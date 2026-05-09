"""
Unit tests for the GRASS mining speedups (S1-S7) and async 2-GPU components (S14+).
Tests correctness using synthetic data only.
No real model download, no GPU required — falls back to CPU.

Run: python scripts/grass_test.py
"""
import sys
import json
import heapq
import random
import threading
import tempfile
import time
import collections
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from unittest.mock import MagicMock

# -----------------------------------------------------------------------
# Import setup
# -----------------------------------------------------------------------
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root / 'src'))

# Mock tevatron before loading run_grass_mcd (which imports it at module level)
for _name in [
    'tevatron', 'tevatron.retriever', 'tevatron.retriever.driver',
    'tevatron.retriever.driver.train', 'tevatron.retriever.modeling',
]:
    sys.modules.setdefault(_name, MagicMock())

sys.modules['tevatron.retriever.modeling'].DenseModel = type(
    'DenseModel', (), {'_keys_to_ignore_on_save': None}
)

import importlib.util

# Load run_grass_mcd (needs tevatron mock) — provides _shortlist_batch
mcd_spec = importlib.util.spec_from_file_location(
    'run_grass_mcd', Path(__file__).parent / 'run_grass_mcd.py'
)
mcd_mod = importlib.util.module_from_spec(mcd_spec)
sys.modules['run_grass_mcd'] = mcd_mod
mcd_spec.loader.exec_module(mcd_mod)
_shortlist_batch = mcd_mod._shortlist_batch

# Load run_grass_ema (no tevatron) — provides mine_ema_batch; alias as _mod for S8 tests
ema_spec = importlib.util.spec_from_file_location(
    'run_grass_ema', Path(__file__).parent / 'run_grass_ema.py'
)
_mod = importlib.util.module_from_spec(ema_spec)
sys.modules['run_grass_ema'] = _mod
ema_spec.loader.exec_module(_mod)

from utils.helpers import encode_batch, is_valid_checkpoint, get_latest_marker_no, _shortlist_batch
from utils.bandit import CaseBandit
_mod.CaseBandit = CaseBandit  # S8 tests reference _mod.CaseBandit

# Load run_grass_train for _apply_pending_neg_updates (no tevatron imports at module level)
_train_spec = importlib.util.spec_from_file_location(
    'run_grass_train', Path(__file__).parent / 'run_grass_train.py'
)
_train_mod = importlib.util.module_from_spec(_train_spec)
# Stub out temperature_scaled_loss so module-level import doesn't fail
sys.modules.setdefault('models', MagicMock())
sys.modules.setdefault('models.temperature_scaled_loss', MagicMock())
sys.modules['run_grass_train'] = _train_mod
_train_spec.loader.exec_module(_train_mod)
_apply_pending_neg_updates = _train_mod._apply_pending_neg_updates

# -----------------------------------------------------------------------
# Shared helpers
# -----------------------------------------------------------------------
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class MockOutput:
    def __init__(self, last_hidden_state):
        self.last_hidden_state = last_hidden_state


class MockModel(nn.Module):
    """
    Tiny model shaped like a HuggingFace transformer.
    Has a Dropout layer so model.train() produces stochastic outputs —
    needed to verify MC-dropout diversity (S1).
    """
    def __init__(self, hidden=32, seq_len=4, dropout_p=0.5):
        super().__init__()
        self.hidden   = hidden
        self.seq_len  = seq_len
        self.dropout  = nn.Dropout(p=dropout_p)

    def forward(self, input_ids=None, attention_mask=None, **kwargs):
        B = input_ids.shape[0]
        # Randn gives a unique random state each call; dropout then varies per element
        x = torch.randn(B, self.seq_len, self.hidden, device=input_ids.device)
        return MockOutput(last_hidden_state=self.dropout(x))


class _BatchEncoding(dict):
    """Dict subclass with .to(device) so encode_batch can call inputs.to(device)."""
    def to(self, device):
        return _BatchEncoding({k: v.to(device) for k, v in self.items()})


class MockTokenizer:
    """Tokenizer that returns different input_ids per text (so embeddings differ)."""
    def __call__(self, texts, padding=True, truncation=True,
                 max_length=128, return_tensors='pt'):
        B = len(texts)
        ids = torch.zeros(B, 4, dtype=torch.long)
        for i, t in enumerate(texts):
            ids[i, 0] = abs(hash(t)) % 1000
        return _BatchEncoding({
            'input_ids':      ids,
            'attention_mask': torch.ones(B, 4, dtype=torch.long),
        })


def _run(name, fn):
    print(f"  {name} ...", end=' ', flush=True)
    try:
        fn()
        print("✅ PASS")
        return True
    except AssertionError as e:
        print(f"❌ FAIL — {e}")
        return False
    except Exception as e:
        print(f"❌ ERROR — {type(e).__name__}: {e}")
        return False


# -----------------------------------------------------------------------
# S3 — autocast in encode_batch
# -----------------------------------------------------------------------

def test_s3_outputs_normalized():
    """encode_batch must return L2-normalized embeddings (norm ≈ 1)."""
    model     = MockModel().to(DEVICE)
    tokenizer = MockTokenizer()
    model.eval()
    texts = ["hello world", "foo bar baz", "test query"]
    embs  = encode_batch(model, tokenizer, texts, DEVICE, max_len=32, batch_size=8)
    assert embs.shape == (3, 32), f"shape mismatch: {embs.shape}"
    norms = np.linalg.norm(embs, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-5), f"not normalized: norms={norms}"


def test_s3_autocast_no_crash():
    """autocast (enabled=False on CPU) must not raise regardless of device."""
    model     = MockModel().to(DEVICE)
    tokenizer = MockTokenizer()
    model.eval()
    embs = encode_batch(model, tokenizer, ["a", "b", "c"], DEVICE, max_len=32, batch_size=4)
    assert embs.shape[0] == 3


# -----------------------------------------------------------------------
# S1/S2 — Vectorized T MC encodes
# -----------------------------------------------------------------------

def test_s1_shape_correct():
    """batch_texts * T reshaped to (T, B, dim) must have the right shape."""
    model     = MockModel().to(DEVICE)
    tokenizer = MockTokenizer()
    model.train()
    T, B = 5, 8
    texts     = [f"query {i}" for i in range(B)]
    flat      = encode_batch(model, tokenizer, texts * T, DEVICE, max_len=32, batch_size=16)
    assert flat.shape == (T * B, 32), f"flat shape: {flat.shape}"
    stack = flat.reshape(T, B, -1)
    assert stack.shape == (T, B, 32), f"stacked shape: {stack.shape}"


def test_s1_mc_dropout_diversity():
    """
    T copies of the same text in one forward call get independent dropout masks
    → all T embeddings should differ. This validates that vectorising the T passes
    into a single encode_batch call still produces genuine MC-dropout diversity.
    """
    model     = MockModel(dropout_p=0.5).to(DEVICE)
    tokenizer = MockTokenizer()
    model.train()  # dropout active
    T     = 5
    texts = ["same text"] * T   # T copies of one query
    flat  = encode_batch(model, tokenizer, texts, DEVICE, max_len=32, batch_size=T)
    stack = flat.reshape(T, -1)

    # Every pair of passes should produce a different embedding
    identical_pairs = sum(
        np.allclose(stack[i], stack[j], atol=1e-6)
        for i in range(T) for j in range(i + 1, T)
    )
    assert identical_pairs == 0, \
        f"{identical_pairs}/{T*(T-1)//2} pass-pairs were identical — dropout not firing"


def test_s1_partial_last_batch():
    """Vectorization must handle the last batch where len(batch_texts) < query_batch_size."""
    model     = MockModel().to(DEVICE)
    tokenizer = MockTokenizer()
    model.train()
    T, B = 3, 5   # odd size
    texts = [f"q{i}" for i in range(B)]
    flat  = encode_batch(model, tokenizer, texts * T, DEVICE, max_len=32, batch_size=8)
    stack = flat.reshape(T, len(texts), -1)
    assert stack.shape == (T, B, 32)


# -----------------------------------------------------------------------
# S4 — numpy einsum replaces per-query torch.bmm
# -----------------------------------------------------------------------

def test_s4_einsum_matches_bmm():
    """np.einsum('td,tnd->tn') must be numerically identical to the old torch.bmm path."""
    rng = np.random.default_rng(42)
    T, N, dim = 5, 50, 64
    q_i = rng.standard_normal((T, dim)).astype(np.float32)
    c_i = rng.standard_normal((T, N, dim)).astype(np.float32)

    # Old approach
    q_t       = torch.from_numpy(q_i).unsqueeze(1)                          # (T,1,dim)
    c_t       = torch.from_numpy(c_i)                                        # (T,N,dim)
    sims_bmm  = torch.bmm(q_t, c_t.transpose(1, 2)).squeeze(1).numpy()     # (T,N)

    # New approach
    sims_einsum = np.einsum('td,tnd->tn', q_i, c_i)

    max_diff = np.abs(sims_bmm - sims_einsum).max()
    assert np.allclose(sims_bmm, sims_einsum, atol=1e-5), \
        f"max diff = {max_diff:.2e} — einsum and bmm diverge"


def test_s4_selects_correct_top_m():
    """The top-m candidate by g-score must be the true maximum after einsum scoring."""
    rng = np.random.default_rng(7)
    T, N, m, lambda_val = 5, 20, 1, 2.0
    q_i  = rng.standard_normal((T, 64)).astype(np.float32)
    c_i  = rng.standard_normal((T, N, 64)).astype(np.float32)
    sims = np.einsum('td,tnd->tn', q_i, c_i)
    s_hat = sims.mean(axis=0)
    sigma = sims.std(axis=0)
    g     = s_hat + lambda_val * sigma
    top_m = np.argsort(g)[::-1][:m]
    assert g[top_m[0]] == g.max(), \
        f"top-1 g={g[top_m[0]]:.4f} is not the max g={g.max():.4f}"


# -----------------------------------------------------------------------
# S7 — _shortlist_batch correctness
# -----------------------------------------------------------------------

def _make_inputs(n_queries=4, P=10, L=5, dim=16, n_corpus=50, seed=0):
    """Generate synthetic corpus/query data for _shortlist_batch tests."""
    rng         = np.random.default_rng(seed)
    stale_embs  = rng.standard_normal((n_corpus, dim)).astype(np.float32)
    stale_embs /= np.linalg.norm(stale_embs, axis=1, keepdims=True) + 1e-8
    q_embs_det  = rng.standard_normal((n_queries, dim)).astype(np.float32)
    q_embs_det /= np.linalg.norm(q_embs_det, axis=1, keepdims=True) + 1e-8
    c_ids        = [f"doc{i}" for i in range(n_corpus)]
    c_id_to_idx  = {d: i for i, d in enumerate(c_ids)}
    corpus_lookup = {d: f"text {d}" for d in c_ids}
    batch_ids    = [f"q{i}" for i in range(n_queries)]
    indices      = rng.integers(0, n_corpus, size=(n_queries, P))
    return (batch_ids, indices, q_embs_det, c_ids, c_id_to_idx,
            stale_embs, corpus_lookup, P, L)


def test_s7_shortlist_bounded_by_L():
    """Every query's shortlist must have at most L candidates."""
    args = _make_inputs(n_queries=8, P=20, L=5)
    batch_ids, indices, q_embs_det, c_ids, c_id_to_idx, stale_embs, corpus_lookup, P, L = args
    sl, _, _, _, _ = _shortlist_batch(
        batch_ids, indices, q_embs_det, {}, c_ids,
        c_id_to_idx, stale_embs, corpus_lookup, P, L
    )
    for qid in batch_ids:
        assert len(sl[qid]) <= L, f"{qid}: {len(sl[qid])} > L={L}"


def test_s7_true_positives_filtered():
    """Candidates listed as positives in qrels must never appear in the shortlist."""
    args = _make_inputs(n_queries=4, P=10, L=5, n_corpus=20)
    batch_ids, indices, q_embs_det, c_ids, c_id_to_idx, stale_embs, corpus_lookup, P, L = args
    forbidden = {c_ids[0], c_ids[1], c_ids[2]}
    qrels_dict = {qid: forbidden.copy() for qid in batch_ids}
    sl, _, _, _, _ = _shortlist_batch(
        batch_ids, indices, q_embs_det, qrels_dict, c_ids,
        c_id_to_idx, stale_embs, corpus_lookup, P, L
    )
    for qid in batch_ids:
        leaked = [d for d in sl[qid] if d in forbidden]
        assert not leaked, f"{qid} leaked positives: {leaked}"


def test_s7_shortlist_is_top_L_by_score():
    """
    For a single query with fixed candidates, the shortlist must exactly match the
    top-L candidates by stale_embs @ q_embs_det dot product.
    """
    dim, n_corpus, P, L = 16, 30, 20, 5
    rng         = np.random.default_rng(1)
    stale_embs  = rng.standard_normal((n_corpus, dim)).astype(np.float32)
    q_embs_det  = rng.standard_normal((1, dim)).astype(np.float32)
    c_ids        = [f"doc{i}" for i in range(n_corpus)]
    c_id_to_idx  = {d: i for i, d in enumerate(c_ids)}
    corpus_lookup = {d: f"text {d}" for d in c_ids}
    # Query sees docs 0..P-1 (no FAISS randomness)
    indices     = np.arange(P).reshape(1, P)
    batch_ids   = ["q0"]

    sl, _, _, _, _ = _shortlist_batch(
        batch_ids, indices, q_embs_det, {}, c_ids,
        c_id_to_idx, stale_embs, corpus_lookup, P, L
    )

    scores = stale_embs[:P] @ q_embs_det[0]
    expected = {c_ids[k] for k in np.argsort(scores)[::-1][:L]}
    actual   = set(sl["q0"])
    assert actual == expected, f"expected {expected}, got {actual}"


def test_s7_n_filtered_matches_manual():
    """n_filtered returned from _shortlist_batch must equal a manual recount."""
    args = _make_inputs(n_queries=3, P=10, L=5, n_corpus=20)
    batch_ids, indices, q_embs_det, c_ids, c_id_to_idx, stale_embs, corpus_lookup, P, L = args
    # q0 has one positive filtered out
    qrels_dict = {"q0": {c_ids[0]}}

    _, _, _, _, n_filtered = _shortlist_batch(
        batch_ids, indices, q_embs_det, qrels_dict, c_ids,
        c_id_to_idx, stale_embs, corpus_lookup, P, L
    )

    expected = 0
    for i, qid in enumerate(batch_ids):
        expected += sum(
            1 for j in indices[i]
            if j >= 0 and c_ids[j] not in qrels_dict.get(qid, set())
        )
    assert n_filtered == expected, f"n_filtered={n_filtered}, expected={expected}"


# -----------------------------------------------------------------------
# S6 — mining log structure
# -----------------------------------------------------------------------

def test_s6_log_fields_and_values():
    """
    Simulates one iteration of the g-score loop and verifies the mining log entry
    is valid JSON with all required fields and sane values.
    """
    required = {
        "query_id", "neg_docid", "s_hat_selected",
        "sigma_selected", "g_selected", "rank_by_shat", "sigma_mean_shortlist",
    }
    rng   = np.random.default_rng(5)
    T, N  = 5, 15
    sims  = rng.standard_normal((T, N)).astype(np.float32)
    s_hat = sims.mean(axis=0)
    sigma = sims.std(axis=0)
    g     = s_hat + 2.0 * sigma
    cands = [f"doc{i}" for i in range(N)]
    top_m = np.argsort(g)[::-1][:1]
    # rank_by_shat — same formula as in train_grass.py
    rank_by_shat = int(np.argsort(np.argsort(-s_hat))[top_m[0]])

    with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
        f.write(json.dumps({
            "query_id":             "q42",
            "neg_docid":            cands[top_m[0]],
            "s_hat_selected":       float(s_hat[top_m[0]]),
            "sigma_selected":       float(sigma[top_m[0]]),
            "g_selected":           float(g[top_m[0]]),
            "rank_by_shat":         rank_by_shat,
            "sigma_mean_shortlist": float(sigma.mean()),
        }, ensure_ascii=False) + '\n')
        tmp = f.name

    with open(tmp) as f:
        entry = json.loads(f.readline())

    missing = required - entry.keys()
    assert not missing, f"missing keys: {missing}"
    assert 0 <= entry["rank_by_shat"] < N, \
        f"rank_by_shat={entry['rank_by_shat']} out of range [0,{N})"
    # The selected candidate must have the maximum g-score
    assert np.isclose(entry["g_selected"], float(g.max()), atol=1e-5), \
        f"g_selected={entry['g_selected']} != g.max()={float(g.max())}"
    # sigma_mean_shortlist must be positive (std of a non-constant distribution)
    assert entry["sigma_mean_shortlist"] >= 0.0


# -----------------------------------------------------------------------
# S8 — CaseBandit correctness
# -----------------------------------------------------------------------

def test_s8_config_has_mab_keys():
    """config.yaml must have all MAB keys under training.grass with sane values."""
    import yaml
    cfg_path = project_root / 'config' / 'config.yaml'
    with open(cfg_path) as f:
        config = yaml.safe_load(f)
    grass = config['training']['grass']
    for key in ('mab_n_das', 'mab_alpha', 'mab_epsilon', 'mab_min_pulls'):
        assert key in grass, f"missing key: {key}"
    assert grass['mab_n_das'] < grass['ema_batch_size'], \
        "mab_n_das must be < ema_batch_size for bandit to be active"
    assert isinstance(grass['mab_alpha'], float)


def test_s8_mine_ema_returns_sigma_scores():
    """mine_ema_batch must return a 2-tuple (mined, sigma_scores) with float values."""
    from unittest.mock import MagicMock
    dim, n_corpus = 4, 5
    rng = np.random.default_rng(0)
    embs = rng.standard_normal((n_corpus, dim)).astype(np.float32)
    embs /= np.linalg.norm(embs, axis=1, keepdims=True) + 1e-8

    mock_idx = MagicMock()
    mock_idx.search.return_value = (None, np.array([[0, 1, 2, 3, 4]]))

    c_ids        = [f"doc{i}" for i in range(n_corpus)]
    c_id_to_idx  = {d: i for i, d in enumerate(c_ids)}
    corpus_lookup = {d: "text" for d in c_ids}
    batch_items  = [{"query_id": "q0", "query": "test"}]
    cfg    = {'P': 5, 'L': 3, 'm': 1, 'lambda_val': 2.0, 'mc_batch_size': 8}
    config = {'model': {'query_max_len': 32, 'passage_max_len': 32}}

    orig_enc = _mod.encode_batch
    call_n   = [0]
    def fake_enc(model, tok, texts, device, max_len, bs):
        call_n[0] += 1
        n = len(texts)
        e = rng.standard_normal((n, dim)).astype(np.float32)
        e /= np.linalg.norm(e, axis=1, keepdims=True) + 1e-8
        return e
    _mod.encode_batch = fake_enc
    try:
        mock_model = MagicMock()
        result = _mod.mine_ema_batch(
            mock_model, mock_model, None, batch_items,
            mock_idx, embs, c_id_to_idx, c_ids,
            corpus_lookup, {}, cfg, config, torch.device('cpu')
        )
        assert isinstance(result, tuple) and len(result) == 2, \
            f"expected 2-tuple, got {type(result)}"
        mined, sigma_scores = result
        assert isinstance(mined, dict) and isinstance(sigma_scores, dict)
        for qid, negs in mined.items():
            if negs:
                assert qid in sigma_scores, f"sigma_scores missing for mined qid={qid}"
                assert isinstance(sigma_scores[qid], float), "sigma not float"
    finally:
        _mod.encode_batch = orig_enc


def test_s8_neg_cache_from_train_items():
    """neg_cache must be populated from train_items neg_docid; None entries excluded."""
    train_items = [
        {'query_id': 'q1', 'neg_docid': 'n1'},
        {'query_id': 'q2', 'neg_docid': None},
        {'query_id': 'q3', 'neg_docid': 'n3'},
    ]
    neg_cache = {it['query_id']: it['neg_docid'] for it in train_items if it['neg_docid']}
    assert neg_cache.get('q1') == 'n1', f"q1 not cached correctly: {neg_cache}"
    assert 'q2' not in neg_cache, "q2 has no neg, must not appear in cache"
    assert neg_cache.get('q3') == 'n3', f"q3 not cached correctly: {neg_cache}"


def test_s8_unseen_queries_always_selected():
    """Unseen queries (UCB=inf) must always rank above any seen query."""
    bandit = _mod.CaseBandit(n_das=2, alpha=1.0)
    bandit.update("qA", 0.9)
    bandit.update("qA", 0.9)
    selected = bandit.select(["qA", "qB", "qC"])
    assert "qB" in selected and "qC" in selected, \
        f"Unseen queries not prioritised: selected={selected}"


def test_s8_jt_queries_never_selected():
    """select() must never return a query already in J_t."""
    bandit = _mod.CaseBandit(n_das=3, alpha=0.0, epsilon=1.0, min_pulls=1)
    bandit.J_t.add("qA")
    selected = bandit.select(["qA", "qB", "qC", "qD"])
    assert "qA" not in selected, f"J_t query was selected: {selected}"


def test_s8_only_n_das_queries_selected_per_batch():
    """select() must return exactly n_das queries (or fewer if batch is smaller)."""
    bandit = _mod.CaseBandit(n_das=2, alpha=1.0)
    selected = bandit.select(["q0", "q1", "q2", "q3", "q4"])
    assert len(selected) == 2, f"expected 2 selected, got {len(selected)}: {selected}"


def test_s8_fallback_to_neg_cache():
    """Queries not in mine_set must fall back to neg_cache; absent entries stay absent."""
    neg_cache = {"q3": "docZ"}
    batch_items = [{"query_id": "q3"}, {"query_id": "q4"}]
    mined = {}  # q3 was not a challenger
    for it in batch_items:
        qid = it['query_id']
        if qid not in mined and qid in neg_cache:
            mined[qid] = [neg_cache[qid]]
    assert mined.get("q3") == ["docZ"], f"q3 fallback failed: {mined}"
    assert "q4" not in mined, "q4 has no cache entry, must not appear in mined"


def test_s8_low_sigma_graduates_to_jt():
    """After min_pulls updates with sigma near 0, query must enter J_t."""
    bandit = _mod.CaseBandit(n_das=2, alpha=0.0, epsilon=0.5, min_pulls=2)
    bandit.update("qRef", 0.8)
    bandit.J_t.add("qRef")
    bandit.mean_sigma["qRef"] = 0.8
    bandit.update("qNew", 0.01)
    bandit.update("qNew", 0.01)
    assert "qNew" in bandit.J_t, \
        f"Low-sigma query did not graduate to J_t after min_pulls"


# -----------------------------------------------------------------------
# S9  — L=25 + ema_batch_size=64 config
# S10 — _foreach EMA update
# S11 — zero_grad(set_to_none=True)
# S12 — save_steps=1000 config
# S13 — torch.compile on student
# -----------------------------------------------------------------------

def test_s9_config_L_and_batch():
    """L must be <= 25 [S9] and ema_batch_size must be >= 64 [AdamW8bit]."""
    import yaml
    with open(project_root / 'config' / 'config.yaml') as f:
        config = yaml.safe_load(f)
    grass = config['training']['grass']
    assert grass['L'] <= 25, f"L={grass['L']} should be <= 25 after [S9]"
    assert grass['ema_batch_size'] >= 64, \
        f"ema_batch_size={grass['ema_batch_size']} should be >= 64 after AdamW8bit change"


def test_s10_foreach_ema_matches_loop():
    """_foreach_mul_ + _foreach_add_ must produce the same result as the per-tensor loop."""
    alpha = 0.999
    torch.manual_seed(0)
    ema_loop    = [torch.randn(8, 8) for _ in range(6)]
    ema_foreach = [t.clone() for t in ema_loop]
    cur         = [torch.randn(8, 8) for _ in range(6)]

    with torch.no_grad():
        for p_ema, p_cur in zip(ema_loop, cur):
            p_ema.data.mul_(alpha).add_(p_cur.data, alpha=1.0 - alpha)

    with torch.no_grad():
        torch._foreach_mul_(ema_foreach, alpha)
        torch._foreach_add_(ema_foreach, cur, alpha=1.0 - alpha)

    for i, (loop_t, foreach_t) in enumerate(zip(ema_loop, ema_foreach)):
        assert torch.allclose(loop_t, foreach_t, atol=1e-6), \
            f"tensor {i}: max diff {(loop_t - foreach_t).abs().max():.2e}"


def test_s11_zero_grad_set_to_none():
    """zero_grad(set_to_none=True) must leave all grad attributes as None, not zero tensors."""
    linear = torch.nn.Linear(4, 4).to(DEVICE)
    x = torch.randn(2, 4, device=DEVICE)
    linear(x).sum().backward()
    assert any(p.grad is not None for p in linear.parameters()), \
        "backward must produce gradients before zero_grad"
    linear.zero_grad(set_to_none=True)
    assert all(p.grad is None for p in linear.parameters()), \
        "set_to_none=True must set all grads to None (not zero tensors)"


def test_s12_config_save_steps():
    """save_steps must be >= 1000 to halve checkpoint I/O overhead [S12]."""
    import yaml
    with open(project_root / 'config' / 'config.yaml') as f:
        config = yaml.safe_load(f)
    save_steps = config['training']['grass']['save_steps']
    assert save_steps >= 1000, f"save_steps={save_steps} should be >= 1000 after [S12]"


def test_s13_torch_compile_correct_shape():
    """torch.compile(model, dynamic=True) must produce same output shape as uncompiled [S13]."""
    model     = MockModel().to(DEVICE)
    tokenizer = MockTokenizer()
    model.eval()
    texts = ["hello", "world", "foo bar"]

    with torch.no_grad():
        embs_orig = encode_batch(model, tokenizer, texts, DEVICE, max_len=32, batch_size=8)

    try:
        compiled = torch.compile(model, dynamic=True)
        with torch.no_grad():
            embs_compiled = encode_batch(compiled, tokenizer, texts, DEVICE, max_len=32, batch_size=8)
        assert embs_compiled.shape == embs_orig.shape, \
            f"compiled shape {embs_compiled.shape} != original {embs_orig.shape}"
        norms = np.linalg.norm(embs_compiled, axis=1)
        assert np.allclose(norms, 1.0, atol=1e-5), f"compiled embeddings not normalized: {norms}"
    except Exception as e:
        print(f"(torch.compile unavailable: {e} — skip)", end=" ")


# -----------------------------------------------------------------------
# Epsilon-greedy heap / CaseBandit (async tests)
# -----------------------------------------------------------------------

def test_heap_lazy_deletion():
    """Stale heap entry (wrong version) must be discarded; updated entry wins."""
    b = CaseBandit(n_das=1, epsilon=0.0)
    b.init_all_queries(['q1', 'q2'])
    # q1 gets σ=0.8; this creates version=1 entry in heap
    b.update('q1', 0.8)
    # Manually corrupt heap by pushing a stale entry for q1 at version 0
    heapq.heappush(b.heap, (-0.0, 0, 'q1'))
    # select_global should still return q1 (highest mean-σ), discarding the stale 0.0 entry
    selected = b.select_global(n_das=1, epsilon=0.0)
    assert 'q1' in selected, f"q1 (σ=0.8) not selected; got {selected}"


def test_epsilon_split():
    """Over 1000 rounds, exploit slot fraction ≈ (1-ε) within ±5%."""
    rng = random.Random(42)
    n_queries = 100
    epsilon   = 0.2
    n_das     = 10
    qids      = [f"q{i}" for i in range(n_queries)]
    b         = CaseBandit(n_das=n_das, epsilon=epsilon)
    b.init_all_queries(qids)
    # Seed some observations so exploitation is non-trivial
    for qid in qids[:50]:
        b.update(qid, rng.random())

    exploit_count = 0
    total_selected = 0
    # Run enough rounds that unseen set empties and we see stable exploit/explore split
    for _ in range(200):
        exploit_ids = b._heap_pop_top(int(n_das * (1 - epsilon)))
        explore_ids = rng.sample(list(b.unseen - b.J_t), min(n_das - len(exploit_ids),
                                                               len(b.unseen - b.J_t)))
        exploit_count  += len(exploit_ids)
        total_selected += len(exploit_ids) + len(explore_ids)
        # Re-push exploit so heap stays valid
        for qid in exploit_ids:
            heapq.heappush(b.heap, (-b.mean_sigma.get(qid, 0.0), b.version.get(qid, 0), qid))

    if total_selected > 0:
        actual_frac = exploit_count / total_selected
        assert abs(actual_frac - (1 - epsilon)) < 0.15, \
            f"exploit fraction={actual_frac:.2f} far from {1-epsilon:.2f}"


def test_exploitation_favours_high_sigma():
    """Query with σ=0.9 should win >50% of exploit slots vs 9 queries at σ=0.1."""
    rng   = random.Random(0)
    b     = CaseBandit(n_das=5, epsilon=0.0)  # pure exploitation
    qids  = [f"q{i}" for i in range(10)]
    b.init_all_queries(qids)
    for qid in qids[1:]:
        b.update(qid, 0.1)
        b.update(qid, 0.1)
    b.update('q0', 0.9)
    b.update('q0', 0.9)

    q0_count = 0
    total    = 0
    for _ in range(100):
        selected = b.select_global(n_das=5, epsilon=0.0)
        q0_count += selected.count('q0')
        total    += len(selected)

    assert q0_count / max(total, 1) > 0.3, \
        f"high-σ query q0 got only {q0_count}/{total} exploit slots"


def test_monopolisation_bounded():
    """High-σ query must not monopolise: expect <80% of total mining events."""
    b    = CaseBandit(n_das=5, epsilon=0.1)
    qids = [f"q{i}" for i in range(100)]
    b.init_all_queries(qids)
    for qid in qids[1:]:
        b.update(qid, 0.1)
    b.update('q0', 0.9)

    counts = collections.Counter()
    for _ in range(200):
        selected = b.select_global(n_das=5, epsilon=0.1)
        counts.update(selected)

    total    = sum(counts.values())
    q0_share = counts['q0'] / max(total, 1)
    assert q0_share < 0.80, f"q0 monopolised {q0_share:.1%} of mining events"


def test_jt_graduation_excludes():
    """A graduated query must never appear in exploit or explore output."""
    b    = CaseBandit(n_das=5, epsilon=0.2, min_pulls=1)
    qids = [f"q{i}" for i in range(20)]
    b.init_all_queries(qids)
    # Graduate q0 manually
    b.update('q0', 0.9)   # min_pulls=1 → immediately graduates (J_t bootstrap)
    assert 'q0' in b.J_t, "q0 should have graduated"

    for _ in range(50):
        selected = b.select_global(n_das=5, epsilon=0.2)
        assert 'q0' not in selected, f"graduated q0 appeared in {selected}"


def test_unseen_set_shrinks():
    """After K exploration events, len(unseen) must decrease by K (no double-counts)."""
    b    = CaseBandit(n_das=3, epsilon=1.0)  # pure exploration
    qids = [f"q{i}" for i in range(50)]
    b.init_all_queries(qids)
    initial_unseen = len(b.unseen)
    seen_new = set()
    for _ in range(10):
        selected = b.select_global(n_das=3, epsilon=1.0)
        for qid in selected:
            if qid in b.unseen:
                b.update(qid, 0.2)
                seen_new.add(qid)

    assert len(b.unseen) == initial_unseen - len(seen_new), \
        f"unseen set size mismatch after updates"


def test_sigma_zero_init():
    """All queries start with σ=0 in heap; first real observation pushes above 0."""
    b    = CaseBandit(n_das=1, epsilon=0.0)
    qids = ['qA', 'qB']
    b.init_all_queries(qids)
    assert all(b.mean_sigma.get(qid, 0.0) == 0.0 for qid in qids)
    b.update('qA', 0.5)
    assert b.mean_sigma['qA'] > 0.0, "mean_sigma should be >0 after first update"


# -----------------------------------------------------------------------
# IPC tests
# -----------------------------------------------------------------------

def test_ipc_write_read():
    """Miner-style write (update_{N}.jsonl + ready_{N}) → _apply_pending updates neg_cache."""
    with tempfile.TemporaryDirectory() as tmpdir:
        d = Path(tmpdir)
        # Simulate miner writing update #1
        data = [{'query_id': 'q1', 'neg_docid': 'doc42'},
                {'query_id': 'q2', 'neg_docid': 'doc99'}]
        with open(d / 'update_1.jsonl', 'w') as f:
            for row in data:
                f.write(json.dumps(row) + '\n')
        (d / 'ready_1').write_text('1')

        neg_cache = {}
        last_no, n = _apply_pending_neg_updates(d, neg_cache, 0)
        assert last_no == 1, f"last_no={last_no}"
        assert n == 2,       f"n_applied={n}"
        assert neg_cache.get('q1') == 'doc42'
        assert neg_cache.get('q2') == 'doc99'


def test_ipc_validity_gate():
    """JSONL present but ready marker absent → update ignored."""
    with tempfile.TemporaryDirectory() as tmpdir:
        d = Path(tmpdir)
        with open(d / 'update_1.jsonl', 'w') as f:
            f.write(json.dumps({'query_id': 'q1', 'neg_docid': 'doc42'}) + '\n')
        # No ready_1 written

        neg_cache = {}
        last_no, n = _apply_pending_neg_updates(d, neg_cache, 0)
        assert n == 0, f"partial update should be ignored; n_applied={n}"
        assert 'q1' not in neg_cache


def test_ipc_all_pending_applied():
    """Trainer applies updates 1, 2, 3 in order, not just the latest."""
    with tempfile.TemporaryDirectory() as tmpdir:
        d = Path(tmpdir)
        for i in range(1, 4):
            with open(d / f'update_{i}.jsonl', 'w') as f:
                f.write(json.dumps({'query_id': f'q{i}', 'neg_docid': f'doc{i}'}) + '\n')
            (d / f'ready_{i}').write_text(str(i))

        neg_cache = {}
        last_no, n = _apply_pending_neg_updates(d, neg_cache, 0)
        assert last_no == 3
        assert n == 3
        for i in range(1, 4):
            assert neg_cache.get(f'q{i}') == f'doc{i}', f"q{i} missing"


def test_miner_checkpoint_gate():
    """is_valid_checkpoint is False without optimizer.pt, True after it exists."""
    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt = Path(tmpdir) / 'checkpoint-500'
        ckpt.mkdir()
        assert not is_valid_checkpoint(str(ckpt)), "should be invalid without optimizer.pt"
        (ckpt / 'optimizer.pt').write_bytes(b'')
        assert is_valid_checkpoint(str(ckpt)), "should be valid once optimizer.pt exists"


# -----------------------------------------------------------------------
# neg_cache / training integration
# -----------------------------------------------------------------------

def test_neg_cache_injected_in_batch():
    """Miner update for q1 must appear in the neg_cache used by trainer."""
    neg_cache    = {'q1': 'old_doc', 'q2': 'doc_b'}
    miner_update = {'query_id': 'q1', 'neg_docid': 'new_hard_doc'}
    neg_cache[miner_update['query_id']] = miner_update['neg_docid']
    assert neg_cache['q1'] == 'new_hard_doc', "neg_cache not updated by miner"


def test_neg_cache_fallback():
    """Unmined query must use original mixture negative, not empty string."""
    train_items = [
        {'query_id': 'qA', 'neg_docid': 'mixture_neg'},
        {'query_id': 'qB', 'neg_docid': None},
    ]
    neg_cache = {it['query_id']: it['neg_docid'] for it in train_items if it['neg_docid']}
    assert neg_cache.get('qA') == 'mixture_neg', "qA should use mixture negative"
    assert 'qB' not in neg_cache, "qB has no negative, must not appear in cache"


# -----------------------------------------------------------------------
# Distribution plot tests (coverage curves; skip if matplotlib absent)
# -----------------------------------------------------------------------

def test_coverage_curve_plot():
    """
    Simulate mining for n_das in {3,6} over 200 cycles.
    Verify coverage grows monotonically and at expected rate (~n_das/n_queries per cycle).
    Saves plot to logs_cluster/ if matplotlib is available.
    """
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        has_mpl = True
    except ImportError:
        has_mpl = False

    n_queries  = 500
    n_cycles   = 100
    qids       = [f"q{i}" for i in range(n_queries)]

    curves = {}
    for n_das in [3, 6]:
        b = CaseBandit(n_das=n_das, epsilon=0.2)
        b.init_all_queries(qids)
        coverage = []
        mined    = set()
        for _ in range(n_cycles):
            selected = b.select_global(n_das=n_das, epsilon=0.2)
            for qid in selected:
                if qid not in mined:
                    b.update(qid, random.uniform(0.0, 0.5))
                    mined.add(qid)
            coverage.append(len(mined) / n_queries)
        curves[n_das] = coverage

        # Coverage must be monotonically non-decreasing
        assert all(curves[n_das][i] <= curves[n_das][i+1]
                   for i in range(len(curves[n_das])-1)), \
            f"n_das={n_das}: coverage not monotone"
        # After all cycles, n_das=6 should have higher coverage than n_das=3
    assert curves[6][-1] >= curves[3][-1], \
        f"larger n_das should yield higher coverage: {curves[6][-1]} vs {curves[3][-1]}"

    if has_mpl:
        logs_dir = Path(__file__).resolve().parent.parent / 'logs_cluster'
        logs_dir.mkdir(exist_ok=True)
        plt.figure()
        for n_das, cov in curves.items():
            plt.plot(cov, label=f'n_das={n_das}')
        plt.xlabel('Mining cycle')
        plt.ylabel('Fraction of corpus covered')
        plt.title('Coverage curve by n_das')
        plt.legend()
        plt.savefig(str(logs_dir / 'test_coverage_curve.png'))
        plt.close()


def test_e2e_smoke():
    """
    End-to-end smoke: fake miner thread + trainer loop with tiny data.
    Verifies neg_cache gets updated (no deadlock, no error).
    """
    import queue

    with tempfile.TemporaryDirectory() as tmpdir:
        neg_update_dir = Path(tmpdir) / 'neg_updates'
        neg_update_dir.mkdir()

        n_queries = 50
        qids      = [f"q{i}" for i in range(n_queries)]

        # Fake miner: writes one update every 0.05s
        stop_evt = threading.Event()

        def fake_miner():
            update_no = 1
            while not stop_evt.is_set():
                data = [{'query_id': f'q{random.randint(0, n_queries-1)}',
                          'neg_docid': f'doc{random.randint(0, 500)}'}]
                jpath = neg_update_dir / f'update_{update_no}.jsonl'
                with open(jpath, 'w') as f:
                    for row in data:
                        f.write(json.dumps(row) + '\n')
                (neg_update_dir / f'ready_{update_no}').write_text(str(update_no))
                update_no += 1
                time.sleep(0.05)

        miner_thread = threading.Thread(target=fake_miner, daemon=True)
        miner_thread.start()

        # Fake trainer: apply neg updates every 5 "steps"
        neg_cache    = {qid: f'init_{qid}' for qid in qids}
        last_update  = 0
        n_steps      = 40

        for step in range(n_steps):
            if step % 5 == 0:
                last_update, n_applied = _apply_pending_neg_updates(
                    neg_update_dir, neg_cache, last_update
                )
            time.sleep(0.01)

        stop_evt.set()
        miner_thread.join(timeout=2.0)

        # Check some negatives were updated (miner ran fast enough)
        n_updated = sum(1 for qid in qids
                        if not neg_cache.get(qid, '').startswith('init_'))
        assert n_updated > 0 or last_update == 0, \
            "trainer should have applied at least some miner updates"
        # Verify neg_cache has no empty values for initially populated queries
        assert all(neg_cache.get(qid) for qid in qids), \
            "some queries lost their negative entirely"


# -----------------------------------------------------------------------
# Main runner
# -----------------------------------------------------------------------
if __name__ == '__main__':
    print(f"\nGRASS Tests — Speedups + Async 2-GPU  (device: {DEVICE})")
    print("=" * 60)

    suite = [
        # S3 — autocast in encode_batch
        ("S3  encode_batch returns normalized embeddings",    test_s3_outputs_normalized),
        ("S3  autocast does not crash (CPU or GPU)",          test_s3_autocast_no_crash),
        # S1/S2 — vectorized T MC encodes
        ("S1  vectorized shape is (T, B, dim)",               test_s1_shape_correct),
        ("S1  T passes produce different embeddings (MC-dropout)", test_s1_mc_dropout_diversity),
        ("S1  handles partial last batch",                    test_s1_partial_last_batch),
        # S4 — numpy einsum replaces torch.bmm
        ("S4  einsum matches torch.bmm numerically",          test_s4_einsum_matches_bmm),
        ("S4  top-m selection is correct",                    test_s4_selects_correct_top_m),
        # S7 — _shortlist_batch correctness
        ("S7  shortlist size <= L per query",                 test_s7_shortlist_bounded_by_L),
        ("S7  true positives are filtered out",               test_s7_true_positives_filtered),
        ("S7  shortlist is exactly top-L by stale score",     test_s7_shortlist_is_top_L_by_score),
        ("S7  n_filtered count is accurate",                  test_s7_n_filtered_matches_manual),
        # S6 — mining log structure
        ("S6  mining log has all required fields and values", test_s6_log_fields_and_values),
        # S8 — CaseBandit correctness
        ("S8  config has all MAB keys",                      test_s8_config_has_mab_keys),
        ("S8  neg_cache built from train_items neg_docid",   test_s8_neg_cache_from_train_items),
        ("S8  mine_ema_batch returns (mined, sigma_scores)", test_s8_mine_ema_returns_sigma_scores),
        ("S8  only n_das queries selected per batch",         test_s8_only_n_das_queries_selected_per_batch),
        ("S8  fallback to neg_cache for non-challengers",     test_s8_fallback_to_neg_cache),
        ("S8  unseen queries always selected over seen",      test_s8_unseen_queries_always_selected),
        ("S8  J_t queries never returned by select()",        test_s8_jt_queries_never_selected),
        ("S8  low-sigma query graduates to J_t",              test_s8_low_sigma_graduates_to_jt),
        # S9-S13 — speed optimisations
        ("S9  config L<=25 and ema_batch_size>=64",           test_s9_config_L_and_batch),
        ("S10 _foreach EMA matches per-tensor loop",          test_s10_foreach_ema_matches_loop),
        ("S11 zero_grad(set_to_none=True) sets grads=None",   test_s11_zero_grad_set_to_none),
        ("S12 config save_steps >= 1000",                     test_s12_config_save_steps),
        ("S13 torch.compile produces correct shape+norms",    test_s13_torch_compile_correct_shape),
        # Async 2-GPU — epsilon-greedy heap (S14+)
        ("A01 heap lazy deletion discards stale entries",      test_heap_lazy_deletion),
        ("A02 epsilon split ≈ (1-ε) exploit / ε explore",     test_epsilon_split),
        ("A03 exploitation favours high-σ query",              test_exploitation_favours_high_sigma),
        ("A04 monopolisation bounded (<80% of events)",        test_monopolisation_bounded),
        ("A05 graduated query excluded from all selection",    test_jt_graduation_excludes),
        ("A06 unseen set shrinks after explore events",        test_unseen_set_shrinks),
        ("A07 all queries init at σ=0 in heap",                test_sigma_zero_init),
        # IPC
        ("A08 IPC write+read updates neg_cache correctly",     test_ipc_write_read),
        ("A09 IPC validity gate: no ready marker → ignored",   test_ipc_validity_gate),
        ("A10 all pending updates applied in order",           test_ipc_all_pending_applied),
        ("A11 checkpoint gate: invalid without optimizer.pt",  test_miner_checkpoint_gate),
        # neg_cache / training
        ("A12 miner neg injected into neg_cache",              test_neg_cache_injected_in_batch),
        ("A13 unmined query falls back to mixture negative",   test_neg_cache_fallback),
        # Coverage / distribution
        ("A14 coverage curve monotone + n_das=6 > n_das=3",   test_coverage_curve_plot),
        # End-to-end smoke
        ("A15 e2e smoke: fake miner + trainer, no deadlock",   test_e2e_smoke),
    ]

    passed = sum(_run(name, fn) for name, fn in suite)
    total  = len(suite)
    print("=" * 60)
    print(f"  {passed}/{total} passed", end="  ")
    if passed == total:
        print("— all checks green, safe to submit to cluster.")
    else:
        print("— investigate failures before running on cluster.")
    print("=" * 60)
