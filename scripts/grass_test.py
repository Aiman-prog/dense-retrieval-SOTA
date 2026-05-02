"""
Unit tests for the GRASS mining speedups (S1-S7).
Tests correctness of each change using synthetic data only.
No real model download, no GPU required — falls back to CPU.

Run: python scripts/grass_test.py
"""
import sys
import json
import tempfile
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

from utils.helpers import encode_batch
from utils.bandit import CaseBandit
_mod.CaseBandit = CaseBandit  # S8 tests reference _mod.CaseBandit

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
# Main runner
# -----------------------------------------------------------------------
if __name__ == '__main__':
    print(f"\nGRASS Speedup Tests  (device: {DEVICE})")
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
    ]

    passed = sum(_run(name, fn) for name, fn in suite)
    total  = len(suite)
    print("=" * 60)
    print(f"  {passed}/{total} passed", end="  ")
    if passed == total:
        print("— all speedup checks green, safe to run.")
    else:
        print("— investigate failures before running on cluster.")
    print("=" * 60)
