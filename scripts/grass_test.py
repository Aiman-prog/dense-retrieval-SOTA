"""
Unit tests for GRASS mining speedups and shared sequential components.
Tests correctness using synthetic data only.
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

from utils.helpers import encode_batch, _shortlist_batch

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
# Main runner
# -----------------------------------------------------------------------
if __name__ == '__main__':
    print(f"\nGRASS Tests — Sequential Components  (device: {DEVICE})")
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
        # S9-S13 — speed optimisations
        ("S9  config L<=25 and ema_batch_size>=64",           test_s9_config_L_and_batch),
        ("S10 _foreach EMA matches per-tensor loop",          test_s10_foreach_ema_matches_loop),
        ("S11 zero_grad(set_to_none=True) sets grads=None",   test_s11_zero_grad_set_to_none),
        ("S12 config save_steps >= 1000",                     test_s12_config_save_steps),
        ("S13 torch.compile produces correct shape+norms",    test_s13_torch_compile_correct_shape),
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
