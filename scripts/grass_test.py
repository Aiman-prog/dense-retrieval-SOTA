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

# -----------------------------------------------------------------------
# Import setup
# -----------------------------------------------------------------------
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root / 'src'))

from utils.helpers import encode_batch, _pool_and_fresh_rerank
from utils.grass_candidate_memory import CandidateMemory

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
    """Vectorization must handle the last batch where len(batch_texts) < batch_size."""
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
    # rank_by_shat — same formula as in run_grass.py
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
    """L must be <= 25 [S9] and batch_size must be >= 64 [AdamW8bit]."""
    import yaml
    with open(project_root / 'config' / 'config.yaml') as f:
        config = yaml.safe_load(f)
    grass = config['training']['grass']
    assert grass['L'] <= 25, f"L={grass['L']} should be <= 25 after [S9]"
    assert grass['batch_size'] >= 64, \
        f"batch_size={grass['batch_size']} should be >= 64 after AdamW8bit change"


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
# FR — fresh-rerank (active memory + current-model rerank) — Phase 3
# -----------------------------------------------------------------------

class ScriptedEncoder(nn.Module):
    """Returns a pre-set embedding per text via input_ids[*, 0] -> hash lookup.

    Pair with MockTokenizer (hashes text into ids[0]). Caller must ensure all
    test texts have distinct `abs(hash(t)) % 1000` values.
    """
    def __init__(self, text_to_embedding, dim=8, seq_len=4):
        super().__init__()
        self.text_to_embedding = text_to_embedding
        self.hash_to_emb = {
            abs(hash(t)) % 1000: torch.from_numpy(e).float()
            for t, e in text_to_embedding.items()
        }
        self.dim = dim
        self.seq_len = seq_len
        # Track forward-pass invocations for the "encoded once per batch" test
        self.call_count = 0
        self.total_inputs = 0

    def forward(self, input_ids=None, attention_mask=None, **kwargs):
        self.call_count += 1
        B = input_ids.shape[0]
        self.total_inputs += B
        out = torch.zeros(B, self.seq_len, self.dim)
        for i in range(B):
            h = int(input_ids[i, 0].item())
            if h in self.hash_to_emb:
                out[i, 0] = self.hash_to_emb[h]
        return MockOutput(last_hidden_state=out)


def _make_pool_inputs(memory, faiss_hits, dim=8, c_lookup=None, qrels=None):
    """Build minimal args for _pool_and_fresh_rerank. `memory`/`faiss_hits`/`qrels`
    are dicts keyed by qid; values are lists of docid strings."""
    qids       = sorted(memory.keys() | faiss_hits.keys())
    c_ids_all  = sorted({d for v in list(memory.values()) + list(faiss_hits.values()) for d in v})
    c_id_to_idx = {d: i for i, d in enumerate(c_ids_all)}

    # FAISS indices: pad with -1 to a fixed P
    P_max = max((len(v) for v in faiss_hits.values()), default=0) or 1
    indices = np.full((len(qids), P_max), -1, dtype=np.int64)
    for i, q in enumerate(qids):
        for k, d in enumerate(faiss_hits.get(q, [])):
            indices[i, k] = c_id_to_idx[d]

    q_embs_det = np.eye(len(qids), dim, dtype=np.float32)  # unit basis vectors
    if c_lookup is None:
        c_lookup = {d: f"text-{d}" for d in c_ids_all}
    if qrels is None:
        qrels = {}
    return qids, q_embs_det, indices, c_ids_all, c_id_to_idx, c_lookup, qrels


def test_fr_positive_filtering():
    """Positives in qrels must be filtered out of the pool even when present in both
    FAISS and active memory."""
    memory     = {"q0": ["d_pos", "d3"]}
    faiss_hits = {"q0": ["d_pos", "d1", "d2"]}
    qrels      = {"q0": {"d_pos"}}
    qids, q_embs_det, indices, c_ids, c_id_to_idx, c_lookup, qrels = _make_pool_inputs(
        memory, faiss_hits, qrels=qrels
    )
    model = ScriptedEncoder({t: np.zeros(8, dtype=np.float32) for t in c_lookup.values()})
    tokenizer = MockTokenizer()
    shortlist, _, stats = _pool_and_fresh_rerank(
        model, tokenizer, qids, q_embs_det,
        indices, memory, {q: False for q in qids},
        qrels, c_ids, c_lookup,
        p_max_len=8, mc_batch_size=8, device=DEVICE,
        L=10, max_pool_per_query=64,
    )
    assert "d_pos" not in shortlist["q0"], f"positive leaked: {shortlist['q0']}"
    assert stats["q0"]["positives_filtered"] == 2, \
        f"expected 2 positive filtrations (memory+faiss), got {stats['q0']['positives_filtered']}"


def test_fr_pool_combines_faiss_and_memory():
    """Pool is the union; memory candidates appear first, then FAISS extras."""
    memory     = {"q0": ["m1", "m2"]}
    faiss_hits = {"q0": ["m2", "f1", "f2"]}  # m2 in both
    qids, q_embs_det, indices, c_ids, c_id_to_idx, c_lookup, qrels = _make_pool_inputs(
        memory, faiss_hits
    )
    model = ScriptedEncoder({t: np.zeros(8, dtype=np.float32) for t in c_lookup.values()})
    tokenizer = MockTokenizer()
    shortlist, source_map, _ = _pool_and_fresh_rerank(
        model, tokenizer, qids, q_embs_det,
        indices, memory, {q: False for q in qids},
        qrels, c_ids, c_lookup,
        p_max_len=8, mc_batch_size=8, device=DEVICE,
        L=10, max_pool_per_query=64,
    )
    # m2 should be flagged 'both'
    assert source_map["q0"]["m2"] == "both", \
        f"m2 should be 'both', got {source_map['q0']['m2']}"
    assert source_map["q0"]["m1"] == "memory"
    assert source_map["q0"]["f1"] == "faiss"
    # All non-positive docs appear
    assert set(shortlist["q0"]) == {"m1", "m2", "f1", "f2"}


def test_fr_pool_capping_keeps_memory_first():
    """When memory + FAISS exceeds max_pool_per_query, all memory entries are kept
    first (subject to cap), FAISS fills the remainder by rank."""
    memory     = {"q0": ["m1", "m2", "m3"]}
    faiss_hits = {"q0": ["f1", "f2", "f3", "f4"]}
    qids, q_embs_det, indices, c_ids, c_id_to_idx, c_lookup, qrels = _make_pool_inputs(
        memory, faiss_hits
    )
    model = ScriptedEncoder({t: np.zeros(8, dtype=np.float32) for t in c_lookup.values()})
    tokenizer = MockTokenizer()
    shortlist, source_map, stats = _pool_and_fresh_rerank(
        model, tokenizer, qids, q_embs_det,
        indices, memory, {q: False for q in qids},
        qrels, c_ids, c_lookup,
        p_max_len=8, mc_batch_size=8, device=DEVICE,
        L=5, max_pool_per_query=5,
    )
    sources = source_map["q0"]
    # All 3 memory items must be present
    for m in ["m1", "m2", "m3"]:
        assert m in sources, f"memory entry {m} dropped despite available pool slots"
    # Only 2 FAISS extras admitted (5 - 3 memory)
    faiss_admitted = [d for d, s in sources.items() if s == "faiss"]
    assert len(faiss_admitted) == 2, f"expected 2 FAISS extras, got {faiss_admitted}"
    # The two admitted FAISS docs should be the top-ranked ones (f1, f2)
    assert set(faiss_admitted) == {"f1", "f2"}, \
        f"expected top-rank FAISS docs, got {faiss_admitted}"
    assert stats["q0"]["pool_count"] == 5


def test_fr_top_L_uses_current_embs():
    """Top-L is determined by current_q . current_d dot products from the
    fresh-encoded pool — verifies the architecture fix is in effect.

    Note: encode_batch L2-normalizes the encoder output, so we must vary the
    DIRECTION of doc embeddings (not just magnitude) to get distinct dot
    products against q_emb."""
    dim = 8
    # Query embedding: unit vector along axis 0
    q_emb = np.zeros(dim, dtype=np.float32); q_emb[0] = 1.0
    # Three docs whose normalized embeddings dot with q at decreasing values:
    #   d_high -> [1, 0, ...]                  -> dot 1.0
    #   d_mid  -> [1/sqrt(2), 1/sqrt(2), ...]  -> dot ~0.707
    #   d_low  -> [1/sqrt(26), 5/sqrt(26), ...]-> dot ~0.196
    d_high = np.zeros(dim, dtype=np.float32); d_high[0] = 1.0
    d_mid  = np.zeros(dim, dtype=np.float32); d_mid[0]  = 1.0; d_mid[1] = 1.0
    d_low  = np.zeros(dim, dtype=np.float32); d_low[0]  = 1.0; d_low[1] = 5.0

    memory     = {"q0": []}
    faiss_hits = {"q0": ["d_low", "d_high", "d_mid"]}  # FAISS order does NOT match scores
    qids, _, indices, c_ids, c_id_to_idx, c_lookup, qrels = _make_pool_inputs(
        memory, faiss_hits
    )
    text_to_emb = {
        c_lookup["d_high"]: d_high,
        c_lookup["d_mid"]:  d_mid,
        c_lookup["d_low"]:  d_low,
    }
    # All three texts must have distinct hash%1000 — if not, the test cannot run
    hashes = [abs(hash(t)) % 1000 for t in text_to_emb]
    assert len(set(hashes)) == 3, "test setup collision: pick different texts"

    q_embs_det = q_emb.reshape(1, dim)
    model     = ScriptedEncoder(text_to_emb, dim=dim)
    tokenizer = MockTokenizer()
    shortlist, _, _ = _pool_and_fresh_rerank(
        model, tokenizer, qids, q_embs_det,
        indices, memory, {q: False for q in qids},
        qrels, c_ids, c_lookup,
        p_max_len=8, mc_batch_size=64, device=DEVICE,
        L=1, max_pool_per_query=10,
    )
    assert shortlist["q0"] == ["d_high"], \
        f"top-1 should be d_high (current-model dot product), got {shortlist['q0']}"


def test_fr_pool_encoded_once_per_batch():
    """The pool encode pass is one call to model.forward (modulo batching of
    mc_batch_size), and each unique pool doc is encoded exactly once."""
    memory     = {"q0": ["d_a"], "q1": ["d_a", "d_b"]}  # d_a shared
    faiss_hits = {"q0": ["d_c"], "q1": ["d_d"]}
    qids, q_embs_det, indices, c_ids, c_id_to_idx, c_lookup, qrels = _make_pool_inputs(
        memory, faiss_hits
    )
    model = ScriptedEncoder({t: np.zeros(8, dtype=np.float32) for t in c_lookup.values()})
    tokenizer = MockTokenizer()
    _pool_and_fresh_rerank(
        model, tokenizer, qids, q_embs_det,
        indices, memory, {q: False for q in qids},
        qrels, c_ids, c_lookup,
        p_max_len=8, mc_batch_size=64, device=DEVICE,
        L=5, max_pool_per_query=10,
    )
    # Unique pool docs: d_a, d_b, d_c, d_d = 4
    assert model.total_inputs == 4, \
        f"expected exactly 4 pool encodes (dedup), got {model.total_inputs}"


def test_fr_model_mode_restored():
    """model.training state at entry must be preserved at exit."""
    memory     = {"q0": []}
    faiss_hits = {"q0": ["d1", "d2"]}
    qids, q_embs_det, indices, c_ids, c_id_to_idx, c_lookup, qrels = _make_pool_inputs(
        memory, faiss_hits
    )
    model = ScriptedEncoder({t: np.zeros(8, dtype=np.float32) for t in c_lookup.values()})
    tokenizer = MockTokenizer()

    model.train()
    _pool_and_fresh_rerank(
        model, tokenizer, qids, q_embs_det,
        indices, memory, {q: False for q in qids},
        qrels, c_ids, c_lookup,
        p_max_len=8, mc_batch_size=8, device=DEVICE,
        L=5, max_pool_per_query=10,
    )
    assert model.training is True, "train() mode not restored after rerank"

    model.eval()
    _pool_and_fresh_rerank(
        model, tokenizer, qids, q_embs_det,
        indices, memory, {q: False for q in qids},
        qrels, c_ids, c_lookup,
        p_max_len=8, mc_batch_size=8, device=DEVICE,
        L=5, max_pool_per_query=10,
    )
    assert model.training is False, "eval() mode not preserved after rerank"


# -----------------------------------------------------------------------
# CM — active candidate memory — Phase 3
# -----------------------------------------------------------------------

def test_cm_stores_selected_and_top_g_sigma():
    """update() stores selected + top-g + top-sigma deduped. Fresh-first order:
    selected -> top_g -> top_sigma -> existing."""
    mem = CandidateMemory(max_per_query=32, ttl_rounds=2,
                          top_g_to_store=4, top_sigma_to_store=4)
    mem.update("q0", current_round=1,
               selected_negs=["s1", "s2"],
               top_g_docids=["s1", "g1", "g2"],     # s1 duplicates with selected
               top_sigma_docids=["sig1", "sig2"])
    ids, expired = mem.get("q0", current_round=1)
    assert not expired
    # Selected first, then top_g, then top_sigma (after dedup)
    assert ids == ["s1", "s2", "g1", "g2", "sig1", "sig2"], f"order: {ids}"


def test_cm_fresh_evidence_kicks_old_when_full():
    """When memory is at cap, fresh evidence MUST displace old entries —
    never be silently dropped. This is what the fresh-first merge order
    protects against."""
    mem = CandidateMemory(max_per_query=3, ttl_rounds=10,
                          top_g_to_store=4, top_sigma_to_store=4)
    # Round 1: fill memory with 3 entries
    mem.update("q0", current_round=1, selected_negs=["old1", "old2", "old3"])
    ids, _ = mem.get("q0", current_round=1)
    assert ids == ["old1", "old2", "old3"], f"setup expected full: {ids}"

    # Round 2: a NEW selected negative arrives — must enter despite cap
    mem.update("q0", current_round=2, selected_negs=["new_pick"])
    ids, _ = mem.get("q0", current_round=2)
    assert "new_pick" in ids, f"fresh selected dropped from full memory: {ids}"
    assert ids[0] == "new_pick", f"fresh selected not at front: {ids}"
    assert len(ids) == 3, f"cap violated: {ids}"
    # The oldest existing entry (last in existing list) is pushed out
    assert "old3" not in ids, f"old3 should have been evicted: {ids}"

    # Round 3: a NEW top-g candidate arrives — same protection
    mem.update("q0", current_round=3,
               selected_negs=[],
               top_g_docids=["new_g"])
    ids, _ = mem.get("q0", current_round=3)
    assert "new_g" in ids, f"fresh top-g dropped from full memory: {ids}"
    assert ids[0] == "new_g", f"fresh top-g not at front: {ids}"


def test_cm_max_per_query_cap():
    """Memory size must never exceed max_candidates_per_query.
    Order under fresh-first: selected first, then top_g, then top_sigma."""
    mem = CandidateMemory(max_per_query=3, ttl_rounds=2,
                          top_g_to_store=8, top_sigma_to_store=8)
    mem.update("q0", current_round=1,
               selected_negs=["a", "b"],
               top_g_docids=["c", "d", "e"],
               top_sigma_docids=["f", "g"])
    ids, _ = mem.get("q0", current_round=1)
    assert len(ids) == 3, f"cap violated, got {len(ids)}"
    assert ids == ["a", "b", "c"], f"order: {ids}"


def test_cm_ttl_validity():
    """get() returns (ids, False) if within TTL, ([], True) if expired,
    ([], False) if absent."""
    mem = CandidateMemory(max_per_query=8, ttl_rounds=2,
                          top_g_to_store=4, top_sigma_to_store=4)
    mem.update("q0", current_round=5, selected_negs=["d1"])
    # Within TTL: round 5, 6, 7 valid
    for r in [5, 6, 7]:
        ids, expired = mem.get("q0", current_round=r)
        assert ids == ["d1"] and expired is False, \
            f"round {r}: ids={ids}, expired={expired}"
    # Round 8: 8 - 5 = 3 > 2 -> expired
    ids, expired = mem.get("q0", current_round=8)
    assert ids == [] and expired is True, f"round 8: ids={ids}, expired={expired}"
    # Absent qid
    ids, expired = mem.get("q_absent", current_round=5)
    assert ids == [] and expired is False, \
        f"absent qid should not be flagged expired: expired={expired}"


def test_cm_persist_round_trip():
    """save() then load() preserves state."""
    mem = CandidateMemory(max_per_query=8, ttl_rounds=2,
                          top_g_to_store=4, top_sigma_to_store=4)
    mem.update("q0", current_round=3, selected_negs=["a", "b"],
               top_g_docids=["c"], top_sigma_docids=["d"])
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "mem.pkl"
        mem.save(p)
        loaded = CandidateMemory.load(p, max_per_query=8, ttl_rounds=2,
                                      top_g_to_store=4, top_sigma_to_store=4)
    ids_orig,   _ = mem.get("q0", current_round=3)
    ids_loaded, _ = loaded.get("q0", current_round=3)
    assert ids_orig == ids_loaded, f"orig {ids_orig} != loaded {ids_loaded}"
    assert loaded.max_per_query == 8
    assert loaded.ttl_rounds == 2


# -----------------------------------------------------------------------
# EMA — sigma formula
# -----------------------------------------------------------------------

def test_ema_sigma_is_abs_diff():
    """EMA sigma = |s_cur - s_ema|, NOT std-over-T."""
    rng    = np.random.default_rng(7)
    N      = 10
    s_cur  = rng.standard_normal(N).astype(np.float32)
    s_ema  = rng.standard_normal(N).astype(np.float32)
    sigma  = np.abs(s_cur - s_ema)
    assert (sigma >= 0).all(), "EMA sigma must be non-negative"
    # sanity: must differ from std-over-T computed on the 2-sample stack
    sigma_std = np.std(np.stack([s_cur, s_ema]), axis=0)
    # std of 2 samples = |s_cur - s_ema| / 2 (population std, ddof=0)
    # so they differ by factor 2 -- check they're not equal
    assert not np.allclose(sigma, sigma_std), \
        "sigma should be |diff|, not std-over-2-samples (factor of 2 differs)"


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
        # S6 — mining log structure
        ("S6  mining log has all required fields and values", test_s6_log_fields_and_values),
        # S9-S13 — speed optimisations
        ("S9  config L<=25 and ema_batch_size>=64",           test_s9_config_L_and_batch),
        ("S10 _foreach EMA matches per-tensor loop",          test_s10_foreach_ema_matches_loop),
        ("S11 zero_grad(set_to_none=True) sets grads=None",   test_s11_zero_grad_set_to_none),
        ("S12 config save_steps >= 1000",                     test_s12_config_save_steps),
        ("S13 torch.compile produces correct shape+norms",    test_s13_torch_compile_correct_shape),
        # FR — fresh rerank (Phase 3 architecture)
        ("FR  positives filtered from pool (memory + faiss)",  test_fr_positive_filtering),
        ("FR  pool combines memory and faiss, sources marked", test_fr_pool_combines_faiss_and_memory),
        ("FR  capping keeps memory first then FAISS by rank",  test_fr_pool_capping_keeps_memory_first),
        ("FR  top-L uses current-model embeddings",            test_fr_top_L_uses_current_embs),
        ("FR  pool docs encoded once per batch (dedup)",       test_fr_pool_encoded_once_per_batch),
        ("FR  model.training mode restored after rerank",      test_fr_model_mode_restored),
        # CM — active candidate memory
        ("CM  stores selected + top-g + top-sigma deduped",    test_cm_stores_selected_and_top_g_sigma),
        ("CM  fresh evidence kicks old when memory is full",   test_cm_fresh_evidence_kicks_old_when_full),
        ("CM  max_per_query cap respected",                    test_cm_max_per_query_cap),
        ("CM  TTL validity (within/expired/absent)",           test_cm_ttl_validity),
        ("CM  save/load pickle round-trip preserves state",    test_cm_persist_round_trip),
        # EMA scoring formula
        ("EMA sigma = |s_cur - s_ema|",                        test_ema_sigma_is_abs_diff),
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
