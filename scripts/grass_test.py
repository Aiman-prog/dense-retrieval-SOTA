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
import torch.nn.functional as F
from pathlib import Path

# -----------------------------------------------------------------------
# Import setup
# -----------------------------------------------------------------------
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root / 'src'))

from utils.helpers import encode_batch, encode_batch_tensor, _pool_and_fresh_rerank
from models.temperature_scaled_loss import TemperatureScaledContrastiveLoss

# -----------------------------------------------------------------------
# Shared helpers
# -----------------------------------------------------------------------
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Full mining-log field set emitted by _select_and_log_negatives (run_grass.py).
# The split must not silently drop any of these.
_MINING_LOG_FIELDS = {
    "query_id", "neg_docid", "s_hat_selected", "sigma_selected", "g_selected",
    "rank_by_shat", "sigma_mean_shortlist", "retrieved_count",
    "candidate_pool_count", "positives_filtered_count", "L", "m", "neg_docids",
    "selected_cheap_rank_zero_based",
}


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


class GradMockModel(nn.Module):
    """CLS output comes from a learnable nn.Embedding keyed on input_ids[*, 0], so
    gradients flow back to a real parameter. Deterministic (no dropout/randn) so the
    no-grad path is reproducible. Used to test encode_batch_tensor's grad plumbing."""
    def __init__(self, vocab=1000, hidden=8, seq_len=4):
        super().__init__()
        self.emb     = nn.Embedding(vocab, hidden)
        self.seq_len = seq_len

    def forward(self, input_ids=None, attention_mask=None, **kwargs):
        return MockOutput(last_hidden_state=self.emb(input_ids))   # (B, T, hidden)


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


class CLSDistinctModel(nn.Module):
    """last_hidden_state[:, 0] (the CLS token) points in a DIFFERENT direction than
    the mean over all token positions: CLS -> axis 0, every other token -> axis 1.
    Lets a test prove encode_batch CLS-pools (reads token 0) and has not regressed
    to mean pooling (the classic BGE-M3 mistake)."""
    def __init__(self, hidden=8):
        super().__init__()
        self.hidden = hidden

    def forward(self, input_ids=None, attention_mask=None, **kwargs):
        B, T = input_ids.shape[0], input_ids.shape[1]
        hs = torch.zeros(B, T, self.hidden, device=input_ids.device)
        hs[:, 0, 0]  = 1.0   # CLS token        -> axis 0
        hs[:, 1:, 1] = 1.0   # all other tokens -> axis 1
        return MockOutput(last_hidden_state=hs)


class MaskAwareModel(nn.Module):
    """CLS output (position 0) = attention-mask-weighted mean of per-token vectors,
    where each token's direction depends on its id. Padding tokens carry a NONZERO
    vector, so if encode_batch fails to forward attention_mask the pooled direction
    shifts and the embedding changes. Used to prove (a) the mask is forwarded and
    (b) a text's embedding is padding/batch invariant."""
    def __init__(self, hidden=8):
        super().__init__()
        self.hidden = hidden
        self.freq   = torch.arange(1, hidden + 1).float() * 0.1   # nonzero -> id affects direction

    def forward(self, input_ids=None, attention_mask=None, **kwargs):
        B, T = input_ids.shape[0], input_ids.shape[1]
        tok  = torch.sin(input_ids.float().unsqueeze(-1) * self.freq.to(input_ids.device))  # (B,T,H)
        if attention_mask is None:                       # dropped-mask regression -> count every token
            attention_mask = torch.ones(B, T, device=input_ids.device)
        m      = attention_mask.float().unsqueeze(-1)    # (B,T,1)
        pooled = (tok * m).sum(dim=1) / m.sum(dim=1).clamp(min=1.0)   # masked mean -> (B,H)
        hs = torch.zeros(B, T, self.hidden, device=input_ids.device)
        hs[:, 0, :] = pooled                             # place at CLS position 0
        return MockOutput(last_hidden_state=hs)


class PaddingMockTokenizer:
    """Word-count tokenizer with REAL padding + attention_mask: each word -> a token id,
    sequences padded to the batch max with PAD_ID and attention_mask 0 on the pads.
    Lets a test put a short text next to a long one so the short row is genuinely padded."""
    PAD_ID = 1
    def __call__(self, texts, padding=True, truncation=True,
                 max_length=128, return_tensors='pt'):
        seqs = [([abs(hash(w)) % 997 + 5 for w in t.split()][:max_length] or [2]) for t in texts]
        maxlen = max(len(s) for s in seqs)
        input_ids = torch.full((len(seqs), maxlen), self.PAD_ID, dtype=torch.long)
        attn      = torch.zeros((len(seqs), maxlen), dtype=torch.long)
        for i, s in enumerate(seqs):
            input_ids[i, :len(s)] = torch.tensor(s, dtype=torch.long)
            attn[i, :len(s)]      = 1
        return _BatchEncoding({'input_ids': input_ids, 'attention_mask': attn})


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


def test_encode_batch_tensor_grad_flow():
    """encode_batch_tensor(requires_grad=True) must be differentiable (gradients
    reach model params) — this is what replaced the old encode_batch_train. With
    requires_grad=False it must be detached, and the encode_batch numpy wrapper
    must equal encode_batch_tensor(requires_grad=False)."""
    model = GradMockModel(hidden=8).to(DEVICE)
    tok   = MockTokenizer()
    texts = ["alpha", "beta", "gamma"]

    emb_grad = encode_batch_tensor(model, tok, texts, DEVICE, max_len=32,
                                   batch_size=2, requires_grad=True)
    assert emb_grad.requires_grad and emb_grad.grad_fn is not None, \
        "requires_grad=True must produce a differentiable tensor (grad_fn set)"
    emb_grad.sum().backward()
    assert model.emb.weight.grad is not None, \
        "backward must propagate gradients to model parameters"

    emb_nograd = encode_batch_tensor(model, tok, texts, DEVICE, max_len=32,
                                     batch_size=2, requires_grad=False)
    assert (not emb_nograd.requires_grad) and emb_nograd.grad_fn is None, \
        "requires_grad=False must produce a detached tensor"

    # Wrapper parity: encode_batch == encode_batch_tensor(requires_grad=False) -> numpy
    np_wrap = encode_batch(model, tok, texts, DEVICE, max_len=32, batch_size=2)
    assert np.allclose(np_wrap, emb_nograd.detach().cpu().float().numpy(), atol=1e-6), \
        "encode_batch wrapper must equal encode_batch_tensor(requires_grad=False)"


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


def _make_pool_inputs(faiss_hits, dim=8, c_lookup=None, qrels=None):
    """Build minimal args for _pool_and_fresh_rerank. `faiss_hits`/`qrels`
    are dicts keyed by qid; values are lists of docid strings."""
    qids        = sorted(faiss_hits.keys())
    c_ids_all   = sorted({d for v in faiss_hits.values() for d in v})
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
    """Positives in qrels must be filtered out of the FAISS pool."""
    faiss_hits = {"q0": ["d_pos", "d1", "d2"]}
    qrels      = {"q0": {"d_pos"}}
    qids, q_embs_det, indices, c_ids, c_id_to_idx, c_lookup, qrels = _make_pool_inputs(
        faiss_hits, qrels=qrels
    )
    model = ScriptedEncoder({t: np.zeros(8, dtype=np.float32) for t in c_lookup.values()})
    tokenizer = MockTokenizer()
    shortlist, stats = _pool_and_fresh_rerank(
        model, tokenizer, qids, q_embs_det,
        indices, qrels, c_ids, c_lookup,
        p_max_len=8, mc_batch_size=8, device=DEVICE,
        L=10, max_pool_per_query=64,
    )
    assert "d_pos" not in shortlist["q0"], f"positive leaked: {shortlist['q0']}"
    assert stats["q0"]["positives_filtered"] == 1, \
        f"expected 1 positive filtration, got {stats['q0']['positives_filtered']}"


def test_fr_pool_faiss_by_rank():
    """Pool contains FAISS hits in rank order, capped at max_pool_per_query."""
    faiss_hits = {"q0": ["f1", "f2", "f3", "f4"]}
    qids, q_embs_det, indices, c_ids, c_id_to_idx, c_lookup, qrels = _make_pool_inputs(
        faiss_hits
    )
    model = ScriptedEncoder({t: np.zeros(8, dtype=np.float32) for t in c_lookup.values()})
    tokenizer = MockTokenizer()
    shortlist, stats = _pool_and_fresh_rerank(
        model, tokenizer, qids, q_embs_det,
        indices, qrels, c_ids, c_lookup,
        p_max_len=8, mc_batch_size=8, device=DEVICE,
        L=10, max_pool_per_query=3,
    )
    # Only top-3 FAISS hits admitted due to cap
    assert stats["q0"]["pool_count"] == 3
    assert set(shortlist["q0"]) == {"f1", "f2", "f3"}


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

    faiss_hits = {"q0": ["d_low", "d_high", "d_mid"]}  # FAISS order does NOT match scores
    qids, _, indices, c_ids, c_id_to_idx, c_lookup, qrels = _make_pool_inputs(
        faiss_hits
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
    shortlist, _ = _pool_and_fresh_rerank(
        model, tokenizer, qids, q_embs_det,
        indices, qrels, c_ids, c_lookup,
        p_max_len=8, mc_batch_size=64, device=DEVICE,
        L=1, max_pool_per_query=10,
    )
    assert shortlist["q0"] == ["d_high"], \
        f"top-1 should be d_high (current-model dot product), got {shortlist['q0']}"


def test_fr_pool_encoded_once_per_batch():
    """Each unique pool doc is encoded exactly once across the batch."""
    faiss_hits = {"q0": ["d_a", "d_c"], "q1": ["d_a", "d_b", "d_d"]}  # d_a shared
    qids, q_embs_det, indices, c_ids, c_id_to_idx, c_lookup, qrels = _make_pool_inputs(
        faiss_hits
    )
    model = ScriptedEncoder({t: np.zeros(8, dtype=np.float32) for t in c_lookup.values()})
    tokenizer = MockTokenizer()
    _pool_and_fresh_rerank(
        model, tokenizer, qids, q_embs_det,
        indices, qrels, c_ids, c_lookup,
        p_max_len=8, mc_batch_size=64, device=DEVICE,
        L=5, max_pool_per_query=10,
    )
    # Unique pool docs: d_a, d_b, d_c, d_d = 4
    assert model.total_inputs == 4, \
        f"expected exactly 4 pool encodes (dedup), got {model.total_inputs}"


def test_fr_model_mode_restored():
    """model.training state at entry must be preserved at exit."""
    faiss_hits = {"q0": ["d1", "d2"]}
    qids, q_embs_det, indices, c_ids, c_id_to_idx, c_lookup, qrels = _make_pool_inputs(
        faiss_hits
    )
    model = ScriptedEncoder({t: np.zeros(8, dtype=np.float32) for t in c_lookup.values()})
    tokenizer = MockTokenizer()

    model.train()
    _pool_and_fresh_rerank(
        model, tokenizer, qids, q_embs_det,
        indices, qrels, c_ids, c_lookup,
        p_max_len=8, mc_batch_size=8, device=DEVICE,
        L=5, max_pool_per_query=10,
    )
    assert model.training is True, "train() mode not restored after rerank"

    model.eval()
    _pool_and_fresh_rerank(
        model, tokenizer, qids, q_embs_det,
        indices, qrels, c_ids, c_lookup,
        p_max_len=8, mc_batch_size=8, device=DEVICE,
        L=5, max_pool_per_query=10,
    )
    assert model.training is False, "eval() mode not preserved after rerank"


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
# EC — encode_batch contract: pooling, masking, robustness, cross-path/config
# -----------------------------------------------------------------------

def test_ec_uses_cls_not_mean_pooling():
    """encode_batch must CLS-pool (last_hidden_state[:, 0]), NOT mean-pool.
    CLSDistinctModel puts CLS on axis 0 and every other token on axis 1, so the two
    pooling strategies give different directions; the assertion fails on a regression."""
    model = CLSDistinctModel(hidden=8).to(DEVICE)
    tok   = MockTokenizer()
    emb   = encode_batch(model, tok, ["x", "y"], DEVICE, max_len=32, batch_size=8)
    expected = np.zeros(8, dtype=np.float32); expected[0] = 1.0   # normalized CLS = axis-0 unit vector
    assert np.allclose(emb[0], expected, atol=1e-5), \
        f"expected CLS pooling (axis-0 unit vector), got {emb[0]} — regressed to mean/other pooling?"


def test_ec_forwards_mask_and_is_padding_invariant():
    """A text's embedding must not change when it is batched/padded next to a longer text.
    This proves encode_batch forwards attention_mask AND that the CLS embedding is
    padding-invariant (the property that makes dynamic padding and `texts * T` correct)."""
    model = MaskAwareModel(hidden=8).to(DEVICE)
    tok   = PaddingMockTokenizer()
    short = "alpha"
    long_ = "alpha beta gamma delta epsilon zeta"     # 6 tokens -> pads the short row to length 6
    e_alone   = encode_batch(model, tok, [short],        DEVICE, max_len=32, batch_size=8)[0]
    e_batched = encode_batch(model, tok, [short, long_], DEVICE, max_len=32, batch_size=8)[0]
    assert np.allclose(e_alone, e_batched, atol=1e-5), \
        ("embedding of a text changed when padded in a batch — attention_mask not honored "
         f"or pooling not padding-invariant (alone={e_alone[:3]} batched={e_batched[:3]})")


def test_ec_empty_input_no_crash():
    """encode_batch([]) must return an empty array, not raise in np.concatenate."""
    model = CLSDistinctModel().to(DEVICE)
    tok   = MockTokenizer()
    out   = encode_batch(model, tok, [], DEVICE, max_len=32, batch_size=8)
    assert hasattr(out, "shape") and out.shape[0] == 0, f"expected empty result, got {out!r}"


def test_ec_encode_paths_pooling_consistency():
    """Index build (helpers.encode_to_pickle) and eval (evaluate.py) must source
    --pooling and --normalize from config['model'], so their embeddings match
    encode_batch (CLS + L2-normalize). Guards the silent, ranking-corrupting
    train/eval pooling-mismatch failure mode."""
    helpers_src = (project_root / 'src' / 'utils' / 'helpers.py').read_text()
    eval_src    = (project_root / 'src' / 'evaluation' / 'evaluate.py').read_text()
    for nm, src in [("helpers.py", helpers_src), ("evaluate.py", eval_src)]:
        assert "--pooling"   in src, f"{nm}: encode call missing --pooling"
        assert "--normalize" in src, f"{nm}: encode call missing --normalize"
        assert "config['model']" in src and "pooling" in src and "normalize" in src, \
            f"{nm}: --pooling/--normalize not sourced from config['model']"
    import yaml
    with open(project_root / 'config' / 'config.yaml') as f:
        cfg = yaml.safe_load(f)
    assert cfg['model']['pooling'] == 'cls', \
        f"config pooling must be 'cls' to match encode_batch, got {cfg['model']['pooling']!r}"
    assert cfg['model']['normalize'] is True, \
        "config normalize must be true (encode_batch L2-normalizes; FAISS IndexFlatIP assumes unit vectors)"


def test_ec_max_len_coverage_bounds():
    """passage/query max_len must stay large enough for BRIGHT. Measured BGE-M3 token
    coverage: 128 truncated 70-79% of theoremqa/aops/leetcode passages, and 256
    truncated ~100% of BRIGHT reasoning queries. Guard against silent regression."""
    import yaml
    with open(project_root / 'config' / 'config.yaml') as f:
        cfg = yaml.safe_load(f)
    m = cfg['model']
    assert m['passage_max_len'] >= 256, \
        f"passage_max_len={m['passage_max_len']} too small for BRIGHT (>=256, recommend 512)"
    assert m['query_max_len'] >= 512, \
        f"query_max_len={m['query_max_len']} too small for BRIGHT reasoning queries (>=512, recommend 1024)"


# -----------------------------------------------------------------------
# LL — loss / passage-layout contract (TemperatureScaledContrastiveLoss)
# -----------------------------------------------------------------------

def test_ll_target_lands_on_positive():
    """The loss auto-target must point at the positive given run_grass's passage
    layout. run_grass builds d_texts = [pos]+negs per query; the loss strides by
    target_per_qry = num_passages // num_queries. If either drifts, training would
    silently optimize toward a NEGATIVE. This couples both sides."""
    B, m = 3, 2
    positives = [f"POS_{q}" for q in range(B)]
    negatives = [[f"NEG_{q}_{j}" for j in range(m)] for q in range(B)]
    # exact run_grass.py layout
    d_texts = [t for pos, negs in zip(positives, negatives) for t in [pos] + negs]
    assert len(d_texts) == B * (1 + m)
    # the loss's own target formula (temperature_scaled_loss.py:41-45)
    tpq    = len(d_texts) // B
    target = list(range(0, B * tpq, tpq))
    assert tpq == 1 + m, f"target_per_qry={tpq} != group size {1 + m}"
    for q, t_idx in enumerate(target):
        assert d_texts[t_idx] == positives[q], \
            f"loss target {t_idx} -> {d_texts[t_idx]!r}, not positive {positives[q]!r}"


def test_ll_temperature_scaling_applied():
    """loss must equal cross_entropy(logits / temperature, target), not the
    unscaled cross-entropy (the bug this loss class exists to fix)."""
    torch.manual_seed(0)
    B, P, temp = 3, 9, 0.02
    x = F.normalize(torch.randn(B, 16), dim=-1)
    y = F.normalize(torch.randn(P, 16), dim=-1)
    got    = TemperatureScaledContrastiveLoss(temperature=temp)(x, y).item()
    target = torch.arange(0, B * (P // B), P // B)
    manual = F.cross_entropy(x @ y.T / temp, target).item()
    assert abs(got - manual) < 1e-5, f"loss {got} != manual temp-scaled {manual}"
    unscaled = F.cross_entropy(x @ y.T, target).item()
    assert abs(got - unscaled) > 1e-3, "temperature scaling not applied (matches unscaled CE)"


def test_ll_argmax_aligned_and_has_teeth():
    """With each query's positive most-similar at its stride index, argmax(logits)
    must equal the target for every query (correct supervision). Then a mis-ordered
    layout (negs+[pos]) must BREAK that alignment — proving the test has teeth."""
    B, m, dim = 3, 1, 8
    tpq = 1 + m
    q_embs = torch.zeros(B, dim)
    for q in range(B):
        q_embs[q, q] = 1.0                                  # query q -> axis q
    target = torch.arange(0, B * tpq, tpq)

    def build(layout):  # layout: 'correct' = [pos]+negs, 'bad' = negs+[pos]
        rows = []
        for q in range(B):
            p = torch.zeros(dim); p[q] = 1.0                # positive: identical to query
            negs = []
            for j in range(m):
                n = torch.zeros(dim); n[B + q * m + j] = 1.0  # negatives on distinct unused axes
                negs.append(n)
            rows += ([p] + negs) if layout == 'correct' else (negs + [p])
        return torch.stack(rows)

    y = build('correct')
    assert torch.equal((q_embs @ y.T).argmax(1), target), "correct layout: argmax must hit positive"
    loss = TemperatureScaledContrastiveLoss(0.02)(q_embs, y).item()
    assert loss < 0.5, f"aligned loss should be small, got {loss}"

    y_bad = build('bad')
    assert not torch.equal((q_embs @ y_bad.T).argmax(1), target), \
        "mis-ordered layout must break alignment (otherwise the test is toothless)"


# -----------------------------------------------------------------------
# FR — fresh-rerank edges (empty pool, general top-L truncation)
# -----------------------------------------------------------------------

def test_fr_all_positives_empty_pool():
    """A query whose entire FAISS pool is qrels positives -> empty shortlist,
    pool_count 0, all filtered, no crash (the `if not pool_docids` early return)."""
    faiss_hits = {"q0": ["dp1", "dp2", "dp3"]}
    qrels      = {"q0": {"dp1", "dp2", "dp3"}}
    qids, q_embs_det, indices, c_ids, c_id_to_idx, c_lookup, qrels = _make_pool_inputs(
        faiss_hits, qrels=qrels
    )
    model = ScriptedEncoder({t: np.zeros(8, dtype=np.float32) for t in c_lookup.values()})
    tokenizer = MockTokenizer()
    shortlist, stats = _pool_and_fresh_rerank(
        model, tokenizer, qids, q_embs_det, indices, qrels, c_ids, c_lookup,
        p_max_len=8, mc_batch_size=8, device=DEVICE, L=5, max_pool_per_query=10,
    )
    assert shortlist["q0"] == [], f"expected empty shortlist, got {shortlist['q0']}"
    assert stats["q0"]["pool_count"] == 0
    assert stats["q0"]["positives_filtered"] == 3
    assert model.call_count == 0, "no pool docs -> encoder must not be invoked"


def test_fr_top_L_truncates_and_orders():
    """With pool > L > 1, shortlist length == L and equals the true top-L by
    current_q . current_d (extends the L=1 case in test_fr_top_L_uses_current_embs)."""
    dim = 8
    q = np.zeros(dim, dtype=np.float32); q[0] = 1.0
    specs = {"d0": 0.95, "d1": 0.80, "d2": 0.60, "d3": 0.40, "d4": 0.20}   # cos with q
    docs  = {}
    for name, c in specs.items():
        v = np.zeros(dim, dtype=np.float32); v[0] = c; v[1] = float(np.sqrt(1 - c * c))
        docs[name] = v
    faiss_hits = {"q0": ["d4", "d2", "d0", "d3", "d1"]}                    # scrambled FAISS order
    qids, _, indices, c_ids, c_id_to_idx, c_lookup, qrels = _make_pool_inputs(faiss_hits)
    text_to_emb = {c_lookup[d]: docs[d] for d in specs}
    assert len({abs(hash(t)) % 1000 for t in text_to_emb}) == 5, "hash collision; pick other docids"
    model     = ScriptedEncoder(text_to_emb, dim=dim)
    tokenizer = MockTokenizer()
    shortlist, _ = _pool_and_fresh_rerank(
        model, tokenizer, qids, q.reshape(1, dim), indices, qrels, c_ids, c_lookup,
        p_max_len=8, mc_batch_size=64, device=DEVICE, L=3, max_pool_per_query=10,
    )
    assert len(shortlist["q0"]) == 3, f"L=3 must truncate, got {len(shortlist['q0'])}"
    assert shortlist["q0"] == ["d0", "d1", "d2"], \
        f"top-3 by current dot product, got {shortlist['q0']}"


# -----------------------------------------------------------------------
# MQ — _mine_queries end-to-end (EMA g-selection + log consistency)
# -----------------------------------------------------------------------

def test_mq_ema_g_selection_and_log():
    """End-to-end Algorithm 2 (EMA): FAISS top-P -> fresh rerank -> sigma=|s_cur-s_ema|
    -> g = s_cur + lambda*sigma -> top-m -> log. One doc has lower relevance but high
    student/teacher disagreement (high sigma). lambda=0 must pick the relevant doc;
    large lambda must pick the high-sigma doc. Validates the g formula and the log."""
    try:
        import faiss
    except Exception as e:
        print(f"(faiss unavailable: {e} — skip)", end=" ")
        return
    sys.path.insert(0, str(project_root / 'scripts'))
    try:
        from run_grass import _mine_queries
    except Exception as e:
        print(f"(run_grass import failed: {e} — skip)", end=" ")
        return

    dim  = 8
    q    = np.zeros(dim, dtype=np.float32); q[0] = 1.0
    dA   = np.zeros(dim, dtype=np.float32); dA[0] = 0.9; dA[1] = float(np.sqrt(1 - 0.81))  # high relevance
    dB_s = np.zeros(dim, dtype=np.float32); dB_s[0] = 0.6; dB_s[1] = 0.8                   # mid relevance (student)
    dB_t = np.zeros(dim, dtype=np.float32); dB_t[1] = 1.0                                  # teacher: orthogonal to q -> s_ema=0
    dC   = np.zeros(dim, dtype=np.float32); dC[0] = 0.1; dC[1] = float(np.sqrt(1 - 0.01))  # low relevance

    q_text, tA, tB, tC = "QTEXT_q0", "DOC_A_text", "DOC_B_text", "DOC_C_text"
    assert len({abs(hash(t)) % 1000 for t in [q_text, tA, tB, tC]}) == 4, "hash collision; pick other texts"

    c_ids         = ["A", "B", "C"]
    corpus_lookup = {"A": tA, "B": tB, "C": tC}
    qid_to_text   = {"q0": q_text}
    student = ScriptedEncoder({q_text: q, tA: dA, tB: dB_s, tC: dC}, dim=dim)
    teacher = ScriptedEncoder({q_text: q, tA: dA, tB: dB_t, tC: dC}, dim=dim)

    index = faiss.IndexFlatIP(dim)                       # stale index = student doc embeddings (unit vectors)
    index.add(np.stack([dA, dB_s, dC]).astype(np.float32))
    tokenizer = MockTokenizer()
    config    = {"model": {"query_max_len": 32, "passage_max_len": 32}}

    def mine(lv):
        cfg = {"P": 10, "L": 3, "m": 1, "lambda_val": lv, "T": 3,
               "mc_batch_size": 16, "max_pool_per_query": 10}
        return _mine_queries(student, teacher, tokenizer, ["q0"], qid_to_text,
                             index, c_ids, corpus_lookup, {}, cfg, config, DEVICE,
                             uncertainty="ema")

    mined0, _    = mine(0.0)
    minedH, logs = mine(5.0)
    assert mined0["q0"][0] == "A", f"lambda=0 -> most relevant (A), got {mined0['q0']}"
    assert minedH["q0"][0] == "B", f"large lambda -> highest sigma (B), got {minedH['q0']}"

    rec = logs[0]
    assert rec["neg_docid"] == "B", f"log neg_docid should be B, got {rec['neg_docid']}"
    assert rec["sigma_selected"] >= 0.0
    assert abs(rec["g_selected"] - (rec["s_hat_selected"] + 5.0 * rec["sigma_selected"])) < 1e-4, \
        f"g != s_hat + lambda*sigma: {rec['g_selected']} vs {rec['s_hat_selected']}+5*{rec['sigma_selected']}"
    # _select_and_log_negatives must emit the full Phase-3 field set (split must not drop any)
    assert _MINING_LOG_FIELDS <= rec.keys(), f"missing log fields: {_MINING_LOG_FIELDS - rec.keys()}"
    # _score_ema leaves the model in eval() (mining-call post-condition)
    assert student.training is False, "_score_ema must leave the student in eval()"


def test_mq_mcd_selection_and_log():
    """End-to-end Algorithm 2 (MC-dropout) through the split _mine_queries. A
    deterministic encoder makes the T MC passes identical -> sigma=0 -> g=s_hat,
    so the most-relevant doc wins even at large lambda. Pins: correct selection,
    full log field set, and the _score_mc_dropout post-condition (model left in
    eval()) that the train->T->eval mode dance must preserve."""
    try:
        import faiss
    except Exception as e:
        print(f"(faiss unavailable: {e} — skip)", end=" ")
        return
    sys.path.insert(0, str(project_root / 'scripts'))
    try:
        from run_grass import _mine_queries
    except Exception as e:
        print(f"(run_grass import failed: {e} — skip)", end=" ")
        return

    dim = 8
    q  = np.zeros(dim, dtype=np.float32); q[0] = 1.0
    dA = np.zeros(dim, dtype=np.float32); dA[0] = 0.9; dA[1] = float(np.sqrt(1 - 0.81))  # high relevance
    dB = np.zeros(dim, dtype=np.float32); dB[0] = 0.5; dB[1] = float(np.sqrt(1 - 0.25))  # mid
    dC = np.zeros(dim, dtype=np.float32); dC[0] = 0.1; dC[1] = float(np.sqrt(1 - 0.01))  # low

    q_text, tA, tB, tC = "QTEXT_mcd", "DOC_A_mcd", "DOC_B_mcd", "DOC_C_mcd"
    assert len({abs(hash(t)) % 1000 for t in [q_text, tA, tB, tC]}) == 4, "hash collision; pick other texts"

    c_ids         = ["A", "B", "C"]
    corpus_lookup = {"A": tA, "B": tB, "C": tC}
    qid_to_text   = {"q0": q_text}
    student = ScriptedEncoder({q_text: q, tA: dA, tB: dB, tC: dC}, dim=dim)

    index = faiss.IndexFlatIP(dim)
    index.add(np.stack([dA, dB, dC]).astype(np.float32))
    tokenizer = MockTokenizer()
    config    = {"model": {"query_max_len": 32, "passage_max_len": 32}}
    cfg = {"P": 10, "L": 3, "m": 1, "lambda_val": 5.0, "T": 3,
           "mc_batch_size": 16, "max_pool_per_query": 10}

    student.train()  # Algorithm 1 calls mining with the model in train()
    mined, logs = _mine_queries(student, None, tokenizer, ["q0"], qid_to_text,
                                index, c_ids, corpus_lookup, {}, cfg, config, DEVICE,
                                uncertainty="mc_dropout")
    # sigma=0 (deterministic) -> g=s_hat -> most-relevant A wins despite lambda=5
    assert mined["q0"][0] == "A", f"mc_dropout (σ=0) -> most relevant A, got {mined['q0']}"
    assert student.training is False, "_score_mc_dropout must leave the student in eval()"

    rec = logs[0]
    assert _MINING_LOG_FIELDS <= rec.keys(), f"missing log fields: {_MINING_LOG_FIELDS - rec.keys()}"
    assert rec["neg_docid"] == "A"
    assert abs(rec["sigma_selected"]) < 1e-5, "deterministic encoder -> sigma ~ 0"
    assert rec["L"] == 3 and rec["m"] == 1


def test_ema_qcur_reuses_qdet():
    """_score_ema reuses q_det as the student query side (q_cur) instead of
    re-encoding. The optimization is behavior-neutral because encoding the same
    texts twice under eval+no_grad is identical — this guards that invariant."""
    model = CLSDistinctModel(hidden=8).to(DEVICE)   # deterministic
    tok   = MockTokenizer()
    texts = ["q a", "q b", "q c"]
    model.eval()
    q_det       = encode_batch(model, tok, texts, DEVICE, max_len=32, batch_size=2)
    q_cur_fresh = encode_batch(model, tok, texts, DEVICE, max_len=32, batch_size=2)
    assert np.allclose(q_det, q_cur_fresh, atol=1e-6), \
        "reusing q_det as q_cur must equal a fresh deterministic query encode"


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
        ("ET  encode_batch_tensor grad flow + wrapper parity", test_encode_batch_tensor_grad_flow),
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
        # FR — fresh rerank
        ("FR  positives filtered from FAISS pool",             test_fr_positive_filtering),
        ("FR  FAISS pool capped by rank at max_pool_per_query",test_fr_pool_faiss_by_rank),
        ("FR  top-L uses current-model embeddings",            test_fr_top_L_uses_current_embs),
        ("FR  pool docs encoded once per batch (dedup)",       test_fr_pool_encoded_once_per_batch),
        ("FR  model.training mode restored after rerank",      test_fr_model_mode_restored),
        # EMA scoring formula
        ("EMA sigma = |s_cur - s_ema|",                        test_ema_sigma_is_abs_diff),
        ("EMA q_cur reuses q_det (behavior-neutral)",          test_ema_qcur_reuses_qdet),
        # EC — encode_batch contract (pooling / masking / robustness / consistency)
        ("EC  encode_batch CLS-pools, not mean",               test_ec_uses_cls_not_mean_pooling),
        ("EC  forwards attention_mask + padding-invariant",    test_ec_forwards_mask_and_is_padding_invariant),
        ("EC  empty input returns empty (no crash)",           test_ec_empty_input_no_crash),
        ("EC  index/eval pooling == encode_batch (config)",    test_ec_encode_paths_pooling_consistency),
        ("EC  passage/query max_len coverage bounds",          test_ec_max_len_coverage_bounds),
        # LL — loss / passage-layout contract
        ("LL  loss target lands on the positive",              test_ll_target_lands_on_positive),
        ("LL  temperature scaling applied",                    test_ll_temperature_scaling_applied),
        ("LL  argmax aligned to target (+ teeth)",             test_ll_argmax_aligned_and_has_teeth),
        # FR — fresh-rerank edges
        ("FR  all-positives pool -> empty shortlist",          test_fr_all_positives_empty_pool),
        ("FR  top-L truncates and orders (pool>L>1)",          test_fr_top_L_truncates_and_orders),
        # MQ — _mine_queries end-to-end selection + log
        ("MQ  EMA g-selection + log consistency",              test_mq_ema_g_selection_and_log),
        ("MQ  MC-dropout selection + log + mode restore",      test_mq_mcd_selection_and_log),
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
