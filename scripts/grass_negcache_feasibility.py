"""
Negative-Cache-inspired Fast-GRASS — pre-training FEASIBILITY analysis (NO training).

Difference from the two-set design (scripts/grass_twoset_feasibility.py):
  Two-set:   per-query active sets A_q; H is just their union; each query scores
             only its own A_q.
  This file: ONE bounded GLOBAL document cache H of size B_doc (a fraction of the
             corpus). H is NOT per-query. Every query scores against ALL of H
             using cached document states Z_H, then picks negatives by the GRASS
             score g = s_hat + lambda*sigma. Only batch queries, positives, and
             the *selected* negatives are fresh-encoded for training.

What this proves: feasibility + sampler correctness, NOT model quality.
  - cache memory fits a budget,
  - refreshing only |H| (not the full corpus) is cheaper than ANCE,
  - scoring Q_batch x Z_H is much cheaper than the current fresh-rerank,
  - the GRASS sampler selects exactly top-m by g with positives masked,
  - the per-batch fresh-encode count drops the P/L candidate rerank,
  - a single real minibatch runs end-to-end with finite loss.

All read-only: no gradient steps, no checkpoints, no data/config mutation.
Heavy steps (query encode, doc-encode probe, real minibatch) need a GPU; the
budget / throughput / counting math is cheap and reuses cached query embeddings.

Run (DelftBlue, inside the pytorch container) — see CLI at bottom of this file.
"""
import argparse
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np

project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root / 'src'))

from utils.helpers import (                                  # noqa: E402
    get_path, load_config, build_faiss_index, _load_qrels, set_seed,
)

GiB = 1e9


# ──────────────────────────────────────────────────────────────────────────────
# Self-contained query/search/candidate machinery
#
# These were previously imported from scripts/grass_twoset_feasibility.py (the
# two-set design). That module was never committed, so the four functions it
# exported — encode_queries, build_candidate_matrix, load_train_queries,
# cached_grass_sampler — plus the _topk_neighbors helper are inlined here to make
# this feasibility probe self-contained. All read-only.
# ──────────────────────────────────────────────────────────────────────────────
def load_train_queries(debug):
    """Unique training queries from the mixture. Returns (qids, qid_to_text)."""
    mix_dir = get_path("processed") / "training_mixture"
    qid_to_text = {}
    for f_path in sorted(mix_dir.glob("*.jsonl")):
        if f_path.name.startswith('.'):
            continue
        with open(f_path) as f:
            for line in f:
                d = json.loads(line)
                if not d.get('positive_passages'):
                    continue
                qid_to_text.setdefault(str(d['query_id']), d['query'])
    qids = list(qid_to_text.keys())
    if debug:
        qids = qids[:2000]
        qid_to_text = {q: qid_to_text[q] for q in qids}
    return qids, qid_to_text


def encode_queries(qids, qid_to_text, base_model, q_max, mc_bs, cache_path,
                   reuse_cache):
    """Encode unique train queries once with the current encoder. Cache to pickle.

    Returns embeddings aligned to `qids` (np.float32, L2-normalised CLS).
    """
    if reuse_cache and cache_path.exists():
        with open(cache_path, 'rb') as f:
            cached = pickle.load(f)
        if cached['qids'] == qids and cached.get('base_model') == base_model:
            print(f"[feas] reusing cached query embeddings: {cache_path}", flush=True)
            return cached['embs']
        print("[feas] query cache stale (qids/model changed) — re-encoding", flush=True)

    import torch
    from transformers import AutoTokenizer, AutoModel
    from utils.helpers import encode_batch

    device    = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    model     = AutoModel.from_pretrained(
        base_model,
        torch_dtype=torch.bfloat16 if device.type == 'cuda' else torch.float32,
    ).to(device)
    model.eval()

    texts = [qid_to_text[q] for q in qids]
    t0    = time.time()
    embs  = encode_batch(model, tokenizer, texts, device, q_max, mc_bs).astype(np.float32)
    print(f"[feas] encoded {len(qids)} queries in {time.time() - t0:.1f}s "
          f"→ dim {embs.shape[1]}", flush=True)

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, 'wb') as f:
        pickle.dump({'qids': qids, 'embs': embs, 'base_model': base_model}, f)

    del model
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    return embs


def _topk_neighbors(q_embs, doc_embs, stale_idx, search_k, force_faiss_cpu):
    """Top-`search_k` corpus indices per query.

    GPU path (default when CUDA is available): a tiled torch matmul on the GPU.
    We deliberately do NOT use faiss-gpu here — its GpuIndexFlat matmul aborts
    with a cuBLAS failure on the small/MIG A100 slice (uncatchable core dump).
    torch's cuBLAS path works on the same device (it already did the encode).

    CPU fallback: faiss IndexFlatIP.search (exact, same result, slower on few cores).
    Returns I (Nq, search_k) int64 of corpus row indices, FAISS/score rank order.
    """
    Nq = q_embs.shape[0]

    use_gpu = not force_faiss_cpu
    if use_gpu:
        try:
            import torch
            use_gpu = torch.cuda.is_available()
        except ImportError:
            use_gpu = False

    if use_gpu:
        import torch
        device = torch.device('cuda')
        docs   = torch.from_numpy(np.ascontiguousarray(doc_embs)).to(device, torch.float32)
        out    = np.empty((Nq, search_k), dtype=np.int64)
        q_tile = 1024
        t0     = time.time()
        for s in range(0, Nq, q_tile):
            e = min(s + q_tile, Nq)
            q = torch.from_numpy(np.ascontiguousarray(q_embs[s:e])).to(device, torch.float32)
            scores = q @ docs.T
            idx    = torch.topk(scores, search_k, dim=1, sorted=True).indices
            out[s:e] = idx.cpu().numpy()
            del q, scores, idx
            if (s // q_tile) % 20 == 0:
                print(f"[feas]   searched {e}/{Nq} queries on GPU "
                      f"({time.time() - t0:.0f}s)", flush=True)
        del docs
        torch.cuda.empty_cache()
        print(f"[feas] GPU torch search done in {time.time() - t0:.0f}s", flush=True)
        return out

    import os
    import faiss
    faiss.omp_set_num_threads(max(int(os.environ.get('OMP_NUM_THREADS', '2')), 1))
    print("[feas] CPU faiss flat search (no/forced-off GPU)", flush=True)
    out  = np.empty((Nq, search_k), dtype=np.int64)
    tile = 4096
    t0   = time.time()
    for s in range(0, Nq, tile):
        e = min(s + tile, Nq)
        _, idx = stale_idx.search(np.ascontiguousarray(q_embs[s:e]), search_k)
        out[s:e] = idx
        if (s // tile) % 10 == 0:
            print(f"[feas]   searched {e}/{Nq} queries "
                  f"({time.time() - t0:.0f}s)", flush=True)
    return out


def build_candidate_matrix(q_embs, stale_idx, stale_embs, c_ids, qrels_dict,
                           qids, P0, force_faiss_cpu):
    """Search the stale index and build C_q (top-P0, positives removed).

    Returns:
      C_mat  (Nq, P0) int32  — corpus indices, -1 padded, score/FAISS rank order
      C_len  (Nq,)    int32  — valid length per query (<= P0)
      pos_idx_per_q  list[set[int]]  — per-query qrels-positive corpus indices,
                               aligned to qids. Contamination is checked PER
                               QUERY (C_q ∩ qrels[q]); a doc that is positive for
                               another query is a legitimate negative here.
    """
    c_id_to_idx = {d: i for i, d in enumerate(c_ids)}
    Nq = len(qids)

    # over-retrieve enough to survive positive removal for the worst-case query.
    max_pos  = max((len(qrels_dict.get(q, ())) for q in qids), default=0)
    search_k = P0 + max(8, max_pos + 4)

    I = _topk_neighbors(q_embs, stale_embs, stale_idx, search_k, force_faiss_cpu)

    C_mat = np.full((Nq, P0), -1, dtype=np.int32)
    C_len = np.zeros(Nq, dtype=np.int32)
    pos_idx_per_q = [
        {c_id_to_idx[d] for d in qrels_dict.get(q, ()) if d in c_id_to_idx}
        for q in qids
    ]

    for r in range(Nq):
        pis  = pos_idx_per_q[r]
        seen = set()
        row  = C_mat[r]
        n    = 0
        for idx in I[r]:
            if n >= P0:
                break
            if idx < 0 or idx in pis or idx in seen:
                continue
            seen.add(idx)
            row[n] = idx
            n += 1
        C_len[r] = n

    return C_mat, C_len, pos_idx_per_q


def cached_grass_sampler(zq, zH_stack, lam, m):
    """Score active docs for one query against the CACHED doc states (MC-dropout).

    zq       : (T, dim)            query's T MC-dropout embeddings
    zH_stack : (n_active, T, dim)  cached T MC embeddings for the query's A_q docs
    Returns  : (top_m_local_idx, s_hat, sigma, g)
    """
    sims  = np.einsum('tk,ntk->nt', zq, zH_stack)
    s_hat = sims.mean(axis=1)
    sigma = sims.std(axis=1)
    g     = s_hat + lam * sigma
    top_m = np.argsort(g)[::-1][:m]
    return top_m, s_hat, sigma, g


# ──────────────────────────────────────────────────────────────────────────────
# Global cache selection
# ──────────────────────────────────────────────────────────────────────────────
def select_global_H(C_mat, C_len, B_doc, n_corpus):
    """Pick the GLOBAL cache H of size B_doc by retrieval frequency.

    A doc that is retrieved (in top-P0, positives already removed) by many queries
    is a globally useful hard negative. Rank corpus docs by that frequency, take
    the top B_doc. Deterministic; ties broken by corpus index.

    `freq` spans the FULL corpus (minlength=n_corpus) so never-retrieved docs keep
    a real slot (frequency 0) instead of vanishing from the index universe.

    Returns:
      H_idx   (B_doc,) int64 corpus indices, frequency-descending
      freq    (n_corpus,) int32 retrieval count per corpus index
    """
    # bincount over all valid candidate indices across every query's C_q.
    flat = C_mat[C_mat >= 0].astype(np.int64)
    freq = np.bincount(flat, minlength=n_corpus).astype(np.int32)
    # top-B_doc by (freq desc, idx asc): negate freq for a stable ascending sort.
    order = np.lexsort((np.arange(freq.size), -freq))
    H_idx = order[:B_doc].astype(np.int64)
    return H_idx, freq


def cache_overlap_stats(C_mat, C_len, H_set, m):
    """Diagnostic (not gating): how much of each query's own hard region survives
    in the global cache. |C_q ∩ H| per query. Reports availability of >= m."""
    Nq = C_mat.shape[0]
    overlap = np.zeros(Nq, dtype=np.int32)
    for r in range(Nq):
        row = C_mat[r]
        overlap[r] = sum(1 for idx in row[row >= 0] if idx in H_set)
    return dict(
        min=int(overlap.min()), mean=float(overlap.mean()),
        median=float(np.median(overlap)),
        ge_m_frac=float((overlap >= m).mean()),
        zero_frac=float((overlap == 0).mean()),
    )


# ──────────────────────────────────────────────────────────────────────────────
# Test 1 — cache budget
# ──────────────────────────────────────────────────────────────────────────────
def test_cache_budget(budget_fracs, operating_frac, n_corpus, dim, T, max_cache_gb):
    """Z_H holds T MC embeddings per cached doc: bytes = T * |H| * dim * dtype.

    The sweep is informational; the PASS is gated on the OPERATING fraction
    (--b_doc_frac), because that is the cache the rest of the run actually uses.
    """
    fracs = sorted(set(budget_fracs) | {operating_frac})
    print("\n[T1] Cache budget  (Z_H = T x |H| x dim ; T={}, dim={}, budget={:.1f} GB)"
          .format(T, dim, max_cache_gb))
    print(f"      {'frac':>6} {'|H|':>10} {'fp32 GB':>9} {'bf16 GB':>9}  fits(bf16)")
    rows = []
    op_fits = False
    for fr in fracs:
        H = int(round(fr * n_corpus))
        gb32 = T * H * dim * 4 / GiB
        gb16 = T * H * dim * 2 / GiB
        fits = gb16 <= max_cache_gb
        is_op = abs(fr - operating_frac) < 1e-9
        op_fits = op_fits or (is_op and fits)
        rows.append((fr, H, gb32, gb16, fits))
        print(f"      {fr:>6.1%} {H:>10,} {gb32:>9.2f} {gb16:>9.2f}  "
              f"{'✅' if fits else '❌'}{'  <- operating' if is_op else ''}")
    ok = op_fits
    print(f"      => {'PASS' if ok else 'FAIL'}: operating frac {operating_frac:.1%} "
          f"{'fits' if ok else 'EXCEEDS'} the {max_cache_gb:.1f} GB bf16 budget")
    return ok, rows


# ──────────────────────────────────────────────────────────────────────────────
# Test 2 — cache encode speed (vs ANCE full-corpus refresh)
# ──────────────────────────────────────────────────────────────────────────────
def test_encode_speed(H_size, n_corpus, T, base_model, corpus_file, c_ids,
                      H_idx, p_max, mc_bs, mc_drop_p, n_probe, skip):
    """Time the REAL MCDP cache build and compare to an ANCE full-corpus refresh.

    The proposed build is the vectorized MCDP encode `EncodeActiveDocs`: encode
    `texts * T` in one batched call with dropout ACTIVE (model.train()). T2 times
    exactly that on a probe, plus a separate clean (eval) pass for the ANCE
    baseline, then extrapolates by passage count.

    PASS if estimated cache-refresh cost <= 0.5x a full-corpus encode (ANCE=1.0x).
    """
    if skip:
        # Conservative analytic estimate: encode cost scales with #passages.
        # MCDP refresh = T passes over |H| ; ANCE refresh = 1 pass over corpus.
        ratio = T * H_size / n_corpus
        ok = ratio <= 0.5
        print(f"\n[T2] Encode speed (SKIPPED real timing; analytic): "
              f"T x |H| / corpus = {T}x{H_size:,}/{n_corpus:,} = {ratio:.2f}x ANCE "
              f"=> {'PASS' if ok else 'FAIL'} (<=0.50x)")
        return ok, dict(mode='analytic', ratio=ratio)

    import torch
    from transformers import AutoTokenizer, AutoModel
    from utils.helpers import encode_batch, _load_corpus_lookup

    device    = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dtype     = torch.bfloat16 if device.type == 'cuda' else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    model     = AutoModel.from_pretrained(base_model, torch_dtype=dtype).to(device)

    corpus_lookup = _load_corpus_lookup(corpus_file)
    probe_idx = H_idx[:n_probe]
    texts     = [corpus_lookup.get(c_ids[i], "") for i in probe_idx]

    # --- ANCE baseline: one clean (eval, no dropout) pass over the probe ---
    model.eval()
    if device.type == 'cuda':
        torch.cuda.synchronize()
    t0 = time.time()
    _ = encode_batch(model, tokenizer, texts, device, p_max, mc_bs)
    if device.type == 'cuda':
        torch.cuda.synchronize()
    clean_dt   = time.time() - t0
    clean_dps  = len(texts) / clean_dt if clean_dt > 0 else float('inf')

    # --- Real MCDP cache build: vectorized `texts * T`, dropout ACTIVE ---
    for mod in model.modules():
        if isinstance(mod, torch.nn.Dropout):
            mod.p = mc_drop_p
    model.train()
    if device.type == 'cuda':
        torch.cuda.synchronize()
    t0 = time.time()
    _ = encode_batch(model, tokenizer, texts * T, device, p_max, mc_bs)
    if device.type == 'cuda':
        torch.cuda.synchronize()
    build_dt   = time.time() - t0                  # time to MC-build |probe| docs
    build_dps  = (len(texts) * T) / build_dt if build_dt > 0 else float('inf')

    # Extrapolate by passage count.
    refresh_s = build_dt * (H_size / len(texts))   # MC-build for all of |H|
    ance_s    = n_corpus / clean_dps               # one clean pass over the corpus
    ratio     = refresh_s / ance_s
    ok = ratio <= 0.5
    del model
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    print(f"\n[T2] Encode speed (real MCDP build, probe={len(texts)}):")
    print(f"      clean pass : {clean_dt:.1f}s = {clean_dps:,.0f} docs/s  (ANCE baseline)")
    print(f"      MCDP build : {build_dt:.1f}s for {len(texts)}x{T} passes "
          f"= {build_dps:,.0f} passes/s")
    print(f"      cache refresh (|H|={H_size:,}, dropout) ~= {refresh_s/60:.1f} min   |   "
          f"ANCE corpus refresh ~= {ance_s/60:.1f} min")
    print(f"      => ratio {ratio:.2f}x ANCE  {'PASS' if ok else 'FAIL'} (<=0.50x)")
    return ok, dict(mode='real', clean_dps=clean_dps, build_dps=build_dps,
                    refresh_s=refresh_s, ance_s=ance_s, ratio=ratio)


# ──────────────────────────────────────────────────────────────────────────────
# Test 3 — batch scoring throughput  (Q_batch x Z_H, MCDP s_hat/sigma)
# ──────────────────────────────────────────────────────────────────────────────
def _score_against_cache(ZQ, ZH, lam, m, pos_mask=None):
    """ZQ (B,T,dim), ZH (H,T,dim) torch. Returns top-m indices (B,m), g (B,H).
    MCDP: per-pass sims, s_hat=mean_t, sigma=std_t, g=s_hat+lam*sigma."""
    import torch
    sims  = torch.einsum('btd,ntd->btn', ZQ, ZH)        # (B,T,H)
    s_hat = sims.mean(dim=1)                              # (B,H)
    # population std (correction=0) to match NumPy ddof=0 used by current GRASS /
    # cached_grass_sampler — otherwise g differs by the Bessel factor sqrt(T/(T-1)).
    sigma = sims.std(dim=1, unbiased=False)              # (B,H)
    g     = s_hat + lam * sigma
    if pos_mask is not None:
        g = g.masked_fill(pos_mask, float('-inf'))       # never select a positive
    topm  = torch.topk(g, m, dim=1).indices              # (B,m)
    return topm, g


def test_scoring_throughput(q_embs, H_size, dim, T, batch_sizes, lam, m,
                            n_corpus, max_epoch_scoring_min, n_iters=10):
    """Time Q_batch x Z_H scoring on GPU (CPU if no GPU). Z_H is synthesised at the
    real target shape (T,H,dim) — throughput is governed by matmul size, not
    content. Q rows are sampled from the real cached query embeddings (tiled to T).

    PASS if estimated full-epoch scoring time <= --max_epoch_scoring_min.
    """
    import torch
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dtype  = torch.float16 if device.type == 'cuda' else torch.float32

    ZH = torch.randn(H_size, T, dim, device=device, dtype=dtype)
    ZH = torch.nn.functional.normalize(ZH, dim=-1)

    Nq = q_embs.shape[0]
    print(f"\n[T3] Scoring throughput  (|H|={H_size:,}, T={T}, dim={dim}, "
          f"device={device.type})")
    print(f"      {'batch':>6} {'ms/batch':>9} {'queries/s':>11} {'epoch min':>10}  verdict")
    best_epoch_min = float('inf')
    rows = []
    for B in batch_sizes:
        # real query rows, tiled to T passes (+ tiny noise so std>0, realistic).
        idx  = np.random.randint(0, Nq, size=B)
        base = torch.from_numpy(q_embs[idx]).to(device, dtype)          # (B,dim)
        ZQ   = base[:, None, :].repeat(1, T, 1)
        ZQ   = ZQ + 0.01 * torch.randn_like(ZQ)
        ZQ   = torch.nn.functional.normalize(ZQ, dim=-1)

        for _ in range(2):                                              # warmup
            _score_against_cache(ZQ, ZH, lam, m)
        if device.type == 'cuda':
            torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(n_iters):
            _score_against_cache(ZQ, ZH, lam, m)
        if device.type == 'cuda':
            torch.cuda.synchronize()
        ms = (time.time() - t0) / n_iters * 1e3
        qps = B / (ms / 1e3) if ms > 0 else float('inf')
        # full-epoch scoring time is over ALL training queries.
        epoch_min = Nq / qps / 60 if qps > 0 else float('inf')
        best_epoch_min = min(best_epoch_min, epoch_min)
        rows.append((B, ms, qps, epoch_min))
        print(f"      {B:>6} {ms:>9.2f} {qps:>11,.0f} {epoch_min:>10.2f}")

    ok = best_epoch_min <= max_epoch_scoring_min
    print(f"      => best full-epoch scoring ~= {best_epoch_min:.2f} min "
          f"(budget {max_epoch_scoring_min:.1f} min)  {'PASS' if ok else 'FAIL'}")
    del ZH
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    return ok, dict(rows=rows, best_epoch_min=best_epoch_min)


# ──────────────────────────────────────────────────────────────────────────────
# Test 4 — sampler correctness (synthetic, with positive masking)
# ──────────────────────────────────────────────────────────────────────────────
def test_sampler_correctness(seed, m=2, lam=1.0, T=4, dim=16, H=20):
    """Known synthetic scores: verify selection is exactly top-m by g, and that a
    masked positive is never selected even if it has the highest g."""
    import torch
    rng = np.random.default_rng(seed)
    ZQ  = torch.from_numpy(rng.standard_normal((1, T, dim)).astype(np.float32))
    ZH  = torch.from_numpy(rng.standard_normal((H, T, dim)).astype(np.float32))

    # reference g via the numpy sampler (sanity that both agree)
    _, s_hat, sigma, g_ref = cached_grass_sampler(ZQ[0].numpy(), ZH.numpy(), lam, H)
    # force a known doc to have the max g, then mask it as a positive.
    pos_doc = int(np.argmax(g_ref))
    pos_mask = torch.zeros(1, H, dtype=torch.bool)
    pos_mask[0, pos_doc] = True

    topm, g = _score_against_cache(ZQ, ZH, lam, m, pos_mask=pos_mask)
    topm = topm[0].tolist()

    # expected = top-m by g EXCLUDING the masked positive
    g_np = g_ref.copy()
    g_np[pos_doc] = -np.inf
    expect = np.argsort(g_np)[::-1][:m].tolist()

    ok_topm = (topm == expect)
    ok_mask = (pos_doc not in topm)
    ok_sigma = bool((sigma > 1e-8).all())
    ok = ok_topm and ok_mask and ok_sigma
    print(f"\n[T4] Sampler correctness (synthetic, m={m}): "
          f"top-m==argsort {ok_topm}, masked-positive-excluded {ok_mask}, "
          f"sigma>0 {ok_sigma}  => {'PASS' if ok else 'FAIL'}")
    return ok, dict(selected=topm, expected=expect, masked_positive=pos_doc)


# ──────────────────────────────────────────────────────────────────────────────
# Test 5 — positive masking / contamination on the REAL global cache
# ──────────────────────────────────────────────────────────────────────────────
def test_positive_masking(H_idx, pos_idx_per_q, qids, m):
    """How many queries have >=1 of their positives inside the GLOBAL H, and does
    masking remove them? Selection over H must mask each query's own positives.

    contamination = #selected negatives that are positives AFTER masking (must=0).
    Also reports how often masking is actually needed (positives present in H).
    """
    H_set = set(int(x) for x in H_idx)
    H_list = list(H_set)
    H_pos = {d: i for i, d in enumerate(H_list)}            # corpus idx -> H slot

    q_with_pos_in_H   = 0
    contamination     = 0          # masked positives selected (must be 0)
    adversarial_cases = 0          # queries where, UNMASKED, a positive WOULD top-m
    for r, q in enumerate(qids):
        pis = pos_idx_per_q[r]
        if not pis:
            continue
        pos_in_H = [H_pos[d] for d in pis if d in H_pos]
        if not pos_in_H:
            continue
        q_with_pos_in_H += 1
        # Adversarial g in real H_idx order: force this query's positives to the
        # TOP so masking is the only thing that can exclude them.
        g = np.random.default_rng(r).standard_normal(len(H_list)).astype(np.float64)
        g[pos_in_H] = 1e9
        unmasked_sel = np.argsort(g)[::-1][:m]
        if any(s in pos_in_H for s in unmasked_sel):
            adversarial_cases += 1               # confirms the test isn't tautological
        # now mask the positives (-inf) and re-select — they must be gone.
        g_masked = g.copy()
        g_masked[pos_in_H] = -np.inf
        sel = np.argsort(g_masked)[::-1][:m]
        contamination += sum(1 for s in sel if s in pos_in_H)

    # PASS needs: no contamination AND (if any masking was exercised) the test was
    # actually adversarial for those cases.
    ok = (contamination == 0) and (q_with_pos_in_H == 0 or adversarial_cases > 0)
    print(f"\n[T5] Positive masking on global H (|H|={len(H_list):,}): "
          f"{q_with_pos_in_H}/{len(qids)} queries have a positive in H; "
          f"{adversarial_cases} adversarial (pos would be top-m unmasked); "
          f"contamination after masking = {contamination}  "
          f"=> {'PASS' if ok else 'FAIL'} (==0)")
    return ok, dict(q_with_pos_in_H=q_with_pos_in_H,
                    adversarial_cases=adversarial_cases, contamination=contamination)


# ──────────────────────────────────────────────────────────────────────────────
# Test 6 — fresh-encoding count per batch (new arch vs current GRASS)
# ──────────────────────────────────────────────────────────────────────────────
def test_fresh_encode_count(batch_size, m, P, L):
    """Per-batch fresh encodes (forward passes that need gradients OR fresh rerank):

      NEW (negative cache): queries(B) + positives(B) + selected negatives(B*m).
        NO per-query FAISS top-P retrieval, NO fresh-encode of P candidates,
        NO fresh rerank to L inside the query loop.
      CURRENT GRASS: queries(B) + positives(B) + P-or-L fresh candidate encodes
        per query for the rerank/uncertainty step (B*L at minimum, up to B*P pool).

    PASS if the new count removes the B*L (and B*P) candidate fresh-rerank term.
    """
    new_cnt  = batch_size * (1 + 1 + m)
    cur_low  = batch_size * (1 + 1 + L)         # fresh-rerank to L
    cur_high = batch_size * (1 + 1 + P)         # pool encode to P
    removed  = (new_cnt < cur_low)              # the L/P candidate term is gone
    print(f"\n[T6] Fresh-encode count per batch (B={batch_size}, m={m}, P={P}, L={L}):")
    print(f"      NEW negcache : {new_cnt:>6}  (B + B + B*m)")
    print(f"      GRASS rerank : {cur_low:>6}  (B + B + B*L)   pool-encode: {cur_high:,}")
    print(f"      => {'PASS' if removed else 'FAIL'}: "
          f"P/L candidate fresh-rerank {'eliminated' if removed else 'still present'} "
          f"({cur_low/new_cnt:.1f}x fewer fresh encodes)")
    return removed, dict(new=new_cnt, cur_L=cur_low, cur_P=cur_high)


# ──────────────────────────────────────────────────────────────────────────────
# Test 7 — one real minibatch end-to-end
# ──────────────────────────────────────────────────────────────────────────────
def test_one_minibatch(C_mat, c_ids, qids, qid_to_text, corpus_file, qrels,
                       base_model, config, B_doc_frac, n_corpus, T, mc_drop_p,
                       lam, m, mc_bs, batch_size):
    """Build a small global cache, score one real minibatch against it, select
    negatives (positives masked), fresh-encode q+pos+selected, compute a
    contrastive loss tensor. PASS if shapes are right and the loss is finite.
    """
    import torch
    import torch.nn.functional as F
    from transformers import AutoTokenizer, AutoModel
    from utils.helpers import encode_batch, _load_corpus_lookup

    device    = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dtype     = torch.bfloat16 if device.type == 'cuda' else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    model     = AutoModel.from_pretrained(base_model, torch_dtype=dtype).to(device)
    corpus_lookup = _load_corpus_lookup(corpus_file)
    q_max = config['model']['query_max_len']
    p_max = config['model']['passage_max_len']
    temp  = config['model'].get('temperature', 0.02)

    # tiny global cache for the test: top-frequency docs, capped small for speed.
    C_len = (C_mat >= 0).sum(axis=1)
    H_small = min(int(round(B_doc_frac * n_corpus)), 2048)
    H_idx, _ = select_global_H(C_mat, C_len, H_small, n_corpus)
    H_idx = [int(x) for x in H_idx]
    H_pos = {d: i for i, d in enumerate(H_idx)}

    # EncodeActiveDocs: T MC passes over H (dropout active) -> Z_H (|H|,T,dim)
    for mod in model.modules():
        if isinstance(mod, torch.nn.Dropout):
            mod.p = mc_drop_p
    model.train()
    H_texts = [corpus_lookup.get(c_ids[i], "") for i in H_idx]
    ZH = encode_batch(model, tokenizer, H_texts * T, device, p_max, mc_bs)
    ZH = ZH.reshape(T, len(H_idx), -1).transpose(1, 0, 2)              # (|H|,T,dim)
    ZH_t = torch.from_numpy(ZH).to(device, torch.float32)
    dim  = ZH.shape[-1]

    # one minibatch of queries; encode T MC passes; score vs cache; mask positives.
    batch_qids = qids[:batch_size]
    q_texts = [qid_to_text[q] for q in batch_qids]
    ZQ = encode_batch(model, tokenizer, q_texts * T, device, q_max, mc_bs)
    ZQ = ZQ.reshape(T, len(batch_qids), -1).transpose(1, 0, 2)
    ZQ_t = torch.from_numpy(ZQ).to(device, torch.float32)

    # positives are stored as docids; map docid -> corpus idx -> H slot, then mask.
    pos_mask = torch.zeros(len(batch_qids), len(H_idx), dtype=torch.bool, device=device)
    c_id_to_idx = {d: i for i, d in enumerate(c_ids)}
    for r, q in enumerate(batch_qids):
        for d in qrels.get(q, ()):
            ci = c_id_to_idx.get(d)
            if ci is not None and ci in H_pos:
                pos_mask[r, H_pos[ci]] = True

    sel, g = _score_against_cache(ZQ_t, ZH_t, lam, m, pos_mask=pos_mask)   # (B,m)
    finite_g = bool(torch.isfinite(g[~torch.isinf(g)]).all())

    # Fresh-encode ONLY q + pos + selected negatives, then a contrastive loss.
    model.eval()                                   # clean encodes for the loss
    neg_corpus_idx = [[H_idx[int(j)] for j in sel[r].tolist()] for r in range(len(batch_qids))]
    pos_docids = []
    for q in batch_qids:
        ps = list(qrels.get(q, ()))
        pos_docids.append(ps[0] if ps else None)
    keep = [r for r, p in enumerate(pos_docids) if p is not None and p in c_id_to_idx]
    if not keep:
        del model
        return False, dict(reason="no batch query had a resolvable positive")

    q_keep   = [q_texts[r] for r in keep]
    pos_keep = [corpus_lookup.get(pos_docids[r], "") for r in keep]
    neg_keep = [corpus_lookup.get(c_ids[neg_corpus_idx[r][0]], "") for r in keep]

    Zq  = torch.from_numpy(encode_batch(model, tokenizer, q_keep,   device, q_max, mc_bs)).to(device, torch.float32)
    Zp  = torch.from_numpy(encode_batch(model, tokenizer, pos_keep, device, p_max, mc_bs)).to(device, torch.float32)
    Zn  = torch.from_numpy(encode_batch(model, tokenizer, neg_keep, device, p_max, mc_bs)).to(device, torch.float32)

    # InfoNCE with in-batch negatives + the mined hard negative (m=1 used for loss).
    docs   = torch.cat([Zp, Zn], dim=0)                       # (2K, dim)
    logits = (Zq @ docs.T) / temp                             # (K, 2K)
    target = torch.arange(len(keep), device=device)           # positive at row r = col r
    loss   = F.cross_entropy(logits, target)
    loss_finite = bool(torch.isfinite(loss).item())

    del model
    if device.type == 'cuda':
        torch.cuda.empty_cache()

    ok = (sel.shape == (len(batch_qids), m)) and finite_g and loss_finite
    print(f"\n[T7] One real minibatch: |H|={len(H_idx):,}, B={len(batch_qids)}, "
          f"sel shape={tuple(sel.shape)}, kept={len(keep)}, "
          f"loss={loss.item():.4f} finite={loss_finite}  => {'PASS' if ok else 'FAIL'}")
    return ok, dict(H=len(H_idx), B=len(batch_qids), sel_shape=tuple(sel.shape),
                    loss=float(loss.item()))


# ──────────────────────────────────────────────────────────────────────────────
# Test 8 — mini runtime (mining-only) vs current GRASS estimate
# ──────────────────────────────────────────────────────────────────────────────
def test_mini_runtime(q_embs, H_size, dim, T, lam, m, n_mine, batch_size,
                      P, encode_docs_per_s, n_corpus_q):
    """Time mining-only (score vs cache + select) for n_mine queries with the new
    architecture. Compare to a current-GRASS estimate, whose per-query cost is
    dominated by fresh-encoding ~P candidates (B*P doc encodes per batch).

    PASS if the projected speedup vs current GRASS is large (>= 5x) from epoch 1.
    """
    import torch
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dtype  = torch.float16 if device.type == 'cuda' else torch.float32

    ZH = torch.nn.functional.normalize(
        torch.randn(H_size, T, dim, device=device, dtype=dtype), dim=-1)

    n_mine = min(n_mine, q_embs.shape[0])
    if device.type == 'cuda':
        torch.cuda.synchronize()
    t0 = time.time()
    for s in range(0, n_mine, batch_size):
        idx  = np.arange(s, min(s + batch_size, n_mine))
        base = torch.from_numpy(q_embs[idx]).to(device, dtype)
        ZQ   = torch.nn.functional.normalize(
            base[:, None, :].repeat(1, T, 1) + 0.01 * torch.randn(len(idx), T, dim, device=device, dtype=dtype),
            dim=-1)
        _score_against_cache(ZQ, ZH, lam, m)
    if device.type == 'cuda':
        torch.cuda.synchronize()
    new_dt = time.time() - t0
    new_qps = n_mine / new_dt if new_dt > 0 else float('inf')

    # current GRASS mining cost estimate: P fresh doc encodes per query dominate.
    # (encode_docs_per_s measured in T2; fall back to a conservative default.)
    dps = encode_docs_per_s if encode_docs_per_s else 1500.0
    cur_qps = dps / P                                    # queries/s if P encodes each
    speedup = new_qps / cur_qps if cur_qps > 0 else float('inf')

    new_epoch_min = n_corpus_q / new_qps / 60
    cur_epoch_min = n_corpus_q / cur_qps / 60
    ok = speedup >= 5.0
    del ZH
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    print(f"\n[T8] Mini runtime (mining-only, {n_mine:,} queries):")
    print(f"      NEW negcache : {new_dt:.2f}s  = {new_qps:,.0f} q/s  "
          f"(full-epoch mining ~= {new_epoch_min:.2f} min)")
    print(f"      GRASS est.   : ~{cur_qps:,.0f} q/s  "
          f"(P={P} fresh encodes/q @ {dps:,.0f} docs/s; full-epoch ~= {cur_epoch_min:.1f} min)")
    print(f"      => speedup ~= {speedup:,.1f}x  {'PASS' if ok else 'FAIL'} (>=5x)")
    return ok, dict(new_qps=new_qps, cur_qps=cur_qps, speedup=speedup,
                    new_epoch_min=new_epoch_min, cur_epoch_min=cur_epoch_min)


# ──────────────────────────────────────────────────────────────────────────────
# Driver
# ──────────────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(description="Negative-Cache Fast-GRASS feasibility (no training)")
    ap.add_argument('--recipe', default='grass')
    ap.add_argument('--budget_fracs', type=float, nargs='+', default=[0.05, 0.10, 0.20],
                    help='B_doc fractions of corpus for the cache-budget sweep')
    ap.add_argument('--b_doc_frac', type=float, default=0.10,
                    help='operating cache fraction used by T2/T3/T5/T7/T8')
    ap.add_argument('--P0', type=int, default=200, help='top stale hits per query (freq basis)')
    ap.add_argument('--P', type=int, default=None, help='current-GRASS pool (default cfg P)')
    ap.add_argument('--L', type=int, default=None, help='current-GRASS shortlist (default cfg L)')
    ap.add_argument('--T', type=int, default=None, help='MC passes (default cfg T)')
    ap.add_argument('--max_cache_gb', type=float, default=10.0,
                    help='T1 budget: max Z_H size (bf16 GB)')
    ap.add_argument('--max_epoch_scoring_min', type=float, default=15.0,
                    help='T3 budget: max full-epoch scoring time (min)')
    ap.add_argument('--batch_sizes', type=int, nargs='+', default=[64, 128])
    ap.add_argument('--mc_batch_size', type=int, default=None,
                    help='encode batch size (default cfg mc_batch_size)')
    ap.add_argument('--encode_probe', type=int, default=512,
                    help='T2 real doc-encode probe size')
    ap.add_argument('--skip_encode_test', action='store_true',
                    help='T2: skip real doc encoding, use analytic estimate only')
    ap.add_argument('--mine_queries', type=int, default=3000, help='T8 mining sample')
    ap.add_argument('--faiss_cpu', action='store_true',
                    help='force CPU faiss search (default GPU torch matmul)')
    ap.add_argument('--max_queries', type=int, default=0, help='subsample for a dry run')
    ap.add_argument('--minibatch_test', action='store_true',
                    help='also run the real one-minibatch end-to-end test (needs GPU)')
    ap.add_argument('--no_reuse_cache', action='store_true')
    ap.add_argument('--debug', action='store_true')
    args = ap.parse_args()

    config = load_config()
    cfg    = config['training'][args.recipe]
    set_seed(config.get('seed', 42))

    base_model = cfg['base_model']
    P0   = args.P0
    P    = args.P if args.P is not None else cfg.get('P', 200)
    L    = args.L if args.L is not None else cfg.get('L', 25)
    T    = args.T if args.T is not None else cfg.get('T', 3)
    m    = cfg.get('m', 1)
    lam  = cfg.get('lambda_val', 1.0)
    q_max = config['model']['query_max_len']
    p_max = config['model']['passage_max_len']
    mc_bs = args.mc_batch_size if args.mc_batch_size is not None else cfg.get('mc_batch_size', 512)
    mc_drop_p = cfg.get('mc_dropout_p', 0.3)
    batch_size = cfg.get('batch_size', 64)

    # Read-only: inputs must already exist (no run_setup()).
    proc        = get_path("processed")
    ance_cfg    = config['training'].get('ance', {})
    corpus_file = proc / ance_cfg.get('corpus_file', 'reasonir_corpus.jsonl')
    qrels_file  = proc / ance_cfg.get('train_qrels_file', 'train_qrels.txt')
    mix_dir     = proc / "training_mixture"
    workdir     = get_path("temp_grass")
    stale_pkl   = workdir / "stale_index" / "corpus.pkl"

    missing = [str(p) for p in (corpus_file, qrels_file, stale_pkl) if not p.exists()]
    if not mix_dir.is_dir() or not any(mix_dir.glob("*.jsonl")):
        missing.append(f"{mix_dir}/*.jsonl")
    if missing:
        sys.exit("[feas] required inputs missing (read-only feasibility; build them "
                 "with the training pipeline first):\n  - " + "\n  - ".join(missing))

    print(f"[feas] loading stale index: {stale_pkl}", flush=True)
    stale_idx, stale_embs, c_ids = build_faiss_index(stale_pkl)
    n_corpus = len(c_ids)
    dim      = stale_idx.d
    qrels    = _load_qrels(qrels_file)

    qids, qid_to_text = load_train_queries(debug=args.debug)
    if args.max_queries and len(qids) > args.max_queries:
        qids = qids[:args.max_queries]
        qid_to_text = {q: qid_to_text[q] for q in qids}
    Nq = len(qids)
    print(f"[feas] {Nq:,} unique train queries | corpus {n_corpus:,} | dim {dim}", flush=True)

    # Reuse the shared query-embedding cache (same encoder + queries as two-set).
    cache_path = workdir / "twoset_feasibility" / "query_embs.pkl"
    q_embs = encode_queries(qids, qid_to_text, base_model, q_max, mc_bs,
                            cache_path, reuse_cache=not args.no_reuse_cache)

    # Candidate matrix (top-P0, positives removed) → frequency basis for global H.
    C_mat, C_len, pos_idx_per_q = build_candidate_matrix(
        q_embs, stale_idx, stale_embs, c_ids, qrels, qids, P0,
        force_faiss_cpu=args.faiss_cpu)

    print("\n" + "=" * 74)
    print("  NEGATIVE-CACHE FAST-GRASS FEASIBILITY")
    print(f"  corpus={n_corpus:,}  queries={Nq:,}  dim={dim}  T={T}  m={m}  "
          f"lambda={lam}  P0={P0}")
    print("=" * 74)

    checks = []

    # T1 — cache budget (gated on the operating fraction)
    ok1, _ = test_cache_budget(args.budget_fracs, args.b_doc_frac, n_corpus, dim, T,
                               args.max_cache_gb)
    checks.append((f"T1 operating cache ({args.b_doc_frac:.0%}) fits budget", ok1))

    # operating cache for the rest
    B_doc = int(round(args.b_doc_frac * n_corpus))
    H_idx, freq = select_global_H(C_mat, C_len, B_doc, n_corpus)
    H_set = set(int(x) for x in H_idx)
    ov = cache_overlap_stats(C_mat, C_len, H_set, m)
    print(f"\n[diag] operating global cache: frac={args.b_doc_frac:.0%} |H|={B_doc:,} "
          f"(freq-ranked). Per-query |C_q ∩ H|: min={ov['min']} med={ov['median']:.0f} "
          f"mean={ov['mean']:.1f} | >=m: {ov['ge_m_frac']:.1%} | zero: {ov['zero_frac']:.1%}")

    # T2 — encode speed (real MCDP build vs ANCE clean corpus pass)
    ok2, info2 = test_encode_speed(B_doc, n_corpus, T, base_model, corpus_file,
                                   c_ids, H_idx, p_max, mc_bs, mc_drop_p,
                                   args.encode_probe, skip=args.skip_encode_test)
    checks.append(("T2 cache refresh <= 0.5x ANCE", ok2))
    enc_dps = info2.get('clean_dps')      # clean encode throughput → GRASS estimate

    # T3 — scoring throughput
    ok3, _ = test_scoring_throughput(q_embs, B_doc, dim, T, args.batch_sizes, lam, m,
                                     n_corpus, args.max_epoch_scoring_min)
    checks.append(("T3 scoring throughput within budget", ok3))

    # T4 — sampler correctness
    ok4, _ = test_sampler_correctness(config.get('seed', 42), m=max(m, 2), lam=lam, T=max(T, 2))
    checks.append(("T4 sampler top-m by g + masking", ok4))

    # T5 — positive masking on the real global cache
    ok5, _ = test_positive_masking(H_idx, pos_idx_per_q, qids, m)
    checks.append(("T5 contamination == 0 after masking", ok5))

    # T6 — fresh-encode count
    ok6, _ = test_fresh_encode_count(batch_size, m, P, L)
    checks.append(("T6 P/L fresh-rerank eliminated", ok6))

    # T7 — one real minibatch (optional, needs GPU)
    if args.minibatch_test:
        ok7, _ = test_one_minibatch(C_mat, c_ids, qids, qid_to_text, corpus_file,
                                    qrels, base_model, config, args.b_doc_frac,
                                    n_corpus, T, mc_drop_p, lam, m, mc_bs, batch_size)
        checks.append(("T7 one-minibatch finite loss", ok7))

    # T8 — mini runtime
    ok8, _ = test_mini_runtime(q_embs, B_doc, dim, T, lam, m, args.mine_queries,
                               batch_size, P, enc_dps, Nq)
    checks.append(("T8 mining speedup >= 5x", ok8))

    # ---- Verdict ----
    print("\n" + "=" * 74)
    print("  VERDICT")
    print("=" * 74)
    passed = 0
    for name, ok in checks:
        print(f"  {'✅ PASS' if ok else '❌ FAIL'}  {name}")
        passed += int(ok)
    print("=" * 74)
    print(f"  {passed}/{len(checks)} checks passed")
    go = passed == len(checks)
    print(f"  {'🟢 NEGATIVE-CACHE ARCHITECTURE LOOKS WORTH RUNNING' if go else '🔴 RECONSIDER — see failed checks above'}")
    print("=" * 74)
    sys.exit(0 if go else 1)


if __name__ == "__main__":
    main()
