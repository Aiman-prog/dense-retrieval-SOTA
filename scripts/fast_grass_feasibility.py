"""
Fast-GRASS mining-speed benchmark (read-only).

Answers "how much faster is Fast-GRASS mining than current GRASS?" by running BOTH
mining paths on the SAME query batches and counting document/query encoder work +
wall-time:

  baseline   : run_grass._mine_queries(..., uncertainty='ema')
               stale FAISS top-P -> fresh-encode/rerank P candidates -> uncertainty
  Fast-GRASS : encode queries -> cache.score (Q x Z_H matmul) -> mask -> select
               + amortized cache maintenance

NO training, gradients, checkpoints, or data/config mutation. The final-loss fresh
encode of selected pos+negs is IDENTICAL in both paths and is excluded from the
mining comparison.

Real run (DelftBlue, inside pytorch_2.1.sif): emits baseline-vs-Fast-GRASS rows/step,
wall-time, and the SPEEDUP RATIO. --synthetic runs ONLY the Fast-GRASS path on CPU
(random embeddings) for correctness/throughput sanity and prints NO speedup ratio
(the baseline needs the full FAISS index + corpus + rerank model).

Supersedes the earlier donor grass_negcache_feasibility.py, which imported the
never-committed grass_twoset_feasibility.py. That donor was deleted in the Aug 2026
cleanup; see D2 in CONSOLIDATION_STATUS.md for where its code ended up.
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root / 'src'))
sys.path.insert(0, str(project_root / 'scripts'))

from utils.helpers import (load_config, get_training_context, get_path,
                           build_faiss_index, encode_to_pickle,
                           _load_corpus_lookup, _load_qrels, encode_batch_tensor,
                           set_seed)
from utils.negative_cache import NegativeCache


# ---- encoder row-counting hook --------------------------------------------

class RowCounter:
    """forward_pre_hook that sums input_ids.shape[0] across encoder calls."""
    def __init__(self):
        self.rows = 0
        self._handles = []

    def reset(self):
        self.rows = 0

    def _hook(self, module, args, kwargs):
        ids = kwargs.get('input_ids') if kwargs else None
        if ids is None and args:
            ids = args[0]
        if ids is not None and hasattr(ids, 'shape'):
            self.rows += int(ids.shape[0])
        return None

    def attach(self, *modules):
        for m in modules:
            if m is not None:
                self._handles.append(
                    m.register_forward_pre_hook(self._hook, with_kwargs=True))
        return self

    def detach(self):
        for h in self._handles:
            h.remove()
        self._handles = []


def _sync(device):
    if device.type == 'cuda':
        torch.cuda.synchronize()


def _fg_cfg(config, args):
    """Build the runtime fast_grass cfg with derived step counts."""
    cfg = dict(config['training']['fast_grass'])
    # This benchmark scores the EMA-style Q×Z_H matmul (cache.score), so it needs a
    # teacher-backed cache regardless of the config default (now mcdp).
    cfg['uncertainty'] = 'ema'
    if args.B_doc is not None:
        cfg['B_doc'] = args.B_doc
    cfg['lambda_val'] = float(cfg['lambda_val'])
    cfg['passage_max_len'] = config['model']['passage_max_len']
    cfg['query_max_len'] = config['model']['query_max_len']
    return cfg


def _encode_q(student, teacher, tokenizer, texts, cfg, device, dtype):
    q_max = cfg['query_max_len']
    bs = cfg.get('mc_batch_size', 256)
    qs = encode_batch_tensor(student, tokenizer, texts, device, q_max, bs,
                             requires_grad=False).to(dtype)
    qt = encode_batch_tensor(teacher, tokenizer, texts, device, q_max, bs,
                             requires_grad=False).to(dtype)
    return qs, qt


# ---- synthetic mode (Fast-GRASS path only, no ratio) ----------------------

def run_synthetic(args):
    print("\n" + "=" * 70)
    print("  FAST-GRASS FEASIBILITY — SYNTHETIC (Fast-GRASS path only)")
    print("  NOTE: no speedup ratio in synthetic mode (baseline needs real")
    print("        FAISS index + corpus + rerank model). Correctness/throughput")
    print("        sanity only; the speedup number comes from the GPU run.")
    print("=" * 70)
    device = torch.device('cpu')
    dim = 64
    B_doc = args.B_doc or 2000
    n_corpus = max(B_doc * 2, 4000)
    rng = np.random.default_rng(0)
    embs = rng.standard_normal((n_corpus, dim)).astype('float32')
    c_ids = [f"d{i}" for i in range(n_corpus)]

    cfg = dict(load_config()['training']['fast_grass'])
    cfg['uncertainty'] = 'ema'   # benchmarks the EMA-style cache.score matmul
    cfg['B_doc'] = B_doc
    cfg['lambda_val'] = float(cfg['lambda_val'])
    cache = NegativeCache.init_uniform(embs, c_ids, cfg, device, dim=dim,
                                       dtype=torch.float32)

    norm = torch.nn.functional.normalize
    times = []
    for _ in range(args.batches):
        q = norm(torch.from_numpy(rng.standard_normal((64, dim)).astype('float32')), dim=-1)
        t0 = time.perf_counter()
        g, _, _ = cache.score(q, q, cfg['lambda_val'])
        g = cache.mask_positives(g, [f"q{i}" for i in range(64)], {})
        slots, _ = cache.select(g, m=cfg['m'], mode=cfg['selection_mode'],
                                beta=cfg['beta'], L=cfg['L'])
        cache.record_selection(slots)
        times.append(time.perf_counter() - t0)

    finite = torch.isfinite(g).all().item()
    print(f"\n  B_doc={cache.B_doc}  dim={dim}  batches={args.batches}")
    print(f"  Fast-GRASS mining/step : {np.mean(times)*1e3:.2f} ms  "
          f"(queries scored, no fresh candidate encode)")
    print(f"  Z_H memory             : {cache.memory_bytes()/1e6:.1f} MB (fp32)")
    print(f"  finite g               : {finite}")
    print(f"  B_doc invariant        : {len(cache.docids) == cache.B_doc}")
    print("=" * 70)
    return 0 if finite and len(cache.docids) == cache.B_doc else 1


# ---- real mode (baseline vs Fast-GRASS, with ratio) -----------------------

def run_real(args):
    import run_grass  # baseline mining path (verified importable)

    config = load_config()
    set_seed(config.get('seed', 42))
    grass_cfg = dict(config['training']['grass'])
    fg_cfg = _fg_cfg(config, args)
    ctx = get_training_context('grass')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # --- shared setup (mirror run_grass.main) ---
    from data.preprocessor import run_setup
    corpus_file, _query_file, qrels_file = run_setup()
    workdir = get_path("temp_grass")
    stale_pkl = workdir / "stale_index" / "corpus.pkl"
    if not stale_pkl.exists():
        stale_pkl.parent.mkdir(parents=True, exist_ok=True)
        print("[feas] building stale ANN index (one-off)...", flush=True)
        encode_to_pickle(ctx['base_model'], corpus_file, stale_pkl, False, ctx, config)
    print(f"[feas] stale index: {stale_pkl}", flush=True)
    stale_idx, stale_embs, c_ids = build_faiss_index(stale_pkl)
    corpus_lookup = _load_corpus_lookup(corpus_file)
    qrels_dict = _load_qrels(qrels_file)

    # --- training queries (positive_passages schema, same as run_grass) ---
    import json
    mix_dir = get_path("processed") / "training_mixture"
    qid_to_text = {}
    for f_path in sorted(mix_dir.glob("*.jsonl")):
        if f_path.name.startswith('.'):
            continue
        with open(f_path) as f:
            for line in f:
                d = json.loads(line)
                if d.get('positive_passages'):
                    qid_to_text[str(d['query_id'])] = d['query']
    all_qids = list(qid_to_text.keys())
    print(f"[feas] {len(all_qids):,} unique train queries | corpus {len(c_ids):,}",
          flush=True)

    # --- models (one student+teacher shared by both paths) ---
    from transformers import AutoTokenizer, AutoModel
    base_model = ctx['base_model']
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    student = AutoModel.from_pretrained(base_model, torch_dtype=torch.bfloat16).to(device).eval()
    teacher = AutoModel.from_pretrained(base_model, torch_dtype=torch.bfloat16).to(device).eval()
    for p in teacher.parameters():
        p.requires_grad_(False)

    # --- cache ---
    cache = NegativeCache.init_uniform(stale_embs, c_ids, fg_cfg, device)
    cache_dtype = cache.Z_student.dtype
    steps_per_epoch = max(len(all_qids) // grass_cfg.get('batch_size', 64), 1)
    fg_cfg['steps_per_epoch'] = args.steps_per_epoch or steps_per_epoch
    fg_cfg['total_steps'] = fg_cfg['steps_per_epoch'] * fg_cfg.get('num_epochs', 2)
    fg_cfg['max_age_steps'] = fg_cfg['max_age_epochs'] * fg_cfg['steps_per_epoch']

    rng = np.random.default_rng(config.get('seed', 42))
    batch_size = grass_cfg.get('batch_size', 64)

    counter = RowCounter().attach(student, teacher)

    base_rows, base_time = [], []
    fg_rows, fg_time = [], []
    reservoir_qids = []

    print(f"\n[feas] mining {args.batches} batches of {batch_size} "
          f"(B_doc={cache.B_doc}, uncertainty=ema)...", flush=True)
    for b in range(args.batches):
        qids = [all_qids[int(i)] for i in rng.integers(0, len(all_qids), batch_size)]
        reservoir_qids.extend(qids)

        # baseline GRASS mining
        counter.reset(); _sync(device); t0 = time.perf_counter()
        with torch.no_grad():
            run_grass._mine_queries(student, teacher, tokenizer, qids, qid_to_text,
                                    stale_idx, c_ids, corpus_lookup, qrels_dict,
                                    grass_cfg, config, device, uncertainty='ema')
        _sync(device)
        base_time.append(time.perf_counter() - t0); base_rows.append(counter.rows)

        # Fast-GRASS mining
        texts = [qid_to_text[q] for q in qids]
        counter.reset(); _sync(device); t0 = time.perf_counter()
        with torch.no_grad():
            qs, qt = _encode_q(student, teacher, tokenizer, texts, fg_cfg, device, cache_dtype)
            g, _, _ = cache.score(qs, qt, fg_cfg['lambda_val'])
            g = cache.mask_positives(g, qids, qrels_dict, inplace=True)
            slots, _ = cache.select(g, m=fg_cfg['m'], mode=fg_cfg['selection_mode'],
                                    beta=fg_cfg['beta'], L=fg_cfg['L'])
            cache.record_selection(slots)
        _sync(device)
        fg_time.append(time.perf_counter() - t0); fg_rows.append(counter.rows)

    # --- amortized maintenance: one full cycle, timed separately ---
    res_n = min(fg_cfg['recent_query_reservoir_size'], len(reservoir_qids))
    res_qids = reservoir_qids[-res_n:]
    with torch.no_grad():
        rqs, rqt = _encode_q(student, teacher, tokenizer,
                             [qid_to_text[q] for q in res_qids], fg_cfg, device, cache_dtype)
    reservoir = {'q_student': rqs, 'q_teacher': rqt, 'qids': res_qids}
    # exercise maintenance at a representative mid-training step
    mid_step = fg_cfg['total_steps'] // 2
    cache.last_refreshed_step[:] = 0  # make some docs eligible for the timed cycle
    counter.reset(); _sync(device); t0 = time.perf_counter()
    maint = cache.maintain(student, teacher, tokenizer, corpus_lookup, c_ids,
                           reservoir, step=mid_step, cfg=fg_cfg, device=device,
                           qrels_dict=qrels_dict)
    _sync(device)
    maint_time = time.perf_counter() - t0
    maint_rows = counter.rows
    counter.detach()

    interval = fg_cfg['cache_update_interval']
    amort_time = maint_time / interval
    amort_rows = maint_rows / interval

    # --- report ---
    bm_rows, fm_rows = np.mean(base_rows), np.mean(fg_rows)
    bm_time, fm_time = np.mean(base_time), np.mean(fg_time)
    fg_total_time = fm_time + amort_time
    row_speedup = bm_rows / max(fm_rows + amort_rows, 1e-9)
    wall_speedup = bm_time / max(fg_total_time, 1e-9)

    print("\n" + "=" * 70)
    print("  FAST-GRASS MINING-SPEED BENCHMARK")
    print(f"  B_doc={cache.B_doc:,}  batch={batch_size}  batches={args.batches}  "
          f"P={grass_cfg['P']} L={grass_cfg['L']}")
    print("=" * 70)
    print(f"\n  Encoder ROWS per mining step:")
    print(f"    baseline GRASS : {bm_rows:>10,.0f}  (queries + P-pool + L-shortlist)")
    print(f"    Fast-GRASS     : {fm_rows:>10,.0f}  (queries only)")
    print(f"                   + {amort_rows:>10,.1f}  (amortized maintenance / step)")
    print(f"    => row speedup : {row_speedup:>8.1f}x")
    print(f"\n  Mining WALL-TIME per step:")
    print(f"    baseline GRASS : {bm_time*1e3:>9.2f} ms")
    print(f"    Fast-GRASS     : {fm_time*1e3:>9.2f} ms  "
          f"(+ {amort_time*1e3:.2f} ms amortized maintenance)")
    print(f"    => wall speedup: {wall_speedup:>8.1f}x")
    print(f"\n  Maintenance (one full cycle): {maint_time:.2f}s, {maint_rows:,} rows; "
          f"refresh={maint['num_refresh']} replace={maint['num_replace']} "
          f"recert={maint['num_recertified_candidates']} "
          f"turnover={maint['cache_turnover_rate']:.4f}")
    print(f"  Z_H memory: {cache.memory_bytes()/1e9:.2f} GB ({cache_dtype})")
    print("\n  NOTE: final-loss fresh-encode of selected pos+negs is IDENTICAL in")
    print("        both paths and is excluded from this mining comparison.")
    print("=" * 70)
    verdict = "FAST-GRASS MINING IS FASTER" if wall_speedup >= 1.0 else "NO SPEEDUP"
    print(f"  {'PASS' if wall_speedup >= 1.0 else 'FAIL'}  {verdict} "
          f"(wall {wall_speedup:.1f}x, rows {row_speedup:.1f}x)")
    print("=" * 70)
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--synthetic', action='store_true',
                    help='Fast-GRASS path only on CPU (no baseline, no ratio)')
    ap.add_argument('--batches', type=int, default=50)
    ap.add_argument('--B_doc', type=int, default=None, help='override cache size')
    ap.add_argument('--uncertainty', default='ema', choices=['ema'],
                    help='estimator (v0: ema)')
    ap.add_argument('--maintain-every', dest='maintain_every', type=int, default=None,
                    help='reserved; maintenance is amortized over cache_update_interval')
    ap.add_argument('--steps-per-epoch', dest='steps_per_epoch', type=int, default=None,
                    help='override derived steps_per_epoch (budget calc)')
    args = ap.parse_args()
    return run_synthetic(args) if args.synthetic else run_real(args)


if __name__ == "__main__":
    sys.exit(main())
