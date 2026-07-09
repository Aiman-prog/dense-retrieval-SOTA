"""
Async Fast-GRASS — Phase 0 timing calibration: MINER side.

Measures ``t_mine_round`` = wall time for the miner to produce ONE full Fast-GRASS
mined round over the full training mixture with FROZEN checkpoint / base-model
weights (async_fast_grass_architecture.md, "Timing Calibration Before Async
Training"). This is miner-only: bounded global cache ``H``, MCDP lazy top-``L``,
T-pass query/document dropout scoring, known-positive masking, and selection.
NO training, NO optimizer, NO eval.

It reuses the Fast-GRASS mining path unchanged: ``run_fast_grass._mine_batch`` (the
same EMA / teacher-free-MCDP dispatcher the trainer uses) over a
``NegativeCache`` initialized uniformly from the existing stale corpus pickle. It
does NOT rebuild a full-corpus ANN index and does NOT do per-query stale FAISS
top-P; the stale pickle is only the cache-init source (must already exist).

End-of-round cache maintenance is timed once (one full cycle) and reported both
standalone and amortized per ``cache_update_interval``, since async v0 runs one
expensive maintenance per mined round.

Reports: ``t_mine_round`` (extrapolated to the full mixture), ``queries_per_second``,
MCDP doc/query encoder calls, unique top-L docs per batch, and cache maintenance
time; writes a JSON record under ``analysis/async_fast_grass_timing/``.

If a trainer-timing ``seconds_per_train_step`` is available (via
``--seconds_per_train_step``, ``--train_timing_json``, or the newest
``train_timing_*.json`` in the output dir), it also computes the cadence:

    async_mine_every_steps = ceil((t_mine_round / seconds_per_train_step) * safety_margin)

Modes:
  real (default) : GPU cluster timing (needs the stale pickle + processed mixture).
  --synthetic    : CPU-only smoke on a tiny mock model + random cache to verify the
                   harness runs (NO representative numbers).

Usage:
  python scripts/fast_grass_mine_timing.py --synthetic
  python scripts/fast_grass_mine_timing.py --B_doc 32000 --L 64 --T 3 --max_queries 128
  python scripts/fast_grass_mine_timing.py --B_doc 32000 --L 64 --T 3 \
      --seconds_per_train_step 0.42
"""
import argparse
import gc
import glob
import json
import math
import pickle
import sys
import time
from collections import deque
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root / 'src'))
sys.path.insert(0, str(project_root / 'scripts'))

from utils.helpers import (
    get_training_context, load_config, get_path,
    _load_corpus_lookup, _load_qrels, set_seed,
)
from utils.negative_cache import NegativeCache, linear_decay
import run_fast_grass

OUT_DIR = project_root / 'analysis' / 'async_fast_grass_timing'


def _sync(device):
    if device.type == 'cuda':
        torch.cuda.synchronize()


def _resolve_train_step(args):
    """seconds_per_train_step from CLI, an explicit JSON, or the newest train JSON."""
    if args.seconds_per_train_step is not None:
        return float(args.seconds_per_train_step), 'cli'
    path = args.train_timing_json
    if path is None:
        cands = sorted(glob.glob(str(OUT_DIR / 'train_timing_*.json')))
        path = cands[-1] if cands else None
    if path and Path(path).exists():
        rec = json.loads(Path(path).read_text())
        spts = rec.get('seconds_per_train_step')
        if spts:
            return float(spts), str(path)
    return None, None


def _async_cadence(t_mine_round, seconds_per_train_step, safety_margin):
    if not seconds_per_train_step or seconds_per_train_step <= 0:
        return None
    return int(math.ceil((t_mine_round / seconds_per_train_step) * safety_margin))


def _write_json(record, tag):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    path = OUT_DIR / f"mine_timing_{tag}_{ts}.json"
    path.write_text(json.dumps(record, indent=2))
    print(f"[mine-timing] wrote {path}", flush=True)
    return path


# ---- real GPU timing -------------------------------------------------------

def run_real(args):
    config = load_config()
    ctx = get_training_context('fast_grass')
    set_seed(config.get('seed', 42))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type != 'cuda':
        print("[mine-timing] WARNING: no CUDA — real-mode numbers are NOT "
              "representative. Use --synthetic for a CPU correctness smoke.",
              flush=True)

    from data.preprocessor import run_setup
    corpus_file, _query_file, qrels_file = run_setup()

    # stale pickle is the ONLY cache-init source. Do NOT rebuild the full-corpus
    # ANN index during timing — require it to already exist and fail clearly.
    workdir = get_path("temp_grass")
    stale_pkl = workdir / "stale_index" / "corpus.pkl"
    if not stale_pkl.exists():
        print(f"[mine-timing] ERROR: stale index not found at {stale_pkl}.\n"
              "  Phase-0 timing does not rebuild the full corpus. Build it once via "
              "run_fast_grass.py / the training launcher, then re-run timing.",
              flush=True)
        return 2
    print(f"[mine-timing] stale pickle (cache-init source): {stale_pkl}", flush=True)
    # Load the stale embeddings pickle DIRECTLY. Do NOT call build_faiss_index — it
    # builds an in-memory FAISS IndexFlatIP over the whole corpus (a full-corpus ANN
    # build) that Fast-GRASS never queries, which the timing plan forbids.
    with open(stale_pkl, 'rb') as f:
        _c_data = pickle.load(f)
    stale_embs = _c_data[0]
    c_ids = [str(x) for x in _c_data[1]]
    del _c_data
    corpus_lookup = _load_corpus_lookup(corpus_file)
    qrels_dict = _load_qrels(qrels_file)

    train_items = run_fast_grass._load_train_items()
    qid_to_text = {it['query_id']: it['query'] for it in train_items}
    all_qids = list(qid_to_text.keys())
    total_queries = len(all_qids)

    batch_size = args.batch_size or config['training']['fast_grass'].get('batch_size', 64)
    steps_per_epoch = max(total_queries // batch_size, 1)
    ns = SimpleNamespace(B_doc=args.B_doc, lambda_val=None, ema_alpha=None,
                         uncertainty=args.uncertainty, T=args.T,
                         mc_dropout_p=args.mc_dropout_p, L=args.L,
                         selection_mode=None, m=args.m, num_epochs=None,
                         no_registry=args.no_registry)
    fg_cfg = run_fast_grass._build_fast_grass_cfg(config, ns, steps_per_epoch)
    uncertainty = fg_cfg['uncertainty']

    # --- frozen model(s) ---
    base_model = ctx['base_model']
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    dtype = torch.bfloat16 if device.type == 'cuda' else torch.float32
    student = AutoModel.from_pretrained(base_model, torch_dtype=dtype).to(device).eval()
    for p in student.parameters():
        p.requires_grad_(False)
    if uncertainty == 'ema':
        teacher = AutoModel.from_pretrained(base_model, torch_dtype=dtype).to(device).eval()
        for p in teacher.parameters():
            p.requires_grad_(False)
    else:
        teacher = None
        # MCDP: set the dropout rate on every Dropout module (mirror the trainer)
        mc_p = fg_cfg.get('mc_dropout_p', 0.3)
        n_layers = 0
        for mod in student.modules():
            if isinstance(mod, torch.nn.Dropout):
                mod.p = mc_p
                n_layers += 1
        print(f"[mine-timing] MCDP dropout p={mc_p} on {n_layers} layers", flush=True)

    cache = NegativeCache.init_uniform(stale_embs, c_ids, fg_cfg, device)
    print(f"[mine-timing] cache H | B_doc={cache.B_doc} | "
          f"Z_H={cache.memory_bytes()/1e9:.2f} GB", flush=True)
    del stale_embs
    gc.collect()

    qids = all_qids[:args.max_queries] if args.max_queries else all_qids
    # ceil so the final partial batch is mined too — a full round must cover every
    # query in the mixture, not floor-drop the tail.
    n_batches = max(int(math.ceil(len(qids) / batch_size)), 1)

    Lc = min(int(fg_cfg['L']), cache.B_doc)
    T = int(fg_cfg.get('T', 3))
    print(f"[mine-timing] uncertainty={uncertainty} | B_doc={cache.B_doc} | "
          f"L={fg_cfg['L']} (used {Lc}) | T={T} | batch_size={batch_size} | "
          f"queries={len(qids)}/{total_queries} | batches={n_batches}", flush=True)
    if uncertainty == 'mcdp':
        print(f"[mine-timing] MCDP worst-case doc encodes/step (pre-dedup) = "
              f"batch_size*L*T = {batch_size * Lc * T}", flush=True)

    reservoir_size = fg_cfg.get('recent_query_reservoir_size', 128)
    res_batches = max(reservoir_size // batch_size + 1, 1)
    reservoir = deque(maxlen=res_batches)

    # --- warmup: one batch (warms cudnn/allocator) ---
    warm_qids = qids[:batch_size]
    if warm_qids:
        run_fast_grass._mine_batch(cache, student, teacher, tokenizer, warm_qids,
                                   qid_to_text, corpus_lookup, qrels_dict, fg_cfg,
                                   device)
    _sync(device)
    # Warmup mining called record_selection, which set cache.selected_indicator for
    # the warmup slots. Reset it (and the cumulative score counter) so timed mining
    # and end-of-round maintenance start from clean, uncontaminated utility state.
    cache.selected_indicator.zero_()
    cache.cache_score_pairs = 0

    # optional: measure the JSONL round-write cost (fix 5A). Off by default. When on,
    # each mined batch's records are written to a temp round.jsonl and the write time
    # is folded into t_mine_round, so it reflects "miner produces a full mined round".
    jsonl_f = None
    jsonl_write_time = 0.0
    work_dir = None
    if args.write_jsonl_timing:
        work_dir = OUT_DIR / f"work_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        work_dir.mkdir(parents=True, exist_ok=True)
        jsonl_f = open(work_dir / "round.jsonl", 'w')
        qid_to_pos = {it['query_id']: it['pos_docid'] for it in train_items}

    # --- timed mining over the (subset of the) mixture ---
    # per_batch_time captures ONLY the _mine_batch compute (synced). JSONL write and
    # bookkeeping are excluded from it and tracked separately.
    per_batch_time, per_batch_q = [], []
    tot_doc_calls = tot_q_calls = tot_est_max = 0
    unique_docs_per_batch = []
    queries_processed = 0

    for b in range(n_batches):
        batch_qids = qids[b * batch_size:(b + 1) * batch_size]
        batch_qids = list(dict.fromkeys(batch_qids))
        if not batch_qids:
            continue
        _sync(device)
        t0 = time.perf_counter()
        mined, slots, q_student, q_teacher, mstats = run_fast_grass._mine_batch(
            cache, student, teacher, tokenizer, batch_qids, qid_to_text,
            corpus_lookup, qrels_dict, fg_cfg, device)
        _sync(device)
        per_batch_time.append(time.perf_counter() - t0)

        per_batch_q.append(len(batch_qids))
        queries_processed += len(batch_qids)
        tot_doc_calls += mstats.get('mcdp_doc_encoder_calls', 0)
        tot_q_calls += mstats.get('mcdp_query_encoder_calls', 0)
        tot_est_max += mstats.get('estimated_max_mcdp_doc_encodes_per_step', 0)
        if 'mcdp_unique_docs' in mstats:
            unique_docs_per_batch.append(mstats['mcdp_unique_docs'])

        if jsonl_f is not None:
            wt0 = time.perf_counter()
            for qid in batch_qids:
                jsonl_f.write(json.dumps({
                    "query_id": qid,
                    "query": qid_to_text[qid],
                    "positive_docid": qid_to_pos.get(qid),
                    "neg_docids": mined.get(qid, []),
                }, ensure_ascii=False) + "\n")
            jsonl_write_time += time.perf_counter() - wt0

        reservoir.append((q_student.detach(),
                          q_teacher.detach() if q_teacher is not None else None,
                          list(batch_qids)))
    if jsonl_f is not None:
        jsonl_f.close()

    mining_compute = float(np.sum(per_batch_time))
    qps = queries_processed / mining_compute if mining_compute > 0 else 0.0

    # --- async round maintenance budget (fix 4) ---
    # round_training_span_steps ~= async_mine_every_steps = the number of trainer
    # steps that will consume this mined round. It sizes the ROUND maintenance budget
    #   maintenance_budget_round = round(rho * B_doc * round_span / steps_per_epoch)
    # (async_fast_grass_architecture.md), NOT the sequential cache_update_interval
    # budget. When not given, bootstrap it from the mining-compute-only cadence (the
    # maintenance term is small and not yet timed at this point).
    spts, spts_src = _resolve_train_step(args)
    mine_full = (mining_compute / queries_processed) * total_queries \
        if queries_processed else mining_compute
    if args.round_training_span_steps is not None:
        round_span = max(int(args.round_training_span_steps), 1)
        round_span_src = 'cli'
    elif spts:
        round_span = max(int(math.ceil(mine_full / spts * args.safety_margin)), 1)
        round_span_src = 'bootstrap_mining_only_cadence'
    else:
        round_span = steps_per_epoch
        round_span_src = 'fallback_steps_per_epoch'

    # cfg copy so NegativeCache's existing budget formula produces the ROUND budget:
    # cache_update_interval := round_span makes _interval_budget == maintenance_budget_round.
    maint_cfg = dict(fg_cfg)
    maint_cfg['cache_update_interval'] = round_span
    mid_step = fg_cfg['total_steps'] // 2
    progress = mid_step / max(fg_cfg['total_steps'], 1)
    rho_mid = linear_decay(fg_cfg['rho_start'], fg_cfg['rho_end'], progress)
    maintenance_budget_round = round(
        rho_mid * cache.B_doc * round_span / max(steps_per_epoch, 1))

    # --- end-of-round maintenance (one full cycle, timed) ---
    qs = torch.cat([e[0] for e in reservoir], dim=0)
    res_qids = [q for e in reservoir for q in e[2]]
    res_n = min(reservoir_size, len(res_qids))
    if teacher is not None:
        qt = torch.cat([e[1] for e in reservoir], dim=0)[-res_n:]
    else:
        qt = None
    reservoir_dict = {'q_student': qs[-res_n:], 'q_teacher': qt,
                      'qids': res_qids[-res_n:]}
    # make some slots eligible so the timed cycle does representative work
    cache.last_refreshed_step[:] = 0
    _sync(device)
    t0 = time.perf_counter()
    maint = cache.maintain(student, teacher, tokenizer, corpus_lookup, c_ids,
                           reservoir_dict, step=mid_step, cfg=maint_cfg, device=device,
                           qrels_dict=qrels_dict)
    _sync(device)
    maint_time = time.perf_counter() - t0

    # --- extrapolate to a full mined round over the whole mixture ---
    jsonl_write_full = (jsonl_write_time / queries_processed) * total_queries \
        if (jsonl_f is not None and queries_processed) else 0.0
    # one round = full-mixture mine + one end-of-round maintenance (+ JSONL write when
    # measured). If write is NOT measured, t_mine_round EXCLUDES that cost (flagged).
    t_mine_round = mine_full + maint_time + jsonl_write_full
    async_mine_every = _async_cadence(t_mine_round, spts, args.safety_margin)

    record = {
        'kind': 'mine_timing',
        'mode': 'real',
        'device': str(device),
        'base_model': base_model,
        'uncertainty': uncertainty,
        'B_doc': cache.B_doc,
        'L': int(fg_cfg['L']),
        'L_used': Lc,
        'T': T,
        'm': fg_cfg['m'],
        'lambda_val': fg_cfg['lambda_val'],
        'selection_mode': fg_cfg['selection_mode'],
        'batch_size': batch_size,
        'no_registry': bool(args.no_registry),
        'total_queries_mixture': total_queries,
        'queries_processed': queries_processed,
        'batches_timed': len(per_batch_time),
        'mining_compute_only_s': mining_compute,
        'queries_per_second': qps,
        'mean_batch_mining_s': float(np.mean(per_batch_time)) if per_batch_time else None,
        'median_batch_mining_s': float(np.median(per_batch_time)) if per_batch_time else None,
        'mcdp_doc_encoder_calls_total': int(tot_doc_calls),
        'mcdp_query_encoder_calls_total': int(tot_q_calls),
        'mcdp_doc_encoder_calls_per_batch_mean': (
            float(tot_doc_calls / len(per_batch_time)) if per_batch_time else None),
        'estimated_max_mcdp_doc_encodes_total': int(tot_est_max),
        'mcdp_unique_topL_docs_per_batch_mean': (
            float(np.mean(unique_docs_per_batch)) if unique_docs_per_batch else None),
        'mcdp_unique_topL_docs_per_batch_max': (
            int(np.max(unique_docs_per_batch)) if unique_docs_per_batch else None),
        'cache_maintenance_time_s': maint_time,
        'cache_maintenance_amortized_per_step_s': maint_time / round_span if round_span else None,
        'round_training_span_steps': round_span,
        'round_training_span_steps_source': round_span_src,
        'maintenance_budget_round': int(maintenance_budget_round),
        'cache_update_interval_config': fg_cfg['cache_update_interval'],
        'maintenance_num_refresh': maint['num_refresh'],
        'maintenance_num_replace': maint['num_replace'],
        'maintenance_num_recertified': maint['num_recertified_candidates'],
        'jsonl_write_timing_included': bool(jsonl_f is not None),
        'jsonl_write_measured_s': jsonl_write_time if jsonl_f is not None else None,
        'jsonl_write_full_extrapolated_s': jsonl_write_full if jsonl_f is not None else None,
        'jsonl_write_excluded_warning': (
            None if jsonl_f is not None else
            "t_mine_round EXCLUDES JSONL round-write cost; pass --write_jsonl_timing "
            "to include it"),
        'jsonl_work_dir': str(work_dir) if work_dir is not None else None,
        'mining_wall_full_mixture_extrapolated_s': mine_full,
        't_mine_round': t_mine_round,
        'seconds_per_train_step': spts,
        'seconds_per_train_step_source': spts_src,
        'safety_margin': args.safety_margin,
        'async_mine_every_steps': async_mine_every,
    }

    _print_report(record)
    tag = f"bdoc{cache.B_doc}_L{fg_cfg['L']}_T{T}_{uncertainty}"
    _write_json(record, tag)

    del student, teacher, cache
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return 0


def _print_report(r):
    print("\n" + "=" * 66)
    print("  ASYNC FAST-GRASS — MINER TIMING (t_mine_round)")
    print("=" * 66)
    print(f"  uncertainty          : {r['uncertainty']}")
    print(f"  B_doc / L / T        : {r['B_doc']:,} / {r['L']} (used {r['L_used']}) / {r['T']}")
    print(f"  queries processed    : {r['queries_processed']:,} / "
          f"{r['total_queries_mixture']:,} (mixture)")
    print(f"  mining compute (sub) : {r['mining_compute_only_s']:.2f} s "
          f"(mine-only, excl. JSONL write)")
    print(f"  queries_per_second   : {r['queries_per_second']:,.1f}")
    if r['uncertainty'] == 'mcdp':
        print(f"  MCDP doc encodes     : {r['mcdp_doc_encoder_calls_total']:,} total "
              f"({r['mcdp_doc_encoder_calls_per_batch_mean']:,.0f}/batch)")
        print(f"    vs worst-case B*L*T: {r['estimated_max_mcdp_doc_encodes_total']:,} "
              f"(dedup savings)")
        print(f"  unique top-L docs/bt : mean {r['mcdp_unique_topL_docs_per_batch_mean']:.1f} "
              f"| max {r['mcdp_unique_topL_docs_per_batch_max']}")
    print(f"  round span (steps)   : {r['round_training_span_steps']:,} "
          f"(source: {r['round_training_span_steps_source']})")
    print(f"  maint budget / round : {r['maintenance_budget_round']:,} docs")
    print(f"  maintenance (1 cycle): {r['cache_maintenance_time_s']:.2f} s "
          f"(refresh={r['maintenance_num_refresh']} replace={r['maintenance_num_replace']} "
          f"recert={r['maintenance_num_recertified']})")
    print(f"  mining full mixture  : {r['mining_wall_full_mixture_extrapolated_s']:.1f} s "
          f"(extrapolated)")
    if r['jsonl_write_timing_included']:
        print(f"  JSONL write (round)  : {r['jsonl_write_full_extrapolated_s']:.1f} s "
              f"(extrapolated; included in t_mine_round)")
    else:
        print("  JSONL write (round)  : EXCLUDED (pass --write_jsonl_timing to include)")
    print(f"  => t_mine_round      : {r['t_mine_round']:.1f} s "
          f"(full mine + 1 maintenance"
          f"{' + JSONL write' if r['jsonl_write_timing_included'] else ''})")
    print("-" * 66)
    if r['async_mine_every_steps'] is not None:
        print(f"  seconds_per_train_step: {r['seconds_per_train_step']:.4f} s "
              f"(source: {r['seconds_per_train_step_source']})")
        print(f"  safety_margin        : {r['safety_margin']}")
        print(f"  => async_mine_every_steps = ceil(t_mine_round / "
              f"seconds_per_train_step * margin)")
        print(f"     = ceil({r['t_mine_round']:.1f} / {r['seconds_per_train_step']:.4f} "
              f"* {r['safety_margin']}) = {r['async_mine_every_steps']:,} steps")
    else:
        print("  async_mine_every_steps: (need seconds_per_train_step — pass "
              "--seconds_per_train_step or run fast_grass_train_timing.py first)")
    print("=" * 66)


# ---- synthetic CPU smoke ---------------------------------------------------

def run_synthetic(args):
    print("\n" + "=" * 66)
    print("  MINER TIMING — SYNTHETIC (CPU, mock model; NO real numbers)")
    print("=" * 66)
    from fast_grass_train_smoke import DropoutMockModel, MockTokenizer

    device = torch.device('cpu')
    dim = 8
    n_corpus = 60
    c_ids = [f"d{i}" for i in range(n_corpus)]
    corpus_lookup = {d: f"document {d} body text" for d in c_ids}
    embs = np.random.default_rng(1).standard_normal((n_corpus, dim)).astype('float32')

    n_q = 24
    qid_to_text = {f"q{i}": f"query number {i}" for i in range(n_q)}
    all_qids = list(qid_to_text)
    qrels_dict = {q: {c_ids[i % n_corpus]} for i, q in enumerate(all_qids)}

    cfg = dict(
        model_name="mine_timing_smoke", uncertainty='mcdp',
        B_doc=20, m=1, selection_mode='topk', lambda_val=1.0, beta=5.0,
        L=4, T=2, mc_dropout_p=0.5,
        ema_alpha=0.999, rho_start=0.50, rho_end=0.10,
        cache_update_interval=2, max_age_epochs=4, utility_ema_decay=0.95,
        utility_floor=0.01, utility_remember_threshold=0.05, K=3, R_fraction=0.25,
        uniform_candidate_fraction=0.75, replacement_candidate_multiplier=2,
        recent_query_reservoir_size=8, reentry_top_k=5, R_size_factor=0.5,
        cache_init_seed=42,
        learning_rate=1e-4, num_epochs=1, batch_size=4, mc_batch_size=16,
        passage_max_len=128, query_max_len=128,
        steps_per_epoch=6, total_steps=6, max_age_steps=24)

    cache = NegativeCache.init_uniform(embs, c_ids, cfg, device, dim=dim)
    student = DropoutMockModel(hidden=dim, p=cfg['mc_dropout_p'])
    student.eval()
    tok = MockTokenizer()

    batch_size = cfg['batch_size']
    n_batches = max(int(math.ceil(len(all_qids) / batch_size)), 1)
    reservoir = deque(maxlen=3)
    times, doc_calls, uniq = [], 0, []
    q_proc = 0
    # accumulate EVERY batch's mined negatives so the contract checks cover the whole
    # round, not just the last batch's `mined`.
    all_mined = {}
    t0round = time.perf_counter()
    for b in range(n_batches):
        bq = all_qids[b * batch_size:(b + 1) * batch_size]
        if not bq:
            continue
        t0 = time.perf_counter()
        mined, slots, q_stu, q_tea, mstats = run_fast_grass._mine_batch(
            cache, student, None, tok, bq, qid_to_text, corpus_lookup,
            qrels_dict, cfg, device)
        times.append(time.perf_counter() - t0)
        q_proc += len(bq)
        doc_calls += mstats.get('mcdp_doc_encoder_calls', 0)
        uniq.append(mstats.get('mcdp_unique_docs', 0))
        all_mined.update(mined)
        reservoir.append((q_stu.detach(), None, list(bq)))
    mining_wall = time.perf_counter() - t0round

    # Snapshot H at selection time (BEFORE maintenance). Mining never mutates the
    # docid set; only maintenance can replace docs, so this is exactly the pool the
    # negatives were selected from — checking from_H against post-maintenance
    # cache.docids would be contaminated by replacements.
    H_at_selection = set(cache.docids)

    # exercise a maintenance cycle (teacher-free)
    qs = torch.cat([e[0] for e in reservoir], dim=0)
    rq = [q for e in reservoir for q in e[2]]
    cache.last_refreshed_step[:] = 0
    t0 = time.perf_counter()
    maint = cache.maintain(student, None, tok, corpus_lookup, c_ids,
                           {'q_student': qs, 'q_teacher': None, 'qids': rq},
                           step=3, cfg=cfg, device=device, qrels_dict=qrels_dict)
    maint_time = time.perf_counter() - t0

    # contract over ALL mined negatives (every query in the round):
    clean = all(d not in qrels_dict.get(q, set())
                for q, negs in all_mined.items() for d in negs)
    from_H = all(d in H_at_selection
                 for q, negs in all_mined.items() for d in negs)
    all_queries_mined = set(all_mined) == set(all_qids)
    b_doc_ok = len(cache.docids) == cache.B_doc

    t_mine_round = mining_wall + maint_time
    spts = 0.05  # mock train step for the cadence-formula smoke
    cadence = _async_cadence(t_mine_round, spts, args.safety_margin)

    ok = (len(times) > 0 and all(np.isfinite(t) and t >= 0 for t in times)
          and doc_calls > 0 and clean and from_H and all_queries_mined
          and b_doc_ok and cadence is not None and cadence >= 1)

    print(f"  batches mined        : {len(times)} | queries {q_proc}")
    print(f"  MCDP doc encodes     : {doc_calls} | unique/batch mean {np.mean(uniq):.1f}")
    print(f"  mining wall          : {mining_wall*1e3:.2f} ms")
    print(f"  maintenance (1 cycle): {maint_time*1e3:.2f} ms "
          f"(refresh={maint['num_refresh']} replace={maint['num_replace']})")
    print(f"  t_mine_round (mock)  : {t_mine_round*1e3:.2f} ms")
    print(f"  async_mine_every_steps (mock spts={spts}) : {cadence}")
    print(f"  all queries mined    : {all_queries_mined} ({len(all_mined)}/{len(all_qids)})")
    print(f"  clean negatives (all): {clean}")
    print(f"  negatives from H     : {from_H} (snapshot pre-maintenance)")
    print(f"  B_doc invariant      : {b_doc_ok}")
    print("=" * 66)
    print(f"  {'PASS' if ok else 'FAIL'}  miner-timing harness runs end to end")
    print("=" * 66)
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--synthetic', action='store_true',
                    help='CPU mock-model smoke (no real numbers)')
    ap.add_argument('--B_doc', type=int, default=None, help='cache size H (e.g. 32000)')
    ap.add_argument('--L', type=int, default=None, help='MCDP lazy top-L (e.g. 64)')
    ap.add_argument('--T', type=int, default=None, help='MCDP dropout passes (e.g. 3)')
    ap.add_argument('--m', type=int, default=None, help='negatives per query')
    ap.add_argument('--mc_dropout_p', type=float, default=None)
    ap.add_argument('--uncertainty', choices=['mcdp', 'ema'], default='mcdp')
    ap.add_argument('--batch_size', type=int, default=None)
    ap.add_argument('--max_queries', type=int, default=None,
                    help='subset the mixture (tiny sanity run, e.g. 128); '
                         't_mine_round is extrapolated to the full mixture')
    ap.add_argument('--no_registry', action='store_true',
                    help='ablation: disable the retired registry R')
    ap.add_argument('--seconds_per_train_step', type=float, default=None,
                    help='from fast_grass_train_timing.py; enables '
                         'async_mine_every_steps')
    ap.add_argument('--train_timing_json', type=str, default=None,
                    help='explicit train_timing_*.json to read '
                         'seconds_per_train_step from')
    ap.add_argument('--safety_margin', type=float, default=1.2,
                    help='async cadence safety margin (default 1.2; doc 1.1-1.25)')
    ap.add_argument('--round_training_span_steps', type=int, default=None,
                    help='trainer steps that consume one mined round; sizes the async '
                         'maintenance_budget_round. Default: bootstrap from the '
                         'mining-only cadence (needs seconds_per_train_step), else '
                         'steps_per_epoch.')
    ap.add_argument('--write_jsonl_timing', action='store_true',
                    help='write a temp mined round.jsonl (under analysis/.../work_*) '
                         'and fold its write time into t_mine_round (fix 5A). Off by '
                         'default, in which case JSONL write cost is excluded.')
    args = ap.parse_args()
    return run_synthetic(args) if args.synthetic else run_real(args)


if __name__ == "__main__":
    sys.exit(main())
