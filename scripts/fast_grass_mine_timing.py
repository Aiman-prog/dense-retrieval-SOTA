"""
Async Fast-GRASS — Phase 0 timing calibration: MINER side (cached-MCDP).

Measures ``t_mine_round`` = wall time for the miner to produce ONE full mined round
over the training mixture with FROZEN checkpoint / base-model weights
(async_fast_grass_architecture.md, "Timing Calibration"). Miner-only: bounded global
cache ``H``, cached-MCDP scoring, known-positive masking, selection, and PERIODIC
IN-ROUND cache maintenance. NO training, NO optimizer, NO eval.

Cached-MCDP, not lazy top-``L``. ``Z_mc[T, B_doc, D]`` is built once from ``T``
genuine dropout passes; every mining batch does ``T`` fresh stochastic QUERY encodes
and ``T`` matmuls over ALL of ``H``, with **zero document encoder calls**
(``mcdp_doc_encoder_calls_mining == 0``, asserted below). The old lazy top-``L``
path (``run_fast_grass._mine_batch_mcdp``) re-encoded the shortlist every batch and
is now oracle-only — it has no launcher path here, and ``--L`` is gone.

Maintenance follows the doc, not the rejected "one big pass at round end" variant:

    maintenance_budget_interval   = round(rho * B_doc * cache_update_interval
                                          / steps_per_epoch)
    maintenance_interval_mined_queries = cache_update_interval * trainer_batch_size

i.e. ``cache_update_interval`` keeps its trainer-step meaning in the BUDGET, while
the miner's execution TRIGGER converts to mined query examples (100*64 = 6400).
Model time is always ``source_checkpoint_step`` — never a miner-local counter.

The stale corpus pickle supplies only the docid sampling/ordering for ``H``; the MC
states come from real dropout passes. No full-corpus ANN rebuild, no per-query
stale FAISS top-P.

``Z_mc`` initialization is timed separately as ``t_cache_mc_init`` and is NOT folded
into ``t_mine_round`` (it happens once per run, as ``mining_meta_initial``); the
speed estimate charges it as async startup.

Reports ``t_mine_round`` (extrapolated to the full mixture, maintenance intervals
included), ``queries_per_second``, three-way encoder accounting, peak GPU memory,
and per-interval maintenance cost; writes JSON to ``analysis/async_fast_grass_timing/``.

If a trainer-timing ``seconds_per_train_step`` is available (via
``--seconds_per_train_step``, ``--train_timing_json``, or the newest
``train_timing_*.json`` in the output dir), it also computes the cadence:

    async_mine_every_steps = ceil((t_mine_round / seconds_per_train_step) * safety_margin)

NOTE: the gate sample must cross at least TWO maintenance thresholds (ordinarily
``--max_queries >= 12800``) or the per-interval maintenance cost is a single sample
and extrapolation is flagged unreliable.

Async EMA is deferred (checkpoints do not carry the EMA teacher), so this script
times ``cached_mcdp`` only.

Modes:
  real (default) : GPU cluster timing (needs the stale pickle + processed mixture).
  --synthetic    : CPU-only smoke on a tiny mock model + random cache to verify the
                   harness runs (NO representative numbers).

Usage:
  python scripts/fast_grass_mine_timing.py --synthetic
  python scripts/fast_grass_mine_timing.py --B_doc 32000 --T 3 --max_queries 12800
  python scripts/fast_grass_mine_timing.py --B_doc 32000 --T 3 \
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
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root / 'src'))
sys.path.insert(0, str(project_root / 'scripts'))
# the --synthetic path reuses mock fixtures that live with the suites in tests/
sys.path.insert(0, str(project_root / 'tests'))

from utils.helpers import (
    get_training_context, load_config, get_path,
    _load_corpus_lookup, _load_qrels, set_seed,
)
from utils.negative_cache import NegativeCache
import run_fast_grass
from async_fast_grass_cached_mcdp import (
    init_Z_mc, mine_batch_cached_mcdp, maintain_interval_cached_mcdp,
    QueryMCReservoir, MaintenanceDriver, maintenance_interval_mined_queries,
)

OUT_DIR = project_root / 'analysis' / 'async_fast_grass_timing'


def _sync(device):
    if device.type == 'cuda':
        torch.cuda.synchronize()


def _reset_peak_mem(device):
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats()


def _peak_mem(device):
    """(allocated, reserved) peak bytes, or (None, None) off CUDA."""
    if device.type != 'cuda':
        return None, None
    return (int(torch.cuda.max_memory_allocated()),
            int(torch.cuda.max_memory_reserved()))


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
    # Async production mode is cached_mcdp; async EMA is deferred because trainer
    # checkpoints do not carry the EMA teacher.
    ns = SimpleNamespace(B_doc=args.B_doc, lambda_val=args.lambda_val,
                         ema_alpha=None, uncertainty='cached_mcdp', T=args.T,
                         mc_dropout_p=args.mc_dropout_p, L=None,
                         selection_mode=None, m=args.m, num_epochs=None,
                         no_registry=args.no_registry)
    fg_cfg = run_fast_grass._build_fast_grass_cfg(config, ns, steps_per_epoch)
    # cached-MCDP scores all of H; L is not a cached-MCDP knob (impl-details:
    # "Remove L from the cached-MCDP configuration and launcher").
    fg_cfg.pop('L', None)
    fg_cfg['batch_size'] = batch_size

    # config.yaml carries the SEQUENTIAL Fast-GRASS defaults; the async doc
    # ("Current Async Defaults") specifies different maintenance pressure. These
    # are not cosmetic: max_age_epochs 4 -> 2 halves the age threshold, so far more
    # slots go over-age and refresh work (a first-order term in t_mine_round) rises.
    # Timing the sequential values would under-measure the configuration we intend
    # to run. CLI flags still win.
    ASYNC_DEFAULTS = {'rho_start': 0.50, 'rho_end': 0.25, 'max_age_epochs': 2}
    applied = {}
    for key, val in ASYNC_DEFAULTS.items():
        cli = getattr(args, key, None)
        chosen = cli if cli is not None else val
        if fg_cfg.get(key) != chosen:
            applied[key] = (fg_cfg.get(key), chosen, 'cli' if cli is not None else 'async_default')
        fg_cfg[key] = chosen
    # max_age_steps is derived from max_age_epochs, so recompute after overriding
    fg_cfg['max_age_steps'] = fg_cfg['max_age_epochs'] * fg_cfg['steps_per_epoch']
    for k, (was, now, src) in applied.items():
        print(f"[mine-timing] {k}: config={was} -> {now} ({src})", flush=True)

    T = int(fg_cfg.get('T', 3))

    # Model time for the whole round. Step 0 would give every slot age 0, so
    # nothing is ever over-age and refresh work is badly under-measured; default to
    # a representative steady-state step instead.
    source_checkpoint_step = (args.source_checkpoint_step
                              if args.source_checkpoint_step is not None
                              else steps_per_epoch)

    # --- frozen model ---
    base_model = ctx['base_model']
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    dtype = torch.bfloat16 if device.type == 'cuda' else torch.float32
    student = AutoModel.from_pretrained(base_model, torch_dtype=dtype).to(device).eval()
    for p in student.parameters():
        p.requires_grad_(False)
    # set the MC dropout rate on every Dropout module (mirrors the trainer);
    # dropout_only() flips just these back on for the stochastic passes.
    mc_p = fg_cfg.get('mc_dropout_p', 0.3)
    n_layers = 0
    for mod in student.modules():
        if isinstance(mod, torch.nn.Dropout):
            mod.p = mc_p
            n_layers += 1
    print(f"[mine-timing] MC dropout p={mc_p} on {n_layers} layers", flush=True)

    # The stale pickle supplies the docid sample/ordering for H only — the MC
    # states below come from real dropout passes, not from these embeddings.
    cache = NegativeCache.init_uniform(stale_embs, c_ids, fg_cfg, device)
    del stale_embs
    gc.collect()

    # --- Z_mc initialization (timed separately; async STARTUP, not round cost) ---
    _reset_peak_mem(device)
    _sync(device)
    t0 = time.perf_counter()
    Z_mc, init_stats = init_Z_mc(cache, corpus_lookup, student, tokenizer, T,
                                 fg_cfg, device)
    _sync(device)
    t_cache_mc_init = time.perf_counter() - t0
    mem_after_init = _peak_mem(device)
    print(f"[mine-timing] cache H | B_doc={cache.B_doc} | T={T} | "
          f"Z_mc={init_stats['cache_mc_bytes']/1e9:.3f} GB + "
          f"Z_mean={init_stats['cache_mean_bytes']/1e9:.3f} GB | "
          f"init {t_cache_mc_init:.1f}s "
          f"({init_stats['init_examples_encoded']:,} examples, "
          f"{init_stats['init_forward_batches']:,} forward batches)", flush=True)

    qids = all_qids[:args.max_queries] if args.max_queries else all_qids
    # ceil so the final partial batch is mined too — a full round must cover every
    # query in the mixture, not floor-drop the tail.
    n_batches = max(int(math.ceil(len(qids) / batch_size)), 1)

    maint_threshold = maintenance_interval_mined_queries(fg_cfg, batch_size)
    print(f"[mine-timing] cached-MCDP | B_doc={cache.B_doc} | T={T} | "
          f"lambda={fg_cfg['lambda_val']} | batch_size={batch_size} | "
          f"queries={len(qids)}/{total_queries} | batches={n_batches}", flush=True)
    print(f"[mine-timing] maintain every {maint_threshold:,} mined queries "
          f"(cache_update_interval={fg_cfg['cache_update_interval']} * "
          f"batch_size={batch_size}) | model step={source_checkpoint_step:,}",
          flush=True)
    if len(qids) < 2 * maint_threshold:
        print(f"[mine-timing] WARNING: sample of {len(qids):,} queries crosses "
              f"< 2 maintenance thresholds ({maint_threshold:,} each). Per-interval "
              f"maintenance cost will be a single sample; use --max_queries "
              f">= {2 * maint_threshold:,} for a reliable gate number.", flush=True)

    reservoir = QueryMCReservoir(fg_cfg.get('recent_query_reservoir_size', 128))
    driver = MaintenanceDriver(fg_cfg, batch_size)

    # --- warmup: one batch (warms cudnn/allocator) ---
    warm_qids = qids[:batch_size]
    if warm_qids:
        mine_batch_cached_mcdp(cache, student, tokenizer, warm_qids,
                               qid_to_text, qrels_dict, T, fg_cfg, device,
                               chunk_size=args.chunk_size)
    _sync(device)
    # Warmup mining called record_selection, which set cache.selected_indicator for
    # the warmup slots. Reset it (and the cumulative score counter) so timed mining
    # and in-round maintenance start from clean, uncontaminated utility state.
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

    # --- timed mining over the (subset of the) mixture, with periodic in-round
    #     maintenance. per_batch_time captures ONLY mining compute (synced);
    #     maintenance and JSONL write are tracked separately.
    per_batch_time = []
    maint_times, maint_records = [], []
    tot_doc_calls_mining = 0
    tot_q_examples = tot_q_batches = 0
    queries_processed = 0
    mem_after_mine = mem_after_maint = (None, None)

    def _run_maintenance():
        """One bounded in-round maintenance interval, timed."""
        q_res, res_qids = reservoir.get()
        _sync(device)
        mt0 = time.perf_counter()
        rec = maintain_interval_cached_mcdp(
            cache, student, tokenizer, corpus_lookup, c_ids,
            (q_res, res_qids), source_checkpoint_step, T, fg_cfg, device,
            qrels_dict=qrels_dict)
        _sync(device)
        maint_times.append(time.perf_counter() - mt0)
        maint_records.append(rec)
        return rec

    for b in range(n_batches):
        batch_qids = qids[b * batch_size:(b + 1) * batch_size]
        batch_qids = list(dict.fromkeys(batch_qids))
        if not batch_qids:
            continue
        _sync(device)
        t0 = time.perf_counter()
        mined, slots, q_mc, mstats = mine_batch_cached_mcdp(
            cache, student, tokenizer, batch_qids, qid_to_text,
            qrels_dict, T, fg_cfg, device, chunk_size=args.chunk_size)
        _sync(device)
        per_batch_time.append(time.perf_counter() - t0)
        if b == 0:
            mem_after_mine = _peak_mem(device)

        queries_processed += len(batch_qids)
        # the defining cached-MCDP invariant
        tot_doc_calls_mining += mstats['mcdp_doc_encoder_calls_mining']
        tot_q_examples += mstats['query_examples_encoded']
        tot_q_batches += mstats['query_forward_batches']

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

        reservoir.add(q_mc, batch_qids)

        # periodic in-round maintenance on the mined-QUERY trigger
        driver.add(len(batch_qids))
        while driver.should_fire():
            driver.consume()
            rec = _run_maintenance()
            if len(maint_times) == 1:
                mem_after_maint = _peak_mem(device)
            print(f"[mine-timing]   maintenance #{driver.n_intervals} after "
                  f"{driver.mined_total:,} mined queries: "
                  f"refresh={rec['num_refresh']} replace={rec['num_replace']} "
                  f"budget={rec['maintenance_budget_interval']} "
                  f"({maint_times[-1]:.2f}s)", flush=True)

    # Round end: fold the remainder, but run a final bounded interval ONLY if
    # useful pending state exists (arch doc, "Cache Maintenance Semantics").
    ran_final_interval = False
    if driver.round_end_should_maintain(cache):
        _run_maintenance()
        ran_final_interval = True
        print(f"[mine-timing]   final round-end interval "
              f"({driver.pending:,} pending queries, {maint_times[-1]:.2f}s)",
              flush=True)

    if jsonl_f is not None:
        jsonl_f.close()

    if tot_doc_calls_mining != 0:
        raise AssertionError(
            f"cached-MCDP mining performed {tot_doc_calls_mining} document encoder "
            f"calls; the architecture requires mcdp_doc_encoder_calls_mining == 0 "
            f"(a regression to lazy fresh-MCDP)")

    mining_compute = float(np.sum(per_batch_time))
    qps = queries_processed / mining_compute if mining_compute > 0 else 0.0
    n_intervals = len(maint_times)
    maint_total = float(np.sum(maint_times)) if maint_times else 0.0
    maint_per_interval = (maint_total / n_intervals) if n_intervals else 0.0

    # --- extrapolate to a full mined round over the whole mixture ---
    # A full round mines every query in the mixture and therefore runs
    # floor(total_queries / maintenance_interval_mined_queries) maintenance
    # intervals. Both terms must be scaled — mining wall alone would understate a
    # round by the entire maintenance cost.
    spts, spts_src = _resolve_train_step(args)
    mine_full = (mining_compute / queries_processed) * total_queries \
        if queries_processed else mining_compute
    n_intervals_full = total_queries // maint_threshold
    maint_full = maint_per_interval * n_intervals_full
    jsonl_write_full = (jsonl_write_time / queries_processed) * total_queries \
        if (jsonl_f is not None and queries_processed) else 0.0

    # one round = full-mixture mine + ALL its in-round maintenance intervals
    # (+ JSONL write when measured; excluded and flagged otherwise).
    t_mine_round = mine_full + maint_full + jsonl_write_full
    async_mine_every = _async_cadence(t_mine_round, spts, args.safety_margin)

    maint_extrapolation_warning = None
    if n_intervals == 0:
        maint_extrapolation_warning = (
            "NO maintenance interval fired during the timed sample, so the "
            "maintenance term of t_mine_round is 0 and the round cost is "
            f"UNDERSTATED. Re-run with --max_queries >= {2 * maint_threshold}.")
    elif n_intervals == 1:
        maint_extrapolation_warning = (
            "only ONE maintenance interval fired; per-interval cost is a single "
            f"sample. Re-run with --max_queries >= {2 * maint_threshold} for a "
            "reliable gate number.")

    def _agg(key):
        return int(sum(r.get(key, 0) for r in maint_records))

    record = {
        'kind': 'mine_timing',
        'mode': 'real',
        'device': str(device),
        'base_model': base_model,
        'uncertainty': 'cached_mcdp',
        'B_doc': cache.B_doc,
        'T': T,
        'm': fg_cfg['m'],
        'lambda_val': fg_cfg['lambda_val'],
        'mc_dropout_p': mc_p,
        'rho_start': fg_cfg['rho_start'],
        'rho_end': fg_cfg['rho_end'],
        'max_age_epochs': fg_cfg['max_age_epochs'],
        'max_age_steps': fg_cfg['max_age_steps'],
        'async_default_overrides': {k: {'config': w, 'used': n, 'source': s}
                                    for k, (w, n, s) in applied.items()},
        'selection_mode': fg_cfg['selection_mode'],
        'batch_size': batch_size,
        'no_registry': bool(args.no_registry),
        'chunk_size': args.chunk_size,
        'total_queries_mixture': total_queries,
        'queries_processed': queries_processed,
        'batches_timed': len(per_batch_time),
        'mining_compute_only_s': mining_compute,
        'queries_per_second': qps,
        'mean_batch_mining_s': float(np.mean(per_batch_time)) if per_batch_time else None,
        'median_batch_mining_s': float(np.median(per_batch_time)) if per_batch_time else None,

        # --- cached-MCDP encoder accounting (examples != encoder calls) ---
        'mcdp_doc_encoder_calls_mining': int(tot_doc_calls_mining),   # must be 0
        'mcdp_query_mc_passes': T,
        'mcdp_query_examples_encoded': int(tot_q_examples),
        'mcdp_query_forward_batches': int(tot_q_batches),
        'mcdp_doc_encoder_calls_maintenance': _agg('maintenance_docs_encoded'),
        'mcdp_docs_encoded_maintenance': _agg('maintenance_docs_encoded'),
        'maintenance_examples_encoded': _agg('maintenance_examples_encoded'),
        'maintenance_forward_batches': _agg('maintenance_forward_batches'),
        'cache_score_pairs': int(cache.cache_score_pairs),

        # --- cache init (async STARTUP, not part of t_mine_round) ---
        't_cache_mc_init': t_cache_mc_init,
        'cache_mc_bytes': init_stats['cache_mc_bytes'],
        'cache_mean_bytes': init_stats['cache_mean_bytes'],
        'init_examples_encoded': init_stats['init_examples_encoded'],
        'init_forward_batches': init_stats['init_forward_batches'],

        # --- periodic in-round maintenance ---
        'maintenance_model_step': int(source_checkpoint_step),
        'cache_update_interval': fg_cfg['cache_update_interval'],
        'maintenance_interval_mined_queries': int(maint_threshold),
        'maintenance_budget_interval': int(
            cache._interval_budget(source_checkpoint_step, fg_cfg)),
        'num_maintenance_intervals': n_intervals,
        'num_maintenance_intervals_full_round': int(n_intervals_full),
        'ran_final_round_end_interval': ran_final_interval,
        'cache_maintenance_time': maint_total,
        'cache_maintenance_time_per_interval_s': maint_per_interval,
        'cache_maintenance_time_full_round_s': maint_full,
        'num_refresh_total': _agg('num_refresh'),
        'num_replace_total': _agg('num_replace'),
        'num_recertified_candidates_total': _agg('num_recertified_candidates'),
        'num_over_age_total': _agg('num_over_age'),
        'over_age_backlog_final': (maint_records[-1]['over_age_backlog']
                                   if maint_records else None),
        'maintenance_extrapolation_warning': maint_extrapolation_warning,

        # --- feasibility is time AND memory ---
        'peak_mem_allocated_bytes': (mem_after_maint[0] or mem_after_mine[0]
                                     or mem_after_init[0]),
        'peak_mem_reserved_bytes': (mem_after_maint[1] or mem_after_mine[1]
                                    or mem_after_init[1]),
        'peak_mem_after_init_bytes': mem_after_init[0],
        'peak_mem_after_mine_bytes': mem_after_mine[0],
        'peak_mem_after_maintenance_bytes': mem_after_maint[0],

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
    tag = f"bdoc{cache.B_doc}_T{T}_cached_mcdp"
    _write_json(record, tag)

    del student, cache, Z_mc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return 0


def _print_report(r):
    print("\n" + "=" * 66)
    print("  ASYNC FAST-GRASS — MINER TIMING (t_mine_round)")
    print("=" * 66)
    print(f"  uncertainty          : {r['uncertainty']} (async EMA deferred)")
    print(f"  B_doc / T / lambda   : {r['B_doc']:,} / {r['T']} / {r['lambda_val']}")
    print(f"  rho / max_age_epochs : {r['rho_start']}->{r['rho_end']} / "
          f"{r['max_age_epochs']} ({r['max_age_steps']:,} steps)"
          f"{'  [async defaults applied]' if r['async_default_overrides'] else ''}")
    print(f"  queries processed    : {r['queries_processed']:,} / "
          f"{r['total_queries_mixture']:,} (mixture)")
    print(f"  mining compute (sub) : {r['mining_compute_only_s']:.2f} s "
          f"(mine-only, excl. maintenance + JSONL write)")
    print(f"  queries_per_second   : {r['queries_per_second']:,.1f}")
    print(f"  doc encodes (mining) : {r['mcdp_doc_encoder_calls_mining']} "
          f"{'✅ cached-MCDP' if r['mcdp_doc_encoder_calls_mining'] == 0 else '❌ REGRESSION'}")
    print(f"  query MC encodes     : {r['mcdp_query_examples_encoded']:,} examples "
          f"in {r['mcdp_query_forward_batches']:,} forward batches "
          f"(T={r['mcdp_query_mc_passes']} passes)")
    print("-" * 66)
    print(f"  Z_mc init (STARTUP)  : {r['t_cache_mc_init']:.1f} s | "
          f"Z_mc {r['cache_mc_bytes']/1e9:.3f} GB + Z_mean "
          f"{r['cache_mean_bytes']/1e9:.3f} GB  [NOT in t_mine_round]")
    if r['peak_mem_reserved_bytes']:
        print(f"  peak GPU memory      : {r['peak_mem_allocated_bytes']/1e9:.2f} GB "
              f"allocated | {r['peak_mem_reserved_bytes']/1e9:.2f} GB reserved")
    print("-" * 66)
    print(f"  maintain every       : {r['maintenance_interval_mined_queries']:,} mined "
          f"queries (cache_update_interval={r['cache_update_interval']} * "
          f"batch_size={r['batch_size']})")
    print(f"  maint budget/interval: {r['maintenance_budget_interval']:,} docs "
          f"@ model step {r['maintenance_model_step']:,}")
    print(f"  intervals measured   : {r['num_maintenance_intervals']} "
          f"(full round: {r['num_maintenance_intervals_full_round']:,})"
          f"{'  [+1 round-end]' if r['ran_final_round_end_interval'] else ''}")
    print(f"  maintenance measured : {r['cache_maintenance_time']:.2f} s total | "
          f"{r['cache_maintenance_time_per_interval_s']:.2f} s/interval "
          f"(refresh={r['num_refresh_total']} replace={r['num_replace_total']} "
          f"recert={r['num_recertified_candidates_total']})")
    print(f"  maintenance doc enc. : {r['mcdp_docs_encoded_maintenance']:,} docs -> "
          f"{r['maintenance_examples_encoded']:,} examples in "
          f"{r['maintenance_forward_batches']:,} forward batches")
    if r['maintenance_extrapolation_warning']:
        print(f"  ⚠️  {r['maintenance_extrapolation_warning']}")
    print("-" * 66)
    print(f"  mining full mixture  : {r['mining_wall_full_mixture_extrapolated_s']:.1f} s "
          f"(extrapolated)")
    print(f"  maintenance / round  : {r['cache_maintenance_time_full_round_s']:.1f} s "
          f"({r['num_maintenance_intervals_full_round']:,} intervals, extrapolated)")
    if r['jsonl_write_timing_included']:
        print(f"  JSONL write (round)  : {r['jsonl_write_full_extrapolated_s']:.1f} s "
              f"(extrapolated; included in t_mine_round)")
    else:
        print("  JSONL write (round)  : EXCLUDED (pass --write_jsonl_timing to include)")
    print(f"  => t_mine_round      : {r['t_mine_round']:.1f} s "
          f"(full mine + all in-round maintenance"
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

    T = 3
    cfg = dict(
        model_name="mine_timing_smoke", uncertainty='cached_mcdp',
        B_doc=20, m=1, selection_mode='topk', lambda_val=0.5, beta=5.0,
        T=T, mc_dropout_p=0.3,
        ema_alpha=0.999, rho_start=0.50, rho_end=0.25,
        cache_update_interval=2, max_age_epochs=2, utility_ema_decay=0.95,
        utility_floor=0.01, utility_remember_threshold=0.05, K=3, R_fraction=0.25,
        uniform_candidate_fraction=0.75, replacement_candidate_multiplier=2,
        recent_query_reservoir_size=8, reentry_top_k=5, R_size_factor=0.5,
        cache_init_seed=42,
        learning_rate=1e-4, num_epochs=1, batch_size=4, mc_batch_size=16,
        passage_max_len=128, query_max_len=128,
        steps_per_epoch=6, total_steps=6, max_age_steps=24)
    source_checkpoint_step = cfg['steps_per_epoch']   # never 0 (age would be 0)

    cache = NegativeCache.init_uniform(embs, c_ids, cfg, device, dim=dim)
    student = DropoutMockModel(hidden=dim, p=cfg['mc_dropout_p'])
    student.eval()
    tok = MockTokenizer()

    t0 = time.perf_counter()
    Z_mc, init_stats = init_Z_mc(cache, corpus_lookup, student, tok, T, cfg, device)
    t_init = time.perf_counter() - t0
    # T genuine dropout passes, not one deterministic embedding repeated
    states_differ = not torch.allclose(Z_mc[0], Z_mc[1], atol=1e-6)

    batch_size = cfg['batch_size']
    n_batches = max(int(math.ceil(len(all_qids) / batch_size)), 1)
    reservoir = QueryMCReservoir(cfg['recent_query_reservoir_size'])
    driver = MaintenanceDriver(cfg, batch_size)
    times, doc_calls_mining, maint_times, maint_records = [], 0, [], []
    q_proc = 0
    # accumulate EVERY batch's mined negatives so the contract checks cover the whole
    # round, not just the last batch's `mined`.
    all_mined = {}
    # H is snapshotted PER BATCH, not once per round: periodic in-round maintenance
    # replaces cache entries while the round is being mined, so early records
    # legitimately come from an older H than later ones. A single pre-round snapshot
    # would spuriously fail once the first replacement lands.
    from_H = True

    t0round = time.perf_counter()
    for b in range(n_batches):
        bq = all_qids[b * batch_size:(b + 1) * batch_size]
        if not bq:
            continue
        H_now = set(cache.docids)
        t0 = time.perf_counter()
        mined, slots, q_mc, mstats = mine_batch_cached_mcdp(
            cache, student, tok, bq, qid_to_text, qrels_dict, T, cfg,
            device, chunk_size=args.chunk_size)
        times.append(time.perf_counter() - t0)
        q_proc += len(bq)
        doc_calls_mining += mstats['mcdp_doc_encoder_calls_mining']
        from_H = from_H and all(d in H_now for negs in mined.values() for d in negs)
        all_mined.update(mined)
        reservoir.add(q_mc, bq)

        driver.add(len(bq))
        while driver.should_fire():
            driver.consume()
            q_res, res_qids = reservoir.get()
            mt0 = time.perf_counter()
            maint_records.append(maintain_interval_cached_mcdp(
                cache, student, tok, corpus_lookup, c_ids,
                (q_res, res_qids), source_checkpoint_step, T, cfg, device,
                qrels_dict=qrels_dict))
            maint_times.append(time.perf_counter() - mt0)
    if driver.round_end_should_maintain(cache):
        q_res, res_qids = reservoir.get()
        mt0 = time.perf_counter()
        maint_records.append(maintain_interval_cached_mcdp(
            cache, student, tok, corpus_lookup, c_ids, (q_res, res_qids),
            source_checkpoint_step, T, cfg, device, qrels_dict=qrels_dict))
        maint_times.append(time.perf_counter() - mt0)
    round_wall = time.perf_counter() - t0round

    # every interval in the round must be stamped with the SAME model time
    steps_fixed = all(r['maintenance_model_step'] == source_checkpoint_step
                      for r in maint_records)
    budgets_fixed = len({r['maintenance_budget_interval'] for r in maint_records}) <= 1

    # contract over ALL mined negatives (every query in the round):
    clean = all(d not in qrels_dict.get(q, set())
                for q, negs in all_mined.items() for d in negs)
    all_queries_mined = set(all_mined) == set(all_qids)
    b_doc_ok = len(cache.docids) == cache.B_doc == len(set(cache.docids))

    t_mine_round = round_wall
    spts = 0.05  # mock train step for the cadence-formula smoke
    cadence = _async_cadence(t_mine_round, spts, args.safety_margin)

    ok = (len(times) > 0 and all(np.isfinite(t) and t >= 0 for t in times)
          and doc_calls_mining == 0 and states_differ and clean and from_H
          and all_queries_mined and b_doc_ok and len(maint_records) > 0
          and steps_fixed and budgets_fixed
          and cadence is not None and cadence >= 1)

    print(f"  batches mined        : {len(times)} | queries {q_proc}")
    print(f"  Z_mc init            : {t_init*1e3:.2f} ms | "
          f"{init_stats['init_examples_encoded']} examples | T states differ: {states_differ}")
    print(f"  doc encodes (mining) : {doc_calls_mining} (must be 0)")
    print(f"  maintain every       : {driver.threshold} mined queries "
          f"-> {len(maint_records)} intervals")
    print(f"  maintenance total    : {np.sum(maint_times)*1e3:.2f} ms "
          f"(refresh={sum(r['num_refresh'] for r in maint_records)} "
          f"replace={sum(r['num_replace'] for r in maint_records)})")
    print(f"  model step fixed     : {steps_fixed} | budget fixed: {budgets_fixed}")
    print(f"  t_mine_round (mock)  : {t_mine_round*1e3:.2f} ms")
    print(f"  async_mine_every_steps (mock spts={spts}) : {cadence}")
    print(f"  all queries mined    : {all_queries_mined} ({len(all_mined)}/{len(all_qids)})")
    print(f"  clean negatives (all): {clean}")
    print(f"  negatives from H     : {from_H} (per-batch snapshot; H changes mid-round)")
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
    ap.add_argument('--T', type=int, default=None,
                    help='cached MC passes per doc AND per query (e.g. 3)')
    ap.add_argument('--m', type=int, default=None, help='negatives per query')
    ap.add_argument('--lambda_val', type=float, default=None,
                    help='g = s_hat + lambda*sigma (async default 0.5; always '
                         'compare against 0)')
    ap.add_argument('--mc_dropout_p', type=float, default=None)
    ap.add_argument('--rho_start', type=float, default=None,
                    help='maintenance budget fraction at start (async default 0.50)')
    ap.add_argument('--rho_end', type=float, default=None,
                    help='maintenance budget fraction at end (async default 0.25; '
                         'config.yaml carries the sequential 0.10)')
    ap.add_argument('--max_age_epochs', type=int, default=None,
                    help='cache staleness cap in epochs (async default 2; '
                         'config.yaml carries the sequential 4). Halving it roughly '
                         'doubles the over-age refresh pressure.')
    ap.add_argument('--batch_size', type=int, default=None)
    ap.add_argument('--chunk_size', type=int, default=None,
                    help='chunk the Q x Z_mc score over cache slots so the T score '
                         'matrices need not be resident at once')
    ap.add_argument('--source_checkpoint_step', type=int, default=None,
                    help='model time for the whole round: cache age, rho/progress '
                         'and last_refreshed_step all use it. Default steps_per_epoch '
                         '(a representative steady state). Do NOT use 0 — every slot '
                         'would have age 0 and refresh work is under-measured.')
    ap.add_argument('--max_queries', type=int, default=None,
                    help='subset the mixture (t_mine_round is extrapolated to the '
                         'full mixture). Must cross >= 2 maintenance thresholds — '
                         'ordinarily >= 12800 — or maintenance cost is a single sample')
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
    ap.add_argument('--write_jsonl_timing', action='store_true',
                    help='write a temp mined round.jsonl (under analysis/.../work_*) '
                         'and fold its write time into t_mine_round (fix 5A). Off by '
                         'default, in which case JSONL write cost is excluded.')
    args = ap.parse_args()
    return run_synthetic(args) if args.synthetic else run_real(args)


if __name__ == "__main__":
    sys.exit(main())
