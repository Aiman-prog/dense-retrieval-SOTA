"""
Async Fast-GRASS — ORCHESTRATOR.

Prepares the step-0 input, starts the miner on GPU 1, runs the trainer on GPU 0,
and evaluates the result. Mirrors ``train_ance.py`` in process orchestration; the
mining algorithm underneath is cached-MCDP, not ANCE.

    run_setup
    build the initial cache: sample B_doc docids, encode every one T times with
      dropout using the BASE model, Z_mean = mean_t
    mine initial_data/ from that cache
    write mining_meta_initial.json + cache_state_initial.pt, then ready_initial LAST
    start the miner subprocess on miner_gpu (it may poll before any checkpoint exists)
    run the trainer subprocess on trainer_gpu in the foreground
    finally: terminate and wait for the miner
    evaluate on BRIGHT

The initial data is produced by the SAME cached-MCDP miner path as every later
round — it is not ANCE mining and uses no per-query FAISS top-P. Its metadata
records ``source_checkpoint="base_model"``, ``source_checkpoint_step=0``.

Usage:
  python scripts/train_async_fast_grass.py
  python scripts/train_async_fast_grass.py --debug --max_rounds 1
"""
import argparse
import json
import math
import os
import pickle
import shutil
import subprocess
import sys
import time
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModel

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root / 'src'))
sys.path.insert(0, str(project_root / 'scripts'))

from utils.helpers import (  # noqa: E402
    get_path, get_training_context, load_config, set_seed, log_startup_config,
    _load_corpus_lookup, _load_qrels, evaluate_bright,
)
from utils.negative_cache import NegativeCache  # noqa: E402
import run_fast_grass  # noqa: E402
from async_fast_grass_cached_mcdp import (  # noqa: E402
    build_async_cfg, mine_batch_cached_mcdp, maintenance_interval_mined_queries,
    steps_per_epoch, QueryMCReservoir, MaintenanceDriver,
    maintain_interval_cached_mcdp, canonicalize_positives,
    UnresolvablePositivesError, validate_refresh_schedule, format_refresh_report,
)
from async_fast_grass_handoff import (  # noqa: E402
    initial_paths, write_ready_initial, latest_committed_round,
)
from async_fast_grass_pilot import (  # noqa: E402
    maybe_apply_manifest, manifest_source_counts, load_manifest, ManifestError,
    evaluate_pilot_gate, format_gate_report, SOURCE_FILES,
)


def _log(msg):
    print(f"[AsyncFG] {msg}", flush=True)


def check_manifest_required(ctx, manifest, recipe):
    """Recipes built around a manifest must not silently run on the full mixture.

    ``async_fast_grass_pilot`` and ``_smoke`` derive every schedule number from their
    manifest: ``steps_per_epoch`` (516 / 16), the ``cache_update_interval`` that sizes
    the maintenance budget, and ``max_age_steps``. Point either at the full 330k
    mixture and you get a different experiment that still looks like a healthy run —
    the pilot would train 10,314 steps instead of 1,032 and nothing downstream would
    say so. An unset shell variable is enough to cause it, so this is a hard error
    rather than a warning.

    Returns an error string, or ``None`` when satisfied.
    """
    if ctx['args'].get('requires_manifest') and not manifest:
        return (f"recipe {recipe!r} requires --manifest but none was given. Its "
                f"steps_per_epoch, maintenance budget and max_age_steps are all "
                f"derived from a manifest-sized mixture; against the full mixture "
                f"this silently becomes a different experiment. Build one with "
                f"scripts/async_fast_grass_pilot.py build-manifest, and check that "
                f"ASYNC_FG_MANIFEST is actually set (an empty shell variable expands "
                f"to no flag at all).")
    return None


def _preflight_paths():
    """Resolve the processed inputs WITHOUT running ``run_setup``.

    Preflight must be a pure inspection: it runs before a long job to confirm what is
    already on disk. Missing files are reported, never created -- building the derived
    set is `python src/data/preprocessor.py`, not a side effect of training.

    Returns ``(corpus_file, qrels_file, missing)``.
    """
    processed = get_path("processed")
    corpus_file = processed / "reasonir_corpus.jsonl"
    qrels_file = processed / "train_qrels.txt"
    mixture_dir = processed / "training_mixture"
    required = (corpus_file, qrels_file, mixture_dir,
                *(mixture_dir / name for name in SOURCE_FILES.values()))
    missing = [str(p) for p in required
               if not p.exists()]
    return corpus_file, qrels_file, missing


def _preflight(corpus_file, qrels_file, debug=False, manifest=None, config=None,
               ctx=None):
    """Validate the REAL processed data before a long GPU job is submitted.

    The trainer resolves docids strictly, so any positive that is absent from
    ``reasonir_corpus.jsonl`` is a hard failure at step 0. ``run_setup`` MD5-dedupes
    passages and remaps only the corpus and qrels — the mixture keeps its original
    positive docid — so this is exactly where that bites. No GPU is touched and
    nothing is regenerated.
    """
    corpus_lookup = _load_corpus_lookup(corpus_file)
    qrels_dict = _load_qrels(qrels_file)
    train_items = run_fast_grass._load_train_items(debug=debug)

    print("\n" + "=" * 66)
    print("  ASYNC FAST-GRASS — PREFLIGHT (no GPU, no regeneration)")
    print("=" * 66)
    print(f"  corpus docs      : {len(corpus_lookup):,}")
    print(f"  qrels queries    : {len(qrels_dict):,}")
    print(f"  mixture items    : {len(train_items):,}")
    if not train_items:
        print("  ❌ the training mixture is EMPTY in the processed schema "
              "(positive_passages). Run preprocessing first.")
        return 2

    # The stale pickle supplies the initial H docid sample. It is a hard failure deep
    # inside build_initial_round, so catch it here rather than after the job starts.
    stale_pkl = get_path("temp_grass") / "stale_index" / "corpus.pkl"
    if stale_pkl.exists():
        print(f"  stale index      : found ({stale_pkl})")
    else:
        print(f"  ❌ stale index MISSING at {stale_pkl}. Async Fast-GRASS does not "
              f"rebuild the corpus index; build it once via run_fast_grass.py.")
        print("=" * 66)
        return 2

    qrels_docids = {d for s in qrels_dict.values() for d in s}
    qrels_missing = [d for d in qrels_docids if d not in corpus_lookup]
    print(f"  qrels docids     : {len(qrels_docids):,} "
          f"({len(qrels_missing):,} not in corpus)")

    raw_missing = sum(1 for it in train_items if it['pos_docid'] not in corpus_lookup)
    print(f"  positives absent from corpus (pre-canonicalization): {raw_missing:,} "
          f"({100 * raw_missing / len(train_items):.2f}%)")

    try:
        out, stats = canonicalize_positives(train_items, qrels_dict, corpus_lookup,
                                            log=lambda m: print(f"  {m}"))
    except UnresolvablePositivesError as e:
        # Not a warning: training on the remainder would shrink the mixture by an
        # unrecorded amount and break comparability with the baselines.
        print(f"  ❌ {e}")
        print("-" * 66)
        print("  FAIL  regenerate the corpus/qrels before running")
        print("=" * 66)
        return 2

    still_bad = [it for it in out if it['pos_docid'] not in corpus_lookup]
    print(f"  after canonicalization: kept {stats['kept']:,}/{stats['total']:,} "
          f"(remapped {stats['remapped']:,}, dropped {stats['dropped']:,}); "
          f"{len(still_bad):,} still unresolvable")

    ok = not still_bad and stats['kept'] == stats['total'] and stats['kept'] > 0

    # --- manifest: same load + apply the orchestrator and miner use ---
    if manifest:
        print("-" * 66)
        try:
            rows = load_manifest(manifest)
            out = maybe_apply_manifest(out, manifest,
                                       log=lambda m: print(f"  {m}"))[0]
        except ManifestError as e:
            print(f"  ❌ {e}")
            print("=" * 66)
            return 2
        counts = manifest_source_counts(rows)
        for source in sorted(counts):
            print(f"    {source:<9}: {counts[source]:>8,}")
        print(f"  manifest total   : {len(out):,}")

    # --- derived schedule + refresh eligibility ---
    if config is not None and ctx is not None:
        batch_size = int(ctx['args'].get('batch_size', 64))
        # build the cfg with the REAL steps_per_epoch: the max_age_epochs fallback
        # multiplies by it, so a placeholder would validate the wrong schedule
        spe = steps_per_epoch(len(out), batch_size)
        cfg = build_async_cfg(config, ctx, spe)
        print("-" * 66)
        print(f"  batch_size       : {batch_size}")
        print(f"  steps_per_epoch  : {spe:,}")
        print(f"  total_steps      : {cfg['total_steps']:,} "
              f"({cfg['num_epochs']} epochs)")
        errors, warnings, info = validate_refresh_schedule(cfg)
        print(format_refresh_report(errors, warnings, info))
        if errors:
            ok = False

    print("-" * 66)
    print(f"  {'PASS' if ok else 'FAIL'}  every mixture item has a strictly "
          f"resolvable positive and refresh can influence training")
    print("=" * 66)
    return 0 if ok else 1


def supervise(trainer, miner, poll_seconds=2.0, grace=120, log=print):
    """Run until the trainer exits, failing the run if the miner dies first.

    A dead miner is not benign: the trainer would run to ``max_steps`` on whatever
    round happened to be current, degenerating into sequential training on stale
    negatives while looking like a successful async run.

    The miner's status is checked BEFORE the loop exits and again AFTER the trainer
    finishes — a short poll interval plus a post-exit check means a miner that dies
    inside the same window as a trainer exit is still caught, instead of the run
    being reported successful.

    Returns ``(miner_failed_returncode_or_None, trainer_returncode)``.
    """
    miner_failed = None
    try:
        while True:
            if miner.poll() is not None and miner.returncode != 0:
                miner_failed = miner.returncode
                log(f"ERROR: miner exited with code {miner_failed} while the "
                    f"trainer was still running — terminating the trainer")
                _stop(trainer, grace)
                break
            try:
                trainer.wait(timeout=poll_seconds)
                break                      # trainer exited; miner re-checked below
            except subprocess.TimeoutExpired:
                pass
            if miner.poll() is not None and miner.returncode == 0:
                # a clean miner exit is expected only under --max_rounds
                log("miner finished cleanly (expected with --max_rounds); the "
                    "trainer keeps consuming the rounds already committed")
                trainer.wait()
                break
    finally:
        # Re-check AFTER the loop: the miner may have died during the same window
        # in which the trainer exited, which the in-loop check would have missed.
        if miner_failed is None and miner.poll() is not None and miner.returncode != 0:
            miner_failed = miner.returncode
            log(f"ERROR: miner exited with code {miner_failed} (detected after the "
                f"trainer finished) — the run consumed stale mined data")
        if miner.poll() is None:
            _stop(miner, grace)
    return miner_failed, trainer.returncode


def _stop(proc, grace):
    """terminate, then kill if it will not go."""
    proc.terminate()
    try:
        proc.wait(timeout=grace)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()


def _load_stale_docids(cfg):
    """Stale corpus pickle -> (embeddings, docids) for the initial H sample.

    The pickle supplies ONLY the docid sample/ordering; every MC state is a real
    dropout pass. Fast-GRASS never queries a full-corpus ANN index, so this is a
    one-off initialization artifact and must already exist.
    """
    stale_pkl = get_path("temp_grass") / "stale_index" / "corpus.pkl"
    if not stale_pkl.exists():
        raise FileNotFoundError(
            f"stale index not found at {stale_pkl}. Async Fast-GRASS does not "
            f"rebuild the full corpus; build it once via run_fast_grass.py, then "
            f"re-run.")
    with open(stale_pkl, 'rb') as f:
        data = pickle.load(f)
    return data[0], [str(x) for x in data[1]]


def mine_initial_data(cache, model, tokenizer, out_path, train_items, qid_to_text,
                      corpus_lookup, qrels_dict, cfg, device, batch_size):
    """Mine the step-0 round with the SAME machinery every later round uses.

    Query-MC reservoir, mined-query maintenance cadence, periodic in-round
    maintenance and the final partial-interval fold — mining the whole mixture
    against a frozen ``H`` would make round 0 behave unlike any other round.

    Model time is 0 (``source_checkpoint_step`` for the base model).

    Extracted so the integration smoke can drive the REAL code path with mocks
    instead of reproducing this loop. Returns a stats dict.
    """
    T = int(cfg['T'])
    n_batches = max(int(math.ceil(len(train_items) / batch_size)), 1)
    qid_to_pos = {it['query_id']: it['pos_docid'] for it in train_items}
    reservoir = QueryMCReservoir(cfg.get('recent_query_reservoir_size', 128))
    driver = MaintenanceDriver(cfg, batch_size)
    c_ids_all = list(corpus_lookup.keys())
    doc_calls, n_maint = 0, 0

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w') as f_out:
        for b in range(n_batches):
            batch = train_items[b * batch_size:(b + 1) * batch_size]
            if not batch:
                continue
            qids = list(dict.fromkeys(it['query_id'] for it in batch))
            mined, _slots, q_mc, mstats = mine_batch_cached_mcdp(
                cache, model, tokenizer, qids, qid_to_text, qrels_dict, T, cfg,
                device, chunk_size=cfg.get('score_chunk_size'))
            doc_calls += mstats['mcdp_doc_encoder_calls_mining']
            for qid in qids:
                f_out.write(json.dumps({
                    'query_id': qid,
                    'query': qid_to_text[qid],
                    'pos_docid': qid_to_pos.get(qid),
                    'neg_docids': mined.get(qid, []),
                }, ensure_ascii=False) + "\n")
            reservoir.add(q_mc, qids)
            driver.add(len(qids))
            while driver.should_fire():
                driver.consume()
                maintain_interval_cached_mcdp(
                    cache, model, tokenizer, corpus_lookup, c_ids_all,
                    reservoir.get(), 0, T, cfg, device, qrels_dict=qrels_dict)
                n_maint += 1
    # final partial interval, on the same rule as a normal round
    if driver.round_end_should_maintain(cache):
        maintain_interval_cached_mcdp(
            cache, model, tokenizer, corpus_lookup, c_ids_all, reservoir.get(), 0,
            T, cfg, device, qrels_dict=qrels_dict)
        n_maint += 1

    if doc_calls != 0:
        raise AssertionError(f"initial mining did {doc_calls} doc encodes; "
                             f"cached-MCDP requires 0")

    # Never persist selected_indicator accumulated across the WHOLE initial mixture:
    # utility is an interval-scoped signal ("selected at least once in a short
    # interval"), so a mixture-wide indicator would make almost every slot look
    # useful to the first real maintenance interval and suppress replacement.
    cache.selected_indicator.zero_()

    return {
        'num_maintenance_intervals': n_maint,
        'maintenance_interval_mined_queries': driver.threshold,
        'mcdp_doc_encoder_calls_mining': doc_calls,
        'num_queries': driver.mined_total,
    }


def build_initial_round(root, cfg, ctx, config, train_items, qid_to_text,
                        corpus_lookup, qrels_dict, device, batch_size):
    """Initial cache + initial mined data + ready_initial. Idempotent."""
    p = initial_paths(root)
    if p['ready'].exists():
        _log("ready_initial already present — skipping initial mine")
        return
    for stale in (p['training_data'],):
        shutil.rmtree(stale, ignore_errors=True)
    p['training_data'].mkdir(parents=True, exist_ok=True)

    stale_embs, c_ids = _load_stale_docids(cfg)
    base_model = ctx['base_model']
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    dtype = torch.bfloat16 if device.type == 'cuda' else torch.float32
    model = AutoModel.from_pretrained(base_model, torch_dtype=dtype).to(device).eval()
    for prm in model.parameters():
        prm.requires_grad_(False)
    for mod in model.modules():
        if isinstance(mod, torch.nn.Dropout):
            mod.p = cfg.get('mc_dropout_p', 0.3)

    T = int(cfg['T'])
    _log(f"initial cache: sampling B_doc={cfg['B_doc']:,} and encoding each doc "
         f"T={T} times with the base model")
    t0 = time.perf_counter()
    cache, init_stats = NegativeCache.init_cached_mcdp(
        stale_embs, c_ids, corpus_lookup, model, tokenizer, cfg, device)
    t_init = time.perf_counter() - t0
    del stale_embs
    _log(f"initial cache built in {t_init:.1f}s | Z_mc="
         f"{init_stats['cache_mc_bytes'] / 1e9:.2f} GB | "
         f"{init_stats['init_examples_encoded']:,} doc examples encoded")

    t0 = time.perf_counter()
    mine_stats = mine_initial_data(
        cache, model, tokenizer, p['training_data'] / "mined.jsonl", train_items,
        qid_to_text, corpus_lookup, qrels_dict, cfg, device, batch_size)
    t_mine = time.perf_counter() - t0
    n_maint = mine_stats['num_maintenance_intervals']
    driver_threshold = mine_stats['maintenance_interval_mined_queries']

    p['mining_meta'].write_text(json.dumps({
        'round_no': 0,
        'source_checkpoint': 'base_model',
        'source_checkpoint_step': 0,
        'B_doc': cache.B_doc, 'T': T, 'm': cfg['m'],
        'lambda_val': cfg['lambda_val'],
        'num_queries': len(train_items),
        'maintenance_interval_mined_queries': driver_threshold,
        'num_maintenance_intervals': n_maint,
        'maintenance_model_step': 0,
        't_cache_mc_init': t_init,
        't_initial_mine': t_mine,
        'mcdp_doc_encoder_calls_mining': 0,
        # initialization doc encodes are recorded HERE, never attributed to a
        # mining round (mcdp_doc_encoder_calls_mining must stay 0 everywhere)
        'init_examples_encoded': init_stats['init_examples_encoded'],
        'init_forward_batches': init_stats['init_forward_batches'],
        'cache_mc_bytes': init_stats['cache_mc_bytes'],
        'registry_deferred': True,
    }, indent=2))
    cache.save_state(p['cache_state'])
    write_ready_initial(root)          # marker LAST
    _log(f"initial round ready: mined {len(train_items):,} queries in {t_mine:.1f}s "
         f"| {n_maint} maintenance interval(s)")

    del cache, model
    if device.type == 'cuda':
        torch.cuda.empty_cache()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--recipe', default='async_fast_grass')
    ap.add_argument('--debug', action='store_true', help='512-item mixture')
    ap.add_argument('--max_rounds', type=int, default=None,
                    help='stop the miner after N rounds (smoke/debug)')
    ap.add_argument('--no_eval', action='store_true')
    ap.add_argument('--no_compile', action='store_true')
    ap.add_argument('--fresh', action='store_true',
                    help='wipe the async handoff root and start a new run. Required '
                         'when stale rounds exist, because Phase 1 has no trainer '
                         'resume: the trainer always restarts at step 0.')
    ap.add_argument('--preflight', action='store_true',
                    help='validate the mixture/corpus/qrels against the REAL '
                         'processed data and exit, without touching a GPU. Run this '
                         'on the cluster before submitting a long job.')
    ap.add_argument('--allow_single_gpu', action='store_true',
                    help='run trainer and miner on ONE GPU. They then serialise and '
                         'contend for memory — for debugging only.')
    ap.add_argument('--lambda_val', type=float, default=None,
                    help='override training.async_fast_grass.lambda_val for the '
                         'lambda sweep. Pinned at submit time so a queued job cannot '
                         'pick up whatever value config.yaml holds when it starts.')
    ap.add_argument('--bootstrap_checkpoint_step', type=int, default=None,
                    help='trainer saves ONE extra checkpoint at this step so the '
                         'miner stops idling at startup. 0/absent = off. This is an '
                         'ABLATION, not a speedup: the extra round mutates the '
                         'persisted cache, so hold it constant across a sweep.')
    ap.add_argument('--run_suffix', default=None,
                    help='isolate this run: appends to the model dir name AND the '
                         'async handoff root, so concurrent sweep arms cannot '
                         'overwrite each other\'s checkpoints or mined rounds.')
    ap.add_argument('--manifest', default=None,
                    help='pilot manifest JSONL (scripts/async_fast_grass_pilot.py '
                         'build-manifest). Restricts and orders the mixture; passed '
                         'through to the miner so both processes see the same set.')
    args = ap.parse_args()

    config = load_config()
    ctx = get_training_context(args.recipe)
    # Both overrides land in ctx['args'] because build_async_cfg starts from
    # dict(ctx['args']) — injecting here keeps the initial round, the miner cfg and
    # the model dir consistent instead of patching three call sites.
    if args.lambda_val is not None:
        ctx['args']['lambda_val'] = float(args.lambda_val)
    if args.run_suffix:
        ctx['args']['model_name'] = f"{ctx['args']['model_name']}_{args.run_suffix}"
    set_seed(config.get('seed', 42))

    # after the lambda_val / run_suffix overrides land in ctx['args'], and ahead of
    # the preflight branch so both the validator and the training path print it
    log_startup_config(args.recipe, ctx)

    # PREFLIGHT FIRST: it must report what is on disk, not resolve paths through the
    # preprocessor -- a validator that leans on the resolver validates nothing.
    manifest_error = check_manifest_required(ctx, args.manifest, args.recipe)

    if args.preflight:
        if manifest_error:
            print(f"\n❌ {manifest_error}")
            return 2
        corpus_file, qrels_file, missing = _preflight_paths()
        if missing:
            print("\n❌ preflight cannot run — these processed inputs are absent:")
            for p in missing:
                print(f"     {p}")
            print("   Preflight never regenerates data; run preprocessing first.")
            return 2
        return _preflight(corpus_file, qrels_file, debug=args.debug,
                          manifest=args.manifest, config=config, ctx=ctx)

    if manifest_error:
        raise RuntimeError(manifest_error)

    from data.preprocessor import require_derived_artifacts
    from data.preprocessor import MIXTURE_FILES, require_mixture_files
    corpus_file, _query_file, qrels_file = require_derived_artifacts()
    require_mixture_files(get_path("processed") / "training_mixture", MIXTURE_FILES)

    # Detect GPUs BEFORE restricting visibility; subprocesses get their own pin.
    n_gpus = torch.cuda.device_count()
    trainer_gpu = str(ctx['args'].get('trainer_gpu', 0))
    miner_gpu = str(ctx['args'].get('miner_gpu', 1))
    if n_gpus < 2 and not args.allow_single_gpu:
        # Silently colocating serialises the two processes and doubles peak memory
        # on one device — the async design assumes they overlap.
        raise RuntimeError(
            f"async Fast-GRASS needs 2 visible GPUs (trainer + miner), found "
            f"{n_gpus}. Submit with --gpus-per-task=2, or pass --allow_single_gpu "
            f"to colocate them for debugging.")
    if n_gpus < 2:
        miner_gpu = trainer_gpu
        _log("WARNING: --allow_single_gpu — trainer and miner share one GPU; they "
             "will serialise and contend for memory")
    os.environ['CUDA_VISIBLE_DEVICES'] = trainer_gpu
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    _log(f"{n_gpus} GPU(s) | trainer -> GPU {trainer_gpu} | miner -> GPU {miner_gpu}")

    root_name = "async_mining" + (f"_{args.run_suffix}" if args.run_suffix else "")
    root = get_path("temp_fast_grass") / root_name
    root.mkdir(parents=True, exist_ok=True)

    # Phase 1 has no trainer resume: the trainer always starts at global_step 0.
    # Leaving stale ready_N rounds in place would have a step-0 trainer immediately
    # consume rounds mined from a PREVIOUS run's checkpoints, and the miner would
    # resume numbering after them — mixing two runs' data in one model.
    existing = latest_committed_round(root)
    has_initial = initial_paths(root)['ready'].exists()
    if existing > 0 or has_initial:
        if not args.fresh:
            raise RuntimeError(
                f"async handoff root {root} already holds a previous run "
                f"(ready_initial={has_initial}, newest committed round={existing}). "
                f"Phase 1 cannot resume a trainer, so continuing would train a "
                f"fresh step-0 model on stale rounds. Re-run with --fresh to wipe "
                f"it, or move the directory aside to keep it.")
        _log(f"--fresh: wiping {root} (had ready_initial={has_initial}, newest "
             f"committed round={existing})")
        shutil.rmtree(root, ignore_errors=True)
        root.mkdir(parents=True, exist_ok=True)

    train_items = run_fast_grass._load_train_items(debug=args.debug)
    if args.debug:
        train_items = train_items[:512]
    qid_to_text = {it['query_id']: it['query'] for it in train_items}
    corpus_lookup = _load_corpus_lookup(corpus_file)
    qrels_dict = _load_qrels(qrels_file)
    # positives in the mixture may carry pre-dedup docids (preprocessor remaps the
    # corpus and qrels, not the mixture); resolve them before anything is mined
    train_items, _canon = canonicalize_positives(train_items, qrels_dict,
                                                 corpus_lookup, log=_log)
    # manifest AFTER canonicalization, exactly as the miner does it
    train_items, manifest_meta = maybe_apply_manifest(train_items, args.manifest,
                                                      log=_log)
    qid_to_text = {it['query_id']: it['query'] for it in train_items}

    batch_size = ctx['args'].get('batch_size', 64)
    spe = steps_per_epoch(len(train_items), batch_size)
    num_epochs = ctx['args'].get('num_epochs', 2)
    max_steps = spe * num_epochs
    cfg = build_async_cfg(config, ctx, spe)
    _log(f"{len(train_items):,} examples | {spe:,} steps/epoch | "
         f"{num_epochs} epochs | {max_steps:,} total steps | maintain every "
         f"{maintenance_interval_mined_queries(cfg, batch_size):,} mined queries")

    # Refresh eligibility is a START-of-run decision: a config in which no refreshed
    # round can reach the trainer produces a run that looks fine and answers nothing.
    refresh_errors, refresh_warnings, refresh_info = validate_refresh_schedule(cfg)
    _log("refresh schedule:\n" + format_refresh_report(
        refresh_errors, refresh_warnings, refresh_info))
    if refresh_errors:
        raise RuntimeError(
            "refusing to start: the cache-refresh schedule cannot influence this run "
            "(" + "; ".join(refresh_errors) + ")")

    output_model_dir = get_path("models") / ctx['args']['model_name']
    output_model_dir.mkdir(parents=True, exist_ok=True)
    # A leftover high-numbered checkpoint from a previous run would permanently
    # shadow every new save and pin the miner to dead weights (train_ance.py:185).
    stale = [d for d in output_model_dir.glob("checkpoint-*") if d.is_dir()]
    for ckpt in stale:
        shutil.rmtree(ckpt, ignore_errors=True)
    if stale:
        _log(f"removed {len(stale)} stale checkpoint(s) from {output_model_dir.name}")

    build_initial_round(root, cfg, ctx, config, train_items, qid_to_text,
                        corpus_lookup, qrels_dict, device, batch_size)

    scripts_dir = Path(__file__).parent
    miner_env = {**os.environ, 'CUDA_VISIBLE_DEVICES': miner_gpu}
    miner_cmd = [
        sys.executable, str(scripts_dir / "run_async_fast_grass_miner.py"),
        '--async_dir', str(root),
        '--output_model_dir', str(output_model_dir),
        '--corpus_file', str(corpus_file),
        '--qrels_file', str(qrels_file),
        '--recipe', args.recipe,
    ]
    if args.lambda_val is not None:
        # The miner rebuilds cfg from the recipe, so the override must travel with
        # it — lambda is a MINING parameter and the trainer never reads it.
        miner_cmd += ['--lambda_val', repr(float(args.lambda_val))]
    if args.manifest:
        # both processes must mine/size the SAME set, or steps_per_epoch and the
        # maintenance budget would be derived from a different mixture than is mined
        miner_cmd += ['--manifest', str(args.manifest)]
    if args.max_rounds:
        miner_cmd += ['--max_rounds', str(args.max_rounds)]
    if args.debug:
        miner_cmd += ['--debug']
    miner = subprocess.Popen(miner_cmd, env=miner_env)
    _log(f"miner started on GPU {miner_gpu} (pid {miner.pid})")

    trainer_env = {**os.environ, 'CUDA_VISIBLE_DEVICES': trainer_gpu}
    trainer_cmd = [
        sys.executable, str(scripts_dir / "run_async_fast_grass_train.py"),
        '--async_dir', str(root),
        '--output_dir', str(output_model_dir),
        '--corpus_file', str(corpus_file),
        '--max_steps', str(max_steps),
        '--steps_per_epoch', str(spe),
        '--recipe', args.recipe,
    ]
    if args.no_compile:
        trainer_cmd += ['--no_compile']
    if args.bootstrap_checkpoint_step is not None:
        trainer_cmd += ['--bootstrap_checkpoint_step',
                        str(int(args.bootstrap_checkpoint_step))]

    trainer = subprocess.Popen(trainer_cmd, env=trainer_env)
    _log(f"trainer started on GPU {trainer_gpu} (pid {trainer.pid})")

    miner_failed, trainer_rc = supervise(trainer, miner, log=_log)
    committed = latest_committed_round(root)
    _log(f"miner stopped | rounds committed: {committed}")

    # --- validity gate -------------------------------------------------------
    # The summary is written FIRST and unconditionally: a failing run is exactly the
    # one whose diagnostics matter, and raising before writing would discard them.
    gate_min_steps = ctx['args'].get('pilot_gate_min_steps')
    trainer_summary = {}
    summary_path = output_model_dir / "async_trainer_summary.json"
    if summary_path.exists():
        trainer_summary = json.loads(summary_path.read_text())

    gate_ok, gate_reasons, gate_details = (True, [], None)
    if gate_min_steps is not None:
        gate_ok, gate_reasons, gate_details = evaluate_pilot_gate(
            root, trainer_summary, output_model_dir, miner_failed,
            int(gate_min_steps))

    (output_model_dir / "async_run_summary.json").write_text(json.dumps({
        'recipe': args.recipe,
        'run_suffix': args.run_suffix,
        'lambda_val': cfg['lambda_val'],
        'manifest': str(args.manifest) if args.manifest else None,
        'manifest_sha256': (manifest_meta or {}).get('sha256'),
        'num_train_items': len(train_items),
        'steps_per_epoch': spe,
        'max_steps': max_steps,
        'max_age_steps': cfg['max_age_steps'],
        'miner_failed': miner_failed,
        'trainer_returncode': trainer_rc,
        'rounds_committed': committed,
        'refresh_info': refresh_info,
        'pilot_gate_min_steps': gate_min_steps,
        'pilot_gate_ok': gate_ok if gate_min_steps is not None else None,
        'pilot_gate_reasons': gate_reasons,
        'pilot_gate_details': gate_details,
    }, indent=2, default=str))

    if miner_failed is not None:
        raise RuntimeError(
            f"miner died (exit {miner_failed}); the run is invalid because the "
            f"trainer would otherwise have continued on stale mined data")
    if trainer_rc != 0:
        raise RuntimeError(f"trainer exited with code {trainer_rc}")

    if gate_min_steps is not None:
        # Nonzero exit, not just a printed FAIL: SLURM would otherwise report the job
        # as successful, and an invalid lambda=0 arm would appear to authorise the
        # nonzero arms. Recipes without the key (the full run) never reach this.
        print(format_gate_report(gate_ok, gate_reasons, gate_details), flush=True)
        if not gate_ok:
            _log(f"run summary written to {output_model_dir/'async_run_summary.json'}")
            return 1

    if not args.no_eval:
        evaluate_bright(ctx, config, output_model_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
