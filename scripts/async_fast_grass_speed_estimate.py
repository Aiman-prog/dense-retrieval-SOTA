"""
Async Fast-GRASS — expected speedup + refresh-cadence estimate (spec item 3).

Consumes the two timing JSONs written by the trainer/miner timing scripts
(``analysis/async_fast_grass_timing/{train,mine}_timing_*.json``) and answers the
pre-implementation question: does overlapping mining (GPU 1) with training (GPU 0)
actually buy meaningful wall-clock, and at what fixed ``async_mine_every_steps``?

This is a pure-Python arithmetic model (no torch, no GPU). Every intermediate
quantity is printed so the assumptions are auditable. The model:

  steps_per_epoch = ceil(total_queries_mixture / batch_size)
  total_steps     = steps_per_epoch * num_epochs

  SEQUENTIAL (mine inline, per batch, every epoch):
    seq_wall = total_steps * t_train_step          # all training steps
             + num_epochs  * t_mine_round          # re-mine the mixture each epoch

    t_mine_round ALREADY contains every periodic in-round maintenance interval,
    so there is NO separate maintenance term here. `cache_maintenance_time` from
    the miner JSON is a diagnostic breakdown only — adding it again (e.g.
    multiplied by num_maintenance_intervals) double-counts maintenance and
    inflates the sequential baseline, which would flatter async.

  ASYNC (miner on GPU 1 overlaps training on GPU 0):
    training never blocks once the initial round exists; a slow miner costs
    staleness (data_age_steps), NOT wall time. But startup is NOT free: the Z_mc
    cache must be built and the first round mined before trainer step 0.
    async_wall = async_startup                     # t_cache_mc_init + first round
               + total_steps * t_train_step        # continuous training
               + n_ckpt      * checkpoint_write_time

  speedup = seq_wall / async_wall

Cadence / staleness diagnostics:
  trainer_steps_per_mining_round = t_mine_round / t_train_step
  mining_rounds_per_epoch        = steps_per_epoch / trainer_steps_per_mining_round
  recommended async_mine_every_steps = ceil(trainer_steps_per_mining_round * margin)

A miner that produces < 1 full round per epoch (mining_rounds_per_epoch < 1) will
let data_age_steps grow past an epoch: negatives go stale. That is flagged.

Inputs may come from JSON files (default: newest in the timing dir) or be given
directly on the CLI, so the script also runs as a self-contained what-if without
any GPU artifacts.

Usage:
  # from the newest timing JSONs
  python scripts/async_fast_grass_speed_estimate.py
  # explicit files
  python scripts/async_fast_grass_speed_estimate.py \
      --train_timing_json analysis/async_fast_grass_timing/train_timing_bs64_m1_*.json \
      --mine_timing_json  analysis/async_fast_grass_timing/mine_timing_bdoc32000_*.json
  # pure what-if (no files)
  python scripts/async_fast_grass_speed_estimate.py \
      --seconds_per_train_step 0.42 --t_mine_round 5400 \
      --total_queries 367000 --batch_size 64 --num_epochs 3
"""
import argparse
import glob
import json
import math
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
OUT_DIR = project_root / 'analysis' / 'async_fast_grass_timing'


def _newest(pattern):
    cands = sorted(glob.glob(str(OUT_DIR / pattern)))
    return cands[-1] if cands else None


def _load_json(path):
    if path is None:
        return {}
    p = Path(path)
    if not p.exists():
        # allow a glob pattern to be passed directly
        hits = sorted(glob.glob(path))
        if not hits:
            print(f"[speed-estimate] WARNING: no file matched {path}", flush=True)
            return {}
        p = Path(hits[-1])
    return json.loads(p.read_text())


def _pick(cli_val, *json_vals, default=None):
    """First non-None among CLI override, then JSON sources, then default."""
    if cli_val is not None:
        return cli_val
    for v in json_vals:
        if v is not None:
            return v
    return default


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--train_timing_json', default=None,
                    help='trainer timing JSON (default: newest train_timing_*.json)')
    ap.add_argument('--mine_timing_json', default=None,
                    help='miner timing JSON (default: newest mine_timing_*.json)')
    ap.add_argument('--sequential_log', default=None,
                    help='optional Fast-GRASS cost_log.jsonl; if given, its summed '
                         'step_wall_time is used as the measured sequential wall '
                         '(ground truth) instead of the analytic estimate')
    # direct numeric overrides (enable a file-free what-if)
    ap.add_argument('--seconds_per_train_step', type=float, default=None)
    ap.add_argument('--t_mine_round', type=float, default=None)
    ap.add_argument('--mining_only', type=float, default=None,
                    help='full-mixture mine wall excl. maintenance; default from mine JSON')
    ap.add_argument('--maint_time', type=float, default=None,
                    help='maintenance wall per ROUND (diagnostic breakdown of '
                         't_mine_round; never re-added to a wall-time total)')
    ap.add_argument('--t_cache_mc_init', type=float, default=None,
                    help='Z_mc build time; charged once as async startup. Default '
                         'from the mine JSON.')
    ap.add_argument('--peak_mem_reserved_bytes', type=float, default=None,
                    help='miner peak reserved GPU bytes (default: from mine JSON)')
    ap.add_argument('--gpu_capacity_bytes', type=float, default=80e9,
                    help='miner GPU capacity for the memory gate (default 80 GB, A100)')
    ap.add_argument('--max_mem_fraction', type=float, default=0.85,
                    help='memory gate: peak reserved must be <= this fraction of '
                         'GPU capacity (default 0.85)')
    ap.add_argument('--checkpoint_write_time', type=float, default=None,
                    help='seconds to write one trainer checkpoint (async handoff '
                         'overhead). Default: from train JSON if present, else 0 '
                         '(and flagged as excluded).')
    ap.add_argument('--total_queries', type=int, default=None)
    ap.add_argument('--batch_size', type=int, default=None)
    ap.add_argument('--num_epochs', type=int, default=3)
    ap.add_argument('--cache_update_interval', type=int, default=None)
    ap.add_argument('--safety_margin', type=float, default=1.2,
                    help='cadence safety margin (doc range 1.1-1.25)')
    ap.add_argument('--min_speedup', type=float, default=1.3,
                    help='acceptance threshold; warns if expected speedup is below it')
    args = ap.parse_args()

    train = _load_json(args.train_timing_json or _newest('train_timing_*.json'))
    mine = _load_json(args.mine_timing_json or _newest('mine_timing_*.json'))

    t_train = _pick(args.seconds_per_train_step,
                    train.get('seconds_per_train_step'))
    t_mine_round = _pick(args.t_mine_round, mine.get('t_mine_round'))
    if t_train is None or t_mine_round is None:
        print("[speed-estimate] ERROR: need both seconds_per_train_step (trainer "
              "JSON or --seconds_per_train_step) and t_mine_round (miner JSON or "
              "--t_mine_round).", flush=True)
        return 2
    if t_train <= 0 or t_mine_round <= 0:
        print("[speed-estimate] ERROR: timings must be positive.", flush=True)
        return 2

    # diagnostic breakdown of t_mine_round — NEVER re-added to a wall-time total
    mining_only = _pick(args.mining_only,
                        mine.get('mining_wall_full_mixture_extrapolated_s'),
                        default=t_mine_round)
    maint_time = _pick(args.maint_time,
                       mine.get('cache_maintenance_time_full_round_s'),
                       mine.get('cache_maintenance_time'), default=0.0)
    n_maint_intervals = _pick(None, mine.get('num_maintenance_intervals_full_round'),
                              mine.get('num_maintenance_intervals'), default=0)
    # async startup: Z_mc build + the initial mined round, both before step 0
    t_cache_init = _pick(args.t_cache_mc_init, mine.get('t_cache_mc_init'),
                         default=0.0)
    total_queries = _pick(args.total_queries, mine.get('total_queries_mixture'))
    batch_size = _pick(args.batch_size, mine.get('batch_size'),
                       train.get('batch_size'))
    if total_queries is None or batch_size is None:
        print("[speed-estimate] ERROR: need total_queries and batch_size (miner "
              "JSON or --total_queries/--batch_size).", flush=True)
        return 2
    cache_update_interval = _pick(args.cache_update_interval,
                                  mine.get('cache_update_interval_config'),
                                  default=100)

    ckpt_write = _pick(args.checkpoint_write_time,
                       train.get('checkpoint_write_time'))
    ckpt_write_excluded = ckpt_write is None
    if ckpt_write is None:
        ckpt_write = 0.0

    num_epochs = args.num_epochs
    steps_per_epoch = max(int(math.ceil(total_queries / batch_size)), 1)
    total_steps = steps_per_epoch * num_epochs

    # --- cadence / staleness ---
    trainer_steps_per_mining_round = t_mine_round / t_train
    mining_rounds_per_epoch = steps_per_epoch / trainer_steps_per_mining_round
    recommended_cadence = max(
        int(math.ceil(trainer_steps_per_mining_round * args.safety_margin)), 1)

    # number of checkpoints the trainer writes over the run (async handoff I/O)
    n_ckpt = max(total_steps // recommended_cadence, 1)

    # --- wall-time model ---
    # t_mine_round already includes every in-round maintenance interval, so the
    # sequential baseline is training + one full re-mine per epoch. Adding a
    # separate maintenance term here would double-count it.
    seq_wall_analytic = total_steps * t_train + num_epochs * t_mine_round
    seq_wall = seq_wall_analytic
    seq_source = 'analytic'
    if args.sequential_log:
        p = Path(args.sequential_log)
        if p.exists():
            measured = 0.0
            with open(p) as f:
                for line in f:
                    try:
                        measured += float(json.loads(line).get('step_wall_time', 0.0))
                    except Exception:
                        continue
            if measured > 0:
                seq_wall = measured
                seq_source = f'measured_cost_log ({p.name})'
        else:
            print(f"[speed-estimate] WARNING: --sequential_log {p} not found; "
                  "using analytic sequential estimate.", flush=True)

    # async STARTUP is not free: the Z_mc cache is built and the first round mined
    # before trainer step 0 can consume anything.
    async_startup = t_cache_init + t_mine_round
    async_wall = (async_startup                     # cache init + initial round
                  + total_steps * t_train           # continuous, non-blocking training
                  + n_ckpt * ckpt_write)            # checkpoint / handoff overhead
    speedup = seq_wall / async_wall if async_wall > 0 else 0.0

    # --- memory gate (feasibility is time AND memory) ---
    peak_reserved = _pick(args.peak_mem_reserved_bytes,
                          mine.get('peak_mem_reserved_bytes'))
    gpu_capacity = args.gpu_capacity_bytes
    mem_frac = (peak_reserved / gpu_capacity) if (peak_reserved and gpu_capacity) else None
    mem_gate_ok = (mem_frac <= args.max_mem_fraction) if mem_frac is not None else None

    # --- warnings ---
    warnings = []
    if mining_rounds_per_epoch < 1.0:
        warnings.append(
            f"miner produces only {mining_rounds_per_epoch:.2f} full rounds/epoch "
            f"(< 1): data_age_steps will exceed one epoch and negatives go stale. "
            f"Lower B_doc/T or accept staleness.")
    if speedup < args.min_speedup:
        warnings.append(
            f"expected speedup {speedup:.2f}x < acceptance threshold "
            f"{args.min_speedup}x: async overlap may not be worth the complexity "
            f"at these settings.")
    if recommended_cadence > steps_per_epoch:
        warnings.append(
            f"recommended async_mine_every_steps ({recommended_cadence}) exceeds "
            f"steps_per_epoch ({steps_per_epoch}): fewer than one checkpoint per "
            f"epoch feeds the miner.")
    # impl-details: "Initial practical range: 1000 to 2000, then retune from logs.
    # Avoid tiny values such as 100 because checkpoints are large."
    if recommended_cadence < 1000:
        warnings.append(
            f"recommended async_mine_every_steps ({recommended_cadence}) is below the "
            f"documented practical range 1000-2000. Checkpoints are large; a tiny "
            f"cadence (e.g. 100) burns wall time on checkpoint I/O. Consider "
            f"pinning async_mine_every_steps=1000.")
    elif recommended_cadence > 2000:
        warnings.append(
            f"recommended async_mine_every_steps ({recommended_cadence}) exceeds the "
            f"documented practical range 1000-2000: the miner is slow relative to "
            f"the trainer, so mined data will be stale on arrival.")
    if ckpt_write_excluded:
        warnings.append(
            "checkpoint_write_time not in trainer JSON and not passed; async_wall "
            "EXCLUDES checkpoint I/O (optimistic). Run fast_grass_train_timing.py "
            "without --no_checkpoint_probe, or pass --checkpoint_write_time.")
    if mem_gate_ok is False:
        warnings.append(
            f"MEMORY GATE FAILED: peak reserved {peak_reserved/1e9:.2f} GB is "
            f"{mem_frac:.0%} of the {gpu_capacity/1e9:.0f} GB miner GPU (limit "
            f"{args.max_mem_fraction:.0%}). Lower B_doc or T.")
    elif mem_gate_ok is None:
        warnings.append(
            "peak reserved memory unknown (miner JSON has none and "
            "--peak_mem_reserved_bytes not passed): the memory gate is UNCHECKED.")
    if mine.get('mcdp_doc_encoder_calls_mining') not in (0, None):
        warnings.append(
            f"ARCHITECTURE GATE FAILED: miner reported "
            f"{mine['mcdp_doc_encoder_calls_mining']} document encoder calls during "
            f"mining; cached-MCDP requires 0 (regression to lazy fresh-MCDP).")
    if mine.get('maintenance_extrapolation_warning'):
        warnings.append(f"miner JSON: {mine['maintenance_extrapolation_warning']}")

    record = {
        'kind': 'async_speed_estimate',
        'inputs': {
            'seconds_per_train_step': t_train,
            't_mine_round': t_mine_round,
            't_cache_mc_init': t_cache_init,
            # diagnostic breakdown of t_mine_round only — not summed into any wall
            'mining_only_s': mining_only,
            'maint_time_s': maint_time,
            'num_maintenance_intervals_per_round': n_maint_intervals,
            'checkpoint_write_time_s': ckpt_write,
            'checkpoint_write_time_excluded': ckpt_write_excluded,
            'peak_mem_reserved_bytes': peak_reserved,
            'gpu_capacity_bytes': gpu_capacity,
            'total_queries_mixture': total_queries,
            'batch_size': batch_size,
            'num_epochs': num_epochs,
            'cache_update_interval': cache_update_interval,
            'safety_margin': args.safety_margin,
            'train_timing_json': args.train_timing_json or _newest('train_timing_*.json'),
            'mine_timing_json': args.mine_timing_json or _newest('mine_timing_*.json'),
        },
        'derived': {
            'steps_per_epoch': steps_per_epoch,
            'total_steps': total_steps,
            'trainer_steps_per_mining_round': trainer_steps_per_mining_round,
            'mining_rounds_per_epoch': mining_rounds_per_epoch,
            'recommended_async_mine_every_steps': recommended_cadence,
            'n_checkpoints_over_run': n_ckpt,
            'async_startup_s': async_startup,
            'peak_mem_fraction_of_gpu': mem_frac,
        },
        'gates': {
            'architecture_zero_mining_doc_encodes': (
                mine.get('mcdp_doc_encoder_calls_mining') == 0
                if 'mcdp_doc_encoder_calls_mining' in mine else None),
            'memory_within_budget': mem_gate_ok,
            'rounds_per_epoch_at_least_one': mining_rounds_per_epoch >= 1.0,
            'speedup_meets_threshold': speedup >= args.min_speedup,
        },
        'estimated_sequential_wall_s': seq_wall,
        'estimated_sequential_wall_source': seq_source,
        'estimated_sequential_wall_analytic_s': seq_wall_analytic,
        'estimated_async_wall_s': async_wall,
        'expected_speedup': speedup,
        'min_speedup_threshold': args.min_speedup,
        'meets_speedup_threshold': speedup >= args.min_speedup,
        'warnings': warnings,
    }

    _print_report(record)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    from datetime import datetime
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    out = OUT_DIR / f"speed_estimate_{ts}.json"
    out.write_text(json.dumps(record, indent=2))
    print(f"[speed-estimate] wrote {out}", flush=True)
    return 0


def _hms(s):
    s = int(s)
    return f"{s // 3600}h {(s % 3600) // 60}m {s % 60}s"


def _print_report(r):
    d = r['derived']
    i = r['inputs']
    print("\n" + "=" * 68)
    print("  ASYNC FAST-GRASS — EXPECTED SPEEDUP & CADENCE ESTIMATE")
    print("=" * 68)
    g = r['gates']
    print(f"  seconds_per_train_step : {i['seconds_per_train_step']:.4f} s")
    print(f"  t_mine_round           : {i['t_mine_round']:.1f} s  "
          f"(breakdown: mine {i['mining_only_s']:.1f} + maint {i['maint_time_s']:.1f} "
          f"over {i['num_maintenance_intervals_per_round']} intervals)")
    print(f"  t_cache_mc_init        : {i['t_cache_mc_init']:.1f} s (once, startup)")
    print(f"  mixture / batch / epochs: {i['total_queries_mixture']:,} / "
          f"{i['batch_size']} / {i['num_epochs']}")
    print(f"  steps_per_epoch        : {d['steps_per_epoch']:,}  "
          f"(total {d['total_steps']:,})")
    print("-" * 68)
    print(f"  trainer_steps_per_mining_round : {d['trainer_steps_per_mining_round']:,.1f}")
    print(f"  mining_rounds_per_epoch        : {d['mining_rounds_per_epoch']:.2f}")
    print(f"  recommended async_mine_every_steps : {d['recommended_async_mine_every_steps']:,} "
          f"(steps/round * margin {i['safety_margin']})")
    print(f"  checkpoints over run   : {d['n_checkpoints_over_run']:,} "
          f"@ {i['checkpoint_write_time_s']:.1f}s each"
          f"{'  (EXCLUDED)' if i['checkpoint_write_time_excluded'] else ''}")
    print("-" * 68)
    print(f"  SEQUENTIAL wall : {_hms(r['estimated_sequential_wall_s'])} "
          f"({r['estimated_sequential_wall_s']:,.0f} s, {r['estimated_sequential_wall_source']})")
    print(f"    = total_steps * t_train + num_epochs * t_mine_round "
          f"(maintenance already inside t_mine_round)")
    print(f"  ASYNC wall      : {_hms(r['estimated_async_wall_s'])} "
          f"({r['estimated_async_wall_s']:,.0f} s)")
    print(f"    = startup {_hms(d['async_startup_s'])} + training + checkpoint I/O")
    print(f"  EXPECTED SPEEDUP: {r['expected_speedup']:.2f}x "
          f"(threshold {r['min_speedup_threshold']}x → "
          f"{'MEETS' if r['meets_speedup_threshold'] else 'BELOW'})")
    print("-" * 68)
    print("  GO / NO-GO")
    _mark = lambda v: 'PASS' if v is True else ('FAIL' if v is False else 'UNKNOWN')
    print(f"    [hard] mining doc encodes == 0 : "
          f"{_mark(g['architecture_zero_mining_doc_encodes'])}")
    mf = d['peak_mem_fraction_of_gpu']
    print(f"    [hard] memory within budget    : {_mark(g['memory_within_budget'])}"
          f"{f'  ({mf:.0%} of GPU)' if mf is not None else ''}")
    print(f"    [soft] >= 1 round per epoch    : "
          f"{_mark(g['rounds_per_epoch_at_least_one'])} "
          f"({d['mining_rounds_per_epoch']:.2f})")
    print(f"    [soft] speedup >= threshold    : "
          f"{_mark(g['speedup_meets_threshold'])} ({r['expected_speedup']:.2f}x)")
    print("    (correctness gates — population std, lambda=0 ranking, qrel leakage —")
    print("     are enforced by scripts/async_fast_grass_cache_semantics_test.py)")
    print("-" * 68)
    if r['warnings']:
        print("  WARNINGS:")
        for w in r['warnings']:
            print(f"    ! {w}")
    else:
        print("  no warnings — cadence and speedup look healthy")
    print("=" * 68)


if __name__ == "__main__":
    sys.exit(main())
