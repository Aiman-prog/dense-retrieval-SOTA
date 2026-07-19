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
             + num_epochs  * mining_only           # re-mine the mixture each epoch
             + n_maint     * maint_time            # amortized cache maintenance

  ASYNC (miner on GPU 1 overlaps training on GPU 0):
    training never blocks once the initial round exists; a slow miner costs
    staleness (data_age_steps), NOT wall time. So:
    async_wall = t_mine_round                      # initial round before step 0
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
                    help='one cache maintenance cycle wall; default from mine JSON')
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

    mining_only = _pick(args.mining_only,
                        mine.get('mining_wall_full_mixture_extrapolated_s'),
                        default=t_mine_round)
    maint_time = _pick(args.maint_time, mine.get('cache_maintenance_time_s'),
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
    # sequential maintenance count (amortized, every cache_update_interval steps)
    n_maint_seq = max(total_steps // max(cache_update_interval, 1), 1)

    # --- wall-time model ---
    seq_wall_analytic = (total_steps * t_train
                         + num_epochs * mining_only
                         + n_maint_seq * maint_time)
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

    async_wall = (t_mine_round                      # initial round before step 0
                  + total_steps * t_train           # continuous, non-blocking training
                  + n_ckpt * ckpt_write)            # checkpoint / handoff overhead
    speedup = seq_wall / async_wall if async_wall > 0 else 0.0

    # --- warnings ---
    warnings = []
    if mining_rounds_per_epoch < 1.0:
        warnings.append(
            f"miner produces only {mining_rounds_per_epoch:.2f} full rounds/epoch "
            f"(< 1): data_age_steps will exceed one epoch and negatives go stale. "
            f"Lower B_doc/L/T or accept staleness.")
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
    if ckpt_write_excluded:
        warnings.append(
            "checkpoint_write_time not in trainer JSON and not passed; async_wall "
            "EXCLUDES checkpoint I/O (optimistic). Pass --checkpoint_write_time.")

    record = {
        'kind': 'async_speed_estimate',
        'inputs': {
            'seconds_per_train_step': t_train,
            't_mine_round': t_mine_round,
            'mining_only_s': mining_only,
            'maint_time_s': maint_time,
            'checkpoint_write_time_s': ckpt_write,
            'checkpoint_write_time_excluded': ckpt_write_excluded,
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
            'n_maintenance_seq': n_maint_seq,
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
    print(f"  seconds_per_train_step : {i['seconds_per_train_step']:.4f} s")
    print(f"  t_mine_round           : {i['t_mine_round']:.1f} s  "
          f"(mine {i['mining_only_s']:.1f} + maint {i['maint_time_s']:.1f})")
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
    print(f"  estimated SEQUENTIAL wall : {_hms(r['estimated_sequential_wall_s'])} "
          f"({r['estimated_sequential_wall_s']:,.0f} s, {r['estimated_sequential_wall_source']})")
    print(f"  estimated ASYNC wall      : {_hms(r['estimated_async_wall_s'])} "
          f"({r['estimated_async_wall_s']:,.0f} s)")
    print(f"  EXPECTED SPEEDUP          : {r['expected_speedup']:.2f}x "
          f"(threshold {r['min_speedup_threshold']}x → "
          f"{'MEETS' if r['meets_speedup_threshold'] else 'BELOW'})")
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
