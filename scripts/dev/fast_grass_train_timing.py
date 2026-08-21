"""
Async Fast-GRASS — Phase 0 timing calibration: TRAINER side.

Measures ``t_train_step`` = wall time for ONE trainer-only fresh-loss optimizer
step on PRE-MINED data, i.e. the exact loss path the async trainer runs while the
miner prepares the next round (async_fast_grass_architecture.md, "Timing
Calibration Before Async Training"):

    fresh-encode queries -> fresh-encode positives -> fresh-encode mined negatives
    -> InfoNCE loss -> backward -> optimizer step -> scheduler step

This is trainer-only: NO online mining, NO cache scoring / maintenance, NO eval.
Negatives are pre-assigned (uniform-random corpus docids) so the step cost is a
faithful stand-in for consuming an already-mined round; the identity of the
negatives never affects the encode+backward cost (fixed by batch_size, m, and
sequence lengths).

The encode / optimizer / compile / loss setup mirrors ``run_fast_grass_pipeline``
so the measured step matches the real trainer. Nothing here edits the Fast-GRASS
core (run_fast_grass.py / helpers.py / negative_cache.py); it only imports from it.

Reports ``seconds_per_train_step`` and ``steps_per_hour`` and writes a JSON record
under ``analysis/async_fast_grass_timing/``.

Modes:
  real (default) : GPU cluster timing on the real base model + training mixture.
  --synthetic    : CPU-only smoke on a tiny mock model to verify the harness runs
                   (NO representative numbers). Catches code bugs before the GPU run.

Usage:
  # local/CPU smoke (correctness only)
  python scripts/dev/fast_grass_train_timing.py --synthetic
  # cluster/GPU (real numbers); tiny sanity first, then full
  python scripts/dev/fast_grass_train_timing.py --max_queries 128 --steps 50
  python scripts/dev/fast_grass_train_timing.py --steps 500
"""
import argparse
import gc
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch.optim import AdamW
from torch.nn.utils import clip_grad_norm_
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup

try:
    import bitsandbytes as bnb
    _BNB_AVAILABLE = True
except ImportError:
    _BNB_AVAILABLE = False

project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root / 'src'))
sys.path.insert(0, str(project_root / 'scripts'))
# the --synthetic path reuses mock fixtures that live with the suites in tests/
sys.path.insert(0, str(project_root / 'tests'))

from utils.helpers import (
    get_training_context, load_config, get_path,
    encode_batch_tensor, _load_corpus_lookup, set_seed,
)
from run_fast_grass import _build_fast_grass_cfg, _load_train_items

OUT_DIR = project_root / 'analysis' / 'async_fast_grass_timing'


# ---- shared: build a pre-mined batch + one optimizer step ------------------

PREMINED_SOURCE_STAND_IN = "uniform_random_stand_in"
PREMINED_SOURCE_REAL = "real_mined_jsonl"


def _uniform_random_stand_in_negatives(train_items, c_ids, m, seed):
    """Assign ``m`` uniform-random corpus docids per item as STAND-IN negatives.

    These are NOT real Fast-GRASS mined negatives — they are a deterministic
    (seeded) placeholder so the trainer has something to fresh-encode. The step
    cost (encode + backward + optimizer) is fixed by batch_size, m, and sequence
    lengths, so the identity of the negatives does not affect ``t_train_step``.
    Source tag: ``uniform_random_stand_in``.
    """
    rng = np.random.default_rng(seed)
    n = len(c_ids)
    negs = {}
    for it in train_items:
        idx = rng.integers(0, n, size=m)
        negs[it['query_id']] = [c_ids[int(i)] for i in idx]
    return negs


def _load_premined_from_dir(premined_dir):
    """Load real mined negatives ``{query_id: [neg_docids]}`` from JSONL files.

    Optional path (``--premined_data_dir``) for later timing on a real mined round
    (e.g. the async miner's output). Accepts either a ``neg_docids`` list or a
    ``negative_passages: [{docid,...}]`` field per record. Missing/short queries
    are back-filled with uniform-random stand-ins by the caller.
    """
    out = {}
    for f_path in sorted(Path(premined_dir).glob("*.jsonl")):
        if f_path.name.startswith('.'):
            continue
        with open(f_path) as f:
            for line in f:
                d = json.loads(line)
                qid = str(d['query_id'])
                if 'neg_docids' in d:
                    out[qid] = [str(x) for x in d['neg_docids']]
                elif 'negative_passages' in d:
                    out[qid] = [str(p['docid']) for p in d['negative_passages']]
    return out


def _train_step(student, tokenizer, loss_fn, optimizer, scheduler, batch_items,
                premined, corpus_lookup, m, q_max, p_max, enc_bs, max_grad_norm,
                device):
    """One trainer-only fresh-loss optimizer step (mirrors run_fast_grass_pipeline
    lines 463-488). Returns the scalar loss (float)."""
    queries, positives, negatives = [], [], []
    for it in batch_items:
        negs = premined.get(it['query_id'])
        if not negs or len(negs) < m:
            continue
        queries.append(it['query'])
        positives.append(corpus_lookup.get(it['pos_docid'], ''))
        negatives.append([corpus_lookup.get(d, '') for d in negs[:m]])
    if not queries:
        return None

    student.train()
    q_embs = encode_batch_tensor(student, tokenizer, queries, device, q_max,
                                 enc_bs, requires_grad=True)
    d_texts = [t for pos, negs in zip(positives, negatives)
               for t in [pos] + negs]
    d_embs = encode_batch_tensor(student, tokenizer, d_texts, device, p_max,
                                 enc_bs, requires_grad=True)
    loss = loss_fn(q_embs, d_embs)

    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    clip_grad_norm_(student.parameters(), max_grad_norm)
    optimizer.step()
    scheduler.step()
    return float(loss.item())


def _sync(device):
    if device.type == 'cuda':
        torch.cuda.synchronize()


def _peak_mem(device):
    """(allocated, reserved) peak bytes, or (None, None) off CUDA."""
    if device.type != 'cuda':
        return None, None
    return (int(torch.cuda.max_memory_allocated()),
            int(torch.cuda.max_memory_reserved()))


def _time_checkpoint_write(student, tokenizer, optimizer, scheduler, out_dir):
    """Time one trainer checkpoint write, ANCE-style with optimizer.pt LAST.

    The async trainer pays this every ``async_mine_every_steps``, and the speed
    estimate charges ``n_checkpoints * checkpoint_write_time`` against async wall
    time. Without a measured value the estimate has to exclude checkpoint I/O and
    flag itself optimistic.

    ``optimizer.pt`` is written last because it is the miner's validity flag
    (``is_valid_checkpoint``): a checkpoint is only readable once it exists.
    """
    import shutil
    ckpt = Path(out_dir) / "checkpoint-timing-probe"
    if ckpt.exists():
        shutil.rmtree(ckpt)
    ckpt.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    try:
        model = getattr(student, '_orig_mod', student)   # unwrap torch.compile
        model.save_pretrained(str(ckpt))
        tokenizer.save_pretrained(str(ckpt))
        torch.save(scheduler.state_dict(), ckpt / "scheduler.pt")
        torch.save(optimizer.state_dict(), ckpt / "optimizer.pt")   # LAST
        dt = time.perf_counter() - t0
    except Exception as e:
        print(f"[train-timing] checkpoint-write probe failed ({e})", flush=True)
        return None, None
    size = sum(f.stat().st_size for f in ckpt.rglob('*') if f.is_file())
    shutil.rmtree(ckpt, ignore_errors=True)
    return dt, int(size)


def _percentiles(times):
    a = np.asarray(times, dtype=np.float64)
    return {
        'seconds_per_train_step_mean': float(a.mean()),
        'seconds_per_train_step_median': float(np.median(a)),
        'seconds_per_train_step_p10': float(np.percentile(a, 10)),
        'seconds_per_train_step_p90': float(np.percentile(a, 90)),
        'seconds_per_train_step_std': float(a.std()),
        'seconds_per_train_step_min': float(a.min()),
        'seconds_per_train_step_max': float(a.max()),
    }


def _write_json(record, tag):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    path = OUT_DIR / f"train_timing_{tag}_{ts}.json"
    path.write_text(json.dumps(record, indent=2))
    print(f"[train-timing] wrote {path}", flush=True)
    return path


# ---- real GPU timing -------------------------------------------------------

def run_real(args):
    config = load_config()
    ctx = get_training_context('fast_grass')
    set_seed(config.get('seed', 42))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type != 'cuda':
        print("[train-timing] WARNING: no CUDA — real-mode numbers are NOT "
              "representative. Use --synthetic for a CPU correctness smoke.",
              flush=True)

    from data.preprocessor import run_setup
    from models.temperature_scaled_loss import TemperatureScaledContrastiveLoss
    corpus_file, _query_file, _qrels_file = run_setup()
    corpus_lookup = _load_corpus_lookup(corpus_file)
    c_ids = list(corpus_lookup.keys())

    train_items = _load_train_items()
    if args.max_queries:
        train_items = train_items[:args.max_queries]

    batch_size = args.batch_size or config['training']['fast_grass'].get('batch_size', 64)
    steps_per_epoch = max(len(train_items) // batch_size, 1)
    ns = SimpleNamespace(B_doc=None, lambda_val=None, ema_alpha=None,
                         uncertainty='mcdp', T=None, mc_dropout_p=None, L=None,
                         selection_mode=None, m=args.m, num_epochs=None)
    fg_cfg = _build_fast_grass_cfg(config, ns, steps_per_epoch)

    m = fg_cfg['m']
    q_max = fg_cfg['query_max_len']
    p_max = fg_cfg['passage_max_len']
    enc_bs = fg_cfg.get('mc_batch_size', 512)
    max_grad_norm = fg_cfg.get('max_grad_norm', 1.0)
    lr = float(fg_cfg['learning_rate'])
    weight_decay = fg_cfg.get('weight_decay', 0.01)
    warmup_ratio = fg_cfg.get('warmup_ratio', 0.1)
    temperature = ctx['temperature']

    seed = config.get('seed', 42)
    if args.premined_data_dir:
        premined = _load_premined_from_dir(args.premined_data_dir)
        stand_in = _uniform_random_stand_in_negatives(train_items, c_ids, m, seed)
        n_filled = 0
        for it in train_items:
            qid = it['query_id']
            if qid not in premined or len(premined[qid]) < m:
                premined[qid] = stand_in[qid]
                n_filled += 1
        premined_source = (PREMINED_SOURCE_REAL if n_filled == 0
                           else f"{PREMINED_SOURCE_REAL}+{n_filled}_uniform_fill")
        print(f"[train-timing] pre-mined negatives from {args.premined_data_dir} "
              f"({n_filled} uniform-fill of {len(train_items)})", flush=True)
    else:
        premined = _uniform_random_stand_in_negatives(train_items, c_ids, m, seed)
        premined_source = PREMINED_SOURCE_STAND_IN
        print("[train-timing] pre-mined negatives = uniform-random STAND-IN "
              "(not real Fast-GRASS negatives; step cost is source-independent)",
              flush=True)

    # --- model + optimizer + compile: mirror run_fast_grass_pipeline ---
    base_model = ctx['base_model']
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    student = AutoModel.from_pretrained(
        base_model,
        torch_dtype=torch.bfloat16 if device.type == 'cuda' else torch.float32
    ).to(device)

    if _BNB_AVAILABLE and device.type == 'cuda':
        optimizer = bnb.optim.AdamW8bit(student.parameters(), lr=lr,
                                        weight_decay=weight_decay)
        print("[train-timing] AdamW8bit enabled", flush=True)
    else:
        if hasattr(student, 'gradient_checkpointing_enable'):
            try:
                student.gradient_checkpointing_enable()
            except Exception:
                pass
        optimizer = AdamW(student.parameters(), lr=lr, weight_decay=weight_decay)
        print("[train-timing] AdamW + gradient checkpointing", flush=True)
    student.train()

    loss_fn = TemperatureScaledContrastiveLoss(temperature=temperature)
    if not args.no_compile and device.type == 'cuda':
        try:
            torch._dynamo.config.suppress_errors = True
            student = torch.compile(student, dynamic=True)
            print("[train-timing] torch.compile enabled", flush=True)
        except Exception as e:
            print(f"[train-timing] torch.compile skipped ({e})", flush=True)

    n_batches = max(len(train_items) // batch_size, 1)
    total_sched = (args.warmup_steps + args.steps)
    warmup_sched = int(warmup_ratio * total_sched)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_sched, total_sched)

    def batch_at(step):
        b = step % n_batches
        return train_items[b * batch_size:(b + 1) * batch_size]

    print(f"[train-timing] base={base_model} | batch_size={batch_size} | m={m} | "
          f"train_group_size={1 + m} | q/p_len={q_max}/{p_max} | "
          f"warmup={args.warmup_steps} timed={args.steps} | items={len(train_items)}",
          flush=True)

    # --- warmup (compile trace, cudnn autotune, allocator warm) ---
    for s in range(args.warmup_steps):
        _train_step(student, tokenizer, loss_fn, optimizer, scheduler,
                    batch_at(s), premined, corpus_lookup, m, q_max, p_max,
                    enc_bs, max_grad_norm, device)
    _sync(device)

    # --- timed steps (per-step wall, synced) ---
    times, losses = [], []
    for s in range(args.steps):
        _sync(device)
        t0 = time.perf_counter()
        loss = _train_step(student, tokenizer, loss_fn, optimizer, scheduler,
                           batch_at(args.warmup_steps + s), premined,
                           corpus_lookup, m, q_max, p_max, enc_bs, max_grad_norm,
                           device)
        _sync(device)
        dt = time.perf_counter() - t0
        if loss is None:
            continue
        times.append(dt)
        losses.append(loss)

    stats = _percentiles(times)
    spts = stats['seconds_per_train_step_median']
    steps_per_hour = 3600.0 / spts if spts > 0 else 0.0
    peak_alloc, peak_reserved = _peak_mem(device)

    # async handoff I/O: the trainer writes a checkpoint every
    # async_mine_every_steps, and the miner cannot start a round without one.
    ckpt_write_time, ckpt_bytes = (None, None)
    if not args.no_checkpoint_probe:
        ckpt_write_time, ckpt_bytes = _time_checkpoint_write(
            student, tokenizer, optimizer, scheduler, args.checkpoint_probe_dir
            or (OUT_DIR / 'ckpt_probe'))
        if ckpt_write_time is not None:
            print(f"[train-timing] checkpoint write: {ckpt_write_time:.2f} s "
                  f"({ckpt_bytes/1e9:.2f} GB, optimizer.pt last)", flush=True)

    record = {
        'kind': 'train_timing',
        'mode': 'real',
        'device': str(device),
        'base_model': base_model,
        'batch_size': batch_size,
        # async handoff cadence knobs consumed by the speed estimate
        'checkpoint_write_time': ckpt_write_time,
        'checkpoint_bytes': ckpt_bytes,
        'ready_poll_steps': fg_cfg.get('logging_steps', 100),
        'peak_mem_allocated_bytes': peak_alloc,
        'peak_mem_reserved_bytes': peak_reserved,
        'm': m,
        'train_group_size': 1 + m,
        'query_max_len': q_max,
        'passage_max_len': p_max,
        'enc_batch_size': enc_bs,
        'compiled': (not args.no_compile) and device.type == 'cuda',
        'bnb_adamw8bit': _BNB_AVAILABLE and device.type == 'cuda',
        'warmup_steps': args.warmup_steps,
        'timed_steps': len(times),
        'max_queries': args.max_queries,
        'premined_source': premined_source,
        'premined_data_dir': args.premined_data_dir,
        'seconds_per_train_step': spts,
        'steps_per_hour': steps_per_hour,
        'mean_loss': float(np.mean(losses)) if losses else None,
        **stats,
    }

    print("\n" + "=" * 66)
    print("  ASYNC FAST-GRASS — TRAINER TIMING (t_train_step)")
    print("=" * 66)
    print(f"  device               : {device}")
    print(f"  batch_size x (1+m)   : {batch_size} x {1 + m}")
    print(f"  timed steps          : {len(times)} (after {args.warmup_steps} warmup)")
    print(f"  seconds_per_train_step (median) : {spts:.4f} s")
    print(f"    mean {stats['seconds_per_train_step_mean']:.4f} | "
          f"p10 {stats['seconds_per_train_step_p10']:.4f} | "
          f"p90 {stats['seconds_per_train_step_p90']:.4f} | "
          f"std {stats['seconds_per_train_step_std']:.4f}")
    print(f"  steps_per_hour       : {steps_per_hour:,.0f}")
    if ckpt_write_time is not None:
        print(f"  checkpoint write     : {ckpt_write_time:.2f} s "
              f"({ckpt_bytes/1e9:.2f} GB) — async handoff I/O")
    else:
        print("  checkpoint write     : NOT MEASURED (async wall will exclude "
              "checkpoint I/O and be flagged optimistic)")
    if peak_reserved:
        print(f"  peak GPU memory      : {peak_alloc/1e9:.2f} GB allocated | "
              f"{peak_reserved/1e9:.2f} GB reserved")
    print("=" * 66)

    tag = f"bs{batch_size}_m{m}"
    _write_json(record, tag)

    del student
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return 0


# ---- synthetic CPU smoke ---------------------------------------------------

def run_synthetic(args):
    print("\n" + "=" * 66)
    print("  TRAINER TIMING — SYNTHETIC (CPU, mock model; NO real numbers)")
    print("=" * 66)
    from fast_grass_train_smoke import GradMockModel, MockTokenizer

    class _Loss(torch.nn.Module):
        """InfoNCE over interleaved [pos, neg0, ...] doc rows (m=1 layout)."""
        def forward(self, q, d):
            g = int(d.shape[0] // q.shape[0])
            scores = q @ d.t()
            target = torch.arange(0, d.shape[0], g, device=q.device)
            return torch.nn.functional.cross_entropy(scores / 0.02, target)

    device = torch.device('cpu')
    dim = 8
    m = args.m
    n_corpus = 40
    c_ids = [f"d{i}" for i in range(n_corpus)]
    corpus_lookup = {d: f"document {d} body text" for d in c_ids}
    train_items = [{'query_id': f"q{i}", 'query': f"query number {i}",
                    'pos_docid': c_ids[i % n_corpus]} for i in range(16)]
    premined = _uniform_random_stand_in_negatives(train_items, c_ids, m, 0)

    student = GradMockModel(hidden=dim)
    tokenizer = MockTokenizer()
    optimizer = AdamW(student.parameters(), lr=1e-3)
    scheduler = get_linear_schedule_with_warmup(optimizer, 1, args.steps + 2)
    loss_fn = _Loss()

    batch_size = 4
    n_batches = max(len(train_items) // batch_size, 1)
    times, losses = [], []
    for s in range(args.steps + 2):
        batch = train_items[(s % n_batches) * batch_size:
                            (s % n_batches + 1) * batch_size]
        t0 = time.perf_counter()
        loss = _train_step(student, tokenizer, loss_fn, optimizer, scheduler,
                           batch, premined, corpus_lookup, m, 128, 128, 16, 1.0,
                           device)
        dt = time.perf_counter() - t0
        if s >= 2 and loss is not None:   # drop warmup
            times.append(dt)
            losses.append(loss)

    ok = (len(times) > 0 and all(np.isfinite(t) and t > 0 for t in times)
          and all(np.isfinite(l) for l in losses))
    stats = _percentiles(times)
    spts = stats['seconds_per_train_step_median']
    sph = 3600.0 / spts if spts > 0 else 0.0
    print(f"  timed steps          : {len(times)}")
    print(f"  seconds_per_train_step (median) : {spts:.6f} s (mock, meaningless)")
    print(f"  steps_per_hour       : {sph:,.0f} (mock)")
    print(f"  finite times + loss  : {ok}")
    print(f"  steps_per_hour finite: {np.isfinite(sph) and sph > 0}")
    print("=" * 66)
    print(f"  {'PASS' if ok else 'FAIL'}  trainer-timing harness runs end to end")
    print("=" * 66)
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--synthetic', action='store_true',
                    help='CPU mock-model smoke (no real numbers)')
    ap.add_argument('--steps', type=int, default=500,
                    help='timed optimizer steps (architecture doc: 500-1000)')
    ap.add_argument('--warmup_steps', type=int, default=20,
                    help='untimed warmup steps (compile trace + cudnn autotune)')
    ap.add_argument('--batch_size', type=int, default=None,
                    help='override config batch_size')
    ap.add_argument('--m', type=int, default=1,
                    help='negatives per query in the pre-mined batch (default 1)')
    ap.add_argument('--max_queries', type=int, default=None,
                    help='subset the mixture (tiny real sanity run, e.g. 128)')
    ap.add_argument('--no_compile', action='store_true',
                    help='disable torch.compile (debug / non-Triton env)')
    ap.add_argument('--no_checkpoint_probe', action='store_true',
                    help='skip the one-off checkpoint-write timing probe (the '
                         'speed estimate then excludes async checkpoint I/O)')
    ap.add_argument('--checkpoint_probe_dir', type=str, default=None,
                    help='where to write the throwaway probe checkpoint '
                         '(default: analysis/async_fast_grass_timing/ckpt_probe)')
    ap.add_argument('--premined_data_dir', type=str, default=None,
                    help='dir of real mined *.jsonl (neg_docids / negative_passages) '
                         'to time on instead of uniform-random stand-in negatives; '
                         'missing queries are back-filled with stand-ins')
    args = ap.parse_args()
    return run_synthetic(args) if args.synthetic else run_real(args)


if __name__ == "__main__":
    sys.exit(main())
