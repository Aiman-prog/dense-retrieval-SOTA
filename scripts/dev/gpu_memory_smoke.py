"""Peak memory and throughput for the BGE p1024 shapes, on one A100.

Runs a handful of real forward/backward steps per arm and reports
`torch.cuda.max_memory_allocated()` and steps/s. It answers one preflight question:
whether the configured q1024/p1024 batches fit in 80 GiB, for training and for
encoding.

Two arms, not five. The in-batch and BGE-ANCE training shapes are the same grouped
contrastive step at the same batch and group, so one probe covers both. The old
`crossbatch` arm ran that same shape at the GradCache chunk size without ever
invoking GradCache, so it measured a chunk-sized forward rather than the arm. The
`ance_paper` arm was RoBERTa-base at q64/p512, which is not where memory goes; the
paper encoder's real allocation risk is the 27 GiB corpus embedding array, and that
is asserted structurally in `tests/ance_paper_test.py` where it costs no GPU hour.

Synthetic text at the configured caps, not the real mixture: the question is whether
the shape fits in 80GB and how fast a step is, and building that answer must not
depend on /scratch artifacts being present. Passage length is sampled around the
measured mixture median, with one cap-length item forced into every batch. Dynamic
padding makes that the conservative peak a real long-tail batch must survive.

    python scripts/dev/gpu_memory_smoke.py                 # both arms
    python scripts/dev/gpu_memory_smoke.py --arms bge_train
"""
import argparse
import sys
import time
from pathlib import Path

import torch

project_root = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(project_root / 'src'))
sys.path.append(str(project_root / 'scripts'))

from utils.helpers import get_training_context, load_config       # noqa: E402

ARMS = ('bge_train', 'bge_encode')
# Measured on the 329,993-row training mixture and the 648,942-row ReasonIR corpus
# (review §7.2/7.3): P50 41 query / 114 passage word-pieces. Padding to the cap is
# what these runs must NOT do, so the synthetic lengths have to vary.
MEDIAN_Q, MEDIAN_P = 41, 114


def _report(label, steps, elapsed, note=""):
    peak = torch.cuda.max_memory_allocated() / 2 ** 30
    print(f"  {label:<22} peak {peak:6.2f} GiB | {steps / elapsed:5.2f} steps/s | "
          f"{elapsed / steps * 1000:7.1f} ms/step {note}", flush=True)
    return peak, steps / elapsed


def _texts(n, tokens):
    """`tokens` whitespace words -- roughly one word-piece each for this purpose."""
    return [" ".join(["passage"] * tokens) for _ in range(n)]


def _varied(n, median, cap):
    """A median-like batch with one cap-length tail item for peak-memory safety."""
    import random
    rng = random.Random(42)
    lengths = [min(cap, max(8, int(rng.gauss(median, median)))) for _ in range(n)]
    if lengths:
        lengths[0] = cap
    return [" ".join(["token"] * length) for length in lengths]


def fail_if_any(failures):
    if failures:
        print(f"FAILED arms: {[arm for arm, _, _ in failures]}", flush=True)
        raise SystemExit(1)


def run_bge_group(label, ctx, batch_size, group, steps, gradient_checkpointing):
    """One grouped contrastive step: the shape in-batch, cross-batch and BGE ANCE share."""
    from transformers import AutoTokenizer
    from tevatron.retriever.modeling import DenseModel
    from tevatron.retriever.arguments import (
        ModelArguments, TevatronTrainingArguments as TrainingArguments)

    if not hasattr(DenseModel, "_keys_to_ignore_on_save"):
        setattr(DenseModel, "_keys_to_ignore_on_save", None)
    tok = AutoTokenizer.from_pretrained(ctx['base_model'])
    model = DenseModel.build(
        ModelArguments(model_name_or_path=ctx['base_model'], pooling=ctx['pooling'],
                       normalize=ctx['normalize'], temperature=ctx['temperature'],
                       attn_implementation='eager'),
        TrainingArguments(output_dir='/tmp/smoke', bf16=True),
        attn_implementation='eager').cuda()
    if gradient_checkpointing:
        model.encoder.gradient_checkpointing_enable()
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-5)

    q = tok(_varied(batch_size, MEDIAN_Q, ctx['max_q']), padding=True, truncation=True,
            max_length=ctx['max_q'], return_tensors='pt')
    p = tok(_varied(batch_size * group, MEDIAN_P, ctx['max_p']), padding=True,
            truncation=True, max_length=ctx['max_p'], return_tensors='pt')
    q = {k: v.cuda() for k, v in q.items()}
    p = {k: v.cuda() for k, v in p.items()}

    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    for _ in range(steps):
        with torch.autocast('cuda', dtype=torch.bfloat16):
            loss = model(query=q, passage=p).loss
        loss.backward()
        opt.step()
        opt.zero_grad(set_to_none=True)
    torch.cuda.synchronize()
    return _report(label, steps, time.time() - t0,
                   f"(q{q['input_ids'].shape[1]}/p{p['input_ids'].shape[1]} padded)")


def run_encode(label, ctx, batch_size, batches):
    from transformers import AutoModel, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(ctx['base_model'])
    model = AutoModel.from_pretrained(ctx['base_model'],
                                      attn_implementation='eager').cuda().eval()
    batch = tok(_texts(batch_size, ctx['max_p']), padding=True, truncation=True,
                max_length=ctx['max_p'], return_tensors='pt')
    batch = {k: v.cuda() for k, v in batch.items()}
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    with torch.no_grad():
        for _ in range(batches):
            with torch.autocast('cuda', dtype=torch.bfloat16):
                model(**batch)
    torch.cuda.synchronize()
    elapsed = time.time() - t0
    peak, rate = _report(label, batches, elapsed,
                         f"= {batch_size * batches / elapsed:.0f} docs/s at the p"
                         f"{ctx['max_p']} CAP (worst case)")
    print(f"  {'':22} -> 648,942 BRIGHT docs would take "
          f"{648942 / (batch_size * batches / elapsed) / 60:.0f} min at that rate; "
          f"set max_encode_seconds well above it", flush=True)
    return peak, rate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--arms', nargs='*', default=list(ARMS), choices=list(ARMS))
    parser.add_argument('--steps', type=int, default=8)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("no CUDA device; this probe only means something on the A100")
    config = load_config()
    print(f"device: {torch.cuda.get_device_name(0)} | "
          f"q{config['model']['query_max_len']}/p{config['model']['passage_max_len']}\n")

    failures = []
    for arm in args.arms:
        print(f"[{arm}]", flush=True)
        try:
            ctx = get_training_context('ance')
            if arm == 'bge_train':
                # batch 64 / group 2 at q1024/p1024 -- the shape in-batch and BGE ANCE
                # both train, and the one gradient checkpointing exists to make fit.
                run_bge_group("batch 64, group 2", ctx, 64, 2, args.steps,
                              ctx['args']['gradient_checkpointing'])
            elif arm == 'bge_encode':
                run_encode(f"eval batch "
                           f"{ctx['args']['per_device_eval_batch_size']}", ctx,
                           ctx['args']['per_device_eval_batch_size'], args.steps)
        except Exception as exc:                                   # noqa: BLE001
            # Report and continue: an OOM in one arm is a measurement, and the other
            # arms still need measuring in the same allocation.
            failures.append((arm, type(exc).__name__, str(exc)))
            print(f"  FAILED: {type(exc).__name__}: {exc}", flush=True)
        finally:
            torch.cuda.empty_cache()
        print(flush=True)

    fail_if_any(failures)


if __name__ == "__main__":
    main()
