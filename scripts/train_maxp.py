"""MaxP training: the in-batch recipe, with documents scored by their best passage.

ANCE's NLL_MultiChunk (model/models.py:84) on BGE-M3. Every document in a batch -- each
row's positive, its supplied negative, and all of them as in-batch candidates for every
other query -- is its first `model.max_passages` passages (utils/maxp.split_passages)
and scores, per query, as its best passage. The temperature-scaled contrastive loss is
taken over those document scores. A long positive is ONE training row again, not one
row per fragment: a footer passage is never asserted relevant on its own.

Own loop, not Tevatron's driver: its collator and EncoderModel.forward hard-code one
vector per passage. GradCache bounds memory, because 128 documents carry between 128
and 512 passages; its RandContext replays the dropout RNG between its two passes.

    python scripts/train_maxp.py --preflight    # CPU, no writes: data and passage counts
    python scripts/train_maxp.py --benchmark    # GPU: s/step, peak memory, projected hours
    python scripts/train_maxp.py [--overwrite]  # the run
"""
import os
import sys
import json
import math
import time
import random
import hashlib
import argparse
import tempfile
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root / 'src'))
sys.path.append(str(project_root / 'scripts'))

from utils.helpers import (get_training_context, load_config, set_seed, log_startup_config,
                           build_run_manifest, prepare_output_dir, assert_training_succeeded,
                           probe_triples_from_mixture, ranking_probe, require_recipe_keys,
                           build_adamw, append_jsonl, TRAINING_LOG_NAME)
from utils.maxp import doc_text, split_passages, maxp_loss
from data.preprocessor import MIXTURE_FILES, require_mixture_files
from run_ance_train import NonFiniteOptimization

os.environ["TRANSFORMERS_ATTENTION_IMPLEMENTATION"] = "eager"

CONSUMED_KEYS = (
    'model_name', 'batch_size', 'learning_rate', 'num_epochs', 'train_group_size',
    'bf16', 'dataloader_num_workers', 'warmup_ratio', 'weight_decay', 'max_grad_norm',
    'logging_steps', 'save_fraction', 'gradient_checkpointing',
    'gc_q_chunk_size', 'gc_p_chunk_size',
)
MAX_HOURS = 22.0    # the 24h wall, less probes, saves and startup


def _digest(text):
    return hashlib.md5(text.strip().encode()).hexdigest()


class MaxPDataset(Dataset):
    """Mixture rows -> (query, [positive, negatives...], known positive digests).

    Positive and negative choice copy Tevatron's TrainDataset.__getitem__, so this arm
    sees the documents the in-batch arm sees: a row's negatives are shuffled with a
    per-row seed and each epoch takes the next window. `epoch` is set by the loop
    before each epoch's DataLoader is built, so worker copies carry it.

    Known positives are keyed by query TEXT across the whole mixture, not by docid:
    the mixture holds raw ids that the derived qrels remap.
    """

    def __init__(self, files, group_size, seed):
        self.rows, self.known = [], {}
        for path in files:
            with open(path, encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        row = json.loads(line)
                        self.rows.append(row)
                        self.known.setdefault(row['query'], set()).update(
                            _digest(doc_text(p)) for p in row['positive_passages'])
        self.group_size, self.seed, self.epoch = group_size, seed, 0

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, item):
        row, n_neg = self.rows[item], self.group_size - 1
        hashed_seed = hash(item + self.seed)
        positives, negatives = row['positive_passages'], row['negative_passages']
        docs = [positives[(hashed_seed + self.epoch) % len(positives)]]
        if len(negatives) < n_neg:
            docs += random.choices(negatives, k=n_neg)
        elif n_neg:
            shuffled = list(negatives)
            random.Random(hashed_seed).shuffle(shuffled)
            offset = self.epoch * n_neg % len(negatives)
            docs += (shuffled * 2)[offset:offset + n_neg]
        return row['query'], [doc_text(d) for d in docs], self.known[row['query']]


def make_collate(tokenizer, max_q, max_p, max_passages):
    """A batch -> queries, flat passages sorted by length, and the document geometry.

    Row i's positive is document i*G. `mask[i, j]` is True when document j is a known
    positive of query i other than its target -- a duplicate of its own positive, or
    another positive of the same query -- so it leaves the denominator.
    """
    def collate(batch):
        queries, groups, known = zip(*batch)
        docs = [d for g in groups for d in g]
        group = len(groups[0])
        split = split_passages(tokenizer, docs, max_p, max_passages)
        flat = [(len(p), j, p) for j, ps in enumerate(split) for p in ps]
        flat.sort(key=lambda t: t[0])          # short chunks pad short (see _split_trimmed)
        passage = tokenizer.pad({'input_ids': [p.tolist() for _, _, p in flat]},
                                return_tensors='pt')
        query = tokenizer(list(queries), truncation=True, max_length=max_q,
                          padding=True, return_tensors='pt')
        digests = [_digest(d) for d in docs]
        target = [i * group for i in range(len(queries))]
        mask = torch.tensor([[j != target[i] and digests[j] in known[i]
                              for j in range(len(docs))] for i in range(len(queries))])
        return {'query': dict(query), 'passage': dict(passage),
                'doc_index': torch.tensor([j for _, j, _ in flat]),
                'target': torch.tensor(target), 'mask': mask}
    return collate


def _split_trimmed(model_input, chunk_size):
    """Tevatron's split_dense_inputs, with each chunk cut to its own longest row.

    Right padding, so trimming drops only pad columns. Passages arrive length-sorted,
    so a chunk of short passages no longer pays for the batch's longest one.
    """
    from tevatron.retriever.gc_trainer import split_dense_inputs
    chunks = split_dense_inputs(model_input, chunk_size)
    for chunk in chunks:
        (key, val), = chunk.items()
        width = int(val['attention_mask'].sum(1).max())
        chunk[key] = {k: v[:, :width] for k, v in val.items()}
    return chunks


def build_model(ctx, bf16, gradient_checkpointing):
    from tevatron.retriever.modeling import DenseModel
    from tevatron.retriever.arguments import (
        ModelArguments, TevatronTrainingArguments as TrainingArguments)
    if not hasattr(DenseModel, "_keys_to_ignore_on_save"):
        setattr(DenseModel, "_keys_to_ignore_on_save", None)
    model_args = ModelArguments(model_name_or_path=ctx['base_model'],
                                pooling=ctx['pooling'], normalize=ctx['normalize'],
                                temperature=ctx['temperature'], attn_implementation='eager')
    # torch_dtype explicitly: EncoderModel.build ignores TrainingArguments.bf16, and the
    # in-batch arm's driver passes it the same way (run_ance_train.py:363-371).
    model = DenseModel.build(model_args, TrainingArguments(output_dir='.', bf16=bf16),
                             attn_implementation='eager',
                             torch_dtype=torch.bfloat16 if bf16 else torch.float32)
    if gradient_checkpointing:
        # Non-reentrant: GradCache runs several forward/backward passes per step
        # (train_crossbatch.py:40-53). On `.encoder`, past Tevatron's broken override.
        model.encoder.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={'use_reentrant': False})
    return model


def make_grad_cache(model, recipe, temperature):
    from grad_cache import GradCache
    from tevatron.retriever.gc_trainer import get_dense_rep
    return GradCache(models=[model, model],
                     chunk_sizes=[recipe['gc_q_chunk_size'], recipe['gc_p_chunk_size']],
                     loss_fn=lambda q, p, **kw: maxp_loss(q, p, temperature=temperature, **kw),
                     split_input_fn=_split_trimmed, get_rep_fn=get_dense_rep, fp16=False)


def train_step(gc, model, optimizer, scheduler, batch, max_grad_norm, step):
    """One GradCache step; finite checks before the optimizer moves anything."""
    batch = {k: ({n: t.cuda() for n, t in v.items()} if isinstance(v, dict) else v.cuda())
             for k, v in batch.items()}
    loss = float(gc({'query': batch['query']}, {'passage': batch['passage']},
                    doc_index=batch['doc_index'], target=batch['target'], mask=batch['mask']))
    if not math.isfinite(loss):
        raise NonFiniteOptimization(f"non-finite loss={loss} at step {step}; no step taken.")
    try:
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm,
                                                   error_if_nonfinite=True)
    except RuntimeError as exc:
        raise NonFiniteOptimization(f"non-finite gradient norm at step {step}: {exc}") from exc
    optimizer.step()
    scheduler.step()
    optimizer.zero_grad(set_to_none=True)
    return loss, float(grad_norm)


def preflight(dataset, tokenizer, max_p, max_passages, batch_size):
    """CPU, no writes: every epoch-0 document splits, and how many passages it costs."""
    counts = {'positive': [], 'negative': []}
    full = max_passages * (max_p - tokenizer.num_special_tokens_to_add())
    over = 0
    for start in range(0, len(dataset), 2048):
        items = [dataset[i] for i in range(start, min(start + 2048, len(dataset)))]
        pos = [g[0] for _, g, _ in items]
        neg = [d for _, g, _ in items for d in g[1:]]
        counts['positive'] += [len(p) for p in split_passages(tokenizer, pos, max_p, max_passages)]
        counts['negative'] += [len(p) for p in split_passages(tokenizer, neg, max_p, max_passages)]
        over += sum(len(ids) > full for ids in
                    tokenizer(pos, add_special_tokens=False, verbose=False)['input_ids'])
    print(f"[preflight] rows: {len(dataset):,}")
    for kind, c in counts.items():
        c = np.asarray(c)
        print(f"[preflight] {kind}: mean {c.mean():.3f} passages/doc; "
              + ", ".join(f"{n}: {(c == n).mean():.2%}" for n in range(1, max_passages + 1)))
    per_doc = np.concatenate([counts['positive'], counts['negative']]).mean()
    print(f"[preflight] positives fully inside {max_passages} passages: "
          f"{1 - over / len(dataset):.4%}")
    print(f"[preflight] projected passages per step: {per_doc * 2 * batch_size:.0f} "
          f"(FirstP: {2 * batch_size}); worst case {max_passages * 2 * batch_size}")
    print("[preflight] ✅ every document splits. Nothing was written.")


def main():
    parser = argparse.ArgumentParser(description="MaxP in-batch training.")
    parser.add_argument('--preflight', action='store_true')
    parser.add_argument('--benchmark', action='store_true')
    parser.add_argument('--benchmark_steps', type=int, default=200)
    parser.add_argument('--overwrite', action='store_true',
                        help="Discard an output dir produced by a DIFFERENT configuration.")
    args = parser.parse_args()

    config = load_config()
    seed = config.get('seed', 42)
    set_seed(seed)
    ctx = get_training_context("maxp")
    recipe = ctx['args']
    require_recipe_keys("maxp", recipe, CONSUMED_KEYS)
    log_startup_config("maxp", ctx)
    max_passages = config['model']['max_passages']

    mixture_files = require_mixture_files(Path(ctx['processed_dir']) / "training_mixture",
                                          MIXTURE_FILES)
    tokenizer = AutoTokenizer.from_pretrained(ctx['base_model'])
    dataset = MaxPDataset(mixture_files, recipe['train_group_size'], seed)
    batch_size, epochs = recipe['batch_size'], recipe['num_epochs']
    steps_per_epoch = math.ceil(len(dataset) / batch_size)
    total_steps = steps_per_epoch * epochs
    save_steps = max(1, int(total_steps * recipe['save_fraction']))
    print(f"[MaxP] {len(dataset):,} rows | {steps_per_epoch:,} steps/epoch | "
          f"{total_steps:,} steps | {max_passages} x {ctx['max_p']} passages/doc", flush=True)

    if args.preflight:
        preflight(dataset, tokenizer, ctx['max_p'], max_passages, batch_size)
        return

    collate = make_collate(tokenizer, ctx['max_q'], ctx['max_p'], max_passages)
    generator = torch.Generator()
    generator.manual_seed(seed)

    def loader():
        return DataLoader(dataset, batch_size=batch_size, shuffle=True, generator=generator,
                          num_workers=recipe['dataloader_num_workers'], collate_fn=collate)

    model = build_model(ctx, bool(recipe['bf16']), recipe['gradient_checkpointing']).cuda()
    model.train()
    optimizer, _ = build_adamw(model.parameters(), lr=recipe['learning_rate'],
                               weight_decay=recipe['weight_decay'], label="maxp")
    scheduler = get_linear_schedule_with_warmup(
        optimizer, math.ceil(recipe['warmup_ratio'] * total_steps), total_steps)
    gc = make_grad_cache(model, recipe, ctx['temperature'])
    step_args = (gc, model, optimizer, scheduler)

    if args.benchmark:
        benchmark(args.benchmark_steps, loader(), step_args, recipe, tokenizer, ctx,
                  max_passages, total_steps, save_steps)
        return

    world_size = int(os.environ.get('WORLD_SIZE', '1'))
    manifest = build_run_manifest(
        "maxp", ctx, recipe, data_files=mixture_files, world_size=world_size,
        negative_pool_size=batch_size * world_size * recipe['train_group_size'] - 1,
        optimizer_steps=total_steps)
    output_dir = prepare_output_dir(ctx['output_dir'], manifest, overwrite=args.overwrite)
    log_path = output_dir / TRAINING_LOG_NAME
    triples = probe_triples_from_mixture(mixture_files)

    def probe(phase, step):
        """Best-effort and recorded either way, as in run_ance_train._probe."""
        try:
            result = ranking_probe(model, tokenizer, triples, torch.device('cuda'),
                                   ctx['max_q'], ctx['max_p'], normalize=ctx['normalize'])
        except Exception as e:                                        # noqa: BLE001
            result = {"error": f"{type(e).__name__}: {e}"}
        append_jsonl(log_path, {"global_step": step, "phase": phase, **result})

    probe("begin", 0)
    step, t0, loss, grad_norm = 0, time.monotonic(), float('nan'), float('nan')
    for epoch in range(epochs):
        dataset.epoch = epoch
        for batch in loader():
            loss, grad_norm = train_step(*step_args, batch, recipe['max_grad_norm'], step)
            step += 1
            if step % recipe['logging_steps'] == 0 or step == total_steps:
                append_jsonl(log_path, {
                    "global_step": step, "epoch": epoch, "loss": loss,
                    "learning_rate": scheduler.get_last_lr()[0], "grad_norm": grad_norm,
                    "passages": len(batch['doc_index']),
                    "train_seconds": round(time.monotonic() - t0, 1)})
                print(f"[MaxP] step={step}/{total_steps} loss={loss:.4f} "
                      f"passages={len(batch['doc_index'])} "
                      f"train_s={time.monotonic() - t0:.0f}", flush=True)
            if step == total_steps // 2:
                probe("mid", step)
            if step % save_steps == 0 and step < total_steps:
                save_checkpoint(model, tokenizer, optimizer, scheduler,
                                output_dir / f"checkpoint-{step}")
    probe("end", step)
    model.save(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    assert_training_succeeded(output_dir, manifest)
    print(f"✅ MaxP training completed. Model saved to: {output_dir}", flush=True)


def save_checkpoint(model, tokenizer, optimizer, scheduler, path):
    path.mkdir(parents=True, exist_ok=True)
    model.save(str(path))
    tokenizer.save_pretrained(str(path))
    torch.save(scheduler.state_dict(), path / "scheduler.pt")
    torch.save(optimizer.state_dict(), path / "optimizer.pt")   # LAST: the validity flag


def benchmark(n_steps, data, step_args, recipe, tokenizer, ctx, max_passages,
              total_steps, save_steps):
    """Real steps, one synthetic worst-case batch and one save -> GO / NO-GO.

    Nothing lands in the output dir; the save goes to node-local scratch.
    """
    model, optimizer, scheduler = step_args[1], step_args[2], step_args[3]
    warm, times, counts = 20, [], []
    torch.cuda.reset_peak_memory_stats()
    for i, batch in zip(range(n_steps), data):
        torch.cuda.synchronize()
        t = time.monotonic()
        train_step(*step_args, batch, recipe['max_grad_norm'], i)
        torch.cuda.synchronize()
        if i >= warm:
            times.append(time.monotonic() - t)
            counts.append(len(batch['doc_index']))
    real_peak = torch.cuda.max_memory_allocated() / 2**30

    # Worst case: every query at the cap and every document at max_passages full passages.
    b, g = recipe['batch_size'], recipe['train_group_size']
    ids = lambda n, length: {'input_ids': torch.full((n, length), 1000),
                             'attention_mask': torch.ones(n, length, dtype=torch.long)}
    worst = {'query': ids(b, ctx['max_q']), 'passage': ids(b * g * max_passages, ctx['max_p']),
             'doc_index': torch.arange(b * g).repeat_interleave(max_passages),
             'target': torch.arange(b) * g, 'mask': torch.zeros(b, b * g, dtype=torch.bool)}
    torch.cuda.reset_peak_memory_stats()
    t = time.monotonic()
    train_step(*step_args, worst, recipe['max_grad_norm'], n_steps)
    torch.cuda.synchronize()
    worst_s, worst_peak = time.monotonic() - t, torch.cuda.max_memory_allocated() / 2**30

    with tempfile.TemporaryDirectory() as tmp:
        t = time.monotonic()
        save_checkpoint(model, tokenizer, optimizer, scheduler, Path(tmp) / "checkpoint")
        save_s = time.monotonic() - t

    per_step = float(np.mean(times))
    hours = (per_step * total_steps + save_s * (total_steps // save_steps)) / 3600
    print(f"[benchmark] {len(times)} timed steps: {per_step:.2f} s/step "
          f"(p95 {np.percentile(times, 95):.2f}), {np.mean(counts):.0f} passages/step "
          f"(max {max(counts)}), peak {real_peak:.1f} GiB")
    print(f"[benchmark] worst case {b * g * max_passages} passages: {worst_s:.1f} s, "
          f"peak {worst_peak:.1f} GiB")
    print(f"[benchmark] checkpoint save {save_s:.0f} s")
    print(f"[benchmark] projected {total_steps:,} steps: {hours:.1f} h "
          f"(limit {MAX_HOURS} h) -> {'GO' if hours < MAX_HOURS else 'NO-GO'}", flush=True)
    if hours >= MAX_HOURS:
        sys.exit(1)


if __name__ == "__main__":
    main()
