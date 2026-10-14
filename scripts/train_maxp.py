"""MaxP training: the in-batch recipe, with documents scored by their best passage.

ANCE's NLL_MultiChunk (model/models.py:84) on BGE-M3. Every document in a batch -- each
row's positive, its supplied negative, and all of them as in-batch candidates for every
other query -- is its first `model.max_passages` passages (utils/maxp.split_passages)
and scores, per query, as its best passage. The temperature-scaled contrastive loss is
taken over those document scores. A long positive is ONE training row again, not one
row per fragment: a footer passage is never asserted relevant on its own.

Own loop, not Tevatron's driver: its collator and EncoderModel.forward hard-code one
vector per passage. The rows still come from Tevatron's TrainDataset, so this arm samples
exactly what the in-batch arm samples. GradCache bounds memory, because 128 documents
carry between 128 and 512 passages; its RandContext replays the dropout RNG.

    python scripts/train_maxp.py --preflight    # CPU, no writes: data and passage counts
    python scripts/train_maxp.py --benchmark    # GPU: s/step, peak memory, projected hours
    python scripts/train_maxp.py [--overwrite]  # the run
"""
import os
import sys
import math
import time
import hashlib
import argparse
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch.utils.data import DataLoader
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
    'logging_steps', 'gradient_checkpointing', 'gc_q_chunk_size', 'gc_p_chunk_size',
)
MAX_HOURS = 22.0    # the 24h wall, less probes, the final save and startup


def _digest(text):
    return hashlib.md5(text.strip().encode()).hexdigest()


def load_train_data(mixture_dir, group_size, seed):
    """Tevatron's TrainDataset -- the in-batch arm's own sampler -- and known positives.

    Its __getitem__ reads the epoch and seed from a trainer; this loop has none, so it
    gets those two fields, and sets `trainer.state.epoch` before each epoch's DataLoader
    is built (worker copies carry it). Known positives are keyed by query TEXT across the
    mixture, not by docid: the mixture holds raw ids that the derived qrels remap.
    """
    from tevatron.retriever.dataset import TrainDataset
    from tevatron.retriever.arguments import DataArguments
    data = TrainDataset(DataArguments(dataset_name='json', train_group_size=group_size,
                                      dataset_path=str(Path(mixture_dir) / '*.jsonl')))
    data.set_trainer(SimpleNamespace(state=SimpleNamespace(epoch=0),
                                     args=SimpleNamespace(seed=seed)))
    known = {}
    for query, positives in zip(data.train_data['query'],
                                data.train_data['positive_passages']):
        known.setdefault(query, set()).update(_digest(doc_text(p)) for p in positives)
    return data, known


def make_collate(tokenizer, max_q, max_p, max_passages, known):
    """A batch -> queries, flat passages sorted by length, and the document geometry.

    Row i's positive is document i*G. `mask[i, j]` is True when document j is a known
    positive of query i other than its target -- a duplicate of its own positive, or
    another positive of the same query -- so it leaves the denominator.
    """
    def collate(batch):
        queries = [q[0] for q, _ in batch]
        docs = [d[0] for _, group in batch for d in group]
        group = len(batch[0][1])
        split = split_passages(tokenizer, docs, max_p, max_passages)
        flat = sorted(((j, p) for j, ps in enumerate(split) for p in ps),
                      key=lambda t: len(t[1]))         # short chunks pad short (_split_trimmed)
        passage = tokenizer.pad({'input_ids': [p.tolist() for _, p in flat]},
                                return_tensors='pt')
        query = tokenizer(queries, truncation=True, max_length=max_q,
                          padding=True, return_tensors='pt')
        digests = [_digest(d) for d in docs]
        target = [i * group for i in range(len(queries))]
        mask = torch.tensor([[j != target[i] and digests[j] in known[q]
                              for j in range(len(docs))] for i, q in enumerate(queries)])
        return {'query': dict(query), 'passage': dict(passage),
                'doc_index': torch.tensor([j for j, _ in flat]),
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
    """One GradCache step; finite checks before the optimizer moves anything. GradCache
    runs its own backward, so run_ance_train.optimization_step cannot be reused."""
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


def preflight(data, tokenizer, max_p, max_passages, batch_size):
    """CPU, no writes: every epoch-0 document splits, and what it costs in passages."""
    pos, per_doc = [], []
    for start in range(0, len(data), 2048):
        items = [data[i] for i in range(start, min(start + 2048, len(data)))]
        split = split_passages(tokenizer, [d[0] for _, g in items for d in g],
                               max_p, max_passages)
        per_doc += [len(ps) for ps in split]
        pos += [len(ps) for ps in split[::len(items[0][1])]]     # each row's first doc
    print(f"[preflight] rows: {len(data):,}")
    print(f"[preflight] positives at the {max_passages}-passage cap: "
          f"{np.mean(np.asarray(pos) == max_passages):.4%}")
    print(f"[preflight] passages per step: {np.mean(per_doc) * 2 * batch_size:.0f} "
          f"(FirstP {2 * batch_size}, cap {max_passages * 2 * batch_size})")
    print("[preflight] ✅ every document splits. Nothing was written.")


def benchmark(n_steps, data, step_args, recipe, total_steps):
    """Timed real steps and their peak memory -> GO / NO-GO. Writes nothing."""
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
    hours = float(np.mean(times)) * total_steps / 3600
    print(f"[benchmark] {len(times)} timed steps: {np.mean(times):.2f} s/step, "
          f"{np.mean(counts):.0f} passages/step (max {max(counts)}), "
          f"peak {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB")
    print(f"[benchmark] projected {total_steps:,} steps: {hours:.1f} h "
          f"(limit {MAX_HOURS} h) -> {'GO' if hours < MAX_HOURS else 'NO-GO'}", flush=True)
    if hours >= MAX_HOURS:
        sys.exit(1)


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

    mixture_dir = Path(ctx['processed_dir']) / "training_mixture"
    mixture_files = require_mixture_files(mixture_dir, MIXTURE_FILES)
    tokenizer = AutoTokenizer.from_pretrained(ctx['base_model'])
    data, known = load_train_data(mixture_dir, recipe['train_group_size'], seed)
    batch_size, epochs = recipe['batch_size'], recipe['num_epochs']
    steps_per_epoch = math.ceil(len(data) / batch_size)
    total_steps = steps_per_epoch * epochs
    print(f"[MaxP] {len(data):,} rows | {steps_per_epoch:,} steps/epoch | "
          f"{total_steps:,} steps | {max_passages} x {ctx['max_p']} passages/doc", flush=True)

    if args.preflight:
        preflight(data, tokenizer, ctx['max_p'], max_passages, batch_size)
        return

    collate = make_collate(tokenizer, ctx['max_q'], ctx['max_p'], max_passages, known)
    generator = torch.Generator()
    generator.manual_seed(seed)

    def loader():
        return DataLoader(data, batch_size=batch_size, shuffle=True, generator=generator,
                          num_workers=recipe['dataloader_num_workers'], collate_fn=collate)

    model = build_model(ctx, bool(recipe['bf16']), recipe['gradient_checkpointing']).cuda()
    model.train()
    optimizer, _ = build_adamw(model.parameters(), lr=recipe['learning_rate'],
                               weight_decay=recipe['weight_decay'], label="maxp")
    scheduler = get_linear_schedule_with_warmup(
        optimizer, math.ceil(recipe['warmup_ratio'] * total_steps), total_steps)
    step_args = (make_grad_cache(model, recipe, ctx['temperature']), model, optimizer, scheduler)

    if args.benchmark:
        benchmark(args.benchmark_steps, loader(), step_args, recipe, total_steps)
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
    step, t0 = 0, time.monotonic()
    for epoch in range(epochs):
        data.trainer.state.epoch = epoch
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
    probe("end", step)
    # No intermediate checkpoints: there is no resume, so nothing would read them.
    model.save(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    assert_training_succeeded(output_dir, manifest)
    print(f"✅ MaxP training completed. Model saved to: {output_dir}", flush=True)


if __name__ == "__main__":
    main()
