"""MaxP training for the in-batch and cross-batch arms.

Every document in a batch -- each row's positive, its supplied negative, and all of
them as in-batch candidates for every query -- is its first `model.max_passages`
passages and scores, per query, as its best passage (`maxp_training`). A long positive
is ONE training row: a footer passage is never asserted relevant on its own.

    --recipe maxp             in-batch: 64 queries on one GPU
    --recipe crossbatch_maxp  cross-batch: torchrun --nproc_per_node 2, 2 x 512 queries
                              pooled into one 1,024-query softmax across ranks

Own loop, not Tevatron's driver, whose collator and EncoderModel.forward hard-code one
vector per passage. Rows still come from Tevatron's TrainDataset, so the in-batch arm
samples exactly what Tevatron's in-batch run sampled.

    python scripts/train_maxp.py --recipe maxp --preflight   # CPU, no writes
    python scripts/train_maxp.py --recipe maxp --benchmark   # GPU: end-to-end s/step
    python scripts/train_maxp.py --recipe maxp               # the run

There is no --overwrite: an occupied output directory is refused, never cleared.
"""
import os
import re
import sys
import math
import time
import argparse
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root / 'src'))
sys.path.append(str(project_root / 'scripts'))

from utils.helpers import (get_training_context, load_config, set_seed, log_startup_config,
                           build_run_manifest, prepare_output_dir, assert_training_succeeded,
                           probe_triples_from_mixture, ranking_probe, require_recipe_keys,
                           build_adamw, append_jsonl, require_training_data,
                           _load_corpus_lookup, _load_qrels, TRAINING_LOG_NAME)
from utils.maxp import split_passages, representation_contract
from data.preprocessor import MIXTURE_FILES, CORPUS_FILE, QRELS_FILE, require_mixture_files
from maxp_training import (ClsEncoder, known_positives, make_collate, make_grad_cache,
                           train_step, write_contract, CHUNK_ROW_SUFFIX)

os.environ["TRANSFORMERS_ATTENTION_IMPLEMENTATION"] = "eager"

CONSUMED_KEYS = (
    'model_name', 'learning_rate', 'num_epochs', 'train_group_size', 'bf16',
    'dataloader_num_workers', 'warmup_ratio', 'weight_decay', 'max_grad_norm',
    'logging_steps', 'save_fraction', 'gradient_checkpointing',
    'gc_q_chunk_size', 'gc_p_chunk_size',
)
OPTIONAL_KEYS = ('batch_size', 'per_device_batch_size', 'world_size')
MAX_HOURS = 22.0    # the 24h wall, less probes, snapshots, the final save and startup


def load_train_data(mixture_dir, group_size, seed):
    """Tevatron's TrainDataset -- the in-batch arm's own sampler.

    Its __getitem__ reads the epoch and seed from a trainer; this loop has none, so it
    gets those two fields and sets `trainer.state.epoch` before each epoch's DataLoader
    is built (worker copies carry it). A chunk-row id is refused here, before a model
    is loaded: the rolled-back split must never train again.
    """
    from tevatron.retriever.dataset import TrainDataset
    from tevatron.retriever.arguments import DataArguments
    data = TrainDataset(DataArguments(dataset_name='json', train_group_size=group_size,
                                      dataset_path=str(Path(mixture_dir) / '*.jsonl')))
    suffix = re.compile(CHUNK_ROW_SUFFIX)
    chunked = [q for q in data.train_data['query_id'] if suffix.search(str(q))]
    if chunked:
        raise ValueError(f"{len(chunked)} chunk-row id(s) in {mixture_dir}, e.g. "
                         f"{chunked[:3]}: that is the rolled-back split, not the original")
    data.set_trainer(SimpleNamespace(state=SimpleNamespace(epoch=0),
                                     args=SimpleNamespace(seed=seed)))
    return data


def build_model(ctx, bf16, gradient_checkpointing):
    """DenseModel (what `save()` writes and evaluation loads) around the HF encoder."""
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
        # Non-reentrant: GradCache runs several forward/backward passes per step, and
        # under DDP reentrant checkpointing marks parameters ready twice
        # (train_crossbatch.py:40-53). On `.encoder`, past Tevatron's broken override.
        model.encoder.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={'use_reentrant': False})
    return model


def preflight(data, tokenizer, max_p, max_passages, batch_size):
    """CPU, no writes: every epoch-0 document splits, and what it costs in passages."""
    pos, per_doc = [], []
    for start in range(0, len(data), 2048):
        items = [data[i] for i in range(start, min(start + 2048, len(data)))]
        split = split_passages(tokenizer, [d[0] for _, g in items for d in g],
                               max_p, max_passages)
        per_doc += [len(ps) for ps in split]
        pos += [len(ps) for ps in split[::len(items[0][1])]]     # each row's first doc
    print(f"[preflight] rows: {len(data):,} (0 chunk-row ids)")
    print(f"[preflight] positives at the {max_passages}-passage cap: "
          f"{np.mean(np.asarray(pos) == max_passages):.4%}")
    print(f"[preflight] passages per step: {np.mean(per_doc) * 2 * batch_size:.0f} "
          f"(FirstP {2 * batch_size}, cap {max_passages * 2 * batch_size})")
    print("[preflight] ✅ every document splits. Nothing was written.")


def benchmark(n_steps, loader, step_args, recipe, total_steps, distributed):
    """End-to-end timed steps -> GO / NO-GO. The timer includes waiting for the loader;
    probes, snapshots and the final save are inside MAX_HOURS's margin. Writes nothing."""
    warm, times, counts = 20, [], []
    torch.cuda.reset_peak_memory_stats()
    it = iter(loader)
    for i in range(n_steps):
        torch.cuda.synchronize()
        t = time.monotonic()
        batch = next(it)
        train_step(*step_args, batch, recipe['max_grad_norm'], i, distributed=distributed)
        torch.cuda.synchronize()
        if i >= warm:
            times.append(time.monotonic() - t)
            counts.append(len(batch['doc_index']))
    hours = float(np.mean(times)) * total_steps / 3600
    print(f"[benchmark] {len(times)} timed steps: {np.mean(times):.2f} s/step end-to-end, "
          f"{np.mean(counts):.0f} passages/step/rank (max {max(counts)}), "
          f"peak {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB")
    print(f"[benchmark] projected {total_steps:,} steps: {hours:.1f} h "
          f"(limit {MAX_HOURS} h) -> {'GO' if hours < MAX_HOURS else 'NO-GO'}", flush=True)
    return hours < MAX_HOURS


def main():
    parser = argparse.ArgumentParser(description="MaxP in-batch / cross-batch training.")
    parser.add_argument('--recipe', choices=('maxp', 'crossbatch_maxp'), default='maxp')
    parser.add_argument('--preflight', action='store_true')
    parser.add_argument('--benchmark', action='store_true')
    parser.add_argument('--benchmark_steps', type=int, default=200)
    args = parser.parse_args()

    distributed = int(os.environ.get('WORLD_SIZE', '1')) > 1
    if distributed:
        dist.init_process_group('nccl')
        torch.cuda.set_device(int(os.environ['LOCAL_RANK']))
    rank, world = (dist.get_rank(), dist.get_world_size()) if distributed else (0, 1)
    main_rank = rank == 0

    config = load_config()
    seed = config.get('seed', 42)
    set_seed(seed)
    ctx = get_training_context(args.recipe)
    recipe = ctx['args']
    require_recipe_keys(args.recipe, recipe, CONSUMED_KEYS, optional=OPTIONAL_KEYS)
    batch_size = recipe.get('per_device_batch_size') or recipe['batch_size']
    if recipe.get('world_size', 1) != world:
        raise RuntimeError(f"recipe {args.recipe} declares world_size "
                           f"{recipe.get('world_size', 1)}, launched with {world}: the "
                           f"negative pool would not be the declared one")
    if main_rank:
        log_startup_config(args.recipe, ctx)
    max_passages = config['model']['max_passages']

    training_dir = Path(ctx['training_dir'])
    _, data_sha = require_training_data(training_dir)
    mixture_files = require_mixture_files(training_dir / "training_mixture", MIXTURE_FILES)
    contract = representation_contract(config, ctx['base_model'], data_sha256=data_sha)
    tokenizer = AutoTokenizer.from_pretrained(ctx['base_model'])
    data = load_train_data(training_dir / "training_mixture",
                           recipe['train_group_size'], seed)

    if args.preflight:
        preflight(data, tokenizer, ctx['max_p'], max_passages, batch_size)
        return

    known = known_positives(
        ({'query': q, 'query_id': i, 'positive_passages': p} for q, i, p in zip(
            data.train_data['query'], data.train_data['query_id'],
            data.train_data['positive_passages'])),
        _load_qrels(training_dir / QRELS_FILE), _load_corpus_lookup(training_dir / CORPUS_FILE))
    collate = make_collate(tokenizer, ctx['max_q'], ctx['max_p'], max_passages, known)
    workers = recipe['dataloader_num_workers']
    if distributed:
        # drop_last keeps every rank's B equal, which the cross-rank pool requires
        sampler = DistributedSampler(data, shuffle=True, seed=seed, drop_last=True)
        loader = lambda: DataLoader(data, batch_size=batch_size, sampler=sampler,
                                    num_workers=workers, collate_fn=collate, drop_last=True)
    else:
        generator = torch.Generator()
        generator.manual_seed(seed)
        sampler = None
        loader = lambda: DataLoader(data, batch_size=batch_size, shuffle=True,
                                    generator=generator, num_workers=workers,
                                    collate_fn=collate)
    steps_per_epoch = len(loader())
    epochs = recipe['num_epochs']
    total_steps = steps_per_epoch * epochs
    save_steps = max(1, int(total_steps * recipe['save_fraction']))
    if main_rank:
        print(f"[MaxP] {args.recipe}: {len(data):,} rows | {world} rank(s) x {batch_size} "
              f"queries | {steps_per_epoch:,} steps/epoch | {total_steps:,} steps | "
              f"{max_passages} x {ctx['max_p']} passages/doc", flush=True)

    device = torch.device('cuda', torch.cuda.current_device())
    model = build_model(ctx, bool(recipe['bf16']), recipe['gradient_checkpointing'])
    model.to(device)
    encoder = ClsEncoder(model.encoder).train()
    if distributed:
        encoder = torch.nn.parallel.DistributedDataParallel(
            encoder, device_ids=[device.index])
    optimizer, _ = build_adamw(encoder.parameters(), lr=recipe['learning_rate'],
                               weight_decay=recipe['weight_decay'], label=args.recipe)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, math.ceil(recipe['warmup_ratio'] * total_steps), total_steps)
    gc = make_grad_cache(encoder, recipe['gc_q_chunk_size'], recipe['gc_p_chunk_size'],
                         ctx['temperature'], distributed=distributed)
    step_args = (gc, encoder, optimizer, scheduler)

    if args.benchmark:
        if sampler is not None:
            sampler.set_epoch(0)
        ok = benchmark(args.benchmark_steps, loader(), step_args, recipe, total_steps,
                       distributed)
        sys.exit(0 if ok else 1)

    output_dir = Path(ctx['output_dir'])
    if main_rank and output_dir.exists() and any(output_dir.iterdir()):
        raise RuntimeError(f"{output_dir} is not empty; a MaxP run never clears a "
                           f"directory. Move it aside or rename the recipe's model_name.")
    manifest = build_run_manifest(
        args.recipe, ctx, recipe, data_files=mixture_files, world_size=world,
        negative_pool_size=batch_size * world * recipe['train_group_size'] - 1,
        optimizer_steps=total_steps, extra={'representation': contract})
    if main_rank:
        prepare_output_dir(output_dir, manifest)       # empty dir: nothing to clear
        write_contract(output_dir, contract)
    log_path = output_dir / TRAINING_LOG_NAME
    triples = probe_triples_from_mixture(mixture_files)

    def probe(phase, step):
        """Best-effort and recorded either way, as in run_ance_train._probe."""
        if not main_rank:
            return
        try:
            result = ranking_probe(model, tokenizer, triples, device, ctx['max_q'],
                                   ctx['max_p'], normalize=ctx['normalize'])
        except Exception as e:                                        # noqa: BLE001
            result = {"error": f"{type(e).__name__}: {e}"}
        append_jsonl(log_path, {"global_step": step, "phase": phase, **result})
        encoder.train()

    probe("begin", 0)
    step, t0 = 0, time.monotonic()
    for epoch in range(epochs):
        data.trainer.state.epoch = epoch
        if sampler is not None:
            sampler.set_epoch(epoch)
        for batch in loader():
            loss, grad_norm = train_step(*step_args, batch, recipe['max_grad_norm'], step,
                                         distributed=distributed)
            step += 1
            if main_rank and (step % recipe['logging_steps'] == 0 or step == total_steps):
                masked = int(batch['mask'].sum())
                append_jsonl(log_path, {
                    "global_step": step, "epoch": epoch, "loss": loss,
                    "learning_rate": scheduler.get_last_lr()[0], "grad_norm": grad_norm,
                    "passages": len(batch['doc_index']), "masked_candidates": masked,
                    "train_seconds": round(time.monotonic() - t0, 1)})
                print(f"[MaxP] step={step}/{total_steps} loss={loss:.4f} "
                      f"passages={len(batch['doc_index'])} masked={masked} "
                      f"train_s={time.monotonic() - t0:.0f}", flush=True)
            if step == total_steps // 2:
                probe("mid", step)
            if main_rank and step % save_steps == 0 and step < total_steps:
                # Model-only rescue snapshot: no optimizer, never reportable. Named
                # interim-* so _newest_model_artifact ignores it
                # (train_ance_warmup.py:219-227).
                model.save(str(output_dir / f"interim-{step}"))
                tokenizer.save_pretrained(str(output_dir / f"interim-{step}"))
    probe("end", step)
    if main_rank:
        model.save(str(output_dir))
        tokenizer.save_pretrained(str(output_dir))
        assert_training_succeeded(output_dir, manifest)
        print(f"✅ MaxP training completed. Model saved to: {output_dir}", flush=True)
    if distributed:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
