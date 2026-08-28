"""ANCE Trainer -- custom training loop, runs on GPU 0.

Mirrors the reference `drivers/run_ann.py`: a step-based `while global_step <
max_steps` loop that polls the work root at `logging_steps` and swaps the DataLoader
in place. Training NEVER stops between ANN refreshes (paper Figure 2, Appendix A.1).

Paper reference: Section 4, Figure 2, Appendix A.1
"""
import os
import sys
import json
import math
import argparse
import torch
from torch.utils.data import DataLoader, Dataset
from pathlib import Path
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root / 'src'))
sys.path.append(str(project_root / 'scripts'))

from utils.helpers import get_training_context, append_jsonl, set_seed, \
                          ranking_probe, probe_triples_from_mixture, build_adamw, \
                          retry_io, TRAINING_LOG_NAME, atomic_write
from ance_mining import RoundError, latest_committed_round, read_round, INITIAL_ROUND

os.environ["TRANSFORMERS_ATTENTION_IMPLEMENTATION"] = "eager"

TRAINER_SUMMARY_NAME = "ance_trainer_summary.json"


class NonFiniteOptimization(RuntimeError):
    """Optimization diverged; stopping rather than saving poisoned weights."""


class NonFiniteLoss(NonFiniteOptimization):
    """The forward pass produced a non-finite loss."""


class NonFiniteGradNorm(NonFiniteOptimization):
    """The backward pass produced non-finite gradients."""


def check_finite_loss(value, step):
    """Raise on a non-finite loss BEFORE it reaches backward().

    One NaN backward pass poisons every parameter. The old loop stepped the
    optimizer unconditionally, saved the diverged checkpoint, exited 0 and let the
    evaluation score it. The loss is only logged every `logging_steps`, so nothing
    downstream would have seen the NaN either.
    """
    if not math.isfinite(float(value)):
        raise NonFiniteLoss(
            f"non-finite loss={value} at step {step}; the optimization diverged. "
            f"No optimizer step is taken and no checkpoint is written.")
    return float(value)


def check_finite_grad_norm(value, step):
    """Raise on a non-finite gradient norm BEFORE optimizer.step().

    `clip_grad_norm_` returns the PRE-clipping norm, and clipping does not rescue a
    non-finite one: the coefficient is `max_norm / (total_norm + 1e-6)`, so an inf or
    NaN total norm yields a non-finite coefficient and non-finite gradients survive
    into the step. A finite loss is no protection -- the loss can be finite while a
    single parameter's gradient overflows.

    Deliberately stricter than `helpers.assert_training_succeeded`, which only WARNS
    on a non-finite `grad_norm`. That is post-hoc log analysis of HF-Trainer runs,
    where the value is a record of something clipping already handled. This loop is
    hand-rolled bf16 with no GradScaler, so a non-finite norm here is a real
    divergence rather than a loss-scaling artefact, and there is nothing downstream
    that would catch the step it would otherwise take.
    """
    if not math.isfinite(float(value)):
        raise NonFiniteGradNorm(
            f"non-finite gradient norm={value} at step {step} (pre-clipping); the "
            f"gradients are already non-finite, so clipping cannot rescue them and "
            f"the step would write NaN into every parameter. No optimizer step is "
            f"taken and no checkpoint is written.")
    return float(value)


def apply_gradients(model, optimizer, scheduler, *, max_grad_norm, step):
    """Clip, verify, then step. Returns the pre-clipping gradient norm.

    One function so the ORDER is testable: the guard must run between clipping and
    `optimizer.step()`, and a test can assert that parameters are unchanged after a
    raise rather than asserting the ordering by reading the source.
    """
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
    value = check_finite_grad_norm(float(grad_norm), step)
    optimizer.step()
    scheduler.step()
    optimizer.zero_grad(set_to_none=True)
    return value


class ANCEDataset(Dataset):
    """Load JSONL training examples from a committed round directory."""

    def __init__(self, data_dir, tokenizer, max_q_len, max_p_len, train_group_size,
                 paper_mode=False):
        self.examples = []
        # Paper mode (data/msmarco_data.py:337-362, run_train.sh --triplet): the round
        # stores 20 mined negatives per query and each becomes its OWN
        # (query, positive, negative) instance. Every loss term still sees exactly one
        # positive and one negative; the group is not widened into an in-batch softmax.
        self.paper_mode = paper_mode
        self.triplets = []
        for f_path in sorted(Path(data_dir).glob("*.jsonl")):
            with open(f_path) as f:
                for line in f:
                    if line.strip():
                        self.examples.append(json.loads(line))
        self.tokenizer = tokenizer
        self.max_q_len = max_q_len
        self.max_p_len = max_p_len
        self.train_group_size = train_group_size
        self._validate()
        if paper_mode:
            for ex in self.examples:
                pos = ex['positive_passages'][0]['text']
                for neg in ex['negative_passages'][:train_group_size - 1]:
                    self.triplets.append((ex['query'], pos, neg['text']))

    def _validate(self):
        """Every record must carry its full complement of real negatives.

        The old loader padded a short group with `passages[-1]`, which for a record
        with no mined negatives is the POSITIVE -- training the loss to push the gold
        document away from its own query. The miner now refuses to publish a round
        that cannot supply the negatives, so a short record here means the round was
        written by something other than the current miner.
        """
        need = self.train_group_size - 1
        for i, ex in enumerate(self.examples):
            if not ex.get('positive_passages'):
                raise ValueError(f"record {i} has no positive_passages")
            if len(ex.get('negative_passages') or []) < need:
                raise ValueError(
                    f"record {i} (query {ex.get('query_id')!r}) carries "
                    f"{len(ex.get('negative_passages') or [])} negative(s), needs "
                    f"{need}. ANCE never pads a group with the positive.")

    def __len__(self):
        return len(self.triplets) if self.paper_mode else len(self.examples)

    def __getitem__(self, idx):
        if self.paper_mode:
            # Dynamic padding is applied by the collate function, not here: MS MARCO
            # passages average ~75 word-pieces against a 512 cap, so padding each item
            # to the cap would multiply the step cost several-fold.
            return self.triplets[idx]
        ex = self.examples[idx]
        passages = (ex['positive_passages'][:1]
                    + ex['negative_passages'][:self.train_group_size - 1])
        q = self.tokenizer(ex['query'], max_length=self.max_q_len,
                           truncation=True, padding='max_length', return_tensors='pt')
        ps = [self.tokenizer(p['text'], max_length=self.max_p_len,
                             truncation=True, padding='max_length', return_tensors='pt')
              for p in passages]
        return {
            'q_input_ids':      q['input_ids'].squeeze(0),
            'q_attention_mask': q['attention_mask'].squeeze(0),
            'p_input_ids':      torch.stack([p['input_ids'].squeeze(0)  for p in ps]),
            'p_attention_mask': torch.stack([p['attention_mask'].squeeze(0) for p in ps]),
        }


def make_paper_collate(tokenizer, max_q, max_p):
    """Pad to the longest sequence in the batch, not to the cap."""
    def collate(batch):
        q, p, n = zip(*batch)
        enc = lambda texts, ml: tokenizer(list(texts), padding=True, truncation=True,
                                          max_length=ml, return_tensors='pt')
        return enc(q, max_q), enc(p, max_p), enc(n, max_p)
    return collate


def make_dataloader(data_dir, tokenizer, ctx, batch_size, generator=None):
    paper = bool(ctx['args'].get('paper_fidelity'))
    # Lengths come from ctx, which has already applied any recipe override.
    ds = ANCEDataset(data_dir, tokenizer, ctx['max_q'], ctx['max_p'],
                     ctx['args']['train_group_size'], paper_mode=paper)
    return DataLoader(ds, batch_size=batch_size, shuffle=True,
                      num_workers=ctx['args']['dataloader_num_workers'],
                      drop_last=True, generator=generator,
                      collate_fn=(make_paper_collate(tokenizer, ctx['max_q'],
                                                     ctx['max_p']) if paper else None))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name_or_path', required=True)
    parser.add_argument('--work_root',  required=True)
    parser.add_argument('--run_id',     required=True)
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--max_steps',  type=int, required=True)
    parser.add_argument('--seed',       type=int, default=42)
    parser.add_argument('--recipe',     default='ance')
    args = parser.parse_args()

    set_seed(args.seed)
    ctx = get_training_context(args.recipe)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    bf16 = bool(ctx['args']['bf16'])
    paper = bool(ctx['args'].get('paper_fidelity'))

    if paper:
        # DenseModel.build goes through AutoModel, which yields a bare RoBERTa and
        # silently drops embeddingHead/norm -- the paper checkpoint cannot load
        # through Tevatron at all. load_ance_encoder refuses anything left random.
        from ance_paper import load_ance_encoder
        model = load_ance_encoder(args.model_name_or_path,
                                  attn_implementation='eager').cuda()
    else:
        from tevatron.retriever.modeling import DenseModel
        from tevatron.retriever.arguments import (
            ModelArguments, TevatronTrainingArguments as TrainingArguments)

        if not hasattr(DenseModel, "_keys_to_ignore_on_save"):
            setattr(DenseModel, "_keys_to_ignore_on_save", None)
        # DenseModel.forward() applies temperature itself. ModelArguments does not
        # forward the attention implementation, so both declarations are required.
        model_args = ModelArguments(
            model_name_or_path=args.model_name_or_path,
            pooling=ctx['pooling'], normalize=ctx['normalize'],
            temperature=ctx['temperature'], attn_implementation='eager')
        train_args = TrainingArguments(output_dir=args.output_dir, bf16=bf16)
        model = DenseModel.build(model_args, train_args,
                                 attn_implementation='eager').cuda()

    batch_size    = ctx['args']['batch_size']
    logging_steps = ctx['args']['logging_steps']       # also the round-poll interval
    save_steps    = ctx['args']['save_steps']          # == the ANN refresh interval m
    work_root     = Path(args.work_root)
    output_dir    = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    generator = torch.Generator()
    generator.manual_seed(args.seed)

    # The initial round is base-model mined; it is committed by the orchestrator and
    # is NOT a refresh. `latest_committed_round` ignores it by design.
    print(f"[Trainer] Loading initial round from {work_root}", flush=True)
    initial_dir, initial_meta = read_round(work_root, INITIAL_ROUND, run_id=args.run_id)
    train_dataloader = make_dataloader(initial_dir, tokenizer, ctx,
                                       batch_size, generator)
    train_iter  = iter(train_dataloader)
    last_ann_no = 0
    rounds = [{'ann_no': INITIAL_ROUND, 'checkpoint': initial_meta.get('checkpoint'),
               'checkpoint_step': 0, 'consumed_at_step': 0, 'consumed_steps': 0}]

    if paper:
        # utils/lamb.py, --optimizer lamb (run_train.sh:110). Absolute warmup_steps,
        # as upstream passes them, rather than a ratio.
        from ance_paper import Lamb
        optimizer = Lamb(model.parameters(), lr=float(ctx['args']['learning_rate']),
                         eps=float(ctx['args']['lamb_eps']),
                         weight_decay=float(ctx['args']['weight_decay']))
        optimizer_spec = {'name': 'Lamb', 'lr': float(ctx['args']['learning_rate']),
                          'eps': float(ctx['args']['lamb_eps']),
                          'weight_decay': float(ctx['args']['weight_decay']),
                          'source': 'microsoft/ANCE utils/lamb.py'}
        warmup_steps = int(ctx['args']['warmup_steps'])
    else:
        # The SAME explicit optimizer GRASS builds. The BRIGHT table compares negative
        # selection, so the optimizer is pinned rather than left to defaults.
        optimizer, optimizer_spec = build_adamw(
            model.parameters(), lr=ctx['args']['learning_rate'],
            weight_decay=ctx['args']['weight_decay'], label='ance')
        # Paper: linear warmup (5000 steps at MARCO scale); scaled by warmup_ratio.
        warmup_steps = int(args.max_steps * ctx['args']['warmup_ratio'])
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, args.max_steps)

    log_path = output_dir / TRAINING_LOG_NAME
    mixture_dir = Path(ctx['processed_dir']) / ctx['args']['mixture_dir']
    probe_triples = probe_triples_from_mixture(sorted(mixture_dir.glob("*.jsonl")))

    def _probe(phase, step):
        """Best-effort, like attach_training_diagnostics._run_probe: a probe is a
        diagnostic and must never end a training run. Recorded either way, so a
        failure can never be mistaken for a passing probe."""
        try:
            result = ranking_probe(model, tokenizer, probe_triples,
                                   torch.device('cuda'), ctx['max_q'], ctx['max_p'],
                                   normalize=ctx['normalize'])
        except Exception as e:                                    # noqa: BLE001
            result = {"error": f"{type(e).__name__}: {e}"}
        append_jsonl(log_path, {"global_step": step, "phase": phase, **result})

    def _write_summary(step):
        """Critical, not best-effort: this file is the ONLY evidence of round
        consumption, and train_ance.py fails the run outright when it is missing. One
        EREMOTEIO on BeeGFS (P11/P14) would otherwise discard a completed run's
        evidence, or leave a stale summary under-reporting the rounds consumed. This
        is the 'critical caller' half of retry_io's contract: retry, then VERIFY the
        postcondition and raise."""
        path = output_dir / TRAINER_SUMMARY_NAME
        payload = {'run_id': args.run_id, 'work_root': str(work_root),
                   'optimizer': optimizer_spec, 'max_steps': args.max_steps,
                   'final_step': step, 'rounds': rounds}

        def _write():
            with atomic_write(path) as f:
                json.dump(payload, f, indent=2, default=str)

        if not retry_io(_write, f"write {path.name}"):
            raise OSError(
                f"could not write {path} after repeated attempts. It is the only "
                f"record of which ANN rounds were consumed, so the run cannot be "
                f"validated without it.")

    if args.max_steps < 1:
        raise ValueError(f"--max_steps must be >= 1, got {args.max_steps}")

    global_step = 0
    _probe("begin", 0)
    _write_summary(0)
    model.train()
    print(f"[Trainer] Starting: max_steps={args.max_steps}, "
          f"logging_steps={logging_steps}, save_steps={save_steps}", flush=True)

    # ── MAIN TRAINING LOOP ───────────────────────────────────────────────────
    while global_step < args.max_steps:

        # Paper: "when the new ANN index is ready, it immediately replaces existing
        # negatives in training, without waiting."
        if global_step > 0 and global_step % logging_steps == 0:
            ann_no = latest_committed_round(work_root)
            if ann_no > last_ann_no:
                try:
                    data_dir, meta = read_round(work_root, ann_no, run_id=args.run_id)
                except RoundError as exc:
                    # Refused, not consumed: a round that cannot prove it belongs to
                    # this run is another experiment's data.
                    print(f"[Trainer] REFUSED round {ann_no}: {exc}", flush=True)
                else:
                    print(f"[Trainer] Step {global_step}: round {ann_no} "
                          f"(checkpoint-{meta.get('checkpoint_step')}) — swapping",
                          flush=True)
                    train_dataloader = make_dataloader(data_dir, tokenizer, ctx,
                                                       batch_size, generator)
                    train_iter  = iter(train_dataloader)
                    last_ann_no = ann_no
                    rounds.append({'ann_no': ann_no,
                                   'checkpoint': meta.get('checkpoint'),
                                   'checkpoint_step': int(meta.get('checkpoint_step') or 0),
                                   'consumed_at_step': global_step,
                                   'consumed_steps': 0})

        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_dataloader)
            batch = next(train_iter)

        if paper:
            # model/models.py:77-81 -- one positive, one negative, raw dot, no
            # temperature. EncoderModel.forward cannot express this: it hard-codes
            # scores.view(B, -1) with target = arange(B) * G, i.e. in-batch softmax.
            from ance_paper import pairwise_nll
            q, pos, neg = ({k: v.cuda() for k, v in b.items()} for b in batch)
            with torch.autocast('cuda', dtype=torch.bfloat16, enabled=bf16):
                loss = pairwise_nll(model(**q), model(**pos), model(**neg))
        else:
            # DenseModel.forward(query, passage) calls self.encoder(**query, ...).
            batch = {k: v.cuda() for k, v in batch.items()}
            B, G, L = batch['p_input_ids'].shape

            with torch.autocast('cuda', dtype=torch.bfloat16, enabled=bf16):
                outputs = model(
                    query={'input_ids':      batch['q_input_ids'],
                           'attention_mask': batch['q_attention_mask']},
                    passage={'input_ids':      batch['p_input_ids'].view(B * G, L),
                             'attention_mask': batch['p_attention_mask'].view(B * G, L)},
                )
                loss = outputs.loss

        loss_value = check_finite_loss(loss.item(), global_step)
        loss.backward()

        # Clips, rejects a non-finite norm, then steps. The returned value is the
        # PRE-clipping norm -- the diagnostic one.
        grad_norm = apply_gradients(model, optimizer, scheduler,
                                    max_grad_norm=ctx['args']['max_grad_norm'],
                                    step=global_step)
        global_step += 1
        rounds[-1]['consumed_steps'] += 1

        if global_step % logging_steps == 0:
            append_jsonl(log_path, {
                "global_step": global_step, "loss": loss_value,
                "learning_rate": scheduler.get_last_lr()[0],
                "grad_norm": float(grad_norm), "ann_no": last_ann_no,
            })
            print(f"[Trainer] step={global_step}/{args.max_steps} "
                  f"loss={loss_value:.4f} ann_no={last_ann_no}", flush=True)
            _write_summary(global_step)

        # Saving a checkpoint is what triggers the next ANN refresh, so save_steps
        # IS the paper's refresh interval m ("update the ANN index once every m
        # batches, i.e. with checkpoint f_k").
        if global_step % save_steps == 0 or global_step == args.max_steps:
            ckpt = output_dir / f"checkpoint-{global_step}"
            ckpt.mkdir(exist_ok=True)
            # AnceEncoder is a plain PreTrainedModel: save_pretrained writes the
            # encoder AND the projection head into one weight file.
            (model.save_pretrained if paper else model.save)(str(ckpt))
            tokenizer.save_pretrained(str(ckpt))
            torch.save(scheduler.state_dict(), ckpt / "scheduler.pt")
            # optimizer.pt LAST: is_valid_checkpoint() reads it as the validity flag.
            torch.save(optimizer.state_dict(), ckpt / "optimizer.pt")
            print(f"[Trainer] Saved checkpoint-{global_step}", flush=True)

    # A terminal record, so the final step is in the log even when max_steps is not
    # a multiple of logging_steps. Without it assert_training_succeeded compares
    # 10300 against a planned 10312 and rejects a run that completed.
    append_jsonl(log_path, {
        "global_step": global_step, "loss": loss_value,
        "learning_rate": scheduler.get_last_lr()[0],
        "grad_norm": float(grad_norm), "ann_no": last_ann_no, "terminal": True,
    })
    _probe("end", global_step)
    _write_summary(global_step)

    (model.save_pretrained if paper else model.save)(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    print("[Trainer] Training complete.", flush=True)


if __name__ == "__main__":
    main()
