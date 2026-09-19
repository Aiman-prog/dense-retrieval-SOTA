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
import bisect
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
from ance_mining import latest_committed_round, read_round, INITIAL_ROUND

os.environ["TRANSFORMERS_ATTENTION_IMPLEMENTATION"] = "eager"

TRAINER_SUMMARY_NAME = "ance_trainer_summary.json"


class NonFiniteOptimization(RuntimeError):
    """Optimization diverged; stopping rather than saving poisoned weights.

    One NaN backward pass poisons every parameter. The old loop stepped the optimizer
    unconditionally, saved the diverged checkpoint, exited 0 and let the evaluation
    score it. Deliberately stricter than `helpers.assert_training_succeeded`, which
    only WARNS on a non-finite grad_norm: that is post-hoc analysis of HF-Trainer runs
    where clipping already handled the value, while this loop is hand-rolled with no
    GradScaler and nothing downstream would catch the step it would otherwise take.
    """


def optimization_step(model, optimizer, scheduler, loss, *, max_grad_norm, step):
    """Check, backward, clip, step. Returns (loss value, pre-clipping grad norm).

    One function so the ORDER is testable: the finite checks must sit before
    `optimizer.step()`, and a test can assert parameters are unchanged after a raise.
    `error_if_nonfinite` is what rejects a non-finite norm -- clipping cannot rescue
    one, because the coefficient `max_norm / (total_norm + 1e-6)` is itself non-finite
    and the bad gradients survive into the step.
    """
    value = float(loss.item())
    if not math.isfinite(value):
        raise NonFiniteOptimization(
            f"non-finite loss={value} at step {step}; the optimization diverged. "
            f"No optimizer step is taken and no checkpoint is written.")
    loss.backward()
    try:
        grad_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), max_grad_norm, error_if_nonfinite=True)
    except RuntimeError as exc:
        raise NonFiniteOptimization(
            f"non-finite gradient norm at step {step} (pre-clipping): {exc}. "
            f"No optimizer step is taken and no checkpoint is written.") from exc
    optimizer.step()
    scheduler.step()
    optimizer.zero_grad(set_to_none=True)
    return value, float(grad_norm)


class ANCEDataset(Dataset):
    """Load JSONL training examples from a committed round directory."""

    def __init__(self, data_dir, train_group_size, paper_mode=False, ragged=False):
        self.examples = []
        # Paper mode (data/msmarco_data.py:337-362, run_train.sh --triplet): the round
        # stores 20 mined negatives per query and each becomes its OWN
        # (query, positive, negative) instance. Every loss term still sees exactly one
        # positive and one negative; the group is not widened into an in-batch softmax.
        self.paper_mode = paper_mode
        for f_path in sorted(Path(data_dir).glob("*.jsonl")):
            with open(f_path) as f:
                for line in f:
                    if line.strip():
                        self.examples.append(json.loads(line))
        # No tokenizer here: both modes hand TEXT to the collate function so padding
        # reaches the longest sequence in the batch rather than the configured cap.
        self.train_group_size = train_group_size
        self.n_negs = train_group_size - 1
        self.ragged = ragged
        if ragged:
            if not paper_mode:
                raise ValueError("ragged indexing yields one (query, positive, "
                                 "negative) triplet per negative, which only paper "
                                 "mode consumes; pass paper_mode=True.")
            self._build_ragged_index()
        else:
            self._validate()

    def _validate(self):
        """Every record must carry its full complement of real negatives.

        The old loader padded a short group with `passages[-1]`, which for a record
        with no mined negatives is the POSITIVE -- training the loss to push the gold
        document away from its own query. The miner now refuses to publish a round
        that cannot supply the negatives, so a short record here means the round was
        written by something other than the current miner.

        Paper mode also relies on this: `__getitem__` addresses negative `i` of record
        `n` by `divmod`, which is only well defined because every record is guaranteed
        at least `n_negs` of them.
        """
        for i, ex in enumerate(self.examples):
            if not ex.get('positive_passages'):
                raise ValueError(f"record {i} has no positive_passages")
            if len(ex.get('negative_passages') or []) < self.n_negs:
                raise ValueError(
                    f"record {i} (query {ex.get('query_id')!r}) carries "
                    f"{len(ex.get('negative_passages') or [])} negative(s), needs "
                    f"{self.n_negs}. ANCE never pads a group with the positive.")

    def _build_ragged_index(self):
        """Index every (query, positive, negative) the mixture actually holds.

        Upstream has no notion of "negatives per query". Its warm-up reads
        triples.train.small.tsv one triplet per LINE (drivers/run_warmup.py:743,
        data/process_fn.py:48-70 -- exactly three tab cells), and its ANN path yields
        one instance per negative (data/msmarco_data.py:355-360, `for neg_pid in
        neg_pids: yield`). A query with 8 negatives is simply 8 lines.

        The uniform-count requirement is OURS: it comes from Tevatron's grouped record
        shape plus divmod addressing. ~1.8% of Tevatron/msmarco-passage records carry
        fewer than 30 BM25 negatives (min 1), passed through verbatim by the
        preprocessor, so demanding 30 aborts the warm-up on upstream's own data.

        ONLY for a static mixture. A mined round must stay on _validate: there a short
        record means the miner published a round it should have discarded, and the
        strict check is the thing that catches it.
        """
        offsets, total = [], 0
        for i, ex in enumerate(self.examples):
            if not ex.get('positive_passages'):
                raise ValueError(f"record {i} has no positive_passages")
            # Capped, not required: train_group_size stays the per-query ceiling.
            n = min(len(ex.get('negative_passages') or []), self.n_negs)
            if n < 1:
                raise ValueError(
                    f"record {i} (query {ex.get('query_id')!r}) carries no negatives. "
                    f"ANCE never pads a group with the positive.")
            offsets.append(total)
            total += n
        self._offsets, self._n_triplets = offsets, total

    def __len__(self):
        if self.ragged:
            return self._n_triplets
        if self.paper_mode:
            return len(self.examples) * self.n_negs
        return len(self.examples)

    def __getitem__(self, idx):
        # Both modes return TEXT and let the collate function tokenize the batch, so
        # padding reaches the longest sequence present rather than the cap. q1024/p1024
        # against a training-mixture median of 41 query and 114 passage word-pieces
        # meant the old `padding='max_length'` paid for roughly an order of magnitude
        # of pad tokens on every step.
        if self.ragged:
            # Same addressing idea as paper mode, over a cumulative count instead of a
            # fixed stride, because the per-record count varies. On uniform data the
            # two agree exactly.
            rec = bisect.bisect_right(self._offsets, idx) - 1
            ex = self.examples[rec]
            return (ex['query'], ex['positive_passages'][0]['text'],
                    ex['negative_passages'][idx - self._offsets[rec]]['text'])
        if self.paper_mode:
            # Addressed, not materialized. A shard of 80k MS MARCO queries at 20
            # negatives is 1.6M triplets, and holding them as tuples bought nothing
            # that this arithmetic does not.
            rec, neg = divmod(idx, self.n_negs)
            ex = self.examples[rec]
            return (ex['query'], ex['positive_passages'][0]['text'],
                    ex['negative_passages'][neg]['text'])
        ex = self.examples[idx]
        passages = (ex['positive_passages'][:1]
                    + ex['negative_passages'][:self.n_negs])
        # Tevatron's TrainCollator indexes the query and every passage as a sequence
        # (collator.py:31-32), so each is wrapped in a one-element list.
        return [ex['query']], [[p['text']] for p in passages]


def make_paper_collate(tokenizer, max_q, max_p):
    """Pad to the longest sequence in the batch, not to the cap.

    The BGE arm uses Tevatron's own TrainCollator; this one exists because the paper
    arm's objective is pairwise and needs the positive and the negative kept apart,
    which the grouped collator cannot express.
    """
    def collate(batch):
        q, p, n = zip(*batch)
        enc = lambda texts, ml: tokenizer(list(texts), padding=True, truncation=True,
                                          max_length=ml, return_tensors='pt')
        return enc(q, max_q), enc(p, max_p), enc(n, max_p)
    return collate


def make_bge_collate(tokenizer, max_q, max_p):
    """Tevatron's pinned TrainCollator, configured from the recipe's lengths.

    The same collator the in-batch and cross-batch arms get through Tevatron's driver,
    so all three BGE arms tokenize identically. It returns a flat (B*G, L) passage
    batch, which is exactly what `EncoderModel.forward` wants -- it derives the group
    width itself from `p_reps.size(0) // q_reps.size(0)`. The custom collator this
    replaces built (B, G, L) only for the training step to `view` it straight back.
    """
    from tevatron.retriever.arguments import DataArguments
    from tevatron.retriever.collator import TrainCollator
    return TrainCollator(
        data_args=DataArguments(query_max_len=max_q, passage_max_len=max_p),
        tokenizer=tokenizer)


def make_dataloader(data_dir, tokenizer, ctx, batch_size, generator=None):
    paper = bool(ctx['args'].get('paper_fidelity'))
    # Lengths come from ctx, which has already applied any recipe override.
    ds = ANCEDataset(data_dir, ctx['args']['train_group_size'], paper_mode=paper)
    make_collate = make_paper_collate if paper else make_bge_collate
    return DataLoader(ds, batch_size=batch_size, shuffle=True,
                      num_workers=ctx['args']['dataloader_num_workers'],
                      drop_last=True, generator=generator,
                      collate_fn=make_collate(tokenizer, ctx['max_q'], ctx['max_p']))


def write_trainer_summary(path, payload):
    """Durably write the refresh-accounting record, or raise.

    Critical, not best-effort: train_ance.py fails the run outright when this file is
    missing, so a lost write must end the run rather than leave it unvalidatable. The
    'critical caller' half of retry_io's contract -- retry, then check and raise.
    """
    def _write():
        with atomic_write(path) as f:
            json.dump(payload, f, indent=2, default=str)

    if not retry_io(_write, f"write {Path(path).name}"):
        raise OSError(
            f"could not write {path} after repeated attempts. It is the only "
            f"record of which ANN rounds were consumed, so the run cannot be "
            f"validated without it.")


def build_schedule(optimizer, warmup_steps, max_steps, scheduler_max_steps):
    """Linear warmup then decay toward the HORIZON, which is not the stopping step.

    Upstream's --single_warmup builds one decay over --max_steps (default 1,000,000)
    while the released comparison point is checkpoint 600K (run_ann.py:174-178).
    Collapsing the two would change the learning rate at every step, so the run would
    not be the same optimization even at an identical stop.

    Returns (scheduler, horizon).
    """
    horizon = scheduler_max_steps or max_steps
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, horizon)
    if horizon != max_steps:
        print(f"[Trainer] linear decay toward step {horizon}; training stops at "
              f"{max_steps} (a labelled deviation, not a shortened schedule)",
              flush=True)
    return scheduler, horizon


def build_trainer_summary(*, run_id, work_root, optimizer, max_steps,
                          scheduler_max_steps, save_steps, final_step, rounds,
                          rounds_completed):
    """Build the durable refresh-accounting record from trainer state.

    Everything countable is DERIVED here from the final step and the consumed-round
    list, rather than maintained as parallel counters in the loop that could drift
    apart from it.

    The three round numbers stay separate in the output, because they answer different
    questions. A checkpoint OPPORTUNITY is a step at which a save happened; a COMPLETED
    round is one the miner published; a CONSUMED round is one this loop actually
    trained on. Reporting a checkpoint opportunity as a refresh is exactly how a static
    run came to be described as ANCE.
    """
    consumed = [r['ann_no'] for r in rounds if r['ann_no'] != INITIAL_ROUND]
    last_ann_no = max(consumed, default=0)
    # A save fires on every save_steps boundary, plus once at max_steps when that is
    # not itself a boundary.
    opportunities = final_step // save_steps
    if final_step == max_steps and final_step % save_steps:
        opportunities += 1
    return {
        'run_id': run_id, 'work_root': str(work_root), 'optimizer': optimizer,
        'max_steps': max_steps, 'scheduler_max_steps': scheduler_max_steps,
        'final_step': final_step, 'rounds': rounds,
        'checkpoint_opportunities': opportunities,
        'rounds_completed': rounds_completed,
        'rounds_consumed': len(consumed),
        # Published while an earlier round was still training, so never seen. The
        # trainer consumes latest-only; this is normal, and recorded so the refresh
        # count cannot be misread as the publication count.
        'rounds_skipped': last_ann_no - len(consumed),
        'terminal_unconsumed': (rounds_completed
                                if rounds_completed > last_ann_no else None),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name_or_path', required=True)
    parser.add_argument('--work_root',  required=True)
    parser.add_argument('--run_id',     required=True)
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--max_steps',  type=int, required=True)
    # The horizon the linear decay is computed against, which is NOT the step the run
    # stops at. Upstream's --single_warmup builds one schedule over --max_steps
    # (default 1,000,000) while the released comparison point is checkpoint 600K, so
    # collapsing the two would change the LR at every step (run_ann.py:174-178).
    parser.add_argument('--scheduler_max_steps', type=int, default=None)
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
        #
        # FP32 parameters with bf16 autocast, deliberately NOT the bf16 weights the
        # BGE arm uses. Upstream runs fp32, and LAMB at lr 1e-6 moves a weight by far
        # less than bf16 can represent, so bf16 master weights would round most
        # updates to zero and the reproduction would silently barely train.
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
        # torch_dtype must be passed EXPLICITLY. EncoderModel.build ignores train_args
        # entirely (tevatron/retriever/modeling/encoder.py:114), so bf16=True on the
        # TrainingArguments above sets no dtype. Tevatron's own driver derives it from
        # training_args.bf16 (driver/train.py:71-84), which is how the in-batch and
        # cross-batch arms get bf16 WEIGHTS; without this line ANCE would train fp32
        # weights under autocast and the BRIGHT comparison would straddle two
        # numerical paths.
        model = DenseModel.build(model_args, train_args,
                                 attn_implementation='eager',
                                 torch_dtype=torch.bfloat16 if bf16
                                 else torch.float32).cuda()

    if ctx['args']['gradient_checkpointing']:
        # DenseModel holds the transformer at `.encoder`; calling
        # gradient_checkpointing_enable() on the wrapper hits Tevatron's own broken
        # override (see train_inbatch.py:37-46). The paper encoder wraps RobertaModel
        # the same way. This is the memory fix that makes batch 64 at q1024/p1024 fit.
        (model.roberta if paper else model.encoder).gradient_checkpointing_enable()
        print("[Trainer] gradient checkpointing enabled", flush=True)

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
               'checkpoint_step': 0, 'consumed_at_step': 0, 'stale_steps': 0,
               'consumed_steps': 0,
               'shard_index': initial_meta.get('shard_index'),
               'n_shards': initial_meta.get('n_shards'),
               'n_shard_queries': initial_meta.get('n_shard_queries')}]

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
    scheduler, horizon = build_schedule(optimizer, warmup_steps, args.max_steps,
                                        args.scheduler_max_steps)

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
        """Written ONCE, after the loop completes. Critical, not best-effort: this file
        is the only evidence of round consumption, and train_ance.py fails the run
        outright when it is missing, so one EREMOTEIO on BeeGFS (P11/P14) would discard
        a completed run's evidence. This is the 'critical caller' half of retry_io's
        contract: retry, then VERIFY the postcondition and raise.

        There is no resume, so a periodic summary was only ever crash forensics, and
        training_log.jsonl plus the SLURM log already carry those. At logging_steps 100
        over a 300K-step budget it was 3,000 full-payload writes and 3,000 directory
        globs for nothing.
        """
        path = output_dir / TRAINER_SUMMARY_NAME
        payload = build_trainer_summary(
            run_id=args.run_id, work_root=work_root, optimizer=optimizer_spec,
            max_steps=args.max_steps, scheduler_max_steps=horizon,
            save_steps=save_steps, final_step=step, rounds=rounds,
            rounds_completed=latest_committed_round(work_root))

        write_trainer_summary(path, payload)

    if args.max_steps < 1:
        raise ValueError(f"--max_steps must be >= 1, got {args.max_steps}")

    global_step = 0
    interval_loss_sum, interval_loss_n = 0.0, 0
    _probe("begin", 0)
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
                # No refusal path. A committed round that cannot prove it belongs to
                # this run is another experiment's data, and continuing past it means
                # training to max_steps on stale negatives -- the P-ANCE-01 failure.
                # read_round raises RoundError and the run ends here.
                data_dir, meta = read_round(work_root, ann_no, run_id=args.run_id)
                print(f"[Trainer] Step {global_step}: round {ann_no} "
                      f"(checkpoint-{meta.get('checkpoint_step')}) — swapping",
                      flush=True)
                train_dataloader = make_dataloader(data_dir, tokenizer, ctx,
                                                   batch_size, generator)
                train_iter  = iter(train_dataloader)
                if ann_no > last_ann_no + 1:
                    # Latest-only consumption: rounds published while this one was
                    # being trained are never seen. Normal, and derived into
                    # rounds_skipped so the refresh count cannot be misread as the
                    # publication count.
                    print(f"[Trainer] skipped round(s) "
                          f"{list(range(last_ann_no + 1, ann_no))} — superseded "
                          f"before the next poll", flush=True)
                last_ann_no = ann_no
                ckpt_step = int(meta.get('checkpoint_step') or 0)
                rounds.append({'ann_no': ann_no,
                               'checkpoint': meta.get('checkpoint'),
                               'checkpoint_step': ckpt_step,
                               'consumed_at_step': global_step,
                               # negative age: how far the trainer had moved past
                               # the checkpoint that produced these negatives
                               'stale_steps': global_step - ckpt_step,
                               'consumed_steps': 0,
                               'dev_ndcg_cut_10': meta.get('dev_ndcg_cut_10'),
                               'shard_index': meta.get('shard_index'),
                               'n_shards': meta.get('n_shards'),
                               'n_shard_queries': meta.get('n_shard_queries'),
                               'mining_seconds': {k: v for k, v in meta.items()
                                                  if k.endswith('_s')}})

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
            # TrainCollator hands back the flat (query, passage) pair Tevatron's own
            # trainer passes to DenseModel.forward. No reshaping: forward recovers the
            # group width from p_reps.size(0) // q_reps.size(0).
            q_batch, p_batch = ({k: v.cuda() for k, v in b.items()} for b in batch)
            with torch.autocast('cuda', dtype=torch.bfloat16, enabled=bf16):
                loss = model(query=q_batch, passage=p_batch).loss

        # Checks, backwards, clips, rejects a non-finite norm, then steps. The
        # returned norm is the PRE-clipping one -- the diagnostic value.
        loss_value, grad_norm = optimization_step(
            model, optimizer, scheduler, loss,
            max_grad_norm=ctx['args']['max_grad_norm'], step=global_step)
        global_step += 1
        rounds[-1]['consumed_steps'] += 1
        interval_loss_sum += loss_value
        interval_loss_n += 1

        if global_step % logging_steps == 0:
            # Both, because they answer different questions. Upstream logs only the
            # interval mean (run_ann.py:296) -- the comparable series -- while the
            # latest batch loss is what the finite-loss guard actually saw.
            interval_mean = interval_loss_sum / max(interval_loss_n, 1)
            interval_loss_sum, interval_loss_n = 0.0, 0
            append_jsonl(log_path, {
                "global_step": global_step, "loss": loss_value,
                "loss_interval_mean": interval_mean,
                "learning_rate": scheduler.get_last_lr()[0],
                "grad_norm": float(grad_norm), "ann_no": last_ann_no,
                "stale_steps": global_step - rounds[-1]['checkpoint_step'],
            })
            print(f"[Trainer] step={global_step}/{args.max_steps} "
                  f"loss={loss_value:.4f} mean={interval_mean:.4f} "
                  f"ann_no={last_ann_no}", flush=True)

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
        "loss_interval_mean": (interval_loss_sum / interval_loss_n
                               if interval_loss_n else loss_value),
        "learning_rate": scheduler.get_last_lr()[0],
        "grad_norm": float(grad_norm), "ann_no": last_ann_no,
        "stale_steps": global_step - rounds[-1]['checkpoint_step'], "terminal": True,
    })
    _probe("end", global_step)
    _write_summary(global_step)

    (model.save_pretrained if paper else model.save)(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    print("[Trainer] Training complete.", flush=True)


if __name__ == "__main__":
    main()
