"""
Async Fast-GRASS — TRAINER process (GPU 0).

Owns gradients, optimizer, scheduler, and checkpoints. Consumes mined JSONL rounds
and NEVER reads or mutates ``H``, ``Z_mc``, ``Z_mean``, utility, or cache age —
gradients come only from FRESH encodings of the query, positive, and mined negative
texts, exactly as in sequential Fast-GRASS.

Loop (async_fast_grass_implementation_details.md, "Trainer Loop")::

    load initial_data
    while global_step < max_steps:
        every ready_poll_steps: swap to the newest ready round (skip older ones)
        fresh-encode query / positive / mined negatives -> InfoNCE -> step
        every async_mine_every_steps: save a checkpoint, optimizer.pt LAST

A swap rebuilds ONLY the dataloader. The optimizer, scheduler and ``global_step``
are untouched, so training is continuous across rounds. If the miner is late the
trainer simply keeps using the current round and ``data_age_steps`` grows; if
several rounds are ready it jumps to the newest and counts the rest as skipped.

``optimizer.pt`` is written last because it is the miner's validity flag
(``is_valid_checkpoint``) — a checkpoint is invisible until it is complete.

Mined records are docid-only (`{query_id, query, pos_docid, neg_docids}`); passage
text is resolved locally through the same ``corpus_lookup`` the miner used, so
rounds stay small over the full mixture.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import torch
from torch.optim import AdamW
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup

try:
    import bitsandbytes as bnb
    _BNB_AVAILABLE = True
except ImportError:
    _BNB_AVAILABLE = False

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root / 'src'))
sys.path.insert(0, str(project_root / 'scripts'))

from utils.helpers import (  # noqa: E402
    get_training_context, load_config, set_seed, encode_batch_tensor,
    _load_corpus_lookup,
)
from async_fast_grass_cached_mcdp import build_async_cfg  # noqa: E402
from async_fast_grass_handoff import (  # noqa: E402
    latest_committed_round, resolve_training_data, read_meta,
)


def _log(msg):
    print(f"[Trainer] {msg}", flush=True)


class MinedRoundValidationError(RuntimeError):
    """A mined round is unusable. Raised before the trainer switches to it."""


class MinedRoundDataset(Dataset):
    """Mined docid-only JSONL -> (query, positive text, negative texts).

    STRICT by design. Every record must parse, carry a positive and at least ``m``
    negatives, and every docid must resolve in ``corpus_lookup``. A missing docid is
    an error, never an empty string: substituting '' would train the model to pull
    queries toward an empty passage, and the loss would look completely healthy
    while doing it. The miner resolves docids through the SAME lookup, so a miss
    means the round and the corpus have genuinely diverged.
    """

    def __init__(self, data_dir, corpus_lookup, m, min_batch=1):
        self.items = []
        self.corpus_lookup = corpus_lookup
        self.m = m
        data_dir = Path(data_dir)
        files = [f for f in sorted(data_dir.glob("*.jsonl"))
                 if not f.name.startswith('.')]
        if not files:
            raise MinedRoundValidationError(f"no *.jsonl files in {data_dir}")

        for f_path in files:
            with open(f_path) as f:
                for lineno, line in enumerate(f, 1):
                    if not line.strip():
                        continue
                    where = f"{f_path.name}:{lineno}"
                    try:
                        d = json.loads(line)
                    except json.JSONDecodeError as e:
                        raise MinedRoundValidationError(
                            f"{where}: malformed JSON ({e})") from e
                    for key in ('query_id', 'query', 'pos_docid', 'neg_docids'):
                        if key not in d:
                            raise MinedRoundValidationError(
                                f"{where}: record is missing {key!r}")
                    negs = d['neg_docids']
                    if not isinstance(negs, list):
                        raise MinedRoundValidationError(
                            f"{where}: neg_docids must be a list, got {type(negs).__name__}")
                    if d['pos_docid'] is None:
                        raise MinedRoundValidationError(f"{where}: pos_docid is null")
                    if len(negs) < m:
                        raise MinedRoundValidationError(
                            f"{where}: {len(negs)} negatives, need m={m}")
                    for docid in [d['pos_docid']] + list(negs[:m]):
                        if docid not in corpus_lookup:
                            raise MinedRoundValidationError(
                                f"{where}: docid {docid!r} is not in the corpus — "
                                f"the mined round and the corpus have diverged")
                    self.items.append((d['query'], d['pos_docid'], list(negs[:m])))

        if len(self.items) < min_batch:
            raise MinedRoundValidationError(
                f"{data_dir} yielded {len(self.items)} usable records, fewer than "
                f"one batch ({min_batch}) — the dataloader would emit nothing with "
                f"drop_last=True and the trainer would spin")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        query, pos_docid, neg_docids = self.items[i]
        # strict: validated at load, so a KeyError here is a real invariant breach
        return {
            'query': query,
            'positive': self.corpus_lookup[pos_docid],
            'negatives': [self.corpus_lookup[d] for d in neg_docids],
        }


def _collate(batch):
    return {
        'queries': [b['query'] for b in batch],
        'positives': [b['positive'] for b in batch],
        'negatives': [b['negatives'] for b in batch],
    }


def make_dataloader(data_dir, corpus_lookup, m, batch_size, num_workers):
    """Build a loader, validating the round first. Raises MinedRoundValidationError."""
    ds = MinedRoundDataset(data_dir, corpus_lookup, m, min_batch=batch_size)
    return DataLoader(ds, batch_size=batch_size, shuffle=True,
                      num_workers=num_workers, drop_last=True,
                      collate_fn=_collate)


def _resolve_bootstrap_step(cfg, cli_value, mine_every):
    """Resolve ``bootstrap_checkpoint_step``; 0/absent means disabled.

    A nonzero value outside ``0 < step < mine_every`` RAISES rather than being
    warned-about and ignored. Silently disabling a knob the operator explicitly set
    would produce a run labelled "bootstrapped" that never wrote the extra
    checkpoint — and because the bootstrap round mutates the persisted cache, that
    mislabels an ablation arm rather than merely wasting a flag.
    """
    raw = cli_value if cli_value is not None else cfg.get('bootstrap_checkpoint_step', 0)
    step = int(raw or 0)
    if step == 0:
        return 0
    if step < 0:
        raise ValueError(
            f"bootstrap_checkpoint_step must be >= 0, got {step}")
    if step >= mine_every:
        # >= would collide with the regular cadence and save the same step twice
        raise ValueError(
            f"bootstrap_checkpoint_step={step} must be < async_mine_every_steps="
            f"{mine_every}; at or above it the regular cadence already checkpoints "
            f"that step. Use 0 to disable.")
    return step


def summarize_round_consumption(records, final_step):
    """How many optimizer steps was each consumed round actually active for?

    ``records`` are the swap events in order, each ``{round_no, consume_step,
    source_checkpoint_step}``. A round stays active until the next swap, so its span is
    the next ``consume_step`` (or ``final_step`` for the last one).

    This is what the run validity gate reads: a round that was published but consumed
    for two steps before being replaced did not shape the weights, and a run resting on
    that is not evidence about lambda. Extracted as a pure function so the test grades
    production code instead of a reimplementation.
    """
    recs = sorted(records, key=lambda r: int(r['consume_step']))
    out = []
    for i, r in enumerate(recs):
        start = int(r['consume_step'])
        end = int(recs[i + 1]['consume_step']) if i + 1 < len(recs) else int(final_step)
        out.append({
            'round_no': int(r['round_no']),
            'consume_step': start,
            'source_checkpoint_step': int(r.get('source_checkpoint_step', 0)),
            'steps_active': max(end - start, 0),
            'async_gap_steps': start - int(r.get('source_checkpoint_step', 0)),
        })
    return out


def should_checkpoint(step, mine_every, max_steps, bootstrap_step=0):
    """Does the trainer write a checkpoint at ``step``? Single source of truth."""
    return (step % mine_every == 0
            or step == max_steps
            or (bootstrap_step and step == bootstrap_step))


def save_checkpoint(model, tokenizer, optimizer, scheduler, output_dir, step):
    """ANCE-style checkpoint. ``optimizer.pt`` LAST — it is the validity flag."""
    ckpt = Path(output_dir) / f"checkpoint-{step}"
    ckpt.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    raw = getattr(model, '_orig_mod', model)      # unwrap torch.compile
    raw.save_pretrained(str(ckpt))
    tokenizer.save_pretrained(str(ckpt))
    torch.save(scheduler.state_dict(), ckpt / "scheduler.pt")
    torch.save(optimizer.state_dict(), ckpt / "optimizer.pt")   # LAST
    return time.perf_counter() - t0


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--async_dir', required=True)
    ap.add_argument('--output_dir', required=True)
    ap.add_argument('--corpus_file', required=True)
    ap.add_argument('--model_name_or_path', default=None)
    ap.add_argument('--max_steps', type=int, required=True)
    ap.add_argument('--steps_per_epoch', type=int, required=True,
                    help='canonical value from the orchestrator; all three '
                         'processes must agree (see cached_mcdp.steps_per_epoch)')
    ap.add_argument('--recipe', default='async_fast_grass')
    ap.add_argument('--no_compile', action='store_true')
    ap.add_argument('--bootstrap_checkpoint_step', type=int, default=None,
                    help='override training.async_fast_grass.bootstrap_checkpoint_step. '
                         '0 disables. See _resolve_bootstrap_step.')
    args = ap.parse_args()

    from models.temperature_scaled_loss import TemperatureScaledContrastiveLoss

    config = load_config()
    ctx = get_training_context(args.recipe)
    set_seed(config.get('seed', 42))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    root = Path(args.async_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    corpus_lookup = _load_corpus_lookup(args.corpus_file)
    # steps_per_epoch is PASSED by the orchestrator, not reconstructed from
    # max_steps: all three processes must agree on it exactly (the miner divides by
    # it in the maintenance budget), and integer division would not round-trip.
    cfg = build_async_cfg(config, ctx, steps_per_epoch=args.steps_per_epoch)

    m = cfg['m']
    batch_size = cfg.get('batch_size', 64)
    enc_bs = cfg.get('mc_batch_size', 512)
    q_max, p_max = cfg['query_max_len'], cfg['passage_max_len']
    lr = float(cfg['learning_rate'])
    ready_poll_steps = cfg.get('ready_poll_steps', cfg.get('logging_steps', 100))
    mine_every = cfg.get('async_mine_every_steps', 1000)
    logging_steps = cfg.get('logging_steps', 100)
    # raises on a nonzero-but-invalid value; 0 keeps behaviour bit-identical
    bootstrap_step = _resolve_bootstrap_step(
        cfg, args.bootstrap_checkpoint_step, mine_every)

    base_model = args.model_name_or_path or ctx['base_model']
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    student = AutoModel.from_pretrained(
        base_model,
        torch_dtype=torch.bfloat16 if device.type == 'cuda' else torch.float32
    ).to(device)

    if _BNB_AVAILABLE and device.type == 'cuda':
        optimizer = bnb.optim.AdamW8bit(student.parameters(), lr=lr,
                                        weight_decay=cfg.get('weight_decay', 0.01))
        _log("AdamW8bit enabled")
    else:
        if hasattr(student, 'gradient_checkpointing_enable'):
            try:
                student.gradient_checkpointing_enable()
            except Exception:
                pass
        optimizer = AdamW(student.parameters(), lr=lr,
                          weight_decay=cfg.get('weight_decay', 0.01))
    scheduler = get_linear_schedule_with_warmup(
        optimizer, int(cfg.get('warmup_ratio', 0.1) * args.max_steps), args.max_steps)
    loss_fn = TemperatureScaledContrastiveLoss(temperature=ctx['temperature'])
    _student_raw = student
    if not args.no_compile and device.type == 'cuda':
        try:
            torch._dynamo.config.suppress_errors = True
            student = torch.compile(student, dynamic=True)
            _log("torch.compile enabled")
        except Exception as e:
            _log(f"torch.compile skipped ({e})")

    # --- start on the step-0 input ---
    active_round = 0
    loader = make_dataloader(resolve_training_data(root, 0), corpus_lookup, m,
                            batch_size, cfg.get('dataloader_num_workers', 2))
    data_iter = iter(loader)
    consume_step, source_checkpoint_step = 0, 0
    rounds_consumed = rounds_skipped = 0
    rejected_rounds = set()      # validated-and-failed; never retried
    # swap log -> async_trainer_summary.json; the run validity gate needs to know how
    # long each mined round was actually trained on, not merely that it was published.
    swap_records = [{'round_no': 0, 'consume_step': 0, 'source_checkpoint_step': 0}]
    _log(f"initial_data: {len(loader.dataset):,} examples | max_steps="
         f"{args.max_steps:,} | poll every {ready_poll_steps} | checkpoint every "
         f"{mine_every} | bootstrap_checkpoint_step="
         f"{bootstrap_step if bootstrap_step else 'off'}")

    global_step = 0
    running_loss = 0.0
    while global_step < args.max_steps:
        # --- maybe swap to the newest ready round (dataloader ONLY) ---
        if global_step % ready_poll_steps == 0:
            latest = latest_committed_round(root)
            if latest > active_round and latest not in rejected_rounds:
                # Build and validate the NEW loader before touching any state: a
                # malformed round must not take the trainer down mid-run, and must
                # not leave it with a half-swapped dataloader either.
                try:
                    new_loader = make_dataloader(
                        resolve_training_data(root, latest), corpus_lookup, m,
                        batch_size, cfg.get('dataloader_num_workers', 2))
                except MinedRoundValidationError as e:
                    rejected_rounds.add(latest)
                    _log(f"REJECTED round {latest}: {e} — staying on round "
                         f"{active_round}")
                else:
                    skipped = max(latest - active_round - 1, 0)
                    rounds_skipped += skipped
                    meta = read_meta(root, latest)
                    source_checkpoint_step = int(meta.get('source_checkpoint_step', 0))
                    consume_step = global_step
                    # optimizer / scheduler / global_step deliberately untouched
                    loader = new_loader
                    data_iter = iter(loader)
                    active_round = latest
                    rounds_consumed += 1
                    swap_records.append({
                        'round_no': latest, 'consume_step': global_step,
                        'source_checkpoint_step': source_checkpoint_step})
                    _log(f"SWAP -> round {latest} at step {global_step} | "
                         f"source_checkpoint_step={source_checkpoint_step} | "
                         f"async_gap_steps={global_step - source_checkpoint_step} | "
                         f"skipped={skipped} | rounds_consumed={rounds_consumed} "
                         f"rounds_skipped={rounds_skipped} | "
                         f"{len(loader.dataset):,} examples")

        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            batch = next(data_iter)

        # --- fresh-loss step: nothing from the miner's embeddings enters here ---
        student.train()
        q_embs = encode_batch_tensor(student, tokenizer, batch['queries'], device,
                                     q_max, enc_bs, requires_grad=True)
        d_texts = [t for pos, negs in zip(batch['positives'], batch['negatives'])
                   for t in [pos] + negs]
        d_embs = encode_batch_tensor(student, tokenizer, d_texts, device, p_max,
                                     enc_bs, requires_grad=True)
        loss = loss_fn(q_embs, d_embs)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        clip_grad_norm_(student.parameters(), cfg.get('max_grad_norm', 1.0))
        optimizer.step()
        scheduler.step()
        global_step += 1
        running_loss += loss.item()

        if global_step % logging_steps == 0:
            _log(f"step={global_step}/{args.max_steps} "
                 f"loss={running_loss / logging_steps:.4f} "
                 f"round={active_round} "
                 f"data_age_steps={global_step - consume_step} "
                 f"async_gap_steps={consume_step - source_checkpoint_step}")
            running_loss = 0.0

        if should_checkpoint(global_step, mine_every, args.max_steps, bootstrap_step):
            dt = save_checkpoint(_student_raw, tokenizer, optimizer, scheduler,
                                 output_dir, global_step)
            _log(f"saved checkpoint-{global_step} (checkpoint_write_time={dt:.1f}s)")

    # final weights next to the checkpoints, for eval
    raw = getattr(student, '_orig_mod', student)
    raw.save_pretrained(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    rounds = summarize_round_consumption(swap_records, global_step)
    (output_dir / "async_trainer_summary.json").write_text(json.dumps({
        'final_step': global_step,
        'max_steps': args.max_steps,
        'steps_per_epoch': args.steps_per_epoch,
        'rounds_consumed': rounds_consumed,
        'rounds_skipped': rounds_skipped,
        'rejected_rounds': sorted(rejected_rounds),
        'rounds': rounds,
    }, indent=2))
    for r in rounds:
        _log(f"round {r['round_no']}: consumed at step {r['consume_step']} for "
             f"{r['steps_active']} steps (async_gap_steps={r['async_gap_steps']})")
    _log(f"done at step {global_step} | rounds_consumed={rounds_consumed} "
         f"rounds_skipped={rounds_skipped}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
