"""GRASS async v2 trainer (GPU 0).

Custom Tevatron training loop with periodic neg-update polling, mirroring
scripts/run_ance_train.py.

Per-round hot-swap: every poll_interval_steps, check update_dir for the
latest ready_N marker. If N > last_applied, rebuild the DataLoader from
update_dir/training_data_N/*.jsonl. Optimiser, scheduler, and global_step
are preserved across swaps.

Checkpoint writes save optimizer.pt LAST — the miner's validity gate.
"""

import argparse
import json
import os
import sys
from pathlib import Path

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

os.environ["TRANSFORMERS_ATTENTION_IMPLEMENTATION"] = "eager"
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root / 'src'))

from utils.helpers import get_training_context, load_config, get_latest_marker_no


class _AsyncV2Dataset(Dataset):
    """Load JSONL training examples (one record per line) from a directory."""
    def __init__(self, data_dir: Path, tokenizer, max_q_len, max_p_len, train_group_size):
        self.examples = []
        for f_path in sorted(Path(data_dir).glob("*.jsonl")):
            with open(f_path) as f:
                for line in f:
                    if line.strip():
                        self.examples.append(json.loads(line))
        self.tokenizer        = tokenizer
        self.max_q_len        = max_q_len
        self.max_p_len        = max_p_len
        self.train_group_size = train_group_size

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        ex = self.examples[idx]
        passages = ex['positive_passages'][:1] + ex['negative_passages'][:self.train_group_size - 1]
        while len(passages) < self.train_group_size:
            passages.append(passages[-1])

        q  = self.tokenizer(ex['query'], max_length=self.max_q_len,
                            truncation=True, padding='max_length', return_tensors='pt')
        ps = [self.tokenizer(p['text'], max_length=self.max_p_len,
                             truncation=True, padding='max_length', return_tensors='pt')
              for p in passages]
        return {
            'q_input_ids':      q['input_ids'].squeeze(0),
            'q_attention_mask': q['attention_mask'].squeeze(0),
            'p_input_ids':      torch.stack([p['input_ids'].squeeze(0)      for p in ps]),
            'p_attention_mask': torch.stack([p['attention_mask'].squeeze(0) for p in ps]),
        }


def _make_dataloader(data_dir, tokenizer, ctx, config, batch_size):
    ds = _AsyncV2Dataset(
        data_dir, tokenizer,
        config['model']['query_max_len'],
        config['model']['passage_max_len'],
        ctx['args']['train_group_size'],
    )
    return DataLoader(
        ds, batch_size=batch_size, shuffle=True,
        num_workers=ctx['args']['dataloader_num_workers'], drop_last=True,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name_or_path', required=True)
    parser.add_argument('--initial_data_dir',   required=True)
    parser.add_argument('--update_dir',         required=True)
    parser.add_argument('--output_dir',         required=True)
    parser.add_argument('--max_steps',          type=int, required=True)
    parser.add_argument('--recipe',             default='grass')
    args = parser.parse_args()

    ctx    = get_training_context(args.recipe)
    config = load_config()
    cfg    = config['training'][args.recipe]
    cfg_v2 = cfg['async_v2']

    from tevatron.retriever.modeling import DenseModel
    from tevatron.retriever.arguments import ModelArguments, TevatronTrainingArguments as TrainingArguments

    if not hasattr(DenseModel, "_keys_to_ignore_on_save"):
        setattr(DenseModel, "_keys_to_ignore_on_save", None)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)

    model_args = ModelArguments(
        model_name_or_path=args.model_name_or_path,
        pooling=ctx['pooling'],
        normalize=ctx['normalize'],
        temperature=ctx['temperature'],
        attn_implementation='eager',
    )
    train_args = TrainingArguments(output_dir=args.output_dir, bf16=True)
    model = DenseModel.build(
        model_args, train_args,
        attn_implementation='eager',
    ).cuda()

    batch_size          = cfg['batch_size']
    logging_steps       = cfg.get('logging_steps', 100)
    save_steps          = int(cfg_v2.get('save_steps', 1000))
    poll_interval_steps = int(cfg_v2.get('poll_interval_steps', 10))

    update_dir = Path(args.update_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    print(f"[Trainer] Loading initial data from {args.initial_data_dir}", flush=True)
    train_dataloader = _make_dataloader(Path(args.initial_data_dir), tokenizer, ctx, config, batch_size)
    train_iter       = iter(train_dataloader)
    last_update_no   = 0

    optimizer    = AdamW(model.parameters(),
                         lr=float(cfg['learning_rate']),
                         weight_decay=float(cfg.get('weight_decay', 0.0)))
    warmup_steps = int(args.max_steps * cfg.get('warmup_ratio', 0.1))
    scheduler    = get_linear_schedule_with_warmup(optimizer, warmup_steps, args.max_steps)

    global_step = 0
    model.train()
    print(f"[Trainer] Starting: max_steps={args.max_steps}, "
          f"poll_interval_steps={poll_interval_steps}, save_steps={save_steps}", flush=True)

    while global_step < args.max_steps:

        # ── Check for new miner update every poll_interval_steps ──────────
        if global_step > 0 and global_step % poll_interval_steps == 0:
            latest = get_latest_marker_no(update_dir, prefix="ready_")
            if latest > last_update_no:
                new_dir = update_dir / f"training_data_{latest}"
                if new_dir.exists() and list(new_dir.glob("*.jsonl")):
                    print(f"[Trainer] step={global_step}: update #{latest} ready → "
                          f"swapping DataLoader", flush=True)
                    train_dataloader = _make_dataloader(new_dir, tokenizer, ctx, config, batch_size)
                    train_iter       = iter(train_dataloader)
                    last_update_no   = latest

        # ── Fetch batch, cycle on exhaustion ──────────────────────────────
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_dataloader)
            batch      = next(train_iter)

        batch = {k: v.cuda() for k, v in batch.items()}
        B, G, L = batch['p_input_ids'].shape

        with torch.autocast('cuda', dtype=torch.bfloat16):
            outputs = model(
                query={
                    'input_ids':      batch['q_input_ids'],
                    'attention_mask': batch['q_attention_mask'],
                },
                passage={
                    'input_ids':      batch['p_input_ids'].view(B * G, L),
                    'attention_mask': batch['p_attention_mask'].view(B * G, L),
                },
            )
            loss = outputs.loss
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.get('max_grad_norm', 1.0))
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)
        global_step += 1

        if global_step % logging_steps == 0:
            print(f"[Trainer] step={global_step}/{args.max_steps} loss={loss.item():.4f} "
                  f"update_no={last_update_no}", flush=True)

        # ── Save checkpoint (optimizer.pt last = miner's validity gate) ───
        if global_step % save_steps == 0 or global_step == args.max_steps:
            ckpt = output_dir / f"checkpoint-{global_step}"
            ckpt.mkdir(exist_ok=True)
            model.save(str(ckpt))                                  # EncoderModel.save → save_pretrained
            tokenizer.save_pretrained(str(ckpt))
            torch.save(scheduler.state_dict(), ckpt / "scheduler.pt")
            torch.save(optimizer.state_dict(), ckpt / "optimizer.pt")  # LAST
            print(f"[Trainer] Saved checkpoint-{global_step}", flush=True)

    # Final model save
    model.save(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    print("[Trainer] Training complete.", flush=True)


if __name__ == "__main__":
    main()
