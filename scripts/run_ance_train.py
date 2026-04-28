"""
ANCE Trainer — custom training loop, runs on GPU 0.
Mirrors run_ann.py's while-loop design: polls ann_dir at logging_steps, swaps DataLoader in-place.
Training NEVER stops between ANN refreshes (paper Figure 2, Appendix A.1).

Paper reference: Section 4, Figure 2, Appendix A.1
"""
import os
import sys
import json
import argparse
import torch
from torch.utils.data import DataLoader, Dataset
from pathlib import Path
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
from torch.optim import AdamW

project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root / 'src'))

from utils.helpers import get_training_context, load_config

os.environ["TRANSFORMERS_ATTENTION_IMPLEMENTATION"] = "eager"


def get_latest_ann_data(ann_dir: Path):
    """Return (ann_no, data_dir) for the most recent completed ANN data (ready_{N} marker exists)."""
    ready_files = list(ann_dir.glob("ready_*"))
    if not ready_files:
        return 0, None
    nos = [int(f.name.split("_")[1]) for f in ready_files if f.name.split("_")[1].isdigit()]
    latest = max(nos)
    return latest, ann_dir / f"training_data_{latest}"


class ANCEDataset(Dataset):
    """Load JSONL training examples from a directory."""
    def __init__(self, data_dir: Path, tokenizer, max_q_len, max_p_len, train_group_size):
        self.examples = []
        for f_path in sorted(data_dir.glob("*.jsonl")):
            with open(f_path) as f:
                for line in f:
                    if line.strip():
                        self.examples.append(json.loads(line))
        self.tokenizer = tokenizer
        self.max_q_len = max_q_len
        self.max_p_len = max_p_len
        self.train_group_size = train_group_size

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        ex = self.examples[idx]
        passages = ex['positive_passages'][:1] + ex['negative_passages'][:self.train_group_size - 1]
        while len(passages) < self.train_group_size:
            passages.append(passages[-1])  # pad if insufficient negatives

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


def make_dataloader(data_dir, tokenizer, ctx, config, batch_size):
    ds = ANCEDataset(data_dir, tokenizer,
                     config['model']['query_max_len'],
                     config['model']['passage_max_len'],
                     ctx['args']['train_group_size'])
    return DataLoader(ds, batch_size=batch_size, shuffle=True,
                      num_workers=ctx['args']['dataloader_num_workers'], drop_last=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name_or_path', required=True)
    parser.add_argument('--initial_data_dir',   required=True)
    parser.add_argument('--ann_dir',            required=True)
    parser.add_argument('--output_dir',         required=True)
    parser.add_argument('--max_steps',          type=int, required=True)
    parser.add_argument('--recipe',             default='ance')
    args = parser.parse_args()

    ctx = get_training_context(args.recipe)
    config = load_config()
    from utils.helpers import set_seed
    set_seed(config.get('seed', 42))
    # Note: temperature scaling is handled by ModelArguments.temperature → DenseModel.self.temperature
    # (encoder.py line 70: loss = self.compute_loss(scores / self.temperature, target))
    # patch_tevatron_loss is NOT called here — it patches gc_trainer which we don't use

    from tevatron.retriever.modeling import DenseModel
    from tevatron.retriever.arguments import ModelArguments, TevatronTrainingArguments as TrainingArguments

    if not hasattr(DenseModel, "_keys_to_ignore_on_save"):
        setattr(DenseModel, "_keys_to_ignore_on_save", None)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)

    # DenseModel.forward() calls self.encoder(**query, ...) so query/passage must be plain dicts.
    # attn_implementation must be passed as hf_kwarg — ModelArguments.attn_implementation is NOT
    # auto-forwarded to AutoModel.from_pretrained(); must be explicit.
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

    batch_size    = ctx['args']['batch_size']
    logging_steps = ctx['args']['logging_steps']       # check for new ANN data at this interval
    save_steps    = ctx['args'].get('save_steps', 1000) # paper: refresh every ~10k batches (scaled)
    ann_dir       = Path(args.ann_dir)
    output_dir    = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    # Load initial dataset (mined from base model before training starts)
    print(f"[Trainer] Loading initial data from {args.initial_data_dir}", flush=True)
    train_dataloader = make_dataloader(Path(args.initial_data_dir), tokenizer, ctx, config, batch_size)
    train_iter  = iter(train_dataloader)
    last_ann_no = 0  # Inferencer writes ready_N; we check for N > last_ann_no

    # Optimizer (paper uses LAMB; AdamW-fused is equivalent for BGE-M3 at batch 64)
    optimizer = AdamW(model.parameters(),
                      lr=float(ctx['args']['learning_rate']),
                      weight_decay=float(ctx['args'].get('weight_decay', 0.0)))
    # Paper: linear warmup (5000 steps at MARCO scale); we scale proportionally with warmup_ratio
    warmup_steps = int(args.max_steps * ctx['args'].get('warmup_ratio', 0.1))
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, args.max_steps)

    global_step = 0
    model.train()
    print(f"[Trainer] Starting: max_steps={args.max_steps}, "
          f"logging_steps={logging_steps}, save_steps={save_steps}", flush=True)

    # ── MAIN TRAINING LOOP (mirrors run_ann.py while-loop) ───────────────────
    while global_step < args.max_steps:

        # ── Check for new ANN data every logging_steps ────────────────────────
        # Paper: "when the new ANN index is ready, it immediately replaces existing negatives"
        if global_step > 0 and global_step % logging_steps == 0:
            ann_no, ann_path = get_latest_ann_data(ann_dir)
            print(f"[Trainer] ANN check at step {global_step}: latest={ann_no}, using={last_ann_no}", flush=True)
            if ann_path is not None and ann_no > last_ann_no:
                print(f"[Trainer] Step {global_step}: ANN #{ann_no} ready — swapping DataLoader",
                      flush=True)
                # In-place DataLoader replacement: training never pauses
                train_dataloader = make_dataloader(ann_path, tokenizer, ctx, config, batch_size)
                train_iter  = iter(train_dataloader)
                last_ann_no = ann_no

        # ── Fetch batch, cycle if dataset exhausted ───────────────────────────
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_dataloader)
            batch = next(train_iter)

        # ── Forward pass via Tevatron DenseModel ──────────────────────────────
        # DenseModel.forward(query, passage) calls self.encoder(**query, ...) — must be plain dicts
        batch = {k: v.cuda() for k, v in batch.items()}
        B, G, L = batch['p_input_ids'].shape  # batch, group_size, seq_len

        with torch.autocast('cuda', dtype=torch.bfloat16):
            outputs = model(
                query={
                    'input_ids':      batch['q_input_ids'],
                    'attention_mask': batch['q_attention_mask'],
                },
                passage={
                    'input_ids':      batch['p_input_ids'].view(B * G, L),
                    'attention_mask': batch['p_attention_mask'].view(B * G, L),
                }
            )
            loss = outputs.loss
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), ctx['args'].get('max_grad_norm', 1.0))
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)
        global_step += 1

        if global_step % logging_steps == 0:
            print(f"[Trainer] step={global_step}/{args.max_steps} loss={loss.item():.4f} "
                  f"ann_no={last_ann_no}", flush=True)

        # ── Save checkpoint (triggers Inferencer to start new ANN generation) ─
        # Paper: "update the ANN index once every m batches, i.e., with checkpoint f_k"
        if global_step % save_steps == 0 or global_step == args.max_steps:
            ckpt = output_dir / f"checkpoint-{global_step}"
            ckpt.mkdir(exist_ok=True)
            model.save(str(ckpt))           # EncoderModel.save() → self.encoder.save_pretrained()
            tokenizer.save_pretrained(str(ckpt))
            torch.save(scheduler.state_dict(), ckpt / "scheduler.pt")
            torch.save(optimizer.state_dict(), ckpt / "optimizer.pt")  # written last: Inferencer validity flag
            print(f"[Trainer] Saved checkpoint-{global_step}", flush=True)

    # Final model save
    model.save(str(output_dir))             # EncoderModel.save() → self.encoder.save_pretrained()
    tokenizer.save_pretrained(str(output_dir))
    print("[Trainer] Training complete.", flush=True)


if __name__ == "__main__":
    main()
