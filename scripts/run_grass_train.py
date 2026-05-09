"""
GRASS Async Trainer — runs on GPU 0 in parallel with the miner.
Trains on mixture negatives, polling neg_update_dir for miner's hard negatives.
Saves checkpoints with optimizer.pt written last (miner validity gate).
"""
import gc
import json
import random
import time
import sys
import argparse
import numpy as np
import torch
from pathlib import Path
from torch.optim import AdamW
from torch.nn.utils import clip_grad_norm_
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup

try:
    import bitsandbytes as bnb
    _BNB_AVAILABLE = True
except ImportError:
    _BNB_AVAILABLE = False

project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root / 'src'))

from utils.helpers import (
    get_path, get_training_context, load_config,
    encode_batch, get_latest_marker_no, _load_corpus_lookup,
)


def encode_batch_train(model, tokenizer, texts, device, max_len, batch_size):
    """Gradient-tracked forward pass for training (no no_grad wrapper)."""
    all_embs = []
    for i in range(0, len(texts), batch_size):
        batch  = texts[i:i + batch_size]
        inputs = tokenizer(batch, padding=True, truncation=True,
                           max_length=max_len, return_tensors='pt').to(device)
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            out = model(**inputs)
        embs = out.last_hidden_state[:, 0, :]
        embs = torch.nn.functional.normalize(embs, dim=-1)
        all_embs.append(embs)
    return torch.cat(all_embs, dim=0)


def _apply_pending_neg_updates(neg_update_dir, neg_cache, last_update_no):
    """Apply all pending miner updates to neg_cache. Returns (latest_no, n_applied)."""
    latest    = get_latest_marker_no(neg_update_dir, prefix="ready_")
    n_applied = 0
    for n in range(last_update_no + 1, latest + 1):
        if not (neg_update_dir / f"ready_{n}").exists():
            continue
        jsonl = neg_update_dir / f"update_{n}.jsonl"
        if not jsonl.exists():
            continue
        with open(jsonl) as f:
            for line in f:
                d = json.loads(line)
                neg_cache[str(d['query_id'])] = d['neg_docid']
                n_applied += 1
    return latest, n_applied


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_model_dir', required=True)
    parser.add_argument('--neg_update_dir',   required=True)
    parser.add_argument('--corpus_file',      required=True)
    parser.add_argument('--recipe',           default='grass')
    args = parser.parse_args()

    from models.temperature_scaled_loss import TemperatureScaledContrastiveLoss

    config = load_config()
    cfg    = config['training'][args.recipe]
    ctx    = get_training_context(args.recipe)

    lr             = float(cfg['learning_rate'])
    num_epochs     = cfg['num_epochs']
    batch_size     = cfg.get('batch_size', 64)
    m              = cfg['m']
    max_grad_norm  = cfg.get('max_grad_norm', 1.0)
    warmup_ratio   = cfg.get('warmup_ratio', 0.1)
    weight_decay   = cfg.get('weight_decay', 0.01)
    logging_steps  = cfg.get('logging_steps', 100)
    save_steps     = cfg.get('save_steps', 500)
    mc_batch_size  = cfg.get('mc_batch_size', 256)
    neg_poll_steps = cfg.get('trainer_neg_poll_steps', 10)
    q_max_len      = config['model']['query_max_len']
    p_max_len      = config['model']['passage_max_len']
    temperature    = ctx['temperature']
    base_model     = cfg['base_model']

    neg_update_dir   = Path(args.neg_update_dir)
    output_model_dir = Path(args.output_model_dir)
    output_model_dir.mkdir(parents=True, exist_ok=True)

    device    = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    model     = AutoModel.from_pretrained(base_model, torch_dtype=torch.bfloat16).to(device)

    if _BNB_AVAILABLE:
        optimizer = bnb.optim.AdamW8bit(model.parameters(), lr=lr, weight_decay=weight_decay)
        print("[Trainer] AdamW8bit enabled", flush=True)
    else:
        model.gradient_checkpointing_enable()
        optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        print("[Trainer] AdamW + gradient checkpointing", flush=True)
    model.train()

    loss_fn    = TemperatureScaledContrastiveLoss(temperature=temperature)
    _model_raw = model
    try:
        model = torch.compile(model, dynamic=True)
        print("[Trainer] torch.compile enabled", flush=True)
    except Exception as e:
        print(f"[Trainer] torch.compile skipped ({e})", flush=True)

    print("[Trainer] Loading corpus...", flush=True)
    corpus_lookup = _load_corpus_lookup(args.corpus_file)

    # Load training data from mixture dir
    mix_dir     = get_path("processed") / "training_mixture"
    train_items = []
    for f_path in sorted(mix_dir.glob("*.jsonl")):
        if f_path.name.startswith('.'):
            continue
        with open(f_path) as f:
            for line in f:
                d    = json.loads(line)
                pos  = d.get('positive_passages', [])
                if not pos:
                    continue
                negs = d.get('negative_passages', [])
                train_items.append({
                    'query_id':  str(d['query_id']),
                    'query':     d['query'],
                    'pos_docid': pos[0]['docid'],
                    'neg_docid': negs[0]['docid'] if negs else None,
                })
    random.shuffle(train_items)
    print(f"[Trainer] {len(train_items)} training examples", flush=True)

    # Init neg_cache from mixture negatives; miner will progressively replace them
    neg_cache = {it['query_id']: it['neg_docid'] for it in train_items if it['neg_docid']}
    last_update_no = 0

    n_batches    = len(train_items) // batch_size
    total_steps  = n_batches * num_epochs
    warmup_steps = int(warmup_ratio * total_steps)
    scheduler    = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    print(f"[Trainer] {total_steps} total steps | batch={batch_size} | "
          f"neg_poll_steps={neg_poll_steps}", flush=True)

    global_step = 0
    t_start     = time.time()

    for epoch in range(num_epochs):
        random.shuffle(train_items)
        epoch_loss = 0.0

        for b in range(n_batches):
            # Poll for new negatives from miner
            if global_step % neg_poll_steps == 0 and neg_update_dir.exists():
                last_update_no, n_applied = _apply_pending_neg_updates(
                    neg_update_dir, neg_cache, last_update_no
                )
                if n_applied > 0:
                    print(f"[Trainer] step={global_step}: applied {n_applied} neg updates "
                          f"(total={last_update_no})", flush=True)

            batch_items = train_items[b * batch_size:(b + 1) * batch_size]

            # Build text lists (skip queries with no negative available)
            queries, positives, negatives = [], [], []
            for item in batch_items:
                neg_docid = neg_cache.get(item['query_id'])
                if not neg_docid:
                    continue
                queries.append(item['query'])
                positives.append(corpus_lookup.get(item['pos_docid'], ''))
                negatives.append([corpus_lookup.get(neg_docid, '')])
            if not queries:
                continue

            # Forward with gradients
            model.train()
            q_embs  = encode_batch_train(model, tokenizer, queries,
                                         device, q_max_len, mc_batch_size)
            d_texts = [t for pos, negs in zip(positives, negatives) for t in [pos] + negs]
            d_embs  = encode_batch_train(model, tokenizer, d_texts,
                                          device, p_max_len, mc_batch_size)
            loss = loss_fn(q_embs, d_embs)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
            scheduler.step()

            epoch_loss  += loss.item()
            global_step += 1

            if global_step % logging_steps == 0:
                elapsed   = time.time() - t_start
                secs_per  = elapsed / global_step
                remaining = secs_per * (total_steps - global_step)
                eta = f"{int(remaining // 3600)}h {int((remaining % 3600) // 60)}m"
                print(f"[Trainer] Ep{epoch+1} step {b+1}/{n_batches} | "
                      f"loss={loss.item():.4f} | ETA {eta}", flush=True)

            # Save checkpoint — optimizer.pt written LAST as validity gate for miner
            if global_step % save_steps == 0:
                ckpt = output_model_dir / f"checkpoint-{global_step}"
                ckpt.mkdir(parents=True, exist_ok=True)
                _model_raw.save_pretrained(str(ckpt))
                tokenizer.save_pretrained(str(ckpt))
                torch.save(scheduler.state_dict(), ckpt / "scheduler.pt")
                torch.save(optimizer.state_dict(), ckpt / "optimizer.pt")
                print(f"[Trainer] Checkpoint saved: {ckpt.name}", flush=True)

        print(f"[Trainer] Epoch {epoch+1} done. avg_loss={epoch_loss / n_batches:.4f}", flush=True)

    _model_raw.save_pretrained(str(output_model_dir))
    tokenizer.save_pretrained(str(output_model_dir))
    print(f"[Trainer] Training complete. Model at {output_model_dir}", flush=True)

    del model, _model_raw
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
