"""Sequential bandit-guided GRASS pipeline (Algorithm 1, plan §3.2).

Workflow per run:
  0. [INIT] grass_sampler(L=1) over all queries → warm-start bandit with g_0
  1. for each epoch:
       - select  = bandit.select_global(coverage * |D|)   (or random.sample if --selection random)
       - mine    = grass_sampler(L=10) on the subset      (writes JSONL, logs g per query)
       - update  = parse mining_log → bandit.update(qid, g)
       - train   = Tevatron 1 epoch using the mined JSONL, resume from previous checkpoint

Each epoch's mined JSONL chains from the previous epoch's via base_jsonl_dir,
so mined hard negatives accumulate across epochs.

CLI flags (see main()):
  --selection {bandit, random}    default bandit
  --coverage  float               default 0.25
  --num_epochs int                default 3
  --model_suffix str              appended to output model name (avoid collisions)
  --lambda_val float              override gap-index λ in cfg (default 1.0)

Algorithm 3 (CaseBandit) lives in src/utils/bandit.py as EpsilonGreedyBandit.
"""

import gc
import json
import random
import sys
import time
from pathlib import Path

import pandas as pd
import torch
from tevatron.retriever.driver.train import main as tevatron_train_main
from tevatron.retriever.modeling import DenseModel

project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root / 'src'))
sys.path.append(str(project_root / 'scripts'))

from utils.helpers import get_path, patch_tevatron_loss
from utils.bandit import EpsilonGreedyBandit
from run_grass_mcd import grass_sampler

# Tevatron Bug Patch (mirrors run_grass_mcd.py)
if not hasattr(DenseModel, "_keys_to_ignore_on_save"):
    setattr(DenseModel, "_keys_to_ignore_on_save", None)


def parse_mining_log(log_path):
    """Parse mining_log.jsonl produced by grass_sampler. Returns {qid: g_selected}."""
    g_values = {}
    with open(log_path) as f:
        for line in f:
            rec = json.loads(line)
            g_values[str(rec['query_id'])] = float(rec['g_selected'])
    return g_values


def run_init_pass(bandit, model_path, stale_idx, stale_embs, c_id_to_idx, c_ids,
                  corpus_lookup, mix_df, qrels_dict, cfg, config, workdir):
    """[INIT] grass_sampler with L=1 over all queries; update bandit with g_0 per query.

    The JSONL files written here are NOT used for training — they exist only because
    grass_sampler always writes them. The init pass exists solely to populate
    bandit.mean_g so the heap is differentiated when epoch 1's select_global is called.
    """
    init_dir = workdir / "init_pass"
    cfg_init = {**cfg, 'L': 1, 'm': 1}  # top-1 candidate, 1 negative
    t0 = time.time()
    print(f"  [INIT] Running grass_sampler with L=1 over {len(mix_df)} queries...", flush=True)
    grass_sampler(model_path, stale_idx, stale_embs, c_id_to_idx, c_ids, corpus_lookup,
                  mix_df, qrels_dict, cfg_init, config, init_dir)
    g_init = parse_mining_log(init_dir / "mining_log.jsonl")
    for qid, g in g_init.items():
        bandit.update(qid, g)
    print(f"  [INIT] Bandit warm-started with g_0 for {len(g_init)} queries "
          f"({(time.time() - t0) / 60:.1f} min)", flush=True)


def train_one_epoch(model_path, jsonl_dir, output_dir, cfg, ctx, config):
    """Tevatron training for exactly one epoch starting from model_path's checkpoint."""
    training_args = [
        '--output_dir',                    str(output_dir),
        '--model_name_or_path',            str(model_path),
        '--dataset_name',                  'json',
        '--dataset_path',                  str(jsonl_dir / "*.jsonl"),
        '--dataset_split',                 'train',
        '--per_device_train_batch_size',   str(cfg['batch_size']),
        '--train_group_size',              str(cfg['train_group_size']),
        '--learning_rate',                 str(cfg['learning_rate']),
        '--num_train_epochs',              '1',
        '--bf16',                          'True',
        '--dtype',                         'bfloat16',
        '--overwrite_output_dir',          'True',
        '--save_strategy',                 cfg['save_strategy'],
        '--save_steps',                    str(cfg.get('save_steps', 1000)),
        '--save_total_limit',              str(cfg['save_total_limit']),
        '--ignore_data_skip',              'True',
        '--warmup_ratio',                  str(cfg.get('warmup_ratio', 0.1)),
        '--weight_decay',                  str(cfg.get('weight_decay', 0.01)),
        '--max_grad_norm',                 str(cfg.get('max_grad_norm', 1.0)),
        '--dataloader_num_workers',        str(cfg['dataloader_num_workers']),
        '--attn_implementation',           'eager',
        '--optim',                         'adamw_torch_fused',
        '--logging_steps',                 str(cfg['logging_steps']),
        '--pooling',                       ctx['pooling'],
        '--normalize',                     str(ctx['normalize']),
        '--temperature',                   str(ctx['temperature']),
        '--seed',                          str(config.get('seed', 42)),
    ]
    sys.argv = ['train.py'] + training_args
    patch_tevatron_loss(ctx['temperature'])
    tevatron_train_main()


def run_seq_bandit_pipeline(stale_idx, stale_embs, c_id_to_idx, c_ids, corpus_lookup,
                            mix_df, qrels_dict, cfg, config, ctx, workdir,
                            selection='bandit', coverage=0.25, num_epochs=3,
                            model_suffix=''):
    """Run sequential bandit pipeline. Returns final model directory path."""
    all_qids   = mix_df['query_id'].astype(str).tolist()
    n_queries  = len(all_qids)
    n_per_epoch = int(coverage * n_queries)
    suffix     = f"_{model_suffix}" if model_suffix else ''
    base_name  = cfg['model_name']

    print(f"  Selection: {selection}", flush=True)
    print(f"  Coverage: {coverage:.2f} → {n_per_epoch}/{n_queries} queries/epoch", flush=True)
    print(f"  Epochs: {num_epochs}", flush=True)
    print(f"  λ (lambda_val): {cfg.get('lambda_val')}", flush=True)
    print(f"  L_main: {cfg.get('L')}, T: {cfg.get('T')}", flush=True)

    # Bandit (always created; only used if selection == 'bandit')
    bandit = EpsilonGreedyBandit(
        epsilon=cfg.get('bandit_epsilon', 0.3),
        alpha=cfg.get('bandit_alpha', 0.5),
    )
    bandit.init_query_pool(all_qids)

    # [INIT] Warm-start bandit (run for both bandit and random — for random it's wasted
    # ~15 min but keeps total wall-clock identical, which matters for paired comparison)
    if selection == 'bandit':
        run_init_pass(bandit, cfg['base_model'], stale_idx, stale_embs, c_id_to_idx, c_ids,
                      corpus_lookup, mix_df, qrels_dict, cfg, config, workdir)
    else:
        print(f"  [INIT] Skipped (selection='random' does not use bandit state)", flush=True)

    current_model      = cfg['base_model']
    prev_mining_dir    = None   # None → grass_sampler uses original training_mixture as base
    final_model_dir    = None

    for epoch in range(1, num_epochs + 1):
        t_epoch = time.time()
        print(f"\n=== Epoch {epoch}/{num_epochs} ({selection}, X={coverage:.2f}) ===", flush=True)

        # 1. Select queries to mine
        if selection == 'bandit':
            selected = set(bandit.select_global(n_per_epoch))
        else:
            selected = set(random.sample(all_qids, n_per_epoch))
        print(f"  Selected {len(selected)} queries", flush=True)

        # 2. Mine the subset (chained JSONL from previous epoch)
        subset_df  = mix_df[mix_df['query_id'].astype(str).isin(selected)].reset_index(drop=True)
        mining_dir = workdir / f"mining_epoch{epoch}{suffix}"
        t_mine = time.time()
        grass_sampler(current_model, stale_idx, stale_embs, c_id_to_idx, c_ids, corpus_lookup,
                      subset_df, qrels_dict, cfg, config, mining_dir,
                      base_jsonl_dir=prev_mining_dir)
        print(f"  Mining: {(time.time() - t_mine) / 60:.1f} min", flush=True)

        # 3. Update bandit from this epoch's mining log
        if selection == 'bandit':
            g_values = parse_mining_log(mining_dir / "mining_log.jsonl")
            for qid, g in g_values.items():
                bandit.update(qid, g)
            print(f"  Bandit updated for {len(g_values)} queries", flush=True)

        # 4. Tevatron 1-epoch training, resume from previous checkpoint
        output_dir = get_path("models") / f"{base_name}_seqbandit{suffix}_e{epoch}"
        t_train = time.time()
        train_one_epoch(current_model, mining_dir, output_dir, cfg, ctx, config)
        print(f"  Training: {(time.time() - t_train) / 60:.1f} min", flush=True)

        current_model   = output_dir
        prev_mining_dir = mining_dir
        final_model_dir = output_dir

        gc.collect()
        torch.cuda.empty_cache()
        print(f"  Epoch {epoch} total: {(time.time() - t_epoch) / 60:.1f} min", flush=True)

    print(f"\n✅ Sequential bandit pipeline done. Final model: {final_model_dir}", flush=True)
    return final_model_dir
