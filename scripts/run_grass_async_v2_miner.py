"""GRASS async v2 miner (GPU 1).

Two modes:
  --init_only : run grass_sampler(L=1) over the FULL query pool; writes
                training_data_0/*.jsonl + bandit_state.pkl. Used by the
                orchestrator before launching the main loop.
  (default)   : poll output_model_dir for newest valid checkpoint, run one
                full bandit round per cycle (X*|D| queries, batched at
                query_batch_size as in run_grass_seq_bandit.py), publish
                training_data_N/*.jsonl + ready_N.

The bandit and GrassSampler internals are reused from src/utils/bandit.py
and scripts/run_grass_mcd.py unchanged.
"""

import argparse
import gc
import json
import os
import pickle
import random
import sys
import time
from pathlib import Path

import pandas as pd
import torch
from tevatron.retriever.modeling import DenseModel

os.environ["TRANSFORMERS_ATTENTION_IMPLEMENTATION"] = "eager"
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root / 'src'))
sys.path.append(str(project_root / 'scripts'))

from utils.helpers import (
    get_path, get_training_context, load_config, build_faiss_index,
    is_valid_checkpoint, set_seed, _load_qrels, _load_corpus_lookup,
)
from utils.bandit import EpsilonGreedyBandit
from run_grass_mcd import grass_sampler

# Tevatron Bug Patch
if not hasattr(DenseModel, "_keys_to_ignore_on_save"):
    setattr(DenseModel, "_keys_to_ignore_on_save", None)


def _parse_mining_log(log_path):
    """Returns {qid: g_selected} from the JSONL emitted by grass_sampler."""
    g_values = {}
    with open(log_path) as f:
        for line in f:
            rec = json.loads(line)
            g_values[str(rec['query_id'])] = float(rec['g_selected'])
    return g_values


def _load_state(stale_pkl, corpus_file, query_file, qrels_file):
    """Load FAISS index + corpus/queries/qrels for both init and main mode."""
    stale_idx, stale_embs, c_ids = build_faiss_index(stale_pkl)
    c_id_to_idx   = {did: i for i, did in enumerate(c_ids)}
    corpus_lookup = _load_corpus_lookup(corpus_file)
    qrels_dict    = _load_qrels(qrels_file)
    mix_df        = pd.read_json(query_file, lines=True)
    return stale_idx, stale_embs, c_ids, c_id_to_idx, corpus_lookup, qrels_dict, mix_df


def _run_init_pass(args, cfg, config, ctx, update_dir, bandit):
    """grass_sampler(L=1, m=1) over the full query pool → training_data_0/ + bandit_state.pkl.

    Writes a full-mixture JSONL (training_data_0/*.jsonl) which the trainer
    uses as its initial dataset. Warm-starts the bandit's mean_g via the
    mining log (same effect as seq_bandit's run_init_pass).
    """
    stale_idx, stale_embs, c_ids, c_id_to_idx, corpus_lookup, qrels_dict, mix_df = \
        _load_state(args.stale_pkl, args.corpus_file, args.query_file, args.qrels_file)

    if args.debug:
        mix_df = mix_df.head(100)
        print("[miner-init] DEBUG: restricting to 100 queries", flush=True)

    bandit.init_query_pool(mix_df['query_id'].astype(str).tolist())

    out_dir  = Path(args.update_dir) / "training_data_0"
    cfg_init = {**cfg, 'L': 1, 'm': 1}
    cfg_init['lambda_val'] = float(args.lambda_val)
    t0 = time.time()
    print(f"[miner-init] grass_sampler(L=1) over {len(mix_df)} queries...", flush=True)
    grass_sampler(cfg['base_model'], stale_idx, stale_embs, c_id_to_idx, c_ids, corpus_lookup,
                  mix_df, qrels_dict, cfg_init, config, out_dir)
    g_init = _parse_mining_log(out_dir / "mining_log.jsonl")
    for qid, g in g_init.items():
        bandit.update(qid, g)
    elapsed_min = (time.time() - t0) / 60.0
    print(f"[miner-init] Bandit warm-started for {len(g_init)} queries ({elapsed_min:.1f} min)",
          flush=True)

    bandit_state = Path(args.update_dir) / "bandit_state.pkl"
    with open(bandit_state, 'wb') as f:
        pickle.dump(bandit, f)
    print(f"[miner-init] Bandit state → {bandit_state}", flush=True)


def _wait_for_new_valid_checkpoint(output_model_dir, last_ckpt, poll_interval):
    """Block until get_last_checkpoint returns a new, fully-written checkpoint.

    Validity gate: trainer writes optimizer.pt last (is_valid_checkpoint).
    """
    from transformers.trainer_utils import get_last_checkpoint
    while True:
        nxt = get_last_checkpoint(str(output_model_dir))
        if nxt is not None and nxt != last_ckpt and is_valid_checkpoint(nxt):
            return nxt
        time.sleep(poll_interval)


def _run_main_loop(args, cfg, config, ctx, update_dir, bandit):
    """Polling loop: each iteration = one full bandit round on the latest checkpoint."""
    stale_idx, stale_embs, c_ids, c_id_to_idx, corpus_lookup, qrels_dict, mix_df = \
        _load_state(args.stale_pkl, args.corpus_file, args.query_file, args.qrels_file)

    if args.debug:
        mix_df = mix_df.head(100)
        print("[miner] DEBUG: restricting to 100 queries", flush=True)

    all_qids = mix_df['query_id'].astype(str).tolist()
    n_per_round = max(1, int(float(args.coverage) * len(all_qids)))
    cfg_round = {**cfg, 'lambda_val': float(args.lambda_val)}

    poll_interval = int(cfg.get('miner_poll_interval', 5))
    selection     = args.selection
    round_num     = 1
    last_ckpt     = None

    print(f"[miner] selection={selection} coverage={args.coverage} "
          f"n_per_round={n_per_round}/{len(all_qids)} lambda={args.lambda_val}", flush=True)

    while True:
        latest = _wait_for_new_valid_checkpoint(args.output_model_dir, last_ckpt, poll_interval)
        ckpt_name = Path(latest).name
        print(f"[miner] Round {round_num}: checkpoint={ckpt_name}", flush=True)

        if selection == 'bandit':
            selected = set(bandit.select_global(n_per_round))
        else:
            selected = set(random.sample(all_qids, n_per_round))
        subset_df = mix_df[mix_df['query_id'].astype(str).isin(selected)].reset_index(drop=True)
        print(f"[miner] Round {round_num}: selected {len(subset_df)} queries", flush=True)

        out_dir       = Path(args.update_dir) / f"training_data_{round_num}"
        base_jsonl_dir = Path(args.update_dir) / f"training_data_{round_num - 1}"

        t_mine = time.time()
        grass_sampler(latest, stale_idx, stale_embs, c_id_to_idx, c_ids, corpus_lookup,
                      subset_df, qrels_dict, cfg_round, config, out_dir,
                      base_jsonl_dir=base_jsonl_dir)
        print(f"[miner] Round {round_num}: mining {(time.time() - t_mine) / 60:.1f} min",
              flush=True)

        if selection == 'bandit':
            g_values = _parse_mining_log(out_dir / "mining_log.jsonl")
            for qid, g in g_values.items():
                bandit.update(qid, g)

        # Write ready marker LAST (trainer's validity gate for this update).
        (Path(args.update_dir) / f"ready_{round_num}").write_text(str(round_num))
        print(f"[miner] Round {round_num} published → ready_{round_num}", flush=True)

        last_ckpt  = latest
        round_num += 1

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--update_dir',        required=True)
    parser.add_argument('--stale_pkl',         required=True)
    parser.add_argument('--corpus_file',       required=True)
    parser.add_argument('--query_file',        required=True)
    parser.add_argument('--qrels_file',        required=True)
    parser.add_argument('--output_model_dir',  required=True)
    parser.add_argument('--recipe',            default='grass')
    parser.add_argument('--coverage',          type=float, default=0.25)
    parser.add_argument('--selection',         choices=['bandit', 'random'], default='bandit')
    parser.add_argument('--lambda_val',        type=float, default=1.0)
    parser.add_argument('--init_only',         action='store_true')
    parser.add_argument('--debug',             action='store_true')
    args = parser.parse_args()

    ctx    = get_training_context(args.recipe)
    config = load_config()
    cfg    = config['training'][args.recipe]
    set_seed(config.get('seed', 42))

    update_dir = Path(args.update_dir)
    update_dir.mkdir(exist_ok=True, parents=True)

    bandit_state = update_dir / "bandit_state.pkl"

    if args.init_only:
        bandit = EpsilonGreedyBandit(
            epsilon=cfg.get('bandit_epsilon', 0.3),
            alpha=cfg.get('bandit_alpha', 0.5),
        )
        _run_init_pass(args, cfg, config, ctx, update_dir, bandit)
        return

    # Main mode: restore bandit produced by the init pass.
    if not bandit_state.exists():
        raise FileNotFoundError(
            f"{bandit_state} missing. Orchestrator must run --init_only before main miner."
        )
    with open(bandit_state, 'rb') as f:
        bandit = pickle.load(f)
    print(f"[miner] Restored bandit state from {bandit_state} "
          f"({len(bandit.mean_g)} queries known)", flush=True)

    _run_main_loop(args, cfg, config, ctx, update_dir, bandit)


if __name__ == "__main__":
    main()
