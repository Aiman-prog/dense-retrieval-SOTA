"""
GRASS Async Orchestrator — mirrors the train_ance.py pattern.
Builds stale FAISS index once, then launches:
  - Miner (GPU 1): MC-dropout uncertainty mining, writes neg updates
  - Trainer (GPU 0): trains on mixture negatives + miner updates
"""
import gc
import os
import sys
import shutil
import subprocess
import torch
from pathlib import Path

os.environ["TRANSFORMERS_ATTENTION_IMPLEMENTATION"] = "eager"
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root / 'src'))

from utils.helpers import (
    get_path, get_training_context, load_config,
    encode_to_pickle, set_seed, evaluate_bright,
)
from data.preprocessor import run_setup


def main():
    import argparse
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--recipe',       default='grass')
    parser.add_argument('--n_das',        type=int,  default=None)
    parser.add_argument('--selection',    type=str,  default=None)
    parser.add_argument('--model_suffix', type=str,  default=None)
    parser.add_argument('--num_epochs',   type=int,  default=None)
    parser.add_argument('--debug',        action='store_true')
    cli_args, _ = parser.parse_known_args()

    config = load_config()
    cfg    = config['training'][cli_args.recipe]
    ctx    = get_training_context(cli_args.recipe)
    set_seed(config.get('seed', 42))

    corpus_file, query_file, qrels_file = run_setup()

    if cli_args.n_das is not None:
        cfg = {**cfg, 'mab_n_das': cli_args.n_das}
    if cli_args.num_epochs is not None:
        cfg = {**cfg, 'num_epochs': cli_args.num_epochs}
    if cli_args.model_suffix is not None:
        cfg = {**cfg, 'model_name': cfg['model_name'] + '_' + cli_args.model_suffix}
    selection = cli_args.selection or cfg.get('selection', 'bandit')
    n_das     = cfg.get('mab_n_das', 5)

    n_gpus    = torch.cuda.device_count()
    miner_gpu = '1' if n_gpus >= 2 else '0'
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    print(f"[GRASS-Async] {n_gpus} GPU(s) — Trainer→GPU 0, Miner→GPU {miner_gpu}", flush=True)

    workdir = get_path("temp_grass_async")
    workdir.mkdir(exist_ok=True, parents=True)

    stale_dir = workdir / "stale_index"
    stale_dir.mkdir(exist_ok=True)
    stale_pkl = stale_dir / "corpus.pkl"
    if not stale_pkl.exists():
        print("[GRASS-Async] Building stale ANN index...", flush=True)
        encode_to_pickle(cfg['base_model'], corpus_file, stale_pkl, False, ctx, config)
    print(f"[GRASS-Async] Stale index ready: {stale_pkl}", flush=True)

    output_model_dir = get_path("models") / (cfg['model_name'] + '_async')
    output_model_dir.mkdir(parents=True, exist_ok=True)
    neg_update_dir = workdir / f"neg_updates_{cfg['model_name']}"
    neg_update_dir.mkdir(exist_ok=True)

    # Remove stale checkpoints so miner doesn't load prior-run weights
    stale_ckpts = sorted(output_model_dir.glob("checkpoint-*"))
    if stale_ckpts:
        for ckpt in stale_ckpts:
            shutil.rmtree(ckpt, ignore_errors=True)
        print(f"[GRASS-Async] Removed {len(stale_ckpts)} stale checkpoint(s)", flush=True)

    # Launch miner on GPU 1 (background, runs until trainer finishes)
    miner_env = {**os.environ, 'CUDA_VISIBLE_DEVICES': miner_gpu}
    miner_cmd = [
        sys.executable, str(Path(__file__).parent / "run_grass_miner.py"),
        '--output_model_dir', str(output_model_dir),
        '--neg_update_dir',   str(neg_update_dir),
        '--stale_pkl',        str(stale_pkl),
        '--query_file',       str(query_file),
        '--corpus_file',      str(corpus_file),
        '--qrels_file',       str(qrels_file),
        '--base_model',       cfg['base_model'],
        '--recipe',           cli_args.recipe,
        '--selection',        selection,
        '--n_das',            str(n_das),
    ]
    miner_proc = subprocess.Popen(miner_cmd, env=miner_env)
    print(f"[GRASS-Async] Miner started on GPU {miner_gpu} (pid {miner_proc.pid})", flush=True)

    # Launch trainer on GPU 0 (foreground — blocks until training completes)
    trainer_env = {**os.environ, 'CUDA_VISIBLE_DEVICES': '0'}
    try:
        subprocess.run([
            sys.executable, str(Path(__file__).parent / "run_grass_train.py"),
            '--output_model_dir', str(output_model_dir),
            '--neg_update_dir',   str(neg_update_dir),
            '--corpus_file',      str(corpus_file),
            '--recipe',           cli_args.recipe,
        ], env=trainer_env, check=True)
    finally:
        miner_proc.terminate()
        miner_proc.wait()
        print("[GRASS-Async] Miner terminated.", flush=True)

    # Evaluate final model on BRIGHT
    evaluate_bright(ctx, config, output_model_dir, temp_workdir_key='temp_grass_async')


if __name__ == "__main__":
    main()
