"""GRASS async v2 orchestrator — Algorithm 4 driver.

Two-GPU async pipeline that preserves sequential mining geometry. Each miner
cycle = one full bandit round of X*|D| queries mined in 64-batch groups
(same as scripts/run_grass_seq_bandit.py). Trainer runs continuously on GPU 0;
miner runs on GPU 1; coordination via filesystem markers.

M-targeted termination: pick M target mining rounds; derive trainer epoch
count via total_epochs = ceil((t_init + M*t_mine) / t_train). Time constants
calibrated once on target hardware (see config.training.grass.async_v2).

Mirrors scripts/train_ance.py's orchestrator pattern (Popen miner on GPU 1,
subprocess.run trainer on GPU 0, SIGTERM miner on trainer exit).
"""

import argparse
import math
import os
import shutil
import subprocess
import sys
from pathlib import Path

from tevatron.retriever.modeling import DenseModel

os.environ["TRANSFORMERS_ATTENTION_IMPLEMENTATION"] = "eager"
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root / 'src'))

from utils.helpers import (
    get_path, get_training_context, load_config, encode_to_pickle,
    count_jsonl_examples, evaluate_bright,
)
from data.preprocessor import run_setup

# Tevatron Bug Patch (mirrors run_grass_mcd.py / train_ance.py)
if not hasattr(DenseModel, "_keys_to_ignore_on_save"):
    setattr(DenseModel, "_keys_to_ignore_on_save", None)


PLACEHOLDER_SENTINEL = "PLACEHOLDER"


def _validate_time_constants(cfg_v2):
    """Hard-error if t_init/t_mine/t_train still carry placeholder values."""
    missing = []
    for key in ('t_init', 't_mine', 't_train'):
        v = cfg_v2.get(key)
        if v is None or v == PLACEHOLDER_SENTINEL or (isinstance(v, str) and not v.replace('.', '').isdigit()):
            missing.append(key)
    if missing:
        raise ValueError(
            f"async_v2 time constants not calibrated: {missing}. "
            f"Set numeric seconds for each in config.training.grass.async_v2 (Step 11)."
        )


def compute_total_epochs(cfg_v2, M):
    """total_epochs = ceil((t_init + M * t_mine) / t_train). All times in seconds."""
    _validate_time_constants(cfg_v2)
    t_init  = float(cfg_v2['t_init'])
    t_mine  = float(cfg_v2['t_mine'])
    t_train = float(cfg_v2['t_train'])
    return math.ceil((t_init + M * t_mine) / t_train)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--recipe',        default='grass')
    parser.add_argument('--M',             type=int,   default=None,
                        help='Target mining rounds. Overrides config.async_v2.M.')
    parser.add_argument('--X',             type=float, default=None,
                        help='Coverage fraction. Overrides config.async_v2.X.')
    parser.add_argument('--selection',     type=str,   default=None, choices=['bandit', 'random'],
                        help='Query selection method. Overrides config.async_v2.selection.')
    parser.add_argument('--lambda_val',    type=float, default=None,
                        help='Gap-index lambda. Overrides cfg.lambda_val for the miner round.')
    parser.add_argument('--model_suffix',  type=str,   default=None)
    parser.add_argument('--debug',         action='store_true')
    parser.add_argument('--case_lite_enabled', action='store_true',
                        help='Enable CASE-Lite candidate-level sampling on the miner (§6). '
                             'Overrides config.async_v2.case_lite.enabled.')
    parser.add_argument('--case_lite_K',   type=int, default=None,
                        help='Override config.async_v2.case_lite.K (CASE-Lite Pareto knob).')
    args = parser.parse_args()

    ctx    = get_training_context(args.recipe)
    config = load_config()
    cfg    = config['training'][args.recipe]
    if 'async_v2' not in cfg:
        raise KeyError(
            "config.training.grass.async_v2 block missing. "
            "Add M, X, selection, lambda_val, t_init, t_mine, t_train, poll_interval_steps, save_steps."
        )
    cfg_v2 = dict(cfg['async_v2'])  # local copy; CLI overrides applied below

    if args.M          is not None: cfg_v2['M']         = args.M
    if args.X          is not None: cfg_v2['X']         = args.X
    if args.selection  is not None: cfg_v2['selection'] = args.selection
    if args.lambda_val is not None: cfg_v2['lambda_val'] = args.lambda_val

    # CASE-Lite CLI overrides. Toggle and K only; other hyperparams stay in config.yaml.
    cl_cfg = cfg_v2.setdefault('case_lite', {})
    if args.case_lite_enabled:
        cl_cfg['enabled'] = True
    if args.case_lite_K is not None:
        cl_cfg['K'] = args.case_lite_K
    # Propagate override back into cfg so the miner subprocess sees it via load_config()
    # (load_config re-reads the file, so we also pass --case_lite_enabled as a CLI flag below).
    case_lite_on = bool(cl_cfg.get('enabled', False))

    corpus_file, query_file, qrels_file = run_setup()

    # GPU placement. Tevatron's encode_to_pickle raises NotImplementedError on multi-GPU,
    # so we pin the orchestrator to GPU 0 for the stale-index build; miner subprocess
    # overrides this with CUDA_VISIBLE_DEVICES=miner_gpu.
    import torch as _torch
    n_gpus    = _torch.cuda.device_count()
    miner_gpu = '1' if n_gpus >= 2 else '0'
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    print(f"[async_v2] {n_gpus} GPU(s) detected. Trainer→GPU 0, Miner→GPU {miner_gpu}", flush=True)

    # Workdir layout (kept distinct from old async-grass under temp_grass_async/v2/).
    # update_dir is suffix-isolated so parallel sbatch jobs don't clobber each other's
    # mining IPC; stale_index is shared (read-only after first build).
    suffix = f"_{args.model_suffix}" if args.model_suffix else ''
    base_workdir   = get_path("temp_grass_async") / "v2"
    stale_dir      = base_workdir / "stale_index"
    update_dir     = base_workdir / f"updates{suffix}"
    stale_dir.mkdir(exist_ok=True, parents=True)
    update_dir.mkdir(exist_ok=True, parents=True)

    # Build stale FAISS index from base model (skip if cached).
    stale_pkl = stale_dir / "corpus.pkl"
    if not stale_pkl.exists():
        print("[async_v2] Building stale ANN index from base model...", flush=True)
        encode_to_pickle(cfg['base_model'], corpus_file, stale_pkl, False, ctx, config)
    print(f"[async_v2] Stale index: {stale_pkl}", flush=True)

    # Compute total_epochs from M (M-targeted termination, Eq. 5.1).
    M = int(cfg_v2['M'])
    n_examples      = count_jsonl_examples(str(get_path("processed") / "training_mixture" / "*.jsonl"))
    if n_examples == 0:
        raise RuntimeError("No training examples found. Run preprocessing first.")
    steps_per_epoch = math.ceil(n_examples / cfg['batch_size'])
    total_epochs    = compute_total_epochs(cfg_v2, M)
    max_steps       = steps_per_epoch * total_epochs
    print(f"[async_v2] M={M} X={cfg_v2['X']} selection={cfg_v2.get('selection','bandit')} "
          f"lambda={cfg_v2.get('lambda_val', cfg.get('lambda_val'))}", flush=True)
    print(f"[async_v2] {n_examples} examples | {steps_per_epoch} steps/epoch | "
          f"{total_epochs} epochs | {max_steps} total steps", flush=True)

    # Output model dir (suffix avoids collision across sweep cells).
    output_model_dir = get_path("models") / f"{cfg['model_name']}_asyncv2{suffix}"
    stale_ckpts = sorted(output_model_dir.glob("checkpoint-*"))
    if stale_ckpts:
        for ckpt in stale_ckpts:
            shutil.rmtree(ckpt, ignore_errors=True)
        print(f"[async_v2] Removed {len(stale_ckpts)} stale checkpoint(s) from {output_model_dir.name}",
              flush=True)
    output_model_dir.mkdir(exist_ok=True, parents=True)

    # ── INIT PASS (one-shot on miner GPU, blocks) ────────────────────────────
    # Writes update_dir/training_data_0/*.jsonl (L=1 mined negs over all queries)
    # and update_dir/bandit_state.pkl (EMA-warmed bandit).
    init_data_dir = update_dir / "training_data_0"
    bandit_state  = update_dir / "bandit_state.pkl"
    if init_data_dir.exists() and bandit_state.exists() and list(init_data_dir.glob("*.jsonl")):
        print(f"[async_v2] Skipping init pass: {init_data_dir} already populated", flush=True)
    else:
        print("[async_v2] Running init pass on miner GPU (subprocess, blocks)...", flush=True)
        init_env = {**os.environ, 'CUDA_VISIBLE_DEVICES': miner_gpu}
        init_cmd = [
            sys.executable, str(Path(__file__).parent / "run_grass_async_v2_miner.py"),
            '--update_dir',       str(update_dir),
            '--stale_pkl',        str(stale_pkl),
            '--corpus_file',      str(corpus_file),
            '--query_file',       str(query_file),
            '--qrels_file',       str(qrels_file),
            '--output_model_dir', str(output_model_dir),
            '--recipe',           args.recipe,
            '--coverage',         str(cfg_v2['X']),
            '--selection',        cfg_v2.get('selection', 'bandit'),
            '--lambda_val',       str(cfg_v2.get('lambda_val', cfg.get('lambda_val', 1.0))),
            '--init_only',
        ]
        if args.debug:
            init_cmd.append('--debug')
        if case_lite_on:
            init_cmd.append('--case_lite_enabled')
            if args.case_lite_K is not None:
                init_cmd += ['--case_lite_K', str(args.case_lite_K)]
        subprocess.run(init_cmd, env=init_env, check=True)
        print(f"[async_v2] Init pass done: {init_data_dir}", flush=True)

    # ── LAUNCH MINER (background, never blocks) ──────────────────────────────
    miner_env = {**os.environ, 'CUDA_VISIBLE_DEVICES': miner_gpu}
    miner_cmd = [
        sys.executable, str(Path(__file__).parent / "run_grass_async_v2_miner.py"),
        '--update_dir',       str(update_dir),
        '--stale_pkl',        str(stale_pkl),
        '--corpus_file',      str(corpus_file),
        '--query_file',       str(query_file),
        '--qrels_file',       str(qrels_file),
        '--output_model_dir', str(output_model_dir),
        '--recipe',           args.recipe,
        '--coverage',         str(cfg_v2['X']),
        '--selection',        cfg_v2.get('selection', 'bandit'),
        '--lambda_val',       str(cfg_v2.get('lambda_val', cfg.get('lambda_val', 1.0))),
    ]
    if args.debug:
        miner_cmd.append('--debug')
    if case_lite_on:
        miner_cmd.append('--case_lite_enabled')
        if args.case_lite_K is not None:
            miner_cmd += ['--case_lite_K', str(args.case_lite_K)]
    miner_proc = subprocess.Popen(miner_cmd, env=miner_env)
    print(f"[async_v2] Miner started on GPU {miner_gpu} (pid {miner_proc.pid})", flush=True)

    # ── LAUNCH TRAINER on GPU 0 (foreground, blocks until max_steps) ─────────
    train_env = {**os.environ, 'CUDA_VISIBLE_DEVICES': '0'}
    try:
        subprocess.run([
            sys.executable, str(Path(__file__).parent / "run_grass_async_v2_trainer.py"),
            '--model_name_or_path', cfg['base_model'],
            '--initial_data_dir',   str(init_data_dir),
            '--update_dir',         str(update_dir),
            '--output_dir',         str(output_model_dir),
            '--max_steps',          str(max_steps),
            '--recipe',             args.recipe,
        ], env=train_env, check=True)
    finally:
        miner_proc.terminate()
        miner_proc.wait()
        print("[async_v2] Miner terminated.", flush=True)

    # ── EVALUATE final model ─────────────────────────────────────────────────
    evaluate_bright(ctx, config, output_model_dir)


if __name__ == "__main__":
    main()
