"""Calibrate t_init, t_mine, t_train on the current hardware.

Async v2's M-targeted termination needs all three time constants:
    total_epochs = ceil((t_init + M * t_mine) / t_train)

This script measures them by running each phase on a small subset and
extrapolating to the full dataset / one full bandit round.

Outputs a YAML snippet you paste into config.training.grass.async_v2.

Both t_mine variants are reported:
  - t_mine (full GRASS L=10 over X·|D| queries)
  - t_mine (CASE-Lite K=6  over X·|D| queries)
Use the one matching your run (case_lite.enabled).

Usage (interactive or via SLURM):
    python scripts/calibrate_async_v2_times.py \
        --n_init 5000 --n_mine 1000 --n_train_steps 200

Defaults take ~25-35 min on a single A100; toggle --skip_train if you only
want miner-side numbers (~10 min).
"""

import argparse
import math
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd
from tevatron.retriever.modeling import DenseModel

os.environ["TRANSFORMERS_ATTENTION_IMPLEMENTATION"] = "eager"
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root / 'src'))
sys.path.append(str(project_root / 'scripts'))

from utils.helpers import (
    get_path, get_training_context, load_config,
    build_faiss_index, _load_qrels, _load_corpus_lookup,
    encode_to_pickle, count_jsonl_examples,
)
from utils.bandit import CaseLiteBandit
from data.preprocessor import run_setup
from run_grass_mcd import grass_sampler
from run_grass_case_lite import case_lite_sampler

if not hasattr(DenseModel, "_keys_to_ignore_on_save"):
    setattr(DenseModel, "_keys_to_ignore_on_save", None)


def _fresh_dir(p: Path) -> Path:
    if p.exists():
        shutil.rmtree(p, ignore_errors=True)
    p.mkdir(parents=True, exist_ok=True)
    return p


def measure_init(args, cfg, config, stale_idx, stale_embs, c_id_to_idx, c_ids,
                 corpus_lookup, qrels_dict, mix_df, work):
    subset   = mix_df.head(args.n_init).reset_index(drop=True)
    out_dir  = _fresh_dir(work / "calibrate_init")
    cfg_init = {**cfg, 'L': 1, 'm': 1, 'lambda_val': 1.0}
    t0 = time.time()
    grass_sampler(cfg['base_model'], stale_idx, stale_embs, c_id_to_idx, c_ids, corpus_lookup,
                  subset, qrels_dict, cfg_init, config, out_dir)
    return time.time() - t0


def measure_mine_full(args, cfg, config, stale_idx, stale_embs, c_id_to_idx, c_ids,
                      corpus_lookup, qrels_dict, mix_df, work):
    subset   = mix_df.head(args.n_mine).reset_index(drop=True)
    out_dir  = _fresh_dir(work / "calibrate_mine_full")
    cfg_mine = {**cfg, 'L': cfg.get('L', 10), 'm': 1, 'lambda_val': 1.0}
    t0 = time.time()
    grass_sampler(cfg['base_model'], stale_idx, stale_embs, c_id_to_idx, c_ids, corpus_lookup,
                  subset, qrels_dict, cfg_mine, config, out_dir)
    return time.time() - t0


def measure_mine_case_lite(args, cfg, config, stale_idx, stale_embs, c_id_to_idx, c_ids,
                           corpus_lookup, qrels_dict, mix_df, work):
    cl_cfg = cfg['async_v2']['case_lite']
    bandit = CaseLiteBandit(
        bucket_boundaries=cl_cfg['bucket_boundaries'],
        initial_slots    =cl_cfg['initial_slots'],
        alpha_b          =float(cl_cfg.get('alpha_b', 0.5)),
        beta             =float(cl_cfg.get('beta',    0.5)),
        gamma            =float(cl_cfg.get('gamma',   0.05)),
        tau              =float(cl_cfg.get('tau',     0.0)),
        lambda_val       =1.0,
    )
    subset   = mix_df.head(args.n_mine).reset_index(drop=True)
    out_dir  = _fresh_dir(work / "calibrate_mine_case_lite")
    cfg_round = {**cfg, 'lambda_val': 1.0}
    t0 = time.time()
    case_lite_sampler(cfg['base_model'], stale_idx, stale_embs, c_id_to_idx, c_ids, corpus_lookup,
                      subset, qrels_dict, cfg_round, config, out_dir, bandit, round_idx=1)
    return time.time() - t0


def measure_train(args, cfg, work):
    """Subprocess the trainer for n_train_steps; return total elapsed (includes startup).

    Startup overhead is ~30-60s and amortises across n_train_steps. With the
    default 200 steps the per-step bias is small (<25%) — enough for picking
    total_epochs. Bump --n_train_steps if you want tighter accuracy.
    """
    initial_data_dir = get_path("processed") / "training_mixture"
    output_dir       = _fresh_dir(work / "calibrate_train_out")
    update_dir       = _fresh_dir(work / "calibrate_train_updates")
    cmd = [
        sys.executable, str(Path(__file__).parent / "run_grass_async_v2_trainer.py"),
        '--model_name_or_path', cfg['base_model'],
        '--initial_data_dir',   str(initial_data_dir),
        '--update_dir',         str(update_dir),
        '--output_dir',         str(output_dir),
        '--max_steps',          str(args.n_train_steps),
        '--recipe',             'grass',
    ]
    t0 = time.time()
    subprocess.run(cmd, check=True)
    return time.time() - t0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--recipe',         default='grass')
    parser.add_argument('--n_init',         type=int, default=5000,
                        help='Queries to mine for t_init measurement.')
    parser.add_argument('--n_mine',         type=int, default=1000,
                        help='Queries to mine for each t_mine variant.')
    parser.add_argument('--n_train_steps',  type=int, default=200,
                        help='Steps to run the trainer for t_train measurement.')
    parser.add_argument('--skip_train',     action='store_true',
                        help='Skip t_train (~15 min faster, useful for miner-only calibration).')
    args = parser.parse_args()

    config = load_config()
    cfg    = config['training'][args.recipe]
    ctx    = get_training_context(args.recipe)

    # Always pin to GPU 0 for calibration (no async).
    import torch as _torch
    os.environ.setdefault('CUDA_VISIBLE_DEVICES', '0')
    print(f"[calibrate] device: {_torch.cuda.get_device_name(0) if _torch.cuda.is_available() else 'CPU'}",
          flush=True)

    corpus_file, query_file, qrels_file = run_setup()
    work = get_path("temp_grass_async") / "v2" / "calibrate"
    work.mkdir(parents=True, exist_ok=True)

    # Reuse the main async_v2 stale FAISS index (build if missing).
    stale_pkl = get_path("temp_grass_async") / "v2" / "stale_index" / "corpus.pkl"
    if not stale_pkl.exists():
        print(f"[calibrate] Building stale ANN index (one-time)...", flush=True)
        stale_pkl.parent.mkdir(parents=True, exist_ok=True)
        encode_to_pickle(cfg['base_model'], corpus_file, stale_pkl, False, ctx, config)
    print(f"[calibrate] Stale index: {stale_pkl}", flush=True)

    print(f"[calibrate] Loading FAISS + corpus + qrels...", flush=True)
    stale_idx, stale_embs, c_ids = build_faiss_index(stale_pkl)
    c_id_to_idx   = {did: i for i, did in enumerate(c_ids)}
    corpus_lookup = _load_corpus_lookup(corpus_file)
    qrels_dict    = _load_qrels(qrels_file)
    mix_df        = pd.read_json(query_file, lines=True)
    n_total       = len(mix_df)

    n_examples       = count_jsonl_examples(str(get_path("processed") / "training_mixture" / "*.jsonl"))
    steps_per_epoch  = math.ceil(n_examples / cfg['batch_size'])
    X                = float(cfg['async_v2']['X'])
    n_mine_per_round = int(X * n_total)

    print(f"[calibrate] n_total queries={n_total}, "
          f"n_mine_per_round (X={X}) = {n_mine_per_round}, "
          f"training mixture {n_examples} examples, {steps_per_epoch} steps/epoch", flush=True)

    # ── T_INIT ──────────────────────────────────────────────────────────────
    print(f"\n[calibrate] Phase 1/{3 if not args.skip_train else 3}: "
          f"T_INIT — grass_sampler(L=1) on {args.n_init} queries...", flush=True)
    t_init_sub  = measure_init(args, cfg, config, stale_idx, stale_embs, c_id_to_idx, c_ids,
                               corpus_lookup, qrels_dict, mix_df, work)
    t_init_full = t_init_sub * (n_total / args.n_init)
    print(f"  ⇒ subset {t_init_sub:.0f}s → extrapolated to {n_total} queries: "
          f"{t_init_full:.0f}s ({t_init_full/60:.1f} min)", flush=True)

    # ── T_MINE (full GRASS) ─────────────────────────────────────────────────
    print(f"\n[calibrate] Phase 2: T_MINE (full GRASS L={cfg.get('L', 10)}, T={cfg['T']}) "
          f"on {args.n_mine} queries...", flush=True)
    t_mfull_sub = measure_mine_full(args, cfg, config, stale_idx, stale_embs, c_id_to_idx, c_ids,
                                    corpus_lookup, qrels_dict, mix_df, work)
    t_mfull     = t_mfull_sub * (n_mine_per_round / args.n_mine)
    print(f"  ⇒ subset {t_mfull_sub:.0f}s → extrapolated to {n_mine_per_round} queries: "
          f"{t_mfull:.0f}s ({t_mfull/60:.1f} min)", flush=True)

    # ── T_MINE (CASE-Lite) ──────────────────────────────────────────────────
    cl_cfg = cfg['async_v2']['case_lite']
    print(f"\n[calibrate] Phase 3: T_MINE (CASE-Lite K={cl_cfg['K']} L_mem={cl_cfg['L_mem']}) "
          f"on {args.n_mine} queries...", flush=True)
    t_mcase_sub = measure_mine_case_lite(args, cfg, config, stale_idx, stale_embs, c_id_to_idx, c_ids,
                                         corpus_lookup, qrels_dict, mix_df, work)
    t_mcase     = t_mcase_sub * (n_mine_per_round / args.n_mine)
    print(f"  ⇒ subset {t_mcase_sub:.0f}s → extrapolated to {n_mine_per_round} queries: "
          f"{t_mcase:.0f}s ({t_mcase/60:.1f} min)", flush=True)

    # ── T_TRAIN ─────────────────────────────────────────────────────────────
    t_train_full = None
    if not args.skip_train:
        print(f"\n[calibrate] Phase 4: T_TRAIN — trainer for {args.n_train_steps} steps "
              f"(includes ~30-60s startup)...", flush=True)
        t_tsub          = measure_train(args, cfg, work)
        t_per_step      = t_tsub / args.n_train_steps
        t_train_full    = t_per_step * steps_per_epoch
        print(f"  ⇒ subset {t_tsub:.0f}s ({t_per_step:.2f}s/step) → extrapolated to "
              f"{steps_per_epoch} steps/epoch: {t_train_full:.0f}s ({t_train_full/60:.1f} min)",
              flush=True)

    # ── Output ──────────────────────────────────────────────────────────────
    M = int(cfg['async_v2']['M'])
    print("\n" + "=" * 70)
    print("CALIBRATION RESULTS")
    print("=" * 70)
    print(f"\nPaste into config/config.yaml under training.grass.async_v2:\n")
    print(f"      t_init:   {t_init_full:>7.0f}    # ~{t_init_full/60:.1f} min, calibrated")
    print(f"      # --- pick the t_mine matching your run --- ")
    print(f"      t_mine:   {t_mfull:>7.0f}    # ~{t_mfull/60:.1f} min, full GRASS L={cfg.get('L',10)} X={X}")
    print(f"      # t_mine: {t_mcase:>7.0f}    # ~{t_mcase/60:.1f} min, CASE-Lite K={cl_cfg['K']} X={X}")
    if t_train_full is not None:
        print(f"      t_train:  {t_train_full:>7.0f}    # ~{t_train_full/60:.1f} min, one epoch of {steps_per_epoch} steps")
    else:
        print(f"      # t_train: <not measured — rerun without --skip_train>")
    print()

    # Sanity-check total_epochs for both regimes.
    if t_train_full is not None:
        te_full = math.ceil((t_init_full + M * t_mfull) / t_train_full)
        te_case = math.ceil((t_init_full + M * t_mcase) / t_train_full)
        print(f"Predicted total_epochs at M={M}:")
        print(f"  full GRASS:  ceil(({t_init_full:.0f} + {M}·{t_mfull:.0f}) / {t_train_full:.0f}) = {te_full}")
        print(f"  CASE-Lite:   ceil(({t_init_full:.0f} + {M}·{t_mcase:.0f}) / {t_train_full:.0f}) = {te_case}")
    print()


if __name__ == "__main__":
    main()
