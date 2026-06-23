"""
Standalone BRIGHT eval for a finished Fast-GRASS checkpoint.

Thin wrapper around the canonical evaluator utils.helpers.evaluate_bright (the same
code training's do_eval and GRASS use) — no reimplementation. Use it to evaluate
checkpoints trained with --no_eval.

evaluate_bright writes scratch into a single shared dir (temp_grass_workdir/final_eval),
so run checkpoints ONE AT A TIME (sequential), not as concurrent jobs:

    for d in /scratch/$USER/.../models/*fg_*_ema; do
        python scripts/run_fast_grass_eval.py --model_dir "$d"
    done

Run on a full gpu-a100 (the config's eval batch / query_max_len=1024 fit there; the
gpu-a100-small MIG slice has only ~9.5GB and OOMs on the long-query encode).
"""
import argparse
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root / 'src'))

from utils.helpers import load_config, get_training_context, evaluate_bright


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--model_dir', required=True,
                    help='path to a trained Fast-GRASS checkpoint / final model dir')
    args = ap.parse_args()

    model_dir = Path(args.model_dir)
    if not model_dir.exists():
        sys.exit(f"[FAST-GRASS EVAL] model_dir not found: {model_dir}")

    config = load_config()
    ctx    = get_training_context('fast_grass')
    print(f"[FAST-GRASS EVAL] Evaluating {model_dir} on BRIGHT...", flush=True)
    evaluate_bright(ctx, config, model_dir, temp_workdir_key='temp_grass')
    print(f"[FAST-GRASS EVAL] Done: {model_dir}", flush=True)


if __name__ == "__main__":
    main()
