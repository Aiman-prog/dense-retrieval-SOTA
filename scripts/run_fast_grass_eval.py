"""
Standalone BRIGHT eval for a finished Fast-GRASS checkpoint.

Run this AFTER a parallel sweep that trained with ``--no_eval``. The BRIGHT eval
writes to a shared scratch dir (temp_grass_workdir/final_eval/<domain>/{c,q}.pkl),
so concurrent evals would clobber each other — evaluate ONE checkpoint at a time:

    for d in models/fg_A_100k_l1 models/fg_B_32k_l1 models/fg_C_32k_noR models/fg_D_32k_l0; do
        python scripts/run_fast_grass_eval.py --model_dir "$d"
    done

Reuses utils.helpers.evaluate_bright (same path GRASS/Fast-GRASS training uses);
no edits to helpers.py / run_grass.py / negative_cache.py.
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
    ap.add_argument('--temp_workdir_key', default='temp_grass',
                    help='scratch-dir key for eval encodes (default temp_grass)')
    args = ap.parse_args()

    model_dir = Path(args.model_dir)
    if not model_dir.exists():
        sys.exit(f"[FAST-GRASS EVAL] model_dir not found: {model_dir}")

    config = load_config()
    ctx    = get_training_context('fast_grass')
    print(f"[FAST-GRASS EVAL] Evaluating {model_dir} on BRIGHT...", flush=True)
    evaluate_bright(ctx, config, model_dir, temp_workdir_key=args.temp_workdir_key)
    print(f"[FAST-GRASS EVAL] Done: {model_dir}", flush=True)


if __name__ == "__main__":
    main()
