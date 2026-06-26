"""
Standalone refresh of the Fast-GRASS stale index.

Re-encodes the full corpus into temp_grass_workdir/stale_index/corpus.pkl using the
Fast-GRASS base model (the InBatch checkpoint, config.training.fast_grass.base_model).
This is the exact one-off that run_fast_grass.py does on first launch — pulled out so the
index can be refreshed on its own after retraining InBatch, without starting a training run.

Usage:
    python scripts/refresh_stale_index.py                 # rebuild from fast_grass.base_model
    python scripts/refresh_stale_index.py --model PATH     # rebuild from an explicit checkpoint
    python scripts/refresh_stale_index.py --keep-existing  # no-op if the pickle already exists
"""

import sys
import argparse
import shutil
from datetime import datetime
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root / 'src'))
sys.path.append(str(project_root / 'scripts'))

from utils.helpers import get_path, get_training_context, load_config, encode_to_pickle


def main():
    parser = argparse.ArgumentParser(description="Rebuild the Fast-GRASS stale index.")
    parser.add_argument('--model', default=None,
                        help="Checkpoint to encode with (default: config.training.fast_grass.base_model)")
    parser.add_argument('--keep-existing', action='store_true',
                        help="Do nothing if the stale index already exists (default: rebuild it).")
    args = parser.parse_args()

    config = load_config()
    ctx = get_training_context('fast_grass')
    model_path = args.model or ctx['base_model']

    # Same corpus the trainer encodes (init source only).
    from data.preprocessor import run_setup
    corpus_file, _query_file, _qrels_file = run_setup()

    stale_pkl = get_path("temp_grass") / "stale_index" / "corpus.pkl"
    stale_pkl.parent.mkdir(parents=True, exist_ok=True)

    if stale_pkl.exists():
        if args.keep_existing:
            print(f"[REFRESH] Stale index already present, --keep-existing set; nothing to do: {stale_pkl}")
            return
        archived = stale_pkl.with_suffix(f".pkl.old_{datetime.now():%Y%m%d_%H%M%S}")
        shutil.move(str(stale_pkl), str(archived))
        print(f"[REFRESH] Archived previous stale index -> {archived}")

    print(f"[REFRESH] Encoding corpus with: {model_path}")
    print(f"[REFRESH] Corpus file:          {corpus_file}")
    print(f"[REFRESH] Output:               {stale_pkl}")
    encode_to_pickle(model_path, corpus_file, stale_pkl, False, ctx, config)
    print(f"[REFRESH] Done. Stale index ready: {stale_pkl}")


if __name__ == "__main__":
    main()
