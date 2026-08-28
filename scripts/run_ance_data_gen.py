"""ANCE Inferencer -- runs on GPU 1 in parallel with training.

Polls the output model dir for new checkpoints, re-encodes the whole corpus, builds
a FAISS IndexFlatIP, mines negatives and commits a round into the run's work root.
The committed `ready_N` marker is the trainer's only completion signal.

Paper reference: Section 4 "Asynchronous Index Refresh", Figure 2, Appendix A.3

Round data is RETAINED, never pruned: the trainer may still be iterating a round
this process considers superseded, and pruning would race it. Runs are fresh-only
under a unique work root, so retention costs disk on a directory that is already
per-run. The per-cycle encode pickles (~36 GB) are still cleaned.
"""
import sys
import time
import pickle
import random
import shutil
import argparse
import traceback
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root / 'src'))
sys.path.append(str(project_root / 'scripts'))

from utils.helpers import (
    get_path, get_training_context, load_config, encode_to_pickle, build_faiss_index,
    is_valid_checkpoint, _load_qrels, _load_corpus_lookup, set_seed, _sha256,
)
from data.preprocessor import MIXTURE_FILES, MSMARCO_ONLY_FILES, require_mixture_files
from ance_mining import (build_round_records, latest_committed_round,
                         mine_from_index, publish_round)


def _checkpoint_step(path):
    """The step a `checkpoint-N` directory records, or 0 if it is not one."""
    name = Path(path).name
    tail = name.split('-')[-1]
    return int(tail) if name.startswith('checkpoint-') and tail.isdigit() else 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_model_dir', required=True)
    parser.add_argument('--work_root',  required=True)
    parser.add_argument('--run_id',     required=True)
    parser.add_argument('--corpus_file', required=True)
    parser.add_argument('--query_file',  required=True)
    parser.add_argument('--qrels_file',  required=True)
    parser.add_argument('--recipe', default='ance')
    args = parser.parse_args()

    from transformers.trainer_utils import get_last_checkpoint

    work_root = Path(args.work_root)
    work_root.mkdir(exist_ok=True, parents=True)
    # prevents FileNotFoundError in get_last_checkpoint before the first save
    Path(args.output_model_dir).mkdir(exist_ok=True, parents=True)

    ctx = get_training_context(args.recipe)
    config = load_config()
    seed = config.get('seed', 42)
    set_seed(seed)
    rng = random.Random(seed)

    poll_interval = ctx['args']['data_gen_poll_interval']
    n_negs        = ctx['args']['train_group_size'] - 1
    mining_depth  = ctx['args']['mining_depth']

    qrels_dict    = _load_qrels(args.qrels_file)
    corpus_lookup = _load_corpus_lookup(args.corpus_file)
    corpus_sha    = _sha256(args.corpus_file)

    mixture_dir = get_path("processed") / ctx['args']['mixture_dir']
    expected = (MSMARCO_ONLY_FILES if ctx['args']['setup_mode'] == 'tevatron_msmarco'
                else MIXTURE_FILES)
    mixture_files = list(require_mixture_files(mixture_dir, expected))

    last_checkpoint = None
    # Numbering continues from this run's own work root, which is empty at startup.
    output_num = latest_committed_round(work_root) + 1
    print(f"[Inferencer] run_id={args.run_id} | polling {args.output_model_dir} every "
          f"{poll_interval}s | mining_depth={mining_depth}, n_negs={n_negs}", flush=True)

    while True:
        next_checkpoint = get_last_checkpoint(str(args.output_model_dir))

        if (next_checkpoint is None or next_checkpoint == last_checkpoint
                or not is_valid_checkpoint(next_checkpoint)):
            print(f"[Inferencer] Polling... "
                  f"last={Path(last_checkpoint).name if last_checkpoint else None} "
                  f"next={Path(next_checkpoint).name if next_checkpoint else None}",
                  flush=True)
            time.sleep(poll_interval)
            continue

        step = _checkpoint_step(next_checkpoint)
        print(f"[Inferencer] Checkpoint {Path(next_checkpoint).name} → "
              f"generating ANN round #{output_num}", flush=True)

        # Paper: "recomputes the encodings of the entire corpus"
        encode_dir = work_root / f"encode_{output_num}"
        encode_dir.mkdir(exist_ok=True, parents=True)
        _t0 = time.time()
        encode_to_pickle(next_checkpoint, args.corpus_file,
                         encode_dir / "corpus.pkl", False, ctx, config)
        print(f"[Inferencer] Corpus encode done in {time.time()-_t0:.1f}s", flush=True)
        _t1 = time.time()
        encode_to_pickle(next_checkpoint, args.query_file,
                         encode_dir / "query.pkl", True, ctx, config)
        print(f"[Inferencer] Query encode done in {time.time()-_t1:.1f}s", flush=True)

        # Paper Eq. 13: D^-_ANCE = ANN_{f(q,d)} \ D^+
        index, _, corpus_ids = build_faiss_index(encode_dir / "corpus.pkl")
        with open(encode_dir / "query.pkl", 'rb') as f:
            q_data = pickle.load(f)

        mined, failures = mine_from_index(
            index, corpus_ids, q_data, mixture_files, qrels_dict,
            n_negs=n_negs, mining_depth=mining_depth, rng=rng)

        # publish_round writes round_meta_N.json before ready_N and refuses to
        # publish at all when a query could not supply its ANN negatives.
        publish_round(
            work_root, output_num,
            records_by_file=build_round_records(mixture_files, mined, corpus_lookup,
                                                n_negs=n_negs),
            meta={'run_id': args.run_id, 'ann_no': output_num,
                  'checkpoint': str(next_checkpoint), 'checkpoint_step': step,
                  'n_queries_mined': len(mined), 'n_sampling_failures': len(failures),
                  'sampling_failures': failures[:20], 'corpus_sha256': corpus_sha})
        print(f"[Inferencer] Round #{output_num} committed from checkpoint-{step} "
              f"({len(mined)} queries). Waiting for the next checkpoint...", flush=True)

        # Free ~36GB of encode pickles. Round data itself is retained.
        shutil.rmtree(encode_dir, ignore_errors=True)

        last_checkpoint = next_checkpoint
        output_num += 1


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        # SIGINT from the orchestrator's shutdown is a normal end, not a failure.
        sys.exit(0)
    except Exception:
        # Loudly, and nonzero: the orchestrator treats any nonzero exit as a failed
        # run, because a silent inferencer death degrades ANCE into static training.
        traceback.print_exc()
        sys.exit(1)
