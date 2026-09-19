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

import numpy as np

project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root / 'src'))
sys.path.append(str(project_root / 'scripts'))

from utils.helpers import (
    get_path, get_training_context, load_config, encode_to_pickle,
    is_valid_checkpoint, _load_qrels, _load_corpus_lookup, set_seed, _sha256,
)
from data.preprocessor import MIXTURE_FILES, MSMARCO_ONLY_FILES, require_mixture_files
from ance_mining import (build_round_records, encode_and_mine,
                         latest_committed_round, publish_round)

# Upstream EvalDevQuery (run_ann_data_gen.py:397-425) searches to 100 and keeps the
# first 50 distinct document ids per query.
DEV_SEARCH_DEPTH = 100
DEV_RANK_DEPTH = 50


def _checkpoint_step(path):
    """The step a `checkpoint-N` directory records, or 0 if it is not one."""
    name = Path(path).name
    tail = name.split('-')[-1]
    return int(tail) if name.startswith('checkpoint-') and tail.isdigit() else 0


class _Phase:
    """Timestamped phase logging, and the per-phase seconds the round meta carries.

    Deliberately not a heartbeat file: a background writer keeps ticking while a
    multi-hour `encode_to_pickle` is deadlocked, so it would report a hung miner as
    healthy. The encode timeout is what actually detects a stall; this only answers
    "what was the miner doing, and for how long".
    """

    def __init__(self):
        self.timings = {}
        self._name = None
        self._t0 = None

    def __call__(self, name):
        self.close()
        self._name, self._t0 = name, time.time()
        print(f"[Inferencer] {time.strftime('%H:%M:%S')} → {name}", flush=True)
        return self

    def close(self):
        if self._name is not None:
            elapsed = time.time() - self._t0
            self.timings[f"{self._name}_s"] = round(elapsed, 1)
            print(f"[Inferencer] {self._name} done in {elapsed:.1f}s", flush=True)
            self._name = None


def dev_ndcg_at_10(index, corpus_ids, dev_pkl, dev_qrels):
    """Checkpoint-aligned dev NDCG@10, mirroring upstream's EvalDevQuery.

    Search to 100, keep the first 50 DISTINCT document ids, score them by -rank, and
    let pytrec_eval compute ndcg_cut_10. The rank-as-score trick is upstream's: only
    the ordering matters to NDCG, and it keeps the metric independent of the
    similarity scale, which is what makes two checkpoints comparable.

    Attributed by the caller to the checkpoint that produced these embeddings, never
    to the trainer's current step -- the two are hundreds of optimizer steps apart by
    the time mining finishes.
    """
    from evaluation.trec_eval_wrapper import TrecEvalWrapper

    with open(dev_pkl, 'rb') as handle:
        embeddings, ids = pickle.load(handle)
    depth = min(DEV_SEARCH_DEPTH, len(corpus_ids))
    _, indices = index.search(embeddings.astype(np.float32), depth)

    run = {}
    for row, qid in enumerate(str(q) for q in ids):
        seen, ranked = set(), {}
        for j in indices[row]:
            if j < 0:
                continue
            docid = corpus_ids[j]
            if docid in seen:
                continue
            seen.add(docid)
            ranked[docid] = -float(len(seen))
            if len(seen) >= DEV_RANK_DEPTH:
                break
        run[qid] = ranked
    return TrecEvalWrapper(dev_qrels).evaluate(run, {'ndcg_cut_10'})['ndcg_cut_10']


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
    chunk_factor  = ctx['args']['ann_chunk_factor']
    encode_timeout = ctx['args']['max_encode_seconds']

    phase = _Phase()
    phase("load")
    qrels_dict    = _load_qrels(args.qrels_file)
    corpus_lookup = _load_corpus_lookup(args.corpus_file)
    corpus_sha    = _sha256(args.corpus_file)

    mixture_dir = get_path("processed") / ctx['args']['mixture_dir']
    expected = (MSMARCO_ONLY_FILES if ctx['args']['setup_mode'] == 'tevatron_msmarco'
                else MIXTURE_FILES)
    mixture_files = list(require_mixture_files(mixture_dir, expected))

    # Online convergence signal. The reference computes it every generated round and
    # stores it with the source checkpoint (run_ann_data_gen.py:331-334); a falling
    # training loss on one stale negative distribution can coexist with worsening
    # retrieval, so loss alone is not enough. BRIGHT declares no eval files -- its
    # reportable path is run_all_evals.py -- so it simply does not compute one.
    dev_queries = dev_qrels = None
    if ctx['args'].get('eval_queries_file'):
        p_dir = get_path("processed")
        dev_queries = p_dir / ctx['args']['eval_queries_file']
        dev_qrels = _load_qrels(p_dir / ctx['args']['eval_qrels_file'])
        print(f"[Inferencer] online dev NDCG@10 over {len(dev_qrels)} judged queries",
              flush=True)
    phase.close()

    last_checkpoint = None
    # Numbering continues from this run's own work root, which is empty at startup.
    output_num = latest_committed_round(work_root) + 1
    print(f"[Inferencer] run_id={args.run_id} | polling {args.output_model_dir} every "
          f"{poll_interval}s | mining_depth={mining_depth}, n_negs={n_negs}, "
          f"ann_chunk_factor={chunk_factor}, max_encode_seconds={encode_timeout}",
          flush=True)

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

        encode_dir = work_root / f"encode_{output_num}"
        encode_dir.mkdir(exist_ok=True, parents=True)

        index, corpus_ids, mined, failures, shard_qids, shard = encode_and_mine(
            next_checkpoint, encode_dir, corpus_file=args.corpus_file,
            query_file=args.query_file, mixture_files=mixture_files,
            qrels_dict=qrels_dict, ctx=ctx, config=config, rng=rng,
            ann_no=output_num, phase=phase)

        dev_ndcg = None
        if dev_queries is not None:
            # The convergence signal, recorded against the checkpoint that produced
            # it. Cheap next to the full-corpus encode the index already paid for.
            phase("dev_ndcg")
            encode_to_pickle(next_checkpoint, dev_queries, encode_dir / "dev.pkl",
                             True, ctx, config, timeout=encode_timeout)
            dev_ndcg = dev_ndcg_at_10(index, corpus_ids, encode_dir / "dev.pkl",
                                      dev_qrels)
            print(f"[Inferencer] checkpoint-{step} dev NDCG@10 = {dev_ndcg:.4f}",
                  flush=True)

        # publish_round writes round_meta_N.json before ready_N and refuses to
        # publish at all when a query could not supply its ANN negatives.
        # `phase.timings` is read while this call is being built, so the meta carries
        # every phase up to and including search. publish_s cannot be inside the file
        # publication itself writes; it is logged instead.
        phase("publish")
        publish_round(
            work_root, output_num,
            records_by_file=build_round_records(mixture_files, mined, corpus_lookup,
                                                n_negs=n_negs,
                                                shard_qids=shard_qids),
            meta={'run_id': args.run_id, 'ann_no': output_num,
                  'checkpoint': str(next_checkpoint), 'checkpoint_step': step,
                  'n_queries_mined': len(mined), 'n_sampling_failures': len(failures),
                  'sampling_failures': failures[:20], 'corpus_sha256': corpus_sha,
                  'dev_ndcg_cut_10': dev_ndcg, 'source_checkpoint_step': step,
                  **shard, **phase.timings})
        phase.close()
        print(f"[Inferencer] Round #{output_num} committed from checkpoint-{step} "
              f"(shard {shard['shard_index']}/{shard['n_shards']}, {len(mined)} "
              f"queries). Waiting for the next checkpoint...", flush=True)

        # Free ~36GB of encode pickles. Round data itself is retained.
        shutil.rmtree(encode_dir, ignore_errors=True)

        phase.timings.clear()
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
