"""
Standalone BRIGHT eval for a finished Fast-GRASS checkpoint — PARALLEL-SAFE.

Each invocation writes all eval scratch (per-domain corpus/query encodings) into a
PER-MODEL directory (default ``<model_dir>/eval_scratch``), so you can run one eval
job per checkpoint **in parallel** without the runs clobbering each other. This is a
self-contained reimplementation of the BRIGHT branch of utils.helpers.evaluate_bright
(which hardcodes a single shared scratch dir under temp_grass_workdir, hence unsafe to
run concurrently) — no edits to helpers.py / run_grass.py / negative_cache.py.

Parallel sweep (one job per model, different GPUs):
    for d in /scratch/$USER/.../models/*fg_*_ema; do
        FG_EVAL_MODEL_DIR="$d" sbatch scripts/run_fast_grass_eval_singularity.sh
    done

Metrics match evaluate_bright: per-domain NDCG@10 (+ MRR) and the mean NDCG@10.
"""
import argparse
import pickle
import sys
from pathlib import Path

import numpy as np

project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root / 'src'))

from utils.helpers import load_config, get_training_context, get_path, encode_to_pickle


def evaluate_bright_isolated(ctx, config, model_path, scratch_dir):
    """BRIGHT multi-domain (or single-set) eval writing to an isolated scratch_dir.

    Faithful to utils.helpers.evaluate_bright, but every intermediate path is under
    ``scratch_dir`` instead of the shared temp_grass workdir, so concurrent evals of
    different checkpoints never race.
    """
    import faiss
    import pandas as pd
    from evaluation.trec_eval_wrapper import TrecEvalWrapper

    args = ctx['args']
    scratch_dir = Path(scratch_dir)
    processed = get_path("processed")

    def _search_and_score(d_corpus, d_queries, d_qrels, eval_dir, top_k, metrics_set):
        eval_dir.mkdir(parents=True, exist_ok=True)
        encode_to_pickle(str(model_path), d_corpus,  eval_dir / "c.pkl", False, ctx, config)
        encode_to_pickle(str(model_path), d_queries, eval_dir / "q.pkl", True,  ctx, config)
        with open(eval_dir / "c.pkl", 'rb') as f: dc = pickle.load(f)
        with open(eval_dir / "q.pkl", 'rb') as f: dq = pickle.load(f)
        idx = faiss.IndexFlatIP(dc[0].shape[1])
        idx.add(dc[0].astype(np.float32))
        s_e, i_e = idx.search(dq[0].astype(np.float32), top_k)
        results = {
            str(dq[1][j]): {str(dc[1][i_e[j][k]]): float(s_e[j][k])
                            for k in range(len(i_e[j])) if i_e[j][k] >= 0}
            for j in range(len(dq[1]))
        }
        qrels_rows = []
        with open(d_qrels) as f:
            for line in f:
                p = line.strip().split()
                if len(p) >= 4:
                    qrels_rows.append({'query_id': p[0], 'doc_id': p[2], 'relevance': p[3]})
        evaluator = TrecEvalWrapper(pd.DataFrame(qrels_rows))
        return evaluator.evaluate(results, metrics_set)

    # single-set path (only if eval_corpus_file is configured; BRIGHT multi-domain otherwise)
    if args.get('eval_corpus_file'):
        d_corpus  = processed / args['eval_corpus_file']
        d_queries = processed / args['eval_queries_file']
        d_qrels   = processed / args['eval_qrels_file']
        if not all(x.exists() for x in [d_corpus, d_queries, d_qrels]):
            print("[Eval] Skipping: eval files not found", flush=True)
            return
        metric = args.get('eval_metric', 'ndcg_cut_10')
        m = _search_and_score(d_corpus, d_queries, d_qrels,
                              scratch_dir / "final_eval",
                              args.get('eval_top_k', 1000), {metric})
        print(f"\n📈 Eval — {metric}={m.get(metric, 0):.4f}", flush=True)
        return

    eval_summary = []
    for domain in config['evaluation'].get('eval_domains', []):
        d_corpus  = processed / f"{domain}_corpus.jsonl"
        d_queries = processed / f"{domain}_queries.jsonl"
        d_qrels   = processed / f"{domain}_qrels.txt"
        if not all(p.exists() for p in [d_corpus, d_queries, d_qrels]):
            print(f"[Eval] Skipping {domain}: files not found", flush=True)
            continue
        m = _search_and_score(d_corpus, d_queries, d_qrels,
                              scratch_dir / "final_eval" / domain,
                              args.get('eval_top_k', 10),
                              {'recip_rank', 'ndcg_cut_10'})
        eval_summary.append({'domain': domain,
                             'ndcg10': m.get('ndcg_cut_10', 0),
                             'mrr': m.get('recip_rank', 0)})
        print(f"[Eval] {domain}: NDCG@10={m.get('ndcg_cut_10', 0):.4f} "
              f"MRR={m.get('recip_rank', 0):.4f}", flush=True)
    if eval_summary:
        df = pd.DataFrame(eval_summary)
        print(f"\n📈 Final Mean NDCG@10: {df['ndcg10'].mean():.4f} "
              f"| Mean MRR: {df['mrr'].mean():.4f}", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--model_dir', required=True,
                    help='path to a trained Fast-GRASS checkpoint / final model dir')
    ap.add_argument('--scratch_dir', default=None,
                    help='per-model eval scratch (default <model_dir>/eval_scratch); '
                         'keep it model-specific so parallel evals never collide')
    args = ap.parse_args()

    model_dir = Path(args.model_dir)
    if not model_dir.exists():
        sys.exit(f"[FAST-GRASS EVAL] model_dir not found: {model_dir}")
    scratch_dir = Path(args.scratch_dir) if args.scratch_dir else model_dir / "eval_scratch"

    config = load_config()
    ctx    = get_training_context('fast_grass')
    print(f"[FAST-GRASS EVAL] Evaluating {model_dir}", flush=True)
    print(f"[FAST-GRASS EVAL] scratch  {scratch_dir}", flush=True)
    evaluate_bright_isolated(ctx, config, model_dir, scratch_dir)
    print(f"[FAST-GRASS EVAL] Done: {model_dir}", flush=True)


if __name__ == "__main__":
    main()
