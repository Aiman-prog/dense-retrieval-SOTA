"""Standalone MS MARCO evaluation for a trained ANCE model."""
import os
import sys
import pickle
import argparse
import numpy as np
import pandas as pd
import faiss
from pathlib import Path

os.environ["TRANSFORMERS_ATTENTION_IMPLEMENTATION"] = "eager"
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root / 'src'))

from utils.helpers import get_path, get_training_context, load_config, encode_to_pickle
from evaluation.trec_eval_wrapper import TrecEvalWrapper


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', required=True)
    parser.add_argument('--recipe', default='ance_msmarco')
    args = parser.parse_args()

    ctx    = get_training_context(args.recipe)
    config = load_config()
    p      = get_path("processed")

    corpus_file  = p / ctx['args']['eval_corpus_file']
    queries_file = p / ctx['args']['eval_queries_file']
    qrels_file   = p / ctx['args']['eval_qrels_file']

    for f in [corpus_file, queries_file, qrels_file]:
        if not f.exists():
            raise FileNotFoundError(f"Missing: {f}")

    eval_dir = get_path(ctx['args']['temp_workdir']) / "final_eval"
    eval_dir.mkdir(parents=True, exist_ok=True)
    corpus_pkl = eval_dir / "c.pkl"
    query_pkl  = eval_dir / "q.pkl"

    if not corpus_pkl.exists():
        print("Encoding corpus (8.8M passages)...", flush=True)
        encode_to_pickle(args.model_path, corpus_file, corpus_pkl, False, ctx, config)
    else:
        print(f"Reusing cached corpus embeddings: {corpus_pkl}", flush=True)

    print("Encoding dev queries...", flush=True)
    encode_to_pickle(args.model_path, queries_file, query_pkl, True, ctx, config)

    with open(corpus_pkl, 'rb') as f: dc = pickle.load(f)
    with open(query_pkl,  'rb') as f: dq = pickle.load(f)

    print(f"FAISS search: {len(dq[1])} queries × {len(dc[1])} passages...", flush=True)
    idx = faiss.IndexFlatIP(dc[0].shape[1])
    idx.add(dc[0].astype(np.float32))
    top_k = ctx['args'].get('eval_top_k', 1000)
    scores, indices = idx.search(dq[0].astype(np.float32), top_k)

    results = {
        str(dq[1][j]): {
            str(dc[1][indices[j][k]]): float(scores[j][k])
            for k in range(top_k) if indices[j][k] >= 0
        }
        for j in range(len(dq[1]))
    }

    qrels_df = pd.read_csv(qrels_file, sep=' ',
                           names=['query_id', 'ignore', 'doc_id', 'relevance'], dtype=str)
    evaluator = TrecEvalWrapper(qrels_df)
    metric    = ctx['args']['eval_metric']
    metrics   = evaluator.evaluate(results, {metric, 'recip_rank'})

    print(f"\n📈 MS MARCO Dev — MRR@10: {metrics.get('recip_rank', 0):.4f}", flush=True)


if __name__ == "__main__":
    main()
