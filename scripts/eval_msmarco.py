"""MS MARCO Dev evaluation for a trained ANCE model.

Reports what the ANCE paper reports: **MRR@10** and **Recall@1000** (Table 1,
0.330 / 0.959 for passage ANCE at 600K steps).

Two things this file used to get wrong, both of which invalidate the number:

* it reused `final_eval/c.pkl` whenever the file existed, under a path that carried
  no model identity at all, so the second model evaluated scored against the FIRST
  model's 8.8M passage embeddings. The cache is now keyed by content -- the model
  weights, config and tokenizer, the corpus, and the encoding settings -- because
  ANCE overwrites its output directory in place on every run, so a path-based tag
  goes stale by construction.
* it reported `recip_rank` over a depth-1000 run, which is MRR@1000, not MRR@10.
  `pytrec_eval` has no `recip_rank_cut`, so MRR@10 is computed by truncating the run
  to depth 10 before scoring. Recall@1000 uses the full-depth run.
"""
import os
import sys
import json
import pickle
import argparse
import numpy as np
import faiss
from pathlib import Path

os.environ["TRANSFORMERS_ATTENTION_IMPLEMENTATION"] = "eager"
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root / 'src'))
sys.path.append(str(project_root / 'scripts'))

from utils.helpers import (get_path, get_data_base_dir, get_training_context,
                           load_config, encode_to_pickle, model_run_tag, _sha256,
                           _load_qrels, require_eval_files, load_training_manifest,
                           check_eval_artifacts,
                           encoding_contract_drift, training_provenance,
                           atomic_write, RUN_MANIFEST_NAME)
from evaluation.trec_eval_wrapper import TrecEvalWrapper

# The official MS MARCO Dev "small" split every published MRR@10 is measured on.
MSMARCO_DEV_QUERIES = 6980
CACHE_SIDECAR = "corpus_cache.json"


def msmarco_paper_comparable(num_judged_queries):
    """Only the official Dev small denominator supports the paper comparison."""
    return int(num_judged_queries) == MSMARCO_DEV_QUERIES


# Table 1, passage ANCE (FirstP) at 600K steps.
PAPER_MRR_AT_10 = 0.330
PAPER_RECALL_AT_1000 = 0.959
REPRODUCTION_TOLERANCE = 0.005


def reproduction_verdict(mrr10, recall1000, paper_comparable):
    """The pass/fail this arm exists to produce, and the deltas behind it.

    `None` when the run is not measured on the official Dev small split: a verdict
    computed on a different denominator would be a comparison to nothing. The
    tolerance is fixed here and stated in the summary so a later reader sees the bar
    the run was held to, rather than one chosen after seeing the number.
    """
    deltas = {'mrr_at_10': round(float(mrr10) - PAPER_MRR_AT_10, 6),
              'recall_1000': round(float(recall1000) - PAPER_RECALL_AT_1000, 6)}
    if not paper_comparable:
        return None, deltas
    return all(abs(d) <= REPRODUCTION_TOLERANCE for d in deltas.values()), deltas


def _model_identity(model_path):
    """Content hashes of everything that decides what this checkpoint encodes.

    `model_run_tag()` hashes the resolved PATH, which is not identity: ANCE writes
    its final model to the same `models/<name>/` on every run, so two different
    checkpoints share a tag. Weights, config and tokenizer are what actually change.
    """
    model_path = Path(model_path)
    parts = {}
    for name in ("model.safetensors", "pytorch_model.bin", "config.json",
                 "tokenizer.json", "tokenizer_config.json", "sentencepiece.bpe.model"):
        f = model_path / name
        if f.is_file():
            parts[name] = _sha256(f)
    if not parts:
        raise FileNotFoundError(
            f"{model_path} holds none of the files that identify a checkpoint; "
            f"refusing to key an embedding cache on a path alone.")
    return parts


def _encoding_identity(ctx):
    """From the EFFECTIVE model config, so a recipe's length override is part of the
    cache identity rather than silently absent from it."""
    model_cfg = ctx['model_cfg']
    return {k: model_cfg.get(k) for k in ("pooling", "normalize", "passage_max_len")}


def _reuse_corpus_cache(corpus_pkl, expected):
    """True only when the sidecar proves the cache was built from `expected`."""
    sidecar = corpus_pkl.parent / CACHE_SIDECAR
    if not corpus_pkl.is_file() or not sidecar.is_file():
        return False
    try:
        recorded = json.loads(sidecar.read_text())
    except ValueError:
        return False
    if recorded == expected:
        return True
    differing = sorted(k for k in set(recorded) | set(expected)
                       if recorded.get(k) != expected.get(k))
    print(f"⚠️  Corpus embedding cache is stale (differing: {differing}); "
          f"re-encoding.", flush=True)
    return False


def _truncate(run_results, depth):
    """Top-`depth` per query, by score. MRR@10 is recip_rank on a depth-10 run."""
    return {qid: dict(sorted(docs.items(), key=lambda kv: -kv[1])[:depth])
            for qid, docs in run_results.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model_path', required=True)
    parser.add_argument('--recipe', default='ance_msmarco')
    parser.add_argument('--results_json', default=None)
    parser.add_argument('--allow-config-drift', dest='allow_config_drift',
                        action='store_true',
                        help="evaluate despite the checkpoint having been trained "
                             "under different pooling/normalization/sequence "
                             "lengths; the exact values are recorded.")
    args = parser.parse_args()

    ctx    = get_training_context(args.recipe)
    config = load_config()
    p      = get_path("processed")

    corpus_file  = p / ctx['args']['eval_corpus_file']
    queries_file = p / ctx['args']['eval_queries_file']
    qrels_file   = p / ctx['args']['eval_qrels_file']
    require_eval_files("msmarco_dev", [corpus_file, queries_file, qrels_file])

    qrels = _load_qrels(qrels_file)
    paper_comparable = msmarco_paper_comparable(len(qrels))
    if not paper_comparable:
        print(f"⚠️  {len(qrels)} judged dev queries, not the {MSMARCO_DEV_QUERIES} of "
              f"the official Dev small split. The MRR@10 below is NOT directly "
              f"comparable to published MS MARCO numbers.", flush=True)

    # The encoding contract the checkpoint was trained under. A model trained with
    # CLS pooling at passage_max_len 512 does not mean the same thing encoded
    # mean-pooled at 128, and nothing else here would catch it.
    train_manifest = load_training_manifest(args.model_path)
    drift = encoding_contract_drift(train_manifest, ctx['model_cfg'])
    if train_manifest is None:
        print(f"⚠️  No {RUN_MANIFEST_NAME} for {args.model_path}; recording "
              f"training_manifest: null.", flush=True)
    if drift and not args.allow_config_drift:
        for key, vals in sorted(drift.items()):
            print(f"   {key}: trained={vals['checkpoint']!r}  "
                  f"evaluating={vals['evaluation']!r}", flush=True)
        raise SystemExit(
            "❌ Evaluation settings differ from the checkpoint's training contract. "
            "Match config/config.yaml, or pass --allow-config-drift.")

    identity = {'model': _model_identity(args.model_path),
                'corpus_sha256': _sha256(corpus_file),
                'encoding': _encoding_identity(ctx)}

    eval_dir = get_path(ctx['args']['temp_workdir']) / "final_eval" / \
        model_run_tag(args.model_path)
    eval_dir.mkdir(parents=True, exist_ok=True)
    corpus_pkl, query_pkl = eval_dir / "c.pkl", eval_dir / "q.pkl"

    if _reuse_corpus_cache(corpus_pkl, identity):
        print(f"♻️  Reusing corpus embeddings: {corpus_pkl}", flush=True)
    else:
        print("Encoding corpus (8.8M passages)...", flush=True)
        (eval_dir / CACHE_SIDECAR).unlink(missing_ok=True)   # never a stale claim
        encode_to_pickle(args.model_path, corpus_file, corpus_pkl, False, ctx, config)
        with atomic_write(eval_dir / CACHE_SIDECAR) as f:
            json.dump(identity, f, indent=2, default=str)

    print("Encoding dev queries...", flush=True)
    encode_to_pickle(args.model_path, queries_file, query_pkl, True, ctx, config)

    with open(corpus_pkl, 'rb') as f: dc = pickle.load(f)
    with open(query_pkl,  'rb') as f: dq = pickle.load(f)

    # Source, judged and encoded query ids must agree BEFORE the search. An encoder
    # that dropped or invented a query breaks the correspondence between the run and
    # the qrels, and the only symptom is a quietly wrong MRR. `excluded=None` says
    # MS MARCO has no exclusion map -- unlike BRIGHT, where a missing one is a bug.
    encoded_query_ids = [str(x) for x in dq[1]]
    check_eval_artifacts("msmarco_dev", qrels, None, queries_file=queries_file,
                         encoded_query_ids=encoded_query_ids)

    top_k = ctx['args']['eval_top_k']
    print(f"FAISS search: {len(dq[1])} queries × {len(dc[1])} passages, "
          f"depth {top_k}...", flush=True)
    idx = faiss.IndexFlatIP(dc[0].shape[1])
    idx.add(dc[0].astype(np.float32))
    depth = min(top_k, len(dc[1]))
    scores, indices = idx.search(dq[0].astype(np.float32), depth)

    run = {
        encoded_query_ids[j]: {str(dc[1][indices[j][k]]): float(scores[j][k])
                               for k in range(depth) if indices[j][k] >= 0}
        for j in range(len(encoded_query_ids))
    }

    evaluator = TrecEvalWrapper(qrels)
    # MRR@10 is recip_rank over a run truncated to depth 10; pytrec_eval has no
    # recip_rank_cut, and recip_rank over the full depth-1000 run is MRR@1000.
    mrr10 = evaluator.evaluate(_truncate(run, 10), {'recip_rank'})['recip_rank']
    deep  = evaluator.evaluate(run, {'recall_1000', 'ndcg_cut_10', 'recip_rank'})

    print("\n" + "=" * 56, flush=True)
    print(f"  MS MARCO Dev — {Path(args.model_path).name}", flush=True)
    print("=" * 56, flush=True)
    paper_mrr = "   (paper ANCE: 0.330)" if paper_comparable else ""
    paper_recall = "   (paper ANCE: 0.959)" if paper_comparable else ""
    print(f"  MRR@10        : {mrr10:.4f}{paper_mrr}", flush=True)
    print(f"  Recall@1000   : {deep['recall_1000']:.4f}{paper_recall}", flush=True)
    print(f"  NDCG@10       : {deep['ndcg_cut_10']:.4f}", flush=True)
    print(f"  MRR@{depth:<9d}: {deep['recip_rank']:.4f}   (not the published metric)",
          flush=True)
    reproduction_pass, deltas = reproduction_verdict(
        mrr10, deep['recall_1000'], paper_comparable)
    if reproduction_pass is None:
        print(f"  Reproduction  : n/a — {len(qrels)} judged queries, not the official "
              f"Dev small {MSMARCO_DEV_QUERIES}", flush=True)
    else:
        verdict = "PASS" if reproduction_pass else "FAIL"
        print(f"  Reproduction  : {verdict}  (±{REPRODUCTION_TOLERANCE} of "
              f"{PAPER_MRR_AT_10}/{PAPER_RECALL_AT_1000}; Δ MRR@10 "
              f"{deltas['mrr_at_10']:+.4f}, Δ R@1000 {deltas['recall_1000']:+.4f})",
              flush=True)
    print("=" * 56, flush=True)

    out = Path(args.results_json) if args.results_json else (
        Path(get_data_base_dir()) / config['paths']['results_dir']
        / model_run_tag(args.model_path) / "msmarco_summary.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    with atomic_write(out) as f:
        json.dump({
            'benchmark': 'msmarco_dev',
            'model': str(Path(args.model_path).resolve()),
            'run_tag': model_run_tag(args.model_path),
            'recipe': args.recipe,
            'search_depth': depth,
            'num_judged_queries': len(qrels),
            'paper_comparable': paper_comparable,
            'reproduction_pass': reproduction_pass,
            'reproduction_target': {'mrr_at_10': PAPER_MRR_AT_10,
                                    'recall_1000': PAPER_RECALL_AT_1000,
                                    'tolerance': REPRODUCTION_TOLERANCE},
            'reproduction_delta': deltas,
            'metrics': {'mrr_at_10': mrr10,
                        'recall_1000': deep['recall_1000'],
                        'ndcg_cut_10': deep['ndcg_cut_10'],
                        f'recip_rank_at_{depth}': deep['recip_rank']},
            'eval_artifact_sha256': {'msmarco_dev': {
                'corpus': _sha256(corpus_file), 'queries': _sha256(queries_file),
                'qrels': _sha256(qrels_file), 'excluded': None}},
            'encoding_identity': identity,
            'training_manifest': training_provenance(train_manifest),
            'config_drift': drift or None,
        }, f, indent=2, default=str)
    print(f"📄 Summary written to {out}", flush=True)


if __name__ == "__main__":
    main()
