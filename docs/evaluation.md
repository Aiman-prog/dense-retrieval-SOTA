# Evaluation Stage

## Purpose

The evaluation stage compares sparse and dense retrieval systems on BRIGHT using
the same per-domain corpora, queries, relevance judgments, and exclusion lists.
It measures retrieval quality only; preprocessing creates the evaluation
artifacts, while training and hard-negative mining are completed beforehand.

## Evaluation Flow

```text
corpus + queries ──> BM25 search or dense encoding + FAISS search
                  ──> remove BRIGHT excluded documents
                  ──> retain top-k eligible results
                  ──> score against qrels
                  ──> aggregate across domains
```

All domains are read from the processed BRIGHT artifacts described in
[`preprocessor.md`](preprocessor.md). Evaluation settings, including domains,
batch size, retrieval depth, pooling, normalization, and sequence lengths, come
from [`config/config.yaml`](../config/config.yaml).

## Retrieval Methods

Dense models encode every document and query with Tevatron using the configured
pooling and normalization settings. Document embeddings are indexed with an exact
FAISS inner-product index, and each query embedding is searched against that
index. The same evaluator is used for standalone checkpoints and for models
produced by in-batch, cross-batch, ANCE, and GRASS training.

BM25 is the sparse baseline. It indexes the same domain corpus with
Lucene/Pyserini and retrieves results using the configured BM25 parameters. Its
ranking algorithm differs from dense retrieval, but its queries, judgments,
exclusion handling, and reported metrics are shared with the dense evaluation.

## BRIGHT Exclusion Protocol

BRIGHT provides a query-specific set of documents that must not be evaluated.
These documents are removed **before** top-k truncation and metric computation.
The search therefore retrieves beyond the requested cutoff by enough positions
to replace excluded hits, filters those hits, and then retains the highest-scored
eligible documents. This prevents excluded documents from consuming ranking
slots and unfairly shortening the evaluated result list.

## Metrics and Aggregation

NDCG@10 is the primary BRIGHT measure. MRR and Recall@1000 are also recorded by
the standalone BM25 and dense runners. Metrics are computed from TREC-format
binary relevance judgments. A judged query missing from a run contributes zero;
queries without judgments do not change the denominator. The final BRIGHT score
is the unweighted mean of per-domain NDCG@10 values, so every configured domain
has equal influence regardless of its query count.

Before retrieval, evaluation verifies that all required files are non-empty,
query IDs match the exclusion-map keys, and qrel query IDs belong to the query
set. Dense evaluation additionally verifies that encoding did not drop or invent
queries. A missing or failed domain prevents a summary from being reported.
Embedding caches and results are keyed by the resolved model path so checkpoints
with the same directory name cannot exchange artifacts.

## Running Evaluation

Evaluate any dense checkpoint across the configured domains with:

```bash
python scripts/run_all_evals.py --model_path /absolute/path/to/model
```

On DelftBlue, submit the corresponding Singularity launcher with
`EVAL_MODEL_PATH` set to the checkpoint path. Run the BM25 baseline separately
with `scripts/run_bm25_evals.py`. A formal sparse/dense comparison uses
`--compare_bm25 <summary.json>`; the BM25 summary must carry matching hashes for
the corpus, queries, qrels, and exclusions. Legacy hashless summaries must be
regenerated because their evaluation inputs cannot be proven identical.
