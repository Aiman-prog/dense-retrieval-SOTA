# Negative-Cache-Inspired Fast-GRASS Cache Architecture

## Goal

Current GRASS is slow because mining still does expensive per-query candidate work:

```text
query -> stale FAISS top-P -> fresh-encode/rerank P docs -> uncertainty on top-L -> select negative
```

Fast-GRASS keeps the GRASS selection idea,

```text
g(q,d) = s_hat(q,d) + lambda * sigma(q,d)
```

but borrows the Negative Cache architecture: use a bounded global cache of stale document states for cheap negative selection, then fresh-encode only the selected positives and negatives for the training loss.

This is not a direct Negative Cache implementation. Fast-GRASS borrows the cache architecture, but replaces Negative Cache's similarity-based Gumbel-Max sampling objective with GRASS uncertainty-aware selection over `g(q,d)`.

## Borrowed, Replaced, Added

Borrowed from Negative Cache:

```text
global bounded cache H
stale cached document states Z_H
bounded cache maintenance through partial refresh/replacement
fresh encoding for selected training docs
```

Replaced:

```text
Negative Cache's similarity-based Gumbel-Max softmax-gradient estimator as the mining/training objective
```

Added from GRASS:

```text
g(q,d) = s_hat(q,d) + lambda * sigma(q,d)
```

This means the Negative Cache convergence or unbiased-gradient claims should not be transferred to Fast-GRASS. The claim is only architectural: Negative-Cache-inspired caching can reduce mining cost.

## Main Objects

```text
H      = bounded global document pool
B_doc  = fixed size of H
Z_H    = cached document representations for documents in H
R      = retired challenger registry with docid-level metadata only
```

`H` is global, not per-query. Every query scores against the same `H`.

```text
q1 -> H
q2 -> H
q3 -> H
```

This is different from current GRASS:

```text
q1 -> C_q1 = stale top-P for q1
q2 -> C_q2 = stale top-P for q2
q3 -> C_q3 = stale top-P for q3
```

`R` is not a second cache. It never participates in per-batch query scoring and
does not store embeddings for mining. It only nominates previously useful
documents during cache maintenance.

## Initialization

Fast-GRASS can reuse the current GRASS startup artifact:

```text
full corpus -> encode once with base model -> stale corpus embeddings / FAISS index
```

This full-corpus stale index is built once at the beginning, as in current GRASS. The difference is what happens after that.

Current GRASS uses the stale index during every mining step:

```text
for each query:
    retrieve top-P from stale FAISS
```

Fast-GRASS should not do that. Instead:

1. Build or load the stale full-corpus embeddings once.
2. Select `B_doc` document ids from the full corpus to initialize `H`.
3. Initialize `Z_H` for only those `B_doc` documents.
4. During training, score queries against `H`, not against per-query FAISS top-`P`.

First implementation:

```text
H = uniform random sample of B_doc docs from the corpus
```

If the stale corpus embedding pickle already exists, the initial deterministic embeddings for selected docs can be copied from it. For estimator-specific states, initialize `Z_H` for the selected docs:

```text
MCDP:     Z_H[d] = T stochastic document embeddings
EMA:      Z_H[d] = student embedding + EMA teacher embedding
Ensemble: Z_H[d] = E ensemble embeddings
```

The stale full-corpus index is therefore only an initialization/source artifact. It is not the per-query mining mechanism anymore.

## Mining Step

For a minibatch, fresh-encode the queries for selection:

```text
Q_batch = encode_queries(batch queries)
```

Then score every query against cached document states in `H`:

```text
scores = Q_batch x Z_H
```

For uncertainty-aware modes, `Z_H` stores the document-side states needed to compute both `s_hat(q,d)` and `sigma(q,d)`.

For cheap estimators such as EMA, score all of `H`. For expensive estimators such as MCDP, first cheap-score `H`, keep an optional top-`L` inside `H`, then compute uncertainty only on that lazy shortlist.

For each query, mask all known positives/qrels before selection if they are present in `H`. Unknown false negatives can still occur, especially in dense QA corpora, so track them qualitatively when inspecting mined negatives.

Then compute the GRASS score:

```text
g(q,d) = s_hat(q,d) + lambda * sigma(q,d)
```

Select negatives per query using the configured selection mode.

Example:

```text
H = [d2, d4, d5, d7, d10]

q1 positive d1
q2 positive d5
q3 positive d9
```

All three queries score against the same cache:

```text
        d2    d4    d5    d7    d10
q1      g     g     g     g     g
q2      g     g    MASK   g     g
q3      g     g     g     g     g
```

If `m = 1`, select one negative per query.

## Selection Modes

First implementation default:

```text
selection_mode = TopK
score all docs in H by g(q,d)
select the top-m docs by g
```

Vision2026-compatible optional mode:

```text
selection_mode = Softmax
score all docs in H by g(q,d)
optionally keep top-L inside H
sample m docs without replacement using p[d] proportional to exp(beta * g(q,d))
```

`TopK` can ignore `L` and select top-`m` directly from `H`. `Softmax` can use `L` to keep sampling focused and cheaper.

The `Softmax` selection mode samples over the GRASS score `g(q,d)`, not over the original Negative Cache softmax-gradient estimator.

## Training Step

After mining, Fast-GRASS trains exactly on the selected documents:

```text
fresh-encode batch queries
fresh-encode selected positives
fresh-encode selected negatives
compute contrastive loss
update model
```

The selected negatives are fresh for the loss. The cache is only used to choose them cheaply.

This is the key Negative Cache influence:

```text
cached stale states for selection
fresh encodings for gradient training
```

Gradients should never flow through stale `Z_H`. `Z_H` is selection-only. The contrastive loss must use fresh encodings of the selected positives and selected negatives.

## Cache Maintenance

Each active cache entry tracks core maintenance metadata:

```text
docid
age
utility_ema
selection_history
```

`selection_history` is lightweight metadata used for replacement and `R`
admission:

```text
selected_count_recent
lifetime_selected_count
peak_utility_ema
```

Use step-based age:

```text
age = current_global_step - last_refreshed_step
```

For v0, use one age per cache entry. Separate document-state and uncertainty-state ages are only needed later if parts of `Z_H[d]` are refreshed independently.

For v0, utility is a selection-frequency heuristic: selected documents update a binary `selected_indicator`, which is folded into `utility_ema` at cache-update time. Future variants may use normalized `g(q,d)` as the utility signal.

Maintenance runs periodically for batching efficiency, but replacement identity
is utility-triggered. The design removes arbitrary time-based eviction, not
periodic maintenance.

Use a linear-decay maintenance budget:

```text
rho_maint(step) = linear_decay(rho_start, rho_end, training_progress)
maintenance_budget_interval =
    round(rho_maint(step) * B_doc * cache_update_interval / steps_per_epoch)
```

Default:

```text
rho_start = 0.50
rho_end = 0.10
```

There is no fixed refresh/replace split:

```text
num_refresh + num_replace <= maintenance_budget_interval
```

Within each interval:

```text
handle urgent over-age docs by utility first:
    refresh useful over-age docs
    replace low-utility over-age docs
then replace persistently low-utility docs
then refresh remaining useful stale docs by refresh_priority
defer anything beyond the shared budget
```

This follows the Negative Cache principle that cache maintenance is bounded by
an update fraction, while the adaptive schedule and utility-triggered
replacement are Fast-GRASS/CASE-inspired engineering additions.

### Refresh

Refresh keeps the same document in `H` but recomputes its cached state:

```text
keep docid d
recompute Z_H[d]
reset/update age
```

Refresh useful stale documents:

```text
refresh_priority = utility_ema * age_norm
```

Useful stale documents should be refreshed, not evicted.

### Replace

Replace removes a persistently low-utility document and inserts a newly
recertified document:

```text
evict old docid from H
nominate candidates from corpus and R
recertify candidates with the current model under no_grad
compute Z_H[new_docid]
insert new docid into H
```

Replacement eligibility:

```text
utility_ema <= utility_floor
OR selected_count_recent == 0 for K maintenance intervals
```

Default:

```text
utility_floor = 0.01
K = 3
```

Uniform corpus exploration remains dominant:

```text
R_fraction = 0.25
uniform_candidate_fraction = 0.75
uniform_candidate_fraction >= 0.75 for v0
```

Build the replacement candidate set as:

```text
candidate_set = uniform_candidates ∪ R_candidates
```

Then:

```text
encode candidate docs with the current doc encoder under no_grad
score candidates against recent_query_reservoir
apply the same known-positive/qrel mask used during mining
candidate_reentry_score = average top-k g(q, d)
reentry_top_k = 5
insert only top num_replace candidates into H
```

Historical usefulness only nominates; current recertification decides re-entry.

### Retired Challenger Registry

When a document is evicted from `H`, it enters `R` only if it previously showed
usefulness:

```text
lifetime_selected_count > 0
OR peak_utility_ema >= utility_remember_threshold
```

`R` stores metadata only:

```text
docid
selection_history
last_seen_step
```

Bound `R` with:

```text
R_size = 0.5 * B_doc
```

When `R` is full, keep the strongest remembered documents by:

```text
R_keep_score = peak_utility_ema
tie_breaker = lifetime_selected_count
```

`R` is a registry, not a second mining cache. It never stores active document
states for per-batch scoring.

### Negative Cache Boundary

Negative Cache contributes:

```text
bounded active cache H
stale cached document states Z_H
fresh loss encoding
bounded maintenance-budget principle
```

Fast-GRASS adds:

```text
adaptive rho schedule
utility-triggered replacement
retired challenger registry R
current-model recertification
```

Do not transfer Negative Cache convergence or unbiased-gradient guarantees to
Fast-GRASS. The claim is architectural and empirical.

## What Happens To P And L

`P` is no longer the main mining budget.

Old GRASS:

```text
P = per-query stale FAISS retrieval depth
```

Fast-GRASS:

```text
B_doc = global cache size
```

`L` is also no longer the old fresh-reranked shortlist from top-`P`. In Fast-GRASS, `L` only means an optional cheap shortlist inside global `H` before final `TopK` or `Softmax` selection.

For the clean first implementation, `TopK` does not need `L`:

```text
score against H
select top-m by g
```

`L` is mainly useful for Softmax sampling or lazy uncertainty computation.

## Why This Should Be Faster

Current GRASS pays a mining cost roughly like:

```text
per query: fresh-encode up to P candidate documents
```

Fast-GRASS changes this to:

```text
per batch: score queries against cached Z_H
per query: fresh-encode only selected negatives for loss
global: refresh/replace a small cache budget
```

The expensive document encoder is no longer applied to `P` candidates per query. It is applied to:

```text
selected positives
selected negatives
cache refresh/replacement docs
```

This is the Negative Cache tradeoff: accept stale approximate selection to avoid repeated fresh candidate encoding, while keeping the actual training loss fresh.

## Evaluation Plan

Ablate:

```text
current GRASS per-query FAISS mining
Fast-GRASS A: no R, uniform corpus replacements only
Fast-GRASS B: R nomination without recertification (diagnostic only)
Fast-GRASS C: R nomination + current recertification
candidate_reentry_score: max-g vs average top-k-g
```

Track cost:

```text
document encoder calls per step
cache scoring cost
cache refresh/replacement cost
recertification encoder calls
recertification wall-clock overhead
ANN queries per epoch
index rebuild count
GPU-hours to target dev score
```

Track retrieval quality:

```text
MRR@10
NDCG@10
Recall@20
Recall@100
```

## Summary

Fast-GRASS is:

```text
Negative Cache structure + GRASS scoring
```

Negative Cache contributes:

```text
bounded global H
stale cached Z_H
bounded maintenance budget
fresh training only on selected docs
```

GRASS contributes:

```text
uncertainty-aware scoring:
g(q,d) = s_hat(q,d) + lambda * sigma(q,d)
```

The main architectural shift is:

```text
from per-query candidate pools C_q
to one global bounded cache H
```
