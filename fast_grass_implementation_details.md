# Fast-GRASS Implementation Details

This file complements `fast_grass_negative_cache_architecture.md`. The architecture is fixed; the choices below are current engineering defaults, not theoretical claims from Negative Cache.

## Current Defaults

| Parameter | Default | Notes |
|---|---:|---|
| `B_doc` | `100_000` | Global cache size; ablate `32k`, `100k`, `512k`. |
| `m` | `1` | One selected negative per query for the first run. |
| `selection_mode` | `TopK` | `Softmax` is optional after TopK works. |
| `lambda_val` | `1.0` | Also run required baseline `lambda_val = 0`. |
| `beta` | `5.0` | Used only for Softmax mode. |
| `L` | `128` | MCDP lazy top-`L` shortlist inside `H` (also the Softmax prefilter). MCDP cost ≈ `batch_size * L * T`, so keep it small — first real runs may prefer `L≈50–64`; larger `L` (e.g. `1024`) is only cheap for Softmax/EMA. |
| `uncertainty` | `MCDP` | Current default uncertainty estimator; EMA remains a diagnostic/baseline. |
| `T` | `3` | MC-dropout stochastic passes on the top-`L` shortlist. |
| `mc_dropout_p` | `0.3` | Dropout probability for MCDP passes; matches current GRASS config. |
| `ema_decay` | `0.999` | EMA diagnostic only; maps to current config key `ema_alpha`. |
| `rho_start` | `0.50` | Initial linear-decay maintenance fraction; higher early while the model changes quickly. |
| `rho_end` | `0.10` | Final linear-decay maintenance fraction. |
| `cache_update_interval` | `100` steps | Batch cache maintenance for efficient encoding. |
| `steps_per_epoch` | derived | Compute from dataloader length. |
| `max_age_epochs` | `4` | Full cache turnover bound target. |
| `max_age_steps` | `max_age_epochs * steps_per_epoch` | Step-level staleness cap. |
| `utility_ema_decay` | `0.95` | Current utility smoothing. |
| `utility_floor` | `0.01` | Low-utility threshold for binary selected/not-selected EMA. |
| `utility_remember_threshold` | `0.05` | Minimum peak utility for admitting an evicted doc to `R` if it was never selected. |
| `K` | `3` | Replace eligibility also triggers after zero recent selections for K maintenance intervals. |
| `R_fraction` | `0.25` | Fraction of replacement candidates nominated by retired challenger registry `R`. |
| `uniform_candidate_fraction` | `0.75` | Uniform corpus candidates remain dominant; keep `>= 0.75` in the first implementation. |
| `replacement_candidate_multiplier` | `2` | Engineering constant: recertify 2x the number of final replacement slots. |
| `recent_query_reservoir_size` | `128` | Engineering constant for recertification query probe. |
| `reentry_top_k` | `5` | Average the top-5 valid `g(q,d)` scores for candidate re-entry. |
| `R_size` | `0.5 * B_doc` | Engineering constant: bounded retired challenger registry size. |
| `cache_init_seed` | training seed | Reproducible uniform initialization. |

Initialize `H` by uniformly sampling `B_doc` corpus docids without replacement using `cache_init_seed`. Do not exclude positives globally; mask known positives/qrels per query during mining.

## Scoring And Selection

EMA mode maintains EMA copies of both query and document encoders.

```text
s_student = q_student dot d_student_cached
s_teacher = q_teacher dot d_teacher_cached
s_hat = s_student
sigma = abs(s_student - s_teacher)
g = s_hat + lambda_val * sigma
```

MCDP should use lazy uncertainty and top-`L`:

```text
cheap-score all H with deterministic/eval-mode cached doc embeddings
use one deterministic/eval-mode query pass for s_hat_cheap
keep top-L inside H by s_hat_cheap
compute T-pass query/document dropout uncertainty only on top-L
select by g
```

The paper-faithful MCDP estimator applies dropout to the query/document pair:

```text
for t in 1..T:
    q_t = encode query with dropout enabled
    d_t = encode each top-L doc with dropout enabled
    s_t(q,d) = q_t dot d_t

s_hat(q,d) = mean_t s_t(q,d)
sigma(q,d) = std_t s_t(q,d)
g(q,d) = s_hat(q,d) + lambda_val * sigma(q,d)
```

Do not compute MCDP over all `B_doc`; that is too expensive and not the intended Fast-GRASS
variant. Fast-GRASS MCDP uses full query/document dropout on the top-`L` shortlist;
query-side-only MCDP is not part of the first implementation.

Softmax mode, when enabled:

```text
mask positives/qrels
optionally keep top-L
logits = beta * g
logits = logits - max(logits)
sample m docs without replacement
```

For `m > 1`, implement weighted sampling without replacement with Gumbel-Top-k over the stabilized logits:

```text
u_i ~ Uniform(0, 1)
gumbel_i = -log(-log(u_i))
select top-m by logits_i + gumbel_i
```

This is only the Softmax-over-`g` sampling implementation, not the original Negative Cache estimator.

## Utility And Cache Updates

Use one age per cache entry:

```text
age = current_global_step - last_refreshed_step
age_norm = min(age / max_age_steps, 1)
```

Utility update runs at each cache update interval:

```text
selected_indicator[d] = 1 if d was selected at least once since the last cache update, else 0
utility_ema[d] = utility_ema_decay * utility_ema[d]
               + (1 - utility_ema_decay) * selected_indicator[d]
```

Reset `selected_indicator` after each cache update. Track lightweight
`selection_history` for replacement and `R` admission:

```text
selected_count_recent
lifetime_selected_count
peak_utility_ema
```

`selected_count_recent` is maintained over maintenance intervals for the
first-implementation replacement eligibility rule.

Compute interval budget:

```text
rho_maint = linear_decay(rho_start, rho_end, training_progress)
maintenance_budget_interval =
    round(rho_maint * B_doc * cache_update_interval / steps_per_epoch)

num_refresh + num_replace <= maintenance_budget_interval
```

Handle maintenance actions in this order:

```text
1. urgent over-age docs by utility:
   refresh useful over-age docs
   replace low-utility over-age docs
2. persistently low-utility replacements
3. remaining useful stale refreshes by refresh_priority
4. defer anything beyond the shared maintenance budget
```

Refresh useful stale docs:

```text
refresh_priority = utility_ema * age_norm
```

Replace low-utility stale docs:

```text
utility_ema <= utility_floor
OR selected_count_recent == 0 for K maintenance intervals
```

Useful stale docs should be refreshed, not evicted.

Retired challenger registry `R`:

```text
R stores docid-level metadata only
R never participates in per-batch query scoring
R only nominates candidates during maintenance
```

Docs enter `R` only if:

```text
lifetime_selected_count > 0
OR peak_utility_ema >= utility_remember_threshold
```

When `R` is full:

```text
R_keep_score = peak_utility_ema
tie_breaker = lifetime_selected_count
```

Replacement candidate generation:

```text
candidate_set = uniform_candidates ∪ R_candidates
uniform_candidate_fraction = 0.75
R_fraction = 0.25
num_candidate_docs = replacement_candidate_multiplier * num_replace
```

Uniform corpus candidates must remain dominant in the first implementation. Historical usefulness
only nominates docs from `R`; current recertification decides re-entry.

Recertification:

```text
encode candidate docs with the current doc encoder under no_grad
score candidates against recent_query_reservoir
apply the same known-positive/qrel mask used during mining
candidate_reentry_score = average top-k g(q, d)
reentry_top_k = 5
insert only top num_replace candidates into H
```

## Baselines And Logging

Baseline variants:

```text
A: no R, uniform corpus replacements only
B: R nomination without recertification (diagnostic only)
C: R nomination + current recertification
D: max-g vs average top-k-g candidate_reentry_score
```

Required ablations:

```text
lambda_val = 0 vs 1
B_doc = 32k vs 100k vs 512k
m = 1 vs 4
TopK vs Softmax
no R vs R + recertification
```

Cost log fields:

```text
global_step
B_doc
selection_mode
num_queries
num_selected_negatives
doc_encoder_calls_loss
doc_encoder_calls_cache_refresh
doc_encoder_calls_cache_replace
cache_score_pairs
num_refresh
num_replace
num_over_age
over_age_backlog
num_R_entries
num_R_candidates
num_uniform_candidates
num_recertified_candidates
replacement_yield_at_K
selected_doc_diversity
cache_turnover_rate
ann_queries
index_rebuilds
step_wall_time
```

Retrieval metrics:

```text
MRR@10
NDCG@10
Recall@20
Recall@100
```
