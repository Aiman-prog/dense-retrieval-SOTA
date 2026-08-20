# Fast-GRASS Lambda Internal Report

This note summarizes the internal Fast-GRASS settings and log-derived counters for the recent
`lambda=0` vs `lambda=1` runs. The goal is to make the architecture choices and observed behavior
easy to explain in a mentor meeting.

## Run Setup

| Setting | Value |
|---|---:|
| Base model | `inbatch_mixed_bge_m3` |
| Global cache size, `B_doc` | `32,000` |
| Epochs | `2` |
| Steps per epoch | `5,156` |
| Total steps | `10,312` |
| Batch size | `64` |
| Negatives per query, `m` | `1` |
| Selection mode | `TopK` |
| Uncertainty estimator | EMA |
| EMA teacher decay, `ema_alpha` | `0.999` |
| Query / passage max length | `1024 / 512` |
| Learning rate | `1e-5` |

No completed `ema_alpha=0.9999` or `ema_alpha=1.0` run was found in the available logs. The
completed `lambda=0` and `lambda=1` runs both used `ema_alpha=0.999`.

## Lambda Comparison

Fast-GRASS scores cached candidates with:

```text
g(q, d) = s_hat(q, d) + lambda_val * sigma(q, d)
```

For the EMA estimator:

```text
s_hat  = q_student dot d_student_cached
sigma  = abs(q_student dot d_student_cached - q_teacher dot d_teacher_cached)
```

| Run | Meaning |
|---|---|
| `lambda=0` | Ignore uncertainty: `g = s_hat` |
| `lambda=1` | Use uncertainty: `g = s_hat + sigma` |

## Negative Selection

Both runs selected the same number of negatives because they used the same batch count and
`m=1`.

| Run | Selected negatives |
|---|---:|
| `lambda=0`, `B_doc=32k` | `659,968` |
| `lambda=1`, `B_doc=32k` | `659,968` |

## Uncertainty Diagnostics

These diagnostics come from the newer `logs_cluster` runs, which added selection-specific fields.

| Metric | `lambda=0` | `lambda=1` |
|---|---:|---:|
| Mean selected `s_hat` | `0.7691` | `0.7518` |
| Mean selected `sigma` | `0.0230` | `0.0413` |
| Mean `lambda * sigma` | `0.0000` | `0.0413` |
| Mean selected `sigma / s_hat` | `3.0%` | `5.5%` |
| Top-1 flip rate vs `lambda=0` | `0.0%` | `40.7%` |

Interpretation: the rerun shows that EMA uncertainty was not inactive. With `lambda=1`, the
uncertainty term changed the selected top-1 negative for about 41% of queries. However, the final
eval scores remained close to `lambda=0`, so the current EMA uncertainty signal changed mining but
did not clearly improve quality.

## Cache Maintenance

| Metric | Value |
|---|---:|
| Cache maintenance interval | every `100` steps |
| Number of maintenance intervals | `103` |
| Maintenance schedule | `rho_start=0.50` to `rho_end=0.10` linear decay |
| Total maintenance actions | `19,070` |
| Refreshes | `614` |
| Replacements | `18,456` |
| Replaced fraction of `H` | `57.7%` |
| Total maintenance action fraction of `H` | `59.6%` |
| Over-age docs | `0` |

The total `19,070` actions exactly matches the design budget:

```text
budget = round(rho_maint * B_doc * cache_update_interval / steps_per_epoch)
```

summed over the 103 maintenance intervals.

## Replacement And Recertification

| Setting / Metric | Value |
|---|---:|
| Replacement candidate multiplier | `2` |
| Recertified candidates | `36,912` |
| Uniform candidates in latest logs | `36,912` |
| `R` candidates in latest logs | `0` |
| Recent query reservoir size | `128` |
| Reentry score | average top-`5` valid `g(q,d)` |
| Positive/qrel masking | applied during mining and recertification |

Replacement is not purely random. Candidate generation is partly random because uniform corpus
documents are sampled, but candidates are recertified against recent query embeddings before entering
`H`.

## Retired Challenger Registry R

`R` is a metadata-only registry. It is not a second cache and never participates in per-batch
query scoring.

| Rule / Metric | Value |
|---|---|
| Stores | docid-level metadata only |
| Per-batch scoring? | No |
| Nomination time | maintenance only |
| Entry rule | `lifetime_selected_count > 0 OR peak_utility_ema >= 0.05` |
| Size limit for `B_doc=32k` | `0.5 * B_doc = 16,000` |
| Intended candidate mix | `75%` uniform, `25%` from `R` |
| Latest logs | `R` unused: `0` entries/candidates |

Older analysis logs did show some `R` activity, but it was weak:

| Older run | Max `R` entries | `R` candidates | `R` share of recertified candidates |
|---|---:|---:|---:|
| `32k, lambda=1, R on` | `510` | `1,919` | `5.2%` |
| `32k, lambda=0, R on` | `818` | `2,363` | `6.4%` |

Likely reason: many replaced docs were never selected, so they failed the `R` admission rule. Also,
useful stale docs are refreshed rather than evicted, so the registry only sees a small tail of
previously useful evictions.

## Mentor Talking Points

- The original motivation for rerunning `lambda=0` and `lambda=1` was that old logs showed small
  `sigma` values and nearly identical eval scores, but they did not show whether uncertainty changed
  the selected negatives.
- The new logs added `flip_rate_vs_lambda0`, selected `sigma`, selected `lambda*sigma`, and
  selected `sigma/s_hat`.
- The important update is that `lambda=1` changed top-1 negative selection for about 41% of queries,
  so the uncertainty term was active.
- Since quality stayed close to `lambda=0`, the next question is not whether EMA uncertainty affects
  mining. It does. The next question is whether EMA is the right uncertainty signal.
- `ema_alpha=0.999` is slow per step, but not frozen over 10k steps. A useful next probe is
  `ema_alpha=1.0` to test a frozen-teacher uncertainty signal.
- `R` did not matter in the latest runs because it never contributed candidates. If `R` is tested
  again, log `R_inserted` / `R_reentry_rate` and consider increasing the audition window `K`.

