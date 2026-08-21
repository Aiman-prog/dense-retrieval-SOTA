# Uncertainty-weighted hard negative mining: pilot experiment

## Question

Hard negative mining selects training negatives by relevance score. We add an
uncertainty bonus:

```
g = s_hat + lambda * sigma
```

`s_hat` is the predicted query-document score. `sigma` is the model's uncertainty on that
score, estimated from stochastic forward passes. `lambda` sets how much uncertainty
influences selection.

The question is whether selecting uncertain negatives trains a better retriever than
selecting on score alone.

## Why this needs a new experiment

An earlier study on the sequential mining pipeline found that the uncertainty term is not
inert. At lambda = 1 it changed the top-ranked negative for about 41% of queries. Yet
retrieval scores stayed close to the lambda = 0 control. So the term changes what is
mined without visibly changing what is learned.

That study could not settle the question, because the asynchronous mining setup carried a
scheduling defect. The document cache was configured to refresh only after a number of
steps equal to the entire training run. Refresh was therefore never triggered in
practice. Every negative was scored against embeddings from the initial model. The
configuration intended for the experiment had never actually run.

This has been corrected and verified. In the control run, 30 to 40% of the cache is now
re-encoded every mining round.

## Design

Three training runs. Identical in every respect except lambda.

| arm | lambda | role |
|---|---|---|
| control | 0 | selection by score alone |
| low dose | 0.3 | uncertainty as a mild tie-breaker |
| medium dose | 0.5 | uncertainty as a substantial reweighting |

Mining schedule, cache size, number of stochastic passes and all optimisation settings
are held fixed across arms. The cache persists across mining rounds, so any change to the
mining schedule would permanently fork a run's trajectory. Holding it constant is what
makes the arms comparable.

## Choosing the doses

Lambda is not comparable across models, because it multiplies an uncertainty whose scale
is model-dependent. Picking round numbers would not be meaningful.

Instead we calibrate by effect. A cheap report-only probe sweeps lambda over
{0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0} on 2,048 stratified queries with 3 independent
stochastic draws. For each value it measures the **flip rate**: the fraction of queries
whose top-ranked negative changes relative to the lambda = 0 control.

| lambda | flip rate |
|---:|---:|
| 0.1 | 6.4% |
| 0.2 | 11.3% |
| **0.3** | **16.7%** |
| **0.5** | **26.6%** |
| 0.7 | 35.7% |
| 1.0 | 56.3% |

We then select one value in each of two pre-specified bands. The low band is 10 to 20%,
the medium band is 20 to 35%. This yields 0.3 and 0.5. Seed-to-seed variation was under
0.6 percentage points, so the selection is stable.

Two properties of the probe are worth stating. No known positive was ever selected as a
negative, at any lambda. And the score gap between the top two candidates is frequently
near zero, which means uncertainty mostly breaks ties among documents the score cannot
separate, rather than overriding a confident preference.

## Scale

The pilot uses a fixed, reproducible 10% sample of the training mixture, stratified across
its three sources and interleaved so that mining order is representative. Each arm trains
1,032 steps and takes under two hours. A full run is roughly ten times that.

Running the pilot for all three arms costs about six hours. Running the full experiment
for three arms would cost about two days. The pilot exists to decide whether that is
worth spending.

## Evaluation

Each arm is evaluated on four BRIGHT domains chosen for topical diversity: biology,
economics, stackoverflow, and theoremqa_questions. The metric is NDCG@10, averaged across
domains.

A candidate is promoted only if all three conditions hold:

1. average NDCG@10 improves by at least 0.005 over the control
2. at least 3 of the 4 domains improve
3. the gain is not produced by a single domain

If both doses promote and differ by less than 0.002, the smaller lambda is preferred.

## What this can and cannot conclude

This is a screening gate, not a hypothesis test. The four domains contribute roughly 520
queries in total, giving a standard error near 0.01 to 0.015 on the averaged metric. The
0.005 promotion threshold therefore sits inside the noise, and a 3-of-4 domain result has
a probability near 0.31 under the null hypothesis.

So a promotion does not demonstrate that uncertainty helps. It means the signal is large
enough to justify a matched full-scale confirmation, which is the run that could
demonstrate it.

A negative result is more informative than a positive one at this scale. If neither dose
clears the gate, the uncertainty term does not earn its cost even with document refresh
working correctly. That is a publishable finding about the method, not a failed
experiment.

## Results

All three arms trained to completion and passed the run-validity gate. Document refresh was
active in every arm. All three were then evaluated on the four development domains.

NDCG@10 by domain:

| domain | lambda 0 | lambda 0.3 | lambda 0.5 |
|---|---:|---:|---:|
| biology | **0.5357** | 0.5329 | 0.5276 |
| economics | **0.2329** | **0.2329** | 0.2295 |
| stackoverflow | **0.1905** | 0.1856 | 0.1835 |
| theoremqa_questions | **0.1313** | 0.1278 | 0.1321 |
| macro | **0.2726** | 0.2698 | 0.2682 |

Recall@1000 by domain:

| domain | lambda 0 | lambda 0.3 | lambda 0.5 |
|---|---:|---:|---:|
| biology | **0.9472** | 0.9448 | 0.9432 |
| economics | 0.7671 | **0.7675** | 0.7571 |
| stackoverflow | **0.6790** | 0.6783 | 0.6745 |
| theoremqa_questions | **0.4422** | 0.4329 | 0.4396 |
| macro | **0.7089** | 0.7059 | 0.7036 |

Macro MRR follows the same ordering: 0.3383, 0.3317, 0.3319.

Neither dose is promoted. Both fail all three conditions, and fail them in the wrong
direction. The change in average NDCG@10 is -0.0028 at lambda 0.3 and -0.0044 at lambda 0.5,
against a requirement of +0.005. Zero of four domains improved, against a requirement of
three. The single case where a treatment arm leads on recall is economics at lambda 0.3, by
0.0004, which is one relevant document for one query out of 103.

## Interpretation

Individually, every one of these differences sits inside the noise floor. The standard error
on the averaged metric is near 0.010 to 0.015, and the observed gaps are 0.003 to 0.004. So
the data do not show that uncertainty weighting is harmful.

What the data do show is the absence of the signal the pilot was built to detect. The
screening threshold was deliberately set below the noise floor so that a real effect would
survive it. Nothing survived it. Average NDCG@10 and average Recall@1000 both decline
monotonically as lambda increases, and across four domains and three metrics no comparison
favours uncertainty by more than rounding.

This closes the alternative explanation that motivated the pilot. The earlier null result on
the sequential pipeline could be dismissed on the grounds that document refresh was never
firing, so nothing downstream of selection could respond. Refresh is now confirmed active,
the probe confirms the uncertainty term is changing selections at these doses, and the miner
diagnostics show a systematic effect that widens over training. Selection genuinely changes.
Retrieval quality does not.

Two independent uncertainty estimators have now been tested, on two different mining
architectures, and both give the same answer. The sequential study used a teacher-student
disagreement estimator at lambda 1. This pilot used dropout-sampled variance at lambda 0.3
and 0.5. In both cases the term reshapes the negative distribution without improving the
retriever.

The defensible claim is therefore specific: adding an uncertainty bonus linearly to the
relevance score does not improve retrieval on this benchmark, for either estimator, on this
base model. Three things remain untested and should be stated as such. The dropout variance
was estimated from only three samples, so it is itself a noisy quantity. Only the additive
combination rule was tried, and the probe showed that the score gap between the top two
candidates is usually near zero, meaning uncertainty is spent almost entirely on breaking
ties among documents relevance cannot separate. And each estimator was tested on only one of
the two pipelines.

The most promising direction is the second of those. A term that only ever acts as a
tie-breaker among indistinguishable candidates is close to a term that cannot help by
construction. That is a more informative finding than the dose sweep itself, and it points at
the combination rule rather than the dosage as the thing to change.

## Status

- dose selection: complete, both bands satisfied
- all three arms: trained, gate-valid, evaluated on four domains
- decision: no promotion. A full-scale confirmation run of these arms is not justified
