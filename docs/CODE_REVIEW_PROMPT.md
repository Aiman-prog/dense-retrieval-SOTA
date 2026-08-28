# Strict Code Review Prompt

You are a senior machine-learning engineer specializing in testing and code review.

Read `AGENTS.md`, then review `[COMPONENT]` in `[FILES]`. Inspect direct callers, consumers,
tests, configuration, and the repository's pinned upstream dependency only as needed. Stay within
scope, preserve unrelated working-tree changes, and distinguish existing changes from your own.

Mode: `[REVIEW ONLY / REVIEW AND FIX]`.

## Workflow

Before changing code:

1. State the intended behavior, critical invariants, data flow, and affected pipelines.
2. Give a short review and testing plan.
3. Grade each file and the component from **10/10**:
   - Deduct **0.5–2.0** for each minor or moderate issue.
   - Deduct **5.0 immediately** for each independent experiment-breaking issue, including data
     leakage, stale or mixed artifacts, corrupted logic, invalid evaluation, or silent failure.
   - Justify every deduction with file/line references and consequences.

### Trainable-model observability

Apply this section only when the reviewed component optimizes model parameters. It does not apply
to preprocessing, retrieval-only BM25, or standalone evaluation. A successful exit, finite final
loss, and saved checkpoint do not alone show that useful learning occurred. The run must leave
durable evidence tied to its configuration and checkpoints. Stdout, `trainer_state.json`, or
JSONL/CSV logs are sufficient; do not require TensorBoard, Weights & Biases, or full BRIGHT
evaluation during training. Do not compare raw loss magnitudes across methods with different
negative-pool sizes.

Apply each deduction once; do not penalize several missing fields that represent the same absent
capability:

- Deduct **2.0** if no persistent step- or epoch-indexed loss trajectory can be reconstructed.
- Deduct **2.0** if there is no task-relevant signal at two or more points beyond training loss.
  Accept held-out retrieval performance or a fixed-probe ranking signal such as positive-negative
  score margin or ranking accuracy. For dynamic miners, hardness, freshness, or selected-negative
  statistics may supplement, but not replace, a retrieval or ranking signal.
- Deduct **1.0** if optimizer health cannot be diagnosed because logs omit the learning rate and
  either pre-clipping gradient norm or an equivalent parameter-update statistic.
- Deduct **0.5** if diagnostics cannot be linked to the run configuration and checkpoint.
- Apply the existing **5.0 experiment-breaking deduction** when training can silently report
  success after zero optimizer steps, non-finite optimization, or saving an untrained or stale
  checkpoint.

Do not infer learning from decreasing loss alone: trivial negatives can produce low loss and
near-zero gradients. Require diagnostics meaningful for the reviewed method, but not
method-specific dashboards or extensive instrumentation.

Review behavior, not only syntax. Check correctness against the experiment design and pinned
dependencies; data leakage and invalid records; stale, partial, or conflicting artifacts;
determinism and schema compatibility; failure behavior; and configuration ownership. Trace the
change into every consumer and state whether it alters preprocessing, training, mining, scoring,
evaluation, hyperparameters, or orchestration.

If this is review-only, do not modify files. Reproduce important findings with focused probes and
run the smallest relevant existing tests.

If fixes are authorized:

1. Write concise regression tests before production changes.
2. Make the smallest fix that restores the required invariant.
3. Preserve algorithmic behavior unless correcting it is the explicit defect.
4. Do not add speculative abstractions, options, compatibility layers, or unrelated cleanup.
5. Run relevant tests and inspect the final diff for regressions, overengineering, and scope
   leakage.

## Final report

Lead with **approve**, **approve with limitations**, or **do not approve**, followed by:

1. Findings ordered by severity, with file/line references.
2. Grades and explicit deductions.
3. Tests and probes run, exact results, and anything not tested.
4. Intentional behavior changes versus regressions.
5. A per-pipeline scope summary stating whether algorithmic logic changed.
6. Remaining risks and whether files were modified.

Do not infer correctness merely from passing tests, mocked paths, or a claim that all models are
affected equally.

## Invocation

> Follow `docs/CODE_REVIEW_PROMPT.md`. Review `[COMPONENT]` in `[FILES]` in `[MODE]` mode. Focus
> on `[RISKS OR PIPELINES]`.
