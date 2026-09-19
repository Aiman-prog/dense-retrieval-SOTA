# `scripts/` — the experiment ladder

Every rung of the thesis comparison has one entry point and one launcher. Submit the launcher;
it sets `DATA_BASE_DIR`, binds `/scratch`, and runs the entry point inside the container.

| # | pipeline | entry point | launcher (`sbatch scripts/launchers/…`) |
|---|---|---|---|
| 0 | BM25 sparse baseline (CPU) | `run_bm25_evals.py` | `run_bm25_singularity.sh` |
| 1 | in-batch negatives | `train_inbatch.py` | `run_inbatch_singularity.sh` |
| 2 | cross-batch (GradCache) | `train_crossbatch.py` | `run_crossbatch_singularity.sh` |
| 3 | ANCE (periodic full-corpus refresh) | `train_ance.py` | `run_ance_singularity.sh` |
| 3b | ANCE round-0 preparation (1 GPU, MS MARCO arm) | `train_ance.py --prepare-initial` | `run_ance_paper_prepare_singularity.sh` |
| 4 | naive GRASS | `run_grass.py` | `run_grass_singularity.sh` |
| 5 | sequential Fast-GRASS | `run_fast_grass.py` | `run_fast_grass_singularity.sh` |
| 6 | async Fast-GRASS (2 GPUs) | `train_async_fast_grass.py` | `run_async_fast_grass_singularity.sh` |
| — | **evaluation, any model** | `run_all_evals.py` | `run_evaluate_singularity.sh` |

Rungs 4→5→6 are three generations of the same miner, each built to cut the previous one's cost.
They are not alternatives to pick between; each imports from the one before it.

The ANCE row is **ANCE-style asynchronous ANN mining under the same BGE-M3/Tevatron
contrastive objective as naive GRASS**. At batch 64/group 2, each query sees one explicit
ANN-mined negative plus 126 cross-example passages. That shared 127-negative objective makes
the BRIGHT comparison about mining, but it is not Microsoft's pairwise RoBERTa/LAMB
reproduction. Optimizer parity is currently guaranteed only for ANCE versus **naive GRASS**;
Fast-GRASS and async Fast-GRASS retain their own optimizer paths.

The paper's own recipe is reachable as `train_ance.py --recipe ance_paper`, via its own
`run_ance_paper_singularity.sh` — the same entry
point, miner and round handoff, with RoBERTa + a projection head, pairwise NLL over raw dot
and LAMB swapped in (`scripts/ance_paper.py`). It is **not a rung of the ladder**: it is a
separate MS MARCO experiment whose job is to show this implementation is faithful, so that
the BRIGHT row can keep GRASS's objective and stay a comparison of mining. It stops at a
fixed 300K optimizer steps while retaining Microsoft's one-million-step scheduler horizon,
not at the released 600K checkpoint, because DelftBlue supplies one miner GPU and a 24-hour
allocation; this limitation must accompany its result.

It also has a prerequisite of its own. `ance_paper` initializes from a 60K BM25 warm-up, and
Microsoft's release is no longer downloadable — both blob URLs return HTTP 409, the issues
have been open since 2022, and the one mirror is bit-identical to the released 600K *final*,
which `assert_permitted_init` refuses precisely so a finished model cannot "reproduce" 0.330
by construction. `train_ance_warmup.py` (`run_ance_warmup_singularity.sh`, recipe
`ance_paper_warmup`) builds the warm-up from `roberta-base` on the BM25 negatives already in
the mixture: one GPU, ~2-3 h, no mining and no rounds. Gate it at MRR@10 ≈ 0.311 (with
`EVAL_ALLOW_DRIFT=1` — it trains at q128/p128 and is consumed at q64/p512), record the hash it
prints as `ance_paper.expected_init_sha256`, then run the reproduction. The substituted
initialization is a second recorded deviation alongside the step budget.

**Prepare round 0 before the training job.** `run_ance_paper_prepare_singularity.sh` mines it
on one GPU (~6.6h) and writes a reusable artifact; the training job adopts it with
`ANCE_INITIAL_ROUND=<dir>` and refuses it unless every input hash matches. Mining it inline
costs the training allocation 6h38m, which is why job 204931 could not fit 300K steps.
`ANCE_OVERWRITE=1` is required when the output directory still holds checkpoints — ANCE
cannot resume, so starting fresh deletes them, and that is now an explicit choice.

`scripts/launchers/run_ance_refresh_repro_singularity.sh` reproduces the refresh-encoder crash
(`P-ANCE-05`) in 2m23s and is kept as that defect's regression probe.

Run `run_ance_warmup_preflight_singularity.sh` before the GPU job. It executes the same code
on the same data with no GPU and no writes, on `compute-p1` in minutes, and the GPU launcher
repeats it as stage 1. The warm-up reads the mixture with `ANCEDataset(ragged=True)`, because
upstream has no per-query negative count and ~1.8% of `Tevatron/msmarco-passage` records carry
fewer than 30 — the strict path stays in force for mined rounds.

## Comparing rows 0, 1 and 2 honestly

**In-batch vs cross-batch is a comparison of two complete recipes, not a controlled
test of negative-pool size.** At the configured settings the two differ in
optimizer-step budget as much as in pool size:

| | queries/step | passages/step | negatives per query | optimizer steps (same data, 2 epochs) |
|---|---|---|---|---|
| in-batch | 64 | 128 | 127 (17 in each epoch's 9-query final batch) | **16x more** |
| cross-batch | 1,024 (512 x 2 ranks) | 2,048 | 2,047 (constant; the final step is padded) | 1x |

One optimizer step consumes 1,024 queries instead of 64, so cross-batch takes 1/16 as
many steps over the same mixture. Any difference in NDCG is attributable to the pool
**and** to that budget. Neither arm isolates the other, so no causal claim about pool
size can be drawn from the pair. Each run records both numbers as
`negative_pool_size` and `optimizer_steps_planned` in its `run_manifest.json`, so the
comparison can always be restated from the artifacts rather than from memory.

Cross-batch is **distributed large-batch training**: the pool is one step's 2,048
passages gathered across 2 ranks by `DistributedContrastiveLoss`. Negatives are not
carried across optimizer steps, and `gradient_accumulation_steps` does not enlarge the
pool — GradCache pools inside a single `training_step`. Launched without `torchrun`,
`is_ddp` is false, the all-gather disappears and the pool silently halves;
`check_batch_invariants` refuses that rather than training on it.

**BM25 vs dense requires the same domains.** `run_bm25_evals.py` always evaluates all
twelve `evaluation.eval_domains`, while `run_evaluate_singularity.sh` defaults to the
four lambda-pilot domains. A default dense run is therefore **not** comparable to the
BM25 baseline. Use `EVAL_DOMAINS=all`, and pass `--compare_bm25 <bm25 summary.json>` to
`run_all_evals.py`, which refuses to write a summary when the two domain sets differ and
when their corpus/query/qrel/exclusion hashes differ. Hashless legacy BM25 summaries are
also refused because their evaluation inputs cannot be verified.

Incidental false negatives (a passage that is a positive for two different queries, both
landing in the same batch) affect every dense row and grow with the pool. Measured, not
assumed: `python scripts/dev/check_neg_contamination.py` reports it alongside explicit
negative-in-qrels contamination. The one explicit hard negative per query
(`train_group_size: 2`) is unchanged by any of this.

## One evaluator

There is a single BRIGHT evaluator. `EVAL_MODEL_PATH` chooses the model — it is **required**:

```bash
MODELS=/scratch/$USER/dense-retrieval-SOTA/models
EVAL_MODEL_PATH=$MODELS/<name> sbatch scripts/launchers/run_evaluate_singularity.sh
```

`EVAL_DOMAINS` defaults to the four pilot domains; `EVAL_DOMAINS=all` runs all twelve, or pass
a comma-separated subset. Sweep checkpoints by looping the same launcher over `checkpoint-*`.

Underneath, `run_all_evals.py` runs one `src/evaluation/evaluate.py` subprocess per domain and
**exits nonzero if any domain fails**, so a partial run cannot be mistaken for a complete one.

MS MARCO is a separate dataset with its own evaluator, `eval_msmarco.py` (blocked — see defect
P6 in `CONSOLIDATION_STATUS.md`).

## Everything else in this directory

**Workers** — spawned by an orchestrator, never run directly:
`run_ance_train.py`, `run_ance_data_gen.py`, `run_async_fast_grass_miner.py`,
`run_async_fast_grass_train.py`.

**Shared internals** — imported by the entry points above:
`async_fast_grass_cached_mcdp.py`, `async_fast_grass_handoff.py`, `async_fast_grass_pilot.py`,
`patch_tevatron.py`, `prepare_models.py`, `refresh_stale_index.py`.

> `refresh_stale_index.py` is a **prerequisite**, not a utility: async Fast-GRASS raises
> `FileNotFoundError` without the stale-index pickle. Run rung 5, or its own launcher, first.

### Why the Python is flat

19 consumers — the test suites, the `dev/` tools, and these scripts themselves — put `scripts/`
on `sys.path` and import by bare module name (`import run_fast_grass`). Subdividing the Python
would break every one of those. Shell files are never imported and each pins
`#SBATCH --chdir=<repo root>`, so they live in `launchers/` at no cost.

## Subdirectories

- **`launchers/`** — every SLURM job script. Repo-relative paths inside; safe to move.
- **`dev/`** — diagnostics, feasibility probes, Phase-0 timing tools and post-hoc analysis.
  Nothing here trains a model or produces a reported result.
