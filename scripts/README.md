# `scripts/` — the experiment ladder

Every rung of the thesis comparison has one entry point and one launcher. Submit the launcher;
it sets `DATA_BASE_DIR`, binds `/scratch`, and runs the entry point inside the container.

| # | pipeline | entry point | launcher (`sbatch scripts/launchers/…`) |
|---|---|---|---|
| 0 | BM25 sparse baseline (CPU) | `run_bm25_evals.py` | `run_bm25_singularity.sh` |
| 1 | in-batch negatives | `train_inbatch.py` | `run_inbatch_singularity.sh` |
| 2 | cross-batch (GradCache) | `train_crossbatch.py` | `run_crossbatch_singularity.sh` |
| 3 | ANCE (periodic full-corpus refresh) | `train_ance.py` | `run_ance_singularity.sh` |
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

The paper's own recipe is reachable as `train_ance.py --recipe ance_paper` — the same entry
point, miner and round handoff, with RoBERTa + a projection head, pairwise NLL over raw dot
and LAMB swapped in (`scripts/ance_paper.py`). It is **not a rung of the ladder**: it is a
separate MS MARCO experiment whose job is to show this implementation is faithful, so that
the BRIGHT row can keep GRASS's objective and stay a comparison of mining. It runs two
epoch-equivalent budgets over 20 expanded triplets/query (~250K steps), not Microsoft's
600K, because DelftBlue supplies one miner GPU and a 24-hour allocation; this limitation
must accompany its result.

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
