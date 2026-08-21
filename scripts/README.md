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
