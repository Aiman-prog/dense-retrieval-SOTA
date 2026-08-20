# DelftBlue GPU checklist — post-consolidation

Seven experiments, all on `main`. **You run these; I cannot.** Nothing below verifies training
*correctness* or retrieval quality — only that the job starts, logs the config it was asked
for, and writes a checkpoint with finite loss.

There is **no `--max_steps` flag on any entry point**, and adding one would violate the
consolidation rules. Where a debug mode exists, use it; where it does not, assert on the
**first checkpoint written with finite loss** rather than a step count.

Run everything from `/home/$USER/dense-retrieval-SOTA`. `sbatch` scripts set `DATA_BASE_DIR`
themselves — **do not rely on it in a login shell**, where it is unset and
`$DATA_BASE_DIR/...` silently collapses to `/...`. Use absolute paths interactively.

---

## Step 0 for every job — the startup block

Consolidation added one uniform block to all six entry points. Check it **first**, before
waiting on anything else:

```bash
grep -A 10 'RESOLVED TRAINING CONFIG' slurm-<jobid>.out
```

Confirm `base_model`, `temperature`, `query_max_len` (**1024**), `passage_max_len` (**512**),
batch size, learning rate, epochs and the recipe name are what you intended.

⚠️ **The `base_model` line is the one that matters.** `get_training_context()` resolves the
model against the HF snapshot cache and **silently falls back to the raw configured string**
when no snapshot directory holds a `config.json`. Four recipes — `ance`, `grass`,
`fast_grass`, `async_fast_grass` — train from
`/scratch/aimanabdulwaha/dense-retrieval-SOTA/models/inbatch_mixed_bge_m3`. If that path is
missing, they will **train cleanly against the wrong weights**. The block prints
`[PATH DOES NOT EXIST]` when that happens:

```bash
grep 'PATH DOES NOT EXIST' slurm-<jobid>.out && echo "STOP — wrong base model"
```

Empty output is what you want.

---

## 1. In-batch — `train_inbatch.py`

```bash
sbatch scripts/run_inbatch_singularity.sh
```

| | |
|---|---|
| allocation | `gpu-a100`, 1 GPU, **`--time=14:00:00`** |
| smoke flag | none — assert on first checkpoint |
| checkpoint cadence | `total_steps // 5`; at 330k / bs 64 / 2 epochs that is step **2062** of 10314 |

⚠️ The launcher's own comment says 14 h is a **temporary OOM-smoke value** and should be
restored to `24:00:00` for a real run. Check before submitting.

**Success signal**
```bash
grep -E 'Total steps:|checkpoint-2062' slurm-<jobid>.out
ls models/inbatch_mixed_bge_m3/checkpoint-*/
grep -oE "'loss': [0-9.]+" slurm-<jobid>.out | head    # finite, not nan/inf
```

---

## 2. Cross-batch — `train_crossbatch.py`

```bash
sbatch scripts/run_crossbatch_singularity.sh
```

| | |
|---|---|
| allocation | `gpu-a100`, **2 GPUs** (`torchrun --nproc_per_node=2`), `--time=24:00:00` |
| smoke flag | none — and this entry point has **no CLI surface at all** |
| checkpoint cadence | hard-coded `--save_steps 100` → first checkpoint early |

The startup block reports **`per_device_batch_size`**, not `batch_size` — this recipe has no
`batch_size` key. Expect `512` (× 2 GPUs = 1024 pool).

**Success signal**
```bash
grep -A 10 'RESOLVED TRAINING CONFIG' slurm-<jobid>.out
ls models/crossbatch_mixed_bge_m3_epoch2/checkpoint-100/
```

---

## 3. ANCE (BRIGHT) — `train_ance.py`

```bash
sbatch scripts/run_ance_singularity.sh
```

| | |
|---|---|
| allocation | `gpu-a100`, **2 GPUs** (Trainer GPU 0 / Inferencer GPU 1), `--time=10:00:00` |
| smoke flag | none |
| checkpoint cadence | `save_steps: 500` |

The startup block reports **`total_epochs`**, not `num_epochs` — the ANCE recipes have no
`num_epochs` key.

**Success signal**
```bash
grep 'GPU(s) detected' slurm-<jobid>.out          # must say 2 GPU(s)
grep 'Initial data written to' slurm-<jobid>.out  # initial mine completed
ls models/ance_mixed_bge_m3/checkpoint-500/
```

Prior run for reference: job 9566838 reached NDCG@10 = 0.1683 with 1 ANN refresh.

---

## 4. Sync GRASS — `run_grass.py`

```bash
GRASS_UNCERTAINTY=mc_dropout sbatch scripts/run_grass_singularity.sh   # or ema
```

| | |
|---|---|
| allocation | `gpu-a100`, 1 GPU, `--time=20:00:00` |
| smoke flag | **`--debug`** (512-item mixture) — use it first |
| checkpoint cadence | `save_steps: 1000` |
| env knobs | `GRASS_UNCERTAINTY`, `GRASS_MODEL_SUFFIX`, `GRASS_NUM_EPOCHS`, `GRASS_P`, `GRASS_L`, `GRASS_LAMBDA` |

First run builds the stale ANN index over the full corpus — budget extra time before step 1.

**Success signal**
```bash
grep 'Stale index ready' slurm-<jobid>.out
grep 'avg_loss=' slurm-<jobid>.out                # finite
ls models/grass_mixed_bge_m3_mc_dropout/checkpoint-1000/
```

---

## 5. Sequential Fast-GRASS — `run_fast_grass.py`

```bash
FAST_GRASS_UNCERTAINTY=mcdp sbatch scripts/run_fast_grass_singularity.sh
```

| | |
|---|---|
| allocation | `gpu-a100`, 1 GPU, `--time=20:00:00` |
| smoke flag | **`--debug`** |
| checkpoint cadence | `save_steps: 1000` |
| env knobs | `FAST_GRASS_{UNCERTAINTY,LAMBDA,B_DOC,L,T,M,MC_DROPOUT_P,SELECTION_MODE,EMA_ALPHA,NUM_EPOCHS,MODEL_SUFFIX,NO_EVAL,NO_REGISTRY}` |

**Success signal**
```bash
grep 'Stale index ready' slurm-<jobid>.out
grep 'Checkpoint saved:' slurm-<jobid>.out
grep 'avg_loss=' slurm-<jobid>.out                # finite
```

Note: this entry point prints its own older config block **as well as** the new
`RESOLVED TRAINING CONFIG` block. Both are correct; the older one was left untouched
deliberately. Some values appear twice.

---

## 6. Async Fast-GRASS — `train_async_fast_grass.py`

```bash
python scripts/train_async_fast_grass.py --preflight       # login node, no GPU (heavy: loads 655k docs)
ASYNC_FG_DEBUG=1 ASYNC_FG_MAX_ROUNDS=1 sbatch scripts/run_async_fast_grass_singularity.sh
sbatch scripts/run_async_fast_grass_singularity.sh          # real run
```

| | |
|---|---|
| allocation | `gpu-a100`, **2 GPUs required**, `--time=20:00:00` |
| smoke flags | `ASYNC_FG_DEBUG=1`, `ASYNC_FG_MAX_ROUNDS=1`, `--preflight` |
| env knobs | `ASYNC_FG_{RECIPE,LAMBDA,MANIFEST,SUFFIX,MAX_ROUNDS,DEBUG,FRESH,NO_EVAL,NO_COMPILE,BOOTSTRAP_CKPT,RUN_TESTS}` |

⚠️ `--fresh` is **required** if the handoff root holds a previous run: Phase 1 has no trainer
resume, so a step-0 trainer would consume rounds mined from an older checkpoint.

⚠️ The job always runs the CPU test gate first (~2 min). `ASYNC_FG_RUN_TESTS` **cannot be
disabled** — the launcher tests emptiness, so even `0` is truthy.

**Success signal**
```bash
grep -A 10 'RESOLVED TRAINING CONFIG' slurm-<jobid>.out
grep 'refresh schedule' -A 6 slurm-<jobid>.out    # must not report errors
grep -E 'async_gap_steps|data_age_steps|rounds_consumed|rounds_skipped' slurm-<jobid>.out
```

Rising `async_gap_steps` with `miner_idle_time ≈ 0` ⇒ the miner is the bottleneck (raise
`async_mine_every_steps`, or lower `B_doc` / `T`). Large `rounds_skipped` ⇒ trainer is
over-checkpointing.

---

## 7. ANCE (MS MARCO) — `train_ance.py --recipe ance_msmarco`

### 🛑 This will NOT run yet — pre-existing defect P1

```bash
sbatch scripts/run_ance_msmarco_singularity.sh    # crashes seconds in, before any GPU work
```

`train_ance.py:159` calls `get_path("temp_ance_msmarco")`. That key is **not** in
`helpers.get_path`'s `path_map`, and `get_path` ends in `path_map.get(key)` — so it returns
`None`, and line 161 raises:

```
TypeError: unsupported operand type(s) for /: 'NoneType' and 'str'
```

**Pre-existing on `baseline` too** — consolidation only made it reachable by bringing the
recipe across. It fails fast and cheap, not hours in. Full write-up: **P1** in
`CONSOLIDATION_STATUS.md`.

**Fix before submitting** (one line, needs your authorisation — deliberately not applied):
```python
# src/utils/helpers.py, in get_path's path_map
"temp_ance_msmarco": base / "temp_ance_msmarco_workdir",
```
Worth also making `get_path` raise on an unknown key instead of returning `None` — that silent
`None` is what let this hide.

Once fixed:

| | |
|---|---|
| allocation | `gpu-a100`, **2 GPUs**, `--time=24:00:00` (~16–18 h estimated in the launcher) |
| checkpoint cadence | `save_steps: 1250` |
| eval | separate job: `sbatch scripts/eval_msmarco_singularity.sh` (1 GPU, 4 h), metric `recip_rank` (MRR) |

**Success signal**
```bash
grep -A 10 'RESOLVED TRAINING CONFIG' slurm-<jobid>.out   # recipe: ance_msmarco
grep 'GPU(s) detected' slurm-<jobid>.out
ls models/ance_msmarco_bge_m3/checkpoint-1250/
```

---

## What is unchanged from pre-consolidation

Verified by `AC-SURFACE-01` (`SURFACE_ALLOWLIST_OK`) against
`archive/main-post-promotion`:

- **no pre-existing `scripts/*.sh` launcher changed by a single byte** — job scripts, SLURM
  headers, `--bind` mounts and env assumptions are exactly as they were;
- the only new launchers are `eval_msmarco_singularity.sh` and
  `run_ance_msmarco_singularity.sh`;
- `config/config.yaml` differs only by the added `training.ance_msmarco` block; no
  pre-existing key changed;
- every entry point's CLI flags, recipe names, config path keys, environment keys and
  `sys.path` handling are byte-for-byte unchanged — the Step-6 logging added output only.

So any behaviour difference you observe on the cluster is **not** consolidation drift in the
job plumbing.

---

## Evaluation

```bash
sbatch scripts/run_evaluate_singularity.sh              # defaults to the 4 pilot domains
EVAL_DOMAINS=all sbatch scripts/run_evaluate_singularity.sh
```

Use **`gpu-a100`, not `gpu-a100-small`** — the small partition ran 2.11 s/it and could not fit
four domains inside its 4 h cap. Per-domain `{domain}_results.json` is written as each domain
finishes, so a timeout is resumable by passing only the gaps.
