# DelftBlue GPU checklist — post-consolidation

Seven experiments, all on `main`. **You run these; I cannot.** Nothing below verifies training
*correctness* or retrieval quality — only that the job starts, logs the config it was asked
for, and writes a checkpoint with finite loss.

There is **no `--max_steps` flag on any entry point**, and adding one would violate the
consolidation rules. Where a smoke mode is reachable, use it; where it is not, assert on the
**first checkpoint written with finite loss** rather than a step count.

---

## Before your first job — three things that will otherwise bite

### 1. `logs/` — now tracked, but confirm it survived your checkout

```bash
ls -d /home/$USER/dense-retrieval-SOTA/logs || mkdir -p /home/$USER/dense-retrieval-SOTA/logs
```

Every launcher writes `--output=logs/<name>_%j.out`, and **SLURM opens that file before the
script body runs** — so an absent directory fails the job with no log to explain why, and the
in-script `mkdir -p logs` some launchers carry is far too late to help. Defect **P2 is fixed**:
`logs/.gitkeep` is tracked and `.gitignore` is now `logs/*` + `!logs/.gitkeep`, so a fresh
clone has the directory. The check above costs nothing if you pulled onto an older checkout.

### 2. Know where the logs and models actually are

Job logs are **not** `slurm-<jobid>.out`. Each launcher names its own, and **stderr is a
separate file** — tracebacks land in `.err`, not `.out`:

| experiment | stdout | stderr |
|---|---|---|
| 1 in-batch | `logs/inbatch_neg_<jobid>.out` | `logs/inbatch_neg_<jobid>.err` |
| 2 cross-batch | `logs/crossbatch_bge_<jobid>.out` | `.err` |
| 3 ANCE BRIGHT | `logs/ance_<jobid>.out` | `.err` |
| 4 sync GRASS | `logs/grass_<jobid>.out` | `.err` |
| 5 seq Fast-GRASS | `logs/fast_grass_<jobid>.out` | `.err` |
| 6 async Fast-GRASS | `logs/async_fg_<jobid>.out` | `.err` |
| 7 ANCE MS MARCO | `logs/ance_msmarco_<jobid>.out` | `.err` |
| MS MARCO eval | `logs/eval_msmarco_<jobid>.out` | `.err` |
| BRIGHT eval | `logs/eval_<jobid>.out` | `.err` |
| stale-index refresh | `logs/refresh_stale_<jobid>.out` | `.err` |

Models are **not** in the repo. `get_path("models")` is `$DATA_BASE_DIR/models`, while the job
runs with `--chdir=/home/$USER/dense-retrieval-SOTA`. Set this in your login shell:

```bash
export MODELS=/scratch/$USER/dense-retrieval-SOTA/models
export PROC=/scratch/$USER/dense-retrieval-SOTA/data/processed
```

⚠️ **`DATA_BASE_DIR` is exported only inside the sbatch scripts.** In a login shell it is
unset, so `$DATA_BASE_DIR/...` silently collapses to `/...`. Use absolute paths interactively.

### 3. Respect the ordering — the seven are not independent

```
1 in-batch ──► produces models/inbatch_mixed_bge_m3, the base_model for 3,4,5,6
                      │
5 seq Fast-GRASS ─────┴──► builds temp_grass_workdir/stale_index/corpus.pkl
                      │         (or: sbatch scripts/run_refresh_stale_index_singularity.sh)
                      ▼
              6 async Fast-GRASS   ← HARD-FAILS without that pickle (defect B2)
7 MS MARCO ──► blocked on a separate data-prep step; see §7
```

`train_async_fast_grass.py:280` raises `FileNotFoundError: stale index not found at …` and
**never builds one itself**. Run experiment 5, or the refresh job (`gpu-a100`, 1 GPU, 4 h),
first.

---

## Step 0 for every training job — the startup block

Consolidation added one uniform block to the **six training entry points**. Check it first,
before waiting on anything else:

```bash
grep -A 10 'RESOLVED TRAINING CONFIG' logs/<name>_<jobid>.out
```

Confirm `base_model`, `temperature`, `query_max_len` (**1024**), `passage_max_len` (**512**),
batch size, learning rate, epochs and the recipe name are what you intended.

⚠️ **The `base_model` line is the one that matters.** `get_training_context()` resolves the
model against the HF snapshot cache and **silently falls back to the raw configured string**
when no snapshot directory holds a `config.json`. Four recipes — `ance`, `grass`,
`fast_grass`, `async_fast_grass` — plus `ance_msmarco` train from
`/scratch/$USER/dense-retrieval-SOTA/models/inbatch_mixed_bge_m3`. If that path is missing they
will **train cleanly against the wrong weights**. The block prints `[PATH DOES NOT EXIST]`:

```bash
grep 'PATH DOES NOT EXIST' logs/<name>_<jobid>.out && echo "STOP — wrong base model"
```

Empty output is what you want.

`eval_msmarco.py` and `run_all_evals.py` are **not** among the six and print no such block.

---

## 1. In-batch — `train_inbatch.py`

```bash
sbatch scripts/run_inbatch_singularity.sh
```

| | |
|---|---|
| allocation | `gpu-a100`, 1 GPU, `--time=24:00:00` |
| smoke flag | none — assert on first checkpoint |
| checkpoint cadence | `total_steps // 5`; at 330k / bs 64 / 2 epochs that is step **2062** of 10314 |

**Success signal**
```bash
grep -E 'Total steps:|RESOLVED TRAINING CONFIG' logs/inbatch_neg_<jobid>.out
ls $MODELS/inbatch_mixed_bge_m3/checkpoint-*/
grep -oE "'loss': [0-9.]+" logs/inbatch_neg_<jobid>.out | head    # finite, not nan/inf
```

This is the **prerequisite for experiments 3–7** — they all train from its output.

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

✅ **Defect P3 is fixed** — the launcher now propagates its exit code, so `sacct` is
trustworthy again. (It previously ended in `echo`, reporting `COMPLETED` on a dead `torchrun`.)

The startup block reports **`per_device_batch_size`**, not `batch_size` — this recipe has no
`batch_size` key. Expect `512` (× 2 GPUs = 1024 pool).

**Success signal**
```bash
grep -A 10 'RESOLVED TRAINING CONFIG' logs/crossbatch_bge_<jobid>.out
tail -40 logs/crossbatch_bge_<jobid>.err        # the traceback lives here, not in .out
ls $MODELS/crossbatch_mixed_bge_m3_epoch2/checkpoint-100/
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
grep 'GPU(s) detected' logs/ance_<jobid>.out          # must say 2 GPU(s)
grep 'Initial data written to' logs/ance_<jobid>.out  # initial mine completed
ls $MODELS/ance_mixed_bge_m3/checkpoint-500/
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
| checkpoint cadence | `save_steps: 1000` |
| env knobs | `GRASS_UNCERTAINTY`, `GRASS_MODEL_SUFFIX`, `GRASS_NUM_EPOCHS`, `GRASS_P`, `GRASS_L`, `GRASS_LAMBDA` |

✅ **Defect P5 is fixed** — `GRASS_DEBUG=1` now reaches `--debug` (512-item mixture):

```bash
GRASS_DEBUG=1 GRASS_UNCERTAINTY=mc_dropout sbatch --time=01:00:00 scripts/run_grass_singularity.sh
```

With the knob unset the command line is byte-identical to before. The interactive form still
works if you want a shell:

```bash
srun --partition=gpu-a100 --gpus-per-task=1 --cpus-per-task=16 --time=00:30:00 \
     --account=Education-EEMCS-MSc-DSAIT --pty bash
cd /home/$USER/dense-retrieval-SOTA
export DATA_BASE_DIR=/scratch/$USER/dense-retrieval-SOTA
export HF_HOME=$DATA_BASE_DIR/data/bright HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
singularity exec --nv --bind /scratch/$USER:/scratch/$USER --bind /home/$USER:/home/$USER \
    /scratch/$USER/containers/pytorch_2.1.sif \
    python -u scripts/run_grass.py --uncertainty mc_dropout --debug
```

First real run builds the stale ANN index over the full corpus — budget extra time before
step 1. That pickle is what experiment 6 later depends on.

**Success signal**
```bash
grep 'Stale index ready' logs/grass_<jobid>.out
grep 'avg_loss=' logs/grass_<jobid>.out                # finite
ls $MODELS/grass_mixed_bge_m3_mc_dropout/checkpoint-1000/
```

---

## 5. Sequential Fast-GRASS — `run_fast_grass.py`

```bash
FAST_GRASS_UNCERTAINTY=mcdp sbatch scripts/run_fast_grass_singularity.sh
```

| | |
|---|---|
| allocation | `gpu-a100`, 1 GPU, `--time=20:00:00` |
| checkpoint cadence | `save_steps: 1000` |
| env knobs | `FAST_GRASS_{UNCERTAINTY,LAMBDA,B_DOC,L,T,M,MC_DROPOUT_P,SELECTION_MODE,EMA_ALPHA,NUM_EPOCHS,MODEL_SUFFIX,NO_EVAL,NO_REGISTRY}` |

✅ Defect P5 fixed here too — `FAST_GRASS_DEBUG=1` reaches `--debug`:

```bash
FAST_GRASS_DEBUG=1 FAST_GRASS_UNCERTAINTY=mcdp sbatch --time=01:00:00 scripts/run_fast_grass_singularity.sh
```

**Success signal**
```bash
grep 'Stale index ready'  logs/fast_grass_<jobid>.out
grep 'Checkpoint saved:'  logs/fast_grass_<jobid>.out
grep 'avg_loss='          logs/fast_grass_<jobid>.out   # finite
```

Note: this entry point prints its own older config block **as well as** the new
`RESOLVED TRAINING CONFIG` block. Both are correct; the older one was left untouched
deliberately, so some values appear twice.

---

## 6. Async Fast-GRASS — `train_async_fast_grass.py`

**Prerequisite: the stale-index pickle must already exist** (experiment 5, or
`sbatch scripts/run_refresh_stale_index_singularity.sh`). Verify before submitting:

```bash
ls -lh /scratch/$USER/dense-retrieval-SOTA/temp_grass_workdir/stale_index/corpus.pkl
```

```bash
python scripts/train_async_fast_grass.py --preflight       # login node, no GPU
ASYNC_FG_DEBUG=1 ASYNC_FG_MAX_ROUNDS=1 ASYNC_FG_FRESH=1 \
    sbatch scripts/run_async_fast_grass_singularity.sh     # smoke
sbatch scripts/run_async_fast_grass_singularity.sh         # real run
```

| | |
|---|---|
| allocation | `gpu-a100`, **2 GPUs required**, `--time=20:00:00` |
| smoke | `ASYNC_FG_DEBUG=1`, `ASYNC_FG_MAX_ROUNDS=1`, plus `--preflight` |
| env knobs | `ASYNC_FG_{RECIPE,LAMBDA,MANIFEST,SUFFIX,MAX_ROUNDS,DEBUG,FRESH,NO_EVAL,NO_COMPILE,BOOTSTRAP_CKPT,RUN_TESTS}` |

⚠️ `--preflight` loads the whole 655k-document corpus **with text** — several GB, heavy for a
login node. It also runs inside every job as step 1b, so the standalone run is optional.

⚠️ **`ASYNC_FG_FRESH=1` is required** if the handoff root holds a previous run: Phase 1 has no
trainer resume, so a step-0 trainer would consume rounds mined from an older checkpoint.

⚠️ The job always runs the CPU test gate first (~2 min). `ASYNC_FG_RUN_TESTS` **cannot be
disabled** — the launcher tests emptiness, so even `0` is truthy.

⚠️ A `*_pilot` or `*_smoke` recipe **requires** `ASYNC_FG_MANIFEST` (absolute path); the
launcher refuses to submit without it.

**Success signal**
```bash
grep -A 10 'RESOLVED TRAINING CONFIG' logs/async_fg_<jobid>.out
grep -A 6 'refresh schedule' logs/async_fg_<jobid>.out   # must not report errors
grep -E 'async_gap_steps|data_age_steps|rounds_consumed|rounds_skipped' logs/async_fg_<jobid>.out
```

Rising `async_gap_steps` with `miner_idle_time ≈ 0` ⇒ the miner is the bottleneck (raise
`async_mine_every_steps`, or lower `B_doc` / `T`). Large `rounds_skipped` ⇒ trainer is
over-checkpointing.

**Exit code 1 with a PASS/FAIL gate block** means the run completed but is invalid evidence
about λ. Do not submit nonzero arms on a failed λ=0 run.

---

## 7. ANCE (MS MARCO) — `train_ance.py --recipe ance_msmarco`

### 🛑 Blocked on data prep. Do not submit yet. (defect P6)

Defect P1 (the `get_path("temp_ance_msmarco")` `TypeError`) **is fixed**. The remaining
blocker is data, and the job **cannot fetch it itself**:

- `run_ance_msmarco_singularity.sh` exports `HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1`;
- `prepare_msmarco_full_corpus` / `_tevatron_train` / `_dev` call `load_dataset()` on
  `Tevatron/msmarco-passage-corpus` and `Tevatron/msmarco-passage`, **two of them with
  `streaming=True`**, which cannot work offline under any circumstances;
- `msmarco_dev_qrels.txt` is not in either dataset at all — the `validation` split has no
  `positive_passages`.

**Do this on a login node (internet available), before submitting:**

```bash
# 1. dev qrels — not obtainable from the HF datasets
wget https://raw.githubusercontent.com/castorini/anserini-tools/master/topics-and-qrels/qrels.msmarco-passage.dev-subset.txt \
     -O /scratch/$USER/dense-retrieval-SOTA/data/processed/msmarco_dev_qrels.txt

# 2. warm the HF cache so the offline job can read it
export HF_HOME=/scratch/$USER/dense-retrieval-SOTA/data/bright
python -c "
from datasets import load_dataset
load_dataset('Tevatron/msmarco-passage-corpus', split='train', trust_remote_code=True)
load_dataset('Tevatron/msmarco-passage', split='train')
load_dataset('Tevatron/msmarco-passage', split='validation')"
```

The streaming calls will still not work from a cold cache inside the offline job. If they
fail, generate the four processed files on a login node instead and let the job find them:
`msmarco_corpus.jsonl`, `msmarco_train_queries.jsonl`, `msmarco_train_qrels.txt`,
`msmarco_dev_queries.jsonl` under `$PROC`. `run_setup` skips whatever already exists.

⚠️ `bug_fixes.md` holds the fuller MS MARCO runbook, but it is **explicitly gitignored**
(`.gitignore:82`, alongside `CLAUDE.md`) and therefore **not on `main`** — it exists only in
your local checkout. Everything needed to unblock experiment 7 is inlined above so this
checklist stands alone.

Then:

```bash
sbatch scripts/run_ance_msmarco_singularity.sh
```

| | |
|---|---|
| allocation | `gpu-a100`, **2 GPUs**, `--time=24:00:00` (~16–18 h estimated in the launcher) |
| checkpoint cadence | `save_steps: 1250` |
| corpus | 8.8M passages; ~35 min per full encode at `per_device_eval_batch_size: 256` |

**This is the only one of the seven never run end to end** — treat the first submission as a
smoke test.

**Success signal**
```bash
grep -A 10 'RESOLVED TRAINING CONFIG' logs/ance_msmarco_<jobid>.out   # recipe: ance_msmarco
grep 'GPU(s) detected' logs/ance_msmarco_<jobid>.out
ls $MODELS/ance_msmarco_bge_m3/checkpoint-1250/
```

### MS MARCO evaluation

```bash
sbatch scripts/eval_msmarco_singularity.sh          # gpu-a100, 1 GPU, 4 h, metric recip_rank (MRR)
```

✅ **Defect P4 is fixed** — the launcher now propagates its exit code and **exits 2** with a
clear message when no checkpoint exists, instead of silently evaluating `--model_path None`.

---

## BRIGHT evaluation

```bash
sbatch scripts/run_evaluate_singularity.sh              # defaults to the 4 pilot domains
EVAL_DOMAINS=all sbatch scripts/run_evaluate_singularity.sh
```

| | |
|---|---|
| allocation | `gpu-a100`, 1 GPU, `--time=02:00:00` |
| default domains | `biology,economics,stackoverflow,theoremqa_questions` |

Use **`gpu-a100`, not `gpu-a100-small`** — the small partition ran 2.11 s/it and could not fit
four domains inside its 4 h cap. Per-domain `{domain}_results.json` is written as each domain
finishes, so a timeout is resumable by passing only the gaps via `EVAL_DOMAINS`.

---

## What is unchanged from pre-consolidation

Verified by `AC-SURFACE-01` (`SURFACE_ALLOWLIST_OK`) against `archive/main-post-promotion`:

- **no pre-existing `scripts/*.sh` launcher changed by a single byte** — job scripts, SLURM
  headers, `--bind` mounts and env assumptions are exactly as they were. One deliberate
  exception, made after that verification and narrowly allowlisted: `run_inbatch_singularity.sh`'s
  `--time` was restored from its temporary `14:00:00` smoke value to `24:00:00`;
- the only new launchers are `eval_msmarco_singularity.sh` and `run_ance_msmarco_singularity.sh`;
- `config/config.yaml` differs only by the added `training.ance_msmarco` block;
- every entry point's CLI flags, recipe names, config path keys, environment keys and
  `sys.path` handling are byte-for-byte unchanged — the Step-6 logging added output only.

So any behaviour difference you observe on the cluster is **not** consolidation drift in the
job plumbing.

Defects **P1–P5 have since been fixed** in an authorised post-consolidation pass; the four
launcher edits are pinned line-for-line by `AC-SURFACE-01` amendments A4/A5 and
mutation-tested. **P6 (MS MARCO offline vs streaming) and D1 (`bug_fixes.md` gitignored)
remain open** — see *Pre-existing defects* in `CONSOLIDATION_STATUS.md`.
