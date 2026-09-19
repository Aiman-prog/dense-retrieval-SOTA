# DelftBlue GPU checklist — post-consolidation

Seven experiments, all on `main`. **You run these; I cannot.** Nothing below verifies training
*correctness* or retrieval quality — only that the job starts, logs the config it was asked
for, and writes a checkpoint with finite loss.

There is **no `--max_steps` flag on any entry point**, and adding one would violate the
consolidation rules. Where a smoke mode is reachable, use it; where it is not, assert on the
**first checkpoint written with finite loss** rather than a step count.

---

## Before your first job — four things that will otherwise bite

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
| 7 ANCE paper (MS MARCO) | `logs/ance_paper_<jobid>.out` | `.err` |
| 7a BM25 warm-up (prereq for 7) | `logs/ance_warmup_<jobid>.out` | `.err` |
| 7a-pre warm-up preflight (CPU) | `logs/ance_warmup_preflight_<jobid>.out` | `.err` |
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
1 in-batch ──► models/inbatch_mixed_bge_m3. A BASELINE ARM ONLY — no longer the
               base_model for anything. 3,4,5,6 now start from BAAI/bge-m3, so
               every arm is "BGE-M3 + one mining strategy" at an equal step budget.
5 seq Fast-GRASS ─────► builds temp_grass_workdir/stale_index/corpus.pkl
                      │  (or: sbatch scripts/launchers/run_refresh_stale_index_singularity.sh)
                      ▼
              6 async Fast-GRASS   ← HARD-FAILS without that pickle (defect B2)
7 MS MARCO ──► needs 7a, the BM25 warm-up, built first (Microsoft's is gone); see §7
```

`train_async_fast_grass.py:280` raises `FileNotFoundError: stale index not found at …` and
**never builds one itself**. Run experiment 5, or the refresh job (`gpu-a100`, 1 GPU, 4 h),
first.

### 4. The environment is `~/.local`, not the container (defect P7)

Verified on the cluster 2026-08-20. `pytorch_2.1.sif` gives you CUDA and a torch 2.1 that
**nothing imports**; it has **no `transformers` at all**. Everything resolves from
`~/.local/lib/python3.10/site-packages`:

| | |
|---|---|
| torch | **2.10.0+cu128** (not the 2.1.0/cu118 every doc used to claim) |
| transformers | 4.40.2 |
| accelerate, peft, datasets, safetensors, faiss-gpu, numpy | exactly the `requirements-hpc.txt` pins |

🚫 **Never set `PYTHONNOUSERSITE`** — it breaks all seven pipelines instantly.

This is benign: all six entry points import cleanly, and every model in `models/` postdates
the 2026-02-22 torch upgrade, so your whole results table came off one stack. But `~/.local`
is unversioned and holds three hand-applied Tevatron patches
(`DELFTBLUE_SETUP.md` §2). Sanity-check it in ~1 min before a long run:

```bash
singularity exec /scratch/$USER/containers/pytorch_2.1.sif python -c "
import torch, transformers
from tevatron.retriever.modeling import DenseModel
print(torch.__version__, transformers.__version__, 'tevatron OK')"
```

Backup of the patched package: `/scratch/$USER/tevatron_patched_20260820.tgz` (93K).
Resolved environment: `docs/DELFTBLUE_ENVIRONMENT.md`.

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
when no snapshot directory holds a `config.json`. `ance`, `grass`, `fast_grass` and
`async_fast_grass` now all train from **`BAAI/bge-m3`** (resolved from the offline hub cache),
the same base as the in-batch baseline. Only `ance_paper` trains from a path,
`/scratch/$USER/dense-retrieval-SOTA/models/ance_bm25_warmup_60k`. If a configured path is
missing the run will **train cleanly against the wrong weights**; the block prints
`[PATH DOES NOT EXIST]`:

```bash
grep 'PATH DOES NOT EXIST' logs/<name>_<jobid>.out && echo "STOP — wrong base model"
```

Empty output is what you want.

`eval_msmarco.py` and `run_all_evals.py` are **not** among the six and print no such block.

---

## 1. In-batch — `train_inbatch.py`

```bash
sbatch scripts/launchers/run_inbatch_singularity.sh
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

**No longer a prerequisite for anything.** Experiments 3–6 now train from `BAAI/bge-m3`,
the same base this arm starts from, so it is a baseline to compare against rather than a
dependency to wait on. The completed `inbatch_mixed_bge_m3` (q1024/p512, 2 epochs, 10,314
steps, 12.2 h) already satisfies the current encoding contract and needs no retrain.

---

## 2. Cross-batch — `train_crossbatch.py`

```bash
sbatch scripts/launchers/run_crossbatch_singularity.sh
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

⚠️ **This one has probably never succeeded.** As of 2026-08-20
`$MODELS/crossbatch_mixed_bge_m3_epoch2/` is an **empty directory**, despite `save_steps=100`
meaning a checkpoint should appear within minutes. P3 is why nobody noticed: the launcher
reported `COMPLETED 0:0` no matter what `torchrun` did. Treat your next run as a first run,
and read the `.err` file.

**Success signal**
```bash
grep -A 10 'RESOLVED TRAINING CONFIG' logs/crossbatch_bge_<jobid>.out
tail -40 logs/crossbatch_bge_<jobid>.err        # the traceback lives here, not in .out
ls $MODELS/crossbatch_mixed_bge_m3_epoch2/checkpoint-100/
```

---

## 3. ANCE (BRIGHT) — `train_ance.py`

```bash
# validate the inputs first — no GPU, nothing written, minutes not hours
srun --partition=compute --time=00:20:00 --cpus-per-task=2 --mem-per-cpu=8000M \
     --account=Education-EEMCS-MSc-DSAIT \
     singularity exec --bind /scratch/$USER:/scratch/$USER \
     --bind /home/$USER:/home/$USER /scratch/$USER/containers/pytorch_2.1.sif \
     python scripts/train_ance.py --preflight

sbatch scripts/launchers/run_ance_singularity.sh
```

| | |
|---|---|
| allocation | `gpu-a100`, **2 GPUs** (Trainer GPU 0 / Inferencer GPU 1), `--time=24:00:00` |
| smoke flag | `--preflight` (input validation only, no GPU) |
| checkpoint cadence | `save_steps: 1000` sets the ANN refresh interval *m*; job 70367 achieved it (~52 min/round vs ~77 min/1000 steps, `stale_steps` flat). The MS MARCO arm does **not** — see §7 |
| encoding | q1024/**p512**, dynamic padding, `per_device_eval_batch_size: 64` |
| query shards | `ann_chunk_factor: 1` — every BRIGHT training query is mined each round |
| stall guard | `max_encode_seconds` — a hung encode raises, the inferencer exits nonzero |

⚠️ **Run `--preflight` before every ANCE submission.** Job 59904 was handed two A100s and
died 1:47 later in `preflight_inputs`; the whole check costs minutes on `compute`. Like the
async one it loads the corpus **with text**, so it is a batch job, never a login node.

⚠️ A positive docid absent from `reasonir_corpus.jsonl` is **not** by itself a defect.
`preprocessor._derive` remaps the corpus and the qrels (whitespace escaping, duplicate-text
collapse) and deliberately leaves the mixture holding raw ids. Preflight canonicalizes and
reports the count; only a positive whose TEXT is missing, or whose canonical owner the
query's qrels do not carry, is real staleness — and that means regenerate the derived
artifacts, not relax the guard.

**The smoke has run — job 64255, A100 80GB.** At the (now retired) q1024/p1024 shape:
`bge_train` batch 64 / group 2 peaked at **60.09 GiB** and **11,290.8 ms/step**;
`bge_encode` at eval batch 64 peaked at **13.80 GiB** and **847.0 ms/step = 76 docs/s**.
Those numbers are what retired p1024: 11.29 s/step is ~32h for two epochs against a 24h
wall, and 60 GiB of 80 with gradient checkpointing already enabled means it cannot be
turned off. Both readings are worst case — `_varied` forces a cap-length item, so every
batch pads to the cap. Re-run it (`sbatch scripts/launchers/run_gpu_smoke_singularity.sh`)
only when a shape changes; it exits nonzero if any requested arm fails. The shard factors
and generous encode hang timeouts are fixed configuration, not auto-tuned by this probe.

⚠️ The smoke's encode line prints "648,942 **BRIGHT** docs" — that count is the ReasonIR
corpus (`gpu_memory_smoke.py:39`), not BRIGHT. Cosmetic bug in the message only.

The startup block reports **`total_epochs`**, not `num_epochs` — the ANCE recipes have no
`num_epochs` key. Startup now fails before encoding unless PyTorch sees at least two GPUs.

**Success signal.** Exit 0 is not success: the run must prove it refreshed and that
it trained.

```bash
grep 'GPU(s) detected' logs/ance_<jobid>.out            # must say 2 GPU(s)
grep 'Initial round committed' logs/ance_<jobid>.out    # base-model round 0
grep 'round .* — swapping' logs/ance_<jobid>.out        # a refresh was CONSUMED
grep 'distinct checkpoints consumed' logs/ance_<jobid>.out
grep 'checkpoint opportunit' logs/ance_<jobid>.out      # opportunities/completed/consumed
grep '\[run\] validated:' logs/ance_<jobid>.out         # assert_training_succeeded
```

**Two** rounds from **distinct** checkpoints are now required, each consumed for at least
`logging_steps`. One refresh followed by static training used to pass; it no longer does.
A checkpoint opportunity, a completed round and a consumed round are three separate
counters in `ance_trainer_summary.json` and must not be read as one number.

Any of these is a hard failure, and the job exits nonzero:

| log line | meaning |
|---|---|
| `inferencer exited with code N` | no refresh; the run degenerated to static negatives |
| `REFUSED round N` | a round could not prove it belongs to this run |
| `never fabricates a negative` | a query could not supply an ANN negative; round discarded |
| `non-finite loss` | diverged; no checkpoint written |
| `non-finite gradient norm` | backward overflowed; no optimizer step or checkpoint written |
| `TimeoutExpired` | an encode hung past `max_encode_seconds` |
| `distinct docid(s)` | the encoded corpus and the corpus file disagree |

⚠️ **Job 9566838 / 0.1683 is quarantined and must not be used as a reference point.**
See `P-ANCE-01` in `CONSOLIDATION_STATUS.md`.

**Reporting.** `train_ance.py` runs no in-job BRIGHT evaluation. The only reportable
path is:

```bash
EVAL_REQUIRE_EXISTING=1 EVAL_DOMAINS=all EVAL_MODEL_PATH=$MODELS/ance_mixed_bge_m3 \
  sbatch scripts/launchers/run_evaluate_singularity.sh
```

---

## 4. Sync GRASS — `run_grass.py`

```bash
GRASS_UNCERTAINTY=mc_dropout sbatch scripts/launchers/run_grass_singularity.sh   # or ema
```

| | |
|---|---|
| allocation | `gpu-a100`, 1 GPU, `--time=20:00:00` |
| checkpoint cadence | `save_steps: 1000` |
| env knobs | `GRASS_UNCERTAINTY`, `GRASS_MODEL_SUFFIX`, `GRASS_NUM_EPOCHS`, `GRASS_P`, `GRASS_L`, `GRASS_LAMBDA` |

✅ **Defect P5 is fixed** — `GRASS_DEBUG=1` now reaches `--debug` (512-item mixture):

```bash
GRASS_DEBUG=1 GRASS_UNCERTAINTY=mc_dropout sbatch --time=01:00:00 scripts/launchers/run_grass_singularity.sh
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
FAST_GRASS_UNCERTAINTY=mcdp sbatch scripts/launchers/run_fast_grass_singularity.sh
```

| | |
|---|---|
| allocation | `gpu-a100`, 1 GPU, `--time=20:00:00` |
| checkpoint cadence | `save_steps: 1000` |
| env knobs | `FAST_GRASS_{UNCERTAINTY,LAMBDA,B_DOC,L,T,M,MC_DROPOUT_P,SELECTION_MODE,EMA_ALPHA,NUM_EPOCHS,MODEL_SUFFIX,NO_EVAL,NO_REGISTRY}` |

✅ Defect P5 fixed here too — `FAST_GRASS_DEBUG=1` reaches `--debug`:

```bash
FAST_GRASS_DEBUG=1 FAST_GRASS_UNCERTAINTY=mcdp sbatch --time=01:00:00 scripts/launchers/run_fast_grass_singularity.sh
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
`sbatch scripts/launchers/run_refresh_stale_index_singularity.sh`). Verify before submitting:

```bash
ls -lh /scratch/$USER/dense-retrieval-SOTA/temp_grass_workdir/stale_index/corpus.pkl
```

```bash
python scripts/train_async_fast_grass.py --preflight       # login node, no GPU
ASYNC_FG_DEBUG=1 ASYNC_FG_MAX_ROUNDS=1 ASYNC_FG_FRESH=1 \
    sbatch scripts/launchers/run_async_fast_grass_singularity.sh     # smoke
sbatch scripts/launchers/run_async_fast_grass_singularity.sh         # real run
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

## 7. MS MARCO data prep — required by `ance_paper`

The `ance_msmarco` BGE sanity recipe that used to occupy this slot is retired
(`P-ANCE-03`). The data preparation below is **not** retired: `ance_paper` shares the same
`setup_mode: tevatron_msmarco`, so nothing on the MS MARCO side runs until this is done.

### 🛑 Blocked on data prep. (defect P6)

The job **cannot fetch its own data**:

- the launcher exports `HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1`;
- `prepare_msmarco_full_corpus` / `_tevatron_train` / `_dev` call `load_dataset()` on
  two separately pinned Tevatron repositories, **two of them with `streaming=True`**;
- `msmarco_dev_qrels.txt` is not in either dataset at all — the `validation` split has no
  `positive_passages`.

**Do this on a login node (internet available), before submitting:**

```bash
# 1. dev qrels — not obtainable from the HF datasets
wget https://raw.githubusercontent.com/castorini/anserini-tools/master/topics-and-qrels/qrels.msmarco-passage.dev-subset.txt \
     -O /scratch/$USER/dense-retrieval-SOTA/data/processed/msmarco_dev_qrels.txt

# 2. build the four derived files while network access is available. The Python
#    snippet reads both immutable revisions from config/config.yaml.
export DATA_BASE_DIR=/scratch/$USER/dense-retrieval-SOTA
export PYTHONPATH=$PWD/src
python - <<'PY'
from data.preprocessor import BRIGHTPreprocessor, set_msmarco_revisions
from utils.helpers import load_config

cfg = load_config()['data']['msmarco_reproduction']
set_msmarco_revisions(passage=cfg['passage_revision'],
                      corpus=cfg['corpus_revision'])
p = BRIGHTPreprocessor()
p.prepare_msmarco_full_corpus()
p.prepare_msmarco_tevatron_train()
p.prepare_msmarco_dev()
PY
```

The offline GPU job consumes the generated artifacts and does not contact Hugging Face:
`msmarco_corpus.jsonl`, `msmarco_train_queries.jsonl`, `msmarco_train_qrels.txt`,
`msmarco_dev_queries.jsonl` under `$PROC`. `run_setup` skips whatever already exists.

⚠️ `bug_fixes.md` holds the fuller MS MARCO runbook, but it is **explicitly gitignored**
(`.gitignore:82`, alongside `CLAUDE.md`) and therefore **not on `main`** — it exists only in
your local checkout. Everything needed to unblock experiment 7 is inlined above so this
checklist stands alone.

With those four files in place, the MS MARCO consumer is the paper-fidelity run — see
**Paper-fidelity ANCE** below for the submit command and its acceptance bar.

| | |
|---|---|
| allocation | `gpu-a100`, **2 GPUs**, `--time=24:00:00` |
| corpus | 8.8M passages; ~35 min per full encode at `per_device_eval_batch_size: 256` |

**Never run end to end** — treat the first submission as a smoke test.

### MS MARCO evaluation

```bash
sbatch scripts/launchers/eval_msmarco_singularity.sh          # gpu-a100, 1 GPU, 4 h, metric recip_rank (MRR)
```

✅ **Defect P4 is fixed** — the launcher now propagates its exit code and **exits 2** with a
clear message when no checkpoint exists, instead of silently evaluating `--model_path None`.

---

## BRIGHT evaluation

```bash
sbatch scripts/launchers/run_evaluate_singularity.sh              # defaults to the 4 pilot domains
EVAL_DOMAINS=all sbatch scripts/launchers/run_evaluate_singularity.sh
```

| | |
|---|---|
| allocation | `gpu-a100`, 1 GPU, `--time=02:00:00` |
| default domains | `biology,economics,stackoverflow,theoremqa_questions` |

Use **`gpu-a100`, not `gpu-a100-small`** — the small partition ran 2.11 s/it and could not fit
four domains inside its 4 h cap. Per-domain `{domain}_results.json` is written as each domain
finishes, so a timeout is resumable by passing only the gaps via `EVAL_DOMAINS`.

---

## Paper-fidelity ANCE — a fixed step budget, stopping short of 600K

Not a rung of the BRIGHT ladder. This runs our ANCE code from a 60K BM25 warm-up
on a fixed step budget: the linear decay is computed against `scheduler_max_steps`
(1,000,000, the supplied command's horizon) while the run stops at `train_stop_steps`.
600K does not fit one 24 h allocation on a single A100 and there is no cross-job resume,
so stopping short is an explicit, recorded hardware deviation — not a 600K claim.

⚠️ **Build the warm-up first — Microsoft's is gone.** Both released blob URLs return
`HTTP 409`, microsoft/ANCE #23/#24/#26 have been open since 2022, and the only mirror is
bit-identical to the released **600K FINAL** (203/203 tensors, max diff 0), which
`assert_permitted_init` refuses as an initialization. `run_ance_warmup_singularity.sh` builds
one from `roberta-base` on the BM25 negatives already in the mixture: **1** GPU, ~2-3 h.

**Preflight it first** — CPU only, no GPU bound, minutes on `compute-p1`. It runs the same
code on the same data: loads the real 5.2 GB mixture and reports its ragged-negative counts,
builds the model under the head-freshness guard, takes a few optimization steps, writes a
rescue checkpoint, saves and reloads through `load_ance_encoder`. Job 70494 burned a GPU
allocation to die on the mixture 2:30 in.

```bash
sbatch scripts/launchers/run_ance_warmup_preflight_singularity.sh   # then, only if it passes:
sbatch scripts/launchers/run_ance_warmup_singularity.sh
```

The GPU launcher also runs the preflight as its own stage 1, so a regression fails in minutes
rather than at the 5 h wall. Re-running it **refuses to overwrite a finished warm-up** unless
`ANCE_WARMUP_OVERWRITE=1`; `save_steps` writes `interim-<step>/` as a rescue artifact, so a
wall-clock kill still leaves usable weights.

Optionally **gate it** — expect MRR@10 around 0.311. `EVAL_ALLOW_DRIFT=1` is **required**:
the warm-up trains at q128/p128 but is consumed at q64/p512, so without it `eval_msmarco.py`
exits on encoding-contract drift before encoding anything. q64/p512 is the deliberate choice
— it is the contract `ance_paper` uses the warm-up under, and the one upstream's 0.311 refers
to.

```bash
EVAL_ALLOW_DRIFT=1 EVAL_MODEL_PATH=$MODELS/ance_bm25_warmup_60k \
  sbatch scripts/launchers/eval_msmarco_singularity.sh
```

⚠️ **SKIPPED for the current warm-up, by decision** (2026-09-15): it costs a GPU slot and
does not change what the run does. Instead the weights are pinned by content and re-verified
on disk (204854, `17e06536…b802cf`), with 76441's own record (loss 15.16 → ~0.1, probe
`rank_acc` 0.5 → 1.0) as evidence it trained. If the 300K run underperforms, rule this out first.

The training run **refuses to start** until `training.ance_paper.expected_init_sha256` matches
the warm-up on disk. Produce it on the cluster with:

```bash
# train_ance_warmup.py PRINTS this on success — copy it from the log. To recompute,
# note transformers 4.40.2 writes model.safetensors (either name is accepted):
python -c "import sys;sys.path.insert(0,'src');from utils.helpers import _sha256;\
           print(_sha256('$MODELS/ance_bm25_warmup_60k/model.safetensors'))"
```

**Prepare round 0 first — one GPU, outside the training allocation.** It costs ~6h38m, and
job 204931 spent that *inside* a 24 h two-GPU job, which is why 300K steps (20.5h) could not
fit. The artifact is reusable: round 0 depends only on the warm-up weights and the corpus.
Upstream splits the same stage out (`commands/run_train.sh --end_output_num 0`).

```bash
sbatch scripts/launchers/run_ance_paper_prepare_singularity.sh      # ~6.6h, 1 GPU
#   stage 1 re-runs --preflight on CPU; stage 2 mines round 0 into
#   $DATA_BASE_DIR/prepared_rounds/ance_paper_initial (override with ANCE_PREPARE_DIR)
#   and writes initial_mining_timings.jsonl — the per-phase breakdown.
```

Its own launcher writes the recipe out, so no unset variable can substitute a different
model:

```bash
# ANCE_INITIAL_ROUND adopts the prepared round; without it round 0 is mined inline and
# the job will not finish 300K steps. Adoption is refused unless the initialization,
# corpus, queries, qrels, mixture, mining settings and seed all match.
# ANCE_OVERWRITE=1 is required when the output dir still holds checkpoints (ANCE cannot
# resume, so starting fresh deletes them) — an explicit choice, never silent.
sbatch --export=ALL,ANCE_INITIAL_ROUND=$DATA_BASE_DIR/prepared_rounds/ance_paper_initial \
       scripts/launchers/run_ance_paper_singularity.sh

# Chain them in one go — the training job accrues queue priority while preparation runs:
#   sbatch --dependency=afterok:<prepare-jobid> --kill-on-invalid-dep=yes \
#     --export=ALL,ANCE_INITIAL_ROUND=...,ANCE_OVERWRITE=1 \
#     scripts/launchers/run_ance_paper_singularity.sh

# EVAL_RECIPE defaults to ance_paper; all three go through the same evaluator.
EVAL_MODEL_PATH=<released-ANCE-600K>       sbatch scripts/launchers/eval_msmarco_singularity.sh
EVAL_MODEL_PATH=$MODELS/ance_bm25_warmup_60k sbatch scripts/launchers/eval_msmarco_singularity.sh
EVAL_MODEL_PATH=$MODELS/ance_paper_roberta   sbatch scripts/launchers/eval_msmarco_singularity.sh
```

Prerequisites: MS MARCO built on the cluster (`P-PRE-03` blocks it locally) — **done**, all
five artifacts including `msmarco_dev_qrels.txt` are on scratch — `roberta-base` in the offline
hub cache, and both `data.msmarco_reproduction.{passage,corpus}_revision` values pinned in
`config/config.yaml`. No checkpoint conversion is needed.

⚠️ The released 600K checkpoint is still worth having as the **evaluator** reference
(`castorini/ance-msmarco-passage` mirrors it): scoring it through our path and recovering
0.330 / 0.959 proves the measurement is right before any of our own numbers are trusted.

**Step 1 — validate the evaluator, before trusting any number it produces.** Run the
released 600K checkpoint through the evaluator first. It must report
`within_paper_tolerance: true`, i.e. MRR@10 within ±0.005 of **0.330** and Recall@1000
within ±0.005 of **0.959**, on the official Dev small split (6,980 judged queries).
**Do not proceed if it misses:** that means the evaluator or the pinned artifacts are
wrong, and every later number is meaningless. A checkpoint that passes this preflight
but misses the acceptance bar below is a training result, not an evaluator failure.

**Step 2 — acceptance for our own run.** `within_paper_tolerance` is a diagnostic here,
not a verdict: at `train_stop_steps` against a 600K reference, our run is *expected* to
land outside the band, and that is a recorded budget deviation rather than a failed
reproduction. What it must actually clear:

- MRR@10 above the locally evaluated 60K warm-up (the warm-up is ~0.311 upstream;
  use the number *your* evaluator produced for it, not the published one);
- Recall@1000 >= 0.949;
- `min_fresh_rounds` (2) consumed rounds from DISTINCT checkpoints in the run manifest, plus the achieved refresh cadence reported beside the configured one;
- `[run] validated:` printed after the full runtime-derived budget.

**There is no resume.** If the job hits the 24 h wall clock it does not continue, and a
timeout is a failed run, not a partial one. 300K steps fits **only because round 0 is
prepared separately**: job 204931 measured 0.246 s/step (300K = 20.5h) plus 6h38m of
inline initial mining = ~27h, and was killed. With `--initial-round` the training job
starts at step 1 and the arithmetic works.

---

## What is unchanged from pre-consolidation

`AC-SURFACE-01` used to verify this against `archive/main-post-promotion`; it is retired
(`P-ANCE-03`). The list below is now a description, not a checked invariant:

- **no pre-existing `scripts/*.sh` launcher changed by a single byte** — job scripts, SLURM
  headers, `--bind` mounts and env assumptions are exactly as they were. One deliberate
  exception, made after that verification and narrowly allowlisted: `run_inbatch_singularity.sh`'s
  `--time` was restored from its temporary `14:00:00` smoke value to `24:00:00`;
- every entry point's CLI flags, recipe names, config path keys, environment keys and
  `sys.path` handling are byte-for-byte unchanged — the Step-6 logging added output only.

So any behaviour difference you observe on the cluster is **not** consolidation drift in the
job plumbing.

Defects **P1–P5 have since been fixed** in an authorised post-consolidation pass; the four
launcher edits are pinned line-for-line by `AC-SURFACE-01` amendments A4/A5 and
mutation-tested. **P6 (MS MARCO offline vs streaming), P7 (the `~/.local` environment) and
D1 (`bug_fixes.md` gitignored) remain open** — see *Pre-existing defects* in `CONSOLIDATION_STATUS.md`.
