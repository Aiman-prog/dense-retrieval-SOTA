# Branch consolidation — dense-retrieval-SOTA

Consolidate this repo to `main` only. Recon is already done; the facts below are **given**, do not re-derive them. Verify only where a step says to.

GPU training runs on DelftBlue and I run it myself. You cannot run it. "It imports" is not evidence training works, and nothing in this task verifies training *correctness* — only that jobs start, log the right config, and don't crash. Do not add criteria implying otherwise.

---

## Rules

1. **Consolidation only.** No refactoring, renaming, or deduplication. Don't modify anything you weren't asked to touch. Collect cleanup ideas in a list at the end; don't act on them. One exception, spelled out in Step 6 (startup logging).
2. **`fast-grass` config wins**, with one carve-out named in "Given facts". Older branches are superseded, not alternatives.
3. **Deletion is a proposal.** Never delete a branch without my explicit approval, and never before its backup tag is pushed.
4. No `Co-Authored-By` / AI-attribution trailers in commit messages.

---

## Given facts (verified, do not re-run recon)

### Branch topology
- `main` and `new-grass` are the **same commit** (`b633376`).
- `main`, `new-grass`, `sequential-grass` are **fully contained in `fast-grass`** (zero unique commits). `main` is an ancestor of `fast-grass`, so promoting is a fast-forward.
- Only three branches hold unreachable work:

| branch | unique commits | last | contents |
|---|---|---|---|
| `baseline` | 6 | 2026-04-12 | MS MARCO ANCE track (see below) |
| `async-grass-v2` | 7 | 2026-05-28 | `train_grass_async_v2.py`, `run_grass_async_v2_{miner,trainer}.py`, `run_grass_case_lite.py`, calibration scripts; also ~2500 lines of committed `logs_cluster/eval_*` and a 629KB `sample_efficient_method.pdf` |
| `good-grass` | 1 | 2026-05-18 | `run_grass_seq.py`, `run_grass_seq_bandit.py` + launchers, `src/utils/bandit.py` work |

- No force-pushes, no rewritten history, no stashes. All branch tips match their `origin/` counterparts. `git fsck` dangling objects are normal amend/rebase debris.

### Components and their real paths (on `fast-grass`)

| component | paths |
|---|---|
| preprocessor | `src/data/preprocessor.py`, entry `run_setup()` at line ~316 |
| in-batch | `scripts/train_inbatch.py` + `scripts/run_inbatch_singularity.sh` |
| cross-batch | `scripts/train_crossbatch.py` + `scripts/run_crossbatch_singularity.sh` |
| ANCE (BRIGHT) | `scripts/train_ance.py`, `run_ance_train.py`, `run_ance_data_gen.py` + `run_ance_singularity.sh` |
| sync GRASS | `scripts/run_grass.py` + `run_grass_singularity.sh` |
| async GRASS | `scripts/train_async_fast_grass.py`, `run_async_fast_grass_{miner,train}.py` + `run_async_fast_grass_singularity.sh` |
| sequential Fast-GRASS | `scripts/run_fast_grass.py` + `run_fast_grass_singularity.sh` |
| **ANCE (MS MARCO)** — 7th, on `baseline` only | `scripts/eval_msmarco.py`, `run_ance_msmarco_singularity.sh`, `eval_msmarco_singularity.sh`; `ance_msmarco` config block; `train_ance.py` `_load_qrels`/`_evaluate`; preprocessor `prepare_msmarco_{full_corpus,tevatron_train,dev}` |

`async-grass-v2`'s async GRASS is a **different architecture** (ANN-refresh) from `fast-grass`'s (cached-MCDP, no ANN rebuild), not an older copy of it. Do not rank them against each other.

### `fast-grass` is newest for everything **except**
- **ANCE**: `baseline:scripts/train_ance.py` is 339 lines vs fast-grass's 242; both diverged from the same 212-line merge-base `1b3d434`. Baseline uniquely has `_load_qrels()` and `_evaluate()`. `run_ance_data_gen.py` is 160 vs 134 lines.
- **preprocessor**: baseline has 3 extra methods and an extended `run_setup()` with a `setup_mode` dispatch.

**Decision (mine, already made):** baseline's extras are a *parallel MS MARCO reproduction track*, not a newer BRIGHT ANCE. Preserve them as the 7th component. Do **not** let them overwrite fast-grass's BRIGHT ANCE or preprocessor behaviour.

### Config
- `model.base_model` (`"BAAI/bge-m3"`) and `model.temperature` (`0.02`) are **identical on every branch**. They are not the drift.
- Real drift, `fast-grass` vs the three older branches: `query_max_len` 1024 vs 256, `passage_max_len` 512 vs 128. **fast-grass wins, pre-approved, do not stop to ask.** Any *other* config value where an older branch would win: stop and flag.
- Config is **not** a single source:
  - `config/config.yaml` is the base.
  - `helpers.get_training_context()` (`src/utils/helpers.py:67`) lets a per-recipe `base_model` override the global. For `grass`, `fast_grass`, `async_fast_grass`, `ance` that override is a hardcoded absolute path `/scratch/aimanabdulwaha/dense-retrieval-SOTA/models/inbatch_mixed_bge_m3` — those four train from a **checkpoint**, not from BGE-M3.
  - Lines 79–88 resolve an HF snapshot dir and **silently fall back** to the raw string when `config.json` is missing. This is the "trains cleanly with the wrong model" hazard. Logging must print the value **after** resolution.

### Preprocessor determinism
- `run_setup()` is deterministic (MD5 dedupe, `drop_duplicates`, insertion order, no timestamps, no RNG). Byte-identical output is a fair criterion **for `run_setup()` only**.
- `prepare_msmarco_train_data()` line ~289 calls `random.shuffle(indices)` **unseeded** despite `seed: 42` in config. Exclude it from the criterion; record it as a cleanup idea (it means the 83,030-row MS MARCO mixture slice is not reproducible).
- `run_setup()` **short-circuits if its three outputs already exist** (line ~343). The fixture dir must be wiped between runs or the diff is vacuously clean.

### Docs are mostly untracked
`async_fast_grass_architecture.md`, `async_fast_grass_implementation_details.md`, `lambda_pilot.md`, `lambda_pilot_experiment_summary.md`, `analysis/` are **untracked**. `CLAUDE.md` is in `.gitignore`. Committed docs on `fast-grass`: `DELFTBLUE_SETUP.md`, `README.md`, `fast_grass_implementation_details.md`, `fast_grass_negative_cache_architecture.md`, `plans/fast_grass_rerun_and_sigma_test_plan.md`. `DELFTBLUE_SETUP.md` differs on `baseline`; fast-grass's wins.

---

## Approach (decided)

**Promote, don't merge.** `main` is an ancestor of `fast-grass`, so:
1. fast-forward `main` to `fast-grass`;
2. then add `baseline`'s MS MARCO track on top as additive commits.

Do **not** merge `good-grass` or `async-grass-v2` — their contents are superseded GRASS generations, and `async-grass-v2` would drag 2500 lines of logs and a PDF into `main`. They are preserved as tags only.

Proposed branch dispositions (Gate A confirms):

| branch | disposition | risk |
|---|---|---|
| `new-grass` | delete outright | none — same commit as `main` |
| `sequential-grass` | delete outright | none — ancestor of `fast-grass` |
| `main` | fast-forwarded to `fast-grass`, kept | none |
| `good-grass` | tag `archive/good-grass`, then delete | none if tag pushed |
| `async-grass-v2` | tag `archive/async-grass-v2`, then delete | none if tag pushed |
| `baseline` | tag `archive/baseline`, cherry-pick MS MARCO track, then delete | see Step 5 |
| `fast-grass` | delete after `main` == `fast-grass` and everything is pushed | none |

---

## Gate A — confirm before any writes

Restate rules 1–3 in your own words, present the disposition table above, and give me the ordered plan with per-step rollback. Then **stop**. No writes, no tags, no commits until I approve.

---

## Gate B — execution, one step at a time

Maintain `CONSOLIDATION_STATUS.md` **in the repo root**, updated *before* moving to the next step, never batched at the end. It must contain: steps done, steps pending, branches merged, branches unmerged-and-therefore-unsafe-to-delete, and the exact next command. If you stop for any reason, a fresh session must be able to resume from that file without re-running recon.

### Step 0 — preprocessor baseline fixture
- Build a fixed small input at `<scratchpad>/preproc_fixture/data/processed/training_mixture/` by sampling the first 200 records from each of `train_msmarco.jsonl`, `train_vl.jsonl`, `train_hq.jsonl`.
- **If those mixture files are not present locally, stop and tell me** — they live on `/scratch` on DelftBlue and I'll have to supply them.
- Run `run_setup()` with `DATA_BASE_DIR` pointed at the fixture root. Record `sha256` of `reasonir_corpus.jsonl`, `train_queries.jsonl`, `train_qrels.txt`.
- Write a re-runnable script for this and commit it under `scripts/` — it is the regression check for every later step. Wiping the three outputs before each run is part of the script.
- Re-run and diff after **every** step below. **Any difference: stop.**

### Step 1 — backup tags
Create and **push** `archive/baseline`, `archive/good-grass`, `archive/async-grass-v2`, `archive/main-pre-consolidation`. Verify each is on the remote before proceeding. Nothing destructive happens before this step completes.

### Step 2 — commit the untracked docs
On `fast-grass`, commit the untracked `.md` files and `analysis/` listed above. Leave `CLAUDE.md` gitignored. Do not commit the loose `.html`/`.png`/`.zip` artifacts or `logs_cluster/fast_grass_*` unless they're small and you flag them first. Re-run Step 0 diff.

### Step 3 — promote `main`
Fast-forward `main` to `fast-grass`, push. Verify `git diff main fast-grass` is empty. Re-run Step 0 diff.
Rollback: `git branch -f main archive/main-pre-consolidation`.

### Step 4 — add MS MARCO track: additive files only
Bring these onto `main` from `baseline` — they exist nowhere else, so there is no conflict:
`scripts/eval_msmarco.py`, `scripts/eval_msmarco_singularity.sh`, `scripts/run_ance_msmarco_singularity.sh`, and the `data.msmarco` + `training.ance_msmarco` config blocks.
Config blocks go in **without touching any existing key**. Re-run Step 0 diff.

### Step 5 — MS MARCO track: the risky part. Gated.
`train_ance.py` (`_load_qrels`, `_evaluate`) and `preprocessor.py` (3 methods + `setup_mode` dispatch in `run_setup()`) need real merges, not file copies.
- Attempt it. Re-run Step 0 diff.
- **If `run_setup()` output changes at all, revert this step and stop.** In that case the MS MARCO track stays on `archive/baseline` alone, and you record that in `CONSOLIDATION_STATUS.md` as a known, accepted gap. Do not try to fix it — that's a later task.
- `eval_msmarco.py` referencing a missing `train_ance.py` helper is an acceptable outcome of reverting; note it.

### Step 6 — startup config logging (the one exception to rule 1)
Every training entry point must log, **before training begins**: resolved `ctx['base_model']` (post-snapshot-resolution), `temperature`, `query_max_len`, `passage_max_len`, `batch_size`, `learning_rate`, `num_epochs`, and the recipe name.

Current state — add only what's missing, change nothing else:
- `scripts/train_crossbatch.py` — logs **nothing**, has no `argparse` at all
- `scripts/train_inbatch.py` — logs batch size + epochs only (line ~71)
- `scripts/run_grass.py`, `scripts/run_fast_grass.py` — check and fill gaps
- `scripts/train_ance.py` — logs base model (lines ~97, ~177) and step budget; fill gaps
- `scripts/train_async_fast_grass.py` — `--preflight` prints some of this; ensure the training path does too

Re-run Step 0 diff. Do not add CLI flags, do not restructure argument parsing.

### Step 7 — doc reference audit (report only)
For `DELFTBLUE_SETUP.md`, `README.md`, `fast_grass_*.md`, `async_fast_grass_*.md`, `lambda_pilot*.md`: check that every file path, module name, class name, and CLI flag referenced still exists on `main`. **Report what doesn't. Do not fix it.** Do not assess whether prose is conceptually accurate.

### Step 8 — deletion proposal
List exactly what you'd delete and the command. **Stop for my approval.** Delete local and remote only after I say so, and only for branches whose tags are confirmed pushed.

---

## Deliverable: DelftBlue checklist (separate file, `GPU_CHECKLIST.md`)

I run these. For each of the seven experiments give me: the exact submit command, expected wall clock, and the **specific success signal** to grep for.

Note on smoke tests: there is **no `--max_steps` flag on any entry point** and adding one violates rule 1. Job wall clocks are 10–24h on `gpu-a100`. Use what exists:
- `run_grass.py`, `run_fast_grass.py`, `train_async_fast_grass.py`: `--debug` (512-item mixture); async also has `--max_rounds N` and `--preflight` (no GPU)
- `train_inbatch.py`, `train_crossbatch.py`, `train_ance.py`: **no debug flag** — assert on *first checkpoint written with finite loss* instead of a step count, using the existing `save_steps` cadence

Per experiment the checklist must confirm: job submits; the startup block from Step 6 shows the **expected** base model, temperature and seq lengths; a checkpoint is written with finite loss; job script, entry point, CLI flags, config paths and env assumptions are unchanged from pre-consolidation.

Flag explicitly that the four recipes training from `/scratch/.../models/inbatch_mixed_bge_m3` will silently fall back to a raw string if that path is missing — the Step 6 log line is what catches it.

---

## At the end

One list of cleanup ideas, not acted on. Seed it with: unseeded `random.shuffle` in `prepare_msmarco_train_data`; `ASYNC_FG_RUN_TESTS` cannot be disabled; committed eval logs and PDFs; `train_crossbatch.py` has no CLI surface.
