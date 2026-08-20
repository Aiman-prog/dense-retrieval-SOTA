# Consolidation status

Resume anchor for the branch consolidation to `main`. A fresh session must be able to
continue from this file alone — no recon re-run. Updated **before** moving to the next step.

Authoritative inputs: `CONSOLIDATION_PROMPT.md` (procedure), `ACCEPTANCE_CRITERIA.md`
(checks), `CLAUDE.md` (repo map). Plan: `~/.claude/plans/desired-final-state-shiny-boot.md`.

---

## Next command

```bash
# Step 7 — doc reference audit, REPORT ONLY. Fix nothing.
#   check every file path / module / class / CLI flag referenced by
#   DELFTBLUE_SETUP.md, README.md, fast_grass_*.md, async_fast_grass_*.md,
#   lambda_pilot*.md against what exists on main.
```

---

## Steps

| # | Step | State |
|---|---|---|
| 0 | preprocessor baseline fixture | **DONE** (`86c74f3`) |
| 1 | backup tags (4), pushed + verified | **DONE** |
| 2 | commit untracked docs on `fast-grass` | **DONE** |
| 3 | fast-forward `main` → `fast-grass`; tag `archive/main-post-promotion`; apply AC-SURFACE-01 amendment | **DONE** |
| 4 | MS MARCO additive files + `training.ance_msmarco` | **DONE** |
| 5 | **gated** merge: preprocessor MS MARCO methods (A3 scope) | **DONE** — gate passed |
| 6 | startup config logging in the 6 entry points | **DONE** |
| 7 | doc reference audit (report only) | pending |
| 8 | deletion proposal — **stop for approval** | pending |
| — | `GPU_CHECKLIST.md` deliverable | pending |

## Branch state

| branch | tip | merged into `main`? | archive tag pushed? | safe to delete? |
|---|---|---|---|---|
| `fast-grass` | `d0571e4` (pushed) | **YES** — `main` fast-forwarded to it | n/a | needs Step 8 approval |
| `main` | `d74e5b3` | — | `archive/main-pre-consolidation` ✅ `archive/main-post-promotion` ✅ | **NO** (kept — this is the target) |
| `new-grass` | `b633376` | same commit as `main` | n/a (same commit) | needs Step 8 approval |
| `sequential-grass` | `8669117` | ancestor of `fast-grass` | n/a (ancestor) | needs Step 8 approval |
| `baseline` | `56a9e15` | **NO** — 6 unique commits | `archive/baseline` ✅ | needs Step 8 approval |
| `good-grass` | `201c7c1` | **NO** — 1 unique commit, never merged by design | `archive/good-grass` ✅ | needs Step 8 approval |
| `async-grass-v2` | `46e548e` | **NO** — 7 unique commits, never merged by design | `archive/async-grass-v2` ✅ | needs Step 8 approval |

**Nothing may be deleted** until Step 8 and explicit approval. Tags being pushed is a
precondition, not permission.

---

## Step 0 — preprocessor baseline fixture — DONE

Commit `86c74f3`, adds `scripts/consolidation_preproc_check.py` (the only file touched).
All pre-existing working-tree modifications and untracked files preserved.

Run it after **every** later step. Any hash change ⇒ roll that step back and stop.

```bash
KMP_DUPLICATE_LIB_OK=TRUE python scripts/consolidation_preproc_check.py
```

### Baseline hashes (recorded pre-consolidation, on `fast-grass` @ `86c74f3`)

```
PREPROC_SHA256 reasonir_corpus.jsonl 7c0b8471ee679599b849d0ea58b4e28b498f743302266f18f3531ebe28914d60
PREPROC_SHA256 train_queries.jsonl   74ae6b8a275878d418f0baed5ba8747b09ccb3e27c666f2f7657214bdd4e6997
PREPROC_SHA256 train_qrels.txt       4e3ae34a1500c628bd720fa9d907eea925f3dbc6ab6b06684e4c0c9c54c17357
```

Fixture summary line, also invariant: `586 unique passages (collapsed 10 duplicate-text
docids)`, `605 unique training queries`, `605 positive pairs`.
Verified byte-identical across two consecutive runs.

### Deviation D1 — fixture source

`CONSOLIDATION_PROMPT.md` Step 0 specifies sampling `train_msmarco.jsonl`, `train_vl.jsonl`,
`train_hq.jsonl` from `training_mixture/`. **Those files do not exist on this machine.**
`$DATA_BASE_DIR` = `/Users/aiamn/scratch/aiamn/dense-retrieval-SOTA`, whose
`data/processed/training_mixture/` holds only `train_reasonir_vl.jsonl` and
`train_reasonir_hq.jsonl` — raw ReasonIR schema (`positives`/`negatives` as plain text
lists), on which `run_setup()` would `KeyError`. No `train_msmarco.jsonl` anywhere locally.

Approved substitute: the first 600 records of `data/processed/train_reasonir.jsonl`, which
does carry `{query_id, query, positive_passages:[{docid,text}], negative_passages:[…]}`,
split 200/200/200 into files with the three specified names.
Source sha256 prefix `365cfc11b60f039c`.

### Deviation D2 — fourth fixture file

The 600 real records contain **no** text appearing under two different docids (the first
such pair is at record ~14474), so `run_setup()`'s `docid_remap` would be empty and both the
MD5 remap branch and the qrels canonicalization consuming it would go unexercised — the exact
logic Step 5's `run_setup()` merge could break. The script therefore also writes
`train_zz_dupes.jsonl`: 5 records re-emitting texts already in the window under fresh docids,
derived from the same 600 records, no RNG, no extra source data.

### Known limitations of the fixture

- ReasonIR-derived: exercises no MS MARCO-specific record shape.
- `run_setup()` uses `mixture_dir.glob("*.jsonl")`, which is readdir-ordered. Stable for a
  fixed directory on one machine — adequate for a before/after regression — but the hashes
  above are **not** portable to another filesystem. Re-baseline if the machine changes.
- Covers `run_setup()` only. `prepare_msmarco_train_data()` calls `random.shuffle()` unseeded
  despite `seed: 42`; it is deliberately excluded (recorded as a cleanup idea).
- The script is `.py`, not the `.sh` named in the plan — it calls `run_setup()` directly, and
  a `.py` also stays clear of `AC-SURFACE-01`'s `scripts/*.sh` allowlist entirely.

---

## Step 1 — backup tags — DONE

Four annotated-free lightweight tags created and pushed to `origin`
(`https://github.com/Aiman-prog/dense-retrieval-SOTA.git`), each verified to match both the
remote ref and its branch tip:

```
archive/baseline                 56a9e15a9   (= baseline)
archive/good-grass               201c7c140   (= good-grass)
archive/async-grass-v2           46e548e88   (= async-grass-v2)
archive/main-pre-consolidation   b6333767d   (= main, pre fast-forward)
```

`STEP1_TAGS_VERIFIED_ON_REMOTE`. Every unreachable-work branch tip is now reachable from a
pushed tag, so the deletion precondition in rule 3 is satisfied for Step 8.

No files changed. Step 0 regression re-run: `STEP0_REGRESSION_OK`, all three hashes identical.

Rollback (tags only, nothing else to undo):
```bash
git tag -d archive/baseline archive/good-grass archive/async-grass-v2 archive/main-pre-consolidation
git push origin --delete archive/baseline archive/good-grass archive/async-grass-v2 archive/main-pre-consolidation
```

---

## Step 2 — commit the untracked docs — DONE

Commit `3740a7c` on `fast-grass`, pushed to `origin` (`4b05c68..3740a7c`).
9 files, +2235/-25 lines, all `.md` or `.json` — verified nothing else was staged.

Committed: `async_fast_grass_architecture.md`, `async_fast_grass_implementation_details.md`,
`lambda_pilot.md`, `lambda_pilot_experiment_summary.md`, the two modified
`fast_grass_*.md`, `analysis/fast_grass_lambda_internal_report.md`, and
`analysis/async_fast_grass_timing/*.json` (2).

`CLAUDE.md` remains gitignored (`.gitignore:78`) — confirmed, not committed.

### Decision D3 — `analysis/` committed docs-only (approved)

`analysis/` is 32 MB, of which ~16 KB is documentation:

| kept | 8 K report + 8 K timing JSON |
|---|---|
| **not tracked** | 4× `cost_log.jsonl` (23.6 M), 4× `mining_log.jsonl` (6.8 M), `fg_logs.tgz` (1.6 M) |

`CONSOLIDATION_PROMPT.md` Step 2 lists `analysis/` wholesale, but also forbids committing
`.zip`/log artifacts without flagging — `fg_logs.tgz` and the per-run logs are exactly that
class, and `main` is permanent. Approved: docs only. The raw logs remain on disk, untracked.

### Flagged, deliberately NOT committed

`Cached-MCDP Fast-GRASS Miner (standalone).html` (524 K),
`Fast-GRASS Miner (standalone)-3.html` (436 K), `Fast-Grass Miner (standalone)-3.png` (192 K),
`new-grass-architecture.png` (276 K), `bib.tex` (4 K), `logs_cluster/fg_32k_logs.zip` (1.3 M),
`logs_cluster/fast_grass_mixed_bge_m3_l{0,1}_32k_ema/` (9.3 M + 9.5 M), `.claude/` (56 K).
Also still untracked: `CONSOLIDATION_PROMPT.md`, `ACCEPTANCE_CRITERIA.md` — see Step 3, where
the A1/A2 amendment makes tracking them a live question.

Step 0 regression re-run: `STEP0_REGRESSION_OK`, all three hashes identical.

Rollback:
```bash
git reset --hard e1f5523 && git push --force-with-lease origin fast-grass
```

---

## Step 3 — promote `main` — DONE

```
main  b6333767d  ->  d0571e44e   (fast-forward)
```

- `git merge --ff-only fast-grass`. **HEAD has 1 parent; 0 merge commits introduced.**
- `git diff main fast-grass` empty; `main` and `fast-grass` are the same commit.
- Pushed `b633376..d0571e4`. `STEP3_PROMOTION_VERIFIED` (local == remote for branch and tag).
- Tag `archive/main-post-promotion` = `d0571e44e`, pushed and verified — this is the
  `AC-SURFACE-01` diff base per amendment A1.

Then, on `main`:
- `87a11e0` — track `CONSOLIDATION_PROMPT.md` + `ACCEPTANCE_CRITERIA.md` (decision D4: both
  were untracked, and amending an untracked file leaves no reviewable diff).
- `d1ba880` — apply amendments **A1** and **A2** to `AC-SURFACE-01` (+26/-13). The amendment
  is also quoted inline in `ACCEPTANCE_CRITERIA.md` itself, above the row.

Amendment verified, not just written: the criterion's python block parses, and a dry run
against the real base passes **launcher set equality, launcher byte-equality, and all six
entry-point CLI/recipe/env/`sys.path`/main fingerprints**. It now fails at exactly one point —
`unexpected added shell files: []` — because Step 4 has not yet added the two MS MARCO
launchers. That is the intended pre-Step-4 state, and it confirms the four assertions that
fired under the old base are gone.

Step 0 regression re-run: `STEP0_REGRESSION_OK`.

Rollback:
```bash
git branch -f main archive/main-pre-consolidation && git push --force-with-lease origin main
git tag -d archive/main-post-promotion && git push origin --delete archive/main-post-promotion
```

---

## Step 4 — MS MARCO track, additive files only — DONE

Commit `fada4ca` on `main`. +169 lines of new scripts, +31 lines of config, nothing else.

**Files** — taken verbatim from `baseline` via `git checkout baseline -- <paths>`; blob hashes
and file modes identical to `baseline` (`100644` for all three, which is already the majority
mode among `main`'s launchers):

- `scripts/eval_msmarco.py`
- `scripts/eval_msmarco_singularity.sh`
- `scripts/run_ance_msmarco_singularity.sh`

**Config** — `training.ance_msmarco` only (28 keys), inserted after the `ance` block.
Per amendment A2, `data.msmarco` already existed and was **not** touched.

Verified additive, not merely "looks additive":
- deleting the new block reproduces `archive/main-post-promotion:config/config.yaml`
  **byte-for-byte** (`CONFIG_ADDITION_EXACT`);
- parsed config minus the block `==` the base config, so no pre-existing value changed;
- `data.msmarco` identical to base.

**`AC-SURFACE-01` now passes end to end** (`SURFACE_ALLOWLIST_OK`), ahead of its Step-6 gate.
Re-run after Step 6 as required.

### Note for Step 5 — the prompt's revert concern does not apply

`CONSOLIDATION_PROMPT.md` Step 5 warns that "`eval_msmarco.py` referencing a missing
`train_ance.py` helper is an acceptable outcome of reverting". It does not reference one:
`eval_msmarco.py` imports only `utils.helpers` and `evaluation.trec_eval_wrapper` from `src/`,
with **no** `train_ance` import and no use of `_load_qrels` / `_evaluate`. So `AC-COMP-08`'s
import check is unaffected by whether Step 5 lands or reverts.

Step 0 regression re-run: `STEP0_REGRESSION_OK`.

Rollback:
```bash
git reset --hard d1ba880
```

---

## Step 5 — GATED merge — DONE, gate PASSED

Commit `f37c74f`. **115 insertions, 0 deletions** in `src/data/preprocessor.py`;
`run_setup()` verified byte-identical to `main`; no pre-existing method changed;
method count 8 -> 11, no duplicate names. The three methods are byte-equal to `baseline`'s.

**Gate:** `STEP5_GATE_PASSED` — all three Step-0 hashes unchanged. This was structurally
guaranteed, not lucky: the change is a class-body append and never touches `run_setup()`.

### The Given facts were wrong here — scope was smaller than billed

`CONSOLIDATION_PROMPT.md` states baseline "uniquely has `_load_qrels()` and `_evaluate()`",
and that the `setup_mode` dispatch must be merged into `preprocessor.run_setup()`. Neither holds:

1. **The `setup_mode` dispatch is not in `preprocessor.py` at all.** It lives in
   `scripts/train_ance.py`'s own `run_setup(recipe_args)`, and that function is **already
   byte-identical on `main` and `baseline`**. Nothing to merge.
2. **`_load_qrels` / `_evaluate` are not baseline-unique.** `fast-grass` refactored them into
   `src/utils/helpers.py` as `_load_qrels` / `evaluate_bright`; `train_ance.py` already imports
   and calls them. Bodies are line-for-line identical to baseline's except: an added docstring,
   `open(f,'r')` -> `open(f)`, a defaulted `temp_workdir_key` parameter, and
   `args['eval_metric']` -> `args.get('eval_metric', 'ndcg_cut_10')`. `evaluate_bright` contains
   the MS MARCO single-file branch verbatim.
3. `main` is additionally **newer**: it removes stale `checkpoint-*` dirs before training, which
   `baseline` does not. Without it `get_last_checkpoint()` permanently shadows new saves.

So the only real gap was the three `prepare_msmarco_*` methods, which `main`'s dispatch already
called but which did not exist -> `AttributeError` on `--recipe ance_msmarco`.

### Amendment A3 (approved) — AC-COMP-08 becomes a no-duplication check

Requiring `_load_qrels`/`_evaluate` in `train_ance.py` would duplicate ~110 lines and force
edits to `main`'s import line and its `evaluate_bright()` call site, undoing the refactor —
against rule 1 (no refactoring) and rule 2 (`fast-grass` wins). `AC-COMP-08` now:
- **requires** `_load_qrels` and `evaluate_bright` in `src/utils/helpers.py`;
- **forbids** `_load_qrels` / `_evaluate` in `scripts/train_ance.py`;
- decides landed-vs-absent on the three preprocessor methods alone.

Strictly stronger than the original, and it encodes the no-duplication constraint in the suite.

`AC-COMP-08`: **PASS** — `MSMARCO_ACCEPT_STATE STEP5_LANDED`.

Rollback:
```bash
git reset --hard fada4ca
```

---

## Step 6 — startup config logging — DONE

Commit `d74e5b3`. One implementation, `log_startup_config()` in `src/utils/helpers.py`;
six call sites. No duplicated logging code.

Every entry point prints, before training begins:

```
==================================================================
RESOLVED TRAINING CONFIG
  recipe               : fast_grass
  base_model           : /scratch/.../models/inbatch_mixed_bge_m3
  base_model source    : as configured (no HF snapshot dir with config.json)  [PATH DOES NOT EXIST]
  temperature          : 0.02
  query_max_len        : 1024
  passage_max_len      : 512
  batch_size           : 64
  learning_rate        : 1e-5
  num_epochs           : 2
==================================================================
```

Grep target: `RESOLVED TRAINING CONFIG`.

- `base_model` is the **post-snapshot-resolution** value. `get_training_context()` falls back
  to the raw configured string when no snapshot dir holds a `config.json`, silently — this
  block is the guard against training cleanly on the wrong weights. An absolute path that is
  not on disk is marked `[PATH DOES NOT EXIST]`; that fires for the four checkpoint recipes
  whenever `/scratch/.../models/inbatch_mixed_bge_m3` is missing.
- Recipes do not share one spelling, so the block reports the key actually present rather than
  inventing one: `per_device_batch_size` for `crossbatch` (it has no `batch_size`),
  `total_epochs` for `ance` / `ance_msmarco` (they have no `num_epochs`).
- Calls sit **after** any CLI override (`--num_epochs`, `--lambda_val`, `--model_suffix`, …)
  so the block reports what will actually run. In the async orchestrator it also sits ahead of
  the `--preflight` branch, so both paths print it.
- No CLI flag added, no argument parsing restructured, no config key introduced.

**Not changed:** `train_inbatch.py`'s and `run_fast_grass.py`'s pre-existing print blocks are
left exactly as they were, per rule 1. A few values therefore appear twice in those two logs.
Removing the older lines would be an unrequested change and could break existing greps.

Step 0 regression re-run: `STEP0_REGRESSION_OK`.

Rollback:
```bash
git reset --hard f37c74f
```

### Mandatory post-Step-6 re-run — every eligible row executed

| row | result |
|---|---|
| AC-SURFACE-01 | **PASS** `SURFACE_ALLOWLIST_OK` — reference diff shows only the two Step-4 launchers + one config hunk; all six entry-point CLI/recipe/env/`sys.path`/main fingerprints unchanged despite the logging edits |
| AC-COMP-01 … 07 | **PASS** (re-run) |
| AC-COMP-08 | **PASS** `MSMARCO_ACCEPT_STATE STEP5_LANDED` (re-run) |
| AC-TEST-01 | **PASS** — 13/13 `CPU_SUITE_OK` |
| AC-INV-06 | **PASS** `SRC_IMPORT_BOUNDARY_OK 11 Python files checked` |

No exit-2 `IMPORT_ENVIRONMENT_OUT_OF_SCOPE` anywhere. `grass_test.py`'s known MQ hash-collision
flake did not occur (`PYTHONHASHSEED=0` is pinned by the criterion).

---

## Pre-existing defects — NOT introduced by consolidation, NOT fixed here

Recorded so they are not rediscovered from scratch. Each says what breaks and why. Fixing them
is out of scope under rule 1; they need a separate, explicitly authorised task.

### P1 — `--recipe ance_msmarco` crashes at startup: `get_path("temp_ance_msmarco")` is `None`

**Symptom.** `TypeError: unsupported operand type(s) for /: 'NoneType' and 'str'` at
`scripts/train_ance.py:161`, a few seconds into the job — before any GPU work.

**Why.** `train_ance.py:159` does `get_path(ctx['args']['temp_workdir'])`. The
`ance_msmarco` recipe sets `temp_workdir: "temp_ance_msmarco"` (`config.yaml:378`), but
`helpers.get_path`'s `path_map` has only `temp_ance`, `temp_grass`, `temp_fast_grass`, and
ends in `path_map.get(key)` — an unknown key returns `None` instead of raising. Line 161 then
does `None / "ann_data"`.

**Scope.** Pre-existing on `baseline` too — `baseline`'s `_evaluate` has the identical
`get_path(args['temp_workdir'])`. Consolidation only made it reachable by bringing the recipe
across. The BRIGHT `ance` recipe is unaffected (`temp_ance` is a known key).

**Fix when authorised.** One line: add `"temp_ance_msmarco": base / "temp_ance_msmarco_workdir"`
to `path_map`. Consider also making `get_path` raise on an unknown key rather than return
`None`, which is what let this stay silent.

**Consequence today.** The MS MARCO track is present, imports, and passes `AC-COMP-08`, but
**will not run**. It fails fast and cheap, not 10 hours in.

---

## Recorded amendments to `ACCEPTANCE_CRITERIA.md` (approved at Gate A, applied at Step 3)

### A1 — `AC-SURFACE-01` `BASE` must be post-promotion `main`, not `b633376`

`AC-SURFACE-01` sets `BASE = archive/main-pre-consolidation`, which Step 1 puts on `main`
@ `b633376`. That commit predates all of `fast-grass`, so the Step-3 fast-forward alone trips
four separate assertions:

- adds 8 launchers (`run_fast_grass_singularity.sh`, `run_async_fast_grass_singularity.sh`,
  `run_async_fast_grass_probe_singularity.sh`, `run_fast_grass_timing_singularity.sh`,
  `run_fast_grass_feasibility_singularity.sh`, `run_negcache_feasibility_singularity.sh`,
  `run_refresh_stale_index_singularity.sh`, `run_twoset_feasibility_singularity.sh`)
  ⇒ `head_sh - base_sh != ALLOWED_NEW_SH`;
- changes 11 pre-existing launchers (incl. `run_inbatch_singularity.sh`,
  `run_crossbatch_singularity.sh`, `run_evaluate_singularity.sh`)
  ⇒ `pre-existing launcher changed`;
- `scripts/run_fast_grass.py` and `scripts/train_async_fast_grass.py` do not exist at
  `b633376` ⇒ `entry point is not present at both revisions`;
- config gains `fast_grass`, `async_fast_grass`, `async_fast_grass_pilot`,
  `async_fast_grass_smoke` and rewrites six `crossbatch` keys.

**Amendment:** `BASE = "archive/main-post-promotion"`, a tag created and pushed at Step 3 on
`main` immediately after the fast-forward. `archive/main-pre-consolidation` still points at
`b633376`, so Step 3's rollback (`git branch -f main archive/main-pre-consolidation`) stays
correct. Every substantive check survives and now bounds exactly Steps 4–6.

### A2 — `data.msmarco` is not a Step-4 addition

`data.msmarco` (`name` + `subset`) is **byte-identical on `main`, `fast-grass` and
`baseline`**. Only `training.ance_msmarco` is new. As written, `AC-SURFACE-01` raises
`"Step-4 blocks unexpectedly existed before consolidation"` unconditionally, and its
byte-removal check can never reproduce the base config.

**Amendment:** the expected additive config is the single `training.ance_msmarco` block —
`ranges` drops `("data", "msmarco")`, and the `base_cfg` pre-existence guard and the
`trimmed` deletion loop follow the same reduction. Step 4 therefore adds **only**
`training.ance_msmarco`.

---

## Acceptance ledger

All rows NOT RUNNABLE YET (gated at Step 3 or later).

| row | gate | state |
|---|---|---|
| AC-SURFACE-01 | Step 6 | **PASS** (post-Step-6 re-run) |
| AC-COMP-01 (preprocessor) | Step 3 | **PASS** (re-run after Step 6) |
| AC-COMP-02 (in-batch) | Step 3 | **PASS** (re-run after Step 6) |
| AC-COMP-03 (cross-batch) | Step 3 | **PASS** (re-run after Step 6) |
| AC-COMP-04 (ANCE BRIGHT) | Step 3 | **PASS** (re-run after Step 6) |
| AC-COMP-05 (sync GRASS) | Step 3 | **PASS** (re-run after Step 6) |
| AC-COMP-06 (async GRASS) | Step 3 | **PASS** (re-run after Step 6) |
| AC-COMP-07 (sequential Fast-GRASS) | Step 3 | **PASS** (re-run after Step 6) |
| AC-COMP-08 (ANCE MS MARCO) | Step 5 | **PASS** `STEP5_LANDED` (re-run after Step 6) |
| AC-TEST-01 | Step 6 | **PASS** — 13/13 `CPU_SUITE_OK` |
| AC-INV-06 | Step 6 | **PASS** — 11 files checked |

**All eligible rows executed and passing.** CPU acceptance proves consolidation mechanics
only — not retrieval quality, not GPU training correctness.

---

## Cleanup ideas — collected, NOT acted on

0. **See "Pre-existing defects" above — P1 is the one that actually breaks a pipeline.**
1. `prepare_msmarco_train_data()` (`src/data/preprocessor.py` ~line 289) calls
   `random.shuffle(indices)` unseeded despite `seed: 42` — the 83,030-row MS MARCO mixture
   slice is not reproducible.
2. `ASYNC_FG_RUN_TESTS` cannot be disabled: the launcher does `${ASYNC_FG_RUN_TESTS:-1}` then
   tests `-n`, so `0` is still truthy and the `SKIPPED` branch is dead code.
3. `async-grass-v2` carries ~2500 lines of committed `logs_cluster/eval_*` and a 629KB
   `sample_efficient_method.pdf`.
4. `scripts/train_crossbatch.py` has no CLI surface at all (no `argparse`).
5. Loose repo-root artifacts on `fast-grass`: three `.html`/`.png` miner diagrams, `bib.tex`,
   `logs_cluster/fg_32k_logs.zip`, two `logs_cluster/fast_grass_*` dirs — deliberately not
   committed at Step 2; decide whether they belong in the repo at all.
