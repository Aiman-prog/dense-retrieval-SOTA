# Consolidation status

Resume anchor for the branch consolidation to `main`. A fresh session must be able to
continue from this file alone — no recon re-run. Updated **before** moving to the next step.

Authoritative inputs: `CONSOLIDATION_PROMPT.md` (procedure), `ACCEPTANCE_CRITERIA.md`
(checks), `CLAUDE.md` (repo map). Plan: `~/.claude/plans/desired-final-state-shiny-boot.md`.

---

## Next command

```
CONSOLIDATION COMPLETE, plus two authorised post-consolidation fixes (P1 and the
in-batch wall clock). `main` is the only branch. All work is on `main` or behind a
pushed `archive/*` tag. `archive/consolidation-verified` marks the exact commit the
acceptance ledger was verified against.
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
| 7 | doc reference audit (report only) | **DONE** |
| 8 | branch deletion — approved and **EXECUTED** |  **DONE** |
| — | `GPU_CHECKLIST.md` deliverable | **DONE** |

## Branch state

**`main` is the only branch, locally and on `origin`.** All six others were deleted at Step 8
with explicit approval, after the pre-deletion gate passed.

| former branch | tip | disposition | recover with |
|---|---|---|---|
| `main` | `0b5463a`+ | **kept — the only branch** | — |
| `new-grass` | `b633376` | deleted; was the same commit as old `main` | `git checkout -b new-grass archive/main-pre-consolidation` |
| `sequential-grass` | `8669117` | deleted; ancestor of `main` | already in `main`'s history |
| `fast-grass` | `d0571e4` | deleted; ancestor of `main` | `git checkout -b fast-grass archive/main-post-promotion` |
| `baseline` | `56a9e15` | deleted; **6 commits never merged** | `git checkout -b baseline archive/baseline` |
| `good-grass` | `201c7c1` | deleted; **1 commit never merged** | `git checkout -b good-grass archive/good-grass` |
| `async-grass-v2` | `46e548e` | deleted; **7 commits never merged** | `git checkout -b async-grass-v2 archive/async-grass-v2` |

All five `archive/*` tags are pushed and **auto-fetch on clone**. Recovery was verified from a
fresh clone *after* deletion: each rebuilt branch reproduced its exact commit and tree SHA.

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

> **Superseded 2026-08-21 by the cleanup pass.** The list above is a record of what was true
> at Step 2 and is left unedited. Since then those artifacts were committed and then
> reorganised: the two `.html` files are `docs/assets/source/{fast-grass-miner,async-cached-mcdp-miner}.html`,
> `new-grass-architecture.png` is `docs/assets/fast-grass-architecture.png`, and the raw run
> logs are under `analysis/runs/`. `Fast-Grass Miner (standalone)-3.png` (a duplicate crop),
> `fg_logs.tgz` and `fg_32k_logs.zip` (byte-identical duplicates of the unpacked logs) were
> deleted, as was `logs_cluster/` once empty. `.claude/skills/` was deleted after salvaging
> **D2**. Recover any of them from history; nothing was lost.

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

## Step 7 — doc reference audit — DONE (report only, nothing fixed)

Scope: `DELFTBLUE_SETUP.md`, `README.md`, `fast_grass_implementation_details.md`,
`fast_grass_negative_cache_architecture.md`, `async_fast_grass_architecture.md`,
`async_fast_grass_implementation_details.md`, `lambda_pilot.md`,
`lambda_pilot_experiment_summary.md`. Every backticked reference outside fenced code blocks
was resolved against `main`. **Conceptual accuracy of the prose was not assessed.**

**324 references checked: 34 file paths, 22 CLI flags, 268 module/class/field names.**

| category | result |
|---|---|
| repo file paths | **0 missing, 0 moved** — every `scripts/…`, `src/…`, `config/…` path resolves |
| CLI flags | **0 missing** — every `--flag` appears in a committed `.py` or `.sh` |
| names | **3 stale** (below) |
| runtime artifacts | 18 references to files produced on `/scratch` (`cache_state_N.pt`, `mining_meta_N.json`, `optimizer.pt`, `async_run_summary.json`, …) plus two Tevatron files (`collator.py`, `dense.py`). Correctly absent from the repo, and every one is named somewhere in code. Not findings. |

### The three stale names — NOT fixed

| where | reference | reality on `main` |
|---|---|---|
| `fast_grass_implementation_details.md:116` | `selection_history` | no such attribute. `NegativeCache` tracks per-slot **`utility_ema`** (`src/utils/negative_cache.py:190`), which is what drives replacement and `R` admission |
| `fast_grass_negative_cache_architecture.md:243` | `selection_history` | same; also appears unbackticked at lines 240 and 388 |
| `async_fast_grass_architecture.md:526` | `active_round_no` | the trainer emits **`round_no`** (`scripts/run_async_fast_grass_train.py:199,314`). `active_round` exists but only as a local variable, never as a log field |

Both are cosmetic doc drift in field naming, not architectural error: the mechanisms described
are real, only the identifiers are wrong. Anyone grepping a log or a cache attribute for the
documented name finds nothing, which is the practical cost.

Audit script: `<scratchpad>/doc_audit.py` — deliberately **not** committed; it is a one-off
report generator, not part of the pipeline.

---

## Step 8 — branch deletion — APPROVED AND EXECUTED

### Pre-deletion verification (all passed before anything was deleted)

1. **Commit coverage** — for all seven branches, commits not reachable from `main` + the five
   tags: **0**. `ALL_COMMITS_COVERED`.
2. **Pipelines** — all seven components plus the preprocessor, lambda-pilot harness, shared
   `src/` and config present on `main`: **33/33 files**. `ALL_PIPELINES_PRESENT`.
3. **Docs** — all 10 architecture/design/setup docs present on `main`.
4. **Baseline contribution** — the three MS MARCO scripts byte-identical to `baseline`, the
   three preprocessor methods present, `training.ance_msmarco` present.
   `BASELINE_MSMARCO_FULLY_LANDED`.
5. **File-level coverage** — every file on every branch accounted for:
   - `new-grass`, `sequential-grass`, `fast-grass`: **0 files** not on `main`;
   - `baseline`: 1 (`scripts/train_grass.py`, superseded by `scripts/run_grass.py`);
   - `good-grass`: 16 (bandit/seq GRASS generation, never merged by design);
   - `async-grass-v2`: 15 (ANN-refresh async architecture, never merged by design).
6. **Pre-deletion gate** — every branch tip `==` its `origin` tip (no unpushed work), all five
   tags on the remote and matching, no tracked working-tree changes.
   `PRE_DELETION_GATE_PASSED`.

### Executed

```bash
git branch -d new-grass sequential-grass fast-grass     # -d: refused if not merged
git branch -D baseline good-grass async-grass-v2        # -D: unmerged, tag-protected
git push origin --delete new-grass sequential-grass fast-grass baseline good-grass async-grass-v2
```

### Post-deletion recovery proof

A fresh `git clone` of the remote now shows **one branch** (`origin/main`) and **five tags**,
all fetched automatically. All three never-merged branches were rebuilt from their tags and
matched exactly:

```
archive/baseline       -> 56a9e15a9  MATCHES  (6 unique commits recovered)
archive/good-grass     -> 201c7c140  MATCHES  (1 unique commit  recovered)
archive/async-grass-v2 -> 46e548e88  MATCHES  (7 unique commits recovered)
```

Spot-checked files that exist nowhere on `main` — `src/utils/bandit.py`,
`scripts/run_grass_seq_bandit.py`, `scripts/train_grass_async_v2.py`,
`scripts/run_grass_case_lite.py` — all recovered from their tags.

**Caveat, recorded deliberately:** recoverable is not the same as visible. `good-grass` and
`async-grass-v2` no longer appear in `git branch -a`, and their work is on `main` nowhere at
all. The tags are named in `CLAUDE.md`'s branch header and in this file.

---

## Pre-existing defects — NOT introduced by consolidation, NOT fixed here

Recorded so they are not rediscovered from scratch. Each says what breaks and why. Fixing them
is out of scope under rule 1; they need a separate, explicitly authorised task.

**P1-P5 are FIXED.** **P6, P7 and D1 are open** (P7 is docs-only and now recorded; no code
change is wanted). P2-P6 were found while auditing `GPU_CHECKLIST.md`
against the launcher bodies; each entry below records what it was and how it was resolved.

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

**FIXED after consolidation** (`archive/consolidation-verified` marks the last pre-fix commit).
`"temp_ance_msmarco": base / "temp_ance_msmarco_workdir"` added to `path_map`; verified that
`get_path` now resolves it and `temp_workdir / "ann_data"` no longer raises. Still open, and
deliberately not done: `get_path` returns `None` for **any** unknown key — a grep confirmed no
caller depends on that, so making it raise is safe but is a separate change.

---

## Post-consolidation fixes (authorised separately, after Step 8)

`archive/consolidation-verified` = `8b50a2c`, the commit the whole acceptance ledger was
verified against. Pushed, so `AC-SURFACE-01` stays reproducible against
`archive/main-post-promotion..archive/consolidation-verified`.

| fix | file | why |
|---|---|---|
| **P1** — add `temp_ance_msmarco` to `path_map` | `src/utils/helpers.py` | `--recipe ance_msmarco` raised `TypeError` at `train_ance.py:161` seconds into the job. Verified fixed. |
| restore in-batch wall clock `14:00:00` → `24:00:00` | `scripts/run_inbatch_singularity.sh` | the value was a temporary OOM-smoke setting; the file's own comment said to restore it |

**Amendment A4 — `AC-SURFACE-01` passes again on `main`.** The row asserts that no pre-existing
launcher changed by a byte, which the authorised in-batch edit broke. Reverting the edit was
not an option (the 14 h value kills a real run), so the criterion now allowlists that one file
**narrowly**: it must differ from `BASE` by exactly one line, that line must be the
`#SBATCH --time=` directive, and its new value must be exactly `#SBATCH --time=24:00:00`.

Mutation-tested — the allowlist is not a loophole. Unmutated `main` passes; all five of these
are rejected:

| mutation | rejection |
|---|---|
| second line changed in the allowlisted launcher | `changed 2 lines, expected 1` |
| `--time` restored to a different value | `change is not the permitted one` |
| a **different** pre-existing launcher edited | `pre-existing launcher changed: run_grass_singularity.sh` |
| a line inserted into the allowlisted launcher | `changed line count` |
| a pre-existing config value changed | `config diff is not exactly the additive Step-4 block` |

`archive/consolidation-verified` still pins the pre-edit state.

Re-verified after both fixes: Step-0 preprocessor hashes unchanged (`STEP0_REGRESSION_OK`),
and 9/9 CPU suites green (`helpers.py` is imported by all of them).

### P2 — `logs/` must exist before any `sbatch`; nothing creates it in time

**Symptom.** Job fails immediately with no output file, so there is nothing to read.

**Why.** Every launcher sets `--output=logs/<name>_%j.out`, and SLURM opens that file **before**
executing the script body. `logs/` is in `.gitignore:54` and nothing under it is tracked, so a
fresh clone lacks it. The `mkdir -p logs` present in `run_grass`, `run_fast_grass`,
`run_async_fast_grass` and `run_evaluate` runs far too late to matter, and `run_inbatch`,
`run_crossbatch`, `run_ance`, `run_ance_msmarco` and `eval_msmarco` do not have it at all.

**Fix when authorised.** Track a `logs/.gitkeep`. Editing the launchers cannot fix it —
anything inside the script already runs after SLURM has tried to open the file.

**Workaround, in `GPU_CHECKLIST.md`:** `mkdir -p /home/$USER/dense-retrieval-SOTA/logs` once.

**FIXED.** `.gitignore` now reads `logs/*` + `!logs/.gitkeep`, and `logs/.gitkeep` is tracked, so a
fresh clone has the directory. Verified: `git add logs/.gitkeep` succeeds while a real
`logs/grass_99999.out` is still ignored by `git add -A logs/`.

### P3 — `run_crossbatch_singularity.sh` always exits 0

**Symptom.** `sacct` reports `COMPLETED` even when `torchrun` dies.

**Why.** The script's last statement is `echo "Job Completed"`, so the exit status is `echo`'s.
Every other training launcher ends in `exit $EXIT_CODE`.

**Fix when authorised.** Capture `EXIT_CODE=$?` after the `singularity exec` and `exit` it, as
the sibling launchers do.

**FIXED.** Captures `EXIT_CODE=$?` after the `singularity exec` and ends in `exit $EXIT_CODE`,
matching its four sibling launchers. The original `echo "Job Completed"` line is retained.

### P4 — `eval_msmarco_singularity.sh` always exits 0, and can evaluate nothing

**Symptom.** Job reports success; the MRR number is absent or meaningless.

**Why.** Two independent problems: the script ends in `echo "Done: $?"`, so its status is
`echo`'s; and `MODEL=$(… get_last_checkpoint('$MODEL_DIR'))` prints the string `None` when the
model directory is absent, after which it runs `eval_msmarco.py --model_path None`.

**Fix when authorised.** Propagate the exit code, and fail fast when `MODEL` is empty or
`None`.

**Detection meanwhile:** `head -1` of the log — `Evaluating checkpoint: None`.

**FIXED.** Propagates the exit code, and exits 2 with a clear message when `get_last_checkpoint`
yields an empty string or `None` instead of evaluating `--model_path None`.

### P5 — `--debug` is unreachable from the GRASS launchers

**Symptom.** No cheap smoke run for experiments 4 and 5.

**Why.** `run_grass.py:455` and `run_fast_grass.py:655` both define
`--debug` (512-item mixture), but `run_grass_singularity.sh` and
`run_fast_grass_singularity.sh` contain **zero** occurrences of `debug` — the `${VAR:+--flag}`
pass-throughs cover the other knobs only. `run_async_fast_grass_singularity.sh` does wire
`ASYNC_FG_DEBUG`.

**Fix when authorised.** Add `${GRASS_DEBUG:+--debug}` / `${FAST_GRASS_DEBUG:+--debug}`,
matching the existing pattern.

**Workaround, in `GPU_CHECKLIST.md`:** an interactive `srun` + `singularity exec` invocation.

**FIXED.** Both launchers now pass `${GRASS_DEBUG:+--debug}` / `${FAST_GRASS_DEBUG:+--debug}`,
matching the existing `${VAR:+--flag}` pattern, and document the knob. Verified: with the knob
unset the expansion is empty and the resulting command line is byte-identical to before.

### P6 — the MS MARCO track needs the network, but its launcher forces offline mode

**Symptom.** `--recipe ance_msmarco` cannot build its inputs; setup fails at the first
`load_dataset`.

**Why.** `run_ance_msmarco_singularity.sh` exports `HF_HUB_OFFLINE=1` and
`TRANSFORMERS_OFFLINE=1`, while `prepare_msmarco_full_corpus`, `prepare_msmarco_tevatron_train`
and `prepare_msmarco_dev` call `load_dataset()` on `Tevatron/msmarco-passage-corpus` and
`Tevatron/msmarco-passage` — the latter two with `streaming=True`, which cannot work offline
at all. Independently, `msmarco_dev_qrels.txt` exists in neither dataset and must be fetched
from anserini-tools.

**Scope.** Pre-existing on `baseline`; consolidation only made the recipe reachable. The
offline exports are correct for the six BRIGHT experiments, whose data is already local.

**Fix when authorised.** Either pre-generate the four processed MS MARCO files on a login node
(`run_setup` skips what exists), or make the offline exports conditional for this recipe.

**Consequence today.** Experiment 7 remains **blocked**, now on data rather than on P1.
`GPU_CHECKLIST.md` §7 carries the login-node prep.

### P7 — the real environment is unversioned `~/.local`, not the container

**Symptom.** Nothing fails. Every doc in the repo describes an environment that does not
exist, and the one that does exist is reconstructible only by hand.

**Why.** `pytorch_2.1.sif` supplies CUDA and a torch 2.1.0 that **nothing imports**, and has
**no `transformers` at all** — `PYTHONNOUSERSITE=1` inside the container gives
`ModuleNotFoundError`. The entire ML stack resolves from
`~/.local/lib/python3.10/site-packages`. Measured on the cluster 2026-08-20:

| package | installed | repo claims |
|---|---|---|
| torch | **2.10.0+cu128** | 2.1.0 + CUDA 11.8 (`README.md`, `setup.sh`, `requirements-hpc.txt`) |
| transformers | 4.40.2 | 4.36.0 (`DELFTBLUE_SETUP.md` §2 rationale) |
| accelerate / peft / datasets / safetensors / faiss-gpu / numpy | exactly the `requirements-hpc.txt` pins | — |
| torchvision / torchaudio | 0.16.0 / 2.1.0 — **2.1-era ABI, stale under torch 2.10** | — |

`torch-2.10.0.dist-info` has no `REQUESTED` marker, so torch was upgraded **as a dependency
of some other `pip install --user`**, not deliberately. Directory mtime: **2026-02-22 23:28**.

**Scope.** Pre-existing and entirely outside consolidation — no launcher, entry point or
config is involved. Affects all seven pipelines equally.

**Assessed impact: low, and empirically closed.**
- All six entry points import cleanly on this stack, `train_crossbatch` included — that is
  the strict test, since it pulls transformers `Trainer` → `accelerate 0.30.1` → `GradCacheTrainer`
  → `grad_cache`.
- Every model under `models/` dates **2026-04-05 → 2026-08-20**, i.e. after the 02-22 upgrade.
  The sole exception is `inbatch_mixed_bge_m3.OLD_baseline` (02-22 17:21, six hours before the
  torch install), which was retired at the cutover. So ANCE 0.1683, the sequential λ ablation
  and the λ pilot all ran on **one consistent stack** — there is no cross-torch comparability
  caveat for the write-up.
- torchvision/torchaudio are ABI-stale but **unimported** (grepped: zero references repo-wide).
  Inert today; a confusing crash the moment anything imports them, including a transformers
  vision code path.

**Residual risk.** `accelerate==0.30.1` is pinned as "0.31+ uses PyTorch 2.2+ pytree APIs" and
now runs under torch 2.10 — the opposite of the intended direction. It works; if the Tevatron
pipelines ever misbehave around autocast, pytree or FSDP, look here first. GRASS/Fast-GRASS
use raw `AutoModel` and are largely insulated.

**Fix applied (docs only, no code):** `README.md`, `DELFTBLUE_SETUP.md`, `requirements.txt`,
`requirements-hpc.txt` and `setup.sh` now state the actual versions and warn against
"fixing" the pins upward. `docs/DELFTBLUE_ENVIRONMENT.md` records the resolved environment.

**Deliberately NOT done.** Not downgrading torch (would break a working stack to satisfy a
comment). Not renaming `pytorch_2.1.sif` (17 launchers reference it; the file on disk really
is called that). Not setting `PYTHONNOUSERSITE` anywhere — it breaks every pipeline.

**Irreplaceable state.** The three Tevatron source patches in `DELFTBLUE_SETUP.md` §2 exist
only in `~/.local` and in `/scratch/$USER/tevatron_patched_20260820.tgz` (93K, made
2026-08-20). Re-verified as applied on that date; `from tevatron.retriever.modeling import
DenseModel` is the check.

### D1 — `bug_fixes.md` is gitignored, so the MS MARCO runbook is not on `main`

`bug_fixes.md` holds the `load_dataset` invocations, the `msmarco_dev_qrels.txt` `wget` and
the `streaming=True` / `split='validation'` notes. It is **explicitly** ignored at
`.gitignore:85`, alongside `CLAUDE.md` — a deliberate local-notes choice, not an oversight,
so it was **not** committed during this pass. `main` is therefore not self-contained for
experiment 7; `GPU_CHECKLIST.md` §7 inlines the essential steps to compensate.

**Decision needed:** un-ignore and commit it, or leave it local. One line of `.gitignore`
either way.

Note it also carries one stale claim: it says `per_device_eval_batch_size` "needs to be
256" and is "NOT yet applied" — `config.yaml` already sets 256 for `ance_msmarco`.

### D2 — `grass_twoset_feasibility.py` exists only on DelftBlue (salvaged from a deleted skill file)

Recorded during the Aug 2026 cleanup, when `.claude/skills/*/SKILL.md` were deleted. This was
the only place the fact was written down, and it cannot be discovered by reading `main`.

`scripts/grass_twoset_feasibility.py` was **never committed to git**. It exists on DelftBlue at
`~/dense-retrieval-SOTA/scripts/grass_twoset_feasibility.py` (~26 K). A local compiled copy
survived at `scripts/__pycache__/grass_twoset_feasibility.cpython-313.pyc` until the same
cleanup pass removed `__pycache__`.

Its four functions (`load_train_queries`, `encode_queries`, `build_candidate_matrix`,
`cached_grass_sampler`) plus a `_topk_neighbors` helper were recovered by disassembling that
`.pyc` and **inlined into `scripts/grass_negcache_feasibility.py`**, which is committed and
self-contained. The reconstruction passed synthetic checks and a real 7/7 cluster run.

Its launcher, `scripts/run_twoset_feasibility_singularity.sh`, was deleted in the cleanup pass
(amendment **A7**) because its target does not exist on `main` and the launcher could never run.

**Open, low priority:** if byte-faithfulness of the inlined functions ever matters, diff them
against the cluster original. The recovery already passes synthetic and real runs, so this is
belt-and-braces only.

---

## Post-consolidation fix pass 2 — launcher defects P2–P5

Commit `f07b188`, on top of `f633d8d`. Four launchers and `.gitignore` changed; **no Python
touched**. `eval_msmarco_singularity.sh` is a Step-4 *new* file, so `AC-SURFACE-01` never
byte-checks it; the three pre-existing launchers are pinned by amendment **A5**.

**Amendment A5** replaces A4's one-line allowlist with an **exact unified-diff pin per file**.
Strictly stronger: the permitted change is now the literal diff, so any extra, altered or
missing line fails, as does any edit to a launcher not on the list.

Mutation-tested — unmutated `main` passes, all six of these are rejected:

| mutation | rejection |
|---|---|
| extra edit inside an allowlisted launcher | `diff is not the permitted one` |
| `exit $EXIT_CODE` neutered to `exit 0` | `diff is not the permitted one` |
| extra flag smuggled into the debug knob | `diff is not the permitted one` |
| a **non**-allowlisted launcher edited | `pre-existing launcher changed: run_ance_singularity.sh` |
| a pinned added line removed | `diff is not the permitted one` |
| pre-existing config value changed | `config diff is not exactly the additive Step-4 block` |

⚠️ The first two attempts at this mutation suite reported every mutation as passing. Both
times the harness was at fault, not the criterion — once because mutations were committed to a
branch the script did not read, once because `sed` patterns silently matched nothing. The
suite now asserts the pattern exists and that the commit actually changed something before
trusting a result. **A mutation test that never fails is reporting on your harness.**

Re-verified after the pass: `STEP0_REGRESSION_OK`, `AC-COMP-02…08` pass, `AC-SURFACE-01`
prints `SURFACE_ALLOWLIST_OK`.

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
