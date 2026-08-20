# Consolidation status

Resume anchor for the branch consolidation to `main`. A fresh session must be able to
continue from this file alone — no recon re-run. Updated **before** moving to the next step.

Authoritative inputs: `CONSOLIDATION_PROMPT.md` (procedure), `ACCEPTANCE_CRITERIA.md`
(checks), `CLAUDE.md` (repo map). Plan: `~/.claude/plans/desired-final-state-shiny-boot.md`.

---

## Next command

```bash
# Step 1 — backup tags (nothing destructive has happened yet)
git tag archive/baseline baseline
git tag archive/good-grass good-grass
git tag archive/async-grass-v2 async-grass-v2
git tag archive/main-pre-consolidation main
git push origin archive/baseline archive/good-grass archive/async-grass-v2 archive/main-pre-consolidation
git ls-remote --tags origin 'archive/*'   # verify all four on the remote
```

---

## Steps

| # | Step | State |
|---|---|---|
| 0 | preprocessor baseline fixture | **DONE** (`86c74f3`) |
| 1 | backup tags (4), pushed + verified | pending |
| 2 | commit untracked docs on `fast-grass` | pending |
| 3 | fast-forward `main` → `fast-grass`; tag `archive/main-post-promotion`; apply AC-SURFACE-01 amendment | pending |
| 4 | MS MARCO additive files + `training.ance_msmarco` | pending |
| 5 | **gated** merge: `train_ance.py` helpers + preprocessor methods/`setup_mode` | pending |
| 6 | startup config logging in the 6 entry points | pending |
| 7 | doc reference audit (report only) | pending |
| 8 | deletion proposal — **stop for approval** | pending |
| — | `GPU_CHECKLIST.md` deliverable | pending |

## Branch state

| branch | tip | merged into `main`? | safe to delete? |
|---|---|---|---|
| `fast-grass` | `86c74f3` | not yet (Step 3) | **NO** |
| `main` | `b633376` | — | **NO** (kept) |
| `new-grass` | `b633376` | same commit as `main` | not yet — no approval, Step 8 |
| `sequential-grass` | `8669117` | ancestor of `fast-grass` | not yet — no approval, Step 8 |
| `baseline` | `56a9e15` | **NO** — 6 unique commits | **NO** — no tag pushed yet |
| `good-grass` | `201c7c1` | **NO** — 1 unique commit, never merged by design | **NO** — no tag pushed yet |
| `async-grass-v2` | `46e548e` | **NO** — 7 unique commits, never merged by design | **NO** — no tag pushed yet |

No archive tags exist yet. **Nothing may be deleted.**

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
| AC-COMP-01…07 | Step 3 | not runnable yet |
| AC-COMP-08 | Step 5 | not runnable yet |
| AC-SURFACE-01 | Step 6 | not runnable yet |
| AC-TEST-01 | Step 6 | not runnable yet |
| AC-INV-06 | Step 6 | not runnable yet |

Every `AC-COMP-*` row is re-run after Step 6.

---

## Cleanup ideas — collected, NOT acted on

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
