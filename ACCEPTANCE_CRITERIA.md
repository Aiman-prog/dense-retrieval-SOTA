# Consolidation acceptance criteria

These criteria assess consolidation mechanics only. They do not assess training correctness or the conceptual accuracy of documentation. All checks are CPU-only. Do not substitute `--preflight` for any row: it loads the full 655k-document corpus and is intentionally excluded.

“Runnable after step N” is an availability state, not a result. Before that step is complete, the row is **NOT RUNNABLE YET**, not failed. Once runnable, exit 0 and the stated success output mean **PASS**; exit 1 or a missing success output mean **FAIL**.

**Mandatory re-run.** Step 6 edits `scripts/train_inbatch.py`, `scripts/train_crossbatch.py`, `scripts/train_ance.py`, `scripts/run_grass.py`, `scripts/run_fast_grass.py`, and `scripts/train_async_fast_grass.py` to add startup logging. Every `AC-COMP-*` row therefore becomes runnable at the step listed but **must be re-run after Step 6**, exactly as Step 0's preprocessor diff is re-run after every step. A first-run PASS is not final: a logging edit that breaks an import is caught only by the re-run. `AC-SURFACE-01`, `AC-TEST-01` and `AC-INV-06` are already gated at Step 6 and need no re-run. Component import commands may also exit 2 with `IMPORT_ENVIRONMENT_OUT_OF_SCOPE`; that is neither pass nor fail and must include an exception narrowly attributable to unavailable CUDA, a missing `/scratch/...` path, or an unavailable Hugging Face cache in forced-offline mode.

The import harnesses set `CUDA_VISIBLE_DEVICES` empty and force Transformers/Hugging Face offline, then use `importlib` in a fresh subprocess without calling `main()`, `run_setup()`, or any training function. Therefore ordinary absence of CUDA, `/scratch`, and cached model files is not exercised. A traceback not matching those narrow environmental signatures exits 1 as a real import error; missing Python packages, syntax errors, bad imports, and application exceptions are consolidation failures.

## AC-SURFACE-01

> **Amendment A6 (baseline hardening pass).** `train_inbatch.py` and
> `train_crossbatch.py` gain `--resume` / `--overwrite` and read `WORLD_SIZE`. Both are
> load-bearing, not conveniences: Tevatron's driver resumes from any `checkpoint-*` in
> the output dir unconditionally, so "fresh unless asked" needs a flag to opt out of,
> and cross-batch launched without `torchrun` silently halves its negative pool, which
> only `WORLD_SIZE` can detect before training starts. The row now allowlists **exactly**
> these additions on **exactly** these two files: any third flag, any other env key, any
> new `get_path`/`get_training_context` key, any removal, and any change to the other
> four entry points still fail. Strictly stronger than a blanket exemption, and pinned
> the way A5 pins launcher diffs.
>
> A6 also extends the pinned diffs of `run_inbatch_singularity.sh` and
> `run_crossbatch_singularity.sh` with `${INBATCH_RESUME:+--resume}`-style knobs, following
> the `${GRASS_DEBUG:+--debug}` pattern the P5 fix already established. Every GPU run in
> this project goes through a launcher, so a flag reachable only by direct `python`
> invocation is not reachable at all. **With both knobs unset the emitted command line is
> byte-identical to before**, so the default `sbatch` path is unchanged.

> **Amended during consolidation (approved at Gate A; evidence in `CONSOLIDATION_STATUS.md`).**
> **A1** — `BASE` is `archive/main-post-promotion` (`main` immediately after the Step-3
> fast-forward), not `archive/main-pre-consolidation`. The latter is `main` @ `b633376`, which
> predates all of `fast-grass`: the fast-forward alone adds 8 launchers, modifies 11 more, and
> two entry points do not exist there, so four assertions fired before Step 4 was reached.
> `archive/main-pre-consolidation` still points at `b633376` and remains the Step-3 rollback anchor.
> **A2** — `data.msmarco` is byte-identical on `main`, `fast-grass` and `baseline`, so it is not
> a Step-4 addition; the sole additive block is `training.ance_msmarco`.
> **A4 (post-consolidation)** — `scripts/run_inbatch_singularity.sh` was edited after Step 8 with
> explicit authorisation, restoring `--time` from a temporary `14:00:00` OOM-smoke value to
> `24:00:00` as the file's own comment instructed. It is allowlisted, but narrowly: the file must
> differ from `BASE` by **exactly one line**, that line must be the `#SBATCH --time=` directive,
> and its new value must be exactly `#SBATCH --time=24:00:00`. Any other edit to this launcher,
> or a second changed line, still fails. `archive/consolidation-verified` pins the pre-edit state.
> **A5 (post-consolidation)** — three more pre-existing launchers were edited with explicit
> authorisation, fixing defects P3 and P5: `run_crossbatch_singularity.sh` now propagates its
> exit code (it ended in `echo`, so `sacct` reported COMPLETED on a failed `torchrun`), and
> `run_grass_singularity.sh` / `run_fast_grass_singularity.sh` now pass `--debug` through
> (the flag existed on both entry points but no launcher wired it, leaving no smoke path;
> with the knob unset the expansion is empty and the command line is byte-identical).
> A4's one-line mechanism is replaced by an **exact unified-diff pin** per file, which is
> strictly stronger: the permitted change is now the literal diff, so any extra, altered or
> missing line fails, as does any edit to a launcher not listed.
> **A6 (post-consolidation)** — the CPU test suites moved from `scripts/` to `tests/`, so the
> two launchers that invoke them by path were edited: `run_async_fast_grass_singularity.sh`
> (6 lines, the pre-GPU correctness gate) and `run_fast_grass_timing_singularity.sh`
> (3 lines). Both diffs are **path-only** — `scripts/<suite>.py` → `tests/<suite>.py`, with no
> other token changed — and are pinned line-for-line by the same A5 mechanism. `AC-TEST-01`'s
> roster moved with them. The suites' own `Path(__file__).resolve().parent.parent` still
> resolves to the repo root from `tests/`, so no `sys.path` bootstrapping changed.

> **A7 (post-consolidation)** — `scripts/run_twoset_feasibility_singularity.sh` was deleted.
> It invoked `scripts/grass_twoset_feasibility.py`, which was **never committed**, so the
> launcher was unrunnable from a clone at every point in its history. This is the first
> allowlisted *removal*, so the row gained `ALLOWED_REMOVED_SH` alongside `ALLOWED_NEW_SH`;
> the set is exact, so deleting any other launcher still fails. Context: **D2** in
> `CONSOLIDATION_STATUS.md`.
>
> **A8 (post-consolidation)** — `scripts/run_negcache_feasibility_singularity.sh` was deleted
> alongside its only target, `scripts/grass_negcache_feasibility.py` (901 lines), which
> `scripts/dev/fast_grass_feasibility.py` had already superseded and which no code imported.
> `ALLOWED_REMOVED_SH` gains one entry; the set stays exact.

> **A9 (post-consolidation)** — three developer launchers moved to `scripts/dev/`
> (`run_fast_grass_feasibility`, `run_fast_grass_timing`, `run_async_fast_grass_probe`).
> A relocation is not exempt from review: `MOVED_SH` maps each old path to its new one **and**
> pins the exact unified diff between them, so the only permitted change is the path rewrite.
> Every pinned line is path-only. The timing launcher's A6 `tests/` lines moved into its
> `MOVED_SH` pin, since one pin must cover the whole file.

> **A10 (post-consolidation)** — every launcher moved to `scripts/launchers/`; `scripts/`
> root is now Python only. Shell files are never imported and each hardcodes
> `#SBATCH --chdir=<repo root>` with repo-relative paths, so a move needs no content edit.
> `ALLOWED_CHANGED_SH` is consequently **empty**: a pre-existing launcher that changed
> without moving now fails outright. `MOVED_SH` carries all 12 relocations, **4 of them
> with an empty diff list** — the strongest pin available, asserting the bytes are
> untouched. `scripts/run_eval_checkpoints_singularity.sh` joins `ALLOWED_REMOVED_SH`;
> its target duplicated `run_all_evals.py` and swallowed per-domain failures.

**Runnable after step:** 6

**Exact command:**

```bash
set -euo pipefail
git diff archive/main-post-promotion..main -- 'scripts/*.sh' 'config/config.yaml'
python - <<'PY'
import ast
import copy
import fnmatch
import re
import subprocess
import sys

BASE = "archive/main-post-promotion"
HEAD = "main"
ALLOWED_NEW_SH = {
    "scripts/launchers/eval_msmarco_singularity.sh",
    "scripts/launchers/run_ance_msmarco_singularity.sh",
}
# Amendment A7. The launcher's only target, scripts/grass_twoset_feasibility.py, was
# never committed, so this script could not run from a clone of `main` at any point in
# its history. Deleting it removes a dead file, not a capability. Recorded as D2 in
# CONSOLIDATION_STATUS.md, which also preserves where the module actually lives.
# Amendment A8. The negcache feasibility probe was superseded by
# scripts/dev/fast_grass_feasibility.py, which says so in its own docstring; the donor
# was imported by no code, gated by no test, and reachable only from this launcher.
ALLOWED_REMOVED_SH = {
    "scripts/run_twoset_feasibility_singularity.sh",
    "scripts/run_negcache_feasibility_singularity.sh",
    "scripts/run_eval_checkpoints_singularity.sh",
}
# Amendments A4 + A5. Pre-existing launchers were edited AFTER consolidation, each
# with explicit authorisation. Allowing them is not a loophole: every permitted
# change is pinned to its EXACT unified diff below, so any additional, altered or
# missing line in these files fails the row, and any edit to a launcher not listed
# here fails immediately.
#   A4  run_inbatch  -- restore --time from a temporary 14:00:00 OOM-smoke value
#   A5  crossbatch   -- propagate the exit code (the script ended in `echo`, so a
#                       failed torchrun was reported COMPLETED by sacct)
#   A5  grass/fast_grass -- wire ${*_DEBUG:+--debug}; the flag existed on the entry
#                       points but no launcher passed it through, so there was no
#                       smoke path. With the knob unset the expansion is empty and
#                       the command line is byte-identical to before.
# Amendment A10. Every pre-existing launcher moved to scripts/launchers/, so
# ALLOWED_CHANGED_SH is now empty: a launcher that changed WITHOUT moving fails.
ALLOWED_CHANGED_SH = {}
MOVED_SH = {
    "scripts/run_ance_singularity.sh": ("scripts/launchers/run_ance_singularity.sh", []),   # pure rename
    "scripts/run_async_fast_grass_probe_singularity.sh": (
        "scripts/dev/run_async_fast_grass_probe_singularity.sh", [
        "-    python -u scripts/async_fast_grass_quality_probe.py --synthetic",
        "+    python -u scripts/dev/async_fast_grass_quality_probe.py --synthetic",
        "-    python -u scripts/async_fast_grass_quality_probe.py --real \\",
        "+    python -u scripts/dev/async_fast_grass_quality_probe.py --real \\",
    ]),
    "scripts/run_async_fast_grass_singularity.sh": (
        "scripts/launchers/run_async_fast_grass_singularity.sh", [
        "-#   ASYNC_FG_LAMBDA=0   ASYNC_FG_SUFFIX=lam0   sbatch scripts/run_async_fast_grass_singularity.sh",
        "-#   ASYNC_FG_LAMBDA=0.5 ASYNC_FG_SUFFIX=lam05  sbatch scripts/run_async_fast_grass_singularity.sh",
        "+#   ASYNC_FG_LAMBDA=0   ASYNC_FG_SUFFIX=lam0   sbatch scripts/launchers/run_async_fast_grass_singularity.sh",
        "+#   ASYNC_FG_LAMBDA=0.5 ASYNC_FG_SUFFIX=lam05  sbatch scripts/launchers/run_async_fast_grass_singularity.sh",
        "-            python -u scripts/async_fast_grass_handoff_test.py",
        "-            python -u scripts/async_fast_grass_cache_semantics_test.py",
        "-            python -u scripts/async_fast_grass_persistence_test.py",
        "-            python -u scripts/async_fast_grass_pilot_test.py",
        "-            python -u scripts/async_fast_grass_integration_smoke.py",
        "-            python -u scripts/fast_grass_test.py",
        "+            python -u tests/async_fast_grass_handoff_test.py",
        "+            python -u tests/async_fast_grass_cache_semantics_test.py",
        "+            python -u tests/async_fast_grass_persistence_test.py",
        "+            python -u tests/async_fast_grass_pilot_test.py",
        "+            python -u tests/async_fast_grass_integration_smoke.py",
        "+            python -u tests/fast_grass_test.py",
    ]),
    # A8: JAVA_HOME moves to /home. Defect P9 -- the /scratch JDK was written during
    # the BeeGFS fault (P11) and its libjli.so is corrupt, so every BM25 job failed at
    # JVM start. The /home copy is verified working (java -version succeeds).
    "scripts/run_bm25_singularity.sh": (
        "scripts/launchers/run_bm25_singularity.sh", [
        '-export JAVA_HOME="/scratch/${USER}/.jdk21"',
        '+export JAVA_HOME="/home/${USER}/.jdk21"   # P9: the /scratch copy was written during the BeeGFS fault and is broken',
    ]),
    # A7: the two resource lines and the [alloc] echo. Job 15039 died host-side with a
    # DataLoader worker SIGBUS on an allocation of 32GB/8CPU for two ranks, where every
    # other GPU launcher here takes 125GB/16CPU for one. The echo prints what the
    # allocation actually granted, because the failing memory term could not be
    # identified after the fact (MaxRSS 10.4GB against a 32GB cap).
    "scripts/run_crossbatch_singularity.sh": (
        "scripts/launchers/run_crossbatch_singularity.sh", [
        '-#SBATCH --cpus-per-task=4          # 2 CPUs per GPU (for 4 workers)',
        '+#SBATCH --cpus-per-task=8           # 2 tasks x 8 = 16 CPUs, matching every other GPU launcher',
        '-#SBATCH --mem-per-gpu=16GB           # 16GB RAM per GPU',
        '+#SBATCH --mem-per-cpu=8000M         # 16 x 8000M = 125GB. --mem-per-gpu=16GB gave 2 ranks 32GB',
        '+                                    # total while in-batch gets 125GB for one; job 15039 died',
        '+                                    # host-side with a DataLoader worker SIGBUS.',
        '+# --- Experiment Knobs (override via env vars before sbatch) ---',
        '+# CROSSBATCH_RESUME=1     # continue a run whose manifest fingerprint matches',
        '+# CROSSBATCH_OVERWRITE=1  # discard an output dir built by a DIFFERENT config',
        '+# nproc_per_node MUST stay 2: train_crossbatch.py refuses any other world size,',
        '+# because a single process drops the all-gather and halves the negative pool.',
        '+',
        "+# What the allocation actually granted. 15039's SIGBUS could not be pinned to a",
        '+# specific memory term after the fact; print the limits so a recurrence is read, not guessed.',
        '+echo "[alloc] cgroup memory.max: $(cat /sys/fs/cgroup/memory.max 2>/dev/null \\',
        '+    || cat /sys/fs/cgroup/memory/memory.limit_in_bytes 2>/dev/null || echo unknown)"',
        '+echo "[alloc] /dev/shm: $(df -h /dev/shm | tail -1)"',
        '+',
        '-    torchrun --nproc_per_node=2 scripts/train_crossbatch.py',
        '+    torchrun --nproc_per_node=2 scripts/train_crossbatch.py \\',
        '+        ${CROSSBATCH_RESUME:+--resume} \\',
        '+        ${CROSSBATCH_OVERWRITE:+--overwrite}',
        '+',
        '+EXIT_CODE=$?',
        '+',
        '+if [ $EXIT_CODE -eq 0 ]; then',
        '+    echo "✅ Cross-batch training completed successfully"',
        '+else',
        '+    echo "❌ Cross-batch training failed with code $EXIT_CODE"',
        '+fi',
        '+',
        '+exit $EXIT_CODE',
    ]),
    "scripts/run_evaluate_singularity.sh": ("scripts/launchers/run_evaluate_singularity.sh", []),   # pure rename
    "scripts/run_fast_grass_feasibility_singularity.sh": (
        "scripts/dev/run_fast_grass_feasibility_singularity.sh", [
        "-    python -u scripts/fast_grass_feasibility.py \\",
        "+    python -u scripts/dev/fast_grass_feasibility.py \\",
    ]),
    "scripts/run_fast_grass_singularity.sh": (
        "scripts/launchers/run_fast_grass_singularity.sh", [
        "+# FAST_GRASS_DEBUG=1      # 512-item mixture smoke run",
        "-        ${FAST_GRASS_NO_EVAL:+--no_eval}",
        "+        ${FAST_GRASS_NO_EVAL:+--no_eval} \\",
        "+        ${FAST_GRASS_DEBUG:+--debug}",
    ]),
    "scripts/run_fast_grass_timing_singularity.sh": (
        "scripts/dev/run_fast_grass_timing_singularity.sh", [
        "-            python -u scripts/async_fast_grass_handoff_test.py",
        "-            python -u scripts/async_fast_grass_cache_semantics_test.py",
        "-            python -u scripts/fast_grass_test.py",
        "-            python -u scripts/fast_grass_mine_timing.py --synthetic",
        "-            python -u scripts/async_fast_grass_quality_probe.py --synthetic",
        "+            python -u tests/async_fast_grass_handoff_test.py",
        "+            python -u tests/async_fast_grass_cache_semantics_test.py",
        "+            python -u tests/fast_grass_test.py",
        "+            python -u scripts/dev/fast_grass_mine_timing.py --synthetic",
        "+            python -u scripts/dev/async_fast_grass_quality_probe.py --synthetic",
        "-        python -u scripts/fast_grass_train_timing.py \\",
        "+        python -u scripts/dev/fast_grass_train_timing.py \\",
        "-    python -u scripts/fast_grass_mine_timing.py \\",
        "+    python -u scripts/dev/fast_grass_mine_timing.py \\",
        "-    python -u scripts/async_fast_grass_speed_estimate.py \\",
        "+    python -u scripts/dev/async_fast_grass_speed_estimate.py \\",
        "-        python -u scripts/async_fast_grass_quality_probe.py --real \\",
        "+        python -u scripts/dev/async_fast_grass_quality_probe.py --real \\",
    ]),
    "scripts/run_grass_singularity.sh": (
        "scripts/launchers/run_grass_singularity.sh", [
        "+# GRASS_DEBUG=1           # 512-item mixture smoke run",
        "-        ${GRASS_LAMBDA:+--lambda_val $GRASS_LAMBDA}",
        "+        ${GRASS_LAMBDA:+--lambda_val $GRASS_LAMBDA} \\",
        "+        ${GRASS_DEBUG:+--debug}",
    ]),
    "scripts/run_inbatch_singularity.sh": (
        "scripts/launchers/run_inbatch_singularity.sh", [
        '-#SBATCH --time=14:00:00   # TEMP: OOM smoke test — passes step-1 mem peak then SLURM kills it. RESTORE to 24:00:00 for real run.',
        '+#SBATCH --time=24:00:00',
        '+# --- Experiment Knobs (override via env vars before sbatch) ---',
        '+# INBATCH_RESUME=1        # continue a run whose manifest fingerprint matches',
        '+# INBATCH_OVERWRITE=1     # discard an output dir built by a DIFFERENT config',
        '+# Default (both unset) starts FRESH: stale checkpoint-* are removed first, which is',
        '+# what stops Tevatron resuming them and reporting success after zero steps.',
        '+',
        '-    python -u scripts/train_inbatch.py',
        '+    python -u scripts/train_inbatch.py \\',
        '+        ${INBATCH_RESUME:+--resume} \\',
        '+        ${INBATCH_OVERWRITE:+--overwrite}',
    ]),
    "scripts/run_refresh_stale_index_singularity.sh": ("scripts/launchers/run_refresh_stale_index_singularity.sh", []),   # pure rename
}
ENTRY_POINTS = [
    "scripts/train_inbatch.py",
    "scripts/train_crossbatch.py",
    "scripts/train_ance.py",
    "scripts/run_grass.py",
    "scripts/run_fast_grass.py",
    "scripts/train_async_fast_grass.py",
]

# Amendment A6 -- the ONLY permitted surface additions, pinned per entry point.
# Anything not listed here still fails, as does any removal. See the amendment note
# above the criterion for why these two exist.
ALLOWED_SURFACE_ADDITIONS = {
    "scripts/train_inbatch.py":   {"cli": {"--resume", "--overwrite"},
                                   "env_keys": {("get", "WORLD_SIZE")}},
    # RANK: only rank 0 writes the checkpoint and the diagnostics log under DDP, so
    # only rank 0 may validate them.
    "scripts/train_crossbatch.py": {"cli": {"--resume", "--overwrite"},
                                    "env_keys": {("get", "WORLD_SIZE"),
                                                 ("get", "RANK")}},
}
SURFACE_FIELDS = ("cli", "recipes", "path_keys", "env_keys", "sys_path_ops",
                  "mains", "main_guards")

def git(*args, text=False):
    return subprocess.check_output(["git", *args], text=text)

def paths(rev, pattern):
    return {p for p in git("ls-tree", "-r", "--name-only", rev, text=True).splitlines()
            if fnmatch.fnmatch(p, pattern)}

base_sh = paths(BASE, "scripts/*.sh")
head_sh = paths(HEAD, "scripts/*.sh")
moved_new = {new for new, _ in MOVED_SH.values()}
if head_sh - base_sh != ALLOWED_NEW_SH | moved_new:
    raise AssertionError(f"unexpected added shell files: {sorted(head_sh - base_sh - ALLOWED_NEW_SH - moved_new)}")
if base_sh - head_sh != ALLOWED_REMOVED_SH | set(MOVED_SH):
    raise AssertionError(f"unexpected removed shell files: {sorted(base_sh - head_sh - ALLOWED_REMOVED_SH - set(MOVED_SH))}")
import difflib

def change_lines(before, after):
    """The exact +/- lines of the unified diff, in order, with no context."""
    return [line for line in difflib.unified_diff(
                before.splitlines(), after.splitlines(), lineterm="", n=0)
            if not line.startswith(("---", "+++", "@@"))]

for path in sorted(base_sh):
    if path in ALLOWED_REMOVED_SH:
        continue  # deleted at HEAD; `git show HEAD:<path>` would fail
    before = git("show", f"{BASE}:{path}", text=True)
    if path in MOVED_SH:
        # relocated: compare against its NEW path and pin that diff exactly
        new_path, expected = MOVED_SH[path]
        after = git("show", f"{HEAD}:{new_path}", text=True)
        actual = change_lines(before, after)
        if actual != expected:
            raise AssertionError(
                f"relocated launcher diff is not the permitted one: {path} -> {new_path}\n"
                f"  expected: {expected}\n  actual:   {actual}")
        continue
    after = git("show", f"{HEAD}:{path}", text=True)
    if before == after:
        continue
    if path not in ALLOWED_CHANGED_SH:
        raise AssertionError(f"pre-existing launcher changed: {path}")
    actual = change_lines(before, after)
    expected = ALLOWED_CHANGED_SH[path]
    if actual != expected:
        raise AssertionError(
            f"allowlisted launcher diff is not the permitted one: {path}\n"
            f"  expected: {expected}\n  actual:   {actual}")

base_text = git("show", f"{BASE}:config/config.yaml", text=True)
head_text = git("show", f"{HEAD}:config/config.yaml", text=True)

def block_range(lines, parent, child):
    parent_i = [i for i, line in enumerate(lines) if line == f"{parent}:\n"]
    if len(parent_i) != 1:
        raise AssertionError(f"expected one top-level {parent} block")
    start_parent = parent_i[0]
    end_parent = next((i for i in range(start_parent + 1, len(lines))
                       if lines[i].strip() and not lines[i].startswith((" ", "\t"))), len(lines))
    starts = [i for i in range(start_parent + 1, end_parent)
              if lines[i] == f"  {child}:\n"]
    if len(starts) != 1:
        raise AssertionError(f"expected one {parent}.{child} block")
    start = starts[0]
    end = next((i for i in range(start + 1, end_parent)
                if lines[i].strip() and len(lines[i]) - len(lines[i].lstrip(" ")) <= 2), end_parent)
    return start, end

head_lines = head_text.splitlines(keepends=True)
ranges = [block_range(head_lines, "training", "ance_msmarco")]

# Removing exactly the Step-4 block, plus at most one adjacent blank line, must
# reproduce the tagged config byte-for-byte. This makes every deletion,
# replacement, relocation, comment edit, or other added hunk fail.
def candidates(lines, ranges):
    choices = [[]]
    for start, end in ranges:
        variants = [(start, end)]
        if start > 0 and not lines[start - 1].strip():
            variants.append((start - 1, end))
        if end < len(lines) and not lines[end].strip():
            variants.append((start, end + 1))
        choices = [old + [new] for old in choices for new in variants]
    return choices

# Config drift that predates the baseline-hardening pass. The preprocessor
# hardening landed after archive/consolidation-verified and this row has been
# FAILING on main ever since -- recorded as defect P8 in CONSOLIDATION_STATUS.md,
# pinned here so it is visible rather than blessed. Do not add to this list.
PRE_EXISTING_CONFIG_DRIFT = [
    '-    examples_config: "Gemini-1.0_reason"',
    '+    examples_config: "examples"',
    '-    subset: "hq"',
    '-    train_file: "train_reasonir.jsonl"',
    '-    msmarco_samples: 83030   # 23.7% - Public data bridge + calibration (prevent forgetting)',
    '-    vl_samples: 149970       # 42.8% - ALL available clean VL (length generalization)',
    '-    hq_samples: 97000        # 27.7% - Nearly all HQ (reasoning-intensive queries)',
    '-    # Total: ~350k samples | Note: VL corrupted before index 95k, only 150k clean samples available',
    '+    msmarco_samples: 83030   # 25.2% - Public data bridge + calibration (prevent forgetting)',
    '+    vl_samples: 149970       # 45.4% - ALL available clean VL (length generalization)',
    '+    hq_samples: 97000        # 29.4% - Nearly all HQ (reasoning-intensive queries)',
    '+    vl_skip_first_n: 95000   # VL rows before this index are corrupted; skipped at generation',
    '+    # Requested total: 330,000. VL writes 149,963 (7 rows past the cutoff are unusable),',
    '+    # so the actual total is 329,993 -- the generators print requested vs written.',
]

# Amendment A6: config.yaml becomes the source of truth for values the two Tevatron
# trainers used to hard-code, and training.overwrite_output_dir goes because it is
# read by nothing and contradicts the fresh-start gate in helpers.prepare_output_dir.
A6_CONFIG_LINES = [
    '+  # Fallback for a recipe that declares no logging_steps of its own.',
    '-  overwrite_output_dir: true',
    '+    logging_steps: 100',
    '+    save_total_limit: 6',
    '+    save_fraction: 0.2        # checkpoint every 20% of total steps',
    '+    gradient_checkpointing: true   # THE memory fix at query_max_len 1024 + eager attention',
    # A7: cross-batch only, and pinned HERE because that is where it falls in the file.
    # 0 removes the DataLoader worker processes entirely, so the shm tensor hand-off and
    # the SIGCHLD path that surfaced job 15039's SIGBUS cannot occur. Every other recipe
    # keeps its worker count.
    '-    dataloader_num_workers: 4',
    '+    dataloader_num_workers: 0   # 0 = collate inline: no worker processes, so no shm hand-off and no worker SIGBUS (job 15039)',
    '+    save_steps: 100',
    '+    save_total_limit: 3',
    '+    gradient_checkpointing: true   # cross-batch omitted this and never reached a checkpoint',
]
EXPECTED_CONFIG_LINES = PRE_EXISTING_CONFIG_DRIFT + A6_CONFIG_LINES

residuals = ["".join(line for i, line in enumerate(head_lines)
                     if not any(start <= i < end for start, end in selected))
             for selected in candidates(head_lines, ranges)]
if not any(residual == base_text for residual in residuals):
    # Not byte-identical to BASE: the remainder must be EXACTLY the pinned lines, so
    # any other deletion, replacement, relocation or added hunk still fails.
    actual = min((change_lines(base_text, residual) for residual in residuals), key=len)
    if actual != EXPECTED_CONFIG_LINES:
        unexpected = [l for l in actual if l not in EXPECTED_CONFIG_LINES]
        missing = [l for l in EXPECTED_CONFIG_LINES if l not in actual]
        raise AssertionError(
            "config diff is not the permitted one (Step-4 block + pinned lines):\n"
            f"  unexpected: {unexpected}\n  missing: {missing}")

try:
    import yaml
except Exception as exc:
    raise AssertionError("PyYAML is required for the config semantic check") from exc
base_cfg = yaml.safe_load(base_text)
head_cfg = yaml.safe_load(head_text)
# data.msmarco is NOT a Step-4 addition: it is byte-identical on main, fast-grass
# and baseline, so it must already be present at BASE. Only training.ance_msmarco
# is new.
if "msmarco" not in base_cfg.get("data", {}):
    raise AssertionError("data.msmarco was expected to pre-exist at BASE")
if "ance_msmarco" in base_cfg.get("training", {}):
    raise AssertionError("Step-4 block unexpectedly existed before consolidation")
trimmed = copy.deepcopy(head_cfg)
for parent, child in (("training", "ance_msmarco"),):
    if child not in trimmed.get(parent, {}):
        raise AssertionError(f"missing required Step-4 block: {parent}.{child}")
    del trimmed[parent][child]

# Amendment A6, semantic half: reverse exactly the keys the baseline-hardening pass
# moved into config.yaml, so every OTHER semantic change still fails.
A6_ADDED_KEYS = {
    ("training", "inbatch"): ("logging_steps", "save_total_limit", "save_fraction",
                              "gradient_checkpointing"),
    ("training", "crossbatch"): ("save_steps", "save_total_limit",
                                 "gradient_checkpointing"),
}
A6_REMOVED_KEYS = {("training",): ("overwrite_output_dir",)}
for (parent, child), keys in A6_ADDED_KEYS.items():
    for key in keys:
        if key not in trimmed[parent][child]:
            raise AssertionError(f"A6 expects {parent}.{child}.{key} to be present")
        del trimmed[parent][child][key]
for parents, keys in A6_REMOVED_KEYS.items():
    node, base_node = trimmed, base_cfg
    for seg in parents:
        node, base_node = node[seg], base_node[seg]
    for key in keys:
        if key in node:
            raise AssertionError(f"A6 expects {'.'.join(parents)}.{key} to be removed")
        node[key] = base_node[key]

# Amendment A7: cross-batch dataloader_num_workers 4 -> 0, so the DataLoader collates
# inline and cannot spawn the worker processes whose SIGBUS killed job 15039. This is
# a VALUE change rather than a key addition, so A6's add/remove machinery does not
# cover it; reverse it by value, and every OTHER value change still fails.
A7_CHANGED_VALUES = {("training", "crossbatch", "dataloader_num_workers"): (4, 0)}
for path, (base_value, head_value) in A7_CHANGED_VALUES.items():
    node = trimmed
    for seg in path[:-1]:
        node = node[seg]
    if node.get(path[-1]) != head_value:
        raise AssertionError(
            f"A7 expects {'.'.join(path)} == {head_value}, found {node.get(path[-1])!r}")
    node[path[-1]] = base_value

if trimmed != base_cfg:
    # Name the paths rather than saying "a value changed": defect P8 (the preprocessor
    # hardening pass) already differs here, and an opaque message made that invisible.
    def diff_paths(a, b, prefix=""):
        if isinstance(a, dict) and isinstance(b, dict):
            out = []
            for key in sorted(set(a) | set(b)):
                out += diff_paths(a.get(key), b.get(key), f"{prefix}.{key}" if prefix else key)
            return out
        return [] if a == b else [prefix]
    raise AssertionError(
        "config values differ outside the Step-4 block and the A6 keys at: "
        + ", ".join(diff_paths(trimmed, base_cfg))
        + "  (see CONSOLIDATION_STATUS.md defect P8)")

# Step-6 logging may add reads for display, but it may not change these public
# surfaces. train_crossbatch.py is inspected statically; --help is never used.
def literal(node):
    return node.value if isinstance(node, ast.Constant) and isinstance(node.value, str) else None

def fingerprint(source, path):
    tree = ast.parse(source, filename=path)
    cli = set()
    recipes = set()
    path_keys = set()
    env_keys = set()
    sys_path_ops = set()
    mains = set()
    main_guards = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == "main":
            mains.add("main")
        if isinstance(node, ast.If) and isinstance(node.test, ast.Compare):
            rendered = ast.unparse(node.test)
            if "__name__" in rendered and "__main__" in rendered:
                main_guards.add(rendered)
        # NOTE: do not fingerprint bare uppercase string literals. They are log
        # text, not environment keys, and Step-6 logging legitimately adds them
        # (train_async_fast_grass.py already carries 'PASS'/'FAIL'). Environment
        # access is captured precisely by the os.environ/os.getenv branches below.
        if not isinstance(node, ast.Call):
            continue
        if isinstance(node.func, ast.Attribute) and node.func.attr == "add_argument":
            cli.update(v for v in map(literal, node.args) if v and v.startswith("-"))
        if isinstance(node.func, ast.Name) and node.func.id in {"get_training_context", "get_path"} and node.args:
            value = literal(node.args[0])
            if value is not None:
                (recipes if node.func.id == "get_training_context" else path_keys).add(value)
        if isinstance(node.func, ast.Attribute) and node.func.attr in {"getenv", "setdefault", "get"} and node.args:
            value = literal(node.args[0])
            owner = ast.unparse(node.func.value)
            if value is not None and ("environ" in owner or owner == "os"):
                env_keys.add((node.func.attr, value))
        if isinstance(node.func, ast.Attribute) and node.func.attr in {"append", "insert"}:
            owner = ast.unparse(node.func.value)
            if owner == "sys.path":
                sys_path_ops.add((node.func.attr, tuple(ast.dump(a, include_attributes=False) for a in node.args)))
    for node in ast.walk(tree):
        if isinstance(node, ast.Subscript) and "environ" in ast.unparse(node.value):
            value = literal(node.slice)
            if value is not None:
                env_keys.add(("subscript", value))
    return cli, recipes, path_keys, env_keys, sys_path_ops, mains, main_guards

for path in ENTRY_POINTS:
    try:
        before = git("show", f"{BASE}:{path}", text=True)
        after = git("show", f"{HEAD}:{path}", text=True)
    except subprocess.CalledProcessError as exc:
        raise AssertionError(f"entry point is not present at both revisions: {path}") from exc
    before_fp, after_fp = fingerprint(before, path), fingerprint(after, path)
    if before_fp == after_fp:
        continue
    allowed = ALLOWED_SURFACE_ADDITIONS.get(path)
    if allowed is None:
        raise AssertionError(f"CLI/config-path/env/sys.path/main surface changed: {path}")
    for field, was, now in zip(SURFACE_FIELDS, before_fp, after_fp):
        permitted = allowed.get(field, set())
        added, removed = now - was, was - now
        if removed:
            raise AssertionError(
                f"{path}: {field} removed {sorted(removed)}; A6 permits additions only")
        if added != permitted:
            raise AssertionError(
                f"{path}: {field} added {sorted(added)}, A6 permits exactly "
                f"{sorted(permitted)}")

print("SURFACE_ALLOWLIST_OK")
PY
```

**Expected output:** The reference diff prints only the two new Step-4 launcher files and the added `training.ance_msmarco` block; it prints no modification to any pre-existing shell script or pre-existing config text. The final line is `SURFACE_ALLOWLIST_OK`. Step-6 logging is in `.py` files and therefore is not displayed by the literal reference pathspec; the static fingerprints verify that those permitted logging additions did not alter entry-point presence, CLI flags, recipe/config path keys, environment keys, or `sys.path` assumptions.

**Pass/fail condition:** **PASS:** command exits 0 and ends with `SURFACE_ALLOWLIST_OK`. **FAIL:** any pre-existing launcher byte changes, other than the four allowlisted launchers whose diffs match their pinned specs exactly (amendments A4, A5); a shell file other than the two Step-4 launchers is added/removed; config differs by anything other than the additive Step-4 block; the block changes an existing value; or any inspected entry point changes its main/CLI/config-path/env/`sys.path` fingerprint.

## AC-COMP-01 (preprocessor)

**Runnable after step:** 3

**Exact command:**

```bash
set -euo pipefail
AC_TMP=$(mktemp -d)
trap 'rm -rf "$AC_TMP"' EXIT
git archive main | tar -x -C "$AC_TMP"
cd "$AC_TMP"
test -f src/data/preprocessor.py
CUDA_VISIBLE_DEVICES='' TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 python -c 'import importlib.util,sys,traceback; p=sys.argv[1]; spec=importlib.util.spec_from_file_location("acceptance_target",p); m=importlib.util.module_from_spec(spec); sys.path[:0]=["src","scripts","."];
try:
 spec.loader.exec_module(m); assert callable(getattr(m,"run_setup")); print("IMPORT_OK",p,"run_setup callable")
except Exception:
 t=traceback.format_exc(); low=t.lower(); oos=(("cuda" in low and any(x in low for x in ("not available","no nvidia driver","not compiled","driver"))) or ("/scratch/" in t and ("filenotfounderror" in low or "no such file" in low)) or any(x in low for x in ("localentrynotfounderror","not found in the cached files","could not locate the requested files in the local cache"))); print(("IMPORT_ENVIRONMENT_OUT_OF_SCOPE\n" if oos else "")+t); sys.exit(2 if oos else 1)' src/data/preprocessor.py
```

**Expected output:** `IMPORT_OK src/data/preprocessor.py run_setup callable`. No preprocessing output is generated or checked.

**Pass/fail condition:** **PASS:** the committed `main` archive contains the module, it imports in the fresh subprocess, and module-level `run_setup` is callable. **FAIL:** file missing, real import error, or `run_setup` absent/non-callable. Exit 2 is the explicitly out-of-scope environmental state described above.

## AC-COMP-02 (in-batch)

**Runnable after step:** 3

**Exact command:**

```bash
set -euo pipefail
AC_TMP=$(mktemp -d)
trap 'rm -rf "$AC_TMP"' EXIT
git archive main | tar -x -C "$AC_TMP"
cd "$AC_TMP"
ENTRY=scripts/train_inbatch.py; LAUNCHER=scripts/launchers/run_inbatch_singularity.sh
test -f "$ENTRY" -a -f "$LAUNCHER"; test "$ENTRY" != "$LAUNCHER"; rg '^[[:space:]]*(python|torchrun)[[:space:]].*scripts/train_inbatch\.py([[:space:]]|$)' "$LAUNCHER"
CUDA_VISIBLE_DEVICES='' TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 python -c 'import importlib.util,sys,traceback; p=sys.argv[1]; spec=importlib.util.spec_from_file_location("acceptance_target",p); m=importlib.util.module_from_spec(spec); sys.path[:0]=["src","scripts","."];
try: spec.loader.exec_module(m); print("IMPORT_OK",p)
except Exception:
 t=traceback.format_exc(); low=t.lower(); oos=(("cuda" in low and any(x in low for x in ("not available","no nvidia driver","not compiled","driver"))) or ("/scratch/" in t and ("filenotfounderror" in low or "no such file" in low)) or any(x in low for x in ("localentrynotfounderror","not found in the cached files","could not locate the requested files in the local cache"))); print(("IMPORT_ENVIRONMENT_OUT_OF_SCOPE\n" if oos else "")+t); sys.exit(2 if oos else 1)' "$ENTRY"
```

**Expected output:** The launcher match containing `scripts/train_inbatch.py`, then `IMPORT_OK scripts/train_inbatch.py`.

**Pass/fail condition:** **PASS:** distinct committed entry/launcher files exist, the launcher invokes this entry, and the entry imports. **FAIL:** any structural check or real import fails. Exit 2 is environmental/out of scope.

## AC-COMP-03 (cross-batch)

**Runnable after step:** 3

**Exact command:**

```bash
set -euo pipefail
AC_TMP=$(mktemp -d)
trap 'rm -rf "$AC_TMP"' EXIT
git archive main | tar -x -C "$AC_TMP"
cd "$AC_TMP"
ENTRY=scripts/train_crossbatch.py; LAUNCHER=scripts/launchers/run_crossbatch_singularity.sh
test -f "$ENTRY" -a -f "$LAUNCHER"; test "$ENTRY" != "$LAUNCHER"; rg '^[[:space:]]*(python|torchrun)[[:space:]].*scripts/train_crossbatch\.py([[:space:]]|$)' "$LAUNCHER"
CUDA_VISIBLE_DEVICES='' TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 python -c 'import importlib.util,sys,traceback; p=sys.argv[1]; spec=importlib.util.spec_from_file_location("acceptance_target",p); m=importlib.util.module_from_spec(spec); sys.path[:0]=["src","scripts","."];
try: spec.loader.exec_module(m); print("IMPORT_OK",p)
except Exception:
 t=traceback.format_exc(); low=t.lower(); oos=(("cuda" in low and any(x in low for x in ("not available","no nvidia driver","not compiled","driver"))) or ("/scratch/" in t and ("filenotfounderror" in low or "no such file" in low)) or any(x in low for x in ("localentrynotfounderror","not found in the cached files","could not locate the requested files in the local cache"))); print(("IMPORT_ENVIRONMENT_OUT_OF_SCOPE\n" if oos else "")+t); sys.exit(2 if oos else 1)' "$ENTRY"
```

**Expected output:** The launcher match containing `scripts/train_crossbatch.py`, then `IMPORT_OK scripts/train_crossbatch.py`. No `--help` invocation occurs.

**Pass/fail condition:** **PASS:** distinct committed entry/launcher files exist, the launcher invokes this entry, and the no-argparse entry imports. **FAIL:** any structural check or real import fails. Exit 2 is environmental/out of scope.

## AC-COMP-04 (ANCE BRIGHT)

**Runnable after step:** 3

**Exact command:**

```bash
set -euo pipefail
AC_TMP=$(mktemp -d)
trap 'rm -rf "$AC_TMP"' EXIT
git archive main | tar -x -C "$AC_TMP"
cd "$AC_TMP"
ENTRY=scripts/train_ance.py; LAUNCHER=scripts/launchers/run_ance_singularity.sh
test -f "$ENTRY" -a -f scripts/run_ance_train.py -a -f scripts/run_ance_data_gen.py -a -f "$LAUNCHER"; test "$ENTRY" != "$LAUNCHER"; rg '^[[:space:]]*(python|torchrun)[[:space:]].*scripts/train_ance\.py([[:space:]]|$)' "$LAUNCHER"
for MODULE in scripts/train_ance.py scripts/run_ance_train.py scripts/run_ance_data_gen.py; do CUDA_VISIBLE_DEVICES='' TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 python -c 'import importlib.util,sys,traceback; p=sys.argv[1]; spec=importlib.util.spec_from_file_location("acceptance_target",p); m=importlib.util.module_from_spec(spec); sys.path[:0]=["src","scripts","."];
try: spec.loader.exec_module(m); print("IMPORT_OK",p)
except Exception:
 t=traceback.format_exc(); low=t.lower(); oos=(("cuda" in low and any(x in low for x in ("not available","no nvidia driver","not compiled","driver"))) or ("/scratch/" in t and ("filenotfounderror" in low or "no such file" in low)) or any(x in low for x in ("localentrynotfounderror","not found in the cached files","could not locate the requested files in the local cache"))); print(("IMPORT_ENVIRONMENT_OUT_OF_SCOPE\n" if oos else "")+t); sys.exit(2 if oos else 1)' "$MODULE"; done
```

**Expected output:** Launcher match for `scripts/train_ance.py` and one `IMPORT_OK` line for each of `train_ance.py`, `run_ance_train.py`, and `run_ance_data_gen.py`.

**Pass/fail condition:** **PASS:** all committed implementation files and the distinct launcher exist, the launcher invokes the BRIGHT orchestrator, and every entry/helper imports in its own subprocess. **FAIL:** any structural check or real import fails. Exit 2 is environmental/out of scope.

## AC-COMP-05 (sync GRASS)

**Runnable after step:** 3

**Exact command:**

```bash
set -euo pipefail
AC_TMP=$(mktemp -d)
trap 'rm -rf "$AC_TMP"' EXIT
git archive main | tar -x -C "$AC_TMP"
cd "$AC_TMP"
ENTRY=scripts/run_grass.py; LAUNCHER=scripts/launchers/run_grass_singularity.sh
test -f "$ENTRY" -a -f "$LAUNCHER"; test "$ENTRY" != "$LAUNCHER"; rg '^[[:space:]]*(python|torchrun)[[:space:]].*scripts/run_grass\.py([[:space:]]|$)' "$LAUNCHER"
CUDA_VISIBLE_DEVICES='' TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 python -c 'import importlib.util,sys,traceback; p=sys.argv[1]; spec=importlib.util.spec_from_file_location("acceptance_target",p); m=importlib.util.module_from_spec(spec); sys.path[:0]=["src","scripts","."];
try: spec.loader.exec_module(m); print("IMPORT_OK",p)
except Exception:
 t=traceback.format_exc(); low=t.lower(); oos=(("cuda" in low and any(x in low for x in ("not available","no nvidia driver","not compiled","driver"))) or ("/scratch/" in t and ("filenotfounderror" in low or "no such file" in low)) or any(x in low for x in ("localentrynotfounderror","not found in the cached files","could not locate the requested files in the local cache"))); print(("IMPORT_ENVIRONMENT_OUT_OF_SCOPE\n" if oos else "")+t); sys.exit(2 if oos else 1)' "$ENTRY"
```

**Expected output:** Launcher match for `scripts/run_grass.py`, then `IMPORT_OK scripts/run_grass.py`.

**Pass/fail condition:** **PASS:** distinct committed entry/launcher files exist, the launcher invokes this entry, and it imports. **FAIL:** any structural check or real import fails. Exit 2 is environmental/out of scope.

## AC-COMP-06 (async GRASS)

**Runnable after step:** 3

**Exact command:**

```bash
set -euo pipefail
AC_TMP=$(mktemp -d)
trap 'rm -rf "$AC_TMP"' EXIT
git archive main | tar -x -C "$AC_TMP"
cd "$AC_TMP"
ENTRY=scripts/train_async_fast_grass.py; LAUNCHER=scripts/launchers/run_async_fast_grass_singularity.sh
test -f "$ENTRY" -a -f scripts/run_async_fast_grass_miner.py -a -f scripts/run_async_fast_grass_train.py -a -f "$LAUNCHER"; test "$ENTRY" != "$LAUNCHER"; rg '^[[:space:]]*(python|torchrun)[[:space:]].*scripts/train_async_fast_grass\.py([[:space:]]|$)' "$LAUNCHER"
for MODULE in scripts/train_async_fast_grass.py scripts/run_async_fast_grass_miner.py scripts/run_async_fast_grass_train.py; do CUDA_VISIBLE_DEVICES='' TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 python -c 'import importlib.util,sys,traceback; p=sys.argv[1]; spec=importlib.util.spec_from_file_location("acceptance_target",p); m=importlib.util.module_from_spec(spec); sys.path[:0]=["src","scripts","."];
try: spec.loader.exec_module(m); print("IMPORT_OK",p)
except Exception:
 t=traceback.format_exc(); low=t.lower(); oos=(("cuda" in low and any(x in low for x in ("not available","no nvidia driver","not compiled","driver"))) or ("/scratch/" in t and ("filenotfounderror" in low or "no such file" in low)) or any(x in low for x in ("localentrynotfounderror","not found in the cached files","could not locate the requested files in the local cache"))); print(("IMPORT_ENVIRONMENT_OUT_OF_SCOPE\n" if oos else "")+t); sys.exit(2 if oos else 1)' "$MODULE"; done
```

**Expected output:** Launcher match for `scripts/train_async_fast_grass.py` and one `IMPORT_OK` line for each orchestrator, miner, and trainer module. The command never uses `--preflight`.

**Pass/fail condition:** **PASS:** all committed implementation files and the distinct launcher exist, the launcher invokes its orchestrator, and all three modules import in separate subprocesses. **FAIL:** any structural check or real import fails. Exit 2 is environmental/out of scope.

## AC-COMP-07 (sequential Fast-GRASS)

**Runnable after step:** 3

**Exact command:**

```bash
set -euo pipefail
AC_TMP=$(mktemp -d)
trap 'rm -rf "$AC_TMP"' EXIT
git archive main | tar -x -C "$AC_TMP"
cd "$AC_TMP"
ENTRY=scripts/run_fast_grass.py; LAUNCHER=scripts/launchers/run_fast_grass_singularity.sh
test -f "$ENTRY" -a -f "$LAUNCHER"; test "$ENTRY" != "$LAUNCHER"; rg '^[[:space:]]*(python|torchrun)[[:space:]].*scripts/run_fast_grass\.py([[:space:]]|$)' "$LAUNCHER"
CUDA_VISIBLE_DEVICES='' TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 python -c 'import importlib.util,sys,traceback; p=sys.argv[1]; spec=importlib.util.spec_from_file_location("acceptance_target",p); m=importlib.util.module_from_spec(spec); sys.path[:0]=["src","scripts","."];
try: spec.loader.exec_module(m); print("IMPORT_OK",p)
except Exception:
 t=traceback.format_exc(); low=t.lower(); oos=(("cuda" in low and any(x in low for x in ("not available","no nvidia driver","not compiled","driver"))) or ("/scratch/" in t and ("filenotfounderror" in low or "no such file" in low)) or any(x in low for x in ("localentrynotfounderror","not found in the cached files","could not locate the requested files in the local cache"))); print(("IMPORT_ENVIRONMENT_OUT_OF_SCOPE\n" if oos else "")+t); sys.exit(2 if oos else 1)' "$ENTRY"
```

**Expected output:** Launcher match for `scripts/run_fast_grass.py`, then `IMPORT_OK scripts/run_fast_grass.py`.

**Pass/fail condition:** **PASS:** distinct committed entry/launcher files exist, the launcher invokes this entry, and it imports. **FAIL:** any structural check or real import fails. Exit 2 is environmental/out of scope.

## AC-COMP-08 (ANCE MS MARCO, conditional)

> **Amended during consolidation (A3; evidence in `CONSOLIDATION_STATUS.md`).**
> `_load_qrels` and `_evaluate` are **not** baseline-unique. `fast-grass` refactored them into
> `src/utils/helpers.py` as `_load_qrels` / `evaluate_bright` — bodies line-for-line identical
> to baseline's apart from a docstring, an `open()` mode, a defaulted `temp_workdir_key`, and a
> defaulted `eval_metric` — and `train_ance.py` already imports and calls them. Demanding them
> in `train_ance.py` would duplicate ~110 lines and undo the refactor. The row now requires
> them **in helpers** and **forbids** them in `train_ance.py`, and the landed/absent split
> turns on the three preprocessor methods alone.

**Runnable after step:** 5

**Exact command:**

```bash
set -euo pipefail
AC_TMP=$(mktemp -d)
trap 'rm -rf "$AC_TMP"' EXIT
git archive main | tar -x -C "$AC_TMP"
cd "$AC_TMP"
test -f scripts/eval_msmarco.py -a -f scripts/launchers/run_ance_msmarco_singularity.sh -a -f scripts/launchers/eval_msmarco_singularity.sh -a -f scripts/train_ance.py -a -f src/data/preprocessor.py
test scripts/eval_msmarco.py != scripts/launchers/eval_msmarco_singularity.sh
test scripts/train_ance.py != scripts/launchers/run_ance_msmarco_singularity.sh
rg '^[[:space:]]*(python|torchrun)[[:space:]].*scripts/eval_msmarco\.py([[:space:]]|$)' scripts/launchers/eval_msmarco_singularity.sh
rg '^[[:space:]]*(python|torchrun)[[:space:]].*scripts/train_ance\.py[[:space:]]+--recipe[[:space:]]+ance_msmarco([[:space:]]|$)' scripts/launchers/run_ance_msmarco_singularity.sh
CUDA_VISIBLE_DEVICES='' TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 python -c 'import importlib.util,sys,traceback; p=sys.argv[1]; spec=importlib.util.spec_from_file_location("acceptance_target",p); m=importlib.util.module_from_spec(spec); sys.path[:0]=["src","scripts","."];
try: spec.loader.exec_module(m); assert callable(getattr(m,"main")); print("IMPORT_OK",p,"main callable")
except Exception:
 t=traceback.format_exc(); low=t.lower(); oos=(("cuda" in low and any(x in low for x in ("not available","no nvidia driver","not compiled","driver"))) or ("/scratch/" in t and ("filenotfounderror" in low or "no such file" in low)) or any(x in low for x in ("localentrynotfounderror","not found in the cached files","could not locate the requested files in the local cache"))); print(("IMPORT_ENVIRONMENT_OUT_OF_SCOPE\n" if oos else "")+t); sys.exit(2 if oos else 1)' scripts/eval_msmarco.py
cd - >/dev/null
python - <<'PY'
import ast
import pathlib
import re
import subprocess

def source(path):
    return subprocess.check_output(["git", "show", f"main:{path}"], text=True)

ance_defs = {n.name for n in ast.walk(ast.parse(source("scripts/train_ance.py")))
             if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))}
helper_defs = {n.name for n in ast.walk(ast.parse(source("src/utils/helpers.py")))
               if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))}
prep_tree = ast.parse(source("src/data/preprocessor.py"))
prep_methods = set()
for node in ast.walk(prep_tree):
    if isinstance(node, ast.ClassDef) and node.name == "BRIGHTPreprocessor":
        prep_methods.update(n.name for n in node.body
                            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)))
# Amendment A3. The qrels loader and the evaluator are NOT baseline-unique: fast-grass
# refactored them out of train_ance.py into src/utils/helpers.py, where train_ance.py
# already imports and calls them. Requiring them in train_ance.py would force ~110
# duplicated lines and undo that refactor. They are therefore required in helpers, and
# forbidden in train_ance.py, which turns this row into a no-duplication check.
required_helpers = {"_load_qrels", "evaluate_bright"}
forbidden_ance = {"_load_qrels", "_evaluate"}
if not required_helpers <= helper_defs:
    raise AssertionError(f"missing from src/utils/helpers.py: {sorted(required_helpers - helper_defs)}")
if forbidden_ance & ance_defs:
    raise AssertionError(f"duplicated into scripts/train_ance.py: {sorted(forbidden_ance & ance_defs)}")
required_prep = {"prepare_msmarco_full_corpus", "prepare_msmarco_tevatron_train", "prepare_msmarco_dev"}
landed = required_prep <= prep_methods
fully_absent = not (required_prep & prep_methods)
if landed:
    print("MSMARCO_ACCEPT_STATE STEP5_LANDED")
elif fully_absent:
    status = pathlib.Path("CONSOLIDATION_STATUS.md")
    if not status.is_file():
        raise AssertionError("Step 5 helpers are absent but CONSOLIDATION_STATUS.md is missing")
    text = status.read_text()
    required = [r"(?is)step\s*5", r"(?is)revert", r"(?is)MS\s*MARCO", r"(?is)accepted\s+gap|known[, -]+accepted\s+gap", r"archive/baseline"]
    if not all(re.search(pattern, text) for pattern in required):
        raise AssertionError("CONSOLIDATION_STATUS.md does not record the accepted Step-5 revert gap and archive/baseline")
    print("MSMARCO_ACCEPT_STATE STEP5_REVERTED_GAP_RECORDED")
else:
    raise AssertionError("partial Step-5 helper state is not accepted")
PY
```

**Expected output:** The two launcher invocation matches, `IMPORT_OK scripts/eval_msmarco.py main callable`, and exactly one accepted state: `MSMARCO_ACCEPT_STATE STEP5_LANDED` or `MSMARCO_ACCEPT_STATE STEP5_REVERTED_GAP_RECORDED`.

**Pass/fail condition:** **PASS (landed):** additive entry/launchers exist and are distinct/wired, evaluation entry imports with callable `main`, `_load_qrels` and `evaluate_bright` are present in `src/utils/helpers.py` and absent from `scripts/train_ance.py`, and all three preprocessor methods are present. **PASS (reverted):** additive entry/launchers exist and are wired, evaluation entry imports with callable `main`, all three preprocessor methods are absent, and `CONSOLIDATION_STATUS.md` explicitly records the Step-5 revert as a known accepted MS MARCO gap retained on `archive/baseline`. **FAIL:** partial helper state, missing status evidence, bad wiring, or real import error. Exit 2 is environmental/out of scope.

## AC-TEST-01 (CLAUDE.md CPU roster)

**Runnable after step:** 6

**Exact command:**

```bash
set -euo pipefail
AC_TMP=$(mktemp -d)
trap 'rm -rf "$AC_TMP"' EXIT
git archive main | tar -x -C "$AC_TMP"
cd "$AC_TMP"
KMP_DUPLICATE_LIB_OK=TRUE PYTHONHASHSEED=0 CUDA_VISIBLE_DEVICES='' TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 python - <<'PY'
import os
import re
import subprocess
import sys

cases = [
    ("async_fast_grass_handoff_test.py", [sys.executable, "tests/async_fast_grass_handoff_test.py"], "count", None),
    ("async_fast_grass_cache_semantics_test.py", [sys.executable, "tests/async_fast_grass_cache_semantics_test.py"], "count", None),
    ("async_fast_grass_persistence_test.py", [sys.executable, "tests/async_fast_grass_persistence_test.py"], "count", None),
    ("async_fast_grass_pilot_test.py", [sys.executable, "tests/async_fast_grass_pilot_test.py"], "count", None),
    ("async_fast_grass_integration_smoke.py", [sys.executable, "tests/async_fast_grass_integration_smoke.py"], "integration", "PASS  async handoff integration"),
    ("fast_grass_test.py", [sys.executable, "tests/fast_grass_test.py"], "count", None),
    ("fast_grass_smoke.py", [sys.executable, "tests/fast_grass_smoke.py"], "count", None),
    ("grass_test.py", [sys.executable, "tests/grass_test.py"], "count", None),
    ("grass_smoke.py", [sys.executable, "tests/grass_smoke.py"], "count", None),
    ("fast_grass_mine_timing.py --synthetic", [sys.executable, "scripts/dev/fast_grass_mine_timing.py", "--synthetic"], "marker", "PASS  miner-timing harness runs end to end"),
    ("fast_grass_train_timing.py --synthetic", [sys.executable, "scripts/dev/fast_grass_train_timing.py", "--synthetic"], "marker", "PASS  trainer-timing harness runs end to end"),
    ("async_fast_grass_speed_estimate.py file-free smoke", [sys.executable, "scripts/dev/async_fast_grass_speed_estimate.py", "--seconds_per_train_step", "1", "--t_mine_round", "10", "--total_queries", "100", "--batch_size", "10", "--num_epochs", "2", "--checkpoint_write_time", "1"], "marker", "ASYNC FAST-GRASS — EXPECTED SPEEDUP & CADENCE ESTIMATE"),
    ("async_fast_grass_quality_probe.py --synthetic", [sys.executable, "scripts/dev/async_fast_grass_quality_probe.py", "--synthetic"], "marker", "PASS  dosage-probe harness runs end to end"),
]

for name, command, kind, marker in cases:
    run = subprocess.run(command, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=os.environ.copy())
    if run.returncode != 0:
        print(run.stdout)
        raise SystemExit(f"FAIL {name}: exit {run.returncode}")
    if kind == "count":
        counts = [(int(a), int(b)) for a, b in re.findall(r"(\d+)\s*/\s*(\d+)(?:\s+checks)?\s+passed", run.stdout)]
        # Require at least one N/N summary and that EVERY summary is all-green.
        # `any` would let a suite pass on one green section while another failed.
        if not counts or not all(total > 0 and passed == total for passed, total in counts):
            print(run.stdout)
            raise SystemExit(f"FAIL {name}: no all-green N/N summary")
    elif kind == "integration":
        if run.stdout.count(marker) != 1:
            print(run.stdout)
            raise SystemExit(f"FAIL {name}: expected its single overall PASS line")
    elif marker not in run.stdout:
        print(run.stdout)
        raise SystemExit(f"FAIL {name}: missing success marker")
    print("CPU_SUITE_OK", name)
PY
```

**Expected output:** Thirteen `CPU_SUITE_OK ...` lines, one for every listed command. Each counted suite reports a dynamically checked all-green `N/N` summary. The integration smoke is accepted only by its single `PASS  async handoff integration` line; no `N/N` grep is used.

**Pass/fail condition:** **PASS:** all thirteen subprocesses exit 0 and satisfy their own dynamic count or named success marker. **FAIL:** any nonzero exit, missing all-green summary, or missing marker. `KMP_DUPLICATE_LIB_OK=TRUE` and `PYTHONHASHSEED=0` must be present exactly as shown.

## AC-INV-06 (`src/` import boundary)

**Runnable after step:** 6

**Exact command:**

```bash
set -euo pipefail
AC_TMP=$(mktemp -d)
trap 'rm -rf "$AC_TMP"' EXIT
git archive main | tar -x -C "$AC_TMP"
cd "$AC_TMP"
python - <<'PY'
import ast
from pathlib import Path

violations = []
files = sorted(Path("src").rglob("*.py"))
for path in files:
    tree = ast.parse(path.read_text(), filename=str(path))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == "scripts" or alias.name.startswith("scripts."):
                    violations.append((path, node.lineno, alias.name))
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            if module == "scripts" or module.startswith("scripts."):
                violations.append((path, node.lineno, module))
        elif isinstance(node, ast.Call) and node.args and isinstance(node.args[0], ast.Constant) and isinstance(node.args[0].value, str):
            target = node.args[0].value
            dynamic = ((isinstance(node.func, ast.Name) and node.func.id == "__import__") or
                       (isinstance(node.func, ast.Attribute) and node.func.attr == "import_module"))
            if dynamic and (target == "scripts" or target.startswith("scripts.")):
                violations.append((path, node.lineno, target))
if violations:
    for path, line, target in violations:
        print(f"SRC_IMPORTS_SCRIPTS {path}:{line} {target}")
    raise SystemExit(1)
print(f"SRC_IMPORT_BOUNDARY_OK {len(files)} Python files checked")
PY
```

**Expected output:** One line matching `SRC_IMPORT_BOUNDARY_OK N Python files checked`, with `N > 0`, and no `SRC_IMPORTS_SCRIPTS` line.

**Pass/fail condition:** **PASS:** every committed Python file under `src/` parses and no direct or literal dynamic import targets `scripts` or `scripts.*`. **FAIL:** parse error, zero/absent success output, or any reported boundary violation.
