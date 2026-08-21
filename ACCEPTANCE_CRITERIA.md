# Consolidation acceptance criteria

These criteria assess consolidation mechanics only. They do not assess training correctness or the conceptual accuracy of documentation. All checks are CPU-only. Do not substitute `--preflight` for any row: it loads the full 655k-document corpus and is intentionally excluded.

“Runnable after step N” is an availability state, not a result. Before that step is complete, the row is **NOT RUNNABLE YET**, not failed. Once runnable, exit 0 and the stated success output mean **PASS**; exit 1 or a missing success output mean **FAIL**.

**Mandatory re-run.** Step 6 edits `scripts/train_inbatch.py`, `scripts/train_crossbatch.py`, `scripts/train_ance.py`, `scripts/run_grass.py`, `scripts/run_fast_grass.py`, and `scripts/train_async_fast_grass.py` to add startup logging. Every `AC-COMP-*` row therefore becomes runnable at the step listed but **must be re-run after Step 6**, exactly as Step 0's preprocessor diff is re-run after every step. A first-run PASS is not final: a logging edit that breaks an import is caught only by the re-run. `AC-SURFACE-01`, `AC-TEST-01` and `AC-INV-06` are already gated at Step 6 and need no re-run. Component import commands may also exit 2 with `IMPORT_ENVIRONMENT_OUT_OF_SCOPE`; that is neither pass nor fail and must include an exception narrowly attributable to unavailable CUDA, a missing `/scratch/...` path, or an unavailable Hugging Face cache in forced-offline mode.

The import harnesses set `CUDA_VISIBLE_DEVICES` empty and force Transformers/Hugging Face offline, then use `importlib` in a fresh subprocess without calling `main()`, `run_setup()`, or any training function. Therefore ordinary absence of CUDA, `/scratch`, and cached model files is not exercised. A traceback not matching those narrow environmental signatures exits 1 as a real import error; missing Python packages, syntax errors, bad imports, and application exceptions are consolidation failures.

## AC-SURFACE-01

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
    "scripts/eval_msmarco_singularity.sh",
    "scripts/run_ance_msmarco_singularity.sh",
}
# Amendment A7. The launcher's only target, scripts/grass_twoset_feasibility.py, was
# never committed, so this script could not run from a clone of `main` at any point in
# its history. Deleting it removes a dead file, not a capability. Recorded as D2 in
# CONSOLIDATION_STATUS.md, which also preserves where the module actually lives.
ALLOWED_REMOVED_SH = {
    "scripts/run_twoset_feasibility_singularity.sh",
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
ALLOWED_CHANGED_SH = {
    "scripts/run_inbatch_singularity.sh": [
        "-#SBATCH --time=14:00:00   # TEMP: OOM smoke test \u2014 passes step-1 mem peak then SLURM kills it. RESTORE to 24:00:00 for real run.",
        "+#SBATCH --time=24:00:00",
    ],
    "scripts/run_crossbatch_singularity.sh": [
        "+EXIT_CODE=$?",
        "+",
        "+if [ $EXIT_CODE -eq 0 ]; then",
        "+    echo \"\u2705 Cross-batch training completed successfully\"",
        "+else",
        "+    echo \"\u274c Cross-batch training failed with code $EXIT_CODE\"",
        "+fi",
        "+",
        "+",
        "+exit $EXIT_CODE",
    ],
    "scripts/run_grass_singularity.sh": [
        "+# GRASS_DEBUG=1           # 512-item mixture smoke run",
        "-        ${GRASS_LAMBDA:+--lambda_val $GRASS_LAMBDA}",
        "+        ${GRASS_LAMBDA:+--lambda_val $GRASS_LAMBDA} \\",
        "+        ${GRASS_DEBUG:+--debug}",
    ],
    "scripts/run_fast_grass_singularity.sh": [
        "+# FAST_GRASS_DEBUG=1      # 512-item mixture smoke run",
        "-        ${FAST_GRASS_NO_EVAL:+--no_eval}",
        "+        ${FAST_GRASS_NO_EVAL:+--no_eval} \\",
        "+        ${FAST_GRASS_DEBUG:+--debug}",
    ],
    "scripts/run_async_fast_grass_singularity.sh": [
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
    ],
    "scripts/run_fast_grass_timing_singularity.sh": [
        "-            python -u scripts/async_fast_grass_handoff_test.py",
        "-            python -u scripts/async_fast_grass_cache_semantics_test.py",
        "-            python -u scripts/fast_grass_test.py",
        "+            python -u tests/async_fast_grass_handoff_test.py",
        "+            python -u tests/async_fast_grass_cache_semantics_test.py",
        "+            python -u tests/fast_grass_test.py",
    ],
}
ENTRY_POINTS = [
    "scripts/train_inbatch.py",
    "scripts/train_crossbatch.py",
    "scripts/train_ance.py",
    "scripts/run_grass.py",
    "scripts/run_fast_grass.py",
    "scripts/train_async_fast_grass.py",
]

def git(*args, text=False):
    return subprocess.check_output(["git", *args], text=text)

def paths(rev, pattern):
    return {p for p in git("ls-tree", "-r", "--name-only", rev, text=True).splitlines()
            if fnmatch.fnmatch(p, pattern)}

base_sh = paths(BASE, "scripts/*.sh")
head_sh = paths(HEAD, "scripts/*.sh")
if head_sh - base_sh != ALLOWED_NEW_SH:
    raise AssertionError(f"unexpected added shell files: {sorted(head_sh - base_sh)}")
if base_sh - head_sh != ALLOWED_REMOVED_SH:
    raise AssertionError(f"unexpected removed shell files: {sorted(base_sh - head_sh)}")
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

if not any("".join(line for i, line in enumerate(head_lines)
                    if not any(start <= i < end for start, end in selected)) == base_text
           for selected in candidates(head_lines, ranges)):
    raise AssertionError("config diff is not exactly the additive Step-4 block")

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
if trimmed != base_cfg:
    raise AssertionError("a pre-existing config value changed")

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
    if fingerprint(before, path) != fingerprint(after, path):
        raise AssertionError(f"CLI/config-path/env/sys.path/main surface changed: {path}")

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
ENTRY=scripts/train_inbatch.py; LAUNCHER=scripts/run_inbatch_singularity.sh
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
ENTRY=scripts/train_crossbatch.py; LAUNCHER=scripts/run_crossbatch_singularity.sh
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
ENTRY=scripts/train_ance.py; LAUNCHER=scripts/run_ance_singularity.sh
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
ENTRY=scripts/run_grass.py; LAUNCHER=scripts/run_grass_singularity.sh
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
ENTRY=scripts/train_async_fast_grass.py; LAUNCHER=scripts/run_async_fast_grass_singularity.sh
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
ENTRY=scripts/run_fast_grass.py; LAUNCHER=scripts/run_fast_grass_singularity.sh
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
test -f scripts/eval_msmarco.py -a -f scripts/run_ance_msmarco_singularity.sh -a -f scripts/eval_msmarco_singularity.sh -a -f scripts/train_ance.py -a -f src/data/preprocessor.py
test scripts/eval_msmarco.py != scripts/eval_msmarco_singularity.sh
test scripts/train_ance.py != scripts/run_ance_msmarco_singularity.sh
rg '^[[:space:]]*(python|torchrun)[[:space:]].*scripts/eval_msmarco\.py([[:space:]]|$)' scripts/eval_msmarco_singularity.sh
rg '^[[:space:]]*(python|torchrun)[[:space:]].*scripts/train_ance\.py[[:space:]]+--recipe[[:space:]]+ance_msmarco([[:space:]]|$)' scripts/run_ance_msmarco_singularity.sh
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
    ("fast_grass_mine_timing.py --synthetic", [sys.executable, "scripts/fast_grass_mine_timing.py", "--synthetic"], "marker", "PASS  miner-timing harness runs end to end"),
    ("fast_grass_train_timing.py --synthetic", [sys.executable, "scripts/fast_grass_train_timing.py", "--synthetic"], "marker", "PASS  trainer-timing harness runs end to end"),
    ("async_fast_grass_speed_estimate.py file-free smoke", [sys.executable, "scripts/async_fast_grass_speed_estimate.py", "--seconds_per_train_step", "1", "--t_mine_round", "10", "--total_queries", "100", "--batch_size", "10", "--num_epochs", "2", "--checkpoint_write_time", "1"], "marker", "ASYNC FAST-GRASS — EXPECTED SPEEDUP & CADENCE ESTIMATE"),
    ("async_fast_grass_quality_probe.py --synthetic", [sys.executable, "scripts/async_fast_grass_quality_probe.py", "--synthetic"], "marker", "PASS  dosage-probe harness runs end to end"),
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
