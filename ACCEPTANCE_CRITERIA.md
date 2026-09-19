# Consolidation acceptance criteria

These criteria assess consolidation mechanics only. They do not assess training correctness or the conceptual accuracy of documentation. All checks are CPU-only. Do not substitute `--preflight` for any row: it loads the full 655k-document corpus and is intentionally excluded.

“Runnable after step N” is an availability state, not a result. Before that step is complete, the row is **NOT RUNNABLE YET**, not failed. Once runnable, exit 0 and the stated success output mean **PASS**; exit 1 or a missing success output mean **FAIL**.

**Mandatory re-run.** Step 6 edits `scripts/train_inbatch.py`, `scripts/train_crossbatch.py`, `scripts/train_ance.py`, `scripts/run_grass.py`, `scripts/run_fast_grass.py`, and `scripts/train_async_fast_grass.py` to add startup logging. Every `AC-COMP-*` row therefore becomes runnable at the step listed but **must be re-run after Step 6**, exactly as Step 0's preprocessor diff is re-run after every step. A first-run PASS is not final: a logging edit that breaks an import is caught only by the re-run. `AC-TEST-01` and `AC-INV-06` are already gated at Step 6 and need no re-run. `AC-SURFACE-01` and `AC-COMP-08` are retired (see their rows). Component import commands may also exit 2 with `IMPORT_ENVIRONMENT_OUT_OF_SCOPE`; that is neither pass nor fail and must include an exception narrowly attributable to unavailable CUDA, a missing `/scratch/...` path, or an unavailable Hugging Face cache in forced-offline mode.

The import harnesses set `CUDA_VISIBLE_DEVICES` empty and force Transformers/Hugging Face offline, then use `importlib` in a fresh subprocess without calling `main()`, `run_setup()`, or any training function. Therefore ordinary absence of CUDA, `/scratch`, and cached model files is not exercised. A traceback not matching those narrow environmental signatures exits 1 as a real import error; missing Python packages, syntax errors, bad imports, and application exceptions are consolidation failures.

## AC-SURFACE-01 — RETIRED

**Status: retired, not failing.** This row pinned `scripts/*.sh` and `config/config.yaml`
against `archive/main-post-promotion` with an exact allowlist, and asserted that
`training.ance_msmarco` was the sole additive config block and
`scripts/launchers/run_ance_msmarco_singularity.sh` the sole additive launcher.

The ANCE refactor deleted both. `ance_msmarco` was a BGE-M3 MS MARCO sanity recipe that was
never run end to end, was blocked on `P6`, and duplicated surface between the reportable
BRIGHT arm (`ance`) and the Microsoft Passage reproduction (`ance_paper`). With the recipe
gone the allowlist describes a tree that no longer exists, so the row cannot be repaired —
only rewritten against a new baseline, which would make it a different criterion.

The property it protected — that launcher and config surface does not drift silently — is
retained by the launchers being hardcoded to one recipe each (no env-selected recipe, so
there is no unset variable that can pick the wrong experiment) and by
`helpers.require_recipe_keys`, which fails a run whose config declares a key nothing reads or
reads a key nothing declares.

Recorded in `CONSOLIDATION_STATUS.md`. Previously tracked as failing under `P8`; that entry
is superseded by this retirement.


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

## AC-COMP-08 (ANCE MS MARCO) — RETIRED

**Status: retired, not failing.** This row required
`scripts/launchers/run_ance_msmarco_singularity.sh` to exist and to invoke
`scripts/train_ance.py --recipe ance_msmarco`. Both the launcher and the recipe were deleted
by the ANCE refactor.

The MS MARCO path itself is not gone and is not untested: `scripts/eval_msmarco.py` now
defaults to `ance_paper`, `scripts/launchers/eval_msmarco_singularity.sh` is unchanged, and
the three preprocessor methods (`prepare_msmarco_full_corpus`,
`prepare_msmarco_tevatron_train`, `prepare_msmarco_dev`) are still exercised by
`tests/preprocessor_test.py`. What this row uniquely covered — a second, non-reportable BGE
port of MS MARCO — no longer exists to cover.

Recorded in `CONSOLIDATION_STATUS.md`.


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
