"""Patch Tevatron to remove Qwen/multimodal dependencies.

Tevatron commit 8f31cd8 added Qwen multimodal support that requires
packages (qwen_omni_utils, Qwen2_5Omni models) not available in our
PyTorch 2.1 container. This script removes all multimodal references
so Tevatron works for text-only dense retrieval.

Usage:
    python scripts/patch_tevatron.py [tevatron_base_path]            # apply
    python scripts/patch_tevatron.py --verify [tevatron_base_path]   # check only

`--verify` is what makes the patches reproducible rather than merely applied once: it
asserts every edit DELFTBLUE_SETUP.md section 2 documents is still in place, including
2.4 (stale bytecode), and prints the resolved dependency versions so a run's environment
can be compared against the one that produced a result. Exits nonzero on any miss.
"""

import sys
import os
import re
import glob

def patch_tevatron(base_path):
    if not os.path.isdir(base_path):
        print(f"ERROR: Tevatron not found at {base_path}")
        sys.exit(1)

    patched = 0

    for pyfile in glob.glob(os.path.join(base_path, "**", "*.py"), recursive=True):
        with open(pyfile, 'r') as f:
            original = f.read()

        content = original

        # 1. Comment out qwen_omni_utils imports (e.g. collator.py)
        # The (?!\s*#) guard makes this idempotent. Without it every run prepended
        # another '# ' to lines that were already commented, so the script produced a
        # different file each time it ran -- collator.py had reached '# # # #' on the
        # cluster -- rewriting sources (and invalidating their bytecode, section 2.4)
        # on every setup.sh.
        content = re.sub(
            r'^(?!\s*#)(.*qwen_omni_utils.*)',
            r'# \1',
            content, flags=re.MULTILINE
        )

        # 2. Comment out Qwen2_5Omni references (e.g. dense.py imports + assignments)
        content = re.sub(
            r'^(?!\s*#)(.*Qwen2_5Omni.*)',
            r'# \1',
            content, flags=re.MULTILINE
        )

        # 3. Remove entire MultiModalDenseModel class definition + body
        content = re.sub(
            r'^class MultiModalDenseModel.*?(?=\nclass |\n[^\s#]|\Z)',
            '',
            content, flags=re.DOTALL | re.MULTILINE
        )

        # 4. Remove visual encoder freezing block (for loop + indented body)
        content = re.sub(
            r'^\s*for param in self\.encoder\.visual\.parameters\(\):.*?\n(?:\s+.*\n)*',
            '',
            content, flags=re.MULTILINE
        )

        # 5. Fix __init__.py: remove MultiModalDenseModel from imports
        content = content.replace(
            'from .dense import DenseModel, MultiModalDenseModel',
            'from .dense import DenseModel'
        )

        # 6. Any MultiModalDenseModel reference rules 3 and 5 did not reach. The
        # multimodal DRIVERS import it as `from tevatron.retriever.modeling import
        # ... MultiModalDenseModel`, a form rule 5's literal replace never matched, so
        # encode_mm.py and train_mm.py were left importing a name that no longer
        # exists. Inert for text-only retrieval, but it means the package is not
        # self-consistent -- and --verify has no way to tell that from a real miss.
        # Runs last so it only mops up what the targeted rules left behind.
        content = re.sub(
            r'^(?!\s*#)(.*MultiModalDenseModel.*)',
            r'# \1',
            content, flags=re.MULTILINE
        )

        if content != original:
            with open(pyfile, 'w') as f:
                f.write(content)
            patched += 1
            print(f"  Patched: {os.path.relpath(pyfile, base_path)}")

    print(f"  Total: {patched} file(s) patched")
    return patched

# The four edits DELFTBLUE_SETUP.md section 2 requires, as (label, check) pairs.
# 2.4 is the one setup.sh never performed: patch_tevatron rewrites dense.py in place,
# so a .pyc with a newer mtime silently shadows the patched source.
def _active_lines(path, pattern):
    """Occurrences that are NOT commented out."""
    if not os.path.isfile(path):
        return [f"{path} is missing"]
    hits = []
    with open(path) as f:
        for i, line in enumerate(f, 1):
            if re.search(pattern, line) and not line.lstrip().startswith("#"):
                hits.append(f"{os.path.basename(path)}:{i}: {line.strip()}")
    return hits


def _stale_pyc(pyfile, pyc):
    """[] unless this .pyc was compiled from a different version of `pyfile`."""
    import struct
    try:
        with open(pyc, 'rb') as f:
            header = f.read(16)
        if len(header) < 16:
            return [f"2.4 unreadable bytecode: {pyc}"]
        flags, mtime, size = struct.unpack('<III', header[4:16])
        if flags & 0b1:
            return []                    # hash-based .pyc: Python validates it itself
        st = os.stat(pyfile)
        if mtime != int(st.st_mtime) & 0xFFFFFFFF or size != st.st_size & 0xFFFFFFFF:
            return [f"2.4 bytecode was compiled from a different {os.path.basename(pyfile)}: {pyc}"]
    except OSError as e:
        return [f"2.4 could not check {pyc}: {e}"]
    return []


def verify_tevatron(base_path):
    """Return a list of problems; empty means every documented patch is in place."""
    problems = []
    if not os.path.isdir(base_path):
        return [f"Tevatron not found at {base_path}"]

    # 2.1 / 2.2 / troubleshooting: no ACTIVE Qwen or multimodal references anywhere.
    pattern = r"qwen_omni_utils|Qwen2_5Omni|MultiModalDenseModel|encoder\.visual"
    for pyfile in glob.glob(os.path.join(base_path, "**", "*.py"), recursive=True):
        problems += [f"2.1/2.2 active Qwen/multimodal reference -> {h}"
                     for h in _active_lines(pyfile, pattern)]

    # 2.3: train.py uses torch.float32 at import time without importing torch.
    train_py = os.path.join(base_path, "retriever", "driver", "train.py")
    if not os.path.isfile(train_py):
        problems.append(f"2.3 {train_py} is missing")
    elif not any(line.startswith("import torch")
                 for line in open(train_py).read().splitlines()):
        problems.append("2.3 train.py does not import torch")

    # 2.4: bytecode compiled from a DIFFERENT version of the source. A timestamp .pyc
    # embeds the source mtime and size it was built from; a fresh pip install matches
    # and is fine, whereas an in-place patch of dense.py leaves a mismatch. Comparing
    # raw mtimes instead would flag every normal install.
    for pyfile in glob.glob(os.path.join(base_path, "**", "*.py"), recursive=True):
        cache = os.path.join(os.path.dirname(pyfile), "__pycache__")
        stem = os.path.splitext(os.path.basename(pyfile))[0]
        for pyc in glob.glob(os.path.join(cache, f"{stem}.cpython-*.pyc")):
            problems += _stale_pyc(pyfile, pyc)
    return problems


def print_environment():
    """The versions a result depends on. GradCache reports a static 0.1.0, so its
    commit -- not its version -- is its identity; direct_url.json carries it when pip
    recorded one."""
    from importlib import metadata
    print("\nResolved environment:")
    for name in ("torch", "transformers", "accelerate", "datasets", "peft",
                 "safetensors", "tevatron", "GradCache", "pyserini",
                 "faiss-cpu", "faiss-gpu"):
        try:
            print(f"  {name:16} {metadata.version(name)}")
        except Exception:                                          # noqa: BLE001
            print(f"  {name:16} <absent>")
    for dist in ("GradCache", "tevatron"):
        try:
            raw = metadata.distribution(dist).read_text("direct_url.json")
            if raw:
                print(f"  {dist} source: {raw.strip()}")
        except Exception:                                          # noqa: BLE001
            pass


if __name__ == "__main__":
    argv = [a for a in sys.argv[1:] if a != "--verify"]
    verify_only = "--verify" in sys.argv[1:]
    base = argv[0] if argv else os.path.expanduser(
        "~/.local/lib/python3.10/site-packages/tevatron"
    )

    if verify_only:
        print(f"Verifying Tevatron patches at: {base}")
        issues = verify_tevatron(base)
        for issue in issues:
            print(f"  ❌ {issue}")
        print_environment()
        if issues:
            print(f"\n❌ {len(issues)} patch problem(s); see DELFTBLUE_SETUP.md section 2")
            sys.exit(1)
        print("\n✅ TEVATRON_PATCHES_VERIFIED (2.1 Qwen, 2.2 MultiModal, 2.3 torch, 2.4 bytecode)")
        sys.exit(0)

    print(f"Patching Tevatron at: {base}")
    patch_tevatron(base)
    # Section 2.4: the in-place rewrite above leaves any existing .pyc newer than its
    # source, which then shadows the patched code.
    removed = 0
    for pyc in glob.glob(os.path.join(base, "**", "__pycache__", "*.pyc"), recursive=True):
        os.remove(pyc)
        removed += 1
    print(f"  Cleared {removed} stale .pyc file(s)")
