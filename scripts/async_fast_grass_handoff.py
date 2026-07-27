"""
Async Fast-GRASS — trainer/miner handoff protocol.

One implementation of the `ready_N` commit protocol, imported by BOTH the miner
(`run_async_fast_grass_miner.py`) and the trainer (`run_async_fast_grass_train.py`)
so the two processes cannot drift apart on layout or commit semantics. The handoff
tests import the same functions rather than grading a private copy.

Layout (async_fast_grass_implementation_details.md, "Handoff Protocol")::

    temp_fast_grass_workdir/async_mining/
      initial_data/ *.jsonl
      mining_meta_initial.json
      cache_state_initial.pt
      ready_initial

      work_N/                      <- miner scratch; the trainer NEVER reads this
        training_data/ *.jsonl
        cache_state.pt
        mining_meta.json

      training_data_N/ *.jsonl
      cache_state_N.pt
      mining_meta_N.json
      ready_N                      <- the ONLY trainer-visible completion signal

`ready_N` is written LAST, through a temporary marker plus atomic rename, after
every other artifact of round N is durable. A round is committed only when its
marker exists: final-path artifacts without one are crash leftovers.

`ready_initial` is the trainer's step-0 input and is handled separately. Numeric
rounds start at `ready_1`; `latest_committed_round` only discovers numeric markers
and intentionally ignores `ready_initial`.
"""
import json
import os
import shutil
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root / 'src'))

from utils.helpers import get_latest_marker_no, is_valid_checkpoint  # noqa: E402


# ---- layout ----------------------------------------------------------------

def round_paths(root, n):
    """Final paths for round ``n``. One place that knows the layout."""
    root = Path(root)
    return {
        'training_data': root / f"training_data_{n}",
        'cache_state': root / f"cache_state_{n}.pt",
        'mining_meta': root / f"mining_meta_{n}.json",
        'ready': root / f"ready_{n}",
        'work': root / f"work_{n}",
    }


def initial_paths(root):
    root = Path(root)
    return {
        'training_data': root / "initial_data",
        'cache_state': root / "cache_state_initial.pt",
        'mining_meta': root / "mining_meta_initial.json",
        'ready': root / "ready_initial",
    }


def work_paths(root, n):
    """Miner scratch paths for round ``n`` (never read by the trainer)."""
    work = Path(root) / f"work_{n}"
    return {
        'work': work,
        'training_data': work / "training_data",
        'cache_state': work / "cache_state.pt",
        'mining_meta': work / "mining_meta.json",
    }


# ---- commit ----------------------------------------------------------------

def latest_committed_round(root):
    """Highest N with a `ready_N` marker, or 0. Ignores `ready_initial`."""
    root = Path(root)
    if not root.exists():
        return 0
    return get_latest_marker_no(root, prefix="ready_")


def publish_round(root, n):
    """Commit round ``n`` from `work_N/` to its final paths, marker LAST.

    Order matters and is fixed by the doc::

        write and close all files under work_N/
        os.replace(work_N/training_data, training_data_N)
        os.replace(work_N/cache_state.pt, cache_state_N.pt)
        os.replace(work_N/mining_meta.json, mining_meta_N.json)
        write ready_N.tmp, then os.replace(ready_N.tmp, ready_N)

    Every step is an atomic rename within one filesystem, so a crash can leave
    final-path artifacts without a marker (recovered by ``reap_orphans``) but can
    never leave a marker pointing at incomplete data.
    """
    root = Path(root)
    w, f = work_paths(root, n), round_paths(root, n)
    missing = [k for k in ('training_data', 'cache_state', 'mining_meta')
               if not w[k].exists()]
    if missing:
        raise FileNotFoundError(
            f"cannot publish round {n}: work_{n}/ is missing {missing}")

    # a re-published round must not collide with leftovers at the final paths
    if f['training_data'].exists():
        shutil.rmtree(f['training_data'], ignore_errors=True)

    os.replace(w['training_data'], f['training_data'])
    os.replace(w['cache_state'], f['cache_state'])
    os.replace(w['mining_meta'], f['mining_meta'])

    tmp = root / f"ready_{n}.tmp"
    tmp.write_text(str(n))
    os.replace(tmp, f['ready'])          # marker last, atomically

    shutil.rmtree(w['work'], ignore_errors=True)
    return f


def write_ready_initial(root):
    """Mark the step-0 input committed. Call only after initial_data/, the initial
    metadata and cache_state_initial.pt are all durable."""
    root = Path(root)
    p = initial_paths(root)
    missing = [k for k in ('training_data', 'cache_state', 'mining_meta')
               if not p[k].exists()]
    if missing:
        raise FileNotFoundError(
            f"cannot write ready_initial: missing {missing}")
    tmp = root / "ready_initial.tmp"
    tmp.write_text("initial")
    os.replace(tmp, p['ready'])
    return p['ready']


# ---- recovery and retention ------------------------------------------------

def reap_orphans(root):
    """Delete artifacts of uncommitted rounds (N > latest committed).

    A crash between the `os.replace` sequence and the marker write leaves
    final-path artifacts with no `ready_N`. Those are NOT committed rounds, and the
    doc explicitly sanctions removing them: "They may delete or overwrite orphaned
    artifacts with larger round numbers." Miner startup only.

    Returns the sorted list of round numbers reaped.
    """
    root = Path(root)
    if not root.exists():
        return []
    committed = latest_committed_round(root)

    def _round_no(name, prefix, suffix=""):
        if not name.startswith(prefix) or not name.endswith(suffix):
            return None
        tail = name[len(prefix):len(name) - len(suffix)] if suffix else name[len(prefix):]
        return int(tail) if tail.isdigit() else None

    # every work_N dir is miner scratch and is always removable, committed or not
    for path in root.glob("work_*"):
        if _round_no(path.name, "work_") is not None:
            shutil.rmtree(path, ignore_errors=True)

    uncommitted = set()
    for pattern, prefix, suffix in (("training_data_*", "training_data_", ""),
                                    ("cache_state_*.pt", "cache_state_", ".pt"),
                                    ("mining_meta_*.json", "mining_meta_", ".json")):
        for path in root.glob(pattern):
            n = _round_no(path.name, prefix, suffix)
            if n is not None and n > committed:
                uncommitted.add(n)

    reaped = []
    for n in sorted(uncommitted):
        p = round_paths(root, n)
        shutil.rmtree(p['training_data'], ignore_errors=True)
        p['cache_state'].unlink(missing_ok=True)
        p['mining_meta'].unlink(missing_ok=True)
        reaped.append(n)
    return reaped


def prune_cache_states(root, keep):
    """Delete `cache_state_N.pt` for N <= newest_committed - keep.

    Cache states are the only large artifact (~250 MB at B_doc=32k, T=3) and the
    only thing pruned. `ready_N` markers are the commit log — removing one would
    un-commit a round and could move ``latest_committed_round`` BACKWARDS.
    `training_data_N/` stays because the miner cannot know which round the trainer
    is still consuming.

    Returns the sorted list of round numbers whose state was deleted.
    """
    root = Path(root)
    keep = max(int(keep), 1)
    committed = latest_committed_round(root)
    cutoff = committed - keep
    deleted = []
    for path in sorted(root.glob("cache_state_*.pt")):
        tail = path.name[len("cache_state_"):-len(".pt")]
        if tail.isdigit() and int(tail) <= cutoff:
            path.unlink(missing_ok=True)
            deleted.append(int(tail))
    return deleted


def resolve_cache_state(root):
    """Cache state to load at miner startup: newest COMMITTED round, else initial.

    Deliberately NOT "the highest-numbered cache_state_N.pt" — a crash can leave an
    unpublished `cache_state_5.pt` next to a committed `ready_3`, and loading the
    orphan would resume from a round the trainer never saw.

    Returns ``(path, round_no)``; ``round_no == 0`` means `cache_state_initial.pt`.
    Raises ``FileNotFoundError`` if neither exists.
    """
    root = Path(root)
    committed = latest_committed_round(root)
    if committed > 0:
        p = round_paths(root, committed)['cache_state']
        if p.exists():
            return p, committed
        raise FileNotFoundError(
            f"round {committed} is committed (ready_{committed} exists) but its "
            f"cache state {p} is missing — it may have been pruned too aggressively")
    p = initial_paths(root)['cache_state']
    if p.exists():
        return p, 0
    raise FileNotFoundError(f"no committed round and no initial cache state at {p}")


def resolve_training_data(root, n):
    """Data dir for round ``n``; ``n == 0`` means `initial_data/`."""
    return (initial_paths(root)['training_data'] if n == 0
            else round_paths(root, n)['training_data'])


def read_meta(root, n):
    """Parse `mining_meta_N.json` (or the initial one for ``n == 0``)."""
    p = (initial_paths(root)['mining_meta'] if n == 0
         else round_paths(root, n)['mining_meta'])
    if not p.exists():
        return {}
    return json.loads(p.read_text())


# ---- checkpoint selection --------------------------------------------------

def newest_valid_checkpoint(out_dir, exclude_step=-1):
    """Newest `checkpoint-N` with `optimizer.pt` present and N > ``exclude_step``.

    `optimizer.pt` is written last by the trainer, so its presence is the validity
    flag (ANCE pattern) — this is what stops the miner reading a half-written
    checkpoint. ``exclude_step`` is the step already mined, so the miner skips
    intermediate checkpoints between rounds instead of queueing stale work.

    Returns ``(step, path)`` or ``None``.
    """
    out_dir = Path(out_dir)
    if not out_dir.exists():
        return None
    best = None
    for ck in out_dir.glob("checkpoint-*"):
        tail = ck.name[len("checkpoint-"):]
        if not tail.isdigit():
            continue
        step = int(tail)
        if step <= exclude_step or not is_valid_checkpoint(ck):
            continue
        if best is None or step > best[0]:
            best = (step, ck)
    return best


def checkpoint_step(path):
    """Parse the optimizer step out of a `checkpoint-N` directory name."""
    name = Path(path).name
    tail = name[len("checkpoint-"):] if name.startswith("checkpoint-") else ""
    if not tail.isdigit():
        raise ValueError(f"not a checkpoint dir: {path}")
    return int(tail)
