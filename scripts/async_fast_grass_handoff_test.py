"""
Async Fast-GRASS — handoff-protocol correctness tests (spec item 4).

CPU-only, no torch model, no GPU. Builds a temp async-mining handoff tree and
asserts the ready-marker / checkpoint-selection contract from
async_fast_grass_implementation_details.md ("Handoff Protocol", "Trainer Loop",
"Miner Loop"), using the REAL helpers the implementation will use:
``get_latest_marker_no`` and ``is_valid_checkpoint`` from src/utils/helpers.py.

Covered:
  - ready_initial is handled separately (step-0 input), NOT a numeric round.
  - numeric rounds start at ready_1.
  - get_latest_marker_no(..., "ready_") ignores ready_initial.
  - trainer consumes the newest ready round and skips stale older ready rounds.
  - trainer never reads work_N (only ready_N gates consumption).
  - a partial round (training_data_N/work_N present but no ready_N) is ignored.
  - a checkpoint is valid only after optimizer.pt exists.
  - miner selects the newest VALID checkpoint not already mined (skips invalid/older).
  - async_gap_steps / data_age_steps arithmetic.

DEFERRED TO PHASE 1 (audited, deliberately not covered here — none of these change
a Phase-0 timing number, and several can only be tested against a real miner):
  - publish ORDER/atomicity: write work_N/* -> os.replace(work_N/training_data,
    training_data_N) -> os.replace cache_state -> os.replace mining_meta ->
    write ready_N.tmp -> os.replace(ready_N.tmp, ready_N). ``_publish_round``
    below writes straight to the final paths, so it exercises the ready-marker
    contract but NOT the crash-safety of publication.
  - restart resolution: load the cache state of the newest COMMITTED ready round,
    never merely the highest-numbered cache_state_N.pt (a crash can leave
    unpublished final-path artifacts with no ready marker).
  - miner must not interrupt a round when a newer checkpoint appears mid-round;
    source_checkpoint_step stays pinned until the round is published.
  - ready_poll_steps (= logging_steps) is the ready-check cadence and is decoupled
    from the async_mine_every_steps checkpoint-save cadence.
  - rounds_consumed / rounds_skipped running counters.
  - dataloader swap without optimizer/scheduler reset; global_step continuous.

Run: python scripts/async_fast_grass_handoff_test.py
"""
import sys
import tempfile
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root / 'src'))

from utils.helpers import get_latest_marker_no, is_valid_checkpoint


# ---- handoff-tree builders (mirror the impl-details "Handoff Protocol") -----

def _fresh_tree():
    tmp = Path(tempfile.mkdtemp(prefix="async_fg_handoff_"))
    root = tmp / "async_mining"
    root.mkdir(parents=True)
    return tmp, root


def _publish_round(root, n, with_ready=True):
    """Publish round N the way the miner does: training_data_N/ then ready_N last."""
    (root / f"training_data_{n}").mkdir(exist_ok=True)
    (root / f"training_data_{n}" / "train.jsonl").write_text('{"query_id":"q"}\n')
    (root / f"cache_state_{n}.pt").write_text("cache")
    (root / f"mining_meta_{n}.json").write_text('{"round_no": %d}' % n)
    if with_ready:
        (root / f"ready_{n}").write_text(str(n))


def _publish_initial(root):
    (root / "initial_data").mkdir(exist_ok=True)
    (root / "initial_data" / "train.jsonl").write_text('{"query_id":"q"}\n')
    (root / "cache_state_initial.pt").write_text("cache")
    (root / "ready_initial").write_text("initial")


def _write_partial_work(root, n):
    """An in-progress round: work_N/ + training_data_N/ present, but NO ready_N."""
    (root / f"work_{n}").mkdir(exist_ok=True)
    (root / f"work_{n}" / "round.jsonl.tmp").write_text('{"partial":true}\n')
    (root / f"training_data_{n}").mkdir(exist_ok=True)
    (root / f"training_data_{n}" / "train.jsonl").write_text('{"partial":true}\n')


def _make_checkpoint(out_dir, step, valid=True):
    ckpt = out_dir / f"checkpoint-{step}"
    ckpt.mkdir(parents=True, exist_ok=True)
    (ckpt / "config.json").write_text("{}")
    (ckpt / "scheduler.pt").write_text("sched")
    if valid:
        (ckpt / "optimizer.pt").write_text("opt")  # written last => validity flag
    return ckpt


# ---- reference selection logic (what the trainer / miner will implement) ----

def newest_valid_checkpoint(out_dir, exclude_step=-1):
    """Miner rule: newest checkpoint-N with optimizer.pt present and N > exclude_step."""
    best = None
    for ck in Path(out_dir).glob("checkpoint-*"):
        tail = ck.name[len("checkpoint-"):]
        if not tail.isdigit():
            continue
        step = int(tail)
        if step <= exclude_step:
            continue
        if not is_valid_checkpoint(ck):
            continue
        if best is None or step > best[0]:
            best = (step, ck)
    return best  # (step, path) or None


def trainer_consume(root, active_round):
    """Trainer rule: consume newest ready round if strictly newer than active.
    Returns (new_active_round, consumed_bool, skipped_rounds)."""
    latest = get_latest_marker_no(root, prefix="ready_")
    if latest > active_round:
        skipped = list(range(active_round + 1, latest))  # older ready rounds jumped over
        return latest, True, skipped
    return active_round, False, []


# ---- tests -----------------------------------------------------------------

def test_ready_initial_ignored_by_marker_scan():
    tmp, root = _fresh_tree()
    _publish_initial(root)
    assert get_latest_marker_no(root, prefix="ready_") == 0, \
        "ready_initial must NOT be counted as a numeric round"


def test_numeric_rounds_start_at_one():
    tmp, root = _fresh_tree()
    _publish_initial(root)
    _publish_round(root, 1)
    assert get_latest_marker_no(root, prefix="ready_") == 1
    # miner's next round number after initial only = 0 + 1 = 1
    _t, root2 = _fresh_tree()[0], _fresh_tree()[1]
    _publish_initial(root2)
    assert get_latest_marker_no(root2, prefix="ready_") + 1 == 1


def test_trainer_consumes_newest_and_skips_stale():
    tmp, root = _fresh_tree()
    _publish_initial(root)
    for n in (1, 2, 3):
        _publish_round(root, n)
    # trainer sitting on round 1 jumps straight to 3, skipping 2
    active, consumed, skipped = trainer_consume(root, active_round=1)
    assert consumed and active == 3, f"expected jump to 3, got {active}"
    assert skipped == [2], f"expected to skip [2], got {skipped}"
    # already newest => no consume
    active2, consumed2, _ = trainer_consume(root, active_round=3)
    assert not consumed2 and active2 == 3


def test_trainer_never_reads_work_dir():
    tmp, root = _fresh_tree()
    _publish_initial(root)
    _publish_round(root, 1)
    _publish_round(root, 2)
    _write_partial_work(root, 3)  # work_3 + training_data_3 but NO ready_3
    latest = get_latest_marker_no(root, prefix="ready_")
    assert latest == 2, f"partial work_3 must not advance latest ready; got {latest}"
    # the consumed data path is training_data_2, never work_3
    consumed_path = root / f"training_data_{latest}"
    assert consumed_path.exists()
    assert (root / "work_3").exists() and not (root / "ready_3").exists()


def test_partial_round_without_ready_is_ignored():
    tmp, root = _fresh_tree()
    _publish_initial(root)
    _publish_round(root, 1, with_ready=True)
    _publish_round(root, 2, with_ready=False)  # files present, ready_2 missing
    assert get_latest_marker_no(root, prefix="ready_") == 1
    active, consumed, _ = trainer_consume(root, active_round=1)
    assert not consumed and active == 1, "must not consume a round with no ready_N"


def test_checkpoint_valid_only_after_optimizer_pt():
    tmp, _root = _fresh_tree()
    out = tmp / "model_out"
    ck = _make_checkpoint(out, 100, valid=False)
    assert not is_valid_checkpoint(ck), "no optimizer.pt => invalid"
    (ck / "optimizer.pt").write_text("opt")
    assert is_valid_checkpoint(ck), "optimizer.pt present => valid"


def test_miner_picks_newest_valid_not_already_mined():
    tmp, _root = _fresh_tree()
    out = tmp / "model_out"
    _make_checkpoint(out, 100, valid=True)
    _make_checkpoint(out, 200, valid=True)
    _make_checkpoint(out, 300, valid=False)  # in-progress, no optimizer.pt
    # newest valid overall is 200 (300 invalid)
    sel = newest_valid_checkpoint(out, exclude_step=-1)
    assert sel is not None and sel[0] == 200, f"expected 200, got {sel}"
    # already mined 200 => nothing newer valid => None (300 still invalid)
    assert newest_valid_checkpoint(out, exclude_step=200) is None
    # 300 completes => now selectable
    (out / "checkpoint-300" / "optimizer.pt").write_text("opt")
    sel2 = newest_valid_checkpoint(out, exclude_step=200)
    assert sel2 is not None and sel2[0] == 300


def test_async_gap_and_data_age_arithmetic():
    # source_checkpoint_step: weights the round was mined from
    # consume_step: trainer step where the round goes active
    # async_gap_steps = consume_step - source_checkpoint_step (fixed at consume)
    # data_age_steps grows while the SAME round stays active
    source_checkpoint_step = 1000
    consume_step = 1250
    async_gap = consume_step - source_checkpoint_step
    assert async_gap == 250
    for reuse in range(0, 400):
        cur_step = consume_step + reuse
        data_age = cur_step - consume_step
        assert data_age == reuse
    # when a newer round is consumed, gap is recomputed and data_age resets
    next_source, next_consume = 1400, 1600
    assert next_consume - next_source == 200
    assert (next_consume - next_consume) == 0  # data_age resets on swap


# ---- harness (mirrors fast_grass_test.py) ----------------------------------

def _run(name, fn):
    print(f"  {name} ...", end=' ', flush=True)
    try:
        fn()
        print("PASS")
        return True
    except AssertionError as e:
        print(f"FAIL — {e}")
        return False
    except Exception as e:
        print(f"ERROR — {type(e).__name__}: {e}")
        return False


TESTS = [
    ("ready_initial ignored by get_latest_marker_no", test_ready_initial_ignored_by_marker_scan),
    ("numeric rounds start at ready_1", test_numeric_rounds_start_at_one),
    ("trainer consumes newest ready, skips stale", test_trainer_consumes_newest_and_skips_stale),
    ("trainer never reads work_N", test_trainer_never_reads_work_dir),
    ("partial round without ready_N ignored", test_partial_round_without_ready_is_ignored),
    ("checkpoint valid only after optimizer.pt", test_checkpoint_valid_only_after_optimizer_pt),
    ("miner picks newest valid, not already mined", test_miner_picks_newest_valid_not_already_mined),
    ("async_gap_steps / data_age_steps arithmetic", test_async_gap_and_data_age_arithmetic),
]


def main():
    print("\nAsync Fast-GRASS handoff-protocol tests")
    print("=" * 55)
    passed = sum(_run(name, fn) for name, fn in TESTS)
    total = len(TESTS)
    print("=" * 55)
    print(f"  {passed}/{total} passed")
    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
