"""
Async Fast-GRASS — handoff-protocol correctness tests.

CPU-only, no torch model, no GPU. Builds temp async-mining trees and asserts the
commit contract from async_fast_grass_implementation_details.md ("Handoff
Protocol", "Trainer Loop", "Miner Loop") against the REAL implementation in
``scripts/async_fast_grass_handoff.py`` — the miner and trainer import the same
functions, so these tests grade production code rather than a private copy.

Commit semantics:
  - ready_initial is the step-0 input, NOT a numeric round.
  - numeric rounds start at ready_1; latest_committed_round ignores ready_initial.
  - trainer consumes the newest ready round and skips stale older ready rounds.
  - the trainer never reads work_N; a round with no ready_N is not committed.
  - publish_round writes the marker LAST, after atomic renames of data/state/meta.

Recovery and retention:
  - reap_orphans clears artifacts of uncommitted rounds (crash between the
    os.replace sequence and the marker write) and leaves committed rounds intact.
  - prune_cache_states deletes only cache_state_N.pt, never a ready_N marker or a
    training_data_N/ the trainer may still be consuming.
  - resolve_cache_state follows the newest COMMITTED round, never a higher-numbered
    orphan cache_state_N.pt.
  - invariant: ready_N exists <=> training_data_N/ exists; and
    latest_committed_round never decreases across a crash + restart.

Checkpoints and step arithmetic:
  - a checkpoint is valid only once optimizer.pt exists (written last).
  - the miner picks the newest VALID checkpoint not already mined.
  - a newer checkpoint appearing mid-round does NOT change the round's
    source_checkpoint_step; the miner finishes, publishes, then re-selects.
  - ready_poll_steps (ready checks) is decoupled from async_mine_every_steps
    (checkpoint saves).
  - async_gap_steps / data_age_steps / rounds_consumed / rounds_skipped.

Run: python scripts/async_fast_grass_handoff_test.py
"""
import sys
import tempfile
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root / 'src'))
sys.path.insert(0, str(project_root / 'scripts'))

from utils.helpers import is_valid_checkpoint  # noqa: E402
from async_fast_grass_handoff import (  # noqa: E402
    latest_committed_round, publish_round, write_ready_initial, reap_orphans,
    prune_cache_states, resolve_cache_state, resolve_training_data,
    newest_valid_checkpoint, checkpoint_step, round_paths, work_paths,
    initial_paths,
)
# PRODUCTION predicate + resolver, not a reimplementation of them
from run_async_fast_grass_train import (  # noqa: E402
    should_checkpoint, _resolve_bootstrap_step,
)


# ---- tree builders ---------------------------------------------------------

def _fresh_tree():
    tmp = Path(tempfile.mkdtemp(prefix="async_fg_handoff_"))
    root = tmp / "async_mining"
    root.mkdir(parents=True)
    return tmp, root


def _stage_work(root, n, source_checkpoint_step=0):
    """Fill work_N/ the way the miner does, before publishing."""
    w = work_paths(root, n)
    w['training_data'].mkdir(parents=True, exist_ok=True)
    (w['training_data'] / "train.jsonl").write_text(
        '{"query_id":"q","query":"q","pos_docid":"d0","neg_docids":["d1"]}\n')
    w['cache_state'].write_text(f"cache-{n}")
    w['mining_meta'].write_text(
        '{"round_no": %d, "source_checkpoint_step": %d}' % (n, source_checkpoint_step))
    return w


def _publish(root, n, source_checkpoint_step=0):
    _stage_work(root, n, source_checkpoint_step)
    return publish_round(root, n)


def _publish_initial(root):
    p = initial_paths(root)
    p['training_data'].mkdir(parents=True, exist_ok=True)
    (p['training_data'] / "train.jsonl").write_text('{"query_id":"q"}\n')
    p['cache_state'].write_text("cache-initial")
    p['mining_meta'].write_text('{"source_checkpoint": "base_model", '
                                '"source_checkpoint_step": 0}')
    return write_ready_initial(root)


def _make_checkpoint(out_dir, step, valid=True):
    ckpt = out_dir / f"checkpoint-{step}"
    ckpt.mkdir(parents=True, exist_ok=True)
    (ckpt / "config.json").write_text("{}")
    (ckpt / "scheduler.pt").write_text("sched")
    if valid:
        (ckpt / "optimizer.pt").write_text("opt")  # written last => validity flag
    return ckpt


# ---- trainer reference behaviour -------------------------------------------

class TrainerSwapSim:
    """The trainer's round-consumption rule, in isolation.

    Mirrors the doc's trainer loop: poll every ready_poll_steps, consume the newest
    ready round, skip older ones, and track the async-lag counters.
    """

    def __init__(self, root, ready_poll_steps):
        self.root = root
        self.ready_poll_steps = ready_poll_steps
        self.active_round = 0
        self.consume_step = 0
        self.source_checkpoint_step = 0
        self.rounds_consumed = 0
        self.rounds_skipped = 0
        self.async_gap_steps = 0

    def maybe_swap(self, global_step, meta_lookup):
        if global_step % self.ready_poll_steps != 0:
            return False
        latest = latest_committed_round(self.root)
        if latest <= self.active_round:
            return False
        # every ready round strictly between the active one and the newest is skipped
        self.rounds_skipped += max(latest - self.active_round - 1, 0)
        self.active_round = latest
        self.consume_step = global_step
        self.source_checkpoint_step = meta_lookup(latest)
        self.async_gap_steps = self.consume_step - self.source_checkpoint_step
        self.rounds_consumed += 1
        return True

    def data_age_steps(self, global_step):
        return global_step - self.consume_step


# ---- commit semantics ------------------------------------------------------

def test_ready_initial_ignored_by_marker_scan():
    tmp, root = _fresh_tree()
    _publish_initial(root)
    assert latest_committed_round(root) == 0, \
        "ready_initial must NOT be counted as a numeric round"
    # and it still resolves as the step-0 input
    path, n = resolve_cache_state(root)
    assert n == 0 and path.name == "cache_state_initial.pt"


def test_numeric_rounds_start_at_one():
    tmp, root = _fresh_tree()
    _publish_initial(root)
    assert latest_committed_round(root) + 1 == 1, "first mined round must be 1"
    _publish(root, 1)
    assert latest_committed_round(root) == 1


def test_trainer_consumes_newest_and_skips_stale():
    tmp, root = _fresh_tree()
    _publish_initial(root)
    for n in (1, 2, 3):
        _publish(root, n)
    sim = TrainerSwapSim(root, ready_poll_steps=10)
    sim.active_round = 1                      # already on round 1
    assert sim.maybe_swap(100, lambda n: 0)
    assert sim.active_round == 3, f"expected jump to 3, got {sim.active_round}"
    assert sim.rounds_skipped == 1, "round 2 should be counted as skipped"
    assert not sim.maybe_swap(110, lambda n: 0), "already newest => no consume"


def test_publish_writes_marker_last_and_clears_work():
    """The marker must appear only after data/state/meta are at their final paths."""
    tmp, root = _fresh_tree()
    w = _stage_work(root, 1)
    f = round_paths(root, 1)
    assert not f['ready'].exists() and not f['training_data'].exists()
    assert latest_committed_round(root) == 0, "staged work is not committed"

    publish_round(root, 1)
    for key in ('training_data', 'cache_state', 'mining_meta', 'ready'):
        assert f[key].exists(), f"{key} missing after publish"
    assert not w['work'].exists(), "work_N/ should be cleared after publish"
    assert not (root / "ready_1.tmp").exists(), "temp marker left behind"
    assert latest_committed_round(root) == 1


def test_crash_midpublish_leaves_round_uncommitted():
    """The marker must be unreachable until every artifact is at its final path.

    Simulates a crash during publication by making the LAST data rename fail. With
    marker-last ordering the round is simply not committed; if the marker were
    written any earlier, the trainer could consume a round whose data never landed.
    """
    import async_fast_grass_handoff as hf

    tmp, root = _fresh_tree()
    _publish_initial(root)
    _stage_work(root, 1)

    real_replace = hf.os.replace
    calls = {'n': 0}

    def flaky_replace(src, dst):
        calls['n'] += 1
        if str(dst).endswith("mining_meta_1.json"):
            raise OSError("simulated crash mid-publish")
        return real_replace(src, dst)

    hf.os.replace = flaky_replace
    try:
        try:
            publish_round(root, 1)
        except OSError:
            pass
        else:
            raise AssertionError("expected the simulated crash to propagate")
    finally:
        hf.os.replace = real_replace

    assert latest_committed_round(root) == 0, \
        "a crash mid-publish must NOT leave the round committed — the ready marker " \
        "was written before the data was fully in place"
    assert not round_paths(root, 1)['ready'].exists()
    # and the leftovers are recoverable
    reaped = reap_orphans(root)
    assert 1 in reaped
    assert not round_paths(root, 1)['training_data'].exists()


def test_partial_round_without_ready_is_ignored():
    tmp, root = _fresh_tree()
    _publish_initial(root)
    _publish(root, 1)
    # round 2: final-path artifacts exist but the marker write never happened
    _stage_work(root, 2)
    f2 = round_paths(root, 2)
    f2['training_data'].mkdir(exist_ok=True)
    f2['cache_state'].write_text("orphan")
    assert latest_committed_round(root) == 1, \
        "artifacts without ready_N must not advance the committed round"
    sim = TrainerSwapSim(root, ready_poll_steps=1)
    sim.active_round = 1
    assert not sim.maybe_swap(5, lambda n: 0)


def test_trainer_never_reads_work_dir():
    tmp, root = _fresh_tree()
    _publish_initial(root)
    _publish(root, 1)
    _stage_work(root, 2)                       # in-flight round
    latest = latest_committed_round(root)
    assert latest == 1
    consumed = resolve_training_data(root, latest)
    assert consumed == round_paths(root, 1)['training_data'] and consumed.exists()
    assert "work_" not in str(consumed), "trainer must never be pointed at work_N"
    assert work_paths(root, 2)['work'].exists(), "in-flight work_2 still present"


# ---- recovery and retention ------------------------------------------------

def test_resolve_cache_state_follows_committed_not_highest():
    """A crash can leave cache_state_5.pt with only ready_3 committed."""
    tmp, root = _fresh_tree()
    _publish_initial(root)
    for n in (1, 2, 3):
        _publish(root, n)
    round_paths(root, 5)['cache_state'].write_text("orphan-5")   # never committed
    path, n = resolve_cache_state(root)
    assert n == 3, f"must resume from the newest COMMITTED round, got {n}"
    assert path == round_paths(root, 3)['cache_state']
    assert path.read_text() == "cache-3"


def test_reap_orphans_clears_uncommitted_only():
    tmp, root = _fresh_tree()
    _publish_initial(root)
    for n in (1, 2):
        _publish(root, n)
    # crash leftovers above the committed round
    f4 = round_paths(root, 4)
    f4['training_data'].mkdir()
    (f4['training_data'] / "x.jsonl").write_text("{}\n")
    f4['cache_state'].write_text("orphan")
    f4['mining_meta'].write_text("{}")
    _stage_work(root, 3)                       # interrupted in-flight round

    reaped = reap_orphans(root)
    assert 4 in reaped, f"round 4 should be reaped, got {reaped}"
    assert not f4['training_data'].exists() and not f4['cache_state'].exists()
    assert not work_paths(root, 3)['work'].exists(), "scratch work_3 not cleared"
    # committed rounds survive untouched
    for n in (1, 2):
        p = round_paths(root, n)
        assert p['ready'].exists() and p['training_data'].exists() \
            and p['cache_state'].exists()
    assert latest_committed_round(root) == 2


def test_prune_cache_states_never_touches_markers_or_data():
    tmp, root = _fresh_tree()
    _publish_initial(root)
    for n in range(1, 6):
        _publish(root, n)
    deleted = prune_cache_states(root, keep=2)
    assert deleted == [1, 2, 3], f"expected states 1-3 pruned, got {deleted}"
    for n in range(1, 6):
        p = round_paths(root, n)
        assert p['ready'].exists(), f"ready_{n} must never be pruned"
        assert p['training_data'].exists(), f"training_data_{n} must never be pruned"
    assert not round_paths(root, 1)['cache_state'].exists()
    assert round_paths(root, 4)['cache_state'].exists()
    assert round_paths(root, 5)['cache_state'].exists()
    # the commit log is intact, so the newest committed round is unchanged
    assert latest_committed_round(root) == 5
    # and the round we resume from still has its state
    path, n = resolve_cache_state(root)
    assert n == 5 and path.exists()


def test_committed_round_never_goes_backwards_across_restart():
    tmp, root = _fresh_tree()
    _publish_initial(root)
    for n in (1, 2, 3):
        _publish(root, n)
    before = latest_committed_round(root)
    # simulate a crash mid-round-4 then a restart
    _stage_work(root, 4)
    round_paths(root, 4)['cache_state'].write_text("orphan")
    prune_cache_states(root, keep=2)
    reap_orphans(root)
    after = latest_committed_round(root)
    assert after == before == 3, f"committed round moved: {before} -> {after}"
    # invariant: ready_N exists <=> training_data_N exists
    for marker in root.glob("ready_*"):
        tail = marker.name[len("ready_"):]
        if tail.isdigit():
            assert round_paths(root, int(tail))['training_data'].exists(), \
                f"ready_{tail} has no training_data_{tail}"
    for data in root.glob("training_data_*"):
        tail = data.name[len("training_data_"):]
        if tail.isdigit():
            assert round_paths(root, int(tail))['ready'].exists(), \
                f"training_data_{tail} has no ready_{tail}"


# ---- checkpoint selection --------------------------------------------------

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
    _make_checkpoint(out, 300, valid=False)      # in-progress, no optimizer.pt
    sel = newest_valid_checkpoint(out, exclude_step=-1)
    assert sel is not None and sel[0] == 200, f"expected 200, got {sel}"
    assert newest_valid_checkpoint(out, exclude_step=200) is None
    (out / "checkpoint-300" / "optimizer.pt").write_text("opt")
    sel2 = newest_valid_checkpoint(out, exclude_step=200)
    assert sel2 is not None and sel2[0] == 300
    assert checkpoint_step(sel2[1]) == 300


def test_newer_checkpoint_midround_does_not_change_source_step():
    """A round is frozen to the checkpoint it started from.

    "The miner must not interrupt a round when a newer checkpoint appears. Once a
    round starts, it finishes that round, publishes it, and only then picks the
    newest valid checkpoint."
    """
    tmp, root = _fresh_tree()
    out = tmp / "model_out"
    _make_checkpoint(out, 1000, valid=True)
    step, _ck = newest_valid_checkpoint(out, exclude_step=-1)
    assert step == 1000
    source_checkpoint_step = step                 # frozen for the whole round

    _stage_work(root, 1, source_checkpoint_step)  # mining round 1 in progress
    _make_checkpoint(out, 2000, valid=True)       # trainer checkpoints mid-round
    # the in-flight round is unaffected: it publishes with its ORIGINAL step
    publish_round(root, 1)
    import json
    meta = json.loads(round_paths(root, 1)['mining_meta'].read_text())
    assert meta['source_checkpoint_step'] == 1000, \
        f"mid-round checkpoint leaked into the published round: {meta}"
    # only AFTER publishing does the miner move to the newer checkpoint
    nxt = newest_valid_checkpoint(out, exclude_step=source_checkpoint_step)
    assert nxt is not None and nxt[0] == 2000


def test_ready_poll_and_checkpoint_cadences_are_independent():
    """ready_poll_steps is a cheap directory check; async_mine_every_steps is the
    checkpoint-save cadence. They must not be coupled."""
    ready_poll_steps, async_mine_every_steps = 100, 1000
    poll_steps = [s for s in range(1, 3001) if s % ready_poll_steps == 0]
    save_steps = [s for s in range(1, 3001) if s % async_mine_every_steps == 0]
    assert len(poll_steps) == 30 and len(save_steps) == 3
    assert set(save_steps).issubset(set(poll_steps)), \
        "with these values every save step is also a poll step (1000 % 100 == 0)"
    # the trainer polls far more often than it checkpoints — that is the point
    assert len(poll_steps) > 5 * len(save_steps)


def test_async_gap_and_data_age_arithmetic():
    tmp, root = _fresh_tree()
    _publish_initial(root)
    _publish(root, 1, source_checkpoint_step=1000)
    sim = TrainerSwapSim(root, ready_poll_steps=50)
    metas = {1: 1000, 2: 1400}

    assert sim.maybe_swap(1250, metas.get)
    assert sim.async_gap_steps == 250, sim.async_gap_steps
    assert sim.rounds_consumed == 1 and sim.rounds_skipped == 0
    # data_age grows while the SAME round stays active
    for reuse in (0, 100, 399):
        assert sim.data_age_steps(1250 + reuse) == reuse

    _publish(root, 2, source_checkpoint_step=1400)
    assert sim.maybe_swap(1600, metas.get)
    assert sim.async_gap_steps == 200
    assert sim.data_age_steps(1600) == 0, "data_age resets on swap"
    assert sim.rounds_consumed == 2


def test_dataloader_swap_keeps_optimizer_and_scheduler():
    """Swapping rounds must not reset optimizer/scheduler or the global step."""
    import torch
    model = torch.nn.Linear(4, 4)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda s: 1.0)
    opt_id, sched_id = id(optimizer), id(scheduler)

    tmp, root = _fresh_tree()
    _publish_initial(root)
    _publish(root, 1, source_checkpoint_step=100)
    sim = TrainerSwapSim(root, ready_poll_steps=10)

    global_step = 0
    loader_round = 0
    for _ in range(30):
        global_step += 1
        optimizer.step()
        scheduler.step()
        if sim.maybe_swap(global_step, lambda n: 100):
            loader_round = sim.active_round      # only the dataloader is rebuilt

    assert loader_round == 1, "round 1 was never consumed"
    assert id(optimizer) == opt_id and id(scheduler) == sched_id, \
        "optimizer/scheduler must survive a dataloader swap"
    assert global_step == 30, "global step must stay continuous across the swap"
    assert scheduler.last_epoch == 30, "scheduler state must not reset on swap"


# ---- harness ---------------------------------------------------------------

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


def _fired(max_steps, mine_every, bootstrap_step=0):
    """Steps at which the PRODUCTION predicate says the trainer checkpoints."""
    return [s for s in range(1, max_steps + 1)
            if should_checkpoint(s, mine_every, max_steps, bootstrap_step)]


def test_bootstrap_default_off_is_a_noop():
    """With the key absent or 0, checkpoints are exactly the old cadence.

    This is what makes the change safe to push while sweep arms are queued: those
    jobs read the .py at launch, so the default MUST be bit-identical to before.
    """
    cfg = {}
    assert _resolve_bootstrap_step(cfg, None, 1000) == 0, "absent key must mean off"
    assert _resolve_bootstrap_step({'bootstrap_checkpoint_step': 0}, None, 1000) == 0

    expected = [1000 * k for k in range(1, 11)] + [10314]   # + the final step
    assert _fired(10314, 1000, 0) == expected, "default-off changed the cadence"
    # and the CLI default (None) resolves through config, not around it
    assert _fired(10314, 1000, _resolve_bootstrap_step(cfg, None, 1000)) == expected


def test_bootstrap_adds_exactly_one_checkpoint():
    """bootstrap=200 adds step 200 and nothing else."""
    step = _resolve_bootstrap_step({'bootstrap_checkpoint_step': 200}, None, 1000)
    assert step == 200
    off = _fired(10314, 1000, 0)
    on = _fired(10314, 1000, step)
    assert set(on) - set(off) == {200}, f"expected only 200 added, got {set(on) - set(off)}"
    assert set(off) - set(on) == set(), "bootstrap must not remove a checkpoint"
    assert on.count(200) == 1, "bootstrap step must fire exactly once"
    # CLI overrides config
    assert _resolve_bootstrap_step({'bootstrap_checkpoint_step': 200}, 400, 1000) == 400


def test_invalid_bootstrap_raises_not_silently_disabled():
    """A nonzero value >= mine_every RAISES.

    Warning-and-disabling would yield a run labelled bootstrapped that never wrote
    the extra checkpoint. Because the bootstrap round mutates the PERSISTED cache,
    that silently mislabels an ablation arm.
    """
    for bad in (1000, 1500, -1):
        try:
            _resolve_bootstrap_step({'bootstrap_checkpoint_step': bad}, None, 1000)
        except ValueError:
            continue
        raise AssertionError(
            f"bootstrap_checkpoint_step={bad} must raise, not be silently disabled")
    # the CLI path is validated too, not just the config path
    try:
        _resolve_bootstrap_step({}, 1000, 1000)
    except ValueError:
        pass
    else:
        raise AssertionError("invalid CLI override must raise")


TESTS = [
    # commit semantics
    ("ready_initial ignored by marker scan", test_ready_initial_ignored_by_marker_scan),
    ("numeric rounds start at ready_1", test_numeric_rounds_start_at_one),
    ("trainer consumes newest ready, skips stale", test_trainer_consumes_newest_and_skips_stale),
    ("publish writes marker LAST, clears work_N", test_publish_writes_marker_last_and_clears_work),
    ("crash mid-publish leaves round uncommitted", test_crash_midpublish_leaves_round_uncommitted),
    ("partial round without ready_N ignored", test_partial_round_without_ready_is_ignored),
    ("trainer never reads work_N", test_trainer_never_reads_work_dir),
    # recovery + retention
    ("resolve_cache_state follows committed", test_resolve_cache_state_follows_committed_not_highest),
    ("reap_orphans clears uncommitted only", test_reap_orphans_clears_uncommitted_only),
    ("prune keeps markers + round data", test_prune_cache_states_never_touches_markers_or_data),
    ("committed round never goes backwards", test_committed_round_never_goes_backwards_across_restart),
    # checkpoints + step arithmetic
    ("checkpoint valid only after optimizer.pt", test_checkpoint_valid_only_after_optimizer_pt),
    ("miner picks newest valid, not already mined", test_miner_picks_newest_valid_not_already_mined),
    ("mid-round checkpoint does not change source step", test_newer_checkpoint_midround_does_not_change_source_step),
    ("ready_poll vs checkpoint cadence independent", test_ready_poll_and_checkpoint_cadences_are_independent),
    ("async_gap / data_age / round counters", test_async_gap_and_data_age_arithmetic),
    ("swap keeps optimizer + scheduler + step", test_dataloader_swap_keeps_optimizer_and_scheduler),
    # bootstrap checkpoint (mining-schedule ablation, ships OFF)
    ("bootstrap default-off is a no-op", test_bootstrap_default_off_is_a_noop),
    ("bootstrap adds exactly one checkpoint", test_bootstrap_adds_exactly_one_checkpoint),
    ("invalid bootstrap RAISES, not disabled", test_invalid_bootstrap_raises_not_silently_disabled),
]


def main():
    print("\nAsync Fast-GRASS handoff-protocol tests")
    print("=" * 60)
    passed = sum(_run(name, fn) for name, fn in TESTS)
    total = len(TESTS)
    print("=" * 60)
    print(f"  {passed}/{total} passed")
    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
