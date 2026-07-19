"""
Async Fast-GRASS — cache-maintenance step-semantics tests (spec item 5).

CPU-only, deterministic, synthetic. Validates the invariant that async in-round
maintenance must use the FROZEN checkpoint's optimizer step as model time
(async_fast_grass_implementation_details.md, "Cache Maintenance Semantics"):

    age = source_checkpoint_step - last_refreshed_step
    rho = linear_decay(..., source_checkpoint_step / total_steps)
    on refresh/replace: last_refreshed_step := source_checkpoint_step

Because a mining round runs MANY maintenance intervals against ONE checkpoint,
every ``cache.maintain(...)`` call in that round must pass the SAME
``source_checkpoint_step`` even though utility state (selected_indicator ->
utility_ema, intervals_since_selected) advances per interval. Passing a
miner-local counter (batch index, mined-query count, interval index) would
corrupt cache age, the rho budget, and ``last_refreshed_step``.

Reuses the NegativeCache test fixtures from fast_grass_test.py.

Covered:
  - N maintenance calls in one round, all step=source_checkpoint_step:
    every slot whose last_refreshed_step changed == source_checkpoint_step.
  - utility advances each interval while step stays fixed (utility_ema of a
    repeatedly-selected slot rises; its lifetime_selected_count increments once
    per interval, proving one utility update per maintenance call).
  - last_refreshed_step never exceeds source_checkpoint_step (age >= 0).
  - B_doc invariant holds after every interval.
  - CONTRAST: passing a miner-local counter as step corrupts last_refreshed_step
    (drifts far below model time) — demonstrating why the counter must not be used.

Run: python scripts/async_fast_grass_cache_semantics_test.py
"""
import sys
from pathlib import Path

import torch

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root / 'src'))
sys.path.insert(0, str(project_root / 'scripts'))

from fast_grass_test import (  # noqa: E402
    make_cfg, make_cache, GradMockModel, MockTokenizer, _rand_unit, DEVICE,
)

SOURCE_CKPT_STEP = 500       # weights the mining round was frozen from
N_INTERVALS = 4              # maintenance intervals inside the one round
OVER_SLOTS = [10, 11, 12, 13]  # forced over-age each interval => guaranteed action
SLOT_SELECTED = 2            # selected every interval => utility_ema must rise


def _mk():
    cfg = make_cfg(uncertainty='mcdp', B_doc=20)  # teacher-free (Z_teacher None)
    cache, cfg, c_ids, embs, corpus_lookup = make_cache(cfg, n_corpus=30, dim=8)
    student = GradMockModel(hidden=8).eval()
    tok = MockTokenizer()
    reservoir = {'q_student': _rand_unit(6, 8, 3), 'q_teacher': None,
                 'qids': [f"rq{i}" for i in range(6)]}
    return cache, cfg, c_ids, corpus_lookup, student, tok, reservoir


def _one_interval(cache, cfg, c_ids, corpus_lookup, student, tok, reservoir, step):
    """Simulate one in-round maintenance interval: mark selections + over-age,
    snapshot last_refreshed_step, run maintain(step), return changed-slot steps."""
    # selection since last interval (the selected slot stays "useful")
    cache.record_selection(torch.tensor([[SLOT_SELECTED]], device=DEVICE))
    # force a disjoint set over-age so maintenance always acts this interval
    for s in OVER_SLOTS:
        cache.last_refreshed_step[s] = 0
    before = cache.last_refreshed_step.clone()
    cache.maintain(student, None, tok, corpus_lookup, c_ids, reservoir,
                   step, cfg, DEVICE, qrels_dict={})
    changed = (cache.last_refreshed_step != before)
    return changed, cache.last_refreshed_step.clone()


def test_all_intervals_use_same_source_checkpoint_step():
    cache, cfg, c_ids, corpus_lookup, student, tok, reservoir = _mk()
    for _ in range(N_INTERVALS):
        changed, lrs = _one_interval(cache, cfg, c_ids, corpus_lookup, student,
                                     tok, reservoir, SOURCE_CKPT_STEP)
        assert changed.any(), "no maintenance action fired this interval"
        # every slot touched this interval must be stamped with the checkpoint step
        assert torch.all(lrs[changed] == SOURCE_CKPT_STEP), \
            "refreshed/replaced slots must be stamped source_checkpoint_step"
        # model-time age must be non-negative: last_refreshed_step <= step
        assert torch.all(cache.last_refreshed_step <= SOURCE_CKPT_STEP), \
            "last_refreshed_step must never exceed model time"
        # B_doc invariant
        assert len(cache.docids) == cache.B_doc == len(set(cache.docids))


def test_utility_advances_while_step_fixed():
    cache, cfg, c_ids, corpus_lookup, student, tok, reservoir = _mk()
    util_selected = []
    for _ in range(N_INTERVALS):
        _one_interval(cache, cfg, c_ids, corpus_lookup, student, tok, reservoir,
                      SOURCE_CKPT_STEP)
        util_selected.append(float(cache.utility_ema[SLOT_SELECTED]))
    # selected-every-interval slot: utility_ema strictly increases across intervals
    for a, b in zip(util_selected, util_selected[1:]):
        assert b > a, f"utility_ema of a repeatedly-selected slot must rise: {util_selected}"
    # one utility update per maintenance call: lifetime_selected_count of the
    # slot selected every interval equals the number of intervals (it stays
    # high-utility, so it is never replaced/reset).
    assert int(cache.lifetime_selected_count[SLOT_SELECTED]) == N_INTERVALS, \
        (f"expected {N_INTERVALS} utility updates, got "
         f"{int(cache.lifetime_selected_count[SLOT_SELECTED])}")


def test_last_refreshed_step_tied_to_checkpoint_not_counter():
    """CONTRAST: a miner-local interval counter as `step` corrupts model time."""
    # correct run: fixed source_checkpoint_step
    c1, cfg1, ids1, look1, stu1, tok1, res1 = _mk()
    for _ in range(N_INTERVALS):
        _one_interval(c1, cfg1, ids1, look1, stu1, tok1, res1, SOURCE_CKPT_STEP)
    assert int(c1.last_refreshed_step.max()) == SOURCE_CKPT_STEP

    # wrong run: pass an incrementing miner-local counter (1,2,3,4)
    c2, cfg2, ids2, look2, stu2, tok2, res2 = _mk()
    for interval_idx in range(1, N_INTERVALS + 1):
        _one_interval(c2, cfg2, ids2, look2, stu2, tok2, res2, interval_idx)
    # corruption is visible: stamped model time is tiny (the counter), not ~500,
    # so age = source_checkpoint_step - last_refreshed_step would be hugely wrong.
    assert int(c2.last_refreshed_step.max()) <= N_INTERVALS, \
        "counter-as-step leaves last_refreshed_step at counter scale (corrupted)"
    assert int(c1.last_refreshed_step.max()) - int(c2.last_refreshed_step.max()) > 100, \
        "correct vs counter runs must diverge in stamped model time"


def test_interval_budget_uses_checkpoint_step():
    """rho/budget must be evaluated at source_checkpoint_step, not a counter."""
    cache, cfg, *_ = _mk()
    b_ckpt = cache._interval_budget(SOURCE_CKPT_STEP, cfg)
    b_counter = cache._interval_budget(1, cfg)  # what a miner-local counter would give
    # progress differs (500/1000 vs 1/1000) => rho differs => budget differs
    assert b_ckpt != b_counter, \
        "budget at checkpoint step must differ from budget at a tiny counter"
    assert b_ckpt >= 0 and b_counter >= 0


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
    ("all intervals stamp source_checkpoint_step", test_all_intervals_use_same_source_checkpoint_step),
    ("utility advances while step fixed", test_utility_advances_while_step_fixed),
    ("last_refreshed_step tied to ckpt, not counter", test_last_refreshed_step_tied_to_checkpoint_not_counter),
    ("interval budget uses checkpoint step", test_interval_budget_uses_checkpoint_step),
]


def main():
    print("\nAsync Fast-GRASS cache-semantics tests")
    print("=" * 55)
    passed = sum(_run(name, fn) for name, fn in TESTS)
    total = len(TESTS)
    print("=" * 55)
    print(f"  {passed}/{total} passed")
    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
