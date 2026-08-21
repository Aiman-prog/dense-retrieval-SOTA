"""
Async Fast-GRASS — cache persistence tests.

CPU-only, deterministic. Covers ``NegativeCache.save_state`` / ``load_state``, the
`cached_mcdp_v1` state that lets a crashed miner resume mid-run
(async_fast_grass_implementation_details.md, "Cache State And Initialization").

Covered:
  - round trip preserves docids, Z_mc, Z_mean/Z_student, utility + selection
    history, and last_refreshed_step.
  - Z_student stays ALIASED to Z_mean after a reload (no third bank, and the
    persisted mean is restored bit-exactly rather than recomputed).
  - the reload reproduces the NEXT RNG-driven cache decision, not merely the
    current embeddings — both the NumPy bit generator (uniform candidate sampling)
    and the torch generator (Gumbel selection) are restored.
  - schema / T / B_doc / dim mismatches are rejected BEFORE any device transfer,
    so an incompatible state never allocates GPU memory.
  - a reloaded cache keeps mining and maintaining correctly.
  - R_doc is deferred: the state carries no registry entries, and a reloaded cache
    still admits nothing.

Run: python tests/async_fast_grass_persistence_test.py
"""
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root / 'src'))
sys.path.insert(0, str(project_root / 'scripts'))

from utils.negative_cache import NegativeCache  # noqa: E402
from fast_grass_test import make_cfg, DropoutMockModel, MockTokenizer, DEVICE  # noqa: E402
from async_fast_grass_cached_mcdp import (  # noqa: E402
    mine_batch_cached_mcdp, maintain_interval_cached_mcdp, QueryMCReservoir,
    encode_queries_mc,
)

DIM, T, N_CORPUS, B_DOC = 16, 3, 60, 20


def _mk():
    cfg = make_cfg(uncertainty='cached_mcdp', B_doc=B_DOC, T=T, lambda_val=0.5,
                   batch_size=8, mc_batch_size=8, miner_mc_batch_size=8,
                   passage_max_len=8, query_max_len=8)
    c_ids = [f"d{i}" for i in range(N_CORPUS)]
    lookup = {d: f"document {d} body text" for d in c_ids}
    embs = np.random.default_rng(0).standard_normal((N_CORPUS, DIM)).astype('float32')
    model = DropoutMockModel(hidden=DIM, p=0.3)
    tok = MockTokenizer()
    cache, _ = NegativeCache.init_cached_mcdp(embs, c_ids, lookup, model, tok, cfg,
                                              DEVICE, dim=DIM)
    return cache, cfg, c_ids, lookup, model, tok


def _dirty(cache):
    """Give the cache non-trivial state so a round trip has something to preserve."""
    cache.record_selection(torch.tensor([[3], [7]], device=DEVICE))
    cache.utility_ema[5] = 0.42
    cache.peak_utility_ema[5] = 0.77
    cache.lifetime_selected_count[5] = 9
    cache.intervals_since_selected[2] = 4
    cache.last_refreshed_step[:] = 123
    cache.last_refreshed_step[1] = 7
    cache.cache_score_pairs = 4242


def _reservoir(model, tok, cfg, n=6):
    res = QueryMCReservoir(cfg['recent_query_reservoir_size'])
    q_mc, _ = encode_queries_mc(model, tok, [f"reservoir q{i}" for i in range(n)],
                                T, DEVICE, cfg)
    res.add(q_mc, [f"rq{i}" for i in range(n)])
    return res.get()


# ---- round trip ------------------------------------------------------------

def test_round_trip_preserves_all_state():
    cache, cfg, *_ = _mk()
    _dirty(cache)
    with tempfile.TemporaryDirectory() as td:
        path = cache.save_state(Path(td) / "cache_state.pt")
        c2 = NegativeCache.load_state(path, cfg, DEVICE, expect_T=T,
                                      expect_B_doc=B_DOC, expect_dim=DIM)
    assert c2.docids == cache.docids, "docids not preserved"
    assert c2.docid_to_slot == cache.docid_to_slot, "slot map not rebuilt"
    assert torch.equal(c2.Z_mc, cache.Z_mc), "Z_mc not preserved"
    assert torch.equal(c2.Z_student, cache.Z_student), "Z_mean/Z_student not preserved"
    for key in ('utility_ema', 'peak_utility_ema', 'selected_indicator',
                'selected_count_recent', 'lifetime_selected_count',
                'intervals_since_selected', 'last_refreshed_step'):
        assert torch.equal(getattr(c2, key), getattr(cache, key)), f"{key} not preserved"
    assert c2.cache_score_pairs == cache.cache_score_pairs
    assert c2.T == T and c2.B_doc == B_DOC and c2.dim == DIM
    assert c2.Z_teacher is None, "cached-MCDP must stay teacher-free"


def test_inconsistent_z_mean_is_rejected():
    """Z_mc is AUTHORITATIVE; a persisted Z_mean that disagrees is corruption.

    Z_mean is derived state and is aliased to Z_student, which ``cheap_scores`` and
    ``_plan_actions`` read. Loading a mean that no longer summarises Z_mc would have
    those score against document states the MC bank does not contain. Reject, do not
    silently restore.
    """
    cache, cfg, *_ = _mk()
    cache.Z_student[0] = torch.zeros_like(cache.Z_student[0])   # corrupt the mean
    with tempfile.TemporaryDirectory() as td:
        path = cache.save_state(Path(td) / "s.pt")
        try:
            NegativeCache.load_state(path, cfg, DEVICE, expect_T=T)
        except ValueError as e:
            assert 'Z_mean' in str(e) and 'authoritative' in str(e), \
                f"rejected for the wrong reason: {e}"
        else:
            raise AssertionError(
                "a Z_mean disagreeing with mean_t(Z_mc) was accepted")


def test_z_mean_is_recomputed_from_z_mc_on_load():
    """A consistent state loads, and the mean comes from Z_mc rather than the file."""
    cache, cfg, *_ = _mk()
    with tempfile.TemporaryDirectory() as td:
        path = cache.save_state(Path(td) / "s.pt")
        c2 = NegativeCache.load_state(path, cfg, DEVICE, expect_T=T)
    expected = c2.Z_mc.float().mean(dim=0).to(c2.Z_mc.dtype)
    assert torch.equal(c2.Z_student, expected), \
        "Z_student must be exactly mean_t(Z_mc) after load"
    assert c2.is_cached_mcdp and c2.Z_mc.shape == cache.Z_mc.shape
    # and it is still the ALIAS the cache scores against
    assert torch.allclose(c2.Z_student.float(), cache.Z_student.float(), atol=1e-2)


def test_reload_reproduces_next_rng_decisions():
    """A save/reload must reproduce the NEXT cache-random decision, not only the
    current embeddings — otherwise a resumed miner diverges from an uninterrupted one.

    Both generators are ADVANCED before saving. ``_gen`` is seeded deterministically
    from ``cache_init_seed`` in ``__init__``, so a cache that has not consumed any
    torch randomness would match a freshly-seeded one by coincidence and the test
    would pass even if the state were never restored.
    """
    cache, cfg, *_ = _mk()
    # advance both generators away from their initial seeds
    cache.rng.integers(0, 10_000, size=17)
    torch.rand(23, generator=cache._gen)

    with tempfile.TemporaryDirectory() as td:
        path = cache.save_state(Path(td) / "s.pt")
        # NumPy bit generator drives uniform candidate sampling
        expect_np = cache.rng.integers(0, 10_000, size=8).tolist()
        # torch generator drives Gumbel softmax selection
        expect_torch = torch.rand(6, generator=cache._gen).tolist()

        c2 = NegativeCache.load_state(path, cfg, DEVICE, expect_T=T)
        got_np = c2.rng.integers(0, 10_000, size=8).tolist()
        got_torch = torch.rand(6, generator=c2._gen).tolist()

    assert got_np == expect_np, \
        f"NumPy RNG not restored: {got_np} != {expect_np} (uniform candidate " \
        f"sampling would diverge after a resume)"
    assert got_torch == expect_torch, \
        "torch RNG not restored (Gumbel selection would diverge after a resume)"


def test_state_carries_no_registry_entries():
    """R_doc deferred: nothing registry-shaped is serialized, and a reloaded cache
    still admits nothing."""
    cache, cfg, c_ids, lookup, model, tok = _mk()
    with tempfile.TemporaryDirectory() as td:
        path = cache.save_state(Path(td) / "s.pt")
        raw = torch.load(path, map_location='cpu', weights_only=False)
        assert 'registry_entries' not in raw and 'registry' not in raw, \
            f"registry unexpectedly serialized: {sorted(raw)}"
        assert raw.get('registry_deferred') is True
        c2 = NegativeCache.load_state(path, cfg, DEVICE, expect_T=T)

    assert len(c2.registry) == 0
    # drive a replacement and confirm nothing is admitted
    c2.last_refreshed_step[:] = 0
    c2.intervals_since_selected[:] = 99
    counters = maintain_interval_cached_mcdp(
        c2, model, tok, lookup, c_ids, _reservoir(model, tok, cfg), 500, T, cfg,
        DEVICE, qrels_dict={})
    assert counters['num_R_candidates'] == 0, "R_doc is deferred; no R candidates"
    assert len(c2.registry) == 0, "evicted docs must not be admitted while R_doc is deferred"


# ---- validation before device transfer -------------------------------------

def test_mismatches_rejected_before_device_transfer():
    cache, cfg, *_ = _mk()
    with tempfile.TemporaryDirectory() as td:
        path = cache.save_state(Path(td) / "s.pt")
        for kwargs, what in (({'expect_T': T + 1}, 'T'),
                             ({'expect_B_doc': B_DOC + 1}, 'B_doc'),
                             ({'expect_dim': DIM + 1}, 'dim'),
                             ({'expect_schema': 'cached_mcdp_v99'}, 'schema')):
            try:
                NegativeCache.load_state(path, cfg, DEVICE, **kwargs)
            except ValueError as e:
                assert what in str(e) or 'schema' in str(e), \
                    f"{what} mismatch raised the wrong error: {e}"
            else:
                raise AssertionError(f"{what} mismatch was NOT rejected")


def test_validation_happens_before_allocation():
    """A rejected state must not have moved tensors to the device.

    Tracked by counting ``Tensor.to`` calls: validation runs on CPU tensors from
    ``map_location='cpu'``, so a mismatch raises before any transfer.
    """
    cache, cfg, *_ = _mk()
    with tempfile.TemporaryDirectory() as td:
        path = cache.save_state(Path(td) / "s.pt")
        calls = {'n': 0}
        real_to = torch.Tensor.to

        def counting_to(self, *a, **kw):
            calls['n'] += 1
            return real_to(self, *a, **kw)

        torch.Tensor.to = counting_to
        try:
            try:
                NegativeCache.load_state(path, cfg, DEVICE, expect_B_doc=B_DOC + 1)
            except ValueError:
                pass
            else:
                raise AssertionError("mismatch was not rejected")
            rejected_calls = calls['n']
        finally:
            torch.Tensor.to = real_to
    assert rejected_calls == 0, \
        f"{rejected_calls} tensor transfers happened before validation rejected the state"


def test_b_doc_clamped_to_small_corpus_and_restarts():
    """A corpus smaller than the configured B_doc yields a SMALLER cache.

    Restart validation must compare against the effective size; comparing against
    the raw config value would make every restart fail on a small corpus.
    """
    small_corpus = 12
    cfg = make_cfg(uncertainty='cached_mcdp', B_doc=B_DOC, T=T, lambda_val=0.5,
                   batch_size=4, mc_batch_size=8, miner_mc_batch_size=8,
                   passage_max_len=8, query_max_len=8)
    c_ids = [f"d{i}" for i in range(small_corpus)]
    lookup = {d: f"document {d} text" for d in c_ids}
    embs = np.random.default_rng(1).standard_normal((small_corpus, DIM)).astype('float32')
    model, tok = DropoutMockModel(hidden=DIM, p=0.3), MockTokenizer()

    cache, _ = NegativeCache.init_cached_mcdp(embs, c_ids, lookup, model, tok, cfg,
                                              DEVICE, dim=DIM)
    assert cache.B_doc == small_corpus < cfg['B_doc'], \
        f"expected clamp to {small_corpus}, got B_doc={cache.B_doc}"
    effective = NegativeCache.effective_B_doc(cfg, len(c_ids))
    assert effective == small_corpus

    with tempfile.TemporaryDirectory() as td:
        path = cache.save_state(Path(td) / "s.pt")
        # the miner's restart path: validate against the EFFECTIVE size
        c2 = NegativeCache.load_state(path, cfg, DEVICE, expect_T=T,
                                      expect_B_doc=effective, expect_dim=DIM)
        assert c2.B_doc == small_corpus
        # validating against the raw configured B_doc must fail loudly, not silently
        try:
            NegativeCache.load_state(path, cfg, DEVICE, expect_B_doc=cfg['B_doc'])
        except ValueError:
            pass
        else:
            raise AssertionError("configured-vs-effective B_doc mismatch not caught")


def test_metadata_tensor_validation():
    """Every metadata tensor is checked for presence, shape and dtype."""
    cache, cfg, *_ = _mk()
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "s.pt"
        cache.save_state(path)
        good = torch.load(path, map_location='cpu', weights_only=False)

        for key, mutate, why in (
                ('utility_ema', lambda s: s.pop('utility_ema'), 'missing'),
                ('last_refreshed_step',
                 lambda s: s.__setitem__('last_refreshed_step',
                                         torch.zeros(3, dtype=torch.long)), 'wrong shape'),
                ('selected_indicator',
                 lambda s: s.__setitem__('selected_indicator',
                                         torch.zeros(B_DOC, dtype=torch.long)), 'wrong dtype'),
                ('docids', lambda s: s.__setitem__('docids', s['docids'][:-1]), 'short'),
                ('docids', lambda s: s.__setitem__(
                    'docids', [s['docids'][0]] * B_DOC), 'duplicates'),
        ):
            state = dict(good)
            state.update({k: v for k, v in good.items()})
            mutate(state)
            torch.save(state, path)
            try:
                NegativeCache.load_state(path, cfg, DEVICE)
            except ValueError:
                pass
            else:
                raise AssertionError(f"{key} ({why}) was accepted")


def test_internally_inconsistent_state_rejected():
    cache, cfg, *_ = _mk()
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "s.pt"
        cache.save_state(path)
        raw = torch.load(path, map_location='cpu', weights_only=False)
        raw['T'] = T + 1                     # header disagrees with Z_mc
        torch.save(raw, path)
        try:
            NegativeCache.load_state(path, cfg, DEVICE)
        except ValueError as e:
            assert 'inconsistent' in str(e).lower() or 'T' in str(e)
        else:
            raise AssertionError("internally inconsistent state was accepted")


def test_generic_maintain_raises_on_cached_mcdp():
    """The EMA maintain() would write ONE deterministic embedding per slot into
    Z_student (the Z_mean alias) while leaving Z_mc untouched — desynchronising the
    mean from the MC bank. It must refuse rather than corrupt."""
    cache, cfg, c_ids, lookup, model, tok = _mk()
    reservoir = {'q_student': torch.nn.functional.normalize(
        torch.randn(6, DIM), dim=-1), 'q_teacher': None,
        'qids': [f"rq{i}" for i in range(6)]}
    try:
        cache.maintain(model, None, tok, lookup, c_ids, reservoir, 500, cfg, DEVICE,
                       qrels_dict={})
    except RuntimeError as e:
        assert 'maintain_cached_mcdp' in str(e), \
            f"guard must point at the right API, got: {e}"
    else:
        raise AssertionError("maintain() silently ran on a cached-MCDP cache")
    # the EMA path itself must still work on an EMA cache
    from fast_grass_test import make_cache
    ema_cfg = make_cfg(uncertainty='ema', B_doc=10)
    ema_cache, ema_cfg, ema_ids, _embs, ema_lookup = make_cache(ema_cfg, n_corpus=20,
                                                                dim=8)
    assert ema_cache.Z_teacher is not None and not ema_cache.is_cached_mcdp
    from fast_grass_test import GradMockModel
    out = ema_cache.maintain(GradMockModel(hidden=8).eval(), GradMockModel(hidden=8).eval(),
                             MockTokenizer(), ema_lookup, ema_ids,
                             {'q_student': torch.nn.functional.normalize(
                                 torch.randn(4, 8), dim=-1),
                              'q_teacher': torch.nn.functional.normalize(
                                 torch.randn(4, 8), dim=-1),
                              'qids': [f"q{i}" for i in range(4)]},
                             5, ema_cfg, DEVICE, qrels_dict={})
    assert 'num_refresh' in out, "EMA maintain() must still work"


# ---- a reloaded cache still works ------------------------------------------

def test_reloaded_cache_mines_and_maintains():
    cache, cfg, c_ids, lookup, model, tok = _mk()
    _dirty(cache)
    with tempfile.TemporaryDirectory() as td:
        path = cache.save_state(Path(td) / "s.pt")
        c2 = NegativeCache.load_state(path, cfg, DEVICE, expect_T=T,
                                      expect_B_doc=B_DOC, expect_dim=DIM)

    qids = [f"q{i}" for i in range(8)]
    q2t = {q: f"query {q}" for q in qids}
    qrels = {q: {c2.docids[i % c2.B_doc]} for i, q in enumerate(qids)}
    mined, _slots, _q, mstats = mine_batch_cached_mcdp(
        c2, model, tok, qids, q2t, qrels, T, cfg, DEVICE)
    assert mstats['mcdp_doc_encoder_calls_mining'] == 0
    H = set(c2.docids)
    for q, negs in mined.items():
        assert negs and all(d in H for d in negs), "negatives must come from H"
        assert all(d not in qrels[q] for d in negs), "positive leaked"

    c2.last_refreshed_step[:] = 0
    counters = maintain_interval_cached_mcdp(
        c2, model, tok, lookup, c_ids, _reservoir(model, tok, cfg), 500, T, cfg,
        DEVICE, qrels_dict=qrels)
    assert counters['num_refresh'] + counters['num_replace'] > 0
    assert counters['maintenance_model_step'] == 500
    assert len(c2.docids) == c2.B_doc == len(set(c2.docids))


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


TESTS = [
    ("round trip preserves all state", test_round_trip_preserves_all_state),
    ("inconsistent Z_mean is REJECTED", test_inconsistent_z_mean_is_rejected),
    ("Z_mean recomputed from Z_mc on load", test_z_mean_is_recomputed_from_z_mc_on_load),
    ("B_doc clamps to small corpus + restarts", test_b_doc_clamped_to_small_corpus_and_restarts),
    ("metadata tensors validated", test_metadata_tensor_validation),
    ("generic maintain() raises on cached-MCDP", test_generic_maintain_raises_on_cached_mcdp),
    ("reload reproduces next RNG decisions", test_reload_reproduces_next_rng_decisions),
    ("state carries no registry (R_doc deferred)", test_state_carries_no_registry_entries),
    ("schema/T/B_doc/dim mismatches rejected", test_mismatches_rejected_before_device_transfer),
    ("validation precedes device transfer", test_validation_happens_before_allocation),
    ("internally inconsistent state rejected", test_internally_inconsistent_state_rejected),
    ("reloaded cache mines + maintains", test_reloaded_cache_mines_and_maintains),
]


def main():
    print("\nAsync Fast-GRASS cache-persistence tests")
    print("=" * 60)
    passed = sum(_run(name, fn) for name, fn in TESTS)
    total = len(TESTS)
    print("=" * 60)
    print(f"  {passed}/{total} passed")
    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
