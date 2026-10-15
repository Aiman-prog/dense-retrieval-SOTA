"""
GRASS uncertainty estimator screen (Phase B) — tests.

CPU-only, deterministic, no model, no downloads, no processed data. Covers the pure
machinery the screen's conclusion rests on:

  sampling      deterministic under a seed, sensitive to it, splits disjoint and
                exactly sized, and a corpus too small is an error not a short sample
  geometry      1 - cos drift, representation dispersion, score dispersion and score
                volatility against hand-computable cases
  statistics    average ranks with shared ties, Spearman on monotone and independent
                data, and that conditioning on a confound REMOVES a correlation that
                is entirely explained by it — the property the gate depends on
  bootstrap     determinism under a fixed seed, and that a resampled statistic brackets
                its point estimate
  budget        captured drift is bounded by the oracle, and ranking by the label
                reproduces the oracle exactly
  discipline    the held-out split shares no document with calibration or validation

Run: python tests/grass_uncertainty_screen_test.py
"""
import sys
from pathlib import Path

import numpy as np

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root / 'scripts'))
sys.path.insert(0, str(project_root / 'src'))
sys.path.insert(0, str(project_root / 'tests'))

from grass_uncertainty_screen import (  # noqa: E402
    sample_and_split_ids, cosine_drift, representation_dispersion, score_dispersion,
    score_volatility, rankdata, partial_spearman, bootstrap_ci, captured_drift,
    analyse_split, encode_deterministic, mc_scalars,
)

SIZES = {'calibration': 40, 'validation': 20, 'test': 20}
IDS = [f'd{i}' for i in range(500)]


# ── sampling ──────────────────────────────────────────────────────────────────

def test_sample_is_deterministic():
    a = sample_and_split_ids(IDS, SIZES, 42)
    b = sample_and_split_ids(IDS, SIZES, 42)
    assert a == b, "same seed produced different splits"


def test_sample_is_seed_sensitive():
    a = sample_and_split_ids(IDS, SIZES, 42)
    b = sample_and_split_ids(IDS, SIZES, 43)
    assert a != b, "different seeds produced identical splits"


def test_splits_are_disjoint_and_exact():
    s = sample_and_split_ids(IDS, SIZES, 42)
    for name, n in SIZES.items():
        assert len(s[name]) == n, f"{name}: {len(s[name])} != {n}"
        assert len(set(s[name])) == n, f"{name} has duplicates"
    cal, val, test = set(s['calibration']), set(s['validation']), set(s['test'])
    assert not (cal & val), "calibration and validation overlap"
    assert not (cal & test), "calibration and test overlap"
    assert not (val & test), "validation and test overlap"


def test_held_out_split_is_isolated():
    """The discipline the screen is built on: nothing used for fitting or for choosing T
    may appear in the held-out read."""
    s = sample_and_split_ids(IDS, SIZES, 42)
    seen = set(s['calibration']) | set(s['validation'])
    assert not (seen & set(s['test'])), "a test document was also fitted or tuned on"


def test_oversized_request_raises():
    try:
        sample_and_split_ids(IDS[:10], SIZES, 42)
    except ValueError as exc:
        assert 'corpus has' in str(exc), f"unhelpful message: {exc}"
        return
    raise AssertionError("a sample larger than the corpus was accepted")


def test_sample_ids_come_from_the_corpus():
    s = sample_and_split_ids(IDS, SIZES, 7)
    drawn = [d for ids in s.values() for d in ids]
    assert set(drawn) <= set(IDS), "sample contains ids absent from the corpus"
    assert len(set(drawn)) == sum(SIZES.values()), "a document was drawn twice"


# ── geometry ──────────────────────────────────────────────────────────────────

def test_cosine_drift_identical_is_zero():
    a = np.random.default_rng(0).normal(size=(6, 8))
    assert np.allclose(cosine_drift(a, a), 0.0, atol=1e-6), "identical rows drifted"


def test_cosine_drift_orthogonal_and_opposite():
    a = np.array([[1.0, 0.0], [1.0, 0.0]])
    b = np.array([[0.0, 1.0], [-1.0, 0.0]])
    got = cosine_drift(a, b)
    assert np.allclose(got, [1.0, 2.0], atol=1e-6), f"orthogonal/opposite wrong: {got}"


def test_representation_dispersion_zero_without_variance():
    z = np.random.default_rng(1).normal(size=(1, 5, 4))
    Z = np.repeat(z, 7, axis=0)
    assert np.allclose(representation_dispersion(Z), 0.0, atol=1e-6), \
        "identical MC samples reported nonzero dispersion"


def test_representation_dispersion_grows_with_spread():
    rng = np.random.default_rng(2)
    base = rng.normal(size=(1, 30, 16))
    tight = np.repeat(base, 8, axis=0) + 0.01 * rng.normal(size=(8, 30, 16))
    loose = np.repeat(base, 8, axis=0) + 0.50 * rng.normal(size=(8, 30, 16))
    assert representation_dispersion(loose).mean() > representation_dispersion(tight).mean(), \
        "wider MC spread did not increase dispersion"


def test_representation_dispersion_rejects_wrong_rank():
    try:
        representation_dispersion(np.zeros((4, 4)))
    except ValueError as exc:
        assert 'T, n, D' in str(exc), f"unhelpful message: {exc}"
        return
    raise AssertionError("a 2-D tensor was accepted as [T, n, D]")


def test_score_dispersion_zero_without_variance():
    z = np.random.default_rng(3).normal(size=(1, 6, 5))
    Z = np.repeat(z, 4, axis=0)
    Q = np.random.default_rng(4).normal(size=(3, 5))
    out = score_dispersion(Z, Q)
    assert out.shape == (6,), f"expected one value per document, got {out.shape}"
    assert np.allclose(out, 0.0, atol=1e-6), "identical samples gave nonzero score spread"


def test_score_volatility_identical_is_zero():
    e = np.random.default_rng(5).normal(size=(7, 5))
    Q = np.random.default_rng(6).normal(size=(4, 5))
    assert np.allclose(score_volatility(e, e, Q), 0.0, atol=1e-6), \
        "an unchanged document reported score volatility"


# ── statistics ────────────────────────────────────────────────────────────────

def test_rankdata_basic_and_ties():
    assert np.allclose(rankdata([10.0, 30.0, 20.0]), [1, 3, 2]), "plain ranks wrong"
    assert np.allclose(rankdata([5.0, 5.0, 9.0]), [1.5, 1.5, 3]), "ties not averaged"
    assert np.allclose(rankdata([2.0, 2.0, 2.0]), [2, 2, 2]), "all-equal ranks wrong"


def test_spearman_monotone_and_independent():
    x = np.arange(60, dtype=float)
    assert partial_spearman(x, x ** 3) > 0.999, "monotone transform was not rank-1"
    assert partial_spearman(x, -x) < -0.999, "reversed order was not rank -1"
    rng = np.random.default_rng(7)
    rho = partial_spearman(rng.normal(size=400), rng.normal(size=400))
    assert abs(rho) < 0.15, f"independent vectors correlated at {rho}"


def test_conditioning_removes_a_pure_confound():
    """u carries NO information about y beyond the control. This is exactly the failure
    the screen must be able to detect, so it has to survive as a test."""
    rng = np.random.default_rng(8)
    control = rng.normal(size=500)
    u = control + 0.01 * rng.normal(size=500)
    y = control + 0.01 * rng.normal(size=500)
    assert partial_spearman(u, y) > 0.9, "setup failed: u and y should look correlated"
    inc = partial_spearman(u, y, (control,))
    assert abs(inc) < 0.25, f"confound survived conditioning: incremental {inc}"


def test_conditioning_keeps_genuine_signal():
    rng = np.random.default_rng(9)
    control = rng.normal(size=500)
    signal = rng.normal(size=500)
    u = signal + 0.05 * rng.normal(size=500)
    y = control + signal + 0.05 * rng.normal(size=500)
    inc = partial_spearman(u, y, (control,))
    assert inc > 0.5, f"genuine incremental information was erased: {inc}"


def test_partial_spearman_without_controls_is_plain():
    rng = np.random.default_rng(10)
    a, b = rng.normal(size=80), rng.normal(size=80)
    assert abs(partial_spearman(a, b) - partial_spearman(a, b, ())) < 1e-12, \
        "empty control set changed the statistic"


# ── bootstrap ─────────────────────────────────────────────────────────────────

def test_bootstrap_is_deterministic_under_seed():
    rng = np.random.default_rng(11)
    x, y = rng.normal(size=120), rng.normal(size=120)
    fn = lambda ix: partial_spearman(x[ix], y[ix])  # noqa: E731
    a = bootstrap_ci(fn, 120, 200, 3)
    b = bootstrap_ci(fn, 120, 200, 3)
    assert a == b, "same seed produced different intervals"
    assert a != bootstrap_ci(fn, 120, 200, 4), "different seeds produced identical intervals"


def test_bootstrap_brackets_a_strong_point_estimate():
    x = np.arange(200, dtype=float)
    y = x + np.random.default_rng(12).normal(scale=0.1, size=200)
    ci = bootstrap_ci(lambda ix: partial_spearman(x[ix], y[ix]), 200, 300, 5)
    assert ci['lower'] > 0.9, f"a near-perfect correlation had lower bound {ci['lower']}"
    assert ci['lower'] <= ci['upper'], "interval is inverted"


# ── budget proxy ──────────────────────────────────────────────────────────────

def test_captured_drift_is_bounded_by_the_oracle():
    rng = np.random.default_rng(13)
    y = rng.random(200)
    u = rng.random(200)
    assert captured_drift(y, y, 20) >= captured_drift(u, y, 20), \
        "an estimator beat the oracle ranking"


def test_captured_drift_oracle_takes_the_largest():
    y = np.array([0.1, 0.9, 0.5, 0.7])
    assert abs(captured_drift(y, y, 2) - 1.6) < 1e-9, "oracle did not take the top two"


def test_captured_drift_caps_at_population():
    y = np.array([1.0, 2.0, 3.0])
    assert abs(captured_drift(y, y, 99) - 6.0) < 1e-9, "budget above n did not cap"


# ── end to end on synthetic columns ───────────────────────────────────────────

def test_analyse_split_reports_every_required_field():
    rng = np.random.default_rng(14)
    n = 300
    cols = {'y': rng.random(n), 'b': rng.random(n), 'v': rng.random(n),
            'u_rep_T3': rng.random(n), 'u_rep_T8': rng.random(n),
            'u_score_T3': rng.random(n), 'u_score_T8': rng.random(n)}
    out = analyse_split(cols, np.arange(n), (3, 8), (64, 128), 50, 1)
    assert out['n'] == n
    for t in (3, 8):
        entry = out[f'T{t}']
        for key in ('raw_spearman', 'incremental_spearman', 'incremental_ci95',
                    'score_incremental_spearman', 'captured_drift'):
            assert key in entry, f"T{t} missing {key}"
        assert entry['incremental_ci95']['lower'] <= entry['incremental_ci95']['upper']
        assert set(entry['captured_drift']) == {'64', '128'}, "budget grid not reported"
    assert out['paired_T_difference']['comparison'] == 'T8 - T3'


def test_analyse_split_detects_a_planted_signal():
    rng = np.random.default_rng(15)
    n = 600
    b = rng.random(n)
    u = rng.random(n)
    cols = {'y': b + u, 'b': b, 'v': rng.random(n),
            'u_rep_T3': u, 'u_rep_T8': u,
            'u_score_T3': rng.random(n), 'u_score_T8': rng.random(n)}
    out = analyse_split(cols, np.arange(n), (3, 8), (64,), 200, 2)
    inc = out['T3']['incremental_spearman']
    assert inc > 0.5, f"a planted incremental signal was not recovered: {inc}"
    assert out['T3']['incremental_ci95']['lower'] > 0, "lower bound did not clear zero"


# ── encoder contract (regressions, both found by a mock smoke) ────────────────

def _mock_pair():
    import torch
    from fast_grass_test import DropoutMockModel, MockTokenizer
    torch.manual_seed(0)
    return DropoutMockModel(hidden=8, p=0.5), MockTokenizer(), torch.device('cpu')


def test_deterministic_encode_ignores_caller_mode():
    """`encode_batch_tensor` suppresses gradients but not dropout. An nn.Module defaults
    to training=True, so without an explicit eval the drift LABEL would itself be
    stochastic and the screen would be measuring noise against noise."""
    import torch
    model, tok, dev = _mock_pair()
    assert model.training, "fixture must start in train mode for this to test anything"
    texts = [f'doc {i}' for i in range(12)]
    a = encode_deterministic(model, tok, texts, dev, 128, 8)
    b = encode_deterministic(model, tok, texts, dev, 128, 8)
    assert torch.allclose(a, b, atol=1e-6), "deterministic encode varied between calls"
    assert model.training, "entry mode was not restored"


def test_deterministic_encode_is_unit_norm():
    model, tok, dev = _mock_pair()
    out = encode_deterministic(model, tok, [f'doc {i}' for i in range(10)], dev, 128, 8)
    assert abs(float(out.norm(dim=-1).mean()) - 1.0) < 1e-5, "embeddings are not normalized"


def test_mc_passes_is_T_not_a_chunk_tally():
    """mc_passes is the LOGICAL pass count. Summing the per-chunk value reported T=8 as
    24 across three chunks, which would misstate the screen's cost."""
    import torch
    model, tok, dev = _mock_pair()
    texts = [f'doc {i}' for i in range(30)]
    Q = torch.nn.functional.normalize(torch.randn(4, 8), dim=-1)
    _, one = mc_scalars(model, tok, texts, dev, 128, 8, 30, Q, (3, 8))
    _, many = mc_scalars(model, tok, texts, dev, 128, 8, 10, Q, (3, 8))
    assert one['mc_passes'] == many['mc_passes'] == 8, (one, many)
    assert one['examples_encoded'] == many['examples_encoded'] == 30 * 8, (one, many)


def test_mc_scalars_T3_is_the_paired_prefix_of_T8():
    """T=3 must be the first three samples of the SAME draw, otherwise the T comparison is
    two independent draws and its paired interval is invalid."""
    import torch
    from utils.helpers import encode_mc
    model, tok, dev = _mock_pair()
    texts = [f'doc {i}' for i in range(20)]
    Q = torch.nn.functional.normalize(torch.randn(4, 8), dim=-1)
    torch.manual_seed(5)
    Z, _ = encode_mc(model, tok, texts, 8, dev, 128, 8)
    want3 = representation_dispersion(Z[:3].float())
    torch.manual_seed(5)
    got, _ = mc_scalars(model, tok, texts, dev, 128, 8, 20, Q, (3, 8))
    assert np.allclose(got['u_rep_T3'], want3, atol=1e-6), "T=3 is not the T=8 prefix"


def test_mc_scalars_shapes_and_keys():
    import torch
    model, tok, dev = _mock_pair()
    texts = [f'doc {i}' for i in range(25)]
    Q = torch.nn.functional.normalize(torch.randn(4, 8), dim=-1)
    out, _ = mc_scalars(model, tok, texts, dev, 128, 8, 10, Q, (3, 8))
    assert set(out) == {'u_rep_T3', 'u_rep_T8', 'u_score_T3', 'u_score_T8'}, sorted(out)
    for key, arr in out.items():
        assert arr.shape == (25,), f"{key}: {arr.shape}"
    assert out['u_rep_T8'].mean() > 1e-4, "MC dispersion collapsed — dropout was not active"


def _run(name, fn):
    print(f"  {name:<52} ... ", end="")
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
    ("sample is deterministic", test_sample_is_deterministic),
    ("sample is seed sensitive", test_sample_is_seed_sensitive),
    ("splits disjoint and exactly sized", test_splits_are_disjoint_and_exact),
    ("held-out split is isolated", test_held_out_split_is_isolated),
    ("oversized request raises", test_oversized_request_raises),
    ("sampled ids come from the corpus", test_sample_ids_come_from_the_corpus),
    ("cosine drift: identical is zero", test_cosine_drift_identical_is_zero),
    ("cosine drift: orthogonal / opposite", test_cosine_drift_orthogonal_and_opposite),
    ("dispersion zero without MC variance", test_representation_dispersion_zero_without_variance),
    ("dispersion grows with MC spread", test_representation_dispersion_grows_with_spread),
    ("dispersion rejects wrong rank", test_representation_dispersion_rejects_wrong_rank),
    ("score dispersion zero without variance", test_score_dispersion_zero_without_variance),
    ("score volatility: unchanged doc is zero", test_score_volatility_identical_is_zero),
    ("rankdata basic and ties", test_rankdata_basic_and_ties),
    ("spearman monotone and independent", test_spearman_monotone_and_independent),
    ("conditioning removes a pure confound", test_conditioning_removes_a_pure_confound),
    ("conditioning keeps genuine signal", test_conditioning_keeps_genuine_signal),
    ("no controls == plain spearman", test_partial_spearman_without_controls_is_plain),
    ("bootstrap deterministic under seed", test_bootstrap_is_deterministic_under_seed),
    ("bootstrap brackets a strong estimate", test_bootstrap_brackets_a_strong_point_estimate),
    ("captured drift bounded by oracle", test_captured_drift_is_bounded_by_the_oracle),
    ("oracle takes the largest", test_captured_drift_oracle_takes_the_largest),
    ("budget above n caps", test_captured_drift_caps_at_population),
    ("analyse_split reports every field", test_analyse_split_reports_every_required_field),
    ("analyse_split detects planted signal", test_analyse_split_detects_a_planted_signal),
    ("deterministic encode ignores caller mode", test_deterministic_encode_ignores_caller_mode),
    ("deterministic encode is unit-norm", test_deterministic_encode_is_unit_norm),
    ("mc_passes is T, not a chunk tally", test_mc_passes_is_T_not_a_chunk_tally),
    ("T=3 is the paired prefix of T=8", test_mc_scalars_T3_is_the_paired_prefix_of_T8),
    ("mc_scalars shapes and keys", test_mc_scalars_shapes_and_keys),
]


def main():
    print("\nGRASS uncertainty estimator screen (Phase B) tests")
    print("=" * 72)
    passed = sum(_run(name, fn) for name, fn in TESTS)
    total = len(TESTS)
    print("=" * 72)
    print(f"  {passed}/{total} passed")
    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
