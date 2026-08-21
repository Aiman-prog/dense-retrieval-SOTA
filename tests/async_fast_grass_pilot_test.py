"""
Async Fast-GRASS — lambda-pilot tests.

CPU-only, deterministic, no downloads and no processed data required. Covers the
machinery added for the lambda pilot:

  manifest      exact stratified counts, uniqueness, determinism, proportional
                interleaving, sha256 stability, strict application to the mixture,
                and that preflight / orchestrator / miner all resolve the SAME
                ordered item list through the SAME production helpers
  refresh       explicit ``max_age_steps`` beats ``max_age_epochs``; the pre-fix
                derived value is REJECTED; first-refresh-eligible checkpoint math
  probe         the lambda grid shares one s_hat/sigma draw; lambda=0 reproduces
                mean-score ranking; band selection, ties, fallback, distinctness and
                the one-survivor case; positives never leak
  diagnostics   the miner's per-batch selection stats, flip rate restricted to TopK,
                and the query-weighted aggregation that lands in mining_meta
  gate          numeric-round requirement, refresh requirement, min-steps requirement,
                the separate smoke threshold, and round-consumption arithmetic
  eval          domain-list routing, nonzero exit on a failed domain,
                --require_existing, and the promotion decision rule
  unchanged     the base recipe is identical to HEAD except the intentional
                max_age_steps correction; --debug still yields 512 HQ-first items

Run: python tests/async_fast_grass_pilot_test.py
"""
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root / 'src'))
sys.path.insert(0, str(project_root / 'scripts'))
# the dosage probe and the pilot decision rule are developer tools under scripts/dev/
sys.path.insert(0, str(project_root / 'scripts' / 'dev'))

from utils.negative_cache import NegativeCache  # noqa: E402
from utils.helpers import load_config, get_training_context  # noqa: E402
from fast_grass_test import make_cfg, DropoutMockModel, MockTokenizer, DEVICE  # noqa: E402

import async_fast_grass_pilot as pilot  # noqa: E402
import async_fast_grass_quality_probe as probe  # noqa: E402
from async_fast_grass_cached_mcdp import (  # noqa: E402
    build_async_cfg, validate_refresh_schedule, steps_per_epoch,
    mine_batch_cached_mcdp, MiningDiagnostics, init_Z_mc,
)
from run_async_fast_grass_train import summarize_round_consumption  # noqa: E402

DIM, T, N_CORPUS, B_DOC = 16, 3, 60, 24

# proportions of the real mixture, scaled down so the suite stays fast
FIXTURE_SIZES = {'msmarco': 830, 'vl': 1499, 'hq': 970}
# query_id prefixes as emitted by src/data/preprocessor.py
FIXTURE_PREFIX = {'msmarco': 'msmarco', 'vl': 'reasonir_vl', 'hq': 'reasonir_hq'}
# the three files src/data/preprocessor.py run_setup writes into training_mixture/
SOURCE_FILES_EXPECTED = ('train_msmarco.jsonl', 'train_vl.jsonl', 'train_hq.jsonl')


# ---- fixtures ---------------------------------------------------------------

def _write_mixture(tmp, sizes=None):
    """A miniature training_mixture/ with the real file names and id prefixes."""
    sizes = sizes or FIXTURE_SIZES
    tmp = Path(tmp)
    tmp.mkdir(parents=True, exist_ok=True)
    for source, n in sizes.items():
        with open(tmp / f"train_{source}.jsonl", 'w') as f:
            for i in range(n):
                f.write(json.dumps({
                    'query_id': f"{FIXTURE_PREFIX[source]}_{i}",
                    'query': f"{source} query {i}",
                    'positive_passages': [{'docid': f"{source}_d{i}", 'text': 'text'}],
                }) + "\n")
    return tmp


def _mixture_items(by_source):
    return [it for source in sorted(by_source) for it in by_source[source]]


def _mk_cache(lam=0.5, m=1, selection_mode='topk'):
    cfg = make_cfg(uncertainty='cached_mcdp', B_doc=B_DOC, T=T, lambda_val=lam, m=m,
                   selection_mode=selection_mode, batch_size=8, mc_batch_size=8,
                   miner_mc_batch_size=8, passage_max_len=8, query_max_len=8)
    c_ids = [f"d{i}" for i in range(N_CORPUS)]
    lookup = {d: f"document {d} body text" for d in c_ids}
    embs = np.random.default_rng(0).standard_normal((N_CORPUS, DIM)).astype('float32')
    model = DropoutMockModel(hidden=DIM, p=0.3)
    tok = MockTokenizer()
    cache, _ = NegativeCache.init_cached_mcdp(embs, c_ids, lookup, model, tok, cfg,
                                              DEVICE, dim=DIM)
    return cache, cfg, lookup, model, tok


def _grid_row(lam, flip, sd=0.0, leaked=0):
    return {'lambda_val': lam, 'flip_rate_mean': flip, 'flip_rate_std': sd,
            'positives_leaked_total': leaked, 'sel_s_hat_mean': 0.0,
            'sel_sigma_mean': 0.0, 'sel_lambda_sigma_mean': 0.0,
            'lambda_sigma_vs_margin_mean': None, 'num_seeds': 3,
            'flip_rate_min': flip, 'flip_rate_max': flip}


# ---- manifest ---------------------------------------------------------------

def test_manifest_exact_counts():
    """pilot10 must be exactly 8303/14997/9700 = 33,000 on the real mixture sizes."""
    with tempfile.TemporaryDirectory() as td:
        real = {'msmarco': 83030, 'vl': 149970, 'hq': 97000}
        by = pilot.load_mixture_with_source(_write_mixture(td, real))
        counts = pilot.resolve_counts(by, preset='pilot10')
        assert counts == {'msmarco': 8303, 'vl': 14997, 'hq': 9700}, counts
        assert sum(counts.values()) == 33000
        rows = pilot.build_manifest(by, counts, seed=42)
        assert len(rows) == 33000, len(rows)
        assert pilot.manifest_source_counts(rows) == counts

        smoke = pilot.resolve_counts(by, preset='smoke1k')
        assert sum(smoke.values()) == 1024, smoke
        assert len(pilot.build_manifest(by, smoke, seed=42)) == 1024


def test_stratified_counts_sum_exactly():
    """Largest-remainder must hit the total exactly, not 'about' it."""
    avail = {'msmarco': 83030, 'vl': 149970, 'hq': 97000}
    for total in (1, 7, 100, 1024, 33001):
        c = pilot.stratified_counts(avail, total)
        assert sum(c.values()) == total, (total, c)
    # proportions preserved to within one record per source
    c = pilot.stratified_counts(avail, 1024)
    pool = sum(avail.values())
    for s, n in c.items():
        assert abs(n - 1024 * avail[s] / pool) < 1.0, (s, n)


def test_manifest_deterministic_and_seed_sensitive():
    with tempfile.TemporaryDirectory() as td:
        by = pilot.load_mixture_with_source(_write_mixture(td))
        counts = {'msmarco': 83, 'vl': 149, 'hq': 97}
        a = pilot.build_manifest(by, counts, seed=42)
        b = pilot.build_manifest(by, counts, seed=42)
        c = pilot.build_manifest(by, counts, seed=7)
        assert a == b, "same seed must reproduce the manifest exactly"
        assert a != c, "a different seed must produce a different manifest"
        # per-source draws are independent of which other sources were requested
        only_hq = pilot.build_manifest(by, {'hq': 97}, seed=42)
        hq_a = [r['query_id'] for r in a if r['source'] == 'hq']
        assert hq_a == [r['query_id'] for r in only_hq], \
            "a source's draw must not depend on the other sources"


def test_manifest_interleaves_sources():
    """A mining batch must be source-mixed: the miner does NOT shuffle."""
    with tempfile.TemporaryDirectory() as td:
        by = pilot.load_mixture_with_source(_write_mixture(td))
        counts = {'msmarco': 830, 'vl': 1499, 'hq': 970}
        rows = pilot.build_manifest(by, counts, seed=42)
        head = rows[:64]
        seen = {r['source'] for r in head}
        assert seen == {'msmarco', 'vl', 'hq'}, seen
        # and roughly in global proportion (within a generous tolerance)
        total = sum(counts.values())
        for s, n in counts.items():
            got = sum(1 for r in head if r['source'] == s)
            assert abs(got / 64 - n / total) < 0.10, (s, got)
        # no long single-source run anywhere
        longest, run = 0, 0
        for i, r in enumerate(rows):
            run = run + 1 if i and r['source'] == rows[i - 1]['source'] else 1
            longest = max(longest, run)
        assert longest <= 5, f"longest single-source run {longest}"


def test_source_files_match_what_preprocessor_writes():
    """The manifest must read exactly the files ``run_setup`` writes, no more.

    Pins ``SOURCE_FILES`` to ``src/data/preprocessor.py`` rather than to a guess: if
    the writer's filenames ever change, this fails here instead of on the cluster.

    The mixture is written by the module's ``__main__`` block (``run_setup`` only
    builds the derived corpus/queries/qrels from an already-present mixture), so this
    checks the module source and the ``prepare_*_train_data`` defaults.
    """
    import inspect
    from data import preprocessor
    src = Path(inspect.getfile(preprocessor)).read_text()
    for name in SOURCE_FILES_EXPECTED:
        assert f'"training_mixture/{name}"' in src, \
            f"preprocessor no longer writes {name}; update pilot.SOURCE_FILES"
    for fn in (preprocessor.BRIGHTPreprocessor.prepare_msmarco_train_data,
               preprocessor.BRIGHTPreprocessor.prepare_vl_train_data,
               preprocessor.BRIGHTPreprocessor.prepare_hq_train_data):
        default = inspect.signature(fn).parameters['filename'].default
        assert default in SOURCE_FILES_EXPECTED, (fn.__name__, default)
    assert set(pilot.SOURCE_FILES.values()) == set(SOURCE_FILES_EXPECTED)

    assert pilot._source_of('train_hq.jsonl') == 'hq'
    assert pilot._source_of('train_msmarco.jsonl') == 'msmarco'
    assert pilot._source_of('reasonir_corpus.jsonl') is None
    assert pilot._source_of('train_something_else.jsonl') is None, \
        "an unrecognised file must be ignored, not become a fourth stratum"


def test_empty_mixture_error_names_the_files():
    with tempfile.TemporaryDirectory() as td:
        (Path(td) / "train_hq.jsonl").write_text(
            json.dumps({'query_id': 'a', 'query': 'q', 'positive_passages': []}) + "\n")
        try:
            pilot.load_mixture_with_source(td)
        except pilot.ManifestError as e:
            assert 'train_hq.jsonl' in str(e) and 'positive_passages' in str(e), str(e)
            return
    raise AssertionError("a mixture with no usable records must raise")


def test_duplicate_query_ids_raise():
    dup = {'a': [{'query_id': 'x', 'query': 'q', 'pos_docid': 'd'}],
           'b': [{'query_id': 'x', 'query': 'q', 'pos_docid': 'd'}]}
    try:
        pilot.assert_unique_query_ids(dup)
    except pilot.ManifestError as e:
        assert 'collision' in str(e).lower()
        return
    raise AssertionError("colliding query_ids must raise, not be silently deduped")


def test_manifest_sha_stable_and_content_sensitive():
    with tempfile.TemporaryDirectory() as td:
        by = pilot.load_mixture_with_source(_write_mixture(td))
        counts = {'msmarco': 83, 'vl': 149, 'hq': 97}
        rows = pilot.build_manifest(by, counts, seed=42)
        p = Path(td) / "m.jsonl"
        m1 = pilot.write_manifest(p, rows, 42, counts, preset='x')
        m2 = pilot.write_manifest(p, rows, 42, counts, preset='x')
        assert m1['sha256'] == m2['sha256'], "sha must be stable across rebuilds"
        m3 = pilot.write_manifest(p, rows[:-1], 42, counts, preset='x')
        assert m3['sha256'] != m1['sha256'], "sha must change with content"
        assert pilot.manifest_meta(p)['sha256'] == m3['sha256']


def test_manifest_round_trip_and_strict_apply():
    with tempfile.TemporaryDirectory() as td:
        by = pilot.load_mixture_with_source(_write_mixture(td))
        counts = {'msmarco': 83, 'vl': 149, 'hq': 97}
        rows = pilot.build_manifest(by, counts, seed=42)
        p = Path(td) / "m.jsonl"
        pilot.write_manifest(p, rows, 42, counts)
        assert pilot.load_manifest(p) == rows

        items = _mixture_items(by)
        applied = pilot.apply_manifest(items, rows)
        assert len(applied) == len(rows)
        assert [a['query_id'] for a in applied] == [r['query_id'] for r in rows], \
            "apply_manifest must impose MANIFEST order, not mixture order"

        try:
            pilot.apply_manifest(items[:5], rows)
        except pilot.ManifestError as e:
            assert 'absent' in str(e)
            return
        raise AssertionError("missing manifest ids must raise, never be dropped")


def test_all_three_processes_load_identical_items():
    """preflight, orchestrator and miner must resolve the SAME ordered list.

    All three call ``maybe_apply_manifest``; this drives that production helper three
    times from three independently loaded mixtures, which is what the processes do.
    """
    with tempfile.TemporaryDirectory() as td:
        mix = _write_mixture(td)
        counts = {'msmarco': 83, 'vl': 149, 'hq': 97}
        by = pilot.load_mixture_with_source(mix)
        rows = pilot.build_manifest(by, counts, seed=42)
        p = Path(td) / "m.jsonl"
        pilot.write_manifest(p, rows, 42, counts)

        results = []
        for _process in range(3):
            items = _mixture_items(pilot.load_mixture_with_source(mix))
            out, meta = pilot.maybe_apply_manifest(items, p, log=lambda _m: None)
            results.append(([it['query_id'] for it in out], meta['sha256']))
        assert results[0] == results[1] == results[2], \
            "the three processes disagree on the manifest-filtered mixture"
        assert len(results[0][0]) == sum(counts.values())


def test_no_manifest_is_a_passthrough():
    items = [{'query_id': 'a', 'query': 'q', 'pos_docid': 'd'}]
    out, meta = pilot.maybe_apply_manifest(items, None, log=lambda _m: None)
    assert out is items and meta is None


# ---- refresh schedule -------------------------------------------------------

def _ctx(recipe):
    ctx = get_training_context(recipe)
    return ctx


def test_explicit_max_age_steps_wins():
    config = load_config()
    ctx = _ctx('async_fast_grass_pilot')
    cfg = build_async_cfg(config, ctx, steps_per_epoch=516)
    assert cfg['max_age_steps'] == 100, cfg['max_age_steps']
    assert cfg['max_age_source'] == 'max_age_steps'
    # ... and is NOT max_age_epochs * steps_per_epoch
    assert cfg['max_age_steps'] != ctx['args']['max_age_epochs'] * 516


def test_max_age_epochs_fallback_reproduces_head():
    """Recipes with no explicit key must behave exactly as before this change."""
    config = load_config()
    ctx = _ctx('async_fast_grass')
    ctx['args'] = {k: v for k, v in ctx['args'].items() if k != 'max_age_steps'}
    cfg = build_async_cfg(config, ctx, steps_per_epoch=5157)
    assert cfg['max_age_steps'] == 2 * 5157 == 10314, cfg['max_age_steps']
    assert cfg['max_age_source'] == 'max_age_epochs'


def test_missing_both_age_keys_raises():
    config = load_config()
    ctx = _ctx('async_fast_grass')
    ctx['args'] = {k: v for k, v in ctx['args'].items()
                   if k not in ('max_age_steps', 'max_age_epochs')}
    try:
        build_async_cfg(config, ctx, steps_per_epoch=100)
    except ValueError as e:
        assert 'max_age' in str(e)
        return
    raise AssertionError("a recipe with no age key must raise, not default silently")


def test_pre_fix_schedule_is_rejected():
    """The exact production bug: max_age_steps == total_steps is a hard error."""
    config = load_config()
    ctx = _ctx('async_fast_grass')
    ctx['args'] = {k: v for k, v in ctx['args'].items() if k != 'max_age_steps'}
    cfg = build_async_cfg(config, ctx, steps_per_epoch=5157)
    errors, _w, _i = validate_refresh_schedule(cfg)
    assert errors, "max_age_steps == total_steps must be an ERROR"
    assert 'influence training' in errors[0]


def test_first_refresh_checkpoint_math():
    base = {'max_age_steps': 100, 'total_steps': 1032, 'async_mine_every_steps': 100,
            'rho_start': 0.5, 'B_doc': 32000, 'cache_update_interval': 10,
            'steps_per_epoch': 516, 'batch_size': 64}
    errors, warnings, info = validate_refresh_schedule(base)
    assert not errors and not warnings, (errors, warnings)
    assert info['first_refresh_checkpoint_step'] == 100
    assert info['initial_interval_budget'] == 310, info['initial_interval_budget']

    late = {**base, 'max_age_steps': 150}
    errors, warnings, info = validate_refresh_schedule(late)
    assert not errors
    assert warnings and 'checkpoint-200' in warnings[0], warnings
    assert info['first_refresh_checkpoint_step'] == 200


def test_zero_budget_is_an_error():
    cfg = {'max_age_steps': 10, 'total_steps': 1000, 'async_mine_every_steps': 10,
           'rho_start': 0.5, 'B_doc': 4, 'cache_update_interval': 1,
           'steps_per_epoch': 10000, 'batch_size': 64}
    errors, _w, _i = validate_refresh_schedule(cfg)
    assert any('no-op' in e for e in errors), errors


def test_recipe_step_arithmetic():
    assert steps_per_epoch(33000, 64) == 516
    assert steps_per_epoch(33000, 64) * 2 == 1032
    assert steps_per_epoch(1024, 64) == 16
    assert steps_per_epoch(1024, 64) * 4 == 64


def test_recipe_blocks_match_the_plan():
    config = load_config()
    t = config['training']
    for name in ('async_fast_grass', 'async_fast_grass_pilot', 'async_fast_grass_smoke'):
        assert name in t, name
        # merge keys must have carried the inherited values through
        assert t[name]['uncertainty'] == 'cached_mcdp'
        assert t[name]['rho_start'] == 0.50
        assert t[name]['cache_init_seed'] == 42

    p = t['async_fast_grass_pilot']
    assert (p['num_epochs'], p['batch_size'], p['B_doc'], p['T'], p['m']) == \
        (2, 64, 32000, 3, 1)
    assert p['selection_mode'] == 'topk' and p['train_group_size'] == 2
    assert p['cache_update_interval'] == 10 and p['async_mine_every_steps'] == 100
    assert p['ready_poll_steps'] == 10 and p['async_poll_interval'] == 10
    assert p['max_age_steps'] == 100 and p['logging_steps'] == 10
    assert p['bootstrap_checkpoint_step'] == 0 and p['pilot_gate_min_steps'] == 128
    assert float(p['learning_rate']) == 1e-5

    s = t['async_fast_grass_smoke']
    assert (s['B_doc'], s['T'], s['num_epochs']) == (512, 3, 4)
    assert s['async_mine_every_steps'] == 10 and s['max_age_steps'] == 10
    assert s['pilot_gate_min_steps'] == 1

    full = t['async_fast_grass']
    assert full['max_age_steps'] == 1000
    assert 'pilot_gate_min_steps' not in full, \
        "the full run must have NO pilot gate, or its exit behaviour changes"
    # distinct model names => distinct model dirs => arms cannot collide
    names = {t[n]['model_name'] for n in
             ('async_fast_grass', 'async_fast_grass_pilot', 'async_fast_grass_smoke')}
    assert len(names) == 3, names


# ---- lambda grid / probe ----------------------------------------------------

def test_grid_shares_one_draw_and_lambda0_is_mean_ranking():
    """All lambdas must come from ONE s_hat/sigma, and lambda=0 == mean ranking."""
    cache, cfg, lookup, model, tok = _mk_cache(lam=0.0, m=2)
    Z_mc = cache.Z_mc
    qids = [f"q{i}" for i in range(8)]
    qid_to_text = {q: f"query {q}" for q in qids}
    qrels = {q: {f"d{i}"} for i, q in enumerate(qids)}

    lambdas = [0.0, 0.1, 0.5, 1.0]
    grid, sig = probe.probe_grid(cache, Z_mc, model, tok, qids, qid_to_text, qrels,
                                 T, lambdas, cfg['m'], cfg, DEVICE, seed=11,
                                 query_batch_size=4)
    assert set(grid) == {float(x) for x in lambdas}
    assert grid[0.0]['flip_rate'] == 0.0, "lambda=0 cannot differ from itself"
    assert sig['sigma_all_zero'] is False

    # explicit check of the algebra the grid relies on
    from async_fast_grass_cached_mcdp import encode_queries_mc, score_cached_mcdp
    torch.manual_seed(11)
    np.random.seed(11)
    q_mc, _ = encode_queries_mc(model, tok, [qid_to_text[q] for q in qids[:4]], T,
                                DEVICE, cfg)
    _g, s_hat, sigma = score_cached_mcdp(q_mc, Z_mc, 0.0)
    for lam in lambdas:
        g_direct = probe.score_cached_mcdp(q_mc, Z_mc, lam)[0]
        assert torch.allclose(g_direct, s_hat + lam * sigma, atol=1e-5), lam
    # lambda=0 ranking is exactly the mean-score ranking
    assert torch.equal(torch.topk(s_hat, k=3, dim=1).indices,
                       torch.topk(s_hat + 0.0 * sigma, k=3, dim=1).indices)


def test_probe_never_selects_a_positive():
    cache, cfg, lookup, model, tok = _mk_cache(lam=0.0, m=2)
    qids = [f"q{i}" for i in range(6)]
    qid_to_text = {q: f"query {q}" for q in qids}
    # make several cache slots positives for every query
    in_cache = cache.docids[:4]
    qrels = {q: set(in_cache) for q in qids}
    grid, _sig = probe.probe_grid(cache, cache.Z_mc, model, tok, qids, qid_to_text,
                                  qrels, T, [0.0, 0.5, 1.0], cfg['m'], cfg, DEVICE,
                                  seed=3, query_batch_size=3)
    for lam, row in grid.items():
        assert row['positives_leaked'] == 0, (lam, row['positives_leaked'])


def test_query_batching_covers_every_query():
    """Chunking must partition the queries exactly — every one scored, none twice.

    It does NOT reproduce the same numbers across chunk sizes: each chunk draws its own
    dropout masks, so ``query_batch_size`` is part of the experiment. Document chunking
    is the one that must be exactly invariant, and that is asserted separately below.
    """
    cache, cfg, lookup, model, tok = _mk_cache(lam=0.0, m=1)
    qids = [f"q{i}" for i in range(12)]
    qid_to_text = {q: f"query {q}" for q in qids}
    qrels = {q: {f"d{i}"} for i, q in enumerate(qids)}
    for bs in (12, 5, 4, 1):
        grid, _s = probe.probe_grid(cache, cache.Z_mc, model, tok, qids, qid_to_text,
                                    qrels, T, [0.0, 0.5], cfg['m'], cfg, DEVICE,
                                    seed=5, query_batch_size=bs)
        for lam in (0.0, 0.5):
            assert grid[lam]['num_queries'] == 12, (bs, lam, grid[lam]['num_queries'])
            assert 0.0 <= grid[lam]['flip_rate'] <= 1.0
        assert grid[0.0]['flip_rate'] == 0.0, bs


def test_document_chunking_is_exactly_invariant():
    """score_chunk_size only splits a matmul, so it must not move any number."""
    cache, cfg, lookup, model, tok = _mk_cache(lam=0.0, m=1)
    qids = [f"q{i}" for i in range(8)]
    qid_to_text = {q: f"query {q}" for q in qids}
    qrels = {q: {f"d{i}"} for i, q in enumerate(qids)}
    outs = []
    for chunk in (None, 7, 3):
        c = dict(cfg)
        c['score_chunk_size'] = chunk
        grid, _s = probe.probe_grid(cache, cache.Z_mc, model, tok, qids, qid_to_text,
                                    qrels, T, [0.0, 0.5], c['m'], c, DEVICE, seed=5,
                                    query_batch_size=8)
        outs.append(grid)
    for lam in (0.0, 0.5):
        base = outs[0][lam]
        for other in outs[1:]:
            assert abs(other[lam]['flip_rate'] - base['flip_rate']) < 1e-9, lam
            assert abs(other[lam]['sel_sigma_mean']
                       - base['sel_sigma_mean']) < 1e-5, lam


def test_select_lambdas_in_band():
    rows = [_grid_row(0.0, 0.0), _grid_row(0.1, 0.05), _grid_row(0.2, 0.12),
            _grid_row(0.3, 0.16), _grid_row(0.5, 0.26), _grid_row(1.0, 0.60)]
    s = probe.select_lambdas(rows)
    # low band [0.10, 0.20), centre 0.15: 0.3 (flip 0.16) is nearer than 0.2 (0.12)
    assert s['selected_low'] == 0.3, s['selected_low']
    # medium band [0.20, 0.35], centre 0.275: only 0.5 (flip 0.26) is in band
    assert s['selected_medium'] == 0.5, s['selected_medium']
    assert s['band_satisfied'] == {'low': True, 'medium': True}
    assert s['n_arms'] == 2
    # 0.1 (flip 0.05) is below the low band and 1.0 (0.60) above the medium band
    assert s['selected_low'] not in (0.1, 1.0)


def test_select_lambdas_never_picks_zero():
    rows = [_grid_row(0.0, 0.15), _grid_row(0.5, 0.60)]
    s = probe.select_lambdas(rows)
    assert s['selected_low'] != 0.0 and s['selected_medium'] != 0.0
    assert s['selected_low'] == 0.5


def test_select_lambdas_tie_prefers_smaller():
    """Equidistant from the band centre -> the SMALLER lambda wins.

    In exact arithmetic |0.13-0.15| == |0.17-0.15|, but in floating point they differ
    by ~2e-17, so without distance quantization the tie-break could never fire and the
    winner would be decided by representation error.
    """
    rows = [_grid_row(0.0, 0.0), _grid_row(0.2, 0.13), _grid_row(0.3, 0.17),
            _grid_row(0.9, 0.30)]
    s = probe.select_lambdas(rows)
    assert s['selected_low'] == 0.2, s['selected_low']
    # a difference LARGER than the tolerance must still decide on distance, not size
    rows2 = [_grid_row(0.0, 0.0), _grid_row(0.2, 0.101), _grid_row(0.3, 0.150)]
    assert probe.select_lambdas(rows2)['selected_low'] == 0.3


def test_select_lambdas_fallback_flags_band():
    rows = [_grid_row(0.0, 0.0), _grid_row(0.5, 0.45), _grid_row(1.0, 0.80)]
    s = probe.select_lambdas(rows)
    assert s['selected_low'] == 0.5
    assert s['band_satisfied']['low'] is False
    assert any('band NOT satisfied' in n for n in s['notes'])


def test_select_lambdas_fallback_stays_distinct():
    """A fallback that would duplicate the low arm must advance to the next one."""
    rows = [_grid_row(0.0, 0.0), _grid_row(0.5, 0.44), _grid_row(0.7, 0.52)]
    s = probe.select_lambdas(rows)
    assert s['selected_low'] == 0.5
    assert s['selected_medium'] == 0.7, s['selected_medium']
    assert s['selected_low'] != s['selected_medium']
    assert s['n_arms'] == 2


def test_select_lambdas_single_survivor_reports_one_arm():
    rows = [_grid_row(0.0, 0.0), _grid_row(0.5, 0.15),
            _grid_row(1.0, 0.30, sd=0.20)]          # rejected: unstable
    s = probe.select_lambdas(rows)
    assert s['num_survivors'] == 1
    assert s['selected_low'] == 0.5
    assert s['selected_medium'] is None
    assert s['n_arms'] == 1
    assert any('ONE nonzero pilot arm' in n for n in s['notes'])


def test_select_lambdas_rejects_unstable_and_leaky():
    rows = [_grid_row(0.0, 0.0),
            _grid_row(0.3, 0.15, sd=0.051),          # SD just over the threshold
            _grid_row(0.5, 0.25, leaked=1)]          # leaked a known positive
    s = probe.select_lambdas(rows)
    assert s['num_survivors'] == 0
    assert s['n_arms'] == 0
    assert s['selected_low'] is None
    reasons = ' '.join(r for row in s['rejected'] for r in row['rejected_because'])
    assert 'SD' in reasons and 'positives' in reasons


def test_probe_grid_always_includes_control():
    assert probe._parse_grid("0.1,0.5")[0] == 0.0, "lambda=0 control must be forced in"
    assert probe._parse_grid("0,0.5") == [0.0, 0.5]


def test_probe_uses_the_named_recipe_not_fast_grass():
    """--recipe must actually reach the model / dropout / B_doc / T / lengths.

    ``run_real`` used to hardcode ``get_training_context('fast_grass')``, so the recipe
    flag would have been accepted and ignored.
    """
    import inspect
    src = inspect.getsource(probe.run_real)
    assert "get_training_context(args.recipe)" in src, \
        "run_real must resolve the recipe from --recipe"
    assert "get_training_context('fast_grass')" not in src, \
        "run_real must not read the SEQUENTIAL fast_grass block"
    assert "_build_fast_grass_cfg" not in src, \
        "run_real must build cfg with build_async_cfg, not the sequential builder"
    assert "build_async_cfg" in src
    # the manifest must be applied BEFORE the max_queries slice, or the sample is
    # mixture-file order (HQ first), not stratified
    assert src.index("maybe_apply_manifest") < src.index("[:args.max_queries]"), \
        "manifest must be applied before slicing to --max_queries"
    # and the async cfg must supply the model knobs
    assert "cfg.get('mc_dropout_p'" in src


# ---- mining diagnostics -----------------------------------------------------

def _mine(cache, cfg, model, tok, lam, m=1, age_step=None, qrels=None):
    cfg = dict(cfg)
    cfg['lambda_val'] = lam
    cfg['m'] = m
    qids = [f"q{i}" for i in range(6)]
    qid_to_text = {q: f"query {q}" for q in qids}
    qrels = qrels if qrels is not None else {q: {f"d{i}"} for i, q in enumerate(qids)}
    return mine_batch_cached_mcdp(cache, model, tok, qids, qid_to_text, qrels, T, cfg,
                                  DEVICE, age_step=age_step)


def test_mining_diagnostics_present():
    cache, cfg, lookup, model, tok = _mk_cache()
    _mined, _slots, _q, stats = _mine(cache, cfg, model, tok, lam=0.5, age_step=250)
    for key in ('sel_s_hat_mean', 'sel_sigma_mean', 'sel_lambda_sigma_mean',
                'flip_rate_vs_lambda0', 'sel_age_mean', 'sel_age_max', 'num_queries'):
        assert key in stats, key
    assert abs(stats['sel_lambda_sigma_mean'] - 0.5 * stats['sel_sigma_mean']) < 1e-6
    # cache was initialised at step 0, so age == age_step
    assert stats['sel_age_mean'] == 250.0, stats['sel_age_mean']
    assert stats['mcdp_doc_encoder_calls_mining'] == 0


def test_age_step_optional():
    cache, cfg, lookup, model, tok = _mk_cache()
    _m, _s, _q, stats = _mine(cache, cfg, model, tok, lam=0.5, age_step=None)
    assert 'sel_age_mean' not in stats


def test_flip_rate_zero_at_lambda_zero_and_positive_when_sigma_bites():
    cache, cfg, lookup, model, tok = _mk_cache()
    _m, _s, _q, s0 = _mine(cache, cfg, model, tok, lam=0.0)
    assert s0['flip_rate_vs_lambda0'] == 0.0, s0['flip_rate_vs_lambda0']
    _m, _s, _q, s9 = _mine(cache, cfg, model, tok, lam=50.0)
    assert s9['flip_rate_vs_lambda0'] > 0.0, \
        "a huge lambda must reorder something, or the diagnostic is broken"
    assert s9['flip_rate_unsupported_reason'] is None


def test_flip_rate_is_none_under_softmax_selection():
    """Gumbel top-k is a SAMPLE; a flip against the lambda=0 argmax is noise."""
    cache, cfg, lookup, model, tok = _mk_cache(selection_mode='softmax')
    _m, _s, _q, stats = _mine(cache, cfg, model, tok, lam=0.5)
    assert stats['flip_rate_vs_lambda0'] is None
    assert 'Gumbel' in (stats['flip_rate_unsupported_reason'] or '')


def test_diagnostics_are_query_weighted():
    d = MiningDiagnostics()
    d.add({'num_queries': 64, 'sel_sigma_mean': 1.0, 'flip_rate_vs_lambda0': 0.5,
           'sel_age_mean': 10.0, 'sel_age_max': 12.0})
    d.add({'num_queries': 16, 'sel_sigma_mean': 5.0, 'flip_rate_vs_lambda0': 0.0,
           'sel_age_mean': 20.0, 'sel_age_max': 40.0})
    s = d.summary()
    assert abs(s['sel_sigma_mean'] - (64 * 1.0 + 16 * 5.0) / 80) < 1e-9, s
    assert abs(s['flip_rate_vs_lambda0'] - (64 * 0.5) / 80) < 1e-9
    assert s['sel_age_max'] == 40.0
    assert s['diagnostics_num_queries'] == 80 and s['diagnostics_num_batches'] == 2


def test_diagnostics_absent_keys_are_none_not_zero():
    d = MiningDiagnostics()
    d.add({'num_queries': 8, 'sel_sigma_mean': 2.0, 'flip_rate_vs_lambda0': None,
           'flip_rate_unsupported_reason': 'softmax'})
    s = d.summary()
    assert s['flip_rate_vs_lambda0'] is None, \
        "an unmeasured flip rate must be None, never 0.0 (which reads as 'no flips')"
    assert s['sel_age_mean'] is None
    assert s['flip_rate_unsupported_reason'] == 'softmax'
    assert json.loads(json.dumps(s))  # must survive the mining_meta round trip


# ---- round consumption + gate ----------------------------------------------

def test_round_consumption_arithmetic():
    recs = [{'round_no': 0, 'consume_step': 0, 'source_checkpoint_step': 0},
            {'round_no': 1, 'consume_step': 300, 'source_checkpoint_step': 100},
            {'round_no': 3, 'consume_step': 800, 'source_checkpoint_step': 600}]
    out = summarize_round_consumption(recs, 1032)
    assert [r['steps_active'] for r in out] == [300, 500, 232], out
    assert [r['async_gap_steps'] for r in out] == [0, 200, 200]
    # a single round spans the whole run
    solo = summarize_round_consumption([recs[0]], 64)
    assert solo[0]['steps_active'] == 64


def _gate(rounds, metas, min_steps=128, miner_failed=None, model_ok=True):
    with tempfile.TemporaryDirectory() as td:
        model_dir = Path(td) / "model"
        model_dir.mkdir()
        if model_ok:
            (model_dir / "config.json").write_text("{}")
            (model_dir / "model.safetensors").write_text("w")
        return pilot.evaluate_pilot_gate(
            Path(td), {'rounds': rounds}, model_dir, miner_failed, min_steps,
            read_meta=lambda n: metas.get(n, {}))


def test_gate_passes_on_a_refreshed_round():
    ok, reasons, details = _gate(
        [{'round_no': 0, 'consume_step': 0, 'steps_active': 300},
         {'round_no': 1, 'consume_step': 300, 'steps_active': 400}],
        {1: {'num_refresh_total': 12, 'num_replace_total': 5}})
    assert ok, reasons
    assert len(details['qualifying_rounds']) == 1
    assert details['rounds_consumed_numeric'][0]['num_replace_total'] == 5


def test_gate_rejects_initial_data_only():
    ok, reasons, _d = _gate([{'round_no': 0, 'consume_step': 0, 'steps_active': 1032}],
                            {})
    assert not ok
    assert any('initial_data' in r for r in reasons), reasons


def test_gate_requires_refresh():
    ok, reasons, _d = _gate(
        [{'round_no': 1, 'consume_step': 100, 'steps_active': 900}],
        {1: {'num_refresh_total': 0, 'num_replace_total': 900}})
    assert not ok
    assert any('num_refresh_total=0' in r for r in reasons), reasons


def test_gate_requires_min_steps():
    rounds = [{'round_no': 1, 'consume_step': 1000, 'steps_active': 32}]
    metas = {1: {'num_refresh_total': 50}}
    ok, reasons, _d = _gate(rounds, metas, min_steps=128)
    assert not ok, "32 steps must fail the pilot's 128"
    assert any('32 steps' in r for r in reasons), reasons
    # the SMOKE threshold accepts exactly this run
    ok_smoke, _r, _d = _gate(rounds, metas, min_steps=1)
    assert ok_smoke, "min_steps=1 must accept a short smoke run"


def test_gate_fails_on_dead_miner_and_missing_model():
    ok, reasons, _d = _gate(
        [{'round_no': 1, 'consume_step': 0, 'steps_active': 900}],
        {1: {'num_refresh_total': 5}}, miner_failed=1)
    assert not ok and any('miner exited' in r for r in reasons), reasons

    ok, reasons, _d = _gate(
        [{'round_no': 1, 'consume_step': 0, 'steps_active': 900}],
        {1: {'num_refresh_total': 5}}, model_ok=False)
    assert not ok and any('no final model' in r for r in reasons), reasons


def test_orchestrator_gate_wiring():
    """Nonzero exit on FAIL, summary written first, and NO gate for the full recipe."""
    import inspect
    import train_async_fast_grass as orch
    src = inspect.getsource(orch.main)
    assert "gate_min_steps = ctx['args'].get('pilot_gate_min_steps')" in src
    assert "if gate_min_steps is not None:" in src, \
        "the gate must be skipped entirely for recipes without the key"
    assert "return 1" in src, "a failed gate must exit nonzero, not just print"
    assert "async_run_summary.json" in src
    assert src.index("async_run_summary.json") < src.index("return 1"), \
        "the run summary must be written BEFORE the failing exit"
    # the full recipe genuinely has no key, so the gate is skipped there
    assert 'pilot_gate_min_steps' not in load_config()['training']['async_fast_grass']


def test_manifest_is_required_by_pilot_and_smoke_recipes():
    """An empty ASYNC_FG_MANIFEST must be a hard error, not a full-mixture run.

    Reproduces the real failure: `${ASYNC_FG_MANIFEST:+--manifest $X}` expands to
    NOTHING when the shell variable is unset, so the job silently ran the smoke recipe
    against all 330,000 items — steps_per_epoch 5,157 instead of 16, and a maintenance
    budget of 0.5*512*1/5157 that rounds to zero. The pilot would not even have failed
    loudly; it would have trained 10,314 steps instead of 1,032.
    """
    import train_async_fast_grass as orch
    for recipe in ('async_fast_grass_pilot', 'async_fast_grass_smoke'):
        ctx = _ctx(recipe)
        assert ctx['args'].get('requires_manifest') is True, recipe
        err = orch.check_manifest_required(ctx, None, recipe)
        assert err and 'requires --manifest' in err, (recipe, err)
        assert orch.check_manifest_required(ctx, '', recipe), \
            "an EMPTY manifest path must fail too, not just a missing one"
        assert orch.check_manifest_required(ctx, '/some/m.jsonl', recipe) is None

    # the full run has no manifest and must stay unaffected
    full = _ctx('async_fast_grass')
    assert 'requires_manifest' not in full['args']
    assert orch.check_manifest_required(full, None, 'async_fast_grass') is None


def test_budget_collapses_without_a_manifest():
    """The exact arithmetic that failed: full mixture + smoke recipe -> zero budget."""
    config = load_config()
    ctx = _ctx('async_fast_grass_smoke')
    cfg = build_async_cfg(config, ctx, steps_per_epoch=steps_per_epoch(330000, 64))
    errors, _w, info = validate_refresh_schedule(cfg)
    assert info['initial_interval_budget'] == 0, info
    assert any('no-op' in e for e in errors), errors
    # with the manifest applied it is healthy again
    cfg_ok = build_async_cfg(config, ctx, steps_per_epoch=steps_per_epoch(1024, 64))
    errors_ok, _w2, info_ok = validate_refresh_schedule(cfg_ok)
    assert not errors_ok, errors_ok
    assert info_ok['initial_interval_budget'] == 16, info_ok


def test_preflight_does_not_regenerate_data():
    """--preflight must inspect paths, never call run_setup (which rebuilds files)."""
    import inspect
    import train_async_fast_grass as orch
    src = inspect.getsource(orch.main)
    pre = src.index("if args.preflight:")
    setup = src.index("from data.preprocessor import run_setup")
    assert pre < setup, \
        "the preflight branch must return BEFORE run_setup is even imported"
    paths_src = inspect.getsource(orch._preflight_paths)
    # the docstring names run_setup; what matters is that it never CALLS it
    assert 'run_setup(' not in paths_src, "_preflight_paths must not invoke run_setup"
    for name in ('reasonir_corpus.jsonl', 'train_qrels.txt', 'training_mixture'):
        assert name in paths_src, name
    # and it reports missing inputs rather than creating them
    assert 'missing' in paths_src


def test_preflight_reports_missing_inputs():
    import train_async_fast_grass as orch
    corpus, qrels, missing = orch._preflight_paths()
    # no processed data in the test environment -> everything is reported missing
    assert isinstance(missing, list)
    assert corpus.name == 'reasonir_corpus.jsonl' and qrels.name == 'train_qrels.txt'


# ---- eval routing + decision ------------------------------------------------

def _fake_results(base, model, domains, scores):
    d = base / model
    d.mkdir(parents=True, exist_ok=True)
    for dom in domains:
        (d / f"{dom}_results.json").write_text(json.dumps({
            'domain': dom, 'model_path': str(d),
            'metrics': {'ndcg_cut_10': scores[dom], 'recip_rank': 0.1,
                        'recall_1000': 0.9}}))


DOMAINS4 = ['biology', 'economics', 'stackoverflow', 'theoremqa_questions']


def test_eval_domain_list_routing_and_validation():
    """--domains must restrict the run and reject unknown names."""
    script = project_root / 'scripts' / 'run_all_evals.py'
    r = subprocess.run([sys.executable, str(script), '--model_path', str(project_root),
                        '--domains', 'biology,not_a_domain'],
                       capture_output=True, text=True)
    assert r.returncode == 1, r.stdout
    assert 'unknown domain' in r.stdout, r.stdout

    src = script.read_text()
    assert '--require_existing' in src
    assert 'failed.append(domain)' in src, "a failed domain must be recorded"
    assert 'sys.exit(1)' in src
    # the swallow is gone: the except block no longer ends the story
    assert 'if failed or absent:' in src


def test_eval_exits_nonzero_when_a_domain_fails():
    """The old loop caught CalledProcessError and still exited 0."""
    import run_all_evals
    import inspect
    src = inspect.getsource(run_all_evals.main)
    assert 'except subprocess.CalledProcessError' in src
    body = src[src.index('except subprocess.CalledProcessError'):]
    assert 'failed.append(domain)' in body.split('rows, absent')[0], \
        "the handler must record the failure before continuing"
    assert 'if failed or absent:' in src, \
        "a recorded failure must still gate the exit code"
    tail = src[src.index('if failed or absent:'):]
    assert 'sys.exit(1)' in tail


def test_require_existing_does_not_prepare():
    import run_all_evals
    import inspect
    src = inspect.getsource(run_all_evals.check_and_prepare_data)
    assert 'require_existing' in src
    guard = src.index('if require_existing:')
    prepare = src.index('BRIGHTLoader()')
    assert guard < prepare, "the guard must short-circuit before any loader runs"
    assert 'missing_domains.append(domain)' in src


def test_decision_promote_and_stop():
    import lambda_pilot_decide as dec
    base = {d: 0.20 for d in DOMAINS4}
    good = {'biology': 0.22, 'economics': 0.22, 'stackoverflow': 0.22,
            'theoremqa_questions': 0.19}
    r = dec.compare(base, good, DOMAINS4)
    assert r['verdict'] == 'promote', r
    assert r['domain_wins'] == 3

    # exactly ON the threshold in decimal terms: float error must not demote it
    #   macro delta = 0.005 but computes as 0.004999999999999977
    edge = {'biology': 0.21, 'economics': 0.21, 'stackoverflow': 0.21,
            'theoremqa_questions': 0.19}
    e = dec.compare(base, edge, DOMAINS4)
    assert e['verdict'] == 'promote', (e['delta'], e['verdict'])

    # a real gain but only 2/4 domains -> stop, not promote
    narrow = {'biology': 0.24, 'economics': 0.23, 'stackoverflow': 0.19,
              'theoremqa_questions': 0.19}
    n = dec.compare(base, narrow, DOMAINS4)
    assert n['domain_wins'] == 2 and n['verdict'] == 'stop', n

    flat = {d: 0.2005 for d in DOMAINS4}
    assert dec.compare(base, flat, DOMAINS4)['verdict'] == 'stop'

    worse = {d: 0.19 for d in DOMAINS4}
    assert dec.compare(base, worse, DOMAINS4)['verdict'] == 'stop'


def test_decision_inconclusive_band():
    import lambda_pilot_decide as dec
    base = {d: 0.20 for d in DOMAINS4}
    mild = {d: 0.2030 for d in DOMAINS4}      # macro delta 0.003
    r = dec.compare(base, mild, DOMAINS4)
    assert r['verdict'] == 'inconclusive', r
    assert 'second pilot seed' in r['reason']


def test_decision_single_domain_is_stopped():
    import lambda_pilot_decide as dec
    base = {d: 0.20 for d in DOMAINS4}
    spike = {'biology': 0.28, 'economics': 0.20, 'stackoverflow': 0.20,
             'theoremqa_questions': 0.20}
    r = dec.compare(base, spike, DOMAINS4)
    assert r['single_domain_driven'] is True
    assert r['verdict'] == 'stop', r
    assert 'single domain' in r['reason']


def test_decision_tie_prefers_smaller_lambda():
    import lambda_pilot_decide as dec
    base = {d: 0.20 for d in DOMAINS4}
    low = {'biology': 0.21, 'economics': 0.21, 'stackoverflow': 0.21,
           'theoremqa_questions': 0.209}
    med = {'biology': 0.2105, 'economics': 0.2105, 'stackoverflow': 0.2105,
           'theoremqa_questions': 0.2095}
    d = dec.decide(base, {'lamLOW': low, 'lamMED': med}, DOMAINS4)
    assert set(d['promoted']) == {'lamLOW', 'lamMED'}, d['promoted']
    assert d['chosen'] == 'lamLOW', d
    assert 'smaller lambda' in d['tie_break_note']


def test_decision_refuses_partial_results():
    import lambda_pilot_decide as dec
    config = load_config()
    with tempfile.TemporaryDirectory() as td:
        base = Path(td)
        _fake_results(base, 'm0', DOMAINS4[:2], {d: 0.2 for d in DOMAINS4})
        orig = dec.results_dir
        dec.results_dir = lambda name, cfg: base / name
        try:
            dec.load_model_results('m0', DOMAINS4, config)
        except dec.MissingResults as e:
            assert 'Refusing to decide' in str(e)
            return
        finally:
            dec.results_dir = orig
    raise AssertionError("a missing domain result must raise, not shrink the macro")


# ---- unchanged behaviour ----------------------------------------------------

def test_run_suffix_isolates_model_dir_and_handoff_root():
    import inspect
    import train_async_fast_grass as orch
    src = inspect.getsource(orch.main)
    assert 'f"{ctx[\'args\'][\'model_name\']}_{args.run_suffix}"' in src \
        or "model_name'] = f\"{ctx['args']['model_name']}_{args.run_suffix}\"" in src, \
        "run_suffix must rename the model dir"
    assert 'root_name = "async_mining" + (f"_{args.run_suffix}"' in src, \
        "run_suffix must also isolate the handoff root, or arms share mined rounds"


def test_debug_mode_unchanged():
    """--debug must still be the 512-item HQ-first smoke, not silently 'fixed'."""
    import inspect
    import train_async_fast_grass as orch
    src = inspect.getsource(orch.main)
    assert 'train_items[:512]' in src
    import run_fast_grass
    loader = inspect.getsource(run_fast_grass._load_train_items)
    assert 'sorted(mix_dir.glob' in loader, \
        "the alphabetical glob is what makes --debug HQ-only; it stays as-is"


def test_base_recipe_only_differs_by_max_age_steps():
    """The full recipe must be HEAD plus the intentional correction, nothing else."""
    config = load_config()
    full = config['training']['async_fast_grass']
    expected = {
        'model_name': 'async_fast_grass_mixed_bge_m3', 'uncertainty': 'cached_mcdp',
        'B_doc': 32000, 'T': 3, 'lambda_val': 0.5, 'mc_dropout_p': 0.3, 'm': 1,
        'selection_mode': 'topk', 'beta': 5.0, 'score_chunk_size': 8192,
        'rho_start': 0.50, 'rho_end': 0.25, 'max_age_epochs': 2,
        'cache_update_interval': 100, 'utility_ema_decay': 0.95, 'utility_floor': 0.01,
        'K': 3, 'replacement_candidate_multiplier': 2,
        'recent_query_reservoir_size': 128, 'cache_init_seed': 42,
        'cache_state_keep': 2, 'miner_mc_batch_size': 128, 'reentry_top_k': 5,
        'async_mine_every_steps': 1000, 'bootstrap_checkpoint_step': 0,
        'ready_poll_steps': 100, 'async_poll_interval': 60, 'trainer_gpu': 0,
        'miner_gpu': 1, 'num_epochs': 2, 'batch_size': 64, 'train_group_size': 2,
        'mc_batch_size': 1024, 'bf16': True, 'dataloader_num_workers': 2,
        'warmup_ratio': 0.1, 'weight_decay': 0.01, 'max_grad_norm': 1.0,
        'logging_steps': 100, 'eval_top_k': 10, 'per_device_eval_batch_size': 256,
    }
    for k, v in expected.items():
        assert full[k] == v, f"{k}: {full[k]!r} != {v!r}"
    # exactly ONE new key relative to HEAD
    new = set(full) - set(expected) - {'base_model', 'learning_rate'}
    assert new == {'max_age_steps'}, f"unexpected new keys in the full recipe: {new}"
    assert full['max_age_steps'] == 1000
    # R_doc knobs still absent, so its deferral stays visible
    for absent in ('R_fraction', 'R_size_factor', 'utility_remember_threshold', 'L'):
        assert absent not in full, absent


def test_src_never_imports_scripts():
    """The pilot module lives under scripts/ precisely because of this invariant."""
    for path in (project_root / 'src').rglob('*.py'):
        text = path.read_text()
        for bad in ('async_fast_grass_pilot', 'import run_fast_grass',
                    'from scripts'):
            assert bad not in text, f"{path} imports scripts/: {bad}"


# ---- runner -----------------------------------------------------------------

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
    ("manifest exact counts (33,000 / 1,024)", test_manifest_exact_counts),
    ("stratified counts sum exactly", test_stratified_counts_sum_exactly),
    ("manifest deterministic + seed sensitive", test_manifest_deterministic_and_seed_sensitive),
    ("manifest interleaves the three sources", test_manifest_interleaves_sources),
    ("source files match preprocessor output", test_source_files_match_what_preprocessor_writes),
    ("empty mixture error names the files", test_empty_mixture_error_names_the_files),
    ("duplicate query_ids raise", test_duplicate_query_ids_raise),
    ("manifest sha stable + content sensitive", test_manifest_sha_stable_and_content_sensitive),
    ("manifest round trip + strict apply", test_manifest_round_trip_and_strict_apply),
    ("3 processes load identical items", test_all_three_processes_load_identical_items),
    ("no manifest is a pass-through", test_no_manifest_is_a_passthrough),
    ("explicit max_age_steps wins", test_explicit_max_age_steps_wins),
    ("max_age_epochs fallback == HEAD", test_max_age_epochs_fallback_reproduces_head),
    ("missing both age keys raises", test_missing_both_age_keys_raises),
    ("pre-fix refresh schedule REJECTED", test_pre_fix_schedule_is_rejected),
    ("first-refresh checkpoint math", test_first_refresh_checkpoint_math),
    ("zero maintenance budget is an error", test_zero_budget_is_an_error),
    ("516 / 1032 and 16 / 64 step counts", test_recipe_step_arithmetic),
    ("recipe blocks match the plan", test_recipe_blocks_match_the_plan),
    ("grid shares one draw; lambda0 == mean", test_grid_shares_one_draw_and_lambda0_is_mean_ranking),
    ("probe never selects a positive", test_probe_never_selects_a_positive),
    ("query batching covers every query", test_query_batching_covers_every_query),
    ("document chunking is exactly invariant", test_document_chunking_is_exactly_invariant),
    ("select_lambdas in-band picks", test_select_lambdas_in_band),
    ("select_lambdas never picks lambda=0", test_select_lambdas_never_picks_zero),
    ("select_lambdas tie -> smaller lambda", test_select_lambdas_tie_prefers_smaller),
    ("select_lambdas fallback flags band", test_select_lambdas_fallback_flags_band),
    ("select_lambdas fallback stays distinct", test_select_lambdas_fallback_stays_distinct),
    ("single survivor -> n_arms 1", test_select_lambdas_single_survivor_reports_one_arm),
    ("unstable / leaky lambdas rejected", test_select_lambdas_rejects_unstable_and_leaky),
    ("grid always includes the control", test_probe_grid_always_includes_control),
    ("probe uses --recipe, not fast_grass", test_probe_uses_the_named_recipe_not_fast_grass),
    ("mining diagnostics present", test_mining_diagnostics_present),
    ("age_step is optional", test_age_step_optional),
    ("flip rate 0 at lambda=0, >0 when sigma bites", test_flip_rate_zero_at_lambda_zero_and_positive_when_sigma_bites),
    ("flip rate None under softmax", test_flip_rate_is_none_under_softmax_selection),
    ("diagnostics are query-weighted", test_diagnostics_are_query_weighted),
    ("absent diagnostics are None, not 0.0", test_diagnostics_absent_keys_are_none_not_zero),
    ("round-consumption arithmetic", test_round_consumption_arithmetic),
    ("gate passes on a refreshed round", test_gate_passes_on_a_refreshed_round),
    ("gate rejects initial_data only", test_gate_rejects_initial_data_only),
    ("gate requires num_refresh_total > 0", test_gate_requires_refresh),
    ("gate min_steps: 128 fails, smoke 1 passes", test_gate_requires_min_steps),
    ("gate fails on dead miner / no model", test_gate_fails_on_dead_miner_and_missing_model),
    ("orchestrator exits nonzero on gate FAIL", test_orchestrator_gate_wiring),
    ("pilot/smoke recipes REQUIRE a manifest", test_manifest_is_required_by_pilot_and_smoke_recipes),
    ("budget collapses to 0 without a manifest", test_budget_collapses_without_a_manifest),
    ("preflight never calls run_setup", test_preflight_does_not_regenerate_data),
    ("preflight reports missing inputs", test_preflight_reports_missing_inputs),
    ("eval --domains routing + validation", test_eval_domain_list_routing_and_validation),
    ("eval exits nonzero on a failed domain", test_eval_exits_nonzero_when_a_domain_fails),
    ("--require_existing does not prepare", test_require_existing_does_not_prepare),
    ("decision promote / stop", test_decision_promote_and_stop),
    ("decision inconclusive band", test_decision_inconclusive_band),
    ("decision stops single-domain gains", test_decision_single_domain_is_stopped),
    ("decision tie -> smaller lambda", test_decision_tie_prefers_smaller_lambda),
    ("decision refuses partial results", test_decision_refuses_partial_results),
    ("run_suffix isolates dir + handoff root", test_run_suffix_isolates_model_dir_and_handoff_root),
    ("--debug behaviour unchanged", test_debug_mode_unchanged),
    ("full recipe == HEAD + max_age_steps", test_base_recipe_only_differs_by_max_age_steps),
    ("src/ never imports scripts/", test_src_never_imports_scripts),
]


def main():
    print("\nAsync Fast-GRASS lambda-pilot tests")
    print("=" * 72)
    passed = sum(_run(name, fn) for name, fn in TESTS)
    total = len(TESTS)
    print("=" * 72)
    print(f"  {passed}/{total} passed")
    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
