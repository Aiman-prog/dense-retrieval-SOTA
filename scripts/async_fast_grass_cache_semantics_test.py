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

Covered (sequential-API step semantics):
  - N maintenance calls in one round, all step=source_checkpoint_step:
    every slot whose last_refreshed_step changed == source_checkpoint_step.
  - utility advances each interval while step stays fixed (utility_ema of a
    repeatedly-selected slot rises; its lifetime_selected_count increments once
    per interval, proving one utility update per maintenance call).
  - last_refreshed_step never exceeds source_checkpoint_step (age >= 0).
  - B_doc invariant holds after every interval.
  - CONTRAST: passing a miner-local counter as step corrupts last_refreshed_step
    (drifts far below model time) — demonstrating why the counter must not be used.

Covered (cached-MCDP, against scripts/async_fast_grass_cached_mcdp.py):
  - cadence: maintenance fires every cache_update_interval * trainer_batch_size
    mined QUERIES (100*64=6400), with the threshold SUBTRACTED on fire so an
    overshooting batch carries its remainder forward.
  - budget identical across every interval of one round (rho/progress is pinned
    to the frozen checkpoint step, so it must not drift interval to interval).
  - replacement candidates encoded exactly once and reused on insertion.
  - round end: fold the remainder, but run a final interval only when useful
    pending state exists.
  - metadata totals aggregate across intervals; mining reports zero document
    encoder calls.
  - cached scoring: explicit mean + population std, lambda=0 == mean ranking,
    chunked == unchunked, positives masked before selection, dropout-only
    context restores every module's entry mode.
  - initialization: T genuinely stochastic document states, Z_mean == mean_t,
    Z_student aliased onto Z_mean.

DEFERRED TO PHASE 1: "unselected slots remain bitwise unchanged" and "slot update
is one atomic commit" are correctness properties of the real
NegativeCache.maintain_cached_mcdp, which does not exist yet — asserting them
against the Phase-0 reference stand-in would only test the stand-in.

Run: python scripts/async_fast_grass_cache_semantics_test.py
"""
import sys
from pathlib import Path

import torch

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root / 'src'))
sys.path.insert(0, str(project_root / 'scripts'))

from fast_grass_test import (  # noqa: E402
    make_cfg, make_cache, make_z_mc, GradMockModel, DropoutMockModel,
    MockTokenizer, _rand_unit, DEVICE,
)
from async_fast_grass_cached_mcdp import (  # noqa: E402
    score_cached_mcdp, dropout_only, init_Z_mc, encode_queries_mc,
    QueryMCReservoir, MaintenanceDriver, maintenance_interval_mined_queries,
    maintain_interval_cached_mcdp, mine_batch_cached_mcdp,
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


# ---- cached-MCDP fixtures --------------------------------------------------

CACHED_T = 3
CACHED_DIM = 16          # >= 16 so a p=0.3 dropout pass can't zero every dim
CACHED_STEP = 500        # source_checkpoint_step for cached-MCDP rounds


def _mk_cached(n_corpus=40, B_doc=20, **cfg_over):
    """Teacher-free cache + Z_mc built from T genuine dropout passes."""
    cfg = make_cfg(uncertainty='cached_mcdp', B_doc=B_doc, T=CACHED_T,
                   lambda_val=0.5, batch_size=64, mc_batch_size=32,
                   passage_max_len=8, query_max_len=8, **cfg_over)
    cache, cfg, c_ids, _embs, corpus_lookup = make_cache(cfg, n_corpus=n_corpus,
                                                        dim=CACHED_DIM)
    student = DropoutMockModel(hidden=CACHED_DIM, p=0.3)
    tok = MockTokenizer()
    Z_mc, init_stats = init_Z_mc(cache, corpus_lookup, student, tok, CACHED_T,
                                 cfg, DEVICE)
    return cache, Z_mc, cfg, c_ids, corpus_lookup, student, tok, init_stats


def _cached_reservoir(student, tok, cfg, n=6):
    qids = [f"rq{i}" for i in range(n)]
    res = QueryMCReservoir(cfg['recent_query_reservoir_size'])
    q_mc, _ = encode_queries_mc(student, tok, [f"reservoir query {i}" for i in
                                               range(n)], CACHED_T, DEVICE, cfg)
    res.add(q_mc, qids)
    return res.get()


def _force_maintenance_work(cache, over_slots=(10, 11, 12, 13),
                            stale_slots=(2, 3)):
    """Make some slots over-age (refresh) and some persistently unselected
    (replace), so a bounded interval always has work to plan."""
    for s in over_slots:
        cache.last_refreshed_step[s] = 0
    for s in stale_slots:
        cache.intervals_since_selected[s] = 99


# ---- cadence (C / B3) ------------------------------------------------------

def test_maintenance_interval_is_update_interval_times_batch_size():
    cfg = make_cfg(async_defaults=True)
    assert cfg['cache_update_interval'] == 100 and cfg['batch_size'] == 64
    thr = maintenance_interval_mined_queries(cfg)
    assert thr == 6400, (
        f"maintenance_interval_mined_queries must be cache_update_interval * "
        f"trainer_batch_size = 100*64 = 6400, got {thr}")
    # explicit override path (a miner may mine in batches != trainer batch size)
    assert maintenance_interval_mined_queries(cfg, batch_size=32) == 3200


def test_cadence_fires_on_mined_queries_and_carries_remainder():
    """Batches that do NOT divide the threshold evenly must still average out:
    on fire the counter SUBTRACTS the threshold instead of resetting to 0."""
    cfg = make_cfg(cache_update_interval=2, batch_size=3)   # threshold = 6
    d = MaintenanceDriver(cfg)
    assert d.threshold == 6
    remainders, fire_at = [], []
    for _ in range(10):
        d.add(4)                       # 4 does not divide 6
        if d.should_fire():
            fire_at.append(d.mined_total)
            d.consume()
            remainders.append(d.counter)
    # 40 mined queries / 6 == 6 intervals; a reset-to-zero driver would give 5
    assert d.n_intervals == 6, f"expected 6 intervals over 40 queries, got {d.n_intervals}"
    assert remainders == [2, 0, 2, 0, 2, 0], f"remainder not carried: {remainders}"
    assert fire_at == [8, 12, 20, 24, 32, 36], f"fired at wrong totals: {fire_at}"
    # long-run cadence is exact: intervals == floor(total / threshold)
    assert d.n_intervals == d.mined_total // d.threshold


def test_cadence_partial_batch_does_not_fire_early():
    cfg = make_cfg(cache_update_interval=10, batch_size=64)  # threshold = 640
    d = MaintenanceDriver(cfg)
    for _ in range(9):
        d.add(64)
        assert not d.should_fire(), "fired before reaching the mined-query threshold"
    d.add(64)
    assert d.should_fire(), "did not fire at exactly the threshold"


# ---- budget fixed within a round (B2) --------------------------------------

def test_budget_identical_across_every_interval_of_a_round():
    """rho/progress is pinned to the frozen checkpoint step, so the per-interval
    budget must be the SAME in every interval of one mining round."""
    cache, Z_mc, cfg, c_ids, lookup, student, tok, _ = _mk_cached()
    reservoir = _cached_reservoir(student, tok, cfg)
    budgets = []
    for _ in range(4):
        _force_maintenance_work(cache)
        cache.record_selection(torch.tensor([[5]], device=DEVICE))
        c = maintain_interval_cached_mcdp(
            cache, Z_mc, student, tok, lookup, c_ids, reservoir, CACHED_STEP,
            CACHED_T, cfg, DEVICE, qrels_dict={})
        budgets.append(c['maintenance_budget_interval'])
        assert c['maintenance_model_step'] == CACHED_STEP
    assert len(set(budgets)) == 1, \
        f"budget drifted across intervals of one round: {budgets}"
    # and it is the documented interval budget, evaluated at the checkpoint step:
    #   round(rho * B_doc * cache_update_interval / steps_per_epoch)
    assert budgets[0] == cache._interval_budget(CACHED_STEP, cfg)
    # a miner-local counter would have produced a different budget
    assert budgets[0] != cache._interval_budget(1, cfg)


def test_utility_folded_before_planning():
    """_update_utility must run BEFORE _interval_budget/_plan_actions, matching
    NegativeCache.maintain (negative_cache.py:308-310).

    Ordering is observable through eligibility: a slot with a stale
    ``intervals_since_selected >= K`` that WAS selected during this interval is
    reset to 0 by the fold. Folding first therefore protects it from the
    low-utility replacement rule; folding afterwards lets planning see the stale
    counter and evict a document that was just useful.
    """
    cache, Z_mc, cfg, c_ids, lookup, student, tok, _ = _mk_cached()
    reservoir = _cached_reservoir(student, tok, cfg)

    slot = 7
    # Make `slot` the ONLY over-age entry, so the bounded budget cannot be spent
    # on other slots before the replace bucket is reached. Then it is the fold
    # alone that decides refresh (kept) vs. replace (evicted).
    cache.last_refreshed_step[:] = CACHED_STEP
    cache.last_refreshed_step[slot] = 0
    cache.intervals_since_selected[slot] = 99      # stale: would be replaced
    cache.record_selection(torch.tensor([[slot]], device=DEVICE))  # ...but just used
    docid_before = cache.docids[slot]
    util_before = float(cache.utility_ema[slot])

    maintain_interval_cached_mcdp(cache, Z_mc, student, tok, lookup, c_ids,
                                  reservoir, CACHED_STEP, CACHED_T, cfg, DEVICE,
                                  qrels_dict={})

    assert int(cache.intervals_since_selected[slot]) == 0, \
        "fold did not reset intervals_since_selected for a selected slot"
    assert cache.docids[slot] == docid_before, (
        "a slot selected during this interval was replaced — planning saw "
        "pre-fold utility state, so _update_utility ran AFTER _plan_actions")
    assert float(cache.utility_ema[slot]) > util_before, "utility_ema did not advance"
    assert not bool(cache.selected_indicator[slot]), \
        "selected_indicator not reset — utility was not folded"
    assert int(cache.lifetime_selected_count[slot]) == 1


# ---- replacement encodes candidates exactly once (B6) ----------------------

def test_replacement_candidates_encoded_once_and_reused():
    cache, Z_mc, cfg, c_ids, lookup, student, tok, _ = _mk_cached()
    reservoir = _cached_reservoir(student, tok, cfg)
    _force_maintenance_work(cache)
    c = maintain_interval_cached_mcdp(cache, Z_mc, student, tok, lookup, c_ids,
                                      reservoir, CACHED_STEP, CACHED_T, cfg,
                                      DEVICE, qrels_dict={})
    assert c['num_replace'] > 0, "no replacement happened; test is vacuous"
    n_cand = c['num_recertified_candidates']
    assert n_cand >= c['num_replace']
    # every encoded doc is either a refresh doc or a candidate — counted ONCE.
    # If insertion re-encoded the chosen candidates this would exceed the sum.
    assert c['maintenance_docs_encoded'] == c['num_refresh'] + n_cand, (
        f"docs encoded {c['maintenance_docs_encoded']} != refresh "
        f"{c['num_refresh']} + candidates {n_cand} (insertion re-encoded?)")
    # three-way accounting: examples = docs * T, and T is the logical pass count
    assert c['maintenance_mc_passes'] == CACHED_T
    assert c['maintenance_examples_encoded'] == c['maintenance_docs_encoded'] * CACHED_T
    assert c['maintenance_forward_batches'] >= CACHED_T
    assert len(cache.docids) == cache.B_doc == len(set(cache.docids))


# ---- round-end partial interval (B7) ---------------------------------------

def test_round_end_final_interval_only_when_pending_state_exists():
    cache, Z_mc, cfg, *_ = _mk_cached()
    d = MaintenanceDriver(make_cfg(cache_update_interval=2, batch_size=3))
    # (a) exactly on a boundary, nothing selected since -> no final interval
    d.add(6)
    d.consume()
    cache.selected_indicator.zero_()
    assert not d.round_end_should_maintain(cache), \
        "ran a final interval with no remainder and no pending selections"
    # (b) remainder but nothing selected -> still nothing useful to fold
    d.add(2)
    assert d.pending == 2
    assert not d.round_end_should_maintain(cache)
    # (c) remainder AND pending selections -> fold them
    cache.record_selection(torch.tensor([[4]], device=DEVICE))
    assert d.round_end_should_maintain(cache), \
        "pending selections at round end must be folded"


# ---- metadata aggregation + zero mining doc encodes (B8) -------------------

def test_metadata_totals_aggregate_and_mining_encodes_no_docs():
    cache, Z_mc, cfg, c_ids, lookup, student, tok, _ = _mk_cached(B_doc=20)
    qids = [f"q{i}" for i in range(8)]
    qid_to_text = {q: f"query number {q}" for q in qids}
    qrels = {q: {cache.docids[i % cache.B_doc]} for i, q in enumerate(qids)}
    reservoir = _cached_reservoir(student, tok, cfg)

    totals = dict(num_refresh_total=0, num_replace_total=0,
                  num_recertified_candidates_total=0,
                  mcdp_docs_encoded_maintenance=0, num_maintenance_intervals=0)
    doc_calls_mining = 0
    per_interval = []
    for _ in range(3):
        mined, _slots, _q, mstats = mine_batch_cached_mcdp(
            cache, Z_mc, student, tok, qids, qid_to_text, qrels, CACHED_T, cfg,
            DEVICE)
        doc_calls_mining += mstats['mcdp_doc_encoder_calls_mining']
        # negatives come from H and never leak a known positive
        H = set(cache.docids)
        for q, negs in mined.items():
            assert all(d in H for d in negs), "negative not drawn from H"
            assert all(d not in qrels[q] for d in negs), "positive leaked"

        _force_maintenance_work(cache)
        c = maintain_interval_cached_mcdp(cache, Z_mc, student, tok, lookup,
                                          c_ids, reservoir, CACHED_STEP,
                                          CACHED_T, cfg, DEVICE, qrels_dict=qrels)
        per_interval.append(c)
        totals['num_refresh_total'] += c['num_refresh']
        totals['num_replace_total'] += c['num_replace']
        totals['num_recertified_candidates_total'] += c['num_recertified_candidates']
        totals['mcdp_docs_encoded_maintenance'] += c['maintenance_docs_encoded']
        totals['num_maintenance_intervals'] += 1

    assert doc_calls_mining == 0, (
        f"cached-MCDP mining must perform ZERO document encoder calls, got "
        f"{doc_calls_mining} — this is the regression guard against lazy fresh-MCDP")
    assert totals['num_maintenance_intervals'] == 3
    assert totals['num_refresh_total'] == sum(c['num_refresh'] for c in per_interval)
    assert totals['mcdp_docs_encoded_maintenance'] == sum(
        c['maintenance_docs_encoded'] for c in per_interval)
    assert totals['mcdp_docs_encoded_maintenance'] > 0, \
        "maintenance encoded nothing; aggregation test is vacuous"


# ---- cached scoring contract (D3) ------------------------------------------

def test_cached_score_matches_explicit_mean_and_population_std():
    q_mc = make_z_mc(CACHED_T, 4, CACHED_DIM, seed=1)
    Z_mc = make_z_mc(CACHED_T, 9, CACHED_DIM, seed=50)
    g, s_hat, sigma = score_cached_mcdp(q_mc, Z_mc, 0.5)
    S = torch.stack([q_mc[t] @ Z_mc[t].t() for t in range(CACHED_T)], dim=0)
    assert torch.allclose(s_hat, S.mean(dim=0), atol=1e-6)
    # population std (correction=0), NOT the sample std
    assert torch.allclose(sigma, S.std(dim=0, unbiased=False), atol=1e-6)
    assert not torch.allclose(sigma, S.std(dim=0, unbiased=True), atol=1e-6), \
        "sigma matches the SAMPLE std — must be population std (correction=0)"
    assert torch.allclose(g, s_hat + 0.5 * sigma, atol=1e-6)
    assert g.shape == (4, 9)


def test_cached_score_lambda_zero_is_mean_ranking():
    q_mc = make_z_mc(CACHED_T, 5, CACHED_DIM, seed=2)
    Z_mc = make_z_mc(CACHED_T, 11, CACHED_DIM, seed=60)
    g0, s_hat, _ = score_cached_mcdp(q_mc, Z_mc, 0.0)
    assert torch.equal(g0, s_hat), "lambda=0 must be exactly the mean score"
    assert torch.equal(g0.argsort(dim=-1), s_hat.argsort(dim=-1))


def test_cached_score_chunked_equals_unchunked():
    q_mc = make_z_mc(CACHED_T, 6, CACHED_DIM, seed=3)
    Z_mc = make_z_mc(CACHED_T, 23, CACHED_DIM, seed=70)
    g_full, _, _ = score_cached_mcdp(q_mc, Z_mc, 0.5)
    for chunk in (1, 4, 7, 23, 64):
        g_c, _, _ = score_cached_mcdp(q_mc, Z_mc, 0.5, chunk_size=chunk)
        assert torch.allclose(g_full, g_c, atol=1e-5), f"chunk_size={chunk} diverged"
        # the contract is on SELECTED DOC IDS, so check the argmax too
        assert torch.equal(g_full.argmax(dim=1), g_c.argmax(dim=1)), \
            f"chunk_size={chunk} changed the selected doc"


def test_positives_masked_before_selection():
    cache, Z_mc, cfg, c_ids, lookup, student, tok, _ = _mk_cached(B_doc=20)
    qids = [f"q{i}" for i in range(5)]
    qid_to_text = {q: f"query {q}" for q in qids}
    # mask each query's would-be top-1 so masking demonstrably changes selection
    q_mc, _ = encode_queries_mc(student, tok, [qid_to_text[q] for q in qids],
                                CACHED_T, DEVICE, cfg)
    g, _, _ = score_cached_mcdp(q_mc, Z_mc, cfg['lambda_val'])
    top1 = g.argmax(dim=1)
    qrels = {q: {cache.docids[int(top1[i])]} for i, q in enumerate(qids)}
    masked = cache.mask_positives(g.clone(), qids, qrels, inplace=False)
    for i, q in enumerate(qids):
        assert masked[i, int(top1[i])] == float('-inf'), "positive not masked"
    slots, docids = cache.select(masked, m=1, mode='topk')
    for i, q in enumerate(qids):
        assert docids[i][0] not in qrels[q], "masked positive was selected"


def test_dropout_only_restores_every_module_mode():
    model = torch.nn.Sequential(
        torch.nn.Linear(4, 4), torch.nn.Dropout(0.5), torch.nn.BatchNorm1d(4))
    for entry_train in (True, False):
        model.train(entry_train)
        before = [m.training for m in model.modules()]
        with dropout_only(model):
            inside = {type(m).__name__: m.training for m in model.modules()}
            assert inside['Dropout'] is True, "dropout must be active for MC passes"
            assert inside['BatchNorm1d'] is False, \
                "only Dropout may be in train mode (no stateful training modules)"
        assert [m.training for m in model.modules()] == before, \
            f"entry modes not restored (entry train={entry_train})"
    # restored even when the body raises
    model.train(False)
    try:
        with dropout_only(model):
            raise RuntimeError("boom")
    except RuntimeError:
        pass
    assert [m.training for m in model.modules()] == [False] * 4


# ---- initialization (D4) ---------------------------------------------------

def test_init_has_t_genuine_stochastic_states():
    cache, Z_mc, cfg, c_ids, lookup, student, tok, stats = _mk_cached()
    assert Z_mc.shape == (CACHED_T, cache.B_doc, CACHED_DIM)
    # the T states of a document must be genuinely different draws, not one
    # deterministic embedding repeated T times
    for t in range(1, CACHED_T):
        assert not torch.allclose(Z_mc[0], Z_mc[t], atol=1e-6), \
            f"pass {t} is identical to pass 0 — cache holds repeated deterministic states"
    assert torch.allclose(cache.Z_student.float(),
                          Z_mc.float().mean(dim=0), atol=1e-2), \
        "Z_mean is not mean_t(Z_mc)"
    assert int(cache.last_refreshed_step.max()) == 0
    assert stats['init_examples_encoded'] == cache.B_doc * CACHED_T
    assert stats['init_mc_passes'] == CACHED_T
    assert stats['cache_mc_bytes'] > 0


def test_z_student_is_aliased_to_z_mean_and_tracks_refresh():
    """Z_student must BE Z_mean (no third bank), and refresh must mutate it in
    place so cheap_scores / _plan_actions never see contradictory doc states."""
    cache, Z_mc, cfg, c_ids, lookup, student, tok, _ = _mk_cached()
    reservoir = _cached_reservoir(student, tok, cfg)
    assert cache.Z_teacher is None, "cached-MCDP is teacher-free"
    before = cache.Z_student.clone()
    _force_maintenance_work(cache)
    c = maintain_interval_cached_mcdp(cache, Z_mc, student, tok, lookup, c_ids,
                                      reservoir, CACHED_STEP, CACHED_T, cfg,
                                      DEVICE, qrels_dict={})
    assert c['num_refresh'] + c['num_replace'] > 0
    assert not torch.allclose(before, cache.Z_student), \
        "Z_student did not change after refresh/replace — alias is stale"
    touched = (cache.last_refreshed_step == CACHED_STEP).nonzero().flatten()
    assert len(touched) > 0
    for s in touched.tolist():
        assert torch.allclose(cache.Z_student[s].float(),
                              Z_mc[:, s, :].float().mean(dim=0), atol=1e-2), \
            f"slot {s}: Z_student out of sync with mean_t(Z_mc)"


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
    # --- step semantics (sequential maintain API) ---
    ("all intervals stamp source_checkpoint_step", test_all_intervals_use_same_source_checkpoint_step),
    ("utility advances while step fixed", test_utility_advances_while_step_fixed),
    ("last_refreshed_step tied to ckpt, not counter", test_last_refreshed_step_tied_to_checkpoint_not_counter),
    ("interval budget uses checkpoint step", test_interval_budget_uses_checkpoint_step),
    # --- cadence: mined-query trigger ---
    ("cadence: interval = update_interval * batch_size", test_maintenance_interval_is_update_interval_times_batch_size),
    ("cadence: fires on mined queries, carries remainder", test_cadence_fires_on_mined_queries_and_carries_remainder),
    ("cadence: partial batch does not fire early", test_cadence_partial_batch_does_not_fire_early),
    # --- cached-MCDP maintenance ---
    ("budget identical across a round's intervals", test_budget_identical_across_every_interval_of_a_round),
    ("utility folded before planning", test_utility_folded_before_planning),
    ("replacement candidates encoded once, reused", test_replacement_candidates_encoded_once_and_reused),
    ("round end: final interval only if pending", test_round_end_final_interval_only_when_pending_state_exists),
    ("metadata aggregates; mining encodes 0 docs", test_metadata_totals_aggregate_and_mining_encodes_no_docs),
    # --- cached scoring contract ---
    ("score: mean + population std", test_cached_score_matches_explicit_mean_and_population_std),
    ("score: lambda=0 == mean ranking", test_cached_score_lambda_zero_is_mean_ranking),
    ("score: chunked == unchunked", test_cached_score_chunked_equals_unchunked),
    ("select: positives masked first", test_positives_masked_before_selection),
    ("dropout_only restores every module mode", test_dropout_only_restores_every_module_mode),
    # --- initialization ---
    ("init: T genuine stochastic states", test_init_has_t_genuine_stochastic_states),
    ("init: Z_student aliases Z_mean", test_z_student_is_aliased_to_z_mean_and_tracks_refresh),
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
