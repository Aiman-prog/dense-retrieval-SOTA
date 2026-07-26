"""
Async Fast-GRASS — cached-MCDP reference implementation (Phase 0).

This is the **feasibility-gate** implementation of the cached-MCDP miner described
in ``async_fast_grass_architecture.md`` / ``async_fast_grass_implementation_details.md``.
It exists so the Phase-0 timing harness and CPU tests can measure and check the
algorithm the docs actually specify, WITHOUT yet committing it to
``src/utils/negative_cache.py``. Phase 1 ports these bodies onto ``NegativeCache``
as ``score_cached_mcdp`` / ``maintain_cached_mcdp``.

Nothing here edits ``negative_cache.py``, ``run_fast_grass.py``, or ``helpers.py``.

Cached-MCDP vs. the sequential lazy top-``L`` MCDP (``run_fast_grass._mine_batch_mcdp``):

    lazy MCDP : cheap-score H -> top-L -> encode query AND those L doc texts
                with T dropout passes, EVERY batch.  (doc encodes per batch > 0)
    cached    : Z_mc[T, B_doc, D] is stored; mining does T fresh stochastic QUERY
                encodes and T matmuls over ALL of H.  (doc encodes per batch == 0)

Document MC states are created at initialization and updated only by periodic
in-round cache maintenance.

Encoder accounting vocabulary (used consistently by the timing script and tests):
``*_mc_passes`` is the logical pass count ``T``; ``*_examples_encoded`` is
``n_texts * T``; ``*_forward_batches`` is the number of real encoder forward calls,
``ceil(n_texts / mc_batch_size) * T``. ``B*T`` is examples, NOT encoder calls.
"""
import contextlib
import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root / 'src'))

from utils.helpers import encode_batch_tensor  # noqa: E402


# ---- stochastic encoding ---------------------------------------------------

@contextlib.contextmanager
def dropout_only(model):
    """Put ``model`` in eval mode but re-enable ONLY ``nn.Dropout`` modules.

    "Frozen for the round" means no parameter updates and no gradients, not
    dropout-off (async_fast_grass_implementation_details.md, "Miner Loop"). Using
    ``model.train()`` would also switch on any other stateful training-mode module
    (BatchNorm running stats, etc.), which the miner must not do.

    Every module's entry mode is captured and restored on exit, including on
    exception.
    """
    entry_modes = {mod: mod.training for mod in model.modules()}
    try:
        model.eval()
        for mod in model.modules():
            if isinstance(mod, nn.Dropout):
                mod.train(True)
        yield model
    finally:
        for mod, was_training in entry_modes.items():
            mod.train(was_training)


def encode_mc(model, tokenizer, texts, T, device, max_len, batch_size,
              dtype=None):
    """Encode ``texts`` through ``T`` genuine dropout passes -> ``[T, n, D]``.

    Each pass is a separate stochastic forward over the full text list, so the
    ``T`` states of a document are independent dropout samples rather than a
    repeated deterministic embedding. No gradients.

    Returns ``(Z, stats)`` where ``stats`` carries the three-way encoder
    accounting for this call.
    """
    if T < 1:
        raise ValueError(f"T must be >= 1, got {T}")
    passes = []
    with dropout_only(model):
        for _ in range(int(T)):
            z = encode_batch_tensor(model, tokenizer, texts, device, max_len,
                                    batch_size, requires_grad=False)
            passes.append(z.detach())
    Z = torch.stack(passes, dim=0)
    if dtype is not None:
        Z = Z.to(dtype=dtype)
    n = len(texts)
    stats = {
        'mc_passes': int(T),
        'examples_encoded': int(n * T),
        'forward_batches': int(math.ceil(n / max(batch_size, 1)) * T) if n else 0,
    }
    return Z, stats


def encode_queries_mc(model, tokenizer, texts, T, device, cfg, dtype=None):
    """``T`` stochastic query passes -> ``[T, B_query, D]``. Thin cfg wrapper."""
    return encode_mc(model, tokenizer, texts, T, device,
                     cfg.get('query_max_len', 128),
                     cfg.get('mc_batch_size', 256), dtype=dtype)


def encode_docs_mc(model, tokenizer, texts, T, device, cfg, dtype=None):
    """``T`` stochastic document passes -> ``[T, n_docs, D]``. Thin cfg wrapper."""
    return encode_mc(model, tokenizer, texts, T, device,
                     cfg.get('passage_max_len', 512),
                     cfg.get('mc_batch_size', 256), dtype=dtype)


# ---- cached-MCDP scoring ---------------------------------------------------

def score_cached_mcdp(q_mc, Z_mc, lambda_val, chunk_size=None):
    """Score a query batch against ALL cache slots from cached MC states.

    ``q_mc``  ``[T, B_query, D]`` — fresh stochastic query states (current ckpt).
    ``Z_mc``  ``[T, B_doc,   D]`` — cached stochastic document states.

    For pass ``t``::

        s_t   = q_mc[t] @ Z_mc[t].T
        s_hat = mean_t(s_t)
        sigma = sqrt(mean_t((s_t - s_hat)^2))     # population std, correction=0
        g     = s_hat + lambda_val * sigma

    Returns ``(g, s_hat, sigma)``, each ``[B_query, B_doc]`` in FP32. All three are
    returned because the signal probe needs ``s_hat`` and ``sigma`` separately, not
    just ``g``.

    The pass index simply pairs independent query and cached document dropout
    samples; no extra document passes and no fresh shortlist are created.

    ``chunk_size`` chunks over cache slots so the ``T`` full score matrices need not
    be resident at once. Moments are computed in FP32 within each chunk, so chunked
    and unchunked results agree to floating-point tolerance.

    Scoring is grad-free. Callers own ``cache_score_pairs`` accounting.
    """
    if q_mc.dim() != 3 or Z_mc.dim() != 3:
        raise ValueError(f"expected [T,B,D] tensors, got q_mc{tuple(q_mc.shape)} "
                         f"Z_mc{tuple(Z_mc.shape)}")
    if q_mc.shape[0] != Z_mc.shape[0]:
        raise ValueError(f"T mismatch: q_mc has {q_mc.shape[0]} passes, Z_mc has "
                         f"{Z_mc.shape[0]}")
    if q_mc.shape[2] != Z_mc.shape[2]:
        raise ValueError(f"dim mismatch: q_mc D={q_mc.shape[2]}, Z_mc D={Z_mc.shape[2]}")

    T, B_query, _ = q_mc.shape
    B_doc = Z_mc.shape[1]
    step = int(chunk_size) if chunk_size else B_doc
    if step < 1:
        raise ValueError(f"chunk_size must be >= 1, got {chunk_size}")

    s_hat = torch.empty((B_query, B_doc), dtype=torch.float32, device=q_mc.device)
    sigma = torch.empty((B_query, B_doc), dtype=torch.float32, device=q_mc.device)

    with torch.no_grad():
        for start in range(0, B_doc, step):
            end = min(start + step, B_doc)
            Zc = Z_mc[:, start:end, :]
            # [T, B_query, C] — exact two-pass mean/std per chunk (T is small), which
            # is numerically safer than accumulating sum/sum-of-squares.
            S = torch.stack(
                [(q_mc[t].to(Zc.dtype) @ Zc[t].t()).float() for t in range(T)],
                dim=0)
            s_hat[:, start:end] = S.mean(dim=0)
            sigma[:, start:end] = S.std(dim=0, unbiased=False)

    g = s_hat + float(lambda_val) * sigma
    return g, s_hat, sigma


# ---- cache initialization --------------------------------------------------

def init_Z_mc(cache, corpus_lookup, model, tokenizer, T, cfg, device):
    """Build ``Z_mc[T, B_doc, D]`` for an existing ``NegativeCache`` slot layout.

    The initial cache cannot repeat a deterministic embedding ``T`` times
    (impl-details, "Cache State And Initialization"): every cached document is
    encoded through ``T`` genuine dropout passes with the current (base) model.

    ``Z_mean = mean_t(Z_mc)`` is **aliased onto ``cache.Z_student``** rather than
    allocated as a third bank, per the "Cached-MCDP Scoring Contract". Maintenance
    mutates it in place so the alias stays valid and ``cheap_scores`` /
    ``_plan_actions`` never see document states that contradict ``Z_mc``.

    Returns ``(Z_mc, stats)``.
    """
    texts = [corpus_lookup[d] for d in cache.docids]
    Z_mc, enc = encode_docs_mc(model, tokenizer, texts, T, device, cfg,
                               dtype=cache.Z_student.dtype)
    Z_mean = Z_mc.float().mean(dim=0).to(cache.Z_student.dtype)

    # alias, not a copy: Z_student IS Z_mean from here on.
    cache.Z_student = Z_mean
    cache.Z_student.requires_grad_(False)
    cache.last_refreshed_step[:] = 0

    stats = {
        'init_docs': len(texts),
        'init_mc_passes': enc['mc_passes'],
        'init_examples_encoded': enc['examples_encoded'],
        'init_forward_batches': enc['forward_batches'],
        'cache_mc_bytes': int(Z_mc.element_size() * Z_mc.nelement()),
        'cache_mean_bytes': int(Z_mean.element_size() * Z_mean.nelement()),
    }
    return Z_mc, stats


# ---- recent query-MC reservoir ---------------------------------------------

class QueryMCReservoir:
    """Bounded rolling buffer of recent no-grad query MC states + their qids.

    Conceptually ``[T, R_query, D]`` plus query IDs for qrel masking during
    replacement recertification (arch doc, "Cache Maintenance Semantics").
    """

    def __init__(self, size):
        self.size = int(size)
        self._q = None          # [T, R, D]
        self._qids = []

    def add(self, q_mc, qids):
        q_mc = q_mc.detach()
        if self._q is None:
            self._q = q_mc
            self._qids = list(qids)
        else:
            self._q = torch.cat([self._q, q_mc], dim=1)
            self._qids = self._qids + list(qids)
        if self._q.shape[1] > self.size:
            self._q = self._q[:, -self.size:, :]
            self._qids = self._qids[-self.size:]

    def get(self):
        """Returns ``(q_mc[T,R,D], qids)`` or ``(None, [])`` when empty."""
        return self._q, list(self._qids)

    def __len__(self):
        return 0 if self._q is None else int(self._q.shape[1])


# ---- maintenance cadence ---------------------------------------------------

def maintenance_interval_mined_queries(cfg, batch_size=None):
    """``cache_update_interval * trainer_batch_size`` (e.g. 100 * 64 = 6400).

    ``cache_update_interval`` keeps its sequential trainer-step meaning inside the
    budget formula; only the miner's *execution trigger* converts to mined query
    examples (arch doc, "Cache Maintenance Semantics").
    """
    bs = batch_size if batch_size is not None else cfg.get('batch_size', 64)
    return int(cfg['cache_update_interval']) * int(bs)


class MaintenanceDriver:
    """Fires one bounded maintenance interval every N mined query examples.

    The trigger is ``>=`` on a running mined-query counter. When it fires the
    threshold is **subtracted** rather than the counter reset to zero, so a batch
    that overshoots carries its remainder into the next interval and the long-run
    cadence stays exact.
    """

    def __init__(self, cfg, batch_size=None):
        self.threshold = maintenance_interval_mined_queries(cfg, batch_size)
        if self.threshold < 1:
            raise ValueError(f"maintenance interval must be >= 1, got {self.threshold}")
        self.counter = 0
        self.n_intervals = 0
        self.mined_total = 0

    def add(self, n_queries):
        self.counter += int(n_queries)
        self.mined_total += int(n_queries)

    def should_fire(self):
        return self.counter >= self.threshold

    def consume(self):
        """Account one fired interval; subtract (do not reset) the threshold."""
        self.counter -= self.threshold
        self.n_intervals += 1

    @property
    def pending(self):
        """Un-consumed mined queries left over at round end."""
        return self.counter

    def round_end_should_maintain(self, cache):
        """Final bounded interval at round end only if useful pending state exists.

        "Fold remaining ``selected_indicator`` if non-empty; run one final bounded
        maintenance interval only if useful pending state exists" (arch doc). A
        round that ended exactly on an interval boundary with nothing selected
        since must NOT pay for an extra maintenance pass.
        """
        return bool(self.counter > 0 and bool(cache.selected_indicator.any()))


# ---- cached-MCDP maintenance ----------------------------------------------

def _zero_maintenance_counters():
    return dict(
        num_refresh=0, num_replace=0, num_over_age=0, over_age_backlog=0,
        num_R_candidates=0, num_uniform_candidates=0,
        num_recertified_candidates=0,
        maintenance_docs_encoded=0, maintenance_mc_passes=0,
        maintenance_examples_encoded=0, maintenance_forward_batches=0,
        cache_turnover_rate=0.0)


def maintain_interval_cached_mcdp(cache, Z_mc, student, tokenizer, corpus_lookup,
                                  all_docids, reservoir, source_checkpoint_step,
                                  T, cfg, device, qrels_dict=None):
    """One bounded cached-MCDP maintenance interval.

    ``source_checkpoint_step`` is **model time** and is identical for every interval
    in a mining round: the miner holds checkpoint-frozen weights, so cache age, the
    rho/progress budget, and ``last_refreshed_step`` must not move within a round.
    Passing a miner-local counter (batch index, mined-query count, interval index)
    would corrupt all three.

    Differences from ``NegativeCache.maintain`` (which stays untouched):
      * refresh encodes each document ``T`` times, not once;
      * **every** replacement candidate is encoded ``T`` times exactly once, and
        recertification scores those states against the query-MC reservoir with the
        same mean+uncertainty ``g`` as normal mining (not a deterministic dot);
      * chosen candidates are inserted by reusing the already-computed states, so
        insertion adds no encoder calls.

    Planning is delegated to the audited ``NegativeCache`` internals
    (``_update_utility`` -> ``_interval_budget`` -> ``_plan_actions``) rather than
    forked, so budget/eligibility semantics cannot drift from the sequential path.

    ``reservoir`` is ``(q_mc[T,R,D], qids)``. When it is empty, maintenance refreshes
    existing slots but skips replacement recertification.

    Returns a counter dict. Mutates ``Z_mc`` and ``cache`` in place.
    """
    qrels_dict = qrels_dict or {}
    entry_mode = student.training
    counters = _zero_maintenance_counters()
    try:
        # 1. Fold utility BEFORE planning — same order as NegativeCache.maintain
        #    (negative_cache.py:308-310). Without this, selected_indicator,
        #    utility_ema and intervals_since_selected never advance and the
        #    eligibility rules see frozen state.
        cache._update_utility(cfg)

        budget = cache._interval_budget(source_checkpoint_step, cfg)
        refresh_slots, replace_slots, diag = cache._plan_actions(
            source_checkpoint_step, cfg, budget)
        counters.update(diag)
        counters['maintenance_budget_interval'] = int(budget)

        # 2. Refresh: T stochastic passes per document; Z_mc, Z_mean and the
        #    timestamp move together.
        if len(refresh_slots):
            slot_list = refresh_slots.tolist()
            texts = [corpus_lookup[cache.docids[s]] for s in slot_list]
            Zr, enc = encode_docs_mc(student, tokenizer, texts, T, device, cfg,
                                     dtype=Z_mc.dtype)
            Z_mc[:, refresh_slots, :] = Zr
            # in-place so the Z_student alias stays valid
            cache.Z_student[refresh_slots] = Zr.float().mean(dim=0).to(
                cache.Z_student.dtype)
            cache.last_refreshed_step[refresh_slots] = source_checkpoint_step
            counters['num_refresh'] = len(slot_list)
            counters['maintenance_docs_encoded'] += len(texts)
            counters['maintenance_mc_passes'] = enc['mc_passes']
            counters['maintenance_examples_encoded'] += enc['examples_encoded']
            counters['maintenance_forward_batches'] += enc['forward_batches']

        # 3. Replace (needs the query-MC reservoir for recertification).
        q_res, res_qids = reservoir if reservoir is not None else (None, [])
        if len(replace_slots) and q_res is not None and len(res_qids):
            rc = _replace_cached_mcdp(cache, Z_mc, replace_slots, student,
                                      tokenizer, corpus_lookup, all_docids,
                                      q_res, res_qids, source_checkpoint_step,
                                      T, cfg, device, qrels_dict)
            for k, v in rc.items():
                if k in ('maintenance_docs_encoded', 'maintenance_examples_encoded',
                         'maintenance_forward_batches'):
                    counters[k] += v
                elif k == 'maintenance_mc_passes':
                    counters[k] = v
                else:
                    counters[k] = v

        counters['cache_turnover_rate'] = (
            counters['num_replace'] / cache.B_doc if cache.B_doc else 0.0)
        counters['maintenance_model_step'] = int(source_checkpoint_step)
        return counters
    finally:
        student.train(entry_mode)


def _replace_cached_mcdp(cache, Z_mc, slots, student, tokenizer, corpus_lookup,
                         all_docids, q_res, res_qids, step, T, cfg, device,
                         qrels_dict):
    """Replacement with cached-MCDP recertification.

    Candidate nomination reuses the registry ``R`` + uniform-corpus policy from
    ``NegativeCache`` (uniform stays the binding constraint). Every candidate is
    encoded for ``T`` passes ONCE; those same states are reused on insertion, so
    ``maintenance_docs_encoded`` counts each candidate exactly once.
    """
    num_replace = len(slots)
    num_cand = cfg['replacement_candidate_multiplier'] * num_replace
    num_uniform_target = int(np.ceil(cfg['uniform_candidate_fraction'] * num_cand))
    num_R_target = max(num_cand - num_uniform_target, 0)

    in_H = set(cache.docids)
    R_cands = [d for d in cache.registry.nominate(num_R_target, cache.rng)
               if d in corpus_lookup and d not in in_H]
    uni = cache._sample_uniform(all_docids, num_cand - len(R_cands),
                                exclude=in_H | set(R_cands), corpus=corpus_lookup)
    cand = list(dict.fromkeys(R_cands + uni))
    out = dict(num_replace=0, num_R_candidates=len(R_cands),
               num_uniform_candidates=len(uni), num_recertified_candidates=0,
               maintenance_docs_encoded=0, maintenance_mc_passes=int(T),
               maintenance_examples_encoded=0, maintenance_forward_batches=0)
    if not cand:
        return out

    cand_texts = [corpus_lookup[d] for d in cand]
    Zc, enc = encode_docs_mc(student, tokenizer, cand_texts, T, device, cfg,
                             dtype=Z_mc.dtype)
    out['num_recertified_candidates'] = len(cand)
    out['maintenance_docs_encoded'] = len(cand)
    out['maintenance_examples_encoded'] = enc['examples_encoded']
    out['maintenance_forward_batches'] = enc['forward_batches']

    # recertify with the SAME mean+uncertainty score as ordinary mining
    g, _s_hat, _sigma = score_cached_mcdp(q_res, Zc, cfg['lambda_val'])
    for r, qid in enumerate(res_qids):
        pos = qrels_dict.get(qid)
        if not pos:
            continue
        cols = [j for j, d in enumerate(cand) if d in pos]
        if cols:
            g[r, cols] = float('-inf')

    k = min(int(cfg['reentry_top_k']), g.shape[0])
    reentry = torch.topk(g, k=k, dim=0).values.mean(dim=0)
    finite = torch.isfinite(reentry)
    order = [i for i in torch.argsort(reentry, descending=True).tolist()
             if finite[i]]
    chosen = order[:num_replace]

    for slot, ci in zip(slots.tolist(), chosen):
        evicted = cache.docids[slot]
        cache.registry.admit(evicted, dict(
            peak_utility_ema=float(cache.peak_utility_ema[slot]),
            lifetime_selected_count=int(cache.lifetime_selected_count[slot])), step)
        # reuse the already-computed MC states — no second encode
        Z_mc[:, slot, :] = Zc[:, ci, :]
        # _insert writes Z_student[slot] (== Z_mean[slot]) in place and resets the
        # new slot's utility/history with timestamp = step.
        cache._insert(slot, cand[ci],
                      Zc[:, ci, :].float().mean(dim=0).to(cache.Z_student.dtype),
                      None, step)
        cache.registry.entries.pop(cand[ci], None)

    out['num_replace'] = len(chosen)
    return out


# ---- mining ----------------------------------------------------------------

def mine_batch_cached_mcdp(cache, Z_mc, student, tokenizer, batch_qids,
                           qid_to_text, qrels_dict, T, cfg, device,
                           chunk_size=None):
    """One virtual mining batch: T query passes + T matmuls over all of H.

    Performs **zero document encoder calls** — that invariant is what separates
    cached-MCDP from the lazy top-``L`` path and is asserted by the timing harness
    and the CPU tests.

    Returns ``(mined, slots, q_mc, stats)``.
    """
    texts = [qid_to_text[q] for q in batch_qids]
    q_mc, enc = encode_queries_mc(student, tokenizer, texts, T, device, cfg)

    g, s_hat, sigma = score_cached_mcdp(q_mc, Z_mc, cfg['lambda_val'],
                                        chunk_size=chunk_size)
    cache.cache_score_pairs += len(batch_qids) * cache.B_doc

    g = cache.mask_positives(g, batch_qids, qrels_dict, inplace=True)
    slots, neg_docids = cache.select(g, m=cfg['m'], mode=cfg['selection_mode'],
                                     beta=cfg.get('beta', 5.0), L=None)
    cache.record_selection(slots)

    mined = {qid: neg_docids[i] for i, qid in enumerate(batch_qids)}
    sel_g = torch.gather(g, 1, slots)
    sel_sigma = torch.gather(sigma, 1, slots)
    stats = {
        'query_mc_passes': enc['mc_passes'],
        'query_examples_encoded': enc['examples_encoded'],
        'query_forward_batches': enc['forward_batches'],
        # the defining invariant of cached-MCDP
        'mcdp_doc_encoder_calls_mining': 0,
        'cache_score_pairs_batch': len(batch_qids) * cache.B_doc,
        'sel_g_mean': float(sel_g.float().mean()),
        'sel_sigma_mean': float(sel_sigma.float().mean()),
        's_hat_mean': float(s_hat[torch.isfinite(s_hat)].float().mean()),
        'sigma_mean': float(sigma.float().mean()),
    }
    return mined, slots, q_mc, stats
