"""
Fast-GRASS negative-cache core: ``NegativeCache`` + ``RetiredRegistry``.

Implements the bounded global document cache ``H`` (size ``B_doc``) holding stale
cached document states ``Z_H`` for cheap GRASS-style negative selection:

    g(q, d) = s_hat(q, d) + lambda_val * sigma(q, d)

EMA estimator (v0): ``Z_H`` stores a student doc embedding and an EMA-teacher doc
embedding per slot, so

    s_hat = q_student . d_student_cached
    sigma = | q_student . d_student_cached  -  q_teacher . d_teacher_cached |

Mining is one ``Q_batch x Z_H`` matmul plus selection — the doc encoder is used
only for the fresh loss encodes (in the trainer) and for *amortized* cache
maintenance here. ``Z_H`` is SELECTION-ONLY: gradients never flow through it and
every cache encode runs under ``torch.no_grad()`` (via
``encode_batch_tensor(..., requires_grad=False)``).

Design is fixed by ``fast_grass_negative_cache_architecture.md`` and the v0
defaults in ``fast_grass_implementation_details.md``. This module is what the
Phase-2 trainer (``run_fast_grass.py``) imports; nothing here is throwaway.
"""
from pathlib import Path

import numpy as np
import torch

from utils.helpers import encode_batch_tensor, encode_mc


def linear_decay(start, end, progress):
    """``start`` at progress<=0, ``end`` at progress>=1, linear in between."""
    p = min(max(float(progress), 0.0), 1.0)
    return start + (end - start) * p


def score_cached_mcdp(q_mc, Z_mc, lambda_val, chunk_size=None):
    """Cached-MCDP score of query MC states against a bank of document MC states.

    ``q_mc`` ``[T, B_query, D]``, ``Z_mc`` ``[T, B_doc, D]``. For pass ``t``::

        s_t   = q_mc[t] @ Z_mc[t].T
        s_hat = mean_t(s_t)
        sigma = sqrt(mean_t((s_t - s_hat)^2))     # population std, correction=0
        g     = s_hat + lambda_val * sigma

    Returns ``(g, s_hat, sigma)``, each ``[B_query, B_doc]`` in FP32 — all three,
    because the signal probe needs ``s_hat`` and ``sigma`` separately.

    The pass index simply pairs independent query and cached document dropout
    samples; no extra document passes and no fresh shortlist are created.

    ``chunk_size`` chunks over document slots so the ``T`` full score matrices need
    not be resident at once; moments are computed in FP32 within each chunk, so
    chunked and unchunked agree to floating-point tolerance.

    Free function so ``NegativeCache.score_cached_mcdp`` (over ``H``) and
    recertification (over a candidate bank) share one implementation, and so tests
    can score raw synthetic tensors without building a cache.
    """
    if q_mc.dim() != 3 or Z_mc.dim() != 3:
        raise ValueError(f"expected [T,B,D] tensors, got q_mc{tuple(q_mc.shape)} "
                         f"Z_mc{tuple(Z_mc.shape)}")
    if q_mc.shape[0] != Z_mc.shape[0]:
        raise ValueError(f"T mismatch: q_mc has {q_mc.shape[0]} passes, Z_mc has "
                         f"{Z_mc.shape[0]}")
    if q_mc.shape[2] != Z_mc.shape[2]:
        raise ValueError(f"dim mismatch: q_mc D={q_mc.shape[2]}, "
                         f"Z_mc D={Z_mc.shape[2]}")

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
            # [T, B_query, C] — exact two-pass mean/std per chunk (T is small),
            # numerically safer than accumulating sum / sum-of-squares.
            S = torch.stack(
                [(q_mc[t].to(Zc.dtype) @ Zc[t].t()).float() for t in range(T)],
                dim=0)
            s_hat[:, start:end] = S.mean(dim=0)
            sigma[:, start:end] = S.std(dim=0, unbiased=False)

    return s_hat + float(lambda_val) * sigma, s_hat, sigma


class RetiredRegistry:
    """Metadata-only registry of evicted-but-previously-useful documents.

    Never stores embeddings and never participates in per-batch query scoring; it
    only *nominates* docids as replacement candidates during cache maintenance
    (current-model recertification then decides re-entry). Bounded at
    ``R_size_factor * B_doc`` entries.
    """

    def __init__(self, max_size, utility_remember_threshold):
        self.max_size = int(max_size)
        self.remember_threshold = float(utility_remember_threshold)
        # docid -> {peak_utility_ema, lifetime_selected_count, last_seen_step}
        self.entries = {}

    def admit(self, docid, meta, step):
        """Admit an evicted doc only if it previously showed usefulness."""
        if not (meta['lifetime_selected_count'] > 0 or
                meta['peak_utility_ema'] >= self.remember_threshold):
            return False
        e = self.entries.get(docid)
        if e is None:
            self.entries[docid] = {
                'peak_utility_ema': float(meta['peak_utility_ema']),
                'lifetime_selected_count': int(meta['lifetime_selected_count']),
                'last_seen_step': int(step),
            }
        else:  # re-eviction: keep the strongest history seen
            e['peak_utility_ema'] = max(e['peak_utility_ema'],
                                        float(meta['peak_utility_ema']))
            e['lifetime_selected_count'] += int(meta['lifetime_selected_count'])
            e['last_seen_step'] = int(step)
        self._evict_if_full()
        return True

    def _evict_if_full(self):
        if len(self.entries) <= self.max_size:
            return
        ranked = sorted(
            self.entries.items(),
            key=lambda kv: (kv[1]['peak_utility_ema'],
                            kv[1]['lifetime_selected_count']),
            reverse=True)
        self.entries = dict(ranked[:self.max_size])

    def nominate(self, n, rng):
        """Sample up to ``n`` docids to nominate as replacement candidates."""
        if n <= 0 or not self.entries:
            return []
        docids = list(self.entries.keys())
        if len(docids) <= n:
            return list(docids)
        idx = rng.choice(len(docids), size=int(n), replace=False)
        return [docids[i] for i in idx]

    def __len__(self):
        return len(self.entries)


class NegativeCache:
    """Bounded global cache ``H`` of ``B_doc`` docs with stale states ``Z_H``.

    Slots ``0..B_doc-1`` each hold one docid, a student doc embedding
    (``Z_student``) and — in EMA mode only — an EMA-teacher doc embedding
    (``Z_teacher``; ``None`` for the teacher-free MCDP estimator), plus
    maintenance metadata (age, utility EMA, selection history). All tensors live
    on ``device``; ``Z_*`` carry no gradient.
    """

    # ---- construction -------------------------------------------------------

    def __init__(self, docids, Z_student, Z_teacher, cfg, device, dim, rng):
        self.device = device
        self.dim = int(dim)
        self.cfg = cfg
        self.rng = rng
        self.B_doc = len(docids)

        self.docids = list(docids)
        self.docid_to_slot = {d: i for i, d in enumerate(self.docids)}

        self.Z_student = Z_student.detach()   # (B_doc, dim) — selection only
        self.Z_student.requires_grad_(False)
        # Z_teacher is EMA-only. MCDP is teacher-free: Z_teacher is None and σ comes
        # from top-L MC-dropout in the trainer, not from a cached teacher state.
        if Z_teacher is not None:
            self.Z_teacher = Z_teacher.detach()
            self.Z_teacher.requires_grad_(False)
        else:
            self.Z_teacher = None

        z_long = lambda: torch.zeros(self.B_doc, dtype=torch.long, device=device)
        z_flt = lambda: torch.zeros(self.B_doc, dtype=torch.float32, device=device)
        self.last_refreshed_step = z_long()
        self.utility_ema = z_flt()
        self.peak_utility_ema = z_flt()
        self.selected_indicator = torch.zeros(self.B_doc, dtype=torch.bool,
                                              device=device)
        self.selected_count_recent = z_long()
        self.lifetime_selected_count = z_long()
        self.intervals_since_selected = z_long()

        # R_doc knobs default to a ZERO-SIZE registry so a config that omits them
        # (the async cached-MCDP block, where R_doc is deferred) structurally
        # cannot admit anything. The sequential/EMA config supplies both keys, so
        # its behaviour is unchanged.
        self.registry = RetiredRegistry(
            max_size=cfg.get('R_size_factor', 0) * self.B_doc,
            utility_remember_threshold=cfg.get('utility_remember_threshold', 0.05))

        # Seeded torch RNG so Gumbel-Softmax selection is reproducible (does not
        # touch the global torch RNG state).
        self._gen = torch.Generator(device=self.device)
        self._gen.manual_seed(int(cfg['cache_init_seed']))

        # cumulative scoring cost (maintenance cost is returned per-call instead)
        self.cache_score_pairs = 0

        # cached-MCDP state: [T, B_doc, dim] stochastic doc samples. None for the
        # EMA/sequential estimator; set by init_cached_mcdp / attach_mc_states.
        self.Z_mc = None
        self.T = None

    @classmethod
    def init_cached_mcdp(cls, stale_embs, c_ids, corpus_lookup, model, tokenizer,
                         cfg, device, dim=None, dtype=None):
        """Build a cached-MCDP cache: ``Z_mc[T, B_doc, D]`` from genuine dropout passes.

        Slot sampling reuses ``init_uniform`` (the stale pickle supplies docid
        ordering only). Every sampled document is then encoded through ``T`` real
        dropout passes with ``model`` — the initial cache must NOT repeat a
        deterministic embedding ``T`` times
        (async_fast_grass_implementation_details.md, "Cache State And Initialization").

        ``Z_mean = mean_t(Z_mc)`` is **aliased onto ``Z_student``** rather than
        allocated as a third bank ("Cached-MCDP Scoring Contract"), so
        ``cheap_scores`` / ``_plan_actions`` can never see document states that
        contradict ``Z_mc``. Refresh and replacement mutate it in place.

        Returns ``(cache, stats)``.
        """
        cache = cls.init_uniform(stale_embs, c_ids, cfg, device, dim=dim, dtype=dtype)
        T = int(cfg['T'])
        texts = [corpus_lookup[d] for d in cache.docids]
        Z_mc, enc = encode_mc(model, tokenizer, texts, T, device,
                              cfg.get('passage_max_len', 512),
                              cache._miner_mc_batch_size(cfg),
                              dtype=cache.Z_student.dtype)
        cache.attach_mc_states(Z_mc)
        cache.last_refreshed_step[:] = 0
        stats = {
            'init_docs': len(texts),
            'init_mc_passes': enc['mc_passes'],
            'init_examples_encoded': enc['examples_encoded'],
            'init_forward_batches': enc['forward_batches'],
            'cache_mc_bytes': cache.mc_memory_bytes(),
        }
        return cache, stats

    def attach_mc_states(self, Z_mc):
        """Install ``Z_mc`` and re-point ``Z_student`` at ``Z_mean = mean_t(Z_mc)``.

        Used by ``init_cached_mcdp`` and by ``load_state`` on restart.
        """
        if Z_mc.dim() != 3 or Z_mc.shape[1] != self.B_doc:
            raise ValueError(
                f"Z_mc must be [T, B_doc={self.B_doc}, D], got {tuple(Z_mc.shape)}")
        self.Z_mc = Z_mc.detach()
        self.Z_mc.requires_grad_(False)
        self.T = int(Z_mc.shape[0])
        Z_mean = Z_mc.float().mean(dim=0).to(Z_mc.dtype)
        # alias, not a copy: Z_student IS Z_mean from here on
        self.Z_student = Z_mean
        self.Z_student.requires_grad_(False)
        self.Z_teacher = None      # cached-MCDP is teacher-free

    @property
    def is_cached_mcdp(self):
        return getattr(self, 'Z_mc', None) is not None

    @staticmethod
    def _miner_mc_batch_size(cfg):
        """Miner-side MC encode batch, falling back to the shared encode batch."""
        return int(cfg.get('miner_mc_batch_size') or cfg.get('mc_batch_size', 256))

    def mc_memory_bytes(self):
        if not self.is_cached_mcdp:
            return 0
        return int(self.Z_mc.element_size() * self.Z_mc.nelement())

    @classmethod
    def init_uniform(cls, stale_embs, c_ids, cfg, device, dim=None, dtype=None):
        """Initialize ``H`` by uniformly sampling ``B_doc`` corpus docs (seeded).

        ``stale_embs[i]`` must be the embedding of ``c_ids[i]`` (the ordered
        ``c_ids <-> rows`` mapping from the stale-index pickle). At init the
        student and teacher states are identical (the EMA teacher == student).
        ``dtype`` defaults to bf16 on CUDA (halves the Z_H footprint) and float32
        on CPU. A teacher state is allocated only for the EMA estimator
        (``cfg['uncertainty'] == 'ema'``); MCDP is teacher-free (``Z_teacher=None``).
        """
        if dtype is None:
            dtype = torch.bfloat16 if device.type == 'cuda' else torch.float32
        rng = np.random.default_rng(cfg['cache_init_seed'])
        n_corpus = len(c_ids)
        B_doc = min(int(cfg['B_doc']), n_corpus)
        sample = np.sort(rng.choice(n_corpus, size=B_doc, replace=False))
        docids = [c_ids[i] for i in sample]

        embs = stale_embs[sample]
        if not torch.is_tensor(embs):
            embs = torch.as_tensor(np.asarray(embs))
        embs = torch.nn.functional.normalize(embs.to(device).float(), dim=-1)
        embs = embs.to(dtype=dtype)
        if dim is None:
            dim = embs.shape[1]
        with_teacher = cfg.get('uncertainty', 'ema') == 'ema'
        Z_teacher = embs.clone() if with_teacher else None
        return cls(docids, embs.clone(), Z_teacher, cfg, device, dim, rng)

    # ---- mining: score / mask / select -------------------------------------

    def score(self, q_student, q_teacher, lambda_val):
        """Score a query batch against all of ``H``.

        Returns ``(g, s_student, sigma)``, each ``(batch, B_doc)``. ``s_student``
        and ``sigma`` are returned for logging. Selection is grad-free: scoring
        runs under ``no_grad`` so a grad-enabled query batch never builds a graph
        through the selection-only cache. EMA-only — raises for a teacher-free
        (MCDP) cache.
        """
        if self.Z_teacher is None:
            raise RuntimeError(
                "score() requires an EMA cache with Z_teacher; use cheap_scores() "
                "for MCDP")
        with torch.no_grad():
            s_student = q_student @ self.Z_student.t()
            s_teacher = q_teacher @ self.Z_teacher.t()
            sigma = (s_student - s_teacher).abs()
            g = s_student + lambda_val * sigma
        self.cache_score_pairs += int(q_student.shape[0]) * self.B_doc
        return g, s_student, sigma

    def cheap_scores(self, q_student):
        """Deterministic student-only cheap scores over all of ``H`` — MCDP's
        top-L ranking stage. ``q_student @ Z_student.t()`` under ``no_grad``;
        returns ``(batch, B_doc)`` and counts the scored pairs like ``score()``.
        Needs no teacher, so it works regardless of estimator.
        """
        with torch.no_grad():
            s = q_student @ self.Z_student.t()
        self.cache_score_pairs += int(q_student.shape[0]) * self.B_doc
        return s

    def score_cached_mcdp(self, q_mc, lambda_val, chunk_size=None):
        """Score a query batch against ALL cache slots from cached MC states.

        ``q_mc`` is ``[T, B_query, D]`` — fresh stochastic query states from the
        current (frozen) checkpoint. For pass ``t``::

            s_t   = q_mc[t] @ Z_mc[t].T
            s_hat = mean_t(s_t)
            sigma = sqrt(mean_t((s_t - s_hat)^2))     # population std, correction=0
            g     = s_hat + lambda_val * sigma

        Returns ``(g, s_hat, sigma)``, each ``[B_query, B_doc]`` in FP32 — all three,
        because the signal probe needs ``s_hat`` and ``sigma`` separately.

        The pass index simply pairs independent query and cached document dropout
        samples; no extra document passes and no fresh shortlist are created. There
        is no top-``L`` in cached-MCDP: every query is scored against all of ``H``.

        ``chunk_size`` chunks over cache slots so the ``T`` full score matrices need
        not be resident at once; moments are computed in FP32 within each chunk, so
        chunked and unchunked agree to floating-point tolerance.

        Grad-free, and counts scored pairs like ``score()``.
        """
        if not self.is_cached_mcdp:
            raise RuntimeError(
                "score_cached_mcdp() requires a cached-MCDP cache; call "
                "init_cached_mcdp()/attach_mc_states() first")
        g, s_hat, sigma = score_cached_mcdp(q_mc, self.Z_mc, lambda_val,
                                            chunk_size=chunk_size)
        self.cache_score_pairs += int(q_mc.shape[1]) * self.B_doc
        return g, s_hat, sigma

    def mask_positives(self, g, batch_qids, qrels_dict, inplace=False):
        """Set ``g[i, slot] = -inf`` for slots holding a known positive of query i."""
        if not inplace:
            g = g.clone()
        for i, qid in enumerate(batch_qids):
            pos = qrels_dict.get(qid)
            if not pos:
                continue
            slots = [self.docid_to_slot[d] for d in pos if d in self.docid_to_slot]
            if slots:
                g[i, slots] = float('-inf')
        return g

    def select(self, g, m, mode='topk', beta=5.0, L=None):
        """Select ``m`` negatives per query. Returns ``(slots, docids)``.

        ``topk``: top-``m`` by ``g``. ``softmax``: Gumbel-top-k without
        replacement over ``beta*g`` (optional top-``L`` prefilter), using the
        cache's seeded RNG. Masked (``-inf``) slots can never be selected.

        Raises ``ValueError`` if any query has fewer than ``m`` finite
        (selectable) slots after masking — selecting a ``-inf`` slot would mean
        emitting a masked positive as a negative.
        """
        if m < 1:
            raise ValueError("m must be >= 1")
        valid = torch.isfinite(g).sum(dim=1)
        if int(valid.min()) < m:
            n_bad = int((valid < m).sum())
            raise ValueError(
                f"{n_bad} query/queries have < m={m} selectable (finite) slots "
                f"after positive masking; cannot pick clean negatives")
        mode = mode.lower()
        if mode == 'topk':
            slots = torch.topk(g, k=m, dim=1).indices
        elif mode == 'softmax':
            if L is not None and L < m:
                raise ValueError(
                    f"softmax selection requires L >= m, got L={L}, m={m} "
                    f"(top-L prefilter would leave fewer than m candidates)")
            slots = self._gumbel_topk(g, m, beta, L)
        else:
            raise ValueError(f"unknown selection_mode: {mode!r}")
        docids = [[self.docids[s] for s in row.tolist()] for row in slots]
        return slots, docids

    def _gumbel_topk(self, g, m, beta, L):
        B, N = g.shape
        logits = (beta * g).float()
        if L is not None and L < N:
            top = torch.topk(logits, k=L, dim=1)
            cand_logits, cand_idx = top.values, top.indices
        else:
            cand_logits = logits
            cand_idx = torch.arange(N, device=g.device).expand(B, N)
        cand_logits = cand_logits - cand_logits.max(dim=1, keepdim=True).values
        u = torch.rand(cand_logits.shape, generator=self._gen,
                       device=cand_logits.device, dtype=torch.float32
                       ).clamp_(min=1e-12, max=1.0)
        gumbel = -torch.log(-torch.log(u))
        perturbed = cand_logits + gumbel
        # masked entries (-inf) must never win
        perturbed = torch.where(torch.isinf(cand_logits), cand_logits, perturbed)
        sel = torch.topk(perturbed, k=m, dim=1).indices
        return torch.gather(cand_idx, 1, sel)

    def record_selection(self, slots):
        """Mark the given slots as selected at least once this interval."""
        self.selected_indicator[slots.reshape(-1)] = True

    def memory_bytes(self):
        """Byte footprint of ``Z_student`` (+ ``Z_teacher`` when present)."""
        b = self.Z_student.element_size() * self.Z_student.nelement()
        if self.Z_teacher is not None:
            b += self.Z_teacher.element_size() * self.Z_teacher.nelement()
        return b

    # ---- maintenance --------------------------------------------------------

    def maintain(self, student, teacher, tokenizer, corpus_lookup, all_docids,
                 recent_query_reservoir, step, cfg, device, qrels_dict=None):
        """Run one bounded maintenance cycle; returns a cost-counter dict.

        ``recent_query_reservoir`` is a dict ``{'q_student','q_teacher','qids'}``
        of recent query embeddings used to recertify replacement candidates
        (``None`` skips replacement, refresh-only). Restores ``student.training``.

        EMA / lazy-sequential MCDP only. On a cached-MCDP cache this would encode
        ONE deterministic embedding per slot into ``Z_student`` (the ``Z_mean``
        alias) while leaving ``Z_mc`` untouched, silently desynchronising the mean
        from the MC bank it is supposed to summarise — so it raises instead.
        """
        if self.is_cached_mcdp:
            raise RuntimeError(
                "maintain() is the EMA/sequential path and would desynchronise "
                "Z_mean from Z_mc on a cached-MCDP cache; call "
                "maintain_cached_mcdp() instead")
        was_training = student.training
        teacher_was_training = teacher.training if teacher is not None else False
        student.eval()
        if teacher is not None:
            teacher.eval()
        try:
            self._update_utility(cfg)
            budget = self._interval_budget(step, cfg)
            refresh_slots, replace_slots, diag = self._plan_actions(step, cfg, budget)

            counters = self._zero_counters()
            counters.update(diag)

            if len(refresh_slots):
                self._refresh(refresh_slots, student, teacher, tokenizer,
                              corpus_lookup, step, cfg, device)
                counters['num_refresh'] = len(refresh_slots)
                counters['doc_encoder_calls_cache_refresh'] = len(refresh_slots)

            if len(replace_slots) and recent_query_reservoir is not None:
                rc = self._replace(replace_slots, student, teacher, tokenizer,
                                   corpus_lookup, all_docids,
                                   recent_query_reservoir, step, cfg, device,
                                   qrels_dict or {})
                counters.update(rc)

            counters['cache_turnover_rate'] = (
                counters['num_replace'] / self.B_doc if self.B_doc else 0.0)
            return counters
        finally:
            if was_training:
                student.train()
            if teacher is not None and teacher_was_training:
                teacher.train()

    @staticmethod
    def _zero_counters():
        return dict(
            num_refresh=0, num_replace=0, num_over_age=0, over_age_backlog=0,
            num_R_candidates=0, num_uniform_candidates=0,
            num_recertified_candidates=0,
            doc_encoder_calls_cache_refresh=0, doc_encoder_calls_cache_replace=0,
            doc_encoder_calls_recertify=0, cache_turnover_rate=0.0)

    def _update_utility(self, cfg):
        decay = cfg['utility_ema_decay']
        ind_f = self.selected_indicator.float()
        self.utility_ema = decay * self.utility_ema + (1.0 - decay) * ind_f
        self.peak_utility_ema = torch.maximum(self.peak_utility_ema,
                                              self.utility_ema)
        self.lifetime_selected_count += self.selected_indicator.long()
        self.selected_count_recent = self.selected_indicator.long()
        self.intervals_since_selected = torch.where(
            self.selected_indicator,
            torch.zeros_like(self.intervals_since_selected),
            self.intervals_since_selected + 1)
        self.selected_indicator = torch.zeros_like(self.selected_indicator)

    def _interval_budget(self, step, cfg):
        total = cfg.get('total_steps', 0)
        progress = (step / total) if total else 0.0
        rho = linear_decay(cfg['rho_start'], cfg['rho_end'], progress)
        budget = round(rho * self.B_doc * cfg['cache_update_interval']
                       / cfg['steps_per_epoch'])
        return max(int(budget), 0)

    def _plan_actions(self, step, cfg, budget):
        """Order maintenance actions per spec and cut at the shared budget.

        1. urgent over-age: refresh useful, replace low-utility
        2. persistently low-utility replacements
        3. remaining useful stale refreshes by refresh_priority = utility*age_norm
        4. defer anything beyond budget (tracked as over_age_backlog)
        """
        max_age = max(int(cfg['max_age_steps']), 1)
        # Grace: a doc is too young to be judged "persistently low-utility" until
        # it has been resident for K maintenance intervals. Without this, freshly
        # inserted docs (utility_ema=0) — and the entire cache at init — are
        # instantly flagged low-utility and churned before proving useful.
        grace_steps = cfg['K'] * cfg['cache_update_interval']
        age_t = step - self.last_refreshed_step
        age = age_t.float().cpu().numpy()
        age_norm = np.minimum(age / max_age, 1.0)
        util = self.utility_ema.cpu().numpy()
        over = age >= max_age
        low = ((self.intervals_since_selected >= cfg['K']) |
               ((self.utility_ema <= cfg['utility_floor']) &
                (age_t >= grace_steps))).cpu().numpy()

        refresh, replace, used = [], [], 0

        def take(slot, bucket):
            nonlocal used
            if used >= budget:
                return False
            bucket.append(int(slot))
            used += 1
            return True

        oa = np.where(over)[0]
        oa_useful = sorted([s for s in oa if not low[s]],
                           key=lambda s: util[s] * age_norm[s], reverse=True)
        oa_low = [s for s in oa if low[s]]
        for s in oa_useful:
            if not take(s, refresh):
                break
        for s in oa_low:
            if not take(s, replace):
                break

        low_only = sorted(np.where(low & ~over)[0], key=lambda s: util[s])
        for s in low_only:
            if not take(s, replace):
                break

        remaining = sorted(
            [s for s in range(self.B_doc) if not over[s] and not low[s]],
            key=lambda s: util[s] * age_norm[s], reverse=True)
        for s in remaining:
            if util[s] * age_norm[s] <= 0:
                break
            if not take(s, refresh):
                break

        num_over = int(over.sum())
        handled_over = sum(1 for s in (refresh + replace) if over[s])
        diag = dict(num_over_age=num_over,
                    over_age_backlog=max(num_over - handled_over, 0))
        to_t = lambda xs: torch.tensor(xs, dtype=torch.long, device=self.device)
        return to_t(refresh), to_t(replace), diag

    def _encode_docs(self, texts, student, teacher, tokenizer, cfg, device):
        max_len = cfg.get('passage_max_len', 512)
        bs = cfg.get('mc_batch_size', 256)
        zs = encode_batch_tensor(student, tokenizer, texts, device, max_len, bs,
                                 requires_grad=False).to(self.Z_student.dtype)
        if self.Z_teacher is not None and teacher is not None:
            zt = encode_batch_tensor(teacher, tokenizer, texts, device, max_len,
                                     bs, requires_grad=False).to(self.Z_teacher.dtype)
            zt = zt.detach()
        else:
            zt = None   # teacher-free (MCDP): student states only
        return zs.detach(), zt

    def _refresh(self, slots, student, teacher, tokenizer, corpus_lookup, step,
                 cfg, device):
        slot_list = slots.tolist()
        texts = [corpus_lookup[self.docids[s]] for s in slot_list]
        zs, zt = self._encode_docs(texts, student, teacher, tokenizer, cfg, device)
        self.Z_student[slots] = zs
        if self.Z_teacher is not None and zt is not None:
            self.Z_teacher[slots] = zt
        self.last_refreshed_step[slots] = step

    def _replace(self, slots, student, teacher, tokenizer, corpus_lookup,
                 all_docids, reservoir, step, cfg, device, qrels_dict):
        num_replace = len(slots)
        num_cand = cfg['replacement_candidate_multiplier'] * num_replace
        # Candidate generation is governed by uniform_candidate_fraction (the
        # BINDING constraint): the uniform pool is sized from it and R fills only
        # the remainder. R_fraction is therefore SECONDARY and effectively capped
        # by the uniform requirement, so config drift in R_fraction can never
        # break uniform dominance.
        num_uniform_target = int(np.ceil(cfg['uniform_candidate_fraction'] * num_cand))
        num_R_target = max(num_cand - num_uniform_target, 0)

        in_H = set(self.docids)
        R_cands = [d for d in self.registry.nominate(num_R_target, self.rng)
                   if d in corpus_lookup and d not in in_H]
        uni = self._sample_uniform(all_docids, num_cand - len(R_cands),
                                   exclude=in_H | set(R_cands), corpus=corpus_lookup)
        cand = list(dict.fromkeys(R_cands + uni))
        if not cand:
            return self._replace_counters(0, len(R_cands), len(uni), 0)

        cand_texts = [corpus_lookup[d] for d in cand]
        czs, czt = self._encode_docs(cand_texts, student, teacher, tokenizer,
                                     cfg, device)

        # recertify: score candidates against the recent-query reservoir.
        # Grad-free like score(): recertification must never build a graph, even
        # if the reservoir tensors happen to require grad. Teacher-free (MCDP):
        # deterministic student-only recert score (g = s_stu, no σ term).
        qs = reservoir['q_student']
        qt = reservoir.get('q_teacher')
        with torch.no_grad():
            s_stu = qs @ czs.t()
            if self.Z_teacher is not None and czt is not None and qt is not None:
                s_tea = qt @ czt.t()
                g = s_stu + cfg['lambda_val'] * (s_stu - s_tea).abs()   # (R, C)
            else:
                g = s_stu   # student-only recert (MCDP)
        for r, qid in enumerate(reservoir['qids']):
            pos = qrels_dict.get(qid)
            if not pos:
                continue
            cols = [j for j, d in enumerate(cand) if d in pos]
            if cols:
                g[r, cols] = float('-inf')

        k = min(int(cfg['reentry_top_k']), g.shape[0])
        reentry = torch.topk(g, k=k, dim=0).values.mean(dim=0)   # (C,)
        # Only candidates with a finite re-entry score are insertable — a fully
        # masked candidate (a known positive of every reservoir query) must never
        # be promoted into H.
        finite = torch.isfinite(reentry)
        order = [i for i in torch.argsort(reentry, descending=True).tolist()
                 if finite[i]]
        chosen = order[:num_replace]

        for slot, ci in zip(slots.tolist(), chosen):
            evicted = self.docids[slot]
            self.registry.admit(evicted, dict(
                peak_utility_ema=float(self.peak_utility_ema[slot]),
                lifetime_selected_count=int(self.lifetime_selected_count[slot])),
                step)
            self._insert(slot, cand[ci], czs[ci],
                         czt[ci] if czt is not None else None, step)
            # a recertified-and-reinserted doc is active again -> leaves R
            self.registry.entries.pop(cand[ci], None)

        return self._replace_counters(len(chosen), len(R_cands), len(uni),
                                      len(cand))

    @staticmethod
    def _replace_counters(num_replace, num_R, num_uniform, num_recert):
        return dict(
            num_replace=num_replace,
            num_R_candidates=num_R,
            num_uniform_candidates=num_uniform,
            num_recertified_candidates=num_recert,
            # ALL replacement-side doc encodes happen during recertification;
            # inserting a chosen candidate reuses its already-computed embedding,
            # so it adds no extra encoder calls (avoids double-counting).
            doc_encoder_calls_recertify=num_recert,
            doc_encoder_calls_cache_replace=0)

    def _sample_uniform(self, all_docids, n, exclude, corpus):
        if n <= 0:
            return []
        out, seen, N = [], set(exclude), len(all_docids)
        budget = 0
        while len(out) < n and budget < 20 * n + 100:
            budget += 1
            d = all_docids[int(self.rng.integers(0, N))]
            if d in seen or d not in corpus:
                continue
            seen.add(d)
            out.append(d)
        return out

    # ---- cached-MCDP maintenance -------------------------------------------

    @staticmethod
    def _zero_mc_counters():
        return dict(
            num_refresh=0, num_replace=0, num_over_age=0, over_age_backlog=0,
            num_recertified_candidates=0,
            # R_doc is DEFERRED for Phase 1: candidates are uniform-only, nothing is
            # admitted to the registry on eviction, so these stay zero by construction.
            num_R_candidates=0, num_uniform_candidates=0,
            maintenance_docs_encoded=0, maintenance_mc_passes=0,
            maintenance_examples_encoded=0, maintenance_forward_batches=0,
            cache_turnover_rate=0.0)

    def maintain_cached_mcdp(self, student, tokenizer, corpus_lookup, all_docids,
                             query_mc_reservoir, step, cfg, device, qrels_dict=None):
        """One bounded cached-MCDP maintenance interval.

        ``step`` is **model time** = ``source_checkpoint_step``, identical for every
        interval in a mining round: the miner holds checkpoint-frozen weights, so
        cache age, the rho/progress budget, and ``last_refreshed_step`` must not move
        within a round. Passing a miner-local counter (batch index, mined-query
        count, interval index) would corrupt all three.

        Differences from the EMA ``maintain`` (which stays untouched):
          * refresh encodes each document ``T`` times, not once;
          * **every** replacement candidate is encoded ``T`` times exactly once, and
            recertification scores those states against the query-MC reservoir with
            the same mean+uncertainty ``g`` as normal mining;
          * chosen candidates are inserted by reusing the already-computed states, so
            insertion adds no encoder calls.

        **`R_doc` is DEFERRED for Phase 1**: candidates come uniformly from the
        corpus excluding documents already in ``H``; there is no registry
        nomination and no admission on eviction. The registry counters are reported
        as zero. ``RetiredRegistry`` and the EMA path are unaffected.

        ``query_mc_reservoir`` is ``(q_mc[T, R_query, D], qids)``. When it is empty,
        maintenance refreshes existing slots but skips replacement recertification.
        """
        if not self.is_cached_mcdp:
            raise RuntimeError("maintain_cached_mcdp() requires a cached-MCDP cache")
        qrels_dict = qrels_dict or {}
        T = int(self.T)
        entry_mode = student.training
        counters = self._zero_mc_counters()
        try:
            # 1. Fold utility BEFORE planning — same order as maintain(). Without
            #    this, selected_indicator / utility_ema / intervals_since_selected
            #    never advance and the eligibility rules see frozen state.
            self._update_utility(cfg)

            budget = self._interval_budget(step, cfg)
            refresh_slots, replace_slots, diag = self._plan_actions(step, cfg, budget)
            counters.update(diag)
            counters['maintenance_budget_interval'] = int(budget)

            # 2. Refresh: T stochastic passes per doc; Z_mc, Z_mean and the
            #    timestamp move together as one commit.
            if len(refresh_slots):
                slot_list = refresh_slots.tolist()
                texts = [corpus_lookup[self.docids[s]] for s in slot_list]
                Zr, enc = encode_mc(student, tokenizer, texts, T, device,
                                    cfg.get('passage_max_len', 512),
                                    self._miner_mc_batch_size(cfg),
                                    dtype=self.Z_mc.dtype)
                self.Z_mc[:, refresh_slots, :] = Zr
                # in place, so the Z_student -> Z_mean alias stays valid
                self.Z_student[refresh_slots] = Zr.float().mean(dim=0).to(
                    self.Z_student.dtype)
                self.last_refreshed_step[refresh_slots] = step
                counters['num_refresh'] = len(slot_list)
                counters['maintenance_docs_encoded'] += len(texts)
                counters['maintenance_mc_passes'] = enc['mc_passes']
                counters['maintenance_examples_encoded'] += enc['examples_encoded']
                counters['maintenance_forward_batches'] += enc['forward_batches']

            # 3. Replace (needs the query-MC reservoir for recertification).
            q_res, res_qids = (query_mc_reservoir if query_mc_reservoir is not None
                               else (None, []))
            if len(replace_slots) and q_res is not None and len(res_qids):
                rc = self._replace_cached_mcdp(
                    replace_slots, student, tokenizer, corpus_lookup, all_docids,
                    q_res, res_qids, step, T, cfg, device, qrels_dict)
                for k, v in rc.items():
                    if k in ('maintenance_docs_encoded',
                             'maintenance_examples_encoded',
                             'maintenance_forward_batches'):
                        counters[k] += v
                    else:
                        counters[k] = v

            counters['cache_turnover_rate'] = (
                counters['num_replace'] / self.B_doc if self.B_doc else 0.0)
            counters['maintenance_model_step'] = int(step)
            return counters
        finally:
            student.train(entry_mode)

    def _replace_cached_mcdp(self, slots, student, tokenizer, corpus_lookup,
                             all_docids, q_res, res_qids, step, T, cfg, device,
                             qrels_dict):
        """Uniform-candidate replacement with cached-MCDP recertification.

        `R_doc` deferred: the whole candidate pool is uniform corpus sampling that
        excludes documents already in ``H``. Every candidate is encoded for ``T``
        passes ONCE and those same states are reused on insertion, so
        ``maintenance_docs_encoded`` counts each candidate exactly once.
        """
        num_replace = len(slots)
        num_cand = cfg['replacement_candidate_multiplier'] * num_replace
        cand = self._sample_uniform(all_docids, num_cand, exclude=set(self.docids),
                                    corpus=corpus_lookup)
        out = dict(num_replace=0, num_R_candidates=0,
                   num_uniform_candidates=len(cand),
                   num_recertified_candidates=0,
                   maintenance_docs_encoded=0, maintenance_mc_passes=int(T),
                   maintenance_examples_encoded=0, maintenance_forward_batches=0)
        if not cand:
            return out

        cand_texts = [corpus_lookup[d] for d in cand]
        Zc, enc = encode_mc(student, tokenizer, cand_texts, T, device,
                            cfg.get('passage_max_len', 512),
                            self._miner_mc_batch_size(cfg), dtype=self.Z_mc.dtype)
        out['num_recertified_candidates'] = len(cand)
        out['maintenance_docs_encoded'] = len(cand)
        out['maintenance_examples_encoded'] = enc['examples_encoded']
        out['maintenance_forward_batches'] = enc['forward_batches']

        # recertify with the SAME mean+uncertainty score as ordinary mining.
        # Uses the free function directly so recertification does NOT inflate
        # cache_score_pairs — it is a maintenance cost, not a mining cost.
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
        # a fully masked candidate (a known positive of every reservoir query) must
        # never be promoted into H
        finite = torch.isfinite(reentry)
        order = [i for i in torch.argsort(reentry, descending=True).tolist()
                 if finite[i]]
        chosen = order[:num_replace]

        for slot, ci in zip(slots.tolist(), chosen):
            # R_doc deferred: no registry.admit(evicted, ...) here.
            self.Z_mc[:, slot, :] = Zc[:, ci, :]     # reuse — no second encode
            # _insert writes Z_student[slot] (== Z_mean[slot]) in place and resets
            # the new slot's utility/history with timestamp = step.
            self._insert(slot, cand[ci],
                         Zc[:, ci, :].float().mean(dim=0).to(self.Z_student.dtype),
                         None, step)

        out['num_replace'] = len(chosen)
        return out

    def _insert(self, slot, docid, zs, zt, step):
        del self.docid_to_slot[self.docids[slot]]
        self.docids[slot] = docid
        self.docid_to_slot[docid] = slot
        self.Z_student[slot] = zs
        if self.Z_teacher is not None and zt is not None:
            self.Z_teacher[slot] = zt
        self.last_refreshed_step[slot] = step
        self.utility_ema[slot] = 0.0
        self.peak_utility_ema[slot] = 0.0
        self.selected_indicator[slot] = False
        self.selected_count_recent[slot] = 0
        self.lifetime_selected_count[slot] = 0
        self.intervals_since_selected[slot] = 0

    # ---- cached-MCDP persistence -------------------------------------------

    CACHED_MCDP_SCHEMA = "cached_mcdp_v1"

    def save_state(self, path):
        """Persist an explicit versioned state dict, not the live Python object.

        Holds enough for an EXACT miner restart: CPU tensors plus both RNG states,
        so a reload reproduces the next cache-random decision, not merely the
        current embeddings.

        `R_doc` is deferred, so the registry is intentionally NOT serialized;
        adding it later requires a schema bump or an optional-key read.
        """
        if not self.is_cached_mcdp:
            raise RuntimeError("save_state() is cached-MCDP only")
        state = {
            'schema_version': self.CACHED_MCDP_SCHEMA,
            'T': int(self.T),
            'B_doc': int(self.B_doc),
            'dim': int(self.dim),
            'dtype': str(self.Z_mc.dtype),
            'docids': list(self.docids),
            'Z_mc': self.Z_mc.detach().to('cpu'),
            # Z_mean is the Z_student alias; stored once and re-aliased on load.
            'Z_mean': self.Z_student.detach().to('cpu'),
            'utility_ema': self.utility_ema.detach().to('cpu'),
            'peak_utility_ema': self.peak_utility_ema.detach().to('cpu'),
            'selected_indicator': self.selected_indicator.detach().to('cpu'),
            'selected_count_recent': self.selected_count_recent.detach().to('cpu'),
            'lifetime_selected_count': self.lifetime_selected_count.detach().to('cpu'),
            'intervals_since_selected': self.intervals_since_selected.detach().to('cpu'),
            'last_refreshed_step': self.last_refreshed_step.detach().to('cpu'),
            'cache_score_pairs': int(self.cache_score_pairs),
            'numpy_bit_generator_state': self.rng.bit_generator.state,
            'torch_generator_state': self._gen.get_state().clone(),
            'registry_deferred': True,   # R_doc not implemented in Phase 1
        }
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(state, path)
        return path

    # metadata tensors that must be present, 1-D, and B_doc-long
    _STATE_META_TENSORS = (
        ('utility_ema', torch.float32),
        ('peak_utility_ema', torch.float32),
        ('selected_indicator', torch.bool),
        ('selected_count_recent', torch.long),
        ('lifetime_selected_count', torch.long),
        ('intervals_since_selected', torch.long),
        ('last_refreshed_step', torch.long),
    )

    @staticmethod
    def effective_B_doc(cfg, n_corpus):
        """Cache size actually realised: ``init_uniform`` clamps to the corpus.

        A configured ``B_doc`` larger than the corpus yields a smaller cache, so
        restart validation must compare against this, not the raw config value.
        """
        return min(int(cfg['B_doc']), int(n_corpus))

    @classmethod
    def load_state(cls, path, cfg, device, *, expect_T=None, expect_B_doc=None,
                   expect_dim=None, expect_schema=None, mean_atol=1e-2):
        """Load a persisted cached-MCDP cache, validating BEFORE any device move.

        ``torch.load`` runs with ``map_location='cpu'`` and EVERY check below runs on
        those CPU tensors, so an incompatible state never allocates device memory.

        **``Z_mc`` is authoritative.** ``Z_mean`` is derived, so it is recomputed
        from ``Z_mc`` on load and the persisted copy is only used to detect
        corruption: if it disagrees beyond ``mean_atol`` the state is REJECTED rather
        than silently loaded with a mean that contradicts the MC bank. Restoring a
        disagreeing mean would leave ``cheap_scores`` / ``_plan_actions`` (which read
        ``Z_student``, the ``Z_mean`` alias) scoring against states that no longer
        match ``Z_mc``.

        ``expect_B_doc`` should be the EFFECTIVE size (see ``effective_B_doc``), not
        the raw configured one, or a corpus smaller than ``B_doc`` makes every
        restart fail.
        """
        state = torch.load(Path(path), map_location='cpu', weights_only=False)
        if not isinstance(state, dict):
            raise ValueError(f"cache state at {path} is not a state dict")

        want_schema = expect_schema or cls.CACHED_MCDP_SCHEMA
        got = state.get('schema_version')
        if got != want_schema:
            raise ValueError(f"cache state schema mismatch: file has {got!r}, "
                             f"expected {want_schema!r}")

        for key in ('T', 'B_doc', 'dim', 'docids', 'Z_mc', 'Z_mean',
                    'numpy_bit_generator_state', 'torch_generator_state'):
            if key not in state:
                raise ValueError(f"cache state is missing required key {key!r}")

        T, B_doc, dim = int(state['T']), int(state['B_doc']), int(state['dim'])
        for name, expected, value in (('T', expect_T, T),
                                      ('B_doc', expect_B_doc, B_doc),
                                      ('dim', expect_dim, dim)):
            if expected is not None and value != int(expected):
                raise ValueError(f"cache state {name} mismatch: file has {value}, "
                                 f"expected {expected}")

        Z_mc, Z_mean = state['Z_mc'], state['Z_mean']
        if tuple(Z_mc.shape) != (T, B_doc, dim):
            raise ValueError(f"cache state Z_mc shape {tuple(Z_mc.shape)} does not "
                             f"match T={T}, B_doc={B_doc}, dim={dim}")
        if tuple(Z_mean.shape) != (B_doc, dim):
            raise ValueError(f"cache state Z_mean shape {tuple(Z_mean.shape)} does "
                             f"not match B_doc={B_doc}, dim={dim}")
        if Z_mean.dtype != Z_mc.dtype:
            raise ValueError(f"cache state dtype mismatch: Z_mc {Z_mc.dtype} vs "
                             f"Z_mean {Z_mean.dtype}")
        if 'dtype' in state and str(Z_mc.dtype) != str(state['dtype']):
            raise ValueError(f"cache state dtype header {state['dtype']!r} does not "
                             f"match Z_mc dtype {Z_mc.dtype}")

        docids = list(state['docids'])
        if len(docids) != B_doc:
            raise ValueError(f"cache state has {len(docids)} docids, expected {B_doc}")
        if len(set(docids)) != B_doc:
            raise ValueError(
                f"cache state docids are not unique ({B_doc - len(set(docids))} "
                f"duplicates) — the slot<->docid bijection is broken")

        for name, dtype in cls._STATE_META_TENSORS:
            if name not in state:
                raise ValueError(f"cache state is missing metadata tensor {name!r}")
            t = state[name]
            if not torch.is_tensor(t) or t.shape != (B_doc,):
                raise ValueError(f"cache state {name} must be a [{B_doc}] tensor, "
                                 f"got {tuple(getattr(t, 'shape', ()))}")
            if t.dtype != dtype:
                raise ValueError(f"cache state {name} dtype {t.dtype}, expected {dtype}")

        # Z_mc is authoritative: derive the mean and reject a disagreeing copy.
        recomputed = Z_mc.float().mean(dim=0).to(Z_mc.dtype)
        if not torch.allclose(recomputed.float(), Z_mean.float(), atol=mean_atol):
            drift = float((recomputed.float() - Z_mean.float()).abs().max())
            raise ValueError(
                f"cache state Z_mean disagrees with mean_t(Z_mc) (max |diff| "
                f"{drift:.3e} > atol {mean_atol}); Z_mc is authoritative, so this "
                f"state is corrupt rather than merely stale")

        # --- everything validated; only now touch the device ---
        rng = np.random.default_rng()
        rng.bit_generator.state = state['numpy_bit_generator_state']
        cache = cls(docids, recomputed.to(device), None, cfg, device, dim, rng)
        cache.attach_mc_states(Z_mc.to(device))
        for name, _dtype in cls._STATE_META_TENSORS:
            setattr(cache, name, state[name].to(device))
        cache.cache_score_pairs = int(state.get('cache_score_pairs', 0))
        cache._gen.set_state(state['torch_generator_state'])
        return cache
