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
import numpy as np
import torch

from utils.helpers import encode_batch_tensor


def linear_decay(start, end, progress):
    """``start`` at progress<=0, ``end`` at progress>=1, linear in between."""
    p = min(max(float(progress), 0.0), 1.0)
    return start + (end - start) * p


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
    (``Z_student``) and an EMA-teacher doc embedding (``Z_teacher``), plus
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
        self.Z_teacher = Z_teacher.detach()
        self.Z_student.requires_grad_(False)
        self.Z_teacher.requires_grad_(False)

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

        self.registry = RetiredRegistry(
            max_size=cfg['R_size_factor'] * self.B_doc,
            utility_remember_threshold=cfg['utility_remember_threshold'])

        # Seeded torch RNG so Gumbel-Softmax selection is reproducible (does not
        # touch the global torch RNG state).
        self._gen = torch.Generator(device=self.device)
        self._gen.manual_seed(int(cfg['cache_init_seed']))

        # cumulative scoring cost (maintenance cost is returned per-call instead)
        self.cache_score_pairs = 0

    @classmethod
    def init_uniform(cls, stale_embs, c_ids, cfg, device, dim=None, dtype=None):
        """Initialize ``H`` by uniformly sampling ``B_doc`` corpus docs (seeded).

        ``stale_embs[i]`` must be the embedding of ``c_ids[i]`` (the ordered
        ``c_ids <-> rows`` mapping from the stale-index pickle). At init the
        student and teacher states are identical (the EMA teacher == student).
        ``dtype`` defaults to bf16 on CUDA (halves the Z_H footprint) and float32
        on CPU.
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
        return cls(docids, embs.clone(), embs.clone(), cfg, device, dim, rng)

    # ---- mining: score / mask / select -------------------------------------

    def score(self, q_student, q_teacher, lambda_val):
        """Score a query batch against all of ``H``.

        Returns ``(g, s_student, sigma)``, each ``(batch, B_doc)``. ``s_student``
        and ``sigma`` are returned for logging. Selection is grad-free: scoring
        runs under ``no_grad`` so a grad-enabled query batch never builds a graph
        through the selection-only cache.
        """
        with torch.no_grad():
            s_student = q_student @ self.Z_student.t()
            s_teacher = q_teacher @ self.Z_teacher.t()
            sigma = (s_student - s_teacher).abs()
            g = s_student + lambda_val * sigma
        self.cache_score_pairs += int(q_student.shape[0]) * self.B_doc
        return g, s_student, sigma

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
        """Byte footprint of ``Z_student`` + ``Z_teacher`` (T1 budget report)."""
        return (self.Z_student.element_size() * self.Z_student.nelement() +
                self.Z_teacher.element_size() * self.Z_teacher.nelement())

    # ---- maintenance --------------------------------------------------------

    def maintain(self, student, teacher, tokenizer, corpus_lookup, all_docids,
                 recent_query_reservoir, step, cfg, device, qrels_dict=None):
        """Run one bounded maintenance cycle; returns a cost-counter dict.

        ``recent_query_reservoir`` is a dict ``{'q_student','q_teacher','qids'}``
        of recent query embeddings used to recertify replacement candidates
        (``None`` skips replacement, refresh-only). Restores ``student.training``.
        """
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
        if teacher is not None:
            zt = encode_batch_tensor(teacher, tokenizer, texts, device, max_len,
                                     bs, requires_grad=False).to(self.Z_teacher.dtype)
        else:
            zt = zs.clone()
        return zs.detach(), zt.detach()

    def _refresh(self, slots, student, teacher, tokenizer, corpus_lookup, step,
                 cfg, device):
        slot_list = slots.tolist()
        texts = [corpus_lookup[self.docids[s]] for s in slot_list]
        zs, zt = self._encode_docs(texts, student, teacher, tokenizer, cfg, device)
        self.Z_student[slots] = zs
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
        # if the reservoir tensors happen to require grad.
        qs, qt = reservoir['q_student'], reservoir['q_teacher']
        with torch.no_grad():
            s_stu = qs @ czs.t()
            s_tea = qt @ czt.t()
            g = s_stu + cfg['lambda_val'] * (s_stu - s_tea).abs()   # (R, C)
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
            self._insert(slot, cand[ci], czs[ci], czt[ci], step)
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

    def _insert(self, slot, docid, zs, zt, step):
        del self.docid_to_slot[self.docids[slot]]
        self.docids[slot] = docid
        self.docid_to_slot[docid] = slot
        self.Z_student[slot] = zs
        self.Z_teacher[slot] = zt
        self.last_refreshed_step[slot] = step
        self.utility_ema[slot] = 0.0
        self.peak_utility_ema[slot] = 0.0
        self.selected_indicator[slot] = False
        self.selected_count_recent[slot] = 0
        self.lifetime_selected_count[slot] = 0
        self.intervals_since_selected[slot] = 0
