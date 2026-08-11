"""
Async Fast-GRASS — cached-MCDP miner utilities.

Phase 0 prototyped the cached-MCDP algorithm here; **Phase 1 ported it into
``src/utils/negative_cache.py``** (`init_cached_mcdp`, `score_cached_mcdp`,
`maintain_cached_mcdp`, `save_state`/`load_state`) and the encoding primitives into
``src/utils/helpers.py`` (`dropout_only`, `encode_mc`). ``src/`` must never import
``scripts/``, so the low-level pieces had to live under ``src/``.

What remains here is genuinely miner-side and has no place on the cache object:
the recent-query MC reservoir, the mined-query maintenance cadence, and the
per-batch mining step. `dropout_only` / `encode_mc` / `score_cached_mcdp` are
**re-exported** so the Phase-0 timing harness and its tests keep importing from one
place, unchanged.

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
import math
import sys
from pathlib import Path

import torch

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root / 'src'))

from utils.helpers import dropout_only, encode_mc  # noqa: E402,F401
from utils.negative_cache import score_cached_mcdp  # noqa: E402,F401


# ---- re-exports (implementation lives under src/) ---------------------------
# src/ must never import scripts/, so the encoding primitives live in helpers and
# the scoring math lives in negative_cache. Re-exported here so the Phase-0 timing
# harness and its tests keep importing cached-MCDP pieces from one place.

__all__ = [
    'dropout_only', 'encode_mc', 'score_cached_mcdp',
    'encode_queries_mc', 'encode_docs_mc', 'init_Z_mc',
    'QueryMCReservoir', 'MaintenanceDriver', 'maintenance_interval_mined_queries',
    'maintain_interval_cached_mcdp', 'mine_batch_cached_mcdp',
    'validate_refresh_schedule', 'format_refresh_report', 'MiningDiagnostics',
]


# ---- runtime config ---------------------------------------------------------

def steps_per_epoch(n_items, batch_size):
    """Canonical steps-per-epoch, shared by orchestrator, trainer and miner.

    Must be identical in all three: the miner divides by it in the maintenance
    budget (``rho * B_doc * cache_update_interval / steps_per_epoch``), so a
    floor-vs-ceil disagreement silently changes how much of the cache is
    maintained per interval.
    """
    return max(math.ceil(int(n_items) / max(int(batch_size), 1)), 1)


class UnresolvablePositivesError(RuntimeError):
    """Some training positives cannot be resolved against the corpus."""


def canonicalize_positives(train_items, qrels_dict, corpus_lookup, log=print):
    """Map each item's ``pos_docid`` onto the docid that actually exists in the corpus.

    ``preprocessor.run_setup`` MD5-dedupes passages by text: duplicate-text docids
    collapse to one canonical entry, and only the CORPUS and the QRELS are remapped.
    The training mixture still carries the original ``positive_passages[0]['docid']``,
    so a positive whose text was a duplicate has a docid that is absent from
    ``reasonir_corpus.jsonl``.

    Qrels are already canonical and keyed by query id, so they are the remap: for an
    unresolvable positive, take a qrels entry for that query that IS in the corpus.

    **Any item that still cannot be resolved is a hard error, not a drop.** Silently
    discarding training examples shrinks the experiment by an unrecorded amount and
    makes the run incomparable to a baseline trained on the full mixture — a much
    worse failure than stopping, because nothing downstream would reveal it. The fix
    is to regenerate the corpus/qrels, not to train on less data.

    Returns ``(items, stats)``.
    """
    out, remapped, unresolved = [], 0, []
    for it in train_items:
        pos = it['pos_docid']
        if pos in corpus_lookup:
            out.append(it)
            continue
        canonical = next((d for d in qrels_dict.get(it['query_id'], ())
                          if d in corpus_lookup), None)
        if canonical is None:
            unresolved.append(it)
            continue
        out.append({**it, 'pos_docid': canonical})
        remapped += 1

    stats = {'total': len(train_items), 'kept': len(out),
             'remapped': remapped, 'dropped': len(unresolved)}
    if remapped:
        log(f"canonicalized positives: {remapped:,}/{len(train_items):,} remapped "
            f"to their canonical docid")
    if unresolved:
        sample = [(it['query_id'], it['pos_docid']) for it in unresolved[:5]]
        raise UnresolvablePositivesError(
            f"{len(unresolved):,}/{len(train_items):,} training items have a "
            f"positive that is absent from the corpus and has no corpus-resident "
            f"qrels entry (e.g. {sample}). Training on the remainder would silently "
            f"shrink the mixture and break comparability with the baselines — "
            f"regenerate reasonir_corpus.jsonl / train_qrels.txt (re-run "
            f"preprocessing) instead.")
    return out, stats


def build_async_cfg(config, ctx, steps_per_epoch):
    """Runtime cfg for the async miner/orchestrator from ``training.async_fast_grass``.

    Separate from ``run_fast_grass._build_fast_grass_cfg`` because that one reads
    the *sequential* block and applies CLI overrides the async processes do not
    have. Derived fields match it so the shared ``NegativeCache`` maintenance code
    behaves identically.

    ``L`` is absent by construction: cached-MCDP scores all of ``H``.

    **``max_age_steps`` precedence.** An explicit non-null ``max_age_steps`` in the
    recipe WINS; ``max_age_epochs`` is only the fallback. The derived form
    (``max_age_epochs * steps_per_epoch``) silently produced ``2 * 5157 = 10314``,
    exactly ``total_steps``, and ``NegativeCache._plan_actions`` tests
    ``age >= max_age_steps`` — so over-age refresh became eligible only on the final
    round and no refreshed round ever reached the trainer. Recipes that omit
    ``max_age_steps`` keep the old value bit-for-bit.
    """
    cfg = dict(ctx['args'])
    cfg.pop('L', None)
    cfg['lambda_val'] = float(cfg['lambda_val'])
    cfg['query_max_len'] = config['model']['query_max_len']
    cfg['passage_max_len'] = config['model']['passage_max_len']
    cfg['steps_per_epoch'] = int(steps_per_epoch)
    cfg['total_steps'] = cfg['steps_per_epoch'] * cfg['num_epochs']

    explicit = cfg.get('max_age_steps')
    if explicit is None:
        if cfg.get('max_age_epochs') is None:
            raise ValueError(
                "the recipe defines neither max_age_steps nor max_age_epochs; cache "
                "maintenance cannot decide when a document state is over-age")
        cfg['max_age_steps'] = int(cfg['max_age_epochs']) * cfg['steps_per_epoch']
        cfg['max_age_source'] = 'max_age_epochs'
    else:
        cfg['max_age_steps'] = int(explicit)
        cfg['max_age_source'] = 'max_age_steps'
    if cfg['max_age_steps'] < 1:
        raise ValueError(f"max_age_steps must be >= 1, got {cfg['max_age_steps']}")
    return cfg


def validate_refresh_schedule(cfg):
    """Can over-age refresh actually influence this run? -> (errors, warnings, info).

    Three things have to line up, and none of them are checked anywhere else:

    * ``max_age_steps < total_steps`` — otherwise no refreshed numeric round can ever
      reach the trainer (the pre-fix default sat exactly ON this boundary);
    * ``max_age_steps <= async_mine_every_steps`` — the miner freezes model time at
      ``source_checkpoint_step``, and slots start at ``last_refreshed_step = 0``, so
      the first round whose checkpoint step reaches ``max_age_steps`` is the first
      one that can refresh anything;
    * the interval budget must be non-zero, or maintenance plans actions it cannot pay
      for.

    Errors are fatal (the orchestrator refuses to start); warnings are printed.
    """
    errors, warnings = [], []
    max_age = int(cfg['max_age_steps'])
    total = int(cfg.get('total_steps', 0))
    mine_every = int(cfg.get('async_mine_every_steps', 0) or 0)

    if total and max_age >= total:
        errors.append(
            f"max_age_steps={max_age} >= total_steps={total}: no refreshed numeric "
            f"round can influence training, because a slot only becomes over-age at "
            f"or after the last checkpoint. Set max_age_steps to about the checkpoint "
            f"cadence (async_mine_every_steps={mine_every or 'n/a'}).")

    first_refresh = None
    if mine_every > 0:
        first_refresh = int(math.ceil(max_age / mine_every)) * mine_every
        if max_age > mine_every:
            warnings.append(
                f"max_age_steps={max_age} > async_mine_every_steps={mine_every}: the "
                f"first numeric round that can refresh is the one mined from "
                f"checkpoint-{first_refresh}, not checkpoint-{mine_every}.")
        if total and first_refresh > total:
            errors.append(
                f"the first refresh-eligible checkpoint would be step {first_refresh}, "
                f"beyond total_steps={total}: refresh can never fire.")

    budget = None
    if cfg.get('B_doc') and cfg.get('steps_per_epoch'):
        budget = round(cfg['rho_start'] * int(cfg['B_doc'])
                       * int(cfg['cache_update_interval'])
                       / int(cfg['steps_per_epoch']))
        if budget < 1:
            errors.append(
                f"the initial maintenance budget rounds to {budget} slots per interval "
                f"(rho_start={cfg['rho_start']} * B_doc={cfg['B_doc']} * "
                f"cache_update_interval={cfg['cache_update_interval']} / "
                f"steps_per_epoch={cfg['steps_per_epoch']}): maintenance would be a "
                f"no-op and nothing could ever be refreshed.")

    info = {
        'max_age_steps': max_age,
        'max_age_source': cfg.get('max_age_source'),
        'total_steps': total,
        'async_mine_every_steps': mine_every,
        'first_refresh_checkpoint_step': first_refresh,
        'initial_interval_budget': budget,
        'maintenance_interval_mined_queries': (
            maintenance_interval_mined_queries(cfg) if cfg.get('cache_update_interval')
            else None),
    }
    return errors, warnings, info


def format_refresh_report(errors, warnings, info):
    """Human-readable block for the preflight / orchestrator log."""
    lines = [
        f"  max_age_steps    : {info['max_age_steps']} "
        f"(from {info['max_age_source']})",
        f"  total_steps      : {info['total_steps']}",
        f"  checkpoint every : {info['async_mine_every_steps']} steps",
        f"  first refresh-eligible checkpoint: "
        f"{info['first_refresh_checkpoint_step']}",
        f"  initial maintenance budget/interval: {info['initial_interval_budget']} "
        f"slots every {info['maintenance_interval_mined_queries']} mined queries",
    ]
    lines += [f"  ⚠️  {w}" for w in warnings]
    lines += [f"  ❌ {e}" for e in errors]
    return "\n".join(lines)


# ---- cfg-shaped encode wrappers --------------------------------------------

def encode_queries_mc(model, tokenizer, texts, T, device, cfg, dtype=None):
    """``T`` stochastic query passes -> ``[T, B_query, D]``."""
    return encode_mc(model, tokenizer, texts, T, device,
                     cfg.get('query_max_len', 128),
                     _miner_mc_batch(cfg), dtype=dtype)


def encode_docs_mc(model, tokenizer, texts, T, device, cfg, dtype=None):
    """``T`` stochastic document passes -> ``[T, n_docs, D]``."""
    return encode_mc(model, tokenizer, texts, T, device,
                     cfg.get('passage_max_len', 512),
                     _miner_mc_batch(cfg), dtype=dtype)


def _miner_mc_batch(cfg):
    """Miner-side MC encode batch, falling back to the shared encode batch."""
    return int(cfg.get('miner_mc_batch_size') or cfg.get('mc_batch_size', 256))


def init_Z_mc(cache, corpus_lookup, model, tokenizer, T, cfg, device):
    """Attach ``Z_mc[T, B_doc, D]`` to an already-sampled cache.

    Thin wrapper over ``encode_mc`` + ``NegativeCache.attach_mc_states`` for callers
    that built ``H`` with ``init_uniform`` (the Phase-0 harness). The orchestrator
    uses ``NegativeCache.init_cached_mcdp`` instead, which does both in one step.

    Returns ``(cache.Z_mc, stats)``.
    """
    texts = [corpus_lookup[d] for d in cache.docids]
    Z_mc, enc = encode_docs_mc(model, tokenizer, texts, T, device, cfg,
                               dtype=cache.Z_student.dtype)
    cache.attach_mc_states(Z_mc)
    cache.last_refreshed_step[:] = 0
    stats = {
        'init_docs': len(texts),
        'init_mc_passes': enc['mc_passes'],
        'init_examples_encoded': enc['examples_encoded'],
        'init_forward_batches': enc['forward_batches'],
        'cache_mc_bytes': cache.mc_memory_bytes(),
        'cache_mean_bytes': int(cache.Z_student.element_size()
                                * cache.Z_student.nelement()),
    }
    return cache.Z_mc, stats

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


# ---- cached-MCDP maintenance ------------------------------------------------

def maintain_interval_cached_mcdp(cache, student, tokenizer, corpus_lookup,
                                  all_docids, reservoir, source_checkpoint_step,
                                  T, cfg, device, qrels_dict=None):
    """One bounded in-round maintenance interval — delegates to the cache.

    Kept as a named entry point because the miner and the timing harness talk in
    terms of *intervals*, while the cache exposes a single maintenance call. ``T``
    is accepted for call-site clarity and validated against the cache's own ``T``.

    `R_doc` is deferred: candidates are uniform-only and registry counters are zero.
    """
    if T is not None and cache.T is not None and int(T) != int(cache.T):
        raise ValueError(f"T mismatch: caller says {T}, cache holds {cache.T}")
    return cache.maintain_cached_mcdp(
        student, tokenizer, corpus_lookup, all_docids, reservoir,
        source_checkpoint_step, cfg, device, qrels_dict=qrels_dict)



# ---- mining ----------------------------------------------------------------

def mine_batch_cached_mcdp(cache, student, tokenizer, batch_qids,
                           qid_to_text, qrels_dict, T, cfg, device,
                           chunk_size=None, age_step=None):
    """One virtual mining batch: T query passes + T matmuls over all of H.

    Performs **zero document encoder calls** — that invariant is what separates
    cached-MCDP from the lazy top-``L`` path and is asserted by the timing harness
    and the CPU tests. There is no top-``L``: every query is scored against all of
    ``H`` from the cached MC states.

    ``age_step`` is **model time** (``source_checkpoint_step``); pass it to get the
    selected-document age diagnostic. ``None`` skips that field — it never affects
    selection, only reporting.

    Returns ``(mined, slots, q_mc, stats)``.
    """
    texts = [qid_to_text[q] for q in batch_qids]
    q_mc, enc = encode_queries_mc(student, tokenizer, texts, T, device, cfg)

    # cache.score_cached_mcdp owns the cache_score_pairs accounting
    g, s_hat, sigma = cache.score_cached_mcdp(q_mc, cfg['lambda_val'],
                                              chunk_size=chunk_size)
    g = cache.mask_positives(g, batch_qids, qrels_dict, inplace=True)
    mode = cfg['selection_mode']
    slots, neg_docids = cache.select(g, m=cfg['m'], mode=mode,
                                     beta=cfg.get('beta', 5.0), L=None)
    cache.record_selection(slots)

    mined = {qid: neg_docids[i] for i, qid in enumerate(batch_qids)}
    sel_g = torch.gather(g, 1, slots)
    sel_sigma = torch.gather(sigma, 1, slots)
    sel_s_hat = torch.gather(s_hat, 1, slots)
    lam = float(cfg['lambda_val'])
    stats = {
        'query_mc_passes': enc['mc_passes'],
        'query_examples_encoded': enc['examples_encoded'],
        'query_forward_batches': enc['forward_batches'],
        # the defining invariant of cached-MCDP
        'mcdp_doc_encoder_calls_mining': 0,
        'cache_score_pairs_batch': len(batch_qids) * cache.B_doc,
        'sel_g_mean': float(sel_g.float().mean()),
        'sel_sigma_mean': float(sel_sigma.float().mean()),
        'sel_s_hat_mean': float(sel_s_hat.float().mean()),
        'sel_lambda_sigma_mean': lam * float(sel_sigma.float().mean()),
        's_hat_mean': float(s_hat[torch.isfinite(s_hat)].float().mean()),
        'sigma_mean': float(sigma.float().mean()),
        'num_queries': len(batch_qids),
    }

    # --- selected document age (model time - last refresh) ---
    if age_step is not None:
        age = (int(age_step) - cache.last_refreshed_step[slots.reshape(-1)]).float()
        stats['sel_age_mean'] = float(age.mean())
        stats['sel_age_max'] = float(age.max())

    # --- flip rate vs lambda=0, TopK ONLY ---
    # `softmax` selection is a Gumbel top-k SAMPLE, so comparing it to a lambda=0
    # argmax would measure sampling noise, not the uncertainty term. Report null
    # rather than a misleading number.
    if str(mode).lower() == 'topk':
        s0 = cache.mask_positives(s_hat.clone(), batch_qids, qrels_dict, inplace=True)
        top0 = torch.argmax(s0, dim=1)
        stats['flip_rate_vs_lambda0'] = float(
            (top0 != slots[:, 0]).float().mean())
        stats['flip_rate_unsupported_reason'] = None
    else:
        stats['flip_rate_vs_lambda0'] = None
        stats['flip_rate_unsupported_reason'] = (
            f"selection_mode={mode!r} draws a Gumbel top-k sample; a flip against the "
            f"lambda=0 argmax would report sampling noise, not the uncertainty term")
    return mined, slots, q_mc, stats


class MiningDiagnostics:
    """Query-weighted accumulator for the per-batch cached-MCDP diagnostics.

    The miner discarded these before: ``mine_batch_cached_mcdp`` already computed
    selected ``s_hat``/``sigma`` every batch and threw them away, so a finished run
    could not answer "did lambda change anything, and against what document ages?".

    Weighting by ``num_queries`` (not by batch count) keeps a short final batch from
    counting as much as a full one.
    """

    _MEAN_KEYS = ('sel_s_hat_mean', 'sel_sigma_mean', 'sel_lambda_sigma_mean',
                  'sel_g_mean', 's_hat_mean', 'sigma_mean', 'flip_rate_vs_lambda0',
                  'sel_age_mean')

    def __init__(self):
        self._sums = {k: 0.0 for k in self._MEAN_KEYS}
        self._weights = {k: 0 for k in self._MEAN_KEYS}
        self.num_queries = 0
        self.num_batches = 0
        self.sel_age_max = None
        self.flip_rate_unsupported_reason = None

    def add(self, stats):
        n = int(stats.get('num_queries', 0))
        self.num_queries += n
        self.num_batches += 1
        for k in self._MEAN_KEYS:
            v = stats.get(k)
            if v is None:
                continue
            self._sums[k] += float(v) * n
            self._weights[k] += n
        if stats.get('sel_age_max') is not None:
            self.sel_age_max = (stats['sel_age_max'] if self.sel_age_max is None
                                else max(self.sel_age_max, stats['sel_age_max']))
        if stats.get('flip_rate_unsupported_reason'):
            self.flip_rate_unsupported_reason = stats['flip_rate_unsupported_reason']

    def summary(self):
        """``mining_meta`` fields. Keys with no observations are ``None``, never 0.0."""
        out = {k: (self._sums[k] / self._weights[k] if self._weights[k] else None)
               for k in self._MEAN_KEYS}
        out['sel_age_max'] = self.sel_age_max
        out['flip_rate_unsupported_reason'] = self.flip_rate_unsupported_reason
        out['diagnostics_num_queries'] = self.num_queries
        out['diagnostics_num_batches'] = self.num_batches
        return out
