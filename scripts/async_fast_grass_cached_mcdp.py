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
    """
    cfg = dict(ctx['args'])
    cfg.pop('L', None)
    cfg['lambda_val'] = float(cfg['lambda_val'])
    cfg['query_max_len'] = config['model']['query_max_len']
    cfg['passage_max_len'] = config['model']['passage_max_len']
    cfg['steps_per_epoch'] = int(steps_per_epoch)
    cfg['total_steps'] = cfg['steps_per_epoch'] * cfg['num_epochs']
    cfg['max_age_steps'] = cfg['max_age_epochs'] * cfg['steps_per_epoch']
    return cfg


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
                           chunk_size=None):
    """One virtual mining batch: T query passes + T matmuls over all of H.

    Performs **zero document encoder calls** — that invariant is what separates
    cached-MCDP from the lazy top-``L`` path and is asserted by the timing harness
    and the CPU tests. There is no top-``L``: every query is scored against all of
    ``H`` from the cached MC states.

    Returns ``(mined, slots, q_mc, stats)``.
    """
    texts = [qid_to_text[q] for q in batch_qids]
    q_mc, enc = encode_queries_mc(student, tokenizer, texts, T, device, cfg)

    # cache.score_cached_mcdp owns the cache_score_pairs accounting
    g, s_hat, sigma = cache.score_cached_mcdp(q_mc, cfg['lambda_val'],
                                              chunk_size=chunk_size)
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
