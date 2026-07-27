"""
Async Fast-GRASS — MINER process (GPU 1).

Owns ALL selection state: the bounded global cache ``H``, ``Z_mc``, ``Z_mean``,
utility, and cache age. The trainer never reads or mutates any of it; this process
never touches gradients, the optimizer, or the loss.

Per round (async_fast_grass_implementation_details.md, "Miner Loop")::

    ckpt = newest valid checkpoint not already mined   (optimizer.pt = validity flag)
    freeze its weights for the WHOLE round             (dropout still on for MC)
    for each virtual batch over the full mixture:
        T stochastic query encodes + T matmuls over all of H   (0 doc encodes)
        mask qrels, select m negatives, record for utility
        every cache_update_interval * batch_size mined queries: maintain H
    fold the remainder; one final bounded interval only if useful state is pending
    write work_N/, then publish -> ready_N LAST

This is ANCE-style orchestration but NOT ANCE mining: no full-corpus ANN rebuild,
no per-query stale FAISS top-P. The stale pickle supplies only the docid sample for
``H`` on a cold start; MC states always come from real dropout passes.

Model time for the whole round is ``source_checkpoint_step``. Cache age, the
rho/progress budget, and ``last_refreshed_step`` are pinned to it and must not move
within a round — a miner-local counter (batch index, mined-query count, interval
index) would corrupt all three.

`R_doc` is DEFERRED: replacement candidates are sampled uniformly from the corpus
excluding documents already in ``H``, then recertified against the query-MC
reservoir. Registry counters are reported as zero.

Run by ``train_async_fast_grass.py``; not usually invoked directly.
"""
import argparse
import json
import math
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root / 'src'))
sys.path.insert(0, str(project_root / 'scripts'))

from utils.helpers import (  # noqa: E402
    get_path, get_training_context, load_config, set_seed,
    _load_corpus_lookup, _load_qrels,
)
from utils.negative_cache import NegativeCache  # noqa: E402
import run_fast_grass  # noqa: E402
from async_fast_grass_cached_mcdp import (  # noqa: E402
    mine_batch_cached_mcdp, maintain_interval_cached_mcdp, QueryMCReservoir,
    MaintenanceDriver, maintenance_interval_mined_queries, build_async_cfg,
    steps_per_epoch, canonicalize_positives,
)
from async_fast_grass_handoff import (  # noqa: E402
    publish_round, reap_orphans, prune_cache_states, resolve_cache_state,
    work_paths, round_paths, newest_valid_checkpoint, read_meta,
)


def _log(msg):
    print(f"[Miner] {msg}", flush=True)


def _load_frozen_student(path, cfg, device):
    """Load checkpoint weights, freeze them, and set the MC dropout rate.

    "Frozen for the round" means no parameter updates and no gradients, NOT
    dropout-off — ``dropout_only`` (inside ``encode_mc``) re-enables just the
    Dropout modules for the stochastic passes.
    """
    dtype = torch.bfloat16 if device.type == 'cuda' else torch.float32
    model = AutoModel.from_pretrained(str(path), torch_dtype=dtype).to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    mc_p = cfg.get('mc_dropout_p', 0.3)
    n = 0
    for mod in model.modules():
        if isinstance(mod, torch.nn.Dropout):
            mod.p = mc_p
            n += 1
    _log(f"loaded {Path(path).name} | MC dropout p={mc_p} on {n} layers")
    return model


def _model_hidden_size(model_path):
    """Embedding dimension from the model config, not from config.yaml.

    ``config/config.yaml`` has no ``embedding_dim`` key, so validating against it
    yields ``None`` and silently disables the dimension check in production. The
    encoder's own ``hidden_size`` is the ground truth.

    Raises rather than returning ``None``: the miner cannot encode anything without
    a readable model config, so an unreadable one is a hard failure — degrading to
    an unvalidated cache dimension would let a mismatched cache load and produce
    silently meaningless scores.
    """
    from transformers import AutoConfig
    try:
        hidden = AutoConfig.from_pretrained(str(model_path)).hidden_size
    except Exception as e:
        raise RuntimeError(
            f"could not read hidden_size from {model_path} ({e}). The miner needs "
            f"this model to encode, so an unreadable config is fatal; it must not "
            f"fall back to skipping cache-dimension validation.") from e
    return int(hidden)


def _cache_age_stats(cache, step):
    age = (step - cache.last_refreshed_step).float()
    return {
        'cache_age_mean_steps': float(age.mean()),
        'cache_age_p95_steps': float(torch.quantile(age, 0.95)) if age.numel() else 0.0,
        'cache_age_max_steps': float(age.max()) if age.numel() else 0.0,
    }


def mine_round(cache, student, tokenizer, train_items, qid_to_text, corpus_lookup,
               qrels_dict, c_ids, cfg, device, round_no, source_checkpoint_step,
               source_checkpoint, root, batch_size, chunk_size=None):
    """Mine one complete round into ``work_N/`` and publish it.

    Returns the ``mining_meta`` dict that was written.
    """
    w = work_paths(root, round_no)
    shutil.rmtree(w['work'], ignore_errors=True)
    w['training_data'].mkdir(parents=True, exist_ok=True)

    T = int(cache.T)
    reservoir = QueryMCReservoir(cfg.get('recent_query_reservoir_size', 128))
    driver = MaintenanceDriver(cfg, batch_size)
    maint_records, maint_time = [], 0.0
    doc_calls_mining = 0
    q_examples = q_batches = 0
    n_batches = max(int(math.ceil(len(train_items) / batch_size)), 1)

    _log(f"round {round_no}: mining {len(train_items):,} queries in {n_batches:,} "
         f"batches @ step {source_checkpoint_step:,} | maintain every "
         f"{driver.threshold:,} mined queries")

    t0 = time.perf_counter()
    out_path = w['training_data'] / "mined.jsonl"
    qid_to_pos = {it['query_id']: it['pos_docid'] for it in train_items}

    with open(out_path, 'w') as f_out:
        for b in range(n_batches):
            batch = train_items[b * batch_size:(b + 1) * batch_size]
            if not batch:
                continue
            batch_qids = list(dict.fromkeys(it['query_id'] for it in batch))

            mined, _slots, q_mc, mstats = mine_batch_cached_mcdp(
                cache, student, tokenizer, batch_qids, qid_to_text, qrels_dict,
                T, cfg, device, chunk_size=chunk_size)

            doc_calls_mining += mstats['mcdp_doc_encoder_calls_mining']
            q_examples += mstats['query_examples_encoded']
            q_batches += mstats['query_forward_batches']

            # docid-only records: the trainer resolves text through the same
            # corpus_lookup, so rounds stay small over the full mixture.
            for qid in batch_qids:
                f_out.write(json.dumps({
                    'query_id': qid,
                    'query': qid_to_text[qid],
                    'pos_docid': qid_to_pos.get(qid),
                    'neg_docids': mined.get(qid, []),
                }, ensure_ascii=False) + "\n")

            reservoir.add(q_mc, batch_qids)

            driver.add(len(batch_qids))
            while driver.should_fire():
                driver.consume()
                mt0 = time.perf_counter()
                maint_records.append(maintain_interval_cached_mcdp(
                    cache, student, tokenizer, corpus_lookup, c_ids,
                    reservoir.get(), source_checkpoint_step, T, cfg, device,
                    qrels_dict=qrels_dict))
                maint_time += time.perf_counter() - mt0
                if len(maint_records) % 10 == 1:
                    r = maint_records[-1]
                    _log(f"  maintenance #{driver.n_intervals} after "
                         f"{driver.mined_total:,} queries: refresh={r['num_refresh']} "
                         f"replace={r['num_replace']} budget="
                         f"{r['maintenance_budget_interval']}")

    # round end: fold the remainder, but only pay for a final interval if useful
    # pending state exists
    if driver.round_end_should_maintain(cache):
        mt0 = time.perf_counter()
        maint_records.append(maintain_interval_cached_mcdp(
            cache, student, tokenizer, corpus_lookup, c_ids, reservoir.get(),
            source_checkpoint_step, T, cfg, device, qrels_dict=qrels_dict))
        maint_time += time.perf_counter() - mt0
        _log(f"  final round-end interval ({driver.pending:,} pending queries)")

    t_mine_round = time.perf_counter() - t0

    if doc_calls_mining != 0:
        raise AssertionError(
            f"cached-MCDP mining performed {doc_calls_mining} document encoder "
            f"calls; the architecture requires 0 (regression to lazy fresh-MCDP)")

    agg = lambda k: int(sum(r.get(k, 0) for r in maint_records))
    meta = {
        'round_no': round_no,
        'source_checkpoint': str(source_checkpoint),
        'source_checkpoint_step': int(source_checkpoint_step),
        'B_doc': cache.B_doc, 'T': T, 'm': cfg['m'],
        'lambda_val': cfg['lambda_val'], 'mc_dropout_p': cfg.get('mc_dropout_p'),
        'selection_mode': cfg['selection_mode'],
        'async_mine_every_steps': cfg.get('async_mine_every_steps'),
        'cache_update_interval': cfg['cache_update_interval'],
        'maintenance_interval_mined_queries': driver.threshold,
        'maintenance_budget_interval': int(
            cache._interval_budget(source_checkpoint_step, cfg)),
        'num_maintenance_intervals': len(maint_records),
        'maintenance_model_step': int(source_checkpoint_step),
        'num_queries': int(driver.mined_total),
        'num_refresh_total': agg('num_refresh'),
        'num_replace_total': agg('num_replace'),
        'num_over_age_total': agg('num_over_age'),
        'over_age_backlog_final': (maint_records[-1]['over_age_backlog']
                                   if maint_records else 0),
        # R_doc DEFERRED: candidates are uniform-only, nothing is admitted to a
        # registry, so these are zero by construction rather than by measurement.
        'num_R_entries': 0,
        'num_R_candidates_total': 0,
        'registry_deferred': True,
        'num_uniform_candidates_total': agg('num_uniform_candidates'),
        'num_recertified_candidates_total': agg('num_recertified_candidates'),
        'cache_turnover_rate_mean': (
            float(np.mean([r['cache_turnover_rate'] for r in maint_records]))
            if maint_records else 0.0),
        # the FINAL committed path, not the work_N scratch path this is written to —
        # a restart reads this to locate the state and work_N no longer exists
        'cache_state_path': str(round_paths(root, round_no)['cache_state']),
        't_mine_round': t_mine_round,
        'queries_per_second': (driver.mined_total / t_mine_round
                               if t_mine_round > 0 else 0.0),
        'mcdp_query_encoder_calls': q_batches,
        'mcdp_query_examples_encoded': q_examples,
        'mcdp_query_mc_passes': T,
        'mcdp_doc_encoder_calls_mining': 0,
        'mcdp_doc_encoder_calls_maintenance': agg('maintenance_forward_batches'),
        'mcdp_docs_encoded_maintenance': agg('maintenance_docs_encoded'),
        'maintenance_examples_encoded': agg('maintenance_examples_encoded'),
        'cache_mc_bytes': cache.mc_memory_bytes(),
        'cache_score_pairs': int(cache.cache_score_pairs),
        'cache_maintenance_time': maint_time,
        **_cache_age_stats(cache, source_checkpoint_step),
    }

    cache.save_state(w['cache_state'])
    w['mining_meta'].write_text(json.dumps(meta, indent=2))
    publish_round(root, round_no)
    _log(f"round {round_no} published: {t_mine_round:.1f}s | "
         f"{meta['queries_per_second']:.1f} q/s | "
         f"{len(maint_records)} maintenance intervals "
         f"(refresh={meta['num_refresh_total']} replace={meta['num_replace_total']})")
    return meta


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--async_dir', required=True, help='async_mining root')
    ap.add_argument('--output_model_dir', required=True, help='trainer checkpoint dir')
    ap.add_argument('--corpus_file', required=True)
    ap.add_argument('--qrels_file', required=True)
    ap.add_argument('--recipe', default='async_fast_grass')
    ap.add_argument('--max_rounds', type=int, default=None,
                    help='stop after N rounds (smoke/debug); default = run forever')
    ap.add_argument('--debug', action='store_true', help='512-item mixture')
    args = ap.parse_args()

    config = load_config()
    ctx = get_training_context(args.recipe)
    set_seed(config.get('seed', 42))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    root = Path(args.async_dir)

    train_items = run_fast_grass._load_train_items(debug=args.debug)
    if args.debug:
        train_items = train_items[:512]
    qid_to_text = {it['query_id']: it['query'] for it in train_items}
    corpus_lookup = _load_corpus_lookup(args.corpus_file)
    qrels_dict = _load_qrels(args.qrels_file)
    c_ids = list(corpus_lookup.keys())
    # positives in the mixture may carry pre-dedup docids; map them onto the
    # canonical ids that actually exist in the corpus (see canonicalize_positives)
    train_items, canon = canonicalize_positives(train_items, qrels_dict,
                                                corpus_lookup, log=_log)
    qid_to_text = {it['query_id']: it['query'] for it in train_items}

    batch_size = ctx['args'].get('batch_size', 64)
    # identical definition in all three processes: the maintenance budget
    # divides by it, so floor-vs-ceil would change how much cache is maintained
    spe = steps_per_epoch(len(train_items), batch_size)
    cfg = build_async_cfg(config, ctx, spe)
    chunk_size = cfg.get('score_chunk_size')

    # Recovery BEFORE loading: a crash can leave final-path artifacts above the
    # newest committed round, and those must not be mistaken for real rounds.
    reaped = reap_orphans(root)
    if reaped:
        _log(f"reaped uncommitted rounds {reaped}")

    state_path, committed = resolve_cache_state(root)
    # Validate schema/T/B_doc/dim against the live cfg BEFORE anything reaches the
    # GPU. B_doc must be the EFFECTIVE size: init_uniform clamps to the corpus, so a
    # corpus smaller than the configured B_doc would otherwise fail every restart.
    effective_b_doc = NegativeCache.effective_B_doc(cfg, len(c_ids))
    # Embedding dim comes from the MODEL, not config: config.yaml has no
    # embedding_dim key, so reading it would silently pass expect_dim=None and skip
    # the check entirely in production while tests (which pass it explicitly)
    # suggested it was covered.
    expect_dim = _model_hidden_size(ctx['base_model'])
    cache = NegativeCache.load_state(
        state_path, cfg, device,
        expect_T=int(cfg['T']), expect_B_doc=effective_b_doc,
        expect_dim=expect_dim)
    _log(f"loaded cache from {state_path.name} (committed round {committed}) | "
         f"B_doc={cache.B_doc} T={cache.T} | Z_mc={cache.mc_memory_bytes()/1e9:.2f} GB")

    tokenizer = AutoTokenizer.from_pretrained(ctx['base_model'])
    round_no = committed + 1
    # On restart, resume AFTER the checkpoint the newest committed round was mined
    # from. Starting at -1 would re-mine that same checkpoint, producing a duplicate
    # round with zero new information and stalling the pipeline for a full round.
    last_mined_step = -1
    if committed > 0:
        prev_meta = read_meta(root, committed)
        last_mined_step = int(prev_meta.get('source_checkpoint_step', -1))
        _log(f"resuming after checkpoint step {last_mined_step} "
             f"(round {committed} was mined from it)")
    poll_interval = ctx['args'].get('async_poll_interval', 60)
    keep = ctx['args'].get('cache_state_keep', 2)
    idle_total = 0.0
    rounds_done = 0

    _log(f"polling {args.output_model_dir} every {poll_interval}s | "
         f"maintain every {maintenance_interval_mined_queries(cfg, batch_size):,} "
         f"mined queries")

    while True:
        sel = newest_valid_checkpoint(args.output_model_dir,
                                      exclude_step=last_mined_step)
        if sel is None:
            # nothing newer to mine — this idle time is the signal that the
            # TRAINER is the bottleneck, not the miner
            time.sleep(poll_interval)
            idle_total += poll_interval
            continue

        step, ckpt = sel
        student = _load_frozen_student(ckpt, cfg, device)
        try:
            mine_round(cache, student, tokenizer, train_items, qid_to_text,
                       corpus_lookup, qrels_dict, c_ids, cfg, device, round_no,
                       step, ckpt, root, batch_size, chunk_size=chunk_size)
        finally:
            del student
            if device.type == 'cuda':
                torch.cuda.empty_cache()

        deleted = prune_cache_states(root, keep)
        if deleted:
            _log(f"pruned cache states {deleted} (kept newest {keep})")

        last_mined_step = step
        round_no += 1
        rounds_done += 1
        _log(f"miner_idle_time so far: {idle_total:.0f}s")
        if args.max_rounds and rounds_done >= args.max_rounds:
            _log(f"reached --max_rounds {args.max_rounds}; exiting")
            return 0


if __name__ == "__main__":
    sys.exit(main())
