"""ANCE negative selection and round commit protocol.

ANCE-local. Two concerns, both of which used to be duplicated verbatim between
`train_ance.py` (initial mine) and `run_ance_data_gen.py` (refresh mine), which is
how the positive-as-negative fallback came to exist in two places at once.

**Selection** follows the reference implementation
(`microsoft/ANCE:drivers/run_ann_data_gen.py`): retrieve `mining_depth` (200)
candidates from the ANN index, drop this query's positives, and sample uniformly
without replacement from what is left. The reference's default path shuffles the
retrieved list and takes negatives in shuffled order; `SelectTopK` slicing is only
its MRR-measurement mode. A query short of negatives simply yields fewer there --
it never falls back to a positive and never pads.

Here the group size is fixed (Tevatron wants exactly `train_group_size` passages),
so "fewer" is not representable. A query that cannot supply its negatives is a
**sampling failure**, and a single one discards the whole round. Fabricating a
negative -- from a positive, from a duplicate, or from a uniform corpus draw -- would
silently change the negative distribution ANCE exists to define.

Candidate ids are NOT deduplicated at selection time. FirstP puts exactly one vector
per corpus row in the index and the preprocessor canonicalizes docids, so a repeated
id would mean the pickle and the corpus disagree -- a per-round
`assert_corpus_ids_unique` says so once instead of every query quietly papering over
it. The reference dedups (`run_ann_data_gen.py:381-383`) because MaxP puts several
chunk vectors under one document id; that index does not exist here.

**Sharding** follows `run_ann_data_gen.py:281-295`. Every training query is encoded,
then the COMPLETE embedding array is sliced into `ann_chunk_factor` contiguous
shards and `ann_no % k` selects this round's. Order matters and is kept: upstream
encodes first and slices second, so this is not the same computation as encoding
only a shard.

**Commit** mirrors `async_fast_grass_handoff.publish_round`: everything is written
under `work_N/`, moved into place by atomic rename, and only then is the `ready_N`
    marker written. Metadata records each JSONL content hash and lands BEFORE the
    marker, so a marker can never point at a round whose provenance is unknown or whose
    contents changed after commit. The initial mine uses
`ready_initial` / `round_meta_initial.json`, which `latest_committed_round` ignores
-- the freshness gate counts numeric rounds only, because a base-model round is not
an ANN refresh.
"""
import os
import sys
import json
import pickle
import shutil
from pathlib import Path

import numpy as np

project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root / 'src'))

from utils.helpers import atomic_write, get_latest_marker_no  # noqa: E402


ROUND_META_NAME = "round_meta.json"
INITIAL_ROUND = "initial"


class RoundError(RuntimeError):
    """A round cannot be published, or cannot be trusted once read."""


class SamplingFailure(RuntimeError):
    """A query could not supply its ANN negatives without fabricating one."""


# ---- corpus / query preconditions -------------------------------------------

def assert_corpus_ids_unique(corpus_ids):
    """One FAISS row per document, checked once per round.

    ANCE's selection treats an ANN result list as a list of distinct documents. Under
    FirstP that holds by construction -- one embedding per corpus row -- so a repeat
    means the encoded pickle and the corpus disagree, and every downstream count
    (mining depth, retained negatives) is quietly wrong. Asserting here is cheaper and
    louder than deduplicating per query, and it does not pretend to support the
    multi-vector index that would actually need dedup.
    """
    corpus_ids = list(corpus_ids)
    if len(set(corpus_ids)) == len(corpus_ids):
        return len(corpus_ids)
    seen, dupes = set(), []
    for docid in corpus_ids:
        if docid in seen and docid not in dupes and len(dupes) < 5:
            dupes.append(docid)
        seen.add(docid)
    raise RoundError(
        f"the encoded corpus holds {len(corpus_ids)} vectors under {len(seen)} "
        f"distinct docid(s), e.g. {dupes}. FirstP emits one vector per corpus row, so "
        f"a repeat means the embedding pickle and the corpus disagree. Negatives are "
        f"not mined against an index whose ids cannot be trusted.")


def query_shard(query_ids, ann_no, chunk_factor):
    """The contiguous rotating shard, exactly as `run_ann_data_gen.py:281-295`.

    Returns ``(shard_index, start, end)`` so the caller slices the embeddings and the
    id array with one set of boundaries. ``queries_per_chunk`` FLOORS and the LAST
    shard runs to the end, so a non-divisible query count places every remainder in
    shard ``k-1``: the k shards stay disjoint and their union is the full array.

    Applied to the COMPLETE encoded query array, never to the mixture before encoding.
    """
    k = max(int(chunk_factor), 1)
    shard = int(ann_no) % k
    n = len(query_ids)
    per_chunk = n // k
    start = per_chunk * shard
    end = n if shard == k - 1 else start + per_chunk
    return shard, start, end


# ---- selection --------------------------------------------------------------

def select_ance_negatives(qid, candidate_ids, positives, *, n_negs, rng):
    """Uniformly sample ``n_negs`` distinct non-positive ANN candidates.

    ``candidate_ids`` is the ANN result list for this query, already truncated to
    the mining depth by the caller's FAISS search -- it is NOT re-sliced here. The
    old code searched to ``mining_depth`` and then sliced to ``mining_depth`` again,
    which reads as a cap while doing nothing.

    ``positives`` is the union of the query's qrels documents and the docids its
    mixture record labels positive. Raises ``SamplingFailure`` rather than returning
    a positive, a duplicate or a corpus-random filler.
    """
    if n_negs < 1:
        raise ValueError(f"n_negs must be >= 1, got {n_negs}")
    pool = [d for d in candidate_ids if d not in positives]
    if len(pool) < n_negs:
        raise SamplingFailure(
            f"query {qid}: {len(pool)} non-positive ANN candidate(s) among "
            f"{len(candidate_ids)} retrieved, need {n_negs}. ANCE never pads with a "
            f"positive or fills from the corpus, so this round cannot be published.")
    return rng.sample(pool, n_negs)


def record_positives(record, qrels_dict):
    """Every docid that must never be this record's negative.

    The qrels entry AND the mixture record's own `positive_passages`: a mixture
    label absent from the qrels file is still a positive for this training example.
    """
    qid = str(record['query_id'])
    positives = set(qrels_dict.get(qid, ()))
    for p in record.get('positive_passages') or []:
        positives.add(str(p['docid']))
    return positives


def build_round_records(mixture_files, mined_negs, corpus_lookup, *, n_negs,
                        shard_qids=None):
    """Rewrite the mixture with mined negatives. Yields (filename, [records]).

    A mined docid with no corpus text is a failure, not an empty passage: an empty
    string trains the model against nothing while looking like a negative.

    ``shard_qids`` is this round's query shard. A record outside it is skipped -- that
    is what makes the round one shard's worth of data. A record INSIDE it with no
    mined negatives is still a failure, so sharding cannot become a silent way to drop
    a query the miner could not serve.
    """
    for path in mixture_files:
        path = Path(path)
        out = []
        with open(path, encoding='utf-8') as handle:
            for line in handle:
                if not line.strip():
                    continue
                record = json.loads(line)
                qid = str(record['query_id'])
                if shard_qids is not None and qid not in shard_qids:
                    continue
                negs = mined_negs.get(qid)
                if negs is None:
                    raise SamplingFailure(
                        f"{path.name}: query {qid} has no mined negatives. Every "
                        f"mixture record in this round's shard must be covered by the "
                        f"mining pass.")
                if len(negs) != n_negs:
                    raise SamplingFailure(
                        f"{path.name}: query {qid} carries {len(negs)} negative(s), "
                        f"expected {n_negs}.")
                passages = []
                for docid in negs:
                    text = corpus_lookup.get(docid)
                    if not text:
                        raise SamplingFailure(
                            f"{path.name}: query {qid} selected docid {docid!r}, "
                            f"which has no text in the corpus.")
                    passages.append({"docid": docid, "text": text})
                record['negative_passages'] = passages
                out.append(record)
        yield path.name, out


def mine_from_index(index, corpus_ids, q_data, mixture_files, qrels_dict, *,
                    n_negs, mining_depth, rng, ann_no=0, chunk_factor=1,
                    search_batch_size=8192, phase=None):
    """Search the ANN index and select negatives for this round's query shard.

    Called by the orchestrator's initial mine and by the Inferencer's refresh mine,
    so the two can never drift -- which is exactly how the positive-as-negative
    fallback came to exist in both at once.

    ``q_data`` is the COMPLETE encoded query array; the shard is sliced out of it
    here, after encoding, as upstream does. Returns
    ``(mined, failures, shard_qids, shard_info)``.
    """
    all_qids = [str(qid) for qid in q_data[1]]
    shard, start, end = query_shard(all_qids, ann_no, chunk_factor)
    shard_ids = all_qids[start:end]
    shard_qids = set(shard_ids)

    if search_batch_size < 1:
        raise ValueError('search_batch_size must be positive')
    phase = phase or (lambda name: None)
    blocks = [index.search(np.asarray(q_data[0][i:min(i + search_batch_size, end)],
                                     dtype=np.float32), mining_depth)[1]
              for i in range(start, end, search_batch_size)]
    indices = np.concatenate(blocks) if blocks else np.empty((0, mining_depth), dtype=int)
    row_of = {qid: i for i, qid in enumerate(shard_ids)}
    encoded = set(all_qids)

    positives_by_qid = {}
    for path in mixture_files:
        with open(path, encoding='utf-8') as handle:
            for line in handle:
                if not line.strip():
                    continue
                record = json.loads(line)
                qid = str(record['query_id'])
                positives_by_qid.setdefault(qid, set()).update(
                    record_positives(record, qrels_dict))

    mined, failures = {}, []
    for qid, positives in positives_by_qid.items():
        if qid not in encoded:
            # Absent from the encode entirely -- still a failure at any shard count.
            failures.append(f"{qid}: not present in the encoded query set")
            continue
        row = row_of.get(qid)
        if row is None:
            continue                      # another shard's query, mined another round
        candidates = [corpus_ids[j] for j in indices[row] if j >= 0]
        try:
            mined[qid] = select_ance_negatives(qid, candidates, positives,
                                               n_negs=n_negs, rng=rng)
        except SamplingFailure as exc:
            failures.append(str(exc))
    shard_info = {'shard_index': shard, 'n_shards': max(int(chunk_factor), 1),
                  'n_shard_queries': len(shard_ids)}
    return mined, failures, shard_qids, shard_info


# ---- round layout -----------------------------------------------------------

def encode_and_mine(checkpoint, staging, *, corpus_file, query_file, mixture_files,
                    qrels_dict, ctx, config, rng, ann_no, phase=None):
    """Encode corpus and queries with `checkpoint`, index them, and mine one shard.

    The orchestrator's base-model round and the Inferencer's refresh rounds are the
    same computation; only the checkpoint, the shard number and the metadata differ.
    Keeping the sequence in one place is what stops the two from drifting apart on
    encode ORDER (all queries before any shard is selected, upstream's order), on the
    encode timeout, on the corpus-id uniqueness assertion, or on the mining arguments.

    `phase` is an optional label callback for the Inferencer's timing telemetry.

    Returns ``(index, corpus_ids, mined, failures, shard_qids, shard)``.
    """
    from utils.helpers import build_faiss_index, encode_to_pickle

    phase = phase or (lambda _label: None)
    timeout = ctx['args']['max_encode_seconds']
    staging = Path(staging)
    staging.mkdir(parents=True, exist_ok=True)

    # Paper: "recomputes the encodings of the entire corpus". The timeout is the stall
    # detector -- it raises, the process exits nonzero, and the orchestrator terminates
    # the trainer rather than letting it finish on stale negatives.
    phase("encode_corpus")
    encode_to_pickle(checkpoint, corpus_file, staging / "corpus.pkl", False, ctx,
                     config, timeout=timeout)
    if hasattr(phase, 'merge_encoder'):
        phase.merge_encoder('corpus', staging / 'corpus.pkl')
    # ALL training queries are encoded before any shard is selected, which is the
    # reference's order (run_ann_data_gen.py:256-295). Encoding only the shard would be
    # cheaper and is NOT the same computation.
    phase("encode_query")
    encode_to_pickle(checkpoint, query_file, staging / "query.pkl", True, ctx,
                     config, timeout=timeout)
    if hasattr(phase, 'merge_encoder'):
        phase.merge_encoder('query', staging / 'query.pkl')

    phase("index")
    index, corpus_embs, corpus_ids = build_faiss_index(
        staging / 'corpus.pkl', backend=ctx['args'].get('faiss_backend', 'cpu'),
        batch_size=ctx['args'].get('faiss_batch_size', 8192))
    del corpus_embs                    # FAISS copied them; 27 GiB on MS MARCO
    assert_corpus_ids_unique(corpus_ids)
    with open(staging / "query.pkl", 'rb') as handle:
        q_data = pickle.load(handle)

    # Paper Eq. 13: D^-_ANCE = ANN_{f(q,d)} \ D^+. Search and selection share one
    # phase because mine_from_index does both in one pass over the shard.
    phase("search")
    mined, failures, shard_qids, shard = mine_from_index(
        index, corpus_ids, q_data, mixture_files, qrels_dict,
        n_negs=ctx['args']['train_group_size'] - 1,
        mining_depth=ctx['args']['mining_depth'], rng=rng,
        ann_no=ann_no, chunk_factor=ctx['args']['ann_chunk_factor'],
        search_batch_size=ctx['args'].get('faiss_batch_size', 8192), phase=phase)
    return index, corpus_ids, mined, failures, shard_qids, shard


def _round_dirname(n):
    return "training_data_initial" if n == INITIAL_ROUND else f"training_data_{n}"


def round_paths(root, n):
    root = Path(root)
    suffix = INITIAL_ROUND if n == INITIAL_ROUND else str(n)
    return {
        'work':          root / f"work_{suffix}",
        'training_data': root / _round_dirname(n),
        'meta':          root / f"round_meta_{suffix}.json",
        'ready':         root / f"ready_{suffix}",
    }


def latest_committed_round(root):
    """Highest N with a numeric ``ready_N``, or 0. Ignores ``ready_initial``."""
    root = Path(root)
    if not root.is_dir():
        return 0
    return get_latest_marker_no(root, prefix="ready_")


# ---- commit -----------------------------------------------------------------

def publish_round(root, n, *, records_by_file, meta):
    """Stage a round under work_*/ and commit it, marker LAST.

    ``meta`` must already carry the round's provenance; the record count is filled in
    here so it cannot disagree with what was actually written.

    ANCE never fabricates a negative, so a round that could not supply one for every
    query is discarded rather than published with substitutes.
    """
    failures = int(meta.get('n_sampling_failures', 0))
    if failures:
        raise RoundError(
            f"round {n}: {failures} query(ies) could not supply an ANN negative. "
            f"ANCE never fabricates a negative, so the round is discarded rather "
            f"than published with substitutes.")

    paths = round_paths(root, n)
    Path(root).mkdir(parents=True, exist_ok=True)
    shutil.rmtree(paths['work'], ignore_errors=True)
    staged = paths['work'] / _round_dirname(n)
    staged.mkdir(parents=True)

    total = 0
    for name, records in records_by_file:
        with atomic_write(staged / name) as handle:
            for record in records:
                handle.write(json.dumps(record, ensure_ascii=False) + '\n')
        total += len(records)

    meta = dict(meta)
    meta['n_records'] = total
    meta['files'] = sorted(p.name for p in staged.glob("*.jsonl"))
    with atomic_write(paths['work'] / ROUND_META_NAME) as handle:
        json.dump(meta, handle, indent=2, default=str)

    if paths['training_data'].exists():
        shutil.rmtree(paths['training_data'], ignore_errors=True)
    os.replace(staged, paths['training_data'])
    os.replace(paths['work'] / ROUND_META_NAME, paths['meta'])   # meta before marker

    tmp = Path(root) / f"{paths['ready'].name}.tmp"
    tmp.write_text(str(n))
    os.replace(tmp, paths['ready'])                              # marker LAST

    shutil.rmtree(paths['work'], ignore_errors=True)
    return paths


def adopt_prepared_round(root, n, *, source_dir, meta):
    """Publish a round mined by a SEPARATE preparation job into this run's work root.

    Keeps round 0's full-corpus encode out of the training allocation; upstream splits
    the same stage out (run_train.sh --end_output_num 0). Commit order matches
    publish_round -- payload, metadata, marker -- so a partial adoption is unobservable.

    It rewrites the metadata with THIS run's id, and that check is the only thing
    stopping a foreign round reaching the trainer, so rebinding is legitimate only after
    the caller has verified the artifact (train_ance.assert_prepared_initial_matches).
    """
    source_dir = Path(source_dir)
    files = sorted(p for p in source_dir.glob("*.jsonl"))
    if not files:
        raise RoundError(
            f"prepared round at {source_dir} holds no .jsonl payload, so there is "
            f"nothing to adopt.")
    if not meta.get('prepared_from'):
        raise RoundError(
            "adopt_prepared_round requires meta['prepared_from'] recording which "
            "artifact this round came from; a round with no adoption provenance is "
            "indistinguishable from one this run mined itself.")

    paths = round_paths(root, n)
    Path(root).mkdir(parents=True, exist_ok=True)
    shutil.rmtree(paths['work'], ignore_errors=True)
    staged = paths['work'] / _round_dirname(n)
    staged.mkdir(parents=True)
    for path in files:
        shutil.copyfile(path, staged / path.name)

    meta = dict(meta)
    meta['files'] = sorted(p.name for p in staged.glob("*.jsonl"))
    with atomic_write(paths['work'] / ROUND_META_NAME) as handle:
        json.dump(meta, handle, indent=2, default=str)

    if paths['training_data'].exists():
        shutil.rmtree(paths['training_data'], ignore_errors=True)
    os.replace(staged, paths['training_data'])
    os.replace(paths['work'] / ROUND_META_NAME, paths['meta'])   # meta before marker

    tmp = Path(root) / f"{paths['ready'].name}.tmp"
    tmp.write_text(str(n))
    os.replace(tmp, paths['ready'])                              # marker LAST

    shutil.rmtree(paths['work'], ignore_errors=True)
    return paths


def read_round(root, n, *, run_id):
    """Validate a committed round and return (data_dir, meta).

    Checks, in order: the marker exists; the metadata exists and parses; the metadata
    belongs to THIS run; the files it claims are present. A round that fails any of
    these is refused, never silently consumed -- a leftover ``ready_7`` from a previous
    run is exactly how another run's negatives used to reach the trainer.

    Content is NOT re-hashed and records are NOT recounted. Every file is written
    through ``atomic_write`` and the directory is published by ``os.replace`` with the
    marker last, so a partially written round cannot be observed in the first place.
    Re-reading every byte of a multi-GB round on the trainer's critical path defended
    against nothing that ordering does not already prevent.
    """
    paths = round_paths(root, n)
    if not paths['ready'].exists():
        raise RoundError(f"round {n}: no {paths['ready'].name} marker")
    if not paths['meta'].is_file():
        raise RoundError(
            f"round {n}: {paths['meta'].name} is missing, so the round's "
            f"provenance cannot be established")
    try:
        meta = json.loads(paths['meta'].read_text())
    except ValueError as exc:
        raise RoundError(f"round {n}: {paths['meta'].name} is not valid JSON") from exc

    if meta.get('run_id') != run_id:
        raise RoundError(
            f"round {n} was mined by run {meta.get('run_id')!r}, not {run_id!r}. "
            f"Training on another run's negatives is not this experiment.")
    if not paths['training_data'].is_dir():
        raise RoundError(f"round {n}: {paths['training_data'].name} is missing")

    present = sorted(p.name for p in paths['training_data'].glob("*.jsonl"))
    claimed = sorted(meta.get('files') or [])
    if present != claimed:
        raise RoundError(
            f"round {n}: files on disk {present} do not match the metadata "
            f"{claimed}")
    return paths['training_data'], meta


# ---- freshness gate ---------------------------------------------------------

def assert_ance_refresh(summary, *, min_fresh_rounds=2, min_consume_steps=1):
    """Refuse to call a run ANCE without repeated, consumed, checkpoint-derived rounds.

    Three things a weaker gate let through, all of which produce a number that looks
    like ANCE and is not:

    * an inferencer that died at startup -- the trainer runs to `max_steps` on the
      base-model round and exits 0. `checkpoint_step > 0` is what makes a round a
      refresh, because the initial round is mined by the base model at step 0;
    * a single refresh consumed for a single optimizer step, followed by tens of
      thousands of static steps. `min_consume_steps` is the logging interval, so a
      qualifying round has to have actually trained something;
    * two rounds mined from the SAME checkpoint, which is one refresh reported twice.
      Only distinct `checkpoint_step` values count, so `min_fresh_rounds` counts
      iterations of the ANCE loop rather than publications.
    """
    rounds = list(summary.get('rounds') or [])
    fresh, seen_checkpoints = [], set()
    for r in rounds:
        step = int(r.get('checkpoint_step') or 0)
        if step <= 0 or int(r.get('consumed_steps') or 0) < min_consume_steps:
            continue
        if step in seen_checkpoints:
            continue
        seen_checkpoints.add(step)
        fresh.append(r)
    if len(fresh) < min_fresh_rounds:
        seen = [{'ann_no': r.get('ann_no'),
                 'checkpoint_step': r.get('checkpoint_step'),
                 'consumed_steps': r.get('consumed_steps')} for r in rounds]
        raise RoundError(
            f"ANCE requires at least {min_fresh_rounds} ANN round(s) mined from "
            f"DISTINCT checkpoints of this run, each consumed for >= "
            f"{min_consume_steps} optimizer step(s); {len(fresh)} qualified. Rounds "
            f"seen: {seen or 'none'}. A run that refreshed once, or not at all, "
            f"trained on effectively static negatives and is not ANCE.")
    return fresh
