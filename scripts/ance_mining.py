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
**sampling failure**, and a round with more than `max_sampling_failures` of them is
never published. Fabricating a negative -- from a positive, from a duplicate, or
from a uniform corpus draw -- would silently change the negative distribution ANCE
exists to define.

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
import shutil
from pathlib import Path

import numpy as np

project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root / 'src'))

from utils.helpers import _sha256, atomic_write, get_latest_marker_no  # noqa: E402


ROUND_META_NAME = "round_meta.json"
INITIAL_ROUND = "initial"


class RoundError(RuntimeError):
    """A round cannot be published, or cannot be trusted once read."""


class SamplingFailure(RuntimeError):
    """A query could not supply its ANN negatives without fabricating one."""


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


def build_round_records(mixture_files, mined_negs, corpus_lookup, *, n_negs):
    """Rewrite the mixture with mined negatives. Yields (filename, [records]).

    A mined docid with no corpus text is a failure, not an empty passage: an empty
    string trains the model against nothing while looking like a negative.
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
                negs = mined_negs.get(qid)
                if negs is None:
                    raise SamplingFailure(
                        f"{path.name}: query {qid} has no mined negatives. Every "
                        f"mixture record must be covered by the mining pass.")
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
                    n_negs, mining_depth, rng):
    """Search the ANN index and select negatives for every mixture query.

    Called by the orchestrator's initial mine and by the Inferencer's refresh mine,
    so the two can never drift -- which is exactly how the positive-as-negative
    fallback came to exist in both at once.
    """
    _, indices = index.search(q_data[0].astype(np.float32), mining_depth)
    row_of = {str(qid): i for i, qid in enumerate(q_data[1])}

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
        row = row_of.get(qid)
        if row is None:
            failures.append(f"{qid}: not present in the encoded query set")
            continue
        candidates = [corpus_ids[j] for j in indices[row] if j >= 0]
        try:
            mined[qid] = select_ance_negatives(qid, candidates, positives,
                                               n_negs=n_negs, rng=rng)
        except SamplingFailure as exc:
            failures.append(str(exc))
    return mined, failures


# ---- round layout -----------------------------------------------------------

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

def publish_round(root, n, *, records_by_file, meta, max_sampling_failures=0):
    """Stage a round under work_*/ and commit it, marker LAST.

    ``meta`` must already carry the round's provenance; the record count and the
    sampling-failure budget are filled in here so they cannot disagree with what was
    actually written.
    """
    failures = int(meta.get('n_sampling_failures', 0))
    if failures > max_sampling_failures:
        raise RoundError(
            f"round {n}: {failures} sampling failure(s) exceed the budget of "
            f"{max_sampling_failures}. ANCE never fabricates a negative, so the "
            f"round is discarded rather than published with substitute negatives.")

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
    meta['file_sha256'] = {name: _sha256(staged / name) for name in meta['files']}
    meta['max_sampling_failures'] = int(max_sampling_failures)
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

    Checks, in order: the marker exists; the metadata exists; the metadata belongs
    to THIS run; the files it claims are present; their content hashes and record count
    match. A round
    that fails any of these is refused, never silently consumed -- a leftover
    ``ready_7`` from a previous run is exactly how another run's negatives used to
    reach the trainer.
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
    claimed_hashes = meta.get('file_sha256')
    if not isinstance(claimed_hashes, dict) or sorted(claimed_hashes) != claimed:
        raise RoundError(
            f"round {n}: content hashes are missing or do not cover exactly the "
            f"claimed files {claimed}")
    counted = 0
    for name in present:
        actual_hash = _sha256(paths['training_data'] / name)
        if actual_hash != claimed_hashes[name]:
            raise RoundError(
                f"round {n}: {name} content hash {actual_hash} does not match "
                f"metadata {claimed_hashes[name]}; the committed round was modified")
        with open(paths['training_data'] / name, encoding='utf-8') as handle:
            counted += sum(1 for line in handle if line.strip())
    if counted != int(meta.get('n_records', -1)):
        raise RoundError(
            f"round {n}: {counted} record(s) on disk, metadata claims "
            f"{meta.get('n_records')}. The round is incomplete.")
    return paths['training_data'], meta


# ---- freshness gate ---------------------------------------------------------

def assert_ance_refresh(summary, *, min_fresh_rounds=1, min_consume_steps=1):
    """Refuse to call a run ANCE without a consumed, checkpoint-derived round.

    A run whose inferencer died at startup trains to `max_steps` on the base-model
    round and exits 0. That is static hard-negative training, not ANCE, and nothing
    in the logs distinguishes it. `checkpoint_step > 0` is what makes a round a
    refresh: the initial round is mined by the base model at step 0.
    """
    rounds = list(summary.get('rounds') or [])
    fresh = [r for r in rounds
             if int(r.get('checkpoint_step') or 0) > 0
             and int(r.get('consumed_steps') or 0) >= min_consume_steps]
    if len(fresh) < min_fresh_rounds:
        seen = [{'ann_no': r.get('ann_no'),
                 'checkpoint_step': r.get('checkpoint_step'),
                 'consumed_steps': r.get('consumed_steps')} for r in rounds]
        raise RoundError(
            f"ANCE requires at least {min_fresh_rounds} ANN round mined from a "
            f"checkpoint of this run and consumed for >= {min_consume_steps} "
            f"optimizer step(s); {len(fresh)} qualified. Rounds seen: {seen or 'none'}. "
            f"A run with no refresh trained on static base-model negatives and is "
            f"not ANCE.")
    return fresh
