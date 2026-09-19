"""ANCE handoff, supervision, freshness and success validation.

The defects these pin, all of which shipped:

* the trainer started at `last_ann_no = 0` while the inferencer numbered from
  `get_latest_marker_no(ann_dir) + 1` in a work root shared by every run, so a
  leftover `ready_7` was swapped in at the first logging step;
* the inferencer was an unsupervised `Popen`, so a run whose miner died trained to
  `max_steps` on base-model negatives and exited 0 as "ANCE";
* nothing checked the loss for NaN and nothing called `assert_training_succeeded`.

Run: python tests/ance_pipeline_test.py
"""
import json
import random
import os
import shutil
import subprocess
import sys
import tempfile
import traceback
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root / 'src'))
sys.path.insert(0, str(project_root / 'scripts'))

from utils.helpers import (                                       # noqa: E402
    RUN_MANIFEST_NAME, TRAINING_LOG_NAME, RunDirectoryError, append_jsonl,
    assert_training_succeeded, build_run_manifest, prepare_output_dir,
)
from ance_mining import (                                         # noqa: E402
    INITIAL_ROUND, RoundError, assert_ance_refresh, latest_committed_round,
    publish_round, read_round, round_paths,
)


def _assert_raises(exc, fn, contains=None):
    try:
        fn()
    except exc as e:
        assert contains is None or contains in str(e), str(e)
        return str(e)
    raise AssertionError(f"expected {exc.__name__}")


RUN_ID = "abc123def456-1700000000"


def _records(n=3, prefix="q"):
    return [{'query_id': f'{prefix}{i}', 'query': f'query {i}',
             'positive_passages': [{'docid': f'p{i}', 'text': f'pos {i}'}],
             'negative_passages': [{'docid': f'n{i}', 'text': f'neg {i}'}]}
            for i in range(n)]


def _publish(root, n, *, run_id=RUN_ID, step=1000, failures=0, records=None):
    return publish_round(
        root, n,
        records_by_file=[("train_hq.jsonl", records if records is not None
                          else _records())],
        meta={'run_id': run_id, 'ann_no': n,
              'checkpoint': f'/models/x/checkpoint-{step}', 'checkpoint_step': step,
              'n_queries_mined': 3, 'n_sampling_failures': failures})


# ---- round commit protocol --------------------------------------------------

def test_marker_is_written_last_and_metadata_first():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        paths = _publish(root, 1)
        assert paths['meta'].is_file() and paths['ready'].is_file()
        meta = json.loads(paths['meta'].read_text())
        assert meta['files'] == ["train_hq.jsonl"] and meta['n_records'] == 3
        # marker mtime is never older than the metadata it vouches for
        assert paths['ready'].stat().st_mtime >= paths['meta'].stat().st_mtime
        assert not paths['work'].exists(), "staging directory left behind"


def test_a_round_without_its_marker_is_invisible():
    """A crash between the rename sequence and the marker leaves final-path
    artifacts. Those are not a committed round."""
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        _publish(root, 1)
        _publish(root, 2)
        round_paths(root, 2)['ready'].unlink()
        assert latest_committed_round(root) == 1
        _assert_raises(RoundError, lambda: read_round(root, 2, run_id=RUN_ID),
                       "no ready_2 marker")


def test_a_foreign_runs_round_is_refused():
    """The core E1 case: a leftover round from a previous run used to be consumed
    at the first logging step because the trainer only compared round numbers."""
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        _publish(root, 7, run_id="OTHER-RUN-9999")
        assert latest_committed_round(root) == 7, "still discoverable by number"
        msg = _assert_raises(RoundError, lambda: read_round(root, 7, run_id=RUN_ID),
                             "not this experiment")
        assert "OTHER-RUN-9999" in msg and RUN_ID in msg, msg


def _corrupt(paths, how):
    """Damage a committed round in one specific way. Each case is a distinct route
    by which a round could reach the trainer without being what it claims."""
    data, meta = paths['training_data'], paths['meta']
    if how == 'no_meta':
        meta.unlink()
    elif how == 'no_data_dir':
        shutil.rmtree(data)
    elif how == 'bad_meta_json':
        meta.write_text("{not json")
    elif how == 'extra_file':
        (data / "stray.jsonl").write_text('{"query_id":"z"}\n')
    elif how == 'missing_claimed_file':
        (data / "train_hq.jsonl").unlink()
    elif how == 'foreign_run':
        payload = json.loads(meta.read_text())
        payload['run_id'] = "OTHER-RUN-9999"
        meta.write_text(json.dumps(payload))
    else:                                                    # pragma: no cover
        raise AssertionError(how)


MALFORMED_ROUNDS = [
    ('no_meta',              "provenance cannot be established"),
    ('bad_meta_json',        "not valid JSON"),
    ('foreign_run',          "not this experiment"),
    ('no_data_dir',          "is missing"),
    ('extra_file',           "do not match the metadata"),
    ('missing_claimed_file', "do not match the metadata"),
]


def test_a_malformed_round_is_always_refused():
    """One table, one assertion per route. Content is deliberately NOT re-hashed --
    atomic_write plus marker-last publication means a partially written round is
    never observable, so re-reading every byte of a multi-GB round on the trainer's
    critical path defended against nothing."""
    for how, message in MALFORMED_ROUNDS:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = _publish(root, 1)
            _corrupt(paths, how)
            _assert_raises(RoundError, lambda: read_round(root, 1, run_id=RUN_ID),
                           message)


def test_sampling_failures_block_publication_entirely():
    """No marker, no data, no metadata: a round that could not supply its ANN
    negatives is discarded rather than published with substitutes."""
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        _assert_raises(RoundError, lambda: _publish(root, 1, failures=1),
                       "never fabricates a negative")
        assert latest_committed_round(root) == 0
        assert not round_paths(root, 1)['ready'].exists()
        assert not round_paths(root, 1)['training_data'].exists()


def test_initial_round_is_not_counted_as_a_numeric_round():
    """`ready_initial` is the step-0 input, not a refresh."""
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        _publish(root, INITIAL_ROUND, step=0)
        assert latest_committed_round(root) == 0
        data_dir, meta = read_round(root, INITIAL_ROUND, run_id=RUN_ID)
        assert meta['checkpoint_step'] == 0
        assert data_dir.name == "training_data_initial"


def test_latest_committed_round_tracks_the_highest_marker():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        _publish(root, INITIAL_ROUND, step=0)
        assert latest_committed_round(root) == 0
        for n in (1, 2, 3):
            _publish(root, n, step=1000 * n)
            assert latest_committed_round(root) == n


# ---- the shared encode+mine sequence ----------------------------------------

def test_both_callers_mine_through_one_shared_sequence():
    """The orchestrator's base-model round and the Inferencer's refresh rounds run the
    SAME code, so they cannot drift apart on encode order, the encode timeout, the
    corpus-id assertion, or the mining arguments.

    Driven with a stub encoder so it stays CPU-only: what is asserted is the sequence,
    not the embeddings.
    """
    import pickle
    import numpy as np
    import ance_mining
    import utils.helpers as helpers

    calls = []

    def fake_encode(model, in_file, out_pkl, is_query, ctx, config, timeout=None):
        calls.append((str(model), Path(in_file).name, bool(is_query), timeout))
        n = 2 if is_query else 3
        ids = [f'q{i}' for i in range(n)] if is_query else [f'd{i}' for i in range(n)]
        vecs = np.eye(n, 8, dtype=np.float32)
        with open(out_pkl, 'wb') as f:
            pickle.dump((vecs, ids), f)

    real = helpers.encode_to_pickle
    helpers.encode_to_pickle = fake_encode
    try:
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            (tmp / "corpus.jsonl").write_text("")
            (tmp / "queries.jsonl").write_text("")
            mixture = tmp / "mix.jsonl"
            mixture.write_text("".join(json.dumps({
                'query_id': f'q{i}', 'query': f'query {i}',
                'positive_passages': [{'docid': 'd0', 'text': 'pos'}],
                'negative_passages': []}) + "\n" for i in range(2)))

            ctx = {'args': {'max_encode_seconds': 4242, 'train_group_size': 2,
                            'mining_depth': 200, 'ann_chunk_factor': 1}}
            def common():
                # A FRESH rng per call: the two rounds are only comparable if the
                # selection starts from the same state, which is also why the miner
                # is seeded rather than left to global random.
                return dict(corpus_file=tmp / "corpus.jsonl",
                            query_file=tmp / "queries.jsonl",
                            mixture_files=[mixture],
                            qrels_dict={'q0': {'d0'}, 'q1': {'d0'}},
                            ctx=ctx, config={}, rng=random.Random(0))

            _, ids_a, mined_a, fails_a, _, shard_a = ance_mining.encode_and_mine(
                "/models/base", tmp / "s0", ann_no=0, **common())
            labels = []
            _, ids_b, mined_b, fails_b, _, shard_b = ance_mining.encode_and_mine(
                "/models/checkpoint-1000", tmp / "s1", ann_no=1,
                phase=labels.append, **common())

    finally:
        helpers.encode_to_pickle = real

    # Same sequence, same timeout, corpus before queries, on both paths.
    assert [c[1:] for c in calls[:2]] == [c[1:] for c in calls[2:]], calls
    assert [c[1] for c in calls[:2]] == ['corpus.jsonl', 'queries.jsonl'], calls
    assert all(c[3] == 4242 for c in calls), "encode timeout not applied on both paths"
    assert calls[0][0] == "/models/base" and calls[2][0] == "/models/checkpoint-1000"

    # Same outputs, and the phase callback is telemetry only -- it changes nothing.
    assert ids_a == ids_b, (ids_a, ids_b)
    assert mined_a == mined_b, (mined_a, mined_b)
    assert not fails_a and not fails_b
    assert shard_a['n_shards'] == shard_b['n_shards'] == 1
    assert labels == ['encode_corpus', 'encode_query', 'index', 'search'], labels


def test_the_shared_sequence_refuses_a_corpus_with_repeated_ids():
    """assert_corpus_ids_unique runs inside the shared path, so neither caller can
    skip it. Duplicate ids mean the same passage can be mined as several distinct
    negatives, and nothing downstream would notice."""
    import pickle
    import numpy as np
    import ance_mining
    import utils.helpers as helpers

    def dup_encode(model, in_file, out_pkl, is_query, ctx, config, timeout=None):
        ids = ['q0'] if is_query else ['d0', 'd0']
        vecs = np.eye(len(ids), 8, dtype=np.float32)
        with open(out_pkl, 'wb') as f:
            pickle.dump((vecs, ids), f)

    real = helpers.encode_to_pickle
    helpers.encode_to_pickle = dup_encode
    try:
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            (tmp / "c.jsonl").write_text("")
            (tmp / "q.jsonl").write_text("")
            _assert_raises(RoundError, lambda: ance_mining.encode_and_mine(
                "/m", tmp / "s", corpus_file=tmp / "c.jsonl",
                query_file=tmp / "q.jsonl", mixture_files=[], qrels_dict={},
                ctx={'args': {'max_encode_seconds': 1, 'train_group_size': 2,
                              'mining_depth': 200, 'ann_chunk_factor': 1}},
                config={}, rng=random.Random(0), ann_no=0), "d0")
    finally:
        helpers.encode_to_pickle = real


# ---- input preflight --------------------------------------------------------

def _preflight_fixture(tmp, *, mixture_query="the query", file_query="the query",
                       positives=("d0",), corpus=("d0",), qrels=("d0",), texts=None):
    """One mixture record and the three artifacts it must agree with.

    `texts` overrides the default `text <docid>` body for a docid, on BOTH sides. It
    is what lets a test express preprocessor._derive's duplicate-text collapse: a
    mixture positive whose docid is absent from the corpus while its TEXT is present
    there under the canonical owner's id.
    """
    texts = texts or {}
    body = lambda d: texts.get(d, f'text {d}')
    tmp = Path(tmp)
    (tmp / "mix.jsonl").write_text(json.dumps({
        'query_id': 'q0', 'query': mixture_query,
        'positive_passages': [{'docid': d, 'text': body(d)} for d in positives],
        'negative_passages': []}) + "\n")
    (tmp / "queries.jsonl").write_text(
        json.dumps({'query_id': 'q0', 'query': file_query}) + "\n")
    (tmp / "qrels.txt").write_text("".join(f"q0 Q0 {d} 1\n" for d in qrels))
    return ([tmp / "mix.jsonl"], tmp / "queries.jsonl", tmp / "qrels.txt",
            {d: body(d) for d in corpus}, {'q0': set(qrels)})


def test_preflight_accepts_consistent_inputs():
    from train_ance import preflight_inputs
    with tempfile.TemporaryDirectory() as tmp:
        assert preflight_inputs(*_preflight_fixture(tmp)) == 1


def test_preflight_rejects_a_query_whose_text_disagrees():
    """The silent one. The miner encodes the QUERY FILE and labels the result with the
    mixture's qid, so a disagreement mines negatives for a different question than the
    one trained on -- and still publishes a complete, valid-looking round."""
    from train_ance import preflight_inputs
    with tempfile.TemporaryDirectory() as tmp:
        args = _preflight_fixture(tmp, mixture_query="what is a cell",
                                  file_query="what is a stack frame")
        _assert_raises(RuntimeError, lambda: preflight_inputs(*args),
                       "text differs from")


def test_preflight_rejects_a_query_with_no_resolvable_positive():
    """Every labelled positive absent from the corpus: the round cannot be mined,
    because select_ance_negatives has no positive to exclude from the candidates."""
    from train_ance import preflight_inputs
    with tempfile.TemporaryDirectory() as tmp:
        args = _preflight_fixture(tmp, positives=("d9",), qrels=("d9",),
                                  corpus=("d0",))
        _assert_raises(RuntimeError, lambda: preflight_inputs(*args),
                       "every labelled positive")


def test_preflight_accepts_a_positive_resolvable_only_through_the_qrels():
    """record_positives unions both sources, so either one resolving is enough."""
    from train_ance import preflight_inputs
    with tempfile.TemporaryDirectory() as tmp:
        args = _preflight_fixture(tmp, positives=(), qrels=("d0",), corpus=("d0",))
        assert preflight_inputs(*args) == 1


# preprocessor._derive remaps the CORPUS and the QRELS but leaves the mixture holding
# raw docids, so a legitimate positive can name an id the corpus does not carry. Job
# 59904 died on 811 of these. The same defect is handled for the async arm by
# async_fast_grass_cached_mcdp.canonicalize_positives.

def test_preflight_resolves_a_whitespace_escaped_positive():
    """_trec_safe_docid percent-encodes an id whose whitespace would break a TREC
    column. The mixture keeps the raw id; the corpus and qrels carry the escaped one."""
    from train_ance import preflight_inputs
    from data.preprocessor import _trec_safe_docid
    raw = "pytorch/Memory Management_4_0.txt"
    escaped = _trec_safe_docid(raw)
    assert escaped != raw and escaped.startswith("trec:")
    with tempfile.TemporaryDirectory() as tmp:
        args = _preflight_fixture(tmp, positives=(raw,), corpus=(escaped,),
                                  qrels=(escaped,), texts={raw: "shared body",
                                                           escaped: "shared body"})
        assert preflight_inputs(*args) == 1


def test_preflight_resolves_a_duplicate_text_positive_through_the_qrels():
    """_derive emits one canonical owner per text and remaps the qrels onto it."""
    from train_ance import preflight_inputs
    with tempfile.TemporaryDirectory() as tmp:
        args = _preflight_fixture(tmp, positives=("dup",), corpus=("d0",),
                                  qrels=("d0",), texts={"dup": "shared body",
                                                        "d0": "shared body"})
        assert preflight_inputs(*args) == 1


def test_preflight_refuses_a_duplicate_whose_owner_the_qrels_do_not_carry():
    """The drift case a bare corpus-membership test could not distinguish: the text
    IS in the corpus, but under a docid this query is not judged against, so the
    exclusion set would miss it and the true positive could be mined as a negative."""
    from train_ance import preflight_inputs
    with tempfile.TemporaryDirectory() as tmp:
        args = _preflight_fixture(tmp, positives=("dup",), corpus=("d0", "d1"),
                                  qrels=("d1",), texts={"dup": "shared body",
                                                        "d0": "shared body"})
        _assert_raises(RuntimeError, lambda: preflight_inputs(*args),
                       "which that query's qrels do not carry")


def test_preflight_refuses_a_positive_whose_text_is_nowhere_in_the_corpus():
    """Genuine staleness: neither remapping explains it, and the query stays
    otherwise resolvable, so `unresolved` alone would let it through."""
    from train_ance import preflight_inputs
    with tempfile.TemporaryDirectory() as tmp:
        args = _preflight_fixture(tmp, positives=("ghost", "d0"), corpus=("d0",),
                                  qrels=("d0",), texts={"ghost": "body nobody has"})
        msg = _assert_raises(RuntimeError, lambda: preflight_inputs(*args),
                             "absent from the corpus under any docid")
        assert "ghost" in msg and "q0" in msg


def test_preflight_builds_the_reverse_map_only_when_a_positive_misses():
    """The map is O(corpus): 8.8M md5s on MS MARCO. It must not be built to answer
    no questions."""
    import train_ance
    real, calls = train_ance._canonical_docids, []

    def counting(corpus_lookup, wanted):
        calls.append(set(wanted))
        return real(corpus_lookup, wanted)

    train_ance._canonical_docids = counting
    try:
        with tempfile.TemporaryDirectory() as tmp:
            assert train_ance.preflight_inputs(*_preflight_fixture(tmp)) == 1
            assert calls == [], f"reverse map built with no misses: {calls}"
            args = _preflight_fixture(tmp, positives=("dup",), corpus=("d0",),
                                      qrels=("d0",), texts={"dup": "shared body",
                                                            "d0": "shared body"})
            assert train_ance.preflight_inputs(*args) == 1
            assert len(calls) == 1 and len(calls[0]) == 1, calls
    finally:
        train_ance._canonical_docids = real


def test_preflight_flag_touches_no_gpu_and_writes_nothing():
    """--preflight exists so this class of failure costs minutes, not a 24h GPU
    allocation (job 59904 died in preflight_inputs after SLURM had handed it 2 A100s).

    It must also be safe to run while a REAL run holds the same output dir, which is
    the whole reason the return sits above prepare_output_dir and create_work_root.
    """
    import train_ance
    tripwires = ('require_ance_gpus', 'prepare_output_dir', 'create_work_root',
                 'mine_initial_round', 'build_run_manifest')
    saved = {n: getattr(train_ance, n) for n in tripwires}
    saved.update({n: getattr(train_ance, n) for n in
                  ('load_config', 'set_seed', 'get_training_context',
                   'require_recipe_keys', 'log_startup_config', 'run_setup',
                   'get_path', 'require_mixture_files', '_load_corpus_lookup',
                   '_load_qrels', 'calculate_training_budget')})
    argv = sys.argv

    def forbidden(name):
        def boom(*a, **k):
            raise AssertionError(f"--preflight reached {name}")
        return boom

    with tempfile.TemporaryDirectory() as tmp:
        mix, queries, qrels, corpus_lookup, qrels_dict = _preflight_fixture(tmp)
        recipe = {'mixture_dir': 'training_mixture', 'setup_mode': 'reasonir_mixture',
                  'batch_size': 64, 'model_name': 'm', 'temp_workdir': 'temp_ance'}
        try:
            for name in tripwires:
                setattr(train_ance, name, forbidden(name))
            train_ance.load_config = lambda: {'seed': 42}
            train_ance.set_seed = lambda *a, **k: None
            train_ance.get_training_context = lambda r: {'args': recipe,
                                                         'base_model': 'bge'}
            train_ance.require_recipe_keys = lambda *a, **k: None
            train_ance.log_startup_config = lambda *a, **k: None
            train_ance.run_setup = lambda r: (Path(tmp) / "c.jsonl", queries, qrels)
            train_ance.get_path = lambda k: Path(tmp)
            train_ance.require_mixture_files = lambda d, e: [mix[0]]
            train_ance._load_corpus_lookup = lambda f: corpus_lookup
            train_ance._load_qrels = lambda f: qrels_dict
            train_ance.calculate_training_budget = lambda n, r: {
                'steps_per_epoch': 1, 'max_steps': 1, 'scheduler_max_steps': 1,
                'triplets_per_query': 1, 'training_instances': 1,
                'queries_per_round': 1, 'ann_chunk_factor': 1}

            sys.argv = ['train_ance.py', '--preflight']
            assert train_ance.main() == 0
            assert not (Path(tmp) / "temp_ance").exists()
        finally:
            sys.argv = argv
            for name, fn in saved.items():
                setattr(train_ance, name, fn)


# ---- freshness gate ---------------------------------------------------------

def _summary(*rounds):
    return {'run_id': RUN_ID, 'rounds': list(rounds)}


def _round(ann_no, step, consumed):
    return {'ann_no': ann_no, 'checkpoint': f'ck-{step}',
            'checkpoint_step': step, 'consumed_steps': consumed}


def test_initial_round_alone_does_not_satisfy_the_gate():
    """The exact shape of a run whose inferencer died at startup."""
    _assert_raises(
        RoundError,
        lambda: assert_ance_refresh(_summary(_round(INITIAL_ROUND, 0, 10312))),
        "is not ANCE")


def test_no_rounds_at_all_does_not_satisfy_the_gate():
    _assert_raises(RoundError, lambda: assert_ance_refresh(_summary()), "none")


def test_a_published_but_never_consumed_round_does_not_count():
    """Mined at step 10000 of 10312 and swapped in with no steps left to train on
    it: the negatives never influenced a single update."""
    _assert_raises(
        RoundError,
        lambda: assert_ance_refresh(_summary(_round(INITIAL_ROUND, 0, 10000),
                                             _round(1, 1000, 0))),
        "0 qualified")


def test_one_consumed_checkpoint_round_satisfies_the_gate():
    """The mechanism, at min_fresh_rounds=1. The DEFAULT is 2 -- see below."""
    fresh = assert_ance_refresh(_summary(_round(INITIAL_ROUND, 0, 900),
                                         _round(1, 1000, 412)), min_fresh_rounds=1)
    assert [r['ann_no'] for r in fresh] == [1]


def test_two_distinct_fresh_rounds_are_required_by_default():
    """One refresh followed by static training is the failure the default exists for.

    A run that refreshes once near the start and then trains tens of thousands of
    steps on that one round is hard-negative training with an ANCE-shaped log.
    """
    one = _summary(_round(INITIAL_ROUND, 0, 900), _round(1, 1000, 9000))
    _assert_raises(RoundError, lambda: assert_ance_refresh(one), "at least 2")
    two = _summary(_round(INITIAL_ROUND, 0, 900), _round(1, 1000, 900),
                   _round(2, 2000, 900))
    assert len(assert_ance_refresh(two)) == 2


def test_two_rounds_from_one_checkpoint_are_one_refresh():
    """Publishing twice from the same checkpoint is one iteration of the ANCE loop.

    The miner numbers rounds monotonically, so two publications carry different
    ann_no values while describing the same model state; counting both would let a
    single refresh satisfy a two-refresh gate.
    """
    summary = _summary(_round(INITIAL_ROUND, 0, 900),
                       _round(1, 1000, 500), _round(2, 1000, 500))
    _assert_raises(RoundError, lambda: assert_ance_refresh(summary), "DISTINCT")
    assert len(assert_ance_refresh(summary, min_fresh_rounds=1)) == 1


def test_min_consume_steps_is_enforced():
    """A round replaced after a handful of steps trained nothing on it.

    train_ance passes the recipe's logging_steps here, so "consumed" means at least
    one full logging interval rather than one optimizer step.
    """
    summary = _summary(_round(INITIAL_ROUND, 0, 900), _round(1, 1000, 5),
                       _round(2, 2000, 5))
    assert_ance_refresh(summary, min_consume_steps=5)
    _assert_raises(RoundError,
                   lambda: assert_ance_refresh(summary, min_consume_steps=6))


def test_min_fresh_rounds_is_enforced():
    summary = _summary(_round(1, 1000, 100), _round(2, 2000, 100))
    assert len(assert_ance_refresh(summary, min_fresh_rounds=2)) == 2
    _assert_raises(RoundError,
                   lambda: assert_ance_refresh(summary, min_fresh_rounds=3))


# ---- encode timeout: the stall detector -------------------------------------

def test_a_stalled_encode_raises_rather_than_hanging_forever():
    """The one thing that catches a live-but-hung inferencer.

    A heartbeat cannot: `encode_to_pickle` shells out to Tevatron and blocks for hours
    on a real corpus, so a background writer keeps ticking straight through a
    deadlocked subprocess and reports it healthy. `subprocess.run(timeout=...)` is
    what actually fires, and the resulting nonzero exit is what the existing
    supervision already turns into a failed run.
    """
    from utils.helpers import encode_to_pickle

    ctx = {'args': {'per_device_eval_batch_size': 8, 'dataloader_num_workers': 0},
           'max_q': 32, 'max_p': 32, 'pooling': 'cls', 'normalize': True}
    config = {'model': {'query_max_len': 32, 'passage_max_len': 32}}
    calls = []

    def fake_run(cmd, **kwargs):
        calls.append(kwargs.get('timeout'))
        if kwargs.get('timeout') is not None:
            raise subprocess.TimeoutExpired(cmd, kwargs['timeout'])
        return subprocess.CompletedProcess(cmd, 0)

    import utils.helpers as helpers
    original = helpers.subprocess.run
    helpers.subprocess.run = fake_run
    try:
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "corpus.pkl"
            # No timeout configured -> the call is made and does not raise.
            encode_to_pickle("m", "in.jsonl", out, False, ctx, config)
            assert calls == [None], calls
            # A timeout is passed through and its expiry propagates.
            _assert_raises(
                subprocess.TimeoutExpired,
                lambda: encode_to_pickle("m", "in.jsonl", out, False, ctx, config,
                                         timeout=1))
            assert calls[-1] == 1, calls
    finally:
        helpers.subprocess.run = original


def test_the_paper_encoder_uses_the_hard_subprocess_timeout():
    from utils.helpers import encode_to_pickle
    import utils.helpers as helpers

    ctx = {'args': {'paper_fidelity': True, 'per_device_eval_batch_size': 8},
           'max_q': 64, 'max_p': 512}
    seen = {}

    def fake_run(cmd, **kwargs):
        seen['cmd'], seen['timeout'] = cmd, kwargs.get('timeout')
        raise subprocess.TimeoutExpired(cmd, kwargs['timeout'])

    original = helpers.subprocess.run
    helpers.subprocess.run = fake_run
    try:
        _assert_raises(
            subprocess.TimeoutExpired,
            lambda: encode_to_pickle("model", "queries.jsonl", "queries.pkl", True,
                                     ctx, {}, timeout=17))
    finally:
        helpers.subprocess.run = original
    assert seen['timeout'] == 17
    assert any(str(part).endswith('scripts/ance_paper.py') for part in seen['cmd'])
    assert '--is_query' in seen['cmd']


def test_gpu_smoke_is_conservative_and_fails_closed():
    from dev.gpu_memory_smoke import _varied, fail_if_any
    assert len(_varied(4, median=10, cap=32)[0].split()) == 32
    assert fail_if_any([]) is None
    msg = _assert_raises(SystemExit,
                         lambda: fail_if_any([('ance_train', 'RuntimeError', 'OOM')]))
    assert msg == '1', msg


# ---- inferencer supervision -------------------------------------------------

class _Proc:
    """Minimal Popen stand-in.

    `rc` is what poll() reports once the process is no longer alive. `exit_after`
    is how many timed-out waits it survives before exiting, which is what lets the
    supervision loop terminate in a test.
    """

    def __init__(self, rc=None, alive=True, exit_after=1):
        self._rc, self.alive, self._exit_after = rc, alive, exit_after
        self.returncode = None if alive else rc
        self.terminated = self.killed = False
        self.waits = 0

    def poll(self):
        return None if self.alive else self._rc

    def wait(self, timeout=None):
        self.waits += 1
        if self.alive and timeout is not None and self.waits <= self._exit_after:
            import subprocess
            raise subprocess.TimeoutExpired("proc", timeout)
        self.alive = False
        self.returncode = self._rc
        return self._rc

    def terminate(self):
        self.terminated = True
        self.alive = False
        self.returncode = self._rc

    def kill(self):
        self.killed = True


def _supervise(*a, **kw):
    from train_ance import supervise
    kw.setdefault('log', lambda *_: None)
    return supervise(*a, poll_seconds=0.0, **kw)


def test_a_dead_inferencer_kills_the_trainer_and_fails_the_run():
    trainer = _Proc(rc=0, alive=True)
    inferencer = _Proc(rc=1, alive=False)
    failed, train_rc = _supervise(trainer, inferencer)
    assert failed == 1, failed
    assert trainer.terminated, "the trainer was left running on stale negatives"


def test_an_inferencer_that_dies_with_the_trainer_is_still_caught():
    """The window the in-loop check alone would miss."""
    trainer = _Proc(rc=0, alive=True)
    inferencer = _Proc(rc=137, alive=True, exit_after=99)

    original = trainer.wait

    def wait(timeout=None):
        inferencer.alive = False           # dies in the same window
        inferencer.returncode = 137
        return original(timeout=None)

    trainer.wait = wait
    failed, _ = _supervise(trainer, inferencer)
    assert failed == 137, failed


def test_a_healthy_run_reports_no_failure():
    trainer = _Proc(rc=0, alive=True, exit_after=2)
    inferencer = _Proc(rc=None, alive=True, exit_after=99)
    failed, train_rc = _supervise(trainer, inferencer)
    assert failed is None, failed
    assert train_rc == 0
    assert inferencer.terminated, "the inferencer must be stopped when training ends"


def test_a_clean_early_inferencer_exit_is_a_failure():
    """INVERTED. `run_ance_data_gen.main()` loops until terminated and has no
    --max_rounds equivalent, so there is no path on which it finishes early and
    legitimately. rc 0 before the trainer is done means refreshes stopped."""
    trainer = _Proc(rc=0, alive=True)
    inferencer = _Proc(rc=0, alive=False)
    failed, _ = _supervise(trainer, inferencer)
    assert failed == 0, failed          # 0 is a failure code here, not "no failure"
    assert failed is not None, "rc 0 must not be read as truthiness"
    assert trainer.terminated, "the trainer was left running with no miner"


def test_our_own_termination_is_not_an_early_exit():
    """After the trainer finishes we terminate the inferencer ourselves. That is a
    normal shutdown and must not be reported as a failed run."""
    trainer = _Proc(rc=0, alive=True, exit_after=1)
    inferencer = _Proc(rc=-15, alive=True, exit_after=99)
    failed, train_rc = _supervise(trainer, inferencer)
    assert failed is None, failed
    assert train_rc == 0
    assert inferencer.terminated


# ---- non-finite loss --------------------------------------------------------

def test_non_finite_loss_is_rejected_before_backward():
    """A NaN forward must not reach backward(). Proved by state: `backward` is
    replaced with a tripwire, so reaching it fails the test rather than passing it."""
    import torch as t
    from run_ance_train import NonFiniteOptimization, optimization_step
    model, optimizer, scheduler, before = _grad_fixture(0.01)
    for bad in (float('nan'), float('inf'), float('-inf')):
        loss = t.tensor(bad, requires_grad=True)
        loss.backward = lambda: (_ for _ in ()).throw(
            AssertionError("backward() reached with a non-finite loss"))
        _assert_raises(NonFiniteOptimization,
                       lambda l=loss: optimization_step(
                           model, optimizer, scheduler, l,
                           max_grad_norm=1.0, step=10),
                       "diverged")
    for prm, was in zip(model.parameters(), before):
        assert t.equal(prm.detach(), was), "a parameter moved despite the raise"


# ---- success validation -----------------------------------------------------

_RECIPE = {'model_name': 'm', 'batch_size': 4, 'train_group_size': 2,
           'learning_rate': 1e-5, 'total_epochs': 2}
_CTX = {'args': _RECIPE, 'base_model': '/nonexistent/base'}


def _mixture(tmp):
    path = Path(tmp) / "train_hq.jsonl"
    with open(path, 'w') as f:
        for rec in _records(5):
            f.write(json.dumps(rec) + '\n')
    return path


def _manifest(tmp, steps=10):
    return build_run_manifest('ance', _CTX, _RECIPE, data_files=[_mixture(tmp)],
                              world_size=1, negative_pool_size=7,
                              optimizer_steps=steps)


def _safetensors(path):
    header = {"t0": {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]}}
    raw = json.dumps(header).encode()
    path.write_bytes(len(raw).to_bytes(8, 'little') + raw + b"\x00" * 4)


def _trained_output(tmp, manifest, *, logged, final, planned=10):
    """An output dir shaped like the ANCE trainer's own writes.

    `logged` are the steps at which the loop's logging_steps branch fired; `final`
    is the terminal record. There is no trainer_state.json: this is not an HF
    Trainer, so the diagnostics log is the ONLY witness.
    """
    out = Path(tmp) / "out"
    prepare_output_dir(out, manifest)
    log = out / TRAINING_LOG_NAME
    append_jsonl(log, {"global_step": 0, "phase": "begin",
                       "rank_acc": 0.5, "margin_mean": 0.0})
    for step in logged:
        append_jsonl(log, {"global_step": step, "loss": 0.5,
                           "learning_rate": 1e-5, "grad_norm": 1.2})
    if final is not None:
        append_jsonl(log, {"global_step": final, "loss": 0.4, "learning_rate": 1e-5,
                           "grad_norm": 1.1, "terminal": True})
    append_jsonl(log, {"global_step": final or (logged[-1] if logged else 0),
                       "phase": "end", "rank_acc": 0.9, "margin_mean": 0.3})
    (out / "config.json").write_text(json.dumps({"model_type": "xlm-roberta"}))
    _safetensors(out / "model.safetensors")
    return out


def test_a_complete_run_validates_via_the_terminal_record():
    """max_steps is not a multiple of logging_steps. Without the terminal record the
    log's last step is 8 against a planned 10 and a COMPLETE run is rejected."""
    with tempfile.TemporaryDirectory() as tmp:
        m = _manifest(tmp, steps=10)
        out = _trained_output(tmp, m, logged=[4, 8], final=10)
        stored = assert_training_succeeded(out, m)
        assert stored['final_global_step'] == 10


def test_without_the_terminal_record_a_complete_run_is_rejected():
    with tempfile.TemporaryDirectory() as tmp:
        m = _manifest(tmp, steps=10)
        out = _trained_output(tmp, m, logged=[4, 8], final=None)
        _assert_raises(RunDirectoryError,
                       lambda: assert_training_succeeded(out, m),
                       "stopped at step 8 of the 10 planned")


def test_a_zero_step_run_is_rejected():
    with tempfile.TemporaryDirectory() as tmp:
        m = _manifest(tmp, steps=10)
        out = _trained_output(tmp, m, logged=[], final=None)
        _assert_raises(RunDirectoryError,
                       lambda: assert_training_succeeded(out, m),
                       "no new optimizer steps")


def test_a_non_finite_loss_in_the_log_is_rejected():
    with tempfile.TemporaryDirectory() as tmp:
        m = _manifest(tmp, steps=10)
        out = _trained_output(tmp, m, logged=[4, 8], final=10)
        append_jsonl(out / TRAINING_LOG_NAME,
                     {"global_step": 6, "loss": float('nan'), "grad_norm": 1.0})
        _assert_raises(RunDirectoryError,
                       lambda: assert_training_succeeded(out, m), "non-finite loss")


def test_manifest_records_the_run_identity_for_the_evaluator():
    with tempfile.TemporaryDirectory() as tmp:
        m = _manifest(tmp, steps=10)
        out = _trained_output(tmp, m, logged=[4, 8], final=10)
        assert_training_succeeded(out, m)
        stored = json.loads((out / RUN_MANIFEST_NAME).read_text())
        assert stored['finished_at'] and stored['fingerprint']
        assert stored['optimizer_steps_planned'] == 10


# ---- non-finite gradients ---------------------------------------------------

def _tiny_model():
    """A real 2-layer encoder wrapped in the real DenseModel, on CPU."""
    from transformers import XLMRobertaConfig, XLMRobertaModel
    from tevatron.retriever.modeling import DenseModel
    cfg = XLMRobertaConfig(vocab_size=64, hidden_size=32, num_hidden_layers=2,
                           num_attention_heads=2, intermediate_size=64,
                           max_position_embeddings=12)
    import torch as t
    t.manual_seed(0)
    return DenseModel(encoder=XLMRobertaModel(cfg), pooling='cls', normalize=True,
                      temperature=0.02)


def _grad_fixture(value):
    """A model whose every gradient is `value`, plus a real AdamW and scheduler."""
    import torch as t
    from transformers import get_linear_schedule_with_warmup
    from utils.helpers import build_adamw
    model = _tiny_model()
    optimizer, _ = build_adamw(model.parameters(), lr=1e-3, weight_decay=0.0,
                               label='test')
    scheduler = get_linear_schedule_with_warmup(optimizer, 0, 10)
    for prm in model.parameters():
        prm.grad = t.full_like(prm, value)
    before = [prm.detach().clone() for prm in model.parameters()]
    return model, optimizer, scheduler, before


def _loss_from(model):
    """A finite scalar whose backward() ADDS zero, leaving `_grad_fixture`'s seeded
    gradients in place. The point is to drive the real code path, not to recompute
    the gradients the fixture deliberately poisoned."""
    import torch as t
    return sum((prm * 0).sum() for prm in model.parameters()) + t.tensor(0.5)


def test_no_step_is_taken_on_non_finite_gradients():
    """The ordering claim, proved by state: parameters must be untouched after the
    raise. clip_grad_norm_ with a non-finite total norm yields a non-finite
    coefficient, so a step here would write NaN into every parameter."""
    import torch as t
    from run_ance_train import NonFiniteOptimization, optimization_step
    for bad in (float('inf'), float('nan')):
        model, optimizer, scheduler, before = _grad_fixture(bad)
        # A FINITE loss with non-finite gradients: the two checks are independent,
        # which is why the loss guard alone is not enough.
        loss = _loss_from(model)
        _assert_raises(NonFiniteOptimization,
                       lambda l=loss: optimization_step(
                           model, optimizer, scheduler, l,
                           max_grad_norm=1.0, step=7),
                       "step 7")
        for prm, was in zip(model.parameters(), before):
            assert t.equal(prm.detach(), was), "a parameter moved despite the raise"
            assert t.isfinite(prm.detach()).all(), "NaN reached the weights"


def test_finite_gradients_do_step():
    import torch as t
    from run_ance_train import optimization_step
    model, optimizer, scheduler, before = _grad_fixture(0.01)
    value, norm = optimization_step(model, optimizer, scheduler, _loss_from(model),
                                    max_grad_norm=1.0, step=1)
    assert t.isfinite(t.tensor(value)) and norm > 0
    assert t.isfinite(t.tensor(norm))
    moved = sum(not t.equal(prm.detach(), was)
                for prm, was in zip(model.parameters(), before))
    assert moved > 0, "no parameter moved on a valid step"
    assert all(prm.grad is None for prm in model.parameters()), "grads not zeroed"


# ---- the trainer summary is critical, not best-effort -----------------------

def test_summary_write_is_retried_then_raises():
    """`ance_trainer_summary.json` is the only evidence of round consumption, and
    train_ance.py fails the run when it is missing. A lost write must raise, not
    leave the run unvalidatable."""
    import utils.helpers as helpers
    calls = {'n': 0}

    def always_fails():
        calls['n'] += 1
        raise OSError("EREMOTEIO")

    assert helpers.retry_io(always_fails, "write summary", attempts=3, delay=0) is False
    assert calls['n'] == 3, calls['n']


def test_the_summary_separates_opportunities_publications_and_consumptions():
    """Every refresh state has its own durable field, all DERIVED from the final step
    and the consumed-round list rather than from parallel counters in the loop.

    Reporting a checkpoint opportunity as a refresh is how a static run came to be
    described as ANCE, so these three must never collapse into one number.
    """
    from run_ance_train import build_trainer_summary
    # Consumed the initial round and rounds 1 and 3; round 2 was superseded before
    # the trainer's next poll, and round 4 was published after the last swap.
    rounds = [_round(INITIAL_ROUND, 0, 100), _round(1, 1000, 200),
              _round(3, 3000, 200)]
    summary = build_trainer_summary(
        run_id=RUN_ID, work_root='/tmp/work', optimizer={'name': 'AdamW'},
        max_steps=3000, scheduler_max_steps=3000, save_steps=1000,
        final_step=3000, rounds=rounds, rounds_completed=4)
    assert summary['checkpoint_opportunities'] == 3      # 1000, 2000, 3000
    assert summary['rounds_completed'] == 4              # the miner published 4
    assert summary['rounds_consumed'] == 2               # initial is not a refresh
    assert summary['rounds_skipped'] == 1                # round 2
    assert summary['terminal_unconsumed'] == 4


def test_a_final_step_off_the_save_boundary_still_counts_its_checkpoint():
    """max_steps always saves, even when it is not a save_steps multiple. Counting
    only the boundaries would under-report the last checkpoint."""
    from run_ance_train import build_trainer_summary
    summary = build_trainer_summary(
        run_id=RUN_ID, work_root='/tmp/work', optimizer={'name': 'AdamW'},
        max_steps=2500, scheduler_max_steps=2500, save_steps=1000,
        final_step=2500, rounds=[_round(INITIAL_ROUND, 0, 100)], rounds_completed=0)
    assert summary['checkpoint_opportunities'] == 3      # 1000, 2000, and 2500
    assert summary['rounds_consumed'] == 0
    assert summary['terminal_unconsumed'] is None


def test_a_terminal_round_published_after_the_last_swap_is_not_consumed():
    """The miner normally publishes one more round than the trainer can use.

    The final checkpoint triggers a mine that finishes after training ends. Recording
    it as an unconsumed publication is honest; counting it as a refresh would inflate
    every run by exactly one.
    """
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        _publish(root, 1, step=1000)
        _publish(root, 2, step=2000)
        published = latest_committed_round(root)
        assert published == 2
        last_ann_no = 1                      # the trainer never reached round 2
        assert (published if published > last_ann_no else None) == 2
        # ...and it is absent from the consumed rounds, so the gate cannot see it.
        summary = _summary(_round(INITIAL_ROUND, 0, 900), _round(1, 1000, 900))
        _assert_raises(RoundError, lambda: assert_ance_refresh(summary), "at least 2")


def test_summary_write_survives_a_transient_failure():
    import utils.helpers as helpers
    calls = {'n': 0}

    def fails_twice():
        calls['n'] += 1
        if calls['n'] < 3:
            raise OSError("EREMOTEIO")

    assert helpers.retry_io(fails_twice, "write summary", attempts=5, delay=0) is True
    assert calls['n'] == 3


def test_trainer_treats_a_failed_summary_write_as_fatal():
    """A write that cannot be completed must raise, not leave the run unvalidatable.

    Driven through the real function with a filesystem that refuses the write, rather
    than asserted by reading the source.
    """
    from run_ance_train import write_trainer_summary
    with tempfile.TemporaryDirectory() as tmp:
        # A directory where the summary file must go: every attempt raises OSError.
        blocked = Path(tmp) / "ance_trainer_summary.json"
        blocked.mkdir()
        _assert_raises(OSError,
                       lambda: write_trainer_summary(blocked, {'run_id': RUN_ID}),
                       "only")


def test_a_successful_summary_write_lands_the_payload():
    from run_ance_train import write_trainer_summary
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "ance_trainer_summary.json"
        write_trainer_summary(path, {'run_id': RUN_ID, 'rounds_consumed': 2})
        assert json.loads(path.read_text())['rounds_consumed'] == 2


# ---- run id -----------------------------------------------------------------

def _fake_manifest(fp="abcdef0123456789", epoch=1700000000.0):
    return {'fingerprint': fp, 'started_at_epoch': epoch}


def test_run_ids_are_unique_within_one_second():
    """A SLURM array launches its tasks in the same second with the same recipe, so
    the fingerprint and the timestamp are both identical across them."""
    from train_ance import build_run_id
    m = _fake_manifest()
    ids = {build_run_id(m) for _ in range(200)}
    assert len(ids) == 200, f"{200 - len(ids)} collision(s)"


def test_run_id_carries_the_fingerprint_and_job_id():
    from train_ance import build_run_id
    os.environ['SLURM_JOB_ID'] = '1234567'
    try:
        rid = build_run_id(_fake_manifest())
    finally:
        del os.environ['SLURM_JOB_ID']
    assert rid.startswith('abcdef012345'), rid
    assert 'j1234567' in rid, rid


def test_run_id_omits_the_job_id_outside_slurm():
    from train_ance import build_run_id
    os.environ.pop('SLURM_JOB_ID', None)
    assert 'j' not in build_run_id(_fake_manifest()).split('-')[-1]


def test_work_root_creation_refuses_an_existing_directory():
    """A collision must be a startup error, not two runs silently sharing a work root
    in which each is the other's "foreign run"."""
    from train_ance import create_work_root
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp) / "runs" / "abc"
        assert create_work_root(root) == root and root.is_dir()
        _assert_raises(RuntimeError, lambda: create_work_root(root),
                       "already exists")


def test_ance_requires_two_visible_gpus():
    from train_ance import require_ance_gpus
    assert require_ance_gpus(2) == 2
    assert require_ance_gpus(4) == 4
    for count in (0, 1):
        _assert_raises(RuntimeError, lambda n=count: require_ance_gpus(n), "2 visible GPUs")


def test_inferencer_help_exits_cleanly():
    result = subprocess.run(
        [sys.executable, str(project_root / 'scripts' / 'run_ance_data_gen.py'), '--help'],
        text=True, capture_output=True, check=False)
    assert result.returncode == 0, (result.returncode, result.stderr)
    assert 'Traceback' not in result.stderr



# ---- prepared initial round (step 4: preparation split out of training) ------
# Round 0 costs 6h38m of a two-GPU allocation (job 204931), so it is mined once by a
# one-GPU job and adopted by training. Adoption REBINDS the round's run_id, which is the
# check that otherwise stops a foreign round reaching the trainer -- so everything below
# pins the conditions under which that rebinding is allowed.

def _identity(**over):
    base = {
        'recipe': 'ance_paper',
        'base_model_path': '/models/warmup',
        'base_model_weights_sha256': 'a' * 64,
        'tokenizer_sha256': {'tokenizer.json': 'b' * 64},
        'corpus_sha256': 'c' * 64,
        'queries_sha256': 'd' * 64,
        'qrels_sha256': 'e' * 64,
        'mixture_sha256': {'train_msmarco.jsonl': 'f' * 64},
        'seed': 42,
        'mining': {'mining_depth': 200, 'n_negs': 20, 'ann_chunk_factor': 5,
                   'query_max_len': 64, 'passage_max_len': 512,
                   'per_device_eval_batch_size': 256, 'paper_fidelity': True},
    }
    base.update(over)
    return base


def _prepared_artifact(root, *, identity=None, records=1):
    """Write what a --prepare-initial job leaves behind."""
    import train_ance as ta
    root = Path(root)
    data = root / "training_data_initial"
    data.mkdir(parents=True, exist_ok=True)
    payload = data / "train_msmarco.jsonl"
    with open(payload, 'w', encoding='utf-8') as fh:
        for i in range(records):
            fh.write(json.dumps({"query": f"q{i}", "positive_passages": [],
                                 "negative_passages": []}) + "\n")
    manifest = {
        'prepared_run_id': 'prepared-deadbeef-j1',
        'prepared_at': '2026-09-16T00:00:00+00:00',
        'code_revision': {'git_sha': '0' * 40, 'git_dirty': False},
        'identity': identity or _identity(),
        'payload_sha256': {payload.name: ta._sha256(payload)},
        'round_meta': {'ann_no': INITIAL_ROUND, 'checkpoint_step': 0,
                       'n_records': records, 'n_queries_mined': records,
                       'files': [payload.name], 'n_sampling_failures': 0},
    }
    (root / ta.PREPARED_MANIFEST_NAME).write_text(json.dumps(manifest, indent=2))
    return manifest


def test_prepared_round_adopted_round_is_readable_by_this_run():
    """The point of the feature: a round mined by another job becomes this run's
    round 0, with its provenance preserved rather than erased."""
    import train_ance as ta
    with tempfile.TemporaryDirectory() as tmp:
        artifact, work = Path(tmp) / "artifact", Path(tmp) / "work"
        _prepared_artifact(artifact)
        work.mkdir()
        ta.adopt_initial_round(artifact, work, run_id=RUN_ID, identity=_identity())
        data_dir, meta = read_round(work, INITIAL_ROUND, run_id=RUN_ID)
        assert data_dir.is_dir()
        assert meta['checkpoint_step'] == 0, "adoption must not fake a checkpoint step"
        assert meta['prepared_from']['prepared_run_id'] == 'prepared-deadbeef-j1'
        assert latest_committed_round(work) == 0, "initial is not a numeric round"


def test_prepared_round_with_different_inputs_is_refused():
    """Every identity field is load-bearing: a round mined from another corpus,
    another initialization or another seed is a different experiment."""
    import train_ance as ta
    mining = dict(_identity()['mining'], mining_depth=50)
    for field, value in (('corpus_sha256', 'z' * 64),
                         ('base_model_weights_sha256', 'z' * 64),
                         ('seed', 7),
                         ('recipe', 'ance'),
                         ('mining', mining)):          # depth/n_negs decide WHICH negatives
        with tempfile.TemporaryDirectory() as tmp:
            artifact, work = Path(tmp) / "artifact", Path(tmp) / "work"
            _prepared_artifact(artifact, identity=_identity(**{field: value}))
            work.mkdir()
            _assert_raises(
                RuntimeError,
                lambda: ta.adopt_initial_round(artifact, work, run_id=RUN_ID,
                                               identity=_identity()),
                field)
            assert not round_paths(work, INITIAL_ROUND)['ready'].exists(), \
                f"{field} mismatch must publish nothing"


def test_prepared_payload_edited_after_preparation_is_refused():
    """The manifest is not trusted about the bytes: hashes are re-checked, because
    the artifact outlives the job that wrote it and sits on shared storage."""
    import train_ance as ta
    with tempfile.TemporaryDirectory() as tmp:
        artifact, work = Path(tmp) / "artifact", Path(tmp) / "work"
        _prepared_artifact(artifact)
        payload = artifact / "training_data_initial" / "train_msmarco.jsonl"
        payload.write_text(payload.read_text() + json.dumps({"query": "extra"}) + "\n")
        work.mkdir()
        _assert_raises(RuntimeError,
                       lambda: ta.adopt_initial_round(artifact, work, run_id=RUN_ID,
                                                      identity=_identity()),
                       "does not match the hash")


def test_a_directory_without_a_prepared_manifest_is_refused():
    """A bare training_data_initial/ is not evidence of anything: no initialization,
    no corpus, no settings. Refuse rather than adopt on filename alone."""
    import train_ance as ta
    with tempfile.TemporaryDirectory() as tmp:
        artifact, work = Path(tmp) / "artifact", Path(tmp) / "work"
        (artifact / "training_data_initial").mkdir(parents=True)
        work.mkdir()
        _assert_raises(RuntimeError,
                       lambda: ta.adopt_initial_round(artifact, work, run_id=RUN_ID,
                                                      identity=_identity()),
                       "prepared_initial.json")


def test_adoption_requires_recorded_provenance():
    """The publication primitive refuses to be the place the audit trail is dropped:
    an adopted round must say which artifact it came from."""
    from ance_mining import adopt_prepared_round
    with tempfile.TemporaryDirectory() as tmp:
        src = Path(tmp) / "src"
        src.mkdir()
        (src / "train.jsonl").write_text(json.dumps({"query": "q"}) + "\n")
        work = Path(tmp) / "work"
        work.mkdir()
        _assert_raises(RoundError,
                       lambda: adopt_prepared_round(work, INITIAL_ROUND,
                                                    source_dir=src,
                                                    meta={'run_id': RUN_ID}),
                       "prepared_from")


def test_an_adopted_initial_round_still_is_not_a_refresh():
    """Adoption changes where round 0 was mined, never what it counts as. A run whose
    only round is an adopted initial has refreshed nothing."""
    summary = {'rounds_consumed': 1, 'consumed': [
        {'ann_no': INITIAL_ROUND, 'checkpoint_step': 0, 'consumed_steps': 5000,
         'prepared_from': {'artifact_dir': '/prepared'}}]}
    _assert_raises(RoundError,
                   lambda: assert_ance_refresh(summary, min_fresh_rounds=2),
                   "DISTINCT checkpoints")


TESTS = [
    ("commit: metadata first, marker last", test_marker_is_written_last_and_metadata_first),
    ("commit: no marker => invisible round", test_a_round_without_its_marker_is_invisible),
    ("mine: both callers share one sequence", test_both_callers_mine_through_one_shared_sequence),
    ("mine: shared sequence refuses duplicate ids", test_the_shared_sequence_refuses_a_corpus_with_repeated_ids),
    ("preflight: consistent inputs accepted", test_preflight_accepts_consistent_inputs),
    ("preflight: disagreeing query text refused", test_preflight_rejects_a_query_whose_text_disagrees),
    ("preflight: no resolvable positive refused", test_preflight_rejects_a_query_with_no_resolvable_positive),
    ("preflight: qrels-only positive accepted", test_preflight_accepts_a_positive_resolvable_only_through_the_qrels),
    ("preflight: whitespace-escaped positive resolved", test_preflight_resolves_a_whitespace_escaped_positive),
    ("preflight: duplicate-text positive resolved via qrels", test_preflight_resolves_a_duplicate_text_positive_through_the_qrels),
    ("preflight: duplicate owner absent from qrels refused", test_preflight_refuses_a_duplicate_whose_owner_the_qrels_do_not_carry),
    ("preflight: text nowhere in corpus refused", test_preflight_refuses_a_positive_whose_text_is_nowhere_in_the_corpus),
    ("preflight: reverse map built only on a miss", test_preflight_builds_the_reverse_map_only_when_a_positive_misses),
    ("preflight: --preflight uses no GPU and writes nothing", test_preflight_flag_touches_no_gpu_and_writes_nothing),
    ("commit: foreign run's round refused", test_a_foreign_runs_round_is_refused),
    ("commit: every malformed round refused", test_a_malformed_round_is_always_refused),
    ("commit: sampling failures block publication", test_sampling_failures_block_publication_entirely),
    ("commit: ready_initial is not a numeric round", test_initial_round_is_not_counted_as_a_numeric_round),
    ("commit: latest tracks the highest marker", test_latest_committed_round_tracks_the_highest_marker),
    ("fresh: initial round alone fails the gate", test_initial_round_alone_does_not_satisfy_the_gate),
    ("fresh: no rounds fails the gate", test_no_rounds_at_all_does_not_satisfy_the_gate),
    ("fresh: unconsumed round does not count", test_a_published_but_never_consumed_round_does_not_count),
    ("fresh: one consumed round passes", test_one_consumed_checkpoint_round_satisfies_the_gate),
    ("fresh: two distinct rounds by default", test_two_distinct_fresh_rounds_are_required_by_default),
    ("fresh: two rounds, one checkpoint = one refresh", test_two_rounds_from_one_checkpoint_are_one_refresh),
    ("fresh: min_consume_steps enforced", test_min_consume_steps_is_enforced),
    ("fresh: min_fresh_rounds enforced", test_min_fresh_rounds_is_enforced),
    ("encode: a stalled encode times out", test_a_stalled_encode_raises_rather_than_hanging_forever),
    ("encode: paper subprocess has a hard timeout", test_the_paper_encoder_uses_the_hard_subprocess_timeout),
    ("smoke: cap-length batch and nonzero failure", test_gpu_smoke_is_conservative_and_fails_closed),
    ("supervise: dead inferencer kills the trainer", test_a_dead_inferencer_kills_the_trainer_and_fails_the_run),
    ("supervise: simultaneous death still caught", test_an_inferencer_that_dies_with_the_trainer_is_still_caught),
    ("supervise: healthy run reports no failure", test_a_healthy_run_reports_no_failure),
    ("supervise: clean EARLY exit is a failure", test_a_clean_early_inferencer_exit_is_a_failure),
    ("supervise: our own termination is not a failure", test_our_own_termination_is_not_an_early_exit),
    ("loss: non-finite rejected before backward", test_non_finite_loss_is_rejected_before_backward),
    ("grad: no step taken on non-finite gradients", test_no_step_is_taken_on_non_finite_gradients),
    ("grad: finite gradients do step", test_finite_gradients_do_step),
    ("summary: retried then raises", test_summary_write_is_retried_then_raises),
    ("summary: opportunities != publications != consumptions", test_the_summary_separates_opportunities_publications_and_consumptions),
    ("summary: final step off the save boundary counts", test_a_final_step_off_the_save_boundary_still_counts_its_checkpoint),
    ("summary: terminal round recorded, not counted", test_a_terminal_round_published_after_the_last_swap_is_not_consumed),
    ("summary: survives a transient failure", test_summary_write_survives_a_transient_failure),
    ("summary: a failed write is fatal", test_trainer_treats_a_failed_summary_write_as_fatal),
    ("summary: a successful write lands the payload", test_a_successful_summary_write_lands_the_payload),
    ("runid: unique within one second", test_run_ids_are_unique_within_one_second),
    ("runid: carries fingerprint and job id", test_run_id_carries_the_fingerprint_and_job_id),
    ("runid: omits job id outside SLURM", test_run_id_omits_the_job_id_outside_slurm),
    ("runid: existing work root refused", test_work_root_creation_refuses_an_existing_directory),
    ("gpu: two visible devices required", test_ance_requires_two_visible_gpus),
    ("cli: inferencer help exits cleanly", test_inferencer_help_exits_cleanly),
    ("success: terminal record validates a complete run", test_a_complete_run_validates_via_the_terminal_record),
    ("success: no terminal record => rejected", test_without_the_terminal_record_a_complete_run_is_rejected),
    ("success: zero-step run rejected", test_a_zero_step_run_is_rejected),
    ("success: non-finite loss in the log rejected", test_a_non_finite_loss_in_the_log_is_rejected),
    ("success: manifest records run identity", test_manifest_records_the_run_identity_for_the_evaluator),
    ("prepared: adopted round readable by this run", test_prepared_round_adopted_round_is_readable_by_this_run),
    ("prepared: different inputs refused", test_prepared_round_with_different_inputs_is_refused),
    ("prepared: edited payload refused", test_prepared_payload_edited_after_preparation_is_refused),
    ("prepared: no manifest => refused", test_a_directory_without_a_prepared_manifest_is_refused),
    ("prepared: adoption needs provenance", test_adoption_requires_recorded_provenance),
    ("prepared: adopted initial is not a refresh", test_an_adopted_initial_round_still_is_not_a_refresh),
]


def _run(name, fn):
    try:
        fn()
    except Exception as e:                                        # noqa: BLE001
        print(f"  ❌ {name}\n       {type(e).__name__}: {e}")
        if os.environ.get("TEST_TRACE"):
            traceback.print_exc()
        return False
    print(f"  ✅ {name}")
    return True


def main():
    print("\nANCE pipeline tests (handoff, supervision, freshness, success)")
    print("=" * 66)
    passed = sum(_run(n, f) for n, f in TESTS)
    print("=" * 66)
    print(f"  {passed}/{len(TESTS)} passed")
    return 0 if passed == len(TESTS) else 1


if __name__ == "__main__":
    sys.exit(main())
