"""ANCE negative selection, against the PRODUCTION selector.

Replaces `test_ance_negative_sampling.py`, which re-implemented `mine_negatives`
inside the test file and so could not fail when either ANCE script changed. Worse,
its `test_fallback_to_pot_when_all_are_positives` certified the defect: when every
ANN candidate was a positive, the old code promoted the positives to negatives, and
when the pool came back empty the loader padded the group with the positive itself.
That test is inverted here.

The reference implementation (`microsoft/ANCE`) samples uniformly from the retrieved
top-k and simply yields fewer negatives when it runs short. A fixed Tevatron group
cannot represent "fewer", so a shortfall is a sampling failure and the round is not
published.

Run: python tests/ance_negative_pool_test.py
"""
import json
import os
import random
import sys
import tempfile
import traceback
from collections import Counter
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root / 'src'))
sys.path.insert(0, str(project_root / 'scripts'))

from ance_mining import (                                          # noqa: E402
    RoundError, SamplingFailure, assert_corpus_ids_unique, build_round_records,
    query_shard, record_positives, select_ance_negatives,
)


def _assert_raises(exc, fn, contains=None):
    try:
        fn()
    except exc as e:
        assert contains is None or contains in str(e), str(e)
        return str(e)
    raise AssertionError(f"expected {exc.__name__}")


def _cands(n, start=0):
    return [f"d{i}" for i in range(start, start + n)]


# ---- selection --------------------------------------------------------------

def test_returns_requested_count():
    got = select_ance_negatives("q1", _cands(200), set(), n_negs=1,
                                rng=random.Random(0))
    assert len(got) == 1


def test_multi_negative_draw_has_no_duplicates():
    for seed in range(50):
        got = select_ance_negatives("q1", _cands(200), set(), n_negs=5,
                                    rng=random.Random(seed))
        assert len(got) == len(set(got)) == 5


def test_positives_are_never_returned():
    positives = {"d0", "d1", "d2"}
    for seed in range(100):
        got = select_ance_negatives("q1", _cands(200), positives, n_negs=1,
                                    rng=random.Random(seed))
        assert not set(got) & positives


def test_sampling_is_uniform_not_top1():
    """The reference shuffles the retrieved list; top-1 slicing is its MRR mode."""
    rng = random.Random(42)
    counts = Counter()
    for _ in range(500):
        counts[select_ance_negatives("q1", _cands(200), set(), n_negs=1, rng=rng)[0]] += 1
    assert len(counts) > 20, f"too little spread: {len(counts)} unique"
    assert counts.most_common(1)[0][1] < 100, "one candidate dominates the draw"


def test_restricted_to_the_retrieved_candidates():
    """The caller's FAISS search sets the depth; nothing outside it may be drawn."""
    candidates = _cands(200)
    universe = set(candidates)
    for seed in range(100):
        got = select_ance_negatives("q1", candidates, set(), n_negs=1,
                                    rng=random.Random(seed))
        assert set(got) <= universe


def test_deterministic_under_a_fixed_seed():
    a = select_ance_negatives("q1", _cands(200), set(), n_negs=3, rng=random.Random(7))
    b = select_ance_negatives("q1", _cands(200), set(), n_negs=3, rng=random.Random(7))
    c = select_ance_negatives("q1", _cands(200), set(), n_negs=3, rng=random.Random(8))
    assert a == b
    assert a != c, "different seeds produced identical draws"


def test_all_positive_pool_fails_it_does_not_return_a_positive():
    """INVERTED from the old suite: the old code returned the positives here."""
    candidates = ["d0", "d1", "d2"]
    _assert_raises(
        SamplingFailure,
        lambda: select_ance_negatives("q1", candidates, {"d0", "d1", "d2"},
                                      n_negs=1, rng=random.Random(0)),
        contains="never pads with a positive")


def test_short_pool_fails_rather_than_padding():
    """The old code repeated the pool to length; a duplicate is not a negative."""
    _assert_raises(
        SamplingFailure,
        lambda: select_ance_negatives("q1", ["d0", "d1"], set(), n_negs=5,
                                      rng=random.Random(0)),
        contains="need 5")


def test_empty_candidate_list_fails():
    _assert_raises(SamplingFailure,
                   lambda: select_ance_negatives("q1", [], set(), n_negs=1,
                                                 rng=random.Random(0)))


def test_mixture_positives_count_even_when_absent_from_qrels():
    record = {'query_id': 'q1', 'query': 'q',
              'positive_passages': [{'docid': 'd5', 'text': 'gold'}]}
    positives = record_positives(record, {'q1': {'d0'}})
    assert positives == {'d0', 'd5'}
    for seed in range(50):
        got = select_ance_negatives("q1", _cands(20), positives, n_negs=1,
                                    rng=random.Random(seed))
        assert got[0] not in positives


# ---- round record construction ----------------------------------------------

def _mixture(tmp, name="train_hq.jsonl", n=3):
    path = Path(tmp) / name
    with open(path, 'w') as f:
        for i in range(n):
            f.write(json.dumps({
                'query_id': f'q{i}', 'query': f'query {i}',
                'positive_passages': [{'docid': f'p{i}', 'text': f'pos {i}'}],
                'negative_passages': [{'docid': 'stale', 'text': 'stale'}],
            }) + '\n')
    return path


def test_round_records_replace_negatives_and_keep_the_positive():
    with tempfile.TemporaryDirectory() as tmp:
        path = _mixture(tmp)
        mined = {f'q{i}': [f'n{i}'] for i in range(3)}
        lookup = {f'n{i}': f'neg text {i}' for i in range(3)}
        (name, records), = build_round_records([path], mined, lookup, n_negs=1)
        assert name == "train_hq.jsonl"
        for i, rec in enumerate(records):
            assert rec['negative_passages'] == [{'docid': f'n{i}',
                                                 'text': f'neg text {i}'}]
            assert rec['positive_passages'][0]['docid'] == f'p{i}'


def test_missing_corpus_text_is_a_failure_not_an_empty_passage():
    with tempfile.TemporaryDirectory() as tmp:
        path = _mixture(tmp, n=1)
        _assert_raises(
            SamplingFailure,
            lambda: list(build_round_records([path], {'q0': ['n0']}, {}, n_negs=1)),
            contains="no text in the corpus")


def test_uncovered_query_is_a_failure_not_a_stale_negative():
    """The old code left the record's ORIGINAL negatives in place when the query
    was absent from the mining pass, silently mixing mined and pre-existing data."""
    with tempfile.TemporaryDirectory() as tmp:
        path = _mixture(tmp, n=1)
        _assert_raises(
            SamplingFailure,
            lambda: list(build_round_records([path], {}, {'n0': 'x'}, n_negs=1)),
            contains="no mined negatives")


def test_wrong_negative_count_is_a_failure():
    with tempfile.TemporaryDirectory() as tmp:
        path = _mixture(tmp, n=1)
        _assert_raises(
            SamplingFailure,
            lambda: list(build_round_records([path], {'q0': []}, {'n0': 'x'},
                                             n_negs=1)),
            contains="0 negative(s)")


# ---- query sharding ---------------------------------------------------------
#
# Upstream: drivers/run_ann_data_gen.py:281-295.
#
#     queries_per_chunk = num_queries // chunk_factor
#     q_start_idx = queries_per_chunk * effective_idx
#     q_end_idx   = num_queries if effective_idx == chunk_factor - 1
#                              else q_start_idx + queries_per_chunk
#
# Expected values below are derived from THAT arithmetic, not from ours.

def _all_shards(ids, k):
    return [ids[start:end] for start, end in
            (query_shard(ids, n, k)[1:] for n in range(k))]


def test_the_shard_partition_matches_upstreams_arithmetic():
    """One table over the whole partition property, at several chunk factors.

    For every k the shards must be disjoint, ordered, and cover the query set exactly;
    `//` floors, so the remainder goes to shard k-1 rather than being spread. A
    "fairer" split would be a different partition and a different set of mined rounds.
    Expected sizes are derived from upstream's arithmetic, not from ours.
    """
    cases = [
        (1000, 1, [1000]),
        (1000, 2, [500, 500]),
        (1000, 5, [200] * 5),
        (1002, 5, [200, 200, 200, 200, 202]),       # remainder -> last shard
        (1000, 7, [142] * 6 + [148]),
        (37, 1, [37]),                              # BRIGHT: one shard, whole set
        (3, 5, [0, 0, 0, 0, 3]),                    # fewer queries than shards
    ]
    for n, k, sizes in cases:
        ids = [f"q{i}" for i in range(n)]
        shards = _all_shards(ids, k)
        assert [len(x) for x in shards] == sizes, (n, k, [len(x) for x in shards])
        flat = [q for shard in shards for q in shard]
        assert flat == ids, (n, k)                  # ordered, nothing dropped
        assert len(set(flat)) == n, (n, k)          # and nothing mined twice


def test_shard_index_rotates_modulo_the_chunk_factor():
    """ann_no 0 is the initial round, which is why it takes shard 0.

    Upstream generates it with --end_output_num 0 from ann_no = -1, so output_num is
    0 and the refresh rounds continue 1, 2, ...
    """
    ids = [f"q{i}" for i in range(100)]
    assert [query_shard(ids, n, 5)[0] for n in range(12)] == \
        [0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 0, 1]
    # chunk_factor 1 never rotates: every round is the whole set.
    assert {query_shard(ids, n, 1) for n in (0, 1, 9)} == {(0, 0, 100)}


def test_sharding_skips_other_shards_but_still_fails_on_an_unmined_member():
    """Sharding must not become a silent way to drop a query the miner could not serve."""
    with tempfile.TemporaryDirectory() as tmp:
        path = _mixture(tmp, n=4)
        corpus = {f"n{i}": f"neg {i}" for i in range(4)}
        mined = {'q0': ['n0'], 'q1': ['n1']}
        # q2/q3 belong to another shard: skipped, not an error.
        out = dict(build_round_records([path], mined, corpus, n_negs=1,
                                       shard_qids={'q0', 'q1'}))
        assert [r['query_id'] for r in out[Path(path).name]] == ['q0', 'q1']
        # q1 IS in the shard but was not mined: still a failure.
        _assert_raises(
            SamplingFailure,
            lambda: list(build_round_records([path], {'q0': ['n0']}, corpus,
                                             n_negs=1, shard_qids={'q0', 'q1'})),
            contains="q1")


# ---- corpus id uniqueness ---------------------------------------------------

def test_repeated_corpus_ids_are_refused_before_mining():
    """Selection treats an ANN result as distinct documents; FirstP makes that true.

    A repeated id means the encoded pickle and the corpus disagree, which silently
    changes what "top 200" and "20 retained negatives" mean. Checked once per round
    instead of deduplicated per query -- per-query dedup would paper over it and only
    a multi-vector (MaxP) index would actually need it.
    """
    assert assert_corpus_ids_unique(["d0", "d1", "d2"]) == 3
    _assert_raises(RoundError,
                   lambda: assert_corpus_ids_unique(["d0", "d1", "d0", "d2", "d1"]),
                   contains="d0")


TESTS = [
    ("select: returns the requested count", test_returns_requested_count),
    ("select: no duplicates in a multi-negative draw", test_multi_negative_draw_has_no_duplicates),
    ("select: positives are never returned", test_positives_are_never_returned),
    ("select: uniform, not top-1", test_sampling_is_uniform_not_top1),
    ("select: restricted to the retrieved candidates", test_restricted_to_the_retrieved_candidates),
    ("select: deterministic under a fixed seed", test_deterministic_under_a_fixed_seed),
    ("select: all-positive pool FAILS (inverted)", test_all_positive_pool_fails_it_does_not_return_a_positive),
    ("select: short pool fails rather than padding", test_short_pool_fails_rather_than_padding),
    ("select: empty candidate list fails", test_empty_candidate_list_fails),
    ("select: mixture positives count too", test_mixture_positives_count_even_when_absent_from_qrels),
    ("round: negatives replaced, positive kept", test_round_records_replace_negatives_and_keep_the_positive),
    ("round: missing corpus text fails", test_missing_corpus_text_is_a_failure_not_an_empty_passage),
    ("round: uncovered query fails", test_uncovered_query_is_a_failure_not_a_stale_negative),
    ("round: wrong negative count fails", test_wrong_negative_count_is_a_failure),
    ("shard: partition matches upstream arithmetic", test_the_shard_partition_matches_upstreams_arithmetic),
    ("shard: rotates modulo chunk_factor", test_shard_index_rotates_modulo_the_chunk_factor),
    ("shard: skips others, fails on an unmined member", test_sharding_skips_other_shards_but_still_fails_on_an_unmined_member),
    ("corpus: repeated docids refused", test_repeated_corpus_ids_are_refused_before_mining),
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
    print("\nANCE negative-pool tests")
    print("=" * 58)
    passed = sum(_run(n, f) for n, f in TESTS)
    print("=" * 58)
    print(f"  {passed}/{len(TESTS)} passed")
    return 0 if passed == len(TESTS) else 1


if __name__ == "__main__":
    sys.exit(main())
