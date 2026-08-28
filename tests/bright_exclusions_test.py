"""
BRIGHT `excluded_ids`: loader preservation and the shared eval-time filter.

BRIGHT's protocol removes each query's excluded documents from the ranking before
scoring. Retrieval truncates at top_k, so the filter must run on an over-retrieved
list -- otherwise excluded hits burn result slots and nothing refills them.

Run: python tests/bright_exclusions_test.py
"""
import json
import os
import sys
import tempfile
import traceback
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root / 'src'))

from data.bright_loader import BRIGHTLoader                      # noqa: E402
from utils.helpers import (                                      # noqa: E402
    apply_exclusions, load_excluded_ids, search_depth,
)


class _FakeSplit:
    def __init__(self, rows): self.rows = rows
    def __len__(self): return len(self.rows)
    def __getitem__(self, k):
        return [r[k] for r in self.rows] if isinstance(k, str) else self.rows[k]


def _loader(examples, documents=None):
    ldr = BRIGHTLoader()
    ldr.examples_dataset = {k: _FakeSplit(v) for k, v in examples.items()}
    ldr.documents_dataset = {k: _FakeSplit(v) for k, v in (documents or {}).items()}
    return ldr


def _ex_row(qid, excluded, gold=('d1',)):
    return {'id': qid, 'query': 'q', 'gold_ids': list(gold), 'excluded_ids': excluded}


def _assert_raises(exc, fn, contains=None):
    try:
        fn()
    except exc as e:
        assert contains is None or contains in str(e), str(e)
        return
    raise AssertionError(f"expected {exc.__name__}")


# ---- loader ----------------------------------------------------------------

def test_real_ids_preserved():
    ldr = _loader({'bio': [_ex_row('q1', ['dA', 'dB'])]})
    assert ldr.get_excluded_ids('bio') == {'q1': ['dA', 'dB']}


def test_na_sentinel_dropped():
    """8 of 12 BRIGHT domains store the literal 'N/A' instead of a doc id."""
    ldr = _loader({'bio': [_ex_row('q1', ['N/A'])]})
    assert ldr.get_excluded_ids('bio') == {'q1': []}


def test_blank_values_dropped():
    ldr = _loader({'bio': [_ex_row('q1', ['  ', '', 'dA'])]})
    assert ldr.get_excluded_ids('bio') == {'q1': ['dA']}


def test_duplicates_collapsed():
    ldr = _loader({'bio': [_ex_row('q1', ['dA', 'dA', 'dB', 'dA'])]})
    assert ldr.get_excluded_ids('bio') == {'q1': ['dA', 'dB']}


def test_non_list_rejected():
    ldr = _loader({'bio': [_ex_row('q1', 'dA')]})
    _assert_raises(ValueError, lambda: ldr.get_excluded_ids('bio'), 'excluded_ids')


def test_data_split_carries_exclusions():
    ldr = _loader({'bio': [_ex_row('q1', ['dA'])]},
                  {'bio': [{'id': 'd1', 'content': 'x'}, {'id': 'dA', 'content': 'y'}]})
    assert ldr.get_data_split('bio')['excluded'] == {'q1': ['dA']}


# ---- shared filter ---------------------------------------------------------

def test_excluded_removed_and_others_promoted():
    run = {'q1': {'dTop': 9.0, 'dB': 8.0, 'dC': 7.0}}
    out = apply_exclusions(run, {'q1': {'dTop'}}, top_k=2)
    assert list(out['q1']) == ['dB', 'dC'], out


def test_rank_one_exclusion_does_not_shrink_results():
    run = {'q1': {f'd{i}': 100.0 - i for i in range(5)}}
    out = apply_exclusions(run, {'q1': {'d0'}}, top_k=4)
    assert len(out['q1']) == 4 and 'd0' not in out['q1']


def test_query_isolation():
    run = {'q1': {'dA': 2.0, 'dB': 1.0}, 'q2': {'dA': 2.0, 'dB': 1.0}}
    out = apply_exclusions(run, {'q1': {'dA'}}, top_k=2)
    assert list(out['q1']) == ['dB'] and list(out['q2']) == ['dA', 'dB']


def test_filtering_precedes_the_top_k_cutoff():
    """The core requirement: an eligible doc below the cutoff must refill a slot
    freed by an exclusion, so the caller still gets a full top_k."""
    run = {'q1': {f'd{i}': 10000.0 - i for i in range(1002)}}
    excluded = {'q1': {'d0', 'd5'}}
    out = apply_exclusions(run, excluded, top_k=1000)
    assert len(out['q1']) == 1000
    assert 'd1000' in out['q1'] and 'd1001' in out['q1'], "lower ranks did not refill"
    assert not ({'d0', 'd5'} & set(out['q1']))


def test_unknown_query_is_untouched():
    run = {'q9': {'dA': 1.0}}
    assert apply_exclusions(run, {'q1': {'dA'}}, top_k=1) == {'q9': {'dA': 1.0}}


def test_search_depth():
    excluded = {'q1': {'a', 'b', 'c'}, 'q2': {'a'}}
    assert search_depth(1000, excluded) == 1003          # deepest query
    assert search_depth(1000, excluded, 'q2') == 1001    # this query
    assert search_depth(1000, excluded, 'absent') == 1000


# ---- on-disk round trip ----------------------------------------------------

def test_load_excluded_ids_round_trip():
    d = Path(tempfile.mkdtemp())
    (d / "bio_excluded.json").write_text(json.dumps({'q1': ['dA', 'dB']}))
    assert load_excluded_ids('bio', d) == {'q1': frozenset({'dA', 'dB'})}


def test_missing_file_raises():
    """Silently treating a missing file as 'no exclusions' would reproduce the old,
    unfiltered numbers without saying so."""
    _assert_raises(FileNotFoundError,
                   lambda: load_excluded_ids('bio', Path(tempfile.mkdtemp())),
                   'bio_excluded.json')


# ---- BM25 job exit status ---------------------------------------------------

def _bm25_main_over_empty_processed_dir():
    """Drive run_bm25_evals.main() with a stub pyserini and an empty processed dir,
    so every domain fails preflight. A job that evaluated nothing must not exit 0."""
    import types
    sys.path.insert(0, str(project_root / 'scripts'))
    stub = types.ModuleType('pyserini'); search = types.ModuleType('pyserini.search')
    lucene = types.ModuleType('pyserini.search.lucene')
    lucene.LuceneSearcher = object
    search.lucene = lucene; stub.search = search
    for name, mod in (('pyserini', stub), ('pyserini.search', search),
                      ('pyserini.search.lucene', lucene)):
        sys.modules.setdefault(name, mod)

    import importlib
    bm25 = importlib.import_module('run_bm25_evals')
    empty = Path(tempfile.mkdtemp())
    original = (bm25.get_path, bm25.check_and_prepare_bm25_data, bm25.preflight_java)
    bm25.get_path = lambda key, *a, **k: empty / key
    bm25.check_and_prepare_bm25_data = lambda *a, **k: None   # needs the HF cache
    # This test is about the exit status after every domain fails, not about Java.
    # The preflight is exercised directly in bm25_provenance_test.py.
    bm25.preflight_java = lambda: "<stubbed>"
    try:
        return bm25.main()
    finally:
        bm25.get_path, bm25.check_and_prepare_bm25_data, bm25.preflight_java = original


def test_bm25_returns_nonzero_when_a_domain_fails():
    rc = _bm25_main_over_empty_processed_dir()
    assert rc not in (0, None), f"BM25 reported success after failing every domain (rc={rc})"


def test_bm25_preflight_requires_the_exclusions_file():
    src = (project_root / 'scripts' / 'run_bm25_evals.py').read_text()
    assert '_excluded.json' in src, "exclusions file is not part of BM25 preflight"


TESTS = [
    ("loader: real ids preserved", test_real_ids_preserved),
    ("loader: 'N/A' sentinel dropped", test_na_sentinel_dropped),
    ("loader: blank values dropped", test_blank_values_dropped),
    ("loader: duplicates collapsed", test_duplicates_collapsed),
    ("loader: non-list rejected", test_non_list_rejected),
    ("loader: data split carries exclusions", test_data_split_carries_exclusions),
    ("filter: excluded removed, others promoted", test_excluded_removed_and_others_promoted),
    ("filter: rank-1 exclusion keeps top_k full", test_rank_one_exclusion_does_not_shrink_results),
    ("filter: query isolation", test_query_isolation),
    ("filter: applied before the top_k cutoff", test_filtering_precedes_the_top_k_cutoff),
    ("filter: unknown query untouched", test_unknown_query_is_untouched),
    ("depth: over-retrieval sizing", test_search_depth),
    ("io: excluded json round trip", test_load_excluded_ids_round_trip),
    ("io: missing file raises", test_missing_file_raises),
    ("bm25: nonzero after a domain failure", test_bm25_returns_nonzero_when_a_domain_fails),
    ("bm25: exclusions in preflight", test_bm25_preflight_requires_the_exclusions_file),
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
    print("\nBRIGHT excluded_ids tests")
    print("=" * 58)
    passed = sum(_run(n, f) for n, f in TESTS)
    print("=" * 58)
    print(f"  {passed}/{len(TESTS)} passed")
    return 0 if passed == len(TESTS) else 1


if __name__ == "__main__":
    sys.exit(main())
