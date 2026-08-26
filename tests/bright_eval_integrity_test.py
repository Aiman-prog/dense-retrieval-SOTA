"""
BRIGHT evaluation boundary: the checks that a reported number belongs to the run
that claims it.

`bright_exclusions_test.py` owns the exclusion filter itself (drop before the top_k
cut). This suite owns everything around it: the artifacts a run consumes must be
present and mutually consistent, embeddings and results of two models with the same
basename must not share a directory, collected results must match the run that was
requested, and an incomplete evaluation must fail instead of averaging what it got.

Preprocessing is out of scope -- nothing here re-checks what the writers in
`src/data/preprocessor.py` already guarantee.

Run: python tests/bright_eval_integrity_test.py
"""
import json
import os
import subprocess
import sys
import tempfile
import traceback
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root / 'src'))
sys.path.insert(0, str(project_root / 'scripts'))

from evaluation.trec_eval_wrapper import TrecEvalWrapper          # noqa: E402
from utils.helpers import (                                       # noqa: E402
    check_eval_artifacts, model_run_tag, require_eval_files,
)


def _assert_raises(exc, fn, contains=None):
    try:
        fn()
    except exc as e:
        assert contains is None or contains in str(e), str(e)
        return
    raise AssertionError(f"expected {exc.__name__}")


def _qrels(pairs):
    qrels = {}
    for q, d in pairs:
        qrels.setdefault(q, set()).add(d)
    return qrels


def _write_queries(path, qids):
    path.write_text(''.join(
        json.dumps({'query_id': q, 'query': 'text'}) + '\n' for q in qids))


# ---- aggregation -----------------------------------------------------------

def test_missing_judged_queries_score_zero():
    """pytrec_eval returns a zero row for an EMPTY ranking but nothing at all for a
    judged query absent from the run. Averaging over what it returned therefore
    shrinks the denominator; every judged query must count."""
    qrels = _qrels([('q1', 'dA'), ('q2', 'dB'), ('q3', 'dC')])
    run = {'q1': {'dA': 1.0}, 'q2': {}}          # q3 never reached the run at all
    m = TrecEvalWrapper(qrels).evaluate(run, {'ndcg_cut_10'})
    assert abs(m['ndcg_cut_10'] - 1.0 / 3.0) < 1e-9, m

    # an unjudged query in the run must not dilute the mean either
    run_extra = dict(run, q9={'dZ': 1.0})
    m2 = TrecEvalWrapper(qrels).evaluate(run_extra, {'ndcg_cut_10'})
    assert abs(m2['ndcg_cut_10'] - 1.0 / 3.0) < 1e-9, m2


# ---- artifact preflight ----------------------------------------------------

def test_preflight_rejects_missing_and_empty_files():
    """A missing or zero-byte artifact must fail before anything is encoded, and the
    message must name the file -- a Tevatron subprocess failure does not."""
    d = Path(tempfile.mkdtemp())
    present = d / "biology_queries.jsonl"
    _write_queries(present, ['q1'])
    empty = d / "biology_qrels.txt"
    empty.write_text("")
    absent = d / "biology_excluded.json"

    require_eval_files("biology", [present])                       # the happy path
    _assert_raises(FileNotFoundError,
                   lambda: require_eval_files("biology", [present, empty]),
                   "biology_qrels.txt")
    _assert_raises(FileNotFoundError,
                   lambda: require_eval_files("biology", [present, absent]),
                   "biology_excluded.json")


# ---- cross-artifact consistency --------------------------------------------

def test_query_and_exclusion_key_sets_must_agree():
    qrels = _qrels([('q1', 'dA')])
    ok = {'q1': frozenset(), 'q2': frozenset({'dX'})}
    check_eval_artifacts('biology', qrels, ok, query_ids=['q1', 'q2'])

    # exclusion map carries a query the run will never score
    _assert_raises(ValueError,
                   lambda: check_eval_artifacts('biology', qrels,
                                                dict(ok, q3=frozenset()),
                                                query_ids=['q1', 'q2']),
                   'q3')
    # a query with no exclusion entry: its exclusions would silently not apply
    _assert_raises(ValueError,
                   lambda: check_eval_artifacts('biology', qrels, {'q1': frozenset()},
                                                query_ids=['q1', 'q2']),
                   'q2')
    # a judged query that is not in the query set scores 0 forever
    _assert_raises(ValueError,
                   lambda: check_eval_artifacts('biology', _qrels([('q7', 'dA')]),
                                                ok, query_ids=['q1', 'q2']),
                   'q7')


def test_encoded_query_ids_must_match_source():
    """Compared as SETS: the encoder may reorder, but it may not drop or invent."""
    d = Path(tempfile.mkdtemp())
    qfile = d / "biology_queries.jsonl"
    _write_queries(qfile, ['q1', 'q2', 'q3'])
    qrels = _qrels([('q1', 'dA')])
    excluded = {q: frozenset() for q in ('q1', 'q2', 'q3')}

    # source ids read from the file, encoder reordered them: fine
    check_eval_artifacts('biology', qrels, excluded, queries_file=qfile,
                         encoded_query_ids=['q3', 'q1', 'q2'])
    # dropped by the encoder
    _assert_raises(ValueError,
                   lambda: check_eval_artifacts('biology', qrels, excluded,
                                                queries_file=qfile,
                                                encoded_query_ids=['q1', 'q2']),
                   'q3')
    # invented by the encoder
    _assert_raises(ValueError,
                   lambda: check_eval_artifacts('biology', qrels, excluded,
                                                queries_file=qfile,
                                                encoded_query_ids=['q1', 'q2', 'q3', 'q4']),
                   'q4')
    # BM25 has no encoded ids and must be able to skip that check
    check_eval_artifacts('biology', qrels, excluded, queries_file=qfile)


# ---- run identity ----------------------------------------------------------

def test_run_tag_isolates_same_basename():
    """`checkpoint-500` from two models must not share embeddings or results."""
    d = Path(tempfile.mkdtemp())
    a, b = d / "run_a" / "checkpoint-500", d / "run_b" / "checkpoint-500"
    a.mkdir(parents=True); b.mkdir(parents=True)
    assert model_run_tag(a) != model_run_tag(b), "same-basename runs collide"
    assert model_run_tag(a) == model_run_tag(str(a)), "tag is not stable for one path"
    assert model_run_tag(a).startswith("checkpoint-500"), "tag is not readable"


def test_collect_results_rejects_foreign_or_nonfinite():
    """A leftover result file from another model, another domain, or one carrying a
    non-finite metric must be reported, never folded into the macro average."""
    import run_all_evals

    d = Path(tempfile.mkdtemp())
    config = {'paths': {'results_dir': 'results'}}
    model = d / "models" / "wanted" / "checkpoint-500"
    other = d / "models" / "other" / "checkpoint-500"
    model.mkdir(parents=True); other.mkdir(parents=True)

    base = d / 'results' / model_run_tag(model)
    base.mkdir(parents=True)

    def write(domain, model_path, recorded_domain=None, ndcg=0.2):
        (base / f"{domain}_results.json").write_text(json.dumps({
            'domain': recorded_domain or domain,
            'model_path': str(model_path),
            'metrics': {'ndcg_cut_10': ndcg, 'recip_rank': 0.1, 'recall_1000': 0.9},
        }))

    write('good', model)
    write('foreign', other)
    write('wrongdomain', model, recorded_domain='biology')
    write('nonfinite', model, ndcg=float('nan'))

    original = run_all_evals.get_data_base_dir, run_all_evals.get_path
    run_all_evals.get_data_base_dir = lambda: d
    run_all_evals.get_path = lambda key, *a, **k: d / key
    try:
        rows, invalid = run_all_evals.collect_results(
            model, ['good', 'foreign', 'wrongdomain', 'nonfinite'], config)
    finally:
        run_all_evals.get_data_base_dir, run_all_evals.get_path = original

    assert [r['domain'] for r in rows] == ['good'], rows
    assert set(invalid) == {'foreign', 'wrongdomain', 'nonfinite'}, invalid


# ---- orchestration ---------------------------------------------------------

def test_empty_domain_selection_exits_nonzero():
    """`--domains ,` evaluated nothing and still printed 'All evaluations complete'."""
    r = subprocess.run(
        [sys.executable, str(project_root / 'scripts' / 'run_all_evals.py'),
         '--model_path', str(project_root), '--domains', ','],
        capture_output=True, text=True)
    assert r.returncode != 0, r.stdout + r.stderr
    assert 'no domains' in (r.stdout + r.stderr).lower(), r.stdout + r.stderr


def test_evaluate_bright_fails_on_incomplete_domain_set():
    """ANCE's post-training eval used to print '[Eval] Skipping <domain>' and average
    the domains that survived. It must raise, and it must raise before the first
    encode rather than after eleven domains of GPU time."""
    import utils.helpers as helpers

    d = Path(tempfile.mkdtemp())
    ctx = {'args': {'temp_workdir': 'temp_x', 'eval_top_k': 10}}
    config = {'evaluation': {'eval_domains': ['biology', 'economics']}}

    def boom(*a, **k):
        raise AssertionError("encoding started before the preflight failed")

    original_get_path, original_encode = helpers.get_path, helpers.encode_to_pickle
    helpers.get_path = lambda key, *a, **k: d / key
    helpers.encode_to_pickle = boom
    try:
        _assert_raises(FileNotFoundError,
                       lambda: helpers.evaluate_bright(ctx, config, d / "model"),
                       'biology')
        _assert_raises(ValueError,
                       lambda: helpers.evaluate_bright(
                           ctx, {'evaluation': {'eval_domains': []}}, d / "model"),
                       'eval_domains')
    finally:
        helpers.get_path, helpers.encode_to_pickle = original_get_path, original_encode


TESTS = [
    ("aggregation: missing judged queries score zero", test_missing_judged_queries_score_zero),
    ("preflight: missing or empty artifact raises", test_preflight_rejects_missing_and_empty_files),
    ("consistency: query ids == exclusion keys, qrels covered", test_query_and_exclusion_key_sets_must_agree),
    ("consistency: encoded ids match source as sets", test_encoded_query_ids_must_match_source),
    ("identity: run tag isolates same basename", test_run_tag_isolates_same_basename),
    ("identity: collect_results rejects foreign/non-finite", test_collect_results_rejects_foreign_or_nonfinite),
    ("orchestration: empty domain selection exits nonzero", test_empty_domain_selection_exits_nonzero),
    ("orchestration: evaluate_bright fails, never skips", test_evaluate_bright_fails_on_incomplete_domain_set),
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
    print("\nBRIGHT evaluation integrity tests")
    print("=" * 58)
    passed = sum(_run(n, f) for n, f in TESTS)
    print("=" * 58)
    print(f"  {passed}/{len(TESTS)} passed")
    return 0 if passed == len(TESTS) else 1


if __name__ == "__main__":
    sys.exit(main())
