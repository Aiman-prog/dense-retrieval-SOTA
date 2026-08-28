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
    RUN_MANIFEST_NAME, check_eval_artifacts, encoding_contract_drift,
    eval_artifact_hashes, load_training_manifest, model_run_tag, require_eval_files,
    training_provenance,
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


def test_excluded_none_is_for_benchmarks_with_no_exclusion_map():
    """MS MARCO has no `{domain}_excluded.json`, so it cannot satisfy the
    key-set check. `excluded=None` skips ONLY that check; the two that catch a
    mismatched run still apply."""
    d = Path(tempfile.mkdtemp())
    qfile = d / "msmarco_dev_queries.jsonl"
    _write_queries(qfile, ['q1', 'q2', 'q3'])
    qrels = _qrels([('q1', 'dA'), ('q2', 'dB')])

    # encoded == source, qrels subset of source: the MS MARCO contract
    check_eval_artifacts('msmarco_dev', qrels, None, queries_file=qfile,
                         encoded_query_ids=['q2', 'q1', 'q3'])
    # a query judged but absent from the query file still raises
    _assert_raises(ValueError,
                   lambda: check_eval_artifacts('msmarco_dev',
                                                _qrels([('q9', 'dA')]), None,
                                                queries_file=qfile,
                                                encoded_query_ids=['q1', 'q2', 'q3']),
                   'q9')
    # the encoder dropped a query
    _assert_raises(ValueError,
                   lambda: check_eval_artifacts('msmarco_dev', qrels, None,
                                                queries_file=qfile,
                                                encoded_query_ids=['q1', 'q2']),
                   'q3')
    # the encoder invented one
    _assert_raises(ValueError,
                   lambda: check_eval_artifacts('msmarco_dev', qrels, None,
                                                queries_file=qfile,
                                                encoded_query_ids=['q1', 'q2', 'q3', 'q4']),
                   'q4')


def test_excluded_none_does_not_weaken_bright():
    """BRIGHT can never reach the None path: load_excluded_ids raises on a missing
    file, so a domain cannot opt out of exclusion filtering this way."""
    src = (project_root / 'src' / 'utils' / 'helpers.py').read_text()
    body = src[src.index('def load_excluded_ids('):src.index('def search_depth(')]
    assert 'raise' in body, "a missing exclusion file no longer raises"
    for caller in ('src/utils/helpers.py', 'src/evaluation/evaluate.py',
                   'scripts/run_bm25_evals.py'):
        text = (project_root / caller).read_text()
        for line in text.splitlines():
            if 'check_eval_artifacts(' in line and 'def ' not in line:
                assert 'None' not in line, f"{caller} passes excluded=None: {line!r}"


def test_msmarco_evaluator_checks_ids_before_searching():
    """The check has to run before the FAISS search, or a wrong run is built first."""
    src = (project_root / 'scripts' / 'eval_msmarco.py').read_text()
    assert 'check_eval_artifacts(' in src, "eval_msmarco does not verify query ids"
    assert src.index('check_eval_artifacts(') < src.index('idx.search('),         "ids are verified after the search"


def test_msmarco_paper_comparison_requires_official_query_count():
    from eval_msmarco import msmarco_paper_comparable
    assert msmarco_paper_comparable(6980) is True
    for count in (0, 1, 6979, 6981, 101093):
        assert msmarco_paper_comparable(count) is False
    src = (project_root / 'scripts' / 'eval_msmarco.py').read_text()
    assert 'if paper_comparable else ""' in src
    assert "'paper_comparable': paper_comparable" in src


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



# ---- evaluation is bound to the checkpoint that produced it ------------------

_TRAINED = {"pooling": "cls", "normalize": True,
            "query_max_len": 1024, "passage_max_len": 512}


def _manifest_dir(tmp, name="model", model=None, **extra):
    d = Path(tmp) / name
    d.mkdir(parents=True, exist_ok=True)
    (d / RUN_MANIFEST_NAME).write_text(json.dumps({
        "recipe": "inbatch", "fingerprint": "abc123", "base_model": "/bge-m3",
        "data_sha256": ["hash-of-a-mixture-since-deleted"],
        "final_global_step": 10314,
        "effective_config": {"model": model if model is not None else dict(_TRAINED)},
        **extra,
    }))
    return d


def test_encoding_contract_drift_is_detected():
    """Pooling/normalize/lengths decide what an embedding means."""
    with tempfile.TemporaryDirectory() as tmp:
        d = _manifest_dir(tmp)
        m = load_training_manifest(d)
        assert encoding_contract_drift(m, dict(_TRAINED)) == {}
        drift = encoding_contract_drift(m, {**_TRAINED, "passage_max_len": 128})
        assert drift == {"passage_max_len": {"checkpoint": 512, "evaluation": 128}}, drift
        drift = encoding_contract_drift(m, {**_TRAINED, "pooling": "mean"})
        assert set(drift) == {"pooling"}, drift


def test_manifest_resolves_from_a_checkpoint_subdir():
    with tempfile.TemporaryDirectory() as tmp:
        d = _manifest_dir(tmp)
        ckpt = d / "checkpoint-2062"
        ckpt.mkdir()
        assert load_training_manifest(ckpt)["fingerprint"] == "abc123"


def test_legacy_checkpoint_without_a_manifest_is_evaluable():
    """Pre-gate checkpoints exist; they warn, not fail, and record a null provenance."""
    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp) / "legacy"
        d.mkdir()
        assert load_training_manifest(d) is None
        assert training_provenance(None) is None
        assert encoding_contract_drift(None, dict(_TRAINED)) == {}


def test_training_data_hashes_are_recorded_not_enforced():
    """A checkpoint stays valid when the mixture that produced it is gone."""
    with tempfile.TemporaryDirectory() as tmp:
        m = load_training_manifest(_manifest_dir(tmp))
        prov = training_provenance(m)
        assert prov["training_data_sha256"] == ["hash-of-a-mixture-since-deleted"]
        assert prov["fingerprint"] == "abc123" and prov["recipe"] == "inbatch"
        # the mixture is absent from this tmpdir entirely, and that is not an error
        assert encoding_contract_drift(m, dict(_TRAINED)) == {}


def test_eval_artifact_hashes_track_every_scoring_input():
    with tempfile.TemporaryDirectory() as tmp:
        proc = Path(tmp)
        (proc / "biology_corpus.jsonl").write_text('{"docid":"d1","text":"a"}\n')
        (proc / "biology_queries.jsonl").write_text('{"query_id":"q1","query":"x"}\n')
        (proc / "biology_qrels.txt").write_text("q1 Q0 d1 1\n")
        (proc / "biology_excluded.json").write_text('{"q1": []}')
        first = eval_artifact_hashes(proc, ["biology"])
        assert set(first["biology"]) == {"corpus", "queries", "qrels", "excluded"}
        assert all(v for v in first["biology"].values())
        # a regenerated corpus changes the digest, which is what makes two runs
        # comparable or not
        (proc / "biology_corpus.jsonl").write_text('{"docid":"d1","text":"CHANGED"}\n')
        assert eval_artifact_hashes(proc, ["biology"])["biology"]["corpus"] != \
            first["biology"]["corpus"]
        # a missing artifact is recorded as null rather than crashing the digest
        (proc / "biology_excluded.json").unlink()
        assert eval_artifact_hashes(proc, ["biology"])["biology"]["excluded"] is None


def test_bm25_comparison_requires_matching_artifact_hashes():
    """A legacy BM25 summary cannot prove it scored the dense run's artifacts."""
    import run_all_evals

    domains = ["biology"]
    current = {"biology": {"corpus": "a", "queries": "b",
                            "qrels": "c", "excluded": "d"}}
    base = {"domains": domains, "run_tag": "bm25", "model": "BM25",
            "macro_ndcg_cut_10": 0.1}

    _assert_raises(
        ValueError,
        lambda: run_all_evals.validate_bm25_comparison(
            {**base, "domains": ["economics"], "eval_artifact_sha256": current},
            domains, current, "wrong-domains.json"),
        "Domain sets differ",
    )
    _assert_raises(
        ValueError,
        lambda: run_all_evals.validate_bm25_comparison(
            base, domains, current, "legacy.json"),
        "eval_artifact_sha256",
    )
    _assert_raises(
        ValueError,
        lambda: run_all_evals.validate_bm25_comparison(
            {**base, "eval_artifact_sha256": {
                "biology": {**current["biology"], "corpus": "different"}}},
            domains, current, "different.json"),
        "different evaluation artifacts",
    )
    record = run_all_evals.validate_bm25_comparison(
        {**base, "eval_artifact_sha256": current}, domains, current, "matching.json")
    assert record["eval_artifacts_verified"] is True


TESTS = [
    ("aggregation: missing judged queries score zero", test_missing_judged_queries_score_zero),
    ("preflight: missing or empty artifact raises", test_preflight_rejects_missing_and_empty_files),
    ("consistency: query ids == exclusion keys, qrels covered", test_query_and_exclusion_key_sets_must_agree),
    ("consistency: encoded ids match source as sets", test_encoded_query_ids_must_match_source),
    ("artifacts: excluded=None for MS MARCO", test_excluded_none_is_for_benchmarks_with_no_exclusion_map),
    ("artifacts: excluded=None cannot weaken BRIGHT", test_excluded_none_does_not_weaken_bright),
    ("artifacts: MS MARCO checks ids before searching", test_msmarco_evaluator_checks_ids_before_searching),
    ("artifacts: paper comparison requires 6,980", test_msmarco_paper_comparison_requires_official_query_count),
    ("identity: run tag isolates same basename", test_run_tag_isolates_same_basename),
    ("identity: collect_results rejects foreign/non-finite", test_collect_results_rejects_foreign_or_nonfinite),
    ("orchestration: empty domain selection exits nonzero", test_empty_domain_selection_exits_nonzero),
    ("orchestration: evaluate_bright fails, never skips", test_evaluate_bright_fails_on_incomplete_domain_set),
    ("provenance: encoding-contract drift detected", test_encoding_contract_drift_is_detected),
    ("provenance: manifest resolves from checkpoint-*", test_manifest_resolves_from_a_checkpoint_subdir),
    ("provenance: legacy checkpoint stays evaluable", test_legacy_checkpoint_without_a_manifest_is_evaluable),
    ("provenance: training hashes recorded not enforced", test_training_data_hashes_are_recorded_not_enforced),
    ("provenance: eval artifact hashes track inputs", test_eval_artifact_hashes_track_every_scoring_input),
    ("compare: BM25 artifact hashes mandatory", test_bm25_comparison_requires_matching_artifact_hashes),
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
