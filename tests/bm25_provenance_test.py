"""
BM25 index provenance and BM25-vs-dense comparability.

`run_bm25_evals.py` used to reuse an index whenever its directory was non-empty, so a
corpus regenerated under different preprocessing, a pyserini upgrade, and a truncated
index from a killed job were all invisible. And the eval launcher defaults to four
pilot domains while BM25 always runs twelve, so the two macro numbers were routinely
not comparable.

Run: python tests/bm25_provenance_test.py
"""
import json
import os
import sys
import tempfile
import traceback
from pathlib import Path
from unittest import mock

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root / 'src'))
sys.path.insert(0, str(project_root / 'scripts'))

import run_bm25_evals as bm                                       # noqa: E402


def _assert_raises(exc, fn, contains=None):
    try:
        fn()
    except exc as e:
        assert contains is None or contains in str(e), str(e)
        return str(e)
    raise AssertionError(f"expected {exc.__name__}")


def _corpus(tmp, text="alpha"):
    corpus_dir = Path(tmp) / "corpus"
    corpus_dir.mkdir(parents=True, exist_ok=True)
    (corpus_dir / "corpus.jsonl").write_text(
        json.dumps({"id": "d1", "contents": text}) + "\n")
    return corpus_dir, corpus_dir / "corpus.jsonl"


def _built_index(tmp):
    """A directory that looks like a finished Lucene index."""
    index_dir = Path(tmp) / "index"
    index_dir.mkdir(parents=True, exist_ok=True)
    (index_dir / "segments_1").write_bytes(b"lucene")
    return index_dir


def _ensure(tmp, **kw):
    """Drive ensure_index with the real builder stubbed out."""
    corpus_dir, corpus_file = _corpus(tmp, kw.pop("text", "alpha"))
    index_dir = Path(tmp) / "index"
    calls = []

    def fake_build(cdir, idir, threads=4):
        calls.append((Path(cdir), Path(idir)))
        idir.mkdir(parents=True, exist_ok=True)
        (idir / "segments_1").write_bytes(b"lucene")

    with mock.patch.object(bm, 'build_lucene_index', fake_build):
        result = bm.ensure_index(corpus_file, corpus_dir, index_dir, **kw)
    return calls, index_dir, result


# ---- index provenance ------------------------------------------------------

def test_missing_index_builds():
    with tempfile.TemporaryDirectory() as tmp:
        calls, index_dir, _ = _ensure(tmp)
        assert len(calls) == 1, calls
        assert (index_dir.parent / bm.META_NAME).is_file()


def test_matching_meta_reuses():
    with tempfile.TemporaryDirectory() as tmp:
        _ensure(tmp)
        calls, _, _ = _ensure(tmp)
        assert calls == [], "a current index must not be rebuilt"


def test_changed_corpus_rebuilds_and_archives():
    with tempfile.TemporaryDirectory() as tmp:
        _ensure(tmp)
        calls, index_dir, _ = _ensure(tmp, text="beta")          # corpus digest moves
        assert len(calls) == 1, calls
        archived = list(index_dir.parent.glob("index.old_*"))
        assert len(archived) == 1, [p.name for p in index_dir.parent.iterdir()]


def test_changed_pyserini_version_rebuilds():
    with tempfile.TemporaryDirectory() as tmp:
        _ensure(tmp)
        meta_path = Path(tmp) / bm.META_NAME
        meta = json.loads(meta_path.read_text())
        meta['pyserini_version'] = "0.0.1-different"
        meta_path.write_text(json.dumps(meta))
        calls, _, _ = _ensure(tmp)
        assert len(calls) == 1, "a pyserini change must invalidate the index"


def test_populated_index_without_meta_is_not_reused():
    """A killed indexer leaves a non-empty directory. Existence is not evidence."""
    with tempfile.TemporaryDirectory() as tmp:
        _built_index(tmp)
        calls, _, _ = _ensure(tmp)
        assert len(calls) == 1, "a partial index must be rebuilt, not searched"


def test_unreadable_meta_is_not_reused():
    with tempfile.TemporaryDirectory() as tmp:
        _ensure(tmp)
        (Path(tmp) / bm.META_NAME).write_text("{not json")
        calls, _, _ = _ensure(tmp)
        assert len(calls) == 1


def test_no_rebuild_raises_on_a_stale_index():
    with tempfile.TemporaryDirectory() as tmp:
        _ensure(tmp)
        _assert_raises(RuntimeError, lambda: _ensure(tmp, text="beta", allow_rebuild=False),
                       "--no-rebuild")


def test_no_rebuild_still_reuses_a_current_index():
    with tempfile.TemporaryDirectory() as tmp:
        _ensure(tmp)
        calls, _, _ = _ensure(tmp, allow_rebuild=False)
        assert calls == []


def test_provenance_excludes_k1_and_b():
    """k1/b are search-time (set_bm25), so they must not force a rebuild."""
    with tempfile.TemporaryDirectory() as tmp:
        _corpus_dir, corpus_file = _corpus(tmp)
        prov = bm.corpus_provenance(corpus_file)
        assert set(prov) == {"corpus_sha256", "corpus_bytes", "pyserini_version"}, prov
        assert len(prov["corpus_sha256"]) == 64


# ---- comparability ---------------------------------------------------------

def _summary(tmp, name, domains):
    path = Path(tmp) / name
    path.write_text(json.dumps({
        'model': 'BM25 (k1=0.9, b=0.4)', 'run_tag': 'bm25_k1-0.9_b-0.4',
        'domains': domains, 'per_domain': [], 'macro_ndcg_cut_10': 0.11}))
    return path


def _run_all_evals_compare(dense_domains, bm25_domains):
    """The comparability gate as run_all_evals applies it, without a GPU run."""
    import run_all_evals                                          # noqa: F401
    with tempfile.TemporaryDirectory() as tmp:
        bm25 = json.loads(_summary(tmp, "s.json", bm25_domains).read_text())
        return set(bm25.get('domains', [])) == set(dense_domains)


def test_domain_sets_matching_is_accepted():
    twelve = ['biology', 'economics', 'stackoverflow', 'theoremqa_questions']
    assert _run_all_evals_compare(twelve, twelve) is True


def test_pilot_subset_vs_full_bm25_is_rejected():
    """The default launcher runs 4 domains; BM25 runs 12. That is not a comparison."""
    pilot = ['biology', 'economics', 'stackoverflow', 'theoremqa_questions']
    full = pilot + ['pony', 'leetcode', 'aops', 'robotics']
    assert _run_all_evals_compare(pilot, full) is False


def test_compare_bm25_flag_and_recording_exist():
    """Pin the surface: the flag, the refusal, and `compared_to` in the summary."""
    src = (project_root / 'scripts' / 'run_all_evals.py').read_text()
    assert '--compare_bm25' in src
    assert "'compared_to': compared_to" in src
    # The gate must precede the macro computation, or a failed comparison would still
    # print a number and overwrite a good summary.
    # Anchored on the macro block itself, not on one metric's expression: the macro
    # computation grew a helper when recall and MRR were added to the report.
    assert src.index('compared_to = None') < src.index("def _macro("), \
        "the comparability gate must run before anything is computed or written"


def test_bm25_summary_is_written_only_for_a_complete_run():
    src = (project_root / 'scripts' / 'run_bm25_evals.py').read_text()
    assert src.index("return 1") < src.index("summary_path"), \
        "the partial-run exit must precede the summary write"
    assert "'domains': domains" in src, "the summary must record the domain set"


def test_search_depth_is_clamped_to_the_corpus():
    src = (project_root / 'scripts' / 'run_bm25_evals.py').read_text()
    assert "min(search_depth(top_k, excluded, qid), corpus_size)" in src, \
        "unclamped depth allocates a queue far larger than the corpus on aops"


def test_results_carry_provenance():
    src = (project_root / 'scripts' / 'run_bm25_evals.py').read_text()
    for field in ('"bm25": {"k1": k1, "b": b}', '"run_tag": run_tag', '**provenance'):
        assert field in src, field



# ---- transactional publication ---------------------------------------------

def test_interrupted_build_publishes_nothing_and_keeps_the_old_index():
    """A killed build must not leave a partial index at the canonical path.

    Building straight into index_dir, with the old meta still beside it, meant a
    half-written index matched stale provenance and was reported as "reusing".
    """
    with tempfile.TemporaryDirectory() as tmp:
        corpus_dir, corpus_file = _corpus(tmp, "alpha")
        index_dir = Path(tmp) / "index"
        index_dir.mkdir(parents=True)
        (index_dir / "segments_1").write_bytes(b"old-but-valid")
        meta = index_dir.parent / bm.META_NAME
        prov = bm.corpus_provenance(corpus_file)
        meta.write_text(json.dumps({**prov, "corpus_sha256": "stale"}))

        def exploding_build(cdir, idir, threads=4):
            idir.mkdir(parents=True, exist_ok=True)
            (idir / "segments_partial").write_bytes(b"half")
            raise RuntimeError("killed mid-build")

        with mock.patch.object(bm, 'build_lucene_index', exploding_build):
            _assert_raises(RuntimeError,
                           lambda: bm.ensure_index(corpus_file, corpus_dir, index_dir))

        assert (index_dir / "segments_1").read_bytes() == b"old-but-valid", \
            "the old index must survive a failed rebuild"
        assert not (index_dir / "segments_partial").exists(), "partial leaked into place"
        assert not list(index_dir.parent.glob("index.building*")), "staging leaked"


def test_archived_index_keeps_its_metadata():
    """Provenance must travel with the index it describes, never be unlinked."""
    with tempfile.TemporaryDirectory() as tmp:
        calls, index_dir, _ = _ensure(tmp, text="alpha")
        meta = index_dir.parent / bm.META_NAME
        first = json.loads(meta.read_text())
        # corpus changes -> archive + rebuild
        (index_dir.parent / "corpus" / "corpus.jsonl").write_text(
            json.dumps({"id": "d1", "contents": "beta"}) + "\n")
        corpus_dir = index_dir.parent / "corpus"
        with mock.patch.object(bm, 'build_lucene_index',
                               lambda c, i, threads=4: (i.mkdir(parents=True, exist_ok=True),
                                                        (i / "segments_1").write_bytes(b"x"))):
            bm.ensure_index(corpus_dir / "corpus.jsonl", corpus_dir, index_dir)
        archived = [d for d in index_dir.parent.iterdir() if ".old_" in d.name]
        assert archived, "old index was not archived"
        kept = archived[0] / bm.META_NAME
        assert kept.is_file(), "archived index lost its provenance"
        assert json.loads(kept.read_text())['corpus_sha256'] == first['corpus_sha256']


# ---- corpus identity against the dense artifacts ---------------------------

def test_corpus_text_mismatch_regenerates():
    """Docid equality is not enough: the same id with different text is a different corpus."""
    with tempfile.TemporaryDirectory() as tmp:
        dense = Path(tmp) / "biology_corpus.jsonl"
        dense.write_text(json.dumps({"docid": "d1", "text": "alpha"}) + "\n")
        bm25 = Path(tmp) / "corpus" / "corpus.jsonl"
        bm25.parent.mkdir(parents=True)
        bm25.write_text(json.dumps({"id": "d1", "contents": "DIFFERENT"}) + "\n")
        assert bm.bm25_corpus_matches_dense(bm25, dense) is False
        bm25.write_text(json.dumps({"id": "d1", "contents": "alpha"}) + "\n")
        assert bm.bm25_corpus_matches_dense(bm25, dense) is True


def test_missing_qrel_document_raises():
    """A judged document absent from the corpus makes the domain unscoreable."""
    with tempfile.TemporaryDirectory() as tmp:
        corpus = Path(tmp) / "corpus.jsonl"
        corpus.write_text(json.dumps({"id": "d1", "contents": "alpha"}) + "\n")
        qrels = Path(tmp) / "q.txt"
        qrels.write_text("q1 Q0 d1 1\nq1 Q0 d_missing 1\n")
        _assert_raises(RuntimeError,
                       lambda: bm.verify_qrel_documents("biology", corpus, qrels),
                       "d_missing")
        qrels.write_text("q1 Q0 d1 1\n")
        bm.verify_qrel_documents("biology", corpus, qrels)      # must not raise


def test_malformed_qrels_row_is_rejected_not_skipped():
    """verify_qrel_documents must go through the strict reader.

    It used to parse qrels by hand with `len(parts) >= 3`, which silently accepted a
    three-column row. _load_qrels is documented as the one strict reader for mining
    AND evaluation and raises on anything that is not four columns; bypassing it let a
    malformed judgement file through as if it were clean.
    """
    with tempfile.TemporaryDirectory() as tmp:
        corpus = Path(tmp) / "corpus.jsonl"
        corpus.write_text(json.dumps({"id": "d1", "contents": "alpha"}) + "\n")
        qrels = Path(tmp) / "q.txt"
        qrels.write_text("q1 Q0 d1\n")            # three columns, not four
        _assert_raises(ValueError,
                       lambda: bm.verify_qrel_documents("biology", corpus, qrels),
                       "four columns")


# ---- Java preflight ---------------------------------------------------------

def test_java_preflight_rejects_a_missing_home():
    with tempfile.TemporaryDirectory() as tmp:
        with mock.patch.dict(os.environ, {"JAVA_HOME": str(Path(tmp) / "nope")}, clear=False):
            _assert_raises(RuntimeError, bm.preflight_java, "JAVA_HOME")


def test_java_preflight_rejects_a_home_without_java():
    with tempfile.TemporaryDirectory() as tmp:
        (Path(tmp) / "bin").mkdir()
        with mock.patch.dict(os.environ, {"JAVA_HOME": tmp}, clear=False):
            _assert_raises(RuntimeError, bm.preflight_java, "bin/java")



def test_undeletable_staging_raises_before_building_into_it():
    """Leftover staging from a killed build must never be built into and published.

    Reproduced: with rmtree failing, segments_STALE from a previous corpus was
    published into the live index, meta was written, and the next run reused it.
    """
    import shutil as _sh
    with tempfile.TemporaryDirectory() as tmp:
        corpus_dir, corpus_file = _corpus(tmp, "alpha")
        index_dir = Path(tmp) / "index"
        staging = index_dir.with_name("index.building")
        staging.mkdir(parents=True)
        (staging / "segments_STALE").write_bytes(b"from-a-different-corpus")

        real = _sh.rmtree
        _sh.rmtree = lambda *a, **k: None          # silently does nothing
        try:
            msg = _assert_raises(RuntimeError,
                                 lambda: bm.ensure_index(corpus_file, corpus_dir, index_dir))
        finally:
            _sh.rmtree = real
        assert "staging" in msg.lower(), msg
        assert not index_dir.exists(), "nothing may be published"


def test_preflight_java_runs_before_pyserini_is_imported():
    """The import initializes the JVM, so a later preflight cannot diagnose it."""
    src = (project_root / 'scripts' / 'run_bm25_evals.py').read_text()
    body = src[src.index("def main("):]
    assert body.index("preflight_java()") < body.index("from pyserini"), \
        "pyserini is imported before the Java preflight"


TESTS = [
    ("index: missing index builds", test_missing_index_builds),
    ("index: matching meta reuses", test_matching_meta_reuses),
    ("index: changed corpus rebuilds and archives", test_changed_corpus_rebuilds_and_archives),
    ("index: changed pyserini version rebuilds", test_changed_pyserini_version_rebuilds),
    ("index: populated dir without meta is not reused", test_populated_index_without_meta_is_not_reused),
    ("index: unreadable meta is not reused", test_unreadable_meta_is_not_reused),
    ("index: --no-rebuild raises on a stale index", test_no_rebuild_raises_on_a_stale_index),
    ("index: --no-rebuild still reuses a current index", test_no_rebuild_still_reuses_a_current_index),
    ("index: k1/b are not index provenance", test_provenance_excludes_k1_and_b),
    ("compare: matching domain sets accepted", test_domain_sets_matching_is_accepted),
    ("compare: pilot subset vs full bm25 rejected", test_pilot_subset_vs_full_bm25_is_rejected),
    ("compare: flag, refusal and recording exist", test_compare_bm25_flag_and_recording_exist),
    ("results: summary only for a complete run", test_bm25_summary_is_written_only_for_a_complete_run),
    ("results: search depth clamped to corpus", test_search_depth_is_clamped_to_the_corpus),
    ("results: carry k1/b, run tag and provenance", test_results_carry_provenance),
    ("index: interrupted build publishes nothing", test_interrupted_build_publishes_nothing_and_keeps_the_old_index),
    ("index: archived index keeps its metadata", test_archived_index_keeps_its_metadata),
    ("corpus: text mismatch regenerates", test_corpus_text_mismatch_regenerates),
    ("corpus: missing qrel document raises", test_missing_qrel_document_raises),
    ("corpus: malformed qrels row is rejected", test_malformed_qrels_row_is_rejected_not_skipped),
    ("java: preflight rejects a missing JAVA_HOME", test_java_preflight_rejects_a_missing_home),
    ("java: preflight rejects a home without bin/java", test_java_preflight_rejects_a_home_without_java),
    ("java: preflight precedes the pyserini import", test_preflight_java_runs_before_pyserini_is_imported),
    ("index: undeletable staging raises", test_undeletable_staging_raises_before_building_into_it),
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
    print("\nBM25 provenance and comparability tests")
    print("=" * 58)
    passed = sum(_run(n, f) for n, f in TESTS)
    print("=" * 58)
    print(f"  {passed}/{len(TESTS)} passed")
    return 0 if passed == len(TESTS) else 1


if __name__ == "__main__":
    sys.exit(main())
