"""
Unit tests for the BRIGHT/ReasonIR preprocessor (src/data/preprocessor.py) and the
BRIGHT loader (src/data/bright_loader.py).

CPU-only, deterministic, no network and no `load_dataset` against the HF hub: every
generator is driven by injected in-memory records, so the suite never touches the
2 GB cache (which was written by an older `datasets` version and no longer loads).

The Tevatron contract group is the exception -- it loads the produced JSONL through
the *installed* tevatron reader, which is the whole point: it pins the pin.

Run: python tests/preprocessor_test.py      (KMP_DUPLICATE_LIB_OK=TRUE on macOS)
"""
import json
import os
import sys
import tempfile
import traceback
from pathlib import Path

import pandas as pd

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root / 'src'))

from data.preprocessor import (  # noqa: E402
    BRIGHTPreprocessor,
    MIXTURE_FILES,
    MSMARCO_ONLY_FILES,
    PreprocessingError,
    require_derived_artifacts,
    require_mixture_files,
    run_setup,
)
from utils.helpers import atomic_write  # noqa: E402
from data.bright_loader import BRIGHTLoader  # noqa: E402


# ---- helpers ----------------------------------------------------------------

def _tmp():
    return Path(tempfile.mkdtemp(prefix="preproc_test_"))


def _read_jsonl(path):
    with open(path, encoding='utf-8') as f:
        return [json.loads(line) for line in f if line.strip()]


def _mix_record(qid, query, pos, neg):
    """pos/neg are lists of (docid, text)."""
    return {
        "query_id": qid,
        "query": query,
        "positive_passages": [{"docid": d, "text": t} for d, t in pos],
        "negative_passages": [{"docid": d, "text": t} for d, t in neg],
    }


def _write_mixture(mix_dir, files):
    """files: {filename: [record, ...]}"""
    mix_dir = Path(mix_dir)
    mix_dir.mkdir(parents=True, exist_ok=True)
    for name, records in files.items():
        with open(mix_dir / name, 'w', encoding='utf-8') as f:
            for r in records:
                f.write(json.dumps(r, ensure_ascii=False) + '\n')
    return mix_dir


def _fill_components(files):
    """Add a benign record for any declared component a fixture omits, so a test about
    something else is not caught by the missing-component check."""
    out = dict(files)
    for i, name in enumerate(MIXTURE_FILES):
        out.setdefault(name, [_mix_record(f"filler_{i}", f"filler {i}",
                                          [(f"fp{i}", f"filler pos {i}")],
                                          [(f"fn{i}", f"filler neg {i}")])])
    return out


def _simple_mixture(mix_dir):
    """A three-file mixture with one duplicate-text passage across records."""
    return _write_mixture(mix_dir, {
        "train_hq.jsonl": [
            _mix_record("reasonir_hq_0", "hq query zero", [("bright_a", "alpha text")],
                        [("hq_neg_0_0", "neg one")]),
        ],
        "train_msmarco.jsonl": [
            _mix_record("msmarco_0", "ms query", [("msmarco_pos_0", "ms positive")],
                        [("msmarco_neg_0", "ms negative")]),
        ],
        "train_vl.jsonl": [
            # "alpha text" repeats under a different docid -> exercises the remap
            _mix_record("reasonir_vl_0", "vl query", [("vl_pos_0_0", "alpha text")],
                        [("vl_neg_0_0", "neg two")]),
        ],
    })


def _reasonir_rows(pairs):
    """Build ReasonIR-shaped rows: query/pos/neg are lists of [title, payload]."""
    rows = []
    for q, pos, neg in pairs:
        rows.append({
            "query": ["instruction prefix ", q],
            "pos": [["", p] for p in pos],
            "neg": [["", n] for n in neg],
        })
    return rows


class _FakeSplit:
    """Minimal stand-in for a datasets.Dataset split (column and row access)."""

    def __init__(self, rows):
        self.rows = rows

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, key):
        if isinstance(key, str):
            return [r[key] for r in self.rows]
        return self.rows[key]

    @property
    def column_names(self):
        return list(self.rows[0].keys()) if self.rows else []


def _loader_with(documents=None, examples=None):
    loader = BRIGHTLoader()
    loader.documents_dataset = {k: _FakeSplit(v) for k, v in (documents or {}).items()}
    loader.examples_dataset = {k: _FakeSplit(v) for k, v in (examples or {}).items()}
    return loader


def _assert_raises(exc, fn, *, contains=None):
    try:
        fn()
    except exc as e:
        if contains:
            for token in ([contains] if isinstance(contains, str) else contains):
                assert token in str(e), f"message {str(e)!r} does not mention {token!r}"
        return
    raise AssertionError(f"expected {exc.__name__}, nothing raised")


# ---- 1. Tevatron contract ---------------------------------------------------

def test_corpus_jsonl_has_exactly_docid_and_text():
    out = _tmp()
    p = BRIGHTPreprocessor(output_dir=out)
    df = pd.DataFrame({'doc_id': ['d1', 'd2'], 'text': ['one', 'two']})
    path = p.prepare_tevatron_corpus(df, filename="c.jsonl")
    assert isinstance(path, Path), f"writer must return Path, got {type(path).__name__}"
    rows = _read_jsonl(path)
    assert [set(r) for r in rows] == [{'docid', 'text'}] * 2, rows[0].keys()


def test_queries_jsonl_has_exactly_queryid_and_query():
    out = _tmp()
    p = BRIGHTPreprocessor(output_dir=out)
    df = pd.DataFrame({'query_id': ['q1'], 'query': ['hello']})
    path = p.prepare_tevatron_queries(df, filename="q.jsonl")
    assert isinstance(path, Path), f"writer must return Path, got {type(path).__name__}"
    rows = _read_jsonl(path)
    assert set(rows[0]) == {'query_id', 'query'}, rows[0].keys()


def _data_args(**kw):
    from tevatron.retriever.arguments import DataArguments
    return DataArguments(**kw)


def test_tevatron_encode_dataset_reads_corpus():
    from tevatron.retriever.dataset import EncodeDataset
    out = _tmp()
    p = BRIGHTPreprocessor(output_dir=out)
    df = pd.DataFrame({'doc_id': ['d1'], 'text': ['passage body']})
    path = p.prepare_tevatron_corpus(df, filename="c.jsonl")
    ds = EncodeDataset(_data_args(dataset_path=str(path), encode_is_query=False,
                                 dataset_cache_dir=str(out / "hf")))
    cid, text, *_ = ds[0]
    assert cid == 'd1' and text == 'passage body', (cid, text)


def test_tevatron_encode_dataset_reads_queries():
    from tevatron.retriever.dataset import EncodeDataset
    out = _tmp()
    p = BRIGHTPreprocessor(output_dir=out)
    df = pd.DataFrame({'query_id': ['q1'], 'query': ['query body']})
    path = p.prepare_tevatron_queries(df, filename="q.jsonl")
    ds = EncodeDataset(_data_args(dataset_path=str(path), encode_is_query=True,
                                 dataset_cache_dir=str(out / "hf")))
    cid, text, *_ = ds[0]
    assert cid == 'q1' and text == 'query body', (cid, text)


def test_tevatron_train_dataset_reads_mixture():
    from tevatron.retriever.dataset import TrainDataset
    out = _tmp()
    mix = _simple_mixture(out / "training_mixture")
    args = _data_args(dataset_path=str(mix / "train_hq.jsonl"), train_group_size=2,
                      dataset_cache_dir=str(out / "hf"))
    ds = TrainDataset(args)
    ds.set_trainer(type('T', (), {
        'state': type('S', (), {'epoch': 0})(),
        'args': type('A', (), {'seed': 42})(),
    })())
    (query, *_), documents = ds[0]
    assert query == "hq query zero", query
    assert [d[0] for d in documents] == ["alpha text", "neg one"], documents


# ---- 2. BRIGHT loader -------------------------------------------------------

def test_qrels_gold_ids_list_only():
    loader = _loader_with(
        documents={'bio': [{'id': 'd1', 'content': 'x'}, {'id': 'd2', 'content': 'y'}]},
        examples={'bio': [{'id': 'q1', 'gold_ids': ['d1', 'd2']}]})
    qrels = loader.get_qrels('bio')
    assert len(qrels) == 2
    assert set(qrels['doc_id']) == {'d1', 'd2'}


def test_qrels_preserves_comma_in_docid():
    """58 real biology doc ids contain a comma; the old split branch shredded them."""
    loader = _loader_with(
        documents={'bio': [{'id': 'doc,with,commas', 'content': 'x'}]},
        examples={'bio': [{'id': 'q1', 'gold_ids': ['doc,with,commas']}]})
    qrels = loader.get_qrels('bio')
    assert list(qrels['doc_id']) == ['doc,with,commas'], list(qrels['doc_id'])


def test_qrels_rejects_scalar_gold_ids():
    loader = _loader_with(
        documents={'bio': [{'id': 'd1', 'content': 'x'}]},
        examples={'bio': [{'id': 'q1', 'gold_ids': 'd1'}]})
    _assert_raises(ValueError, lambda: loader.get_qrels('bio'), contains='gold_ids')


def test_qrels_deduplicates_pairs():
    """theoremqa_questions ships 178 duplicate (query_id, doc_id) pairs."""
    loader = _loader_with(
        documents={'bio': [{'id': 'd1', 'content': 'x'}]},
        examples={'bio': [{'id': 'q1', 'gold_ids': ['d1', 'd1', 'd1']}]})
    qrels = loader.get_qrels('bio')
    assert len(qrels) == 1, f"duplicates not collapsed: {len(qrels)}"


def test_qrels_rejects_blank_ids():
    loader = _loader_with(
        documents={'bio': [{'id': 'd1', 'content': 'x'}]},
        examples={'bio': [{'id': 'q1', 'gold_ids': ['   ']}]})
    _assert_raises(ValueError, lambda: loader.get_qrels('bio'))


def test_qrels_rejects_gold_id_missing_from_corpus():
    loader = _loader_with(
        documents={'bio': [{'id': 'd1', 'content': 'x'}]},
        examples={'bio': [{'id': 'q1', 'gold_ids': ['d_absent']}]})
    _assert_raises(ValueError, lambda: loader.get_qrels('bio'), contains='d_absent')


def test_corpus_rejects_conflicting_duplicate_docids():
    loader = _loader_with(
        documents={'bio': [{'id': 'd1', 'content': 'x'}, {'id': 'd1', 'content': 'DIFFERENT'}]},
        examples={'bio': []})
    _assert_raises(ValueError, lambda: loader.get_corpus('bio'), contains='d1')


def test_corpus_allows_identical_duplicate_docids():
    loader = _loader_with(
        documents={'bio': [{'id': 'd1', 'content': 'x'}, {'id': 'd1', 'content': 'x'}]},
        examples={'bio': []})
    corpus = loader.get_corpus('bio')
    assert len(corpus) == 1, len(corpus)


def test_id_map_allows_identical_duplicate_ids():
    """188,002 doc ids really are shared by aops and theoremqa_questions."""
    loader = _loader_with(documents={
        'aops': [{'id': 'shared', 'content': 'same text'}],
        'theoremqa_questions': [{'id': 'shared', 'content': 'same text'}]})
    assert loader.get_all_documents_id_map() == {'shared': 'same text'}


def test_id_map_rejects_conflicting_text():
    loader = _loader_with(documents={
        'aops': [{'id': 'shared', 'content': 'text A'}],
        'biology': [{'id': 'shared', 'content': 'text B'}]})
    _assert_raises(ValueError, loader.get_all_documents_id_map,
                   contains=['shared', 'aops', 'biology'])


def test_loader_init_does_not_download():
    """__init__ must be a pure constructor -- no hidden ReasonIR fetch."""
    import data.bright_loader as bl
    calls = []
    original = bl.load_dataset
    bl.load_dataset = lambda *a, **k: calls.append(a) or (_ for _ in ()).throw(
        AssertionError("load_dataset called from __init__"))
    try:
        BRIGHTLoader()
    finally:
        bl.load_dataset = original
    assert not calls, "BRIGHTLoader.__init__ hit the network"


def test_bright_examples_subset_comes_from_config():
    assert BRIGHTLoader().examples_config == 'examples'


def test_example_domains_require_corpora():
    loader = _loader_with(
        documents={'bio': [{'id': 'd1', 'content': 'x'}]},
        examples={'bio': [], 'orphan': [{'id': 'q1', 'gold_ids': ['d1']}]})
    _assert_raises(ValueError, loader.validate_example_domains_have_corpora,
                   contains='orphan')


# ---- 3. ReasonIR generation -------------------------------------------------

def test_hq_skips_blank_and_identical():
    out = _tmp()
    p = BRIGHTPreprocessor(output_dir=out)
    id2doc = {'bright_a': 'alpha', 'bright_b': 'beta'}
    rows = _reasonir_rows([
        ("good query", ["bright_a"], ["a negative"]),
        ("", ["bright_a"], ["a negative"]),          # blank query
        ("blank neg", ["bright_a"], ["   "]),         # blank negative
        ("pos eq neg", ["bright_a"], ["alpha"]),      # negative == positive text
        ("unmapped", ["not_in_bright"], ["a negative"]),
    ])
    path = p.prepare_hq_train_data(id2doc=id2doc, records=rows, filename="hq.jsonl")
    written = _read_jsonl(path)
    assert len(written) == 1, [r['query'] for r in written]
    assert written[0]['query'] == "good query"


def test_hq_ids_from_source_index():
    out = _tmp()
    p = BRIGHTPreprocessor(output_dir=out)
    rows = _reasonir_rows([
        ("", ["bright_a"], ["n"]),            # dropped, but still consumes index 0
        ("kept", ["bright_a"], ["n"]),
    ])
    path = p.prepare_hq_train_data(id2doc={'bright_a': 'alpha'}, records=rows,
                                   filename="hq.jsonl")
    written = _read_jsonl(path)
    assert written[0]['query_id'] == "reasonir_hq_1", written[0]['query_id']


def test_vl_ids_unique_across_multiple_passages():
    out = _tmp()
    p = BRIGHTPreprocessor(output_dir=out)
    rows = _reasonir_rows([("q", ["pos one", "pos two"], ["neg one", "neg two"])])
    path = p.prepare_vl_train_data(records=rows, filename="vl.jsonl", skip_first_n=0)
    rec = _read_jsonl(path)[0]
    ids = [d['docid'] for d in rec['positive_passages'] + rec['negative_passages']]
    assert len(set(ids)) == 4, ids


def test_vl_skip_first_n_applies():
    out = _tmp()
    p = BRIGHTPreprocessor(output_dir=out)
    rows = _reasonir_rows([(f"q{i}", ["p"], ["n"]) for i in range(5)])
    path = p.prepare_vl_train_data(records=rows, filename="vl.jsonl", skip_first_n=3)
    written = _read_jsonl(path)
    assert [r['query'] for r in written] == ["q3", "q4"], [r['query'] for r in written]
    assert written[0]['query_id'] == "reasonir_vl_3", written[0]['query_id']


def test_vl_skip_first_n_default_comes_from_config():
    from utils.helpers import load_config
    import inspect
    cfg = load_config()
    assert 'vl_skip_first_n' in cfg['data']['mixed_training'], \
        "vl_skip_first_n must live in config/config.yaml, not at the call site"
    sig = inspect.signature(BRIGHTPreprocessor.prepare_vl_train_data)
    assert sig.parameters['skip_first_n'].default is None, \
        "skip_first_n must default to None and resolve from config"


def test_generation_reports_written_and_skipped():
    import io
    from contextlib import redirect_stdout
    out = _tmp()
    p = BRIGHTPreprocessor(output_dir=out)
    rows = _reasonir_rows([("q", ["p"], ["n"]), ("blank", ["p"], ["  "])])
    buf = io.StringIO()
    with redirect_stdout(buf):
        p.prepare_vl_train_data(records=rows, filename="vl.jsonl", skip_first_n=0, limit=2)
    assert "1 written" in buf.getvalue() and "1 skipped" in buf.getvalue(), buf.getvalue()


# ---- 4. MS MARCO ------------------------------------------------------------

def _msmarco_rows(n=50):
    return [{'query': f'q{i}', 'positive': f'pos{i}', 'negative': f'neg{i}'}
            for i in range(n)]


def test_msmarco_seeded_reproducible():
    a, b = _tmp(), _tmp()
    rows = _msmarco_rows()
    p1 = BRIGHTPreprocessor(output_dir=a).prepare_msmarco_train_data(
        records=rows, filename="m.jsonl", limit=10, seed=42)
    p2 = BRIGHTPreprocessor(output_dir=b).prepare_msmarco_train_data(
        records=rows, filename="m.jsonl", limit=10, seed=42)
    assert Path(p1).read_bytes() == Path(p2).read_bytes(), "seeded output is not reproducible"


def test_msmarco_seed_changes_output():
    a, b = _tmp(), _tmp()
    rows = _msmarco_rows()
    p1 = BRIGHTPreprocessor(output_dir=a).prepare_msmarco_train_data(
        records=rows, filename="m.jsonl", limit=10, seed=42)
    p2 = BRIGHTPreprocessor(output_dir=b).prepare_msmarco_train_data(
        records=rows, filename="m.jsonl", limit=10, seed=7)
    assert Path(p1).read_bytes() != Path(p2).read_bytes(), "seed had no effect"


def test_msmarco_limit_none_takes_everything():
    out = _tmp()
    path = BRIGHTPreprocessor(output_dir=out).prepare_msmarco_train_data(
        records=_msmarco_rows(12), filename="m.jsonl", limit=None, seed=42)
    assert len(_read_jsonl(path)) == 12


def test_msmarco_missing_positive_raises():
    out = _tmp()
    rows = [{'query': 'q0', 'positive': '', 'negative': 'n'}]
    _assert_raises(PreprocessingError,
                   lambda: BRIGHTPreprocessor(output_dir=out).prepare_msmarco_train_data(
                       records=rows, filename="m.jsonl", limit=None, seed=42),
                   contains=['msmarco_0', 'positive'])


def test_msmarco_missing_negative_raises():
    out = _tmp()
    rows = [{'query': 'q0', 'positive': 'p', 'negative': '   '}]
    _assert_raises(PreprocessingError,
                   lambda: BRIGHTPreprocessor(output_dir=out).prepare_msmarco_train_data(
                       records=rows, filename="m.jsonl", limit=None, seed=42),
                   contains=['msmarco_0', 'negative'])


def test_msmarco_identical_pair_is_skipped_not_fatal():
    """22.8% of the real triplet corpus has positive == negative, starting at row 0.
    Raising there would abort generation and publish no mixture at all."""
    out = _tmp()
    rows = [{'query': 'q0', 'positive': 'same', 'negative': 'same'},
            {'query': 'q1', 'positive': 'p', 'negative': 'n'}]
    path = BRIGHTPreprocessor(output_dir=out).prepare_msmarco_train_data(
        records=rows, filename="m.jsonl", limit=None, seed=42)
    written = _read_jsonl(path)
    assert len(written) == 1 and written[0]['query'] == 'q1', written


def test_msmarco_output_loads_in_pinned_tevatron():
    from tevatron.retriever.dataset import TrainDataset
    out = _tmp()
    rows = [{'query': 'q0', 'positive': 'a positive', 'negative': 'a negative'}]
    path = BRIGHTPreprocessor(output_dir=out).prepare_msmarco_train_data(
        records=rows, filename="m.jsonl", limit=None, seed=42)
    ds = TrainDataset(_data_args(dataset_path=str(path), train_group_size=2,
                                 dataset_cache_dir=str(out / "hf")))
    ds.set_trainer(type('T', (), {'state': type('S', (), {'epoch': 0})(),
                                  'args': type('A', (), {'seed': 42})()})())
    (query, *_), documents = ds[0]
    assert query == 'q0' and [d[0] for d in documents] == ['a positive', 'a negative']


# ---- 5. run_setup invariants ------------------------------------------------

def _setup(root):
    processed = Path(root) / "data" / "processed"
    return run_setup(mixture_dir=processed / "training_mixture", output_dir=processed)


def test_run_setup_positive_first_canonical():
    """A text seen as a positive first owns the canonical docid."""
    root = _tmp()
    processed = root / "data" / "processed"
    _write_mixture(processed / "training_mixture", _fill_components({
        "train_hq.jsonl": [_mix_record("q1", "one", [("POS", "shared text")], [("N1", "n1")])],
        "train_vl.jsonl": [_mix_record("q2", "two", [("P2", "p2")], [("NEG", "shared text")])],
    }))
    corpus_path, _, _ = _setup(root)
    corpus = {r['docid']: r['text'] for r in _read_jsonl(corpus_path)}
    assert 'POS' in corpus and 'NEG' not in corpus, sorted(corpus)


def test_run_setup_qrels_remapped_to_canonical():
    root = _tmp()
    processed = root / "data" / "processed"
    _write_mixture(processed / "training_mixture", _fill_components({
        "train_hq.jsonl": [_mix_record("q1", "one", [("CANON", "shared")], [("n1", "x")])],
        "train_vl.jsonl": [_mix_record("q2", "two", [("DUPE", "shared")], [("n2", "y")])],
    }))
    corpus_path, _, qrels_path = _setup(root)
    lines = [l.split() for l in Path(qrels_path).read_text().splitlines()]
    mapped = {q: d for q, _, d, _ in lines}
    assert mapped['q2'] == 'CANON', mapped
    docids = {r['docid'] for r in _read_jsonl(corpus_path)}
    assert 'DUPE' not in docids, docids


def test_run_setup_uses_declared_file_order():
    """Components are read in MIXTURE_FILES order, not readdir order, so the docid
    that wins a duplicated text is deterministic."""
    root = _tmp()
    processed = root / "data" / "processed"
    _write_mixture(processed / "training_mixture", _fill_components({
        MIXTURE_FILES[-1]: [_mix_record("q2", "two", [("FROM_LAST", "shared")], [("n2", "y")])],
        MIXTURE_FILES[0]: [_mix_record("q1", "one", [("FROM_FIRST", "shared")], [("n1", "x")])],
    }))
    corpus_path, _, _ = _setup(root)
    docids = {r['docid'] for r in _read_jsonl(corpus_path)}
    assert 'FROM_FIRST' in docids and 'FROM_LAST' not in docids, docids


def test_stray_jsonl_is_ignored_and_kept():
    """Legacy/unrelated JSONL in the mixture dir must be neither read nor deleted."""
    root = _tmp()
    processed = root / "data" / "processed"
    mix = _simple_mixture(processed / "training_mixture")
    stray = mix / "train_reasonir_vl.jsonl"          # legacy schema, would KeyError if read
    stray.write_text(json.dumps({"query_id": "legacy", "query": "x",
                                 "positives": ["p"], "negatives": ["n"]}) + "\n")
    _, queries_path, _ = _setup(root)
    assert "legacy" not in {r['query_id'] for r in _read_jsonl(queries_path)}
    assert stray.is_file(), "a stray file was deleted"


def test_run_setup_full_coverage():
    """Every qrel docid must exist in the corpus, every query in the queries file."""
    root = _tmp()
    processed = root / "data" / "processed"
    _simple_mixture(processed / "training_mixture")
    corpus_path, queries_path, qrels_path = _setup(root)
    corpus_ids = {r['docid'] for r in _read_jsonl(corpus_path)}
    query_ids = {r['query_id'] for r in _read_jsonl(queries_path)}
    for line in Path(qrels_path).read_text().splitlines():
        qid, _, did, _ = line.split()
        assert did in corpus_ids, f"qrel docid {did} absent from corpus"
        assert qid in query_ids, f"qrel query {qid} absent from queries"


def test_run_setup_returns_paths():
    root = _tmp()
    _simple_mixture(root / "data" / "processed" / "training_mixture")
    for p in _setup(root):
        assert isinstance(p, Path), f"run_setup must return Paths, got {type(p).__name__}"


# ---- 6. Failure before mutation ---------------------------------------------

def test_legacy_schema_raises_naming_file_and_field():
    root = _tmp()
    processed = root / "data" / "processed"
    mix = _write_mixture(processed / "training_mixture", _fill_components({}))
    with open(mix / "train_vl.jsonl", 'w') as f:
        f.write(json.dumps({"query_id": "q1", "query": "x",
                            "positives": ["p"], "negatives": ["n"]}) + "\n")
    _assert_raises(PreprocessingError, lambda: _setup(root),
                   contains=['train_vl.jsonl', 'positive_passages'])


def test_duplicate_query_id_raises():
    root = _tmp()
    processed = root / "data" / "processed"
    _write_mixture(processed / "training_mixture", _fill_components({
        "train_hq.jsonl": [_mix_record("dupe_q", "one", [("p1", "a")], [("n1", "b")])],
        "train_vl.jsonl": [_mix_record("dupe_q", "two", [("p2", "c")], [("n2", "d")])],
    }))
    _assert_raises(PreprocessingError, lambda: _setup(root), contains='dupe_q')


def test_conflicting_docid_text_raises():
    root = _tmp()
    processed = root / "data" / "processed"
    _write_mixture(processed / "training_mixture", _fill_components({
        "train_hq.jsonl": [_mix_record("q1", "one", [("SAME_ID", "text A")], [("n1", "b")])],
        "train_vl.jsonl": [_mix_record("q2", "two", [("SAME_ID", "text B")], [("n2", "d")])],
    }))
    _assert_raises(PreprocessingError, lambda: _setup(root), contains='SAME_ID')


def test_existing_output_is_refused():
    """Never silently reuse or overwrite: a derived file already on disk is fatal."""
    root = _tmp()
    processed = root / "data" / "processed"
    _simple_mixture(processed / "training_mixture")
    _setup(root)
    _assert_raises(PreprocessingError, lambda: _setup(root),
                   contains=['reasonir_corpus.jsonl', 'Delete'])


def test_single_leftover_output_is_refused():
    root = _tmp()
    processed = root / "data" / "processed"
    _simple_mixture(processed / "training_mixture")
    processed.mkdir(parents=True, exist_ok=True)
    (processed / "train_qrels.txt").write_text("stale\n")
    _assert_raises(PreprocessingError, lambda: _setup(root), contains='train_qrels.txt')


# ---- 7. Transactional publication -------------------------------------------

def _break_qrels(monkey_target, boom="qrels exploded"):
    """Make qrels writing fail, leaving corpus and queries already written."""
    import data.preprocessor as pp
    original = pp.BRIGHTPreprocessor.prepare_trec_qrels

    def fail(self, *a, **k):
        raise RuntimeError(boom)
    pp.BRIGHTPreprocessor.prepare_trec_qrels = fail
    return lambda: setattr(pp.BRIGHTPreprocessor, 'prepare_trec_qrels', original)


def test_qrels_write_failure_publishes_nothing():
    root = _tmp()
    processed = root / "data" / "processed"
    _simple_mixture(processed / "training_mixture")
    restore = _break_qrels(None)
    try:
        _assert_raises(RuntimeError, lambda: _setup(root), contains='qrels exploded')
    finally:
        restore()
    for name in ("reasonir_corpus.jsonl", "train_queries.jsonl", "train_qrels.txt"):
        assert not (processed / name).exists(), f"{name} survived a failed build"


def test_publish_failure_rolls_back_earlier_outputs():
    """Failure *during* publication must remove the files already published."""
    import data.preprocessor as pp
    root = _tmp()
    processed = root / "data" / "processed"
    _simple_mixture(processed / "training_mixture")
    original = pp.os.replace
    seen = []

    def flaky(src, dst):
        if str(dst).endswith("train_qrels.txt"):
            raise OSError("publish exploded")
        seen.append(str(dst))
        return original(src, dst)

    pp.os.replace = flaky
    try:
        _assert_raises(OSError, lambda: _setup(root), contains='publish exploded')
    finally:
        pp.os.replace = original
    assert seen, "test did not reach publication"
    for name in ("reasonir_corpus.jsonl", "train_queries.jsonl", "train_qrels.txt"):
        assert not (processed / name).exists(), f"{name} survived a failed publish"


def test_retry_after_failure_succeeds():
    root = _tmp()
    processed = root / "data" / "processed"
    _simple_mixture(processed / "training_mixture")
    restore = _break_qrels(None)
    try:
        _assert_raises(RuntimeError, lambda: _setup(root))
    finally:
        restore()
    corpus_path, queries_path, qrels_path = _setup(root)
    assert all(Path(p).stat().st_size > 0 for p in (corpus_path, queries_path, qrels_path))


def test_failed_build_leaves_no_staging_dir():
    root = _tmp()
    processed = root / "data" / "processed"
    _simple_mixture(processed / "training_mixture")
    restore = _break_qrels(None)
    try:
        _assert_raises(RuntimeError, lambda: _setup(root))
    finally:
        restore()
    leftovers = [q.name for q in processed.iterdir() if q.is_dir() and q.name != "training_mixture"]
    assert not leftovers, f"staging left behind: {leftovers}"


# ---- 8. atomic_write --------------------------------------------------------

def test_atomic_write_concurrent_writers_do_not_collide():
    """Two writers to one destination must not share a temp path: the loser used to
    truncate the winner's file mid-write."""
    import threading
    d = Path(tempfile.mkdtemp())
    target = d / "out.txt"
    payloads = ["A" * 5000, "B" * 5000]
    barrier = threading.Barrier(2)

    def write(text):
        with atomic_write(target) as f:
            f.write(text[:100])
            barrier.wait()          # both writers are mid-write simultaneously
            f.write(text[100:])

    threads = [threading.Thread(target=write, args=(t,)) for t in payloads]
    for t in threads: t.start()
    for t in threads: t.join()

    assert target.read_text() in payloads, "final file is not exactly one payload"
    assert [q.name for q in d.iterdir()] == ["out.txt"], list(d.iterdir())


# ---- 9. Build vs consume ----------------------------------------------------

def test_mixed_mode_rejects_a_missing_component():
    """The experiment is declared, not inferred from whichever files survived."""
    root = _tmp()
    processed = root / "data" / "processed"
    _write_mixture(processed / "training_mixture", {
        "train_hq.jsonl": [_mix_record("q1", "one", [("p1", "a")], [("n1", "b")])],
        "train_vl.jsonl": [_mix_record("q2", "two", [("p2", "c")], [("n2", "d")])],
    })
    _assert_raises(PreprocessingError, lambda: _setup(root),
                   contains='train_msmarco.jsonl')


def test_msmarco_only_mode_accepts_its_single_file():
    root = _tmp()
    processed = root / "data" / "processed"
    _write_mixture(processed / "training_mixture", {
        "train_msmarco.jsonl": [_mix_record("m1", "q", [("p", "a")], [("n", "b")])],
        "train_vl.jsonl": [_mix_record("v1", "q", [("vp", "c")], [("vn", "d")])],
    })
    _, queries_path, _ = run_setup(mixture_dir=processed / "training_mixture",
                                   output_dir=processed,
                                   expected_files=MSMARCO_ONLY_FILES)
    assert {r['query_id'] for r in _read_jsonl(queries_path)} == {'m1'}, "read a non-MS file"


def test_require_derived_artifacts_after_a_build():
    root = _tmp()
    processed = root / "data" / "processed"
    _simple_mixture(processed / "training_mixture")
    built = _setup(root)
    assert require_derived_artifacts(processed) == tuple(built)


def test_require_derived_artifacts_reports_what_is_missing():
    root = _tmp()
    processed = root / "data" / "processed"
    _simple_mixture(processed / "training_mixture")
    _setup(root)
    (processed / "train_qrels.txt").write_text("")          # incomplete, not absent
    _assert_raises(PreprocessingError,
                   lambda: require_derived_artifacts(processed),
                   contains=['train_qrels.txt', 'empty'])


def test_consumers_never_import_the_builder():
    """Training and mining read derived artifacts; only the preprocessor builds them."""
    consumers = [project_root / 'scripts' / n for n in (
        'run_grass.py', 'run_fast_grass.py', 'train_async_fast_grass.py',
        'refresh_stale_index.py', 'train_ance.py')]
    consumers += sorted((project_root / 'scripts' / 'dev').glob('*.py'))
    offenders = []
    for path in consumers:
        text = path.read_text()
        if 'consolidation_preproc_check' in path.name:
            continue                                        # the fixture builds on purpose
        if 'import run_setup' in text or 'preprocessor import run_setup' in text:
            offenders.append(path.name)
    assert not offenders, f"still import the builder: {offenders}"


def test_require_mixture_files_is_recipe_aware_and_strict():
    root = _tmp()
    mix = _simple_mixture(root / "training_mixture")
    assert tuple(p.name for p in require_mixture_files(mix, MIXTURE_FILES)) == MIXTURE_FILES
    assert tuple(p.name for p in require_mixture_files(
        mix, MSMARCO_ONLY_FILES, reject_unexpected=False)) == MSMARCO_ONLY_FILES

    (mix / "legacy.jsonl").write_text("{}\n")
    _assert_raises(PreprocessingError,
                   lambda: require_mixture_files(mix, MIXTURE_FILES),
                   contains='legacy.jsonl')


def test_mixed_training_consumers_validate_the_file_set():
    consumers = [project_root / 'scripts' / name for name in (
        'train_inbatch.py', 'train_crossbatch.py', 'run_grass.py',
        'run_fast_grass.py', 'train_async_fast_grass.py', 'train_ance.py')]
    missing = [path.name for path in consumers
               if 'require_mixture_files' not in path.read_text()]
    assert not missing, f"mixed-data consumers do not validate their file set: {missing}"


def test_ance_msmarco_partial_train_set_is_rebuilt():
    sys.path.insert(0, str(project_root / 'scripts'))
    import train_ance

    root = _tmp()
    mixture_dir = root / 'msmarco_training_mixture'
    mixture_dir.mkdir()
    (root / 'msmarco_corpus.jsonl').write_text('{}\n')
    (mixture_dir / 'train_msmarco.jsonl').write_text('{}\n')

    calls = []
    original_get_path = train_ance.get_path
    original_prepare = train_ance.BRIGHTPreprocessor.prepare_msmarco_tevatron_train

    def fake_prepare(self, **_kwargs):
        calls.append(True)
        mixture = self.output_dir / 'msmarco_training_mixture' / 'train_msmarco.jsonl'
        mixture.parent.mkdir(parents=True, exist_ok=True)
        mixture.write_text('{}\n')
        queries = self.output_dir / 'msmarco_train_queries.jsonl'
        qrels = self.output_dir / 'msmarco_train_qrels.txt'
        queries.write_text('{}\n')
        qrels.write_text('q Q0 d 1\n')
        return mixture, queries, qrels

    train_ance.get_path = lambda _key: root
    train_ance.BRIGHTPreprocessor.prepare_msmarco_tevatron_train = fake_prepare
    try:
        paths = train_ance.run_setup({
            'setup_mode': 'tevatron_msmarco',
            'corpus_file': 'msmarco_corpus.jsonl',
            'train_queries_file': 'msmarco_train_queries.jsonl',
            'train_qrels_file': 'msmarco_train_qrels.txt',
            'mixture_dir': 'msmarco_training_mixture',
            'eval_queries_file': None,
        })
    finally:
        train_ance.get_path = original_get_path
        train_ance.BRIGHTPreprocessor.prepare_msmarco_tevatron_train = original_prepare

    assert calls, "partial MS MARCO train artifacts were not rebuilt together"
    assert all(path.is_file() and path.stat().st_size for path in paths), paths


# ---- 10. MS MARCO count enforcement -----------------------------------------

def test_msmarco_reaches_requested_count_despite_collisions():
    """Collisions are skipped, and scanning continues until `limit` are emitted."""
    rows = []
    for i in range(40):
        rows.append({'query': f'q{i}', 'positive': f'p{i}',
                     'negative': (f'p{i}' if i % 2 else f'n{i}')})   # half collide
    path = BRIGHTPreprocessor(output_dir=_tmp()).prepare_msmarco_train_data(
        records=rows, filename="m.jsonl", limit=20, seed=42)
    assert len(_read_jsonl(path)) == 20


def test_msmarco_exhaustion_raises_and_publishes_nothing():
    out = _tmp()
    rows = [{'query': f'q{i}', 'positive': 'same', 'negative': 'same'} for i in range(5)]
    rows.append({'query': 'ok', 'positive': 'p', 'negative': 'n'})
    _assert_raises(PreprocessingError,
                   lambda: BRIGHTPreprocessor(output_dir=out).prepare_msmarco_train_data(
                       records=rows, filename="m.jsonl", limit=4, seed=42),
                   contains=['requested', 'inspected', 'skipped'])
    assert not (out / "m.jsonl").exists(), "underfilled output was published"


def test_msmarco_blank_query_raises_and_publishes_nothing():
    out = _tmp()
    _assert_raises(PreprocessingError,
                   lambda: BRIGHTPreprocessor(out).prepare_msmarco_train_data(
                       records=[{'query': '  ', 'positive': 'p', 'negative': 'n'}],
                       filename="m.jsonl", limit=1), contains='query')
    assert not (out / "m.jsonl").exists()


def test_full_generation_refuses_existing_derived_before_source_work():
    import data.preprocessor as pp
    out = _tmp()
    (out / pp.CORPUS_FILE).write_text('old\n')
    calls = []
    original = pp.BRIGHTPreprocessor.prepare_msmarco_train_data
    def source_started(*_a, **_k):
        calls.append(True)
        raise AssertionError("source generation should not start")
    pp.BRIGHTPreprocessor.prepare_msmarco_train_data = source_started
    config = {'data': {'mixed_training': {'msmarco_samples': 1, 'vl_samples': 1,
                                           'hq_samples': 1},
                       'msmarco': {'name': 'ms', 'subset': 'triplet'},
                       'reasonir': {'name': 'reasonir'}}}
    try:
        _assert_raises(PreprocessingError,
                       lambda: pp._generate_training_mixture(
                           BRIGHTPreprocessor(out), object(), config),
                       contains='derived')
    finally:
        pp.BRIGHTPreprocessor.prepare_msmarco_train_data = original
    assert not calls, "source generation started before the clean-target preflight"


# ---- 11. MS MARCO reproduction track ----------------------------------------

def test_reproduction_skips_records_without_negatives():
    from data.preprocessor import _reproduction_record_ok
    assert _reproduction_record_ok({'query': 'q',
                                    'positive_passages': [{'docid': 'a', 'text': 't'}],
                                    'negative_passages': [{'docid': 'b', 'text': 'u'}]})
    assert not _reproduction_record_ok({'query': ' ', 'positive_passages': [1],
                                        'negative_passages': [2]})
    assert not _reproduction_record_ok({'positive_passages': [], 'negative_passages': [1]})
    assert not _reproduction_record_ok({'positive_passages': [1], 'negative_passages': []})
    assert not _reproduction_record_ok({'positive_passages': [1]})


def test_reproduction_records_load_in_pinned_tevatron():
    """train_group_size=2 needs a real negative; an empty list used to reach Tevatron."""
    import data.preprocessor as pp
    from tevatron.retriever.dataset import TrainDataset
    out = _tmp()
    records = [
        {"query_id": "bad", "query": "q", "positive_passages": [
            {"docid": "p0", "text": "positive"}], "negative_passages": []},
        {"query_id": "good", "query": "q", "positive_passages": [
            {"docid": "p1", "text": "a positive"}], "negative_passages": [
            {"docid": "n1", "text": "a negative"}]},
    ]
    original = pp.load_dataset
    pp.load_dataset = lambda *_a, **_k: records
    try:
        path, _, _ = BRIGHTPreprocessor(out).prepare_msmarco_tevatron_train()
    finally:
        pp.load_dataset = original

    exported = _read_jsonl(path)
    assert [row['query_id'] for row in exported] == ['good'], exported
    ds = TrainDataset(_data_args(dataset_path=str(path), train_group_size=2,
                                 dataset_cache_dir=str(out / "hf")))
    ds.set_trainer(type('T', (), {'state': type('S', (), {'epoch': 0})(),
                                  'args': type('A', (), {'seed': 42})()})())
    (_q, *_), documents = ds[0]
    assert [d[0] for d in documents] == ['a positive', 'a negative']


# ---- runner -----------------------------------------------------------------

TESTS = [
    ("tevatron: corpus JSONL is exactly {docid,text}", test_corpus_jsonl_has_exactly_docid_and_text),
    ("tevatron: queries JSONL is exactly {query_id,query}", test_queries_jsonl_has_exactly_queryid_and_query),
    ("tevatron: EncodeDataset resolves corpus", test_tevatron_encode_dataset_reads_corpus),
    ("tevatron: EncodeDataset resolves queries", test_tevatron_encode_dataset_reads_queries),
    ("tevatron: TrainDataset resolves mixture", test_tevatron_train_dataset_reads_mixture),

    ("bright: gold_ids list handled", test_qrels_gold_ids_list_only),
    ("bright: comma in docid survives", test_qrels_preserves_comma_in_docid),
    ("bright: scalar gold_ids rejected", test_qrels_rejects_scalar_gold_ids),
    ("bright: duplicate qrel pairs collapsed", test_qrels_deduplicates_pairs),
    ("bright: blank gold id rejected", test_qrels_rejects_blank_ids),
    ("bright: gold id absent from corpus rejected", test_qrels_rejects_gold_id_missing_from_corpus),
    ("bright: conflicting duplicate docid rejected", test_corpus_rejects_conflicting_duplicate_docids),
    ("bright: identical duplicate docid collapsed", test_corpus_allows_identical_duplicate_docids),
    ("bright: id map allows identical dupes", test_id_map_allows_identical_duplicate_ids),
    ("bright: id map rejects conflicts", test_id_map_rejects_conflicting_text),
    ("bright: __init__ does not download", test_loader_init_does_not_download),
    ("bright: configured examples subset", test_bright_examples_subset_comes_from_config),
    ("bright: every examples domain has a corpus", test_example_domains_require_corpora),

    ("hq: blank/identical rows skipped", test_hq_skips_blank_and_identical),
    ("hq: ids derived from source index", test_hq_ids_from_source_index),
    ("vl: ids unique across passages", test_vl_ids_unique_across_multiple_passages),
    ("vl: skip_first_n applied", test_vl_skip_first_n_applies),
    ("vl: skip_first_n default from config", test_vl_skip_first_n_default_comes_from_config),
    ("gen: reports written and skipped", test_generation_reports_written_and_skipped),

    ("msmarco: seeded output reproducible", test_msmarco_seeded_reproducible),
    ("msmarco: seed changes output", test_msmarco_seed_changes_output),
    ("msmarco: limit=None takes everything", test_msmarco_limit_none_takes_everything),
    ("msmarco: missing positive raises", test_msmarco_missing_positive_raises),
    ("msmarco: missing negative raises", test_msmarco_missing_negative_raises),
    ("msmarco: identical pair skipped", test_msmarco_identical_pair_is_skipped_not_fatal),
    ("msmarco: output loads in tevatron", test_msmarco_output_loads_in_pinned_tevatron),

    ("run_setup: positive-first canonical id", test_run_setup_positive_first_canonical),
    ("run_setup: qrels remapped to canonical", test_run_setup_qrels_remapped_to_canonical),
    ("run_setup: declared file order", test_run_setup_uses_declared_file_order),
    ("run_setup: stray jsonl ignored, kept", test_stray_jsonl_is_ignored_and_kept),
    ("run_setup: corpus/query coverage complete", test_run_setup_full_coverage),
    ("run_setup: returns Paths", test_run_setup_returns_paths),

    ("fail: legacy schema names file+field", test_legacy_schema_raises_naming_file_and_field),
    ("fail: duplicate query_id", test_duplicate_query_id_raises),
    ("fail: conflicting docid text", test_conflicting_docid_text_raises),
    ("fail: existing output refused", test_existing_output_is_refused),
    ("fail: single leftover output refused", test_single_leftover_output_is_refused),

    ("txn: qrels write failure publishes nothing", test_qrels_write_failure_publishes_nothing),
    ("txn: publish failure rolls back", test_publish_failure_rolls_back_earlier_outputs),
    ("txn: retry after failure succeeds", test_retry_after_failure_succeeds),
    ("txn: no staging dir left behind", test_failed_build_leaves_no_staging_dir),
    ("atomic_write: concurrent writers", test_atomic_write_concurrent_writers_do_not_collide),

    ("mode: mixed rejects missing component", test_mixed_mode_rejects_a_missing_component),
    ("mode: ms-only accepts one file", test_msmarco_only_mode_accepts_its_single_file),
    ("consume: verifier accepts a built set", test_require_derived_artifacts_after_a_build),
    ("consume: verifier reports what is missing", test_require_derived_artifacts_reports_what_is_missing),
    ("consume: trainers never import the builder", test_consumers_never_import_the_builder),
    ("consume: mixture validator is recipe-aware", test_require_mixture_files_is_recipe_aware_and_strict),
    ("consume: every mixed trainer validates files", test_mixed_training_consumers_validate_the_file_set),
    ("ance-ms: partial train set rebuilt", test_ance_msmarco_partial_train_set_is_rebuilt),

    ("msmarco: reaches count despite collisions", test_msmarco_reaches_requested_count_despite_collisions),
    ("msmarco: exhaustion raises, publishes nothing", test_msmarco_exhaustion_raises_and_publishes_nothing),
    ("msmarco: blank query raises, publishes nothing", test_msmarco_blank_query_raises_and_publishes_nothing),
    ("gen: existing derived output blocks source work", test_full_generation_refuses_existing_derived_before_source_work),
    ("repro: records without negatives skipped", test_reproduction_skips_records_without_negatives),
    ("repro: records load in pinned tevatron", test_reproduction_records_load_in_pinned_tevatron),
]


def _run(name, fn):
    try:
        fn()
    except Exception as e:                                    # noqa: BLE001
        print(f"  ❌ {name}\n       {type(e).__name__}: {e}")
        if os.environ.get("PREPROC_TEST_TRACE"):
            traceback.print_exc()
        return False
    print(f"  ✅ {name}")
    return True


def main():
    print("\nPreprocessor / BRIGHT-loader unit tests")
    print("=" * 62)
    passed = sum(_run(name, fn) for name, fn in TESTS)
    total = len(TESTS)
    print("=" * 62)
    print(f"  {passed}/{total} passed")
    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
