"""Preprocess BRIGHT / ReasonIR data for Tevatron.

Writes the files fine-tuning opens: the training mixture, the per-domain BRIGHT eval
files, and the corpus/queries/qrels derived from the mixture for hard-negative mining.
"""

import argparse
import hashlib
import json
import random
import os
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Dict, Iterable, Optional
from urllib.parse import quote

import pandas as pd
from datasets import load_dataset

project_root = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(project_root / 'src'))

from utils.helpers import load_config, get_path, atomic_write

CORPUS_FILE = "reasonir_corpus.jsonl"
QUERIES_FILE = "train_queries.jsonl"
QRELS_FILE = "train_qrels.txt"

REQUIRED_MIXTURE_FIELDS = ("query_id", "query", "positive_passages", "negative_passages")

# The component files run_setup derives from, in read order. Explicit rather than a
# glob: a stray or legacy *.jsonl left in the directory used to be silently ingested.
MIXTURE_FILES = ("train_hq.jsonl", "train_msmarco.jsonl", "train_vl.jsonl")

# An MS-MARCO-only experiment declares itself; it is never inferred from the fact
# that the other components happen to be absent.
MSMARCO_ONLY_FILES = ("train_msmarco.jsonl",)

REBUILD_HINT = "regenerate with `python src/data/preprocessor.py`"


class PreprocessingError(RuntimeError):
    """The inputs cannot be preprocessed as given."""


# ---- small shared helpers ---------------------------------------------------

def _text(value) -> str:
    """Convert a value to text, treating a missing value as empty text."""
    return "" if value is None else str(value)


def _reasonir_query(entry) -> str:
    """Return the query text from ReasonIR's [instruction, query] value."""
    seq = entry.get("query", [])
    if isinstance(seq, (list, tuple)):
        return _text(seq[-1]) if seq else ""
    return _text(seq)


def _reasonir_payloads(entry, key) -> list:
    """Take the value from each ReasonIR [instruction, value] pair.

    VL values are passage text. HQ positive values are BRIGHT document IDs.
    """
    return [_text(item[1] if isinstance(item, (list, tuple)) else item)
            for item in (entry.get(key) or [])]


def _reproduction_record_ok(entry) -> bool:
    """Check that an MS MARCO reproduction row is usable by Tevatron.

    A group size of two requires a query, a positive, and a negative.
    """
    return bool(_text(entry.get('query')).strip() and entry.get('positive_passages') and entry.get('negative_passages'))


def _is_valid(query, positives, negatives) -> bool:
    """Reject a training row with blanks or the same text on both sides."""
    if not query.strip() or not positives or not negatives:
        return False
    if any(not t.strip() for t in positives + negatives):
        return False
    return not ({t.strip() for t in positives} & {t.strip() for t in negatives})


def _trec_safe_docid(value) -> str:
    """Escape a document ID only when whitespace would break TREC columns."""
    docid = _text(value)
    return f"trec:{quote(docid, safe='')}" if any(c.isspace() for c in docid) else docid


class BRIGHTPreprocessor:
    """Write BRIGHT / ReasonIR data in the formats Tevatron and Pyserini read."""

    def __init__(self, output_dir: Optional[str] = None):
        self.output_dir = Path(output_dir) if output_dir else get_path("processed")
        self.output_dir.mkdir(parents=True, exist_ok=True)

    # ---- writers ------------------------------------------------------------

    def prepare_tevatron_corpus(self, corpus: pd.DataFrame,
                                filename: str = "corpus.jsonl") -> Path:
        """Corpus JSONL for Tevatron's EncodeDataset: exactly {docid, text}."""
        output_path = self.output_dir / filename
        print(f"Processing {len(corpus):,} documents for {filename}...", flush=True)
        with atomic_write(output_path) as f:
            for _, row in corpus.iterrows():
                doc = {"docid": str(row['doc_id']),
                       "text": row['text'] if pd.notna(row['text']) else ""}
                f.write(json.dumps(doc, ensure_ascii=False) + '\n')
        return output_path

    def prepare_tevatron_queries(self, queries: pd.DataFrame,
                                 filename: str = "queries.jsonl") -> Path:
        """Query JSONL for Tevatron's EncodeDataset: exactly {query_id, query}."""
        output_path = self.output_dir / filename
        print(f"Processing {len(queries):,} queries for {filename}...", flush=True)
        with atomic_write(output_path) as f:
            for _, row in queries.iterrows():
                item = {"query_id": str(row['query_id']),
                        "query": row['query'] if pd.notna(row['query']) else ""}
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
        return output_path

    def prepare_pyserini_corpus(self, corpus: pd.DataFrame, output_dir) -> Path:
        """Corpus JSONL in Pyserini/Lucene format {"id", "contents"}."""
        output_path = Path(output_dir) / "corpus.jsonl"
        print(f"Writing {len(corpus):,} documents in Pyserini format to {output_path}...",
              flush=True)
        with atomic_write(output_path) as f:
            for _, row in corpus.iterrows():
                doc = {"id": str(row['doc_id']),
                       "contents": row['text'] if pd.notna(row['text']) else ""}
                f.write(json.dumps(doc, ensure_ascii=False) + '\n')
        return output_path

    def prepare_trec_qrels(self, qrels: pd.DataFrame,
                           filename: str = "qrels.txt") -> Path:
        """TREC qrels: `query_id Q0 doc_id relevance`."""
        output_path = self.output_dir / filename
        with atomic_write(output_path) as f:
            for _, row in qrels.iterrows():
                qid, docid = str(row['query_id']), str(row['doc_id'])
                if (not qid or not docid or any(c.isspace() for c in qid) or
                        any(c.isspace() for c in docid)):
                    raise PreprocessingError(
                        f"TREC qrels identifiers must be nonblank single tokens: "
                        f"query_id={qid!r}, doc_id={docid!r}")
                f.write(f"{qid} Q0 {docid} "
                        f"{int(row.get('relevance', 1))}\n")
        print(f"Saved TREC qrels to {output_path}", flush=True)
        return output_path

    def prepare_bright_excluded(self, excluded: Dict[str, list],
                                filename: str = "excluded.json") -> Path:
        """Per-query excluded doc ids, read back by every BRIGHT evaluation path."""
        output_path = self.output_dir / filename
        with atomic_write(output_path) as f:
            json.dump(excluded, f)
        print(f"Saved {len(excluded):,} query exclusion lists to {output_path}", flush=True)
        return output_path

    # ---- training-mixture generators ---------------------------------------

    def prepare_hq_train_data(self,
                              id2doc: Dict[str, str],
                              dataset_name: str = "reasonir/reasonir-data",
                              cache_dir: Optional[str] = None,
                              filename: str = "train_hq.jsonl",
                              limit: Optional[int] = None,
                              records: Optional[Iterable] = None) -> Path:
        """HQ split: positives are BRIGHT doc ids resolved through `id2doc`.

        Ids come from the source row index and the passage index, so they stay unique
        however many passages a row carries.
        """
        output_path = self.output_dir / filename
        if records is None:
            print("📥 Loading ReasonIR HQ dataset...", flush=True)
            cache = Path(cache_dir) if cache_dir else get_path("bright")
            records = load_dataset(dataset_name, "hq", cache_dir=str(cache))['train']

        written = skipped = 0
        with atomic_write(output_path) as f:
            for idx, entry in enumerate(records):
                if limit is not None and written >= limit:
                    break
                query = _reasonir_query(entry)
                doc_ids = _reasonir_payloads(entry, 'pos')
                positives = [(d, id2doc[d]) for d in doc_ids if d in id2doc]
                negatives = _reasonir_payloads(entry, 'neg')

                if not _is_valid(query, [t for _, t in positives], negatives):
                    skipped += 1
                    continue

                f.write(json.dumps({
                    "query_id": f"reasonir_hq_{idx}",
                    "query": query,
                    "positive_passages": [{"docid": str(d), "text": t}
                                          for d, t in positives],
                    "negative_passages": [{"docid": f"hq_neg_{idx}_{j}", "text": t}
                                          for j, t in enumerate(negatives)],
                }, ensure_ascii=False) + '\n')
                written += 1

        print(f"✅ HQ: {written:,} written, {skipped:,} skipped", flush=True)
        return output_path

    def prepare_vl_train_data(self,
                              dataset_name: str = "reasonir/reasonir-data",
                              cache_dir: Optional[str] = None,
                              filename: str = "train_vl.jsonl",
                              limit: Optional[int] = None,
                              skip_first_n: Optional[int] = None,
                              records: Optional[Iterable] = None) -> Path:
        """VL split: passages are direct text, no BRIGHT mapping.

        `skip_first_n` defaults to `data.mixed_training.vl_skip_first_n` in
        config.yaml -- the leading rows of the VL split are corrupted.
        """
        if skip_first_n is None:
            skip_first_n = load_config()['data']['mixed_training']['vl_skip_first_n']

        output_path = self.output_dir / filename
        if records is None:
            print("📥 Loading ReasonIR VL dataset...", flush=True)
            cache = Path(cache_dir) if cache_dir else get_path("bright")
            records = load_dataset(dataset_name, "vl", cache_dir=str(cache))['train']
        if skip_first_n:
            print(f"   ⚠️  Skipping first {skip_first_n:,} corrupted samples...", flush=True)

        written = skipped = 0
        with atomic_write(output_path) as f:
            for idx, entry in enumerate(records):
                if idx < skip_first_n:
                    continue
                if limit is not None and written >= limit:
                    break
                query = _reasonir_query(entry)
                positives = _reasonir_payloads(entry, 'pos')
                negatives = _reasonir_payloads(entry, 'neg')

                if not _is_valid(query, positives, negatives):
                    skipped += 1
                    continue

                f.write(json.dumps({
                    "query_id": f"reasonir_vl_{idx}",
                    "query": query,
                    "positive_passages": [{"docid": f"vl_pos_{idx}_{j}", "text": t}
                                          for j, t in enumerate(positives)],
                    "negative_passages": [{"docid": f"vl_neg_{idx}_{j}", "text": t}
                                          for j, t in enumerate(negatives)],
                }, ensure_ascii=False) + '\n')
                written += 1

        print(f"✅ VL: {written:,} written, {skipped:,} skipped", flush=True)
        return output_path

    def prepare_msmarco_train_data(self,
                                   dataset_name: str = "sentence-transformers/msmarco-hard-negatives",
                                   subset: str = "triplet",
                                   cache_dir: Optional[str] = None,
                                   filename: str = "train_msmarco.jsonl",
                                   limit: Optional[int] = None,
                                   records: Optional[Iterable] = None,
                                   seed: Optional[int] = None) -> Path:
        """MS MARCO triplets, sampled with a local `random.Random(seed)` so the slice
        is reproducible and cannot be perturbed by an unrelated `random.*` call."""
        if seed is None:
            seed = load_config().get('seed', 42)

        output_path = self.output_dir / filename
        if records is None:
            print(f"📥 Loading MS MARCO dataset ({subset})...", flush=True)
            cache = Path(cache_dir) if cache_dir else get_path("bright")
            records = load_dataset(dataset_name, subset, split='train',
                                   cache_dir=str(cache))

        indices = list(range(len(records)))
        random.Random(seed).shuffle(indices)

        written = skipped = inspected = 0
        with atomic_write(output_path) as f:
            for idx in indices:
                if limit is not None and written >= limit:
                    break
                inspected += 1
                entry = records[idx]
                query = _text(entry['query'])
                positive = _text(entry['positive'])
                negative = _text(entry['negative'])

                if not query.strip() or not positive.strip():
                    missing = "query" if not query.strip() else "positive passage"
                    raise PreprocessingError(
                        f"msmarco_{idx}: record has no {missing}")
                if not negative.strip():
                    raise PreprocessingError(
                        f"msmarco_{idx}: record has no negative passage")
                if positive.strip() == negative.strip():
                    # 22.8% of the triplet corpus, starting at row 0. Raising here
                    # would abort generation and publish no mixture at all.
                    skipped += 1
                    continue

                f.write(json.dumps({
                    "query_id": f"msmarco_{idx}",
                    "query": query,
                    "positive_passages": [{"docid": f"msmarco_pos_{idx}", "text": positive}],
                    "negative_passages": [{"docid": f"msmarco_neg_{idx}", "text": negative}],
                }, ensure_ascii=False) + '\n')
                written += 1

            if limit is not None and written < limit:
                # Raised inside atomic_write, so the underfilled file is discarded
                # rather than published as if it were the requested slice.
                raise PreprocessingError(
                    f"MS MARCO: requested {limit:,} records but only {written:,} were "
                    f"usable; inspected all {inspected:,} source records and skipped "
                    f"{skipped:,} where the negative equalled the positive")

        print(f"✅ MS MARCO: {written:,} written, {skipped:,} skipped", flush=True)
        return output_path

    # ---- MS MARCO reproduction track ---------------------------------------

    def prepare_msmarco_full_corpus(self,
                                    dataset_name: str = "Tevatron/msmarco-passage-corpus",
                                    cache_dir: Optional[str] = None,
                                    filename: str = "msmarco_corpus.jsonl") -> Path:
        """Write all 8.8M MS MARCO passages with real passage IDs for FAISS indexing."""
        cache = Path(cache_dir) if cache_dir else get_path("bright")
        print("📥 Loading MS MARCO full corpus (~8.8M passages)...", flush=True)
        dataset = load_dataset(dataset_name, split='train', cache_dir=str(cache),
                               trust_remote_code=True)
        corpus_df = pd.DataFrame({'doc_id': dataset['docid'], 'text': dataset['text']})
        print(f"   Loaded {len(corpus_df):,} passages", flush=True)
        return self.prepare_tevatron_corpus(corpus_df, filename=filename)

    def prepare_msmarco_tevatron_train(self,
                                       dataset_name: str = "Tevatron/msmarco-passage",
                                       cache_dir: Optional[str] = None,
                                       mixture_filename: str = "msmarco_training_mixture/train_msmarco.jsonl",
                                       queries_filename: str = "msmarco_train_queries.jsonl",
                                       qrels_filename: str = "msmarco_train_qrels.txt") -> tuple:
        """From the Tevatron/msmarco-passage train split, write the training mixture,
        the train queries and the train qrels.

        Uses streaming=True: the underlying parquet files mix train records (with
        positive_passages/negative_passages) and dev records (query_id/query only), so
        building the Arrow cache raises DatasetGenerationCastError. Records without
        positive_passages are skipped.
        """
        cache = Path(cache_dir) if cache_dir else get_path("bright")
        print("📥 Streaming Tevatron/msmarco-passage train split...", flush=True)

        mixture_path = self.output_dir / mixture_filename
        queries = {}
        qrel_rows = []
        count = 0
        skipped = 0

        stream = load_dataset(dataset_name, split='train', cache_dir=str(cache),
                              trust_remote_code=True, streaming=True)
        with atomic_write(mixture_path) as f:
            for entry in stream:
                if not _reproduction_record_ok(entry):
                    skipped += 1
                    continue
                pos = entry['positive_passages']
                qid = str(entry['query_id'])
                f.write(json.dumps({
                    "query_id": qid,
                    "query": entry['query'],
                    "positive_passages": pos,
                    "negative_passages": entry['negative_passages'],
                }, ensure_ascii=False) + '\n')
                queries[qid] = entry['query']
                for p in pos:
                    qrel_rows.append({'query_id': qid, 'doc_id': str(p['docid']),
                                      'relevance': 1})
                count += 1
                if count % 10000 == 0:
                    print(f"   {count:,} / ~400,782 records written...", flush=True)

        print(f"   Wrote {count:,} training records (skipped {skipped:,} "
              f"schema-mismatch rows) → {mixture_path}", flush=True)

        q_path = self.prepare_tevatron_queries(
            pd.DataFrame([{'query_id': k, 'query': v} for k, v in queries.items()]),
            filename=queries_filename)
        qr_path = self.prepare_trec_qrels(
            pd.DataFrame(qrel_rows).drop_duplicates(), filename=qrels_filename)
        return mixture_path, q_path, qr_path

    def prepare_msmarco_dev(self,
                            dataset_name: str = "Tevatron/msmarco-passage",
                            cache_dir: Optional[str] = None) -> tuple:
        """Write dev queries JSONL and dev qrels for MRR@10 evaluation.

        Streams for the same mixed-schema reason as prepare_msmarco_tevatron_train.
        """
        cache = Path(cache_dir) if cache_dir else get_path("bright")
        print("📥 Streaming Tevatron/msmarco-passage dev split...", flush=True)

        queries = {}
        qrel_rows = []
        stream = load_dataset(dataset_name, split='validation', cache_dir=str(cache),
                              trust_remote_code=True, streaming=True)
        for entry in stream:
            qid = str(entry['query_id'])
            queries[qid] = entry['query']
            for pos in entry.get('positive_passages') or []:
                qrel_rows.append({'query_id': qid, 'doc_id': str(pos['docid']),
                                  'relevance': 1})

        print(f"   Streamed {len(queries):,} dev queries", flush=True)
        q_path = self.prepare_tevatron_queries(
            pd.DataFrame([{'query_id': k, 'query': v} for k, v in queries.items()]),
            filename="msmarco_dev_queries.jsonl")

        if qrel_rows:
            qr_path = self.prepare_trec_qrels(
                pd.DataFrame(qrel_rows).drop_duplicates(),
                filename="msmarco_dev_qrels.txt")
        else:
            print("   ⚠️  No positive_passages in validation split — qrels not written.")
            print("      Download official qrels manually: "
                  "msmarco.blob.core.windows.net/msmarcoranking/qrels.dev.small.tsv.gz")
            qr_path = self.output_dir / "msmarco_dev_qrels.txt"
        return q_path, qr_path


# ---- mixture loading --------------------------------------------------------

def require_mixture_files(mixture_dir, expected_files,
                          reject_unexpected: bool = True) -> tuple:
    """Return the declared non-empty mixture files or fail without mutating data.

    Training consumers use the strict default because they pass a JSONL glob to
    Tevatron; rejecting extra JSONL files prevents a legacy export from joining the
    experiment. Builders may set ``reject_unexpected=False`` and ignore unrelated
    files while deriving from the explicitly declared set.
    """
    mixture_dir = Path(mixture_dir)
    if isinstance(expected_files, (str, bytes)):
        raise TypeError("expected_files must be a sequence of filenames, not a string")
    expected = tuple(expected_files)
    if not expected or len(set(expected)) != len(expected):
        raise ValueError("expected_files must contain unique filenames")
    if any(Path(name).name != name or not name.endswith('.jsonl') for name in expected):
        raise ValueError(f"expected_files must be JSONL filenames: {expected}")
    if not mixture_dir.is_dir():
        raise PreprocessingError(f"training mixture directory not found: {mixture_dir}")

    paths = tuple(mixture_dir / name for name in expected)
    bad = [f"{path.name} ({'missing' if not path.is_file() else 'empty'})"
           for path in paths if not path.is_file() or path.stat().st_size == 0]
    if bad:
        raise PreprocessingError(
            f"{mixture_dir} is missing required mixture data: {', '.join(bad)}. "
            f"{REBUILD_HINT}")

    if reject_unexpected:
        unexpected = sorted(path.name for path in mixture_dir.glob('*.jsonl')
                            if not path.name.startswith('.') and path.name not in expected)
        if unexpected:
            raise PreprocessingError(
                f"unexpected JSONL files in {mixture_dir}: {', '.join(unexpected)}. "
                "Move them outside the mixture directory before training.")
    return paths


def _mixture_files(mixture_dir: Path, expected) -> list:
    """The declared component files, in order. Every one of them must be present.

    Which experiment this is comes from `expected`, not from whichever files survived
    in the directory. Anything not listed is ignored and left untouched.
    """
    return list(require_mixture_files(mixture_dir, expected,
                                      reject_unexpected=False))


def require_derived_artifacts(output_dir=None,
                              corpus_file: str = CORPUS_FILE,
                              queries_file: str = QUERIES_FILE,
                              qrels_file: str = QRELS_FILE):
    """Read-only: the three derived paths, or an error saying which are unusable.

    Consumers -- training, mining, probes -- call this. Only the preprocessor builds.
    """
    output_dir = Path(output_dir) if output_dir else get_path("processed")
    paths = (output_dir / corpus_file, output_dir / queries_file, output_dir / qrels_file)
    bad = [f"{p.name} ({'missing' if not p.is_file() else 'empty'})"
           for p in paths if not p.is_file() or p.stat().st_size == 0]
    if bad:
        raise PreprocessingError(
            f"derived artifacts are not usable in {output_dir}: {', '.join(bad)}. "
            f"{REBUILD_HINT}")
    return paths


def _publish(staging: Path, output_dir: Path, names) -> None:
    """Publish staged files as an all-or-nothing set.

    Staging and output share a filesystem, so each move is atomic. If one fails,
    remove files already moved so consumers never see an incomplete set.
    """
    published = []
    try:
        for name in names:
            os.replace(staging / name, output_dir / name)
            published.append(output_dir / name)
    except BaseException:
        for path in published:
            path.unlink(missing_ok=True)
        raise


def _load_mixture(files) -> list:
    """Parse and validate every record before anything is written.

    Raises on the first structural problem, naming file, line and field, so a legacy
    export fails immediately instead of as a KeyError deep inside a DataFrame.
    """
    records = []
    seen_queries = {}
    doc_origin = {}

    for path in files:
        with open(path, encoding='utf-8') as handle:
            for lineno, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                record = json.loads(line)

                missing = [f for f in REQUIRED_MIXTURE_FIELDS if f not in record]
                if missing:
                    raise PreprocessingError(
                        f"{path.name}:{lineno} missing {', '.join(missing)} (found "
                        f"{sorted(record)}). Legacy ReasonIR exports use positives/"
                        f"negatives; {REBUILD_HINT}")

                qid = str(record['query_id'])
                if qid in seen_queries:
                    # Ids are <source>_<row index>, unique within a file, so a collision
                    # means the same mixture file is in the directory twice (a stray
                    # backup copy) and the dataset would silently double.
                    raise PreprocessingError(
                        f"duplicate query_id {qid!r} at {path.name}:{lineno}, first seen "
                        f"in {seen_queries[qid]} -- is a copy of that file in the "
                        f"mixture directory?")
                seen_queries[qid] = path.name

                for column in ('positive_passages', 'negative_passages'):
                    for passage in record[column]:
                        docid = str(passage['docid'])
                        text = _text(passage['text'])
                        previous = doc_origin.get(docid)
                        if previous is not None and previous[1] != text:
                            raise PreprocessingError(
                                f"docid {docid!r} maps to two different texts: first at "
                                f"{previous[0]}, again at {path.name}:{lineno}")
                        doc_origin[docid] = (f"{path.name}:{lineno}", text)

                records.append((qid, record))

    if not records:
        raise PreprocessingError(
            f"the training mixture is empty: {', '.join(f.name for f in files)}")
    return records


# ---- run_setup --------------------------------------------------------------

def run_setup(mixture_dir=None,
              output_dir=None,
              expected_files=MIXTURE_FILES,
              corpus_file: str = CORPUS_FILE,
              queries_file: str = QUERIES_FILE,
              qrels_file: str = QRELS_FILE):
    """Derive the corpus, queries and qrels every mining pipeline encodes against.

    Every passage in the mixture, deduplicated by text hash, becomes the corpus the
    FAISS / stale ANN index is built from; positives are walked before negatives so a
    text that is ever a positive keeps its positive's docid as canonical. Qrels carry
    the same canonical ids, so mining can exclude true positives.

    Clean build only: an existing output is refused, never reused or overwritten. The
    three files are written to a staging directory and published together, so a failure
    anywhere leaves none of them behind. `expected_files` declares which mixture this
    is; every one of them must be present. Consumers use `require_derived_artifacts`.
    """
    output_dir = Path(output_dir) if output_dir else get_path("processed")
    mixture_dir = Path(mixture_dir) if mixture_dir else output_dir / "training_mixture"

    corpus_path = output_dir / corpus_file
    queries_path = output_dir / queries_file
    qrels_path = output_dir / qrels_file
    outputs = [corpus_path, queries_path, qrels_path]

    existing = [p.name for p in outputs if p.exists()]
    if existing:
        raise PreprocessingError(
            f"refusing to overwrite {', '.join(existing)} in {output_dir}. Delete the "
            f"whole derived-output set ({', '.join(p.name for p in outputs)}) and re-run.")

    print("🛠️ Running setup: building corpus, queries, and qrels from training mixture...",
          flush=True)
    files = _mixture_files(mixture_dir, expected_files)
    records = _load_mixture(files)

    output_dir.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(dir=output_dir, prefix=".staging_"))
    try:
        _derive(records, BRIGHTPreprocessor(output_dir=staging),
                corpus_file, queries_file, qrels_file)
        _publish(staging, output_dir, (corpus_file, queries_file, qrels_file))
    finally:
        shutil.rmtree(staging, ignore_errors=True)

    return corpus_path, queries_path, qrels_path


def _derive(records, preprocessor, corpus_file, queries_file, qrels_file) -> None:
    """Write the three derived files into the preprocessor's (staging) directory."""

    # --- Corpus ---
    # Positives before negatives, globally: a text that is ever a positive keeps its
    # positive's docid as canonical, so a duplicate of a positive can never be
    # retrieved under a second docid and selected as a hard negative.
    canonical = {}
    docid_remap = {}
    emitted_docids = {}
    escaped_docids = set()
    duplicate_docids = set()
    corpus_rows = []
    for column in ('positive_passages', 'negative_passages'):
        for _, record in records:
            for passage in record[column]:
                raw_docid = str(passage['docid'])
                docid = _trec_safe_docid(raw_docid)
                text = _text(passage['text'])
                digest = hashlib.md5(text.strip().encode()).hexdigest()
                owner = canonical.get(digest)
                if owner is None:
                    previous = emitted_docids.get(docid)
                    if previous is not None and previous != digest:
                        raise PreprocessingError(
                            f"TREC-safe document ID collision for {raw_docid!r}")
                    canonical[digest] = docid
                    emitted_docids[docid] = digest
                    corpus_rows.append({'doc_id': docid, 'text': text})
                    if docid != raw_docid:
                        docid_remap[raw_docid] = docid
                        escaped_docids.add(raw_docid)
                elif owner != raw_docid:
                    docid_remap[raw_docid] = owner
                    duplicate_docids.add(raw_docid)

    preprocessor.prepare_tevatron_corpus(pd.DataFrame(corpus_rows), filename=corpus_file)
    print(f"  Corpus: {len(corpus_rows):,} unique passages "
          f"(collapsed {len(duplicate_docids):,} duplicate-text docids; "
          f"escaped {len(escaped_docids):,} whitespace docids)", flush=True)

    # --- Queries ---
    queries_df = pd.DataFrame([{'query_id': qid, 'query': _text(record['query'])}
                               for qid, record in records])
    preprocessor.prepare_tevatron_queries(queries_df, filename=queries_file)
    print(f"  Queries: {len(queries_df):,} unique training queries", flush=True)

    # --- Qrels ---
    pos_pairs = []
    seen_pairs = set()
    for qid, record in records:
        for passage in record['positive_passages']:
            docid = str(passage['docid'])
            docid = docid_remap.get(docid, docid)
            if (qid, docid) in seen_pairs:
                continue
            seen_pairs.add((qid, docid))
            pos_pairs.append({'query_id': qid, 'doc_id': docid, 'relevance': 1})
    preprocessor.prepare_trec_qrels(pd.DataFrame(pos_pairs), filename=qrels_file)
    print(f"  Qrels: {len(pos_pairs):,} positive pairs", flush=True)


# ---- CLI --------------------------------------------------------------------

def _generate_training_mixture(preprocessor, loader, config):
    """Build every component in staging, then publish the mixture as one unit.

    A source that fails part way must not leave a mixture that looks complete.
    """
    if any((preprocessor.output_dir / n).exists() for n in (CORPUS_FILE, QUERIES_FILE, QRELS_FILE)):
        raise PreprocessingError("refusing to replace the mixture while derived artifacts exist")
    mixed = config['data']['mixed_training']
    mixture_dir = preprocessor.output_dir / "training_mixture"
    mixture_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 80)
    print("🎯 Generating Mixed Training Dataset")
    print("=" * 80)

    staging = Path(tempfile.mkdtemp(dir=mixture_dir, prefix=".staging_"))
    staged = BRIGHTPreprocessor(output_dir=staging)
    try:
        print("\n[1/3] MS MARCO Data")
        msmarco_cfg = config['data'].get('msmarco', {})
        staged.prepare_msmarco_train_data(
            dataset_name=msmarco_cfg['name'], subset=msmarco_cfg['subset'],
            filename="train_msmarco.jsonl", limit=mixed['msmarco_samples'])

        print("\n[2/3] VL Data")
        staged.prepare_vl_train_data(
            dataset_name=config['data']['reasonir']['name'],
            filename="train_vl.jsonl", limit=mixed['vl_samples'])

        print("\n[3/3] HQ Data")
        staged.prepare_hq_train_data(
            id2doc=loader.get_all_documents_id_map(),
            dataset_name=config['data']['reasonir']['name'],
            filename="train_hq.jsonl", limit=mixed['hq_samples'])

        _publish(staging, mixture_dir, MIXTURE_FILES)
    finally:
        shutil.rmtree(staging, ignore_errors=True)

    print(f"\n✅ Training data generated in: {mixture_dir}", flush=True)
    return mixture_dir


def _generate_eval_data(preprocessor, loader, config):
    print("\n" + "=" * 80)
    print("🌐 Generating BRIGHT Evaluation Data (Domains)")
    print("=" * 80)
    for domain in config['evaluation'].get('eval_domains', []):
        print(f"Processing Domain: {domain}")
        data = loader.get_data_split(domain)
        preprocessor.prepare_tevatron_corpus(data['corpus'], filename=f"{domain}_corpus.jsonl")
        preprocessor.prepare_tevatron_queries(data['queries'], filename=f"{domain}_queries.jsonl")
        preprocessor.prepare_trec_qrels(data['qrels'], filename=f"{domain}_qrels.txt")
        preprocessor.prepare_bright_excluded(data['excluded'],
                                             filename=f"{domain}_excluded.json")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--derive-only', action='store_true',
                        help="rebuild only the corpus/queries/qrels from the existing "
                             "training_mixture/, skipping the hours of downloading and "
                             "regenerating the mixture and eval files")
    args = parser.parse_args(argv)

    config = load_config()
    preprocessor = BRIGHTPreprocessor()

    loader = None
    if not args.derive_only:
        from data.bright_loader import BRIGHTLoader
        loader = BRIGHTLoader()
        loader.load_dataset()
        _generate_training_mixture(preprocessor, loader, config)

    # Derive straight after the mixture it depends on, before the unrelated BRIGHT
    # evaluation files -- so the two halves are always produced by the same run.
    print("\n" + "=" * 80)
    print("🧾 Deriving corpus / queries / qrels from the mixture")
    print("=" * 80)
    run_setup()

    if loader is not None:
        _generate_eval_data(preprocessor, loader, config)

    print(f"\n✅ All preprocessing complete! Files are in: {preprocessor.output_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
