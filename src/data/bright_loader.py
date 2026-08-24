"""Load the BRIGHT dataset from HuggingFace and extract per-domain splits.

Construction is a pure constructor: nothing is downloaded until `load_dataset()` is
called explicitly, and failures propagate instead of being printed and swallowed.
"""

import os
import sys
from typing import Dict, Optional
from pathlib import Path

import pandas as pd
from datasets import load_dataset, DatasetDict

project_root = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(project_root / 'src'))

from utils.helpers import get_data_base_dir, load_config


class BRIGHTLoader:
    """Loader for the BRIGHT dataset from HuggingFace."""

    def __init__(self, config_path: str = "config/config.yaml"):
        self.config = load_config(config_path)

        bright_cfg = self.config['data']['bright']
        self.dataset_name = bright_cfg['name']
        self.examples_config = bright_cfg['examples_config']
        self.cache_dir = Path(get_data_base_dir()) / bright_cfg.get('cache_dir', 'data/bright')

        self.documents_dataset = None
        self.examples_dataset = None

    # ---- loading ------------------------------------------------------------

    def load_dataset(self, cache_dir: Optional[str] = None) -> Dict[str, DatasetDict]:
        """Load the BRIGHT 'documents' and examples subsets."""
        cache = cache_dir or self.cache_dir
        os.makedirs(cache, exist_ok=True)

        print(f"Loading BRIGHT 'documents' from: {self.dataset_name}")
        print(f"Using cache directory: {cache}")
        self.documents_dataset = load_dataset(self.dataset_name, 'documents', cache_dir=cache)

        print(f"Loading BRIGHT '{self.examples_config}' (queries/qrels) "
              f"from: {self.dataset_name}")
        self.examples_dataset = load_dataset(self.dataset_name, self.examples_config,
                                             cache_dir=cache)

        self.validate_example_domains_have_corpora()
        return {'documents': self.documents_dataset, 'examples': self.examples_dataset}

    # ---- validation ---------------------------------------------------------

    def validate_example_domains_have_corpora(self) -> None:
        """Every examples domain must have a matching documents domain.

        An examples domain with no corpus can only ever score zero, so this is an
        error rather than the informational print it used to be.
        """
        self._require_loaded()
        doc_domains = set(self.documents_dataset.keys())
        example_domains = set(self.examples_dataset.keys())
        orphans = sorted(example_domains - doc_domains)
        if orphans:
            raise ValueError(
                f"BRIGHT examples domains have no matching documents domain: {orphans}. "
                f"Available documents domains: {sorted(doc_domains)}")
        print(f"Domains: {len(doc_domains)} documents, {len(example_domains)} examples "
              f"(overlap verified)")

    def _require_loaded(self) -> None:
        if self.documents_dataset is None or self.examples_dataset is None:
            raise ValueError("Dataset not loaded. Call load_dataset() first.")

    def _documents(self, domain: str):
        if self.documents_dataset is None:
            raise ValueError("Dataset not loaded. Call load_dataset() first.")
        if domain not in self.documents_dataset:
            raise ValueError(f"Domain '{domain}' not found in documents.")
        return self.documents_dataset[domain]

    def _examples(self, domain: str):
        if self.examples_dataset is None:
            raise ValueError("Dataset not loaded. Call load_dataset() first.")
        if domain not in self.examples_dataset:
            raise ValueError(f"Domain '{domain}' not found in examples.")
        return self.examples_dataset[domain]

    # ---- extraction ---------------------------------------------------------

    def get_corpus(self, domain: str) -> pd.DataFrame:
        """Corpus for a domain as a (doc_id, text) DataFrame.

        A doc id repeated with identical content collapses; repeated with different
        content it is an error, because whichever row wins silently decides what the
        index vector for that id actually encodes.
        """
        domain_data = self._documents(domain)
        rows = {}
        for doc_id, content in zip(domain_data['id'], domain_data['content']):
            doc_id = str(doc_id)
            if not doc_id.strip():
                raise ValueError(f"blank document id in BRIGHT domain '{domain}'")
            text = "" if content is None else str(content)
            previous = rows.get(doc_id)
            if previous is not None and previous != text:
                raise ValueError(
                    f"document id {doc_id!r} appears twice in BRIGHT domain '{domain}' "
                    f"with different content")
            rows[doc_id] = text
        return pd.DataFrame({'doc_id': list(rows), 'text': list(rows.values())})

    def get_queries(self, domain: str) -> pd.DataFrame:
        """Queries for a domain as a (query_id, query) DataFrame."""
        domain_data = self._examples(domain)
        return pd.DataFrame({
            'query_id': [str(q) for q in domain_data['id']],
            'query': ["" if q is None else str(q) for q in domain_data['query']],
        })

    def get_qrels(self, domain: str) -> pd.DataFrame:
        """Qrels for a domain, expanded from `gold_ids` into TREC triples.

        `gold_ids` is `list<string>` in every BRIGHT config. It is handled as such and
        nothing else: the old fallback split a scalar on commas, and 58 biology document
        ids contain a comma, so that branch could only ever corrupt real ids.
        Duplicate (query_id, doc_id) pairs are collapsed -- theoremqa_questions ships
        178 of them.
        """
        domain_data = self._examples(domain)
        corpus_ids = set(self.get_corpus(domain)['doc_id'])

        pairs = []
        seen = set()
        for i in range(len(domain_data)):
            row = domain_data[i]
            qid = str(row['id'])
            golds = row['gold_ids']
            if not isinstance(golds, (list, tuple)):
                raise ValueError(
                    f"BRIGHT domain '{domain}' query {qid!r}: gold_ids must be a list, "
                    f"got {type(golds).__name__} ({golds!r})")
            for doc_id in golds:
                doc_id = str(doc_id)
                if not doc_id.strip():
                    raise ValueError(
                        f"BRIGHT domain '{domain}' query {qid!r} has a blank gold id")
                if doc_id not in corpus_ids:
                    raise ValueError(
                        f"BRIGHT domain '{domain}' query {qid!r} has gold id {doc_id!r}, "
                        f"which is absent from the domain corpus")
                if (qid, doc_id) in seen:
                    continue
                seen.add((qid, doc_id))
                pairs.append({'query_id': qid, 'doc_id': doc_id, 'relevance': 1})

        return pd.DataFrame(pairs, columns=['query_id', 'doc_id', 'relevance'])

    def get_excluded_ids(self, domain: str) -> Dict[str, list]:
        """Per-query documents BRIGHT's protocol removes from the ranking.

        Most domains store the literal 'N/A' rather than a doc id; aops,
        theoremqa_questions and leetcode carry real ids, up to 11,224 for one query.
        Sentinels and blanks are dropped and each query is de-duplicated in place.
        """
        domain_data = self._examples(domain)
        excluded = {}
        for i in range(len(domain_data)):
            row = domain_data[i]
            raw = row['excluded_ids']
            if not isinstance(raw, (list, tuple)):
                raise ValueError(
                    f"BRIGHT domain '{domain}' query {str(row['id'])!r}: excluded_ids "
                    f"must be a list, got {type(raw).__name__} ({raw!r})")
            keep = {}
            for doc_id in raw:
                doc_id = str(doc_id)
                if doc_id.strip() and doc_id != 'N/A':
                    keep[doc_id] = None
            excluded[str(row['id'])] = list(keep)
        return excluded

    def get_data_split(self, domain: str) -> Dict[str, pd.DataFrame]:
        """Corpus, queries and qrels for one domain. In BRIGHT this is the eval set."""
        return {
            'corpus': self.get_corpus(domain),
            'queries': self.get_queries(domain),
            'qrels': self.get_qrels(domain),
            'excluded': self.get_excluded_ids(domain),
        }

    def get_all_documents_id_map(self) -> Dict[str, str]:
        """Map doc_id -> text across ALL domains, for resolving ReasonIR-HQ positives.

        Follows the ReasonIR dataset card:
        https://huggingface.co/datasets/reasonir/reasonir-data
        """
        self._require_loaded()
        id2doc = {}
        origin = {}
        for task_name, domain_data in self.documents_dataset.items():
            before = len(id2doc)
            for doc_id, content in zip(domain_data['id'], domain_data['content']):
                doc_id = str(doc_id)
                text = "" if content is None else str(content)
                # aops and theoremqa_questions genuinely share 188,002 ids with
                # identical text; the same id carrying different text would mean
                # whichever domain loaded last silently decided the HQ positive.
                if doc_id in id2doc and id2doc[doc_id] != text:
                    raise ValueError(
                        f"BRIGHT document id {doc_id!r} has different text in "
                        f"'{origin[doc_id]}' and '{task_name}'")
                id2doc[doc_id] = text
                origin[doc_id] = task_name
            print(f"   {task_name}: {len(id2doc) - before:,} documents "
                  f"(total {len(id2doc):,})", flush=True)
        print(f"✅ ID-to-text mapping for {len(id2doc):,} documents across "
              f"{len(self.documents_dataset)} domains", flush=True)
        return id2doc


if __name__ == "__main__":
    loader = BRIGHTLoader()
    loader.load_dataset()

    test_domain = 'biology'
    data = loader.get_data_split(test_domain)
    print(f"\n--- {test_domain.upper()} SANITY CHECK ---")
    print(f"Corpus size:  {len(data['corpus']):,}")
    print(f"Queries size: {len(data['queries'])}")
    print(f"Qrels size:   {len(data['qrels'])}")
    print(f"Sample Query: {data['queries'].iloc[0]['query'][:100]}...")
