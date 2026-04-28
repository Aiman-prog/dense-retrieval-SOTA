"""Preprocess BRIGHT data for Tevatron."""

import os
import sys
import json
import hashlib
import pandas as pd
from pathlib import Path
from typing import Optional, Dict, List
from datasets import load_dataset
import numpy as np

project_root = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(project_root / 'src'))

from utils.helpers import get_data_base_dir, load_config, get_path

class BRIGHTPreprocessor:
    """Preprocess BRIGHT data for Tevatron training and evaluation."""
    
    def __init__(self, output_dir: Optional[str] = None):
        """Initialize preprocessor."""
        # 2. KEY CHANGE: Use get_path to resolve the processed directory automatically
        self.output_dir = Path(output_dir) if output_dir else get_path("processed")
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def prepare_tevatron_corpus(self, corpus: pd.DataFrame, filename: str = "corpus.jsonl") -> str:
        """Save corpus in Tevatron JSONL format for encoding."""
        output_path = self.output_dir / filename
        print(f"Processing {len(corpus)} documents for {filename}...")
        
        with open(output_path, 'w', encoding='utf-8') as f:
            for _, row in corpus.iterrows():
                doc = {
                    "text_id": str(row['doc_id']),
                    "docid": str(row['doc_id']),
                    "text": row['text'] if pd.notna(row['text']) else "" 
                }
                f.write(json.dumps(doc, ensure_ascii=False) + '\n')
        return str(output_path)

    def prepare_tevatron_queries(self, queries: pd.DataFrame, filename: str = "queries.jsonl") -> str:
        """
        Save queries in Tevatron JSONL format for encoding.
        INCLUDES BOTH KEYS: 'text_id' (for Tevatron) and 'query_id' (for reference).
        """
        output_path = self.output_dir / filename
        print(f"Processing {len(queries)} queries for {filename}...")
        
        with open(output_path, 'w', encoding='utf-8') as f:
            for _, row in queries.iterrows():
                content = row['query'] if pd.notna(row['query']) else ""
                
                # TRICK: Include BOTH keys!
                query_item = {
                    "docid": str(row['query_id']),    # Legacy trick
                    "text_id": str(row['query_id']),  # Legacy support
                    "query_id": str(row['query_id']), # Clarity
                    
                    # --- THE FIX ---
                    "query": content,  # <--- CRITICAL: Tevatron encoder looks for 'query'
                    "text": content    # Fallback: Keep 'text' just in case
                }
                f.write(json.dumps(query_item, ensure_ascii=False) + '\n')
                
        return output_path

    def prepare_pyserini_corpus(self, corpus: pd.DataFrame, output_dir: Path) -> str:
        """
        Save corpus in Pyserini/Lucene format {"id": ..., "contents": ...}.

        Args:
            corpus: DataFrame with columns doc_id, text
            output_dir: Directory to write corpus.jsonl (created if needed)
        Returns:
            Path to written file
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / "corpus.jsonl"
        print(f"Writing {len(corpus)} documents in Pyserini format to {output_path}...")
        with open(output_path, 'w', encoding='utf-8') as f:
            for _, row in corpus.iterrows():
                doc = {
                    "id": str(row['doc_id']),
                    "contents": row['text'] if pd.notna(row['text']) else "",
                }
                f.write(json.dumps(doc, ensure_ascii=False) + '\n')
        return str(output_path)

    def prepare_trec_qrels(self, qrels: pd.DataFrame, filename: str = "qrels.txt") -> str:
        """
        Save QRELS in TREC format for evaluation (trec_eval).
        Format: query_id Q0 doc_id relevance
        """
        output_path = self.output_dir / filename
        
        with open(output_path, 'w', encoding='utf-8') as f:
            for _, row in qrels.iterrows():
                qid = str(row['query_id'])
                did = str(row['doc_id'])
                rel = int(row.get('relevance', 1))
                f.write(f"{qid} Q0 {did} {rel}\n")
                
        print(f"Saved TREC qrels to {output_path}")
        return output_path
    
    def prepare_hq_train_data(self,
                              id2doc: Dict[str, str],
                              dataset_name: str = "reasonir/reasonir-data",
                              cache_dir: Optional[str] = None,
                              filename: str = "train_hq.jsonl",
                              limit: Optional[int] = None) -> str:
        """
        Prepare HQ (Hard Query) training data from ReasonIR.
        Positive documents are BRIGHT doc IDs that need mapping via id2doc.
        
        Args:
            id2doc: Dictionary mapping BRIGHT doc IDs to document text
            dataset_name: HuggingFace dataset name
            cache_dir: Cache directory for datasets
            filename: Output filename
            limit: Maximum number of samples to process
        
        Returns:
            Path to the generated file
        """
        output_path = self.output_dir / filename
        cache = Path(cache_dir) if cache_dir else get_path("bright")
        
        print(f"📥 Loading ReasonIR HQ dataset...")
        dataset = load_dataset(dataset_name, "hq", cache_dir=str(cache))
        
        count = 0
        skipped = 0
        
        with open(output_path, 'w', encoding='utf-8') as f:
            for entry in dataset['train']:
                if limit is not None and count >= limit:
                    break

                # 1. Query Extraction
                query_seq = entry.get("query", [])
                query_text = query_seq[-1] if isinstance(query_seq, list) else query_seq
                
                # 2. Extract Passages
                positive_passages = []
                negative_passages = []
                
                # Process positive passages (need BRIGHT mapping)
                for item in entry.get('pos', []):
                    doc_id = item[1] if isinstance(item, list) else item
                    if doc_id in id2doc:
                        positive_passages.append({
                            "docid": str(doc_id), 
                            "text": id2doc[doc_id]
                        })
                
                # Process negative passages (direct text)
                for item in entry.get('neg', []):
                    text = item[1] if isinstance(item, list) else item
                    text = str(text)
                    negative_passages.append({
                        "docid": f"hq_neg_{count}",
                        "text": text
                    })

                # 3. Validation: Only save if we have both pos and neg
                if not positive_passages or not negative_passages:
                    skipped += 1
                    continue
                
                # 4. Construct and Write Record
                record = {
                    "query_id": f"reasonir_hq_{count}",
                    "query": query_text,
                    "positive_passages": positive_passages,
                    "negative_passages": negative_passages
                }
                f.write(json.dumps(record, ensure_ascii=False) + '\n')
                count += 1
                
        print(f"✅ HQ Complete! Saved: {count:,} | Skipped: {skipped:,}")
        return str(output_path)

    def prepare_vl_train_data(self,
                              dataset_name: str = "reasonir/reasonir-data",
                              cache_dir: Optional[str] = None,
                              filename: str = "train_vl.jsonl",
                              limit: Optional[int] = None,
                              skip_first_n: int = 95000) -> str:
        """
        Prepare VL (Varied-Length) training data from ReasonIR.
        Documents are direct text (no BRIGHT mapping needed).

        Args:
            dataset_name: HuggingFace dataset name
            cache_dir: Cache directory for datasets
            filename: Output filename
            limit: Maximum number of samples to process
            skip_first_n: Skip first N samples (default 95000 to avoid corrupted data)

        Returns:
            Path to the generated file
        """
        output_path = self.output_dir / filename
        cache = Path(cache_dir) if cache_dir else get_path("bright")

        print(f"📥 Loading ReasonIR VL dataset...")
        if skip_first_n > 0:
            print(f"   ⚠️  Skipping first {skip_first_n:,} corrupted samples...")

        dataset = load_dataset(dataset_name, "vl", cache_dir=str(cache))

        count = 0
        skipped = 0

        with open(output_path, 'w', encoding='utf-8') as f:
            for idx, entry in enumerate(dataset['train']):
                # Skip corrupted samples
                if idx < skip_first_n:
                    continue

                if limit is not None and count >= limit:
                    break

                # 1. Query Extraction
                query_seq = entry.get("query", [])
                query_text = query_seq[-1] if isinstance(query_seq, list) else query_seq

                # 2. Extract Passages (all direct text for VL)
                positive_passages = []
                negative_passages = []

                # Process positive passages (direct text)
                for item in entry.get('pos', []):
                    text = item[1] if isinstance(item, list) else item
                    positive_passages.append({
                        "docid": f"vl_pos_{count}",
                        "text": str(text)
                    })

                # Process negative passages (direct text)
                for item in entry.get('neg', []):
                    text = item[1] if isinstance(item, list) else item
                    negative_passages.append({
                        "docid": f"vl_neg_{count}",
                        "text": str(text)
                    })

                # 3. Validation: Only save if we have both pos and neg
                if not positive_passages or not negative_passages:
                    skipped += 1
                    continue

                # 4. Construct and Write Record
                record = {
                    "query_id": f"reasonir_vl_{count}",
                    "query": query_text,
                    "positive_passages": positive_passages,
                    "negative_passages": negative_passages
                }
                f.write(json.dumps(record, ensure_ascii=False) + '\n')
                count += 1

        print(f"✅ VL Complete! Saved: {count:,} | Skipped: {skipped:,}")
        return str(output_path)

    def prepare_msmarco_train_data(self,
                                   dataset_name: str = "sentence-transformers/msmarco-hard-negatives",
                                   subset: str = "triplet",
                                   cache_dir: Optional[str] = None,
                                   filename: str = "train_msmarco.jsonl",
                                   limit: Optional[int] = None) -> str:
        """
        Prepare MS MARCO training data with optional row limit.
        """
        import random

        output_path = self.output_dir / filename
        cache = Path(cache_dir) if cache_dir else get_path("bright")

        print(f"📥 Loading MS MARCO dataset ({subset})...")
        dataset = load_dataset(dataset_name, subset, split='train', cache_dir=str(cache))

        total = len(dataset)
        # Shuffle indices for randomness, then iterate until we collect `limit` clean records
        indices = list(range(total))
        random.shuffle(indices)
        print(f"   Collecting up to {limit:,} clean records from {total:,} examples (skipping pos==neg)...")

        count = 0
        skipped = 0
        with open(output_path, 'w', encoding='utf-8') as f:
            for idx in indices:
                if limit is not None and count >= limit:
                    break
                entry = dataset[idx]
                pos_text = entry['positive']
                neg_text = entry['negative']
                if neg_text.strip() == pos_text.strip():
                    skipped += 1
                    continue
                record = {
                    "query_id": f"msmarco_{count}",
                    "query": entry['query'],
                    "positive_passages": [{"docid": f"msmarco_pos_{count}", "text": pos_text}],
                    "negative_passages": [{"docid": f"msmarco_neg_{count}", "text": neg_text}]
                }
                f.write(json.dumps(record, ensure_ascii=False) + '\n')
                count += 1

        print(f"✅ MS MARCO Complete! Saved: {count:,} | Skipped (pos==neg): {skipped:,}")
        return str(output_path)

def run_setup():
    """
    Prepares the three processed files required by any training script that does
    encoding or hard-negative mining (i.e. train_ance.py, train_grass.py).
    Skips silently if all three files already exist and are non-empty.

    What it does:
      1. Reads all JSONL files from the training_mixture directory (MS MARCO, VL, HQ).
      2. Collects every passage that appears as a positive or negative across all records,
         deduplicates by doc_id, and writes them to `reasonir_corpus.jsonl` in Tevatron
         format — this is the full training corpus used to build the FAISS / stale ANN index.
      3. Extracts unique (query_id, query) pairs from the mixture and writes them to
         `train_queries.jsonl` in Tevatron format — used to encode all training queries
         during index building or GrassSampler.
      4. Collects all (query_id, positive_doc_id) pairs from the mixture and writes them
         to `train_qrels.txt` in TREC format — used during mining to filter out true
         positives so they are never selected as hard negatives.

    Returns:
        Tuple[Path, Path, Path]: absolute paths to
            (reasonir_corpus.jsonl, train_queries.jsonl, train_qrels.txt)
    """
    corpus_path = get_path("processed") / "reasonir_corpus.jsonl"
    queries_path = get_path("processed") / "train_queries.jsonl"
    qrels_path = get_path("processed") / "train_qrels.txt"
    mixture_dir = get_path("processed") / "training_mixture"

    if all(p.exists() and p.stat().st_size > 0 for p in [corpus_path, queries_path, qrels_path]):
        print("⏩ Skipping setup: processed files already exist.", flush=True)
        return corpus_path, queries_path, qrels_path

    print("🛠️ Running setup: building corpus, queries, and qrels from training mixture...", flush=True)
    preprocessor = BRIGHTPreprocessor()
    mix_files = [f for f in mixture_dir.glob("*.jsonl") if not f.name.startswith('.')]

    # Load all mixture files into a single DataFrame
    mix_dfs = []
    for f in mix_files:
        df = pd.read_json(f, lines=True)
        if 'query_text' in df.columns:
            df = df.rename(columns={'query_text': 'query'})
        mix_dfs.append(df)
    mix_df = pd.concat(mix_dfs, ignore_index=True)

    # --- Corpus ---
    # Collect every passage (positive and negative) from all records, deduplicate by text
    # hash so that passages with identical content but different docids collapse to one
    # canonical entry. This prevents GRASS/ANCE from retrieving a duplicate of a positive
    # under a different docid and selecting it as a hard negative.
    all_passages = []
    for col in ['positive_passages', 'negative_passages']:
        for record_list in mix_df[col]:
            all_passages.extend(record_list)
    passages_df = (pd.DataFrame(all_passages)
                   .rename(columns={'docid': 'doc_id'})[['doc_id', 'text']])
    passages_df['_hash'] = passages_df['text'].apply(
        lambda t: hashlib.md5(t.strip().encode()).hexdigest()
    )
    # First docid seen for each unique text becomes canonical
    hash_to_canonical = (passages_df.drop_duplicates(subset=['_hash'])
                         .set_index('_hash')['doc_id'])
    passages_df['canonical_id'] = passages_df['_hash'].map(hash_to_canonical)
    # Remap dict: only entries where docid != canonical (i.e. true duplicates)
    docid_remap = {
        row['doc_id']: row['canonical_id']
        for _, row in passages_df[passages_df['doc_id'] != passages_df['canonical_id']].iterrows()
    }
    corpus_df = (passages_df.drop_duplicates(subset=['_hash'])
                 [['doc_id', 'text']])
    preprocessor.prepare_tevatron_corpus(corpus_df, filename="reasonir_corpus.jsonl")
    print(f"  Corpus: {len(corpus_df)} unique passages "
          f"(collapsed {len(docid_remap)} duplicate-text docids)", flush=True)

    # --- Queries ---
    # One entry per unique query_id — used to encode training queries for ANN search.
    queries_df = mix_df[['query_id', 'query']].drop_duplicates(subset=['query_id'])
    preprocessor.prepare_tevatron_queries(queries_df, filename="train_queries.jsonl")
    print(f"  Queries: {len(queries_df)} unique training queries", flush=True)

    # --- Qrels ---
    # All (query_id, positive_doc_id) pairs in TREC format.
    # Used during mining to exclude true positives from the hard negative candidates.
    pos_pairs = []
    for _, row in mix_df.iterrows():
        for pos in row['positive_passages']:
            orig_id = str(pos['docid'])
            canonical_id = docid_remap.get(orig_id, orig_id)
            pos_pairs.append({'query_id': str(row['query_id']), 'doc_id': canonical_id, 'relevance': 1})
    preprocessor.prepare_trec_qrels(pd.DataFrame(pos_pairs).drop_duplicates(), filename="train_qrels.txt")
    print(f"  Qrels: {len(pos_pairs)} positive pairs", flush=True)

    return corpus_path, queries_path, qrels_path


if __name__ == "__main__":
    from data.bright_loader import BRIGHTLoader
    
    config = load_config()
    loader = BRIGHTLoader() 
    loader.load_dataset()
    id2doc = loader.get_all_documents_id_map()
    
    preprocessor = BRIGHTPreprocessor()
    
    # 1. TRAINING DATA GENERATION - Save to training_mixture/
    mixture_dir = preprocessor.output_dir / "training_mixture"
    mixture_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "=" * 80)
    print("🎯 Generating Mixed Training Dataset")
    print("   Tevatron will read all .jsonl files in training_mixture/")
    print("=" * 80)
    
    mixed_config = config['data'].get('mixed_training', {})
    
    # Generate MS MARCO
    print("\n[1/3] MS MARCO Data")
    preprocessor.prepare_msmarco_train_data(
        dataset_name=config['data'].get('msmarco', {}).get('name', 'sentence-transformers/msmarco-hard-negatives'),
        subset=config['data'].get('msmarco', {}).get('subset', 'triplet'),
        filename="training_mixture/train_msmarco.jsonl",
        limit=mixed_config.get('msmarco_samples', 20000)
    )
    
    # Generate VL
    print("\n[2/3] VL Data")
    preprocessor.prepare_vl_train_data(
        dataset_name=config['data']['reasonir']['name'],
        filename="training_mixture/train_vl.jsonl",
        limit=mixed_config.get('vl_samples', 233000),
        skip_first_n=95000  # Skip corrupted early samples
    )
    
    # Generate HQ
    print("\n[3/3] HQ Data")
    preprocessor.prepare_hq_train_data(
        id2doc=id2doc,
        dataset_name=config['data']['reasonir']['name'],
        filename="training_mixture/train_hq.jsonl",
        limit=mixed_config.get('hq_samples', 97000)
    )
    
    total = mixed_config.get('msmarco_samples', 20000) + mixed_config.get('vl_samples', 233000) + mixed_config.get('hq_samples', 97000)
    print(f"\n✅ Training data generated in: {mixture_dir}")
    print(f"   Total samples: ~{total:,} across 3 files")
    print(f"   Tevatron will load: {mixture_dir / '*.jsonl'}")

    # 3. EVALUATION DATA GENERATION
    print("\n" + "=" * 80)
    print("🌐 Generating BRIGHT Evaluation Data (Domains)")
    print("=" * 80)

    eval_domains = config['evaluation'].get('eval_domains', [])
    for domain in eval_domains:
        print(f"Processing Domain: {domain}")
        domain_data = loader.get_data_split(domain)
        
        preprocessor.prepare_tevatron_corpus(domain_data['corpus'], filename=f"{domain}_corpus.jsonl")
        preprocessor.prepare_tevatron_queries(domain_data['queries'], filename=f"{domain}_queries.jsonl")
        preprocessor.prepare_trec_qrels(domain_data['qrels'], filename=f"{domain}_qrels.txt")

    print(f"\n✅ All preprocessing complete! Files are in: {preprocessor.output_dir}")