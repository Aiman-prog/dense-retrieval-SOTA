"""Preprocess BRIGHT data for Tevatron."""

import os
import sys
import json
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
    
    def prepare_reasonir_hq_train_data(self,
                                       id2doc: Dict[str, str],
                                       dataset_name: str = "reasonir/reasonir-data",
                                       subset: str = "hq",
                                       cache_dir: Optional[str] = None,
                                       filename: str = "train_reasonir.jsonl") -> str:
        """Prepare ReasonIR-HQ training data for GitHub Tevatron."""
        output_path = self.output_dir / filename
        
        # 3. KEY CHANGE: Use get_path('bright') to find the cache consistently
        cache = Path(cache_dir) if cache_dir else get_path("bright")
        cache.mkdir(parents=True, exist_ok=True)
        
        print(f"Loading ReasonIR dataset from: {cache}")
        hq_dataset = load_dataset(dataset_name, subset, cache_dir=str(cache))
        count = 0
        skipped = 0
        
        with open(output_path, 'w', encoding='utf-8') as f:
            for entry in hq_dataset['train']:
                # Extract query text
                query_seq = entry.get("query", [])
                if isinstance(query_seq, list) and len(query_seq) >= 2:
                    query_text = query_seq[1]
                elif isinstance(query_seq, list) and len(query_seq) == 1:
                    query_text = query_seq[0]
                elif isinstance(query_seq, str):
                    query_text = query_seq
                else:
                    skipped += 1
                    continue
                
                # Extract positive passages (as list of dicts with docid and text)
                pos_docs = entry.get("pos", [])
                positive_passages = []
                for pos in pos_docs:
                    if isinstance(pos, list) and len(pos) >= 2:
                        doc_id = str(pos[1])
                        if doc_id in id2doc:
                            positive_passages.append({
                                "docid": doc_id,
                                "text": id2doc[doc_id]
                            })
                    elif isinstance(pos, str):
                        # If it's already a string, treat as docid
                        if pos in id2doc:
                            positive_passages.append({
                                "docid": pos,
                                "text": id2doc[pos]
                            })
                
                if not positive_passages:
                    skipped += 1
                    continue
                
                neg_docs = entry.get("neg", []) # Get the hard negatives from the dataset
                negative_passages = []
                for neg in neg_docs:
                    if isinstance(neg, list) and len(neg) >= 2:
                        doc_id = str(neg[1])
                        if doc_id in id2doc:
                            negative_passages.append({
                                "docid": doc_id,
                                "text": id2doc[doc_id]
                            })
                # Standard Tevatron format: positive_passages and negative_passages
                record = {
                    "query_id": f"reasonir_{count}",
                    "query": query_text,
                    "positive_passages": positive_passages,  # List of dicts: [{"docid": "...", "text": "..."}]
                    "negative_passages": negative_passages
                }
                
                f.write(json.dumps(record, ensure_ascii=False) + '\n')
                count += 1

        return str(output_path)

if __name__ == "__main__":
    from data.bright_loader import BRIGHTLoader
    
    # 4. CLEANER BOILERPLATE: Just call load_config()
    config = load_config()
    
    print("=" * 80)
    print("Generating ReasonIR-HQ Training Data")
    print("=" * 80)
    
    loader = BRIGHTLoader() # Uses config/config.yaml by default now
    loader.load_dataset()
    id2doc = loader.get_all_documents_id_map()
    
    preprocessor = BRIGHTPreprocessor()
    
    # 5. KEY CHANGE: Match the new 'data' structure in YAML
    reasonir_cfg = config['data']['reasonir']
    
    train_file_path = preprocessor.prepare_reasonir_hq_train_data(
        id2doc=id2doc,
        dataset_name=reasonir_cfg['name'],
        subset=reasonir_cfg['subset'],
        # cache_dir is handled internally by preprocessor using get_path("bright")
        filename=reasonir_cfg.get('train_file', 'train_reasonir.jsonl')
    )
    
    print(f"\n✅ Training data generated: {train_file_path}")
    print("=" * 80)

    print("\n" + "=" * 80)
    print("Generating BRIGHT Evaluation Data (Domains)")
    print("=" * 80)

    # Pull domains from the evaluation section of config
    eval_domains = config['evaluation'].get('eval_domains', [])
    
    for domain in eval_domains:
        print(f"\n🌐 Processing Domain: {domain}")
        
        # 1. Get raw data from loader
        domain_data = loader.get_data_split(domain)
        
        # 2. Create the Corpus JSONL (What evaluate.py was missing!)
        preprocessor.prepare_tevatron_corpus(
            domain_data['corpus'], 
            filename=f"{domain}_corpus.jsonl"
        )
        
        # 3. Create the Queries JSONL
        preprocessor.prepare_tevatron_queries(
            domain_data['queries'], 
            filename=f"{domain}_queries.jsonl"
        )
        
        # 4. Create the Qrels TXT
        preprocessor.prepare_trec_qrels(
            domain_data['qrels'], 
            filename=f"{domain}_qrels.txt"
        )

    print(f"\n✅ All preprocessing complete! Files are in: {preprocessor.output_dir}")