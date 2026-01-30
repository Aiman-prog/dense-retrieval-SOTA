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
        """Prepare ReasonIR training data. Automatically handles text extraction for VL."""
        output_path = self.output_dir / filename
        cache = Path(cache_dir) if cache_dir else get_path("bright")
        
        print(f"📥 Loading ReasonIR {subset.upper()} dataset...")
        dataset = load_dataset(dataset_name, subset, cache_dir=str(cache))
        
        count = 0
        skipped = 0
        
        with open(output_path, 'w', encoding='utf-8') as f:
            for entry in dataset['train']:
                # 1. Query Extraction
                query_seq = entry.get("query", [])
                query_text = query_seq[-1] if isinstance(query_seq, list) else query_seq
                
                # 2. Restructured Unified Extraction Logic (No enumerate)
                passages = {"pos": [], "neg": []}
                
                for key in ["pos", "neg"]:
                    raw_list = entry.get(key, [])
                    for item in raw_list:
                        content_or_id = item[1] if isinstance(item, list) else item
                        
                        if subset == "hq" and key == "pos":
                            # HQ Logic: Map ID to BRIGHT Text
                            if content_or_id in id2doc:
                                passages[key].append({
                                    "docid": str(content_or_id), 
                                    "text": id2doc[content_or_id]
                                })
                        else:
                            # VL Logic: Use raw text directly, matching old ID style
                            passages[key].append({
                                "docid": f"vl_{key}_{count}", 
                                "text": str(content_or_id)
                            })

                # 3. Validation: Tevatron MUST have at least one positive
                # Replace your previous validation check with this:
                if not passages["pos"] or not passages["neg"]:
                    skipped += 1
                    continue
                
                record = {
                    "query_id": f"reasonir_{subset}_{count}",
                    "query": query_text,
                    "positive_passages": passages["pos"],
                    "negative_passages": passages["neg"]
                }
                f.write(json.dumps(record, ensure_ascii=False) + '\n')
                count += 1
        print(f"✅ {subset.upper()} Complete! Processed: {count:,} | Skipped: {skipped:,}")
        return str(output_path)

if __name__ == "__main__":
    from data.bright_loader import BRIGHTLoader
    
    config = load_config()
    loader = BRIGHTLoader() 
    loader.load_dataset()
    id2doc = loader.get_all_documents_id_map()
    
    preprocessor = BRIGHTPreprocessor()
    
    # 1. TRAINING DATA GENERATION
    # ReasonIR-8B is trained on a mixture of HQ and VL
    mixture_dir = preprocessor.output_dir / "training_mixture"
    mixture_dir.mkdir(parents=True, exist_ok=True)
    
    for subset_type in ['hq', 'vl']:
        print("=" * 80)
        print(f"🚀 Generating ReasonIR-{subset_type.upper()} Training Data")
        print("=" * 80)
        
        reasonir_cfg = config['data']['reasonir']
        
        # Point the filename to the new subdirectory
        filename = f"training_mixture/train_reasonir_{subset_type}.jsonl"
        
        train_file_path = preprocessor.prepare_reasonir_hq_train_data(
            id2doc=id2doc,
            dataset_name=reasonir_cfg['name'],
            subset=subset_type, 
            filename=filename
        )
        
        print(f"\n✅ {subset_type.upper()} Data generated: {train_file_path}")

    # 2. EVALUATION DATA GENERATION
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