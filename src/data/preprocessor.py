"""Preprocess BRIGHT data for Tevatron."""

import os
import sys
import json
import pandas as pd
from pathlib import Path
from typing import Optional, Dict, List
from datasets import load_dataset

# Import helper function (handle both package and direct import)
try:
    from utils.helpers import get_data_base_dir
except ImportError:
    # Fallback for relative import
    project_root = Path(__file__).resolve().parent.parent.parent
    if str(project_root / 'src') not in sys.path:
        sys.path.insert(0, str(project_root / 'src'))
    from utils.helpers import get_data_base_dir

class BRIGHTPreprocessor:
    """Preprocess BRIGHT data for Tevatron training and evaluation."""
    
    def __init__(self, output_dir: Optional[str] = None):
        """
        Initialize preprocessor.
        Args:
            output_dir: Optional override. Defaults to DATA_BASE_DIR/data/processed
        """
        if output_dir:
            self.output_dir = output_dir
        else:
            base_dir = get_data_base_dir()
            self.output_dir = os.environ.get('PROCESSED_DATA_DIR') or f'{base_dir}/data/processed'
        
        os.makedirs(self.output_dir, exist_ok=True)

    def prepare_tevatron_corpus(self, corpus: pd.DataFrame, filename: str = "corpus.jsonl") -> str:
        """
        Save corpus in Tevatron JSONL format for encoding.
        INCLUDES BOTH KEYS: 'text_id' (for Tevatron) and 'docid' (for reference).
        """
        output_path = os.path.join(self.output_dir, filename)
        print(f"Processing {len(corpus)} documents for {filename}...")
        
        with open(output_path, 'w', encoding='utf-8') as f:
            for _, row in corpus.iterrows():
                # TRICK: Include BOTH keys!
                doc = {
                    "text_id": str(row['doc_id']),  # REQUIRED by Tevatron to avoid crash
                    "docid": str(row['doc_id']),    # REQUIRED: Tevatron looks for 'docid'
                    "text": row['text'] if pd.notna(row['text']) else "" 
                }
                f.write(json.dumps(doc, ensure_ascii=False) + '\n')
        
        return output_path

    def prepare_tevatron_queries(self, queries: pd.DataFrame, filename: str = "queries.jsonl") -> str:
        """
        Save queries in Tevatron JSONL format for encoding.
        INCLUDES BOTH KEYS: 'text_id' (for Tevatron) and 'query_id' (for reference).
        """
        output_path = os.path.join(self.output_dir, filename)
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
        output_path = os.path.join(self.output_dir, filename)
        
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
        """
        Prepare ReasonIR-HQ training data for GitHub Tevatron.
        Standard format: {"query_id": "...", "query": "...", "positive_passages": [{"docid": "...", "text": "..."}], "negative_passages": []}
        """
        output_path = os.path.join(self.output_dir, filename)
        print(f"Preparing ReasonIR-HQ training data for GitHub Tevatron...")
        
        # Use consistent cache directory
        cache_dir = os.environ.get('HF_DATASETS_CACHE') or os.environ.get('HF_HOME')
        if not cache_dir:
            base_dir = get_data_base_dir()
            cache_dir = f'{base_dir}/data/bright'
        
        os.makedirs(cache_dir, exist_ok=True)
        print(f"Loading ReasonIR dataset: {dataset_name} (subset: {subset})...")
        hq_dataset = load_dataset(dataset_name, subset, cache_dir=cache_dir)
        
        # Format into Tevatron JSONL format with positive_passages (full text)
        print(f"Formatting training data to {output_path}...")
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
        
        print(f"Saved {count} training examples to {output_path}")
        if skipped > 0:
            print(f"Skipped {skipped} examples (missing docs or bad format)")
        return output_path

if __name__ == "__main__":
    from data.bright_loader import BRIGHTLoader
    from utils.helpers import load_config
    
    project_root = Path(__file__).resolve().parent.parent.parent
    config = load_config(str(project_root / 'config' / 'config.yaml'))
    
    print("=" * 80)
    print("Generating ReasonIR-HQ Training Data")
    print("=" * 80)
    
    # 1. Load BRIGHT dataset and create ID mapping (needed for ReasonIR-HQ)
    print("Step 1: Loading BRIGHT dataset for document ID mapping...")
    loader = BRIGHTLoader(config_path='config/config.yaml')
    loader.load_dataset()
    id2doc = loader.get_all_documents_id_map()
    print(f"✅ Created ID-to-text mapping for {len(id2doc)} documents")
    
    # 2. Prepare ReasonIR-HQ training data
    print("\nStep 2: Preparing ReasonIR-HQ training data...")
    preprocessor = BRIGHTPreprocessor()
    reasonir_config = config['dataset']['reasonir']
    
    train_file_path = preprocessor.prepare_reasonir_hq_train_data(
        id2doc=id2doc,
        dataset_name=reasonir_config['name'],
        subset=reasonir_config['subset'],
        cache_dir=reasonir_config.get('cache_dir'),
        filename='train_reasonir.jsonl'
    )
    
    print(f"\n✅ Training data generated: {train_file_path}")
    print("=" * 80)