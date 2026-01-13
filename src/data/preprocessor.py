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
                    "docid": str(row['doc_id']),    # REQUIRED for clarity/mapping
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
                # TRICK: Include BOTH keys!
                query = {
                    "text_id": str(row['query_id']), # REQUIRED by Tevatron
                    "query_id": str(row['query_id']), # REQUIRED for clarity
                    "text": row['query'] if pd.notna(row['query']) else ""
                }
                f.write(json.dumps(query, ensure_ascii=False) + '\n')
                
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
        Prepare ReasonIR-HQ training data for Tevatron.
        """
        output_path = os.path.join(self.output_dir, filename)
        print(f"Preparing ReasonIR-HQ training data...")
        
        # Use consistent cache directory
        cache_dir = os.environ.get('HF_DATASETS_CACHE') or os.environ.get('HF_HOME')
        if not cache_dir:
            base_dir = get_data_base_dir()
            cache_dir = f'{base_dir}/data/bright'
        
        os.makedirs(cache_dir, exist_ok=True)
        print(f"Loading ReasonIR dataset: {dataset_name} (subset: {subset})...")
        hq_dataset = load_dataset(dataset_name, subset, cache_dir=cache_dir)
        
        # Format into Tevatron JSONL format
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
                
                # Extract positive documents
                pos_docs = entry.get("pos", [])
                pos_texts = []
                for pos in pos_docs:
                    if isinstance(pos, list) and len(pos) >= 2:
                        doc_id = pos[1]
                        if doc_id in id2doc:
                            pos_texts.append(id2doc[doc_id])
                    elif isinstance(pos, str):
                         pos_texts.append(pos)
                
                if not pos_texts:
                    skipped += 1
                    continue
                
                # Create record
                record = {
                    "query_id": f"reasonir_{count}",
                    "query": query_text,
                    "positives": pos_texts,
                    "negatives": []
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
    
    # 1. Load BRIGHT
    loader = BRIGHTLoader(config_path='config/config.yaml')
    loader.load_dataset()
    
    # 2. Example: Prepare Biology
    domain = 'biology'
    # Auto-detect scratch space on DelftBlue, otherwise use project root
    user = os.environ.get('USER', '')
    scratch_path = Path(f'/scratch/{user}/dense-retrieval-SOTA')
    if scratch_path.exists() and scratch_path.is_dir():
        # On DelftBlue: use scratch space (faster, more space)
        output_dir = str(scratch_path / 'data' / 'processed' / 'eval' / domain)
        print(f"Using scratch space: {output_dir}")
    else:
        # Local: use project root
        output_dir = str(project_root / 'data' / 'processed' / 'eval' / domain)
        print(f"Using project root: {output_dir}")
    preprocessor = BRIGHTPreprocessor(output_dir=output_dir)
    
    try:
        data = loader.get_data_split(domain)
        print(f"Preparing data for {domain}...")
        preprocessor.prepare_tevatron_corpus(data['corpus'], 'corpus.jsonl')
        preprocessor.prepare_tevatron_queries(data['queries'], 'queries.jsonl')
        preprocessor.prepare_trec_qrels(data['qrels'], 'qrels.txt')
        print("✅ Done!")
    except Exception as e:
        print(f"Error: {e}")