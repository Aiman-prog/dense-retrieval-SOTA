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
    project_root = Path(__file__).parent.parent.parent
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
            # Always use DelftBlue structure: /scratch/${USER}/dense-retrieval-SOTA/data/processed
            # Can override with PROCESSED_DATA_DIR env var if needed
            base_dir = get_data_base_dir()
            self.output_dir = os.environ.get('PROCESSED_DATA_DIR') or f'{base_dir}/data/processed'
        
        os.makedirs(self.output_dir, exist_ok=True)

    def prepare_tevatron_corpus(self, corpus: pd.DataFrame, filename: str = "corpus.jsonl") -> str:
        """
        Save corpus in Tevatron JSONL format: {"id": "doc_id", "text": "content"}
        """
        output_path = os.path.join(self.output_dir, filename)
        print(f"Processing {len(corpus)} documents for {filename}...")
        
        with open(output_path, 'w', encoding='utf-8') as f:
            for _, row in corpus.iterrows():
                # Tevatron expects 'id' and 'text' keys
                doc = {
                    "id": str(row['doc_id']),
                    "text": row['text'] if pd.notna(row['text']) else "" 
                }
                f.write(json.dumps(doc, ensure_ascii=False) + '\n')
        
        return output_path

    def prepare_tevatron_queries(self, queries: pd.DataFrame, filename: str = "queries.jsonl") -> str:
        """
        Save queries in Tevatron JSONL format: {"id": "query_id", "text": "content"}
        """
        output_path = os.path.join(self.output_dir, filename)
        print(f"Processing {len(queries)} queries for {filename}...")
        
        with open(output_path, 'w', encoding='utf-8') as f:
            for _, row in queries.iterrows():
                query = {
                    "id": str(row['query_id']),
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
                # Ensure strings
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
        
        ReasonIR-HQ provides queries with document IDs (not full texts).
        This method:
        1. Loads ReasonIR-HQ dataset
        2. Maps document IDs to texts using BRIGHT id2doc mapping
        3. Formats into Tevatron JSONL format
        
        Based on approach from ReasonIR dataset card:
        https://huggingface.co/datasets/reasonir/reasonir-data
        
        Args:
            id2doc: Dictionary mapping document ID -> document text (from BRIGHT)
            dataset_name: ReasonIR dataset name (default: "reasonir/reasonir-data")
            subset: Dataset subset to use (default: "hq" for hard-query)
            cache_dir: Optional cache directory for HuggingFace datasets
            filename: Output filename
            
        Returns:
            Path to the created training file
        """
        output_path = os.path.join(self.output_dir, filename)
        print(f"Preparing ReasonIR-HQ training data...")
        
        # Load ReasonIR-HQ dataset
        print(f"Loading ReasonIR dataset: {dataset_name} (subset: {subset})...")
        hq_dataset = load_dataset(dataset_name, subset, cache_dir=cache_dir)
        
        # Process the dataset to map document IDs to texts
        def process_pos_id2doc(entry):
            """Map document IDs in 'pos' column to actual document texts."""
            pos_docs = entry["pos"]
            res = []
            for pos in pos_docs:
                if isinstance(pos, list) and len(pos) >= 2:
                    instruction, doc_id = pos[0], pos[1]
                    # Map doc_id to actual text
                    if doc_id in id2doc:
                        doc_text = id2doc[doc_id]
                        res.append([instruction, doc_text])
                    else:
                        print(f"Warning: Document ID '{doc_id}' not found in BRIGHT mapping")
                else:
                    print(f"Warning: Unexpected format in 'pos' column: {pos}")
            entry["pos"] = res
            return entry
        
        # Apply mapping
        print("Mapping document IDs to texts...")
        hq_dataset = hq_dataset.map(process_pos_id2doc)
        
        # Format into Tevatron JSONL format
        print(f"Formatting training data to {output_path}...")
        count = 0
        skipped = 0
        
        with open(output_path, 'w', encoding='utf-8') as f:
            for entry in hq_dataset['train']:
                # Extract query text
                # ReasonIR-HQ query format: ["instruction", "actual query"]
                query_seq = entry.get("query", [])
                if isinstance(query_seq, list) and len(query_seq) >= 2:
                    # Use the actual query (second element), not the instruction
                    query_text = query_seq[1]
                elif isinstance(query_seq, list) and len(query_seq) == 1:
                    query_text = query_seq[0]
                elif isinstance(query_seq, str):
                    query_text = query_seq
                else:
                    print(f"Warning: Unexpected query format: {query_seq}")
                    skipped += 1
                    continue
                
                # Extract positive documents
                pos_docs = entry.get("pos", [])
                pos_texts = []
                for pos in pos_docs:
                    if isinstance(pos, list) and len(pos) >= 2:
                        # pos format: [instruction, doc_text]
                        doc_text = pos[1]
                        pos_texts.append(doc_text)
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
                    "negatives": []  # Empty for in-batch negative training
                }
                
                f.write(json.dumps(record, ensure_ascii=False) + '\n')
                count += 1
        
        print(f"Saved {count} training examples to {output_path}")
        if skipped > 0:
            print(f"Skipped {skipped} examples due to formatting issues")
        return output_path

if __name__ == "__main__":
    # Example usage for ReasonIR-HQ training data preparation
    from bright_loader import BRIGHTLoader
    from utils.helpers import load_config
    
    # Resolve paths relative to project root
    project_root = Path(__file__).parent.parent.parent
    
    # Load config
    config = load_config(str(project_root / 'config' / 'config.yaml'))
    
    # 1. Load BRIGHT dataset and create ID mapping (needed for ReasonIR-HQ)
    loader = BRIGHTLoader(config_path='config/config.yaml')
    loader.load_dataset()
    id2doc = loader.get_all_documents_id_map()
    
    # 2. Prepare ReasonIR-HQ training data (using config values)
    preprocessor = BRIGHTPreprocessor(output_dir=str(project_root / 'data/processed'))
    reasonir_config = config['dataset']['reasonir']
    train_file = preprocessor.prepare_reasonir_hq_train_data(
        id2doc=id2doc,
        dataset_name=reasonir_config['name'],
        subset=reasonir_config['subset'],
        cache_dir=reasonir_config.get('cache_dir'),
        filename='train_reasonir.jsonl'
    )
    print(f"ReasonIR-HQ training data saved to: {train_file}")
    
    # 3. Example: Prepare BRIGHT evaluation data for a domain
    domain = 'biology'
    data = loader.get_data_split(domain)
    
    # Corpus & Queries for evaluation
    preprocessor.prepare_tevatron_corpus(data['corpus'], f'{domain}_corpus.jsonl')
    preprocessor.prepare_tevatron_queries(data['queries'], f'{domain}_queries.jsonl')
    
    # Qrels for Evaluation
    preprocessor.prepare_trec_qrels(data['qrels'], f'{domain}_qrels.txt')
