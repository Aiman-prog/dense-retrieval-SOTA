"""Preprocess BRIGHT data for Tevatron."""

import os
import json
import pandas as pd
from pathlib import Path
from typing import Optional, Dict, List

class BRIGHTPreprocessor:
    """Preprocess BRIGHT data for Tevatron training and evaluation."""
    
    def __init__(self, output_dir: str = "data/processed"):
        """
        Initialize preprocessor.
        Args:
            output_dir: Directory to save processed files.
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

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

    def prepare_train_data(self, 
                          queries: pd.DataFrame, 
                          corpus: pd.DataFrame, 
                          qrels: pd.DataFrame, 
                          filename: str = "train.jsonl",
                          hard_negatives: Optional[Dict[str, List[str]]] = None) -> str:
        """
        Prepare Training Data in standard Tevatron JSONL format.
        
        Format per line:
        {
            "query_id": "q1",
            "query": "query text...",
            "positives": ["pos_doc_text_1", "pos_doc_text_2"],
            "negatives": ["neg_doc_text_1", ...]  <-- Optional (used for ANCE/RocketQA)
        }
        """
        output_path = os.path.join(self.output_dir, filename)
        print(f"Preparing training pairs for {filename}...")
        
        # 1. Lookup Tables for speed
        corpus_map = dict(zip(corpus['doc_id'].astype(str), corpus['text']))
        query_map = dict(zip(queries['query_id'].astype(str), queries['query']))
        
        # 2. Group Qrels by Query ID
        # We want: qid -> [doc_id1, doc_id2]
        # This fixes the "single pair" bug from the old code
        grouped_qrels = qrels.groupby('query_id')['doc_id'].apply(list).to_dict()
        
        with open(output_path, 'w', encoding='utf-8') as f:
            count = 0
            for qid, pos_doc_ids in grouped_qrels.items():
                qid = str(qid)
                
                # Skip if query is missing (data safety)
                if qid not in query_map:
                    continue
                
                # Get Positive Texts
                pos_texts = [corpus_map[str(pid)] for pid in pos_doc_ids if str(pid) in corpus_map]
                if not pos_texts: 
                    continue
                
                # Build Record
                record = {
                    "query_id": qid,
                    "query": query_map[qid],
                    "positives": pos_texts,
                    "negatives": [] 
                }
                
                # Add Hard Negatives if provided (For ANCE/RocketQA)
                if hard_negatives and qid in hard_negatives:
                    neg_ids = hard_negatives[qid]
                    neg_texts = [corpus_map[str(nid)] for nid in neg_ids if str(nid) in corpus_map]
                    record["negatives"] = neg_texts
                
                f.write(json.dumps(record, ensure_ascii=False) + '\n')
                count += 1
                
        print(f"Saved {count} training examples to {output_path}")
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

if __name__ == "__main__":
    # Example usage
    from bright_loader import BRIGHTLoader
    
    # Resolve paths relative to project root
    project_root = Path(__file__).parent.parent.parent
    
    # 1. Load Data
    loader = BRIGHTLoader(config_path='config/config.yaml')
    loader.load_dataset()
    
    # 2. Get Domain Data
    domain = 'biology'
    data = loader.get_data_split(domain) # Uses the new unified getter
    
    # 3. Preprocess
    preprocessor = BRIGHTPreprocessor(output_dir=str(project_root / 'data/processed'))
    
    # Corpus & Queries
    preprocessor.prepare_tevatron_corpus(data['corpus'], f'{domain}_corpus.jsonl')
    preprocessor.prepare_tevatron_queries(data['queries'], f'{domain}_queries.jsonl')
    
    # Training Data (JSONL)
    preprocessor.prepare_train_data(
        data['queries'], 
        data['corpus'], 
        data['qrels'], 
        f'{domain}_train.jsonl'
    )
    
    # Qrels for Evaluation
    preprocessor.prepare_trec_qrels(data['qrels'], f'{domain}_qrels.txt')