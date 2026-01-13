import sys
import argparse
import pickle
import os
import json
import pandas as pd
import numpy as np
from pathlib import Path

# Fix imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
try:
    from src.evaluation.trec_eval_wrapper import TrecEvalWrapper
except ImportError:
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from evaluation.trec_eval_wrapper import TrecEvalWrapper

def load_id_map(jsonl_path, key_type="doc"):
    """Creates a mapping from Row Index (0,1,2) -> Real String ID."""
    if not jsonl_path or not os.path.exists(jsonl_path):
        return None
    print(f"DEBUG: Building ID map from {jsonl_path}...")
    id_map = {}
    try:
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for idx, line in enumerate(f):
                data = json.loads(line)
                if key_type == "doc":
                    # Priority: docid -> text_id -> row_index
                    real_id = data.get('docid') or data.get('text_id') or str(idx)
                else:
                    # Priority: query_id -> text_id -> row_index
                    real_id = data.get('query_id') or data.get('text_id') or str(idx)
                id_map[str(idx)] = str(real_id)
        print(f"DEBUG: Mapped {len(id_map)} IDs.")
        return id_map
    except Exception as e:
        print(f"⚠️ Warning: Could not load ID map: {e}")
        return None

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--qrels", required=True)
    parser.add_argument("--ranking", required=True)
    parser.add_argument("--domain", required=True)
    parser.add_argument("--corpus_file", required=True)  # Required for index mapping
    parser.add_argument("--queries_file", required=True) # Required for index mapping
    args = parser.parse_args()
    
    print(f"DEBUG: STARTING EVALUATION for {args.domain}")

    # 1. Load Maps (Index -> String ID)
    doc_map = load_id_map(args.corpus_file, "doc")
    query_map = load_id_map(args.queries_file, "query")

    if not doc_map or not query_map:
        print("❌ CRITICAL: Failed to build ID maps. Cannot map Faiss indices to IDs.")
        sys.exit(1)

    if not os.path.exists(args.ranking):
        print("❌ CRITICAL: Ranking file missing.")
        sys.exit(1)

    print("DEBUG: Loading Ranking File...")
    try:
        with open(args.ranking, 'rb') as f:
            raw = pickle.load(f)

        run = {}

        # === HANDLER FOR TUPLE FORMAT (Scores, Indices) ===
        # This matches your file inspection: tuple of length 2, first item is numpy array
        if isinstance(raw, tuple) and len(raw) == 2:
            print("DEBUG: Detected (Scores, Indices) matrix format.")
            scores_matrix = raw[0]
            indices_matrix = raw[1]

            # scores_matrix shape is (num_queries, k)
            num_queries = len(scores_matrix)
            
            for q_idx in range(num_queries):
                # 1. Map Row Index 'q_idx' -> Real Query ID
                q_idx_str = str(q_idx)
                if q_idx_str in query_map:
                    real_qid = query_map[q_idx_str]
                else:
                    # Fallback if map incomplete
                    real_qid = q_idx_str

                if real_qid not in run:
                    run[real_qid] = {}

                # 2. Iterate through Top-K docs for this query
                q_scores = scores_matrix[q_idx]
                q_doc_indices = indices_matrix[q_idx]

                for score, doc_idx in zip(q_scores, q_doc_indices):
                    # 3. Map Doc Index 'doc_idx' -> Real Doc ID
                    d_idx_str = str(doc_idx)
                    
                    if d_idx_str in doc_map:
                        real_docid = doc_map[d_idx_str]
                    else:
                        real_docid = d_idx_str
                    
                    # Store as float
                    run[real_qid][real_docid] = float(score)

        # === FALLBACK HANDLER (Dictionary Format) ===
        elif isinstance(raw, dict):
            print("DEBUG: Detected Dictionary format.")
            for qid, docs in raw.items():
                real_qid = str(qid)
                if real_qid in query_map: real_qid = query_map[real_qid]
                
                run[real_qid] = {}
                for docid, score in docs.items():
                    real_docid = str(docid)
                    if real_docid in doc_map: real_docid = doc_map[real_docid]
                    
                    if hasattr(score, 'item'): score = score.item()
                    run[real_qid][real_docid] = float(score)

        else:
            print(f"❌ ERROR: Unknown data format: {type(raw)}")
            sys.exit(1)

        # 5. Evaluate
        print("DEBUG: Loading Qrels...")
        qrels = pd.read_csv(args.qrels, sep=r'\s+', header=None, 
                          names=['query_id', 'Q0', 'doc_id', 'relevance'],
                          dtype={'query_id': str, 'doc_id': str, 'relevance': int})
        
        print("DEBUG: Running TrecEval...")
        evaluator = TrecEvalWrapper(qrels)
        metrics = evaluator.evaluate(run, {'recip_rank', 'ndcg_cut_10', 'recall_10'})
        
        print(f"\nRESULTS for {args.domain}:")
        print(f"MRR:       {metrics.get('recip_rank', 0):.4f}")
        print(f"NDCG@10:   {metrics.get('ndcg_cut_10', 0):.4f}")
        print(f"Recall@10: {metrics.get('recall_10', 0):.4f}\n")

    except Exception as e:
        print(f"\n❌ PYTHON ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()