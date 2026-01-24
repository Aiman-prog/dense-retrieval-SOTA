"""
Final Corrected Evaluation Script.
LOGIC: Untouched original logic (nested fallbacks, dictionary loops).
PATHS/VARS: Resolved via helpers.get_data_base_dir() and config.yaml.
"""

import os
import sys
import pickle
import argparse
import subprocess
import numpy as np 
import pandas as pd
from pathlib import Path
import json
import faiss
import shutil

# 1. Hardware Fix - Environment level
os.environ["TRANSFORMERS_ATTENTION_IMPLEMENTATION"] = "eager"

# Resolve project root and add to sys.path
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(project_root / 'src'))

# Import your specific helpers
from utils.helpers import load_config, get_data_base_dir
from evaluation.trec_eval_wrapper import TrecEvalWrapper

def main():
    # Load config to map variables dynamically
    config = load_config()
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--domain", type=str, default="biology")
    # Pull k from config.yaml evaluation section
    parser.add_argument("--k", type=int, default=config['evaluation'].get('top_k', 1000))
    parser.add_argument("--batch_size", type=int, default=128)
    args = parser.parse_args()

    # --- PATH FIX: Use helpers and config.yaml ---
    # get_data_base_dir() handles the switch between /home/aiman... and /Users/aiamn...
    base_dir = Path(get_data_base_dir())
    
    # processed_dir resolved from config.yaml paths section
    processed_dir = base_dir / config['paths']['processed_dir']
    
    # evaluation dir
    eval_dir = base_dir / 'data' / 'evaluation' / args.domain
    eval_dir.mkdir(parents=True, exist_ok=True)

    # File Paths constructed using the resolved directories
    corpus_file = processed_dir / f"{args.domain}_corpus.jsonl"
    queries_file = processed_dir / f"{args.domain}_queries.jsonl"
    qrels_file = processed_dir / f"{args.domain}_qrels.txt"
    
    corpus_pkl = eval_dir / 'corpus_emb' / 'corpus.pkl'
    query_pkl = eval_dir / 'query_emb' / 'query.pkl'
    ranking_pkl = eval_dir / 'ranking.pkl'

    # --- STEP 1: ENCODE (Logic Untouched) ---
    print(f"\n🚀 Step 1: Encoding with Tevatron Driver...")
    
    # Pull bf16 from evaluation config
    bf16_arg = str(config['evaluation'].get('bf16', False))

    for input_f, output_p, is_q in [(corpus_file, corpus_pkl, False), (queries_file, query_pkl, True)]:
        output_p.parent.mkdir(parents=True, exist_ok=True)
        
        # Base arguments (Logic Untouched)
        cmd = [
            sys.executable, '-m', 'tevatron.retriever.driver.encode',
            '--output_dir', str(output_p.parent),
            '--model_name_or_path', args.model_path,
            '--bf16', bf16_arg,
            '--per_device_eval_batch_size', str(args.batch_size),
            '--dataset_name', 'json',
            '--dataset_path', str(input_f),
            '--encode_output_path', str(output_p),
            '--attn_implementation', 'eager'
        ]
        
        if is_q:
            # Fallback Logic (Logic Untouched)
            q_len = str(config['model'].get('query_max_len', 128))
            temp_cmd = cmd + ['--encode_is_query', '--query_max_len', q_len]
            res = subprocess.run(temp_cmd)
            if res.returncode != 0:
                print("⚠️ Retrying with fallback flags...")
                temp_cmd = cmd + ['--encode_is_qry', '--q_max_len', q_len]
                subprocess.run(temp_cmd, check=True)
        else:
            p_len = str(config['model'].get('passage_max_len', 512))
            subprocess.run(cmd + ['--passage_max_len', p_len], check=True)

    # --- STEP 2: LOAD REFRESHED DATA (Logic Untouched) ---
    print(f"\n📂 Step 2: Loading freshly encoded embeddings...")
    with open(corpus_pkl, 'rb') as f:
        c_data = pickle.load(f)
        corpus_embs = c_data[0].astype(np.float32)
        corpus_ids = [str(x) for x in c_data[1]]

    with open(query_pkl, 'rb') as f:
        q_data = pickle.load(f)
        query_embs = q_data[0].astype(np.float32)
        query_ids = [str(x) for x in q_data[1]]

    # --- STEP 3: FAISS SEARCH (Logic Untouched) ---
    print(f"🔍 Step 3: Running FAISS IndexFlatIP (k={args.k})...")
    index = faiss.IndexFlatIP(corpus_embs.shape[1])
    index.add(corpus_embs)
    scores_mat, indices_mat = index.search(query_embs, args.k)

    # --- STEP 4: VIGILANTE CHECK (Logic Untouched) ---
    qrels_df = pd.read_csv(qrels_file, sep=' ', names=['qid', 'ignore', 'docid', 'rel'], dtype=str)
    print("\n" + "="*60 + "\n🕵️  VIGILANTE CHECK: QUERY 0\n" + "="*60)
    q0_id = str(query_ids[0])
    top_doc_id = str(corpus_ids[indices_mat[0][0]])
    truth_docs = qrels_df[qrels_df['qid'] == q0_id]['docid'].values
    
    print(f"1. Query ID from Pickle:   '{q0_id}'")
    print(f"2. Top Result from FAISS:  '{top_doc_id}'")
    print(f"3. Valid Answers in QRELs: {truth_docs[:3]}...") 
    
    match = top_doc_id in truth_docs
    print(f"\n👉 MATCH CHECK: {match}")
    print("="*60 + "\n")

    # --- STEP 5: RUN FULL EVALUATION (Logic Untouched) ---
    print("📊 Step 5: Running TrecEvalWrapper...")
    run_results = {}
    for i in range(len(query_ids)):
        qid_str = str(query_ids[i])
        run_results[qid_str] = {}
        for j in range(args.k):
            idx = indices_mat[i][j]
            if idx < 0: continue
            run_results[qid_str][str(corpus_ids[idx])] = float(scores_mat[i][j])

    # Re-read for wrapper format
    eval_qrels = pd.read_csv(qrels_file, sep=' ', names=['query_id', 'ignore', 'doc_id', 'relevance'], dtype=str)
    evaluator = TrecEvalWrapper(eval_qrels)
    metrics = evaluator.evaluate(run_results, {'recip_rank', 'ndcg_cut_10', 'recall_10'})
    
    print(f"\n" + "*"*40 + f"\nFINAL RESULTS: {args.domain}\n" + "*"*40)
    print(f"MRR:       {metrics.get('recip_rank', 0):.4f}")
    print(f"NDCG@10:   {metrics.get('ndcg_cut_10', 0):.4f}")
    print(f"Recall@10: {metrics.get('recall_10', 0):.4f}\n" + "*"*40)

if __name__ == "__main__":
    main()