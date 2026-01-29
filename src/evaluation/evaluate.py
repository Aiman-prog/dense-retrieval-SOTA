"""
Reverted Evaluation Script.
LOGIC: Uses standard FAISS CPU indexing.
DIAGNOSTICS: Kept prints to monitor environment status in .out file.
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
import torch 

# Hardware Fix
os.environ["TRANSFORMERS_ATTENTION_IMPLEMENTATION"] = "eager"

project_root = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(project_root / 'src'))

from utils.helpers import load_config, get_data_base_dir
from evaluation.trec_eval_wrapper import TrecEvalWrapper

def main():
    config = load_config()
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--domain", type=str, default="biology")
    parser.add_argument("--k", type=int, default=config['evaluation'].get('top_k', 1000))
    parser.add_argument("--batch_size", type=int, default=128)
    args = parser.parse_args()

    # --- HARDWARE DIAGNOSTIC ---
    # Kept this so you can verify the environment in the .out log
    print("\n" + "="*40, flush=True)
    print("🔍 HARDWARE DIAGNOSTIC", flush=True)
    print("="*40, flush=True)
    
    cuda_available = torch.cuda.is_available()
    print(f"PyTorch Version:  {torch.__version__}", flush=True)
    print(f"CUDA Available:   {cuda_available}", flush=True)
    if cuda_available:
        print(f"GPU Device:       {torch.cuda.get_device_name(0)}", flush=True)
    
    print(f"FAISS Version:    {faiss.__version__}", flush=True)
    print("="*40 + "\n", flush=True)

    # --- PATH SETUP ---
    base_dir = Path(get_data_base_dir())
    processed_dir = base_dir / config['paths']['processed_dir']
    eval_dir = base_dir / 'data' / 'evaluation' / args.domain
    eval_dir.mkdir(parents=True, exist_ok=True)

    corpus_file = processed_dir / f"{args.domain}_corpus.jsonl"
    queries_file = processed_dir / f"{args.domain}_queries.jsonl"
    qrels_file = processed_dir / f"{args.domain}_qrels.txt"
    
    corpus_pkl = eval_dir / 'corpus_emb' / 'corpus.pkl'
    query_pkl = eval_dir / 'query_emb' / 'query.pkl'

    # --- STEP 1: ENCODE ---
    print(f"🚀 Step 1: Encoding with Tevatron Driver...", flush=True)

    # FORCE THIS TO TRUE for A100 MIG slices to avoid Sgemm crashes
    bf16_arg = "True" 

    for input_f, output_p, is_q in [(corpus_file, corpus_pkl, False), (queries_file, query_pkl, True)]:
        output_p.parent.mkdir(parents=True, exist_ok=True)
        cmd = [
            sys.executable, '-m', 'tevatron.retriever.driver.encode',
            '--output_dir', str(output_p.parent),
            '--model_name_or_path', args.model_path,
            '--bf16', bf16_arg,        # MUST BE TRUE
            '--fp16', 'False',
            '--per_device_eval_batch_size', str(args.batch_size),
            '--dataset_name', 'json',
            '--dataset_path', str(input_f),
            '--encode_output_path', str(output_p),
            '--attn_implementation', 'eager',
            '--dataloader_num_workers', '2' 
        ]
        
        if is_q:
            q_len = str(config['model'].get('query_max_len', 128))
            temp_cmd = cmd + ['--encode_is_query', '--query_max_len', q_len]
            res = subprocess.run(temp_cmd)
            if res.returncode != 0:
                print("⚠️ Retrying with fallback flags...", flush=True)
                temp_cmd = cmd + ['--encode_is_qry', '--q_max_len', q_len]
                subprocess.run(temp_cmd, check=True)
        else:
            p_len = str(config['model'].get('passage_max_len', 512))
            subprocess.run(cmd + ['--passage_max_len', p_len], check=True)

    # --- STEP 2: LOAD ---
    print(f"\n📂 Step 2: Loading embeddings...", flush=True)
    with open(corpus_pkl, 'rb') as f:
        c_data = pickle.load(f)
        corpus_embs = c_data[0].astype(np.float32)
        corpus_ids = [str(x) for x in c_data[1]]

    with open(query_pkl, 'rb') as f:
        q_data = pickle.load(f)
        query_embs = q_data[0].astype(np.float32)
        query_ids = [str(x) for x in q_data[1]]

    # --- STEP 3: FAISS SEARCH (Standard CPU) ---
    
    print(f"🔍 Step 3: Running FAISS IndexFlatIP (CPU)...", flush=True)
    index = faiss.IndexFlatIP(corpus_embs.shape[1])
    index.add(corpus_embs)
    scores_mat, indices_mat = index.search(query_embs, args.k)

    # --- STEP 4: VIGILANTE CHECK ---
    qrels_df = pd.read_csv(qrels_file, sep=' ', names=['qid', 'ignore', 'docid', 'rel'], dtype=str)
    print("\n" + "="*60 + "\n🕵️  VIGILANTE CHECK: QUERY 0\n" + "="*60, flush=True)
    q0_id = str(query_ids[0])
    top_doc_id = str(corpus_ids[indices_mat[0][0]])
    truth_docs = qrels_df[qrels_df['qid'] == q0_id]['docid'].values
    print(f"Query ID: {q0_id} | Top Result: {top_doc_id} | Match: {top_doc_id in truth_docs}", flush=True)
    print("="*60 + "\n", flush=True)

    # --- STEP 5: EVALUATION ---
    print("📊 Step 5: Running Evaluation Wrapper...", flush=True)
    run_results = {}
    for i in range(len(query_ids)):
        qid_str = str(query_ids[i])
        run_results[qid_str] = {str(corpus_ids[indices_mat[i][j]]): float(scores_mat[i][j]) 
                               for j in range(args.k) if indices_mat[i][j] >= 0}

    eval_qrels = pd.read_csv(qrels_file, sep=' ', names=['query_id', 'ignore', 'doc_id', 'relevance'], dtype=str)
    evaluator = TrecEvalWrapper(eval_qrels)
    metrics = evaluator.evaluate(run_results, {'recip_rank', 'ndcg_cut_10', 'recall_10'})
    
    print(f"\nFINAL RESULTS: {args.domain}\n" + "*"*40, flush=True)
    print(f"MRR:       {metrics.get('recip_rank', 0):.4f}", flush=True)
    print(f"NDCG@10:   {metrics.get('ndcg_cut_10', 0):.4f}", flush=True)
    print(f"Recall@10: {metrics.get('recall_10', 0):.4f}\n" + "*"*40, flush=True)

if __name__ == "__main__":
    main()