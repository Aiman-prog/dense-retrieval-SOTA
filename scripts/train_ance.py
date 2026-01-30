import os
import sys
import json
import random
import subprocess
import numpy as np
import pandas as pd
import faiss
import torch
import pickle
from pathlib import Path

# Hardware Fix for A100/DelftBlue
os.environ["TRANSFORMERS_ATTENTION_IMPLEMENTATION"] = "eager"

# Standard Project Imports
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root / 'src'))

from utils.helpers import get_path, get_training_context, load_config
from data.bright_loader import BRIGHTLoader
from data.preprocessor import BRIGHTPreprocessor

def diagnostic():
    """Environment check for Slurm logs, inspired by evaluate.py."""
    print("\n" + "="*40, flush=True)
    print("🔍 ANCE HARDWARE & ENV DIAGNOSTIC", flush=True)
    print("="*40, flush=True)
    cuda_available = torch.cuda.is_available()
    print(f"PyTorch Version:  {torch.__version__}", flush=True)
    print(f"CUDA Available:   {cuda_available}", flush=True)
    if cuda_available:
        print(f"GPU Device:       {torch.cuda.get_device_name(0)}", flush=True)
    print(f"FAISS Version:    {faiss.__version__}", flush=True)
    print("="*40 + "\n", flush=True)

def run_setup():
    """
    STEPS 1-4: Initialize and prepare the ReasonIR training environment.
    Uses existing Loader and Preprocessor methods to avoid reinventing the wheel.
    """
    print("🛠️  STARTING ANCE SETUP", flush=True)
    
    # --- Step 1: Initialize Unified Logic ---
    config = load_config()
    loader = BRIGHTLoader()
    loader.load_dataset()  # Fetches BRIGHT 'documents' and ReasonIR-HQ
    preprocessor = BRIGHTPreprocessor()
    
    mixture_dir = get_path("processed") / "training_mixture"
    
    # --- Step 2: Global Corpus Construction (The Haystack) ---
    print("\n🚀 Step 2: Constructing Master Corpus (HQ + VL)...", flush=True)
    
    # 2.1 Get HQ Documents from BRIGHT (Standard mapping)
    id2doc_map = loader.get_all_documents_id_map()
    hq_corpus_df = pd.DataFrame([
        {"doc_id": str(k), "text": v} for k, v in id2doc_map.items()
    ])

    # 2.2 Get VL Documents (extracting from current VL mixture)
    vl_file = mixture_dir / "train_reasonir_vl.jsonl"
    if not vl_file.exists():
        raise FileNotFoundError(f"❌ Could not find VL training file at {vl_file}")
        
    vl_raw_df = pd.read_json(vl_file, lines=True)
    
    # Flatten passages to get unique docid/text pairs for VL
    all_vl_psgs = []
    for col in ['positive_passages', 'negative_passages']:
        if col in vl_raw_df.columns:
            for record_list in vl_raw_df[col]:
                all_vl_psgs.extend(record_list)
    
    vl_corpus_df = pd.DataFrame(all_vl_psgs).rename(columns={'docid': 'doc_id'})
    
    # 2.3 Merge and Deduplicate to create the ReasonIR Universe
    combined_corpus = pd.concat([hq_corpus_df, vl_corpus_df]).drop_duplicates(subset=['doc_id'])
    corpus_path = preprocessor.prepare_tevatron_corpus(combined_corpus, filename="reasonir_corpus.jsonl")
    print(f"✅ Created Master Corpus: {len(combined_corpus):,} docs", flush=True)

    # --- Step 3: Training Queries Extraction (No-Leakage) ---
    print("\n🚀 Step 3: Extracting Training Queries...", flush=True)
    
    all_queries = []
    # Filter only .jsonl files and ignore hidden files
    mixture_files = [f for f in mixture_dir.glob("*.jsonl") if not f.name.startswith('.')]
    
    for f in mixture_files:
        try:
            temp_df = pd.read_json(f, lines=True)
            # Clean column names (remove whitespace)
            temp_df.columns = temp_df.columns.astype(str).str.strip()
            
            if 'query' in temp_df.columns:
                all_queries.append(temp_df[['query_id', 'query']])
            else:
                print(f"⚠️  Warning: 'query' column missing in {f.name}. Found: {temp_df.columns.tolist()}", flush=True)
        except Exception as e:
            print(f"❌ Error reading {f.name}: {e}", flush=True)

    if not all_queries:
        raise ValueError("❌ No valid queries found in training mixture files.")
        
    queries_df = pd.concat(all_queries).drop_duplicates(subset=['query_id'])
    queries_path = preprocessor.prepare_tevatron_queries(queries_df, filename="train_queries.jsonl")
    print(f"✅ Extracted {len(queries_df):,} unique training queries", flush=True)

    # --- Step 4: ANCE QRELS Generation ---
    print("\n🚀 Step 4: Generating Ground-Truth QRELs for Training...", flush=True)
    
    all_pos_pairs = []
    for f in mixture_files:
        try:
            temp_df = pd.read_json(f, lines=True)
            if 'query_id' in temp_df.columns and 'positive_passages' in temp_df.columns:
                for _, row in temp_df.iterrows():
                    qid = str(row['query_id'])
                    for pos_psg in row['positive_passages']:
                        all_pos_pairs.append({
                            'query_id': qid,
                            'doc_id': str(pos_psg['docid']),
                            'relevance': 1
                        })
        except Exception:
            continue
            
    qrels_df = pd.DataFrame(all_pos_pairs).drop_duplicates()
    qrels_path = preprocessor.prepare_trec_qrels(qrels_df, filename="train_qrels.txt")

    return corpus_path, queries_path, qrels_path

def main():
    diagnostic()
    
    # Execute Steps 1-4
    corpus_file, query_file, qrels_file = run_setup()
    
    # Load training context for ANCE
    ctx = get_training_context("ance")
    
    print("\n" + "="*40, flush=True)
    print("✅ INITIAL SETUP COMPLETE", flush=True)
    print("="*40, flush=True)
    print(f"Corpus:  {corpus_file}")
    print(f"Queries: {query_file}")
    print(f"Qrels:   {qrels_file}")
    print("="*40 + "\n", flush=True)
    
    # NEXT STEPS (To be implemented in the episode loop):
    # 1. Encode corpus and queries using current model.
    # 2. Global Mining via FAISS (sample hard negatives).
    # 3. Update mixture JSONLs.
    # 4. Train 1 Epoch.

if __name__ == "__main__":
    main()