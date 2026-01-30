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
from tevatron.retriever.driver.train import main as tevatron_train_main
from tevatron.retriever.modeling import DenseModel

# Hardware & Project Setup
os.environ["TRANSFORMERS_ATTENTION_IMPLEMENTATION"] = "eager"
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root / 'src'))

from utils.helpers import get_path, get_training_context, load_config
from data.bright_loader import BRIGHTLoader
from data.preprocessor import BRIGHTPreprocessor
from evaluation.trec_eval_wrapper import TrecEvalWrapper

# 🩹 Tevatron Bug Patch: Ensures compatibility during model saving
if not hasattr(DenseModel, "_keys_to_ignore_on_save"):
    setattr(DenseModel, "_keys_to_ignore_on_save", None)

def diagnostic():
    """Verify hardware environment and FAISS availability."""
    print("\n" + "="*40, flush=True)
    print("🔍 ANCE HARDWARE & ENV DIAGNOSTIC", flush=True)
    print(f"PyTorch Version:  {torch.__version__}")
    print(f"CUDA Available:   {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU Device:       {torch.cuda.get_device_name(0)}")
    print(f"FAISS Version:    {faiss.__version__}")
    print("="*40 + "\n", flush=True)
    
def run_setup():
    """
    STEPS 1-4: Initialize and prepare the ReasonIR training environment.
    """
    print("🛠️ Running ANCE Setup...")
    
    # --- Step 1: Initialize Unified Logic ---
    config = load_config()
    loader = BRIGHTLoader()
    loader.load_dataset()  # Loads BRIGHT 'documents' and ReasonIR-HQ
    preprocessor = BRIGHTPreprocessor()
    
    mixture_dir = get_path("processed") / "training_mixture"
    
    # --- Step 2: Global Corpus Construction (The Haystack) ---
    print("🚀 Step 2: Constructing Master Corpus (HQ + VL)...")
    
    # 2.1 Get HQ Documents from BRIGHT (using id2doc map logic)
    id2doc_map = loader.get_all_documents_id_map()
    hq_corpus_df = pd.DataFrame([
        {"doc_id": k, "text": v} for k, v in id2doc_map.items()
    ])

    # 2.2 Get VL Documents (extracting from current VL mixture)
    vl_file = mixture_dir / "train_reasonir_vl.jsonl"
    vl_raw_df = pd.read_json(vl_file, lines=True)
    
    # Flatten all passages in VL to get unique text/id pairs
    all_vl_psgs = []
    for col in ['positive_passages', 'negative_passages']:
        for record_list in vl_raw_df[col]:
            all_vl_psgs.extend(record_list)
    
    vl_corpus_df = pd.DataFrame(all_vl_psgs).rename(columns={'docid': 'doc_id'})
    
    # 2.3 Merge and Deduplicate
    combined_corpus = pd.concat([hq_corpus_df, vl_corpus_df]).drop_duplicates(subset=['doc_id'])
    corpus_path = preprocessor.prepare_tevatron_corpus(combined_corpus, filename="reasonir_corpus.jsonl")

    # --- Step 3: Training Queries Extraction (The Questions) ---
    print("🚀 Step 3: Extracting Training Queries (No-Leakage)...")
    
    # Read queries directly from the mixture files to avoid touching eval domains
    mixture_files = list(mixture_dir.glob("*.jsonl"))
    queries_df = pd.concat([pd.read_json(f, lines=True)[['query_id', 'query']] for f in mixture_files])
    queries_df.drop_duplicates(subset=['query_id'], inplace=True)
    
    queries_path = preprocessor.prepare_tevatron_queries(queries_df, filename="train_queries.jsonl")

    # --- Step 4: ANCE QRELS Generation ---
    print("🚀 Step 4: Generating Ground-Truth QRELs for Training...")
    
    all_pos_pairs = []
    for f in mixture_files:
        temp_df = pd.read_json(f, lines=True)
        for _, row in temp_df.iterrows():
            qid = row['query_id']
            for pos_psg in row['positive_passages']:
                all_pos_pairs.append({
                    'query_id': qid,
                    'doc_id': str(pos_psg['docid']),
                    'relevance': 1
                })
    
    qrels_df = pd.DataFrame(all_pos_pairs)
    qrels_path = preprocessor.prepare_trec_qrels(qrels_df, filename="train_qrels.txt")

    return corpus_path, queries_path, qrels_path

def main():
    diagnostic()
    corpus_file, query_file, qrels_file = run_setup()
    
    ctx = get_training_context("ance")
    config = load_config()
    current_model_path = ctx['base_model']
    workdir = get_path("temp_ance")
    workdir.mkdir(exist_ok=True, parents=True)

    # Load Qrels for False Positive Filtering (Phase B)
    qrels_df = pd.read_csv(qrels_file, sep=' ', names=['query_id', 'Q0', 'doc_id', 'relevance'], dtype=str)
    qrels_dict = qrels_df.groupby('query_id')['doc_id'].apply(set).to_dict()

    for ep in range(1, ctx['args']['num_episodes'] + 1):
        print(f"\n{'='*20} 🚀 ANCE EPISODE {ep} {'='*20}", flush=True)
        ep_dir = workdir / f"ep_{ep}"
        ep_dir.mkdir(exist_ok=True)

        # --- PHASE A: ENCODE TRAINING SET ---
        corpus_pkl, query_pkl = ep_dir / "corpus.pkl", ep_dir / "query.pkl"
        for inp, outp, is_q in [(corpus_file, corpus_pkl, False), (query_file, query_pkl, True)]:
            print(f"📡 Encoding {'queries' if is_q else 'corpus'}...", flush=True)
            cmd = [
                sys.executable, '-m', 'tevatron.retriever.driver.encode',
                '--output_dir', str(outp.parent), '--model_name_or_path', current_model_path,
                '--bf16', 'True', '--per_device_eval_batch_size', str(ctx['args']['per_device_eval_batch_size']),
                '--dataset_name', 'json', '--dataset_path', str(inp), '--encode_output_path', str(outp)
            ]
            cmd += ['--encode_is_query', '--query_max_len', str(ctx['max_q'])] if is_q else ['--passage_max_len', str(ctx['max_p'])]
            subprocess.run(cmd, check=True)

        # --- PHASE B: GLOBAL MINING (FAISS CPU) ---
        print(f"🔍 Mining Hard Negatives...", flush=True)
        with open(corpus_pkl, 'rb') as f: c_data = pickle.load(f)
        with open(query_pkl, 'rb') as f: q_data = pickle.load(f)
        
        idx = faiss.IndexFlatIP(c_data[0].shape[1])
        idx.add(c_data[0].astype(np.float32))
        _, indices_mat = idx.search(q_data[0].astype(np.float32), ctx['args']['mining_depth'])

        # --- PHASE C: UPDATE MIXTURE WITH SAMPLED HARD NEGATIVES ---
        print(f"📝 Constructing mined mixture...", flush=True)
        mined_negs = {}
        c_ids = [str(x) for x in c_data[1]]
        for i, qid in enumerate([str(x) for x in q_data[1]]):
            potential_ids = [c_ids[idx] for idx in indices_mat[i] if idx >= 0]
            known_pos = qrels_dict.get(qid, set())
            # False Positive Filtering: Remove docs that are actually relevant
            true_negs = [did for did in potential_ids if did not in known_pos]
            mined_negs[qid] = random.choice(true_negs) if true_negs else potential_ids[0]

        mined_mixture_dir = ep_dir / "mined_mixture"
        mined_mixture_dir.mkdir(exist_ok=True)
        for f_path in list((get_path("processed") / "training_mixture").glob("*.jsonl")):
            if f_path.name.startswith('.'): continue
            out_f = mined_mixture_dir / f_path.name
            with open(f_path, 'r') as f_in, open(out_f, 'w') as f_out:
                for line in f_in:
                    data = json.loads(line)
                    qid = str(data['query_id'])
                    if qid in mined_negs:
                        # Swap static negative with mined negative
                        data['negative_passages'] = [{"docid": mined_negs[qid], "text": "ANCE_MINED"}]
                    f_out.write(json.dumps(data, ensure_ascii=False) + '\n')

        # --- PHASE D: TRAIN ONE EPOCH ---
        output_model_dir = get_path("models") / f"ance_ep{ep}"
        print(f"🏋️ Training Episode {ep}...", flush=True)
        training_args = [
            '--output_dir', str(output_model_dir), '--model_name_or_path', current_model_path,
            '--dataset_name', 'json', '--dataset_path', str(mined_mixture_dir / "*.jsonl"),
            '--corpus_path', str(corpus_file), '--per_device_train_batch_size', str(ctx['args']['batch_size']),
            '--train_group_size', str(ctx['args']['train_group_size']), '--learning_rate', str(ctx['args']['learning_rate']),
            '--num_train_epochs', '1', '--bf16', 'True', '--dtype', 'bfloat16', '--gradient_checkpointing', 'True', '--overwrite_output_dir', 'True'
        ]
        sys.argv = ['train.py'] + training_args
        tevatron_train_main()
        current_model_path = str(output_model_dir)

        # --- PHASE E: ZERO-SHOT EVALUATION ---
        print(f"\n📊 Zero-Shot Evaluation (Ep {ep})...", flush=True)
        eval_summary = []
        for domain in config['evaluation'].get('eval_domains', []):
            d_corpus = get_path("processed") / f"{domain}_corpus.jsonl"
            d_queries = get_path("processed") / f"{domain}_queries.jsonl"
            d_qrels = get_path("processed") / f"{domain}_qrels.txt"
            d_eval_dir = ep_dir / "eval" / domain
            d_eval_dir.mkdir(parents=True, exist_ok=True)
            dc_pkl, dq_pkl = d_eval_dir / "c.pkl", d_eval_dir / "q.pkl"

            for inp, outp, is_q in [(d_corpus, dc_pkl, False), (d_queries, dq_pkl, True)]:
                cmd = [sys.executable, '-m', 'tevatron.retriever.driver.encode', '--output_dir', str(outp.parent), '--model_name_or_path', current_model_path, '--bf16', 'True', '--per_device_eval_batch_size', str(ctx['args']['per_device_eval_batch_size']), '--dataset_name', 'json', '--dataset_path', str(inp), '--encode_output_path', str(outp)]
                cmd += ['--encode_is_query', '--query_max_len', str(ctx['max_q'])] if is_q else ['--passage_max_len', str(ctx['max_p'])]
                subprocess.run(cmd, check=True)

            with open(dc_pkl, 'rb') as f: dc_d = pickle.load(f)
            with open(dq_pkl, 'rb') as f: dq_d = pickle.load(f)
            idx = faiss.IndexFlatIP(dc_d[0].shape[1])
            idx.add(dc_d[0].astype(np.float32))
            s, i = idx.search(dq_d[0].astype(np.float32), 10)
            
            run_results = {str(dq_d[1][idx_q]): {str(dc_d[1][i[idx_q][idx_doc]]): float(s[idx_q][idx_doc]) for idx_doc in range(len(i[idx_q])) if i[idx_q][idx_doc] >= 0} for idx_q in range(len(dq_d[1]))}
            evaluator = TrecEvalWrapper(pd.read_csv(d_qrels, sep=' ', names=['query_id', 'Q0', 'doc_id', 'relevance'], dtype=str))
            m = evaluator.evaluate(run_results, {'recip_rank', 'ndcg_cut_10'})
            eval_summary.append({'domain': domain, 'ndcg10': m.get('ndcg_cut_10', 0)})
        
        print(f"📈 Episode {ep} Mean NDCG@10: {pd.DataFrame(eval_summary)['ndcg10'].mean():.4f}", flush=True)

if __name__ == "__main__":
    main()