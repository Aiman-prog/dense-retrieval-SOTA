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

# 🩹 Tevatron Bug Patch
if not hasattr(DenseModel, "_keys_to_ignore_on_save"):
    setattr(DenseModel, "_keys_to_ignore_on_save", None)

def run_setup():
    """Logic for Steps 1-4: Skips if files exist."""
    corpus_path = get_path("processed") / "reasonir_corpus.jsonl"
    queries_path = get_path("processed") / "train_queries.jsonl"
    qrels_path = get_path("processed") / "train_qrels.txt"
    mixture_dir = get_path("processed") / "training_mixture"

    if all(p.exists() and p.stat().st_size > 0 for p in [corpus_path, queries_path, qrels_path]):
        print("⏩ Skipping setup: Target files found.", flush=True)
        return corpus_path, queries_path, qrels_path

    print("🛠️ Running ANCE Setup...", flush=True)
    loader = BRIGHTLoader()
    loader.load_dataset()
    preprocessor = BRIGHTPreprocessor()
    
    # Corpus
    id2doc_map = loader.get_all_documents_id_map()
    hq_df = pd.DataFrame([{"doc_id": str(k), "text": v} for k, v in id2doc_map.items()])
    vl_df = pd.read_json(mixture_dir / "train_reasonir_vl.jsonl", lines=True)
    all_vl = []
    for col in ['positive_passages', 'negative_passages']:
        for record_list in vl_df[col]: all_vl.extend(record_list)
    vl_corpus = pd.DataFrame(all_vl).rename(columns={'docid': 'doc_id'})
    combined = pd.concat([hq_df, vl_corpus]).drop_duplicates(subset=['doc_id'])
    preprocessor.prepare_tevatron_corpus(combined, filename="reasonir_corpus.jsonl")

    # Queries
    mix_files = [f for f in mixture_dir.glob("*.jsonl") if not f.name.startswith('.')]
    q_dfs = []
    for f in mix_files:
        df = pd.read_json(f, lines=True)
        if 'query_text' in df.columns: df = df.rename(columns={'query_text': 'query'})
        q_dfs.append(df[['query_id', 'query']])
    queries_df = pd.concat(q_dfs).drop_duplicates(subset=['query_id'])
    preprocessor.prepare_tevatron_queries(queries_df, filename="train_queries.jsonl")

    # Qrels
    pos_pairs = []
    for f in mix_files:
        df = pd.read_json(f, lines=True)
        for _, row in df.iterrows():
            for pos in row['positive_passages']:
                pos_pairs.append({'query_id': str(row['query_id']), 'doc_id': str(pos['docid']), 'relevance': 1})
    preprocessor.prepare_trec_qrels(pd.DataFrame(pos_pairs).drop_duplicates(), filename="train_qrels.txt")

    return corpus_path, queries_path, qrels_path

def main():
    corpus_file, query_file, qrels_file = run_setup()
    ctx = get_training_context("ance")
    config = load_config()
    current_model_path = ctx['base_model']
    workdir = get_path("temp_ance")
    workdir.mkdir(exist_ok=True, parents=True)

    # Robust Qrel Load (Inspiration: evaluate.py file reading)
    qrels_data = []
    with open(qrels_file, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 4: qrels_data.append({'qid': parts[0], 'did': parts[2]})
    qrels_dict = pd.DataFrame(qrels_data).groupby('qid')['did'].apply(set).to_dict()

    for ep in range(1, ctx['args']['num_episodes'] + 1):
        print(f"\n🚀 ANCE EPISODE {ep}", flush=True)
        ep_dir = workdir / f"ep_{ep}"
        ep_dir.mkdir(exist_ok=True)

        # --- PHASE A: ENCODE (Inspiration: evaluate.py logic) ---
        for inp, outp, is_q in [(corpus_file, ep_dir/"corpus.pkl", False), (query_file, ep_dir/"query.pkl", True)]:
            cmd = [
                sys.executable, '-m', 'tevatron.retriever.driver.encode',
                '--output_dir', str(outp.parent), '--model_name_or_path', current_model_path,
                '--bf16', 'True', '--fp16', 'False', 
                '--per_device_eval_batch_size', str(ctx['args']['per_device_eval_batch_size']),
                '--dataset_name', 'json', '--dataset_path', str(inp),
                '--encode_output_path', str(outp), '--attn_implementation', 'eager',
                '--dataloader_num_workers', str(ctx['args']['dataloader_num_workers']),
                '--pooling', ctx['pooling'],
            ]
            if is_q:
                q_len = str(config['model'].get('query_max_len', 128))
                # Tevatron Version Fallback Logic
                try:
                    subprocess.run(cmd + ['--encode_is_query', '--query_max_len', q_len], check=True)
                except subprocess.CalledProcessError:
                    subprocess.run(cmd + ['--encode_is_qry', '--q_max_len', q_len], check=True)
            else:
                subprocess.run(cmd + ['--passage_max_len', str(config['model'].get('passage_max_len', 512))], check=True)

        # --- PHASE B & C: MINE & UPDATE ---
        with open(ep_dir/"corpus.pkl", 'rb') as f: c_data = pickle.load(f)
        with open(ep_dir/"query.pkl", 'rb') as f: q_data = pickle.load(f)
        idx = faiss.IndexFlatIP(c_data[0].shape[1]); idx.add(c_data[0].astype(np.float32))
        _, indices = idx.search(q_data[0].astype(np.float32), ctx['args']['mining_depth'])
        
        mined_negs = {}
        c_ids = [str(x) for x in c_data[1]]
        for i, qid in enumerate([str(x) for x in q_data[1]]):
            pot = [c_ids[idx] for idx in indices[i] if idx >= 0]
            true_negs = [d for d in pot if d not in qrels_dict.get(qid, set())]
            mined_negs[qid] = random.choice(true_negs) if true_negs else pot[0]

        mix_out = ep_dir / "mined_mixture"; mix_out.mkdir(exist_ok=True)
        for f_path in (get_path("processed") / "training_mixture").glob("*.jsonl"):
            if f_path.name.startswith('.'): continue
            with open(f_path, 'r') as f_in, open(mix_out/f_path.name, 'w') as f_out:
                for line in f_in:
                    d = json.loads(line)
                    if str(d['query_id']) in mined_negs:
                        d['negative_passages'] = [{"docid": mined_negs[str(d['query_id'])], "text": "ANCE_MINED"}]
                    f_out.write(json.dumps(d, ensure_ascii=False) + '\n')

        # --- PHASE D: TRAIN (Inspiration: train_inbatch.py logic) ---
        output_model_dir = get_path("models") / f"ance_ep{ep}"
        training_args = [
            '--output_dir', str(output_model_dir), '--model_name_or_path', current_model_path,
            '--dataset_name', 'json', '--dataset_path', str(mix_out / "*.jsonl"),
            '--corpus_path', str(corpus_file), '--per_device_train_batch_size', str(ctx['args']['batch_size']),
            '--train_group_size', str(ctx['args']['train_group_size']), '--learning_rate', str(ctx['args']['learning_rate']),
            '--num_train_epochs', str(ctx['args']['num_epochs']), '--bf16', 'True', '--dtype', 'bfloat16',
            '--gradient_checkpointing', str(ctx['args']['gradient_checkpointing']),
            '--overwrite_output_dir', 'True',   # Clears "toxic" old settings
            '--save_strategy', ctx['args']['save_strategy'],            # Stops 100GB checkpoint spike
            '--save_total_limit', str(ctx['args']['save_total_limit']),          # Safety: only keep 1 model copy
            '--ignore_data_skip', 'True',       # Forces batch size 64 (resets counter)
            '--attn_implementation', 'eager', '--optim', 'adamw_torch_fused', '--logging_steps', str(ctx['args']['logging_steps']),
            '--pooling', ctx['pooling'],
            '--normalize', str(ctx['normalize']),
            '--temperature', str(ctx['temperature']),
        ]
        sys.argv = ['train.py'] + training_args
        tevatron_train_main()
        current_model_path = str(output_model_dir)

        # --- PHASE E: EVALUATE (Strict Inspiration: evaluate.py loop) ---
        eval_summary = []
        for domain in config['evaluation'].get('eval_domains', []):
            d_corpus = get_path("processed") / f"{domain}_corpus.jsonl"
            d_queries = get_path("processed") / f"{domain}_queries.jsonl"
            d_qrels = get_path("processed") / f"{domain}_qrels.txt"
            d_eval = ep_dir / "eval" / domain; d_eval.mkdir(parents=True, exist_ok=True)
            
            # Re-use Phase A encoding style with version fallback
            for inp, outp, is_q in [(d_corpus, d_eval/"c.pkl", False), (d_queries, d_eval/"q.pkl", True)]:
                cmd_eval = [sys.executable, '-m', 'tevatron.retriever.driver.encode', '--output_dir', str(outp.parent), '--model_name_or_path', current_model_path, '--bf16', 'True', '--fp16', 'False', '--per_device_eval_batch_size', str(ctx['args']['per_device_eval_batch_size']), '--dataset_name', 'json', '--dataset_path', str(inp), '--encode_output_path', str(outp), '--attn_implementation', 'eager', '--pooling', ctx['pooling']]
                if is_q:
                    try: subprocess.run(cmd_eval + ['--encode_is_query', '--query_max_len', q_len], check=True)
                    except subprocess.CalledProcessError: subprocess.run(cmd_eval + ['--encode_is_qry', '--q_max_len', q_len], check=True)
                else: subprocess.run(cmd_eval + ['--passage_max_len', str(config['model'].get('passage_max_len', 512))], check=True)

            with open(d_eval/"c.pkl", 'rb') as f: dc = pickle.load(f)
            with open(d_eval/"q.pkl", 'rb') as f: dq = pickle.load(f)
            idx_e = faiss.IndexFlatIP(dc[0].shape[1]); idx_e.add(dc[0].astype(np.float32))
            eval_top_k = ctx['args'].get('eval_top_k', 10)
            s_e, i_e = idx_e.search(dq[0].astype(np.float32), eval_top_k)
            
            results = {str(dq[1][j]): {str(dc[1][i_e[j][k]]): float(s_e[j][k]) for k in range(len(i_e[j])) if i_e[j][k] >= 0} for j in range(len(dq[1]))}
            eval_qrels_data = []
            with open(d_qrels, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 4:
                        eval_qrels_data.append({'query_id': parts[0], 'doc_id': parts[2], 'relevance': parts[3]})
            eval_eval_df = pd.DataFrame(eval_qrels_data)
            evaluator = TrecEvalWrapper(eval_eval_df)
            metrics = evaluator.evaluate(results, {'recip_rank', 'ndcg_cut_10'})
            eval_summary.append({'domain': domain, 'ndcg10': metrics.get('ndcg_cut_10', 0)})
        
        print(f"📈 Ep {ep} Mean NDCG@10: {pd.DataFrame(eval_summary)['ndcg10'].mean():.4f}", flush=True)

if __name__ == "__main__":
    main()