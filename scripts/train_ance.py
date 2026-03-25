import os
import sys
import gc
import json
import random
import argparse
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

from utils.helpers import get_path, get_training_context, load_config, \
                          encode_to_pickle, build_faiss_index, patch_tevatron_loss
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
    preprocessor = BRIGHTPreprocessor()
    mix_files = [f for f in mixture_dir.glob("*.jsonl") if not f.name.startswith('.')]

    # Read all mixture files once
    mix_dfs = []
    for f in mix_files:
        df = pd.read_json(f, lines=True)
        if 'query_text' in df.columns: df = df.rename(columns={'query_text': 'query'})
        mix_dfs.append(df)
    mix_df = pd.concat(mix_dfs, ignore_index=True)

    # Corpus: all passages from mixture files (same source as crossbatch/inbatch)
    all_passages = []
    for col in ['positive_passages', 'negative_passages']:
        for record_list in mix_df[col]:
            all_passages.extend(record_list)
    corpus_df = pd.DataFrame(all_passages).rename(columns={'docid': 'doc_id'})[['doc_id', 'text']].drop_duplicates(subset=['doc_id'])
    preprocessor.prepare_tevatron_corpus(corpus_df, filename="reasonir_corpus.jsonl")
    print(f"Corpus: {len(corpus_df)} passages", flush=True)

    # Queries
    queries_df = mix_df[['query_id', 'query']].drop_duplicates(subset=['query_id'])
    preprocessor.prepare_tevatron_queries(queries_df, filename="train_queries.jsonl")

    # Qrels
    pos_pairs = []
    for _, row in mix_df.iterrows():
        for pos in row['positive_passages']:
            pos_pairs.append({'query_id': str(row['query_id']), 'doc_id': str(pos['docid']), 'relevance': 1})
    preprocessor.prepare_trec_qrels(pd.DataFrame(pos_pairs).drop_duplicates(), filename="train_qrels.txt")

    return corpus_path, queries_path, qrels_path

def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--start_episode', type=int, default=1)
    cli_args, _ = parser.parse_known_args()
    start_ep = cli_args.start_episode

    corpus_file, query_file, qrels_file = run_setup()

    # Load corpus text lookup once — used in Phase C to write real hard negative text
    corpus_lookup = {}
    with open(corpus_file) as f:
        for line in f:
            d = json.loads(line)
            corpus_lookup[d['docid']] = d['text']
    print(f"Loaded corpus lookup: {len(corpus_lookup)} passages", flush=True)

    ctx = get_training_context("ance")
    config = load_config()
    current_model_path = ctx['base_model']
    if start_ep > 1:
        current_model_path = str(get_path("models") / f"ance_ep{start_ep - 1}")
        print(f"▶️  Resuming ANCE from episode {start_ep}, using model: {current_model_path}", flush=True)
    workdir = get_path("temp_ance")
    workdir.mkdir(exist_ok=True, parents=True)

    # Robust Qrel Load (Inspiration: evaluate.py file reading)
    qrels_data = []
    with open(qrels_file, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 4: qrels_data.append({'qid': parts[0], 'did': parts[2]})
    qrels_dict = pd.DataFrame(qrels_data).groupby('qid')['did'].apply(set).to_dict()

    for ep in range(start_ep, ctx['args']['num_episodes'] + 1):
        print(f"\n🚀 ANCE EPISODE {ep}", flush=True)
        ep_dir = workdir / f"ep_{ep}"
        ep_dir.mkdir(exist_ok=True)

        # --- PHASE A: ENCODE ---
        encode_to_pickle(current_model_path, corpus_file, ep_dir/"corpus.pkl", False, ctx, config)
        encode_to_pickle(current_model_path, query_file,  ep_dir/"query.pkl",  True,  ctx, config)

        # --- PHASE B & C: MINE & UPDATE ---
        idx, _, c_ids = build_faiss_index(ep_dir/"corpus.pkl")
        with open(ep_dir/"query.pkl", 'rb') as f: q_data = pickle.load(f)
        _, indices = idx.search(q_data[0].astype(np.float32), ctx['args']['mining_depth'])
        
        n_negs = ctx['args']['train_group_size'] - 1  # e.g. 5 when train_group_size=6
        mined_negs = {}
        for i, qid in enumerate([str(x) for x in q_data[1]]):
            pot = [c_ids[j] for j in indices[i] if j >= 0]
            true_negs = [d for d in pot if d not in qrels_dict.get(qid, set())]
            candidates = true_negs if true_negs else pot
            # Top-n hardest: earliest in FAISS-ranked list = highest similarity
            if len(candidates) >= n_negs:
                mined_negs[qid] = candidates[:n_negs]
            else:
                # Pad by repeating available candidates
                mined_negs[qid] = (candidates * (n_negs // len(candidates) + 1))[:n_negs]

        mix_out = ep_dir / "mined_mixture"; mix_out.mkdir(exist_ok=True)
        for f_path in (get_path("processed") / "training_mixture").glob("*.jsonl"):
            if f_path.name.startswith('.'): continue
            with open(f_path, 'r') as f_in, open(mix_out/f_path.name, 'w') as f_out:
                for line in f_in:
                    d = json.loads(line)
                    if str(d['query_id']) in mined_negs:
                        d['negative_passages'] = [
                            {"docid": neg_id, "text": corpus_lookup.get(neg_id, "")}
                            for neg_id in mined_negs[str(d['query_id'])]
                        ]
                    f_out.write(json.dumps(d, ensure_ascii=False) + '\n')

        # --- PHASE D: TRAIN (Inspiration: train_inbatch.py logic) ---
        output_model_dir = get_path("models") / f"ance_ep{ep}"
        training_args = [
            '--output_dir', str(output_model_dir), '--model_name_or_path', current_model_path,
            '--dataset_name', 'json', '--dataset_path', str(mix_out / "*.jsonl"),
            '--dataset_split', 'train', '--per_device_train_batch_size', str(ctx['args']['batch_size']),
            '--train_group_size', str(ctx['args']['train_group_size']), '--learning_rate', str(ctx['args']['learning_rate']),
            '--num_train_epochs', str(ctx['args']['num_epochs']), '--bf16', 'True', '--dtype', 'bfloat16',
            '--overwrite_output_dir', 'True',   # Clears "toxic" old settings
            '--save_strategy', ctx['args']['save_strategy'],
            '--save_steps', str(ctx['args'].get('save_steps', 500)),
            '--save_total_limit', str(ctx['args']['save_total_limit']),
            '--ignore_data_skip', 'True',       # Forces batch size 64 (resets counter)
            '--warmup_ratio', str(ctx['args'].get('warmup_ratio', 0.0)),
            '--weight_decay', str(ctx['args'].get('weight_decay', 0.0)),
            '--max_grad_norm', str(ctx['args'].get('max_grad_norm', 1.0)),
            '--dataloader_num_workers', str(ctx['args']['dataloader_num_workers']),
            '--attn_implementation', 'eager', '--optim', 'adamw_torch_fused', '--logging_steps', str(ctx['args']['logging_steps']),
            '--pooling', ctx['pooling'],
            '--normalize', str(ctx['normalize']),
            '--temperature', str(ctx['temperature']),
        ]
        sys.argv = ['train.py'] + training_args
        
        patch_tevatron_loss(ctx['temperature'])
        tevatron_train_main()
        current_model_path = str(output_model_dir)

        # Free training model from GPU before encoding subprocesses start next episode
        gc.collect()
        torch.cuda.empty_cache()

        # --- PHASE E: EVALUATE (Strict Inspiration: evaluate.py loop) ---
        eval_summary = []
        for domain in config['evaluation'].get('eval_domains', []):
            d_corpus = get_path("processed") / f"{domain}_corpus.jsonl"
            d_queries = get_path("processed") / f"{domain}_queries.jsonl"
            d_qrels = get_path("processed") / f"{domain}_qrels.txt"
            d_eval = ep_dir / "eval" / domain; d_eval.mkdir(parents=True, exist_ok=True)
            
            # Re-use encode_to_pickle for eval encoding
            encode_to_pickle(current_model_path, d_corpus,  d_eval/"c.pkl", False, ctx, config)
            encode_to_pickle(current_model_path, d_queries, d_eval/"q.pkl", True,  ctx, config)

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