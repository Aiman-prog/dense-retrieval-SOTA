import os
import sys
import gc
import json
import argparse
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

from utils.helpers import get_path, get_training_context, load_config, \
                          encode_to_pickle, build_faiss_index, patch_tevatron_loss
from data.preprocessor import run_setup

# 🩹 Tevatron Bug Patch
if not hasattr(DenseModel, "_keys_to_ignore_on_save"):
    setattr(DenseModel, "_keys_to_ignore_on_save", None)


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

        # --- PHASE E: EVALUATE ---
        for domain in config['evaluation'].get('eval_domains', []):
            subprocess.run([
                sys.executable, str(project_root / 'src/evaluation/evaluate.py'),
                '--model_path', current_model_path,
                '--domain', domain,
            ], check=True)

        scores = [
            json.load(open(get_path("results") / f"{domain}_results.json"))['metrics'].get('ndcg_cut_10', 0)
            for domain in config['evaluation'].get('eval_domains', [])
        ]
        print(f"📈 Ep {ep} Mean NDCG@10: {sum(scores) / len(scores):.4f}", flush=True)

if __name__ == "__main__":
    main()