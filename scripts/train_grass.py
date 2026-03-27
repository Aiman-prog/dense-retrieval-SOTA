import os
import sys
import gc
import json
import subprocess
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from transformers import AutoTokenizer, AutoModel
from tevatron.retriever.driver.train import main as tevatron_train_main
from tevatron.retriever.modeling import DenseModel

# Hardware & Project Setup
os.environ["TRANSFORMERS_ATTENTION_IMPLEMENTATION"] = "eager"
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root / 'src'))

from utils.helpers import get_path, get_training_context, load_config, \
                          encode_to_pickle, build_faiss_index, patch_tevatron_loss
from data.preprocessor import run_setup

# Tevatron Bug Patch
if not hasattr(DenseModel, "_keys_to_ignore_on_save"):
    setattr(DenseModel, "_keys_to_ignore_on_save", None)


def encode_batch(model, tokenizer, texts, device, max_len, batch_size):
    """
    Encode a list of texts in batches using the given model.
    Model must already be in train() mode for MC-dropout to be active.
    Returns a (N, dim) float32 numpy array of L2-normalized CLS embeddings.
    """
    all_embs = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        inputs = tokenizer(batch, padding=True, truncation=True,
                           max_length=max_len, return_tensors='pt').to(device)
        with torch.no_grad():
            out = model(**inputs)
        embs = out.last_hidden_state[:, 0, :]  # CLS pooling
        embs = torch.nn.functional.normalize(embs, dim=-1)
        all_embs.append(embs.cpu().float().numpy())
    return np.concatenate(all_embs, axis=0)


def grass_sampler(model_path, stale_idx, c_ids, corpus_lookup, mix_df,
                  qrels_dict, cfg, config, out_dir):
    """
    GrassSampler (Algorithm 2): mines hard negatives for all training queries using
    a stale ANN index and MC-dropout uncertainty estimation.

    For each training query:
      1. Retrieve top-P candidate doc IDs from the stale ANN index (index can be stale —
         Algorithm 2, Line 1). True positives from qrels are filtered out.
      2. Encode all unique candidates and queries T times with dropout enabled (MC-dropout).
         This gives T sets of embeddings sampled from the model's approximate posterior.
      3. Compute per-candidate statistics across T passes:
           s_hat(q, d) = mean similarity  — estimates expected hardness
           sigma(q, d) = std similarity   — estimates model uncertainty about this doc
      4. Score each candidate: g(q, d) = s_hat + lambda * sigma
         High g means the doc is both confusing (hard) and uncertain (informative).
      5. Select the top-m candidates by g as the new hard negatives for this query.

    Queries are processed in batches (query_batch_size) to keep peak GPU memory bounded.
    Writes updated mixture JSONL files to out_dir with negative_passages replaced.
    """
    P                = cfg['P']
    m                = cfg['m']
    T                = cfg['T']
    lambda_val       = cfg['lambda_val']
    mc_batch_size    = cfg.get('mc_batch_size', 256)
    query_batch_size = cfg.get('query_batch_size', 64)
    q_max_len        = config['model']['query_max_len']
    p_max_len        = config['model']['passage_max_len']

    device    = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model     = AutoModel.from_pretrained(model_path, torch_dtype=torch.bfloat16).to(device)
    model.train()  # enables dropout for MC passes
    print(f"  Loaded model for GrassSampler (T={T}, P={P}, m={m}, lambda={lambda_val})", flush=True)

    query_ids  = mix_df['query_id'].astype(str).tolist()
    query_texts = mix_df['query'].tolist()
    n_queries  = len(query_ids)
    n_batches  = (n_queries + query_batch_size - 1) // query_batch_size
    print(f"  Processing {n_queries} queries in {n_batches} batches (batch_size={query_batch_size})...", flush=True)

    # Process queries in batches to avoid holding (T × N_all_queries × dim) in memory.
    # Peak memory per batch: T × query_batch_size × P × dim (candidates).
    mined_negs = {}
    for b, batch_start in enumerate(range(0, n_queries, query_batch_size)):
        batch_ids   = query_ids[batch_start:batch_start + query_batch_size]
        batch_texts = query_texts[batch_start:batch_start + query_batch_size]

        # Encode this batch of queries T times with dropout active — (T, B, dim)
        q_embs_all   = [encode_batch(model, tokenizer, batch_texts, device, q_max_len, mc_batch_size)
                        for _ in range(T)]
        q_embs_stack = np.stack(q_embs_all, axis=0)  # (T, B, dim)

        # Use mean query embedding across T passes for stable ANN retrieval
        q_embs_mean = q_embs_stack.mean(axis=0).astype(np.float32)
        _, indices  = stale_idx.search(q_embs_mean, P)

        # Collect unique candidate doc IDs for this batch, filtering true positives
        batch_query_cands = {}
        batch_cand_ids_set = set()
        for i, qid in enumerate(batch_ids):
            cands     = [c_ids[j] for j in indices[i] if j >= 0]
            true_negs = [d for d in cands if d not in qrels_dict.get(qid, set())]
            batch_query_cands[qid] = true_negs if true_negs else cands
            batch_cand_ids_set.update(batch_query_cands[qid])

        batch_cand_ids    = list(batch_cand_ids_set)
        batch_cand_texts  = [corpus_lookup.get(did, "") for did in batch_cand_ids]
        batch_cand_to_idx = {did: i for i, did in enumerate(batch_cand_ids)}

        # Encode all unique candidates for this batch T times — (T, N_batch_cands, dim)
        c_embs_all   = [encode_batch(model, tokenizer, batch_cand_texts, device, p_max_len, mc_batch_size)
                        for _ in range(T)]
        c_embs_stack = np.stack(c_embs_all, axis=0)  # (T, N_batch_cands, dim)

        # Compute g = s_hat + lambda * sigma and select top-m for each query in this batch
        for i, qid in enumerate(batch_ids):
            cands = batch_query_cands[qid]
            if not cands:
                continue
            cand_idxs = [batch_cand_to_idx[d] for d in cands]

            # q_t: (T, 1, dim)  @  c_t: (T, dim, N_cands_q)  →  (T, N_cands_q)
            q_t  = torch.from_numpy(q_embs_stack[:, i, :]).unsqueeze(1)      # (T, 1, dim)
            c_t  = torch.from_numpy(c_embs_stack[:, cand_idxs, :])           # (T, N_cands_q, dim)
            sims = torch.bmm(q_t, c_t.transpose(1, 2)).squeeze(1).numpy()    # (T, N_cands_q)

            s_hat = sims.mean(axis=0)  # expected similarity — measures hardness
            sigma = sims.std(axis=0)   # std across passes  — measures model uncertainty
            g     = s_hat + lambda_val * sigma

            top_m = np.argsort(g)[::-1][:m]
            mined_negs[qid] = [cands[k] for k in top_m]

        if (b + 1) % 10 == 0:
            print(f"  Batch {b + 1}/{n_batches} done.", flush=True)

    # Free model from GPU before Tevatron training loads it
    del model
    gc.collect()
    torch.cuda.empty_cache()

    # Write updated mixture files — same structure as original, negative_passages replaced
    out_dir.mkdir(exist_ok=True, parents=True)
    for f_path in (get_path("processed") / "training_mixture").glob("*.jsonl"):
        if f_path.name.startswith('.'): continue
        with open(f_path, 'r') as f_in, open(out_dir / f_path.name, 'w') as f_out:
            for line in f_in:
                d   = json.loads(line)
                qid = str(d['query_id'])
                if qid in mined_negs:
                    d['negative_passages'] = [
                        {"docid": neg_id, "text": corpus_lookup.get(neg_id, "")}
                        for neg_id in mined_negs[qid]
                    ]
                f_out.write(json.dumps(d, ensure_ascii=False) + '\n')

    print(f"  GrassSampler done. Negatives updated for {len(mined_negs)} queries.", flush=True)


def main():
    corpus_file, query_file, qrels_file = run_setup()

    # Load corpus text lookup — used to retrieve passage text when writing mined negatives
    corpus_lookup = {}
    with open(corpus_file) as f:
        for line in f:
            d = json.loads(line)
            corpus_lookup[d['docid']] = d['text']
    print(f"Loaded corpus lookup: {len(corpus_lookup)} passages", flush=True)

    ctx    = get_training_context("grass")
    config = load_config()
    cfg    = config['training']['grass']

    workdir = get_path("base") / "temp_grass_workdir"
    workdir.mkdir(exist_ok=True, parents=True)

    # Load qrels for positive filtering during GrassSampler
    qrels_data = []
    with open(qrels_file, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 4: qrels_data.append({'qid': parts[0], 'did': parts[2]})
    qrels_dict = pd.DataFrame(qrels_data).groupby('qid')['did'].apply(set).to_dict()

    # Load deduplicated training queries for GrassSampler encoding.
    # run_setup() already extracted unique queries and wrote them to train_queries.jsonl.
    mix_df = pd.read_json(query_file, lines=True)

    print(f"✅ Setup complete. corpus_lookup={len(corpus_lookup)} passages, qrels={len(qrels_dict)} queries, mix_df={len(mix_df)} unique queries.", flush=True)

    # Build stale ANN index from base model — never refreshed (Algorithm 2, Line 1)
    stale_dir = workdir / "stale_index"
    stale_dir.mkdir(exist_ok=True)
    stale_pkl = stale_dir / "corpus.pkl"
    if not stale_pkl.exists():
        print("📦 Building stale ANN index from base model...", flush=True)
        encode_to_pickle(cfg['base_model'], corpus_file, stale_pkl, False, ctx, config)
    stale_idx, _, c_ids = build_faiss_index(stale_pkl)
    print(f"✅ Stale index ready: {len(c_ids)} passages", flush=True)

    # # --- GRASSSAMPLER: mine hard negatives using stale ANN + MC-dropout ---
    # mix_out = workdir / "grass_train"
    # print("🔍 Running GrassSampler...", flush=True)
    # grass_sampler(cfg['base_model'], stale_idx, c_ids, corpus_lookup,
    #               mix_df, qrels_dict, cfg, config, mix_out)

    # # --- TRAIN: one epoch on mined negatives ---
    # output_model_dir = get_path("models") / cfg['model_name']
    # training_args = [
    #     '--output_dir', str(output_model_dir), '--model_name_or_path', cfg['base_model'],
    #     '--dataset_name', 'json', '--dataset_path', str(mix_out / "*.jsonl"),
    #     '--dataset_split', 'train', '--per_device_train_batch_size', str(cfg['batch_size']),
    #     '--train_group_size', str(cfg['train_group_size']), '--learning_rate', str(cfg['learning_rate']),
    #     '--num_train_epochs', str(cfg['num_epochs']), '--bf16', 'True', '--dtype', 'bfloat16',
    #     '--overwrite_output_dir', 'True',
    #     '--save_strategy', cfg['save_strategy'],
    #     '--save_steps', str(cfg.get('save_steps', 500)),
    #     '--save_total_limit', str(cfg['save_total_limit']),
    #     '--ignore_data_skip', 'True',
    #     '--warmup_ratio', str(cfg.get('warmup_ratio', 0.1)),
    #     '--weight_decay', str(cfg.get('weight_decay', 0.01)),
    #     '--max_grad_norm', str(cfg.get('max_grad_norm', 1.0)),
    #     '--dataloader_num_workers', str(cfg['dataloader_num_workers']),
    #     '--attn_implementation', 'eager', '--optim', 'adamw_torch_fused',
    #     '--logging_steps', str(cfg['logging_steps']),
    #     '--pooling', ctx['pooling'],
    #     '--normalize', str(ctx['normalize']),
    #     '--temperature', str(ctx['temperature']),
    # ]
    # sys.argv = ['train.py'] + training_args
    # patch_tevatron_loss(ctx['temperature'])
    # tevatron_train_main()

    # gc.collect()
    # torch.cuda.empty_cache()

    # # --- EVALUATE ---
    # for domain in config['evaluation'].get('eval_domains', []):
    #     subprocess.run([
    #         sys.executable, str(project_root / 'src/evaluation/evaluate.py'),
    #         '--model_path', str(output_model_dir),
    #         '--domain', domain,
    #     ], check=True)

    # scores = [
    #     json.load(open(get_path("results") / f"{domain}_results.json"))['metrics'].get('ndcg_cut_10', 0)
    #     for domain in config['evaluation'].get('eval_domains', [])
    # ]
    # print(f"📈 GRASS Mean NDCG@10: {sum(scores) / len(scores):.4f}", flush=True)
    # print(f"✅ GRASS complete. Model saved to: {output_model_dir}", flush=True)


if __name__ == "__main__":
    main()
