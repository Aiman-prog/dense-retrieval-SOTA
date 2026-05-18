"""Helper utility functions for Path and Context Management."""

import sys
import subprocess
import pickle
import json
import yaml
import os
import numpy as np
import faiss
import torch
from pathlib import Path
from typing import Dict, Any

def load_config(config_path: str = "config/config.yaml"):
    """
    Finds the project root and loads the config file.
    """
    # 1. Get the directory where THIS file (helpers.py) lives
    # 2. Go up two levels to reach the project root (src/utils -> project_root)
    project_root = Path(__file__).resolve().parent.parent.parent
    
    full_path = project_root / config_path
    
    if not full_path.exists():
        raise FileNotFoundError(f"❌ Config not found at {full_path}. Check your folder structure!")
        
    with open(full_path, 'r') as f:
        return yaml.safe_load(f)

def get_data_base_dir() -> Path:
    """Get base directory for all data, returning a Path object."""
    if 'DATA_BASE_DIR' in os.environ:
        return Path(os.environ['DATA_BASE_DIR'])
    
    user = os.environ.get('USER', os.environ.get('USERNAME', 'user'))
    return Path(f'/scratch/{user}/dense-retrieval-SOTA')

def get_path(key: str, model_name: str = None) -> Path:
    """
    Centralized path resolver.
    Example: get_path('processed') -> /scratch/user/.../data/processed
    """
    config = load_config()
    base = get_data_base_dir()
    p_cfg = config['paths']
    
    path_map = {
        "base": base,
        "data": base / p_cfg['data_dir'],
        "processed": base / p_cfg['processed_dir'],
        "bright": base / p_cfg['bright_cache'],
        "models": base / p_cfg['models_dir'],
        "results": base / p_cfg['results_dir'],
        "temp_ance": base / "temp_ance_workdir",
        "temp_grass": base / "temp_grass_workdir",
        "temp_grass_async": base / "temp_grass_async_workdir",
        "temp_grass_seq": base / "temp_grass_seq_workdir",
    }
    
    if model_name:
        return path_map["models"] / model_name
    return path_map.get(key)

def get_training_context(training_type: str = "inbatch") -> Dict[str, Any]:
    config = load_config()
    recipe = config['training'][training_type]
    model_name = recipe.get('base_model') or config['model']['base_model']
    
    # Force absolute path resolution
    cache_base = get_path("bright").resolve() / "hub"
    repo_id = model_name.replace("/", "--")
    snapshot_dir = cache_base / f"models--{repo_id}" / "snapshots"
    
    final_base_model = model_name # Default fallback

    if snapshot_dir.exists():
        # Filter out hidden files and get actual directories
        snapshots = [d for d in snapshot_dir.iterdir() if d.is_dir()]
        if snapshots:
            # Sort to get the most recent or consistent one
            chosen_snapshot = sorted(snapshots)[-1]
            # Check if config.json is there (exists() or is_symlink() for HF cache)
            cfg = chosen_snapshot / "config.json"
            if cfg.exists() or cfg.is_symlink():
                final_base_model = str(chosen_snapshot)

    return {
        "args": recipe,
        "base_model": final_base_model,
        "max_q": config['model']['query_max_len'],
        "max_p": config['model']['passage_max_len'],
        "pooling": config['model'].get('pooling', 'cls'),
        "normalize": config['model'].get('normalize', False),
        "temperature": config['model'].get('temperature', 0.02),
        "processed_dir": get_path("processed"),
        "output_dir": get_path("models", recipe['model_name']),
        "cache_dir": str(get_path("bright").resolve())
    }


def encode_to_pickle(model_path, input_file, output_pkl, is_query, ctx, config):
    """Run Tevatron encode subprocess and save embeddings to a pickle file."""
    cmd = [
        sys.executable, '-m', 'tevatron.retriever.driver.encode',
        '--output_dir', str(output_pkl.parent),
        '--model_name_or_path', model_path,
        '--bf16', 'True', '--fp16', 'False',
        '--per_device_eval_batch_size', str(ctx['args']['per_device_eval_batch_size']),
        '--dataset_name', 'json', '--dataset_path', str(input_file),
        '--encode_output_path', str(output_pkl),
        '--attn_implementation', 'eager',
        '--dataloader_num_workers', str(ctx['args']['dataloader_num_workers']),
        '--pooling', ctx['pooling'],
        '--normalize', str(ctx['normalize']),
    ]
    if is_query:
        q_len = str(config['model'].get('query_max_len', 128))
        try:
            subprocess.run(cmd + ['--encode_is_query', '--query_max_len', q_len], check=True)
        except subprocess.CalledProcessError:
            subprocess.run(cmd + ['--encode_is_qry', '--q_max_len', q_len], check=True)
    else:
        subprocess.run(cmd + ['--passage_max_len', str(config['model'].get('passage_max_len', 512))], check=True)


def build_faiss_index(corpus_pkl_path):
    """Load corpus pickle and build a FAISS IndexFlatIP. Returns (index, embeddings, ids)."""
    with open(corpus_pkl_path, 'rb') as f:
        c_data = pickle.load(f)
    embeddings = c_data[0].astype(np.float32)
    ids = [str(x) for x in c_data[1]]
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    return index, embeddings, ids


def patch_tevatron_loss(temperature):
    """Monkey-patch Tevatron's GradCache trainer to use temperature-scaled contrastive loss."""
    from models.temperature_scaled_loss import (
        TemperatureScaledContrastiveLoss,
        DistributedTemperatureScaledContrastiveLoss,
    )
    import tevatron.retriever.gc_trainer as gc_module

    class SimpleContrastiveLossPatched(TemperatureScaledContrastiveLoss):
        def __init__(self):
            super().__init__(temperature=temperature)

    class DistributedContrastiveLossPatched(DistributedTemperatureScaledContrastiveLoss):
        def __init__(self, n_target=0, scale_loss=True):
            super().__init__(temperature=temperature, n_target=n_target, scale_loss=scale_loss)

    gc_module.SimpleContrastiveLoss = SimpleContrastiveLossPatched
    gc_module.DistributedContrastiveLoss = DistributedContrastiveLossPatched


def set_seed(seed: int):
    import random as _random
    import numpy as _np
    import torch as _torch
    _random.seed(seed)
    _np.random.seed(seed)
    _torch.manual_seed(seed)
    if _torch.cuda.is_available():
        _torch.cuda.manual_seed_all(seed)


def encode_batch(model, tokenizer, texts, device, max_len, batch_size):
    """
    Encode a list of texts in mini-batches, returning L2-normalised CLS embeddings.
    Runs under no_grad with bf16 autocast (disabled on CPU).
    Used by MC-dropout mining (grass_sampler in run_grass_mcd).
    """
    all_embs = []
    for i in range(0, len(texts), batch_size):
        batch  = texts[i:i + batch_size]
        inputs = tokenizer(batch, padding=True, truncation=True,
                           max_length=max_len, return_tensors='pt').to(device)
        # [S3] autocast — enabled=False on CPU so safe to run locally.
        with torch.no_grad(), torch.autocast(device_type='cuda', dtype=torch.bfloat16,
                                              enabled=device.type == 'cuda'):
            out = model(**inputs)
        embs = out.last_hidden_state[:, 0, :]
        embs = torch.nn.functional.normalize(embs, dim=-1)
        all_embs.append(embs.cpu().float().numpy())
    return np.concatenate(all_embs, axis=0)


def count_jsonl_examples(pattern: str) -> int:
    """Count total lines across all JSONL files matching a glob pattern."""
    import glob as glob_module
    total = 0
    for path in glob_module.glob(pattern):
        with open(path) as f:
            total += sum(1 for line in f if line.strip())
    return total


# ── Shared IPC / IO utilities (used by ANCE + GRASS async) ──────────────────

def is_valid_checkpoint(ckpt_path: str) -> bool:
    """Checkpoint is fully written once optimizer.pt exists (trainer writes it last)."""
    return (Path(ckpt_path) / "optimizer.pt").exists()


def get_latest_marker_no(directory: Path, prefix: str = "ready_") -> int:
    """Return the highest N from files named {prefix}{N} in directory, or 0 if none."""
    nos = [int(f.name[len(prefix):]) for f in directory.glob(f"{prefix}*")
           if f.name[len(prefix):].isdigit()]
    return max(nos) if nos else 0


def _load_qrels(qrels_file) -> dict:
    """Load TREC qrels file. Returns {qid: set(docids)}."""
    import pandas as pd
    data = []
    with open(qrels_file) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 4:
                data.append({'qid': parts[0], 'did': parts[2]})
    return pd.DataFrame(data).groupby('qid')['did'].apply(set).to_dict() if data else {}


def _load_corpus_lookup(corpus_file) -> dict:
    """Load corpus JSONL. Returns {docid: text}."""
    lookup = {}
    with open(corpus_file) as f:
        for line in f:
            d = json.loads(line)
            lookup[d['docid']] = d['text']
    return lookup


def _shortlist_batch(batch_ids, indices, q_embs_det, qrels_dict, c_ids,
                     c_id_to_idx, stale_embs, corpus_lookup, P, L):
    """
    CPU shortlisting: filter true positives from P ANN candidates per query,
    then keep top-L by cheap stale_embs @ q_cur dot product.
    Safe for background thread execution (numpy BLAS releases the GIL).
    Returns (batch_query_shortlist, shortlist_ids, shortlist_texts, shortlist_to_idx, n_filtered).
    """
    batch_query_cands = {}
    for i, qid in enumerate(batch_ids):
        cands = [c_ids[j] for j in indices[i]
                 if j >= 0 and c_ids[j] not in qrels_dict.get(qid, set())]
        batch_query_cands[qid] = cands

    batch_query_shortlist  = {}
    shortlist_cand_ids_set = set()
    for i, qid in enumerate(batch_ids):
        cands = batch_query_cands[qid]
        if not cands:
            batch_query_shortlist[qid] = []
            continue
        stale_idxs = [c_id_to_idx[d] for d in cands]
        scores     = stale_embs[stale_idxs] @ q_embs_det[i]
        top_l      = np.argsort(scores)[::-1][:L]
        shortlist  = [cands[k] for k in top_l]
        batch_query_shortlist[qid] = shortlist
        shortlist_cand_ids_set.update(shortlist)

    shortlist_ids    = list(shortlist_cand_ids_set)
    shortlist_texts  = [corpus_lookup.get(did, "") for did in shortlist_ids]
    shortlist_to_idx = {did: idx for idx, did in enumerate(shortlist_ids)}
    n_filtered = sum(len(v) for v in batch_query_cands.values())
    return batch_query_shortlist, shortlist_ids, shortlist_texts, shortlist_to_idx, n_filtered


def evaluate_bright(ctx, config, model_path, temp_workdir_key=None):
    """Multi-domain BRIGHT evaluation (or single-set if eval_corpus_file set in ctx.args)."""
    import pickle
    import pandas as pd
    from evaluation.trec_eval_wrapper import TrecEvalWrapper

    args = ctx['args']
    if temp_workdir_key is None:
        temp_workdir_key = args.get('temp_workdir', 'temp_grass_async')
    temp_dir = get_path(temp_workdir_key)

    if args.get('eval_corpus_file'):
        p         = get_path("processed")
        d_corpus  = p / args['eval_corpus_file']
        d_queries = p / args['eval_queries_file']
        d_qrels   = p / args['eval_qrels_file']
        if not all(x.exists() for x in [d_corpus, d_queries, d_qrels]):
            print("[Eval] Skipping: eval files not found", flush=True)
            return
        eval_dir = temp_dir / "final_eval"
        eval_dir.mkdir(parents=True, exist_ok=True)
        encode_to_pickle(str(model_path), d_corpus,  eval_dir / "c.pkl", False, ctx, config)
        encode_to_pickle(str(model_path), d_queries, eval_dir / "q.pkl", True,  ctx, config)
        with open(eval_dir / "c.pkl", 'rb') as f: dc = pickle.load(f)
        with open(eval_dir / "q.pkl", 'rb') as f: dq = pickle.load(f)
        idx_e = faiss.IndexFlatIP(dc[0].shape[1])
        idx_e.add(dc[0].astype(np.float32))
        s_e, i_e = idx_e.search(dq[0].astype(np.float32), args.get('eval_top_k', 1000))
        results = {
            str(dq[1][j]): {str(dc[1][i_e[j][k]]): float(s_e[j][k])
                             for k in range(len(i_e[j])) if i_e[j][k] >= 0}
            for j in range(len(dq[1]))
        }
        eval_qrels_data = []
        with open(d_qrels) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 4:
                    eval_qrels_data.append({'query_id': parts[0], 'doc_id': parts[2],
                                            'relevance': parts[3]})
        metric = args.get('eval_metric', 'ndcg_cut_10')
        evaluator = TrecEvalWrapper(pd.DataFrame(eval_qrels_data))
        metrics = evaluator.evaluate(results, {metric})
        print(f"\n📈 Eval — {metric}={metrics.get(metric, 0):.4f}", flush=True)
    else:
        eval_summary = []
        for domain in config['evaluation'].get('eval_domains', []):
            d_corpus  = get_path("processed") / f"{domain}_corpus.jsonl"
            d_queries = get_path("processed") / f"{domain}_queries.jsonl"
            d_qrels   = get_path("processed") / f"{domain}_qrels.txt"
            if not all(p.exists() for p in [d_corpus, d_queries, d_qrels]):
                print(f"[Eval] Skipping {domain}: files not found", flush=True)
                continue
            eval_dir = temp_dir / "final_eval" / domain
            eval_dir.mkdir(parents=True, exist_ok=True)
            encode_to_pickle(str(model_path), d_corpus,  eval_dir / "c.pkl", False, ctx, config)
            encode_to_pickle(str(model_path), d_queries, eval_dir / "q.pkl", True,  ctx, config)
            with open(eval_dir / "c.pkl", 'rb') as f: dc = pickle.load(f)
            with open(eval_dir / "q.pkl", 'rb') as f: dq = pickle.load(f)
            idx_e = faiss.IndexFlatIP(dc[0].shape[1])
            idx_e.add(dc[0].astype(np.float32))
            eval_top_k = args.get('eval_top_k', 10)
            s_e, i_e = idx_e.search(dq[0].astype(np.float32), eval_top_k)
            results = {
                str(dq[1][j]): {str(dc[1][i_e[j][k]]): float(s_e[j][k])
                                 for k in range(len(i_e[j])) if i_e[j][k] >= 0}
                for j in range(len(dq[1]))
            }
            eval_qrels_data = []
            with open(d_qrels) as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 4:
                        eval_qrels_data.append({'query_id': parts[0], 'doc_id': parts[2],
                                                'relevance': parts[3]})
            evaluator = TrecEvalWrapper(pd.DataFrame(eval_qrels_data))
            metrics = evaluator.evaluate(results, {'recip_rank', 'ndcg_cut_10'})
            eval_summary.append({'domain': domain, 'ndcg10': metrics.get('ndcg_cut_10', 0)})
            print(f"[Eval] {domain}: NDCG@10={metrics.get('ndcg_cut_10', 0):.4f}", flush=True)
        if eval_summary:
            mean_ndcg = pd.DataFrame(eval_summary)['ndcg10'].mean()
            print(f"\n📈 Final Mean NDCG@10: {mean_ndcg:.4f}", flush=True)