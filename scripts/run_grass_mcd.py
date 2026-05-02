import gc
import json
import sys
import time
import numpy as np
import torch
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from transformers import AutoTokenizer, AutoModel
from tevatron.retriever.driver.train import main as tevatron_train_main
from tevatron.retriever.modeling import DenseModel

project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root / 'src'))

from utils.helpers import get_path, encode_batch, patch_tevatron_loss

# Tevatron Bug Patch
if not hasattr(DenseModel, "_keys_to_ignore_on_save"):
    setattr(DenseModel, "_keys_to_ignore_on_save", None)


def _shortlist_batch(batch_ids, indices, q_embs_det, qrels_dict, c_ids,
                     c_id_to_idx, stale_embs, corpus_lookup, P, L):
    """
    [S7] CPU shortlisting — designed to run in a background thread while the GPU
    executes the T MC query encodes. All inputs are read-only (numpy arrays and
    Python dicts/lists), so concurrent access is safe. numpy BLAS releases the
    GIL during the dot-product calls, enabling genuine CPU/GPU parallelism.

    Returns (batch_query_shortlist, shortlist_ids, shortlist_texts, shortlist_to_idx, n_filtered).
    """
    # Filter true positives out of each query's P ANN candidates
    batch_query_cands = {}
    for i, qid in enumerate(batch_ids):
        cands = [c_ids[j] for j in indices[i]
                 if j >= 0 and c_ids[j] not in qrels_dict.get(qid, set())]
        batch_query_cands[qid] = cands

    # Shortlist to top-L per query using cheap stale-embedding dot products
    batch_query_shortlist  = {}
    shortlist_cand_ids_set = set()
    for i, qid in enumerate(batch_ids):
        cands = batch_query_cands[qid]
        if not cands:
            batch_query_shortlist[qid] = []
            continue
        stale_idxs = [c_id_to_idx[d] for d in cands]
        scores     = stale_embs[stale_idxs] @ q_embs_det[i]  # numpy BLAS, releases GIL
        top_l      = np.argsort(scores)[::-1][:L]
        shortlist  = [cands[k] for k in top_l]
        batch_query_shortlist[qid] = shortlist
        shortlist_cand_ids_set.update(shortlist)

    shortlist_ids    = list(shortlist_cand_ids_set)
    shortlist_texts  = [corpus_lookup.get(did, "") for did in shortlist_ids]
    shortlist_to_idx = {did: idx for idx, did in enumerate(shortlist_ids)}
    n_filtered = sum(len(v) for v in batch_query_cands.values())
    return batch_query_shortlist, shortlist_ids, shortlist_texts, shortlist_to_idx, n_filtered


def grass_sampler(model_path, stale_idx, stale_embs, c_id_to_idx, c_ids, corpus_lookup, mix_df,
                  qrels_dict, cfg, config, out_dir):
    """
    GrassSampler (Algorithm 2): mines hard negatives for all training queries using
    a stale ANN index and MC-dropout uncertainty estimation.

    Hyperparameters (from cfg):
      P          — pool size: candidates retrieved per query from the stale ANN index
      L          — shortlist size: top-L candidates by cheap score before MC-dropout (L <= P)
      m          — hard negatives selected per query (= train_group_size - 1)
      T          — MC-dropout forward passes for uncertainty estimation
      lambda_val — trade-off weight: higher lambda promotes more uncertain negatives

    Writes updated mixture JSONL files to out_dir with negative_passages replaced.
    """
    P                = cfg['P']
    L                = cfg['L']
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
    mc_dropout_p = cfg.get('mc_dropout_p', 0.1)
    if mc_dropout_p != 0.1:
        n_dropout = sum(1 for mod in model.modules() if isinstance(mod, torch.nn.Dropout))
        for module in model.modules():
            if isinstance(module, torch.nn.Dropout):
                module.p = mc_dropout_p
        print(f"  MC-dropout p set to {mc_dropout_p} ({n_dropout} Dropout layers)", flush=True)
    print(f"  Loaded model for GrassSampler (T={T}, P={P}, m={m}, lambda={lambda_val})", flush=True)

    query_ids   = mix_df['query_id'].astype(str).tolist()
    query_texts = mix_df['query'].tolist()
    n_queries   = len(query_ids)
    n_batches   = (n_queries + query_batch_size - 1) // query_batch_size
    print(f"  Processing {n_queries} queries in {n_batches} batches (batch_size={query_batch_size})...", flush=True)

    mined_negs   = {}
    t_loop_start = time.time()

    # [S6] Open mining log for per-query uncertainty analysis after training
    log_path = out_dir / "mining_log.jsonl"
    log_path.parent.mkdir(exist_ok=True, parents=True)
    mining_log_f = open(log_path, 'w')

    # [S7] One persistent background thread — CPU shortlisting runs here while GPU
    # does T MC query encodes.
    cpu_exec = ThreadPoolExecutor(max_workers=1)

    for b, batch_start in enumerate(range(0, n_queries, query_batch_size)):
        batch_ids   = query_ids[batch_start:batch_start + query_batch_size]
        batch_texts = query_texts[batch_start:batch_start + query_batch_size]

        # Deterministic query encoding for ANN retrieval and shortlisting
        model.eval()
        q_embs_det = encode_batch(model, tokenizer, batch_texts, device, q_max_len, mc_batch_size)
        model.train()

        _, indices = stale_idx.search(q_embs_det, P)

        # [S7] Launch CPU filter+shortlist in background immediately after FAISS
        shortlist_fut = cpu_exec.submit(
            _shortlist_batch, batch_ids, indices, q_embs_det, qrels_dict,
            c_ids, c_id_to_idx, stale_embs, corpus_lookup, P, L
        )

        # [S1] Vectorize T MC query encodes into one forward pass
        q_embs_flat  = encode_batch(model, tokenizer, batch_texts * T, device, q_max_len, mc_batch_size)
        q_embs_stack = q_embs_flat.reshape(T, len(batch_texts), -1)  # (T, B, dim)

        # [S7] Join shortlisting — should be done while GPU was encoding
        batch_query_shortlist, shortlist_ids, shortlist_texts, shortlist_to_idx, n_filtered = shortlist_fut.result()

        # [S2] Vectorize T MC candidate encodes — guard against empty shortlist
        if shortlist_texts:
            c_embs_flat  = encode_batch(model, tokenizer, shortlist_texts * T, device, p_max_len, mc_batch_size)
            c_embs_stack = c_embs_flat.reshape(T, len(shortlist_texts), -1)  # (T, N_shortlist, dim)
        else:
            c_embs_stack = None

        batch_sigmas, batch_g = [], []
        for i, qid in enumerate(batch_ids):
            cands = batch_query_shortlist[qid]
            # [S4] guard handles c_embs_stack=None (empty shortlist batch)
            if not cands or c_embs_stack is None:
                continue
            cand_idxs = [shortlist_to_idx[d] for d in cands]

            # [S4] numpy einsum — faster than per-query torch bmm for tiny (T, 1, ~50) matrices
            q_i  = q_embs_stack[:, i, :]
            c_i  = c_embs_stack[:, cand_idxs, :]
            sims = np.einsum('td,tnd->tn', q_i, c_i)  # (T, N_cands)

            s_hat = sims.mean(axis=0)
            sigma = sims.std(axis=0)
            g     = s_hat + lambda_val * sigma

            top_m = np.argsort(g)[::-1][:m]
            mined_negs[qid] = [cands[k] for k in top_m]
            batch_sigmas.append(sigma.mean())
            batch_g.append(g.mean())

            # [S6] Log per-query mining stats
            rank_by_shat = int(np.argsort(np.argsort(-s_hat))[top_m[0]])
            mining_log_f.write(json.dumps({
                "query_id":             qid,
                "neg_docid":            cands[top_m[0]],
                "s_hat_selected":       float(s_hat[top_m[0]]),
                "sigma_selected":       float(sigma[top_m[0]]),
                "g_selected":           float(g[top_m[0]]),
                "rank_by_shat":         rank_by_shat,
                "sigma_mean_shortlist": float(sigma.mean()),
            }, ensure_ascii=False) + '\n')

        if b < 3 or (b + 1) % 100 == 0:
            elapsed   = time.time() - t_loop_start
            secs_per  = elapsed / (b + 1)
            remaining = secs_per * (n_batches - b - 1)
            eta       = f"{int(remaining // 3600)}h {int((remaining % 3600) // 60)}m"
            n_raw = len(batch_ids) * P
            print(f"  Batch {b+1}/{n_batches} | ETA {eta} | "
                  f"P→L filter: {n_filtered}/{n_raw} → shortlist {len(shortlist_ids)} unique | "
                  f"sigma mean: {np.mean(batch_sigmas):.5f} | "
                  f"g mean: {np.mean(batch_g):.4f}", flush=True)

    # [S6] Flush and close mining log
    mining_log_f.close()
    print(f"  Mining log written to {log_path}", flush=True)

    # [S7] Shut down background thread pool
    cpu_exec.shutdown(wait=False)

    del model
    gc.collect()
    torch.cuda.empty_cache()

    # Write updated mixture files
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


def run_mcd_pipeline(stale_idx, stale_embs, c_id_to_idx, c_ids, corpus_lookup, mix_df,
                     qrels_dict, cfg, config, ctx, workdir):
    """Mine hard negatives with MC-dropout then train with Tevatron. Returns output_model_dir."""
    mix_out = workdir / "grass_train"
    print("🔍 Running GrassSampler (MC-dropout)...", flush=True)
    grass_sampler(cfg['base_model'], stale_idx, stale_embs, c_id_to_idx, c_ids, corpus_lookup,
                  mix_df, qrels_dict, cfg, config, mix_out)

    output_model_dir = get_path("models") / (cfg['model_name'] + '_mcdp')
    training_args = [
        '--output_dir',                    str(output_model_dir),
        '--model_name_or_path',            cfg['base_model'],
        '--dataset_name',                  'json',
        '--dataset_path',                  str(mix_out / "*.jsonl"),
        '--dataset_split',                 'train',
        '--per_device_train_batch_size',   str(cfg['batch_size']),
        '--train_group_size',              str(cfg['train_group_size']),
        '--learning_rate',                 str(cfg['learning_rate']),
        '--num_train_epochs',              str(cfg['num_epochs']),
        '--bf16',                          'True',
        '--dtype',                         'bfloat16',
        '--overwrite_output_dir',          'True',
        '--save_strategy',                 cfg['save_strategy'],
        '--save_steps',                    str(cfg.get('save_steps', 500)),
        '--save_total_limit',              str(cfg['save_total_limit']),
        '--ignore_data_skip',              'True',
        '--warmup_ratio',                  str(cfg.get('warmup_ratio', 0.1)),
        '--weight_decay',                  str(cfg.get('weight_decay', 0.01)),
        '--max_grad_norm',                 str(cfg.get('max_grad_norm', 1.0)),
        '--dataloader_num_workers',        str(cfg['dataloader_num_workers']),
        '--attn_implementation',           'eager',
        '--optim',                         'adamw_torch_fused',
        '--logging_steps',                 str(cfg['logging_steps']),
        '--pooling',                       ctx['pooling'],
        '--normalize',                     str(ctx['normalize']),
        '--temperature',                   str(ctx['temperature']),
        '--seed',                          str(config.get('seed', 42)),
    ]
    sys.argv = ['train.py'] + training_args
    patch_tevatron_loss(ctx['temperature'])
    tevatron_train_main()
    return output_model_dir
