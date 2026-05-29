import gc
import json
import sys
import time
import numpy as np
import torch
from pathlib import Path
from transformers import AutoTokenizer, AutoModel
from transformers.trainer_utils import get_last_checkpoint
from tevatron.retriever.driver.train import main as tevatron_train_main
from tevatron.retriever.modeling import DenseModel

project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root / 'src'))

from utils.helpers import (
    get_path, encode_batch, patch_tevatron_loss,
    _pool_and_fresh_rerank,
)
from utils.grass_candidate_memory import CandidateMemory

# Tevatron Bug Patch
if not hasattr(DenseModel, "_keys_to_ignore_on_save"):
    setattr(DenseModel, "_keys_to_ignore_on_save", None)


def grass_sampler(model_path, stale_idx, stale_embs, c_id_to_idx, c_ids, corpus_lookup, mix_df,
                  qrels_dict, cfg, config, out_dir, base_jsonl_dir=None,
                  current_round=1, memory=None):
    """
    GrassSampler (Algorithm 2): mines hard negatives using a stale ANN index
    plus optional active candidate memory, fresh current-model rerank for the
    top-L shortlist, and MC-dropout for uncertainty.

    Hyperparameters (from cfg):
      P          - candidates retrieved per query from stale ANN index
      L          - top-L shortlist after fresh current-model rerank
      m          - hard negatives per query (= train_group_size - 1)
      T          - MC-dropout forward passes
      lambda_val - g = s_hat + lambda * sigma

    Architecture:
      stale FAISS top-P + active memory -> dedup + filter positives ->
      deterministic current-model fresh rerank -> top-L ->
      T-pass MC query encode -> T-pass MC candidate encode -> s_hat/sigma/g ->
      top-m -> update memory.

    Writes updated mixture JSONL files to out_dir with negative_passages replaced.
    """
    P                = cfg['P']
    L                = cfg['L']
    m                = cfg['m']
    T                = cfg['T']
    lambda_val       = cfg['lambda_val']
    mc_batch_size    = cfg.get('mc_batch_size', 256)
    query_batch_size = cfg.get('query_batch_size', 64)
    max_pool_per_query = cfg.get('max_pool_per_query', P)
    q_max_len        = config['model']['query_max_len']
    p_max_len        = config['model']['passage_max_len']

    cache_cfg            = cfg.get('candidate_cache', {})
    cache_enabled        = cache_cfg.get('enabled', False)
    top_g_to_store       = cache_cfg.get('top_g_to_store', 8)
    top_sigma_to_store   = cache_cfg.get('top_sigma_to_store', 8)
    use_memory           = cache_enabled and memory is not None

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
    print(f"  Loaded model for GrassSampler (T={T}, P={P}, m={m}, lambda={lambda_val}, "
          f"memory={'on' if use_memory else 'off'})",
          flush=True)

    query_ids   = mix_df['query_id'].astype(str).tolist()
    query_texts = mix_df['query'].tolist()
    n_queries   = len(query_ids)
    n_batches   = (n_queries + query_batch_size - 1) // query_batch_size
    print(f"  Processing {n_queries} queries in {n_batches} batches (batch_size={query_batch_size})...", flush=True)

    mined_negs   = {}
    t_loop_start = time.time()

    log_path = out_dir / "mining_log.jsonl"
    log_path.parent.mkdir(exist_ok=True, parents=True)
    mining_log_f = open(log_path, 'w')

    for b, batch_start in enumerate(range(0, n_queries, query_batch_size)):
        batch_ids   = query_ids[batch_start:batch_start + query_batch_size]
        batch_texts = query_texts[batch_start:batch_start + query_batch_size]

        # Deterministic query encoding for ANN retrieval and fresh-rerank
        model.eval()
        q_embs_det = encode_batch(model, tokenizer, batch_texts, device, q_max_len, mc_batch_size)
        model.train()

        _, indices = stale_idx.search(q_embs_det, P)

        memory_per_query   = {}
        memory_expired_map = {}
        if use_memory:
            for qid in batch_ids:
                ids, expired = memory.get(qid, current_round)
                memory_per_query[qid]   = ids
                memory_expired_map[qid] = expired

        batch_query_shortlist, source_map, pool_stats = _pool_and_fresh_rerank(
            model, tokenizer, batch_ids, q_embs_det,
            indices, memory_per_query, memory_expired_map,
            qrels_dict, c_ids, corpus_lookup,
            p_max_len, mc_batch_size, device,
            L, max_pool_per_query,
        )

        # T-pass MC query encode (dropout active via model.train())
        q_embs_flat  = encode_batch(model, tokenizer, batch_texts * T, device, q_max_len, mc_batch_size)
        q_embs_stack = q_embs_flat.reshape(T, len(batch_texts), -1)  # (T, B, dim)

        # T-pass MC candidate encode over the deduped top-L docs across the batch
        shortlist_cand_ids_set = set()
        for qid in batch_ids:
            shortlist_cand_ids_set.update(batch_query_shortlist.get(qid, []))
        shortlist_ids    = list(shortlist_cand_ids_set)
        shortlist_texts  = [corpus_lookup.get(d, "") for d in shortlist_ids]
        shortlist_to_idx = {d: i for i, d in enumerate(shortlist_ids)}

        if shortlist_texts:
            c_embs_flat  = encode_batch(model, tokenizer, shortlist_texts * T, device, p_max_len, mc_batch_size)
            c_embs_stack = c_embs_flat.reshape(T, len(shortlist_texts), -1)
        else:
            c_embs_stack = None
        n_filtered = sum(stats['positives_filtered'] for stats in pool_stats.values())

        batch_sigmas, batch_g = [], []
        for i, qid in enumerate(batch_ids):
            cands = batch_query_shortlist[qid]
            if not cands or c_embs_stack is None:
                continue
            cand_idxs = [shortlist_to_idx[d] for d in cands]

            q_i  = q_embs_stack[:, i, :]
            c_i  = c_embs_stack[:, cand_idxs, :]
            sims = np.einsum('td,tnd->tn', q_i, c_i)  # (T, N_cands)

            s_hat = sims.mean(axis=0)
            sigma = sims.std(axis=0)
            g     = s_hat + lambda_val * sigma

            top_m_idxs = np.argsort(g)[::-1][:m]
            top_m_docs = [cands[k] for k in top_m_idxs]
            mined_negs[qid] = top_m_docs
            batch_sigmas.append(sigma.mean())
            batch_g.append(g.mean())

            selected_docid = top_m_docs[0]
            rank_by_shat   = int(np.argsort(np.argsort(-s_hat))[top_m_idxs[0]])

            # Update active memory with selected + top-g + top-sigma from this round
            if use_memory:
                top_g_idxs     = np.argsort(g)[::-1][:top_g_to_store]
                top_sigma_idxs = np.argsort(sigma)[::-1][:top_sigma_to_store]
                memory.update(
                    qid, current_round,
                    selected_negs    = top_m_docs,
                    top_g_docids     = [cands[k] for k in top_g_idxs],
                    top_sigma_docids = [cands[k] for k in top_sigma_idxs],
                    top_g_value      = float(g[top_m_idxs[0]]),
                )

            log_record = {
                "query_id":             qid,
                "neg_docid":            selected_docid,
                "s_hat_selected":       float(s_hat[top_m_idxs[0]]),
                "sigma_selected":       float(sigma[top_m_idxs[0]]),
                "g_selected":           float(g[top_m_idxs[0]]),
                "rank_by_shat":         rank_by_shat,
                "sigma_mean_shortlist": float(sigma.mean()),
            }
            stats = pool_stats.get(qid, {})
            log_record.update({
                "retrieved_count":          stats.get('retrieved', 0),
                "memory_count":             stats.get('memory_count', 0),
                "candidate_pool_count":     stats.get('pool_count', 0),
                "positives_filtered_count": stats.get('positives_filtered', 0),
                "L":                        L,
                "m":                        m,
                "neg_docids":               top_m_docs,
                "selected_cheap_rank_zero_based": int(top_m_idxs[0]),
                "selected_source":          source_map.get(qid, {}).get(selected_docid, 'faiss'),
                "cache_used":               bool(stats.get('cache_used', False)),
                "faiss_used":               bool(stats.get('faiss_used', False)),
                "memory_expired":           bool(stats.get('memory_expired', False)),
            })
            mining_log_f.write(json.dumps(log_record, ensure_ascii=False) + '\n')

        if b < 3 or (b + 1) % 100 == 0:
            elapsed   = time.time() - t_loop_start
            secs_per  = elapsed / (b + 1)
            remaining = secs_per * (n_batches - b - 1)
            eta       = f"{int(remaining // 3600)}h {int((remaining % 3600) // 60)}m"
            n_raw = len(batch_ids) * P
            print(f"  Batch {b+1}/{n_batches} | ETA {eta} | "
                  f"P->L filter: {n_filtered}/{n_raw} -> shortlist {len(shortlist_ids)} unique | "
                  f"sigma mean: {np.mean(batch_sigmas) if batch_sigmas else 0:.5f} | "
                  f"g mean: {np.mean(batch_g) if batch_g else 0:.4f}", flush=True)

    mining_log_f.close()
    print(f"  Mining log written to {log_path}", flush=True)

    del model
    gc.collect()
    torch.cuda.empty_cache()

    # Write updated mixture files (base = previous epoch's output, or original mixture)
    out_dir.mkdir(exist_ok=True, parents=True)
    base_dir = base_jsonl_dir if base_jsonl_dir is not None else (get_path("processed") / "training_mixture")
    for f_path in base_dir.glob("*.jsonl"):
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


def _detect_completed_epochs(output_dir):
    """Returns the number of fully-completed training epochs in `output_dir`,
    based on the latest HF checkpoint's trainer_state.json. Returns 0 if the
    directory is missing, empty, has no valid checkpoint, or the state file is
    unreadable.

    Used to support resume after preempted/failed cluster runs: a job that died
    after epoch N will resume mining + training at epoch N+1 instead of wiping
    prior progress.
    """
    if not output_dir.exists():
        return 0
    last_ckpt = get_last_checkpoint(str(output_dir))
    if last_ckpt is None:
        return 0
    state_path = Path(last_ckpt) / "trainer_state.json"
    if not state_path.exists():
        return 0
    try:
        with open(state_path) as f:
            state = json.load(f)
        return int(state.get('epoch', 0))
    except (json.JSONDecodeError, ValueError, OSError, KeyError):
        return 0


def train_one_epoch_mcd(model_path, jsonl_dir, output_dir, cfg, ctx, config,
                        cumulative_epochs):
    """Train one more epoch into a shared output_dir.

    Tevatron's main() auto-detects the latest checkpoint in `output_dir` via
    `get_last_checkpoint` and calls `trainer.train(resume_from_checkpoint=True)`,
    which restores model + optimizer + LR scheduler + RNG + trainer_state. So
    AdamW moments and scheduler step counter carry across epochs.

    `cumulative_epochs` is the *total* number of epochs the trainer should
    believe it is training for (1 on first call, 2 on second, etc.). HF Trainer
    computes `max_steps = cumulative_epochs * steps_per_epoch` from the new
    dataset's size, so the scheduler horizon grows each call. With
    `--ignore_data_skip True`, swapping the mined JSONL path between calls is
    safe (no dataloader fast-forward).

    Caveat: the LR scheduler horizon is recomputed per call, so the decay curve
    is segmented (each segment treats the *new* max_steps as the horizon).
    Optimizer state continuity (the main goal) is preserved.

    `--save_strategy epoch` is forced here so the end-of-epoch checkpoint lands
    at the exact step boundary; that is the checkpoint the next call resumes
    from.
    """
    training_args = [
        '--output_dir',                    str(output_dir),
        '--model_name_or_path',            str(model_path),
        '--dataset_name',                  'json',
        '--dataset_path',                  str(jsonl_dir / "*.jsonl"),
        '--dataset_split',                 'train',
        '--per_device_train_batch_size',   str(cfg['batch_size']),
        '--train_group_size',              str(cfg['train_group_size']),
        '--learning_rate',                 str(cfg['learning_rate']),
        '--num_train_epochs',              str(cumulative_epochs),
        '--bf16',                          'True',
        '--dtype',                         'bfloat16',
        '--overwrite_output_dir',          'True',
        '--save_strategy',                 'epoch',
        '--save_total_limit',              str(cfg.get('save_total_limit', 1)),
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


def run_mcd_pipeline(stale_idx, stale_embs, c_id_to_idx, c_ids, corpus_lookup, mix_df,
                     qrels_dict, cfg, config, ctx, workdir):
    """Repeated MC-dropout mining: for each of `num_epochs`, re-mine negatives
    with the *current* checkpoint via grass_sampler (FAISS top-P + optional
    CandidateMemory -> current-model fresh rerank -> top-L -> MC-dropout
    uncertainty -> top-m), then train 1 Tevatron epoch starting from that
    checkpoint. The chained mining JSONL (`base_jsonl_dir=prev`) preserves
    previous negatives for any query not re-mined.

    Training continuity: shared output_dir + Tevatron's auto-detected
    resume_from_checkpoint preserves optimizer + LR scheduler step counter
    across epochs. The LR scheduler horizon is recomputed per call, so the
    decay curve is segmented (acceptable trade-off; see train_one_epoch_mcd).

    Resume-after-preemption: if `output_dir` contains a checkpoint from a prior
    job, this pipeline detects the completed epoch count via
    `_detect_completed_epochs` and skips already-trained epochs. Never wipes
    prior state.

    Returns the final epoch's model directory.
    """
    num_epochs = cfg.get('num_epochs', 3)
    base_name  = cfg['model_name']

    cache_cfg     = cfg.get('candidate_cache', {})
    cache_enabled = cache_cfg.get('enabled', False)
    memory_path   = workdir / f"candidate_memory_{base_name}.pkl"
    memory = None
    if cache_enabled:
        memory = CandidateMemory.load(
            memory_path,
            max_per_query      = cache_cfg.get('max_candidates_per_query', 64),
            ttl_rounds         = cache_cfg.get('ttl_rounds', 2),
            top_g_to_store     = cache_cfg.get('top_g_to_store', 8),
            top_sigma_to_store = cache_cfg.get('top_sigma_to_store', 8),
        )
        print(f"  Active memory loaded: {len(memory)} queries (ttl={memory.ttl_rounds})", flush=True)

    output_dir = get_path("models") / f"{base_name}_mcdp"
    completed_epochs = _detect_completed_epochs(output_dir)

    if completed_epochs >= num_epochs:
        print(f"  All {num_epochs} epochs already completed in {output_dir}. Nothing to do.", flush=True)
        return output_dir

    start_epoch = completed_epochs + 1
    output_dir.mkdir(parents=True, exist_ok=True)

    if completed_epochs > 0:
        print(f"  Resume detected: {completed_epochs}/{num_epochs} epochs already completed.", flush=True)
        print(f"    -> Re-mining + training will start at epoch {start_epoch}.", flush=True)
        print(f"    -> Mining for epoch {start_epoch} re-runs (mining is not checkpointed; "
              f"output is deterministic from the same source checkpoint).", flush=True)
    else:
        print(f"  Fresh run: no prior checkpoint detected in {output_dir}.", flush=True)
    print(f"  Training continuity: shared output_dir + resume_from_checkpoint", flush=True)
    print(f"    -> AdamW moments + scheduler step counter preserved across epochs", flush=True)
    print(f"    -> LR scheduler horizon recomputed per call (segmented decay, not continuous)", flush=True)

    current_model   = cfg['base_model'] if start_epoch == 1 else output_dir
    prev_mining_dir = None
    if start_epoch > 1:
        prev_candidate = workdir / f"mining_epoch{start_epoch - 1}_{base_name}"
        if prev_candidate.exists():
            prev_mining_dir = prev_candidate
            print(f"  Chaining JSONL from prior epoch: {prev_mining_dir}", flush=True)
        else:
            print(f"  No prior mining dir at {prev_candidate}; epoch {start_epoch} "
                  f"will mine from the original training mixture.", flush=True)

    for epoch in range(start_epoch, num_epochs + 1):
        t_epoch = time.time()
        print(f"\n=== MCD repeated mining — epoch {epoch}/{num_epochs} ===", flush=True)
        print(f"  current_model (for mining): {current_model}", flush=True)

        mining_dir = workdir / f"mining_epoch{epoch}_{base_name}"
        t_mine = time.time()
        grass_sampler(
            current_model, stale_idx, stale_embs, c_id_to_idx, c_ids, corpus_lookup,
            mix_df, qrels_dict, cfg, config, mining_dir,
            base_jsonl_dir=prev_mining_dir,
            current_round=epoch,
            memory=memory,
        )
        print(f"  Mining (epoch {epoch}): {(time.time() - t_mine) / 60:.1f} min", flush=True)

        if cache_enabled and memory is not None:
            memory.save(memory_path)
            print(f"  Active memory persisted: {len(memory)} queries -> {memory_path}", flush=True)

        # Train: epoch 1 starts from base_model; epochs 2+ resume from output_dir's
        # latest checkpoint. cumulative_epochs grows so HF Trainer's max_steps =
        # cumulative_epochs * steps_per_epoch, leaving room for one more epoch.
        train_model_path = cfg['base_model'] if epoch == 1 else output_dir
        t_train = time.time()
        train_one_epoch_mcd(
            train_model_path, mining_dir, output_dir, cfg, ctx, config,
            cumulative_epochs=epoch,
        )
        print(f"  Training (epoch {epoch}): {(time.time() - t_train) / 60:.1f} min", flush=True)

        current_model   = output_dir
        prev_mining_dir = mining_dir

        gc.collect()
        torch.cuda.empty_cache()
        print(f"  Epoch {epoch} total: {(time.time() - t_epoch) / 60:.1f} min", flush=True)

    print(f"\n✅ MCD repeated mining done. Final model: {output_dir}", flush=True)
    return output_dir
