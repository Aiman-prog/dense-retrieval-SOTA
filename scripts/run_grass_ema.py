import gc
import json
import random
import time
import sys
import numpy as np
import torch
from pathlib import Path
from torch.optim import AdamW
from torch.nn.utils import clip_grad_norm_
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup

try:
    import bitsandbytes as bnb
    _BNB_AVAILABLE = True
except ImportError:
    _BNB_AVAILABLE = False

project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root / 'src'))

from utils.helpers import get_path, encode_batch, _pool_and_fresh_rerank
from utils.grass_candidate_memory import CandidateMemory


def encode_batch_train(model, tokenizer, texts, device, max_len, batch_size):
    """
    Encode texts with gradient tracking for the training forward pass.
    Unlike encode_batch, this does NOT wrap in torch.no_grad — the computation
    graph is retained so loss.backward() can update the model.
    Uses autocast for bfloat16 efficiency.
    Returns a (N, dim) float tensor on device with gradients.
    """
    all_embs = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        inputs = tokenizer(batch, padding=True, truncation=True,
                           max_length=max_len, return_tensors='pt').to(device)
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            out = model(**inputs)
        embs = out.last_hidden_state[:, 0, :]
        embs = torch.nn.functional.normalize(embs, dim=-1)
        all_embs.append(embs)
    return torch.cat(all_embs, dim=0)  # (N, dim), grad retained


def mine_ema_batch(model, ema_model, tokenizer, batch_items,
                   stale_idx, stale_embs, c_id_to_idx, c_ids,
                   corpus_lookup, qrels_dict, cfg, config, device,
                   current_round=1, memory=None):
    """Per-batch EMA mining (Algorithm 2, EMA branch).

    Architecture:
      student encode q -> FAISS top-P + active memory -> deterministic student
      fresh rerank -> top-L -> re-encode top-L cleanly with student (s_cur) and
      EMA teacher (s_ema) -> sigma = |s_cur - s_ema|, g = s_cur + lambda*sigma
      -> top-m -> memory update.

    All encoding is no_grad. Returns (mined, sigma_scores, log_records).
    """
    P             = cfg['P']
    L             = cfg['L']
    m             = cfg['m']
    lambda_val    = cfg['lambda_val']
    mc_batch_size = cfg.get('mc_batch_size', 256)
    max_pool_per_query = cfg.get('max_pool_per_query', P)
    q_max_len     = config['model']['query_max_len']
    p_max_len     = config['model']['passage_max_len']

    cache_cfg            = cfg.get('candidate_cache', {})
    cache_enabled        = cache_cfg.get('enabled', False)
    top_g_to_store       = cache_cfg.get('top_g_to_store', 8)
    top_sigma_to_store   = cache_cfg.get('top_sigma_to_store', 8)
    use_memory           = cache_enabled and memory is not None

    query_ids   = [item['query_id'] for item in batch_items]
    query_texts = [item['query']    for item in batch_items]

    # Deterministic query encode with student (eval, no_grad). EMA query encode
    # below is used for the EMA scoring stage only.
    prev_training = model.training
    model.eval()
    try:
        q_cur = encode_batch(model,     tokenizer, query_texts, device, q_max_len, mc_batch_size)
        q_ema = encode_batch(ema_model, tokenizer, query_texts, device, q_max_len, mc_batch_size)
    finally:
        if prev_training:
            model.train()

    _, indices = stale_idx.search(q_cur, P)

    # Build top-L shortlist per query
    memory_per_query   = {}
    memory_expired_map = {}
    if use_memory:
        for qid in query_ids:
            ids, expired = memory.get(qid, current_round)
            memory_per_query[qid]   = ids
            memory_expired_map[qid] = expired

    batch_query_shortlist, source_map, pool_stats = _pool_and_fresh_rerank(
        model, tokenizer, query_ids, q_cur,
        indices, memory_per_query, memory_expired_map,
        qrels_dict, c_ids, corpus_lookup,
        p_max_len, mc_batch_size, device,
        L, max_pool_per_query,
    )

    # Re-encode top-L cleanly for EMA scoring with BOTH models (eval, no_grad).
    # Deduped across the batch to encode each candidate text once per model.
    shortlist_cand_ids_set = set()
    for qid in query_ids:
        shortlist_cand_ids_set.update(batch_query_shortlist.get(qid, []))
    shortlist_ids    = list(shortlist_cand_ids_set)
    shortlist_texts  = [corpus_lookup.get(d, "") for d in shortlist_ids]
    shortlist_to_idx = {d: i for i, d in enumerate(shortlist_ids)}

    if shortlist_texts:
        prev_training = model.training
        model.eval()
        try:
            c_cur = encode_batch(model,     tokenizer, shortlist_texts, device, p_max_len, mc_batch_size)
            c_ema = encode_batch(ema_model, tokenizer, shortlist_texts, device, p_max_len, mc_batch_size)
        finally:
            if prev_training:
                model.train()
    else:
        c_cur = c_ema = None

    # Score top-L: sigma = |s_cur - s_ema|, g = s_cur + lambda * sigma
    mined        = {}
    sigma_scores = {}
    log_records  = []
    for i, qid in enumerate(query_ids):
        cands = batch_query_shortlist.get(qid, [])
        if not cands or c_cur is None:
            continue
        cidxs = [shortlist_to_idx[d] for d in cands]
        s_cur = q_cur[i] @ c_cur[cidxs].T
        s_ema = q_ema[i] @ c_ema[cidxs].T
        sigma = np.abs(s_cur - s_ema)
        g     = s_cur + lambda_val * sigma

        if len(cands) < m:
            mined[qid] = cands
            continue
        top_m_idxs = np.argsort(g)[::-1][:m]
        top_m_docs = [cands[k] for k in top_m_idxs]
        mined[qid]        = top_m_docs
        sigma_scores[qid] = float(sigma[top_m_idxs[0]])

        selected_docid = top_m_docs[0]
        rank_by_shat   = int(np.argsort(np.argsort(-s_cur))[top_m_idxs[0]])

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
            "s_hat_selected":       float(s_cur[top_m_idxs[0]]),
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
        log_records.append(log_record)

    return mined, sigma_scores, log_records


def train_with_ema_grass(stale_idx, stale_embs, c_id_to_idx, c_ids,
                          corpus_lookup, qrels_dict, cfg, config, ctx, debug=False):
    """
    EMA GRASS training loop (Algorithm 1 + Algorithm 2 EMA branch).

    Replaces grass_sampler() + tevatron_train_main() for EMA mode.
    Mines negatives per batch (interleaved with weight updates) using
    Teacher-Student disagreement instead of bulk MC-dropout upfront.

    Expected runtime: ~4h (vs ~22h for MC-dropout).
    """
    from models.temperature_scaled_loss import TemperatureScaledContrastiveLoss

    base_model    = cfg['base_model']
    model_name    = cfg['model_name'] + '_ema'
    ema_alpha     = cfg.get('ema_alpha', 0.999)
    lr            = float(cfg['learning_rate'])
    num_epochs    = cfg['num_epochs']
    batch_size    = cfg.get('ema_batch_size', 32)
    m             = cfg['m']
    max_grad_norm = cfg.get('max_grad_norm', 1.0)
    warmup_ratio  = cfg.get('warmup_ratio', 0.1)
    weight_decay  = cfg.get('weight_decay', 0.01)
    logging_steps = cfg.get('logging_steps', 100)
    save_steps    = cfg.get('save_steps', 500)
    mc_batch_size = cfg.get('mc_batch_size', 256)
    q_max_len     = config['model']['query_max_len']
    p_max_len     = config['model']['passage_max_len']
    temperature   = ctx['temperature']

    device    = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tokenizer = AutoTokenizer.from_pretrained(base_model)

    # Student model
    model = AutoModel.from_pretrained(base_model, torch_dtype=torch.bfloat16).to(device)
    # AdamW8bit frees ~3.9GB VRAM (optimizer states 4.5GB→0.56GB) — enables ema_batch_size=64
    # without gradient checkpointing. Falls back to AdamW + gradient checkpointing if bnb absent.
    if _BNB_AVAILABLE:
        optimizer = bnb.optim.AdamW8bit(model.parameters(), lr=lr, weight_decay=weight_decay)
        print("  [AdamW8bit] 8-bit Adam enabled — gradient checkpointing OFF", flush=True)
    else:
        model.gradient_checkpointing_enable()
        optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        print("  [AdamW8bit] bitsandbytes not found — AdamW + gradient checkpointing ON "
              "(pip install bitsandbytes for ~2x speedup)", flush=True)
    model.train()

    # EMA teacher — frozen copy, always eval, no grad. Never compiled: [S10] _foreach EMA
    # iterates raw parameters which breaks with compiled module wrappers.
    ema_model = AutoModel.from_pretrained(base_model, torch_dtype=torch.bfloat16).to(device)
    for p in ema_model.parameters():
        p.requires_grad_(False)
    ema_model.eval()

    loss_fn = TemperatureScaledContrastiveLoss(temperature=temperature)

    # [S13] torch.compile on student only — 15-25% faster forward passes via kernel fusion.
    # _model_raw kept separately: compiled wrapper may not expose HF save_pretrained.
    _model_raw = model
    try:
        model = torch.compile(model, dynamic=True)
        print("  [S13] torch.compile enabled on student", flush=True)
    except Exception as e:
        print(f"  [S13] torch.compile skipped ({e})", flush=True)

    # Load training data from mixture JSONL
    mix_dir     = get_path("processed") / "training_mixture"
    train_items = []
    for f_path in sorted(mix_dir.glob("*.jsonl")):
        if f_path.name.startswith('.'): continue
        with open(f_path) as f:
            for line in f:
                d   = json.loads(line)
                pos = d.get('positive_passages', [])
                if not pos: continue
                negs = d.get('negative_passages', [])
                train_items.append({
                    'query_id':  str(d['query_id']),
                    'query':     d['query'],
                    'pos_docid': pos[0]['docid'],
                    'neg_docid': negs[0]['docid'] if negs else None,  # [S8] mixture negative
                })
    if debug:
        train_items = train_items[:200]
        print("🐛 DEBUG mode: 200 training items", flush=True)
    random.shuffle(train_items)

    # neg_cache — mixture negatives already in train_items, no second file pass needed
    neg_cache = {it['query_id']: it['neg_docid'] for it in train_items if it['neg_docid']}

    n_batches    = len(train_items) // batch_size
    total_steps  = n_batches * num_epochs
    warmup_steps = int(warmup_ratio * total_steps)
    scheduler    = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    output_model_dir = get_path("models") / model_name
    output_model_dir.mkdir(parents=True, exist_ok=True)

    # Active candidate memory (optional)
    cache_cfg     = cfg.get('candidate_cache', {})
    cache_enabled = cache_cfg.get('enabled', False)
    memory_path   = get_path("temp_grass") / "candidate_memory_ema.pkl"
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

    log_path = output_model_dir / "mining_log.jsonl"
    mining_log_f = open(log_path, 'w')

    print(f"  EMA GRASS: {len(train_items)} examples | batch={batch_size} | "
          f"{total_steps} steps | ema_alpha={ema_alpha} | m={m}", flush=True)

    global_step = 0
    mining_round = 0
    t_start      = time.time()

    for epoch in range(num_epochs):
        random.shuffle(train_items)
        epoch_loss = 0.0

        for b in range(n_batches):
            batch_items = train_items[b * batch_size:(b + 1) * batch_size]

            # 1. Mine negatives for every query in the batch (no MAB subsetting)
            mining_round += 1
            mined = {}
            mined_sub, sigma_scores, log_records = mine_ema_batch(
                model, ema_model, tokenizer, batch_items,
                stale_idx, stale_embs, c_id_to_idx, c_ids,
                corpus_lookup, qrels_dict, cfg, config, device,
                current_round=mining_round, memory=memory,
            )
            for it in batch_items:
                qid = it['query_id']
                if mined_sub.get(qid):
                    mined[qid]     = mined_sub[qid]
                    neg_cache[qid] = mined_sub[qid][0]
            for rec in log_records:
                mining_log_f.write(json.dumps(rec, ensure_ascii=False) + '\n')

            # Non-mined queries fall back to neg_cache
            for it in batch_items:
                qid = it['query_id']
                if qid not in mined and qid in neg_cache:
                    mined[qid] = [neg_cache[qid]]

            # 2. Build text lists — skip queries with insufficient negatives
            queries, positives, negatives = [], [], []
            for item in batch_items:
                negs = mined.get(item['query_id'], [])
                if len(negs) < m: continue
                queries.append(item['query'])
                positives.append(corpus_lookup.get(item['pos_docid'], ""))
                negatives.append([corpus_lookup.get(d, "") for d in negs[:m]])
            if not queries: continue

            # 3. Forward WITH gradients (gradient_checkpointing active on model)
            model.train()
            q_embs = encode_batch_train(model, tokenizer, queries,
                                        device, q_max_len, mc_batch_size)
            # Layout: [pos0, neg0_0..neg0_m-1, pos1, neg1_0..neg1_m-1, ...]
            # TemperatureScaledContrastiveLoss auto-targets: [0, m+1, 2*(m+1), ...]
            d_texts = [t for pos, negs in zip(positives, negatives) for t in [pos] + negs]
            d_embs  = encode_batch_train(model, tokenizer, d_texts,
                                         device, p_max_len, mc_batch_size)
            loss = loss_fn(q_embs, d_embs)

            # 4. Backward
            optimizer.zero_grad(set_to_none=True)  # [S11] avoids 2.2GB gradient tensor write
            loss.backward()
            clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
            scheduler.step()

            # 5. EMA update — [S10] _foreach fuses ~240 tensor ops into 2 kernel launches
            _ema_ps = list(ema_model.parameters())
            _cur_ps = list(model.parameters())
            with torch.no_grad():
                torch._foreach_mul_(_ema_ps, ema_alpha)
                torch._foreach_add_(_ema_ps, _cur_ps, alpha=1.0 - ema_alpha)

            epoch_loss  += loss.item()
            global_step += 1

            if global_step % logging_steps == 0:
                elapsed   = time.time() - t_start
                secs_per  = elapsed / global_step
                remaining = secs_per * (total_steps - global_step)
                eta = f"{int(remaining // 3600)}h {int((remaining % 3600) // 60)}m"
                print(f"  [Epoch {epoch+1} step {b+1}/{n_batches}] "
                      f"loss={loss.item():.4f} | ETA {eta}", flush=True)

            if global_step % save_steps == 0:
                _model_raw.save_pretrained(str(output_model_dir))
                tokenizer.save_pretrained(str(output_model_dir))

        print(f"  Epoch {epoch+1} done. avg_loss={epoch_loss / n_batches:.4f}", flush=True)

    _model_raw.save_pretrained(str(output_model_dir))
    tokenizer.save_pretrained(str(output_model_dir))
    mining_log_f.close()
    if memory is not None:
        memory.save(memory_path)
        print(f"  Active memory saved: {len(memory)} queries -> {memory_path}", flush=True)
    print(f"  EMA GRASS done. Model saved to {output_model_dir}", flush=True)

    del model, ema_model, _model_raw
    gc.collect()
    torch.cuda.empty_cache()

    return output_model_dir
