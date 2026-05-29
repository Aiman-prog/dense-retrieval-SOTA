"""
GRASS Sequential Mode — interleaved training + MC-dropout mining on a single GPU.
Mines n_das queries every mine_every training steps for exact coverage control.

Coverage per epoch ≈ n_das × (steps_per_epoch / mine_every) / N_queries
  e.g. n_das=30, mine_every=2, 5734 steps/epoch, 330K queries → ~26%
"""
import gc
import json
import random
import time
import sys
import argparse
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

from utils.helpers import (
    get_path, get_training_context, load_config,
    encode_batch, encode_to_pickle, build_faiss_index,
    _load_qrels, _load_corpus_lookup,
    _pool_and_fresh_rerank, set_seed, evaluate_bright,
)
from utils.grass_candidate_memory import CandidateMemory


def encode_batch_train(model, tokenizer, texts, device, max_len, batch_size):
    """Gradient-tracked forward pass for training (no no_grad wrapper)."""
    all_embs = []
    for i in range(0, len(texts), batch_size):
        batch  = texts[i:i + batch_size]
        inputs = tokenizer(batch, padding=True, truncation=True,
                           max_length=max_len, return_tensors='pt').to(device)
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            out = model(**inputs)
        embs = out.last_hidden_state[:, 0, :]
        embs = torch.nn.functional.normalize(embs, dim=-1)
        all_embs.append(embs)
    return torch.cat(all_embs, dim=0)


def _mine_queries(model, tokenizer, qids, qid_to_text,
                  stale_idx, stale_embs, c_id_to_idx, c_ids,
                  corpus_lookup, qrels_dict, cfg, config, device,
                  current_round=1, memory=None):
    """MC-dropout mining for a specific list of qids.

    Architecture:
      stale FAISS top-P + active memory -> dedup + filter positives ->
      deterministic current-model fresh rerank (eval+no_grad, model mode
      saved and restored) -> top-L -> T MC-dropout passes -> top-m by g.

    Note: model is in train() mode when called from the seq training loop.
    _pool_and_fresh_rerank saves model.training and restores it after pool
    encoding, so the T MC passes below run with dropout active as intended.

    Returns ({qid: (neg_docid, sigma)}, log_records).
    """
    P, L, T = cfg['P'], cfg['L'], cfg.get('T', 5)
    m, lv   = cfg['m'], cfg['lambda_val']
    mc_bs   = cfg.get('mc_batch_size', 512)
    max_pool_per_query = cfg.get('max_pool_per_query', P)
    q_max   = config['model']['query_max_len']
    p_max   = config['model']['passage_max_len']

    cache_cfg            = cfg.get('candidate_cache', {})
    cache_enabled        = cache_cfg.get('enabled', False)
    top_g_to_store       = cache_cfg.get('top_g_to_store', 8)
    top_sigma_to_store   = cache_cfg.get('top_sigma_to_store', 8)
    use_memory           = cache_enabled and memory is not None

    texts = [qid_to_text[q] for q in qids]

    # Deterministic query encode (eval + no_grad inside encode_batch)
    model.eval()
    q_det = encode_batch(model, tokenizer, texts, device, q_max, mc_bs)
    model.train()  # restore train mode for T MC passes

    _, indices = stale_idx.search(q_det, P)

    memory_per_query   = {}
    memory_expired_map = {}
    if use_memory:
        for qid in qids:
            ids, expired = memory.get(qid, current_round)
            memory_per_query[qid]   = ids
            memory_expired_map[qid] = expired

    batch_query_shortlist, source_map, pool_stats = _pool_and_fresh_rerank(
        model, tokenizer, qids, q_det,
        indices, memory_per_query, memory_expired_map,
        qrels_dict, c_ids, corpus_lookup,
        p_max, mc_bs, device,
        L, max_pool_per_query,
    )
    shortlist_cand_ids_set = set()
    for qid in qids:
        shortlist_cand_ids_set.update(batch_query_shortlist.get(qid, []))
    shortlist_ids    = list(shortlist_cand_ids_set)
    shortlist_texts  = [corpus_lookup.get(d, "") for d in shortlist_ids]
    shortlist_to_idx = {d: i for i, d in enumerate(shortlist_ids)}

    if not shortlist_texts:
        return {}, []

    # T MC-dropout passes — vectorized; model is still in train() mode (dropout active)
    q_mc = encode_batch(model, tokenizer, texts * T,
                        device, q_max, mc_bs).reshape(T, len(texts), -1)
    c_mc = encode_batch(model, tokenizer, shortlist_texts * T,
                        device, p_max, mc_bs).reshape(T, len(shortlist_texts), -1)

    results     = {}
    log_records = []
    for i, qid in enumerate(qids):
        cands = batch_query_shortlist.get(qid, [])
        if not cands:
            continue
        cidxs = [shortlist_to_idx[d] for d in cands]
        sims  = np.einsum('td,tnd->tn', q_mc[:, i, :], c_mc[:, cidxs, :])
        s_hat = sims.mean(axis=0)
        sigma = sims.std(axis=0)
        g     = s_hat + lv * sigma

        top_m_idxs = np.argsort(g)[::-1][:m]
        top_m_docs = [cands[k] for k in top_m_idxs]
        results[qid] = (top_m_docs[0], float(sigma[top_m_idxs[0]]))

        selected_docid = top_m_docs[0]
        rank_by_shat   = int(np.argsort(np.argsort(-s_hat))[top_m_idxs[0]])

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
        log_records.append(log_record)

    return results, log_records


def run_seq_pipeline(stale_idx, stale_embs, c_id_to_idx, c_ids,
                     corpus_lookup, qrels_dict, cfg, config, ctx,
                     debug=False, n_das_override=None, mine_every_override=None):
    """Mine-every-N-steps GRASS pipeline. Returns output_model_dir.

    Assumes train_grass.py already loaded stale FAISS state, corpus, qrels.
    Loads its own train_items from data/processed/training_mixture/*.jsonl since
    it needs the full item shape (query_id, query, pos_docid, neg_docid)."""
    n_das      = n_das_override      or cfg.get('n_das', cfg.get('mab_n_das', 5))
    mine_every = mine_every_override or cfg.get('mine_every', 2)

    # Load training data from mixture dir
    mix_dir     = get_path("processed") / "training_mixture"
    train_items = []
    for f_path in sorted(mix_dir.glob("*.jsonl")):
        if f_path.name.startswith('.'):
            continue
        with open(f_path) as f:
            for line in f:
                d    = json.loads(line)
                pos  = d.get('positive_passages', [])
                if not pos:
                    continue
                negs = d.get('negative_passages', [])
                train_items.append({
                    'query_id':  str(d['query_id']),
                    'query':     d['query'],
                    'pos_docid': pos[0]['docid'],
                    'neg_docid': negs[0]['docid'] if negs else None,
                })
    if debug:
        train_items = train_items[:512]
        print("[SEQ] DEBUG: 512 items", flush=True)
    random.shuffle(train_items)

    neg_cache   = {it['query_id']: it['neg_docid'] for it in train_items if it['neg_docid']}
    qid_to_text = {it['query_id']: it['query'] for it in train_items}
    all_qids    = list(qid_to_text.keys())
    print(f"[SEQ] {len(train_items)} training examples | {len(all_qids)} unique queries",
          flush=True)

    print(f"[SEQ] Random subset mining: n_das={n_das}", flush=True)

    from models.temperature_scaled_loss import TemperatureScaledContrastiveLoss

    lr            = float(cfg['learning_rate'])
    num_epochs    = cfg['num_epochs']
    batch_size    = cfg.get('batch_size', 64)
    mc_batch_size = cfg.get('mc_batch_size', 512)
    max_grad_norm = cfg.get('max_grad_norm', 1.0)
    warmup_ratio  = cfg.get('warmup_ratio', 0.1)
    weight_decay  = cfg.get('weight_decay', 0.01)
    logging_steps = cfg.get('logging_steps', 100)
    save_steps    = cfg.get('save_steps', 1000)
    q_max_len     = config['model']['query_max_len']
    p_max_len     = config['model']['passage_max_len']
    temperature   = ctx['temperature']
    mc_dropout_p  = cfg.get('mc_dropout_p', 0.3)

    output_model_dir = get_path("models") / (cfg['model_name'] + '_seq')
    output_model_dir.mkdir(parents=True, exist_ok=True)

    device    = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tokenizer = AutoTokenizer.from_pretrained(cfg['base_model'])
    model     = AutoModel.from_pretrained(cfg['base_model'],
                                          torch_dtype=torch.bfloat16).to(device)

    if mc_dropout_p != 0.1:
        n_layers = sum(1 for m in model.modules() if isinstance(m, torch.nn.Dropout))
        for module in model.modules():
            if isinstance(module, torch.nn.Dropout):
                module.p = mc_dropout_p
        print(f"[SEQ] MC-dropout p={mc_dropout_p} on {n_layers} layers", flush=True)

    if _BNB_AVAILABLE:
        optimizer = bnb.optim.AdamW8bit(model.parameters(), lr=lr, weight_decay=weight_decay)
        print("[SEQ] AdamW8bit enabled", flush=True)
    else:
        model.gradient_checkpointing_enable()
        optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        print("[SEQ] AdamW + gradient checkpointing", flush=True)
    model.train()

    loss_fn    = TemperatureScaledContrastiveLoss(temperature=temperature)
    _model_raw = model
    try:
        torch._dynamo.config.suppress_errors = True
        model = torch.compile(model, dynamic=True)
        print("[SEQ] torch.compile enabled", flush=True)
    except Exception as e:
        print(f"[SEQ] torch.compile skipped ({e})", flush=True)

    n_batches    = len(train_items) // batch_size
    total_steps  = n_batches * num_epochs
    warmup_steps = int(warmup_ratio * total_steps)
    scheduler    = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    est_coverage = n_das * (n_batches / mine_every) / max(1, len(all_qids))
    print(f"[SEQ] {total_steps} total steps | mine_every={mine_every} | "
          f"n_das={n_das}", flush=True)
    print(f"[SEQ] Expected coverage/epoch ≈ {est_coverage:.1%} "
          f"({n_das} × {n_batches}/{mine_every} / {len(all_qids)})", flush=True)

    # Active candidate memory (optional)
    cache_cfg     = cfg.get('candidate_cache', {})
    cache_enabled = cache_cfg.get('enabled', False)
    memory_path   = get_path("temp_grass_seq") / "candidate_memory.pkl"
    memory = None
    if cache_enabled:
        memory = CandidateMemory.load(
            memory_path,
            max_per_query      = cache_cfg.get('max_candidates_per_query', 64),
            ttl_rounds         = cache_cfg.get('ttl_rounds', 2),
            top_g_to_store     = cache_cfg.get('top_g_to_store', 8),
            top_sigma_to_store = cache_cfg.get('top_sigma_to_store', 8),
        )
        print(f"[SEQ] Active memory loaded: {len(memory)} queries (ttl={memory.ttl_rounds})",
              flush=True)

    mining_log_path = output_model_dir / "mining_log.jsonl"
    mining_log_f    = open(mining_log_path, 'w')

    global_step  = 0
    mining_round = 0
    t_start      = time.time()

    for epoch in range(num_epochs):
        random.shuffle(train_items)
        epoch_loss = 0.0
        n_mined_ep = 0  # total mining events this epoch (>= unique queries mined)

        for b in range(n_batches):
            # Mining round — every mine_every training steps
            if global_step % mine_every == 0:
                selected = random.sample(all_qids, min(n_das, len(all_qids)))
                selected = [q for q in selected if q in qid_to_text]

                if selected:
                    mining_round += 1
                    mined, log_records = _mine_queries(
                        model, tokenizer, selected, qid_to_text,
                        stale_idx, stale_embs, c_id_to_idx, c_ids,
                        corpus_lookup, qrels_dict, cfg, config, device,
                        current_round=mining_round, memory=memory,
                    )
                    for qid, (neg_docid, sigma) in mined.items():
                        neg_cache[qid] = neg_docid
                    for rec in log_records:
                        mining_log_f.write(json.dumps(rec, ensure_ascii=False) + '\n')
                    n_mined_ep += len(mined)
                    model.train()

            # Training step
            batch_items = train_items[b * batch_size:(b + 1) * batch_size]
            queries, positives, negatives = [], [], []
            for item in batch_items:
                neg_docid = neg_cache.get(item['query_id'])
                if not neg_docid:
                    continue
                queries.append(item['query'])
                positives.append(corpus_lookup.get(item['pos_docid'], ''))
                negatives.append([corpus_lookup.get(neg_docid, '')])
            if not queries:
                continue

            model.train()
            q_embs  = encode_batch_train(model, tokenizer, queries,
                                         device, q_max_len, mc_batch_size)
            d_texts = [t for pos, negs in zip(positives, negatives) for t in [pos] + negs]
            d_embs  = encode_batch_train(model, tokenizer, d_texts,
                                          device, p_max_len, mc_batch_size)
            loss = loss_fn(q_embs, d_embs)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
            scheduler.step()

            epoch_loss  += loss.item()
            global_step += 1

            if global_step % logging_steps == 0:
                elapsed   = time.time() - t_start
                secs_per  = elapsed / global_step
                remaining = secs_per * (total_steps - global_step)
                eta       = f"{int(remaining // 3600)}h {int((remaining % 3600) // 60)}m"
                ep_cov    = n_mined_ep / max(1, len(all_qids))
                print(f"[SEQ] Ep{epoch+1} step {b+1}/{n_batches} | "
                      f"loss={loss.item():.4f} | ep_coverage={ep_cov:.1%} | ETA {eta}",
                      flush=True)

            if global_step % save_steps == 0:
                ckpt = output_model_dir / f"checkpoint-{global_step}"
                ckpt.mkdir(parents=True, exist_ok=True)
                _model_raw.save_pretrained(str(ckpt))
                tokenizer.save_pretrained(str(ckpt))
                torch.save(scheduler.state_dict(), ckpt / "scheduler.pt")
                torch.save(optimizer.state_dict(), ckpt / "optimizer.pt")
                print(f"[SEQ] Checkpoint saved: {ckpt.name}", flush=True)

        ep_cov = n_mined_ep / max(1, len(all_qids))
        print(f"[SEQ] Epoch {epoch+1} done. "
              f"avg_loss={epoch_loss / n_batches:.4f} | "
              f"epoch_coverage={ep_cov:.1%} | mined_events={n_mined_ep}",
              flush=True)

    _model_raw.save_pretrained(str(output_model_dir))
    tokenizer.save_pretrained(str(output_model_dir))
    mining_log_f.close()
    if memory is not None:
        memory.save(memory_path)
        print(f"[SEQ] Active memory saved: {len(memory)} queries -> {memory_path}",
              flush=True)
    print(f"[SEQ] Training complete. Model at {output_model_dir}", flush=True)

    del model, _model_raw
    gc.collect()
    torch.cuda.empty_cache()

    evaluate_bright(ctx, config, output_model_dir, temp_workdir_key='temp_grass_seq')
    return output_model_dir


def main():
    """Standalone entry point. Loads everything then calls run_seq_pipeline."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--recipe',       default='grass')
    parser.add_argument('--n_das',        type=int,   default=None)
    parser.add_argument('--mine_every',   type=int,   default=None)
    parser.add_argument('--model_suffix', type=str,   default=None)
    parser.add_argument('--num_epochs',   type=int,   default=None)
    parser.add_argument('--debug',        action='store_true')
    args = parser.parse_args()

    config = load_config()
    cfg    = config['training'][args.recipe]
    ctx    = get_training_context(args.recipe)
    set_seed(config.get('seed', 42))

    if args.num_epochs is not None:
        cfg = {**cfg, 'num_epochs': args.num_epochs}
    if args.model_suffix is not None:
        cfg = {**cfg, 'model_name': cfg['model_name'] + '_' + args.model_suffix}

    from data.preprocessor import run_setup
    corpus_file, query_file, qrels_file = run_setup()

    workdir   = get_path("temp_grass_seq")
    workdir.mkdir(exist_ok=True, parents=True)
    stale_pkl = workdir / "stale_index" / "corpus.pkl"
    stale_pkl.parent.mkdir(exist_ok=True)
    if not stale_pkl.exists():
        print("[SEQ] Building stale ANN index...", flush=True)
        encode_to_pickle(cfg['base_model'], corpus_file, stale_pkl, False, ctx, config)
    print(f"[SEQ] Stale index ready: {stale_pkl}", flush=True)

    stale_idx, stale_embs, c_ids = build_faiss_index(stale_pkl)
    c_id_to_idx   = {did: i for i, did in enumerate(c_ids)}
    corpus_lookup = _load_corpus_lookup(corpus_file)
    qrels_dict    = _load_qrels(qrels_file)

    run_seq_pipeline(stale_idx, stale_embs, c_id_to_idx, c_ids,
                     corpus_lookup, qrels_dict, cfg, config, ctx,
                     debug=args.debug,
                     n_das_override=args.n_das,
                     mine_every_override=args.mine_every)


if __name__ == "__main__":
    main()
