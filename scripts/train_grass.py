import os
import sys
import time
import gc
import json
import random
import subprocess
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from torch.optim import AdamW
from torch.nn.utils import clip_grad_norm_
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup
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
                   corpus_lookup, qrels_dict, cfg, config, device):
    """
    Per-batch EMA mining (Algorithm 2, EMA branch).

    For each query in the batch:
      1. ANN search with q_cur (current model) — paper Alg2 Line 1
      2. Shortlist top-L by stale_embs @ q_cur — approximates paper Line 3 cheap pass
      3. Encode shortlist with both current and EMA model
      4. ŝ[d] = s_cur (cheap-pass score, not updated in EMA branch — paper Lines 12-14)
         σ[d] = |s_cur - s_ema|
         g[d] = s_cur + lambda * sigma  →  top-m negatives

    All encoding is done under torch.no_grad. Returns dict {query_id → [neg_docid, ...]}.
    """
    P             = cfg['P']
    L             = cfg['L']
    m             = cfg['m']
    lambda_val    = cfg['lambda_val']
    mc_batch_size = cfg.get('mc_batch_size', 256)
    q_max_len     = config['model']['query_max_len']
    p_max_len     = config['model']['passage_max_len']

    query_ids   = [item['query_id'] for item in batch_items]
    query_texts = [item['query']    for item in batch_items]

    # Encode queries with both models — no grad needed for mining
    model.eval()
    with torch.no_grad():
        q_cur = encode_batch(model,     tokenizer, query_texts, device, q_max_len, mc_batch_size)
        q_ema = encode_batch(ema_model, tokenizer, query_texts, device, q_max_len, mc_batch_size)
    model.train()

    # ANN search with q_cur — paper Alg2 Line 1 uses current model query, NOT ema
    _, indices = stale_idx.search(q_cur, P)

    # Per-query: filter positives, shortlist top-L using stale embeddings @ q_cur
    batch_query_shortlist  = {}
    shortlist_cand_ids_set = set()
    for i, qid in enumerate(query_ids):
        cands = [c_ids[j] for j in indices[i]
                 if j >= 0 and c_ids[j] not in qrels_dict.get(qid, set())]
        if not cands:
            batch_query_shortlist[qid] = []
            continue
        stale_idxs = [c_id_to_idx[d] for d in cands]
        scores     = stale_embs[stale_idxs] @ q_cur[i]  # cheap pass ≈ paper Line 3
        top_l      = np.argsort(scores)[::-1][:L]
        shortlist  = [cands[k] for k in top_l]
        batch_query_shortlist[qid] = shortlist
        shortlist_cand_ids_set.update(shortlist)

    shortlist_ids    = list(shortlist_cand_ids_set)
    shortlist_texts  = [corpus_lookup.get(d, "") for d in shortlist_ids]
    shortlist_to_idx = {d: i for i, d in enumerate(shortlist_ids)}

    # Encode shortlist with both models
    with torch.no_grad():
        c_cur = encode_batch(model,     tokenizer, shortlist_texts, device, p_max_len, mc_batch_size)
        c_ema = encode_batch(ema_model, tokenizer, shortlist_texts, device, p_max_len, mc_batch_size)

    # Compute g = s_cur + lambda * |s_cur - s_ema| and select top-m (paper Lines 12-14)
    mined = {}
    for i, qid in enumerate(query_ids):
        cands = batch_query_shortlist[qid]
        if len(cands) < m:
            mined[qid] = cands
            continue
        cidxs = [shortlist_to_idx[d] for d in cands]
        s_cur = q_cur[i] @ c_cur[cidxs].T   # (N_cands,) — ŝ[d] per paper
        s_ema = q_ema[i] @ c_ema[cidxs].T   # (N_cands,)
        sigma = np.abs(s_cur - s_ema)        # σ[d] — temporal disagreement
        g     = s_cur + lambda_val * sigma
        top_m = np.argsort(g)[::-1][:m]
        mined[qid] = [cands[k] for k in top_m]

    return mined


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

    # Student model — gradient checkpointing prevents OOM with full backprop on BGE-M3
    model = AutoModel.from_pretrained(base_model, torch_dtype=torch.bfloat16).to(device)
    model.gradient_checkpointing_enable()
    model.train()

    # EMA teacher — frozen copy, always eval, no grad
    ema_model = AutoModel.from_pretrained(base_model, torch_dtype=torch.bfloat16).to(device)
    for p in ema_model.parameters():
        p.requires_grad_(False)
    ema_model.eval()

    loss_fn   = TemperatureScaledContrastiveLoss(temperature=temperature)
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

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
                train_items.append({
                    'query_id': str(d['query_id']),
                    'query':    d['query'],
                    'pos_docid': pos[0]['docid'],
                })
    if debug:
        train_items = train_items[:200]
        print("🐛 DEBUG mode: 200 training items", flush=True)
    random.shuffle(train_items)

    n_batches    = len(train_items) // batch_size
    total_steps  = n_batches * num_epochs
    warmup_steps = int(warmup_ratio * total_steps)
    scheduler    = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    output_model_dir = get_path("models") / model_name
    output_model_dir.mkdir(parents=True, exist_ok=True)

    print(f"  EMA GRASS: {len(train_items)} examples | batch={batch_size} | "
          f"{total_steps} steps | ema_alpha={ema_alpha} | m={m}", flush=True)

    global_step = 0
    t_start     = time.time()

    for epoch in range(num_epochs):
        random.shuffle(train_items)
        epoch_loss = 0.0

        for b in range(n_batches):
            batch_items = train_items[b * batch_size:(b + 1) * batch_size]

            # 1. Mine negatives per batch — all under no_grad
            mined = mine_ema_batch(
                model, ema_model, tokenizer, batch_items,
                stale_idx, stale_embs, c_id_to_idx, c_ids,
                corpus_lookup, qrels_dict, cfg, config, device
            )

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
            optimizer.zero_grad()
            loss.backward()
            clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
            scheduler.step()

            # 5. EMA update — every gradient step (Algorithm 1 Line 9)
            with torch.no_grad():
                for p_ema, p_cur in zip(ema_model.parameters(), model.parameters()):
                    p_ema.data.mul_(ema_alpha).add_(p_cur.data, alpha=1.0 - ema_alpha)

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
                model.save_pretrained(str(output_model_dir))
                tokenizer.save_pretrained(str(output_model_dir))

        print(f"  Epoch {epoch+1} done. avg_loss={epoch_loss / n_batches:.4f}", flush=True)

    model.save_pretrained(str(output_model_dir))
    tokenizer.save_pretrained(str(output_model_dir))
    print(f"  EMA GRASS done. Model saved to {output_model_dir}", flush=True)

    del model, ema_model
    gc.collect()
    torch.cuda.empty_cache()

    return output_model_dir


def grass_sampler(model_path, stale_idx, stale_embs, c_id_to_idx, c_ids, corpus_lookup, mix_df,
                  qrels_dict, cfg, config, out_dir):
    """
    GrassSampler (Algorithm 2): mines hard negatives for all training queries using
    a stale ANN index and MC-dropout uncertainty estimation.

    Args:
      model_path   — path to the base model used for encoding (HuggingFace checkpoint)
      stale_idx    — FAISS IndexFlatIP built from base model corpus embeddings (can be stale)
      c_ids        — list of corpus doc IDs parallel to the FAISS index rows
      corpus_lookup — dict mapping doc ID → passage text for candidate encoding
      mix_df       — DataFrame of training queries (columns: query_id, query)
      qrels_dict   — dict mapping query_id → set of positive doc IDs (for filtering true positives)
      cfg          — grass config block from config.yaml (contains P, L, m, T, lambda_val, etc.)
      config       — full config dict (used for model max lengths)
      out_dir      — output directory where updated JSONL training files are written

    Hyperparameters (from cfg):
      P          — pool size: number of candidates retrieved per query from the stale ANN index
      L          — shortlist size: top-L candidates by cheap score before MC-dropout (L <= P)
      m          — number of hard negatives selected per query (= train_group_size - 1)
      T          — number of MC-dropout forward passes for uncertainty estimation
      lambda_val — trade-off weight: higher lambda promotes more uncertain (exploratory) negatives

    Algorithm:
      1. Retrieve top-P candidate doc IDs from the stale ANN index (index can be stale —
         Algorithm 2, Line 1). True positives from qrels are filtered out.
      2. One cheap eval-mode pass scores all P candidates; shortlist to top-L likely confusers.
      3. Run T MC-dropout passes on the shortlist only. Each pass samples a slightly different
         model due to dropout, giving T similarity scores per (query, candidate) pair.
      4. Compute per-candidate statistics across T passes:
           s_hat(q, d) = mean similarity  — estimates expected hardness
           sigma(q, d) = std similarity   — estimates model uncertainty about this doc
      5. Score each candidate: g(q, d) = s_hat + lambda * sigma
         High g means the doc is both confusing (hard) and uncertain (informative).
      6. Select the top-m candidates by g as the new hard negatives for this query.

    Queries are processed in batches (query_batch_size) to keep peak GPU memory bounded.
    Writes updated mixture JSONL files to out_dir with negative_passages replaced.
    """
    P                = cfg['P']           # candidates retrieved per query from stale ANN
    L                = cfg['L']           # shortlist: top-L by cheap pass before MC-dropout (L <= P)
    m                = cfg['m']           # hard negatives selected per query (train_group_size - 1)
    T                = cfg['T']           # MC-dropout forward passes for uncertainty estimation
    lambda_val       = cfg['lambda_val']  # weight of uncertainty term in g = s_hat + lambda * sigma
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
        n_dropout = sum(1 for m in model.modules() if isinstance(m, torch.nn.Dropout))
        for module in model.modules():
            if isinstance(module, torch.nn.Dropout):
                module.p = mc_dropout_p
        print(f"  MC-dropout p set to {mc_dropout_p} ({n_dropout} Dropout layers)", flush=True)
    print(f"  Loaded model for GrassSampler (T={T}, P={P}, m={m}, lambda={lambda_val})", flush=True)

    query_ids  = mix_df['query_id'].astype(str).tolist()
    query_texts = mix_df['query'].tolist()
    n_queries  = len(query_ids)
    n_batches  = (n_queries + query_batch_size - 1) // query_batch_size
    print(f"  Processing {n_queries} queries in {n_batches} batches (batch_size={query_batch_size})...", flush=True)

    # Process queries in batches to avoid holding (T × N_all_queries × dim) in memory.
    # Peak memory per batch: T × query_batch_size × L × dim (shortlisted candidates).
    mined_negs = {}
    t_loop_start = time.time()
    for b, batch_start in enumerate(range(0, n_queries, query_batch_size)):
        batch_ids   = query_ids[batch_start:batch_start + query_batch_size]
        batch_texts = query_texts[batch_start:batch_start + query_batch_size]

        # Deterministic query encoding for ANN retrieval and shortlisting (Algorithm 2, Lines 1 & 3)
        model.eval()
        q_embs_det = encode_batch(model, tokenizer, batch_texts, device, q_max_len, mc_batch_size)
        model.train()

        _, indices = stale_idx.search(q_embs_det, P)

        # T MC-dropout query encodings for uncertainty scoring (Algorithm 2, Lines 6-7)
        q_embs_all   = [encode_batch(model, tokenizer, batch_texts, device, q_max_len, mc_batch_size)
                        for _ in range(T)]
        q_embs_stack = np.stack(q_embs_all, axis=0)  # (T, B, dim)

        # Collect candidate doc IDs for this batch, filtering true positives
        batch_query_cands = {}
        for i, qid in enumerate(batch_ids):
            cands = [c_ids[j] for j in indices[i] if j >= 0 and c_ids[j] not in qrels_dict.get(qid, set())]
            batch_query_cands[qid] = cands

        # Stage 1: shortlist to top-L using pre-computed stale embeddings (no re-encoding needed)
        batch_query_shortlist = {}
        shortlist_cand_ids_set = set()
        for i, qid in enumerate(batch_ids):
            cands = batch_query_cands[qid]
            if not cands:
                batch_query_shortlist[qid] = cands
                continue
            stale_idxs = [c_id_to_idx[d] for d in cands]
            scores     = stale_embs[stale_idxs] @ q_embs_det[i]   # (N_cands_q,)
            top_l      = np.argsort(scores)[::-1][:L]
            shortlist  = [cands[k] for k in top_l]
            batch_query_shortlist[qid] = shortlist
            shortlist_cand_ids_set.update(shortlist)

        # Stage 2: T MC-dropout passes only on shortlisted candidates (Algorithm 2, Lines 5-8)
        shortlist_ids   = list(shortlist_cand_ids_set)
        shortlist_texts = [corpus_lookup.get(did, "") for did in shortlist_ids]
        shortlist_to_idx = {did: i for i, did in enumerate(shortlist_ids)}

        c_embs_all   = [encode_batch(model, tokenizer, shortlist_texts, device, p_max_len, mc_batch_size)
                        for _ in range(T)]
        c_embs_stack = np.stack(c_embs_all, axis=0)  # (T, N_shortlist, dim)

        # Compute g = s_hat + lambda * sigma and select top-m for each query in this batch
        batch_sigmas, batch_g = [], []
        for i, qid in enumerate(batch_ids):
            cands = batch_query_shortlist[qid]
            if not cands:
                continue
            cand_idxs = [shortlist_to_idx[d] for d in cands]

            # q_t: (T, 1, dim)  @  c_t: (T, dim, N_cands_q)  →  (T, N_cands_q)
            q_t  = torch.from_numpy(q_embs_stack[:, i, :]).unsqueeze(1)      # (T, 1, dim)
            c_t  = torch.from_numpy(c_embs_stack[:, cand_idxs, :])           # (T, N_cands_q, dim)
            sims = torch.bmm(q_t, c_t.transpose(1, 2)).squeeze(1).numpy()    # (T, N_cands_q)

            s_hat = sims.mean(axis=0)  # expected similarity — measures hardness
            sigma = sims.std(axis=0)   # std across passes  — measures model uncertainty
            g     = s_hat + lambda_val * sigma

            top_m = np.argsort(g)[::-1][:m]
            mined_negs[qid] = [cands[k] for k in top_m]
            batch_sigmas.append(sigma.mean())
            batch_g.append(g.mean())

        if b < 3 or (b + 1) % 100 == 0:
            elapsed   = time.time() - t_loop_start
            secs_per  = elapsed / (b + 1)
            remaining = secs_per * (n_batches - b - 1)
            eta       = f"{int(remaining // 3600)}h {int((remaining % 3600) // 60)}m"
            n_filtered = sum(len(v) for v in batch_query_cands.values())
            n_raw = len(batch_ids) * P
            print(f"  Batch {b+1}/{n_batches} | ETA {eta} | "
                  f"P→L filter: {n_filtered}/{n_raw} → shortlist {len(shortlist_ids)} unique | "
                  f"sigma mean: {np.mean(batch_sigmas):.5f} | "
                  f"g mean: {np.mean(batch_g):.4f}", flush=True)

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
    import argparse
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--mode', type=str, default=None,
                        help='Override uncertainty_mode from config: "ema" or "mc_dropout"')
    cli_args, _ = parser.parse_known_args()

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

    workdir = get_path("temp_grass")
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

    if cli_args.debug:
        mix_df = mix_df.head(100)
        cfg = {**cfg, 'T': 2, 'P': 20, 'L': 5, 'm': 2, 'query_batch_size': 10}
        print("🐛 DEBUG mode: 100 queries, T=2, P=20, L=5", flush=True)

    print(f"✅ Setup complete. corpus_lookup={len(corpus_lookup)} passages, qrels={len(qrels_dict)} queries, mix_df={len(mix_df)} unique queries.", flush=True)

    # Build stale ANN index from base model — never refreshed (Algorithm 2, Line 1)
    stale_dir = workdir / "stale_index"
    stale_dir.mkdir(exist_ok=True)
    stale_pkl = stale_dir / "corpus.pkl"
    if not stale_pkl.exists():
        print("📦 Building stale ANN index from base model...", flush=True)
        encode_to_pickle(cfg['base_model'], corpus_file, stale_pkl, False, ctx, config)
    stale_idx, stale_embs, c_ids = build_faiss_index(stale_pkl)
    # Build a fast doc-ID → embedding-row lookup for the cheap shortlisting pass
    c_id_to_idx = {did: i for i, did in enumerate(c_ids)}
    print(f"✅ Stale index ready: {len(c_ids)} passages", flush=True)

    uncertainty_mode = cli_args.mode or cfg.get('uncertainty_mode', 'mc_dropout')
    print(f"\n{'='*50}", flush=True)
    print(f"  GRASS MODE: {uncertainty_mode.upper()}", flush=True)
    print(f"{'='*50}\n", flush=True)

    if uncertainty_mode == 'ema':
        # --- EMA GRASS: per-batch mining + teacher-student disagreement (~4h) ---
        output_model_dir = train_with_ema_grass(
            stale_idx, stale_embs, c_id_to_idx, c_ids,
            corpus_lookup, qrels_dict, cfg, config, ctx, debug=cli_args.debug
        )
    else:
        # --- MC-DROPOUT GRASS: bulk mining upfront + Tevatron training (~22h) ---
        mix_out = workdir / "grass_train"
        print("🔍 Running GrassSampler (MC-dropout)...", flush=True)
        grass_sampler(cfg['base_model'], stale_idx, stale_embs, c_id_to_idx, c_ids, corpus_lookup,
                      mix_df, qrels_dict, cfg, config, mix_out)

        output_model_dir = get_path("models") / (cfg['model_name'] + '_mcdp')
        training_args = [
            '--output_dir', str(output_model_dir), '--model_name_or_path', cfg['base_model'],
            '--dataset_name', 'json', '--dataset_path', str(mix_out / "*.jsonl"),
            '--dataset_split', 'train', '--per_device_train_batch_size', str(cfg['batch_size']),
            '--train_group_size', str(cfg['train_group_size']), '--learning_rate', str(cfg['learning_rate']),
            '--num_train_epochs', str(cfg['num_epochs']), '--bf16', 'True', '--dtype', 'bfloat16',
            '--overwrite_output_dir', 'True',
            '--save_strategy', cfg['save_strategy'],
            '--save_steps', str(cfg.get('save_steps', 500)),
            '--save_total_limit', str(cfg['save_total_limit']),
            '--ignore_data_skip', 'True',
            '--warmup_ratio', str(cfg.get('warmup_ratio', 0.1)),
            '--weight_decay', str(cfg.get('weight_decay', 0.01)),
            '--max_grad_norm', str(cfg.get('max_grad_norm', 1.0)),
            '--dataloader_num_workers', str(cfg['dataloader_num_workers']),
            '--attn_implementation', 'eager', '--optim', 'adamw_torch_fused',
            '--logging_steps', str(cfg['logging_steps']),
            '--pooling', ctx['pooling'],
            '--normalize', str(ctx['normalize']),
            '--temperature', str(ctx['temperature']),
        ]
        sys.argv = ['train.py'] + training_args
        patch_tevatron_loss(ctx['temperature'])
        tevatron_train_main()

    gc.collect()
    torch.cuda.empty_cache()

    # --- EVALUATE ---
    for domain in config['evaluation'].get('eval_domains', []):
        subprocess.run([
            sys.executable, str(project_root / 'src/evaluation/evaluate.py'),
            '--model_path', str(output_model_dir),
            '--domain', domain,
        ], check=True)

    scores = [
        json.load(open(get_path("results") / f"{domain}_results.json"))['metrics'].get('ndcg_cut_10', 0)
        for domain in config['evaluation'].get('eval_domains', [])
    ]
    print(f"📈 GRASS Mean NDCG@10: {sum(scores) / len(scores):.4f}", flush=True)
    print(f"✅ GRASS complete. Model saved to: {output_model_dir}", flush=True)


if __name__ == "__main__":
    main()
