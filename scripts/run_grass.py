"""
GRASS — paper-faithful Algorithm 1 with a pluggable Algorithm 2 σ estimator.

Reference: Venky & Avishek, "The GRASS Retriever" (2025-10-15).

Outer loop (Algorithm 1):
  for each epoch:
    for each minibatch:
      for each (q, pos) in batch:
        negs[q] ← GRASSSAMPLER(q, pos, I, P, L, k, λ)
      train one InfoNCE step on (q, pos, negs[q])
      optimizer.step(); scheduler.step()
      if uncertainty == 'ema': EMA-update the teacher

Inner sampler (Algorithm 2, `_mine_queries`):
  stale FAISS top-P
  → filter qrels positives
  → _pool_and_fresh_rerank (current-model cheap rerank, paper line 2-3)
  → top-L by ŝ
  → uncertainty σ via {MC-dropout, EMA teacher–student}
  → g = ŝ + λσ
  → top-m by g

Pick the σ estimator with `--uncertainty {mc_dropout,ema}`.
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
    get_path, get_training_context, load_config, log_startup_config,
    encode_batch, encode_batch_tensor, encode_to_pickle, build_faiss_index,
    _load_qrels, _load_corpus_lookup,
    _pool_and_fresh_rerank, set_seed, evaluate_bright,
)


def _update_ema(student, teacher, alpha):
    """EMA teacher update: θ_t ← α·θ_t + (1-α)·θ_s. Foreach kernels for speed."""
    with torch.no_grad():
        s_params = list(student.parameters())
        t_params = list(teacher.parameters())
        torch._foreach_mul_(t_params, alpha)
        torch._foreach_add_(t_params, s_params, alpha=1.0 - alpha)


def _fresh_shortlists(student, tokenizer, qids, texts, stale_idx, c_ids,
                      corpus_lookup, qrels_dict, P, L, max_pool_per_query,
                      q_max, mc_bs, p_max, device):
    """Shared shell (Algorithm 2 lines 1–3): deterministic student query encode
    → FAISS top-(P+1) → _pool_and_fresh_rerank → per-query top-L shortlist.

    Returns (q_det, batch_query_shortlist, pool_stats, shortlist_texts,
    shortlist_to_idx). `q_det` is exposed so the EMA estimator can reuse it as
    s_cur's query side instead of re-encoding.
    """
    # Deterministic student query encode (eval + no_grad inside encode_batch).
    # _pool_and_fresh_rerank saves/restores model.training, so we don't need to.
    student.eval()
    q_det = encode_batch(student, tokenizer, texts, device, q_max, mc_bs)

    # Algorithm 2 line 1: R(q; I, P+1) \ {d+} — over-retrieve by one so that
    # filtering d+ in _pool_and_fresh_rerank still leaves P FAISS candidates.
    _, indices = stale_idx.search(q_det, P + 1)

    batch_query_shortlist, pool_stats = _pool_and_fresh_rerank(
        student, tokenizer, qids, q_det,
        indices, qrels_dict, c_ids, corpus_lookup,
        p_max, mc_bs, device,
        L, max_pool_per_query,
    )

    shortlist_cand_ids_set = set()
    for qid in qids:
        shortlist_cand_ids_set.update(batch_query_shortlist.get(qid, []))
    shortlist_ids    = list(shortlist_cand_ids_set)
    shortlist_texts  = [corpus_lookup.get(d, "") for d in shortlist_ids]
    shortlist_to_idx = {d: i for i, d in enumerate(shortlist_ids)}

    return q_det, batch_query_shortlist, pool_stats, shortlist_texts, shortlist_to_idx


def _score_mc_dropout(student, tokenizer, texts, shortlist_texts,
                      T, q_max, p_max, mc_bs, device):
    """MC-dropout σ estimator: T stochastic student passes (dropout active),
    σ = std over T, ŝ = mean over T. Returns a score(i, cidxs) -> (s_hat, sigma).

    Post-condition: leaves the student in eval(); the Algorithm 1 caller resets
    to train() before the training encode.
    """
    student.train()
    q_mc = encode_batch(student, tokenizer, texts * T,
                        device, q_max, mc_bs).reshape(T, len(texts), -1)
    c_mc = encode_batch(student, tokenizer, shortlist_texts * T,
                        device, p_max, mc_bs).reshape(T, len(shortlist_texts), -1)
    student.eval()  # restored to a known state; caller resets to train()

    def score(i, cidxs):
        sims  = np.einsum('td,tnd->tn', q_mc[:, i, :], c_mc[:, cidxs, :])
        s_hat = sims.mean(axis=0)
        sigma = sims.std(axis=0)
        return s_hat, sigma

    return score


def _score_ema(student, teacher, tokenizer, texts, shortlist_texts, q_det,
               q_max, p_max, mc_bs, device):
    """EMA teacher–student σ estimator: σ = |s_cur − s_ema|, ŝ = s_cur. Clean
    student + teacher passes (eval + no_grad). Returns score(i, cidxs).

    The student query side reuses `q_det` (already encoded under the same
    eval+no_grad state) instead of re-encoding the query batch.
    """
    if teacher is None:
        raise ValueError("uncertainty='ema' requires a teacher model")
    student.eval()
    q_cur = q_det  # reuse deterministic student query encode (same eval+no_grad state)
    q_ema = encode_batch(teacher, tokenizer, texts, device, q_max, mc_bs)
    c_cur = encode_batch(student, tokenizer, shortlist_texts, device, p_max, mc_bs)
    c_ema = encode_batch(teacher, tokenizer, shortlist_texts, device, p_max, mc_bs)

    def score(i, cidxs):
        s_cur = q_cur[i] @ c_cur[cidxs].T
        s_ema = q_ema[i] @ c_ema[cidxs].T
        sigma = np.abs(s_cur - s_ema)
        return s_cur, sigma

    return score


def _select_and_log_negatives(qids, batch_query_shortlist, shortlist_to_idx,
                              score, pool_stats, m, lv, L):
    """g = ŝ + λσ → top-m by g per query → mined dict + mining-log records.
    Queries with an empty shortlist are absent from the dict (and unlogged).
    """
    mined       = {}
    log_records = []
    for i, qid in enumerate(qids):
        cands = batch_query_shortlist.get(qid, [])
        if not cands:
            continue
        cidxs = [shortlist_to_idx[d] for d in cands]
        s_hat, sigma = score(i, cidxs)
        g = s_hat + lv * sigma

        top_m_idxs = np.argsort(g)[::-1][:m]
        top_m_docs = [cands[k] for k in top_m_idxs]
        mined[qid] = top_m_docs

        selected_docid = top_m_docs[0]
        rank_by_shat   = int(np.argsort(np.argsort(-s_hat))[top_m_idxs[0]])

        stats = pool_stats.get(qid, {})
        log_record = {
            "query_id":                      qid,
            "neg_docid":                     selected_docid,
            "s_hat_selected":                float(s_hat[top_m_idxs[0]]),
            "sigma_selected":                float(sigma[top_m_idxs[0]]),
            "g_selected":                    float(g[top_m_idxs[0]]),
            "rank_by_shat":                  rank_by_shat,
            "sigma_mean_shortlist":          float(sigma.mean()),
            "retrieved_count":               stats.get('retrieved', 0),
            "candidate_pool_count":          stats.get('pool_count', 0),
            "positives_filtered_count":      stats.get('positives_filtered', 0),
            "L":                             L,
            "m":                             m,
            "neg_docids":                    top_m_docs,
            "selected_cheap_rank_zero_based": int(top_m_idxs[0]),
        }
        log_records.append(log_record)

    return mined, log_records


def _mine_queries(student, teacher, tokenizer, qids, qid_to_text,
                  stale_idx, c_ids, corpus_lookup, qrels_dict,
                  cfg, config, device,
                  uncertainty='mc_dropout'):
    """Algorithm 2: mine top-m hard negatives for `qids` with the current model.

    Shared shell (_fresh_shortlists):
      student.eval() query encode → FAISS top-P → _pool_and_fresh_rerank → top-L
    Estimator branch:
      mc_dropout (_score_mc_dropout) — σ = std over T, ŝ = mean over T.
      ema        (_score_ema)        — σ = |s_cur − s_ema|, ŝ = s_cur.
    Selection (_select_and_log_negatives): g = ŝ + λσ → top-m.

    Returns ({qid: [neg_docid_1..neg_docid_m]}, log_records).
    Queries whose pool is empty after positive filtering are absent from the dict.
    """
    P, L = cfg['P'], cfg['L']
    T    = cfg.get('T', 5)
    m, lv = cfg['m'], cfg['lambda_val']
    mc_bs = cfg.get('mc_batch_size', 512)
    max_pool_per_query = cfg.get('max_pool_per_query', P)
    q_max = config['model']['query_max_len']
    p_max = config['model']['passage_max_len']

    texts = [qid_to_text[q] for q in qids]

    q_det, batch_query_shortlist, pool_stats, shortlist_texts, shortlist_to_idx = \
        _fresh_shortlists(student, tokenizer, qids, texts, stale_idx, c_ids,
                          corpus_lookup, qrels_dict, P, L, max_pool_per_query,
                          q_max, mc_bs, p_max, device)

    if not shortlist_texts:
        return {}, []

    if uncertainty == 'mc_dropout':
        score = _score_mc_dropout(student, tokenizer, texts, shortlist_texts,
                                  T, q_max, p_max, mc_bs, device)
    elif uncertainty == 'ema':
        score = _score_ema(student, teacher, tokenizer, texts, shortlist_texts,
                           q_det, q_max, p_max, mc_bs, device)
    else:
        raise ValueError(f"unknown uncertainty estimator: {uncertainty!r}")

    return _select_and_log_negatives(qids, batch_query_shortlist, shortlist_to_idx,
                                     score, pool_stats, m, lv, L)


def run_grass_pipeline(stale_idx, c_ids, corpus_lookup, qrels_dict,
                       cfg, config, ctx, uncertainty, debug=False):
    """Algorithm 1 outer loop: per-minibatch mine → train → step. Returns output dir."""
    # Load training mixture
    mix_dir     = get_path("processed") / "training_mixture"
    train_items = []
    for f_path in sorted(mix_dir.glob("*.jsonl")):
        if f_path.name.startswith('.'):
            continue
        with open(f_path) as f:
            for line in f:
                d   = json.loads(line)
                pos = d.get('positive_passages', [])
                if not pos:
                    continue
                train_items.append({
                    'query_id':  str(d['query_id']),
                    'query':     d['query'],
                    'pos_docid': pos[0]['docid'],
                })
    if debug:
        train_items = train_items[:512]
        print("[GRASS] DEBUG: 512 items", flush=True)
    random.shuffle(train_items)

    qid_to_text = {it['query_id']: it['query'] for it in train_items}
    print(f"[GRASS] {len(train_items)} training examples | "
          f"{len(qid_to_text)} unique queries", flush=True)

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
    ema_alpha     = cfg.get('ema_alpha', 0.999)
    m             = cfg['m']

    output_model_dir = get_path("models") / (cfg['model_name'] + f'_{uncertainty}')
    output_model_dir.mkdir(parents=True, exist_ok=True)

    device    = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # base_model from ctx (get_training_context) for HF-snapshot resolution; must
    # match the model used to build the stale FAISS index in main().
    base_model = ctx['base_model']
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    student   = AutoModel.from_pretrained(base_model,
                                          torch_dtype=torch.bfloat16).to(device)

    teacher = None
    if uncertainty == 'ema':
        teacher = AutoModel.from_pretrained(base_model,
                                            torch_dtype=torch.bfloat16).to(device)
        teacher.eval()
        for p in teacher.parameters():
            p.requires_grad_(False)
        print(f"[GRASS] EMA teacher initialized | alpha={ema_alpha}", flush=True)

    if uncertainty == 'mc_dropout' and mc_dropout_p != 0.1:
        n_layers = sum(1 for mod in student.modules() if isinstance(mod, torch.nn.Dropout))
        for module in student.modules():
            if isinstance(module, torch.nn.Dropout):
                module.p = mc_dropout_p
        print(f"[GRASS] MC-dropout p={mc_dropout_p} on {n_layers} layers", flush=True)

    if _BNB_AVAILABLE:
        optimizer = bnb.optim.AdamW8bit(student.parameters(), lr=lr, weight_decay=weight_decay)
        print("[GRASS] AdamW8bit enabled", flush=True)
    else:
        student.gradient_checkpointing_enable()
        optimizer = AdamW(student.parameters(), lr=lr, weight_decay=weight_decay)
        print("[GRASS] AdamW + gradient checkpointing", flush=True)
    student.train()

    loss_fn      = TemperatureScaledContrastiveLoss(temperature=temperature)
    _student_raw = student  # keep raw handle for save_pretrained (compile wraps the module)
    try:
        torch._dynamo.config.suppress_errors = True
        student = torch.compile(student, dynamic=True)
        print("[GRASS] torch.compile enabled on student", flush=True)
    except Exception as e:
        print(f"[GRASS] torch.compile skipped ({e})", flush=True)

    n_batches    = len(train_items) // batch_size
    total_steps  = n_batches * num_epochs
    warmup_steps = int(warmup_ratio * total_steps)
    scheduler    = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    print(f"[GRASS] Algorithm 1 | uncertainty={uncertainty} | "
          f"batch_size={batch_size} | m={m} | {total_steps} total steps",
          flush=True)

    mining_log_path = output_model_dir / "mining_log.jsonl"
    mining_log_f    = open(mining_log_path, 'w')

    global_step  = 0
    mining_round = 0
    t_start      = time.time()

    for epoch in range(num_epochs):
        random.shuffle(train_items)
        epoch_loss  = 0.0
        epoch_steps = 0

        for b in range(n_batches):
            batch_items = train_items[b * batch_size:(b + 1) * batch_size]
            batch_qids = list(dict.fromkeys(it['query_id'] for it in batch_items))
            if not batch_qids:
                continue

            mining_round += 1
            mined, log_records = _mine_queries(
                student, teacher, tokenizer, batch_qids, qid_to_text,
                stale_idx, c_ids, corpus_lookup, qrels_dict,
                cfg, config, device,
                uncertainty=uncertainty,
            )
            for rec in log_records:
                mining_log_f.write(json.dumps(rec, ensure_ascii=False) + '\n')

            queries, positives, negatives = [], [], []
            for it in batch_items:
                negs = mined.get(it['query_id'])
                if not negs or len(negs) < m:
                    continue
                queries.append(it['query'])
                positives.append(corpus_lookup.get(it['pos_docid'], ''))
                negatives.append([corpus_lookup.get(d, '') for d in negs[:m]])
            if not queries:
                continue

            student.train()
            q_embs  = encode_batch_tensor(student, tokenizer, queries,
                                          device, q_max_len, mc_batch_size,
                                          requires_grad=True)
            d_texts = [t for pos, negs in zip(positives, negatives) for t in [pos] + negs]
            d_embs  = encode_batch_tensor(student, tokenizer, d_texts,
                                          device, p_max_len, mc_batch_size,
                                          requires_grad=True)
            loss = loss_fn(q_embs, d_embs)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            clip_grad_norm_(student.parameters(), max_grad_norm)
            optimizer.step()
            scheduler.step()

            if uncertainty == 'ema':
                _update_ema(_student_raw, teacher, ema_alpha)

            epoch_loss  += loss.item()
            epoch_steps += 1
            global_step += 1

            if global_step % logging_steps == 0:
                elapsed   = time.time() - t_start
                secs_per  = elapsed / global_step
                remaining = secs_per * (total_steps - global_step)
                eta       = f"{int(remaining // 3600)}h {int((remaining % 3600) // 60)}m"
                print(f"[GRASS] Ep{epoch+1} step {b+1}/{n_batches} | "
                      f"loss={loss.item():.4f} | mined={len(mined)}/{len(batch_qids)} "
                      f"| ETA {eta}",
                      flush=True)

            if global_step % save_steps == 0:
                ckpt = output_model_dir / f"checkpoint-{global_step}"
                ckpt.mkdir(parents=True, exist_ok=True)
                _student_raw.save_pretrained(str(ckpt))
                tokenizer.save_pretrained(str(ckpt))
                torch.save(scheduler.state_dict(), ckpt / "scheduler.pt")
                torch.save(optimizer.state_dict(), ckpt / "optimizer.pt")
                print(f"[GRASS] Checkpoint saved: {ckpt.name}", flush=True)

        avg = epoch_loss / max(1, epoch_steps)
        print(f"[GRASS] Epoch {epoch+1} done. avg_loss={avg:.4f} | "
              f"trained_batches={epoch_steps}/{n_batches}",
              flush=True)

    _student_raw.save_pretrained(str(output_model_dir))
    tokenizer.save_pretrained(str(output_model_dir))
    mining_log_f.close()
    print(f"[GRASS] Training complete. Model at {output_model_dir}", flush=True)

    del student, _student_raw
    if teacher is not None:
        del teacher
    gc.collect()
    torch.cuda.empty_cache()

    evaluate_bright(ctx, config, output_model_dir, temp_workdir_key='temp_grass')
    return output_model_dir


def main():
    """Standalone entry point. Sets up shared state, then runs Algorithm 1."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--recipe',       default='grass')
    parser.add_argument('--uncertainty',  choices=['mc_dropout', 'ema'],
                        default='mc_dropout',
                        help='Algorithm 2 σ estimator')
    parser.add_argument('--model_suffix', type=str, default=None)
    parser.add_argument('--num_epochs',   type=int, default=None)
    parser.add_argument('--P',            type=int,   default=None, help='override cfg.P (pool size)')
    parser.add_argument('--L',            type=int,   default=None, help='override cfg.L (shortlist)')
    parser.add_argument('--lambda_val',   type=float, default=None, help='override cfg.lambda_val (σ weight)')
    parser.add_argument('--debug',        action='store_true')
    args = parser.parse_args()

    config = load_config()
    cfg    = config['training'][args.recipe]
    ctx    = get_training_context(args.recipe)
    set_seed(config.get('seed', 42))

    if args.num_epochs is not None:
        cfg = {**cfg, 'num_epochs': args.num_epochs}
    if args.P is not None:
        cfg = {**cfg, 'P': args.P}
    if args.L is not None:
        cfg = {**cfg, 'L': args.L}
    if args.lambda_val is not None:
        cfg = {**cfg, 'lambda_val': args.lambda_val}
    if args.model_suffix is not None:
        cfg = {**cfg, 'model_name': cfg['model_name'] + '_' + args.model_suffix}

    # after the CLI overrides, so the block reports what will actually run
    log_startup_config(args.recipe, ctx, cfg)

    from data.preprocessor import run_setup
    corpus_file, query_file, qrels_file = run_setup()

    workdir   = get_path("temp_grass")
    workdir.mkdir(exist_ok=True, parents=True)
    stale_pkl = workdir / "stale_index" / "corpus.pkl"
    stale_pkl.parent.mkdir(exist_ok=True)
    if not stale_pkl.exists():
        print("[GRASS] Building stale ANN index...", flush=True)
        # Same base_model source as the in-process encoder (run_grass_pipeline)
        # so the stale index and the query/pool encoder share one checkpoint.
        encode_to_pickle(ctx['base_model'], corpus_file, stale_pkl, False, ctx, config)
    print(f"[GRASS] Stale index ready: {stale_pkl}", flush=True)

    stale_idx, stale_embs, c_ids = build_faiss_index(stale_pkl)
    corpus_lookup = _load_corpus_lookup(corpus_file)
    qrels_dict    = _load_qrels(qrels_file)

    run_grass_pipeline(stale_idx, c_ids, corpus_lookup, qrels_dict,
                       cfg, config, ctx,
                       uncertainty=args.uncertainty,
                       debug=args.debug)


if __name__ == "__main__":
    main()
