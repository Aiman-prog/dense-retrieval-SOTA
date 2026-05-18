"""
GRASS Async Miner — runs on GPU 1 in parallel with the trainer.
Polls output_model_dir for new checkpoints → MC-dropout uncertainty mining → writes neg updates.
Signals trainer via neg_update_dir/update_{N}.jsonl + ready_{N} marker (ready written last).
"""
import gc
import json
import os
import sys
import time
import random
import argparse
import numpy as np
import torch
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from transformers import AutoTokenizer, AutoModel
from transformers.trainer_utils import get_last_checkpoint

project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root / 'src'))

from utils.helpers import (
    get_path, get_training_context, load_config, build_faiss_index,
    is_valid_checkpoint, get_latest_marker_no,
    _load_qrels, _load_corpus_lookup, encode_batch, _shortlist_batch,
)
from utils.bandit import CaseBandit


def _write_update(mined, neg_update_dir, update_num):
    """Write update_{N}.jsonl then ready_{N} marker (validity gate)."""
    jsonl_path = neg_update_dir / f"update_{update_num}.jsonl"
    with open(jsonl_path, 'w') as f:
        for qid, neg_docids in mined.items():
            if neg_docids:
                f.write(json.dumps({'query_id': qid, 'neg_docid': neg_docids[0]},
                                   ensure_ascii=False) + '\n')
    (neg_update_dir / f"ready_{update_num}").write_text(str(update_num))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_model_dir', required=True)
    parser.add_argument('--neg_update_dir',   required=True)
    parser.add_argument('--stale_pkl',        required=True)
    parser.add_argument('--query_file',       required=True)
    parser.add_argument('--corpus_file',      required=True)
    parser.add_argument('--qrels_file',       required=True)
    parser.add_argument('--base_model',       required=True)
    parser.add_argument('--recipe',           default='grass')
    parser.add_argument('--selection',        default='bandit', choices=['bandit', 'random'])
    parser.add_argument('--n_das',            type=int, default=None)
    args = parser.parse_args()

    config = load_config()
    cfg    = config['training'][args.recipe]

    P             = cfg['P']
    L             = cfg['L']
    m             = cfg['m']
    T             = cfg.get('T', 3)
    lambda_val    = cfg['lambda_val']
    mc_batch_size = cfg.get('mc_batch_size', 256)
    mc_dropout_p  = cfg.get('mc_dropout_p', 0.1)
    poll_interval = cfg.get('miner_poll_interval', 5)
    n_das         = args.n_das or cfg.get('mab_n_das', 5)
    bandit_eps       = cfg.get('bandit_epsilon', 0.2)
    bandit_eps_start = cfg.get('bandit_epsilon_start', 0.8)
    q_max_len     = config['model']['query_max_len']
    p_max_len     = config['model']['passage_max_len']

    neg_update_dir = Path(args.neg_update_dir)
    neg_update_dir.mkdir(exist_ok=True, parents=True)
    Path(args.output_model_dir).mkdir(exist_ok=True, parents=True)

    print("[Miner] Loading qrels + corpus...", flush=True)
    qrels_dict    = _load_qrels(args.qrels_file)
    corpus_lookup = _load_corpus_lookup(args.corpus_file)

    print("[Miner] Loading queries...", flush=True)
    import pandas as pd
    query_df         = pd.read_json(args.query_file, lines=True)
    all_query_ids    = query_df['query_id'].astype(str).tolist()
    qid_to_text      = {str(r['query_id']): r['query'] for _, r in query_df.iterrows()}
    print(f"[Miner] {len(all_query_ids)} queries loaded", flush=True)

    print("[Miner] Building stale FAISS index...", flush=True)
    stale_idx, stale_embs, c_ids = build_faiss_index(args.stale_pkl)
    c_id_to_idx = {did: i for i, did in enumerate(c_ids)}
    print(f"[Miner] Stale index: {len(c_ids)} passages", flush=True)

    device    = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    model     = AutoModel.from_pretrained(args.base_model, torch_dtype=torch.bfloat16).to(device)
    model.train()
    if mc_dropout_p != 0.1:
        n_layers = 0
        for module in model.modules():
            if isinstance(module, torch.nn.Dropout):
                module.p = mc_dropout_p
                n_layers += 1
        print(f"[Miner] MC-dropout p={mc_dropout_p} on {n_layers} layers", flush=True)

    if device.type == 'cuda':
        try:
            model = torch.compile(model)
            print("[Miner] torch.compile enabled", flush=True)
        except Exception as e:
            print(f"[Miner] torch.compile skipped ({e})", flush=True)

    bandit = None
    if args.selection == 'bandit':
        one_sweep = max(1, len(all_query_ids) // n_das)
        bandit = CaseBandit(
            n_das=n_das, epsilon=bandit_eps,
            epsilon_start=bandit_eps_start,
            decay_cycles=one_sweep,
            stale_cycles=one_sweep,
        )
        bandit.init_all_queries(all_query_ids)
        print(
            f"[Miner] CaseBandit ready: n_das={n_das}, "
            f"ε {bandit_eps_start}→{bandit_eps} over {one_sweep} cycles, "
            f"stale_cycles={one_sweep}",
            flush=True,
        )

    cpu_exec       = ThreadPoolExecutor(max_workers=1)
    last_ckpt      = None
    update_num     = get_latest_marker_no(neg_update_dir, prefix="ready_") + 1
    print(f"[Miner] Loop started (T={T}, P={P}, L={L}, selection={args.selection})", flush=True)

    while True:
        # 1. Reload model if trainer wrote a new valid checkpoint
        next_ckpt = get_last_checkpoint(args.output_model_dir)
        if next_ckpt and next_ckpt != last_ckpt and is_valid_checkpoint(next_ckpt):
            print(f"[Miner] Loading checkpoint: {Path(next_ckpt).name}", flush=True)
            old_model = model
            model = AutoModel.from_pretrained(next_ckpt, torch_dtype=torch.bfloat16).to(device)
            model.train()
            if mc_dropout_p != 0.1:
                for module in model.modules():
                    if isinstance(module, torch.nn.Dropout):
                        module.p = mc_dropout_p
            if device.type == 'cuda':
                try:
                    model = torch.compile(model)
                except Exception:
                    pass
            del old_model
            gc.collect()
            torch.cuda.empty_cache()
            last_ckpt = next_ckpt

        # 2. Select queries to mine
        if bandit is not None:
            selected_ids = bandit.select_global(n_das=n_das)
        else:
            selected_ids = random.sample(all_query_ids, min(n_das, len(all_query_ids)))

        selected_ids   = [qid for qid in selected_ids if qid in qid_to_text]
        selected_texts = [qid_to_text[qid] for qid in selected_ids]
        if not selected_ids:
            time.sleep(poll_interval)
            continue

        # 3. Deterministic encode for ANN search + cheap shortlisting score
        model.eval()
        q_embs_det = encode_batch(model, tokenizer, selected_texts, device, q_max_len, mc_batch_size)
        model.train()

        _, indices = stale_idx.search(q_embs_det, P)

        # CPU shortlisting in background while GPU does T MC encodes
        shortlist_fut = cpu_exec.submit(
            _shortlist_batch, selected_ids, indices, q_embs_det, qrels_dict,
            c_ids, c_id_to_idx, stale_embs, corpus_lookup, P, L
        )

        # Vectorized T MC-dropout encodes (T copies in one batch → reshape)
        q_flat  = encode_batch(model, tokenizer, selected_texts * T, device, q_max_len, mc_batch_size)
        q_stack = q_flat.reshape(T, len(selected_texts), -1)  # (T, B, dim)

        batch_sl, sl_ids, sl_texts, sl_to_idx, _ = shortlist_fut.result()

        if sl_texts:
            c_flat  = encode_batch(model, tokenizer, sl_texts * T, device, p_max_len, mc_batch_size)
            c_stack = c_flat.reshape(T, len(sl_texts), -1)  # (T, N_sl, dim)
        else:
            c_stack = None

        # 4. Score and select top-m negatives
        mined   = {}
        sigmas  = []   # top-candidate σ per query this cycle
        for i, qid in enumerate(selected_ids):
            cands = batch_sl.get(qid, [])
            if not cands or c_stack is None:
                continue
            cidxs = [sl_to_idx[d] for d in cands]
            q_i   = q_stack[:, i, :]           # (T, dim)
            c_i   = c_stack[:, cidxs, :]       # (T, n_cands, dim)
            sims  = np.einsum('td,tnd->tn', q_i, c_i)  # (T, n_cands)
            s_hat = sims.mean(axis=0)
            sigma = sims.std(axis=0)
            g     = s_hat + lambda_val * sigma
            top_m = np.argsort(g)[::-1][:m]
            mined[qid] = [cands[k] for k in top_m]
            top_sigma = float(sigma[top_m[0]])
            top_g     = float(g[top_m[0]])
            sigmas.append(top_sigma)
            if bandit is not None:
                bandit.update(qid, top_g)

        # 5. Write update JSONL + ready marker
        if mined:
            _write_update(mined, neg_update_dir, update_num)

            # Diagnostic: σ distribution for this cycle
            if sigmas:
                arr = np.array(sigmas)
                n_exploit = int(n_das * (1 - bandit._current_epsilon())) if bandit else n_das
                n_explore = len(selected_ids) - n_exploit
                print(
                    f"[Miner] #{update_num} | queries={len(mined)} | "
                    f"σ mean={arr.mean():.4f} std={arr.std():.4f} "
                    f"min={arr.min():.4f} max={arr.max():.4f} | "
                    f"exploit={n_exploit} explore={n_explore}" +
                    (f" | J_t={len(bandit.J_t)} explore_pool={len(bandit.unseen)}"
                     f" ε={bandit._current_epsilon():.2f}"
                     if bandit else ""),
                    flush=True,
                )
            else:
                print(f"[Miner] #{update_num}: no shortlist candidates (shortlist empty)",
                      flush=True)

            update_num += 1


if __name__ == "__main__":
    main()
