"""CASE-Lite challenger sampler.

Implements §6 of grass-report.tex: bucket-UCB challenger allocation over
a per-query incumbent + small challenger set, with a margin-violation
reward against the positive document.

Parallel to grass_sampler in run_grass_mcd.py. Reuses encode_batch and
_shortlist_batch from src/utils/helpers.py. Designed to be called from
run_grass_async_v2_miner.py when --case_lite_enabled is on.
"""

import gc
import json
import random
import sys
import time
import numpy as np
import torch
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from transformers import AutoTokenizer, AutoModel
from tevatron.retriever.modeling import DenseModel

project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root / 'src'))

from utils.helpers import encode_batch, _shortlist_batch

# Tevatron Bug Patch
if not hasattr(DenseModel, "_keys_to_ignore_on_save"):
    setattr(DenseModel, "_keys_to_ignore_on_save", None)


def _pick_e_t(qid, shortlist_ids, shortlist_to_cheap_rank, corpus_lookup,
              incumbent_docid, slots, bandit):
    """Build E_t(q) = {incumbent} ∪ challengers.

    Returns a list of dicts: [{docid, text, cheap_rank, bucket, is_incumbent}, ...].
    """
    # Pool candidates per bucket from shortlist (1-indexed ranks).
    per_bucket = [[] for _ in range(bandit.n_buckets)]
    for d in shortlist_ids:
        rank = shortlist_to_cheap_rank[d]  # 1-indexed
        per_bucket[bandit.bucket_of(rank)].append(d)

    # Bootstrap incumbent if missing: cheap-rank-1 of S(q).
    if incumbent_docid is None or incumbent_docid not in corpus_lookup:
        incumbent_docid = shortlist_ids[0] if shortlist_ids else None
    if incumbent_docid is None:
        return []

    chosen = set([incumbent_docid])
    challengers = []
    target_total = sum(slots)  # K - 1

    def _draw(b, n_slot):
        """Draw up to n_slot challengers from bucket b, deterministic in b==0, random elsewhere."""
        pool = [d for d in per_bucket[b] if d not in chosen]
        if not pool or n_slot <= 0:
            return []
        if b == 0:
            pool_sorted = sorted(pool, key=lambda d: shortlist_to_cheap_rank[d])
            return pool_sorted[:n_slot]
        return random.sample(pool, min(n_slot, len(pool)))

    # First pass: per-bucket allocation per the slot prior / Bucket-UCB split.
    for b in range(bandit.n_buckets):
        for d in _draw(b, slots[b]):
            chosen.add(d)
            challengers.append(d)

    # Backfill: if some buckets ran short of candidates, greedily fill from any
    # bucket with remaining pool until we hit target_total. Preserves nominal K.
    if len(challengers) < target_total:
        for b in range(bandit.n_buckets):
            need = target_total - len(challengers)
            if need <= 0:
                break
            for d in _draw(b, need):
                chosen.add(d)
                challengers.append(d)

    E_t = []
    # Incumbent first (so we can check is_incumbent via index 0 later).
    inc_rank = shortlist_to_cheap_rank.get(incumbent_docid)
    if inc_rank is None:
        # Incumbent isn't in current shortlist; assign synthetic rank > L_mem so it falls in last bucket.
        inc_rank = max(shortlist_to_cheap_rank.values(), default=0) + 1
    E_t.append({
        'docid':        incumbent_docid,
        'text':         corpus_lookup[incumbent_docid],
        'cheap_rank':   inc_rank,
        'bucket':       bandit.bucket_of(inc_rank),
        'is_incumbent': True,
    })
    for d in challengers:
        E_t.append({
            'docid':        d,
            'text':         corpus_lookup[d],
            'cheap_rank':   shortlist_to_cheap_rank[d],
            'bucket':       bandit.bucket_of(shortlist_to_cheap_rank[d]),
            'is_incumbent': False,
        })
    return E_t


def case_lite_sampler(model_path, stale_idx, stale_embs, c_id_to_idx, c_ids,
                      corpus_lookup, mix_df, qrels_dict, cfg, config, out_dir,
                      case_lite_bandit, round_idx, base_jsonl_dir=None):
    """CASE-Lite per-query candidate-level mining (§6).

    Returns:
      round_rewards: {bucket_idx: [r, ...]} — fed to case_lite_bandit.update_round().
    """
    cl_cfg = cfg['async_v2']['case_lite']
    P                = int(cl_cfg.get('P', cfg.get('P', 50)))
    L_mem            = int(cl_cfg['L_mem'])
    K                = int(cl_cfg['K'])
    T                = int(cfg['T'])
    lambda_val       = float(cfg['lambda_val'])
    mc_batch_size    = int(cfg.get('mc_batch_size', 256))
    query_batch_size = int(cfg.get('query_batch_size', 64))
    q_max_len        = config['model']['query_max_len']
    p_max_len        = config['model']['passage_max_len']

    device    = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model     = AutoModel.from_pretrained(model_path, torch_dtype=torch.bfloat16).to(device)
    mc_dropout_p = cfg.get('mc_dropout_p', 0.1)
    if mc_dropout_p != 0.1:
        for module in model.modules():
            if isinstance(module, torch.nn.Dropout):
                module.p = mc_dropout_p

    print(f"  Loaded model for CaseLiteSampler "
          f"(T={T}, P={P}, L_mem={L_mem}, K={K}, λ={lambda_val}, "
          f"γ={case_lite_bandit.gamma}, τ={case_lite_bandit.tau})", flush=True)

    query_ids   = mix_df['query_id'].astype(str).tolist()
    query_texts = mix_df['query'].tolist()
    n_queries   = len(query_ids)
    n_batches   = (n_queries + query_batch_size - 1) // query_batch_size
    print(f"  Processing {n_queries} queries in {n_batches} batches "
          f"(batch_size={query_batch_size}, round={round_idx})...", flush=True)

    # Pre-compute round-1 slots once (round_idx-conditioned; same for every query in this round).
    slots = case_lite_bandit.allocate_slots(K, round_idx)
    print(f"  Bucket slots this round: {slots} (sum={sum(slots)} = K-1)", flush=True)

    mined_negs        = {}
    round_rewards     = {b: [] for b in range(case_lite_bandit.n_buckets)}
    n_promoted        = 0
    n_kept            = 0
    t_loop_start      = time.time()

    log_path = out_dir / "mining_log.jsonl"
    log_path.parent.mkdir(exist_ok=True, parents=True)
    log_f = open(log_path, 'w')

    cpu_exec = ThreadPoolExecutor(max_workers=1)

    for b, batch_start in enumerate(range(0, n_queries, query_batch_size)):
        batch_ids   = query_ids[batch_start:batch_start + query_batch_size]
        batch_texts = query_texts[batch_start:batch_start + query_batch_size]

        # Deterministic query encoding (eval mode) for FAISS + s_det.
        model.eval()
        q_det = encode_batch(model, tokenizer, batch_texts, device, q_max_len, mc_batch_size)
        model.train()

        _, indices = stale_idx.search(q_det, P)
        shortlist_fut = cpu_exec.submit(
            _shortlist_batch, batch_ids, indices, q_det, qrels_dict,
            c_ids, c_id_to_idx, stale_embs, corpus_lookup, P, L_mem
        )

        # MC-mode query encoding (T passes), vectorized.
        q_mc_flat  = encode_batch(model, tokenizer, batch_texts * T, device, q_max_len, mc_batch_size)
        q_mc_stack = q_mc_flat.reshape(T, len(batch_texts), -1)  # (T, B, dim)

        batch_query_shortlist, shortlist_ids, shortlist_texts, shortlist_to_idx, n_filtered = \
            shortlist_fut.result()

        # Build per-query E_t(q) and gather positives.
        per_q_Et       = []  # parallel to batch_ids; entries are lists or None
        positives      = []  # parallel; (pos_docid, pos_text) or None
        for i, qid in enumerate(batch_ids):
            shortlist = batch_query_shortlist.get(qid, [])
            if not shortlist:
                per_q_Et.append(None)
                positives.append(None)
                continue
            # cheap_rank: 1-indexed position in this query's shortlist.
            cheap_rank_map = {d: rank + 1 for rank, d in enumerate(shortlist)}
            E_t = _pick_e_t(qid, shortlist, cheap_rank_map, corpus_lookup,
                            case_lite_bandit.incumbent.get(qid), slots, case_lite_bandit)
            if not E_t:
                per_q_Et.append(None)
                positives.append(None)
                continue
            per_q_Et.append(E_t)

            pos_set = qrels_dict.get(qid, set())
            if pos_set:
                pos_docid = sorted(pos_set)[0]
                pos_text  = corpus_lookup.get(pos_docid)
                if pos_text is not None:
                    positives.append((pos_docid, pos_text))
                else:
                    positives.append(None)
            else:
                positives.append(None)

        # Collect unique docs across batch for encoding. Maintain stable ordering.
        eval_doc_texts = []
        eval_doc_idx   = {}  # docid -> idx in eval_doc_texts
        mc_doc_texts   = []
        mc_doc_idx     = {}
        for E_t, pos in zip(per_q_Et, positives):
            if E_t is None:
                continue
            for cand in E_t:
                did = cand['docid']
                if did not in eval_doc_idx:
                    eval_doc_idx[did] = len(eval_doc_texts)
                    eval_doc_texts.append(cand['text'])
                if did not in mc_doc_idx:
                    mc_doc_idx[did] = len(mc_doc_texts)
                    mc_doc_texts.append(cand['text'])
            if pos is not None:
                pos_did, pos_text = pos
                if pos_did not in eval_doc_idx:
                    eval_doc_idx[pos_did] = len(eval_doc_texts)
                    eval_doc_texts.append(pos_text)

        if not eval_doc_texts:
            continue

        # Encode docs: eval mode (for s_det, including positives) + MC mode (for σ, no positives).
        model.eval()
        eval_doc_embs = encode_batch(model, tokenizer, eval_doc_texts,
                                     device, p_max_len, mc_batch_size)  # (N_eval, dim)
        model.train()

        if mc_doc_texts:
            mc_doc_flat  = encode_batch(model, tokenizer, mc_doc_texts * T,
                                        device, p_max_len, mc_batch_size)
            mc_doc_stack = mc_doc_flat.reshape(T, len(mc_doc_texts), -1)  # (T, N_mc, dim)
        else:
            mc_doc_stack = None

        # Per-query scoring + promotion + reward.
        for i, qid in enumerate(batch_ids):
            E_t = per_q_Et[i]
            if not E_t or mc_doc_stack is None:
                continue
            pos = positives[i]

            # Vectorised g per candidate across E_t.
            cand_eval_idxs = [eval_doc_idx[c['docid']] for c in E_t]
            cand_mc_idxs   = [mc_doc_idx[c['docid']]   for c in E_t]

            q_det_i = q_det[i]                    # (dim,)
            q_mc_i  = q_mc_stack[:, i, :]         # (T, dim)
            c_det_E = eval_doc_embs[cand_eval_idxs]      # (|E_t|, dim)
            c_mc_E  = mc_doc_stack[:, cand_mc_idxs, :]   # (T, |E_t|, dim)

            s_det_E = c_det_E @ q_det_i                                  # (|E_t|,)
            sims    = np.einsum('td,tnd->tn', q_mc_i, c_mc_E)            # (T, |E_t|)
            s_hat   = sims.mean(axis=0)
            sigma   = sims.std(axis=0)
            g       = s_hat + lambda_val * sigma                          # (|E_t|,)

            # Promotion: B_t = g(d) - g(d_inc) + w_t(d).
            g_inc = g[0]  # incumbent is index 0 by construction
            N_total = sum(case_lite_bandit.N_b)
            w_t = np.array([
                case_lite_bandit.beta * np.sqrt(np.log(1 + N_total) / (1 + case_lite_bandit.N_b[c['bucket']]))
                for c in E_t
            ])
            B_t = g - g_inc + w_t   # B_t[0] == w_t[0] (challenger comparison only meaningful for i>=1)

            chosen_idx = 0
            if len(E_t) > 1:
                challenger_B = B_t[1:]
                max_idx_rel  = int(np.argmax(challenger_B))
                if challenger_B[max_idx_rel] > case_lite_bandit.tau:
                    chosen_idx = max_idx_rel + 1
                    n_promoted += 1
                    case_lite_bandit.incumbent[qid] = E_t[chosen_idx]['docid']
                else:
                    n_kept += 1
            mined_negs[qid] = [E_t[chosen_idx]['docid']]

            # Margin-violation reward per evaluated candidate.
            if pos is not None:
                pos_idx   = eval_doc_idx[pos[0]]
                s_det_pos = float(eval_doc_embs[pos_idx] @ q_det_i)
            else:
                s_det_pos = None

            for k, cand in enumerate(E_t):
                if s_det_pos is not None:
                    r_margin = max(0.0, float(s_det_E[k]) - s_det_pos + case_lite_bandit.gamma)
                    round_rewards[cand['bucket']].append(r_margin)
                else:
                    r_margin = None

                log_f.write(json.dumps({
                    'query_id':     qid,
                    'docid':        cand['docid'],
                    'cheap_rank':   cand['cheap_rank'],
                    'bucket':       cand['bucket'],
                    'is_incumbent': cand['is_incumbent'],
                    'chosen':       (k == chosen_idx),
                    'g_selected':   float(g[k]),
                    's_hat':        float(s_hat[k]),
                    'sigma':        float(sigma[k]),
                    's_det':        float(s_det_E[k]),
                    'r_margin':     r_margin,
                    'promoted':     (k == chosen_idx and not cand['is_incumbent']),
                }, ensure_ascii=False) + '\n')

        if b < 3 or (b + 1) % 100 == 0:
            elapsed   = time.time() - t_loop_start
            secs_per  = elapsed / (b + 1)
            remaining = secs_per * (n_batches - b - 1)
            eta       = f"{int(remaining // 3600)}h {int((remaining % 3600) // 60)}m"
            print(f"  Batch {b+1}/{n_batches} | ETA {eta} | "
                  f"promoted {n_promoted} / kept {n_kept}", flush=True)

    log_f.close()
    cpu_exec.shutdown(wait=False)
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Write updated mixture files (only negative_passages changes).
    out_dir.mkdir(exist_ok=True, parents=True)
    from utils.helpers import get_path
    base_dir = base_jsonl_dir if base_jsonl_dir is not None else (get_path("processed") / "training_mixture")
    for f_path in base_dir.glob("*.jsonl"):
        if f_path.name.startswith('.'):
            continue
        with open(f_path, 'r') as f_in, open(out_dir / f_path.name, 'w') as f_out:
            for line in f_in:
                d   = json.loads(line)
                qid = str(d['query_id'])
                if qid in mined_negs:
                    d['negative_passages'] = [
                        {'docid': neg_id, 'text': corpus_lookup.get(neg_id, '')}
                        for neg_id in mined_negs[qid]
                    ]
                f_out.write(json.dumps(d, ensure_ascii=False) + '\n')

    print(f"  CaseLiteSampler done. Promoted {n_promoted}, kept {n_kept}, "
          f"updated {len(mined_negs)} queries. Round rewards per bucket: "
          f"{[len(round_rewards[b]) for b in range(case_lite_bandit.n_buckets)]}",
          flush=True)
    return round_rewards
