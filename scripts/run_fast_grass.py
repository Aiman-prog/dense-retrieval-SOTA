"""
Fast-GRASS — Algorithm 1 trainer over a bounded global negative cache.

Same outer loop as GRASS (mine -> fresh-encode selected pos+negs -> InfoNCE step ->
optimizer/scheduler step -> [EMA-update teacher]), but mining no longer does a per-query
stale FAISS top-P + fresh rerank. Instead it selects ``m`` negatives per query against
ONE bounded global cache ``H`` (size ``B_doc``) of stale doc states (Mining Step in
fast_grass_negative_cache_architecture.md), via ``--uncertainty``:

  ema  — student+teacher cached states; g = s_stu + λ|s_stu − s_tea| over all H.
  mcdp — TEACHER-FREE (default): cheap deterministic student score over all H → top-L,
         then full query/document MC-dropout (T passes) σ on that top-L shortlist.

The cache is kept fresh by amortized maintenance every ``cache_update_interval`` steps.
The fresh-loss encode of selected pos+negs is IDENTICAL to GRASS — the cache only chooses
which docs to encode. Gradients never flow through the selection-only cache ``Z_H``.

  _mine_batch{,_ema,_mcdp} — estimator dispatcher + the two miners (analog of
                             run_grass._mine_queries / _score_ema / _score_mc_dropout)
  _selection_diag          — shared per-step σ-testability + cost diagnostics
  run_fast_grass_pipeline  — Algorithm 1 loop (unit-testable; mirrors run_grass_pipeline)
  _build_fast_grass_cfg    — runtime cfg with CLI overrides + derived step counts
  main                     — cluster entry point (stale index reuse, cache init, dispatch)

Reuses run_grass._update_ema and the GRASS optimizer/compile/eval scaffolding by import;
does NOT edit run_grass.py / helpers.py / negative_cache.py.
"""
import gc
import json
import random
import time
import sys
import argparse
from collections import deque
from pathlib import Path

import numpy as np
import torch
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
sys.path.append(str(project_root / 'scripts'))

from utils.helpers import (
    get_path, get_training_context, load_config, log_startup_config,
    encode_batch_tensor, encode_to_pickle, build_faiss_index,
    _load_qrels, _load_corpus_lookup, set_seed, evaluate_bright,
)
from utils.negative_cache import NegativeCache
from run_grass import _update_ema  # module-level EMA teacher update (run_grass.py:56)


# ---- mining: the Fast-GRASS analog of run_grass._mine_queries --------------
#
# One dispatcher (`_mine_batch`) over two estimators that share the training loop:
#   EMA  (`_mine_batch_ema`)  — student+teacher cached states, σ=|s_stu−s_tea| over H.
#   MCDP (`_mine_batch_mcdp`) — TEACHER-FREE: cheap deterministic top-L inside H,
#                               then full query/document MC-dropout σ on that top-L.
# Both return `(mined, slots, q_student, q_teacher, stats)` (MCDP's q_teacher is None).


def _selection_diag(s_hat_full, sigma_full, g_masked, s_hat_masked, slots, lam,
                    flat_docids):
    """Shared per-step σ-testability diagnostics (both estimators).

    ``flip_rate_vs_lambda0``: top-1 by g (= ŝ+λσ, masked) vs top-1 by ŝ alone
    (same mask). Nonzero ⇒ σ moves the pick; ~0 ⇒ uncertainty is inert at this
    λ/estimator. All inputs are ``(B, B_doc)``; ``slots`` is ``(B, m)``.
    """
    with torch.no_grad():
        sel_s_hat = float(s_hat_full.gather(1, slots).float().mean())
        sel_sigma = float(sigma_full.gather(1, slots).float().mean())
        flip = float((g_masked.argmax(dim=1) != s_hat_masked.argmax(dim=1))
                     .float().mean())
    return {
        'sel_s_hat_mean': sel_s_hat,
        'sel_sigma_mean': sel_sigma,
        'sel_lambda_sigma_mean': lam * sel_sigma,
        'sel_sigma_over_s_hat': (sel_sigma / sel_s_hat) if sel_s_hat else 0.0,
        'flip_rate_vs_lambda0': flip,
        'selected_doc_diversity': (len(set(flat_docids)) / len(flat_docids))
                                  if flat_docids else 0.0,
    }


def _mine_batch_ema(cache, student, teacher, tokenizer, batch_qids, qid_to_text,
                    qrels_dict, fg_cfg, device):
    """EMA miner: score the batch against all of ``Z_H`` (one matmul,
    σ=|s_stu−s_tea|), mask positives, select. No per-query FAISS/rerank. The
    selection query embeddings are returned for the recert reservoir (no extra
    encode). Returns ``(mined, slots, q_student, q_teacher, stats)``.
    """
    q_max = fg_cfg['query_max_len']
    bs = fg_cfg.get('mc_batch_size', 256)
    texts = [qid_to_text[q] for q in batch_qids]

    # Deterministic selection encode: eval (no dropout) + no_grad; restore modes.
    student_was_training = student.training
    teacher_was_training = teacher.training
    student.eval()
    teacher.eval()
    try:
        q_student = encode_batch_tensor(student, tokenizer, texts, device, q_max,
                                        bs, requires_grad=False).to(cache.Z_student.dtype)
        q_teacher = encode_batch_tensor(teacher, tokenizer, texts, device, q_max,
                                        bs, requires_grad=False).to(cache.Z_teacher.dtype)
    finally:
        student.train(student_was_training)
        teacher.train(teacher_was_training)

    g, s_hat, sigma = cache.score(q_student, q_teacher, fg_cfg['lambda_val'])
    g = cache.mask_positives(g, batch_qids, qrels_dict, inplace=True)
    slots, neg_docids = cache.select(g, m=fg_cfg['m'], mode=fg_cfg['selection_mode'],
                                     beta=fg_cfg['beta'], L=fg_cfg['L'])
    cache.record_selection(slots)

    mined = {qid: neg_docids[i] for i, qid in enumerate(batch_qids)}
    flat = [d for row in neg_docids for d in row]
    lam = float(fg_cfg['lambda_val'])
    s_hat_masked = cache.mask_positives(s_hat.clone(), batch_qids, qrels_dict,
                                        inplace=True)
    stats = {
        's_hat_mean': float(s_hat.float().mean()),
        'sigma_mean': float(sigma.float().mean()),
        **_selection_diag(s_hat, sigma, g, s_hat_masked, slots, lam, flat),
    }
    return mined, slots, q_student, q_teacher, stats


def _mine_batch_mcdp(cache, student, tokenizer, batch_qids, qid_to_text,
                     corpus_lookup, qrels_dict, fg_cfg, device):
    """Teacher-free MCDP miner (lazy top-L, full query/document dropout).

    1. Deterministic cheap query encode (eval + no_grad).
    2. Cheap-score all of ``H`` (student-only), mask positives, top-``L`` per query.
    3. Dedup the batch-wide top-L union → encode each unique doc ONCE with T
       dropout passes (query side too): actual doc encodes = unique_docs·T, far
       below the batch_size·L·T worst case when queries share hard candidates.
    4. Per query: ŝ=mean_T, σ=std_T over its valid top-L; g=ŝ+λσ scattered into a
       full ``(B, B_doc)`` grid (``-inf`` elsewhere) so ``cache.select`` picks
       top-m and its finite-slot guard protects over-masked rows.

    Returns ``(mined, slots, q_student, None, stats)`` — no teacher.
    """
    q_max = fg_cfg['query_max_len']
    p_max = fg_cfg['passage_max_len']
    bs = fg_cfg.get('mc_batch_size', 256)
    T = int(fg_cfg.get('T', 3))
    m = fg_cfg['m']
    lam = float(fg_cfg['lambda_val'])
    Lc = min(int(fg_cfg['L']), cache.B_doc)
    B = len(batch_qids)
    texts = [qid_to_text[q] for q in batch_qids]

    student_was_training = student.training
    try:
        # 1. deterministic cheap query encode (eval, no_grad)
        student.eval()
        q_det = encode_batch_tensor(student, tokenizer, texts, device, q_max, bs,
                                    requires_grad=False).to(cache.Z_student.dtype)

        # 2. cheap-score all H (student-only), mask positives, take top-L per query
        s_cheap = cache.cheap_scores(q_det)
        s_cheap = cache.mask_positives(s_cheap, batch_qids, qrels_dict, inplace=True)
        topl = torch.topk(s_cheap, k=Lc, dim=1).indices                # (B, Lc)
        # masked positives (-inf) that slid into top-L on a tiny/over-masked H are
        # invalid: never encode or select them.
        valid = torch.isfinite(s_cheap.gather(1, topl))                # (B, Lc)

        # Fail early & clearly (mirrors cache.select's finite-slot guard, but before
        # any dropout encode): every query must retain >= m finite candidates.
        n_valid_per_q = valid.sum(dim=1)                               # (B,)
        n_short = int((n_valid_per_q < m).sum())
        if n_short:
            raise ValueError(
                f"MCDP top-L has fewer than m={m} finite candidates for {n_short} "
                f"queries after qrel masking.")

        # 3. dedup the batch-wide top-L union → encode each unique doc once
        union_slots = torch.unique(topl[valid])
        if union_slots.numel() == 0:
            raise ValueError(
                "No finite MCDP candidates after qrel masking; cannot run top-L "
                "dropout mining.")
        slot2u = torch.full((cache.B_doc,), -1, dtype=torch.long, device=device)
        slot2u[union_slots] = torch.arange(union_slots.numel(), device=device)
        union_docids = [cache.docids[s] for s in union_slots.tolist()]
        # strict lookup: a docid missing from corpus_lookup must fail loudly rather
        # than silently encoding an empty string.
        union_texts = [corpus_lookup[d] for d in union_docids]

        # 4. full query/document MC-dropout: T stochastic passes (train, no_grad)
        student.train()
        q_mc = encode_batch_tensor(student, tokenizer, texts * T, device, q_max, bs,
                                   requires_grad=False).reshape(T, B, -1).float()
        d_mc = encode_batch_tensor(student, tokenizer, union_texts * T, device,
                                   p_max, bs, requires_grad=False
                                   ).reshape(T, len(union_texts), -1).float()

        # 5. per-query ŝ/σ/g over valid top-L → scatter into full (B, B_doc) grids
        g_full = torch.full((B, cache.B_doc), float('-inf'), device=device)
        s_hat_full = torch.full((B, cache.B_doc), float('-inf'), device=device)
        sigma_full = torch.zeros((B, cache.B_doc), device=device)
        filled = torch.zeros((B, cache.B_doc), dtype=torch.bool, device=device)
        for i in range(B):
            vslots = topl[i][valid[i]]                 # (V_i,) valid slots
            if vslots.numel() == 0:
                continue
            uidx = slot2u[vslots]                      # (V_i,) union indices
            sims = torch.einsum('td,tvd->tv', q_mc[:, i, :], d_mc[:, uidx, :])
            s_hat_i = sims.mean(dim=0)
            sigma_i = sims.std(dim=0, unbiased=False)  # population std (matches GRASS)
            g_full[i, vslots] = s_hat_i + lam * sigma_i
            s_hat_full[i, vslots] = s_hat_i
            sigma_full[i, vslots] = sigma_i
            filled[i, vslots] = True
    finally:
        student.train(student_was_training)

    # belt-and-suspenders: masked positives can never be selected
    g_full = cache.mask_positives(g_full, batch_qids, qrels_dict, inplace=True)
    slots, neg_docids = cache.select(g_full, m=m, mode=fg_cfg['selection_mode'],
                                     beta=fg_cfg['beta'], L=fg_cfg['L'])
    cache.record_selection(slots)

    mined = {qid: neg_docids[i] for i, qid in enumerate(batch_qids)}
    flat = [d for row in neg_docids for d in row]
    unique_docs = len(union_texts)
    any_filled = bool(filled.any())
    stats = {
        's_hat_mean': float(s_hat_full[filled].float().mean()) if any_filled else 0.0,
        'sigma_mean': float(sigma_full[filled].float().mean()) if any_filled else 0.0,
        **_selection_diag(s_hat_full, sigma_full, g_full, s_hat_full, slots, lam, flat),
        'mcdp_L_used': int(Lc),
        'mcdp_T': int(T),
        'mcdp_unique_docs': int(unique_docs),
        'mcdp_query_encoder_calls': int(B * T),
        'mcdp_doc_encoder_calls': int(unique_docs * T),
        'estimated_max_mcdp_doc_encodes_per_step': int(B * Lc * T),
    }
    return mined, slots, q_det, None, stats


def _mine_batch(cache, student, teacher, tokenizer, batch_qids, qid_to_text,
                corpus_lookup, qrels_dict, fg_cfg, device):
    """Dispatch to the estimator named by ``fg_cfg['uncertainty']`` (`ema`|`mcdp`).
    ``corpus_lookup`` is used only by MCDP (to encode top-L doc texts); EMA ignores it.
    """
    if fg_cfg.get('uncertainty', 'ema') == 'mcdp':
        return _mine_batch_mcdp(cache, student, tokenizer, batch_qids, qid_to_text,
                                corpus_lookup, qrels_dict, fg_cfg, device)
    return _mine_batch_ema(cache, student, teacher, tokenizer, batch_qids,
                           qid_to_text, qrels_dict, fg_cfg, device)


# ---- runtime config --------------------------------------------------------

def _build_fast_grass_cfg(config, args, steps_per_epoch):
    """Runtime fast_grass cfg: config defaults + CLI overrides + derived steps.

    The feasibility script (fast_grass_feasibility.py:_fg_cfg) only sets B_doc /
    lambda_val / max lengths and derives the step fields separately in run_real;
    the trainer needs the full derived set, so this is its own helper.
    """
    cfg = dict(config['training']['fast_grass'])
    if getattr(args, 'B_doc', None) is not None:
        cfg['B_doc'] = args.B_doc
    if getattr(args, 'lambda_val', None) is not None:
        cfg['lambda_val'] = args.lambda_val
    if getattr(args, 'ema_alpha', None) is not None:
        cfg['ema_alpha'] = args.ema_alpha
    if getattr(args, 'uncertainty', None) is not None:
        cfg['uncertainty'] = args.uncertainty
    if getattr(args, 'T', None) is not None:
        cfg['T'] = args.T
    if getattr(args, 'mc_dropout_p', None) is not None:
        cfg['mc_dropout_p'] = args.mc_dropout_p
    if getattr(args, 'L', None) is not None:
        cfg['L'] = args.L
    if getattr(args, 'selection_mode', None) is not None:
        cfg['selection_mode'] = args.selection_mode
    if getattr(args, 'm', None) is not None:
        cfg['m'] = args.m
    if getattr(args, 'num_epochs', None) is not None:
        cfg['num_epochs'] = args.num_epochs
    # --no_registry FULLY disables R: no nominations (uniform-only candidates) AND
    # no admission (RetiredRegistry(max_size=0) drops every admit).
    if getattr(args, 'no_registry', False):
        cfg['uniform_candidate_fraction'] = 1.0
        cfg['R_size_factor'] = 0

    cfg['lambda_val'] = float(cfg['lambda_val'])
    cfg['passage_max_len'] = config['model']['passage_max_len']
    cfg['query_max_len'] = config['model']['query_max_len']
    cfg['steps_per_epoch'] = int(steps_per_epoch)
    cfg['total_steps'] = cfg['steps_per_epoch'] * cfg['num_epochs']
    cfg['max_age_steps'] = cfg['max_age_epochs'] * cfg['steps_per_epoch']
    return cfg


# ---- Algorithm 1 loop ------------------------------------------------------

def run_fast_grass_pipeline(cache, c_ids, corpus_lookup, qrels_dict, qid_to_text,
                            train_items, cfg, config, ctx, device,
                            models=None, compile_model=True, do_eval=True,
                            output_model_dir=None, debug=False):
    """Algorithm 1 over the global cache. Returns the output model dir.

    Model ownership is the pipeline's: with ``models=None`` it loads student /
    teacher / tokenizer itself (so ``main`` never double-loads); the smoke injects
    mocks. ``compile_model=False`` skips torch.compile (mocks/debug);
    ``do_eval=False`` skips the BRIGHT eval (smoke). ``output_model_dir`` lets the
    smoke checkpoint into a temp dir instead of a real ``models/`` path.
    """
    from models.temperature_scaled_loss import TemperatureScaledContrastiveLoss

    lr            = float(cfg['learning_rate'])
    num_epochs    = cfg['num_epochs']
    batch_size    = cfg.get('batch_size', 64)
    enc_bs        = cfg.get('mc_batch_size', 512)
    max_grad_norm = cfg.get('max_grad_norm', 1.0)
    warmup_ratio  = cfg.get('warmup_ratio', 0.1)
    weight_decay  = cfg.get('weight_decay', 0.01)
    logging_steps = cfg.get('logging_steps', 100)
    save_steps    = cfg.get('save_steps', 1000)
    q_max_len     = cfg['query_max_len']
    p_max_len     = cfg['passage_max_len']
    temperature   = ctx['temperature']
    ema_alpha     = cfg.get('ema_alpha', 0.999)
    m             = cfg['m']
    update_every  = cfg['cache_update_interval']
    reservoir_size = cfg.get('recent_query_reservoir_size', 128)

    if debug:
        train_items = train_items[:512]
        print("[FAST-GRASS] DEBUG: 512 items", flush=True)

    # --- estimator + models: pipeline-owned unless injected ---
    # EMA keeps a teacher (Z_teacher + recert). MCDP is teacher-free: teacher=None
    # everywhere and σ comes from top-L MC-dropout in _mine_batch_mcdp.
    uncertainty = cfg.get('uncertainty', 'ema')
    if models is not None:
        student = models['student']
        tokenizer = models['tokenizer']
        teacher = models.get('teacher') if uncertainty == 'ema' else None
    else:
        base_model = ctx['base_model']
        tokenizer = AutoTokenizer.from_pretrained(base_model)
        student = AutoModel.from_pretrained(base_model,
                                            torch_dtype=torch.bfloat16).to(device)
        if uncertainty == 'ema':
            teacher = AutoModel.from_pretrained(base_model,
                                                torch_dtype=torch.bfloat16).to(device)
            teacher.eval()
            for p in teacher.parameters():
                p.requires_grad_(False)
            print(f"[FAST-GRASS] EMA teacher initialized | alpha={ema_alpha}", flush=True)
        else:
            teacher = None
            print("[FAST-GRASS] MCDP (teacher-free): σ from top-L MC-dropout", flush=True)

    if output_model_dir is None:
        output_model_dir = get_path("models") / (cfg['model_name'] + f"_{cfg['uncertainty']}")
    output_model_dir = Path(output_model_dir)
    output_model_dir.mkdir(parents=True, exist_ok=True)

    if _BNB_AVAILABLE and device.type == 'cuda':
        optimizer = bnb.optim.AdamW8bit(student.parameters(), lr=lr, weight_decay=weight_decay)
        print("[FAST-GRASS] AdamW8bit enabled", flush=True)
    else:
        if hasattr(student, 'gradient_checkpointing_enable'):
            try:
                student.gradient_checkpointing_enable()
            except Exception:
                pass
        optimizer = AdamW(student.parameters(), lr=lr, weight_decay=weight_decay)
        print("[FAST-GRASS] AdamW + gradient checkpointing", flush=True)
    student.train()

    # MCDP: set the dropout rate on every Dropout module before compile so the
    # T-pass stochastic encodes use it (mirror run_grass.py:308-313).
    if uncertainty == 'mcdp':
        mc_dropout_p = cfg.get('mc_dropout_p', 0.3)
        n_layers = sum(1 for mod in student.modules()
                       if isinstance(mod, torch.nn.Dropout))
        for mod in student.modules():
            if isinstance(mod, torch.nn.Dropout):
                mod.p = mc_dropout_p
        print(f"[FAST-GRASS] MCDP dropout p={mc_dropout_p} on {n_layers} layers",
              flush=True)

    loss_fn      = TemperatureScaledContrastiveLoss(temperature=temperature)
    _student_raw = student  # raw handle for save_pretrained / EMA / cache encodes
    if compile_model:
        try:
            torch._dynamo.config.suppress_errors = True
            student = torch.compile(student, dynamic=True)
            print("[FAST-GRASS] torch.compile enabled on student", flush=True)
        except Exception as e:
            print(f"[FAST-GRASS] torch.compile skipped ({e})", flush=True)

    n_batches    = max(len(train_items) // batch_size, 1)
    total_steps  = n_batches * num_epochs
    warmup_steps = int(warmup_ratio * total_steps)
    scheduler    = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    # MCDP cost guardrail: worst-case doc encodes/step = batch_size · L · T (before
    # top-L union dedup). Warn if it is large; not a hard block.
    if uncertainty == 'mcdp':
        est = batch_size * int(cfg['L']) * int(cfg.get('T', 3))
        if est > 25_000:
            print(f"[FAST-GRASS] WARNING: MCDP worst-case batch_size*L*T={est} "
                  f"> 25000 — mining may be very slow; consider lowering L/T",
                  flush=True)

    print(f"[FAST-GRASS] Algorithm 1 | uncertainty={uncertainty} | B_doc={cache.B_doc} | "
          f"selection={cfg['selection_mode']} | m={m} | "
          f"batch_size={batch_size} | {total_steps} total steps", flush=True)

    mining_log_f = open(output_model_dir / "mining_log.jsonl", 'w')
    cost_log_f   = open(output_model_dir / "cost_log.jsonl", 'w')

    # rolling recent-query reservoir (reuses no-grad selection embeddings; no
    # extra encode). Hold enough recent batches to fill reservoir_size queries.
    res_batches = max(reservoir_size // batch_size + 1, 1)
    reservoir = deque(maxlen=res_batches)

    global_step = 0
    prev_score_pairs = 0
    t_start = time.time()

    for epoch in range(num_epochs):
        random.shuffle(train_items)
        epoch_loss, epoch_steps = 0.0, 0

        for b in range(n_batches):
            step_t0 = time.time()
            batch_items = train_items[b * batch_size:(b + 1) * batch_size]
            batch_qids = list(dict.fromkeys(it['query_id'] for it in batch_items))
            if not batch_qids:
                continue

            # --- 1. mine (EMA: matmul against H | MCDP: cheap top-L + dropout) ---
            mined, slots, q_student, q_teacher, mstats = _mine_batch(
                cache, student, teacher, tokenizer, batch_qids, qid_to_text,
                corpus_lookup, qrels_dict, cfg, device)

            # --- 2. push selection embeddings into the recert reservoir ---
            #        (q_teacher is None in teacher-free MCDP mode)
            reservoir.append((q_student.detach(),
                              q_teacher.detach() if q_teacher is not None else None,
                              list(batch_qids)))

            # --- 3. training step (IDENTICAL to GRASS — fresh loss encode) ---
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
            q_embs  = encode_batch_tensor(student, tokenizer, queries, device,
                                          q_max_len, enc_bs, requires_grad=True)
            d_texts = [t for pos, negs in zip(positives, negatives)
                       for t in [pos] + negs]
            d_embs  = encode_batch_tensor(student, tokenizer, d_texts, device,
                                          p_max_len, enc_bs, requires_grad=True)
            loss = loss_fn(q_embs, d_embs)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            clip_grad_norm_(student.parameters(), max_grad_norm)
            optimizer.step()
            scheduler.step()

            # --- 4. EMA-update the teacher (EMA mode only; MCDP is teacher-free) ---
            if teacher is not None:
                _update_ema(_student_raw, teacher, ema_alpha)

            epoch_loss  += loss.item()
            epoch_steps += 1
            global_step += 1

            # --- 5. periodic amortized cache maintenance ---
            maint = None
            if global_step % update_every == 0:
                qs = torch.cat([e[0] for e in reservoir], dim=0)
                qids = [q for e in reservoir for q in e[2]]
                res_n = min(reservoir_size, len(qids))
                # teacher-free MCDP: no q_teacher → recert uses student-only scores
                if teacher is not None:
                    qt = torch.cat([e[1] for e in reservoir], dim=0)[-res_n:]
                else:
                    qt = None
                reservoir_dict = {'q_student': qs[-res_n:],
                                  'q_teacher': qt,
                                  'qids': qids[-res_n:]}
                maint = cache.maintain(_student_raw, teacher, tokenizer,
                                       corpus_lookup, c_ids, reservoir_dict,
                                       global_step, cfg, device,
                                       qrels_dict=qrels_dict)

            # --- 6. logging (mining stats + per-step cost DELTAS) ---
            mining_log_f.write(json.dumps({
                "global_step": global_step,
                "num_queries": len(batch_qids),
                "num_selected_negatives": len(queries) * m,
                **mstats,
            }, ensure_ascii=False) + '\n')

            score_pairs_step = cache.cache_score_pairs - prev_score_pairs
            prev_score_pairs = cache.cache_score_pairs
            doc_loss_calls = len(queries) + len(d_texts)
            cost_rec = {
                "global_step": global_step,
                "B_doc": cache.B_doc,
                "selection_mode": cfg['selection_mode'],
                "num_queries": len(batch_qids),
                "num_selected_negatives": len(queries) * m,
                "doc_encoder_calls_loss": doc_loss_calls,
                "doc_encoder_calls_cache_refresh": (maint or {}).get('doc_encoder_calls_cache_refresh', 0),
                "doc_encoder_calls_cache_replace": (maint or {}).get('doc_encoder_calls_cache_replace', 0),
                # MCDP uncertainty encode cost (0 for EMA): actual doc/query encodes
                # this step plus the pre-dedup worst case (batch_size·L·T).
                "doc_encoder_calls_mcdp": mstats.get('mcdp_doc_encoder_calls', 0),
                "query_encoder_calls_mcdp": mstats.get('mcdp_query_encoder_calls', 0),
                "estimated_max_mcdp_doc_encodes_per_step": mstats.get(
                    'estimated_max_mcdp_doc_encodes_per_step', 0),
                "cache_score_pairs": score_pairs_step,
                "num_refresh": (maint or {}).get('num_refresh', 0),
                "num_replace": (maint or {}).get('num_replace', 0),
                "num_over_age": (maint or {}).get('num_over_age', 0),
                "over_age_backlog": (maint or {}).get('over_age_backlog', 0),
                "num_R_entries": len(cache.registry),
                "num_R_candidates": (maint or {}).get('num_R_candidates', 0),
                "num_uniform_candidates": (maint or {}).get('num_uniform_candidates', 0),
                "num_recertified_candidates": (maint or {}).get('num_recertified_candidates', 0),
                "replacement_yield_at_K": (
                    maint['num_replace'] / max(1, maint['num_recertified_candidates'])
                    if maint else None),
                "selected_doc_diversity": mstats['selected_doc_diversity'],
                "cache_turnover_rate": (maint or {}).get('cache_turnover_rate', 0.0),
                "ann_queries": 0,
                "index_rebuilds": 0,
                "step_wall_time": time.time() - step_t0,
            }
            cost_log_f.write(json.dumps(cost_rec, ensure_ascii=False) + '\n')

            if global_step % logging_steps == 0:
                elapsed   = time.time() - t_start
                secs_per  = elapsed / global_step
                remaining = secs_per * (total_steps - global_step)
                eta       = f"{int(remaining // 3600)}h {int((remaining % 3600) // 60)}m"
                print(f"[FAST-GRASS] Ep{epoch+1} step {b+1}/{n_batches} | "
                      f"loss={loss.item():.4f} | mined={len(queries)}/{len(batch_qids)} "
                      f"| R={len(cache.registry)} | ETA {eta}", flush=True)

            if global_step % save_steps == 0:
                ckpt = output_model_dir / f"checkpoint-{global_step}"
                ckpt.mkdir(parents=True, exist_ok=True)
                _student_raw.save_pretrained(str(ckpt))
                tokenizer.save_pretrained(str(ckpt))
                torch.save(scheduler.state_dict(), ckpt / "scheduler.pt")
                torch.save(optimizer.state_dict(), ckpt / "optimizer.pt")
                print(f"[FAST-GRASS] Checkpoint saved: {ckpt.name}", flush=True)

        avg = epoch_loss / max(1, epoch_steps)
        print(f"[FAST-GRASS] Epoch {epoch+1} done. avg_loss={avg:.4f} | "
              f"trained_batches={epoch_steps}/{n_batches}", flush=True)

    _student_raw.save_pretrained(str(output_model_dir))
    tokenizer.save_pretrained(str(output_model_dir))
    mining_log_f.close()
    cost_log_f.close()
    total_train_time = time.time() - t_start
    print(f"[FAST-GRASS] Training complete. Model at {output_model_dir} | "
          f"total train time {total_train_time/3600:.2f}h "
          f"({total_train_time:.0f}s, {global_step} steps)", flush=True)

    if do_eval:
        del student, _student_raw, teacher
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        evaluate_bright(ctx, config, output_model_dir, temp_workdir_key='temp_grass')

    return output_model_dir


# ---- cluster entry point ---------------------------------------------------

def _load_train_items(debug=False):
    """Load the GRASS training mixture (positive_passages schema)."""
    mix_dir = get_path("processed") / "training_mixture"
    train_items = []
    for f_path in sorted(mix_dir.glob("*.jsonl")):
        if f_path.name.startswith('.'):
            continue
        with open(f_path) as f:
            for line in f:
                d = json.loads(line)
                pos = d.get('positive_passages', [])
                if not pos:
                    continue
                train_items.append({
                    'query_id':  str(d['query_id']),
                    'query':     d['query'],
                    'pos_docid': pos[0]['docid'],
                })
    return train_items


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model_suffix',   type=str,   default=None)
    parser.add_argument('--num_epochs',     type=int,   default=None)
    parser.add_argument('--B_doc',          type=int,   default=None)
    parser.add_argument('--lambda_val',     type=float, default=None)
    parser.add_argument('--ema_alpha',      type=float, default=None,
                        help='EMA teacher decay; 1.0 = frozen base-model teacher '
                             '(σ = how much fine-tuning moved each doc score)')
    parser.add_argument('--selection_mode', choices=['topk', 'softmax'], default=None)
    parser.add_argument('--m',              type=int,   default=None)
    parser.add_argument('--uncertainty',    choices=['ema', 'mcdp'], default=None,
                        help='σ estimator (default from config: mcdp). '
                             'mcdp = teacher-free top-L MC-dropout; ema = baseline')
    parser.add_argument('--T',              type=int,   default=None,
                        help='MCDP stochastic passes (default 3)')
    parser.add_argument('--mc_dropout_p',   type=float, default=None,
                        help='MCDP dropout probability (default 0.3)')
    parser.add_argument('--L',              type=int,   default=None,
                        help='MCDP lazy top-L shortlist inside H / softmax prefilter '
                             '(default 128; MCDP cost ~ batch_size*L*T)')
    parser.add_argument('--no_registry',    action='store_true',
                        help='ablation: fully disable the retired registry R')
    parser.add_argument('--no_eval',        action='store_true',
                        help='skip the post-training BRIGHT eval (run it later, '
                             'sequentially, via run_all_evals.py / '
                             'run_evaluate_singularity.sh — avoids the shared '
                             'eval-scratch race when sweeps run in parallel)')
    parser.add_argument('--debug',          action='store_true')
    args = parser.parse_args()

    config = load_config()
    cfg    = config['training']['fast_grass']
    ctx    = get_training_context('fast_grass')
    set_seed(config.get('seed', 42))

    if args.model_suffix is not None:
        cfg = {**cfg, 'model_name': cfg['model_name'] + '_' + args.model_suffix}

    # after the CLI overrides, so the block reports what will actually run
    log_startup_config('fast_grass', ctx, cfg)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    from data.preprocessor import run_setup
    corpus_file, _query_file, qrels_file = run_setup()

    # stale index: reuse if present; build ONCE if missing; never rebuild here.
    workdir   = get_path("temp_grass")
    workdir.mkdir(exist_ok=True, parents=True)
    stale_pkl = workdir / "stale_index" / "corpus.pkl"
    stale_pkl.parent.mkdir(exist_ok=True)
    if not stale_pkl.exists():
        print("[FAST-GRASS] Building stale index (one-off, init source only)...", flush=True)
        encode_to_pickle(ctx['base_model'], corpus_file, stale_pkl, False, ctx, config)
    print(f"[FAST-GRASS] Stale index ready: {stale_pkl}", flush=True)

    # build_faiss_index gives the ordered (embeddings, c_ids) the cache inits from;
    # the FAISS index itself is unused (no per-query ANN in Fast-GRASS).
    _stale_idx, stale_embs, c_ids = build_faiss_index(stale_pkl)
    corpus_lookup = _load_corpus_lookup(corpus_file)
    qrels_dict    = _load_qrels(qrels_file)

    train_items = _load_train_items(debug=args.debug)
    if args.debug:
        train_items = train_items[:512]
    random.shuffle(train_items)
    qid_to_text = {it['query_id']: it['query'] for it in train_items}
    batch_size  = cfg.get('batch_size', 64)
    steps_per_epoch = max(len(train_items) // batch_size, 1)
    print(f"[FAST-GRASS] {len(train_items)} training examples | "
          f"{len(qid_to_text)} unique queries | steps/epoch={steps_per_epoch}",
          flush=True)

    fg_cfg = _build_fast_grass_cfg(config, args, steps_per_epoch)
    fg_cfg['model_name'] = cfg['model_name']

    # --- CONFIG PARAMETER PRINTS (resolved values, after CLI overrides) ---
    print("\n" + "="*48, flush=True)
    print("🛠️  VERIFYING FAST-GRASS CONFIGURATION", flush=True)
    print(f"▶️  Model name (out):  {fg_cfg['model_name']}_{fg_cfg['uncertainty']}", flush=True)
    print(f"▶️  Base model:        {ctx['base_model']}", flush=True)
    print(f"▶️  B_doc:             {fg_cfg['B_doc']}", flush=True)
    print(f"▶️  lambda_val:        {fg_cfg['lambda_val']}", flush=True)
    print(f"▶️  m (negs/query):    {fg_cfg['m']}", flush=True)
    print(f"▶️  selection_mode:    {fg_cfg['selection_mode']}", flush=True)
    print(f"▶️  uncertainty:       {fg_cfg['uncertainty']}", flush=True)
    if fg_cfg['uncertainty'] == 'mcdp':
        _bs = fg_cfg.get('batch_size', batch_size)
        _Lc = min(int(fg_cfg['L']), int(fg_cfg['B_doc']))
        print(f"▶️  T (MCDP passes):   {fg_cfg.get('T', 3)}", flush=True)
        print(f"▶️  mc_dropout_p:      {fg_cfg.get('mc_dropout_p', 0.3)}", flush=True)
        print(f"▶️  L (top-L):         {fg_cfg['L']}  (used {_Lc})", flush=True)
        print(f"▶️  MCDP max encodes:  {_bs * _Lc * int(fg_cfg.get('T', 3))}/step "
              f"(batch_size*L*T, pre top-L union dedup)", flush=True)
    else:
        print(f"▶️  ema_alpha:         {fg_cfg['ema_alpha']}", flush=True)
    print(f"▶️  registry R:        {'DISABLED (--no_registry)' if args.no_registry else 'enabled'}", flush=True)
    print(f"▶️  post-train eval:   {'SKIPPED (--no_eval)' if args.no_eval else 'enabled'}", flush=True)
    print(f"▶️  num_epochs:        {fg_cfg['num_epochs']}", flush=True)
    print(f"▶️  batch_size:        {fg_cfg.get('batch_size', batch_size)}", flush=True)
    print(f"▶️  steps/epoch:       {steps_per_epoch}  (total {fg_cfg['total_steps']})", flush=True)
    print(f"▶️  query/passage len: {fg_cfg['query_max_len']} / {fg_cfg['passage_max_len']}", flush=True)
    print(f"▶️  learning_rate:     {fg_cfg['learning_rate']}", flush=True)
    print("="*48 + "\n", flush=True)

    cache = NegativeCache.init_uniform(stale_embs, c_ids, fg_cfg, device)
    print(f"[FAST-GRASS] Cache H initialized | B_doc={cache.B_doc} | "
          f"Z_H={cache.memory_bytes()/1e9:.2f} GB", flush=True)

    # Fast-GRASS does no per-query ANN: the FAISS index + full stale embeddings
    # are only an init source. Free them once H is built to reclaim CPU RAM.
    del _stale_idx, stale_embs
    gc.collect()

    run_fast_grass_pipeline(cache, c_ids, corpus_lookup, qrels_dict, qid_to_text,
                            train_items, fg_cfg, config, ctx, device,
                            models=None, compile_model=True,
                            do_eval=not args.no_eval, debug=args.debug)


if __name__ == "__main__":
    main()
