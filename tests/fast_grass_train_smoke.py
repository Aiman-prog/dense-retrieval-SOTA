"""
Fast-GRASS trainer smoke — CPU-only, deterministic end-to-end drive of
run_fast_grass_pipeline with mocks (no run_setup, no GPU, no compile, no BRIGHT eval).

Builds a tiny synthetic setup (mock student/teacher, mock tokenizer, ~40-doc corpus,
~12 train items, B_doc=8, cache_update_interval=2, 1 epoch), runs the real Algorithm-1
pipeline, and asserts the training/cache/logging contract holds.

Run: python tests/fast_grass_train_smoke.py
"""
import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root / 'src'))
sys.path.insert(0, str(project_root / 'scripts'))


# ---- mocks (mirror GradMockModel/MockTokenizer in fast_grass_test.py) ------

class _MockOutput:
    def __init__(self, last_hidden_state):
        self.last_hidden_state = last_hidden_state


class GradMockModel(nn.Module):
    """CLS = nn.Embedding(input_ids[:,0]); real params so loss has a gradient."""
    def __init__(self, vocab=1000, hidden=8):
        super().__init__()
        self.emb = nn.Embedding(vocab, hidden)
        self.config = type('C', (), {'hidden_size': hidden})()

    def forward(self, input_ids=None, attention_mask=None, **kwargs):
        return _MockOutput(last_hidden_state=self.emb(input_ids))

    def save_pretrained(self, path):
        Path(path).mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), Path(path) / "pytorch_model.smoke")


class DropoutMockModel(GradMockModel):
    """Like GradMockModel but with active dropout, so mining encodes differ
    between train() and eval() — used to prove _mine_batch forces eval()."""
    def __init__(self, vocab=1000, hidden=8, p=0.5):
        super().__init__(vocab=vocab, hidden=hidden)
        self.drop = nn.Dropout(p=p)

    def forward(self, input_ids=None, attention_mask=None, **kwargs):
        return _MockOutput(last_hidden_state=self.drop(self.emb(input_ids)))


class _BatchEncoding(dict):
    def to(self, device):
        return _BatchEncoding({k: v.to(device) for k, v in self.items()})


class MockTokenizer:
    def __call__(self, texts, padding=True, truncation=True,
                 max_length=128, return_tensors='pt'):
        ids = torch.zeros(len(texts), 4, dtype=torch.long)
        for i, t in enumerate(texts):
            ids[i, 0] = abs(hash(t)) % 1000
        return _BatchEncoding({'input_ids': ids,
                               'attention_mask': torch.ones(len(texts), 4,
                                                            dtype=torch.long)})

    def save_pretrained(self, path):
        Path(path).mkdir(parents=True, exist_ok=True)
        (Path(path) / "tokenizer.smoke").write_text("ok")


def run_smoke():
    print("\nFast-GRASS Trainer Smoke Test")
    print("=" * 56)
    checks = []

    from utils.negative_cache import NegativeCache
    import run_fast_grass

    # API surface
    api_ok = all(callable(getattr(run_fast_grass, n, None)) for n in
                 ["run_fast_grass_pipeline", "_mine_batch",
                  "_build_fast_grass_cfg", "main"])
    checks.append(("run_fast_grass public API present", api_ok))

    device = torch.device('cpu')
    dim = 8

    # synthetic corpus + cache
    n_corpus = 40
    c_ids = [f"d{i}" for i in range(n_corpus)]
    corpus_lookup = {d: f"document {d} body text" for d in c_ids}
    embs = np.random.default_rng(0).standard_normal((n_corpus, dim)).astype('float32')

    # tiny train mixture
    train_items = [{'query_id': f"q{i}", 'query': f"query number {i}",
                    'pos_docid': c_ids[i % n_corpus]} for i in range(12)]
    qid_to_text = {it['query_id']: it['query'] for it in train_items}
    # a known positive per query so masking is exercised
    qrels_dict = {it['query_id']: {it['pos_docid']} for it in train_items}

    cfg = dict(
        model_name="fast_grass_smoke", uncertainty='ema',
        B_doc=8, m=1, selection_mode='topk', lambda_val=1.0, beta=5.0, L=1024,
        ema_alpha=0.999, rho_start=0.50, rho_end=0.10,
        cache_update_interval=2, max_age_epochs=4, utility_ema_decay=0.95,
        utility_floor=0.01, utility_remember_threshold=0.05, K=3, R_fraction=0.25,
        uniform_candidate_fraction=0.75, replacement_candidate_multiplier=2,
        recent_query_reservoir_size=8, reentry_top_k=5, R_size_factor=0.5,
        cache_init_seed=42,
        learning_rate=1e-4, num_epochs=1, batch_size=4, mc_batch_size=16,
        max_grad_norm=1.0, warmup_ratio=0.1, weight_decay=0.01,
        logging_steps=1, save_steps=2,
        passage_max_len=128, query_max_len=128,
        steps_per_epoch=3, total_steps=3, max_age_steps=12)

    cache = NegativeCache.init_uniform(embs, c_ids, cfg, device, dim=dim)
    cache_docids_before = list(cache.docids)

    student = GradMockModel(hidden=dim)
    teacher = GradMockModel(hidden=dim)
    teacher.load_state_dict(student.state_dict())
    for p in teacher.parameters():
        p.requires_grad_(False)
    tok = MockTokenizer()
    ctx = {'temperature': 0.02, 'base_model': 'mock'}
    config = {'model': {'query_max_len': 128, 'passage_max_len': 512}}

    # capture loss across steps by wrapping the loss fn? simplest: read mining/cost logs
    with tempfile.TemporaryDirectory() as tmp:
        out_dir = run_fast_grass.run_fast_grass_pipeline(
            cache, c_ids, corpus_lookup, qrels_dict, qid_to_text, train_items,
            cfg, config, ctx, device,
            models={"student": student, "teacher": teacher, "tokenizer": tok},
            compile_model=False, do_eval=False, output_model_dir=Path(tmp) / "model",
            debug=False)

        out_dir = Path(out_dir)
        cost_lines = [json.loads(l) for l in
                      (out_dir / "cost_log.jsonl").read_text().splitlines()]
        mining_lines = [json.loads(l) for l in
                        (out_dir / "mining_log.jsonl").read_text().splitlines()]

        # 1) finite, positive loss-encode calls every step
        finite_steps = (len(cost_lines) >= 1 and
                        all(np.isfinite(r['step_wall_time']) and
                            r['doc_encoder_calls_loss'] > 0 for r in cost_lines))
        checks.append(("loss-encode happens every step (finite cost log)", finite_steps))

        # 2) B_doc invariant after maintenance fired
        maint_fired = any(r['num_refresh'] or r['num_replace'] for r in cost_lines) \
            or any(r['global_step'] % cfg['cache_update_interval'] == 0 for r in cost_lines)
        b_doc_ok = (len(cache.docids) == cache.B_doc and
                    len(set(cache.docids)) == cache.B_doc)
        checks.append(("B_doc invariant after maintain fired", maint_fired and b_doc_ok))

        # 3) mined negatives never include a masked positive
        no_pos_neg = True
        cache_after = set(cache.docids)
        for it in train_items:
            # the qrels positive for this query must never have been emitted; we
            # re-mine deterministically once to check the contract directly
            pass
        # direct contract check via a fresh mine on current cache
        qb = [it['query_id'] for it in train_items[:4]]
        mined, slots, qs_e, qt_e, _ = run_fast_grass._mine_batch(
            cache, student, teacher, tok, qb, qid_to_text, corpus_lookup,
            qrels_dict, cfg, device)
        for qid in qb:
            for d in mined[qid]:
                if d in qrels_dict.get(qid, set()):
                    no_pos_neg = False
        checks.append(("mined negatives exclude known positives", no_pos_neg))

        # 4) cost log cache_score_pairs is per-step (non-cumulative): each step
        #    scores exactly num_queries * B_doc pairs
        per_step_ok = all(
            r['cache_score_pairs'] == r['num_queries'] * cache.B_doc
            for r in cost_lines)
        checks.append(("cost log cache_score_pairs is per-step (not cumulative)",
                       per_step_ok))

        # 5) cost log has the full design field set
        required_fields = {
            "global_step", "B_doc", "selection_mode", "num_queries",
            "num_selected_negatives", "doc_encoder_calls_loss",
            "doc_encoder_calls_cache_refresh", "doc_encoder_calls_cache_replace",
            "cache_score_pairs",
            "num_refresh", "num_replace", "num_over_age", "over_age_backlog",
            "num_R_entries", "num_R_candidates", "num_uniform_candidates",
            "num_recertified_candidates", "replacement_yield_at_K",
            "selected_doc_diversity", "cache_turnover_rate", "ann_queries",
            "index_rebuilds", "step_wall_time"}
        fields_ok = required_fields.issubset(set(cost_lines[0].keys()))
        checks.append(("cost log has full design field set", fields_ok))

        # 6) mining log written with expected keys
        mining_ok = (len(mining_lines) == len(cost_lines) and
                     all({"global_step", "num_queries", "s_hat_mean",
                          "selected_doc_diversity"}.issubset(r) for r in mining_lines))
        checks.append(("mining log written with expected keys", mining_ok))

        # 7) a checkpoint + final model dir saved under the tmp dir
        ckpts = list(out_dir.glob("checkpoint-*"))
        final_ok = (out_dir.exists() and
                    (out_dir / "tokenizer.smoke").exists() and len(ckpts) >= 1)
        checks.append(("checkpoint + final model saved under tmpdir", final_ok))

    # 8) mining is deterministic eval/no_grad even when the student is in train()
    #    mode (post-step), and the prior mode is restored. A dropout model makes
    #    train-mode encodes stochastic, so two consecutive mines would differ if
    #    _mine_batch failed to force eval(). Also bites if mode isn't restored.
    drop_student = DropoutMockModel(hidden=dim)
    drop_teacher = DropoutMockModel(hidden=dim)
    drop_teacher.load_state_dict(drop_student.state_dict())
    drop_student.train(); drop_teacher.train()
    qb = [it['query_id'] for it in train_items[:4]]
    _, _, qs1, _, _ = run_fast_grass._mine_batch(
        cache, drop_student, drop_teacher, tok, qb, qid_to_text, corpus_lookup,
        qrels_dict, cfg, device)
    student_mode_restored = drop_student.training and drop_teacher.training
    _, _, qs2, _, _ = run_fast_grass._mine_batch(
        cache, drop_student, drop_teacher, tok, qb, qid_to_text, corpus_lookup,
        qrels_dict, cfg, device)
    deterministic = torch.allclose(qs1.float(), qs2.float())
    checks.append(("mining forces deterministic eval + restores train mode",
                   deterministic and student_mode_restored))

    print()
    ok = 0
    for name, passed in checks:
        print(f"  {'PASS' if passed else 'FAIL'}  {name}")
        ok += bool(passed)
    print("=" * 56)
    print(f"  {ok}/{len(checks)} checks passed")
    return 0 if ok == len(checks) else 1


def run_mcdp_smoke():
    """Teacher-free MCDP pipeline drive: verifies dispatch, top-L-only encoding,
    cost accounting, positive masking, negatives-from-H, and no cached gradients."""
    print("\nFast-GRASS MCDP Trainer Smoke Test")
    print("=" * 56)
    checks = []

    from utils.negative_cache import NegativeCache
    import run_fast_grass

    device = torch.device('cpu')
    dim = 8
    # B_doc=30 with L=2 makes the top-L union strictly smaller than H, so the
    # "MCDP encodes only top-L, not all H" check is meaningful.
    n_corpus = 30
    c_ids = [f"d{i}" for i in range(n_corpus)]
    corpus_lookup = {d: f"document {d} body text" for d in c_ids}
    embs = np.random.default_rng(1).standard_normal((n_corpus, dim)).astype('float32')

    train_items = [{'query_id': f"q{i}", 'query': f"query number {i}",
                    'pos_docid': c_ids[i % n_corpus]} for i in range(12)]
    qid_to_text = {it['query_id']: it['query'] for it in train_items}
    qrels_dict = {it['query_id']: {it['pos_docid']} for it in train_items}

    cfg = dict(
        model_name="fast_grass_mcdp_smoke", uncertainty='mcdp',
        B_doc=30, m=1, selection_mode='topk', lambda_val=1.0, beta=5.0,
        L=2, T=2, mc_dropout_p=0.5,
        ema_alpha=0.999, rho_start=0.50, rho_end=0.10,
        cache_update_interval=2, max_age_epochs=4, utility_ema_decay=0.95,
        utility_floor=0.01, utility_remember_threshold=0.05, K=3, R_fraction=0.25,
        uniform_candidate_fraction=0.75, replacement_candidate_multiplier=2,
        recent_query_reservoir_size=8, reentry_top_k=5, R_size_factor=0.5,
        cache_init_seed=42,
        learning_rate=1e-4, num_epochs=1, batch_size=4, mc_batch_size=16,
        max_grad_norm=1.0, warmup_ratio=0.1, weight_decay=0.01,
        logging_steps=1, save_steps=2,
        passage_max_len=128, query_max_len=128,
        steps_per_epoch=3, total_steps=3, max_age_steps=12)

    cache = NegativeCache.init_uniform(embs, c_ids, cfg, device, dim=dim)
    checks.append(("MCDP cache init teacher-free (Z_teacher None)",
                   cache.Z_teacher is None))

    student = DropoutMockModel(hidden=dim, p=cfg['mc_dropout_p'])
    tok = MockTokenizer()
    ctx = {'temperature': 0.02, 'base_model': 'mock'}
    config = {'model': {'query_max_len': 128, 'passage_max_len': 512}}

    with tempfile.TemporaryDirectory() as tmp:
        # NOTE: no 'teacher' in models — MCDP must run teacher-free end to end.
        out_dir = run_fast_grass.run_fast_grass_pipeline(
            cache, c_ids, corpus_lookup, qrels_dict, qid_to_text, train_items,
            cfg, config, ctx, device,
            models={"student": student, "tokenizer": tok},
            compile_model=False, do_eval=False, output_model_dir=Path(tmp) / "model",
            debug=False)
        out_dir = Path(out_dir)
        cost_lines = [json.loads(l) for l in
                      (out_dir / "cost_log.jsonl").read_text().splitlines()]
        mining_lines = [json.loads(l) for l in
                        (out_dir / "mining_log.jsonl").read_text().splitlines()]

        checks.append(("teacher-free after training (Z_teacher None)",
                       cache.Z_teacher is None))

        # MCDP mining-log carries the full diagnostic set
        mfields = {"s_hat_mean", "sigma_mean", "sel_s_hat_mean", "sel_sigma_mean",
                   "sel_lambda_sigma_mean", "flip_rate_vs_lambda0",
                   "selected_doc_diversity", "mcdp_L_used", "mcdp_T",
                   "mcdp_unique_docs", "mcdp_query_encoder_calls",
                   "mcdp_doc_encoder_calls",
                   "estimated_max_mcdp_doc_encodes_per_step"}
        mining_ok = len(mining_lines) >= 1 and all(mfields.issubset(r)
                                                   for r in mining_lines)
        checks.append(("MCDP mining log has full diagnostic set", mining_ok))

        # cost log carries MCDP encode-cost (doc AND query) and worst-case estimate
        cfields = {"doc_encoder_calls_mcdp", "query_encoder_calls_mcdp",
                   "estimated_max_mcdp_doc_encodes_per_step"}
        cost_ok = (all(cfields.issubset(r) for r in cost_lines) and
                   all(r["doc_encoder_calls_mcdp"] > 0 for r in cost_lines))
        checks.append(("MCDP cost log has doc+query encode fields (>0)", cost_ok))

        # only top-L is dropout-encoded: unique docs <= num_queries*L and < B_doc
        topl_ok = all(
            r["mcdp_unique_docs"] <= r["num_queries"] * r["mcdp_L_used"] and
            r["mcdp_unique_docs"] < cache.B_doc
            for r in mining_lines)
        checks.append(("MCDP encodes only top-L union (<B_doc, <=Q*L)", topl_ok))

        # cost accounting identities (dedup savings visible: actual vs est-max)
        acct_ok = all(
            r["mcdp_doc_encoder_calls"] == r["mcdp_unique_docs"] * r["mcdp_T"] and
            r["mcdp_query_encoder_calls"] == r["num_queries"] * r["mcdp_T"] and
            r["estimated_max_mcdp_doc_encodes_per_step"] ==
                r["num_queries"] * r["mcdp_L_used"] * r["mcdp_T"]
            for r in mining_lines)
        checks.append(("MCDP cost accounting identities hold", acct_ok))

    # fresh mine on the trained cache: masking + from-H + teacher-free return
    qb = [it['query_id'] for it in train_items[:4]]
    mined, slots, q_stu, q_tea, _ = run_fast_grass._mine_batch(
        cache, student, None, tok, qb, qid_to_text, corpus_lookup,
        qrels_dict, cfg, device)
    cache_docids = set(cache.docids)
    clean = all(d not in qrels_dict.get(qid, set())
                for qid in qb for d in mined[qid])
    from_H = all(d in cache_docids for qid in qb for d in mined[qid])
    checks.append(("MCDP mined negatives exclude known positives", clean))
    checks.append(("MCDP mined negatives come from H", from_H))
    checks.append(("MCDP miner returns q_teacher=None", q_tea is None))
    checks.append(("cached Z_student is grad-free (selection-only)",
                   not cache.Z_student.requires_grad))

    print()
    ok = 0
    for name, passed in checks:
        print(f"  {'PASS' if passed else 'FAIL'}  {name}")
        ok += bool(passed)
    print("=" * 56)
    print(f"  {ok}/{len(checks)} checks passed")
    return 0 if ok == len(checks) else 1


if __name__ == "__main__":
    rc_ema = run_smoke()
    rc_mcdp = run_mcdp_smoke()
    sys.exit(rc_ema or rc_mcdp)
