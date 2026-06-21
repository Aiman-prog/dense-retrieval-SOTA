"""
Fast-GRASS smoke test — local CPU-only wiring checks:
negative-cache public API, config.training.fast_grass keys, and one tiny
synthetic init -> score -> mask -> select -> maintain cache cycle.

Run: python scripts/fast_grass_smoke.py
"""
import sys
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root / 'src'))


class _Out:
    def __init__(self, h):
        self.last_hidden_state = h


class _Model(nn.Module):
    def __init__(self, hidden=8):
        super().__init__()
        self.emb = nn.Embedding(1000, hidden)
        self.config = type('C', (), {'hidden_size': hidden})()

    def forward(self, input_ids=None, attention_mask=None, **kw):
        return _Out(self.emb(input_ids))


class _Enc(dict):
    def to(self, device):
        return _Enc({k: v.to(device) for k, v in self.items()})


class _Tok:
    def __call__(self, texts, padding=True, truncation=True,
                 max_length=128, return_tensors='pt'):
        ids = torch.zeros(len(texts), 4, dtype=torch.long)
        for i, t in enumerate(texts):
            ids[i, 0] = abs(hash(t)) % 1000
        return _Enc({'input_ids': ids,
                     'attention_mask': torch.ones(len(texts), 4, dtype=torch.long)})


def run_smoke():
    print("\nFast-GRASS Smoke Test")
    print("=" * 50)
    checks = []

    # 1) config.training.fast_grass present with the v0 knobs
    import yaml
    with open(project_root / "config" / "config.yaml") as f:
        cfg_all = yaml.safe_load(f)
    fg = cfg_all["training"].get("fast_grass", {})
    required = ["B_doc", "m", "selection_mode", "lambda_val", "beta", "L",
               "uncertainty", "ema_alpha", "rho_start", "rho_end",
               "cache_update_interval", "max_age_epochs", "utility_ema_decay",
               "utility_floor", "utility_remember_threshold", "K", "R_fraction",
               "uniform_candidate_fraction", "replacement_candidate_multiplier",
               "recent_query_reservoir_size", "reentry_top_k", "R_size_factor",
               "cache_init_seed"]
    missing = [k for k in required if k not in fg]
    checks.append((f"config.training.fast_grass has all v0 keys "
                   f"(missing={missing})", not missing))

    # 2) public API importable + callable
    from utils.negative_cache import NegativeCache, RetiredRegistry, linear_decay
    api_ok = all(callable(x) for x in [NegativeCache, RetiredRegistry, linear_decay])
    methods = ["init_uniform", "score", "mask_positives", "select",
               "record_selection", "maintain", "memory_bytes"]
    api_ok = api_ok and all(hasattr(NegativeCache, m) for m in methods)
    checks.append(("negative_cache public API present", api_ok))

    # 3) tiny synthetic end-to-end cache cycle on CPU
    device = torch.device('cpu')
    dim, n = 8, 24
    cfg = dict(
        B_doc=8, m=1, selection_mode='topk', lambda_val=1.0, beta=5.0, L=1024,
        uncertainty='ema', ema_alpha=0.999, rho_start=0.5, rho_end=0.1,
        cache_update_interval=100, max_age_steps=4, utility_ema_decay=0.95,
        utility_floor=0.01, utility_remember_threshold=0.05, K=3, R_fraction=0.25,
        uniform_candidate_fraction=0.75, replacement_candidate_multiplier=2,
        recent_query_reservoir_size=4, reentry_top_k=5, R_size_factor=0.5,
        cache_init_seed=42, steps_per_epoch=100, total_steps=1000,
        passage_max_len=128, mc_batch_size=64)
    embs = np.random.default_rng(0).standard_normal((n, dim)).astype('float32')
    c_ids = [f"d{i}" for i in range(n)]
    corpus = {d: f"text {d}" for d in c_ids}
    cache = NegativeCache.init_uniform(embs, c_ids, cfg, device, dim=dim)

    model, tok = _Model().eval(), _Tok()
    norm = torch.nn.functional.normalize
    qs = norm(torch.randn(4, dim), dim=-1)
    g, s_hat, sigma = cache.score(qs, qs, cfg['lambda_val'])
    g = cache.mask_positives(g, [f"q{i}" for i in range(4)], {'q0': {cache.docids[0]}})
    slots, docids = cache.select(g, m=cfg['m'], mode='topk')
    cache.record_selection(slots)
    cycle_ok = (g.shape == (4, cache.B_doc) and slots.shape == (4, 1) and
                torch.isfinite(s_hat).all() and torch.isfinite(sigma).all())
    checks.append(("score/mask/select cycle: finite, correct shapes", cycle_ok))

    # 4) maintain runs, keeps B_doc invariant, returns cost counters
    cache.intervals_since_selected[:] = cfg['K']
    cache.utility_ema[:] = 0.0
    reservoir = {'q_student': qs, 'q_teacher': qs, 'qids': [f"q{i}" for i in range(4)]}
    counters = cache.maintain(model, model, tok, corpus, c_ids, reservoir,
                              step=50, cfg=cfg, device=device, qrels_dict={})
    maint_ok = (len(cache.docids) == cache.B_doc and
                'num_replace' in counters and 'cache_turnover_rate' in counters)
    checks.append(("maintain: B_doc invariant + cost counters", maint_ok))

    # 5) memory report is positive
    checks.append(("memory_bytes() > 0", cache.memory_bytes() > 0))

    print()
    ok = 0
    for name, passed in checks:
        print(f"  {'PASS' if passed else 'FAIL'}  {name}")
        ok += bool(passed)
    print("=" * 50)
    print(f"  {ok}/{len(checks)} checks passed")
    return 0 if ok == len(checks) else 1


if __name__ == "__main__":
    sys.exit(run_smoke())
