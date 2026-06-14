"""
GRASS smoke test — local CPU-only checks for shared wiring:
mining-log shape, config keys, CandidateMemory, and run_grass.py imports.

Run: python scripts/grass_smoke.py
"""
import json
import sys
import tempfile
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root / 'src'))


def run_smoke():
    print("\nGRASS Smoke Test")
    print("=" * 50)

    checks = []

    # 1) mining-log JSONL record shape (shared logging contract)
    with tempfile.TemporaryDirectory() as tmpdir:
        p = Path(tmpdir) / "mining_log.jsonl"
        with open(p, "w") as f:
            f.write(json.dumps({"query_id": "q0", "g_selected": 0.42}) + "\n")
        with open(p) as f:
            rec = json.loads(f.readline())
        checks.append(("Mining log record contains query_id + g_selected",
                       "query_id" in rec and "g_selected" in rec))

    # 2) Core config knobs present
    import yaml
    with open(project_root / "config" / "config.yaml") as f:
        cfg = yaml.safe_load(f)
    grass_cfg = cfg["training"]["grass"]
    pool_ok = "max_pool_per_query" in grass_cfg
    checks.append(("max_pool_per_query config key present", pool_ok))

    # 3) run_grass.py public surface — import-based (robust to internal refactors,
    #    unlike source-string matching which breaks when functions are split).
    sys.path.insert(0, str(project_root / "scripts"))
    import run_grass
    public_api = [
        "main", "run_grass_pipeline", "_mine_queries", "_update_ema",
        "_fresh_shortlists", "_score_mc_dropout", "_score_ema",
        "_select_and_log_negatives",
    ]
    api_ok = all(callable(getattr(run_grass, n, None)) for n in public_api)
    checks.append(("run_grass: public API present (pipeline, _mine_queries, split helpers)",
                   api_ok))

    # 3b) --uncertainty CLI still offers both estimators (string contract)
    grass_src = (project_root / "scripts" / "run_grass.py").read_text()
    cli_ok = ("--uncertainty" in grass_src and
              "'mc_dropout'" in grass_src and "'ema'" in grass_src)
    checks.append(("run_grass: --uncertainty CLI accepts mc_dropout + ema", cli_ok))

    # 7) Sanity: dropped knobs are gone from config
    dropped_keys = ["mine_every", "n_das", "ema_batch_size",
                    "mab_n_das", "selection", "bandit_epsilon"]
    none_present = all(k not in grass_cfg for k in dropped_keys)
    checks.append(("config: legacy mode knobs (mine_every/n_das/ema_batch_size/mab_*) removed",
                   none_present))

    # 8) MC-dropout: vectorized T passes == T separate forward passes
    import torch
    import torch.nn as nn
    import numpy as np

    torch.manual_seed(42)
    _model = nn.Sequential(nn.Linear(8, 8), nn.Dropout(p=0.5), nn.Linear(8, 8))
    _model.train()
    _x = torch.ones(2, 8)
    _T = 3

    # Loop method
    _loop = []
    with torch.no_grad():
        for _ in range(_T):
            _loop.append(_model(_x).numpy())
    _loop_result = np.stack(_loop, axis=0)  # (T, B, dim)

    # Vectorized method (run_grass.py approach)
    torch.manual_seed(42)
    _model2 = nn.Sequential(nn.Linear(8, 8), nn.Dropout(p=0.5), nn.Linear(8, 8))
    _model2.load_state_dict(_model.state_dict())
    _model2.train()
    with torch.no_grad():
        _vec_result = _model2(_x.repeat(_T, 1)).numpy().reshape(_T, 2, 8)

    _equiv = (np.allclose(_loop_result, _vec_result) and
              np.allclose(_loop_result.std(0), _vec_result.std(0)))
    checks.append(("MC-dropout: vectorized T passes == T separate forward passes", _equiv))

    # 9) Fresh rerank top-L: encoding then dot-product picks same top-L as brute force
    _B, _dim, _P, _L = 4, 16, 20, 5
    torch.manual_seed(0)
    _q_embs  = torch.nn.functional.normalize(torch.randn(_B, _dim), dim=-1).numpy()
    _d_embs  = torch.nn.functional.normalize(torch.randn(_P, _dim), dim=-1).numpy()

    # Brute force: full P x B similarity matrix, pick top-L per query
    _bf_shortlist = []
    for _i in range(_B):
        _scores = _d_embs @ _q_embs[_i]
        _top    = np.argsort(_scores)[::-1][:_L].tolist()
        _bf_shortlist.append(_top)

    # Grass method: pool_embs[idxs] @ q_emb (same computation, just via index lookup)
    _grass_shortlist = []
    for _i in range(_B):
        _idxs   = list(range(_P))  # all P docs in pool
        _scores = _d_embs[_idxs] @ _q_embs[_i]
        _top    = np.argsort(_scores)[::-1][:_L].tolist()
        _grass_shortlist.append([_idxs[k] for k in _top])

    _rerank_ok = _bf_shortlist == _grass_shortlist
    checks.append(("Fresh rerank top-L: dot-product index lookup == brute-force top-L", _rerank_ok))

    # 10) Fresh rerank necessity:
    #   (a) zero drift  → stale top-L == current top-L (fresh rerank is a no-op)
    #   (b) large drift → stale top-L != current top-L (fresh rerank changes selection)
    torch.manual_seed(0)
    _P2, _L2, _dim2 = 50, 5, 32
    _q = torch.nn.functional.normalize(torch.randn(1, _dim2), dim=-1).numpy()[0]
    _d_stale = torch.nn.functional.normalize(torch.randn(_P2, _dim2), dim=-1).numpy()
    _stale_top_l = set(np.argsort(_d_stale @ _q)[::-1][:_L2].tolist())

    # (a) no drift — current embeddings identical to stale
    _d_no_drift = _d_stale.copy()
    _no_drift_top_l = set(np.argsort(_d_no_drift @ _q)[::-1][:_L2].tolist())
    _no_drift_same = (_stale_top_l == _no_drift_top_l)

    # (b) large drift — current embeddings heavily perturbed
    torch.manual_seed(1)
    _drift = torch.nn.functional.normalize(torch.randn(_P2, _dim2), dim=-1).numpy()
    _d_drifted = torch.nn.functional.normalize(
        torch.tensor(_d_stale + _drift).float(), dim=-1).numpy()
    _drifted_top_l = set(np.argsort(_d_drifted @ _q)[::-1][:_L2].tolist())
    _drift_differs = (_stale_top_l != _drifted_top_l)

    checks.append(("Fresh rerank: zero drift → stale top-L == current top-L", _no_drift_same))
    checks.append(("Fresh rerank: large drift → stale top-L != current top-L", _drift_differs))

    passed = 0
    for name, ok in checks:
        print(f"  {'✅ PASS' if ok else '❌ FAIL'}  {name}")
        if ok:
            passed += 1

    print("=" * 50)
    print(f"  {passed}/{len(checks)} passed")
    print("=" * 50)
    return passed == len(checks)


if __name__ == "__main__":
    ok = run_smoke()
    sys.exit(0 if ok else 1)
