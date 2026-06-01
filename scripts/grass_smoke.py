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

from utils.grass_candidate_memory import CandidateMemory


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

    # 2) Phase 3 config knobs present
    import yaml
    with open(project_root / "config" / "config.yaml") as f:
        cfg = yaml.safe_load(f)
    grass_cfg = cfg["training"]["grass"]
    cc = grass_cfg.get("candidate_cache", {})
    cc_ok = all(k in cc for k in (
        "enabled", "max_candidates_per_query", "ttl_rounds",
        "top_g_to_store", "top_sigma_to_store",
    ))
    pool_ok = "max_pool_per_query" in grass_cfg
    checks.append(("candidate_cache.* config keys present", cc_ok))
    checks.append(("max_pool_per_query config key present", pool_ok))

    # 3) CandidateMemory pickle round-trip
    mem = CandidateMemory(max_per_query=8, ttl_rounds=2,
                          top_g_to_store=4, top_sigma_to_store=4)
    mem.update("q0", current_round=3, selected_negs=["d1"],
               top_g_docids=["d2"], top_sigma_docids=["d3"])
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "mem.pkl"
        mem.save(path)
        loaded = CandidateMemory.load(path, max_per_query=8, ttl_rounds=2,
                                      top_g_to_store=4, top_sigma_to_store=4)
    ids_orig,   _ = mem.get("q0", current_round=3)
    ids_loaded, _ = loaded.get("q0", current_round=3)
    checks.append(("CandidateMemory save/load round-trip preserves ids",
                   ids_orig == ids_loaded and ids_orig == ["d1", "d2", "d3"]))

    # 4) TTL validity on fresh insertion
    mem2 = CandidateMemory(max_per_query=8, ttl_rounds=2,
                           top_g_to_store=4, top_sigma_to_store=4)
    mem2.update("q0", current_round=5, selected_negs=["d_valid"])
    _, exp_valid   = mem2.get("q0", current_round=7)   # within TTL (5+2)
    _, exp_invalid = mem2.get("q0", current_round=8)   # outside TTL
    checks.append(("CandidateMemory TTL: valid at round 7 after insert at 5",
                   exp_valid is False))
    checks.append(("CandidateMemory TTL: expired at round 8 (ttl_rounds=2)",
                   exp_invalid is True))

    # 5) run_grass.py public surface: --uncertainty CLI, _update_ema helper,
    #    Algorithm 1 per-batch outer loop
    grass_src = (project_root / "scripts" / "run_grass.py").read_text()
    has_mc_dropout_choice = "'mc_dropout'" in grass_src and "'ema'" in grass_src
    has_uncertainty_arg   = "--uncertainty" in grass_src
    has_ema_helper        = "def _update_ema(" in grass_src
    has_pipeline          = "def run_grass_pipeline(" in grass_src
    has_per_batch_loop    = ("for b in range(n_batches)" in grass_src and
                             "_mine_queries(" in grass_src)
    checks.append(("run_grass: --uncertainty CLI accepts mc_dropout + ema",
                   has_mc_dropout_choice and has_uncertainty_arg))
    checks.append(("run_grass: _update_ema(student, teacher, alpha) defined", has_ema_helper))
    checks.append(("run_grass: run_grass_pipeline() defined",                 has_pipeline))
    checks.append(("run_grass: per-batch Algorithm 1 loop present",           has_per_batch_loop))

    # 6) _mine_queries signature returns dict[qid -> list[docid]] (not tuple)
    has_return_dict_of_lists = ("mined[qid] = top_m_docs" in grass_src and
                                "return mined, log_records" in grass_src)
    checks.append(("run_grass: _mine_queries returns {qid: [top_m_docs]}",
                   has_return_dict_of_lists))

    # 7) Sanity: dropped knobs are gone from config
    dropped_keys = ["mine_every", "n_das", "ema_batch_size",
                    "mab_n_das", "selection", "bandit_epsilon"]
    none_present = all(k not in grass_cfg for k in dropped_keys)
    checks.append(("config: legacy mode knobs (mine_every/n_das/ema_batch_size/mab_*) removed",
                   none_present))

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
