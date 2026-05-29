"""
GRASS sequential smoke test — local CPU-only checks for bandit wiring.

Run: python scripts/grass_smoke.py
"""
import json
import sys
import tempfile
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root / 'src'))

from utils.bandit import EpsilonGreedyBandit
from utils.grass_candidate_memory import CandidateMemory


def run_smoke():
    print("\nGRASS Sequential Smoke Test")
    print("=" * 50)

    checks = []

    # 1) init_query_pool + select_global returns bounded, distinct query IDs
    bandit = EpsilonGreedyBandit(epsilon=0.3, alpha=0.5)
    qids = [f"q{i}" for i in range(20)]
    bandit.init_query_pool(qids)
    picked = bandit.select_global(8)
    checks.append(("Bandit returns <= n distinct queries", len(picked) <= 8 and len(set(picked)) == len(picked)))

    # 2) update() should bias exploitation toward updated query when epsilon=0
    exploit_bandit = EpsilonGreedyBandit(epsilon=0.0, alpha=1.0)
    exploit_bandit.init_query_pool(["q0", "q1", "q2"])
    exploit_bandit.update("q1", 1.0)
    picked2 = exploit_bandit.select_global(1)
    checks.append(("Bandit exploitation picks highest-updated query", picked2 == ["q1"]))

    # 3) parse/mining-log style JSONL record shape (shared sequential logging contract)
    with tempfile.TemporaryDirectory() as tmpdir:
        p = Path(tmpdir) / "mining_log.jsonl"
        with open(p, "w") as f:
            f.write(json.dumps({"query_id": "q0", "g_selected": 0.42}) + "\n")
        with open(p) as f:
            rec = json.loads(f.readline())
        checks.append(("Mining log record contains query_id + g_selected", "query_id" in rec and "g_selected" in rec))

    # 4) Phase 3 config knobs present
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

    # 5) CandidateMemory pickle round-trip
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

    # 6) TTL validity on fresh insertion
    mem2 = CandidateMemory(max_per_query=8, ttl_rounds=2,
                           top_g_to_store=4, top_sigma_to_store=4)
    mem2.update("q0", current_round=5, selected_negs=["d_valid"])
    _, exp_valid   = mem2.get("q0", current_round=7)   # within TTL (5+2)
    _, exp_invalid = mem2.get("q0", current_round=8)   # outside TTL
    checks.append(("CandidateMemory TTL: valid at round 7 after insert at 5",
                   exp_valid is False))
    checks.append(("CandidateMemory TTL: expired at round 8 (ttl_rounds=2)",
                   exp_invalid is True))

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
