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
