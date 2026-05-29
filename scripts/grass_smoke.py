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

    # 7) MCD repeated-mining control flow present (epoch loop + per-epoch helpers)
    mcd_src = (project_root / "scripts" / "run_grass_mcd.py").read_text()
    loop_ok  = "for epoch in range(start_epoch, num_epochs + 1)" in mcd_src
    chain_ok = "base_jsonl_dir=prev_mining_dir" in mcd_src and "current_model" in mcd_src
    helper_ok = "def train_one_epoch_mcd" in mcd_src
    mem_persist_ok = "memory.save(memory_path)" in mcd_src
    checks.append(("run_grass_mcd: per-epoch loop present", loop_ok))
    checks.append(("run_grass_mcd: chains JSONL + checkpoint across epochs", chain_ok))
    checks.append(("run_grass_mcd: train_one_epoch_mcd helper defined", helper_ok))
    checks.append(("run_grass_mcd: persists CandidateMemory after mining", mem_persist_ok))

    # 8) MCD training continuity: shared output_dir + cumulative epochs + resume-safe flags
    shared_out_ok    = "output_dir = get_path(\"models\") / f\"{base_name}_mcdp\"" in mcd_src
    cumulative_ok    = "cumulative_epochs=epoch" in mcd_src and "cumulative_epochs" in mcd_src
    ignore_skip_ok   = "'--ignore_data_skip'" in mcd_src and "'True'" in mcd_src
    save_epoch_ok    = "'--save_strategy'" in mcd_src and "'epoch'" in mcd_src
    checks.append(("run_grass_mcd: shared output_dir for resume continuity", shared_out_ok))
    checks.append(("run_grass_mcd: cumulative_epochs grows per call", cumulative_ok))
    checks.append(("run_grass_mcd: ignore_data_skip True for new-mined data", ignore_skip_ok))
    checks.append(("run_grass_mcd: save_strategy=epoch for clean resume boundary", save_epoch_ok))

    # 9) MCD resume-after-preemption: detects completed epochs, never wipes output_dir
    no_wipe_ok       = "shutil.rmtree(output_dir)" not in mcd_src and "rmtree" not in mcd_src
    detect_helper_ok = "def _detect_completed_epochs" in mcd_src
    resume_loop_ok   = "range(start_epoch, num_epochs + 1)" in mcd_src
    short_circuit_ok = "completed_epochs >= num_epochs" in mcd_src
    checks.append(("run_grass_mcd: no destructive wipe of output_dir", no_wipe_ok))
    checks.append(("run_grass_mcd: _detect_completed_epochs helper present", detect_helper_ok))
    checks.append(("run_grass_mcd: loop starts at resumed epoch", resume_loop_ok))
    checks.append(("run_grass_mcd: skips work if all epochs already done", short_circuit_ok))

    # 10) _detect_completed_epochs unit behavior on synthetic checkpoint dirs
    import sys as _sys
    _sys.path.insert(0, str(project_root / "scripts"))
    from run_grass_mcd import _detect_completed_epochs
    with tempfile.TemporaryDirectory() as td:
        # absent dir -> 0
        absent = Path(td) / "absent"
        ok_absent = _detect_completed_epochs(absent) == 0
        # empty dir -> 0
        empty = Path(td) / "empty"
        empty.mkdir()
        ok_empty = _detect_completed_epochs(empty) == 0
        # dir with checkpoint-100/trainer_state.json (epoch=2.0) -> 2
        good = Path(td) / "good"
        ckpt = good / "checkpoint-100"
        ckpt.mkdir(parents=True)
        with open(ckpt / "trainer_state.json", "w") as f:
            json.dump({"epoch": 2.0, "global_step": 100}, f)
        ok_good = _detect_completed_epochs(good) == 2
        # corrupted trainer_state -> 0 (safe fallback)
        bad = Path(td) / "bad"
        bad_ckpt = bad / "checkpoint-50"
        bad_ckpt.mkdir(parents=True)
        with open(bad_ckpt / "trainer_state.json", "w") as f:
            f.write("not json {")
        ok_bad = _detect_completed_epochs(bad) == 0
    checks.append(("_detect_completed_epochs: absent dir -> 0",          ok_absent))
    checks.append(("_detect_completed_epochs: empty dir -> 0",           ok_empty))
    checks.append(("_detect_completed_epochs: epoch=2.0 ckpt -> 2",      ok_good))
    checks.append(("_detect_completed_epochs: corrupted state -> 0",     ok_bad))

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
