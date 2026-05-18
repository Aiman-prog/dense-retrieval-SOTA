"""GRASS async v2 smoke test — no GPU, no model load, no internet.

Verifies the wiring between the orchestrator, miner, trainer, and config:
  1. All three async v2 scripts import without error.
  2. config.training.grass.async_v2 has the required keys.
  3. compute_total_epochs hard-errors on PLACEHOLDER constants.
  4. compute_total_epochs returns a sane integer for numeric inputs.
  5. EpsilonGreedyBandit pickle round-trip preserves state (init-pass → main miner IPC).

Run: python scripts/grass_smoke.py
"""

import importlib.util
import pickle
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root / 'src'))


def run_smoke():
    print("\nGRASS Async v2 Wiring Check")
    print("=" * 50)

    checks = []
    scripts_dir = Path(__file__).parent

    mods = {}
    for name, file in [
        ("orch",    "train_grass_async_v2.py"),
        ("miner",   "run_grass_async_v2_miner.py"),
        ("trainer", "run_grass_async_v2_trainer.py"),
    ]:
        spec = importlib.util.spec_from_file_location(f"_v2_{name}", scripts_dir / file)
        mod  = importlib.util.module_from_spec(spec)
        try:
            spec.loader.exec_module(mod)
            mods[name] = mod
            checks.append((f"Import {file}", True))
        except Exception as e:
            checks.append((f"Import {file} ({e.__class__.__name__}: {e})", False))
            mods[name] = None

    from utils.helpers import load_config
    cfg_v2 = None
    try:
        full = load_config()
        cfg_v2 = full['training']['grass']['async_v2']
        required = ['M', 'X', 'selection', 'lambda_val',
                    't_init', 't_mine', 't_train',
                    'poll_interval_steps', 'save_steps']
        missing = [k for k in required if k not in cfg_v2]
        checks.append((f"async_v2 config block has required keys "
                       f"(missing: {missing or 'none'})", not missing))
    except Exception as e:
        checks.append((f"Load async_v2 config block ({e.__class__.__name__}: {e})", False))

    orch = mods.get("orch")
    if orch is not None and cfg_v2 is not None:
        try:
            orch.compute_total_epochs({**cfg_v2, 't_init': 'PLACEHOLDER'}, M=3)
            checks.append(("compute_total_epochs rejects PLACEHOLDER", False))
        except ValueError:
            checks.append(("compute_total_epochs rejects PLACEHOLDER", True))
        except Exception as e:
            checks.append((f"compute_total_epochs PLACEHOLDER check "
                           f"({e.__class__.__name__})", False))

    if orch is not None:
        try:
            te = orch.compute_total_epochs(
                {'t_init': 900, 't_mine': 1800, 't_train': 5700}, M=3
            )
            ok = isinstance(te, int) and te == 2  # ceil((900 + 3*1800) / 5700) = 2
            checks.append((f"compute_total_epochs numeric → {te} (expect 2)", ok))
        except Exception as e:
            checks.append((f"compute_total_epochs numeric ({e.__class__.__name__}: {e})", False))

    try:
        from utils.bandit import EpsilonGreedyBandit
        b1 = EpsilonGreedyBandit(epsilon=0.3, alpha=0.5)
        b1.init_query_pool([f"q{i}" for i in range(5)])
        b1.update("q0", 0.7)
        blob = pickle.dumps(b1)
        b2   = pickle.loads(blob)
        ok = (b2.epsilon == b1.epsilon and b2.alpha == b1.alpha
              and abs(b2.mean_g.get("q0", 0.0) - b1.mean_g["q0"]) < 1e-9
              and b2._all_qids == b1._all_qids)
        checks.append(("EpsilonGreedyBandit pickle round-trip preserves state", ok))
    except Exception as e:
        checks.append((f"Bandit pickle round-trip ({e.__class__.__name__}: {e})", False))

    passed = 0
    for name, ok in checks:
        status = "PASS" if ok else "FAIL"
        print(f"  [{status}]  {name}")
        if ok:
            passed += 1
    print("=" * 50)
    print(f"  {passed}/{len(checks)} passed")
    print("=" * 50)
    return passed == len(checks)


if __name__ == "__main__":
    sys.exit(0 if run_smoke() else 1)
