"""
Evaluate all training checkpoints for a model across BRIGHT domains.
Discovers checkpoint directories (0%, 20%, 40%, 60%, 80%, 100%) and runs
evaluate.py on each, saving aggregated results to JSON.
"""

import os
import sys
import re
import json
import argparse
import subprocess
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root / 'src'))

from utils.helpers import load_config, get_data_base_dir, get_path


def discover_checkpoints(model_dir: Path, config) -> list:
    """
    Discover all checkpoints in a model directory.
    Returns list of dicts: [{pct, step, path}, ...]

    Order:
    - 0%   = pretrained base model (before any training)
    - 20%  = checkpoint-N (lowest step)
    - ...
    - 80%  = checkpoint-N (highest step)
    - 100% = model_dir itself (final model)
    """
    checkpoints = []

    # 0% = the pretrained base model
    base_model = config['model']['base_model']
    # Try to resolve from HF cache
    cache_base = get_path("bright").resolve() / "hub"
    repo_id = base_model.replace("/", "--")
    snapshot_dir = cache_base / f"models--{repo_id}" / "snapshots"

    base_path = base_model  # fallback to HF name
    if snapshot_dir.exists():
        snapshots = [d for d in snapshot_dir.iterdir() if d.is_dir()]
        if snapshots:
            chosen = sorted(snapshots)[-1]
            if (chosen / "config.json").exists():
                base_path = str(chosen)

    checkpoints.append({"pct": 0, "step": 0, "path": base_path})

    # Find checkpoint-* directories
    ckpt_dirs = sorted(
        [d for d in model_dir.iterdir() if d.is_dir() and re.match(r'checkpoint-\d+', d.name)],
        key=lambda d: int(re.search(r'checkpoint-(\d+)', d.name).group(1))
    )

    if ckpt_dirs:
        total_ckpts = len(ckpt_dirs) + 1  # +1 for the final model
        for i, ckpt_dir in enumerate(ckpt_dirs, start=1):
            step = int(re.search(r'checkpoint-(\d+)', ckpt_dir.name).group(1))
            pct = round(100 * i / total_ckpts)
            checkpoints.append({"pct": pct, "step": step, "path": str(ckpt_dir)})

    # 100% = the final model (root of model_dir)
    if (model_dir / "config.json").exists():
        # Estimate final step from last checkpoint
        final_step = checkpoints[-1]["step"] if len(checkpoints) > 1 else 0
        checkpoints.append({"pct": 100, "step": final_step, "path": str(model_dir)})

    return checkpoints


def evaluate_checkpoint(ckpt_path: str, domains: list, config) -> dict:
    """Run evaluate.py for a single checkpoint across all domains. Returns metrics dict."""
    eval_script = project_root / "src" / "evaluation" / "evaluate.py"
    domain_results = {}

    for domain in domains:
        print(f"    Evaluating {domain}...", flush=True)
        cmd = [
            sys.executable, str(eval_script),
            "--model_path", ckpt_path,
            "--domain", domain,
        ]

        try:
            subprocess.run(cmd, check=True)
            # Read the JSON result that evaluate.py saves
            base_dir = Path(get_data_base_dir())
            result_file = base_dir / config['paths']['results_dir'] / f"{domain}_results.json"
            if result_file.exists():
                with open(result_file) as f:
                    data = json.load(f)
                domain_results[domain] = data["metrics"]
            else:
                print(f"    Warning: No result file for {domain}", flush=True)
                domain_results[domain] = {}
        except subprocess.CalledProcessError:
            print(f"    Failed for {domain}, skipping...", flush=True)
            domain_results[domain] = {}

    return domain_results


def main():
    config = load_config()

    parser = argparse.ArgumentParser(description="Evaluate all checkpoints for a trained model")
    parser.add_argument("--model_dir", type=str, required=True,
                        help="Path to the trained model directory (contains checkpoint-* dirs)")
    args = parser.parse_args()

    model_dir = Path(args.model_dir)
    if not model_dir.exists():
        print(f"ERROR: Model directory not found: {model_dir}")
        sys.exit(1)

    domains = config['evaluation'].get('eval_domains', [])
    checkpoints = discover_checkpoints(model_dir, config)

    print(f"\nFound {len(checkpoints)} checkpoints:")
    for ckpt in checkpoints:
        print(f"  {ckpt['pct']:3d}% (step {ckpt['step']}): {ckpt['path']}")
    print()

    # Evaluate each checkpoint
    all_results = {"checkpoints": []}

    for ckpt in checkpoints:
        print(f"\n{'='*60}")
        print(f"Evaluating {ckpt['pct']}% checkpoint (step {ckpt['step']})")
        print(f"Path: {ckpt['path']}")
        print(f"{'='*60}")

        domain_metrics = evaluate_checkpoint(ckpt["path"], domains, config)

        all_results["checkpoints"].append({
            "pct": ckpt["pct"],
            "step": ckpt["step"],
            "path": ckpt["path"],
            "domains": domain_metrics
        })

    # Save aggregated results
    base_dir = Path(get_data_base_dir())
    model_name = model_dir.name
    output_dir = base_dir / config['paths']['results_dir'] / model_name
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "checkpoint_results.json"

    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2)

    print(f"\nAll checkpoint results saved to: {output_file}")


if __name__ == "__main__":
    main()
