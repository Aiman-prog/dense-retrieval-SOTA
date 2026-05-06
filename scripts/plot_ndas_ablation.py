"""
Plot n_das ablation: x=n_das, y=avg metric across BRIGHT domains.
Reads from results/bright_benchmark/{model_name}/{domain}_results.json
"""
import json
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root / 'src'))
from utils.helpers import load_config, get_data_base_dir

DOMAINS = [
    "biology", "earth_science", "economics", "psychology", "robotics",
    "stackoverflow", "sustainable_living", "pony", "leetcode", "aops",
    "theoremqa_theorems", "theoremqa_questions",
]

METRICS = {
    "recip_rank":   "MRR@10",
    "ndcg_cut_10":  "NDCG@10",
    "recall_1000":  "Recall@1000",
}

COLORS = {
    "recip_rank":  "#e15759",
    "ndcg_cut_10": "#f28e2b",
    "recall_1000": "#4e79a7",
}


def extract_ndas(model_name: str) -> int | None:
    m = re.search(r'ndas(\d+)', model_name)
    return int(m.group(1)) if m else None


def load_model_results(results_dir: Path, model_name: str) -> dict | None:
    """Returns {metric_key: avg_across_domains} or None if incomplete."""
    model_dir = results_dir / model_name
    if not model_dir.exists():
        return None

    sums = {k: 0.0 for k in METRICS}
    counts = {k: 0 for k in METRICS}

    for domain in DOMAINS:
        rf = model_dir / f"{domain}_results.json"
        if not rf.exists():
            print(f"  ⚠  missing: {model_name}/{domain}_results.json")
            continue
        data = json.loads(rf.read_text())
        for mk in METRICS:
            val = data.get("metrics", {}).get(mk)
            if val is not None:
                sums[mk] += val
                counts[mk] += 1

    if all(c == 0 for c in counts.values()):
        return None

    return {mk: (sums[mk] / counts[mk] if counts[mk] else 0.0) for mk in METRICS}


def per_domain_table(results_dir: Path, models: list[tuple[int, str]]):
    """Print a per-domain breakdown table for all models."""
    print("\n" + "=" * 80)
    print("PER-DOMAIN Recall@1000")
    print("=" * 80)
    header = f"{'Domain':<28}" + "".join(f"  ndas{n:<4}" for n, _ in models)
    print(header)
    print("-" * len(header))
    for domain in DOMAINS:
        row = f"{domain:<28}"
        for _, model_name in models:
            rf = results_dir / model_name / f"{domain}_results.json"
            if rf.exists():
                val = json.loads(rf.read_text()).get("metrics", {}).get("recall_1000", float("nan"))
                row += f"  {val:.4f}  "
            else:
                row += f"  {'N/A':<8}"
        print(row)


def main():
    config = load_config()
    base_dir = Path(get_data_base_dir())
    results_dir = base_dir / config['paths']['results_dir']

    if not results_dir.exists():
        print(f"❌ Results directory not found: {results_dir}")
        print("   Re-run evaluations first — see run_evaluate_singularity.sh")
        sys.exit(1)

    # Discover ndas model dirs
    ndas_models: list[tuple[int, str]] = []
    for d in sorted(results_dir.iterdir()):
        if not d.is_dir():
            continue
        n = extract_ndas(d.name)
        if n is not None:
            ndas_models.append((n, d.name))

    ndas_models.sort()

    if not ndas_models:
        print("❌ No ndas model result directories found under", results_dir)
        sys.exit(1)

    print(f"Found {len(ndas_models)} ndas models: {[n for n, _ in ndas_models]}")

    # Load avg metrics per model
    rows = []
    for n_das, model_name in ndas_models:
        avgs = load_model_results(results_dir, model_name)
        if avgs is None:
            print(f"  ⚠  skipping {model_name} (no results)")
            continue
        rows.append((n_das, model_name, avgs))
        print(f"  ndas={n_das:>2}  MRR={avgs['recip_rank']:.4f}  "
              f"NDCG={avgs['ndcg_cut_10']:.4f}  Recall={avgs['recall_1000']:.4f}")

    if not rows:
        print("❌ No complete results found.")
        sys.exit(1)

    xs = [r[0] for r in rows]

    # ── Main plot: avg metrics vs n_das ───────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 5))

    for mk, label in METRICS.items():
        ys = [r[2][mk] for r in rows]
        ax.plot(xs, ys, marker='o', linewidth=2, markersize=7,
                color=COLORS[mk], label=label)
        for x, y in zip(xs, ys):
            ax.annotate(f"{y:.3f}", (x, y), textcoords="offset points",
                        xytext=(0, 8), ha='center', fontsize=8,
                        color=COLORS[mk])

    ax.set_xlabel("n_das (challengers per batch)", fontsize=12)
    ax.set_ylabel("Avg metric across 12 BRIGHT domains", fontsize=12)
    ax.set_title("GRASS EMA — CaseBandit n_das ablation", fontsize=13, fontweight='bold')
    ax.set_xticks(xs)
    ax.set_xticklabels([str(x) for x in xs])
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=0)

    out_avg = project_root / "results" / "ndas_ablation_avg.png"
    out_avg.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_avg, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"\n📊 Saved: {out_avg}")

    # ── Per-domain Recall@1000 heatmap ───────────────────────────────────────
    recall_matrix = []
    for domain in DOMAINS:
        row_vals = []
        for n_das, model_name, _ in rows:
            rf = results_dir / model_name / f"{domain}_results.json"
            if rf.exists():
                v = json.loads(rf.read_text()).get("metrics", {}).get("recall_1000", 0.0)
            else:
                v = 0.0
            row_vals.append(v)
        recall_matrix.append(row_vals)

    mat = np.array(recall_matrix)
    fig2, ax2 = plt.subplots(figsize=(max(6, len(rows) * 1.4), 7))
    im = ax2.imshow(mat, aspect='auto', cmap='YlOrRd', vmin=0, vmax=1)
    ax2.set_xticks(range(len(rows)))
    ax2.set_xticklabels([f"ndas={n}" for n, _, _ in rows], fontsize=10)
    ax2.set_yticks(range(len(DOMAINS)))
    ax2.set_yticklabels(DOMAINS, fontsize=9)
    ax2.set_title("Recall@1000 per domain — n_das ablation", fontsize=12, fontweight='bold')
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            ax2.text(j, i, f"{mat[i, j]:.3f}", ha='center', va='center',
                     fontsize=7.5, color='black' if mat[i, j] < 0.7 else 'white')
    plt.colorbar(im, ax=ax2, shrink=0.8)

    out_heat = project_root / "results" / "ndas_ablation_heatmap.png"
    fig2.savefig(out_heat, dpi=150, bbox_inches='tight')
    plt.close(fig2)
    print(f"📊 Saved: {out_heat}")

    # ── Print per-domain table ─────────────────────────────────────────────
    per_domain_table(results_dir, [(n, m) for n, m, _ in rows])


if __name__ == "__main__":
    main()
