"""
Plot Recall@1000 (and optionally MRR, NDCG@10) across training checkpoints.
Reads checkpoint_results.json produced by eval_checkpoints.py.

Can run locally (no GPU needed):
    python scripts/plot_recall_curve.py --results_file results/inbatch_reasonir_neg/checkpoint_results.json
"""

import json
import argparse
import numpy as np
from pathlib import Path

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend (works without display)
import matplotlib.pyplot as plt


METRIC_LABELS = {
    "recall_1000": "Recall@1000",
    "ndcg_cut_10": "NDCG@10",
    "recip_rank": "MRR",
}

# Distinct colors for up to 12 domains
COLORS = [
    '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
    '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf',
    '#aec7e8', '#ffbb78',
]


def load_results(results_file: str) -> dict:
    with open(results_file) as f:
        return json.load(f)


def plot_metric_curve(data: dict, metric: str, output_path: str, show_domains: bool = True):
    """
    Plot a single metric across checkpoints.

    Args:
        data: The checkpoint_results.json content
        metric: pytrec_eval metric name (e.g. 'recall_1000')
        output_path: Where to save the PNG
        show_domains: If True, plot individual domain lines. Always plots average.
    """
    checkpoints = data["checkpoints"]
    pcts = [ckpt["pct"] for ckpt in checkpoints]

    # Collect all domains from the first checkpoint that has results
    all_domains = set()
    for ckpt in checkpoints:
        all_domains.update(ckpt.get("domains", {}).keys())
    all_domains = sorted(all_domains)

    if not all_domains:
        print(f"No domain results found, skipping {metric} plot.")
        return

    fig, ax = plt.subplots(figsize=(8, 5))

    # Per-domain lines
    domain_values = {}
    for domain in all_domains:
        values = []
        for ckpt in checkpoints:
            domain_metrics = ckpt.get("domains", {}).get(domain, {})
            values.append(domain_metrics.get(metric, 0.0) * 100)  # Convert to percentage
        domain_values[domain] = values

    if show_domains:
        for i, domain in enumerate(all_domains):
            color = COLORS[i % len(COLORS)]
            ax.plot(pcts, domain_values[domain], marker='o', markersize=4,
                    color=color, alpha=0.6, linewidth=1, label=domain)

    # Average across domains
    avg_values = []
    for j in range(len(checkpoints)):
        vals = [domain_values[d][j] for d in all_domains if domain_values[d][j] > 0]
        avg_values.append(np.mean(vals) if vals else 0.0)

    ax.plot(pcts, avg_values, marker='s', markersize=6, color='black',
            linewidth=2.5, label='Average', zorder=10)

    ax.set_xlabel("Training Progress (%)", fontsize=12)
    ax.set_ylabel(f"{METRIC_LABELS.get(metric, metric)} (%)", fontsize=12)
    ax.set_title(f"{METRIC_LABELS.get(metric, metric)} vs Training Progress", fontsize=13)
    ax.set_xticks(pcts)
    ax.set_xticklabels([f"{p}%" for p in pcts])
    ax.legend(fontsize=8, loc='lower right', ncol=2)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Plot recall/metric curves from checkpoint evaluation results")
    parser.add_argument("--results_file", type=str, required=True,
                        help="Path to checkpoint_results.json")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Directory to save plots (defaults to same dir as results_file)")
    parser.add_argument("--metrics", nargs='+', default=["recall_1000"],
                        help="Metrics to plot (default: recall_1000). Options: recall_1000, ndcg_cut_10, recip_rank")
    parser.add_argument("--no_domains", action="store_true",
                        help="Only plot the average line, hide individual domains")
    args = parser.parse_args()

    results_path = Path(args.results_file)
    if not results_path.exists():
        print(f"ERROR: Results file not found: {results_path}")
        return

    output_dir = Path(args.output_dir) if args.output_dir else results_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    data = load_results(str(results_path))

    for metric in args.metrics:
        output_file = output_dir / f"{metric}_curve.png"
        plot_metric_curve(data, metric, str(output_file), show_domains=not args.no_domains)

    # Also generate a combined 3-metric plot if multiple metrics requested
    if len(args.metrics) > 1:
        plot_combined(data, args.metrics, str(output_dir / "combined_metrics.png"),
                      show_domains=not args.no_domains)


def plot_combined(data: dict, metrics: list, output_path: str, show_domains: bool = False):
    """Plot multiple metrics on subplots."""
    checkpoints = data["checkpoints"]
    pcts = [ckpt["pct"] for ckpt in checkpoints]

    all_domains = set()
    for ckpt in checkpoints:
        all_domains.update(ckpt.get("domains", {}).keys())
    all_domains = sorted(all_domains)

    fig, axes = plt.subplots(1, len(metrics), figsize=(6 * len(metrics), 5))
    if len(metrics) == 1:
        axes = [axes]

    for ax, metric in zip(axes, metrics):
        domain_values = {}
        for domain in all_domains:
            values = []
            for ckpt in checkpoints:
                domain_metrics = ckpt.get("domains", {}).get(domain, {})
                values.append(domain_metrics.get(metric, 0.0) * 100)
            domain_values[domain] = values

        if show_domains:
            for i, domain in enumerate(all_domains):
                color = COLORS[i % len(COLORS)]
                ax.plot(pcts, domain_values[domain], marker='o', markersize=3,
                        color=color, alpha=0.5, linewidth=1, label=domain)

        avg_values = []
        for j in range(len(checkpoints)):
            vals = [domain_values[d][j] for d in all_domains if domain_values[d][j] > 0]
            avg_values.append(np.mean(vals) if vals else 0.0)

        ax.plot(pcts, avg_values, marker='s', markersize=5, color='black',
                linewidth=2, label='Average', zorder=10)

        ax.set_xlabel("Training Progress (%)", fontsize=10)
        ax.set_ylabel(f"{METRIC_LABELS.get(metric, metric)} (%)", fontsize=10)
        ax.set_title(METRIC_LABELS.get(metric, metric), fontsize=11)
        ax.set_xticks(pcts)
        ax.set_xticklabels([f"{p}%" for p in pcts])
        ax.grid(True, alpha=0.3)

    axes[-1].legend(fontsize=7, loc='lower right')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
