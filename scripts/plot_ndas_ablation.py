"""
Plot NDCG@10 vs n_das for bandit ablation, with random n_das=8 baseline.

Parses SLURM .out logs automatically or accepts manual values.

Usage (auto):   python scripts/plot_ndas_ablation.py --log_dir logs/
Usage (manual): python scripts/plot_ndas_ablation.py --manual 2:0.21 4:0.24 8:0.27 16:0.28 32:0.27 --random 0.23 --ance 0.168
"""
import argparse
import re
import sys
from pathlib import Path


def parse_ndcg_from_log(log_path):
    text = Path(log_path).read_text()
    matches = re.findall(r'Final Mean NDCG@10:\s*([\d.]+)', text)
    return float(matches[-1]) if matches else None


def parse_ndas_from_log(log_path):
    text = Path(log_path).read_text()
    m = re.search(r'N_DAS=(\d+)', text)
    return int(m.group(1)) if m else None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--log_dir', type=str, default='logs')
    parser.add_argument('--manual', nargs='+', default=None,
                        help='n_das:ndcg pairs e.g. 2:0.21 4:0.24 8:0.27')
    parser.add_argument('--random', type=float, default=None,
                        help='NDCG@10 for GRASS-random at n_das=8')
    parser.add_argument('--ance', type=float, default=None,
                        help='ANCE baseline NDCG@10')
    parser.add_argument('--out', type=str, default='logs_cluster/ndas_ablation.png')
    args = parser.parse_args()

    try:
        import matplotlib.pyplot as plt
        import matplotlib
        matplotlib.rcParams.update({
            'font.family': 'serif', 'font.size': 11,
            'axes.spines.top': False, 'axes.spines.right': False,
        })
    except ImportError:
        print("pip install matplotlib")
        sys.exit(1)

    if args.manual:
        bandit = {int(k): float(v) for item in args.manual for k, v in [item.split(':')]}
    else:
        bandit = {}
        for log in sorted(Path(args.log_dir).glob('grass_async_*.out')):
            text = log.read_text()
            if 'SELECTION=random' in text:
                continue
            ndas = parse_ndas_from_log(log)
            ndcg = parse_ndcg_from_log(log)
            if ndas and ndcg:
                bandit[ndas] = ndcg
                print(f"  n_das={ndas} NDCG@10={ndcg:.4f}  ({log.name})")
        if not bandit:
            print("No completed bandit logs found. Use --manual.")
            sys.exit(1)

    ndas_vals = sorted(bandit.keys())
    ndcg_vals = [bandit[n] for n in ndas_vals]

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(ndas_vals, ndcg_vals, marker='o', color='#1f77b4',
            linewidth=2, markersize=7, label='GRASS-bandit')

    if args.random is not None:
        ax.axhline(args.random, color='#ff7f0e', linestyle='--',
                   linewidth=1.5, label='GRASS-random (n\_das=8)')
    if args.ance is not None:
        ax.axhline(args.ance, color='#2ca02c', linestyle=':',
                   linewidth=1.5, label='ANCE baseline')

    ax.set_xlabel('$n_{\\mathrm{das}}$ (queries mined per cycle)')
    ax.set_ylabel('NDCG@10 (BRIGHT, mean over 12 domains)')
    ax.set_title('GRASS: Coverage vs Retrieval Quality')
    ax.set_xticks(ndas_vals)
    ax.legend(frameon=False)
    ax.grid(axis='y', alpha=0.3)

    Path(args.out).parent.mkdir(exist_ok=True, parents=True)
    fig.tight_layout()
    fig.savefig(args.out, dpi=150)
    print(f"Saved: {args.out}")


if __name__ == '__main__':
    main()
