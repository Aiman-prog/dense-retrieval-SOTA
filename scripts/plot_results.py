"""
GRASS n_das Ablation — Recall@1000 by Domain Group + Training Time.

Compares GRASS runs with different n_das values (challengers per batch).
Fill in the placeholder values (None → float) then run:
    python scripts/plot_results.py

Domain groups:
  stackoverflow                        → "StackExchange"
  mean(leetcode, aops)                 → "Coding"
  mean(theoremqa_theorems,
       theoremqa_questions)            → "Theorem"

All Recall@1000 values are in [0, 1].
Training time is wall-clock hours for that specific run (mining + fine-tuning).
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# ============================================================
# ✏️  FILL IN YOUR VALUES HERE
# ============================================================

# n_das values you ran
N_DAS = [1, 4, 8, 16, 32]

# Recall@1000 per n_das setting, per domain group.
# Each inner list: [stackoverflow, coding_avg, theorem_avg]
#   coding_avg  = mean(leetcode, aops)
#   theorem_avg = mean(theoremqa_theorems, theoremqa_questions)
RECALL = {
    #       stackoverflow  coding  theorem
    1:      [0.63,         0.41,   0.35],
    4:      [0.67,         0.41,   0.36],
    8:      [0.68,         0.44,   0.37],
    16:     [0.68,         0.46,   0.39],
    32:     [0.68,         0.45,   0.39],
}

# Wall-clock training hours for each n_das run (mining + Tevatron fine-tuning).
# Higher n_das = more queries mined per batch = longer runtime.
TRAIN_HOURS = {
    1:  8.27,   # e.g. 4.2
    4:  11.20,   # e.g. 5.8
    8:  15.46,   # e.g. 7.1
    16: 13.42,   # e.g. 9.5
    32: 10.16,   # e.g. 14.0
}

# ============================================================
# Plot config
# ============================================================

DOMAIN_LABELS  = ["StackExchange", "Coding", "Theorem"]
DOMAIN_COLORS  = ["#4C72B0", "#55A868", "#C44E52"]   # blue, green, red
TIME_COLOR     = "#DD8800"
BAR_WIDTH      = 0.22
GROUP_PADDING  = 0.12   # extra gap between model groups

FIG_W, FIG_H   = 11, 5.5
OUTPUT_PATH    = "results/recall_vs_time.pdf"   # also saves .png

# ============================================================
# Build plot
# ============================================================

def _resolve(v, fallback=0.0):
    """Return float or fallback when value is None (unfilled placeholder)."""
    return v if v is not None else fallback


fig, ax1 = plt.subplots(figsize=(FIG_W, FIG_H))
ax2 = ax1.twinx()

n_settings = len(N_DAS)
n_domains  = len(DOMAIN_LABELS)

# x positions: one cluster per n_das value, bars offset per domain
group_width = n_domains * BAR_WIDTH + GROUP_PADDING
x_centers   = np.arange(n_settings) * group_width
offsets     = (np.arange(n_domains) - n_domains / 2 + 0.5) * BAR_WIDTH

# --- Recall bars ---
for d_idx, (label, color) in enumerate(zip(DOMAIN_LABELS, DOMAIN_COLORS)):
    values = [_resolve(RECALL[n][d_idx]) for n in N_DAS]
    bars   = ax1.bar(
        x_centers + offsets[d_idx],
        values,
        width=BAR_WIDTH,
        color=color,
        alpha=0.85,
        label=label,
        zorder=3,
    )
    # Annotate each bar with its value (skip zeros from unfilled placeholders)
    for bar, v in zip(bars, values):
        if v > 0:
            ax1.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.008,
                f"{v:.2f}",
                ha='center', va='bottom',
                fontsize=7.5, color='#333333',
            )

# --- Training time line (right axis) ---
time_vals = [_resolve(TRAIN_HOURS[n]) for n in N_DAS]
ax2.plot(
    x_centers, time_vals,
    color=TIME_COLOR, linewidth=1.8,
    linestyle='--', marker='D',
    markersize=7, zorder=4,
    label="Training time (h)",
)
for x, t in zip(x_centers, time_vals):
    if t > 0:
        ax2.text(
            x, t + max(time_vals) * 0.03,
            f"{t:.1f}h",
            ha='center', va='bottom',
            fontsize=8, color=TIME_COLOR, fontweight='bold',
        )

# --- Axes formatting ---
ax1.set_ylabel("Recall@1000", fontsize=12)
ax1.set_ylim(0, 1.12)
ax1.set_xticks(x_centers)
ax1.set_xticklabels([f"n_das={n}" for n in N_DAS], fontsize=10)
ax1.yaxis.grid(True, linestyle='--', alpha=0.5, zorder=0)
ax1.set_axisbelow(True)

max_time = max(t for t in time_vals if t > 0) if any(t > 0 for t in time_vals) else 20
ax2.set_ylabel("Training time (hours)", fontsize=12, color=TIME_COLOR)
ax2.tick_params(axis='y', labelcolor=TIME_COLOR)
ax2.set_ylim(0, max_time * 1.35)

ax1.set_xlabel("Challengers per batch (n_das)", fontsize=12, labelpad=8)
ax1.set_title(
    "GRASS n_das Ablation — Recall@1000 by Domain Group",
    fontsize=13, fontweight='bold', pad=12,
)

# --- Legend ---
domain_patches = [
    mpatches.Patch(color=c, alpha=0.85, label=l)
    for c, l in zip(DOMAIN_COLORS, DOMAIN_LABELS)
]
time_line = plt.Line2D(
    [0], [0], color=TIME_COLOR, linewidth=1.8,
    linestyle='--', marker='D', markersize=7,
    label="Training time (h)",
)
ax1.legend(
    handles=domain_patches + [time_line],
    loc='upper left', fontsize=9,
    framealpha=0.9, edgecolor='#cccccc',
)

plt.tight_layout()

import os
os.makedirs("results", exist_ok=True)
fig.savefig(OUTPUT_PATH, bbox_inches='tight', dpi=150)
fig.savefig(OUTPUT_PATH.replace('.pdf', '.png'), bbox_inches='tight', dpi=150)
print(f"Saved → {OUTPUT_PATH}")
plt.show()
