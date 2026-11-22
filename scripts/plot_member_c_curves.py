"""Validation Top-1 curves for E1 (A), E2 K=4 (B), E3 / E4 / E4-R (C), seed 42.

Reads committed 200-epoch CSV logs only (5k validation split; no test set).
Usage: python scripts/plot_member_c_curves.py [out.png]
"""
import csv
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
A = ROOT / "Baseline Implementation & Training Infrastructure (Member A)/results/baselines"
RUNS = [  # label, csv, colour (fixed categorical order of the reference palette)
    ("E1 random (A, A40)", A / "e1_final_s42.csv", "#2a78d6"),
    ("E2 dual_edge K=4 (B, A800)", ROOT / "results/member_b/raw_logs/b_dual_edge_k4_s42_seq.csv", "#eb6834"),
    ("E3 seam_aware (C, 4090)", ROOT / "results/member_c/raw_logs/c_e3_k4_s42.csv", "#1baf7a"),
    ("E4 seam_dp (C, 4090)", ROOT / "results/member_c/raw_logs/c_e4_k4_s42.csv", "#eda100"),
    ("E4-R seam_random (C, 4090)", ROOT / "results/member_c/raw_logs/c_e4r_k4_s42.csv", "#e87ba4"),
]
INK, MUTED, GRID, SURFACE = "#1f1f1e", "#6b6a64", "#e4e3df", "#fcfcfb"


def load(path):
    rows = list(csv.DictReader(open(path)))
    return [int(r["epoch"]) for r in rows], [100 * float(r["val_accuracy"]) for r in rows]


def style(ax):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def main():
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "results/member_c/validation_curves.png"
    data = [(label, *load(path), colour) for label, path, colour in RUNS]
    fig, (left, right) = plt.subplots(1, 2, figsize=(11, 4.2), facecolor=SURFACE,
                                      gridspec_kw={"width_ratios": [1.15, 1]})
    for ax in (left, right):
        style(ax)
    for label, ep, acc, colour in data:
        left.plot(ep, acc, color=colour, linewidth=1.4, label=label)
        right.plot(ep[150:], acc[150:], color=colour, linewidth=1.6)
    # End labels, nudged apart vertically so close final values do not overlap.
    gap = 0.28
    ends = sorted(((acc[-1], label.split(" ")[0], ep[-1]) for label, ep, acc, _ in data), reverse=True)
    placed = []
    for value, short, last in ends:
        y = value if not placed or placed[-1] - value >= gap else placed[-1] - gap
        placed.append(y)
        right.annotate(f"{short}  {value:.2f}", xy=(last, value), xytext=(last + 1.5, y),
                       textcoords="data", va="center", fontsize=8.5, color=INK)
    left.set_title("Validation Top-1, all 200 epochs", loc="left", fontsize=11, color=INK)
    right.set_title("Epochs 151-200 (labels: final-epoch value)", loc="left", fontsize=11, color=INK)
    left.set_ylim(0, 85)
    right.set_xlim(150, 214)
    for ax in (left, right):
        ax.set_xlabel("epoch", color=MUTED, fontsize=9)
    left.set_ylabel("val Top-1 (%)", color=MUTED, fontsize=9)
    left.legend(frameon=False, fontsize=8.5, loc="lower right", labelcolor=INK)
    fig.text(0.01, 0.005, "seed 42, fixed 5k validation split from the official train set; single seed, "
             "different GPUs per member; not test-set results.", fontsize=8, color=MUTED)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    print(out)


if __name__ == "__main__":
    main()
