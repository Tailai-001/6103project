"""Plot the paired augmentation audit; intervals are not classifier uncertainty."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def main():
    folder = ROOT / 'results/local_audit'
    evidence = json.loads((folder / 'paired_boundary_evidence.json').read_text())
    requests = [
        ('E3', 'E2', 'all_valid', 'E3 vs E2: padding excluded'),
        ('E4', 'E3', 'no_new_image_edge', 'E4 vs E3: padding + new edge excluded'),
        ('E4', 'E4-R', 'no_new_image_edge', 'E4 vs E4-R: padding + new edge excluded'),
        ('E4', 'E4-R', 'accepted_shape_close', 'E4 vs E4-R: accepted, similar shape'),
    ]
    selected = []
    for candidate, reference, subset, label in requests:
        row = next(r for r in evidence['comparisons']
                   if r['candidate'] == candidate and r['reference'] == reference
                   and r['subset'] == subset and r['variant'] == 'without_any_padding')
        selected.append((row, label))
    fig, ax = plt.subplots(figsize=(10, 4.2), layout='constrained')
    for y, (row, label) in enumerate(selected):
        point = row['difference']
        lo, hi = row['ci95']
        ax.errorbar(point, y, xerr=np.array([[point-lo], [hi-point]]),
                    fmt='o', color='#19698a', capsize=4, lw=2)
        ax.annotate(f"n={row['paired_samples']:,}", (hi, y),
                    xytext=(8, 0), textcoords='offset points', va='center', fontsize=9)
    ax.axvline(0, color='#777777', lw=1, ls='--')
    ax.set_yticks(range(len(selected)), [label for row, label in selected])
    ax.invert_yaxis()
    ax.set_xlim(-0.061, 0.004)
    ax.set_xlabel('Candidate minus reference: mean absolute RGB jump (lower favours candidate)')
    ax.set_title('Boundary improvement survives padding and edge sensitivity checks', loc='left', pad=18)
    ax.grid(axis='x', alpha=0.2)
    ax.spines[['top', 'right']].set_visible(False)
    fig.supxlabel('95% paired batch bootstrap intervals; 20 training-image batches. '
                  'Shape subset is observational. No classification inference.', fontsize=9)
    for suffix in ('png', 'pdf'):
        fig.savefig(folder / f'paired_boundary_intervals.{suffix}', dpi=180)
    plt.close(fig)


if __name__ == '__main__':
    main()
