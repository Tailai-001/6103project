"""Build report table and metric distribution figure from D's raw boundary audit."""
import csv
import json
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from compile_results import FIGURES, RESULTS, TABLES, format_value, write_csv


def main():
    base = RESULTS / "member_d"
    summary = json.loads((base / "boundary_summary.json").read_text(encoding="utf-8"))
    samples = defaultdict(list)
    with (base / "boundary_samples.csv").open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            samples[row["arm"]].append(row)

    table_rows = []
    for arm, stats in summary["arms"].items():
        seam = stats["seam"]
        if len(samples[arm]) != stats["samples"]:
            raise ValueError(f"{arm}: raw sample count disagrees with summary")
        row = dict(
            method=arm, samples=stats["samples"], na_with_all=seam["all"]["na_images"],
            na_without_any_padding=seam["without_any_padding"]["na_images"],
            rgb_all=seam["all"]["pair_weighted_rgb"],
            rgb_without_double_padding=seam["without_double_padding"]["pair_weighted_rgb"],
            rgb_without_any_padding=seam["without_any_padding"]["pair_weighted_rgb"],
            rgb_without_double_black=seam["without_double_black"]["pair_weighted_rgb"],
            pair_fraction_removed_any_padding=seam["without_any_padding"]["removed_pair_fraction"],
            area_ratio=stats["area_ratio"], boundary_length=stats["boundary_length"],
            fallback_rate=stats.get("fallback_rate"), bends=stats.get("bends"),
            edge_snap_rate=stats.get("edge_snap_rate_among_changed"),
        )
        table_rows.append(row)
    write_csv(TABLES / "d_boundary_quality.csv", table_rows)
    text = [
        "# Boundary quality", "",
        "Paired, forced-mix audit on CIFAR-100 training images. RGB values are weighted by eligible cross-mask pairs.",
        "Boundary length counts internal four-neighbor pairs. Area uses all inputs. Fallback and bends are over",
        "nonempty base rectangles. Edge snap is over changed masks. N/A indicates no applicable observations.", "",
        "| Method | RGB all | RGB excluding two padded endpoints | RGB excluding any padded endpoint | Area ratio | Boundary length | Fallback (%) | Bends | Edge snap (%) | N/A after any-padding exclusion |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in table_rows:
        ratios = [None if row[key] is None else 100 * row[key] for key in ("fallback_rate", "edge_snap_rate")]
        text.append(f'| {row["method"]} | {format_value(row["rgb_all"], 4)} | '
                    f'{format_value(row["rgb_without_double_padding"], 4)} | {format_value(row["rgb_without_any_padding"], 4)} | '
                    f'{format_value(row["area_ratio"], 3)} | {format_value(row["boundary_length"])} | '
                    f'{format_value(ratios[0])} | {format_value(row["bends"])} | {format_value(ratios[1])} | '
                    f'{row["na_without_any_padding"]} |')
    (TABLES / "d_boundary_quality.md").write_text("\n".join(text) + "\n", encoding="utf-8")

    figure, panels = plt.subplots(2, 2, figsize=(11, 6.8), layout="constrained")
    for ax, (title, value) in zip(panels.flat, (
        ("RGB discontinuity per image · all pairs", lambda r: float(r["sum_all"]) / int(r["count_all"]) if int(r["count_all"]) else None),
        ("RGB discontinuity per image · crop padding excluded",
         lambda r: float(r["sum_without_any_padding"]) / int(r["count_without_any_padding"]) if int(r["count_without_any_padding"]) else None),
        ("Actual replaced area (%)", lambda r: 100 * float(r["area_ratio"])),
        ("Final internal boundary length · pixel pairs", lambda r: int(r["count_all"]) or None),
    )):
        values = [[v for row in group if (v := value(row)) is not None] for group in samples.values()]
        boxes = ax.boxplot(values, showfliers=False, patch_artist=True)
        for patch, method in zip(boxes["boxes"], samples):
            patch.set_facecolor("#bc7450" if method in ("E4", "E4-R") else "#76a2b4")
        ax.set_xticks(range(1, len(samples) + 1), list(samples))
        ax.set_title(title, loc="left")
        ax.grid(axis="y", alpha=0.23)
    figure.suptitle(f'CIFAR-100 training split · {table_rows[0]["samples"]:,} paired inputs per method')
    FIGURES.mkdir(parents=True, exist_ok=True)
    figure.savefig(FIGURES / "d_boundary_distributions.png", dpi=170)
    plt.close(figure)


if __name__ == "__main__":
    main()
