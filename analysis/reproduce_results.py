"""Rebuild report tables and figures from public evidence, without data or weights."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from compile_results import ROOT, build_runs, formal_log, plot_report, write_csv, write_tables
from plot_boundary import main as plot_boundary


def confirmation():
    folder = ROOT / "results/local_audit"
    saved = json.loads((folder / "local_confirmation.json").read_text())
    with (folder / "validation_predictions.csv").open(newline="") as stream:
        predictions = list(csv.DictReader(stream))
    indices = np.load(ROOT / "data/splits/val_indices.npy", allow_pickle=False)
    if ([int(r["validation_position"]) for r in predictions] != list(range(5000))
            or [int(r["cifar_train_index"]) for r in predictions] != indices.tolist()):
        raise ValueError("Saved predictions do not match the fixed ordered validation split")
    labels = np.array([int(r["true_label"]) for r in predictions])
    rows, curves = [], {}
    for item in saved["runs"]:
        arm = item["arm"]
        path = folder / "raw_logs" / (item["run_id"] + ".csv")
        logs = formal_log(path)
        if hashlib.sha256(path.read_bytes()).hexdigest() != item["log_sha256"]:
            raise ValueError(f"Saved log hash disagrees for {arm}")
        accuracy = [float(row["val_accuracy"]) for row in logs]
        if any(not math.isfinite(x) or not 0 <= x <= 1 for x in accuracy):
            raise ValueError(f"Invalid accuracy in {arm}")
        best = max(logs, key=lambda row: float(row["val_accuracy"]))
        pred = np.array([int(r[arm]) for r in predictions])
        if any(np.any((a < 0) | (a >= 100)) for a in (labels, pred)):
            raise ValueError("Predicted classes must be integers in [0,99]")
        matrix = np.bincount(100 * labels + pred, minlength=10000).reshape(100, 100)
        top1 = 100 * np.trace(matrix) / len(pred)
        macro_f1 = np.mean(2 * matrix.diagonal() / np.maximum(matrix.sum(0) + matrix.sum(1), 1))
        if (abs(top1 - 100 * float(best["val_accuracy"])) > 1e-8
                or abs(macro_f1 - item["cpu_checkpoint_macro_f1"]) > 1e-12):
            raise ValueError(f"Predictions and reported metrics disagree for {arm}")
        rows.append(dict(method={"e1": "E1", "e4": "E4", "e4r": "E4-R"}[arm],
                         run_id=item["run_id"], host="M5 Max", seed=43,
                         epochs=200, best_epoch=int(best["epoch"]),
                         best_val_top1_percent=float(top1),
                         last10_val_top1_mean_percent=100 * float(np.mean(accuracy[-10:])),
                         val_macro_f1=float(macro_f1)))
        curves[rows[-1]["method"]] = accuracy
    return rows, curves


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=ROOT / "outputs/reproduced")
    args = parser.parse_args()
    out = args.out_dir.resolve()
    # Never overwrite the committed evidence tree through the aggregate entry point.
    evidence = (ROOT / "results").resolve()
    if out == evidence or evidence in out.parents or out == ROOT:
        parser.error("Choose a separate output directory, for example outputs/reproduced")
    tables, figures = out / "tables", out / "figures"
    tables.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)
    runs = build_runs()
    write_tables(runs, tables)
    plot_report(runs, tables, figures)
    plot_boundary(out)
    local, curves = confirmation()
    write_csv(tables / "local_confirmation.csv", local)
    lines = ["# Same-host confirmation", "", "M5 Max, seed 43, fixed validation split.", "",
             "| Method | Best validation Top-1 (%) | Last 10 mean (%) | Macro-F1 |",
             "|---|---:|---:|---:|"]
    for row in local:
        lines.append(f'| {row["method"]} | {row["best_val_top1_percent"]:.2f} | '
                     f'{row["last10_val_top1_mean_percent"]:.3f} | {row["val_macro_f1"]:.5f} |')
    lines += ["", "Recomputed from public training logs and saved validation predictions.",
              "No checkpoint was reloaded by this command; no official test result is claimed."]
    (tables / "local_confirmation.md").write_text("\n".join(lines) + "\n")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(9, 4.5), layout="constrained")
    for name, values in curves.items():
        ax.plot(range(1, 201), 100 * np.array(values), label=name)
    ax.set(xlabel="Epoch", ylabel="Validation Top-1 (%)", title="Same-host confirmation · M5 Max · seed 43")
    ax.grid(alpha=.2); ax.legend()
    fig.savefig(figures / "local_confirmation_curves.png", dpi=170)
    plt.close(fig)
    inputs = [ROOT / row["log_path"] for row, _ in runs]
    inputs += list((ROOT / "results/local_audit/raw_logs").glob("*.csv"))
    inputs += [ROOT / "results/local_audit/validation_predictions.csv",
               ROOT / "results/member_d/boundary_samples.csv", ROOT / "results/member_d/boundary_summary.json"]
    manifest = {"status": "passed", "historical_runs": len(runs), "confirmation_runs": len(local),
                "dataset_required": False, "checkpoints_required": False,
                "inputs_sha256": {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
                                  for path in inputs}}
    (out / "reproduction_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Reproduced {len(runs)} historical + {len(local)} confirmation runs and boundary figures in {out}")


if __name__ == "__main__":
    main()
