"""Summarize C's formal runs from logs/*.csv into results/member_c/summary.{json,csv}.

Reports validation-set numbers only (seed 42, fixed 5k split). Never reads the test set.
"""
import argparse
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNS = [("E3", "seam_aware", "c_e3_k4_s42"), ("E4", "seam_dp", "c_e4_k4_s42"),
        ("E4-R", "seam_random", "c_e4r_k4_s42")]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--suffix", default="")
    args = parser.parse_args()
    rows = []
    for experiment, policy, run in RUNS:
        name = run + ("_" + args.suffix if args.suffix else "")
        log = ROOT / "logs" / f"{name}.csv"
        if not log.exists():
            print(f"missing: {log}")
            continue
        metrics = list(csv.DictReader(log.open()))
        best = max(metrics, key=lambda x: float(x["val_accuracy"]))  # first max wins ties
        rows.append({
            "run_id": name, "experiment": experiment, "policy": policy, "seed": 42,
            "epochs_logged": len(metrics),
            "best_val_top1_percent": round(100 * float(best["val_accuracy"]), 2),
            "best_epoch": int(best["epoch"]), "val_loss_at_best": float(best["val_loss"]),
            "final_val_top1_percent": round(100 * float(metrics[-1]["val_accuracy"]), 2),
            "mean_epoch_seconds": sum(float(x["epoch_seconds"]) for x in metrics) / len(metrics),
            "sum_epoch_seconds": sum(float(x["epoch_seconds"]) for x in metrics),
            "log_path": f"logs/{name}.csv", "checkpoint_path": f"checkpoints/{name}/best.pth",
        })
    out = ROOT / "results/member_c"
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps(rows, indent=2))
    if rows:
        with (out / "summary.csv").open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    print(json.dumps(rows, indent=2))
    print("NOTE: seed 42, validation split only; not test-set results, not a multi-seed mean.")


if __name__ == "__main__":
    main()
