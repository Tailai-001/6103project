"""Summarize C's formal runs from logs/*.csv into results/member_c/summary.{json,csv}.

Reports validation-set numbers only (seed 42, fixed 5k split). Never reads the test set.
"""
import argparse
import csv
import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from member_c_suite import ARMS  # noqa: E402

RUNS = [(exp, stem, run) for exp, stem, run in ARMS.values()]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--suffix", default="")
    args = parser.parse_args()
    rows = []
    evaluation = {}
    eval_path = ROOT / "results/member_c/checkpoint_validation.json"
    if eval_path.exists():
        evaluation = {r["run_id"]: r for r in json.loads(eval_path.read_text())}
    for experiment, stem, run in RUNS:
        name = run + ("_" + args.suffix if args.suffix else "")
        log = ROOT / "logs" / f"{name}.csv"
        if not log.exists():
            continue
        aug = yaml.safe_load((ROOT / "configs" / f"{stem}.yaml").read_text())["augmentation"]
        metrics = list(csv.DictReader(log.open()))
        best = max(metrics, key=lambda x: float(x["val_accuracy"]))  # first max wins ties
        rows.append({
            "run_id": name, "experiment": experiment, "policy": aug["policy"], "seed": 42,
            "epochs_logged": len(metrics),
            "best_val_top1_percent": round(100 * float(best["val_accuracy"]), 2),
            "best_epoch": int(best["epoch"]), "val_loss_at_best": float(best["val_loss"]),
            "final_val_top1_percent": round(100 * float(metrics[-1]["val_accuracy"]), 2),
            "last10_mean_val_top1_percent": round(sum(100 * float(x["val_accuracy"]) for x in metrics[-10:])
                                                  / len(metrics[-10:]), 3),
            "mean_epoch_seconds": sum(float(x["epoch_seconds"]) for x in metrics) / len(metrics),
            "sum_epoch_seconds": sum(float(x["epoch_seconds"]) for x in metrics),
            "val_macro_f1": evaluation.get(name, {}).get("val_macro_f1", ""),
            "config": f"configs/{stem}.yaml",
            "augmentation": {k: v for k, v in aug.items() if k != "alpha" and k != "p"},
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
