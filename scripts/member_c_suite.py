"""Register and run C's jobs (E3 / E4 / E4-R, controls, ablations) through the single training entry point.

Adapted from B's scripts/member_b_suite.py: every job calls the shared train.py with a
committed config, is recorded in experiments/registry.csv before it starts, and existing
runs are never overwritten. Jobs run sequentially (B measured that concurrent runs on one
GPU make walltimes uninterpretable).

  --phase smoke   E4, 2 epochs x 2 batches        (pipeline check, minutes)
  --phase pilot   E4, 5 full epochs               (prints a 200-epoch time estimate)
  --phase formal  200 epochs for the selected arms
"""
import argparse
import csv
import hashlib
import json
import string
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ARMS = {  # arm -> (experiment id, config stem, run name)
    # main runs
    "e3": ("E3", "c_seam_aware_k4_s42", "c_e3_k4_s42"),
    "e4": ("E4", "c_seam_dp_k4_s42", "c_e4_k4_s42"),
    "e4r": ("E4-R", "c_seam_random_k4_s42", "c_e4r_k4_s42"),
    # same-hardware controls (C's GPU)
    "ctrl_e2": ("E2 control", "c_ctrl_dual_edge_k4_s42", "c_ctrl_e2_k4_s42"),
    "ctrl_e1": ("E1 control", "c_ctrl_random_s42", "c_ctrl_e1_s42"),
    # single-change ablations around the main E3 / E4 settings
    "e3_b05": ("E3 ablation", "c_abl_seam_aware_k4_b05_s42", "c_e3_b05_k4_s42"),
    "e3_b2": ("E3 ablation", "c_abl_seam_aware_k4_b2_s42", "c_e3_b2_k4_s42"),
    "e4_r1": ("E4 ablation", "c_abl_seam_dp_k4_r1_s42", "c_e4_r1_k4_s42"),
    "e4_r3": ("E4 ablation", "c_abl_seam_dp_k4_r3_s42", "c_e4_r3_k4_s42"),
    "e4_mu0": ("E4 ablation", "c_abl_seam_dp_k4_mu0_s42", "c_e4_mu0_k4_s42"),
    "e4_mu02": ("E4 ablation", "c_abl_seam_dp_k4_mu02_s42", "c_e4_mu02_k4_s42"),
}
MAIN_ARMS = ["e3", "e4", "e4r"]
FORMAL_EPOCHS = 200


def git_commit(root):
    out = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True)
    if out.returncode != 0:
        return ""
    dirty = subprocess.run(["git", "status", "--porcelain", "--", "augmentations", "train.py",
                            "configs", "models", "data"], cwd=root, capture_output=True, text=True)
    return out.stdout.strip() + ("-dirty" if dirty.stdout.strip() else "")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=["smoke", "pilot", "formal"], default="formal")
    parser.add_argument("--arms", nargs="+", choices=list(ARMS), default=MAIN_ARMS)
    parser.add_argument("--suffix", default="")
    parser.add_argument("--dry-run", action="store_true", help="print the commands only")
    args = parser.parse_args()
    if any(c not in string.ascii_letters + string.digits + "-_" for c in args.suffix):
        parser.error("suffix may contain only letters, digits, hyphens and underscores")

    root = Path(__file__).resolve().parents[1]
    registry = root / "experiments/registry.csv"
    with registry.open() as f:
        fields = next(csv.reader(f))
    split_hash = ";".join(f"{p.name}:{hashlib.sha256(p.read_bytes()).hexdigest()}"
                          for p in sorted((root / "data/splits").glob("*.npy")))
    commit = git_commit(root)
    if commit.endswith("-dirty") and args.phase == "formal" and not args.dry_run:
        sys.exit("Uncommitted changes in training/augmentation code; commit before formal runs.")

    if args.phase == "formal":
        jobs = [(ARMS[arm][2], arm, []) for arm in args.arms]
    else:
        extra = ["--epochs", "2", "--max-batches", "2"] if args.phase == "smoke" else ["--epochs", "5"]
        jobs = [(f"c_{args.phase}", "e4", extra)]

    import torch
    device_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    timing_path = root / "results/member_c/aug_timing.json"
    timing = json.loads(timing_path.read_text()) if timing_path.exists() else None

    codes = []
    for name, arm, extra in jobs:
        if args.suffix:
            name += "_" + args.suffix
        experiment, stem, _ = ARMS[arm]
        config_path = f"configs/{stem}.yaml"
        cmd = [sys.executable, "-u", "-B", "train.py", "--config", config_path, "--run-name", name, *extra]
        if args.dry_run:
            print(" ".join(cmd))
            continue
        with registry.open() as f:
            existing = list(csv.DictReader(f))
        if (any(x["run_id"] == name for x in existing) or (root / f"logs/{name}.csv").exists()
                or (root / f"logs/{name}.stdout.log").exists() or (root / f"checkpoints/{name}").exists()):
            raise FileExistsError(f"Existing run {name}; preserve it and pass a fresh --suffix")

        row = {k: "" for k in fields}
        import yaml
        policy = yaml.safe_load((root / config_path).read_text())["augmentation"]["policy"]
        row.update(run_id=name, experiment=experiment, policy=policy, seed=42,
                   git_commit=commit, config_path=config_path, split_sha256=split_hash,
                   environment=f"checkpoints/{name}/environment.json", device=device_name,
                   started_at=datetime.now(timezone.utc).isoformat(), status="running",
                   log_uri=f"logs/{name}.csv", checkpoint_uri=f"checkpoints/{name}/best.pth",
                   notes=f"phase={args.phase}; overrides={extra}; workers=1; validation only; "
                         f"seam parameters PROVISIONAL (not yet selected on validation)")
        if timing:
            for rec in timing["results"]:
                if rec["config"] == stem:
                    row["augmentation_ms_per_batch"] = rec["augmentation_ms_per_batch"]
                    row["notes"] += f"; augmentation timing isolated before training on {timing['gpu']}"
        with registry.open("a", newline="") as f:
            csv.DictWriter(f, fieldnames=fields).writerow(row)

        print(f"[{datetime.now():%H:%M:%S}] start {name}: {' '.join(cmd[3:])}", flush=True)
        start = time.perf_counter()
        (root / "logs").mkdir(exist_ok=True)
        with (root / f"logs/{name}.stdout.log").open("x") as log:
            result = subprocess.run(cmd, cwd=root, stdout=log, stderr=subprocess.STDOUT)
        row["status"] = "completed" if result.returncode == 0 else "failed"
        row["total_training_seconds"] = time.perf_counter() - start
        row["notes"] += f"; exit_code={result.returncode}"
        log_path = root / f"logs/{name}.csv"
        metrics = list(csv.DictReader(log_path.open())) if log_path.exists() else []
        if metrics:
            row["val_top1_percent"] = 100 * max(float(x["val_accuracy"]) for x in metrics)

        with registry.open() as f:
            rows = [row if x["run_id"] == name else x for x in csv.DictReader(f)]
        temp = registry.with_suffix(".tmp")
        with temp.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        temp.replace(registry)
        print(json.dumps(row), flush=True)

        if args.phase == "pilot" and metrics:
            # Skip epoch 1 (warm-up: cudnn autotune, worker start-up).
            steady = [float(x["epoch_seconds"]) for x in metrics[1:]] or [float(metrics[0]["epoch_seconds"])]
            per_epoch = sum(steady) / len(steady)
            print(f"Pilot E4: {per_epoch:.1f} s/epoch -> 200 epochs ~ {per_epoch * FORMAL_EPOCHS / 3600:.2f} h "
                  f"per E4 run (E4-R similar, E3 faster)", flush=True)
        codes.append(result.returncode)
        if result.returncode != 0:
            print(f"{name} failed (exit {result.returncode}); see logs/{name}.stdout.log. "
                  f"Stopping; later arms were not started.", flush=True)
            break

    if any(codes):
        raise SystemExit(next(c for c in codes if c))


if __name__ == "__main__":
    main()
