"""Synchronized per-batch augmentation timing for C's arms (and the E2 reference).

Follows B's timing protocol (member_b_diagnostics.py): batch 128, float32, 10 warm-up
and 50 timed calls, CUDA synchronize around each call, augmentation only (no data
loading, no model). Writes results/member_c/aug_timing.json.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from augmentations import cutmix_batch  # noqa: E402

sys.path.insert(0, str(ROOT / "scripts"))
from member_c_suite import ARMS  # noqa: E402

# E2 reference plus every C arm (main, controls, ablations), in ARMS order, no duplicates.
CONFIGS = list(dict.fromkeys(["b_dual_edge_k4_s42"] + [stem for _, stem, _ in ARMS.values()]))
KEYS = {"alpha": "alpha", "p": "p", "k": "k", "gamma": "gamma", "beta": "beta",
        "seam_radius": "seam_radius", "seam_mu": "seam_mu", "seam_min_width": "seam_min_width",
        "seam_area_budget": "seam_area_budget", "seam_random_mu": "seam_random_mu"}


def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repetitions", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--out", default=str(ROOT / "results/member_c/aug_timing.json"))
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(0)
    images = torch.rand(args.batch_size, 3, 32, 32, device=device)
    labels = torch.randint(0, 100, (args.batch_size,), device=device)
    results = []
    for name in CONFIGS:
        aug = yaml.safe_load((ROOT / "configs" / f"{name}.yaml").read_text())["augmentation"]
        kwargs = {dst: aug[src] for src, dst in KEYS.items() if src in aug}
        for _ in range(args.warmup):
            cutmix_batch(images, labels, policy=aug["policy"], **kwargs)
        times = []
        for _ in range(args.repetitions):
            sync(device)
            start = time.perf_counter()
            cutmix_batch(images, labels, policy=aug["policy"], **kwargs)
            sync(device)
            times.append((time.perf_counter() - start) * 1000)
        times.sort()
        record = {"config": name, "policy": aug["policy"], "p": aug["p"],
                  "augmentation_ms_per_batch": sum(times) / len(times),
                  "median_ms": times[len(times) // 2]}
        results.append(record)
        print(json.dumps(record), flush=True)

    out = {"device": str(device),
           "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
           "torch": torch.__version__, "dtype": "torch.float32", "batch_size": args.batch_size,
           "warmup": args.warmup, "repetitions": args.repetitions,
           "timing": "synchronized, complete augmentation only (timing p = config p), "
                     "excludes data loading and model", "results": results}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
