"""Validate a saved checkpoint or evaluate a frozen model on official CIFAR-100 test."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from models.resnet import build_resnet18
from scripts.evaluate_member_c_val import evaluate


ROOT = Path(__file__).resolve().parent


def verify_checkpoint_log(state, log, require_formal=False):
    with log.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    epochs = [int(row["epoch"]) for row in rows]
    if not epochs or epochs != list(range(1, len(epochs) + 1)):
        raise ValueError("Expected a continuous, ordered training log starting at epoch 1")
    accuracies = [float(row["val_accuracy"]) for row in rows]
    checkpoint_accuracy = float(state["best_val_accuracy"])
    if any(not math.isfinite(value) or not 0 <= value <= 1
           for value in [*accuracies, checkpoint_accuracy]):
        raise ValueError("Expected finite validation accuracy values in [0,1]")
    best = max(rows, key=lambda row: float(row["val_accuracy"]))
    if int(best["epoch"]) != state["epoch"] or abs(float(best["val_accuracy"]) - state["best_val_accuracy"]) > 1e-10:
        raise ValueError("Checkpoint and earliest best validation epoch disagree")
    if require_formal and (len(rows) != state["config"]["training"]["epochs"]
                           or any(int(row["train_samples"]) != 45000 for row in rows)):
        raise ValueError("Official test requires complete formal training with 45,000 images per epoch")


def load_split(data_root, split_dir, split_name):
    data = datasets.CIFAR100(str(data_root), train=split_name == "validation",
                              transform=transforms.ToTensor(), download=False)
    if split_name == "test":
        if len(data) != 10000:
            raise ValueError("Expected 10,000 official CIFAR-100 test images")
        return data
    indices = np.load(split_dir / "val_indices.npy", allow_pickle=False)
    if (indices.shape != (5000,) or not np.issubdtype(indices.dtype, np.integer)
            or np.unique(indices).size != 5000 or indices.min() < 0 or indices.max() >= len(data)):
        raise ValueError("Expected 5,000 valid, distinct indices for fixed validation")
    return Subset(data, indices)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--log", type=Path, help="Training CSV used for checkpoint selection")
    parser.add_argument("--split", choices=("validation", "test"), default="validation")
    parser.add_argument("--data-root", type=Path, default=ROOT / "data/cifar100")
    parser.add_argument("--split-dir", type=Path, default=ROOT / "data/splits")
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    if args.batch_size < 1 or args.num_workers < 0:
        parser.error("Batch size must be positive and workers must be nonnegative")
    if args.split == "test" and not args.log:
        parser.error("Official test requires --log from the frozen formal run")
    device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    if device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA is unavailable")
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    if args.log:
        verify_checkpoint_log(state, args.log, require_formal=args.split == "test")
    model = build_resnet18().to(device)
    model.load_state_dict(state["model_state_dict"])
    model.eval()
    data = load_split(args.data_root, args.split_dir, args.split)
    loader = DataLoader(data, batch_size=args.batch_size, num_workers=args.num_workers,
                        shuffle=False, pin_memory=device == "cuda")
    n, accuracy, macro_f1, loss = evaluate(model, loader, device)
    result = {
        "run_id": args.run_id, "split": args.split, "samples": n, "top1_percent": 100 * accuracy,
        "macro_f1": macro_f1, "loss": loss, "checkpoint_epoch": state["epoch"],
        "checkpoint_logged_validation_top1_percent": 100 * state["best_val_accuracy"],
        "checkpoint_sha256": hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
        "fixed_validation_indices_sha256": hashlib.sha256((args.split_dir / "val_indices.npy").read_bytes()).hexdigest(),
        "augmentation": state["config"]["augmentation"], "seed": state["config"]["experiment"]["seed"],
        "device": device, "torch": str(torch.__version__),
    }
    rendered = json.dumps(result, indent=2, allow_nan=False) + "\n"
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
