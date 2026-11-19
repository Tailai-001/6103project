"""AI6103 shared training entry point (Member A).

Expected local modules:
  data/dataset.py: build_dataloaders(...)
  models/resnet.py: build_resnet18(num_classes=100)
  augmentations/__init__.py: cutmix_batch(...)

E0 is runnable with the 'none' policy. E1-E4 work as their policies are
implemented in augmentations.cutmix_batch; the training loop needs no changes.
"""

import argparse
import hashlib
import platform
import csv
import json
import math
import os
import random
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml

from augmentations import cutmix_batch
from data.dataset import build_dataloaders
from models.resnet import build_resnet18


CIFAR100_MEAN = (0.5071, 0.4867, 0.4408)
CIFAR100_STD = (0.2675, 0.2565, 0.2761)
LOG_FIELDS = [
    "epoch", "learning_rate", "train_loss", "train_accuracy",
    "val_loss", "val_accuracy", "train_seconds", "epoch_seconds",
    "train_samples",
]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def normalize_images(images: torch.Tensor) -> torch.Tensor:
    """Normalize after CutMix/seam processing, which requires RGB [0, 1]."""
    mean = images.new_tensor(CIFAR100_MEAN).view(1, 3, 1, 1)
    std = images.new_tensor(CIFAR100_STD).view(1, 3, 1, 1)
    return (images - mean) / std


def apply_augmentation(images, labels, aug_cfg):
    mixed, labels_a, labels_b, lam, metadata = cutmix_batch(
        images=images,
        labels=labels,
        policy=aug_cfg["policy"],
        alpha=aug_cfg.get("alpha", 1.0),
        p=aug_cfg.get("p", 0.5),
        k=aug_cfg.get("k", 4),
        gamma=aug_cfg.get("gamma", 0.5),
        beta=aug_cfg.get("beta", 0.0),
        seam_radius=aug_cfg.get("seam_radius", 0),
        seam_mu=aug_cfg.get("seam_mu", 0.0),
        seam_min_width=aug_cfg.get("seam_min_width", 2),
        seam_area_budget=aug_cfg.get("seam_area_budget", 0.1),
        seam_random_mu=aug_cfg.get("seam_random_mu", None),
    )

    n = images.size(0)
    if mixed.shape != images.shape:
        raise ValueError(f"Mixed images: expected {tuple(images.shape)}, got {tuple(mixed.shape)}")
    if mixed.device != images.device:
        raise ValueError("Augmentation must return images on the original device")
    if labels_a.shape != labels.shape or labels_b.shape != labels.shape:
        raise ValueError("Augmentation must return per-sample labels with shape [N]")
    if labels_a.device != labels.device or labels_b.device != labels.device:
        raise ValueError("Augmentation labels must stay on the original device")
    if not isinstance(lam, torch.Tensor) or lam.shape != (n,):
        raise ValueError(f"lam_actual must be a tensor of shape [{n}]")
    if lam.device != images.device:
        raise ValueError("lam_actual must stay on the original device")
    if not torch.isfinite(lam).all().item() or (lam < 0).any().item() or (lam > 1).any().item():
        raise ValueError("lam_actual must contain finite weights in [0,1]")
    if not isinstance(metadata, dict):
        raise ValueError("metadata must be a dictionary")
    return mixed, labels_a, labels_b, lam, metadata


def train_one_epoch(model, loader, optimizer, device, aug_cfg, max_batches=None):
    if max_batches is not None and max_batches < 1:
        raise ValueError("--max-batches must be >= 1")

    model.train()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    policy = aug_cfg["policy"]

    if device.type == "cuda":
        torch.cuda.synchronize()
    start = time.perf_counter()

    for batch_idx, (images, labels) in enumerate(loader):
        if max_batches is not None and batch_idx >= max_batches:
            break

        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        mixed, labels_a, labels_b, lam, _ = apply_augmentation(images, labels, aug_cfg)
        model_inputs = normalize_images(mixed)

        optimizer.zero_grad(set_to_none=True)
        logits = model(model_inputs)
        loss_a = F.cross_entropy(logits, labels_a, reduction="none")
        loss_b = F.cross_entropy(logits, labels_b, reduction="none")
        loss = (lam * loss_a + (1.0 - lam) * loss_b).mean()
        if not torch.isfinite(loss).item():
            raise FloatingPointError(f"Non-finite loss in batch {batch_idx}")
        loss.backward()
        optimizer.step()

        n = labels.size(0)
        total_loss += loss.detach().item() * n
        total_samples += n
        # Accuracy against the original labels is meaningful only for E0.
        if policy == "none":
            total_correct += (logits.argmax(dim=1) == labels).sum().item()

        if (batch_idx + 1) % 50 == 0:
            print(f"  Batch {batch_idx + 1}/{len(loader)} | loss={loss.item():.4f}", flush=True)

    if total_samples == 0:
        raise RuntimeError("No training batches were processed")
    if device.type == "cuda":
        torch.cuda.synchronize()

    return {
        "loss": total_loss / total_samples,
        "accuracy": (total_correct / total_samples) if policy == "none" else None,
        "seconds": time.perf_counter() - start,
        "samples": total_samples,
    }


@torch.no_grad()
def validate(model, loader, device):
    model.eval()
    total_loss = total_correct = total_samples = 0
    for images, labels in loader:
        images = normalize_images(images.to(device, non_blocking=True))
        labels = labels.to(device, non_blocking=True)
        logits = model(images)
        loss = F.cross_entropy(logits, labels, reduction="sum")
        total_loss += loss.item()
        total_correct += (logits.argmax(dim=1) == labels).sum().item()
        total_samples += labels.size(0)
    if total_samples == 0:
        raise RuntimeError("Validation dataset is empty")
    return {"loss": total_loss / total_samples, "accuracy": total_correct / total_samples}


def save_checkpoint(path, model, optimizer, scheduler, epoch, best_accuracy, config):
    temporary = Path(str(path) + ".tmp")
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "best_val_accuracy": best_accuracy,
        "config": config,
    }, temporary)
    os.replace(temporary, path)


def safe_run_name(name):
    if not name or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for c in name):
        raise ValueError("Run name may contain only letters, digits, '-' and '_'")
    return name


def main():
    parser = argparse.ArgumentParser(description="AI6103 CIFAR-100 training")
    parser.add_argument("--config", default="configs/main.yaml")
    parser.add_argument("--policy", default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--max-batches", type=int, default=None,
                        help="Debug only: limit training batches per epoch; validation stays full")
    parser.add_argument("--run-name", default=None)
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError("Config must be a YAML mapping")
    if args.policy is not None:
        config["augmentation"]["policy"] = args.policy
    if args.epochs is not None:
        config["training"]["epochs"] = args.epochs
    if args.seed is not None:
        config["experiment"]["seed"] = args.seed
    if config["training"]["epochs"] < 1:
        raise ValueError("epochs must be >= 1")
    if args.max_batches is not None and args.max_batches < 1:
        raise ValueError("--max-batches must be >= 1")

    seed = int(config["experiment"]["seed"])
    policy = config["augmentation"]["policy"]
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    default_name = f"{config['experiment']['name']}_{policy}_seed{seed}"
    if args.max_batches is not None:
        default_name += "_smoke"
    run_name = safe_run_name(args.run_name or default_name)
    checkpoint_dir = Path(config["output"]["checkpoint_dir"]) / run_name
    log_dir = Path(config["output"]["log_dir"])
    log_file = log_dir / f"{run_name}.csv"

    # Prevent accidental overwrites, including smoke vs formal runs.
    if checkpoint_dir.exists() or log_file.exists():
        raise FileExistsError(
            f"Run '{run_name}' already exists. Choose another --run-name; "
            "existing checkpoints/logs were not modified."
        )
    checkpoint_dir.mkdir(parents=True, exist_ok=False)
    log_dir.mkdir(parents=True, exist_ok=True)

    print(f"Device: {device}", flush=True)
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)
    print(f"Run: {run_name} | policy: {policy} | seed: {seed}", flush=True)

    train_loader, val_loader = build_dataloaders(
        data_root=config["data"]["root"],
        split_dir=config["data"]["split_dir"],
        batch_size=config["data"]["batch_size"],
        num_workers=config["data"]["num_workers"],
        seed=seed,
    )
    print(f"Train samples: {len(train_loader.dataset)}; val samples: {len(val_loader.dataset)}", flush=True)

    model = build_resnet18(num_classes=config["model"]["num_classes"]).to(device)
    tr_cfg = config["training"]
    if tr_cfg.get("optimizer", "sgd").lower() != "sgd":
        raise ValueError("This training entry currently supports optimizer: sgd")
    if tr_cfg.get("scheduler", "cosine").lower() != "cosine":
        raise ValueError("This training entry currently supports scheduler: cosine")
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=tr_cfg["learning_rate"],
        momentum=tr_cfg["momentum"],
        weight_decay=tr_cfg["weight_decay"],
    )
    epochs = int(tr_cfg["epochs"])
    checkpoint_interval = int(tr_cfg.get("checkpoint_interval", 1))
    if checkpoint_interval < 1:
        raise ValueError("checkpoint_interval must be >= 1")
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    with open(checkpoint_dir / "config.json", "w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2, ensure_ascii=False)
    with open(checkpoint_dir / "environment.json", "w", encoding="utf-8") as handle:
        json.dump({
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "python": os.sys.version.split()[0],
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "device": str(device),
            "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
            "debug_max_batches": args.max_batches,
            "platform": platform.platform(),
            "split_sha256": {name: hashlib.sha256((Path(config["data"]["split_dir"])/name).read_bytes()).hexdigest()
                             for name in ("train_indices.npy", "val_indices.npy")},
            "source_sha256": {str(path): hashlib.sha256(path.read_bytes()).hexdigest()
                              for folder in ("augmentations", "models", "data")
                              for path in sorted(Path(folder).glob("*.py"))},
            "train_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        }, handle, indent=2)
    with open(log_file, "w", newline="", encoding="utf-8") as handle:
        csv.DictWriter(handle, fieldnames=LOG_FIELDS).writeheader()

    best_accuracy = -1.0
    for epoch in range(1, epochs + 1):
        epoch_start = time.perf_counter()
        lr = optimizer.param_groups[0]["lr"]
        print(f"\n===== Epoch {epoch}/{epochs} | lr={lr:.6g} =====", flush=True)

        train_metrics = train_one_epoch(
            model, train_loader, optimizer, device,
            aug_cfg=config["augmentation"], max_batches=args.max_batches,
        )
        val_metrics = validate(model, val_loader, device)
        scheduler.step()

        elapsed = time.perf_counter() - epoch_start
        train_acc_str = (f"{train_metrics['accuracy']:.4f}"
                         if train_metrics["accuracy"] is not None else "N/A (mixed labels)")
        print(f"Train loss: {train_metrics['loss']:.4f} | train acc: {train_acc_str}", flush=True)
        print(f"Val loss: {val_metrics['loss']:.4f} | val acc: {val_metrics['accuracy']:.4f}", flush=True)
        print(f"Train time: {train_metrics['seconds']:.2f}s | epoch time: {elapsed:.2f}s", flush=True)

        with open(log_file, "a", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=LOG_FIELDS)
            writer.writerow({
                "epoch": epoch, "learning_rate": lr,
                "train_loss": train_metrics["loss"],
                "train_accuracy": train_metrics["accuracy"],
                "val_loss": val_metrics["loss"],
                "val_accuracy": val_metrics["accuracy"],
                "train_seconds": train_metrics["seconds"],
                "epoch_seconds": elapsed,
                "train_samples": train_metrics["samples"],
            })

        if val_metrics["accuracy"] > best_accuracy:
            best_accuracy = val_metrics["accuracy"]
            save_checkpoint(checkpoint_dir / "best.pth", model, optimizer,
                            scheduler, epoch, best_accuracy, config)
            print("New best checkpoint saved", flush=True)
        if epoch % checkpoint_interval == 0 or epoch == epochs:
            save_checkpoint(checkpoint_dir / "last.pth", model, optimizer,
                            scheduler, epoch, best_accuracy, config)

    print(f"\nFinished. Best validation accuracy: {best_accuracy:.4f}", flush=True)
    print(f"Logs: {log_file}", flush=True)
    print(f"Checkpoints: {checkpoint_dir}", flush=True)


if __name__ == "__main__":
    main()
