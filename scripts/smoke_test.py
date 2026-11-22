"""Exercise every policy, the shared trainer, evaluation and checkpoint I/O."""
import argparse
import json
import math
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
from data.dataset import build_dataloaders
from models.resnet import build_resnet18
from train import apply_augmentation, normalize_images, save_checkpoint, set_seed, train_one_epoch, validate
from scripts.evaluate_member_c_val import evaluate

POLICIES = ("none", "random", "source_edge", "dual_edge", "seam_aware", "seam_dp", "seam_random")


def run(device="cpu", data_root=None, threads=2):
    torch.set_num_threads(threads)
    set_seed(42)
    device = torch.device(device)
    aug_device = torch.device("cpu") if device.type == "mps" else device
    if data_root is None:
        train_batch = (torch.rand(4, 3, 32, 32), torch.arange(4))
        val_batch = (torch.rand(4, 3, 32, 32), torch.arange(4))
        data_kind = "synthetic tensors; no dataset required"
    else:
        train_loader, val_loader = build_dataloaders(data_root=data_root,
            split_dir=ROOT / "data/splits", batch_size=4, num_workers=0, seed=42)
        train_batch, val_batch = next(iter(train_loader)), next(iter(val_loader))
        data_kind = "one batch each from the fixed CIFAR-100 training and validation splits"
    model = build_resnet18().to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=.01, momentum=.9)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=len(POLICIES))
    checked = []
    for policy in POLICIES:
        aug = dict(policy=policy, alpha=1., p=1., k=4,
                   gamma=0. if policy == "source_edge" else .5, beta=1.,
                   seam_radius=2, seam_mu=.05, seam_min_width=2,
                   seam_area_budget=.2, seam_random_mu=.225)
        images, labels = (item.to(aug_device) for item in train_batch)
        mixed, _, _, lam, meta = apply_augmentation(images, labels, aug)
        expected = 1. - meta["mask"].float().mean((1, 2, 3))
        torch.testing.assert_close(lam.float(), expected)
        if not torch.isfinite(mixed).all():
            raise ValueError(f"{policy}: non-finite mixed image")
        before = model.fc.weight.detach().clone()
        metrics = train_one_epoch(model, [train_batch], optimizer, device, aug,
                                  max_batches=1, augmentation_device=aug_device)
        if metrics["samples"] != 4 or not math.isfinite(metrics["loss"]):
            raise ValueError(f"{policy}: invalid one-batch training result")
        if torch.equal(before, model.fc.weight):
            raise ValueError(f"{policy}: optimizer did not update the classifier")
        checked.append({"policy": policy, "samples": metrics["samples"], "status": "passed"})
        scheduler.step()
    clean = validate(model, [val_batch], device)
    n, accuracy, f1, loss = evaluate(model, [val_batch], device)
    if n != 4 or not all(math.isfinite(v) for v in (accuracy, f1, loss, clean["loss"])):
        raise ValueError("Invalid smoke evaluation")
    config = {"experiment": {"seed": 42}, "augmentation": aug,
              "training": {"epochs": len(POLICIES)}}
    # Temporary checkpoints test the real save/load format and are removed on exit.
    with tempfile.TemporaryDirectory(prefix="ai6103-smoke-") as folder:
        checkpoint = Path(folder) / "best.pth"
        save_checkpoint(checkpoint, model, optimizer, scheduler, len(POLICIES), accuracy, config)
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        restored = build_resnet18().to(device)
        restored.load_state_dict(state["model_state_dict"])
        model.eval(); restored.eval()
        inputs = normalize_images(val_batch[0].to(device))
        with torch.no_grad():
            torch.testing.assert_close(model(inputs), restored(inputs))
    return {"status": "passed", "scope": "execution check only; not a classification experiment",
            "data": data_kind, "device": str(device), "augmentation_device": str(aug_device),
            "torch": str(torch.__version__), "policies": checked,
            "shared_training_loop": "passed", "clean_evaluation": "passed",
            "checkpoint_roundtrip": "passed"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "cuda", "mps"), default="cpu")
    parser.add_argument("--data-root", type=Path, help="Use an existing CIFAR-100 archive; never download")
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("--threads must be positive")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA is unavailable")
    if args.device == "mps" and not torch.backends.mps.is_available():
        parser.error("Apple MPS is unavailable")
    result = run(args.device, args.data_root, args.threads)
    text = json.dumps(result, indent=2, allow_nan=False) + "\n"
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text, encoding="utf-8")
    print(text, end="")


if __name__ == "__main__":
    main()
