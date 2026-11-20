"""Re-evaluate C's best checkpoints on the fixed clean validation split; adds Macro-F1.

Adapted from B's scripts/evaluate_member_b_val.py (same metric definitions, so C and B
numbers sit in one table): reloads checkpoints/<run>/best.pth, recomputes validation
Top-1, loss and Macro-F1, and checks that accuracy and epoch match the training log.
Only runs whose best.pth and log exist are evaluated; results are merged by run_id into
results/member_c/checkpoint_validation.json so evaluations from several machines combine.
Never touches the test set.
"""
import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
import torch  # noqa: E402
from data.dataset import build_dataloaders  # noqa: E402
from models.resnet import build_resnet18  # noqa: E402
from train import normalize_images, set_seed  # noqa: E402
from member_c_suite import ARMS  # noqa: E402


@torch.no_grad()
def evaluate(model, loader, device):
    counts = torch.zeros(100, 100, dtype=torch.int64, device=device)
    loss_sum = 0.0
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        logits = model(normalize_images(x))
        loss_sum += torch.nn.functional.cross_entropy(logits, y, reduction="sum").item()
        counts += torch.bincount(100 * y + logits.argmax(1), minlength=10000).reshape(100, 100)
    counts = counts.cpu()
    tp = counts.diag().double()
    total = int(counts.sum())
    macro_f1 = float((2 * tp / (counts.sum(0) + counts.sum(1)).clamp_min(1)).mean())
    return total, float(tp.sum()) / total, macro_f1, loss_sum / total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--suffix", default="")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--out", default=str(ROOT / "results/member_c/checkpoint_validation.json"))
    args = parser.parse_args()
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, loader = build_dataloaders(data_root=str(ROOT / "data/cifar100"),
                                  split_dir=str(ROOT / "data/splits"), num_workers=args.num_workers)
    out = Path(args.out)
    merged = {r["run_id"]: r for r in json.loads(out.read_text())} if out.exists() else {}
    script_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    failures = []
    for experiment, stem, run in ARMS.values():
        name = run + ("_" + args.suffix if args.suffix else "")
        ckpt, log = ROOT / "checkpoints" / name / "best.pth", ROOT / "logs" / f"{name}.csv"
        if not ckpt.exists() or not log.exists():
            continue
        state = torch.load(ckpt, map_location="cpu", weights_only=True)
        model = build_resnet18().to(device)
        model.load_state_dict(state["model_state_dict"])
        model.eval()
        total, accuracy, macro_f1, loss = evaluate(model, loader, device)
        rows = list(csv.DictReader(log.open()))
        best = max(rows, key=lambda row: float(row["val_accuracy"]))  # first max = earliest
        verified = (total == 5000 and state["epoch"] == int(best["epoch"])
                    and abs(accuracy - float(best["val_accuracy"])) < 1e-10)
        if not verified:
            failures.append(name)
        merged[name] = dict(run_id=name, experiment=experiment, config=f"configs/{stem}.yaml",
                            split="fixed_validation_only", samples=total, device=str(device),
                            num_workers=args.num_workers, evaluation_script_sha256=script_hash,
                            checkpoint_epoch=state["epoch"], val_top1_percent=100 * accuracy,
                            val_macro_f1=macro_f1, val_loss=loss,
                            logged_accuracy_verified=verified,
                            logged_val_top1_percent=100 * float(best["val_accuracy"]))
        print(json.dumps(merged[name]), flush=True)
        del model, state
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(list(merged.values()), indent=2))
    if failures:
        # Keep the file (it records the mismatch) but make the failure visible.
        sys.exit(f"checkpoint vs logged validation mismatch: {failures}")


if __name__ == "__main__":
    main()
