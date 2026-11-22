"""Check CIFAR-100 and the supplied split; downloading requires --download."""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from torchvision.datasets import CIFAR100
from data.split import verify_split


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=ROOT / "data/cifar100")
    parser.add_argument("--download", action="store_true", help="Explicitly download the official archive")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    try:
        data = CIFAR100(str(args.data_root), train=True, download=args.download)
    except RuntimeError as error:
        parser.error(f"{error} Run with --download, or set --data-root to an existing archive.")
    result = verify_split(ROOT / "data/splits", targets=data.targets)
    result.update(data_root=str(args.data_root), official_training_images=len(data),
                  official_test_evaluated=False)
    text = json.dumps(result, indent=2) + "\n"
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text, encoding="utf-8")
    print(text, end="")


if __name__ == "__main__":
    main()
