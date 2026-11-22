"""Validate the supplied split without overwriting it or importing scikit-learn.

The original generation script is retained in the archived Member A handoff.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
EXPECTED_SHA256 = {
    "train_indices.npy": "24520246ef0988e61bb0fe0f5c66da3a1f944be0569ccb5f0afc2536b92dfc15",
    "val_indices.npy": "2c80060ed0fa8eba98b2023be25aa56794982c9d6bf762dd1eb29205835d67c2",
}


def verify_split(split_dir=ROOT / "data/splits", targets=None):
    split_dir = Path(split_dir)
    hashes = {name: hashlib.sha256((split_dir / name).read_bytes()).hexdigest()
              for name in EXPECTED_SHA256}
    if hashes != EXPECTED_SHA256:
        raise ValueError("Split hashes do not match the supplied fixed split; restore the committed indices")
    train = np.load(split_dir / "train_indices.npy", allow_pickle=False)
    val = np.load(split_dir / "val_indices.npy", allow_pickle=False)
    if (train.shape != (45000,) or val.shape != (5000,)
            or not np.issubdtype(train.dtype, np.integer) or not np.issubdtype(val.dtype, np.integer)
            or np.unique(train).size != 45000 or np.unique(val).size != 5000
            or np.intersect1d(train, val).size
            or not np.array_equal(np.sort(np.r_[train, val]), np.arange(50000))):
        raise ValueError("Expected a disjoint, complete 45,000/5,000 partition")
    if targets is not None:
        labels = np.asarray(targets)
        if (labels.shape != (50000,) or not np.issubdtype(labels.dtype, np.integer)
                or np.any((labels < 0) | (labels >= 100))):
            raise ValueError("Expected 50,000 CIFAR-100 training labels")
        if not (np.all(np.bincount(labels[train], minlength=100) == 450)
                and np.all(np.bincount(labels[val], minlength=100) == 50)):
            raise ValueError("Expected 450 training and 50 validation samples per class")
    return {"status": "passed", "train_samples": len(train), "validation_samples": len(val),
            "sha256": hashes, "class_balance_checked": targets is not None}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split-dir", type=Path, default=ROOT / "data/splits")
    args = parser.parse_args()
    print(json.dumps(verify_split(args.split_dir), indent=2))
