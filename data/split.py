
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from torchvision.datasets import CIFAR100

SEED = 42
DATA_ROOT = Path("data/cifar100")
SPLIT_DIR = Path("data/splits")

SPLIT_DIR.mkdir(parents=True, exist_ok=True)


dataset = CIFAR100(
    root=str(DATA_ROOT),
    train=True,
    download=False
)

labels = np.array(dataset.targets)
indices = np.arange(len(dataset))


train_idx, val_idx = train_test_split(
    indices,
    test_size=0.1,
    random_state=SEED,
    stratify=labels,
    shuffle=True
)


np.save(SPLIT_DIR / "train_indices.npy", train_idx)
np.save(SPLIT_DIR / "val_indices.npy", val_idx)


assert len(train_idx) == 45000
assert len(val_idx) == 5000
assert len(set(train_idx) & set(val_idx)) == 0
assert len(set(train_idx) | set(val_idx)) == 50000

train_counts = np.bincount(labels[train_idx], minlength=100)
val_counts = np.bincount(labels[val_idx], minlength=100)

assert np.all(train_counts == 450)
assert np.all(val_counts == 50)

print("Dataset split completed.")
print("Training samples:", len(train_idx))
print("Validation samples:", len(val_idx))
print("Number of classes:", len(np.unique(labels)))
print("Random seed:", SEED)
print("Split saved to:", SPLIT_DIR)
