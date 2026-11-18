
import random
import numpy as np
import torch

from pathlib import Path
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def build_dataloaders(
    data_root="data/cifar100",
    split_dir="data/splits",
    batch_size=128,
    num_workers=4,
    seed=42,
):
    data_root = Path(data_root)
    split_dir = Path(split_dir)

    # Training augmentation
    # Keep images in [0, 1], without normalization.
    # CutMix and seam scoring will run before normalization.
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])

    val_transform = transforms.ToTensor()

    # Separate dataset objects ensure different transforms
    # for training and validation.
    train_full = datasets.CIFAR100(
        root=str(data_root),
        train=True,
        download=False,
        transform=train_transform,
    )

    val_full = datasets.CIFAR100(
        root=str(data_root),
        train=True,
        download=False,
        transform=val_transform,
    )

    train_indices = np.load(
        split_dir / "train_indices.npy"
    )
    val_indices = np.load(
        split_dir / "val_indices.npy"
    )

    assert len(train_indices) == 45000
    assert len(val_indices) == 5000
    assert len(
        np.intersect1d(train_indices, val_indices)
    ) == 0

    train_dataset = Subset(train_full, train_indices)
    val_dataset = Subset(val_full, val_indices)

    generator = torch.Generator()
    generator.manual_seed(seed)

    use_pin_memory = torch.cuda.is_available()

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=use_pin_memory,
        worker_init_fn=seed_worker,
        generator=generator,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=use_pin_memory,
        worker_init_fn=seed_worker,
    )

    return train_loader, val_loader


if __name__ == "__main__":
    train_loader, val_loader = build_dataloaders(
        num_workers=0
    )

    images, labels = next(iter(train_loader))

    print("Train batches:", len(train_loader))
    print("Validation batches:", len(val_loader))
    print("Images:", images.shape)
    print("Labels:", labels.shape)
    print("Pixel range:", images.min().item(),
          images.max().item())

    assert images.shape[1:] == (3, 32, 32)
    assert images.min() >= 0
    assert images.max() <= 1

    print("DataLoader test passed!")
