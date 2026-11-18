"""Shared augmentation entry point for AI6103 E0–E4."""
import torch
from .cutmix import random_cutmix


def cutmix_batch(images, labels, policy="none", alpha=1.0, p=0.5,
                 k=4, gamma=0.5, beta=0.0, seam_radius=0, seam_mu=0.0):
    n, _, h, w = images.shape
    if policy == "none":
        return (images, labels, labels,
                torch.ones(n, dtype=images.dtype, device=images.device),
                {"policy": "none", "mask": torch.zeros((n, 1, h, w),
                 dtype=torch.bool, device=images.device), "boxes": None})
    if policy == "random":
        return random_cutmix(images, labels, alpha=alpha, p=p)
    raise NotImplementedError(f"Policy {policy!r} is not integrated yet")
