"""Unified A/B/C augmentation API. seam_random fails until implemented."""
import torch
from .cutmix import random_cutmix
from .source_edge import source_edge_cutmix
from .dual_edge import dual_edge_cutmix
from .seam_aware import seam_aware_cutmix
from .seam_dp import seam_dp_cutmix


def cutmix_batch(images, labels, policy="none", alpha=1.0, p=0.5,
                 k=4, gamma=0.5, beta=0.0, seam_radius=0, seam_mu=0.0,
                 seam_min_width=2, seam_area_budget=0.1):
    # seam_min_width / seam_area_budget only affect seam_dp (and later seam_random);
    # their defaults are provisional until frozen with D.
    if (images.ndim != 4 or images.shape[1] != 3 or min(images.shape) < 1
            or not images.is_floating_point() or labels.shape != (images.shape[0],)
            or labels.device != images.device):
        raise ValueError("Expected nonempty floating RGB NCHW and labels [N] on same device")
    n, _, h, w = images.shape
    if policy == "none":
        return (images, labels, labels, torch.ones(n, dtype=images.dtype, device=images.device),
                {"policy": "none", "mask": torch.zeros((n,1,h,w), dtype=torch.bool, device=images.device), "boxes": None})
    if policy == "random":
        return random_cutmix(images, labels, alpha, p)
    if policy in ("source_edge", "dual_edge"):
        fn = source_edge_cutmix if policy == "source_edge" else dual_edge_cutmix
        return fn(images, labels, alpha, p, k, gamma)
    if policy == "seam_aware":
        return seam_aware_cutmix(images, labels, alpha, p, k, gamma, beta)
    if policy == "seam_dp":
        return seam_dp_cutmix(images, labels, alpha, p, k, gamma, beta,
                              seam_radius, seam_mu, seam_min_width, seam_area_budget)
    raise NotImplementedError(f"Policy {policy!r} belongs to C and is not integrated")
