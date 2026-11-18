"""Original CutMix for AI6103, adapted to per-sample masks and shared API.

Based on CutMix (Yun et al., ICCV 2019) and the official rand_bbox
sampling convention. Source and target use identical spatial coordinates.
"""
import math
import numpy as np
import torch


def random_cutmix(images: torch.Tensor, labels: torch.Tensor,
                  alpha: float = 1.0, p: float = 0.5):
    """Apply CutMix independently to batch samples, with probability p each.

    Images must be float NCHW RGB in [0,1], before normalization.
    Returns mixed_images, labels_a, labels_b, lam_actual[N], metadata.
    """
    if images.ndim != 4 or labels.ndim != 1 or images.shape[0] != labels.shape[0]:
        raise ValueError("Expected images [N,C,H,W] and labels [N]")
    if not images.is_floating_point():
        raise ValueError("images must be floating-point tensors")
    if alpha <= 0 or not 0 <= p <= 1:
        raise ValueError("alpha must be > 0 and p must be in [0,1]")

    n, _, h, w = images.shape
    perm = torch.randperm(n, device=images.device)
    labels_a, labels_b = labels, labels[perm]
    source = images[perm]  # source is read from ORIGINAL, unmodified batch
    mixed = images.clone()
    masks = torch.zeros((n, 1, h, w), dtype=torch.bool, device=images.device)
    boxes = torch.zeros((n, 4), dtype=torch.int64, device=images.device)

    # Crop sizes follow the original rand_bbox convention:
    # cut_rat = sqrt(1 - lambda), integer width/height, random center, clip.
    for i in range(n):
        if torch.rand((), device=images.device).item() >= p:
            continue
        lam_sample = float(np.random.beta(alpha, alpha))
        cut_rat = math.sqrt(1.0 - lam_sample)
        cut_w = int(w * cut_rat)
        cut_h = int(h * cut_rat)
        cx = int(torch.randint(w, (1,), device=images.device).item())
        cy = int(torch.randint(h, (1,), device=images.device).item())
        left = max(cx - cut_w // 2, 0)
        right = min(cx + cut_w // 2, w)
        top = max(cy - cut_h // 2, 0)
        bottom = min(cy + cut_h // 2, h)
        boxes[i] = torch.tensor((top, left, bottom, right), device=images.device)
        if bottom > top and right > left:
            masks[i, :, top:bottom, left:right] = True
            mixed[i, :, top:bottom, left:right] = source[i, :, top:bottom, left:right]

    lam_actual = 1.0 - masks.flatten(1).float().mean(dim=1)
    lam_actual = lam_actual.to(dtype=images.dtype)
    metadata = {"policy": "random", "mask": masks, "boxes": boxes, "permutation": perm}
    return mixed, labels_a, labels_b, lam_actual, metadata
