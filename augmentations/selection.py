"""Shared E2/E2-S implementation, independent of classification training."""
import math
import torch
from .candidates import sample_candidates, boxes_to_mask
from .edge_utils import candidate_edge_scores


def edge_cutmix(images, labels, policy, alpha=1.0, p=0.5, k=4, gamma=0.5):
    if not math.isfinite(gamma) or gamma < 0:
        raise ValueError("gamma must be finite and nonnegative")
    candidates, base = sample_candidates(images, labels, alpha, p, k)
    perm = base["permutation"]
    source, target, score = candidate_edge_scores(
        images, perm, candidates, 0.0 if policy == "source_edge" else gamma)
    selected = score.argmax(1)  # torch argmax returns first index on ties
    row = torch.arange(images.shape[0], device=images.device)
    boxes = candidates[row, selected]
    mask = boxes_to_mask(boxes, *images.shape[-2:])
    mixed = torch.where(mask, images[perm], images)
    lam = (1 - mask.flatten(1).float().mean(1)).to(images.dtype)
    return mixed, labels, labels[perm], lam, {
        "policy": policy, "mask": mask, "boxes": boxes,
        "base_boxes": base["boxes"], "permutation": perm,
        "candidates": candidates, "selected_indices": selected,
        "source_edge_scores": source, "target_edge_scores": target,
        "candidate_scores": score,
    }
