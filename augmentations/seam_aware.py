"""E3 seam_aware policy (Member C): dual-edge score minus a seam penalty.

    S_seam(R) = E_B(R) - gamma * E_A(R) - beta * D_hat_seam(R)

Shares everything else with B's E2 (augmentations/selection.py): the same
A-sampled base box, the same K translated candidates (equal actual area),
the same Sobel normalisation, same-coordinate pasting and first-index ties.
The seam term is RNG-free, so with beta=0 the output equals dual_edge exactly.
"""
import math

import torch

from .candidates import boxes_to_mask, sample_candidates
from .edge_utils import candidate_edge_scores
from .seam_utils import candidate_seam_costs


def seam_aware_cutmix(images, labels, alpha=1.0, p=0.5, k=4, gamma=0.5,
                      beta=0.0, seam_eps=1e-3):
    if not math.isfinite(gamma) or gamma < 0:
        raise ValueError("gamma must be finite and nonnegative")
    if not math.isfinite(beta) or beta < 0:
        raise ValueError("beta must be finite and nonnegative")
    candidates, base = sample_candidates(images, labels, alpha, p, k)
    perm = base["permutation"]
    source_edge, target_edge, edge_score = candidate_edge_scores(
        images, perm, candidates, gamma)
    seam_hat, seam_raw, seam_count = candidate_seam_costs(
        images, perm, candidates, seam_eps)
    score = edge_score - beta * seam_hat.to(edge_score.dtype)
    selected = score.argmax(1)  # first index on ties, same as E2
    row = torch.arange(images.shape[0], device=images.device)
    boxes = candidates[row, selected]
    mask = boxes_to_mask(boxes, *images.shape[-2:])
    mixed = torch.where(mask, images[perm], images)
    lam = (1 - mask.flatten(1).float().mean(1)).to(images.dtype)
    return mixed, labels, labels[perm], lam, {
        "policy": "seam_aware", "mask": mask, "boxes": boxes,
        "base_boxes": base["boxes"], "permutation": perm,
        "candidates": candidates, "selected_indices": selected,
        "source_edge_scores": source_edge, "target_edge_scores": target_edge,
        "edge_scores": edge_score, "candidate_scores": score,
        "seam_costs": seam_hat, "seam_costs_raw": seam_raw,
        "seam_counts": seam_count,
        # Selected-box diagnostics for D; count == 0 means "not applicable".
        "seam_cost": seam_hat[row, selected],
        "seam_cost_raw": seam_raw[row, selected],
        "seam_count": seam_count[row, selected],
    }
