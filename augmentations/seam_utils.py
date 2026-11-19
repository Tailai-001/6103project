"""Member C: seam (boundary) cost shared by seam_aware, seam_dp and seam_random.

Definition (README §1.3, docs/experiment_protocol.md):

    D_seam(M) = 1 / (3 |B(M)|) * sum_{(p,q) in B(M)} sum_c |x_B,c(p) - x_A,c(q)|

B(M) is the set of in-image 4-neighbour pixel pairs (p, q) with M(p)=1 (source
pixel) and M(q)=0 (target pixel). Pixels outside the image are never counted,
so a patch touching the image border has no seam on that side.

Implementation note: on such a pair the mixed image equals x_B at p and x_A at
q, so the cost is simply the mean absolute RGB difference of the *mixed* image
across neighbouring pixels whose mask values differ. This avoids building pair
lists and works for any binary mask (rectangles for E3, curved masks for E4).

Inputs must be RGB in [0, 1], before model normalization.
"""
import torch


def _check_inputs(source, target, mask):
    if source.ndim != 4 or source.shape[1] != 3 or not source.is_floating_point():
        raise ValueError("source must be floating RGB [N,3,H,W]")
    if target.shape != source.shape or target.dtype != source.dtype:
        raise ValueError("target must match source shape and dtype")
    n, _, h, w = source.shape
    if mask.dtype != torch.bool:
        raise ValueError("mask must be a bool tensor (True = source pixel)")
    if mask.shape == (n, h, w):
        mask = mask[:, None]
    if mask.shape != (n, 1, h, w):
        raise ValueError("mask must be [N,H,W] or [N,1,H,W]")
    if not (source.device == target.device == mask.device):
        raise ValueError("source, target and mask must be on the same device")
    return mask


def seam_cost(source, target, mask):
    """Per-sample seam cost of pasting `source` into `target` under `mask`.

    Args:
        source: [N,3,H,W] float RGB in [0,1] (x_B, the pasted image).
        target: [N,3,H,W] float RGB in [0,1] (x_A, the receiving image).
        mask:   [N,H,W] or [N,1,H,W] bool, True where source pixels are used.

    Returns:
        cost:  [N] float, mean per-channel |x_B(p) - x_A(q)| over seam pairs.
               Set to 0 where count == 0 (empty/full mask, or a mask that
               covers whole rows/columns with no internal boundary).
        count: [N] int64, |B(M)|, number of in-image seam pairs. Callers must
               treat count == 0 as "not applicable" in boundary metrics.

    Accumulates in float32 (float64 inputs stay float64); output cost has the
    accumulation dtype so tiny differences are not lost in half precision.
    """
    mask = _check_inputs(source, target, mask)
    acc = torch.float64 if source.dtype == torch.float64 else torch.float32
    mixed = torch.where(mask, source, target).to(acc)

    # Horizontal neighbours (left/right) and vertical neighbours (up/down).
    seam_h = mask[..., :, 1:] != mask[..., :, :-1]          # [N,1,H,W-1]
    seam_v = mask[..., 1:, :] != mask[..., :-1, :]          # [N,1,H-1,W]
    diff_h = (mixed[..., :, 1:] - mixed[..., :, :-1]).abs().mean(1, keepdim=True)
    diff_v = (mixed[..., 1:, :] - mixed[..., :-1, :]).abs().mean(1, keepdim=True)

    total = (diff_h * seam_h).flatten(1).sum(1) + (diff_v * seam_v).flatten(1).sum(1)
    count = seam_h.flatten(1).sum(1) + seam_v.flatten(1).sum(1)
    cost = torch.where(count > 0, total / count.clamp_min(1).to(acc),
                       torch.zeros_like(total))
    return cost, count


def image_seam_scale(images):
    """Image-level scale [N]: mean per-channel |RGB difference| over ALL in-image
    4-neighbour pairs of each image, i.e. its own natural local variation.

    Used to normalise D_seam so that D_hat = 1 means "the seam is as abrupt as
    the images' own texture". One scale per image, never per candidate.
    """
    if images.ndim != 4 or images.shape[1] != 3 or not images.is_floating_point():
        raise ValueError("images must be floating RGB [N,3,H,W]")
    acc = torch.float64 if images.dtype == torch.float64 else torch.float32
    x = images.to(acc)
    dh = (x[..., :, 1:] - x[..., :, :-1]).abs().mean(1).flatten(1)
    dv = (x[..., 1:, :] - x[..., :-1, :]).abs().mean(1).flatten(1)
    pairs = dh.shape[1] + dv.shape[1]
    if pairs == 0:  # 1x1 images have no neighbours
        return x.new_zeros(x.shape[0])
    return (dh.sum(1) + dv.sum(1)) / pairs


def normalized_seam_cost(source, target, mask, eps=1e-3):
    """D_hat = D_seam / max(scale, eps), scale = mean of the two images' scales.

    eps is in [0,1] RGB units; it keeps constant or near-constant image pairs
    finite. Pairs with count == 0 (no internal seam) get D_hat = 0 and must be
    reported as not applicable. Returns (d_hat, d_raw, count).
    """
    if not eps > 0:
        raise ValueError("eps must be positive")
    d_raw, count = seam_cost(source, target, mask)
    scale = 0.5 * (image_seam_scale(source) + image_seam_scale(target))
    d_hat = torch.where(count > 0, d_raw / scale.clamp_min(eps), torch.zeros_like(d_raw))
    return d_hat, d_raw, count


def candidate_seam_costs(images, permutation, candidates, eps=1e-3):
    """Seam costs [N,K] for every candidate box, source = images[permutation].

    candidates: int64 [N,K,4] half-open boxes shared with B's edge scores.
    Deterministic and RNG-free, so adding it cannot change any random draw.
    Returns (d_hat, d_raw, count), each [N,K].
    """
    n, k = candidates.shape[:2]
    _, c, h, w = images.shape
    masks = box_masks(candidates, h, w).reshape(n * k, h, w)
    src = images[permutation][:, None].expand(n, k, c, h, w).reshape(n * k, c, h, w)
    tgt = images[:, None].expand(n, k, c, h, w).reshape(n * k, c, h, w)
    d_hat, d_raw, count = normalized_seam_cost(src, tgt, masks, eps)
    return d_hat.view(n, k), d_raw.view(n, k), count.view(n, k)


def box_masks(boxes, height, width):
    """Bool masks [N,K,H,W] from int64 half-open boxes [N,K,4] = (top,left,bottom,right).

    Same coordinate convention as A's random_cutmix and B's candidates:
    mask[..., top:bottom, left:right] = True.
    """
    if boxes.shape[-1] != 4 or boxes.dtype != torch.int64:
        raise ValueError("boxes must be int64 [...,4] (top,left,bottom,right)")
    top, left, bottom, right = (b[..., None, None] for b in boxes.unbind(-1))
    rows = torch.arange(height, device=boxes.device)[:, None]
    cols = torch.arange(width, device=boxes.device)[None, :]
    return (rows >= top) & (rows < bottom) & (cols >= left) & (cols < right)
