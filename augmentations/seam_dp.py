"""E4 seam_dp (Member C) — work in progress.

Step 1 (this file so far): the single-boundary dynamic programme.

Starting from the E3 rectangle, the top/bottom edges stay fixed and one
vertical edge (left or right) may shift by an offset d_y in [-r, r] on each
row y, with |d_y - d_{y-1}| <= 1 (connected boundary). Given a per-row,
per-offset local cost c(y, d), we minimise

    sum_y c(y, d_y) + mu * sum_{y>0} |d_y - d_{y-1}|

with the recurrence (README §1.4)

    DP(y, d) = c(y, d) + min_{s in {-1,0,1}} [ DP(y-1, d+s) + mu * |s| ].

This is exact for that local objective only; it is not a global optimum of the
full D_seam of the final mask (top/bottom edges and horizontal connectors are
not in it), which must be recomputed afterwards.

Step 2: the local cost c (edge_cost_table). For the right edge of box
(top, left, bottom, right) placed at column X = right + d on row y, the patch
covers columns < X, so the only seam pair on that row and side is
(source(y, X-1), target(y, X)); c is that pair's channel-mean |RGB diff|.
The left edge at L = left + d pairs (source(y, L), target(y, L-1)). These are
exactly the horizontal pairs of the full D_seam on the box rows.

Step 3: coupling (solve_pair_dp, paths_to_mask). The two edges are solved
jointly over states (a, b) = (left offset, right offset), so no edge is
privileged by solving order and the result is the exact optimum of

    sum_y [c_L(y, a_y) + c_R(y, b_y)] + mu * sum_y (|a_y - a_{y-1}| + |b_y - b_{y-1}|)

subject to (right + b_y) - (left + a_y) >= min_width on every row. With
min_width >= 1 every row is one contiguous run, so the edges never cross.

Step 4: area budget and fallback (finalize_boundaries, shared with E4-R),
and optimize_boundaries, which recomputes the full D_seam of the final mask
(including top/bottom edges and bend connectors not seen by the DP).
The area rule is checked after solving: |A_final - A_base| <= area_budget *
A_base, otherwise the sample falls back to its E3 rectangle with a recorded
reason. The budget value is provisional until frozen with D.

Step 5: seam_dp_cutmix, the E4 policy. It first runs E3 (seam_aware) to
choose the rectangle with the same RNG draws, candidates and selection, then
reshapes that rectangle with optimize_boundaries. Labels use the final mask.
With seam_radius = 0 the output equals seam_aware exactly.
"""
import math

import torch

from .seam_aware import seam_aware_cutmix
from .seam_utils import image_seam_scale, normalized_seam_cost


def _offset_preference(r, device):
    """Offset column indices ordered 0, -1, +1, -2, +2, ... (as columns r, r-1, r+1, ...).

    Used to break exact ties deterministically in favour of the original
    rectangle edge (offset 0), then smaller shifts, then the left/up side.
    """
    order = [r]
    for k in range(1, r + 1):
        order += [r - k, r + k]
    return torch.tensor(order, device=device)


def solve_boundary_dp(cost, mu=0.0, allowed=None, row_valid=None):
    """Minimum-cost connected boundary path, batched over N.

    Args:
        cost:    float [N, R, 2r+1]; cost[n, y, j] is c(y, d) for offset
                 d = j - r on row y of image n. Must be finite where allowed.
        mu:      bending penalty per one-pixel step between adjacent rows (>= 0).
        allowed: optional bool [N, R, 2r+1]; False cells may not be used
                 (image borders, minimum width, the other edge, ...).
        row_valid: optional bool [N, R], True on a prefix of rows. Rows after
                 the prefix are padding (boxes of different heights in one
                 batch): they are skipped exactly and copy the last real
                 row's offset. An image with no valid row returns offset 0,
                 total 0 and feasible True (nothing to optimise).

    Returns:
        offsets:  int64 [N, R] in [-r, r]; zeros where infeasible.
        total:    float [N] objective value; +inf where infeasible.
        feasible: bool [N]; False if no connected path through allowed cells.

    Ties: at every step the predecessor is chosen in the order straight,
    from-left (d-1), from-right (d+1); the final row prefers offset 0, then
    smaller |d|, then negative d. So with r = 0 the path is all zeros.
    """
    if cost.ndim != 3 or not cost.is_floating_point():
        raise ValueError("cost must be floating [N, R, 2r+1]")
    n, rows, width = cost.shape
    if rows < 1 or width < 1 or width % 2 == 0:
        raise ValueError("need R >= 1 rows and an odd number 2r+1 of offsets")
    if not math.isfinite(mu) or mu < 0:
        raise ValueError("mu must be finite and nonnegative")
    if allowed is None:
        allowed = torch.ones_like(cost, dtype=torch.bool)
    elif allowed.shape != cost.shape or allowed.dtype != torch.bool:
        raise ValueError("allowed must be bool with the same shape as cost")
    if row_valid is None:
        row_valid = torch.ones((n, rows), dtype=torch.bool, device=cost.device)
    elif row_valid.shape != (n, rows) or row_valid.dtype != torch.bool:
        raise ValueError("row_valid must be bool [N, R]")
    elif (row_valid[:, 1:] & ~row_valid[:, :-1]).any():
        raise ValueError("row_valid must be a prefix of True rows per image")
    if not torch.isfinite(cost[allowed & row_valid[..., None]]).all():
        raise ValueError("cost must be finite on allowed cells")

    r = width // 2
    acc = torch.float64 if cost.dtype == torch.float64 else torch.float32
    inf = torch.tensor(float("inf"), dtype=acc, device=cost.device)
    c = torch.where(allowed, cost.to(acc), inf)

    dp = torch.where(row_valid[:, :1], c[:, 0], torch.zeros_like(c[:, 0]))  # [N, W]
    # back[n, y, j] = predecessor column on row y-1 for cell (y, j).
    back = torch.zeros((n, rows, width), dtype=torch.int64, device=cost.device)
    cols = torch.arange(width, device=cost.device)
    for y in range(1, rows):
        # Candidates in tie-break order: straight, from d-1, from d+1.
        straight = dp
        from_left = torch.cat((inf.expand(n, 1), dp[:, :-1]), 1) + mu
        from_right = torch.cat((dp[:, 1:], inf.expand(n, 1)), 1) + mu
        stacked = torch.stack((straight, from_left, from_right), -1)   # [N, W, 3]
        best, which = stacked.min(-1)       # min returns first index on ties
        shift = torch.tensor([0, -1, 1], device=cost.device)[which]
        valid = row_valid[:, y:y + 1]
        # Padding rows pass DP values through unchanged (straight back-pointer).
        back[:, y] = torch.where(valid, (cols + shift).clamp(0, width - 1), cols)
        dp = torch.where(valid, c[:, y] + best, dp)

    order = _offset_preference(r, cost.device)
    last_vals = dp[:, order]
    total, pos = last_vals.min(1)
    col = order[pos]
    feasible = torch.isfinite(total)

    path = torch.empty((n, rows), dtype=torch.int64, device=cost.device)
    path[:, -1] = col
    row_idx = torch.arange(n, device=cost.device)
    for y in range(rows - 1, 0, -1):
        col = back[row_idx, y, col]
        path[:, y - 1] = col
    offsets = torch.where(feasible[:, None], path - r, torch.zeros_like(path))
    return offsets, total, feasible


def edge_cost_table(source, target, boxes, side, radius, eps=1e-3, normalize=True):
    """Local costs c(y, d) for shifting one vertical edge of each box.

    Args:
        source, target: float RGB [N,3,H,W] in [0,1] (x_B pasted into x_A).
        boxes:  int64 [N,4] half-open (top, left, bottom, right), e.g. E3 boxes.
        side:   "left" or "right".
        radius: r >= 0; offsets d = -r..r (table column j <-> d = j - r).
        eps:    floor of the image-level scale (RGB units), as in E3.
        normalize: divide by the same image-level scale as E3's D_hat, so that
                 mu is measured in D_hat units. False returns raw RGB costs.

    Returns:
        cost:      float [N, H, 2r+1]; table row i <-> image row top + i.
        allowed:   bool  [N, H, 2r+1]; False where the edge would leave the
                   image (right edge X in [1, W], left edge L in [0, W-1]).
        row_valid: bool  [N, H]; True for i < bottom - top (padding after).

    An edge exactly on the image border has no in-image seam pair on that
    row, so its cost is 0, matching D_seam. This makes the border attractive;
    the area budget (later step) bounds how far an edge can move there.
    Width/crossing constraints between the two edges are added later.
    """
    if side not in ("left", "right"):
        raise ValueError("side must be 'left' or 'right'")
    if isinstance(radius, bool) or not isinstance(radius, int) or radius < 0:
        raise ValueError("radius must be a nonnegative integer")
    if source.ndim != 4 or source.shape[1] != 3 or source.shape != target.shape:
        raise ValueError("source/target must be matching RGB [N,3,H,W]")
    if boxes.shape != (source.shape[0], 4) or boxes.dtype != torch.int64:
        raise ValueError("boxes must be int64 [N,4]")
    n, _, h, w = source.shape
    dev = source.device
    acc = torch.float64 if source.dtype == torch.float64 else torch.float32
    src, tgt = source.to(acc), target.to(acc)

    # pair[n, y, x] for x = 0..W: seam cost of an edge placed at column x.
    # Right edge at x pairs src[x-1] | tgt[x]; left edge at x pairs src[x] | tgt[x-1].
    # x = 0 and x = W are image borders (cost 0; validity handled below).
    if side == "right":
        inner = (src[..., :, :-1] - tgt[..., :, 1:]).abs().mean(1)     # x = 1..W-1
    else:
        inner = (src[..., :, 1:] - tgt[..., :, :-1]).abs().mean(1)     # x = 1..W-1
    zeros = inner.new_zeros(n, h, 1)
    pair = torch.cat((zeros, inner, zeros), -1)                        # [N,H,W+1]
    if normalize:
        scale = 0.5 * (image_seam_scale(source) + image_seam_scale(target))
        pair = pair / scale.clamp_min(eps)[:, None, None]

    top, left, bottom, right = boxes.unbind(-1)
    base = right if side == "right" else left
    offsets = torch.arange(-radius, radius + 1, device=dev)
    pos = base[:, None] + offsets[None, :]                             # [N, 2r+1]
    lo, hi = (1, w) if side == "right" else (0, w - 1)
    in_image = (pos >= lo) & (pos <= hi)

    i = torch.arange(h, device=dev)
    row_valid = i[None, :] < (bottom - top)[:, None]                   # [N, H]
    img_row = (top[:, None] + i[None, :]).clamp(0, h - 1)              # [N, H]
    rows = pair.gather(1, img_row[..., None].expand(n, h, w + 1))      # [N,H,W+1]
    cols = pos.clamp(0, w)[:, None, :].expand(n, h, 2 * radius + 1)
    cost = rows.gather(2, cols)
    allowed = in_image[:, None, :].expand(n, h, 2 * radius + 1) & row_valid[..., None]
    cost = torch.where(allowed, cost, torch.zeros_like(cost))
    # Padding rows stay "allowed" so the solver's pass-through is unconstrained.
    allowed = allowed | ~row_valid[..., None]
    return cost, allowed, row_valid


# Predecessor shifts (sa, sb) in tie-break order: straight, one edge moves,
# both move. Cell (a, b) on row y comes from (a + sa, b + sb) on row y - 1.
_PAIR_SHIFTS = ((0, 0), (-1, 0), (1, 0), (0, -1), (0, 1),
                (-1, -1), (-1, 1), (1, -1), (1, 1))


def _pair_final_order(r, device):
    """Flat (a, b) indices ordered by preference for the final row:
    fewer total pixels of shift first, then smaller max shift, then a, b."""
    offs = range(-r, r + 1)
    pairs = sorted(((da, db) for da in offs for db in offs),
                   key=lambda t: (abs(t[0]) + abs(t[1]), max(abs(t[0]), abs(t[1])), t[0], t[1]))
    width = 2 * r + 1
    return torch.tensor([(da + r) * width + (db + r) for da, db in pairs], device=device)


def solve_pair_dp(cost_left, allowed_left, cost_right, allowed_right, boxes,
                  mu=0.0, min_width=1, row_valid=None):
    """Jointly optimise the left and right edges of each box.

    Args:
        cost_left, cost_right:       float [N, R, 2r+1] (from edge_cost_table).
        allowed_left, allowed_right: bool  [N, R, 2r+1].
        boxes:     int64 [N,4] (top, left, bottom, right); only left/right used
                   for the width constraint.
        mu:        bending penalty per pixel of shift change, per edge.
        min_width: every valid row must satisfy right' - left' >= min_width (>= 1).
        row_valid: bool [N, R] prefix mask (padding rows are passed through).

    Returns:
        left_off, right_off: int64 [N, R] offsets (zeros where infeasible).
        total:   float [N], objective value (+inf where infeasible).
        feasible: bool [N].
    """
    if cost_left.shape != cost_right.shape or cost_left.ndim != 3:
        raise ValueError("cost tables must share shape [N, R, 2r+1]")
    for a in (allowed_left, allowed_right):
        if a.shape != cost_left.shape or a.dtype != torch.bool:
            raise ValueError("allowed masks must be bool and match the cost shape")
    n, rows, width = cost_left.shape
    if width % 2 == 0 or rows < 1:
        raise ValueError("need R >= 1 rows and an odd number 2r+1 of offsets")
    if boxes.shape != (n, 4) or boxes.dtype != torch.int64:
        raise ValueError("boxes must be int64 [N,4]")
    if not math.isfinite(mu) or mu < 0:
        raise ValueError("mu must be finite and nonnegative")
    if isinstance(min_width, bool) or not isinstance(min_width, int) or min_width < 1:
        raise ValueError("min_width must be an integer >= 1")
    dev = cost_left.device
    if row_valid is None:
        row_valid = torch.ones((n, rows), dtype=torch.bool, device=dev)
    elif row_valid.shape != (n, rows) or row_valid.dtype != torch.bool:
        raise ValueError("row_valid must be bool [N, R]")
    elif (row_valid[:, 1:] & ~row_valid[:, :-1]).any():
        raise ValueError("row_valid must be a prefix of True rows per image")

    r = width // 2
    acc = torch.float64 if cost_left.dtype == torch.float64 else torch.float32
    inf = torch.tensor(float("inf"), dtype=acc, device=dev)
    offs = torch.arange(-r, r + 1, device=dev)
    # Width of row for state (a, b): (right + b) - (left + a).
    row_width = ((boxes[:, 3] - boxes[:, 1])[:, None, None]
                 + offs[None, None, :] - offs[None, :, None])           # [N, A, B]
    width_ok = row_width >= min_width

    ok = (allowed_left[..., :, None] & allowed_right[..., None, :]
          & width_ok[:, None])                                          # [N, R, A, B]
    c = cost_left.to(acc)[..., :, None] + cost_right.to(acc)[..., None, :]
    if not torch.isfinite(c[ok & row_valid[..., None, None]]).all():
        raise ValueError("costs must be finite on allowed cells")
    c = torch.where(ok, c, inf)

    dp = torch.where(row_valid[:, :1, None], c[:, 0], torch.zeros_like(c[:, 0]))  # [N,A,B]
    back = torch.zeros((n, rows, width, width), dtype=torch.int64, device=dev)
    padded = torch.full((n, width + 2, width + 2), float("inf"), dtype=acc, device=dev)
    for y in range(1, rows):
        padded[:, 1:-1, 1:-1] = dp
        cands = []
        for sa, sb in _PAIR_SHIFTS:
            prev = padded[:, 1 + sa:1 + sa + width, 1 + sb:1 + sb + width]
            cands.append(prev + mu * (abs(sa) + abs(sb)))
        best, which = torch.stack(cands, -1).min(-1)   # first index wins ties
        valid = row_valid[:, y, None, None]
        back[:, y] = torch.where(valid, which, torch.zeros_like(which))  # 0 = straight
        dp = torch.where(valid, c[:, y] + best, dp)

    order = _pair_final_order(r, dev)
    total, pos = dp.flatten(1)[:, order].min(1)
    flat = order[pos]
    a, b = flat // width, flat % width
    feasible = torch.isfinite(total)

    shifts = torch.tensor(_PAIR_SHIFTS, device=dev)                     # [9, 2]
    left_path = torch.empty((n, rows), dtype=torch.int64, device=dev)
    right_path = torch.empty_like(left_path)
    left_path[:, -1], right_path[:, -1] = a, b
    idx = torch.arange(n, device=dev)
    for y in range(rows - 1, 0, -1):
        s = shifts[back[idx, y, a, b]]
        a, b = a + s[:, 0], b + s[:, 1]
        left_path[:, y - 1], right_path[:, y - 1] = a, b
    zero = torch.zeros_like(left_path)
    left_off = torch.where(feasible[:, None], left_path - r, zero)
    right_off = torch.where(feasible[:, None], right_path - r, zero)
    return left_off, right_off, total, feasible


def paths_to_mask(boxes, left_off, right_off, height, width):
    """Bool mask [N,1,H,W]: row top + i covers columns [left + a_i, right + b_i).

    Rows outside [top, bottom) are False. Offsets are per table row i, as
    returned by solve_pair_dp; padding rows beyond the box height are ignored.
    With all-zero offsets this equals the rectangle mask of `boxes`.
    """
    n = boxes.shape[0]
    if left_off.shape != right_off.shape or left_off.shape[0] != n:
        raise ValueError("offset tensors must be [N, R] and match boxes")
    dev = boxes.device
    top, left, bottom, right = boxes.unbind(-1)
    rows_needed = height
    pad = rows_needed - left_off.shape[1]
    if pad > 0:  # allow tables shorter than H (rows beyond are padding anyway)
        left_off = torch.cat((left_off, left_off.new_zeros(n, pad)), 1)
        right_off = torch.cat((right_off, right_off.new_zeros(n, pad)), 1)
    y = torch.arange(height, device=dev)
    i = (y[None, :] - top[:, None])                                      # [N, H]
    in_box = (i >= 0) & (y[None, :] < bottom[:, None])
    i = i.clamp(0, left_off.shape[1] - 1)
    row_left = left[:, None] + left_off.gather(1, i)                     # [N, H]
    row_right = right[:, None] + right_off.gather(1, i)
    x = torch.arange(width, device=dev)
    mask = ((x[None, None, :] >= row_left[..., None])
            & (x[None, None, :] < row_right[..., None]) & in_box[..., None])
    return mask[:, None]


# Per-sample outcome codes, stored as int64 in metadata["dp_status"].
STATUS_ACCEPTED = 0      # DP/random boundary kept (may equal the rectangle)
STATUS_NO_PATCH = 1      # empty base box (sample not mixed): nothing to do
STATUS_TOO_SMALL = 2     # base width < min_width: fallback to rectangle
STATUS_INFEASIBLE = 3    # no connected path under the constraints: fallback
STATUS_AREA = 4          # final area outside the budget: fallback
STATUS_NAMES = ("accepted", "no_patch", "too_small", "infeasible", "area")


def finalize_boundaries(boxes, left_off, right_off, feasible, height, width,
                        min_width=1, area_budget=0.1):
    """Apply the shared size/feasibility/area rules and build final masks.

    Used identically by E4 (DP paths) and E4-R (random paths), so both arms
    obey the same fallback behaviour.

    Args:
        boxes: int64 [N,4] base (E3) rectangles.
        left_off, right_off: int64 [N,R] candidate offsets per box row.
        feasible: bool [N] whether a valid path was found.
        min_width: rows narrower than this were forbidden; boxes whose base
                   width is below it are not reshaped ("too_small").
        area_budget: allowed |A_final - A_base| as a fraction of A_base (>= 0).

    Returns dict with:
        mask [N,1,H,W] bool, left_off/right_off [N,R] (zeroed on fallback),
        status [N] int64 (STATUS_*), area_base [N], area_final [N] (int64),
        changed [N] bool (final mask differs from the rectangle).
    """
    if not math.isfinite(area_budget) or area_budget < 0:
        raise ValueError("area_budget must be finite and nonnegative")
    n = boxes.shape[0]
    dev = boxes.device
    top, left, bottom, right = boxes.unbind(-1)
    box_h, box_w = (bottom - top).clamp_min(0), (right - left).clamp_min(0)
    area_base = box_h * box_w

    candidate = paths_to_mask(boxes, left_off, right_off, height, width)
    area_cand = candidate.flatten(1).sum(1)
    within = (area_cand - area_base).abs().double() <= area_budget * area_base.double() + 1e-9

    status = torch.full((n,), STATUS_ACCEPTED, dtype=torch.int64, device=dev)
    status = torch.where(~within, torch.full_like(status, STATUS_AREA), status)
    status = torch.where(~feasible, torch.full_like(status, STATUS_INFEASIBLE), status)
    status = torch.where(box_w < min_width, torch.full_like(status, STATUS_TOO_SMALL), status)
    status = torch.where(area_base == 0, torch.full_like(status, STATUS_NO_PATCH), status)

    keep = status == STATUS_ACCEPTED
    left_off = torch.where(keep[:, None], left_off, torch.zeros_like(left_off))
    right_off = torch.where(keep[:, None], right_off, torch.zeros_like(right_off))
    mask = paths_to_mask(boxes, left_off, right_off, height, width)
    rect = paths_to_mask(boxes, torch.zeros_like(left_off), torch.zeros_like(right_off),
                         height, width)
    changed = (mask != rect).flatten(1).any(1)
    return {"mask": mask, "left_offsets": left_off, "right_offsets": right_off,
            "status": status, "area_base": area_base,
            "area_final": mask.flatten(1).sum(1), "changed": changed}


def optimize_boundaries(source, target, boxes, radius=2, mu=0.0, min_width=1,
                        area_budget=0.1, eps=1e-3):
    """E4 core: DP-optimise the left/right edges of the E3 boxes.

    source/target: RGB [N,3,H,W] in [0,1] (source = images[perm]).
    Returns the finalize_boundaries dict plus:
        dp_objective [N] (normalised local objective, inf if infeasible),
        seam_cost / seam_cost_raw / seam_count [N]: full D_seam of the FINAL
            mask (all four sides and bend connectors), count==0 = n/a,
        seam_cost_base / seam_cost_raw_base / seam_count_base [N]: same for the
            E3 rectangle, for before/after diagnostics.
    """
    h, w = source.shape[-2:]
    cl, al, rv = edge_cost_table(source, target, boxes, "left", radius, eps)
    cr, ar, _ = edge_cost_table(source, target, boxes, "right", radius, eps)
    lo, ro, objective, feasible = solve_pair_dp(cl, al, cr, ar, boxes, mu, min_width, rv)
    out = finalize_boundaries(boxes, lo, ro, feasible, h, w, min_width, area_budget)
    d_hat, d_raw, count = normalized_seam_cost(source, target, out["mask"], eps)
    rect = paths_to_mask(boxes, torch.zeros_like(lo), torch.zeros_like(ro), h, w)
    b_hat, b_raw, b_count = normalized_seam_cost(source, target, rect, eps)
    out.update({"dp_objective": objective,
                "seam_cost": d_hat, "seam_cost_raw": d_raw, "seam_count": count,
                "seam_cost_base": b_hat, "seam_cost_raw_base": b_raw,
                "seam_count_base": b_count})
    return out


def seam_dp_cutmix(images, labels, alpha=1.0, p=0.5, k=4, gamma=0.5, beta=0.0,
                   seam_radius=2, seam_mu=0.0, min_width=2, area_budget=0.1,
                   seam_eps=1e-3):
    """E4 policy: E3 rectangle, then DP-optimised left/right edges.

    Returns mixed_images, labels_a, labels_b, lam_actual[N], metadata, with
    lam_actual computed from the FINAL mask. Metadata keeps all E3 fields
    (candidate scores etc.) under their E3 names, except that "mask",
    "policy" and the selected-box seam fields describe the final E4 mask;
    E3's rectangle values are kept as "seam_cost_base" etc.
    """
    if isinstance(seam_radius, bool) or not isinstance(seam_radius, int) or seam_radius < 0:
        raise ValueError("seam_radius must be a nonnegative integer")
    mixed, labels_a, labels_b, _, meta = seam_aware_cutmix(
        images, labels, alpha, p, k, gamma, beta, seam_eps)
    perm, boxes = meta["permutation"], meta["boxes"]
    out = optimize_boundaries(images[perm], images, boxes, seam_radius, seam_mu,
                              min_width, area_budget, seam_eps)
    mask = out["mask"]
    mixed = torch.where(mask, images[perm], images)
    lam = (1 - mask.flatten(1).float().mean(1)).to(images.dtype)
    meta = dict(meta)
    meta.update({
        "policy": "seam_dp", "mask": mask,
        "seam_radius": seam_radius, "seam_mu": seam_mu,
        "min_width": min_width, "area_budget": area_budget,
        "dp_status": out["status"], "dp_changed": out["changed"],
        "dp_objective": out["dp_objective"],
        "left_offsets": out["left_offsets"], "right_offsets": out["right_offsets"],
        "area_base": out["area_base"], "area_final": out["area_final"],
        "seam_cost": out["seam_cost"], "seam_cost_raw": out["seam_cost_raw"],
        "seam_count": out["seam_count"],
        "seam_cost_base": out["seam_cost_base"],
        "seam_cost_raw_base": out["seam_cost_raw_base"],
        "seam_count_base": out["seam_count_base"],
    })
    return mixed, labels_a, labels_b, lam, meta
