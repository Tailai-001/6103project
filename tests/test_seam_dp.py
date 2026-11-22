"""Tests for E4 seam_dp (Member C), step 1: single-boundary DP solver."""
import itertools
import pathlib
import sys

import pytest
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from augmentations.seam_dp import (  # noqa: E402
    STATUS_ACCEPTED, STATUS_AREA, STATUS_INFEASIBLE, STATUS_NO_PATCH,
    STATUS_TOO_SMALL, edge_cost_table, finalize_boundaries, optimize_boundaries,
    paths_to_mask, solve_boundary_dp, solve_pair_dp)
from augmentations.seam_utils import (  # noqa: E402
    box_masks, image_seam_scale, normalized_seam_cost, seam_cost)


def brute_force(cost, mu, allowed):
    """Enumerate every offset sequence for one image; return (best_total, best_paths)."""
    rows, width = cost.shape
    r = width // 2
    best, paths = float("inf"), []
    for seq in itertools.product(range(-r, r + 1), repeat=rows):
        if any(abs(a - b) > 1 for a, b in zip(seq, seq[1:])):
            continue
        if not all(allowed[y, d + r] for y, d in enumerate(seq)):
            continue
        total = sum(cost[y, d + r].item() for y, d in enumerate(seq))
        total += mu * sum(abs(a - b) for a, b in zip(seq, seq[1:]))
        if total < best - 1e-9:
            best, paths = total, [seq]
        elif abs(total - best) <= 1e-9:
            paths.append(seq)
    return best, paths


def path_objective(cost, mu, offsets):
    r = cost.shape[1] // 2
    total = sum(cost[y, d + r].item() for y, d in enumerate(offsets))
    return total + mu * sum(abs(a - b) for a, b in zip(offsets, offsets[1:]))


def test_worked_example_from_explanation():
    # Columns are offsets -1, 0, +1; mu = 0.1. Greedy per-row minima would be
    # 0, +1, -1, 0, which jumps 2 pixels between rows 1 and 2 (illegal).
    cost = torch.tensor([[[0.5, 0.2, 0.6],
                          [0.4, 0.7, 0.1],
                          [0.1, 0.8, 0.9],
                          [0.6, 0.3, 0.4]]])
    offsets, total, feasible = solve_boundary_dp(cost, mu=0.1)
    assert offsets.tolist() == [[0, -1, -1, 0]]
    assert total.item() == pytest.approx(1.2)
    assert feasible.tolist() == [True]
    # Straight E3 edge (all offsets 0) costs 2.0.
    assert path_objective(cost[0], 0.1, [0, 0, 0, 0]) == pytest.approx(2.0)


@pytest.mark.parametrize("rows,r,mu", [(1, 2, 0.3), (4, 1, 0.0), (5, 2, 0.2), (6, 2, 1.0), (4, 3, 0.05)])
def test_matches_brute_force(rows, r, mu):
    g = torch.Generator().manual_seed(rows * 10 + r)
    cost = torch.rand(8, rows, 2 * r + 1, generator=g)
    allowed = torch.rand(8, rows, 2 * r + 1, generator=g) > 0.2
    offsets, total, feasible = solve_boundary_dp(cost, mu, allowed)
    for i in range(8):
        best, paths = brute_force(cost[i], mu, allowed[i])
        if not paths:
            assert not feasible[i] and total[i].item() == float("inf")
            assert offsets[i].abs().sum().item() == 0
            continue
        assert feasible[i]
        assert total[i].item() == pytest.approx(best, abs=1e-5)
        assert tuple(offsets[i].tolist()) in paths


def test_returned_path_is_valid():
    g = torch.Generator().manual_seed(3)
    cost = torch.rand(16, 32, 5, generator=g)
    offsets, total, _ = solve_boundary_dp(cost, mu=0.1)
    assert offsets.min() >= -2 and offsets.max() <= 2
    assert (offsets[:, 1:] - offsets[:, :-1]).abs().max() <= 1
    for i in range(16):
        assert total[i].item() == pytest.approx(path_objective(cost[i], 0.1, offsets[i].tolist()), abs=1e-4)


def test_radius_zero_keeps_the_straight_edge():
    cost = torch.rand(4, 10, 1)
    offsets, total, feasible = solve_boundary_dp(cost, mu=0.5)
    assert offsets.eq(0).all() and feasible.all()
    assert torch.allclose(total, cost.sum((1, 2)))


def test_ties_prefer_the_original_edge():
    # Constant cost: every path ties on c, so mu > 0 or the tie rule must
    # keep the straight original edge.
    cost = torch.zeros(2, 6, 5)
    offsets, total, _ = solve_boundary_dp(cost, mu=0.0)
    assert offsets.eq(0).all() and total.eq(0).all()
    offsets, _, _ = solve_boundary_dp(cost, mu=0.3)
    assert offsets.eq(0).all()


def test_large_mu_straightens_the_path():
    g = torch.Generator().manual_seed(7)
    cost = torch.rand(4, 8, 5, generator=g)
    offsets, _, _ = solve_boundary_dp(cost, mu=100.0)
    assert (offsets[:, 1:] == offsets[:, :-1]).all()


def test_allowed_mask_forces_detour_and_infeasibility():
    cost = torch.zeros(2, 3, 3)
    allowed = torch.ones(2, 3, 3, dtype=torch.bool)
    allowed[0, 1, 1] = False                 # middle row may not use offset 0
    allowed[1, 1, :] = False                 # image 1: whole middle row blocked
    offsets, total, feasible = solve_boundary_dp(cost, mu=0.1, allowed=allowed)
    assert feasible.tolist() == [True, False]
    # Shifting the whole edge to -1 avoids the blocked cell with no bends
    # (cost 0), beating a 0,-1,0 detour (cost 0.2); left wins the tie with +1.
    assert offsets[0].tolist() == [-1, -1, -1]
    assert total[0].item() == 0.0
    assert offsets[1].eq(0).all() and total[1].item() == float("inf")


def test_unreachable_jump_is_infeasible():
    # Row 0 only allows -2, row 1 only allows +2: a 4-pixel jump.
    cost = torch.zeros(1, 2, 5)
    allowed = torch.zeros(1, 2, 5, dtype=torch.bool)
    allowed[0, 0, 0] = True
    allowed[0, 1, 4] = True
    _, _, feasible = solve_boundary_dp(cost, allowed=allowed)
    assert feasible.tolist() == [False]


def test_dtype_and_inputs():
    cost = torch.rand(2, 4, 3).half()
    offsets, total, _ = solve_boundary_dp(cost, mu=0.1)
    assert offsets.dtype == torch.int64 and total.dtype == torch.float32
    with pytest.raises(ValueError):
        solve_boundary_dp(torch.rand(1, 4, 4))           # even width
    with pytest.raises(ValueError):
        solve_boundary_dp(torch.rand(1, 4, 3), mu=-1.0)
    with pytest.raises(ValueError):
        solve_boundary_dp(torch.rand(1, 4, 3), allowed=torch.ones(1, 4, 3))
    bad = torch.rand(1, 4, 3)
    bad[0, 2, 1] = float("nan")
    with pytest.raises(ValueError):
        solve_boundary_dp(bad)


# ---------- step 2: padded rows in the solver ----------

def test_padding_rows_equal_truncated_problem():
    g = torch.Generator().manual_seed(11)
    cost = torch.rand(6, 7, 5, generator=g)
    allowed = torch.rand(6, 7, 5, generator=g) > 0.15
    lengths = [7, 5, 3, 1, 4, 0]
    row_valid = torch.arange(7)[None, :] < torch.tensor(lengths)[:, None]
    # Garbage in padding must not matter.
    cost = torch.where(row_valid[..., None], cost, torch.full_like(cost, 99.0))
    offsets, total, feasible = solve_boundary_dp(cost, 0.2, allowed | ~row_valid[..., None], row_valid)
    for i, L in enumerate(lengths):
        if L == 0:
            assert offsets[i].eq(0).all() and total[i].item() == 0 and feasible[i]
            continue
        o, t, f = solve_boundary_dp(cost[i:i + 1, :L], 0.2, allowed[i:i + 1, :L])
        assert f.item() == feasible[i].item()
        if f.item():
            assert torch.equal(offsets[i, :L], o[0])
            assert total[i].item() == pytest.approx(t.item(), abs=1e-6)
            assert offsets[i, L:].eq(offsets[i, L - 1]).all()   # padding copies last row


def test_row_valid_must_be_prefix():
    rv = torch.tensor([[True, False, True]])
    with pytest.raises(ValueError):
        solve_boundary_dp(torch.rand(1, 3, 3), row_valid=rv)


# ---------- step 2: local cost table c(y, d) ----------

def test_cost_table_hand_values():
    # 1 image, 2x4. Right edge of box rows 0..1, cols 1..3 (right=3).
    src = torch.zeros(1, 3, 2, 4)
    tgt = torch.zeros(1, 3, 2, 4)
    src[0, :, 0, 2] = 0.9      # row 0: source col 2 vs target col 3 -> 0.9
    tgt[0, :, 1, 3] = 0.4      # row 1: source col 2 vs target col 3 -> 0.4
    src[0, :, 0, 1] = 0.5      # row 0, d=-1: source col 1 vs target col 2 -> 0.5
    boxes = torch.tensor([[0, 1, 2, 3]])
    cost, allowed, row_valid = edge_cost_table(src, tgt, boxes, "right", 1, normalize=False)
    assert cost.shape == (1, 2, 3)
    assert cost[0, 0].tolist() == pytest.approx([0.5, 0.9, 0.0])   # d=+1 -> X=4=W border
    assert cost[0, 1].tolist() == pytest.approx([0.0, 0.4, 0.0])
    assert allowed[0].all() and row_valid[0].all()


def test_offset_zero_costs_sum_to_full_seam_for_full_height_box():
    # Full-height box -> no top/bottom seam, so left+right offset-0 costs
    # must add up to seam_cost total exactly (count = 2 * H pairs).
    g = torch.Generator().manual_seed(5)
    n, h, w = 4, 9, 12
    src, tgt = torch.rand(n, 3, h, w, generator=g), torch.rand(n, 3, h, w, generator=g)
    boxes = torch.tensor([[0, 2, h, 7], [0, 4, h, 10], [0, 1, h, 2], [0, 5, h, 11]])
    left, _, _ = edge_cost_table(src, tgt, boxes, "left", 2, normalize=False)
    right, _, _ = edge_cost_table(src, tgt, boxes, "right", 2, normalize=False)
    mask = box_masks(boxes[:, None], h, w)[:, 0]
    d, count = seam_cost(src, tgt, mask)
    assert count.tolist() == [2 * h] * n
    assert torch.allclose(left[:, :, 2].sum(1) + right[:, :, 2].sum(1), d * count, atol=1e-5)


def test_shifted_offsets_match_shifted_boxes():
    # c(y, d) for box B equals c(y, 0) for B with that edge moved by d.
    g = torch.Generator().manual_seed(8)
    src, tgt = torch.rand(1, 3, 10, 10, generator=g), torch.rand(1, 3, 10, 10, generator=g)
    box = torch.tensor([[2, 3, 8, 6]])
    for side, col in (("left", 1), ("right", 3)):
        table, _, _ = edge_cost_table(src, tgt, box, side, 2)
        for d in (-2, -1, 1, 2):
            moved = box.clone()
            moved[0, col] += d
            ref, _, _ = edge_cost_table(src, tgt, moved, side, 0)
            assert torch.allclose(table[0, :6, d + 2], ref[0, :6, 0])


def test_rows_align_with_box_top_and_padding():
    src, tgt = torch.rand(1, 3, 8, 8), torch.rand(1, 3, 8, 8)
    box = torch.tensor([[3, 2, 6, 5]])
    cost, allowed, row_valid = edge_cost_table(src, tgt, box, "right", 1, normalize=False)
    assert row_valid[0].tolist() == [True] * 3 + [False] * 5
    for i in range(3):
        y = 3 + i
        expected = (src[0, :, y, 4] - tgt[0, :, y, 5]).abs().mean()
        assert cost[0, i, 1].item() == pytest.approx(expected.item())
    assert cost[0, 3:].eq(0).all() and allowed[0, 3:].all()


def test_image_border_rules():
    src, tgt = torch.rand(2, 3, 6, 6), torch.rand(2, 3, 6, 6)
    boxes = torch.tensor([[1, 0, 4, 6], [1, 1, 4, 5]])
    for side in ("left", "right"):
        cost, allowed, _ = edge_cost_table(src, tgt, boxes, side, 2, normalize=False)
        if side == "right":   # box 0 right=6=W: d>0 leaves image, d=0 costs 0
            assert allowed[0, :3].tolist() == [[True, True, True, False, False]] * 3
            assert cost[0, :3, 2].eq(0).all()
            assert allowed[1, :3].tolist() == [[True, True, True, True, False]] * 3  # X<=W
            assert cost[1, :3, 3].eq(0).all()                                     # X=W border
        else:                 # box 0 left=0: d<0 leaves image, d=0 costs 0
            assert allowed[0, :3].tolist() == [[False, False, True, True, True]] * 3
            assert cost[0, :3, 2].eq(0).all()
            assert allowed[1, :3].tolist() == [[False, True, True, True, True]] * 3  # L>=0
            assert cost[1, :3, 1].eq(0).all()


def test_normalisation_uses_e3_scale():
    g = torch.Generator().manual_seed(2)
    src, tgt = torch.rand(2, 3, 7, 7, generator=g), torch.rand(2, 3, 7, 7, generator=g)
    boxes = torch.tensor([[1, 1, 5, 5], [0, 2, 7, 6]])
    raw, _, _ = edge_cost_table(src, tgt, boxes, "right", 1, normalize=False)
    norm, _, _ = edge_cost_table(src, tgt, boxes, "right", 1)
    scale = 0.5 * (image_seam_scale(src) + image_seam_scale(tgt))
    assert torch.allclose(norm, raw / scale[:, None, None])
    const = torch.full((1, 3, 5, 5), 0.3)
    c, _, _ = edge_cost_table(const, const, torch.tensor([[1, 1, 4, 4]]), "left", 1)
    assert torch.isfinite(c).all() and c.eq(0).all()


def test_cost_table_feeds_solver_and_radius_zero():
    g = torch.Generator().manual_seed(4)
    src, tgt = torch.rand(5, 3, 12, 12, generator=g), torch.rand(5, 3, 12, 12, generator=g)
    boxes = torch.tensor([[0, 0, 12, 6], [2, 3, 9, 8], [4, 4, 5, 9], [1, 6, 11, 12], [3, 3, 3, 3]])
    for side in ("left", "right"):
        cost, allowed, rv = edge_cost_table(src, tgt, boxes, side, 2)
        offsets, total, feasible = solve_boundary_dp(cost, 0.1, allowed, rv)
        assert feasible.all() and torch.isfinite(total).all()
        base = boxes[:, 1 if side == "left" else 3]
        pos = base[:, None] + offsets
        lo, hi = (0, 11) if side == "left" else (1, 12)
        assert ((pos >= lo) & (pos <= hi)).all()
        cost0, allowed0, rv0 = edge_cost_table(src, tgt, boxes, side, 0)
        offsets0, _, _ = solve_boundary_dp(cost0, 0.1, allowed0, rv0)
        assert offsets0.eq(0).all()


def test_cost_table_bad_inputs():
    x = torch.rand(1, 3, 4, 4)
    box = torch.tensor([[0, 0, 2, 2]])
    with pytest.raises(ValueError):
        edge_cost_table(x, x, box, "top", 1)
    with pytest.raises(ValueError):
        edge_cost_table(x, x, box, "left", -1)
    with pytest.raises(ValueError):
        edge_cost_table(x, x, box.int(), "left", 1)


# ---------- step 3: joint left/right edges ----------

def brute_force_pair(cl, al, cr, ar, base_width, mu, min_width):
    rows, width = cl.shape
    r = width // 2
    seqs = [seq for seq in itertools.product(range(-r, r + 1), repeat=rows)
            if all(abs(x - y) <= 1 for x, y in zip(seq, seq[1:]))]
    best, sols = float("inf"), []
    for sa in seqs:
        if not all(al[y, d + r] for y, d in enumerate(sa)):
            continue
        for sb in seqs:
            if not all(ar[y, d + r] for y, d in enumerate(sb)):
                continue
            if any(base_width + b - a < min_width for a, b in zip(sa, sb)):
                continue
            t = path_objective(cl, mu, list(sa)) + path_objective(cr, mu, list(sb))
            if t < best - 1e-9:
                best, sols = t, [(sa, sb)]
            elif abs(t - best) <= 1e-9:
                sols.append((sa, sb))
    return best, sols


@pytest.mark.parametrize("rows,r,base_width,min_width,mu",
                         [(4, 1, 3, 2, 0.1), (3, 2, 2, 1, 0.0), (4, 1, 2, 2, 0.3), (2, 2, 1, 1, 0.2), (1, 1, 1, 1, 0.0)])
def test_pair_dp_matches_brute_force(rows, r, base_width, min_width, mu):
    g = torch.Generator().manual_seed(rows + 7 * r + 13 * base_width)
    n, w = 6, 2 * r + 1
    cl, cr = torch.rand(n, rows, w, generator=g), torch.rand(n, rows, w, generator=g)
    al = torch.rand(n, rows, w, generator=g) > 0.15
    ar = torch.rand(n, rows, w, generator=g) > 0.15
    boxes = torch.tensor([[0, 5, rows, 5 + base_width]] * n)
    lo, ro, total, feasible = solve_pair_dp(cl, al, cr, ar, boxes, mu, min_width)
    for i in range(n):
        best, sols = brute_force_pair(cl[i], al[i], cr[i], ar[i], base_width, mu, min_width)
        if not sols:
            assert not feasible[i] and total[i].item() == float("inf")
            assert lo[i].eq(0).all() and ro[i].eq(0).all()
            continue
        assert feasible[i]
        assert total[i].item() == pytest.approx(best, abs=1e-5)
        assert (tuple(lo[i].tolist()), tuple(ro[i].tolist())) in sols


def test_pair_dp_equals_separate_dp_when_width_is_slack():
    g = torch.Generator().manual_seed(21)
    cl, cr = torch.rand(5, 9, 5, generator=g), torch.rand(5, 9, 5, generator=g)
    ones = torch.ones_like(cl, dtype=torch.bool)
    boxes = torch.tensor([[0, 2, 9, 20]] * 5)          # width 18 >> 2r
    lo, ro, total, _ = solve_pair_dp(cl, ones, cr, ones, boxes, mu=0.15)
    l1, tl, _ = solve_boundary_dp(cl, 0.15)
    r1, tr, _ = solve_boundary_dp(cr, 0.15)
    assert torch.allclose(total, tl + tr, atol=1e-5)
    assert torch.equal(lo, l1) and torch.equal(ro, r1)   # same tie rules


def test_pair_dp_never_crosses_and_respects_min_width():
    g = torch.Generator().manual_seed(9)
    # Costs push the left edge right and the right edge left (towards crossing).
    n, rows, r = 8, 10, 2
    cl = torch.rand(n, rows, 5, generator=g) * 0.1 + torch.linspace(1, 0, 5)
    cr = torch.rand(n, rows, 5, generator=g) * 0.1 + torch.linspace(0, 1, 5)
    ones = torch.ones_like(cl, dtype=torch.bool)
    boxes = torch.tensor([[0, 4, rows, 7]] * n)        # base width 3
    for mw in (1, 2, 3):
        lo, ro, _, feasible = solve_pair_dp(cl, ones, cr, ones, boxes, 0.05, mw)
        assert feasible.all()
        assert ((3 + ro - lo) >= mw).all()
        assert (lo[:, 1:] - lo[:, :-1]).abs().max() <= 1
        assert (ro[:, 1:] - ro[:, :-1]).abs().max() <= 1
    # Base width 3 with r=2 reaches at most 3 + 2 + 2 = 7, so min_width 8 is infeasible.
    _, _, total, feasible = solve_pair_dp(cl, ones, cr, ones, boxes, 0.05, 8)
    assert not feasible.any() and torch.isinf(total).all()


def test_pair_dp_radius_zero_and_padding():
    g = torch.Generator().manual_seed(1)
    src, tgt = torch.rand(4, 3, 10, 10, generator=g), torch.rand(4, 3, 10, 10, generator=g)
    boxes = torch.tensor([[0, 0, 10, 5], [2, 3, 7, 9], [5, 5, 6, 6], [4, 4, 4, 4]])
    cl, al, rv = edge_cost_table(src, tgt, boxes, "left", 0)
    cr, ar, _ = edge_cost_table(src, tgt, boxes, "right", 0)
    lo, ro, _, feasible = solve_pair_dp(cl, al, cr, ar, boxes, 0.2, 1, rv)
    assert feasible.all() and lo.eq(0).all() and ro.eq(0).all()
    mask = paths_to_mask(boxes, lo, ro, 10, 10)
    assert torch.equal(mask[:, 0], box_masks(boxes[:, None], 10, 10)[:, 0])


def horizontal_seam_sum(src, tgt, mask):
    """Sum of |src(p)-tgt(q)| channel-mean over left/right neighbour seam pairs."""
    m = mask[:, 0]
    mixed = torch.where(mask, src, tgt)
    diff = (mixed[..., 1:] - mixed[..., :-1]).abs().mean(1)
    return (diff * (m[..., 1:] != m[..., :-1])).flatten(1).sum(1)


def test_objective_equals_horizontal_seam_of_final_mask():
    # With mu = 0 the DP objective (raw costs) is exactly the sum of all
    # left/right-neighbour seam pairs of the mask it produces; top/bottom
    # edges and bend connectors are vertical pairs, deliberately not in it.
    g = torch.Generator().manual_seed(17)
    n, h, w, r = 6, 16, 16, 2
    src, tgt = torch.rand(n, 3, h, w, generator=g), torch.rand(n, 3, h, w, generator=g)
    boxes = torch.tensor([[2, 3, 12, 9], [0, 0, 16, 8], [5, 10, 11, 16],
                          [1, 1, 3, 15], [4, 6, 13, 8], [7, 7, 8, 9]])
    cl, al, rv = edge_cost_table(src, tgt, boxes, "left", r, normalize=False)
    cr, ar, _ = edge_cost_table(src, tgt, boxes, "right", r, normalize=False)
    lo, ro, total, feasible = solve_pair_dp(cl, al, cr, ar, boxes, 0.0, 1, rv)
    assert feasible.all()
    mask = paths_to_mask(boxes, lo, ro, h, w)
    assert torch.allclose(total, horizontal_seam_sum(src, tgt, mask), atol=1e-5)
    # And the optimised mask never has a higher horizontal seam sum than the box.
    box = box_masks(boxes[:, None], h, w)[:, :1]
    assert (horizontal_seam_sum(src, tgt, mask) <= horizontal_seam_sum(src, tgt, box) + 1e-6).all()


def test_paths_to_mask_rows_are_contiguous_and_in_box_rows():
    boxes = torch.tensor([[1, 2, 4, 6]])
    lo = torch.tensor([[0, -1, 1]])
    ro = torch.tensor([[1, 0, -1]])
    mask = paths_to_mask(boxes, lo, ro, 6, 8)[0, 0]
    expected = torch.zeros(6, 8, dtype=torch.bool)
    expected[1, 2:7] = True
    expected[2, 1:6] = True
    expected[3, 3:5] = True
    assert torch.equal(mask, expected)


def test_pair_dp_bad_inputs():
    c = torch.rand(1, 3, 3)
    ok = torch.ones(1, 3, 3, dtype=torch.bool)
    box = torch.tensor([[0, 0, 3, 3]])
    with pytest.raises(ValueError):
        solve_pair_dp(c, ok, c, ok, box, min_width=0)
    with pytest.raises(ValueError):
        solve_pair_dp(c, ok, c, ok, box, mu=-0.1)
    with pytest.raises(ValueError):
        solve_pair_dp(c, ok, torch.rand(1, 3, 5), torch.ones(1, 3, 5, dtype=torch.bool), box)


# ---------- step 4: area budget, fallback, full-seam recomputation ----------

def smooth_pair(n, h=32, w=32, seed=0):
    g = torch.Generator().manual_seed(seed)
    up = lambda: torch.nn.functional.interpolate(
        torch.rand(n, 3, h // 4, w // 4, generator=g), size=(h, w),
        mode="bilinear", align_corners=False)
    return up(), up()


BOXES = torch.tensor([[4, 4, 20, 18], [0, 10, 32, 26], [8, 0, 24, 9], [3, 3, 3, 9],
                      [10, 12, 22, 13], [2, 2, 30, 30], [12, 20, 28, 32], [5, 6, 9, 30]])


def test_radius_zero_is_exactly_the_e3_rectangle():
    src, tgt = smooth_pair(8)
    out = optimize_boundaries(src, tgt, BOXES, radius=0, mu=0.1, min_width=2)
    rect = box_masks(BOXES[:, None], 32, 32)
    assert torch.equal(out["mask"], rect)
    assert not out["changed"].any()
    assert torch.equal(out["area_final"], out["area_base"])
    assert torch.allclose(out["seam_cost"], out["seam_cost_base"])


def test_status_codes_and_fallback_masks():
    src, tgt = smooth_pair(8)
    out = optimize_boundaries(src, tgt, BOXES, radius=2, mu=0.0, min_width=2, area_budget=0.1)
    st = out["status"]
    assert st[3].item() == STATUS_NO_PATCH            # zero-height box
    assert st[4].item() == STATUS_TOO_SMALL           # width 1 < min_width 2
    rect = box_masks(BOXES[:, None], 32, 32)
    fallback = st != STATUS_ACCEPTED
    assert torch.equal(out["mask"][fallback], rect[fallback])
    assert out["left_offsets"][fallback].eq(0).all() and out["right_offsets"][fallback].eq(0).all()


def test_accepted_masks_respect_budget_and_geometry():
    src, tgt = smooth_pair(64, seed=3)
    g = torch.Generator().manual_seed(4)
    t = torch.randint(0, 16, (64,), generator=g)
    l = torch.randint(0, 16, (64,), generator=g)
    boxes = torch.stack((t, l, t + torch.randint(4, 17, (64,), generator=g),
                         l + torch.randint(4, 17, (64,), generator=g)), 1)
    for budget in (0.0, 0.05, 0.2):
        out = optimize_boundaries(src, tgt, boxes, radius=2, mu=0.05, min_width=2, area_budget=budget)
        acc = out["status"] == STATUS_ACCEPTED
        diff = (out["area_final"] - out["area_base"]).abs().double()
        assert (diff[acc] <= budget * out["area_base"][acc].double() + 1e-9).all()
        lo, ro = out["left_offsets"], out["right_offsets"]
        h = (boxes[:, 2] - boxes[:, 0])
        for i in range(64):
            k = h[i].item()
            a, b = lo[i, :k], ro[i, :k]
            assert ((boxes[i, 3] - boxes[i, 1]) + b - a >= 2).all()          # min width
            assert (a[1:] - a[:-1]).abs().max().item() <= 1 if k > 1 else True
            assert (a.abs() <= 2).all() and (b.abs() <= 2).all()
        m = out["mask"][:, 0]
        rows_outside = torch.arange(32)[None, :, None]
        outside = (rows_outside < boxes[:, 0, None, None]) | (rows_outside >= boxes[:, 2, None, None])
        assert not (m & outside).any()                                         # top/bottom fixed


def test_zero_budget_only_keeps_area_neutral_paths_and_huge_budget_never_area_fails():
    src, tgt = smooth_pair(8, seed=5)
    tight = optimize_boundaries(src, tgt, BOXES, radius=2, mu=0.0, min_width=2, area_budget=0.0)
    acc = tight["status"] == STATUS_ACCEPTED
    assert torch.equal(tight["area_final"][acc], tight["area_base"][acc])
    loose = optimize_boundaries(src, tgt, BOXES, radius=2, mu=0.0, min_width=2, area_budget=1e6)
    assert not (loose["status"] == STATUS_AREA).any()


def test_infeasible_falls_back():
    src, tgt = smooth_pair(2, seed=6)
    boxes = torch.tensor([[2, 2, 10, 6], [4, 4, 12, 9]])       # widths 4 and 5
    out = optimize_boundaries(src, tgt, boxes, radius=1, min_width=8, area_budget=1.0)
    # widths < 8 -> too_small takes precedence over infeasible
    assert out["status"].tolist() == [STATUS_TOO_SMALL] * 2
    # Force infeasibility with a valid base width via finalize directly.
    lo = torch.zeros(2, 32, dtype=torch.long)
    res = finalize_boundaries(boxes, lo, lo, torch.tensor([False, True]), 32, 32, 1, 0.1)
    assert res["status"].tolist() == [STATUS_INFEASIBLE, STATUS_ACCEPTED]
    assert torch.equal(res["mask"], box_masks(boxes[:, None], 32, 32))


def test_full_seam_is_recomputed_on_final_mask():
    src, tgt = smooth_pair(8, seed=7)
    out = optimize_boundaries(src, tgt, BOXES, radius=2, mu=0.05, min_width=2, area_budget=0.2)
    d_hat, d_raw, count = normalized_seam_cost(src, tgt, out["mask"])
    assert torch.allclose(out["seam_cost"], d_hat) and torch.equal(out["seam_count"], count)
    assert torch.allclose(out["seam_cost_raw"], d_raw)
    rect = box_masks(BOXES[:, None], 32, 32)
    b_hat, _, b_count = normalized_seam_cost(src, tgt, rect)
    assert torch.allclose(out["seam_cost_base"], b_hat) and torch.equal(out["seam_count_base"], b_count)


def test_finalize_rejects_bad_budget():
    z = torch.zeros(1, 4, dtype=torch.long)
    with pytest.raises(ValueError):
        finalize_boundaries(torch.tensor([[0, 0, 4, 4]]), z, z, torch.tensor([True]), 8, 8, 1, -0.1)
