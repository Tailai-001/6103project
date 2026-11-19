"""Tests for augmentations/seam_utils.py (Member C, step 1: seam cost)."""
import pathlib
import sys

import pytest
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from augmentations.seam_utils import box_masks, seam_cost  # noqa: E402


def reference_seam_cost(source, target, mask):
    """Slow, literal implementation of the README formula for one image.

    Enumerates every in-image 4-neighbour pair (p, q) with M(p)=1, M(q)=0 and
    averages |x_B,c(p) - x_A,c(q)| over pairs and the 3 channels.
    """
    _, h, w = source.shape
    total, count = 0.0, 0
    for y in range(h):
        for x in range(w):
            if not mask[y, x]:
                continue
            for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                qy, qx = y + dy, x + dx
                if 0 <= qy < h and 0 <= qx < w and not mask[qy, qx]:
                    total += (source[:, y, x] - target[:, qy, qx]).abs().mean().item()
                    count += 1
    return (total / count if count else 0.0), count


def test_hand_computed_single_pixel():
    # 3x3 image, mask = centre pixel only -> 4 seam pairs (up/down/left/right).
    source = torch.zeros(1, 3, 3, 3)
    source[0, :, 1, 1] = 1.0
    target = torch.zeros(1, 3, 3, 3)
    target[0, :, 0, 1] = 0.8   # up    -> |1-0.8| = 0.2
    target[0, :, 2, 1] = 1.0   # down  -> 0.0
    target[0, :, 1, 0] = 0.5   # left  -> 0.5
    target[0, :, 1, 2] = 0.3   # right -> 0.7
    mask = torch.zeros(1, 3, 3, dtype=torch.bool)
    mask[0, 1, 1] = True
    cost, count = seam_cost(source, target, mask)
    assert count.tolist() == [4]
    assert cost.item() == pytest.approx((0.2 + 0.0 + 0.5 + 0.7) / 4)


def test_channels_are_averaged():
    # Only the red channel differs by 0.9 -> per-pair cost 0.9 / 3.
    source = torch.zeros(1, 3, 1, 2)
    target = torch.zeros(1, 3, 1, 2)
    source[0, 0, 0, 0] = 0.9
    mask = torch.tensor([[[True, False]]])
    cost, count = seam_cost(source, target, mask)
    assert count.item() == 1
    assert cost.item() == pytest.approx(0.3)


def test_matches_reference_on_random_masks():
    g = torch.Generator().manual_seed(0)
    source = torch.rand(6, 3, 9, 7, generator=g)
    target = torch.rand(6, 3, 9, 7, generator=g)
    mask = torch.rand(6, 9, 7, generator=g) > 0.5
    cost, count = seam_cost(source, target, mask)
    for i in range(6):
        ref_cost, ref_count = reference_seam_cost(source[i], target[i], mask[i])
        assert count[i].item() == ref_count
        assert cost[i].item() == pytest.approx(ref_cost, abs=1e-6)


def test_interior_rectangle_count_is_perimeter():
    # A 3x4 box strictly inside an 8x8 image has 2*3 + 2*4 = 14 seam pairs.
    boxes = torch.tensor([[[2, 2, 5, 6]]])
    mask = box_masks(boxes, 8, 8)[:, 0]
    x = torch.rand(1, 3, 8, 8)
    _, count = seam_cost(x, torch.rand(1, 3, 8, 8), mask)
    assert count.item() == 14


def test_border_sides_are_not_counted():
    # Box touching top and left borders: only bottom (4) and right (3) sides count.
    boxes = torch.tensor([[[0, 0, 3, 4]]])
    mask = box_masks(boxes, 8, 8)[:, 0]
    _, count = seam_cost(torch.rand(1, 3, 8, 8), torch.rand(1, 3, 8, 8), mask)
    assert count.item() == 4 + 3
    # Full-height left half: only the single vertical boundary (8 pairs).
    boxes = torch.tensor([[[0, 0, 8, 4]]])
    mask = box_masks(boxes, 8, 8)[:, 0]
    _, count = seam_cost(torch.rand(1, 3, 8, 8), torch.rand(1, 3, 8, 8), mask)
    assert count.item() == 8


def test_empty_and_full_masks_are_not_applicable():
    source, target = torch.rand(2, 3, 5, 5), torch.rand(2, 3, 5, 5)
    mask = torch.zeros(2, 5, 5, dtype=torch.bool)
    mask[1] = True
    cost, count = seam_cost(source, target, mask)
    assert count.tolist() == [0, 0]
    assert cost.tolist() == [0.0, 0.0]
    assert torch.isfinite(cost).all()


def test_same_constant_colour_has_zero_cost():
    # The seam compares *different* neighbouring pixels p and q, so pasting a
    # textured image into itself is not zero cost (it measures the image's own
    # local variation). Zero is expected only when both sides share one colour.
    colour = torch.tensor([0.2, 0.5, 0.9])[None, :, None, None]
    source = colour.expand(1, 3, 6, 6).clone()
    target = colour.expand(1, 3, 6, 6).clone()
    mask = box_masks(torch.tensor([[[1, 1, 4, 4]]]), 6, 6)[:, 0]
    cost, count = seam_cost(source, target, mask)
    assert count.item() > 0
    assert cost.item() == 0.0


def test_box_masks_match_slicing():
    boxes = torch.tensor([[[1, 2, 4, 7], [0, 0, 0, 3]], [[3, 0, 6, 6], [2, 2, 3, 3]]])
    masks = box_masks(boxes, 6, 7)
    assert masks.shape == (2, 2, 6, 7) and masks.dtype == torch.bool
    for i in range(2):
        for k in range(2):
            t, l, b, r = boxes[i, k].tolist()
            expected = torch.zeros(6, 7, dtype=torch.bool)
            expected[t:b, l:r] = True
            assert torch.equal(masks[i, k], expected)


def test_mask_shapes_and_half_precision():
    source = torch.rand(2, 3, 4, 4).half()
    target = torch.rand(2, 3, 4, 4).half()
    mask = torch.rand(2, 4, 4) > 0.5
    c1, n1 = seam_cost(source, target, mask)
    c2, n2 = seam_cost(source, target, mask[:, None])
    assert torch.equal(c1, c2) and torch.equal(n1, n2)
    assert c1.dtype == torch.float32 and n1.dtype == torch.int64


def test_rejects_bad_inputs():
    x = torch.rand(1, 3, 4, 4)
    with pytest.raises(ValueError):
        seam_cost(x, x, torch.ones(1, 4, 4))            # float mask
    with pytest.raises(ValueError):
        seam_cost(x, x[:, :, :3], torch.ones(1, 4, 4, dtype=torch.bool))
    with pytest.raises(ValueError):
        seam_cost(x[:, :1], x[:, :1], torch.ones(1, 4, 4, dtype=torch.bool))
