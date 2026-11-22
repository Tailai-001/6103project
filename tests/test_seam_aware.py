"""Tests for E3 seam_aware (Member C, step 2).

Run: .venv/bin/python -m pytest tests -v
"""
import pathlib
import sys
from unittest.mock import patch

import numpy as np
import pytest
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from augmentations import cutmix_batch  # noqa: E402
from augmentations.edge_utils import candidate_edge_scores  # noqa: E402
from augmentations.seam_aware import seam_aware_cutmix  # noqa: E402
from augmentations.seam_utils import (  # noqa: E402
    box_masks, image_seam_scale, normalized_seam_cost, seam_cost)


def seed():
    np.random.seed(42)
    torch.manual_seed(42)


@pytest.fixture
def batch():
    seed()
    return torch.rand(12, 3, 17, 23), torch.arange(12)


def call(x, y, policy, **kw):
    seed()
    return cutmix_batch(x, y, policy=policy, **kw)


# ---------- degeneracy and shared rules with E2 ----------

@pytest.mark.parametrize("k", [1, 4, 8])
def test_beta_zero_equals_dual_edge_exactly(batch, k):
    x, y = batch
    dual = call(x, y, "dual_edge", p=1, k=k, gamma=0.5)
    seam = call(x, y, "seam_aware", p=1, k=k, gamma=0.5, beta=0.0)
    for i in range(4):
        assert torch.equal(dual[i], seam[i])
    for key in ("boxes", "candidates", "selected_indices", "permutation"):
        assert torch.equal(dual[4][key], seam[4][key])


def test_same_candidates_and_area_as_dual_edge_for_any_beta(batch):
    # beta only changes WHICH candidate wins; candidates and areas are shared.
    x, y = batch
    dual = call(x, y, "dual_edge", p=1, k=8)
    seam = call(x, y, "seam_aware", p=1, k=8, beta=5.0)
    assert torch.equal(dual[4]["candidates"], seam[4]["candidates"])
    assert torch.equal(dual[3], seam[3])  # identical lam_actual (equal area)


def test_beta_does_not_consume_rng(batch):
    x, y = batch
    call(x, y, "seam_aware", p=1, k=8, beta=3.0)
    after_seam = torch.rand(5)
    call(x, y, "dual_edge", p=1, k=8)
    after_dual = torch.rand(5)
    assert torch.equal(after_seam, after_dual)


# ---------- selection is exactly argmax of the documented score ----------

def test_selection_matches_independent_recomputation(batch):
    x, y = batch
    beta, gamma = 2.0, 0.5
    _, _, _, _, m = call(x, y, "seam_aware", p=1, k=8, gamma=gamma, beta=beta)
    perm, cand = m["permutation"], m["candidates"]
    _, _, edge = candidate_edge_scores(x, perm, cand, gamma)
    scale = 0.5 * (image_seam_scale(x[perm]) + image_seam_scale(x))
    n, k = cand.shape[:2]
    for i in range(n):
        scores = []
        for j in range(k):
            mask = box_masks(cand[i:i + 1, j:j + 1], 17, 23)[:, 0]
            d, c = seam_cost(x[perm][i:i + 1], x[i:i + 1], mask)
            d_hat = (d / scale[i].clamp_min(1e-3)) if c.item() > 0 else torch.zeros(1)
            scores.append(edge[i, j].item() - beta * d_hat.item())
        assert m["selected_indices"][i].item() == int(np.argmax(scores))
        assert m["candidate_scores"][i].tolist() == pytest.approx(scores, abs=1e-5)


def test_prefers_low_seam_when_edges_tie():
    # Target: left half black, right half white. Source: all white.
    # Box 0 sits in the black region, box 1 in the white region; both have zero
    # edge scores, so E2 keeps box 0 (first index) but E3 must pick box 1.
    h = w = 8
    target = torch.zeros(3, h, w)
    target[:, :, 4:] = 1.0
    source = torch.ones(3, h, w)
    x = torch.stack([target, source])
    boxes = torch.tensor([[[2, 0, 6, 3], [2, 5, 6, 8]]] * 2)
    base = {"permutation": torch.tensor([1, 0]), "boxes": boxes[:, 0]}
    with patch("augmentations.seam_aware.sample_candidates", return_value=(boxes, base)):
        e2_like = seam_aware_cutmix(x, torch.arange(2), k=2, beta=0.0)[4]
        e3 = seam_aware_cutmix(x, torch.arange(2), k=2, beta=1.0)[4]
    assert e2_like["edge_scores"][0, 0] == e2_like["edge_scores"][0, 1]
    assert e2_like["selected_indices"][0].item() == 0
    assert e3["selected_indices"][0].item() == 1
    assert e3["seam_cost_raw"][0].item() == 0.0        # white pasted on white
    assert e3["seam_costs_raw"][0, 0].item() == pytest.approx(1.0)  # white on black


# ---------- output contract, same checks as E2 ----------

def test_pixels_labels_area_and_no_mutation(batch):
    x, y = batch
    before = x.clone()
    mixed, ya, yb, lam, m = call(x, y, "seam_aware", p=1, k=8, beta=1.0)
    assert torch.equal(x, before)
    assert torch.equal(mixed, torch.where(m["mask"], before[m["permutation"]], before))
    assert torch.equal(ya, y) and torch.equal(yb, y[m["permutation"]])
    assert torch.equal(lam, 1 - m["mask"].flatten(1).float().mean(1))
    assert m["mask"].dtype == torch.bool and m["policy"] == "seam_aware"
    for key in ("seam_cost", "seam_cost_raw", "seam_count"):
        assert m[key].shape == (12,)
    assert torch.isfinite(m["candidate_scores"]).all()


def test_p0_no_mixing_and_finite(batch):
    x, y = batch
    mixed, _, _, lam, m = call(x, y, "seam_aware", p=0, k=4, beta=1.0)
    assert torch.equal(mixed, x)
    assert torch.equal(lam, torch.ones(12))
    assert (m["seam_count"] == 0).all()   # empty masks -> not applicable
    assert torch.isfinite(m["candidate_scores"]).all()


def test_constant_images_and_singleton():
    x = torch.full((1, 3, 6, 6), 0.4)
    mixed, _, _, lam, m = seam_aware_cutmix(x, torch.zeros(1, dtype=torch.long),
                                            p=1, k=4, beta=10.0)
    assert torch.isfinite(m["candidate_scores"]).all()
    assert torch.isfinite(lam).all()


def test_invalid_parameters(batch):
    x, y = batch
    for bad in (-1.0, float("nan"), float("inf")):
        with pytest.raises(ValueError):
            call(x, y, "seam_aware", beta=bad)
    with pytest.raises(ValueError):
        call(x, y, "seam_aware", gamma=-1.0)


# ---------- normalisation helpers ----------

def test_image_seam_scale_known_values():
    x = torch.zeros(2, 3, 2, 2)
    x[1, :, :, 1] = 1.0  # two horizontal pairs differ by 1, vertical pairs equal
    scale = image_seam_scale(x)
    assert scale.tolist() == pytest.approx([0.0, 2 / 4])


def test_normalized_cost_finite_for_constant_but_different_images():
    source = torch.ones(1, 3, 6, 6)
    target = torch.zeros(1, 3, 6, 6)
    mask = box_masks(torch.tensor([[[1, 1, 4, 4]]]), 6, 6)[:, 0]
    d_hat, d_raw, count = normalized_seam_cost(source, target, mask, eps=1e-3)
    assert d_raw.item() == pytest.approx(1.0) and count.item() == 12
    assert torch.isfinite(d_hat).all()
    assert d_hat.item() == pytest.approx(1.0 / 1e-3)
