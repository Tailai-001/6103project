"""Tests for the E4 seam_dp policy (Member C, step 5): integration and contract."""
import pathlib
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from augmentations import cutmix_batch  # noqa: E402
from augmentations.seam_dp import STATUS_ACCEPTED, STATUS_NO_PATCH  # noqa: E402
from augmentations.seam_utils import normalized_seam_cost  # noqa: E402


def seed():
    np.random.seed(42)
    torch.manual_seed(42)


def smooth_batch(n=16, seed_=0):
    g = torch.Generator().manual_seed(seed_)
    x = torch.nn.functional.interpolate(torch.rand(n, 3, 8, 8, generator=g), size=32,
                                        mode="bilinear", align_corners=False)
    return x, torch.arange(n) % 100


def call(x, y, policy, **kw):
    seed()
    return cutmix_batch(x, y, policy=policy, **kw)


E4 = dict(p=1, k=4, gamma=0.5, beta=1.0, seam_radius=2, seam_mu=0.05,
          seam_min_width=2, seam_area_budget=0.2)


def test_radius_zero_equals_seam_aware_exactly():
    x, y = smooth_batch()
    e3 = call(x, y, "seam_aware", p=1, k=4, gamma=0.5, beta=1.0)
    e4 = call(x, y, "seam_dp", **{**E4, "seam_radius": 0})
    for i in range(4):
        assert torch.equal(e3[i], e4[i])
    for key in ("mask", "boxes", "selected_indices", "permutation", "candidates"):
        assert torch.equal(e3[4][key], e4[4][key])
    assert torch.allclose(e3[4]["seam_cost"], e4[4]["seam_cost"])


def test_default_cutmix_batch_radius_is_zero():
    # cutmix_batch keeps seam_radius=0 by default, so an E4 config MUST set it.
    x, y = smooth_batch()
    e3 = call(x, y, "seam_aware", p=1, beta=1.0)
    e4 = call(x, y, "seam_dp", p=1, beta=1.0)
    assert torch.equal(e3[0], e4[0])


def test_shares_e3_selection_and_rng():
    x, y = smooth_batch()
    e3 = call(x, y, "seam_aware", p=1, k=4, gamma=0.5, beta=1.0)
    after_e3 = torch.rand(4)
    e4 = call(x, y, "seam_dp", **E4)
    after_e4 = torch.rand(4)
    assert torch.equal(after_e3, after_e4)                       # no extra RNG
    assert torch.equal(e3[4]["boxes"], e4[4]["boxes"])          # same base rectangle
    assert torch.equal(e3[4]["permutation"], e4[4]["permutation"])
    assert torch.allclose(e3[4]["seam_cost"], e4[4]["seam_cost_base"])


def test_output_contract_and_labels_from_final_mask():
    x, y = smooth_batch(32, seed_=1)
    before = x.clone()
    mixed, ya, yb, lam, m = call(x, y, "seam_dp", **E4)
    assert torch.equal(x, before)                                # no mutation
    perm = m["permutation"]
    assert torch.equal(mixed, torch.where(m["mask"], before[perm], before))
    assert torch.equal(ya, y) and torch.equal(yb, y[perm])
    assert torch.equal(lam, 1 - m["mask"].flatten(1).float().mean(1))
    assert torch.equal(m["area_final"], m["mask"].flatten(1).sum(1))
    assert m["policy"] == "seam_dp" and m["mask"].dtype == torch.bool
    assert ((m["dp_status"] >= 0) & (m["dp_status"] <= 4)).all()
    assert (m["dp_changed"] & (m["dp_status"] == STATUS_ACCEPTED)).any()   # E4 actually acts
    d_hat, _, count = normalized_seam_cost(before[perm], before, m["mask"])
    assert torch.allclose(m["seam_cost"], d_hat) and torch.equal(m["seam_count"], count)


def test_p0_means_no_patch_and_no_mixing():
    x, y = smooth_batch()
    mixed, _, _, lam, m = call(x, y, "seam_dp", **{**E4, "p": 0})
    assert torch.equal(mixed, x) and lam.eq(1).all()
    assert (m["dp_status"] == STATUS_NO_PATCH).all()


def test_invalid_parameters():
    x, y = smooth_batch(4)
    for bad in ({"seam_radius": -1}, {"seam_radius": 1.5}, {"seam_mu": -0.1},
                {"seam_min_width": 0}, {"seam_area_budget": -0.1}, {"beta": -1.0}):
        with pytest.raises(ValueError):
            call(x, y, "seam_dp", **{**E4, **bad})


def test_one_real_training_step_through_train_py():
    # Uses the shared training loop and the real CIFAR ResNet-18 on a fake
    # 1-batch loader (no dataset download): proves seam_dp is wired end to end.
    train = pytest.importorskip("train")
    from models.resnet import build_resnet18
    x, y = smooth_batch(8, seed_=2)
    model = build_resnet18(num_classes=100)
    opt = torch.optim.SGD(model.parameters(), lr=0.01)
    aug_cfg = {"policy": "seam_dp", "alpha": 1.0, "p": 1.0, "k": 4, "gamma": 0.5,
               "beta": 1.0, "seam_radius": 2, "seam_mu": 0.05,
               "seam_min_width": 2, "seam_area_budget": 0.2}
    seed()
    stats = train.train_one_epoch(model, [(x, y)], opt, torch.device("cpu"), aug_cfg)
    assert np.isfinite(stats["loss"])
