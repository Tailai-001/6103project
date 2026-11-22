"""Tests for E4-R seam_random (Member C): matched random-boundary control."""
import pathlib
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from augmentations import cutmix_batch  # noqa: E402
from augmentations.seam_dp import (  # noqa: E402
    STATUS_ACCEPTED, STATUS_AREA, STATUS_NO_PATCH, edge_constraints, edge_cost_table)
from augmentations.seam_random import random_boundaries  # noqa: E402
from augmentations.seam_utils import box_masks, normalized_seam_cost  # noqa: E402


def seed(s=42):
    np.random.seed(s)
    torch.manual_seed(s)


def smooth_batch(n=32, seed_=0):
    g = torch.Generator().manual_seed(seed_)
    x = torch.nn.functional.interpolate(torch.rand(n, 3, 8, 8, generator=g), size=32,
                                        mode="bilinear", align_corners=False)
    return x, torch.arange(n) % 100


def call(x, y, policy, s=42, **kw):
    seed(s)
    return cutmix_batch(x, y, policy=policy, **kw)


CFG = dict(p=1, k=4, gamma=0.5, beta=1.0, seam_radius=2, seam_mu=0.05,
           seam_min_width=2, seam_area_budget=0.2)


def test_constraints_helper_matches_cost_table():
    x, _ = smooth_batch(4)
    boxes = torch.tensor([[0, 0, 32, 10], [5, 20, 9, 32], [3, 3, 3, 9], [10, 1, 30, 31]])
    for side in ("left", "right"):
        _, allowed, rv = edge_cost_table(x, x.flip(0), boxes, side, 2)
        a2, rv2 = edge_constraints(boxes, 32, 32, side, 2)
        assert torch.equal(allowed, a2) and torch.equal(rv, rv2)


def test_radius_zero_equals_seam_aware_exactly():
    x, y = smooth_batch()
    e3 = call(x, y, "seam_aware", p=1, k=4, gamma=0.5, beta=1.0)
    e4r = call(x, y, "seam_random", **{**CFG, "seam_radius": 0})
    for i in range(4):
        assert torch.equal(e3[i], e4r[i])
    assert torch.equal(e3[4]["mask"], e4r[4]["mask"])


def test_same_rectangle_and_feasibility_as_e4():
    x, y = smooth_batch(64, seed_=1)
    e4 = call(x, y, "seam_dp", **CFG)[4]
    e4r = call(x, y, "seam_random", **CFG)[4]
    assert torch.equal(e4["boxes"], e4r["boxes"])
    assert torch.equal(e4["permutation"], e4r["permutation"])
    assert torch.allclose(e4["seam_cost_base"], e4r["seam_cost_base"])
    # Statuses other than the area check depend only on the shared constraints.
    not_area = (e4["dp_status"] != STATUS_AREA) & (e4r["dp_status"] != STATUS_AREA)
    assert torch.equal(e4["dp_status"][not_area], e4r["dp_status"][not_area])
    assert torch.equal(e4["area_base"], e4r["area_base"])


def test_paths_differ_from_e4_and_ignore_content():
    x, y = smooth_batch(64, seed_=2)
    e4 = call(x, y, "seam_dp", **CFG)[4]
    e4r = call(x, y, "seam_random", **CFG)[4]
    assert not torch.equal(e4["mask"], e4r["mask"])
    # Content independence: the random boundary for fixed boxes and noise
    # does not change when the images change.
    boxes = e4r["boxes"]
    g1, g2 = torch.Generator().manual_seed(5), torch.Generator().manual_seed(5)
    a = random_boundaries(x, x.flip(0), boxes, 2, 0.05, 2, 0.2, generator=g1)
    b = random_boundaries(torch.rand_like(x), torch.rand_like(x), boxes, 2, 0.05, 2, 0.2,
                          generator=g2)
    assert torch.equal(a["mask"], b["mask"])


def test_geometry_constraints_hold():
    x, y = smooth_batch(64, seed_=3)
    _, _, _, _, m = call(x, y, "seam_random", **CFG)
    boxes, lo, ro = m["boxes"], m["left_offsets"], m["right_offsets"]
    acc = m["dp_status"] == STATUS_ACCEPTED
    diff = (m["area_final"] - m["area_base"]).abs().double()
    assert (diff[acc] <= 0.2 * m["area_base"][acc].double() + 1e-9).all()
    for i in range(64):
        k = (boxes[i, 2] - boxes[i, 0]).item()
        if k == 0:
            continue
        a, b = lo[i, :k], ro[i, :k]
        assert ((boxes[i, 3] - boxes[i, 1]) + b - a >= 2).all() or not acc[i]
        assert (a.abs() <= 2).all() and (b.abs() <= 2).all()
        if k > 1:
            assert (a[1:] - a[:-1]).abs().max() <= 1 and (b[1:] - b[:-1]).abs().max() <= 1
    rows = torch.arange(32)[None, :, None]
    outside = (rows < boxes[:, 0, None, None]) | (rows >= boxes[:, 2, None, None])
    assert not (m["mask"][:, 0] & outside).any()
    rect = box_masks(boxes[:, None], 32, 32)
    assert torch.equal(m["mask"][~acc], rect[~acc])


def test_reproducible_and_no_global_rng_consumption():
    x, y = smooth_batch()
    a = call(x, y, "seam_random", **CFG)
    after_a = torch.rand(4)
    b = call(x, y, "seam_random", **CFG)
    assert torch.equal(a[0], b[0]) and torch.equal(a[4]["mask"], b[4]["mask"])
    call(x, y, "seam_aware", p=1, k=4, gamma=0.5, beta=1.0)
    after_e3 = torch.rand(4)
    assert torch.equal(after_a, after_e3)           # same global RNG position as E3
    c = call(x, y, "seam_random", s=7, **CFG)
    assert not torch.equal(a[4]["mask"], c[4]["mask"])


def test_output_contract_and_full_seam():
    x, y = smooth_batch(16, seed_=4)
    before = x.clone()
    mixed, ya, yb, lam, m = call(x, y, "seam_random", **CFG)
    perm = m["permutation"]
    assert torch.equal(x, before)
    assert torch.equal(mixed, torch.where(m["mask"], before[perm], before))
    assert torch.equal(ya, y) and torch.equal(yb, y[perm])
    assert torch.equal(lam, 1 - m["mask"].flatten(1).float().mean(1))
    assert m["policy"] == "seam_random"
    d_hat, _, count = normalized_seam_cost(before[perm], before, m["mask"])
    assert torch.allclose(m["seam_cost"], d_hat) and torch.equal(m["seam_count"], count)
    assert (m["bends"] >= 0).all()


def test_p0_and_invalid_parameters():
    x, y = smooth_batch(4)
    mixed, _, _, lam, m = call(x, y, "seam_random", **{**CFG, "p": 0})
    assert torch.equal(mixed, x) and lam.eq(1).all()
    assert (m["dp_status"] == STATUS_NO_PATCH).all()
    for bad in ({"seam_radius": -1}, {"seam_mu": -1.0}, {"seam_min_width": 0},
                {"seam_area_budget": -1.0}):
        with pytest.raises(ValueError):
            call(x, y, "seam_random", **{**CFG, **bad})
    with pytest.raises(ValueError):
        call(x, y, "no_such_policy")


def test_bends_statistic():
    from augmentations.seam_dp import finalize_boundaries
    boxes = torch.tensor([[0, 4, 4, 10], [2, 2, 3, 8]])
    lo = torch.tensor([[0, 1, 1, 0, 0, 0], [0, 0, 0, 0, 0, 0]])
    ro = torch.tensor([[0, 0, -1, -1, 0, 0], [1, 1, 0, 0, 0, 0]])
    out = finalize_boundaries(boxes, lo, ro, torch.tensor([True, True]), 8, 16, 1, 1.0)
    # Box 0 (4 rows): left 0,1,1,0 -> 2; right 0,0,-1,-1 -> 1. Box 1 has 1 row -> 0.
    assert out["bends"].tolist() == [3, 0]


def test_one_real_training_step_through_train_py():
    train = pytest.importorskip("train")
    from models.resnet import build_resnet18
    x, y = smooth_batch(8, seed_=5)
    model = build_resnet18(num_classes=100)
    opt = torch.optim.SGD(model.parameters(), lr=0.01)
    aug_cfg = {"policy": "seam_random", "alpha": 1.0, "p": 1.0, "k": 4, "gamma": 0.5,
               "beta": 1.0, "seam_radius": 2, "seam_mu": 0.05,
               "seam_min_width": 2, "seam_area_budget": 0.2}
    seed()
    stats = train.train_one_epoch(model, [(x, y)], opt, torch.device("cpu"), aug_cfg)
    assert np.isfinite(stats["loss"])


# ---------- E4-R's own bend penalty (seam_random_mu) ----------

def test_seam_random_mu_none_means_seam_mu():
    x, y = smooth_batch(32, seed_=6)
    a = call(x, y, "seam_random", **CFG)
    b = call(x, y, "seam_random", **CFG, seam_random_mu=CFG["seam_mu"])
    assert torch.equal(a[0], b[0]) and torch.equal(a[4]["mask"], b[4]["mask"])
    assert a[4]["bend_mu"] == CFG["seam_mu"] and a[4]["seam_mu"] == CFG["seam_mu"]


def test_seam_random_mu_is_used_and_recorded():
    x, y = smooth_batch(128, seed_=7)
    lo = call(x, y, "seam_random", **CFG, seam_random_mu=0.0)[4]
    hi = call(x, y, "seam_random", **CFG, seam_random_mu=0.6)[4]
    assert hi["bend_mu"] == 0.6 and hi["seam_mu"] == CFG["seam_mu"]
    assert torch.equal(lo["boxes"], hi["boxes"])                # same E3 rectangle
    assert hi["bends"].float().mean() < lo["bends"].float().mean()


def test_seam_random_mu_does_not_affect_e4():
    x, y = smooth_batch(16, seed_=8)
    a = call(x, y, "seam_dp", **CFG)
    b = call(x, y, "seam_dp", **CFG, seam_random_mu=0.9)
    assert torch.equal(a[0], b[0]) and a[4]["bend_mu"] == CFG["seam_mu"]


def test_seam_random_mu_rejects_negative():
    x, y = smooth_batch(4)
    with pytest.raises(ValueError):
        call(x, y, "seam_random", **CFG, seam_random_mu=-0.1)


def test_train_py_passes_seam_random_mu():
    train = pytest.importorskip("train")
    x, y = smooth_batch(8, seed_=9)
    cfg = {"policy": "seam_random", "alpha": 1.0, "p": 1.0, "k": 4, "gamma": 0.5,
           "beta": 1.0, "seam_radius": 2, "seam_mu": 0.05, "seam_min_width": 2,
           "seam_area_budget": 0.2, "seam_random_mu": 0.3}
    seed()
    meta = train.apply_augmentation(x, y, cfg)[4]
    assert meta["bend_mu"] == 0.3


def test_calibration_script_smoke(tmp_path):
    import json
    import subprocess
    root = pathlib.Path(__file__).resolve().parents[1]
    out = tmp_path / "cal.json"
    res = subprocess.run([sys.executable, str(root / "scripts" / "calibrate_seam_random_mu.py"),
                          "--synthetic", "--batches", "1", "--batch-size", "32",
                          "--grid", "0,0.3", "--out", str(out)],
                         capture_output=True, text=True, cwd=root, timeout=120)
    assert res.returncode == 0, res.stderr
    data = json.loads(out.read_text())
    assert data["chosen_seam_random_mu"] in (0.0, 0.3)
    assert data["e4r"]["0.3000"]["bends"] <= data["e4r"]["0.0000"]["bends"]
