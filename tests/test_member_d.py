"""Regression coverage for padding provenance and official-test eligibility."""
import pytest
import torch

from analysis.boundary_audit import seam_totals, summarize
from evaluate import verify_checkpoint_log


def test_real_black_pixel_is_distinct_from_crop_padding():
    images = torch.tensor([[[[0.0, 0.0, 1.0]]]]).expand(1, 3, 1, 3)
    mask = torch.tensor([[[False, True, False]]])
    padded = torch.tensor([[[True, False, False]]])
    sums, counts = seam_totals(images, mask, padded)
    assert counts.tolist() == [[2, 2, 1, 1]]
    assert sums.tolist() == [[1.0, 1.0, 1.0, 1.0]]


def test_no_boundary_is_missing_instead_of_zero():
    row = {"area_ratio": 0}
    for key in ("all", "without_double_padding", "without_any_padding", "without_double_black"):
        row[f"count_{key}"] = 0
        row[f"sum_{key}"] = 0
    stats = summarize([row])
    assert stats["boundary_length"] is None
    assert stats["seam"]["all"]["pair_weighted_rgb"] is None
    assert stats["seam"]["all"]["na_images"] == 1


def test_official_test_rejects_partial_runs_and_later_ties(tmp_path):
    log = tmp_path / "run.csv"
    log.write_text("epoch,val_accuracy,train_samples\n1,0.8,45000\n2,0.8,45000\n")
    state = dict(epoch=1, best_val_accuracy=0.8, config=dict(training=dict(epochs=2)))
    verify_checkpoint_log(state, log, require_formal=True)
    with pytest.raises(ValueError, match="earliest best"):
        verify_checkpoint_log(dict(state, epoch=2), log, require_formal=True)
    log.write_text("epoch,val_accuracy,train_samples\n1,0.8,256\n")
    with pytest.raises(ValueError, match="complete formal"):
        verify_checkpoint_log(state, log, require_formal=True)
