"""Tests for metrics and loss.

These numbers end up in every public claim the project makes, so the arithmetic
is checked against cases computed by hand rather than against the
implementation's own output.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from iceroute.model.losses import (
    IGNORE_INDEX,
    CombinedLoss,
    DiceLoss,
    inverse_frequency_weights,
)
from iceroute.model.metrics import ConfusionMatrix

# ---------------------------------------------------------------------------
# Confusion matrix
# ---------------------------------------------------------------------------


def test_perfect_prediction_scores_one():
    cm = ConfusionMatrix(3)
    y = np.array([[0, 1], [2, 2]])
    cm.update(y, y)
    assert cm.miou() == pytest.approx(1.0)
    assert cm.accuracy() == pytest.approx(1.0)


def test_iou_matches_hand_computation():
    """Two classes, one confusion, worked out by hand.

    truth      pred
    0 0        0 1
    1 1        1 1

    class 0: TP=1, FP=0, FN=1  -> IoU 1/2
    class 1: TP=2, FP=1, FN=0  -> IoU 2/3
    mIoU = (0.5 + 0.6667) / 2 = 0.58333
    """
    cm = ConfusionMatrix(2)
    cm.update(np.array([[0, 0], [1, 1]]), np.array([[0, 1], [1, 1]]))
    iou = cm.iou_per_class()
    assert iou[0] == pytest.approx(0.5)
    assert iou[1] == pytest.approx(2 / 3)
    assert cm.miou() == pytest.approx(0.5833333, abs=1e-6)


def test_ignore_index_excluded_entirely():
    cm = ConfusionMatrix(2)
    y = np.array([0, 1, IGNORE_INDEX, IGNORE_INDEX])
    p = np.array([0, 1, 1, 0])
    cm.update(y, p)
    assert cm.matrix.sum() == 2, "ignored pixels must not enter the matrix"
    assert cm.miou() == pytest.approx(1.0)


def test_absent_class_excluded_from_miou():
    """A class with no ground truth must not drag the average.

    Class 2 never appears in the truth. Scoring it 0 would penalise the model
    for the split's composition.
    """
    cm = ConfusionMatrix(3)
    cm.update(np.array([0, 0, 1, 1]), np.array([0, 0, 1, 1]))
    assert cm.n_classes_scored() == 2
    assert cm.miou() == pytest.approx(1.0)


def test_hallucinated_class_still_penalised():
    """Predicting an absent class must lower the score via the true class."""
    clean = ConfusionMatrix(3)
    clean.update(np.array([0, 0, 1, 1]), np.array([0, 0, 1, 1]))

    halluc = ConfusionMatrix(3)
    halluc.update(np.array([0, 0, 1, 1]), np.array([0, 2, 1, 1]))

    assert halluc.miou() < clean.miou()
    assert halluc.n_classes_predicted_absent() == 1


def test_binary_accuracy_collapses_ice_classes():
    """Confusing two ice types must not hurt the ice/water score."""
    cm = ConfusionMatrix(3)  # 0 = water, 1 and 2 = ice
    cm.update(np.array([1, 1, 2, 2]), np.array([2, 2, 1, 1]))
    assert cm.binary_accuracy() == pytest.approx(1.0)
    assert cm.miou() == pytest.approx(0.0)  # every per-class IoU is 0


def test_binary_accuracy_counts_water_confusion():
    cm = ConfusionMatrix(3)
    cm.update(np.array([0, 0, 1, 1]), np.array([0, 1, 1, 1]))
    assert cm.binary_accuracy() == pytest.approx(0.75)


def test_update_accepts_torch_tensors():
    cm = ConfusionMatrix(2)
    cm.update(torch.tensor([0, 1]), torch.tensor([0, 1]))
    assert cm.matrix.sum() == 2


def test_matrix_orientation_is_truth_rows():
    """Row = truth, column = prediction. Getting this backwards silently
    transposes every published confusion matrix."""
    cm = ConfusionMatrix(2)
    cm.update(np.array([0]), np.array([1]))  # truth 0, predicted 1
    assert cm.matrix[0, 1] == 1
    assert cm.matrix[1, 0] == 0


def test_empty_matrix_does_not_crash():
    cm = ConfusionMatrix(3)
    assert np.isnan(cm.miou())
    assert np.isnan(cm.accuracy())
    cm.format_table()  # must not raise


# ---------------------------------------------------------------------------
# Class weights
# ---------------------------------------------------------------------------


def test_rare_class_gets_more_weight():
    w = inverse_frequency_weights({0: 0.9, 1: 0.1}, 2, normalize=False)
    assert w[1] > w[0]
    assert w[0] == pytest.approx(1 / 0.9)


def test_weights_are_clipped():
    """Without a ceiling a 0.01% class would earn a weight of 10,000."""
    w = inverse_frequency_weights({0: 0.9999, 1: 0.0001}, 2, clip_max=10.0, normalize=False)
    assert w[1] == pytest.approx(10.0)


def test_absent_class_gets_max_weight_not_infinity():
    w = inverse_frequency_weights({0: 1.0}, 2, clip_max=10.0, normalize=False)
    assert np.isfinite(w[1])
    assert w[1] == pytest.approx(10.0)


def test_normalisation_preserves_mean():
    w = inverse_frequency_weights({0: 0.5, 1: 0.3, 2: 0.2}, 3, normalize=True)
    assert float(w.sum()) == pytest.approx(3.0)


# ---------------------------------------------------------------------------
# Losses
# ---------------------------------------------------------------------------


def test_dice_is_near_zero_for_a_confident_correct_prediction():
    logits = torch.full((1, 2, 4, 4), -10.0)
    logits[:, 1] = 10.0
    target = torch.ones(1, 4, 4, dtype=torch.long)
    assert float(DiceLoss(2)(logits, target)) < 0.05


def test_dice_ignores_masked_pixels():
    """Two batches differing only in the ignored region must score the same."""
    logits = torch.randn(1, 3, 4, 4, generator=torch.Generator().manual_seed(0))
    a = torch.zeros(1, 4, 4, dtype=torch.long)
    b = a.clone()
    a[:, 2:] = IGNORE_INDEX
    b[:, 2:] = IGNORE_INDEX
    loss = DiceLoss(3)
    assert float(loss(logits, a)) == pytest.approx(float(loss(logits, b)))


def test_all_ignored_batch_returns_finite_zero():
    """An entirely invalid batch must not produce NaN and kill the run."""
    logits = torch.randn(1, 3, 4, 4, requires_grad=True)
    target = torch.full((1, 4, 4), IGNORE_INDEX, dtype=torch.long)
    out = CombinedLoss(3)(logits, target)
    assert torch.isfinite(out["loss"])
    assert float(out["loss"].detach()) == pytest.approx(0.0)
    out["loss"].backward()  # graph must stay connected


def test_combined_loss_backward_produces_gradients():
    logits = torch.randn(2, 3, 8, 8, requires_grad=True)
    target = torch.randint(0, 3, (2, 8, 8))
    target[0, 0, :] = IGNORE_INDEX
    out = CombinedLoss(3, ce_weight=1.0, dice_weight=0.5)(logits, target)
    out["loss"].backward()
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()


def test_fill_value_never_reaches_one_hot():
    """255 as a class id would blow up scatter in one_hot."""
    logits = torch.randn(1, 6, 4, 4)
    target = torch.full((1, 4, 4), IGNORE_INDEX, dtype=torch.long)
    target[0, 0, 0] = 3
    out = CombinedLoss(6)(logits, target)
    assert torch.isfinite(out["loss"])


def test_class_weights_move_with_the_module():
    """A bare tensor attribute would stay on CPU and raise mid-training."""
    loss = CombinedLoss(3, class_weights=torch.ones(3))
    assert "class_weights" in dict(loss.named_buffers())


# ---------------------------------------------------------------------------
# Which model the measurement runs on
# ---------------------------------------------------------------------------


def test_the_deployed_config_comes_from_the_picker_not_a_default(tmp_path, monkeypatch):
    """The bug this guards already fired, on 2026-09-09.

    ``report/metrics.py`` defaulted to ``configs/train3.yaml`` — the *baseline*
    checkpoint — while the deployed model is whatever ``pick_model.py`` wrote into
    ``artifacts/best_config.txt`` (D-025). Both files load, both produce numbers,
    so ``make report`` silently restated the headline compression ratio as 347:1
    instead of 439:1 with no measurement code having changed. It went unnoticed
    because report had not been re-run since the model switch.
    """
    from iceroute.report import metrics as mx

    picked = tmp_path / "configs/train3_lovasz.yaml"
    picked.parent.mkdir(parents=True)
    picked.write_text("output: {best_checkpoint: lovasz.pt}\n")
    marker = tmp_path / "best_config.txt"
    marker.write_text(str(picked) + "\n")

    monkeypatch.setattr(mx, "DEPLOYED", marker)
    assert mx.deployed_config() == picked


def test_a_missing_marker_falls_back_rather_than_crashing(tmp_path, monkeypatch):
    from iceroute.report import metrics as mx

    monkeypatch.setattr(mx, "DEPLOYED", tmp_path / "missing.txt")
    assert mx.deployed_config(Path("configs/train3.yaml")) == Path("configs/train3.yaml")


def test_a_marker_pointing_at_a_deleted_config_falls_back(tmp_path, monkeypatch):
    """A dangling pointer must not take the whole report down right before it is
    needed; it must degrade to the baseline and say so through the fallback."""
    from iceroute.report import metrics as mx

    marker = tmp_path / "best_config.txt"
    marker.write_text(str(tmp_path / "deleted.yaml") + "\n")
    monkeypatch.setattr(mx, "DEPLOYED", marker)
    assert mx.deployed_config(Path("configs/train3.yaml")) == Path("configs/train3.yaml")


def test_the_real_repository_measures_on_the_deployed_model():
    """metrics.json must record which model produced the compression figure, and
    it must be the deployed one. Without the record a stale file cannot be told
    from a fresh one measured on the wrong network."""
    import json

    mpath, marker = Path("artifacts/metrics.json"), Path("artifacts/best_config.txt")
    if not (mpath.exists() and marker.exists()):
        pytest.skip("metrics.json or best_config.txt missing")
    comp = json.loads(mpath.read_text()).get("compression") or {}
    if not comp or comp.get("error"):
        pytest.skip("compression not measured")
    want = yaml.safe_load(Path(marker.read_text().strip()).read_text())
    assert comp.get("checkpoint") == want["output"]["best_checkpoint"]
