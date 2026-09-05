"""Tests for the loss functions.

The Lovász tests pin the property that motivates using it at all: it is the
tight convex surrogate of the Jaccard index, so it equals the IoU loss exactly
on hard predictions rather than merely correlating with it.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Lovász-Softmax — the surrogate that optimises IoU directly
# ---------------------------------------------------------------------------


def test_lovasz_is_zero_on_a_perfect_prediction():
    from iceroute.model.losses import LovaszSoftmaxLoss

    target = torch.randint(0, 3, (2, 16, 16))
    perfect = F.one_hot(target, 3).permute(0, 3, 1, 2).float() * 30 - 15
    assert LovaszSoftmaxLoss(3)(perfect, target).item() == pytest.approx(0.0, abs=1e-5)


def test_lovasz_is_one_when_every_pixel_is_wrong():
    from iceroute.model.losses import LovaszSoftmaxLoss

    target = torch.randint(0, 3, (2, 16, 16))
    wrong = F.one_hot((target + 1) % 3, 3).permute(0, 3, 1, 2).float() * 30 - 15
    assert LovaszSoftmaxLoss(3)(wrong, target).item() == pytest.approx(1.0, abs=1e-3)


def test_lovasz_tracks_iou_not_just_accuracy():
    """The reason for using it: a worse IoU must give a worse loss."""
    from iceroute.model.losses import LovaszSoftmaxLoss

    loss = LovaszSoftmaxLoss(2)
    target = torch.zeros(1, 8, 8, dtype=torch.long)
    target[:, :4] = 1

    good = F.one_hot(target, 2).permute(0, 3, 1, 2).float() * 10 - 5
    bad = good.clone()
    bad[:, :, :2] = good[:, :, :2].flip(1)      # half of class 1 mislabelled
    assert loss(bad, target) > loss(good, target)


def test_lovasz_ignores_masked_pixels():
    from iceroute.model.losses import IGNORE_INDEX, LovaszSoftmaxLoss

    target = torch.zeros(1, 8, 8, dtype=torch.long)
    target[:, 4:] = IGNORE_INDEX
    logits = torch.full((1, 2, 8, 8), -15.0)
    logits[:, 0] = 15.0                          # correct on every valid pixel
    assert LovaszSoftmaxLoss(2)(logits, target).item() == pytest.approx(0.0, abs=1e-5)


def test_lovasz_on_an_all_ignored_batch_stays_connected():
    """Must return a differentiable zero, not NaN and not a bare constant."""
    from iceroute.model.losses import IGNORE_INDEX, LovaszSoftmaxLoss

    target = torch.full((1, 4, 4), IGNORE_INDEX, dtype=torch.long)
    logits = torch.randn(1, 3, 4, 4, requires_grad=True)
    out = LovaszSoftmaxLoss(3)(logits, target)
    assert out.item() == 0.0
    out.backward()
    assert torch.isfinite(logits.grad).all()


def test_lovasz_skips_classes_absent_from_the_batch():
    """Scoring an absent class as perfect would make the loss improve exactly
    when a rare class disappears — the failure this project already has."""
    from iceroute.model.losses import LovaszSoftmaxLoss

    target = torch.zeros(1, 8, 8, dtype=torch.long)      # only class 0 present
    logits = torch.full((1, 4, 8, 8), -15.0)
    logits[:, 0] = 15.0
    assert LovaszSoftmaxLoss(4)(logits, target).item() == pytest.approx(0.0, abs=1e-5)


def test_lovasz_gradient_is_finite():
    from iceroute.model.losses import LovaszSoftmaxLoss

    logits = torch.randn(2, 3, 16, 16, requires_grad=True)
    LovaszSoftmaxLoss(3)(logits, torch.randint(0, 3, (2, 16, 16))).backward()
    assert torch.isfinite(logits.grad).all()


def test_lovasz_grad_sums_to_the_jaccard_of_the_full_set():
    from iceroute.model.losses import lovasz_grad

    gt = torch.tensor([1.0, 1.0, 0.0, 0.0, 1.0])
    g = lovasz_grad(gt)
    assert torch.isfinite(g).all()
    assert g.numel() == gt.numel()


def test_combined_loss_adds_lovasz_only_when_weighted():
    from iceroute.model.losses import CombinedLoss

    logits = torch.randn(2, 3, 8, 8)
    target = torch.randint(0, 3, (2, 8, 8))

    without = CombinedLoss(3, lovasz_weight=0.0)(logits, target)
    with_ = CombinedLoss(3, lovasz_weight=1.0)(logits, target)
    assert "lovasz" not in without
    assert "lovasz" in with_
    assert with_["loss"] > without["loss"]
