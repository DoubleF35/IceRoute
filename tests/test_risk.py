"""Chance-constrained decision rule: only ever more cautious than argmax."""

from __future__ import annotations

import numpy as np
import pytest

from iceroute.model.risk import _confusion, chance_constrained_labels

# class 2 is the most dangerous (lowest RIV), class 0 the safest
RIV = np.array([3.0, 1.0, -1.0])


def _px(*probs):
    """(n_classes, 1, n) from a list of per-pixel probability vectors."""
    return np.array(probs, dtype="float32").T[:, None, :]


def test_a_budget_of_one_is_argmax():
    p = _px([0.6, 0.3, 0.1], [0.2, 0.2, 0.6], [0.1, 0.8, 0.1])
    out = chance_constrained_labels(p, RIV, 1.0)
    np.testing.assert_array_equal(out, p.argmax(0))


def test_a_plausible_dangerous_class_wins():
    p = _px([0.85, 0.05, 0.10])
    assert chance_constrained_labels(p, RIV, 0.10)[0, 0] == 2
    assert chance_constrained_labels(p, RIV, 0.11)[0, 0] == 0


def test_never_less_cautious_than_argmax():
    rng = np.random.default_rng(0)
    p = rng.dirichlet([0.5, 0.5, 0.5], size=(40, 40)).transpose(2, 0, 1).astype("float32")
    am = p.argmax(0)
    for alpha in (0.9, 0.5, 0.2, 0.05):
        out = chance_constrained_labels(p, RIV, alpha)
        assert (RIV[out] <= RIV[am]).all(), alpha


def test_smaller_budget_is_monotonically_more_cautious():
    rng = np.random.default_rng(1)
    p = rng.dirichlet([1, 1, 1], size=(30, 30)).transpose(2, 0, 1).astype("float32")
    prev = chance_constrained_labels(p, RIV, 0.9)
    for alpha in (0.5, 0.3, 0.1, 0.02):
        cur = chance_constrained_labels(p, RIV, alpha)
        assert (RIV[cur] <= RIV[prev]).all()
        prev = cur


@pytest.mark.parametrize("alpha", [0.0, -0.1, 1.5])
def test_alpha_out_of_range(alpha):
    with pytest.raises(ValueError):
        chance_constrained_labels(_px([1, 0, 0]), RIV, alpha)


def test_shape_checks():
    with pytest.raises(ValueError):
        chance_constrained_labels(np.ones((3, 4)), RIV, 0.5)
    with pytest.raises(ValueError):
        chance_constrained_labels(_px([1, 0, 0]), RIV[:2], 0.5)


def test_confusion_ignores_unlabelled_pixels():
    truth = np.array([[0, 1, 255], [2, 2, 1]])
    pred = np.array([[0, 2, 0], [2, 1, 1]])
    cm = _confusion(truth, pred, 3, 255)
    assert cm.sum() == 5
    assert cm[1, 2] == 1 and cm[2, 2] == 1 and cm[1, 1] == 1
