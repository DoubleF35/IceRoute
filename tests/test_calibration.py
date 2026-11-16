"""Calibration maths on synthetic data with a known answer."""

from __future__ import annotations

import numpy as np
import pytest

from iceroute.model.calibration import (
    IGNORE,
    apply_temperature,
    brier,
    fit_temperature,
    nll,
    reliability,
    sample_pixels,
    summarise,
    top_class_reliability,
)


def _softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def _overconfident(n=60_000, c=3, t_true=2.5, seed=0):
    """Labels drawn from softmax(z / t_true), model reports softmax(z).

    The model is overconfident by exactly t_true, so temperature scaling
    should recover it.
    """
    rng = np.random.default_rng(seed)
    z = rng.normal(0.0, 3.0, size=(n, c))
    p_real = _softmax(z / t_true)
    labels = (rng.random(n)[:, None] > np.cumsum(p_real, axis=1)).sum(axis=1)
    return _softmax(z), labels


def test_temperature_one_is_identity():
    p = _softmax(np.random.default_rng(1).normal(size=(100, 4)))
    np.testing.assert_allclose(apply_temperature(p, 1.0), p, atol=1e-6)


def test_temperature_never_changes_the_argmax():
    p = _softmax(np.random.default_rng(2).normal(size=(500, 3)) * 4)
    for t in (0.3, 0.9, 1.7, 6.0):
        assert (apply_temperature(p, t).argmax(1) == p.argmax(1)).all()


def test_temperature_above_one_softens():
    p = _softmax(np.array([[4.0, 0.0, 0.0]]))
    assert apply_temperature(p, 3.0).max() < p.max()


def test_temperature_must_be_positive():
    with pytest.raises(ValueError):
        apply_temperature(np.full((1, 2), 0.5), 0.0)


def test_fit_recovers_the_true_temperature():
    p, y = _overconfident(t_true=2.5)
    assert fit_temperature(p, y) == pytest.approx(2.5, rel=0.08)


def test_calibrated_model_needs_no_temperature():
    p, y = _overconfident(t_true=1.0, seed=3)
    assert fit_temperature(p, y) == pytest.approx(1.0, rel=0.08)


def test_scaling_lowers_ece_and_nll_on_an_overconfident_model():
    p, y = _overconfident(t_true=2.5, seed=4)
    t = fit_temperature(p, y)
    q = apply_temperature(p, t)
    assert top_class_reliability(q, y).ece < 0.5 * top_class_reliability(p, y).ece
    assert nll(q, y) < nll(p, y)
    assert brier(q, y) < brier(p, y)


def test_ece_is_zero_when_confidence_matches_frequency():
    conf = np.repeat([0.25, 0.75], 4)
    correct = np.array([1, 0, 0, 0, 1, 1, 1, 0])
    rel = reliability(conf, correct, n_bins=4)
    assert rel.ece == pytest.approx(0.0)
    assert rel.count.sum() == 8


def test_ece_of_a_fully_overconfident_model():
    rel = reliability(np.full(10, 0.99), np.zeros(10), n_bins=10)
    assert rel.ece == pytest.approx(0.99)
    assert rel.mce == pytest.approx(0.99)


def test_confidence_zero_and_one_land_in_the_edge_bins():
    rel = reliability(np.array([0.0, 1.0]), np.array([0, 1]), n_bins=5)
    assert rel.count[0] == 1 and rel.count[-1] == 1


def test_sample_pixels_skips_ignored_and_respects_n():
    proba = np.random.default_rng(5).random((3, 20, 20)).astype("float32")
    truth = np.full((20, 20), IGNORE, dtype="uint8")
    truth[:5, :5] = 1
    p, y = sample_pixels(proba, truth, 10, np.random.default_rng(0))
    assert p.shape == (10, 3) and (y == 1).all()
    p, y = sample_pixels(proba, truth, 1000, np.random.default_rng(0))
    assert len(y) == 25


def test_summary_counts_hopeless_errors():
    # two wrong pixels, one of which leaves 0.5% to the truth
    p = np.array([[0.995, 0.005, 0.0], [0.6, 0.4, 0.0], [0.1, 0.1, 0.8]])
    y = np.array([1, 1, 2])
    s = summarise(p, y, ["a", "b", "c"])
    assert s["accuracy"] == pytest.approx(1 / 3)
    assert s["errors_true_class_below_1pct"] == pytest.approx(0.5)
    assert set(s["classwise_ece"]) == {"a", "b", "c"}
