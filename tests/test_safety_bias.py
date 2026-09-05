"""Tests for the danger-biased decision rule.

The rule turned out not to help on the real model — the errors are confident,
not uncertain (D-018) — but the code stays because the *measurement* is the
value: it is what proves an inference-time fix cannot close the gap.
"""

from __future__ import annotations

import numpy as np
import pytest

from iceroute.data.loader import load_config
from iceroute.model.safety_bias import (
    TradeOff,
    choose_tau,
    danger_order,
    format_curve,
    safety_biased_labels,
    sweep_tau,
)
from iceroute.routing.polaris import PolarisTable

CFG = load_config("configs/data.yaml")
MAPPING = {int(k): v for k, v in CFG["sod_to_polaris"].items()}
TABLE = PolarisTable.load()
OPEN_WATER, NEW_ICE, YOUNG_ICE, THIN_FYI, THICK_FYI, OLD_ICE = range(6)


def _proba(pairs: list[tuple[int, float]], shape=(4, 4)) -> np.ndarray:
    """A probability field where each class has a constant probability."""
    p = np.zeros((6, *shape), dtype="float32")
    for cls, v in pairs:
        p[cls] = v
    return p / p.sum(axis=0, keepdims=True)


# ---------------------------------------------------------------------------
# The rule
# ---------------------------------------------------------------------------


def test_danger_order_ranks_old_ice_below_open_water():
    riv = danger_order("PC6", MAPPING, table=TABLE)
    assert riv[OLD_ICE] < riv[THICK_FYI] < riv[OPEN_WATER]


def test_tau_one_is_exactly_argmax():
    """The default must not quietly change behaviour."""
    rng = np.random.default_rng(0)
    p = rng.random((6, 32, 32)).astype("float32")
    p /= p.sum(axis=0, keepdims=True)
    riv = danger_order("PC6", MAPPING, table=TABLE)
    assert np.array_equal(safety_biased_labels(p, riv, 1.0), p.argmax(0).astype("uint8"))


def test_a_close_second_that_is_more_dangerous_wins():
    """0.45 open water vs 0.44 old ice: at tau=0.9 the vessel is told old ice."""
    p = _proba([(OPEN_WATER, 0.45), (OLD_ICE, 0.44), (THIN_FYI, 0.11)])
    riv = danger_order("PC6", MAPPING, table=TABLE)

    assert np.all(safety_biased_labels(p, riv, 1.0) == OPEN_WATER)
    assert np.all(safety_biased_labels(p, riv, 0.9) == OLD_ICE)


def test_a_distant_second_does_not_win():
    """The rule breaks ties; it must not overturn a confident prediction.

    This is the property that makes it useless on the real model, where the
    true class sits at 0.003 — and it is the correct behaviour regardless.
    """
    p = _proba([(OPEN_WATER, 0.97), (OLD_ICE, 0.003), (THIN_FYI, 0.027)])
    riv = danger_order("PC6", MAPPING, table=TABLE)
    assert np.all(safety_biased_labels(p, riv, 0.5) == OPEN_WATER)
    assert np.all(safety_biased_labels(p, riv, 0.3) == OPEN_WATER)


def test_lower_tau_is_never_less_conservative():
    """Monotonicity: relaxing the plausibility bar cannot make the map safer-looking."""
    rng = np.random.default_rng(3)
    p = rng.random((6, 24, 24)).astype("float32")
    p /= p.sum(axis=0, keepdims=True)
    riv = danger_order("IA", MAPPING, table=TABLE)

    prev = None
    for tau in (1.0, 0.8, 0.6, 0.4):
        mean_riv = riv[safety_biased_labels(p, riv, tau)].mean()
        if prev is not None:
            assert mean_riv <= prev + 1e-9
        prev = mean_riv


def test_rejects_a_degenerate_tau():
    riv = danger_order("PC6", MAPPING, table=TABLE)
    p = _proba([(OPEN_WATER, 1.0)])
    for bad in (0.0, -0.1, 1.5):
        with pytest.raises(ValueError, match="tau"):
            safety_biased_labels(p, riv, bad)


def test_rejects_mismatched_class_counts():
    p = _proba([(OPEN_WATER, 1.0)])
    with pytest.raises(ValueError, match="risk value"):
        safety_biased_labels(p, np.zeros(3))


def test_rejects_a_two_dimensional_input():
    with pytest.raises(ValueError, match=r"n_classes, H, W"):
        safety_biased_labels(np.zeros((4, 4)), np.zeros(4))


# ---------------------------------------------------------------------------
# The sweep and the operating point
# ---------------------------------------------------------------------------


def test_sweep_returns_a_row_per_threshold():
    p = _proba([(OPEN_WATER, 0.6), (OLD_ICE, 0.4)], shape=(16, 16))
    truth = np.full((16, 16), OLD_ICE, dtype="uint8")
    sic = np.full((16, 16), 10, dtype="uint8")
    curve = sweep_tau(p, truth, "PC6", MAPPING, taus=[1.0, 0.8, 0.5],
                      table=TABLE, sic=sic)
    assert [t.tau for t in curve] == [1.0, 0.8, 0.5]


def test_sweep_shows_the_trade_it_is_meant_to_show():
    """Caution must cost something, or the curve is not informative."""
    rng = np.random.default_rng(5)
    truth = rng.integers(0, 6, size=(48, 48)).astype("uint8")
    p = np.zeros((6, 48, 48), dtype="float32")
    for c in range(6):
        p[c] = (truth == c) * 0.5 + 0.1
    p /= p.sum(axis=0, keepdims=True)
    sic = np.full((48, 48), 8, dtype="uint8")

    curve = sweep_tau(p, truth, "PC6", MAPPING, taus=[1.0, 0.3], table=TABLE, sic=sic)
    assert curve[1].unsafe_fraction <= curve[0].unsafe_fraction
    assert curve[1].overcautious_fraction >= curve[0].overcautious_fraction


def test_choose_tau_refuses_to_relax_the_budget_silently():
    """When nothing meets both constraints it must say so, not pick anyway."""
    curve = [
        TradeOff(1.0, 0.45, 0.97, 0.264, 0.0001, 5.70, 2.0),
        TradeOff(0.3, 0.465, 0.968, 0.250, 0.0007, 5.11, 1.8),
    ]
    tau, why = choose_tau(curve, max_unsafe=0.05)
    assert "NESSUN punto" in why
    assert "non allargato in silenzio" in why
    assert tau == 0.3


def test_choose_tau_takes_the_least_conservative_feasible_point():
    curve = [
        TradeOff(1.0, 0.50, 0.97, 0.12, 0.00, 4.0, 1.0),
        TradeOff(0.8, 0.48, 0.96, 0.04, 0.01, 3.0, 0.5),
        TradeOff(0.5, 0.42, 0.95, 0.01, 0.05, 2.0, 0.2),
    ]
    tau, why = choose_tau(curve, max_unsafe=0.05)
    assert tau == 0.8
    assert "entro il budget" in why


def test_choose_tau_will_not_pick_a_useless_map():
    """A map that calls everything old ice has zero unsafe pixels and no value.

    The mIoU floor is what stops the selector doing that.
    """
    curve = [
        TradeOff(1.0, 0.50, 0.97, 0.12, 0.00, 4.0, 1.0),
        TradeOff(0.1, 0.05, 0.30, 0.00, 0.90, 9.0, -8.0),
    ]
    tau, why = choose_tau(curve, max_unsafe=0.05, min_miou_ratio=0.80)
    assert tau != 0.1, "ha scelto una mappa senza informazione pur di azzerare il rischio"
    assert tau == 1.0
    assert "non allargato in silenzio" in why


def test_choose_tau_says_so_when_no_point_is_usable_at_all():
    """If even the most permissive setting is below the floor, the decision rule
    cannot help and the message must put the blame on the model."""
    curve = [
        TradeOff(0.5, 0.10, 0.40, 0.20, 0.30, 8.0, 3.0),
        TradeOff(0.3, 0.05, 0.30, 0.00, 0.90, 9.0, -8.0),
    ]
    tau, why = choose_tau(curve, max_unsafe=0.05, min_miou_ratio=2.0)
    assert tau == 1.0
    assert "problema è del modello" in why


def test_choose_tau_on_an_empty_curve_falls_back_to_argmax():
    tau, why = choose_tau([])
    assert tau == 1.0
    assert "argmax" in why


def test_format_curve_is_readable():
    out = format_curve([TradeOff(1.0, 0.45, 0.97, 0.264, 0.0001, 5.70, 2.0)])
    assert "tau" in out
    assert "PERMISSIVA" in out
    assert "26.40%" in out
