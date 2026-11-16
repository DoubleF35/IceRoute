"""Tests for the RIO error metric.

The one that carries the safety argument is
``test_unsafe_and_overcautious_are_not_averaged_together``: an error that makes
ice look safer than it is must never be netted off against an error in the
cautious direction.
"""

from __future__ import annotations

import numpy as np
import pytest

from iceroute.data.loader import load_config
from iceroute.model.metrics import IGNORE_INDEX
from iceroute.model.rio_error import (
    evaluate_all_classes,
    free_confusions,
    rio_error,
)
from iceroute.routing.polaris import PolarisTable

CFG = load_config("configs/data.yaml")
MAPPING = {int(k): v for k, v in CFG["sod_to_polaris"].items()}
TABLE = PolarisTable.load()
CLASSES = ["PC6", "PC7", "IA"]

OPEN_WATER, NEW_ICE, YOUNG_ICE, THIN_FYI, THICK_FYI, OLD_ICE = range(6)


def _field(value: int, shape=(32, 32)):
    return np.full(shape, value, dtype="uint8")


def _sic(tenths: int, shape=(32, 32)):
    return np.full(shape, tenths, dtype="uint8")


# ---------------------------------------------------------------------------
# The premise: some confusions are free, others are not
# ---------------------------------------------------------------------------


def test_new_ice_and_young_ice_are_operationally_identical():
    """The fact that motivates this whole module.

    If it ever stops being true — a different POLARIS table, a different SOD
    mapping — the argument "that confusion is harmless" stops being true with
    it, and this test is what says so.
    """
    pairs = free_confusions(MAPPING, CLASSES, table=TABLE)
    assert (NEW_ICE, YOUNG_ICE) in pairs


def test_old_ice_and_thick_fyi_are_not_free():
    assert (THICK_FYI, OLD_ICE) not in free_confusions(MAPPING, CLASSES, table=TABLE)


def test_a_free_confusion_costs_nothing_in_rio():
    truth, pred = _field(NEW_ICE), _field(YOUNG_ICE)
    err = rio_error(truth, pred, "PC6", MAPPING, table=TABLE, sic=_sic(10))
    assert err.rio_mae == pytest.approx(0.0)
    assert err.unsafe_fraction == 0.0
    assert err.exact_level_agreement == 1.0


def test_a_costly_confusion_shows_up():
    """Old ice called Thick FYI: -3 becomes -1, six points of RIO at full
    concentration."""
    err = rio_error(_field(OLD_ICE), _field(THICK_FYI), "PC6", MAPPING, table=TABLE, sic=_sic(10))
    assert err.rio_mae > 10
    assert err.rio_bias > 0  # predicted safer than reality


# ---------------------------------------------------------------------------
# Perfect and degenerate cases
# ---------------------------------------------------------------------------


def test_perfect_prediction_has_zero_error():
    truth = np.array([[OPEN_WATER, OLD_ICE], [THICK_FYI, THIN_FYI]], dtype="uint8")
    err = rio_error(truth, truth.copy(), "PC6", MAPPING, table=TABLE, sic=_sic(8, (2, 2)))
    assert err.rio_mae == 0.0
    assert err.rio_bias == 0.0
    assert err.unsafe_fraction == 0.0
    assert err.overcautious_fraction == 0.0
    assert err.exact_level_agreement == 1.0
    assert "No pixel" in err.notes[0]


def test_ignored_pixels_are_excluded():
    truth = _field(OLD_ICE)
    truth[:16] = IGNORE_INDEX
    pred = _field(THICK_FYI)
    err = rio_error(truth, pred, "PC6", MAPPING, table=TABLE, sic=_sic(10))
    assert err.n_pixels == 16 * 32


def test_all_ignored_raises_rather_than_returning_zero():
    """A metric of nothing must not read as a perfect score."""
    truth = _field(IGNORE_INDEX)
    with pytest.raises(ValueError, match="No annotated pixel"):
        rio_error(truth, _field(OPEN_WATER), "PC6", MAPPING, table=TABLE, sic=_sic(10))


def test_mask_valid_further_restricts_the_pixels():
    truth, pred = _field(OLD_ICE), _field(THICK_FYI)
    mask = np.zeros((32, 32), dtype=bool)
    mask[:8] = True
    err = rio_error(truth, pred, "PC6", MAPPING, table=TABLE, sic=_sic(10), mask_valid=mask)
    assert err.n_pixels == 8 * 32


# ---------------------------------------------------------------------------
# The asymmetry — the safety case
# ---------------------------------------------------------------------------


def test_unsafe_and_overcautious_are_not_averaged_together():
    """Half the field predicted safer, half more dangerous.

    A single symmetric error metric would report these as cancelling. They do
    not cancel: one costs time, the other costs a ship.
    """
    truth = np.empty((32, 32), dtype="uint8")
    truth[:16] = OLD_ICE  # genuinely dangerous
    truth[16:] = OPEN_WATER  # genuinely safe

    pred = np.empty((32, 32), dtype="uint8")
    pred[:16] = OPEN_WATER  # UNSAFE: danger called safe
    pred[16:] = OLD_ICE  # cautious: safe called dangerous

    err = rio_error(truth, pred, "IA", MAPPING, table=TABLE, sic=_sic(10))
    assert err.unsafe_fraction > 0
    assert err.overcautious_fraction > 0
    # The bias nearly cancels; the two fractions must not.
    assert abs(err.rio_bias) < err.rio_mae
    assert err.unsafe_fraction + err.overcautious_fraction > 0.9


def test_predicting_danger_where_there_is_none_is_not_unsafe():
    err = rio_error(_field(OPEN_WATER), _field(OLD_ICE), "IA", MAPPING, table=TABLE, sic=_sic(10))
    assert err.unsafe_fraction == 0.0
    assert err.overcautious_fraction > 0.0
    assert err.rio_bias < 0


def test_predicting_safety_where_there_is_danger_is_unsafe():
    err = rio_error(_field(OLD_ICE), _field(OPEN_WATER), "IA", MAPPING, table=TABLE, sic=_sic(10))
    assert err.unsafe_fraction > 0.0
    assert err.overcautious_fraction == 0.0
    assert err.worst_unsafe_delta > 0
    assert "overestimates" in err.notes[0]


def test_level_confusion_rows_sum_to_the_pixel_count():
    truth = np.tile(np.array([OPEN_WATER, OLD_ICE], dtype="uint8"), (32, 16))
    pred = np.tile(np.array([THICK_FYI, THIN_FYI], dtype="uint8"), (32, 16))
    err = rio_error(truth, pred, "PC6", MAPPING, table=TABLE, sic=_sic(10))
    assert err.level_confusion.sum() == err.n_pixels


def test_low_concentration_dilutes_the_error():
    """RIO weights by concentration, so the same class error at 2/10 matters
    less than at 10/10. A metric that ignored SIC would overstate thin ice."""
    strong = rio_error(
        _field(OLD_ICE), _field(OPEN_WATER), "PC6", MAPPING, table=TABLE, sic=_sic(10)
    )
    weak = rio_error(_field(OLD_ICE), _field(OPEN_WATER), "PC6", MAPPING, table=TABLE, sic=_sic(2))
    assert weak.rio_mae < strong.rio_mae


# ---------------------------------------------------------------------------
# Across vessel classes
# ---------------------------------------------------------------------------


def test_evaluate_all_classes_covers_every_vessel():
    truth = np.tile(np.array([OPEN_WATER, OLD_ICE], dtype="uint8"), (32, 16))
    pred = np.tile(np.array([OPEN_WATER, THICK_FYI], dtype="uint8"), (32, 16))
    out = evaluate_all_classes(truth, pred, CLASSES, MAPPING, table=TABLE, sic=_sic(9))

    assert set(out["per_ice_class"]) == set(CLASSES)
    assert out["worst_unsafe_fraction"] is not None
    assert [NEW_ICE, YOUNG_ICE] in out["operationally_free_confusions"]


def test_the_weakest_vessel_is_never_the_most_forgiving():
    """A 1A crosses an operational boundary where a PC6 does not, so its unsafe
    fraction cannot be the lowest for the same misclassification."""
    truth = _field(THICK_FYI)
    pred = _field(THIN_FYI)
    out = evaluate_all_classes(truth, pred, CLASSES, MAPPING, table=TABLE, sic=_sic(10))
    unsafe = {k: v["unsafe_fraction"] for k, v in out["per_ice_class"].items()}
    assert unsafe["IA"] >= min(unsafe.values())


def test_summary_names_both_directions():
    err = rio_error(_field(OLD_ICE), _field(THICK_FYI), "PC6", MAPPING, table=TABLE, sic=_sic(10))
    text = err.summary()
    assert "PERMISSIVE" in text
    assert "cautious" in text
