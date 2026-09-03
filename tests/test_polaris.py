"""POLARIS tests — SPEC §8.2 requires hand-computed cases.

Every expected value below is worked out by hand from MSC.1/Circ.1519 Table 1.3
and written into the test, so the test checks the implementation against the
circular rather than against itself.
"""

from __future__ import annotations

import numpy as np
import pytest

from iceroute.routing.polaris import (
    ELEVATED_SPEED_LIMIT_KN,
    FILL_VALUE,
    Operation,
    PolarisTable,
    RioThresholds,
    classify_rio,
    compute_rio,
    is_navigable,
    speed_limit_kn,
    summarise,
)

# The mapping from configs/data.yaml: always the most conservative column of
# those a SOD class spans (DECISIONS.md D-008).
MAPPING = {
    0: "ice_free",
    1: "new_ice",
    2: "grey_white_ice",
    3: "thin_fyi_2nd_stage",
    4: "thick_fyi",
    5: "heavy_multi_year_ice",
}


@pytest.fixture(scope="module")
def table() -> PolarisTable:
    return PolarisTable.load()


# ---------------------------------------------------------------------------
# The table itself
# ---------------------------------------------------------------------------


def test_table_matches_the_circular(table):
    """Spot values transcribed from Table 1.3 by hand."""
    assert table.riv("PC6", "thick_fyi") == -1
    assert table.riv("PC7", "heavy_multi_year_ice") == -3
    assert table.riv("IA", "thin_fyi_2nd_stage") == 0
    assert table.riv("IB", "grey_white_ice") == 1
    assert table.riv("IC", "heavy_multi_year_ice") == -8
    assert table.riv("PC1", "heavy_multi_year_ice") == 1
    assert table.riv("Not Ice Strengthened", "new_ice") == 1


def test_ice_free_is_plus_three_for_everyone(table):
    """Table 1.3: the Ice-Free column is +3 for every ice class."""
    for cls in table.ice_classes:
        assert table.riv(cls, "ice_free") == 3, cls


def test_risk_never_increases_with_thicker_ice(table):
    """Along a row, RIV must be monotonically non-increasing.

    Ice types are printed in order of severity, so a row that goes up would be
    a transcription error.
    """
    for cls in table.ice_classes:
        values = [table.riv(cls, t) for t in table.ice_types]
        assert all(a >= b for a, b in zip(values, values[1:], strict=False)), (
            f"{cls}: la riga non è monotona: {values}"
        )


def test_stronger_class_is_never_worse(table):
    """Down a column, a weaker class must never score better than a stronger one."""
    order = ["PC1", "PC2", "PC3", "PC4", "PC5", "PC6", "PC7"]
    for ice_type in table.ice_types:
        vals = [table.riv(c, ice_type) for c in order]
        assert all(a >= b for a, b in zip(vals, vals[1:], strict=False)), (
            f"{ice_type}: {dict(zip(order, vals, strict=True))}"
        )


def test_every_class_has_a_source(table):
    """SPEC rule 2: an unsourced value must be flagged, not used silently."""
    assert table.provisional_classes() == []


def test_required_demo_classes_exist(table):
    """SPEC §8.2: PC6, PC7 and a reinforced non-PC class."""
    for cls in ("PC6", "PC7", "IA"):
        assert cls in table.ice_classes


# ---------------------------------------------------------------------------
# Hand-computed RIO cases
# ---------------------------------------------------------------------------


def test_all_open_water_is_maximum_rio():
    """SPEC §8.2 named case. 10/10 ice-free at RIV +3 gives RIO = 30."""
    sod = np.zeros((4, 4), dtype="uint8")
    rio = compute_rio(sod, "PC7", MAPPING)
    assert np.allclose(rio, 30.0)


def test_old_ice_full_concentration_pc7_is_no_go():
    """SPEC §8.2 named case.

    10/10 heavy multi-year ice, PC7: RIV = -3, so RIO = 10 * -3 = -30.
    That is below -10, hence special consideration: no-go.
    """
    sod = np.full((4, 4), 5, dtype="uint8")
    rio = compute_rio(sod, "PC7", MAPPING)
    assert np.allclose(rio, -30.0)
    assert not is_navigable(rio, "PC7").any()


def test_partial_concentration_includes_the_open_water_term():
    """The error this guards is dropping the ice-free remainder.

    7/10 thick first-year ice for PC6: RIV(PC6, thick_fyi) = -1.
        RIO = 7 * (-1) + 3 * (+3) = -7 + 9 = +2   -> navigable, normal
    Omitting the open-water term would give -7, which reads as elevated risk
    and would steer the route away from water a PC6 handles normally.
    """
    sod = np.full((2, 2), 4, dtype="uint8")
    sic = np.full((2, 2), 7, dtype="uint8")
    rio = compute_rio(sod, "PC6", MAPPING, sic=sic)
    assert np.allclose(rio, 2.0)
    assert is_navigable(rio, "PC6").all()


def test_same_ice_different_class_changes_the_verdict():
    """The demo moment: switching ice class must change navigability.

    5/10 thick first-year ice.
      PC6: RIV -1  ->  5*(-1) + 5*3 = +10   normal
      IB : RIV -4  ->  5*(-4) + 5*3 = -5    elevated band, but IB is below PC7
                                            so it is special consideration: no-go
    """
    sod = np.full((2, 2), 4, dtype="uint8")
    sic = np.full((2, 2), 5, dtype="uint8")

    rio_pc6 = compute_rio(sod, "PC6", MAPPING, sic=sic)
    assert np.allclose(rio_pc6, 10.0)
    assert is_navigable(rio_pc6, "PC6").all()

    rio_ib = compute_rio(sod, "IB", MAPPING, sic=sic)
    assert np.allclose(rio_ib, -5.0)
    assert not is_navigable(rio_ib, "IB").any()


def test_missing_sic_assumes_full_concentration():
    """SPEC §8.2: a known, declared limitation — the pessimistic assumption."""
    sod = np.full((2, 2), 4, dtype="uint8")
    without = compute_rio(sod, "PC6", MAPPING)
    with_full = compute_rio(sod, "PC6", MAPPING, sic=np.full((2, 2), 10, dtype="uint8"))
    assert np.allclose(without, with_full)
    assert np.allclose(without, -10.0)  # 10 * -1


def test_water_cells_ignore_a_contradictory_sic():
    """If the SOD map says open water, concentration is zero whatever SIC says."""
    sod = np.zeros((2, 2), dtype="uint8")
    rio = compute_rio(sod, "PC6", MAPPING, sic=np.full((2, 2), 10, dtype="uint8"))
    assert np.allclose(rio, 30.0)


def test_invalid_cells_are_nan_not_navigable():
    """An unpredicted cell must never look like navigable water."""
    sod = np.array([[FILL_VALUE, 0], [4, FILL_VALUE]], dtype="uint8")
    rio = compute_rio(sod, "PC6", MAPPING)
    assert np.isnan(rio[0, 0]) and np.isnan(rio[1, 1])
    assert np.isfinite(rio[0, 1])
    nav = is_navigable(rio, "PC6")
    assert not nav[0, 0] and not nav[1, 1]


def test_mask_valid_overrides_prediction():
    sod = np.zeros((2, 2), dtype="uint8")
    mask = np.array([[True, False], [False, True]])
    rio = compute_rio(sod, "PC6", MAPPING, mask_valid=mask)
    assert np.isfinite(rio[0, 0])
    assert np.isnan(rio[0, 1])


# ---------------------------------------------------------------------------
# Thresholds
# ---------------------------------------------------------------------------


def test_threshold_bands_for_polar_class():
    """Table 1.1, PC1-PC7 column."""
    t = RioThresholds()
    assert t.classify(5.0, "PC6") is Operation.NORMAL
    assert t.classify(0.0, "PC6") is Operation.NORMAL
    assert t.classify(-5.0, "PC6") is Operation.ELEVATED
    assert t.classify(-10.0, "PC6") is Operation.ELEVATED
    assert t.classify(-10.1, "PC6") is Operation.SPECIAL


def test_threshold_bands_below_polar_class():
    """Table 1.1, second column: the middle band is already special."""
    t = RioThresholds()
    assert t.classify(1.0, "IA") is Operation.NORMAL
    assert t.classify(-5.0, "IA") is Operation.SPECIAL
    assert t.classify(-20.0, "IA") is Operation.SPECIAL


def test_unknown_rio_classifies_as_unknown():
    assert RioThresholds().classify(float("nan"), "PC6") is Operation.UNKNOWN


def test_classify_raster_codes():
    rio = np.array([[10.0, -5.0], [-20.0, np.nan]])
    codes = classify_rio(rio, "PC6")
    assert codes.tolist() == [[0, 1], [2, 3]]


def test_classify_raster_below_polar_class():
    rio = np.array([[10.0, -5.0]])
    assert classify_rio(rio, "IA").tolist() == [[0, 2]]


# ---------------------------------------------------------------------------
# Speed limits — Table 1.2
# ---------------------------------------------------------------------------


def test_speed_limits_match_table_1_2():
    assert ELEVATED_SPEED_LIMIT_KN["PC1"] == 11.0
    assert ELEVATED_SPEED_LIMIT_KN["PC2"] == 8.0
    assert ELEVATED_SPEED_LIMIT_KN["PC3"] == 5.0
    assert ELEVATED_SPEED_LIMIT_KN["PC5"] == 5.0
    assert ELEVATED_SPEED_LIMIT_KN["PC6"] == 3.0
    assert ELEVATED_SPEED_LIMIT_KN["PC7"] == 3.0


def test_no_speed_limit_in_normal_operation():
    assert speed_limit_kn("PC6", Operation.NORMAL) is None


def test_speed_limit_applies_in_elevated_band():
    assert speed_limit_kn("PC6", Operation.ELEVATED) == 3.0


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------


def test_summary_fractions_sum_to_one():
    rng = np.random.default_rng(0)
    sod = rng.integers(0, 6, size=(50, 50)).astype("uint8")
    s = summarise(compute_rio(sod, "PC6", MAPPING), "PC6")
    total = s["fraction_normal"] + s["fraction_elevated"] + s["fraction_no_go"]
    assert total == pytest.approx(1.0, abs=1e-9)


def test_summary_on_fully_unknown_raster():
    sod = np.full((5, 5), FILL_VALUE, dtype="uint8")
    s = summarise(compute_rio(sod, "PC6", MAPPING), "PC6")
    assert s["fraction_unknown"] == pytest.approx(1.0)
    assert s["rio_min"] is None
