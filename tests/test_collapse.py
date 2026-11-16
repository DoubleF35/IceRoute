"""Tests for the three-class fallback scheme (SPEC §7.4).

Two properties carry the argument for this reduction, and both are asserted
against the POLARIS circular rather than taken on trust:
``test_merging_rounds_towards_danger`` and
``test_conservative_member_agrees_across_vessel_classes``. If either stopped
holding, the fallback would silently start manufacturing permissive errors.
"""

from __future__ import annotations

import numpy as np
import pytest

from iceroute.data.collapse import (
    GROUPS,
    N_CLASSES,
    NAMES,
    collapse,
    collapse_proba,
    conservative_member,
    describe,
    lookup_table,
    polaris_mapping,
)
from iceroute.data.loader import load_config
from iceroute.model.metrics import IGNORE_INDEX
from iceroute.routing.polaris import PolarisTable

CFG = load_config("configs/data.yaml")
MAPPING = {int(k): v for k, v in CFG["sod_to_polaris"].items()}
TABLE = PolarisTable.load()
VESSELS = ["PC6", "PC7", "IA"]

OPEN_WATER, NEW_ICE, YOUNG_ICE, THIN_FYI, THICK_FYI, OLD_ICE = range(6)


# ---------------------------------------------------------------------------
# The grouping itself
# ---------------------------------------------------------------------------


def test_every_sod_class_is_assigned_exactly_once():
    assigned = [c for olds in GROUPS.values() for c in olds]
    assert sorted(assigned) == list(range(6))


def test_collapse_maps_the_expected_classes():
    src = np.array([OPEN_WATER, NEW_ICE, YOUNG_ICE, THIN_FYI, THICK_FYI, OLD_ICE], "uint8")
    assert collapse(src).tolist() == [0, 1, 1, 1, 2, 2]


def test_ignored_pixels_stay_ignored():
    """Mapping 255 to class 0 would relabel every unannotated pixel as open
    water — wrong, and permissive in the dangerous direction."""
    assert collapse(np.array([IGNORE_INDEX], "uint8"))[0] == IGNORE_INDEX
    assert lookup_table()[IGNORE_INDEX] == IGNORE_INDEX


def test_unknown_class_ids_do_not_become_open_water():
    """Anything outside the known scheme must land on ignore, not on water."""
    assert collapse(np.array([7, 99, 200], "uint8")).tolist() == [IGNORE_INDEX] * 3


def test_collapse_preserves_shape_and_dtype():
    src = np.random.default_rng(0).integers(0, 6, size=(17, 23)).astype("uint8")
    out = collapse(src)
    assert out.shape == src.shape
    assert out.dtype == np.uint8


def test_collapse_is_idempotent_on_already_collapsed_ids():
    """0, 1 and 2 are valid in both schemes; applying twice must not shift them."""
    once = collapse(np.array([OPEN_WATER, NEW_ICE, YOUNG_ICE], "uint8"))
    assert collapse(once).tolist() == once.tolist()


# ---------------------------------------------------------------------------
# Why the merge is safe — checked against the circular
# ---------------------------------------------------------------------------


def test_conservative_member_agrees_across_vessel_classes():
    """One POLARIS ice type can only represent a group if the most dangerous
    member is the same for every vessel. It is — but that is a property of
    MSC.1/Circ.1519, not a guarantee, so it is checked."""
    for olds in GROUPS.values():
        _, agrees = conservative_member(olds, MAPPING, VESSELS, table=TABLE)
        assert agrees, f"il membro più cauto di {olds} cambia con la classe di nave"


def test_merging_rounds_towards_danger():
    """The central safety property.

    Each collapsed class must take the risk value of its worst member, for
    every vessel. If it took the best, the reduction would turn dangerous ice
    into navigable ice — the exact failure the fallback is meant to avoid.
    """
    m3 = polaris_mapping(MAPPING, VESSELS, table=TABLE)
    for new, olds in GROUPS.items():
        for vessel in VESSELS:
            merged = TABLE.riv(vessel, m3[new])
            members = [TABLE.riv(vessel, MAPPING[o]) for o in olds]
            assert merged == min(members), (
                f"classe {new} per {vessel}: RIV {merged}, il peggiore fra i "
                f"membri è {min(members)}"
            )


def test_new_ice_and_young_ice_merge_for_free():
    """They carry identical risk values, so this part of the merge costs
    nothing at all in POLARIS terms (D-015)."""
    for vessel in VESSELS:
        assert TABLE.riv(vessel, MAPPING[NEW_ICE]) == TABLE.riv(vessel, MAPPING[YOUNG_ICE])


def test_polaris_mapping_covers_every_collapsed_class():
    m3 = polaris_mapping(MAPPING, VESSELS, table=TABLE)
    assert set(m3) == set(GROUPS)
    assert all(t in TABLE.ice_types for t in m3.values())


def test_collapsed_risk_values_stay_ordered():
    """Water must remain safer than thin ice, and thin than thick."""
    m3 = polaris_mapping(MAPPING, VESSELS, table=TABLE)
    for vessel in VESSELS:
        riv = [TABLE.riv(vessel, m3[c]) for c in range(N_CLASSES)]
        assert riv[0] > riv[1] > riv[2], f"{vessel}: {riv}"


# ---------------------------------------------------------------------------
# Probabilities
# ---------------------------------------------------------------------------


def test_collapsed_probabilities_sum_within_groups():
    p = np.zeros((6, 2, 2), dtype="float32")
    p[NEW_ICE] = 0.2
    p[YOUNG_ICE] = 0.3
    p[THIN_FYI] = 0.1
    p[OPEN_WATER] = 0.4
    out = collapse_proba(p)
    assert out.shape == (N_CLASSES, 2, 2)
    assert np.allclose(out[1], 0.6)
    assert np.allclose(out[0], 0.4)


def test_collapsed_probabilities_stay_normalised():
    rng = np.random.default_rng(2)
    p = rng.random((6, 8, 8)).astype("float32")
    p /= p.sum(axis=0, keepdims=True)
    assert np.allclose(collapse_proba(p).sum(axis=0), 1.0, atol=1e-5)


def test_summing_beats_maximum_when_mass_is_spread():
    """A group whose probability is split across members must not be
    under-reported — which is what taking the max would do."""
    p = np.zeros((6, 1, 1), dtype="float32")
    p[OPEN_WATER] = 0.4
    p[NEW_ICE] = p[YOUNG_ICE] = p[THIN_FYI] = 0.2  # 0.6 together, 0.2 each
    out = collapse_proba(p)
    assert out[1, 0, 0] > out[0, 0, 0]


def test_collapse_proba_rejects_a_bad_shape():
    with pytest.raises(ValueError, match=r"n_classes, H, W"):
        collapse_proba(np.zeros((6, 4)))


# ---------------------------------------------------------------------------
# The disclaimer
# ---------------------------------------------------------------------------


def test_describe_says_the_problem_was_reduced():
    """SPEC §7.4 forbids adjusting the number; the label is what keeps a
    three-class score from being read as a six-class one."""
    text = describe(0.272)
    assert "RIDOTTO" in text
    assert "PIÙ FACILE" in text
    assert "0,60" in text
    assert "0.272" in text


def test_names_cover_every_class():
    assert set(NAMES) == set(GROUPS)


# ---------------------------------------------------------------------------
# Class weights
# ---------------------------------------------------------------------------


def test_class_weights_use_the_collapsed_frequencies(tmp_path):
    """The cache holds six-class labels. Weighting three classes with the first
    three six-class frequencies would be silently wrong."""
    import json

    from iceroute.model.losses import load_class_weights

    stats = tmp_path / "s.json"
    stats.write_text(
        json.dumps(
            {
                "class_frequency": {
                    "0": 0.70,
                    "1": 0.01,
                    "2": 0.02,
                    "3": 0.03,
                    "4": 0.15,
                    "5": 0.09,
                }
            }
        )
    )

    w = load_class_weights(stats, N_CLASSES, collapse=True, normalize=False, clip_max=100.0)
    assert w is not None
    # Thin group is 0.01+0.02+0.03 = 0.06 of pixels, thick 0.15+0.09 = 0.24.
    assert w[1] > w[2] > w[0], f"pesi non ordinati per rarità: {w.tolist()}"
    assert w[0] == pytest.approx(1 / 0.70, rel=1e-3)
    assert w[1] == pytest.approx(1 / 0.06, rel=1e-3)
    assert w[2] == pytest.approx(1 / 0.24, rel=1e-3)


def test_confusion_matrix_silently_drops_out_of_range_truth():
    """The mechanism behind a bug that produced a flattering, plausible mIoU.

    Scoring six-class truth against three-class predictions does not raise: it
    discards every pixel whose label is >= n_classes, so classes 3, 4 and 5
    vanish and the survivors are compared under different semantics. Pinning the
    behaviour here is what makes the requirement to collapse the truth explicit.
    """
    from iceroute.model.metrics import ConfusionMatrix

    cm = ConfusionMatrix(N_CLASSES)
    truth = np.array([0, 1, 2, 3, 4, 5])
    cm.update(truth, np.zeros(6, dtype="int64"))
    assert cm.matrix.sum() == 3, "i pixel 3, 4 e 5 vengono scartati senza errore"


def test_collapsed_mapping_separates_thin_from_thick():
    """The second half of the same bug.

    Pricing collapsed class 1 with the six-class SOD mapping makes it "New ice"
    and class 2 "Young ice" — which carry identical risk values, so every RIO
    error cancels and the report shows 0.00 MAE with 0% permissive. The most
    flattering possible wrong answer.
    """
    six = MAPPING
    three = polaris_mapping(MAPPING, VESSELS, table=TABLE)

    for vessel in VESSELS:
        wrong = TABLE.riv(vessel, six[1]) - TABLE.riv(vessel, six[2])
        right = TABLE.riv(vessel, three[1]) - TABLE.riv(vessel, three[2])
        assert wrong == 0, "presupposto del bug: New ice e Young ice hanno lo stesso RIV"
        assert right != 0, f"{vessel}: ghiaccio sottile e spesso devono differire"


def test_collapsed_weights_still_respect_the_clip(tmp_path):
    """The ceiling that stops a rare class dominating every gradient applies
    after merging too, not only to the six-class frequencies."""
    import json

    from iceroute.model.losses import load_class_weights

    stats = tmp_path / "s.json"
    stats.write_text(
        json.dumps(
            {
                "class_frequency": {
                    "0": 0.94,
                    "1": 0.001,
                    "2": 0.002,
                    "3": 0.003,
                    "4": 0.03,
                    "5": 0.024,
                }
            }
        )
    )
    w = load_class_weights(stats, N_CLASSES, collapse=True, normalize=False, clip_max=10.0)
    assert float(w.max()) <= 10.0
