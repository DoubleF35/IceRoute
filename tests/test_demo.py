"""Tests for precompute and the demo's KPI cards.

The card tests are the ones that matter. A KPI card is the last thing between a
measurement and an investor, and the failure mode is not a crash — it is a
plausible number appearing where no measurement exists.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from iceroute.demo.app import (
    downlink_card,
    fmt,
    onboard_card,
    rio_card,
    transit_card,
)
from iceroute.demo.precompute import (
    CLASS_COLOURS,
    NODATA_COLOUR,
    colour_mask,
    colour_rio,
    save_png,
    stretch,
)

# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


def test_stretch_uses_percentiles_not_min_max():
    """One bright scatterer must not flatten the rest of the scene.

    This is the difference between a readable SAR image on a projector and a
    black rectangle with one white dot.
    """
    band = np.concatenate([np.random.default_rng(0).normal(0, 1, 9999), [500.0]])
    valid = np.ones(band.shape, dtype=bool)
    out = stretch(band, valid)
    # With min/max the bulk would collapse to a couple of levels.
    assert out[:-1].std() > 20


def test_stretch_ignores_invalid_pixels():
    band = np.zeros(1000)
    band[:500] = 1000.0  # invalid junk
    valid = np.zeros(1000, dtype=bool)
    valid[500:] = True
    band[500:] = np.linspace(0, 1, 500)
    out = stretch(band, valid)
    assert out[500:].max() > out[500:].min()


def test_stretch_survives_a_constant_band():
    out = stretch(np.full(100, 3.0), np.ones(100, dtype=bool))
    assert out.shape == (100,)
    assert np.isfinite(out).all()


def test_stretch_of_nothing_is_black_not_an_error():
    out = stretch(np.zeros((4, 4)), np.zeros((4, 4), dtype=bool))
    assert out.shape == (4, 4)
    assert (out == 0).all()


def test_mask_colours_are_distinct():
    """They have to stay separable on a washed-out projector."""
    vals = list(CLASS_COLOURS.values()) + [NODATA_COLOUR]
    assert len(set(vals)) == len(vals)


def test_colour_mask_marks_invalid_pixels_as_nodata():
    """An unannotated pixel must not be painted as open water."""
    labels = np.zeros((4, 4), dtype="uint8")
    valid = np.ones((4, 4), dtype=bool)
    valid[0, 0] = False
    out = colour_mask(labels, valid)
    assert tuple(out[0, 0]) == NODATA_COLOUR
    assert tuple(out[1, 1]) == CLASS_COLOURS[0]


def test_colour_rio_uses_the_polaris_boundaries():
    """Green above 0, yellow between -10 and 0, red below."""
    rio = np.array([[5.0, -5.0, -20.0]])
    out = colour_rio(rio, "PC6")
    assert len({tuple(c) for c in out[0]}) == 3


def test_png_downscaling_does_not_invent_classes(tmp_path):
    """Striding, not averaging: averaging class ids creates classes the model
    never predicted."""
    from PIL import Image

    labels = np.zeros((16, 16, 3), dtype="uint8")
    labels[::2] = CLASS_COLOURS[2]
    labels[1::2] = CLASS_COLOURS[0]
    save_png(labels, tmp_path / "m.png", downscale=2)

    back = np.array(Image.open(tmp_path / "m.png"))
    present = {tuple(c) for c in back.reshape(-1, 3)}
    assert present <= {CLASS_COLOURS[0], CLASS_COLOURS[2]}


def test_saved_png_has_the_expected_size(tmp_path):
    from PIL import Image

    arr = np.zeros((100, 80), dtype="uint8")
    save_png(arr, tmp_path / "a.png", downscale=4)
    assert Image.open(tmp_path / "a.png").size == (20, 25)


# ---------------------------------------------------------------------------
# KPI cards — where a fabricated number would reach the room
# ---------------------------------------------------------------------------


def _entry(**over) -> dict:
    base = {
        "route_exists": True,
        "comparison": {
            "iceroute": {"time_hours": 23.7, "distance_km": 528.0, "min_rio": 10.0},
            "baseline": {
                "time_hours": 30.0,
                "feasible": True,
                "blocked_by_ice": False,
                "n_no_go_cells": 0,
                "n_no_data_cells": 0,
            },
            "time_saved_hours": 6.3,
            "time_saved_percent": 21.0,
        },
        "n_waypoints": 9,
    }
    base["comparison"].update(over.pop("comparison", {}))
    base.update(over)
    return base


def test_transit_card_shows_the_saving_when_it_is_real():
    html = transit_card(_entry(), provisional_speed=False)
    assert "23.7 h" in html
    assert "+6.3 h" in html


def test_transit_card_claims_nothing_against_an_impossible_route():
    """SPEC rule 2: you cannot save hours against a voyage POLARIS forbids."""
    e = _entry(
        comparison={
            "baseline": {
                "time_hours": None,
                "feasible": False,
                "blocked_by_ice": True,
                "n_no_go_cells": 93,
                "n_no_data_cells": 0,
            },
            "time_saved_hours": None,
            "time_saved_percent": None,
        }
    )
    html = transit_card(e, provisional_speed=False)
    assert "vietata dal ghiaccio" in html
    assert "93" in html
    assert "+" not in html.split("</div>")[1]  # no fabricated delta


def test_transit_card_distinguishes_no_coverage_from_blocked_by_ice():
    """Conflating them would claim the ice stopped a voyage a data gap stopped."""
    e = _entry(
        comparison={
            "baseline": {
                "time_hours": None,
                "feasible": False,
                "blocked_by_ice": False,
                "n_no_go_cells": 0,
                "n_no_data_cells": 139,
            },
            "time_saved_hours": None,
            "time_saved_percent": None,
        }
    )
    html = transit_card(e, provisional_speed=False)
    assert "non è il ghiaccio" in html
    assert "vietata dal ghiaccio" not in html


def test_transit_card_discloses_the_provisional_speed():
    """The absolute hours scale with an assumption that has no source (P-05)."""
    assert "senza fonte" in transit_card(_entry(), provisional_speed=True)
    assert "senza fonte" not in transit_card(_entry(), provisional_speed=False)


def test_transit_card_handles_a_missing_route():
    html = transit_card({"route_exists": False, "message": "disconnessa"}, False)
    assert "nessuna rotta" in html


@pytest.mark.parametrize(
    ("rio", "expected"),
    [(10.0, "normale"), (-5.0, "elevato"), (-20.0, "speciale")],
)
def test_rio_card_labels_the_polaris_band(rio, expected):
    html = rio_card(
        _entry(comparison={"iceroute": {"time_hours": 1.0, "distance_km": 1.0, "min_rio": rio}})
    )
    assert expected in html


def test_cards_say_not_measured_rather_than_zero():
    """The project's central rule, at the last point before the audience."""
    assert "non misurato" in downlink_card({})
    assert "non misurata" in onboard_card({})
    assert "0.00" not in downlink_card({})
    assert "0.00" not in onboard_card({})


def test_downlink_card_flags_a_provisional_ratio():
    html = downlink_card(
        {"compression": {"ratio_median": 1012.0, "provisional": True, "n_scenes": 3}}
    )
    assert "1012:1" in html
    assert "limite superiore" in html


def test_downlink_card_without_the_flag_makes_no_caveat():
    html = downlink_card(
        {"compression": {"ratio_median": 640.0, "provisional": False, "n_scenes": 5}}
    )
    assert "640:1" in html
    assert "limite superiore" not in html


def test_onboard_card_reports_measured_size_and_latency():
    html = onboard_card(
        {
            "quantization": {
                "size": {"int8_mb": 2.049},
                "latency": {"single_thread": {"int8": {"median_ms_per_tile": 36.2}}},
                "accuracy": {"miou_drop": -0.002},
            }
        }
    )
    assert "2.05 MB" in html
    assert "36 ms" in html
    assert "non degrada" in html


def test_fmt_never_prints_a_number_it_was_not_given():
    assert fmt(None) == "non misurato"
    assert fmt("boh") == "non misurato"
    assert fmt(3.14159) == "3.1"


# ---------------------------------------------------------------------------
# The precomputed index
# ---------------------------------------------------------------------------


def test_labels_resolve_to_the_reference_file_when_one_exists(tmp_path):
    """Official test scenes carry no targets; their charts sit alongside."""
    from iceroute.demo.precompute import resolve_labels

    scene = tmp_path / "s_prep.nc"
    scene.touch()
    assert resolve_labels(scene) is None  # no reference yet

    ref = tmp_path / "s_prep_reference.nc"
    ref.touch()
    assert resolve_labels(scene) == ref


def test_labels_resolution_never_points_a_scene_at_itself(tmp_path):
    """A reference file passed in by mistake must not become its own label
    source — that would score a chart against itself."""
    from iceroute.demo.precompute import resolve_labels

    ref = tmp_path / "s_prep_reference.nc"
    ref.touch()
    assert resolve_labels(ref) != ref


def test_a_training_scene_needs_no_external_labels(tmp_path):
    """Scenes from the frozen split carry their own targets."""
    from iceroute.demo.precompute import resolve_labels

    scene = tmp_path / "20180101T000000_dmi_prep.nc"
    scene.touch()
    assert resolve_labels(scene) is None


def test_index_declares_it_routed_on_the_prediction():
    """Routing on the analyst's chart would demo an analysis we did not produce.
    If that ever changes, the index must stop claiming otherwise."""
    from pathlib import Path

    path = Path("data/demo/index.json")
    if not path.exists():
        pytest.skip("precompute non ancora eseguito")
    index = json.loads(path.read_text())
    assert index.get("generated_from_predicted_mask") is True
    assert "maschera PREDETTA" in index.get("note", "")


def test_scene_scores_reads_the_evaluation_not_the_demo_index():
    """The warning banner's numbers must come from the last evaluation.

    Copying them into the demo index would leave the banner saying 0.25 after a
    retrain moved the scene to 0.60.
    """
    from iceroute.demo.app import scene_scores

    metrics = {
        "evaluation": {
            "scenes": {
                "per_scene": [
                    {"scene_id": "a.nc", "miou": 0.25, "rio": {"worst_unsafe_fraction": 0.52}},
                    {"scene_id": "b.nc", "miou": 0.77, "rio": {"worst_unsafe_fraction": 0.01}},
                ]
            }
        }
    }
    assert scene_scores(metrics, "a.nc") == (0.25, 0.52)
    assert scene_scores(metrics, "b.nc") == (0.77, 0.01)


def test_scene_scores_returns_nothing_when_the_scene_was_not_evaluated():
    """So the banner says 'non misurato' rather than showing another scene's
    numbers under this scene's name."""
    from iceroute.demo.app import scene_scores

    assert scene_scores({}, "a.nc") == (None, None)
    assert scene_scores(
        {"evaluation": {"scenes": {"per_scene": [{"scene_id": "z.nc", "miou": 0.9}]}}},
        "a.nc",
    ) == (None, None)


def test_the_worst_case_scene_is_not_also_in_the_carousel():
    """It is kept to answer "and when does it fail?", not to be shown as typical."""
    from pathlib import Path

    import yaml

    cfg = yaml.safe_load(Path("configs/demo.yaml").read_text())
    worst = (cfg.get("worst_case") or {}).get("id")
    assert worst, "configs/demo.yaml deve dichiarare un caso peggiore"
    assert worst not in [s["id"] for s in cfg["scenes"]]


def test_the_designated_worst_case_really_is_the_worst(tmp_path):
    """D-024: the carousel is chosen partly on model performance, so that claim
    has to keep being true.

    The original selection used only ice-chart structure and put the single
    worst scene of the twenty into the demo. Nothing failed — the routes were
    drawn, the UI worked, and half the pixels sent a ship into ice POLARIS
    forbids. This test fails if a retrain ever reorders the ranking and leaves
    the config behind.
    """
    from pathlib import Path

    import yaml

    metrics_path = Path("artifacts/metrics.json")
    if not metrics_path.exists():
        pytest.skip("valutazione non ancora eseguita")
    per = ((json.loads(metrics_path.read_text()).get("evaluation") or {}).get("scenes") or {}).get(
        "per_scene"
    ) or []
    scored = {r["scene_id"]: (r.get("rio") or {}).get("worst_unsafe_fraction") for r in per}

    cfg = yaml.safe_load(Path("configs/demo.yaml").read_text())
    worst_id = cfg["worst_case"]["id"]
    if not isinstance(scored.get(worst_id), float):
        pytest.skip("il caso peggiore non compare nella valutazione corrente")

    for entry in cfg["scenes"]:
        got = scored.get(entry["id"])
        if isinstance(got, float):
            assert got < scored[worst_id], (
                f"{entry['id']} è più permissiva ({got:.1%}) del caso peggiore "
                f"dichiarato {worst_id} ({scored[worst_id]:.1%}): "
                "aggiorna configs/demo.yaml"
            )


def test_index_records_whether_routes_differ_by_vessel_class():
    """D-012: the demo's strongest moment needs this, so it is recorded per
    scene and decided on evidence rather than hope."""
    from pathlib import Path

    path = Path("data/demo/index.json")
    if not path.exists():
        pytest.skip("precompute non ancora eseguito")
    for s in json.loads(path.read_text())["scenes"]:
        assert "routes_differ" in s


# ---------------------------------------------------------------------------
# "But your route goes through the thick ice"
# ---------------------------------------------------------------------------


def test_conditions_along_separates_ice_type_from_concentration():
    """The objection the demo used to invite and could not answer.

    A class map shows ice *type*. POLARIS scores type times *concentration*.
    Heavy ice at 3/10 is normal operation; the same ice at 8/10 is forbidden.
    Identical colour on screen, opposite verdict.
    """
    import numpy as np

    from iceroute.routing.polaris import conditions_along

    names = ["Acqua libera", "Ghiaccio sottile", "Ghiaccio spesso"]
    sod = np.full((4, 4), 2, dtype="uint8")  # all thick ice
    sic = np.full((4, 4), 3.0)  # ...at 3/10
    # RIO = 3(-3) + 7(+3) = +12 for a PC6
    rio = np.full((4, 4), 12.0)
    cells = [(0, i) for i in range(4)]

    c = conditions_along(cells, sod, sic, rio, "PC6", names)
    assert c["by_class"] == {"Ghiaccio spesso": 4}
    assert c["worst_class_cells"] == 4
    assert c["worst_class_sic_median"] == 3.0
    assert c["fraction_normal"] == 1.0
    assert c["fraction_no_go"] == 0.0


def test_the_same_ice_type_at_high_concentration_is_reported_as_forbidden():
    import numpy as np

    from iceroute.routing.polaris import conditions_along

    names = ["Acqua libera", "Ghiaccio sottile", "Ghiaccio spesso"]
    sod = np.full((4, 4), 2, dtype="uint8")
    sic = np.full((4, 4), 8.0)
    rio = np.full((4, 4), -18.0)  # 8(-3) + 2(+3)
    c = conditions_along([(0, i) for i in range(4)], sod, sic, rio, "PC6", names)

    assert c["by_class"] == {"Ghiaccio spesso": 4}  # same class as above
    assert c["worst_class_sic_median"] == 8.0  # different concentration
    assert c["fraction_no_go"] == 1.0  # opposite verdict


def test_conditions_of_an_empty_route_are_empty():
    import numpy as np

    from iceroute.routing.polaris import conditions_along

    assert conditions_along([], np.zeros((2, 2)), None, np.zeros((2, 2)), "PC6", ["a", "b"]) == {}


def test_every_demo_route_stays_in_normal_operation():
    """The claim the UI now makes out loud, checked against the real build.

    If a route ever does cross a forbidden cell, the banner saying "attraversa
    il bianco, ed è regolare" becomes a false statement on a projector.
    """
    from pathlib import Path

    path = Path("data/demo/index.json")
    if not path.exists():
        pytest.skip("precompute non ancora eseguito")

    checked = 0
    for scene in json.loads(path.read_text()).get("scenes", []):
        for ice_class, entry in (scene.get("classes") or {}).items():
            if not entry.get("route_exists"):
                continue
            c = entry.get("conditions") or {}
            assert c, f"{scene['scene_id']}/{ice_class}: condizioni non misurate"
            assert c["fraction_no_go"] == 0.0, (
                f"{scene['scene_id']}/{ice_class}: la rotta attraversa "
                f"{100 * c['fraction_no_go']:.0f}% di celle vietate"
            )
            assert c["rio_min"] is not None and c["rio_min"] >= 0, (
                f"{scene['scene_id']}/{ice_class}: RIO minimo {c['rio_min']}"
            )
            checked += 1
    assert checked, "nessuna rotta verificata"


def test_the_layer_that_decides_is_the_one_shown_first():
    """RIO leads. The class map cannot show passability, and leading with it is
    what made the route look illegal to the first person who looked."""
    from iceroute.demo.app import LAYERS

    first = next(iter(LAYERS))
    assert first.startswith("RIO"), f"il primo livello è {first!r}, non RIO"
    assert LAYERS[first][1], "il livello RIO deve poter mostrare le rotte"


def test_concentration_is_available_as_its_own_layer():
    """Without it the demo can assert why the route is legal but not show it."""
    from iceroute.demo.app import LAYERS

    assert "Concentrazione" in LAYERS
    assert LAYERS["Concentrazione"] == ("sic", True)
