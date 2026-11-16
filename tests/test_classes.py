"""Tests for the deck's ice-class table.

The table exists because a hand-written replacement for a wrong table produced a
second wrong one: three rows that read as equally trustworthy, when thin ice
scores 0.136. So the property these tests protect is not formatting — it is that
the page cannot present a weak class as a strong one, and cannot quietly drop
the one error with a safety consequence.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from iceroute.report import classes as cl

ASM = {
    "segmentation": {"scheme": "3 classi", "classes": {
        0: {"name": "Acqua libera", "polaris_type": "ice_free"},
        1: {"name": "Ghiaccio sottile", "polaris_type": "thin_fyi_2nd_stage"},
        2: {"name": "Ghiaccio spesso", "polaris_type": "heavy_multi_year_ice"},
    }},
    "vessel": {"classes_shown": ["PC6", "IA"]},
}


def _metrics(cm=None):
    return {"evaluation": {"scenes": {
        "n_scenes": 20,
        "aggregate": {
            "miou": 0.5324, "binary_accuracy": 0.9059,
            "per_class": {
                "Acqua libera": {"iou": 0.846, "support_px": 168_151_799},
                "Ghiaccio sottile": {"iou": 0.136, "support_px": 24_779_694},
                "Ghiaccio spesso": {"iou": 0.615, "support_px": 109_062_176},
            },
            # rows = truth, columns = predicted
            "confusion_matrix": cm or [[928, 12, 60], [56, 200, 745], [137, 87, 776]],
        },
        "per_scene": [{"rio": {"worst_unsafe_fraction": f}}
                      for f in (0.004, 0.009, 0.02, 0.413)],
    }}}


# ---------------------------------------------------------------------------
# The weak class cannot hide
# ---------------------------------------------------------------------------


def test_the_weakest_class_shows_its_score():
    page = cl.render(cl.collect(_metrics(), ASM, cl.RV_TABLE))
    assert "0.136" in page, "il ghiaccio sottile deve mostrare il proprio IoU"
    assert "0.846" in page and "0.615" in page


def test_the_dangerous_direction_is_named_and_quantified():
    """Thick ice called open water is the only mistake that can put a ship
    somewhere it may not go. It has to be a number, not an adjective."""
    c = cl.collect(_metrics(), ASM, cl.RV_TABLE)
    assert c["permissive_thick_as_water"] == pytest.approx(137 / 1000 * 100, abs=0.1)
    assert "13.7%" in cl.render(c)


def test_the_safe_direction_is_distinguished_from_the_dangerous_one():
    """Thin-called-thick and thick-called-water are both errors and are not the
    same kind of error. A table that only reported accuracy would hide that."""
    c = cl.collect(_metrics(), ASM, cl.RV_TABLE)
    rows = {r.name: r for r in c["rows"]}
    assert rows["Ghiaccio sottile"].mistaken_for == "Ghiaccio spesso"
    assert rows["Ghiaccio spesso"].mistaken_for == "Acqua libera"


def test_the_worst_case_is_reported_not_just_the_median():
    c = cl.collect(_metrics(), ASM, cl.RV_TABLE)
    assert c["unsafe_max"] == pytest.approx(0.413)
    assert "41.3%" in cl.render(c)


# ---------------------------------------------------------------------------
# POLARIS, because the class alone decides nothing
# ---------------------------------------------------------------------------


def test_each_class_carries_its_risk_value_per_vessel():
    c = cl.collect(_metrics(), ASM, cl.RV_TABLE)
    rows = {r.name: r for r in c["rows"]}
    assert rows["Acqua libera"].riv["PC6"] == 3
    assert rows["Ghiaccio spesso"].riv["PC6"] < 0
    assert rows["Ghiaccio spesso"].riv["IA"] < rows["Ghiaccio spesso"].riv["PC6"]


def test_the_formula_is_on_the_page():
    """Without it the table invites exactly the reading D-031 caught: that the
    class alone says whether you may go."""
    assert "RIO = C" in cl.render(cl.collect(_metrics(), ASM, cl.RV_TABLE))


def test_ridges_are_addressed_rather_than_silently_dropped():
    page = cl.render(cl.collect(_metrics(), ASM, cl.RV_TABLE))
    assert "ridge" in page.lower() and "non può imparare" in page


# ---------------------------------------------------------------------------
# Missing measurements
# ---------------------------------------------------------------------------


def test_no_evaluation_means_no_invented_scores():
    c = cl.collect({}, ASM, cl.RV_TABLE)
    page = cl.render(c)
    assert "non misurato" in page
    assert "0.000" not in page


def test_a_confusion_matrix_of_the_wrong_shape_is_ignored_not_guessed():
    m = _metrics(cm=[[1, 0], [0, 1]])
    c = cl.collect(m, ASM, cl.RV_TABLE)
    assert c["permissive_thick_as_water"] is None


def test_a_missing_risk_table_still_produces_a_page():
    c = cl.collect(_metrics(), ASM, Path("nessun/file.csv"))
    assert "—" in cl.render(c)


# ---------------------------------------------------------------------------
# Against the real repository
# ---------------------------------------------------------------------------


def test_the_generated_file_matches_the_current_measurements():
    """If this fails, `make report` is due — the deck table has drifted from the
    model, which is the whole failure mode A-11 is about."""
    out = Path("pitch/class_table.md")
    if not (out.exists() and Path("artifacts/metrics.json").exists()):
        pytest.skip("class_table.md o metrics.json assenti")
    import json
    c = cl.collect(json.loads(Path("artifacts/metrics.json").read_text()),
                   yaml.safe_load(Path("configs/assumptions.yaml").read_text()),
                   cl.RV_TABLE)
    assert out.read_text() == cl.render(c), "pitch/class_table.md è indietro: make report"


def test_the_real_table_has_three_classes_and_none_of_them_is_a_ridge():
    out = Path("pitch/class_table.md")
    if not out.exists():
        pytest.skip("class_table.md assente")
    text = out.read_text()
    for name in ("Open water", "Thin ice", "Thick ice"):
        assert f"**{name}**" in text

    # "Deformed Ice" is allowed to appear — the page names it in order to say it
    # does not exist — but never as a row of the table we are claiming to output.
    rows = [ln for ln in text.splitlines() if ln.startswith("| **")]
    assert len(rows) == 3, rows
    assert not any("Deformed" in ln or "Ridge" in ln for ln in rows)
