"""Tests for presentation mode.

This page replaces the slides at a five-minute demo night, so it fails in a
particular way: not with an exception, but with a screen that looks finished and
says something slightly untrue, in front of people, with a timer running. The
tests are aimed there — at refusals surviving into the page, at the one number
nobody records, and at data that is computed and then dropped on the floor.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from iceroute.pitch import present as pr


def _entry(blocked=True, normal=1.0, no_go=227, base_time=None):
    return {
        "route_exists": True,
        "comparison": {
            "iceroute": {"time_hours": 15.613, "distance_km": 346.99, "min_rio": 0.0},
            "baseline": {"time_hours": base_time, "distance_km": 315.898, "min_rio": -18.0,
                         "feasible": not blocked, "blocked_by_ice": blocked,
                         "n_no_go_cells": no_go, "n_no_data_cells": 101},
            "detour_km": 31.092,
        },
        "conditions": {"fraction_normal": normal},
    }


_OVERLAY = {"grid_shape": [1246, 1253],
            "baseline": {"path": [[0, c] for c in range(748)], "no_go": []},
            "iceroute": {"path": [[1, c] for c in range(449)], "waypoints": [[1, 1]]}}


# ---------------------------------------------------------------------------
# The number nobody recorded
# ---------------------------------------------------------------------------


def test_the_direct_route_never_claims_a_share_of_normal_cells():
    """Only the forbidden count is measured for the baseline.

    The index records forbidden cells and no-data cells; the remainder is not
    split between normal operation and elevated risk anywhere. Reporting "56% in
    navigazione normale" would be the complement of two measurements presented
    as a third, which is the kind of number this project does not print.
    """
    b = pr.route_block(_entry(), _OVERLAY)["base"]
    assert "normale" not in json.dumps(b, ensure_ascii=False)
    assert b["cells"] == "227 celle vietate su 748"


def test_our_route_does_report_it_because_it_is_measured():
    assert pr.route_block(_entry(), _OVERLAY)["ours"]["cells"] == "100% in navigazione normale"


def test_a_blocked_direct_route_shows_no_transit_time():
    """D-030: precompute refuses to time a route nobody can sail, and the
    refusal has to survive the trip onto the projector."""
    b = pr.route_block(_entry(blocked=True), _OVERLAY)["base"]
    assert b["time"] is None and b["blocked"] is True


def test_a_sailable_direct_route_does_show_one():
    b = pr.route_block(_entry(blocked=False, base_time=14.0), _OVERLAY)["base"]
    assert b["time"] == "14,0 h"


def test_an_overlay_without_a_baseline_path_reports_no_cell_count():
    b = pr.route_block(_entry(), {"grid_shape": [10, 10]})["base"]
    assert b["cells"] is None, "senza il percorso non c'è un totale su cui fare la frazione"


# ---------------------------------------------------------------------------
# Scene extent is a claim like any other
# ---------------------------------------------------------------------------


def test_extent_uses_the_configured_pixel_spacing():
    assert pr.extent([4997, 5351], 80) == "400 × 428 km"


def test_extent_without_a_spacing_is_absent_not_assumed():
    """80 m is right for AI4Arctic and is still not something to hardcode here:
    if assumptions.yaml stops saying it, the screen says nothing."""
    assert pr.extent([4997, 5351], None) is None


def test_extent_of_a_malformed_shape_is_absent():
    assert pr.extent([4997], 80) is None and pr.extent(None, 80) is None


# ---------------------------------------------------------------------------
# Italian formatting, because the audience reads it out loud
# ---------------------------------------------------------------------------


def test_thousands_and_decimals_are_italian():
    assert pr.it(169558) == f"169{pr.NBSP}558"
    assert pr.it(55.2522, 1) == "55,3"


def test_grouping_uses_a_space_that_cannot_wrap():
    """A number broken across two lines reads as two numbers from the back of
    a room."""
    assert " " not in pr.it(169558) and pr.NBSP in pr.it(169558)


# ---------------------------------------------------------------------------
# The page
# ---------------------------------------------------------------------------


def _ctx():
    return {
        "scenes": [{
            "id": "S.nc", "label": "20180707", "region": "Kara", "acquired": "2018-07-07",
            "extent": "400 × 428 km", "sar": "data:image/jpeg;base64,AA",
            "mask_pred": "data:image/png;base64,AA", "mask_geo": "data:image/png;base64,AA",
            "thumb": "data:image/jpeg;base64,AA", "ar_swath": 1.07, "ar_geo": 1.006,
            "classes": {"PC6": {"exists": True, "svg": "<svg/>",
                                "routes": pr.route_block(_entry(), _OVERLAY),
                                "payload": None}},
        }],
        "skipped": ["worst.nc"], "classes": ["PC6"], "legend": [("Acqua libera", "#0a285a")],
        "route_colours": {"ours": "#ff2d95", "base": "#f5f5f5", "nogo": "#ff2020"},
        "model": {"int8": "2,05 MB", "binary": "0,906", "miou": "0,532",
                  "n_scenes": 20, "scheme": "3 classi"},
    }


def test_the_page_declares_its_charset():
    """Opened with a double click there is no server to send a Content-Type.
    Without this every accented Italian word is mojibake on the projector —
    which is exactly how this shipped the first time it was generated."""
    assert '<meta charset="utf-8">' in pr.build_page(_ctx())[:200]


def test_the_page_is_self_contained():
    page = pr.build_page(_ctx())
    for remote in ("http://", "https://", "//cdn", "<link"):
        assert remote not in page, f"riferimento esterno: {remote}"


def test_the_page_carries_no_arithmetic():
    """Same contract as standalone: Python formats, the browser selects. A page
    that cannot compute cannot compute something wrong on stage."""
    page = pr.build_page(_ctx())
    js = page.split("const D = ")[1]
    for op in ("/ 1000", "* 100", "toFixed", "Math.round(p."):
        assert op not in js.split("</script>")[-1], f"conto nel browser: {op}"


def test_the_legend_is_read_from_the_palette_not_retyped():
    """One channel out is invisible and still makes the legend a claim about
    the image instead of a reading of it. The live UI has that slip."""
    from iceroute.demo.precompute import CLASS_COLOURS

    index = {"class_names": {"0": "Acqua libera", "1": "Ghiaccio sottile",
                             "2": "Ghiaccio spesso"}}
    got = pr.class_legend(index)
    assert [c for _, c in got] == [pr._rgb(CLASS_COLOURS[i]) for i in (0, 1, 2)]
    assert [n for n, _ in got] == ["Acqua libera", "Ghiaccio sottile", "Ghiaccio spesso"]


def test_a_class_the_index_does_not_name_is_left_out_of_the_legend():
    assert pr.class_legend({"class_names": {"0": "Acqua libera"}}) == [
        ("Acqua libera", pr._rgb((10, 40, 90)))]


def test_the_route_colours_come_from_the_overlay_module():
    from iceroute.demo import overlay as ov

    page = pr.build_page(_ctx())
    assert pr._rgb(ov.COL_ICEROUTE) == "#ff2d95"
    assert "--ours:#ff2d95" in page


# ---------------------------------------------------------------------------
# Against the real build
# ---------------------------------------------------------------------------


def _real():
    page = Path("present.html")
    if not page.exists():
        pytest.skip("present.html assente: make present")
    text = page.read_text()
    return text, json.loads(re.search(r"const D = (\{.*?\}); const STEPS", text, re.S).group(1))


def test_every_scene_carries_its_per_class_data():
    """This failed on the first run and only showed up in the browser console:
    the per-class block was built and never attached to the scene, so the page
    threw on the first key press. Computed and then dropped is D-027's shape."""
    _, d = _real()
    assert d["scenes"], "nessuna scena"
    for s in d["scenes"]:
        assert s.get("classes"), f"{s['label']}: nessuna classe"
        for ic in d["classes"]:
            assert ic in s["classes"], f"{s['label']}: manca {ic}"


def test_every_route_has_geometry_and_numbers():
    _, d = _real()
    for s in d["scenes"]:
        for ic, e in s["classes"].items():
            if not e.get("exists"):
                continue
            assert "<polyline" in e["svg"], f"{s['label']}/{ic}: SVG senza tracce"
            assert e["routes"]["ours"]["distance"], f"{s['label']}/{ic}: nessuna distanza"


def test_every_scene_knows_the_shape_of_its_own_rasters():
    """The aspect ratios are what stop the image overflowing its figure and
    being silently clipped — the bottom of the scene missing from the screen."""
    _, d = _real()
    for s in d["scenes"]:
        assert s["ar_swath"] and s["ar_geo"], f"{s['label']}: rapporti mancanti"


def test_the_worst_case_scene_is_not_in_presentation_mode():
    """It has no compression measurement, so steps 1 and 4 would show two holes.
    It stays in the full demo, where the gap is explained rather than projected."""
    _, d = _real()
    index = json.loads(Path("data/demo/index.json").read_text())
    worst = {s["scene_id"] for s in index["scenes"] if s.get("worst_case")}
    assert worst, "l'indice non marca più nessun caso peggiore"
    assert not ({s["id"] for s in d["scenes"]} & worst)
    assert d["skipped"], "la scena esclusa va dichiarata, non fatta sparire"


def test_every_scene_in_the_page_has_its_compression_measured():
    """The converse of the test above: nothing on these five screens is a hole."""
    _, d = _real()
    for s in d["scenes"]:
        for e in s["classes"].values():
            if e.get("exists") and e.get("payload"):
                assert e["payload"]["measured"] is True, f"{s['label']}: compressione stimata"
                assert e["payload"]["ratio"], f"{s['label']}: nessun rapporto"
