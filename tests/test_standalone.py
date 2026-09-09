"""Tests for the single-file offline demo.

This page is the thing that gets opened when the real demo will not start, in
front of investors, by someone who has no way to check it. So the tests care
about two properties above all: that it carries no logic of its own, and that
it cannot show a number Python decided to withhold.
"""

from __future__ import annotations

import json
import re

import pytest

from iceroute.pitch import standalone as sa


def _entry(hours=24.0, min_rio=6.0, feasible=False, blocked=True):
    return {"comparison": {
        "iceroute": {"time_hours": hours, "min_rio": min_rio, "distance_km": 300.0},
        "baseline": {"time_hours": 22.0 if feasible else None, "feasible": feasible,
                     "blocked_by_ice": blocked, "n_no_go_cells": 12 if blocked else 0},
    }}


# ---------------------------------------------------------------------------
# The refusal survives the trip into the browser
# ---------------------------------------------------------------------------


def test_a_route_in_normal_operation_gets_a_value_at_every_speed():
    t = sa.transit_table(_entry(min_rio=6.0), "PC6", 12.0)
    assert len(t["values"]) == len(sa.SPEEDS)
    assert all(v is not None for v in t["values"])


def test_a_route_through_the_elevated_band_is_null_at_every_speed():
    """The page must not be able to show a figure Python refused to compute.

    Baking the table rather than doing the arithmetic in JavaScript is what
    makes this impossible rather than merely unlikely: there is no rule in the
    browser that could disagree.
    """
    t = sa.transit_table(_entry(min_rio=-4.0), "PC6", 12.0)
    assert all(v is None for v in t["values"])
    assert "inventerebbe" in t["note"]


def test_the_table_matches_the_live_ui_exactly():
    """Same function, so the offline copy cannot drift by a rounding rule."""
    from iceroute.demo.estimate import rescale_transit

    t = sa.transit_table(_entry(hours=24.0, min_rio=6.0), "PC6", 12.0)
    for speed, got in zip(sa.SPEEDS, t["values"], strict=True):
        want = rescale_transit(24.0, 6.0, nominal_kn=12.0,
                               chosen_kn=speed, ice_class="PC6").hours
        assert got == pytest.approx(round(want, 2))


def test_an_impossible_direct_route_has_no_baseline_column():
    t = sa.transit_table(_entry(feasible=False), "PC6", 12.0)
    assert all(v is None for v in t["baseline"])


def test_a_feasible_direct_route_does_have_one():
    t = sa.transit_table(_entry(feasible=True, blocked=False), "PC6", 12.0)
    assert all(v is not None for v in t["baseline"])


def test_the_nominal_speed_is_a_dial_position():
    """Otherwise the page opens on a value it cannot represent."""
    assert 12.0 in sa.SPEEDS


# ---------------------------------------------------------------------------
# The overlay is the same geometry, in a different medium
# ---------------------------------------------------------------------------


def _ov(no_go=None):
    return {
        "grid_shape": [40, 60],
        "iceroute": {"path": [[r, r] for r in range(2, 20)], "waypoints": [[2, 2], [19, 19]]},
        "baseline": {"path": [[20, c] for c in range(2, 58)], "no_go": no_go or []},
        "start": [2, 2], "goal": [19, 19],
    }


def test_the_svg_declares_the_grid_it_was_built_for():
    """The two-grid trap again: an SVG scaled to the wrong viewBox draws a
    plausible line in the wrong place."""
    svg = sa.svg_overlay(_ov())
    assert 'viewBox="0 0 60 40"' in svg


def test_both_tracks_reach_the_svg():
    svg = sa.svg_overlay(_ov())
    assert 'class="lay-baseline"' in svg and 'class="lay-iceroute"' in svg
    assert svg.count("<polyline") >= 4       # casing + core, twice


def test_forbidden_stretches_are_runs_not_one_marker_per_cell():
    svg = sa.svg_overlay(_ov(no_go=[[20, c] for c in range(10, 30)]))
    assert 'class="nogo"' in svg
    assert svg.count("<circle") <= 4         # waypoints and endpoints only


def test_an_isolated_forbidden_cell_becomes_a_marker():
    svg = sa.svg_overlay(_ov(no_go=[[20, 10]]))
    assert 'class="nogod"' in svg


def test_no_overlay_is_no_svg():
    assert sa.svg_overlay({}) == ""


# ---------------------------------------------------------------------------
# The page itself
# ---------------------------------------------------------------------------


def _ctx():
    return {
        "scenes": [{"id": "S.nc", "region": "Kara", "acquired": "2020-07-01",
                    "coverage": 0.8, "worst_case": False, "files": {"sar": "a.png"},
                    "classes": {"PC6": {"route_exists": True, "svg": "", "rio_img": None,
                                        "comparison": _entry()["comparison"],
                                        "n_waypoints": 5, "polaris": {}, "conditions": {},
                                        "transit": {"values": [], "baseline": [], "note": ""},
                                        "payload": None, "message": ""}}}],
        "images": {"a.png": "data:image/png;base64,AAAA"},
        "speeds": sa.SPEEDS, "nominal_kn": 12.0, "hours_per_day": 24.0,
        "scheme": "3 classi", "checkpoint": "m.pt",
        "aggregate": {"miou": 0.5324, "binary_accuracy": 0.9059},
        "n_eval_scenes": 20, "compression": 439.2, "quant": {"int8_mb": 2.05},
    }


def test_the_page_is_self_contained():
    """It has to open from a USB stick with the network off. Anything fetched
    from a host is a blank panel on stage."""
    page = sa.build_page(_ctx())
    for remote in ("http://", "https://", "//cdn", "<link"):
        assert remote not in page, f"riferimento esterno: {remote}"
    assert page.count("<script") == 2        # the data blob and the renderer


def test_the_page_says_the_live_computation_is_missing():
    """Hiding the button would make the offline copy quietly claim a capability
    it does not have."""
    page = sa.build_page(_ctx())
    assert "Non disponibile in questa copia" in page
    assert "Calcola ora" in page


def test_the_embedded_data_is_valid_json():
    page = sa.build_page(_ctx())
    blob = re.search(r"const D = (\{.*?\});</script>", page, re.S).group(1)
    assert json.loads(blob)["nominal_kn"] == 12.0


def test_the_footer_quotes_the_measured_scores_not_placeholders():
    page = sa.build_page(_ctx())
    assert "0.532" in page and "0.906" in page and "20 scene di test ufficiali" in page


def test_a_page_without_an_evaluation_states_nothing_rather_than_zero():
    ctx = _ctx() | {"aggregate": {}, "n_eval_scenes": None}
    page = sa.build_page(ctx)
    assert "mIoU 0.000" not in page and "0 scene di test" not in page


# ---------------------------------------------------------------------------
# Against the real build
# ---------------------------------------------------------------------------


def test_the_generated_page_carries_every_demo_route():
    from pathlib import Path

    page = Path("demo.html")
    index = Path("data/demo/index.json")
    if not (page.exists() and index.exists()):
        pytest.skip("demo.html o data/demo/ assenti")

    text = page.read_text()
    blob = json.loads(re.search(r"const D = (\{.*?\});</script>", text, re.S).group(1))
    stored = {s["scene_id"] for s in json.loads(index.read_text())["scenes"] if s.get("classes")}
    assert {s["id"] for s in blob["scenes"]} == stored

    for s in blob["scenes"]:
        for ice_class, e in s["classes"].items():
            if not e["route_exists"]:
                continue
            assert e["svg"], f"{s['id']}/{ice_class}: rotta senza SVG"
            assert 'class="lay-iceroute"' in e["svg"]
            assert e["transit"]["values"], f"{s['id']}/{ice_class}: nessuna tabella dei tempi"


# ---------------------------------------------------------------------------
# The link time is baked, like everything else
# ---------------------------------------------------------------------------


def test_the_offline_copy_carries_the_link_time_too():
    """Same rule as the transit table: computed in Python, chosen in the browser.
    The page cannot arrive at a transmission time of its own."""
    from pathlib import Path

    page = Path("demo.html")
    if not page.exists():
        pytest.skip("demo.html assente")
    blob = json.loads(re.search(r"const D = (\{.*?\});</script>",
                                page.read_text(), re.S).group(1))
    pays = [c["payload"] for s in blob["scenes"] for c in s["classes"].values()
            if c.get("payload")]
    assert pays, "nessun payload nella pagina"

    measured = [p for p in pays if p.get("total")]
    assert measured, "nessun payload con byte misurati"
    for p in measured:
        lk = p["link"]
        assert lk and lk["s"] > 0 and lk["kbps"] == 88, lk
        assert lk["sbd"] >= 1

    # The worst-case scene has no compression measurement — `make report` covers
    # the carousel only — so it has no bytes, and therefore must have no seconds.
    # A transmission time next to an unmeasured payload would be pure invention.
    for p in pays:
        if not p.get("total"):
            assert p["link"] is None, "tempo di link senza byte misurati"


def test_without_assumptions_the_page_shows_no_transmission_time():
    """A missing source must produce silence, not a default speed."""
    from pathlib import Path

    demo_dir = Path("data/demo")
    if not (demo_dir / "index.json").exists():
        pytest.skip("data/demo assente")
    ctx = sa.collect(demo_dir, {}, {})
    for s in ctx["scenes"]:
        for c in s["classes"].values():
            if c.get("payload"):
                assert c["payload"]["link"] is None
