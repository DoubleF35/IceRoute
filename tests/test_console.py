"""Tests for the console rendering of the demo.

This page puts geography on screen — footprints, a pass direction, a satellite
on a ground track — and geography is the easiest thing on which to be
confidently wrong. So the tests care about one property above all: that
everything drawn at a coordinate was *read* from a scene, and that the parts
with no data behind them are absent rather than approximated.
"""

from __future__ import annotations

import html
import json
import math
import re
from pathlib import Path

import pytest

from iceroute.pitch import console as co

# ---------------------------------------------------------------------------
# Bearings and distances
# ---------------------------------------------------------------------------


def test_a_bearing_due_north_is_zero_and_due_east_is_ninety():
    assert co.azimuth((0.0, 0.0), (0.0, 10.0)) == pytest.approx(0.0, abs=0.01)
    assert co.azimuth((0.0, 0.0), (10.0, 0.0)) == pytest.approx(90.0, abs=0.01)
    assert co.azimuth((0.0, 10.0), (0.0, 0.0)) == pytest.approx(180.0, abs=0.01)


def test_a_degree_of_latitude_is_about_111_km():
    assert co.great_circle_km((0.0, 0.0), (0.0, 1.0)) == pytest.approx(111.2, abs=0.5)


def test_the_acquisition_timestamp_is_reformatted_not_reinterpreted():
    assert co._stamp("20180707T113313") == "2018-07-07 11:33:13Z"


def test_a_product_name_yields_the_platform_and_the_absolute_orbit():
    m = co.PRODUCT.match(
        "S1B_EW_GRDM_1SDH_20180707T113313_20180707T113413_011704_015877_83D1")
    assert m and m.group("platform") == "S1B" and int(m.group("orbit")) == 11704
    assert m.group("mode") == "EW" and m.group("pol") == "1SDH"


def test_a_name_that_is_not_a_sentinel_product_matches_nothing():
    """Then the product panel is empty, which is the right answer: inventing a
    platform for a file that does not name one is the failure this page is
    most exposed to."""
    assert co.PRODUCT.match("qualcosa_altro.nc") is None


# ---------------------------------------------------------------------------
# The forbidden-cell trace
# ---------------------------------------------------------------------------


def _overlay(n=100, no_go=(), no_data=()):
    return {"baseline": {"path": [[0, i] for i in range(n)],
                         "no_go": [[0, i] for i in no_go],
                         "no_data": [[0, i] for i in no_data]}}


def test_the_trace_counts_the_cells_the_overlay_actually_marks():
    a = co.admissibility(_overlay(100, no_go=range(10, 20)), buckets=10)
    assert a["n_cells"] == 100 and a["n_no_go"] == 10
    assert a["bars"][1] == [1.0, 0.0], "il secondo decimo è tutto vietato"
    assert a["bars"][0] == [0.0, 0.0]


def test_a_route_with_nothing_forbidden_draws_a_flat_trace():
    a = co.admissibility(_overlay(60), buckets=12)
    assert a["n_no_go"] == 0 and all(b == [0.0, 0.0] for b in a["bars"])


def test_no_baseline_path_means_no_trace_rather_than_an_empty_chart():
    assert co.admissibility({"baseline": {}}) is None


def test_cells_outside_the_chart_are_a_separate_series_from_forbidden_ones():
    """"Fuori copertura" and "vietata da POLARIS" are different statements —
    one is missing data, the other is a decision — and the page must not blend
    them into one bar."""
    a = co.admissibility(_overlay(20, no_go=(0,), no_data=(10,)), buckets=20)
    assert a["bars"][0] == [1.0, 0.0] and a["bars"][10] == [0.0, 1.0]
    assert a["n_no_go"] == 1 and a["n_no_data"] == 1


# ---------------------------------------------------------------------------
# Nothing is computed in the browser
# ---------------------------------------------------------------------------


_PAL = {"rio": ["#2ecc71", "#f1c40f", "#e74c3c"],
        "classes": ["#0a285a", "#00c8ff", "#ffffff"]}


def _entry():
    return {
        "route_exists": True,
        "polaris": {"rio_min": 0.0, "rio_max": 30.0},
        "conditions": {"n_cells": 449, "fraction_normal": 1.0, "fraction_elevated": 0.0,
                       "fraction_no_go": 0.0, "rio_median": 26.0,
                       "by_class": {"0": 98, "1": 55, "2": 296}},
    }


def test_bar_widths_are_resolved_in_python_as_percentages():
    """A browser that turns fraction_no_go into a width is a browser deciding
    how big a risk looks."""
    p = co.panels(_entry(), None, None, None, _PAL)
    assert p["rio_meter"]["segs"] == [[100.0, "#2ecc71"], [0.0, "#f1c40f"],
                                      [0.0, "#e74c3c"]]
    assert sum(w for w, _ in p["mix"]["segs"]) == pytest.approx(100.0, abs=0.1)


def test_a_class_mix_uses_the_palette_the_masks_were_painted_with():
    p = co.panels(_entry(), None, None, None, _PAL)
    assert [c for _, c in p["mix"]["segs"]] == _PAL["classes"]


def test_a_scene_without_a_route_gets_no_meter_instead_of_an_empty_one():
    p = co.panels({"route_exists": False}, None, None, None, _PAL)
    assert p["rio_meter"] is None and p["mix"] is None and p["trace"] is None


def test_a_scene_without_its_raw_file_gets_no_product_rows():
    p = co.panels(_entry(), None, None, None, _PAL)
    assert p["prod"] == [], "senza NetCDF grezzo non c'è un prodotto da dichiarare"


def test_numbers_are_grouped_and_pointed_in_italian():
    assert co.it(169558) == f"169{co.NBSP}558"
    assert co.it(2.604, 3) == "2,604"
    assert " " not in co.it(169558), "il raggruppamento non deve poter andare a capo"


# ---------------------------------------------------------------------------
# The log states facts, and only facts it has
# ---------------------------------------------------------------------------


def test_the_log_carries_no_timestamps_it_does_not_have():
    """A console log reads as a clock ticking. Nothing in data/demo records
    when a stage ran, so the sequence numbers are the honest version; a
    plausible [23:07:21] would not be."""
    rows = co.event_log({"id": "S.nc", "shape": [10, 20], "warnings": []}, "PC6",
                        _entry(), None, None, {}, None)
    joined = " ".join(m for _, _, m in rows)
    assert ":" not in joined.replace("11:33", "") or "Z" in joined
    assert [n for n, _, _ in rows] == [f"{i:02d}" for i in range(len(rows))]


def test_the_log_omits_the_stages_whose_measurements_are_missing():
    rows = co.event_log({"id": "S.nc", "shape": None, "warnings": []}, "PC6",
                        {"route_exists": False}, None, None, {}, None)
    tags = {t for _, t, _ in rows}
    assert "TILE" not in tags and "PRODOTTO" not in tags and "A*" not in tags


def test_a_scene_warning_reaches_the_log():
    rows = co.event_log({"id": "S.nc", "shape": None, "warnings": ["mancano dati"]},
                        "PC6", {"route_exists": False}, None, None, {}, None)
    assert any(t == "AVVISO" and "mancano dati" in m for _, t, m in rows)


# ---------------------------------------------------------------------------
# Against the real build
# ---------------------------------------------------------------------------


def _real_ctx():
    demo = Path("data/demo")
    if not (demo / "index.json").exists():
        pytest.skip("data/demo assente")
    import yaml
    metrics = (json.loads(Path("artifacts/metrics.json").read_text())
               if Path("artifacts/metrics.json").exists() else {})
    assumptions = (yaml.safe_load(Path("configs/assumptions.yaml").read_text())
                   if Path("configs/assumptions.yaml").exists() else {})
    return co.collect(demo, metrics, assumptions, Path("data/raw/test"))


def test_every_footprint_sits_where_its_route_does():
    """The outline comes from the scene's geolocation grid and the route from
    the transmitted GeoJSON — two different files. If they disagree, one of
    them is being drawn in the wrong place."""
    ctx = _real_ctx()
    checked = 0
    for s in ctx["scenes"]:
        f = s.get("footprint")
        if not f:
            continue
        lons = [p[0] for p in f["ring"]]
        lats = [p[1] for p in f["ring"]]
        for e in s["classes"].values():
            for lon, lat in e.get("route_lonlat") or []:
                assert min(lons) - 0.5 <= lon <= max(lons) + 0.5, f"{s['id']}: lon fuori"
                assert min(lats) - 0.5 <= lat <= max(lats) + 0.5, f"{s['id']}: lat fuori"
                checked += 1
    assert checked, "nessuna rotta confrontata con la propria impronta"


def test_every_pass_is_near_polar():
    """Sentinel-1 flies a sun-synchronous orbit, so an along-track bearing far
    from 190 or 10 degrees means the wrong grid axis was read — which is
    exactly the mistake the dimension names in these files invite."""
    ctx = _real_ctx()
    for s in ctx["scenes"]:
        f = s.get("footprint")
        if not f:
            continue
        az = f["track_az"]
        off = min(abs(az - 190), abs(az - 10), abs(az - 370))
        assert off < 25, f"{s['id']}: traccia {az}°, non è un passaggio quasi polare"


def test_the_look_direction_is_square_to_the_track():
    """Sentinel-1 is right-looking: range sits about ninety degrees off the
    track. This is what distinguishes the two axes, and it is measured."""
    ctx = _real_ctx()
    for s in ctx["scenes"]:
        f = s.get("footprint")
        if not f:
            continue
        d = (f["look_az"] - f["track_az"]) % 360
        assert 75 < d < 105, f"{s['id']}: vista a {d}° dalla traccia"


def test_the_footprint_matches_the_pixel_grid_it_came_from():
    """Extent measured on the geolocation grid against extent implied by the
    raster size and the pixel spacing. Two independent routes to the same
    number; if they part company, one of them is wrong."""
    ctx = _real_ctx()
    for s in ctx["scenes"]:
        f, shape = s.get("footprint"), s.get("shape")
        if not (f and shape and f.get("pixel_spacing_m")):
            continue
        along = shape[0] * f["pixel_spacing_m"] / 1000
        across = shape[1] * f["pixel_spacing_m"] / 1000
        assert math.isclose(f["along_km"], along, rel_tol=0.05), f"{s['id']} lungo traccia"
        assert math.isclose(f["across_km"], across, rel_tol=0.05), f"{s['id']} in range"


def test_the_page_is_self_contained_and_declares_its_charset():
    ctx = _real_ctx()
    page = co.build_page(ctx)
    assert '<meta charset="utf-8">' in page[:120]
    for remote in ("http://", "https://", "//cdn", "<link"):
        assert remote not in page, f"riferimento esterno: {remote}"


def test_the_page_says_out_loud_that_it_has_no_coastlines():
    """The one absence a map must admit to: a viewer who assumes a basemap is
    there will read the empty sphere as ocean."""
    page = co.build_page(_real_ctx())
    assert "Nessuna linea" in page and "costa" in page


# ---------------------------------------------------------------------------
# What arrives on the bridge
# ---------------------------------------------------------------------------


_ROUTE = ('{"type":"FeatureCollection","features":[{"type":"Feature","geometry":'
          '{"type":"LineString","coordinates":[[-76.567482,56.331759],'
          '[-76.690202,56.200049]]},"properties":{"ice_class":"PC6"}}]}')


def _paid(text=None, route=None, **extra):
    body = text if text is not None else _ROUTE
    import textwrap
    return {"payload": {"text": textwrap.fill(body, width=78),
                        "route": route if route is not None else len(body.encode()),
                        "n_waypoints": 2, **extra}}


def test_the_dump_is_the_transmitted_file_and_nothing_else():
    """Concatenating the chunks has to give back the exact bytes, because the
    byte count printed beside them is a claim about this text."""
    b = co.bridge(_paid())
    joined = "".join(re.sub(r"<[^>]+>", "", h) for _, h in b["lines"])
    assert html.unescape(joined) == _ROUTE
    assert b["bytes"] == len(_ROUTE.encode())
    assert json.loads(html.unescape(joined))["type"] == "FeatureCollection"


def test_the_offsets_are_byte_offsets():
    b = co.bridge(_paid(), width=64)
    assert [int(o, 16) for o, _ in b["lines"]] == list(range(0, len(_ROUTE), 64))
    assert b["ruler"] is True


def test_a_byte_count_that_disagrees_refuses_to_draw_a_dump():
    """A dump that contradicts the number printed next to it is worse than no
    dump: both look authoritative and one of them is wrong."""
    b = co.bridge(_paid(route=999))
    assert b["mismatch"] is True and b["declared"] == 999
    assert "lines" not in b


def test_no_route_means_no_packet_rather_than_an_empty_frame():
    assert co.bridge({}) is None
    assert co.bridge({"payload": {"text": ""}}) is None


def test_a_non_ascii_payload_drops_the_ruler_instead_of_misnumbering_it():
    """Character offsets stop being byte offsets the moment a multi-byte
    character appears, and a ruler that is quietly wrong is the kind of detail
    nobody checks."""
    b = co.bridge(_paid(text='{"nome":"Rotta nord-est — Grønland"}'))
    assert b["ruler"] is False and all(o == "" for o, _ in b["lines"])


def test_the_colouring_marks_strings_and_numbers_and_escapes_the_rest():
    out = co.tint('{"a":-76.5,"b":"<x>"}')
    assert '<i class="n">-76.5</i>' in out
    assert '<i class="s">&quot;a&quot;</i>' in out
    assert "&lt;x&gt;" in out and "<x>" not in out


def test_the_byte_rows_quote_only_what_was_measured():
    b = co.bridge(_paid(mask=169558, meta=60, total=170186, ratio=324.7,
                        sar=55_300_000, link={"s": 15, "terminal": "Iridium Certus 100",
                                              "kbps": 88, "sbd": 2}))
    labels = {k for k, _, _ in b["rows"]}
    assert "maschera del ghiaccio" in labels and "compressione" in labels
    assert "in messaggi SBD" in labels
    b2 = co.bridge(_paid())
    assert {k for k, _, _ in b2["rows"]} == {"rotta · 2 waypoint"}, \
        "senza misure di compressione non si inventano righe"


def test_every_real_packet_parses_as_the_file_it_claims_to_be():
    """Against the build: for every scene and vessel class, the text on screen
    is valid GeoJSON and weighs exactly what the panel says it weighs."""
    ctx = _real_ctx()
    checked = 0
    for s in ctx["scenes"]:
        for ic, e in s["classes"].items():
            b = (e.get("panels") or {}).get("bridge")
            if not b:
                assert not e.get("route_exists"), f"{s['id']}/{ic}: rotta senza pacchetto"
                continue
            assert not b.get("mismatch"), f"{s['id']}/{ic}: byte incoerenti"
            joined = html.unescape(
                "".join(re.sub(r"<[^>]+>", "", h) for _, h in b["lines"]))
            assert len(joined.encode()) == b["bytes"], f"{s['id']}/{ic}: byte diversi"
            assert json.loads(joined)["type"] == "FeatureCollection"
            assert e["payload"]["route"] == b["bytes"], f"{s['id']}/{ic}: conteggio diverso"
            checked += 1
    assert checked, "nessun pacchetto verificato"


# ---------------------------------------------------------------------------
# The two grids, made to coincide
# ---------------------------------------------------------------------------


def test_the_geocoded_sar_lands_on_the_same_raster_as_the_mask():
    """The whole point of the layer: SAR, mask, concentration, RIO and the
    routes on one grid, so switching layers stops moving the scene."""
    from PIL import Image

    ctx = _real_ctx()
    checked = 0
    for s in ctx["scenes"]:
        files = s.get("files") or {}
        if "sar_geo" not in files:
            continue
        geo = Image.open(Path("data/demo/images") / files["mask_geo"]).size
        sar = Image.open(co.CACHE_DIR / files["sar_geo"]).size
        assert sar == geo, f"{s['id']}: SAR {sar} contro maschera {geo}"
        checked += 1
    if not checked:
        pytest.skip("nessun SAR geocodificato in questa build")


def test_a_grid_that_does_not_match_the_overlay_produces_no_layer(tmp_path):
    """Refusing is the point. A layer one cell out would put the route in the
    wrong place while looking perfectly convincing — D-027, again."""
    raw = Path("data/raw/test/20180707T113313_cis_prep.nc")
    if not raw.is_file():
        pytest.skip("scena grezza assente")
    assert co.sar_geocoded(raw, [10, 10], 500.0, tmp_path) is None
    assert not list(tmp_path.glob("*.png")), "non deve lasciare un raster inutilizzabile"


def test_a_cached_raster_of_the_wrong_shape_is_rebuilt_not_served(tmp_path):
    """Presence is not correctness: a PNG left from another resolution would
    otherwise sail past the shape check it was meant to fail."""
    from PIL import Image

    raw = Path("data/raw/test/20180707T113313_cis_prep.nc")
    stale = tmp_path / f"{raw.stem}_sargeo.png"
    Image.new("L", (7, 9)).save(stale)
    assert co.sar_geocoded(raw, [10, 10], 500.0, tmp_path) is None
    assert not stale.exists(), "la cache obsoleta va rimossa, non restituita"


def test_the_layer_is_offered_only_when_a_scene_actually_has_it():
    ctx = _real_ctx()
    ids = [x[0] for x in ctx["layers"]]
    has = any("sar_geo" in (s.get("files") or {}) for s in ctx["scenes"])
    assert ("sar_geo" in ids) == has
    if has:
        assert ids.index("sar_geo") < ids.index("sar"), \
            "il livello geocodificato sta accanto a quello a swath, per il confronto"
