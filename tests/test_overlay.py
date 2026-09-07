"""Tests for the route overlay.

The bug these exist to prevent is not a crash. For weeks the deck's route
figure and the demo both drew *nothing*: the drawing code looked up a property
nobody wrote, skipped every track, saved a clean PNG and reported success. A
figure captioned "the direct route is forbidden" shipped with no route on it.

So the first test here asserts the only thing that would have caught that —
that pixels actually change — and the second asserts that the mistake which
made it plausible (two grids, one of them wrong) now raises instead of
silently producing a picture.
"""

from __future__ import annotations

import json

import numpy as np
import pytest
from PIL import Image

from iceroute.demo import overlay as ov

pytest.importorskip("PIL")


# A 40x60 grid: a diagonal "our route" and a straight "direct route" across it.
GRID = [40, 60]


def _overlay(no_go=None, waypoints=None) -> dict:
    return {
        "grid_shape": GRID,
        "resolution_m": 500.0,
        "iceroute": {
            "path": [[r, r] for r in range(2, 38)],
            "waypoints": waypoints if waypoints is not None else [[2, 2], [37, 37]],
        },
        "baseline": {
            "path": [[20, c] for c in range(2, 58)],
            "no_go": no_go or [],
            "no_data": [],
            "feasible": not no_go,
        },
        "start": [2, 2],
        "goal": [37, 37],
    }


def _blank(w=60, h=40) -> Image.Image:
    return Image.new("RGB", (w, h), (0, 0, 0))


def _px(img: Image.Image) -> np.ndarray:
    return np.asarray(img.convert("RGB"))


def _changed(before: Image.Image, after: Image.Image) -> int:
    return int((_px(before) != _px(after)).any(-1).sum())


def _has_colour(img: Image.Image, rgb: tuple) -> bool:
    return bool((_px(img) == np.array(rgb)).all(-1).any())


# ---------------------------------------------------------------------------
# The regression that motivated the module
# ---------------------------------------------------------------------------


def test_drawing_a_route_actually_changes_pixels():
    """The test that was missing. Everything else here is detail."""
    img = _blank()
    out = ov.draw(img, _overlay(), show="both")
    assert _changed(img, out) > 50, "la rotta non è stata disegnata"


def test_nothing_is_drawn_when_nothing_is_asked_for():
    img = _blank()
    assert _changed(img, ov.draw(img, _overlay(), show="none")) == 0
    assert _changed(img, ov.draw(img, {}, show="both")) == 0


@pytest.mark.parametrize("mode", ["baseline", "iceroute", "both"])
def test_every_mode_draws_something_and_they_differ(mode):
    img = _blank()
    drawn = {m: ov.draw(img, _overlay(), show=m) for m in ("baseline", "iceroute", "both")}
    assert _changed(img, drawn[mode]) > 20
    assert _changed(drawn["baseline"], drawn["iceroute"]) > 0
    assert _changed(drawn["both"], drawn["baseline"]) > 0


# ---------------------------------------------------------------------------
# The two-grid trap
# ---------------------------------------------------------------------------


def test_the_wrong_grid_raises_instead_of_drawing_nothing():
    """A route belongs to the geocoded grid. Paired with the raw swath it is
    meaningless, and the old code's answer was to quietly draw nothing."""
    with pytest.raises(ov.GridMismatch, match="incompatibile"):
        ov.draw(_blank(90, 40), _overlay(), show="both")


def test_an_overlay_without_a_declared_grid_is_refused():
    bad = _overlay()
    del bad["grid_shape"]
    with pytest.raises(ov.GridMismatch):
        ov.draw(_blank(), bad, show="both")


def test_a_proportionally_decimated_image_is_still_drawn():
    """Halving both axes is legitimate: the check is that they agree, not that
    the scale is 1."""
    img = _blank(30, 20)
    assert _changed(img, ov.draw(img, _overlay(), show="both")) > 20


def test_the_tolerance_is_tighter_than_the_mismatch_it_must_catch():
    """The real raw-vs-geocoded pairing differs by 2.8%. A tolerance set just
    under that would pass the next scene whose shapes happen to be closer."""
    ov.draw(_blank(60, 40), _overlay(), show="both")           # exact: fine
    with pytest.raises(ov.GridMismatch):
        ov.draw(_blank(61, 40), _overlay(), show="both")       # 1.7%: refused


# ---------------------------------------------------------------------------
# Forbidden cells
# ---------------------------------------------------------------------------


def test_forbidden_cells_become_contiguous_runs():
    cells = [[20, c] for c in range(10, 20)]
    assert [len(r) for r in ov.no_go_runs(cells)] == [10]


def test_a_gap_starts_a_new_run():
    cells = [[20, 10], [20, 11], [20, 40], [20, 41], [20, 42]]
    assert [len(r) for r in ov.no_go_runs(cells)] == [2, 3]


def test_no_forbidden_cells_is_no_runs():
    assert ov.no_go_runs([]) == []


def test_forbidden_cells_are_painted_over_the_direct_route():
    """The evidence has to be visible, not merely counted in the index."""
    plain = ov.draw(_blank(), _overlay(), show="baseline")
    marked = ov.draw(_blank(), _overlay(no_go=[[20, c] for c in range(20, 40)]),
                     show="baseline")
    assert _changed(plain, marked) > 10
    assert _has_colour(marked, ov.COL_NO_GO), "il tratto vietato non è rosso"


# ---------------------------------------------------------------------------
# Separation — the measurement that picks the deck's lead scene
# ---------------------------------------------------------------------------


class _Grid:
    shape = (40, 60)
    resolution_m = 500.0


def test_identical_tracks_have_no_separation():
    path = [[r, r] for r in range(10)]
    assert ov.separation(path, path, _Grid())["max_px"] == 0.0


def test_separation_grows_with_the_detour():
    base = [[20, c] for c in range(60)]
    near = ov.separation([[18, c] for c in range(60)], base, _Grid())
    far = ov.separation([[2, c] for c in range(60)], base, _Grid())
    assert far["max_px"] > near["max_px"]
    assert far["max_km"] == pytest.approx(far["max_px"] * 0.5)
    assert 0 < far["pct_diagonal"] < 100


def test_separation_of_an_empty_track_is_not_a_crash():
    assert ov.separation([], [[1, 1]], _Grid()) == {}


# ---------------------------------------------------------------------------
# The downlink product must not grow
# ---------------------------------------------------------------------------


def test_the_geojson_does_not_carry_the_per_cell_lists():
    """The compression ratio is a measured claim about this file's size.

    Putting the drawable geometry in here would have been the obvious place for
    it, and would have inflated the payload by two orders of magnitude to carry
    something only the ground UI reads.
    """
    from iceroute.routing.astar import Route
    from iceroute.routing.baseline import BaselineResult

    route = Route(cells=[(0, 0), (1, 1)], lonlat=[(0.0, 70.0), (0.1, 70.1)],
                  time_hours=1.0, distance_km=10.0, ice_class="PC6")
    res = BaselineResult(route=route, crosses_no_go=True, n_no_go_cells=2,
                         n_slow_cells=0, crosses_land=False, feasible=False,
                         no_go_cells=[(0, 0), (1, 1)], no_data_cells=[(2, 2)])

    props = res.as_geojson()["properties"]
    assert props["n_no_go_cells"] == 2, "il conteggio serve e deve restare"
    for banned in ("no_go_cells", "no_data_cells", "cells"):
        assert banned not in props, f"{banned} gonfia il prodotto del downlink"


def test_the_baseline_still_counts_what_it_now_also_lists():
    """The counts and the cell lists are two views of one thing; if they ever
    disagree, one of the two is lying to the UI."""
    from iceroute.routing.astar import Route
    from iceroute.routing.baseline import BaselineResult

    route = Route(cells=[(0, 0)], lonlat=[(0.0, 70.0)], time_hours=1.0,
                  distance_km=1.0, ice_class="IA")
    res = BaselineResult(route=route, crosses_no_go=True, n_no_go_cells=3,
                         n_slow_cells=0, crosses_land=False, feasible=False,
                         n_no_data_cells=1,
                         no_go_cells=[(0, 0), (0, 1), (0, 2)], no_data_cells=[(5, 5)])
    assert len(res.no_go_cells) == res.n_no_go_cells
    assert len(res.no_data_cells) == res.n_no_data_cells


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def test_a_missing_overlay_loads_as_empty(tmp_path):
    assert ov.load(tmp_path / "assente.json") == {}


def test_loading_a_directory_is_empty_not_an_exception(tmp_path):
    """Callers build this path from an index entry that may have no overlay
    name, which lands on the directory itself."""
    assert ov.load(tmp_path) == {}


def test_a_written_overlay_round_trips(tmp_path):
    p = tmp_path / "o.json"
    p.write_text(json.dumps(_overlay()))
    assert ov.load(p)["grid_shape"] == GRID


# ---------------------------------------------------------------------------
# Against the real build
# ---------------------------------------------------------------------------


def test_every_route_in_the_demo_can_actually_be_drawn():
    """End to end, on what the demo will really open.

    The unit tests above all use a synthetic overlay, which is exactly the
    blind spot that let the original bug live: the code was self-consistent and
    disagreed with the data. This pairs each real overlay with the real raster
    the UI resolves for it and insists that pixels change.
    """
    from pathlib import Path

    index_path = Path("data/demo/index.json")
    if not index_path.exists():
        pytest.skip("precompute non ancora eseguito")
    index = json.loads(index_path.read_text())

    checked = 0
    for scene in index.get("scenes", []):
        geo_png = Path("data/demo/images") / (scene.get("files") or {}).get("mask_geo", "")
        if not geo_png.is_file():
            pytest.fail(f"{scene['scene_id']}: manca la maschera geocodificata, "
                        f"nessuna rotta è disegnabile")
        for ice_class, entry in (scene.get("classes") or {}).items():
            if not entry.get("route_exists"):
                continue
            name = entry.get("overlay")
            assert name, f"{scene['scene_id']}/{ice_class}: rotta senza overlay"
            data = ov.load(Path("data/demo/overlays") / name)
            assert data, f"{name} assente o vuoto"

            img = Image.open(geo_png)
            assert _changed(img, ov.draw(img, data, show="both")) > 100
            checked += 1

    assert checked, "nessuna rotta verificata: la demo non ha rotte"
