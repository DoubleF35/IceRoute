"""Tests for the baseline route and waypoint simplification.

The one that protects the pitch is
``test_no_time_saving_is_claimed_against_an_impossible_route``: reporting hours
saved against a voyage POLARIS says cannot be made would be a fabricated
number, and it is exactly the sort of thing a due-diligence reader pulls on.
"""

from __future__ import annotations

import numpy as np
import pytest

from iceroute.routing.astar import Route, find_route
from iceroute.routing.baseline import compare, geodesic_baseline, geodesic_points
from iceroute.routing.cost import SpeedModel, build_cost_grid
from iceroute.routing.geocode import POLAR_CRS, GeoGrid, geodesic_distance_m
from iceroute.routing.simplify import (
    douglas_peucker,
    simplify_route,
    turn_angles_deg,
)

RES = 500.0


def make_grid(nrow: int, ncol: int) -> GeoGrid:
    return GeoGrid(
        x=np.arange(ncol) * RES,
        y=-2_000_000.0 + np.arange(nrow)[::-1] * RES,
        crs=POLAR_CRS,
        resolution_m=RES,
    )


# ---------------------------------------------------------------------------
# Geodesic sampling
# ---------------------------------------------------------------------------


def test_geodesic_points_include_the_endpoints():
    pts = geodesic_points(-20.0, 75.0, -10.0, 78.0, 50)
    assert len(pts) == 50
    assert pts[0] == pytest.approx((-20.0, 75.0))
    assert pts[-1] == pytest.approx((-10.0, 78.0))


def test_geodesic_samples_are_evenly_spaced():
    pts = geodesic_points(-20.0, 75.0, -10.0, 75.0, 21)
    seg = [geodesic_distance_m(*a, *b) for a, b in zip(pts, pts[1:], strict=False)]
    assert max(seg) / min(seg) < 1.05


def test_geodesic_rejects_degenerate_sampling():
    with pytest.raises(ValueError):
        geodesic_points(-20.0, 75.0, -10.0, 78.0, 1)


# ---------------------------------------------------------------------------
# Baseline
# ---------------------------------------------------------------------------


def test_open_water_baseline_is_feasible():
    n = 30
    grid = make_grid(n, n)
    cost = build_cost_grid(np.full((n, n), 30.0), SpeedModel("PC6", 12.0))
    res = geodesic_baseline(grid, cost, (5, 5), (25, 25), open_water_speed_kn=12.0)
    assert res.feasible
    assert res.n_no_go_cells == 0


def test_baseline_is_never_longer_than_the_routed_path():
    """The geodesic is the shortest path. If IceRoute came out shorter, the
    comparison would be measuring something else."""
    n = 40
    grid = make_grid(n, n)
    rio = np.full((n, n), 30.0)
    rio[18:22, 5:35] = -5.0
    cost = build_cost_grid(rio, SpeedModel("PC6", 12.0))

    route = find_route(cost, grid, (5, 20), (35, 20), rio=rio)
    base = geodesic_baseline(grid, cost, (5, 20), (35, 20), open_water_speed_kn=12.0)
    assert base.route.distance_km <= route.distance_km * 1.001


def test_baseline_reports_crossing_no_go_ice():
    n = 30
    grid = make_grid(n, n)
    rio = np.full((n, n), 30.0)
    rio[14:17, :] = -50.0
    cost = build_cost_grid(rio, SpeedModel("PC6", 12.0))

    res = geodesic_baseline(grid, cost, (5, 15), (25, 15), open_water_speed_kn=12.0, rio=rio)
    assert res.crosses_no_go
    assert res.n_no_go_cells > 0
    assert not res.feasible
    assert "POLARIS" in res.note


def test_infeasible_baseline_withholds_the_time_in_geojson():
    n = 30
    grid = make_grid(n, n)
    rio = np.full((n, n), 30.0)
    rio[14:17, :] = -50.0
    cost = build_cost_grid(rio, SpeedModel("PC6", 12.0))
    res = geodesic_baseline(grid, cost, (5, 15), (25, 15), open_water_speed_kn=12.0)
    assert res.as_geojson()["properties"]["transit_time_hours"] is None


def test_baseline_flags_land_crossing():
    n = 30
    grid = make_grid(n, n)
    land = np.zeros((n, n), dtype=bool)
    land[14:17, :] = True
    cost = build_cost_grid(np.full((n, n), 30.0), SpeedModel("PC6", 12.0), land_mask=land)
    res = geodesic_baseline(grid, cost, (5, 15), (25, 15), open_water_speed_kn=12.0, land_mask=land)
    assert res.crosses_land
    assert not res.feasible


def test_baseline_time_is_distance_over_open_water_speed():
    n = 30
    grid = make_grid(n, n)
    cost = build_cost_grid(np.full((n, n), 30.0), SpeedModel("PC6", 12.0))
    res = geodesic_baseline(grid, cost, (5, 5), (25, 25), open_water_speed_kn=12.0)
    expected = res.route.distance_km * 1000 / (12.0 * 0.514444) / 3600
    assert res.route.time_hours == pytest.approx(expected, rel=1e-6)


# ---------------------------------------------------------------------------
# Comparison — where a fabricated number would enter
# ---------------------------------------------------------------------------


def test_saving_is_reported_when_the_baseline_is_feasible():
    n = 41
    grid = make_grid(n, n)
    rio = np.full((n, n), 30.0)
    rio[18:23, 3:38] = -5.0
    cost = build_cost_grid(rio, SpeedModel("PC6", 12.0))

    route = find_route(cost, grid, (5, 20), (35, 20), rio=rio)
    base = geodesic_baseline(grid, cost, (5, 20), (35, 20), open_water_speed_kn=12.0, rio=rio)
    out = compare(route, base)

    assert out["baseline"]["feasible"]
    assert out["time_saved_hours"] is not None
    assert out["time_saved_percent"] is not None


def test_no_time_saving_is_claimed_against_an_impossible_route():
    """SPEC rule 2. You cannot save hours against a voyage that cannot be made.

    The honest claim in this case is stronger anyway: IceRoute finds a route
    where the naive one does not exist.
    """
    n = 41
    grid = make_grid(n, n)
    rio = np.full((n, n), 30.0)
    rio[18:23, 5:38] = -50.0  # a no-go wall with a gap on the left
    cost = build_cost_grid(rio, SpeedModel("PC6", 12.0))

    route = find_route(cost, grid, (5, 20), (35, 20), rio=rio)
    base = geodesic_baseline(grid, cost, (5, 20), (35, 20), open_water_speed_kn=12.0, rio=rio)
    out = compare(route, base)

    assert not out["baseline"]["feasible"]
    assert out["time_saved_hours"] is None
    assert out["time_saved_percent"] is None
    assert out["baseline"]["time_hours"] is None
    assert "non è percorribile" in out["headline"]


def test_comparison_reports_the_detour():
    n = 41
    grid = make_grid(n, n)
    rio = np.full((n, n), 30.0)
    rio[18:23, 3:38] = -5.0
    cost = build_cost_grid(rio, SpeedModel("PC6", 12.0))
    route = find_route(cost, grid, (5, 20), (35, 20), rio=rio)
    base = geodesic_baseline(grid, cost, (5, 20), (35, 20), open_water_speed_kn=12.0)
    assert compare(route, base)["detour_km"] > 0


# ---------------------------------------------------------------------------
# Simplification
# ---------------------------------------------------------------------------


def test_douglas_peucker_keeps_only_the_endpoints_of_a_straight_line():
    pts = np.array([[float(i) * 500, 0.0] for i in range(50)])
    assert douglas_peucker(pts, 100.0) == [0, 49]


def test_douglas_peucker_keeps_a_real_corner():
    pts = np.array([[0.0, 0.0], [5000.0, 0.0], [5000.0, 5000.0]])
    assert douglas_peucker(pts, 100.0) == [0, 1, 2]


def test_douglas_peucker_never_drops_the_endpoints():
    rng = np.random.default_rng(0)
    pts = rng.normal(scale=5000, size=(200, 2))
    idx = douglas_peucker(pts, 10_000.0)
    assert idx[0] == 0
    assert idx[-1] == len(pts) - 1


def test_douglas_peucker_handles_short_inputs():
    assert douglas_peucker(np.zeros((1, 2)), 100.0) == [0]
    assert douglas_peucker(np.zeros((2, 2)), 100.0) == [0, 1]


def test_turn_angles_are_zero_on_a_straight_line():
    pts = np.array([[0.0, 0.0], [1000.0, 0.0], [2000.0, 0.0]])
    assert turn_angles_deg(pts) == pytest.approx([0.0], abs=1e-9)


def test_turn_angle_of_a_right_angle():
    pts = np.array([[0.0, 0.0], [1000.0, 0.0], [1000.0, 1000.0]])
    assert turn_angles_deg(pts) == pytest.approx([90.0], abs=1e-6)


def _long_route(grid, n=60):
    cells = [(5 + i // 2, 5 + i) for i in range(n)]
    return Route(
        cells=cells,
        lonlat=[grid.lonlat_of(r, c) for r, c in cells],
        time_hours=1.0,
        distance_km=10.0,
        ice_class="PC6",
    )


def test_simplification_respects_the_waypoint_cap():
    """SPEC §8.5: at most 40 waypoints."""
    grid = make_grid(80, 80)
    out = simplify_route(_long_route(grid, 70), grid, tolerance_m=1.0, max_waypoints=40)
    assert len(out) <= 40


def test_simplification_preserves_the_endpoints():
    """Truncating a route to fit a cap would move its destination."""
    grid = make_grid(80, 80)
    route = _long_route(grid, 70)
    out = simplify_route(route, grid, tolerance_m=2000.0)
    assert out.lonlat[0] == pytest.approx(route.lonlat[0])
    assert out.lonlat[-1] == pytest.approx(route.lonlat[-1])
    assert not any("arrivo" in w for w in out.warnings)


def test_simplification_reduces_a_straight_route_to_two_points():
    grid = make_grid(80, 80)
    cells = [(10, 5 + i) for i in range(60)]
    route = Route(cells, [grid.lonlat_of(r, c) for r, c in cells], 1.0, 10.0, "PC6")
    assert len(simplify_route(route, grid, tolerance_m=1000.0)) == 2


def test_simplification_reports_deviation_and_sharp_turns():
    grid = make_grid(80, 80)
    cells = [(10, 5 + i) for i in range(20)] + [(10 + i, 24) for i in range(1, 20)]
    route = Route(cells, [grid.lonlat_of(r, c) for r, c in cells], 1.0, 10.0, "PC6")
    out = simplify_route(route, grid, tolerance_m=500.0, max_turn_deg=45.0)
    assert out.max_turn_deg > 45.0
    assert out.sharp_turns
    assert any("virate" in w for w in out.warnings)


def test_simplified_geojson_is_wellformed():
    grid = make_grid(80, 80)
    gj = simplify_route(_long_route(grid, 40), grid).as_geojson({"kind": "iceroute"})
    assert gj["geometry"]["type"] == "LineString"
    assert gj["properties"]["kind"] == "iceroute"
    assert gj["properties"]["n_original_cells"] == 40
