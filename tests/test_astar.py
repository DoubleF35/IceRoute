"""Routing tests: cost model and A*.

The two that matter most for credibility:

* ``test_16_connectivity_beats_8_on_a_diagonal`` — quantifies the digitisation
  error the spec warns about, on a case where the true answer is known exactly.
* ``test_long_step_cannot_jump_a_one_cell_barrier`` — a knight move that
  tunnels through a coastline produces a route that is optimal on the grid and
  impossible at sea.
"""

from __future__ import annotations

import numpy as np
import pytest

from iceroute.routing.astar import (
    NEIGHBOURS_8,
    NEIGHBOURS_16,
    NoRouteFound,
    _intermediate_cells,
    find_route,
)
from iceroute.routing.cost import (
    KNOT_TO_MS,
    SpeedModel,
    build_cost_grid,
    build_land_mask,
)
from iceroute.routing.geocode import POLAR_CRS, GeoGrid, geodesic_distance_m

RES = 500.0


def make_grid(nrow: int, ncol: int, res: float = RES) -> GeoGrid:
    """A grid near the standard parallel, so the scale factor is ~1 and the
    expected distances are easy to reason about by hand."""
    x0, y0 = 0.0, -2_000_000.0  # roughly 70N on the -45 meridian
    return GeoGrid(
        x=x0 + np.arange(ncol) * res,
        y=y0 + np.arange(nrow)[::-1] * res,
        crs=POLAR_CRS,
        resolution_m=res,
    )


def open_water_cost(nrow, ncol, speed_kn=12.0, ice_class="PC6"):
    rio = np.full((nrow, ncol), 30.0)
    return build_cost_grid(rio, SpeedModel(ice_class, speed_kn))


# ---------------------------------------------------------------------------
# Speed model
# ---------------------------------------------------------------------------


def test_open_water_gets_full_speed():
    sm = SpeedModel("PC6", 12.0)
    assert sm.speed_kn(np.array([30.0]))[0] == 12.0


def test_elevated_band_gets_the_table_1_2_cap():
    """PC6 in the elevated band is capped at 3 knots by the circular."""
    sm = SpeedModel("PC6", 12.0)
    assert sm.speed_kn(np.array([-5.0]))[0] == 3.0


def test_below_no_go_threshold_is_impassable():
    sm = SpeedModel("PC6", 12.0)
    assert sm.speed_kn(np.array([-15.0]))[0] == 0.0


def test_below_polar_class_has_no_elevated_band():
    """For IA the middle band is already 'special consideration': no transit."""
    sm = SpeedModel("IA", 12.0)
    assert sm.speed_kn(np.array([-5.0]))[0] == 0.0
    assert sm.elevated_cap_kn is None


def test_cap_never_exceeds_the_vessel_speed():
    """A slow ship is not sped up by a regulatory cap above its cruising speed."""
    sm = SpeedModel("PC1", 5.0)  # PC1 cap is 11 kn
    assert sm.elevated_cap_kn == 5.0


def test_unknown_cells_are_impassable():
    sm = SpeedModel("PC6", 12.0)
    assert sm.speed_kn(np.array([np.nan]))[0] == 0.0


def test_linear_profile_ramps_between_the_bounds():
    sm = SpeedModel("PC6", 12.0, band_profile="linear")
    at_zero = sm.speed_kn(np.array([-0.001]))[0]
    at_edge = sm.speed_kn(np.array([-9.999]))[0]
    assert at_edge < at_zero
    assert 3.0 <= at_edge <= 12.0


def test_rejects_bad_configuration():
    with pytest.raises(ValueError):
        SpeedModel("PC6", 0.0)
    with pytest.raises(ValueError, match="band_profile"):
        SpeedModel("PC6", 12.0, band_profile="quadratic")


# ---------------------------------------------------------------------------
# Land mask
# ---------------------------------------------------------------------------


def test_land_mask_from_distance_minimum():
    d = np.array([[0.0, 1.0], [2.0, 3.0]])
    assert build_land_mask(d, (2, 2)).tolist() == [[True, False], [False, False]]


def test_land_mask_buffer_expands():
    d = np.ones((5, 5))
    d[2, 2] = 0.0
    assert build_land_mask(d, (5, 5), buffer_cells=1).sum() == 9


def test_no_distance_layer_means_no_land():
    assert not build_land_mask(None, (3, 3)).any()


def test_land_blocks_the_cell():
    rio = np.full((3, 3), 30.0)
    land = np.zeros((3, 3), dtype=bool)
    land[1, 1] = True
    cg = build_cost_grid(rio, SpeedModel("PC6", 12.0), land_mask=land)
    assert not cg.passable[1, 1]


# ---------------------------------------------------------------------------
# Connectivity
# ---------------------------------------------------------------------------


def test_neighbour_sets_are_the_right_size():
    assert len(NEIGHBOURS_8) == 8
    assert len(NEIGHBOURS_16) == 16
    assert len(set(NEIGHBOURS_16)) == 16


def test_short_steps_have_no_intermediates():
    for dr, dc in NEIGHBOURS_8:
        assert _intermediate_cells(dr, dc) == []


def test_long_steps_declare_their_intermediates():
    for dr, dc in NEIGHBOURS_16[8:]:
        mids = _intermediate_cells(dr, dc)
        assert len(mids) == 2
        for mr, mc in mids:
            assert abs(mr) <= abs(dr) and abs(mc) <= abs(dc)


# ---------------------------------------------------------------------------
# A*
# ---------------------------------------------------------------------------


def test_straight_line_distance_is_exact():
    n = 20
    grid = make_grid(n, n)
    route = find_route(open_water_cost(n, n), grid, (10, 2), (10, 17))
    assert route.distance_km == pytest.approx(15 * RES / 1000, rel=0.02)


def test_travel_time_matches_distance_over_speed():
    n = 20
    grid = make_grid(n, n)
    route = find_route(open_water_cost(n, n, speed_kn=12.0), grid, (10, 2), (10, 17))
    expected_h = (route.distance_km * 1000) / (12.0 * KNOT_TO_MS) / 3600
    assert route.time_hours == pytest.approx(expected_h, rel=0.01)


def test_16_connectivity_beats_8_on_a_diagonal():
    """The digitisation error SPEC §8.4 warns about, measured.

    A 1:2 slope is representable exactly on a 16-connected grid and only
    approximable on an 8-connected one, which must therefore travel further.
    """
    n = 40
    grid = make_grid(n, n)
    cost = open_water_cost(n, n)
    start, goal = (2, 2), (12, 22)  # dr=10, dc=20 — exactly a 1:2 slope

    r8 = find_route(cost, grid, start, goal, connectivity=8)
    r16 = find_route(cost, grid, start, goal, connectivity=16)

    # Ground truth is the geodesic between the endpoints, not the planar hypot.
    # Using the planar figure here would silently reintroduce the 0.5% scale
    # error that D-010 exists to remove.
    lon0, lat0 = grid.lonlat_of(*start)
    lon1, lat1 = grid.lonlat_of(*goal)
    true_km = geodesic_distance_m(lon0, lat0, lon1, lat1) / 1000

    assert r16.distance_km == pytest.approx(true_km, rel=0.005)
    assert r8.distance_km > r16.distance_km
    # The 8-connected excess is the ~8% the spec cites.
    assert (r8.distance_km / true_km - 1) > 0.03


def test_route_prefers_open_water_over_slow_ice():
    """The demo's whole argument: fastest is not shortest.

    A slow corridor lies straight ahead; open water detours around it. The
    route must take the detour, and it must be longer in distance and shorter
    in time.
    """
    n = 41
    rio = np.full((n, n), 30.0)
    rio[18:23, :] = -5.0          # elevated band: PC6 capped at 3 kn
    rio[18:23, 0:3] = 30.0        # a gap at the left edge
    grid = make_grid(n, n)
    cost = build_cost_grid(rio, SpeedModel("PC6", 12.0))

    route = find_route(cost, grid, (5, 20), (35, 20), rio=rio)
    straight = np.hypot(30, 0) * RES / 1000

    assert route.distance_km > straight, "avrebbe dovuto deviare"
    slow_h = (5 * RES) / (3.0 * KNOT_TO_MS) / 3600
    assert route.time_hours < (straight * 1000) / (12 * KNOT_TO_MS) / 3600 + slow_h


def test_long_step_cannot_jump_a_one_cell_barrier():
    """A knight move must not tunnel through a coastline.

    Without the intermediate check a (1,2) step hops a one-cell wall, and the
    route is optimal on the grid and impossible at sea.
    """
    n = 21
    rio = np.full((n, n), 30.0)
    land = np.zeros((n, n), dtype=bool)
    land[:, 10] = True            # a wall one cell thick, fully closed
    grid = make_grid(n, n)
    cost = build_cost_grid(rio, SpeedModel("PC6", 12.0), land_mask=land)

    with pytest.raises(NoRouteFound) as exc:
        find_route(cost, grid, (10, 2), (10, 18))
    assert exc.value.reason == "disconnected"


def test_route_passes_through_a_gap_in_the_barrier():
    n = 21
    rio = np.full((n, n), 30.0)
    land = np.zeros((n, n), dtype=bool)
    land[:, 10] = True
    land[3, 10] = False           # one gate
    grid = make_grid(n, n)
    cost = build_cost_grid(rio, SpeedModel("PC6", 12.0), land_mask=land)

    route = find_route(cost, grid, (10, 2), (10, 18))
    assert (3, 10) in route.cells


def test_ice_class_changes_feasibility():
    """SPEC §8.4 and §10.5: the moment the demo is built around.

    A corridor of ice scores in the elevated band. PC6 may transit it under a
    speed cap; IA may not, because below PC7 that band is already special
    consideration.
    """
    n = 21
    rio = np.full((n, n), -5.0)
    grid = make_grid(n, n)

    pc6 = build_cost_grid(rio, SpeedModel("PC6", 12.0))
    route = find_route(pc6, grid, (2, 2), (18, 18), rio=rio)
    assert route.time_hours > 0

    ia = build_cost_grid(rio, SpeedModel("IA", 12.0))
    with pytest.raises(NoRouteFound):
        find_route(ia, grid, (2, 2), (18, 18))


def test_blocked_endpoint_says_so_clearly():
    n = 10
    rio = np.full((n, n), 30.0)
    rio[2, 2] = -50.0
    grid = make_grid(n, n)
    cost = build_cost_grid(rio, SpeedModel("PC6", 12.0))

    with pytest.raises(NoRouteFound) as exc:
        find_route(cost, grid, (2, 2), (8, 8))
    assert exc.value.reason == "endpoint_blocked"
    assert "partenza" in str(exc.value)


def test_out_of_bounds_endpoint():
    grid = make_grid(10, 10)
    with pytest.raises(NoRouteFound) as exc:
        find_route(open_water_cost(10, 10), grid, (0, 0), (50, 50))
    assert exc.value.reason == "bounds"


def test_same_start_and_goal_is_a_zero_route():
    grid = make_grid(10, 10)
    route = find_route(open_water_cost(10, 10), grid, (5, 5), (5, 5))
    assert route.time_hours == 0.0
    assert len(route.cells) == 1


def test_route_cells_are_contiguous():
    """Consecutive cells must be within one allowed step of each other."""
    n = 30
    grid = make_grid(n, n)
    route = find_route(open_water_cost(n, n), grid, (3, 3), (26, 20))
    allowed = set(NEIGHBOURS_16)
    for (r0, c0), (r1, c1) in zip(route.cells, route.cells[1:], strict=False):
        assert (r1 - r0, c1 - c0) in allowed


def test_min_rio_is_reported():
    n = 21
    rio = np.full((n, n), 30.0)
    rio[10, :] = 5.0
    grid = make_grid(n, n)
    route = find_route(build_cost_grid(rio, SpeedModel("PC6", 12.0)), grid, (2, 10), (18, 10), rio=rio)
    assert route.min_rio == pytest.approx(5.0)


def test_geojson_is_wellformed():
    n = 20
    grid = make_grid(n, n)
    gj = find_route(open_water_cost(n, n), grid, (2, 2), (17, 17)).as_geojson()
    assert gj["type"] == "Feature"
    assert gj["geometry"]["type"] == "LineString"
    assert len(gj["geometry"]["coordinates"][0]) == 2
    assert "transit_time_hours" in gj["properties"]
    for lon, lat in gj["geometry"]["coordinates"]:
        assert -180 <= lon <= 180
        assert -90 <= lat <= 90
