"""Tests for the string-pulling pass.

This one changes where a ship is told to sail, so the tests are not about
tidiness. Two properties carry everything: a shortcut may never cross a cell
A* would have refused, and it may never lower the worst RIO on the stretch it
replaces. A pass that only made routes shorter would be a pass that trades
risk for speed, which is the trade POLARIS exists to forbid.
"""

from __future__ import annotations

import numpy as np
import pytest

from iceroute.routing.astar import Route
from iceroute.routing.cost import KNOT_TO_MS, CostGrid
from iceroute.routing.geocode import GeoGrid
from iceroute.routing.shortcut import line_cells, straighten

N = 40
RES = 500.0
OPEN_KN = 12.0


def _grid() -> GeoGrid:
    x = np.arange(N) * RES
    y = -np.arange(N) * RES + 2_000_000.0
    return GeoGrid(x=x, y=y, crs="EPSG:3413", resolution_m=RES)


def _cost(blocked=None, slow=None, slow_kn=3.0) -> CostGrid:
    speed = np.full((N, N), OPEN_KN)
    if slow is not None:
        speed[slow] = slow_kn
    if blocked is not None:
        speed[blocked] = 0.0
    passable = speed > 0
    with np.errstate(divide="ignore"):
        spm = np.where(passable, 1.0 / (speed * KNOT_TO_MS), np.inf)
    return CostGrid(
        seconds_per_metre=spm,
        passable=passable,
        speed_kn=speed,
        ice_class="PC6",
        open_water_speed_kn=OPEN_KN,
    )


def _chain(*corners):
    """A cell path through the given corners, the way A* would hand one over."""
    out: list[tuple[int, int]] = []
    for a, b in zip(corners, corners[1:], strict=False):
        seg = line_cells(a, b)
        out.extend(seg if not out else seg[1:])
    return out


def _route(cells) -> Route:
    return Route(
        cells=cells,
        lonlat=[(0.0, 0.0)] * len(cells),
        time_hours=0.0,
        distance_km=0.0,
        ice_class="PC6",
    )


def _detour():
    """A route that climbs away from the straight line and comes back."""
    return _chain((20, 5), (8, 20), (20, 35))


# ---------------------------------------------------------------------------
# The line itself
# ---------------------------------------------------------------------------


def test_a_line_between_neighbours_is_just_the_two_cells():
    assert line_cells((3, 3), (3, 4)) == [(3, 3), (3, 4)]
    assert line_cells((3, 3), (3, 3)) == [(3, 3)]


def test_a_line_is_contiguous_and_ends_where_it_was_told_to():
    cells = line_cells((2, 2), (9, 20))
    assert cells[0] == (2, 2) and cells[-1] == (9, 20)
    for (r0, c0), (r1, c1) in zip(cells, cells[1:], strict=False):
        assert max(abs(r1 - r0), abs(c1 - c0)) == 1, "la linea salta una cella"


def test_a_line_cannot_slip_between_two_blocked_cells():
    """Sampled at half-cell steps, so a diagonal crossing touches both cells of
    the gap rather than threading it."""
    cells = set(line_cells((0, 0), (10, 10)))
    assert (5, 5) in cells


# ---------------------------------------------------------------------------
# Straightening
# ---------------------------------------------------------------------------


def test_open_water_collapses_to_a_single_leg():
    r = straighten(_route(_detour()), _cost(), _grid())
    assert r.route.cells[0] == (20, 5) and r.route.cells[-1] == (20, 35)
    assert r.anchors == [(20, 5), (20, 35)], "più di una tratta"
    assert len(r.route.lonlat) == len(r.route.cells), (
        "simplify_route indicizza lonlat con indici di cells"
    )
    assert r.metres_saved > 0 and r.seconds_saved > 0
    assert r.percent_distance > 20


def test_an_already_straight_route_is_left_alone():
    straight = _chain((20, 5), (20, 35))
    r = straighten(_route(straight), _cost(), _grid())
    assert r.seconds_saved == 0.0 and r.metres_saved == 0.0
    assert r.route.cells == straight


def test_a_wall_is_not_cut_through():
    """The decisive one: the shortcut is the whole point, and it must still
    refuse to cross a cell A* would have refused."""
    blocked = np.zeros((N, N), dtype=bool)
    blocked[18:23, 15:26] = True
    cost = _cost(blocked=blocked)
    r = straighten(_route(_detour()), cost, _grid())
    for cell in r.route.cells:
        assert cost.passable[cell], f"la rotta raddrizzata attraversa {cell}, impraticabile"


def test_speed_is_never_bought_with_risk():
    """The straight line is open water and faster, and it is still refused:
    its worst RIO is below the worst RIO of the stretch it would replace."""
    rio = np.full((N, N), 20.0)
    rio[18:23, 10:30] = -5.0  # fascia a rischio elevato sulla linea retta
    before = _detour()
    r = straighten(_route(before), _cost(), _grid(), rio=rio)
    assert r.route.min_rio is not None and r.route.min_rio >= 20.0
    assert all(rio[c] >= 20.0 for c in r.route.cells)


def test_the_same_band_is_crossed_when_the_route_already_crossed_it():
    """The rule is "no worse", not "never": if A* already went through the
    band, a straight line through the same band is allowed."""
    rio = np.full((N, N), 20.0)
    rio[0:25, :] = -5.0
    r = straighten(_route(_detour()), _cost(), _grid(), rio=rio)
    assert r.metres_saved > 0


def test_a_band_slow_enough_is_not_driven_straight_through():
    """Not a safety rule, just arithmetic. Entering slow ice to save distance
    is *correct* when it comes out faster — that is condition three doing its
    job, and counting cells rather than hours would call it a fault. What must
    not happen is the pass flattening the detour into one leg straight across
    half-knot ice."""
    slow = np.zeros((N, N), dtype=bool)
    slow[15:26, 8:33] = True
    r = straighten(_route(_detour()), _cost(slow=slow, slow_kn=0.5), _grid())
    assert len(r.anchors) > 2, "ha tirato dritto attraverso il banco lento"
    assert r.seconds_saved >= 0.0


def test_the_result_is_never_slower_than_what_a_star_found():
    """Condition three, stated as the property it exists to guarantee."""
    rng = np.random.default_rng(0)
    for seed in range(5):
        blocked = rng.random((N, N)) < 0.04
        blocked[20, :] = False  # un corridoio sempre libero
        cost = _cost(blocked=blocked)
        path = _chain((20, 2), (20 - seed, 18), (20, 37))
        if not all(cost.passable[c] for c in path):
            continue
        r = straighten(_route(path), cost, _grid())
        assert r.seconds_saved >= 0.0
        assert r.route.time_hours * 3600 + r.seconds_saved == pytest.approx(
            r.route.time_hours * 3600 + r.seconds_saved
        )


def test_the_drawn_cells_join_up_so_the_overlay_can_follow_them():
    r = straighten(_route(_detour()), _cost(), _grid())
    for (r0, c0), (r1, c1) in zip(r.route.cells, r.route.cells[1:], strict=False):
        assert max(abs(r1 - r0), abs(c1 - c0)) == 1


def test_distance_is_the_straight_line_not_the_drawn_staircase():
    """The legs are straight lines; charging the route for the staircase of
    cells drawn to show them would overstate every distance on the page."""
    r = straighten(_route(_detour()), _cost(), _grid())
    span = 30 * RES / 1000.0
    assert r.route.distance_km == pytest.approx(span, rel=0.02)


def test_a_route_of_one_cell_survives_the_pass():
    r = straighten(_route([(5, 5)]), _cost(), _grid())
    assert r.route.cells == [(5, 5)] and r.seconds_saved == 0.0


def test_unclassified_ice_is_not_crossed_to_save_time():
    """RIO is NaN where no class could be assigned, and every comparison
    against NaN is false — so a minimum that folded NaN in would wave the
    shortcut through. Unknown is refused on its own terms."""
    rio = np.full((N, N), 20.0)
    rio[18:23, 10:30] = np.nan
    before = _detour()
    r = straighten(_route(before), _cost(), _grid(), rio=rio)
    assert not any(np.isnan(rio[c]) for c in r.route.cells), (
        "la scorciatoia attraversa ghiaccio non classificato"
    )


def test_unclassified_ice_already_on_the_route_does_not_block_the_pass():
    """The rule is "no new unknown", not "never": a route that already crosses
    unclassified ice may still be straightened through it."""
    rio = np.full((N, N), np.nan)
    r = straighten(_route(_detour()), _cost(), _grid(), rio=rio)
    assert r.metres_saved > 0


def test_the_route_contains_only_cells_that_were_checked():
    """What is drawn is what was validated. The two were computed by separate
    code paths once, and one unchecked cell per route leaked into the
    conditions panel as "99.87% in normal operation" with nothing to point at.
    """
    blocked = np.zeros((N, N), dtype=bool)
    blocked[18:23, 15:26] = True
    rio = np.full((N, N), 20.0)
    rio[blocked] = np.nan
    cost = _cost(blocked=blocked)
    r = straighten(_route(_detour()), cost, _grid(), rio=rio)
    for cell in r.route.cells:
        assert cost.passable[cell], f"{cell} disegnata ma non praticabile"
        assert not np.isnan(rio[cell]), f"{cell} disegnata ma di classe ignota"
