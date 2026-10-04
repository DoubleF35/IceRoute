"""Pull the slack out of an A* path. SPEC §8.5.

A* on a 16-connected grid returns an optimal path *for that grid*, and the
grid is the problem: every step has to land on a cell centre, so a leg that
ought to run at 20 degrees is built out of steps at 0, 26.6 and 45. The result
is a route that visibly wanders in open water, which is both slower than it
needs to be and the first thing anyone looking at the screen notices.

This is the string-pulling pass. Walk the path, and wherever the straight line
between two of its cells is navigable, replace everything between them with
that straight line.

Three conditions, and the second is the one that matters
--------------------------------------------------------
A shortcut is taken only when

1. **every cell the straight line crosses is passable** — the same test A*
   applies, on the same cost grid, so this cannot route through anything A*
   would have refused;
2. **the lowest RIO on the shortcut is no lower than the lowest RIO on the
   stretch it replaces, and it crosses no ice of unknown class that stretch did
   not already cross** — speed is never bought with risk. Without the first the
   pass would cut a corner through marginal ice to save ten minutes, which is
   the trade POLARIS exists to prevent. Without the second it would do the same
   through ice nobody has classified and get away with it, because RIO is NaN
   where it is unknown and every comparison against NaN is false;
3. **it is actually faster**, by the same accounting A* uses.

Anything that fails one of the three is left exactly as A* drew it.

What it costs and what it cannot do
-----------------------------------
The pass is greedy and runs to a fixed point, which is not provably optimal —
a globally optimal any-angle path needs Theta* in the search itself, not a
post-pass. It is strictly an improvement on the input, though: condition 3
means a leg is only replaced by a faster one, so the result can never be
slower than what A* found.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np

from iceroute.routing.astar import NEIGHBOURS_16, Route, _intermediate_cells
from iceroute.routing.cost import KNOT_TO_MS, CostGrid
from iceroute.routing.geocode import GeoGrid

log = logging.getLogger(__name__)

# How far ahead a single shortcut may reach, in cells. A straight leg longer
# than this is not refused on principle; it is simply not searched for, because
# the scan is quadratic in the span and the gain past a few hundred cells is
# nil — by then the line has almost certainly hit something.
MAX_SPAN = 400

# A leg must be *meaningfully* faster, not faster by a rounding error. Without
# this an exactly straight path still gets rebuilt, because the two ways of
# summing the same time differ in the twelfth decimal, and the pass reports
# having saved half a nanosecond.
REL_GAIN = 1e-6


def line_cells(a: tuple[int, int], b: tuple[int, int]) -> list[tuple[int, int]]:
    """Every cell the straight segment a-b passes through, in order.

    Sampled at half-cell steps and de-duplicated, which is the conservative
    reading: a cell is "crossed" if the line comes within half a cell of its
    centre, so a shortcut cannot slip diagonally between two blocked cells.
    """
    (r0, c0), (r1, c1) = a, b
    span = max(abs(r1 - r0), abs(c1 - c0))
    if span == 0:
        return [a]
    n = span * 2
    out: list[tuple[int, int]] = []
    seen: set[tuple[int, int]] = set()
    for k in range(n + 1):
        t = k / n
        cell = (int(round(r0 + (r1 - r0) * t)), int(round(c0 + (c1 - c0) * t)))
        if cell not in seen:
            seen.add(cell)
            out.append(cell)
    return out


@dataclass
class Straightened:
    """The pulled path, and what pulling it bought."""

    route: Route
    n_legs: int
    n_cells_before: int
    seconds_saved: float
    metres_saved: float
    # The corners of the straight legs. ``route.cells`` holds the cells they are
    # drawn through, one lon/lat each, because `simplify_route` indexes
    # ``route.lonlat`` with indices into ``route.cells`` — break that and the
    # pipeline raises IndexError two scenes later, which is how this was found.
    anchors: list = field(default_factory=list)

    @property
    def percent_time(self) -> float:
        before = self.route.time_hours * 3600 + self.seconds_saved
        return 100.0 * self.seconds_saved / before if before > 0 else 0.0

    @property
    def percent_distance(self) -> float:
        before = self.route.distance_km * 1000 + self.metres_saved
        return 100.0 * self.metres_saved / before if before > 0 else 0.0


_STEPS_16 = frozenset(NEIGHBOURS_16)


def drawn_cells(a, b) -> list[tuple[int, int]]:
    """The cells a leg is drawn through — and exactly the ones it was checked on.

    One of A*'s own moves is drawn the way A* draws it, endpoint to endpoint.
    Interpolating it instead put a cell into the route that no check had seen,
    which `conditions_along` then counted: one cell per route, enough to turn
    "100% in normal operation" into 99.87% with nothing to point at.
    """
    dr, dc = b[0] - a[0], b[1] - a[1]
    return [a, b] if (dr, dc) in _STEPS_16 else line_cells(a, b)


def _worst_rio(cells, rio) -> tuple[float, bool]:
    """Lowest known RIO on these cells, and whether any of them is unknown.

    RIO is NaN where no class could be assigned. Folding that into a minimum
    disarms the risk test silently, since every comparison against NaN is
    false — so unknown is carried separately and refused on its own terms.
    """
    if rio is None:
        return np.inf, False
    worst, unknown = np.inf, False
    for r, c in cells:
        v = float(rio[r, c])
        if np.isnan(v):
            unknown = True
        elif v < worst:
            worst = v
    return worst, unknown


def _leg(a, b, spm, rio, inv_scale, res) -> tuple[float, float, float, list] | None:
    """Ground length, time and worst RIO of a straight leg, or None if blocked.

    A leg that *is* one of A*'s own moves is judged by A*'s own rule — the
    cells :func:`astar._intermediate_cells` names, costed as the mean of the
    two endpoints. Re-digitising those short steps with the sampler below
    amounts to a second, slightly different notion of "passable", and the two
    disagreeing is how a perfectly good improvement got thrown away on twelve
    of eighteen routes.

    Longer legs are new geometry and get the sampler: the straight distance,
    scale-corrected at the endpoints the way A* corrects a step, times the mean
    cost along the line.
    """
    dr, dc = b[0] - a[0], b[1] - a[1]
    planar = float(np.hypot(dr, dc)) * res
    ground = planar * 0.5 * (inv_scale[a] + inv_scale[b])

    if (dr, dc) in _STEPS_16:
        cells = [a, *((a[0] + mr, a[1] + mc) for mr, mc in _intermediate_cells(dr, dc)), b]
        for r, c in cells:
            if not np.isfinite(spm[r, c]):
                return None
        worst, unknown = _worst_rio(cells, rio)
        time = ground * 0.5 * (spm[a] + spm[b])
        # Drawn as A* drew it — the two endpoints — not re-interpolated. The
        # cells that end up in the route are the cells that were checked, and
        # `conditions_along` counts exactly what is drawn: interpolating here
        # slipped one unchecked cell per route into the conditions.
        return ground, float(time), worst, unknown, drawn_cells(a, b)

    cells = line_cells(a, b)
    costs = np.empty(len(cells))
    for i, (r, c) in enumerate(cells):
        s = spm[r, c]
        if not np.isfinite(s):
            return None
        costs[i] = s
    worst, unknown = _worst_rio(cells, rio)
    return ground, ground * float(costs.mean()), worst, unknown, cells


def _straight_chain(anchors, spm, rio, inv_scale, res):
    """Length, time, worst known RIO and unknown-ice flag of a chain of straight
    legs, or None if any leg is blocked."""
    if len(anchors) == 1:
        w, u = _worst_rio(anchors, rio)
        return 0.0, 0.0, w, u
    total_d = total_t = 0.0
    worst, unknown = np.inf, False
    for a, b in zip(anchors, anchors[1:], strict=False):
        leg = _leg(a, b, spm, rio, inv_scale, res)
        if leg is None:
            return None
        d, tt, w, u, _ = leg
        total_d += d
        total_t += tt
        worst = min(worst, w)
        unknown = unknown or u
    return total_d, total_t, worst, unknown


def _subpath(cells, spm, rio, inv_scale, res) -> tuple[float, float, float]:
    """Ground length, time and worst RIO of a stretch of the A* path."""
    total_d = total_t = 0.0
    for (r0, c0), (r1, c1) in zip(cells, cells[1:], strict=False):
        planar = float(np.hypot(r1 - r0, c1 - c0)) * res
        ground = planar * 0.5 * (inv_scale[r0, c0] + inv_scale[r1, c1])
        total_d += ground
        total_t += ground * 0.5 * (spm[r0, c0] + spm[r1, c1])
    worst, unknown = _worst_rio(cells, rio)
    return total_d, total_t, worst, unknown


def straighten(
    route: Route,
    cost: CostGrid,
    grid: GeoGrid,
    *,
    rio: np.ndarray | None = None,
    max_span: int = MAX_SPAN,
    max_passes: int = 3,
) -> Straightened:
    """Replace wandering stretches with straight legs wherever that is legal.

    Returns the input untouched — as a :class:`Straightened` reporting nothing
    saved — when nothing qualifies, which is the right answer for a route that
    is already tight or is threading a channel it cannot leave.
    """
    spm = cost.seconds_per_metre
    res = float(grid.resolution_m)
    inv_scale = 1.0 / grid.scale_factors()

    nodes = list(route.cells)
    # A*'s own accounting, step by step. Measuring the input as a chain of
    # straight legs would re-digitise its knight moves and could call the path
    # it just produced impassable.
    d_before, t_before, _, _ = _subpath(nodes, spm, rio, inv_scale, res)
    n_before = len(nodes)
    touched = False

    # The anchors are the real geometry: the legs between them are straight
    # lines, and their length is the straight distance. Measuring the drawn
    # cells instead would charge the route for a staircase it does not sail.
    anchors = list(nodes)
    for _ in range(max_passes):
        out = [anchors[0]]
        i, changed = 0, False
        while i < len(anchors) - 1:
            taken = None
            top = min(len(anchors) - 1, i + max_span)
            for j in range(top, i + 1, -1):
                leg = _leg(anchors[i], anchors[j], spm, rio, inv_scale, res)
                if leg is None:
                    continue
                _, t_leg, worst, unknown, _ = leg
                sub = _straight_chain(anchors[i:j + 1], spm, rio, inv_scale, res)
                if sub is None:
                    continue
                _, t_sub, worst_sub, unknown_sub = sub
                # Never buy speed with risk: not with a lower RIO, and not by
                # crossing ice nobody classified that the old stretch avoided.
                if rio is not None and (worst < worst_sub
                                        or (unknown and not unknown_sub)):
                    continue
                if t_leg >= t_sub * (1.0 - REL_GAIN):
                    continue
                taken = j
                break
            if taken is None:
                out.append(anchors[i + 1])
                i += 1
            else:
                out.append(anchors[taken])
                i = taken
                changed = True
        anchors = out
        touched = touched or changed
        if not changed:
            break

    if not touched:
        # Nothing qualified. Returning the input untouched is exact; recomputing
        # it through a different summation would report a saved nanosecond.
        return Straightened(route=route, n_legs=max(1, len(nodes) - 1),
                            n_cells_before=n_before, seconds_saved=0.0,
                            metres_saved=0.0, anchors=list(nodes))

    chain = _straight_chain(anchors, spm, rio, inv_scale, res)
    if chain is None:                      # non dovrebbe accadere: ogni tratta e' stata verificata
        log.warning("raddrizzamento %s: catena non percorribile, tengo A*", route.ice_class)
        return Straightened(route=route, n_legs=len(route.cells),
                            n_cells_before=n_before, seconds_saved=0.0, metres_saved=0.0)
    d_after, t_after, worst_after, _ = chain

    # Drawn geometry: the cells each straight leg crosses, so the overlay shows
    # the line that was actually costed.
    drawn: list[tuple[int, int]] = list(anchors[:1])
    for a, b in zip(anchors, anchors[1:], strict=False):
        drawn.extend(drawn_cells(a, b)[1:])

    # One lon/lat per drawn cell: `simplify_route` indexes this list with
    # indices into `cells`, and Douglas-Peucker recovers the corners anyway.
    lonlat = [grid.lonlat_of(r, c) for r, c in drawn]
    straight = Route(
        cells=drawn,
        lonlat=lonlat,
        time_hours=t_after / 3600.0,
        distance_km=d_after / 1000.0,
        ice_class=route.ice_class,
        min_rio=None if rio is None else worst_after,
        mean_speed_kn=(d_after / t_after / KNOT_TO_MS) if t_after > 0 else None,
        expanded_nodes=route.expanded_nodes,
        segment_speeds_kn=list(route.segment_speeds_kn),
    )
    saved_s, saved_m = t_before - t_after, d_before - d_after
    log.info(
        "raddrizzamento %s: %d celle -> %d tratte, %.2f km e %.1f min in meno (%.2f%% tempo)",
        route.ice_class, n_before, len(anchors) - 1, saved_m / 1000, saved_s / 60,
        100 * saved_s / t_before if t_before else 0.0,
    )
    return Straightened(
        route=straight,
        n_legs=max(1, len(anchors) - 1),
        n_cells_before=n_before,
        seconds_saved=max(0.0, saved_s),
        metres_saved=max(0.0, saved_m),
        anchors=list(anchors),
    )
