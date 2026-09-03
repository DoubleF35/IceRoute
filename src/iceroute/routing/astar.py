"""A* over the cost grid. SPEC §8.4.

Sixteen-connected, not eight. On an 8-connected grid the optimal path
overestimates the true length of a diagonal route by about 8% — the
digitisation error of restricting headings to multiples of 45 degrees — and on
a demo whose whole point is a percentage saving, that error is larger than the
effect being shown.

Two things the obvious implementation gets wrong:

* **Knight moves can jump a coastline.** A (1,2) step crosses the cells between
  its endpoints. Without checking them, a route can hop over a one-cell spit of
  land or a narrow ice ridge. Every long step verifies its intermediates.

* **A cell step is not a constant ground distance.** EPSG:3413 stretches with
  latitude, so the metres covered by one cell differ across the grid. The step
  distance is divided by the local scale factor, consistent with
  ``geocode.ground_distance_m``.

The heuristic is straight-line distance divided by the open-water speed. No
cell can be crossed faster than that, so it never overestimates and the path is
optimal.
"""

from __future__ import annotations

import heapq
import logging
from dataclasses import dataclass, field

import numpy as np

from iceroute.routing.cost import KNOT_TO_MS, CostGrid
from iceroute.routing.geocode import GeoGrid

log = logging.getLogger(__name__)

# 8-connected: the four axial and four diagonal steps.
NEIGHBOURS_8 = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]

# 16-connected adds the eight (1,2)-type steps, which allow headings at
# multiples of ~26.6 degrees and cut the digitisation error from ~8% to ~2%.
NEIGHBOURS_16 = NEIGHBOURS_8 + [
    (-1, -2), (-1, 2), (1, -2), (1, 2),
    (-2, -1), (-2, 1), (2, -1), (2, 1),
]


class NoRouteFound(RuntimeError):
    """No passable path exists between the endpoints.

    Not an error condition to hide: SPEC §8.4 makes this a scenario to show on
    purpose — "with a 1A this route does not exist, with a PC6 it does".
    """

    def __init__(self, message: str, *, expanded: int = 0, reason: str = ""):
        super().__init__(message)
        self.expanded = expanded
        self.reason = reason


@dataclass
class Route:
    """A path across the grid, with everything the demo needs to describe it."""

    cells: list[tuple[int, int]]
    lonlat: list[tuple[float, float]]
    time_hours: float
    distance_km: float
    ice_class: str
    min_rio: float | None = None
    mean_speed_kn: float | None = None
    expanded_nodes: int = 0
    segment_speeds_kn: list[float] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.cells)

    def as_geojson(self, properties: dict | None = None) -> dict:
        """GeoJSON LineString in WGS84, as SPEC §8.6 requires."""
        props = {
            "ice_class": self.ice_class,
            "transit_time_hours": round(self.time_hours, 3),
            "distance_km": round(self.distance_km, 3),
            "n_waypoints": len(self.lonlat),
        }
        if self.min_rio is not None:
            props["min_rio_on_route"] = round(self.min_rio, 2)
        if self.mean_speed_kn is not None:
            props["mean_speed_kn"] = round(self.mean_speed_kn, 2)
        props.update(properties or {})
        return {
            "type": "Feature",
            "geometry": {
                "type": "LineString",
                "coordinates": [[round(lon, 6), round(lat, 6)] for lon, lat in self.lonlat],
            },
            "properties": props,
        }

    def summary(self) -> str:
        return (
            f"{self.ice_class}: {self.time_hours:.1f} h su {self.distance_km:.0f} km "
            f"({len(self.cells)} celle, RIO minimo "
            f"{'—' if self.min_rio is None else f'{self.min_rio:.1f}'})"
        )


def _intermediate_cells(dr: int, dc: int) -> list[tuple[int, int]]:
    """Cells a long step passes through, relative to its origin.

    Only the (1,2)-type steps need this; axial and diagonal steps touch nothing
    in between. Both candidates are returned and both must be passable, which
    is the conservative reading: a ship should not thread a one-cell gap
    diagonally.
    """
    if abs(dr) <= 1 and abs(dc) <= 1:
        return []
    if abs(dr) == 2:
        return [(dr // 2, 0), (dr // 2, dc)]
    return [(0, dc // 2), (dr, dc // 2)]


def find_route(
    cost: CostGrid,
    grid: GeoGrid,
    start: tuple[int, int],
    goal: tuple[int, int],
    *,
    connectivity: int = 16,
    heuristic_weight: float = 1.0,
    max_expansions: int = 20_000_000,
    rio: np.ndarray | None = None,
) -> Route:
    """Least-time path from ``start`` to ``goal``, both as ``(row, col)``.

    Raises
    ------
    NoRouteFound
        If either endpoint is impassable, or no path exists, or the expansion
        budget runs out. The message says which.
    """
    nrow, ncol = cost.shape
    passable = cost.passable
    spm = cost.seconds_per_metre

    for name, (r, c) in (("partenza", start), ("arrivo", goal)):
        if not (0 <= r < nrow and 0 <= c < ncol):
            raise NoRouteFound(f"Punto di {name} fuori dalla griglia: ({r}, {c})", reason="bounds")
        if not passable[r, c]:
            raise NoRouteFound(
                f"Il punto di {name} ({r}, {c}) non è praticabile per una nave "
                f"{cost.ice_class}: è terra o ghiaccio oltre il limite POLARIS.",
                reason="endpoint_blocked",
            )

    if start == goal:
        lon, lat = grid.lonlat_of(*start)
        return Route([start], [(lon, lat)], 0.0, 0.0, cost.ice_class)

    res = grid.resolution_m
    # Per-cell scale factor: one cell step is not the same ground distance at
    # 70N and at 85N.
    inv_scale = 1.0 / grid.scale_factors()

    offsets = NEIGHBOURS_16 if connectivity == 16 else NEIGHBOURS_8
    steps = [
        (dr, dc, float(np.hypot(dr, dc)) * res, _intermediate_cells(dr, dc))
        for dr, dc in offsets
    ]

    # Admissible: nothing moves faster than open water, so this never
    # overestimates the remaining time.
    fastest_spm = cost.min_seconds_per_metre()
    gx, gy = grid.x[goal[1]], grid.y[goal[0]]
    scale_goal = float(1.0 / inv_scale[goal])

    def heuristic(r: int, c: int) -> float:
        planar = np.hypot(grid.x[c] - gx, grid.y[r] - gy)
        return float(planar) * scale_goal * fastest_spm * heuristic_weight

    g_score = np.full(nrow * ncol, np.inf, dtype="float64")
    came_from = np.full(nrow * ncol, -1, dtype="int64")
    closed = np.zeros(nrow * ncol, dtype=bool)

    start_flat = start[0] * ncol + start[1]
    goal_flat = goal[0] * ncol + goal[1]
    g_score[start_flat] = 0.0

    open_heap: list[tuple[float, int]] = [(heuristic(*start), start_flat)]
    expanded = 0

    while open_heap:
        _, current = heapq.heappop(open_heap)
        if closed[current]:
            continue
        if current == goal_flat:
            break
        closed[current] = True
        expanded += 1
        if expanded > max_expansions:
            raise NoRouteFound(
                f"Superato il budget di espansioni ({max_expansions}). "
                f"Griglia troppo grande o euristica inefficace.",
                expanded=expanded,
                reason="budget",
            )

        r, c = divmod(current, ncol)
        cur_g = g_score[current]
        cur_spm = spm[r, c]

        for dr, dc, planar, mids in steps:
            nr, nc = r + dr, c + dc
            if not (0 <= nr < nrow and 0 <= nc < ncol) or not passable[nr, nc]:
                continue
            # A long step must not tunnel through land or unnavigable ice.
            if mids and not all(passable[r + mr, c + mc] for mr, mc in mids):
                continue

            nflat = nr * ncol + nc
            if closed[nflat]:
                continue

            # Ground distance for this step, scale-corrected at the midpoint.
            ground = planar * 0.5 * (inv_scale[r, c] + inv_scale[nr, nc])
            # Mean cost of the two cells: the step spends half its length in each.
            tentative = cur_g + ground * 0.5 * (cur_spm + spm[nr, nc])

            if tentative < g_score[nflat]:
                g_score[nflat] = tentative
                came_from[nflat] = current
                heapq.heappush(open_heap, (tentative + heuristic(nr, nc), nflat))
    else:
        raise NoRouteFound(
            f"Nessun percorso esistente per una nave {cost.ice_class} fra "
            f"{start} e {goal}. Il ghiaccio o la terra bloccano ogni via.",
            expanded=expanded,
            reason="disconnected",
        )

    # --- reconstruct ------------------------------------------------------
    path: list[int] = []
    node = goal_flat
    while node != -1:
        path.append(node)
        node = came_from[node]
    path.reverse()

    cells = [divmod(n, ncol) for n in path]
    lonlat = [grid.lonlat_of(r, c) for r, c in cells]

    distance_m = 0.0
    speeds: list[float] = []
    for (r0, c0), (r1, c1) in zip(cells, cells[1:], strict=False):
        planar = float(np.hypot(r1 - r0, c1 - c0)) * res
        ground = planar * 0.5 * (inv_scale[r0, c0] + inv_scale[r1, c1])
        distance_m += ground
        speeds.append(float(0.5 * (cost.speed_kn[r0, c0] + cost.speed_kn[r1, c1])))

    time_hours = float(g_score[goal_flat]) / 3600.0

    min_rio = None
    if rio is not None:
        vals = [rio[r, c] for r, c in cells if np.isfinite(rio[r, c])]
        min_rio = float(min(vals)) if vals else None

    mean_speed = (
        distance_m / (time_hours * 3600.0) / KNOT_TO_MS if time_hours > 0 else None
    )

    log.info(
        "rotta %s: %.1f h, %.0f km, %d celle, %d nodi espansi",
        cost.ice_class, time_hours, distance_m / 1000, len(cells), expanded,
    )

    return Route(
        cells=cells,
        lonlat=lonlat,
        time_hours=time_hours,
        distance_km=distance_m / 1000.0,
        ice_class=cost.ice_class,
        min_rio=min_rio,
        mean_speed_kn=mean_speed,
        expanded_nodes=expanded,
        segment_speeds_kn=speeds,
    )
