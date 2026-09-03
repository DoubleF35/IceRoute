"""The comparison route: what a ship would do without IceRoute. SPEC §8.5.

A great-circle track between the same two waypoints, with the land mask
applied but the ice ignored. That is the honest comparison — "what would a
vessel do with only a nautical chart" — and it is what the demo's delta is
measured against.

Two properties make it honest rather than a straw man:

* it is the **shortest** path, so IceRoute cannot win on distance, only on
  time. If IceRoute were merely shorter the comparison would prove nothing.
* when it crosses ice the vessel cannot legally transit, that is **reported,
  not silently priced in**. Quietly slowing the baseline through a no-go cell
  would invent a transit that POLARIS says must not happen, and would inflate
  our saving with a number nobody can defend.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
from pyproj import Geod

from iceroute.routing.astar import Route
from iceroute.routing.cost import KNOT_TO_MS, CostGrid
from iceroute.routing.geocode import GeoGrid

log = logging.getLogger(__name__)

GEOD = Geod(ellps="WGS84")


@dataclass
class BaselineResult:
    """A geodesic route and what happens to it when the ice is taken into account."""

    route: Route
    crosses_no_go: bool
    n_no_go_cells: int
    n_slow_cells: int
    crosses_land: bool
    feasible: bool
    note: str = ""

    def as_geojson(self) -> dict:
        gj = self.route.as_geojson(
            {
                "kind": "baseline",
                "feasible": self.feasible,
                "crosses_no_go": self.crosses_no_go,
                "n_no_go_cells": self.n_no_go_cells,
                "crosses_land": self.crosses_land,
                "note": self.note,
            }
        )
        if not self.feasible:
            # A time for an infeasible track would be a fabricated number.
            gj["properties"]["transit_time_hours"] = None
        return gj


def geodesic_points(
    lon0: float, lat0: float, lon1: float, lat1: float, n_points: int = 200
) -> list[tuple[float, float]]:
    """Points along the great circle, endpoints included."""
    if n_points < 2:
        raise ValueError("n_points deve essere almeno 2")
    inter = GEOD.npts(lon0, lat0, lon1, lat1, n_points - 2)
    return [(lon0, lat0), *[(float(a), float(b)) for a, b in inter], (lon1, lat1)]


def _densify_cells(grid: GeoGrid, points: list[tuple[float, float]]) -> list[tuple[int, int]]:
    """Cells the track passes through, de-duplicated and in order.

    The geodesic is sampled finely enough that consecutive samples land in the
    same or adjacent cells; taking unique consecutive cells then gives the
    traversed set without gaps.
    """
    cells: list[tuple[int, int]] = []
    for lon, lat in points:
        try:
            rc = grid.index_of(lon, lat)
        except ValueError:
            continue
        if not cells or rc != cells[-1]:
            cells.append(rc)
    return cells


def geodesic_baseline(
    grid: GeoGrid,
    cost: CostGrid,
    start: tuple[int, int],
    goal: tuple[int, int],
    *,
    open_water_speed_kn: float,
    rio: np.ndarray | None = None,
    land_mask: np.ndarray | None = None,
    samples: int = 500,
) -> BaselineResult:
    """Great-circle track between the two waypoints, evaluated against the ice.

    The transit time assumes open-water speed throughout, which is exactly the
    assumption a master with only a paper chart would make. Where the track
    crosses cells the vessel may not enter, the result is marked infeasible and
    the time is withheld.
    """
    lon0, lat0 = grid.lonlat_of(*start)
    lon1, lat1 = grid.lonlat_of(*goal)

    # Sample densely enough that no cell is skipped: at least a couple of
    # samples per cell along the track.
    total_m = float(GEOD.inv(lon0, lat0, lon1, lat1)[2])
    n = max(samples, int(2 * total_m / grid.resolution_m) + 2)
    points = geodesic_points(lon0, lat0, lon1, lat1, n)
    cells = _densify_cells(grid, points)

    if not cells:
        raise ValueError("La rotta geodetica non interseca la griglia.")

    n_no_go = sum(1 for r, c in cells if not cost.passable[r, c])
    n_slow = sum(
        1
        for r, c in cells
        if cost.passable[r, c] and cost.speed_kn[r, c] < open_water_speed_kn
    )
    on_land = (
        bool(any(land_mask[r, c] for r, c in cells)) if land_mask is not None else False
    )

    time_hours = (total_m / (open_water_speed_kn * KNOT_TO_MS)) / 3600.0

    min_rio = None
    if rio is not None:
        vals = [rio[r, c] for r, c in cells if np.isfinite(rio[r, c])]
        min_rio = float(min(vals)) if vals else None

    feasible = n_no_go == 0 and not on_land
    if on_land:
        note = "La rotta geodetica attraversa terraferma: non percorribile."
    elif n_no_go:
        note = (
            f"La rotta geodetica attraversa {n_no_go} celle oltre il limite POLARIS "
            f"per una nave {cost.ice_class}. Il tempo di transito non è definito: "
            f"quella rotta non si può percorrere."
        )
    elif n_slow:
        note = (
            f"Percorribile, ma {n_slow} celle sono in regime di rischio elevato: "
            f"il tempo mostrato assume velocità piena e quindi è ottimistico."
        )
    else:
        note = "Percorribile in acqua libera per tutta la lunghezza."

    route = Route(
        cells=cells,
        lonlat=[grid.lonlat_of(r, c) for r, c in cells],
        time_hours=time_hours,
        distance_km=total_m / 1000.0,
        ice_class=cost.ice_class,
        min_rio=min_rio,
        mean_speed_kn=open_water_speed_kn,
    )

    log.info(
        "baseline %s: %.0f km, %s (%d celle no-go, %d a velocità ridotta)",
        cost.ice_class,
        total_m / 1000,
        "percorribile" if feasible else "NON percorribile",
        n_no_go,
        n_slow,
    )

    return BaselineResult(
        route=route,
        crosses_no_go=n_no_go > 0,
        n_no_go_cells=n_no_go,
        n_slow_cells=n_slow,
        crosses_land=on_land,
        feasible=feasible,
        note=note,
    )


def compare(route: Route, baseline: BaselineResult) -> dict:
    """The numbers behind the demo's headline KPI card.

    ``time_saved_hours`` is ``None`` when the baseline is infeasible, because
    there is no honest number to put there: you cannot save time against a
    voyage that cannot be made. What is reported instead is that IceRoute finds
    a route where the naive one does not exist — a stronger claim, and a true
    one.
    """
    out: dict = {
        "iceroute": {
            "time_hours": round(route.time_hours, 3),
            "distance_km": round(route.distance_km, 3),
            "min_rio": None if route.min_rio is None else round(route.min_rio, 2),
            "n_waypoints": len(route.cells),
        },
        "baseline": {
            "time_hours": round(baseline.route.time_hours, 3) if baseline.feasible else None,
            "distance_km": round(baseline.route.distance_km, 3),
            "min_rio": None if baseline.route.min_rio is None else round(baseline.route.min_rio, 2),
            "feasible": baseline.feasible,
            "n_no_go_cells": baseline.n_no_go_cells,
            "note": baseline.note,
        },
        "detour_km": round(route.distance_km - baseline.route.distance_km, 3),
    }

    if baseline.feasible:
        saved = baseline.route.time_hours - route.time_hours
        out["time_saved_hours"] = round(saved, 3)
        out["time_saved_percent"] = (
            round(100 * saved / baseline.route.time_hours, 2)
            if baseline.route.time_hours > 0
            else None
        )
        out["headline"] = (
            f"{abs(saved):.1f} h {'risparmiate' if saved > 0 else 'in più'} "
            f"({abs(out['time_saved_percent']):.0f}%)"
        )
    else:
        out["time_saved_hours"] = None
        out["time_saved_percent"] = None
        out["headline"] = (
            f"La rotta diretta non è percorribile da una nave {route.ice_class}. "
            f"IceRoute ne trova una in {route.time_hours:.1f} h."
        )

    return out
