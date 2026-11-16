"""Choose origin and destination for a scene. Closes docs/decisions.md P-07.

The demo compares the same voyage across ice classes, and that only works if
both endpoints are navigable for **every** class shown. Pick them for PC6 and a
1A fails immediately with "start blocked", which is a dull and slightly
dishonest way to lose: the interesting statement is *"for a 1A there is no way
through"*, not *"a 1A cannot leave port"*.

So the endpoints are chosen inside the region navigable by the **weakest** class
under consideration, and inside one connected component of it. Whether a route
then exists for each class is left to A* to answer.

Ordering note: the "farthest pair" is approximated by projecting the component
onto its principal axis and taking the extremes. The exact diameter of a set of
180,000 cells is quadratic and unnecessary — we need a long, plausible voyage,
not a provably maximal one.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from iceroute.routing.cost import CostGrid
from iceroute.routing.geocode import GeoGrid

log = logging.getLogger(__name__)


@dataclass
class Waypoints:
    """A start/goal pair with the provenance of the choice."""

    start: tuple[int, int]
    goal: tuple[int, int]
    start_lonlat: tuple[float, float]
    goal_lonlat: tuple[float, float]
    straight_km: float
    component_cells: int
    n_components: int
    limiting_class: str
    auto: bool = True

    def as_dict(self) -> dict:
        return {
            "start": {
                "row": self.start[0],
                "col": self.start[1],
                "lon": round(self.start_lonlat[0], 5),
                "lat": round(self.start_lonlat[1], 5),
            },
            "goal": {
                "row": self.goal[0],
                "col": self.goal[1],
                "lon": round(self.goal_lonlat[0], 5),
                "lat": round(self.goal_lonlat[1], 5),
            },
            "straight_km": round(self.straight_km, 2),
            "component_cells": self.component_cells,
            "n_components": self.n_components,
            "limiting_class": self.limiting_class,
            "auto_selected": self.auto,
        }


def largest_component(passable: np.ndarray) -> tuple[np.ndarray, int, int]:
    """Boolean mask of the largest connected navigable region.

    Eight-connected, matching the routing's own connectivity: a region A* can
    traverse must count as one region here, or endpoints get rejected from a
    component the router would happily cross.
    """
    from scipy.ndimage import label

    lab, n = label(passable, structure=np.ones((3, 3), dtype=bool))
    if n == 0:
        return np.zeros_like(passable), 0, 0
    sizes = np.bincount(lab.ravel())
    sizes[0] = 0
    biggest = int(sizes.argmax())
    return lab == biggest, int(sizes[biggest]), int(n)


def choose_waypoints(
    grid: GeoGrid,
    costs: dict[str, CostGrid],
    *,
    limiting_class: str | None = None,
    min_separation_km: float = 50.0,
) -> Waypoints:
    """Pick a start and goal navigable for every class in ``costs``.

    Parameters
    ----------
    costs
        One cost grid per ice class the demo will show.
    limiting_class
        Which class defines the navigable region. Defaults to whichever has the
        fewest passable cells — the weakest.

    Raises
    ------
    ValueError
        If no component is large enough to hold a voyage of at least
        ``min_separation_km``. Better an explicit failure than a two-cell route
        that renders as a dot.
    """
    if not costs:
        raise ValueError("At least one CostGrid is required.")

    if limiting_class is None:
        limiting_class = min(costs, key=lambda k: int(costs[k].passable.sum()))

    # Navigable for every class at once.
    common = np.ones(grid.shape, dtype=bool)
    for cg in costs.values():
        common &= cg.passable

    mask, size, n_comp = largest_component(common)
    if size == 0:
        raise ValueError(
            "No cell is navigable for all the requested classes: "
            f"{sorted(costs)}. The scene cannot be used for the comparison."
        )

    cells = np.argwhere(mask)
    centred = cells - cells.mean(axis=0)
    # Principal axis of the component; its extremes are a long, plausible voyage.
    axis = np.linalg.svd(centred, full_matrices=False)[2][0]
    proj = centred @ axis
    start = tuple(int(v) for v in cells[proj.argmin()])
    goal = tuple(int(v) for v in cells[proj.argmax()])

    from iceroute.routing.geocode import geodesic_distance_m

    slon, slat = grid.lonlat_of(*start)
    glon, glat = grid.lonlat_of(*goal)
    straight_km = geodesic_distance_m(slon, slat, glon, glat) / 1000.0

    if straight_km < min_separation_km:
        raise ValueError(
            f"The largest navigable component spans only {straight_km:.0f} km "
            f"(minimum required {min_separation_km:.0f} km). Scene not suitable for the demo."
        )

    log.info(
        "automatic waypoints: %s → %s, %.0f km in a straight line "
        "(component of %d cells out of %d components, limiting class %s)",
        start,
        goal,
        straight_km,
        size,
        n_comp,
        limiting_class,
    )

    return Waypoints(
        start=start,
        goal=goal,
        start_lonlat=(slon, slat),
        goal_lonlat=(glon, glat),
        straight_km=straight_km,
        component_cells=size,
        n_components=n_comp,
        limiting_class=limiting_class,
    )


def waypoints_from_config(grid: GeoGrid, entry: dict, limiting_class: str = "") -> Waypoints:
    """Build waypoints from explicit lon/lat in ``configs/routing.yaml``.

    Hand-picked waypoints take precedence over automatic ones: real port pairs
    on the Northern Sea Route are a more representative voyage than the
    extremes of a connected component.
    """
    from iceroute.routing.geocode import geodesic_distance_m

    o, d = entry["origin"], entry["destination"]
    start = grid.index_of(float(o["lon"]), float(o["lat"]))
    goal = grid.index_of(float(d["lon"]), float(d["lat"]))
    slon, slat = grid.lonlat_of(*start)
    glon, glat = grid.lonlat_of(*goal)
    return Waypoints(
        start=start,
        goal=goal,
        start_lonlat=(slon, slat),
        goal_lonlat=(glon, glat),
        straight_km=geodesic_distance_m(slon, slat, glon, glat) / 1000.0,
        component_cells=0,
        n_components=0,
        limiting_class=limiting_class,
        auto=False,
    )
