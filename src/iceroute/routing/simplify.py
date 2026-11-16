"""Reduce a grid path to a waypoint list a bridge could actually use. SPEC §8.5.

An A* path over a 500 m grid has hundreds of cells and turns at every step.
That is not a route, it is a raster. Douglas-Peucker reduces it to the corners
that matter, and a sanity pass flags turns no vessel would make.

Simplification is done in **projected metres**, not in degrees. A tolerance in
degrees means something different at 70°N than at 85°N, since a degree of
longitude shortens by more than half between them.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np

from iceroute.routing.astar import Route
from iceroute.routing.geocode import GeoGrid, geodesic_distance_m

log = logging.getLogger(__name__)


@dataclass
class SimplifiedRoute:
    """A reduced waypoint list, with whatever the sanity check found."""

    lonlat: list[tuple[float, float]]
    xy: list[tuple[float, float]]
    n_original: int
    max_turn_deg: float
    sharp_turns: list[int] = field(default_factory=list)
    max_deviation_m: float = 0.0
    warnings: list[str] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.lonlat)

    def as_geojson(self, properties: dict | None = None) -> dict:
        props = {
            "n_waypoints": len(self.lonlat),
            "n_original_cells": self.n_original,
            "max_turn_deg": round(self.max_turn_deg, 1),
            "max_deviation_m": round(self.max_deviation_m, 1),
        }
        if self.sharp_turns:
            props["sharp_turn_indices"] = self.sharp_turns
        if self.warnings:
            props["warnings"] = self.warnings
        props.update(properties or {})
        return {
            "type": "Feature",
            "geometry": {
                "type": "LineString",
                "coordinates": [[round(lon, 6), round(lat, 6)] for lon, lat in self.lonlat],
            },
            "properties": props,
        }


def _perpendicular_distance(pt: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
    """Distance from ``pt`` to the segment ``ab`` (not the infinite line).

    Using the segment rather than the line matters at the ends of a run: a
    point beyond the segment should measure to the endpoint, otherwise
    Douglas-Peucker keeps corners it does not need.
    """
    ab = b - a
    denom = float(ab @ ab)
    if denom == 0.0:
        return float(np.linalg.norm(pt - a))
    t = float(np.clip((pt - a) @ ab / denom, 0.0, 1.0))
    return float(np.linalg.norm(pt - (a + t * ab)))


def douglas_peucker(points: np.ndarray, tolerance_m: float) -> list[int]:
    """Indices of the points to keep. Iterative, so a long route cannot
    overflow the recursion limit."""
    n = len(points)
    if n <= 2:
        return list(range(n))

    keep = np.zeros(n, dtype=bool)
    keep[0] = keep[-1] = True
    stack = [(0, n - 1)]

    while stack:
        i, j = stack.pop()
        if j <= i + 1:
            continue
        a, b = points[i], points[j]
        best_d, best_k = -1.0, -1
        for k in range(i + 1, j):
            d = _perpendicular_distance(points[k], a, b)
            if d > best_d:
                best_d, best_k = d, k
        if best_d > tolerance_m:
            keep[best_k] = True
            stack.append((i, best_k))
            stack.append((best_k, j))

    return np.flatnonzero(keep).tolist()


def turn_angles_deg(points: np.ndarray) -> np.ndarray:
    """Course change at each interior vertex, in degrees. 0 means straight on."""
    if len(points) < 3:
        return np.zeros(0)
    v1 = points[1:-1] - points[:-2]
    v2 = points[2:] - points[1:-1]
    n1 = np.linalg.norm(v1, axis=1)
    n2 = np.linalg.norm(v2, axis=1)
    ok = (n1 > 0) & (n2 > 0)
    cos = np.ones(len(v1))
    cos[ok] = np.clip((v1[ok] * v2[ok]).sum(axis=1) / (n1[ok] * n2[ok]), -1.0, 1.0)
    return np.degrees(np.arccos(cos))


def simplify_route(
    route: Route,
    grid: GeoGrid,
    *,
    tolerance_m: float = 2000.0,
    max_waypoints: int = 40,
    max_turn_deg: float = 60.0,
) -> SimplifiedRoute:
    """Reduce a route to at most ``max_waypoints`` vertices.

    If the tolerance leaves too many points the tolerance is raised and the
    reduction repeated, rather than truncating the list — dropping the tail of
    a route would move its destination.
    """
    xy = np.array([[grid.x[c], grid.y[r]] for r, c in route.cells], dtype="float64")
    n_original = len(xy)

    if n_original <= 2:
        idx = list(range(n_original))
        tol = tolerance_m
    else:
        tol = tolerance_m
        idx = douglas_peucker(xy, tol)
        # Doubling converges quickly and keeps the geometry honest, where
        # truncation would silently relocate the destination.
        for _ in range(20):
            if len(idx) <= max_waypoints:
                break
            tol *= 1.6
            idx = douglas_peucker(xy, tol)

    kept = xy[idx]
    lonlat = [route.lonlat[i] for i in idx]

    # How far the simplified line strays from the original path.
    max_dev = 0.0
    for a_i, b_i in zip(idx, idx[1:], strict=False):
        a, b = xy[a_i], xy[b_i]
        for k in range(a_i + 1, b_i):
            max_dev = max(max_dev, _perpendicular_distance(xy[k], a, b))

    angles = turn_angles_deg(kept)
    sharp = [int(i + 1) for i, ang in enumerate(angles) if ang > max_turn_deg]

    warnings: list[str] = []
    if len(idx) > max_waypoints:
        warnings.append(
            f"{len(idx)} waypoint dopo la semplificazione, oltre il massimo di "
            f"{max_waypoints}: la tolleranza non è bastata."
        )
    if sharp:
        warnings.append(
            f"{len(sharp)} virate oltre {max_turn_deg:.0f}°: verificare che siano "
            f"realistiche per una nave."
        )
    if len(lonlat) >= 2:
        endpoint_shift = geodesic_distance_m(*route.lonlat[-1], *lonlat[-1])
        if endpoint_shift > 1.0:
            warnings.append(f"L'arrivo si è spostato di {endpoint_shift:.0f} m.")

    log.info(
        "semplificazione: %d celle → %d waypoint (tolleranza %.0f m, "
        "deviazione massima %.0f m, %d virate strette)",
        n_original,
        len(idx),
        tol,
        max_dev,
        len(sharp),
    )

    return SimplifiedRoute(
        lonlat=lonlat,
        xy=[(float(a), float(b)) for a, b in kept],
        n_original=n_original,
        max_turn_deg=float(angles.max()) if angles.size else 0.0,
        sharp_turns=sharp,
        max_deviation_m=max_dev,
        warnings=warnings,
    )
