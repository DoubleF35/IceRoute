"""From RIO to a traversal cost in time. SPEC §8.3.

The cost of crossing a cell is **time**, not distance. That is the whole point:
it is what makes the shortest route and the fastest route different, and the
difference is what the demo shows.

The speed model is not a curve someone invented. It is the three levels of
operation of MSC.1/Circ.1519 §1.3.2 made executable:

    RIO >= 0         normal operation        full speed
    -10 <= RIO < 0   elevated risk           the speed cap of Table 1.2
    RIO < -10        special consideration   impassable

Only one number in the chain lacks a normative source: the open-water speed,
which depends on the vessel and not on the regulation. It is marked
PROVVISORIO in ``configs/routing.yaml`` and :func:`provisional_parameters`
surfaces it so the UI can say so out loud.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np

from iceroute.routing.polaris import (
    ELEVATED_SPEED_LIMIT_KN,
    POLAR_CLASSES,
    RioThresholds,
    classify_rio,
)

log = logging.getLogger(__name__)

KNOT_TO_MS = 0.514444
IMPASSABLE = np.inf


@dataclass
class SpeedModel:
    """Maps RIO to speed in knots for one vessel class.

    Attributes
    ----------
    ice_class
        e.g. ``"PC6"``.
    open_water_speed_kn
        Speed under normal operation. **Unsourced** — see the module docstring.
    band_profile
        ``"step"`` applies the Table 1.2 cap flat across the elevated band,
        which is what the circular literally says. ``"linear"`` ramps from
        open-water speed down to the cap as RIO goes 0 to -10; more plausible
        physically, but it is our invention and produces transit times that
        cannot be defended by citing the circular.
    """

    ice_class: str
    open_water_speed_kn: float
    band_profile: str = "step"
    thresholds: RioThresholds = field(default_factory=RioThresholds)
    elevated_limits: dict[str, float] = field(default_factory=lambda: dict(ELEVATED_SPEED_LIMIT_KN))

    def __post_init__(self):
        if self.open_water_speed_kn <= 0:
            raise ValueError("open_water_speed_kn deve essere positiva")
        if self.band_profile not in ("step", "linear"):
            raise ValueError(f"band_profile sconosciuto: {self.band_profile}")

    @property
    def elevated_cap_kn(self) -> float | None:
        """Speed cap in the elevated band, or None if the band is not navigable.

        Below PC7 the middle band already requires special consideration, so
        there is no cap to apply — the cells are simply out.
        """
        if self.ice_class not in POLAR_CLASSES:
            return None
        cap = self.elevated_limits.get(self.ice_class)
        # A cap above the vessel's own cruising speed is not a restriction.
        return min(cap, self.open_water_speed_kn) if cap is not None else None

    def speed_kn(self, rio: np.ndarray) -> np.ndarray:
        """Speed in knots per cell. Zero where the cell cannot be entered."""
        rio = np.asarray(rio, dtype="float64")
        codes = classify_rio(rio, self.ice_class, self.thresholds)

        speed = np.zeros(rio.shape, dtype="float64")
        speed[codes == 0] = self.open_water_speed_kn

        cap = self.elevated_cap_kn
        if cap is not None:
            elevated = codes == 1
            if self.band_profile == "step":
                speed[elevated] = cap
            else:
                # Linear ramp across RIO in [-10, 0).
                frac = np.clip(rio / self.thresholds.elevated_min, 0.0, 1.0)
                speed[elevated] = (
                    self.open_water_speed_kn + (cap - self.open_water_speed_kn) * frac
                )[elevated]

        # codes 2 (special consideration) and 3 (unknown) stay at zero.
        return speed


@dataclass
class CostGrid:
    """Per-cell traversal cost, in seconds per metre travelled.

    Storing seconds-per-metre rather than seconds-per-cell keeps the cost
    independent of which of the 16 neighbours is used: the edge cost is this
    value times the ground distance of that particular step.
    """

    seconds_per_metre: np.ndarray
    passable: np.ndarray
    speed_kn: np.ndarray
    ice_class: str
    open_water_speed_kn: float

    @property
    def shape(self) -> tuple[int, int]:
        return self.seconds_per_metre.shape

    @property
    def passable_fraction(self) -> float:
        return float(self.passable.mean())

    def min_seconds_per_metre(self) -> float:
        """The fastest any cell can be crossed — the admissible heuristic bound."""
        return 1.0 / (self.open_water_speed_kn * KNOT_TO_MS)


def build_land_mask(
    distance_to_land: np.ndarray | None,
    shape: tuple[int, int],
    *,
    land_threshold: float | None = None,
    buffer_cells: int = 0,
) -> np.ndarray:
    """Boolean mask, True where the cell is land and must not be entered.

    ``distance_to_land`` from the AI4Arctic product is standard-scaled, so the
    absolute value carries no metres. What is reliable is its ordering: the
    minimum is attained on land. When no threshold is given, cells sitting at
    the array minimum are treated as land.

    SPEC §8.3 allows an external coastline (GSHHG / Natural Earth) instead. The
    dataset layer is used here because it is already co-registered with the
    scene; a mis-registered external coastline would be worse than none.
    """
    if distance_to_land is None:
        return np.zeros(shape, dtype=bool)

    d = np.asarray(distance_to_land, dtype="float64")
    if not np.isfinite(d).any():
        return np.zeros(shape, dtype=bool)

    thr = land_threshold if land_threshold is not None else np.nanmin(d)
    land = np.isfinite(d) & (d <= thr)

    if buffer_cells > 0 and land.any():
        from scipy.ndimage import binary_dilation

        # A route that shaves the coastline is technically optimal and
        # nautically absurd. The structuring element is a full 3x3 square, not
        # scipy's default cross: a cross dilates into a diamond and leaves the
        # diagonal clearance short of the requested buffer, which is the one
        # direction a grid route is most likely to cut the corner.
        land = binary_dilation(land, structure=np.ones((3, 3), dtype=bool), iterations=buffer_cells)

    return land


def build_cost_grid(
    rio: np.ndarray,
    speed_model: SpeedModel,
    *,
    land_mask: np.ndarray | None = None,
) -> CostGrid:
    """Turn a RIO raster into traversal costs."""
    speed = speed_model.speed_kn(rio)

    if land_mask is not None:
        speed = np.where(land_mask, 0.0, speed)

    passable = speed > 0
    speed_ms = speed * KNOT_TO_MS

    with np.errstate(divide="ignore", invalid="ignore"):
        spm = np.where(passable, 1.0 / speed_ms, IMPASSABLE)

    log.info(
        "costo %s: %.1f%% praticabile (%.1f%% a velocità piena, %.1f%% a velocità ridotta)",
        speed_model.ice_class,
        100 * passable.mean(),
        100 * np.mean(speed == speed_model.open_water_speed_kn),
        100 * np.mean(passable & (speed < speed_model.open_water_speed_kn)),
    )

    return CostGrid(
        seconds_per_metre=spm,
        passable=passable,
        speed_kn=speed,
        ice_class=speed_model.ice_class,
        open_water_speed_kn=speed_model.open_water_speed_kn,
    )


def speed_model_from_config(cfg: dict, ice_class: str) -> SpeedModel:
    """Build a :class:`SpeedModel` from ``configs/routing.yaml``."""
    sm = cfg.get("speed_model", {})
    vessel = cfg.get("vessel", {})
    return SpeedModel(
        ice_class=ice_class,
        open_water_speed_kn=float(vessel["open_water_speed_kn"]),
        band_profile=str(sm.get("band_profile", "step")),
        elevated_limits={k: float(v) for k, v in sm.get("elevated_speed_limit_kn", {}).items()}
        or dict(ELEVATED_SPEED_LIMIT_KN),
    )


def provisional_parameters(cfg: dict) -> list[str]:
    """Config keys marked ``[PROVVISORIO]``, for the UI to declare.

    SPEC rule 2: a value without a source is shown as such, not shown silently.
    Parsed from the comments so that the marker and the value cannot drift
    apart — there is only one place to change.
    """
    from pathlib import Path

    path = cfg.get("__path__")
    if not path or not Path(path).exists():
        return []

    out: list[str] = []
    pending: str | None = None
    for line in Path(path).read_text().splitlines():
        stripped = line.strip()
        if "[PROVVISORIO]" in stripped:
            if not stripped.startswith("#"):
                key = stripped.split(":", 1)[0].strip()
                out.append(key)
            else:
                pending = stripped
        elif pending and stripped and not stripped.startswith("#") and ":" in stripped:
            out.append(stripped.split(":", 1)[0].strip())
            pending = None
    return out
