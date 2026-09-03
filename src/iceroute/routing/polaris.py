"""POLARIS Risk Index Outcome from a predicted ice chart. SPEC §8.2.

Implements IMO MSC.1/Circ.1519, "Guidance on Methodologies for Assessing
Operational Capabilities and Limitations in Ice".

    RIO = C1*RIV1 + C2*RIV2 + ... + Cn*RIVn

where ``Ci`` is the concentration of ice type *i* in tenths and ``RIVi`` its
Risk Index Value for the vessel's ice class (circular §1.2.2).

The part that is easy to get wrong: the concentrations must sum to ten tenths,
and **open water is one of the terms**. A cell with 7/10 thick first-year ice
is also 3/10 ice-free, and ice-free carries RIV +3 for every ice class. Drop
that term and every partially-covered cell scores far more dangerous than it
is, which would push routes away from perfectly navigable water.

Every number here comes from the circular. See ``data/polaris_rv.csv`` and
DECISIONS.md D-007.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import numpy as np

FILL_VALUE = 255

# RIO is in tenths-of-concentration times risk value, so an all-open-water cell
# scores 10 * 3 = 30 and the scale runs from +30 down to -80.
RIO_OPEN_WATER = 30.0

# Sentinel for cells where RIO cannot be computed (no prediction available).
RIO_UNKNOWN = np.nan

POLAR_CLASSES = frozenset({"PC1", "PC2", "PC3", "PC4", "PC5", "PC6", "PC7"})


class Operation(Enum):
    """The three levels of operation defined in circular §1.3.2."""

    NORMAL = "normal"
    ELEVATED = "elevated"
    SPECIAL = "special"          # "operation subject to special consideration"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class RioThresholds:
    """Table 1.1 of the circular.

    The bands differ by vessel category. For PC1-PC7 the middle band is
    *elevated operational risk*, which is navigable under a speed limit. For
    anything below PC7 the same band already requires special consideration,
    which IceRoute treats as no-go.
    """

    elevated_min: float = -10.0   # RIO >= 0 normal; -10 <= RIO < 0 middle band
    special_max: float = -10.0    # RIO < -10 special consideration

    def classify(self, rio: float, ice_class: str) -> Operation:
        if not np.isfinite(rio):
            return Operation.UNKNOWN
        if rio >= 0:
            return Operation.NORMAL
        if rio >= self.elevated_min:
            return Operation.ELEVATED if ice_class in POLAR_CLASSES else Operation.SPECIAL
        return Operation.SPECIAL


# Table 1.2: recommended speed limits in elevated-risk conditions, in knots.
# Applies to Polar Class vessels only, since below PC7 the band is already
# "special consideration".
ELEVATED_SPEED_LIMIT_KN: dict[str, float] = {
    "PC1": 11.0,
    "PC2": 8.0,
    "PC3": 5.0,
    "PC4": 5.0,
    "PC5": 5.0,
    "PC6": 3.0,   # "Below PC5"
    "PC7": 3.0,
}


class PolarisTable:
    """The Risk Index Value matrix, loaded from ``data/polaris_rv.csv``."""

    def __init__(self, rows: dict[str, dict[str, int]], sources: dict[str, str]):
        self.rows = rows
        self.sources = sources

    @classmethod
    def load(cls, path: str | Path = "data/polaris_rv.csv") -> PolarisTable:
        path = Path(path)
        with path.open() as fh:
            lines = [ln for ln in fh if not ln.lstrip().startswith("#")]
        reader = csv.DictReader(lines)
        rows: dict[str, dict[str, int]] = {}
        sources: dict[str, str] = {}
        for rec in reader:
            name = (rec.pop("ice_class") or "").strip()
            if not name:
                continue
            sources[name] = (rec.pop("source", "") or "").strip()
            rows[name] = {k: int(v) for k, v in rec.items() if v not in (None, "")}
        if not rows:
            raise ValueError(f"{path}: nessuna riga di risk value trovata")
        return cls(rows, sources)

    @property
    def ice_classes(self) -> list[str]:
        return list(self.rows)

    @property
    def ice_types(self) -> list[str]:
        return list(next(iter(self.rows.values())))

    def riv(self, ice_class: str, ice_type: str) -> int:
        try:
            return self.rows[ice_class][ice_type]
        except KeyError as exc:
            raise KeyError(
                f"Nessun risk value per ({ice_class}, {ice_type}). "
                f"Classi: {self.ice_classes}. Tipi: {self.ice_types}."
            ) from exc

    def provisional_classes(self) -> list[str]:
        """Ice classes whose source line is missing or marked PROVVISORIO.

        SPEC rule 2 requires these to be surfaced in the UI rather than used
        silently.
        """
        return [
            c for c, s in self.sources.items() if not s or "PROVVISORIO" in s.upper()
        ]

    def riv_lookup(self, ice_class: str, sod_to_polaris: dict[int, str], n_classes: int = 6) -> np.ndarray:
        """Vector of RIV indexed by SOD class id, for vectorised lookup."""
        out = np.zeros(n_classes, dtype="float64")
        for sod_class, ice_type in sod_to_polaris.items():
            sod_class = int(sod_class)
            if 0 <= sod_class < n_classes:
                out[sod_class] = self.riv(ice_class, ice_type)
        return out


def compute_rio(
    sod: np.ndarray,
    ice_class: str,
    sod_to_polaris: dict[int, str],
    *,
    table: PolarisTable | None = None,
    sic: np.ndarray | None = None,
    mask_valid: np.ndarray | None = None,
    n_classes: int = 6,
    water_class: int = 0,
) -> np.ndarray:
    """RIO raster from a predicted SOD map.

    Parameters
    ----------
    sod
        Predicted stage-of-development class per cell.
    ice_class
        Vessel ice class, e.g. ``"PC6"``.
    sod_to_polaris
        Mapping from SOD class id to a column of the risk value table. See
        ``configs/data.yaml``; the mapping always picks the most conservative
        column of those a SOD class spans (DECISIONS.md D-008).
    sic
        Sea ice concentration in tenths, 0-10. **If omitted, full concentration
        is assumed wherever ice is predicted.** That is the pessimistic
        assumption and SPEC §8.2 requires it to be declared as a known
        limitation rather than buried — a 6/10 regime scored as 10/10 yields a
        route more cautious than reality demands.
    mask_valid
        Cells with a usable prediction. Elsewhere the result is NaN, never a
        number that looks navigable.

    Returns
    -------
    Float array of RIO. ``+30`` is entirely open water; NaN is unknown.
    """
    table = table or PolarisTable.load()
    riv = table.riv_lookup(ice_class, sod_to_polaris, n_classes)
    riv_open = float(table.riv(ice_class, "ice_free"))

    sod_arr = np.asarray(sod)
    known = (sod_arr != FILL_VALUE) & (sod_arr < n_classes)
    if mask_valid is not None:
        known &= np.asarray(mask_valid, dtype=bool)

    safe = np.where(known, sod_arr, 0).astype("int64")
    riv_cell = riv[safe]

    if sic is None:
        # No concentration information: ice cells are treated as fully covered.
        conc = np.where(safe == water_class, 0.0, 10.0)
    else:
        sic_arr = np.asarray(sic, dtype="float64")
        conc = np.clip(np.where(sic_arr == FILL_VALUE, 10.0, sic_arr), 0.0, 10.0)
        # Open water by ice type means zero ice concentration regardless of what
        # the SIC chart says, otherwise the two products can contradict.
        conc = np.where(safe == water_class, 0.0, conc)

    # The open-water remainder. Omitting this is the classic error.
    rio = conc * riv_cell + (10.0 - conc) * riv_open
    return np.where(known, rio, RIO_UNKNOWN)


def classify_rio(
    rio: np.ndarray, ice_class: str, thresholds: RioThresholds | None = None
) -> np.ndarray:
    """Map a RIO raster to operation levels, as an integer code array.

    ``0`` normal, ``1`` elevated, ``2`` special consideration, ``3`` unknown.
    Integer codes rather than an object array so the result can be used as a
    raster and rendered directly.
    """
    thresholds = thresholds or RioThresholds()
    rio = np.asarray(rio, dtype="float64")
    out = np.full(rio.shape, 3, dtype="uint8")

    finite = np.isfinite(rio)
    out[finite & (rio >= 0)] = 0
    mid = finite & (rio < 0) & (rio >= thresholds.elevated_min)
    out[mid] = 1 if ice_class in POLAR_CLASSES else 2
    out[finite & (rio < thresholds.special_max)] = 2
    return out


def is_navigable(rio: np.ndarray, ice_class: str, thresholds: RioThresholds | None = None) -> np.ndarray:
    """Cells a vessel of this class may enter.

    Normal and elevated-risk operation are navigable; special consideration is
    not. Unknown cells are **not** navigable: routing a ship through a cell
    where the model made no prediction is not a defensible thing to do.
    """
    return classify_rio(rio, ice_class, thresholds) <= 1


def speed_limit_kn(ice_class: str, operation: Operation) -> float | None:
    """Recommended speed cap in knots, or None if unrestricted / not navigable."""
    if operation is Operation.ELEVATED:
        return ELEVATED_SPEED_LIMIT_KN.get(ice_class)
    return None


def summarise(rio: np.ndarray, ice_class: str) -> dict:
    """Summary of a RIO raster, for the demo KPI card and metrics.json."""
    codes = classify_rio(rio, ice_class)
    finite = np.isfinite(rio)
    total = int(finite.sum())
    return {
        "ice_class": ice_class,
        "rio_min": float(np.nanmin(rio)) if total else None,
        "rio_max": float(np.nanmax(rio)) if total else None,
        "rio_mean": float(np.nanmean(rio)) if total else None,
        "fraction_normal": float((codes == 0).sum() / total) if total else 0.0,
        "fraction_elevated": float((codes == 1).sum() / total) if total else 0.0,
        "fraction_no_go": float((codes == 2).sum() / total) if total else 0.0,
        "fraction_unknown": float((codes == 3).mean()),
    }
