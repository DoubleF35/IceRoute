"""Read one AI4Arctic scene into memory as a typed structure.

The loader does exactly two non-obvious things, and both exist because of what
``inspect.py`` found on the real data (docs/decisions.md D-006):

1. **Validity comes from the target, not from the SAR.** The SAR channels pad
   the swath with a literal ``0.0`` — not NaN, no ``_FillValue`` attribute — on
   96.6% of the verified scene. Any mask derived from the SAR would treat that
   padding as measurement. The ice chart marks the same region invalid (NaN or
   255), so ``mask_valid`` is derived from the chart and the two were verified
   to coincide to within 0.2 percentage points.

2. **No normalisation happens here.** The dataset ships standard-scaled, the
   incidence angle included (it ranges roughly -1.8 to 0.9, so it is not
   degrees). Re-scaling would be a second, silent transformation on top of one
   we did not perform and cannot invert.

Memory note: a scene is 4996x5246. Held as float64 that is 210 MB per variable
and six variables would be 1.3 GB, so SAR channels are cast to float32 and
targets to uint8 on the way in.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

# The chart's ignore value. Verified against SIC/SOD/FLOE_LOOKUP['mask'] in the
# official toolkit and against the real file.
FILL_VALUE = 255

SCENE_RE = re.compile(r"^(\d{8})T(\d{6})_(dmi|cis)_prep")

# Parses e.g. "..._icechart_dmi_202010160825_North_RIC.nc" and the CIS variant.
_DMI_REGION_RE = re.compile(r"_icechart_dmi_\d{12}_([A-Za-z]+)_RIC")
_CIS_REGION_RE = re.compile(r"_icechart_cis_(SGRDI[A-Z]*)_")


@dataclass(frozen=True)
class SceneMeta:
    """Identity and provenance of a scene."""

    scene_id: str
    acquired: datetime | None
    ice_service: str  # 'dmi' or 'cis'
    region: str  # e.g. 'dmi:North', 'cis:SGRDIEA', or 'unknown'
    pixel_spacing_m: float
    original_id: str
    flip: int  # global attribute; see geocode.py
    shape: tuple[int, int]

    @property
    def platform(self) -> str:
        """S1A or S1B, recoverable only from original_id."""
        return self.original_id[:3] if self.original_id[:2] == "S1" else "unknown"


@dataclass
class Scene:
    """One AI4Arctic scene, ready for patch extraction.

    All 2D arrays share the same shape. ``mask_valid`` is the single source of
    truth for which pixels may be used: never index the SAR without it.
    """

    hh: np.ndarray  # float32, standard-scaled  (nersc_sar_primary)
    hv: np.ndarray  # float32, standard-scaled  (nersc_sar_secondary)
    incidence_angle: np.ndarray  # float32, standard-scaled — NOT degrees
    distance_to_land: np.ndarray  # float32, standard-scaled
    sod: np.ndarray  # uint8, classes 0-5, FILL_VALUE where invalid
    sic: np.ndarray  # uint8, tenths 0-10, FILL_VALUE where invalid
    mask_valid: np.ndarray  # bool — True where the ice chart annotates
    lat_grid: np.ndarray  # float64, low-resolution (21x21)
    lon_grid: np.ndarray  # float64, low-resolution (21x21)
    meta: SceneMeta
    extras: dict[str, np.ndarray] = field(default_factory=dict)

    @property
    def shape(self) -> tuple[int, int]:
        return self.hh.shape  # type: ignore[return-value]

    @property
    def valid_fraction(self) -> float:
        return float(self.mask_valid.mean())

    def channels(self, names: list[str]) -> np.ndarray:
        """Stack named channels into ``(C, H, W)`` float32 for the model."""
        available = {
            "hh": self.hh,
            "hv": self.hv,
            "incidence_angle": self.incidence_angle,
            "distance_to_land": self.distance_to_land,
        }
        missing = [n for n in names if n not in available]
        if missing:
            raise KeyError(f"Channels not available: {missing}. Allowed: {sorted(available)}")
        return np.stack([available[n] for n in names], axis=0)

    def summary(self) -> str:
        counts = class_counts(self.sod, self.mask_valid)
        dist = " ".join(
            f"{k}:{100 * v / max(sum(counts.values()), 1):.0f}%" for k, v in sorted(counts.items())
        )
        return (
            f"{self.meta.scene_id}  {self.shape[0]}x{self.shape[1]}  "
            f"{self.meta.region}  {self.meta.acquired:%Y-%m-%d}\n"
            f"  annotated: {100 * self.valid_fraction:.1f}%  ·  SOD  {dist}"
        )


def class_counts(labels: np.ndarray, mask_valid: np.ndarray) -> dict[int, int]:
    """Per-class pixel counts over valid pixels only."""
    vals, counts = np.unique(labels[mask_valid], return_counts=True)
    return {int(v): int(c) for v, c in zip(vals, counts, strict=True)}


def parse_region(original_id: str, ice_service: str) -> str:
    """Recover the geographic region from the original product name.

    The ready-to-train filename carries only a date and dmi/cis, so the region
    has to come from here. Two grammars, one per ice service.
    """
    if m := _DMI_REGION_RE.search(original_id):
        return f"dmi:{m.group(1)}"
    if m := _CIS_REGION_RE.search(original_id):
        return f"cis:{m.group(1)}"
    return f"{ice_service}:unknown" if ice_service else "unknown"


def _read(ds: Any, name: str, dtype: str, fill: float | int | None = None) -> np.ndarray:
    """Read a variable, filling masked entries with ``fill``."""
    raw = ds.variables[name][:]
    if np.ma.isMaskedArray(raw):
        if fill is None:
            arr = np.ma.filled(raw.astype("float64"), np.nan)
        else:
            arr = np.ma.filled(raw, fill)
    else:
        arr = np.asarray(raw)
    return arr.astype(dtype, copy=False)


def _invalid_mask(raw: np.ndarray) -> np.ndarray:
    """True where a target array carries no annotation.

    Covers both conventions: netCDF4 hands back NaN for masked entries, and the
    underlying encoding uses 255.
    """
    invalid = ~np.isfinite(raw)
    invalid |= raw == FILL_VALUE
    return invalid


def reference_path(path: str | Path) -> Path:
    """Where the AI4Arctic challenge test set keeps a scene's labels.

    The official test scenes ship without targets — that is what makes them a
    test set — and the charts live alongside in ``*_prep_reference.nc``.
    Verified on disk rather than taken from documentation (docs/decisions.md D-006).
    """
    path = Path(path)
    return path.with_name(path.name.replace("_prep.nc", "_prep_reference.nc"))


def load_scene(
    path: str | Path,
    config: dict,
    *,
    load_extras: bool = False,
    labels_from: str | Path | None = None,
) -> Scene:
    """Load a scene using the variable names in ``configs/data.yaml``.

    Parameters
    ----------
    path
        Path to a ``*_prep.nc`` file.
    config
        Parsed ``configs/data.yaml``. Variable names are read from it so that a
        rename in the dataset is a config change, never a code change.
    load_extras
        Also load the AMSR2 and ERA5 blocks into ``Scene.extras``. Off by
        default: the base model does not use them and they are 28 more arrays.
    labels_from
        Read SIC and SOD from this file instead of from ``path``. Needed for the
        official challenge test set, whose scenes carry no targets. The
        reference file uses the same variable names and full resolution, so
        nothing else changes.
    """
    import netCDF4

    path = Path(path)
    var = config["variables"]
    geom = var["geometry"]
    tgt = var["targets"]

    with netCDF4.Dataset(path, "r") as ds:
        # --- targets first: they define validity -------------------------
        if labels_from is not None:
            ref = Path(labels_from)
            if not ref.exists():
                raise FileNotFoundError(
                    f"Labels missing: {ref}. The official test scenes do not contain "
                    f"them; they live in the _prep_reference.nc file alongside."
                )
            with netCDF4.Dataset(ref, "r") as rds:
                sod_raw = _read(rds, tgt["sod"], "float64")
                sic_raw = _read(rds, tgt["sic"], "float64")
            if sod_raw.shape != _read(ds, var["sar_hh"], "float32").shape:
                raise ValueError(
                    f"The label grid {sod_raw.shape} does not match the SAR grid "
                    f"in {path.name}: these are two different scenes."
                )
        else:
            sod_raw = _read(ds, tgt["sod"], "float64")
            sic_raw = _read(ds, tgt["sic"], "float64")

        invalid = _invalid_mask(sod_raw)
        mask_valid = ~invalid

        sod = np.where(invalid, FILL_VALUE, sod_raw).astype("uint8")
        sic = np.where(_invalid_mask(sic_raw), FILL_VALUE, sic_raw).astype("uint8")

        # --- inputs -------------------------------------------------------
        hh = _read(ds, var["sar_hh"], "float32")
        hv = _read(ds, var["sar_hv"], "float32")
        inc = _read(ds, var["incidence_angle"], "float32")
        dist = _read(ds, var["distance_to_land"], "float32")

        lat = _read(ds, geom["latitude"], "float64")
        lon = _read(ds, geom["longitude"], "float64")

        # --- metadata -----------------------------------------------------
        gattr = config.get("global_attrs", {})

        def attr(key: str, default: str = "") -> str:
            name = gattr.get(key, key)
            return str(getattr(ds, name, default))

        scene_id = attr("scene_id", path.name)
        original_id = attr("original_id")
        ice_service = attr("ice_service")

        try:
            spacing = float(attr("pixel_spacing", "80"))
        except ValueError:
            spacing = 80.0
        try:
            flip = int(float(attr("flip", "0")))
        except ValueError:
            flip = 0

        extras: dict[str, np.ndarray] = {}
        if load_extras:
            for group in ("passive_amsr2", "weather_era5"):
                for name in var.get(group, []):
                    if name in ds.variables:
                        extras[name] = _read(ds, name, "float32")

    acquired = None
    if m := SCENE_RE.match(scene_id):
        acquired = datetime.strptime(m.group(1) + m.group(2), "%Y%m%d%H%M%S")

    meta = SceneMeta(
        scene_id=scene_id,
        acquired=acquired,
        ice_service=ice_service,
        region=parse_region(original_id, ice_service),
        pixel_spacing_m=spacing,
        original_id=original_id,
        flip=flip,
        shape=hh.shape,  # type: ignore[arg-type]
    )

    scene = Scene(
        hh=hh,
        hv=hv,
        incidence_angle=inc,
        distance_to_land=dist,
        sod=sod,
        sic=sic,
        mask_valid=mask_valid,
        lat_grid=lat,
        lon_grid=lon,
        meta=meta,
        extras=extras,
    )

    _sanity_check(scene)
    return scene


def _sanity_check(scene: Scene) -> None:
    """Fail loudly on the corruptions that would otherwise surface at training.

    Every check here corresponds to something that was actually observed in the
    data or that would silently poison the loss.
    """
    shapes = {
        "hh": scene.hh.shape,
        "hv": scene.hv.shape,
        "incidence_angle": scene.incidence_angle.shape,
        "sod": scene.sod.shape,
        "mask_valid": scene.mask_valid.shape,
    }
    if len(set(shapes.values())) != 1:
        raise ValueError(f"Misaligned shapes in {scene.meta.scene_id}: {shapes}")

    # A label outside 0-5 that is not the fill value means the SOD lookup was
    # applied wrongly, and CrossEntropy would index out of bounds.
    labels = scene.sod[scene.mask_valid]
    if labels.size:
        bad = np.unique(labels[(labels > 5) & (labels != FILL_VALUE)])
        if bad.size:
            raise ValueError(
                f"{scene.meta.scene_id}: SOD classes out of range on valid pixels: {bad.tolist()}. "
                f"Expected 0-5, or {FILL_VALUE} with mask_valid=False."
            )

    # The invariant that lets us mask on the target alone (D-006).
    if scene.mask_valid.any():
        if (scene.sod[scene.mask_valid] == FILL_VALUE).any():
            raise ValueError(
                f"{scene.meta.scene_id}: pixels with SOD={FILL_VALUE} but mask_valid=True."
            )
        if not np.isfinite(scene.hh[scene.mask_valid]).all():
            raise ValueError(f"{scene.meta.scene_id}: NaN in the SAR on pixels marked valid.")


def load_config(path: str | Path = "configs/data.yaml") -> dict:
    import yaml

    return yaml.safe_load(Path(path).read_text())


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Load a scene and print a summary.")
    ap.add_argument("scene", type=Path)
    ap.add_argument("--config", type=Path, default=Path("configs/data.yaml"))
    args = ap.parse_args()

    sc = load_scene(args.scene, load_config(args.config))
    print(sc.summary())
    print(
        f"  platform: {sc.meta.platform}  ·  spacing: {sc.meta.pixel_spacing_m:.0f} m  ·  flip: {sc.meta.flip}"
    )
    print(
        f"  geo grid: {sc.lat_grid.shape}  lat {sc.lat_grid.min():.2f}..{sc.lat_grid.max():.2f}  "
        f"lon {sc.lon_grid.min():.2f}..{sc.lon_grid.max():.2f}"
    )
