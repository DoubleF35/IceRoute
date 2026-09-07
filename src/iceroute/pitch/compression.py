"""Measure the downlink compression ratio. SPEC §10.2. Closes P-06.

The claim is that IceRoute downlinks *mask + route* instead of the raw scene,
at roughly 200:1. This module measures the ratio on real bytes and, if the
number is not 200:1, the number in the pitch becomes the measured one — SPEC
§10.2, no exceptions.

Choosing the numerator honestly
-------------------------------
The ratio is only as trustworthy as what sits on top of the fraction, and there
are several defensible "raw scene sizes" that differ by an order of magnitude.
Picking the largest would inflate the result for free, so all are reported and
the *smallest defensible* one is designated:

* ``sar_channels_png`` — the two polarisation channels compressed losslessly.
  **This is the designated numerator.** Any real payload compresses before
  transmitting, so comparing against uncompressed data would be beating a
  baseline nobody would fly. It matters concretely here: measured scenes are
  7-58% exact-zero swath padding, and compression removes that at once. Quoting
  the uncompressed ratio would be charging the baseline full price for
  transmitting nothing.
* ``sar_channels_uncompressed`` — the same channels at 2 bytes per pixel.
  Reported for reference; roughly 1.7x more generous to us.
* ``netcdf_file`` — the AI4Arctic ``.nc`` on disk. Reported for context and
  *not* used: it bundles AMSR2 brightness temperatures, ERA5 fields and
  ancillary grids that no CubeSat would downlink.

Why the mask source must be declared
------------------------------------
This is the trap in this measurement, and it is not obvious.

An ice chart is drawn by a human analyst as **polygons**, so the ground-truth
mask is piecewise constant over large areas and PNG compresses it
extraordinarily well — far below its own per-pixel entropy, because the spatial
structure is what is being exploited. A U-Net's output is not like that: it is
pixel-wise, speckled at boundaries, and irregular. It will compress
*substantially worse*.

Measuring the ratio on ground truth and presenting it as the operational number
therefore overstates it, by an amount only a real predicted mask can reveal.
``measure_scene`` requires ``mask_source`` to be declared and marks every result
accordingly; ``summarise`` refuses to hand back an unqualified headline for
ground-truth masks.

The denominator
---------------
Mask + route + metadata, exactly what goes down. The mask is written as a
paletted PNG: lossless, 3 bits of real information per pixel, and decodable by
anything. Run-length encoding would likely beat it — SPEC §10.2 names it as the
improvement to mention if the ratio disappoints — so a run-length estimate is
computed alongside for comparison, and clearly marked as an estimate rather
than a shipped format.
"""

from __future__ import annotations

import io
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

log = logging.getLogger("compression")

# Bytes per pixel a SAR payload would send per polarisation channel. Sentinel-1
# GRD products are 16-bit unsigned integers; AI4Arctic redistributes them
# standard-scaled as float32, which is a property of the research dataset, not
# of the sensor. Counting 4 bytes here would inflate the ratio 2x for free.
SAR_BYTES_PER_PIXEL = 2
N_SAR_CHANNELS = 2


@dataclass
class CompressionResult:
    """Every quantity behind the headline ratio, so it can be recomputed."""

    scene_id: str
    shape: tuple[int, int]
    n_pixels: int
    raw_sar_bytes: int
    raw_netcdf_bytes: int
    raw_sar_png_bytes: int
    mask_png_bytes: int
    route_bytes: int
    metadata_bytes: int
    n_classes: int
    mask_source: str = "predicted"
    coverage: float = float("nan")
    sar_zero_fraction: float = float("nan")
    mask_rle_estimate_bytes: int = 0
    notes: list[str] = field(default_factory=list)

    @property
    def payload_bytes(self) -> int:
        return self.mask_png_bytes + self.route_bytes + self.metadata_bytes

    @property
    def ratio(self) -> float:
        """The designated ratio: losslessly compressed SAR over payload."""
        return self.raw_sar_png_bytes / max(self.payload_bytes, 1)

    @property
    def ratio_vs_uncompressed(self) -> float:
        """Against raw bytes. More generous to us; reported, not quoted."""
        return self.raw_sar_bytes / max(self.payload_bytes, 1)

    @property
    def is_provisional(self) -> bool:
        """True when the ratio was measured on an analyst-drawn chart."""
        return self.mask_source != "predicted"

    def as_dict(self) -> dict:
        return {
            "scene_id": self.scene_id,
            "shape": list(self.shape),
            "n_pixels": self.n_pixels,
            "mask_source": self.mask_source,
            "provisional": self.is_provisional,
            "ice_chart_coverage": None if np.isnan(self.coverage) else round(self.coverage, 4),
            "sar_zero_fraction": (
                None if np.isnan(self.sar_zero_fraction) else round(self.sar_zero_fraction, 4)
            ),
            "numerator": {
                "designated": "sar_channels_png",
                "sar_channels_png": self.raw_sar_png_bytes,
                "sar_channels_uncompressed": self.raw_sar_bytes,
                "netcdf_file": self.raw_netcdf_bytes,
                "bytes_per_pixel_per_channel": SAR_BYTES_PER_PIXEL,
                "n_channels": N_SAR_CHANNELS,
            },
            "denominator": {
                "mask_png": self.mask_png_bytes,
                "route_geojson": self.route_bytes,
                "metadata": self.metadata_bytes,
                "total": self.payload_bytes,
                "mask_rle_estimate": self.mask_rle_estimate_bytes,
            },
            "ratio": round(self.ratio, 1),
            "ratio_vs_uncompressed_sar": round(self.ratio_vs_uncompressed, 1),
            "n_classes": self.n_classes,
            "notes": self.notes,
        }


def encode_mask_png(mask: np.ndarray, n_classes: int) -> bytes:
    """Lossless paletted PNG of a class map.

    Mode ``P`` rather than ``L``: an 8-bit greyscale image of class ids 0-5 is
    almost black and compresses similarly, but a palette makes the file
    self-describing and directly viewable, which matters for something a ground
    station has to interpret without our code.
    """
    from PIL import Image

    if mask.dtype != np.uint8:
        mask = mask.astype("uint8")

    img = Image.fromarray(mask, mode="P")
    # A palette that is distinguishable rather than pretty; the demo has its own.
    palette = []
    for i in range(256):
        if i < n_classes:
            v = int(255 * i / max(n_classes - 1, 1))
            palette += [v, 90, 255 - v]
        else:
            palette += [0, 0, 0]
    img.putpalette(palette)

    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=True, compress_level=9)
    return buf.getvalue()


def encode_sar_png(channels: np.ndarray) -> int:
    """Bytes for the SAR channels compressed losslessly. The strict numerator.

    The float32 standard-scaled values are requantised to 16 bits over their own
    range before compressing, because that is the precision a payload would
    actually send. Compressing float32 directly would measure the dataset's
    storage format instead of the satellite's.
    """
    from PIL import Image

    total = 0
    for ch in channels:
        finite = np.isfinite(ch)
        if not finite.any():
            continue
        lo, hi = float(ch[finite].min()), float(ch[finite].max())
        scale = (hi - lo) or 1.0
        q = np.clip((ch - lo) / scale, 0, 1)
        q = (q * 65535).astype("uint16")

        buf = io.BytesIO()
        # No explicit mode: passing one to reinterpret dtypes is deprecated and
        # removed in Pillow 13 (October 2026 — the same month as the pitch).
        # fromarray infers I;16 from a uint16 array.
        Image.fromarray(q).save(buf, format="PNG", optimize=True, compress_level=9)
        total += buf.getbuffer().nbytes
    return total


def rle_estimate_bytes(mask: np.ndarray) -> int:
    """Size of a row-wise run-length encoding: 2 bytes run length + 1 byte class.

    An estimate of an alternative, not a shipped format. SPEC §10.2 asks for
    this comparison so that "what would improve the ratio" has a number attached
    instead of being a hand-wave.
    """
    flat = mask.ravel()
    if flat.size == 0:
        return 0
    # Count runs without materialising them.
    n_runs = 1 + int((flat[1:] != flat[:-1]).sum())
    return n_runs * 3


def downlink_route(collection: dict | None) -> dict | None:
    """Strip a demo GeoJSON down to what the satellite would actually send.

    Every ``data/demo/routes/*.geojson`` holds two tracks: the computed route
    and the geodesic it is compared against. The comparison exists to argue a
    point to a human on the ground — a satellite has no reason to transmit the
    straight line between two points whose coordinates the recipient supplied.

    Counting it as payload measured the wrong thing, and not by a little: the
    baseline is sampled at every cell it crosses, so it ran to 26 KB against
    430 bytes of real route. The error was conservative — it understated the
    ratio — but a downlink figure that includes a file the downlink does not
    carry is not a downlink figure.

    Used by both the measurement and the UI panel that shows the payload, so
    the bytes on screen and the bytes in the deck cannot disagree.
    """
    if not collection:
        return None
    kept = [
        f for f in collection.get("features", [])
        if (f.get("properties") or {}).get("kind") != "baseline"
    ]
    return {**collection, "features": kept} if kept else None


def measure_scene(
    scene_id: str,
    mask: np.ndarray,
    sar_channels: np.ndarray,
    *,
    n_classes: int,
    mask_source: str,
    netcdf_path: Path | None = None,
    route_geojson: dict | None = None,
    metadata: dict | None = None,
    coverage: float = float("nan"),
) -> CompressionResult:
    """Measure the ratio for one scene against a given mask.

    Parameters
    ----------
    mask_source
        ``"predicted"`` for a model output, ``"ground_truth"`` for an analyst's
        ice chart. Required, with no default, because the two give very
        different answers and the difference is invisible in the result. See
        the module docstring.
    """
    if mask_source not in {"predicted", "ground_truth"}:
        raise ValueError(
            f"mask_source deve essere 'predicted' o 'ground_truth', non {mask_source!r}. "
            f"Il rapporto misurato sulle due cose è diverso e va etichettato."
        )
    if mask.shape != sar_channels.shape[1:]:
        raise ValueError(
            f"Maschera {mask.shape} e canali SAR {sar_channels.shape[1:]} non coincidono: "
            f"il rapporto confronterebbe due scene diverse."
        )

    n_px = int(mask.size)
    raw_sar = n_px * SAR_BYTES_PER_PIXEL * N_SAR_CHANNELS

    mask_png = encode_mask_png(mask, n_classes)
    sar_png = encode_sar_png(sar_channels)
    zero_frac = float((sar_channels[0] == 0.0).mean())

    route_bytes = (
        len(json.dumps(route_geojson, separators=(",", ":")).encode()) if route_geojson else 0
    )
    meta_bytes = len(json.dumps(metadata or {}, separators=(",", ":")).encode())

    notes = []
    if route_geojson is None:
        notes.append(
            "Nessuna rotta inclusa: il denominatore è la sola maschera. "
            "Il rapporto reale sarà leggermente più basso."
        )
    if mask_source == "ground_truth":
        notes.append(
            "PROVVISORIO: misurato sull'ice chart, non su una maschera predetta. "
            "L'ice chart è disegnata a poligoni dall'analista, quindi si comprime molto "
            "meglio di quanto farà l'uscita della U-Net. Questo numero è un LIMITE "
            "SUPERIORE, non il valore operativo."
        )
    if zero_frac > 0.10:
        notes.append(
            f"Il {zero_frac:.0%} del SAR è padding a 0.0 esatto. Il numeratore designato "
            f"è quello compresso proprio per non far pagare alla baseline la "
            f"trasmissione del nulla."
        )

    result = CompressionResult(
        scene_id=scene_id,
        shape=tuple(mask.shape),
        n_pixels=n_px,
        raw_sar_bytes=raw_sar,
        raw_netcdf_bytes=netcdf_path.stat().st_size if netcdf_path and netcdf_path.exists() else 0,
        raw_sar_png_bytes=sar_png,
        mask_png_bytes=len(mask_png),
        route_bytes=route_bytes,
        metadata_bytes=meta_bytes,
        n_classes=n_classes,
        mask_source=mask_source,
        coverage=coverage,
        sar_zero_fraction=zero_frac,
        mask_rle_estimate_bytes=rle_estimate_bytes(mask),
        notes=notes,
    )

    if result.mask_rle_estimate_bytes < result.mask_png_bytes:
        gain = result.mask_png_bytes / max(result.mask_rle_estimate_bytes, 1)
        result.notes.append(
            f"Un RLE per righe starebbe in {result.mask_rle_estimate_bytes / 1024:.0f} KB "
            f"contro i {result.mask_png_bytes / 1024:.0f} KB del PNG ({gain:.1f}× meglio): "
            f"è la strada indicata dalla SPEC §10.2 se il rapporto va migliorato."
        )
    return result


def summarise(results: list[CompressionResult]) -> dict:
    """Aggregate over scenes. Reports the range, not just the mean.

    A single averaged ratio hides that a scene of open water compresses far
    better than a scene of broken ice — and the demo will show one specific
    scene, whose number a viewer can check against this.
    """
    if not results:
        return {"error": "Nessuna scena misurata."}

    ratios = [r.ratio for r in results]
    loose = [r.ratio_vs_uncompressed for r in results]
    provisional = [r for r in results if r.is_provisional]

    out = {
        "n_scenes": len(results),
        "ratio_mean": round(float(np.mean(ratios)), 1),
        "ratio_median": round(float(np.median(ratios)), 1),
        "ratio_min": round(float(np.min(ratios)), 1),
        "ratio_max": round(float(np.max(ratios)), 1),
        "ratio_vs_uncompressed_median": round(float(np.median(loose)), 1),
        "designated_numerator": "sar_channels_png",
        "mask_sources": sorted({r.mask_source for r in results}),
        "provisional": bool(provisional),
        "per_scene": [r.as_dict() for r in results],
    }
    if provisional:
        out["provisional_reason"] = (
            f"{len(provisional)}/{len(results)} scene misurate sull'ice chart invece che "
            f"su una maschera predetta. Numero da trattare come limite superiore finché "
            f"non si rimisura con il modello addestrato (P-06)."
        )
    return out


def verdict(summary: dict, claimed: float = 200.0) -> str:
    """Compare the measurement to the claim. SPEC §10.2 leaves no discretion."""
    if "error" in summary:
        return summary["error"]

    measured = summary["ratio_median"]
    if summary.get("provisional"):
        # Refuse to hand back a quotable headline from an analyst-drawn chart.
        return (
            f"Rapporto {measured:.0f}:1 misurato sull'ICE CHART, non su una maschera "
            f"predetta: è un LIMITE SUPERIORE e non va nel pitch. L'ice chart è a "
            f"poligoni e si comprime molto meglio dell'uscita di una rete. "
            f"Rimisurare dopo il training. (P-06 resta aperta.)"
        )

    if measured >= claimed:
        return (
            f"Rapporto misurato {measured:.0f}:1, sopra i {claimed:.0f}:1 dichiarati, "
            f"già contro una baseline SAR compressa senza perdita. Si usa il misurato."
        )
    return (
        f"Rapporto misurato {measured:.0f}:1 contro i {claimed:.0f}:1 dichiarati. "
        f"SPEC §10.2: nel pitch va **{measured:.0f}:1**, non {claimed:.0f}:1."
    )
