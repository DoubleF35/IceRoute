"""The two routes, drawn. SPEC §9.3.

Every ``data/demo/routes/*.geojson`` has always held two tracks — ours and the
geodesic a master would steer with only a paper chart — and until now nothing
drew either of them. The comparison was computed, stored, and never shown.

Why this is a module and not four lines in the UI
-------------------------------------------------
Because the coordinate convention is the whole problem, and it needs one home.

A scene lives on **two different grids**. The SAR and the mask PNGs are the raw
swath, decimated (≈1311×1249). The routes are computed after geocoding, on a
500 m EPSG:3413 grid (≈976×956) — a different shape, a different projection,
and not a crop of the other. The previous attempt at this overlay scaled route
cells by the *raw* scene shape and looked up a ``cells`` property that nothing
ever wrote, so it silently drew nothing at all. A figure whose entire job was
"IceRoute finds a route the direct track cannot take" shipped as a bare mask
with a caption, and nobody noticed, because there is no way to notice: it does
not raise, it does not warn, it just draws no line.

So the overlay carries its own ``grid_shape`` and :func:`draw` **refuses** to
paint an image that does not match it. Getting this wrong should be loud.

Two rules about what gets drawn
-------------------------------
* **The line is the full A\\* path; the dots are what actually goes down.** The
  simplified waypoint list is the downlink product, and showing both makes the
  compression claim visible instead of asserted: this whole voyage, those eight
  points.
* **A forbidden cell gets a marker.** "POLARIS blocks this track" is an
  assertion; twelve crosses on twelve pixels is evidence. The count already
  lived in the index — the positions are what make it arguable on stage.

Colours are chosen against the two backdrops this can be drawn on, not for
looks. The ice mask uses deep blue / cyan / white and RIO uses green / yellow /
red, so both of those hue families are unusable for a line. Magenta appears in
neither. Every stroke also gets a dark casing underneath, the way road maps do
it, which is what makes a thin line survive an arbitrary background on a washed
out projector.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - typing only
    from PIL.Image import Image

log = logging.getLogger(__name__)

OVERLAY_DIR = "overlays"

# Casing first, core on top. Widths are for a 1:1 grid and scale with the image.
CASING = (8, 12, 20)  # near-black, the demo background
COL_ICEROUTE = (255, 45, 149)  # magenta: absent from both palettes
COL_BASELINE = (245, 245, 245)  # white, dashed — reads as a line, not a region
COL_NO_GO = (255, 32, 32)  # red: the cells POLARIS forbids
COL_ENDPOINT = (255, 255, 255)

DASH_ON, DASH_OFF = 14, 10


class GridMismatch(ValueError):
    """The image and the overlay are not on the same grid.

    Raised rather than tolerated. The failure this replaces was a route drawn
    on the wrong grid, or not drawn at all, with no diagnostic either way.
    """


# ---------------------------------------------------------------------------
# Build — runs in precompute, where the grid is still in hand
# ---------------------------------------------------------------------------


def separation(ours: list, base: list, grid) -> dict:
    """How far the two tracks visibly pull apart.

    The deck's route figure has one job — show that IceRoute goes somewhere the
    direct track cannot — and a scene where the two lines sit on top of each
    other does not do it, however many cells POLARIS forbids underneath. That
    is a property of the picture, so it gets measured rather than eyeballed,
    and whichever scene leads the figure is then a consequence of the data
    instead of whichever one happened to come first in the index.

    Reported as a fraction of the image diagonal as well as in kilometres:
    what makes a slide readable is angular separation on the wall, not ground
    distance, and the two do not rank scenes the same way.
    """
    import numpy as np

    a, b = np.asarray(ours, dtype=float), np.asarray(base, dtype=float)
    if a.size == 0 or b.size == 0:
        return {}
    px = float(np.sqrt(((a[:, None, :] - b[None, :, :]) ** 2).sum(-1)).min(1).max())
    rows, cols = grid.shape
    return {
        "max_px": round(px, 1),
        "max_km": round(px * grid.resolution_m / 1000.0, 1),
        "pct_diagonal": round(100 * px / (rows**2 + cols**2) ** 0.5, 2),
    }


def build(grid, route, simplified, baseline, start, goal) -> dict:
    """Everything needed to draw one (scene, vessel class), in grid pixels.

    Kept out of the GeoJSON on purpose. That file is what the satellite sends
    down and its size is part of a measured claim; a per-cell list would grow
    it by two orders of magnitude to carry something only the ground UI wants.
    """

    def rc(lon: float, lat: float) -> list[int] | None:
        try:
            r, c = grid.index_of(lon, lat)
        except ValueError:
            return None
        return [int(r), int(c)]

    waypoints = [p for p in (rc(lon, lat) for lon, lat in simplified.lonlat) if p]

    return {
        "separation": separation(route.cells, baseline.route.cells, grid),
        "grid_shape": [int(grid.shape[0]), int(grid.shape[1])],
        "resolution_m": float(grid.resolution_m),
        "iceroute": {
            "path": [[int(r), int(c)] for r, c in route.cells],
            "waypoints": waypoints,
        },
        "baseline": {
            "path": [[int(r), int(c)] for r, c in baseline.route.cells],
            "no_go": [[int(r), int(c)] for r, c in baseline.no_go_cells],
            "no_data": [[int(r), int(c)] for r, c in baseline.no_data_cells],
            "feasible": bool(baseline.feasible),
        },
        "start": [int(start[0]), int(start[1])],
        "goal": [int(goal[0]), int(goal[1])],
    }


def load(path: Path) -> dict:
    # ``is_file`` rather than ``exists``: callers build this path from an index
    # entry that may carry no overlay name, which lands on the directory.
    return json.loads(path.read_text()) if path.is_file() else {}


# ---------------------------------------------------------------------------
# Draw
# ---------------------------------------------------------------------------


def _project(overlay: dict, img: Image) -> tuple[float, float]:
    """Grid cell -> image pixel scale, refusing a grid that is not this one.

    The image may legitimately be a decimated version of the grid, so the two
    scale factors only have to *agree*, not be 1. What they must not do is
    disagree, which is what happens when the overlay is paired with a raster
    from the raw swath instead of the geocoded one.
    """
    rows, cols = overlay.get("grid_shape") or (0, 0)
    if not rows or not cols:
        raise GridMismatch("l'overlay non dichiara grid_shape: non è disegnabile")
    sx, sy = img.width / cols, img.height / rows
    # 1%, not a round 5%. A legitimately decimated raster is ceil(n/k) on each
    # axis, so the two scale factors differ only by that rounding — under 0.5%
    # even for aggressive factors. The mismatch this is really guarding against
    # (a geocoded route paired with the raw swath) came out at 2.8%, which is
    # uncomfortably close: the margin should sit well below the thing it
    # catches, not just above the thing it tolerates.
    if abs(sx - sy) > 0.01 * max(sx, sy):
        raise GridMismatch(
            f"immagine {img.width}x{img.height} incompatibile con la griglia "
            f"{cols}x{rows}: fattori di scala {sx:.3f} e {sy:.3f}. Quasi certamente "
            f"è un raster della scena grezza invece che di quella geocodificata."
        )
    return sx, sy


def _pts(cells, sx: float, sy: float) -> list[tuple[float, float]]:
    return [(c * sx + sx / 2, r * sy + sy / 2) for r, c in cells]


def _dashed(pts: list[tuple[float, float]], on: float, off: float) -> list[list]:
    """Split a polyline into dashes of constant on-screen length."""
    runs, cur, carry, drawing = [], [pts[0]], 0.0, True
    for a, b in zip(pts, pts[1:], strict=False):
        seg = ((b[0] - a[0]) ** 2 + (b[1] - a[1]) ** 2) ** 0.5
        if seg == 0:
            continue
        t = 0.0
        while t < seg:
            want = (on if drawing else off) - carry
            step = min(want, seg - t)
            t += step
            carry += step
            p = (a[0] + (b[0] - a[0]) * t / seg, a[1] + (b[1] - a[1]) * t / seg)
            if drawing:
                cur.append(p)
            if carry >= (on if drawing else off) - 1e-9:
                if drawing and len(cur) > 1:
                    runs.append(cur)
                drawing, carry = not drawing, 0.0
                cur = [p]
    if drawing and len(cur) > 1:
        runs.append(cur)
    return runs


def _stroke(d, pts, colour, width: int, *, casing: int = 3) -> None:
    if len(pts) < 2:
        return
    if casing:
        d.line(pts, fill=CASING, width=width + 2 * casing, joint="curve")
    d.line(pts, fill=colour, width=width, joint="curve")


def no_go_runs(cells: list, gap: int = 2) -> list[list]:
    """Forbidden cells grouped into the contiguous stretches they form.

    ``geodesic_baseline`` collects them walking the track, so consecutive
    entries are consecutive along the route and adjacency is enough to group.

    This exists because drawing one marker per cell is both ugly and wrong. A
    scene where the direct track ploughs through a floe yields three hundred
    forbidden cells; three hundred overlapping markers render as a black smear,
    and they misdescribe the situation anyway. It is not three hundred separate
    incidents, it is one stretch the vessel may not enter. Runs say that.
    """
    runs: list[list] = []
    for cell in cells:
        r, c = cell
        if runs and max(abs(r - runs[-1][-1][0]), abs(c - runs[-1][-1][1])) <= gap:
            runs[-1].append(cell)
        else:
            runs.append([cell])
    return runs


def _cross(d, xy, size: float, colour, width: int) -> None:
    x, y = xy
    for a, b in (
        ((x - size, y - size), (x + size, y + size)),
        ((x - size, y + size), (x + size, y - size)),
    ):
        d.line([a, b], fill=CASING, width=width + 4)
    for a, b in (
        ((x - size, y - size), (x + size, y + size)),
        ((x - size, y + size), (x + size, y - size)),
    ):
        d.line([a, b], fill=colour, width=width)


def _dot(d, xy, r: float, fill) -> None:
    x, y = xy
    d.ellipse([x - r - 2, y - r - 2, x + r + 2, y + r + 2], fill=CASING)
    d.ellipse([x - r, y - r, x + r, y + r], fill=fill)


def draw(img: Image, overlay: dict, *, show: str = "both", scale: float = 1.0) -> Image:
    """Paint the selected tracks onto a copy of ``img``.

    ``show`` is one of ``baseline``, ``iceroute``, ``both`` or ``none`` — the
    three-state reveal the demo is built around, plus the off position.
    """
    if show == "none" or not overlay:
        return img

    from PIL import ImageDraw

    sx, sy = _project(overlay, img)
    out = img.convert("RGB").copy()
    d = ImageDraw.Draw(out)

    w = max(1, round(3 * scale))
    base, ours = overlay.get("baseline") or {}, overlay.get("iceroute") or {}

    if show in ("baseline", "both") and base.get("path"):
        pts = _pts(base["path"], sx, sy)
        for run in _dashed(pts, DASH_ON * scale, DASH_OFF * scale):
            _stroke(d, run, COL_BASELINE, max(1, w - 1), casing=2)
        # The evidence, drawn over the track so nothing covers it: solid red
        # where the vessel may not go. A run of one or two cells has no length
        # to speak of, so it gets a marker instead of an invisible segment.
        for run in no_go_runs(base.get("no_go") or []):
            xy = _pts(run, sx, sy)
            if len(xy) > 2:
                _stroke(d, xy, COL_NO_GO, w, casing=1)
            else:
                _cross(d, xy[0], 5 * scale, COL_NO_GO, max(1, round(2 * scale)))

    if show in ("iceroute", "both") and ours.get("path"):
        _stroke(d, _pts(ours["path"], sx, sy), COL_ICEROUTE, w, casing=3)
        for xy in _pts(ours.get("waypoints") or [], sx, sy):
            _dot(d, xy, 3.5 * scale, COL_ICEROUTE)

    for key in ("start", "goal"):
        if overlay.get(key):
            _dot(d, _pts([overlay[key]], sx, sy)[0], 5 * scale, COL_ENDPOINT)
    return out
