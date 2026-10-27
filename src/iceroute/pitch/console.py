"""The demo as an operations console, with the real geography attached.

``demo.html`` is the thing that gets shown and it stays exactly as it is. This
is a second rendering of the same build — ``console.html`` — with a globe, the
true swath footprints, and the Sentinel-1 product metadata that the raw scenes
have been carrying all along and that nothing has ever displayed.

Everything on this page is measured
-----------------------------------
That is the constraint, and it is the reason several obvious things are *not*
here.

* **The footprints are real.** Each AI4Arctic scene carries a 21x21
  geolocation grid (``sar_grid2d_latitude`` / ``sar_grid2d_longitude``). The
  outline drawn on the globe is the border of that grid, at its true
  coordinates. Nothing is placed by hand.
* **The product metadata is real.** ``original_id`` is the full Sentinel-1
  product name, so the platform (S1A or S1B), the mode, the polarisation, the
  absolute orbit number and the sixty-second acquisition window are read, not
  invented.
* **The pass direction is measured, not assumed.** The along-track axis is
  axis 0 — despite being named ``sar_sample_2dgrid_points`` — which the
  geolocation grid settles: its azimuth comes out at 192-201 degrees, a
  descending near-polar track, while axis 1 sits about 92 degrees off it, which
  is Sentinel-1's right-looking range direction. The names in the file are
  transposed relative to the array order; the measurement is not.
* **There are no coastlines**, because there is no coastline dataset in this
  project and this page does not reach the network. The globe carries a
  graticule and the Arctic Circle, which are definitions rather than data, and
  says so. Drawing a plausible landmass would be the one kind of lie this
  screen cannot afford.
* **There is no orbit ring.** A full orbit would have to be modelled from
  elements nobody here has. What is drawn instead is the ground track *of the
  acquisition itself*: sixty seconds long, in the measured along-track
  direction, which is a thing that actually happened.

Not a second source of truth
----------------------------
Every number comes from :func:`standalone.collect` — the same function that
builds ``demo.html`` — and the legend colours from :func:`standalone.palette`.
This module adds geography and formats strings; it recomputes nothing. Two
renderers that each derive their own figures is the failure D-025, D-027 and
D-031 are all about.

The one thing the browser computes is the orthographic projection, because the
globe turns. That is display geometry, not measurement: every figure printed
anywhere on the page is still formatted in Python, and the projection cannot
make a number say something Python did not.
"""

from __future__ import annotations

import argparse
import html
import json
import logging
import math
import re
import sys
from pathlib import Path

import yaml

from iceroute.pitch import standalone as sa
from iceroute.routing.geocode import POLAR_CRS

log = logging.getLogger("console")

DEMO_DIR = Path("data/demo")
RAW_DIR = Path("data/raw/test")
METRICS = Path("artifacts/metrics.json")
ASSUMPTIONS = Path("configs/assumptions.yaml")
OUT = Path("console.html")
# Rasters this page builds and the demo does not. Kept apart from
# data/demo/images so preflight does not count them as orphans of the build.
CACHE_DIR = Path("data/demo/console")

ARCTIC_CIRCLE = 66.5633      # definition, not a measurement

# S1B_EW_GRDM_1SDH_20180707T113313_20180707T113413_011704_015877_83D1
PRODUCT = re.compile(
    r"^(?P<platform>S1[AB])_(?P<mode>\w{2})_(?P<product>\w{4})_(?P<pol>\w{4})_"
    r"(?P<t0>\d{8}T\d{6})_(?P<t1>\d{8}T\d{6})_(?P<orbit>\d{6})_(?P<take>\w{6})_"
    r"(?P<uid>\w{4})"
)

NBSP = "\u00a0"


def it(n: float | int, decimals: int = 0) -> str:
    """Italian digit grouping and decimal comma, non-breaking between groups."""
    return f"{n:,.{decimals}f}".replace(",", NBSP).replace(".", ",")


# ---------------------------------------------------------------------------
# Geography, read off the scenes
# ---------------------------------------------------------------------------


def azimuth(p0: tuple[float, float], p1: tuple[float, float]) -> float:
    """Initial bearing from p0 to p1, degrees clockwise from north."""
    lo0, la0 = math.radians(p0[0]), math.radians(p0[1])
    lo1, la1 = math.radians(p1[0]), math.radians(p1[1])
    d = lo1 - lo0
    y = math.sin(d) * math.cos(la1)
    x = math.cos(la0) * math.sin(la1) - math.sin(la0) * math.cos(la1) * math.cos(d)
    return (math.degrees(math.atan2(y, x)) + 360) % 360


def great_circle_km(p0: tuple[float, float], p1: tuple[float, float]) -> float:
    lo0, la0 = math.radians(p0[0]), math.radians(p0[1])
    lo1, la1 = math.radians(p1[0]), math.radians(p1[1])
    a = (math.sin((la1 - la0) / 2) ** 2
         + math.cos(la0) * math.cos(la1) * math.sin((lo1 - lo0) / 2) ** 2)
    return 6371.0 * 2 * math.asin(math.sqrt(a))


def _stamp(s: str) -> str:
    """20180707T113313 -> 2018-07-07 11:33:13Z, as written in the product name."""
    return f"{s[0:4]}-{s[4:6]}-{s[6:8]} {s[9:11]}:{s[11:13]}:{s[13:15]}Z"


def footprint(raw: Path, *, step: int = 2) -> dict | None:
    """The real swath outline and the real product identity, or nothing.

    Returns ``None`` when the raw scene is not on this disk: the globe then
    simply has one fewer footprint, which is better than a placed guess.
    """
    if not raw.is_file():
        return None
    import xarray as xr

    with xr.open_dataset(raw) as ds:
        la = ds["sar_grid2d_latitude"].values
        lo = ds["sar_grid2d_longitude"].values
        original = str(ds.attrs.get("original_id", ""))
        raw_spacing = ds.attrs.get("pixel_spacing")
    # netCDF attributes come back as numpy scalars, which json refuses.
    spacing = None if raw_spacing is None else float(raw_spacing)

    def pt(i, j):
        return [round(float(lo[i, j]), 4), round(float(la[i, j]), 4)]

    n, m = la.shape
    ring = ([pt(0, j) for j in range(0, m, step)]
            + [pt(i, m - 1) for i in range(0, n, step)]
            + [pt(n - 1, j) for j in range(m - 1, -1, -step)]
            + [pt(i, 0) for i in range(n - 1, -1, -step)])

    mid = n // 2, m // 2
    # Axis 0 is along-track. Measured, not taken from the dimension name: its
    # bearing lands on a descending near-polar track while axis 1 sits ~92
    # degrees off it, which is the right-looking range direction.
    a0, a1 = (pt(0, mid[1]), pt(n - 1, mid[1])), (pt(mid[0], 0), pt(mid[0], m - 1))
    track = azimuth(tuple(a0[0]), tuple(a0[1]))

    out = {
        "ring": ring,
        "corners": [pt(0, 0), pt(0, m - 1), pt(n - 1, m - 1), pt(n - 1, 0)],
        "centre": [round(float(lo.mean()), 4), round(float(la.mean()), 4)],
        "track_az": round(track, 1),
        "look_az": round(azimuth(tuple(a1[0]), tuple(a1[1])), 1),
        "along_km": round(great_circle_km(tuple(a0[0]), tuple(a0[1])), 1),
        "across_km": round(great_circle_km(tuple(a1[0]), tuple(a1[1])), 1),
        # The sixty seconds of ground track that produced this scene, as a
        # segment through the footprint centre along the measured bearing.
        "track_line": [pt(0, mid[1]), pt(n - 1, mid[1])],
        "pass": "discendente" if 90 < track < 270 else "ascendente",
        "pixel_spacing_m": spacing,
    }
    if (m_ := PRODUCT.match(original)):
        g = m_.groupdict()
        out |= {
            "platform": g["platform"], "mode": g["mode"], "level": g["product"],
            "pol": g["pol"], "orbit": int(g["orbit"]), "take": g["take"],
            "uid": g["uid"], "t0": _stamp(g["t0"]), "t1": _stamp(g["t1"]),
            "seconds": round((int(g["t1"][13:15]) - int(g["t0"][13:15])
                              + 60 * (int(g["t1"][11:13]) - int(g["t0"][11:13]))), 0),
        }
    return out


def sar_geocoded(raw: Path, grid_shape, resolution_m: float, cache: Path) -> Path | None:
    """The SAR put on the routing grid, so the layers stop jumping.

    ``sar`` is the decimated swath; the mask, the concentration, the RIO raster
    and the routes are EPSG:3413 at 500 m. Switching between them moves the
    scene, because the picture really has changed projection — correct, and
    still the thing that makes the two halves of the demo look like two
    different places.

    The fix is not to warp anything by hand. The SAR is handed to the *same*
    ``geocode_scene`` call that built the routing grid, as an extra band, so it
    lands on that grid by construction rather than by agreement. The shape is
    then checked against the overlay's own ``grid_shape``, and a mismatch
    returns nothing: a layer that is one cell out would put the route in the
    wrong place while looking perfectly convincing, which is D-027 again.

    Stretching happens *before* resampling, on the native band, exactly as
    ``precompute`` does it for the swath SAR — so the geocoded layer carries
    the same grey mapping and the two are comparable rather than merely
    similar.
    """
    out = cache / f"{raw.stem}_sargeo.png"
    if out.is_file():
        # The cache is checked against the grid, not trusted for being present.
        # A PNG left over from a different resolution would otherwise sail past
        # the shape check below and misplace every route drawn on it.
        from PIL import Image

        with Image.open(out) as im:
            cached = [im.height, im.width]
        if cached == list(grid_shape):
            return out
        log.info("  cache obsoleta per %s (%s contro %s): rigenero",
                 raw.name, cached, list(grid_shape))
        out.unlink()
    if not raw.is_file():
        return None

    from iceroute.data.loader import load_config, load_scene
    from iceroute.demo.precompute import resolve_labels, save_png, stretch
    from iceroute.routing.geocode import geocode_scene

    scene = load_scene(raw, load_config(Path("configs/data.yaml")),
                       labels_from=resolve_labels(raw))
    hv8 = stretch(scene.hv, scene.mask_valid)
    geo = geocode_scene(scene, resolution_m=resolution_m, extra={"sar": hv8})
    if list(geo.shape) != list(grid_shape):
        log.warning("  %s: griglia %s contro %s dell'overlay — livello non generato",
                    raw.name, list(geo.shape), list(grid_shape))
        return None

    img = geo.extra["sar"].copy()
    img[~geo.mask_valid] = 0
    cache.mkdir(parents=True, exist_ok=True)
    save_png(img, out, downscale=1)
    return out


def frame(fp: dict | None, grid_shape, resolution_m: float) -> dict | None:
    """How far the swath is turned inside its own raster, and how far it can be
    enlarged without losing a corner of it.

    The geocoded grid is north-up; the swath is flown at about 192 degrees. So
    the scene sits in its raster as a diamond with four empty corners, and on
    screen it reads as a crooked picture in a mostly empty box. The angle is
    not guessed: the four corners of the geolocation grid are projected into
    EPSG:3413 — the grid the raster is actually on — and the tilt is measured
    from them.

    ``scale`` is the largest enlargement that still contains all four corners.
    Filling the frame by cropping would be trading empty pixels for scene, and
    the corner that disappears is always somebody's ice.
    """
    if not (fp and fp.get("corners") and grid_shape and resolution_m):
        return None
    from pyproj import Transformer

    tr = Transformer.from_crs("EPSG:4326", POLAR_CRS, always_xy=True)
    # Image axes: x grows with projected x, y grows downward as projected y falls.
    pts = []
    for lon, lat in fp["corners"]:
        x, y = tr.transform(lon, lat)
        pts.append((x, -y))

    # Tilt of the along-track edge, then the smallest turn that squares it up.
    (x0, y0), (_, _), (_, _), (x3, y3) = pts
    tilt = math.degrees(math.atan2(y3 - y0, x3 - x0))
    # Rounded here, before the fit is measured: the browser applies this value,
    # and sizing the enlargement for a slightly different angle quietly crops
    # the corners — half a hundredth of a degree across 419 km is 38 metres.
    rot = round(((90.0 - tilt) + 45.0) % 90.0 - 45.0, 2)

    c, s = math.cos(math.radians(rot)), math.sin(math.radians(rot))
    xs = [x * c - y * s for x, y in pts]
    ys = [x * s + y * c for x, y in pts]
    w, h = max(xs) - min(xs), max(ys) - min(ys)
    if w <= 0 or h <= 0:
        return None
    rows, cols = grid_shape
    scale = min(cols * resolution_m / w, rows * resolution_m / h)
    # Rounded *down*. To the nearest would push it over by a hair — fifteen
    # metres on this scene — and the guarantee here is "never crops", not
    # "almost never crops".
    scale = math.floor(max(1.0, scale) * 1e4) / 1e4
    return {"rot": rot, "scale": scale}


def route_lonlat(routes_dir: Path, name: str) -> list[list[float]] | None:
    """The transmitted route, at its true coordinates."""
    p = routes_dir / (name or "")
    if not (name and p.is_file()):
        return None
    for f in json.loads(p.read_text()).get("features") or []:
        geom = f.get("geometry") or {}
        if geom.get("type") == "LineString":
            return [[round(x, 4), round(y, 4)] for x, y in geom["coordinates"]]
    return None


# ---------------------------------------------------------------------------
# The console's own panels, from the same build
# ---------------------------------------------------------------------------


def inference_stats(files: dict) -> dict | None:
    raw = files.get("inference_stats")
    if not isinstance(raw, str):
        return None
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return None


def admissibility(overlay: dict, buckets: int = 180) -> dict | None:
    """Where the direct route is forbidden, cell by cell, bucketed for drawing.

    This is the one trace on the page, and it is entirely measured: the overlay
    records the baseline's full path and the subset of its cells POLARIS
    forbids. Each bucket is the fraction of its cells that are no-go, so a tall
    bar is a real stretch of ice the ship may not enter.
    """
    base = overlay.get("baseline") or {}
    path = [tuple(c) for c in base.get("path") or []]
    if not path:
        return None
    no_go = {tuple(c) for c in base.get("no_go") or []}
    no_data = {tuple(c) for c in base.get("no_data") or []}
    size = max(1, math.ceil(len(path) / buckets))
    bars = []
    for i in range(0, len(path), size):
        chunk = path[i:i + size]
        bars.append([
            round(sum(c in no_go for c in chunk) / len(chunk), 3),
            round(sum(c in no_data for c in chunk) / len(chunk), 3),
        ])
    return {"bars": bars, "n_cells": len(path), "n_no_go": len(no_go),
            "n_no_data": len(no_data)}


def event_log(scene: dict, ice_class: str, entry: dict, stats: dict | None,
              fp: dict | None, overlay: dict, adm: dict | None) -> list[list[str]]:
    """One line per fact the build actually recorded. No invented timestamps.

    A console log reads as a clock ticking, and this one deliberately does not
    have one: nothing in ``data/demo`` records when a stage ran. Sequence
    numbers are honest about that; a fabricated ``[23:07:21]`` would not be.
    """
    rows: list[list[str]] = []

    def add(tag: str, text: str) -> None:
        rows.append([f"{len(rows):02d}", tag, text])

    shape = scene.get("shape") or (stats or {}).get("scene_shape")
    if shape:
        px = (fp or {}).get("pixel_spacing_m")
        extent = (f" · {it(shape[0] * px / 1000)}{NBSP}×{NBSP}{it(shape[1] * px / 1000)} km"
                  if px else "")
        add("SCENE", f"{scene['id']} · {it(shape[0])}×{it(shape[1])} px{extent}")
    if fp and fp.get("platform"):
        add("PRODOTTO", f"{fp['platform']} {fp['mode']} {fp['level']} {fp['pol']} · "
                        f"orbita assoluta {it(fp['orbit'])} · take {fp['take']}")
        add("ACQUIS.", f"{fp['t0']} → {fp['t1']} · {it(fp.get('seconds') or 0)} s · "
                       f"passaggio {fp['pass']} · traccia {it(fp['track_az'], 1)}°")
    if fp:
        add("IMPRONTA", f"{it(fp['along_km'], 1)} km lungo traccia × "
                        f"{it(fp['across_km'], 1)} km in range · centro "
                        f"{it(fp['centre'][1], 2)}°N {it(abs(fp['centre'][0]), 2)}°O")
    if stats:
        add("TILE", f"{it(stats['n_tiles'])} tile da {stats['tile_size']} px · "
                    f"sovrapposizione {stats['overlap']} · batch {stats['batch_size']}")
        add("INFER", f"{it(stats['seconds'], 3)} s su {str(stats['device']).upper()} · "
                     f"picco {it(stats['peak_accumulator_mb'], 1)} MB contro "
                     f"{it(stats['naive_accumulator_mb'], 1)} MB del metodo ingenuo")
    gs, res = overlay.get("grid_shape"), overlay.get("resolution_m")
    if gs and res:
        add("GEOCODE", f"{it(gs[0])}×{it(gs[1])} celle a {it(res)} m")
    pol = entry.get("polaris") or {}
    cond = entry.get("conditions") or {}
    if pol:
        add("POLARIS", f"{ice_class} · RIO min {it(pol.get('rio_min') or 0, 0)} · "
                       f"max {it(pol.get('rio_max') or 0, 0)} · normale "
                       f"{it((cond.get('fraction_normal') or 0) * 100)}%")
    cmp_ = entry.get("comparison") or {}
    ours = cmp_.get("iceroute") or {}
    if entry.get("route_exists") and ours:
        add("A*", f"{it(cond.get('n_cells') or 0)} celle · {entry.get('n_waypoints')} "
                  f"waypoint · {it(ours.get('distance_km') or 0)} km")
    if adm:
        verdict = "NON PERCORRIBILE" if adm["n_no_go"] else "percorribile"
        add("DIRETTA", f"{it(adm['n_cells'])} celle · {it(adm['n_no_go'])} vietate · "
                       f"{it(adm['n_no_data'])} fuori copertura · {verdict}")
    pay = entry.get("payload") or {}
    if pay.get("total"):
        add("PAYLOAD", f"maschera {it(pay['mask'])} B + rotta {it(pay['route'])} B + "
                       f"metadati {it(pay['meta'])} B = {it(pay['total'] / 1000)} KB")
    for w in scene.get("warnings") or []:
        add("AVVISO", str(w)[:120])
    return rows


_TOKENS = re.compile(r'("(?:[^"\\]|\\.)*")|(-?\d+(?:\.\d+)?)')


def tint(chunk: str) -> str:
    """Colour JSON by token, one fixed-width chunk at a time.

    Chunks are cut on byte boundaries so the offsets stay true, which means a
    string or a number can be sliced in half. A partial token simply fails to
    match and renders plain — wrong-looking colour on a cut token is cosmetic,
    a lying offset would not be.
    """
    out, i = [], 0
    for m in _TOKENS.finditer(chunk):
        out.append(html.escape(chunk[i:m.start()]))
        cls = "s" if m.group(1) else "n"
        out.append(f'<i class="{cls}">{html.escape(m.group(0))}</i>')
        i = m.end()
    out.append(html.escape(chunk[i:]))
    return "".join(out)


def bridge(entry: dict, width: int = 64) -> dict | None:
    """The bytes that actually reach the ship, laid out like a dump.

    ``payload.text`` is the transmitted feature, wrapped for display; the
    separators are ``(",", ":")`` so there are no spaces to wrap on and joining
    the lines restores the exact string — asserted here rather than assumed,
    because the whole panel is a claim about how many bytes arrive.
    """
    pay = entry.get("payload") or {}
    text = pay.get("text")
    if not text:
        return None
    flat = "".join(text.splitlines())
    n = len(flat.encode())
    if pay.get("route") and n != pay["route"]:
        # Rather than print a dump that disagrees with the byte count beside it.
        return {"mismatch": True, "declared": pay["route"], "rebuilt": n}

    # Offsets are byte offsets. ASCII makes character and byte indices the same;
    # if that ever stops holding the ruler would quietly start lying, so it is
    # checked and the ruler is dropped instead.
    ruler = flat.isascii()
    lines = [[f"{i:04x}" if ruler else "", tint(flat[i:i + width])]
             for i in range(0, len(flat), width)]

    link = pay.get("link") or {}
    rows = [
        ["maschera del ghiaccio", f"{it(pay['mask'])} B"] if pay.get("mask") else None,
        [f"rotta · {pay.get('n_waypoints')} waypoint", f"{it(pay['route'])} B"]
        if pay.get("route") else None,
        ["metadati", f"{it(pay['meta'])} B"] if pay.get("meta") else None,
        ["totale trasmesso", f"{it(pay['total'] / 1000)} KB"] if pay.get("total") else None,
        ["contro il radar", f"{it(pay['sar'] / 1e6, 1)} MB"] if pay.get("sar") else None,
        ["compressione", f"{it(pay['ratio'], 1)}:1"] if pay.get("ratio") else None,
        ["sul collegamento", f"≈ {it(link['s'])} s · {link['terminal']}"]
        if link.get("s") is not None else None,
        ["in messaggi SBD", f"{it(link['sbd'])}"] if link.get("sbd") else None,
    ]
    return {
        "lines": lines, "bytes": n, "ruler": ruler,
        "n_waypoints": pay.get("n_waypoints"),
        "head": (f"{it(n)} byte · {pay.get('n_waypoints')} waypoint"
                 + (f" · {it(link['sbd'])} messaggi SBD" if link.get("sbd") else "")),
        "rows": [[*r, ""] for r in rows if r],
        "note": pay.get("note") or "",
    }


def panels(entry: dict, fp: dict | None, stats: dict | None,
           adm: dict | None, pal: dict) -> dict:
    """Every row and every bar width, resolved here.

    The browser places strings and percentages it was handed. Computing a bar
    width from ``fraction_no_go`` in JavaScript would be the browser deciding
    how big a risk looks, which is the one thing this page must not delegate.
    """
    f, s = fp or {}, stats or {}
    cond = entry.get("conditions") or {}
    pol = entry.get("polaris") or {}

    def row(k, v, small=False):
        return None if v in (None, "") else [k, v, "sm" if small else ""]

    prod = [
        row("piattaforma", f"{f.get('platform','')} · {f.get('mode','')} "
                           f"{f.get('level','')} {f.get('pol','')}".strip(), True)
        if f.get("platform") else None,
        row("orbita assoluta", it(f["orbit"]) if f.get("orbit") else None),
        row("acquisizione", f.get("t0"), True),
        row("fine", f.get("t1"), True),
        row("durata", f"{it(f['seconds'])} s" if f.get("seconds") else None, True),
        row("passaggio", f"{f.get('pass','')} · {it(f['track_az'], 1)}°"
            if f.get("track_az") is not None else None, True),
        row("vista", f"destra · {it(f['look_az'], 1)}°"
            if f.get("look_az") is not None else None, True),
        row("impronta", f"{it(f['along_km'], 1)} × {it(f['across_km'], 1)} km"
            if f.get("along_km") else None, True),
        row("centro", f"{it(f['centre'][1], 2)}° {it(f['centre'][0], 2)}°"
            if f.get("centre") else None, True),
        row("passo pixel", f"{it(f['pixel_spacing_m'])} m"
            if f.get("pixel_spacing_m") else None, True),
    ]

    onboard = [
        row("tile", f"{it(s['n_tiles'])} × {s['tile_size']} px") if s.get("n_tiles") else None,
        row("sovrapposizione", f"{s['overlap']} px" if s.get("overlap") else None, True),
        row("batch", str(s["batch_size"]) if s.get("batch_size") else None, True),
        row("inferenza", f"{it(s['seconds'], 3)} s" if s.get("seconds") else None),
        row("dispositivo", str(s["device"]).upper() if s.get("device") else None, True),
        row("picco memoria", f"{it(s['peak_accumulator_mb'], 1)} MB"
            if s.get("peak_accumulator_mb") else None),
        row("metodo ingenuo", f"{it(s['naive_accumulator_mb'], 1)} MB"
            if s.get("naive_accumulator_mb") else None, True),
    ]

    def pct(v):
        return round(100 * v, 2) if isinstance(v, int | float) else 0.0

    rio_meter = None
    if entry.get("route_exists") and cond.get("n_cells"):
        rio_meter = {
            "label": "celle della rotta per livello POLARIS",
            "value": f"{it(cond['n_cells'])} celle",
            "segs": [[pct(cond.get("fraction_normal")), pal["rio"][0]],
                     [pct(cond.get("fraction_elevated")), pal["rio"][1]],
                     [pct(cond.get("fraction_no_go")), pal["rio"][2]]],
        }
    rio_rows = [
        row("rio minimo", it(pol["rio_min"]) if pol.get("rio_min") is not None else None),
        row("rio mediano", it(cond["rio_median"])
            if cond.get("rio_median") is not None else None),
        row("rio massimo", it(pol["rio_max"]) if pol.get("rio_max") is not None else None),
        row("classe peggiore", cond.get("worst_class"), True),
        row("sua concentrazione", f"{it(cond['worst_class_sic_median'])}/10"
            if cond.get("worst_class_sic_median") is not None else None, True),
        row("suo rio mediano", it(cond["worst_class_rio_median"])
            if cond.get("worst_class_rio_median") is not None else None, True),
    ]

    by = cond.get("by_class") or {}
    total = sum(by.values())
    mix = None
    if total:
        mix = {"label": "tipi di ghiaccio attraversati",
               "value": f"{it(total)} celle",
               "segs": [[round(100 * by[k] / total, 2), pal["classes"][int(k)]]
                        for k in sorted(by, key=int) if int(k) < len(pal["classes"])]}

    trace = None
    if adm:
        trace = {
            "bars": [[round(max(3.0, 100 * (g or d)), 1),
                      "g" if g else ("d" if d else "")] for g, d in adm["bars"]],
            "note": (f"{it(adm['n_cells'])} celle lungo la rotta diretta · "
                     f"{it(adm['n_no_go'])} vietate da POLARIS · "
                     f"{it(adm['n_no_data'])} fuori copertura. Ogni barra è una porzione "
                     f"del percorso: l'altezza è la frazione delle sue celle vietate."),
        }

    return {"prod": [r for r in prod if r], "onboard": [r for r in onboard if r],
            "rio_meter": rio_meter, "rio_rows": [r for r in rio_rows if r],
            "mix": mix, "trace": trace, "bridge": bridge(entry)}


def collect(demo_dir: Path, metrics: dict, assumptions: dict, raw_dir: Path,
            cache_dir: Path = CACHE_DIR) -> dict:
    """standalone's context, plus geography and the console's own strings."""
    ctx = sa.collect(demo_dir, metrics, assumptions)
    index = json.loads((demo_dir / "index.json").read_text())
    by_id = {s["scene_id"]: s for s in index.get("scenes") or []}

    missing_raw: list[str] = []
    for scene in ctx["scenes"]:
        src = by_id.get(scene["id"]) or {}
        fp = footprint(raw_dir / scene["id"])
        if fp is None:
            missing_raw.append(scene["id"])
        stats = inference_stats(src.get("files") or {})
        scene |= {"footprint": fp, "stats": stats, "shape": src.get("shape"),
                  "warnings": src.get("warnings") or [],
                  "inference_seconds": src.get("inference_seconds")}

        grid_shape = grid_res = None
        for ic, entry in scene["classes"].items():
            raw_entry = ((src.get("classes") or {}).get(ic)) or {}
            ov_name = raw_entry.get("overlay") or ""
            overlay = {}
            if ov_name and (demo_dir / "overlays" / ov_name).is_file():
                overlay = json.loads((demo_dir / "overlays" / ov_name).read_text())
                grid_shape = grid_shape or overlay.get("grid_shape")
                grid_res = grid_res or overlay.get("resolution_m")
            adm = admissibility(overlay) if overlay else None
            entry |= {
                "adm": adm,
                "route_lonlat": route_lonlat(demo_dir / "routes",
                                             raw_entry.get("geojson") or ""),
                "grid": {"shape": overlay.get("grid_shape"),
                         "res": overlay.get("resolution_m")},
                "log": event_log(scene, ic, entry, stats, fp, overlay, adm),
            }
            entry["panels"] = panels(entry, fp, stats, adm, sa.palette())
        scene["_grid"] = (grid_shape, grid_res)

    # The SAR on the routing grid, built from the same geocode call that made
    # that grid. Done per scene, after its overlay has said what the grid is.
    any_geo = False
    for scene in ctx["scenes"]:
        shape, res = scene.pop("_grid", (None, None))
        if not (shape and res):
            scene["frame"] = None
            continue
        scene["frame"] = frame(scene.get("footprint"), shape, res)
        png = sar_geocoded(raw_dir / scene["id"], shape, res, cache_dir)
        if png is None:
            continue
        scene["files"]["sar_geo"] = png.name
        if png.name not in ctx["images"]:
            ctx["images"][png.name] = sa.data_uri(png, photographic=True)
        any_geo = True

    layers = list(sa.LAYERS)
    if any_geo:
        at = next((i for i, lay in enumerate(layers) if lay[0] == "sar"), len(layers))
        layers.insert(at, ("sar_geo", "SAR sulla griglia delle rotte", True))

    ctx |= {
        "palette": sa.palette(),
        "arctic_circle": ARCTIC_CIRCLE,
        "missing_raw": missing_raw,
        "layers": layers,
        "modes": sa.MODES,
    }
    return ctx


# Palette and type scale taken from Blueprint (palantir/blueprint,
# @blueprintjs/core 6.21.0, Apache-2.0), read from `packages/colors/src/_colors.scss`
# and `packages/core/src/common/_variables.scss` rather than eyeballed. The
# components are not Blueprint's — only its tokens are — so nothing here claims
# to be a Blueprint app.
#
# Two choices came out of measuring contrast rather than taste. `gray3` is
# 4.35:1 on the elevated surface, under the 4.5 threshold, and that is exactly
# where the nine-pixel labels sit: so the faint tier steps up to `gray4` on
# anything raised. And `blue3` is 3.43:1 as text, fine as a fill behind white
# and not fine as a label, so links and accents use `blue5` while `blue3` is
# only ever a background.
CSS = """
:root{
  /* dark-gray1 / dark-gray2 / dark-gray3 — app, panel, elevated */
  --bg:#1c2127; --panel:#252a31; --raise:#2f343c;
  /* dark-gray5 / dark-gray4 — borders and inner rules */
  --line:#404854; --rule:#383e47;
  /* light-gray5 / gray5 / gray4. Blueprint's muted tier is gray4 and its
     small type is 12px; these labels are 8-10px, where gray4 falls to 4.3:1
     on a raised panel and gray3 to 3.4:1. Measured, so both tiers step up one
     stop and every piece of text on this page clears 4.5:1. */
  --text:#f6f7f9; --dim:#c5cbd3; --faint:#abb3bf; --strong:#ffffff;
  /* blue3 fills, blue5 text; green5 reads on every surface here */
  --accent:#2d72d2; --accent-text:#8abbff; --good:#72ca9b;
  --sel:#2d72d238; --glow:#2d72d233;
  /* Cromie del globo. Stanno qui e non nei fogli di stile dei singoli pezzi
     perche' la legenda usa gli stessi token: un campione e la linea che
     descrive non possono piu' divergere. */
  --geo-scene:#4c90f0; --geo-scene-fill:#4c90f01f;   /* blue4 */
  --geo-track:#ec9a3c; --geo-track-dim:#ec9a3c66;    /* orange4 */
  --geo-arc:#5f6b7c; --geo-grid:#404854;             /* gray1, dark-gray5 */
  --geo-limb:#5f6b7c;
  /* L'alone dietro i marcatori e' il fondo dell'app: su un grigio piu' chiaro
     un alone quasi nero smette di separare e diventa un contorno. */
  --halo:#1c2127;
  --ok-t:#72ca9b; --bad-t:#fa999c; --warn-t:#fbb360; --info-t:#abb3bf;
  --radius:4px;
  --sans:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Oxygen,Ubuntu,Cantarell,
    "Open Sans","Helvetica Neue",sans-serif;
  --mono:ui-monospace,"SF Mono",SFMono-Regular,Menlo,Consolas,monospace;
}
*{box-sizing:border-box}
html,body{margin:0;height:100%;background:var(--bg);color:var(--text);
  overflow:auto hidden;font:14px/1.28581 var(--sans);-webkit-font-smoothing:antialiased}
b,strong{color:var(--strong);font-weight:600}

/* Below this the four columns would start eating each other's content; a
   horizontal scrollbar is honest about running out of room, clipping is not. */
#app{display:grid;height:100vh;min-width:1100px;
  grid-template-columns:200px minmax(300px,1fr) minmax(320px,1.15fr) 270px;
  grid-template-rows:38px 1fr 190px 22px;gap:1px;background:var(--line)}
#bar{grid-column:1/-1;background:var(--raise);display:flex;align-items:center;gap:0}
#foot{grid-column:1/-1;background:var(--raise);display:flex;align-items:center;
  font-size:10px;color:var(--faint);overflow:hidden}
#foot div{padding:0 10px;border-right:1px solid var(--line);line-height:21px;white-space:nowrap}
#foot div:last-child{border-right:0;margin-left:auto;color:var(--faint)}
#foot b{color:var(--dim);font-weight:500}

.pane{background:var(--panel);display:flex;flex-direction:column;min-width:0;min-height:0}
.hd{display:flex;align-items:center;gap:8px;padding:5px 9px;background:var(--raise);
  border-bottom:1px solid var(--line);flex-shrink:0}
.hd h2{margin:0;font-size:9px;letter-spacing:.17em;text-transform:uppercase;
  color:var(--dim);font-weight:700}
.hd .rt{margin-left:auto;font-size:9.5px;color:var(--faint);letter-spacing:.06em}
.bd{flex:1;min-height:0;overflow:auto;padding:8px 9px}
.bd.flush{padding:0;overflow:hidden}

#bar .mark{display:flex;align-items:center;gap:7px;padding:0 12px;font-weight:700;
  letter-spacing:.2em;font-size:12px;color:var(--text)}
#bar .mark i{width:8px;height:8px;background:var(--accent);display:block;
  box-shadow:0 0 0 3px var(--glow)}
#bar .cell{padding:0 11px;border-left:1px solid var(--line);height:100%;display:flex;
  flex-direction:column;justify-content:center;gap:1px}
#bar .k{font-size:8px;letter-spacing:.15em;text-transform:uppercase;color:var(--faint)}
#bar .v{font-size:11px;color:var(--text)}
#bar .sp{flex:1}
#clock{padding:0 14px;border-left:1px solid var(--line);font-size:18px;font-weight:600;
  letter-spacing:.06em;color:var(--text);height:100%;display:flex;align-items:center}

.row{display:flex;align-items:center;gap:7px;padding:4px 9px;cursor:pointer;
  border-left:2px solid transparent;border-bottom:1px solid var(--rule);color:var(--dim);
  font-size:11px}
.row:hover{background:var(--raise);color:var(--text)}
.row.on{background:var(--sel);background:var(--sel);color:var(--text);border-left-color:var(--accent)}
.row .id{flex:1;min-width:0;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.row .sub{font-size:9px;color:var(--faint);letter-spacing:.04em}
.pill{font-size:8px;font-weight:700;letter-spacing:.1em;padding:2px 5px;border-radius:var(--radius);
  text-transform:uppercase;border:1px solid transparent;flex-shrink:0}
/* Opaque, not a tint. A translucent tag borrows whatever is under it, so the
   same "vietata" read 4.37:1 on a selected row and 4.9:1 on an unselected one
   — a label that changes legibility with the selection. These are the same
   15%/30% blends, resolved once over the panel. */
.pill.ok{background:#304241;color:var(--ok-t);border-color:#3c5a51}
.pill.bad{background:#453b41;color:var(--bad-t);border-color:#654c51}
.pill.warn{background:#453e38;color:var(--warn-t);border-color:#66533f}
.pill.info{background:#393e46;color:var(--info-t);border-color:#4d535c}

#globe{width:100%;height:100%;display:block;cursor:grab;overflow:hidden}
#globe:active{cursor:grabbing}
.gr{fill:none;stroke:var(--geo-grid);stroke-width:.5}
.limb{fill:var(--bg);stroke:var(--geo-limb);stroke-width:1}
.arc{fill:none;stroke:var(--geo-arc);stroke-width:.9;stroke-dasharray:3 3}
.fp{fill:var(--geo-scene-fill);stroke:var(--geo-scene);stroke-width:1}
.fp.on{fill:var(--c-ours-fill);stroke:var(--c-ours);stroke-width:1.6}
.dot{fill:var(--geo-scene);stroke:var(--halo);stroke-width:.7}
.dot.on{fill:var(--c-ours);stroke-width:1}
.trk{fill:none;stroke:var(--geo-track);stroke-width:1.8}
.orb{fill:none;stroke:var(--geo-track-dim);stroke-width:.8;stroke-dasharray:2 5}
.orb.moving{stroke:var(--geo-track);stroke-width:1.3}
.satg .body{fill:var(--geo-track);stroke:var(--halo);stroke-width:.6}
.satg .pan{fill:var(--geo-limb);stroke:var(--halo);stroke-width:.5}
.satg .ant{stroke:var(--geo-track);stroke-width:.7;fill:none}
.rt{fill:none;stroke:var(--c-ours);stroke-width:1.6}
.sat{fill:var(--geo-track);stroke:var(--halo);stroke-width:1}
.lbl{fill:var(--dim);font-size:8.5px;letter-spacing:.08em}

.map{position:relative;line-height:0;background:var(--halo);height:100%;
  display:flex;align-items:center;justify-content:center;overflow:hidden}
/* Rotating the stack turns the raster and its overlay together, so the routes
   stay on the pixels they were computed for. */
.mapin{position:relative;max-width:100%;max-height:100%;
  transition:transform .35s ease}
.mapin img{max-width:100%;max-height:100%;display:block}
.ovwrap{position:absolute;inset:0}
svg.ov{width:100%;height:100%;display:block;overflow:visible}
.cas{fill:none;stroke:var(--halo);stroke-width:9;vector-effect:non-scaling-stroke;
  stroke-linejoin:round;stroke-linecap:round}
.base{fill:none;stroke:var(--c-base);stroke-width:2.5;stroke-dasharray:9 6;
  vector-effect:non-scaling-stroke}
.ours{fill:none;stroke:var(--c-ours);stroke-width:3.5;vector-effect:non-scaling-stroke;
  stroke-linejoin:round;stroke-linecap:round}
.nogo{fill:none;stroke:var(--c-nogo);stroke-width:4.5;vector-effect:non-scaling-stroke;
  stroke-linecap:round}
.nogod{fill:var(--c-nogo);stroke:var(--halo);stroke-width:1.5;vector-effect:non-scaling-stroke}
.wp{fill:var(--c-ours);stroke:var(--halo);stroke-width:2;vector-effect:non-scaling-stroke}
.end{fill:#fff;stroke:var(--halo);stroke-width:2;vector-effect:non-scaling-stroke}
body[data-mode=none] .lay-baseline,body[data-mode=none] .lay-iceroute,
body[data-mode=baseline] .lay-iceroute,body[data-mode=iceroute] .lay-baseline{display:none}

dl{display:grid;grid-template-columns:auto 1fr;gap:3px 10px;margin:0}
dt{color:var(--faint);font-size:10px;letter-spacing:.07em;text-transform:uppercase;
  white-space:nowrap}
dd{margin:0;text-align:right;color:var(--text);font-variant-numeric:tabular-nums;
  overflow:hidden;text-overflow:ellipsis}
dd.sm{font-size:10.5px;color:var(--dim)}
.sep{height:1px;background:var(--rule);margin:8px -9px}

.meter{margin:6px 0}
.meter .t{display:flex;justify-content:space-between;font-size:10px;color:var(--faint);
  margin-bottom:3px;letter-spacing:.06em}
.meter .t b{color:var(--text);font-variant-numeric:tabular-nums}
.track{height:7px;background:var(--bg);border:1px solid var(--rule);display:flex;overflow:hidden}
.track i{display:block;height:100%}

#dump{font-size:10.5px;line-height:1.75}
#dump .ln{display:flex;gap:13px}
#dump .off{color:var(--faint);flex-shrink:0;user-select:none;letter-spacing:.05em}
#dump .hx{color:var(--dim);word-break:break-all;white-space:pre-wrap}
#dump i{font-style:normal}
#dump i.s{color:var(--ok-t)}
#dump i.n{color:var(--geo-track)}
#dump .hdr{display:flex;gap:10px;align-items:baseline;margin-bottom:7px;
  padding-bottom:6px;border-bottom:1px solid var(--rule)}
#dump .hdr b{color:var(--text);font-size:13px}
#dump .hdr span{color:var(--faint);font-size:9.5px;letter-spacing:.1em;
  text-transform:uppercase}

#log{font-size:10.5px;line-height:1.65}
#log .l{display:flex;gap:9px;white-space:nowrap}
#log .n{color:var(--faint)}
#log .t{color:var(--accent);width:74px;flex-shrink:0;letter-spacing:.08em}
#log .m{color:var(--dim);overflow:hidden;text-overflow:ellipsis}
#log .l.w .t{color:var(--warn-t)}

.legend{display:flex;gap:11px;flex-wrap:wrap;padding:6px 9px;border-top:1px solid var(--rule);
  flex-shrink:0}
.legend span{display:flex;align-items:center;gap:5px;font-size:9.5px;color:var(--dim)}
.legend i{width:9px;height:9px;border-radius:var(--radius);display:inline-block}
.legend i.ln{height:3px;width:17px;border-radius:1px}

.tabs{display:flex;flex-shrink:0;background:var(--raise);border-bottom:1px solid var(--line)}
.tabs button{background:none;border:0;border-right:1px solid var(--line);color:var(--faint);
  padding:5px 11px;font:inherit;font-size:9px;letter-spacing:.14em;text-transform:uppercase;
  cursor:pointer;font-weight:700}
.tabs button.on{color:var(--text);background:var(--panel);box-shadow:inset 0 -2px 0 var(--accent)}
.tabs button:hover{color:var(--dim)}
.tabs .rt{margin-left:auto;padding:0 10px;align-self:center;font-size:9.5px;
  color:var(--faint);letter-spacing:.06em}

.bars{display:flex;align-items:flex-end;gap:1px;height:56px;padding-top:4px}
.bars i{flex:1;min-width:1px;background:var(--line);align-self:flex-end}
.bars i.g{background:var(--c-nogo)}
.bars i.d{background:var(--geo-limb)}
.note{font-size:9.5px;color:var(--faint);line-height:1.5;margin-top:7px}
::-webkit-scrollbar{width:8px;height:8px}
::-webkit-scrollbar-track{background:var(--panel)}
::-webkit-scrollbar-thumb{background:var(--line)}
"""

JS = r"""
const $ = s => document.querySelector(s);
const st = {scene: 0, cls: null, layer: 'rio', mode: 'both', lon: 0, spin: true};
const S = () => D.scenes[st.scene];
const E = () => S().classes[st.cls];

// ---------------------------------------------------------------------------
// Orthographic projection. Display geometry only: every figure printed on this
// page was formatted in Python, and the globe cannot change one.
// ---------------------------------------------------------------------------
const CX = 150, CY = 150, rad = Math.PI / 180;
let R = 118, lat0 = 62, phase = 0;
// Moving between scenes bends the satellite from one pass onto the next
// instead of teleporting it. Both circles are real; the bend between them is
// a transition, and the caption says so — these acquisitions are years apart.
let orbFrom = null, orbTo = null, orbT = 1;
const ORB_STEP = 0.012;                       // ~1,4 s di deviazione a 60 fps
const ease = x => x * x * (3 - 2 * x);

function orbitNow() {
  if (!orbTo) return null;
  if (!orbFrom || orbT >= 1) return orbTo;
  const t = ease(orbT);
  const A = norm(slerp(orbFrom[0], orbTo[0], t));
  let N = slerp(orbFrom[1], orbTo[1], t);
  N = norm(sub(N, mul(A, dot(A, N))));         // riortogonalizza: resta un cerchio
  return [A, N];
}

function aimAt(f) {
  const next = f && f.track_line ? frameOf(f.track_line) : null;
  if (!next) { orbFrom = null; orbTo = null; orbT = 1; return; }
  const cur = orbitNow();
  orbFrom = cur; orbTo = next; orbT = cur ? 0 : 1;
}

function proj(lon, lat) {
  const p = lat * rad, l = (lon - st.lon) * rad, p0 = lat0 * rad;
  const cosc = Math.sin(p0) * Math.sin(p) + Math.cos(p0) * Math.cos(p) * Math.cos(l);
  if (cosc < 0) return null;                       // dietro al lembo
  return [CX + R * Math.cos(p) * Math.sin(l),
          CY - R * (Math.cos(p0) * Math.sin(p) - Math.sin(p0) * Math.cos(p) * Math.cos(l))];
}

// A great circle through two real points: the sixty-second ground track,
// continued. Nothing orbital is assumed — the circle is the plane those two
// measured points and the centre of the Earth already define. It is drawn
// dashed, and the sixty seconds actually acquired stay solid over it.
function toVec(lon, lat) {
  const p = lat * rad, l = lon * rad;
  return [Math.cos(p) * Math.cos(l), Math.cos(p) * Math.sin(l), Math.sin(p)];
}
function toLonLat(v) {
  return [Math.atan2(v[1], v[0]) / rad,
          Math.asin(Math.max(-1, Math.min(1, v[2]))) / rad];
}
const dot = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
const mul = (a, k) => [a[0] * k, a[1] * k, a[2] * k];
const add = (a, b) => [a[0] + b[0], a[1] + b[1], a[2] + b[2]];
const sub = (a, b) => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
function norm(a) { const m = Math.hypot(a[0], a[1], a[2]); return m ? mul(a, 1 / m) : a; }

// A pass as an orthonormal frame: P(th) = A cos th + N sin th sweeps the whole
// circle. Keeping the frame rather than the two points is what lets one pass
// be interpolated into the next.
function frameOf(trackLine) {
  const A = toVec(trackLine[0][0], trackLine[0][1]);
  const B = toVec(trackLine[1][0], trackLine[1][1]);
  const N = norm(sub(B, mul(A, dot(A, B))));
  return Math.hypot(N[0], N[1], N[2]) ? [A, N] : null;
}

function slerp(a, b, t) {
  let d = Math.max(-1, Math.min(1, dot(a, b)));
  if (d < 0) { b = mul(b, -1); d = -d; }        // via breve
  if (d > 0.9995) return norm(add(mul(a, 1 - t), mul(b, t)));
  const w = Math.acos(d), s = Math.sin(w);
  return add(mul(a, Math.sin((1 - t) * w) / s), mul(b, Math.sin(t * w) / s));
}

function circleOf(fr) {
  const [A, N] = fr;
  return th => toLonLat(add(mul(A, Math.cos(th)), mul(N, Math.sin(th))));
}

function pathOf(pts, close) {
  let d = '', pen = false;
  for (const [lo, la] of pts) {
    const q = proj(lo, la);
    if (!q) { pen = false; continue; }
    d += (pen ? 'L' : 'M') + q[0].toFixed(1) + ' ' + q[1].toFixed(1);
    pen = true;
  }
  return d + (close && d ? 'Z' : '');
}

function graticule() {
  const out = [];
  for (let lon = -180; lon < 180; lon += 30) {
    const pts = []; for (let la = -90; la <= 90; la += 3) pts.push([lon, la]);
    out.push(pathOf(pts, false));
  }
  for (let la = -60; la <= 80; la += 20) {
    const pts = []; for (let lo = -180; lo <= 180; lo += 3) pts.push([lo, la]);
    out.push(pathOf(pts, false));
  }
  return out.filter(Boolean).map(d => `<path class="gr" d="${d}"/>`).join('');
}

function drawGlobe() {
  const g = [];
  g.push(`<circle class="limb" cx="${CX}" cy="${CY}" r="${R}"/>`);
  g.push(graticule());
  const circle = []; for (let lo = -180; lo <= 180; lo += 3) circle.push([lo, D.arctic_circle]);
  const ac = pathOf(circle, false);
  if (ac) g.push(`<path class="arc" d="${ac}"/>`);

  D.scenes.forEach((s, i) => {
    const f = s.footprint; if (!f) return;
    const d = pathOf(f.ring, true);
    if (d) g.push(`<path class="fp${i === st.scene ? ' on' : ''}" d="${d}"/>`);
    const c = proj(f.centre[0], f.centre[1]);
    if (c) g.push(`<circle class="dot${i === st.scene ? ' on' : ''}" `
      + `cx="${c[0].toFixed(1)}" cy="${c[1].toFixed(1)}" r="${i === st.scene ? 3 : 2}"/>`);
  });

  const f = S().footprint, e = E();
  if (f) {
    const t = pathOf(f.track_line, false);
    if (t) g.push(`<path class="trk" d="${t}"/>`);
    if (e && e.route_lonlat) {
      const r = pathOf(e.route_lonlat, false);
      if (r) g.push(`<path class="rt" d="${r}"/>`);
    }
    // The satellite travels the great circle the sixty seconds of real ground
    // track lie on. That circle is geometry those two measured points already
    // imply; the acquisition itself stays solid and bright over it.
    const fr = orbitNow();
    const gc = fr ? circleOf(fr) : null;
    if (gc) {
      const ring = [];
      for (let k = 0; k <= 180; k++) ring.push(gc(k / 180 * 2 * Math.PI));
      const d = pathOf(ring, false);
      if (d) g.push(`<path class="orb${orbT < 1 ? ' moving' : ''}" d="${d}"/>`);

      const th = phase * 2 * Math.PI;
      const here = gc(th), ahead = gc(th + 0.02);
      const q = proj(here[0], here[1]), q2 = proj(ahead[0], ahead[1]);
      if (q) {
        // Fixed screen size on purpose: an icon that grew with the zoom ended
        // up larger than the swath it had just imaged.
        const ang = q2 ? Math.atan2(q2[1] - q[1], q2[0] - q[0]) * 180 / Math.PI : 0;
        g.push(`<g class="satg" transform="translate(${q[0].toFixed(1)} `
          + `${q[1].toFixed(1)}) rotate(${ang.toFixed(1)})">`
          + `<line class="ant" x1="0" y1="0" x2="0" y2="5"/>`
          + `<rect class="pan" x="-1.4" y="-5.2" width="2.8" height="3.4"/>`
          + `<rect class="pan" x="-1.4" y="1.8" width="2.8" height="3.4"/>`
          + `<rect class="body" x="-2.6" y="-1.7" width="5.2" height="3.4" rx=".6"/></g>`);
        g.push(`<text class="lbl" x="${(q[0] + 9).toFixed(1)}" `
          + `y="${(q[1] - 5).toFixed(1)}">${f.platform || ''}</text>`);
      }
    }
  }
  $('#globe').innerHTML = g.join('');
}

// ---------------------------------------------------------------------------
// Panels. Each one places strings Python prepared.
// ---------------------------------------------------------------------------
function dlOf(rows) {
  return '<dl>' + rows.filter(Boolean)
    .map(([k, v, c]) => `<dt>${k}</dt><dd class="${c || ''}">${v}</dd>`).join('') + '</dl>';
}

// Widths arrive as percentages Python already resolved.
function meter(m) {
  if (!m) return '';
  return `<div class="meter"><div class="t"><span>${m.label}</span><b>${m.value}</b></div>`
    + `<div class="track">` + m.segs.map(([w, col]) =>
        `<i style="width:${w}%;background:${col}"></i>`).join('') + `</div></div>`;
}

function render() {
  const s = S(), e = E(), f = s.footprint || {}, stats = s.stats || {};
  document.body.dataset.mode = st.mode;

  $('#tracks').innerHTML = D.scenes.map((x, i) => {
    const ex = x.classes[st.cls] || Object.values(x.classes)[0] || {};
    const base = (ex.comparison || {}).baseline || {};
    const pill = x.worst_case ? ['warn', 'peggiore']
      : !ex.route_exists ? ['bad', 'nessuna']
      : base.blocked_by_ice ? ['ok', 'vietata'] : ['info', 'libera'];
    return `<div class="row${i === st.scene ? ' on' : ''}" data-s="${i}">
      <span class="id">${x.id.slice(0, 8)}<span class="sub"> ${x.region || ''}</span></span>
      <span class="pill ${pill[0]}">${pill[1]}</span></div>`;
  }).join('');

  $('#classes').innerHTML = Object.keys(s.classes).map(k => {
    const x = s.classes[k], b = (x.comparison || {}).baseline || {};
    const pill = !x.route_exists ? ['bad', 'nessuna']
      : b.blocked_by_ice ? ['ok', 'diretta vietata'] : ['info', 'diretta libera'];
    return `<div class="row${k === st.cls ? ' on' : ''}" data-c="${k}">
      <span class="id">${k}</span><span class="pill ${pill[0]}">${pill[1]}</span></div>`;
  }).join('');

  $('#layers').innerHTML = D.layers.filter(l => l[0] !== 'rio' || e).map(l =>
    `<div class="row${l[0] === st.layer ? ' on' : ''}" data-l="${l[0]}">
      <span class="id">${l[1]}</span></div>`).join('');
  $('#modesbox').innerHTML = D.modes.map(m =>
    `<div class="row${m[0] === st.mode ? ' on' : ''}" data-m="${m[0]}">
      <span class="id">${m[1]}</span></div>`).join('');

  const routable = (D.layers.find(l => l[0] === st.layer) || [])[2];
  const name = st.layer === 'rio' ? (e || {}).rio_img : s.files[st.layer];
  $('#img').src = name ? D.images[name] : '';
  $('#img').style.display = name ? '' : 'none';
  $('#ov').innerHTML = (routable && e) ? e.svg : '';
  // Only the geocoded layers are tilted inside their raster; the swath ones
  // are already square with their own grid.
  const fr = routable ? s.frame : null;
  $('#mapin').style.transform = fr
    ? `rotate(${fr.rot}deg) scale(${fr.scale})` : '';
  $('#maptitle').textContent = (D.layers.find(l => l[0] === st.layer) || [0, st.layer])[1];
  $('#mapmeta').textContent = e && e.grid.shape
    ? `${e.grid.shape[0]}×${e.grid.shape[1]} @ ${e.grid.res} m`
    : (s.shape ? `${s.shape[0]}×${s.shape[1]} px` : '');

  const P = e.panels || {};
  $('#prod').innerHTML = P.prod && P.prod.length ? dlOf(P.prod)
    : '<div class="note">Scena grezza assente: nessun metadato di prodotto.</div>';
  $('#onboard').innerHTML = P.onboard && P.onboard.length ? dlOf(P.onboard)
    : '<div class="note">Statistiche di inferenza non registrate per questa scena.</div>';
  $('#polaris').innerHTML = P.rio_meter
    ? meter(P.rio_meter) + dlOf(P.rio_rows || [])
    : '<div class="note">Nessuna rotta per questa classe.</div>';
  $('#classmix').innerHTML = meter(P.mix);
  $('#trace').innerHTML = P.trace
    ? '<div class="bars">' + P.trace.bars.map(([h, c]) =>
        `<i class="${c}" style="height:${h}%"></i>`).join('') + '</div>'
      + `<div class="note">${P.trace.note}</div>`
    : '<div class="note">Nessuna rotta diretta da analizzare.</div>';

  $('#downlink').innerHTML = (P.bridge && P.bridge.rows) ? dlOf(P.bridge.rows)
    : '<div class="note">Nessuna rotta: non c\'è payload.</div>';

  // The transmitted file, byte for byte. Offsets, colouring and byte count all
  // come from Python; the browser places them.
  const br = P.bridge;
  $('#dump').innerHTML = !br ? '<div class="note">Nessuna rotta: niente da trasmettere.</div>'
    : br.mismatch
    ? `<div class="note" style="color:var(--warn-t)">Il file ricostruito pesa ${br.rebuilt} B
       ma il conteggio dice ${br.declared} B: non mostro un dump che contraddice
       il numero accanto.</div>`
    : `<div class="hdr"><b>${br.head}</b>
        <span>${br.ruler ? 'offset in byte' : 'offset non mostrati: file non ASCII'}</span></div>`
      + br.lines.map(([off, hx]) =>
          `<div class="ln"><span class="off">${off}</span><span class="hx">${hx}</span></div>`
        ).join('')
      + (br.note ? `<div class="note">${br.note}</div>` : '');
  $('#paneinfo').textContent = br && br.bytes
    ? `${br.bytes} byte · è tutto ciò che il satellite manda alla nave` : '';

  $('#log').innerHTML = (e.log || []).map(([n, t, m]) =>
    `<div class="l${t === 'AVVISO' ? ' w' : ''}"><span class="n">${n}</span>`
    + `<span class="t">${t}</span><span class="m">${m}</span></div>`).join('');

  $('#b-scene').textContent = s.id;
  $('#b-plat').textContent = f.platform ? `${f.platform} ${f.mode}` : '—';
  $('#b-orbit').textContent = f.orbit || '—';
  $('#f-scene').textContent = s.id;
  $('#f-cls').textContent = st.cls;
  $('#f-layer').textContent = st.layer;
  $('#f-mode').textContent = routable ? st.mode : 'n/d';
  drawGlobe();
}

// ---------------------------------------------------------------------------
document.querySelectorAll('.tabs button').forEach(b => b.onclick = () => {
  document.querySelectorAll('.tabs button').forEach(x => x.classList.toggle('on', x === b));
  $('#dump').hidden = b.dataset.t !== 'bridge';
  $('#log').hidden = b.dataset.t !== 'log';
});

addEventListener('click', ev => {
  const r = ev.target.closest('.row'); if (!r) return;
  if (r.dataset.s != null) {
    st.scene = +r.dataset.s;
    if (!S().classes[st.cls]) st.cls = Object.keys(S().classes)[0];
    // Il globo resta dov'è: a muoversi è il satellite, che devia sulla
    // traiettoria della scena scelta.
    aimAt(S().footprint);
  }
  if (r.dataset.c) st.cls = r.dataset.c;
  if (r.dataset.l) st.layer = r.dataset.l;
  if (r.dataset.m) st.mode = r.dataset.m;
  render();
});

let drag = null;
$('#globe').addEventListener('pointerdown', e => { drag = [e.clientX, e.clientY, st.lon, lat0]; st.spin = false; });
// Il globo non insegue piu' la scena, quindi serve un modo per tornarci.
$('#globe').addEventListener('dblclick', () => {
  const f = S().footprint;
  if (f) { st.lon = f.centre[0]; lat0 = f.centre[1]; drawGlobe(); }
});
$('#globe').addEventListener('wheel', e => {
  e.preventDefault();
  R = Math.max(90, Math.min(1400, R * (e.deltaY < 0 ? 1.12 : 1 / 1.12)));
  drawGlobe();
}, {passive: false});
addEventListener('pointerup', () => { drag = null; });
addEventListener('pointermove', e => {
  if (!drag) return;
  st.lon = drag[2] - (e.clientX - drag[0]) * 0.45;
  lat0 = Math.max(-89, Math.min(89, drag[3] + (e.clientY - drag[1]) * 0.35));
  drawGlobe();
});
$('#spin').onclick = () => { st.spin = !st.spin; $('#spin').textContent = st.spin ? 'ruota ⏸' : 'ruota ▶'; };

function tick() {
  const d = new Date();
  $('#clock').textContent = [d.getHours(), d.getMinutes(), d.getSeconds()]
    .map(x => String(x).padStart(2, '0')).join(':');
  phase = (phase + 0.0011) % 1;    // giro completo in ~15 s, continuo
  if (orbT < 1) orbT = Math.min(1, orbT + ORB_STEP);
  if (!drag) { if (st.spin) st.lon -= 0.12; drawGlobe(); }
  requestAnimationFrame(tick);
}

st.cls = Object.keys(D.scenes[0].classes)[0];
const f0 = D.scenes[0].footprint;
if (f0) { st.lon = f0.centre[0]; lat0 = f0.centre[1]; aimAt(f0); }
render(); tick();
"""


def build_page(ctx: dict) -> str:
    pal = ctx["palette"]
    agg = ctx["aggregate"]
    root = (f":root{{--c-ours:{pal['ours']};--c-base:{pal['base']};"
            f"--c-nogo:{pal['nogo']};--c-ours-fill:{pal['ours']}22}}")

    def cell(k: str, v: str, eid: str = "") -> str:
        i = f' id="{eid}"' if eid else ""
        return (f'<div class="cell"><span class="k">{html.escape(k)}</span>'
                f'<span class="v"{i}>{v}</span></div>')

    leg_cls = "".join(
        f'<span><i style="background:{c}"></i>{html.escape(n)}</span>'
        for n, c in zip(["acqua libera", "ghiaccio sottile", "ghiaccio spesso"],
                        pal["classes"], strict=False))

    miou = sa.dec(agg.get("miou")) if isinstance(agg.get("miou"), float) else "—"
    ckpt = Path(str(ctx.get("checkpoint") or "")).name or "—"

    warn = ""
    if ctx["missing_raw"]:
        warn = ('<div class="note" style="color:var(--warn-t)">Scene grezze assenti: '
                + html.escape(", ".join(s[:14] for s in ctx["missing_raw"]))
                + " — niente impronta sul globo per queste.</div>")

    return f"""<!doctype html><html lang="it"><head><meta charset="utf-8">
<title>IceRoute — console</title>
<meta name="viewport" content="width=device-width,initial-scale=1">
<style>{CSS}{root}</style></head><body data-mode="both">

<div id="app">
  <div id="bar">
    <div class="mark"><i></i>ICEROUTE</div>
    {cell("scena", "—", "b-scene")}
    {cell("piattaforma", "—", "b-plat")}
    {cell("orbita", "—", "b-orbit")}
    {cell("modello", html.escape(ckpt))}
    {cell("miou", miou)}
    <div class="sp"></div>
    <div id="clock">--:--:--</div>
  </div>

  <div class="pane">
    <div class="hd"><h2>Tracce</h2><span class="rt">{len(ctx["scenes"])}</span></div>
    <div class="bd flush" style="overflow:auto" id="tracks"></div>
    <div class="hd"><h2>Classe nave</h2></div>
    <div class="bd flush" id="classes"></div>
    <div class="hd"><h2>Livello</h2></div>
    <div class="bd flush" id="layers"></div>
    <div class="hd"><h2>Rotte</h2></div>
    <div class="bd flush" id="modesbox"></div>
  </div>

  <div class="pane">
    <div class="hd"><h2>Globo</h2>
      <span class="rt">trascina · rotella zoom · doppio clic centra &nbsp;
      <button id="spin" style="background:none;border:0;color:inherit;
        font:inherit;cursor:pointer">ruota ⏸</button></span></div>
    <div class="bd flush"><svg id="globe" viewBox="0 0 300 300"
      preserveAspectRatio="xMidYMid meet"></svg></div>
    <div class="legend">
      <span><i style="background:var(--geo-scene);border-radius:50%"></i>scene</span>
      <span><i class="ln" style="background:var(--geo-scene)"></i>impronta swath</span>
      <span><i class="ln" style="background:var(--geo-track)"></i>acquisizione (60&#8239;s)</span>
      <span><i class="ln" style="background:var(--geo-track-dim)"></i>cerchio massimo del passaggio</span>
      <span><i class="ln" style="background:{pal['ours']}"></i>rotta</span>
      <span><i class="ln" style="background:var(--geo-arc)"></i>Circolo Polare Artico</span>
    </div>
    <div class="bd" style="flex:0 0 auto;border-top:1px solid var(--rule)">
      <div class="note">Reticolato ogni 30°/20° e Circolo Polare Artico a
      {sa.dec(ctx['arctic_circle'], 4)}°: definizioni, non misure. <b>Nessuna linea di
      costa</b> — in questo progetto non c'è un dataset di coste e la pagina non va in rete.
      Impronte, traccia e rotta sono alle coordinate vere lette dalle scene. Il tratto
      pieno è l'acquisizione vera; il tratteggio è <b>lo stesso cerchio massimo
      continuato</b> — geometria che quei due punti già definiscono, non un'orbita
      ricavata da elementi che qui non abbiamo. Cambiando scena il satellite
      <b>devia</b> da un cerchio all'altro: i due passaggi sono veri, la curva
      fra loro è una transizione — queste acquisizioni distano anni, non
      un'orbita.{warn}</div>
    </div>
  </div>

  <div class="pane">
    <div class="hd"><h2 id="maptitle">Vista</h2><span class="rt" id="mapmeta"></span></div>
    <div class="bd flush"><div class="map"><div class="mapin" id="mapin">
      <img id="img" alt="livello"><div id="ov" class="ovwrap"></div></div></div></div>
    <div class="legend">{leg_cls}
      <span><i class="ln" style="background:{pal['base']}"></i>diretta</span>
      <span><i class="ln" style="background:{pal['nogo']}"></i>vietata</span>
      <span><i class="ln" style="background:{pal['ours']}"></i>IceRoute</span>
    </div>
  </div>

  <div class="pane" style="overflow:auto">
    <div class="hd"><h2>Prodotto</h2><span class="rt">Sentinel-1</span></div>
    <div class="bd" id="prod"></div>
    <div class="hd"><h2>A bordo</h2><span class="rt">inferenza</span></div>
    <div class="bd" id="onboard"></div>
    <div class="hd"><h2>POLARIS</h2><span class="rt">MSC.1/Circ.1519</span></div>
    <div class="bd" id="polaris"></div>
    <div class="hd"><h2>Downlink</h2><span class="rt">misurato</span></div>
    <div class="bd" id="downlink"></div>
  </div>

  <div class="pane" style="grid-column:1/3">
    <div class="tabs">
      <button data-t="bridge" class="on">Quello che arriva in plancia</button>
      <button data-t="log">Registro della costruzione</button>
      <span class="rt" id="paneinfo"></span>
    </div>
    <div class="bd" id="dump"></div>
    <div class="bd" id="log" hidden></div>
  </div>

  <div class="pane" style="grid-column:3/5">
    <div class="hd"><h2>Rotta diretta · celle vietate lungo il percorso</h2>
      <span class="rt">misurato cella per cella</span></div>
    <div class="bd">
      <div id="trace"></div>
      <div class="sep"></div>
      <div id="classmix"></div>
    </div>
  </div>

  <div id="foot">
    <div>scena <b id="f-scene">—</b></div>
    <div>classe <b id="f-cls">—</b></div>
    <div>livello <b id="f-layer">—</b></div>
    <div>rotte <b id="f-mode">—</b></div>
    <div>ogni valore da data/demo/index.json e artifacts/metrics.json</div>
  </div>
</div>

<script>const D = {json.dumps(ctx, ensure_ascii=False)};</script>
<script>{JS}</script>
</body></html>"""


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Genera console.html: la demo come console.")
    ap.add_argument("--demo-dir", type=Path, default=DEMO_DIR)
    ap.add_argument("--raw-dir", type=Path, default=RAW_DIR)
    ap.add_argument("--metrics", type=Path, default=METRICS)
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    if not (args.demo_dir / "index.json").exists():
        log.error("%s/index.json assente: lancia make precompute.", args.demo_dir)
        return 1
    metrics = json.loads(args.metrics.read_text()) if args.metrics.exists() else {}
    assumptions = yaml.safe_load(ASSUMPTIONS.read_text()) if ASSUMPTIONS.exists() else {}

    ctx = collect(args.demo_dir, metrics, assumptions, args.raw_dir)
    args.out.write_text(build_page(ctx), encoding="utf-8")

    with_fp = sum(1 for s in ctx["scenes"] if s.get("footprint"))
    log.info("%s — %.1f MB, %d scene, %d con impronta reale",
             args.out, args.out.stat().st_size / 1e6, len(ctx["scenes"]), with_fp)
    if ctx["missing_raw"]:
        log.info("  senza NetCDF grezzo: %s",
                 ", ".join(s[:14] for s in ctx["missing_raw"]))
    log.info("  Doppio clic per aprirla. Nessun server, nessuna rete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
