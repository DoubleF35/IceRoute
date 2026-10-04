"""Presentation mode: one screen per beat, driven from the keyboard.

Five minutes, a timer, no slides — the product itself is the talk. This builds
``present.html``: a single file that opens with a double click, runs with the
network off, and holds the whole deck's worth of imagery inline.

**Why a static page and not a Streamlit page.** Streamlit has no keyboard
events. The usual workaround puts JavaScript in a ``components.html`` iframe and
has it click a hidden button in the parent document — which works until the
first key press, because changing the slide triggers a rerun, the rerun
recreates the iframe, and the iframe loses focus. The second arrow never
arrives. A presentation remote sends PageDown to whatever holds focus, so that
failure mode is the whole feature. In plain HTML, ``keydown`` on ``document`` is
two lines and cannot be rebuilt out from under itself.

**There is no logic in the JavaScript**, the same contract ``standalone`` keeps.
Every number on screen is formatted in Python from ``data/demo/index.json`` and
``artifacts/metrics.json``; the browser picks a prepared string. A page that
cannot compute cannot compute something wrong in front of an audience.

**Two grids, and they fall either side of a step boundary.** ``sar`` and
``mask_pred`` are the decimated swath; ``mask_geo`` and the routes are EPSG:3413
at 500 m. So step 2 lays the mask on the SAR, where they match, and step 3 moves
to the geocoded raster, where the routes match. Drawing a route over the SAR
would reproduce D-027 exactly — ``overlay.draw`` raises ``GridMismatch`` rather
than let it happen, and the frame visibly shifts between the two steps because
the picture really has changed projection.
"""

from __future__ import annotations

import argparse
import html
import json
import logging
import sys
from pathlib import Path

import yaml

from iceroute.demo import overlay as ov
from iceroute.demo.estimate import link_time, payload
from iceroute.pitch.standalone import data_uri, svg_overlay

log = logging.getLogger("present")

DEMO_DIR = Path("data/demo")
METRICS = Path("artifacts/metrics.json")
ASSUMPTIONS = Path("configs/assumptions.yaml")
OUT = Path("present.html")

STAGE_PX = 1400       # longest edge of a raster that has to fill a projector
THUMB_PX = 420

# Digit grouping uses a non-breaking space on purpose: "169 558 B" must not be
# allowed to wrap between the 169 and the 558 at the end of a line, which on a
# projector reads as two numbers.
NBSP = "\u00a0"


def _rgb(t: tuple[int, int, int]) -> str:
    return "#{:02x}{:02x}{:02x}".format(*t)


def it(n: float | int, decimals: int = 0) -> str:
    """Italian digit grouping. The audience reads 55,3 MB, not 55.3."""
    return f"{n:,.{decimals}f}".replace(",", NBSP).replace(".", ",")


def class_legend(index: dict) -> list[tuple[str, str]]:
    """The legend, built from the palette the masks were actually drawn with.

    Retyping the hex values is how a legend starts describing a picture that has
    moved: the first draft of this file carried ``#0a2859`` for a class drawn in
    ``#0a285a``, copied from the live UI, which carries the same slip. One
    channel out is invisible on a projector and still makes the legend a claim
    about the image rather than a reading of it.
    """
    from iceroute.demo.precompute import CLASS_COLOURS

    names = index.get("class_names") or {}
    out = []
    for cid, rgb in sorted(CLASS_COLOURS.items()):
        name = names.get(str(cid)) or names.get(cid)
        if name:
            out.append((name, _rgb(rgb)))
    return out


def extent(shape, spacing_m: float | None) -> str | None:
    """Scene size on the ground, or nothing.

    ``spacing_m`` comes from ``configs/assumptions.yaml``; the demo index does
    not carry it. Without it this returns None rather than assuming 80 m,
    because a scene extent is a claim like any other.
    """
    if not spacing_m or not shape or len(shape) != 2:
        return None
    return f"{it(shape[0] * spacing_m / 1000)} × {it(shape[1] * spacing_m / 1000)} km"


def route_block(entry: dict, overlay: dict) -> dict:
    """The two routes side by side, each with only what is actually known.

    The direct route is the asymmetric one on purpose. Its transit time is
    ``None`` whenever the ice blocks it — ``precompute`` refuses to time a route
    nobody can sail, which is D-030 — and its share of cells *in normal
    operation* is not recorded anywhere: the index gives forbidden cells and
    no-data cells, and the remainder could be normal or elevated risk. So this
    reports the forbidden count, which is measured and is the point of the
    comparison, and never the complement, which would be a guess.
    """
    cmp_ = entry.get("comparison") or {}
    ours, base = cmp_.get("iceroute") or {}, cmp_.get("baseline") or {}
    cond = entry.get("conditions") or {}

    base_cells = len((overlay.get("baseline") or {}).get("path") or [])
    no_go = base.get("n_no_go_cells")
    no_data = base.get("n_no_data_cells")

    def km(v):
        return f"{it(v, 0)} km" if isinstance(v, int | float) else None

    def hours(v):
        return f"{it(v, 1)} h" if isinstance(v, int | float) else None

    normal = cond.get("fraction_normal")
    return {
        "ours": {
            "distance": km(ours.get("distance_km")),
            "time": hours(ours.get("time_hours")),
            "cells": (f"{it(normal * 100)}% in navigazione normale"
                      if isinstance(normal, float) else None),
            "rio": (f"RIO minimo {ours['min_rio']:+.0f}"
                    if isinstance(ours.get("min_rio"), int | float) else None),
        },
        "base": {
            "distance": km(base.get("distance_km")),
            "time": hours(base.get("time_hours")),
            "blocked": bool(base.get("blocked_by_ice")),
            "cells": (f"{it(no_go)} celle vietate su {it(base_cells)}"
                      if isinstance(no_go, int) and base_cells else None),
            "nodata": (f"{it(no_data)} fuori copertura" if no_data else None),
            "rio": (f"RIO minimo {base['min_rio']:+.0f}"
                    if isinstance(base.get("min_rio"), int | float) else None),
            "note": base.get("note") or "",
        },
        "detour": (f"{it(cmp_['detour_km'])} km in più"
                   if isinstance(cmp_.get("detour_km"), int | float) else None),
    }


def payload_block(entry: dict, metrics: dict, scene_id: str, assumptions: dict) -> dict | None:
    """What leaves the satellite, through the same function the live UI uses."""
    p = payload(entry, DEMO_DIR / "routes", metrics, scene_id)
    if p is None:
        return None
    lt = link_time(p.total_bytes, p.route_bytes, assumptions)
    rows = [("Maschera del ghiaccio (PNG)", p.mask_bytes),
            (f"Rotta · {p.n_waypoints} waypoint", p.route_bytes),
            ("Metadati", p.metadata_bytes)]
    return {
        "rows": [(n, f"{it(v)} B" if isinstance(v, int) else "non misurato") for n, v in rows],
        "total": f"{it(p.total_bytes / 1000, 0)} KB" if p.total_bytes else None,
        "total_bytes": p.total_bytes,
        "route": f"{it(p.route_bytes)} byte",
        "route_bytes": p.route_bytes,
        "sar": f"{it(p.sar_bytes / 1e6, 1)} MB" if p.sar_bytes else None,
        "sar_bytes": p.sar_bytes,
        "ratio": f"{it(p.ratio, 1)}:1" if p.ratio else None,
        "measured": p.measured,
        # Phrased as the live UI phrases it: seconds on the slowest terminal in
        # assumptions.yaml, and the route in SBD messages. Derived from rated
        # peak, which the sentence says out loud rather than implying a stopwatch.
        "link": (f"≈ {lt.seconds:.0f} s su {lt.terminal}"
                 + (f" · la rotta sta in {lt.sbd_messages} messaggi SBD"
                    if lt.sbd_messages else "")
                 if lt.seconds is not None else ""),
        # Bar lengths. Pure presentation, but computed here so the browser is not
        # the thing deciding how big "a lot more" looks.
        "bar": (round(100 * p.total_bytes / p.sar_bytes, 3)
                if p.sar_bytes and p.total_bytes else None),
    }


def collect(demo_dir: Path, metrics: dict, assumptions: dict) -> dict:
    index = json.loads((demo_dir / "index.json").read_text())
    spacing = ((assumptions.get("dataset") or {}).get("pixel_spacing_m"))
    ice_classes = index.get("ice_classes") or ["PC6", "PC7", "IA"]

    scenes, skipped = [], []
    for s in index.get("scenes") or []:
        # The worst-case scene is out of presentation mode: `make report` never
        # measured its compression, so steps 1 and 4 would have two holes in
        # them. It stays in the full demo, where it earns its place.
        if s.get("worst_case"):
            skipped.append(s["scene_id"])
            continue

        files = s.get("files") or {}
        img_dir = demo_dir / "images"

        def uri(key, *, photo, px=STAGE_PX, _f=files, _d=img_dir):
            name = _f.get(key)
            p = _d / name if isinstance(name, str) and name.endswith(".png") else None
            return data_uri(p, photographic=photo, max_px=px) if p and p.is_file() else ""

        def ratio(key, _f=files, _d=img_dir):
            """Aspect ratio of a layer, measured, not assumed.

            A stacked image inside a flex column has no definite height to
            resolve ``max-height:100%`` against, so it overflows its figure and
            gets silently clipped — the bottom of the scene simply is not on the
            projector. Giving the stack the real ratio lets the browser fit the
            box within both limits, and the overlay, which positions against that
            box, lands on the pixels it was computed for.
            """
            from PIL import Image

            name = _f.get(key)
            p = _d / name if isinstance(name, str) and name.endswith(".png") else None
            if not (p and p.is_file()):
                return None
            with Image.open(p) as im:
                return round(im.width / im.height, 6)

        entry_classes = {}
        for ic in ice_classes:
            e = (s.get("classes") or {}).get(ic) or {}
            if not e.get("route_exists"):
                entry_classes[ic] = {"exists": False}
                continue
            o = ov.load(demo_dir / "overlays" / (e.get("overlay") or ""))
            entry_classes[ic] = {
                "exists": True,
                "svg": svg_overlay(o),
                "routes": route_block(e, o),
                "payload": payload_block(e, metrics, s["scene_id"], assumptions),
            }

        scenes.append({
            "id": s["scene_id"],
            "label": s["scene_id"][:8],
            "region": s.get("region", ""),
            "acquired": s.get("acquired", ""),
            "extent": extent(s.get("shape"), spacing),
            "sar": uri("sar", photo=True),
            "mask_pred": uri("mask_pred", photo=False),
            "mask_geo": uri("mask_geo", photo=False),
            "thumb": uri("sar", photo=True, px=THUMB_PX),
            "ar_swath": ratio("sar"),          # sar + mask_pred, stessa griglia
            "ar_geo": ratio("mask_geo"),       # mask_geo + rotte, stessa griglia
            "classes": entry_classes,
        })

    ev = (metrics.get("evaluation") or {}).get("scenes") or {}
    agg = ev.get("aggregate") or {}
    q = (metrics.get("quantization") or {}).get("size") or {}
    return {
        "scenes": scenes, "skipped": skipped, "classes": ice_classes,
        "legend": class_legend(index),
        "route_colours": {"ours": _rgb(ov.COL_ICEROUTE), "base": _rgb(ov.COL_BASELINE),
                          "nogo": _rgb(ov.COL_NO_GO)},
        "model": {
            "int8": f"{it(q['int8_mb'], 2)} MB" if q.get("int8_mb") else None,
            "binary": f"{agg['binary_accuracy']:.3f}".replace(".", ",")
                      if isinstance(agg.get("binary_accuracy"), float) else None,
            "miou": f"{agg['miou']:.3f}".replace(".", ",")
                    if isinstance(agg.get("miou"), float) else None,
            "n_scenes": ev.get("n_scenes"),
            "scheme": index.get("scheme", ""),
        },
    }


# ---------------------------------------------------------------------------
# The page
# ---------------------------------------------------------------------------

CSS = """
*{box-sizing:border-box;margin:0;padding:0}
html,body{height:100%;overflow:hidden;background:#06080c;color:#e8eef7;
  font-family:ui-sans-serif,system-ui,-apple-system,"Segoe UI",Roboto,sans-serif;
  -webkit-font-smoothing:antialiased}
#stage{position:fixed;inset:0;display:flex;align-items:center;justify-content:center}
.step{position:absolute;inset:0;display:none;flex-direction:column;
  padding:clamp(20px,3.2vh,44px) clamp(24px,3.4vw,64px)}
.step.on{display:flex}
.fig{position:relative;flex:1;min-height:0;display:flex;align-items:center;justify-content:center}
.fig img{max-width:100%;max-height:100%;object-fit:contain;display:block}
.stack{position:relative;max-width:100%;max-height:100%;aspect-ratio:var(--ar,1);
  margin:auto}
.stack img{width:100%;height:100%;object-fit:fill;display:block}
.stack .top{position:absolute;inset:0;width:100%;height:100%;opacity:0;
  transition:opacity .9s ease}
.step.on .stack .top.show{opacity:.86}
.ov{position:absolute;inset:0;width:100%;height:100%;overflow:visible}
.ov .cas{fill:none;stroke:#080c14;stroke-width:10;stroke-linejoin:round;
  stroke-linecap:round;vector-effect:non-scaling-stroke}
.ov .base{fill:none;stroke:var(--base);stroke-width:4.5;stroke-dasharray:14 10;
  vector-effect:non-scaling-stroke}
.ov .ours{fill:none;stroke:var(--ours);stroke-width:6;stroke-linejoin:round;
  stroke-linecap:round;vector-effect:non-scaling-stroke}
.ov .nogo{fill:none;stroke:var(--nogo);stroke-width:11;stroke-linecap:round;
  vector-effect:non-scaling-stroke}
.ov .nogod{fill:var(--nogo)}
.ov .wp{fill:var(--ours);stroke:#080c14;stroke-width:1.5;vector-effect:non-scaling-stroke}
.ov .end{fill:#fff;stroke:#080c14;stroke-width:2;vector-effect:non-scaling-stroke}

h1{font-size:clamp(54px,9vw,148px);font-weight:800;letter-spacing:-.035em;line-height:.96}
h2{font-size:clamp(20px,2.5vw,40px);font-weight:700;letter-spacing:-.02em;margin-bottom:.35em}
.lede{font-size:clamp(17px,1.9vw,30px);color:#9fb1c7;margin-top:.6em;max-width:22ch}
.big{font-size:clamp(34px,5.2vw,86px);font-weight:800;letter-spacing:-.03em;line-height:1}
.mid{font-size:clamp(20px,2.4vw,40px);font-weight:700;line-height:1.05}
.cap{font-size:clamp(12px,1.05vw,17px);color:#8ea1b8;letter-spacing:.02em}
.mono{font-variant-numeric:tabular-nums;font-family:ui-monospace,SFMono-Regular,Menlo,monospace}

.bar{display:flex;align-items:baseline;gap:clamp(16px,2vw,34px);margin-bottom:clamp(14px,2.4vh,34px)}
.rail{height:clamp(26px,4.4vh,60px);background:#121926;border-radius:6px;position:relative;flex:1;
  min-width:0;overflow:hidden}
.rail i{position:absolute;inset:0 auto 0 0;display:block;border-radius:6px}
.rail .sar{width:100%;background:linear-gradient(90deg,#1f3a63,#2d5a99)}
/* 170 KB contro 55 MB è lo 0,3%: a scala onesta la barra è una scheggia. Una
   scheggia invisibile però si legge come un errore di rendering, non come un
   numero piccolo — quindi resta in scala ma con un minimo e un alone, così si
   vede che c'è ed è evidente quanto sia poca. */
.rail .pkt{background:var(--ours);min-width:7px;
  box-shadow:0 0 0 1px #fff6,0 0 18px 3px var(--ours)}

.grid{display:grid;gap:clamp(10px,1.4vw,22px);grid-template-columns:repeat(5,1fr);flex:1;min-height:0}
.card{background:#0d131d;border:1px solid #1b2634;border-radius:10px;overflow:hidden;
  display:flex;flex-direction:column;cursor:pointer;transition:border-color .15s,transform .15s}
.card:hover,.card:focus-visible{border-color:var(--ours);transform:translateY(-3px);outline:none}
.card img{width:100%;flex:1;min-height:0;object-fit:cover}
.card b{padding:8px 10px 3px;font-size:clamp(12px,1vw,16px)}
.card span{padding:0 10px 9px;font-size:clamp(10px,.82vw,13px);color:#7d90a8}

.cols{display:flex;gap:clamp(18px,2.6vw,48px);flex:1 1 auto;min-height:0}
.step[data-step="4"] h2{margin-bottom:clamp(14px,2.6vh,36px)}
.step[data-step="4"] .cols{margin-top:clamp(16px,3vh,44px)}
.panel{width:clamp(320px,27vw,500px);display:flex;flex-direction:column;
  justify-content:center;gap:clamp(12px,2.2vh,30px)}
.box{background:#0d131d;border:1px solid #1b2634;border-left-width:4px;border-radius:9px;
  padding:clamp(11px,1.5vh,20px) clamp(13px,1.2vw,22px)}
.box.ours{border-left-color:var(--ours)}
.box.base{border-left-color:var(--base)}
.box h3{font-size:clamp(16px,1.35vw,24px);font-weight:700;margin-bottom:.45em;letter-spacing:.01em}
.box p{font-size:clamp(17px,1.5vw,27px);color:#b9c8da;line-height:1.5}
.box .no{color:#ff6b6b;font-weight:700}

table{border-collapse:collapse;font-size:clamp(14px,1.4vw,24px)}
td{padding:.28em 0}
td:first-child{color:#9fb1c7;padding-right:clamp(18px,2.4vw,46px)}
td:last-child{text-align:right;font-variant-numeric:tabular-nums}
tr.sum td{border-top:1px solid #243246;padding-top:.5em;font-weight:800;color:#e8eef7}

.legend{display:flex;gap:clamp(12px,1.6vw,28px);flex-wrap:wrap;align-items:center}
.legend span{display:flex;align-items:center;gap:8px;font-size:clamp(15px,1.3vw,22px);color:#b9c8da}
.legend i{width:clamp(12px,1.1vw,18px);height:clamp(12px,1.1vw,18px);border-radius:3px;
  display:inline-block;border:1px solid #2a3647}
.legend i.line{height:4px;border-radius:2px;border:none;width:clamp(22px,2vw,34px)}

.corner{position:absolute;right:clamp(24px,3.4vw,64px);top:clamp(20px,3.2vh,44px);
  background:#0d131dcc;border:1px solid #1b2634;border-radius:9px;
  padding:clamp(10px,1.4vh,18px) clamp(13px,1.2vw,22px);text-align:right;backdrop-filter:blur(6px)}
.over{position:absolute;left:clamp(24px,3.4vw,64px);bottom:clamp(20px,3.2vh,44px);
  background:#06080cd9;border-radius:10px;padding:clamp(12px,1.6vh,22px) clamp(16px,1.6vw,30px);
  backdrop-filter:blur(6px)}

#hud{position:fixed;left:0;right:0;bottom:0;display:flex;align-items:center;
  justify-content:space-between;padding:10px clamp(16px,2vw,28px);gap:14px;
  font-size:13px;color:#5e7290;pointer-events:none;z-index:5}
#hud>*{pointer-events:auto}
.chips{display:flex;gap:7px}
.chip{background:#0d131d;border:1px solid #223049;color:#9fb1c7;border-radius:999px;
  padding:5px 13px;font-size:clamp(11px,1vw,15px);cursor:pointer;font-weight:600}
.chip.on{background:var(--ours);border-color:var(--ours);color:#fff}
.nav{display:flex;gap:7px}
.nav button{background:#0d131d;border:1px solid #223049;color:#8ea1b8;border-radius:7px;
  width:34px;height:28px;cursor:pointer;font-size:14px;line-height:1}
.nav button:hover{border-color:#3a5071;color:#e8eef7}
#dots{color:#4e617a;font-size:12px;font-variant-numeric:tabular-nums}
#clock{position:fixed;right:12px;top:12px;z-index:6;display:none;
  background:#0d131dd9;border:1px solid #223049;border-radius:8px;padding:6px 12px;
  font-size:19px;font-weight:700;font-variant-numeric:tabular-nums;color:#8ea1b8}
#clock.on{display:block}
#clock.warn{color:#f1c40f;border-color:#6b5a16}
#clock.over{color:#ff6b6b;border-color:#7a2a2a}
#help{position:fixed;inset:0;background:#06080cf2;z-index:9;display:none;
  align-items:center;justify-content:center}
#help.on{display:flex}
#help table{font-size:clamp(14px,1.3vw,21px)}
#help kbd{background:#141d2b;border:1px solid #28364b;border-radius:5px;padding:2px 9px;
  font-family:ui-monospace,monospace;font-size:.9em}
"""

JS = r"""
const $ = s => document.querySelector(s);
const N = D.scenes.length;
let si = 0, ci = 0, step = 0, t0 = null, tick = null;

const esc = s => (s ?? '');
const scene = () => D.scenes[si];
const klass = () => D.classes[ci];
const entry = () => scene().classes[klass()] || {exists: false};

function paint() {
  const s = scene(), e = entry(), r = e.routes || {}, p = e.payload || {};

  $('#s1img').src = s.sar;
  $('#s1ext').textContent = esc(s.extent);
  $('#s1sar').textContent = esc(p.sar) || '—';
  $('#s1name').textContent = s.label + ' · ' + esc(s.region);
  $('#s1extw').style.display = s.extent ? '' : 'none';
  $('#s1sarw').style.display = p.sar ? '' : 'none';

  $('#s2base').src = s.sar;
  $('#s2top').src = s.mask_pred;
  if (s.ar_swath) $('#s2stack').style.setProperty('--ar', s.ar_swath);
  if (s.ar_geo) $('#s3stack').style.setProperty('--ar', s.ar_geo);

  $('#s3img').src = s.mask_geo;
  $('#s3ov').innerHTML = esc(e.svg);
  const ours = r.ours || {}, base = r.base || {};
  $('#s3ours').innerHTML = [ours.distance, ours.time, ours.cells, ours.rio]
    .filter(Boolean).map(x => `<p>${x}</p>`).join('');
  $('#s3base').innerHTML =
    (base.blocked ? '<p class="no">VIETATA DAL GHIACCIO</p>' : '')
    + [base.distance, base.time, base.cells, base.nodata, base.rio]
      .filter(Boolean).map(x => `<p>${x}</p>`).join('');
  $('#s3detour').textContent = esc(r.detour);

  $('#s4rows').innerHTML = (p.rows || []).map(
    ([n, v]) => `<tr><td>${n}</td><td class="mono">${v}</td></tr>`).join('')
    + (p.total ? `<tr class="sum"><td>Totale trasmesso</td><td class="mono">${p.total}</td></tr>` : '');
  $('#s4sar').textContent = esc(p.sar) || '—';
  $('#s4pkt').textContent = esc(p.total) || '—';
  $('#s4ratio').textContent = esc(p.ratio) || '—';
  $('#s4route').textContent = esc(p.route) || '—';
  $('#s4link').textContent = esc(p.link);
  $('#s4bar').style.width = (p.bar != null ? Math.max(p.bar, 0.35) : 0) + '%';
  $('#s4warn').style.display = (p.measured === false) ? '' : 'none';

  document.querySelectorAll('.chip').forEach((c, i) => c.classList.toggle('on', i === ci));
  $('#dots').textContent = step + '/' + (STEPS - 1);
}

function show(n) {
  step = Math.max(0, Math.min(STEPS - 1, n));
  document.querySelectorAll('.step').forEach(
    el => el.classList.toggle('on', +el.dataset.step === step));
  $('#s2top').classList.remove('show');
  if (step === 2) requestAnimationFrame(() => setTimeout(
    () => $('#s2top').classList.add('show'), 260));
  $('.chips').style.visibility = (step === 3 || step === 4) ? '' : 'hidden';
  paint();
}

function pickScene(i) { si = i; show(1); }
function setClass(i) { ci = i; paint(); }

function fmtClock(ms) {
  const over = ms < 0, s = Math.round(Math.abs(ms) / 1000);
  return (over ? '+' : '') + Math.floor(s / 60) + ':' + String(s % 60).padStart(2, '0');
}
function toggleTimer() {
  const c = $('#clock');
  if (tick) { clearInterval(tick); tick = null; t0 = null; c.classList.remove('on'); return; }
  t0 = Date.now() + 5 * 60 * 1000;
  c.classList.add('on');
  const draw = () => {
    const left = t0 - Date.now();
    c.textContent = fmtClock(left);
    c.classList.toggle('warn', left <= 60000 && left > 0);
    c.classList.toggle('over', left <= 0);
  };
  draw(); tick = setInterval(draw, 1000);
}

addEventListener('keydown', ev => {
  const k = ev.key;
  if (k === '?' || k === 'h' || k === 'H') { $('#help').classList.toggle('on'); return; }
  if ($('#help').classList.contains('on') && k === 'Escape') {
    $('#help').classList.remove('on'); return;
  }
  if (k === 'ArrowRight' || k === 'ArrowDown' || k === ' ' || k === 'PageDown'
      || k === 'Enter') { ev.preventDefault(); show(step + 1); return; }
  if (k === 'ArrowLeft' || k === 'ArrowUp' || k === 'PageUp' || k === 'Backspace') {
    ev.preventDefault(); show(step - 1); return;
  }
  if (k === 't' || k === 'T') { toggleTimer(); return; }
  if (k === 'c' || k === 'C') { setClass((ci + 1) % D.classes.length); return; }
  if (k === 'f' || k === 'F') {
    (document.fullscreenElement ? document.exitFullscreen() :
      document.documentElement.requestFullscreen()).catch(() => {});
    return;
  }
  if (k === 'Home') { show(0); return; }
  if (k >= '1' && k <= '9') {
    const i = +k - 1;
    if (step === 5) { if (i < N) pickScene(i); }
    else if (i < STEPS) show(i);
  }
});

$('#prev').onclick = () => show(step - 1);
$('#next').onclick = () => show(step + 1);
document.querySelectorAll('.chip').forEach((c, i) => c.onclick = () => setClass(i));
document.querySelectorAll('.card').forEach((c, i) => {
  c.onclick = () => pickScene(i);
  c.onkeydown = ev => { if (ev.key === 'Enter') pickScene(i); };
});
show(0);
"""


def build_page(ctx: dict) -> str:
    m, rc = ctx["model"], ctx["route_colours"]
    leg = "".join(f'<span><i style="background:{c}"></i>{html.escape(n)}</span>'
                  for n, c in ctx["legend"])
    chips = "".join(f'<button class="chip">{html.escape(c)}</button>' for c in ctx["classes"])
    cards = "".join(
        f'<div class="card" tabindex="0"><img src="{s["thumb"]}" alt="">'
        f'<b>{html.escape(s["label"])}</b>'
        f'<span>{html.escape(s["region"])} · {html.escape(s["acquired"])}</span></div>'
        for s in ctx["scenes"])

    corner2 = "".join(
        f'<div class="mid">{v}</div><div class="cap">{html.escape(k)}</div>'
        for k, v in [("INT8, a bordo", m["int8"]),
                     ("ghiaccio/acqua, test ufficiale", m["binary"])] if v)
    n_eval = (f'<div class="cap">{m["n_scenes"]} scene mai viste in addestramento</div>'
              if m.get("n_scenes") else "")

    # The charset is not optional: opened with a double click there is no server
    # to send a Content-Type, the browser falls back to latin-1, and every
    # accented word in the page turns to mojibake on the projector.
    return f"""<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>IceRoute — presentazione</title>
<style>:root{{--ours:{rc['ours']};--base:{rc['base']};--nogo:{rc['nogo']}}}{CSS}</style>
<div id="stage">

  <section class="step" data-step="0" style="justify-content:center">
    <h1>IceRoute</h1>
    <p class="lede">Il satellite guarda il ghiaccio, decide a bordo, e manda alla nave la
      rotta. Non l'immagine.</p>
    <p class="cap" style="margin-top:clamp(28px,5vh,70px)">spazio per iniziare · ? per i tasti</p>
  </section>

  <section class="step" data-step="1">
    <div class="fig"><img id="s1img" alt=""></div>
    <div class="over">
      <div class="cap" id="s1name"></div>
      <div class="big" id="s1extw"><span id="s1ext"></span></div>
      <div class="mid" id="s1sarw" style="margin-top:.35em;color:#9fb1c7">
        <span id="s1sar"></span> di radar, compresso senza perdita</div>
    </div>
  </section>

  <section class="step" data-step="2">
    <div class="fig"><span class="stack" id="s2stack">
      <img id="s2base" alt=""><img id="s2top" class="top" alt="">
    </span></div>
    <div class="legend" style="margin-top:clamp(10px,1.6vh,22px)">{leg}</div>
    <div class="corner">{corner2}{n_eval}</div>
  </section>

  <section class="step" data-step="3">
    <div class="cols">
      <div class="fig"><span class="stack" id="s3stack">
        <img id="s3img" alt=""><span id="s3ov"></span>
      </span></div>
      <div class="panel">
        <div class="box base"><h3>Rotta diretta</h3><div id="s3base"></div></div>
        <div class="box ours"><h3>IceRoute</h3><div id="s3ours"></div>
          <p class="cap" style="margin-top:.5em" id="s3detour"></p></div>
        <div class="legend">
          <span><i class="line" style="background:var(--base)"></i>diretta</span>
          <span><i class="line" style="background:var(--nogo)"></i>vietata</span>
          <span><i class="line" style="background:var(--ours)"></i>IceRoute</span>
        </div>
      </div>
    </div>
  </section>

  <section class="step" data-step="4" style="justify-content:center">
    <h2>Cosa scende a terra</h2>
    <div class="bar"><div class="mid" style="width:5.5em">Radar</div>
      <div class="rail"><i class="sar"></i></div>
      <div class="mid mono" id="s4sar" style="width:4.6em;text-align:right"></div></div>
    <div class="bar"><div class="mid" style="width:5.5em">Pacchetto</div>
      <div class="rail"><i class="pkt" id="s4bar"></i></div>
      <div class="mid mono" id="s4pkt" style="width:4.6em;text-align:right"></div></div>
    <div class="cols" style="align-items:center;gap:clamp(24px,5vw,90px);flex:0 0 auto">
      <table><tbody id="s4rows"></tbody></table>
      <div>
        <div class="big" id="s4ratio"></div>
        <div class="cap">su questa scena</div>
        <div class="mid" style="margin-top:.7em"><span id="s4route"></span></div>
        <div class="cap">la sola rotta</div>
        <div class="cap" id="s4link" style="margin-top:.7em;max-width:30ch"></div>
        <div class="cap" id="s4warn" style="color:#f1c40f;margin-top:.5em">
          compressione non misurata su questa scena</div>
      </div>
    </div>
  </section>

  <section class="step" data-step="5">
    <h2>Scegliete voi la scena</h2>
    <div class="grid">{cards}</div>
    <p class="cap" style="margin-top:clamp(10px,1.6vh,20px)">
      clic o 1–{len(ctx['scenes'])} · si riparte dal radar</p>
  </section>

</div>

<div id="clock"></div>
<div id="help"><table><tbody>
  <tr><td><kbd>→</kbd> <kbd>spazio</kbd> <kbd>PagGiù</kbd></td><td>avanti</td></tr>
  <tr><td><kbd>←</kbd> <kbd>PagSu</kbd></td><td>indietro</td></tr>
  <tr><td><kbd>C</kbd></td><td>cambia classe di nave</td></tr>
  <tr><td><kbd>T</kbd></td><td>timer 5 minuti</td></tr>
  <tr><td><kbd>F</kbd></td><td>schermo intero</td></tr>
  <tr><td><kbd>1</kbd>–<kbd>5</kbd></td><td>passo diretto · sulla griglia, scena</td></tr>
  <tr><td><kbd>Home</kbd></td><td>torna al titolo</td></tr>
  <tr><td><kbd>?</kbd></td><td>chiudi</td></tr>
</tbody></table></div>

<div id="hud">
  <div class="chips">{chips}</div>
  <div id="dots"></div>
  <div class="nav"><button id="prev">‹</button><button id="next">›</button></div>
</div>

<script>const D = {json.dumps(ctx, ensure_ascii=False)}; const STEPS = 6;</script>
<script>{JS}</script>
"""


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Genera present.html: modalità presentazione.")
    ap.add_argument("--demo-dir", type=Path, default=DEMO_DIR)
    ap.add_argument("--metrics", type=Path, default=METRICS)
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    if not (args.demo_dir / "index.json").exists():
        log.error("%s/index.json assente: lancia make precompute.", args.demo_dir)
        return 1
    metrics = json.loads(args.metrics.read_text()) if args.metrics.exists() else {}
    assumptions = yaml.safe_load(ASSUMPTIONS.read_text()) if ASSUMPTIONS.exists() else {}

    ctx = collect(args.demo_dir, metrics, assumptions)
    if not ctx["scenes"]:
        log.error("Nessuna scena utilizzabile.")
        return 1
    args.out.write_text(build_page(ctx), encoding="utf-8")

    mb = args.out.stat().st_size / 1e6
    log.info("%s — %.1f MB, %d scene, %d classi di nave",
             args.out, mb, len(ctx["scenes"]), len(ctx["classes"]))
    if ctx["skipped"]:
        log.info("  fuori: %s (compressione non misurata)",
                 ", ".join(s[:14] for s in ctx["skipped"]))
    log.info("  Doppio clic per aprirla. Nessun server, nessuna rete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
