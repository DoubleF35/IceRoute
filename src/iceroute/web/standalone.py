"""Freeze the demo into one HTML file that runs with nothing installed.

``make demo`` needs Python, a virtualenv, Streamlit and a server that stays up.
In a live demo with the network off, all four of those are things that can go
wrong, and the usual mitigation — "record a backup video" — is worse than this
in every way: a video cannot be steered when someone asks "what about a PC7",
and it silently goes stale the moment a number changes.

This produces ``demo.html``: one file, a few megabytes, double-click, no
Python, no server, no network. Everything that does not need the model works —
scenes, vessel classes, layers, both routes, the downlink payload, the speed
and fuel dials. "Compute now" cannot, and the page says so rather than hiding
the button.

Not a second UI
---------------
It is a *renderer*, and it is built to stay that way, because two things that
draw the same truth eventually disagree — the failure this project has now hit
three times (docs/decisions.md D-025, D-027, D-031).

* **Every number comes from ``data/demo/index.json``**, the same file the live
  UI reads. Nothing is recomputed here and nothing is typed in.
* **Route geometry and colours come from ``demo/overlay.py``**, the same
  constants the live UI draws with. The SVG is a different medium, not a
  different decision.
* **There is no logic in the JavaScript.** The transit time at each speed, and
  the *refusal* to give one where POLARIS caps the speed, are computed here by
  :func:`estimate.rescale_transit` and baked into a lookup table. The browser
  picks an entry. It cannot round differently, and it cannot decide to show a
  number Python decided to withhold.

The one exception is fuel: hours divided by a day, times a rate the viewer
sets. A single multiplication with no judgement in it.
"""

from __future__ import annotations

import argparse
import base64
import html
import io
import json
import logging
from pathlib import Path

import yaml

from iceroute.demo import overlay as ov
from iceroute.demo.estimate import HOURS_PER_DAY, link_time, payload, rescale_transit

log = logging.getLogger("standalone")

# The dial positions, resolved in Python. Matches the live UI's slider range.
SPEEDS = [round(8.0 + 0.5 * i, 1) for i in range(25)]

MAX_PX = 900  # longest edge of an embedded raster
JPEG_QUALITY = 78

LAYERS = [
    ("rio", "RIO — where the ship may go", True),
    ("mask_geo", "Ice type", True),
    ("sic", "Concentration", True),
    ("sar", "SAR", False),
    ("mask_truth", "Reference chart", False),
]
MODES = [
    ("both", "Both"),
    ("baseline", "Direct route"),
    ("iceroute", "IceRoute"),
    ("none", "None"),
]


def data_uri(path: Path, *, photographic: bool, max_px: int = MAX_PX) -> str:
    """A raster, downscaled and inlined.

    Categorical layers stay PNG and are resampled with nearest neighbour:
    averaging class ids invents classes nobody predicted, which on the RIO layer
    would mean inventing navigability. Only the SAR, which is a photograph, is
    resampled smoothly and stored as JPEG.

    ``max_px`` defaults to the size this page needs; a caller that shows one
    raster full-screen can ask for more, because one raster filling a large
    display is a different job from forty-two thumbnails in a scrolling page.
    """
    from PIL import Image

    img = Image.open(path).convert("RGB")
    if max(img.size) > max_px:
        scale = max_px / max(img.size)
        size = (max(1, round(img.width * scale)), max(1, round(img.height * scale)))
        img = img.resize(size, Image.LANCZOS if photographic else Image.NEAREST)

    buf = io.BytesIO()
    if photographic:
        img.save(buf, format="JPEG", quality=JPEG_QUALITY, optimize=True)
        mime = "jpeg"
    else:
        img.convert("P", palette=Image.ADAPTIVE, colors=32).save(buf, format="PNG", optimize=True)
        mime = "png"
    return f"data:image/{mime};base64,{base64.b64encode(buf.getvalue()).decode()}"


def svg_overlay(data: dict) -> str:
    """The two tracks as SVG, from the same coordinates and colours as the app.

    Widths use ``vector-effect: non-scaling-stroke`` so a line stays legible
    however the browser scales the image — the equivalent of drawing at grid
    resolution in :func:`overlay.draw`.
    """
    if not data:
        return ""
    rows, cols = data["grid_shape"]
    base, ours = data.get("baseline") or {}, data.get("iceroute") or {}

    def pts(cells) -> str:
        return " ".join(f"{c + 0.5},{r + 0.5}" for r, c in cells)

    parts = [
        f'<svg viewBox="0 0 {cols} {rows}" preserveAspectRatio="none" class="ov">',
        '<g class="lay-baseline">',
    ]
    if base.get("path"):
        parts.append(f'<polyline points="{pts(base["path"])}" class="cas"/>')
        parts.append(f'<polyline points="{pts(base["path"])}" class="base"/>')
    for run in ov.no_go_runs(base.get("no_go") or []):
        if len(run) > 2:
            parts.append(f'<polyline points="{pts(run)}" class="nogo"/>')
        else:
            r, c = run[0]
            parts.append(f'<circle cx="{c + 0.5}" cy="{r + 0.5}" r="4" class="nogod"/>')
    parts.append('</g><g class="lay-iceroute">')
    if ours.get("path"):
        parts.append(f'<polyline points="{pts(ours["path"])}" class="cas"/>')
        parts.append(f'<polyline points="{pts(ours["path"])}" class="ours"/>')
    for r, c in ours.get("waypoints") or []:
        parts.append(f'<circle cx="{c + 0.5}" cy="{r + 0.5}" r="4" class="wp"/>')
    parts.append("</g>")
    for key in ("start", "goal"):
        if data.get(key):
            r, c = data[key]
            parts.append(f'<circle cx="{c + 0.5}" cy="{r + 0.5}" r="6" class="end"/>')
    parts.append("</svg>")
    return "".join(parts)


def transit_table(entry: dict, ice_class: str, nominal: float) -> dict:
    """Transit hours at every dial position, or ``null`` where Python refused.

    Baked rather than computed in the browser on purpose. The interesting case
    is the refusal — a route through the elevated risk band has a component
    capped at a fixed number of knots that the dial does not move, and scaling
    it would invent time. That judgement stays in one language.
    """
    ours = (entry.get("comparison") or {}).get("iceroute") or {}
    base = (entry.get("comparison") or {}).get("baseline") or {}
    values, baseline, note = [], [], ""
    for v in SPEEDS:
        t = rescale_transit(
            ours.get("time_hours"),
            ours.get("min_rio"),
            nominal_kn=nominal,
            chosen_kn=v,
            ice_class=ice_class,
        )
        values.append(None if t.hours is None else round(t.hours, 2))
        note = note or t.reason
        b = rescale_transit(
            base.get("time_hours") if base.get("feasible") else None,
            ours.get("min_rio"),
            nominal_kn=nominal,
            chosen_kn=v,
            ice_class=ice_class,
        )
        baseline.append(None if b.hours is None else round(b.hours, 2))
    return {"values": values, "baseline": baseline, "note": note}


def collect(demo_dir: Path, metrics: dict, assumptions: dict | None = None) -> dict:
    """Everything the page shows, read out of the demo build."""
    index = json.loads((demo_dir / "index.json").read_text())
    nominal = float(index.get("open_water_speed_kn") or 12.0)
    scenes = [s for s in index.get("scenes", []) if s.get("classes")]
    scenes.sort(key=lambda s: bool(s.get("worst_case")))

    images: dict[str, str] = {}
    out_scenes = []
    for s in scenes:
        files = s.get("files") or {}
        keys = {"sar": True, "mask_geo": False, "mask_truth": False, "sic": False}
        for key, photographic in keys.items():
            name = files.get(key)
            p = demo_dir / "images" / (name or "")
            if name and p.is_file() and name not in images:
                images[name] = data_uri(p, photographic=photographic)

        classes = {}
        for ic, e in (s.get("classes") or {}).items():
            rio_name = files.get(f"rio_{ic}")
            rp = demo_dir / "images" / (rio_name or "")
            if rio_name and rp.is_file() and rio_name not in images:
                images[rio_name] = data_uri(rp, photographic=False)

            pay = (
                payload(e, demo_dir / "routes", metrics, s["scene_id"])
                if e.get("route_exists")
                else None
            )
            classes[ic] = {
                "route_exists": bool(e.get("route_exists")),
                "message": (e.get("message") or "")[:220],
                "rio_img": rio_name,
                "svg": svg_overlay(ov.load(demo_dir / ov.OVERLAY_DIR / (e.get("overlay") or ""))),
                "comparison": e.get("comparison") or {},
                "n_waypoints": e.get("n_waypoints"),
                "polaris": e.get("polaris") or {},
                "conditions": e.get("conditions") or {},
                "transit": transit_table(e, ic, nominal) if e.get("route_exists") else [],
                "payload": None
                if pay is None
                else {
                    "route": pay.route_bytes,
                    "mask": pay.mask_bytes,
                    "meta": pay.metadata_bytes,
                    "sar": pay.sar_bytes,
                    "total": pay.total_bytes,
                    "ratio": pay.ratio,
                    "n_waypoints": pay.n_waypoints,
                    "text": pay.route_text,
                    "note": pay.note,
                    # Computed here, not in the browser: same rule as the transit
                    # table. If assumptions.yaml has no link rate, this is null and
                    # the page shows nothing rather than a plausible number.
                    "link": (
                        lambda lt: (
                            None
                            if lt.seconds is None
                            else {
                                "s": round(lt.seconds),
                                "terminal": lt.terminal,
                                "kbps": lt.kbps,
                                "sbd": lt.sbd_messages,
                            }
                        )
                    )(link_time(pay.total_bytes, pay.route_bytes, assumptions or {})),
                },
            }

        out_scenes.append(
            {
                "id": s["scene_id"],
                "region": s.get("region", ""),
                "acquired": (s.get("acquired") or "")[:10],
                "coverage": s.get("coverage"),
                "worst_case": bool(s.get("worst_case")),
                "files": files,
                "classes": classes,
            }
        )

    ev = (metrics.get("evaluation") or {}).get("scenes") or {}
    return {
        "scenes": out_scenes,
        "images": images,
        "speeds": SPEEDS,
        "nominal_kn": nominal,
        "hours_per_day": HOURS_PER_DAY,
        "scheme": index.get("scheme", ""),
        "checkpoint": index.get("checkpoint", ""),
        "aggregate": ev.get("aggregate") or {},
        "n_eval_scenes": ev.get("n_scenes"),
        "compression": (metrics.get("compression") or {}).get("ratio_median"),
        "quant": (metrics.get("quantization") or {}).get("size") or {},
    }


def palette() -> dict:
    """Every legend colour, read from the module that paints the raster.

    Retyping them is how a legend starts describing a picture that has moved:
    this file carried ``#0a2859`` for open water, which ``precompute`` paints in
    ``#0a285a``. One channel out is invisible on screen and still turns the
    legend from a reading of the image into a claim about it. The other eleven
    happened to be right, which is the argument for deriving all twelve rather
    than fixing one.
    """
    from iceroute.demo.precompute import CLASS_COLOURS, RIO_COLOURS, SIC_COLOURS

    def h(t):
        return "#{:02x}{:02x}{:02x}".format(*t)

    return {
        "classes": [h(CLASS_COLOURS[i]) for i in sorted(CLASS_COLOURS)],
        "rio": [h(RIO_COLOURS[i]) for i in (0, 1, 2)],
        "sic": {str(k): h(v) for k, v in SIC_COLOURS.items()},
        "ours": h(ov.COL_ICEROUTE),
        "base": h(ov.COL_BASELINE),
        "nogo": h(ov.COL_NO_GO),
    }


# ---------------------------------------------------------------------------
# The page
# ---------------------------------------------------------------------------

# An operator console, not a web page: a command bar, a rail, panels with title
# strips, a status line, 2px corners and one weight of rule. Three choices carry
# most of it.
#
# **The interface accent is blue; magenta belongs to the data.** The tabs, the
# selected rail row and the sliders used to be #ff2d95 — the same colour the
# IceRoute track is drawn in. On a screen whose subject is the map, chrome that
# borrows a series colour makes the eye ask whether a highlighted tab is part of
# the route. Interaction is now #4a9eff and magenta appears only where it means
# "this is our route".
#
# **Numbers are monospaced and tabular everywhere.** Figures that change when
# you move a slider must not reflow the line they sit on.
#
# **Density over air.** 13px base, 2px radii, one hairline between things.
# Measurements sit next to each other so they can be compared, which is what the
# page is for.
CSS = """
:root{
  --bg:#0a0e14; --panel:#0f141c; --raise:#131a24; --line:#1c2531; --rule:#161d27;
  --text:#dbe4ef; --dim:#7e90a6; --faint:#5a6b80; --accent:#4a9eff;
  --mono:ui-monospace,"SF Mono",SFMono-Regular,Menlo,Consolas,monospace;
}
*{box-sizing:border-box}
html,body{margin:0;background:var(--bg);color:var(--text);
  font:13px/1.45 ui-sans-serif,system-ui,-apple-system,"Segoe UI",Roboto,Helvetica,Arial,sans-serif;
  -webkit-font-smoothing:antialiased}
a{color:var(--accent)}
b,strong{color:#eef4fb;font-weight:650}
code{font-family:var(--mono);font-size:.92em;color:#9fc6f0;background:#121a26;
  border:1px solid var(--rule);border-radius:2px;padding:0 4px}
::-webkit-scrollbar{width:10px;height:10px}
::-webkit-scrollbar-track{background:var(--bg)}
::-webkit-scrollbar-thumb{background:#1f2a38;border:2px solid var(--bg);border-radius:0}
::-webkit-scrollbar-thumb:hover{background:#2b3a4d}

/* -- command bar ---------------------------------------------------------- */
#top{position:fixed;top:0;left:0;right:0;height:42px;z-index:20;display:flex;
  align-items:center;gap:14px;padding:0 14px;background:var(--raise);
  border-bottom:1px solid var(--line)}
#top .mark{display:flex;align-items:center;gap:8px;font-weight:700;letter-spacing:.14em;
  font-size:12px;text-transform:uppercase;color:#eef4fb}
#top .mark i{width:9px;height:9px;background:var(--ours);display:inline-block;
  box-shadow:0 0 0 3px #ff2d9524}
#top .what{color:var(--faint);font-size:10.5px;text-transform:uppercase;
  letter-spacing:.14em;border-left:1px solid var(--line);padding-left:14px}
#top .sp{flex:1}
#top .meta{display:flex;gap:0;align-items:stretch;font-size:10.5px}
#top .meta div{padding:0 13px;border-left:1px solid var(--line);display:flex;
  flex-direction:column;justify-content:center;gap:2px}
#top .meta .k{color:var(--faint);text-transform:uppercase;letter-spacing:.12em;font-size:9px}
#top .meta .v{font-family:var(--mono);color:var(--text);font-size:11.5px}

/* -- frame ---------------------------------------------------------------- */
.wrap{display:grid;grid-template-columns:238px 1fr;min-height:100vh;
  padding:42px 0 26px}
aside{background:var(--panel);border-right:1px solid var(--line);
  position:sticky;top:42px;height:calc(100vh - 68px);overflow-y:auto}
aside h3{font-size:9.5px;letter-spacing:.15em;text-transform:uppercase;color:var(--faint);
  margin:0;font-weight:700;padding:9px 12px 8px;background:var(--raise);
  border-bottom:1px solid var(--line);border-top:1px solid var(--line)}
aside h3:first-child{border-top:0}
label.opt{display:flex;align-items:center;gap:8px;padding:6px 12px 6px 10px;cursor:pointer;
  font-size:12.5px;color:var(--dim);border-left:2px solid transparent;
  border-bottom:1px solid var(--rule);font-family:var(--mono);letter-spacing:-.01em}
label.opt:hover{background:#141d29;color:var(--text)}
label.opt input{appearance:none;width:9px;height:9px;margin:0;border:1px solid #3a4a5e;
  border-radius:50%;flex-shrink:0;background:transparent}
label.opt.on{background:#152235;color:#eef4fb;border-left-color:var(--accent)}
label.opt.on input{border-color:var(--accent);background:var(--accent);
  box-shadow:inset 0 0 0 2px #152235}
main{padding:16px 18px 30px;max-width:1680px;min-width:0}

/* -- panels --------------------------------------------------------------- */
.panel{background:var(--panel);border:1px solid var(--line);border-radius:2px;margin-bottom:12px}
.panel>.ph{display:flex;align-items:center;gap:10px;padding:7px 11px;background:var(--raise);
  border-bottom:1px solid var(--line)}
.panel>.ph h2{margin:0;font-size:9.5px;letter-spacing:.15em;text-transform:uppercase;
  color:var(--dim);font-weight:700}
.panel>.ph .rt{margin-left:auto;font-family:var(--mono);font-size:10.5px;color:var(--faint)}
.panel>.pb{padding:11px}
.panel>.pb.flush{padding:0}

h1{font-size:15px;margin:0;font-weight:650;letter-spacing:.01em;font-family:var(--mono)}
h1 small{font-size:11px;color:var(--faint);font-weight:400;font-family:var(--mono)}
h2{font-size:9.5px;letter-spacing:.15em;text-transform:uppercase;color:var(--dim);
  margin:16px 0 7px;font-weight:700}
/* the side column starts on the map panel's top edge, not 16px under it */
#side>h2:first-child{margin-top:0}
/* the transmitted file is the point of this pane: room enough not to wrap mid-token */
#pane-down .cols{grid-template-columns:1.4fr 1fr}

/* -- the metric strip ----------------------------------------------------- */
.grid{display:grid;gap:12px;grid-template-columns:repeat(4,1fr);margin:0 0 12px}
.kpi{background:var(--panel);border:1px solid var(--line);border-radius:2px;padding:10px 12px}
.kpi .lab{font-size:9.5px;letter-spacing:.14em;text-transform:uppercase;color:var(--faint);
  font-weight:700;margin-bottom:6px}
.kpi .val{font-size:23px;font-weight:600;font-family:var(--mono);line-height:1.1;
  color:#eef4fb;font-variant-numeric:tabular-nums;letter-spacing:-.02em}
.kpi .sub{font-size:11.5px;color:var(--dim);margin-top:6px;line-height:1.4}
/* One instrument of four readings, not four cards: a single frame with hairlines
   so the figures sit on a common baseline and can be read across. */
#kpis{gap:0;border:1px solid var(--line);border-radius:2px;background:var(--panel)}
#kpis .kpi{border:0;border-left:1px solid var(--line);border-radius:0;padding:11px 14px}
#kpis .kpi:first-child{border-left:0}

/* -- badges --------------------------------------------------------------- */
.tag{display:inline-block;font-size:9px;font-weight:800;letter-spacing:.1em;padding:2px 6px;
  border-radius:2px;text-transform:uppercase;vertical-align:middle;
  border:1px solid transparent}
.ok{background:#0c2f23;color:#45d19a;border-color:#15523c}
.warn{background:#30280f;color:#e8b54a;border-color:#4e4119}
.bad{background:#2f1717;color:#ec8585;border-color:#532525}
.info{background:#16202c;color:#7d92ad;border-color:#243040}

/* -- map ------------------------------------------------------------------ */
.cols{display:grid;grid-template-columns:3fr 1fr;gap:12px;align-items:start}
.map{position:relative;background:#080c12;overflow:hidden;line-height:0;
  border-bottom:1px solid var(--line)}
.map img{width:100%;display:block}
.ovwrap{position:absolute;inset:0}
svg.ov{width:100%;height:100%;display:block}
.cas{fill:none;stroke:#080c14;stroke-width:9;vector-effect:non-scaling-stroke;
  stroke-linejoin:round;stroke-linecap:round}
.base{fill:none;stroke:var(--c-base);stroke-width:2.5;stroke-dasharray:9 6;
  vector-effect:non-scaling-stroke}
.ours{fill:none;stroke:var(--c-ours);stroke-width:3.5;vector-effect:non-scaling-stroke;
  stroke-linejoin:round;stroke-linecap:round}
.nogo{fill:none;stroke:var(--c-nogo);stroke-width:4.5;vector-effect:non-scaling-stroke;
  stroke-linecap:round}
.nogod{fill:var(--c-nogo);stroke:#080c14;stroke-width:1.5;vector-effect:non-scaling-stroke}
.wp{fill:var(--c-ours);stroke:#080c14;stroke-width:2;vector-effect:non-scaling-stroke}
.end{fill:#fff;stroke:#080c14;stroke-width:2;vector-effect:non-scaling-stroke}
body[data-mode=none] .lay-baseline,body[data-mode=none] .lay-iceroute,
body[data-mode=baseline] .lay-iceroute,body[data-mode=iceroute] .lay-baseline{display:none}

.legend{display:flex;gap:14px;flex-wrap:wrap;margin:0 0 7px}
.legend span{display:flex;align-items:center;gap:6px;font-size:11.5px;color:var(--dim);
  font-family:var(--mono)}
.legend i{width:11px;height:11px;border-radius:2px;display:inline-block;
  border:1px solid #00000066}
.note{font-size:11.5px;color:var(--dim);margin:7px 0;line-height:1.5}

.box{border-radius:2px;padding:10px 12px;margin:9px 0;font-size:12px;line-height:1.55;
  border-left-width:2px;border-left-style:solid}
.box.i{background:#101a28;border:1px solid var(--line);border-left-color:var(--accent);
  color:#b4c8de}
.box.w{background:#231e0f;border:1px solid #3c3414;border-left-color:#e8b54a;color:#e0c589}
.box.e{background:#241212;border:1px solid #401d1d;border-left-color:#ec8585;color:#eaa2a2}
.box.g{background:#0c2219;border:1px solid #16402c;border-left-color:#45d19a;color:#8ed6b0}

/* -- tabs ----------------------------------------------------------------- */
.tabs{display:flex;gap:0;margin:0;background:var(--raise);border:1px solid var(--line);
  border-bottom:0;border-radius:2px 2px 0 0}
.tabs button{background:none;border:0;border-right:1px solid var(--line);
  border-bottom:2px solid transparent;color:var(--faint);padding:8px 16px;font-size:10px;
  cursor:pointer;font-family:inherit;text-transform:uppercase;letter-spacing:.13em;
  font-weight:700}
.tabs button:hover{color:var(--dim);background:#172231}
.tabs button.on{color:#eef4fb;border-bottom-color:var(--accent);background:var(--panel)}
.panes{border:1px solid var(--line);border-top:0;border-radius:0 0 2px 2px;
  background:var(--panel)}
.pane{display:none;padding:13px} .pane.on{display:block}

table{border-collapse:collapse;font-size:12px;width:100%}
td{padding:6px 14px 6px 0;color:var(--dim);border-bottom:1px solid var(--rule)}
tr:last-child td{border-bottom:0}
td.n{text-align:right;font-family:var(--mono);color:#eef4fb;font-variant-numeric:tabular-nums;
  white-space:nowrap;padding-right:0}
pre{background:#080c12;border:1px solid var(--line);border-radius:2px;padding:11px;
  font-size:11px;color:#8fd6a8;overflow-x:auto;white-space:pre-wrap;word-break:break-all;
  font-family:var(--mono);line-height:1.5;max-height:360px;overflow-y:auto}

input[type=range]{width:100%;accent-color:var(--accent);height:3px}
.dial{display:grid;grid-template-columns:1fr 1fr;gap:22px;margin:12px 0 4px}
.dial label{font-size:10px;color:var(--faint);text-transform:uppercase;letter-spacing:.13em;
  font-weight:700;display:block;margin-bottom:7px}
.dial b{color:#eef4fb;font-family:var(--mono);font-size:13px;text-transform:none;
  letter-spacing:0;font-variant-numeric:tabular-nums}

/* -- status line ---------------------------------------------------------- */
#status{position:fixed;left:0;right:0;bottom:0;height:26px;z-index:20;display:flex;
  align-items:center;gap:0;background:var(--raise);border-top:1px solid var(--line);
  font-family:var(--mono);font-size:10.5px;color:var(--faint);overflow:hidden}
#status div{padding:0 12px;border-right:1px solid var(--line);white-space:nowrap;
  line-height:25px}
#status div:last-child{border-right:0;margin-left:auto;color:#3f4e60}
#status b{color:var(--text);font-weight:500}

footer{color:var(--faint);font-size:11px;border-top:1px solid var(--line);padding-top:12px;
  margin-top:18px;line-height:1.6}

@media(max-width:1000px){
  .wrap{grid-template-columns:1fr}
  aside{position:static;height:auto}
  .grid,#kpis{grid-template-columns:repeat(2,1fr)}
  #kpis .kpi:nth-child(3){border-left:0}
  .cols{grid-template-columns:1fr}
  #top .meta div:nth-child(n+2){display:none}
}
"""

JS = r"""
const $ = s => document.querySelector(s);
const st = {scene: 0, cls: null, layer: 'rio', mode: 'both', tab: 'down'};
// One number convention for the whole page: comma-grouped thousands, decimal
// point. It once grouped thousands with an Italian locale ("94.220 B") while
// printing decimals with a dot ("15.6 h") — two conventions in the same card,
// and "94.220" read as a decimal turns the mask into ninety-four bytes. Every
// displayed figure goes through here.
const fmt = (v, d = 1) => (typeof v === 'number') ? v.toFixed(d) : 'not measured';
const B = n => (typeof n === 'number') ? n.toLocaleString('en-US') + ' B' : 'not measured';

function scene() { return D.scenes[st.scene]; }
function entry() { return scene().classes[st.cls]; }

function radios(host, items, key, cur) {
  host.innerHTML = items.map(([v, t]) =>
    `<label class="opt${v === cur ? ' on' : ''}"><input type="radio" name="${key}"
      value="${v}"${v === cur ? ' checked' : ''}>${t}</label>`).join('');
  host.querySelectorAll('input').forEach(i => i.onchange = () => { pick(key, i.value); });
}

function pick(key, v) {
  if (key === 'scene') { st.scene = +v; if (!scene().classes[st.cls]) st.cls = Object.keys(scene().classes)[0]; }
  if (key === 'cls') st.cls = v;
  if (key === 'layer') st.layer = v;
  if (key === 'mode') st.mode = v;
  render();
}

function imgName() {
  const s = scene();
  if (st.layer === 'rio') return entry() ? entry().rio_img : null;
  return s.files[st.layer];
}

function kpi(lab, val, sub) {
  return `<div class="kpi"><div class="lab">${lab}</div><div class="val">${val}</div>
          <div class="sub">${sub || ''}</div></div>`;
}

function render() {
  const s = scene(), e = entry(), c = (e && e.comparison) || {}, ours = c.iceroute || {},
        base = c.baseline || {};
  document.body.dataset.mode = st.mode;

  radios($('#r-scene'), D.scenes.map((x, i) => [i, (x.worst_case ? '⚠ ' : '') +
      x.id.slice(0, 8) + ' · ' + (x.region || '?').slice(0, 20)]), 'scene', st.scene);
  radios($('#r-cls'), Object.keys(s.classes).map(k => [k, k]), 'cls', st.cls);
  radios($('#r-layer'), D.layers.filter(l => l[0] !== 'rio' || e).map(l => [l[0], l[1]]),
         'layer', st.layer);
  const routable = (D.layers.find(l => l[0] === st.layer) || [])[2];
  $('#modes').style.display = routable ? '' : 'none';
  $('#notroutable').style.display = routable ? 'none' : '';
  radios($('#r-mode'), D.modes, 'mode', st.mode);

  $('#head').textContent = s.id;
  $('#headrt').textContent = `${s.acquired} · chart coverage `
      + Math.round(100 * (s.coverage || 0)) + '%';
  $('#worst').innerHTML = s.worst_case ? `<div class="box w"><b>This is the scene the
      model gets most wrong</b>, out of the 20 in the official test set. It is here on
      purpose: to answer "and when it is wrong?" instead of pretending it does not
      happen. The route below <b>is not reliable</b>.</div>` : '';

  let transit;
  if (!e || !e.route_exists) transit = kpi('Transit time',
      '<span style="color:#f08a8a">no route</span>', (e && e.message) || '');
  else if (typeof c.time_saved_hours === 'number' && c.time_saved_hours > 0)
    transit = kpi('Transit time', fmt(ours.time_hours) + ' h',
      `<span style="color:#4ade9b">${fmt(c.time_saved_hours)} h
       (${Math.round(c.time_saved_percent)}%)</span> against ${fmt(base.time_hours)} h for the direct route`);
  else if (base.blocked_by_ice) transit = kpi('Transit time', fmt(ours.time_hours) + ' h',
      `<span class="tag ok">the direct route is forbidden by the ice</span>
       ${base.n_no_go_cells} cells beyond the POLARIS limit`);
  else if (base.feasible) transit = kpi('Transit time', fmt(ours.time_hours) + ' h',
      `<span class="tag info">the direct route is navigable</span> IceRoute follows it:
       ${fmt(c.detour_km)} km longer`);
  else transit = kpi('Transit time', fmt(ours.time_hours) + ' h',
      '<span class="tag info">no comparison available</span> the direct route leaves the coverage');

  const rio = ours.min_rio, band = rio == null ? ['info', 'not measured']
      : rio >= 0 ? ['ok', 'normal operation']
      : rio >= -10 ? ['warn', 'elevated risk'] : ['bad', 'special consideration'];
  $('#kpis').innerHTML = transit
    + kpi('Minimum RIO on the route', rio == null ? '—' : (rio >= 0 ? '+' : '') + fmt(rio),
          `<span class="tag ${band[0]}">${band[1]}</span> IMO MSC.1/Circ.1519 thresholds`)
    + kpi('Downlink volume', D.compression ? Math.round(D.compression) + ':1' : 'not measured',
          'median · numerator: losslessly compressed SAR')
    + kpi('On-board inference', D.quant.int8_mb ? fmt(D.quant.int8_mb, 2) + ' MB' : 'not measured',
          'INT8, 1 CPU thread');

  const n = imgName();
  $('#img').src = n ? D.images[n] : '';
  $('#img').style.display = n ? '' : 'none';
  $('#svg').innerHTML = (routable && e) ? e.svg : '';

  const L = {
    rio: [[['Normal operation', D.palette.rio[0]], ['Elevated risk', D.palette.rio[1]],
           ['Special consideration', D.palette.rio[2]]],
          'Risk Index Outcome for a ' + st.cls + ' vessel. <b>This is the layer that decides</b> where the ship may go.'],
    mask_geo: [[['Open water', D.palette.classes[0]],
                ['Thin ice', D.palette.classes[1]],
                ['Thick ice', D.palette.classes[2]]],
          'Mask <b>predicted by the model</b>, not the analyst\'s chart.'],
    sic: [D.sic_legend, 'Fraction of the sea covered by ice, in tenths. It is <b>the other half</b> of the POLARIS decision: <code>RIO = C × RIV(type) + (10−C) × RIV(open water)</code>.'],
    sar: [[], 'HV band, percentile stretch. This is the model\'s input.'],
    mask_truth: [[], 'The chart drawn by the analyst, for comparison only. It does not enter the computation.'],
  }[st.layer] || [[], ''];
  const routeLeg = routable && st.mode !== 'none' ? [
    ...(st.mode !== 'iceroute' ? [['Direct route', D.palette.base],
                                 ['Cell forbidden by POLARIS', D.palette.nogo]] : []),
    ...(st.mode !== 'baseline' ? [['IceRoute', D.palette.ours]] : [])] : [];
  const chip = a => a.map(([t, col]) => `<span><i style="background:${col}"></i>${t}</span>`).join('');
  $('#legend').innerHTML = (routeLeg.length ? `<div class="legend">${chip(routeLeg)}</div>` : '')
      + (L[0].length ? `<div class="legend">${chip(L[0])}</div>` : '')
      + `<div class="note">${L[1]}</div>`;

  const cond = (e && e.conditions) || {};
  $('#cond').innerHTML = (st.layer === 'mask_geo' && cond.worst_class_cells) ? `<div class="box i">
      <b>The route crosses the white, and that is allowed.</b>
      ${Math.round(100 * cond.worst_class_cells / cond.n_cells)}% of its cells are
      classified "${cond.worst_class.toLowerCase()}" — but at a median concentration of
      <b>${fmt(cond.worst_class_sic_median, 0)}/10</b>, which gives RIO
      <b>${fmt(cond.worst_class_rio_median, 0)}</b>: normal operation.<br><br>
      POLARIS does not decide on the ice type but on <code>RIO = C × RIV(type) + (10−C) ×
      RIV(open water)</code>. Open <b>Concentration</b> for the other half, or <b>RIO</b> for
      the verdict.<br><br>Over the whole route: <b>${Math.round(100 * cond.fraction_normal)}%
      of the cells in normal operation</b>, minimum RIO ${fmt(cond.rio_min, 0)}.</div>` : '';

  $('#side').innerHTML = !e || !e.route_exists
    ? `<div class="box e">${(e && e.message) || 'No route found.'}</div>`
    : `<h2>Route</h2><div class="kpi"><div class="lab">Distance</div>
       <div class="val">${Math.round(ours.distance_km)} km</div>
       <div class="sub">${e.n_waypoints} waypoints transmitted to the ground</div></div>
       <h2>Direct route</h2>` + (base.blocked_by_ice
        ? `<div class="box e">Forbidden: ${base.n_no_go_cells} cells beyond the POLARIS limit
           for a ${st.cls} vessel.</div>`
        : base.feasible
        ? `<div class="box g">Navigable in ${fmt(base.time_hours)} h. IceRoute deviates by
           ${fmt(c.detour_km)} km.</div>`
        : `<div class="box i">Outside the chart's coverage: it is not the ice that stops it.</div>`);

  // Chrome, not data: what is on screen, named. No value appears here that is
  // not already shown, formatted, somewhere else on the page.
  $('#maptitle').textContent = (D.layers.find(l => l[0] === st.layer) || [0, st.layer])[1];
  $('#mapmeta').textContent = routable ? 'geocoded grid · routes drawable'
                                       : 'scene grid · routes not drawable';
  $('#st-scene').textContent = s.id;
  $('#st-cls').textContent = st.cls;
  $('#st-layer').textContent = st.layer;
  $('#st-mode').textContent = routable ? st.mode : 'n/a';

  renderPanes();
}

function renderPanes() {
  const e = entry(), p = e && e.payload;
  $('#pane-down').innerHTML = !p ? '<div class="box i">No route: no payload.</div>'
    : `<div class="cols"><div>
        <table><tr><td>Ice mask (PNG)</td><td class="n">${B(p.mask)}</td></tr>
        <tr><td>Route (${p.n_waypoints} waypoints, GeoJSON)</td><td class="n">${B(p.route)}</td></tr>
        <tr><td>Metadata</td><td class="n">${B(p.meta)}</td></tr></table>
        ${p.total && p.sar ? `<div class="kpi" style="margin-top:14px">
          <div class="lab">Total transmitted, for this scene</div>
          <div class="val">${Math.round(p.total / 1024)} KB</div>
          <div class="sub">against ${fmt(p.sar / 1048576)} MB of losslessly compressed
          SAR — <b>${Math.round(p.ratio)}:1</b>. The card at the top reports the
          <b>median</b>.</div></div>` : `<div class="note">${p.note || ''}</div>`}
        ${p.link ? `<div class="kpi" style="margin-top:12px">
          <div class="lab">Time on the link</div>
          <div class="val">&asymp; ${p.link.s} s</div>
          <div class="sub">on ${p.link.terminal} at ${p.link.kbps} kbps, the slowest
          terminal listed in <code>configs/assumptions.yaml</code>.${p.link.sbd ?
          ` The route alone fits in <b>${p.link.sbd} Iridium SBD messages</b>.` : ''}
          Derived from the operator's advertised peak rate: real throughput
          is lower.</div></div>` : ''}
        <div class="note">The comparison route is <b>not</b> in here: it is computed on the
        ground to make the argument, and a satellite has no reason to transmit the straight
        line between two points the recipient gave it in the first place.</div></div>
        <div><div class="note">The file that reaches the ship, in full — ${p.route} bytes:</div>
        <pre>${p.text.replace(/</g, '&lt;')}</pre></div></div>`;

  const t = e && e.transit;
  if (!t || !t.values.length) { $('#pane-time').innerHTML =
      '<div class="box i">No route: nothing to estimate.</div>'; return; }
  const i = +$('#sp').value, v = D.speeds[i], h = t.values[i], bh = t.baseline[i],
        burn = +$('#bu').value, c = (e.comparison || {}), base = c.baseline || {};
  $('#spv').textContent = fmt(v) + ' kn';
  $('#buv').textContent = burn ? burn + ' t/day' : 'not set';
  const fuel = (h != null && burn) ? h / D.hours_per_day * burn : null;
  $('#cards-time').innerHTML =
      (h == null ? kpi('IceRoute transit', 'cannot be rescaled', t.note)
                 : kpi('IceRoute transit', fmt(h) + ' h',
                       `at ${fmt(v)} kn · ${Math.round((c.iceroute || {}).distance_km)} km`))
    + (base.blocked_by_ice
        ? kpi('Direct route', 'forbidden', `${base.n_no_go_cells} cells beyond the POLARIS limit.
             <b>No reference time</b>: there is no slower voyage to save time against.`)
        : bh != null ? kpi('Direct route', fmt(bh) + ' h', 'navigable: IceRoute follows it')
        : kpi('Direct route', 'outside coverage', 'it is not the ice that stops it'))
    + kpi('Fuel', fuel == null ? 'not set' : fmt(fuel) + ' t',
        fuel == null ? 'needs a consumption rate, and we have none with a source (P-09)'
        : `<span class="tag warn">your assumption</span> ${burn} t/day ×
           ${fmt(h / D.hours_per_day, 2)} days`);
}

function tab(which) {
  st.tab = which;
  document.querySelectorAll('.tabs button').forEach(b => b.classList.toggle('on', b.dataset.t === which));
  document.querySelectorAll('.pane').forEach(p => p.classList.toggle('on', p.id === 'pane-' + which));
}

st.cls = Object.keys(D.scenes[0].classes)[0];
$('#sp').max = D.speeds.length - 1;
$('#sp').value = D.speeds.indexOf(D.nominal_kn);
$('#sp').oninput = renderPanes;
$('#bu').oninput = renderPanes;
document.querySelectorAll('.tabs button').forEach(b => b.onclick = () => tab(b.dataset.t));
render(); tab('down');
"""


def dec(v: float | None, places: int = 3) -> str | None:
    """Fixed decimal places with a decimal point, the page's one number convention."""
    return None if v is None else f"{v:.{places}f}"


def build_page(ctx: dict) -> str:
    agg = ctx["aggregate"]
    pal = palette()
    foot = " · ".join(
        x
        for x in [
            f"scheme: {ctx['scheme']}" if ctx["scheme"] else "",
            f"mIoU {dec(agg.get('miou'))}" if isinstance(agg.get("miou"), float) else "",
            f"ice/water {dec(agg.get('binary_accuracy'))}"
            if isinstance(agg.get("binary_accuracy"), float)
            else "",
            f"{ctx['n_eval_scenes']} official test scenes" if ctx.get("n_eval_scenes") else "",
        ]
        if x
    )

    data = dict(ctx)
    data["layers"] = LAYERS
    data["modes"] = MODES
    data["palette"] = pal
    data["sic_legend"] = [
        ["0/10 — open water", pal["sic"]["0"]],
        ["3/10", pal["sic"]["3"]],
        ["6/10", pal["sic"]["6"]],
        ["10/10 — full cover", pal["sic"]["10"]],
    ]

    # The command bar carries provenance, not measurements: which checkpoint, which
    # scheme, how many held-out scenes. The numbers live in the strip below, where
    # they can be compared; up here is the answer to "what am I looking at".
    def cell(k: str, v: str | None) -> str:
        return (
            (
                f'<div><span class="k">{html.escape(k)}</span>'
                f'<span class="v">{html.escape(v)}</span></div>'
            )
            if v
            else ""
        )

    ckpt = Path(str(ctx.get("checkpoint") or "")).name
    meta = (
        cell("model", ckpt or None)
        + cell("scheme", ctx.get("scheme") or None)
        + cell("test", f"{ctx['n_eval_scenes']} scenes" if ctx.get("n_eval_scenes") else None)
        + cell("miou", dec(agg.get("miou")) if isinstance(agg.get("miou"), float) else None)
    )

    root = (
        f":root{{--c-ours:{pal['ours']};--c-base:{pal['base']};--c-nogo:{pal['nogo']};"
        f"--ours:{pal['ours']}}}"
    )

    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<title>IceRoute — demo</title><meta name="viewport" content="width=device-width,initial-scale=1">
<style>{CSS}{root}</style></head><body data-mode="both">

<div id="top">
  <div class="mark"><i></i>IceRoute</div>
  <div class="what">routing through sea ice · console</div>
  <div class="sp"></div>
  <div class="meta">{meta}</div>
</div>

<div class="wrap">
<aside>
  <h3>Scene</h3><div id="r-scene"></div>
  <h3>Vessel class</h3><div id="r-cls"></div>
  <h3>Layers</h3><div id="r-layer"></div>
  <div id="modes"><h3>Routes on the map</h3><div id="r-mode"></div></div>
  <div id="notroutable" class="note" style="display:none;padding:10px 12px">Routes cannot
   be drawn on this layer: it is the raster of the raw scene, while the route is computed
   on the 500 m geocoded grid.</div>
</aside>

<main>
  <div class="panel"><div class="ph">
    <h1 id="head"></h1>
    <span class="rt" id="headrt"></span>
  </div></div>

  <div id="worst"></div>
  <div class="grid" id="kpis"></div>

  <div class="cols">
    <div>
      <div class="panel">
        <div class="ph"><h2 id="maptitle">View</h2><span class="rt" id="mapmeta"></span></div>
        <div class="map"><img id="img" alt="layer"><div id="svg" class="ovwrap"></div></div>
        <div class="pb"><div id="legend"></div></div>
      </div>
      <div id="cond"></div>
    </div>
    <div id="side"></div>
  </div>

  <div class="tabs">
    <button data-t="down" class="on">What comes down</button>
    <button data-t="time">Time and fuel</button>
    <button data-t="live">Compute now</button>
  </div>
  <div class="panes">
  <div class="pane on" id="pane-down"></div>
  <div class="pane" id="pane-time">
    <div class="box w"><b>These two values are not our measurements.</b> The open-water
     speed is the only unsourced number in <code>configs/routing.yaml</code> (P-05) and
     fuel consumption is not in the project at all (P-09). Move them: what changes below is
     how much the assumption weighs, not how good IceRoute is.</div>
    <div class="dial">
      <div><label>Open-water speed — <b id="spv"></b></label>
        <input type="range" id="sp" min="0" step="1"></div>
      <div><label>Consumption at full speed — <b id="buv"></b></label>
        <input type="range" id="bu" min="0" max="120" step="5" value="0"></div>
    </div>
    <div class="grid" id="cards-time" style="grid-template-columns:repeat(3,1fr)"></div>
    <div class="note">The rescaling is <b>exact</b>, and only when the whole route stays in
     normal operation (RIO ≥ 0): there every leg runs at full speed and the time is
     proportional to 1/v. If it enters elevated risk, POLARIS sets a limit in knots that the
     slider does not move, and the number is withheld instead of being invented. Every value
     is computed upstream in Python: this page reads them, it does not recompute them.</div>
  </div>
  <div class="pane" id="pane-live">
    <div class="box i"><b>Not available in this copy.</b> Recomputing a route means
     re-reading the raw NetCDF and running the network: that needs Python, the model and about
     15 seconds of GPU, and none of the three fits inside an HTML file.<br><br>
     In the full demo (<code>make demo</code>) the button is there: it reruns the whole chain
     on the raw scene and compares the result with the precomputed one — so far identical cell
     for cell. This page is the <b>safety net</b>: it is for when the full demo will not start.</div>
  </div>
  </div>

  <footer>{html.escape(foot)}<br>
   Self-contained page built by <code>make demo-static</code>: no external resources, no
   server, works with the network off. Every number comes from
   <code>data/demo/index.json</code> and <code>artifacts/metrics.json</code>, the same files
   the full demo reads.<br>Model: <code>{html.escape(str(ctx.get("checkpoint") or "—"))}</code>
  </footer>
</main></div>

<div id="status">
  <div>scene <b id="st-scene">—</b></div>
  <div>class <b id="st-cls">—</b></div>
  <div>layer <b id="st-layer">—</b></div>
  <div>routes <b id="st-mode">—</b></div>
  <div>data/demo/index.json · artifacts/metrics.json</div>
</div>

<script>const D = {json.dumps(data, ensure_ascii=False)};</script>
<script>{JS}</script>
</body></html>"""


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Freeze the demo into a single self-contained HTML file."
    )
    ap.add_argument("--demo-dir", type=Path, default=Path("data/demo"))
    ap.add_argument("--metrics", type=Path, default=Path("artifacts/metrics.json"))
    ap.add_argument("--out", type=Path, default=Path("demo.html"))
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)-7s %(message)s")

    if not (args.demo_dir / "index.json").exists():
        log.error("%s missing: run 'make precompute' first.", args.demo_dir / "index.json")
        return 1
    metrics = json.loads(args.metrics.read_text()) if args.metrics.exists() else {}
    asm_path = Path("configs/assumptions.yaml")
    assumptions = yaml.safe_load(asm_path.read_text()) if asm_path.exists() else {}

    ctx = collect(args.demo_dir, metrics, assumptions)
    args.out.write_text(build_page(ctx), encoding="utf-8")
    mb = args.out.stat().st_size / 1e6
    log.info(
        "%s — %.1f MB, %d scenes, %d embedded images",
        args.out,
        mb,
        len(ctx["scenes"]),
        len(ctx["images"]),
    )
    log.info("Opens with a double-click. No server, no network, no Python.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
