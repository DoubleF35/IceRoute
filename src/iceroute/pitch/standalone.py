"""Freeze the demo into one HTML file. SPEC §10.4.

``make demo`` needs Python, a virtualenv, Streamlit and a server that stays up.
On 7 October all four of those are things that can go wrong in front of
investors, and the mitigation on the list was "record a backup video" — which
is worse than this in every way: a video cannot be steered when someone asks
"what about a PC7", and it silently goes stale the moment a number changes.

This produces ``demo.html``: one file, a few megabytes, double-click, no
Python, no server, no network. Everything that does not need the model works —
scenes, vessel classes, layers, both routes, the downlink payload, the speed
and fuel dials. "Calcola ora" cannot, and the page says so rather than hiding
the button.

Not a second UI
---------------
It is a *renderer*, and it is built to stay that way, because two things that
draw the same truth eventually disagree — the failure this project has now hit
three times (D-025, D-027, D-031).

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

MAX_PX = 900          # longest edge of an embedded raster
JPEG_QUALITY = 78

LAYERS = [
    ("rio", "RIO — dove si può andare", True),
    ("mask_geo", "Tipo di ghiaccio", True),
    ("sic", "Concentrazione", True),
    ("sar", "SAR", False),
    ("mask_truth", "Carta di riferimento", False),
]
MODES = [("both", "Entrambe"), ("baseline", "Rotta diretta"),
         ("iceroute", "IceRoute"), ("none", "Nessuna")]


def data_uri(path: Path, *, photographic: bool, max_px: int = MAX_PX) -> str:
    """A raster, downscaled and inlined.

    Categorical layers stay PNG and are resampled with nearest neighbour:
    averaging class ids invents classes nobody predicted, which on the RIO layer
    would mean inventing navigability. Only the SAR, which is a photograph, is
    resampled smoothly and stored as JPEG.

    ``max_px`` defaults to the size this page needs; the presentation build asks
    for more, because one raster filling a projector is a different job from
    forty-two thumbnails in a scrolling page.
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

    parts = [f'<svg viewBox="0 0 {cols} {rows}" preserveAspectRatio="none" class="ov">',
             '<g class="lay-baseline">']
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
        t = rescale_transit(ours.get("time_hours"), ours.get("min_rio"),
                            nominal_kn=nominal, chosen_kn=v, ice_class=ice_class)
        values.append(None if t.hours is None else round(t.hours, 2))
        note = note or t.reason
        b = rescale_transit(base.get("time_hours") if base.get("feasible") else None,
                            ours.get("min_rio"), nominal_kn=nominal,
                            chosen_kn=v, ice_class=ice_class)
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

            pay = payload(e, demo_dir / "routes", metrics, s["scene_id"]) if e.get(
                "route_exists") else None
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
                "payload": None if pay is None else {
                    "route": pay.route_bytes, "mask": pay.mask_bytes,
                    "meta": pay.metadata_bytes, "sar": pay.sar_bytes,
                    "total": pay.total_bytes, "ratio": pay.ratio,
                    "n_waypoints": pay.n_waypoints, "text": pay.route_text,
                    "note": pay.note,
                    # Computed here, not in the browser: same rule as the transit
                    # table. If assumptions.yaml has no link rate, this is null and
                    # the page shows nothing rather than a plausible number.
                    "link": (lambda lt: None if lt.seconds is None else {
                        "s": round(lt.seconds), "terminal": lt.terminal,
                        "kbps": lt.kbps, "sbd": lt.sbd_messages,
                    })(link_time(pay.total_bytes, pay.route_bytes, assumptions or {})),
                },
            }

        out_scenes.append({
            "id": s["scene_id"], "region": s.get("region", ""),
            "acquired": (s.get("acquired") or "")[:10],
            "coverage": s.get("coverage"), "worst_case": bool(s.get("worst_case")),
            "files": files, "classes": classes,
        })

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


# ---------------------------------------------------------------------------
# The page
# ---------------------------------------------------------------------------

CSS = """
*{box-sizing:border-box} html,body{margin:0;background:#0a0e14;color:#eaf1fa;
 font:15px/1.5 -apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,Helvetica,Arial,sans-serif}
a{color:#7fb2ff}
.wrap{display:grid;grid-template-columns:230px 1fr;min-height:100vh}
aside{background:#0d131c;border-right:1px solid #1d2633;padding:20px 16px;
 position:sticky;top:0;height:100vh;overflow-y:auto}
aside h3{font-size:11px;letter-spacing:1.2px;text-transform:uppercase;color:#6b7f99;
 margin:20px 0 8px;font-weight:700}
aside h3:first-child{margin-top:0}
label.opt{display:block;padding:5px 8px;border-radius:6px;cursor:pointer;font-size:13.5px;
 color:#9fb1c7}
label.opt:hover{background:#141d2a}
label.opt input{margin-right:8px;accent-color:#ff2d95}
label.opt.on{background:#182234;color:#eaf1fa}
main{padding:24px 30px 60px;max-width:1500px}
h1{font-size:31px;margin:0 0 4px} h1 small{font-size:14px;color:#6b7f99;font-weight:400}
h2{font-size:17px;margin:26px 0 10px}
.grid{display:grid;gap:12px;grid-template-columns:repeat(4,1fr);margin:16px 0}
.kpi{background:#111823;border:1px solid #1d2633;border-radius:12px;padding:15px 17px}
.kpi .lab{font-size:10.5px;letter-spacing:1.2px;text-transform:uppercase;color:#6b7f99;
 font-weight:700;margin-bottom:6px}
.kpi .val{font-size:28px;font-weight:700;font-family:ui-monospace,Menlo,monospace;
 line-height:1.15}
.kpi .sub{font-size:12.5px;color:#8397ae;margin-top:6px}
.tag{display:inline-block;font-size:10px;font-weight:800;letter-spacing:.7px;padding:2px 7px;
 border-radius:4px;text-transform:uppercase}
.ok{background:#0d3b2a;color:#4ade9b}.warn{background:#3d3212;color:#f0b849}
.bad{background:#3b1c1c;color:#f08a8a}.info{background:#1b2735;color:#7d92ad}
.cols{display:grid;grid-template-columns:3fr 1fr;gap:22px;align-items:start}
.map{position:relative;background:#0d131c;border:1px solid #1d2633;border-radius:12px;
 overflow:hidden;line-height:0}
.map img{width:100%;display:block}
.ovwrap{position:absolute;inset:0}
svg.ov{width:100%;height:100%;display:block}
.cas{fill:none;stroke:#080c14;stroke-width:9;vector-effect:non-scaling-stroke;
 stroke-linejoin:round;stroke-linecap:round}
.base{fill:none;stroke:#f5f5f5;stroke-width:2.5;stroke-dasharray:9 6;
 vector-effect:non-scaling-stroke}
.ours{fill:none;stroke:#ff2d95;stroke-width:3.5;vector-effect:non-scaling-stroke;
 stroke-linejoin:round;stroke-linecap:round}
.nogo{fill:none;stroke:#ff2020;stroke-width:4.5;vector-effect:non-scaling-stroke;
 stroke-linecap:round}
.nogod{fill:#ff2020;stroke:#080c14;stroke-width:1.5;vector-effect:non-scaling-stroke}
.wp{fill:#ff2d95;stroke:#080c14;stroke-width:2;vector-effect:non-scaling-stroke}
.end{fill:#fff;stroke:#080c14;stroke-width:2;vector-effect:non-scaling-stroke}
body[data-mode=none] .lay-baseline,body[data-mode=none] .lay-iceroute,
body[data-mode=baseline] .lay-iceroute,body[data-mode=iceroute] .lay-baseline{display:none}
.legend{display:flex;gap:16px;flex-wrap:wrap;margin:10px 0 4px}
.legend span{display:flex;align-items:center;gap:7px;font-size:13px;color:#9fb1c7}
.legend i{width:14px;height:14px;border-radius:3px;display:inline-block}
.note{font-size:13px;color:#8397ae;margin:8px 0}
.box{border-radius:10px;padding:13px 15px;margin:12px 0;font-size:13.5px;line-height:1.55}
.box.i{background:#101d2e;border:1px solid #1d3450;color:#b9cde6}
.box.w{background:#2a2410;border:1px solid #4a3d15;color:#e8cf92}
.box.e{background:#2b1414;border:1px solid #4d2020;color:#f0a8a8}
.box.g{background:#0e2a1e;border:1px solid #17482f;color:#94e0b8}
.tabs{display:flex;gap:4px;margin:26px 0 0;border-bottom:1px solid #1d2633}
.tabs button{background:none;border:0;border-bottom:2px solid transparent;color:#6b7f99;
 padding:9px 15px;font-size:14px;cursor:pointer;font-family:inherit}
.tabs button.on{color:#ff2d95;border-bottom-color:#ff2d95}
.pane{display:none;padding-top:18px} .pane.on{display:block}
table{border-collapse:collapse;font-size:13.5px} td{padding:4px 16px 4px 0;color:#9fb1c7}
td.n{text-align:right;font-family:ui-monospace,Menlo,monospace;color:#eaf1fa}
pre{background:#0d131c;border:1px solid #1d2633;border-radius:8px;padding:13px;
 font-size:11.5px;color:#8fd6a8;overflow-x:auto;white-space:pre-wrap;word-break:break-all}
input[type=range]{width:100%;accent-color:#ff2d95}
.dial{display:grid;grid-template-columns:1fr 1fr;gap:26px;margin:14px 0 4px}
.dial label{font-size:12.5px;color:#9fb1c7}
.dial b{color:#eaf1fa;font-family:ui-monospace,Menlo,monospace}
footer{color:#55677d;font-size:12.5px;border-top:1px solid #1d2633;padding-top:14px;
 margin-top:34px}
@media(max-width:900px){.wrap{grid-template-columns:1fr}aside{position:static;height:auto}
 .grid{grid-template-columns:repeat(2,1fr)}.cols{grid-template-columns:1fr}}
"""

JS = r"""
const $ = s => document.querySelector(s);
const st = {scene: 0, cls: null, layer: 'rio', mode: 'both', tab: 'down'};
const fmt = (v, d = 1) => (typeof v === 'number') ? v.toFixed(d) : 'non misurato';
const B = n => (typeof n === 'number') ? n.toLocaleString('it-IT') + ' B' : 'non misurato';

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

  $('#head').innerHTML = `IceRoute <small>${s.id} · ${s.acquired} · copertura carta
      ${Math.round(100 * (s.coverage || 0))}%</small>`;
  $('#worst').innerHTML = s.worst_case ? `<div class="box w"><b>Questa è la scena su cui il
      modello sbaglia di più</b>, fra le 20 del test ufficiale. È qui apposta: serve a
      rispondere a «e quando sbaglia?» invece di far finta che non succeda. La rotta qui
      sotto <b>non è affidabile</b>.</div>` : '';

  let transit;
  if (!e || !e.route_exists) transit = kpi('Tempo di transito',
      '<span style="color:#f08a8a">nessuna rotta</span>', (e && e.message) || '');
  else if (typeof c.time_saved_hours === 'number' && c.time_saved_hours > 0)
    transit = kpi('Tempo di transito', fmt(ours.time_hours) + ' h',
      `<span style="color:#4ade9b">${c.time_saved_hours.toFixed(1)} h
       (${Math.round(c.time_saved_percent)}%)</span> contro ${fmt(base.time_hours)} h della diretta`);
  else if (base.blocked_by_ice) transit = kpi('Tempo di transito', fmt(ours.time_hours) + ' h',
      `<span class="tag ok">la rotta diretta è vietata dal ghiaccio</span>
       ${base.n_no_go_cells} celle oltre il limite POLARIS`);
  else if (base.feasible) transit = kpi('Tempo di transito', fmt(ours.time_hours) + ' h',
      `<span class="tag info">la rotta diretta è percorribile</span> IceRoute la segue:
       ${fmt(c.detour_km)} km in più`);
  else transit = kpi('Tempo di transito', fmt(ours.time_hours) + ' h',
      '<span class="tag info">confronto non disponibile</span> la diretta esce dalla copertura');

  const rio = ours.min_rio, band = rio == null ? ['info', 'non misurato']
      : rio >= 0 ? ['ok', 'navigazione normale']
      : rio >= -10 ? ['warn', 'rischio elevato'] : ['bad', 'considerazione speciale'];
  $('#kpis').innerHTML = transit
    + kpi('RIO minimo sulla rotta', rio == null ? '—' : (rio >= 0 ? '+' : '') + fmt(rio),
          `<span class="tag ${band[0]}">${band[1]}</span> soglie IMO MSC.1/Circ.1519`)
    + kpi('Volume di downlink', D.compression ? Math.round(D.compression) + ':1' : 'non misurato',
          'mediana · numeratore: SAR compresso senza perdita')
    + kpi('Inferenza a bordo', D.quant.int8_mb ? D.quant.int8_mb.toFixed(2) + ' MB' : 'non misurata',
          'INT8, 1 thread CPU');

  const n = imgName();
  $('#img').src = n ? D.images[n] : '';
  $('#img').style.display = n ? '' : 'none';
  $('#svg').innerHTML = (routable && e) ? e.svg : '';

  const L = {
    rio: [[['Navigazione normale', '#2ecc71'], ['Rischio elevato', '#f1c40f'],
           ['Considerazione speciale', '#e74c3c']],
          'Risk Index Outcome per una nave ' + st.cls + '. <b>È questo il livello che decide</b> dove la nave può andare.'],
    mask_geo: [[['Acqua libera', '#0a2859'], ['Ghiaccio sottile', '#00c8ff'],
                ['Ghiaccio spesso', '#ffffff']],
          'Maschera <b>predetta dal modello</b>, non la carta dell\'analista.'],
    sic: [D.sic_legend, 'Frazione di mare coperta da ghiaccio, in decimi. È <b>l\'altra metà</b> della decisione POLARIS: <code>RIO = C × RIV(tipo) + (10−C) × RIV(acqua libera)</code>.'],
    sar: [[], 'Banda HV, stiramento per percentili. È l\'ingresso del modello.'],
    mask_truth: [[], 'La carta disegnata dall\'analista, solo per confronto. Non entra nel calcolo.'],
  }[st.layer] || [[], ''];
  const routeLeg = routable && st.mode !== 'none' ? [
    ...(st.mode !== 'iceroute' ? [['Rotta diretta', '#f5f5f5'], ['Cella vietata da POLARIS', '#ff2020']] : []),
    ...(st.mode !== 'baseline' ? [['IceRoute', '#ff2d95']] : [])] : [];
  const chip = a => a.map(([t, col]) => `<span><i style="background:${col}"></i>${t}</span>`).join('');
  $('#legend').innerHTML = (routeLeg.length ? `<div class="legend">${chip(routeLeg)}</div>` : '')
      + (L[0].length ? `<div class="legend">${chip(L[0])}</div>` : '')
      + `<div class="note">${L[1]}</div>`;

  const cond = (e && e.conditions) || {};
  $('#cond').innerHTML = (st.layer === 'mask_geo' && cond.worst_class_cells) ? `<div class="box i">
      <b>La rotta attraversa il bianco, ed è regolare.</b>
      ${Math.round(100 * cond.worst_class_cells / cond.n_cells)}% delle sue celle sono
      classificate «${cond.worst_class.toLowerCase()}» — ma a una concentrazione mediana di
      <b>${fmt(cond.worst_class_sic_median, 0)}/10</b>, che dà RIO
      <b>${fmt(cond.worst_class_rio_median, 0)}</b>: navigazione normale.<br><br>
      POLARIS non decide sul tipo di ghiaccio ma su <code>RIO = C × RIV(tipo) + (10−C) ×
      RIV(acqua libera)</code>. Apri <b>Concentrazione</b> per l'altra metà, o <b>RIO</b> per
      il verdetto.<br><br>Sull'intera rotta: <b>${Math.round(100 * cond.fraction_normal)}%
      delle celle in navigazione normale</b>, RIO minimo ${fmt(cond.rio_min, 0)}.</div>` : '';

  $('#side').innerHTML = !e || !e.route_exists
    ? `<div class="box e">${(e && e.message) || 'Nessuna rotta trovata.'}</div>`
    : `<h2>Rotta</h2><div class="kpi"><div class="lab">Distanza</div>
       <div class="val">${Math.round(ours.distance_km)} km</div>
       <div class="sub">${e.n_waypoints} waypoint trasmessi a terra</div></div>
       <h2>Rotta diretta</h2>` + (base.blocked_by_ice
        ? `<div class="box e">Vietata: ${base.n_no_go_cells} celle oltre il limite POLARIS
           per una nave ${st.cls}.</div>`
        : base.feasible
        ? `<div class="box g">Percorribile in ${fmt(base.time_hours)} h. IceRoute devia di
           ${fmt(c.detour_km)} km.</div>`
        : `<div class="box i">Fuori dalla copertura della carta: non è il ghiaccio a fermarla.</div>`);

  renderPanes();
}

function renderPanes() {
  const e = entry(), p = e && e.payload;
  $('#pane-down').innerHTML = !p ? '<div class="box i">Nessuna rotta: non c\'è payload.</div>'
    : `<div class="cols"><div>
        <table><tr><td>Maschera del ghiaccio (PNG)</td><td class="n">${B(p.mask)}</td></tr>
        <tr><td>Rotta (${p.n_waypoints} waypoint, GeoJSON)</td><td class="n">${B(p.route)}</td></tr>
        <tr><td>Metadati</td><td class="n">${B(p.meta)}</td></tr></table>
        ${p.total && p.sar ? `<div class="kpi" style="margin-top:14px">
          <div class="lab">Totale trasmesso, su questa scena</div>
          <div class="val">${Math.round(p.total / 1024)} KB</div>
          <div class="sub">contro ${(p.sar / 1048576).toFixed(1)} MB di SAR compresso senza
          perdita — <b>${Math.round(p.ratio)}:1</b>. La scheda in alto riporta la
          <b>mediana</b>.</div></div>` : `<div class="note">${p.note || ''}</div>`}
        ${p.link ? `<div class="kpi" style="margin-top:12px">
          <div class="lab">Tempo sul collegamento</div>
          <div class="val">&asymp; ${p.link.s} s</div>
          <div class="sub">su ${p.link.terminal} a ${p.link.kbps} kbps, il terminale più
          lento fra quelli in <code>configs/assumptions.yaml</code>.${p.link.sbd ?
          ` La sola rotta sta in <b>${p.link.sbd} messaggi Iridium SBD</b>.` : ''}
          Derivato dalla velocità di picco dichiarata dall'operatore: il throughput reale
          è più basso.</div></div>` : ''}
        <div class="note">La rotta di confronto <b>non</b> è qui dentro: è calcolata a terra
        per argomentare, e un satellite non ha ragione di trasmettere la linea retta fra due
        punti che il destinatario gli ha dato lui.</div></div>
        <div><div class="note">Il file che arriva alla nave, per intero — ${p.route} byte:</div>
        <pre>${p.text.replace(/</g, '&lt;')}</pre></div></div>`;

  const t = e && e.transit;
  if (!t || !t.values.length) { $('#pane-time').innerHTML =
      '<div class="box i">Nessuna rotta: niente da stimare.</div>'; return; }
  const i = +$('#sp').value, v = D.speeds[i], h = t.values[i], bh = t.baseline[i],
        burn = +$('#bu').value, c = (e.comparison || {}), base = c.baseline || {};
  $('#spv').textContent = v.toFixed(1) + ' kn';
  $('#buv').textContent = burn ? burn + ' t/giorno' : 'non impostato';
  const fuel = (h != null && burn) ? h / D.hours_per_day * burn : null;
  $('#cards-time').innerHTML =
      (h == null ? kpi('Transito IceRoute', 'non riscalabile', t.note)
                 : kpi('Transito IceRoute', h.toFixed(1) + ' h',
                       `a ${v.toFixed(1)} kn · ${Math.round((c.iceroute || {}).distance_km)} km`))
    + (base.blocked_by_ice
        ? kpi('Rotta diretta', 'vietata', `${base.n_no_go_cells} celle oltre il limite POLARIS.
             <b>Nessun tempo di riferimento</b>: non esiste un viaggio più lento da cui risparmiare.`)
        : bh != null ? kpi('Rotta diretta', bh.toFixed(1) + ' h', 'percorribile: IceRoute la segue')
        : kpi('Rotta diretta', 'fuori copertura', 'non è il ghiaccio a fermarla'))
    + kpi('Carburante', fuel == null ? 'non impostato' : fuel.toFixed(1) + ' t',
        fuel == null ? 'serve un consumo, e non ne abbiamo uno con una fonte (P-09)'
        : `<span class="tag warn">assunzione tua</span> ${burn} t/giorno ×
           ${(h / D.hours_per_day).toFixed(2)} giorni`);
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


def build_page(ctx: dict) -> str:
    agg = ctx["aggregate"]
    foot = " · ".join(x for x in [
        f"schema: {ctx['scheme']}" if ctx["scheme"] else "",
        f"mIoU {agg['miou']:.3f}" if isinstance(agg.get("miou"), float) else "",
        f"ghiaccio/acqua {agg['binary_accuracy']:.3f}"
        if isinstance(agg.get("binary_accuracy"), float) else "",
        f"{ctx['n_eval_scenes']} scene di test ufficiali" if ctx.get("n_eval_scenes") else "",
    ] if x)

    data = dict(ctx)
    data["layers"] = LAYERS
    data["modes"] = MODES
    data["sic_legend"] = [["0/10 — acqua libera", "#181626"], ["3/10", "#7c2a68"],
                          ["6/10", "#de5c42"], ["10/10 — copertura totale", "#fff078"]]

    return f"""<!doctype html><html lang="it"><head><meta charset="utf-8">
<title>IceRoute — demo</title><meta name="viewport" content="width=device-width,initial-scale=1">
<style>{CSS}</style></head><body data-mode="both">
<div class="wrap">
<aside>
  <h3>Scena</h3><div id="r-scene"></div>
  <h3>Classe della nave</h3><div id="r-cls"></div>
  <h3>Livelli</h3><div id="r-layer"></div>
  <div id="modes"><h3>Rotte sulla mappa</h3><div id="r-mode"></div></div>
  <div id="notroutable" class="note" style="display:none">Su questo livello le rotte non sono
   disegnabili: è il raster della scena grezza, mentre la rotta è calcolata sulla griglia
   geocodificata a 500 m.</div>
</aside>
<main>
  <h1 id="head"></h1>
  <div id="worst"></div>
  <div class="grid" id="kpis"></div>
  <div class="cols">
    <div>
      <div class="map"><img id="img" alt="livello"><div id="svg" class="ovwrap"></div></div>
      <div id="legend"></div>
      <div id="cond"></div>
    </div>
    <div id="side"></div>
  </div>
  <div class="tabs">
    <button data-t="down" class="on">Cosa scende a terra</button>
    <button data-t="time">Tempo e carburante</button>
    <button data-t="live">Calcola ora</button>
  </div>
  <div class="pane on" id="pane-down"></div>
  <div class="pane" id="pane-time">
    <div class="box w"><b>Questi due valori non sono nostre misure.</b> La velocità in acqua
     libera è l'unico numero senza fonte in <code>configs/routing.yaml</code> (P-05) e il
     consumo non è nel progetto (P-09). Muovili: quello che cambia sotto è quanto pesa
     l'assunzione, non quanto è brava IceRoute.</div>
    <div class="dial">
      <div><label>Velocità in acqua libera — <b id="spv"></b></label>
        <input type="range" id="sp" min="0" step="1"></div>
      <div><label>Consumo a velocità piena — <b id="buv"></b></label>
        <input type="range" id="bu" min="0" max="120" step="5" value="0"></div>
    </div>
    <div class="grid" id="cards-time" style="grid-template-columns:repeat(3,1fr)"></div>
    <div class="note">Il riscalamento è <b>esatto</b>, e solo quando la rotta resta tutta in
     operazione normale (RIO ≥ 0): lì ogni tratto va a velocità piena e il tempo è
     proporzionale a 1/v. Se entra in rischio elevato, POLARIS fissa un limite in nodi che il
     cursore non muove, e il numero non viene mostrato invece di essere inventato. Tutti i
     valori sono calcolati in Python a monte: questa pagina li legge, non li ricalcola.</div>
  </div>
  <div class="pane" id="pane-live">
    <div class="box i"><b>Non disponibile in questa copia.</b> Ricalcolare una rotta significa
     rileggere il NetCDF grezzo ed eseguire la rete: servono Python, il modello e circa 15
     secondi di GPU, e nessuna delle tre cose sta dentro un file HTML.<br><br>
     Nella demo vera (<code>make demo</code>) il pulsante c'è, rifà l'intera catena sulla
     scena grezza e confronta il risultato con quello precalcolato — finora identico cella per
     cella. Questa pagina è la <b>rete di sicurezza</b>: serve quando la demo vera non parte.</div>
  </div>
  <footer>{html.escape(foot)}<br>
   Pagina autonoma generata da <code>make demo-static</code>: nessuna risorsa esterna, nessun
   server, funziona con la rete staccata. Ogni numero viene da
   <code>data/demo/index.json</code> e <code>artifacts/metrics.json</code>, gli stessi file
   che legge la demo vera.<br>Modello: <code>{html.escape(str(ctx.get('checkpoint') or '—'))}</code>
  </footer>
</main></div>
<script>const D = {json.dumps(data, ensure_ascii=False)};</script>
<script>{JS}</script>
</body></html>"""


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Congela la demo in un solo file HTML. SPEC §10.4.")
    ap.add_argument("--demo-dir", type=Path, default=Path("data/demo"))
    ap.add_argument("--metrics", type=Path, default=Path("artifacts/metrics.json"))
    ap.add_argument("--out", type=Path, default=Path("demo.html"))
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)-7s %(message)s")

    if not (args.demo_dir / "index.json").exists():
        log.error("%s assente: lancia prima 'make precompute'.", args.demo_dir / "index.json")
        return 1
    metrics = json.loads(args.metrics.read_text()) if args.metrics.exists() else {}
    asm_path = Path("configs/assumptions.yaml")
    assumptions = yaml.safe_load(asm_path.read_text()) if asm_path.exists() else {}

    ctx = collect(args.demo_dir, metrics, assumptions)
    args.out.write_text(build_page(ctx), encoding="utf-8")
    mb = args.out.stat().st_size / 1e6
    log.info("%s — %.1f MB, %d scene, %d immagini incorporate",
             args.out, mb, len(ctx["scenes"]), len(ctx["images"]))
    log.info("Apribile con doppio clic. Nessun server, nessuna rete, nessun Python.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
