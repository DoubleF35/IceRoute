"""The interactive demo.

Not a tool — a 90-second visual argument. Everything that does not serve the
argument has been left out.

Two rules shaped this file:

* **The main path reads only precomputed files.** No inference, no geocoding,
  no routing while the viewer waits. The demo has to start in under two
  seconds and must not depend on PyTorch initialising during a live demo.

  There are exactly two deliberate exceptions, both behind a button, both of
  which fail into a message box rather than a traceback. **Compute now** re-runs
  the entire chain on the raw NetCDF — the same ``process_scene`` the product
  runs, not a copy — and reports whether the result matches the precomputed one.
  The second times the INT8 model on a single CPU thread, kept separate because
  the GPU in this laptop is not the hardware that would fly.

  Neither weakens the first rule: if either breaks, the precomputed answer is
  still on screen and the demo continues.
* **It never invents a number.** Anything not measured reads as "not measured",
  and anything measured under an assumption without a source is labelled as
  provisional right where it appears. ``configs/assumptions.yaml`` is one click
  away in the footer — deliberately, because it shows rigour and defuses the
  question "where do these numbers come from" before it is asked.

Run with ``make demo``.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import streamlit as st
import yaml

from iceroute.demo import overlay as overlay_mod

DEMO_DIR = Path("data/demo")
ASSUMPTIONS = Path("configs/assumptions.yaml")
METRICS = Path("artifacts/metrics.json")

# Dark, high contrast: demo rooms are often dark and a projector washes out mid
# tones. No pastels anywhere.
CSS = """
<style>
.stApp {background:#0a0e14}
#MainMenu, footer, header {visibility:hidden}
.block-container {padding-top:2.2rem; max-width:1500px}
h1,h2,h3 {color:#eaf1fa !important}
.kpi {background:#111823; border:1px solid #1d2633; border-radius:12px;
      padding:16px 18px; height:100%}
.kpi .lab {font-size:11px; letter-spacing:1.2px; text-transform:uppercase;
           color:#6b7f99; font-weight:700; margin-bottom:6px}
.kpi .val {font-size:30px; font-weight:700; color:#eaf1fa; line-height:1.15;
           font-family:ui-monospace,Menlo,monospace}
.kpi .sub {font-size:12.5px; color:#8397ae; margin-top:6px}
.tag {display:inline-block; font-size:10px; font-weight:800; letter-spacing:.7px;
      padding:2px 7px; border-radius:4px; text-transform:uppercase}
.ok{background:#0d3b2a;color:#4ade9b} .warn{background:#3d3212;color:#f0b849}
.bad{background:#3b1c1c;color:#f08a8a} .info{background:#1b2735;color:#7d92ad}
.legend {display:flex; gap:18px; flex-wrap:wrap; margin:8px 0 2px}
.legend span {display:flex; align-items:center; gap:7px; font-size:13px; color:#9fb1c7}
.legend i {width:15px; height:15px; border-radius:3px; display:inline-block}
.foot {color:#55677d; font-size:12.5px; border-top:1px solid #1d2633;
       padding-top:12px; margin-top:26px}
</style>
"""

CLASS_COLOURS = {
    "Open water": "#0a2859",
    "Thin ice": "#00c8ff",
    "Thick ice": "#ffffff",
}
RIO_LEVELS = [
    ("Normal operation", "#2ecc71"),
    ("Elevated risk", "#f1c40f"),
    ("Special consideration", "#e74c3c"),
]

# Which raster each layer shows, and whether a route can be drawn on it.
#
# Only the geocoded rasters are routable. The SAR and the analyst's chart are
# the raw swath: the route cells are indexed in the 500 m EPSG:3413 grid and
# mean nothing there. Rather than quietly not drawing — the failure this whole
# feature exists to fix — the UI says why the tracks are unavailable.
LAYERS = {
    "RIO — where the ship may go": ("rio_{ice_class}", True),
    "Ice type": ("mask_geo", True),
    "Concentration": ("sic", True),
    "SAR": ("sar", False),
    "Reference chart": ("mask_truth", False),
}
# RIO leads, and the class map is no longer called "the ice mask".
#
# Both changes come from watching someone read the old default and reach the
# wrong conclusion out loud: the route crosses a lot of white, the legend said
# "thick ice", so the route looked illegal. It is not — those cells are
# thick ice at 3/10 concentration, RIO +12 — but the class map cannot show that,
# because POLARIS decides on type *times* concentration and the class map is
# only the type. Leading with the layer that carries the actual decision, and
# naming the other two for what they each are, stops the demo from inviting an
# objection it then has to talk its way out of.
SIC_LEGEND = [
    ("0/10 — open water", "#181626"),
    ("3/10", "#7c2a68"),
    ("6/10", "#de5c42"),
    ("10/10 — full cover", "#fff078"),
]
ROUTE_MODES = {
    "Direct route": "baseline",
    "IceRoute": "iceroute",
    "Both": "both",
    "None": "none",
}
ROUTE_LEGEND = {
    "baseline": [
        ("Direct route (nautical chart)", "#f5f5f5"),
        ("Cell forbidden by POLARIS", "#ff2020"),
    ],
    "iceroute": [("IceRoute", "#ff2d95"), ("Waypoints sent to the ground", "#ff2d95")],
}


# ---------------------------------------------------------------------------
# Loading — cached, so switching scenes never shows a progress bar
# ---------------------------------------------------------------------------


@st.cache_data(show_spinner=False)
def load_index() -> dict:
    path = DEMO_DIR / "index.json"
    return json.loads(path.read_text()) if path.exists() else {}


@st.cache_data(show_spinner=False)
def load_json(path_str: str) -> dict:
    p = Path(path_str)
    return json.loads(p.read_text()) if p.exists() else {}


@st.cache_data(show_spinner=False)
def load_assumptions() -> dict:
    return yaml.safe_load(ASSUMPTIONS.read_text()) if ASSUMPTIONS.exists() else {}


def fmt(v, spec="{:.1f}", missing="not measured"):
    return spec.format(v) if isinstance(v, int | float) else missing


def scene_scores(metrics: dict, scene_id: str) -> tuple[float | None, float | None]:
    """mIoU and worst permissive fraction for one scene, from the evaluation.

    Read from ``metrics.json`` rather than carried in the demo index on purpose:
    these are the numbers the warning banner shows, and a number copied into a
    second file is a number that will still say 0.25 after the next training
    run. If the evaluation has not been run, this returns ``None`` and the
    banner says so instead of inventing a figure.
    """
    per = ((metrics.get("evaluation") or {}).get("scenes") or {}).get("per_scene") or []
    for row in per:
        if row.get("scene_id") == scene_id:
            unsafe = (row.get("rio") or {}).get("worst_unsafe_fraction")
            return row.get("miou"), unsafe
    return None, None


def kpi(label: str, value: str, sub: str = "") -> str:
    return (
        f'<div class="kpi"><div class="lab">{label}</div>'
        f'<div class="val">{value}</div><div class="sub">{sub}</div></div>'
    )


@st.cache_data(show_spinner=False)
def composite(img_path: str, ov_path: str, show: str) -> tuple[bytes | None, str]:
    """The layer with the routes painted on, as PNG bytes plus any complaint.

    This is the one thing the UI computes rather than reads, and it is worth
    being precise about why that does not break the "no runtime work" rule.
    Drawing a polyline is a few hundred integer coordinates through Pillow: no
    model, no torch, no geocoding, no A\\*, a couple of milliseconds, and the
    result is cached per (layer, mode). Precomputing it instead would mean a
    PNG for every scene x class x layer x mode — well over a hundred files, in
    the one directory that is already carrying orphans.

    The error is *returned* rather than raised or logged: the caller has to
    decide what to show, and a cached function that writes to session state
    would report the problem on the first call and stay silent on every reload
    after it — which is exactly the kind of quiet that let the missing overlay
    survive this long.
    """
    import io

    from PIL import Image

    p = Path(img_path)
    if not p.exists():
        return None, f"Image missing: {p.name}"

    img = Image.open(p)
    ov = overlay_mod.load(Path(ov_path)) if ov_path else {}
    try:
        img = overlay_mod.draw(img, ov, show=show)
    except overlay_mod.GridMismatch as exc:
        return None, str(exc)

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue(), ""


# ---------------------------------------------------------------------------
# KPI cards
# ---------------------------------------------------------------------------


def transit_card(entry: dict, provisional_speed: bool) -> str:
    """Transit time, and the delta only when there is an honest one."""
    if not entry.get("route_exists"):
        return kpi(
            "Transit time",
            '<span style="color:#f08a8a">no route</span>',
            entry.get("message", "")[:90],
        )

    cmp_ = entry["comparison"]
    ours = cmp_["iceroute"]["time_hours"]
    # Points at the tab that recomputes this, because otherwise the page shows
    # two different transit times for the same route and neither says why.
    note = (
        ("assumes 12 kn in open water (unsourced) — change it under “Time and fuel”")
        if provisional_speed
        else ""
    )

    if cmp_.get("time_saved_hours") is not None:
        saved = cmp_["time_saved_hours"]
        pct = cmp_.get("time_saved_percent")
        base_h = fmt(cmp_["baseline"]["time_hours"])
        if saved > 0:
            body = (
                f'<span style="color:#4ade9b">{saved:+.1f} h ({pct:+.0f}%)</span> '
                f"against {base_h} h on the direct route"
            )
        else:
            # The direct route is legal and IceRoute all but follows it. There
            # is nothing to win here, and dressing a 1% loss in warning colours
            # would be editorialising a non-event. The honest reading is the
            # better one anyway: no detour was invented where none was needed.
            body = (
                '<span class="tag info">the direct route is navigable</span> '
                f"IceRoute follows it: {cmp_['detour_km']:+.1f} km and "
                f"{abs(saved):.1f} h longer than {base_h} h"
            )
        return kpi("Transit time", f"{fmt(ours)} h", body + (f"<br>{note}" if note else ""))

    # No comparable baseline. Say which of the two reasons applies — they mean
    # completely different things and conflating them would be a false claim.
    base = cmp_["baseline"]
    if base.get("blocked_by_ice"):
        sub = (
            '<span class="tag ok">the direct route is blocked by ice</span> '
            f"{base['n_no_go_cells']} cells beyond the POLARIS limit"
        )
    else:
        sub = (
            '<span class="tag info">comparison not available</span> '
            "the direct route leaves the chart coverage: it is not the ice "
            "that stops it"
        )
    return kpi("Transit time", f"{fmt(ours)} h", sub + (f"<br>{note}" if note else ""))


def rio_card(entry: dict) -> str:
    if not entry.get("route_exists"):
        return kpi("Minimum RIO on route", "—", "no route to assess")
    rio = entry["comparison"]["iceroute"].get("min_rio")
    if rio is None:
        return kpi("Minimum RIO on route", "not measured", "")
    if rio >= 0:
        tag, txt = "ok", "normal operation"
    elif rio >= -10:
        tag, txt = "warn", "elevated risk — POLARIS speed limit"
    else:
        tag, txt = "bad", "special consideration"
    return kpi(
        "Minimum RIO on route",
        f"{rio:+.1f}",
        f'<span class="tag {tag}">{txt}</span> IMO MSC.1/Circ.1519 thresholds',
    )


def downlink_card(metrics: dict) -> str:
    comp = metrics.get("compression") or {}
    ratio = comp.get("ratio_median")
    if not ratio:
        return kpi("Downlink volume", "not measured", "needs <code>make report</code>")
    prov = comp.get("provisional")
    sub = f"{comp.get('n_scenes', 0)} scenes · numerator: losslessly compressed SAR"
    if prov:
        sub = '<span class="tag warn">upper bound</span> ' + sub
    return kpi("Downlink volume", f"{ratio:.0f}:1", sub)


def onboard_card(metrics: dict) -> str:
    q = metrics.get("quantization") or {}
    size = (q.get("size") or {}).get("int8_mb")
    lat = ((q.get("latency") or {}).get("single_thread") or {}).get("int8", {})
    ms = lat.get("median_ms_per_tile")
    if not size:
        return kpi("On-board inference", "not measured", "needs <code>make quantize</code>")
    acc = q.get("accuracy") or {}
    drop = acc.get("miou_drop")
    extra = (
        f" · INT8 costs {abs(drop):.3f} mIoU"
        if isinstance(drop, float) and drop > 0
        else " · INT8 does not degrade measurably"
    )
    return kpi(
        "On-board inference",
        f"{size:.2f} MB",
        f"{fmt(ms, '{:.0f}')} ms per tile, 1 CPU thread{extra}",
    )


# ---------------------------------------------------------------------------
# Page
# ---------------------------------------------------------------------------


def main() -> None:
    st.set_page_config(page_title="IceRoute", layout="wide", initial_sidebar_state="expanded")
    st.markdown(CSS, unsafe_allow_html=True)

    index = load_index()
    if not index or not index.get("scenes"):
        st.title("IceRoute")
        st.error(
            "No precomputed data in `data/demo/`. Run **`make precompute`**.\n\n"
            "The UI only reads files generated beforehand: it computes nothing live, "
            "so it starts in under two seconds and does not depend on PyTorch at demo time."
        )
        return

    scenes = [s for s in index["scenes"] if s.get("classes")]
    # The worst case goes last, always. Left in the normal rotation it would
    # read as a representative scene, which is the opposite of why it is here.
    scenes.sort(key=lambda s: bool(s.get("worst_case")))
    if not scenes:
        st.title("IceRoute")
        st.error("No usable scene: none of them has a route.")
        return

    metrics = load_json(str(METRICS))
    assumptions = load_assumptions()
    prov_speed = bool(index.get("open_water_speed_is_provisional"))

    # -- sidebar ------------------------------------------------------------
    with st.sidebar:
        st.markdown("### Scene")
        labels = {
            f"{'⚠ ' if s.get('worst_case') else ''}{s['scene_id'][:8]}"
            f" · {s.get('region', '?')[:22]}": s
            for s in scenes
        }
        chosen = st.radio("Scene", list(labels), label_visibility="collapsed")
        scene = labels[chosen]

        st.markdown("### Vessel class")
        available = list(scene["classes"])
        # If the route never changes with the class, the selector is theatre.
        # Say so rather than letting someone discover it mid-demo.
        if not scene.get("routes_differ") and len(available) > 1:
            st.caption(
                "On this scene the route does not change with the class: the open-water "
                "corridor is wide enough for all of them."
            )
        ice_class = st.radio("Class", available, label_visibility="collapsed")

        st.markdown("### Layers")
        # RIO leads: the class map alone cannot show passability (D-031).
        layer = st.radio("Layer", list(LAYERS), index=0, label_visibility="collapsed")
        key_tpl, routable = LAYERS[layer]

        st.markdown("### Routes on the map")
        if routable:
            mode = ROUTE_MODES[
                st.radio("Routes", list(ROUTE_MODES), index=2, label_visibility="collapsed")
            ]
        else:
            mode = "none"
            st.caption(
                "Routes cannot be drawn on this layer: it is the raster of the raw "
                "scene, while the route is computed on the geocoded 500 m grid. "
                "Switch to **Ice type** or **RIO**."
            )

    entry = scene["classes"][ice_class]

    # -- header + KPI -------------------------------------------------------
    st.markdown(
        f"# IceRoute <span style='font-size:16px;color:#6b7f99'>"
        f"{scene['scene_id']} · {scene.get('acquired', '')[:10]} · "
        f"chart coverage {100 * scene.get('coverage', 0):.0f}%</span>",
        unsafe_allow_html=True,
    )

    if scene.get("worst_case"):
        miou, unsafe = scene_scores(metrics, scene["scene_id"])
        agg = (((metrics.get("evaluation") or {}).get("scenes") or {}).get("aggregate") or {}).get(
            "miou"
        )
        unsafe_pct = 100 * unsafe if isinstance(unsafe, int | float) else None
        st.warning(
            "**This is the scene the model gets most wrong** of the 20 in the "
            "official test set. It is here on purpose: it answers “and when does it "
            "fail?” instead of pretending that it does not.\n\n"
            f"mIoU **{fmt(miou, '{:.2f}')}** against a mean of **{fmt(agg, '{:.2f}')}**. "
            f"Permissive fraction **{fmt(unsafe_pct, '{:.0f}%')}**: "
            "on this share of pixels the model declares navigable ice that POLARIS "
            "forbids for that vessel class. The route below **is not reliable**, "
            "and that is why it is shown."
        )

    cols = st.columns(4, gap="small")
    for col, html_ in zip(
        cols,
        [
            transit_card(entry, prov_speed),
            rio_card(entry),
            downlink_card(metrics),
            onboard_card(metrics),
        ],
        strict=True,
    ):
        col.markdown(html_, unsafe_allow_html=True)

    st.write("")

    # -- map ----------------------------------------------------------------
    left, right = st.columns([3, 1], gap="medium")
    with left:
        key = key_tpl.format(ice_class=ice_class)
        img = DEMO_DIR / "images" / scene["files"].get(key, "")
        # No route on this scene/class means no overlay file; the layer still
        # renders, which is the point — "no route" is a real outcome.
        name = entry.get("overlay") or ""
        ov = str(DEMO_DIR / overlay_mod.OVERLAY_DIR / name) if name and mode != "none" else ""
        png, problem = composite(str(img), ov, mode)
        if png:
            st.image(png, use_container_width=True)
        else:
            st.warning(problem or f"Image missing: {key}")

        if mode != "none":
            shown = [x for m, x in ROUTE_LEGEND.items() if mode in (m, "both")]
            st.markdown(
                '<div class="legend">'
                + "".join(
                    f'<span><i style="background:{c}"></i>{n}</span>'
                    for group in shown
                    for n, c in group
                )
                + "</div>",
                unsafe_allow_html=True,
            )

        if layer == "Ice type":
            st.markdown(
                '<div class="legend">'
                + "".join(
                    f'<span><i style="background:{c}"></i>{n}</span>'
                    for n, c in CLASS_COLOURS.items()
                )
                + "</div>",
                unsafe_allow_html=True,
            )
            st.caption(
                "Mask **predicted by the model**, not the analyst's chart, "
                "reprojected to EPSG:3413 at 500 m."
            )
            conditions_note(entry, ice_class)
        elif layer.startswith("RIO"):
            st.markdown(
                '<div class="legend">'
                + "".join(f'<span><i style="background:{c}"></i>{n}</span>' for n, c in RIO_LEVELS)
                + "</div>",
                unsafe_allow_html=True,
            )
            st.caption(
                f"Risk Index Outcome for a {ice_class} vessel, IMO MSC.1/Circ.1519 "
                f"thresholds. **This is the layer that decides** where the ship may "
                f"go, and it is the one A\\* computes the route on."
            )
        elif layer == "Concentration":
            st.markdown(
                '<div class="legend">'
                + "".join(f'<span><i style="background:{c}"></i>{n}</span>' for n, c in SIC_LEGEND)
                + "</div>",
                unsafe_allow_html=True,
            )
            st.caption(
                "Fraction of the sea covered by ice, in tenths. It is **the other half** "
                "of the POLARIS decision: `RIO = C × RIV(type) + (10−C) × RIV(open "
                "water)`. The same ice type at 3/10 and at 8/10 gives two opposite "
                "verdicts."
            )
        elif layer == "Reference chart":
            st.caption(
                "The ice chart drawn by the analyst, shown for comparison only. "
                "It plays no part in computing the route."
            )

    with right:
        st.markdown("#### Route")
        if entry.get("route_exists"):
            cmp_ = entry["comparison"]
            st.metric("Distance", f"{cmp_['iceroute']['distance_km']:.0f} km")
            st.metric(
                "Waypoints",
                entry.get("n_waypoints", "—"),
                help="The only points sent to the ground: the line on the map "
                "is the full path, the dots are what gets transmitted.",
            )

            # The direct route's verdict, in words, next to the map that just
            # drew it. Three outcomes, and they are not interchangeable.
            base = cmp_["baseline"]
            st.markdown("#### Direct route")
            if base.get("blocked_by_ice"):
                st.error(
                    f"Forbidden: {base['n_no_go_cells']} cells beyond the POLARIS "
                    f"limit for a {ice_class} vessel."
                )
            elif base.get("feasible"):
                st.success(
                    f"Navigable in {fmt(base['time_hours'])} h. "
                    f"IceRoute deviates by {cmp_['detour_km']:+.1f} km."
                )
            else:
                st.info(
                    "Outside the chart coverage: it is not the ice that stops it, "
                    "we simply have no map there."
                )

            gj = DEMO_DIR / "routes" / entry.get("geojson", "")
            if gj.exists():
                st.download_button(
                    "Download GeoJSON",
                    gj.read_bytes(),
                    file_name=gj.name,
                    mime="application/geo+json",
                )
        else:
            st.error(entry.get("message", "No route found.")[:200])

        pol = entry.get("polaris", {})
        if pol:
            st.markdown("#### POLARIS on the scene")
            st.caption(
                f"forbidden cells: {pol.get('n_no_go', '—')} · "
                f"minimum RIO: {fmt(pol.get('rio_min'), '{:+.1f}')}"
            )

    st.write("")
    tab_down, tab_time, tab_live = st.tabs(
        ["What reaches the ground", "Time and fuel", "Compute now"]
    )
    with tab_down:
        downlink_panel(scene, entry)
    with tab_time:
        assumptions_panel(entry, ice_class, index)
    with tab_live:
        live_panel(scene, ice_class)
        onboard_panel()

    # -- footer -------------------------------------------------------------
    render_footer(metrics, index, assumptions)


# ---------------------------------------------------------------------------
# What the satellite sends
# ---------------------------------------------------------------------------


def conditions_note(entry: dict, ice_class: str) -> None:
    """Answer "but your route goes through the thick ice" before it is asked.

    On the class map the route crosses a great deal of white, the legend calls
    the white thick ice, and the obvious conclusion is that the route is
    illegal. It is not, and the reason is not visible on this layer: POLARIS
    scores type *times* concentration, and the same ice at 3/10 and at 8/10
    gives opposite verdicts. The numbers below are measured along the route in
    precompute, not restated here, so the rebuttal is evidence rather than
    reassurance.
    """
    c = entry.get("conditions") or {}
    if not c.get("worst_class_cells"):
        return

    pct = 100 * c["worst_class_cells"] / c["n_cells"]
    normal = 100 * (c.get("fraction_normal") or 0.0)
    st.info(
        f"**The route crosses the white, and it is legal.** {pct:.0f}% of its cells "
        f"are classified “{c['worst_class'].lower()}” — but at a median concentration "
        f"of **{c['worst_class_sic_median']:.0f}/10**, which gives "
        f"RIO **{fmt(c.get('worst_class_rio_median'), '{:+.0f}')}**: normal operation.\n\n"
        f"POLARIS does not decide on the ice type but on "
        f"`RIO = C × RIV(type) + (10−C) × RIV(open water)`. This layer shows only "
        f"the type: open **Concentration** for the other half, or **RIO** for the "
        f"verdict.\n\n"
        f"Over the whole route: **{normal:.0f}% of cells in normal operation**, "
        f"minimum RIO {fmt(c.get('rio_min'), '{:+.0f}')} for a {ice_class} vessel."
    )


def downlink_panel(scene: dict, entry: dict) -> None:
    """The product, in bytes. The strongest measured claim we have."""
    from iceroute.demo.estimate import link_time, payload

    if not entry.get("route_exists"):
        st.info("No route on this scene: there is no payload to transmit.")
        return

    p = payload(entry, DEMO_DIR / "routes", load_json(str(METRICS)), scene["scene_id"])
    if p is None:
        st.warning("Route file missing: run `make precompute`.")
        return

    rows = [
        ("Ice mask (PNG)", p.mask_bytes),
        (f"Route ({p.n_waypoints} waypoints, GeoJSON)", p.route_bytes),
        ("Metadata", p.metadata_bytes),
    ]
    body = "".join(
        f'<tr><td style="padding:3px 14px 3px 0;color:#9fb1c7">{n}</td>'
        f'<td style="text-align:right;font-family:ui-monospace">'
        f"{'not measured' if v is None else f'{v:,} B'.replace(',', '.')}</td></tr>"
        for n, v in rows
    )
    left, right = st.columns([1, 1], gap="medium")
    with left:
        st.markdown(f"<table>{body}</table>", unsafe_allow_html=True)
        if p.total_bytes and p.sar_bytes:
            # "on this scene", because the card at the top of the page shows
            # the median across scenes and the two numbers differ.
            st.markdown(
                kpi(
                    "Total transmitted, on this scene",
                    f"{p.total_bytes / 1024:.0f} KB",
                    f"against {p.sar_bytes / 1024**2:.1f} MB of losslessly compressed SAR "
                    f"— <b>{p.ratio:.0f}:1</b>. The card at the top reports the "
                    f"<b>median</b> over the measured scenes.",
                ),
                unsafe_allow_html=True,
            )
        lt = link_time(p.total_bytes, p.route_bytes, load_assumptions())
        if lt.seconds is not None:
            sbd = (
                f" The route alone fits in <b>{lt.sbd_messages} Iridium SBD messages</b>."
                if lt.sbd_messages
                else ""
            )
            st.markdown(
                kpi(
                    "Time on the link",
                    f"≈ {lt.seconds:.0f} s",
                    f"on {lt.terminal} at {lt.kbps} kbps, the slowest terminal among those "
                    f"in configs/assumptions.yaml.{sbd} Derived from the operator's rated "
                    f"peak speed: real throughput is lower.",
                ),
                unsafe_allow_html=True,
            )
        if p.note:
            st.caption(p.note)
        st.caption(
            "The comparison route is **not** in here: it is computed on the ground "
            "to make the comparison, and a satellite has no reason to transmit the "
            "straight line between two points the recipient gave it in the first place."
        )
    with right:
        st.caption(f"The file that reaches the ship, in full — {p.route_bytes} bytes:")
        st.code(p.route_text, language="json")


# ---------------------------------------------------------------------------
# Time and fuel: the viewer's assumption, not our measurement
# ---------------------------------------------------------------------------


def assumptions_panel(entry: dict, ice_class: str, index: dict) -> None:
    from iceroute.demo.estimate import fuel_tonnes, rescale_transit

    if not entry.get("route_exists"):
        st.info("No route: nothing to estimate.")
        return

    ours = entry["comparison"]["iceroute"]
    nominal = float(index.get("open_water_speed_kn") or 12.0)

    st.warning(
        "**These two values are not our measurements.** The open-water speed is "
        "the only unsourced number in `configs/routing.yaml` (P-05) and fuel "
        "consumption is not in the project (P-09). Move them: what changes below is "
        "how much the assumption weighs, not how good IceRoute is."
    )
    c1, c2 = st.columns(2)
    speed = c1.slider("Open-water speed (knots)", 8.0, 20.0, nominal, 0.5)
    burn = c2.slider(
        "Consumption at full speed (t/day)",
        0.0,
        120.0,
        0.0,
        5.0,
        help="0 = not set: without a consumption figure no fuel is shown.",
    )

    t = rescale_transit(
        ours.get("time_hours"),
        ours.get("min_rio"),
        nominal_kn=nominal,
        chosen_kn=speed,
        ice_class=ice_class,
    )
    base = entry["comparison"]["baseline"]

    cols = st.columns(3)
    if t.hours is None:
        cols[0].markdown(
            kpi("IceRoute transit", "not rescalable", t.reason[:150]), unsafe_allow_html=True
        )
    else:
        cols[0].markdown(
            kpi(
                "IceRoute transit",
                f"{t.hours:.1f} h",
                f"at {speed:.1f} kn · {ours['distance_km']:.0f} km",
            ),
            unsafe_allow_html=True,
        )

    if base.get("blocked_by_ice"):
        cols[1].markdown(
            kpi(
                "Direct route",
                "forbidden",
                f"{base['n_no_go_cells']} cells beyond the POLARIS limit for a "
                f"{ice_class} vessel. <b>No reference time</b>: there is no slower "
                f"voyage to save time against.",
            ),
            unsafe_allow_html=True,
        )
    elif base.get("feasible"):
        bt = rescale_transit(
            base.get("time_hours"),
            ours.get("min_rio"),
            nominal_kn=nominal,
            chosen_kn=speed,
            ice_class=ice_class,
        )
        delta = (bt.hours - t.hours) if (bt.hours and t.hours) else None
        cols[1].markdown(
            kpi(
                "Direct route",
                f"{fmt(bt.hours)} h",
                "navigable: IceRoute follows it"
                + (f", {delta:+.1f} h" if delta is not None else ""),
            ),
            unsafe_allow_html=True,
        )
    else:
        cols[1].markdown(
            kpi("Direct route", "outside coverage", "it is not the ice that stops it"),
            unsafe_allow_html=True,
        )

    f = fuel_tonnes(t.hours, burn)
    cols[2].markdown(
        kpi(
            "Fuel",
            "not set" if f is None else f"{f:.1f} t",
            "needs a consumption figure, and we have none with a source (P-09)"
            if f is None
            else f'<span class="tag warn">your assumption</span> {burn:.0f} t/day × '
            f"{t.hours / 24:.2f} days",
        ),
        unsafe_allow_html=True,
    )

    from iceroute.demo.estimate import ELEVATED_CAP_KN

    cap = ELEVATED_CAP_KN.get(ice_class)
    st.caption(
        "The rescaling is **exact**, not approximate, and only applies when the route "
        "stays entirely in normal operation (RIO ≥ 0): there every leg runs at full "
        "speed and the time is proportional to 1/v. If the route enters elevated risk, "
        + (
            f"POLARIS sets a {cap:.0f} kn limit (Table 1.2) that this slider does not move"
            if cap
            else "POLARIS already makes that band impassable for this class"
        )
        + ", and the number is withheld instead of being invented."
    )


# ---------------------------------------------------------------------------
# Run the whole chain, now
# ---------------------------------------------------------------------------


@st.cache_resource(show_spinner=False)
def _model(train_config: str):
    from iceroute.demo.live import load_model

    return load_model(Path(train_config))


def live_panel(scene: dict, ice_class: str) -> None:
    """The answer to "how do I know these aren't pictures you made earlier"."""
    import tempfile

    from iceroute.demo import live
    from iceroute.demo import overlay as ov

    raw = Path("data/raw/test") / scene["scene_id"]
    cfg = Path("artifacts/best_config.txt")
    picked = cfg.read_text().strip() if cfg.exists() else "configs/train3.yaml"

    st.caption(
        "Re-reads the raw NetCDF and redoes everything: inference, geocoding, POLARIS, "
        "A\\*. It is not a second program — it is **the same function** that generated "
        "what you are looking at. About 15 s on a GPU. If it fails, the good result "
        "stays above and the demo continues."
    )
    if not raw.exists():
        st.info(
            f"Raw scene not present in `{raw.parent}`: the live computation "
            f"needs the original NetCDF."
        )
        return

    if not st.button("Compute now from the raw scene", type="primary"):
        return

    slot, out = st.empty(), Path(tempfile.mkdtemp(prefix="iceroute_live_"))
    with st.spinner("…"):
        model, device = _model(picked)
        run = live.compute(
            raw,
            model=model,
            device=device,
            out_dir=out,
            train_config=Path(picked),
            on_stage=lambda st_: slot.markdown(_stages_md(st_)),
        )
    slot.markdown(_stages_md(run.stages))

    if not run.ok:
        st.error(f"Live computation failed: {run.error}")
        return

    st.success(
        f"**{run.total_seconds:.1f} s** on {run.device.upper()}, from raw scene to waypoints."
    )

    stem = scene["scene_id"].replace(".nc", "")
    verdict = live.agreement(
        ov.load(out / "overlays" / f"{stem}__{ice_class}.json"),
        ov.load(DEMO_DIR / "overlays" / f"{stem}__{ice_class}.json"),
    )
    (st.success if verdict.get("identical") else st.warning)(
        f"Comparison with the precomputed result: {verdict['note']}"
    )

    png = out / "images" / f"{stem}_maskgeo.png"
    live_ov = out / "overlays" / f"{stem}__{ice_class}.json"
    if png.exists() and live_ov.exists():
        img, problem = composite(str(png), str(live_ov), "both")
        if img:
            st.image(
                img, caption="Computed just now, not loaded from disk.", use_container_width=True
            )
        else:
            st.warning(problem)


def onboard_panel() -> None:
    """The other half of "does it run": the quantised model, on one CPU thread."""
    st.markdown("---")
    st.caption(
        "The computation above runs on this laptop's GPU with the fp32 model: "
        "it proves the **chain** is real. The satellite, however, would carry the INT8 "
        "model on a CPU, and that is a different machine — measuring it separately "
        "avoids passing the laptop GPU off as on-board hardware."
    )
    if st.button("Time the INT8 model on 1 CPU thread"):
        run_live_inference()


def _stages_md(stages) -> str:
    rows = []
    for s in stages:
        mark = "✓" if s.done else "▸"
        secs = f"`{s.seconds:5.1f} s`" if s.done else "…"
        rows.append(f"{mark} {s.name} &nbsp; {secs}")
    return "  \n".join(rows)


def run_live_inference() -> None:
    """Time the INT8 model on one CPU thread.

    Deliberately kept alongside the full live run, because they measure
    different machines. "Compute now" uses the fp32 network on this laptop's
    GPU: it proves the *chain* is real. This measures the quantised model on a
    single CPU thread, which is the thing that would actually fly. Letting the
    GPU run stand in for the onboard claim would be showing hardware the
    satellite does not have.

    Wrapped so that a failure is a message in a box, never a traceback across
    the screen.
    """
    try:
        import numpy as np
        import onnxruntime as ort

        # The deployed config, not a fixed filename. Reading configs/train3.yaml
        # here meant timing whatever model that file pointed at, which stopped
        # being the deployed one the moment the picker changed its mind.
        picked = Path("artifacts/best_config.txt")
        name_cfg = picked.read_text().strip() if picked.exists() else "configs/train3.yaml"
        cfg = yaml.safe_load(Path(name_cfg).read_text())
        path = Path(cfg["output"]["onnx_int8"])
        if not path.exists():
            st.warning(f"INT8 model missing: {path}. Run `make quantize`.")
            return

        sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        name = sess.get_inputs()[0].name
        n_ch = int(cfg["model"]["in_channels"])
        x = np.random.default_rng(0).standard_normal((1, n_ch, 256, 256)).astype("float32")

        sess.run(None, {name: x})  # warm-up, excluded from the timing
        t0 = time.perf_counter()
        sess.run(None, {name: x})
        ms = (time.perf_counter() - t0) * 1000

        st.success(
            f"One 256×256 tile in **{ms:.0f} ms** · model "
            f"{path.stat().st_size / 1024**2:.2f} MB, INT8, 1 CPU thread "
            f"({name_cfg})"
        )
    except Exception as exc:  # noqa: BLE001 — must never break the demo
        st.warning(f"Live inference failed: {exc}")


def render_footer(metrics: dict, index: dict, assumptions: dict) -> None:
    ev = metrics.get("evaluation") or {}
    agg = (ev.get("scenes") or {}).get("aggregate") or {}
    miou = agg.get("miou")
    binacc = agg.get("binary_accuracy")
    n_scenes = (ev.get("scenes") or {}).get("n_scenes")

    bits = [f"scheme: {index.get('scheme', '—')}"]
    if isinstance(miou, float):
        bits.append(f"mIoU {miou:.3f}")
    if isinstance(binacc, float):
        bits.append(f"ice/water {binacc:.3f}")
    if n_scenes:
        bits.append(f"{n_scenes} official test scenes")

    st.markdown(f'<div class="foot">{" · ".join(bits)}</div>', unsafe_allow_html=True)

    with st.expander("Assumptions"):
        st.caption(
            "Every number shown derives from here. Entries without a source are marked "
            "provisional and do not appear in any economic claim."
        )
        econ = assumptions.get("economics", {})
        if not econ.get("show_dollar_savings", False):
            st.info("**No dollar savings.** " + econ.get("reason", "").strip())
        if ASSUMPTIONS.exists():
            st.code(ASSUMPTIONS.read_text(), language="yaml")


if __name__ == "__main__":
    main()
