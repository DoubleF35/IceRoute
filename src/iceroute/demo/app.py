"""The demo. SPEC §9.

Not a tool — a 90-second visual argument. Everything that does not serve the
argument has been left out.

Two rules shaped this file:

* **The main path reads only precomputed files.** No inference, no geocoding,
  no routing while the audience waits. The demo has to start in under two
  seconds and must not depend on PyTorch initialising on a stage.

  There are exactly two deliberate exceptions, both behind a button, both of
  which fail into a message box rather than a traceback. **Calcola ora** re-runs
  the entire chain on the raw NetCDF — the same ``process_scene`` the product
  runs, not a copy — and reports whether the result matches the precomputed one.
  The second times the INT8 model on a single CPU thread, kept separate because
  the GPU in this laptop is not the hardware that would fly.

  Neither weakens the first rule: if either breaks, the precomputed answer is
  still on screen and the talk continues.
* **It never invents a number.** Anything not measured reads as "non misurato",
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

# Dark, high contrast: conference rooms are dark and a projector washes out mid
# tones (SPEC §9.4). No pastels anywhere.
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

CLASS_COLOURS = {"Acqua libera": "#0a2859", "Ghiaccio sottile": "#00c8ff",
                 "Ghiaccio spesso": "#ffffff"}
RIO_LEVELS = [("Navigazione normale", "#2ecc71"), ("Rischio elevato", "#f1c40f"),
              ("Considerazione speciale", "#e74c3c")]

# Which raster each layer shows, and whether a route can be drawn on it.
#
# Only the geocoded rasters are routable. The SAR and the analyst's chart are
# the raw swath: the route cells are indexed in the 500 m EPSG:3413 grid and
# mean nothing there. Rather than quietly not drawing — the failure this whole
# feature exists to fix — the UI says why the tracks are unavailable.
LAYERS = {
    "SAR": ("sar", False),
    "Maschera del ghiaccio": ("mask_geo", True),
    "RIO": ("rio_{ice_class}", True),
    "Carta di riferimento": ("mask_truth", False),
}
ROUTE_MODES = {"Rotta diretta": "baseline", "IceRoute": "iceroute",
               "Entrambe": "both", "Nessuna": "none"}
ROUTE_LEGEND = {
    "baseline": [("Rotta diretta (carta nautica)", "#f5f5f5"),
                 ("Cella vietata da POLARIS", "#ff2020")],
    "iceroute": [("IceRoute", "#ff2d95"), ("Waypoint trasmessi a terra", "#ff2d95")],
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


def fmt(v, spec="{:.1f}", missing="non misurato"):
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
    return (f'<div class="kpi"><div class="lab">{label}</div>'
            f'<div class="val">{value}</div><div class="sub">{sub}</div></div>')


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
        return None, f"Immagine assente: {p.name}"

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
# KPI cards — SPEC §9.3
# ---------------------------------------------------------------------------


def transit_card(entry: dict, provisional_speed: bool) -> str:
    """Transit time, and the delta only when there is an honest one."""
    if not entry.get("route_exists"):
        return kpi("Tempo di transito",
                   '<span style="color:#f08a8a">nessuna rotta</span>',
                   entry.get("message", "")[:90])

    cmp_ = entry["comparison"]
    ours = cmp_["iceroute"]["time_hours"]
    note = "assume 12 kn in acqua libera (senza fonte)" if provisional_speed else ""

    if cmp_.get("time_saved_hours") is not None:
        saved = cmp_["time_saved_hours"]
        pct = cmp_.get("time_saved_percent")
        base_h = fmt(cmp_["baseline"]["time_hours"])
        if saved > 0:
            body = (f'<span style="color:#4ade9b">{saved:+.1f} h ({pct:+.0f}%)</span> '
                    f"contro {base_h} h della rotta diretta")
        else:
            # The direct route is legal and IceRoute all but follows it. There
            # is nothing to win here, and dressing a 1% loss in warning colours
            # would be editorialising a non-event. The honest reading is the
            # better one anyway: no detour was invented where none was needed.
            body = ('<span class="tag info">la rotta diretta è percorribile</span> '
                    f"IceRoute la segue: {cmp_['detour_km']:+.1f} km e "
                    f"{abs(saved):.1f} h in più su {base_h} h")
        return kpi("Tempo di transito", f"{fmt(ours)} h",
                   body + (f"<br>{note}" if note else ""))

    # No comparable baseline. Say which of the two reasons applies — they mean
    # completely different things and conflating them would be a false claim.
    base = cmp_["baseline"]
    if base.get("blocked_by_ice"):
        sub = ('<span class="tag ok">la rotta diretta è vietata dal ghiaccio</span> '
               f'{base["n_no_go_cells"]} celle oltre il limite POLARIS')
    else:
        sub = ('<span class="tag info">confronto non disponibile</span> '
               "la rotta diretta esce dalla copertura della carta: non è il ghiaccio "
               "a fermarla")
    return kpi("Tempo di transito", f"{fmt(ours)} h", sub + (f"<br>{note}" if note else ""))


def rio_card(entry: dict) -> str:
    if not entry.get("route_exists"):
        return kpi("RIO minimo sulla rotta", "—", "nessuna rotta da valutare")
    rio = entry["comparison"]["iceroute"].get("min_rio")
    if rio is None:
        return kpi("RIO minimo sulla rotta", "non misurato", "")
    if rio >= 0:
        tag, txt = "ok", "navigazione normale"
    elif rio >= -10:
        tag, txt = "warn", "rischio elevato — limite di velocità POLARIS"
    else:
        tag, txt = "bad", "considerazione speciale"
    return kpi("RIO minimo sulla rotta", f"{rio:+.1f}",
               f'<span class="tag {tag}">{txt}</span> soglie IMO MSC.1/Circ.1519')


def downlink_card(metrics: dict) -> str:
    comp = metrics.get("compression") or {}
    ratio = comp.get("ratio_median")
    if not ratio:
        return kpi("Volume di downlink", "non misurato",
                   "serve <code>make report</code>")
    prov = comp.get("provisional")
    sub = (f"{comp.get('n_scenes', 0)} scene · numeratore: SAR compresso senza perdita")
    if prov:
        sub = '<span class="tag warn">limite superiore</span> ' + sub
    return kpi("Volume di downlink", f"{ratio:.0f}:1", sub)


def onboard_card(metrics: dict) -> str:
    q = metrics.get("quantization") or {}
    size = (q.get("size") or {}).get("int8_mb")
    lat = ((q.get("latency") or {}).get("single_thread") or {}).get("int8", {})
    ms = lat.get("median_ms_per_tile")
    if not size:
        return kpi("Inferenza a bordo", "non misurata", "serve <code>make quantize</code>")
    acc = q.get("accuracy") or {}
    drop = acc.get("miou_drop")
    extra = (f" · INT8 costa {abs(drop):.3f} di mIoU" if isinstance(drop, float) and drop > 0
             else " · INT8 non degrada in modo misurabile")
    return kpi("Inferenza a bordo", f"{size:.2f} MB",
               f"{fmt(ms, '{:.0f}')} ms per tile, 1 thread CPU{extra}")


# ---------------------------------------------------------------------------
# Page
# ---------------------------------------------------------------------------


def main() -> None:
    st.set_page_config(page_title="IceRoute", layout="wide",
                       initial_sidebar_state="expanded")
    st.markdown(CSS, unsafe_allow_html=True)

    index = load_index()
    if not index or not index.get("scenes"):
        st.title("IceRoute")
        st.error(
            "Nessun dato precomputato in `data/demo/`. Lancia **`make precompute`**.\n\n"
            "La UI legge solo file già generati: non calcola niente dal vivo, "
            "così parte in meno di due secondi e non dipende da PyTorch sul palco."
        )
        return

    scenes = [s for s in index["scenes"] if s.get("classes")]
    # The worst case goes last, always. Left in the normal rotation it would
    # read as a representative scene, which is the opposite of why it is here.
    scenes.sort(key=lambda s: bool(s.get("worst_case")))
    if not scenes:
        st.title("IceRoute")
        st.error("Nessuna scena utilizzabile: tutte prive di rotte.")
        return

    metrics = load_json(str(METRICS))
    assumptions = load_assumptions()
    prov_speed = bool(index.get("open_water_speed_is_provisional"))

    # -- sidebar ------------------------------------------------------------
    with st.sidebar:
        st.markdown("### Scena")
        labels = {
            f"{'⚠ ' if s.get('worst_case') else ''}{s['scene_id'][:8]}"
            f" · {s.get('region', '?')[:22]}": s
            for s in scenes
        }
        chosen = st.radio("Scena", list(labels), label_visibility="collapsed")
        scene = labels[chosen]

        st.markdown("### Classe della nave")
        available = list(scene["classes"])
        # If the route never changes with the class, the selector is theatre.
        # Say so rather than letting someone discover it on stage.
        if not scene.get("routes_differ") and len(available) > 1:
            st.caption(
                "Su questa scena la rotta non cambia con la classe: il corridoio "
                "di acqua libera è largo abbastanza per tutte."
            )
        ice_class = st.radio("Classe", available, label_visibility="collapsed")

        st.markdown("### Livelli")
        layer = st.radio("Livello", list(LAYERS), index=1, label_visibility="collapsed")
        key_tpl, routable = LAYERS[layer]

        st.markdown("### Rotte sulla mappa")
        if routable:
            mode = ROUTE_MODES[
                st.radio("Rotte", list(ROUTE_MODES), index=2, label_visibility="collapsed")
            ]
        else:
            mode = "none"
            st.caption(
                "Su questo livello le rotte non sono disegnabili: è il raster della "
                "scena grezza, mentre la rotta è calcolata sulla griglia geocodificata "
                "a 500 m. Passa a **Maschera del ghiaccio** o **RIO**."
            )

    entry = scene["classes"][ice_class]

    # -- header + KPI -------------------------------------------------------
    st.markdown(
        f"# IceRoute <span style='font-size:16px;color:#6b7f99'>"
        f"{scene['scene_id']} · {scene.get('acquired', '')[:10]} · "
        f"copertura carta {100 * scene.get('coverage', 0):.0f}%</span>",
        unsafe_allow_html=True,
    )

    if scene.get("worst_case"):
        miou, unsafe = scene_scores(metrics, scene["scene_id"])
        agg = (((metrics.get("evaluation") or {}).get("scenes") or {})
               .get("aggregate") or {}).get("miou")
        unsafe_pct = 100 * unsafe if isinstance(unsafe, int | float) else None
        st.warning(
            "**Questa è la scena su cui il modello sbaglia di più**, fra le 20 del "
            "test ufficiale. È qui apposta: serve a rispondere a «e quando "
            "sbaglia?» invece di far finta che non succeda.\n\n"
            f"mIoU **{fmt(miou, '{:.2f}')}** contro **{fmt(agg, '{:.2f}')}** di media. "
            f"Frazione permissiva **{fmt(unsafe_pct, '{:.0f}%')}**: "
            "su questa quota di pixel il modello dichiara percorribile del ghiaccio "
            "che POLARIS vieta a quella classe di nave. La rotta qui sotto **non è "
            "affidabile** ed è mostrata per questo."
        )

    cols = st.columns(4, gap="small")
    for col, html_ in zip(
        cols,
        [transit_card(entry, prov_speed), rio_card(entry),
         downlink_card(metrics), onboard_card(metrics)],
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
        # renders, which is the point — "nessuna rotta" is a real outcome.
        name = entry.get("overlay") or ""
        ov = str(DEMO_DIR / overlay_mod.OVERLAY_DIR / name) if name and mode != "none" else ""
        png, problem = composite(str(img), ov, mode)
        if png:
            st.image(png, use_container_width=True)
        else:
            st.warning(problem or f"Immagine assente: {key}")

        if mode != "none":
            shown = [x for m, x in ROUTE_LEGEND.items() if mode in (m, "both")]
            st.markdown(
                '<div class="legend">'
                + "".join(f'<span><i style="background:{c}"></i>{n}</span>'
                          for group in shown for n, c in group)
                + "</div>", unsafe_allow_html=True)

        if layer == "Maschera del ghiaccio":
            st.markdown(
                '<div class="legend">'
                + "".join(f'<span><i style="background:{c}"></i>{n}</span>'
                          for n, c in CLASS_COLOURS.items())
                + "</div>", unsafe_allow_html=True)
            st.caption(
                "Maschera **predetta dal modello**, non la carta dell'analista, "
                "riproiettata in EPSG:3413 a 500 m. È su questa che è calcolata "
                "la rotta, ed è per questo che le rotte si disegnano qui."
            )
        elif layer == "RIO":
            st.markdown(
                '<div class="legend">'
                + "".join(f'<span><i style="background:{c}"></i>{n}</span>'
                          for n, c in RIO_LEVELS)
                + "</div>", unsafe_allow_html=True)
            st.caption(f"Risk Index Outcome per una nave {ice_class}, soglie IMO.")
        elif layer == "Carta di riferimento":
            st.caption(
                "La carta del ghiaccio disegnata dall'analista, mostrata solo per "
                "confronto. Non entra nel calcolo della rotta."
            )

    with right:
        st.markdown("#### Rotta")
        if entry.get("route_exists"):
            cmp_ = entry["comparison"]
            st.metric("Distanza", f"{cmp_['iceroute']['distance_km']:.0f} km")
            st.metric("Waypoint", entry.get("n_waypoints", "—"),
                      help="I soli punti che scendono a terra: la linea sulla mappa "
                           "è il percorso completo, i pallini sono ciò che si trasmette.")

            # The direct route's verdict, in words, next to the map that just
            # drew it. Three outcomes, and they are not interchangeable.
            base = cmp_["baseline"]
            st.markdown("#### Rotta diretta")
            if base.get("blocked_by_ice"):
                st.error(f"Vietata: {base['n_no_go_cells']} celle oltre il limite "
                         f"POLARIS per una nave {ice_class}.")
            elif base.get("feasible"):
                st.success(f"Percorribile in {fmt(base['time_hours'])} h. "
                           f"IceRoute devia di {cmp_['detour_km']:+.1f} km.")
            else:
                st.info("Fuori dalla copertura della carta: non è il ghiaccio a "
                        "fermarla, lì non abbiamo una mappa.")

            gj = DEMO_DIR / "routes" / entry.get("geojson", "")
            if gj.exists():
                st.download_button("Scarica GeoJSON", gj.read_bytes(),
                                   file_name=gj.name, mime="application/geo+json")
        else:
            st.error(entry.get("message", "Nessuna rotta trovata.")[:200])

        pol = entry.get("polaris", {})
        if pol:
            st.markdown("#### POLARIS sulla scena")
            st.caption(
                f"celle vietate: {pol.get('n_no_go', '—')} · "
                f"RIO minimo: {fmt(pol.get('rio_min'), '{:+.1f}')}"
            )

    st.write("")
    tab_down, tab_time, tab_live = st.tabs(
        ["Cosa scende a terra", "Tempo e carburante", "Calcola ora"]
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


def downlink_panel(scene: dict, entry: dict) -> None:
    """The product, in bytes. The strongest measured claim we have."""
    from iceroute.demo.estimate import payload

    if not entry.get("route_exists"):
        st.info("Nessuna rotta su questa scena: non c'è payload da trasmettere.")
        return

    p = payload(entry, DEMO_DIR / "routes", load_json(str(METRICS)), scene["scene_id"])
    if p is None:
        st.warning("File della rotta assente: lancia `make precompute`.")
        return

    rows = [("Maschera del ghiaccio (PNG)", p.mask_bytes),
            (f"Rotta ({p.n_waypoints} waypoint, GeoJSON)", p.route_bytes),
            ("Metadati", p.metadata_bytes)]
    body = "".join(
        f'<tr><td style="padding:3px 14px 3px 0;color:#9fb1c7">{n}</td>'
        f'<td style="text-align:right;font-family:ui-monospace">'
        f'{"non misurato" if v is None else f"{v:,} B".replace(",", ".")}</td></tr>'
        for n, v in rows
    )
    left, right = st.columns([1, 1], gap="medium")
    with left:
        st.markdown(f'<table>{body}</table>', unsafe_allow_html=True)
        if p.total_bytes and p.sar_bytes:
            st.markdown(
                kpi("Totale trasmesso", f"{p.total_bytes / 1024:.0f} KB",
                    f"contro {p.sar_bytes / 1024**2:.1f} MB di SAR compresso senza perdita "
                    f"— <b>{p.ratio:.0f}:1</b>"),
                unsafe_allow_html=True)
        if p.note:
            st.caption(p.note)
        st.caption(
            "La rotta di confronto **non** è qui dentro: è calcolata a terra per "
            "argomentare, e un satellite non ha ragione di trasmettere la linea retta "
            "fra due punti che il destinatario gli ha dato lui."
        )
    with right:
        st.caption(f"Il file che arriva alla nave, per intero — {p.route_bytes} byte:")
        st.code(p.route_text, language="json")


# ---------------------------------------------------------------------------
# Time and fuel: the viewer's assumption, not our measurement
# ---------------------------------------------------------------------------


def assumptions_panel(entry: dict, ice_class: str, index: dict) -> None:
    from iceroute.demo.estimate import fuel_tonnes, rescale_transit

    if not entry.get("route_exists"):
        st.info("Nessuna rotta: niente da stimare.")
        return

    ours = entry["comparison"]["iceroute"]
    nominal = float(index.get("open_water_speed_kn") or 12.0)

    st.warning(
        "**Questi due valori non sono nostre misure.** La velocità in acqua libera è "
        "l'unico numero senza fonte in `configs/routing.yaml` (P-05) e il consumo non "
        "è nel progetto (P-09). Muovili: quello che cambia sotto è quanto pesa "
        "l'assunzione, non quanto è brava IceRoute."
    )
    c1, c2 = st.columns(2)
    speed = c1.slider("Velocità in acqua libera (nodi)", 8.0, 20.0, nominal, 0.5)
    burn = c2.slider("Consumo a velocità piena (t/giorno)", 0.0, 120.0, 0.0, 5.0,
                     help="0 = non impostato: senza un consumo non mostriamo carburante.")

    t = rescale_transit(ours.get("time_hours"), ours.get("min_rio"),
                        nominal_kn=nominal, chosen_kn=speed, ice_class=ice_class)
    base = entry["comparison"]["baseline"]

    cols = st.columns(3)
    if t.hours is None:
        cols[0].markdown(kpi("Transito IceRoute", "non riscalabile", t.reason[:150]),
                         unsafe_allow_html=True)
    else:
        cols[0].markdown(
            kpi("Transito IceRoute", f"{t.hours:.1f} h",
                f"a {speed:.1f} kn · {ours['distance_km']:.0f} km"), unsafe_allow_html=True)

    if base.get("blocked_by_ice"):
        cols[1].markdown(
            kpi("Rotta diretta", "vietata",
                f'{base["n_no_go_cells"]} celle oltre il limite POLARIS per una nave '
                f'{ice_class}. <b>Nessun tempo di riferimento</b>: non esiste un viaggio '
                f'più lento da cui risparmiare.'), unsafe_allow_html=True)
    elif base.get("feasible"):
        bt = rescale_transit(base.get("time_hours"), ours.get("min_rio"),
                             nominal_kn=nominal, chosen_kn=speed, ice_class=ice_class)
        delta = (bt.hours - t.hours) if (bt.hours and t.hours) else None
        cols[1].markdown(
            kpi("Rotta diretta", f"{fmt(bt.hours)} h",
                "percorribile: IceRoute la segue"
                + (f", {delta:+.1f} h" if delta is not None else "")),
            unsafe_allow_html=True)
    else:
        cols[1].markdown(kpi("Rotta diretta", "fuori copertura",
                             "non è il ghiaccio a fermarla"), unsafe_allow_html=True)

    f = fuel_tonnes(t.hours, burn)
    cols[2].markdown(
        kpi("Carburante", "non impostato" if f is None else f"{f:.1f} t",
            "serve un consumo, e non ne abbiamo uno con una fonte (P-09)"
            if f is None else
            f'<span class="tag warn">assunzione tua</span> {burn:.0f} t/giorno × '
            f'{t.hours / 24:.2f} giorni'), unsafe_allow_html=True)

    from iceroute.demo.estimate import ELEVATED_CAP_KN

    cap = ELEVATED_CAP_KN.get(ice_class)
    st.caption(
        "Il riscalamento è **esatto**, non approssimato, e solo quando la rotta resta "
        "tutta in operazione normale (RIO ≥ 0): lì ogni tratto va a velocità piena e il "
        "tempo è proporzionale a 1/v. Se la rotta entra in rischio elevato, POLARIS fissa "
        + (f"un limite di {cap:.0f} kn (Tabella 1.2) che questo cursore non muove"
           if cap else "che per questa classe la fascia è già impraticabile")
        + ", e il numero non viene mostrato invece di essere inventato."
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
        "Rilegge il NetCDF grezzo e rifà tutto: inferenza, geocodifica, POLARIS, A\\*. "
        "Non è un secondo programma — è **la stessa funzione** che ha generato quello "
        "che stai guardando. Circa 15 s su GPU. Se fallisce, sopra resta il risultato "
        "buono e la demo continua."
    )
    if not raw.exists():
        st.info(f"Scena grezza non presente in `{raw.parent}`: il calcolo dal vivo "
                f"richiede il NetCDF originale.")
        return

    if not st.button("Calcola ora dalla scena grezza", type="primary"):
        return

    slot, out = st.empty(), Path(tempfile.mkdtemp(prefix="iceroute_live_"))
    with st.spinner("…"):
        model, device = _model(picked)
        run = live.compute(raw, model=model, device=device, out_dir=out,
                           train_config=Path(picked),
                           on_stage=lambda st_: slot.markdown(_stages_md(st_)))
    slot.markdown(_stages_md(run.stages))

    if not run.ok:
        st.error(f"Calcolo dal vivo non riuscito: {run.error}")
        return

    st.success(f"**{run.total_seconds:.1f} s** su {run.device.upper()}, "
               f"dalla scena grezza al waypoint.")

    stem = scene["scene_id"].replace(".nc", "")
    verdict = live.agreement(
        ov.load(out / "overlays" / f"{stem}__{ice_class}.json"),
        ov.load(DEMO_DIR / "overlays" / f"{stem}__{ice_class}.json"),
    )
    (st.success if verdict.get("identical") else st.warning)(
        f"Confronto col precomputato: {verdict['note']}")

    png = out / "images" / f"{stem}_maskgeo.png"
    live_ov = out / "overlays" / f"{stem}__{ice_class}.json"
    if png.exists() and live_ov.exists():
        img, problem = composite(str(png), str(live_ov), "both")
        if img:
            st.image(img, caption="Calcolata adesso, non caricata da disco.",
                     use_container_width=True)
        else:
            st.warning(problem)


def onboard_panel() -> None:
    """The other half of "does it run": the quantised model, on one CPU thread."""
    st.markdown("---")
    st.caption(
        "Il calcolo qui sopra gira sulla GPU di questo portatile con il modello fp32: "
        "dimostra che la **catena** è vera. Il satellite però monterebbe il modello "
        "INT8 su CPU, ed è una macchina diversa — misurarla separatamente evita di far "
        "passare la GPU del portatile per hardware di bordo."
    )
    if st.button("Cronometra il modello INT8 su 1 thread CPU"):
        run_live_inference()


def _stages_md(stages) -> str:
    rows = []
    for s in stages:
        mark = "✓" if s.done else "▸"
        secs = f"`{s.seconds:5.1f} s`" if s.done else "…"
        rows.append(f"{mark} {s.name} &nbsp; {secs}")
    return "  \n".join(rows)


def run_live_inference() -> None:
    """Time the INT8 model on one CPU thread. SPEC §9.2.

    Deliberately kept alongside the full live run, because they measure
    different machines. "Calcola ora" uses the fp32 network on this laptop's
    GPU: it proves the *chain* is real. This measures the quantised model on a
    single CPU thread, which is the thing that would actually fly. Letting the
    GPU run stand in for the onboard claim would be showing hardware the
    satellite does not have.

    Wrapped so that a failure is a message in a box, never a traceback across
    the projector.
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
            st.warning(f"Modello INT8 assente: {path}. Lancia `make quantize`.")
            return

        sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        name = sess.get_inputs()[0].name
        n_ch = int(cfg["model"]["in_channels"])
        x = np.random.default_rng(0).standard_normal((1, n_ch, 256, 256)).astype("float32")

        sess.run(None, {name: x})          # warm-up, excluded from the timing
        t0 = time.perf_counter()
        sess.run(None, {name: x})
        ms = (time.perf_counter() - t0) * 1000

        st.success(f"Una tile 256×256 in **{ms:.0f} ms** · modello "
                   f"{path.stat().st_size / 1024**2:.2f} MB, INT8, 1 thread CPU "
                   f"({name_cfg})")
    except Exception as exc:  # noqa: BLE001 — must never break the demo
        st.warning(f"Inferenza dal vivo non riuscita: {exc}")


def render_footer(metrics: dict, index: dict, assumptions: dict) -> None:
    ev = metrics.get("evaluation") or {}
    agg = (ev.get("scenes") or {}).get("aggregate") or {}
    miou = agg.get("miou")
    binacc = agg.get("binary_accuracy")
    n_scenes = (ev.get("scenes") or {}).get("n_scenes")

    bits = [f"schema: {index.get('scheme', '—')}"]
    if isinstance(miou, float):
        bits.append(f"mIoU {miou:.3f}")
    if isinstance(binacc, float):
        bits.append(f"ghiaccio/acqua {binacc:.3f}")
    if n_scenes:
        bits.append(f"{n_scenes} scene di test ufficiali")

    st.markdown(f'<div class="foot">{" · ".join(bits)}</div>', unsafe_allow_html=True)

    with st.expander("Assunzioni"):
        st.caption(
            "Ogni numero mostrato deriva da qui. Le voci senza fonte sono marcate "
            "provvisorie e non compaiono in affermazioni economiche."
        )
        econ = assumptions.get("economics", {})
        if not econ.get("show_dollar_savings", False):
            st.info(
                "**Nessun risparmio in dollari.** " + econ.get("reason", "").strip()
            )
        if ASSUMPTIONS.exists():
            st.code(ASSUMPTIONS.read_text(), language="yaml")


if __name__ == "__main__":
    main()
