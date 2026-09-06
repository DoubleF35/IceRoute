"""The demo. SPEC §9.

Not a tool — a 90-second visual argument. Everything that does not serve the
argument has been left out.

Two rules shaped this file:

* **It reads only precomputed files.** No inference, no geocoding, no routing at
  runtime in the main path. The demo has to start in under two seconds and must
  not depend on PyTorch initialising on a stage. The one exception is the "Run
  inference now" button, which proves this is not a video and which fails
  softly: if it breaks, the rest of the demo carries on.
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
        colour = "#4ade9b" if saved > 0 else "#f0b849"
        sub = f"contro {fmt(cmp_['baseline']['time_hours'])} h della rotta diretta"
        return kpi("Tempo di transito", f"{fmt(ours)} h",
                   f'<span style="color:{colour}">{saved:+.1f} h ({pct:+.0f}%)</span> — {sub}'
                   + (f"<br>{note}" if note else ""))

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
        layer = st.radio(
            "Livello", ["SAR", "Maschera del ghiaccio", "RIO", "Carta di riferimento"],
            label_visibility="collapsed",
        )
        show_route = st.checkbox("Mostra la rotta", value=True)

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
        key = {"SAR": "sar", "Maschera del ghiaccio": "mask_pred",
               "RIO": f"rio_{ice_class}", "Carta di riferimento": "mask_truth"}[layer]
        img = DEMO_DIR / "images" / scene["files"].get(key, "")
        if img.exists():
            st.image(str(img), use_container_width=True)
        else:
            st.warning(f"Immagine assente: {key}")

        if layer == "Maschera del ghiaccio":
            st.markdown(
                '<div class="legend">'
                + "".join(f'<span><i style="background:{c}"></i>{n}</span>'
                          for n, c in CLASS_COLOURS.items())
                + "</div>", unsafe_allow_html=True)
            st.caption(
                "Maschera **predetta dal modello**, non la carta dell'analista. "
                "È su questa che è calcolata la rotta."
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
            c = entry["comparison"]["iceroute"]
            st.metric("Distanza", f"{c['distance_km']:.0f} km")
            st.metric("Waypoint", entry.get("n_waypoints", "—"))
            if show_route:
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

        with st.expander("Prova che gira davvero"):
            st.caption(
                "Esegue il modello INT8 su una patch, adesso. Se fallisce, il resto "
                "della demo continua a funzionare."
            )
            if st.button("Esegui l'inferenza ora"):
                run_live_inference()

    # -- footer -------------------------------------------------------------
    render_footer(metrics, index, assumptions)


def run_live_inference() -> None:
    """The controlled exception to 'no runtime inference'. SPEC §9.2.

    Wrapped so that a failure is a message in a box, never a traceback across
    the projector.
    """
    try:
        import numpy as np
        import onnxruntime as ort

        cfg = yaml.safe_load(Path("configs/train3.yaml").read_text())
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
                   f"{path.stat().st_size / 1024**2:.2f} MB, INT8, CPU")
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
