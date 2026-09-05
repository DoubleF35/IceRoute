"""The five figures for the deck. SPEC §10.3.

Each one answers a question an investor actually asks, in order:

1. ``works``        — SAR, predicted mask, chart, side by side
2. ``useful``       — IceRoute route against the baseline, KPI in overlay
3. ``onboard``      — model size and latency, fp32 against INT8
4. ``business``     — raw scene bytes against downlinked payload, log scale
5. ``confusion``    — the confusion matrix, for technical due diligence

300 DPI, transparent background, type sized to be read from the back of a room.

Every figure refuses to draw a number that was not measured. A bar chart with a
plausible height and no measurement behind it is worse than a missing figure,
because a missing figure prompts a question and a wrong bar does not.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np

log = logging.getLogger("figures")

DPI = 300
# Same palette as the demo, so the deck and the live run look like one product.
COL_WATER, COL_THIN, COL_THICK = "#0a2859", "#00c8ff", "#ffffff"
COL_ROUTE, COL_BASE = "#00e08a", "#8899aa"
COL_INK, COL_MUTED = "#eaf1fa", "#8397ae"


def _setup():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.size": 13,
        "axes.labelsize": 14,
        "axes.titlesize": 15,
        "figure.facecolor": "none",
        "axes.facecolor": "none",
        "savefig.facecolor": "none",
        "savefig.transparent": True,
        "text.color": COL_INK,
        "axes.labelcolor": COL_INK,
        "xtick.color": COL_MUTED,
        "ytick.color": COL_MUTED,
        "axes.edgecolor": COL_MUTED,
    })
    return plt


def _save(fig, path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=DPI, bbox_inches="tight", transparent=True)
    import matplotlib.pyplot as plt

    plt.close(fig)
    log.info("  %s", path)
    return path.name


def _missing(plt, path: Path, title: str, why: str) -> str:
    """A placeholder that says what is missing, rather than a plausible chart."""
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.axis("off")
    ax.text(0.5, 0.62, title, ha="center", fontsize=16, color=COL_INK)
    ax.text(0.5, 0.40, why, ha="center", fontsize=12, color="#f0b849", wrap=True)
    return _save(fig, path)


# ---------------------------------------------------------------------------
# 1. It works
# ---------------------------------------------------------------------------


def figure_works(demo_dir: Path, out: Path, scene: dict | None = None) -> str:
    """SAR, prediction and chart side by side, from the precomputed PNGs."""
    plt = _setup()
    if scene is None:
        return _missing(plt, out, "Segmentazione", "Manca data/demo/: lancia make precompute.")

    import matplotlib.image as mpimg

    panels = [("SAR (HV)", "sar"), ("Predizione IceRoute", "mask_pred"),
              ("Carta dell'analista", "mask_truth")]
    fig, axes = plt.subplots(1, 3, figsize=(16, 6))
    for ax, (title, key) in zip(axes, panels, strict=True):
        path = demo_dir / "images" / scene["files"].get(key, "")
        ax.axis("off")
        ax.set_title(title, pad=12)
        if path.exists():
            ax.imshow(mpimg.imread(path))
        else:
            ax.text(0.5, 0.5, "assente", ha="center", color="#f0b849")
    fig.suptitle(
        f"{scene['scene_id'][:15]} · {scene.get('region', '')} · "
        f"inferenza {scene.get('inference_seconds', 0):.1f} s",
        y=0.04, fontsize=12, color=COL_MUTED,
    )
    return _save(fig, out)


# ---------------------------------------------------------------------------
# 2. It is useful
# ---------------------------------------------------------------------------


def figure_useful(demo_dir: Path, out: Path, scene: dict | None, ice_class: str) -> str:
    """The route over the mask, with the KPI that survives scrutiny."""
    plt = _setup()
    entry = (scene or {}).get("classes", {}).get(ice_class) if scene else None
    if not entry or not entry.get("route_exists"):
        return _missing(plt, out, "Rotta", "Nessuna rotta precomputata per questa classe.")

    import matplotlib.image as mpimg

    fig, ax = plt.subplots(figsize=(10, 9))
    ax.axis("off")
    img_path = demo_dir / "images" / scene["files"].get("mask_pred", "")
    if img_path.exists():
        img = mpimg.imread(img_path)
        ax.imshow(img)
        h, w = img.shape[:2]
        gj = demo_dir / "routes" / entry.get("geojson", "")
        if gj.exists():
            _draw_routes(ax, json.loads(gj.read_text()), scene, w, h)

    cmp_ = entry["comparison"]
    saved = cmp_.get("time_saved_hours")
    if saved is not None:
        headline = f"{cmp_['iceroute']['time_hours']:.1f} h · {saved:+.1f} h sulla diretta"
    elif cmp_["baseline"].get("blocked_by_ice"):
        headline = (f"{cmp_['iceroute']['time_hours']:.1f} h · la rotta diretta e' "
                    f"vietata dal ghiaccio ({cmp_['baseline']['n_no_go_cells']} celle)")
    else:
        headline = f"{cmp_['iceroute']['time_hours']:.1f} h · confronto non disponibile"

    ax.set_title(f"{ice_class} — {headline}", pad=14)
    ax.text(
        0.5, -0.04,
        f"RIO minimo sulla rotta {cmp_['iceroute'].get('min_rio', float('nan')):+.1f} · "
        f"{entry.get('n_waypoints', '?')} waypoint · soglie IMO MSC.1/Circ.1519",
        transform=ax.transAxes, ha="center", fontsize=11, color=COL_MUTED,
    )
    return _save(fig, out)


def _draw_routes(ax, collection: dict, scene: dict, w: int, h: int) -> None:
    """Overlay the GeoJSON tracks, mapped from lon/lat back to image pixels."""
    rows, cols = scene["shape"]
    for feat in collection.get("features", []):
        coords = feat.get("geometry", {}).get("coordinates") or []
        if len(coords) < 2:
            continue
        props = feat.get("properties", {})
        # The GeoJSON carries lon/lat; the PNG is the raw scene grid. Waypoint
        # row/col are the reliable link, so use them where present.
        cells = props.get("cells")
        if not cells:
            continue
        y = [c[0] * h / rows for c in cells]
        x = [c[1] * w / cols for c in cells]
        if props.get("kind") == "baseline":
            ax.plot(x, y, "--", color=COL_BASE, lw=2.2, label="rotta diretta")
        else:
            ax.plot(x, y, "-", color=COL_ROUTE, lw=3.0, label="IceRoute")
    if ax.get_legend_handles_labels()[0]:
        ax.legend(loc="lower right", framealpha=0.3, facecolor="#111823")


# ---------------------------------------------------------------------------
# 3. It fits onboard
# ---------------------------------------------------------------------------


def figure_onboard(metrics: dict, out: Path) -> str:
    """Size and latency, fp32 against INT8. Both measured or neither drawn."""
    plt = _setup()
    q = metrics.get("quantization") or {}
    size, lat = q.get("size"), (q.get("latency") or {}).get("single_thread")
    if not size or not lat:
        return _missing(plt, out, "A bordo", "Nessuna misura: lancia make quantize.")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.6))

    for ax, vals, unit, title in (
        (ax1, [size["fp32_mb"], size["int8_mb"]], "MB", "Dimensione del modello"),
        (ax2, [lat["fp32"]["median_ms_per_tile"], lat["int8"]["median_ms_per_tile"]],
         "ms", "Latenza per tile (CPU, 1 thread)"),
    ):
        bars = ax.bar(["fp32", "INT8"], vals, color=[COL_MUTED, COL_THIN], width=0.55)
        ax.set_title(title, pad=10)
        ax.set_ylabel(unit)
        ax.spines[["top", "right"]].set_visible(False)
        for b, v in zip(bars, vals, strict=True):
            ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.1f}",
                    ha="center", va="bottom", fontsize=13, color=COL_INK)
        ax.set_ylim(0, max(vals) * 1.22)
        ax.text(0.5, -0.20, f"{vals[0] / vals[1]:.2f}× piu' piccolo/veloce"
                if vals[1] else "", transform=ax.transAxes, ha="center",
                fontsize=11, color=COL_THIN)

    drop = (q.get("accuracy") or {}).get("miou_drop")
    if isinstance(drop, float):
        note = ("INT8 non degrada in modo misurabile" if drop <= 0.002
                else f"INT8 costa {drop:.3f} di mIoU")
        fig.text(0.5, -0.08, note, ha="center", fontsize=12, color=COL_MUTED)
    return _save(fig, out)


# ---------------------------------------------------------------------------
# 4. It is the business
# ---------------------------------------------------------------------------


def figure_business(metrics: dict, out: Path) -> str:
    """Raw scene against downlinked payload, log scale."""
    plt = _setup()
    comp = metrics.get("compression") or {}
    per = comp.get("per_scene") or []
    if not per:
        return _missing(plt, out, "Downlink", "Nessuna misura: lancia make report.")

    s = per[0]
    raw = s["numerator"]["sar_channels_png"]
    payload = s["denominator"]["total"]

    fig, ax = plt.subplots(figsize=(8.5, 5))
    bars = ax.bar(
        ["Scena SAR\n(compressa)", "Payload IceRoute\n(maschera + rotta)"],
        [raw, payload], color=[COL_MUTED, COL_THIN], width=0.5,
    )
    ax.set_yscale("log")
    ax.set_ylabel("byte (scala logaritmica)")
    ax.spines[["top", "right"]].set_visible(False)
    for b, v in zip(bars, [raw, payload], strict=True):
        label = f"{v / 1e6:.1f} MB" if v > 1e6 else f"{v / 1e3:.0f} KB"
        ax.text(b.get_x() + b.get_width() / 2, v * 1.15, label,
                ha="center", fontsize=14, color=COL_INK)

    ratio = comp.get("ratio_median", raw / max(payload, 1))
    title = f"{ratio:.0f}:1"
    if comp.get("provisional"):
        title += "  (limite superiore)"
    ax.set_title(title, pad=14, fontsize=20)
    ax.text(0.5, -0.16,
            "numeratore: canali SAR compressi senza perdita, non byte grezzi",
            transform=ax.transAxes, ha="center", fontsize=10.5, color=COL_MUTED)
    return _save(fig, out)


# ---------------------------------------------------------------------------
# 5. Due diligence
# ---------------------------------------------------------------------------


def figure_confusion(metrics: dict, out: Path) -> str:
    plt = _setup()
    ev = metrics.get("evaluation") or {}
    agg = (ev.get("scenes") or {}).get("aggregate") or ev.get("patches") or {}
    matrix = agg.get("confusion_matrix")
    if not matrix:
        return _missing(plt, out, "Matrice di confusione", "Lancia make eval.")

    m = np.array(matrix, dtype="float64")
    support = m.sum(axis=1, keepdims=True)
    norm = np.divide(m, support, out=np.zeros_like(m), where=support > 0)
    names = list((agg.get("per_class") or {}).keys()) or [str(i) for i in range(len(m))]

    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    im = ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(names)), names, rotation=30, ha="right")
    ax.set_yticks(range(len(names)), names)
    ax.set_xlabel("Predizione")
    ax.set_ylabel("Carta dell'analista")
    for i in range(len(names)):
        for j in range(len(names)):
            if support[i] > 0:
                ax.text(j, i, f"{norm[i, j]:.2f}", ha="center", va="center",
                        fontsize=11, color="white" if norm[i, j] > 0.55 else "#111823")
    ax.set_title(f"mIoU {agg.get('miou', float('nan')):.3f} · "
                 f"ghiaccio/acqua {agg.get('binary_accuracy', float('nan')):.3f}", pad=14)
    fig.colorbar(im, ax=ax, label="frazione della classe vera")
    return _save(fig, out)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Genera le figure del deck. SPEC §10.3.")
    ap.add_argument("--demo-dir", type=Path, default=Path("data/demo"))
    ap.add_argument("--metrics", type=Path, default=Path("artifacts/metrics.json"))
    ap.add_argument("--out", type=Path, default=Path("artifacts/figures"))
    ap.add_argument("--ice-class", default="PC6")
    args = ap.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)-7s %(message)s")

    metrics = json.loads(args.metrics.read_text()) if args.metrics.exists() else {}
    index_path = args.demo_dir / "index.json"
    index = json.loads(index_path.read_text()) if index_path.exists() else {}

    # Lead with a scene that has a route for the reference class: a figure of a
    # scene with no route argues the opposite of what it is there to argue.
    scenes = [s for s in index.get("scenes", []) if s.get("classes")]
    lead = next(
        (s for s in scenes
         if s["classes"].get(args.ice_class, {}).get("route_exists")),
        scenes[0] if scenes else None,
    )

    log.info("Figure in %s", args.out)
    figure_works(args.demo_dir, args.out / "01_works.png", lead)
    figure_useful(args.demo_dir, args.out / "02_useful.png", lead, args.ice_class)
    figure_onboard(metrics, args.out / "03_onboard.png")
    figure_business(metrics, args.out / "04_business.png")
    figure_confusion(metrics, args.out / "05_confusion.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
