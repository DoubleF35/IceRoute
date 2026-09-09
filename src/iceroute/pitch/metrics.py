"""Consolidate every measurement, and rule on what the deck may claim. SPEC §10.3.

This module produces no numbers of its own. It gathers what the other stages
measured, measures the downlink ratio on the **predicted** masks, and then does
the part that matters: for each public claim, it says whether a measurement
supports it, contradicts it, or is missing.

That last step exists because the failure mode of a pitch is not a wrong
calculation — it is a number that was true of a prototype in June, was never
re-measured, and gets read aloud in October. ``artifacts/claims.md`` is the
artefact Federico reads before writing a slide.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import yaml

from iceroute.data.collapse import N_CLASSES
from iceroute.data.loader import load_config, load_scene
from iceroute.demo.precompute import resolve_labels
from iceroute.pitch.compression import downlink_route, measure_scene, summarise, verdict

DEPLOYED = Path("artifacts/best_config.txt")


def deployed_config(fallback: Path = Path("configs/train3.yaml")) -> Path:
    """The config the model picker chose, not a filename frozen in an argparse default.

    ``configs/train3.yaml`` names the *baseline* checkpoint. The deployed model is
    whatever ``scripts/pick_model.py`` last wrote into ``artifacts/best_config.txt``
    (D-025: the Lovász run). Defaulting to the fixed name measured a model nobody
    ships — and did so silently, because both files load and both produce numbers.

    That is exactly how the compression figure moved from 439:1 to 347:1 on
    2026-09-09 without a line of measurement code changing: ``make report`` had not
    been re-run since the model switch, so the wrong default had never fired.
    """
    if DEPLOYED.exists():
        picked = Path(DEPLOYED.read_text().strip())
        if picked.exists():
            return picked
    return fallback


log = logging.getLogger("metrics")


# ---------------------------------------------------------------------------
# Downlink, measured on predicted masks — closes P-06
# ---------------------------------------------------------------------------


def measure_downlink(
    demo_dir: Path,
    data_cfg: dict,
    raw_dir: Path,
    *,
    checkpoint: Path | None = None,
    device: str = "cpu",
    limit: int = 5,
) -> dict:
    """Compression ratio on the masks the model actually produces.

    D-016 measured this on the analyst's chart and refused to quote the result,
    because a chart is drawn as polygons and compresses far better than a
    network's speckled output. This runs the model so the number is the
    operational one.
    """
    index_path = demo_dir / "index.json"
    if not index_path.exists():
        return {"error": "data/demo/index.json assente: lancia make precompute."}
    index = json.loads(index_path.read_text())
    if not index.get("generated_from_predicted_mask"):
        return {"error": "L'indice viene da una classifica su ice chart, non dal modello."}

    import torch

    from iceroute.model.evaluate import load_checkpoint
    from iceroute.model.inference import predict_scene

    if checkpoint is None or not Path(checkpoint).exists():
        return {"error": f"Checkpoint assente: {checkpoint}"}
    model, _ = load_checkpoint(Path(checkpoint), device)
    channels = ["hh", "hv", "incidence_angle"]

    results = []
    for entry in index.get("scenes", [])[:limit]:
        path = raw_dir / entry["scene_id"]
        if not path.exists():
            continue
        scene = load_scene(path, data_cfg, labels_from=resolve_labels(path))
        pred, _, _ = predict_scene(
            model, scene.channels(channels), n_classes=N_CLASSES, device=device
        )
        mask = np.where(scene.mask_valid, pred, 255).astype("uint8")

        route = None
        first = next(iter(entry.get("classes", {}).values()), {})
        gj = demo_dir / "routes" / first.get("geojson", "")
        if gj.exists():
            # Only the transmitted track. The same file also holds the
            # ground-side comparison geodesic, 26 KB the satellite never sends.
            route = downlink_route(json.loads(gj.read_text()))

        results.append(measure_scene(
            scene.meta.scene_id, mask, np.stack([scene.hh, scene.hv]),
            n_classes=N_CLASSES, mask_source="predicted", netcdf_path=path,
            route_geojson=route, coverage=float(scene.mask_valid.mean()),
            metadata={"scene_id": scene.meta.scene_id, "crs": "EPSG:3413"},
        ))
        log.info("  %s: %.0f:1", scene.meta.scene_id[:18], results[-1].ratio)
        del scene, pred
        if device == "cuda":
            torch.cuda.empty_cache()

    if not results:
        return {"error": "Nessuna scena misurabile."}
    out = summarise(results)
    out["verdict"] = verdict(out, claimed=200.0)
    return out


# ---------------------------------------------------------------------------
# Ruling on the public claims
# ---------------------------------------------------------------------------


def _fmt(v, spec="{:.3f}"):
    return spec.format(v) if isinstance(v, int | float) and np.isfinite(v) else "non misurato"


def adjudicate(metrics: dict) -> list[dict]:
    """For each claim in the June deck: supported, contradicted, or unmeasured.

    Deliberately blunt. A row that says "contraddetta" with both numbers next to
    it is what stops a slide going out unchanged.
    """
    q = metrics.get("quantization") or {}
    size = q.get("size") or {}
    lat = ((q.get("latency") or {}).get("single_thread") or {}).get("int8") or {}
    acc = q.get("accuracy") or {}
    ev = metrics.get("evaluation") or {}
    agg = (ev.get("scenes") or {}).get("aggregate") or {}
    comp = metrics.get("compression") or {}

    rows = []

    def add(claim_id, claimed, measured, status, action):
        rows.append({"id": claim_id, "claimed": claimed, "measured": measured,
                     "status": status, "action": action})

    int8_kb = size.get("int8_kb")
    add("A-02", "58 KB di pesi INT8",
        f"{int8_kb:.0f} KB" if int8_kb else "non misurato",
        "contraddetta" if int8_kb else "non misurata",
        f"Sostituire con {size.get('int8_mb', 0):.2f} MB. A "
        f"{size.get('bytes_per_parameter_int8', 0):.2f} byte/parametro i 58 KB sono "
        f"aritmeticamente impossibili." if int8_kb else "Lancia make quantize.")

    drop = acc.get("miou_drop")
    add("A-04", "1–3% di mIoU perso con INT8",
        f"{drop:+.4f}" if isinstance(drop, float) else "non misurato",
        "contraddetta (in meglio)" if isinstance(drop, float) and drop <= 0.01
        else "non misurata",
        "Dire che il degrado misurato e' nullo: e' meglio di quanto dichiarato."
        if isinstance(drop, float) else "Lancia make quantize.")

    ratio = comp.get("ratio_median")
    add("P-06 / compressione", "~200:1",
        f"{ratio:.0f}:1" if ratio else "non misurato",
        ("provvisoria" if comp.get("provisional") else "sostenuta")
        if ratio else "non misurata",
        "Misurata sull'ice chart: non citabile finche' non e' sulla maschera predetta."
        if comp.get("provisional") else
        (f"Usare {ratio:.0f}:1." if ratio else "Lancia make report."))

    miou = agg.get("miou")
    add("mIoU", "6 stadi di sviluppo del ghiaccio",
        f"{_fmt(miou)} su 3 classi" if miou else "non misurato",
        "ridimensionata",
        "Il deck deve dire 3 classi: acqua libera, ghiaccio sottile, ghiaccio spesso. "
        "Il ripiego SPEC §7.4 e' attivo (D-020).")

    add("A-03", "5,3 ms per tile (proiezione GPU@SAT)",
        f"{lat.get('median_ms_per_tile', float('nan')):.0f} ms su CPU 1 thread"
        if lat else "non misurata",
        "grandezza diversa",
        "La latenza CPU non valida la proiezione GPU@SAT: sono due cose. "
        "Etichettare la proiezione come tale.")

    unsafe = (ev.get("scenes") or {}).get("worst_unsafe_fraction")
    if isinstance(unsafe, float):
        add("sicurezza", "(nessuna affermazione pubblica)",
            f"{unsafe:.1%} permissiva nel caso peggiore",
            "da dichiarare",
            "E' la frazione di pixel dove una rotta passerebbe in ghiaccio vietato. "
            "Va detta prima che la trovi qualcun altro.")
    return rows


def write_claims(rows: list[dict], path: Path) -> None:
    lines = [
        "# Cosa il deck puo' dire, e cosa no",
        "",
        "> Generato da `make report`. Ogni riga confronta un'affermazione pubblica",
        "> con una misura. Non modificare a mano: si rigenera.",
        "",
        "| ID | Dichiarato | Misurato | Stato | Azione |",
        "|---|---|---|---|---|",
    ]
    lines += [
        f"| {r['id']} | {r['claimed']} | **{r['measured']}** | {r['status']} | {r['action']} |"
        for r in rows
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Consolida le misure per il pitch. SPEC §10.3.")
    ap.add_argument("--metrics", type=Path, default=Path("artifacts/metrics.json"))
    ap.add_argument("--demo-dir", type=Path, default=Path("data/demo"))
    ap.add_argument("--data-config", type=Path, default=Path("configs/data.yaml"))
    ap.add_argument("--train-config", type=Path, default=None,
                    help="default: il config schierato, da artifacts/best_config.txt")
    ap.add_argument("--raw-dir", type=Path, default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--skip-downlink", action="store_true")
    args = ap.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)-7s %(message)s")

    metrics = json.loads(args.metrics.read_text()) if args.metrics.exists() else {}
    data_cfg = load_config(args.data_config)
    train_path = args.train_config or deployed_config()
    train_cfg = yaml.safe_load(train_path.read_text())
    log.info("Modello: %s", train_path)

    if args.raw_dir is None:
        demo_cfg_path = Path("configs/demo.yaml")
        demo_cfg = yaml.safe_load(demo_cfg_path.read_text()) if demo_cfg_path.exists() else {}
        args.raw_dir = Path(demo_cfg.get("source_dir", "data/raw/train"))

    if not args.skip_downlink:
        import torch

        device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
        log.info("Compressione sulle maschere predette...")
        metrics["compression"] = measure_downlink(
            args.demo_dir, data_cfg, args.raw_dir,
            checkpoint=Path(train_cfg["output"]["best_checkpoint"]), device=device,
        )
        # The number carries the model that produced it. Without this a stale
        # metrics.json cannot be told from a fresh one measured on the wrong net.
        metrics["compression"]["checkpoint"] = train_cfg["output"]["best_checkpoint"]
        metrics["compression"]["train_config"] = str(train_path)
        v = metrics["compression"].get("verdict") or metrics["compression"].get("error")
        log.info("  %s", v)

    args.metrics.write_text(json.dumps(metrics, indent=2))

    rows = adjudicate(metrics)
    write_claims(rows, Path("artifacts/claims.md"))
    log.info("Verdetto sulle affermazioni in artifacts/claims.md")
    for r in rows:
        log.info("  %-20s %-14s %s", r["id"], r["status"], r["measured"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
