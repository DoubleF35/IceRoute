"""Generate everything the demo displays, ahead of time.

The UI reads **only** files produced here. Nothing is inferred, geocoded or
routed while someone is watching, because the demo has to start in under two
seconds and must not depend on PyTorch initialising correctly on whatever
machine it is shown from.

The one thing that matters most in this file
--------------------------------------------
Routing runs on the **predicted** mask, not on the ice chart. Routing on the
chart would demonstrate an ice analysis we did not produce — the whole claim is
that the satellite computes this itself. So the chain here is

    scene -> model -> predicted mask -> geocode -> RIO -> A* -> route

and the ground-truth chart is carried alongside purely so the UI can show the
comparison honestly. Every route in ``data/demo/`` is therefore a route through
ice as IceRoute *believes* it to be, which is exactly what a ship would get.

Rendering is tuned for a projector in a dark room, not for a laptop screen: a
percentile stretch rather than min/max (one bright outlier flattens everything
else), and a palette with no pastels.
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import yaml

from iceroute.data.collapse import N_CLASSES, collapse, polaris_mapping
from iceroute.data.collapse import NAMES as CLASS_NAMES
from iceroute.data.loader import load_config, load_scene, reference_path
from iceroute.demo import overlay as overlay_mod
from iceroute.model.metrics import IGNORE_INDEX
from iceroute.routing.astar import NoRouteFound, find_route
from iceroute.routing.baseline import compare, geodesic_baseline
from iceroute.routing.cost import build_cost_grid, build_land_mask, speed_model_from_config
from iceroute.routing.geocode import geocode_scene
from iceroute.routing.polaris import (
    PolarisTable,
    classify_rio,
    compute_rio,
    conditions_along,
    summarise,
)
from iceroute.routing.shortcut import straighten
from iceroute.routing.simplify import simplify_route
from iceroute.routing.waypoints import choose_waypoints, waypoints_from_config

log = logging.getLogger("precompute")

# High contrast, no pastels: a projector washes out mid tones and these have to
# stay separable from across a room.
CLASS_COLOURS = {
    0: (10, 40, 90),  # open water — deep blue
    1: (0, 200, 255),  # thin ice — bright cyan
    2: (255, 255, 255),  # thick ice — white
}
NODATA_COLOUR = (26, 26, 26)

# Sea ice concentration, in tenths. A separate hue family on purpose: it must
# not be mistaken for either the ice-class palette (blue/cyan/white) or the RIO
# one (green/yellow/red), because its whole job is to be the *third* thing —
# the half of the POLARIS decision that neither of those shows.
#
# This layer exists because of a real failure. The demo drew routes over the
# class mask, the mask calls a whole third of the scene "thick ice", and the
# route crosses it — which reads as a bug and is not one: those cells are at
# 3/10 concentration, RIO +12. Without a concentration layer the demo could not
# show why, only assert it.
SIC_COLOURS = {
    0: (24, 22, 38),
    1: (52, 30, 74),
    2: (86, 36, 98),
    3: (124, 42, 104),
    4: (160, 52, 98),
    5: (194, 68, 84),
    6: (222, 92, 66),
    7: (240, 124, 48),
    8: (250, 162, 36),
    9: (252, 202, 48),
    10: (255, 240, 120),
}

# RIO is rendered around the two POLARIS decision boundaries, not on a smooth
# ramp: the eye should read "may I go here", not "what number is this".
RIO_COLOURS = {
    0: (46, 204, 113),  # normal
    1: (241, 196, 15),  # elevated risk
    2: (231, 76, 60),  # special consideration
    3: (40, 40, 40),  # unknown / outside coverage
}


@dataclass
class SceneAssets:
    """What one scene contributes to the demo."""

    scene_id: str
    region: str
    acquired: str | None
    shape: list[int]
    coverage: float
    files: dict[str, str] = field(default_factory=dict)
    classes: dict = field(default_factory=dict)
    waypoints: dict = field(default_factory=dict)
    inference_seconds: float = 0.0
    routes_differ: bool = False
    # A scene kept in the build precisely *because* the model does badly on it.
    # It is not part of the carousel: it exists so "show me where it fails" has
    # an answer ready. The UI must label it, or it becomes an accidental claim.
    worst_case: bool = False
    warnings: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


def stretch(band: np.ndarray, valid: np.ndarray, lo: float = 2.0, hi: float = 98.0) -> np.ndarray:
    """Percentile stretch to 8-bit.

    Min/max would let a single bright scatterer compress everything else into a
    few grey levels — common in SAR, and it looks like a broken image on a
    projector.
    """
    px = band[valid]
    if px.size == 0:
        return np.zeros(band.shape, dtype="uint8")
    a, b = np.percentile(px, [lo, hi])
    if b - a < 1e-9:
        b = a + 1.0
    out = np.clip((band - a) / (b - a), 0, 1)
    return (out * 255).astype("uint8")


def save_png(array: np.ndarray, path: Path, *, downscale: int = 4) -> str:
    """Write a PNG, downscaled by simple striding.

    Striding rather than averaging: the mask is categorical, and averaging class
    ids invents classes that were never predicted. The SAR is decimated the same
    way so the two layers stay pixel-aligned in the UI.
    """
    from PIL import Image

    arr = array[::downscale, ::downscale]
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr).save(path, optimize=True)
    return path.name


def colour_mask(labels: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """RGB rendering of a class map."""
    out = np.zeros((*labels.shape, 3), dtype="uint8")
    out[:] = NODATA_COLOUR
    for cls, rgb in CLASS_COLOURS.items():
        out[(labels == cls) & valid] = rgb
    return out


def colour_sic(sic: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """RGB rendering of ice concentration in tenths."""
    conc = np.clip(np.rint(np.asarray(sic, dtype="float64")), 0, 10).astype("int64")
    out = np.zeros((*conc.shape, 3), dtype="uint8")
    out[:] = NODATA_COLOUR
    for tenth, rgb in SIC_COLOURS.items():
        out[(conc == tenth) & valid] = rgb
    return out


def colour_rio(rio: np.ndarray, ice_class: str) -> np.ndarray:
    """RGB rendering of a RIO raster by POLARIS operation level."""
    levels = classify_rio(rio, ice_class)
    out = np.zeros((*levels.shape, 3), dtype="uint8")
    for lvl, rgb in RIO_COLOURS.items():
        out[levels == lvl] = rgb
    return out


# ---------------------------------------------------------------------------
# One scene
# ---------------------------------------------------------------------------


def predict_mask(model, scene, channels: list[str], eval_cfg: dict, device: str):
    """Run the model over a whole scene and collapse to the demo scheme."""
    from iceroute.model.inference import predict_scene

    labels, _, stats = predict_scene(
        model,
        scene.channels(channels),
        n_classes=N_CLASSES,
        tile_size=int(eval_cfg.get("tile_size", 256)),
        overlap=int(eval_cfg.get("tile_overlap", 32)),
        batch_size=int(eval_cfg.get("batch_size", 8)),
        device=device,
    )
    return labels, stats


def resolve_labels(scene_path: Path) -> Path | None:
    """Where a scene's ice chart lives.

    The official challenge test scenes carry no targets — that is what makes
    them a test set — and their charts sit alongside in ``*_prep_reference.nc``.
    Resolved by looking on disk rather than by a flag, so pointing precompute at
    either directory just works and cannot be pointed at the wrong one.
    """
    ref = reference_path(scene_path)
    return ref if ref != scene_path and ref.exists() else None


def process_scene(
    scene_path: Path,
    *,
    model,
    data_cfg: dict,
    routing_cfg: dict,
    train_cfg: dict,
    out_dir: Path,
    table: PolarisTable,
    device: str,
    downscale: int,
    use_chart: bool = False,
    worst_case: bool = False,
    on_stage: Callable[[str], None] | None = None,
) -> SceneAssets:
    """Everything the UI needs for one scene.

    ``use_chart`` routes on the analyst's chart instead of the model. That is
    **not** a demo mode — it exists to rank candidate scenes on the ice itself,
    independently of a model that may still be training. Runs in this mode set
    ``generated_from_predicted_mask: false`` in the index, so a ranking pass can
    never be mistaken for a demo build.

    ``on_stage`` is called as each phase begins. It exists so the demo's "run it
    now" button can show the chain advancing **through this same function**: a
    second implementation of the pipeline for the live path would make "watch it
    compute" a claim about code the product does not use.
    """
    step = on_stage or (lambda _name: None)
    step("Reading the scene from disk")
    labels_from = resolve_labels(scene_path)
    scene = load_scene(scene_path, data_cfg, labels_from=labels_from)
    stem = scene.meta.scene_id.replace(".nc", "")
    channels = list(train_cfg["data"].get("channels") or ["hh", "hv", "incidence_angle"])
    ice_classes = list(routing_cfg["vessel"]["classes"])
    v_open = float(routing_cfg["vessel"]["open_water_speed_kn"])
    astar_cfg, simp_cfg = routing_cfg["astar"], routing_cfg["simplify"]

    truth = collapse(np.where(scene.mask_valid, scene.sod, IGNORE_INDEX).astype("uint8"))

    if use_chart:
        pred, inf_stats, infer_s = truth, None, 0.0
    else:
        step("U-Net inference on the full scene")
        t0 = time.monotonic()
        pred, inf_stats = predict_mask(model, scene, channels, train_cfg.get("eval", {}), device)
        infer_s = time.monotonic() - t0

    assets = SceneAssets(
        scene_id=scene.meta.scene_id,
        region=scene.meta.region,
        acquired=scene.meta.acquired.isoformat() if scene.meta.acquired else None,
        shape=list(scene.shape),
        coverage=round(float(scene.mask_valid.mean()), 4),
        inference_seconds=round(infer_s, 2),
        worst_case=worst_case,
    )

    # -- rasters ------------------------------------------------------------
    step("Rendering the masks")
    img_dir = out_dir / "images"
    hv8 = stretch(scene.hv, scene.mask_valid)
    assets.files["sar"] = save_png(hv8, img_dir / f"{stem}_sar.png", downscale=downscale)
    assets.files["mask_pred"] = save_png(
        colour_mask(pred, scene.mask_valid), img_dir / f"{stem}_mask.png", downscale=downscale
    )
    assets.files["mask_truth"] = save_png(
        colour_mask(truth, scene.mask_valid), img_dir / f"{stem}_truth.png", downscale=downscale
    )

    # -- geocode, on the PREDICTED mask ------------------------------------
    # geocode_scene resamples whatever sits in scene.sod, so the prediction is
    # substituted before geocoding. The chart stays available as scene.sic etc.
    step("Geocoding to EPSG:3413 at 500 m")
    scene.sod = pred.astype("uint8")
    geo = geocode_scene(scene, resolution_m=float(routing_cfg["grid"]["resolution_m"]))
    land = build_land_mask(
        geo.distance_to_land, geo.shape, buffer_cells=int(astar_cfg["land_buffer_cells"])
    )
    mapping = polaris_mapping(
        {int(k): v for k, v in data_cfg["sod_to_polaris"].items()}, ice_classes, table=table
    )

    # The mask again, this time on the geocoded grid. The raw-swath version
    # above is the honest picture of what the model saw; this one is the only
    # backdrop a route can legally be drawn on, because the route cells are
    # indexed in *this* grid and no other. Keeping both, clearly named, is what
    # stops someone pairing a track with the wrong raster (see demo/overlay.py).
    assets.files["mask_geo"] = save_png(
        colour_mask(geo.sod, geo.mask_valid), img_dir / f"{stem}_maskgeo.png", downscale=1
    )
    if geo.sic is not None:
        assets.files["sic"] = save_png(
            colour_sic(geo.sic, geo.mask_valid), img_dir / f"{stem}_sic.png", downscale=1
        )

    step("POLARIS RIO and cost grid")
    rios, costs = {}, {}
    for ic in ice_classes:
        rio = compute_rio(
            geo.sod,
            ic,
            mapping,
            table=table,
            sic=geo.sic,
            mask_valid=geo.mask_valid,
            n_classes=N_CLASSES,
        )
        rios[ic] = rio
        costs[ic] = build_cost_grid(rio, speed_model_from_config(routing_cfg, ic), land_mask=land)
        assets.files[f"rio_{ic}"] = save_png(
            colour_rio(rio, ic), img_dir / f"{stem}_rio_{ic}.png", downscale=1
        )

    # -- waypoints ----------------------------------------------------------
    step("Choosing the waypoints")
    entry = (routing_cfg.get("waypoints") or {}).get(scene.meta.scene_id)
    try:
        wp = waypoints_from_config(geo.grid, entry) if entry else choose_waypoints(geo.grid, costs)
    except ValueError as exc:
        assets.warnings.append(f"Waypoints could not be selected: {exc}")
        log.warning("  %s", assets.warnings[-1])
        return assets
    assets.waypoints = wp.as_dict()

    # -- routes -------------------------------------------------------------
    step("A*, simplification and comparison with the direct route")
    routes_dir = out_dir / "routes"
    signatures = []
    for ic in ice_classes:
        cost, rio = costs[ic], rios[ic]
        entry_out: dict = {"polaris": summarise(rio, ic)}
        try:
            route = find_route(
                cost,
                geo.grid,
                wp.start,
                wp.goal,
                connectivity=int(astar_cfg["connectivity"]),
                heuristic_weight=float(astar_cfg["heuristic_weight"]),
                max_expansions=int(astar_cfg["max_expansions"]),
                rio=rio,
            )
        except NoRouteFound as exc:
            # An expected outcome, and an informative one: "with this class
            # the route does not exist" is a result, not a failure.
            entry_out.update({"route_exists": False, "reason": exc.reason, "message": str(exc)})
            assets.classes[ic] = entry_out
            signatures.append(None)
            log.info("  %s: no route (%s)", ic, exc.reason)
            continue

        # String pulling, before anything is measured off the route: A* is
        # optimal for the grid, and the grid is what makes it wander.
        if bool(astar_cfg.get("straighten", True)):
            pulled = straighten(route, cost, geo.grid, rio=rio)
            route = pulled.route
            entry_out["straightened"] = {
                "legs": pulled.n_legs,
                "cells_before": pulled.n_cells_before,
                "seconds_saved": round(pulled.seconds_saved, 1),
                "metres_saved": round(pulled.metres_saved, 1),
                "percent_time": round(pulled.percent_time, 3),
                "percent_distance": round(pulled.percent_distance, 3),
            }

        base = geodesic_baseline(
            geo.grid,
            cost,
            wp.start,
            wp.goal,
            open_water_speed_kn=v_open,
            rio=rio,
            land_mask=land,
            coverage=geo.mask_valid,
        )
        simple = simplify_route(
            route,
            geo.grid,
            tolerance_m=float(simp_cfg["tolerance_m"]),
            max_waypoints=int(simp_cfg["max_waypoints"]),
            max_turn_deg=float(simp_cfg["max_turn_deg"]),
        )
        path = routes_dir / f"{stem}__{ic}.geojson"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {
                    "type": "FeatureCollection",
                    "features": [
                        simple.as_geojson(
                            {
                                "kind": "iceroute",
                                "ice_class": ic,
                                "transit_time_hours": round(route.time_hours, 3),
                                "distance_km": round(route.distance_km, 3),
                                "min_rio_on_route": None
                                if route.min_rio is None
                                else round(route.min_rio, 2),
                            }
                        ),
                        base.as_geojson(),
                    ],
                },
                indent=1,
            )
        )

        # The drawable form of the same two tracks, in grid pixels. Separate
        # file, never inside the GeoJSON: that one goes down the downlink and
        # its byte count is part of the compression figure.
        ov_path = out_dir / overlay_mod.OVERLAY_DIR / f"{stem}__{ic}.json"
        ov_path.parent.mkdir(parents=True, exist_ok=True)
        ov = overlay_mod.build(geo.grid, route, simple, base, wp.start, wp.goal)
        ov_path.write_text(json.dumps(ov))
        entry_out["separation"] = ov.get("separation") or {}
        # The answer to "but your route goes through the thick ice". Measured
        # here, where the rasters exist, so the UI states it instead of the
        # viewer inferring the opposite from a class map.
        entry_out["conditions"] = conditions_along(
            route.cells, geo.sod, geo.sic, rio, ic, CLASS_NAMES
        )

        entry_out.update(
            {
                "route_exists": True,
                "geojson": path.name,
                "overlay": ov_path.name,
                "comparison": compare(route, base),
                "n_waypoints": len(simple),
                "simplify_warnings": simple.warnings,
            }
        )
        assets.classes[ic] = entry_out
        signatures.append(tuple(route.cells))
        log.info("  %s: %s", ic, entry_out["comparison"]["headline"])

    # docs/decisions.md D-012: the vessel-class comparison only means something
    # if the route actually changes with the class. Recorded per scene so the
    # five demo scenes can be picked on evidence.
    distinct = {s for s in signatures if s is not None}
    assets.routes_differ = len(distinct) > 1 or None in signatures
    if not assets.routes_differ:
        assets.warnings.append(
            "Every vessel class produces the same route: this scene does not "
            "support the 'change class and there is no way through' comparison (D-012)."
        )

    if inf_stats is not None:
        assets.files["inference_stats"] = json.dumps(inf_stats.as_dict())
    return assets


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Generate data/demo/ for the UI.")
    ap.add_argument("--config", type=Path, default=Path("configs/routing.yaml"))
    ap.add_argument("--data-config", type=Path, default=Path("configs/data.yaml"))
    ap.add_argument(
        "--train-config",
        type=Path,
        default=None,
        help="default: the deployed config, from artifacts/best_config.txt",
    )
    ap.add_argument("--checkpoint", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=Path("data/demo"))
    ap.add_argument("--demo-config", type=Path, default=Path("configs/demo.yaml"))
    ap.add_argument("--raw-dir", type=Path, default=None)
    ap.add_argument("--scenes", nargs="*", default=None)
    ap.add_argument("--downscale", type=int, default=4, help="PNG downscale factor")
    ap.add_argument("--device", default=None)
    ap.add_argument(
        "--use-chart",
        action="store_true",
        help="route on the ice chart instead of the model: used to RANK "
        "candidate scenes, not to build the demo",
    )
    args = ap.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s  %(levelname)-7s %(message)s", datefmt="%H:%M:%S"
    )

    import torch

    from iceroute.model.evaluate import load_checkpoint

    data_cfg = load_config(args.data_config)
    routing_cfg = yaml.safe_load(args.config.read_text())
    from iceroute.report.metrics import deployed_config

    train_path = args.train_config or deployed_config()
    train_cfg = yaml.safe_load(train_path.read_text())
    log.info("Model: %s", train_path)
    table = PolarisTable.load()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    model, ckpt_path = None, None
    if args.use_chart:
        log.warning(
            "RANKING MODE: routes are computed on the ice chart, not on the "
            "model. Used to choose candidate scenes. Do NOT use the output as the demo."
        )
    else:
        ckpt_path = args.checkpoint or Path(train_cfg["output"]["best_checkpoint"])
        if not ckpt_path.exists():
            log.error("Checkpoint missing: %s. Run 'make train' first.", ckpt_path)
            return 1
        model, _ = load_checkpoint(ckpt_path, device)

    # Scene source, in order of precedence: explicit --scenes, then the demo
    # config, then whatever directory was given. The config exists so the choice
    # of scenes is a reviewed decision in git, not a flag someone typed once.
    demo_cfg = yaml.safe_load(args.demo_config.read_text()) if args.demo_config.exists() else {}
    if args.raw_dir is None:
        args.raw_dir = Path(demo_cfg.get("source_dir", "data/raw/train"))

    # Each entry is (scene_id, is_worst_case). The worst case is built alongside
    # the carousel rather than in a separate run: a scene that only gets
    # regenerated when someone remembers to is a scene that will be stale when
    # it is needed, and this one exists to be shown under questioning.
    worst_id = ((demo_cfg.get("worst_case") or {}).get("id")) if demo_cfg else None

    if args.scenes:
        scenes = list(args.scenes)
    elif demo_cfg.get("scenes"):
        scenes = [e["id"] for e in demo_cfg["scenes"]]
        if worst_id and worst_id not in scenes:
            scenes.append(worst_id)
        log.info(
            "Demo scenes from %s: %d%s",
            args.demo_config,
            len(scenes),
            f" (including 1 worst case: {worst_id[:18]})" if worst_id else "",
        )
    elif args.raw_dir.name == "test":
        # Official test scenes are not in the frozen split; enumerate from disk,
        # and only those whose reference labels are present.
        scenes = [
            p.name for p in sorted(args.raw_dir.glob("*_prep.nc")) if resolve_labels(p) is not None
        ]
    else:
        split = yaml.safe_load(Path(data_cfg.get("split_file", "configs/split.yaml")).read_text())[
            "split"
        ]
        scenes = list(split["demo"])

    present = [s for s in scenes if (args.raw_dir / s).exists()]
    if not present:
        log.error("No demo scene in %s.", args.raw_dir)
        return 1

    args.out.mkdir(parents=True, exist_ok=True)
    results = []
    for i, sid in enumerate(present, 1):
        is_worst = sid == worst_id
        log.info("[%d/%d] %s%s", i, len(present), sid, "  [WORST CASE]" if is_worst else "")
        try:
            results.append(
                process_scene(
                    args.raw_dir / sid,
                    model=model,
                    data_cfg=data_cfg,
                    routing_cfg=routing_cfg,
                    train_cfg=train_cfg,
                    out_dir=args.out,
                    table=table,
                    device=device,
                    downscale=args.downscale,
                    use_chart=args.use_chart,
                    worst_case=is_worst,
                )
            )
        except Exception as exc:  # noqa: BLE001 — one bad scene must not stop the batch
            log.exception("  failed: %s", exc)

    index = {
        "generated_from_predicted_mask": not args.use_chart,
        "note": (
            "Routes are computed on the mask PREDICTED by the model, not on the ice "
            "chart. Routing on the chart would demonstrate an analysis we did not "
            "produce ourselves."
            if not args.use_chart
            else "RANKING: routes computed on the ICE CHART, not on the model. Only used "
            "to choose candidate scenes. Not an output usable as the demo."
        ),
        "scheme": "3 classes (fallback, docs/decisions.md D-020)",
        "class_names": CLASS_NAMES,
        "ice_classes": routing_cfg["vessel"]["classes"],
        "open_water_speed_kn": routing_cfg["vessel"]["open_water_speed_kn"],
        "open_water_speed_is_provisional": True,
        "checkpoint": str(ckpt_path) if ckpt_path else None,
        "scenes": [
            {k: v for k, v in a.__dict__.items() if k != "files"} | {"files": a.files}
            for a in results
        ],
    }
    (args.out / "index.json").write_text(json.dumps(index, indent=2))

    # The worst case is not part of the carousel, so it must not dilute (or
    # flatter) the count of scenes that carry the vessel-class moment.
    carousel = [a for a in results if not a.worst_case]
    differing = [a.scene_id for a in carousel if a.routes_differ]
    log.info("=" * 66)
    log.info(
        "Scenes generated: %d (%d in the carousel + %d worst case)",
        len(results),
        len(carousel),
        len(results) - len(carousel),
    )
    log.info(
        "Scenes where the route CHANGES with the vessel class: %d/%d%s",
        len(differing),
        len(carousel),
        f" — {', '.join(s[:15] for s in differing)}" if differing else "",
    )
    # Which scene actually *looks* like the claim. A scene can forbid the
    # direct route in three hundred cells and still draw two lines on top of
    # each other; on screen that argues nothing. Ranked here so the choice of
    # which scene to open first is made against a measurement.
    ranked = sorted(
        (
            (a.scene_id, ic, (e.get("separation") or {}).get("pct_diagonal") or 0.0)
            for a in carousel
            for ic, e in a.classes.items()
            if e.get("route_exists")
        ),
        key=lambda t: -t[2],
    )
    if ranked:
        log.info("Visible separation between the two routes (% of the diagonal):")
        for sid, ic, pct in ranked[:4]:
            log.info("    %-16s %-4s %5.2f%%", sid[:16], ic, pct)

    if not differing:
        log.warning(
            "No scene supports the 'change class and there is no way through' "
            "comparison. Consider removing the vessel-class selector from the UI "
            "BEFORE building the interface around it."
        )
    log.info("Index written to %s", args.out / "index.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
