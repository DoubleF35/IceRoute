"""How much does the demo's headline depend on the unsourced speed? Closes P-05.

``open_water_speed_kn`` has no citable source (P-05), and every transit time the
demo prints scales with it. The usual reflex is to hunt for a "typical" value
and move on, but that only replaces one assumption with a better-dressed one.

The question worth answering is different: **does the number we actually quote
move when the assumption moves?** Two numbers behave very differently here.

* Absolute hours scale with the speed almost directly. They are not defensible
  without a source, and never will be.
* The *percentage* saved against the direct route is a ratio, and much of the
  speed cancels. Not entirely: POLARIS Table 1.2 caps speed in the elevated-risk
  band at a fixed value in knots, so raising the open-water speed widens the gap
  between clear water and marginal ice, which can change the optimal route.

So the sweep is not a formality. It measures how wide the plausible band is, and
that turns P-05 from "an unsourced number that invalidates the claim" into "a
bounded assumption whose effect on the claim is measured".

**Waypoints are chosen once**, at the nominal speed, and held fixed across the
sweep. They are normally derived from the cost grid, so letting them move would
confound a change in route with a change in endpoints.
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
from pathlib import Path

import numpy as np
import yaml

from iceroute.data.collapse import N_CLASSES, polaris_mapping
from iceroute.data.loader import load_config, load_scene
from iceroute.demo.precompute import resolve_labels
from iceroute.routing.astar import NoRouteFound, find_route
from iceroute.routing.baseline import compare, geodesic_baseline
from iceroute.routing.cost import build_cost_grid, build_land_mask, speed_model_from_config
from iceroute.routing.geocode import geocode_scene
from iceroute.routing.polaris import PolarisTable, compute_rio
from iceroute.routing.waypoints import choose_waypoints

log = logging.getLogger("speed")

# The band to sweep. Bottom end is a cautious loaded transit; the top is the
# open-water capability quoted for modern ice-class cargo vessels. The nominal
# 12,0 of configs/routing.yaml sits inside it.
DEFAULT_SPEEDS = [10.0, 12.0, 14.2, 16.0, 19.5]


def sweep_scene(
    scene_path: Path,
    *,
    model,
    data_cfg: dict,
    routing_cfg: dict,
    train_cfg: dict,
    table: PolarisTable,
    device: str,
    speeds: list[float],
) -> dict:
    from iceroute.model.inference import predict_scene

    channels = list(train_cfg["data"].get("channels") or ["hh", "hv", "incidence_angle"])
    ice_classes = list(routing_cfg["vessel"]["classes"])
    astar_cfg, grid_cfg = routing_cfg["astar"], routing_cfg["grid"]

    scene = load_scene(scene_path, data_cfg, labels_from=resolve_labels(scene_path))
    pred, _, _ = predict_scene(model, scene.channels(channels), n_classes=N_CLASSES, device=device)
    scene.sod = pred.astype("uint8")

    geo = geocode_scene(scene, resolution_m=float(grid_cfg["resolution_m"]))
    land = build_land_mask(
        geo.distance_to_land, geo.shape, buffer_cells=int(astar_cfg["land_buffer_cells"])
    )
    mapping = polaris_mapping(
        {int(k): v for k, v in data_cfg["sod_to_polaris"].items()}, ice_classes, table=table
    )

    # RIO does not depend on speed, so it is computed once and reused.
    rios = {
        ic: compute_rio(
            geo.sod,
            ic,
            mapping,
            table=table,
            sic=geo.sic,
            mask_valid=geo.mask_valid,
            n_classes=N_CLASSES,
        )
        for ic in ice_classes
    }

    nominal = {
        ic: build_cost_grid(rios[ic], speed_model_from_config(routing_cfg, ic), land_mask=land)
        for ic in ice_classes
    }
    try:
        wp = choose_waypoints(geo.grid, nominal)
    except ValueError as exc:
        return {"scene_id": scene.meta.scene_id, "error": f"waypoint non selezionabili: {exc}"}

    out: dict = {"scene_id": scene.meta.scene_id, "by_class": {}}
    for ic in ice_classes:
        rows = []
        for v in speeds:
            cfg = copy.deepcopy(routing_cfg)
            cfg["vessel"]["open_water_speed_kn"] = float(v)
            cost = build_cost_grid(rios[ic], speed_model_from_config(cfg, ic), land_mask=land)
            row: dict = {"open_water_speed_kn": v}
            try:
                route = find_route(
                    cost,
                    geo.grid,
                    wp.start,
                    wp.goal,
                    connectivity=int(astar_cfg["connectivity"]),
                    heuristic_weight=float(astar_cfg["heuristic_weight"]),
                    max_expansions=int(astar_cfg["max_expansions"]),
                    rio=rios[ic],
                )
            except NoRouteFound as exc:
                rows.append(row | {"route_exists": False, "reason": exc.reason})
                continue

            base = geodesic_baseline(
                geo.grid,
                cost,
                wp.start,
                wp.goal,
                open_water_speed_kn=float(v),
                rio=rios[ic],
                land_mask=land,
                coverage=geo.mask_valid,
            )
            # compare() is the demo's own logic, reused rather than reimplemented:
            # it already refuses to report a saving against an infeasible
            # baseline (SPEC rule 2), and the point of this sweep is to measure
            # the quantity the demo actually displays.
            cmp = compare(route, base)
            row |= {
                "route_exists": True,
                "hours": route.time_hours,
                "distance_km": route.distance_km,
                "baseline_feasible": bool(base.feasible),
                "baseline_hours": (cmp["baseline"] or {}).get("time_hours"),
                "saved_percent": cmp.get("time_saved_percent"),
            }
            rows.append(row)
        out["by_class"][ic] = rows
    return out


def summarise(results: list[dict]) -> dict:
    """Spread of the quoted quantities across the swept speeds."""
    hours, pct = [], []
    for r in results:
        for rows in (r.get("by_class") or {}).values():
            h = [x["hours"] for x in rows if x.get("hours")]
            p = [x["saved_percent"] for x in rows if isinstance(x.get("saved_percent"), float)]
            # Spread within one scene+class, i.e. attributable to speed alone.
            if len(h) > 1:
                hours.append((max(h) - min(h)) / min(h))
            if len(p) > 1:
                pct.append(max(p) - min(p))
    return {
        "n_series": len(hours),
        # Absolute hours: expected to move a lot.
        "hours_relative_spread_median": float(np.median(hours)) if hours else None,
        "hours_relative_spread_max": float(max(hours)) if hours else None,
        # Percentage saved: the number we would actually quote.
        "saved_percent_spread_median_points": float(np.median(pct)) if pct else None,
        "saved_percent_spread_max_points": float(max(pct)) if pct else None,
        "n_comparable_series": len(pct),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Sensibilità alla velocità in acqua libera. P-05.")
    ap.add_argument("--config", type=Path, default=Path("configs/routing.yaml"))
    ap.add_argument("--data-config", type=Path, default=Path("configs/data.yaml"))
    ap.add_argument("--train-config", type=Path, default=None)
    ap.add_argument("--demo-config", type=Path, default=Path("configs/demo.yaml"))
    ap.add_argument("--speeds", type=float, nargs="*", default=DEFAULT_SPEEDS)
    ap.add_argument("--device", default=None)
    ap.add_argument("--out", type=Path, default=Path("artifacts/speed_sensitivity.json"))
    args = ap.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)-7s %(message)s")

    if args.train_config is None:
        picked = Path("artifacts/best_config.txt")
        args.train_config = Path(
            picked.read_text().strip() if picked.exists() else "configs/train3.yaml"
        )

    import torch

    from iceroute.model.evaluate import load_checkpoint

    data_cfg = load_config(args.data_config)
    routing_cfg = yaml.safe_load(args.config.read_text())
    train_cfg = yaml.safe_load(args.train_config.read_text())
    demo_cfg = yaml.safe_load(args.demo_config.read_text())
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    model, _ = load_checkpoint(Path(train_cfg["output"]["best_checkpoint"]), device)
    table = PolarisTable.load()

    raw_dir = Path(demo_cfg.get("source_dir", "data/raw/test"))
    scenes = [e["id"] for e in demo_cfg["scenes"]]

    log.info("Velocità sotto esame: %s nodi", ", ".join(f"{v:g}" for v in args.speeds))
    results = []
    for i, sid in enumerate(scenes, 1):
        path = raw_dir / sid
        if not path.exists():
            continue
        log.info("[%d/%d] %s", i, len(scenes), sid[:20])
        try:
            results.append(
                sweep_scene(
                    path,
                    model=model,
                    data_cfg=data_cfg,
                    routing_cfg=routing_cfg,
                    train_cfg=train_cfg,
                    table=table,
                    device=device,
                    speeds=list(args.speeds),
                )
            )
        except Exception as exc:  # noqa: BLE001 — one bad scene must not stop the sweep
            log.exception("  fallita: %s", exc)
        if device == "cuda":
            torch.cuda.empty_cache()

    agg = summarise(results)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps({"speeds_kn": args.speeds, "aggregate": agg, "per_scene": results}, indent=2)
    )

    log.info("=" * 66)
    hs = agg["hours_relative_spread_median"]
    if hs is not None:
        log.info(
            "Ore di transito: variano del %.0f%% (mediana) sull'intervallo di velocità.", 100 * hs
        )
        log.info("  → NON citabili senza una fonte per la velocità. P-05 resta su questo.")
    # A small spread means "stable" only if there is enough of it to be a
    # spread. The first version reported "0,0 punti → STABILE, usalo nel deck"
    # off two series out of fifteen: that is not stability, it is a quantity
    # that is almost never defined. Coverage is checked before dispersion.
    n_ok, n_all = agg["n_comparable_series"], agg["n_series"]
    log.info("Risparmio percentuale: definito in %d serie su %d.", n_ok, n_all)
    if n_ok < max(3, n_all // 2):
        log.info("  → NON è una metrica utilizzabile su queste scene. Le scene demo sono")
        log.info("    state scelte perché la rotta diretta è VIETATA dal ghiaccio, quindi")
        log.info("    nella maggior parte dei casi non esiste una baseline percorribile")
        log.info("    contro cui calcolare un risparmio (SPEC regola 2).")
    elif agg["saved_percent_spread_max_points"] < 5.0:
        log.info(
            "  → varia di %.1f punti al massimo: STABILE rispetto alla velocità.",
            agg["saved_percent_spread_max_points"],
        )
    else:
        log.info(
            "  → varia di %.1f punti: serve comunque la fonte.",
            agg["saved_percent_spread_max_points"],
        )

    log.info("")
    log.info("CONSEGUENZA PER IL DECK. Nessuna delle due grandezze temporali regge da sola:")
    log.info("le ore dipendono da un'assunzione senza fonte, la percentuale quasi non")
    log.info("esiste. L'affermazione difendibile è CATEGORICA e non usa la velocità:")
    log.info("«la rotta diretta è vietata da POLARIS, IceRoute ne trova una che non lo è».")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
