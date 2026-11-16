"""Run the full routing pipeline over the demo scenes. SPEC Gate 3.

    scene -> geocode -> POLARIS RIO -> cost -> A* -> baseline -> simplify

Produces, for every (scene, ice class) pair in ``data/demo/routes/``:
GeoJSON for the IceRoute track and for the baseline, transit time and distance
for both, and the minimum RIO encountered — the safety metric that says the
route is not merely faster but inside operational limits (SPEC §8.6).

Run with ``make route``.
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import yaml

from iceroute.data.loader import load_config, load_scene
from iceroute.routing.astar import NoRouteFound, find_route
from iceroute.routing.baseline import compare, geodesic_baseline
from iceroute.routing.cost import build_cost_grid, build_land_mask, speed_model_from_config
from iceroute.routing.geocode import geocode_scene
from iceroute.routing.polaris import PolarisTable, compute_rio, summarise
from iceroute.routing.simplify import simplify_route
from iceroute.routing.waypoints import choose_waypoints, waypoints_from_config

log = logging.getLogger("route")


def process_scene(
    scene_path: Path,
    data_cfg: dict,
    routing_cfg: dict,
    out_dir: Path,
    table: PolarisTable,
) -> dict:
    """Geocode one scene and route it for every configured ice class."""
    mapping = {int(k): v for k, v in data_cfg["sod_to_polaris"].items()}
    classes = list(routing_cfg["vessel"]["classes"])
    v_open = float(routing_cfg["vessel"]["open_water_speed_kn"])
    astar_cfg = routing_cfg["astar"]
    simp_cfg = routing_cfg["simplify"]

    scene = load_scene(scene_path, data_cfg)
    geo = geocode_scene(scene, resolution_m=float(routing_cfg["grid"]["resolution_m"]))

    land = build_land_mask(
        geo.distance_to_land, geo.shape, buffer_cells=int(astar_cfg["land_buffer_cells"])
    )

    rios, costs = {}, {}
    for ice_class in classes:
        rio = compute_rio(
            geo.sod, ice_class, mapping, table=table, sic=geo.sic, mask_valid=geo.mask_valid
        )
        rios[ice_class] = rio
        costs[ice_class] = build_cost_grid(
            rio, speed_model_from_config(routing_cfg, ice_class), land_mask=land
        )

    # Hand-picked waypoints win over automatic ones; see waypoints.py.
    entry = (routing_cfg.get("waypoints") or {}).get(scene.meta.scene_id)
    try:
        wp = waypoints_from_config(geo.grid, entry) if entry else choose_waypoints(geo.grid, costs)
    except ValueError as exc:
        log.warning("%s: waypoint non selezionabili — %s", scene.meta.scene_id, exc)
        return {
            "scene_id": scene.meta.scene_id,
            "region": scene.meta.region,
            "usable": False,
            "reason": str(exc),
        }

    result: dict = {
        "scene_id": scene.meta.scene_id,
        "region": scene.meta.region,
        "acquired": scene.meta.acquired.isoformat() if scene.meta.acquired else None,
        "grid": {
            "shape": list(geo.shape),
            "resolution_m": geo.grid.resolution_m,
            "crs": geo.grid.crs,
            "coverage": round(float(geo.mask_valid.mean()), 4),
        },
        "waypoints": wp.as_dict(),
        "usable": True,
        "classes": {},
    }

    routes_dir = out_dir / "routes"
    routes_dir.mkdir(parents=True, exist_ok=True)
    stem = scene.meta.scene_id.replace(".nc", "")

    for ice_class in classes:
        cost, rio = costs[ice_class], rios[ice_class]
        entry_out: dict = {"polaris": summarise(rio, ice_class)}

        try:
            t0 = time.monotonic()
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
            elapsed = time.monotonic() - t0
        except NoRouteFound as exc:
            # An expected outcome, not a failure: SPEC §8.4 wants this shown.
            log.info("  %s: nessuna rotta (%s)", ice_class, exc.reason)
            entry_out.update({"route_exists": False, "reason": exc.reason, "message": str(exc)})
            result["classes"][ice_class] = entry_out
            continue

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

        collection = {
            "type": "FeatureCollection",
            "features": [
                simple.as_geojson(
                    {
                        "kind": "iceroute",
                        "ice_class": ice_class,
                        "transit_time_hours": round(route.time_hours, 3),
                        "distance_km": round(route.distance_km, 3),
                        "min_rio_on_route": None
                        if route.min_rio is None
                        else round(route.min_rio, 2),
                    }
                ),
                base.as_geojson(),
            ],
        }
        path = routes_dir / f"{stem}__{ice_class}.geojson"
        path.write_text(json.dumps(collection, indent=1))

        entry_out.update(
            {
                "route_exists": True,
                "geojson": str(path.relative_to(out_dir.parent.parent))
                if out_dir.is_absolute()
                else str(path),
                "comparison": compare(route, base),
                "n_waypoints": len(simple),
                "simplify_warnings": simple.warnings,
                "astar_seconds": round(elapsed, 2),
                "astar_expanded": route.expanded_nodes,
            }
        )
        result["classes"][ice_class] = entry_out
        log.info(
            "  %s: %s  (%d waypoint, A* %.1fs)",
            ice_class,
            entry_out["comparison"]["headline"],
            len(simple),
            elapsed,
        )

    return result


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Geocodifica, POLARIS e A* sulle scene demo.")
    ap.add_argument("--config", type=Path, default=Path("configs/routing.yaml"))
    ap.add_argument("--data-config", type=Path, default=Path("configs/data.yaml"))
    ap.add_argument("--out", type=Path, default=Path("data/demo"))
    ap.add_argument("--raw-dir", type=Path, default=Path("data/raw/train"))
    ap.add_argument("--scenes", nargs="*", default=None, help="override delle scene demo")
    args = ap.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s  %(levelname)-7s %(message)s", datefmt="%H:%M:%S"
    )

    data_cfg = load_config(args.data_config)
    routing_cfg = yaml.safe_load(args.config.read_text())
    table = PolarisTable.load()

    provisional = table.provisional_classes()
    if provisional:
        log.warning("Risk value PROVVISORI per: %s — la demo deve dichiararlo.", provisional)

    if args.scenes:
        scenes = list(args.scenes)
    else:
        split = yaml.safe_load(Path(data_cfg.get("split_file", "configs/split.yaml")).read_text())[
            "split"
        ]
        scenes = list(split["demo"])

    present = [s for s in scenes if (args.raw_dir / s).exists()]
    if len(present) < len(scenes):
        log.warning(
            "%d scene demo non ancora scaricate: %s",
            len(scenes) - len(present),
            ", ".join(s for s in scenes if s not in present),
        )
    if not present:
        log.error("Nessuna scena demo disponibile in %s.", args.raw_dir)
        return 1

    args.out.mkdir(parents=True, exist_ok=True)
    results = []
    for i, sid in enumerate(present, 1):
        log.info("[%d/%d] %s", i, len(present), sid)
        try:
            results.append(
                process_scene(args.raw_dir / sid, data_cfg, routing_cfg, args.out, table)
            )
        except Exception as exc:  # noqa: BLE001 — one bad scene must not stop the batch
            log.exception("  fallita: %s", exc)
            results.append({"scene_id": sid, "usable": False, "reason": str(exc)})

    index_path = args.out / "routes_index.json"
    index_path.write_text(
        json.dumps(
            {
                "open_water_speed_kn": routing_cfg["vessel"]["open_water_speed_kn"],
                "open_water_speed_is_provisional": True,
                "grid_resolution_m": routing_cfg["grid"]["resolution_m"],
                "crs": routing_cfg["grid"]["crs"],
                "ice_classes": routing_cfg["vessel"]["classes"],
                "polaris_source": "IMO MSC.1/Circ.1519 Tabelle 1.1, 1.2, 1.3",
                "scenes": results,
            },
            indent=2,
        )
    )

    usable = [r for r in results if r.get("usable")]
    n_routes = sum(1 for r in usable for c in r["classes"].values() if c.get("route_exists"))
    log.info("=" * 66)
    log.info("Scene utilizzabili: %d/%d", len(usable), len(present))
    log.info(
        "Rotte trovate: %d su %d combinazioni",
        n_routes,
        len(usable) * len(routing_cfg["vessel"]["classes"]),
    )
    log.info("Indice in %s", index_path)
    return 0 if usable else 1


if __name__ == "__main__":
    raise SystemExit(main())
