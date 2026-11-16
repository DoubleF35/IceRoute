"""Risk-aware decision rules: trading map accuracy for navigational safety.

``argmax`` minimises classification error, and classification error is
symmetric: calling thick ice thin costs the same as calling thin ice thick.
POLARIS is not symmetric. The first error can route a vessel into ice it is not
certified for; the second adds a detour. On the official test set the deployed
model puts 5.6% of pixels (41% in the worst scene) in a *less* restrictive
POLARIS level than the ice chart (``docs/model-card.md``).

This module measures what a different decision rule buys, on the same model,
with no retraining. Two rules are compared against argmax:

* **relative** (``safety_bias.safety_biased_labels``): among classes whose
  probability is at least ``tau`` times the top one, take the most dangerous.
  Scale-free, but ``tau`` has no probabilistic meaning.
* **chance-constrained** (new here): treat a cell as the most dangerous class
  whose *calibrated* probability is at least ``alpha``. With calibrated
  probabilities ``alpha`` reads as a risk budget: "plan as if the ice were
  thick wherever there is at least an alpha chance that it is". This is why
  calibration (``iceroute.model.calibration``) comes first: on raw, confident
  probabilities ``alpha = 0.05`` would mean something closer to 0.01.

For each rule and threshold it reports mIoU, ice/water accuracy, and the RIO
error split by direction (unsafe vs. over-cautious) for every ship class, so an
operating point can be argued from the curve rather than tuned until a number
looks good. Both rules are one comparison per pixel at inference time, so they
are available on an on-board accelerator too.

Usage
-----
    python -m iceroute.model.risk          # needs artifacts/calibration.json

Writes ``artifacts/risk_curve.json`` and ``docs/img/risk_curve.png``.
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)

ALPHAS = [0.5, 0.3, 0.2, 0.1, 0.05, 0.02]
TAUS = [0.9, 0.7, 0.5, 0.3, 0.1]


def chance_constrained_labels(proba: np.ndarray, riv: np.ndarray, alpha: float) -> np.ndarray:
    """The most dangerous class whose probability is at least ``alpha``.

    Parameters
    ----------
    proba
        ``(n_classes, H, W)`` probabilities, ideally calibrated.
    riv
        POLARIS risk index value per class for the vessel (lower = more
        dangerous), from ``safety_bias.danger_order``.
    alpha
        Risk budget in (0, 1]. Where no class reaches ``alpha`` (possible only
        when ``alpha > 1 / n_classes``) the pixel keeps its argmax label, so
        the rule can only ever be *more* cautious than argmax, never less.
    """
    if not 0.0 < alpha <= 1.0:
        raise ValueError(f"alpha must lie in (0, 1], not {alpha}")
    if proba.ndim != 3:
        raise ValueError(f"Expected (n_classes, H, W), got {proba.shape}")
    if len(riv) != proba.shape[0]:
        raise ValueError(f"{len(riv)} risk values for {proba.shape[0]} classes")

    argmax = proba.argmax(axis=0)
    eligible = proba >= alpha
    # the argmax class is always eligible: it is the fallback, and a class the
    # model prefers can never be excluded by a budget meant to add caution
    np.put_along_axis(eligible, argmax[None], True, axis=0)
    order = np.argsort(riv, kind="stable")  # most dangerous first
    first = eligible[order].argmax(axis=0)
    return order[first].astype("uint8")


# ---------------------------------------------------------------------------
# Accumulating a curve over many scenes
# ---------------------------------------------------------------------------


@dataclass
class _Point:
    """Running totals for one (rule, threshold) across scenes."""

    rule: str
    threshold: float
    confusion: np.ndarray
    rio: dict[str, dict[str, float]] = field(default_factory=dict)  # class -> sums

    def add_rio(self, ice_class: str, n: int, err) -> None:
        acc = self.rio.setdefault(
            ice_class, {"n": 0, "unsafe": 0.0, "overcautious": 0.0, "agree": 0.0, "mae": 0.0}
        )
        acc["n"] += n
        acc["unsafe"] += err.unsafe_fraction * n
        acc["overcautious"] += err.overcautious_fraction * n
        acc["agree"] += err.exact_level_agreement * n
        acc["mae"] += err.rio_mae * n

    def as_dict(self) -> dict:
        from iceroute.model.metrics import ConfusionMatrix

        cm = ConfusionMatrix(self.confusion.shape[0])
        cm.matrix = self.confusion.copy()
        out = {
            "rule": self.rule,
            "threshold": self.threshold,
            "miou": round(cm.miou(), 4),
            "binary_accuracy": round(cm.binary_accuracy(), 4),
            "per_ship_class": {},
        }
        for ic, a in self.rio.items():
            n = max(a["n"], 1)
            out["per_ship_class"][ic] = {
                "unsafe_fraction": round(a["unsafe"] / n, 5),
                "overcautious_fraction": round(a["overcautious"] / n, 5),
                "operational_agreement": round(a["agree"] / n, 4),
                "rio_mae": round(a["mae"] / n, 3),
            }
        return out


def _confusion(truth: np.ndarray, pred: np.ndarray, n_classes: int, ignore: int) -> np.ndarray:
    valid = truth != ignore
    idx = truth[valid].astype(np.int64) * n_classes + pred[valid].astype(np.int64)
    return np.bincount(idx, minlength=n_classes * n_classes).reshape(n_classes, n_classes)


def plot_curve(points: list[dict], ice_class: str, path: Path) -> bool:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:  # pragma: no cover
        return False

    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    styles = {
        "argmax": dict(color="#1a202c", marker="*", s=180, zorder=5),
        "relative": dict(color="#718096", marker="o"),
        "chance": dict(color="#2b6cb0", marker="s"),
    }
    labels = {
        "argmax": "argmax (deployed)",
        "relative": "relative rule (τ)",
        "chance": "chance-constrained (α, calibrated)",
    }
    for rule in ("relative", "chance", "argmax"):
        pts = [p for p in points if p["rule"] == rule]
        if not pts:
            continue
        x = [100 * p["per_ship_class"][ice_class]["unsafe_fraction"] for p in pts]
        y = [p["miou"] for p in pts]
        st = dict(styles[rule])
        size = st.pop("s", 40)
        if rule != "argmax":
            ax.plot(x, y, color=st["color"], lw=1.2, alpha=0.7)
        ax.scatter(x, y, s=size, label=labels[rule], **st)
        if rule != "argmax":
            sym = "τ" if rule == "relative" else "α"
            for p, xi, yi in zip(pts, x, y, strict=True):
                ax.annotate(
                    f"{sym}={p['threshold']:g}",
                    (xi, yi),
                    textcoords="offset points",
                    xytext=(5, 4),
                    fontsize=7,
                    color=st["color"],
                )
    ax.set_xlabel(f"unsafe pixels for a {ice_class} vessel (%)  ← safer")
    ax.set_ylabel("mIoU (3 classes)")
    ax.set_title("Safety vs. map accuracy on the official test set, same model, no retraining")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return True


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Safety/accuracy curves for risk-aware decision rules."
    )
    ap.add_argument("--config", type=Path, default=Path("configs/train3_lovasz.yaml"))
    ap.add_argument("--data-config", type=Path, default=Path("configs/data.yaml"))
    ap.add_argument("--routing-config", type=Path, default=Path("configs/routing.yaml"))
    ap.add_argument("--calibration", type=Path, default=Path("artifacts/calibration.json"))
    ap.add_argument("--test-dir", type=Path, default=Path("data/raw/test"))
    ap.add_argument("--max-scenes", type=int, default=20)
    ap.add_argument("--device", default=None)
    ap.add_argument("--out", type=Path, default=Path("artifacts/risk_curve.json"))
    ap.add_argument("--figure", type=Path, default=Path("docs/img/risk_curve.png"))
    args = ap.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)-7s %(message)s")

    import torch
    import yaml

    from iceroute.data.collapse import collapse, polaris_mapping
    from iceroute.data.loader import load_config, load_scene
    from iceroute.demo.precompute import resolve_labels
    from iceroute.model.calibration import apply_temperature
    from iceroute.model.evaluate import load_checkpoint
    from iceroute.model.inference import predict_scene
    from iceroute.model.rio_error import IGNORE_INDEX, rio_error
    from iceroute.model.safety_bias import danger_order, safety_biased_labels
    from iceroute.routing.polaris import PolarisTable

    if not args.calibration.exists():
        log.error("%s missing: run `make calibrate` first.", args.calibration)
        return 1
    temperature = float(json.loads(args.calibration.read_text())["temperature"])

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    data_cfg = load_config(args.data_config)
    train_cfg = yaml.safe_load(args.config.read_text())
    routing_cfg = yaml.safe_load(args.routing_config.read_text())
    ice_classes = list(routing_cfg["vessel"]["classes"])
    n_classes = int(train_cfg["model"]["n_classes"])
    channels = list(train_cfg["data"].get("channels") or ["hh", "hv", "incidence_angle"])
    eval_cfg = train_cfg.get("eval", {})
    table = PolarisTable.load()
    base_map = {int(k): v for k, v in data_cfg["sod_to_polaris"].items()}
    mapping = polaris_mapping(base_map, ice_classes)
    model, _ = load_checkpoint(Path(train_cfg["output"]["best_checkpoint"]), device)

    paths = [p for p in sorted(args.test_dir.glob("*_prep.nc")) if resolve_labels(p) is not None][
        : args.max_scenes
    ]
    if not paths:
        log.error("No labelled test scenes in %s", args.test_dir)
        return 1

    # One danger ordering per ship class; the label map is then scored for
    # *that* vessel, because "most dangerous" depends on the vessel.
    rivs = {ic: danger_order(ic, mapping, table=table, n_classes=n_classes) for ic in ice_classes}
    grid = [("argmax", 1.0)] + [("relative", t) for t in TAUS] + [("chance", a) for a in ALPHAS]
    points: dict[tuple[str, float, str], _Point] = {}

    for i, path in enumerate(paths, 1):
        t0 = time.monotonic()
        scene = load_scene(path, data_cfg, labels_from=resolve_labels(path))
        _, proba, _ = predict_scene(
            model,
            scene.channels(channels),
            n_classes=n_classes,
            tile_size=int(eval_cfg.get("tile_size", 256)),
            overlap=int(eval_cfg.get("tile_overlap", 32)),
            device=device,
            return_proba=True,
            amp=device == "cuda",
        )
        proba = apply_temperature(proba, temperature, axis=0).astype("float32")
        truth = collapse(scene.sod.astype("uint8")).astype("int64")
        truth[~scene.mask_valid] = IGNORE_INDEX
        n_valid = int((truth != IGNORE_INDEX).sum())

        for ic in ice_classes:
            riv = rivs[ic]
            for rule, thr in grid:
                if rule == "argmax":
                    pred = proba.argmax(axis=0).astype("uint8")
                elif rule == "relative":
                    pred = safety_biased_labels(proba, riv, thr)
                else:
                    pred = chance_constrained_labels(proba, riv, thr)
                key = (rule, thr, ic)
                pt = points.get(key)
                if pt is None:
                    pt = points[key] = _Point(rule, thr, np.zeros((n_classes, n_classes), np.int64))
                pt.confusion += _confusion(truth, pred, n_classes, IGNORE_INDEX)
                try:
                    err = rio_error(
                        truth,
                        pred,
                        ic,
                        mapping,
                        table=table,
                        sic=scene.sic,
                        mask_valid=scene.mask_valid,
                        n_classes=n_classes,
                    )
                    pt.add_rio(ic, err.n_pixels or n_valid, err)
                except ValueError as exc:
                    log.warning("  %s %s %s=%g: %s", path.name[:15], ic, rule, thr, exc)
        log.info("[%2d/%d] %s  (%.0f s)", i, len(paths), path.name[:24], time.monotonic() - t0)
        del scene, proba, truth
        if device == "cuda":
            torch.cuda.empty_cache()

    # Merge per-ship-class accumulators: mIoU is per vessel too, because the
    # biased label map depends on the vessel's danger ordering.
    results = {}
    for ic in ice_classes:
        rows = []
        for rule, thr in grid:
            pt = points.get((rule, thr, ic))
            if pt is not None:
                rows.append(pt.as_dict())
        results[ic] = rows

    out = {
        "config": str(args.config),
        "temperature": temperature,
        "split": "official AI4Arctic test set",
        "n_scenes": len(paths),
        "weighting": "pixel-weighted over all scenes",
        "rules": {
            "argmax": "plain argmax (deployed)",
            "relative": "most dangerous class with p >= tau * p_max (safety_bias.py)",
            "chance": "most dangerous class with calibrated p >= alpha (risk.py)",
        },
        "curves": results,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2))
    strictest = ice_classes[-1]
    plot_curve(results[strictest], strictest, args.figure)

    log.info("=" * 72)
    for ic in ice_classes:
        log.info("%s", ic)
        log.info("  %-10s %6s %8s %9s %10s", "rule", "thr", "mIoU", "unsafe", "cautious")
        for r in results[ic]:
            s = r["per_ship_class"].get(ic, {})
            log.info(
                "  %-10s %6.2f %8.3f %8.2f%% %9.2f%%",
                r["rule"],
                r["threshold"],
                r["miou"],
                100 * s.get("unsafe_fraction", float("nan")),
                100 * s.get("overcautious_fraction", float("nan")),
            )
    log.info("Written %s and %s", args.out, args.figure)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
