"""Evaluate the trained model. SPEC §7.3 and §7.4.

Produces the numbers the pitch is allowed to quote, and the one comparison that
decides whether the U-Net deserves to exist at all.

The honest baseline
-------------------
Sea ice is bright in HV and open water is dark. A single threshold on that one
channel separates them surprisingly well — so any claim that a neural network
is needed has to be measured against it, not asserted. This module fits that
threshold two ways:

* **fitted** — chosen on the training split, applied to the evaluation split.
  This is what an engineer could actually deploy.
* **oracle** — the best threshold *in hindsight on the evaluation split
  itself*. It is not deployable and it is not fair; that is the point. It is
  the strongest possible version of the cheap baseline, and beating it is the
  only result worth putting on a slide.

The threshold baseline is binary by construction: it cannot distinguish young
ice from old ice, and old ice is precisely what POLARIS penalises hardest. So
the comparison is reported on *both* axes — ice/water, where the threshold is
competitive, and the 6-class mIoU, where it structurally cannot compete. Saying
only the second would be selective quoting.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import yaml

from iceroute.data.loader import load_config, load_scene
from iceroute.data.patches import CHANNELS, PatchDataset
from iceroute.model.inference import predict_scene
from iceroute.model.metrics import IGNORE_INDEX, ConfusionMatrix
from iceroute.model.rio_error import evaluate_all_classes
from iceroute.model.unet import build_model

log = logging.getLogger("evaluate")

WATER_CLASS = 0


# ---------------------------------------------------------------------------
# Checkpoint
# ---------------------------------------------------------------------------


def load_checkpoint(path: Path, device: str = "cpu") -> tuple[torch.nn.Module, dict]:
    """Load a training checkpoint and rebuild the model it describes.

    The config travels inside the checkpoint, so evaluation cannot silently use
    a different architecture from the one that was trained — a mismatch would
    surface as a confusing state_dict error at best and wrong numbers at worst.
    """
    ckpt = torch.load(path, map_location=device, weights_only=False)
    if "model" not in ckpt or "config" not in ckpt:
        raise ValueError(
            f"{path} non sembra un checkpoint di train.py: mancano 'model' o 'config'."
        )
    model = build_model(ckpt["config"]["model"])
    model.load_state_dict(ckpt["model"])
    model.eval().to(device)
    log.info(
        "checkpoint %s: epoca %s, val_miou %.4f, %s parametri",
        path.name, ckpt.get("epoch", "?"), ckpt.get("val_miou", float("nan")),
        f"{model.param_count():,}",
    )
    return model, ckpt


# ---------------------------------------------------------------------------
# Patch-level evaluation
# ---------------------------------------------------------------------------


@torch.no_grad()
def evaluate_patches(
    model: torch.nn.Module,
    dataset: PatchDataset,
    *,
    n_classes: int,
    batch_size: int = 16,
    device: str = "cpu",
    amp: bool = True,
) -> ConfusionMatrix:
    """Accumulate a confusion matrix over a patch dataset."""
    from torch.utils.data import DataLoader

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    cm = ConfusionMatrix(n_classes)
    dev = torch.device(device)
    use_amp = bool(amp) and dev.type == "cuda"

    for batch in loader:
        x, y = batch[0].to(dev), batch[1]
        with torch.autocast("cuda", dtype=torch.float16, enabled=use_amp):
            logits = model(x)
        cm.update(y, logits.float().argmax(dim=1).cpu())
    return cm


# ---------------------------------------------------------------------------
# The HV threshold baseline
# ---------------------------------------------------------------------------


@dataclass
class ThresholdBaseline:
    """A single-threshold ice/water classifier and how well it does."""

    threshold: float
    binary_accuracy: float
    binary_iou: float
    miou_6class: float
    fitted_on: str
    n_pixels: int

    def as_dict(self) -> dict:
        return {
            "threshold_hv_scaled": round(self.threshold, 5),
            "binary_accuracy": round(self.binary_accuracy, 4),
            "binary_iou": round(self.binary_iou, 4),
            "miou_6class": round(self.miou_6class, 4),
            "fitted_on": self.fitted_on,
            "n_pixels": self.n_pixels,
        }


def _binary_accuracy_at(hv: np.ndarray, is_ice: np.ndarray, thresholds: np.ndarray) -> np.ndarray:
    """Accuracy of ``hv > t`` against ``is_ice``, vectorised over thresholds.

    Done with two sorted cumulative counts rather than a loop over thresholds:
    at tens of millions of pixels the loop would take minutes.
    """
    order = np.argsort(hv, kind="stable")
    hv_sorted = hv[order]
    ice_sorted = is_ice[order].astype("int64")

    total = ice_sorted.size
    n_ice = int(ice_sorted.sum())
    # For a cut at index k: pixels [0, k) are predicted water, [k, n) ice.
    cum_ice = np.concatenate([[0], np.cumsum(ice_sorted)])
    k = np.searchsorted(hv_sorted, thresholds, side="right")
    correct_water = k - cum_ice[k]              # water-labelled pixels below the cut
    correct_ice = n_ice - cum_ice[k]            # ice-labelled pixels above it
    return (correct_water + correct_ice) / max(total, 1)


def fit_hv_threshold(
    hv: np.ndarray, sod: np.ndarray, *, n_candidates: int = 512
) -> tuple[float, float]:
    """Threshold on HV maximising ice/water accuracy. Returns (threshold, accuracy).

    Candidates are quantiles of the observed HV rather than a uniform grid: the
    channel is standard-scaled, so its useful range is data-dependent and a
    fixed grid would waste most of its points outside it.
    """
    keep = sod != IGNORE_INDEX
    hv, sod = hv[keep].astype("float64"), sod[keep]
    if hv.size == 0:
        raise ValueError("Nessun pixel valido per calibrare la soglia HV.")

    is_ice = sod != WATER_CLASS
    if is_ice.all() or not is_ice.any():
        raise ValueError(
            "I pixel validi sono tutti ghiaccio o tutta acqua: la soglia non è definita."
        )

    qs = np.linspace(0.001, 0.999, n_candidates)
    cands = np.unique(np.quantile(hv, qs))
    acc = _binary_accuracy_at(hv, is_ice, cands)
    best = int(np.argmax(acc))
    return float(cands[best]), float(acc[best])


def score_hv_threshold(
    hv: np.ndarray, sod: np.ndarray, threshold: float, *, n_classes: int, fitted_on: str
) -> ThresholdBaseline:
    """Score a fixed HV threshold on labelled pixels.

    The 6-class mIoU is computed by giving the baseline its best possible shot:
    every pixel it calls "ice" is assigned the *most frequent* ice class in the
    ground truth. That is generous — it uses label statistics no threshold has
    access to — and it still cannot exceed 2 of 6 classes. Handicapping it
    further would prove nothing.
    """
    keep = sod != IGNORE_INDEX
    hv, sod = hv[keep], sod[keep]
    if hv.size == 0:
        raise ValueError("Nessun pixel valido per valutare la soglia HV.")

    ice_labels = sod[sod != WATER_CLASS]
    majority_ice = (
        int(np.bincount(ice_labels, minlength=n_classes).argmax())
        if ice_labels.size
        else WATER_CLASS
    )

    pred = np.where(hv > threshold, majority_ice, WATER_CLASS).astype("int64")
    cm = ConfusionMatrix(n_classes)
    cm.update(sod, pred)

    return ThresholdBaseline(
        threshold=threshold,
        binary_accuracy=cm.binary_accuracy(WATER_CLASS),
        binary_iou=cm.binary_iou(WATER_CLASS),
        miou_6class=cm.miou(),
        fitted_on=fitted_on,
        n_pixels=int(hv.size),
    )


def collect_patch_pixels(
    dataset: PatchDataset, *, hv_channel: str = "hv", max_pixels: int = 40_000_000, seed: int = 0
) -> tuple[np.ndarray, np.ndarray]:
    """Flatten a patch dataset into (hv, sod) arrays, subsampled if huge."""
    if hv_channel not in dataset.channels:
        raise ValueError(
            f"Il canale '{hv_channel}' non è nel dataset ({dataset.channels}): "
            f"la baseline a soglia non è calcolabile."
        )
    ch = dataset.channels.index(hv_channel)

    hv_parts, sod_parts, total = [], [], 0
    for i in range(len(dataset)):
        x, y, m = dataset[i]
        hv_parts.append(np.asarray(x[ch]).ravel())
        lab = np.asarray(y).ravel().astype("int64")
        lab[~np.asarray(m).ravel()] = IGNORE_INDEX
        sod_parts.append(lab)
        total += lab.size
        if total >= max_pixels:
            log.info("baseline HV: campionamento fermato a %d px", total)
            break

    hv = np.concatenate(hv_parts)
    sod = np.concatenate(sod_parts)
    if hv.size > max_pixels:
        rng = np.random.default_rng(seed)
        idx = rng.choice(hv.size, size=max_pixels, replace=False)
        hv, sod = hv[idx], sod[idx]
    return hv, sod


# ---------------------------------------------------------------------------
# Whole-scene evaluation
# ---------------------------------------------------------------------------


def evaluate_scene(
    model: torch.nn.Module,
    scene_path: Path,
    data_cfg: dict,
    *,
    n_classes: int,
    channels: list[str],
    tile_size: int = 256,
    overlap: int = 32,
    batch_size: int = 8,
    device: str = "cpu",
    ice_classes: list[str] | None = None,
) -> tuple[ConfusionMatrix, dict]:
    """Tiled inference over a whole scene, scored against its ice chart.

    Scores it twice: as a classification (mIoU, the SPEC §7.4 gate) and as a
    POLARIS input (RIO error, what the route actually depends on). Neither
    replaces the other — see ``rio_error``.
    """
    scene = load_scene(scene_path, data_cfg)
    x = scene.channels(channels)

    labels, _, stats = predict_scene(
        model, x, n_classes=n_classes, tile_size=tile_size, overlap=overlap,
        batch_size=batch_size, device=device,
    )

    truth = scene.sod.astype("int64").copy()
    truth[~scene.mask_valid] = IGNORE_INDEX
    cm = ConfusionMatrix(n_classes)
    cm.update(truth, labels)

    info = stats.as_dict()
    info["scene_id"] = scene.meta.scene_id
    info["region"] = scene.meta.region
    info["valid_fraction"] = round(scene.valid_fraction, 4)

    if ice_classes:
        try:
            info["rio"] = evaluate_all_classes(
                truth, labels, ice_classes,
                {int(k): v for k, v in data_cfg["sod_to_polaris"].items()},
                sic=scene.sic, mask_valid=scene.mask_valid, n_classes=n_classes,
            )
        except (ValueError, KeyError) as exc:
            log.warning("  errore RIO non calcolabile: %s", exc)
            info["rio"] = {"error": str(exc)}

    return cm, info


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def write_confusion_csv(cm: ConfusionMatrix, path: Path, class_names: dict[int, str]) -> None:
    """Raw counts, so anyone can recompute every metric we report."""
    path.parent.mkdir(parents=True, exist_ok=True)
    names = [class_names.get(i, str(i)) for i in range(cm.n_classes)]
    with path.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["verita\\predizione", *names, "supporto"])
        for i, row in enumerate(cm.matrix):
            w.writerow([names[i], *row.tolist(), int(row.sum())])


def plot_confusion(cm: ConfusionMatrix, path: Path, class_names: dict[int, str]) -> bool:
    """Row-normalised confusion matrix as a PNG. Returns False if unplottable."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        log.warning("matplotlib non disponibile: salto la figura.")
        return False

    norm = cm.normalized()
    names = [class_names.get(i, str(i)) for i in range(cm.n_classes)]
    support = cm.support()

    fig, ax = plt.subplots(figsize=(7.2, 6.2), dpi=150)
    im = ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(cm.n_classes), names, rotation=40, ha="right")
    ax.set_yticks(range(cm.n_classes), names)
    ax.set_xlabel("Predizione")
    ax.set_ylabel("Verità (ice chart)")
    ax.set_title(
        f"Matrice di confusione — mIoU {cm.miou():.3f} "
        f"({cm.n_classes_scored()}/{cm.n_classes} classi con ground truth)"
    )

    for i in range(cm.n_classes):
        for j in range(cm.n_classes):
            if support[i] == 0:
                continue
            v = norm[i, j]
            ax.text(
                j, i, f"{v:.2f}", ha="center", va="center",
                color="white" if v > 0.55 else "black", fontsize=8,
            )
    # A class with no ground truth has an empty row; say so instead of showing
    # a band of zeros that reads as "the model got everything wrong here".
    for i in range(cm.n_classes):
        if support[i] == 0:
            ax.text(
                cm.n_classes / 2 - 0.5, i, "nessun pixel di verità in questo split",
                ha="center", va="center", fontsize=7, style="italic", color="0.35",
            )

    fig.colorbar(im, ax=ax, label="frazione della classe vera")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    return True


def threshold_verdict(model_cm: ConfusionMatrix, oracle: ThresholdBaseline) -> dict:
    """Does the U-Net actually beat a threshold? SPEC §7.4 wants this stated."""
    d_bin = model_cm.binary_accuracy(WATER_CLASS) - oracle.binary_accuracy
    d_miou = model_cm.miou() - oracle.miou_6class
    beats_binary = d_bin > 0

    if beats_binary:
        verdict = (
            f"La U-Net batte la soglia ottimale anche su ghiaccio/acqua "
            f"({d_bin:+.3f}) oltre che sulle 6 classi ({d_miou:+.3f})."
        )
    else:
        # The defensible position when the threshold wins on binary. It usually
        # does, and pretending otherwise is the fastest way to lose the room.
        verdict = (
            f"Su ghiaccio/acqua la soglia ottimale è competitiva ({d_bin:+.3f}): "
            f"è un problema facile. Il valore della U-Net è nelle 6 classi "
            f"({d_miou:+.3f}), che è ciò che POLARIS richiede — una soglia non "
            f"distingue young ice da old ice, e il RIO fra le due cambia di segno."
        )

    return {
        "delta_binary_accuracy": round(d_bin, 4),
        "delta_miou": round(d_miou, 4),
        "unet_beats_threshold_on_binary": bool(beats_binary),
        "unet_beats_threshold_on_miou": bool(d_miou > 0),
        "verdict": verdict,
    }


def report_bias(split: str) -> str:
    """State plainly how the reported number is biased. SPEC rule 2.

    The frozen split (SPEC §5.3) has no held-out test set, so there is no
    unbiased estimate available and pretending otherwise would be the exact
    failure mode rule 2 exists to prevent. Both options have a named defect:

    * ``val`` — early stopping picked the checkpoint that maximises val mIoU, so
      this number is the *best epoch as judged by this very split*. Optimistic
      by an unknown amount, typically a point or two.
    * ``demo`` — never used for training or model selection, so no selection
      bias. But the scenes were hand-picked to be visually convincing, which is
      its own bias and pushes in the same direction.

    Quote the val number, with this caveat attached.

    The real fix is already downloading: the AI4Arctic challenge ships an
    official held-out test set (20 scenes plus 20 ``_prep_reference.nc`` label
    files, 2.53 GB) into ``data/raw/test/``. That is an unbiased estimate and it
    is what the pitch should quote. It is not wired up here yet on purpose —
    the labels live in a separate reference file whose variable names have not
    been verified on disk, and SPEC §5.4 is explicit about not trusting
    documented names. Run ``inspect.py`` on a reference file first.
    """
    if split == "val":
        return (
            "Stima OTTIMISTICA: l'early stopping ha scelto il checkpoint che massimizza "
            "il mIoU su questo stesso split (SPEC §5.3 non prevede un test set). "
            "Non è una stima out-of-sample."
        )
    return (
        "Split demo: mai usato per training né per selezione del modello, quindi nessun "
        "bias di selezione — ma le scene sono state scelte a mano perché visivamente "
        "convincenti, che è un bias nella stessa direzione."
    )


def check_thresholds(cm: ConfusionMatrix, eval_cfg: dict) -> list[str]:
    """Compare against the SPEC §7.4 acceptance thresholds."""
    warns = []
    miou, binacc = cm.miou(), cm.binary_accuracy(WATER_CLASS)
    if np.isfinite(miou) and miou < float(eval_cfg.get("min_miou", 0.60)):
        warns.append(
            f"mIoU {miou:.3f} sotto la soglia SPEC §7.4 di "
            f"{eval_cfg['min_miou']}: valutare il ripiego al problema binario."
        )
    if np.isfinite(binacc) and binacc < float(eval_cfg.get("min_binary_accuracy", 0.90)):
        warns.append(
            f"Accuratezza ghiaccio/acqua {binacc:.3f} sotto la soglia SPEC §7.4 di "
            f"{eval_cfg['min_binary_accuracy']}."
        )
    return warns


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Valuta il modello: patch, scene intere, baseline.")
    ap.add_argument("--config", type=Path, default=Path("configs/train.yaml"))
    ap.add_argument("--data-config", type=Path, default=Path("configs/data.yaml"))
    ap.add_argument("--checkpoint", type=Path, default=None)
    # SPEC §5.3 freezes three splits: train, val, demo. There is no test split,
    # so `val` is the only non-training data available for metrics — and it is
    # the split early stopping selected the checkpoint on. See report_bias().
    ap.add_argument("--split", default="val", choices=["val", "demo"])
    ap.add_argument("--out", type=Path, default=Path("artifacts"))
    ap.add_argument("--raw-dir", type=Path, default=Path("data/raw/train"))
    ap.add_argument("--scenes", type=int, default=3, help="quante scene intere valutare")
    ap.add_argument("--device", default=None)
    ap.add_argument("--skip-baseline", action="store_true")
    ap.add_argument(
        "--ice-classes", nargs="*", default=["PC6", "PC7", "IA"],
        help="classi di nave per cui misurare l'errore su RIO",
    )
    args = ap.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s  %(levelname)-7s %(message)s", datefmt="%H:%M:%S"
    )

    cfg = yaml.safe_load(args.config.read_text())
    data_cfg = load_config(args.data_config)
    n_classes = int(cfg["model"]["n_classes"])
    eval_cfg = cfg.get("eval", {})
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    ckpt_path = args.checkpoint or Path(cfg["output"]["best_checkpoint"])
    if not ckpt_path.exists():
        log.error("Checkpoint assente: %s. Lancia prima 'make train'.", ckpt_path)
        return 1
    model, ckpt = load_checkpoint(ckpt_path, device)

    class_names = ckpt.get("class_names") or {
        i: v["name"] for i, v in data_cfg["sod_classes"].items() if i != IGNORE_INDEX
    }
    class_names = {int(k): v for k, v in class_names.items()}

    split_file = Path(data_cfg.get("split_file", "configs/split.yaml"))
    split = yaml.safe_load(split_file.read_text())["split"]
    channels = list(cfg["data"].get("channels") or CHANNELS[:3])
    cache_dir = Path(cfg["data"]["cache_dir"])

    report: dict = {
        "checkpoint": str(ckpt_path),
        "checkpoint_epoch": ckpt.get("epoch"),
        "split": args.split,
        "split_bias": report_bias(args.split),
        "device": device,
        "n_parameters": model.param_count(),
        "channels": channels,
    }
    log.warning("%s", report["split_bias"])

    # -- patches ------------------------------------------------------------
    log.info("Valutazione su patch (%s)…", args.split)
    ds = PatchDataset(cache_dir, split[args.split], channels=channels, augment=False)
    cm = evaluate_patches(
        model, ds, n_classes=n_classes, batch_size=int(eval_cfg.get("batch_size", 8)),
        device=device,
    )
    log.info("\n%s", cm.format_table(class_names))
    report["patches"] = cm.as_dict(class_names)
    report["patches"]["n_patches"] = len(ds)

    write_confusion_csv(cm, args.out / "figures" / f"confusion_{args.split}.csv", class_names)
    plot_confusion(cm, args.out / "figures" / f"confusion_{args.split}.png", class_names)

    # -- HV threshold baseline ---------------------------------------------
    if not args.skip_baseline:
        log.info("Baseline a soglia su HV…")
        try:
            hv_tr, sod_tr = collect_patch_pixels(
                PatchDataset(cache_dir, split["train"], channels=channels, augment=False)
            )
            thr_fit, _ = fit_hv_threshold(hv_tr, sod_tr)
            del hv_tr, sod_tr

            hv_ev, sod_ev = collect_patch_pixels(ds)
            fitted = score_hv_threshold(
                hv_ev, sod_ev, thr_fit, n_classes=n_classes, fitted_on="train"
            )
            thr_oracle, _ = fit_hv_threshold(hv_ev, sod_ev)
            oracle = score_hv_threshold(
                hv_ev, sod_ev, thr_oracle, n_classes=n_classes, fitted_on=args.split
            )

            report["baseline_hv_threshold"] = {
                "fitted": fitted.as_dict(),
                "oracle": oracle.as_dict(),
                "comparison": threshold_verdict(cm, oracle),
            }
            log.info(
                "  soglia fitted %.3f → gh/acqua %.4f · oracle %.3f → %.4f | U-Net %.4f",
                fitted.threshold, fitted.binary_accuracy,
                oracle.threshold, oracle.binary_accuracy, cm.binary_accuracy(),
            )
            log.info("  %s", report["baseline_hv_threshold"]["comparison"]["verdict"])
        except ValueError as exc:
            log.warning("Baseline a soglia non calcolabile: %s", exc)
            report["baseline_hv_threshold"] = {"error": str(exc)}

    # -- whole scenes -------------------------------------------------------
    scene_ids = [s for s in split[args.split] if (args.raw_dir / s).exists()][: args.scenes]
    if scene_ids:
        log.info("Inferenza a scena intera su %d scene…", len(scene_ids))
        scene_cm = ConfusionMatrix(n_classes)
        per_scene = []
        ice_classes = list(args.ice_classes)
        for sid in scene_ids:
            one, info = evaluate_scene(
                model, args.raw_dir / sid, data_cfg, n_classes=n_classes, channels=channels,
                tile_size=int(eval_cfg.get("tile_size", 256)),
                overlap=int(eval_cfg.get("tile_overlap", 32)),
                batch_size=int(eval_cfg.get("batch_size", 8)), device=device,
                ice_classes=ice_classes,
            )
            scene_cm.matrix += one.matrix
            info["miou"] = one.miou()
            info["binary_accuracy"] = one.binary_accuracy()
            per_scene.append(info)
            log.info("  %s: mIoU %.4f, %.1fs", sid, one.miou(), info["seconds"])

            rio = info.get("rio", {})
            for ic, r in (rio.get("per_ice_class") or {}).items():
                if "error" not in r:
                    log.info(
                        "    %s  RIO MAE %.2f · decisione corretta %.1f%% · "
                        "PERMISSIVA %.2f%%",
                        ic, r["rio_mae"], 100 * r["operational_agreement"],
                        100 * r["unsafe_fraction"],
                    )

        # The worst unsafe fraction across scenes and vessel classes. This is
        # the safety number: pixels the model calls navigable that POLARIS
        # forbids. It is a maximum, not a mean — averaging away the worst scene
        # is how a fleet ends up trusting a chart that fails in one region.
        unsafe = [
            r["unsafe_fraction"]
            for s in per_scene
            for r in (s.get("rio", {}).get("per_ice_class") or {}).values()
            if "error" not in r
        ]
        report["scenes"] = {
            "aggregate": scene_cm.as_dict(class_names),
            "per_scene": per_scene,
            "n_scenes": len(scene_ids),
            "worst_unsafe_fraction": round(max(unsafe), 5) if unsafe else None,
        }
        if unsafe:
            log.warning(
                "Frazione PERMISSIVA peggiore su tutte le scene e classi di nave: "
                "%.2f%%. Sono i pixel dove una rotta passerebbe in ghiaccio che "
                "POLARIS vieta.",
                100 * max(unsafe),
            )
        write_confusion_csv(scene_cm, args.out / "figures" / "confusion_scenes.csv", class_names)
        plot_confusion(scene_cm, args.out / "figures" / "confusion_scenes.png", class_names)
        log.info("Scene intere:\n%s", scene_cm.format_table(class_names))
    else:
        log.warning("Nessuna scena dello split '%s' presente in %s.", args.split, args.raw_dir)

    # -- gate ---------------------------------------------------------------
    warnings = check_thresholds(cm, eval_cfg)
    report["warnings"] = warnings
    report["gate_2_metrics_pass"] = not warnings
    for w in warnings:
        log.warning(w)

    out_path = args.out / "metrics.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    existing = json.loads(out_path.read_text()) if out_path.exists() else {}
    existing["evaluation"] = report
    out_path.write_text(json.dumps(existing, indent=2))
    log.info("Metriche in %s", out_path)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
