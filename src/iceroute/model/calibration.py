"""Confidence calibration: how much should the router trust a probability?

D-018 established that the model's errors are *confident*: on the pixels it
gets wrong it puts a median 0.95 on the wrong class and 0.03 on the right one.
That is a statement about one quantity (the probability attached to a
prediction), and it matters downstream for a precise reason: any routing rule
that reads probabilities, such as "avoid a cell if P(thick ice) > 0.2", is only
as safe as those probabilities are honest. A model that says 0.95 and is right
80% of the time turns a 20% risk budget into a 5% one without anyone noticing.

This module measures that gap and closes the part of it that can be closed
cheaply.

What is measured
----------------
* **ECE** (expected calibration error): bin pixels by top-class confidence,
  and average |accuracy - confidence| over bins, weighted by bin size. 15
  equal-width bins, the convention of Guo et al. (2017).
* **Class-wise ECE**: the same, for each one-vs-rest event "class k". The
  router cares about one event in particular, "this cell is thick ice", and the
  top-class ECE can look fine while that event is badly calibrated.
* **NLL** and **Brier score**: proper scoring rules, so a better number cannot
  be bought by moving confidence around without being more right.

What is fitted
--------------
**Temperature scaling** (Guo et al., 2017): one scalar T, fitted on the
validation scenes by minimising NLL, then frozen and applied to the test
scenes. It divides the log-probabilities by T before the softmax. It cannot
change a single predicted label (argmax is invariant to it), so accuracy, mIoU
and every route computed from labels stay exactly as they were; it only
changes how much the probabilities claim.

The inference path blends overlapping tiles in *probability* space, so the
logits are not available. Temperature is applied to ``log p`` of the blended
probabilities instead. Inside a tile this is exact (log-softmax differs from the
logits by a per-pixel constant, which the softmax cancels); in the blended
overlap bands it is a close approximation.

What it cannot do
-----------------
Temperature scaling is a monotone, global transform. It can make 0.95 mean
"right 80% of the time"; it cannot move probability onto the true class of an
individual pixel. Accuracy is identical before and after, and the mean
confidence the model places on its *wrong* answers is reported before and
after: it falls, but it stays well above chance, which is the part of D-018
that only a better model can fix.

Usage
-----
    python -m iceroute.model.calibration --config configs/train3_lovasz.yaml

Writes ``artifacts/calibration.json`` and ``docs/img/reliability.png``.
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)

IGNORE = 255
EPS = 1e-7
N_BINS = 15


# ---------------------------------------------------------------------------
# Core maths (pure numpy, unit-tested)
# ---------------------------------------------------------------------------


def apply_temperature(proba: np.ndarray, temperature: float, axis: int = -1) -> np.ndarray:
    """Rescale probabilities by a temperature: ``softmax(log(p) / T)``.

    ``T > 1`` softens (less confident), ``T < 1`` sharpens, ``T = 1`` is the
    identity. Works on any array where ``axis`` indexes the classes.
    """
    if temperature <= 0:
        raise ValueError(f"temperature must be > 0, got {temperature}")
    z = np.log(np.clip(proba, EPS, 1.0)) / temperature
    z = z - z.max(axis=axis, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=axis, keepdims=True)


def nll(proba: np.ndarray, labels: np.ndarray) -> float:
    """Mean negative log-likelihood of the true class. ``proba`` is ``(N, C)``."""
    p_true = proba[np.arange(len(labels)), labels]
    return float(-np.log(np.clip(p_true, EPS, 1.0)).mean())


def brier(proba: np.ndarray, labels: np.ndarray) -> float:
    """Multi-class Brier score: mean squared distance to the one-hot truth."""
    onehot = np.zeros_like(proba)
    onehot[np.arange(len(labels)), labels] = 1.0
    return float(((proba - onehot) ** 2).sum(axis=1).mean())


@dataclass(frozen=True)
class Reliability:
    """Per-bin statistics behind an ECE number, kept for the diagram."""

    edges: np.ndarray  # (n_bins + 1,)
    confidence: np.ndarray  # mean predicted probability per bin (nan if empty)
    accuracy: np.ndarray  # observed frequency per bin (nan if empty)
    count: np.ndarray  # pixels per bin

    @property
    def ece(self) -> float:
        n = self.count.sum()
        if n == 0:
            return float("nan")
        filled = self.count > 0
        gap = np.abs(self.accuracy[filled] - self.confidence[filled])
        return float((gap * self.count[filled]).sum() / n)

    @property
    def mce(self) -> float:
        filled = self.count > 0
        if not filled.any():
            return float("nan")
        return float(np.abs(self.accuracy[filled] - self.confidence[filled]).max())

    def to_dict(self) -> dict:
        def clean(a: np.ndarray) -> list:
            return [None if not np.isfinite(v) else round(float(v), 6) for v in a]

        return {
            "ece": self.ece,
            "mce": self.mce,
            "edges": [round(float(e), 6) for e in self.edges],
            "confidence": clean(self.confidence),
            "accuracy": clean(self.accuracy),
            "count": [int(c) for c in self.count],
        }


def reliability(confidence: np.ndarray, correct: np.ndarray, n_bins: int = N_BINS) -> Reliability:
    """Bin ``confidence`` into equal-width bins on [0, 1] and measure each bin.

    ``correct`` is boolean (or 0/1): whether the event the confidence refers to
    actually happened. For top-class calibration that is "prediction == truth";
    for class-wise calibration it is "truth == k".
    """
    confidence = np.asarray(confidence, dtype=np.float64)
    correct = np.asarray(correct, dtype=np.float64)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    # right-closed bins, with 0.0 falling into the first one
    idx = np.clip(np.searchsorted(edges, confidence, side="left") - 1, 0, n_bins - 1)
    count = np.bincount(idx, minlength=n_bins).astype(np.int64)
    sum_conf = np.bincount(idx, weights=confidence, minlength=n_bins)
    sum_corr = np.bincount(idx, weights=correct, minlength=n_bins)
    with np.errstate(invalid="ignore", divide="ignore"):
        conf = np.where(count > 0, sum_conf / np.maximum(count, 1), np.nan)
        acc = np.where(count > 0, sum_corr / np.maximum(count, 1), np.nan)
    return Reliability(edges=edges, confidence=conf, accuracy=acc, count=count)


def top_class_reliability(proba: np.ndarray, labels: np.ndarray, n_bins: int = N_BINS):
    pred = proba.argmax(axis=1)
    return reliability(proba.max(axis=1), pred == labels, n_bins)


def classwise_reliability(
    proba: np.ndarray, labels: np.ndarray, n_bins: int = N_BINS
) -> list[Reliability]:
    return [reliability(proba[:, k], labels == k, n_bins) for k in range(proba.shape[1])]


def fit_temperature(proba: np.ndarray, labels: np.ndarray, bounds=(0.05, 20.0)) -> float:
    """The temperature that minimises NLL on ``(proba, labels)``.

    One-dimensional and convex in ``log T`` for this objective, so a bounded
    scalar search is enough; no learning rate, no epochs, nothing to tune.
    """
    from scipy.optimize import minimize_scalar

    logp = np.log(np.clip(proba, EPS, 1.0))
    rows = np.arange(len(labels))

    def objective(log_t: float) -> float:
        z = logp / np.exp(log_t)
        z = z - z.max(axis=1, keepdims=True)
        lse = np.log(np.exp(z).sum(axis=1))
        return float((lse - z[rows, labels]).mean())

    res = minimize_scalar(
        objective, bounds=(np.log(bounds[0]), np.log(bounds[1])), method="bounded"
    )
    return float(np.exp(res.x))


def summarise(proba: np.ndarray, labels: np.ndarray, class_names: list[str]) -> dict:
    """Every calibration number for one set of pixels."""
    top = top_class_reliability(proba, labels)
    per_class = classwise_reliability(proba, labels)
    pred = proba.argmax(axis=1)
    wrong = pred != labels
    p_true_wrong = proba[np.arange(len(labels)), labels][wrong]
    return {
        "n_pixels": len(labels),
        "accuracy": float((~wrong).mean()),
        "nll": nll(proba, labels),
        "brier": brier(proba, labels),
        "top_class": top.to_dict(),
        "classwise_ece": {name: rel.ece for name, rel in zip(class_names, per_class, strict=True)},
        "errors_true_class_below_1pct": float((p_true_wrong < 0.01).mean()) if wrong.any() else 0.0,
        "mean_confidence_on_errors": float(proba.max(axis=1)[wrong].mean())
        if wrong.any()
        else float("nan"),
    }


def sample_pixels(
    proba: np.ndarray, truth: np.ndarray, n: int, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    """Draw ``n`` scored pixels from a scene: ``(n, C)`` probabilities and labels.

    ``proba`` is ``(C, H, W)``; ``truth`` is ``(H, W)`` with ``IGNORE`` where
    there is no label. A uniform draw per scene keeps one huge scene from
    swamping the fit; the cost is that per-pixel weighting across scenes is
    by scene, which is stated in the output.
    """
    rows, cols = np.nonzero(truth != IGNORE)
    if len(rows) == 0:
        return np.empty((0, proba.shape[0]), np.float32), np.empty(0, np.int64)
    pick = rng.choice(len(rows), size=min(n, len(rows)), replace=False)
    r, c = rows[pick], cols[pick]
    return proba[:, r, c].T.astype(np.float32), truth[r, c].astype(np.int64)


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------


def plot_reliability(
    before: Reliability, after: Reliability, temperature: float, path: Path
) -> bool:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:  # pragma: no cover
        log.warning("matplotlib not available: no reliability diagram")
        return False

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.6), sharey=True)
    centers = 0.5 * (before.edges[:-1] + before.edges[1:])
    width = before.edges[1] - before.edges[0]
    for ax, rel, title in (
        (axes[0], before, "Uncalibrated"),
        (axes[1], after, f"Temperature-scaled (T = {temperature:.2f})"),
    ):
        filled = rel.count > 0
        ax.plot([0, 1], [0, 1], ls="--", lw=1, color="#888888", label="perfect calibration")
        ax.bar(
            centers[filled],
            rel.accuracy[filled],
            width=width * 0.9,
            color="#2b6cb0",
            alpha=0.85,
            label="observed accuracy",
        )
        ax.bar(
            centers[filled],
            rel.confidence[filled] - rel.accuracy[filled],
            bottom=rel.accuracy[filled],
            width=width * 0.9,
            color="#e53e3e",
            alpha=0.35,
            label="gap",
        )
        ax.set_title(f"{title}\nECE = {100 * rel.ece:.1f}%")
        ax.set_xlabel("predicted confidence (top class)")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("accuracy")
    axes[0].legend(loc="upper left", fontsize=8, frameon=False)
    fig.suptitle("Reliability on the official AI4Arctic test scenes", fontsize=11)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return True


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _collect(paths, *, model, data_cfg, train_cfg, device, per_scene, rng):
    import torch

    from iceroute.data.collapse import collapse
    from iceroute.data.loader import load_scene
    from iceroute.demo.precompute import resolve_labels
    from iceroute.model.inference import predict_scene

    n_classes = int(train_cfg["model"]["n_classes"])
    channels = list(train_cfg["data"].get("channels") or ["hh", "hv", "incidence_angle"])
    collapse_classes = bool(train_cfg["data"].get("collapse_classes"))
    eval_cfg = train_cfg.get("eval", {})
    probas, labels = [], []
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
        truth = np.where(scene.mask_valid, scene.sod, IGNORE).astype("uint8")
        if collapse_classes:
            truth = collapse(truth)
        p, y = sample_pixels(proba, truth, per_scene, rng)
        probas.append(p)
        labels.append(y)
        log.info(
            "[%2d/%d] %s  %d px sampled  (%.0f s)",
            i,
            len(paths),
            path.name[:24],
            len(y),
            time.monotonic() - t0,
        )
        del scene, proba, truth
        if device == "cuda":
            torch.cuda.empty_cache()
    return np.concatenate(probas), np.concatenate(labels)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Fit temperature scaling on validation scenes, measure calibration on test."
    )
    ap.add_argument("--config", type=Path, default=Path("configs/train3_lovasz.yaml"))
    ap.add_argument("--data-config", type=Path, default=Path("configs/data.yaml"))
    ap.add_argument("--split", type=Path, default=Path("configs/split.yaml"))
    ap.add_argument("--train-dir", type=Path, default=Path("data/raw/train"))
    ap.add_argument("--test-dir", type=Path, default=Path("data/raw/test"))
    ap.add_argument("--pixels-per-scene", type=int, default=200_000)
    ap.add_argument("--max-test-scenes", type=int, default=20)
    ap.add_argument("--device", default=None)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--out", type=Path, default=Path("artifacts/calibration.json"))
    ap.add_argument("--figure", type=Path, default=Path("docs/img/reliability.png"))
    args = ap.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)-7s %(message)s")

    import torch
    import yaml

    from iceroute.data.collapse import NAMES as CLASS_NAMES_3
    from iceroute.data.loader import load_config
    from iceroute.demo.precompute import resolve_labels
    from iceroute.model.evaluate import load_checkpoint

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    if device == "cpu":
        torch.set_num_threads(max(1, (torch.get_num_threads() or 1)))
    data_cfg = load_config(args.data_config)
    train_cfg = yaml.safe_load(args.config.read_text())
    split = yaml.safe_load(args.split.read_text())["split"]
    model, _ = load_checkpoint(Path(train_cfg["output"]["best_checkpoint"]), device)
    class_names = [CLASS_NAMES_3[k] for k in sorted(CLASS_NAMES_3)]
    rng = np.random.default_rng(args.seed)

    val_paths = [args.train_dir / s for s in split["val"] if (args.train_dir / s).exists()]
    test_paths = [
        p for p in sorted(args.test_dir.glob("*_prep.nc")) if resolve_labels(p) is not None
    ][: args.max_test_scenes]
    if not val_paths or not test_paths:
        log.error(
            "Need validation scenes in %s and test scenes in %s", args.train_dir, args.test_dir
        )
        return 1

    log.info("Fitting on %d validation scenes (%s)", len(val_paths), device)
    kw = dict(
        model=model,
        data_cfg=data_cfg,
        train_cfg=train_cfg,
        device=device,
        per_scene=args.pixels_per_scene,
        rng=rng,
    )
    p_val, y_val = _collect(val_paths, **kw)
    temperature = fit_temperature(p_val, y_val)
    log.info("Fitted temperature T = %.3f", temperature)

    log.info("Measuring on %d official test scenes", len(test_paths))
    p_test, y_test = _collect(test_paths, **kw)
    p_test_cal = apply_temperature(p_test, temperature, axis=1)

    result = {
        "config": str(args.config),
        "checkpoint": train_cfg["output"]["best_checkpoint"],
        "method": "temperature scaling on log of tile-blended probabilities (Guo et al., 2017)",
        "temperature": temperature,
        "fit": {
            "split": "frozen validation split (configs/split.yaml)",
            "n_scenes": len(val_paths),
            "n_pixels": len(y_val),
            "nll_before": nll(p_val, y_val),
            "nll_after": nll(apply_temperature(p_val, temperature, axis=1), y_val),
        },
        "test": {
            "split": "official AI4Arctic test set (never used for training or selection)",
            "n_scenes": len(test_paths),
            "pixels_per_scene": args.pixels_per_scene,
            "weighting": "uniform sample per scene: scenes weigh equally, not by area",
            "before": summarise(p_test, y_test, class_names),
            "after": summarise(p_test_cal, y_test, class_names),
        },
        "n_bins": N_BINS,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2))
    plot_reliability(
        top_class_reliability(p_test, y_test),
        top_class_reliability(p_test_cal, y_test),
        temperature,
        args.figure,
    )

    b, a = result["test"]["before"], result["test"]["after"]
    log.info("=" * 64)
    log.info("T = %.3f   (fitted on validation, applied to test)", temperature)
    log.info("ECE    %.4f -> %.4f", b["top_class"]["ece"], a["top_class"]["ece"])
    log.info("NLL    %.4f -> %.4f", b["nll"], a["nll"])
    log.info("Brier  %.4f -> %.4f", b["brier"], a["brier"])
    for name in class_names:
        log.info(
            "class-wise ECE %-10s %.4f -> %.4f",
            name,
            b["classwise_ece"][name],
            a["classwise_ece"][name],
        )
    log.info(
        "accuracy %.4f -> %.4f (unchanged by construction); "
        "mean confidence on wrong pixels %.3f -> %.3f",
        b["accuracy"],
        a["accuracy"],
        b["mean_confidence_on_errors"],
        a["mean_confidence_on_errors"],
    )
    log.info("Written %s and %s", args.out, args.figure)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
