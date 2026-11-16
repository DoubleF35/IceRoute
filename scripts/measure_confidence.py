"""Quantify *how* the model is wrong, not just how often. Evidence for D-018.

The mIoU says a fraction of pixels are wrong. It does not say whether the model
was hesitating between two candidates or certain of the wrong answer, and that
distinction decides which fixes are worth trying at all.

If errors are near-ties, then temperature scaling, threshold tuning, reordering
the decision toward the more dangerous class — the whole family of inference-time
corrections — has something to work with: the correct class is sitting just
below the winner, and moving the margin promotes it.

If instead the model puts almost all its mass on a wrong class and near zero on
the truth, that family is closed. There is no margin to tune, because the right
answer is not among the candidates being compared. Only different features, a
different architecture or different supervision can help.

This script measures which of the two we are in, on the official test set, so
the answer is a number rather than an intuition.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import yaml

from iceroute.data.collapse import collapse
from iceroute.data.loader import load_config, load_scene
from iceroute.demo.precompute import resolve_labels

log = logging.getLogger("confidence")

IGNORE = 255


def measure(proba: np.ndarray, pred: np.ndarray, truth: np.ndarray) -> dict:
    """Distribution of confidence over the pixels the model got wrong.

    ``proba`` is ``(n_classes, H, W)``. Pixels whose truth is the ignore value
    take no part: they have no correct answer to be wrong about.
    """
    scored = truth != IGNORE
    wrong = scored & (pred != truth)
    n_wrong = int(wrong.sum())
    if n_wrong == 0:
        return {"n_scored": int(scored.sum()), "n_wrong": 0}

    rows, cols = np.nonzero(wrong)
    p_pred = proba[pred[rows, cols], rows, cols]  # mass on the wrong winner
    p_true = proba[truth[rows, cols], rows, cols]  # mass on the right answer

    return {
        "n_scored": int(scored.sum()),
        "n_wrong": n_wrong,
        "error_rate": n_wrong / max(int(scored.sum()), 1),
        # On a wrong pixel, how sure was it of the wrong answer?
        "median_p_predicted": float(np.median(p_pred)),
        # ...and how much did it leave for the truth?
        "median_p_true": float(np.median(p_true)),
        "mean_p_true": float(p_true.mean()),
        # The number that closes the door: errors where the truth was not even
        # a candidate. A margin-based fix cannot promote a class holding 1%.
        "frac_true_below_1pct": float((p_true < 0.01).mean()),
        "frac_true_below_10pct": float((p_true < 0.10).mean()),
        # The complement: errors a decision rule could plausibly have caught.
        "frac_near_tie": float((p_pred - p_true < 0.10).mean()),
    }


def combine(per_scene: list[dict]) -> dict:
    """Pool the per-scene figures, weighting by how many errors each contributed.

    A plain mean over scenes would let a scene with two thousand errors count
    as much as one with two million.
    """
    usable = [r for r in per_scene if r.get("n_wrong")]
    if not usable:
        return {"n_wrong": 0}
    w = np.array([r["n_wrong"] for r in usable], dtype=float)
    out = {"n_scenes": len(usable), "n_wrong": int(w.sum())}
    for key in (
        "median_p_predicted",
        "median_p_true",
        "mean_p_true",
        "frac_true_below_1pct",
        "frac_true_below_10pct",
        "frac_near_tie",
    ):
        vals = np.array([r[key] for r in usable], dtype=float)
        out[key] = float((vals * w).sum() / w.sum())
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Misura la confidenza sugli errori. D-018.")
    ap.add_argument(
        "--train-config",
        type=Path,
        default=None,
        help="default: artifacts/best_config.txt, cioè il modello scelto",
    )
    ap.add_argument("--data-config", type=Path, default=Path("configs/data.yaml"))
    ap.add_argument("--raw-dir", type=Path, default=Path("data/raw/test"))
    ap.add_argument("--scenes", type=int, default=20)
    ap.add_argument("--device", default=None)
    ap.add_argument("--out", type=Path, default=Path("artifacts/confidence.json"))
    args = ap.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)-7s %(message)s")

    if args.train_config is None:
        picked = Path("artifacts/best_config.txt")
        args.train_config = Path(
            picked.read_text().strip() if picked.exists() else "configs/train3.yaml"
        )
    log.info("Modello: %s", args.train_config)

    import torch

    from iceroute.model.evaluate import load_checkpoint
    from iceroute.model.inference import predict_scene

    data_cfg = load_config(args.data_config)
    train_cfg = yaml.safe_load(args.train_config.read_text())
    n_classes = int(train_cfg["model"]["n_classes"])
    collapse_classes = bool(train_cfg["data"].get("collapse_classes"))
    channels = list(train_cfg["data"].get("channels") or ["hh", "hv", "incidence_angle"])
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    model, _ = load_checkpoint(Path(train_cfg["output"]["best_checkpoint"]), device)

    paths = [p for p in sorted(args.raw_dir.glob("*_prep.nc")) if resolve_labels(p) is not None][
        : args.scenes
    ]
    if not paths:
        log.error("Nessuna scena con etichette in %s", args.raw_dir)
        return 1

    per_scene = []
    for i, path in enumerate(paths, 1):
        scene = load_scene(path, data_cfg, labels_from=resolve_labels(path))
        pred, proba, _ = predict_scene(
            model,
            scene.channels(channels),
            n_classes=n_classes,
            device=device,
            return_proba=True,
        )
        truth = np.where(scene.mask_valid, scene.sod, IGNORE).astype("uint8")
        if collapse_classes:
            truth = collapse(truth)

        row = measure(proba, pred.astype("uint8"), truth)
        row["scene_id"] = scene.meta.scene_id
        per_scene.append(row)
        log.info(
            "[%2d/%d] %s  errori %5.1f%%  p(sbagliata) %.3f  p(vera) %.4f",
            i,
            len(paths),
            scene.meta.scene_id[:18],
            100 * row.get("error_rate", 0.0),
            row.get("median_p_predicted", float("nan")),
            row.get("median_p_true", float("nan")),
        )
        del scene, proba, pred
        if device == "cuda":
            torch.cuda.empty_cache()

    agg = combine(per_scene)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(
            {
                "config": str(args.train_config),
                "n_classes": n_classes,
                "split": "test_ufficiale",
                "aggregate": agg,
                "per_scene": per_scene,
            },
            indent=2,
        )
    )

    log.info("=" * 64)
    if not agg.get("n_wrong"):
        log.info("Nessun errore da analizzare.")
        return 0
    log.info("Su %s pixel sbagliati, in mediana il modello mette:", f"{agg['n_wrong']:,}")
    log.info("  %.3f sulla classe SBAGLIATA che ha scelto", agg["median_p_predicted"])
    log.info("  %.4f sulla classe VERA", agg["median_p_true"])
    log.info(
        "  il %.1f%% degli errori lascia alla classe vera meno dell'1%%",
        100 * agg["frac_true_below_1pct"],
    )
    log.info(
        "  il %.1f%% degli errori è un quasi-pareggio (margine < 0,10)", 100 * agg["frac_near_tie"]
    )
    log.info("")
    # The verdict keys on frac_near_tie, and getting this wrong once already
    # produced the opposite conclusion from the data. The first version tested
    # frac_true_below_1pct > 0.5: at 44,6% it printed "worth trying margin
    # fixes" while only 3,4% of errors were actually near-ties.
    #
    # The two are not complements. An error where the true class holds 3% is
    # not below 1%, but it is nowhere near winning either — the winner holds
    # 95%. Only frac_near_tie answers the question being asked, which is
    # whether there is a margin small enough for a decision rule to close.
    tie = agg["frac_near_tie"]
    if tie < 0.20:
        log.info(
            "VERDETTO: il modello sbaglia con sicurezza. Solo il %.1f%% degli errori", 100 * tie
        )
        log.info("ha un margine abbastanza stretto da poter essere ribaltato. Le correzioni")
        log.info("a tempo di inferenza (temperatura, soglie, riordino della decisione) non")
        log.info("hanno su cosa lavorare: agiscono sul margine fra candidati, e il candidato")
        log.info("giusto non è in gara. Vedi D-018.")
    else:
        log.info(
            "VERDETTO: il %.1f%% degli errori è un quasi-pareggio: una correzione sul", 100 * tie
        )
        log.info("margine potrebbe ribaltarne una parte. Vale la pena misurarla.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
