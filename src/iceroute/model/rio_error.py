"""Measure the segmentation error in the quantity that drives the decision.

mIoU treats every class confusion as equally bad. POLARIS does not. For a PC6
vessel the Risk Index Values are:

    Open water +3 | New ice +2 | Young ice +2 | Thin FYI +1 | Thick FYI -1 | Old ice -3

so calling New ice "Young ice" costs **exactly zero** — they carry the same RIV
for every vessel class we ship — while calling Old ice "Thick FYI" moves the RIO
by two points per tenth of concentration and can turn a forbidden cell into a
navigable one.

This module measures the error in RIO and, more importantly, in the
**operational decision** POLARIS derives from it. Two properties make it the
right metric for this project:

* It is what the product actually outputs. The route depends on RIO, not on
  class ids, so an error that does not move RIO does not move the route.
* Its errors are **asymmetric**, and the asymmetry is the safety case.
  Predicting a cell as more dangerous than it is costs time. Predicting it as
  safer than it is routes a ship into ice it cannot handle. Averaging the two
  into one number would hide precisely the failure that matters.

This does **not** replace mIoU, and the 0.60 acceptance threshold on mIoU
still stands. A model that scored perfectly here and badly on mIoU would be one whose
errors happen to fall in operationally cheap places — worth knowing, not worth
hiding. Both numbers get reported.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np

from iceroute.model.metrics import IGNORE_INDEX
from iceroute.routing.polaris import (
    RIO_UNKNOWN,
    PolarisTable,
    classify_rio,
    compute_rio,
)

log = logging.getLogger(__name__)

# Operation level codes from classify_rio: 0 normal, 1 elevated, 2 special.
LEVEL_NAMES = {0: "normal", 1: "elevated risk", 2: "special consideration", 3: "unknown"}


@dataclass
class RioError:
    """How wrong the RIO map is, and in which direction."""

    ice_class: str
    n_pixels: int
    rio_mae: float
    rio_bias: float
    rio_rmse: float
    exact_level_agreement: float
    unsafe_fraction: float
    overcautious_fraction: float
    level_confusion: np.ndarray
    worst_unsafe_delta: float = 0.0
    notes: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "ice_class": self.ice_class,
            "n_pixels": self.n_pixels,
            "rio_mae": round(self.rio_mae, 4),
            "rio_bias": round(self.rio_bias, 4),
            "rio_rmse": round(self.rio_rmse, 4),
            "operational_agreement": round(self.exact_level_agreement, 4),
            "unsafe_fraction": round(self.unsafe_fraction, 5),
            "overcautious_fraction": round(self.overcautious_fraction, 5),
            "worst_unsafe_delta": round(self.worst_unsafe_delta, 3),
            "level_confusion": self.level_confusion.tolist(),
            "notes": self.notes,
        }

    def summary(self) -> str:
        return (
            f"{self.ice_class}: RIO MAE {self.rio_mae:.2f} (bias {self.rio_bias:+.2f}), "
            f"correct operational decision {self.exact_level_agreement:.1%}, "
            f"PERMISSIVE {self.unsafe_fraction:.2%}, cautious {self.overcautious_fraction:.2%}"
        )


def rio_error(
    truth_sod: np.ndarray,
    pred_sod: np.ndarray,
    ice_class: str,
    sod_to_polaris: dict[int, str],
    *,
    table: PolarisTable | None = None,
    sic: np.ndarray | None = None,
    mask_valid: np.ndarray | None = None,
    n_classes: int = 6,
) -> RioError:
    """Compare the RIO implied by a prediction against the RIO implied by truth.

    Both maps use the **same** concentration field, so the difference isolates
    the stage-of-development error. Feeding a predicted concentration into one
    side and the true one into the other would confound two error sources.

    ``unsafe_fraction`` is the headline: the fraction of annotated pixels the
    model places in a *less* restrictive operational level than reality. Those
    are the pixels where a route would be planned through ice POLARIS forbids.
    """
    table = table or PolarisTable.load()

    valid = truth_sod != IGNORE_INDEX
    if mask_valid is not None:
        valid &= np.asarray(mask_valid, dtype=bool)
    if not valid.any():
        raise ValueError("No annotated pixel: the RIO error is undefined.")

    rio_true = compute_rio(
        truth_sod,
        ice_class,
        sod_to_polaris,
        table=table,
        sic=sic,
        mask_valid=valid,
        n_classes=n_classes,
    )
    rio_pred = compute_rio(
        pred_sod,
        ice_class,
        sod_to_polaris,
        table=table,
        sic=sic,
        mask_valid=valid,
        n_classes=n_classes,
    )

    both = valid & np.isfinite(rio_true) & np.isfinite(rio_pred)
    if not both.any():
        raise ValueError("No pixel with a defined RIO on both maps.")

    t, p = rio_true[both], rio_pred[both]
    delta = p - t

    lvl_true = classify_rio(np.where(both, rio_true, RIO_UNKNOWN), ice_class)[both]
    lvl_pred = classify_rio(np.where(both, rio_pred, RIO_UNKNOWN), ice_class)[both]

    # Levels are ordered by severity: 0 normal < 1 elevated < 2 special. A
    # prediction with a *lower* level than truth understates the danger.
    unsafe = lvl_pred < lvl_true
    cautious = lvl_pred > lvl_true

    conf = np.zeros((4, 4), dtype="int64")
    np.add.at(conf, (lvl_true, lvl_pred), 1)

    notes = []
    if unsafe.any():
        worst = float(delta[unsafe].max())
        notes.append(
            f"In the worst case the model overestimates the RIO by {worst:.1f} points: "
            f"that is the distance between a cell declared navigable and reality."
        )
    else:
        worst = 0.0
        notes.append("No pixel was classified as less dangerous than the truth.")

    return RioError(
        ice_class=ice_class,
        n_pixels=int(both.sum()),
        rio_mae=float(np.abs(delta).mean()),
        rio_bias=float(delta.mean()),
        rio_rmse=float(np.sqrt((delta**2).mean())),
        exact_level_agreement=float((lvl_pred == lvl_true).mean()),
        unsafe_fraction=float(unsafe.mean()),
        overcautious_fraction=float(cautious.mean()),
        level_confusion=conf,
        worst_unsafe_delta=worst,
        notes=notes,
    )


def free_confusions(
    sod_to_polaris: dict[int, str],
    ice_classes: list[str],
    *,
    table: PolarisTable | None = None,
) -> list[tuple[int, int]]:
    """SOD class pairs that carry the same RIV for **every** vessel class.

    Confusing such a pair costs nothing operationally. Reporting this alongside
    mIoU is what turns "the model is bad at New ice" into a statement someone
    can act on: if New ice and Young ice are operationally identical, that
    particular confusion is not a defect worth spending capacity on.
    """
    table = table or PolarisTable.load()
    classes = sorted(int(k) for k in sod_to_polaris)
    out = []
    for i, a in enumerate(classes):
        for b in classes[i + 1 :]:
            same = all(
                table.riv(ic, sod_to_polaris[a]) == table.riv(ic, sod_to_polaris[b])
                for ic in ice_classes
            )
            if same:
                out.append((a, b))
    return out


def evaluate_all_classes(
    truth_sod: np.ndarray,
    pred_sod: np.ndarray,
    ice_classes: list[str],
    sod_to_polaris: dict[int, str],
    *,
    table: PolarisTable | None = None,
    sic: np.ndarray | None = None,
    mask_valid: np.ndarray | None = None,
    n_classes: int = 6,
) -> dict:
    """RIO error for every vessel class the demo shows.

    The weakest vessel is the strictest test: the same misclassification moves
    a 1A across an operational boundary where a PC6 stays comfortably inside
    one, so the unsafe fraction is generally worst there.
    """
    table = table or PolarisTable.load()
    per_class = {}
    for ic in ice_classes:
        try:
            per_class[ic] = rio_error(
                truth_sod,
                pred_sod,
                ic,
                sod_to_polaris,
                table=table,
                sic=sic,
                mask_valid=mask_valid,
                n_classes=n_classes,
            ).as_dict()
        except ValueError as exc:
            per_class[ic] = {"error": str(exc)}

    scored = [v for v in per_class.values() if "error" not in v]
    worst = max((v["unsafe_fraction"] for v in scored), default=None)

    return {
        "per_ice_class": per_class,
        "worst_unsafe_fraction": worst,
        "operationally_free_confusions": [
            list(pair) for pair in free_confusions(sod_to_polaris, ice_classes, table=table)
        ],
        "note": (
            "unsafe_fraction = fraction of pixels the model places in an operational "
            "level LESS restrictive than the truth. It is the error that takes a ship "
            "into ice POLARIS forbids for it, and the only asymmetric one: being too "
            "cautious costs time, being too permissive costs the ship."
        ),
    }
