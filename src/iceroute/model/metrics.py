"""Segmentation metrics accumulated over a confusion matrix.

Everything derives from one ``n_classes x n_classes`` matrix built incrementally,
so a whole validation set is summarised in a few kilobytes and per-class IoU,
mIoU, accuracy and the binary ice/water score all stay mutually consistent —
they are different reductions of the same counts, not separate passes.

Convention: rows are ground truth, columns are prediction.
"""

from __future__ import annotations

import numpy as np
import torch

IGNORE_INDEX = 255


class ConfusionMatrix:
    """Accumulate predictions into a confusion matrix and reduce it to metrics."""

    def __init__(self, n_classes: int, ignore_index: int = IGNORE_INDEX):
        self.n_classes = n_classes
        self.ignore_index = ignore_index
        self.matrix = np.zeros((n_classes, n_classes), dtype="int64")

    def reset(self) -> None:
        self.matrix[:] = 0

    def update(self, target: torch.Tensor | np.ndarray, pred: torch.Tensor | np.ndarray) -> None:
        """Add a batch. Both arguments are class ids, not logits."""
        t = (
            target.detach().cpu().numpy()
            if isinstance(target, torch.Tensor)
            else np.asarray(target)
        )
        p = pred.detach().cpu().numpy() if isinstance(pred, torch.Tensor) else np.asarray(pred)
        t, p = t.ravel(), p.ravel()

        keep = (t != self.ignore_index) & (t < self.n_classes)
        if not keep.any():
            return
        t, p = t[keep].astype("int64"), np.clip(p[keep], 0, self.n_classes - 1).astype("int64")

        # bincount on the flattened (truth, prediction) pair is far faster than
        # a Python loop and exact for the sizes involved.
        idx = t * self.n_classes + p
        counts = np.bincount(idx, minlength=self.n_classes**2)
        self.matrix += counts.reshape(self.n_classes, self.n_classes)

    # -- reductions ---------------------------------------------------------

    def iou_per_class(self) -> np.ndarray:
        """IoU per class. NaN where a class is absent from both truth and prediction."""
        tp = np.diag(self.matrix).astype("float64")
        fp = self.matrix.sum(axis=0) - tp
        fn = self.matrix.sum(axis=1) - tp
        union = tp + fp + fn
        with np.errstate(divide="ignore", invalid="ignore"):
            iou = np.where(union > 0, tp / union, np.nan)
        return iou

    def miou(self) -> float:
        """Mean IoU over classes that appear in the ground truth.

        A class with no ground-truth pixels in this split is excluded from the
        average. Scoring it 0 would punish the model for the split's
        composition; scoring it 1 would flatter it. Excluding it is the
        standard convention and the only defensible one.

        Hallucination is still penalised: predicting an absent class turns
        those pixels into false negatives for whichever class was actually
        there, which lowers *that* class's IoU. Nothing is swept away — see
        ``n_classes_scored`` for how many classes the average covers.
        """
        iou = self.iou_per_class()
        scored = iou[self.support() > 0]
        return (
            float(np.nanmean(scored)) if scored.size and np.isfinite(scored).any() else float("nan")
        )

    def n_classes_scored(self) -> int:
        """How many classes the mIoU average covers, i.e. have ground truth here."""
        return int((self.support() > 0).sum())

    def n_classes_predicted_absent(self) -> int:
        """Classes the model predicted that have no ground truth in this split."""
        return int(((self.support() == 0) & (self.matrix.sum(axis=0) > 0)).sum())

    def accuracy(self) -> float:
        total = self.matrix.sum()
        return float(np.diag(self.matrix).sum() / total) if total else float("nan")

    def accuracy_per_class(self) -> np.ndarray:
        support = self.matrix.sum(axis=1).astype("float64")
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(support > 0, np.diag(self.matrix) / support, np.nan)

    def f1_per_class(self) -> np.ndarray:
        tp = np.diag(self.matrix).astype("float64")
        fp = self.matrix.sum(axis=0) - tp
        fn = self.matrix.sum(axis=1) - tp
        denom = 2 * tp + fp + fn
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(denom > 0, 2 * tp / denom, np.nan)

    def binary_accuracy(self, water_class: int = 0) -> float:
        """Ice vs water accuracy — SPEC §7.4 requires it separately from mIoU.

        Collapses every ice class into one. It is the number a sceptical
        engineer will ask for, because a single threshold on HV already gets
        close to it.
        """
        m = self.matrix
        water_true = m[water_class, :].sum()
        ice_rows = [i for i in range(self.n_classes) if i != water_class]
        correct_water = m[water_class, water_class]
        correct_ice = m[np.ix_(ice_rows, ice_rows)].sum()
        total = water_true + m[ice_rows, :].sum()
        return float((correct_water + correct_ice) / total) if total else float("nan")

    def binary_iou(self, water_class: int = 0) -> float:
        """IoU of the 'ice' class after collapsing all ice types."""
        m = self.matrix
        ice = [i for i in range(self.n_classes) if i != water_class]
        tp = m[np.ix_(ice, ice)].sum()
        fp = m[water_class, ice].sum()
        fn = m[ice, water_class].sum()
        union = tp + fp + fn
        return float(tp / union) if union else float("nan")

    def support(self) -> np.ndarray:
        return self.matrix.sum(axis=1)

    def normalized(self) -> np.ndarray:
        """Row-normalised matrix (recall per class), for plotting."""
        support = self.matrix.sum(axis=1, keepdims=True).astype("float64")
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(support > 0, self.matrix / support, 0.0)

    def as_dict(self, class_names: dict[int, str] | None = None) -> dict:
        names = class_names or {i: str(i) for i in range(self.n_classes)}
        iou = self.iou_per_class()
        f1 = self.f1_per_class()
        acc = self.accuracy_per_class()
        support = self.support()
        return {
            "miou": self.miou(),
            "n_classes_scored": self.n_classes_scored(),
            "n_classes_predicted_absent": self.n_classes_predicted_absent(),
            "accuracy": self.accuracy(),
            "binary_accuracy": self.binary_accuracy(),
            "binary_iou": self.binary_iou(),
            "per_class": {
                names.get(i, str(i)): {
                    "iou": None if np.isnan(iou[i]) else float(iou[i]),
                    "f1": None if np.isnan(f1[i]) else float(f1[i]),
                    "accuracy": None if np.isnan(acc[i]) else float(acc[i]),
                    "support_px": int(support[i]),
                }
                for i in range(self.n_classes)
            },
            "confusion_matrix": self.matrix.tolist(),
        }

    def format_table(self, class_names: dict[int, str] | None = None) -> str:
        names = class_names or {i: str(i) for i in range(self.n_classes)}
        iou, f1, acc, sup = (
            self.iou_per_class(),
            self.f1_per_class(),
            self.accuracy_per_class(),
            self.support(),
        )
        total = max(int(sup.sum()), 1)
        lines = [f"  {'classe':<14}{'IoU':>8}{'F1':>8}{'recall':>8}{'supporto':>12}{'%':>7}"]
        lines.append("  " + "-" * 57)
        for i in range(self.n_classes):
            fmt = lambda v: "     —" if np.isnan(v) else f"{v:6.3f}"  # noqa: E731
            lines.append(
                f"  {names.get(i, str(i)):<14}{fmt(iou[i]):>8}{fmt(f1[i]):>8}{fmt(acc[i]):>8}"
                f"{int(sup[i]):>12,}{100 * sup[i] / total:>6.1f}%"
            )
        lines.append("  " + "-" * 57)
        absent = self.n_classes_predicted_absent()
        note = f", {absent} predette ma assenti" if absent else ""
        lines.append(
            f"  {'mIoU':<14}{self.miou():>8.3f}   "
            f"(media su {self.n_classes_scored()}/{self.n_classes} classi con ground truth{note})"
        )
        lines.append(f"  {'accuratezza':<14}{self.accuracy():>8.3f}")
        lines.append(f"  {'ghiaccio/acqua':<14}{self.binary_accuracy():>8.3f}")
        return "\n".join(lines)
