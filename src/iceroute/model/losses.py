"""Loss functions for masked multi-class segmentation.

SPEC §7.2: cross-entropy with ``ignore_index`` for invalid pixels, plus Dice on
the valid ones, with configurable weights.

Why both. Cross-entropy optimises per-pixel likelihood and is dominated by
whichever class covers the most area, which in an Arctic scene is usually a
single ice type. Dice is computed per class and normalised by that class's own
area, so a class covering 2% of the scene contributes as much to the gradient
as one covering 60%. Old ice is rare and is exactly what the routing decisions
turn on, so it needs the help.

Class weights come from ``artifacts/data_stats.json``, i.e. from the measured
distribution of the actual cached patches — never hand-tuned.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

IGNORE_INDEX = 255


def inverse_frequency_weights(
    class_frequency: dict[str, float] | dict[int, float],
    n_classes: int,
    *,
    clip_max: float = 10.0,
    normalize: bool = True,
) -> torch.Tensor:
    """Weights inversely proportional to class frequency.

    Two guards matter here. A class absent from the cache gets the maximum
    weight rather than dividing by zero. And every weight is clipped: without a
    ceiling, a class at 0.01% of pixels would earn a weight of 10,000 and a
    handful of pixels would dominate every gradient step.
    """
    freq = {int(k): float(v) for k, v in class_frequency.items()}
    w = torch.empty(n_classes, dtype=torch.float32)
    for c in range(n_classes):
        f = freq.get(c, 0.0)
        w[c] = clip_max if f <= 0.0 else min(1.0 / f, clip_max)
    if normalize:
        w = w * n_classes / w.sum()
    return w


def load_class_weights(
    stats_path: str | Path,
    n_classes: int,
    *,
    clip_max: float = 10.0,
    normalize: bool = True,
    collapse: bool = False,
) -> torch.Tensor | None:
    """Read the measured class distribution and turn it into weights.

    ``collapse`` folds the six SOD frequencies into the three-class scheme
    first. The cache still stores six-class labels, so without this the weights
    would be indexed by the wrong scheme — silently, since three of six entries
    would simply go unused and the remaining three would carry the frequencies
    of open water, new ice and young ice.
    """
    p = Path(stats_path)
    if not p.exists():
        return None
    stats = json.loads(p.read_text())
    freq = stats.get("class_frequency")
    if not freq:
        return None

    if collapse:
        from iceroute.data.collapse import GROUPS

        merged = {
            new: sum(float(freq.get(str(o), freq.get(o, 0.0))) for o in olds)
            for new, olds in GROUPS.items()
        }
        freq = merged

    return inverse_frequency_weights(freq, n_classes, clip_max=clip_max, normalize=normalize)


class DiceLoss(nn.Module):
    """Soft multi-class Dice over valid pixels.

    Classes absent from both prediction and target in a batch are excluded from
    the mean rather than scored as a perfect 1.0 — counting them would make the
    loss look better precisely when a rare class is missing.
    """

    def __init__(self, n_classes: int, *, smooth: float = 1.0, ignore_index: int = IGNORE_INDEX):
        super().__init__()
        self.n_classes = n_classes
        self.smooth = smooth
        self.ignore_index = ignore_index

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        valid = target != self.ignore_index
        if not valid.any():
            return logits.sum() * 0.0  # keeps the graph connected

        probs = F.softmax(logits, dim=1)
        # Clamp before one-hot: ignored positions still hold 255 and would blow
        # up scatter. They are zeroed by the mask immediately after.
        safe = torch.where(valid, target, torch.zeros_like(target))
        onehot = F.one_hot(safe, self.n_classes).permute(0, 3, 1, 2).float()

        m = valid.unsqueeze(1).float()
        probs = probs * m
        onehot = onehot * m

        dims = (0, 2, 3)
        inter = (probs * onehot).sum(dims)
        denom = probs.sum(dims) + onehot.sum(dims)

        present = onehot.sum(dims) > 0
        if not present.any():
            return logits.sum() * 0.0

        dice = (2.0 * inter + self.smooth) / (denom + self.smooth)
        return 1.0 - dice[present].mean()


def lovasz_grad(gt_sorted: torch.Tensor) -> torch.Tensor:
    """Gradient of the Lovász extension of the Jaccard loss.

    From Berman, Triki & Blaschko (CVPR 2018). ``gt_sorted`` is the ground-truth
    indicator permuted by decreasing prediction error. The cumulative sums give
    the Jaccard index at each prefix; differencing turns it into the per-element
    gradient, which is what makes the surrogate exactly equal to the IoU loss on
    the vertices of the hypercube.
    """
    p = gt_sorted.numel()
    gts = gt_sorted.sum()
    intersection = gts - gt_sorted.float().cumsum(0)
    union = gts + (1 - gt_sorted).float().cumsum(0)
    jaccard = 1.0 - intersection / union
    if p > 1:
        jaccard[1:p] = jaccard[1:p] - jaccard[0 : p - 1]
    return jaccard


class LovaszSoftmaxLoss(nn.Module):
    """A convex surrogate for the IoU loss, optimised directly.

    Cross-entropy optimises per-pixel likelihood; Dice optimises a smooth
    approximation of overlap. Neither is the metric we are judged on. Lovász is:
    it is the tight convex surrogate of the Jaccard index, so lowering it lowers
    IoU rather than something correlated with it.

    Only classes **present in the batch** are scored. A class absent from both
    target and prediction has an undefined IoU, and counting it as a perfect 1.0
    would make the loss improve precisely when a rare class goes missing — which
    is the failure mode this project has (D-018).
    """

    def __init__(self, n_classes: int, *, ignore_index: int = IGNORE_INDEX):
        super().__init__()
        self.n_classes = n_classes
        self.ignore_index = ignore_index

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        probs = F.softmax(logits, dim=1)
        # (B, C, H, W) -> (N, C), dropping ignored pixels once rather than
        # masking inside the per-class loop.
        probs = probs.permute(0, 2, 3, 1).reshape(-1, probs.shape[1])
        labels = target.reshape(-1)

        valid = labels != self.ignore_index
        if not valid.any():
            return logits.sum() * 0.0  # keeps the graph connected
        probs, labels = probs[valid], labels[valid]

        losses = []
        for cls in range(self.n_classes):
            fg = (labels == cls).float()
            if fg.sum() == 0:
                continue
            errors = (fg - probs[:, cls]).abs()
            errors_sorted, perm = torch.sort(errors, dim=0, descending=True)
            losses.append(torch.dot(errors_sorted, lovasz_grad(fg[perm])))

        if not losses:
            return logits.sum() * 0.0
        return torch.stack(losses).mean()


class CombinedLoss(nn.Module):
    """``ce_weight * CrossEntropy + dice_weight * Dice``.

    Parameters
    ----------
    n_classes
        Number of output channels.
    ce_weight, dice_weight
        Mixing coefficients from ``configs/train.yaml``.
    class_weights
        Per-class weights for the cross-entropy term, typically from
        :func:`load_class_weights`.
    """

    def __init__(
        self,
        n_classes: int,
        *,
        ce_weight: float = 1.0,
        dice_weight: float = 0.5,
        class_weights: torch.Tensor | None = None,
        ignore_index: int = IGNORE_INDEX,
        label_smoothing: float = 0.0,
        lovasz_weight: float = 0.0,
    ):
        super().__init__()
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight
        self.lovasz_weight = lovasz_weight
        self.ignore_index = ignore_index
        # Registered as a buffer so .to(device) moves it with the module; a bare
        # tensor attribute would stay on CPU and raise mid-training.
        self.register_buffer(
            "class_weights", class_weights if class_weights is not None else torch.empty(0)
        )
        self.label_smoothing = label_smoothing
        self.dice = DiceLoss(n_classes, ignore_index=ignore_index)
        self.lovasz = (
            LovaszSoftmaxLoss(n_classes, ignore_index=ignore_index) if lovasz_weight else None
        )

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> dict[str, torch.Tensor]:
        w = self.class_weights if self.class_weights.numel() else None
        ce = F.cross_entropy(
            logits,
            target,
            weight=w,
            ignore_index=self.ignore_index,
            label_smoothing=self.label_smoothing,
        )
        # An all-invalid batch makes cross_entropy return NaN rather than zero.
        if torch.isnan(ce):
            ce = logits.sum() * 0.0

        dice = self.dice(logits, target) if self.dice_weight else logits.sum() * 0.0
        total = self.ce_weight * ce + self.dice_weight * dice

        out = {"loss": total, "ce": ce.detach(), "dice": dice.detach()}
        if self.lovasz is not None:
            lov = self.lovasz(logits, target)
            out["loss"] = total + self.lovasz_weight * lov
            out["lovasz"] = lov.detach()
        return out


def build_loss(cfg: dict, n_classes: int, stats_path: str | Path) -> CombinedLoss:
    """Build the loss from the ``loss:`` block of ``configs/train.yaml``."""
    lc = cfg.get("loss", {})
    cw_cfg = lc.get("class_weights", {}) or {}
    weights = None
    if cw_cfg.get("mode") == "inverse_frequency":
        weights = load_class_weights(
            stats_path,
            n_classes,
            clip_max=float(cw_cfg.get("clip_max", 10.0)),
            normalize=bool(cw_cfg.get("normalize", True)),
            collapse=bool(cfg.get("data", {}).get("collapse_classes", False)),
        )
    return CombinedLoss(
        n_classes,
        ce_weight=float(lc.get("ce_weight", 1.0)),
        dice_weight=float(lc.get("dice_weight", 0.5)),
        class_weights=weights,
        ignore_index=int(lc.get("ignore_index", IGNORE_INDEX)),
        lovasz_weight=float(lc.get("lovasz_weight", 0.0)),
    )
