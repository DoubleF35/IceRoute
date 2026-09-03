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
) -> torch.Tensor | None:
    """Read the measured class distribution and turn it into weights."""
    p = Path(stats_path)
    if not p.exists():
        return None
    stats = json.loads(p.read_text())
    freq = stats.get("class_frequency")
    if not freq:
        return None
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
    ):
        super().__init__()
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight
        self.ignore_index = ignore_index
        # Registered as a buffer so .to(device) moves it with the module; a bare
        # tensor attribute would stay on CPU and raise mid-training.
        self.register_buffer(
            "class_weights", class_weights if class_weights is not None else torch.empty(0)
        )
        self.label_smoothing = label_smoothing
        self.dice = DiceLoss(n_classes, ignore_index=ignore_index)

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
        return {"loss": total, "ce": ce.detach(), "dice": dice.detach()}


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
        )
    return CombinedLoss(
        n_classes,
        ce_weight=float(lc.get("ce_weight", 1.0)),
        dice_weight=float(lc.get("dice_weight", 0.5)),
        class_weights=weights,
        ignore_index=int(lc.get("ignore_index", IGNORE_INDEX)),
    )
