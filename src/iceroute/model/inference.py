"""Whole-scene inference by overlapping tiles. SPEC §7.3.

A scene is ~5000x5000 pixels and the model was trained on 256x256 patches, so
inference is tiled. Two things make a naive tiling wrong, and both are visible
in the output as a grid of seams:

* **A convolution near a tile edge has no context on one side.** Its prediction
  there is systematically worse than in the tile centre. Tiles therefore
  overlap and are blended with a window that tapers towards the edges, so the
  centre of a tile — where the model saw context on all sides — dominates.
* **argmax cannot be blended.** Averaging class *ids* is meaningless. The
  accumulation happens on softmax probabilities and argmax is taken once, at
  the end, on the blended field.

Memory
------
The obvious implementation allocates an ``(n_classes, H, W)`` float32
accumulator: for a 5000x5000 scene with 6 classes that is 600 MB, and a second
100 MB for the weights. Instead the scene is consumed in **strips of tile
rows**, and a strip is finalised as soon as no later tile can touch it. Peak
accumulator memory becomes ``(n_classes, tile, W)`` — about 30 MB for the same
scene, a 20x reduction.

That is not incidental tidiness. The pitch claims this pipeline belongs on a
CubeSat; an implementation whose memory grows with scene area would contradict
the claim on the first slide.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import torch

log = logging.getLogger(__name__)

# Two different constants, and conflating them is a real bug: the 2D window is
# an outer product, so its minimum is RAMP_FLOOR**2. Guarding the division with
# RAMP_FLOOR itself would clamp a *legitimate* corner weight of 9.8e-4 up to
# 1e-3 and scale that pixel's probabilities by 0.978 — small, invisible on a
# map, and wrong.
#
# RAMP_FLOOR: where the taper starts. Only has to be > 0 so an edge pixel keeps
# some weight; normalisation makes its exact value irrelevant for pixels a
# single tile covers.
RAMP_FLOOR = 1e-3
# DIV_EPS: purely a 0/0 guard. tile_starts() guarantees every pixel is covered,
# so this should never actually bind; it is small enough that it cannot perturb
# a real weight if it ever did.
DIV_EPS = 1e-12


def blend_window(tile: int, overlap: int) -> np.ndarray:
    """A 2D taper: linear ramp over ``overlap`` pixels, flat in the middle.

    Separable, so it is built from the 1D profile by an outer product. A Hann
    window over the whole tile would also work but throws away the centre
    pixels' full weight for no benefit — the model's predictions there are the
    ones we most want to keep.
    """
    if overlap <= 0:
        return np.ones((tile, tile), dtype="float32")

    ramp_len = min(overlap, tile // 2)
    w = np.ones(tile, dtype="float32")
    ramp = np.linspace(RAMP_FLOOR, 1.0, ramp_len + 2, dtype="float32")[1:-1]
    w[:ramp_len] = ramp
    w[tile - ramp_len :] = ramp[::-1]
    return np.outer(w, w).astype("float32")


def tile_starts(extent: int, tile: int, stride: int) -> list[int]:
    """Start offsets covering ``extent`` with the last tile flush to the edge.

    Letting the final tile hang past the edge and padding it would feed the
    model invented pixels at exactly the place we can least verify the output.
    Shifting it back to ``extent - tile`` costs a little extra overlap instead,
    which the blend absorbs.
    """
    if extent <= tile:
        return [0]
    starts = list(range(0, extent - tile + 1, stride))
    if starts[-1] != extent - tile:
        starts.append(extent - tile)
    return starts


@dataclass
class InferenceStats:
    """What the run cost, for the numbers the pitch is allowed to quote."""

    scene_shape: tuple[int, int]
    n_tiles: int
    tile_size: int
    overlap: int
    seconds: float
    peak_accumulator_mb: float
    device: str
    batch_size: int
    naive_accumulator_mb: float = 0.0
    warnings: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "scene_shape": list(self.scene_shape),
            "n_tiles": self.n_tiles,
            "tile_size": self.tile_size,
            "overlap": self.overlap,
            "seconds": round(self.seconds, 3),
            "peak_accumulator_mb": round(self.peak_accumulator_mb, 2),
            "naive_accumulator_mb": round(self.naive_accumulator_mb, 2),
            "device": self.device,
            "batch_size": self.batch_size,
            "warnings": self.warnings,
        }


def _pad_to_tile(x: np.ndarray, tile: int) -> tuple[np.ndarray, int, int]:
    """Edge-pad a scene smaller than one tile. Returns the array and the pads."""
    _, h, w = x.shape
    ph, pw = max(0, tile - h), max(0, tile - w)
    if ph or pw:
        x = np.pad(x, ((0, 0), (0, ph), (0, pw)), mode="edge")
    return x, ph, pw


@torch.no_grad()
def predict_scene(
    model: torch.nn.Module,
    channels: np.ndarray,
    *,
    n_classes: int,
    tile_size: int = 256,
    overlap: int = 32,
    batch_size: int = 8,
    device: str | torch.device = "cpu",
    return_proba: bool = False,
    amp: bool = True,
) -> tuple[np.ndarray, np.ndarray | None, InferenceStats]:
    """Segment a whole scene.

    Parameters
    ----------
    channels
        ``(C, H, W)`` float32, already standard-scaled exactly as in training.
    return_proba
        Also return the blended ``(n_classes, H, W)`` probabilities. This
        allocates the full-scene array the streaming design exists to avoid, so
        it is opt-in: the demo needs only the class map.

    Returns
    -------
    (labels, proba_or_none, stats)
        ``labels`` is ``(H, W)`` uint8 of class ids.
    """
    import time

    if channels.ndim != 3:
        raise ValueError(f"Attese (C, H, W), ricevuto {channels.shape}")
    if overlap >= tile_size:
        raise ValueError(f"overlap ({overlap}) deve essere < tile_size ({tile_size})")

    t0 = time.monotonic()
    dev = torch.device(device)
    model = model.to(dev).eval()

    orig_h, orig_w = channels.shape[1], channels.shape[2]
    x, pad_h, pad_w = _pad_to_tile(np.ascontiguousarray(channels, dtype="float32"), tile_size)
    _, height, width = x.shape

    stride = tile_size - overlap
    rows = tile_starts(height, tile_size, stride)
    cols = tile_starts(width, tile_size, stride)
    window = blend_window(tile_size, overlap)

    labels = np.zeros((height, width), dtype="uint8")
    proba_full = (
        np.zeros((n_classes, height, width), dtype="float32") if return_proba else None
    )

    # The streaming accumulator spans only the rows a live tile row can touch.
    strip_h = tile_size
    acc = np.zeros((n_classes, strip_h, width), dtype="float32")
    acc_w = np.zeros((strip_h, width), dtype="float32")
    peak_mb = (acc.nbytes + acc_w.nbytes) / 1024**2

    use_amp = bool(amp) and dev.type == "cuda"
    n_tiles = 0

    # Rows already accumulated at the top of the buffer by *previous* tile rows.
    # Getting this wrong is silent: zeroing the whole buffer each iteration
    # throws away the previous row's contribution to exactly the overlap band,
    # which disables vertical blending while still producing a plausible map.
    carry = 0

    for ri, r0 in enumerate(rows):
        # The buffer represents rows [r0, r0 + tile_size). Its first `carry`
        # rows already hold contributions; the rest start empty.
        acc[:, carry:] = 0.0
        acc_w[carry:] = 0.0

        # Every tile in this row, in batches.
        for b0 in range(0, len(cols), batch_size):
            batch_cols = cols[b0 : b0 + batch_size]
            crops = np.stack(
                [x[:, r0 : r0 + tile_size, c0 : c0 + tile_size] for c0 in batch_cols]
            )
            t = torch.from_numpy(crops).to(dev, non_blocking=True)

            with torch.autocast("cuda", dtype=torch.float16, enabled=use_amp):
                logits = model(t)
            # softmax in fp32: fp16 probabilities lose resolution in the tail,
            # and the tail is where the rare old-ice classes live.
            probs = torch.softmax(logits.float(), dim=1).cpu().numpy()

            for k, c0 in enumerate(batch_cols):
                acc[:, :, c0 : c0 + tile_size] += probs[k] * window
                acc_w[:, c0 : c0 + tile_size] += window
            n_tiles += len(batch_cols)

        # Rows that no later tile row can reach are final.
        # tile_starts() is strictly increasing, so this is always positive.
        finalise_to = rows[ri + 1] if ri + 1 < len(rows) else height
        n_final = finalise_to - r0

        block = acc[:, :n_final] / np.maximum(acc_w[:n_final], DIV_EPS)
        labels[r0:finalise_to] = block.argmax(axis=0).astype("uint8")
        if proba_full is not None:
            proba_full[:, r0:finalise_to] = block

        # Shift the still-live tail to the front: those rows are inside the next
        # tile row's footprint and must keep what this row contributed to them.
        carry = tile_size - n_final
        if carry > 0:
            acc[:, :carry] = acc[:, n_final : n_final + carry]
            acc_w[:carry] = acc_w[n_final : n_final + carry]

    labels = labels[:orig_h, :orig_w]
    if proba_full is not None:
        proba_full = proba_full[:, :orig_h, :orig_w]

    stats = InferenceStats(
        scene_shape=(orig_h, orig_w),
        n_tiles=n_tiles,
        tile_size=tile_size,
        overlap=overlap,
        seconds=time.monotonic() - t0,
        peak_accumulator_mb=peak_mb,
        naive_accumulator_mb=(n_classes * height * width * 4 + height * width * 4) / 1024**2,
        device=str(dev),
        batch_size=batch_size,
    )
    if pad_h or pad_w:
        stats.warnings.append(
            f"Scena più piccola di un tile: padding di {pad_h}x{pad_w} px replicando il bordo."
        )

    log.info(
        "inferenza %dx%d: %d tile, %.1fs, accumulatore %.0f MB (naive %.0f MB)",
        orig_h, orig_w, n_tiles, stats.seconds, peak_mb, stats.naive_accumulator_mb,
    )
    return labels, proba_full, stats


class OnnxSessionModel:
    """Adapter making an ONNX Runtime session look like ``torch.nn.Module``.

    Lets ``predict_scene`` run the INT8 model without a second code path. The
    quantized model must be evaluated by *exactly* the same tiling and blending
    as fp32, or the reported mIoU degradation would confound quantization error
    with a change of inference procedure.
    """

    def __init__(self, session, input_name: str | None = None):
        self.session = session
        self.input_name = input_name or session.get_inputs()[0].name
        self.output_name = session.get_outputs()[0].name

    def to(self, device):  # noqa: ARG002 — the session owns its providers
        return self

    def eval(self):
        return self

    def __call__(self, t: torch.Tensor) -> torch.Tensor:
        out = self.session.run(
            [self.output_name], {self.input_name: t.detach().cpu().numpy().astype("float32")}
        )[0]
        return torch.from_numpy(out)
