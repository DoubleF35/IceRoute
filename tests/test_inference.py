"""Tests for whole-scene tiled inference.

The one that matters is ``test_streaming_matches_a_naive_full_accumulator``:
the streaming implementation exists to bound memory, and a bug in its
finalisation logic would corrupt a horizontal band of every scene — which is
exactly the kind of defect that looks like "the model is bad at ice margins".
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from torch import nn

from iceroute.model.inference import (
    DIV_EPS,
    RAMP_FLOOR,
    blend_window,
    predict_scene,
    tile_starts,
)

N_CLASSES = 4


class TileEdgeModel(nn.Module):
    """Output depends on position *within the tile*, not just on the pixel.

    This is the only kind of model that can detect an accumulator bug. A
    position-independent model returns the same probability vector for a pixel
    no matter which tile covers it, so every weighting scheme — including a
    broken one that discards a whole tile row's contribution — produces an
    identical argmax. Real convolutional networks are position-dependent near
    tile edges, which is why the blending exists at all; this model reproduces
    that property in the crudest possible way.
    """

    def __init__(self, n_classes: int = N_CLASSES):
        super().__init__()
        self.n_classes = n_classes

    def forward(self, x):
        b, _, h, w = x.shape
        row = torch.linspace(-3.0, 3.0, h).view(1, h, 1).expand(b, h, w)
        col = torch.linspace(-3.0, 3.0, w).view(1, 1, w).expand(b, h, w)
        out = torch.full((b, self.n_classes, h, w), -6.0)
        out[:, 0] = x[:, 0] - row - col
        out[:, 1] = x[:, 0] + row + col
        return out


class ConstantModel(nn.Module):
    """Predicts a fixed class everywhere. Isolates the tiling from the model."""

    def __init__(self, cls: int, n_classes: int = N_CLASSES):
        super().__init__()
        self.cls, self.n_classes = cls, n_classes

    def forward(self, x):
        out = torch.full((x.shape[0], self.n_classes, x.shape[2], x.shape[3]), -10.0)
        out[:, self.cls] = 10.0
        return out


class ThresholdModel(nn.Module):
    """Class 1 where channel 0 is positive, class 0 elsewhere.

    Position-dependent, so a tiling bug shows up as a misplaced boundary
    instead of being masked by a uniform output.
    """

    def __init__(self, n_classes: int = N_CLASSES):
        super().__init__()
        self.n_classes = n_classes

    def forward(self, x):
        out = torch.full((x.shape[0], self.n_classes, x.shape[2], x.shape[3]), -10.0)
        hot = x[:, 0] > 0
        out[:, 1] = torch.where(hot, 10.0, -10.0)
        out[:, 0] = torch.where(hot, -10.0, 10.0)
        return out


# ---------------------------------------------------------------------------
# Window and tiling geometry
# ---------------------------------------------------------------------------


def test_blend_window_peaks_in_the_centre():
    w = blend_window(64, 16)
    assert w[32, 32] == pytest.approx(1.0)
    assert w[0, 0] < w[32, 32]


def test_blend_window_is_never_zero():
    """A zero weight would make a single-tile pixel divide 0/0."""
    assert blend_window(64, 16).min() > 0


def test_the_division_guard_cannot_perturb_a_real_weight():
    """Regression: the guard used to be RAMP_FLOOR, above the window's own
    minimum of RAMP_FLOOR**2, so it clamped legitimate corner weights and
    scaled those pixels' probabilities by ~0.978."""
    assert blend_window(256, 32).min() > DIV_EPS
    assert DIV_EPS < RAMP_FLOOR**2


def test_blend_window_is_symmetric():
    w = blend_window(64, 16)
    assert np.allclose(w, w[::-1, :])
    assert np.allclose(w, w[:, ::-1])


def test_blend_window_without_overlap_is_flat():
    assert np.all(blend_window(32, 0) == 1.0)


def test_tile_starts_cover_the_full_extent():
    for extent in (256, 300, 512, 1000, 1024):
        starts = tile_starts(extent, 256, 224)
        assert starts[0] == 0
        assert starts[-1] + 256 == extent, f"ultima tile non a filo per extent={extent}"


def test_tile_starts_handles_an_extent_below_one_tile():
    assert tile_starts(100, 256, 224) == [0]


def test_tile_starts_never_leaves_a_gap():
    starts = tile_starts(1000, 256, 224)
    covered = np.zeros(1000, dtype=bool)
    for s in starts:
        covered[s : s + 256] = True
    assert covered.all()


# ---------------------------------------------------------------------------
# Correctness of the streaming reconstruction
# ---------------------------------------------------------------------------


def test_constant_model_gives_a_constant_map():
    x = np.random.default_rng(0).standard_normal((3, 700, 640)).astype("float32")
    labels, _, _ = predict_scene(
        ConstantModel(2), x, n_classes=N_CLASSES, tile_size=256, overlap=32, amp=False
    )
    assert labels.shape == (700, 640)
    assert np.all(labels == 2)


def test_output_shape_matches_the_input_exactly():
    """Padding for undersized scenes must not leak into the returned array."""
    x = np.zeros((3, 100, 137), dtype="float32")
    labels, _, stats = predict_scene(
        ConstantModel(1), x, n_classes=N_CLASSES, tile_size=256, overlap=32, amp=False
    )
    assert labels.shape == (100, 137)
    assert stats.warnings and "padding" in stats.warnings[0]


def test_a_spatial_boundary_lands_in_the_right_place():
    """A vertical edge in the input must appear at the same column in the output."""
    h, w, edge = 600, 700, 421
    x = np.full((3, h, w), -1.0, dtype="float32")
    x[0, :, edge:] = 1.0

    labels, _, _ = predict_scene(
        ThresholdModel(), x, n_classes=N_CLASSES, tile_size=256, overlap=32, amp=False
    )
    assert np.all(labels[:, : edge - 1] == 0)
    assert np.all(labels[:, edge + 1 :] == 1)


def _naive_predict(model, x, *, n_classes, tile_size, overlap):
    """Reference implementation: one full-scene accumulator, no streaming."""
    from iceroute.model.inference import tile_starts as ts

    _, h, w = x.shape
    stride = tile_size - overlap
    win = blend_window(tile_size, overlap)
    acc = np.zeros((n_classes, h, w), dtype="float32")
    acc_w = np.zeros((h, w), dtype="float32")

    with torch.no_grad():
        for r0 in ts(h, tile_size, stride):
            for c0 in ts(w, tile_size, stride):
                crop = x[:, r0 : r0 + tile_size, c0 : c0 + tile_size][None]
                probs = torch.softmax(model(torch.from_numpy(crop)).float(), dim=1).numpy()[0]
                acc[:, r0 : r0 + tile_size, c0 : c0 + tile_size] += probs * win
                acc_w[r0 : r0 + tile_size, c0 : c0 + tile_size] += win

    return (acc / np.maximum(acc_w, DIV_EPS)).argmax(axis=0).astype("uint8")


@pytest.mark.parametrize("shape", [(800, 700), (756, 600), (513, 513)])
def test_streaming_matches_a_naive_full_accumulator(shape):
    """The memory optimisation must not change a single pixel.

    Uses a tile-position-dependent model on purpose. With a position-independent
    one this assertion holds even when vertical blending is completely broken,
    which is how the original carry-over bug survived its first test.

    The shapes cover the case where the last tile row is shifted flush to the
    edge and therefore finalises a different number of rows from the others.
    """
    rng = np.random.default_rng(7)
    h, w = shape
    x = rng.standard_normal((3, h, w)).astype("float32")
    x[0] += np.linspace(-2, 2, h)[:, None] + np.linspace(-2, 2, w)[None, :]

    model = TileEdgeModel()
    streamed, _, stats = predict_scene(
        model, x, n_classes=N_CLASSES, tile_size=256, overlap=32, batch_size=4, amp=False
    )
    naive = _naive_predict(model, x, n_classes=N_CLASSES, tile_size=256, overlap=32)

    assert np.array_equal(streamed, naive)
    # And the whole point: it used far less memory to get there.
    assert stats.peak_accumulator_mb < stats.naive_accumulator_mb / 2


def test_vertical_blending_actually_happens():
    """Directly pins the bug the carry-over fixes.

    Discarding the previous tile row's contribution changes the result near a
    horizontal seam. Comparing against a reference that blends correctly is the
    assertion; this test differs from the one above only in being narrow enough
    to name what broke.
    """
    rng = np.random.default_rng(11)
    h, w = 704, 300  # three tile rows at stride 224
    x = rng.standard_normal((3, h, w)).astype("float32")
    model = TileEdgeModel()

    streamed, _, _ = predict_scene(
        model, x, n_classes=N_CLASSES, tile_size=256, overlap=32, amp=False
    )
    naive = _naive_predict(model, x, n_classes=N_CLASSES, tile_size=256, overlap=32)

    seam = slice(224, 256)  # rows the first two tile rows share
    assert np.array_equal(streamed[seam], naive[seam])


def test_streaming_saves_memory_on_a_realistic_scene():
    """A 5000x5000 scene is the real case; the saving must be an order of magnitude."""
    x = np.zeros((3, 300, 300), dtype="float32")
    _, _, stats = predict_scene(
        ConstantModel(0, n_classes=6), x, n_classes=6, tile_size=256, overlap=32, amp=False
    )
    assert stats.peak_accumulator_mb < stats.naive_accumulator_mb


def test_batching_does_not_change_the_result():
    rng = np.random.default_rng(3)
    x = rng.standard_normal((3, 600, 600)).astype("float32")
    model = ThresholdModel()
    a, _, _ = predict_scene(model, x, n_classes=N_CLASSES, batch_size=1, amp=False)
    b, _, _ = predict_scene(model, x, n_classes=N_CLASSES, batch_size=8, amp=False)
    assert np.array_equal(a, b)


def test_returned_probabilities_are_normalised():
    x = np.zeros((3, 400, 400), dtype="float32")
    _, proba, _ = predict_scene(
        ConstantModel(1), x, n_classes=N_CLASSES, amp=False, return_proba=True
    )
    assert proba is not None
    assert proba.shape == (N_CLASSES, 400, 400)
    assert np.allclose(proba.sum(axis=0), 1.0, atol=1e-4)


def test_overlap_must_be_smaller_than_the_tile():
    x = np.zeros((3, 300, 300), dtype="float32")
    with pytest.raises(ValueError, match="overlap"):
        predict_scene(ConstantModel(0), x, n_classes=N_CLASSES, tile_size=64, overlap=64)


def test_rejects_input_without_a_channel_axis():
    with pytest.raises(ValueError, match=r"C, H, W"):
        predict_scene(ConstantModel(0), np.zeros((300, 300), dtype="float32"), n_classes=N_CLASSES)


def test_stats_count_every_tile():
    x = np.zeros((3, 700, 700), dtype="float32")
    _, _, stats = predict_scene(
        ConstantModel(0), x, n_classes=N_CLASSES, tile_size=256, overlap=32, amp=False
    )
    expected = len(tile_starts(700, 256, 224)) ** 2
    assert stats.n_tiles == expected
