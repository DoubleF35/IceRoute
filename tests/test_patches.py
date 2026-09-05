"""Tests for the patch pipeline — SPEC Gate 1.

Three of these guard invariants that, if broken, produce a model that trains
without error and is quietly wrong:

* a target pixel carrying the fill value while its mask says "valid" would feed
  class id 255 to CrossEntropy;
* a scene appearing in both train and val inflates mIoU and the inflation is
  invisible until demo day;
* augmentation that interpolates a categorical target invents classes that were
  never in the chart.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import yaml

from iceroute.data.loader import FILL_VALUE, load_config
from iceroute.data.patches import (
    CHANNELS,
    PatchDataset,
    SceneScan,
    allocate_uniform_quota,
    is_margin_patch,
    iter_windows,
)

REPO = Path(__file__).resolve().parent.parent
CACHE = REPO / "data" / "processed"
STATS = REPO / "artifacts" / "data_stats.json"


# ---------------------------------------------------------------------------
# Pure logic — always runs
# ---------------------------------------------------------------------------


def test_windows_cover_the_edges():
    """The last row and column must be flush with the border.

    A plain range() stride leaves a strip of the scene never sampled, and on an
    Arctic scene the swath edge is exactly where the incidence-angle effect is
    strongest.
    """
    wins = list(iter_windows((500, 700), size=256, stride=192))
    rows = {r for r, _ in wins}
    cols = {c for _, c in wins}
    assert max(rows) == 500 - 256
    assert max(cols) == 700 - 256


def test_windows_empty_when_scene_smaller_than_patch():
    assert list(iter_windows((100, 100), size=256, stride=192)) == []


def test_margin_requires_two_substantial_classes():
    valid = np.ones((10, 10), dtype=bool)

    uniform = np.full((10, 10), 4, dtype="uint8")
    assert not is_margin_patch(uniform, valid, min_classes=2, min_fraction=0.1)

    half = np.full((10, 10), 4, dtype="uint8")
    half[:5] = 0
    assert is_margin_patch(half, valid, min_classes=2, min_fraction=0.1)


def test_margin_ignores_speckle():
    """A handful of stray pixels of a second class is not an ice margin."""
    valid = np.ones((10, 10), dtype=bool)
    speckled = np.full((10, 10), 4, dtype="uint8")
    speckled[0, :3] = 0  # 3% of pixels
    assert not is_margin_patch(speckled, valid, min_classes=2, min_fraction=0.1)


def test_margin_excludes_fill_value():
    """FILL_VALUE is not a class and must not count toward the class tally."""
    valid = np.ones((10, 10), dtype=bool)
    labels = np.full((10, 10), 4, dtype="uint8")
    labels[:5] = FILL_VALUE
    assert not is_margin_patch(labels, valid, min_classes=2, min_fraction=0.1)


def _scan(sid: str, n_margin: int, n_uniform: int) -> SceneScan:
    return SceneScan(
        scene_id=sid,
        region="test",
        month=1,
        valid_fraction=1.0,
        n_candidates=n_margin + n_uniform,
        margin=[(i, 0) for i in range(n_margin)],
        uniform=[(i, 1) for i in range(n_uniform)],
    )


def test_global_quota_hits_the_target_ratio():
    """The balance is global, not per scene.

    This is the regression test for the bug that motivated the two-pass design:
    a scene with zero margin patches used to contribute every uniform patch it
    had, dragging the realised margin fraction from 40% down to 7%.
    """
    scans = [_scan("no-margin", 0, 235), _scan("some", 20, 164), _scan("more", 30, 185)]
    quota = allocate_uniform_quota(scans, target_margin=0.40, balance="discard")

    total_margin = sum(len(s.margin) for s in scans)
    total_uniform = sum(quota.values())
    realised = total_margin / (total_margin + total_uniform)
    assert realised == pytest.approx(0.40, abs=0.02)

    # The margin-free scene still contributes variety, just not dominance.
    assert 0 < quota["no-margin"] < 235


def test_quota_never_exceeds_availability():
    scans = [_scan("a", 100, 5), _scan("b", 100, 3)]
    quota = allocate_uniform_quota(scans, target_margin=0.40, balance="discard")
    assert quota["a"] <= 5
    assert quota["b"] <= 3


def test_quota_without_margins_keeps_everything():
    """Degenerate case must not divide by zero or silently drop all data."""
    scans = [_scan("a", 0, 10), _scan("b", 0, 20)]
    quota = allocate_uniform_quota(scans, target_margin=0.40, balance="discard")
    assert quota == {"a": 10, "b": 20}


def test_sampler_mode_keeps_every_patch():
    """D-017: discarding to hit the ratio threw away 85% of the data.

    In sampler mode the cache keeps everything and the ratio is a training-time
    concern. This is the assertion that the data actually survives.
    """
    scans = [_scan("no-margin", 0, 235), _scan("some", 20, 164), _scan("more", 30, 185)]
    discard = allocate_uniform_quota(scans, target_margin=0.40, balance="discard")
    sampler = allocate_uniform_quota(scans, target_margin=0.40, balance="sampler")

    assert sampler == {"no-margin": 235, "some": 164, "more": 185}
    assert sum(sampler.values()) > 5 * sum(discard.values())


def test_sampler_is_the_default():
    scans = [_scan("a", 10, 100)]
    assert allocate_uniform_quota(scans, target_margin=0.40) == {"a": 100}


def test_unknown_balance_mode_is_rejected():
    with pytest.raises(ValueError, match="balance"):
        allocate_uniform_quota([_scan("a", 1, 1)], target_margin=0.4, balance="magic")


# ---------------------------------------------------------------------------
# Split integrity — always runs
# ---------------------------------------------------------------------------


def test_split_has_no_scene_overlap():
    """SPEC §5.3: split per scene, never per patch.

    Patches of one scene in both train and val leak, and produce metrics that
    look excellent right up until the demo.
    """
    split_path = REPO / "configs" / "split.yaml"
    if not split_path.exists():
        pytest.skip("split non ancora congelato")
    split = yaml.safe_load(split_path.read_text())["split"]

    train, val, demo = set(split["train"]), set(split["val"]), set(split["demo"])
    assert not train & val, f"scene in train e val: {sorted(train & val)}"
    assert not train & demo, f"scene in train e demo: {sorted(train & demo)}"
    assert not val & demo, f"scene in val e demo: {sorted(val & demo)}"
    assert len(train) + len(val) + len(demo) == len(train | val | demo)


def test_split_is_frozen():
    split_path = REPO / "configs" / "split.yaml"
    if not split_path.exists():
        pytest.skip("split non ancora congelato")
    assert yaml.safe_load(split_path.read_text())["split"]["frozen"] is True


def test_polaris_mapping_covers_every_class():
    """Every SOD class the model can emit must map to a POLARIS column.

    An unmapped class means polaris.py has no risk value for a pixel the model
    predicts — the joint between model and routing, SPEC §6.3.
    """
    cfg = load_config(REPO / "configs" / "data.yaml")
    classes = {int(k) for k in cfg["sod_classes"] if int(k) != FILL_VALUE}
    mapped = {int(k) for k in cfg["sod_to_polaris"]}
    assert classes == mapped, f"classi senza mappatura POLARIS: {sorted(classes - mapped)}"

    import csv

    with (REPO / "data" / "polaris_rv.csv").open() as fh:
        rows = [r for r in csv.reader(fh) if r and not r[0].startswith("#")]
    columns = set(rows[0])
    unknown = set(cfg["sod_to_polaris"].values()) - columns
    assert not unknown, f"colonne inesistenti in polaris_rv.csv: {sorted(unknown)}"


# ---------------------------------------------------------------------------
# Cache invariants — need `make data` to have run
# ---------------------------------------------------------------------------

_cached = sorted(CACHE.glob("*_x.npy"))
needs_cache = pytest.mark.skipif(not _cached, reason="cache patch assente: lancia 'make data'")


@needs_cache
def test_fill_value_always_matches_the_mask():
    """SPEC Gate 1: no target pixel may be FILL_VALUE while marked valid."""
    for xp in _cached[:5]:
        stem = xp.name[:-6]
        y = np.load(CACHE / f"{stem}_y.npy", mmap_mode="r")
        m = np.load(CACHE / f"{stem}_m.npy", mmap_mode="r")
        y_valid = np.asarray(y)[np.asarray(m)]
        assert not (y_valid == FILL_VALUE).any(), f"{stem}: FILL_VALUE su pixel valido"


@needs_cache
def test_labels_are_in_range():
    for xp in _cached[:5]:
        stem = xp.name[:-6]
        y = np.asarray(np.load(CACHE / f"{stem}_y.npy", mmap_mode="r"))
        m = np.asarray(np.load(CACHE / f"{stem}_m.npy", mmap_mode="r"))
        assert y[m].max() <= 5, f"{stem}: classe fuori range"


@needs_cache
def test_no_nan_in_cached_inputs():
    """NaN in the input propagates to the loss and destroys a training run."""
    for xp in _cached[:5]:
        x = np.asarray(np.load(xp, mmap_mode="r"))
        assert np.isfinite(x).all(), f"{xp.name}: NaN o inf nella cache"


@needs_cache
def test_cached_shapes_are_consistent():
    cfg = load_config(REPO / "configs" / "data.yaml")
    size = int(cfg["patches"]["size"])
    for xp in _cached[:5]:
        stem = xp.name[:-6]
        x = np.load(xp, mmap_mode="r")
        y = np.load(CACHE / f"{stem}_y.npy", mmap_mode="r")
        m = np.load(CACHE / f"{stem}_m.npy", mmap_mode="r")
        assert x.shape == (len(x), len(CHANNELS), size, size)
        assert y.shape == (len(x), size, size)
        assert m.shape == (len(x), size, size)


@needs_cache
def test_stats_file_records_class_distribution():
    """SPEC Gate 1 requires the distribution to be logged, not just computed."""
    assert STATS.exists(), "artifacts/data_stats.json mancante"
    stats = json.loads(STATS.read_text())
    for key in ("n_patches", "class_pixel_counts", "class_frequency", "margin_fraction"):
        assert key in stats, f"chiave mancante in data_stats.json: {key}"
    assert stats["n_patches"] > 0
    assert sum(stats["class_frequency"].values()) == pytest.approx(1.0, abs=1e-6)


@needs_cache
def test_dataset_augmentation_preserves_label_values():
    """Flips and rot90 must permute pixels, never invent class values."""
    stem = _cached[0].name[:-6]
    ds = PatchDataset(CACHE, [stem + ".nc"], augment=True, seed=0)
    x0, y0, _ = ds[0]
    before = set(np.unique(np.asarray(np.load(CACHE / f"{stem}_y.npy", mmap_mode="r")[0])).tolist())
    for i in range(min(8, len(ds))):
        _, y, _ = ds[i]
        assert set(np.unique(y).tolist()) <= before | {FILL_VALUE}
    assert x0.shape[0] == len(ds.channels)


@needs_cache
def test_dataset_channel_selection():
    stem = _cached[0].name[:-6]
    ds = PatchDataset(CACHE, [stem + ".nc"], channels=["hv"], augment=False)
    x, _, _ = ds[0]
    assert x.shape[0] == 1


# ---------------------------------------------------------------------------
# Sampler weights — the balancing that replaced discarding data (D-017)
# ---------------------------------------------------------------------------


def _fake_cache(tmp_path, n_margin: int, n_uniform: int, size: int = 8):
    """A minimal cache bundle with known margin flags."""
    n = n_margin + n_uniform
    stem = "20200101T000000_fake_prep"
    np.save(tmp_path / f"{stem}_x.npy", np.zeros((n, len(CHANNELS), size, size), "float32"))
    np.save(tmp_path / f"{stem}_y.npy", np.zeros((n, size, size), "uint8"))
    np.save(tmp_path / f"{stem}_m.npy", np.ones((n, size, size), bool))
    flags = np.zeros(n, dtype="bool")
    flags[:n_margin] = True
    np.save(tmp_path / f"{stem}_f.npy", flags)
    return stem + ".nc"


def test_sampler_weights_hit_the_target_in_expectation(tmp_path):
    """The whole point: a draw contains a margin with probability `target`.

    Checked as an exact sum over the weights rather than by sampling, so the
    test cannot pass by luck.
    """
    sid = _fake_cache(tmp_path, n_margin=10, n_uniform=190)
    ds = PatchDataset(tmp_path, [sid], augment=False)

    w = ds.sampler_weights(0.40)
    assert w is not None
    assert w.sum() == pytest.approx(1.0)
    assert w[ds.is_margin].sum() == pytest.approx(0.40)
    assert w[~ds.is_margin].sum() == pytest.approx(0.60)


def test_scarce_margins_are_upweighted_not_duplicated(tmp_path):
    """10 margin patches among 200 must each be drawn far more often."""
    sid = _fake_cache(tmp_path, n_margin=10, n_uniform=190)
    ds = PatchDataset(tmp_path, [sid], augment=False)
    w = ds.sampler_weights(0.40)

    assert len(ds) == 200, "nessuna patch deve essere stata scartata"
    assert w[0] / w[-1] == pytest.approx((0.40 / 10) / (0.60 / 190))


def test_sampler_weights_decline_when_a_group_is_empty(tmp_path):
    """No margins at all: weighting is undefined, so fall back rather than
    divide by zero or silently upweight nothing."""
    sid = _fake_cache(tmp_path, n_margin=0, n_uniform=50)
    assert PatchDataset(tmp_path, [sid], augment=False).sampler_weights(0.40) is None

    sid2 = _fake_cache(tmp_path, n_margin=50, n_uniform=0)
    assert PatchDataset(tmp_path, [sid2], augment=False).sampler_weights(0.40) is None


def test_sampler_weights_reject_a_degenerate_target(tmp_path):
    sid = _fake_cache(tmp_path, n_margin=10, n_uniform=90)
    ds = PatchDataset(tmp_path, [sid], augment=False)
    assert ds.sampler_weights(0.0) is None
    assert ds.sampler_weights(1.0) is None


def test_cache_without_flags_falls_back_safely(tmp_path):
    """An older cache has no _f.npy. It must read as 'no margins' and disable
    weighting, not mislabel arbitrary patches as margins."""
    sid = _fake_cache(tmp_path, n_margin=10, n_uniform=90)
    (tmp_path / sid.replace(".nc", "_f.npy")).unlink()

    ds = PatchDataset(tmp_path, [sid], augment=False)
    assert len(ds) == 100
    assert not ds.is_margin.any()
    assert ds.sampler_weights(0.40) is None


def test_margin_flags_line_up_across_several_scenes(tmp_path):
    """Flags are concatenated per bundle; a misalignment would upweight the
    wrong patches in every scene after the first."""
    for i, (nm, nu) in enumerate([(3, 7), (5, 5)]):
        n = nm + nu
        stem = f"2020010{i}T000000_fake_prep"
        np.save(tmp_path / f"{stem}_x.npy", np.zeros((n, len(CHANNELS), 8, 8), "float32"))
        np.save(tmp_path / f"{stem}_y.npy", np.zeros((n, 8, 8), "uint8"))
        np.save(tmp_path / f"{stem}_m.npy", np.ones((n, 8, 8), bool))
        f = np.zeros(n, bool)
        f[:nm] = True
        np.save(tmp_path / f"{stem}_f.npy", f)

    ds = PatchDataset(
        tmp_path, ["20200100T000000_fake_prep.nc", "20200101T000000_fake_prep.nc"],
        augment=False,
    )
    assert len(ds) == 20
    assert ds.is_margin.size == 20
    assert int(ds.is_margin.sum()) == 8
    assert ds.margin_fraction() == pytest.approx(0.4)
