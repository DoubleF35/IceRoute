"""Tests for evaluation, the HV baseline, ONNX export and quantization.

The baseline tests carry the weight here. If ``fit_hv_threshold`` is wrong, the
comparison that justifies the whole neural network is wrong, and it is wrong in
the flattering direction — a badly-fitted threshold makes the U-Net look good
for free.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from iceroute.model.evaluate import (
    WATER_CLASS,
    check_thresholds,
    fit_hv_threshold,
    score_hv_threshold,
    threshold_verdict,
    write_confusion_csv,
)
from iceroute.model.export import check_parity, export_onnx
from iceroute.model.metrics import IGNORE_INDEX, ConfusionMatrix
from iceroute.model.quantize import PatchCalibrationReader, size_report
from iceroute.model.unet import UNet, UNetConfig

N_CLASSES = 6


def _separable(n=20000, sep=3.0, seed=0):
    """HV values with ice brighter than water by a known margin."""
    rng = np.random.default_rng(seed)
    water = rng.normal(0.0, 1.0, n // 2)
    ice = rng.normal(sep, 1.0, n // 2)
    hv = np.concatenate([water, ice]).astype("float32")
    sod = np.concatenate([np.zeros(n // 2), np.full(n // 2, 4)]).astype("int64")
    return hv, sod


# ---------------------------------------------------------------------------
# Threshold fitting
# ---------------------------------------------------------------------------


def test_threshold_lands_between_the_two_populations():
    hv, sod = _separable(sep=6.0)
    thr, acc = fit_hv_threshold(hv, sod)
    assert 1.0 < thr < 5.0
    assert acc > 0.99


def test_threshold_accuracy_degrades_with_overlap():
    """A sanity check on the fit: harder data must score worse."""
    _, easy = fit_hv_threshold(*_separable(sep=6.0))
    _, hard = fit_hv_threshold(*_separable(sep=0.5))
    assert easy > hard


def test_fitted_threshold_is_actually_optimal():
    """The vectorised search must match an exhaustive one.

    This is the test that would catch an off-by-one in the cumulative-count
    trick, which would otherwise silently return a slightly wrong threshold.
    """
    hv, sod = _separable(n=2000, sep=2.0, seed=3)
    thr, acc = fit_hv_threshold(hv, sod)

    is_ice = sod != WATER_CLASS
    brute = max(float(((hv > t) == is_ice).mean()) for t in np.unique(hv))
    assert acc == pytest.approx(brute, abs=1e-3)


def test_threshold_ignores_fill_pixels():
    """Ignored pixels must not move the threshold."""
    hv, sod = _separable(sep=5.0)
    thr_clean, _ = fit_hv_threshold(hv, sod)

    noise = np.full(5000, 50.0, dtype="float32")
    hv2 = np.concatenate([hv, noise])
    sod2 = np.concatenate([sod, np.full(5000, IGNORE_INDEX)]).astype("int64")
    thr_dirty, _ = fit_hv_threshold(hv2, sod2)

    assert thr_dirty == pytest.approx(thr_clean, rel=0.05)


def test_threshold_refuses_single_class_data():
    """No threshold separates ice from water when there is only ice."""
    hv = np.random.default_rng(0).normal(size=1000).astype("float32")
    sod = np.full(1000, 3, dtype="int64")
    with pytest.raises(ValueError, match="all ice or all water"):
        fit_hv_threshold(hv, sod)


def test_threshold_refuses_empty_data():
    with pytest.raises(ValueError):
        fit_hv_threshold(np.array([], dtype="float32"), np.array([], dtype="int64"))


# ---------------------------------------------------------------------------
# Scoring the baseline
# ---------------------------------------------------------------------------


def test_scored_baseline_is_strong_on_binary_and_weak_on_six_classes():
    """The central claim of the baseline comparison, as an executable assertion.

    A threshold can only ever emit two labels, so with four ice classes present
    its 6-class mIoU is structurally capped well below its binary accuracy.
    """
    rng = np.random.default_rng(1)
    n = 6000
    hv = np.concatenate([rng.normal(0, 0.7, n), rng.normal(5, 0.7, 4 * n)]).astype("float32")
    sod = np.concatenate([np.zeros(n), np.repeat([1, 2, 4, 5], n)]).astype("int64")

    thr, _ = fit_hv_threshold(hv, sod)
    res = score_hv_threshold(hv, sod, thr, n_classes=N_CLASSES, fitted_on="test")

    assert res.binary_accuracy > 0.95
    assert res.miou_6class < 0.45


def test_oracle_threshold_is_never_worse_than_a_fitted_one():
    """By construction the in-hindsight threshold is an upper bound."""
    hv_tr, sod_tr = _separable(sep=2.0, seed=1)
    hv_ev, sod_ev = _separable(sep=2.0, seed=2)

    thr_fit, _ = fit_hv_threshold(hv_tr, sod_tr)
    thr_or, _ = fit_hv_threshold(hv_ev, sod_ev)

    fitted = score_hv_threshold(hv_ev, sod_ev, thr_fit, n_classes=N_CLASSES, fitted_on="train")
    oracle = score_hv_threshold(hv_ev, sod_ev, thr_or, n_classes=N_CLASSES, fitted_on="test")
    assert oracle.binary_accuracy >= fitted.binary_accuracy - 1e-9


def test_verdict_is_honest_when_the_threshold_wins_on_binary():
    """Never-invent-numbers applied to a comparison: if the cheap baseline wins, say so."""
    cm = ConfusionMatrix(N_CLASSES)
    truth = np.array([0, 0, 1, 1, 4, 4, 5, 5])
    cm.update(truth, np.array([0, 1, 1, 1, 4, 4, 5, 5]))

    strong = score_hv_threshold(
        np.array([-1.0, -1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], dtype="float32"),
        truth,
        0.0,
        n_classes=N_CLASSES,
        fitted_on="test",
    )
    out = threshold_verdict(cm, strong)
    assert set(out) >= {"delta_binary_accuracy", "delta_miou", "verdict"}
    if not out["unet_beats_threshold_on_binary"]:
        assert "optimal threshold is competitive" in out["verdict"]


# ---------------------------------------------------------------------------
# Gate thresholds
# ---------------------------------------------------------------------------


def test_frozen_split_has_no_test_key():
    """Pins the contract that made evaluate.py crash on its own default.

    configs/split.yaml freezes train/val/demo only. Code that reaches for ``split["test"]``
    raises KeyError the first time it runs against the real config — which is
    after training, i.e. hours in.
    """
    from pathlib import Path

    import yaml

    cfg = Path("configs/split.yaml")
    if not cfg.exists():
        pytest.skip("split not frozen yet")
    split = yaml.safe_load(cfg.read_text())["split"]
    assert set(split) - {"frozen", "seed", "note"} == {"train", "val", "demo"}


def test_reported_bias_is_never_silent():
    """Every split is biased or not, and the report must say which and how."""
    from iceroute.model.evaluate import report_bias

    assert "OPTIMISTIC" in report_bias("val")
    assert "hand-picked" in report_bias("demo")
    assert "UNBIASED" in report_bias("official_test")


def test_reference_path_maps_a_scene_to_its_labels():
    """The official test scenes ship without targets; the charts sit alongside."""
    from pathlib import Path

    from iceroute.data.loader import reference_path

    p = Path("data/raw/test/20180124T194759_dmi_prep.nc")
    assert reference_path(p).name == "20180124T194759_dmi_prep_reference.nc"
    assert reference_path(p).parent == p.parent


def test_reference_path_is_idempotent_on_a_reference():
    """Applying it twice must not produce _reference_reference."""
    from pathlib import Path

    from iceroute.data.loader import reference_path

    ref = reference_path(Path("a/s_prep.nc"))
    assert reference_path(ref) == ref


def test_missing_reference_file_raises_a_useful_error(tmp_path):
    """Silently scoring against absent labels would report a fabricated mIoU."""
    from iceroute.data.loader import load_config, load_scene

    scene = tmp_path / "s_prep.nc"
    scene.touch()
    with pytest.raises((FileNotFoundError, OSError)) as exc:
        load_scene(
            scene, load_config("configs/data.yaml"), labels_from=tmp_path / "s_prep_reference.nc"
        )
    assert exc.value is not None


@pytest.mark.skipif(
    not Path("data/raw/test").exists()
    or not any(Path("data/raw/test").glob("*_prep_reference.nc")),
    reason="official test set not downloaded",
)
def test_official_test_scene_loads_with_external_labels():
    """End-to-end on the real files: scene from one, labels from another.

    The variable names in the reference file were read off disk, not taken
    from documentation (docs/decisions.md D-006). This is what keeps that true.
    """
    from iceroute.data.loader import load_config, load_scene

    ref = sorted(Path("data/raw/test").glob("*_prep_reference.nc"))[0]
    scene_path = ref.with_name(ref.name.replace("_prep_reference.nc", "_prep.nc"))
    if not scene_path.exists():
        pytest.skip("matching scene not downloaded")

    scene = load_scene(scene_path, load_config("configs/data.yaml"), labels_from=ref)
    assert scene.sod.shape == scene.hh.shape
    assert scene.mask_valid.any(), "no annotated pixel in the reference file"
    assert set(np.unique(scene.sod[scene.mask_valid]).tolist()) <= set(range(6))


def test_gate_warns_below_the_acceptance_thresholds():
    cm = ConfusionMatrix(3)
    cm.update(np.array([0, 1, 2, 0, 1, 2]), np.array([0, 0, 0, 0, 0, 0]))
    warns = check_thresholds(cm, {"min_miou": 0.60, "min_binary_accuracy": 0.90})
    assert len(warns) == 2
    assert any("mIoU" in w for w in warns)


def test_gate_is_silent_on_a_perfect_matrix():
    cm = ConfusionMatrix(3)
    labels = np.array([0, 1, 2, 0, 1, 2])
    cm.update(labels, labels)
    assert check_thresholds(cm, {"min_miou": 0.60, "min_binary_accuracy": 0.90}) == []


def test_confusion_csv_round_trips_the_counts(tmp_path):
    cm = ConfusionMatrix(3)
    cm.update(np.array([0, 0, 1, 2]), np.array([0, 1, 1, 2]))
    path = tmp_path / "cm.csv"
    write_confusion_csv(cm, path, {0: "Water", 1: "New", 2: "Old"})

    lines = path.read_text().strip().splitlines()
    assert lines[0].startswith("truth\\prediction")
    assert lines[1].split(",")[:4] == ["Water", "1", "1", "0"]


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------


def _tiny_model(seed: int = 0):
    """A small UNet with deterministic weights.

    Seeded so the parity assertions test the exporter and not the draw. An
    unseeded model makes the test's outcome depend on where random init happens
    to put the decision boundaries.
    """
    torch.manual_seed(seed)
    return UNet(
        UNetConfig(
            in_channels=2, n_classes=3, encoder_channels=[8, 16], bottleneck_channels=16, groups=4
        )
    )


def test_onnx_export_is_faithful(tmp_path):
    """A drifting export would invalidate every downstream measurement."""
    model = _tiny_model()
    path = export_onnx(model, tmp_path / "m.onnx", in_channels=2, tile_size=64)
    report = check_parity(model, path, in_channels=2, tile_size=64, n_samples=2)
    assert report.passed, report.note
    assert report.max_abs_logit_diff < 1e-4


def test_parity_check_is_not_fooled_by_a_different_model(tmp_path):
    """Exporting model A and comparing against model B must fail loudly."""
    exported = _tiny_model(seed=0)
    path = export_onnx(exported, tmp_path / "m.onnx", in_channels=2, tile_size=64)

    other = _tiny_model(seed=1)  # same shape, different weights
    report = check_parity(other, path, in_channels=2, tile_size=64, n_samples=2)
    assert not report.passed
    assert "NOT faithful" in report.note


def test_parity_is_reproducible_under_load(tmp_path):
    """The gate must not depend on how busy the machine is.

    Multi-threaded convolution sums partial results in completion order, so a
    logit's low bits move with CPU contention and an argmax tie can flip. This
    was observed once: the parity check failed during a training run and passed
    three times in a row afterwards. check_parity now pins one thread; repeated
    calls must agree exactly.
    """
    model = _tiny_model()
    path = export_onnx(model, tmp_path / "m.onnx", in_channels=2, tile_size=64)
    runs = [
        check_parity(model, path, in_channels=2, tile_size=64, n_samples=2, seed=7)
        for _ in range(3)
    ]
    assert len({r.max_abs_logit_diff for r in runs}) == 1
    assert all(r.argmax_disagreement == runs[0].argmax_disagreement for r in runs)


def test_exported_graph_accepts_a_batch(tmp_path):
    """Tiled inference feeds batches, so the batch axis must be dynamic."""
    import onnxruntime as ort

    model = _tiny_model()
    path = export_onnx(model, tmp_path / "m.onnx", in_channels=2, tile_size=64)
    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    x = np.zeros((5, 2, 64, 64), dtype="float32")
    out = sess.run(None, {sess.get_inputs()[0].name: x})[0]
    assert out.shape == (5, 3, 64, 64)


# ---------------------------------------------------------------------------
# Quantization helpers
# ---------------------------------------------------------------------------


class _FakeDataset:
    def __init__(self, n):
        self.n = n

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        return np.full((2, 8, 8), float(i), dtype="float32"), None, None


def test_calibration_reader_spreads_across_the_dataset():
    """Taking the first N patches would sample one or two scenes only."""
    reader = PatchCalibrationReader(_FakeDataset(1000), 10, "input")
    firsts = [float(b["input"].flat[0]) for b in reader._batches]
    assert firsts[0] == 0.0
    assert firsts[-1] == 999.0
    assert max(firsts) - min(firsts) > 900


def test_calibration_reader_is_exhaustible_and_rewindable():
    reader = PatchCalibrationReader(_FakeDataset(20), 4, "input")
    assert sum(1 for _ in iter(reader.get_next, None)) == 4
    assert reader.get_next() is None
    reader.rewind()
    assert reader.get_next() is not None


def test_calibration_reader_caps_at_the_dataset_size():
    reader = PatchCalibrationReader(_FakeDataset(5), 200, "input")
    assert reader.n_samples == 5


def test_calibration_reader_refuses_an_empty_dataset():
    with pytest.raises(ValueError, match="calibration"):
        PatchCalibrationReader(_FakeDataset(0), 10, "input")


def test_size_report_exposes_bytes_per_parameter(tmp_path):
    """The arithmetic that makes an impossible size claim obvious.

    INT8 weights cannot occupy meaningfully less than one byte per parameter,
    so this field is the check that catches a headline number like "58 KB for
    1.96M parameters" without needing to rerun anything.
    """
    fp32 = tmp_path / "a.bin"
    int8 = tmp_path / "b.bin"
    fp32.write_bytes(b"\0" * 4_000_000)
    int8.write_bytes(b"\0" * 1_000_000)

    rep = size_report(fp32, int8, n_parameters=1_000_000)
    assert rep["compression_ratio"] == pytest.approx(4.0)
    assert rep["bytes_per_parameter_int8"] == pytest.approx(1.0)


def test_onnx_session_model_matches_torch(tmp_path):
    """The INT8 graph must be scorable through the same tiling as fp32."""
    from iceroute.model.export import onnx_session
    from iceroute.model.inference import OnnxSessionModel, predict_scene

    model = _tiny_model().eval()
    path = export_onnx(model, tmp_path / "m.onnx", in_channels=2, tile_size=64)
    wrapped = OnnxSessionModel(onnx_session(path))

    x = np.random.default_rng(0).standard_normal((2, 200, 180)).astype("float32")
    a, _, _ = predict_scene(model, x, n_classes=3, tile_size=64, overlap=8, amp=False)
    b, _, _ = predict_scene(wrapped, x, n_classes=3, tile_size=64, overlap=8, amp=False)

    assert (a == b).mean() > 0.999


def test_onnx_session_model_returns_a_tensor(tmp_path):
    from iceroute.model.export import onnx_session
    from iceroute.model.inference import OnnxSessionModel

    model = _tiny_model()
    path = export_onnx(model, tmp_path / "m.onnx", in_channels=2, tile_size=64)
    wrapped = OnnxSessionModel(onnx_session(path))
    out = wrapped(torch.zeros(1, 2, 64, 64))
    assert isinstance(out, torch.Tensor)
    assert out.shape == (1, 3, 64, 64)
