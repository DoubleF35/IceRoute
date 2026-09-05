"""Tests for the downlink compression measurement.

The tests that matter are the ones that stop a flattering number escaping:
``test_ground_truth_measurement_refuses_to_produce_a_headline`` and
``test_a_speckled_mask_compresses_worse_than_a_polygon_mask``. The second one
demonstrates the effect the first one guards against.
"""

from __future__ import annotations

import numpy as np
import pytest

from iceroute.pitch.compression import (
    N_SAR_CHANNELS,
    SAR_BYTES_PER_PIXEL,
    CompressionResult,
    encode_mask_png,
    measure_scene,
    rle_estimate_bytes,
    summarise,
    verdict,
)

H, W = 512, 512


def _polygon_mask(seed=0):
    """A mask like an analyst's chart: a few large, contiguous regions."""
    mask = np.zeros((H, W), dtype="uint8")
    mask[: H // 3] = 0
    mask[H // 3 : 2 * H // 3] = 4
    mask[2 * H // 3 :] = 5
    mask[:, : W // 4] = 255
    return mask


def _speckled_mask(seed=0):
    """A mask like a network's output: the same regions, noisy at pixel level."""
    rng = np.random.default_rng(seed)
    mask = _polygon_mask()
    flip = rng.random((H, W)) < 0.15
    noise = rng.integers(0, 6, size=(H, W)).astype("uint8")
    out = np.where(flip, noise, mask).astype("uint8")
    out[mask == 255] = 255
    return out


def _sar(seed=0):
    rng = np.random.default_rng(seed)
    return rng.standard_normal((2, H, W)).astype("float32")


# ---------------------------------------------------------------------------
# Encoding
# ---------------------------------------------------------------------------


def test_mask_png_is_lossless():
    """A downlinked mask that changed a class in transit would be worthless."""
    import io

    from PIL import Image

    mask = _speckled_mask()
    data = encode_mask_png(mask, 6)
    back = np.array(Image.open(io.BytesIO(data)))
    assert np.array_equal(back, mask)


def test_mask_png_beats_one_byte_per_pixel():
    assert len(encode_mask_png(_polygon_mask(), 6)) < H * W


def test_a_speckled_mask_compresses_worse_than_a_polygon_mask():
    """The reason mask_source has to be declared.

    An ice chart is polygons; a U-Net output is speckled. Measuring on the
    former and quoting it as the operational number overstates the ratio.
    """
    polygon = len(encode_mask_png(_polygon_mask(), 6))
    speckled = len(encode_mask_png(_speckled_mask(), 6))
    assert speckled > polygon * 2, (
        f"atteso un divario netto, ottenuto {polygon} vs {speckled} byte"
    )


def test_rle_estimate_counts_runs_not_pixels():
    uniform = np.zeros((10, 100), dtype="uint8")
    assert rle_estimate_bytes(uniform) == 3          # one run

    striped = np.tile(np.arange(2, dtype="uint8"), (10, 50))
    assert rle_estimate_bytes(striped) == 1000 * 3   # every pixel its own run


def test_rle_handles_an_empty_mask():
    assert rle_estimate_bytes(np.zeros((0, 0), dtype="uint8")) == 0


# ---------------------------------------------------------------------------
# Measurement
# ---------------------------------------------------------------------------


def test_measure_requires_a_declared_mask_source():
    with pytest.raises(TypeError):
        measure_scene("s", _polygon_mask(), _sar(), n_classes=6)  # type: ignore[call-arg]


def test_measure_rejects_an_unknown_mask_source():
    with pytest.raises(ValueError, match="mask_source"):
        measure_scene("s", _polygon_mask(), _sar(), n_classes=6, mask_source="guessed")


def test_measure_rejects_mismatched_shapes():
    """Comparing a mask and SAR of different sizes compares two scenes."""
    with pytest.raises(ValueError, match="non coincidono"):
        measure_scene(
            "s", np.zeros((10, 10), dtype="uint8"), _sar(), n_classes=6,
            mask_source="predicted",
        )


def test_designated_numerator_is_the_compressed_one():
    """Quoting the uncompressed ratio charges the baseline for sending padding."""
    r = measure_scene("s", _polygon_mask(), _sar(), n_classes=6, mask_source="predicted")
    assert r.ratio == pytest.approx(r.raw_sar_png_bytes / r.payload_bytes)
    assert r.ratio <= r.ratio_vs_uncompressed
    assert r.as_dict()["numerator"]["designated"] == "sar_channels_png"


def test_uncompressed_numerator_uses_sensor_precision():
    """float32 is the research dataset's format, not the sensor's."""
    r = measure_scene("s", _polygon_mask(), _sar(), n_classes=6, mask_source="predicted")
    assert r.raw_sar_bytes == H * W * SAR_BYTES_PER_PIXEL * N_SAR_CHANNELS


def test_payload_includes_the_route():
    route = {"type": "FeatureCollection", "features": [{"x": list(range(500))}]}
    without = measure_scene(
        "s", _polygon_mask(), _sar(), n_classes=6, mask_source="predicted"
    )
    with_route = measure_scene(
        "s", _polygon_mask(), _sar(), n_classes=6, mask_source="predicted",
        route_geojson=route,
    )
    assert with_route.payload_bytes > without.payload_bytes
    assert with_route.ratio < without.ratio


def test_missing_route_is_disclosed():
    r = measure_scene("s", _polygon_mask(), _sar(), n_classes=6, mask_source="predicted")
    assert any("Nessuna rotta" in n for n in r.notes)


def test_sar_padding_is_disclosed():
    """7-58% of a real scene is exact-zero swath padding; it must be visible."""
    sar = _sar()
    sar[:, :, : W // 2] = 0.0
    r = measure_scene("s", _polygon_mask(), sar, n_classes=6, mask_source="predicted")
    assert r.sar_zero_fraction == pytest.approx(0.5, abs=0.01)
    assert any("padding" in n for n in r.notes)


# ---------------------------------------------------------------------------
# The guard on the headline number
# ---------------------------------------------------------------------------


def test_ground_truth_measurement_refuses_to_produce_a_headline():
    """SPEC rule 2. An upper bound must not be quotable as the real figure."""
    r = measure_scene(
        "s", _polygon_mask(), _sar(), n_classes=6, mask_source="ground_truth"
    )
    assert r.is_provisional
    assert any("PROVVISORIO" in n for n in r.notes)

    s = summarise([r])
    assert s["provisional"]
    text = verdict(s)
    assert "LIMITE SUPERIORE" in text
    assert "non va nel pitch" in text


def test_predicted_measurement_produces_a_usable_verdict():
    r = measure_scene("s", _speckled_mask(), _sar(), n_classes=6, mask_source="predicted")
    s = summarise([r])
    assert not s["provisional"]
    assert "LIMITE SUPERIORE" not in verdict(s)


def test_verdict_replaces_the_claim_when_measurement_is_lower():
    """SPEC §10.2: if it is not 200:1, the pitch number becomes the measured one."""
    r = measure_scene("s", _speckled_mask(), _sar(), n_classes=6, mask_source="predicted")
    s = summarise([r])
    s["ratio_median"] = 47.0
    text = verdict(s, claimed=200.0)
    assert "47:1" in text
    assert "non 200:1" in text


def test_summary_reports_the_range_not_just_the_mean():
    """Open water and broken ice compress very differently."""
    results = [
        measure_scene(f"s{i}", m, _sar(i), n_classes=6, mask_source="predicted")
        for i, m in enumerate([_polygon_mask(), _speckled_mask(), _speckled_mask(2)])
    ]
    s = summarise(results)
    assert s["ratio_min"] < s["ratio_max"]
    assert s["n_scenes"] == 3
    assert len(s["per_scene"]) == 3


def test_summary_of_nothing_is_an_error_not_a_number():
    assert "error" in summarise([])


def test_result_dict_carries_everything_needed_to_recompute():
    r = measure_scene("s", _polygon_mask(), _sar(), n_classes=6, mask_source="predicted")
    d = r.as_dict()
    assert d["numerator"]["sar_channels_png"] > 0
    assert d["denominator"]["total"] == r.payload_bytes
    assert d["denominator"]["mask_png"] + d["denominator"]["route_geojson"] \
        + d["denominator"]["metadata"] == d["denominator"]["total"]


def test_ratio_is_finite_when_the_payload_is_empty():
    """Division guard: an empty payload must not raise."""
    r = CompressionResult(
        scene_id="s", shape=(1, 1), n_pixels=1, raw_sar_bytes=100,
        raw_netcdf_bytes=0, raw_sar_png_bytes=100, mask_png_bytes=0,
        route_bytes=0, metadata_bytes=0, n_classes=6,
    )
    assert np.isfinite(r.ratio)
