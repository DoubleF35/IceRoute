"""Tests for the assumptions sync.

The file this writes is shown verbatim to investors and is, per SPEC §10.1, the
answer to "based on what?". So the tests care less about the arithmetic than
about two properties: that writing a number never costs a comment, and that the
module refuses rather than guesses.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from iceroute.report import assumptions as asm

SAMPLE = """\
meta:
  updated: 2026-09-05

downlink:
  # Perché una scena sola e non tre mediane.
  ratio_scene: null            # SCRITTO da pitch/assumptions.py
  raw_scene_bytes: null        # SCRITTO
  payload_bytes: null          # SCRITTO
  ratio: null                  # SCRITTO
  ratio_min: null
  ratio_max: null
  provisional: true
  numerator_definition: >-
    Canali SAR compressi senza perdita.

model:
  parameters: 1964182
  onnx_int8_bytes: null
  latency_ms_per_tile: null
  int8_degradation: null
  provisional: true
  latency_note: >-
    Misurata a 1 thread.
"""


def _metrics(ratios=(281.0, 439.2, 770.0), median=439.2):
    return {
        "evaluation": {"n_parameters": 1964131},
        "quantization": {
            "size": {"int8_bytes": 2148589},
            "latency": {"single_thread": {"int8": {"median_ms_per_tile": 39.744}}},
            "accuracy": {"miou_drop": 0.0046},
        },
        "compression": {
            "ratio_median": median,
            "ratio_min": min(ratios),
            "ratio_max": max(ratios),
            "per_scene": [
                {
                    "scene_id": f"s{i}.nc",
                    "ratio": r,
                    "numerator": {"sar_channels_png": int(r * 100_000)},
                    "denominator": {"total": 100_000},
                }
                for i, r in enumerate(ratios)
            ],
        },
    }


# ---------------------------------------------------------------------------
# Writing a number must not cost a comment
# ---------------------------------------------------------------------------


def test_every_comment_survives():
    """``yaml.safe_dump`` would delete all of these. The comments are the
    justification, and the justification is the point of the file."""
    out = asm.sync(_metrics(), SAMPLE).text
    for comment in (
        "# Perché una scena sola e non tre mediane.",
        "# SCRITTO da pitch/assumptions.py",
        "Canali SAR compressi senza perdita.",
        "Misurata a 1 thread.",
    ):
        assert comment in out, f"perso: {comment}"


def test_untouched_keys_are_byte_identical():
    out = asm.sync(_metrics(), SAMPLE).text
    for line in ("meta:", "  updated: 2026-09-05", "  numerator_definition: >-"):
        assert line in out.split("\n")


def test_the_file_stays_valid_yaml():
    assert yaml.safe_load(asm.sync(_metrics(), SAMPLE).text)["downlink"]["ratio"] == 439.2


# ---------------------------------------------------------------------------
# It refuses rather than guesses
# ---------------------------------------------------------------------------


def test_a_key_that_does_not_exist_is_an_error_not_an_append():
    """If a measurement has no home in the file, a person decides where it goes
    and what to say about it. Appending a bare key would produce a number with
    no explanation — exactly what §10.1 forbids."""
    with pytest.raises(asm.SyncError, match="non esiste"):
        asm.set_value(SAMPLE, "downlink", "inventata", 1)


def test_a_hand_written_value_is_never_clobbered():
    text = SAMPLE.replace("  ratio: null", "  ratio: da rimisurare dopo il freeze")
    with pytest.raises(asm.SyncError, match="a mano"):
        asm.sync(_metrics(), text)


def test_a_section_that_does_not_exist_is_an_error():
    with pytest.raises(asm.SyncError):
        asm.set_value(SAMPLE, "inesistente", "ratio", 1)


def test_a_key_in_another_section_is_not_confused_for_this_one():
    """Both sections carry `provisional`. Editing the wrong one would flip a
    claim from provisional to measured on the strength of a different measure."""
    out = asm.set_value(SAMPLE, "model", "provisional", False)
    d = yaml.safe_load(out)
    assert d["model"]["provisional"] is False
    assert d["downlink"]["provisional"] is True


# ---------------------------------------------------------------------------
# The numbers have to survive being divided by the audience
# ---------------------------------------------------------------------------


def test_the_three_downlink_numbers_come_from_one_scene():
    d = yaml.safe_load(asm.sync(_metrics(), SAMPLE).text)["downlink"]
    assert round(d["raw_scene_bytes"] / d["payload_bytes"], 1) == d["ratio"]


def test_the_representative_scene_is_the_median_one():
    got = asm.representative_scene(_metrics())
    assert got["scene"] == "s1.nc" and got["ratio"] == 439.2


def test_an_even_number_of_scenes_still_picks_a_real_scene():
    """With an even count the median matches no scene. Picking the nearest keeps
    the division honest; interpolating would print a payload nobody measured."""
    m = _metrics(ratios=(100.0, 200.0, 300.0, 400.0), median=250.0)
    got = asm.representative_scene(m)
    assert got["ratio"] in (200.0, 300.0)
    assert round(got["raw"] / got["payload"], 1) == got["ratio"]


def test_no_compression_measured_means_no_representative_scene():
    assert asm.representative_scene({"compression": {}}) is None


# ---------------------------------------------------------------------------
# Provisional is a consequence, not an opinion
# ---------------------------------------------------------------------------


def test_everything_measured_clears_provisional():
    d = yaml.safe_load(asm.sync(_metrics(), SAMPLE).text)
    assert d["downlink"]["provisional"] is False
    assert d["model"]["provisional"] is False


def test_a_missing_measurement_keeps_its_section_provisional():
    m = _metrics()
    del m["quantization"]["size"]["int8_bytes"]
    res = asm.sync(m, SAMPLE)
    d = yaml.safe_load(res.text)
    assert d["model"]["provisional"] is True, "una misura manca: la sezione resta provvisoria"
    assert d["downlink"]["provisional"] is False, "l'altra sezione è misurata per intero"
    assert "model.onnx_int8_bytes" in res.missing


# ---------------------------------------------------------------------------
# The file changes when a number changes, and at no other time
# ---------------------------------------------------------------------------


def test_running_twice_changes_nothing():
    """No timestamp, on purpose: a diff in this file has to mean a measurement
    moved, not that someone ran make report again."""
    once = asm.sync(_metrics(), SAMPLE).text
    twice = asm.sync(_metrics(), once)
    assert twice.text == once
    assert not twice.changed


def test_a_changed_measurement_is_reported_as_changed():
    once = asm.sync(_metrics(), SAMPLE).text
    m = _metrics()
    m["quantization"]["size"]["int8_bytes"] = 9999999
    res = asm.sync(m, once)
    assert res.changed == ["model.onnx_int8_bytes"]


def test_the_wrong_hardcoded_parameter_count_gets_corrected():
    """1964182 was typed by hand and is 51 off the measured value."""
    d = yaml.safe_load(asm.sync(_metrics(), SAMPLE).text)
    assert d["model"]["parameters"] == 1964131


# ---------------------------------------------------------------------------
# Against the real repository
# ---------------------------------------------------------------------------


def test_the_real_file_is_in_sync():
    """This is the check preflight runs. It failing means make report is due."""
    if not (Path("artifacts/metrics.json").exists() and Path("configs/assumptions.yaml").exists()):
        pytest.skip("metrics.json o assumptions.yaml assenti")
    res = asm.run(Path("."), dry_run=True)
    assert not res.changed, f"configs/assumptions.yaml è indietro: {res.changed}"


def test_the_real_file_still_refuses_dollars():
    """The one thing this module must never be able to switch on. SPEC §10.1,
    P-09: no source for fuel price or daily consumption, so no currency."""
    if not Path("configs/assumptions.yaml").exists():
        pytest.skip("assumptions.yaml assente")
    econ = yaml.safe_load(Path("configs/assumptions.yaml").read_text())["economics"]
    assert econ["show_dollar_savings"] is False
    assert econ["show_hours_saved"] is True
