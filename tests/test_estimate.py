"""Tests for the downlink panel and the assumption-driven estimates.

These guard two different things. The payload half must keep measuring what
the satellite actually sends — the bug it was written after counted 26 KB of
ground-side comparison track as transmitted data. The time-and-fuel half must
keep *refusing* to produce a number when the number would be invented, which is
the only reason it is defensible to show one at all.
"""

from __future__ import annotations

import json

import pytest

from iceroute.demo import estimate
from iceroute.demo.estimate import (
    ELEVATED_CAP_KN,
    fuel_tonnes,
    payload,
    rescale_transit,
)
from iceroute.report.compression import downlink_route


def _collection(n_baseline_points=1000):
    return {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "properties": {"kind": "iceroute", "ice_class": "PC6"},
                "geometry": {
                    "type": "LineString",
                    "coordinates": [[0.0, 70.0], [1.0, 71.0], [2.0, 72.0]],
                },
            },
            {
                "type": "Feature",
                "properties": {"kind": "baseline", "ice_class": "PC6"},
                "geometry": {
                    "type": "LineString",
                    "coordinates": [[i / 100, 70.0] for i in range(n_baseline_points)],
                },
            },
        ],
    }


# ---------------------------------------------------------------------------
# What actually goes down
# ---------------------------------------------------------------------------


def test_the_comparison_track_is_not_part_of_the_payload():
    """The bug. The baseline is computed on the ground to argue a point; the
    satellite has no reason to send the straight line between two points its
    recipient chose."""
    sent = downlink_route(_collection())
    kinds = [f["properties"]["kind"] for f in sent["features"]]
    assert kinds == ["iceroute"]


def test_stripping_the_baseline_shrinks_the_payload_by_orders_of_magnitude():
    whole = len(json.dumps(_collection(), separators=(",", ":")))
    sent = len(json.dumps(downlink_route(_collection()), separators=(",", ":")))
    assert sent * 10 < whole, "il payload non si è ridotto: la baseline è ancora dentro"


def test_a_collection_with_only_a_baseline_yields_no_payload():
    only_base = {
        "type": "FeatureCollection",
        "features": [f for f in _collection()["features"] if f["properties"]["kind"] == "baseline"],
    }
    assert downlink_route(only_base) is None


def test_no_collection_is_no_payload():
    assert downlink_route(None) is None
    assert downlink_route({}) is None


def test_payload_counts_the_route_and_says_what_it_could_not_measure(tmp_path):
    routes = tmp_path / "routes"
    routes.mkdir()
    (routes / "r.geojson").write_text(json.dumps(_collection()))

    p = payload({"geojson": "r.geojson"}, routes, {}, "SCENA.nc")
    assert p.n_waypoints == 3
    assert 0 < p.route_bytes < 400
    assert p.mask_bytes is None and p.total_bytes is None and p.ratio is None
    assert not p.measured and "non misurat" in p.note


def test_payload_uses_the_measured_mask_and_sar_when_they_exist(tmp_path):
    routes = tmp_path / "routes"
    routes.mkdir()
    (routes / "r.geojson").write_text(json.dumps(_collection()))
    metrics = {
        "compression": {
            "per_scene": [
                {
                    "scene_id": "SCENA.nc",
                    "denominator": {"mask_png": 100_000, "metadata": 60},
                    "numerator": {"sar_channels_png": 50_000_000},
                }
            ]
        }
    }

    p = payload({"geojson": "r.geojson"}, routes, metrics, "SCENA.nc")
    assert p.measured
    assert p.total_bytes == 100_000 + p.route_bytes + 60
    assert p.ratio == pytest.approx(50_000_000 / p.total_bytes)


def test_a_missing_route_file_is_no_payload_not_a_crash(tmp_path):
    assert payload({"geojson": "assente.geojson"}, tmp_path, {}, "S.nc") is None


# ---------------------------------------------------------------------------
# Time, under an assumption
# ---------------------------------------------------------------------------


def test_rescaling_is_exact_when_the_route_stayed_in_normal_operation():
    """At RIO >= 0 every segment runs at open-water speed, so the total is
    strictly proportional to 1/v — not an approximation."""
    t = rescale_transit(24.0, 6.0, nominal_kn=12.0, chosen_kn=24.0, ice_class="PC6")
    assert t.exact and t.hours == pytest.approx(12.0)


def test_the_nominal_speed_returns_the_measured_time_unchanged():
    t = rescale_transit(23.7, 0.0, nominal_kn=12.0, chosen_kn=12.0, ice_class="PC6")
    assert t.hours == pytest.approx(23.7)


def test_a_route_through_the_elevated_band_is_refused_not_approximated():
    """The one that makes this panel honest.

    POLARIS caps the elevated band at a fixed number of knots — 3.0 for PC6 —
    which the slider does not move. Scaling the whole transit by v_nominal/v
    would invent time on the stretch the regulation controls.
    """
    t = rescale_transit(23.7, -4.0, nominal_kn=12.0, chosen_kn=18.0, ice_class="PC6")
    assert t.hours is None
    assert not t.exact
    assert "3 kn" in t.reason and "inventerebbe" in t.reason


@pytest.mark.parametrize("ice_class", ["PC6", "PC7"])
def test_the_refusal_quotes_the_cap_for_that_vessel_class(ice_class):
    t = rescale_transit(20.0, -2.0, nominal_kn=12.0, chosen_kn=15.0, ice_class=ice_class)
    assert f"{ELEVATED_CAP_KN[ice_class]:.0f} kn" in t.reason


def test_a_non_polar_class_says_the_band_is_simply_out():
    """Below PC7 the elevated band already needs special consideration, so
    there is no speed cap to quote — the cells are not navigable at all."""
    t = rescale_transit(20.0, -2.0, nominal_kn=12.0, chosen_kn=15.0, ice_class="IA")
    assert t.hours is None
    assert "3 kn" not in t.reason


def test_an_unknown_min_rio_blocks_the_rescale():
    """Absent evidence that the route stayed in normal operation is not
    evidence that it did."""
    t = rescale_transit(20.0, None, nominal_kn=12.0, chosen_kn=15.0, ice_class="PC6")
    assert t.hours is None and "non noto" in t.reason


@pytest.mark.parametrize("hours,chosen", [(None, 12.0), (20.0, 0.0), (20.0, -3.0)])
def test_nonsense_inputs_produce_nothing(hours, chosen):
    assert (
        rescale_transit(hours, 5.0, nominal_kn=12.0, chosen_kn=chosen, ice_class="PC6").hours
        is None
    )


# ---------------------------------------------------------------------------
# Fuel
# ---------------------------------------------------------------------------


def test_fuel_is_hours_times_a_rate():
    assert fuel_tonnes(48.0, 30.0) == pytest.approx(60.0)


def test_no_consumption_means_no_fuel_figure_and_never_a_zero():
    """A zero on a dashboard reads as a measurement. P-09 has no source, so the
    honest output is nothing at all."""
    assert fuel_tonnes(48.0, None) is None
    assert fuel_tonnes(48.0, 0.0) is None
    assert fuel_tonnes(None, 30.0) is None


# ---------------------------------------------------------------------------
# What the payload costs on the link — A-10
# ---------------------------------------------------------------------------

ASM_LINK = {
    "link": {
        "iridium_certus_100_kbps": 88,
        "inmarsat_fleetbroadband_kbps": 432,
        "iridium_sbd_message_bytes": 300,
    }
}


def test_no_link_rate_means_no_number():
    """Third-party specs live in assumptions.yaml with a source. Without them
    the panel says nothing rather than picking a plausible speed."""
    lt = estimate.link_time(147986, 431, {})
    assert lt.seconds is None and "assumptions" in lt.reason


def test_the_slowest_terminal_is_the_one_quoted():
    """Quoting FleetBroadband's three seconds would be quoting the best case at
    someone who may not have it. The claim has to survive the worst link."""
    lt = estimate.link_time(147986, 431, ASM_LINK)
    assert lt.kbps == 88
    assert lt.seconds == pytest.approx(147986 * 8 / 88_000)


def test_the_route_is_counted_in_whole_sbd_messages():
    """431 bytes is 1.4 messages, and you cannot send 0.4 of one."""
    assert estimate.link_time(147986, 431, ASM_LINK).sbd_messages == 2
    assert estimate.link_time(147986, 300, ASM_LINK).sbd_messages == 1
    assert estimate.link_time(147986, 301, ASM_LINK).sbd_messages == 2


def test_no_payload_means_no_link_time():
    assert estimate.link_time(None, 431, ASM_LINK).seconds is None
    assert estimate.link_time(0, 431, ASM_LINK).seconds is None


def test_a_missing_sbd_size_still_gives_the_seconds():
    """The two facts are independent: one missing must not suppress the other."""
    lt = estimate.link_time(147986, 431, {"link": {"iridium_certus_100_kbps": 88}})
    assert lt.seconds is not None and lt.sbd_messages is None


def test_the_real_assumptions_put_the_payload_in_seconds_not_minutes():
    """This is the answer to the only serious objection A-10 invites. If it ever
    stops being a small number, the deck claim needs revisiting, not the wording."""
    from pathlib import Path

    import yaml

    if not Path("configs/assumptions.yaml").exists():
        pytest.skip("assumptions.yaml assente")
    a = yaml.safe_load(Path("configs/assumptions.yaml").read_text())
    lt = estimate.link_time(a["downlink"]["payload_bytes"], 431, a)
    assert lt.seconds is not None and lt.seconds < 60, f"{lt.seconds:.0f} s non è più 'tredici'"
