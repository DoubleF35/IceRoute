"""Geocoding tests — a hard gate for everything downstream.

The mandatory one is ``test_distance_error_under_one_percent``. If it fails the
routing is meaningless, because every transit time the demo shows is derived
from these distances.

The subtlety it guards: EPSG:3413 is true-scale at 70°N only. At 78°N a planar
measurement understates ground distance by 1.9% and at 85°N by 2.8%, so the
naive hypot that looks obviously correct fails this gate outright.
"""

from __future__ import annotations

import numpy as np
import pytest
from pyproj import Transformer

from iceroute.routing.geocode import (
    POLAR_CRS,
    GeoGrid,
    build_target_grid,
    geodesic_distance_m,
    ground_distance_m,
    interpolate_gcp_grid,
    scale_factor,
)

TO_POLAR = Transformer.from_crs("EPSG:4326", POLAR_CRS, always_xy=True)

# Real waypoints in the Arctic, spanning the latitudes the demo scenes cover.
CASES = [
    ("Fram Strait E-W", -10.0, 78.0, -5.0, 78.0),
    ("Fram Strait N-S", -10.0, 78.0, -10.0, 79.0),
    ("NE Greenland diagonal", -20.0, 76.0, -15.0, 77.5),
    ("near the standard parallel", -45.0, 70.0, -40.0, 70.5),
    ("high Arctic", -30.0, 83.0, -25.0, 83.5),
    ("across the central meridian", -50.0, 75.0, -40.0, 75.0),
    ("long leg", -25.0, 72.0, -10.0, 79.0),
]


@pytest.mark.parametrize("name,lon0,lat0,lon1,lat1", CASES)
def test_distance_error_under_one_percent(name, lon0, lat0, lon1, lat1):
    """The gate: scale-corrected planar distance within 1% of geodesic."""
    x0, y0 = TO_POLAR.transform(lon0, lat0)
    x1, y1 = TO_POLAR.transform(lon1, lat1)

    geodesic = geodesic_distance_m(lon0, lat0, lon1, lat1)
    corrected = ground_distance_m(x0, y0, x1, y1, lat_mid=(lat0 + lat1) / 2)

    rel = abs(corrected - geodesic) / geodesic
    assert rel < 0.01, (
        f"{name}: relative error {100 * rel:.2f}% (geodesic {geodesic:.0f} m, "
        f"corrected {corrected:.0f} m). The gate requires < 1%."
    )


@pytest.mark.parametrize("name,lon0,lat0,lon1,lat1", CASES)
def test_uncorrected_distance_is_the_trap(name, lon0, lat0, lon1, lat1):
    """The correction must actually be doing something.

    Documents the failure mode: without dividing by the scale factor, the
    planar distance is systematically short. If this ever starts passing at
    high latitude, the correction has been silently removed.
    """
    x0, y0 = TO_POLAR.transform(lon0, lat0)
    x1, y1 = TO_POLAR.transform(lon1, lat1)
    geodesic = geodesic_distance_m(lon0, lat0, lon1, lat1)
    planar = float(np.hypot(x1 - x0, y1 - y0))
    corrected = ground_distance_m(x0, y0, x1, y1, lat_mid=(lat0 + lat1) / 2)

    assert planar <= geodesic * 1.0001, "the planar distance should underestimate"
    assert abs(corrected - geodesic) <= abs(planar - geodesic) + 1.0


def test_scale_factor_is_one_at_the_standard_parallel():
    assert scale_factor(70.0) == pytest.approx(1.0, abs=1e-6)


def test_scale_factor_decreases_going_north():
    lats = np.array([70.0, 75.0, 80.0, 85.0])
    k = np.asarray(scale_factor(lats))
    assert np.all(np.diff(k) < 0)
    assert k[-1] == pytest.approx(0.9717, abs=1e-3)


def test_scale_factor_accepts_scalars_and_arrays():
    assert isinstance(scale_factor(75.0), float)
    out = scale_factor(np.array([[70.0, 75.0], [80.0, 85.0]]))
    assert out.shape == (2, 2)


# ---------------------------------------------------------------------------
# Grid mechanics
# ---------------------------------------------------------------------------


def test_grid_row_zero_is_north():
    """Raster convention. Getting it backwards flips every map in the demo."""
    grid = build_target_grid(np.array([0.0, 1000.0]), np.array([0.0, 1000.0]), 500.0)
    assert grid.y[0] > grid.y[-1]


def _grid_covering(lon0, lon1, lat0, lat1, res=500.0):
    """Grid covering a lat/lon box.

    Built from a dense mesh, not from two corners: a lat/lon rectangle is not a
    rectangle in polar stereographic, so a bounding box taken from the corners
    alone does not contain the whole region.
    """
    lons, lats = np.meshgrid(np.linspace(lon0, lon1, 25), np.linspace(lat0, lat1, 25))
    x, y = TO_POLAR.transform(lons, lats)
    return build_target_grid(x, y, res)


def test_grid_roundtrip_index_and_coordinates():
    grid = _grid_covering(-20.0, -10.0, 76.0, 79.0)
    for lon, lat in [(-18.0, 77.0), (-12.0, 78.5)]:
        r, c = grid.index_of(lon, lat)
        lon2, lat2 = grid.lonlat_of(r, c)
        # Half a 500 m cell is at most ~0.005° of latitude.
        assert abs(lat2 - lat) < 0.01
        assert abs(lon2 - lon) < 0.05


def test_point_outside_grid_raises():
    """A waypoint off the scene must fail loudly, not clamp to the border."""
    grid = build_target_grid(
        *TO_POLAR.transform(np.array([-20.0, -19.0]), np.array([76.0, 76.5])), 500.0
    )
    with pytest.raises(ValueError, match="outside the grid"):
        grid.index_of(30.0, 60.0)


def test_grid_transform_is_north_up():
    grid = build_target_grid(np.array([0.0, 5000.0]), np.array([0.0, 5000.0]), 500.0)
    ox, xres, xskew, oy, yskew, yres = grid.transform
    assert xres == 500.0
    assert yres == -500.0
    assert xskew == yskew == 0.0


# ---------------------------------------------------------------------------
# GCP interpolation
# ---------------------------------------------------------------------------


def _synthetic_gcp(gh=21, gw=21, lat0=75.0, lat1=79.0, lon0=-27.0, lon1=-5.0):
    lat = np.linspace(lat1, lat0, gh)[:, None] * np.ones((1, gw))
    lon = np.ones((gh, 1)) * np.linspace(lon0, lon1, gw)[None, :]
    return lat, lon


def test_interpolation_reproduces_the_nodes():
    """At the GCP positions the interpolant must return the GCPs themselves."""
    lat, lon = _synthetic_gcp()
    h, w = 4996, 5246
    xf, yf = interpolate_gcp_grid(lat, lon, (h, w))

    x_gcp, y_gcp = TO_POLAR.transform(lon, lat)
    rows = np.linspace(0, h - 1, lat.shape[0]).round().astype(int)
    cols = np.linspace(0, w - 1, lat.shape[1]).round().astype(int)

    # Node k sits at pixel k*(h-1)/20, which is fractional for most k. Sampling
    # the full-resolution array forces those positions to integers, and half a
    # pixel is 40 m on the ground — so the tolerance is a property of the test,
    # not of the interpolation. The corners land on exact integers and are
    # checked tightly below.
    assert np.allclose(xf[np.ix_(rows, cols)], x_gcp, atol=100.0, rtol=0)
    assert np.allclose(yf[np.ix_(rows, cols)], y_gcp, atol=100.0, rtol=0)

    for r, c, gr, gc in [
        (0, 0, 0, 0),
        (0, w - 1, 0, -1),
        (h - 1, 0, -1, 0),
        (h - 1, w - 1, -1, -1),
    ]:
        assert xf[r, c] == pytest.approx(x_gcp[gr, gc], abs=1e-3)
        assert yf[r, c] == pytest.approx(y_gcp[gr, gc], abs=1e-3)


def test_interpolation_output_shape():
    lat, lon = _synthetic_gcp()
    xf, yf = interpolate_gcp_grid(lat, lon, (500, 600))
    assert xf.shape == yf.shape == (500, 600)


def test_interpolation_rejects_degenerate_grid():
    with pytest.raises(ValueError, match="too small"):
        interpolate_gcp_grid(np.array([[75.0]]), np.array([[-20.0]]), (100, 100))


def test_implied_pixel_spacing_matches_declaration():
    """The linear GCP-to-pixel correspondence must reproduce 80 m pixels.

    This is the check that established the mapping in the first place: if the
    node spacing implied 40 m or 160 m, the assumed correspondence would be
    wrong and every distance downstream would scale with it.
    """
    lat, lon = _synthetic_gcp()
    h, w = 4996, 5246
    xf, yf = interpolate_gcp_grid(lat, lon, (h, w))

    mid = h // 2
    step_x = abs(xf[mid, w // 2 + 100] - xf[mid, w // 2]) / 100
    lat_mid = 77.0
    ground = step_x / scale_factor(lat_mid)
    assert 60.0 < ground < 110.0, f"implied step {ground:.1f} m, expected ~80 m"


# ---------------------------------------------------------------------------
# Categorical resampling
# ---------------------------------------------------------------------------


def test_nearest_resampling_invents_no_class():
    """Bilinear between class 2 and class 4 would produce class 3.

    SOD is categorical: an interpolated value is an ice type the analyst never
    recorded, and it would flow straight into the POLARIS risk lookup.
    """
    from iceroute.routing.geocode import _resample_nearest

    n = 400
    xs = np.repeat(np.linspace(0.0, 4000.0, n), n)
    ys = np.tile(np.linspace(0.0, 4000.0, n), n)
    vals = np.where(xs < 2000.0, 2, 4).astype("uint8")

    grid = build_target_grid(xs, ys, 500.0)
    out = _resample_nearest(xs, ys, vals, grid, 255, "uint8")

    present = set(np.unique(out).tolist()) - {255}
    assert present <= {2, 4}, f"invented classes: {sorted(present - {2, 4})}"


def test_resampling_marks_uncovered_cells_with_fill():
    from iceroute.routing.geocode import _resample_nearest

    xs = np.array([0.0, 100.0])
    ys = np.array([0.0, 100.0])
    vals = np.array([3, 3], dtype="uint8")
    grid = GeoGrid(
        x=np.arange(-2000.0, 2001.0, 500.0),
        y=np.arange(2000.0, -2001.0, -500.0),
        crs=POLAR_CRS,
        resolution_m=500.0,
    )
    out = _resample_nearest(xs, ys, vals, grid, 255, "uint8")
    assert (out == 255).any()
    assert (out == 3).any()
