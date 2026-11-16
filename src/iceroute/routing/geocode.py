"""SAR geometry to a regular EPSG:3413 grid.

If this step is wrong, everything downstream of it is wrong too, so the
reasoning is written out.

An AI4Arctic scene is in SAR geometry: a pixel is not a constant ground
distance, and the grid is not north-aligned. Running A* on the raw array gives
distances that are wrong and a route that looks plausible and is false.

Three things had to be established against the real data rather than assumed:

1. **The GCP grid maps linearly onto the array.** The netCDF carries a coarse
   21x21 lat/lon grid. Measuring the geodesic step between adjacent nodes gives
   81.6 m and 80.1 m per pixel against a declared ``pixel_spacing`` of 80, so
   node ``(i,j)`` sits at pixel ``(i*(H-1)/(gh-1), j*(W-1)/(gw-1))``.

2. **No flip correction is needed**, despite the undocumented ``flip: 1``
   global attribute. Verified independently: under direct correspondence the
   array corners that the grid places on Greenland are exactly the corners
   where ``distance_map`` bottoms out, and the corners it places in the Fram
   Strait are where ``distance_map`` is high. A flipped interpretation would
   anti-correlate. See docs/decisions.md D-009.

3. **Planar distance in EPSG:3413 is not ground distance.** The projection is
   true-scale at 70°N only; at 78°N the scale factor is 0.9806, so a planar
   measurement understates the real distance by 1.9%, and by 2.8% at 85°N. That
   alone would fail the 1% distance-accuracy gate (scale-corrected planar
   distance within 1% of the WGS84 geodesic, enforced in the tests) and would
   bias every transit time. Use
   :func:`ground_distance_m`, never a bare hypot.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from functools import lru_cache

import numpy as np
from pyproj import Geod, Proj, Transformer

log = logging.getLogger(__name__)

# WGS 84 / NSIDC Sea Ice Polar Stereographic North. The standard for Arctic sea
# ice; true scale at 70°N, central meridian -45°.
POLAR_CRS = "EPSG:3413"
WGS84 = "EPSG:4326"

GEOD = Geod(ellps="WGS84")


@lru_cache(maxsize=4)
def _proj(crs: str = POLAR_CRS) -> Proj:
    return Proj(crs)


@lru_cache(maxsize=4)
def _to_polar(crs: str = POLAR_CRS) -> Transformer:
    return Transformer.from_crs(WGS84, crs, always_xy=True)


@lru_cache(maxsize=4)
def _to_wgs(crs: str = POLAR_CRS) -> Transformer:
    return Transformer.from_crs(crs, WGS84, always_xy=True)


def scale_factor(lat: np.ndarray | float, crs: str = POLAR_CRS) -> np.ndarray | float:
    """Point scale factor of the projection at the given latitude(s).

    For a conformal projection like polar stereographic the meridional and
    parallel scale factors are equal, so one number per point suffices.
    Computed by PROJ rather than from a spherical approximation.
    """
    p = _proj(crs)
    lat_arr = np.atleast_1d(np.asarray(lat, dtype="float64"))
    lon_arr = np.full_like(lat_arr, -45.0)  # scale is longitude-independent here
    k = np.asarray(p.get_factors(lon_arr, lat_arr).meridional_scale, dtype="float64")
    return float(k[0]) if np.isscalar(lat) or np.ndim(lat) == 0 else k.reshape(np.shape(lat))


def ground_distance_m(
    x0: float, y0: float, x1: float, y1: float, lat_mid: float, crs: str = POLAR_CRS
) -> float:
    """Ground distance between two projected points, scale-corrected.

    ``lat_mid`` is the latitude at which to evaluate the scale factor —
    normally the midpoint of the segment. For a 500 m cell the factor is
    effectively constant across the step.
    """
    planar = float(np.hypot(x1 - x0, y1 - y0))
    return planar / float(scale_factor(lat_mid, crs))


def geodesic_distance_m(lon0: float, lat0: float, lon1: float, lat1: float) -> float:
    """True geodesic distance on the WGS84 ellipsoid. The reference for tests."""
    return float(GEOD.inv(lon0, lat0, lon1, lat1)[2])


@dataclass
class GeoGrid:
    """A regular grid in a projected CRS.

    ``x`` and ``y`` are 1D cell-centre coordinates in metres. ``y`` is stored
    descending so that row 0 is the northern edge, matching raster convention.
    """

    x: np.ndarray
    y: np.ndarray
    crs: str
    resolution_m: float

    @property
    def shape(self) -> tuple[int, int]:
        return (len(self.y), len(self.x))

    @property
    def transform(self) -> tuple[float, float, float, float, float, float]:
        """GDAL-style affine transform for writing GeoTIFF."""
        half = self.resolution_m / 2
        return (self.x[0] - half, self.resolution_m, 0.0, self.y[0] + half, 0.0, -self.resolution_m)

    def meshgrid(self) -> tuple[np.ndarray, np.ndarray]:
        return np.meshgrid(self.x, self.y)

    def to_lonlat(self) -> tuple[np.ndarray, np.ndarray]:
        """Longitude and latitude of every cell centre."""
        xx, yy = self.meshgrid()
        lon, lat = _to_wgs(self.crs).transform(xx, yy)
        return lon, lat

    def latitudes(self) -> np.ndarray:
        return self.to_lonlat()[1]

    def scale_factors(self) -> np.ndarray:
        """Per-cell scale factor, for converting planar steps to ground metres."""
        return np.asarray(scale_factor(self.latitudes(), self.crs))

    def index_of(self, lon: float, lat: float) -> tuple[int, int]:
        """Nearest cell to a geographic point, as ``(row, col)``.

        Raises ``ValueError`` if the point falls outside the grid — a waypoint
        off the scene has to be an explicit failure, not a silent clamp to the
        border that would produce a route starting somewhere else.
        """
        x, y = _to_polar(self.crs).transform(lon, lat)
        if not (self.x.min() - self.resolution_m <= x <= self.x.max() + self.resolution_m):
            raise ValueError(f"Point ({lon:.3f}, {lat:.3f}) outside the grid in x")
        if not (self.y.min() - self.resolution_m <= y <= self.y.max() + self.resolution_m):
            raise ValueError(f"Point ({lon:.3f}, {lat:.3f}) outside the grid in y")
        col = int(np.argmin(np.abs(self.x - x)))
        row = int(np.argmin(np.abs(self.y - y)))
        return row, col

    def lonlat_of(self, row: int, col: int) -> tuple[float, float]:
        lon, lat = _to_wgs(self.crs).transform(self.x[col], self.y[row])
        return float(lon), float(lat)


def interpolate_gcp_grid(
    lat_gcp: np.ndarray, lon_gcp: np.ndarray, shape: tuple[int, int], crs: str = POLAR_CRS
) -> tuple[np.ndarray, np.ndarray]:
    """Expand the coarse GCP grid to full resolution, in projected coordinates.

    Interpolation happens on projected x/y rather than on lat/lon. Longitude is
    cyclic and converges at the pole, so interpolating it directly misbehaves
    near the dateline and near 90°N — precisely where Arctic scenes live.
    Projected metres are well behaved everywhere in the grid's domain.

    Returns full-resolution ``(x, y)`` arrays of ``shape``.
    """
    from scipy.interpolate import RectBivariateSpline

    gh, gw = lat_gcp.shape
    if gh < 2 or gw < 2:
        raise ValueError(f"GCP grid too small: {lat_gcp.shape}")

    x_gcp, y_gcp = _to_polar(crs).transform(lon_gcp, lat_gcp)

    h, w = shape
    # GCP node (i,j) sits at pixel (i*(h-1)/(gh-1), j*(w-1)/(gw-1)) — verified
    # against the declared 80 m pixel spacing, see module docstring.
    src_rows = np.linspace(0, h - 1, gh)
    src_cols = np.linspace(0, w - 1, gw)
    dst_rows = np.arange(h, dtype="float64")
    dst_cols = np.arange(w, dtype="float64")

    # Cubic where there are enough nodes, linear otherwise. A 21x21 grid over a
    # 5000-pixel scene is smooth, so cubic adds accuracy without ringing.
    k = 3 if min(gh, gw) > 3 else 1
    xf = RectBivariateSpline(src_rows, src_cols, x_gcp, kx=k, ky=k)(dst_rows, dst_cols)
    yf = RectBivariateSpline(src_rows, src_cols, y_gcp, kx=k, ky=k)(dst_rows, dst_cols)
    return xf, yf


def build_target_grid(
    x: np.ndarray, y: np.ndarray, resolution_m: float, crs: str = POLAR_CRS, pad_cells: int = 2
) -> GeoGrid:
    """Regular grid covering the extent of the projected swath, snapped to the resolution."""
    xmin = np.floor(np.nanmin(x) / resolution_m) * resolution_m - pad_cells * resolution_m
    xmax = np.ceil(np.nanmax(x) / resolution_m) * resolution_m + pad_cells * resolution_m
    ymin = np.floor(np.nanmin(y) / resolution_m) * resolution_m - pad_cells * resolution_m
    ymax = np.ceil(np.nanmax(y) / resolution_m) * resolution_m + pad_cells * resolution_m

    xs = np.arange(xmin, xmax + resolution_m, resolution_m)
    ys = np.arange(ymax, ymin - resolution_m, -resolution_m)  # descending: north first
    return GeoGrid(x=xs, y=ys, crs=crs, resolution_m=resolution_m)


@dataclass
class GeocodedScene:
    """A scene resampled onto a regular projected grid."""

    grid: GeoGrid
    sod: np.ndarray  # uint8, FILL_VALUE outside coverage
    mask_valid: np.ndarray  # bool
    sic: np.ndarray | None = None
    distance_to_land: np.ndarray | None = None
    scene_id: str = ""
    # Extra bands resampled onto *this* grid, by name. Empty unless asked for.
    extra: dict = field(default_factory=dict)

    @property
    def shape(self) -> tuple[int, int]:
        return self.grid.shape


def _resample_nearest(
    src_x: np.ndarray,
    src_y: np.ndarray,
    values: np.ndarray,
    grid: GeoGrid,
    fill,
    dtype: str,
) -> np.ndarray:
    """Nearest-neighbour resample of a categorical raster onto ``grid``.

    Nearest neighbour is not a performance choice: SOD is categorical, and
    bilinear interpolation between class 2 and class 4 would invent class 3,
    an ice type the chart never recorded.

    Implemented by binning source pixels into destination cells rather than
    with a KD-tree. The source is 26 million points; a tree query would cost
    minutes per scene, while binning is a single pass and is exact for a
    downsampling ratio this large (80 m source into 500 m cells).
    """
    res = grid.resolution_m
    x0 = grid.x[0] - res / 2
    y0 = grid.y[0] + res / 2
    nrow, ncol = grid.shape

    col = np.floor((src_x - x0) / res).astype("int64")
    row = np.floor((y0 - src_y) / res).astype("int64")

    inside = (col >= 0) & (col < ncol) & (row >= 0) & (row < nrow)
    flat = np.full(nrow * ncol, fill, dtype=dtype)
    if inside.any():
        idx = row[inside] * ncol + col[inside]
        # Later writes win. With an ~6x downsample each destination cell
        # receives ~39 source pixels; taking one of them is the definition of
        # nearest-neighbour subsampling and cannot invent a class.
        flat[idx] = values[inside]
    return flat.reshape(nrow, ncol)


def geocode_scene(
    scene,
    resolution_m: float = 500.0,
    crs: str = POLAR_CRS,
    subsample: int | None = None,
    extra: dict[str, np.ndarray] | None = None,
) -> GeocodedScene:
    """Project a :class:`~iceroute.data.loader.Scene` onto a regular grid.

    Parameters
    ----------
    scene
        A loaded scene.
    resolution_m
        Target cell size. Default 500 m: the native 80 m is pointless for
        routing and makes A* crawl.
    subsample
        Take every n-th source pixel before resampling. ``None`` picks a factor
        that keeps the source about twice as dense as the target, which is
        enough to leave no destination cell empty while cutting the work.
    extra
        Further source-shaped arrays to put on the same grid, returned in
        :attr:`GeocodedScene.extra`. This exists so a caller that wants, say,
        the SAR on the routing grid gets it from *this* call rather than
        rebuilding the grid itself — two grids built separately are two grids
        that can disagree, which is docs/decisions.md D-027.
    """
    from iceroute.data.loader import FILL_VALUE

    h, w = scene.shape
    xf, yf = interpolate_gcp_grid(scene.lat_grid, scene.lon_grid, (h, w), crs)

    if subsample is None:
        native = float(getattr(scene.meta, "pixel_spacing_m", 80.0)) or 80.0
        subsample = max(1, int(resolution_m / (2 * native)))

    sl = (slice(None, None, subsample), slice(None, None, subsample))
    sx, sy = xf[sl].ravel(), yf[sl].ravel()

    grid = build_target_grid(sx, sy, resolution_m, crs)

    sod = _resample_nearest(sx, sy, scene.sod[sl].ravel(), grid, FILL_VALUE, "uint8")
    valid = _resample_nearest(sx, sy, scene.mask_valid[sl].ravel(), grid, False, "bool")
    sic = _resample_nearest(sx, sy, scene.sic[sl].ravel(), grid, FILL_VALUE, "uint8")
    dist = _resample_nearest(sx, sy, scene.distance_to_land[sl].ravel(), grid, np.nan, "float32")

    # A cell can only be valid if it actually received a source pixel.
    valid &= sod != FILL_VALUE

    log.info(
        "geocoding %s: %dx%d @ %.0f m (source %dx%d @ %.0f m, subsample %d) — %.1f%% covered",
        scene.meta.scene_id,
        *grid.shape,
        resolution_m,
        h,
        w,
        getattr(scene.meta, "pixel_spacing_m", 80.0),
        subsample,
        100 * valid.mean(),
    )

    resampled = {}
    for name, band in (extra or {}).items():
        fill = np.nan if np.issubdtype(np.asarray(band).dtype, np.floating) else 0
        resampled[name] = _resample_nearest(
            sx, sy, np.asarray(band)[sl].ravel(), grid, fill, str(np.asarray(band).dtype)
        )

    return GeocodedScene(
        grid=grid,
        sod=sod,
        mask_valid=valid,
        sic=sic,
        distance_to_land=dist,
        scene_id=scene.meta.scene_id,
        extra=resampled,
    )
