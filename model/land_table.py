"""Build land tables: how much land in each 1-degree cell is suitable for wind and for PV.

The inputs are the MODIS land-cover class fractions (MCD12C1, 0.05 degrees), the World
Database on Protected Areas and the GEBCO terrain grid (data/README.md). For each cell centred
on an integer coordinate the builder

1. averages MODIS class fractions over the cell with exact spherical weights and applies the
   suitability factor of each class for wind and for PV (Salmon and Banares-Alcantara 2022,
   Table 1; Verschuur et al. 2024);
2. removes protected land (WDPA polygons) and, unless skipped, land steeper than
   ``max_slope_degrees`` (slopes computed from GEBCO), and records the share of the cell's land
   in each slope class (SLOPE_CLASSES), from which model/site_costs.py takes the slope factor
   of the build cost;
3. keeps a share of what remains (``land_competition_fraction``: 1.0 for the full build,
   0.2 or 0.02 for the derived tables);
4. converts areas to power: 200 km2/GW for wind, and for fixed PV the row packing of a First
   Solar module at the latitude's tilt (van de Ven et al. 2021).

The union of wind and PV land is the classwise nested lower bound described in land.py.

    python -m model.land_table --output <table.csv> --land-cover <MCD12C1.hdf> \
        --protected-area <WDPA_0.shp> --protected-area <WDPA_1.shp> --protected-area <WDPA_2.shp> \
        --slope-raster <GEBCO.nc> --max-slope-degrees 20
    python -m model.land_table --base-csv <100 % table> --output <20 % table> --land-competition-fraction 0.2

The full global build takes several hours (ARC job arc/jobs/00_build_land_constraints.sh); a
share variant takes seconds.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Tuple

import geopandas as gpd
import numpy as np
import pandas as pd
import xarray as xr

from .config import BATHYMETRY_FILE, EARTH_RADIUS_KM, LAND_COVER_FILE, SLOPE_CLASS_COLUMNS, SLOPE_CLASSES, repo_path
from .parameters import file_record, parameter
from .land import (
    CLASSWISE_NESTED_UNION_AREA_COLUMN,
    CLASSWISE_NESTED_UNION_AVAILABILITY_COLUMN,
    CLASSWISE_NESTED_UNION_METHOD,
    CLASSWISE_NESTED_UNION_VERSION,
    REQUIRED_CELL_ANCHOR,
    RENEWABLE_UNION_METHOD_COLUMN,
    RENEWABLE_UNION_METHOD_VERSION_COLUMN,
)

LOGGER = logging.getLogger(__name__)

MODIS_WATER_CLASS = 0
DEFAULT_LAND_COMPETITION_FRACTION = 1.0
# The land assumptions, with their sources: inputs/parameters.yaml, section land
WIND_LAND_USE_KM2_PER_GW = float(parameter("land.wind_land_use_km2_per_gw"))
# Land above the last slope class bound is excluded
DEFAULT_MAX_SLOPE_DEGREES = SLOPE_CLASSES[-2][1]
PROTECTED_AREA_STATUSES = frozenset(str(status).lower() for status in parameter("land.protected_area_statuses"))

# Fixed-tilt PV rows
FIRST_SOLAR_MODULE_WIDTH_M = float(parameter("land.solar_module_width_m"))
FIRST_SOLAR_MODULE_LENGTH_M = float(parameter("land.solar_module_length_m"))
FIRST_SOLAR_MODULE_POWER_KW = float(parameter("land.solar_module_power_kw"))
SOLAR_DECLINATION_DEG = float(parameter("physical.earth_axial_tilt_deg"))  # at the solstices
SOLAR_MIN_TILT_DEG = float(parameter("land.solar_min_tilt_deg"))
SOLAR_MAX_TILT_DEG = float(parameter("land.solar_max_tilt_deg"))
SOLAR_MIN_SUN_ALTITUDE_DEG = float(parameter("land.solar_min_sun_altitude_deg"))
SOLAR_GCR_MIN = float(parameter("land.solar_min_ground_cover_ratio"))
SOLAR_GCR_MAX = float(parameter("land.solar_max_ground_cover_ratio"))

# Share of each IGBP land-cover class (MODIS MCD12C1 Land Cover Type 1) that each technology may
# use; unlisted classes give none
MODIS_CLASS_WIND_AVAILABILITY = {int(k): float(v) for k, v in parameter("land.wind_class_availability").items()}
MODIS_CLASS_SOLAR_AVAILABILITY = {int(k): float(v) for k, v in parameter("land.solar_class_availability").items()}

FINAL_COLUMNS = [
    "latitude",
    "longitude",
    "cell_anchor",
    "availability",
    "area",
    "onshore_land_pct",
    "offshore_sea_pct",
    "onshore_area_km2",
    "offshore_area_km2",
    "elevation_m",
    "protected_area_pct",
    "slope_suitable_land_pct",
    "steep_slope_pct",
    *SLOPE_CLASS_COLUMNS,
    "land_exclusion_factor",
    "land_competition_fraction",
    "constrained_onshore_area_km2",
    "wind_onshore_availability",
    "wind_offshore_availability",
    "wind_availability",
    "solar_availability",
    CLASSWISE_NESTED_UNION_AVAILABILITY_COLUMN,
    "renewable_union_availability",
    RENEWABLE_UNION_METHOD_COLUMN,
    RENEWABLE_UNION_METHOD_VERSION_COLUMN,
    "solar_area_km2",
    "wind_area_km2",
    "wind_onshore_area_km2",
    "wind_offshore_area_km2",
    CLASSWISE_NESTED_UNION_AREA_COLUMN,
    "renewable_union_area_km2",
    "wind_density_mw_per_km2",
    "solar_density_mw_per_km2",
    "max_power_solar_mw",
    "max_power_wind_mw",
    "max_capacity_mw",
]


def _format_eta(seconds: float) -> str:
    total_seconds = int(round(max(float(seconds), 0.0)))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours:d}h {minutes:02d}m {secs:02d}s"
    if minutes:
        return f"{minutes:d}m {secs:02d}s"
    return f"{secs:d}s"


def _wind_availability_map(include_offshore_wind: bool) -> dict[int, float]:
    availability = dict(MODIS_CLASS_WIND_AVAILABILITY)
    if not include_offshore_wind:
        availability[MODIS_WATER_CLASS] = 0.0
    return availability


def wind_density(latitudes: pd.Series, scale: float = 1.0, land_use_km2_per_gw: float = WIND_LAND_USE_KM2_PER_GW) -> pd.Series:
    """Wind power density (MW/km2), the same everywhere."""
    base_density = 1000.0 / float(land_use_km2_per_gw)
    return pd.Series(np.full(latitudes.shape[0], base_density * scale), index=latitudes.index)


def _solar_density_raw(latitudes: np.ndarray) -> np.ndarray:
    lat_abs = np.abs(latitudes)
    beta = np.clip(lat_abs, SOLAR_MIN_TILT_DEG, SOLAR_MAX_TILT_DEG)
    solar_alt = np.clip(90.0 - lat_abs - SOLAR_DECLINATION_DEG, SOLAR_MIN_SUN_ALTITUDE_DEG, 89.0)
    beta_rad = np.deg2rad(beta)
    solar_alt_rad = np.deg2rad(solar_alt)
    row_pitch_m = FIRST_SOLAR_MODULE_LENGTH_M * (np.sin(beta_rad) + np.cos(beta_rad) / np.tan(solar_alt_rad))
    row_pitch_m = np.clip(row_pitch_m, FIRST_SOLAR_MODULE_LENGTH_M, np.inf)
    gcr = np.clip(FIRST_SOLAR_MODULE_WIDTH_M / row_pitch_m, SOLAR_GCR_MIN, SOLAR_GCR_MAX)
    module_area_m2 = FIRST_SOLAR_MODULE_WIDTH_M * FIRST_SOLAR_MODULE_LENGTH_M
    module_power_mw = FIRST_SOLAR_MODULE_POWER_KW / 1000.0
    area_per_mw_m2 = (module_area_m2 / module_power_mw) / gcr
    return 1_000_000.0 / area_per_mw_m2


def solar_density(latitudes: pd.Series, scale: float = 1.0, base_land_use_km2_per_mw: float | None = None) -> pd.Series:
    """Fixed-PV power density (MW/km2) from row packing at each latitude.

    With ``base_land_use_km2_per_mw`` the curve is scaled so that its equator value is
    1 / base_land_use (the technology YAML's land use).
    """
    density_mw_per_km2 = _solar_density_raw(latitudes.to_numpy())
    if base_land_use_km2_per_mw is not None:
        base_land_use = float(base_land_use_km2_per_mw)
        if base_land_use <= 0:
            raise ValueError("base_land_use_km2_per_mw must be positive")
        model_equator_density = float(_solar_density_raw(np.array([0.0]))[0])
        if model_equator_density > 0:
            density_mw_per_km2 *= (1.0 / base_land_use) / model_equator_density
    return pd.Series(density_mw_per_km2 * scale, index=latitudes.index)


@dataclass
class LandAvailabilityConfig:
    """Inputs and settings of one land build."""

    output_csv: Path
    land_cover_path: Path = LAND_COVER_FILE
    bathymetry_path: Path = BATHYMETRY_FILE
    coarse_degree: float = 1.0
    fine_degree: float = 0.05
    lat_bounds: Tuple[float, float] = (-75.0, 75.0)
    include_offshore_wind: bool = False
    land_competition_fraction: float = DEFAULT_LAND_COMPETITION_FRACTION
    protected_area_paths: Tuple[Path, ...] = tuple()
    slope_raster_path: Path | None = None
    max_slope_degrees: float = DEFAULT_MAX_SLOPE_DEGREES
    skip_slope_exclusion: bool = False

    def resolved(self) -> "LandAvailabilityConfig":
        return LandAvailabilityConfig(
            output_csv=repo_path(self.output_csv),
            land_cover_path=repo_path(self.land_cover_path),
            bathymetry_path=repo_path(self.bathymetry_path),
            coarse_degree=self.coarse_degree,
            fine_degree=self.fine_degree,
            lat_bounds=self.lat_bounds,
            include_offshore_wind=bool(self.include_offshore_wind),
            land_competition_fraction=float(self.land_competition_fraction),
            protected_area_paths=tuple(repo_path(path) for path in self.protected_area_paths),
            slope_raster_path=repo_path(self.slope_raster_path) if self.slope_raster_path is not None else None,
            max_slope_degrees=float(self.max_slope_degrees),
            skip_slope_exclusion=bool(self.skip_slope_exclusion),
        )


# ---------------------------------------------------------------------------------------------
# Cell geometry. A cell labelled (lat, lon) covers [lat - 0.5, lat + 0.5) x [lon - 0.5, lon + 0.5),
# like the weather nodes; the cell at longitude -180 straddles the dateline.
# ---------------------------------------------------------------------------------------------

def _half_cell(coarse_degree: float) -> float:
    """Degrees from a cell's label (its centre) to its south and west edges."""
    return 0.5 * float(coarse_degree)


def _bin_coordinates(values, coarse_degree: float) -> np.ndarray:
    """Label of the cell containing each coordinate (pixel centres expected)."""
    return np.floor((np.asarray(values, dtype=float) + _half_cell(coarse_degree)) / coarse_degree) * coarse_degree


def _wrap_longitude_labels(values) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    return np.where(values >= 180.0, values - 360.0, values)


def _cell_area_km2(latitudes: pd.Series, delta_deg: float) -> pd.Series:
    south = latitudes.to_numpy(dtype=float) - _half_cell(delta_deg)
    band_height = np.sin(np.deg2rad(south + delta_deg)) - np.sin(np.deg2rad(south))
    return pd.Series(np.abs((EARTH_RADIUS_KM ** 2) * np.deg2rad(delta_deg) * band_height), index=latitudes.index)


# ---------------------------------------------------------------------------------------------
# Land cover
# ---------------------------------------------------------------------------------------------

def _reduce_band(field: np.ndarray, group_size: int, row_weights: np.ndarray) -> np.ndarray:
    """Area-weighted mean of a (group_size, n_fine_lon) band for each coarse column.

    Fine rows are weighted by their spherical area (row_weights); fine columns within a coarse
    cell all have the same area, so a plain mean applies there.
    """
    blocks = field.reshape(group_size, -1, group_size)
    return np.tensordot(row_weights, blocks, axes=([0], [0])).mean(axis=1)


def _aggregate_availability_from_hdf4(config: LandAvailabilityConfig) -> pd.DataFrame:
    """Aggregate MODIS land-cover suitability directly from the HDF4 input.

    Class fractions are averaged over each coarse cell with exact spherical
    row weights, so a cell's availability is an area fraction, not a pixel
    fraction (identical to the legacy land step's per-pixel weighting).

    Some netCDF4 builds (common on macOS/Homebrew) do not enable the HDF4 feature
    set, which makes `xr.open_dataset(..., engine="netcdf4")` fail for MODIS .hdf
    inputs. This fallback uses `pyhdf` (HDF4) and aggregates straight to the 1° grid
    without expanding to a massive long-form dataframe.
    """

    try:
        from pyhdf.SD import SD, SDC
    except Exception as exc:  # pragma: no cover
        raise ImportError(
            "Reading MODIS .hdf requires 'pyhdf' when netCDF4 lacks HDF4 support. "
            "Install pyhdf (and system HDF4 libs) or use a netCDF4 build with HDF4 enabled."
        ) from exc

    if config.fine_degree <= 0 or config.coarse_degree <= 0:
        raise ValueError("fine_degree and coarse_degree must be positive")

    group_size = int(round(config.coarse_degree / config.fine_degree))
    if not np.isclose(group_size * config.fine_degree, config.coarse_degree):
        raise ValueError(
            "coarse_degree must be an integer multiple of fine_degree for HDF4 aggregation "
            f"(coarse_degree={config.coarse_degree}, fine_degree={config.fine_degree})."
        )

    factors: np.ndarray
    water_class = int(MODIS_WATER_CLASS)
    sd = SD(str(config.land_cover_path), SDC.READ)
    try:
        sds = sd.select("Land_Cover_Type_1_Percent")
        data_name, rank, dims, _dtype, _nattrs = sds.info()
        if rank != 3:
            raise ValueError(
                f"Unexpected MODIS variable shape for {data_name!r}: rank={rank}, dims={dims}"
            )

        n_y, n_x, n_classes = dims

        wind_map = _wind_availability_map(config.include_offshore_wind)
        wind_factors = np.zeros(int(n_classes), dtype=float)
        wind_onshore_factors = np.zeros(int(n_classes), dtype=float)
        for klass, weight in wind_map.items():
            if 0 <= int(klass) < n_classes:
                wind_factors[int(klass)] = float(weight)
                if int(klass) != water_class:
                    wind_onshore_factors[int(klass)] = float(weight)

        solar_factors = np.zeros(int(n_classes), dtype=float)
        for klass, weight in MODIS_CLASS_SOLAR_AVAILABILITY.items():
            if 0 <= int(klass) < n_classes:
                solar_factors[int(klass)] = float(weight)
        # With no sub-class spatial masks, max(...) is the minimum possible
        # wind/solar union: it assumes the smaller eligible footprint is fully
        # nested inside the larger footprint for every MODIS class.
        renewable_union_factors = np.maximum(wind_onshore_factors, solar_factors)

        if not (0 <= water_class < n_classes):
            raise ValueError(
                f"Water class index {water_class} out of bounds for n_classes={n_classes}."
            )

        # Fine row y spans latitudes [90 - fine * (y + 1), 90 - fine * y] and fine
        # column x spans longitudes [-180 + fine * x, -180 + fine * (x + 1)].  A
        # coarse band of group_size rows starting at y0 covers [top - coarse, top]
        # with top = 90 - fine * y0.  Bands are shifted by half a cell so that the
        # label sits at the cell centre, and the first coarse column then wraps
        # the dateline.
        if group_size % 2:
            raise ValueError(
                "Centred cells need an even coarse/fine ratio so that cell edges "
                "fall on MODIS pixel edges."
            )
        shift = group_size // 2
        lat_min, lat_max = config.lat_bounds
        band_starts = np.arange(shift, n_y - group_size + 1, group_size)
        band_tops = 90.0 - config.fine_degree * band_starts
        band_labels = (
            band_tops - config.coarse_degree + _half_cell(config.coarse_degree)
        )
        keep = (band_labels >= lat_min - 1e-9) & (band_labels <= lat_max + 1e-9)
        band_starts = band_starts[keep]
        band_labels = band_labels[keep]

        x_end = (n_x // group_size) * group_size
        if x_end == 0:
            raise ValueError("Longitude dimension too small for requested aggregation.")
        if x_end != n_x:
            raise ValueError(
                "Centred cells need the MODIS longitude axis to divide exactly into coarse cells."
            )

        lon_coarse = -180.0 + config.coarse_degree * np.arange(x_end // group_size)

        rows: list[dict[str, float | str]] = []

        for y0, lat_coarse in zip(band_starts, band_labels):
            # Read a single coarse latitude band (group_size rows) for all longitudes/classes.
            cube = sds[int(y0) : int(y0) + group_size, 0:x_end, :]
            cube = np.asarray(cube, dtype=float) / 100.0
            # Rotate the fine columns eastward so that coarse column 0 covers
            # [-180 - coarse / 2, -180 + coarse / 2] across the dateline.
            cube = np.roll(cube, shift, axis=1)
            # Fine row r of the band covers [top - fine * (r + 1), top - fine * r].
            top = 90.0 - config.fine_degree * float(y0)
            row_edges = np.deg2rad(top - config.fine_degree * np.arange(group_size + 1))
            row_weights = np.sin(row_edges[:-1]) - np.sin(row_edges[1:])
            row_weights = row_weights / row_weights.sum()

            # Weighted availability: sum_class(class_fraction * factor)
            wind_component_fine = np.tensordot(cube, wind_factors, axes=([2], [0]))
            solar_component_fine = np.tensordot(cube, solar_factors, axes=([2], [0]))
            wind_onshore_fine = np.tensordot(cube, wind_onshore_factors, axes=([2], [0]))
            renewable_union_fine = np.tensordot(
                cube,
                renewable_union_factors,
                axes=([2], [0]),
            )
            water_fraction_fine = cube[:, :, water_class]
            wind_offshore_fine = (
                water_fraction_fine if config.include_offshore_wind else np.zeros_like(water_fraction_fine)
            )

            # Reduce to coarse cells: area-weighted mean across fine y and fine x.
            # Shapes: (group_size, x_end) -> (n_lon_coarse,)
            wind_band = _reduce_band(wind_component_fine, group_size, row_weights)
            wind_onshore_band = _reduce_band(wind_onshore_fine, group_size, row_weights)
            wind_offshore_band = _reduce_band(wind_offshore_fine, group_size, row_weights)
            solar_band = _reduce_band(solar_component_fine, group_size, row_weights)
            renewable_union_band = _reduce_band(renewable_union_fine, group_size, row_weights)
            water_band = _reduce_band(water_fraction_fine, group_size, row_weights)

            for (
                lon,
                wind_val,
                wind_onshore_val,
                wind_offshore_val,
                solar_val,
                renewable_union_val,
                water_val,
            ) in zip(
                lon_coarse,
                wind_band,
                wind_onshore_band,
                wind_offshore_band,
                solar_band,
                renewable_union_band,
                water_band,
            ):
                wind_val = float(np.clip(wind_val, 0.0, 1.0))
                wind_onshore_val = float(np.clip(wind_onshore_val, 0.0, 1.0))
                wind_offshore_val = float(np.clip(wind_offshore_val, 0.0, 1.0))
                solar_val = float(np.clip(solar_val, 0.0, 1.0))
                renewable_union_val = float(np.clip(renewable_union_val, 0.0, 1.0))
                water_val = float(np.clip(water_val, 0.0, 1.0))
                onshore = float(np.clip(1.0 - water_val, 0.0, 1.0))
                avail = float(max(wind_val, solar_val))
                rows.append(
                    {
                        "latitude": float(lat_coarse),
                        "longitude": float(lon),
                        "wind_onshore_availability": wind_onshore_val,
                        "wind_offshore_availability": wind_offshore_val,
                        "wind_availability": wind_val,
                        "solar_availability": solar_val,
                        CLASSWISE_NESTED_UNION_AVAILABILITY_COLUMN: renewable_union_val,
                        "renewable_union_availability": renewable_union_val,
                        RENEWABLE_UNION_METHOD_COLUMN: CLASSWISE_NESTED_UNION_METHOD,
                        RENEWABLE_UNION_METHOD_VERSION_COLUMN: CLASSWISE_NESTED_UNION_VERSION,
                        "onshore_land_pct": onshore * 100.0,
                        "availability": avail,
                    }
                )
    finally:
        sd.end()

    df = pd.DataFrame.from_records(rows)
    if df.empty:
        raise ValueError("No cells produced; check lat_bounds and input grid assumptions.")

    return df


def _attach_bathymetry_depth(df: pd.DataFrame, bathymetry_path: Path) -> pd.DataFrame:
    """Add elevation_m (positive above sea level) sampled at each cell centre."""
    if not bathymetry_path.exists():
        LOGGER.warning("Bathymetry file %s not found; offshore capacities will be zero.", bathymetry_path)
        output = df.copy()
        output["elevation_m"] = np.nan
        return output
    with xr.open_dataset(bathymetry_path) as dataset:
        variable_name = "depths" if "depths" in dataset.data_vars else next(iter(dataset.data_vars))
        centre_lon = df["longitude"].to_numpy(dtype=float)
        centre_lon = np.where(centre_lon >= 180.0, centre_lon - 360.0, centre_lon)
        points = xr.Dataset(coords={
            "points": np.arange(len(df)),
            "latitude": ("points", df["latitude"].to_numpy(dtype=float)),
            "longitude": ("points", centre_lon),
        })
        selected = dataset[variable_name].sel(latitude=points["latitude"], longitude=points["longitude"], method="nearest")
    output = df.copy()
    # The raw data are positive for ocean depth and negative for land elevation
    output["elevation_m"] = -selected.to_numpy().astype(float)
    return output


# ---------------------------------------------------------------------------------------------
# Protected areas
# ---------------------------------------------------------------------------------------------

# WDPA polygons are rasterised on a grid this many times finer than the land cover (0.005
# degrees for MODIS's 0.05): a sub-pixel is protected when its centre lies inside a polygon.
PROTECTED_SUBPIXELS = 10
# Largest number of sub-pixel centres tested against one polygon at once
_PROTECTED_POINT_BLOCK = 4_000_000


def _valid_geometries(frame: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    frame = frame[frame.geometry.notna() & ~frame.geometry.is_empty].copy()
    if frame.empty:
        return frame
    try:
        frame = frame.set_geometry(frame.geometry.make_valid())
    except AttributeError:  # pragma: no cover - depends on shapely/geopandas version
        frame = frame.set_geometry(frame.geometry.buffer(0))
    return frame[frame.geometry.notna() & ~frame.geometry.is_empty]


def _filter_protected_areas(frame: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    if frame.empty:
        return frame

    output = frame.copy()
    if "STATUS" in output.columns:
        status = output["STATUS"].astype(str).str.lower().str.strip()
        output = output[status.isin(PROTECTED_AREA_STATUSES)]

    if "REALM" in output.columns:
        realm = output["REALM"].astype(str).str.lower().str.strip()
        output = output[realm != "marine"]
    elif "MARINE" in output.columns:
        marine = output["MARINE"].astype(str).str.lower().str.strip()
        output = output[~marine.isin({"2", "true", "marine"})]

    return _valid_geometries(output)


def _rasterise_polygons(geometries, north: float, n_rows: int, n_cols: int, step: float) -> np.ndarray:
    """Boolean grid (north-up, from -180 degrees) of the sub-pixels whose centre lies in a polygon."""
    import shapely

    mask = np.zeros((n_rows, n_cols), dtype=bool)
    south = north - n_rows * step
    for geometry in geometries:
        min_x, min_y, max_x, max_y = geometry.bounds
        min_y, max_y = max(min_y, south), min(max_y, north)
        if max_y <= min_y:
            continue
        first_col = max(int(np.ceil((min_x + 180.0) / step - 0.5)), 0)
        last_col = min(int(np.floor((max_x + 180.0) / step - 0.5)), n_cols - 1)
        first_row = max(int(np.ceil((north - max_y) / step - 0.5)), 0)
        last_row = min(int(np.floor((north - min_y) / step - 0.5)), n_rows - 1)
        if last_col < first_col or last_row < first_row:
            continue
        shapely.prepare(geometry)
        xs = -180.0 + step * (np.arange(first_col, last_col + 1) + 0.5)
        rows_per_block = max(1, _PROTECTED_POINT_BLOCK // len(xs))
        for block_start in range(first_row, last_row + 1, rows_per_block):
            block_end = min(block_start + rows_per_block, last_row + 1)
            ys = north - step * (np.arange(block_start, block_end) + 0.5)
            grid_x, grid_y = np.meshgrid(xs, ys)
            inside = shapely.contains_xy(geometry, grid_x, grid_y)
            mask[block_start:block_end, first_col:last_col + 1] |= inside
    return mask


def _protected_land_pct(
    df: pd.DataFrame,
    protected_area_paths: Tuple[Path, ...],
    land_cover_path: Path,
    coarse_degree: float,
    fine_degree: float,
) -> pd.DataFrame:
    """Protected land as a percentage of each cell's area, measured on the land cover's grid.

    Each land-cover pixel's protected share (WDPA polygons rasterised at fine_degree /
    PROTECTED_SUBPIXELS) is weighted by the pixel's land fraction (one minus its MODIS water
    fraction) and its spherical area. Water inside a protected area, the sea of a coastal park or
    a lake, therefore does not count against the cell's land, and protected land never exceeds
    the land the table counts. (Until October 2026 the whole protected area, water included, was
    divided by the cell's land, which removed the land of hundreds of coastal cells.)
    """
    output = df.copy()
    output["protected_area_pct"] = 0.0
    if not protected_area_paths:
        return output
    for path in protected_area_paths:
        if not path.exists():
            raise FileNotFoundError(f"Protected-area vector file not found: {path}")
    from pyhdf.SD import SD, SDC

    group = int(round(coarse_degree / fine_degree))
    half = group // 2
    step = fine_degree / PROTECTED_SUBPIXELS
    n_cols = int(round(360.0 / fine_degree))
    labels = np.sort(output["latitude"].astype(float).unique())
    chunk_degree = max(coarse_degree, 5.0)
    chunks = [labels[(labels >= edge) & (labels < edge + chunk_degree)]
              for edge in np.arange(labels.min(), labels.max() + chunk_degree, chunk_degree)]
    chunks = [chunk for chunk in chunks if len(chunk)]
    LOGGER.info("Protected land: %d latitude chunks across %d WDPA source files.", len(chunks), len(protected_area_paths))
    stage_started = time.perf_counter()
    results = []
    sd = SD(str(land_cover_path), SDC.READ)
    try:
        sds = sd.select("Land_Cover_Type_1_Percent")
        for index, chunk in enumerate(chunks, start=1):
            chunk_started = time.perf_counter()
            # the chunk's cells as a contiguous band of whole cells
            cell_rows = np.arange(chunk.max(), chunk.min() - coarse_degree / 2, -coarse_degree)
            north = float(chunk.max()) + _half_cell(coarse_degree)
            south = float(chunk.min()) - _half_cell(coarse_degree)
            first_row = int(round((90.0 - north) / fine_degree))
            last_row = int(round((90.0 - south) / fine_degree))
            water = np.asarray(sds[first_row:last_row, 0:n_cols, MODIS_WATER_CLASS:MODIS_WATER_CLASS + 1],
                               dtype=float)[:, :, 0] / 100.0

            geometries = []
            for path in protected_area_paths:
                protected = gpd.read_file(path, bbox=(-180.0, south, 180.0, north))
                if protected.empty:
                    continue
                protected = protected.set_crs("EPSG:4326") if protected.crs is None else protected.to_crs("EPSG:4326")
                geometries.extend(_filter_protected_areas(protected).geometry)
            mask = _rasterise_polygons(geometries, north, (last_row - first_row) * PROTECTED_SUBPIXELS,
                                       n_cols * PROTECTED_SUBPIXELS, step)
            share = mask.reshape(last_row - first_row, PROTECTED_SUBPIXELS, n_cols, PROTECTED_SUBPIXELS).mean(axis=(1, 3))

            row_north = north - fine_degree * np.arange(last_row - first_row)
            pixel_km2 = (EARTH_RADIUS_KM ** 2) * np.deg2rad(fine_degree) * (
                np.sin(np.deg2rad(row_north)) - np.sin(np.deg2rad(row_north - fine_degree)))
            protected_land = share * (1.0 - water).clip(0.0, 1.0) * pixel_km2[:, None]
            # coarse column b covers [b - 180.5, b - 179.5): roll by half a cell, as the land cover does
            protected_land = np.roll(protected_land, half, axis=1)
            per_cell = protected_land.reshape(len(cell_rows), group, -1, group).sum(axis=(1, 3))
            areas = _cell_area_km2(pd.Series(cell_rows), coarse_degree).to_numpy()
            lon_labels = -180.0 + coarse_degree * np.arange(per_cell.shape[1])
            for row, lat in enumerate(cell_rows):
                results.append(pd.DataFrame({"latitude": float(lat), "longitude": lon_labels,
                                             "protected_area_pct": 100.0 * per_cell[row] / areas[row]}))
            elapsed = time.perf_counter() - stage_started
            LOGGER.info(
                "Protected-land chunk %d/%d lat [%.1f, %.1f): %d polygons, chunk=%s, elapsed=%s, eta=%s",
                index, len(chunks), south, north, len(geometries),
                _format_eta(time.perf_counter() - chunk_started), _format_eta(elapsed),
                _format_eta(elapsed / index * (len(chunks) - index)),
            )
    finally:
        sd.end()

    table = pd.concat(results, ignore_index=True)
    output = output.drop(columns=["protected_area_pct"]).merge(table, on=["latitude", "longitude"], how="left")
    output["protected_area_pct"] = output["protected_area_pct"].fillna(0.0).clip(0.0, 100.0)
    return output


# ---------------------------------------------------------------------------------------------
# Slope
# ---------------------------------------------------------------------------------------------

def _coord_name(dataset: xr.Dataset, candidates: tuple[str, ...]) -> str:
    lower = {name.lower(): name for name in list(dataset.coords) + list(dataset.dims)}
    for candidate in candidates:
        match = lower.get(candidate.lower())
        if match is not None:
            return match
    raise ValueError(f"Could not identify coordinate from candidates: {candidates}")


def _coord_slice(values: np.ndarray, lower: float, upper: float) -> slice:
    if values[0] <= values[-1]:
        return slice(lower, upper)
    return slice(upper, lower)


def _slope_suitable_land_pct_from_raster(
    df: pd.DataFrame,
    raster_path: Path,
    coarse_degree: float,
    max_slope_degrees: float,
) -> pd.DataFrame:
    if not raster_path.exists():
        raise FileNotFoundError(f"Slope/elevation raster not found: {raster_path}")
    offset_deg = _half_cell(coarse_degree)

    frames: list[pd.DataFrame] = []
    requested_lats = sorted(float(value) for value in df["latitude"].dropna().unique())
    LOGGER.info(
        "Slope screening: %d latitude rows from %s with max slope %.1f degrees.",
        len(requested_lats),
        raster_path,
        max_slope_degrees,
    )
    stage_started = time.perf_counter()

    with xr.open_dataset(raster_path) as dataset:
        lat_name = _coord_name(dataset, ("lat", "latitude", "y"))
        lon_name = _coord_name(dataset, ("lon", "longitude", "x"))
        variable_name = "elevation" if "elevation" in dataset.data_vars else next(
            name for name in dataset.data_vars if name.lower() != "crs"
        )
        elevation = dataset[variable_name]

        lat_values = dataset[lat_name].values.astype(float)
        lon_values = dataset[lon_name].values.astype(float)
        dlat = abs(float(np.median(np.diff(lat_values))))
        dlon = abs(float(np.median(np.diff(lon_values))))
        rows_per_cell = int(round(coarse_degree / dlat))
        cols_per_cell = int(round(coarse_degree / dlon))
        if (
            rows_per_cell <= 0
            or cols_per_cell <= 0
            or not np.isclose(rows_per_cell * dlat, coarse_degree, rtol=1e-4, atol=1e-6)
            or not np.isclose(cols_per_cell * dlon, coarse_degree, rtol=1e-4, atol=1e-6)
        ):
            raise ValueError(
                "Slope raster resolution must divide evenly into the coarse grid "
                f"(coarse_degree={coarse_degree}, dlat={dlat}, dlon={dlon})."
            )

        n_lon = (len(lon_values) // cols_per_cell) * cols_per_cell
        if len(lon_values) > 1 and lon_values[1] < lon_values[0]:
            raise ValueError("Slope raster longitudes must be ascending.")
        cols_shift = int(round(offset_deg / dlon))
        if n_lon != len(lon_values):
            raise ValueError(
                "Centred cells need the slope raster longitude axis to divide exactly into coarse cells."
            )
        # After rolling the columns eastward by cols_shift, coarse column b holds
        # raster columns [b * cols_per_cell - cols_shift, (b + 1) * cols_per_cell - cols_shift).
        column_lons = lon_values[0] + dlon * (np.arange(n_lon) - cols_shift)
        lon_cells = _wrap_longitude_labels(
            _bin_coordinates(column_lons.reshape(-1, cols_per_cell).mean(axis=1), coarse_degree)
        )
        dy_m = EARTH_RADIUS_KM * 1000.0 * np.deg2rad(dlat)

        for lat_index, cell_lat in enumerate(requested_lats, start=1):
            lat_lower = cell_lat - offset_deg
            lat_upper = lat_lower + coarse_degree
            pad_lower = lat_lower - dlat * 1.5
            pad_upper = lat_upper + dlat * 1.5

            band = elevation.sel({lat_name: _coord_slice(lat_values, pad_lower, pad_upper)})
            band = band.isel({lon_name: slice(0, n_lon)}).transpose(lat_name, lon_name)
            band_lats = band[lat_name].values.astype(float)
            center_mask = (band_lats >= lat_lower) & (band_lats < lat_upper)
            if center_mask.sum() == 0:
                continue

            elevations = band.to_numpy().astype(np.float32)
            elevations = np.roll(elevations, cols_shift, axis=1)
            grad_y = np.gradient(elevations, axis=0) / dy_m
            dx_m = (
                EARTH_RADIUS_KM
                * 1000.0
                * np.cos(np.deg2rad(band_lats))
                * np.deg2rad(dlon)
            )
            dx_m = np.maximum(dx_m, 1.0)
            grad_x = np.gradient(elevations, axis=1) / dx_m[:, None]
            slope = np.rad2deg(np.arctan(np.hypot(grad_x, grad_y)))

            land = elevations[center_mask, :n_lon] > 0
            cell_slope = slope[center_mask, :n_lon]

            def land_pct(mask: np.ndarray) -> np.ndarray:
                counts = mask.reshape(mask.shape[0], -1, cols_per_cell).sum(axis=(0, 2))
                return 100.0 * np.divide(
                    counts, land_counts, out=np.zeros_like(counts, dtype=float), where=land_counts > 0
                )

            land_counts = land.reshape(land.shape[0], -1, cols_per_cell).sum(axis=(0, 2))
            row = {
                "latitude": cell_lat,
                "longitude": lon_cells,
                "slope_suitable_land_pct": land_pct(land & (cell_slope <= max_slope_degrees)),
            }
            lower_bound = -np.inf
            for column, upper_bound in SLOPE_CLASSES:
                row[column] = land_pct(land & (cell_slope > lower_bound) & (cell_slope <= upper_bound))
                lower_bound = upper_bound
            frames.append(pd.DataFrame(row))

            if lat_index == 1 or lat_index % 10 == 0 or lat_index == len(requested_lats):
                elapsed = time.perf_counter() - stage_started
                eta = (elapsed / lat_index) * (len(requested_lats) - lat_index)
                LOGGER.info(
                    "Slope row %d/%d (lat %.0f): elapsed=%s, eta=%s",
                    lat_index,
                    len(requested_lats),
                    cell_lat,
                    _format_eta(elapsed),
                    _format_eta(eta),
                )

    output = df.drop(columns=["slope_suitable_land_pct", "steep_slope_pct", *SLOPE_CLASS_COLUMNS], errors="ignore")
    if frames:
        slope_df = pd.concat(frames, ignore_index=True)
        output = output.merge(slope_df, on=["latitude", "longitude"], how="left")
    else:
        for column in ("slope_suitable_land_pct", *SLOPE_CLASS_COLUMNS):
            output[column] = np.nan
    # Cells outside the latitude rows read count as flat
    missing = output["slope_suitable_land_pct"].isna()
    output["slope_suitable_land_pct"] = output["slope_suitable_land_pct"].fillna(100.0).clip(0.0, 100.0)
    output["steep_slope_pct"] = 100.0 - output["slope_suitable_land_pct"]
    for index, column in enumerate(SLOPE_CLASS_COLUMNS):
        output.loc[missing, column] = 100.0 if index == 0 else 0.0
    return output


# ---------------------------------------------------------------------------------------------
# Exclusions, land share and the table
# ---------------------------------------------------------------------------------------------

def _apply_land_exclusions(availability: pd.DataFrame, cfg: LandAvailabilityConfig) -> pd.DataFrame:
    output = _protected_land_pct(
        availability.copy(),
        cfg.protected_area_paths,
        cfg.land_cover_path,
        cfg.coarse_degree,
        cfg.fine_degree,
    )
    if cfg.skip_slope_exclusion or cfg.slope_raster_path is None:
        output["slope_suitable_land_pct"] = 100.0
        output["steep_slope_pct"] = 0.0
        for column in SLOPE_CLASS_COLUMNS:  # unknown without the terrain grid
            output[column] = np.nan
    else:
        output = _slope_suitable_land_pct_from_raster(output, cfg.slope_raster_path, cfg.coarse_degree, cfg.max_slope_degrees)

    onshore_fraction = (output["onshore_land_pct"] / 100.0).clip(0.0, 1.0)
    protected_fraction = (output["protected_area_pct"] / 100.0).clip(0.0, 1.0)
    protected_factor = pd.Series(0.0, index=output.index)
    onshore_mask = onshore_fraction > 0
    protected_factor.loc[onshore_mask] = 1.0 - protected_fraction.loc[onshore_mask] / onshore_fraction.loc[onshore_mask]
    protected_factor = protected_factor.clip(0.0, 1.0)

    slope_factor = (output["slope_suitable_land_pct"] / 100.0).clip(0.0, 1.0)
    output["land_exclusion_factor"] = (protected_factor * slope_factor).clip(0.0, 1.0)
    output["constrained_onshore_area_km2"] = output["onshore_area_km2"] * output["land_exclusion_factor"]

    output["solar_availability"] = (output["solar_availability"] * output["land_exclusion_factor"]).clip(0.0, 1.0)
    output["wind_onshore_availability"] = (output["wind_onshore_availability"] * output["land_exclusion_factor"]).clip(0.0, 1.0)
    for union_column in (CLASSWISE_NESTED_UNION_AVAILABILITY_COLUMN, "renewable_union_availability"):
        if union_column in output.columns:
            output[union_column] = (output[union_column] * output["land_exclusion_factor"]).clip(0.0, 1.0)
    output["wind_offshore_availability"] = output["wind_offshore_availability"].clip(0.0, 1.0)
    output["wind_availability"] = (output["wind_onshore_availability"] + output["wind_offshore_availability"]).clip(0.0, 1.0)
    output["availability"] = output[["wind_availability", "solar_availability"]].max(axis=1)
    return output


def _uniform_land_competition_fraction(
    df: pd.DataFrame,
    source_land_competition_fraction: float | None = None,
) -> float:
    if "land_competition_fraction" not in df.columns:
        if source_land_competition_fraction is None:
            return 1.0
        return float(np.clip(source_land_competition_fraction, 0.0, 1.0))
    raw_values = pd.to_numeric(df["land_competition_fraction"], errors="coerce").dropna().unique()
    if len(raw_values) == 0:
        if source_land_competition_fraction is None:
            return 1.0
        return float(np.clip(source_land_competition_fraction, 0.0, 1.0))
    if len(raw_values) > 1 and not np.allclose(raw_values, raw_values[0]):
        raise ValueError("land_competition_fraction must be uniform within a max-capacities CSV.")
    return float(raw_values[0])


def _validate_versioned_union_aliases(df: pd.DataFrame) -> None:
    """Ensure compatibility aliases do not diverge from the v1 estimate."""

    for versioned, generic in (
        (
            CLASSWISE_NESTED_UNION_AVAILABILITY_COLUMN,
            "renewable_union_availability",
        ),
        (CLASSWISE_NESTED_UNION_AREA_COLUMN, "renewable_union_area_km2"),
    ):
        if versioned not in df.columns or generic not in df.columns:
            continue
        versioned_values = pd.to_numeric(df[versioned], errors="coerce")
        generic_values = pd.to_numeric(df[generic], errors="coerce")
        if not np.allclose(
            versioned_values.to_numpy(),
            generic_values.to_numpy(),
            equal_nan=True,
        ):
            raise ValueError(
                f"Compatibility alias {generic!r} diverges from versioned "
                f"renewable-union column {versioned!r}."
            )


def _ensure_versioned_union_compatibility_aliases(df: pd.DataFrame) -> None:
    """Add generic aliases only when versioned provenance is already present."""

    for versioned, generic in (
        (
            CLASSWISE_NESTED_UNION_AVAILABILITY_COLUMN,
            "renewable_union_availability",
        ),
        (CLASSWISE_NESTED_UNION_AREA_COLUMN, "renewable_union_area_km2"),
    ):
        if versioned in df.columns and generic not in df.columns:
            df[generic] = df[versioned]


def apply_land_competition_scenario(
    df: pd.DataFrame,
    land_competition_fraction: float,
    source_land_competition_fraction: float | None = None,
) -> pd.DataFrame:
    output = df.copy()
    _ensure_versioned_union_compatibility_aliases(output)
    _validate_versioned_union_aliases(output)
    current_fraction = _uniform_land_competition_fraction(
        output,
        source_land_competition_fraction=source_land_competition_fraction,
    )
    target_fraction = float(np.clip(land_competition_fraction, 0.0, 1.0))

    if current_fraction <= 0 and target_fraction > 0:
        raise ValueError(
            "Cannot rescale a land-competition scenario from a CSV with zero available onshore competition share. "
            "Regenerate from a baseline or a non-zero source CSV."
        )

    if current_fraction <= 0:
        current_fraction = 1.0

    if "solar_area_km2" not in output.columns and "solar_availability" in output.columns and "area" in output.columns:
        output["solar_area_km2"] = output["area"] * output["solar_availability"]

    if (
        "renewable_union_area_km2" not in output.columns
        and "renewable_union_availability" in output.columns
        and "area" in output.columns
    ):
        output["renewable_union_area_km2"] = (
            output["area"] * output["renewable_union_availability"]
        )
    if (
        CLASSWISE_NESTED_UNION_AREA_COLUMN not in output.columns
        and CLASSWISE_NESTED_UNION_AVAILABILITY_COLUMN in output.columns
        and "area" in output.columns
    ):
        output[CLASSWISE_NESTED_UNION_AREA_COLUMN] = (
            output["area"] * output[CLASSWISE_NESTED_UNION_AVAILABILITY_COLUMN]
        )

    required_wind_area_columns = {"wind_onshore_area_km2", "wind_offshore_area_km2"}
    missing_wind_area_columns = sorted(required_wind_area_columns - set(output.columns))
    if missing_wind_area_columns:
        raise KeyError(
            "Rescaling land-competition scenarios now requires explicit wind area columns: "
            + ", ".join(missing_wind_area_columns)
        )

    wind_onshore_area_km2 = output["wind_onshore_area_km2"].clip(lower=0.0)
    wind_offshore_area_km2 = output["wind_offshore_area_km2"].clip(lower=0.0)
    output["wind_onshore_area_km2"] = wind_onshore_area_km2
    output["wind_offshore_area_km2"] = wind_offshore_area_km2

    if "wind_onshore_availability" not in output.columns and "area" in output.columns:
        output["wind_onshore_availability"] = np.where(
            output["area"] > 0.0,
            wind_onshore_area_km2 / output["area"],
            0.0,
        )
    if "wind_offshore_availability" not in output.columns and "area" in output.columns:
        output["wind_offshore_availability"] = np.where(
            output["area"] > 0.0,
            wind_offshore_area_km2 / output["area"],
            0.0,
        )

    base_land_exclusion_factor = (
        output["land_exclusion_factor"] / current_fraction
        if "land_exclusion_factor" in output.columns
        else None
    )
    base_constrained_onshore_area = (
        output["constrained_onshore_area_km2"] / current_fraction
        if "constrained_onshore_area_km2" in output.columns
        else None
    )
    base_solar_availability = output["solar_availability"] / current_fraction
    base_wind_onshore_availability = output["wind_onshore_availability"] / current_fraction
    base_solar_area_km2 = output["solar_area_km2"] / current_fraction
    base_wind_onshore_area_km2 = wind_onshore_area_km2 / current_fraction
    base_renewable_union_availability = {
        column: output[column] / current_fraction
        for column in (
            CLASSWISE_NESTED_UNION_AVAILABILITY_COLUMN,
            "renewable_union_availability",
        )
        if column in output.columns
    }
    base_renewable_union_area_km2 = {
        column: output[column] / current_fraction
        for column in (
            CLASSWISE_NESTED_UNION_AREA_COLUMN,
            "renewable_union_area_km2",
        )
        if column in output.columns
    }

    output["land_competition_fraction"] = target_fraction
    if base_land_exclusion_factor is not None:
        output["land_exclusion_factor"] = (base_land_exclusion_factor * target_fraction).clip(0.0, 1.0)
    if base_constrained_onshore_area is not None:
        output["constrained_onshore_area_km2"] = (base_constrained_onshore_area * target_fraction).clip(lower=0.0)
    output["solar_availability"] = (base_solar_availability * target_fraction).clip(0.0, 1.0)
    output["wind_onshore_availability"] = (base_wind_onshore_availability * target_fraction).clip(0.0, 1.0)
    for column, base_values in base_renewable_union_availability.items():
        output[column] = (base_values * target_fraction).clip(0.0, 1.0)
    output["wind_offshore_availability"] = output["wind_offshore_availability"].clip(0.0, 1.0)
    output["wind_availability"] = (
        output["wind_onshore_availability"] + output["wind_offshore_availability"]
    ).clip(0.0, 1.0)

    output["solar_area_km2"] = (base_solar_area_km2 * target_fraction).clip(lower=0.0)
    output["wind_onshore_area_km2"] = (base_wind_onshore_area_km2 * target_fraction).clip(lower=0.0)
    output["wind_offshore_area_km2"] = output["wind_offshore_area_km2"].clip(lower=0.0)
    output["wind_area_km2"] = output["wind_onshore_area_km2"] + output["wind_offshore_area_km2"]
    for column, base_values in base_renewable_union_area_km2.items():
        output[column] = (base_values * target_fraction).clip(lower=0.0)

    output["max_power_solar_mw"] = (
        output["solar_area_km2"] * output["solar_density_mw_per_km2"]
    ).clip(lower=0.0)
    output["max_power_wind_mw"] = (
        output["wind_area_km2"] * output["wind_density_mw_per_km2"]
    ).clip(lower=0.0)
    output["max_capacity_mw"] = output["max_power_solar_mw"] + output["max_power_wind_mw"]
    # Keep the historical generic availability column fractional.  The
    # versioned classwise union is a nested-overlap lower bound; its generic
    # renewable_union_* aliases are retained only for compatibility.
    output["availability"] = output[["wind_availability", "solar_availability"]].max(axis=1)
    _validate_versioned_union_aliases(output)
    return output


# The cells a campaign runs: onshore cells with suitable land (inputs/parameters.yaml, run_cells)
RUN_CELLS_MIN_LAND_PCT = float(parameter("run_cells.min_land_pct"))
RUN_CELLS_MIN_SUITABLE_KM2 = float(parameter("run_cells.min_suitable_km2"))
RUN_CELLS_SOUTH_LIMIT = float(parameter("run_cells.south_limit_deg"))


def run_cells(full_table: pd.DataFrame) -> pd.DataFrame:
    """Cells with at least RUN_CELLS_MIN_LAND_PCT % land, RUN_CELLS_MIN_SUITABLE_KM2 of PV plus
    onshore wind area in the 100 % table, and north of RUN_CELLS_SOUTH_LIMIT, as `lat, lon` (the
    `--global-locations` list of a campaign)."""
    if not np.allclose(pd.to_numeric(full_table["land_competition_fraction"]), 1.0):
        raise ValueError("The run cells are defined on the 100 % table")
    keep = (
        (full_table["onshore_land_pct"] >= RUN_CELLS_MIN_LAND_PCT)
        & (full_table["solar_area_km2"] + full_table["wind_onshore_area_km2"] >= RUN_CELLS_MIN_SUITABLE_KM2)
        & (full_table["latitude"] > RUN_CELLS_SOUTH_LIMIT)
    )
    cells = full_table.loc[keep, ["latitude", "longitude"]].rename(columns={"latitude": "lat", "longitude": "lon"})
    return cells.sort_values(["lat", "lon"]).reset_index(drop=True)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 24), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_provenance(output_csv: Path, rows: int, settings: dict) -> None:
    """<table>_provenance.json beside a land table: its hash, the parameters file and the settings."""
    provenance = {
        "written_by": "python -m model.land_table",
        "rows": int(rows),
        "sha256": _sha256(output_csv),
        "parameters": file_record(),
        "settings": settings,
    }
    output_csv.with_name(output_csv.stem + "_provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")


def write_land_competition_variant(
    base_csv: Path,
    output_csv: Path,
    land_competition_fraction: float,
    source_land_competition_fraction: float | None = None,
) -> pd.DataFrame:
    df = pd.read_csv(base_csv)
    df = apply_land_competition_scenario(
        df,
        land_competition_fraction,
        source_land_competition_fraction=source_land_competition_fraction,
    )
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)
    _write_provenance(output_csv, len(df), {
        "base_csv": str(base_csv), "base_sha256": _sha256(base_csv),
        "land_competition_fraction": land_competition_fraction,
        "source_land_competition_fraction": source_land_competition_fraction,
    })
    LOGGER.info(
        "Wrote land-competition variant %s from %s with fraction %.3f",
        output_csv,
        base_csv,
        land_competition_fraction,
    )
    return df


def build_land_availability_table(config: LandAvailabilityConfig) -> pd.DataFrame:
    """The full land table of one build (see the module docstring)."""
    cfg = config.resolved()
    if cfg.land_cover_path.suffix.lower() != ".hdf":
        raise ValueError(f"land_cover_path must be the MODIS MCD12C1 HDF file, got {cfg.land_cover_path}")
    availability = _aggregate_availability_from_hdf4(cfg)
    availability["cell_anchor"] = REQUIRED_CELL_ANCHOR
    availability["area"] = _cell_area_km2(availability["latitude"], cfg.coarse_degree)
    availability = _attach_bathymetry_depth(availability, cfg.bathymetry_path)

    availability["offshore_sea_pct"] = (100.0 - availability["onshore_land_pct"]).clip(lower=0.0, upper=100.0)
    availability["onshore_area_km2"] = availability["area"] * availability["onshore_land_pct"] / 100.0
    availability["offshore_area_km2"] = availability["area"] * availability["offshore_sea_pct"] / 100.0
    availability = _apply_land_exclusions(availability, cfg)

    # Siting area: cell area x class suitability x exclusions (x the land share, below)
    availability["solar_area_km2"] = availability["area"] * availability["solar_availability"]
    availability["wind_area_km2"] = availability["area"] * availability["wind_availability"]
    availability["wind_onshore_area_km2"] = availability["area"] * availability["wind_onshore_availability"]
    availability["wind_offshore_area_km2"] = availability["area"] * availability["wind_offshore_availability"]
    availability[CLASSWISE_NESTED_UNION_AREA_COLUMN] = availability["area"] * availability[CLASSWISE_NESTED_UNION_AVAILABILITY_COLUMN]
    availability["renewable_union_area_km2"] = availability[CLASSWISE_NESTED_UNION_AREA_COLUMN]

    centre_latitudes = availability["latitude"].astype(float)
    availability["solar_density_mw_per_km2"] = solar_density(centre_latitudes, 1.0)
    availability["wind_density_mw_per_km2"] = wind_density(centre_latitudes, 1.0, WIND_LAND_USE_KM2_PER_GW)

    availability = apply_land_competition_scenario(availability, cfg.land_competition_fraction)

    availability["max_power_solar_mw"] = (availability["solar_area_km2"] * availability["solar_density_mw_per_km2"]).clip(lower=0.0)
    availability["max_power_wind_mw"] = (availability["wind_area_km2"] * availability["wind_density_mw_per_km2"]).clip(lower=0.0)
    availability["max_capacity_mw"] = availability["max_power_solar_mw"] + availability["max_power_wind_mw"]
    availability["availability"] = availability[["wind_availability", "solar_availability"]].max(axis=1)

    for column in FINAL_COLUMNS:
        if column not in availability.columns:
            raise ValueError(f"Missing expected column '{column}' in aggregated table.")
    return availability[FINAL_COLUMNS].sort_values(["latitude", "longitude"]).reset_index(drop=True)


def write_land_availability_table(config: LandAvailabilityConfig) -> pd.DataFrame:
    cfg = config.resolved()
    df = build_land_availability_table(cfg)
    cfg.output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(cfg.output_csv, index=False)
    settings = {key: (str(value) if isinstance(value, Path) else value) for key, value in asdict(cfg).items()}
    settings["protected_area_paths"] = [str(path) for path in cfg.protected_area_paths]
    _write_provenance(cfg.output_csv, len(df), settings)
    LOGGER.info("Wrote %s rows to %s", len(df), cfg.output_csv)
    return df


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a land table, or derive a land-share variant of one.")
    parser.add_argument("--output", type=str, required=True, help="Destination CSV path.")
    parser.add_argument("--land-competition-fraction", type=float, default=DEFAULT_LAND_COMPETITION_FRACTION,
                        help="Share of the suitable onshore land available to ammonia production.")
    parser.add_argument("--base-csv", type=str, default=None,
                        help="Existing table to rescale to --land-competition-fraction, without rebuilding.")
    parser.add_argument("--source-land-competition-fraction", type=float, default=None,
                        help="Share already in --base-csv when it predates the land_competition_fraction column.")
    parser.add_argument("--land-cover", type=str, default=None, help="MODIS MCD12C1 .hdf file.")
    parser.add_argument("--bathymetry", type=str, default=None, help="Coarse bathymetry NetCDF for elevation_m.")
    parser.add_argument("--min-lat", type=float, default=-75.0)
    parser.add_argument("--max-lat", type=float, default=75.0)
    parser.add_argument("--include-offshore-wind", action="store_true", help="Count water as offshore-wind area.")
    parser.add_argument("--protected-area", action="append", default=[], help="WDPA polygon file; repeat for each file.")
    parser.add_argument("--slope-raster", type=str, default=None, help="Elevation NetCDF (GEBCO) for the slope screen.")
    parser.add_argument("--max-slope-degrees", type=float, default=DEFAULT_MAX_SLOPE_DEGREES)
    parser.add_argument("--skip-slope-exclusion", action="store_true", help="Keep land of any slope.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = _parse_args(argv)
    if args.base_csv is not None:
        write_land_competition_variant(
            base_csv=repo_path(args.base_csv),
            output_csv=repo_path(args.output),
            land_competition_fraction=args.land_competition_fraction,
            source_land_competition_fraction=args.source_land_competition_fraction,
        )
        return
    write_land_availability_table(LandAvailabilityConfig(
        output_csv=Path(args.output),
        land_cover_path=Path(args.land_cover) if args.land_cover else LAND_COVER_FILE,
        bathymetry_path=Path(args.bathymetry) if args.bathymetry else BATHYMETRY_FILE,
        lat_bounds=(args.min_lat, args.max_lat),
        include_offshore_wind=args.include_offshore_wind,
        land_competition_fraction=args.land_competition_fraction,
        protected_area_paths=tuple(Path(path) for path in args.protected_area),
        slope_raster_path=None if args.skip_slope_exclusion or not args.slope_raster else Path(args.slope_raster),
        max_slope_degrees=args.max_slope_degrees,
        skip_slope_exclusion=args.skip_slope_exclusion,
    ))


if __name__ == "__main__":
    main()
