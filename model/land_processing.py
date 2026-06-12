"""Utilities for deriving max-capacity inputs for the run_global workflow.

The module aggregates the 0.05 degree MODIS land-cover tiles to 1 degree cells,
applies technology-specific suitability factors, optional protected-area and
slope exclusions, and estimates maximum installable renewable capacity (MW).
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Tuple

import geopandas as gpd
import numpy as np
import pandas as pd
import xarray as xr
from shapely.geometry import box
from shapely.errors import GEOSException

try:
    from . import data_paths
except ImportError:  # pragma: no cover - fallback for direct execution
    PACKAGE_ROOT = Path(__file__).resolve().parent
    if str(PACKAGE_ROOT) not in sys.path:
        sys.path.insert(0, str(PACKAGE_ROOT))
    import data_paths  # type: ignore

LOGGER = logging.getLogger(__name__)

REPO_ROOT = data_paths.REPO_ROOT
DATA_DIR = data_paths.DATA_DIR
DEFAULT_LAND_COVER_FILE = data_paths.LAND_COVER_FILE
DEFAULT_BATHYMETRY_FILE = data_paths.BATHYMETRY_FILE
DEFAULT_OUTPUT_CSV = data_paths.MAX_CAPACITIES_FILE
EARTH_RADIUS_KM = 6371.0
MODIS_WATER_CLASS = 0
WIND_LAND_USE_KM2_PER_GW = 200.0  # Salmon et al. (2021)
DEFAULT_LAND_SCENARIO_NAME = "baseline"
LAND_COMPETITION_SCENARIOS = {
    DEFAULT_LAND_SCENARIO_NAME: 1.0,
    "paper_2pct": 0.02,
    "high_50pct": 0.50,
}
DEFAULT_LAND_COMPETITION_FRACTION = LAND_COMPETITION_SCENARIOS[DEFAULT_LAND_SCENARIO_NAME]
DEFAULT_MAX_SLOPE_DEGREES = 15.0
EQUAL_AREA_CRS = "EPSG:6933"
DEFAULT_SLOPE_RASTER_FILE = data_paths.GEBCO_SLOPE_FILE
DEFAULT_PROTECTED_AREA_PATHS = data_paths.WDPA_SHAPEFILES
# First Solar Series 6 (2018) constants used in van de Ven et al. (2021) style packing
FIRST_SOLAR_MODULE_WIDTH_M = 1.21
FIRST_SOLAR_MODULE_LENGTH_M = 2.06
FIRST_SOLAR_MODULE_POWER_KW = 0.45
SOLAR_DECLINATION_DEG = 23.44
SOLAR_MIN_TILT_DEG = 15.0
SOLAR_MAX_TILT_DEG = 55.0
SOLAR_MIN_SUN_ALTITUDE_DEG = 5.0
SOLAR_GCR_MIN = 0.05
SOLAR_GCR_MAX = 0.8

# Suitability factors expressed separately for wind and solar siting. The generic
# availability (used for quick inspection) is the max of the two technology maps.
#
# IGBP Land Cover Type 1 classes (MODIS MCD12C1):
#   0  Water bodies                    6  Closed shrublands        12  Croplands
#   1  Evergreen needleleaf forests    7  Open shrublands          13  Urban and built-up lands
#   2  Evergreen broadleaf forests     8  Woody savannas           14  Cropland / natural veg mosaics
#   3  Deciduous needleleaf forests    9  Savannas                 15  Permanent snow and ice
#   4  Deciduous broadleaf forests    10  Grasslands               16  Barren / sparsely vegetated
#   5  Mixed forests                  11  Permanent wetlands       17  Unclassified (treated as 0)
#
# Onshore land-availability factors follow Salmon & Bañares-Alcántara (2022),
# Table 1 / Verschuur et al. (2024) supplementary material.
# Wind: open terrain + cropland/mosaics (turbines coexist with agriculture).
# Solar: excludes croplands/mosaics but includes small urban rooftop fraction.
# The base map keeps the historical offshore-wind option available; the default
# LandAvailabilityConfig sets include_offshore_wind=False for Salmon/Verschuur-
# style onshore siting.
MODIS_CLASS_WIND_AVAILABILITY = {
    0: 1.0,    # Water — offshore wind (not in onshore reference)
    6: 0.5,    # Closed shrublands
    7: 0.5,    # Open shrublands
    8: 0.2,    # Woody savannas
    9: 0.2,    # Savannas
    10: 0.2,   # Grasslands
    12: 0.05,  # Croplands
    14: 0.05,  # Cropland / natural vegetation mosaics
    16: 1.0,   # Barren / sparsely vegetated
}

MODIS_CLASS_SOLAR_AVAILABILITY = {
    6: 0.5,    # Closed shrublands
    7: 0.5,    # Open shrublands
    8: 0.2,    # Woody savannas
    9: 0.2,    # Savannas
    10: 0.2,   # Grasslands
    13: 0.03,  # Urban and built-up lands (rooftop solar)
    16: 1.0,   # Barren / sparsely vegetated
}

FINAL_COLUMNS = [
    "latitude",
    "longitude",
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
    "land_exclusion_factor",
    "land_competition_fraction",
    "constrained_onshore_area_km2",
    "wind_onshore_availability",
    "wind_offshore_availability",
    "wind_availability",
    "solar_availability",
    "solar_area_km2",
    "wind_area_km2",
    "wind_onshore_area_km2",
    "wind_offshore_area_km2",
    "wind_density_mw_per_km2",
    "solar_density_mw_per_km2",
    "max_power_solar_mw",
    "max_power_wind_mw",
    "max_capacity_mw",
]


def land_competition_fraction_for_scenario(name: str) -> float:
    key = str(name).strip()
    try:
        return float(LAND_COMPETITION_SCENARIOS[key])
    except KeyError as exc:
        raise KeyError(
            f"Unknown land-availability scenario {name!r}. "
            f"Choose from {', '.join(sorted(LAND_COMPETITION_SCENARIOS))}."
        ) from exc


def land_availability_output_csv(name: str, data_dir: Path = DATA_DIR) -> Path:
    land_competition_fraction_for_scenario(name)
    return Path(data_dir) / f"max_capacities_{name}.csv"


def _format_eta(seconds: float) -> str:
    seconds = max(float(seconds), 0.0)
    total_seconds = int(round(seconds))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours:d}h {minutes:02d}m {secs:02d}s"
    if minutes:
        return f"{minutes:d}m {secs:02d}s"
    return f"{secs:d}s"


def _resolve_path(path: str | Path | None, default: Path) -> Path:
    if path is None:
        return default
    resolved = Path(path)
    if not resolved.is_absolute():
        resolved = REPO_ROOT / resolved
    return resolved


def _resolve_optional_path(path: str | Path | None) -> Path | None:
    if path is None:
        return None
    resolved = Path(path)
    if not resolved.is_absolute():
        resolved = REPO_ROOT / resolved
    return resolved


def _resolve_path_tuple(paths: Iterable[str | Path] | None) -> Tuple[Path, ...]:
    if paths is None:
        return tuple()
    return tuple(path for item in paths if (path := _resolve_optional_path(item)) is not None)


def _wind_availability_map(include_offshore_wind: bool) -> dict[int, float]:
    availability = dict(MODIS_CLASS_WIND_AVAILABILITY)
    if not include_offshore_wind:
        availability[MODIS_WATER_CLASS] = 0.0
    return availability


def _wind_density(
    latitudes: pd.Series,
    scale: float = 1.0,
    land_use_km2_per_gw: float = WIND_LAND_USE_KM2_PER_GW,
) -> pd.Series:
    base_density = 1000.0 / float(land_use_km2_per_gw)  # MW per km²
    densities = np.full(latitudes.shape[0], base_density * scale)
    return pd.Series(densities, index=latitudes.index)


def _solar_density_raw(latitudes: np.ndarray) -> np.ndarray:
    lat_abs = np.abs(latitudes)
    beta = np.clip(lat_abs, SOLAR_MIN_TILT_DEG, SOLAR_MAX_TILT_DEG)
    solar_alt = 90.0 - lat_abs - SOLAR_DECLINATION_DEG
    solar_alt = np.clip(solar_alt, SOLAR_MIN_SUN_ALTITUDE_DEG, 89.0)

    beta_rad = np.deg2rad(beta)
    solar_alt_rad = np.deg2rad(solar_alt)
    row_pitch_m = FIRST_SOLAR_MODULE_LENGTH_M * (
        np.sin(beta_rad) + np.cos(beta_rad) / np.tan(solar_alt_rad)
    )
    row_pitch_m = np.clip(row_pitch_m, FIRST_SOLAR_MODULE_LENGTH_M, np.inf)

    gcr = FIRST_SOLAR_MODULE_WIDTH_M / row_pitch_m
    gcr = np.clip(gcr, SOLAR_GCR_MIN, SOLAR_GCR_MAX)

    module_area_m2 = FIRST_SOLAR_MODULE_WIDTH_M * FIRST_SOLAR_MODULE_LENGTH_M
    module_power_mw = FIRST_SOLAR_MODULE_POWER_KW / 1000.0
    area_per_mw_m2 = (module_area_m2 / module_power_mw) / gcr
    return 1_000_000.0 / area_per_mw_m2


def _solar_density(
    latitudes: pd.Series,
    scale: float = 1.0,
    base_land_use_km2_per_mw: float | None = None,
) -> pd.Series:
    density_mw_per_km2 = _solar_density_raw(latitudes.to_numpy())

    if base_land_use_km2_per_mw is not None:
        base_land_use = float(base_land_use_km2_per_mw)
        if base_land_use <= 0:
            raise ValueError("base_land_use_km2_per_mw must be positive")
        target_equator_density = 1.0 / base_land_use
        model_equator_density = float(_solar_density_raw(np.array([0.0]))[0])
        if model_equator_density > 0:
            density_mw_per_km2 *= target_equator_density / model_equator_density

    densities = density_mw_per_km2 * scale
    return pd.Series(densities, index=latitudes.index)


@dataclass
class LandAvailabilityConfig:
    """Runtime knobs for the land-cover aggregation pipeline."""

    land_cover_path: Path = DEFAULT_LAND_COVER_FILE
    bathymetry_path: Path = DEFAULT_BATHYMETRY_FILE
    output_csv: Path = DEFAULT_OUTPUT_CSV
    coarse_degree: float = 1.0
    fine_degree: float = 0.05
    lat_bounds: Tuple[float, float] = (-75.0, 75.0)
    include_offshore_wind: bool = False
    land_competition_fraction: float = DEFAULT_LAND_COMPETITION_FRACTION
    protected_area_paths: Tuple[Path, ...] = tuple()
    protected_area_csv: Path | None = None
    slope_raster_path: Path | None = None
    slope_exclusion_csv: Path | None = None
    max_slope_degrees: float = DEFAULT_MAX_SLOPE_DEGREES
    skip_slope_exclusion: bool = False

    def resolved(self) -> "LandAvailabilityConfig":
        return LandAvailabilityConfig(
            land_cover_path=_resolve_path(self.land_cover_path, DEFAULT_LAND_COVER_FILE),
            bathymetry_path=_resolve_path(self.bathymetry_path, DEFAULT_BATHYMETRY_FILE),
            output_csv=_resolve_path(self.output_csv, DEFAULT_OUTPUT_CSV),
            coarse_degree=self.coarse_degree,
            fine_degree=self.fine_degree,
            lat_bounds=self.lat_bounds,
            include_offshore_wind=bool(self.include_offshore_wind),
            land_competition_fraction=float(self.land_competition_fraction),
            protected_area_paths=_resolve_path_tuple(self.protected_area_paths),
            protected_area_csv=_resolve_optional_path(self.protected_area_csv),
            slope_raster_path=_resolve_optional_path(self.slope_raster_path),
            slope_exclusion_csv=_resolve_optional_path(self.slope_exclusion_csv),
            max_slope_degrees=float(self.max_slope_degrees),
            skip_slope_exclusion=bool(self.skip_slope_exclusion),
        )


def _cell_area_km2(latitudes: pd.Series, delta_deg: float) -> pd.Series:
    radians = np.deg2rad(latitudes.to_numpy())
    radians_upper = np.deg2rad(latitudes.to_numpy() + delta_deg)
    band_height = np.sin(radians_upper) - np.sin(radians)
    area = (EARTH_RADIUS_KM ** 2) * np.deg2rad(delta_deg) * band_height
    return pd.Series(np.abs(area), index=latitudes.index)

def _load_land_cover_frame(config: LandAvailabilityConfig) -> pd.DataFrame:
    with xr.open_dataset(config.land_cover_path, engine="netcdf4") as dataset:
        df = dataset["Land_Cover_Type_1_Percent"].to_dataframe().reset_index()

    df = df.rename(
        columns={
            "YDim:MOD12C1": "y_index",
            "XDim:MOD12C1": "x_index",
            "Num_IGBP_Classes:MOD12C1": "modis_class",
            "Land_Cover_Type_1_Percent": "percentage",
        }
    )

    df["class_fraction"] = df["percentage"].astype(float) / 100.0
    df["latitude"] = 90.0 - config.fine_degree * df["y_index"]
    df["longitude"] = -180.0 + config.fine_degree * df["x_index"]

    df["latitude"] = np.floor(df["latitude"] / config.coarse_degree) * config.coarse_degree
    df["longitude"] = np.floor(df["longitude"] / config.coarse_degree) * config.coarse_degree
    df = df[(df["latitude"] >= config.lat_bounds[0]) & (df["latitude"] <= config.lat_bounds[1])]

    return df[["latitude", "longitude", "modis_class", "class_fraction"]]


def _aggregate_availability_from_hdf4(config: LandAvailabilityConfig) -> pd.DataFrame:
    """Aggregate MODIS land-cover suitability directly from the HDF4 input.

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

        if not (0 <= water_class < n_classes):
            raise ValueError(
                f"Water class index {water_class} out of bounds for n_classes={n_classes}."
            )

        # Compute y indices that intersect the latitude bounds.
        # lat = 90 - fine_degree * y
        lat_min, lat_max = config.lat_bounds
        y_start = int(np.floor((90.0 - lat_max) / config.fine_degree))
        y_end = int(np.ceil((90.0 - lat_min) / config.fine_degree))
        y_start = max(0, min(n_y, y_start))
        y_end = max(0, min(n_y, y_end))

        # Ensure we only process complete coarse bands.
        y_start = (y_start // group_size) * group_size
        y_end = (y_end // group_size) * group_size

        # Similarly, ensure longitude dimension divides neatly.
        x_end = (n_x // group_size) * group_size
        if x_end == 0:
            raise ValueError("Longitude dimension too small for requested aggregation.")

        lon_coarse = -180.0 + config.coarse_degree * np.arange(x_end // group_size)

        rows: list[dict[str, float]] = []

        for y0 in range(y_start, y_end, group_size):
            # Read a single coarse latitude band (group_size rows) for all longitudes/classes.
            cube = sds[y0 : y0 + group_size, 0:x_end, :]
            cube = np.asarray(cube, dtype=float) / 100.0

            # Weighted availability: sum_class(class_fraction * factor)
            wind_component_fine = np.tensordot(cube, wind_factors, axes=([2], [0]))
            solar_component_fine = np.tensordot(cube, solar_factors, axes=([2], [0]))
            wind_onshore_fine = np.tensordot(cube, wind_onshore_factors, axes=([2], [0]))
            water_fraction_fine = cube[:, :, water_class]
            wind_offshore_fine = (
                water_fraction_fine if config.include_offshore_wind else np.zeros_like(water_fraction_fine)
            )

            # Reduce to coarse cells: average across fine y and fine x.
            # Shapes: (group_size, x_end) -> (n_lon_coarse,)
            wind_band = wind_component_fine.reshape(group_size, -1, group_size).mean(axis=(0, 2))
            wind_onshore_band = wind_onshore_fine.reshape(group_size, -1, group_size).mean(axis=(0, 2))
            wind_offshore_band = wind_offshore_fine.reshape(group_size, -1, group_size).mean(axis=(0, 2))
            solar_band = solar_component_fine.reshape(group_size, -1, group_size).mean(axis=(0, 2))
            water_band = water_fraction_fine.reshape(group_size, -1, group_size).mean(axis=(0, 2))

            lat_center = 90.0 - config.fine_degree * float(y0)
            lat_coarse = np.floor(lat_center / config.coarse_degree) * config.coarse_degree

            for lon, wind_val, wind_onshore_val, wind_offshore_val, solar_val, water_val in zip(
                lon_coarse,
                wind_band,
                wind_onshore_band,
                wind_offshore_band,
                solar_band,
                water_band,
            ):
                wind_val = float(np.clip(wind_val, 0.0, 1.0))
                wind_onshore_val = float(np.clip(wind_onshore_val, 0.0, 1.0))
                wind_offshore_val = float(np.clip(wind_offshore_val, 0.0, 1.0))
                solar_val = float(np.clip(solar_val, 0.0, 1.0))
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


def _aggregate_availability(df: pd.DataFrame, include_offshore_wind: bool) -> pd.DataFrame:
    grouped = (
        df.groupby(["latitude", "longitude", "modis_class"], as_index=False)["class_fraction"]
        .mean()
    )

    wind_map = _wind_availability_map(include_offshore_wind)
    grouped["wind_component"] = (
        grouped["class_fraction"] * grouped["modis_class"].map(wind_map).fillna(0.0)
    )
    wind_onshore_map = dict(wind_map)
    wind_onshore_map[MODIS_WATER_CLASS] = 0.0
    grouped["wind_onshore_component"] = (
        grouped["class_fraction"] * grouped["modis_class"].map(wind_onshore_map).fillna(0.0)
    )
    grouped["solar_component"] = (
        grouped["class_fraction"] * grouped["modis_class"].map(MODIS_CLASS_SOLAR_AVAILABILITY).fillna(0.0)
    )

    agg = grouped.groupby(["latitude", "longitude"], as_index=False).agg(
        wind_availability=("wind_component", "sum"),
        wind_onshore_availability=("wind_onshore_component", "sum"),
        solar_availability=("solar_component", "sum"),
    )
    agg["wind_availability"] = agg["wind_availability"].clip(0.0, 1.0)
    agg["wind_onshore_availability"] = agg["wind_onshore_availability"].clip(0.0, 1.0)
    agg["solar_availability"] = agg["solar_availability"].clip(0.0, 1.0)

    water_fraction = (
        grouped.loc[grouped["modis_class"] == MODIS_WATER_CLASS, ["latitude", "longitude", "class_fraction"]]
        .groupby(["latitude", "longitude"], as_index=False)
        .sum()
        .rename(columns={"class_fraction": "water_fraction"})
    )
    availability = agg.merge(water_fraction, on=["latitude", "longitude"], how="left")
    availability["water_fraction"] = availability["water_fraction"].fillna(0.0).clip(0.0, 1.0)
    availability["wind_offshore_availability"] = (
        availability["water_fraction"] if include_offshore_wind else 0.0
    )
    availability["onshore_land_pct"] = (1.0 - availability["water_fraction"]).clip(0.0, 1.0) * 100.0

    availability["availability"] = availability[["wind_availability", "solar_availability"]].max(axis=1)
    availability["availability"] = availability["availability"].clip(0.0, 1.0)

    return availability.drop(columns=["water_fraction"])


def _attach_bathymetry_depth(df: pd.DataFrame, bathymetry_path: Path) -> pd.DataFrame:
    if not bathymetry_path.exists():
        LOGGER.warning("Bathymetry file %s not found; offshore capacities will be zero.", bathymetry_path)
        output = df.copy()
        output["elevation_m"] = np.nan
        return output

    with xr.open_dataset(bathymetry_path) as dataset:
        variable_name = "depths" if "depths" in dataset.data_vars else next(iter(dataset.data_vars))
        depth_da = dataset[variable_name]
        points = xr.Dataset(
            coords={
                "points": np.arange(len(df)),
                "latitude": ("points", df["latitude"].to_numpy()),
                "longitude": ("points", df["longitude"].to_numpy()),
            }
        )
        selected = depth_da.sel(
            latitude=points["latitude"],
            longitude=points["longitude"],
            method="nearest",
        )

    output = df.copy()
    # Flip sign: raw data has positive=ocean-depth, negative=land-elevation.
    # We store elevation_m with standard convention: positive=above sea level.
    output["elevation_m"] = -selected.to_numpy().astype(float)
    return output


def _merge_fraction_csv(
    df: pd.DataFrame,
    path: Path,
    output_column: str,
    value_columns: tuple[str, ...],
    default_value: float,
    coarse_degree: float,
) -> pd.DataFrame:
    """Merge a precomputed percentage/fraction mask onto the land grid."""
    if not path.exists():
        raise FileNotFoundError(f"Mask CSV not found: {path}")

    values = pd.read_csv(path)
    lower_columns = {str(column).strip().lower(): column for column in values.columns}
    lat_col = lower_columns.get("latitude") or lower_columns.get("lat")
    lon_col = lower_columns.get("longitude") or lower_columns.get("lon")
    value_col = next(
        (lower_columns.get(candidate.lower()) for candidate in value_columns if candidate.lower() in lower_columns),
        None,
    )
    if lat_col is None or lon_col is None or value_col is None:
        raise ValueError(
            f"{path} must contain latitude/longitude and one of {', '.join(value_columns)}."
        )

    cleaned = values[[lat_col, lon_col, value_col]].rename(
        columns={lat_col: "latitude", lon_col: "longitude", value_col: output_column}
    )
    cleaned["latitude"] = (
        np.floor(cleaned["latitude"].astype(float) / coarse_degree) * coarse_degree
    )
    cleaned["longitude"] = (
        np.floor(cleaned["longitude"].astype(float) / coarse_degree) * coarse_degree
    )
    cleaned[output_column] = pd.to_numeric(cleaned[output_column], errors="coerce")
    if cleaned[output_column].max(skipna=True) <= 1.0:
        cleaned[output_column] *= 100.0
    cleaned = cleaned.groupby(["latitude", "longitude"], as_index=False)[output_column].mean()

    output = df.drop(columns=[output_column], errors="ignore").merge(
        cleaned,
        on=["latitude", "longitude"],
        how="left",
    )
    output[output_column] = output[output_column].fillna(default_value).clip(0.0, 100.0)
    return output


def _cell_geometries(df: pd.DataFrame, coarse_degree: float) -> gpd.GeoDataFrame:
    cells = df[["latitude", "longitude"]].drop_duplicates().reset_index(drop=True)
    cells["cell_id"] = np.arange(len(cells), dtype=int)
    cells["geometry"] = [
        box(lon, lat, lon + coarse_degree, lat + coarse_degree)
        for lat, lon in zip(cells["latitude"], cells["longitude"])
    ]
    return gpd.GeoDataFrame(cells, geometry="geometry", crs="EPSG:4326")


def _valid_geometries(frame: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    frame = frame[frame.geometry.notna() & ~frame.geometry.is_empty].copy()
    if frame.empty:
        return frame
    try:
        frame = frame.set_geometry(frame.geometry.make_valid())
    except AttributeError:  # pragma: no cover - depends on shapely/geopandas version
        frame = frame.set_geometry(frame.geometry.buffer(0))
    return frame[frame.geometry.notna() & ~frame.geometry.is_empty]


def _repair_geometry_series(geometries: gpd.GeoSeries) -> gpd.GeoSeries:
    geometries = geometries[geometries.notna() & ~geometries.is_empty]
    if geometries.empty:
        return geometries
    try:
        repaired = geometries.make_valid()
    except AttributeError:  # pragma: no cover - depends on shapely/geopandas version
        repaired = geometries.buffer(0)
    repaired = gpd.GeoSeries(repaired, index=geometries.index, crs=geometries.crs)
    return repaired[repaired.notna() & ~repaired.is_empty]


def _filter_protected_areas(frame: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    if frame.empty:
        return frame

    output = frame.copy()
    if "STATUS" in output.columns:
        status = output["STATUS"].astype(str).str.lower().str.strip()
        output = output[status.isin({"designated", "inscribed", "established"})]

    if "REALM" in output.columns:
        realm = output["REALM"].astype(str).str.lower().str.strip()
        output = output[realm != "marine"]
    elif "MARINE" in output.columns:
        marine = output["MARINE"].astype(str).str.lower().str.strip()
        output = output[~marine.isin({"2", "true", "marine"})]

    return _valid_geometries(output)


def _protected_area_checkpoint_path(output_csv: Path) -> Path:
    return output_csv.with_name(f"{output_csv.stem}.protected_area_checkpoint.csv")


def _load_protected_area_checkpoint(
    checkpoint_path: Path,
    cells: gpd.GeoDataFrame,
    cell_areas: pd.Series,
) -> tuple[pd.Series, set[int]]:
    protected_area_m2 = pd.Series(0.0, index=cell_areas.index)
    if not checkpoint_path.exists():
        return protected_area_m2, set()

    cached = pd.read_csv(checkpoint_path)
    required_columns = {"latitude", "longitude", "protected_area_pct"}
    if not required_columns.issubset(cached.columns):
        LOGGER.warning(
            "Ignoring protected-area checkpoint %s because required columns are missing.",
            checkpoint_path,
        )
        return protected_area_m2, set()

    cached = cached[["latitude", "longitude", "protected_area_pct"]].copy()
    cached["protected_area_pct"] = (
        pd.to_numeric(cached["protected_area_pct"], errors="coerce")
        .fillna(0.0)
        .clip(0.0, 100.0)
    )
    cached = cached.drop_duplicates(subset=["latitude", "longitude"], keep="last")
    cached = cells[["cell_id", "latitude", "longitude"]].merge(
        cached,
        on=["latitude", "longitude"],
        how="inner",
    )
    if cached.empty:
        return protected_area_m2, set()

    cached_ids = cached["cell_id"].astype(int)
    protected_area_m2.loc[cached_ids] = (
        cell_areas.loc[cached_ids].to_numpy()
        * cached["protected_area_pct"].to_numpy()
        / 100.0
    )
    processed_cell_ids = set(cached_ids.tolist())
    LOGGER.info(
        "Loaded protected-area checkpoint %s for %d/%d cells.",
        checkpoint_path,
        len(processed_cell_ids),
        len(cells),
    )
    return protected_area_m2, processed_cell_ids


def _write_protected_area_checkpoint(
    checkpoint_path: Path,
    protected_area_m2: pd.Series,
    cell_areas: pd.Series,
    cells: gpd.GeoDataFrame,
    processed_cell_ids: set[int],
) -> None:
    if not processed_cell_ids:
        return

    processed_index = pd.Index(sorted(processed_cell_ids), dtype=int, name="cell_id")
    protected_fraction = (protected_area_m2.loc[processed_index] / cell_areas.loc[processed_index]).clip(0.0, 1.0)
    checkpoint = (
        protected_fraction.rename("protected_area_pct")
        .reset_index()
        .merge(cells[["cell_id", "latitude", "longitude"]], on="cell_id", how="left")
    )
    checkpoint["protected_area_pct"] *= 100.0
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint[["latitude", "longitude", "protected_area_pct"]].sort_values(
        ["latitude", "longitude"]
    ).to_csv(checkpoint_path, index=False)


def _geometry_union(geometries: gpd.GeoSeries):
    try:
        if hasattr(geometries, "union_all"):
            return geometries.union_all()
        return geometries.unary_union
    except GEOSException:
        repaired = _repair_geometry_series(geometries)
        if repaired.empty:
            return None
        if hasattr(repaired, "union_all"):
            return repaired.union_all()
        return repaired.unary_union


def _overlay_intersection_with_retry(
    left: gpd.GeoDataFrame,
    right: gpd.GeoDataFrame,
) -> gpd.GeoDataFrame:
    try:
        return gpd.overlay(left, right, how="intersection", keep_geom_type=False)
    except GEOSException:
        repaired_left = _valid_geometries(left)
        repaired_right = _valid_geometries(right)
        if repaired_left.empty or repaired_right.empty:
            return gpd.GeoDataFrame(columns=list(left.columns), geometry="geometry", crs=left.crs)
        return gpd.overlay(
            repaired_left,
            repaired_right,
            how="intersection",
            keep_geom_type=False,
        )


def _protected_area_pct_from_vectors(
    df: pd.DataFrame,
    protected_area_paths: Tuple[Path, ...],
    coarse_degree: float,
    checkpoint_path: Path | None = None,
) -> pd.DataFrame:
    output = df.copy()
    output["protected_area_pct"] = 0.0
    if not protected_area_paths:
        return output

    for path in protected_area_paths:
        if not path.exists():
            raise FileNotFoundError(f"Protected-area vector file not found: {path}")

    cells = _cell_geometries(output, coarse_degree)
    cells_equal_area = cells.to_crs(EQUAL_AREA_CRS)
    cell_areas = cells_equal_area.set_index("cell_id").geometry.area
    if checkpoint_path is not None:
        protected_area_m2, processed_cell_ids = _load_protected_area_checkpoint(
            checkpoint_path,
            cells,
            cell_areas,
        )
    else:
        protected_area_m2 = pd.Series(0.0, index=cell_areas.index)
        processed_cell_ids = set()

    def persist_checkpoint(cell_ids: set[int]) -> None:
        if not cell_ids:
            return
        processed_cell_ids.update(cell_ids)
        if checkpoint_path is not None:
            _write_protected_area_checkpoint(
                checkpoint_path,
                protected_area_m2,
                cell_areas,
                cells,
                processed_cell_ids,
            )

    lat_min = float(cells["latitude"].min())
    lat_max = float(cells["latitude"].max() + coarse_degree)
    chunk_degree = max(coarse_degree, 5.0)
    chunk_edges = np.arange(lat_min, lat_max + chunk_degree, chunk_degree)
    chunk_pairs = list(zip(chunk_edges[:-1], chunk_edges[1:]))
    LOGGER.info(
        "Protected-area overlay: %d latitude chunks across %d WDPA source files.",
        len(chunk_pairs),
        len(protected_area_paths),
    )
    stage_started = time.perf_counter()

    for chunk_index, (chunk_start, chunk_end) in enumerate(chunk_pairs, start=1):
        chunk_started = time.perf_counter()
        chunk_cells = cells[
            (cells["latitude"] >= chunk_start)
            & (cells["latitude"] < chunk_end)
        ]
        if chunk_cells.empty:
            elapsed = time.perf_counter() - stage_started
            eta = (elapsed / chunk_index) * (len(chunk_pairs) - chunk_index)
            LOGGER.info(
                "Protected-area chunk %d/%d lat [%.0f, %.0f): no cells, elapsed=%s, eta=%s",
                chunk_index,
                len(chunk_pairs),
                chunk_start,
                chunk_end,
                _format_eta(elapsed),
                _format_eta(eta),
            )
            continue

        chunk_cell_ids = set(chunk_cells["cell_id"].astype(int))
        if chunk_cell_ids.issubset(processed_cell_ids):
            elapsed = time.perf_counter() - stage_started
            eta = (elapsed / chunk_index) * (len(chunk_pairs) - chunk_index)
            LOGGER.info(
                "Protected-area chunk %d/%d lat [%.0f, %.0f): restored from checkpoint, elapsed=%s, eta=%s",
                chunk_index,
                len(chunk_pairs),
                chunk_start,
                chunk_end,
                _format_eta(elapsed),
                _format_eta(eta),
            )
            continue

        bbox = tuple(chunk_cells.total_bounds)
        protected_frames: list[gpd.GeoDataFrame] = []
        for path in protected_area_paths:
            protected = gpd.read_file(path, bbox=bbox)
            if protected.empty:
                continue
            if protected.crs is None:
                protected = protected.set_crs("EPSG:4326")
            else:
                protected = protected.to_crs("EPSG:4326")
            protected = _filter_protected_areas(protected)
            if not protected.empty:
                protected_frames.append(protected[["geometry"]])

        if not protected_frames:
            elapsed = time.perf_counter() - stage_started
            eta = (elapsed / chunk_index) * (len(chunk_pairs) - chunk_index)
            LOGGER.info(
                "Protected-area chunk %d/%d lat [%.0f, %.0f): no WDPA intersections in bbox, elapsed=%s, eta=%s",
                chunk_index,
                len(chunk_pairs),
                chunk_start,
                chunk_end,
                _format_eta(elapsed),
                _format_eta(eta),
            )
            persist_checkpoint(chunk_cell_ids)
            continue

        protected = gpd.GeoDataFrame(
            pd.concat(protected_frames, ignore_index=True),
            geometry="geometry",
            crs="EPSG:4326",
        ).to_crs(EQUAL_AREA_CRS)
        protected = _valid_geometries(protected)
        protected_union = _geometry_union(protected.geometry)
        if protected_union is None or protected_union.is_empty:
            elapsed = time.perf_counter() - stage_started
            eta = (elapsed / chunk_index) * (len(chunk_pairs) - chunk_index)
            LOGGER.info(
                "Protected-area chunk %d/%d lat [%.0f, %.0f): empty dissolved WDPA geometry, elapsed=%s, eta=%s",
                chunk_index,
                len(chunk_pairs),
                chunk_start,
                chunk_end,
                _format_eta(elapsed),
                _format_eta(eta),
            )
            persist_checkpoint(chunk_cell_ids)
            continue

        protected_union_frame = gpd.GeoDataFrame(
            geometry=[protected_union],
            crs=EQUAL_AREA_CRS,
        )
        chunk_cells_equal_area = cells_equal_area[
            cells_equal_area["cell_id"].isin(chunk_cell_ids)
        ][["cell_id", "geometry"]]
        intersections = _overlay_intersection_with_retry(
            chunk_cells_equal_area,
            protected_union_frame,
        )
        if not intersections.empty:
            areas = intersections.geometry.area.groupby(intersections["cell_id"]).sum()
            protected_area_m2 = protected_area_m2.add(areas, fill_value=0.0)

        persist_checkpoint(chunk_cell_ids)

        elapsed = time.perf_counter() - stage_started
        eta = (elapsed / chunk_index) * (len(chunk_pairs) - chunk_index)
        LOGGER.info(
            "Protected-area chunk %d/%d lat [%.0f, %.0f): %d cells, %d WDPA features, %d overlaps, chunk=%s, elapsed=%s, eta=%s",
            chunk_index,
            len(chunk_pairs),
            chunk_start,
            chunk_end,
            len(chunk_cells_equal_area),
            len(protected),
            len(intersections),
            _format_eta(time.perf_counter() - chunk_started),
            _format_eta(elapsed),
            _format_eta(eta),
        )

    protected_fraction = (protected_area_m2 / cell_areas).clip(0.0, 1.0)
    protected_pct = (
        protected_fraction.rename("protected_area_pct")
        .reset_index()
        .merge(cells[["cell_id", "latitude", "longitude"]], on="cell_id", how="left")
    )
    protected_pct["protected_area_pct"] *= 100.0

    output = output.drop(columns=["protected_area_pct"], errors="ignore").merge(
        protected_pct[["latitude", "longitude", "protected_area_pct"]],
        on=["latitude", "longitude"],
        how="left",
    )
    output["protected_area_pct"] = output["protected_area_pct"].fillna(0.0).clip(0.0, 100.0)
    return output


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
        lon_cells = (
            np.floor(
                lon_values[:n_lon].reshape(-1, cols_per_cell).mean(axis=1) / coarse_degree
            )
            * coarse_degree
        )
        dy_m = EARTH_RADIUS_KM * 1000.0 * np.deg2rad(dlat)

        for lat_index, cell_lat in enumerate(requested_lats, start=1):
            lat_lower = cell_lat
            lat_upper = cell_lat + coarse_degree
            pad_lower = lat_lower - dlat * 1.5
            pad_upper = lat_upper + dlat * 1.5

            band = elevation.sel({lat_name: _coord_slice(lat_values, pad_lower, pad_upper)})
            band = band.isel({lon_name: slice(0, n_lon)}).transpose(lat_name, lon_name)
            band_lats = band[lat_name].values.astype(float)
            center_mask = (band_lats >= lat_lower) & (band_lats < lat_upper)
            if center_mask.sum() == 0:
                continue

            elevations = band.to_numpy().astype(np.float32)
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
            suitable = land & (slope[center_mask, :n_lon] <= max_slope_degrees)
            land_counts = land.reshape(land.shape[0], -1, cols_per_cell).sum(axis=(0, 2))
            suitable_counts = suitable.reshape(suitable.shape[0], -1, cols_per_cell).sum(axis=(0, 2))
            suitable_pct = np.divide(
                suitable_counts,
                land_counts,
                out=np.zeros_like(suitable_counts, dtype=float),
                where=land_counts > 0,
            )
            suitable_pct *= 100.0
            frames.append(
                pd.DataFrame(
                    {
                        "latitude": cell_lat,
                        "longitude": lon_cells,
                        "slope_suitable_land_pct": suitable_pct,
                    }
                )
            )

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

    output = df.drop(columns=["slope_suitable_land_pct", "steep_slope_pct"], errors="ignore")
    if frames:
        slope_df = pd.concat(frames, ignore_index=True)
        output = output.merge(slope_df, on=["latitude", "longitude"], how="left")
    else:
        output["slope_suitable_land_pct"] = np.nan
    output["slope_suitable_land_pct"] = (
        output["slope_suitable_land_pct"].fillna(100.0).clip(0.0, 100.0)
    )
    output["steep_slope_pct"] = 100.0 - output["slope_suitable_land_pct"]
    return output


def _apply_land_exclusions(
    availability: pd.DataFrame,
    cfg: LandAvailabilityConfig,
) -> pd.DataFrame:
    output = availability.copy()

    if cfg.protected_area_csv is not None:
        output = _merge_fraction_csv(
            output,
            cfg.protected_area_csv,
            "protected_area_pct",
            ("protected_area_pct", "protected_fraction"),
            0.0,
            cfg.coarse_degree,
        )
    else:
        output = _protected_area_pct_from_vectors(
            output,
            cfg.protected_area_paths,
            cfg.coarse_degree,
            _protected_area_checkpoint_path(cfg.output_csv),
        )

    if cfg.skip_slope_exclusion:
        output["slope_suitable_land_pct"] = 100.0
        output["steep_slope_pct"] = 0.0
    elif cfg.slope_exclusion_csv is not None:
        output = _merge_fraction_csv(
            output,
            cfg.slope_exclusion_csv,
            "slope_suitable_land_pct",
            ("slope_suitable_land_pct", "slope_suitable_fraction", "non_steep_land_pct"),
            100.0,
            cfg.coarse_degree,
        )
        output["steep_slope_pct"] = 100.0 - output["slope_suitable_land_pct"]
    elif cfg.slope_raster_path is not None:
        output = _slope_suitable_land_pct_from_raster(
            output,
            cfg.slope_raster_path,
            cfg.coarse_degree,
            cfg.max_slope_degrees,
        )
    else:
        output["slope_suitable_land_pct"] = 100.0
        output["steep_slope_pct"] = 0.0

    onshore_fraction = (output["onshore_land_pct"] / 100.0).clip(0.0, 1.0)
    protected_fraction = (output["protected_area_pct"] / 100.0).clip(0.0, 1.0)
    protected_factor = pd.Series(0.0, index=output.index)
    onshore_mask = onshore_fraction > 0
    protected_factor.loc[onshore_mask] = (
        1.0 - protected_fraction.loc[onshore_mask] / onshore_fraction.loc[onshore_mask]
    )
    protected_factor = protected_factor.clip(0.0, 1.0)

    slope_factor = (output["slope_suitable_land_pct"] / 100.0).clip(0.0, 1.0)
    output["land_exclusion_factor"] = (
        protected_factor * slope_factor
    ).clip(0.0, 1.0)
    output["constrained_onshore_area_km2"] = (
        output["onshore_area_km2"] * output["land_exclusion_factor"]
    )

    output["solar_availability"] = (
        output["solar_availability"] * output["land_exclusion_factor"]
    ).clip(0.0, 1.0)
    output["wind_onshore_availability"] = (
        output["wind_onshore_availability"] * output["land_exclusion_factor"]
    ).clip(0.0, 1.0)
    output["wind_offshore_availability"] = output["wind_offshore_availability"].clip(0.0, 1.0)
    output["wind_availability"] = (
        output["wind_onshore_availability"] + output["wind_offshore_availability"]
    ).clip(0.0, 1.0)
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


def apply_land_competition_scenario(
    df: pd.DataFrame,
    land_competition_fraction: float,
    source_land_competition_fraction: float | None = None,
) -> pd.DataFrame:
    output = df.copy()
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

    output["land_competition_fraction"] = target_fraction
    if base_land_exclusion_factor is not None:
        output["land_exclusion_factor"] = (base_land_exclusion_factor * target_fraction).clip(0.0, 1.0)
    if base_constrained_onshore_area is not None:
        output["constrained_onshore_area_km2"] = (base_constrained_onshore_area * target_fraction).clip(lower=0.0)
    output["solar_availability"] = (base_solar_availability * target_fraction).clip(0.0, 1.0)
    output["wind_onshore_availability"] = (base_wind_onshore_availability * target_fraction).clip(0.0, 1.0)
    output["wind_offshore_availability"] = output["wind_offshore_availability"].clip(0.0, 1.0)
    output["wind_availability"] = (
        output["wind_onshore_availability"] + output["wind_offshore_availability"]
    ).clip(0.0, 1.0)

    output["solar_area_km2"] = (base_solar_area_km2 * target_fraction).clip(lower=0.0)
    output["wind_onshore_area_km2"] = (base_wind_onshore_area_km2 * target_fraction).clip(lower=0.0)
    output["wind_offshore_area_km2"] = output["wind_offshore_area_km2"].clip(lower=0.0)
    output["wind_area_km2"] = output["wind_onshore_area_km2"] + output["wind_offshore_area_km2"]

    output["max_power_solar_mw"] = (
        output["solar_area_km2"] * output["solar_density_mw_per_km2"]
    ).clip(lower=0.0)
    output["max_power_wind_mw"] = (
        output["wind_area_km2"] * output["wind_density_mw_per_km2"]
    ).clip(lower=0.0)
    output["max_capacity_mw"] = output["max_power_solar_mw"] + output["max_power_wind_mw"]
    output["availability"] = (output["max_capacity_mw"] > 0).astype(float)
    return output


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
    LOGGER.info(
        "Wrote land-competition variant %s from %s with fraction %.3f",
        output_csv,
        base_csv,
        land_competition_fraction,
    )
    return df





def build_land_availability_table(config: LandAvailabilityConfig | None = None) -> pd.DataFrame:
    cfg = (config or LandAvailabilityConfig()).resolved()

    if cfg.land_cover_path.suffix.lower() == ".csv":
        raise ValueError(
            "land_cover_path must point to the MODIS land-cover HDF; CSV shortcuts have been removed."
        )

    if cfg.land_cover_path.suffix.lower() == ".hdf":
        availability = _aggregate_availability_from_hdf4(cfg)
    else:
        try:
            land_cover = _load_land_cover_frame(cfg)
            availability = _aggregate_availability(land_cover, cfg.include_offshore_wind)
        except OSError as exc:
            message = str(exc)
            if "Attempt to use feature that was not turned on when netCDF was built" in message or "NetCDF: Attempt" in message:
                LOGGER.warning(
                    "netCDF4 cannot read %s as HDF4; falling back to pyhdf-based reader.",
                    cfg.land_cover_path,
                )
                availability = _aggregate_availability_from_hdf4(cfg)
            else:
                raise
    availability["area"] = _cell_area_km2(availability["latitude"], cfg.coarse_degree)
    availability = _attach_bathymetry_depth(availability, cfg.bathymetry_path)

    availability["offshore_sea_pct"] = (100.0 - availability["onshore_land_pct"]).clip(lower=0.0, upper=100.0)
    availability["onshore_area_km2"] = availability["area"] * availability["onshore_land_pct"] / 100.0
    availability["offshore_area_km2"] = availability["area"] * availability["offshore_sea_pct"] / 100.0
    availability = _apply_land_exclusions(availability, cfg)

    # Effective siting area: cell area weighted by MODIS-class suitability,
    # protected/slope exclusions, and optionally a post-processed land-
    # competition share.
    availability["solar_area_km2"] = availability["area"] * availability["solar_availability"]
    availability["wind_area_km2"] = availability["area"] * availability["wind_availability"]
    availability["wind_onshore_area_km2"] = (
        availability["area"] * availability["wind_onshore_availability"]
    )
    availability["wind_offshore_area_km2"] = (
        availability["area"] * availability["wind_offshore_availability"]
    )

    availability["solar_density_mw_per_km2"] = _solar_density(availability["latitude"], 1.0)
    availability["wind_density_mw_per_km2"] = _wind_density(
        availability["latitude"], 1.0, WIND_LAND_USE_KM2_PER_GW
    )

    availability = apply_land_competition_scenario(availability, cfg.land_competition_fraction)

    availability["max_power_solar_mw"] = (
        availability["solar_area_km2"] * availability["solar_density_mw_per_km2"]
    ).clip(lower=0.0)
    availability["max_power_wind_mw"] = (
        availability["wind_area_km2"] * availability["wind_density_mw_per_km2"]
    ).clip(lower=0.0)
    availability["max_capacity_mw"] = (
        availability["max_power_solar_mw"]
        + availability["max_power_wind_mw"]
    )
    availability["availability"] = (availability["max_capacity_mw"] > 0).astype(float)

    for column in FINAL_COLUMNS:
        if column not in availability.columns:
            raise ValueError(f"Missing expected column '{column}' in aggregated table.")

    result = availability[FINAL_COLUMNS]
    result = result.sort_values(["latitude", "longitude"]).reset_index(drop=True)
    return result


def write_land_availability_table(config: LandAvailabilityConfig | None = None) -> pd.DataFrame:
    cfg = (config or LandAvailabilityConfig()).resolved()
    df = build_land_availability_table(cfg)
    cfg.output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(cfg.output_csv, index=False)
    checkpoint_path = _protected_area_checkpoint_path(cfg.output_csv)
    if checkpoint_path.exists():
        checkpoint_path.unlink()
        LOGGER.info("Removed protected-area checkpoint %s after successful run", checkpoint_path)
    LOGGER.info("Wrote %s rows to %s", len(df), cfg.output_csv)
    return df


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate the max-capacities CSV for run_global.")
    parser.add_argument(
        "--land-cover",
        type=str,
        default=None,
        help="Path to the MODIS land-cover .hdf file (HDF4/NetCDF).",
    )
    parser.add_argument(
        "--bathymetry",
        type=str,
        default=None,
        help="Path to the bathymetry NetCDF file used to split offshore fixed/floating wind.",
    )
    parser.add_argument(
        "--base-csv",
        type=str,
        default=None,
        help=(
            "Existing max-capacities CSV to rescale to a different land competition fraction quickly, "
            "without recomputing MODIS/protected-area/slope overlays."
        ),
    )
    parser.add_argument("--output", type=str, default=None, help="Destination CSV path.")
    parser.add_argument(
        "--min-lat",
        type=float,
        default=-75.0,
        help="Lower latitude bound (degrees) to include in the aggregation.",
    )
    parser.add_argument(
        "--max-lat",
        type=float,
        default=75.0,
        help="Upper latitude bound (degrees) to include in the aggregation.",
    )
    parser.add_argument(
        "--include-offshore-wind",
        action="store_true",
        help="Include water cells as offshore-wind siting area. Default is onshore-only.",
    )
    parser.add_argument(
        "--land-scenario",
        type=str,
        choices=sorted(LAND_COMPETITION_SCENARIOS),
        default=None,
        help=(
            "Named land-availability scenario. When supplied, overrides "
            "--land-competition-fraction and defaults output naming to "
            "max_capacities_<scenario>.csv."
        ),
    )
    parser.add_argument(
        "--land-competition-fraction",
        type=float,
        default=DEFAULT_LAND_COMPETITION_FRACTION,
        help="Fraction of otherwise suitable onshore land available to shipping fuel demand.",
    )
    parser.add_argument(
        "--source-land-competition-fraction",
        type=float,
        default=None,
        help=(
            "Land competition fraction already baked into --base-csv when the source CSV predates the "
            "land_competition_fraction column."
        ),
    )
    parser.add_argument(
        "--protected-area",
        action="append",
        default=[],
        help="Protected-area vector file to exclude; may be supplied multiple times.",
    )
    parser.add_argument(
        "--protected-area-csv",
        type=str,
        default=None,
        help="Optional precomputed protected-area percentage CSV.",
    )
    parser.add_argument(
        "--slope-raster",
        type=str,
        default=None,
        help="Optional elevation raster/NetCDF used to exclude steep onshore land.",
    )
    parser.add_argument(
        "--slope-exclusion-csv",
        type=str,
        default=None,
        help="Optional precomputed slope suitability percentage CSV.",
    )
    parser.add_argument(
        "--max-slope-degrees",
        type=float,
        default=DEFAULT_MAX_SLOPE_DEGREES,
        help="Maximum terrain slope allowed when --slope-raster is supplied.",
    )
    parser.add_argument(
        "--skip-slope-exclusion",
        action="store_true",
        help="Keep all slopes in the land-availability build and do not apply a slope cutoff.",
    )
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    args = _parse_args()
    land_competition_fraction = (
        land_competition_fraction_for_scenario(args.land_scenario)
        if args.land_scenario is not None
        else args.land_competition_fraction
    )
    default_output_csv = (
        land_availability_output_csv(args.land_scenario)
        if args.land_scenario is not None
        else DEFAULT_OUTPUT_CSV
    )
    output_csv = _resolve_path(args.output, default_output_csv) if args.output else default_output_csv

    if args.base_csv is not None:
        base_csv = _resolve_path(args.base_csv, DEFAULT_OUTPUT_CSV)
        write_land_competition_variant(
            base_csv=base_csv,
            output_csv=output_csv,
            land_competition_fraction=land_competition_fraction,
            source_land_competition_fraction=args.source_land_competition_fraction,
        )
        return

    config = LandAvailabilityConfig(
        land_cover_path=_resolve_path(args.land_cover, DEFAULT_LAND_COVER_FILE),
        bathymetry_path=_resolve_path(args.bathymetry, DEFAULT_BATHYMETRY_FILE),
        output_csv=output_csv,
        lat_bounds=(args.min_lat, args.max_lat),
        include_offshore_wind=args.include_offshore_wind,
        land_competition_fraction=land_competition_fraction,
        protected_area_paths=tuple(Path(path) for path in args.protected_area),
        protected_area_csv=_resolve_optional_path(args.protected_area_csv),
        slope_raster_path=(None if args.skip_slope_exclusion else _resolve_optional_path(args.slope_raster)),
        slope_exclusion_csv=_resolve_optional_path(args.slope_exclusion_csv),
        max_slope_degrees=args.max_slope_degrees,
        skip_slope_exclusion=args.skip_slope_exclusion,
    )
    write_land_availability_table(config)


if __name__ == "__main__":
    main()
