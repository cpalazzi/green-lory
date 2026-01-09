"""Utilities for deriving land-availability inputs for the run_global workflow.

The module aggregates the 0.05° MODIS land-cover tiles to 1° cells, applies
simple suitability factors for wind/solar siting, and estimates a maximum
installable capacity (in MW) that can be consumed directly by run_global.
"""
from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import xarray as xr

LOGGER = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data"
DEFAULT_LAND_COVER_FILE = DATA_DIR / "MCD12C1.A2022001.061.2023244164746.hdf"
DEFAULT_OUTPUT_CSV = DATA_DIR / "20251222_land_max_capacity.csv"
EARTH_RADIUS_KM = 6371.0
MODIS_WATER_CLASS = 0
WIND_LAND_USE_KM2_PER_GW = 200.0  # Salmon et al. (2021)
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
MODIS_CLASS_WIND_AVAILABILITY = {
    0: 1.0,  # Open water / offshore
    6: 0.5,  # Closed shrublands
    7: 0.5,  # Open shrublands
    8: 0.15,  # Woody savannas
    9: 0.2,  # Savannas
    10: 0.2,  # Grasslands
    16: 1.0,  # Barren / sparsely vegetated
}

FINAL_COLUMNS = [
    "latitude",
    "longitude",
    "availability",
    "area",
    "wind_availability",
    "solar_availability",
    "wind_area_km2",
    "solar_area_km2",
    "onshore_land_pct",
    "wind_density_mw_per_km2",
    "solar_density_mw_per_km2",
    "wind_max_capacity",
    "solar_max_capacity",
    "max_capacity",
]


def _resolve_path(path: str | Path | None, default: Path) -> Path:
    if path is None:
        return default
    resolved = Path(path)
    if not resolved.is_absolute():
        resolved = REPO_ROOT / resolved
    return resolved
def _wind_density(latitudes: pd.Series, scale: float = 1.0) -> pd.Series:
    base_density = 1000.0 / WIND_LAND_USE_KM2_PER_GW  # MW per km²
    densities = np.full(latitudes.shape[0], base_density * scale)
    return pd.Series(densities, index=latitudes.index)


def _solar_density(latitudes: pd.Series, scale: float = 1.0) -> pd.Series:
    lat_abs = np.abs(latitudes.to_numpy())
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
    density_mw_per_km2 = 1_000_000.0 / area_per_mw_m2
    densities = density_mw_per_km2 * scale
    return pd.Series(densities, index=latitudes.index)


@dataclass
class LandAvailabilityConfig:
    """Runtime knobs for the land-cover aggregation pipeline."""

    land_cover_path: Path = DEFAULT_LAND_COVER_FILE
    output_csv: Path = DEFAULT_OUTPUT_CSV
    coarse_degree: float = 1.0
    fine_degree: float = 0.05
    lat_bounds: Tuple[float, float] = (-75.0, 75.0)
    reference_capacity_csv: Path | None = None

    def resolved(self) -> "LandAvailabilityConfig":
        return LandAvailabilityConfig(
            land_cover_path=_resolve_path(self.land_cover_path, DEFAULT_LAND_COVER_FILE),
            output_csv=_resolve_path(self.output_csv, DEFAULT_OUTPUT_CSV),
            coarse_degree=self.coarse_degree,
            fine_degree=self.fine_degree,
            lat_bounds=self.lat_bounds,
            reference_capacity_csv=(
                _resolve_path(self.reference_capacity_csv, DEFAULT_OUTPUT_CSV)
                if self.reference_capacity_csv is not None
                else None
            ),
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

        factors = np.zeros(int(n_classes), dtype=float)
        for klass, weight in MODIS_CLASS_WIND_AVAILABILITY.items():
            if 0 <= int(klass) < n_classes:
                factors[int(klass)] = float(weight)

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

            # wind_component_fine = sum_class(class_fraction * factor)
            wind_component_fine = np.tensordot(cube, factors, axes=([2], [0]))
            water_fraction_fine = cube[:, :, water_class]

            # Reduce to coarse cells: average across fine y and fine x.
            # Shapes: (group_size, x_end) -> (n_lon_coarse,)
            wind_band = wind_component_fine.reshape(group_size, -1, group_size).mean(axis=(0, 2))
            water_band = water_fraction_fine.reshape(group_size, -1, group_size).mean(axis=(0, 2))

            lat_center = 90.0 - config.fine_degree * float(y0)
            lat_coarse = np.floor(lat_center / config.coarse_degree) * config.coarse_degree

            for lon, wind_val, water_val in zip(lon_coarse, wind_band, water_band):
                wind_val = float(np.clip(wind_val, 0.0, 1.0))
                water_val = float(np.clip(water_val, 0.0, 1.0))
                onshore = float(np.clip(1.0 - water_val, 0.0, 1.0))
                solar_val = onshore
                avail = float(max(wind_val, solar_val))
                rows.append(
                    {
                        "latitude": float(lat_coarse),
                        "longitude": float(lon),
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


def _aggregate_availability(df: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        df.groupby(["latitude", "longitude", "modis_class"], as_index=False)["class_fraction"]
        .mean()
    )

    grouped["wind_component"] = (
        grouped["class_fraction"] * grouped["modis_class"].map(MODIS_CLASS_WIND_AVAILABILITY).fillna(0.0)
    )

    availability = (
        grouped.groupby(["latitude", "longitude"], as_index=False)["wind_component"]
        .sum()
        .rename(columns={"wind_component": "wind_availability"})
    )
    availability["wind_availability"] = availability["wind_availability"].clip(0.0, 1.0)

    water_fraction = (
        grouped.loc[grouped["modis_class"] == MODIS_WATER_CLASS, ["latitude", "longitude", "class_fraction"]]
        .groupby(["latitude", "longitude"], as_index=False)
        .sum()
        .rename(columns={"class_fraction": "water_fraction"})
    )
    availability = availability.merge(water_fraction, on=["latitude", "longitude"], how="left")
    availability["water_fraction"] = availability["water_fraction"].fillna(0.0).clip(0.0, 1.0)
    availability["onshore_land_pct"] = (1.0 - availability["water_fraction"]).clip(0.0, 1.0) * 100.0

    availability["solar_availability"] = (availability["onshore_land_pct"] / 100.0).clip(0.0, 1.0)
    availability["availability"] = availability[["wind_availability", "solar_availability"]].max(axis=1)
    availability["availability"] = availability["availability"].clip(0.0, 1.0)

    return availability.drop(columns=["water_fraction"])


def _merge_reference_data(base: pd.DataFrame, reference_path: Path | None) -> pd.DataFrame:
    if reference_path is None:
        return base
    if not reference_path.exists():
        LOGGER.warning("Reference CSV %s not found; skipping merge.", reference_path)
        return base

    reference = pd.read_csv(reference_path)
    reference.columns = [col.lower() for col in reference.columns]
    required = {"latitude", "longitude"}
    missing = required - set(reference.columns)
    if missing:
        LOGGER.warning("Reference CSV missing columns: %s", ", ".join(sorted(missing)))
        return base

    merged = base.merge(reference, on=["latitude", "longitude"], how="left", suffixes=("", "_ref"))
    override_columns = [
        "max_capacity",
        "wind_max_capacity",
        "solar_max_capacity",
    ]
    for column in override_columns:
        ref_column = f"{column}_ref"
        if ref_column not in merged.columns:
            continue
        if column not in merged.columns:
            merged[column] = np.nan
        merged[column] = merged[ref_column].fillna(merged[column])
        merged = merged.drop(columns=[ref_column])

    return merged


def build_land_availability_table(config: LandAvailabilityConfig | None = None) -> pd.DataFrame:
    cfg = (config or LandAvailabilityConfig()).resolved()

    if cfg.land_cover_path.suffix.lower() == ".csv":
        raise ValueError(
            "land_cover_path must point to the MODIS land-cover HDF; CSV shortcuts have been removed."
        )

    try:
        land_cover = _load_land_cover_frame(cfg)
        availability = _aggregate_availability(land_cover)
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

    availability["wind_area_km2"] = availability["wind_availability"] * availability["area"]
    availability["solar_area_km2"] = availability["solar_availability"] * availability["area"]
    availability["wind_density_mw_per_km2"] = _wind_density(availability["latitude"], 1.0)
    availability["solar_density_mw_per_km2"] = _solar_density(availability["latitude"], 1.0)
    availability["wind_max_capacity"] = (
        availability["wind_area_km2"] * availability["wind_density_mw_per_km2"]
    )
    availability["solar_max_capacity"] = (
        availability["solar_area_km2"] * availability["solar_density_mw_per_km2"]
    )
    availability["wind_max_capacity"] = availability["wind_max_capacity"].clip(lower=0.0)
    availability["solar_max_capacity"] = availability["solar_max_capacity"].clip(lower=0.0)
    availability["max_capacity"] = (
        availability["wind_max_capacity"] + availability["solar_max_capacity"]
    )

    merged = _merge_reference_data(availability, cfg.reference_capacity_csv)

    for column in FINAL_COLUMNS:
        if column not in merged.columns:
            raise ValueError(f"Missing expected column '{column}' in aggregated table.")

    merged = merged[FINAL_COLUMNS]
    merged = merged.sort_values(["latitude", "longitude"]).reset_index(drop=True)
    return merged


def write_land_availability_table(config: LandAvailabilityConfig | None = None) -> pd.DataFrame:
    cfg = (config or LandAvailabilityConfig()).resolved()
    df = build_land_availability_table(cfg)
    cfg.output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(cfg.output_csv, index=False)
    LOGGER.info("Wrote %s rows to %s", len(df), cfg.output_csv)
    return df


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate the land-availability CSV for run_global.")
    parser.add_argument(
        "--land-cover",
        type=str,
        default=None,
        help="Path to the MODIS land-cover .hdf file (HDF4/NetCDF).",
    )
    parser.add_argument("--output", type=str, default=None, help="Destination CSV path.")
    parser.add_argument(
        "--reference-csv",
        type=str,
        default=None,
        help="Optional CSV with Latitude/Longitude pairs containing known capacity overrides.",
    )
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
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    config = LandAvailabilityConfig(
        land_cover_path=_resolve_path(args.land_cover, DEFAULT_LAND_COVER_FILE),
        output_csv=_resolve_path(args.output, DEFAULT_OUTPUT_CSV) if args.output else DEFAULT_OUTPUT_CSV,
        lat_bounds=(args.min_lat, args.max_lat),
        reference_capacity_csv=_resolve_path(args.reference_csv, DEFAULT_OUTPUT_CSV)
        if args.reference_csv
        else None,
    )
    write_land_availability_table(config)


if __name__ == "__main__":
    main()
