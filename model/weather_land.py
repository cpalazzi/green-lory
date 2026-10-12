"""The weather of a cell's land: a correction of the 1-degree profiles in cells with water.

The 1-degree profiles (data/weather_data/salmon2022_pv_wind_2019_hourly_1deg) average the whole
cell, sea and lakes included: at equal sea share, cells centred at sea and on land have the same
wind, and it rises with the sea share. A plant sits on the cell's land, so in a cell that is
mostly sea it would get a marine wind at an onshore cost (up to 10 % lower LCOA than inland
neighbours in the key run; CAMPAIGN_PLAN.md, review of 8 October 2026).

The correction keeps the profiles and rescales their wind to the land, with ERA5 capacity
factors at 0.25 degrees (atlite, Vestas V112 on land and NREL 5 MW at sea; the green-condor
stores on ARC). For each cell with water, month and hour of day,

    factor = sum over hours of the land-weighted mean of the cell's ERA5 points
             / sum over hours of their area-weighted mean,

each 0.25-degree point weighted by the land (or total) area of its box within the cell. Two
safeguards, because ERA5's coarse coast is unreliable where land is steep (on the Atacama coast
its land points get a capacity factor of 0.003 where wind farms run at 20-30 %):

- in calm bins (the whole-cell mean below CALM_CF) the cell's annual ratio replaces the bin's;
- the corrected wind is scaled back up, if needed, to the mean wind of the cell's fully inland
  neighbours (at least 99 % land), and never above the uncorrected profile: the correction
  removes the sea's contribution but does not make a coast less windy than its hinterland
  (coastal land is slightly windier than inland on average: +0.03 capacity factor).

A profile hour is multiplied by the factor of its month and hour (capped at 1). PV is not
corrected: the stores hold no PV at sea, and PV over sea and land differs little. Cells without
water are unchanged.

    python -m model.weather_land weights            # land and total area of each point's box (GLWD v2)
    python -m model.weather_land ratios --weights inputs/weather/era5_point_weights_1deg.csv \\
        --geography inputs/site_costs/cell_geography.csv \\
        --stores <green-condor outputs> --legacy-dir <weather dir> --output <correction.nc>

`weights` runs where the lakes database is (data/README.md); `ratios` runs on ARC in a job
(arc/jobs/10_weather_land_correction.sh), with an environment that reads zarr.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd

from .config import EARTH_RADIUS_KM, GLWD_AREA_DIR, INPUTS_DIR
from .parameters import file_record, parameter

LOGGER = logging.getLogger(__name__)

WEIGHTS_CSV = INPUTS_DIR / "weather" / "era5_point_weights_1deg.csv"
POINT_STEP = 0.25
# The 5 x 5 ERA5 points whose boxes overlap a cell centred on (lat, lon): point (i, j) sits at
# (lat + 0.5 - 0.25 i, lon - 0.5 + 0.25 j); the boxes of the edge points are half inside.
POINTS = 5
# Boundaries of the point boxes within a cell, in 15-arc-second pixels from its north (west) edge
_GLWD_BLOCKS = (0, 30, 90, 150, 210, 240)
GLWD_PIXELS_PER_DEGREE = 240
GLWD_NORTH = 84.0
GLWD_NODATA = 255
GLWD_OPEN_WATER = ("01", "02", "03", "04", "05", "06")  # lakes, reservoirs, rivers
# The correction's thresholds (inputs/parameters.yaml, weather.land_correction): cells with less
# land than LAND_SHARE_FULL are corrected; bins whose whole-cell ERA5 wind averages below CALM_CF
# take the annual ratio; a neighbour with at least INLAND_LAND_SHARE land counts as fully inland
LAND_SHARE_FULL = float(parameter("weather.land_correction.full_land_share"))
TECHS = ("wind",)
CALM_CF = float(parameter("weather.land_correction.calm_capacity_factor"))
INLAND_LAND_SHARE = float(parameter("weather.land_correction.inland_land_share"))
OPEN_WATER_COLUMNS = ("fresh_lake_km2", "saline_lake_km2", "reservoir_km2", "large_river_km2", "other_permanent_water_km2")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 24), b""):
            digest.update(chunk)
    return digest.hexdigest()


# ---------------------------------------------------------------------------------------------
# Weights (local)
# ---------------------------------------------------------------------------------------------

def _block_sums(field: np.ndarray) -> np.ndarray:
    """(240 rows, 360 x 240 columns) -> (360 cells, 5, 5) sums over the point boxes."""
    rows = np.add.reduceat(field, _GLWD_BLOCKS[:-1], axis=0)  # (5, 86400)
    cells = rows.reshape(POINTS, 360, GLWD_PIXELS_PER_DEGREE)
    return np.add.reduceat(cells, _GLWD_BLOCKS[:-1], axis=2).transpose(1, 0, 2)  # (360, 5, 5)


def build_weights(glwd_dir: Path = GLWD_AREA_DIR, latitudes: list[int] | None = None) -> pd.DataFrame:
    """Land and total area (km2) of each point box within each cell that has land and water.

    Land is the lakes database's land mask less its open water (classes 01-06); the sea is
    outside the mask.
    """
    import rasterio
    from rasterio.windows import Window

    if latitudes is None:
        latitudes = list(range(-55, 84))
    step = 1.0 / GLWD_PIXELS_PER_DEGREE
    sources = {code: rasterio.open(glwd_dir / f"GLWD_v2_0_class_{code}_ha_x10.tif") for code in ("00", *GLWD_OPEN_WATER)}
    frames = []
    try:
        for lat in latitudes:
            top = int(round((GLWD_NORTH - (lat + 0.5)) * GLWD_PIXELS_PER_DEGREE))
            window = Window(0, top, 360 * GLWD_PIXELS_PER_DEGREE, GLWD_PIXELS_PER_DEGREE)
            north_edges = lat + 0.5 - step * np.arange(GLWD_PIXELS_PER_DEGREE)
            pixel_km2 = EARTH_RADIUS_KM ** 2 * np.deg2rad(step) * (
                np.sin(np.deg2rad(north_edges)) - np.sin(np.deg2rad(north_edges - step)))
            valid = sources["00"].read(1, window=window) != GLWD_NODATA
            water_km2 = np.zeros(valid.shape)
            for code in GLWD_OPEN_WATER:
                values = sources[code].read(1, window=window).astype(float)
                water_km2 += np.where(values == GLWD_NODATA, 0.0, values) / 1000.0  # hectares x 10 -> km2
            land = np.clip(valid * pixel_km2[:, None] - water_km2, 0.0, None)
            total = np.broadcast_to(pixel_km2[:, None], valid.shape)
            half = GLWD_PIXELS_PER_DEGREE // 2  # cell b covers b - 180.5 to b - 179.5 degrees
            land_sums = _block_sums(np.roll(land, half, axis=1))
            total_sums = _block_sums(np.roll(total, half, axis=1))
            share = land_sums.sum(axis=(1, 2)) / total_sums.sum(axis=(1, 2))
            keep = (share > 0.0) & (share < LAND_SHARE_FULL)
            for b in np.flatnonzero(keep):
                row = {"latitude": float(lat), "longitude": float(b - 180), "land_share": share[b]}
                row.update({f"land_{i}{j}": land_sums[b, i, j] for i in range(POINTS) for j in range(POINTS)})
                row.update({f"total_{i}{j}": total_sums[b, i, j] for i in range(POINTS) for j in range(POINTS)})
                frames.append(row)
    finally:
        for source in sources.values():
            source.close()
    return pd.DataFrame(frames)


# ---------------------------------------------------------------------------------------------
# Ratios (ARC)
# ---------------------------------------------------------------------------------------------

class _PointRows:
    """Rows of ERA5 points (all longitudes, all hours) read from the tile stores, cached."""

    def __init__(self, stores: Path):
        import zarr

        self.tiles = []
        for path in sorted(stores.glob("global_cf_2019_tiles_*_*.zarr")):
            group = zarr.open_group(str(path), mode="r")
            self.tiles.append((np.asarray(group["lat"][:], dtype=float), group))
        if not self.tiles:
            raise FileNotFoundError(f"No global_cf_2019_tiles_*.zarr in {stores}")
        self.cache: dict[float, dict[str, np.ndarray]] = {}

    def row(self, lat: float) -> dict[str, np.ndarray]:
        key = round(lat, 4)
        if key not in self.cache:
            for lats, group in self.tiles:
                match = np.flatnonzero(np.isclose(lats, lat))
                if len(match):
                    y = int(match[0])
                    self.cache[key] = {"wind": np.asarray(group["cf_wind"][y, :, :], dtype=np.float32)}  # (lon, time)
                    break
            else:
                raise KeyError(f"No ERA5 point row at latitude {lat}")
            for old in [k for k in self.cache if k > key + 0.6]:  # rows north of this cell are done
                del self.cache[old]
        return self.cache[key]


def compute_bin_sums(weights: pd.DataFrame, rows, hours: pd.DatetimeIndex) -> dict[str, np.ndarray]:
    """Sums over each month-and-hour bin of the cells' land-weighted and area-weighted ERA5 wind.

    rows.row(lat) gives the ERA5 points of one latitude: {"wind": (1440 longitudes, hours)}.
    """
    n = len(weights)
    land = weights[[f"land_{i}{j}" for i in range(POINTS) for j in range(POINTS)]].to_numpy().reshape(n, POINTS, POINTS)
    total = weights[[f"total_{i}{j}" for i in range(POINTS) for j in range(POINTS)]].to_numpy().reshape(n, POINTS, POINTS)
    bins = (hours.month.to_numpy() - 1) * 24 + hours.hour.to_numpy()
    land_bins = np.zeros((n, 288))
    cell_bins = np.zeros((n, 288))
    order = np.argsort(-weights["latitude"].to_numpy(), kind="stable")
    started = time.perf_counter()
    by_row = weights.iloc[order].groupby("latitude", sort=False).indices
    for count, (lat, positions) in enumerate(by_row.items(), start=1):
        cells = order[positions]
        point_rows = [rows.row(lat + 0.5 - POINT_STEP * i) for i in range(POINTS)]
        lon = weights["longitude"].to_numpy()[cells]
        columns = (np.round((lon[:, None] - 0.5 + POINT_STEP * np.arange(POINTS)[None, :] + 180.0) / POINT_STEP).astype(int)) % 1440
        series = np.stack([point_rows[i]["wind"][columns] for i in range(POINTS)], axis=1)  # (cells, 5, 5, time)
        valid = np.isfinite(series)
        series = np.where(valid, series, 0.0)
        land_w = land[cells][:, :, :, None] * valid
        total_w = total[cells][:, :, :, None] * valid
        land_mean = (series * land_w).sum(axis=(1, 2)) / np.maximum(land_w.sum(axis=(1, 2)), 1e-12)
        cell_mean = (series * total_w).sum(axis=(1, 2)) / np.maximum(total_w.sum(axis=(1, 2)), 1e-12)
        block_land, block_cell = np.zeros((len(cells), 288)), np.zeros((len(cells), 288))
        np.add.at(block_land.T, bins, land_mean.T)
        np.add.at(block_cell.T, bins, cell_mean.T)
        land_bins[cells], cell_bins[cells] = block_land, block_cell
        if count % 10 == 0:
            LOGGER.info("Weather land correction: latitude %.0f, %d of %d rows, %.0f s", lat, count, len(by_row),
                        time.perf_counter() - started)
    return {"land_bins": land_bins, "cell_bins": cell_bins, "hours_per_bin": np.bincount(bins, minlength=288)}


def wind_factors(land_bins: np.ndarray, cell_bins: np.ndarray, hours_per_bin: np.ndarray) -> np.ndarray:
    """Land/whole-cell ratio per bin, the annual ratio in calm bins (cells, 288)."""
    annual = np.divide(land_bins.sum(axis=1), cell_bins.sum(axis=1), out=np.ones(len(land_bins)),
                       where=cell_bins.sum(axis=1) > 1e-9)
    ratio = np.divide(land_bins, cell_bins, out=np.ones_like(land_bins), where=cell_bins > 1e-12)
    calm = cell_bins / np.maximum(hours_per_bin, 1) < CALM_CF
    return np.where(calm, annual[:, None], ratio)


def apply_inland_floor(factors: np.ndarray, legacy: np.ndarray, bins: np.ndarray, floor: np.ndarray) -> dict[str, np.ndarray]:
    """Scale each cell's factors so that its corrected mean wind is at least
    min(floor, uncorrected mean); floor is NaN where a cell has no fully inland neighbour."""
    uncorrected = legacy.mean(axis=1)
    corrected = np.minimum(legacy * factors[:, bins], 1.0).mean(axis=1)
    target = np.where(np.isnan(floor), -np.inf, np.minimum(floor, uncorrected))
    scaled = factors.copy()
    for _ in range(5):  # the cap at 1 makes the scaling slightly short; a few rounds close it
        short = corrected < target - 1e-6
        if not short.any():
            break
        scaled[short] *= (target[short] / np.maximum(corrected[short], 1e-9))[:, None]
        corrected = np.minimum(legacy * scaled[:, bins], 1.0).mean(axis=1)
    return {"factors": scaled, "uncorrected_mean": uncorrected, "corrected_mean": corrected,
            "floor_applied": factors.mean(axis=1) < scaled.mean(axis=1) - 1e-9}


def inland_floor(cells: pd.DataFrame, geography: pd.DataFrame, legacy_mean: dict) -> np.ndarray:
    """Mean legacy wind of each cell's fully inland neighbours (the 8 around it), NaN if none."""
    land = geography["land_km2"] - geography[list(OPEN_WATER_COLUMNS)].sum(axis=1)
    inland = geography.loc[land / geography["cell_area_km2"] >= INLAND_LAND_SHARE, ["latitude", "longitude"]]
    inland_keys = set(zip(inland["latitude"].round(4), inland["longitude"].round(4)))
    floors = np.full(len(cells), np.nan)
    for k, (lat, lon) in enumerate(zip(cells["latitude"], cells["longitude"])):
        values = []
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                key = (round(lat + dy, 4), round(((lon + dx + 180.0) % 360.0) - 180.0, 4))
                if (dy or dx) and key in inland_keys and key in legacy_mean:
                    values.append(legacy_mean[key])
        if values:
            floors[k] = float(np.mean(values))
    return floors


def _legacy_wind(legacy_dir: Path) -> tuple[dict, callable]:
    """The legacy wind stack in memory: annual means by cell, and a cell's hourly series."""
    import xarray as xr

    from .weather import resource_files

    arrays = []
    for path in resource_files(legacy_dir, "windpowers"):
        with xr.open_dataset(path) as dataset:
            arrays.append(dataset["Wind"].astype(np.float32).load())
    means = {}
    for array in arrays:
        annual = array.mean("time").values
        for i, lat in enumerate(array["latitude"].values):
            for j, lon in enumerate(array["longitude"].values):
                means[(round(float(lat), 4), round(float(lon), 4))] = float(annual[i, j])

    def series(lat: float, lon: float) -> np.ndarray:
        for array in arrays:
            if lon in array["longitude"].values:
                return array.sel(latitude=lat, longitude=lon).values
        raise KeyError(f"No legacy wind at ({lat}, {lon})")

    return means, series


def alignment_check(weights: pd.DataFrame, series, stores: Path, sample: int = 40) -> dict:
    """Correlation of the legacy wind profile with the ERA5 whole-cell mean at lags of -3 to 3 h,
    for a sample of cells (a check that both are hourly UTC series of the same weather)."""
    rows = _PointRows(stores)
    rng = np.random.default_rng(0)
    picks = rng.choice(len(weights), size=min(sample, len(weights)), replace=False)
    best = []
    for k in picks:
        lat, lon = float(weights["latitude"].iat[k]), float(weights["longitude"].iat[k])
        total = weights.iloc[k][[f"total_{i}{j}" for i in range(POINTS) for j in range(POINTS)]].to_numpy(float).reshape(POINTS, POINTS)
        columns = (np.round((lon - 0.5 + POINT_STEP * np.arange(POINTS) + 180.0) / POINT_STEP).astype(int)) % 1440
        era5 = sum(total[i, j] * rows.row(lat + 0.5 - POINT_STEP * i)["wind"][columns[j]] for i in range(POINTS) for j in range(POINTS)) / total.sum()
        old = series(lat, lon)
        lags = range(-3, 4)
        correlations = [np.corrcoef(np.roll(era5, lag), old)[0, 1] for lag in lags]
        best.append((list(lags)[int(np.nanargmax(correlations))], float(np.nanmax(correlations)),
                     float(np.nanmean(old)), float(np.nanmean(era5))))
    frame = pd.DataFrame(best, columns=["best_lag_h", "correlation", "legacy_mean_cf", "era5_cell_mean_cf"])
    return {"cells": int(len(frame)), "best_lag_counts": {int(k): int(v) for k, v in frame.best_lag_h.value_counts().items()},
            "median_correlation": float(frame.correlation.median()),
            "median_legacy_over_era5": float((frame.legacy_mean_cf / frame.era5_cell_mean_cf).median())}


def write_correction(weights_csv: Path, geography_csv: Path, stores: Path, legacy_dir: Path, output: Path) -> None:
    import xarray as xr

    weights = pd.read_csv(weights_csv)
    hours = pd.date_range("2019-01-01", periods=8760, freq="h")
    bins = (hours.month.to_numpy() - 1) * 24 + hours.hour.to_numpy()
    sums = compute_bin_sums(weights, _PointRows(stores), hours)
    factors = wind_factors(sums["land_bins"], sums["cell_bins"], sums["hours_per_bin"])
    legacy_mean, series = _legacy_wind(legacy_dir)
    legacy = np.stack([series(lat, lon) for lat, lon in zip(weights["latitude"], weights["longitude"])])
    floor = inland_floor(weights, pd.read_csv(geography_csv), legacy_mean)
    result = apply_inland_floor(factors, legacy, bins, floor)
    check = alignment_check(weights, series, stores)
    LOGGER.info("Alignment check: %s; floor applied in %d of %d cells", check, int(result["floor_applied"].sum()), len(weights))
    hours_per_bin = np.maximum(sums["hours_per_bin"], 1)
    dataset = xr.Dataset(
        {
            "factor": (("cell", "tech", "month", "hour"), result["factors"].reshape(len(weights), 1, 12, 24).astype(np.float32)),
            "era5_land_mean_cf": ("cell", (sums["land_bins"].sum(axis=1) / hours_per_bin.sum()).astype(np.float32)),
            "era5_cell_mean_cf": ("cell", (sums["cell_bins"].sum(axis=1) / hours_per_bin.sum()).astype(np.float32)),
            "legacy_mean_cf": ("cell", result["uncorrected_mean"].astype(np.float32)),
            "corrected_mean_cf": ("cell", result["corrected_mean"].astype(np.float32)),
            "inland_floor_cf": ("cell", floor.astype(np.float32)),
            "floor_applied": ("cell", result["floor_applied"]),
            "land_share": ("cell", weights["land_share"].to_numpy(np.float32)),
            "latitude": ("cell", weights["latitude"].to_numpy(float)),
            "longitude": ("cell", weights["longitude"].to_numpy(float)),
        },
        coords={"tech": list(TECHS), "month": np.arange(1, 13), "hour": np.arange(24)},
        attrs={
            "description": "Factors that rescale the wind of the 1-degree profiles of cells with water to the cell's "
                           "land (model/weather_land.py); hours are UTC; PV is not corrected",
            "calm_cf": CALM_CF, "inland_land_share": INLAND_LAND_SHARE,
            "parameters_yaml": file_record()["path"], "parameters_sha256": file_record()["sha256"],
            "weights_csv": str(weights_csv), "weights_sha256": _sha256(weights_csv),
            "geography_csv": str(geography_csv), "geography_sha256": _sha256(geography_csv),
            "era5_stores": str(stores), "alignment_check": json.dumps(check),
        },
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_netcdf(output, engine="h5netcdf", encoding={"factor": {"zlib": True, "complevel": 4}})
    LOGGER.info("Wrote %s (%d cells)", output, len(weights))


# ---------------------------------------------------------------------------------------------
# Applying the correction (read by model/weather.py)
# ---------------------------------------------------------------------------------------------

class LandCorrection:
    """The factors of a correction file, by cell; cells without a row are left unchanged."""

    def __init__(self, path: str | Path):
        import xarray as xr

        with xr.open_dataset(path) as dataset:
            self.factor = dataset["factor"].values.astype(np.float64)  # (cell, tech, month, hour)
            self.techs = [str(t) for t in dataset["tech"].values]
            keys = zip(np.round(dataset["latitude"].values, 4), np.round(dataset["longitude"].values, 4))
            self.index = {(float(lat), float(lon)): k for k, (lat, lon) in enumerate(keys)}
        self.path = Path(path)

    def factors(self, lat: float, lon: float, month: np.ndarray, hour: np.ndarray) -> dict[str, np.ndarray] | None:
        """Hourly factors per technology for a cell, or None when the cell has no water."""
        k = self.index.get((round(float(lat), 4), round(float(lon), 4)))
        if k is None:
            return None
        return {tech: self.factor[k, t][month - 1, hour] for t, tech in enumerate(self.techs)}


# ---------------------------------------------------------------------------------------------
# Command line
# ---------------------------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    weights = commands.add_parser("weights", help="land and total area of each ERA5 point box per cell")
    weights.add_argument("--output", type=Path, default=WEIGHTS_CSV)
    ratios = commands.add_parser("ratios", help="the correction factors from the ERA5 stores (ARC)")
    ratios.add_argument("--weights", type=Path, default=WEIGHTS_CSV)
    ratios.add_argument("--geography", type=Path, default=INPUTS_DIR / "site_costs" / "cell_geography.csv")
    ratios.add_argument("--stores", type=Path, required=True)
    ratios.add_argument("--legacy-dir", type=Path, required=True)
    ratios.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if args.command == "weights":
        frame = build_weights()
        args.output.parent.mkdir(parents=True, exist_ok=True)
        frame.round(4).to_csv(args.output, index=False)
        print(f"wrote {args.output}: {len(frame)} cells with land and water")
    else:
        write_correction(args.weights, args.geography, args.stores, args.legacy_dir, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
