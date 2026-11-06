"""One-degree PV and wind profiles from the ERA5 2019 cutout (CAMPAIGN_PLAN.md, section 7a).

They replace Salmon's profiles (data/weather_data/salmon2022_pv_wind_2019_hourly_1deg), whose
turbine, hub height and panel are not documented and whose PV is DC output per MW of modules
(data/README.md). Built with atlite 0.4.1 from the green-condor cutout on ARC (0.25 degrees,
8,760 hours of 2019), with the settings of inputs/parameters.yaml (weather.profiles):

- wind: a reference turbine with its wind speeds stretched to DEA 2050's specific power, at DEA's
  hub height; the cutout's 100 m wind is extrapolated with its roughness (log law), then
  normalised by the air density of the standard atmosphere at the hub;
- PV: fixed at the latitude-optimal tilt and single-axis tracking, output per MW of AC (DEA's cost
  basis): atlite's output per MW of modules x DEA's DC/AC ratio, capped at 1.

Each one-degree cell (centred on whole degrees, as the weather stack) gets a land profile, its
ERA5 points weighted by their land area within the cell, and a sea profile, weighted by their
water area. The weights are those of model/weather_land.py for cells with land and water
(inputs/weather/era5_point_weights_1deg.csv); other cells are land if their land share
(inputs/site_costs/cell_geography.csv) reaches weather.land_correction.full_land_share and sea if
they have no land, with a land profile too, from all their points, since the land table counts
atolls the geography does not. A land run's stack holds the land profiles (Solar, SolarTracking, WindPowers);
a sea run's (stack --sea) holds the sea profile of the offshore turbine as its WindPowers, and no
PV output. The wake factor (weather.wind_wake_factor) is applied when the profiles are loaded
(model/weather.py), not here.

    python -m model.weather_profiles check --cutout <cutout.nc> --output <report.json>
    python -m model.weather_profiles convert --cutout <cutout.nc> --cells <south>:<north> --output <band.nc>
    python -m model.weather_profiles stack --bands <band dir> --output <weather dir>

`check` and `convert` need atlite: they run on ARC in jobs (arc/jobs/11_weather_profiles.sh), with
the green-condor environment.
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
import xarray as xr

from .config import EARTH_RADIUS_KM, INPUTS_DIR
from .parameters import file_record, parameter

LOGGER = logging.getLogger(__name__)

WEIGHTS_CSV = INPUTS_DIR / "weather" / "era5_point_weights_1deg.csv"
GEOGRAPHY_CSV = INPUTS_DIR / "site_costs" / "cell_geography.csv"
WIND_ONSHORE = dict(parameter("weather.profiles.wind_onshore"))
WIND_OFFSHORE = dict(parameter("weather.profiles.wind_offshore"))
SMOOTH_POWER_CURVE = bool(parameter("weather.profiles.smooth_power_curve"))
AIR_DENSITY_CORRECTION = bool(parameter("weather.profiles.air_density_correction"))
PV_PANEL = str(parameter("weather.profiles.pv_panel"))
PV_ORIENTATION = str(parameter("weather.profiles.pv_orientation"))
PV_TRACKING = str(parameter("weather.profiles.pv_tracking"))
DC_AC_RATIO = float(parameter("weather.profiles.dc_ac_ratio"))
FULL_LAND_SHARE = float(parameter("weather.land_correction.full_land_share"))
WIND_WAKE_FACTOR = float(parameter("weather.wind_wake_factor"))

# The cells of the weather stack and its files: three longitude bands per technology, as
# Salmon's (unsuffixed 180 W to 61 W, 1: 60 W to 59 E, 2: 60 E to 179 E)
CELL_LATITUDES = np.arange(-90, 90)
CELL_LONGITUDES = np.arange(-180, 180)
LONGITUDE_BANDS = (("", -180, -61), ("1", -60, 59), ("2", 60, 179))
# band variable -> (file-name prefix, NetCDF variable), as model/weather.py reads them
STACK_FILES = {
    "solar": ("Solar", "Solar"),
    "solar_tracking": ("SolarTracking", "Solar"),
    "wind": ("WindPowers", "Wind"),
}

# Standard atmosphere (ISO 2533) below 11 km: rho / rho0 = (1 - L h / T0)^(g M / (R L) - 1)
SEA_LEVEL_TEMPERATURE_K = 288.15
LAPSE_RATE_K_PER_M = 0.0065
DENSITY_EXPONENT = 9.80665 * 0.0289644 / (8.31446 * LAPSE_RATE_K_PER_M) - 1.0  # 4.2559

# The land share of a cell without land, so that it has a land profile (band_cells)
MIN_LAND_SHARE = 1e-6

# DEA's reference conditions for the check: 2050 full-load hours per MW (onshore and offshore
# wind net of wakes; PV per MW of AC) and the cells that stand for them (Denmark and its North Sea)
DEA_FULL_LOAD_HOURS = {"wind": 3800.0, "wind_offshore": 4650.0, "solar": 1515.0, "solar_tracking": 1742.0}
DEA_LAND_CELLS = ((55, 9), (56, 9), (57, 9), (56, 10), (55, 12))
DEA_SEA_CELLS = ((55, 6), (55, 7), (56, 6), (56, 7))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 24), b""):
            digest.update(chunk)
    return digest.hexdigest()


# ---------------------------------------------------------------------------------------------
# Turbines, air density and PV
# ---------------------------------------------------------------------------------------------

def specific_power_w_per_m2(power_mw: float, rotor_m: float) -> float:
    """Rated power per swept area of the rotor."""
    return power_mw * 1e6 / (np.pi * (rotor_m / 2.0) ** 2)


def speed_scale(settings: dict) -> float:
    """(DEA / reference specific power)^(1/3), the factor on the reference curve's wind speeds."""
    target = specific_power_w_per_m2(settings["power_mw"], settings["rotor_m"])
    reference = specific_power_w_per_m2(settings["reference_power_mw"], settings["reference_rotor_m"])
    return (target / reference) ** (1.0 / 3.0)


def stretched_turbine(reference: dict, settings: dict) -> dict:
    """The reference power curve with its wind speeds x speed_scale, at DEA's hub height."""
    return {**reference, "V": np.asarray(reference["V"], dtype=float) * speed_scale(settings),
            "POW": np.asarray(reference["POW"], dtype=float), "hub_height": float(settings["hub_height_m"])}


def air_density_ratio(height_m):
    """Air density over its sea-level value in the standard atmosphere (heights below 11 km)."""
    height = np.clip(height_m, 0.0, None)
    return (1.0 - LAPSE_RATE_K_PER_M * height / SEA_LEVEL_TEMPERATURE_K) ** DENSITY_EXPONENT


def pv_per_mw_ac(output_per_mw_modules, dc_ac_ratio: float = DC_AC_RATIO):
    """PV output per MW of AC: output per MW of modules x DC/AC, capped at the inverter's 1."""
    return np.minimum(output_per_mw_modules * dc_ac_ratio, 1.0)


def convert_wind_at_hub(ds: xr.Dataset, turbine: dict, density_correction: bool,
                        interpolation_method: str = "logarithmic") -> xr.DataArray:
    """atlite's wind conversion with the wind speed at the hub normalised by air density.

    A conversion function for Cutout.convert_and_aggregate: as atlite.convert.convert_wind, but
    the hub-height speed is multiplied by (rho / rho0)^(1/3), rho the standard atmosphere's at the
    point's height (the cutout's `height`) plus the hub height.
    """
    from atlite import wind as atlite_wind

    speeds, power, hub_height = turbine["V"], turbine["POW"], turbine["hub_height"]
    rated = float(np.max(power))
    hub_speed = atlite_wind.extrapolate_wind_speed(ds, to_height=hub_height, method=interpolation_method)
    if density_correction:
        hub_speed = hub_speed * air_density_ratio(ds["height"] + hub_height) ** (1.0 / 3.0)
    output = xr.apply_ufunc(lambda v: np.interp(v, speeds, power / rated), hub_speed,
                            dask="parallelized", output_dtypes=[hub_speed.dtype])
    output.attrs["units"] = "MWh/MW"
    return output.rename("specific generation")


# ---------------------------------------------------------------------------------------------
# Point weights of each cell
# ---------------------------------------------------------------------------------------------

def _interval_overlap(lower_a, upper_a, lower_b, upper_b):
    return np.clip(np.minimum(upper_a, upper_b) - np.maximum(lower_a, lower_b), 0.0, None)


def _latitude_weights(point_lats: np.ndarray, step: float, cell_lat: float) -> np.ndarray:
    """Area factor (km2 per degree of longitude) of each point's box within the cell's latitudes."""
    south = np.maximum(np.maximum(point_lats - step / 2, cell_lat - 0.5), -90.0)
    north = np.minimum(np.minimum(point_lats + step / 2, cell_lat + 0.5), 90.0)
    inside = north > south
    band = np.sin(np.deg2rad(np.where(inside, north, 0.0))) - np.sin(np.deg2rad(np.where(inside, south, 0.0)))
    return np.where(inside, EARTH_RADIUS_KM ** 2 * np.deg2rad(1.0) * band, 0.0)


def _longitude_weights(point_lons: np.ndarray, step: float, cell_lon: float) -> np.ndarray:
    """Degrees of longitude of each point's box within the cell (across the date line too)."""
    offset = (point_lons - cell_lon + 180.0) % 360.0 - 180.0
    return _interval_overlap(offset - step / 2, offset + step / 2, -0.5, 0.5)


def cell_point_weights(point_lats: np.ndarray, point_lons: np.ndarray, cells: pd.DataFrame,
                       point_weights: pd.DataFrame | None = None) -> pd.DataFrame:
    """Land and total area (km2) of each ERA5 point's box within each cell.

    point_lats and point_lons are the cutout's coordinates (regular). cells holds latitude,
    longitude and land_share. point_weights (model/weather_land.py, 0.25-degree points) gives the
    land of cells with land and water; elsewhere a cell's land is all of it (land_share at least
    FULL_LAND_SHARE) or none (no land), and in between, without the table, its land share.
    Returns one row per (cell, point) with total_km2 > 0: cell, iy, ix, total_km2, land_km2.
    """
    lat_step = float(np.median(np.diff(point_lats)))
    lon_step = float(np.median(np.diff(point_lons)))
    exact = point_weights is not None and np.isclose(lat_step, 0.25) and np.isclose(lon_step, 0.25)
    table = None
    if exact:
        table = point_weights.set_index(["latitude", "longitude"])
    rows = []
    for position, cell in enumerate(cells.itertuples(index=False)):
        lat_w = _latitude_weights(point_lats, lat_step, cell.latitude)
        lon_w = _longitude_weights(point_lons, lon_step, cell.longitude)
        iy, ix = np.flatnonzero(lat_w > 0), np.flatnonzero(lon_w > 0)
        if not len(iy) or not len(ix):
            continue
        total = lat_w[iy][:, None] * lon_w[ix][None, :]
        key = (float(cell.latitude), float(cell.longitude))
        if table is not None and key in table.index:
            record = table.loc[key]
            land = np.zeros_like(total)
            for i in range(5):
                for j in range(5):
                    y = cell.latitude + 0.5 - 0.25 * i
                    x = (cell.longitude - 0.5 + 0.25 * j + 180.0) % 360.0 - 180.0
                    a = np.flatnonzero(np.isclose(point_lats[iy], y))
                    b = np.flatnonzero(np.isclose((point_lons[ix] + 180.0) % 360.0 - 180.0, x))
                    if len(a) and len(b):
                        land[a[0], b[0]] = min(float(record[f"land_{i}{j}"]), total[a[0], b[0]])
        elif cell.land_share >= FULL_LAND_SHARE:
            land = total.copy()
        elif cell.land_share <= 0.0:
            land = np.zeros_like(total)
        else:
            land = total * float(cell.land_share)
        grid_y, grid_x = np.meshgrid(iy, ix, indexing="ij")
        rows.append(pd.DataFrame({"cell": position, "iy": grid_y.ravel(), "ix": grid_x.ravel(),
                                  "total_km2": total.ravel(), "land_km2": land.ravel()}))
    if not rows:
        return pd.DataFrame(columns=["cell", "iy", "ix", "total_km2", "land_km2"])
    return pd.concat(rows, ignore_index=True)


def aggregation_matrix(weights: pd.DataFrame, kind: str, n_cells: int, n_lat: int, n_lon: int):
    """Sparse matrix (cells x points, in atlite's grid order: latitude outer, longitude inner)
    of each cell's land ('land') or water ('sea') weights, rows summing to 1 or 0 (no such area)."""
    from scipy.sparse import csr_matrix

    values = weights["land_km2"].to_numpy() if kind == "land" else (weights["total_km2"] - weights["land_km2"]).clip(lower=0.0).to_numpy()
    sums = np.bincount(weights["cell"].to_numpy(), weights=values, minlength=n_cells)
    keep = values > 0
    rows = weights["cell"].to_numpy()[keep]
    data = values[keep] / sums[rows]
    columns = weights["iy"].to_numpy()[keep] * n_lon + weights["ix"].to_numpy()[keep]
    return csr_matrix((data, (rows, columns)), shape=(n_cells, n_lat * n_lon)), sums > 0


def band_cells(south: int, north: int, geography: pd.DataFrame | None = None) -> pd.DataFrame:
    """The cells of latitudes south to north (inclusive), all longitudes, with their land share."""
    lat, lon = np.meshgrid(np.arange(south, north + 1), CELL_LONGITUDES, indexing="ij")
    cells = pd.DataFrame({"latitude": lat.ravel().astype(float), "longitude": lon.ravel().astype(float)})
    if geography is None:
        geography = pd.read_csv(GEOGRAPHY_CSV, usecols=["latitude", "longitude", "cell_area_km2", "land_km2"])
    shares = geography.assign(land_share=geography["land_km2"] / geography["cell_area_km2"])[["latitude", "longitude", "land_share"]]
    cells = cells.merge(shares, on=["latitude", "longitude"], how="left")
    # beyond the geography table: sea. A cell of no land still gets a land profile from all its
    # points, its sea profile unchanged: the land table, from MODIS, holds atolls the geography lacks
    cells["land_share"] = cells["land_share"].fillna(0.0).clip(lower=MIN_LAND_SHARE)
    return cells


# ---------------------------------------------------------------------------------------------
# Conversion (atlite)
# ---------------------------------------------------------------------------------------------

def _turbine(settings: dict, smooth: bool) -> dict:
    from atlite.resource import get_windturbineconfig, windturbine_smooth

    turbine = stretched_turbine(get_windturbineconfig(settings["reference_turbine"]), settings)
    return windturbine_smooth(turbine) if smooth else turbine


TECHNOLOGIES = ("wind", "wind_offshore", "solar", "solar_tracking")


def convert_cells(cutout, cells: pd.DataFrame, point_weights: pd.DataFrame | None,
                  technologies: tuple[str, ...] = TECHNOLOGIES, smooth: bool = SMOOTH_POWER_CURVE) -> xr.Dataset:
    """The land and sea profiles of the given cells from an atlite cutout covering them."""
    point_lats = np.asarray(cutout.data["y"].values, dtype=float)
    point_lons = np.asarray(cutout.data["x"].values, dtype=float)
    weights = cell_point_weights(point_lats, point_lons, cells, point_weights)
    n = len(cells)
    matrices = {kind: aggregation_matrix(weights, kind, n, len(point_lats), len(point_lons)) for kind in ("land", "sea")}
    index = pd.RangeIndex(n, name="cell")
    runs = {
        "wind": ("land", lambda m: cutout.convert_and_aggregate(
            convert_func=convert_wind_at_hub, matrix=m, index=index, turbine=_turbine(WIND_ONSHORE, smooth),
            density_correction=AIR_DENSITY_CORRECTION)),
        "wind_offshore": ("sea", lambda m: cutout.convert_and_aggregate(
            convert_func=convert_wind_at_hub, matrix=m, index=index, turbine=_turbine(WIND_OFFSHORE, smooth),
            density_correction=AIR_DENSITY_CORRECTION)),
        "solar": ("land", lambda m: cutout.pv(panel=PV_PANEL, orientation=PV_ORIENTATION, matrix=m, index=index)),
        "solar_tracking": ("land", lambda m: cutout.pv(panel=PV_PANEL, orientation=PV_ORIENTATION, tracking=PV_TRACKING,
                                                       matrix=m, index=index)),
    }
    profiles = {}
    started = time.perf_counter()
    for name in technologies:
        kind, run = runs[name]
        matrix, has_area = matrices[kind]
        values = np.asarray(run(matrix).transpose("time", "cell").values, dtype=np.float64)
        if name.startswith("solar"):
            values = pv_per_mw_ac(values)
        values[:, ~has_area] = np.nan
        profiles[name] = values
        LOGGER.info("%s: %d cells with %s area (%.0f s)", name, int(has_area.sum()), kind, time.perf_counter() - started)
    times = pd.DatetimeIndex(cutout.data["time"].values)
    lats = np.unique(cells["latitude"].to_numpy())
    lons = np.unique(cells["longitude"].to_numpy())
    position = pd.MultiIndex.from_frame(cells[["latitude", "longitude"]])
    full = pd.MultiIndex.from_product([lats, lons], names=["latitude", "longitude"])
    order = position.get_indexer(full)
    data = {}
    for name, values in profiles.items():
        grid = np.full((len(times), len(full)), np.nan)
        present = order >= 0
        grid[:, present] = values[:, order[present]]
        data[name] = (("time", "latitude", "longitude"), grid.reshape(len(times), len(lats), len(lons)).astype(np.float32))
    dataset = xr.Dataset(data, coords={"time": times, "latitude": lats.astype(np.int32), "longitude": lons.astype(np.int32)})
    dataset.attrs.update(settings_record())
    return dataset


def settings_record() -> dict:
    """The settings behind the profiles, for the files' attributes and the report."""
    record = {"parameters": json.dumps(file_record()), "pv_panel": PV_PANEL, "pv_orientation": PV_ORIENTATION,
              "pv_tracking": PV_TRACKING, "dc_ac_ratio": DC_AC_RATIO, "air_density_correction": int(AIR_DENSITY_CORRECTION),
              "smooth_power_curve": int(SMOOTH_POWER_CURVE)}
    for label, settings in (("wind_onshore", WIND_ONSHORE), ("wind_offshore", WIND_OFFSHORE)):
        record[f"{label}_reference_turbine"] = settings["reference_turbine"]
        record[f"{label}_hub_height_m"] = float(settings["hub_height_m"])
        record[f"{label}_speed_scale"] = round(speed_scale(settings), 6)
    try:
        from importlib.metadata import version

        record["atlite_version"] = version("atlite")
    except Exception:  # noqa: BLE001
        pass
    return record


def _open_cutout(path: Path, south: float, north: float, west: float | None = None, east: float | None = None):
    import atlite

    cutout = atlite.Cutout(path=path)
    selection = {"y": slice(south - 0.75, north + 0.75)}
    if west is not None and east is not None:
        selection["x"] = slice(west - 0.75, east + 0.75)
    return cutout.sel(**selection)


def convert_band(cutout_path: Path, south: int, north: int, output: Path) -> None:
    """Profiles of cell latitudes south to north (inclusive), every longitude, to one NetCDF."""
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    cells = band_cells(south, north)
    cutout = _open_cutout(cutout_path, south, north)
    dataset = convert_cells(cutout, cells, pd.read_csv(WEIGHTS_CSV))
    dataset.attrs["cutout"] = str(cutout_path)
    dataset.attrs["cell_latitudes"] = f"{south}:{north}"
    output.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_netcdf(output)
    LOGGER.info("wrote %s", output)


def check(cutout_path: Path, output: Path) -> dict:
    """Full-load hours at DEA's reference cells against DEA's (wind after the wake factor)."""
    cells = sorted(set(DEA_LAND_CELLS) | set(DEA_SEA_CELLS))
    frame = band_cells(min(c[0] for c in cells), max(c[0] for c in cells))
    frame = frame[frame["longitude"].between(min(c[1] for c in cells), max(c[1] for c in cells))].reset_index(drop=True)
    cutout = _open_cutout(cutout_path, frame["latitude"].min(), frame["latitude"].max(),
                          frame["longitude"].min(), frame["longitude"].max())
    weights = pd.read_csv(WEIGHTS_CSV)
    profiles = convert_cells(cutout, frame, weights)
    # the wind once more with the other smoothing setting, as the calibration choice it informs
    other = convert_cells(cutout, frame, weights, technologies=("wind", "wind_offshore"), smooth=not SMOOTH_POWER_CURVE)
    report = {"cutout": str(cutout_path), "settings": settings_record(), "wake_factor": WIND_WAKE_FACTOR, "technologies": {}}

    def hours_at(dataset, name):
        chosen = DEA_SEA_CELLS if name == "wind_offshore" else DEA_LAND_CELLS
        factor = WIND_WAKE_FACTOR if name.startswith("wind") else 1.0
        hours = {f"{lat},{lon}": round(float(dataset[name].sel(latitude=lat, longitude=lon).sum()) * factor, 1)
                 for lat, lon in chosen}
        return hours, float(np.nanmean(list(hours.values())))

    for name, target in DEA_FULL_LOAD_HOURS.items():
        hours, mean = hours_at(profiles, name)
        entry = {"dea_full_load_hours": target, "cells": hours, "mean": round(mean, 1), "ratio_to_dea": round(mean / target, 4)}
        if name.startswith("wind"):
            _, other_mean = hours_at(other, name)
            entry[f"mean_with_smoothing_{str(not SMOOTH_POWER_CURVE).lower()}"] = round(other_mean, 1)
            entry[f"ratio_with_smoothing_{str(not SMOOTH_POWER_CURVE).lower()}"] = round(other_mean / target, 4)
        report["technologies"][name] = entry
    speed = cutout.data["wnd100m"]
    report["sea_cells_mean_wind_100m_m_per_s"] = {
        f"{lat},{lon}": round(float(speed.sel(y=slice(lat - 0.5, lat + 0.5), x=slice(lon - 0.5, lon + 0.5)).mean()), 2)
        for lat, lon in DEA_SEA_CELLS}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    return report


# ---------------------------------------------------------------------------------------------
# Stack (any machine)
# ---------------------------------------------------------------------------------------------

def write_stack(bands: list[Path], output_dir: Path, sea: bool = False) -> dict:
    """Stitch band files into the weather stack: per technology three longitude bands, every
    latitude of -90 to 89 (NaN outside the bands, and at sea where a cell has no sea).

    A land run's stack holds the wind on land; a sea run's (sea=True) the wind at sea and no PV
    output, so the same plant is solved there without PV (model/sea_cells.py)."""
    output_dir.mkdir(parents=True, exist_ok=False)
    combined = xr.concat([xr.open_dataset(path, chunks={"time": 744}) for path in sorted(bands)], dim="latitude").sortby("latitude")
    if combined["latitude"].to_index().duplicated().any():
        raise ValueError("band files overlap in latitude")
    combined = combined.reindex(latitude=CELL_LATITUDES.astype(np.int32), longitude=CELL_LONGITUDES.astype(np.int32))
    if sea:
        combined["wind"] = combined["wind_offshore"]
        combined["solar"] = combined["solar_tracking"] = xr.zeros_like(combined["wind_offshore"])
    record = {"bands": {path.name: _sha256(path) for path in sorted(bands)}, "attributes": dict(combined.attrs), "files": {}, "sea": sea}
    for name, (prefix, variable) in STACK_FILES.items():
        for suffix, west, east in LONGITUDE_BANDS:
            part = combined[name].sel(longitude=slice(west, east)).rename(variable).to_dataset()
            part.attrs.update(combined.attrs)
            path = output_dir / f"{prefix}{suffix}.nc"
            part.to_netcdf(path, encoding={variable: {"dtype": "float32", "zlib": True, "complevel": 4}})
            record["files"][path.name] = _sha256(path)
    (output_dir / "provenance.json").write_text(json.dumps(record, indent=2) + "\n")
    return record


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    check_parser = sub.add_parser("check", help="full-load hours at DEA's reference cells")
    check_parser.add_argument("--cutout", type=Path, required=True)
    check_parser.add_argument("--output", type=Path, required=True)
    convert_parser = sub.add_parser("convert", help="profiles of one band of cell latitudes")
    convert_parser.add_argument("--cutout", type=Path, required=True)
    convert_parser.add_argument("--cells", required=True, help="south:north cell latitudes, inclusive")
    convert_parser.add_argument("--output", type=Path, required=True)
    stack_parser = sub.add_parser("stack", help="stitch band files into a weather directory")
    stack_parser.add_argument("--bands", type=Path, required=True)
    stack_parser.add_argument("--output", type=Path, required=True)
    stack_parser.add_argument("--sea", action="store_true", help="a sea run's stack: the wind at sea, no PV output")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if args.command == "check":
        report = check(args.cutout, args.output)
        for name, entry in report["technologies"].items():
            other = {key: value for key, value in entry.items() if key.startswith("mean_with")}
            print(f"{name}: {entry['mean']:.0f} h against DEA {entry['dea_full_load_hours']:.0f} h ({entry['ratio_to_dea']:.3f})"
                  + "".join(f"; {key.replace('mean_', '').replace('_', ' ')}: {value:.0f} h" for key, value in other.items()))
    elif args.command == "convert":
        south, north = (int(value) for value in args.cells.split(":"))
        convert_band(args.cutout, south, north, args.output)
    else:
        record = write_stack(sorted(args.bands.glob("band_*.nc")), args.output, sea=args.sea)
        print(f"wrote {len(record['files'])} files to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
