"""Hourly PV and wind output per unit of capacity for each 1-degree cell.

The profiles are a stack of NetCDF files in one directory (runs.yaml `weather_dir`): fixed PV
(`Solar*.nc`, variable `Solar`), tracking PV (`SolarTracking*.nc`, variable `Solar`) and wind
(`WindPowers*.nc`, variable `Wind`). Each technology may be split into numbered longitude
bands, which are stitched together on load.

    stack = ProfileStack(directory)          # loads the stack once
    frame = cell_profiles(stack, lat, lon)   # one cell's year, columns the plant expects
    frame = resample(frame, 4)               # 4-hour blocks for a coarser time step

The profiles average the whole 1-degree cell, sea and lakes included. With a land correction
(ProfileStack(directory, land_correction=<file>), model/weather_land.py) the profiles of cells
with water are rescaled to the cell's land, by month and hour of day; runs set it in runs.yaml
(`weather_land_correction`).
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from .config import repo_path
from .parameters import parameter

LOGGER = logging.getLogger(__name__)

# Wind output after wake losses inside the farm (inputs/parameters.yaml)
WIND_WAKE_FACTOR = float(parameter("weather.wind_wake_factor"))

# resource name -> (file-name prefix, NetCDF variable)
RESOURCES = {
    "solar": ("solar", "Solar"),
    "wind": ("windpowers", "Wind"),
    "solar_tracking": ("solartracking", "Solar"),
}


def resource_files(directory: Path, prefix: str) -> list[Path]:
    """The NetCDF files of one technology: `<prefix>.nc` and `<prefix><band number>.nc`."""
    files = []
    for candidate in sorted(directory.glob("*.nc")):
        stem = candidate.stem.lower()
        if not stem.startswith(prefix):
            continue
        suffix = stem[len(prefix):]
        if suffix and not suffix.isdigit():
            continue
        files.append(candidate)
    return files


class ProfileStack:
    """All profiles of one weather directory, loaded into memory once (``cache=True``).

    Pool workers inherit a loaded stack by fork, without copying it.
    """

    def __init__(self, directory: str | Path, cache: bool = True, land_correction: str | Path | None = None):
        self.path = repo_path(directory)
        self._datasets: list[xr.Dataset] = []
        self.resources: dict[str, xr.DataArray] = {}
        self.land_correction = None
        for name, (prefix, variable) in RESOURCES.items():
            files = resource_files(self.path, prefix)
            if not files:
                raise FileNotFoundError(f"No NetCDF files for '{prefix}' in {self.path}")
            dataset = xr.open_mfdataset([str(file) for file in files], combine="by_coords", parallel=False)
            if variable not in dataset:
                dataset.close()
                raise ValueError(f"Variable '{variable}' not present in the '{prefix}' files of {self.path}")
            LOGGER.info("Loaded %s stack (%s files)", prefix, len(files))
            data_array = dataset[variable].load() if cache else dataset[variable]
            self._datasets.append(dataset)
            self.resources[name] = data_array
        if land_correction is not None:
            from .weather_land import LandCorrection

            self.land_correction = LandCorrection(repo_path(land_correction))
            times = pd.DatetimeIndex(self.resources["wind"]["time"].values)
            self._month, self._hour = times.month.to_numpy(), times.hour.to_numpy()
            LOGGER.info("Land correction of %d cells with water from %s", len(self.land_correction.index), land_correction)

    def close(self) -> None:
        for dataset in self._datasets:
            try:
                dataset.close()
            except Exception:  # noqa: BLE001
                LOGGER.debug("Failed to close a weather dataset", exc_info=True)
        self._datasets.clear()

    def __enter__(self) -> "ProfileStack":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


def cell_profiles(stack: ProfileStack, lat: float, lon: float) -> pd.DataFrame:
    """One cell's hourly profiles: solar, wind (after wake losses), solar_tracking, grid, ramp_dummy.

    grid (0) and ramp_dummy (1) are the availability columns of the plant's two auxiliary
    generators.
    """
    selection = dict(latitude=lat, longitude=lon)
    factors = None
    if stack.land_correction is not None:
        factors = stack.land_correction.factors(lat, lon, stack._month, stack._hour)
    columns: dict[str, np.ndarray] = {}
    for resource in ("solar", "wind", "solar_tracking"):
        values = stack.resources[resource].sel(selection, method="nearest").values
        if factors is not None:
            # the cell's land: wind by the wind factor (at most rated output); PV by a PV factor
            # only if the file has one (the current one does not)
            if resource == "wind":
                values = np.minimum(values * factors["wind"], 1.0)
            elif "solar" in factors:
                values = values * factors["solar"]
        if resource == "wind":
            values = values * WIND_WAKE_FACTOR
        columns[resource] = values
    frame = pd.DataFrame(columns)
    frame["grid"] = 0.0
    frame["ramp_dummy"] = 1.0
    return frame


def resample(frame: pd.DataFrame, hours: int) -> pd.DataFrame:
    """Block means over `hours` consecutive hours (constant columns pass through unchanged)."""
    if hours <= 1:
        return frame
    n_out = len(frame) // hours
    return pd.DataFrame({
        column: frame[column].values[: n_out * hours].reshape(n_out, hours).mean(axis=1)
        for column in frame.columns
    })
