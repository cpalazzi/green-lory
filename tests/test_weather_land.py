"""model/weather_land.py: the land correction of the 1-degree profiles (no ERA5 stores needed)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import xarray as xr

from model import weather_land

HOURS = pd.date_range("2019-01-01", periods=8760, freq="h")
BINS = (HOURS.month.to_numpy() - 1) * 24 + HOURS.hour.to_numpy()


class FakeRows:
    """ERA5 points west of 151 E are land with wind 0.3, the rest sea with wind 0.6."""

    def __init__(self, hours):
        lons = -180.0 + 0.25 * np.arange(1440)
        self.wind = np.where(lons[:, None] < 151.0, 0.3, 0.6) * np.ones((1, len(hours)))

    def row(self, lat):
        return {"wind": self.wind}


def _weights(land):
    row = {"latitude": -23.0, "longitude": 151.0, "land_share": land.sum() / 25.0}
    row.update({f"land_{i}{j}": land[i, j] for i in range(5) for j in range(5)})
    row.update({f"total_{i}{j}": 1.0 for i in range(5) for j in range(5)})
    return pd.DataFrame([row])


def test_factors_rescale_the_cell_mean_to_its_land():
    land = np.zeros((5, 5))
    land[:, :2] = 1.0  # the two western point columns (150.5 and 150.75 E) are land
    sums = weather_land.compute_bin_sums(_weights(land), FakeRows(HOURS), HOURS)
    factors = weather_land.wind_factors(sums["land_bins"], sums["cell_bins"], sums["hours_per_bin"])
    # whole cell: 2 land columns at 0.3 and 3 sea columns at 0.6 -> 0.48; land 0.3
    np.testing.assert_allclose(factors, 0.3 / 0.48, rtol=1e-6)


def test_calm_bins_take_the_annual_ratio():
    land_bins = np.array([[3.0, 0.01, 6.0]])
    cell_bins = np.array([[6.0, 0.001, 6.0]])  # the middle bin is calm (mean 0.001 over one hour)
    factors = weather_land.wind_factors(land_bins, cell_bins, np.array([30, 1, 30]))
    np.testing.assert_allclose(factors, [[0.5, (3.0 + 0.01 + 6.0) / (6.0 + 0.001 + 6.0), 1.0]])


def test_floor_keeps_a_coast_at_least_as_windy_as_its_hinterland():
    legacy = np.full((3, 8760), 0.4)
    factors = np.full((3, 288), 0.1)  # ERA5 says the land has a tenth of the cell's wind
    floor = np.array([0.3, np.nan, 0.6])
    result = weather_land.apply_inland_floor(factors, legacy, BINS, floor)
    np.testing.assert_allclose(result["corrected_mean"], [0.3, 0.04, 0.4], rtol=1e-6)
    assert list(result["floor_applied"]) == [True, False, True]


def test_inland_floor_averages_the_fully_inland_neighbours():
    cells = pd.DataFrame({"latitude": [0.0], "longitude": [-180.0]})
    geography = pd.DataFrame({
        "latitude": [1.0, 0.0, -1.0], "longitude": [-180.0, 179.0, -179.0],
        "cell_area_km2": [100.0] * 3, "land_km2": [100.0, 100.0, 50.0],
        **{column: [0.0, 0.0, 0.0] for column in weather_land.OPEN_WATER_COLUMNS},
    })
    means = {(1.0, -180.0): 0.2, (0.0, 179.0): 0.4, (-1.0, -179.0): 0.9}
    # (-1, -179) is half sea, so only the other two count; (0, 179) is across the dateline
    np.testing.assert_allclose(weather_land.inland_floor(cells, geography, means), [0.3])


def _correction_file(path, wind_factor):
    factor = np.ones((1, 1, 12, 24))
    factor[0, 0] = wind_factor
    xr.Dataset({"factor": (("cell", "tech", "month", "hour"), factor),
                "latitude": ("cell", [-23.0]), "longitude": ("cell", [151.0])},
               coords={"tech": ["wind"]}).to_netcdf(path)


def test_correction_file_is_read_by_cell(tmp_path):
    path = tmp_path / "correction.nc"
    _correction_file(path, 1.0)
    factor = xr.open_dataset(path)["factor"].values
    factor[0, 0, 0, 3] = 0.5
    xr.Dataset({"factor": (("cell", "tech", "month", "hour"), factor),
                "latitude": ("cell", [-23.0]), "longitude": ("cell", [151.0])},
               coords={"tech": ["wind"]}).to_netcdf(tmp_path / "edited.nc")
    correction = weather_land.LandCorrection(tmp_path / "edited.nc")
    hours = pd.date_range("2019-01-01", periods=48, freq="h")
    factors = correction.factors(-23.0, 151.0, hours.month.to_numpy(), hours.hour.to_numpy())
    assert factors["wind"][3] == 0.5 and factors["wind"][27] == 0.5 and factors["wind"][4] == 1.0
    assert correction.factors(-23.0, 117.0, hours.month.to_numpy(), hours.hour.to_numpy()) is None


def test_profiles_of_cells_with_water_are_rescaled(tmp_path):
    from model import weather

    times = pd.date_range("2019-01-01", periods=48, freq="h")
    coords = {"time": times, "latitude": [-23, -22], "longitude": [150, 151]}
    shape = (48, 2, 2)
    for name, variable, value in (("Solar", "Solar", 0.5), ("SolarTracking", "Solar", 0.6), ("WindPowers", "Wind", 0.8)):
        xr.Dataset({variable: (("time", "latitude", "longitude"), np.full(shape, value))}, coords=coords).to_netcdf(tmp_path / f"{name}.nc")
    correction = tmp_path / "correction" / "factors.nc"
    correction.parent.mkdir()
    _correction_file(correction, 1.5)  # wind on the land stronger than the cell mean: capped at 1

    plain = weather.ProfileStack(tmp_path)
    corrected = weather.ProfileStack(tmp_path, land_correction=correction)
    coastal = weather.cell_profiles(corrected, -23, 151)
    np.testing.assert_allclose(coastal["wind"], 1.0 * weather.WIND_WAKE_FACTOR)
    np.testing.assert_allclose(coastal["solar"], 0.5)  # PV is not corrected
    inland = weather.cell_profiles(corrected, -23, 150)
    pd.testing.assert_frame_equal(inland, weather.cell_profiles(plain, -23, 150))
