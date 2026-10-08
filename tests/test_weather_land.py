"""model/weather_land.py: the land correction of the 1-degree profiles (no ERA5 stores needed)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from model import weather_land


class FakeRows:
    """ERA5 points west of 151 E are land with wind 0.3, the rest sea with wind 0.6; PV is 0.2
    everywhere by day (06-18 UTC)."""

    def __init__(self, hours):
        lons = -180.0 + 0.25 * np.arange(1440)
        day = np.asarray((hours.hour >= 6) & (hours.hour < 18))
        self.wind = np.where(lons[:, None] < 151.0, 0.3, 0.6) * np.ones((1, len(hours)))
        self.solar = np.outer(np.ones(1440), np.where(day, 0.2, 0.0))

    def row(self, lat):
        return {"wind": self.wind, "solar": self.solar}


def _weights(land):
    row = {"latitude": -23.0, "longitude": 151.0, "land_share": land.sum() / 25.0}
    row.update({f"land_{i}{j}": land[i, j] for i in range(5) for j in range(5)})
    row.update({f"total_{i}{j}": 1.0 for i in range(5) for j in range(5)})
    return pd.DataFrame([row])


def test_factors_rescale_the_cell_mean_to_its_land():
    hours = pd.date_range("2019-01-01", periods=8760, freq="h")
    land = np.zeros((5, 5))
    land[:, :2] = 1.0  # the two western point columns (150.5 and 150.75 E) are land
    result = weather_land.compute_ratios(_weights(land), FakeRows(hours), hours)
    # whole cell: 2 land columns at 0.3 and 3 sea columns at 0.6 -> 0.48; land 0.3
    np.testing.assert_allclose(result["ratio"][0, 0], 0.3 / 0.48, rtol=1e-6)
    np.testing.assert_allclose(result["means"][0, 0], [0.3, 0.48], rtol=1e-6)
    solar = result["ratio"][0, 1]
    np.testing.assert_allclose(solar[:, 6:18], 1.0, rtol=1e-6)  # the same PV on land and sea
    np.testing.assert_allclose(solar[:, :6], 1.0)  # night: no ratio, factor 1


def test_correction_file_is_read_by_cell(tmp_path):
    factor = np.ones((1, 2, 12, 24))
    factor[0, 0, 0, 3] = 0.5
    path = tmp_path / "correction.nc"
    xr.Dataset({"factor": (("cell", "tech", "month", "hour"), factor),
                "latitude": ("cell", [-23.0]), "longitude": ("cell", [151.0])},
               coords={"tech": ["wind", "solar"]}).to_netcdf(path)
    correction = weather_land.LandCorrection(path)
    hours = pd.date_range("2019-01-01", periods=48, freq="h")
    factors = correction.factors(-23.0, 151.0, hours.month.to_numpy(), hours.hour.to_numpy())
    assert factors["wind"][3] == 0.5 and factors["wind"][27] == 0.5 and factors["wind"][4] == 1.0
    assert correction.factors(-23.0, 117.0, hours.month.to_numpy(), hours.hour.to_numpy()) is None
