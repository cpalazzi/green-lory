"""model/weather_profiles.py: the one-degree profiles from the ERA5 cutout (no atlite needed)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from model import weather, weather_profiles as wp
from model.config import EARTH_RADIUS_KM

POINT_LATS = np.arange(-90.0, 90.01, 0.25)
POINT_LONS = np.arange(-180.0, 180.0, 0.25)


def test_dea_turbines_have_dea_specific_power():
    assert wp.specific_power_w_per_m2(6.0, 165.0) == pytest.approx(280.6, abs=0.1)
    assert wp.specific_power_w_per_m2(35.0, 380.0) == pytest.approx(308.6, abs=0.1)
    assert wp.speed_scale(wp.WIND_ONSHORE) == pytest.approx((280.6 / 222.8) ** (1 / 3), abs=1e-3)  # 1.080
    assert wp.speed_scale(wp.WIND_OFFSHORE) == pytest.approx((308.6 / 331.6) ** (1 / 3), abs=1e-3)  # 0.976


def test_stretched_curve_gives_the_reference_output_at_the_stretched_speed():
    reference = {"V": np.array([0.0, 3.0, 10.0, 12.0, 25.0]), "POW": np.array([0.0, 0.1, 5.0, 7.0, 7.0]),
                 "hub_height": 100.0, "P": 7.0}
    turbine = wp.stretched_turbine(reference, wp.WIND_ONSHORE)
    scale = wp.speed_scale(wp.WIND_ONSHORE)
    assert turbine["hub_height"] == 110.0
    for speed in (3.0, 7.0, 11.0):
        assert np.interp(speed * scale, turbine["V"], turbine["POW"]) == pytest.approx(np.interp(speed, reference["V"], reference["POW"]))


def test_air_density_follows_the_standard_atmosphere():
    assert wp.air_density_ratio(0.0) == pytest.approx(1.0)
    assert wp.air_density_ratio(-20.0) == pytest.approx(1.0)  # below sea level counts as sea level
    assert wp.air_density_ratio(3000.0) == pytest.approx(0.742, abs=1e-3)
    assert wp.air_density_ratio(4000.0) == pytest.approx(0.669, abs=1e-3)


def test_pv_per_mw_of_ac_scales_by_dc_ac_and_caps_at_one():
    np.testing.assert_allclose(wp.pv_per_mw_ac(np.array([0.0, 0.4, 0.8, 0.9])), [0.0, 0.5, 1.0, 1.0])


def test_convert_takes_a_band_south_of_the_equator(monkeypatch):
    # "-75:-61" after a space reads as an option, so the job script passes --cells=-75:-61
    seen = {}
    monkeypatch.setattr(wp, "convert_band", lambda cutout, south, north, output: seen.update(south=south, north=north))
    wp.main(["convert", "--cutout", "cutout.nc", "--cells=-75:-61", "--output", "band.nc"])
    assert seen == {"south": -75, "north": -61}
    script = (Path(__file__).resolve().parents[1] / "arc" / "jobs" / "11_weather_profiles.sh").read_text()
    assert '--cells="${SOUTH}:${NORTH}"' in script


def test_a_cell_of_no_land_still_gets_a_land_profile():
    geography = pd.DataFrame({"latitude": [-16.0], "longitude": [-144.0], "cell_area_km2": [11800.0], "land_km2": [0.0]})
    cells = wp.band_cells(-16, -16, geography)
    lats, lons = np.array([-16.25, -16.0, -15.75]), np.arange(-180.0, 180.0, 0.25)
    weights = wp.cell_point_weights(lats, lons, cells)
    land, has_land = wp.aggregation_matrix(weights, "land", len(cells), len(lats), len(lons))
    sea, has_sea = wp.aggregation_matrix(weights, "sea", len(cells), len(lats), len(lons))
    assert has_land.all() and has_sea.all()
    np.testing.assert_allclose(land.toarray(), sea.toarray())  # all its points, for both


def _cell(lat, lon, share):
    return pd.DataFrame({"latitude": [float(lat)], "longitude": [float(lon)], "land_share": [share]})


def test_point_boxes_add_up_to_the_cell_area():
    weights = wp.cell_point_weights(POINT_LATS, POINT_LONS, _cell(-23, 151, 1.0))
    area = EARTH_RADIUS_KM ** 2 * np.deg2rad(1.0) * (np.sin(np.deg2rad(-22.5)) - np.sin(np.deg2rad(-23.5)))
    assert len(weights) == 25
    assert weights["total_km2"].sum() == pytest.approx(area, rel=1e-9)
    corner = weights[(weights.iy == weights.iy.min()) & (weights.ix == weights.ix.min())]["total_km2"].iloc[0]
    centre = weights["total_km2"].max()
    assert corner == pytest.approx(centre / 4, rel=0.01)  # a corner point's box is a quarter inside


def test_cells_on_the_date_line_take_points_from_both_sides():
    weights = wp.cell_point_weights(POINT_LATS, POINT_LONS, _cell(10, -180, 0.0))
    lons = POINT_LONS[weights["ix"].unique()]
    assert set(np.round(lons, 2)) == {-180.0, -179.75, -179.5, 179.5, 179.75}


def test_land_and_sea_weights_follow_the_point_table():
    land = np.zeros((5, 5))
    land[:, :2] = 1.0  # the two western point columns are land (km2, below the box areas)
    row = {"latitude": -23.0, "longitude": 151.0, "land_share": 0.3}
    row.update({f"land_{i}{j}": land[i, j] for i in range(5) for j in range(5)})
    row.update({f"total_{i}{j}": 1.0 for i in range(5) for j in range(5)})
    weights = wp.cell_point_weights(POINT_LATS, POINT_LONS, _cell(-23, 151, 0.3), pd.DataFrame([row]))
    lons = POINT_LONS[weights["ix"]]
    assert set(np.round(lons[weights["land_km2"] > 0], 2)) == {150.5, 150.75}
    matrix, has_area = wp.aggregation_matrix(weights, "land", 1, len(POINT_LATS), len(POINT_LONS))
    assert has_area.all() and matrix.sum() == pytest.approx(1.0)
    sea, _ = wp.aggregation_matrix(weights, "sea", 1, len(POINT_LATS), len(POINT_LONS))
    assert sea.sum() == pytest.approx(1.0)


def test_cells_without_the_table_are_all_land_or_all_sea():
    cells = pd.concat([_cell(50, 10, 1.0), _cell(50, -30, 0.0)], ignore_index=True)
    weights = wp.cell_point_weights(POINT_LATS, POINT_LONS, cells)
    _, land_cells = wp.aggregation_matrix(weights, "land", 2, len(POINT_LATS), len(POINT_LONS))
    _, sea_cells = wp.aggregation_matrix(weights, "sea", 2, len(POINT_LATS), len(POINT_LONS))
    assert land_cells.tolist() == [True, False] and sea_cells.tolist() == [False, True]


def test_stack_is_read_by_the_weather_loader(tmp_path):
    times = pd.date_range("2019-01-01", periods=48, freq="h")
    lats = np.array([-23, -22], dtype=np.int32)
    lons = wp.CELL_LONGITUDES.astype(np.int32)
    shape = (len(times), len(lats), len(lons))
    band = xr.Dataset({name: (("time", "latitude", "longitude"), np.full(shape, value, dtype=np.float32))
                       for name, value in (("solar", 0.2), ("solar_tracking", 0.3), ("wind", 0.4), ("wind_offshore", 0.5))},
                      coords={"time": times, "latitude": lats, "longitude": lons})
    (tmp_path / "bands").mkdir()
    band.to_netcdf(tmp_path / "bands" / "band_-23_-22.nc")
    record = wp.write_stack(sorted((tmp_path / "bands").glob("band_*.nc")), tmp_path / "stack")
    assert sorted(record["files"]) == sorted(f"{prefix}{suffix}.nc" for prefix in ("Solar", "SolarTracking", "WindPowers")
                                            for suffix in ("", "1", "2"))
    stack = weather.ProfileStack(tmp_path / "stack")
    frame = weather.cell_profiles(stack, -23, 151)
    assert frame["solar"].iloc[0] == pytest.approx(0.2)
    assert frame["solar_tracking"].iloc[0] == pytest.approx(0.3)
    assert frame["wind"].iloc[0] == pytest.approx(0.4 * weather.WIND_WAKE_FACTOR)
    assert np.isnan(stack.resources["solar"].sel(latitude=10, longitude=0).values).all()  # outside the band
    stack.close()
    wp.write_stack(sorted((tmp_path / "bands").glob("band_*.nc")), tmp_path / "sea", sea=True)
    with weather.ProfileStack(tmp_path / "sea") as sea:
        frame = weather.cell_profiles(sea, -23, 151)
    assert frame["wind"].iloc[0] == pytest.approx(0.5 * weather.WIND_WAKE_FACTOR)  # the wind at sea
    assert frame["solar"].iloc[0] == 0.0 and frame["solar_tracking"].iloc[0] == 0.0
