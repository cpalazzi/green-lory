"""model/site_geography.py: geometry, water sources, site lookup and the remoteness fit (no rasters)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from model import site_geography as geo
from model.config import EARTH_RADIUS_KM


def test_distances_and_areas_on_the_sphere():
    distance, index = geo.nearest_km([0.0, 10.0], [0.0, 0.0], [0.0, 10.0, 50.0], [1.0, 1.0, 1.0])
    np.testing.assert_allclose(distance, [111.19, 109.50], atol=0.01)
    assert list(index) == [0, 1]
    assert geo.great_circle_km([0.0], [0.0], [0.0], [180.0])[0] == pytest.approx(np.pi * EARTH_RADIUS_KM)
    assert geo.band_area_km2(90.0, -90.0, 360.0) == pytest.approx(4 * np.pi * EARTH_RADIUS_KM ** 2)


def test_cell_blocks_centre_cells_on_whole_degrees():
    pixels = 4  # pixels per degree
    band = np.tile(np.arange(360 * pixels), (pixels, 1))  # column numbers
    blocks = geo._cell_blocks(band, pixels)
    # cell 0 is centred on -180: half a degree east of the dateline and half a degree west of it
    assert list(blocks[0, 0]) == [1438, 1439, 0, 1]
    assert list(blocks[0, 180]) == [718, 719, 720, 721]  # the cell centred on 0 degrees


def test_gsw_tiles_cover_the_cell():
    assert geo._gsw_tile_names(-28, 137) == [("130E_20S", 130.0, -20.0)]
    assert [name for name, _, _ in geo._gsw_tile_names(-30, 140)] == ["130E_20S", "130E_30S", "140E_20S", "140E_30S"]
    assert [name for name, _, _ in geo._gsw_tile_names(40, 51)] == ["50E_50N", "50E_40N"]
    assert [name for name, _, _ in geo._gsw_tile_names(41, -112)] == ["120W_50N"]
    assert all(west >= -180 for _, west, _ in geo._gsw_tile_names(0, -180))


def _water_cells():
    area = 10_000.0
    return pd.DataFrame({
        "latitude": [0.0, 0.0, 0.0, 0.0],
        "longitude": [0.0, 1.0, 2.0, 5.0],
        "cell_area_km2": [area] * 4,
        "sea_km2": [5_000.0, 0.0, 0.0, 0.0],
        "fresh_lake_km2": [0.0, 900.0, 0.0, 2_000.0],
        "saline_lake_km2": [0.0, 0.0, 3_000.0, 0.0],
        "reservoir_km2": [0.0, 0.0, 0.0, 0.0],
        "large_river_km2": [0.0, 0.0, 0.0, 0.0],
        "other_permanent_water_km2": [0.0, 0.0, 0.0, 0.0],
    })


def test_water_sources_need_ten_percent_of_the_cell():
    sources = geo.water_sources(_water_cells())
    assert list(sources.water_source) == [True, False, True, True]  # 9 % of lake is not enough
    assert list(sources.water_source_kind) == ["sea", "sea", "lake", "lake"]
    np.testing.assert_allclose(sources.water_source_distance_km, [0.0, 111.19, 0.0, 0.0], atol=0.01)
    without_saline = geo.water_sources(_water_cells(), classes=("sea_km2", "fresh_lake_km2"))
    assert list(without_saline.water_source) == [True, False, False, True]


def test_large_rivers_need_one_percent_of_the_cell():
    cells = _water_cells()
    cells["fresh_lake_km2"] = 0.0  # cells 2 and 4 now have no standing water
    cells["large_river_km2"] = [0.0, 100.0, 0.0, 99.0]  # 1 % of the second cell, 0.99 % of the fourth
    sources = geo.water_sources(cells)
    assert list(sources.water_source) == [True, True, True, False]
    assert list(sources.water_source_kind) == ["sea", "river", "lake", "lake"]  # the fourth's nearest is the saline lake
    without_rivers = geo.water_sources(cells, river_classes=())
    assert list(without_rivers.water_source) == [True, False, True, False]


def _gazetteer():
    return pd.DataFrame({
        "geonameid": [1, 2, 3, 4, 5],
        "name": ["Richmond", "Richmond", "Campbell Town", "Cardowan", "Campbelltown"],
        "asciiname": ["Richmond", "Richmond", "Campbell Town", "Cardowan", "Campbelltown"],
        "admin1": ["04", "02", "06", "04", "02"],
        "feature_class": ["P", "P", "P", "S", "P"],
        "feature_code": ["PPL", "PPL", "PPL", "FRM", "PPL"],
        "population": [758, 5418, 982, 0, 13193],
        "latitude": [-20.7, -33.6, -41.9, -22.1, -34.1],
        "longitude": [143.1, 150.8, 147.5, 149.1, 150.8],
    })


def test_sites_are_found_in_their_state():
    gazetteer = _gazetteer()
    assert geo.locate_site(gazetteer, "QLD", "Richmond")["geonameid"] == 1
    assert geo.locate_site(gazetteer, "TAS", "Campbelltown")["geonameid"] == 3  # the alias
    assert geo.locate_site(gazetteer, "QLD", "Cardowan")["feature_class"] == "S"  # no town of that name
    with pytest.raises(LookupError):
        geo.locate_site(gazetteer, "VIC", "Richmond")


def _sites_for_fit():
    rng = np.random.default_rng(3)
    minutes = np.linspace(0.0, 600.0, 41)
    sites = pd.DataFrame({f"travel_min_city_{label}": rng.uniform(0, 900, len(minutes)) for label in geo.CITY_SIZES})
    sites["travel_min_city_20k"] = minutes
    return sites, 1.0 + 0.12 / 60.0 * np.minimum(minutes, 240.0)


def test_installation_fit_recovers_city_size_slope_and_saturation():
    sites, factor = _sites_for_fit()
    best = geo.fit_installation(sites, factor)["best"]
    assert best["city_size"] == "20k"
    assert best["saturation_min"] == 240.0
    assert best["b_per_hour"] == pytest.approx(0.12)
    assert best["rmse"] == pytest.approx(0.0, abs=1e-12)


def test_equipment_fit_is_a_line_through_one_with_the_table_cap():
    distance = np.array([0.0, 100.0, 300.0, 600.0])
    fit = geo.fit_equipment(distance, 1.0 + 0.0001 * distance)
    assert fit["a_per_1000_km"] == pytest.approx(0.1)
    assert fit["cap"] == pytest.approx(1.06)
    assert fit["rmse"] == pytest.approx(0.0, abs=1e-12)
