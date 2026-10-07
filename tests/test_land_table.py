"""model/land_table.py: cell geometry, MODIS aggregation, the wind/PV union and land shares.

Cells are centred on integer coordinates: the cell labelled (lat, lon) covers
[lat - 0.5, lat + 0.5) x [lon - 0.5, lon + 0.5), like the weather nodes. The September 2026
build labelled each MODIS band by its north edge (a one-degree mismatch); these tests pin the
geometry, including the cell that straddles the dateline. MODIS is replaced by a synthetic
0.5-degree class cube read through a fake pyhdf.
"""
from __future__ import annotations

import sys
import types
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd
import pytest
from shapely.geometry import MultiPolygon

from model.land import (
    CLASSWISE_NESTED_UNION_AREA_COLUMN,
    CLASSWISE_NESTED_UNION_AVAILABILITY_COLUMN,
    CLASSWISE_NESTED_UNION_METHOD,
    CLASSWISE_NESTED_UNION_VERSION,
    RENEWABLE_UNION_METHOD_COLUMN,
    RENEWABLE_UNION_METHOD_VERSION_COLUMN,
)
from model.land_table import (
    EARTH_RADIUS_KM,
    FINAL_COLUMNS,
    LandAvailabilityConfig,
    _aggregate_availability_from_hdf4,
    _apply_land_exclusions,
    _bin_coordinates,
    _cell_area_km2,
    _cell_polygon,
    _wrap_longitude_labels,
    apply_land_competition_scenario,
    build_land_availability_table,
    solar_density,
)

FINE = 0.5
N_Y = int(round(180 / FINE))
N_X = int(round(360 / FINE))
BARREN = 16
WATER = 0


def _cube(boxes, classes=None):
    """Synthetic MODIS percentage cube: water everywhere except the boxes (south, west, north, east),
    which get ``classes`` ({class: percent}, default all barren)."""
    classes = classes or {BARREN: 100.0}
    cube = np.zeros((N_Y, N_X, 17), dtype=float)
    cube[:, :, WATER] = 100.0
    for south, west, north, east in boxes:
        y0, y1 = int(round((90.0 - north) / FINE)), int(round((90.0 - south) / FINE))
        x0, x1 = int(round((west + 180.0) / FINE)), int(round((east + 180.0) / FINE))
        cube[y0:y1, x0:x1, WATER] = 0.0
        for klass, percent in classes.items():
            cube[y0:y1, x0:x1, klass] = percent
    return cube


def _row_share(south, north, cell_south, cell_north):
    """Area share of the latitude band [south, north] within a cell [cell_south, cell_north]."""
    s = np.sin(np.deg2rad([south, north, cell_south, cell_north]))
    return float((s[1] - s[0]) / (s[3] - s[2]))


def _fake_pyhdf(cube):
    class FakeDataset:
        def info(self):
            return ("Land_Cover_Type_1_Percent", 3, cube.shape, 0, 0)

        def __getitem__(self, key):
            return cube[key]

    class FakeFile:
        def __init__(self, *_args, **_kwargs):
            pass

        def select(self, _name):
            return FakeDataset()

        def end(self):
            pass

    sd_module = types.ModuleType("pyhdf.SD")
    sd_module.SD = FakeFile
    sd_module.SDC = types.SimpleNamespace(READ=0)
    package = types.ModuleType("pyhdf")
    package.SD = sd_module
    return {"pyhdf": package, "pyhdf.SD": sd_module}


def _config(**overrides):
    settings = dict(output_csv=Path("unused.csv"), land_cover_path=Path("synthetic.hdf"), fine_degree=FINE,
                    coarse_degree=1.0, lat_bounds=(-3.0, 15.0), include_offshore_wind=False)
    settings.update(overrides)
    return LandAvailabilityConfig(**settings)


def _aggregate(cube, **overrides):
    with mock.patch.dict(sys.modules, _fake_pyhdf(cube)):
        table = _aggregate_availability_from_hdf4(_config(**overrides))
    return table.set_index(["latitude", "longitude"])


# --- geometry ----------------------------------------------------------------------------------

def test_binning_and_longitude_wrap():
    np.testing.assert_array_equal(_bin_coordinates([10.49, 10.5, -0.5, -0.51], 1.0), [10.0, 11.0, 0.0, -1.0])
    np.testing.assert_array_equal(_wrap_longitude_labels([180.0, 179.0, -180.0]), [-180.0, 179.0, -180.0])


def test_cell_polygons_and_dateline_split():
    assert _cell_polygon(0.0, 179.0, 1.0).bounds == (178.5, -0.5, 179.5, 0.5)
    straddling = _cell_polygon(0.0, -180.0, 1.0)
    assert isinstance(straddling, MultiPolygon)
    assert sorted(p.bounds for p in straddling.geoms) == [(-180.0, -0.5, -179.5, 0.5), (179.5, -0.5, 180.0, 0.5)]
    assert straddling.area == pytest.approx(1.0)


def test_cell_area_uses_the_centred_south_edge():
    lat = pd.Series([0.0, 60.0])
    expected = (EARTH_RADIUS_KM**2) * np.deg2rad(1.0) * (np.sin(np.deg2rad(lat + 0.5)) - np.sin(np.deg2rad(lat - 0.5)))
    np.testing.assert_allclose(_cell_area_km2(lat, 1.0), expected)


# --- MODIS aggregation ---------------------------------------------------------------------------

def test_a_corner_aligned_box_splits_into_four_quarters():
    table = _aggregate(_cube([(10.0, 20.0, 11.0, 21.0)]))
    # Half the columns of each cell, and the northern or southern half of its rows weighted by
    # spherical area (exact row weights, not pixel counts)
    lower = 0.5 * _row_share(10.0, 10.5, 9.5, 10.5)
    upper = 0.5 * _row_share(10.5, 11.0, 10.5, 11.5)
    assert lower + upper == pytest.approx(0.5, abs=1e-3)
    assert lower != pytest.approx(0.25, abs=1e-9)
    for cell, share in [((10.0, 20.0), lower), ((10.0, 21.0), lower), ((11.0, 20.0), upper), ((11.0, 21.0), upper)]:
        assert table.loc[cell, "solar_availability"] == pytest.approx(share), cell
        assert table.loc[cell, "onshore_land_pct"] == pytest.approx(100.0 * share), cell
    assert table.loc[(9.0, 20.0), "solar_availability"] == pytest.approx(0.0)
    assert table.loc[(12.0, 21.0), "solar_availability"] == pytest.approx(0.0)


def test_a_centred_box_is_reproduced_by_its_own_label():
    table = _aggregate(_cube([(9.5, 19.5, 10.5, 20.5)]))
    assert table.loc[(10.0, 20.0), "solar_availability"] == pytest.approx(1.0)
    assert table.loc[(11.0, 20.0), "solar_availability"] == pytest.approx(0.0)
    assert table.loc[(10.0, 21.0), "solar_availability"] == pytest.approx(0.0)


def test_the_cell_at_minus_180_straddles_the_dateline():
    table = _aggregate(_cube([(0.0, 179.5, 1.0, 180.0), (0.0, -180.0, 1.0, -179.5)]))
    assert table.loc[(0.0, -180.0), "solar_availability"] == pytest.approx(_row_share(0.0, 0.5, -0.5, 0.5))
    assert table.loc[(1.0, -180.0), "solar_availability"] == pytest.approx(_row_share(0.5, 1.0, 0.5, 1.5))
    assert table.loc[(0.0, 179.0), "solar_availability"] == pytest.approx(0.0)


def test_labels_are_integers_within_bounds():
    table = _aggregate(_cube([])).reset_index()
    assert np.all(np.isclose(table["latitude"], np.round(table["latitude"])))
    assert (table["latitude"].min(), table["latitude"].max()) == (-3.0, 15.0)
    assert (table["longitude"].min(), table["longitude"].max()) == (-180.0, 179.0)
    assert len(table) == 19 * 360


def test_centred_cells_need_an_even_pixel_ratio():
    cube = np.zeros((180, 360, 17), dtype=float)
    cube[:, :, WATER] = 100.0
    with pytest.raises(ValueError):
        _aggregate(cube, fine_degree=1.0)


def test_the_table_records_the_centred_anchor_and_its_geometry():
    config = _config(bathymetry_path=Path("missing_bathymetry.nc"), lat_bounds=(9.0, 12.0), skip_slope_exclusion=True)
    with mock.patch.dict(sys.modules, _fake_pyhdf(_cube([(10.0, 20.0, 11.0, 21.0)]))):
        table = build_land_availability_table(config).set_index(["latitude", "longitude"])
    assert "cell_anchor" in FINAL_COLUMNS and set(table["cell_anchor"]) == {"center"}
    area = float(_cell_area_km2(pd.Series([10.0]), 1.0).iloc[0])
    assert table.loc[(10.0, 20.0), "area"] == pytest.approx(area)
    assert table.loc[(10.0, 20.0), "solar_density_mw_per_km2"] == pytest.approx(float(solar_density(pd.Series([10.0]), 1.0).iloc[0]))
    assert table.loc[(10.0, 20.0), "solar_area_km2"] == pytest.approx(0.5 * _row_share(10.0, 10.5, 9.5, 10.5) * area)


# --- the wind/PV union ---------------------------------------------------------------------------

CELL_00 = (-0.5, -0.5, 0.5, 0.5)


@pytest.mark.parametrize("classes,wind,solar,union", [
    ({12: 100.0}, 0.05, 0.0, 0.05),    # cropland: wind only
    ({13: 100.0}, 0.0, 0.03, 0.03),    # urban: PV only
    ({6: 100.0}, 0.5, 0.5, 0.5),       # shrubland: shared
    ({12: 20.0, 13: 30.0, 6: 50.0}, 0.26, 0.259, 0.269),
])
def test_union_is_the_sum_of_classwise_maxima(classes, wind, solar, union):
    cell = _aggregate(_cube([CELL_00], classes), lat_bounds=(0.0, 0.0)).loc[(0.0, 0.0)]
    assert cell["wind_onshore_availability"] == pytest.approx(wind)
    assert cell["solar_availability"] == pytest.approx(solar)
    assert cell["renewable_union_availability"] == pytest.approx(union)
    assert cell[CLASSWISE_NESTED_UNION_AVAILABILITY_COLUMN] == pytest.approx(union)
    assert cell[RENEWABLE_UNION_METHOD_COLUMN] == CLASSWISE_NESTED_UNION_METHOD
    assert cell[RENEWABLE_UNION_METHOD_VERSION_COLUMN] == CLASSWISE_NESTED_UNION_VERSION
    # A lower bound: at least the larger technology, at most their sum
    assert max(wind, solar) - 1e-12 <= cell["renewable_union_availability"] <= wind + solar + 1e-12


def test_union_columns_are_part_of_the_output_schema():
    for column in (CLASSWISE_NESTED_UNION_AVAILABILITY_COLUMN, CLASSWISE_NESTED_UNION_AREA_COLUMN,
                   "renewable_union_availability", "renewable_union_area_km2",
                   RENEWABLE_UNION_METHOD_COLUMN, RENEWABLE_UNION_METHOD_VERSION_COLUMN):
        assert column in FINAL_COLUMNS


def test_union_survives_land_exclusions():
    frame = pd.DataFrame({
        "latitude": [0.0], "longitude": [0.0], "area": [1_000.0], "onshore_land_pct": [100.0],
        "onshore_area_km2": [1_000.0], "solar_availability": [0.3], "wind_onshore_availability": [0.2],
        "wind_offshore_availability": [0.0], "wind_availability": [0.2],
        CLASSWISE_NESTED_UNION_AVAILABILITY_COLUMN: [0.4], "renewable_union_availability": [0.4], "availability": [0.3],
    })

    def protected_half(input_frame, *_args, **_kwargs):
        result = input_frame.copy()
        result["protected_area_pct"] = 50.0
        return result

    with mock.patch("model.land_table._protected_area_pct_from_vectors", side_effect=protected_half):
        result = _apply_land_exclusions(frame, _config(skip_slope_exclusion=True)).iloc[0]
    assert result["land_exclusion_factor"] == pytest.approx(0.5)
    assert result[CLASSWISE_NESTED_UNION_AVAILABILITY_COLUMN] == pytest.approx(0.2)
    assert result["renewable_union_availability"] == pytest.approx(0.2)
    assert result["availability"] == pytest.approx(0.15)


# --- land shares --------------------------------------------------------------------------------

def _baseline_frame() -> pd.DataFrame:
    return pd.DataFrame({
        "latitude": [0.0], "longitude": [0.0], "area": [1_000.0], "land_exclusion_factor": [0.8],
        "constrained_onshore_area_km2": [800.0], "wind_onshore_availability": [0.2],
        "wind_offshore_availability": [0.0], "wind_availability": [0.2], "solar_availability": [0.3],
        CLASSWISE_NESTED_UNION_AVAILABILITY_COLUMN: [0.4], "renewable_union_availability": [0.4],
        "wind_onshore_area_km2": [200.0], "wind_offshore_area_km2": [0.0], "wind_area_km2": [200.0],
        "solar_area_km2": [300.0], CLASSWISE_NESTED_UNION_AREA_COLUMN: [400.0], "renewable_union_area_km2": [400.0],
        RENEWABLE_UNION_METHOD_COLUMN: [CLASSWISE_NESTED_UNION_METHOD],
        RENEWABLE_UNION_METHOD_VERSION_COLUMN: [CLASSWISE_NESTED_UNION_VERSION],
        "wind_density_mw_per_km2": [5.0], "solar_density_mw_per_km2": [100.0],
    })


def test_two_and_fifty_percent_shares_are_linear_and_not_compounded():
    baseline = _baseline_frame()
    two = apply_land_competition_scenario(baseline, 0.02).iloc[0]
    direct = apply_land_competition_scenario(baseline, 0.50).iloc[0]
    sequential = apply_land_competition_scenario(apply_land_competition_scenario(baseline, 0.02), 0.50).iloc[0]
    assert two["renewable_union_availability"] == pytest.approx(0.008)
    assert two["renewable_union_area_km2"] == pytest.approx(8.0)
    assert two[CLASSWISE_NESTED_UNION_AVAILABILITY_COLUMN] == pytest.approx(0.008)
    assert two[CLASSWISE_NESTED_UNION_AREA_COLUMN] == pytest.approx(8.0)
    assert two["availability"] == pytest.approx(0.006)
    assert direct["renewable_union_availability"] == pytest.approx(0.2)
    assert direct["renewable_union_area_km2"] == pytest.approx(200.0)
    assert direct["renewable_union_availability"] == pytest.approx(sequential["renewable_union_availability"])
    assert direct["renewable_union_area_km2"] == pytest.approx(sequential["renewable_union_area_km2"])
    assert direct["renewable_union_area_km2"] / two["renewable_union_area_km2"] == pytest.approx(25.0)


def test_a_table_without_union_columns_still_rescales():
    legacy = _baseline_frame().drop(columns=[
        CLASSWISE_NESTED_UNION_AVAILABILITY_COLUMN, CLASSWISE_NESTED_UNION_AREA_COLUMN,
        "renewable_union_availability", "renewable_union_area_km2",
        RENEWABLE_UNION_METHOD_COLUMN, RENEWABLE_UNION_METHOD_VERSION_COLUMN,
    ])
    result = apply_land_competition_scenario(legacy, 0.02)
    for column in ("renewable_union_availability", "renewable_union_area_km2",
                   CLASSWISE_NESTED_UNION_AVAILABILITY_COLUMN, CLASSWISE_NESTED_UNION_AREA_COLUMN):
        assert column not in result.columns
    assert result.iloc[0]["solar_area_km2"] == pytest.approx(6.0)


def test_the_versioned_column_creates_the_generic_alias_not_the_reverse():
    versioned = _baseline_frame().drop(columns=["renewable_union_availability", "renewable_union_area_km2"])
    result = apply_land_competition_scenario(versioned, 0.02)
    assert result["renewable_union_availability"].equals(result[CLASSWISE_NESTED_UNION_AVAILABILITY_COLUMN])
    assert result["renewable_union_area_km2"].equals(result[CLASSWISE_NESTED_UNION_AREA_COLUMN])
