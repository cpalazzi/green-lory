"""Cell anchoring of the land build: which 1-degree box a (latitude, longitude) label denotes.

The September 2026 build labelled each MODIS band by its north edge while the
area, protected-area and slope overlays assumed the label was the south-west
corner (a one-degree mismatch).  These tests pin the geometry for both anchors,
including the centred cell that straddles the dateline.
"""
from __future__ import annotations

import sys
import types
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd
from shapely.geometry import MultiPolygon, Polygon

from model.land_processing import (
    CELL_ANCHORS,
    DEFAULT_CELL_ANCHOR,
    EARTH_RADIUS_KM,
    FINAL_COLUMNS,
    LandAvailabilityConfig,
    _aggregate_availability_from_hdf4,
    _bin_coordinates,
    _cell_area_km2,
    _cell_centre_latitudes,
    _cell_polygon,
    _solar_density,
    _wrap_longitude_labels,
    build_land_availability_table,
)

FINE = 0.5
N_Y = int(round(180 / FINE))
N_X = int(round(360 / FINE))
BARREN = 16
WATER = 0


def _cube(barren_boxes):
    """Synthetic MODIS percentage cube: water everywhere except barren boxes (south, west, north, east)."""
    cube = np.zeros((N_Y, N_X, 17), dtype=float)
    cube[:, :, WATER] = 100.0
    for south, west, north, east in barren_boxes:
        y0 = int(round((90.0 - north) / FINE))
        y1 = int(round((90.0 - south) / FINE))
        x0 = int(round((west + 180.0) / FINE))
        x1 = int(round((east + 180.0) / FINE))
        cube[y0:y1, x0:x1, WATER] = 0.0
        cube[y0:y1, x0:x1, BARREN] = 100.0
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


def _aggregate(cube, anchor, lat_bounds=(-3.0, 15.0), fine=FINE):
    config = LandAvailabilityConfig(
        land_cover_path=Path("synthetic.hdf"),
        fine_degree=fine,
        coarse_degree=1.0,
        lat_bounds=lat_bounds,
        include_offshore_wind=False,
        cell_anchor=anchor,
    )
    with mock.patch.dict(sys.modules, _fake_pyhdf(cube)):
        table = _aggregate_availability_from_hdf4(config)
    return table.set_index(["latitude", "longitude"])


class CellLabelGeometryTest(unittest.TestCase):
    def test_centre_is_the_default_and_only_two_anchors_exist(self):
        self.assertEqual(DEFAULT_CELL_ANCHOR, "center")
        self.assertEqual(set(CELL_ANCHORS), {"center", "southwest"})
        with self.assertRaises(ValueError):
            LandAvailabilityConfig(cell_anchor="north").resolved()

    def test_binning_follows_the_anchor_convention(self):
        np.testing.assert_array_equal(_bin_coordinates([10.49, 10.5, -0.5, -0.51], 1.0, "center"), [10.0, 11.0, 0.0, -1.0])
        np.testing.assert_array_equal(_bin_coordinates([10.99, 10.0, -0.01], 1.0, "southwest"), [10.0, 10.0, -1.0])
        np.testing.assert_array_equal(_wrap_longitude_labels([180.0, 179.0, -180.0]), [-180.0, 179.0, -180.0])

    def test_cell_polygons_and_dateline_split(self):
        self.assertEqual(_cell_polygon(0.0, 179.0, 1.0, "southwest").bounds, (179.0, 0.0, 180.0, 1.0))
        self.assertEqual(_cell_polygon(0.0, 179.0, 1.0, "center").bounds, (178.5, -0.5, 179.5, 0.5))
        straddling = _cell_polygon(0.0, -180.0, 1.0, "center")
        self.assertIsInstance(straddling, MultiPolygon)
        self.assertEqual(sorted(p.bounds for p in straddling.geoms), [(-180.0, -0.5, -179.5, 0.5), (179.5, -0.5, 180.0, 0.5)])
        self.assertAlmostEqual(straddling.area, 1.0)
        self.assertIsInstance(_cell_polygon(0.0, -180.0, 1.0, "southwest"), Polygon)

    def test_cell_area_uses_the_anchored_south_edge(self):
        lat = pd.Series([0.0, 60.0])
        expected_centre = (EARTH_RADIUS_KM**2) * np.deg2rad(1.0) * (np.sin(np.deg2rad(lat + 0.5)) - np.sin(np.deg2rad(lat - 0.5)))
        expected_southwest = (EARTH_RADIUS_KM**2) * np.deg2rad(1.0) * (np.sin(np.deg2rad(lat + 1.0)) - np.sin(np.deg2rad(lat)))
        np.testing.assert_allclose(_cell_area_km2(lat, 1.0, "center"), expected_centre)
        np.testing.assert_allclose(_cell_area_km2(lat, 1.0, "southwest"), expected_southwest)
        np.testing.assert_allclose(_cell_centre_latitudes(lat, 1.0, "center"), [0.0, 60.0])
        np.testing.assert_allclose(_cell_centre_latitudes(lat, 1.0, "southwest"), [0.5, 60.5])


class Hdf4AggregationAnchorTest(unittest.TestCase):
    def test_southwest_label_denotes_the_box_north_east_of_it(self):
        table = _aggregate(_cube([(10.0, 20.0, 11.0, 21.0)]), "southwest")
        self.assertAlmostEqual(table.loc[(10.0, 20.0), "solar_availability"], 1.0)
        self.assertAlmostEqual(table.loc[(10.0, 20.0), "onshore_land_pct"], 100.0)
        # The September build labelled this band 11 (north edge): must be zero now.
        self.assertAlmostEqual(table.loc[(11.0, 20.0), "solar_availability"], 0.0)
        self.assertAlmostEqual(table.loc[(9.0, 20.0), "solar_availability"], 0.0)
        self.assertAlmostEqual(table.loc[(10.0, 19.0), "solar_availability"], 0.0)

    def test_centre_label_splits_a_corner_aligned_box_into_four_quarters(self):
        table = _aggregate(_cube([(10.0, 20.0, 11.0, 21.0)]), "center")
        # Half the columns of each cell, and the northern or southern half of its rows
        # weighted by spherical area (exact row weights, not pixel counts).
        lower = 0.5 * _row_share(10.0, 10.5, 9.5, 10.5)
        upper = 0.5 * _row_share(10.5, 11.0, 10.5, 11.5)
        self.assertAlmostEqual(lower + upper, 0.5, places=3)
        self.assertNotAlmostEqual(lower, 0.25, places=9)
        for cell, share in [((10.0, 20.0), lower), ((10.0, 21.0), lower), ((11.0, 20.0), upper), ((11.0, 21.0), upper)]:
            self.assertAlmostEqual(table.loc[cell, "solar_availability"], share, msg=str(cell))
            self.assertAlmostEqual(table.loc[cell, "onshore_land_pct"], 100.0 * share, msg=str(cell))
        self.assertAlmostEqual(table.loc[(9.0, 20.0), "solar_availability"], 0.0)
        self.assertAlmostEqual(table.loc[(12.0, 21.0), "solar_availability"], 0.0)

    def test_centred_box_is_reproduced_exactly_by_its_own_label(self):
        table = _aggregate(_cube([(9.5, 19.5, 10.5, 20.5)]), "center")
        self.assertAlmostEqual(table.loc[(10.0, 20.0), "solar_availability"], 1.0)
        self.assertAlmostEqual(table.loc[(11.0, 20.0), "solar_availability"], 0.0)
        self.assertAlmostEqual(table.loc[(10.0, 21.0), "solar_availability"], 0.0)

    def test_centred_cell_at_minus_180_straddles_the_dateline(self):
        cube = _cube([(0.0, 179.5, 1.0, 180.0), (0.0, -180.0, 1.0, -179.5)])
        centred = _aggregate(cube, "center")
        self.assertAlmostEqual(centred.loc[(0.0, -180.0), "solar_availability"], _row_share(0.0, 0.5, -0.5, 0.5))
        self.assertAlmostEqual(centred.loc[(1.0, -180.0), "solar_availability"], _row_share(0.5, 1.0, 0.5, 1.5))
        self.assertAlmostEqual(centred.loc[(0.0, 179.0), "solar_availability"], 0.0)
        southwest = _aggregate(cube, "southwest")
        self.assertAlmostEqual(southwest.loc[(0.0, 179.0), "solar_availability"], 0.5)
        self.assertAlmostEqual(southwest.loc[(0.0, -180.0), "solar_availability"], 0.5)

    def test_labels_are_integers_within_bounds_for_both_anchors(self):
        cube = _cube([])
        for anchor in CELL_ANCHORS:
            table = _aggregate(cube, anchor, lat_bounds=(-3.0, 15.0)).reset_index()
            self.assertTrue(np.all(np.isclose(table["latitude"], np.round(table["latitude"]))), anchor)
            self.assertEqual(table["latitude"].min(), -3.0, anchor)
            self.assertEqual(table["latitude"].max(), 15.0, anchor)
            self.assertEqual(table["longitude"].min(), -180.0, anchor)
            self.assertEqual(table["longitude"].max(), 179.0, anchor)
            self.assertEqual(len(table), 19 * 360, anchor)

    def test_centred_cells_need_an_even_pixel_ratio(self):
        cube = np.zeros((180, 360, 17), dtype=float)
        cube[:, :, WATER] = 100.0
        with self.assertRaises(ValueError):
            _aggregate(cube, "center", fine=1.0)


class TableAnchorColumnTest(unittest.TestCase):
    def _table(self, anchor):
        config = LandAvailabilityConfig(
            land_cover_path=Path("synthetic.hdf"),
            bathymetry_path=Path("missing_bathymetry.nc"),
            fine_degree=FINE,
            coarse_degree=1.0,
            lat_bounds=(9.0, 12.0),
            include_offshore_wind=False,
            skip_slope_exclusion=True,
            cell_anchor=anchor,
        )
        with mock.patch.dict(sys.modules, _fake_pyhdf(_cube([(10.0, 20.0, 11.0, 21.0)]))):
            return build_land_availability_table(config).set_index(["latitude", "longitude"])

    def test_anchor_is_recorded_and_geometry_follows_it(self):
        self.assertIn("cell_anchor", FINAL_COLUMNS)
        for anchor in CELL_ANCHORS:
            table = self._table(anchor)
            self.assertEqual(set(table["cell_anchor"]), {anchor})
            lat = pd.Series([10.0])
            self.assertAlmostEqual(table.loc[(10.0, 20.0), "area"], float(_cell_area_km2(lat, 1.0, anchor).iloc[0]))
            centre = 10.0 if anchor == "center" else 10.5
            self.assertAlmostEqual(
                table.loc[(10.0, 20.0), "solar_density_mw_per_km2"],
                float(_solar_density(pd.Series([centre]), 1.0).iloc[0]),
            )
        self.assertAlmostEqual(self._table("southwest").loc[(10.0, 20.0), "solar_area_km2"], float(_cell_area_km2(pd.Series([10.0]), 1.0, "southwest").iloc[0]))
        self.assertAlmostEqual(
            self._table("center").loc[(10.0, 20.0), "solar_area_km2"],
            0.5 * _row_share(10.0, 10.5, 9.5, 10.5) * float(_cell_area_km2(pd.Series([10.0]), 1.0, "center").iloc[0]),
        )


if __name__ == "__main__":
    unittest.main()
