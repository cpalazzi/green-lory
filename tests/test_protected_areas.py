"""model/protected_areas.py: the protected land and water of each cell.

MODIS is replaced by the synthetic class cube of tests/test_land_table.py (water everywhere
except the boxes given), read through a fake pyhdf; the polygons are written to a GeoPackage.
"""
from __future__ import annotations

import sys
from pathlib import Path
from unittest import mock

import geopandas as gpd
import pandas as pd
import pytest
from shapely.geometry import box

from model.land_table import _cell_area_km2
from model.protected_areas import _rasterise_polygons, protected_area_table, protected_polygons
from test_land_table import FINE, _cube, _fake_pyhdf


def test_polygons_are_rasterised_by_subpixel_centre():
    # a 0.2 x 0.1 degree box on a 0.05-degree grid covering 0.5 S to 0.5 N from -180
    mask = _rasterise_polygons([box(-179.9, 0.0, -179.7, 0.1)], 0.5, 20, 7200, 0.05)
    assert mask.sum() == 4 * 2
    assert mask[8:10, 2:6].all()


def test_polygons_count_by_status_and_realm():
    frame = gpd.GeoDataFrame({"STATUS": ["Designated", "Designated", "Inscribed", "Proposed"],
                              "REALM": ["Terrestrial", "Coastal", "Marine", "Marine"]},
                             geometry=[box(0, 0, 1, 1)] * 4, crs="EPSG:4326")
    assert protected_polygons(frame)["realm"].tolist() == ["terrestrial", "coastal", "marine"]
    # WDPA releases before the REALM field code the realm in MARINE
    legacy = gpd.GeoDataFrame({"STATUS": "Designated", "MARINE": ["0", "1", "2"]}, geometry=[box(0, 0, 1, 1)] * 3,
                              crs="EPSG:4326")
    assert protected_polygons(legacy)["realm"].tolist() == ["terrestrial", "coastal", "marine"]


def test_land_and_water_count_the_polygons_of_their_realms(tmp_path):
    # cell (0, 0): its west half is land, its east half water. A coastal park covers the cell's
    # north half east of 0.25 W: a quarter of the land and half of the water. A terrestrial area
    # and a proposed marine area cover the south half of the water: the first counts on land
    # only, where it covers nothing, and the second does not count
    cube = _cube([(-0.5, -0.5, 0.5, 0.0)])
    path = tmp_path / "wdpa.gpkg"
    south_water = box(0.0, -0.5, 0.5, 0.0)
    gpd.GeoDataFrame({"STATUS": ["Designated", "Designated", "Proposed"], "REALM": ["Coastal", "Terrestrial", "Marine"]},
                     geometry=[box(-0.25, 0.0, 0.5, 0.5), south_water, south_water], crs="EPSG:4326").to_file(path)
    cells = pd.DataFrame({"latitude": [0.0, 1.0], "longitude": [0.0, 0.0]})
    with mock.patch.dict(sys.modules, _fake_pyhdf(cube)):
        table = protected_area_table(cells, (path,), Path("synthetic.hdf"), 1.0, FINE).set_index(["latitude", "longitude"])
    area = _cell_area_km2(pd.Series([0.0, 1.0]), 1.0).to_numpy()
    coast, open_sea = table.loc[(0.0, 0.0)], table.loc[(1.0, 0.0)]
    assert coast["land_km2"] == pytest.approx(area[0] / 2, rel=1e-6)
    assert coast["protected_land_pct"] == pytest.approx(25.0, abs=0.01)
    assert coast["water_km2"] == pytest.approx(area[0] / 2, rel=1e-6)
    assert coast["protected_water_pct"] == pytest.approx(50.0, abs=0.01)
    assert open_sea["water_km2"] == pytest.approx(area[1], rel=1e-6)
    assert (open_sea["land_km2"], open_sea["protected_land_pct"], open_sea["protected_water_pct"]) == (0.0, 0.0, 0.0)
