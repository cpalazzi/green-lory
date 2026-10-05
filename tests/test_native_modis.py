from pathlib import Path

import numpy as np
import pytest
import shapely

from reconciliation.land.core import Cell, RADIUS_KM
from reconciliation.land.native_modis import NativeTile, cell_polygon, class_histogram, metadata_value, sinusoidal_xy


def test_metadata_exact_key_does_not_match_localversion():
    text = 'OBJECT = LOCALVERSIONID\nVALUE = "6.0.1"\nEND_OBJECT = LOCALVERSIONID\nOBJECT = VERSIONID\nVALUE = 61\nEND_OBJECT = VERSIONID\n\0'
    assert metadata_value(text, "VERSIONID") == "61"
    with pytest.raises(ValueError):
        metadata_value(text, "MISSING")


def test_water_mapping_and_fill_rejection():
    hist = class_histogram(np.array([17, 16, 7]), np.array([1., 2., 3.]))
    assert hist[0] == 1 and hist[16] == 2 and hist[7] == 3
    with pytest.raises(ValueError):
        class_histogram(np.array([255]), np.array([1.]))
    assert class_histogram(np.array([255]), np.array([0.])).sum() == 0


@pytest.mark.parametrize("lat,lon", [(-23, 117), (-23, -69), (-21, 135), (0, 0), (60, 0)])
def test_sinusoidal_area_matches_geographic_cell(lat, lon):
    radius = 6371007.181
    cell = Cell(lat, lon)
    polygon = cell_polygon(cell, radius)
    area = shapely.area(polygon)*(RADIUS_KM*1000/radius)**2/1e6
    assert area == pytest.approx(cell.area_km2, abs=1e-5)


def test_coordinate_sampling_uses_row_direction_and_pixel_edges():
    data = np.array([[1, 2], [3, 4]], dtype=np.uint8)
    tile = NativeTile(Path("synthetic"), 6371007.181, 0., 1000., 500., 500., {"LC_Type1": data})
    lat = np.rad2deg(np.array([750., 250.])/tile.radius)
    lon = np.rad2deg(np.array([250., 750.])/(tile.radius*np.cos(np.deg2rad(lat))))
    np.testing.assert_array_equal(tile.sample(lon, lat)["LC_Type1"], [1, 4])
    with pytest.raises(ValueError):
        tile.sample(-1, 0)


def test_real_pilot_files_identity_and_area_when_present():
    root = Path(__file__).resolve().parents[1]
    sources = root/"results/campaigns/land_reconcile_20260914_v1/sources/native-modis-c61-2022"
    files = sorted(sources.glob("*.hdf"))
    if not files:
        pytest.skip("Native source data are optional outside the local campaign")
    for path in files:
        tile = NativeTile.read(path)
        lon = {"h11v11": -69, "h28v11": 117, "h30v11": 135}[path.name.split(".")[2]]
        lat = -21 if lon == 135 else -23
        ledger = tile.unmasked_ledger(Cell(lat, lon))
        assert ledger["class_areas_km2"].sum() == pytest.approx(Cell(lat, lon).area_km2, abs=1e-5)
        assert ledger["qc_areas_km2"][255] == 0
