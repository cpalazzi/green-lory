"""Canonical repository data paths with temporary legacy fallbacks."""
from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data"
WEATHER_DATA_DIR = DATA_DIR / "weather_data"
DEA_REFERENCE_DIR = DATA_DIR / "dea_reference"

LAND_COVER_FILE = DATA_DIR / "MCD12C1.A2022001.061.2023244164746.hdf"
MAX_CAPACITIES_FILE = DATA_DIR / "max_capacities.csv"
COUNTRIES_GEOJSON = DATA_DIR / "countries.geojson"
PORT_LOCATIONS_CSV = DATA_DIR / "port_locations.csv"
TRAVEL_TIME_CSV = DATA_DIR / "travel_time_by_cell.csv"

BATHYMETRY_CANONICAL_FILE = DATA_DIR / "model_bathymetry.nc"
BATHYMETRY_LEGACY_FILE = WEATHER_DATA_DIR / "model_bathymetry.nc"

GEBCO_SLOPE_CANONICAL_FILE = DATA_DIR / "GEBCO_2025_sub_ice.nc"
GEBCO_SLOPE_LEGACY_FILE = DATA_DIR / "external" / "gebco" / "GEBCO_2025_sub_ice.nc"

WDPA_SHAPEFILE_NAME = "WDPA_Feb2026_Public_shp-polygons.shp"


def _first_existing_path(*candidates: Path) -> Path:
    if not candidates:
        raise ValueError("At least one candidate path is required.")
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def wdpa_shapefile(index: int) -> Path:
    stem = f"WDPA_Feb2026_Public_shp_{int(index)}"
    return _first_existing_path(
        DATA_DIR / stem / WDPA_SHAPEFILE_NAME,
        DATA_DIR / "external" / "wdpa" / stem / WDPA_SHAPEFILE_NAME,
    )


BATHYMETRY_FILE = _first_existing_path(BATHYMETRY_CANONICAL_FILE, BATHYMETRY_LEGACY_FILE)
GEBCO_SLOPE_FILE = _first_existing_path(GEBCO_SLOPE_CANONICAL_FILE, GEBCO_SLOPE_LEGACY_FILE)
WDPA_SHAPEFILES = tuple(wdpa_shapefile(index) for index in range(3))