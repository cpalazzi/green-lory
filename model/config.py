"""Paths and physical constants shared by the model.

Large data files live under data/ (data/README.md gives their sources and hashes) and small
tracked inputs under inputs/. Every path here is absolute, so the model runs from any working
directory. Where the Mac and ARC keep a file in different folders, the first existing location
is used.
"""
from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data"
INPUTS_DIR = REPO_ROOT / "inputs"
PLANTS_DIR = REPO_ROOT / "model" / "plants"


def _first_existing(*candidates: Path) -> Path:
    """The first candidate that exists, else the first one (so error messages name it)."""
    return next((path for path in candidates if path.exists()), candidates[0])


# Hourly 2019 output per unit of capacity for fixed PV, tracking PV and wind on a 1-degree
# grid, as used by Salmon and Banares-Alcantara (2022). Runs set weather_dir in runs.yaml.
WEATHER_DIR = DATA_DIR / "weather_data" / "salmon2022_pv_wind_2019_hourly_1deg"

# Land-table inputs (model/land_table.py). ARC keeps them in data/, the Mac in data/land/.
LAND_COVER_FILE = DATA_DIR / "MCD12C1.A2022001.061.2023244164746.hdf"
GEBCO_FILE = _first_existing(
    DATA_DIR / "GEBCO_2025_sub_ice.nc", DATA_DIR / "land" / "GEBCO_2025_sub_ice.nc"
)
WDPA_SHAPEFILES = tuple(
    _first_existing(
        DATA_DIR / f"WDPA_Feb2026_Public_shp_{index}" / "WDPA_Feb2026_Public_shp-polygons.shp",
        DATA_DIR / "land" / f"WDPA_Feb2026_Public_shp_{index}" / "WDPA_Feb2026_Public_shp-polygons.shp",
    )
    for index in range(3)
)
BATHYMETRY_FILE = DATA_DIR / "model_bathymetry.nc"
COUNTRIES_GEOJSON = DATA_DIR / "countries.geojson"

# The 1,377 ports of the green-porpoise shipping network (Verschuur et al. 2024)
PORTS_CSV = DATA_DIR / "ports" / "verschuur2024_ports.csv"

# Defaults: the improved runs' YAML (DEA 2050 costs, co-located land use) and the plant bundle
DEFAULT_TECH_YAML = INPUTS_DIR / "tech_config_ammonia_plant_2050_dea_colocated.yaml"
DEFAULT_PLANT_DIR = PLANTS_DIR / "ammonia_plant_1mt"

# Higher heating values used to convert energy flows to tonnes
HYDROGEN_HHV_MWH_PER_T = 39.4
AMMONIA_HHV_MWH_PER_T = 6.25


def repo_path(path: str | Path) -> Path:
    """A path as given if absolute, otherwise relative to the repository root."""
    path = Path(path)
    return path if path.is_absolute() else REPO_ROOT / path
