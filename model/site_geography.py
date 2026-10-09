"""Site geography: the per-cell and per-site inputs of the site costs, read from global rasters.

model/site_costs.py turns these inputs into the cost factors of each cell. This module runs once,
on a machine that holds the data of data/README.md (it needs rasterio), and writes small tables
to inputs/site_costs/:

    python -m model.site_geography cells    # cell_geography.csv
    python -m model.site_geography sites    # calibration_sites.csv
    python -m model.site_geography fit      # remoteness_fit.json and remoteness_fit.md

cells
    Every 1-degree cell centred on an integer coordinate (the land tables' cells) from 55 S to
    83 N, sea included (atolls too small for the lakes database's land mask still have land in
    the land tables):
    - the travel time to the nearest city of at least N people, for each N of CITY_SIZES: the
      median over the cell's land pixels (Nelson et al. 2019, 30 arc-seconds; minutes; empty
      where the cell has no land pixel);
    - the great-circle distance from the cell centre to the nearest of the 1,377 ports of the
      shipping network (data/ports/verschuur2024_ports.csv);
    - water: the area of sea (outside the land mask of GLWD v2), of each kind of inland water
      (WATER_CLASSES), whether the cell holds a water source (WATER_SOURCE_CLASSES covering at
      least WATER_SOURCE_MIN_SHARE of the cell, or RIVER_SOURCE_CLASSES at least
      RIVER_SOURCE_MIN_SHARE) and the great-circle distance between cell centres to the nearest
      cell that does.
sites
    The sites of the two AEMO location-factor tables (inputs/site_costs/*_location_factors.csv),
    located with the GeoNames gazetteer of Australia, with their travel times (at the site) and
    their distances to the nearest network port and to their state capital's port.
fit
    The remoteness factors of model/site_costs.py fitted to the sites of each table:
    equipment = 1 + a x distance (capped at the table's largest factor) and installation =
    1 + b x min(travel time to a city of at least N people, saturation), N and the saturation
    chosen by leave-one-out cross-validation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from .config import (
    EARTH_RADIUS_KM,
    GEONAMES_AU_FILE,
    GLWD_AREA_DIR,
    NELSON_TRAVEL_TIME_DIR,
    PORTS_CSV,
    SITE_COSTS_DIR,
)

LOGGER = logging.getLogger(__name__)

CELL_GEOGRAPHY_CSV = SITE_COSTS_DIR / "cell_geography.csv"
# Results of the remote permanence reads, kept so that a rerun reads only new cells
GSW_PERMANENT_WATER_CSV = SITE_COSTS_DIR / "gsw_permanent_water.csv"
CALIBRATION_SITES_CSV = SITE_COSTS_DIR / "calibration_sites.csv"
REMOTENESS_FIT_JSON = SITE_COSTS_DIR / "remoteness_fit.json"
LOCATION_FACTOR_TABLES = {
    "aurecon2024": SITE_COSTS_DIR / "aurecon2024_location_factors.csv",
    "ghd2025": SITE_COSTS_DIR / "ghd2025_location_factors.csv",
}
GHD_CAPITAL_FACTORS_CSV = SITE_COSTS_DIR / "ghd2025_capital_installation_factors.csv"

# Nelson et al. (2019): travel_time_to_cities_<k>.tif. Layers 1-9 hold cities in population
# bands (1: 5-50 million, 2: 1-5 million, 3: 0.5-1 million, 4: 200-500 thousand, 5: 100-200
# thousand, 8: 10-20 thousand), layers 10-12 every city above 20, 50 and 5 thousand. The time
# to the nearest city of at least N people is the smallest time over the layers that cover N.
CITY_SIZES = {
    "5k": (12,),
    "10k": (8, 10),
    "20k": (10,),
    "50k": (11,),
    "100k": (1, 2, 3, 4, 5),
    "200k": (1, 2, 3, 4),
    "500k": (1, 2, 3),
    "1m": (1, 2),
    "5m": (1,),
}
NELSON_NODATA = 65535
NELSON_PIXELS_PER_DEGREE = 120
NELSON_NORTH, NELSON_SOUTH = 85.0, -60.0

# GLWD v2 (Lehner et al. 2025), area of each class per 15 arc-second pixel in hectares x 10;
# pixels outside its land mask (the sea) hold no data. Columns of cell_geography.csv by class.
WATER_CLASSES = {
    "fresh_lake_km2": ("01",),
    "saline_lake_km2": ("02",),
    "reservoir_km2": ("03",),
    "large_river_km2": ("04", "05"),
    "other_permanent_water_km2": ("06",),
    "ephemeral_water_km2": ("20", "21"),
    "salt_pan_km2": ("32",),
}
GLWD_NODATA = 255
GLWD_PIXELS_PER_DEGREE = 240
GLWD_NORTH, GLWD_SOUTH = 84.0, -56.0

# Water sources (decision of 7 October 2026, CAMPAIGN_PLAN.md section 0b): the sea and
# permanent lakes and reservoirs, GLWD classes 01 freshwater lake, 02 saline lake, 03 reservoir
# and 06 other permanent waterbody; not ephemeral water (20, 21) or salt pans (32). A cell holds
# a source if the sea and these waters cover at least 10 % of it, the share the spatial inputs
# used before (then any water of the 2022 land cover).
WATER_SOURCE_CLASSES = ("sea_km2", "fresh_lake_km2", "saline_lake_km2", "reservoir_km2", "other_permanent_water_km2")
WATER_SOURCE_MIN_SHARE = 0.10
# Large rivers (decision of 9 October 2026): GLWD classes 04 large river and 05 large estuarine
# river are sources where they cover at least 1 % of the cell (about 120 km2 at the equator, a
# river about 1 km wide across the cell). A river is long and narrow, so the 10 % of standing
# water would leave out the Amazon, Congo and Siberian rivers beside which cells would pay for
# hundreds of kilometres of pipeline.
RIVER_SOURCE_CLASSES = ("large_river_km2",)
RIVER_SOURCE_MIN_SHARE = 0.01
# GLWD's saline lakes include salt lakes that are dry most years (Kati Thanda-Lake Eyre, Lake
# Torrens, Makgadikgadi, Salar de Uyuni). Where a cell would hold a source only through its
# saline lake, the lake counts only as far as the JRC Global Surface Water occurrence layer
# (Pekel et al. 2016, version 1.4, 1984-2021, 0.00025 degrees) shows water in at least half of
# the observations. The tiles are read remotely, cell by cell; the result is kept in the table.
GSW_OCCURRENCE_URL = (
    "https://storage.googleapis.com/global-surface-water/downloads2021/occurrence/occurrence_{tile}v1_4_2021.tif"
)
GSW_PERMANENT_OCCURRENCE_PCT = 50
GSW_PIXELS_PER_DEGREE = 4000

# GeoNames admin1 codes of the Australian states; Canberra (ACT) is listed under NSW by GHD
STATE_ADMIN1 = {"ACT": "01", "NSW": "02", "NT": "03", "QLD": "04", "SA": "05", "TAS": "06", "VIC": "07", "WA": "08"}
# Table names that the gazetteer spells differently (state, name in table) -> (admin1, name)
SITE_ALIASES = {
    ("SA", "Kingston SE"): ("05", "Kingston South East"),
    ("TAS", "Campbelltown"): ("06", "Campbell Town"),
    ("TAS", "Derwent Valley"): ("06", "New Norfolk"),  # seat of the Derwent Valley Council
    ("TAS", "Derwent Valley Council"): ("06", "New Norfolk"),
    ("QLD", "Texas"): ("04", "Texas, Goodiwindi"),
    ("NSW", "Canberra"): ("01", "Canberra"),
}
# The state capitals and their ports in the network port set (Sydney: Botany Bay, the main
# cargo port; Adelaide: Port Adelaide). Aurecon set its equipment factors by distance from the
# capital; GeoNames ids of the capitals.
STATE_CAPITALS = {
    "QLD": ("Brisbane", 2174003, "port174"),
    "NSW": ("Sydney", 2147714, "port161"),
    "VIC": ("Melbourne", 2158177, "port729"),
    "SA": ("Adelaide", 2078025, "port929"),
    "TAS": ("Hobart", 2163355, "port469"),
}


# ---------------------------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------------------------

def _unit_vectors(lat_deg, lon_deg) -> np.ndarray:
    lat, lon = np.deg2rad(np.asarray(lat_deg, float)), np.deg2rad(np.asarray(lon_deg, float))
    return np.column_stack([np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)])


def nearest_km(lat, lon, target_lat, target_lon) -> tuple[np.ndarray, np.ndarray]:
    """Great-circle distance (km) from each point to the nearest target, and that target's index."""
    from scipy.spatial import cKDTree

    chord, index = cKDTree(_unit_vectors(target_lat, target_lon)).query(_unit_vectors(lat, lon), k=1)
    return 2.0 * np.arcsin(np.clip(chord / 2.0, 0.0, 1.0)) * EARTH_RADIUS_KM, index


def great_circle_km(lat1, lon1, lat2, lon2) -> np.ndarray:
    a, b = _unit_vectors(lat1, lon1), _unit_vectors(lat2, lon2)
    return np.arccos(np.clip((a * b).sum(axis=1), -1.0, 1.0)) * EARTH_RADIUS_KM


def band_area_km2(north_deg, south_deg, width_deg) -> np.ndarray:
    """Area of a latitude-longitude box on the sphere."""
    return EARTH_RADIUS_KM ** 2 * np.deg2rad(width_deg) * (np.sin(np.deg2rad(north_deg)) - np.sin(np.deg2rad(south_deg)))


def _cell_blocks(band: np.ndarray, pixels_per_degree: int) -> np.ndarray:
    """A band of whole-degree rows as (rows, 360 cells, pixels): cell b covers b - 180.5 to b - 179.5 E."""
    rolled = np.roll(band, pixels_per_degree // 2, axis=1)
    return rolled.reshape(band.shape[0], 360, pixels_per_degree)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 24), b""):
            digest.update(chunk)
    return digest.hexdigest()


# ---------------------------------------------------------------------------------------------
# Cells
# ---------------------------------------------------------------------------------------------

def _travel_time_rows(nelson_dir: Path, latitudes: list[int]) -> pd.DataFrame:
    import rasterio
    from rasterio.windows import Window

    layers = sorted({layer for group in CITY_SIZES.values() for layer in group})
    sources = {layer: rasterio.open(nelson_dir / f"travel_time_to_cities_{layer}.tif") for layer in layers}
    try:
        frames = []
        for lat in latitudes:
            top = int(round((NELSON_NORTH - (lat + 0.5)) * NELSON_PIXELS_PER_DEGREE))
            window = Window(0, top, 360 * NELSON_PIXELS_PER_DEGREE, NELSON_PIXELS_PER_DEGREE)
            bands = {layer: source.read(1, window=window) for layer, source in sources.items()}
            row = {"latitude": float(lat), "longitude": np.arange(-180.0, 180.0)}
            for label, group in CITY_SIZES.items():
                minutes = np.minimum.reduce([bands[layer] for layer in group])
                cells = _cell_blocks(minutes, NELSON_PIXELS_PER_DEGREE).transpose(1, 0, 2).reshape(360, -1)
                cells = np.where(cells >= NELSON_NODATA, np.nan, cells.astype(np.float32))
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", category=RuntimeWarning)  # cells without land
                    row[f"travel_min_city_{label}_median"] = np.nanmedian(cells, axis=1)
            frames.append(pd.DataFrame(row))
        return pd.concat(frames, ignore_index=True)
    finally:
        for source in sources.values():
            source.close()


def _water_rows(glwd_dir: Path, latitudes: list[int]) -> pd.DataFrame:
    import rasterio
    from rasterio.windows import Window

    codes = ["00"] + sorted({code for group in WATER_CLASSES.values() for code in group})
    sources = {code: rasterio.open(glwd_dir / f"GLWD_v2_0_class_{code}_ha_x10.tif") for code in codes}
    try:
        frames = []
        step = 1.0 / GLWD_PIXELS_PER_DEGREE
        for lat in latitudes:
            top = int(round((GLWD_NORTH - (lat + 0.5)) * GLWD_PIXELS_PER_DEGREE))
            window = Window(0, top, 360 * GLWD_PIXELS_PER_DEGREE, GLWD_PIXELS_PER_DEGREE)
            north_edges = lat + 0.5 - step * np.arange(GLWD_PIXELS_PER_DEGREE)
            pixel_km2 = band_area_km2(north_edges, north_edges - step, step)  # one pixel per row
            land = _cell_blocks(sources["00"].read(1, window=window) != GLWD_NODATA, GLWD_PIXELS_PER_DEGREE)
            land_pixels = land.sum(axis=2)  # (rows, cells)
            row = {
                "latitude": float(lat),
                "longitude": np.arange(-180.0, 180.0),
                "cell_area_km2": band_area_km2(lat + 0.5, lat - 0.5, 1.0),
                "land_km2": (land_pixels * pixel_km2[:, None]).sum(axis=0),
                "sea_km2": ((GLWD_PIXELS_PER_DEGREE - land_pixels) * pixel_km2[:, None]).sum(axis=0),
            }
            for column, group in WATER_CLASSES.items():
                total = np.zeros(360)
                for code in group:
                    values = _cell_blocks(sources[code].read(1, window=window), GLWD_PIXELS_PER_DEGREE)
                    total += np.where(values == GLWD_NODATA, 0, values).sum(axis=(0, 2), dtype=np.int64) / 1000.0
                row[column] = total  # hectares x 10 -> km2
            frames.append(pd.DataFrame(row))
        return pd.concat(frames, ignore_index=True)
    finally:
        for source in sources.values():
            source.close()


def _gsw_tile_names(lat: float, lon: float) -> list[tuple[str, float, float]]:
    """GSW tiles (10 degrees, named by their north-west corner) that overlap a cell."""
    tiles = []
    for west in range(int(np.floor((lon - 0.5) / 10.0)) * 10, int(np.ceil((lon + 0.5) / 10.0)) * 10, 10):
        if not -180 <= west < 180:
            continue  # the part of the dateline cell beyond 180 degrees (no saline lakes there)
        for north in range(int(np.ceil((lat + 0.5) / 10.0)) * 10, int(np.floor((lat - 0.5) / 10.0)) * 10, -10):
            name = f"{abs(west)}{'E' if west >= 0 else 'W'}_{abs(north)}{'N' if north >= 0 else 'S'}"
            tiles.append((name, float(west), float(north)))
    return tiles


def permanent_water_km2(lat: float, lon: float, cell_area_km2: float, threshold_pct: int = GSW_PERMANENT_OCCURRENCE_PCT) -> float:
    """Area of the cell where GSW saw water in at least threshold_pct % of the observations."""
    import rasterio
    from rasterio.errors import RasterioIOError
    from rasterio.windows import from_bounds

    water_pixels = 0
    options = {"GDAL_DISABLE_READDIR_ON_OPEN": "EMPTY_DIR", "GDAL_HTTP_MERGE_CONSECUTIVE_RANGES": "YES",
               "GDAL_HTTP_MAX_RETRY": "5", "GDAL_HTTP_RETRY_DELAY": "5", "GDAL_HTTP_TIMEOUT": "120",
               "GDAL_HTTP_CONNECTTIMEOUT": "30"}
    with rasterio.Env(**options):
        for name, west, north in _gsw_tile_names(lat, lon):
            bounds = (max(lon - 0.5, west), max(lat - 0.5, north - 10.0), min(lon + 0.5, west + 10.0), min(lat + 0.5, north))
            try:
                with rasterio.open("/vsicurl/" + GSW_OCCURRENCE_URL.format(tile=name)) as source:
                    occurrence = source.read(1, window=from_bounds(*bounds, source.transform))
            except RasterioIOError as exc:
                if "404" in str(exc) or "not recognized" in str(exc) or "No such file" in str(exc):
                    LOGGER.warning("GSW tile %s not available; counted as no water", name)
                    continue
                raise  # a failed read must not pass for dry land
            water_pixels += int(((occurrence >= threshold_pct) & (occurrence <= 100)).sum())
    return cell_area_km2 * water_pixels / GSW_PIXELS_PER_DEGREE ** 2


def _read_gsw_cache(path: Path) -> dict[tuple[float, float], float]:
    if not path.exists():
        return {}
    frame = pd.read_csv(path)
    if not (frame["occurrence_threshold_pct"] == GSW_PERMANENT_OCCURRENCE_PCT).all():
        raise ValueError(f"{path} was read at another occurrence threshold; delete it to read again")
    return {(float(lat), float(lon)): float(km2) for lat, lon, km2 in
            frame[["latitude", "longitude", "gsw_permanent_water_km2"]].itertuples(index=False)}


def _append_gsw_cache(path: Path, key: tuple[float, float], km2: float) -> None:
    new = not path.exists()
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as handle:
        if new:
            handle.write("latitude,longitude,gsw_permanent_water_km2,occurrence_threshold_pct,read_utc\n")
        stamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        handle.write(f"{key[0]:g},{key[1]:g},{km2:.3f},{GSW_PERMANENT_OCCURRENCE_PCT},{stamp}\n")


def water_sources(
    cells: pd.DataFrame,
    classes=WATER_SOURCE_CLASSES,
    min_share: float = WATER_SOURCE_MIN_SHARE,
    river_classes=RIVER_SOURCE_CLASSES,
    river_min_share: float = RIVER_SOURCE_MIN_SHARE,
) -> pd.DataFrame:
    """Whether each cell holds a water source, its kind, and the distance to the nearest source cell.

    A cell holds a source if the standing water of `classes` covers `min_share` of it or the
    rivers of `river_classes` cover `river_min_share` (no rivers: river_classes=()). The kind is
    sea or lake by the larger standing water, or river where only the rivers make the source.
    """
    source_km2 = cells[list(classes)].sum(axis=1)
    standing = source_km2 >= min_share * cells["cell_area_km2"]
    river = cells[list(river_classes)].sum(axis=1) >= river_min_share * cells["cell_area_km2"]
    is_source = standing | river
    lake_km2 = source_km2 - cells["sea_km2"]
    kind = np.where(~standing, "river", np.where(cells["sea_km2"] >= lake_km2, "sea", "lake"))
    sources = cells.loc[is_source]
    distance, index = nearest_km(cells["latitude"], cells["longitude"], sources["latitude"], sources["longitude"])
    distance = np.where(is_source, 0.0, distance)
    return pd.DataFrame(
        {
            "water_source": is_source.to_numpy(),
            "water_source_distance_km": distance,
            "water_source_kind": np.asarray(kind)[is_source.to_numpy().nonzero()[0][index]],
        },
        index=cells.index,
    )


def build_cell_geography(
    nelson_dir: Path = NELSON_TRAVEL_TIME_DIR,
    glwd_dir: Path = GLWD_AREA_DIR,
    ports_csv: Path = PORTS_CSV,
    latitudes: list[int] | None = None,
    gsw_cache_csv: Path = GSW_PERMANENT_WATER_CSV,
) -> pd.DataFrame:
    """The per-cell table of the module docstring."""
    if latitudes is None:
        latitudes = list(range(int(GLWD_SOUTH + 1), int(GLWD_NORTH)))  # -55 ... 83
    started = time.perf_counter()
    water = _water_rows(glwd_dir, latitudes)
    LOGGER.info("Water areas of %d latitude rows in %.0f s", len(latitudes), time.perf_counter() - started)
    # Saline lakes count only where they are permanent (see GSW_OCCURRENCE_URL)
    others = [column for column in WATER_SOURCE_CLASSES if column != "saline_lake_km2"]
    threshold = WATER_SOURCE_MIN_SHARE * water["cell_area_km2"]
    candidates = water.index[(water[others].sum(axis=1) < threshold)
                             & (water[others].sum(axis=1) + water["saline_lake_km2"] >= threshold)]
    started = time.perf_counter()
    water["gsw_permanent_water_km2"] = np.nan
    known = _read_gsw_cache(gsw_cache_csv)
    for count, index in enumerate(candidates, start=1):
        key = (float(water.at[index, "latitude"]), float(water.at[index, "longitude"]))
        if key not in known:
            known[key] = permanent_water_km2(*key, water.at[index, "cell_area_km2"])
            _append_gsw_cache(gsw_cache_csv, key, known[key])
        water.at[index, "gsw_permanent_water_km2"] = known[key]
        if count % 20 == 0:
            LOGGER.info("Permanence of saline lakes: %d of %d cells", count, len(candidates))
    LOGGER.info("Permanence of %d saline-lake cells in %.0f s", len(candidates), time.perf_counter() - started)
    water["saline_lake_counted_km2"] = water["saline_lake_km2"]
    water.loc[candidates, "saline_lake_counted_km2"] = np.minimum(
        water.loc[candidates, "saline_lake_km2"], water.loc[candidates, "gsw_permanent_water_km2"])
    counted = (*others, "saline_lake_counted_km2")
    water = water.join(water_sources(water, classes=counted))
    # the distances with every saline lake, with none and without the rivers, to show what the
    # permanence test and the river rule change
    for suffix, options in (
        ("with_all_saline_lakes", {"classes": WATER_SOURCE_CLASSES}),
        ("without_saline_lakes", {"classes": tuple(others)}),
        ("without_rivers", {"classes": counted, "river_classes": ()}),
    ):
        water[f"water_source_distance_km_{suffix}"] = water_sources(water, **options)["water_source_distance_km"]

    cells = water
    ports = pd.read_csv(ports_csv)
    cells["port_distance_km"], index = nearest_km(cells["latitude"], cells["longitude"], ports["lat"], ports["lon"])
    cells["nearest_port_id"] = ports["id"].to_numpy()[index]

    nelson_rows = [lat for lat in latitudes if NELSON_SOUTH + 0.5 <= lat <= NELSON_NORTH - 0.5]
    started = time.perf_counter()
    travel = _travel_time_rows(nelson_dir, nelson_rows)
    LOGGER.info("Travel times of %d latitude rows in %.0f s", len(nelson_rows), time.perf_counter() - started)
    cells = cells.merge(travel, on=["latitude", "longitude"], how="left")
    return cells.sort_values(["latitude", "longitude"]).reset_index(drop=True)


# ---------------------------------------------------------------------------------------------
# Calibration sites
# ---------------------------------------------------------------------------------------------

GEONAMES_COLUMNS = [
    "geonameid", "name", "asciiname", "alternatenames", "latitude", "longitude", "feature_class",
    "feature_code", "country_code", "cc2", "admin1", "admin2", "admin3", "admin4", "population",
    "elevation", "dem", "timezone", "modified",
]


def load_gazetteer(path: Path = GEONAMES_AU_FILE) -> pd.DataFrame:
    return pd.read_csv(path, sep="\t", header=None, names=GEONAMES_COLUMNS, quoting=3, keep_default_na=False,
                       dtype={"admin1": str, "admin2": str, "feature_class": str, "feature_code": str},
                       low_memory=False)


def locate_site(gazetteer: pd.DataFrame, state: str, site: str) -> pd.Series:
    """The gazetteer entry of a table site: a populated place of that name in the state (the most
    populous), else any feature of that name there."""
    admin1, name = SITE_ALIASES.get((state, site), (STATE_ADMIN1[state], site))
    in_state = gazetteer[gazetteer["admin1"] == admin1]
    named = in_state[(in_state["name"].str.lower() == name.lower()) | (in_state["asciiname"].str.lower() == name.lower())]
    if named.empty:
        raise LookupError(f"{site} ({state}) is not in the gazetteer as {name!r}")
    places = named[named["feature_class"] == "P"]
    pool = places if not places.empty else named
    return pool.sort_values("population", ascending=False).iloc[0]


def _sample_travel_times(nelson_dir: Path, lat: np.ndarray, lon: np.ndarray) -> pd.DataFrame:
    """Travel time at each point: the pixel's, or the nearest land pixel's within 3 pixels."""
    import rasterio
    from rasterio.windows import Window

    layers = sorted({layer for group in CITY_SIZES.values() for layer in group})
    radius = 3
    offsets = np.array([(dy, dx) for dy in range(-radius, radius + 1) for dx in range(-radius, radius + 1)])
    order = np.argsort(np.hypot(offsets[:, 0], offsets[:, 1]), kind="stable")
    out = {f"travel_min_city_{label}": np.full(len(lat), np.nan) for label in CITY_SIZES}
    sources = {layer: rasterio.open(nelson_dir / f"travel_time_to_cities_{layer}.tif") for layer in layers}
    try:
        for i, (y, x) in enumerate(zip(lat, lon)):
            row = int((NELSON_NORTH - y) * NELSON_PIXELS_PER_DEGREE)
            col = int((x + 180.0) * NELSON_PIXELS_PER_DEGREE)
            window = Window(col - radius, row - radius, 2 * radius + 1, 2 * radius + 1)
            patches = {layer: source.read(1, window=window, boundless=True, fill_value=NELSON_NODATA)
                       for layer, source in sources.items()}
            for label, group in CITY_SIZES.items():
                patch = np.minimum.reduce([patches[layer] for layer in group])
                for k in order:
                    value = patch[radius + offsets[k][0], radius + offsets[k][1]]
                    if value < NELSON_NODATA:
                        out[f"travel_min_city_{label}"][i] = float(value)
                        break
    finally:
        for source in sources.values():
            source.close()
    return pd.DataFrame(out)


def build_calibration_sites(
    tables: dict[str, Path] = LOCATION_FACTOR_TABLES,
    gazetteer_path: Path = GEONAMES_AU_FILE,
    nelson_dir: Path = NELSON_TRAVEL_TIME_DIR,
    ports_csv: Path = PORTS_CSV,
) -> pd.DataFrame:
    """Every site of the location-factor tables with its location, travel times and port distances."""
    gazetteer = load_gazetteer(gazetteer_path)
    frames = []
    for source, path in tables.items():
        table = pd.read_csv(path)
        table.insert(0, "source", source)
        frames.append(table)
    sites = pd.concat(frames, ignore_index=True)
    located = [locate_site(gazetteer, state, site) for state, site in zip(sites["state"], sites["site"])]
    sites["geonameid"] = [int(entry["geonameid"]) for entry in located]
    sites["geonames_name"] = [entry["name"] for entry in located]
    sites["geonames_feature"] = [f"{entry['feature_class']}.{entry['feature_code']}" for entry in located]
    sites["geonames_population"] = [int(entry["population"]) for entry in located]
    sites["latitude"] = [float(entry["latitude"]) for entry in located]
    sites["longitude"] = [float(entry["longitude"]) for entry in located]

    ports = pd.read_csv(ports_csv).set_index("id")
    sites["port_distance_km"], index = nearest_km(sites["latitude"], sites["longitude"], ports["lat"], ports["lon"])
    sites["nearest_port"] = ports["name"].to_numpy()[index]
    capitals = gazetteer.set_index("geonameid")
    capital_state = sites["state"].replace({"ACT": "NSW"})
    capital_ids = [STATE_CAPITALS[state][1] for state in capital_state]
    capital_ports = [STATE_CAPITALS[state][2] for state in capital_state]
    sites["capital_distance_km"] = great_circle_km(
        sites["latitude"], sites["longitude"],
        capitals.loc[capital_ids, "latitude"].astype(float), capitals.loc[capital_ids, "longitude"].astype(float),
    )
    sites["capital_port_distance_km"] = great_circle_km(
        sites["latitude"], sites["longitude"], ports.loc[capital_ports, "lat"], ports.loc[capital_ports, "lon"],
    )
    travel = _sample_travel_times(nelson_dir, sites["latitude"].to_numpy(), sites["longitude"].to_numpy())
    return pd.concat([sites, travel], axis=1)


# ---------------------------------------------------------------------------------------------
# The fit
# ---------------------------------------------------------------------------------------------

def _through_origin(x: np.ndarray, y: np.ndarray) -> float:
    denominator = float(x @ x)
    return float(x @ y) / denominator if denominator > 0 else 0.0


def fit_equipment(distance_km: np.ndarray, factor: np.ndarray) -> dict:
    """equipment = 1 + a x distance, capped at the largest factor of the table."""
    cap = float(factor.max() - 1.0)
    a = _through_origin(distance_km, factor - 1.0)
    predicted = 1.0 + np.minimum(a * distance_km, cap)
    return {
        "a_per_1000_km": a * 1000.0,
        "cap": 1.0 + cap,
        "rmse": float(np.sqrt(np.mean((predicted - factor) ** 2))),
        "residuals": (factor - predicted).tolist(),
    }


def _installation_prediction(minutes: np.ndarray, saturation: float, b: float) -> np.ndarray:
    return 1.0 + b * np.minimum(minutes, saturation)


def fit_installation(sites: pd.DataFrame, factor: np.ndarray, saturation_step_min: float = 30.0) -> dict:
    """installation = 1 + b x min(travel time to a city of at least N, saturation).

    For each N and each saturation (every saturation_step_min minutes up to the largest travel
    time of the sites, which is the cap when nothing saturates earlier), b is the least-squares
    slope through 1 at the city; N and the saturation are those of the smallest leave-one-out
    error.
    """
    y = factor - 1.0
    candidates = []
    for label in CITY_SIZES:
        minutes = sites[f"travel_min_city_{label}"].to_numpy(float)
        largest = float(np.nanmax(minutes))
        saturations = sorted({*np.arange(saturation_step_min, largest, saturation_step_min), largest})
        for saturation in saturations:
            x = np.minimum(minutes, saturation)
            b = _through_origin(x, y)
            residuals = y - b * x
            loo = residuals / (1.0 - x * x / float(x @ x))  # leave-one-out residuals of a one-slope fit
            candidates.append({
                "city_size": label,
                "saturation_min": float(saturation),
                "b_per_hour": b * 60.0,
                "rmse": float(np.sqrt(np.mean(residuals ** 2))),
                "loocv_rmse": float(np.sqrt(np.mean(loo ** 2))),
                "factor_at_saturation": 1.0 + b * saturation,
            })
    table = pd.DataFrame(candidates).sort_values("loocv_rmse").reset_index(drop=True)
    best = table.iloc[0].to_dict()
    minutes = sites[f"travel_min_city_{best['city_size']}"].to_numpy(float)
    predicted = _installation_prediction(minutes, best["saturation_min"], best["b_per_hour"] / 60.0)
    best["residuals"] = (factor - predicted).tolist()
    best_by_size = table.groupby("city_size", sort=False).head(1).set_index("city_size")
    return {"best": best, "best_by_city_size": best_by_size.reindex(list(CITY_SIZES)).reset_index().to_dict("records")}


def fit_remoteness(sites: pd.DataFrame, capital_factors_csv: Path = GHD_CAPITAL_FACTORS_CSV) -> dict:
    """The fit of each table (see the module docstring)."""
    capital_factor = pd.read_csv(capital_factors_csv).set_index("state")["installation_factor_relative_to_melbourne"]
    results = {}
    for source, group in sites.groupby("source", sort=False):
        group = group.reset_index(drop=True)
        if source == "aurecon2024":
            installation = group["installation_cost_factor"].to_numpy(float)
            installation_basis = "installation cost factor of Table 11-1 (relative to the state capital)"
            distance_column = "capital_port_distance_km"
        else:
            # GHD's factors include each capital's price level (Table 8.2); the fit keeps remoteness only
            installation = (group["installation_om_cost_factor"] / group["state"].replace({"ACT": "NSW"}).map(capital_factor)).to_numpy(float)
            installation_basis = "installation factor of Table 8.3 divided by the state capital's factor of Table 8.2"
            distance_column = "port_distance_km"
        equipment = fit_equipment(group[distance_column].to_numpy(float), group["equipment_cost_factor"].to_numpy(float))
        equipment["distance"] = distance_column
        results[source] = {
            "sites": int(len(group)),
            "equipment": equipment,
            "installation": {**fit_installation(group, installation), "basis": installation_basis},
            "site_names": group["site"].tolist(),
            "installation_observed": installation.tolist(),
        }
    return results


def write_fit_report(fit: dict, sites: pd.DataFrame, path: Path) -> None:
    lines = ["# Remoteness factors fitted to the AEMO location-factor tables", "",
             "Written by `python -m model.site_geography fit`; read with `model/site_costs.py`.", ""]
    for source, result in fit.items():
        equipment, installation = result["equipment"], result["installation"]["best"]
        lines += [
            f"## {source} ({result['sites']} sites)", "",
            f"- Equipment: 1 + {equipment['a_per_1000_km']:.4f} per 1,000 km of {equipment['distance']}, "
            f"capped at {equipment['cap']:.2f}; RMSE {equipment['rmse']:.3f}.",
            f"- Installation ({result['installation']['basis']}): 1 + {installation['b_per_hour']:.4f} per hour "
            f"to a city of at least {installation['city_size']} people, saturating at "
            f"{installation['saturation_min']:.0f} min (factor {installation['factor_at_saturation']:.2f}); "
            f"RMSE {installation['rmse']:.3f}, leave-one-out RMSE {installation['loocv_rmse']:.3f}.",
            "", "Best saturation for each city size:", "",
            "| city size | saturation (min) | b per hour | factor at saturation | RMSE | leave-one-out RMSE |",
            "|---|---|---|---|---|---|",
        ]
        for row in result["installation"]["best_by_city_size"]:
            lines.append(f"| {row['city_size']} | {row['saturation_min']:.0f} | {row['b_per_hour']:.4f} | "
                         f"{row['factor_at_saturation']:.2f} | {row['rmse']:.3f} | {row['loocv_rmse']:.3f} |")
        group = sites[sites["source"] == source].reset_index(drop=True)
        lines += ["", "| site | state | travel (min) | installation observed | fitted | equipment observed | fitted |",
                  "|---|---|---|---|---|---|---|"]
        travel = group[f"travel_min_city_{installation['city_size']}"]
        for i, site in enumerate(result["site_names"]):
            observed = result["installation_observed"][i]
            lines.append(
                f"| {site} | {group.at[i, 'state']} | {travel[i]:.0f} | {observed:.2f} | "
                f"{observed - installation['residuals'][i]:.2f} | {group.at[i, 'equipment_cost_factor']:.2f} | "
                f"{group.at[i, 'equipment_cost_factor'] - equipment['residuals'][i]:.2f} |"
            )
        lines.append("")
    path.write_text("\n".join(lines) + "\n")


# ---------------------------------------------------------------------------------------------
# Command line
# ---------------------------------------------------------------------------------------------

def _write_with_provenance(frame: pd.DataFrame, path: Path, inputs: dict[str, Path], decimals: int = 3) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.round(decimals).to_csv(path, index=False)
    provenance = {
        "written_by": "python -m model.site_geography",
        "rows": int(len(frame)),
        "sha256": _file_sha256(path),
        "inputs": {name: {"path": str(source), "sha256": _file_sha256(source)} for name, source in inputs.items()},
    }
    path.with_name(path.stem + "_provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    cells = commands.add_parser("cells", help="per-cell travel times, port distances and water")
    cells.add_argument("--output", type=Path, default=CELL_GEOGRAPHY_CSV)
    cells.add_argument("--min-lat", type=int, default=None)
    cells.add_argument("--max-lat", type=int, default=None)
    sites = commands.add_parser("sites", help="the calibration sites")
    sites.add_argument("--output", type=Path, default=CALIBRATION_SITES_CSV)
    fit = commands.add_parser("fit", help="the remoteness fit")
    fit.add_argument("--sites", type=Path, default=CALIBRATION_SITES_CSV)
    fit.add_argument("--output", type=Path, default=REMOTENESS_FIT_JSON)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    if args.command == "cells":
        latitudes = None
        if args.min_lat is not None or args.max_lat is not None:
            latitudes = list(range(args.min_lat if args.min_lat is not None else -55,
                                   (args.max_lat if args.max_lat is not None else 83) + 1))
        frame = build_cell_geography(latitudes=latitudes)
        inputs = {"ports": PORTS_CSV, "gsw_permanent_water": GSW_PERMANENT_WATER_CSV}
        inputs.update({f"nelson_cities_{layer}": NELSON_TRAVEL_TIME_DIR / f"travel_time_to_cities_{layer}.tif"
                       for layer in sorted({layer for group in CITY_SIZES.values() for layer in group})})
        inputs.update({f"glwd_class_{code}": GLWD_AREA_DIR / f"GLWD_v2_0_class_{code}_ha_x10.tif"
                       for code in ["00"] + sorted({c for group in WATER_CLASSES.values() for c in group})})
        _write_with_provenance(frame, args.output, inputs)
    elif args.command == "sites":
        frame = build_calibration_sites()
        inputs = {**LOCATION_FACTOR_TABLES, "gazetteer": GEONAMES_AU_FILE, "ports": PORTS_CSV}
        inputs.update({f"nelson_cities_{layer}": NELSON_TRAVEL_TIME_DIR / f"travel_time_to_cities_{layer}.tif"
                       for layer in sorted({layer for group in CITY_SIZES.values() for layer in group})})
        _write_with_provenance(frame, args.output, inputs, decimals=5)
    else:
        sites_frame = pd.read_csv(args.sites)
        result = fit_remoteness(sites_frame)
        args.output.write_text(json.dumps({"sites_sha256": _file_sha256(args.sites), "fits": result}, indent=2) + "\n")
        write_fit_report(result, sites_frame, args.output.with_suffix(".md"))
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
