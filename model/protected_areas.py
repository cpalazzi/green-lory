"""Protected areas: the share of each 1-degree cell's land and water in protected areas.

The World Database on Protected Areas (WDPA, February 2026, data/README.md) gives the polygons
and the MODIS land cover (MCD12C1, 0.05 degrees) each pixel's water fraction. For every cell
centred on an integer coordinate from 89 S to 89 N, protected_area_table measures

- the cell's land and water (land_km2, water_km2): the spherical area of each land-cover pixel
  times its land fraction (one minus its water fraction) or its water fraction;
- the parts of them in protected areas (protected_land_km2, protected_water_km2) and their shares
  of the land and of the water (protected_land_pct, protected_water_pct; 0 where the cell has none).

A polygon counts if its status is one of land.protected_area_statuses: on land if its realm is
one of land.protected_area_realms, on water if its realm is one of offshore.protected_area_realms
(inputs/parameters.yaml). A coastal area counts on both. The polygons are rasterised on a grid
PROTECTED_SUBPIXELS times finer than the land cover: a sub-pixel is protected if its centre lies
inside a polygon, and each pixel's protected share is the share of its sub-pixels that are.

The land tables leave the protected land out of the land a plant may use (model/land_table.py)
and the sea runs leave the protected water out of the sea (model/sea_cells.py). The table is
written by

    python -m model.site_geography protected-areas  # inputs/site_costs/protected_areas.csv
"""
from __future__ import annotations

import logging
import time
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

from .config import EARTH_RADIUS_KM
from .land_table import MODIS_WATER_CLASS, _format_eta, _half_cell
from .parameters import parameter

LOGGER = logging.getLogger(__name__)

PROTECTED_AREA_STATUSES = frozenset(str(status).lower() for status in parameter("land.protected_area_statuses"))
LAND_REALMS = frozenset(str(realm).lower() for realm in parameter("land.protected_area_realms"))
WATER_REALMS = frozenset(str(realm).lower() for realm in parameter("offshore.protected_area_realms"))
# The realm codes of WDPA releases before the REALM field (MARINE: 0 terrestrial, 1 coastal, 2 marine)
LEGACY_MARINE_REALMS = {"0": "terrestrial", "false": "terrestrial", "1": "coastal", "2": "marine", "true": "marine"}
# Sub-pixels per land-cover pixel side (0.005 degrees for MODIS's 0.05)
PROTECTED_SUBPIXELS = 10
# Largest number of sub-pixel centres tested against one polygon at once
_POINT_BLOCK = 4_000_000
KEYS = ["latitude", "longitude"]
COLUMNS = [*KEYS, "land_km2", "protected_land_km2", "protected_land_pct",
           "water_km2", "protected_water_km2", "protected_water_pct"]


def _valid_geometries(frame: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    frame = frame[frame.geometry.notna() & ~frame.geometry.is_empty].copy()
    if frame.empty:
        return frame
    try:
        frame = frame.set_geometry(frame.geometry.make_valid())
    except AttributeError:  # pragma: no cover - depends on shapely/geopandas version
        frame = frame.set_geometry(frame.geometry.buffer(0))
    return frame[frame.geometry.notna() & ~frame.geometry.is_empty]


def protected_polygons(frame: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """The polygons of frame that count on land or on water, with their realm in lower case
    (realm; empty in a file without realms, whose polygons count on both)."""
    output = frame.copy()
    if "STATUS" in output.columns:
        output = output[output["STATUS"].astype(str).str.lower().str.strip().isin(PROTECTED_AREA_STATUSES)]
    if "REALM" in output.columns:
        output["realm"] = output["REALM"].astype(str).str.lower().str.strip()
    elif "MARINE" in output.columns:
        marine = output["MARINE"].astype(str).str.lower().str.strip()
        output["realm"] = marine.map(LEGACY_MARINE_REALMS).fillna(marine)
    else:
        output["realm"] = ""
    output = output[output["realm"].isin(LAND_REALMS | WATER_REALMS | {""})]
    return _valid_geometries(output)


def _read_polygons(path: Path, bbox=None) -> gpd.GeoDataFrame:
    """The polygons of one WDPA file that count (protected_polygons), in degrees."""
    polygons = gpd.read_file(path, bbox=bbox)
    if polygons.empty:
        return polygons.assign(realm=pd.Series(dtype=str))
    polygons = polygons.set_crs("EPSG:4326") if polygons.crs is None else polygons.to_crs("EPSG:4326")
    return protected_polygons(polygons)


def _rasterise_polygons(geometries, north: float, n_rows: int, n_cols: int, step: float) -> np.ndarray:
    """Boolean grid (north-up, from -180 degrees) of the sub-pixels whose centre lies in a polygon."""
    import shapely

    mask = np.zeros((n_rows, n_cols), dtype=bool)
    south = north - n_rows * step
    for geometry in geometries:
        min_x, min_y, max_x, max_y = geometry.bounds
        min_y, max_y = max(min_y, south), min(max_y, north)
        if max_y <= min_y:
            continue
        first_col = max(int(np.ceil((min_x + 180.0) / step - 0.5)), 0)
        last_col = min(int(np.floor((max_x + 180.0) / step - 0.5)), n_cols - 1)
        first_row = max(int(np.ceil((north - max_y) / step - 0.5)), 0)
        last_row = min(int(np.floor((north - min_y) / step - 0.5)), n_rows - 1)
        if last_col < first_col or last_row < first_row:
            continue
        shapely.prepare(geometry)
        xs = -180.0 + step * (np.arange(first_col, last_col + 1) + 0.5)
        rows_per_block = max(1, _POINT_BLOCK // len(xs))
        for block_start in range(first_row, last_row + 1, rows_per_block):
            block_end = min(block_start + rows_per_block, last_row + 1)
            ys = north - step * (np.arange(block_start, block_end) + 0.5)
            grid_x, grid_y = np.meshgrid(xs, ys)
            inside = shapely.contains_xy(geometry, grid_x, grid_y)
            mask[block_start:block_end, first_col:last_col + 1] |= inside
    return mask


def protected_area_table(
    cells: pd.DataFrame,
    protected_area_paths: tuple[Path, ...],
    land_cover_path: Path,
    coarse_degree: float = 1.0,
    fine_degree: float = 0.05,
) -> pd.DataFrame:
    """The table of the module docstring for the cells of cells (latitude, longitude).

    The cells are measured a band of latitude at a time. Each WDPA file is read for the band, and
    the polygons of each realm are rasterised once: a pixel's protected share on land is the share
    of its sub-pixels inside a polygon of LAND_REALMS, and on water inside one of WATER_REALMS.
    """
    from pyhdf.SD import SD, SDC

    for path in protected_area_paths:
        if not Path(path).exists():
            raise FileNotFoundError(f"Protected-area vector file not found: {path}")
    group = int(round(coarse_degree / fine_degree))
    half = group // 2
    step = fine_degree / PROTECTED_SUBPIXELS
    n_cols = int(round(360.0 / fine_degree))
    labels = np.sort(cells["latitude"].astype(float).unique())
    chunk_degree = max(coarse_degree, 5.0)
    chunks = [labels[(labels >= edge) & (labels < edge + chunk_degree)]
              for edge in np.arange(labels.min(), labels.max() + chunk_degree, chunk_degree)]
    chunks = [chunk for chunk in chunks if len(chunk)]
    LOGGER.info("Protected areas: %d latitude chunks across %d WDPA files.", len(chunks), len(protected_area_paths))
    started = time.perf_counter()
    frames = []
    sd = SD(str(land_cover_path), SDC.READ)
    try:
        sds = sd.select("Land_Cover_Type_1_Percent")
        for index, chunk in enumerate(chunks, start=1):
            chunk_started = time.perf_counter()
            # the chunk's cells as a contiguous band of whole cells
            cell_rows = np.arange(chunk.max(), chunk.min() - coarse_degree / 2, -coarse_degree)
            north = float(chunk.max()) + _half_cell(coarse_degree)
            south = float(chunk.min()) - _half_cell(coarse_degree)
            first_row = int(round((90.0 - north) / fine_degree))
            last_row = int(round((90.0 - south) / fine_degree))
            n_rows = last_row - first_row
            water = np.asarray(sds[first_row:last_row, 0:n_cols, MODIS_WATER_CLASS:MODIS_WATER_CLASS + 1],
                               dtype=float)[:, :, 0] / 100.0
            row_north = north - fine_degree * np.arange(n_rows)
            pixel_km2 = (EARTH_RADIUS_KM ** 2) * np.deg2rad(fine_degree) * (
                np.sin(np.deg2rad(row_north)) - np.sin(np.deg2rad(row_north - fine_degree)))
            land_km2 = (1.0 - water).clip(0.0, 1.0) * pixel_km2[:, None]
            water_km2 = water.clip(0.0, 1.0) * pixel_km2[:, None]

            polygons = pd.concat([_read_polygons(path, bbox=(-180.0, south, 180.0, north)) for path in protected_area_paths],
                                 ignore_index=True)
            masks = {realm: _rasterise_polygons(realm_polygons.geometry, north, n_rows * PROTECTED_SUBPIXELS,
                                                n_cols * PROTECTED_SUBPIXELS, step)
                     for realm, realm_polygons in polygons.groupby("realm")}

            def protected_share(realms: frozenset) -> np.ndarray:
                mask = np.zeros((n_rows * PROTECTED_SUBPIXELS, n_cols * PROTECTED_SUBPIXELS), dtype=bool)
                for realm, realm_mask in masks.items():
                    if realm in realms or realm == "":
                        mask |= realm_mask
                return mask.reshape(n_rows, PROTECTED_SUBPIXELS, n_cols, PROTECTED_SUBPIXELS).mean(axis=(1, 3))

            pixels = {"land_km2": land_km2, "protected_land_km2": protected_share(LAND_REALMS) * land_km2,
                      "water_km2": water_km2, "protected_water_km2": protected_share(WATER_REALMS) * water_km2}
            # coarse column b covers [b - 180.5, b - 179.5): roll by half a cell, as the land cover does
            per_cell = {name: np.roll(values, half, axis=1).reshape(len(cell_rows), group, -1, group).sum(axis=(1, 3))
                        for name, values in pixels.items()}
            lon_labels = -180.0 + coarse_degree * np.arange(n_cols // group)
            for row, lat in enumerate(cell_rows):
                frames.append(pd.DataFrame({"latitude": float(lat), "longitude": lon_labels,
                                            **{name: values[row] for name, values in per_cell.items()}}))
            elapsed = time.perf_counter() - started
            LOGGER.info(
                "Protected-area chunk %d/%d lat [%.1f, %.1f): %d polygons, chunk=%s, elapsed=%s, eta=%s",
                index, len(chunks), south, north, len(polygons), _format_eta(time.perf_counter() - chunk_started),
                _format_eta(elapsed), _format_eta(elapsed / index * (len(chunks) - index)),
            )
    finally:
        sd.end()

    table = pd.concat(frames, ignore_index=True)
    for surface in ("land", "water"):
        area = table[f"{surface}_km2"].where(table[f"{surface}_km2"] > 0)
        table[f"protected_{surface}_pct"] = (100.0 * table[f"protected_{surface}_km2"] / area).fillna(0.0).clip(0.0, 100.0)
    return cells[KEYS].merge(table, on=KEYS, how="left")[COLUMNS]
