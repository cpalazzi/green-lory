"""Each 1-degree cell's country: one assignment for finance tables, results and supplier tables.

A cell's country is the one covering the largest share of its box in data/countries.geojson
(Natural Earth admin-0 polygons). Cells that no polygon touches (small islands missing from the
map, and the sea) take the country of the nearest cell that has one, with the distance between
the cell centres. ISO 3166 alpha-3 codes come from inputs/cells/country_reference_20260928_v1.csv.

The table covers the cells of the land tables (latitudes 75 S to 75 N) and is built once, with a
provenance file beside it:

    python -m model.countries          # inputs/cells/cell_countries.csv

Results report the country covering the cell (empty where none does); a finance table keyed by
country reaches every cell through this table, the nearest country included.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from functools import lru_cache
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from .config import CELLS_DIR, COUNTRIES_GEOJSON, repo_path

CELL_COUNTRIES_CSV = CELLS_DIR / "cell_countries.csv"
COUNTRY_REFERENCE_CSV = CELLS_DIR / "country_reference_20260928_v1.csv"
# The rows of the land tables (model/land_table.py, lat_bounds)
CELL_LATITUDES = range(-75, 76)
COUNTRY_ASSIGNMENT_AREA_CRS = "EPSG:6933"


def load_countries(path: str | Path = COUNTRIES_GEOJSON):
    """Country polygons with a 'country' column (Natural Earth admin-0 style GeoJSON)."""
    import geopandas as gpd

    path = repo_path(path)
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}: add a Natural Earth admin-0 GeoJSON to tag cells by country.")
    frame = gpd.read_file(path)
    for candidate in ("ADMIN", "NAME", "name", "COUNTRY", "Country"):
        if candidate in frame.columns:
            frame = frame.rename(columns={candidate: "country"})
            break
    if "country" not in frame.columns:
        raise ValueError(f"{path} needs a 'country'/'ADMIN' column and valid geometries.")
    return frame[["country", "geometry"]]


def country_lookup(world, locations: Sequence[tuple[float, float]], cell_size_deg: float = 1.0) -> dict[tuple[float, float], str]:
    """Each cell's country: the one covering the largest share of the cell's 1-degree box."""
    import geopandas as gpd
    from shapely.errors import GEOSException
    from shapely.geometry import box

    lookup = {(float(lat), float(lon)): "" for lat, lon in locations}
    if not locations or world.empty:
        return lookup
    countries = world[["country", "geometry"]].dropna(subset=["geometry"]).copy()
    if countries.empty:
        return lookup
    if countries.crs is None:
        countries = countries.set_crs("EPSG:4326")
    # Repair invalid geometries, which can otherwise crash GEOS overlay operations
    if not countries.geometry.is_valid.all():
        countries["geometry"] = countries.geometry.buffer(0)
        countries = countries.dropna(subset=["geometry"])
        countries = countries[countries.geometry.is_valid].copy()
        if countries.empty:
            return lookup

    half = float(cell_size_deg) / 2.0
    points = [(float(lat), float(lon)) for lat, lon in locations]
    cells = gpd.GeoDataFrame(
        {"latitude": [lat for lat, _ in points], "longitude": [lon for _, lon in points]},
        geometry=[box(lon - half, lat - half, lon + half, lat + half) for lat, lon in points],
        crs=countries.crs,
    )
    candidates = gpd.sjoin(cells, countries, how="left", predicate="intersects")
    candidates = candidates.dropna(subset=["index_right"]).reset_index().rename(columns={"index": "cell_idx"})
    if candidates.empty:
        return lookup
    countries_equal_area = countries.to_crs(COUNTRY_ASSIGNMENT_AREA_CRS).reset_index(drop=True)
    candidates_equal_area = gpd.GeoDataFrame(
        candidates[["cell_idx", "latitude", "longitude", "country", "index_right"]].copy(),
        geometry=candidates.geometry,
        crs=cells.crs,
    ).to_crs(COUNTRY_ASSIGNMENT_AREA_CRS)
    cell_geometries = gpd.GeoSeries(candidates_equal_area.geometry.reset_index(drop=True), crs=COUNTRY_ASSIGNMENT_AREA_CRS)
    country_geometries = gpd.GeoSeries(
        countries_equal_area.geometry.iloc[candidates_equal_area["index_right"].astype(int).to_numpy()].reset_index(drop=True),
        crs=COUNTRY_ASSIGNMENT_AREA_CRS,
    )
    try:
        overlap_area = cell_geometries.intersection(country_geometries).area
    except GEOSException:
        overlap_area = cell_geometries.buffer(0).intersection(country_geometries.buffer(0)).area
    candidates_equal_area = candidates_equal_area.reset_index(drop=True)
    candidates_equal_area["overlap_area"] = overlap_area.to_numpy()
    candidates_equal_area = candidates_equal_area[candidates_equal_area["overlap_area"] > 0]
    best = candidates_equal_area.sort_values(["cell_idx", "overlap_area"], ascending=[True, False])
    best = best.drop_duplicates(subset=["cell_idx"], keep="first")
    for row in best.itertuples(index=False):
        lookup[(float(row.latitude), float(row.longitude))] = str(row.country) if pd.notna(row.country) else ""
    return lookup


def build_cell_countries(latitudes=CELL_LATITUDES, countries_path: Path = COUNTRIES_GEOJSON,
                         reference_csv: Path = COUNTRY_REFERENCE_CSV) -> pd.DataFrame:
    """The table of the module docstring: latitude, longitude, iso3, country, assignment, distance_km."""
    from .site_geography import nearest_km

    cells = [(float(lat), float(lon)) for lat in latitudes for lon in range(-180, 180)]
    lookup = country_lookup(load_countries(countries_path), cells)
    reference = pd.read_csv(repo_path(reference_csv))
    iso3 = dict(zip(reference["country"], reference["iso3"]))
    named = sorted({name for name in lookup.values() if name})
    unknown = [name for name in named if name not in iso3]
    if unknown:
        raise ValueError(f"{reference_csv} has no ISO 3166 code for {unknown}")
    table = pd.DataFrame({"latitude": [lat for lat, _ in cells], "longitude": [lon for _, lon in cells],
                          "country": [lookup[cell] for cell in cells]})
    covered = table["country"] != ""
    table["assignment"] = np.where(covered, "overlap", "nearest")
    table["distance_km"] = 0.0
    source = table[covered]
    distance, index = nearest_km(table.loc[~covered, "latitude"], table.loc[~covered, "longitude"],
                                 source["latitude"], source["longitude"])
    table.loc[~covered, "country"] = source["country"].to_numpy()[index]
    table.loc[~covered, "distance_km"] = np.round(distance, 1)
    table["iso3"] = table["country"].map(iso3)
    return table[["latitude", "longitude", "iso3", "country", "assignment", "distance_km"]]


@lru_cache(maxsize=None)
def cell_countries(path: Path = CELL_COUNTRIES_CSV) -> pd.DataFrame:
    """The cell-country table, indexed by (latitude, longitude)."""
    return pd.read_csv(repo_path(path)).set_index(["latitude", "longitude"])


def _sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", type=Path, default=CELL_COUNTRIES_CSV)
    args = parser.parse_args(argv)
    table = build_cell_countries()
    output = repo_path(args.output)
    table.to_csv(output, index=False)
    provenance = {
        "written_by": "python -m model.countries",
        "rows": int(len(table)),
        "sha256": _sha256(output),
        "inputs": {name: {"path": str(path), "sha256": _sha256(repo_path(path))}
                   for name, path in (("countries", COUNTRIES_GEOJSON), ("country_reference", COUNTRY_REFERENCE_CSV))},
        "cells_by_assignment": table["assignment"].value_counts().to_dict(),
    }
    output.with_name(output.stem + "_provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(f"wrote {output}: {len(table)} cells, {provenance['cells_by_assignment']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
