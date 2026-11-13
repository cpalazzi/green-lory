"""Each 1-degree cell's country, and each country's region and rate in the finance tables.

Countries. A cell's country is the one covering the largest share of its box in Natural Earth's
1:10m map of countries and territories (data/ne_10m_admin_0_countries/, version 5.1.1), which has a
polygon for every territory and small island. A territory keeps its own ISO 3166 code (Greenland,
French Polynesia), which the network model matches to its ports. An area without a code of its own
takes the code of the country that ISO 3166 counts it in (Natural Earth's ADM0_ISO: Somaliland
Somalia, Northern Cyprus Cyprus, Kosovo Serbia), and areas without any ISO code (Siachen Glacier,
Bir Tawil, a few reefs) are left out, so that their land counts for the neighbouring countries.
Cells that no country touches (the sea of the land tables' rows) take the country of the nearest
cell that has one, with the distance between the cell centres.

Regions. Ameli et al. (2021) give one cost of capital for each of the 16 regions of the TIAM-UCL
model, and list the countries of each region (Supplementary Table 4,
inputs/finance/ameli2021_tiam_ucl_regions.csv). Each country or territory takes the region of its
sovereign country (Natural Earth's SOVEREIGNT), as a plant at sea takes the rate of its zone's
sovereign. A sovereign that the list does not name takes the region of the list's catch-all for its
continent (Natural Earth's REGION_UN; CONTINENT_REGIONS): Other Africa, Other Latin America for the
Americas, and Other Asia, which also takes the Pacific islands, since the list's Australia region
holds only Australia and New Zealand; the list has no catch-all for Europe, whose unnamed
micro-states take Western Europe. Antarctica takes the highest rate of each series, as international
waters do (offshore in inputs/parameters.yaml).

The tables cover the cells of the land tables (latitudes 75 S to 75 N) and are built once, with
provenance files beside them:

    python -m model.countries

writes inputs/cells/cell_countries.csv, the country reference of the supplier export
(COUNTRY_REFERENCE_CSV) and inputs/finance/ameli2021_{lowcarbon,reduced}_wacc_by_country.csv, from
the regional rates of inputs/finance/ameli2021_wacc_by_region.csv. Results report the country
covering the cell (empty where none does); a finance table keyed by country reaches every cell
through the cell table, the nearest country included.
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

from .config import CELLS_DIR, COUNTRIES_10M_SHAPEFILE, COUNTRIES_GEOJSON, FINANCE_DIR, repo_path

CELL_COUNTRIES_CSV = CELLS_DIR / "cell_countries.csv"
# The names of the cell table and their codes, for the supplier export of runs whose result tables
# take their country names from it (model/suppliers.py --country-reference)
COUNTRY_REFERENCE_CSV = CELLS_DIR / "country_reference_20261113_v1.csv"
REGION_LIST_CSV = FINANCE_DIR / "ameli2021_tiam_ucl_regions.csv"
REGION_RATES_CSV = FINANCE_DIR / "ameli2021_wacc_by_region.csv"
# The finance tables by country, one for each series of regional rates (columns of REGION_RATES_CSV)
FINANCE_TABLES = {series: FINANCE_DIR / f"ameli2021_{series}_wacc_by_country.csv" for series in ("lowcarbon", "reduced")}
# The region of the list's catch-all for a continent (Natural Earth's REGION_UN), for sovereign
# countries the list does not name (module docstring)
CONTINENT_REGIONS = {"Africa": "AFR", "Americas": "CSA", "Asia": "ODA", "Oceania": "ODA", "Europe": "WEU"}
ANTARCTICA = "Antarctica"
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


def load_units(path: str | Path = COUNTRIES_10M_SHAPEFILE):
    """The countries and territories of Natural Earth's admin-0 map, one row per map unit: its code
    (iso3) and name (country) under the rules of the module docstring, its sovereign's code
    (sovereign_iso3) and the sovereign's continent (continent). Units without any ISO code are
    left out."""
    import geopandas as gpd

    path = repo_path(path)
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}: download Natural Earth's 1:10m admin-0 countries (data/README.md).")
    frame = gpd.read_file(path)[["ADMIN", "SOVEREIGNT", "ISO_A3_EH", "ADM0_ISO", "REGION_UN", "geometry"]]
    own = frame["ISO_A3_EH"].where(frame["ISO_A3_EH"] != "-99")
    codes = set(own.dropna())
    # an area without a code of its own takes the code that ISO 3166 counts it in, else, if it is
    # a dependency (Christmas Island, Clipperton), its sovereign's
    sovereign_code = frame["SOVEREIGNT"].map(frame[frame["ADMIN"] == frame["SOVEREIGNT"]].drop_duplicates("ADMIN")
                                             .set_index("ADMIN")["ISO_A3_EH"])
    frame["iso3"] = (own.fillna(frame["ADM0_ISO"].where(frame["ADM0_ISO"].isin(codes)))
                     .fillna(sovereign_code.where(sovereign_code.isin(codes))))
    frame = frame.dropna(subset=["iso3"]).reset_index(drop=True)
    home = frame[frame["ADMIN"] == frame["SOVEREIGNT"]].drop_duplicates("ADMIN").set_index("ADMIN")
    named_by_code = frame[frame["ISO_A3_EH"] == frame["iso3"]].drop_duplicates("iso3").set_index("iso3")
    frame["country"] = frame["iso3"].map(named_by_code["ADMIN"])
    # the sovereign of a unit: SOVEREIGNT's own unit, or, for an area counted in another country,
    # that country's sovereign
    sovereign_name = frame["SOVEREIGNT"].where(frame["ISO_A3_EH"] == frame["iso3"],
                                               frame["iso3"].map(named_by_code["SOVEREIGNT"]))
    frame["sovereign_iso3"] = sovereign_name.map(home["iso3"]).fillna(frame["iso3"])
    frame["continent"] = frame["sovereign_iso3"].map(named_by_code["REGION_UN"]).fillna(frame["REGION_UN"])
    return frame[["iso3", "country", "sovereign_iso3", "continent", "geometry"]]


def country_regions(units: pd.DataFrame, region_list: pd.DataFrame | None = None) -> pd.DataFrame:
    """Each code's name, sovereign, region of the finance tables (ameli_region; empty for
    Antarctica, which takes the highest rate) and the basis of that region (region_basis)."""
    if region_list is None:
        region_list = pd.read_csv(repo_path(REGION_LIST_CSV), keep_default_na=False)
    listed = region_list[region_list["iso3"] != ""].set_index("iso3")["region"]
    if listed.index.duplicated().any():
        raise ValueError(f"{REGION_LIST_CSV} lists {sorted(listed.index[listed.index.duplicated()])} more than once")
    table = units[["iso3", "country", "sovereign_iso3", "continent"]].drop_duplicates("iso3").reset_index(drop=True)
    in_list = table["sovereign_iso3"].isin(listed.index)
    table["ameli_region"] = table["sovereign_iso3"].map(listed).fillna(table["continent"].map(CONTINENT_REGIONS))
    table["region_basis"] = np.where(in_list, "sovereign listed", "continent " + table["continent"].astype(str))
    unknown = table[table["ameli_region"].isna() & (table["continent"] != ANTARCTICA)]
    if len(unknown):
        raise ValueError(f"no region for {unknown[['iso3', 'continent']].to_dict('records')}: add the continent to CONTINENT_REGIONS")
    table.loc[table["ameli_region"].isna(), "region_basis"] = "Antarctica, the highest rate"
    return table.sort_values("iso3").reset_index(drop=True)


def finance_tables(regions: pd.DataFrame, rates: pd.DataFrame | None = None) -> dict[str, pd.DataFrame]:
    """One finance table by country for each series of regional rates: every technology at the
    rate of the country's region, and Antarctica at the highest rate of the series."""
    if rates is None:
        rates = pd.read_csv(repo_path(REGION_RATES_CSV))
    rates = rates.set_index("region")
    missing = sorted(set(regions["ameli_region"].dropna()) - set(rates.index))
    if missing:
        raise ValueError(f"{REGION_RATES_CSV} has no rates for the regions {missing}")
    tables = {}
    for series in FINANCE_TABLES:
        rate = regions["ameli_region"].map(rates[series]).fillna(float(rates[series].max()))
        tables[series] = pd.DataFrame({"iso3": regions["iso3"], "country": regions["country"], "tech": "all",
                                       "interest_rate": rate, "ameli_region": regions["ameli_region"].fillna(""),
                                       "sovereign_iso3": regions["sovereign_iso3"], "region_basis": regions["region_basis"]})
    return tables


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


def build_cell_countries(units, latitudes=CELL_LATITUDES) -> pd.DataFrame:
    """The cell table of the module docstring: latitude, longitude, iso3, country, assignment
    (overlap or nearest) and distance_km (to the nearest cell's centre)."""
    from .site_geography import nearest_km

    # one geometry per code, so that the areas counted in one country add up
    world = units.dissolve(by="iso3").reset_index()[["iso3", "geometry"]].rename(columns={"iso3": "country"})
    cells = [(float(lat), float(lon)) for lat in latitudes for lon in range(-180, 180)]
    lookup = country_lookup(world, cells)
    table = pd.DataFrame({"latitude": [lat for lat, _ in cells], "longitude": [lon for _, lon in cells],
                          "iso3": [lookup[cell] for cell in cells]})
    covered = table["iso3"] != ""
    table["assignment"] = np.where(covered, "overlap", "nearest")
    table["distance_km"] = 0.0
    source = table[covered]
    distance, index = nearest_km(table.loc[~covered, "latitude"], table.loc[~covered, "longitude"],
                                 source["latitude"], source["longitude"])
    table.loc[~covered, "iso3"] = source["iso3"].to_numpy()[index]
    table.loc[~covered, "distance_km"] = np.round(distance, 1)
    table["country"] = table["iso3"].map(units.drop_duplicates("iso3").set_index("iso3")["country"])
    return table[["latitude", "longitude", "iso3", "country", "assignment", "distance_km"]]


@lru_cache(maxsize=None)
def cell_countries(path: Path = CELL_COUNTRIES_CSV) -> pd.DataFrame:
    """The cell-country table, indexed by (latitude, longitude)."""
    return pd.read_csv(repo_path(path)).set_index(["latitude", "longitude"])


def _sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _write(frame: pd.DataFrame, path: Path, inputs: dict[str, Path], **extra) -> None:
    output = repo_path(path)
    frame.to_csv(output, index=False)
    provenance = {"written_by": "python -m model.countries", "rows": int(len(frame)), "sha256": _sha256(output),
                  "inputs": {name: {"path": str(Path(source).relative_to(repo_path("."))), "sha256": _sha256(repo_path(source))}
                             for name, source in inputs.items()}, **extra}
    output.with_name(output.stem + "_provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(f"wrote {output.relative_to(repo_path('.'))}: {len(frame):,} rows")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.parse_args(argv)
    units = load_units()
    regions = country_regions(units)
    cells = build_cell_countries(units)
    sources = {"countries": repo_path(COUNTRIES_10M_SHAPEFILE), "countries_attributes": repo_path(COUNTRIES_10M_SHAPEFILE).with_suffix(".dbf")}
    _write(cells, CELL_COUNTRIES_CSV, sources, cells_by_assignment=cells["assignment"].value_counts().to_dict())
    _write(regions[["country", "iso3"]], COUNTRY_REFERENCE_CSV, sources)
    region_sources = {**sources, "region_list": repo_path(REGION_LIST_CSV), "region_rates": repo_path(REGION_RATES_CSV)}
    for series, table in finance_tables(regions).items():
        _write(table, FINANCE_TABLES[series], region_sources, series=series,
               codes_by_region_basis=table["region_basis"].value_counts().to_dict())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
