"""Finance inputs: the per-cell interest-rate (WACC) tables that runs pin as ``finance_csv``.

A run with one interest rate everywhere sets ``interest_rate`` in runs.yaml and needs no table.
A finance table has one row per cell and technology, ``lat, lon, tech, interest_rate`` (a
fraction), for the 15 technologies of the plant. ``model.technology`` reads it as the per-cell
override table, and runs.yaml pins each file's sha256. The tables are generated (gitignored).

- ``ameli_rates``: the WACC of Ameli et al. (2021, Fig. 2; ``inputs/finance/ameli_tiam_wacc_fig2.csv``)
  by TIAM-UCL region, each cell's region taken from its country (``inputs/finance/tiam_regions.csv``);
- ``fill_from_nearest``: cells that a table does not cover take the rates of the nearest
  covered cell (great-circle distance), and the fills are listed.

    python -m model.finance ameli --land-csv <land table> --scenario low-carbon-regional --output-dir inputs/finance
    python -m model.finance fill --land-csv <land table> --table <finance csv> --output-dir inputs/finance

Every file is written deterministically: identical inputs give identical bytes.
"""
from __future__ import annotations

import argparse
import hashlib
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from .config import COUNTRIES_GEOJSON, EARTH_RADIUS_KM, FINANCE_DIR

AMELI_WACC_TABLE = FINANCE_DIR / "ameli_tiam_wacc_fig2.csv"
# The TIAM-UCL region of each country of data/countries.geojson (Ameli et al. 2021, Table 1)
TIAM_REGIONS_CSV = FINANCE_DIR / "tiam_regions.csv"
SCENARIO_COLUMN = {
    "reduced": "reduced_wacc_pct",
    "low-carbon-regional": "low_carbon_wacc_pct",
    "high-carbon-regional": "high_carbon_wacc_pct",
}
SCENARIO_TAG = {"reduced": "amelired", "low-carbon-regional": "ameli_lowcarbon", "high-carbon-regional": "ameli_highcarbon"}

# The technologies a finance table covers: every component of the plant bundle
TECHNOLOGIES = (
    "accumulated_penalty", "ammonia", "ammonia_synthesis", "battery_pcs_charge", "battery_pcs_discharge",
    "battery_storage", "compressed_hydrogen_store", "electrolysis", "hydrogen_compression",
    "hydrogen_from_storage", "hydrogen_fuel_cell", "penalty_link", "solar", "solar_tracking", "wind",
)

UNASSIGNED = "UNASSIGNED"


@lru_cache(maxsize=None)
def _tiam_regions(path: Path = TIAM_REGIONS_CSV) -> dict[str, str]:
    table = pd.read_csv(path)
    if table["country"].duplicated().any():
        raise ValueError(f"{path} lists a country twice")
    return dict(zip(table["country"], table["tiam_region_code"]))


def tiam_region(country: str | float | None) -> str:
    """The TIAM-UCL region code of a country name (UNASSIGNED for none, Antarctica, or unknown)."""
    if country is None or pd.isna(country) or country == "":
        return UNASSIGNED
    return _tiam_regions().get(str(country), UNASSIGNED)


def land_cells(land_csv: str | Path) -> pd.DataFrame:
    """The cells (lat, lon) of a land table, sorted."""
    cells = pd.read_csv(land_csv, usecols=["latitude", "longitude"]).drop_duplicates().sort_values(["latitude", "longitude"])
    return cells.rename(columns={"latitude": "lat", "longitude": "lon"}).reset_index(drop=True)


def ameli_rates(countries: pd.Series, scenario: str, wacc_table: str | Path = AMELI_WACC_TABLE) -> pd.DataFrame:
    """Each cell's TIAM region and Ameli WACC (a fraction; NaN where the region is unassigned).

    countries: the country name of each cell (index kept), for example from
    model.surface.country_lookup.
    """
    wacc = pd.read_csv(wacc_table).set_index("tiam_region_code")
    rates = wacc[SCENARIO_COLUMN[scenario]].astype(float) / 100.0
    regions = countries.map(tiam_region)
    return pd.DataFrame({"country": countries, "tiam_region_code": regions, "interest_rate": regions.map(rates)})


def fill_from_nearest(cells: pd.DataFrame, table: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """The finance table restricted to cells, with uncovered cells given the nearest covered cell's rates.

    Returns (table, fills); fills lists each filled cell, its source cell and the distance.
    """
    techs = sorted(table["tech"].unique())
    rates = table.pivot_table(index=["lat", "lon"], columns="tech", values="interest_rate", aggfunc="first")
    covered = rates.index.to_frame(index=False)
    key = cells.merge(covered.assign(_covered=1), on=["lat", "lon"], how="left")
    missing = key[key["_covered"].isna()][["lat", "lon"]].reset_index(drop=True)
    fill_rows = []
    if not missing.empty:
        rad = np.deg2rad
        c_lat, c_lon = rad(covered["lat"].to_numpy()), rad(covered["lon"].to_numpy())
        for lat, lon in zip(missing["lat"], missing["lon"]):
            d = np.arccos(np.clip(np.sin(rad(lat)) * np.sin(c_lat) + np.cos(rad(lat)) * np.cos(c_lat) * np.cos(rad(lon) - c_lon), -1, 1))
            j = int(np.argmin(d))
            src_lat, src_lon = float(covered["lat"].iloc[j]), float(covered["lon"].iloc[j])
            fill_rows.append({"lat": lat, "lon": lon, "source_lat": src_lat, "source_lon": src_lon, "distance_km": float(d[j] * EARTH_RADIUS_KM)})
    fills = pd.DataFrame(fill_rows, columns=["lat", "lon", "source_lat", "source_lon", "distance_km"])
    frames = [table[["lat", "lon", "tech", "interest_rate"]]]
    for _, f in fills.iterrows():
        src = rates.loc[(f.source_lat, f.source_lon)]
        frames.append(pd.DataFrame({"lat": f.lat, "lon": f.lon, "tech": techs, "interest_rate": [float(src[t]) for t in techs]}))
    filled = pd.concat(frames, ignore_index=True)
    filled = filled.merge(cells, on=["lat", "lon"], how="inner").sort_values(["lat", "lon", "tech"])
    return filled, fills.sort_values(["lat", "lon"])


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, float_format="%.6g")
    print(f"{_sha256(path)}  {path}  rows={len(frame)}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    ameli = commands.add_parser("ameli", help="Ameli WACC by TIAM region; cells without a country take the nearest cell's rate")
    ameli.add_argument("--scenario", choices=sorted(SCENARIO_COLUMN), required=True)
    ameli.add_argument("--countries", type=Path, default=COUNTRIES_GEOJSON)
    fill = commands.add_parser("fill", help="an existing finance table extended to every cell by nearest-cell fill")
    fill.add_argument("--table", type=Path, required=True)
    fill.add_argument("--name", default="amelired_interest_inputs_2050_center", help="output file name without .csv")
    for command in (ameli, fill):
        command.add_argument("--land-csv", type=Path, required=True, help="land table whose latitude/longitude labels define the cells")
        command.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)

    cells = land_cells(args.land_csv)
    if args.command == "ameli":
        from .surface import country_lookup, load_countries

        lookup = country_lookup(load_countries(args.countries), list(cells.itertuples(index=False, name=None)))
        countries = pd.Series([lookup[(float(lat), float(lon))] for lat, lon in zip(cells.lat, cells.lon)], index=cells.index)
        rates = ameli_rates(countries, args.scenario)
        assigned = cells[rates.interest_rate.notna()]
        table = (assigned.merge(pd.DataFrame({"tech": sorted(TECHNOLOGIES)}), how="cross")
                 .merge(cells.assign(interest_rate=rates.interest_rate), on=["lat", "lon"]))
        filled, fills = fill_from_nearest(cells, table)
        tag = SCENARIO_TAG[args.scenario]
        _write(filled, args.output_dir / f"{tag}_interest_inputs_2050_center.csv")
        _write(fills, args.output_dir / f"{tag}_interest_fill_2050_center.csv")
        _write(pd.concat([cells, rates], axis=1), args.output_dir / f"{tag}_country_regions_2050_center.csv")
    else:
        filled, fills = fill_from_nearest(cells, pd.read_csv(args.table))
        _write(filled, args.output_dir / f"{args.name}.csv")
        _write(fills, args.output_dir / f"{args.name.replace('_inputs_', '_fill_')}.csv")


if __name__ == "__main__":
    main()
