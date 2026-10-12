"""Finance inputs: the per-cell interest-rate (WACC) tables that runs pin as ``finance_csv``.

A finance table has one row per cell and technology, ``lat, lon, tech, interest_rate`` (a
fraction), for the 15 technologies of the plant. ``model.technology`` reads it as the per-cell
override table, and runs.yaml pins each file's sha256.

- ``uniform_table``: one rate everywhere (the flat-finance runs, 5 %);
- ``ameli_rates``: the WACC of Ameli et al. (2021, Fig. 2; ``inputs/finance/ameli_tiam_wacc_fig2.csv``)
  by TIAM-UCL region, each cell's region taken from its country;
- ``fill_from_nearest``: cells that a table does not cover take the rates of the nearest
  covered cell (great-circle distance), and the fills are listed.

    python -m model.finance uniform --land-csv <land table> --rate 0.05 --output-dir inputs/finance
    python -m model.finance ameli --land-csv <land table> --scenario low-carbon-regional --output-dir inputs/finance
    python -m model.finance fill --land-csv <land table> --table <finance csv> --output-dir inputs/finance

Every file is written deterministically: identical inputs give identical bytes.
"""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

from .config import COUNTRIES_GEOJSON, EARTH_RADIUS_KM, FINANCE_DIR

AMELI_WACC_TABLE = FINANCE_DIR / "ameli_tiam_wacc_fig2.csv"
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

# TIAM-UCL regions of Ameli et al. (2021, Table 1), with the country names of
# data/countries.geojson; territories are placed with the country that governs them.
AFRICA = {
    "Algeria", "Angola", "Benin", "Botswana", "Burkina Faso", "Burundi",
    "Cameroon", "Central African Republic", "Chad", "Democratic Republic of the Congo",
    "Djibouti", "Egypt", "Equatorial Guinea", "Eritrea", "Ethiopia", "Gabon",
    "Gambia", "Ghana", "Guinea", "Guinea-Bissau", "Ivory Coast", "Kenya",
    "Lesotho", "Liberia", "Libya", "Madagascar", "Malawi", "Mali", "Mauritania",
    "Morocco", "Mozambique", "Namibia", "Niger", "Nigeria", "Republic of the Congo",
    "Rwanda", "Senegal", "Sierra Leone", "Somalia", "Somaliland", "South Africa",
    "South Sudan", "Sudan", "Togo", "Tunisia", "Uganda",
    "United Republic of Tanzania", "Western Sahara", "Zambia", "Zimbabwe", "eSwatini",
}
CENTRAL_SOUTH_AMERICA = {
    "Argentina", "Belize", "Bolivia", "Brazil", "Chile", "Colombia", "Costa Rica",
    "Cuba", "Dominican Republic", "Ecuador", "El Salvador", "Falkland Islands",
    "Guatemala", "Guyana", "Haiti", "Honduras", "Jamaica", "Nicaragua", "Panama",
    "Paraguay", "Peru", "Puerto Rico", "Suriname", "The Bahamas",
    "Trinidad and Tobago", "Uruguay", "Venezuela",
}
WESTERN_EUROPE = {
    "Austria", "Belgium", "Denmark", "Finland", "France", "Germany", "Greece",
    "Iceland", "Ireland", "Italy", "Luxembourg", "Malta", "Netherlands",
    "New Caledonia", "Norway", "Portugal", "Spain", "Sweden", "Switzerland",
    "French Southern and Antarctic Lands",
}
EASTERN_EUROPE = {
    "Albania", "Bosnia and Herzegovina", "Bulgaria", "Croatia", "Cyprus", "Czechia",
    "Hungary", "Kosovo", "Montenegro", "North Macedonia", "Northern Cyprus",
    "Poland", "Republic of Serbia", "Romania", "Slovakia", "Slovenia",
}
FORMER_SOVIET_UNION = {
    "Armenia", "Azerbaijan", "Belarus", "Estonia", "Georgia", "Kazakhstan",
    "Kyrgyzstan", "Latvia", "Lithuania", "Moldova", "Russia", "Tajikistan",
    "Turkmenistan", "Ukraine", "Uzbekistan",
}
MIDDLE_EAST = {
    "Iran", "Iraq", "Israel", "Jordan", "Kuwait", "Lebanon", "Oman", "Palestine",
    "Qatar", "Saudi Arabia", "Syria", "Turkey", "United Arab Emirates", "Yemen",
}
OTHER_DEVELOPING_ASIA = {
    "Afghanistan", "Bangladesh", "Bhutan", "Brunei", "Cambodia", "East Timor",
    "Indonesia", "Laos", "Malaysia", "Myanmar", "Nepal", "North Korea", "Pakistan",
    "Papua New Guinea", "Philippines", "Solomon Islands", "Sri Lanka", "Taiwan",
    "Thailand", "Vanuatu", "Vietnam",
}
AUSTRALIA_REGION = {"Australia", "Fiji", "New Zealand"}
UNASSIGNED = "UNASSIGNED"


def tiam_region(country: str | float | None) -> str:
    """The TIAM-UCL region code of a country name (UNASSIGNED for none, Antarctica, or unknown)."""
    if country is None or pd.isna(country) or country == "":
        return UNASSIGNED
    country = str(country)
    single = {"Mexico": "MEX", "India": "IND", "South Korea": "SKO", "China": "CHI", "Mongolia": "CHI",
              "Canada": "CAN", "Greenland": "CAN", "United States of America": "USA", "United Kingdom": "UK",
              "Japan": "JAP"}
    if country in single:
        return single[country]
    for members, code in ((AFRICA, "AFR"), (CENTRAL_SOUTH_AMERICA, "CSA"), (MIDDLE_EAST, "MEA"),
                          (OTHER_DEVELOPING_ASIA, "ODA"), (AUSTRALIA_REGION, "AUS"), (EASTERN_EUROPE, "EEU"),
                          (FORMER_SOVIET_UNION, "FSU"), (WESTERN_EUROPE, "WEU")):
        if country in members:
            return code
    return UNASSIGNED


def land_cells(land_csv: str | Path) -> pd.DataFrame:
    """The cells (lat, lon) of a land table, sorted."""
    cells = pd.read_csv(land_csv, usecols=["latitude", "longitude"]).drop_duplicates().sort_values(["latitude", "longitude"])
    return cells.rename(columns={"latitude": "lat", "longitude": "lon"}).reset_index(drop=True)


def uniform_table(cells: pd.DataFrame, rate: float, techs: tuple[str, ...] = TECHNOLOGIES) -> pd.DataFrame:
    """One interest rate for every technology at every cell."""
    grid = cells[["lat", "lon"]].merge(pd.DataFrame({"tech": sorted(techs)}), how="cross")
    grid["interest_rate"] = rate
    return grid.sort_values(["lat", "lon", "tech"])


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
    uniform = commands.add_parser("uniform", help="one rate everywhere")
    uniform.add_argument("--rate", type=float, default=0.05)
    ameli = commands.add_parser("ameli", help="Ameli WACC by TIAM region; cells without a country take the nearest cell's rate")
    ameli.add_argument("--scenario", choices=sorted(SCENARIO_COLUMN), required=True)
    ameli.add_argument("--countries", type=Path, default=COUNTRIES_GEOJSON)
    fill = commands.add_parser("fill", help="an existing finance table extended to every cell by nearest-cell fill")
    fill.add_argument("--table", type=Path, required=True)
    fill.add_argument("--name", default="amelired_interest_inputs_2050_center", help="output file name without .csv")
    for command in (uniform, ameli, fill):
        command.add_argument("--land-csv", type=Path, required=True, help="land table whose latitude/longitude labels define the cells")
        command.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)

    cells = land_cells(args.land_csv)
    if args.command == "uniform":
        tag = f"{args.rate:g}".replace(".", "p")
        _write(uniform_table(cells, args.rate), args.output_dir / f"uniform_interest_inputs_{tag}_2050.csv")
    elif args.command == "ameli":
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
