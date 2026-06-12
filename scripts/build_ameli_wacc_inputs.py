#!/usr/bin/env python3
"""Build Ameli spatial WACC override inputs.

This is intended to be run after notebooks/02_spatial_cost_inputs.ipynb.
It reads the notebook's per-location/per-tech spatial cost CSV, assigns each
1-degree cell to a TIAM-UCL WACC region, and writes:

1. a WACC-only override CSV for flat-cost + spatial-WACC runs; and
2. a combined CSV preserving existing spatial build/water costs while replacing
    interest_rate with the selected Ameli Fig. 2 WACC series.

The WACC values are transcribed from Ameli et al. (2021) Fig. 2. The paper's
reduced scenario gives most regions a 5.1% WACC by 2050; developed regions with
already lower values retain those lower values.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import geopandas as gpd
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_SPATIAL_INPUT = REPO_ROOT / "inputs" / "spatial_cost_inputs.csv"
DEFAULT_WACC_TABLE = REPO_ROOT / "inputs" / "ameli_tiam_wacc_fig2.csv"
DEFAULT_COUNTRIES = REPO_ROOT / "data" / "countries.geojson"
DEFAULT_LOCATIONS_INPUT = REPO_ROOT / "data" / "max_capacities_paper_2pct_slope15.csv"
SCENARIO_COLUMN = {
    "reduced": "reduced_wacc_pct",
    "low-carbon-regional": "low_carbon_wacc_pct",
    "high-carbon-regional": "high_carbon_wacc_pct",
}

DEFAULT_OUTPUT_TAG = {
    "reduced": "amelired",
    "low-carbon-regional": "ameli_lowcarbon",
    "high-carbon-regional": "ameli_highcarbon",
}

# Country names follow data/countries.geojson. The regional aggregation follows
# Ameli et al. Table 1, expanded pragmatically to countries/territories present
# in the local Natural Earth-derived country file.
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


def find_repo_root(start: Path) -> Path:
    for path in [start.resolve(), *start.resolve().parents]:
        if (path / "model").is_dir() and (path / "inputs").is_dir():
            return path
    raise RuntimeError("Could not locate green-lory repo root")


def classify_country(country: str | float | None) -> str:
    if country is None or pd.isna(country):
        return "UNASSIGNED"
    country = str(country)
    if country in AFRICA:
        return "AFR"
    if country == "Mexico":
        return "MEX"
    if country in CENTRAL_SOUTH_AMERICA:
        return "CSA"
    if country == "India":
        return "IND"
    if country in MIDDLE_EAST:
        return "MEA"
    if country in OTHER_DEVELOPING_ASIA:
        return "ODA"
    if country == "South Korea":
        return "SKO"
    if country in {"China", "Mongolia"}:
        return "CHI"
    if country in AUSTRALIA_REGION:
        return "AUS"
    if country in EASTERN_EUROPE:
        return "EEU"
    if country in FORMER_SOVIET_UNION:
        return "FSU"
    if country in {"Canada", "Greenland"}:
        return "CAN"
    if country == "United States of America":
        return "USA"
    if country == "United Kingdom":
        return "UK"
    if country in WESTERN_EUROPE:
        return "WEU"
    if country == "Japan":
        return "JAP"
    if country == "Antarctica":
        return "UNASSIGNED"
    return "UNASSIGNED"


def assign_countries(locations: pd.DataFrame, countries_path: Path) -> pd.DataFrame:
    world = gpd.read_file(countries_path)
    if "country" not in world.columns:
        raise ValueError(f"{countries_path} must contain a 'country' column")

    points = gpd.GeoDataFrame(
        locations.copy(),
        geometry=gpd.points_from_xy(locations["lon"], locations["lat"]),
        crs="EPSG:4326",
    )
    joined = gpd.sjoin(points, world[["country", "geometry"]], predicate="intersects", how="left")
    joined = joined[~joined.index.duplicated(keep="first")]

    out = locations.copy()
    out["country"] = joined.reindex(locations.index)["country"].to_numpy()
    out["tiam_region_code"] = out["country"].map(classify_country)
    out["mapping_note"] = out["tiam_region_code"].where(
        out["tiam_region_code"] != "UNASSIGNED",
        "unassigned_ocean_or_unmapped_country_defaulted_to_reduced_5p1pct",
    )
    return out


def _write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    print(f"Wrote {path.relative_to(REPO_ROOT)} ({len(df):,} rows, {path.stat().st_size / 1024 / 1024:.1f} MB)")


def _locations_from_csv(path: Path | None) -> pd.DataFrame:
    if path is None or not path.exists():
        return pd.DataFrame(columns=["lat", "lon"])
    df = pd.read_csv(path)
    if "max_capacity_mw" in df.columns:
        max_capacity = pd.to_numeric(df["max_capacity_mw"], errors="coerce").fillna(0.0)
        df = df.loc[max_capacity > 0].copy()
    cols = {col.lower(): col for col in df.columns}
    lat_col = cols.get("lat") or cols.get("latitude")
    lon_col = cols.get("lon") or cols.get("longitude")
    if lat_col is None or lon_col is None:
        raise ValueError(f"{path} must contain lat/lon or latitude/longitude columns")
    return (
        df[[lat_col, lon_col]]
        .rename(columns={lat_col: "lat", lon_col: "lon"})
        .drop_duplicates()
        .reset_index(drop=True)
    )


def _default_output_tag(scenario: str) -> str:
    return DEFAULT_OUTPUT_TAG.get(scenario, f"ameli_{scenario.replace('-', '_')}")


def _default_output_path(tag: str, kind: str) -> Path:
    if kind == "region":
        return REPO_ROOT / "inputs" / f"{tag}_wacc_tiam_regions_2050.csv"
    if kind == "country":
        return REPO_ROOT / "inputs" / f"{tag}_wacc_country_map_2050.csv"
    if kind == "interest":
        return REPO_ROOT / "inputs" / f"{tag}_interest_inputs_2050.csv"
    if kind == "combined":
        return REPO_ROOT / "inputs" / f"spatial_cost_inputs_{tag}_2050.csv"
    raise ValueError(f"Unsupported output kind: {kind}")


def build_outputs(args: argparse.Namespace) -> None:
    spatial = pd.read_csv(args.spatial_input)
    required = {"lat", "lon", "tech", "interest_rate"}
    missing = required - set(spatial.columns)
    if missing:
        raise ValueError(f"{args.spatial_input} is missing required columns: {sorted(missing)}")

    wacc = pd.read_csv(args.wacc_table)
    scenario_col = SCENARIO_COLUMN[args.scenario]
    wacc["interest_rate"] = wacc[scenario_col].astype(float) / 100.0

    # The reduced scenario is a floor/cap-like convergence in Fig. 2. For ocean
    # cells and countries outside the local mapping, use 5.1%, the reduced value
    # assigned to most regions. Offshore cells are not part of the Verschuur paper
    # onshore-only analysis, but this keeps green-lory offshore-inclusive runs feasible.
    default_interest_rate = 0.051

    spatial_locations = spatial[["lat", "lon"]].drop_duplicates().reset_index(drop=True)
    spatial_coord_set = set(map(tuple, spatial_locations[["lat", "lon"]].itertuples(index=False, name=None)))
    extra_locations = _locations_from_csv(args.locations_input)
    locations = (
        pd.concat([spatial_locations, extra_locations], ignore_index=True)
        .drop_duplicates(subset=["lat", "lon"])
        .reset_index(drop=True)
    )
    extra_coord_set = set(map(tuple, extra_locations[["lat", "lon"]].itertuples(index=False, name=None)))
    extra_only_coords = extra_coord_set - spatial_coord_set
    locations = assign_countries(locations, args.countries)

    rate_lookup = wacc.set_index("tiam_region_code")["interest_rate"].to_dict()
    name_lookup = wacc.set_index("tiam_region_code")["tiam_region_name"].to_dict()
    locations["tiam_region_name"] = locations["tiam_region_code"].map(name_lookup)
    locations["ameli_interest_rate"] = (
        locations["tiam_region_code"].map(rate_lookup).fillna(default_interest_rate)
    )
    locations["ameli_wacc_pct"] = locations["ameli_interest_rate"] * 100.0

    country_map = (
        locations[["country", "tiam_region_code", "tiam_region_name", "ameli_interest_rate", "ameli_wacc_pct", "mapping_note"]]
        .drop_duplicates()
        .sort_values(["tiam_region_code", "country"], na_position="last")
        .reset_index(drop=True)
    )

    # Per-tech WACC-only file: preserve only the columns run_global requires.
    # This uses the full location set, so it can be paired with the paper-capacity
    # land CSV even where the current spatial-cost notebook filtered out cells.
    techs = spatial[["tech"]].drop_duplicates().sort_values("tech").reset_index(drop=True)
    interest_grid = locations[["lat", "lon"]].merge(techs, how="cross")
    loc_rates = locations[["lat", "lon", "ameli_interest_rate"]]
    with_rates = interest_grid.merge(loc_rates, on=["lat", "lon"], how="left")
    with_rates["interest_rate"] = with_rates["ameli_interest_rate"].fillna(default_interest_rate)
    interest_only = with_rates[["lat", "lon", "tech", "interest_rate"]].copy()

    # Combined file: keep existing spatial build/water columns and overwrite only interest_rate.
    combined = spatial.copy()
    combined = combined.merge(loc_rates, on=["lat", "lon"], how="left")
    combined["interest_rate"] = combined["ameli_interest_rate"].fillna(default_interest_rate)
    combined = combined.drop(columns=["ameli_interest_rate"])

    _write_csv(wacc, args.region_output)
    _write_csv(country_map, args.country_output)
    _write_csv(interest_only, args.interest_output)
    _write_csv(combined, args.combined_output)

    if extra_only_coords:
        print("\nWarning: combined output is limited to the spatial-input grid.")
        print(
            f"  locations-input contributes {len(extra_only_coords):,} extra positive-capacity cells "
            f"not present in {args.spatial_input.name}."
        )
        print("  The WACC-only output covers them, but the combined output cannot.")
        print("  Regenerate the spatial base on the matching land grid before using the combined file.")

    print("\nInterest-rate summary:")
    print(locations["ameli_interest_rate"].describe().to_string())
    print("\nAssigned TIAM regions by location:")
    print(locations["tiam_region_code"].value_counts(dropna=False).sort_index().to_string())

    sample_countries: Iterable[str] = ["Australia", "Chile", "Argentina", "United States of America", "Japan"]
    print("\nSelected country mappings:")
    print(
        country_map[country_map["country"].isin(sample_countries)]
        .sort_values("country")
        .to_string(index=False)
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scenario",
        choices=sorted(SCENARIO_COLUMN),
        default="reduced",
        help="Which Ameli Fig. 2 WACC series to use. Verschuur cites the reduced scenario.",
    )
    parser.add_argument("--spatial-input", type=Path, default=DEFAULT_SPATIAL_INPUT)
    parser.add_argument(
        "--locations-input",
        type=Path,
        default=DEFAULT_LOCATIONS_INPUT,
        help=(
            "Optional CSV with the full grid to cover in the WACC-only output. "
            "Defaults to the paper 2pct slope15 max-capacity grid when present."
        ),
    )
    parser.add_argument("--wacc-table", type=Path, default=DEFAULT_WACC_TABLE)
    parser.add_argument("--countries", type=Path, default=DEFAULT_COUNTRIES)
    parser.add_argument(
        "--output-tag",
        default=None,
        help=(
            "Short token used in default output filenames. "
            "Defaults to 'amelired' for the reduced scenario."
        ),
    )
    parser.add_argument(
        "--region-output",
        type=Path,
        default=None,
    )
    parser.add_argument("--country-output", type=Path, default=None)
    parser.add_argument("--interest-output", type=Path, default=None)
    parser.add_argument("--combined-output", type=Path, default=None)
    args = parser.parse_args()
    output_tag = args.output_tag or _default_output_tag(args.scenario)
    args.output_tag = output_tag
    if args.region_output is None:
        args.region_output = _default_output_path(output_tag, "region")
    if args.country_output is None:
        args.country_output = _default_output_path(output_tag, "country")
    if args.interest_output is None:
        args.interest_output = _default_output_path(output_tag, "interest")
    if args.combined_output is None:
        args.combined_output = _default_output_path(output_tag, "combined")
    return args


if __name__ == "__main__":
    build_outputs(parse_args())
