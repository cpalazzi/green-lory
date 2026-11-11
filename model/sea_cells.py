"""Plants at sea: a sea run's land table, site-cost table and cell list, from a land table.

A sea run solves the land runs' plant with wind at sea and no PV, at the site, with no export
cable. It differs from a land run only in its data, so a coastal cell can hold a land plant in
the land run and a sea plant in the sea run:

- its land table (sea_land_table) gives the plant the cell's sea outside protected areas (less
  the protected share of the cell's water, inputs/site_costs/protected_areas.csv:
  model/protected_areas.py), at the land table's share, as wind area, and no PV area;
- its weather stack (python -m model.weather_profiles stack --sea) holds the wind at sea and no
  PV output;
- its site-cost table (sea_site_cost_table) has the columns of a land run's; a flat run's
  (flat_site_cost_table) keeps only the depth-class costs. Wind and the plant are costed by depth
  class: wind_equipment_ratio and wind_build_ratio are wind's costs at sea
  over the YAML's onshore costs, and plant_multiplier scales every other technology.

A cell holds a sea plant if enough of it is sea (sea_cells). Nothing is left out for ice: frozen
sea counts as sea, and the cold rule prices it at the rule's latitudes. A cell's country is its
economic zone; cells outside every zone are international waters. A finance run reads the rates
of the zones, by cell (sea_finance_table).

    python -m model.sea_cells --land <land table> --land-output <csv> --site-costs-output <csv> \
        --locations-output <csv> [--finance <table by country> --finance-output <csv>]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .config import DEFAULT_TECH_YAML, repo_path
from .parameters import file_record, parameter
from .site_costs import (
    CELL_GEOGRAPHY_CSV,
    DEPTH_MULTIPLIER,
    LABOUR_MULTIPLIER,
    REMOTE_BUILD_AT_ANCHOR,
    REMOTE_CITY_SIZE,
    REMOTE_EQUIPMENT_AT_ANCHOR,
    _sha256,
    cold_rule,
    load_ports,
    remote_multiplier,
    travel_min_to_city,
)
from .site_geography import CELL_ELEVATION_CSV, PORT_TRAVEL_TIMES_CSV, PROTECTED_AREAS_CSV, SEA_ZONES_CSV, nearest_km

# The values below, with their sources: inputs/parameters.yaml, section offshore
_OFFSHORE = parameter("offshore")
MIN_SEA_SHARE = float(_OFFSHORE["min_sea_share"])
FIXED_MAX_DEPTH_M = float(_OFFSHORE["fixed_max_depth_m"])
FLOATING_MAX_DEPTH_M = float(_OFFSHORE["floating_max_depth_m"])
TURBINE_EUR_PER_MW = float(_OFFSHORE["turbine_eur_per_mw"])
ARRAY_CABLES_EUR_PER_MW = float(_OFFSHORE["array_cables_eur_per_mw"])
DEVELOPMENT_EUR_PER_MW = float(_OFFSHORE["development_eur_per_mw"])
_FIXED = _OFFSHORE["fixed_foundation"]
FLOATING_FOUNDATION_EUR_PER_MW = float(_OFFSHORE["floating_foundation_eur_per_mw"])
PLANT_MULTIPLIERS = {name: float(value) for name, value in _OFFSHORE["plant_multiplier"].items()}
DEEP_WIND_MULTIPLIER = float(_OFFSHORE["deep_wind_multiplier"])
REGION_OF_SMALL_STATE = {iso3: region for region, states in _OFFSHORE["ameli_region_of_small_states"].items() for iso3 in states}
KEYS = ["latitude", "longitude"]


def depth_class(depth_m) -> np.ndarray:
    """fixed, floating or deep, by the cell's mean depth."""
    depth = np.asarray(depth_m, float)
    return np.select([depth <= FIXED_MAX_DEPTH_M, depth <= FLOATING_MAX_DEPTH_M], ["fixed", "floating"], "deep")


def fixed_foundation_eur_per_mw(depth_m) -> np.ndarray:
    """DEA's foundation at its reference depth, scaled by DEA's function of depth."""
    def function(depth):
        return float(_FIXED["function_base_eur_per_mw"]) + float(_FIXED["function_eur_per_mw_per_m"]) * depth

    return float(_FIXED["eur_per_mw_at_reference"]) * function(np.asarray(depth_m, float)) / function(float(_FIXED["reference_depth_m"]))


def wind_cost_ratios(depth_m, wind_tech_cost: float, wind_build_cost: float) -> tuple[np.ndarray, np.ndarray]:
    """Wind's equipment and build costs at sea over the YAML's onshore costs. The turbine is the
    equipment; the foundation, array cables and development are the build part. In deep water
    every technology, wind included, costs its onshore cost times a speculative multiplier."""
    depth = np.asarray(depth_m, float)
    kind = depth_class(depth)
    foundation = np.where(kind == "fixed", fixed_foundation_eur_per_mw(depth), FLOATING_FOUNDATION_EUR_PER_MW)
    build = (foundation + ARRAY_CABLES_EUR_PER_MW + DEVELOPMENT_EUR_PER_MW) / wind_build_cost
    deep = kind == "deep"
    return np.where(deep, DEEP_WIND_MULTIPLIER, TURBINE_EUR_PER_MW / wind_tech_cost), np.where(deep, DEEP_WIND_MULTIPLIER, build)


def sea_cells(land_table: pd.DataFrame, geography: pd.DataFrame, elevation: pd.DataFrame, zones: pd.DataFrame,
              protected: pd.DataFrame | None = None) -> pd.DataFrame:
    """The land table's cells that hold a sea plant, with their depth, class and zone, and the
    protected share of their water (protected_water_pct of the protected-area table protected, 0
    without it)."""
    cells = (land_table[[*KEYS, "area", "offshore_area_km2"]]
             .merge(elevation[[*KEYS, "sea_depth_m"]], on=KEYS)
             .merge(zones, on=KEYS)
             .merge(geography[[*KEYS, "cell_area_km2", "sea_km2"]], on=KEYS, how="left"))
    if protected is not None:
        cells = cells.merge(protected[[*KEYS, "protected_water_pct"]], on=KEYS, how="left")
    cells["protected_water_pct"] = cells["protected_water_pct"].fillna(0.0) if protected is not None else 0.0
    # the geography stops at 55 S and 83 N: beyond it the land table's water counts as sea
    cells["sea_share"] = (cells["sea_km2"] / cells["cell_area_km2"]).fillna(cells["offshore_area_km2"] / cells["area"])
    cells = cells[(cells["sea_share"] >= MIN_SEA_SHARE) & cells["sea_depth_m"].notna()]
    cells = cells.drop(columns=["sea_km2", "cell_area_km2", "offshore_area_km2"]).reset_index(drop=True)
    cells["depth_class"] = depth_class(cells["sea_depth_m"])
    return cells


def sea_wind_km2(area_km2, sea_share, protected_water_pct, share):
    """The wind area of a sea plant: the cell's sea less the protected share of its water, at the share."""
    return area_km2 * sea_share * (1.0 - np.asarray(protected_water_pct, float) / 100.0) * share


def sea_land_table(land_table: pd.DataFrame, cells: pd.DataFrame) -> pd.DataFrame:
    """The land table's rows of the sea cells, the sea outside protected areas at the table's
    share as the wind area (sea_wind_km2), with the protected share of the water (protected_water_pct).

    A cell's budget reads wind_onshore_area_km2 whatever the site, so the column holds the sea
    here; in a sea run's results, "onshore" in the capacity columns means the plant's own area."""
    table = land_table.merge(cells[[*KEYS, "sea_share", "protected_water_pct"]], on=KEYS)
    wind_km2 = sea_wind_km2(table["area"], table["sea_share"], table["protected_water_pct"], table["land_competition_fraction"])
    share = wind_km2 / table["area"]
    for column in ("wind_area_km2", "wind_onshore_area_km2", "wind_offshore_area_km2", "renewable_union_area_km2",
                   "renewable_union_area_km2_classwise_nested_v1"):
        table[column] = wind_km2
    for column in ("availability", "wind_availability", "wind_offshore_availability", "renewable_union_availability",
                   "renewable_union_availability_classwise_nested_v1"):
        table[column] = share
    table["wind_onshore_availability"] = 0.0
    table["solar_availability"] = 0.0
    table["solar_area_km2"] = 0.0
    table["max_power_solar_mw"] = 0.0
    table["max_power_wind_mw"] = wind_km2 * table["wind_density_mw_per_km2"]
    table["max_capacity_mw"] = table["max_power_wind_mw"]
    return table.drop(columns="sea_share")


def sea_site_cost_table(cells: pd.DataFrame, ports: pd.DataFrame, *, wind_tech_cost: float, wind_build_cost: float,
                        water_source_usd_per_m3: float) -> pd.DataFrame:
    """The site costs of the sea cells: the columns of a land run's table (model/site_costs.py)."""
    port_km, port_index = nearest_km(cells["latitude"], cells["longitude"], ports["lat"], ports["lon"])
    port_min = ports[f"travel_min_city_{REMOTE_CITY_SIZE}"].to_numpy()[port_index]
    travel_min, _ = travel_min_to_city(np.full(len(cells), np.nan), port_min, port_km, at_sea=True)
    equipment_ratio, build_ratio = wind_cost_ratios(cells["sea_depth_m"], wind_tech_cost, wind_build_cost)
    return pd.DataFrame({
        "latitude": cells["latitude"],
        "longitude": cells["longitude"],
        "sea_depth_m": cells["sea_depth_m"],
        "depth_class": cells["depth_class"],
        "eez_iso3": cells["eez_iso3"],
        "port_distance_km": port_km,
        "nearest_port_id": ports["id"].to_numpy()[port_index],
        "travel_time_to_city_min": travel_min,
        "travel_time_source": "port",
        "equipment_cost_multiplier": remote_multiplier(travel_min, REMOTE_EQUIPMENT_AT_ANCHOR),
        "installation_multiplier": remote_multiplier(travel_min, REMOTE_BUILD_AT_ANCHOR),
        "cold_multiplier": cold_rule(cells["latitude"], np.zeros(len(cells))),
        "plant_multiplier": cells["depth_class"].map(PLANT_MULTIPLIERS).to_numpy(),
        "wind_equipment_ratio": equipment_ratio,
        "wind_build_ratio": build_ratio,
        "slope_multiplier": 1.0,
        "depth_multiplier": DEPTH_MULTIPLIER,
        "labour_multiplier": LABOUR_MULTIPLIER,
        # desalinated at the site, so no pipeline
        "water_source_kind": "sea",
        "water_source_distance_km": 0.0,
        "water_source_cost_usd_per_m3": float(water_source_usd_per_m3),
        "water_pipeline_cost_usd_per_m3": 0.0,
        "water_cost_usd_per_m3": float(water_source_usd_per_m3),
        "land_cost_usd_per_km2_year": 0.0,
    })


def flat_site_cost_table(table: pd.DataFrame) -> pd.DataFrame:
    """A flat run's site costs at sea (bflat): wind and the plant at their depth-class costs, the
    base costs of a plant at sea, without the location factors that a bspat run adds on top."""
    return table.assign(equipment_cost_multiplier=1.0, installation_multiplier=1.0, cold_multiplier=1.0,
                        slope_multiplier=1.0, depth_multiplier=1.0, labour_multiplier=1.0)


def sea_finance_table(cells: pd.DataFrame, by_country: pd.DataFrame) -> pd.DataFrame:
    """A finance table by cell, every technology at the rate of the cell's zone: the country's
    row, or its region's for a small state without one, and in international waters the highest
    rate of by_country (offshore in inputs/parameters.yaml)."""
    rates = by_country.set_index("iso3")["interest_rate"].to_dict()
    regions = by_country.groupby("ameli_region")["interest_rate"].first().to_dict()
    rates.update({iso3: regions[region] for iso3, region in REGION_OF_SMALL_STATE.items()})
    zones = cells["eez_iso3"]
    unknown = sorted(set(zones.dropna()) - set(rates))
    if unknown:
        raise ValueError(f"Zones without a rate: {unknown}; add them to offshore.ameli_region_of_small_states")
    return pd.DataFrame({"lat": cells["latitude"], "lon": cells["longitude"], "tech": "all",
                         "interest_rate": zones.map(rates).fillna(float(by_country["interest_rate"].max()))})


def write_sea_inputs(land_csv: Path, land_output: Path, site_costs_output: Path, locations_output: Path, *,
                     tech_yaml: Path = DEFAULT_TECH_YAML, geography_csv: Path = CELL_GEOGRAPHY_CSV,
                     elevation_csv: Path = CELL_ELEVATION_CSV, zones_csv: Path = SEA_ZONES_CSV,
                     ports_csv: Path = PORT_TRAVEL_TIMES_CSV, protected_csv: Path = PROTECTED_AREAS_CSV,
                     finance_csv: Path | None = None, finance_output: Path | None = None,
                     flat_site_costs_output: Path | None = None) -> pd.DataFrame:
    from .technology import load_tech_inputs, load_tech_meta

    land_table = pd.read_csv(land_csv)
    cells = sea_cells(land_table, pd.read_csv(geography_csv), pd.read_csv(elevation_csv), pd.read_csv(zones_csv),
                      pd.read_csv(protected_csv))
    wind = load_tech_inputs(tech_yaml)["wind"]
    table = sea_site_cost_table(cells, load_ports(ports_csv), wind_tech_cost=float(wind["tech_cost_per_mw"]),
                                wind_build_cost=float(wind["build_cost_per_mw"]),
                                water_source_usd_per_m3=float(load_tech_meta(tech_yaml)["water_cost_baseline_usd_per_m3"]))
    outputs = [land_output, site_costs_output, locations_output] + ([flat_site_costs_output] if flat_site_costs_output else [])
    for path in outputs:
        path.parent.mkdir(parents=True, exist_ok=True)
    sea_land_table(land_table, cells).to_csv(land_output, index=False)
    table.round(6).to_csv(site_costs_output, index=False)
    if flat_site_costs_output:
        flat_site_cost_table(table).round(6).to_csv(flat_site_costs_output, index=False)
    cells[KEYS].rename(columns={"latitude": "lat", "longitude": "lon"}).to_csv(locations_output, index=False)
    if finance_csv is not None:
        sea_finance_table(cells, pd.read_csv(finance_csv)).to_csv(finance_output, index=False)
    provenance = {
        "written_by": "python -m model.sea_cells",
        "rows": int(len(table)),
        "outputs": {path.name: _sha256(path) for path in outputs},
        "inputs": {name: {"path": str(path), "sha256": _sha256(path)}
                   for name, path in (("land_table", land_csv), ("tech_yaml", tech_yaml), ("cell_geography", geography_csv),
                                      ("cell_elevation", elevation_csv), ("sea_zones", zones_csv), ("port_travel_times", ports_csv),
                                      ("protected_areas", protected_csv))},
        "parameters_yaml": file_record(),
    }
    for path in [site_costs_output] + outputs[3:]:
        path.with_name(path.stem + "_provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    return table


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--land", type=Path, required=True, help="the land table the sea run shares its cells and share with")
    parser.add_argument("--land-output", type=Path, required=True)
    parser.add_argument("--site-costs-output", type=Path, required=True)
    parser.add_argument("--locations-output", type=Path, required=True)
    parser.add_argument("--tech-yaml", type=Path, default=DEFAULT_TECH_YAML)
    parser.add_argument("--finance", type=Path, help="Ameli table by country, for a finance table by cell")
    parser.add_argument("--finance-output", type=Path)
    parser.add_argument("--flat-site-costs-output", type=Path, help="the site costs of the flat runs at sea (bflat)")
    args = parser.parse_args(argv)
    table = write_sea_inputs(repo_path(args.land), repo_path(args.land_output), repo_path(args.site_costs_output),
                             repo_path(args.locations_output), tech_yaml=repo_path(args.tech_yaml),
                             finance_csv=repo_path(args.finance) if args.finance else None,
                             finance_output=repo_path(args.finance_output) if args.finance_output else None,
                             flat_site_costs_output=repo_path(args.flat_site_costs_output) if args.flat_site_costs_output else None)
    print(f"wrote {len(table)} sea cells: {table['depth_class'].value_counts().to_dict()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
