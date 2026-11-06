"""Site costs: every location factor of the plant's cost, in one table per land build.

For each onshore cell of a land table, site_cost_table() gives

- equipment_cost_multiplier, on the equipment part of every technology's overnight cost
  (tech_cost_per_mw), and installation_multiplier, on the build part (build_cost_per_mw): the
  remote factor, which rises with log2(1 + d), d the travel time in days from a city of at least
  20,000 people, through a value at the anchor, a fly-in community (remote_multiplier). Every
  point gets one: the travel time is the raster's, or the nearest port's plus the sea leg
  (travel_min_to_city; travel_time_source records which);
- cold_multiplier, on both parts of every technology: at high latitude or altitude (cold_rule),
  the altitude being the mean of the cell's land (cell_elevation.csv);
- slope_multiplier, on the build part of PV and wind only: the mean factor of the slope classes
  of the land the plant's share takes, flattest first (SLOPE_CLASS_FACTORS);
- depth_multiplier and labour_multiplier, both 1 (inputs/parameters.yaml): the table covers
  onshore cells, and there is no wage level by country (finance differences are the finance
  table's job);
- the water price (USD/m3, the site-cost currency of the technology YAML): the source price
  plus a pipeline from the nearest water source (water_pipeline_cost).

The remote and cold rules are in inputs/parameters.yaml with their sources. A technology's
equipment multiplier is equipment x cold, and its build multiplier installation x labour x depth
x cold, times the slope factor for PV and wind; fixed O&M is a fraction of the overnight cost, so
it follows both multipliers (model/technology.py). A sea run's table (model/sea_cells.py) has the
same columns; in a land run plant_multiplier, wind_equipment_ratio and wind_build_ratio are 1.

    python -m model.site_costs --land <land table> --output <site costs csv>
    green-lory site-costs --land <land table> --output <site costs csv>

The table is written with a provenance file beside it (inputs and their hashes, every
parameter); runs pin its hash in runs.yaml (site_costs_csv).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from .config import DEFAULT_TECH_YAML, SITE_COSTS_DIR, SLOPE_CLASS_COLUMNS, repo_path
from .parameters import eur2020_per, file_record, parameter
from .site_geography import CELL_ELEVATION_CSV, PORT_TRAVEL_TIMES_CSV, nearest_km

CELL_GEOGRAPHY_CSV = SITE_COSTS_DIR / "cell_geography.csv"
ONSHORE_LAND_PCT_MIN = 1e-6
MINUTES_PER_DAY = 1440.0

# The values below, with their sources: inputs/parameters.yaml, section site_costs.
_REMOTE = parameter("site_costs.remote_factor")
REMOTE_CITY_SIZE = str(_REMOTE["city_size"])
REMOTE_ANCHOR_DAYS = float(_REMOTE["anchor_days"])
REMOTE_BUILD_AT_ANCHOR = float(_REMOTE["build_at_anchor"])
REMOTE_EQUIPMENT_AT_ANCHOR = float(_REMOTE["equipment_at_anchor"])
SEA_SPEED_KM_PER_H = float(_REMOTE["sea_speed_km_per_h"])
SEA_LEG_INCLUDED_KM = float(_REMOTE["sea_leg_included_km"])
_COLD = parameter("site_costs.cold")
COLD_MULTIPLIER = float(_COLD["multiplier"])
COLD_MIN_ABS_LATITUDE_DEG = float(_COLD["min_abs_latitude_deg"])
COLD_MIN_ELEVATION_M = float(_COLD["min_elevation_m"])

# Build-cost factor of each slope class up to the slope limit (the land tables exclude the rest)
_SLOPE_FACTORS = [float(factor) for factor in parameter("site_costs.slope_class_factors")]
if len(_SLOPE_FACTORS) != len(SLOPE_CLASS_COLUMNS) - 1:
    raise ValueError("site_costs.slope_class_factors needs one factor per slope class bound (land.slope_class_upper_deg)")
SLOPE_CLASS_FACTORS = dict(zip(SLOPE_CLASS_COLUMNS[:-1], _SLOPE_FACTORS))
SLOPE_TECHS = tuple(parameter("site_costs.slope_techs"))
DEPTH_MULTIPLIER = float(parameter("site_costs.depth_multiplier"))
LABOUR_MULTIPLIER = float(parameter("site_costs.labour_multiplier"))

# Water pipeline (site_costs.water_pipeline): unit costs of the Texas Water Development Board's
# Uniform Costing Model in December 2017 dollars, sized for one reference plant's flow
_PIPELINE = parameter("site_costs.water_pipeline")
UCM_RURAL_SOIL_USD_PER_M = {float(diameter_mm): float(usd) for diameter_mm, usd in _PIPELINE["rural_soil_usd_per_m"].items()}
UCM_PIPE_ENGINEERING_CONTINGENCY = float(_PIPELINE["pipe_engineering_contingency"])
UCM_OTHER_ENGINEERING_CONTINGENCY = float(_PIPELINE["other_engineering_contingency"])
UCM_ENVIRONMENTAL_USD_PER_KM = float(_PIPELINE["environmental_usd_per_km"])
UCM_BOOSTER_STATION_USD = float(_PIPELINE["booster_station_usd"])
UCM_PIPE_OM_FRACTION = float(_PIPELINE["pipe_om_fraction"])
UCM_PUMP_OM_FRACTION = float(_PIPELINE["pump_om_fraction"])
UCM_POWER_USD_PER_KWH = float(_PIPELINE["power_usd_per_kwh"])
HAZEN_WILLIAMS_C = float(_PIPELINE["hazen_williams_c"])
GRAVITY_M_PER_S2 = float(parameter("physical.gravity_m_per_s2"))
WATER_DENSITY_KG_PER_M3 = float(parameter("physical.water_density_kg_per_m3"))
BOOSTER_HEAD_M = float(_PIPELINE["booster_lift_m"])
PUMP_EFFICIENCY = float(_PIPELINE["pump_efficiency"])
PIPELINE_DISCOUNT_RATE = float(_PIPELINE["discount_rate"])
PIPE_LIFETIME_YEARS = float(_PIPELINE["pipe_lifetime_years"])
PUMP_LIFETIME_YEARS = float(_PIPELINE["pump_lifetime_years"])
# The pipeline is costed in the site-cost currency, USD2020, which the model converts to EUR2020
# at the 2020 rate; so each UCM value is brought to EUR2020 by the shared currency rule from its
# price year, then stated in USD2020
UCM_COST_TO_USD2020 = eur2020_per("USD", int(_PIPELINE["unit_cost_price_year"])) / eur2020_per("USD", 2020)
UCM_POWER_TO_USD2020 = eur2020_per("USD", int(_PIPELINE["power_price_year"])) / eur2020_per("USD", 2020)
REFERENCE_AMMONIA_T_PER_YEAR = float(parameter("plant.reference_output_mt_per_year")) * 1e6
PIPELINE_MATERIALS_SHARE = float(_PIPELINE["materials_share"])


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 24), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _annuity(rate: float, years: float) -> float:
    factor = (1.0 + rate) ** years
    return rate * factor / (factor - 1.0)


# ---------------------------------------------------------------------------------------------
# The factors
# ---------------------------------------------------------------------------------------------

def travel_min_to_city(raster_min, port_min, port_km, at_sea=False) -> tuple[np.ndarray, np.ndarray]:
    """Minutes from a city, and whether they run through the port.

    The raster's time where it has a value; otherwise, and at sea, the port's own time plus the
    sea leg. At sea the first SEA_LEG_INCLUDED_KM are in DEA's offshore costs, so do not count."""
    raster = np.asarray(raster_min, float)
    by_port = np.isnan(raster) | at_sea
    leg_km = np.maximum(np.asarray(port_km, float) - np.where(at_sea, SEA_LEG_INCLUDED_KM, 0.0), 0.0)
    return np.where(by_port, np.asarray(port_min, float) + leg_km / SEA_SPEED_KM_PER_H * 60.0, raster), by_port


def remote_multiplier(travel_min, at_anchor: float) -> np.ndarray:
    """1 in a city, at_anchor at REMOTE_ANCHOR_DAYS, the same step for each doubling of (1 + days)."""
    days = np.asarray(travel_min, float) / MINUTES_PER_DAY
    return 1.0 + (at_anchor - 1.0) * np.log2(1.0 + days) / np.log2(1.0 + REMOTE_ANCHOR_DAYS)


def cold_rule(latitude, elevation_m) -> np.ndarray:
    """COLD_MULTIPLIER at or beyond the latitude or at or above the elevation, otherwise 1."""
    cold = ((np.abs(np.asarray(latitude, float)) >= COLD_MIN_ABS_LATITUDE_DEG)
            | (np.asarray(elevation_m, float) >= COLD_MIN_ELEVATION_M))
    return np.where(cold, COLD_MULTIPLIER, 1.0)


def slope_multiplier(class_pcts: pd.DataFrame, land_share: float) -> np.ndarray:
    """Mean factor of the land a plant's share takes, flattest class first.

    class_pcts holds, per cell, the percentage of the cell's land in each class up to the slope
    limit (SLOPE_CLASS_FACTORS); land_share is the share of the cell's suitable land the plant may
    use (the land table's land_competition_fraction). Suitable land is taken to spread over the
    classes like all land up to the limit.
    """
    shares = class_pcts[list(SLOPE_CLASS_FACTORS)].to_numpy(float)
    total = shares.sum(axis=1, keepdims=True)
    shares = np.divide(shares, total, out=np.zeros_like(shares), where=total > 0)
    remaining = np.full(len(shares), float(land_share))
    taken_area = np.zeros(len(shares))
    weighted = np.zeros(len(shares))
    for column, factor in zip(range(shares.shape[1]), SLOPE_CLASS_FACTORS.values()):
        taken = np.minimum(shares[:, column], remaining)
        weighted += factor * taken
        taken_area += taken
        remaining -= taken
    return np.divide(weighted, taken_area, out=np.ones(len(shares)), where=taken_area > 0)


def water_pipeline_cost(flow_m3_per_year: float) -> dict:
    """Yearly cost of a water pipeline per m3 delivered and per km (USD2020), the diameter of the
    smallest yearly cost among UCM's standard sizes.

    capital_usd_per_m3_km: pipe and booster stations (annuity plus O&M), which the cell's
    multipliers scale; energy_usd_per_m3_km: friction pumping, unscaled.
    """
    flow = flow_m3_per_year / (365.0 * 86400.0)  # m3/s
    options = []
    for diameter_mm, usd_per_m in UCM_RURAL_SOIL_USD_PER_M.items():
        diameter_m = diameter_mm / 1000.0
        # Hazen-Williams friction loss, metric form (head in m, flow in m3/s, diameter in m)
        head_m_per_km = 1000.0 * 10.67 * flow ** 1.852 / (HAZEN_WILLIAMS_C ** 1.852 * diameter_m ** 4.87)
        pipe_usd_per_km = (usd_per_m * 1000.0 * (1.0 + UCM_PIPE_ENGINEERING_CONTINGENCY)
                           + UCM_ENVIRONMENTAL_USD_PER_KM) * UCM_COST_TO_USD2020
        stations_usd_per_km = (head_m_per_km / BOOSTER_HEAD_M * UCM_BOOSTER_STATION_USD
                               * (1.0 + UCM_OTHER_ENGINEERING_CONTINGENCY) * UCM_COST_TO_USD2020)
        capital_per_year = (pipe_usd_per_km * (_annuity(PIPELINE_DISCOUNT_RATE, PIPE_LIFETIME_YEARS) + UCM_PIPE_OM_FRACTION)
                            + stations_usd_per_km * (_annuity(PIPELINE_DISCOUNT_RATE, PUMP_LIFETIME_YEARS) + UCM_PUMP_OM_FRACTION))
        pump_kw = WATER_DENSITY_KG_PER_M3 * GRAVITY_M_PER_S2 * flow * head_m_per_km / PUMP_EFFICIENCY / 1000.0
        energy_per_year = pump_kw * 8760.0 * UCM_POWER_USD_PER_KWH * UCM_POWER_TO_USD2020
        options.append({
            "diameter_mm": diameter_mm,
            "velocity_m_per_s": flow / (math.pi * diameter_m ** 2 / 4.0),
            "friction_head_m_per_km": head_m_per_km,
            "pipe_usd2020_per_km": pipe_usd_per_km,
            "stations_usd2020_per_km": stations_usd_per_km,
            "capital_usd_per_m3_km": capital_per_year / flow_m3_per_year,
            "energy_usd_per_m3_km": energy_per_year / flow_m3_per_year,
        })
    best = min(options, key=lambda option: option["capital_usd_per_m3_km"] + option["energy_usd_per_m3_km"])
    return {**best, "flow_m3_per_year": flow_m3_per_year, "options": options}


# ---------------------------------------------------------------------------------------------
# The table
# ---------------------------------------------------------------------------------------------

def load_ports(path: Path = PORT_TRAVEL_TIMES_CSV) -> pd.DataFrame:
    """The ports with a travel time: a few lie out in the water on the raster, and the next port
    stands in for them."""
    ports = pd.read_csv(path)
    return ports[ports[f"travel_min_city_{REMOTE_CITY_SIZE}"].notna()].reset_index(drop=True)


def site_cost_table(
    land_table: pd.DataFrame,
    geography: pd.DataFrame,
    ports: pd.DataFrame,
    elevation: pd.DataFrame,
    *,
    water_source_usd_per_m3: float,
    water_m3_per_t: float,
) -> tuple[pd.DataFrame, dict]:
    """Per-cell site costs for the land table's cells with onshore land (module docstring)."""
    missing = [column for column in SLOPE_CLASS_COLUMNS if column not in land_table.columns]
    if missing:
        raise ValueError(f"The land table has no slope classes ({', '.join(missing)}); rebuild it with model.land_table")
    shares = pd.to_numeric(land_table["land_competition_fraction"]).unique()
    if len(shares) != 1:
        raise ValueError("land_competition_fraction must be uniform within a land table")
    # onshore land above rounding noise (sea cells can carry 1e-14 % from the spherical weights)
    cells = land_table[land_table["onshore_land_pct"] > ONSHORE_LAND_PCT_MIN][["latitude", "longitude", *SLOPE_CLASS_COLUMNS]]
    # the geography covers 55 S to 83 N (the lakes database); land cells beyond it (Antarctica) get
    # no site costs, and runs refuse any cell without a row
    in_range = cells["latitude"].between(geography["latitude"].min(), geography["latitude"].max())
    beyond = int((~in_range).sum())
    table = cells[in_range].merge(geography, on=["latitude", "longitude"], how="left", validate="one_to_one")
    table = table.merge(elevation[["latitude", "longitude", "land_elevation_m"]], on=["latitude", "longitude"], how="left")
    uncovered = table["port_distance_km"].isna()
    if uncovered.any():
        examples = ", ".join(f"({lat:g}, {lon:g})" for lat, lon in table.loc[uncovered, ["latitude", "longitude"]].head(5).to_numpy())
        raise ValueError(f"{int(uncovered.sum())} land cells have no geography row, e.g. {examples}")

    port_km, port_index = nearest_km(table["latitude"], table["longitude"], ports["lat"], ports["lon"])
    travel_min, by_port = travel_min_to_city(table[f"travel_min_city_{REMOTE_CITY_SIZE}_median"],
                                             ports[f"travel_min_city_{REMOTE_CITY_SIZE}"].to_numpy()[port_index], port_km)
    pipeline = water_pipeline_cost(water_m3_per_t * REFERENCE_AMMONIA_T_PER_YEAR)
    out = pd.DataFrame({"latitude": table["latitude"], "longitude": table["longitude"]})
    out["port_distance_km"] = port_km
    out["nearest_port_id"] = ports["id"].to_numpy()[port_index]
    out["travel_time_to_city_min"] = travel_min
    out["travel_time_source"] = np.where(by_port, "port", "raster")
    out["equipment_cost_multiplier"] = remote_multiplier(travel_min, REMOTE_EQUIPMENT_AT_ANCHOR)
    out["installation_multiplier"] = remote_multiplier(travel_min, REMOTE_BUILD_AT_ANCHOR)
    out["cold_multiplier"] = cold_rule(table["latitude"], table["land_elevation_m"])
    out["plant_multiplier"] = 1.0
    out["wind_equipment_ratio"] = 1.0
    out["wind_build_ratio"] = 1.0
    out["slope_multiplier"] = slope_multiplier(table, float(shares[0]))
    out["depth_multiplier"] = DEPTH_MULTIPLIER
    out["labour_multiplier"] = LABOUR_MULTIPLIER
    pv_wind_build = out["installation_multiplier"] * out["labour_multiplier"] * out["depth_multiplier"] * out["slope_multiplier"]
    out["water_source_kind"] = table["water_source_kind"]
    out["water_source_distance_km"] = table["water_source_distance_km"]
    out["water_source_cost_usd_per_m3"] = float(water_source_usd_per_m3)
    out["water_pipeline_cost_usd_per_m3"] = out["water_source_distance_km"] * (
        pipeline["capital_usd_per_m3_km"] * out["cold_multiplier"] * (PIPELINE_MATERIALS_SHARE * out["equipment_cost_multiplier"]
                                                                      + (1.0 - PIPELINE_MATERIALS_SHARE) * pv_wind_build)
        + pipeline["energy_usd_per_m3_km"]
    )
    out["water_cost_usd_per_m3"] = out["water_source_cost_usd_per_m3"] + out["water_pipeline_cost_usd_per_m3"]
    out["land_cost_usd_per_km2_year"] = 0.0
    parameters = {
        "land_cells_beyond_the_geography": beyond,
        "cells_by_port": int(by_port.sum()),
        "remote_factor": _REMOTE,
        "cold": _COLD,
        "land_competition_fraction": float(shares[0]),
        "slope_class_factors": SLOPE_CLASS_FACTORS,
        "slope_techs": list(SLOPE_TECHS),
        "water_source_usd_per_m3": float(water_source_usd_per_m3),
        "water_pipeline": {key: value for key, value in pipeline.items() if key != "options"},
        "pipeline_materials_share": PIPELINE_MATERIALS_SHARE,
    }
    return out.reset_index(drop=True), parameters


# ---------------------------------------------------------------------------------------------
# Reading a table for a run
# ---------------------------------------------------------------------------------------------

def load_site_cost_overrides(path: str | Path, tech_names) -> dict:
    """A site-cost table as cell overrides: {(lat, lon): {tech: {...}, "__location__": {...}}}.

    Every technology gets an equipment multiplier (equipment x cold) and a build multiplier
    (installation x labour x depth x cold, times the slope factor for PV and wind), wind times its
    ratios and the others times plant_multiplier; the location entry carries the water price and
    the land cost.
    """
    from .plant import normalize_component_name

    frame = pd.read_csv(repo_path(path))
    names = [name for name in (normalize_component_name(tech) for tech in tech_names) if name is not None]
    index = {}
    for row in frame.itertuples(index=False):
        base_build = row.installation_multiplier * row.labour_multiplier * row.depth_multiplier * row.cold_multiplier
        entry = {
            name: {
                "equipment_cost_multiplier": float(row.equipment_cost_multiplier * row.cold_multiplier
                                                   * (row.wind_equipment_ratio if name == "wind" else row.plant_multiplier)),
                "build_cost_multiplier": float(base_build * (row.slope_multiplier if name in SLOPE_TECHS else 1.0)
                                               * (row.wind_build_ratio if name == "wind" else row.plant_multiplier)),
            }
            for name in names
        }
        entry["__location__"] = {
            "water_cost_usd_per_m3": float(row.water_cost_usd_per_m3),
            "land_cost_usd_per_km2_year": float(row.land_cost_usd_per_km2_year),
        }
        index[(round(float(row.latitude), 4), round(float(row.longitude), 4))] = entry
    return index


def merge_overrides(finance: dict, site_costs: dict) -> dict:
    """The run's interest rates with the site-cost multipliers and water price on top, cell by cell."""
    merged = {}
    for key in set(finance) | set(site_costs):
        cell = {tech: dict(params) for tech, params in finance.get(key, {}).items()}
        for tech, params in site_costs.get(key, {}).items():
            cell.setdefault(tech, {}).update(params)
        merged[key] = cell
    return merged


# ---------------------------------------------------------------------------------------------
# Command line
# ---------------------------------------------------------------------------------------------

def write_site_cost_table(
    land_csv: Path,
    output: Path,
    *,
    tech_yaml: Path = DEFAULT_TECH_YAML,
    geography_csv: Path = CELL_GEOGRAPHY_CSV,
    ports_csv: Path = PORT_TRAVEL_TIMES_CSV,
    elevation_csv: Path = CELL_ELEVATION_CSV,
) -> pd.DataFrame:
    from .technology import load_tech_meta

    meta = load_tech_meta(tech_yaml)
    table, parameters = site_cost_table(
        pd.read_csv(land_csv),
        pd.read_csv(geography_csv),
        load_ports(ports_csv),
        pd.read_csv(elevation_csv),
        # The source price is the YAML's baseline (the flat runs' price): the desalination price
        # at sea, lake and river alike. Fresh water could be treated more cheaply; charging the
        # desalination price is a deliberate shadow price that protects freshwater for other uses
        # (decisions of 7 and 9 October 2026, stated in the paper).
        water_source_usd_per_m3=float(meta["water_cost_baseline_usd_per_m3"]),
        water_m3_per_t=float(meta["water_usage_m3_per_t_nh3"]),
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    table.round(6).to_csv(output, index=False)
    provenance = {
        "written_by": "python -m model.site_costs",
        "rows": int(len(table)),
        "sha256": _sha256(output),
        "currency": f"{meta.get('spatial_cost_currency', 'USD')}{meta.get('spatial_cost_price_year', '')}",
        "inputs": {
            name: {"path": str(path), "sha256": _sha256(path)}
            for name, path in (("land_table", land_csv), ("tech_yaml", tech_yaml),
                               ("cell_geography", geography_csv), ("port_travel_times", ports_csv),
                               ("cell_elevation", elevation_csv))
        },
        "parameters_yaml": file_record(),
        "parameters": parameters,
    }
    output.with_name(output.stem + "_provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    return table


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--land", type=Path, required=True, help="land table (with slope classes)")
    parser.add_argument("--output", type=Path, required=True, help="site-cost table to write")
    parser.add_argument("--tech-yaml", type=Path, default=DEFAULT_TECH_YAML,
                        help="technology YAML: the water source price, the water use and the currency")
    parser.add_argument("--geography", type=Path, default=CELL_GEOGRAPHY_CSV)
    parser.add_argument("--port-travel-times", type=Path, default=PORT_TRAVEL_TIMES_CSV)
    parser.add_argument("--cell-elevation", type=Path, default=CELL_ELEVATION_CSV)


def run(args: argparse.Namespace) -> int:
    table = write_site_cost_table(
        repo_path(args.land), repo_path(args.output), tech_yaml=repo_path(args.tech_yaml),
        geography_csv=repo_path(args.geography), ports_csv=repo_path(args.port_travel_times),
        elevation_csv=repo_path(args.cell_elevation),
    )
    print(f"wrote {args.output}: {len(table)} cells")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_arguments(parser)
    return run(parser.parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
