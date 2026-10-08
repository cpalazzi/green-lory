"""Site costs: every location factor of the plant's cost, in one table per land build.

For each onshore cell of a land table, site_cost_table() gives

- equipment_cost_multiplier, on the equipment part of every technology's overnight cost
  (tech_cost_per_mw): 1 + a x great-circle distance to the nearest port of the shipping network,
  capped. It stands for transporting plant and materials (AEMO's equipment cost factor, based
  on the nearest port since its 2025 inputs);
- installation_multiplier, on the build part (build_cost_per_mw): 1 + b x min(travel time to the
  nearest city of at least N people, saturation). It stands for labour, materials and
  mobilisation from the city (AEMO's installation cost factor);
- slope_multiplier, on the build part of PV and wind only: the mean factor of the slope classes
  of the land the plant's share takes, flattest first (SLOPE_CLASS_FACTORS);
- depth_multiplier and labour_multiplier, both 1: the table covers onshore cells, and there is no
  wage level by country (finance differences are the finance table's job);
- the water price (USD/m3, the site-cost currency of the technology YAML): the source price
  plus a pipeline from the nearest water source (water_pipeline_cost).

a, b, N and the saturation come from inputs/site_costs/remoteness_fit.json, fitted by
model/site_geography.py to one of the AEMO location-factor tables (REMOTENESS_SOURCES). A
technology's build multiplier is installation x labour x depth, times the slope factor for PV
and wind; fixed O&M is a fraction of the overnight cost, so it follows both multipliers
(model/technology.py).

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

CELL_GEOGRAPHY_CSV = SITE_COSTS_DIR / "cell_geography.csv"
ONSHORE_LAND_PCT_MIN = 1e-6
REMOTENESS_FIT_JSON = SITE_COSTS_DIR / "remoteness_fit.json"

# The table the remoteness factors are fitted to (model/site_geography.py fits both):
#   ghd2025      GHD (2025), 2025 Energy Technology Cost and Technical Parameter Review, draft
#                report Rev C for AEMO, Table 8.3: equipment factors by distance from a major
#                port, installation factors from Rawlinsons (2025) regional cost maps (divided
#                by the state capital's factor, Table 8.2, to keep remoteness only)
#   aurecon2024  Aurecon (2024), 2024 Energy Technology Cost and Technical Parameter Review,
#                Rev 3, Table 11-1: equipment factors by distance from the state capital,
#                installation factors derived from Rawlinsons (2024) location factors
# The default is the more recent table (proposed 7 October 2026, for review).
REMOTENESS_SOURCES = ("ghd2025", "aurecon2024")
DEFAULT_REMOTENESS_SOURCE = "ghd2025"

# Build-cost factor of each slope class (decided 7 October 2026; CAMPAIGN_PLAN.md section 0a):
# no uplift up to 8.5 degrees (15 % grade, the limit of standard single-axis trackers), the
# middle of NREL's hilly (1.1-1.5) and mountainous (1.2-2.0) terrain multipliers for civil works
# (Lopez et al. 2025, Table 12) above; land above 20 degrees is excluded by the land table.
SLOPE_CLASS_FACTORS = dict(zip(SLOPE_CLASS_COLUMNS[:3], (1.0, 1.3, 1.6)))
SLOPE_TECHS = ("solar", "solar_tracking", "wind")

# Water pipeline: unit costs of the Texas Water Development Board's Uniform Costing Model
# (HDR and Freese and Nichols 2018, User's Guide v2.0, Appendix C, Tables 1 and 3; December 2017
# dollars), sized for one reference plant's flow. Rural pipelines in soil, with appurtenances,
# by diameter (inches, USD per foot):
UCM_RURAL_SOIL_USD_PER_FT = {6: 24, 8: 38, 10: 52, 12: 65, 14: 79, 16: 92, 18: 106, 20: 120, 24: 147}
UCM_PIPE_ENGINEERING_CONTINGENCY = 0.30  # engineering, legal and contingencies, pipes
UCM_OTHER_ENGINEERING_CONTINGENCY = 0.35  # the same, all other facilities
UCM_ENVIRONMENTAL_USD_PER_MILE = 25_000.0  # environmental studies and mitigation, pipelines
UCM_BOOSTER_STATION_USD = 2.76e6  # smallest booster pump station of the capacity table (5 MGD)
UCM_PIPE_OM_FRACTION = 0.01  # O&M per year, share of capital (pipelines)
UCM_PUMP_OM_FRACTION = 0.025  # the same, pump stations
UCM_POWER_USD_PER_KWH = 0.08  # power cost of the 2021 regional water plans
# Hydraulics: Hazen-Williams C of 120; stations lift up to a 200 psi pipe class less a 25 psi
# residual pressure (123 m); pumps 75 % efficient; no lift for the elevation gain (section 0b).
HAZEN_WILLIAMS_C = 120.0
BOOSTER_HEAD_M = (200.0 - 25.0) * 0.70307
PUMP_EFFICIENCY = 0.75
# Annuity of the pipeline: the base 5 % rate of the uniform-finance runs, 45 years for pipes (as
# the district-heating proxy it replaces) and 20 years for pump stations (UCM's debt period)
PIPELINE_DISCOUNT_RATE = 0.05
PIPE_LIFETIME_YEARS = 45.0
PUMP_LIFETIME_YEARS = 20.0
# US CPI-U (BLS series CUUR0000SA0): December 2017 246.524, 2020 annual average 258.811
USD_DEC2017_TO_USD2020 = 258.811 / 246.524
# Reference flow: one 1 Mt/yr plant at the YAML's water use (1.5 m3 per tonne of ammonia)
REFERENCE_AMMONIA_T_PER_YEAR = 1.0e6
# Share of the pipeline's capital cost that is materials, on the equipment multiplier; the rest
# (installation and soft costs) takes the cell's build multiplier. DEA Technology Data for
# Energy Transport, district-heating transmission line (113): materials 30 %.
PIPELINE_MATERIALS_SHARE = 0.30


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

def equipment_multiplier(port_distance_km, a_per_1000_km: float, cap: float) -> np.ndarray:
    """1 + a x distance to the nearest port, at most cap."""
    return 1.0 + np.minimum(a_per_1000_km / 1000.0 * np.asarray(port_distance_km, float), cap - 1.0)


def installation_multiplier(travel_min, b_per_hour: float, saturation_min: float) -> np.ndarray:
    """1 + b x travel time to the nearest city, the time capped at the saturation.

    Cells without a travel time (no land pixel with data) take the saturation value."""
    minutes = np.asarray(travel_min, float)
    minutes = np.where(np.isnan(minutes), saturation_min, minutes)
    return 1.0 + b_per_hour / 60.0 * np.minimum(minutes, saturation_min)


def slope_multiplier(class_pcts: pd.DataFrame, land_share: float) -> np.ndarray:
    """Mean factor of the land a plant's share takes, flattest class first.

    class_pcts holds, per cell, the percentage of the cell's land in each class up to 20 degrees
    (SLOPE_CLASS_FACTORS); land_share is the share of the cell's suitable land the plant may use
    (the land table's land_competition_fraction). Suitable land is taken to spread over the
    classes like all land up to 20 degrees.
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
    feet_per_km = 1000.0 / 0.3048
    options = []
    for diameter_in, usd_per_ft in UCM_RURAL_SOIL_USD_PER_FT.items():
        diameter_m = diameter_in * 0.0254
        head_m_per_km = 1000.0 * 10.67 * flow ** 1.852 / (HAZEN_WILLIAMS_C ** 1.852 * diameter_m ** 4.87)
        pipe_usd_per_km = (usd_per_ft * feet_per_km * (1.0 + UCM_PIPE_ENGINEERING_CONTINGENCY)
                           + UCM_ENVIRONMENTAL_USD_PER_MILE / 1.609344) * USD_DEC2017_TO_USD2020
        stations_usd_per_km = (head_m_per_km / BOOSTER_HEAD_M * UCM_BOOSTER_STATION_USD
                               * (1.0 + UCM_OTHER_ENGINEERING_CONTINGENCY) * USD_DEC2017_TO_USD2020)
        capital_per_year = (pipe_usd_per_km * (_annuity(PIPELINE_DISCOUNT_RATE, PIPE_LIFETIME_YEARS) + UCM_PIPE_OM_FRACTION)
                            + stations_usd_per_km * (_annuity(PIPELINE_DISCOUNT_RATE, PUMP_LIFETIME_YEARS) + UCM_PUMP_OM_FRACTION))
        energy_per_year = 1000.0 * 9.81 * flow * head_m_per_km / PUMP_EFFICIENCY / 1000.0 * 8760.0 * UCM_POWER_USD_PER_KWH
        options.append({
            "diameter_in": diameter_in,
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

def load_remoteness_fit(path: Path = REMOTENESS_FIT_JSON, source: str = DEFAULT_REMOTENESS_SOURCE) -> dict:
    if source not in REMOTENESS_SOURCES:
        raise ValueError(f"remoteness source must be one of {REMOTENESS_SOURCES}")
    fit = json.loads(Path(path).read_text())["fits"][source]
    installation = fit["installation"]["best"]
    return {
        "source": source,
        "equipment_a_per_1000_km": float(fit["equipment"]["a_per_1000_km"]),
        "equipment_cap": float(fit["equipment"]["cap"]),
        "installation_city_size": str(installation["city_size"]),
        "installation_b_per_hour": float(installation["b_per_hour"]),
        "installation_saturation_min": float(installation["saturation_min"]),
    }


def site_cost_table(
    land_table: pd.DataFrame,
    geography: pd.DataFrame,
    remoteness: dict,
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
    uncovered = table["port_distance_km"].isna()
    if uncovered.any():
        examples = ", ".join(f"({lat:g}, {lon:g})" for lat, lon in table.loc[uncovered, ["latitude", "longitude"]].head(5).to_numpy())
        raise ValueError(f"{int(uncovered.sum())} land cells have no geography row, e.g. {examples}")

    travel_column = f"travel_min_city_{remoteness['installation_city_size']}_median"
    pipeline = water_pipeline_cost(water_m3_per_t * REFERENCE_AMMONIA_T_PER_YEAR)
    out = pd.DataFrame({"latitude": table["latitude"], "longitude": table["longitude"]})
    out["port_distance_km"] = table["port_distance_km"]
    out["nearest_port_id"] = table["nearest_port_id"]
    out["travel_time_to_city_min"] = table[travel_column]
    out["equipment_cost_multiplier"] = equipment_multiplier(
        table["port_distance_km"], remoteness["equipment_a_per_1000_km"], remoteness["equipment_cap"])
    out["installation_multiplier"] = installation_multiplier(
        table[travel_column], remoteness["installation_b_per_hour"], remoteness["installation_saturation_min"])
    out["slope_multiplier"] = slope_multiplier(table, float(shares[0]))
    out["depth_multiplier"] = 1.0
    out["labour_multiplier"] = 1.0
    pv_wind_build = out["installation_multiplier"] * out["labour_multiplier"] * out["depth_multiplier"] * out["slope_multiplier"]
    out["water_source_kind"] = table["water_source_kind"]
    out["water_source_distance_km"] = table["water_source_distance_km"]
    out["water_source_cost_usd_per_m3"] = float(water_source_usd_per_m3)
    out["water_pipeline_cost_usd_per_m3"] = out["water_source_distance_km"] * (
        pipeline["capital_usd_per_m3_km"] * (PIPELINE_MATERIALS_SHARE * out["equipment_cost_multiplier"]
                                             + (1.0 - PIPELINE_MATERIALS_SHARE) * pv_wind_build)
        + pipeline["energy_usd_per_m3_km"]
    )
    out["water_cost_usd_per_m3"] = out["water_source_cost_usd_per_m3"] + out["water_pipeline_cost_usd_per_m3"]
    out["land_cost_usd_per_km2_year"] = 0.0
    parameters = {
        "land_cells_beyond_the_geography": beyond,
        "remoteness": remoteness,
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

    Every technology gets the equipment multiplier and a build multiplier (installation x labour
    x depth, times the slope factor for PV and wind); the location entry carries the water price
    and the land cost.
    """
    from .plant import normalize_component_name

    frame = pd.read_csv(repo_path(path))
    names = [name for name in (normalize_component_name(tech) for tech in tech_names) if name is not None]
    index = {}
    for row in frame.itertuples(index=False):
        base_build = row.installation_multiplier * row.labour_multiplier * row.depth_multiplier
        entry = {
            name: {
                "equipment_cost_multiplier": float(row.equipment_cost_multiplier),
                "build_cost_multiplier": float(base_build * (row.slope_multiplier if name in SLOPE_TECHS else 1.0)),
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
    """Finance overrides (interest rates) with site-cost overrides on top, cell by cell."""
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
    fit_json: Path = REMOTENESS_FIT_JSON,
    remoteness_source: str = DEFAULT_REMOTENESS_SOURCE,
) -> pd.DataFrame:
    from .technology import load_tech_meta

    meta = load_tech_meta(tech_yaml)
    table, parameters = site_cost_table(
        pd.read_csv(land_csv),
        pd.read_csv(geography_csv),
        load_remoteness_fit(fit_json, remoteness_source),
        # The source price is the YAML's baseline (the flat runs' price): the desalination price
        # at sea and lake alike. Lake water could be treated more cheaply; charging the
        # desalination price is a deliberate shadow price that protects freshwater for other uses
        # (decision of 7 October 2026, stated in the paper).
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
                               ("cell_geography", geography_csv), ("remoteness_fit", fit_json))
        },
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
    parser.add_argument("--remoteness-fit", type=Path, default=REMOTENESS_FIT_JSON)
    parser.add_argument("--remoteness-source", choices=REMOTENESS_SOURCES, default=DEFAULT_REMOTENESS_SOURCE)


def run(args: argparse.Namespace) -> int:
    table = write_site_cost_table(
        repo_path(args.land), repo_path(args.output), tech_yaml=repo_path(args.tech_yaml),
        geography_csv=repo_path(args.geography), fit_json=repo_path(args.remoteness_fit),
        remoteness_source=args.remoteness_source,
    )
    print(f"wrote {args.output}: {len(table)} cells")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_arguments(parser)
    return run(parser.parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
