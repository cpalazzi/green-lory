"""Global orchestration helpers for looping the ammonia plant model over many locations."""
from __future__ import annotations

import argparse
import json
import copy
import io
import logging
import math
import multiprocessing
import os
import sys
import time
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Sequence, Tuple

import geopandas as gpd
import numpy as np
import pandas as pd
import yaml

try:
    from . import main as plant_main
    from . import location_tools as lt
    from . import data_store as results_store
    from . import data_paths
    from . import land_processing
    from . import land_capacity
    from . import land_union
    from . import result_accounting
except ImportError:  # pragma: no cover - fallback for direct execution
    PACKAGE_ROOT = Path(__file__).resolve().parent
    if str(PACKAGE_ROOT) not in sys.path:
        sys.path.insert(0, str(PACKAGE_ROOT))
    import main as plant_main  # type: ignore
    import location_tools as lt  # type: ignore
    import data_store as results_store  # type: ignore
    import data_paths  # type: ignore
    import land_processing  # type: ignore
    import land_capacity  # type: ignore
    import land_union  # type: ignore
    import result_accounting  # type: ignore

LOGGER = logging.getLogger(__name__)

REPO_ROOT = data_paths.REPO_ROOT
DEFAULT_WEATHER_DIR = data_paths.WEATHER_DATA_DIR
DEFAULT_LAND_CSV = data_paths.MAX_CAPACITIES_FILE
DEFAULT_TECH_YAML = REPO_ROOT / "inputs" / "tech_config_ammonia_plant_2030_dea.yaml"
RENEWABLES = ["wind", "solar", "solar_tracking"]
_LAT_LON_TOLERANCE = 0.125  # match within 1/8th degree
# Techno-economic inputs are now expected to be pre-processed into the CSV bundle.
# Interest-rate defaults are therefore not sourced from a runtime YAML file.
DEFAULT_INTEREST_RATES: Dict[str, float] = {}

WIND_GENERATORS = ["wind"]

# Module-level state populated by _worker_init() in each pool worker process.
_WORKER_STATE: Dict[str, Any] = {}
# Loaded once by the main process before forking; inherited by workers via CoW.
_SHARED_DATASET: Any = None


# Gurobi settings for a fresh-network retry after a numerical termination.  Barrier without
# crossover (the fast default in model/main.py) occasionally ends SUBOPTIMAL on hard cells;
# dual simplex with numeric focus is slower but robust.  Same set as the supply pilots.
ROBUST_GUROBI_OPTIONS = {"Method": 1, "DualReductions": 0, "NumericFocus": 3, "InfUnbdInfo": 1}
# Retry chain after a numerical termination, cheapest first (probes of 28 Sep 2026 on stalling
# cells: baseline barrier 5 s; barrier with crossover about one minute; dual simplex about
# fifteen minutes).  Each attempt rebuilds the network from scratch.
CROSSOVER_GUROBI_OPTIONS = {"Method": 2, "Crossover": 1, "BarConvTol": 1e-8}
NUMERICAL_RETRY_OPTIONS = (CROSSOVER_GUROBI_OPTIONS, ROBUST_GUROBI_OPTIONS)
NUMERICAL_RETRY_CONDITIONS = frozenset({"suboptimal", "numeric", "numerical", "infeasible_or_unbounded"})


def _solver_failure(error: BaseException | None):
    """The OptimizationFailure behind a wrapped error, or None."""
    while error is not None:
        if isinstance(error, plant_main.OptimizationFailure):
            return error
        error = error.__cause__
    return None


def _gurobi_selected() -> bool:
    return os.environ.get("GREEN_LORY_SOLVER", "gurobi").strip().lower() == "gurobi"


def _run_single_location_with_numerical_retry(*, fail_fast: bool, **kwargs) -> Tuple[str, Dict[str, Any] | None]:
    """Run one cell; after a suboptimal or numerical Gurobi termination, rebuild the network and
    solve again with each option set of NUMERICAL_RETRY_OPTIONS in turn (barrier with crossover,
    then dual simplex).  Other failures follow the fail_fast contract: raised when fail_fast,
    otherwise logged and returned as a status without results."""
    lat, lon = kwargs.get("lat"), kwargs.get("lon")
    attempts = [None, *NUMERICAL_RETRY_OPTIONS]
    first_condition = ""
    for index, options in enumerate(attempts):
        try:
            status, results = _run_single_location(fail_fast=True, solver_options_override=options, **kwargs)
        except RuntimeError as error:
            failure = _solver_failure(error)
            retryable = (
                failure is not None
                and failure.condition in NUMERICAL_RETRY_CONDITIONS
                and _gurobi_selected()
                and index < len(attempts) - 1
            )
            if not retryable:
                if fail_fast:
                    raise
                LOGGER.warning("Cell (%s, %s) failed after %d attempt(s): %s", lat, lon, index + 1, error)
                message = str(error)
                return ("setup" if message.startswith("Network setup failed") else "solver"), None
            if index == 0:
                first_condition = failure.condition
            LOGGER.warning(
                "Numerical termination %s at (%s, %s) on attempt %d; retrying on a fresh network with %s",
                failure.condition, lat, lon, index + 1, attempts[index + 1],
            )
            continue
        if results is not None:
            results["solver_numerical_retry"] = index > 0
            results["solver_retry_attempts"] = index
            results["solver_retry_termination"] = first_condition
            results["solver_retry_options"] = json.dumps(options, sort_keys=True) if options else ""
        return status, results
    raise AssertionError("retry chain exhausted without a result or an exception")


def _annuity_factor(interest_rate: float, lifetime_years: float) -> float:
    if lifetime_years <= 0:
        raise ValueError("lifetime_years must be positive")
    if abs(interest_rate) < 1e-12:
        return 1.0 / lifetime_years
    factor = (1.0 + interest_rate) ** lifetime_years
    return interest_rate * factor / (factor - 1.0)


def _annualised_capital_cost(
    overnight_cost: float,
    interest_rate: float,
    lifetime_years: float,
    fixed_om_fraction: float,
) -> float:
    crf = _annuity_factor(float(interest_rate), float(lifetime_years))
    return float(overnight_cost) * (crf + float(fixed_om_fraction))


def _deep_merge_dicts(base: Dict[str, Any], overlay: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merge a small YAML overlay without mutating either input."""
    merged: Dict[str, Any] = copy.deepcopy(base)
    for key, value in overlay.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge_dicts(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def _load_tech_config(
    path: str | Path,
    _seen: set[Path] | None = None,
) -> Dict[str, Any]:
    """Load a tech YAML, resolving an optional relative ``extends`` chain."""
    resolved = Path(path).expanduser().resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"Tech YAML not found: {resolved}")
    seen = set() if _seen is None else set(_seen)
    if resolved in seen:
        chain = " -> ".join(str(item) for item in [*seen, resolved])
        raise ValueError(f"Circular tech YAML extends chain: {chain}")
    seen.add(resolved)

    raw = yaml.safe_load(resolved.read_text()) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"Tech YAML must contain a mapping: {resolved}")
    parent = raw.pop("extends", None)
    if parent is None:
        return raw
    parent_path = Path(parent)
    if not parent_path.is_absolute():
        parent_path = resolved.parent / parent_path
    return _deep_merge_dicts(_load_tech_config(parent_path, seen), raw)


def _load_tech_inputs(path: str | Path | None) -> Dict[str, dict]:
    resolved = _resolve_path(path) or DEFAULT_TECH_YAML
    data = _load_tech_config(resolved)
    currency = data.get("currency")
    if currency:
        os.environ["GREEN_LORY_CURRENCY"] = str(currency).strip().upper()
    techs = data.get("techs")
    if not isinstance(techs, dict):
        LOGGER.warning("Tech YAML %s missing top-level 'techs' mapping; skipping overrides.", resolved)
        return {}
    out: Dict[str, dict] = {}
    for name, raw in techs.items():
        if not isinstance(raw, dict):
            continue
        normalized = plant_main.normalize_component_name(name)
        if normalized is None:
            continue
        out[normalized] = raw
    # Pass through the grid_backstop section (if present) directly — it is not a
    # PyPSA component and does not need normalization.  Absence means the backstop
    # is disabled (the YAML comment controls this).
    grid_backstop = data.get("grid_backstop")
    if isinstance(grid_backstop, dict):
        out["grid_backstop"] = grid_backstop
    return out


def _load_tech_meta(path: str | Path | None) -> Dict[str, Any]:
    resolved = _resolve_path(path) or DEFAULT_TECH_YAML
    data = _load_tech_config(resolved)
    meta: Dict[str, Any] = {}
    if "currency" in data:
        meta["currency"] = str(data["currency"]).strip().upper()
    for key in (
        "model_currency_price_year",
        "spatial_cost_currency",
        "spatial_cost_price_year",
        "spatial_cost_to_model_currency",
    ):
        if key in data:
            meta[key] = data[key]
    if "water_usage_m3_per_t_nh3" in data:
        meta["water_usage_m3_per_t_nh3"] = float(data["water_usage_m3_per_t_nh3"])
    if "water_cost_baseline_usd_per_m3" in data:
        meta["water_cost_baseline_usd_per_m3"] = float(data["water_cost_baseline_usd_per_m3"])
    return meta


def _apply_yaml_base_costs(
    network,
    tech_inputs: Dict[str, dict],
    aggregation_count: int,
    time_step: float,
    temporal_accounting_mode: str = "snapshot_weighted",
) -> None:
    """Unconditionally set capital_cost for every tech using YAML defaults.

    This ensures the network always reflects the current tech config, regardless
    of whether per-location finance overrides are present.  _apply_finance_overrides
    may later adjust individual techs where an override CSV provides different
    interest_rate or build_cost_multiplier values.

    Convention:
    - YAML overnight costs are quoted on an output basis (MW_out / MWh_out).
    - PyPSA Links are sized on bus0 (input) MW, so link capital_cost is
      converted to per-MW_in/year by multiplying by bus0→bus1 efficiency.
    - Corrected runs use physical MWh and PyPSA snapshot weights.  Store-cost
      scaling is retained only in the explicit ``legacy_scaled`` mode.
    """
    if not tech_inputs:
        return

    store_scale = (
        float(time_step) * float(aggregation_count)
        if temporal_accounting_mode == "legacy_scaled"
        else 1.0
    )

    for name, raw in tech_inputs.items():
        if not isinstance(raw, dict):
            continue
        component_type = str(raw.get("component_type", "")).lower()
        lifetime_years = float(raw.get("lifetime_years", 20.0))
        interest_rate = float(raw.get("interest_rate", 0.07))
        fixed_om_fraction = float(raw.get("fixed_om_fraction", 0.0))

        if component_type in {"generator", "link"}:
            overnight = raw.get("overnight_cost_per_mw")
            if overnight is None:
                continue
            annual_out = _annualised_capital_cost(
                float(overnight), interest_rate, lifetime_years, fixed_om_fraction
            )

            if component_type == "generator":
                if name in network.generators.index:
                    network.generators.loc[name, "capital_cost"] = annual_out
                continue

            # Link: YAML cost is per MW_out on bus1; convert to per MW_in on bus0.
            if name in network.links.index:
                efficiency = float(network.links.loc[name, "efficiency"])
                if efficiency <= 0:
                    raise ValueError(
                        f"Link '{name}' has non-positive efficiency; cannot compute capital_cost"
                    )
                network.links.loc[name, "capital_cost"] = annual_out * efficiency

        elif component_type == "store":
            overnight = raw.get("overnight_cost_per_mwh")
            if overnight is None:
                continue
            annual = _annualised_capital_cost(
                float(overnight), interest_rate, lifetime_years, fixed_om_fraction
            )
            if name in network.stores.index:
                network.stores.loc[name, "capital_cost"] = annual * store_scale


def _apply_finance_overrides(
    network,
    tech_inputs: Dict[str, dict],
    overrides: Dict[str, Dict[str, float]] | None,
    aggregation_count: int,
    time_step: float,
    temporal_accounting_mode: str = "snapshot_weighted",
) -> None:
    """Apply per-location interest_rate and build_cost_multiplier overrides.

    Called AFTER _apply_yaml_base_costs to adjust capital_cost where the finance
    override CSV specifies a different interest_rate or build_cost_multiplier.

    Convention:
    - Build cost multiplier applies to build_cost (labor + remoteness sensitive).
    - PyPSA Links are sized on an input basis (bus0 MW_in), so link capital_cost
      must be currency/MW_in/year.
    - Store scaling is applied only in the explicit legacy temporal mode.
    """
    if not overrides or not tech_inputs:
        return

    for tech, params in overrides.items():
        if tech == "__location__":
            continue
        rate = params.get("interest_rate")
        build_mult = params.get("build_cost_multiplier", 1.0)
        if rate is None:
            continue
        normalized = plant_main.normalize_component_name(tech)
        if normalized is None:
            continue
        raw = tech_inputs.get(normalized)
        if not raw:
            continue

        component_type = str(raw.get("component_type", "")).lower()
        lifetime_years = float(raw.get("lifetime_years", 20.0))
        fixed_om_fraction = float(raw.get("fixed_om_fraction", 0.0))

        if component_type in {"generator", "link"}:
            # Compute effective overnight cost with build_cost_multiplier applied
            tech_cost = raw.get("tech_cost_per_mw")
            build_cost = raw.get("build_cost_per_mw")
            if tech_cost is None or build_cost is None:
                raise ValueError(
                    f"Tech '{normalized}' must define tech_cost_per_mw and build_cost_per_mw; "
                    "unsplit overnight-cost inputs are no longer supported."
                )

            effective_build_cost = float(build_cost) * float(build_mult)
            overnight = float(tech_cost) + effective_build_cost

            annual_out = _annualised_capital_cost(overnight, float(rate), lifetime_years, fixed_om_fraction)

            if component_type == "generator":
                if normalized in network.generators.index:
                    network.generators.loc[normalized, "capital_cost"] = annual_out
                continue

            # Link: YAML is MW_out on bus1; PyPSA needs MW_in on bus0.
            if normalized in network.links.index:
                efficiency = float(network.links.loc[normalized, "efficiency"])
                if efficiency <= 0:
                    raise ValueError(f"Link '{normalized}' has non-positive efficiency; cannot apply finance overrides")
                network.links.loc[normalized, "capital_cost"] = annual_out * efficiency

        elif component_type == "store":
            # Compute effective overnight cost with build_cost_multiplier applied
            tech_cost = raw.get("tech_cost_per_mwh")
            build_cost = raw.get("build_cost_per_mwh")
            if tech_cost is None or build_cost is None:
                raise ValueError(
                    f"Store '{normalized}' must define tech_cost_per_mwh and build_cost_per_mwh; "
                    "unsplit overnight-cost inputs are no longer supported."
                )

            effective_build_cost = float(build_cost) * float(build_mult)
            overnight = float(tech_cost) + effective_build_cost

            annual = _annualised_capital_cost(overnight, float(rate), lifetime_years, fixed_om_fraction)
            if normalized in network.stores.index:
                store_scale = (
                    float(time_step) * float(aggregation_count)
                    if temporal_accounting_mode == "legacy_scaled"
                    else 1.0
                )
                network.stores.loc[normalized, "capital_cost"] = annual * store_scale


def _link_capacity_mw_out(network, link_name: str) -> float:
    if link_name not in network.links.index:
        return 0.0
    eff_col = "output_basis_efficiency" if "output_basis_efficiency" in network.links.columns else "efficiency"
    efficiency = float(network.links.at[link_name, eff_col]) if eff_col in network.links.columns else 1.0
    if math.isnan(efficiency):
        efficiency = 1.0
    p_nom = network.links.at[link_name, "p_nom_opt"]
    if pd.isna(p_nom):
        return 0.0
    return float(p_nom) * float(efficiency)


def _component_capacity_for_cost(network, component_type: str, name: str) -> float:
    if component_type == "generator":
        if name not in network.generators.index:
            return 0.0
        p_nom = network.generators.at[name, "p_nom_opt"]
        return 0.0 if pd.isna(p_nom) else float(p_nom)
    if component_type == "link":
        return _link_capacity_mw_out(network, name)
    if component_type == "store":
        if name not in network.stores.index:
            return 0.0
        e_nom = network.stores.at[name, "e_nom_opt"]
        return 0.0 if pd.isna(e_nom) else float(e_nom)
    return 0.0


def _compute_headline_splits(
    network,
    tech_inputs: Dict[str, dict],
    overrides: Dict[str, Dict[str, float]] | None,
    aggregation_count: int,
    time_step: float,
    temporal_accounting_mode: str = "snapshot_weighted",
) -> Dict[str, float]:
    """Compute headline cost splits without double counting.

    Returns percentages of total annual cost for:
    - build_cost_pct: build portion of principal recovery
    - tech_cost_pct: tech portion of principal recovery
    - om_cost_pct: fixed O&M
    - interest_pct: interest portion of capital recovery
    Also returns weighted build_cost_multiplier.
    """
    total_cost = float(network.objective) if network.objective else 0.0
    if total_cost <= 0:
        return {}

    total_build_principal = 0.0
    total_tech_principal = 0.0
    total_interest = 0.0
    total_fixed_om = 0.0
    base_build_cost = 0.0
    weighted_build_cost = 0.0

    overrides = overrides or {}
    store_scale = (
        float(aggregation_count) * float(time_step)
        if temporal_accounting_mode == "legacy_scaled"
        else 1.0
    )

    for name, raw in tech_inputs.items():
        if not isinstance(raw, dict):
            continue
        component_type = str(raw.get("component_type", "")).lower()
        capacity = _component_capacity_for_cost(network, component_type, name)
        if capacity <= 0:
            continue

        tech_cost = raw.get("tech_cost_per_mw") if component_type in {"generator", "link"} else raw.get("tech_cost_per_mwh")
        build_cost = raw.get("build_cost_per_mw") if component_type in {"generator", "link"} else raw.get("build_cost_per_mwh")
        if tech_cost is None or build_cost is None:
            continue

        override_params = overrides.get(name, {})
        build_mult = float(override_params.get("build_cost_multiplier", 1.0))
        tech_cost_total = float(tech_cost) * capacity
        build_cost_total_base = float(build_cost) * capacity
        build_cost_total = build_cost_total_base * build_mult

        lifetime_years = float(raw.get("lifetime_years", 20.0))
        interest_rate = float(override_params.get("interest_rate", raw.get("interest_rate", 0.07)))
        fixed_om_fraction = float(raw.get("fixed_om_fraction", 0.0))

        if lifetime_years <= 0:
            continue

        overnight_total = tech_cost_total + build_cost_total
        if component_type == "store":
            overnight_total *= store_scale
        principal_total = overnight_total / lifetime_years
        annual_capital = overnight_total * _annuity_factor(interest_rate, lifetime_years)
        interest_total = max(0.0, annual_capital - principal_total)
        fixed_om_total = overnight_total * fixed_om_fraction

        if component_type == "store":
            tech_cost_total *= store_scale
            build_cost_total *= store_scale

        total_tech_principal += tech_cost_total / lifetime_years
        total_build_principal += build_cost_total / lifetime_years
        total_interest += interest_total
        total_fixed_om += fixed_om_total
        base_build_cost += (
            build_cost_total_base * store_scale
            if component_type == "store"
            else build_cost_total_base
        )
        weighted_build_cost += build_cost_total

    build_mult_applied = None
    if base_build_cost > 0:
        build_mult_applied = weighted_build_cost / base_build_cost

    return {
        "build_cost_multiplier": build_mult_applied if build_mult_applied is not None else 1.0,
        "build_cost_pct": total_build_principal / total_cost * 100.0,
        "tech_cost_pct": total_tech_principal / total_cost * 100.0,
        "om_cost_pct": total_fixed_om / total_cost * 100.0,
        "interest_pct": total_interest / total_cost * 100.0,
    }


def _order_results_columns(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    currency_code = str(df["currency"].iloc[0]).strip().lower() if "currency" in df.columns else "usd"
    lcoa_col = f"lcoa_{currency_code}_per_t"
    plant_lcoa_col = f"lcoa_plant_{currency_code}_per_t"
    total_cost_col = f"total_cost_{currency_code}_per_year"
    plant_total_cost_col = f"plant_objective_cost_{currency_code}_per_year"
    headline_identity_col = f"headline_cost_identity_residual_{currency_code}_per_year"
    water_cost_col = f"water_cost_{currency_code}_per_t"
    land_cost_col = f"land_cost_{currency_code}_per_t"

    headline = [
        "latitude",
        "longitude",
        "country",
        "scenario_id",
        "run_id",
        "manifest_sha256",
        "land_constraint",
        "capacity_rule",
        "land_allocation",
        "temporal_accounting_mode",
        "ramp_limit_basis",
        "currency",
        lcoa_col,
        plant_lcoa_col,
        "annual_ammonia_demand_mwh",
        "annual_ammonia_production_t",
        "gridless_ammonia_production_t",
        total_cost_col,
        plant_total_cost_col,
        "site_costs_in_headline",
        "site_cost_source_currency",
        "site_cost_source_to_output_fx",
        "build_cost_multiplier",
        "build_cost_pct",
        "tech_cost_pct",
        "om_cost_pct",
        "interest_pct",
        "water_cost_pct",
        "land_cost_pct",
        "other_cost_pct",
        "plant_cost_pct",
        "headline_cost_share_total_pct",
        "headline_cost_share_residual_pct",
        headline_identity_col,
        "cost_percentage_basis",
        "uses_grid_backstop",
        "grid_energy_mwh",
        "grid_energy_share",
        "grid_energy_reference_mwh",
        "grid_energy_reference_basis",
        "grid_energy_tolerance_mwh",
        "grid_free_result_status",
        "is_gridless_feasible",
        "simulated_hours",
        "is_full_year_result",
    ]

    land_water = [
        "water_cost_usd_per_m3",
        "water_usage_m3_per_t_nh3",
        water_cost_col,
        "land_cost_usd_per_km2_year",
        "land_used_km2",
        land_cost_col,
        "land_capacity_cap_mw",
        "land_onshore_pct",
        "offshore_sea_pct",
        "land_cell_area_km2",
        "onshore_area_km2",
        "offshore_area_km2",
        "elevation_m",
        "area_cap_mw",
        "max_ammonia_capacity_t",
        "max_ammonia_capacity_mtpa",
        "max_onshore_ammonia_capacity_t",
        "max_onshore_ammonia_capacity_mtpa",
        "max_gridless_onshore_ammonia_capacity_t",
        "max_gridless_onshore_ammonia_capacity_mtpa",
        "scaled_design_max_onshore_ammonia_capacity_t",
        "scaled_design_max_onshore_ammonia_capacity_mtpa",
        "scaled_design_max_gridless_onshore_ammonia_capacity_t",
        "scaled_design_max_gridless_onshore_ammonia_capacity_mtpa",
        "scaled_design_renewable_capacity_scale_factor",
        "scaled_design_limiting_constraint",
        "preferred_supplier_capacity_column",
        "legacy_capacity_alias_method",
        "renewable_union_area_source",
        "renewable_union_area_is_conservative_fallback",
        "renewable_union_area_is_lower_bound_approximation",
        "renewable_union_area_method",
        "renewable_union_area_method_version",
        "capacity_limit_technology",
        "onshore_capacity_limit_technology",
        "renewable_capacity_scale_factor",
        "onshore_renewable_capacity_scale_factor",
        "wind_mw_per_t_nh3",
        "solar_mw_per_t_nh3",
        "protected_area_pct",
        "slope_suitable_land_pct",
        "steep_slope_pct",
        "land_exclusion_factor",
        "land_competition_fraction",
        "constrained_onshore_area_km2",
        "solar_area_used_km2",
        "wind_area_used_km2",
        "wind_onshore_area_km2",
        "wind_offshore_area_km2",
        "wind_density_mw_per_km2",
        "solar_density_mw_per_km2",
        "solar_tracking_density_mw_per_km2",
        "renewable_union_area_km2",
        land_union.CLASSWISE_NESTED_UNION_AREA_COLUMN,
        "max_power_wind_mw",
        "max_power_solar_mw",
    ]

    capacities = [
        "wind_mw",
        "solar_mw",
        "solar_tracking_mw",
        "grid_mw",
        "ramp_dummy_mw",
        "electrolysis_mw",
        "hydrogen_compression_mw",
        "hydrogen_from_storage_mw",
        "ammonia_synthesis_mw",
        "battery_pcs_mw",
        "hydrogen_fuel_cell_mw",
        "penalty_link_mwh",
        "ammonia_mwh",
        "compressed_hydrogen_store_mwh",
        "battery_storage_mwh",
        "accumulated_penalty_mwh",
        "hydrogen_storage_capacity_t",
    ]

    component_costs = [
        col
        for col in df.columns
        if (col.startswith("tech_share_") or col.startswith("lcoa_tech_"))
        and col != "tech_share_other_pct"
    ]
    interest_rates = [col for col in df.columns if col.startswith("interest_rate_")]
    tail = ["interest_overrides_applied"]

    ordered = []
    for section in [headline, land_water, capacities, component_costs, interest_rates, tail]:
        for col in section:
            if col in df.columns and col not in ordered:
                ordered.append(col)

    remaining = [col for col in df.columns if col not in ordered]
    return df[ordered + remaining]



class _ProgressTracker:
    def __init__(self, total: int | None):
        self.total = total
        self.count = 0
        self.bar_width = 28
        self._supports_inline = bool(getattr(sys.stderr, "isatty", lambda: False)())
        self._emit_every = 1
        if self.total and self.total > 0:
            # In non-TTY contexts (e.g., notebooks), emit roughly every 1%.
            self._emit_every = max(1, self.total // 100)

    def update(self, lat: float, lon: float, status: str = "") -> None:
        self.count += 1
        stream = sys.stderr if self._supports_inline else sys.stdout
        if self.total and self.total > 0:
            filled = int(self.bar_width * self.count / self.total)
            bar = "#" * filled + "-" * (self.bar_width - filled)
            percent = self.count / self.total * 100.0
            prefix = f"[{bar}] {self.count}/{self.total} ({percent:5.1f}%)"
            newline = self.count >= self.total
        else:
            prefix = f"[{self.count}]"
            newline = True
        suffix = status.upper() if status else ""
        message = f"{prefix} {suffix:>8} lat={lat:7.2f} lon={lon:7.2f}"
        if self.total and self.total > 0 and self._supports_inline:
            stream.write("\r" + message)
            if newline:
                stream.write("\n")
        else:
            should_emit = (
                self.total is None
                or self.count == 1
                or (self.total is not None and self.count >= self.total)
                or self.count % self._emit_every == 0
                or (status and status.lower() != "done")
            )
            if should_emit:
                stream.write(message + "\n")
        stream.flush()


def _resolve_path(value: str | Path | None) -> Path | None:
    if value is None:
        return None
    path = Path(value)
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path


@contextmanager
def _quiet_logging(enabled: bool):
    if not enabled:
        yield
        return
    previous_disable_level = logging.root.manager.disable
    # Disable WARNING and below so the progress bar doesn't get interleaved with log output.
    logging.disable(logging.WARNING)
    logger_names = [
        "gurobipy",
        "linopy",
        "linopy.constants",
        "linopy.io",
        "linopy.model",
        "model.location_tools",
        "model.run_global",
        "pypsa",
        "pypsa.network",
        "pypsa.network.io",
        "pypsa.optimization",
        "pypsa.optimization.constraints",
        "pypsa.optimization.optimize",
    ]
    previous_levels: Dict[str, int] = {}
    for name in logger_names:
        logger = logging.getLogger(name)
        previous_levels[name] = logger.level
        logger.setLevel(logging.ERROR)
    try:
        yield
    finally:
        for name, level in previous_levels.items():
            logging.getLogger(name).setLevel(level)
        logging.disable(previous_disable_level)


@contextmanager
def _suppress_solver_streams(enabled: bool):
    if not enabled:
        yield
        return
    stdout_buffer = io.StringIO()
    stderr_buffer = io.StringIO()
    with redirect_stdout(stdout_buffer), redirect_stderr(stderr_buffer):
        yield


@contextmanager
def _override_env(var: str, value: str, enabled: bool):
    if not enabled:
        yield
        return
    previous = os.environ.get(var)
    os.environ[var] = value
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop(var, None)
        else:
            os.environ[var] = previous


def _load_interest_table(path: str | Path | None) -> pd.DataFrame:
    resolved = _resolve_path(path)
    if resolved is None:
        return pd.DataFrame(columns=["lat", "lon", "tech", "interest_rate"])
    if not resolved.exists():
        raise FileNotFoundError(f"Requested override CSV not found: {resolved}")
    df = pd.read_csv(resolved)
    column_map = {col.lower(): col for col in df.columns}
    for required in ("lat", "lon", "tech", "interest_rate"):
        if required not in column_map:
            raise ValueError(
                f"Interest CSV {resolved} must include '{required}' column (found {list(df.columns)})"
            )
    cleaned = df.rename(columns={column_map[k]: k for k in ("lat", "lon", "tech", "interest_rate")})
    cleaned["tech"] = cleaned["tech"].map(plant_main.normalize_component_name)
    return cleaned


def _load_land_table(path: str | Path | None) -> pd.DataFrame | None:
    resolved = _resolve_path(path)
    if resolved is None:
        return None
    if not resolved.exists():
        raise FileNotFoundError(f"Max-capacities CSV not found: {resolved}")
    df = pd.read_csv(resolved)
    df.columns = [col.lower() for col in df.columns]
    expected = {"latitude", "longitude"}
    if not expected.issubset(df.columns):
        LOGGER.warning("Land CSV %s is missing %s", resolved, expected - set(df.columns))
        return None
    # Backfill elevation_m from bathymetry NetCDF when the CSV column is absent or all-NaN.
    if "elevation_m" not in df.columns or df["elevation_m"].isna().all():
        bathymetry_path = data_paths.BATHYMETRY_FILE
        if bathymetry_path.exists():
            # Sample at the cell centre implied by the table's own anchoring.
            anchor = (
                str(df["cell_anchor"].dropna().iloc[0])
                if "cell_anchor" in df.columns and df["cell_anchor"].notna().any()
                else "southwest"
            )
            df = land_processing._attach_bathymetry_depth(df, bathymetry_path, 1.0, anchor)
            LOGGER.info("Backfilled elevation_m from %s", bathymetry_path)
    return df


def _validate_versioned_union_land_table(
    frame: pd.DataFrame | None,
    path: Path,
) -> None:
    """Require the publication-grade, versioned v1 union lower bound."""

    if frame is None:
        raise ValueError(f"Land table could not be loaded: {path}")
    required = {
        land_union.CLASSWISE_NESTED_UNION_AREA_COLUMN,
        land_union.RENEWABLE_UNION_METHOD_COLUMN,
        land_union.RENEWABLE_UNION_METHOD_VERSION_COLUMN,
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(
            f"Land table {path} lacks versioned renewable-union provenance: "
            + ", ".join(missing)
        )

    methods = set(frame[land_union.RENEWABLE_UNION_METHOD_COLUMN].dropna().astype(str))
    versions = set(
        frame[land_union.RENEWABLE_UNION_METHOD_VERSION_COLUMN].dropna().astype(str)
    )
    if methods != {land_union.CLASSWISE_NESTED_UNION_METHOD}:
        raise ValueError(
            f"Land table {path} has unexpected renewable-union methods: {sorted(methods)}"
        )
    if versions != {land_union.CLASSWISE_NESTED_UNION_VERSION}:
        raise ValueError(
            f"Land table {path} has unexpected renewable-union versions: {sorted(versions)}"
        )

    versioned = pd.to_numeric(
        frame[land_union.CLASSWISE_NESTED_UNION_AREA_COLUMN], errors="coerce"
    )
    if versioned.isna().any() or (versioned < 0.0).any():
        raise ValueError(
            f"Land table {path} contains invalid values in "
            f"{land_union.CLASSWISE_NESTED_UNION_AREA_COLUMN}"
        )
    if "renewable_union_area_km2" in frame.columns:
        generic = pd.to_numeric(frame["renewable_union_area_km2"], errors="coerce")
        if generic.isna().any() or not np.allclose(
            versioned.to_numpy(), generic.to_numpy(), rtol=0.0, atol=1e-9
        ):
            raise ValueError(
                f"Land table {path} has a generic renewable-union alias that "
                "diverges from the versioned v1 column"
            )


def _solar_base_land_use_from_tech_inputs(tech_inputs: Dict[str, dict]) -> float | None:
    for key in ("solar", "solar_tracking"):
        raw = tech_inputs.get(key)
        if not isinstance(raw, dict):
            continue
        value = raw.get("land_use_km2_per_mw")
        if value is None:
            continue
        base_land_use = float(value)
        if base_land_use > 0:
            return base_land_use
    return None


def _solar_land_uses_from_tech_inputs(
    tech_inputs: Dict[str, dict],
) -> tuple[float | None, float | None]:
    def _land_use(name: str) -> float | None:
        raw = tech_inputs.get(name)
        if not isinstance(raw, dict):
            return None
        value = raw.get("land_use_km2_per_mw")
        if value is None or float(value) <= 0:
            return None
        return float(value)

    return _land_use("solar"), _land_use("solar_tracking")


def _apply_spatial_solar_density_from_tech_config(
    land_df: pd.DataFrame | None,
    tech_inputs: Dict[str, dict],
) -> pd.DataFrame | None:
    if land_df is None or land_df.empty:
        return land_df
    if "latitude" not in land_df.columns:
        return land_df

    updated = land_df.copy()
    if "solar_density_method" in updated.columns:
        methods = set(updated["solar_density_method"].fillna("").astype(str))
        if methods != {"explicit_fixed_pv_density_v1"}:
            raise ValueError(f"Unknown or mixed explicit solar density methods: {methods}")
        if "solar_density_mw_per_km2" not in updated.columns:
            raise ValueError("Explicit solar density method requires solar_density_mw_per_km2")
        density = pd.to_numeric(updated["solar_density_mw_per_km2"], errors="coerce")
        if not np.isfinite(density).all() or (density <= 0).any():
            raise ValueError("Explicit fixed-PV densities must be finite and positive")
        # New versioned inputs own their geometry/normalization. Do not silently
        # replace their density with the historical latitude-packing equation.
        updated["solar_density_mw_per_km2"] = density
    else:
        # Preserve the existing contract for all unversioned historical inputs.
        base_land_use = _solar_base_land_use_from_tech_inputs(tech_inputs)
        if base_land_use is None:
            return land_df
        updated["solar_density_mw_per_km2"] = land_processing._solar_density(
            updated["latitude"],
            scale=1.0,
            base_land_use_km2_per_mw=base_land_use,
        )
    fixed_land_use, tracking_land_use = _solar_land_uses_from_tech_inputs(tech_inputs)
    if fixed_land_use is not None and tracking_land_use is not None:
        updated["solar_tracking_density_mw_per_km2"] = (
            updated["solar_density_mw_per_km2"]
            * fixed_land_use
            / tracking_land_use
        )

    if "solar_area_km2" in updated.columns:
        updated["max_power_solar_mw"] = (
            updated["solar_area_km2"] * updated["solar_density_mw_per_km2"]
        ).clip(lower=0.0)
        if "max_power_wind_mw" in updated.columns:
            updated["max_capacity_mw"] = (
                updated["max_power_wind_mw"]
                + updated["max_power_solar_mw"]
            )

    return updated


def _read_country_file(path: str | Path) -> gpd.GeoDataFrame | None:
    try:
        frame = gpd.read_file(path)
    except Exception as exc:  # noqa: BLE001
        LOGGER.debug("Failed to read country boundaries from %s: %s", path, exc)
        return None
    rename_map = {}
    for candidate in ["ADMIN", "NAME", "name", "COUNTRY", "Country"]:
        if candidate in frame.columns:
            rename_map[candidate] = "country"
            break
    if rename_map:
        frame = frame.rename(columns=rename_map)
    if "country" not in frame.columns:
        LOGGER.debug("Country column missing in %s (columns=%s)", path, list(frame.columns))
        return None
    return frame[["country", "geometry"]]


def _build_interest_lookup(interest_df: pd.DataFrame) -> Dict[Tuple[float, float], Dict[str, Dict[str, float]]]:
    """Pre-index overrides CSV into ``{(lat, lon): overrides_dict}`` for O(1) lookup.

    The returned structure mirrors what ``_interest_overrides`` would produce
    per location, but computed once instead of scanning the full DataFrame for
    every cell.
    """
    if interest_df.empty:
        return {}

    lookup: Dict[Tuple[float, float], Dict[str, Dict[str, float]]] = {}

    for _, row in interest_df.iterrows():
        key = (round(float(row["lat"]), 4), round(float(row["lon"]), 4))
        entry = lookup.setdefault(key, {})
        tech = row["tech"]
        entry.setdefault(tech, {})

        if "interest_rate" in row.index and pd.notna(row["interest_rate"]):
            entry[tech]["interest_rate"] = float(row["interest_rate"])
        if "build_cost_multiplier" in row.index and pd.notna(row["build_cost_multiplier"]):
            entry[tech]["build_cost_multiplier"] = float(row["build_cost_multiplier"])

        loc_params = entry.setdefault("__location__", {})
        if "land_cost_usd_per_km2_year" in row.index and pd.notna(row["land_cost_usd_per_km2_year"]):
            loc_params["land_cost_usd_per_km2_year"] = float(row["land_cost_usd_per_km2_year"])
        if "water_cost_usd_per_m3" in row.index and pd.notna(row["water_cost_usd_per_m3"]):
            loc_params["water_cost_usd_per_m3"] = float(row["water_cost_usd_per_m3"])

    # Remove empty __location__ entries
    for key in lookup:
        if not lookup[key].get("__location__"):
            lookup[key].pop("__location__", None)

    return lookup


def _interest_overrides_fast(
    lookup: Dict[Tuple[float, float], Dict[str, Dict[str, float]]],
    lat: float,
    lon: float,
) -> Dict[str, Dict[str, float]] | None:
    """O(1) interest override lookup using the pre-indexed dict."""
    key = (round(float(lat), 4), round(float(lon), 4))
    result = lookup.get(key)
    return result if result else None


def _interest_overrides(interest_df: pd.DataFrame, lat: float, lon: float) -> Dict[str, Dict[str, float]] | None:
    """Load interest_rate, build_cost_multiplier, and spatial cost parameters from overrides CSV.
    
    Returns dict with structure:
    {
        'solar': {'interest_rate': 0.058, 'build_cost_multiplier': 1.2, ...},
        ...
    }
    Location-level params (water_cost_usd_per_m3, land_cost_usd_per_km2_year) are 
    stored once per location and shared across all techs.
    """
    if interest_df.empty:
        return None
    mask = (interest_df["lat"] - lat).abs() <= _LAT_LON_TOLERANCE
    mask &= (interest_df["lon"] - lon).abs() <= _LAT_LON_TOLERANCE
    subset = interest_df.loc[mask]
    if subset.empty:
        return None
    
    overrides: Dict[str, Dict[str, float]] = {}
    location_params: Dict[str, float] = {}
    
    for _, row in subset.iterrows():
        tech = row["tech"]
        overrides.setdefault(tech, {})
        
        # Tech-specific parameters
        if "interest_rate" in row and pd.notna(row["interest_rate"]):
            overrides[tech]["interest_rate"] = float(row["interest_rate"])
        
        if "build_cost_multiplier" in row and pd.notna(row["build_cost_multiplier"]):
            overrides[tech]["build_cost_multiplier"] = float(row["build_cost_multiplier"])
        
        # Location-level parameters (store once, will be reused for all techs at this location)
        if "land_cost_usd_per_km2_year" in row and pd.notna(row["land_cost_usd_per_km2_year"]):
            location_params["land_cost_usd_per_km2_year"] = float(row["land_cost_usd_per_km2_year"])
        
        if "water_cost_usd_per_m3" in row and pd.notna(row["water_cost_usd_per_m3"]):
            location_params["water_cost_usd_per_m3"] = float(row["water_cost_usd_per_m3"])
    
    # Store location params in a special key that won't conflict with tech names
    if location_params:
        overrides["__location__"] = location_params
    
    return overrides or None


def _build_land_lookup(land_df: pd.DataFrame | None) -> Dict[Tuple[float, float], pd.Series]:
    """Pre-index the max-capacities DataFrame into ``{(lat, lon): row}`` for O(1) lookup."""
    if land_df is None:
        return {}
    lookup: Dict[Tuple[float, float], pd.Series] = {}
    for idx, row in land_df.iterrows():
        key = (round(float(row["latitude"]), 4), round(float(row["longitude"]), 4))
        lookup[key] = row
    return lookup


def _match_land_row_fast(
    lookup: Dict[Tuple[float, float], pd.Series],
    lat: float,
    lon: float,
) -> pd.Series | None:
    """O(1) land row lookup using the pre-indexed dict."""
    key = (round(float(lat), 4), round(float(lon), 4))
    return lookup.get(key)


def _match_land_row(land_df: pd.DataFrame, lat: float, lon: float) -> pd.Series | None:
    if land_df is None:
        return None
    mask = (land_df["latitude"] - lat).abs() <= _LAT_LON_TOLERANCE
    mask &= (land_df["longitude"] - lon).abs() <= _LAT_LON_TOLERANCE
    subset = land_df.loc[mask]
    if subset.empty:
        return None
    return subset.iloc[0]

def _set_component_cap(network, tech_name: str, cap_value: float) -> float:
    if cap_value is None or not np.isfinite(float(cap_value)):
        return 0.0
    cap = max(0.0, float(cap_value))

    if tech_name in network.generators.index:
        if not bool(network.generators.at[tech_name, "p_nom_extendable"]):
            return 0.0
        network.generators.loc[tech_name, "p_nom_max"] = cap
        return cap

    if tech_name in network.links.index:
        if not bool(network.links.at[tech_name, "p_nom_extendable"]):
            return 0.0
        network.links.loc[tech_name, "p_nom_max"] = cap
        return cap

    if tech_name in network.stores.index:
        if not bool(network.stores.at[tech_name, "e_nom_extendable"]):
            return 0.0
        network.stores.loc[tech_name, "e_nom_max"] = cap
        return cap

    return 0.0


def _extract_capacity_caps_from_row(row: pd.Series) -> Tuple[Dict[str, float], Dict[str, float]]:
    power_caps: Dict[str, float] = {}
    energy_caps: Dict[str, float] = {}

    for column in row.index:
        value = row.get(column)
        if pd.isna(value):
            continue
        key = str(column).strip().lower()
        if key.startswith("max_power_") and key.endswith("_mw"):
            tech = key[len("max_power_") : -len("_mw")]
            if tech:
                power_caps[tech] = max(0.0, float(value))
            continue
        if key.startswith("max_energy_") and key.endswith("_mwh"):
            tech = key[len("max_energy_") : -len("_mwh")]
            if tech:
                energy_caps[tech] = max(0.0, float(value))

    return power_caps, energy_caps


def _apply_component_caps(
    network,
    power_caps: Dict[str, float],
    energy_caps: Dict[str, float],
) -> float:
    setattr(network, "_shared_solar_cap_mw", None)
    setattr(network, "_shared_wind_cap_mw", None)
    total_cap = 0.0

    solar_cap = power_caps.get("solar")
    solar_tracking_cap = power_caps.get("solar_tracking")
    shared_wind_cap = power_caps.get("wind")

    if shared_wind_cap is not None:
        setattr(network, "_shared_wind_cap_mw", max(0.0, float(shared_wind_cap)))

    if solar_cap is not None and solar_tracking_cap is None:
        _set_component_cap(network, "solar", solar_cap)
        _set_component_cap(network, "solar_tracking", solar_cap)
        setattr(network, "_shared_solar_cap_mw", float(solar_cap))
    elif solar_tracking_cap is not None and solar_cap is None:
        _set_component_cap(network, "solar", solar_tracking_cap)
        _set_component_cap(network, "solar_tracking", solar_tracking_cap)
        setattr(network, "_shared_solar_cap_mw", float(solar_tracking_cap))
    else:
        if solar_cap is not None:
            _set_component_cap(network, "solar", solar_cap)
        if solar_tracking_cap is not None:
            _set_component_cap(network, "solar_tracking", solar_tracking_cap)
        if (
            solar_cap is not None
            and solar_tracking_cap is not None
            and abs(float(solar_cap) - float(solar_tracking_cap)) < 1e-9
        ):
            setattr(network, "_shared_solar_cap_mw", float(solar_cap))

    for tech, cap in power_caps.items():
        if tech in {"solar", "solar_tracking", "wind"}:
            continue
        _set_component_cap(network, tech, cap)

    for tech, cap in energy_caps.items():
        _set_component_cap(network, tech, cap)

    total_cap += sum(cap for tech, cap in power_caps.items() if tech != "wind")
    return total_cap


def _apply_renewable_land_budget(
    network,
    budget: land_capacity.RenewableLandBudget,
    land_allocation: land_capacity.LandAllocation,
) -> float:
    """Attach area-based land constraints and conservative component bounds."""
    land_allocation = land_capacity.normalise_land_allocation(land_allocation)

    setattr(network, "_renewable_land_budget", budget)
    setattr(network, "_land_allocation", land_allocation)
    # Disable the old MW-sum constraints; Linopy consumes the area budget above.
    setattr(network, "_shared_solar_cap_mw", None)
    setattr(network, "_shared_wind_cap_mw", None)

    wind_area = budget.wind_area_km2
    solar_area = budget.solar_area_km2

    total_cap = 0.0
    component_specs = (
        ("wind", wind_area, budget.wind_density_mw_per_km2),
        ("solar", solar_area, budget.fixed_solar_density_mw_per_km2),
        (
            "solar_tracking",
            solar_area,
            budget.tracking_solar_density_mw_per_km2,
        ),
    )
    for component, area_km2, density_mw_per_km2 in component_specs:
        if area_km2 is None or density_mw_per_km2 is None:
            continue
        cap_mw = max(0.0, float(area_km2)) * max(0.0, float(density_mw_per_km2))
        total_cap += _set_component_cap(network, component, cap_mw)
    return total_cap


def _apply_land_caps(
    network, land_df: pd.DataFrame | None, lat: float, lon: float
) -> Tuple[float | None, pd.Series | None]:
    row = _match_land_row(land_df, lat, lon) if land_df is not None else None
    if row is None:
        return None, None

    power_caps, energy_caps = _extract_capacity_caps_from_row(row)
    has_explicit_power_caps = any(
        str(col).lower().startswith("max_power_") and str(col).lower().endswith("_mw")
        for col in row.index
    )
    if has_explicit_power_caps:
        for tech in WIND_GENERATORS + ["solar"]:
            power_caps.setdefault(tech, 0.0)

    if power_caps or energy_caps:
        total_cap = _apply_component_caps(network, power_caps, energy_caps)
        return (total_cap if total_cap > 0 else None), row

    raise ValueError(
        f"Land CSV row at ({lat}, {lon}) must include explicit max_power_*_mw or max_energy_*_mwh columns."
    )


def _apply_land_caps_fast(
    network,
    land_lookup: Dict[Tuple[float, float], pd.Series],
    lat: float,
    lon: float,
) -> Tuple[float | None, pd.Series | None]:
    """Same as ``_apply_land_caps`` but uses the pre-indexed lookup dict."""
    row = _match_land_row_fast(land_lookup, lat, lon)
    if row is None:
        return None, None

    power_caps, energy_caps = _extract_capacity_caps_from_row(row)
    has_explicit_power_caps = any(
        str(col).lower().startswith("max_power_") and str(col).lower().endswith("_mw")
        for col in row.index
    )
    if has_explicit_power_caps:
        for tech in WIND_GENERATORS + ["solar"]:
            power_caps.setdefault(tech, 0.0)

    if power_caps or energy_caps:
        total_cap = _apply_component_caps(network, power_caps, energy_caps)
        return (total_cap if total_cap > 0 else None), row

    raise ValueError(
        f"Land CSV row at ({lat}, {lon}) must include explicit max_power_*_mw or max_energy_*_mwh columns."
    )


def _land_metadata_from_row(row: pd.Series | None) -> Dict[str, float]:
    if row is None:
        return {}

    metadata: Dict[str, float] = {}

    def _maybe(name: str, scale: float = 1.0) -> float | None:
        value = row.get(name)
        if pd.isna(value):
            return None
        return float(value) * scale

    onshore = _maybe("onshore_land_pct", 1.0)
    if onshore is not None:
        metadata["land_onshore_pct"] = onshore
        metadata["offshore_sea_pct"] = max(0.0, 100.0 - onshore)

    area = _maybe("area")
    if area is not None:
        metadata["land_cell_area_km2"] = area

    onshore_area = _maybe("onshore_area_km2")
    if onshore_area is not None:
        metadata["onshore_area_km2"] = onshore_area

    offshore_area = _maybe("offshore_area_km2")
    if offshore_area is not None:
        metadata["offshore_area_km2"] = offshore_area

    bathymetry_depth = _maybe("elevation_m")
    if bathymetry_depth is not None:
        metadata["elevation_m"] = bathymetry_depth

    max_power_solar_mw = _maybe("max_power_solar_mw")
    max_power_wind_mw = _maybe("max_power_wind_mw")
    if max_power_solar_mw is not None or max_power_wind_mw is not None:
        metadata["area_cap_mw"] = (max_power_solar_mw or 0.0) + (max_power_wind_mw or 0.0)

    solar_area = _maybe("solar_area_km2")
    if solar_area is not None:
        metadata["solar_area_used_km2"] = solar_area
        metadata["solar_available_area_km2"] = solar_area

    wind_area = _maybe("wind_area_km2")
    if wind_area is not None:
        metadata["wind_area_used_km2"] = wind_area
        metadata["wind_available_area_km2"] = wind_area

    for column in [
        "protected_area_pct",
        "slope_suitable_land_pct",
        "steep_slope_pct",
        "land_exclusion_factor",
        "land_competition_fraction",
        "constrained_onshore_area_km2",
        "wind_onshore_area_km2",
        "wind_offshore_area_km2",
        "wind_density_mw_per_km2",
        "solar_density_mw_per_km2",
        "solar_tracking_density_mw_per_km2",
        "renewable_union_area_km2",
        "renewable_union_availability",
        land_union.CLASSWISE_NESTED_UNION_AREA_COLUMN,
    ]:
        value = _maybe(column)
        if value is not None:
            metadata[column] = value

    for column in row.index:
        key = str(column).strip().lower()
        if key.startswith("max_power_") and key.endswith("_mw"):
            value = _maybe(key)
            if value is not None:
                metadata[key] = value
        if key.startswith("max_energy_") and key.endswith("_mwh"):
            value = _maybe(key)
            if value is not None:
                metadata[key] = value

    return metadata


def _numeric_from_mapping(mapping: Dict[str, Any] | pd.Series, *names: str) -> float | None:
    for name in names:
        try:
            value = mapping.get(name)  # type: ignore[attr-defined]
        except AttributeError:
            value = None
        if value is None or pd.isna(value):
            continue
        try:
            return float(value)
        except (TypeError, ValueError):
            continue
    return None


def _bounded_fraction(value: float) -> float:
    return float(np.clip(float(value), 0.0, 1.0))


def _gridless_fraction_from_results(results: Dict[str, Any], annual_t: float) -> float:
    gridless_t = _numeric_from_mapping(results, "gridless_ammonia_production_t")
    if gridless_t is not None:
        return _bounded_fraction(gridless_t / annual_t)

    gridless_fraction = _numeric_from_mapping(results, "gridless_energy_fraction")
    if gridless_fraction is not None:
        return _bounded_fraction(gridless_fraction)

    grid_energy_mwh = _numeric_from_mapping(results, "grid_energy_mwh")
    if grid_energy_mwh is not None and grid_energy_mwh <= 1e-9:
        return 1.0

    grid_mw = _numeric_from_mapping(results, "grid_mw", "grid")
    if grid_mw is not None and grid_mw <= 1e-9:
        return 1.0

    return math.nan


def _renewable_capacity_scale(
    wind_used_mw: float,
    solar_used_mw: float,
    wind_cap_mw: float | None,
    solar_cap_mw: float | None,
) -> tuple[float, str]:
    scale_limits: list[tuple[str, float]] = []
    missing_required_cap = False

    if wind_used_mw > 1e-9:
        if wind_cap_mw is None:
            missing_required_cap = True
        else:
            scale_limits.append(("wind", max(0.0, float(wind_cap_mw)) / wind_used_mw))

    if solar_used_mw > 1e-9:
        if solar_cap_mw is None:
            missing_required_cap = True
        else:
            scale_limits.append(("solar", max(0.0, float(solar_cap_mw)) / solar_used_mw))

    if missing_required_cap:
        return math.nan, "unknown"
    if not scale_limits:
        return 1.0, "none"

    technology, scale_factor = min(scale_limits, key=lambda item: item[1])
    return max(0.0, float(scale_factor)), technology


def _onshore_wind_cap_from_land_row(land_row: pd.Series | None) -> float | None:
    if land_row is None:
        return None

    wind_onshore_area_km2 = _numeric_from_mapping(land_row, "wind_onshore_area_km2")
    wind_density_mw_per_km2 = _numeric_from_mapping(land_row, "wind_density_mw_per_km2")
    if wind_onshore_area_km2 is None or wind_density_mw_per_km2 is None:
        return None

    return max(0.0, float(wind_onshore_area_km2)) * max(0.0, float(wind_density_mw_per_km2))


def _capacity_from_scale(annual_t: float, scale_factor: float) -> float:
    if not math.isfinite(scale_factor):
        return math.nan
    return annual_t * scale_factor


def _estimate_paper_ammonia_capacity(
    results: Dict[str, Any],
    land_row: pd.Series | None,
) -> Dict[str, Any]:
    """Estimate paper-style land-constrained ammonia capacity from solved sizing.

    The solved renewable capacities encode the full plant efficiency chain, so this
    scales ammonia output by available renewable build-out rather than recreating
    electrolyser, storage, Haber-Bosch, or ASU conversion logic.
    """
    output: Dict[str, Any] = {
        "max_ammonia_capacity_t": math.nan,
        "max_ammonia_capacity_mtpa": math.nan,
        "max_onshore_ammonia_capacity_t": math.nan,
        "max_onshore_ammonia_capacity_mtpa": math.nan,
        "max_gridless_onshore_ammonia_capacity_t": math.nan,
        "max_gridless_onshore_ammonia_capacity_mtpa": math.nan,
        "capacity_limit_technology": "unknown",
        "onshore_capacity_limit_technology": "unknown",
        "renewable_capacity_scale_factor": math.nan,
        "onshore_renewable_capacity_scale_factor": math.nan,
        "wind_mw_per_t_nh3": math.nan,
        "solar_mw_per_t_nh3": math.nan,
    }

    annual_t = _numeric_from_mapping(results, "annual_ammonia_production_t")
    if annual_t is None or annual_t <= 0:
        for column in [
            "max_ammonia_capacity_t",
            "max_ammonia_capacity_mtpa",
            "max_onshore_ammonia_capacity_t",
            "max_onshore_ammonia_capacity_mtpa",
            "max_gridless_onshore_ammonia_capacity_t",
            "max_gridless_onshore_ammonia_capacity_mtpa",
        ]:
            output[column] = 0.0
        return output

    annual_t = float(annual_t)
    wind_used_mw = max(0.0, _numeric_from_mapping(results, "wind_mw", "wind") or 0.0)
    solar_used_mw = max(
        0.0,
        (_numeric_from_mapping(results, "solar_mw", "solar") or 0.0)
        + (_numeric_from_mapping(results, "solar_tracking_mw", "solar_tracking") or 0.0),
    )

    output["wind_mw_per_t_nh3"] = wind_used_mw / annual_t
    output["solar_mw_per_t_nh3"] = solar_used_mw / annual_t

    wind_cap_mw = None
    solar_cap_mw = None
    if land_row is not None:
        wind_cap_mw = _numeric_from_mapping(land_row, "max_power_wind_mw")
        solar_cap_mw = _numeric_from_mapping(land_row, "max_power_solar_mw")

    scale_factor, limit_technology = _renewable_capacity_scale(
        wind_used_mw,
        solar_used_mw,
        wind_cap_mw,
        solar_cap_mw,
    )
    max_capacity_t = _capacity_from_scale(annual_t, scale_factor)

    output["renewable_capacity_scale_factor"] = scale_factor
    output["capacity_limit_technology"] = limit_technology
    output["max_ammonia_capacity_t"] = max_capacity_t
    output["max_ammonia_capacity_mtpa"] = max_capacity_t / 1_000_000.0

    wind_onshore_cap_mw = _onshore_wind_cap_from_land_row(land_row)
    onshore_scale_factor, onshore_limit_technology = _renewable_capacity_scale(
        wind_used_mw,
        solar_used_mw,
        wind_onshore_cap_mw,
        solar_cap_mw,
    )
    max_onshore_capacity_t = _capacity_from_scale(annual_t, onshore_scale_factor)

    output["onshore_renewable_capacity_scale_factor"] = onshore_scale_factor
    output["onshore_capacity_limit_technology"] = onshore_limit_technology
    output["max_onshore_ammonia_capacity_t"] = max_onshore_capacity_t
    output["max_onshore_ammonia_capacity_mtpa"] = max_onshore_capacity_t / 1_000_000.0

    gridless_fraction = _gridless_fraction_from_results(results, annual_t)
    if math.isfinite(max_onshore_capacity_t) and math.isfinite(gridless_fraction):
        gridless_onshore_capacity_t = max_onshore_capacity_t * gridless_fraction
    else:
        gridless_onshore_capacity_t = math.nan

    output["max_gridless_onshore_ammonia_capacity_t"] = gridless_onshore_capacity_t
    output["max_gridless_onshore_ammonia_capacity_mtpa"] = (
        gridless_onshore_capacity_t / 1_000_000.0
    )
    return output


def _compose_interest_rates(
    overrides: Dict[str, Dict[str, float]] | None,
) -> Dict[str, float]:
    rates: Dict[str, float] = {k: v for k, v in DEFAULT_INTEREST_RATES.items() if v is not None}
    if overrides:
        for tech, params in overrides.items():
            rate = params.get("interest_rate")
            if rate is None:
                continue
            normalized = plant_main.normalize_component_name(tech)
            if normalized is None:
                continue
            rates[normalized] = float(rate)
    return rates


def _load_country_shapes() -> gpd.GeoDataFrame:
    fallback = REPO_ROOT / "data" / "countries.geojson"
    if not fallback.exists():
        raise FileNotFoundError(
            "Missing data/countries.geojson – add a Natural Earth admin-0 GeoJSON to tag locations."
        )
    frame = _read_country_file(fallback)
    if frame is None:
        raise ValueError(
            "Unable to parse data/countries.geojson. Ensure it includes a 'country'/'ADMIN' column and valid geometries."
        )
    return frame


def _country_for(world: gpd.GeoDataFrame, lat: float, lon: float) -> str:
    return lt.country_for_cell_overlap(world, lat, lon)


def _build_country_lookup(
    world: gpd.GeoDataFrame,
    locations: List[Tuple[float, float]],
) -> Dict[Tuple[float, float], str]:
    """Batch-assign countries using the largest country overlap per cell footprint."""
    return lt.build_country_lookup_by_cell_overlap(world, locations)


def _build_weather_frame(dataset: lt.all_locations, lat: float, lon: float, aggregation_count: int) -> pd.DataFrame:
    """Extract weather profiles directly from cached xarray data.

    This bypasses the ``renewable_data`` class to avoid per-cell overhead
    (DataFrame copies, aggregation with count=1, etc.).
    """
    sel = dict(latitude=lat, longitude=lon)
    columns: dict[str, np.ndarray] = {}
    wake_loss = 0.93
    for resource in ("solar", "wind", "solar_tracking"):
        arr = dataset.resources.get(resource)
        if arr is None:
            continue
        values = arr.sel(sel, method="nearest").values
        if resource == "wind":
            values = values * wake_loss
        columns[resource] = values

    if not columns:
        raise ValueError(f"No weather resources for ({lat}, {lon})")

    frame = pd.DataFrame(columns)

    if "wind" not in frame.columns and "onshore_wind" in frame.columns:
        frame["wind"] = frame["onshore_wind"]
    if "solar_tracking" not in frame.columns and "solar" in frame.columns:
        frame["solar_tracking"] = frame["solar"]
    frame["grid"] = 0.0
    frame["ramp_dummy"] = 1.0
    return frame


def _resample_weather_frame(weather_frame: pd.DataFrame, timestep_hours: int) -> pd.DataFrame:
    """Downsample an hourly weather frame to a coarser timestep by block-averaging.

    For capacity-factor columns (solar, wind, solar_tracking) the mean of each
    block gives the correct average power availability.  Constant columns
    (grid=0, ramp_dummy=1) pass through unchanged.

    Args:
        weather_frame: Hourly weather DataFrame (8760 rows typically).
        timestep_hours: Target resolution in hours (e.g. 3 for 3-hourly).

    Returns:
        Resampled DataFrame with ``len(weather_frame) // timestep_hours`` rows.
    """
    if timestep_hours <= 1:
        return weather_frame
    n = len(weather_frame)
    n_out = n // timestep_hours
    trimmed_len = n_out * timestep_hours
    result: dict[str, np.ndarray] = {}
    for col in weather_frame.columns:
        arr = weather_frame[col].values[:trimmed_len]
        result[col] = arr.reshape(n_out, timestep_hours).mean(axis=1)
    return pd.DataFrame(result)


def _default_locations(land_df: pd.DataFrame | None) -> List[Tuple[float, float]]:
    """Return (lat, lon) pairs for all cells with positive capacity.

    Filters on ``max_capacity_mw > 0``.
    Ocean cells are included — offshore cost premiums are handled via
    ``build_cost_multiplier`` in the spatial cost inputs CSV.
    """
    if land_df is None:
        raise ValueError("A max-capacities CSV is required when no explicit locations are supplied.")
    if "max_capacity_mw" in land_df.columns:
        filtered = land_df[land_df["max_capacity_mw"] > 0]
    elif "availability" in land_df.columns:
        filtered = land_df[land_df["availability"] > 0]
    else:
        filtered = land_df

    return list(zip(filtered["latitude"].astype(float), filtered["longitude"].astype(float)))


def _estimate_land_used_km2(
    network,
    land_row: pd.Series | None,
    tech_inputs: Dict[str, dict],
) -> float | None:
    if not tech_inputs or land_row is None:
        return None
    budget = land_capacity.build_renewable_land_budget(
        land_row,
        tech_inputs,
        onshore=True,
        allow_conservative_union_fallback=True,
    )
    capacities = {
        name: (
            max(0.0, float(network.generators.at[name, "p_nom_opt"]))
            if name in network.generators.index
            and pd.notna(network.generators.at[name, "p_nom_opt"])
            else 0.0
        )
        for name in ("wind", "solar", "solar_tracking")
    }
    used = land_capacity.renewable_land_use_from_capacities(
        wind_mw=capacities["wind"],
        solar_mw=capacities["solar"],
        solar_tracking_mw=capacities["solar_tracking"],
        budget=budget,
    )
    return used.total_km2


def _run_single_location(
    lat: float,
    lon: float,
    dataset: lt.all_locations,
    interest_lookup: Dict[Tuple[float, float], Dict[str, Dict[str, float]]],
    tech_inputs: Dict[str, dict],
    tech_meta: Dict[str, Any],
    land_lookup: Dict[Tuple[float, float], pd.Series],
    base_network,
    aggregation_count: int,
    time_step: float,
    max_snapshots: int | None,
    quiet: bool,
    land_constraint: str = "after_solve",
    capacity_rule: str = "scaled_reference_design",
    land_allocation: land_capacity.LandAllocation = "colocated",
    allow_conservative_union_fallback: bool = True,
    temporal_accounting_mode: str = "snapshot_weighted",
    ramp_limit_basis: str = "per_hour",
    include_site_costs: bool = True,
    fail_fast: bool = False,
    solver_options_override: dict | None = None,
) -> Tuple[str, Dict[str, Any] | None]:
    """Run optimisation for a single (lat, lon).  With fail_fast=True any exception
    propagates immediately instead of being swallowed — use for debugging.
    """
    t0 = time.perf_counter()

    try:
        weather_frame = _build_weather_frame(dataset, lat, lon, aggregation_count)
        # Resample to coarser timestep if requested (e.g. 3h blocks).
        if time_step > 1:
            weather_frame = _resample_weather_frame(weather_frame, int(time_step))
        if max_snapshots is not None:
            weather_frame = weather_frame.iloc[:max_snapshots].copy()
    except Exception as exc:  # noqa: BLE001
        if fail_fast:
            raise RuntimeError(f"Weather extraction failed at ({lat}, {lon}): {exc}") from exc
        LOGGER.warning("Skipping location (%s, %s) due to weather extraction error: %s", lat, lon, exc)
        return "weather", None

    t_weather = time.perf_counter()

    try:
        overrides = _interest_overrides_fast(interest_lookup, lat, lon)
        interest_rates = _compose_interest_rates(overrides)

        # Extract location-level spatial cost parameters from overrides.
        location_params = overrides.get("__location__", {}) if overrides else {}
        water_cost_usd_per_m3 = location_params.get("water_cost_usd_per_m3")
        land_cost_usd_per_km2_year = location_params.get("land_cost_usd_per_km2_year")

        # Apply YAML defaults if location-specific values are missing.
        if water_cost_usd_per_m3 is None:
            water_cost_usd_per_m3 = tech_meta.get("water_cost_baseline_usd_per_m3")
        water_usage_m3_per_t = tech_meta.get("water_usage_m3_per_t_nh3")

        # Deep-copy the pre-built base network instead of re-importing CSVs.
        network = copy.deepcopy(base_network)
        network._ramp_limit_basis = ramp_limit_basis
        network._currency = str(tech_meta.get("currency", "USD")).strip().upper()

        # Always apply YAML default costs first (the CSV bundle may be stale).
        _apply_yaml_base_costs(
            network,
            tech_inputs=tech_inputs,
            aggregation_count=aggregation_count,
            time_step=time_step,
            temporal_accounting_mode=temporal_accounting_mode,
        )
        # Then apply per-location finance overrides on top (if any).
        _apply_finance_overrides(
            network,
            tech_inputs=tech_inputs,
            overrides=overrides,
            aggregation_count=aggregation_count,
            time_step=time_step,
            temporal_accounting_mode=temporal_accounting_mode,
        )
        land_row = _match_land_row_fast(land_lookup, lat, lon)
        if land_row is None:
            raise ValueError(f"No land row found for ({lat}, {lon})")
        land_allocation = land_capacity.normalise_land_allocation(land_allocation)
        land_budget = land_capacity.build_renewable_land_budget(
            land_row,
            tech_inputs,
            onshore=True,
            allow_conservative_union_fallback=allow_conservative_union_fallback,
        )
        if land_constraint == "in_solve":
            land_cap = _apply_renewable_land_budget(
                network,
                land_budget,
                land_allocation,
            )
        elif land_constraint == "after_solve":
            land_cap = None
            # The reference design is intentionally independent of the land
            # table.  Remove finite placeholder bounds from the plant bundle so
            # they cannot masquerade as a land or resource constraint.
            for renewable in RENEWABLES:
                if (
                    renewable in network.generators.index
                    and bool(network.generators.at[renewable, "p_nom_extendable"])
                ):
                    network.generators.at[renewable, "p_nom_max"] = math.inf
        else:
            raise ValueError("land_constraint must be 'after_solve' or 'in_solve'")
    except Exception as exc:  # noqa: BLE001
        if fail_fast:
            raise RuntimeError(f"Network setup failed at ({lat}, {lon}): {exc}") from exc
        LOGGER.warning("Network setup failed for (%s, %s): %s", lat, lon, exc)
        return "setup", None

    t_setup = time.perf_counter()

    try:
        with _suppress_solver_streams(quiet):
            results = plant_main.main(
                n=network,
                weather_data=weather_frame,
                multi_site=True,
                aggregation_count=aggregation_count,
                time_step=time_step,
                interest_rates=interest_rates,
                water_cost_usd_per_m3=water_cost_usd_per_m3,
                water_usage_m3_per_t=water_usage_m3_per_t,
                land_cost_usd_per_km2_year=land_cost_usd_per_km2_year,
                land_used_km2=None,
                solver_options_override=solver_options_override,
            )
    except Exception as exc:  # noqa: BLE001
        if fail_fast:
            raise RuntimeError(f"Optimisation failed at ({lat}, {lon}): {exc}") from exc
        LOGGER.warning("Optimisation failed for (%s, %s): %s", lat, lon, exc)
        return "solver", None

    t_solve = time.perf_counter()

    renewable_land_use = land_capacity.renewable_land_use_from_results(results, land_budget)
    land_used_km2 = renewable_land_use.total_km2
    if land_used_km2 is not None:
        results["land_used_km2"] = float(land_used_km2)
        results["wind_land_used_km2"] = renewable_land_use.wind_km2
        results["fixed_solar_land_used_km2"] = renewable_land_use.fixed_solar_km2
        results["tracking_solar_land_used_km2"] = renewable_land_use.tracking_solar_km2
        results["solar_land_used_km2"] = renewable_land_use.solar_km2

    if land_cap is not None:
        results["land_capacity_cap"] = land_cap
        results["land_capacity_cap_mw"] = land_cap
    if land_row is not None:
        results.update(_land_metadata_from_row(land_row))
    results["land_constraint"] = land_constraint
    results["capacity_rule"] = capacity_rule
    results["land_allocation"] = land_allocation
    results["wind_land_exclusive_fraction_effective"] = (
        land_capacity.effective_wind_land_exclusive_fraction(land_allocation, land_budget)
    )
    results["temporal_accounting_mode"] = temporal_accounting_mode
    results["ramp_limit_basis"] = ramp_limit_basis
    results["scenario_id"] = os.environ.get("ARC_SCENARIO_ID", "adhoc")
    results["run_id"] = os.environ.get("ARC_RUN_ID", "adhoc")
    results["manifest_sha256"] = os.environ.get("ARC_MANIFEST_SHA256", "")
    results["interest_overrides_applied"] = bool(overrides)

    # Headline cost splits (no double counting).
    headline_splits = _compute_headline_splits(
        network,
        tech_inputs,
        overrides,
        aggregation_count=aggregation_count,
        time_step=time_step,
        temporal_accounting_mode=temporal_accounting_mode,
    )
    if headline_splits:
        results.update(headline_splits)

    output_currency = str(tech_meta.get("currency", results.get("currency", "USD"))).strip().upper()
    source_currency = str(tech_meta.get("spatial_cost_currency", "USD")).strip().upper()
    source_to_output_fx = tech_meta.get("spatial_cost_to_model_currency")
    if source_to_output_fx is None:
        if source_currency != output_currency:
            raise ValueError(
                "Tech YAML must define spatial_cost_to_model_currency when site-cost "
                "and model currencies differ"
            )
        source_to_output_fx = 1.0
    results = result_accounting.finalize_site_cost_accounting(
        results,
        output_currency=output_currency,
        source_currency=source_currency,
        source_to_output_fx=float(source_to_output_fx),
        water_cost_source_per_m3=water_cost_usd_per_m3,
        water_usage_m3_per_t_nh3=(
            water_usage_m3_per_t if water_cost_usd_per_m3 is not None else None
        ),
        land_cost_source_per_km2_year=(
            float(land_cost_usd_per_km2_year)
            if land_cost_usd_per_km2_year is not None
            else 0.0
        ),
        land_used_km2=land_used_km2,
        include_site_costs_in_headline=include_site_costs,
    )

    results = result_accounting.finalize_headline_cost_percentages(
        results,
        plant_split_percentages=headline_splits,
    )

    results.setdefault("grid_energy_mwh", 0.0)
    if "power_bus_generator_supply_mwh" not in results:
        raise ValueError(
            "Solved results are missing the electrical grid-share denominator "
            "'power_bus_generator_supply_mwh'"
        )
    grid_reference_mwh = float(results["power_bus_generator_supply_mwh"])
    results = result_accounting.finalize_grid_reporting(
        results,
        electrical_reference_energy_mwh=grid_reference_mwh,
        absolute_tolerance_mwh=1.0,
        relative_tolerance=1e-6,
    )

    if capacity_rule == "solved_quantity":
        if land_constraint != "in_solve":
            raise ValueError("solved_quantity requires land_constraint='in_solve'")
        results.update(land_capacity.report_land_feasible_quantity(
            results, land_budget, land_allocation=land_allocation,
        ))
        return "done", results
    if capacity_rule != "scaled_reference_design":
        raise ValueError(
            "Only capacity_rule='scaled_reference_design' is implemented in this campaign phase"
        )
    paper_capacity = land_capacity.estimate_scaled_design_capacity(
        results,
        land_budget,
        land_allocation=land_allocation,
    )
    results.update(paper_capacity.to_mapping())

    # Preserve the historical result contract separately.  In these legacy
    # aliases max_ammonia_capacity_* uses total wind (including offshore), while
    # max_onshore_* uses onshore wind; solar tracking is folded into the old
    # fixed-solar power cap.  Green Porpoise should use the explicit
    # scaled_design_* columns for corrected supplier bounds.
    results.update(_estimate_paper_ammonia_capacity(results, land_row))
    results["legacy_capacity_alias_method"] = (
        "historical_independent_total_vs_onshore_power_caps_v0"
    )
    results["preferred_supplier_capacity_column"] = (
        "scaled_design_max_gridless_onshore_ammonia_capacity_t"
    )

    t_end = time.perf_counter()
    LOGGER.debug(
        "Timing (%s, %s): weather=%.3fs setup=%.3fs solve=%.3fs post=%.3fs total=%.3fs",
        lat, lon,
        t_weather - t0,
        t_setup - t_weather,
        t_solve - t_setup,
        t_end - t_solve,
        t_end - t0,
    )

    return "done", results


def _worker_init(
    interest_lookup: Dict,
    land_lookup: Dict,
    base_network: Any,
    tech_inputs: Dict[str, dict],
    tech_meta: Dict[str, Any],
    aggregation_count: int,
    time_step: float,
    max_snapshots: int | None,
    quiet: bool,
    land_constraint: str,
    capacity_rule: str,
    land_allocation: land_capacity.LandAllocation,
    allow_conservative_union_fallback: bool,
    temporal_accounting_mode: str,
    ramp_limit_basis: str,
    include_site_costs: bool,
    fail_fast: bool = False,
) -> None:
    """Initialise per-worker state.  Called once when each pool worker starts.

    The dataset is inherited from the parent process via fork (copy-on-write),
    so we do not re-open it here.  This keeps memory usage at a constant ~15 GB
    regardless of the number of workers.
    """
    _WORKER_STATE["dataset"] = _SHARED_DATASET  # CoW-inherited from parent
    _WORKER_STATE["interest_lookup"] = interest_lookup
    _WORKER_STATE["land_lookup"] = land_lookup
    _WORKER_STATE["base_network"] = base_network
    _WORKER_STATE["tech_inputs"] = tech_inputs
    _WORKER_STATE["tech_meta"] = tech_meta
    _WORKER_STATE["aggregation_count"] = aggregation_count
    _WORKER_STATE["time_step"] = time_step
    _WORKER_STATE["max_snapshots"] = max_snapshots
    _WORKER_STATE["quiet"] = quiet
    _WORKER_STATE["land_constraint"] = land_constraint
    _WORKER_STATE["capacity_rule"] = capacity_rule
    _WORKER_STATE["land_allocation"] = land_allocation
    _WORKER_STATE["allow_conservative_union_fallback"] = allow_conservative_union_fallback
    _WORKER_STATE["temporal_accounting_mode"] = temporal_accounting_mode
    _WORKER_STATE["ramp_limit_basis"] = ramp_limit_basis
    _WORKER_STATE["include_site_costs"] = include_site_costs
    _WORKER_STATE["fail_fast"] = fail_fast


def _worker_task(lat_lon: Tuple[float, float]) -> Tuple[float, float, str, Dict[str, Any] | None]:
    """Entry point for pool workers.  Returns (lat, lon, status, results).

    With fail_fast=True (set via _worker_init / _WORKER_STATE) any exception
    propagates out of the worker.  The pool pickles it and re-raises it in the
    parent at the imap_unordered iteration point — giving an immediate, full
    traceback rather than a silent skip.
    """
    lat, lon = lat_lon
    ws = _WORKER_STATE
    fail_fast = ws.get("fail_fast", False)
    if fail_fast:
        # Let any exception propagate — pool will re-raise it in the parent.
        status, results = _run_single_location_with_numerical_retry(
            lat=lat,
            lon=lon,
            dataset=ws["dataset"],
            interest_lookup=ws["interest_lookup"],
            tech_inputs=ws["tech_inputs"],
            tech_meta=ws["tech_meta"],
            land_lookup=ws["land_lookup"],
            base_network=ws["base_network"],
            aggregation_count=ws["aggregation_count"],
            time_step=ws["time_step"],
            max_snapshots=ws["max_snapshots"],
            quiet=ws["quiet"],
            land_constraint=ws["land_constraint"],
            capacity_rule=ws["capacity_rule"],
            land_allocation=ws["land_allocation"],
            allow_conservative_union_fallback=ws["allow_conservative_union_fallback"],
            temporal_accounting_mode=ws["temporal_accounting_mode"],
            ramp_limit_basis=ws["ramp_limit_basis"],
            include_site_costs=ws["include_site_costs"],
            fail_fast=True,
        )
    else:
        try:
            status, results = _run_single_location_with_numerical_retry(
                lat=lat,
                lon=lon,
                dataset=ws["dataset"],
                interest_lookup=ws["interest_lookup"],
                tech_inputs=ws["tech_inputs"],
                tech_meta=ws["tech_meta"],
                land_lookup=ws["land_lookup"],
                base_network=ws["base_network"],
                aggregation_count=ws["aggregation_count"],
                time_step=ws["time_step"],
                max_snapshots=ws["max_snapshots"],
                quiet=ws["quiet"],
                land_constraint=ws["land_constraint"],
                capacity_rule=ws["capacity_rule"],
                land_allocation=ws["land_allocation"],
                allow_conservative_union_fallback=ws["allow_conservative_union_fallback"],
                temporal_accounting_mode=ws["temporal_accounting_mode"],
                ramp_limit_basis=ws["ramp_limit_basis"],
                include_site_costs=ws["include_site_costs"],
                fail_fast=False,
            )
        except Exception as exc:  # noqa: BLE001
            LOGGER.error("Unhandled error in worker at (%s, %s): %s", lat, lon, exc, exc_info=True)
            status, results = "error", None
    return lat, lon, status, results


def run_global(
    locations: Sequence[Tuple[float, float]] | None = None,
    weather_dir: str | Path | None = None,
    land_csv: str | Path | None = None,
    override_csv: str | Path | None = None,
    tech_yaml: str | Path | None = None,
    plant_dir: str | Path | None = None,
    aggregation_count: int = 1,
    time_step: float = 1.0,
    max_snapshots: int | None = None,
    output_csv: str | Path | None = None,
    quiet: bool = False,
    threads_per_worker: int | None = None,
    num_workers: int = 1,
    lon_min: float | None = None,
    lon_max: float | None = None,
    fail_fast: bool = False,
    ensure_feasibility: bool = True,
    land_constraint: str = "after_solve",
    capacity_rule: str = "scaled_reference_design",
    land_allocation: land_capacity.LandAllocation = "colocated",
    allow_conservative_union_fallback: bool = True,
    temporal_accounting_mode: str = "snapshot_weighted",
    ramp_limit_basis: str = "per_hour",
    include_site_costs: bool = True,
) -> pd.DataFrame:
    """Run the ammonia plant optimisation for every requested location.

    Args:
        locations: Iterable of (lat, lon) pairs. If omitted, the function iterates over every
            coordinate contained in the max-capacities CSV.
        weather_dir: Directory containing the NetCDF stacks consumed by `location_tools`.
        land_csv: CSV with at least `Latitude` and `Longitude` columns plus any available
            max-capacity metadata.
        override_csv: Optional per-location override CSV. Supported columns include
            `interest_rate`, `build_cost_multiplier`, `land_cost_usd_per_km2_year`,
            `water_cost_usd_per_m3`, depth/remoteness multipliers, and related spatial
            cost inputs when present.
        tech_yaml: Optional tech-config YAML used to source financing defaults and currency metadata.
        aggregation_count: Snapshot aggregation factor forwarded to the weather loader.
        time_step: Duration of each snapshot in hours.
        max_snapshots: Optional hard cap on the number of weather snapshots to simulate per
            location (useful for quick smoke tests and notebooks).
        output_csv: Optional destination for the aggregated results table.
        threads_per_worker: Solver thread count (forwarded to GREEN_LORY_SOLVER_THREADS).
        lon_min: Optional minimum longitude bound (inclusive) for location filtering.
        lon_max: Optional maximum longitude bound (exclusive) for location filtering.
        fail_fast: When True any per-location failure (weather, setup, or solver) raises
            immediately with a full traceback instead of being logged and skipped.  Use
            during development/debugging; leave False for production batch runs.
        ensure_feasibility: When True the grid generator is made extendable at high-but-
            finite cost (same as the ramp-penalty link: ~110 MEUR/MW capital +
            1 MEUR/MWh marginal), guaranteeing every cell has a feasible LP.  Cells that
            rely on the backstop will have grid_energy_mwh > 0 in results and very high
            LCOA.  Set to False to reproduce the original hard-infeasibility behaviour.
    """
    requested_threads = 1 if threads_per_worker is None else int(threads_per_worker)
    if aggregation_count != 1:
        raise ValueError(
            "run_global requires aggregation_count=1; use time_step for explicit resampling"
        )
    if not float(time_step).is_integer() or float(time_step) < 1:
        raise ValueError("time_step must be a positive whole number of hours")
    timestep_hours = int(time_step)
    if 8760 % timestep_hours != 0:
        raise ValueError("time_step must divide the 8760-hour model year exactly")
    if temporal_accounting_mode not in {"snapshot_weighted", "legacy_scaled"}:
        raise ValueError(
            "temporal_accounting_mode must be 'snapshot_weighted' or 'legacy_scaled'"
        )
    if ramp_limit_basis not in {"per_hour", "legacy_per_snapshot"}:
        raise ValueError("ramp_limit_basis must be 'per_hour' or 'legacy_per_snapshot'")
    if land_constraint not in {"after_solve", "in_solve"}:
        raise ValueError("land_constraint must be 'after_solve' or 'in_solve'")
    if capacity_rule not in {"scaled_reference_design", "solved_quantity"}:
        raise ValueError("capacity_rule must be 'scaled_reference_design' or 'solved_quantity'")
    if capacity_rule == "solved_quantity" and land_constraint != "in_solve":
        raise ValueError("capacity_rule='solved_quantity' requires land_constraint='in_solve'")
    if capacity_rule == "scaled_reference_design" and land_constraint != "after_solve":
        raise ValueError(
            "capacity_rule='scaled_reference_design' requires land_constraint='after_solve'; "
            "scaling a land-constrained reference design would reintroduce circular "
            "1 Mt admission and double-apply the land budget"
        )
    land_allocation = land_capacity.normalise_land_allocation(land_allocation)

    with _quiet_logging(quiet), _override_env("GREEN_LORY_SOLVER_LOG", "0", quiet):
        weather_dir_path = _resolve_path(weather_dir) or DEFAULT_WEATHER_DIR
        land_csv_path = _resolve_path(land_csv or DEFAULT_LAND_CSV)
        override_csv_path = _resolve_path(override_csv) if override_csv is not None else None
        tech_yaml_path = _resolve_path(tech_yaml or DEFAULT_TECH_YAML)
        plant_dir_path = _resolve_path(
            plant_dir or os.environ.get("ARC_PLANT_DIR", "basic_ammonia_plant")
        )
        if plant_dir_path is None or not plant_dir_path.is_dir():
            raise FileNotFoundError(f"Plant directory not found: {plant_dir_path}")
        required_plant_files = {
            "buses.csv",
            "generators.csv",
            "links.csv",
            "loads.csv",
            "stores.csv",
        }
        missing_plant_files = sorted(
            name for name in required_plant_files if not (plant_dir_path / name).is_file()
        )
        if missing_plant_files:
            raise FileNotFoundError(
                f"Plant directory {plant_dir_path} is missing: {', '.join(missing_plant_files)}"
            )

        tech_inputs = _load_tech_inputs(tech_yaml_path)
        tech_meta = _load_tech_meta(tech_yaml_path)
        land_df = _load_land_table(land_csv_path)
        if not allow_conservative_union_fallback:
            _validate_versioned_union_land_table(land_df, land_csv_path)
        land_df = _apply_spatial_solar_density_from_tech_config(land_df, tech_inputs)
        world = _load_country_shapes()
        store = results_store.Data_store()

        if locations is None:
            location_list = _default_locations(land_df)
        else:
            location_list = [(float(lat), float(lon)) for lat, lon in locations]

        # Apply longitude segmentation if bounds are specified.
        if lon_min is not None or lon_max is not None:
            lo = float(lon_min) if lon_min is not None else -180.0
            hi = float(lon_max) if lon_max is not None else 180.0
            location_list = [(la, lo_) for la, lo_ in location_list if lo <= lo_ < hi]
            LOGGER.info("Longitude filter [%.1f, %.1f): %d locations", lo, hi, len(location_list))

        total_locations = len(location_list)
        progress = _ProgressTracker(total_locations)

        with _override_env(
            "GREEN_LORY_SOLVER_THREADS",
            str(requested_threads),
            threads_per_worker is not None,
        ):
            interest_df = _load_interest_table(override_csv_path)

            # ── Pre-computation phase ──────────────────────────────────────
            t_pre = time.perf_counter()

            # Pre-index lookup tables for O(1) per-cell access.
            interest_lookup = _build_interest_lookup(interest_df)
            land_lookup = _build_land_lookup(land_df)
            country_lookup = _build_country_lookup(world, location_list)

            # Build the base PyPSA network once; deep-copy per cell.
            # Load all weather data into RAM here in the main process.  For the
            # parallel path the forked workers inherit the loaded numpy arrays
            # via copy-on-write at zero extra RAM cost.  For the serial path the
            # same object is used directly.
            global _SHARED_DATASET
            _SHARED_DATASET = lt.all_locations(str(weather_dir_path), cache_resources=True)
            probe_lat, probe_lon = location_list[0] if location_list else (0.0, 0.0)
            probe_weather = _build_weather_frame(_SHARED_DATASET, probe_lat, probe_lon, aggregation_count)
            # Apply the same resampling that per-cell code will use.
            if time_step > 1:
                probe_weather = _resample_weather_frame(probe_weather, int(time_step))
            n_snapshots = max_snapshots if max_snapshots is not None else len(probe_weather)
            base_network = plant_main.generate_network(
                n_snapshots,
                str(plant_dir_path),
                aggregation_count=aggregation_count,
                time_step=time_step,
                temporal_accounting_mode=temporal_accounting_mode,
            )
            base_network._currency = str(tech_meta.get("currency", "USD")).strip().upper()
            base_network._ramp_limit_basis = ramp_limit_basis

            # ── Feasibility backstop ───────────────────────────────────────
            # When ensure_feasibility=True AND the tech YAML contains a
            # grid_backstop section, the grid generator acts as a relief valve:
            # the LP is always feasible, and grid_energy_mwh > 0 in results
            # flags locations that genuinely cannot meet demand from local
            # renewables alone.  Parameters are defined in the tech YAML —
            # comment out the grid_backstop section there to disable.
            _grid_backstop_cfg = tech_inputs.get("grid_backstop") if ensure_feasibility else None
            if _grid_backstop_cfg and "grid" in base_network.generators.index:
                _GRID_P_NOM = float(_grid_backstop_cfg.get("p_nom_mw", 20_000.0))
                _GRID_MARGINAL = float(_grid_backstop_cfg.get("marginal_cost_eur_per_mwh", 100_000.0))
                base_network.generators.at["grid", "p_nom_extendable"] = False
                base_network.generators.at["grid", "p_nom"] = _GRID_P_NOM
                base_network.generators.at["grid", "capital_cost"] = 0.0
                base_network.generators.at["grid", "marginal_cost"] = _GRID_MARGINAL
                LOGGER.info(
                    "ensure_feasibility=True: grid backstop active "
                    "(p_nom=%.0f MW, marginal=%.0f EUR/MWh, capital=0)",
                    _GRID_P_NOM, _GRID_MARGINAL,
                )
            elif ensure_feasibility:
                LOGGER.warning(
                    "ensure_feasibility=True but no grid_backstop section found in "
                    "tech YAML; backstop disabled. Comment back in the grid_backstop "
                    "block or remove --no-ensure-feasibility to re-enable."
                )

            LOGGER.info(
                "Pre-computation complete in %.1fs: %d interest entries, %d land entries, %d country entries, base network built (%d snapshots)",
                time.perf_counter() - t_pre,
                len(interest_lookup),
                len(land_lookup),
                len(country_lookup),
                n_snapshots,
            )

            # Open output file for incremental writing (provides partial-results
            # recovery on crash and progress monitoring via file size).
            out_fh: Any = None
            header_written = False
            if output_csv:
                output_path = _resolve_path(output_csv) or Path(output_csv)
                output_path.parent.mkdir(parents=True, exist_ok=True)
                out_fh = open(output_path, "w", newline="", encoding="utf-8")

            def _write_csv_row(lat: float, lon: float, country: str, res: Dict[str, Any]) -> None:
                nonlocal header_written
                if out_fh is None:
                    return
                row = {"latitude": lat, "longitude": lon, "country": country, **res}
                row_df = pd.DataFrame([row])
                if not header_written:
                    row_df.to_csv(out_fh, index=False)
                    header_written = True
                else:
                    row_df.to_csv(out_fh, index=False, header=False)
                out_fh.flush()

            try:
                if num_workers > 1:
                    # ── Parallel path: dataset loaded once, shared via fork CoW ──
                    LOGGER.info(
                        "Starting parallel run: %d workers x %d threads = %d CPUs",
                        num_workers, requested_threads, num_workers * requested_threads,
                    )
                    _mp_ctx = multiprocessing.get_context("fork" if sys.platform != "win32" else "spawn")
                    # Track submitted vs received so any worker-process deaths
                    # (SIGKILL / OOM) produce a retryable _failed_<ts>.csv rather
                    # than silently dropping locations.  chunksize=1 ensures at most
                    # one location is lost per worker death (vs two with chunksize=2).
                    _received: set[tuple[float, float]] = set()
                    with _mp_ctx.Pool(
                        processes=num_workers,
                        initializer=_worker_init,
                        initargs=(
                            interest_lookup, land_lookup, base_network,
                            tech_inputs, tech_meta, aggregation_count, time_step, max_snapshots, quiet,
                            land_constraint, capacity_rule, land_allocation,
                            allow_conservative_union_fallback, temporal_accounting_mode,
                            ramp_limit_basis, include_site_costs,
                            fail_fast,
                        ),
                    ) as pool:
                        try:
                            for lat, lon, status, results in pool.imap_unordered(
                                _worker_task, location_list, chunksize=1,
                            ):
                                _received.add((lat, lon))
                                if results is not None:
                                    country = country_lookup.get((lat, lon), _country_for(world, lat, lon))
                                    store.add_location(lat, lon, country, results)
                                    _write_csv_row(lat, lon, country, results)
                                progress.update(lat, lon, status)
                        except Exception as _pool_exc:  # noqa: BLE001 – infrastructure crash (OOM/SIGKILL/broken pipe)
                            # Per-cell solver errors are already caught inside _worker_task
                            # and returned as (lat, lon, "error", None), so any exception
                            # that reaches here is an infrastructure-level crash (worker
                            # killed by OOM, SLURM, Gurobi licence limit, etc.).
                            # Always re-raise after writing the dropped-locations CSV so
                            # the SLURM job exits non-zero and sends the failure email.
                            _dropped = [loc for loc in location_list if loc not in _received]
                            LOGGER.error(
                                "Pool iteration aborted after %d/%d locations: %s — "
                                "%d location(s) were never processed.",
                                len(_received), len(location_list), _pool_exc, len(_dropped),
                            )
                            if _dropped and output_csv:
                                from datetime import datetime as _dt  # noqa: PLC0415
                                _ts = _dt.now().strftime("%Y%m%d-%H%M%S")
                                _failed_path = Path(output_csv).with_stem(
                                    Path(output_csv).stem + f"_failed_{_ts}"
                                )
                                pd.DataFrame(_dropped, columns=["lat", "lon"]).to_csv(
                                    _failed_path, index=False
                                )
                                LOGGER.error(
                                    "Dropped locations written to %s — resubmit with "
                                    "--locations-csv '%s'",
                                    _failed_path, _failed_path,
                                )
                            raise  # always propagate — pool is dead
                else:
                    # ── Serial path: use _SHARED_DATASET directly ──
                    for lat, lon in location_list:
                        try:
                            status, results = _run_single_location_with_numerical_retry(
                                lat=lat,
                                lon=lon,
                                dataset=_SHARED_DATASET,
                                interest_lookup=interest_lookup,
                                tech_inputs=tech_inputs,
                                tech_meta=tech_meta,
                                land_lookup=land_lookup,
                                base_network=base_network,
                                aggregation_count=aggregation_count,
                                time_step=time_step,
                                max_snapshots=max_snapshots,
                                quiet=quiet,
                                land_constraint=land_constraint,
                                capacity_rule=capacity_rule,
                                land_allocation=land_allocation,
                                allow_conservative_union_fallback=allow_conservative_union_fallback,
                                temporal_accounting_mode=temporal_accounting_mode,
                                ramp_limit_basis=ramp_limit_basis,
                                include_site_costs=include_site_costs,
                                fail_fast=fail_fast,
                            )
                        except Exception as exc:  # noqa: BLE001
                            if fail_fast:
                                raise
                            LOGGER.error(
                                "Unhandled error at (%s, %s) — skipping: %s",
                                lat, lon, exc, exc_info=True,
                            )
                            status, results = "error", None
                        if results is not None:
                            country = country_lookup.get((lat, lon), _country_for(world, lat, lon))
                            store.add_location(lat, lon, country, results)
                            _write_csv_row(lat, lon, country, results)
                        progress.update(lat, lon, status)
            finally:
                if _SHARED_DATASET is not None:
                    _SHARED_DATASET.close()
                    _SHARED_DATASET = None
                if out_fh is not None:
                    out_fh.close()

        df = pd.DataFrame.from_dict(store.collated_results, orient='index')
        df = _order_results_columns(df)
        if output_csv:
            # Rewrite ordered CSV (the incremental file may have unordered columns).
            output_path = _resolve_path(output_csv) or Path(output_csv)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(output_path, index=False)
        return df


def load_locations_csv(path: str | Path) -> List[Tuple[float, float]]:
    """Read a CSV with ``lat`` and ``lon`` columns and return a list of (lat, lon) tuples."""
    df = pd.read_csv(path)
    for column in ("lat", "lon"):
        if column not in df.columns:
            raise ValueError(f"Locations CSV {path} must include '{column}' column")
    return list(zip(df["lat"], df["lon"]))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the green ammonia model over many locations.")
    parser.add_argument("--locations-csv", type=str, help="CSV with columns lat,lon to restrict the sweep.")
    parser.add_argument(
        "--tech-yaml",
        type=str,
        default=None,
        help="Path to the tech-config YAML (overrides the built-in default).",
    )
    parser.add_argument(
        "--plant-dir",
        type=str,
        default=None,
        help="Plant CSV bundle controlling topology, efficiencies, loads, and ramp limits.",
    )
    parser.add_argument(
        "--override-csv",
        type=str,
        default=None,
        help=(
            "Spatial override CSV (lat,lon,tech plus optional interest/build/water/land columns); may include "
            "build_cost_multiplier, land_cost_usd_per_km2_year, and water_cost_usd_per_m3. "
            "Generate with notebook 02_spatial_cost_inputs."
        ),
    )
    parser.add_argument("--land-csv", type=str, default=None, help="Optional override for the max-capacities CSV.")
    parser.add_argument("--output-csv", type=str, default="results/run_global.csv", help="Where to write aggregated results.")
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress solver and PyPSA logging when running large batches.",
    )
    parser.add_argument(
        "--max-snapshots",
        type=int,
        default=None,
        help="Limit the number of weather snapshots per location (useful for notebooks/smoke tests).",
    )
    parser.add_argument(
        "--threads-per-worker",
        type=int,
        default=None,
        help="Solver thread count (forwarded to GREEN_LORY_SOLVER_THREADS).",
    )
    parser.add_argument("--limit", type=int, default=None, help="Optional limit on the number of locations to process.")
    parser.add_argument("--lon-min", type=float, default=None, help="Minimum longitude (inclusive) for location filtering.")
    parser.add_argument("--lon-max", type=float, default=None, help="Maximum longitude (exclusive) for location filtering.")
    parser.add_argument(
        "--time-step",
        type=float,
        default=1.0,
        help="Timestep resolution in hours (default: 1.0). E.g. 3.0 for 3-hourly snapshots.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=1,
        help="Number of parallel worker processes (default: 1 = serial). Set >1 to enable multiprocessing.",
    )
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        default=False,
        help="Raise immediately on any per-location failure instead of logging and skipping.",
    )
    parser.add_argument(
        "--no-ensure-feasibility",
        dest="ensure_feasibility",
        action="store_false",
        default=True,
        help="Disable the grid backstop and allow hard LP infeasibility (reproduces original behaviour).",
    )
    parser.add_argument(
        "--land-constraint",
        choices=("after_solve", "in_solve"),
        default="after_solve",
        help=("after_solve: optimise the 1 Mt/yr reference design without land limits and apply the land "
             "budget afterwards as a capacity multiplier; in_solve: bound wind and PV inside the optimisation."),
    )
    parser.add_argument(
        "--capacity-rule",
        choices=("scaled_reference_design", "solved_quantity"),
        default="scaled_reference_design",
        help=("scaled_reference_design: scale the unconstrained reference design until wind, PV or their shared "
             "footprint fills the land (Salmon/Verschuur rule); solved_quantity: report the production solved "
             "under in_solve land limits (a feasible quantity, not a maximum)."),
    )
    parser.add_argument(
        "--land-allocation",
        choices=land_capacity.LAND_ALLOCATIONS,
        default="colocated",
        help=("Renewable land allocation rule: colocated (base; only the exclusive fraction of the wind "
              "footprint competes with PV for the shared budget) or exclusive (September 2026 rule)."),
    )
    parser.add_argument(
        "--no-conservative-union-fallback",
        dest="allow_conservative_union_fallback",
        action="store_false",
        default=True,
        help=(
            "Require the versioned classwise-nested v1 renewable-union lower-bound "
            "columns and provenance in the land table."
        ),
    )
    parser.add_argument(
        "--temporal-accounting-mode",
        choices=("snapshot_weighted", "legacy_scaled"),
        default="snapshot_weighted",
    )
    parser.add_argument(
        "--ramp-limit-basis",
        choices=("per_hour", "legacy_per_snapshot"),
        default="per_hour",
    )
    parser.add_argument(
        "--exclude-site-costs",
        dest="include_site_costs",
        action="store_false",
        default=True,
        help="Keep water and land out of the headline for strict plant-gate replication.",
    )
    args = parser.parse_args()

    if args.locations_csv:
        requested_locations = load_locations_csv(args.locations_csv)
        if args.limit is not None:
            requested_locations = requested_locations[: args.limit]
    else:
        requested_locations = None

    results_df = run_global(
        locations=requested_locations,
        tech_yaml=args.tech_yaml,
        plant_dir=args.plant_dir,
        override_csv=args.override_csv,
        land_csv=args.land_csv,
        time_step=args.time_step,
        max_snapshots=args.max_snapshots,
        output_csv=args.output_csv,
        quiet=args.quiet,
        threads_per_worker=args.threads_per_worker,
        num_workers=args.num_workers,
        lon_min=args.lon_min,
        lon_max=args.lon_max,
        fail_fast=args.fail_fast,
        ensure_feasibility=args.ensure_feasibility,
        land_constraint=args.land_constraint,
        capacity_rule=args.capacity_rule,
        land_allocation=args.land_allocation,
        allow_conservative_union_fallback=args.allow_conservative_union_fallback,
        temporal_accounting_mode=args.temporal_accounting_mode,
        ramp_limit_basis=args.ramp_limit_basis,
        include_site_costs=args.include_site_costs,
    )
    print(f"Processed {len(results_df)} locations. Results saved to {args.output_csv}.")
