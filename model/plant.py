"""One ammonia plant at one site.

The plant is a PyPSA network read from a plant bundle, a folder of CSV files under
model/plants/ (buses, generators, links, loads, stores). Wind and PV feed a power bus; an
electrolyser, a hydrogen compressor and store, a battery and a Haber-Bosch loop (ammonia
synthesis with its air separation unit) turn the power into a constant ammonia output.

    network = build_network(8760, plant_dir, time_step=1.0, temporal_accounting_mode="snapshot_weighted")
    results = solve(network, weather_frame, time_step=1.0)

build_network() reads the bundle once; callers copy the network for each cell, set its costs
(technology.py) and pass one cell's weather to solve(). solve() adds the plant's own
constraints (ammonia ramp limits, the coupling of the battery charger and discharger and, when
land is enforced in the solve, the land budget), solves with Gurobi through linopy and returns
the plant results: cost, production and every capacity.

Time accounting. ``snapshot_weighted`` represents each snapshot's duration with PyPSA snapshot
weights, so stores are in MWh whatever the time step. ``legacy_scaled`` reproduces the
published 4-hour method, which scaled store costs and reported store energies by the time step
instead; it is kept only for the reconciliation runs.
"""
from __future__ import annotations

import copy
import json
import logging
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pypsa

from .config import AMMONIA_HHV_MWH_PER_T, HYDROGEN_HHV_MWH_PER_T, repo_path

LOGGER = logging.getLogger(__name__)

TEMPORAL_ACCOUNTING_MODES = ("snapshot_weighted", "legacy_scaled")
RAMP_LIMIT_BASES = ("per_hour", "legacy_per_snapshot")
RENEWABLES = ("wind", "solar", "solar_tracking")

# Component names of older bundles and YAMLs -> the names used in the plant bundles
COMPONENT_NAMES = {
    "Power": "power",
    "Ammonia": "ammonia",
    "PowerStorage": "power_storage",
    "Hydrogen": "hydrogen",
    "HydrogenStorage": "hydrogen_storage",
    "RampPenalty": "ramp_penalty",
    "RampPenaltyDest": "ramp_penalty_dest",
    "Wind": "wind",
    "OnshoreWind": "wind",
    "Solar": "solar",
    "SolarTracking": "solar_tracking",
    "Grid": "grid",
    "RampDummy": "ramp_dummy",
    "Electrolysis": "electrolysis",
    "HydrogenCompression": "hydrogen_compression",
    "HydrogenFromStorage": "hydrogen_from_storage",
    "HB": "ammonia_synthesis",
    "BatteryInterfaceIn": "battery_pcs_charge",
    "BatteryInterfaceOut": "battery_pcs_discharge",
    "HydrogenFuelCell": "hydrogen_fuel_cell",
    "PenaltyLink": "penalty_link",
    "CompressedH2Store": "compressed_hydrogen_store",
    "Battery": "battery_storage",
    "BatteryStorage": "battery_storage",
    "AccumulatedPenalty": "accumulated_penalty",
}

PLANT_BUNDLE_FILES = ("buses.csv", "generators.csv", "links.csv", "loads.csv", "stores.csv")


def normalize_component_name(name: str | None) -> str | None:
    """The bundle name of a component (older names are mapped, others pass through)."""
    if name is None:
        return None
    return COMPONENT_NAMES.get(name, name)


# ---------------------------------------------------------------------------------------------
# Building the network
# ---------------------------------------------------------------------------------------------

def check_plant_bundle(plant_dir: str | Path) -> Path:
    """The bundle folder, after checking that it holds every component file."""
    path = repo_path(plant_dir)
    if not path.is_dir():
        raise FileNotFoundError(f"Plant directory not found: {path}")
    missing = [name for name in PLANT_BUNDLE_FILES if not (path / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Plant directory {path} is missing: {', '.join(missing)}")
    return path


def build_network(
    n_snapshots: int,
    plant_dir: str | Path,
    *,
    time_step: float,
    temporal_accounting_mode: str,
) -> pypsa.Network:
    """Read a plant bundle into a network of ``n_snapshots`` snapshots of ``time_step`` hours."""
    if temporal_accounting_mode not in TEMPORAL_ACCOUNTING_MODES:
        raise ValueError(f"temporal_accounting_mode must be one of {TEMPORAL_ACCOUNTING_MODES}")
    network = pypsa.Network()
    network.import_from_csv_folder(repo_path(plant_dir))
    network.set_snapshots(range(int(n_snapshots)))
    network._temporal_accounting_mode = temporal_accounting_mode
    network._snapshot_hours = float(time_step)
    if temporal_accounting_mode == "snapshot_weighted":
        network.snapshot_weightings.loc[:, :] = float(time_step)

    # Bundles may omit carriers.csv; add every carrier the components refer to
    existing = set(network.carriers.index)
    referenced = set()
    for component in (network.buses, network.generators, network.links, network.stores, network.loads):
        if not component.empty and "carrier" in component.columns:
            referenced.update(component["carrier"].dropna().unique())
    for carrier in sorted(filter(None, referenced)):
        if carrier not in existing:
            network.add("Carrier", carrier)

    if temporal_accounting_mode == "legacy_scaled":
        # The published convention, kept byte for byte for the reconciliation runs
        network.stores.capital_cost *= time_step
        network.links.marginal_cost *= 24 * 366 / n_snapshots
    return network


def set_ammonia_output(network: pypsa.Network, mt_per_year: float) -> pypsa.Network:
    """A copy of the network whose constant ammonia load is ``mt_per_year`` Mt/yr."""
    if not np.isfinite(mt_per_year) or mt_per_year <= 0:
        raise ValueError("Quantity must be positive and finite")
    if list(network.loads.index) != ["ammonia"] or not network.loads_t.p_set.empty:
        raise ValueError("The plant must have one constant ammonia load")
    network = copy.deepcopy(network)
    network.loads.at["ammonia", "p_set"] = mt_per_year * 1e6 * AMMONIA_HHV_MWH_PER_T / 8760
    return network


# Which PV the plant may build: both kinds, fixed tilt only, or single-axis tracking only
PV_POLICIES = ("both", "fixed", "tracking")


def restrict_pv(network: pypsa.Network, policy: str) -> pypsa.Network:
    """Switch off the PV technology a policy excludes (``fixed`` drops tracking PV)."""
    if policy not in PV_POLICIES:
        raise ValueError(f"Unknown PV policy {policy!r}; choose from {PV_POLICIES}")
    excluded = {"both": None, "fixed": "solar_tracking", "tracking": "solar"}[policy]
    if excluded is not None:
        if excluded not in network.generators.index:
            raise ValueError(f"Missing {excluded} generator")
        network.generators.loc[excluded, ["p_nom", "p_nom_max"]] = 0.0
        network.generators.loc[excluded, "p_nom_extendable"] = False
    return network


# Generator name -> weather column used when the generator has no column of its own
WEATHER_PROFILE_ALIASES = {"solar_tracking": ("solar",)}


def apply_weather(network: pypsa.Network, weather: pd.DataFrame) -> pypsa.Network:
    """Set each generator's hourly availability from the weather frame's column of that name."""
    frame = pd.DataFrame(weather).reset_index(drop=True)
    if len(frame) != len(network.snapshots):
        raise ValueError(
            f"Weather/profile length mismatch: got {len(frame)} rows for {len(network.snapshots)} snapshots."
        )
    missing = []
    for generator in network.generators.index:
        # The grid generator is a backstop (switched off in every bundle): always available,
        # whatever the weather frame's zero-filled grid column says
        if generator == "grid":
            network.generators_t.p_max_pu[generator] = 1.0
            continue
        column = generator if generator in frame.columns else next(
            (alias for alias in WEATHER_PROFILE_ALIASES.get(generator, ()) if alias in frame.columns), None
        )
        if column is None and generator == "ramp_dummy":
            network.generators_t.p_max_pu[generator] = 1.0
            continue
        if column is None:
            missing.append(generator)
            continue
        network.generators_t.p_max_pu[generator] = frame[column].to_numpy()
    if missing:
        raise ValueError(f"Missing weather columns for generators: {', '.join(sorted(missing))}")
    return network


# ---------------------------------------------------------------------------------------------
# Solving
# ---------------------------------------------------------------------------------------------

class OptimizationFailure(RuntimeError):
    """A solve that did not end optimal, with the solver's status and termination condition."""

    def __init__(self, status, condition):
        self.status = str(status)
        self.condition = str(condition)
        super().__init__(f"Optimisation failed with status {status} ({condition}).")

    def __reduce__(self):
        return type(self), (self.status, self.condition)


def require_optimal_termination(status, condition) -> None:
    # Linopy can label a Gurobi SUBOPTIMAL end as status 'ok'; only 'optimal' is accepted.
    if str(status) != "ok" or str(condition) != "optimal":
        raise OptimizationFailure(status, condition)


def solver_options(solver: str, solver_threads: int | None = None, override: dict | None = None) -> dict | None:
    """Solver parameters for one solve.

    Gurobi: barrier (Method 2) without crossover through the direct Python API, about 24 times
    faster than the LP-file path for this problem, with BarConvTol 1e-4 (about 19 % faster,
    under 0.1 EUR/t on LCOA). A stalled barrier without crossover ends SUBOPTIMAL on degenerate
    cells; the surface runner then retries with other settings (surface.NUMERICAL_RETRY_OPTIONS).
    GREEN_LORY_SOLVER_OPTIONS_JSON (for example {"Crossover": 1}) changes the first attempt
    campaign-wide; ``override`` is a retry's own setting.
    """
    options = None
    if solver == "gurobi":
        quiet = os.environ.get("GREEN_LORY_SOLVER_LOG", "0").lower() in {"0", "false", "no", "off"}
        options = {
            "OutputFlag": 0 if quiet else 1,
            "LogToConsole": 0 if quiet else 1,
            "Method": 2,
            "Crossover": 0,
            "BarConvTol": 1e-4,
        }
        if solver_threads is not None:
            options["Threads"] = solver_threads
        raw = os.environ.get("GREEN_LORY_SOLVER_OPTIONS_JSON", "").strip()
        if raw:
            environment_options = json.loads(raw)
            if not isinstance(environment_options, dict):
                raise ValueError("GREEN_LORY_SOLVER_OPTIONS_JSON must be a JSON object")
            options.update(environment_options)
    elif solver == "highs":
        options = {"solver": "ipm"}
        if solver_threads is not None:
            options["threads"] = solver_threads
    if override is not None:
        options = {**(options or {}), **override}
    return options


def solve(
    network: pypsa.Network,
    weather: pd.DataFrame,
    *,
    time_step: float,
    interest_rates: dict[str, float] | None = None,
    water_cost_usd_per_m3: float | None = None,
    water_usage_m3_per_t: float | None = None,
    land_cost_usd_per_km2_year: float | None = None,
    land_used_km2: float | None = None,
    solver_options_override: dict | None = None,
) -> dict:
    """Solve the plant for one cell's weather and return its results (see plant_results).

    The solver comes from GREEN_LORY_SOLVER (default gurobi) and its thread count from
    GREEN_LORY_SOLVER_THREADS.
    """
    apply_weather(network, weather)
    solver = os.environ.get("GREEN_LORY_SOLVER", "gurobi").lower()
    threads_raw = os.environ.get("GREEN_LORY_SOLVER_THREADS", "").strip()
    threads = int(threads_raw) if threads_raw else None
    if threads is not None and threads < 1:
        raise ValueError("GREEN_LORY_SOLVER_THREADS must be >= 1")

    optimize_kwargs = {
        "solver_name": solver,
        "extra_functionality": extra_constraints,
        "io_api": "direct",  # Gurobi's Python API, no LP file
    }
    options = solver_options(solver, threads, solver_options_override)
    if options is not None:
        optimize_kwargs["solver_options"] = options

    # PyPSA builds its own link ramp constraints before extra_functionality runs
    prepare_ammonia_ramp_constraints(network)
    status, condition = network.optimize(**optimize_kwargs)
    require_optimal_termination(status, condition)

    output = plant_results(
        network,
        time_step=time_step,
        interest_rates=interest_rates,
        water_cost_usd_per_m3=water_cost_usd_per_m3,
        water_usage_m3_per_t=water_usage_m3_per_t,
        land_cost_usd_per_km2_year=land_cost_usd_per_km2_year,
        land_used_km2=land_used_km2,
    )
    output.update(solver_name=solver, solver_status=str(status), solver_termination=str(condition))
    return output


# ---------------------------------------------------------------------------------------------
# Constraints beyond PyPSA's own
# ---------------------------------------------------------------------------------------------

_AMMONIA_RAMP_RATE_ATTRS = {
    "up": "_ammonia_synthesis_ramp_limit_up_per_hour",
    "down": "_ammonia_synthesis_ramp_limit_down_per_hour",
}


def prepare_ammonia_ramp_constraints(network: pypsa.Network) -> None:
    """Move the Haber-Bosch ramp limits out of PyPSA's hands before the model is built.

    PyPSA reads ``Link.ramp_limit_*`` as a fraction of capacity per snapshot. In the
    ``per_hour`` basis the bundle's values are hourly rates, which extra_constraints() scales by
    the snapshot duration; left in place, PyPSA's native per-snapshot limit would add a second,
    tighter constraint at multi-hour steps. The ``legacy_per_snapshot`` basis keeps the
    historical behaviour and is left untouched.
    """
    if getattr(network, "_ramp_limit_basis", "legacy_per_snapshot") != "per_hour":
        return
    if "ammonia_synthesis" not in network.links.index:
        return
    for direction, private_attr in _AMMONIA_RAMP_RATE_ATTRS.items():
        column = f"ramp_limit_{direction}"
        if column not in network.links.columns:
            continue
        native_rate = network.links.at["ammonia_synthesis", column]
        if pd.notna(native_rate):
            setattr(network, private_attr, float(native_rate))
        network.links.at["ammonia_synthesis", column] = np.nan


def _ammonia_ramp_rate(network: pypsa.Network, direction: str) -> float:
    preserved = getattr(network, _AMMONIA_RAMP_RATE_ATTRS[direction], np.nan)
    if pd.notna(preserved):
        return float(preserved)
    column = f"ramp_limit_{direction}"
    if "ammonia_synthesis" in network.links.index and column in network.links.columns:
        native = network.links.at["ammonia_synthesis", column]
        if pd.notna(native):
            return float(native)
    return np.nan


def _ammonia_ramp_limit_per_snapshot(network: pypsa.Network, direction: str) -> float:
    rate = _ammonia_ramp_rate(network, direction)
    if pd.isna(rate):
        return np.nan
    per_hour = getattr(network, "_ramp_limit_basis", "legacy_per_snapshot") == "per_hour"
    multiplier = float(getattr(network, "_snapshot_hours", 1.0)) if per_hour else 1.0
    return min(1.0, rate * multiplier)


def extra_constraints(network: pypsa.Network, snapshots) -> None:
    """linopy constraints added to PyPSA's model (passed as extra_functionality)."""
    if getattr(network, "model", None) is None:
        raise RuntimeError("Network model not initialised; use n.optimize(extra_functionality=...).")
    model = network.model
    sns = snapshots if snapshots is not None else network.snapshots

    land_budget = getattr(network, "_renewable_land_budget", None)
    land_allocation = getattr(network, "_land_allocation", None)
    if land_budget is not None and land_allocation is not None:
        from .land import add_land_constraints

        add_land_constraints(model, land_budget, land_allocation)

    # The battery charger carries the PCS capital cost and the discharger only a token cost, so
    # their ratings are locked together (otherwise discharge power would be free).
    try:
        charge = model["Link-p_nom"].sel(name="battery_pcs_charge")
        discharge = model["Link-p_nom"].sel(name="battery_pcs_discharge")
    except KeyError:
        charge = discharge = None
    if charge is not None and discharge is not None:
        efficiency = network.links.at["battery_pcs_discharge", "efficiency"]
        model.add_constraints(charge == discharge / efficiency, name="battery_pcs_balance")

    # Haber-Bosch ramp limits, between consecutive snapshots (cyclic)
    try:
        dispatch = model["Link-p"].sel(name="ammonia_synthesis", snapshot=sns)
    except KeyError:
        dispatch = None
    if dispatch is not None and "ammonia_synthesis" in network.links.index:
        previous = dispatch.roll(snapshot=1, roll_coords=False)
        try:
            capacity = model["Link-p_nom"].sel(name="ammonia_synthesis")
        except KeyError:
            capacity = network.links.loc["ammonia_synthesis", "p_nom"]
        ramp_down = _ammonia_ramp_limit_per_snapshot(network, "down")
        ramp_up = _ammonia_ramp_limit_per_snapshot(network, "up")
        if pd.notna(ramp_down):
            model.add_constraints(previous - dispatch <= ramp_down * capacity, name="ammonia_synthesis_ramp_down")
        if pd.notna(ramp_up):
            model.add_constraints(dispatch - previous <= ramp_up * capacity, name="ammonia_synthesis_ramp_up")


# ---------------------------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------------------------

# label -> (component type, component name) of the cost shares reported per technology
COST_COMPONENT_SPECS = {
    "wind": ("generator", "wind"),
    "solar": ("generator", "solar"),
    "solar_tracking": ("generator", "solar_tracking"),
    "electrolyser": ("link", "electrolysis"),
    "ammonia_synthesis": ("link", "ammonia_synthesis"),
    "hydrogen_compression": ("link", "hydrogen_compression"),
    "hydrogen_store": ("store", "compressed_hydrogen_store"),
    "ammonia_store": ("store", "ammonia"),
    "battery": ("store", "battery_storage"),
}

INTEREST_RATE_COLUMN_ALIASES = {
    "electrolysis": "electrolyser",
    "compressed_hydrogen_store": "hydrogen_store",
    "ammonia": "ammonia_store",
}


def _component_table(network, component_type):
    return {"generator": network.generators, "link": network.links, "store": network.stores}[component_type]


def _component_capacity(network, component_type, name) -> float:
    table = _component_table(network, component_type)
    if name not in table.index:
        return 0.0
    value = table.at[name, "p_nom_opt" if component_type in {"generator", "link"} else "e_nom_opt"]
    return 0.0 if pd.isna(value) else float(value)


def _component_capex(network, component_type, name) -> float:
    table = _component_table(network, component_type)
    if name not in table.index:
        return 0.0
    capital_cost = table.at[name, "capital_cost"]
    capacity = _component_capacity(network, component_type, name)
    if pd.isna(capital_cost) or capacity <= 0:
        return 0.0
    return float(capital_cost) * capacity


def _snapshot_weightings(network) -> pd.Series:
    weightings = getattr(network.snapshot_weightings, "generators", None)
    if weightings is None:
        return pd.Series(1.0, index=network.snapshots)
    weightings = pd.Series(weightings, index=network.snapshots)
    return weightings.reindex(network.snapshots).ffill().bfill().fillna(1.0)


def _component_dispatch_cost(network, component_type, name, weightings) -> float:
    if component_type == "generator":
        table, dispatch = network.generators, getattr(network.generators_t, "p", None)
    elif component_type == "link":
        table, dispatch = network.links, getattr(network.links_t, "p0", None)
    else:
        table, dispatch = network.stores, None
    if name not in table.index:
        return 0.0
    marginal_cost = table.at[name, "marginal_cost"] if "marginal_cost" in table.columns else 0.0
    if pd.isna(marginal_cost) or marginal_cost == 0 or dispatch is None or name not in dispatch:
        return 0.0
    flow = dispatch[name].reindex(weightings.index, fill_value=0.0)
    return float((flow * weightings).sum() * float(marginal_cost))


def _total_capex(network) -> float:
    total = 0.0
    for table, column in ((network.generators, "p_nom_opt"), (network.links, "p_nom_opt"), (network.stores, "e_nom_opt")):
        if not table.empty:
            total += float((table["capital_cost"].fillna(0.0) * table[column].fillna(0.0)).sum())
    return total


def _component_cost_breakdown(network):
    weightings = _snapshot_weightings(network)
    total_cost = float(network.objective)
    components = {}
    tracked_total = 0.0
    for label, (component_type, name) in COST_COMPONENT_SPECS.items():
        capex = _component_capex(network, component_type, name)
        opex = _component_dispatch_cost(network, component_type, name, weightings)
        components[label] = {"capex": capex, "opex": opex, "total": capex + opex}
        tracked_total += capex + opex
    return components, total_cost - tracked_total, _total_capex(network), total_cost


def _weighted_interest_rate(network, interest_rates) -> float | None:
    """Capex-weighted mean interest rate (falling back to capacity-weighted, then plain mean)."""
    if not interest_rates:
        return None
    total_weight = weighted_sum = 0.0
    for tech, rate in interest_rates.items():
        if rate is None:
            continue
        for component_type in ("generator", "link", "store"):
            table = _component_table(network, component_type)
            if tech not in table.index:
                continue
            capacity = _component_capacity(network, component_type, tech)
            if capacity <= 0:
                break
            try:
                capital_cost = float(table.loc[tech].get("capital_cost", 0.0))
            except Exception:  # noqa: BLE001
                capital_cost = 0.0
            if not np.isfinite(capital_cost):
                capital_cost = 0.0
            weight = capital_cost * capacity
            if weight > 0:
                total_weight += weight
                weighted_sum += weight * float(rate)
            break
    if total_weight > 0:
        return weighted_sum / total_weight

    total_capacity = weighted_sum = 0.0
    for tech, rate in interest_rates.items():
        if rate is None:
            continue
        for component_type in ("generator", "link", "store"):
            table = _component_table(network, component_type)
            if tech in table.index:
                capacity = _component_capacity(network, component_type, tech)
                if capacity > 0:
                    total_capacity += capacity
                    weighted_sum += capacity * float(rate)
                break
    if total_capacity > 0:
        return weighted_sum / total_capacity
    valid = [float(rate) for rate in interest_rates.values() if rate is not None]
    return sum(valid) / len(valid) if valid else None


def _link_capacity_on_output_basis(network, series: pd.Series) -> pd.Series:
    """Link ratings from PyPSA's input basis (bus0) to the output basis (bus1): MW_out = MW_in x eff."""
    links = network.links
    if links.empty or series.empty:
        return series
    column = "output_basis_efficiency" if "output_basis_efficiency" in links.columns else "efficiency"
    if column not in links.columns:
        return series
    return series.astype(float) * links[column].reindex(series.index).astype(float).fillna(1.0)


def store_reporting_scale(network, time_step: float = 1.0) -> float:
    """Factor from solved store energy to physical MWh (1 except in legacy_scaled networks)."""
    if getattr(network, "_temporal_accounting_mode", "legacy_scaled") == "snapshot_weighted":
        return 1.0
    return float(time_step)


def plant_results(
    network: pypsa.Network,
    *,
    time_step: float = 1.0,
    interest_rates: dict[str, float] | None = None,
    water_cost_usd_per_m3: float | None = None,
    water_usage_m3_per_t: float | None = None,
    land_cost_usd_per_km2_year: float | None = None,
    land_used_km2: float | None = None,
) -> dict:
    """Headline results of a solved plant: cost, LCOA, production, cost shares and capacities.

    Generator and link capacities are reported under the component name in MW (links on the
    output basis), stores in MWh; add_unit_suffixes() later appends the units to the names.
    Water and land costs given here are only recorded; accounting.py puts them in the headline.
    """
    out: dict = {}
    currency = str(getattr(network, "_currency", "USD")).strip().upper()
    slug = currency.lower()
    mode = getattr(network, "_temporal_accounting_mode", "legacy_scaled")
    snapshot_hours = float(getattr(network, "_snapshot_hours", time_step))
    if mode == "snapshot_weighted":
        objective_weights = network.snapshot_weightings["objective"].reindex(network.snapshots)
        simulated_hours = float(objective_weights.sum())
    else:
        objective_weights = pd.Series(snapshot_hours, index=network.snapshots)
        simulated_hours = float(len(network.snapshots) * snapshot_hours)

    def energy_mwh(series) -> float:
        return float((pd.Series(series, index=network.snapshots, dtype=float) * objective_weights).sum())

    load = float(network.loads.p_set.values[0])
    production_hours = simulated_hours if mode == "snapshot_weighted" else 8760.0
    production = load / AMMONIA_HHV_MWH_PER_T * production_hours
    total_cost = float(network.objective)
    out["currency"] = currency
    out["temporal_accounting_mode"] = mode
    out["snapshot_hours"] = snapshot_hours
    out["simulated_hours"] = simulated_hours
    out["is_full_year_result"] = bool(abs(simulated_hours - 8760.0) < 1e-6)
    out[f"lcoa_{slug}_per_t"] = total_cost / production if production > 0 else np.nan
    out["annual_ammonia_demand_mwh"] = load * production_hours
    out["annual_ammonia_production_t"] = production
    out[f"total_cost_{slug}_per_year"] = total_cost

    components, other_cost, total_capital, denominator = _component_cost_breakdown(network)
    if denominator == 0:
        denominator = total_cost
    other_cost = max(other_cost, 0.0)
    for label, values in components.items():
        out[f"tech_share_{label}_pct"] = values["total"] / denominator * 100 if denominator else np.nan
        out[f"lcoa_tech_{label}_{slug}_per_t"] = values["total"] / production if production > 0 else np.nan
    out["other_cost_pct"] = other_cost / denominator * 100 if denominator else np.nan
    out[f"lcoa_tech_other_{slug}_per_t"] = other_cost / production if production > 0 else np.nan
    out["capital_cost_share_pct"] = total_capital / denominator * 100 if denominator else np.nan

    if water_cost_usd_per_m3 is not None and production > 0:
        usage = 1.5 if water_usage_m3_per_t is None else float(water_usage_m3_per_t)
        out["water_cost_usd_per_m3"] = float(water_cost_usd_per_m3)
        out["water_usage_m3_per_t_nh3"] = usage
        out[f"water_cost_{slug}_per_t"] = float(water_cost_usd_per_m3) * usage
    if land_cost_usd_per_km2_year is not None and land_used_km2 is not None and production > 0:
        out["land_cost_usd_per_km2_year"] = float(land_cost_usd_per_km2_year)
        out["land_used_km2"] = float(land_used_km2)
        out[f"land_cost_{slug}_per_t"] = float(land_cost_usd_per_km2_year) * float(land_used_km2) / production

    if interest_rates:
        average = _weighted_interest_rate(network, interest_rates)
        if average is not None:
            out["interest_rate_overall"] = average
        for tech, rate in interest_rates.items():
            if rate is not None:
                out[f"interest_rate_{INTEREST_RATE_COLUMN_ALIASES.get(tech, tech)}"] = float(rate)

    for generator in network.generators.index:
        out[generator] = network.generators.loc[generator, "p_nom_opt"]
    link_capacities = _link_capacity_on_output_basis(network, network.links["p_nom_opt"])
    for link in network.links.index:
        out[link] = link_capacities.loc[link]
    # The penalty link is a fictitious relief valve: report the energy through it, not its size
    if "penalty_link" in network.links.index:
        flows = network.links_t.p0
        out["penalty_link"] = energy_mwh(flows["penalty_link"]) if "penalty_link" in flows.columns else 0.0
    if "grid" in network.generators.index:
        dispatch = network.generators_t.p
        out["grid_energy_mwh"] = energy_mwh(dispatch["grid"]) if "grid" in dispatch.columns else 0.0
    if not network.generators_t.p.empty:
        power_generators = [g for g in network.generators.index if network.generators.loc[g, "bus"] == "power"]
        out["power_bus_generator_supply_mwh"] = float(
            sum(energy_mwh(network.generators_t.p[g]) for g in power_generators)
        )
    if "battery_pcs_charge" in out or "battery_pcs_discharge" in out:
        charge = float(out.pop("battery_pcs_charge", 0.0) or 0.0)
        discharge = float(out.pop("battery_pcs_discharge", 0.0) or 0.0)
        out["battery_pcs_mw"] = max(charge, discharge)
    scale = store_reporting_scale(network, time_step)
    for store in network.stores.index:
        out[store] = network.stores.loc[store, "e_nom_opt"] * scale
    out["hydrogen_storage_capacity"] = (
        network.stores.loc["compressed_hydrogen_store", "e_nom_opt"] * scale / HYDROGEN_HHV_MWH_PER_T
    )
    return out


_MW_COMPONENTS = {
    "wind", "solar", "solar_tracking", "grid", "ramp_dummy", "electrolysis", "hydrogen_compression",
    "hydrogen_from_storage", "ammonia_synthesis", "hydrogen_fuel_cell",
}
# penalty_link is reported as the energy through it (MWh), not as a capacity
_MWH_COMPONENTS = {"ammonia", "compressed_hydrogen_store", "battery", "battery_storage", "accumulated_penalty", "penalty_link"}
_RENAMED = {
    "hydrogen_storage_capacity": "hydrogen_storage_capacity_t",
    "land_capacity_cap": "land_capacity_cap_mw",
}


def add_unit_suffixes(results: dict) -> dict:
    """A copy of plant results with units on component names (wind -> wind_mw, ammonia -> ammonia_mwh)."""
    renamed = {}
    for key, value in results.items():
        lower = str(key).lower()
        if lower in _MW_COMPONENTS:
            key = f"{lower}_mw"
        elif lower in _MWH_COMPONENTS:
            key = f"{lower}_mwh"
        else:
            key = _RENAMED.get(key, key)
        renamed[key] = value
    return renamed
