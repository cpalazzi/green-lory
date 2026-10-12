"""Technology costs: the technology YAML and how its costs reach the plant network.

A technology YAML (inputs/tech/tech_config_*.yaml) lists every plant component under `techs` with
its overnight cost split into equipment (`tech_cost_per_mw`) and installation, balance of plant
and grid (`build_cost_per_mw`), their total (`overnight_cost_per_mw`, checked), its lifetime and
fixed O&M as a fraction of capex. Interest rates come from the run, never from the YAML.
`extends: <other.yaml>` makes a YAML a short overlay of another. Top-level keys set the model
currency, the water use and baseline water price, and the exchange rate of the site-cost inputs.

The YAML also gives each link's conversion recipe: `carriers_in` and `carriers_out`, the flows
by bus at one operating point, in any common unit (per MW of input or per MW of output).
apply_link_efficiencies() turns a recipe into the link's `efficiency` (bus0 to bus1) and, for a
third port, `efficiency2`. A technology may also state operating values: `min_load_fraction`,
`ramp_limit_up` and `ramp_limit_down` (fractions of capacity per hour), `variable_cost_per_mwh`
and, for a store, `standing_loss_fraction_per_day` (apply_operating_parameters). The plant bundle
supplies only the topology and the formulation devices (prepare_base_network checks that the
YAML covers every other component).

Costs reach the network once per cell (apply_costs): each technology's equipment part times the
cell's equipment_cost_multiplier plus its build part times the build_cost_multiplier, annualised
at the cell's interest rate with the capital recovery factor plus fixed O&M.

The cell's interest rates come from the run: one rate for every technology and cell
(uniform_interest_overrides), or a finance table (finance_overrides). A finance table is keyed
by cell (lat, lon) or by country (iso3, reaching cells through model/countries.py); each row
gives one technology's rate, or every other technology's (tech "all"). Every technology at
every cell a run solves must get a rate. A site-cost table (model/site_costs.py) adds the
multipliers and the water price.

Links are rated on PyPSA's input side (bus0) while the YAML quotes output capacity, so link
costs are multiplied by the link efficiency. In the legacy_scaled time accounting, store costs
are also multiplied by the time step (store_cost_scale) and link variable costs by
24 x 366 / snapshots (plant.legacy_marginal_cost_scale).
"""
from __future__ import annotations

import copy
import logging
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from .config import DEFAULT_TECH_YAML, repo_path
from .plant import legacy_marginal_cost_scale, normalize_component_name

LOGGER = logging.getLogger(__name__)

# Location-level columns of the override table (not per technology)
LOCATION_COST_COLUMNS = ("land_cost_usd_per_km2_year", "water_cost_usd_per_m3")
# What every generator, link and store of a technology YAML must state (the costs by unit:
# tech_cost_per_mw, build_cost_per_mw and their total overnight_cost_per_mw, or per MWh for stores)
COST_KEYS = ("lifetime_years", "fixed_om_fraction")
COSTED_TYPES = ("generator", "link", "store")
# Operating values a technology may state: the network attribute each sets and the components it
# applies to. A technology without one has none (no minimum load, no ramp limit, no variable
# cost, no standing loss).
OPERATING_KEYS = {
    "min_load_fraction": ("p_min_pu", ("generators", "links")),
    "ramp_limit_up": ("ramp_limit_up", ("links",)),
    "ramp_limit_down": ("ramp_limit_down", ("links",)),
    "variable_cost_per_mwh": ("marginal_cost", ("generators", "links")),
    "standing_loss_fraction_per_day": ("standing_loss", ("stores",)),
}
# Components of the plant bundle that are formulation devices, not technologies: the switched-off
# grid and the generator feeding the ramp penalty. The YAML gives every other component's values.
FORMULATION_DEVICES = ("grid", "ramp_dummy")
TECH_META_KEYS = (
    "model_currency_price_year",
    "spatial_cost_currency",
    "spatial_cost_price_year",
    "spatial_cost_to_model_currency",
)


# ---------------------------------------------------------------------------------------------
# The technology YAML
# ---------------------------------------------------------------------------------------------

def _deep_merge(base: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge an overlay into a copy of base."""
    merged = copy.deepcopy(base)
    for key, value in overlay.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def load_tech_config(path: str | Path, _seen: frozenset[Path] = frozenset()) -> dict[str, Any]:
    """A technology YAML with its `extends` chain resolved (paths relative to the YAML)."""
    resolved = Path(path).expanduser().resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"Tech YAML not found: {resolved}")
    if resolved in _seen:
        chain = " -> ".join(str(item) for item in [*_seen, resolved])
        raise ValueError(f"Circular tech YAML extends chain: {chain}")
    raw = yaml.safe_load(resolved.read_text()) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"Tech YAML must contain a mapping: {resolved}")
    parent = raw.pop("extends", None)
    if parent is None:
        return raw
    parent_path = Path(parent)
    if not parent_path.is_absolute():
        parent_path = resolved.parent / parent_path
    return _deep_merge(load_tech_config(parent_path, _seen | {resolved}), raw)


def tech_yaml_files(path: str | Path) -> list[Path]:
    """The YAML and every file its `extends` chain reads, in order (for manifests)."""
    files = []
    current = Path(path).expanduser().resolve()
    while True:
        if current in files:
            raise ValueError(f"Circular tech YAML extends chain: {current}")
        files.append(current)
        parent = (yaml.safe_load(current.read_text()) or {}).get("extends")
        if parent is None:
            return files
        current = (current.parent / parent).resolve()


def load_tech_inputs(path: str | Path | None) -> dict[str, dict]:
    """The `techs` mapping of a technology YAML, keyed by plant-bundle component name."""
    resolved = repo_path(path) if path is not None else DEFAULT_TECH_YAML
    techs = load_tech_config(resolved).get("techs")
    if not isinstance(techs, dict):
        LOGGER.warning("Tech YAML %s has no top-level 'techs' mapping", resolved)
        return {}
    out = {}
    for name, raw in techs.items():
        normalized = normalize_component_name(name)
        if isinstance(raw, dict) and normalized is not None:
            out[normalized] = raw
    # Every costed component states its costs, lifetime and O&M: the model has no defaults of its own
    problems = []
    for name, raw in out.items():
        component_type = str(raw.get("component_type", "")).lower()
        if component_type not in COSTED_TYPES:
            continue
        unit = "mwh" if component_type == "store" else "mw"
        keys = (*COST_KEYS, f"tech_cost_per_{unit}", f"build_cost_per_{unit}", f"overnight_cost_per_{unit}")
        problems += [f"{name}.{key} is missing" for key in keys if key not in raw]
        if "interest_rate" in raw:
            problems.append(f"{name}.interest_rate: interest rates come from the run, not the YAML")
        parts = [raw.get(f"{part}_cost_per_{unit}") for part in ("tech", "build", "overnight")]
        if None not in parts and abs(float(parts[0]) + float(parts[1]) - float(parts[2])) > 1e-6 * max(1.0, abs(float(parts[2]))):
            problems.append(f"{name}: tech_cost_per_{unit} + build_cost_per_{unit} = {float(parts[0]) + float(parts[1]):g}, "
                            f"not overnight_cost_per_{unit} = {float(parts[2]):g}")
    if problems:
        raise ValueError(f"Tech YAML {resolved}: {'; '.join(problems)}")
    return out


def load_tech_meta(path: str | Path | None) -> dict[str, Any]:
    """The YAML's top-level settings: currency, water use and price, site-cost exchange rate."""
    resolved = repo_path(path) if path is not None else DEFAULT_TECH_YAML
    data = load_tech_config(resolved)
    meta: dict[str, Any] = {}
    if "currency" in data:
        meta["currency"] = str(data["currency"]).strip().upper()
    for key in TECH_META_KEYS:
        if key in data:
            meta[key] = data[key]
    if "water_usage_m3_per_t_nh3" in data:
        meta["water_usage_m3_per_t_nh3"] = float(data["water_usage_m3_per_t_nh3"])
    if "water_cost_baseline_usd_per_m3" in data:
        meta["water_cost_baseline_usd_per_m3"] = float(data["water_cost_baseline_usd_per_m3"])
    return meta


# ---------------------------------------------------------------------------------------------
# Annualised costs on the network
# ---------------------------------------------------------------------------------------------

def apply_link_efficiencies(network, tech_inputs: dict[str, dict]) -> None:
    """Set each link's conversion coefficients from its YAML recipe.

    efficiency = carriers_out[bus1] / carriers_in[bus0]. A third port (bus2: the power drawn by
    hydrogen compression, the hydrogen drawn by ammonia synthesis) gets efficiency2 =
    -carriers_in[bus2] / carriers_in[bus0], or +carriers_out[bus2] / carriers_in[bus0] for a
    product. Every link has a recipe (check_plant_coverage). Must run before the costs are
    set, which convert link costs with the efficiency.
    """
    for name in network.links.index:
        raw = tech_inputs.get(name) or {}
        carriers_in, carriers_out = raw.get("carriers_in"), raw.get("carriers_out")
        if not carriers_in or not carriers_out:
            continue
        bus0, bus1 = network.links.at[name, "bus0"], network.links.at[name, "bus1"]
        if bus0 not in carriers_in or bus1 not in carriers_out:
            raise ValueError(f"techs.{name}: the recipe must give carriers_in.{bus0} and carriers_out.{bus1}")
        flow_in = float(carriers_in[bus0])
        if flow_in <= 0 or float(carriers_out[bus1]) <= 0:
            raise ValueError(f"techs.{name}: carriers_in.{bus0} and carriers_out.{bus1} must be positive")
        network.links.at[name, "efficiency"] = float(carriers_out[bus1]) / flow_in
        bus2 = network.links.at[name, "bus2"] if "bus2" in network.links.columns else ""
        if isinstance(bus2, str) and bus2.strip():
            if bus2 in carriers_in:
                network.links.at[name, "efficiency2"] = -float(carriers_in[bus2]) / flow_in
            elif bus2 in carriers_out:
                network.links.at[name, "efficiency2"] = float(carriers_out[bus2]) / flow_in
            else:
                raise ValueError(f"techs.{name}: the link has bus2 '{bus2}' but the recipe omits it")


def check_plant_coverage(network, tech_inputs: dict[str, dict]) -> None:
    """Every generator, link and store of the bundle except the formulation devices needs its costs
    in the YAML, and every link its recipe: the bundle itself holds no technology values."""
    gaps = []
    for component, unit in (("generators", "mw"), ("links", "mw"), ("stores", "mwh")):
        for name in getattr(network, component).index:
            if name in FORMULATION_DEVICES:
                continue
            raw = tech_inputs.get(name) or {}
            if raw.get(f"tech_cost_per_{unit}") is None or raw.get(f"build_cost_per_{unit}") is None:
                gaps.append(f"{name}.tech_cost_per_{unit}/build_cost_per_{unit}")
            if component == "links" and not (raw.get("carriers_in") and raw.get("carriers_out")):
                gaps.append(f"{name}.carriers_in/carriers_out")
    if gaps:
        raise ValueError(f"The technology YAML lacks plant values: {', '.join(gaps)}")


def apply_operating_parameters(network, tech_inputs: dict[str, dict]) -> None:
    """Set minimum loads, ramp limits, variable costs and standing losses from the YAML.

    PyPSA applies a store's standing loss per hour, so a loss per day d becomes 1 - (1 - d)^(1/24).
    A link's variable cost (per MWh of its input) follows the legacy time scaling of the bundle's
    own variable costs (plant.legacy_marginal_cost_scale).
    """
    for name, raw in tech_inputs.items():
        for key, (attribute, components) in OPERATING_KEYS.items():
            if key not in raw:
                continue
            component = next((c for c in components if name in getattr(network, c).index), None)
            if component is None:
                raise ValueError(f"techs.{name}.{key}: the plant has no {' or '.join(components)} component {name}")
            value = float(raw[key])
            if key == "standing_loss_fraction_per_day":
                value = 1.0 - (1.0 - value) ** (1.0 / 24.0)
            elif key == "variable_cost_per_mwh" and component == "links":
                value *= legacy_marginal_cost_scale(network)
            getattr(network, component).at[name, attribute] = value


def prepare_base_network(network, tech_inputs: dict[str, dict]) -> None:
    """Give the bundle's network its technology values: efficiencies and operating values (the
    costs follow per cell, apply_costs)."""
    check_plant_coverage(network, tech_inputs)
    apply_link_efficiencies(network, tech_inputs)
    apply_operating_parameters(network, tech_inputs)


def annuity_factor(interest_rate: float, lifetime_years: float) -> float:
    """Capital recovery factor r (1 + r)^n / ((1 + r)^n - 1); 1/n at a zero rate."""
    if lifetime_years <= 0:
        raise ValueError("lifetime_years must be positive")
    if abs(interest_rate) < 1e-12:
        return 1.0 / lifetime_years
    factor = (1.0 + interest_rate) ** lifetime_years
    return interest_rate * factor / (factor - 1.0)


def annualised_capital_cost(
    overnight_cost: float, interest_rate: float, lifetime_years: float, fixed_om_fraction: float
) -> float:
    """Yearly cost of one unit of capacity: overnight cost x (annuity factor + fixed O&M fraction)."""
    crf = annuity_factor(float(interest_rate), float(lifetime_years))
    return float(overnight_cost) * (crf + float(fixed_om_fraction))


def store_cost_scale(time_step: float, temporal_accounting_mode: str) -> float:
    """Store-cost factor of the time accounting (the time step in legacy_scaled, else 1)."""
    return float(time_step) if temporal_accounting_mode == "legacy_scaled" else 1.0


def _set_annual_cost(network, name: str, component_type: str, annual: float, store_scale: float) -> None:
    if component_type == "generator":
        if name in network.generators.index:
            network.generators.loc[name, "capital_cost"] = annual
    elif component_type == "link":
        # YAML costs are per MW of output (bus1); PyPSA rates links on input (bus0)
        if name in network.links.index:
            efficiency = float(network.links.loc[name, "efficiency"])
            if efficiency <= 0:
                raise ValueError(f"Link '{name}' has non-positive efficiency; cannot compute capital_cost")
            network.links.loc[name, "capital_cost"] = annual * efficiency
    elif component_type == "store":
        if name in network.stores.index:
            network.stores.loc[name, "capital_cost"] = annual * store_scale


def apply_costs(
    network,
    tech_inputs: dict[str, dict],
    overrides: dict[str, dict[str, float]] | None,
    *,
    time_step: float,
    temporal_accounting_mode: str,
) -> None:
    """Set every technology's capital cost for one cell.

    The overnight cost is tech_cost x equipment_cost_multiplier + build_cost x
    build_cost_multiplier (multipliers of 1 where the cell has no site costs), annualised at the
    cell's interest rate, which the run's finance must give for every technology.
    """
    store_scale = store_cost_scale(time_step, temporal_accounting_mode)
    overrides = overrides or {}
    for name, raw in tech_inputs.items():
        component_type = str(raw.get("component_type", "")).lower()
        if component_type not in COSTED_TYPES:
            continue
        params = overrides.get(name) or {}
        if "interest_rate" not in params:
            raise ValueError(f"No interest rate for {name}: the run's finance must cover every technology")
        unit = "mwh" if component_type == "store" else "mw"
        overnight = (float(raw[f"tech_cost_per_{unit}"]) * float(params.get("equipment_cost_multiplier", 1.0))
                     + float(raw[f"build_cost_per_{unit}"]) * float(params.get("build_cost_multiplier", 1.0)))
        annual = annualised_capital_cost(
            overnight, float(params["interest_rate"]), float(raw["lifetime_years"]), float(raw["fixed_om_fraction"])
        )
        _set_annual_cost(network, name, component_type, annual, store_scale)


# ---------------------------------------------------------------------------------------------
# Finance: one interest rate per run, or a finance table
# ---------------------------------------------------------------------------------------------

ALL_TECHNOLOGIES = "all"
FINANCE_KEYS = {"cell": ("lat", "lon"), "country": ("iso3",)}
# Columns a finance table may not carry: they belong to the site-cost table
SITE_COST_COLUMNS = ("build_cost_multiplier", "equipment_cost_multiplier", *LOCATION_COST_COLUMNS)


def _cell_key(lat: float, lon: float) -> tuple[float, float]:
    return round(float(lat), 4), round(float(lon), 4)


def costed_technologies(tech_inputs: dict[str, dict]) -> list[str]:
    """The technologies that carry a capital cost, so need an interest rate."""
    return sorted(name for name, raw in tech_inputs.items() if str(raw.get("component_type", "")).lower() in COSTED_TYPES)


def load_finance_table(path: str | Path) -> tuple[pd.DataFrame, str]:
    """A finance table and its key: "cell" (columns lat, lon) or "country" (iso3).

    Each row gives the interest rate (a fraction) of one technology at a cell or in a country, or,
    with tech "all", of every technology the table does not name there. Other columns are kept
    for the reader and ignored, except site-cost columns, which belong to the site-cost table.
    """
    resolved = repo_path(path)
    frame = pd.read_csv(resolved)
    frame = frame.rename(columns={column: str(column).strip().lower() for column in frame.columns})
    keyed = [key for key, columns in FINANCE_KEYS.items() if set(columns) <= set(frame.columns)]
    if len(keyed) != 1:
        raise ValueError(f"{resolved}: a finance table is keyed either by cell (lat, lon) or by country (iso3)")
    key = keyed[0]
    missing = [column for column in ("tech", "interest_rate") if column not in frame.columns]
    forbidden = [column for column in SITE_COST_COLUMNS if column in frame.columns]
    if missing or forbidden:
        raise ValueError(f"{resolved}: missing columns {missing}; site-cost columns {forbidden} belong to the site-cost table")
    frame = frame[[*FINANCE_KEYS[key], "tech", "interest_rate"]].copy()
    tech = frame["tech"].astype(str).str.strip()
    frame["tech"] = [ALL_TECHNOLOGIES if name.lower() == ALL_TECHNOLOGIES else normalize_component_name(name) for name in tech]
    if key == "country":
        frame["iso3"] = frame["iso3"].astype(str).str.strip().str.upper()
    else:
        frame["lat"], frame["lon"] = frame["lat"].astype(float).round(4), frame["lon"].astype(float).round(4)
    rates = pd.to_numeric(frame["interest_rate"], errors="coerce")
    bad = frame[rates.isna() | (rates < 0) | (rates >= 1)]
    if len(bad):
        raise ValueError(f"{resolved}: interest_rate must be a fraction in [0, 1); e.g. {bad.head(3).to_dict('records')}")
    duplicated = frame[frame.duplicated([*FINANCE_KEYS[key], "tech"], keep=False)]
    if len(duplicated):
        raise ValueError(f"{resolved}: rows given twice, e.g. {duplicated.head(3).to_dict('records')}")
    frame["interest_rate"] = rates
    return frame, key


def finance_overrides(path: str | Path, cells, tech_inputs: dict[str, dict]) -> tuple[dict, dict]:
    """A finance table's interest rates for the cells, and the cells it leaves without one.

    Each technology at a cell takes the rate of its own row for the cell (or the cell's country,
    model/countries.py), or else the "all" row's. Returns ({cell: {tech: {"interest_rate": r}}},
    {cell: [technologies without a rate]}); a run must not solve a cell of the second.
    """
    table, key = load_finance_table(path)
    techs = costed_technologies(tech_inputs)
    unknown = sorted(set(table["tech"]) - set(techs) - {ALL_TECHNOLOGIES})
    if unknown:
        raise ValueError(f"{path}: technologies the plant does not have: {unknown}; use {techs} or 'all'")
    if key == "cell":
        keys = list(zip(table["lat"], table["lon"]))
    else:
        keys = list(table["iso3"])
        from .countries import cell_countries

        assigned = cell_countries()["iso3"]
        country_of = dict(zip((_cell_key(lat, lon) for lat, lon in assigned.index), assigned.to_numpy()))
    specific = {(k, tech): rate for k, tech, rate in zip(keys, table["tech"], table["interest_rate"]) if tech != ALL_TECHNOLOGIES}
    default = {k: rate for k, tech, rate in zip(keys, table["tech"], table["interest_rate"]) if tech == ALL_TECHNOLOGIES}
    resolved: dict = {}  # the rates of one key, shared read-only by its cells
    index, gaps = {}, {}
    for lat, lon in cells:
        cell = _cell_key(lat, lon)
        k = cell if key == "cell" else country_of.get(cell)
        if k not in resolved:
            rates = {tech: specific.get((k, tech), default.get(k)) for tech in techs}
            resolved[k] = ({tech: {"interest_rate": float(rate)} for tech, rate in rates.items() if rate is not None},
                           [tech for tech, rate in rates.items() if rate is None])
        rates, missing = resolved[k]
        if missing:
            gaps[cell] = missing
        else:
            index[cell] = rates
    return index, gaps


def describe_finance_gaps(gaps: dict, limit: int = 5) -> str:
    examples = "; ".join(f"({lat:g}, {lon:g}): {', '.join(techs)}" for (lat, lon), techs in list(gaps.items())[:limit])
    count = f"{len(gaps)} cell" + ("" if len(gaps) == 1 else "s")
    return f"{count} without an interest rate for some technologies, e.g. {examples}"


def uniform_interest_overrides(cells, tech_inputs: dict[str, dict], rate: float) -> dict[tuple[float, float], dict[str, dict[str, float]]]:
    """The interest rates of a run with one rate: every technology at every cell takes it (the
    cells share one read-only mapping)."""
    rates = {name: {"interest_rate": float(rate)} for name in costed_technologies(tech_inputs)}
    return {_cell_key(lat, lon): rates for lat, lon in cells}


def cell_overrides(index: dict, lat: float, lon: float) -> dict[str, dict[str, float]] | None:
    """One cell's overrides, or None."""
    return index.get(_cell_key(lat, lon)) or None


def interest_rates(overrides: dict[str, dict[str, float]] | None) -> dict[str, float]:
    """The cell's interest rate per technology."""
    rates = {}
    for tech, params in (overrides or {}).items():
        rate = params.get("interest_rate")
        name = normalize_component_name(tech)
        if rate is not None and name is not None:
            rates[name] = float(rate)
    return rates
