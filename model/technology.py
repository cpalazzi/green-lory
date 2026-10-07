"""Technology costs: the technology YAML and how its costs reach the plant network.

A technology YAML (inputs/tech_config_*.yaml) lists every plant component under `techs` with
its overnight cost split into equipment (`tech_cost_per_mw`) and installation, balance of plant
and grid (`build_cost_per_mw`), its lifetime, fixed O&M as a fraction of capex and a default
interest rate. `extends: <other.yaml>` makes a YAML a short overlay of another. Top-level keys
set the model currency, the water use and baseline water price, and the exchange rate of the
site-cost inputs.

The YAML also gives each link's conversion recipe: `carriers_in` and `carriers_out`, the flows
by bus at one operating point, in any common unit (per MW of input or per MW of output).
apply_link_efficiencies() turns a recipe into the link's `efficiency` (bus0 to bus1) and, for a
third port, `efficiency2`; the plant bundle supplies only the topology.

Costs reach the network in two steps for each cell:
1. apply_base_costs() annualises every YAML cost with the capital recovery factor plus fixed O&M.
2. apply_cell_overrides() replaces the interest rate, and multiplies the installation part by
   the cell's build_cost_multiplier, wherever the cell's override table gives a value.

The override table (columns lat, lon, tech, interest_rate and optionally
build_cost_multiplier, water_cost_usd_per_m3, land_cost_usd_per_km2_year) is read once and
indexed by cell (index_override_table, cell_overrides).

Links are rated on PyPSA's input side (bus0) while the YAML quotes output capacity, so link
costs are multiplied by the link efficiency. In the legacy_scaled time accounting, store costs
are also multiplied by the time step (see plant.py).
"""
from __future__ import annotations

import copy
import logging
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from .config import DEFAULT_TECH_YAML, repo_path
from .plant import normalize_component_name

LOGGER = logging.getLogger(__name__)

# Location-level columns of the override table (not per technology)
LOCATION_COST_COLUMNS = ("land_cost_usd_per_km2_year", "water_cost_usd_per_m3")
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
    product. Links without a recipe keep the bundle's values. Must run before the costs are
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


def apply_base_costs(
    network, tech_inputs: dict[str, dict], *, time_step: float, temporal_accounting_mode: str
) -> None:
    """Set every component's capital cost from the YAML's overnight cost and default interest rate."""
    store_scale = store_cost_scale(time_step, temporal_accounting_mode)
    for name, raw in tech_inputs.items():
        component_type = str(raw.get("component_type", "")).lower()
        unit = "mwh" if component_type == "store" else "mw"
        overnight = raw.get(f"overnight_cost_per_{unit}")
        if component_type not in {"generator", "link", "store"} or overnight is None:
            continue
        annual = annualised_capital_cost(
            float(overnight),
            float(raw.get("interest_rate", 0.07)),
            float(raw.get("lifetime_years", 20.0)),
            float(raw.get("fixed_om_fraction", 0.0)),
        )
        _set_annual_cost(network, name, component_type, annual, store_scale)


def apply_cell_overrides(
    network,
    tech_inputs: dict[str, dict],
    overrides: dict[str, dict[str, float]] | None,
    *,
    time_step: float,
    temporal_accounting_mode: str,
) -> None:
    """Recompute the capital cost of each technology the cell overrides.

    The overnight cost becomes tech_cost + build_cost x build_cost_multiplier, annualised at
    the cell's interest rate. Called after apply_base_costs(); technologies without an
    override row keep their base cost.
    """
    if not overrides:
        return
    store_scale = store_cost_scale(time_step, temporal_accounting_mode)
    for tech, params in overrides.items():
        if tech == "__location__" or params.get("interest_rate") is None:
            continue
        name = normalize_component_name(tech)
        raw = tech_inputs.get(name) if name is not None else None
        if not raw:
            continue
        component_type = str(raw.get("component_type", "")).lower()
        if component_type not in {"generator", "link", "store"}:
            continue
        unit = "mwh" if component_type == "store" else "mw"
        tech_cost, build_cost = raw.get(f"tech_cost_per_{unit}"), raw.get(f"build_cost_per_{unit}")
        if tech_cost is None or build_cost is None:
            raise ValueError(
                f"Tech '{name}' must define tech_cost_per_{unit} and build_cost_per_{unit}; "
                "unsplit overnight-cost inputs are no longer supported."
            )
        overnight = float(tech_cost) + float(build_cost) * float(params.get("build_cost_multiplier", 1.0))
        annual = annualised_capital_cost(
            overnight,
            float(params["interest_rate"]),
            float(raw.get("lifetime_years", 20.0)),
            float(raw.get("fixed_om_fraction", 0.0)),
        )
        _set_annual_cost(network, name, component_type, annual, store_scale)


# ---------------------------------------------------------------------------------------------
# The per-cell override table
# ---------------------------------------------------------------------------------------------

def load_override_table(path: str | Path | None) -> pd.DataFrame:
    """The override CSV with lower-case key columns and component names normalised."""
    if path is None:
        return pd.DataFrame(columns=["lat", "lon", "tech", "interest_rate"])
    resolved = repo_path(path)
    if not resolved.exists():
        raise FileNotFoundError(f"Requested override CSV not found: {resolved}")
    frame = pd.read_csv(resolved)
    columns = {column.lower(): column for column in frame.columns}
    for required in ("lat", "lon", "tech", "interest_rate"):
        if required not in columns:
            raise ValueError(f"Override CSV {resolved} must include '{required}' column (found {list(frame.columns)})")
    frame = frame.rename(columns={columns[key]: key for key in ("lat", "lon", "tech", "interest_rate")})
    frame["tech"] = frame["tech"].map(normalize_component_name)
    return frame


def _cell_key(lat: float, lon: float) -> tuple[float, float]:
    return round(float(lat), 4), round(float(lon), 4)


def index_override_table(frame: pd.DataFrame) -> dict[tuple[float, float], dict[str, dict[str, float]]]:
    """Index an override table by cell: {(lat, lon): {tech: {...}, "__location__": {...}}}."""
    index: dict[tuple[float, float], dict[str, dict[str, float]]] = {}
    for _, row in frame.iterrows():
        entry = index.setdefault(_cell_key(row["lat"], row["lon"]), {})
        tech = entry.setdefault(row["tech"], {})
        for column in ("interest_rate", "build_cost_multiplier"):
            if column in row.index and pd.notna(row[column]):
                tech[column] = float(row[column])
        location = entry.setdefault("__location__", {})
        for column in LOCATION_COST_COLUMNS:
            if column in row.index and pd.notna(row[column]):
                location[column] = float(row[column])
    for entry in index.values():
        if not entry.get("__location__"):
            entry.pop("__location__", None)
    return index


def cell_overrides(index: dict, lat: float, lon: float) -> dict[str, dict[str, float]] | None:
    """One cell's overrides, or None."""
    return index.get(_cell_key(lat, lon)) or None


def interest_rates(overrides: dict[str, dict[str, float]] | None) -> dict[str, float]:
    """The cell's interest rate per component (only those the override table sets)."""
    rates = {}
    for tech, params in (overrides or {}).items():
        rate = params.get("interest_rate")
        name = normalize_component_name(tech)
        if rate is not None and name is not None:
            rates[name] = float(rate)
    return rates
