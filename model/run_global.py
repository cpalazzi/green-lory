"""Global orchestration helpers for looping the ammonia plant model over many locations."""
from __future__ import annotations

import argparse
import io
import logging
import math
import os
import sys
import time
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Sequence, Tuple

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import Point
import yaml

try:
    from . import main as plant_main
    from . import location_tools as lt
    from . import data_store as results_store
    from . import land_processing
except ImportError:  # pragma: no cover - fallback for direct execution
    PACKAGE_ROOT = Path(__file__).resolve().parent
    if str(PACKAGE_ROOT) not in sys.path:
        sys.path.insert(0, str(PACKAGE_ROOT))
    import main as plant_main  # type: ignore
    import location_tools as lt  # type: ignore
    import data_store as results_store  # type: ignore
    import land_processing  # type: ignore

LOGGER = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_WEATHER_DIR = REPO_ROOT / "data"
DEFAULT_LAND_CSV = DEFAULT_WEATHER_DIR / "20251222_max_capacities.csv"
DEFAULT_INTEREST_CSV = REPO_ROOT / "inputs" / "example_finance_overrides_spatial.csv"
DEFAULT_TECH_YAML = REPO_ROOT / "inputs" / "tech_config_ammonia_plant_2030_dea.yaml"
RENEWABLES = ["wind", "solar", "solar_tracking"]
_LAT_LON_TOLERANCE = 0.125  # match within 1/8th degree
# Techno-economic inputs are now expected to be pre-processed into the CSV bundle.
# Interest-rate defaults are therefore not sourced from a runtime YAML file.
DEFAULT_INTEREST_RATES: Dict[str, float] = {}

WIND_GENERATORS = ["wind"]


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


def _load_tech_inputs(path: str | Path | None) -> Dict[str, dict]:
    resolved = _resolve_path(path) or DEFAULT_TECH_YAML
    if not resolved.exists():
        LOGGER.warning("Tech YAML %s not found; financing overrides will not affect costs.", resolved)
        return {}
    data = yaml.safe_load(resolved.read_text()) or {}
    currency = data.get("currency")
    if currency and not os.environ.get("GREEN_LORY_CURRENCY"):
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
    return out


def _load_tech_meta(path: str | Path | None) -> Dict[str, float]:
    resolved = _resolve_path(path) or DEFAULT_TECH_YAML
    if not resolved.exists():
        return {}
    data = yaml.safe_load(resolved.read_text()) or {}
    meta: Dict[str, float] = {}
    if "water_usage_m3_per_t_nh3" in data:
        meta["water_usage_m3_per_t_nh3"] = float(data["water_usage_m3_per_t_nh3"])
    if "water_cost_baseline_usd_per_m3" in data:
        meta["water_cost_baseline_usd_per_m3"] = float(data["water_cost_baseline_usd_per_m3"])
    return meta


def _apply_finance_overrides(
    network,
    tech_inputs: Dict[str, dict],
    overrides: Dict[str, Dict[str, float]] | None,
    aggregation_count: int,
    time_step: float,
) -> None:
    """Apply per-tech interest_rate and build_cost_multiplier overrides by recomputing capital_cost.

    Convention:
    - YAML overnight costs are quoted on an HHV output basis.
    - Build cost multiplier applies to build_cost (labor + remoteness sensitive).
    - PyPSA Links are sized on an input basis (bus0 MW_in), so link capital_cost
      must be USD/MW_in/year.
    - Stores are scaled in generate_network() by time_step * aggregation_count;
      we mirror that scaling here so the objective remains consistent.
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
            overnight_base = raw.get("overnight_cost_per_mw")
            
            if overnight_base is None:
                continue
            
            # Use new split if available, otherwise fall back to total overnight cost
            if tech_cost is not None and build_cost is not None:
                effective_build_cost = float(build_cost) * float(build_mult)
                overnight = float(tech_cost) + effective_build_cost
            else:
                # Backward compat: if no split, apply multiplier to entire overnight cost
                overnight = float(overnight_base) * float(build_mult)
            
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
            overnight_base = raw.get("overnight_cost_per_mwh")
            
            if overnight_base is None:
                continue
            
            if tech_cost is not None and build_cost is not None:
                effective_build_cost = float(build_cost) * float(build_mult)
                overnight = float(tech_cost) + effective_build_cost
            else:
                overnight = float(overnight_base) * float(build_mult)
            
            annual = _annualised_capital_cost(overnight, float(rate), lifetime_years, fixed_om_fraction)
            if normalized in network.stores.index:
                network.stores.loc[normalized, "capital_cost"] = annual * float(time_step) * float(aggregation_count)


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
) -> Dict[str, float]:
    """Compute headline cost splits without double counting.

    Returns percentages of total annual cost for:
    - build_cost_pct: build portion of principal recovery
    - tech_cost_pct: tech portion of principal recovery
    - om_cost_pct: fixed O&M
    - interest_pct: interest portion of capital recovery
    Also returns weighted build_cost_multiplier_applied.
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
    store_scale = float(aggregation_count) * float(time_step)

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
        base_build_cost += build_cost_total_base
        weighted_build_cost += build_cost_total

    build_mult_applied = None
    if base_build_cost > 0:
        build_mult_applied = weighted_build_cost / base_build_cost

    return {
        "build_cost_multiplier_applied": build_mult_applied if build_mult_applied is not None else 1.0,
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
    total_cost_col = f"total_cost_{currency_code}_per_year"
    water_cost_col = f"water_cost_{currency_code}_per_t"
    land_cost_col = f"land_cost_{currency_code}_per_t"

    headline = [
        "latitude",
        "longitude",
        "country",
        "currency",
        lcoa_col,
        "annual_ammonia_demand_mwh",
        "annual_ammonia_production_t",
        total_cost_col,
        "build_cost_multiplier_applied",
        "build_cost_pct",
        "tech_cost_pct",
        "om_cost_pct",
        "interest_pct",
        "water_cost_pct",
        "land_cost_pct",
        "other_cost_pct",
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
        "bathymetry_depth_m",
        "area_cap_mw",
        "solar_area_used_km2",
        "wind_area_used_km2",
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
        "penalty_link_mw",
        "ammonia_mwh",
        "compressed_hydrogen_store_mwh",
        "battery_storage_mwh",
        "accumulated_penalty_mwh",
        "hydrogen_storage_capacity_t",
    ]

    component_costs = [
        col
        for col in df.columns
        if (col.startswith("cost_share_") or col.startswith("lcoa_component_"))
        and col != "cost_share_other_pct"
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
        LOGGER.warning("Interest CSV %s not found; continuing without overrides.", resolved)
        return pd.DataFrame(columns=["lat", "lon", "tech", "interest_rate"])
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
        legacy = resolved.parent / "20251222_land_max_capacity.csv"
        if path in {None, DEFAULT_LAND_CSV} and legacy.exists():
            resolved = legacy
        else:
            LOGGER.warning("Max-capacities CSV %s not found; skipping capacity caps.", resolved)
            return None
    df = pd.read_csv(resolved)
    df.columns = [col.lower() for col in df.columns]
    expected = {"latitude", "longitude"}
    if not expected.issubset(df.columns):
        LOGGER.warning("Land CSV %s is missing %s", resolved, expected - set(df.columns))
        return None
    return df


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


def _apply_spatial_solar_density_from_tech_config(
    land_df: pd.DataFrame | None,
    tech_inputs: Dict[str, dict],
) -> pd.DataFrame | None:
    if land_df is None or land_df.empty:
        return land_df
    if "latitude" not in land_df.columns:
        return land_df

    base_land_use = _solar_base_land_use_from_tech_inputs(tech_inputs)
    if base_land_use is None:
        return land_df

    updated = land_df.copy()
    updated["solar_density_mw_per_km2"] = land_processing._solar_density(
        updated["latitude"],
        scale=1.0,
        base_land_use_km2_per_mw=base_land_use,
    )

    if "onshore_area_km2" in updated.columns:
        updated["max_power_solar_mw"] = (
            updated["onshore_area_km2"] * updated["solar_density_mw_per_km2"]
        ).clip(lower=0.0)
        if "max_power_wind_mw" in updated.columns:
            updated["max_capacity_mw"] = (
                updated["max_power_wind_mw"]
                + updated["max_power_solar_mw"]
            )
        # Backward-compatible aliases used by older plotting/report helpers.
        updated["solar_max_capacity"] = updated["max_power_solar_mw"]

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


def _match_land_row(land_df: pd.DataFrame, lat: float, lon: float) -> pd.Series | None:
    if land_df is None:
        return None
    mask = (land_df["latitude"] - lat).abs() <= _LAT_LON_TOLERANCE
    mask &= (land_df["longitude"] - lon).abs() <= _LAT_LON_TOLERANCE
    subset = land_df.loc[mask]
    if subset.empty:
        return None
    return subset.iloc[0]


LEGACY_POWER_CAP_COLUMN_MAP = {
    "solar_max_capacity": "solar",
    "wind_max_capacity": "wind",
}


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

    for legacy_col, tech in LEGACY_POWER_CAP_COLUMN_MAP.items():
        if tech in power_caps:
            continue
        value = row.get(legacy_col)
        if pd.notna(value):
            power_caps[tech] = max(0.0, float(value))

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
    has_legacy_power_caps = any(
        pd.notna(row.get(col))
        for col in LEGACY_POWER_CAP_COLUMN_MAP
    )
    if has_explicit_power_caps:
        for tech in WIND_GENERATORS + ["solar"]:
            power_caps.setdefault(tech, 0.0)
    elif has_legacy_power_caps:
        for tech in WIND_GENERATORS:
            power_caps.setdefault(tech, 0.0)

    if power_caps or energy_caps:
        total_cap = _apply_component_caps(network, power_caps, energy_caps)
        return (total_cap if total_cap > 0 else None), row

    fallback = row.get("max_capacity_mw")
    if pd.isna(fallback):
        fallback = row.get("max_capacity")
    if pd.notna(fallback):
        fallback_val = max(0.0, float(fallback))
        fallback_caps = {
            "wind": fallback_val,
            "solar": fallback_val,
        }
        total_cap = _apply_component_caps(network, fallback_caps, {})
        return (total_cap if total_cap > 0 else None), row

    return None, row


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
    if onshore is None:
        onshore = _maybe("onshore_fraction", 100.0)
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

    bathymetry_depth = _maybe("bathymetry_depth_m")
    if bathymetry_depth is not None:
        metadata["bathymetry_depth_m"] = bathymetry_depth

    max_capacity = _maybe("max_capacity_mw")
    if max_capacity is None:
        max_capacity = _maybe("max_capacity")
    if max_capacity is not None:
        metadata["area_cap_mw"] = max_capacity

    solar_area = _maybe("solar_area_km2")
    if solar_area is None:
        solar_area = _maybe("onshore_area_km2")
    if solar_area is not None:
        metadata["solar_area_used_km2"] = solar_area

    wind_area = _maybe("wind_area_km2")
    if wind_area is None:
        wind_area = _maybe("onshore_wind_area_km2")
    if wind_area is not None:
        metadata["wind_area_used_km2"] = wind_area

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
    point = Point(lon, lat)
    matches = world[world.intersects(point)]
    if matches.empty:
        return "None"
    return matches.iloc[0].country


def _build_weather_frame(dataset: lt.all_locations, lat: float, lon: float, aggregation_count: int) -> pd.DataFrame:
    location = lt.renewable_data(
        dataset,
        latitude=lat,
        longitude=lon,
        renewables=RENEWABLES,
        aggregation_variable=aggregation_count,
    )
    frame = location.concat.copy()
    if "Weights" in frame.columns:
        frame = frame.drop(columns="Weights")
    if "wind" not in frame.columns and "onshore_wind" in frame.columns:
        frame["wind"] = frame["onshore_wind"]
    if "solar_tracking" not in frame.columns and "solar" in frame.columns:
        frame["solar_tracking"] = frame["solar"]
    defaults = {"grid": 0.0, "ramp_dummy": 1.0}
    for column, default_value in defaults.items():
        if column not in frame.columns:
            frame[column] = default_value
    frame = frame.reset_index(drop=True)
    return frame


def _default_locations(land_df: pd.DataFrame | None) -> List[Tuple[float, float]]:
    if land_df is None:
        raise ValueError("A max-capacities CSV is required when no explicit locations are supplied.")
    if "max_capacity_mw" in land_df.columns:
        filtered = land_df[land_df["max_capacity_mw"] > 0]
    elif "max_capacity" in land_df.columns:
        filtered = land_df[land_df["max_capacity"] > 0]
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
    if not tech_inputs:
        return None

    land_used_km2 = 0.0
    solar_density = None
    if land_row is not None:
        raw_density = land_row.get("solar_density_mw_per_km2")
        if pd.notna(raw_density) and float(raw_density) > 0:
            solar_density = float(raw_density)

    total_solar_capacity = 0.0
    for gen_name in ("solar", "solar_tracking"):
        if gen_name not in network.generators.index:
            continue
        value = network.generators.at[gen_name, "p_nom_opt"]
        if pd.isna(value):
            continue
        total_solar_capacity += max(0.0, float(value))

    if total_solar_capacity > 0:
        if solar_density is not None and solar_density > 0:
            land_used_km2 += total_solar_capacity / solar_density
        else:
            land_use = tech_inputs.get("solar", {}).get("land_use_km2_per_mw")
            if land_use is not None:
                land_used_km2 += total_solar_capacity * float(land_use)

    for gen_name in ("wind",):
        if gen_name not in network.generators.index:
            continue
        value = network.generators.at[gen_name, "p_nom_opt"]
        if pd.isna(value):
            continue
        capacity = max(0.0, float(value))
        if capacity <= 0:
            continue
        land_use = tech_inputs.get("wind", {}).get("land_use_km2_per_mw")
        if land_use is not None:
            land_used_km2 += capacity * float(land_use)

    return land_used_km2


def _run_single_location(
    lat: float,
    lon: float,
    dataset: lt.all_locations,
    interest_df: pd.DataFrame,
    tech_inputs: Dict[str, dict],
    tech_meta: Dict[str, float],
    land_df: pd.DataFrame | None,
    aggregation_count: int,
    time_step: float,
    max_snapshots: int | None,
    quiet: bool,
) -> Tuple[str, Dict[str, Any] | None]:
    try:
        weather_frame = _build_weather_frame(dataset, lat, lon, aggregation_count)
        if max_snapshots is not None:
            weather_frame = weather_frame.iloc[:max_snapshots].copy()
    except Exception as exc:  # noqa: BLE001
        LOGGER.warning("Skipping location (%s, %s) due to weather extraction error: %s", lat, lon, exc)
        return "weather", None

    overrides = _interest_overrides(interest_df, lat, lon)
    interest_rates = _compose_interest_rates(overrides)

    # Extract location-level spatial cost parameters from overrides.
    location_params = overrides.get("__location__", {}) if overrides else {}
    water_cost_usd_per_m3 = location_params.get("water_cost_usd_per_m3")
    land_cost_usd_per_km2_year = location_params.get("land_cost_usd_per_km2_year")

    # Apply YAML defaults if location-specific values are missing.
    if water_cost_usd_per_m3 is None:
        water_cost_usd_per_m3 = tech_meta.get("water_cost_baseline_usd_per_m3")
    water_usage_m3_per_t = tech_meta.get("water_usage_m3_per_t_nh3")

    network = plant_main.generate_network(
        len(weather_frame),
        "basic_ammonia_plant",
        aggregation_count=aggregation_count,
        time_step=time_step,
        tech_config_overrides=overrides,
    )
    _apply_finance_overrides(
        network,
        tech_inputs=tech_inputs,
        overrides=overrides,
        aggregation_count=aggregation_count,
        time_step=time_step,
    )
    land_cap, land_row = _apply_land_caps(network, land_df, lat, lon)

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
            )
    except Exception as exc:  # noqa: BLE001
        LOGGER.warning("Optimisation failed for (%s, %s): %s", lat, lon, exc)
        return "solver", None

    land_used_km2 = _estimate_land_used_km2(network, land_row, tech_inputs)
    if land_used_km2 is not None:
        results["land_used_km2"] = float(land_used_km2)
        production_t = results.get("annual_ammonia_production_t")
        if (
            land_cost_usd_per_km2_year is not None
            and production_t is not None
            and float(production_t) > 0
        ):
            currency_code = str(results.get("currency", "USD")).strip().lower()
            land_cost_per_t = float(land_cost_usd_per_km2_year) * float(land_used_km2) / float(production_t)
            results["land_cost_usd_per_km2_year"] = float(land_cost_usd_per_km2_year)
            results[f"land_cost_{currency_code}_per_t"] = land_cost_per_t

    # Backfill explicit currency columns if legacy outputs are returned.
    currency_code = os.environ.get("GREEN_LORY_CURRENCY", "USD").strip().upper()
    currency_slug = currency_code.lower()
    if f"lcoa_{currency_slug}_per_t" not in results:
        if "lcoa_currency_per_t" in results:
            results[f"lcoa_{currency_slug}_per_t"] = results["lcoa_currency_per_t"]
        elif "lcoa_usd_per_t" in results:
            results[f"lcoa_{currency_slug}_per_t"] = results["lcoa_usd_per_t"]
    if "currency" not in results:
        results["currency"] = currency_code
    if (
        f"total_cost_{currency_slug}_per_year" not in results
        and "total_cost_currency_per_year" in results
    ):
        results[f"total_cost_{currency_slug}_per_year"] = results[
            "total_cost_currency_per_year"
        ]
    if (
        f"total_cost_{currency_slug}_per_year" not in results
        and "total_cost_usd_per_year" in results
    ):
        results[f"total_cost_{currency_slug}_per_year"] = results[
            "total_cost_usd_per_year"
        ]

    if land_cap is not None:
        results["land_capacity_cap"] = land_cap
    if land_row is not None:
        results.update(_land_metadata_from_row(land_row))
    results["interest_overrides_applied"] = bool(overrides)

    # Headline cost splits (no double counting).
    headline_splits = _compute_headline_splits(
        network,
        tech_inputs,
        overrides,
        aggregation_count=aggregation_count,
        time_step=time_step,
    )
    if headline_splits:
        results.update(headline_splits)

    # Add water/land cost percentages and rescale headline splits to include them.
    currency_code = str(results.get("currency", "USD")).strip().lower()
    total_cost_col = f"total_cost_{currency_code}_per_year"
    water_cost_col = f"water_cost_{currency_code}_per_t"
    land_cost_col = f"land_cost_{currency_code}_per_t"

    production = results.get("annual_ammonia_production_t")
    total_cost = results.get(total_cost_col)
    water_cost_per_t = results.get(water_cost_col)
    land_cost_per_t = results.get(land_cost_col)

    if production and total_cost:
        extra_cost = 0.0
        if water_cost_per_t is not None:
            extra_cost += float(water_cost_per_t) * float(production)
        if land_cost_per_t is not None:
            extra_cost += float(land_cost_per_t) * float(production)

        total_with_extras = float(total_cost) + extra_cost
        if total_with_extras > 0:
            if water_cost_per_t is not None:
                results["water_cost_pct"] = float(water_cost_per_t) * float(production) / total_with_extras * 100.0
            if land_cost_per_t is not None:
                results["land_cost_pct"] = float(land_cost_per_t) * float(production) / total_with_extras * 100.0

            # Rescale headline splits to keep 100% stack including water/land.
            scale = float(total_cost) / total_with_extras
            for key in ("build_cost_pct", "tech_cost_pct", "om_cost_pct", "interest_pct"):
                if key in results and results[key] is not None:
                    results[key] = float(results[key]) * scale

    return "done", results


def run_global(
    locations: Sequence[Tuple[float, float]] | None = None,
    weather_dir: str | Path | None = None,
    land_csv: str | Path | None = None,
    interest_csv: str | Path | None = None,
    tech_yaml: str | Path | None = None,
    aggregation_count: int = 1,
    time_step: float = 1.0,
    max_snapshots: int | None = None,
    output_csv: str | Path | None = None,
    quiet: bool = False,
    threads_per_worker: int | None = None,
    lon_min: float | None = None,
    lon_max: float | None = None,
) -> pd.DataFrame:
    """Run the ammonia plant optimisation for every requested location (serial).

    Args:
        locations: Iterable of (lat, lon) pairs. If omitted, the function iterates over every
            coordinate contained in the max-capacities CSV.
        weather_dir: Directory containing the NetCDF stacks consumed by `location_tools`.
        land_csv: CSV with at least `Latitude` and `Longitude` columns plus any available
            max-capacity metadata.
        interest_csv: CSV with `lat`, `lon`, `tech`, `interest_rate` columns to override
            financing assumptions per technology. Additional spatial columns such as
            `build_cost_multiplier`, `land_cost_usd_per_km2_year`, and
            `water_cost_usd_per_m3` are supported when present.
        tech_yaml: Optional tech-config YAML used to source financing defaults and currency metadata.
        aggregation_count: Snapshot aggregation factor forwarded to the weather loader.
        time_step: Duration of each snapshot in hours.
        max_snapshots: Optional hard cap on the number of weather snapshots to simulate per
            location (useful for quick smoke tests and notebooks).
        output_csv: Optional destination for the aggregated results table.
        threads_per_worker: Solver thread count (forwarded to GREEN_LORY_SOLVER_THREADS).
        lon_min: Optional minimum longitude bound (inclusive) for location filtering.
        lon_max: Optional maximum longitude bound (exclusive) for location filtering.
    """
    requested_threads = 1 if threads_per_worker is None else int(threads_per_worker)

    with _quiet_logging(quiet), _override_env("GREEN_LORY_SOLVER_LOG", "0", quiet):
        weather_dir_path = _resolve_path(weather_dir) or DEFAULT_WEATHER_DIR
        land_csv_path = _resolve_path(land_csv or DEFAULT_LAND_CSV)
        interest_csv_path = _resolve_path(interest_csv or DEFAULT_INTEREST_CSV)
        tech_yaml_path = _resolve_path(tech_yaml or DEFAULT_TECH_YAML)

        tech_inputs = _load_tech_inputs(tech_yaml_path)
        tech_meta = _load_tech_meta(tech_yaml_path)
        land_df = _load_land_table(land_csv_path)
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
            dataset = lt.all_locations(str(weather_dir_path), cache_resources=True)
            interest_df = _load_interest_table(interest_csv_path)

            # Open output file for incremental writing (provides partial-results
            # recovery on crash and progress monitoring via file size).
            out_fh: Any = None
            header_written = False
            if output_csv:
                output_path = _resolve_path(output_csv) or Path(output_csv)
                output_path.parent.mkdir(parents=True, exist_ok=True)
                out_fh = open(output_path, "w", newline="", encoding="utf-8")

            try:
                for lat, lon in location_list:
                    status, results = _run_single_location(
                        lat=lat,
                        lon=lon,
                        dataset=dataset,
                        interest_df=interest_df,
                        tech_inputs=tech_inputs,
                        tech_meta=tech_meta,
                        land_df=land_df,
                        aggregation_count=aggregation_count,
                        time_step=time_step,
                        max_snapshots=max_snapshots,
                        quiet=quiet,
                    )
                    if results is not None:
                        country = _country_for(world, lat, lon)
                        store.add_location(lat, lon, country, results)
                        if out_fh is not None:
                            row = {"latitude": lat, "longitude": lon, "country": country, **results}
                            row_df = pd.DataFrame([row])
                            if not header_written:
                                row_df.to_csv(out_fh, index=False)
                                header_written = True
                            else:
                                row_df.to_csv(out_fh, index=False, header=False)
                            out_fh.flush()
                    progress.update(lat, lon, status)
            finally:
                dataset.close()
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
        "--interest-csv",
        type=str,
        default=None,
        help=(
            "CSV with lat,lon,tech,interest_rate overrides; may also include "
            "build_cost_multiplier, land_cost_usd_per_km2_year, and water_cost_usd_per_m3."
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
        interest_csv=args.interest_csv,
        land_csv=args.land_csv,
        max_snapshots=args.max_snapshots,
        output_csv=args.output_csv,
        quiet=args.quiet,
        threads_per_worker=args.threads_per_worker,
        lon_min=args.lon_min,
        lon_max=args.lon_max,
    )
    print(f"Processed {len(results_df)} locations. Results saved to {args.output_csv}.")
