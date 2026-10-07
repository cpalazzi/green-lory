"""Land: how much wind and PV a cell can hold, and how that limits its ammonia capacity.

A land table (built by land_table.py, one row per 1-degree cell, centred on the cell) gives
each cell's area suitable for wind, for PV and for either of them (their union), and the power
density of each technology. load_land_table() reads and checks a table; a cell's row becomes a
RenewableLandBudget.

Wind and PV may share land. Under the ``colocated`` allocation only a small exclusive fraction
of the wind footprint (pads, roads, substations; 3 % from Denholm et al. 2009) competes with PV
for the shared budget; under ``exclusive`` all of it does.

Two capacity rules turn a budget into ammonia capacity:
- ``scaled_reference_design`` (campaign runs): solve the cost-optimal 1 Mt/yr plant without land
  limits, then scale it until wind, PV or their shared footprint fills the budget. The LCOA does
  not depend on land; the capacity does (estimate_scaled_design_capacity).
- ``solved_quantity`` (finite-site supply curves): enforce the budget in the solve
  (apply_land_budget, add_land_constraints) and report the quantity solved, which is a feasible
  quantity, not a maximum (report_land_feasible_quantity).

The union of the wind and PV areas is not known pixel by pixel. Version 1 of the land table
assumes the smaller eligible area of each MODIS class lies inside the larger one and sums the
classwise maximum: the smallest possible union, a lower bound.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Mapping

import numpy as np
import pandas as pd

from .config import repo_path

# Provenance of the renewable-union area in a land table (written by land_table.py)
CLASSWISE_NESTED_UNION_METHOD = "classwise_nested_overlap_lower_bound"
CLASSWISE_NESTED_UNION_VERSION = "v1"
CLASSWISE_NESTED_UNION_SOURCE = "explicit_classwise_nested_v1_lower_bound"
CLASSWISE_NESTED_UNION_AVAILABILITY_COLUMN = "renewable_union_availability_classwise_nested_v1"
CLASSWISE_NESTED_UNION_AREA_COLUMN = "renewable_union_area_km2_classwise_nested_v1"
RENEWABLE_UNION_METHOD_COLUMN = "renewable_union_method"
RENEWABLE_UNION_METHOD_VERSION_COLUMN = "renewable_union_method_version"

# Land tables must be centred builds (23 September 2026 or later); earlier tables labelled each
# cell by the MODIS band one degree south of it.
REQUIRED_CELL_ANCHOR = "center"

LandAllocation = Literal["colocated", "exclusive"]
LAND_ALLOCATIONS: tuple[LandAllocation, ...] = ("colocated", "exclusive")

# Fraction of the wind footprint that excludes PV (Denholm et al. 2009, NREL/TP-6A2-45834,
# Table 1: direct impact 1.0 +/- 0.7 ha/MW of a 34.5 +/- 22.4 ha/MW total). A technology
# YAML can override it (wind.land_use_exclusive_fraction).
# TODO: replace by a co-location study value that also covers PV row losses from turbine
# shading, wake spacing and access setbacks.
DEFAULT_WIND_LAND_EXCLUSIVE_FRACTION = 0.03

UnionAreaSource = Literal["explicit_classwise_nested_v1_lower_bound", "explicit_legacy_unversioned", "explicit", "missing"]

_EPS = 1e-9


def normalise_land_allocation(name: str) -> LandAllocation:
    """Validate the allocation name (``colocated`` or ``exclusive``)."""
    if name in LAND_ALLOCATIONS:
        return name  # type: ignore[return-value]
    raise ValueError(f"Unknown land_allocation {name!r}; choose from {LAND_ALLOCATIONS}")


def effective_wind_land_exclusive_fraction(land_allocation: LandAllocation, budget: "RenewableLandBudget") -> float:
    """Fraction of the wind footprint that counts against the shared PV/wind budget."""
    if land_allocation == "exclusive":
        return 1.0
    return float(budget.wind_land_exclusive_fraction)


def _number(mapping: Any, *names: str) -> float | None:
    """The first finite number stored under one of ``names``."""
    for name in names:
        try:
            value = mapping.get(name)
        except (AttributeError, TypeError):
            value = None
        if value is None:
            continue
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(number):
            return number
    return None


def _nonnegative(value: float | None) -> float | None:
    return None if value is None else max(0.0, float(value))


def _positive(value: float | None) -> float | None:
    return None if value is None or value <= 0.0 else float(value)


def _tech_land_use(tech_inputs: Mapping[str, Any], tech: str) -> float | None:
    raw = tech_inputs.get(tech)
    return _positive(_number(raw, "land_use_km2_per_mw")) if isinstance(raw, Mapping) else None


# ---------------------------------------------------------------------------------------------
# A cell's land budget
# ---------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class RenewableLandBudget:
    """One cell's areas for wind, PV and their union (km2), and the power densities (MW/km2)."""

    wind_area_km2: float | None
    solar_area_km2: float | None
    renewable_union_area_km2: float | None
    wind_density_mw_per_km2: float | None
    fixed_solar_density_mw_per_km2: float | None
    tracking_solar_density_mw_per_km2: float | None
    union_area_source: UnionAreaSource
    wind_area_source: str
    tracking_density_source: str
    wind_land_exclusive_fraction: float = DEFAULT_WIND_LAND_EXCLUSIVE_FRACTION

    @property
    def tracking_land_multiplier(self) -> float:
        """Tracking-PV land per MW relative to fixed-PV land per MW."""
        fixed, tracking = _positive(self.fixed_solar_density_mw_per_km2), _positive(self.tracking_solar_density_mw_per_km2)
        return math.nan if fixed is None or tracking is None else fixed / tracking

    @classmethod
    def from_land_row(cls, land_row: Any, tech_inputs: Mapping[str, Any], *, onshore: bool = True) -> "RenewableLandBudget":
        return build_renewable_land_budget(land_row, tech_inputs, onshore=onshore)


def build_renewable_land_budget(land_row: Any, tech_inputs: Mapping[str, Any], *, onshore: bool = True) -> RenewableLandBudget:
    """A cell's budget from its land-table row and the technology YAML.

    The table carries the latitude-adjusted fixed-PV density; tracking PV gets the fixed
    density times the YAML's fixed/tracking land-use ratio.
    """
    if land_row is None:
        raise ValueError("land_row is required to construct a renewable land budget")

    if onshore:
        wind_area = _nonnegative(_number(land_row, "wind_onshore_area_km2"))
        wind_area_source = "wind_onshore_area_km2"
        if wind_area is None:
            wind_area = _nonnegative(_number(land_row, "wind_area_km2"))
            wind_area_source = "wind_area_km2"
    else:
        wind_area = _nonnegative(_number(land_row, "wind_area_km2"))
        wind_area_source = "wind_area_km2"
    solar_area = _nonnegative(_number(land_row, "solar_area_km2"))

    wind_land_use = _tech_land_use(tech_inputs, "wind")
    fixed_land_use = _tech_land_use(tech_inputs, "solar")
    tracking_land_use = _tech_land_use(tech_inputs, "solar_tracking")

    wind_density = _positive(_number(land_row, "wind_density_mw_per_km2"))
    if wind_density is None and wind_land_use is not None:
        wind_density = 1.0 / wind_land_use
    fixed_density = _positive(_number(land_row, "solar_density_mw_per_km2"))
    if fixed_density is None and fixed_land_use is not None:
        fixed_density = 1.0 / fixed_land_use

    if fixed_density is not None and fixed_land_use is not None and tracking_land_use is not None:
        tracking_density = fixed_density * fixed_land_use / tracking_land_use
        tracking_density_source = "tech_land_use_ratio"
    else:
        tracking_density = _positive(_number(land_row, "solar_tracking_density_mw_per_km2"))
        if tracking_density is not None:
            tracking_density_source = "solar_tracking_density_mw_per_km2"
        elif tracking_land_use is not None:
            tracking_density = 1.0 / tracking_land_use
            tracking_density_source = "solar_tracking_land_use"
        else:
            tracking_density = fixed_density
            tracking_density_source = "fixed_solar_compatibility"

    # Older tables: reconstruct areas from the MW caps
    if wind_area is None and wind_density is not None:
        wind_cap = _nonnegative(_number(land_row, "max_power_wind_mw"))
        if wind_cap is not None:
            wind_area = wind_cap / wind_density
            wind_area_source = "max_power_wind_mw"
    if solar_area is None and fixed_density is not None:
        solar_cap = _nonnegative(_number(land_row, "max_power_solar_mw"))
        if solar_cap is not None:
            solar_area = solar_cap / fixed_density

    versioned_union = _nonnegative(_number(land_row, CLASSWISE_NESTED_UNION_AREA_COLUMN))
    unversioned_union = _nonnegative(_number(land_row, "renewable_union_area_km2", "onshore_renewable_union_area_km2"))
    if (
        versioned_union is not None
        and unversioned_union is not None
        and not math.isclose(versioned_union, unversioned_union, rel_tol=1e-12, abs_tol=1e-9)
    ):
        raise ValueError(f"renewable_union_area_km2 compatibility alias diverges from {CLASSWISE_NESTED_UNION_AREA_COLUMN}")
    if versioned_union is not None:
        union_area, union_area_source = versioned_union, CLASSWISE_NESTED_UNION_SOURCE
    elif unversioned_union is not None:
        union_area, union_area_source = unversioned_union, "explicit_legacy_unversioned"
    else:
        union_area, union_area_source = None, "missing"

    exclusive_fraction = DEFAULT_WIND_LAND_EXCLUSIVE_FRACTION
    wind_inputs = tech_inputs.get("wind") if isinstance(tech_inputs, Mapping) else None
    if isinstance(wind_inputs, Mapping):
        raw_fraction = _number(wind_inputs, "land_use_exclusive_fraction")
        if raw_fraction is not None:
            if not 0.0 <= raw_fraction <= 1.0:
                raise ValueError("wind.land_use_exclusive_fraction must lie between 0 and 1")
            exclusive_fraction = float(raw_fraction)

    return RenewableLandBudget(
        wind_area_km2=wind_area,
        solar_area_km2=solar_area,
        renewable_union_area_km2=union_area,
        wind_density_mw_per_km2=wind_density,
        fixed_solar_density_mw_per_km2=fixed_density,
        tracking_solar_density_mw_per_km2=_positive(tracking_density),
        union_area_source=union_area_source,
        wind_area_source=wind_area_source,
        tracking_density_source=tracking_density_source,
        wind_land_exclusive_fraction=exclusive_fraction,
    )


# ---------------------------------------------------------------------------------------------
# Land used by a capacity mix
# ---------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class RenewableLandUse:
    wind_km2: float
    fixed_solar_km2: float
    tracking_solar_km2: float

    @property
    def solar_km2(self) -> float:
        return self.fixed_solar_km2 + self.tracking_solar_km2

    @property
    def total_km2(self) -> float:
        return self.wind_km2 + self.solar_km2

    def scaled(self, factor: float) -> "RenewableLandUse":
        return RenewableLandUse(self.wind_km2 * factor, self.fixed_solar_km2 * factor, self.tracking_solar_km2 * factor)


def _area_for_capacity(capacity_mw: float, density: float | None, tech: str) -> float:
    capacity = max(0.0, float(capacity_mw))
    if capacity <= _EPS:
        return 0.0
    usable_density = _positive(density)
    if usable_density is None:
        raise ValueError(f"Missing positive {tech} density for {capacity:g} MW of installed capacity")
    return capacity / usable_density


def renewable_land_use_from_capacities(
    *, wind_mw: float, solar_mw: float, solar_tracking_mw: float, budget: RenewableLandBudget
) -> RenewableLandUse:
    """Land occupied by a wind and PV capacity mix."""
    return RenewableLandUse(
        wind_km2=_area_for_capacity(wind_mw, budget.wind_density_mw_per_km2, "wind"),
        fixed_solar_km2=_area_for_capacity(solar_mw, budget.fixed_solar_density_mw_per_km2, "fixed-solar"),
        tracking_solar_km2=_area_for_capacity(solar_tracking_mw, budget.tracking_solar_density_mw_per_km2, "tracking-solar"),
    )


def renewable_land_use_from_results(results: Mapping[str, Any], budget: RenewableLandBudget) -> RenewableLandUse:
    """Land occupied by the capacities in a result mapping (with or without unit suffixes)."""
    return renewable_land_use_from_capacities(
        wind_mw=max(0.0, _number(results, "wind_mw", "wind") or 0.0),
        solar_mw=max(0.0, _number(results, "solar_mw", "solar") or 0.0),
        solar_tracking_mw=max(0.0, _number(results, "solar_tracking_mw", "solar_tracking") or 0.0),
        budget=budget,
    )


def _bounded_gridless_fraction(results: Mapping[str, Any], annual_t: float) -> float:
    gridless_t = _number(results, "gridless_ammonia_production_t")
    if gridless_t is not None and annual_t > 0:
        return min(1.0, max(0.0, gridless_t / annual_t))
    gridless_fraction = _number(results, "gridless_energy_fraction")
    if gridless_fraction is not None:
        return min(1.0, max(0.0, gridless_fraction))
    return math.nan


def _required_area(value: float | None, label: str) -> float:
    if value is None:
        raise ValueError(f"Missing {label} in renewable land budget")
    return max(0.0, float(value))


# ---------------------------------------------------------------------------------------------
# Capacity rules
# ---------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class ScaledDesignCapacity:
    land_allocation: LandAllocation
    reference_production_t: float
    scale_factor: float
    capacity_t: float
    gridless_capacity_t: float
    limiting_constraint: str
    reference_land_use: RenewableLandUse
    wind_mw_per_t_nh3: float
    fixed_solar_mw_per_t_nh3: float
    tracking_solar_mw_per_t_nh3: float
    union_area_source: str
    wind_land_exclusive_fraction_effective: float = 1.0

    @property
    def capacity_mtpa(self) -> float:
        return self.capacity_t / 1_000_000.0

    @property
    def gridless_capacity_mtpa(self) -> float:
        return self.gridless_capacity_t / 1_000_000.0

    def to_mapping(self) -> dict[str, Any]:
        """The scaled-design capacity columns of a surface result row."""
        versioned = self.union_area_source == CLASSWISE_NESTED_UNION_SOURCE
        production = self.reference_production_t

        def per_tonne(value: float) -> float:
            return value / production if production > 0 else math.nan

        return {
            "capacity_rule": "scaled_reference_design",
            "land_allocation": self.land_allocation,
            "wind_land_exclusive_fraction_effective": self.wind_land_exclusive_fraction_effective,
            "scaled_design_max_onshore_ammonia_capacity_t": self.capacity_t,
            "scaled_design_max_onshore_ammonia_capacity_mtpa": self.capacity_mtpa,
            "scaled_design_max_gridless_onshore_ammonia_capacity_t": self.gridless_capacity_t,
            "scaled_design_max_gridless_onshore_ammonia_capacity_mtpa": self.gridless_capacity_mtpa,
            "scaled_design_renewable_capacity_scale_factor": self.scale_factor,
            "scaled_design_limiting_constraint": self.limiting_constraint,
            "renewable_union_area_source": self.union_area_source,
            "renewable_union_area_is_lower_bound_approximation": versioned,
            "renewable_union_area_method": CLASSWISE_NESTED_UNION_METHOD if versioned else self.union_area_source,
            "renewable_union_area_method_version": CLASSWISE_NESTED_UNION_VERSION if versioned else "",
            "wind_land_km2_per_t_nh3": per_tonne(self.reference_land_use.wind_km2),
            "solar_land_km2_per_t_nh3": per_tonne(self.reference_land_use.solar_km2),
            "renewable_land_km2_per_t_nh3": per_tonne(self.reference_land_use.total_km2),
            "wind_mw_per_t_nh3": self.wind_mw_per_t_nh3,
            "fixed_solar_mw_per_t_nh3": self.fixed_solar_mw_per_t_nh3,
            "solar_tracking_mw_per_t_nh3": self.tracking_solar_mw_per_t_nh3,
            "solar_mw_per_t_nh3": self.fixed_solar_mw_per_t_nh3 + self.tracking_solar_mw_per_t_nh3,
            "preferred_supplier_capacity_column": "scaled_design_max_gridless_onshore_ammonia_capacity_t",
        }


def estimate_scaled_design_capacity(
    results: Mapping[str, Any], budget: RenewableLandBudget, *, land_allocation: LandAllocation = "colocated"
) -> ScaledDesignCapacity:
    """Scale an unconstrained reference design until it fills the land budget.

    The reference mix is kept in proportion. A scale below one stays below one: a 1 Mt/yr
    reference plant that does not fit is never promoted to the supplier threshold.
    """
    land_allocation = normalise_land_allocation(land_allocation)
    fraction = effective_wind_land_exclusive_fraction(land_allocation, budget)
    annual_t = _number(results, "annual_ammonia_production_t")
    if annual_t is None:
        raise ValueError("results must include annual_ammonia_production_t")
    annual_t = max(0.0, annual_t)

    wind_mw = max(0.0, _number(results, "wind_mw", "wind") or 0.0)
    fixed_solar_mw = max(0.0, _number(results, "solar_mw", "solar") or 0.0)
    tracking_solar_mw = max(0.0, _number(results, "solar_tracking_mw", "solar_tracking") or 0.0)
    land_use = renewable_land_use_from_capacities(
        wind_mw=wind_mw, solar_mw=fixed_solar_mw, solar_tracking_mw=tracking_solar_mw, budget=budget
    )

    constraints: list[tuple[str, float]] = []

    def add_constraint(label: str, available_km2: float | None, used_km2: float) -> None:
        if used_km2 > _EPS:
            constraints.append((label, _required_area(available_km2, f"{label} area") / used_km2))

    add_constraint("wind", budget.wind_area_km2, land_use.wind_km2)
    add_constraint("solar", budget.solar_area_km2, land_use.solar_km2)
    add_constraint("renewable_union", budget.renewable_union_area_km2, land_use.solar_km2 + fraction * land_use.wind_km2)

    if constraints:
        limiting_constraint, scale_factor = min(constraints, key=lambda item: item[1])
        scale_factor = max(0.0, float(scale_factor))
    else:
        limiting_constraint, scale_factor = "none", 1.0

    capacity_t = annual_t * scale_factor
    gridless_fraction = _bounded_gridless_fraction(results, annual_t)
    denominator = annual_t if annual_t > 0 else math.nan
    return ScaledDesignCapacity(
        land_allocation=land_allocation,
        wind_land_exclusive_fraction_effective=fraction,
        reference_production_t=annual_t,
        scale_factor=scale_factor,
        capacity_t=capacity_t,
        gridless_capacity_t=capacity_t * gridless_fraction if math.isfinite(gridless_fraction) else math.nan,
        limiting_constraint=limiting_constraint,
        reference_land_use=land_use,
        wind_mw_per_t_nh3=wind_mw / denominator,
        fixed_solar_mw_per_t_nh3=fixed_solar_mw / denominator,
        tracking_solar_mw_per_t_nh3=tracking_solar_mw / denominator,
        union_area_source=budget.union_area_source,
    )


def report_land_feasible_quantity(
    results: Mapping[str, Any], budget: RenewableLandBudget, *, land_allocation: LandAllocation = "colocated"
) -> dict[str, Any]:
    """Report a quantity solved with land in the solve: checked against the budget, never rescaled."""
    land_allocation = normalise_land_allocation(land_allocation)
    production = _number(results, "annual_ammonia_production_t")
    if production is None or production <= 0:
        raise ValueError("Solved quantity requires positive annual production")
    used = renewable_land_use_from_results(results, budget)
    exclusive_fraction = effective_wind_land_exclusive_fraction(land_allocation, budget)
    output: dict[str, Any] = {}
    for name, consumption, raw_area in (
        ("wind", used.wind_km2, budget.wind_area_km2),
        ("solar", used.solar_km2, budget.solar_area_km2),
        ("renewable_union", used.solar_km2 + exclusive_fraction * used.wind_km2, budget.renewable_union_area_km2),
    ):
        area = _required_area(raw_area, name)
        if consumption > area + max(1e-5, area * 1e-6):
            raise ValueError(f"Solved plant exceeds {name} land budget: {consumption} > {area}")
        output[f"{name}_land_slack_km2"] = area - consumption
    fraction = _bounded_gridless_fraction(results, production)
    output.update({
        "capacity_rule": "solved_quantity",
        "land_allocation": land_allocation,
        "wind_land_exclusive_fraction_effective": exclusive_fraction,
        "land_feasible_ammonia_production_t": production,
        "land_feasible_gridless_ammonia_production_t": production * fraction,
        "quantity_is_maximum": False,
        "preferred_supplier_capacity_column": "",
        "renewable_union_area_source": budget.union_area_source,
        "renewable_union_area_is_lower_bound_approximation": budget.union_area_source == CLASSWISE_NESTED_UNION_SOURCE,
    })
    return output


# ---------------------------------------------------------------------------------------------
# Land in the solve
# ---------------------------------------------------------------------------------------------

def _set_component_cap(network, name: str, cap_value: float) -> float:
    """Bound an extendable component's capacity; returns the bound set (0 if none)."""
    if cap_value is None or not np.isfinite(float(cap_value)):
        return 0.0
    cap = max(0.0, float(cap_value))
    for table, extendable, maximum in (
        (network.generators, "p_nom_extendable", "p_nom_max"),
        (network.links, "p_nom_extendable", "p_nom_max"),
        (network.stores, "e_nom_extendable", "e_nom_max"),
    ):
        if name in table.index:
            if not bool(table.at[name, extendable]):
                return 0.0
            table.loc[name, maximum] = cap
            return cap
    return 0.0


def apply_land_budget(network, budget: RenewableLandBudget, land_allocation: LandAllocation) -> float:
    """Enforce a cell's land budget in the solve: store it for add_land_constraints() and
    bound wind and PV by their own areas. Returns the sum of the MW bounds."""
    setattr(network, "_renewable_land_budget", budget)
    setattr(network, "_land_allocation", normalise_land_allocation(land_allocation))
    total_cap = 0.0
    for component, area_km2, density in (
        ("wind", budget.wind_area_km2, budget.wind_density_mw_per_km2),
        ("solar", budget.solar_area_km2, budget.fixed_solar_density_mw_per_km2),
        ("solar_tracking", budget.solar_area_km2, budget.tracking_solar_density_mw_per_km2),
    ):
        if area_km2 is not None and density is not None:
            total_cap += _set_component_cap(network, component, max(0.0, float(area_km2)) * max(0.0, float(density)))
    return total_cap


def add_land_constraints(model, budget: RenewableLandBudget, land_allocation: LandAllocation) -> None:
    """linopy constraints: wind land, PV land and the shared footprint within the budget."""

    def generator_capacity(name):
        try:
            return model["Generator-p_nom"].sel(name=name)
        except KeyError:
            return None

    wind_nom, solar_nom, tracking_nom = (generator_capacity(name) for name in ("wind", "solar", "solar_tracking"))
    wind_land = (
        wind_nom / float(budget.wind_density_mw_per_km2)
        if wind_nom is not None and budget.wind_density_mw_per_km2 else None
    )
    fixed_solar_land = (
        solar_nom / float(budget.fixed_solar_density_mw_per_km2)
        if solar_nom is not None and budget.fixed_solar_density_mw_per_km2 else None
    )
    tracking_solar_land = (
        tracking_nom / float(budget.tracking_solar_density_mw_per_km2)
        if tracking_nom is not None and budget.tracking_solar_density_mw_per_km2 else None
    )
    solar_terms = [term for term in (fixed_solar_land, tracking_solar_land) if term is not None]
    solar_land = sum(solar_terms[1:], solar_terms[0]) if solar_terms else None
    fraction = effective_wind_land_exclusive_fraction(normalise_land_allocation(land_allocation), budget)
    union_terms = [term for term in (
        (wind_land * fraction) if (wind_land is not None and fraction > 0) else None,
        solar_land,
    ) if term is not None]
    union_land = sum(union_terms[1:], union_terms[0]) if union_terms else None

    if wind_land is not None and budget.wind_area_km2 is not None:
        model.add_constraints(wind_land <= float(budget.wind_area_km2), name="wind_land_area_cap")
    if solar_land is not None and budget.solar_area_km2 is not None:
        model.add_constraints(solar_land <= float(budget.solar_area_km2), name="solar_land_area_cap")
    if union_land is not None and budget.renewable_union_area_km2 is not None:
        model.add_constraints(union_land <= float(budget.renewable_union_area_km2), name="renewable_union_land_area_cap")


# ---------------------------------------------------------------------------------------------
# Land tables
# ---------------------------------------------------------------------------------------------

def _validate_union_provenance(frame: pd.DataFrame, path: Path) -> None:
    """Require the versioned v1 union lower bound, with a consistent generic alias."""
    required = {CLASSWISE_NESTED_UNION_AREA_COLUMN, RENEWABLE_UNION_METHOD_COLUMN, RENEWABLE_UNION_METHOD_VERSION_COLUMN}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Land table {path} lacks versioned renewable-union provenance: " + ", ".join(missing))
    methods = set(frame[RENEWABLE_UNION_METHOD_COLUMN].dropna().astype(str))
    versions = set(frame[RENEWABLE_UNION_METHOD_VERSION_COLUMN].dropna().astype(str))
    if methods != {CLASSWISE_NESTED_UNION_METHOD}:
        raise ValueError(f"Land table {path} has unexpected renewable-union methods: {sorted(methods)}")
    if versions != {CLASSWISE_NESTED_UNION_VERSION}:
        raise ValueError(f"Land table {path} has unexpected renewable-union versions: {sorted(versions)}")
    versioned = pd.to_numeric(frame[CLASSWISE_NESTED_UNION_AREA_COLUMN], errors="coerce")
    if versioned.isna().any() or (versioned < 0.0).any():
        raise ValueError(f"Land table {path} contains invalid values in {CLASSWISE_NESTED_UNION_AREA_COLUMN}")
    if "renewable_union_area_km2" in frame.columns:
        generic = pd.to_numeric(frame["renewable_union_area_km2"], errors="coerce")
        if generic.isna().any() or not np.allclose(versioned.to_numpy(), generic.to_numpy(), rtol=0.0, atol=1e-9):
            raise ValueError(f"Land table {path} has a generic renewable-union alias that diverges from the versioned v1 column")


def _fixed_and_tracking_land_use(tech_inputs: Mapping[str, dict]) -> tuple[float | None, float | None]:
    def land_use(name: str) -> float | None:
        raw = tech_inputs.get(name)
        if not isinstance(raw, dict) or raw.get("land_use_km2_per_mw") is None:
            return None
        value = float(raw["land_use_km2_per_mw"])
        return value if value > 0 else None

    return land_use("solar"), land_use("solar_tracking")


def _apply_tech_solar_density(frame: pd.DataFrame, tech_inputs: Mapping[str, dict]) -> pd.DataFrame:
    """Set the PV densities the run uses from the technology YAML.

    Tables with an explicit density method keep their own fixed-PV density. Otherwise the
    fixed-PV density is the latitude packing of land_table.solar_density, normalised to the
    YAML's equator land use; tracking PV is that density times the fixed/tracking land-use
    ratio. The PV power caps follow.
    """
    from .land_table import solar_density

    if frame.empty or "latitude" not in frame.columns:
        return frame
    updated = frame.copy()
    fixed_land_use, tracking_land_use = _fixed_and_tracking_land_use(tech_inputs)
    if "solar_density_method" in updated.columns:
        methods = set(updated["solar_density_method"].fillna("").astype(str))
        if methods != {"explicit_fixed_pv_density_v1"}:
            raise ValueError(f"Unknown or mixed explicit solar density methods: {methods}")
        if "solar_density_mw_per_km2" not in updated.columns:
            raise ValueError("Explicit solar density method requires solar_density_mw_per_km2")
        density = pd.to_numeric(updated["solar_density_mw_per_km2"], errors="coerce")
        if not np.isfinite(density).all() or (density <= 0).any():
            raise ValueError("Explicit fixed-PV densities must be finite and positive")
        updated["solar_density_mw_per_km2"] = density
    else:
        base_land_use = fixed_land_use if fixed_land_use is not None else tracking_land_use
        if base_land_use is None:
            return frame
        updated["solar_density_mw_per_km2"] = solar_density(
            updated["latitude"], scale=1.0, base_land_use_km2_per_mw=base_land_use
        )
    if fixed_land_use is not None and tracking_land_use is not None:
        updated["solar_tracking_density_mw_per_km2"] = (
            updated["solar_density_mw_per_km2"] * fixed_land_use / tracking_land_use
        )
    if "solar_area_km2" in updated.columns:
        updated["max_power_solar_mw"] = (updated["solar_area_km2"] * updated["solar_density_mw_per_km2"]).clip(lower=0.0)
        if "max_power_wind_mw" in updated.columns:
            updated["max_capacity_mw"] = updated["max_power_wind_mw"] + updated["max_power_solar_mw"]
    return updated


def load_land_table(path: str | Path, tech_inputs: Mapping[str, dict]) -> pd.DataFrame:
    """A centred land table with lower-case columns, checked, with the run's PV densities."""
    resolved = repo_path(path)
    if not resolved.exists():
        raise FileNotFoundError(f"Land table not found: {resolved}")
    frame = pd.read_csv(resolved)
    frame.columns = [column.lower() for column in frame.columns]
    missing = {"latitude", "longitude"} - set(frame.columns)
    if missing:
        raise ValueError(f"Land CSV {resolved} is missing {sorted(missing)}")
    anchors = set(frame["cell_anchor"].dropna().astype(str)) if "cell_anchor" in frame.columns else set()
    if anchors != {REQUIRED_CELL_ANCHOR}:
        raise ValueError(
            f"Land CSV {resolved} must be a centred build (cell_anchor={REQUIRED_CELL_ANCHOR}), "
            f"found {sorted(anchors) if anchors else 'no cell_anchor column'}; tables built "
            "before 23 Sep 2026 are offset by one degree"
        )
    _validate_union_provenance(frame, resolved)
    return _apply_tech_solar_density(frame, tech_inputs)


def _cell_key(lat: float, lon: float) -> tuple[float, float]:
    return round(float(lat), 4), round(float(lon), 4)


def index_land_table(frame: pd.DataFrame | None) -> dict[tuple[float, float], pd.Series]:
    """Index a land table by cell: {(lat, lon): row}."""
    if frame is None:
        return {}
    return {_cell_key(row["latitude"], row["longitude"]): row for _, row in frame.iterrows()}


def cell_land_row(index: dict, lat: float, lon: float) -> pd.Series | None:
    return index.get(_cell_key(lat, lon))


_LAND_METADATA_COLUMNS = (
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
    CLASSWISE_NESTED_UNION_AREA_COLUMN,
)


def land_metadata(row: pd.Series | None) -> dict[str, float]:
    """The land-table values copied into a cell's result row."""
    if row is None:
        return {}
    metadata: dict[str, float] = {}

    def value(name: str) -> float | None:
        raw = row.get(name)
        return None if pd.isna(raw) else float(raw)

    onshore = value("onshore_land_pct")
    if onshore is not None:
        metadata["land_onshore_pct"] = onshore
        metadata["offshore_sea_pct"] = max(0.0, 100.0 - onshore)
    for source, target in (
        ("area", "land_cell_area_km2"),
        ("onshore_area_km2", "onshore_area_km2"),
        ("offshore_area_km2", "offshore_area_km2"),
        ("elevation_m", "elevation_m"),
    ):
        if value(source) is not None:
            metadata[target] = value(source)
    max_solar, max_wind = value("max_power_solar_mw"), value("max_power_wind_mw")
    if max_solar is not None or max_wind is not None:
        metadata["area_cap_mw"] = (max_solar or 0.0) + (max_wind or 0.0)
    if value("solar_area_km2") is not None:
        metadata["solar_area_used_km2"] = metadata["solar_available_area_km2"] = value("solar_area_km2")
    if value("wind_area_km2") is not None:
        metadata["wind_area_used_km2"] = metadata["wind_available_area_km2"] = value("wind_area_km2")
    for column in _LAND_METADATA_COLUMNS:
        if value(column) is not None:
            metadata[column] = value(column)
    for column in row.index:
        key = str(column).strip().lower()
        if (key.startswith("max_power_") and key.endswith("_mw")) or (key.startswith("max_energy_") and key.endswith("_mwh")):
            if value(key) is not None:
                metadata[key] = value(key)
    return metadata
