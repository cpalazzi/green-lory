"""Pure renewable-land accounting and ammonia-capacity scaling helpers.

This module deliberately contains no PyPSA or solver imports.  It separates the
land budget from the plant-design solve so that a cost-optimal reference plant
can be scaled after the fact without first forcing that plant under the same
land constraint.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Literal, Mapping

from .land_union import (
    CLASSWISE_NESTED_UNION_AREA_COLUMN,
    CLASSWISE_NESTED_UNION_METHOD,
    CLASSWISE_NESTED_UNION_SOURCE,
    CLASSWISE_NESTED_UNION_VERSION,
)


LandAllocation = Literal["colocated", "exclusive"]
LAND_ALLOCATIONS: tuple[LandAllocation, ...] = (
    "colocated",
    "exclusive",
)

# Wind and PV may share land.  A wind farm's footprint (about 200 km2/GW of
# turbine spacing) is almost entirely open ground; only the turbine pads, roads
# and substations exclude other uses.  Denholm et al. (NREL/TP-6A2-45834, 2009,
# Table 1) report a direct-impact area of 1.0 +/- 0.7 ha/MW (0.3 permanent +
# 0.7 temporary) against a total area of 34.5 +/- 22.4 ha/MW, i.e. about 3 % of
# the footprint.  ``colocated`` therefore counts only that fraction of the wind
# footprint against the shared budget that PV also draws on; ``exclusive``
# counts the whole footprint (the September 2026 campaign rule).  The fraction
# can be overridden per technology YAML (``wind.land_use_exclusive_fraction``).
# TODO: replace the direct-impact fraction by a co-location study value that also
# covers PV row losses from turbine shading, wake spacing and access setbacks.
DEFAULT_WIND_LAND_EXCLUSIVE_FRACTION = 0.03

def normalise_land_allocation(name: str) -> LandAllocation:
    """Validate the allocation name (``colocated`` or ``exclusive``)."""

    if name in LAND_ALLOCATIONS:
        return name  # type: ignore[return-value]
    raise ValueError(
        f"Unknown land_allocation {name!r}; choose from {LAND_ALLOCATIONS}"
    )


def effective_wind_land_exclusive_fraction(
    land_allocation: LandAllocation, budget: "RenewableLandBudget"
) -> float:
    """Fraction of the wind footprint that counts against the shared PV/wind budget."""

    if land_allocation == "exclusive":
        return 1.0
    return float(budget.wind_land_exclusive_fraction)

_EPS = 1e-9

UnionAreaSource = Literal[
    "explicit_classwise_nested_v1_lower_bound",
    "explicit_legacy_unversioned",
    "explicit",
    "conservative_max",
    "missing",
]


def _number(mapping: Any, *names: str) -> float | None:
    """Return the first finite numeric value available under ``names``."""

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
    if value is None:
        return None
    return max(0.0, float(value))


def _positive(value: float | None) -> float | None:
    if value is None or value <= 0.0:
        return None
    return float(value)


def _tech_land_use(tech_inputs: Mapping[str, Any], tech: str) -> float | None:
    raw = tech_inputs.get(tech)
    if not isinstance(raw, Mapping):
        return None
    return _positive(_number(raw, "land_use_km2_per_mw"))


@dataclass(frozen=True)
class RenewableLandBudget:
    """Available renewable areas and technology-specific power densities.

    ``renewable_union_area_km2`` is an estimate of the total physical area
    suitable for at least one onshore renewable technology.  New land tables
    provide the versioned classwise-nested v1 lower bound explicitly.  It uses
    ``max(wind_factor, solar_factor)`` within each MODIS class because the
    pixel-level overlap is unknown; it must not be described as an exact union.
    Older unversioned tables and the still more conservative aggregate
    ``max(wind_area, solar_area)`` fallback receive distinct source labels.
    """

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
    def uses_conservative_union_fallback(self) -> bool:
        return self.union_area_source == "conservative_max"

    @property
    def tracking_land_multiplier(self) -> float:
        """Tracking land per MW relative to fixed PV land per MW."""

        fixed_density = _positive(self.fixed_solar_density_mw_per_km2)
        tracking_density = _positive(self.tracking_solar_density_mw_per_km2)
        if fixed_density is None or tracking_density is None:
            return math.nan
        return fixed_density / tracking_density

    @classmethod
    def from_land_row(
        cls,
        land_row: Any,
        tech_inputs: Mapping[str, Any],
        *,
        onshore: bool = True,
        allow_conservative_union_fallback: bool = True,
    ) -> "RenewableLandBudget":
        return build_renewable_land_budget(
            land_row,
            tech_inputs,
            onshore=onshore,
            allow_conservative_union_fallback=allow_conservative_union_fallback,
        )


def build_renewable_land_budget(
    land_row: Any,
    tech_inputs: Mapping[str, Any],
    *,
    onshore: bool = True,
    allow_conservative_union_fallback: bool = True,
) -> RenewableLandBudget:
    """Construct a land budget from one land-table row and technology inputs.

    The land table carries the latitude-adjusted fixed-PV density.  Tracking PV
    receives a distinct density by applying the fixed/tracking land-use ratio
    from the technology YAML.  With the canonical inputs (0.01 versus
    0.02 km2/MW), tracking density is half fixed-PV density.
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

    # Prefer the YAML ratio applied to the row's latitude-adjusted fixed-PV
    # density.  This preserves the spatial packing adjustment while correcting
    # tracking's larger footprint.
    if (
        fixed_density is not None
        and fixed_land_use is not None
        and tracking_land_use is not None
    ):
        tracking_density = fixed_density * fixed_land_use / tracking_land_use
        tracking_density_source = "tech_land_use_ratio"
    else:
        tracking_density = _positive(
            _number(land_row, "solar_tracking_density_mw_per_km2")
        )
        if tracking_density is not None:
            tracking_density_source = "solar_tracking_density_mw_per_km2"
        elif tracking_land_use is not None:
            tracking_density = 1.0 / tracking_land_use
            tracking_density_source = "solar_tracking_land_use"
        else:
            # Compatibility for configurations that do not define tracking as a
            # distinct technology.  Canonical Green Lory YAMLs do define it.
            tracking_density = fixed_density
            tracking_density_source = "fixed_solar_compatibility"

    # Older tables can reconstruct areas from MW caps where necessary.
    if wind_area is None and wind_density is not None:
        wind_cap = _nonnegative(_number(land_row, "max_power_wind_mw"))
        if wind_cap is not None:
            wind_area = wind_cap / wind_density
            wind_area_source = "max_power_wind_mw"
    if solar_area is None and fixed_density is not None:
        solar_cap = _nonnegative(_number(land_row, "max_power_solar_mw"))
        if solar_cap is not None:
            solar_area = solar_cap / fixed_density

    versioned_union_area = _nonnegative(
        _number(land_row, CLASSWISE_NESTED_UNION_AREA_COLUMN)
    )
    unversioned_union_area = _nonnegative(
        _number(
            land_row,
            "renewable_union_area_km2",
            "onshore_renewable_union_area_km2",
        )
    )
    if (
        versioned_union_area is not None
        and unversioned_union_area is not None
        and not math.isclose(
            versioned_union_area,
            unversioned_union_area,
            rel_tol=1e-12,
            abs_tol=1e-9,
        )
    ):
        raise ValueError(
            "renewable_union_area_km2 compatibility alias diverges from "
            f"{CLASSWISE_NESTED_UNION_AREA_COLUMN}"
        )

    if versioned_union_area is not None:
        union_area = versioned_union_area
        union_area_source: UnionAreaSource = CLASSWISE_NESTED_UNION_SOURCE
    elif unversioned_union_area is not None:
        union_area = unversioned_union_area
        union_area_source = "explicit_legacy_unversioned"
    elif (
        allow_conservative_union_fallback
        and wind_area is not None
        and solar_area is not None
    ):
        # Any physical union is at least the larger individual suitable area and
        # at most their sum.  max(...) is therefore conservative, never expansive.
        union_area = max(wind_area, solar_area)
        union_area_source = "conservative_max"
    else:
        union_area = None
        union_area_source = "missing"

    wind_inputs = tech_inputs.get("wind") if isinstance(tech_inputs, Mapping) else None
    exclusive_fraction = DEFAULT_WIND_LAND_EXCLUSIVE_FRACTION
    if isinstance(wind_inputs, Mapping):
        raw_fraction = _number(wind_inputs, "land_use_exclusive_fraction")
        if raw_fraction is not None:
            if not 0.0 <= raw_fraction <= 1.0:
                raise ValueError(
                    "wind.land_use_exclusive_fraction must lie between 0 and 1"
                )
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
        return RenewableLandUse(
            wind_km2=self.wind_km2 * factor,
            fixed_solar_km2=self.fixed_solar_km2 * factor,
            tracking_solar_km2=self.tracking_solar_km2 * factor,
        )


def _area_for_capacity(capacity_mw: float, density: float | None, tech: str) -> float:
    capacity = max(0.0, float(capacity_mw))
    if capacity <= _EPS:
        return 0.0
    usable_density = _positive(density)
    if usable_density is None:
        raise ValueError(f"Missing positive {tech} density for {capacity:g} MW of installed capacity")
    return capacity / usable_density


def renewable_land_use_from_capacities(
    *,
    wind_mw: float,
    solar_mw: float,
    solar_tracking_mw: float,
    budget: RenewableLandBudget,
) -> RenewableLandUse:
    """Calculate physical land occupied by a renewable capacity mix."""

    return RenewableLandUse(
        wind_km2=_area_for_capacity(
            wind_mw, budget.wind_density_mw_per_km2, "wind"
        ),
        fixed_solar_km2=_area_for_capacity(
            solar_mw, budget.fixed_solar_density_mw_per_km2, "fixed-solar"
        ),
        tracking_solar_km2=_area_for_capacity(
            solar_tracking_mw,
            budget.tracking_solar_density_mw_per_km2,
            "tracking-solar",
        ),
    )


def renewable_land_use_from_results(
    results: Mapping[str, Any],
    budget: RenewableLandBudget,
) -> RenewableLandUse:
    """Calculate land occupied by renewable capacities in a result mapping."""

    return renewable_land_use_from_capacities(
        wind_mw=max(0.0, _number(results, "wind_mw", "wind") or 0.0),
        solar_mw=max(0.0, _number(results, "solar_mw", "solar") or 0.0),
        solar_tracking_mw=max(
            0.0,
            _number(results, "solar_tracking_mw", "solar_tracking") or 0.0,
        ),
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
        """Return only versioned, onshore-aware capacity outputs.

        Historical ``max_ammonia_capacity_*`` columns have a different contract:
        they use total wind (including offshore) while their ``max_onshore_*``
        companions use onshore wind.  One onshore calculation cannot populate
        both honestly, so those compatibility fields are produced separately by
        :mod:`model.run_global`'s frozen historical estimator.
        """

        output: dict[str, Any] = {
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
            "renewable_union_area_is_conservative_fallback": (
                self.union_area_source == "conservative_max"
            ),
            "renewable_union_area_is_lower_bound_approximation": (
                self.union_area_source == CLASSWISE_NESTED_UNION_SOURCE
            ),
            "renewable_union_area_method": (
                CLASSWISE_NESTED_UNION_METHOD
                if self.union_area_source == CLASSWISE_NESTED_UNION_SOURCE
                else self.union_area_source
            ),
            "renewable_union_area_method_version": (
                CLASSWISE_NESTED_UNION_VERSION
                if self.union_area_source == CLASSWISE_NESTED_UNION_SOURCE
                else ""
            ),
            "wind_land_km2_per_t_nh3": (
                self.reference_land_use.wind_km2 / self.reference_production_t
                if self.reference_production_t > 0
                else math.nan
            ),
            "solar_land_km2_per_t_nh3": (
                self.reference_land_use.solar_km2 / self.reference_production_t
                if self.reference_production_t > 0
                else math.nan
            ),
            "renewable_land_km2_per_t_nh3": (
                self.reference_land_use.total_km2 / self.reference_production_t
                if self.reference_production_t > 0
                else math.nan
            ),
            "wind_mw_per_t_nh3": self.wind_mw_per_t_nh3,
            "fixed_solar_mw_per_t_nh3": self.fixed_solar_mw_per_t_nh3,
            "solar_tracking_mw_per_t_nh3": self.tracking_solar_mw_per_t_nh3,
            "solar_mw_per_t_nh3": (
                self.fixed_solar_mw_per_t_nh3
                + self.tracking_solar_mw_per_t_nh3
            ),
        }

        return output


def _required_area(value: float | None, label: str) -> float:
    if value is None:
        raise ValueError(f"Missing {label} in renewable land budget")
    return max(0.0, float(value))


def report_land_feasible_quantity(
    results: Mapping[str, Any],
    budget: RenewableLandBudget,
    *,
    land_allocation: LandAllocation = "colocated",
) -> dict[str, Any]:
    """Report a solved quantity, never rescaling it or labelling it a maximum."""
    land_allocation = normalise_land_allocation(land_allocation)
    production = _number(results, "annual_ammonia_production_t")
    if production is None or production <= 0:
        raise ValueError("Solved quantity requires positive annual production")
    used = renewable_land_use_from_results(results, budget)
    exclusive_fraction = effective_wind_land_exclusive_fraction(land_allocation, budget)
    constraints = [
        ("wind", used.wind_km2, budget.wind_area_km2),
        ("solar", used.solar_km2, budget.solar_area_km2),
        ("renewable_union", used.solar_km2 + exclusive_fraction * used.wind_km2, budget.renewable_union_area_km2),
    ]
    output: dict[str, Any] = {}
    for name, consumption, raw_area in constraints:
        area = _required_area(raw_area, name)
        tolerance = max(1e-5, area * 1e-6)
        if consumption > area + tolerance:
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
        "renewable_union_area_is_lower_bound_approximation": (
            budget.union_area_source == CLASSWISE_NESTED_UNION_SOURCE),
    })
    return output


def estimate_scaled_design_capacity(
    results: Mapping[str, Any],
    budget: RenewableLandBudget,
    *,
    land_allocation: LandAllocation = "colocated",
) -> ScaledDesignCapacity:
    """Scale an unconstrained reference design against a renewable land budget.

    The reference mix is preserved proportionally.  A scale below one is kept
    below one; this function never promotes a land-infeasible 1 Mt reference
    plant to the downstream supplier threshold.
    """

    land_allocation = normalise_land_allocation(land_allocation)
    fraction = effective_wind_land_exclusive_fraction(land_allocation, budget)

    annual_t = _number(results, "annual_ammonia_production_t")
    if annual_t is None:
        raise ValueError("results must include annual_ammonia_production_t")
    annual_t = max(0.0, annual_t)

    wind_mw = max(0.0, _number(results, "wind_mw", "wind") or 0.0)
    fixed_solar_mw = max(0.0, _number(results, "solar_mw", "solar") or 0.0)
    tracking_solar_mw = max(
        0.0,
        _number(results, "solar_tracking_mw", "solar_tracking") or 0.0,
    )
    land_use = renewable_land_use_from_capacities(
        wind_mw=wind_mw,
        solar_mw=fixed_solar_mw,
        solar_tracking_mw=tracking_solar_mw,
        budget=budget,
    )

    constraints: list[tuple[str, float]] = []

    def add_constraint(label: str, available_km2: float | None, used_km2: float) -> None:
        if used_km2 <= _EPS:
            return
        available = _required_area(available_km2, f"{label} area")
        constraints.append((label, available / used_km2))

    add_constraint("wind", budget.wind_area_km2, land_use.wind_km2)
    add_constraint("solar", budget.solar_area_km2, land_use.solar_km2)
    add_constraint(
        "renewable_union",
        budget.renewable_union_area_km2,
        land_use.solar_km2 + fraction * land_use.wind_km2,
    )

    if constraints:
        limiting_constraint, scale_factor = min(constraints, key=lambda item: item[1])
        scale_factor = max(0.0, float(scale_factor))
    else:
        limiting_constraint, scale_factor = "none", 1.0

    capacity_t = annual_t * scale_factor
    gridless_fraction = _bounded_gridless_fraction(results, annual_t)
    gridless_capacity_t = (
        capacity_t * gridless_fraction
        if math.isfinite(gridless_fraction)
        else math.nan
    )

    denominator = annual_t if annual_t > 0 else math.nan
    return ScaledDesignCapacity(
        land_allocation=land_allocation,
        wind_land_exclusive_fraction_effective=fraction,
        reference_production_t=annual_t,
        scale_factor=scale_factor,
        capacity_t=capacity_t,
        gridless_capacity_t=gridless_capacity_t,
        limiting_constraint=limiting_constraint,
        reference_land_use=land_use,
        wind_mw_per_t_nh3=wind_mw / denominator,
        fixed_solar_mw_per_t_nh3=fixed_solar_mw / denominator,
        tracking_solar_mw_per_t_nh3=tracking_solar_mw / denominator,
        union_area_source=budget.union_area_source,
    )
