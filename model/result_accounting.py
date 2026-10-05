"""Pure helpers for final result accounting and grid-backstop reporting.

The optimisation objective contains plant costs in the model currency.  Water
and land inputs may use a different source currency and are added after the
solve.  This module keeps that boundary explicit: callers must supply the FX
conversion and the returned headline identity is always

    headline cost = plant objective + included water cost + included land cost.

Grid-backstop dispatch is classified separately.  In particular, this module
never infers counterfactual grid-free production by multiplying ammonia output
by a renewable-energy fraction.  A solve is either grid-free within a stated
tolerance, or its grid-free production and LCOA are unknown (NaN).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping


def _currency_code(value: str, field_name: str) -> str:
    code = str(value).strip().upper()
    if len(code) != 3 or not code.isalpha():
        raise ValueError(f"{field_name} must be a three-letter currency code")
    return code


def _finite_nonnegative(value: float, field_name: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be finite and non-negative") from exc
    if not math.isfinite(number) or number < 0:
        raise ValueError(f"{field_name} must be finite and non-negative")
    return number


def _finite_positive(value: float, field_name: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be finite and positive") from exc
    if not math.isfinite(number) or number <= 0:
        raise ValueError(f"{field_name} must be finite and positive")
    return number


def finalize_site_cost_accounting(
    results: Mapping[str, Any],
    *,
    output_currency: str,
    source_currency: str,
    source_to_output_fx: float,
    water_cost_source_per_m3: float | None = None,
    water_usage_m3_per_t_nh3: float | None = None,
    land_cost_source_per_km2_year: float | None = None,
    land_used_km2: float | None = None,
    include_site_costs_in_headline: bool = True,
) -> dict[str, Any]:
    """Return results with reconciled plant, water, land, and headline costs.

    ``source_to_output_fx`` is the number of units of output currency per one
    unit of source currency.  It is mandatory even when both currencies match;
    use ``1.0`` in that case.  This prevents implicit USD/EUR relabelling.

    ``include_site_costs_in_headline=False`` preserves a historical plant-gate
    comparison while still reporting the converted site-cost diagnostics.

    The input mapping is not mutated.  Calling the function more than once is
    safe: once present, ``plant_objective_cost_*`` remains the base cost rather
    than the already augmented headline total.
    """

    output_code = _currency_code(output_currency, "output_currency")
    source_code = _currency_code(source_currency, "source_currency")
    output_slug = output_code.lower()
    source_slug = source_code.lower()
    fx = _finite_positive(source_to_output_fx, "source_to_output_fx")
    if source_code == output_code and not math.isclose(fx, 1.0, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError(
            "source_to_output_fx must be 1.0 when source and output currencies match"
        )

    output = dict(results)
    production = _finite_positive(
        output.get("annual_ammonia_production_t"),
        "annual_ammonia_production_t",
    )

    headline_total_key = f"total_cost_{output_slug}_per_year"
    plant_total_key = f"plant_objective_cost_{output_slug}_per_year"
    plant_total_raw = output.get(plant_total_key, output.get(headline_total_key))
    if plant_total_raw is None:
        raise ValueError(
            f"results must include '{headline_total_key}' or '{plant_total_key}'"
        )
    plant_total = _finite_nonnegative(plant_total_raw, plant_total_key)

    if water_cost_source_per_m3 is None:
        if water_usage_m3_per_t_nh3 is not None:
            raise ValueError(
                "water_usage_m3_per_t_nh3 requires water_cost_source_per_m3"
            )
        water_cost_per_m3 = 0.0
        water_usage_per_t = 0.0
    else:
        if water_usage_m3_per_t_nh3 is None:
            raise ValueError(
                "water_cost_source_per_m3 requires water_usage_m3_per_t_nh3"
            )
        water_cost_per_m3 = _finite_nonnegative(
            water_cost_source_per_m3,
            "water_cost_source_per_m3",
        )
        water_usage_per_t = _finite_nonnegative(
            water_usage_m3_per_t_nh3,
            "water_usage_m3_per_t_nh3",
        )

    if land_cost_source_per_km2_year is None:
        if land_used_km2 is not None:
            raise ValueError(
                "land_used_km2 requires land_cost_source_per_km2_year"
            )
        land_cost_per_km2_year = 0.0
        land_area = 0.0
    else:
        if land_used_km2 is None:
            raise ValueError(
                "land_cost_source_per_km2_year requires land_used_km2"
            )
        land_cost_per_km2_year = _finite_nonnegative(
            land_cost_source_per_km2_year,
            "land_cost_source_per_km2_year",
        )
        land_area = _finite_nonnegative(land_used_km2, "land_used_km2")

    water_source_per_t = water_cost_per_m3 * water_usage_per_t
    water_source_per_year = water_source_per_t * production
    land_source_per_year = land_cost_per_km2_year * land_area
    land_source_per_t = land_source_per_year / production

    water_output_per_year = water_source_per_year * fx
    land_output_per_year = land_source_per_year * fx
    water_output_per_t = water_output_per_year / production
    land_output_per_t = land_output_per_year / production
    site_output_per_year = water_output_per_year + land_output_per_year
    component_total = (
        plant_total + site_output_per_year
        if include_site_costs_in_headline
        else plant_total
    )
    headline_total = component_total

    output["currency"] = output_code
    output["site_cost_source_currency"] = source_code
    output["site_cost_output_currency"] = output_code
    output["site_cost_source_to_output_fx"] = fx
    output["site_costs_in_headline"] = bool(include_site_costs_in_headline)

    output[plant_total_key] = plant_total
    output[f"lcoa_plant_{output_slug}_per_t"] = plant_total / production

    output[f"water_cost_{source_slug}_per_m3"] = water_cost_per_m3
    output["water_usage_m3_per_t_nh3"] = water_usage_per_t
    output[f"water_cost_{source_slug}_per_t"] = water_source_per_t
    output[f"water_cost_{source_slug}_per_year"] = water_source_per_year
    output[f"water_cost_{output_slug}_per_t"] = water_output_per_t
    output[f"water_cost_{output_slug}_per_year"] = water_output_per_year

    output[f"land_cost_{source_slug}_per_km2_year"] = land_cost_per_km2_year
    output["land_used_km2"] = land_area
    output[f"land_cost_{source_slug}_per_t"] = land_source_per_t
    output[f"land_cost_{source_slug}_per_year"] = land_source_per_year
    output[f"land_cost_{output_slug}_per_t"] = land_output_per_t
    output[f"land_cost_{output_slug}_per_year"] = land_output_per_year

    output[f"site_cost_{output_slug}_per_year"] = site_output_per_year
    output[headline_total_key] = headline_total
    output[f"lcoa_{output_slug}_per_t"] = headline_total / production

    if headline_total > 0:
        output["plant_cost_pct"] = plant_total / headline_total * 100.0
        output["water_cost_pct"] = (
            water_output_per_year / headline_total * 100.0
            if include_site_costs_in_headline
            else 0.0
        )
        output["land_cost_pct"] = (
            land_output_per_year / headline_total * 100.0
            if include_site_costs_in_headline
            else 0.0
        )
    else:
        output["plant_cost_pct"] = 0.0
        output["water_cost_pct"] = 0.0
        output["land_cost_pct"] = 0.0

    # This is useful in output validation and should be exactly zero apart from
    # floating-point roundoff introduced by external serialization.
    output[f"headline_cost_identity_residual_{output_slug}_per_year"] = (
        headline_total - component_total
    )
    return output


HEADLINE_PLANT_SPLIT_KEYS = (
    "build_cost_pct",
    "tech_cost_pct",
    "om_cost_pct",
    "interest_pct",
)


def finalize_headline_cost_percentages(
    results: Mapping[str, Any],
    *,
    plant_split_percentages: Mapping[str, float],
    tolerance_pct: float = 1e-8,
) -> dict[str, Any]:
    """Put plant subcomponents and included site costs on one denominator.

    ``plant_split_percentages`` contains shares of the *plant objective*.  The
    returned ``build_cost_pct``/``tech_cost_pct``/``om_cost_pct``/
    ``interest_pct`` fields are instead shares of the final headline cost.
    ``other_cost_pct`` is calculated only after those known plant components
    and the included water and land costs have been allocated.  It is therefore
    a genuine residual, rather than the unrelated component-tracking residual
    produced by :mod:`model.auxiliary`.

    ``plant_cost_pct`` remains a subtotal for diagnostics and must not be added
    to its four subcomponents.  The seven mutually exclusive headline fields
    are the four plant splits, ``other_cost_pct``, ``water_cost_pct``, and
    ``land_cost_pct``.
    """

    output = dict(results)
    tolerance = _finite_nonnegative(tolerance_pct, "tolerance_pct")
    plant_headline_pct = _finite_nonnegative(
        output.get("plant_cost_pct"),
        "plant_cost_pct",
    )
    water_pct = _finite_nonnegative(output.get("water_cost_pct", 0.0), "water_cost_pct")
    land_pct = _finite_nonnegative(output.get("land_cost_pct", 0.0), "land_cost_pct")

    subtotal = math.fsum((plant_headline_pct, water_pct, land_pct))
    if not math.isclose(subtotal, 100.0, rel_tol=0.0, abs_tol=tolerance):
        raise ValueError(
            "plant_cost_pct plus included water_cost_pct and land_cost_pct "
            f"must equal 100%; got {subtotal:.12g}%"
        )

    plant_relative: dict[str, float] = {}
    for key in HEADLINE_PLANT_SPLIT_KEYS:
        plant_relative[key] = _finite_nonnegative(
            plant_split_percentages.get(key, 0.0),
            f"plant_split_percentages['{key}']",
        )

    known_plant_pct = math.fsum(plant_relative.values())
    if known_plant_pct > 100.0 + tolerance:
        raise ValueError(
            "known plant cost splits exceed the plant objective: "
            f"{known_plant_pct:.12g}%"
        )

    plant_to_headline = plant_headline_pct / 100.0
    scaled_known = {
        key: value * plant_to_headline
        for key, value in plant_relative.items()
    }
    for key, value in scaled_known.items():
        output[key] = value

    # Calculate the residual last and from the headline total.  This absorbs
    # only floating-point roundoff plus plant-objective costs not represented by
    # the four explicit splits (for example marginal dispatch costs).
    allocated_without_other = math.fsum(
        (*scaled_known.values(), water_pct, land_pct)
    )
    other_pct = 100.0 - allocated_without_other
    if other_pct < -tolerance:
        raise ValueError(
            "known headline cost shares exceed 100%; "
            f"residual other cost would be {other_pct:.12g}%"
        )
    if other_pct < 0.0:
        other_pct = 0.0
    output["other_cost_pct"] = other_pct

    headline_total = math.fsum(
        (*scaled_known.values(), other_pct, water_pct, land_pct)
    )
    output["headline_cost_share_total_pct"] = headline_total
    output["headline_cost_share_residual_pct"] = 100.0 - headline_total
    output["cost_percentage_basis"] = "headline_total"
    return output


@dataclass(frozen=True)
class GridUseClassification:
    """Classification of numerical versus material grid-backstop dispatch."""

    raw_grid_energy_mwh: float
    grid_energy_mwh: float
    electrical_reference_energy_mwh: float
    grid_energy_share: float
    tolerance_mwh: float
    uses_grid_backstop: bool


def classify_grid_use(
    grid_energy_mwh: float,
    electrical_reference_energy_mwh: float,
    *,
    absolute_tolerance_mwh: float = 1.0,
    relative_tolerance: float = 1e-6,
) -> GridUseClassification:
    """Classify grid use using the greater of absolute and relative tolerance.

    ``electrical_reference_energy_mwh`` is total solved generator supply into
    the power bus, including the grid backstop.  Using a chemical-energy demand
    or ammonia-production denominator would change both the reported share and
    the relative tolerance and is therefore rejected by this API's semantics.

    Tiny negative dispatch caused by solver tolerances is retained in
    ``raw_grid_energy_mwh`` and clamped to zero for classification.
    """

    try:
        raw_grid_energy = float(grid_energy_mwh)
    except (TypeError, ValueError) as exc:
        raise ValueError("grid_energy_mwh must be finite") from exc
    if not math.isfinite(raw_grid_energy):
        raise ValueError("grid_energy_mwh must be finite")
    reference_energy = _finite_positive(
        electrical_reference_energy_mwh,
        "electrical_reference_energy_mwh",
    )
    absolute_tolerance = _finite_nonnegative(
        absolute_tolerance_mwh,
        "absolute_tolerance_mwh",
    )
    relative = _finite_nonnegative(relative_tolerance, "relative_tolerance")

    tolerance = max(absolute_tolerance, relative * reference_energy)
    if raw_grid_energy < -tolerance:
        raise ValueError(
            "grid_energy_mwh is materially negative; only numerical solver noise may be clamped"
        )
    classified_grid_energy = max(0.0, raw_grid_energy)
    if classified_grid_energy > reference_energy + tolerance:
        raise ValueError(
            "grid_energy_mwh exceeds total electrical generator supply; "
            "the reference-energy denominator is inconsistent"
        )
    return GridUseClassification(
        raw_grid_energy_mwh=raw_grid_energy,
        grid_energy_mwh=classified_grid_energy,
        electrical_reference_energy_mwh=reference_energy,
        grid_energy_share=classified_grid_energy / reference_energy,
        tolerance_mwh=tolerance,
        uses_grid_backstop=classified_grid_energy > tolerance,
    )


def finalize_grid_reporting(
    results: Mapping[str, Any],
    *,
    electrical_reference_energy_mwh: float,
    absolute_tolerance_mwh: float = 1.0,
    relative_tolerance: float = 1e-6,
) -> dict[str, Any]:
    """Return results with honest grid-free reporting.

    If backstop dispatch is numerical noise, the solved plant is classified as
    grid-free and its full production and headline LCOA are retained.  If grid
    dispatch is material, counterfactual grid-free production and LCOA are
    unknown and therefore set to NaN.  No proportional attribution is made.
    """

    output = dict(results)
    # Remove the former independently calculated tolerance alias, if a caller
    # passes an older result mapping through this canonical finalizer.
    output.pop("grid_backstop_tolerance_mwh", None)
    if "grid_energy_mwh" not in output:
        raise ValueError("results must include 'grid_energy_mwh'")

    classification = classify_grid_use(
        output["grid_energy_mwh"],
        electrical_reference_energy_mwh,
        absolute_tolerance_mwh=absolute_tolerance_mwh,
        relative_tolerance=relative_tolerance,
    )
    output["grid_energy_raw_mwh"] = classification.raw_grid_energy_mwh
    output["grid_energy_mwh"] = classification.grid_energy_mwh
    output["grid_energy_reference_mwh"] = (
        classification.electrical_reference_energy_mwh
    )
    output["grid_energy_reference_basis"] = "power_bus_generator_supply"
    output["grid_energy_share"] = classification.grid_energy_share
    output["grid_energy_tolerance_mwh"] = classification.tolerance_mwh
    output["uses_grid_backstop"] = classification.uses_grid_backstop
    output["is_gridless_feasible"] = not classification.uses_grid_backstop

    currency = _currency_code(output.get("currency", ""), "results['currency']")
    currency_slug = currency.lower()
    production = output.get("annual_ammonia_production_t")
    headline_lcoa = output.get(f"lcoa_{currency_slug}_per_t")

    # Remove any stale currency-specific counterfactual inherited from earlier
    # proportional reporting before writing the one canonical output field.
    for key in [key for key in output if key.startswith("lcoa_gridless_")]:
        output.pop(key)

    if classification.uses_grid_backstop:
        output["grid_free_result_status"] = "requires_grid_free_resolve"
        output["gridless_energy_fraction"] = math.nan
        output["gridless_ammonia_production_t"] = math.nan
        output[f"lcoa_gridless_{currency_slug}_per_t"] = math.nan
    else:
        output["grid_free_result_status"] = "equivalent_within_tolerance"
        output["gridless_energy_fraction"] = 1.0
        output["gridless_ammonia_production_t"] = (
            _finite_nonnegative(production, "annual_ammonia_production_t")
            if production is not None
            else math.nan
        )
        output[f"lcoa_gridless_{currency_slug}_per_t"] = (
            _finite_nonnegative(headline_lcoa, f"lcoa_{currency_slug}_per_t")
            if headline_lcoa is not None
            else math.nan
        )

    return output
