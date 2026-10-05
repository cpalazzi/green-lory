import math
import unittest

from model.land_capacity import (
    RenewableLandBudget,
    estimate_scaled_design_capacity,
    renewable_land_use_from_results,
)
from model.land_union import (
    CLASSWISE_NESTED_UNION_AREA_COLUMN,
    CLASSWISE_NESTED_UNION_METHOD,
    CLASSWISE_NESTED_UNION_SOURCE,
    CLASSWISE_NESTED_UNION_VERSION,
)


TECH_INPUTS = {
    "wind": {"land_use_km2_per_mw": 0.2},
    "solar": {"land_use_km2_per_mw": 0.01},
    "solar_tracking": {"land_use_km2_per_mw": 0.02},
}


def make_land_row(fraction: float = 1.0) -> dict[str, float]:
    return {
        "wind_onshore_area_km2": 1_000.0 * fraction,
        "solar_area_km2": 100.0 * fraction,
        "renewable_union_area_km2": 1_050.0 * fraction,
        CLASSWISE_NESTED_UNION_AREA_COLUMN: 1_050.0 * fraction,
        "wind_density_mw_per_km2": 5.0,
        "solar_density_mw_per_km2": 100.0,
    }


class RenewableLandBudgetTest(unittest.TestCase):
    def test_versioned_union_lower_bound_is_preferred_and_labelled(self):
        row = make_land_row()

        budget = RenewableLandBudget.from_land_row(row, TECH_INPUTS)

        self.assertEqual(budget.renewable_union_area_km2, 1_050.0)
        self.assertEqual(budget.union_area_source, CLASSWISE_NESTED_UNION_SOURCE)

        mapping = estimate_scaled_design_capacity(
            {
                "annual_ammonia_production_t": 1_000_000.0,
                "wind_mw": 0.0,
                "solar_mw": 100.0,
                "solar_tracking_mw": 0.0,
            },
            budget,
        ).to_mapping()
        self.assertTrue(mapping["renewable_union_area_is_lower_bound_approximation"])
        self.assertEqual(
            mapping["renewable_union_area_method"], CLASSWISE_NESTED_UNION_METHOD
        )
        self.assertEqual(
            mapping["renewable_union_area_method_version"],
            CLASSWISE_NESTED_UNION_VERSION,
        )

    def test_divergent_union_compatibility_alias_is_rejected(self):
        row = make_land_row()
        row["renewable_union_area_km2"] = 999_999.0

        with self.assertRaisesRegex(ValueError, "compatibility alias diverges"):
            RenewableLandBudget.from_land_row(row, TECH_INPUTS)

    def test_unversioned_explicit_union_is_not_mislabelled_as_v1(self):
        row = make_land_row()
        row.pop(CLASSWISE_NESTED_UNION_AREA_COLUMN)

        budget = RenewableLandBudget.from_land_row(row, TECH_INPUTS)

        self.assertEqual(budget.union_area_source, "explicit_legacy_unversioned")

    def test_tracking_uses_twice_fixed_pv_area(self):
        budget = RenewableLandBudget.from_land_row(make_land_row(), TECH_INPUTS)

        self.assertEqual(budget.fixed_solar_density_mw_per_km2, 100.0)
        self.assertEqual(budget.tracking_solar_density_mw_per_km2, 50.0)
        self.assertEqual(budget.tracking_land_multiplier, 2.0)
        self.assertEqual(budget.tracking_density_source, "tech_land_use_ratio")

        land_use = renewable_land_use_from_results(
            {
                "wind_mw": 0.0,
                "solar_mw": 300.0,
                "solar_tracking_mw": 200.0,
            },
            budget,
        )
        self.assertEqual(land_use.fixed_solar_km2, 3.0)
        self.assertEqual(land_use.tracking_solar_km2, 4.0)
        self.assertEqual(land_use.solar_km2, 7.0)

    def test_old_land_row_marks_conservative_union_fallback(self):
        row = make_land_row()
        row.pop("renewable_union_area_km2")
        row.pop(CLASSWISE_NESTED_UNION_AREA_COLUMN)

        budget = RenewableLandBudget.from_land_row(row, TECH_INPUTS)

        self.assertEqual(budget.renewable_union_area_km2, 1_000.0)
        self.assertTrue(budget.uses_conservative_union_fallback)
        self.assertEqual(budget.union_area_source, "conservative_max")


class ScaledDesignCapacityTest(unittest.TestCase):
    def test_scale_below_one_is_not_floored_to_reference_plant(self):
        budget = RenewableLandBudget(
            wind_area_km2=59.2,
            solar_area_km2=1_000.0,
            renewable_union_area_km2=1_000.0,
            wind_density_mw_per_km2=5.0,
            fixed_solar_density_mw_per_km2=100.0,
            tracking_solar_density_mw_per_km2=50.0,
            union_area_source="explicit",
            wind_area_source="wind_onshore_area_km2",
            tracking_density_source="tech_land_use_ratio",
        )
        results = {
            "annual_ammonia_production_t": 1_000_000.0,
            "gridless_ammonia_production_t": 1_000_000.0,
            "wind_mw": 500.0,  # 100 km2 in the reference design
            "solar_mw": 0.0,
            "solar_tracking_mw": 0.0,
        }

        capacity = estimate_scaled_design_capacity(
            results,
            budget,
            land_allocation="exclusive",
        )

        self.assertAlmostEqual(capacity.scale_factor, 0.592)
        self.assertAlmostEqual(capacity.capacity_t, 592_000.0)
        self.assertEqual(capacity.limiting_constraint, "wind")
        mapped = capacity.to_mapping()
        self.assertAlmostEqual(
            mapped["scaled_design_max_onshore_ammonia_capacity_t"], 592_000.0
        )
        self.assertNotIn("max_ammonia_capacity_t", mapped)

    def test_union_can_bind_when_individual_technology_areas_do_not(self):
        budget = RenewableLandBudget(
            wind_area_km2=100.0,
            solar_area_km2=100.0,
            renewable_union_area_km2=100.0,
            wind_density_mw_per_km2=5.0,
            fixed_solar_density_mw_per_km2=100.0,
            tracking_solar_density_mw_per_km2=50.0,
            union_area_source="explicit",
            wind_area_source="wind_onshore_area_km2",
            tracking_density_source="tech_land_use_ratio",
        )
        results = {
            "annual_ammonia_production_t": 1_000_000.0,
            "wind_mw": 400.0,  # 80 km2
            "solar_mw": 4_000.0,  # 40 km2
            "solar_tracking_mw": 0.0,
        }

        shared = estimate_scaled_design_capacity(
            results,
            budget,
            land_allocation="exclusive",
        )
        colocated_default = estimate_scaled_design_capacity(
            results,
            budget,
            land_allocation="colocated",
        )
        overlap_budget = RenewableLandBudget(
            **{**budget.__dict__, "wind_land_exclusive_fraction": 0.0}
        )
        complete_overlap = estimate_scaled_design_capacity(
            results,
            overlap_budget,
            land_allocation="colocated",
        )

        self.assertEqual(shared.limiting_constraint, "renewable_union")
        self.assertAlmostEqual(shared.scale_factor, 100.0 / 120.0)
        self.assertAlmostEqual(shared.wind_land_exclusive_fraction_effective, 1.0)
        # colocated: only 3 % of the 80 km2 wind footprint competes with PV for the union budget
        self.assertEqual(colocated_default.limiting_constraint, "wind")
        self.assertAlmostEqual(colocated_default.scale_factor, 1.25)
        self.assertAlmostEqual(colocated_default.wind_land_exclusive_fraction_effective, 0.03)
        self.assertAlmostEqual(
            min(100.0 / (40.0 + 0.03 * 80.0), 100.0 / 80.0, 100.0 / 40.0), 1.25
        )
        self.assertEqual(complete_overlap.limiting_constraint, "wind")
        self.assertAlmostEqual(complete_overlap.scale_factor, 1.25)

    def test_colocated_union_binds_through_wind_exclusive_fraction(self):
        budget = RenewableLandBudget(
            wind_area_km2=1_000.0,
            solar_area_km2=1_000.0,
            renewable_union_area_km2=100.0,
            wind_density_mw_per_km2=5.0,
            fixed_solar_density_mw_per_km2=100.0,
            tracking_solar_density_mw_per_km2=50.0,
            union_area_source="explicit",
            wind_area_source="wind_onshore_area_km2",
            tracking_density_source="tech_land_use_ratio",
            wind_land_exclusive_fraction=0.5,
        )
        results = {
            "annual_ammonia_production_t": 1_000_000.0,
            "wind_mw": 400.0,  # 80 km2, of which 40 km2 count against the shared budget
            "solar_mw": 4_000.0,  # 40 km2
            "solar_tracking_mw": 0.0,
        }
        capacity = estimate_scaled_design_capacity(results, budget, land_allocation="colocated")
        self.assertEqual(capacity.limiting_constraint, "renewable_union")
        self.assertAlmostEqual(capacity.scale_factor, 100.0 / 80.0)
        exclusive = estimate_scaled_design_capacity(results, budget, land_allocation="exclusive")
        self.assertAlmostEqual(exclusive.scale_factor, 100.0 / 120.0)

    def test_solved_quantity_report_records_the_exclusive_fraction_not_the_gridless_fraction(self):
        from model.land_capacity import report_land_feasible_quantity

        budget = RenewableLandBudget(
            wind_area_km2=1_000.0, solar_area_km2=1_000.0, renewable_union_area_km2=1_000.0,
            wind_density_mw_per_km2=5.0, fixed_solar_density_mw_per_km2=100.0,
            tracking_solar_density_mw_per_km2=50.0, union_area_source="explicit",
            wind_area_source="wind_onshore_area_km2", tracking_density_source="tech_land_use_ratio",
            wind_land_exclusive_fraction=0.03,
        )
        results = {"annual_ammonia_production_t": 1_000_000.0, "gridless_ammonia_production_t": 1_000_000.0,
                   "wind_mw": 400.0, "solar_mw": 4_000.0, "solar_tracking_mw": 0.0}
        report = report_land_feasible_quantity(results, budget, land_allocation="colocated")
        self.assertAlmostEqual(report["wind_land_exclusive_fraction_effective"], 0.03)
        self.assertAlmostEqual(report["land_feasible_gridless_ammonia_production_t"], 1_000_000.0)
        self.assertAlmostEqual(report["renewable_union_land_slack_km2"], 1_000.0 - (40.0 + 0.03 * 80.0))
        exclusive = report_land_feasible_quantity(results, budget, land_allocation="exclusive")
        self.assertAlmostEqual(exclusive["wind_land_exclusive_fraction_effective"], 1.0)
        self.assertAlmostEqual(exclusive["renewable_union_land_slack_km2"], 1_000.0 - 120.0)

    def test_only_the_two_allocation_names_are_accepted(self):
        from model.land_capacity import normalise_land_allocation

        self.assertEqual(normalise_land_allocation("colocated"), "colocated")
        self.assertEqual(normalise_land_allocation("exclusive"), "exclusive")
        for old_name in ("technology_shared", "paper_union", "independent_legacy", "shared"):
            with self.assertRaises(ValueError):
                normalise_land_allocation(old_name)

    def test_budget_reads_wind_exclusive_fraction_from_tech_inputs(self):
        inputs = {**TECH_INPUTS, "wind": {**TECH_INPUTS["wind"], "land_use_exclusive_fraction": 0.1}}
        budget = RenewableLandBudget.from_land_row(make_land_row(0.02), inputs)
        self.assertAlmostEqual(budget.wind_land_exclusive_fraction, 0.1)
        default = RenewableLandBudget.from_land_row(make_land_row(0.02), TECH_INPUTS)
        self.assertAlmostEqual(default.wind_land_exclusive_fraction, 0.03)
        with self.assertRaises(ValueError):
            RenewableLandBudget.from_land_row(
                make_land_row(0.02), {**TECH_INPUTS, "wind": {**TECH_INPUTS["wind"], "land_use_exclusive_fraction": 1.5}}
            )

    def test_capacity_is_monotonic_with_competition_fraction(self):
        results = {
            "annual_ammonia_production_t": 1_000_000.0,
            "wind_mw": 500.0,
            "solar_mw": 2_000.0,
            "solar_tracking_mw": 500.0,
        }
        capacities = []
        for fraction in (0.02, 0.50, 1.0):
            budget = RenewableLandBudget.from_land_row(
                make_land_row(fraction),
                TECH_INPUTS,
            )
            capacities.append(
                estimate_scaled_design_capacity(
                    results,
                    budget,
                    land_allocation="exclusive",
                ).capacity_t
            )

        self.assertLess(capacities[0], capacities[1])
        self.assertLess(capacities[1], capacities[2])
        self.assertAlmostEqual(capacities[1] / capacities[0], 25.0)
        self.assertAlmostEqual(capacities[2] / capacities[1], 2.0)
        self.assertTrue(all(math.isfinite(value) for value in capacities))


if __name__ == "__main__":
    unittest.main()
