import math
import unittest

from model.result_accounting import (
    classify_grid_use,
    finalize_grid_reporting,
    finalize_headline_cost_percentages,
    finalize_site_cost_accounting,
)


class SiteCostAccountingTest(unittest.TestCase):
    def test_explicit_fx_and_headline_identity(self):
        original = {
            "currency": "EUR",
            "annual_ammonia_production_t": 1_000_000.0,
            "total_cost_eur_per_year": 200_000_000.0,
            "lcoa_eur_per_t": 200.0,
        }

        result = finalize_site_cost_accounting(
            original,
            output_currency="EUR",
            source_currency="USD",
            source_to_output_fx=0.875,
            water_cost_source_per_m3=2.0,
            water_usage_m3_per_t_nh3=1.5,
            land_cost_source_per_km2_year=1_000.0,
            land_used_km2=100.0,
        )

        self.assertEqual(original["total_cost_eur_per_year"], 200_000_000.0)
        self.assertEqual(result["plant_objective_cost_eur_per_year"], 200_000_000.0)
        self.assertEqual(result["lcoa_plant_eur_per_t"], 200.0)
        self.assertEqual(result["water_cost_usd_per_t"], 3.0)
        self.assertEqual(result["water_cost_eur_per_year"], 2_625_000.0)
        self.assertEqual(result["land_cost_usd_per_year"], 100_000.0)
        self.assertEqual(result["land_cost_eur_per_year"], 87_500.0)
        self.assertEqual(result["total_cost_eur_per_year"], 202_712_500.0)
        self.assertEqual(result["lcoa_eur_per_t"], 202.7125)
        self.assertAlmostEqual(
            result["plant_cost_pct"]
            + result["water_cost_pct"]
            + result["land_cost_pct"],
            100.0,
        )
        self.assertEqual(
            result["headline_cost_identity_residual_eur_per_year"],
            0.0,
        )

    def test_finalization_is_idempotent(self):
        inputs = {
            "currency": "USD",
            "annual_ammonia_production_t": 10.0,
            "total_cost_usd_per_year": 1_000.0,
        }
        once = finalize_site_cost_accounting(
            inputs,
            output_currency="USD",
            source_currency="USD",
            source_to_output_fx=1.0,
            water_cost_source_per_m3=2.0,
            water_usage_m3_per_t_nh3=1.5,
        )
        twice = finalize_site_cost_accounting(
            once,
            output_currency="USD",
            source_currency="USD",
            source_to_output_fx=1.0,
            water_cost_source_per_m3=2.0,
            water_usage_m3_per_t_nh3=1.5,
        )

        self.assertEqual(once["total_cost_usd_per_year"], 1_030.0)
        self.assertEqual(twice["total_cost_usd_per_year"], 1_030.0)
        self.assertEqual(twice["plant_objective_cost_usd_per_year"], 1_000.0)

    def test_replication_can_report_site_costs_without_adding_them(self):
        result = finalize_site_cost_accounting(
            {
                "currency": "EUR",
                "annual_ammonia_production_t": 1_000_000.0,
                "total_cost_eur_per_year": 200_000_000.0,
            },
            output_currency="EUR",
            source_currency="USD",
            source_to_output_fx=0.875,
            water_cost_source_per_m3=2.0,
            water_usage_m3_per_t_nh3=1.5,
            land_cost_source_per_km2_year=0.0,
            land_used_km2=100.0,
            include_site_costs_in_headline=False,
        )

        self.assertFalse(result["site_costs_in_headline"])
        self.assertEqual(result["water_cost_eur_per_year"], 2_625_000.0)
        self.assertEqual(result["total_cost_eur_per_year"], 200_000_000.0)
        self.assertEqual(result["lcoa_eur_per_t"], 200.0)
        self.assertEqual(result["plant_cost_pct"], 100.0)
        self.assertEqual(result["water_cost_pct"], 0.0)

    def test_headline_component_percentages_sum_to_100_with_true_residual(self):
        site_accounted = finalize_site_cost_accounting(
            {
                "currency": "EUR",
                "annual_ammonia_production_t": 100.0,
                "total_cost_eur_per_year": 800.0,
                # This is an overlapping residual from a different breakdown;
                # headline reconciliation must replace it, not scale it.
                "other_cost_pct": 75.0,
            },
            output_currency="EUR",
            source_currency="EUR",
            source_to_output_fx=1.0,
            water_cost_source_per_m3=1.0,
            water_usage_m3_per_t_nh3=1.0,
            land_cost_source_per_km2_year=100.0,
            land_used_km2=1.0,
        )
        result = finalize_headline_cost_percentages(
            site_accounted,
            plant_split_percentages={
                "build_cost_pct": 10.0,
                "tech_cost_pct": 20.0,
                "om_cost_pct": 30.0,
                "interest_pct": 5.0,
            },
        )

        mutually_exclusive = (
            "build_cost_pct",
            "tech_cost_pct",
            "om_cost_pct",
            "interest_pct",
            "other_cost_pct",
            "water_cost_pct",
            "land_cost_pct",
        )
        self.assertEqual(math.fsum(result[key] for key in mutually_exclusive), 100.0)
        self.assertEqual(result["other_cost_pct"], 28.0)
        self.assertEqual(result["plant_cost_pct"], 80.0)
        self.assertEqual(result["headline_cost_share_total_pct"], 100.0)
        self.assertEqual(result["headline_cost_share_residual_pct"], 0.0)

    def test_incomplete_or_implicit_cost_inputs_fail(self):
        inputs = {
            "annual_ammonia_production_t": 1_000_000.0,
            "total_cost_eur_per_year": 200_000_000.0,
        }
        with self.assertRaisesRegex(ValueError, "source_to_output_fx"):
            finalize_site_cost_accounting(
                inputs,
                output_currency="EUR",
                source_currency="USD",
                source_to_output_fx=0.0,
            )
        with self.assertRaisesRegex(ValueError, "water_usage_m3_per_t_nh3"):
            finalize_site_cost_accounting(
                inputs,
                output_currency="EUR",
                source_currency="USD",
                source_to_output_fx=0.875,
                water_cost_source_per_m3=2.0,
            )
        with self.assertRaisesRegex(ValueError, "must be 1.0"):
            finalize_site_cost_accounting(
                {
                    "annual_ammonia_production_t": 1_000_000.0,
                    "total_cost_usd_per_year": 200_000_000.0,
                },
                output_currency="USD",
                source_currency="USD",
                source_to_output_fx=0.875,
            )


class GridUseReportingTest(unittest.TestCase):
    def test_relative_tolerance_classifies_solver_noise_as_gridless(self):
        classification = classify_grid_use(
            5.0,
            9_000_000.0,
            absolute_tolerance_mwh=1.0,
            relative_tolerance=1e-6,
        )

        self.assertEqual(classification.tolerance_mwh, 9.0)
        self.assertFalse(classification.uses_grid_backstop)

        result = finalize_grid_reporting(
            {
                "currency": "EUR",
                "annual_ammonia_production_t": 1_000_000.0,
                "lcoa_eur_per_t": 225.0,
                "grid_energy_mwh": 5.0,
                "gridless_energy_fraction": 0.25,
                "gridless_ammonia_production_t": 250_000.0,
                "lcoa_gridless_eur_per_t": -1e12,
            },
            electrical_reference_energy_mwh=9_000_000.0,
        )

        self.assertFalse(result["uses_grid_backstop"])
        self.assertTrue(result["is_gridless_feasible"])
        self.assertEqual(result["gridless_energy_fraction"], 1.0)
        self.assertEqual(result["gridless_ammonia_production_t"], 1_000_000.0)
        self.assertEqual(result["lcoa_gridless_eur_per_t"], 225.0)
        self.assertEqual(result["grid_free_result_status"], "equivalent_within_tolerance")
        self.assertEqual(result["grid_energy_reference_mwh"], 9_000_000.0)
        self.assertEqual(result["grid_energy_reference_basis"], "power_bus_generator_supply")
        self.assertEqual(result["grid_energy_share"], 5.0 / 9_000_000.0)

    def test_material_grid_use_never_creates_proportional_counterfactual(self):
        result = finalize_grid_reporting(
            {
                "currency": "EUR",
                "annual_ammonia_production_t": 1_000_000.0,
                "lcoa_eur_per_t": 225_000.0,
                "grid_energy_mwh": 9_000_000.0,
                "gridless_energy_fraction": 0.01,
                "gridless_ammonia_production_t": 10_000.0,
                "lcoa_gridless_eur_per_t": -1e12,
            },
            electrical_reference_energy_mwh=9_100_000.0,
        )

        self.assertTrue(result["uses_grid_backstop"])
        self.assertFalse(result["is_gridless_feasible"])
        self.assertTrue(math.isnan(result["gridless_energy_fraction"]))
        self.assertTrue(math.isnan(result["gridless_ammonia_production_t"]))
        self.assertTrue(math.isnan(result["lcoa_gridless_eur_per_t"]))
        self.assertEqual(result["grid_free_result_status"], "requires_grid_free_resolve")
        self.assertEqual(result["grid_energy_share"], 9_000_000.0 / 9_100_000.0)

    def test_tiny_negative_dispatch_is_preserved_and_clamped(self):
        result = finalize_grid_reporting(
            {
                "currency": "EUR",
                "annual_ammonia_production_t": 1_000_000.0,
                "lcoa_eur_per_t": 225.0,
                "grid_energy_mwh": -1e-9,
            },
            electrical_reference_energy_mwh=9_000_000.0,
        )

        self.assertEqual(result["grid_energy_raw_mwh"], -1e-9)
        self.assertEqual(result["grid_energy_mwh"], 0.0)
        self.assertFalse(result["uses_grid_backstop"])

    def test_materially_negative_dispatch_fails(self):
        with self.assertRaisesRegex(ValueError, "materially negative"):
            classify_grid_use(
                -10.0,
                1_000_000.0,
                absolute_tolerance_mwh=1.0,
                relative_tolerance=0.0,
            )

    def test_grid_energy_cannot_exceed_electrical_supply_denominator(self):
        with self.assertRaisesRegex(ValueError, "denominator is inconsistent"):
            classify_grid_use(
                101.0,
                100.0,
                absolute_tolerance_mwh=0.5,
                relative_tolerance=0.0,
            )


if __name__ == "__main__":
    unittest.main()
