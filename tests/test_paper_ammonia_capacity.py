import math
import sys
import types
import unittest
from types import SimpleNamespace

import pandas as pd

# The estimator tested here is pure pandas/math logic.  Importing run_global also
# imports the full solver stack, so keep this unit test independent of a local
# Pyomo installation.
pyomo_module = types.ModuleType("pyomo")
pyomo_environ_module = types.ModuleType("pyomo.environ")
pyomo_module.environ = pyomo_environ_module
sys.modules.setdefault("pyomo", pyomo_module)
sys.modules.setdefault("pyomo.environ", pyomo_environ_module)

from model.run_global import (
    _compute_headline_splits,
    _estimate_paper_ammonia_capacity,
    _apply_spatial_solar_density_from_tech_config,
    run_global,
)


class PaperAmmoniaCapacityTest(unittest.TestCase):
    def test_explicit_density_survives_runtime_and_legacy_remains_unchanged(self):
        tech={"solar":{"land_use_km2_per_mw":.01},"solar_tracking":{"land_use_km2_per_mw":.02}}
        land=pd.DataFrame({"latitude":[-23.],"solar_density_mw_per_km2":[83.446157],"solar_area_km2":[100.]})
        legacy=_apply_spatial_solar_density_from_tech_config(land,tech)
        self.assertAlmostEqual(legacy.solar_density_mw_per_km2.iloc[0],73.599292,places=5)
        land["solar_density_method"]="explicit_fixed_pv_density_v1"
        explicit=_apply_spatial_solar_density_from_tech_config(land,tech)
        self.assertEqual(explicit.solar_density_mw_per_km2.iloc[0],83.446157)
        self.assertAlmostEqual(explicit.solar_tracking_density_mw_per_km2.iloc[0],83.446157/2)
        self.assertAlmostEqual(explicit.max_power_solar_mw.iloc[0],8344.6157)

    def test_explicit_density_fails_closed(self):
        for density,method in [(0.,"explicit_fixed_pv_density_v1"),(float("nan"),"explicit_fixed_pv_density_v1"),(100.,"typo")]:
            with self.subTest(density=density,method=method):
                land=pd.DataFrame({"latitude":[0.],"solar_density_mw_per_km2":[density],"solar_density_method":[method]})
                with self.assertRaises(ValueError):
                    _apply_spatial_solar_density_from_tech_config(land,{})

    def test_onshore_capacity_can_exceed_solved_plant_and_be_wind_limited(self):
        results = {
            "annual_ammonia_production_t": 1_000_000.0,
            "gridless_ammonia_production_t": 800_000.0,
            "wind_mw": 500.0,
            "solar_mw": 300.0,
            "solar_tracking_mw": 200.0,
        }
        land_row = pd.Series(
            {
                "max_power_wind_mw": 2_000.0,
                "max_power_solar_mw": 3_000.0,
                "wind_onshore_area_km2": 750.0,
                "wind_density_mw_per_km2": 2.0,
            }
        )

        capacity = _estimate_paper_ammonia_capacity(results, land_row)

        self.assertEqual(capacity["capacity_limit_technology"], "wind")
        self.assertEqual(capacity["onshore_capacity_limit_technology"], "wind")
        self.assertEqual(capacity["renewable_capacity_scale_factor"], 4.0)
        self.assertEqual(capacity["onshore_renewable_capacity_scale_factor"], 3.0)
        self.assertEqual(capacity["max_onshore_ammonia_capacity_t"], 3_000_000.0)
        self.assertGreater(capacity["max_onshore_ammonia_capacity_t"], 1_000_000.0)
        self.assertEqual(capacity["max_gridless_onshore_ammonia_capacity_t"], 2_400_000.0)
        self.assertEqual(capacity["wind_mw_per_t_nh3"], 0.0005)
        self.assertEqual(capacity["solar_mw_per_t_nh3"], 0.0005)

    def test_legacy_store_scaling_does_not_bias_build_multiplier(self):
        network = SimpleNamespace(
            objective=1_000.0,
            generators=pd.DataFrame(),
            links=pd.DataFrame(),
            stores=pd.DataFrame(
                {"e_nom_opt": [2.0]}, index=["compressed_hydrogen_store"]
            ),
        )
        tech_inputs = {
            "compressed_hydrogen_store": {
                "component_type": "store",
                "tech_cost_per_mwh": 60.0,
                "build_cost_per_mwh": 40.0,
                "lifetime_years": 20.0,
                "interest_rate": 0.05,
                "fixed_om_fraction": 0.0,
            }
        }

        splits = _compute_headline_splits(
            network,
            tech_inputs,
            overrides=None,
            aggregation_count=1,
            time_step=4.0,
            temporal_accounting_mode="legacy_scaled",
        )

        self.assertAlmostEqual(splits["build_cost_multiplier"], 1.0)

    def test_scaled_design_rejects_land_enforced_reference_design(self):
        with self.assertRaisesRegex(ValueError, "requires land_constraint='after_solve'"):
            run_global(land_constraint="in_solve", capacity_rule="scaled_reference_design")

    def test_missing_land_row_is_graceful(self):
        results = {
            "annual_ammonia_production_t": 1_000_000.0,
            "gridless_ammonia_production_t": 500_000.0,
            "wind_mw": 500.0,
            "solar_mw": 500.0,
        }

        capacity = _estimate_paper_ammonia_capacity(results, None)

        self.assertEqual(capacity["capacity_limit_technology"], "unknown")
        self.assertEqual(capacity["onshore_capacity_limit_technology"], "unknown")
        self.assertTrue(math.isnan(capacity["renewable_capacity_scale_factor"]))
        self.assertTrue(math.isnan(capacity["onshore_renewable_capacity_scale_factor"]))
        self.assertTrue(math.isnan(capacity["max_ammonia_capacity_t"]))
        self.assertTrue(math.isnan(capacity["max_onshore_ammonia_capacity_t"]))
        self.assertTrue(math.isnan(capacity["max_gridless_onshore_ammonia_capacity_t"]))
        self.assertEqual(capacity["wind_mw_per_t_nh3"], 0.0005)
        self.assertEqual(capacity["solar_mw_per_t_nh3"], 0.0005)


if __name__ == "__main__":
    unittest.main()
