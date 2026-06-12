import math
import sys
import types
import unittest

import pandas as pd

# The estimator tested here is pure pandas/math logic.  Importing run_global also
# imports the full solver stack, so keep this unit test independent of a local
# Pyomo installation.
pyomo_module = types.ModuleType("pyomo")
pyomo_environ_module = types.ModuleType("pyomo.environ")
pyomo_module.environ = pyomo_environ_module
sys.modules.setdefault("pyomo", pyomo_module)
sys.modules.setdefault("pyomo.environ", pyomo_environ_module)

from model.run_global import _estimate_paper_ammonia_capacity


class PaperAmmoniaCapacityTest(unittest.TestCase):
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
