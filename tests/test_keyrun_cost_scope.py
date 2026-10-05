"""Flat cost scopes with a uniform WACC (key runs of 23 September 2026) beside the Ameli scope."""
from __future__ import annotations

import unittest
from pathlib import Path

from arc.merge_and_qa_campaign import _flat_cost_scope_expectations
from arc.write_campaign_manifest import build_flat_amelired_cost_scope, flat_finance_token

ROOT = Path(__file__).resolve().parents[1]
DEA_YAML = ROOT / "inputs/tech_config_ammonia_plant_2050_dea_colocated.yaml"
UNIFORM = ROOT / "inputs/uniform_interest_inputs_0p05_2050.csv"
AMELI = ROOT / "inputs/amelired_interest_inputs_2050.csv"


def _manifest(scope_id, mode, rate=0.05):
    finance = {"mode": mode, "source": "override_csv", "override_column": "interest_rate"}
    if mode == "uniform_wacc":
        finance["rate"] = rate
    return {
        "cost_scope": {
            "schema_version": 1, "id": scope_id, "finance": finance,
            "spatial_build_and_remoteness": {"mode": "none_flat", "override_columns_present": [], "expected_result_build_cost_multiplier": 1.0},
            "water": {"mode": "uniform_yaml_baseline", "source": "resolved_tech_config.water_cost_baseline_usd_per_m3", "source_currency": "USD",
                      "expected_result_cost_usd_per_m3": 2.0, "model_currency": "EUR", "source_to_model_currency": 0.875506, "expected_result_cost_model_currency_per_m3": 1.751012, "usage_m3_per_t_nh3": 1.5, "included_in_headline": True},
            "land_rent": {"mode": "none_zero", "source": "no_land_rent_input", "expected_result_cost_usd_per_km2_year": 0.0, "included_in_headline": False},
        },
        "inputs": {"override_csv": {"columns": ["lat", "lon", "tech", "interest_rate"]}},
    }


class QaGateTest(unittest.TestCase):
    def test_uniform_and_ameli_scopes_are_accepted(self):
        for scope_id, mode in [("flat_wacc5_uniform_baseline_water_no_land_rent", "uniform_wacc"), ("flat_amelired_uniform_baseline_water_no_land_rent", "ameli_reduced_wacc")]:
            out = _flat_cost_scope_expectations(_manifest(scope_id, mode), {"include_site_costs": True})
            self.assertEqual(out["water_cost_usd_per_m3"], 2.0, scope_id)

    def test_finance_mode_must_match_the_scope_id(self):
        for scope_id, mode in [("flat_wacc5_uniform_baseline_water_no_land_rent", "ameli_reduced_wacc"), ("flat_amelired_uniform_baseline_water_no_land_rent", "uniform_wacc"), ("flat_spatial_uniform_baseline_water_no_land_rent", "uniform_wacc")]:
            with self.assertRaises(ValueError, msg=scope_id):
                _flat_cost_scope_expectations(_manifest(scope_id, mode), {"include_site_costs": True})


class ManifestBuilderTest(unittest.TestCase):
    def test_tokens(self):
        self.assertEqual(flat_finance_token("flat_wacc5_uniform_baseline_water_no_land_rent"), "wacc5")
        self.assertEqual(flat_finance_token("flat_amelired_uniform_baseline_water_no_land_rent"), "amelired")

    @unittest.skipUnless(UNIFORM.exists() and AMELI.exists() and DEA_YAML.exists(), "campaign inputs not present")
    def test_override_must_agree_with_the_declared_finance(self):
        scope = build_flat_amelired_cost_scope(tech_yaml=DEA_YAML, override_csv=UNIFORM, include_site_costs=True, cost_scope_id="flat_wacc5_uniform_baseline_water_no_land_rent", flat_water_model_currency_per_m3=2.0)
        self.assertEqual(scope["finance"]["mode"], "uniform_wacc")
        self.assertAlmostEqual(scope["water"]["expected_result_cost_model_currency_per_m3"], 2.0, places=7)
        self.assertEqual(scope["water"]["model_currency"], "EUR")
        with self.assertRaises(ValueError):
            build_flat_amelired_cost_scope(tech_yaml=DEA_YAML, override_csv=UNIFORM, include_site_costs=True, cost_scope_id="flat_wacc5_uniform_baseline_water_no_land_rent", flat_water_model_currency_per_m3=1.75)
        self.assertAlmostEqual(scope["finance"]["rate"], 0.05)
        self.assertEqual(scope["id"], "flat_wacc5_uniform_baseline_water_no_land_rent")
        ameli = build_flat_amelired_cost_scope(tech_yaml=DEA_YAML, override_csv=AMELI, include_site_costs=True)
        self.assertEqual(ameli["finance"]["mode"], "ameli_reduced_wacc")
        with self.assertRaises(ValueError):
            build_flat_amelired_cost_scope(tech_yaml=DEA_YAML, override_csv=AMELI, include_site_costs=True, cost_scope_id="flat_wacc5_uniform_baseline_water_no_land_rent")
        with self.assertRaises(ValueError):
            build_flat_amelired_cost_scope(tech_yaml=DEA_YAML, override_csv=UNIFORM, include_site_costs=True)


if __name__ == "__main__":
    unittest.main()
