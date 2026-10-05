import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest

import pandas as pd

from arc.merge_and_qa_campaign import _run
from arc.write_campaign_manifest import validate_override_coordinate_coverage


class CampaignQaTest(unittest.TestCase):
    def _fixture(
        self,
        root: Path,
        *,
        include_site_costs: bool = True,
    ) -> tuple[SimpleNamespace, dict]:
        scenario = (
            "central_way2050_flat_amelired_1h_tracking_explicit_compressor_dea_tank"
            if include_site_costs
            else "rep_way2050_flat_amelired_4h_tracking_nominal_h2"
        )
        land_allocation = "colocated" if include_site_costs else "exclusive"
        temporal_mode = "snapshot_weighted" if include_site_costs else "legacy_scaled"
        ramp_basis = "per_hour" if include_site_costs else "legacy_per_snapshot"
        time_step_hours = 1.0 if include_site_costs else 4.0
        manifest_path = root / "manifest.json"
        manifest = {
            "campaign_id": "qa-test",
            "run_id": "run-1",
            "scenario": {"id": scenario},
            "inputs": {
                "override_csv": {
                    "columns": ["lat", "lon", "tech", "interest_rate"]
                }
            },
            "execution": {
                "stage": "global",
                "fail_fast": True,
                "land_constraint": "after_solve",
                "capacity_rule": "scaled_reference_design",
                "land_allocation": land_allocation,
                "temporal_accounting_mode": temporal_mode,
                "ramp_limit_basis": ramp_basis,
                "include_site_costs": include_site_costs,
                "time_step_hours": time_step_hours,
                "simulated_hours": 8760.0,
                "expected_full_year": True,
            },
            "cost_scope": {
                "schema_version": 1,
                "id": "flat_amelired_uniform_baseline_water_no_land_rent",
                "finance": {
                    "mode": "ameli_reduced_wacc",
                    "source": "override_csv",
                    "override_column": "interest_rate",
                },
                "spatial_build_and_remoteness": {
                    "mode": "none_flat",
                    "override_columns_present": [],
                    "expected_result_build_cost_multiplier": 1.0,
                },
                "water": {
                    "mode": "uniform_yaml_baseline",
                    "source": "resolved_tech_config.water_cost_baseline_usd_per_m3",
                    "source_currency": "USD",
                    "expected_result_cost_usd_per_m3": 2.0, "model_currency": "EUR", "source_to_model_currency": 0.875506, "expected_result_cost_model_currency_per_m3": 1.751012,
                    "usage_m3_per_t_nh3": 1.5,
                    "included_in_headline": include_site_costs,
                },
                "land_rent": {
                    "mode": "none_zero",
                    "source": "no_land_rent_input",
                    "source_currency": "USD",
                    "expected_result_cost_usd_per_km2_year": 0.0,
                    "included_in_headline": False,
                },
            },
        }
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        manifest_sha = hashlib.sha256(manifest_path.read_bytes()).hexdigest()

        land_path = root / "land.csv"
        pd.DataFrame(
            [{"latitude": 0.0, "longitude": 0.0, "max_capacity_mw": 1.0}]
        ).to_csv(land_path, index=False)

        row = {
            "latitude": 0.0,
            "longitude": 0.0,
            "country": "Testland",
            "currency": "EUR",
            "annual_ammonia_production_t": 1.0,
            "max_ammonia_capacity_t": 1.0,
            # A valid global LCOA row may have no corrected onshore supply.
            "max_onshore_ammonia_capacity_t": 0.0,
            "max_gridless_onshore_ammonia_capacity_t": 0.0,
            "scaled_design_max_onshore_ammonia_capacity_t": 0.0,
            "scaled_design_max_gridless_onshore_ammonia_capacity_t": 0.0,
            "wind_mw_per_t_nh3": 1.0,
            "solar_mw_per_t_nh3": 1.0,
            "lcoa_eur_per_t": 100.0,
            "lcoa_plant_eur_per_t": 99.0,
            "total_cost_eur_per_year": 100.0,
            "grid_energy_mwh": 0.0,
            "grid_energy_reference_mwh": 10.0,
            "grid_energy_share": 0.0,
            "grid_energy_tolerance_mwh": 1.0,
            "uses_grid_backstop": False,
            "grid_free_result_status": "equivalent_within_tolerance",
            "is_gridless_feasible": True,
            "interest_overrides_applied": True,
            "scenario_id": scenario,
            "run_id": "run-1",
            "manifest_sha256": manifest_sha,
            "land_constraint": "after_solve",
            "capacity_rule": "scaled_reference_design",
            "land_allocation": land_allocation,
            "temporal_accounting_mode": temporal_mode,
            "ramp_limit_basis": ramp_basis,
            "site_costs_in_headline": include_site_costs,
            "build_cost_multiplier": 1.0,
            "water_cost_usd_per_m3": 2.0,
            "land_cost_usd_per_km2_year": 0.0,
            "water_cost_pct": 1.0 if include_site_costs else 0.0,
            "land_cost_pct": 0.0,
            "snapshot_hours": time_step_hours,
            "simulated_hours": 8760.0,
            "is_full_year_result": True,
            "headline_cost_identity_residual_eur_per_year": 0.0,
            "headline_cost_share_total_pct": 100.0,
            "headline_cost_share_residual_pct": 0.0,
            "renewable_union_area_source": "explicit_classwise_nested_v1_lower_bound",
            "renewable_union_area_is_conservative_fallback": False,
            "renewable_union_area_is_lower_bound_approximation": True,
            "renewable_union_area_method": "classwise_nested_overlap_lower_bound",
            "renewable_union_area_method_version": "v1",
            "legacy_capacity_alias_method": (
                "historical_independent_total_vs_onshore_power_caps_v0"
            ),
            "preferred_supplier_capacity_column": (
                "scaled_design_max_gridless_onshore_ammonia_capacity_t"
            ),
        }
        input_path = root / "shard.csv"
        pd.DataFrame([row]).to_csv(input_path, index=False)

        args = SimpleNamespace(
            input=[str(input_path)],
            expected_input_count=1,
            output=str(root / "merged.csv"),
            qa_output=str(root / "qa.json"),
            manifest=str(manifest_path),
            scenario_id=scenario,
            stage="global",
            expected_locations=str(land_path),
            expected_locations_kind="land",
            expected_currency="EUR",
            require_interest_overrides=True,
            require_full_year=True,
        )
        return args, row

    def test_global_qa_accepts_and_counts_zero_corrected_supplier_capacity(self):
        with tempfile.TemporaryDirectory() as temporary:
            args, _ = self._fixture(Path(temporary))

            report = _run(args)

            self.assertEqual(report["status"], "passed")
            self.assertEqual(report["zero_corrected_supplier_capacity_rows"], 1)
            self.assertEqual(report["positive_corrected_supplier_capacity_rows"], 0)

    def test_global_qa_rejects_nonfinite_lcoa(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            args, row = self._fixture(root)
            row["lcoa_eur_per_t"] = float("nan")
            pd.DataFrame([row]).to_csv(args.input[0], index=False)

            with self.assertRaisesRegex(ValueError, "non-finite"):
                _run(args)

    def test_replication_accepts_baseline_water_outside_headline(self):
        with tempfile.TemporaryDirectory() as temporary:
            args, _ = self._fixture(
                Path(temporary), include_site_costs=False
            )

            report = _run(args)

            self.assertEqual(report["status"], "passed")
            self.assertFalse(
                report["cost_scope"]["water"]["included_in_headline"]
            )

    def test_qa_rejects_result_cost_scope_mismatches(self):
        cases = (
            ("build_cost_multiplier", 1.1, "Expected build_cost_multiplier=1.0"),
            ("water_cost_usd_per_m3", 3.0, "Expected water_cost_usd_per_m3=2.0"),
            (
                "land_cost_usd_per_km2_year",
                100.0,
                "Expected land_cost_usd_per_km2_year=0.0",
            ),
            ("land_cost_pct", 0.1, "Expected land_cost_pct=0.0"),
            (
                "water_cost_pct",
                0.0,
                "headline-active but water_cost_pct is not positive",
            ),
        )
        for column, value, message in cases:
            with self.subTest(column=column), tempfile.TemporaryDirectory() as temporary:
                args, row = self._fixture(Path(temporary))
                row[column] = value
                pd.DataFrame([row]).to_csv(args.input[0], index=False)

                with self.assertRaisesRegex(ValueError, message):
                    _run(args)

    def test_qa_rejects_spatial_columns_in_flat_override_manifest(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            args, _ = self._fixture(root)
            manifest_path = Path(args.manifest)
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["inputs"]["override_csv"]["columns"].append(
                "build_cost_multiplier"
            )
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "interest-only override CSV"):
                _run(args)


class OverrideCoverageTest(unittest.TestCase):
    def _write_override(self, path: Path, coordinates: list[tuple[float, float]]) -> None:
        pd.DataFrame(
            [
                {"lat": lat, "lon": lon, "tech": "wind", "interest_rate": 0.051}
                for lat, lon in coordinates
            ]
        ).to_csv(path, index=False)

    def test_explicit_location_coverage_accepts_override_superset(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            override = root / "override.csv"
            locations = root / "locations.csv"
            land = root / "unused-land.csv"
            self._write_override(override, [(0.0, 0.0), (1.0, 1.0)])
            pd.DataFrame([{"lat": 1.0, "lon": 1.0}]).to_csv(locations, index=False)

            report = validate_override_coordinate_coverage(
                override_csv=override,
                land_csv=land,
                locations_csv=locations,
            )

            self.assertEqual(report["expected_coordinate_count"], 1)
            self.assertEqual(report["override_coordinate_count"], 2)

    def test_active_global_land_coverage_ignores_zero_capacity_cells(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            override = root / "override.csv"
            land = root / "land.csv"
            self._write_override(override, [(0.0, 0.0)])
            pd.DataFrame(
                [
                    {"latitude": 0.0, "longitude": 0.0, "max_capacity_mw": 1.0},
                    {"latitude": 1.0, "longitude": 1.0, "max_capacity_mw": 0.0},
                ]
            ).to_csv(land, index=False)

            report = validate_override_coordinate_coverage(
                override_csv=override,
                land_csv=land,
                locations_csv=None,
            )

            self.assertEqual(report["expected_source"], "active_land_csv")
            self.assertEqual(report["expected_coordinate_count"], 1)

    def test_missing_override_coordinate_fails_preflight(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            override = root / "override.csv"
            locations = root / "locations.csv"
            self._write_override(override, [(0.0, 0.0)])
            pd.DataFrame([{"lat": 2.0, "lon": 2.0}]).to_csv(locations, index=False)

            with self.assertRaisesRegex(ValueError, "missing=1"):
                validate_override_coordinate_coverage(
                    override_csv=override,
                    land_csv=root / "unused.csv",
                    locations_csv=locations,
                )


if __name__ == "__main__":
    unittest.main()
