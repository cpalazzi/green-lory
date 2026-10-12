"""arc/qa.py: the merge and QA gate of a surface, and the land-table gate."""
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
import tempfile
import unittest

import pandas as pd

from arc.manifest import validate_coordinate_coverage, validate_finance_coverage
from arc.qa import (
    EXPECTED_RENEWABLE_UNION_METHOD,
    EXPECTED_RENEWABLE_UNION_METHOD_VERSION,
    LandCampaignInputError,
    _run,
    validate_land_campaign_input,
)


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
                "finance_csv": {
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
                    "source": "finance_csv",
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

    def test_qa_rejects_site_cost_columns_in_a_finance_table(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            args, _ = self._fixture(root)
            manifest_path = Path(args.manifest)
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["inputs"]["finance_csv"]["columns"].append(
                "build_cost_multiplier"
            )
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "no site-cost columns"):
                _run(args)


class CoverageTest(unittest.TestCase):
    def _write_table(self, path: Path, coordinates: list[tuple[float, float]]) -> None:
        pd.DataFrame([{"latitude": lat, "longitude": lon, "value": 1.0} for lat, lon in coordinates]).to_csv(path, index=False)

    def test_explicit_location_coverage_accepts_a_table_superset(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            table, locations = root / "site_costs.csv", root / "locations.csv"
            self._write_table(table, [(0.0, 0.0), (1.0, 1.0)])
            pd.DataFrame([{"lat": 1.0, "lon": 1.0}]).to_csv(locations, index=False)
            report = validate_coordinate_coverage(table_csv=table, land_csv=root / "unused-land.csv", locations_csv=locations)
            self.assertEqual((report["expected_coordinate_count"], report["table_coordinate_count"]), (1, 2))

    def test_active_global_land_coverage_ignores_zero_capacity_cells(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            table, land = root / "site_costs.csv", root / "land.csv"
            self._write_table(table, [(0.0, 0.0)])
            pd.DataFrame([
                {"latitude": 0.0, "longitude": 0.0, "max_capacity_mw": 1.0},
                {"latitude": 1.0, "longitude": 1.0, "max_capacity_mw": 0.0},
            ]).to_csv(land, index=False)
            report = validate_coordinate_coverage(table_csv=table, land_csv=land, locations_csv=None)
            self.assertEqual((report["expected_source"], report["expected_coordinate_count"]), ("active_land_csv", 1))

    def test_missing_coordinate_fails_preflight(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            table, locations = root / "site_costs.csv", root / "locations.csv"
            self._write_table(table, [(0.0, 0.0)])
            pd.DataFrame([{"lat": 2.0, "lon": 2.0}]).to_csv(locations, index=False)
            with self.assertRaisesRegex(ValueError, "missing=1"):
                validate_coordinate_coverage(table_csv=table, land_csv=root / "unused.csv", locations_csv=locations)


class FinanceCoverageTest(unittest.TestCase):
    TECH_YAML = Path(__file__).resolve().parents[1] / "inputs/tech/tech_config_ammonia_plant_2050_dea_colocated.yaml"

    def _check(self, rows, locations=((-23.0, -69.0), (-23.0, 117.0))):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            finance, cells = root / "finance.csv", root / "locations.csv"
            pd.DataFrame(rows).to_csv(finance, index=False)
            pd.DataFrame([{"lat": lat, "lon": lon} for lat, lon in locations]).to_csv(cells, index=False)
            return validate_finance_coverage(finance_csv=finance, tech_yaml=self.TECH_YAML, land_csv=root / "unused.csv",
                                             locations_csv=cells)

    def test_a_country_table_with_defaults_covers_the_cells(self):
        report = self._check([
            {"iso3": "CHL", "tech": "all", "interest_rate": 0.09},
            {"iso3": "CHL", "tech": "solar", "interest_rate": 0.07},
            {"iso3": "AUS", "tech": "all", "interest_rate": 0.06},
        ])
        self.assertEqual((report["key"], report["expected_coordinate_count"]), ("country", 2))

    def test_a_country_without_rates_fails_preflight(self):
        with self.assertRaisesRegex(ValueError, r"1 cell without an interest rate.*\(-23, 117\)"):
            self._check([{"iso3": "AUS", "tech": "solar", "interest_rate": 0.06},
                         {"iso3": "CHL", "tech": "all", "interest_rate": 0.09}])

    def test_unknown_technology_names_are_refused(self):
        with self.assertRaisesRegex(ValueError, "technologies the plant does not have"):
            self._check([{"iso3": "CHL", "tech": "onshore_wind", "interest_rate": 0.09}])


# --- the land-table gate --------------------------------------------------------------------

class ValidateLandCampaignInputTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)

    def tearDown(self):
        self.temporary.cleanup()

    @staticmethod
    def _row(latitude=0.0, longitude=0.0):
        return {
            "latitude": latitude,
            "longitude": longitude,
            "cell_anchor": "center",
            "availability": 0.6,
            "area": 100.0,
            "land_competition_fraction": 0.02,
            "wind_onshore_availability": 0.04,
            "wind_offshore_availability": 0.01,
            "wind_availability": 0.05,
            "solar_availability": 0.06,
            "renewable_union_availability_classwise_nested_v1": 0.07,
            "renewable_union_availability": 0.07,
            "renewable_union_method": EXPECTED_RENEWABLE_UNION_METHOD,
            "renewable_union_method_version": (
                EXPECTED_RENEWABLE_UNION_METHOD_VERSION
            ),
            "wind_onshore_area_km2": 4.0,
            "wind_offshore_area_km2": 1.0,
            "wind_area_km2": 5.0,
            "solar_area_km2": 6.0,
            "renewable_union_area_km2_classwise_nested_v1": 7.0,
            "renewable_union_area_km2": 7.0,
            "max_power_solar_mw": 600.0,
            "max_power_wind_mw": 25.0,
            "max_capacity_mw": 625.0,
        }

    def _write(self, rows, name="land.csv"):
        path = self.root / name
        fieldnames = list(rows[0]) if rows else list(self._row())
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        return path

    def _assert_invalid(self, rows, pattern):
        path = self._write(rows)
        with self.assertRaisesRegex(LandCampaignInputError, pattern):
            validate_land_campaign_input(path, expected_land_fraction=0.02)

    def test_valid_table_returns_reproducible_summary(self):
        path = self._write([self._row(1.0, 2.0), self._row(-1.0, -2.0)])

        summary = validate_land_campaign_input(
            path, expected_land_fraction=0.02
        )

        coordinate_payload = "-1.00000000,-2.00000000\n1.00000000,2.00000000"
        self.assertEqual(summary["status"], "passed")
        self.assertEqual(summary["cell_anchor"], "center")
        self.assertEqual(summary["row_count"], 2)
        self.assertEqual(summary["land_competition_fraction"], 0.02)
        self.assertEqual(
            summary["file_sha256"], hashlib.sha256(path.read_bytes()).hexdigest()
        )
        self.assertEqual(
            summary["coordinate_sha256"],
            hashlib.sha256(coordinate_payload.encode()).hexdigest(),
        )

    def test_cli_emits_json_summary(self):
        path = self._write([self._row()])
        completed = subprocess.run(
            [
                sys.executable,
                "-m",
                "arc.qa",
                "land",
                str(path),
                "--expected-land-fraction",
                "0.02",
            ],
            check=False,
            capture_output=True,
            text=True,
            cwd=Path(__file__).parents[1],
        )

        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertEqual(json.loads(completed.stdout)["status"], "passed")

    def test_rejects_empty_table_and_missing_required_column(self):
        empty_path = self._write([])
        with self.assertRaisesRegex(LandCampaignInputError, "no data rows"):
            validate_land_campaign_input(
                empty_path, expected_land_fraction=0.02
            )

        row = self._row()
        del row["max_capacity_mw"]
        self._assert_invalid([row], "missing required columns: max_capacity_mw")

    def test_rejects_duplicate_or_invalid_coordinates(self):
        self._assert_invalid(
            [self._row(1.0, 2.0), self._row(1.0 + 1e-10, 2.0)],
            "Duplicate coordinate",
        )
        for column, value, pattern in (
            ("latitude", "nan", "non-finite"),
            ("longitude", "inf", "non-finite"),
            ("latitude", 90.1, "latitude outside"),
            ("longitude", -180.1, "longitude outside"),
        ):
            with self.subTest(column=column, value=value):
                row = self._row()
                row[column] = value
                self._assert_invalid([row], pattern)

    def test_rejects_nonuniform_or_unexpected_fraction(self):
        second = self._row(1.0, 1.0)
        second["land_competition_fraction"] = 0.03
        self._assert_invalid(
            [self._row(), second], "land_competition_fraction is not uniform"
        )

        path = self._write([self._row()])
        with self.assertRaisesRegex(
            LandCampaignInputError, "Expected land_competition_fraction"
        ):
            validate_land_campaign_input(path, expected_land_fraction=0.50)

    def test_rejects_wrong_union_method_or_version(self):
        for column, value in (
            ("renewable_union_method", "exact_geometric_union"),
            ("renewable_union_method_version", "v2"),
        ):
            with self.subTest(column=column):
                row = self._row()
                row[column] = value
                self._assert_invalid([row], f"expected {column}")

    def test_rejects_tables_that_are_not_centred_builds(self):
        missing = self._row()
        del missing["cell_anchor"]
        self._assert_invalid([missing], "missing required columns: cell_anchor")
        for anchor in ("southwest", ""):
            with self.subTest(anchor=anchor):
                row = self._row()
                row["cell_anchor"] = anchor
                self._assert_invalid([row], "expected cell_anchor='center'")

    def test_rejects_nonfinite_or_negative_area_and_capacity(self):
        for column, value, pattern in (
            ("wind_onshore_area_km2", -1.0, "must be nonnegative"),
            ("max_capacity_mw", "nan", "non-finite"),
            ("max_power_solar_mw", "inf", "non-finite"),
        ):
            with self.subTest(column=column):
                row = self._row()
                row[column] = value
                self._assert_invalid([row], pattern)

    def test_rejects_both_divergent_generic_aliases(self):
        for column, value in (
            ("renewable_union_area_km2", 7.1),
            ("renewable_union_availability", 0.08),
        ):
            with self.subTest(column=column):
                row = self._row()
                row[column] = value
                self._assert_invalid([row], "compatibility alias")

    def test_rejects_availability_outside_unit_interval(self):
        for column, value in (
            ("availability", -0.01),
            ("wind_availability", 1.01),
        ):
            with self.subTest(column=column):
                row = self._row()
                row[column] = value
                self._assert_invalid([row], "outside \\[0, 1\\]")

    def test_rejects_union_outside_physical_bounds(self):
        for union, pattern in (
            (5.9, "below max"),
            (10.1, "exceeds wind_onshore\\+solar"),
        ):
            with self.subTest(union=union):
                row = self._row()
                row["renewable_union_area_km2_classwise_nested_v1"] = union
                row["renewable_union_area_km2"] = union
                self._assert_invalid([row], pattern)

    def test_accepts_roundoff_at_physical_union_bounds(self):
        for union in (6.0 - 5e-9, 10.0 + 5e-9):
            with self.subTest(union=union):
                row = self._row()
                row["renewable_union_area_km2_classwise_nested_v1"] = union
                row["renewable_union_area_km2"] = union
                path = self._write([row], name=f"land-{union}.csv")

                summary = validate_land_campaign_input(
                    path, expected_land_fraction=0.02
                )

                self.assertEqual(summary["status"], "passed")


if __name__ == "__main__":
    unittest.main()


def test_site_cost_results_must_carry_the_table_values(tmp_path):
    import pytest

    from arc.qa import _check_site_costs

    table = tmp_path / "site_costs.csv"
    pd.DataFrame({
        "latitude": [-23.0, -23.0], "longitude": [117.0, 151.0],
        "equipment_cost_multiplier": [1.05, 1.01], "installation_multiplier": [1.34, 1.09],
        "slope_multiplier": [1.2, 1.0], "depth_multiplier": [1.0, 1.0], "labour_multiplier": [1.0, 1.0],
        "water_cost_usd_per_m3": [7.2, 2.28], "land_cost_usd_per_km2_year": [0.0, 0.0],
    }).to_csv(table, index=False)
    expectations = {"site_costs_csv": table, "site_costs_sha256": None}
    good = pd.DataFrame({"latitude": [-23.0, -23.0], "longitude": [117.0, 151.0],
                         "equipment_cost_multiplier": [1.05, 1.01], "build_cost_multiplier": [1.5, 1.09],
                         "water_cost_usd_per_m3": [7.2, 2.28]})
    assert _check_site_costs(good, expectations) == {"site_cost_cells_checked": 2}
    for column, value, message in (("equipment_cost_multiplier", 1.0, "equipment"), ("build_cost_multiplier", 1.7, "range"),
                                   ("water_cost_usd_per_m3", 2.28, "water")):
        bad = good.copy()
        bad.loc[0, column] = value
        with pytest.raises(ValueError, match=message):
            _check_site_costs(bad, expectations)

