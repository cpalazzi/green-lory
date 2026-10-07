import csv
import shlex
import tempfile
import unittest
from pathlib import Path

import yaml

from model.runs import (
    RUN_STORE_FILE,
    RUNS_FILE,
    WRAPPER_VARIABLES,
    RunDefinitionError,
    describe_entry,
    load_networks,
    load_runs,
    markdown_table,
    shell_assignments,
)


def _assignments(text):
    return dict(line.split("=", 1) for line in text.splitlines())


class RunDefinitionsTest(unittest.TestCase):
    def test_repository_runs_load_and_name_both_green_lory_roles(self):
        runs = load_runs(RUNS_FILE)
        roles = {run["role"] for run in runs.values() if run["runner"] == "green_lory"}
        self.assertIn("reconciling", roles)
        self.assertIn("improved", roles)
        self.assertIn("rep_way2050_flat_amelired_4h_tracking_nominal_h2", markdown_table(runs))

    def test_key_run_resolves_to_the_settings_it_was_submitted_with(self):
        values = {
            key: shlex.split(value)[0]
            for key, value in _assignments(
                shell_assignments("gl_dea2050_wacc5_bflat_wflat_land20c_fixed")
            ).items()
        }
        self.assertEqual(set(values), set(WRAPPER_VARIABLES.values()))
        self.assertEqual(values["TECH_YAML"], "inputs/tech_config_ammonia_plant_2050_dea_colocated.yaml")
        self.assertEqual(values["PLANT_DIR"], "model/plants/ammonia_plant_1mt")
        self.assertEqual(values["TIME_STEP"], "1.0")
        self.assertEqual(values["INCLUDE_SITE_COSTS"], "1")
        self.assertEqual(values["FLAT_WATER_MODEL_PER_M3"], "2.0")
        self.assertEqual(values["EXPECTED_LAND_COMPETITION_FRACTION"], "0.20")
        self.assertEqual(values["LAND_ALLOCATION"], "colocated")

    def test_reconciling_run_keeps_the_legacy_accounting(self):
        values = _assignments(shell_assignments("rep_way2050_flat_amelired_4h_tracking_nominal_h2"))
        self.assertEqual(values["TIME_STEP"], "4.0")
        self.assertEqual(values["TEMPORAL_ACCOUNTING_MODE"], "legacy_scaled")
        self.assertEqual(values["RAMP_LIMIT_BASIS"], "legacy_per_snapshot")
        self.assertEqual(values["INCLUDE_SITE_COSTS"], "0")
        self.assertEqual(values["FLAT_WATER_MODEL_PER_M3"], "1.751012")

    def test_unknown_and_non_wrapper_runs_are_refused(self):
        with self.assertRaisesRegex(RunDefinitionError, "Unknown run id"):
            shell_assignments("oat_way2050_flat_amelired_1h_tracking_nominal_h2")
        with self.assertRaisesRegex(RunDefinitionError, "runner 'derived'"):
            shell_assignments("gl_dea2050_wacc5_bflat_wflat_land2c_fixed")

    def test_incomplete_or_invalid_green_lory_runs_are_rejected(self):
        run = load_runs(RUNS_FILE)["gl_dea2050_wacc5_bflat_wflat_land20c_fixed"]
        for change, pattern in (
            ({"land_allocation": "shared"}, "land_allocation"),
            ({"site_costs_in_headline": "yes"}, "true or false"),
            ({"time_step_h": 7}, "divide 8760"),
        ):
            with self.subTest(change=change), tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp) / "runs.yaml"
                path.write_text(yaml.safe_dump({"runs": {"bad": {**run, **change}}}))
                with self.assertRaisesRegex(RunDefinitionError, pattern):
                    load_runs(path)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "runs.yaml"
            incomplete = {key: value for key, value in run.items() if key != "tech_yaml"}
            path.write_text(yaml.safe_dump({"runs": {"bad": incomplete}}))
            with self.assertRaisesRegex(RunDefinitionError, "missing tech_yaml"):
                load_runs(path)

    def test_networks_name_their_supplier_tables_and_september_letters(self):
        runs = load_runs(RUNS_FILE)
        networks = load_networks(RUNS_FILE)
        for network in networks.values():
            self.assertTrue(network["supplier_run"] == "archived" or network["supplier_run"] in runs)
        letters = {network.get("report_name"): network_id for network_id, network in networks.items()}
        self.assertEqual(letters["A2"], "gpo_archived_iso1000_1mt")
        self.assertEqual(letters["E"], "gpo_ll_xcost45_ameli_bflat_wnone_landarch_track_4h_iso1000_1mt")
        network_d = networks[letters["D"]]
        self.assertEqual(network_d["supplier_run"], "rep_way2050_flat_amelired_4h_tracking_nominal_h2_legacyrule")
        self.assertEqual(network_d["routes"], "iso1000")  # filled in from network_defaults
        self.assertEqual(networks["gpo_archived_replay"]["routes"], "archived pipelines")  # overrides the default
        text = describe_entry("d", runs, networks)
        self.assertIn("derived from rep_way2050_flat_amelired_4h_tracking_nominal_h2 (reconciling)", text)
        with self.assertRaisesRegex(RunDefinitionError, "not a run id, network id or report name"):
            describe_entry("Z", runs, networks)

    def test_every_defined_run_and_network_has_an_execution_record(self):
        # a new definition gets a run_store.csv row (status planned) before it is submitted
        with RUN_STORE_FILE.open(newline="") as handle:
            recorded = {row["run_id"] for row in csv.DictReader(handle)}
        defined = set(load_runs(RUNS_FILE)) | set(load_networks(RUNS_FILE))
        self.assertEqual(sorted(defined - recorded), [])

    def test_inconsistent_network_and_derived_definitions_are_rejected(self):
        run = load_runs(RUNS_FILE)["gl_dea2050_wacc5_bflat_wflat_land20c_fixed"]
        derived = {"runner": "derived", "role": "sensitivity", "derived_from": "base"}
        for document, pattern in (
            ({"runs": {"base": run, "d": {**derived, "derived_from": "missing"}}}, "derived_from"),
            ({"runs": {"base": run, "d": {**derived, "derived_from": "d"}}}, "forms a cycle"),
            ({"runs": {"base": run}, "networks": {"n": {"supplier_run": "missing"}}}, "supplier_run 'missing'"),
            ({"runs": {"base": run}, "networks": {"base": {"supplier_run": "archived"}}}, "both a run and a network"),
            (
                {"runs": {"base": run, "d": derived}, "networks": {
                    "n1": {"supplier_run": "base", "report_name": "D"},
                    "n2": {"supplier_run": "d", "report_name": "d"},
                }},
                "used twice",
            ),
        ):
            with self.subTest(pattern=pattern), tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp) / "runs.yaml"
                path.write_text(yaml.safe_dump(document))
                with self.assertRaisesRegex(RunDefinitionError, pattern):
                    load_networks(path)


if __name__ == "__main__":
    unittest.main()
