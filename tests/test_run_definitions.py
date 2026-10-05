import shlex
import tempfile
import unittest
from pathlib import Path

import yaml

from arc.run_definitions import (
    RUNS_FILE,
    WRAPPER_VARIABLES,
    RunDefinitionError,
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
        self.assertEqual(values["PLANT_DIR"], "basic_ammonia_plant_2050")
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


if __name__ == "__main__":
    unittest.main()
