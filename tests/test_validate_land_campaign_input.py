import csv
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from arc.validate_land_campaign_input import (
    EXPECTED_RENEWABLE_UNION_METHOD,
    EXPECTED_RENEWABLE_UNION_METHOD_VERSION,
    LandCampaignInputError,
    validate_land_campaign_input,
)


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
        script = Path(__file__).parents[1] / "arc" / "validate_land_campaign_input.py"

        completed = subprocess.run(
            [
                sys.executable,
                str(script),
                str(path),
                "--expected-land-fraction",
                "0.02",
            ],
            check=False,
            capture_output=True,
            text=True,
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
