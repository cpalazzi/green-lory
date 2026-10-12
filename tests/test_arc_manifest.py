"""arc/manifest.py: verifying a manifest's inputs, the flat cost scopes of the key runs and the
site-cost scopes of S1/S2."""
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from arc.qa import _flat_cost_scope_expectations

from arc.manifest import (
    build_cost_scope,
    build_flat_amelired_cost_scope,
    flat_finance_token,
    ManifestInputError,
    PLANT_FILES,
    main,
    verify_manifest_inputs,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _file_record(path: Path) -> dict:
    resolved = path.resolve()
    return {
        "path": str(path),
        "resolved_path": str(resolved),
        "size_bytes": resolved.stat().st_size,
        "sha256": _sha256(resolved),
    }


def _weather_record(path: Path) -> dict:
    resolved = path.resolve()
    files = []
    for item in sorted(resolved.glob("*.nc")):
        stat = item.stat()
        files.append(
            {
                "name": item.name,
                "size_bytes": stat.st_size,
                "mtime_ns": stat.st_mtime_ns,
            }
        )
    listing = json.dumps(files, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )
    return {
        "path": str(path),
        "resolved_path": str(resolved),
        "file_count": len(files),
        "listing_sha256": hashlib.sha256(listing).hexdigest(),
        "files": files,
    }


class VerifyCampaignManifestInputsTest(unittest.TestCase):
    def _fixture(self, root: Path) -> tuple[Path, dict[str, Path]]:
        tech_dir = root / "tech"
        tech_dir.mkdir()
        tech_base = tech_dir / "base.yaml"
        tech_child = tech_dir / "child.yaml"
        tech_base.write_text("cost: 1\n", encoding="utf-8")
        tech_child.write_text("extends: base.yaml\n", encoding="utf-8")

        plant_dir = root / "plant"
        plant_dir.mkdir()
        plant_paths = {}
        for index, name in enumerate(PLANT_FILES):
            path = plant_dir / name
            path.write_text(f"column\n{index}\n", encoding="utf-8")
            plant_paths[name] = path

        override = root / "override.csv"
        land = root / "land.csv"
        locations = root / "locations.csv"
        override.write_text("country,value\nAA,1\n", encoding="utf-8")
        land.write_text("latitude,longitude\n1,2\n", encoding="utf-8")
        locations.write_text("latitude,longitude\n3,4\n", encoding="utf-8")

        weather_dir = root / "weather"
        weather_dir.mkdir()
        (weather_dir / "a.nc").write_bytes(b"weather-a")
        (weather_dir / "b.nc").write_bytes(b"weather-b")
        (weather_dir / "ignored.txt").write_text("ignored", encoding="utf-8")

        manifest = {
            "inputs": {
                "tech_yaml": {
                    "extends_chain": [
                        _file_record(tech_base),
                        _file_record(tech_child),
                    ]
                },
                "plant_bundle": {
                    "files": {
                        name: _file_record(path)
                        for name, path in plant_paths.items()
                    }
                },
                "finance_csv": _file_record(override),
                "land_csv": _file_record(land),
                "locations_csv": _file_record(locations),
                "weather": _weather_record(weather_dir),
            }
        }
        manifest_path = root / "manifest.json"
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        paths = {
            "override": override,
            "weather": weather_dir,
            "plant_buses": plant_paths["buses.csv"],
        }
        return manifest_path, paths

    def test_accepts_unchanged_manifest_inputs(self):
        with tempfile.TemporaryDirectory() as temporary:
            manifest_path, _ = self._fixture(Path(temporary))

            result = verify_manifest_inputs(manifest_path)

            self.assertEqual(result["byte_hashed_file_count"], 11)
            self.assertEqual(result["weather_file_count"], 2)

    def test_cli_prints_concise_success(self):
        with tempfile.TemporaryDirectory() as temporary:
            manifest_path, _ = self._fixture(Path(temporary))
            output = io.StringIO()

            with patch("sys.stdout", output):
                main(["verify", "--manifest", str(manifest_path)])

            self.assertEqual(
                output.getvalue(),
                "Campaign inputs verified: 11 byte-hashed files; "
                "2 weather files\n",
            )

    def test_rejects_same_size_byte_mutation(self):
        with tempfile.TemporaryDirectory() as temporary:
            manifest_path, paths = self._fixture(Path(temporary))
            paths["override"].write_text(
                "country,value\nAA,2\n", encoding="utf-8"
            )

            with self.assertRaisesRegex(
                ManifestInputError, "inputs.finance_csv SHA-256 changed"
            ):
                verify_manifest_inputs(manifest_path)

    def test_rejects_missing_plant_file(self):
        with tempfile.TemporaryDirectory() as temporary:
            manifest_path, paths = self._fixture(Path(temporary))
            paths["plant_buses"].unlink()

            with self.assertRaisesRegex(
                ManifestInputError, r"plant_bundle.*buses.csv.*is missing"
            ):
                verify_manifest_inputs(manifest_path)

    def test_rejects_weather_listing_mutation(self):
        with tempfile.TemporaryDirectory() as temporary:
            manifest_path, paths = self._fixture(Path(temporary))
            (paths["weather"] / "added.nc").write_bytes(b"new-weather")

            with self.assertRaisesRegex(
                ManifestInputError, "inputs.weather listing changed"
            ):
                verify_manifest_inputs(manifest_path)


# --- flat cost scopes (key runs of 23 September 2026) ----------------------------------------

ROOT = Path(__file__).resolve().parents[1]
DEA_YAML = ROOT / "inputs/tech/tech_config_ammonia_plant_2050_dea_colocated.yaml"


def _finance_csv(path: Path, rates) -> Path:
    """A small finance table: one row per (cell, rate) for two technologies."""
    rows = [f"{-23 + i},117,{tech},{rate}" for i, rate in enumerate(rates) for tech in ("solar", "wind")]
    path.write_text("lat,lon,tech,interest_rate\n" + "\n".join(rows) + "\n")
    return path


def _manifest(scope_id, mode, rate=0.05, source="finance_csv"):
    finance = {"mode": mode, "source": source, "override_column": "interest_rate"}
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
        "inputs": ({"interest_rate": rate} if source == "interest_rate"
                   else {"finance_csv": {"columns": ["lat", "lon", "tech", "interest_rate"]}}),
    }


class QaGateTest(unittest.TestCase):
    def test_uniform_and_ameli_scopes_are_accepted(self):
        for scope_id, mode in [("flat_wacc5_uniform_baseline_water_no_land_rent", "uniform_wacc"), ("flat_amelired_uniform_baseline_water_no_land_rent", "ameli_reduced_wacc")]:
            out = _flat_cost_scope_expectations(_manifest(scope_id, mode), {"include_site_costs": True})
            self.assertEqual(out["water_cost_usd_per_m3"], 2.0, scope_id)

    def test_one_interest_rate_is_accepted_when_recorded(self):
        scope_id = "flat_wacc5_uniform_baseline_water_no_land_rent"
        manifest = _manifest(scope_id, "uniform_wacc", source="interest_rate")
        self.assertEqual(_flat_cost_scope_expectations(manifest, {"include_site_costs": True})["water_cost_usd_per_m3"], 2.0)
        del manifest["inputs"]["interest_rate"]
        with self.assertRaisesRegex(ValueError, "inputs.interest_rate"):
            _flat_cost_scope_expectations(manifest, {"include_site_costs": True})

    def test_finance_mode_must_match_the_scope_id(self):
        for scope_id, mode in [("flat_wacc5_uniform_baseline_water_no_land_rent", "ameli_reduced_wacc"), ("flat_amelired_uniform_baseline_water_no_land_rent", "uniform_wacc"), ("flat_spatial_uniform_baseline_water_no_land_rent", "uniform_wacc")]:
            with self.assertRaises(ValueError, msg=scope_id):
                _flat_cost_scope_expectations(_manifest(scope_id, mode), {"include_site_costs": True})


class ManifestBuilderTest(unittest.TestCase):
    def test_tokens(self):
        self.assertEqual(flat_finance_token("flat_wacc5_uniform_baseline_water_no_land_rent"), "wacc5")
        self.assertEqual(flat_finance_token("flat_amelired_uniform_baseline_water_no_land_rent"), "amelired")

    def test_the_rates_must_agree_with_the_declared_finance(self):
        wacc5 = "flat_wacc5_uniform_baseline_water_no_land_rent"
        scope = build_flat_amelired_cost_scope(tech_yaml=DEA_YAML, interest_rate=0.05, include_site_costs=True,
                                               cost_scope_id=wacc5, flat_water_model_currency_per_m3=2.0)
        self.assertEqual((scope["finance"]["mode"], scope["finance"]["source"]), ("uniform_wacc", "interest_rate"))
        self.assertAlmostEqual(scope["finance"]["rate"], 0.05)
        self.assertAlmostEqual(scope["water"]["expected_result_cost_model_currency_per_m3"], 2.0, places=7)
        self.assertEqual(scope["water"]["model_currency"], "EUR")
        self.assertEqual(scope["id"], wacc5)
        with self.assertRaises(ValueError):  # the water price disagrees with the YAML
            build_flat_amelired_cost_scope(tech_yaml=DEA_YAML, interest_rate=0.05, include_site_costs=True,
                                           cost_scope_id=wacc5, flat_water_model_currency_per_m3=1.75)
        with self.assertRaisesRegex(ValueError, "uniform interest rate of 0.05"):
            build_flat_amelired_cost_scope(tech_yaml=DEA_YAML, interest_rate=0.07, include_site_costs=True, cost_scope_id=wacc5)
        with self.assertRaisesRegex(ValueError, "country-varying"):
            build_flat_amelired_cost_scope(tech_yaml=DEA_YAML, interest_rate=0.05, include_site_costs=True)
        with tempfile.TemporaryDirectory() as tmp:
            uniform = _finance_csv(Path(tmp) / "uniform.csv", [0.05, 0.05])
            countries = _finance_csv(Path(tmp) / "countries.csv", [0.05, 0.08])
            table_scope = build_flat_amelired_cost_scope(tech_yaml=DEA_YAML, finance_csv=uniform, include_site_costs=True, cost_scope_id=wacc5)
            self.assertEqual(table_scope["finance"]["source"], "finance_csv")
            ameli = build_flat_amelired_cost_scope(tech_yaml=DEA_YAML, finance_csv=countries, include_site_costs=True)
            self.assertEqual(ameli["finance"]["mode"], "ameli_reduced_wacc")
            with self.assertRaises(ValueError):
                build_flat_amelired_cost_scope(tech_yaml=DEA_YAML, finance_csv=countries, include_site_costs=True, cost_scope_id=wacc5)
            with self.assertRaisesRegex(ValueError, "not both"):
                build_flat_amelired_cost_scope(tech_yaml=DEA_YAML, finance_csv=uniform, interest_rate=0.05,
                                               include_site_costs=True, cost_scope_id=wacc5)


def _site_cost_table(path: Path, *, source_price=2.284393, land_cost=0.0, slope=1.2, water_at_source=None) -> Path:
    """Two cells as model/site_costs.py writes them (prices rounded to 6 decimals); the second holds
    its own water source, so pays the source price."""
    water_at_source = source_price if water_at_source is None else water_at_source
    path.write_text(
        "latitude,longitude,equipment_cost_multiplier,installation_multiplier,slope_multiplier,depth_multiplier,"
        "labour_multiplier,water_source_cost_usd_per_m3,water_cost_usd_per_m3,land_cost_usd_per_km2_year\n"
        f"-23,117,1.05,1.34,{slope},1,1,{source_price},7.2,{land_cost}\n"
        f"-23,151,1.01,1.09,1,1,1,{source_price},{water_at_source},{land_cost}\n"
    )
    return path


class SiteCostScopeTest(unittest.TestCase):
    def test_site_cost_scope_takes_finance_from_the_run_and_costs_from_the_table(self):
        with tempfile.TemporaryDirectory() as tmp:
            table = _site_cost_table(Path(tmp) / "site_costs.csv")
            scope = build_cost_scope(tech_yaml=DEA_YAML, interest_rate=0.05, include_site_costs=True,
                                     cost_scope_id="site_costs_wacc5_no_land_rent",
                                     flat_water_model_currency_per_m3=2.0, site_costs_csv=table)
            self.assertEqual(scope["finance"]["mode"], "uniform_wacc")
            self.assertEqual(scope["spatial_build_and_remoteness"]["mode"], "site_cost_table")
            self.assertEqual(scope["water"]["mode"], "site_cost_table")
            self.assertAlmostEqual(scope["water"]["source_price_model_currency_per_m3"], 2.0, places=6)
            with self.assertRaisesRegex(ValueError, "needs a site-cost table"):
                build_cost_scope(tech_yaml=DEA_YAML, interest_rate=0.05, include_site_costs=True,
                                 cost_scope_id="site_costs_wacc5_no_land_rent")
            with self.assertRaisesRegex(ValueError, "flat scope"):
                build_cost_scope(tech_yaml=DEA_YAML, interest_rate=0.05, include_site_costs=True,
                                 cost_scope_id="flat_wacc5_uniform_baseline_water_no_land_rent", site_costs_csv=table)

    def test_site_cost_tables_must_match_the_yaml_and_carry_no_land_rent(self):
        for kwargs, pattern in (({"source_price": 2.0}, "not the YAML baseline"), ({"land_cost": 5.0}, "zero land cost"),
                                ({"slope": 0.9}, "at least 1"), ({"water_at_source": 2.2}, "below the source price")):
            with self.subTest(kwargs=kwargs), tempfile.TemporaryDirectory() as tmp:
                table = _site_cost_table(Path(tmp) / "site_costs.csv", **kwargs)
                with self.assertRaisesRegex(ValueError, pattern):
                    build_cost_scope(tech_yaml=DEA_YAML, interest_rate=0.05, include_site_costs=True,
                                     cost_scope_id="site_costs_wacc5_no_land_rent", site_costs_csv=table)


if __name__ == "__main__":
    unittest.main()
