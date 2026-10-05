import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from arc.verify_campaign_manifest_inputs import (
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
                "override_csv": _file_record(override),
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

            with patch("sys.argv", ["verify", "--manifest", str(manifest_path)]):
                with patch("sys.stdout", output):
                    main()

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
                ManifestInputError, "inputs.override_csv SHA-256 changed"
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


if __name__ == "__main__":
    unittest.main()
