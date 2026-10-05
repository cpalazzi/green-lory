#!/usr/bin/env python3
"""Verify that a campaign manifest's source inputs have not changed."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


PLANT_FILES = (
    "network.csv",
    "buses.csv",
    "generators.csv",
    "links.csv",
    "loads.csv",
    "stores.csv",
)


class ManifestInputError(ValueError):
    """Raised when a manifest is malformed or a recorded input has changed."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _mapping(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ManifestInputError(f"{label} must be a JSON object")
    return value


def _file_record_path(record: dict[str, Any], label: str) -> Path:
    value = record.get("resolved_path")
    if not isinstance(value, str) or not value:
        raise ManifestInputError(f"{label}.resolved_path must be a non-empty string")
    return Path(value)


def _verify_file_record(record_value: Any, label: str) -> Path:
    record = _mapping(record_value, label)
    path = _file_record_path(record, label)
    if not path.exists():
        raise ManifestInputError(f"{label} is missing: {path}")
    if not path.is_file():
        raise ManifestInputError(f"{label} is not a regular file: {path}")

    expected_size = record.get("size_bytes")
    if not isinstance(expected_size, int) or isinstance(expected_size, bool):
        raise ManifestInputError(f"{label}.size_bytes must be an integer")
    actual_size = path.stat().st_size
    if actual_size != expected_size:
        raise ManifestInputError(
            f"{label} size changed: expected {expected_size} bytes, "
            f"found {actual_size} bytes at {path}"
        )

    expected_sha256 = record.get("sha256")
    if not isinstance(expected_sha256, str) or len(expected_sha256) != 64:
        raise ManifestInputError(f"{label}.sha256 must be a 64-character string")
    actual_sha256 = _sha256(path)
    if actual_sha256 != expected_sha256:
        raise ManifestInputError(
            f"{label} SHA-256 changed: expected {expected_sha256}, "
            f"found {actual_sha256} at {path}"
        )
    return path


def _weather_listing(path: Path) -> list[dict[str, Any]]:
    files = []
    for item in sorted(path.glob("*.nc")):
        stat = item.stat()
        files.append(
            {
                "name": item.name,
                "size_bytes": stat.st_size,
                "mtime_ns": stat.st_mtime_ns,
            }
        )
    return files


def _listing_sha256(files: list[dict[str, Any]]) -> str:
    payload = json.dumps(files, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )
    return hashlib.sha256(payload).hexdigest()


def _verify_weather(record_value: Any) -> int:
    label = "inputs.weather"
    record = _mapping(record_value, label)
    path = _file_record_path(record, label)
    if not path.exists():
        raise ManifestInputError(f"{label} directory is missing: {path}")
    if not path.is_dir():
        raise ManifestInputError(f"{label} is not a directory: {path}")

    expected_count = record.get("file_count")
    if not isinstance(expected_count, int) or isinstance(expected_count, bool):
        raise ManifestInputError(f"{label}.file_count must be an integer")
    expected_sha256 = record.get("listing_sha256")
    if not isinstance(expected_sha256, str) or len(expected_sha256) != 64:
        raise ManifestInputError(
            f"{label}.listing_sha256 must be a 64-character string"
        )

    files = _weather_listing(path)
    actual_sha256 = _listing_sha256(files)
    if len(files) != expected_count or actual_sha256 != expected_sha256:
        raise ManifestInputError(
            f"{label} listing changed: expected {expected_count} NetCDF files "
            f"with listing SHA-256 {expected_sha256}, found {len(files)} files "
            f"with listing SHA-256 {actual_sha256} at {path}; identity uses "
            "filename, size_bytes, and mtime_ns"
        )
    return len(files)


def verify_manifest_inputs(manifest_path: Path) -> dict[str, int]:
    """Verify all source-input identities recorded in ``manifest_path``."""

    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ManifestInputError(f"manifest is missing: {manifest_path}") from exc
    except json.JSONDecodeError as exc:
        raise ManifestInputError(
            f"manifest is not valid JSON: {manifest_path}: {exc}"
        ) from exc
    manifest = _mapping(manifest, "manifest")
    inputs = _mapping(manifest.get("inputs"), "inputs")

    verified_file_count = 0

    tech = _mapping(inputs.get("tech_yaml"), "inputs.tech_yaml")
    chain = tech.get("extends_chain")
    if not isinstance(chain, list) or not chain:
        raise ManifestInputError(
            "inputs.tech_yaml.extends_chain must be a non-empty array"
        )
    for index, record in enumerate(chain):
        _verify_file_record(record, f"inputs.tech_yaml.extends_chain[{index}]")
        verified_file_count += 1

    plant = _mapping(inputs.get("plant_bundle"), "inputs.plant_bundle")
    plant_files = _mapping(plant.get("files"), "inputs.plant_bundle.files")
    missing_records = [name for name in PLANT_FILES if name not in plant_files]
    unexpected_records = [name for name in plant_files if name not in PLANT_FILES]
    if missing_records or unexpected_records:
        details = []
        if missing_records:
            details.append(f"missing records: {', '.join(missing_records)}")
        if unexpected_records:
            details.append(f"unexpected records: {', '.join(unexpected_records)}")
        raise ManifestInputError(
            "inputs.plant_bundle.files does not match the manifest schema ("
            + "; ".join(details)
            + ")"
        )
    for name in PLANT_FILES:
        _verify_file_record(
            plant_files[name], f"inputs.plant_bundle.files[{name!r}]"
        )
        verified_file_count += 1

    for name in ("override_csv", "land_csv"):
        _verify_file_record(inputs.get(name), f"inputs.{name}")
        verified_file_count += 1

    if "locations_csv" in inputs:
        _verify_file_record(inputs["locations_csv"], "inputs.locations_csv")
        verified_file_count += 1

    source = manifest.get("source", {})
    if source is not None:
        source = _mapping(source, "source")
        if "release_inventory" in source:
            _verify_file_record(
                source["release_inventory"], "source.release_inventory"
            )
            verified_file_count += 1

    weather_file_count = _verify_weather(inputs.get("weather"))
    return {
        "byte_hashed_file_count": verified_file_count,
        "weather_file_count": weather_file_count,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    try:
        result = verify_manifest_inputs(Path(args.manifest))
    except (ManifestInputError, OSError) as exc:
        raise SystemExit(f"Campaign input verification failed: {exc}") from exc
    print(
        "Campaign inputs verified: "
        f"{result['byte_hashed_file_count']} byte-hashed files; "
        f"{result['weather_file_count']} weather files"
    )


if __name__ == "__main__":
    main()
