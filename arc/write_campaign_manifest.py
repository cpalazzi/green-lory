#!/usr/bin/env python3
"""Write or update an immutable Green Lory campaign-run manifest."""

from __future__ import annotations

import argparse
import re
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import yaml


PLANT_FILES = (
    "network.csv",
    "buses.csv",
    "generators.csv",
    "links.csv",
    "loads.csv",
    "stores.csv",
)


def _zero_or_one(value: str) -> bool:
    if value == "1":
        return True
    if value == "0":
        return False
    raise argparse.ArgumentTypeError("expected 0 or 1")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--campaign-id", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--scenario-id", required=True)
    parser.add_argument("--scenario-class", required=True)
    parser.add_argument("--stage", required=True)
    parser.add_argument("--description", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--tech-yaml", required=True)
    parser.add_argument("--plant-dir", required=True)
    parser.add_argument("--override-csv", required=True)
    parser.add_argument("--land-csv", required=True)
    parser.add_argument("--weather-dir", required=True)
    parser.add_argument("--locations-csv")
    parser.add_argument("--time-step-hours", required=True, type=float)
    parser.add_argument("--max-snapshots", type=int)
    parser.add_argument("--simulated-hours", required=True, type=float)
    parser.add_argument("--expected-full-year", required=True, type=_zero_or_one)
    parser.add_argument("--num-workers", required=True, type=int)
    parser.add_argument("--threads-per-worker", required=True, type=int)
    parser.add_argument("--slurm-cpus", type=int, default=48)
    parser.add_argument("--slurm-memory", default="370G")
    parser.add_argument("--mail-user", default="carlo.palazzi@eng.ox.ac.uk")
    parser.add_argument("--mail-type", default="BEGIN,END,FAIL")
    parser.add_argument("--ensure-feasibility", required=True, type=_zero_or_one)
    parser.add_argument("--land-constraint", choices=("after_solve", "in_solve"), required=True)
    parser.add_argument("--capacity-rule", choices=("scaled_reference_design",), required=True)
    parser.add_argument(
        "--land-allocation",
        choices=("colocated", "exclusive"),
        required=True,
    )
    parser.add_argument(
        "--allow-conservative-union-fallback",
        required=True,
        type=_zero_or_one,
    )
    parser.add_argument(
        "--temporal-accounting-mode",
        choices=("legacy_scaled", "snapshot_weighted"),
        required=True,
    )
    parser.add_argument(
        "--ramp-limit-basis",
        choices=("legacy_per_snapshot", "per_hour"),
        required=True,
    )
    parser.add_argument("--include-site-costs", required=True, type=_zero_or_one)
    parser.add_argument(
        "--cost-scope",
        type=_flat_cost_scope_id,
        required=True,
        help=(
            "flat_<finance>_uniform_baseline_water_no_land_rent with finance 'amelired' "
            "(Ameli reduced WACC by country) or 'waccN' (uniform N percent)."
        ),
    )
    parser.add_argument(
        "--flat-water-model-currency-per-m3",
        type=float,
        default=None,
        help="Declared uniform baseline water price in the model currency (e.g. 2.0 EUR2020/m3); checked against the tech YAML.",
    )
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--source-diff-sha256", required=True)
    parser.add_argument("--release-source-inventory")
    parser.add_argument("--expected-currency", default="EUR")
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _file_record(path: Path) -> dict[str, Any]:
    resolved = path.expanduser().resolve()
    stat = resolved.stat()
    return {
        "path": str(path),
        "resolved_path": str(resolved),
        "size_bytes": stat.st_size,
        "sha256": _sha256(resolved),
    }


def _release_inventory_record(path: Path) -> dict[str, Any]:
    record = _file_record(path)
    payload = json.loads(path.expanduser().resolve().read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1:
        raise ValueError(
            f"Unsupported release source inventory schema in {path}: "
            f"{payload.get('schema_version')!r}"
        )
    tree_sha256 = payload.get("tree_sha256")
    entry_count = payload.get("entry_count")
    if not isinstance(tree_sha256, str) or len(tree_sha256) != 64:
        raise ValueError(f"Invalid tree_sha256 in release source inventory: {path}")
    if not isinstance(entry_count, int) or entry_count <= 0:
        raise ValueError(f"Invalid entry_count in release source inventory: {path}")
    record.update(
        {
            "tree_sha256": tree_sha256,
            "entry_count": entry_count,
        }
    )
    return record


def _csv_columns(path: Path) -> list[str]:
    with path.expanduser().resolve().open(newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        try:
            columns = next(reader)
        except StopIteration as exc:
            raise ValueError(f"CSV has no header: {path}") from exc
    normalized = [str(column).strip() for column in columns]
    if not normalized or any(not column for column in normalized):
        raise ValueError(f"CSV has an empty column name: {path}")
    if len(normalized) != len(set(normalized)):
        raise ValueError(f"CSV has duplicate column names: {path}")
    return normalized


def _coordinate_columns(columns: list[str], path: Path) -> tuple[str, str]:
    lower = {column.lower(): column for column in columns}
    latitude = lower.get("lat") or lower.get("latitude")
    longitude = lower.get("lon") or lower.get("longitude")
    if latitude is None or longitude is None:
        raise ValueError(f"CSV must contain latitude/longitude coordinates: {path}")
    return latitude, longitude


def _coordinate(value: str, field: str, path: Path, line_number: int) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"Invalid {field} coordinate at {path}:{line_number}: {value!r}"
        ) from exc
    if not math.isfinite(number):
        raise ValueError(
            f"Non-finite {field} coordinate at {path}:{line_number}: {value!r}"
        )
    return round(number, 4)


def _csv_coordinate_set(
    path: Path,
    *,
    active_only: bool = False,
) -> set[tuple[float, float]]:
    resolved = path.expanduser().resolve()
    with resolved.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        columns = [str(column).strip() for column in (reader.fieldnames or [])]
        latitude, longitude = _coordinate_columns(columns, path)
        lower = {column.lower(): column for column in columns}
        activity_column = lower.get("max_capacity_mw") or lower.get("availability")
        if active_only and activity_column is None:
            raise ValueError(
                f"Active land coverage requires max_capacity_mw or availability: {path}"
            )

        coordinates: set[tuple[float, float]] = set()
        for line_number, row in enumerate(reader, start=2):
            if active_only:
                try:
                    active_value = float(row[activity_column])
                except (TypeError, ValueError) as exc:
                    raise ValueError(
                        f"Invalid {activity_column} at {path}:{line_number}"
                    ) from exc
                if not math.isfinite(active_value):
                    raise ValueError(
                        f"Non-finite {activity_column} at {path}:{line_number}"
                    )
                if active_value <= 0.0:
                    continue
            coordinates.add(
                (
                    _coordinate(row[latitude], "latitude", path, line_number),
                    _coordinate(row[longitude], "longitude", path, line_number),
                )
            )
    if not coordinates:
        qualifier = "active " if active_only else ""
        raise ValueError(f"CSV contains no {qualifier}coordinates: {path}")
    return coordinates


def validate_override_coordinate_coverage(
    *,
    override_csv: Path,
    land_csv: Path,
    locations_csv: Path | None,
) -> dict[str, Any]:
    """Require an override coordinate for every location that will be run."""

    override_coordinates = _csv_coordinate_set(override_csv)
    if locations_csv is not None:
        expected_coordinates = _csv_coordinate_set(locations_csv)
        expected_source = "explicit_locations_csv"
        expected_path = locations_csv
    else:
        expected_coordinates = _csv_coordinate_set(land_csv, active_only=True)
        expected_source = "active_land_csv"
        expected_path = land_csv

    missing = sorted(expected_coordinates - override_coordinates)
    if missing:
        raise ValueError(
            "Override CSV does not cover every run coordinate: "
            f"missing={len(missing)} sample={missing[:10]}, "
            f"override={override_csv}, expected={expected_path}"
        )
    return {
        "expected_source": expected_source,
        "expected_coordinate_count": len(expected_coordinates),
        "override_coordinate_count": len(override_coordinates),
        "missing_coordinate_count": 0,
    }


def _deep_merge(base: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in overlay.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _load_tech_chain(
    path: Path,
    seen: set[Path] | None = None,
) -> tuple[dict[str, Any], list[Path]]:
    resolved = path.expanduser().resolve()
    visited = set() if seen is None else set(seen)
    if resolved in visited:
        raise ValueError(f"Circular tech YAML extends chain at {resolved}")
    visited.add(resolved)
    raw = yaml.safe_load(resolved.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"Tech YAML must contain a mapping: {resolved}")
    parent = raw.pop("extends", None)
    if parent is None:
        return raw, [resolved]
    parent_path = Path(parent)
    if not parent_path.is_absolute():
        parent_path = resolved.parent / parent_path
    parent_data, chain = _load_tech_chain(parent_path, visited)
    return _deep_merge(parent_data, raw), [*chain, resolved]


def _tech_yaml_record(path: Path) -> dict[str, Any]:
    resolved_config, chain = _load_tech_chain(path)
    canonical = json.dumps(
        resolved_config, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return {
        "path": str(path),
        "resolved_path": str(path.expanduser().resolve()),
        "extends_chain": [_file_record(item) for item in chain],
        "resolved_config_sha256": hashlib.sha256(canonical).hexdigest(),
    }


FLAT_COST_SCOPE_PATTERN = re.compile(
    r"^flat_(amelired|wacc\d+(?:p\d+)?)_uniform_baseline_water_no_land_rent$"
)


def _flat_cost_scope_id(value: str) -> str:
    if not FLAT_COST_SCOPE_PATTERN.match(value):
        raise argparse.ArgumentTypeError(f"Unsupported cost scope: {value}")
    return value


def flat_finance_token(cost_scope_id: str) -> str:
    match = FLAT_COST_SCOPE_PATTERN.match(cost_scope_id)
    if match is None:
        raise ValueError(f"Unsupported cost scope: {cost_scope_id}")
    return match.group(1)


def _uniform_override_rate(override_csv: Path) -> float | None:
    """Return the single interest rate when the override is uniform, else None.

    Uses the csv module only: the wrapper validates with the login node's plain python."""
    seen: set[float] = set()
    with override_csv.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        column = next((c for c in reader.fieldnames or [] if c.strip().lower() == "interest_rate"), None)
        if column is None:
            raise ValueError(f"{override_csv} has no interest_rate column")
        for row in reader:
            seen.add(round(float(row[column]), 12))
            if len(seen) > 1:
                return None
    if not seen:
        raise ValueError(f"{override_csv} has no rows")
    return float(next(iter(seen)))


def build_flat_amelired_cost_scope(
    *,
    tech_yaml: Path,
    override_csv: Path,
    include_site_costs: bool,
    cost_scope_id: str = "flat_amelired_uniform_baseline_water_no_land_rent",
    flat_water_model_currency_per_m3: float | None = None,
) -> dict[str, Any]:
    """Structured cost scope for a flat campaign: interest-only override, uniform baseline water,
    no build/remoteness multipliers, no land rent.  Finance is Ameli reduced WACC (``amelired``)
    or a uniform rate (``waccN``, checked against the override CSV)."""
    finance_token = flat_finance_token(cost_scope_id)
    resolved_config, _ = _load_tech_chain(tech_yaml)
    override_columns = _csv_columns(override_csv)
    normalized_columns = [column.lower() for column in override_columns]
    expected_columns = ["lat", "lon", "tech", "interest_rate"]
    if normalized_columns != expected_columns:
        raise ValueError(
            "flat_amelired requires an interest-only override CSV with columns "
            f"{expected_columns}; found {override_columns} in {override_csv}"
        )

    source_currency = str(resolved_config.get("spatial_cost_currency", "")).strip().upper()
    if source_currency != "USD":
        raise ValueError(
            "flat_amelired baseline site costs require spatial_cost_currency=USD "
            f"in the resolved tech config; found {source_currency!r}"
        )
    try:
        water_cost = float(resolved_config["water_cost_baseline_usd_per_m3"])
        water_usage = float(resolved_config["water_usage_m3_per_t_nh3"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            "Resolved tech config must define numeric water_cost_baseline_usd_per_m3 "
            "and water_usage_m3_per_t_nh3"
        ) from exc
    try:
        source_to_model = float(resolved_config["spatial_cost_to_model_currency"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            "Resolved tech config must define numeric spatial_cost_to_model_currency"
        ) from exc
    model_currency = str(resolved_config.get("currency", "")).strip().upper() or "EUR"
    water_cost_model = water_cost * source_to_model
    if flat_water_model_currency_per_m3 is not None and not math.isclose(
        water_cost_model, float(flat_water_model_currency_per_m3), rel_tol=1e-5, abs_tol=1e-8
    ):
        raise ValueError(
            f"Flat baseline water cost is {water_cost_model:.6f} {model_currency}/m3 in the resolved tech "
            f"config ({water_cost} {source_currency}/m3 x {source_to_model}); the scenario declares "
            f"{flat_water_model_currency_per_m3} {model_currency}/m3"
        )
    if not math.isfinite(water_usage) or water_usage <= 0.0:
        raise ValueError("water_usage_m3_per_t_nh3 must be finite and positive")

    uniform_rate = _uniform_override_rate(override_csv)
    if finance_token == "amelired":
        if uniform_rate is not None:
            raise ValueError(
                f"amelired cost scope requires country-varying interest rates; {override_csv} is uniform"
            )
        finance: dict[str, Any] = {
            "mode": "ameli_reduced_wacc",
            "source": "override_csv",
            "override_column": "interest_rate",
        }
    else:
        expected_rate = float(finance_token[4:].replace("p", ".")) / 100.0
        if uniform_rate is None or not math.isclose(uniform_rate, expected_rate, abs_tol=1e-9):
            raise ValueError(
                f"{cost_scope_id} requires a uniform interest rate of {expected_rate} in {override_csv}; "
                f"found {'non-uniform rates' if uniform_rate is None else uniform_rate}"
            )
        finance = {
            "mode": "uniform_wacc",
            "source": "override_csv",
            "override_column": "interest_rate",
            "rate": expected_rate,
        }
    return {
        "schema_version": 1,
        "id": cost_scope_id,
        "finance": finance,
        "spatial_build_and_remoteness": {
            "mode": "none_flat",
            "override_columns_present": [],
            "expected_result_build_cost_multiplier": 1.0,
        },
        "water": {
            "mode": "uniform_yaml_baseline",
            "source": "resolved_tech_config.water_cost_baseline_usd_per_m3",
            "source_currency": source_currency,
            "expected_result_cost_usd_per_m3": water_cost,
            "model_currency": model_currency,
            "source_to_model_currency": source_to_model,
            "expected_result_cost_model_currency_per_m3": water_cost_model,
            "usage_m3_per_t_nh3": water_usage,
            "included_in_headline": bool(include_site_costs),
        },
        "land_rent": {
            "mode": "none_zero",
            "source": "no_land_rent_input",
            "source_currency": source_currency,
            "expected_result_cost_usd_per_km2_year": 0.0,
            "included_in_headline": False,
        },
    }


def _plant_record(path: Path) -> dict[str, Any]:
    records: dict[str, dict[str, Any]] = {}
    aggregate = hashlib.sha256()
    for name in PLANT_FILES:
        record = _file_record(path / name)
        records[name] = record
        aggregate.update(name.encode("utf-8"))
        aggregate.update(record["sha256"].encode("ascii"))
    return {
        "path": str(path),
        "resolved_path": str(path.expanduser().resolve()),
        "aggregate_sha256": aggregate.hexdigest(),
        "files": records,
    }


def _weather_record(path: Path) -> dict[str, Any]:
    resolved = path.expanduser().resolve()
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
    listing = json.dumps(files, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return {
        "path": str(path),
        "resolved_path": str(resolved),
        "file_count": len(files),
        "listing_sha256": hashlib.sha256(listing).hexdigest(),
        "files": files,
        "note": "Weather identity hashes filenames, byte sizes, and mtimes; NetCDF payloads are not re-hashed.",
    }


def _immutable_projection(manifest: dict[str, Any]) -> dict[str, Any]:
    return {
        key: manifest[key]
        for key in (
            "schema_version",
            "campaign_id",
            "run_id",
            "scenario",
            "execution",
            "cost_scope",
            "source",
            "inputs",
            "artifacts",
        )
    }


def main() -> None:
    args = _parse_args()
    now = datetime.now(timezone.utc).isoformat()
    output = Path(args.output)

    tech_yaml_path = Path(args.tech_yaml)
    override_csv_path = Path(args.override_csv)
    land_csv_path = Path(args.land_csv)
    locations_csv_path = Path(args.locations_csv) if args.locations_csv else None
    override_record = _file_record(override_csv_path)
    override_record["columns"] = _csv_columns(override_csv_path)
    override_record["coordinate_coverage"] = validate_override_coordinate_coverage(
        override_csv=override_csv_path,
        land_csv=land_csv_path,
        locations_csv=locations_csv_path,
    )

    inputs: dict[str, Any] = {
        "tech_yaml": _tech_yaml_record(tech_yaml_path),
        "plant_bundle": _plant_record(Path(args.plant_dir)),
        "override_csv": override_record,
        "land_csv": _file_record(land_csv_path),
        "weather": _weather_record(Path(args.weather_dir)),
    }
    if locations_csv_path is not None:
        inputs["locations_csv"] = _file_record(locations_csv_path)

    cost_scope = build_flat_amelired_cost_scope(
        tech_yaml=tech_yaml_path,
        override_csv=override_csv_path,
        include_site_costs=args.include_site_costs,
        cost_scope_id=args.cost_scope,
        flat_water_model_currency_per_m3=args.flat_water_model_currency_per_m3,
    )

    manifest: dict[str, Any] = {
        "schema_version": 1,
        "campaign_id": args.campaign_id,
        "run_id": args.run_id,
        "scenario": {
            "id": args.scenario_id,
            "class": args.scenario_class,
            "description": args.description,
        },
        "execution": {
            "stage": args.stage,
            "time_step_hours": args.time_step_hours,
            "max_snapshots": args.max_snapshots,
            "simulated_hours": args.simulated_hours,
            "expected_full_year": args.expected_full_year,
            "num_workers": args.num_workers,
            "threads_per_worker": args.threads_per_worker,
            "fail_fast": True,
            "ensure_feasibility": args.ensure_feasibility,
            "land_constraint": args.land_constraint,
            "capacity_rule": args.capacity_rule,
            "land_allocation": args.land_allocation,
            "allow_conservative_union_fallback": args.allow_conservative_union_fallback,
            "temporal_accounting_mode": args.temporal_accounting_mode,
            "ramp_limit_basis": args.ramp_limit_basis,
            "include_site_costs": args.include_site_costs,
            "expected_currency": args.expected_currency,
            "slurm_profile": {
                "partition": "short",
                "nodes": 1,
                "tasks": 1,
                "cpus_per_task": args.slurm_cpus,
                "memory": args.slurm_memory,
                "walltime": "12:00:00",
                "mail_user": args.mail_user,
                "mail_type": args.mail_type,
            },
        },
        "cost_scope": cost_scope,
        "source": {
            "git_commit": args.source_commit,
            "dirty_diff_sha256": args.source_diff_sha256,
        },
        "inputs": inputs,
        "artifacts": {
            "run_dir": args.run_dir,
            "manifest": str(output),
        },
        "created_utc": now,
    }
    if args.release_source_inventory:
        manifest["source"]["release_inventory"] = _release_inventory_record(
            Path(args.release_source_inventory)
        )
    immutable_payload = json.dumps(
        _immutable_projection(manifest), sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    manifest["immutable_sha256"] = hashlib.sha256(immutable_payload).hexdigest()

    if output.exists():
        raise SystemExit(f"Refusing to overwrite immutable manifest: {output}")

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(output)
    print(f"Manifest: {output}")


if __name__ == "__main__":
    main()
