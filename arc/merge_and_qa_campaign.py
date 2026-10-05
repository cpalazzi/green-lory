#!/usr/bin/env python3
"""Merge explicit Green Lory shard CSVs and enforce campaign QA gates."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import re
from pathlib import Path
import sys
from typing import Any

import pandas as pd


REQUIRED_COLUMNS = (
    "latitude",
    "longitude",
    "country",
    "currency",
    "annual_ammonia_production_t",
    "max_ammonia_capacity_t",
    "max_onshore_ammonia_capacity_t",
    "max_gridless_onshore_ammonia_capacity_t",
    "scaled_design_max_onshore_ammonia_capacity_t",
    "scaled_design_max_gridless_onshore_ammonia_capacity_t",
    "wind_mw_per_t_nh3",
    "solar_mw_per_t_nh3",
    "lcoa_eur_per_t",
    "grid_energy_mwh",
    "grid_energy_reference_mwh",
    "grid_energy_share",
    "grid_energy_tolerance_mwh",
    "uses_grid_backstop",
    "interest_overrides_applied",
    "scenario_id",
    "run_id",
    "manifest_sha256",
    "land_constraint",
    "capacity_rule",
    "land_allocation",
    "temporal_accounting_mode",
    "ramp_limit_basis",
    "site_costs_in_headline",
    "build_cost_multiplier",
    "water_cost_usd_per_m3",
    "land_cost_usd_per_km2_year",
    "water_cost_pct",
    "land_cost_pct",
    "snapshot_hours",
    "simulated_hours",
    "is_full_year_result",
    "total_cost_eur_per_year",
    "lcoa_plant_eur_per_t",
    "headline_cost_identity_residual_eur_per_year",
    "headline_cost_share_total_pct",
    "headline_cost_share_residual_pct",
    "grid_free_result_status",
    "is_gridless_feasible",
    "renewable_union_area_source",
    "renewable_union_area_is_conservative_fallback",
    "renewable_union_area_is_lower_bound_approximation",
    "renewable_union_area_method",
    "renewable_union_area_method_version",
    "legacy_capacity_alias_method",
    "preferred_supplier_capacity_column",
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", action="append", required=True)
    parser.add_argument("--expected-input-count", type=int, required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--qa-output", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--scenario-id", required=True)
    parser.add_argument("--stage", choices=("smoke", "diagnostic", "global"), required=True)
    parser.add_argument("--expected-locations", required=True)
    parser.add_argument(
        "--expected-locations-kind",
        choices=("explicit", "land"),
        required=True,
    )
    parser.add_argument("--expected-currency", default="EUR")
    parser.add_argument("--require-interest-overrides", action="store_true")
    parser.add_argument("--require-full-year", action="store_true")
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _coordinate_columns(frame: pd.DataFrame) -> tuple[str, str]:
    columns = {str(column).strip().lower(): str(column) for column in frame.columns}
    lat = columns.get("latitude") or columns.get("lat")
    lon = columns.get("longitude") or columns.get("lon")
    if lat is None or lon is None:
        raise ValueError("Table must contain latitude/longitude or lat/lon columns")
    return lat, lon


def _coordinate_tuples(frame: pd.DataFrame) -> list[tuple[float, float]]:
    lat, lon = _coordinate_columns(frame)
    coordinates = zip(
        pd.to_numeric(frame[lat], errors="raise"),
        pd.to_numeric(frame[lon], errors="raise"),
    )
    return [(round(float(la), 8), round(float(lo), 8)) for la, lo in coordinates]


def _coordinate_sha256(coordinates: set[tuple[float, float]]) -> str:
    payload = "\n".join(f"{lat:.8f},{lon:.8f}" for lat, lon in sorted(coordinates))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _expected_coordinates(path: Path, kind: str) -> set[tuple[float, float]]:
    frame = pd.read_csv(path)
    if kind == "land":
        lower = {str(column).strip().lower(): column for column in frame.columns}
        if "max_capacity_mw" in lower:
            values = pd.to_numeric(frame[lower["max_capacity_mw"]], errors="raise")
            frame = frame.loc[values > 0]
        elif "availability" in lower:
            values = pd.to_numeric(frame[lower["availability"]], errors="raise")
            frame = frame.loc[values > 0]
    coordinates = _coordinate_tuples(frame)
    if len(coordinates) != len(set(coordinates)):
        raise ValueError(f"Expected-locations source contains duplicate coordinates: {path}")
    if not coordinates:
        raise ValueError(f"Expected-locations source is empty after filtering: {path}")
    return set(coordinates)


def _truthy_series(series: pd.Series) -> pd.Series:
    truthy = {"1", "true", "yes", "y", "t"}
    return series.map(lambda value: str(value).strip().lower() in truthy)


def _require_constant_text(frame: pd.DataFrame, column: str, expected: str) -> None:
    values = set(frame[column].dropna().astype(str))
    if values != {str(expected)}:
        raise ValueError(f"Expected {column}={expected!r}, found {sorted(values)}")


def _require_constant_bool(frame: pd.DataFrame, column: str, expected: bool) -> None:
    values = set(_truthy_series(frame[column]))
    if values != {expected}:
        raise ValueError(f"Expected {column}={expected}, found {sorted(values)}")


def _require_finite_numeric(
    frame: pd.DataFrame,
    columns: tuple[str, ...],
    *,
    positive: bool = False,
) -> None:
    for column in columns:
        values = pd.to_numeric(frame[column], errors="raise")
        if not values.map(lambda value: pd.notna(value) and math.isfinite(float(value))).all():
            raise ValueError(f"Column {column} contains non-finite values")
        if positive and not (values > 0.0).all():
            raise ValueError(f"Column {column} must be positive in every result row")


def _require_constant_numeric(
    frame: pd.DataFrame,
    column: str,
    expected: float,
    *,
    absolute_tolerance: float = 1e-12,
) -> None:
    values = pd.to_numeric(frame[column], errors="raise")
    matches = values.map(
        lambda value: pd.notna(value)
        and math.isfinite(float(value))
        and math.isclose(
            float(value),
            float(expected),
            rel_tol=0.0,
            abs_tol=absolute_tolerance,
        )
    )
    if not matches.all():
        found = sorted(set(float(value) for value in values.dropna()))
        raise ValueError(f"Expected {column}={expected}, found {found[:10]}")


def _flat_cost_scope_expectations(
    manifest: dict[str, Any],
    execution: dict[str, Any],
) -> dict[str, Any]:
    scope = manifest.get("cost_scope")
    if not isinstance(scope, dict):
        raise ValueError("Manifest is missing structured cost_scope metadata")
    scope_id = str(scope.get("id"))
    scope_match = re.match(
        r"^flat_(amelired|wacc\d+(?:p\d+)?)_uniform_baseline_water_no_land_rent$", scope_id
    )
    if scope_match is None:
        raise ValueError(f"Unsupported manifest cost scope: {scope.get('id')!r}")

    finance = scope.get("finance", {})
    expected_mode = "ameli_reduced_wacc" if scope_match.group(1) == "amelired" else "uniform_wacc"
    if finance.get("mode") != expected_mode:
        raise ValueError(f"Manifest cost scope {scope_id} must declare finance mode {expected_mode}")
    if expected_mode == "uniform_wacc" and not isinstance(finance.get("rate"), (int, float)):
        raise ValueError("Uniform-WACC cost scope must record its rate")
    if finance.get("override_column") != "interest_rate":
        raise ValueError("Manifest finance scope must use the interest_rate override")

    build = scope.get("spatial_build_and_remoteness", {})
    if build.get("mode") != "none_flat":
        raise ValueError("Manifest cost scope must disable spatial build/remoteness")
    if build.get("override_columns_present") != []:
        raise ValueError("Flat cost scope cannot declare spatial build/remoteness columns")

    water = scope.get("water", {})
    if water.get("mode") != "uniform_yaml_baseline":
        raise ValueError("Manifest water scope must use the uniform YAML baseline")
    if water.get("source") != "resolved_tech_config.water_cost_baseline_usd_per_m3":
        raise ValueError("Manifest water scope has an unexpected source")
    if water.get("source_currency") != "USD":
        raise ValueError("Manifest baseline water cost must be denominated in USD")

    land = scope.get("land_rent", {})
    if land.get("mode") != "none_zero" or land.get("source") != "no_land_rent_input":
        raise ValueError("Manifest must declare that no land-rent input is available")
    if land.get("included_in_headline") is not False:
        raise ValueError("A missing land-rent input cannot be declared headline-active")

    override_columns = (
        manifest.get("inputs", {}).get("override_csv", {}).get("columns")
    )
    if not isinstance(override_columns, list):
        raise ValueError("Manifest must record override CSV column names")
    normalized_columns = [str(column).strip().lower() for column in override_columns]
    if normalized_columns != ["lat", "lon", "tech", "interest_rate"]:
        raise ValueError(
            "flat_amelired requires an interest-only override CSV; "
            f"manifest records {override_columns}"
        )

    try:
        build_multiplier = float(build["expected_result_build_cost_multiplier"])
        water_cost = float(water["expected_result_cost_usd_per_m3"])
        land_cost = float(land["expected_result_cost_usd_per_km2_year"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("Manifest cost scope has incomplete numeric expectations") from exc
    if not math.isclose(build_multiplier, 1.0, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError("Flat cost scope must expect build_cost_multiplier=1")
    try:
        water_cost_model = float(water["expected_result_cost_model_currency_per_m3"])
        source_to_model = float(water["source_to_model_currency"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            "Manifest water scope must record the baseline in model currency and the conversion"
        ) from exc
    if not math.isclose(water_cost * source_to_model, water_cost_model, rel_tol=1e-5, abs_tol=1e-8):
        raise ValueError("Manifest water scope is inconsistent between source and model currency")
    if not math.isclose(land_cost, 0.0, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError("No-land-rent scope must expect zero land cost")

    include_site_costs = bool(execution.get("include_site_costs"))
    if water.get("included_in_headline") is not include_site_costs:
        raise ValueError(
            "Manifest water headline inclusion contradicts execution.include_site_costs"
        )
    return {
        "build_cost_multiplier": build_multiplier,
        "water_cost_usd_per_m3": water_cost,
        "water_cost_model_currency_per_m3": water_cost_model,
        "land_cost_usd_per_km2_year": land_cost,
        "water_included_in_headline": include_site_costs,
    }


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _run(args: argparse.Namespace) -> dict[str, Any]:
    input_paths = [Path(value) for value in args.input]
    if len(input_paths) != args.expected_input_count:
        raise ValueError(
            f"Expected {args.expected_input_count} explicit inputs, received {len(input_paths)}"
        )
    if len(input_paths) != len(set(input_paths)):
        raise ValueError("The explicit input list contains duplicate paths")

    manifest_path = Path(args.manifest)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest_sha = _sha256(manifest_path)
    if manifest.get("scenario", {}).get("id") != args.scenario_id:
        raise ValueError("Manifest scenario ID does not match the requested scenario")
    if manifest.get("execution", {}).get("stage") != args.stage:
        raise ValueError("Manifest stage does not match the requested stage")
    if manifest.get("execution", {}).get("fail_fast") is not True:
        raise ValueError("Manifest does not record fail_fast=true")
    execution = manifest.get("execution", {})
    cost_scope_expectations = _flat_cost_scope_expectations(manifest, execution)

    frames: list[pd.DataFrame] = []
    input_records: list[dict[str, Any]] = []
    failed_sidecars: list[str] = []
    for path in input_paths:
        if not path.is_file():
            raise FileNotFoundError(f"Explicit shard output is missing: {path}")
        if path.stat().st_size == 0:
            raise ValueError(f"Explicit shard output is empty: {path}")
        frame = pd.read_csv(path)
        if frame.empty:
            raise ValueError(f"Explicit shard contains zero rows: {path}")
        missing = [column for column in REQUIRED_COLUMNS if column not in frame.columns]
        if missing:
            raise ValueError(f"{path} is missing required columns: {', '.join(missing)}")
        frames.append(frame)
        input_records.append(
            {
                "path": str(path),
                "rows": len(frame),
                "sha256": _sha256(path),
            }
        )
        failed_sidecars.extend(
            str(item) for item in sorted(path.parent.glob(f"{path.stem}_failed_*.csv"))
        )

    if failed_sidecars:
        raise ValueError(
            "Failure sidecar CSVs exist beside explicit inputs: " + ", ".join(failed_sidecars)
        )

    combined = pd.concat(frames, ignore_index=True)
    if combined[["latitude", "longitude"]].isna().any().any():
        raise ValueError("Merged coordinates contain missing values")
    duplicate_mask = combined.duplicated(subset=["latitude", "longitude"], keep=False)
    if duplicate_mask.any():
        sample = combined.loc[duplicate_mask, ["latitude", "longitude"]].head(10)
        raise ValueError(f"Duplicate result coordinates found:\n{sample.to_string(index=False)}")

    actual_coordinates = set(_coordinate_tuples(combined))
    expected_coordinates = _expected_coordinates(
        Path(args.expected_locations), args.expected_locations_kind
    )
    missing_coordinates = sorted(expected_coordinates - actual_coordinates)
    unexpected_coordinates = sorted(actual_coordinates - expected_coordinates)
    if missing_coordinates or unexpected_coordinates:
        raise ValueError(
            "Result coordinate set does not match expected locations: "
            f"missing={len(missing_coordinates)} sample={missing_coordinates[:10]}, "
            f"unexpected={len(unexpected_coordinates)} sample={unexpected_coordinates[:10]}"
        )

    currencies = set(combined["currency"].dropna().astype(str).str.upper())
    if currencies != {args.expected_currency.upper()}:
        raise ValueError(
            f"Expected currency {args.expected_currency.upper()}, found {sorted(currencies)}"
        )

    _require_finite_numeric(
        combined,
        (
            "annual_ammonia_production_t",
            "lcoa_eur_per_t",
            "lcoa_plant_eur_per_t",
            "total_cost_eur_per_year",
            "grid_energy_reference_mwh",
        ),
        positive=True,
    )
    capacity_columns = (
        "max_ammonia_capacity_t",
        "max_onshore_ammonia_capacity_t",
        "max_gridless_onshore_ammonia_capacity_t",
        "scaled_design_max_onshore_ammonia_capacity_t",
        "scaled_design_max_gridless_onshore_ammonia_capacity_t",
    )
    _require_finite_numeric(combined, capacity_columns)
    for column in capacity_columns:
        if not (pd.to_numeric(combined[column], errors="raise") >= 0.0).all():
            raise ValueError(f"Column {column} must be non-negative in every result row")
    _require_finite_numeric(
        combined,
        (
            "grid_energy_mwh",
            "grid_energy_share",
            "grid_energy_tolerance_mwh",
            "headline_cost_identity_residual_eur_per_year",
            "headline_cost_share_total_pct",
            "headline_cost_share_residual_pct",
            "build_cost_multiplier",
            "water_cost_usd_per_m3",
            "land_cost_usd_per_km2_year",
            "water_cost_pct",
            "land_cost_pct",
        ),
    )
    _require_constant_numeric(
        combined,
        "build_cost_multiplier",
        cost_scope_expectations["build_cost_multiplier"],
    )
    _require_constant_numeric(
        combined,
        "water_cost_usd_per_m3",
        cost_scope_expectations["water_cost_usd_per_m3"],
    )
    _require_constant_numeric(
        combined,
        "land_cost_usd_per_km2_year",
        cost_scope_expectations["land_cost_usd_per_km2_year"],
    )
    _require_constant_numeric(combined, "land_cost_pct", 0.0)
    water_cost_percentages = pd.to_numeric(
        combined["water_cost_pct"], errors="raise"
    )
    if cost_scope_expectations["water_included_in_headline"]:
        if not (water_cost_percentages > 0.0).all():
            raise ValueError(
                "Uniform baseline water is headline-active but water_cost_pct is not positive"
            )
    else:
        _require_constant_numeric(combined, "water_cost_pct", 0.0)

    share_total = pd.to_numeric(
        combined["headline_cost_share_total_pct"], errors="raise"
    )
    share_residual = pd.to_numeric(
        combined["headline_cost_share_residual_pct"], errors="raise"
    )
    if not (share_total.sub(100.0).abs() <= 1e-8).all():
        raise ValueError("Headline cost shares do not sum to 100%")
    if not (share_residual.abs() <= 1e-8).all():
        raise ValueError("Headline cost-share residual is non-zero")

    identity_residual = pd.to_numeric(
        combined["headline_cost_identity_residual_eur_per_year"], errors="raise"
    ).abs()
    headline_cost = pd.to_numeric(combined["total_cost_eur_per_year"], errors="raise").abs()
    identity_tolerance = pd.Series(
        [max(1e-3, 1e-10 * value) for value in headline_cost],
        index=combined.index,
    )
    if not (identity_residual <= identity_tolerance).all():
        raise ValueError("Headline annual-cost identity residual exceeds tolerance")

    _require_constant_text(
        combined,
        "legacy_capacity_alias_method",
        "historical_independent_total_vs_onshore_power_caps_v0",
    )
    _require_constant_text(
        combined,
        "preferred_supplier_capacity_column",
        "scaled_design_max_gridless_onshore_ammonia_capacity_t",
    )
    _require_constant_text(
        combined, "grid_free_result_status", "equivalent_within_tolerance"
    )
    _require_constant_bool(combined, "is_gridless_feasible", True)

    union_sources = set(combined["renewable_union_area_source"].dropna().astype(str))
    allowed_union_sources = {
        "explicit_classwise_nested_v1_lower_bound",
        "explicit_legacy_unversioned",
        "conservative_max",
    }
    if not union_sources or not union_sources.issubset(allowed_union_sources):
        raise ValueError(f"Unexpected renewable-union source(s): {sorted(union_sources)}")
    if args.stage != "smoke":
        _require_constant_text(
            combined,
            "renewable_union_area_source",
            "explicit_classwise_nested_v1_lower_bound",
        )
        _require_constant_bool(
            combined, "renewable_union_area_is_conservative_fallback", False
        )
        _require_constant_bool(
            combined, "renewable_union_area_is_lower_bound_approximation", True
        )
        _require_constant_text(
            combined,
            "renewable_union_area_method",
            "classwise_nested_overlap_lower_bound",
        )
        _require_constant_text(
            combined, "renewable_union_area_method_version", "v1"
        )
    else:
        fallback_flags = _truthy_series(
            combined["renewable_union_area_is_conservative_fallback"]
        )
        expected_fallback = combined["renewable_union_area_source"].astype(str).eq(
            "conservative_max"
        )
        if not fallback_flags.equals(expected_fallback):
            raise ValueError("Smoke union fallback flag contradicts its source label")

    if args.require_interest_overrides and not _truthy_series(
        combined["interest_overrides_applied"]
    ).all():
        raise ValueError("One or more rows report interest_overrides_applied=false")

    _require_constant_text(combined, "scenario_id", args.scenario_id)
    _require_constant_text(combined, "run_id", manifest["run_id"])
    _require_constant_text(
        combined, "manifest_sha256", manifest_sha
    )
    for column in (
        "land_constraint",
        "capacity_rule",
        "land_allocation",
        "temporal_accounting_mode",
        "ramp_limit_basis",
    ):
        _require_constant_text(combined, column, execution[column])
    _require_constant_bool(
        combined, "site_costs_in_headline", bool(execution["include_site_costs"])
    )
    _require_constant_bool(combined, "uses_grid_backstop", False)

    snapshot_hours = pd.to_numeric(combined["snapshot_hours"], errors="raise")
    if not (snapshot_hours == float(execution["time_step_hours"])).all():
        raise ValueError("snapshot_hours does not match the manifest timestep")
    simulated_hours = pd.to_numeric(combined["simulated_hours"], errors="raise")
    if not (simulated_hours == float(execution["simulated_hours"])).all():
        raise ValueError("simulated_hours does not match the manifest")
    if args.require_full_year:
        if execution.get("expected_full_year") is not True:
            raise ValueError("Full-year QA requested but manifest does not expect a full year")
        if not _truthy_series(combined["is_full_year_result"]).all():
            raise ValueError("One or more diagnostic/global rows are not full-year results")

    metadata = {
        "campaign_id": manifest["campaign_id"],
        "run_id": manifest["run_id"],
        "scenario_id": args.scenario_id,
        "run_stage": args.stage,
        "source_manifest_sha256": manifest_sha,
    }
    for column, value in metadata.items():
        if column in combined.columns:
            existing = set(combined[column].dropna().astype(str))
            if existing and existing != {str(value)}:
                raise ValueError(f"Conflicting existing metadata in column {column}")
        combined[column] = value

    combined = combined.sort_values(["latitude", "longitude"]).reset_index(drop=True)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    combined.to_csv(temporary, index=False)
    temporary.replace(output_path)

    return {
        "schema_version": 1,
        "status": "passed",
        "checked_utc": datetime.now(timezone.utc).isoformat(),
        "campaign_id": manifest["campaign_id"],
        "run_id": manifest["run_id"],
        "scenario_id": args.scenario_id,
        "stage": args.stage,
        "manifest": str(manifest_path),
        "manifest_sha256": manifest_sha,
        "inputs": input_records,
        "output": str(output_path),
        "output_sha256": _sha256(output_path),
        "row_count": len(combined),
        "expected_row_count": len(expected_coordinates),
        "coordinate_sha256": _coordinate_sha256(actual_coordinates),
        "duplicate_coordinate_count": 0,
        "missing_coordinate_count": 0,
        "unexpected_coordinate_count": 0,
        "currency": args.expected_currency.upper(),
        "interest_overrides_required": args.require_interest_overrides,
        "full_year_required": args.require_full_year,
        "zero_corrected_supplier_capacity_rows": int(
            (
                pd.to_numeric(
                    combined["scaled_design_max_gridless_onshore_ammonia_capacity_t"],
                    errors="raise",
                )
                <= 0.0
            ).sum()
        ),
        "positive_corrected_supplier_capacity_rows": int(
            (
                pd.to_numeric(
                    combined["scaled_design_max_gridless_onshore_ammonia_capacity_t"],
                    errors="raise",
                )
                > 0.0
            ).sum()
        ),
        "scenario_execution": execution,
        "cost_scope": manifest["cost_scope"],
    }


def main() -> None:
    args = _parse_args()
    qa_path = Path(args.qa_output)
    try:
        report = _run(args)
    except Exception as exc:  # noqa: BLE001 - persist the failed gate before exiting non-zero
        report = {
            "schema_version": 1,
            "status": "failed",
            "checked_utc": datetime.now(timezone.utc).isoformat(),
            "scenario_id": args.scenario_id,
            "stage": args.stage,
            "error_type": type(exc).__name__,
            "error": str(exc),
        }
        _write_json_atomic(qa_path, report)
        print(f"QA failed; report: {qa_path}", file=sys.stderr)
        raise

    _write_json_atomic(qa_path, report)
    print(f"Merged {len(args.input)} explicit input file(s)")
    print(f"Rows: {report['row_count']}")
    print(f"Output: {report['output']}")
    print(f"QA passed: {qa_path}")


if __name__ == "__main__":
    main()
