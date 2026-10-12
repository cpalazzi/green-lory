"""Quality gates of a campaign: the land table before a run, the merged surface after it.

- ``land``: a centred land table is fit for a campaign: required columns, the expected uniform
  land share, the classwise nested union area of PV and wind, aliases that agree, values in
  range, one row per cell. ``arc/submit_lory_sequence.sh`` runs it in its prepare job.
- ``merge``: the shard CSVs of a run, named explicitly (never found by wildcard), merged into
  ``merged/global_run_results.csv``, then checked against the manifest: every expected cell
  once, the run's settings and cost scope on every row, finite results, full-year results
  where required. The report ``qa/validation.json`` is written whether the gate passes or
  fails; only ``"status": "passed"`` makes a surface accepted.

    python -m arc.qa land <land csv> --expected-land-fraction 0.2
    python -m arc.qa merge --input <shard csv> [--input ...] --expected-input-count 4 ...
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
import re
from pathlib import Path
import sys
from typing import Any, Iterable, Mapping, Sequence

import pandas as pd


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _coordinate_sha256(coordinates: Iterable[tuple[float, float]]) -> str:
    payload = "\n".join(
        f"{latitude:.8f},{longitude:.8f}"
        for latitude, longitude in sorted(coordinates)
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


# --- the land table ------------------------------------------------------------------------------

CLASSWISE_NESTED_UNION_AREA_COLUMN = (
    "renewable_union_area_km2_classwise_nested_v1"
)
CLASSWISE_NESTED_UNION_AVAILABILITY_COLUMN = (
    "renewable_union_availability_classwise_nested_v1"
)
GENERIC_UNION_AREA_COLUMN = "renewable_union_area_km2"
GENERIC_UNION_AVAILABILITY_COLUMN = "renewable_union_availability"
RENEWABLE_UNION_METHOD_COLUMN = "renewable_union_method"
RENEWABLE_UNION_METHOD_VERSION_COLUMN = "renewable_union_method_version"
EXPECTED_RENEWABLE_UNION_METHOD = "classwise_nested_overlap_lower_bound"
EXPECTED_RENEWABLE_UNION_METHOD_VERSION = "v1"
# Tables built before 23 Sep 2026 have no cell_anchor column and are offset by one degree.
CELL_ANCHOR_COLUMN = "cell_anchor"
EXPECTED_CELL_ANCHOR = "center"

LAND_REQUIRED_COLUMNS = (
    "latitude",
    "longitude",
    CELL_ANCHOR_COLUMN,
    "land_competition_fraction",
    "wind_onshore_area_km2",
    "solar_area_km2",
    CLASSWISE_NESTED_UNION_AREA_COLUMN,
    GENERIC_UNION_AREA_COLUMN,
    CLASSWISE_NESTED_UNION_AVAILABILITY_COLUMN,
    GENERIC_UNION_AVAILABILITY_COLUMN,
    RENEWABLE_UNION_METHOD_COLUMN,
    RENEWABLE_UNION_METHOD_VERSION_COLUMN,
    "max_power_solar_mw",
    "max_power_wind_mw",
    "max_capacity_mw",
)

COORDINATE_DECIMAL_PLACES = 8
# Match the campaign runtime's alias contract: only CSV round-trip noise at the
# absolute scale is tolerated; large values do not receive a looser allowance.
ALIAS_REL_TOLERANCE = 0.0
ALIAS_ABS_TOLERANCE = 1e-9
FRACTION_REL_TOLERANCE = 1e-9
FRACTION_ABS_TOLERANCE = 1e-12
UNION_REL_TOLERANCE = 1e-9
UNION_ABS_TOLERANCE_KM2 = 1e-9
RANGE_ABS_TOLERANCE = 1e-12


class LandCampaignInputError(ValueError):
    """Raised when a land campaign input violates a scientific invariant."""


def _number(row: Mapping[str, str], column: str, row_number: int) -> float:
    raw_value = row.get(column)
    try:
        value = float(raw_value)
    except (TypeError, ValueError) as exc:
        raise LandCampaignInputError(
            f"Row {row_number}: {column} must be numeric, found {raw_value!r}"
        ) from exc
    if not math.isfinite(value):
        raise LandCampaignInputError(
            f"Row {row_number}: {column} contains non-finite value {raw_value!r}"
        )
    return value


def _is_area_or_capacity_column(column: str) -> bool:
    return (
        column == "area"
        or column == CLASSWISE_NESTED_UNION_AREA_COLUMN
        or column.endswith("_area_km2")
        or column == "max_capacity_mw"
        or (column.startswith("max_power_") and column.endswith("_mw"))
    )


def _is_availability_column(column: str) -> bool:
    return (
        column == "availability"
        or column == CLASSWISE_NESTED_UNION_AVAILABILITY_COLUMN
        or column.endswith("_availability")
    )


def _close(left: float, right: float, *, rel_tol: float, abs_tol: float) -> bool:
    return math.isclose(left, right, rel_tol=rel_tol, abs_tol=abs_tol)


def _union_tolerance(*values: float) -> float:
    scale = max(1.0, *(abs(value) for value in values))
    return UNION_ABS_TOLERANCE_KM2 + UNION_REL_TOLERANCE * scale


def _canonical_coordinate(latitude: float, longitude: float) -> tuple[float, float]:
    return (
        round(latitude, COORDINATE_DECIMAL_PLACES),
        round(longitude, COORDINATE_DECIMAL_PLACES),
    )


def _read_rows(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        fieldnames = list(reader.fieldnames or [])
        if not fieldnames:
            raise LandCampaignInputError(f"Land table has no CSV header: {path}")
        if len(fieldnames) != len(set(fieldnames)):
            raise LandCampaignInputError(f"Land table has duplicate CSV columns: {path}")
        missing = sorted(set(LAND_REQUIRED_COLUMNS) - set(fieldnames))
        if missing:
            raise LandCampaignInputError(
                f"Land table is missing required columns: {', '.join(missing)}"
            )

        rows: list[dict[str, str]] = []
        for row_number, row in enumerate(reader, start=2):
            if None in row:
                raise LandCampaignInputError(
                    f"Row {row_number}: contains fields beyond the CSV header"
                )
            rows.append(row)

    if not rows:
        raise LandCampaignInputError(f"Land table contains no data rows: {path}")
    return fieldnames, rows


def validate_land_campaign_input(
    path: str | Path,
    *,
    expected_land_fraction: float,
) -> dict[str, object]:
    """Validate ``path`` and return a compact, reproducible identity summary."""

    csv_path = Path(path)
    expected_fraction = float(expected_land_fraction)
    if not math.isfinite(expected_fraction) or not 0.0 <= expected_fraction <= 1.0:
        raise LandCampaignInputError(
            "expected_land_fraction must be finite and between 0 and 1"
        )

    fieldnames, rows = _read_rows(csv_path)
    numeric_area_and_capacity_columns = tuple(
        column for column in fieldnames if _is_area_or_capacity_column(column)
    )
    availability_columns = tuple(
        column for column in fieldnames if _is_availability_column(column)
    )

    coordinates: set[tuple[float, float]] = set()
    fractions: list[float] = []

    for row_number, row in enumerate(rows, start=2):
        latitude = _number(row, "latitude", row_number)
        longitude = _number(row, "longitude", row_number)
        if not -90.0 <= latitude <= 90.0:
            raise LandCampaignInputError(
                f"Row {row_number}: latitude outside [-90, 90]: {latitude}"
            )
        if not -180.0 <= longitude <= 180.0:
            raise LandCampaignInputError(
                f"Row {row_number}: longitude outside [-180, 180]: {longitude}"
            )
        coordinate = _canonical_coordinate(latitude, longitude)
        if coordinate in coordinates:
            raise LandCampaignInputError(
                "Duplicate coordinate after 8-decimal normalization: "
                f"({coordinate[0]:.8f}, {coordinate[1]:.8f})"
            )
        coordinates.add(coordinate)

        anchor = row.get(CELL_ANCHOR_COLUMN)
        if anchor != EXPECTED_CELL_ANCHOR:
            raise LandCampaignInputError(
                f"Row {row_number}: expected {CELL_ANCHOR_COLUMN}="
                f"{EXPECTED_CELL_ANCHOR!r}, found {anchor!r}"
            )

        fraction = _number(row, "land_competition_fraction", row_number)
        if not -RANGE_ABS_TOLERANCE <= fraction <= 1.0 + RANGE_ABS_TOLERANCE:
            raise LandCampaignInputError(
                f"Row {row_number}: land_competition_fraction outside [0, 1]: "
                f"{fraction}"
            )
        fractions.append(fraction)

        method = row.get(RENEWABLE_UNION_METHOD_COLUMN)
        if method != EXPECTED_RENEWABLE_UNION_METHOD:
            raise LandCampaignInputError(
                f"Row {row_number}: expected {RENEWABLE_UNION_METHOD_COLUMN}="
                f"{EXPECTED_RENEWABLE_UNION_METHOD!r}, found {method!r}"
            )
        version = row.get(RENEWABLE_UNION_METHOD_VERSION_COLUMN)
        if version != EXPECTED_RENEWABLE_UNION_METHOD_VERSION:
            raise LandCampaignInputError(
                f"Row {row_number}: expected {RENEWABLE_UNION_METHOD_VERSION_COLUMN}="
                f"{EXPECTED_RENEWABLE_UNION_METHOD_VERSION!r}, found {version!r}"
            )

        numeric_values: dict[str, float] = {}
        for column in numeric_area_and_capacity_columns:
            value = _number(row, column, row_number)
            if value < 0.0:
                raise LandCampaignInputError(
                    f"Row {row_number}: {column} must be nonnegative, found {value}"
                )
            numeric_values[column] = value

        availability_values: dict[str, float] = {}
        for column in availability_columns:
            value = _number(row, column, row_number)
            if not -RANGE_ABS_TOLERANCE <= value <= 1.0 + RANGE_ABS_TOLERANCE:
                raise LandCampaignInputError(
                    f"Row {row_number}: {column} outside [0, 1]: {value}"
                )
            availability_values[column] = value

        versioned_area = numeric_values[CLASSWISE_NESTED_UNION_AREA_COLUMN]
        generic_area = numeric_values[GENERIC_UNION_AREA_COLUMN]
        if not _close(
            versioned_area,
            generic_area,
            rel_tol=ALIAS_REL_TOLERANCE,
            abs_tol=ALIAS_ABS_TOLERANCE,
        ):
            raise LandCampaignInputError(
                f"Row {row_number}: compatibility alias {GENERIC_UNION_AREA_COLUMN} "
                f"diverges from {CLASSWISE_NESTED_UNION_AREA_COLUMN}"
            )

        versioned_availability = availability_values[
            CLASSWISE_NESTED_UNION_AVAILABILITY_COLUMN
        ]
        generic_availability = availability_values[GENERIC_UNION_AVAILABILITY_COLUMN]
        if not _close(
            versioned_availability,
            generic_availability,
            rel_tol=ALIAS_REL_TOLERANCE,
            abs_tol=ALIAS_ABS_TOLERANCE,
        ):
            raise LandCampaignInputError(
                f"Row {row_number}: compatibility alias "
                f"{GENERIC_UNION_AVAILABILITY_COLUMN} diverges from "
                f"{CLASSWISE_NESTED_UNION_AVAILABILITY_COLUMN}"
            )

        wind_area = numeric_values["wind_onshore_area_km2"]
        solar_area = numeric_values["solar_area_km2"]
        lower_bound = max(wind_area, solar_area)
        upper_bound = wind_area + solar_area
        tolerance = _union_tolerance(
            versioned_area, wind_area, solar_area, lower_bound, upper_bound
        )
        if versioned_area < lower_bound - tolerance:
            raise LandCampaignInputError(
                f"Row {row_number}: versioned renewable union {versioned_area} km2 "
                f"is below max(wind_onshore, solar)={lower_bound} km2"
            )
        if versioned_area > upper_bound + tolerance:
            raise LandCampaignInputError(
                f"Row {row_number}: versioned renewable union {versioned_area} km2 "
                f"exceeds wind_onshore+solar={upper_bound} km2"
            )

    reference_fraction = fractions[0]
    for row_offset, fraction in enumerate(fractions[1:], start=3):
        if not _close(
            fraction,
            reference_fraction,
            rel_tol=FRACTION_REL_TOLERANCE,
            abs_tol=FRACTION_ABS_TOLERANCE,
        ):
            raise LandCampaignInputError(
                "land_competition_fraction is not uniform: "
                f"row 2 has {reference_fraction}, row {row_offset} has {fraction}"
            )
    if not _close(
        reference_fraction,
        expected_fraction,
        rel_tol=FRACTION_REL_TOLERANCE,
        abs_tol=FRACTION_ABS_TOLERANCE,
    ):
        raise LandCampaignInputError(
            f"Expected land_competition_fraction={expected_fraction}, "
            f"found {reference_fraction}"
        )

    return {
        "cell_anchor": EXPECTED_CELL_ANCHOR,
        "coordinate_sha256": _coordinate_sha256(coordinates),
        "file": str(csv_path.expanduser().resolve()),
        "file_sha256": _sha256(csv_path),
        "land_competition_fraction": reference_fraction,
        "renewable_union_method": EXPECTED_RENEWABLE_UNION_METHOD,
        "renewable_union_method_version": EXPECTED_RENEWABLE_UNION_METHOD_VERSION,
        "row_count": len(rows),
        "status": "passed",
    }


def _fraction_argument(value: str) -> float:
    try:
        fraction = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be numeric") from exc
    if not math.isfinite(fraction) or not 0.0 <= fraction <= 1.0:
        raise argparse.ArgumentTypeError("must be finite and between 0 and 1")
    return fraction


def _land_command(args: argparse.Namespace) -> int:
    try:
        summary = validate_land_campaign_input(
            args.land_csv,
            expected_land_fraction=args.expected_land_fraction,
        )
    except (LandCampaignInputError, OSError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(summary, sort_keys=True, separators=(",", ":")))
    return 0


# --- the merged surface ---------------------------------------------------------------------

SURFACE_REQUIRED_COLUMNS = (
    "latitude",
    "longitude",
    "country",
    "currency",
    "annual_ammonia_production_t",
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
    "renewable_union_area_is_lower_bound_approximation",
    "renewable_union_area_method",
    "renewable_union_area_method_version",
    "preferred_supplier_capacity_column",
)


def _add_merge_arguments(parser: argparse.ArgumentParser) -> None:
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

    # the rates come from a finance table (interest-only) or one rate for every technology and cell
    inputs = manifest.get("inputs", {})
    if finance.get("source") == "interest_rate":
        if not isinstance(inputs.get("interest_rate"), (int, float)):
            raise ValueError("Manifest finance from interest_rate must record inputs.interest_rate")
    else:
        override_columns = inputs.get("override_csv", {}).get("columns")
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



def _site_cost_scope_expectations(manifest: dict[str, Any], execution: dict[str, Any]) -> dict[str, Any]:
    """Expectations of a site_costs_* scope: the results must carry the table's values per cell."""
    scope = manifest["cost_scope"]
    scope_id = str(scope.get("id"))
    finance = scope.get("finance", {})
    token = re.match(r"^site_costs_(amelired|ameli|wacc\d+(?:p\d+)?)_no_land_rent$", scope_id).group(1)
    expected_mode = {"amelired": "ameli_reduced_wacc", "ameli": "ameli_wacc"}.get(token, "uniform_wacc")
    if finance.get("mode") != expected_mode or finance.get("override_column") != "interest_rate":
        raise ValueError(f"Manifest cost scope {scope_id} must declare finance mode {expected_mode} from interest_rate")
    build = scope.get("spatial_build_and_remoteness", {})
    water = scope.get("water", {})
    if build.get("mode") != "site_cost_table" or water.get("mode") != "site_cost_table":
        raise ValueError(f"Manifest cost scope {scope_id} must take build costs and water from the site-cost table")
    land = scope.get("land_rent", {})
    if land.get("mode") != "none_zero" or land.get("included_in_headline") is not False:
        raise ValueError("Manifest must declare that no land-rent input is available")
    record = manifest.get("inputs", {}).get("site_costs_csv")
    if not isinstance(record, dict) or not record.get("resolved_path"):
        raise ValueError(f"Manifest cost scope {scope_id} needs inputs.site_costs_csv")
    include_site_costs = bool(execution.get("include_site_costs"))
    if water.get("included_in_headline") is not include_site_costs:
        raise ValueError("Manifest water headline inclusion contradicts execution.include_site_costs")
    return {
        "mode": "site_cost_table",
        "site_costs_csv": Path(record["resolved_path"]),
        "site_costs_sha256": record.get("sha256"),
        "land_cost_usd_per_km2_year": 0.0,
        "water_included_in_headline": include_site_costs,
    }


def _cost_scope_expectations(manifest: dict[str, Any], execution: dict[str, Any]) -> dict[str, Any]:
    scope = manifest.get("cost_scope")
    if isinstance(scope, dict) and str(scope.get("id", "")).startswith("site_costs_"):
        return _site_cost_scope_expectations(manifest, execution)
    return {"mode": "flat", **_flat_cost_scope_expectations(manifest, execution)}


def _check_site_costs(combined: pd.DataFrame, expectations: dict[str, Any]) -> dict[str, Any]:
    """Every cell carries the site-cost table's equipment multiplier and water price, and a
    build multiplier between its installation factor and that times its slope factor."""
    path = expectations["site_costs_csv"]
    if expectations.get("site_costs_sha256") and _sha256(path) != expectations["site_costs_sha256"]:
        raise ValueError(f"Site-cost table changed since the manifest was written: {path}")
    table = pd.read_csv(path)
    table["latitude"], table["longitude"] = table["latitude"].round(4), table["longitude"].round(4)
    results = combined.assign(latitude=combined["latitude"].round(4), longitude=combined["longitude"].round(4))
    merged = results.merge(table, on=["latitude", "longitude"], how="left", suffixes=("", "_table"), indicator=True)
    if (merged["_merge"] != "both").any():
        raise ValueError(f"{int((merged['_merge'] != 'both').sum())} result cells have no site-cost row")
    if "equipment_cost_multiplier" not in combined.columns:
        raise ValueError("Results with site costs must report equipment_cost_multiplier")
    equipment = pd.to_numeric(merged["equipment_cost_multiplier"], errors="raise")
    if not ((equipment - merged["equipment_cost_multiplier_table"]).abs() <= 1e-9).all():
        raise ValueError("Result equipment multipliers differ from the site-cost table")
    base_build = merged["installation_multiplier"] * merged["labour_multiplier"] * merged["depth_multiplier"]
    build = pd.to_numeric(merged["build_cost_multiplier"], errors="raise")
    if not ((build >= base_build - 1e-9) & (build <= base_build * merged["slope_multiplier"] + 1e-9)).all():
        raise ValueError("Result build multipliers lie outside the site-cost table's range")
    water = pd.to_numeric(merged["water_cost_usd_per_m3"], errors="raise")
    if not ((water - merged["water_cost_usd_per_m3_table"]).abs() <= 1e-9 * merged["water_cost_usd_per_m3_table"].abs().clip(lower=1.0)).all():
        raise ValueError("Result water prices differ from the site-cost table")
    return {"site_cost_cells_checked": int(len(merged))}

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
    cost_scope_expectations = _cost_scope_expectations(manifest, execution)

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
        missing = [column for column in SURFACE_REQUIRED_COLUMNS if column not in frame.columns]
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
    if cost_scope_expectations["mode"] == "site_cost_table":
        site_cost_check = _check_site_costs(combined, cost_scope_expectations)
    else:
        site_cost_check = None
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
                "Water is headline-active but water_cost_pct is not positive"
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
        "preferred_supplier_capacity_column",
        "scaled_design_max_gridless_onshore_ammonia_capacity_t",
    )
    _require_constant_text(
        combined, "grid_free_result_status", "equivalent_within_tolerance"
    )
    _require_constant_bool(combined, "is_gridless_feasible", True)

    # Every stage reads a versioned land table (the model refuses others)
    _require_constant_text(combined, "renewable_union_area_source", "explicit_classwise_nested_v1_lower_bound")
    _require_constant_bool(combined, "renewable_union_area_is_lower_bound_approximation", True)
    _require_constant_text(combined, "renewable_union_area_method", "classwise_nested_overlap_lower_bound")
    _require_constant_text(combined, "renewable_union_area_method_version", "v1")

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
        "cost_scope_mode": cost_scope_expectations["mode"],
        **(site_cost_check or {}),
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


def merge_and_qa(args: argparse.Namespace) -> None:
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


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    land = commands.add_parser("land", help="check a land table before a campaign")
    land.add_argument("land_csv", help="Land campaign CSV to validate")
    land.add_argument(
        "--expected-land-fraction",
        required=True,
        type=_fraction_argument,
        help="Required uniform land_competition_fraction, for example 0.02",
    )
    _add_merge_arguments(commands.add_parser("merge", help="merge a run's shards and apply the QA gate"))
    args = parser.parse_args(argv)
    if args.command == "land":
        return _land_command(args)
    merge_and_qa(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
