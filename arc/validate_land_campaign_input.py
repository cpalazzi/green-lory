#!/usr/bin/env python3
"""Validate a publication-grade Green Lory land campaign CSV.

This command intentionally uses only the Python standard library so it can run
as a small Slurm dependency gate before the scientific environment is loaded.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Iterable, Mapping, Sequence


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

REQUIRED_COLUMNS = (
    "latitude",
    "longitude",
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


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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


def _coordinate_sha256(coordinates: Iterable[tuple[float, float]]) -> str:
    payload = "\n".join(
        f"{latitude:.8f},{longitude:.8f}"
        for latitude, longitude in sorted(coordinates)
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _read_rows(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        fieldnames = list(reader.fieldnames or [])
        if not fieldnames:
            raise LandCampaignInputError(f"Land table has no CSV header: {path}")
        if len(fieldnames) != len(set(fieldnames)):
            raise LandCampaignInputError(f"Land table has duplicate CSV columns: {path}")
        missing = sorted(set(REQUIRED_COLUMNS) - set(fieldnames))
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


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("land_csv", help="Land campaign CSV to validate")
    parser.add_argument(
        "--expected-land-fraction",
        required=True,
        type=_fraction_argument,
        help="Required uniform land_competition_fraction, for example 0.02",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
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


if __name__ == "__main__":
    raise SystemExit(main())
