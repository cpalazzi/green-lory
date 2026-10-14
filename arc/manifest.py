"""Run manifests: what a campaign run uses, written before its shards start, checked after.

A manifest (``<run dir>/manifest.json``) is byte-immutable. It records the source (commit,
release inventory), the resolved settings of the run, its cost scope, and the identity of
every input: sha256 of the technology YAML chain, the plant bundle, the finance table (and the
cell-country table when it is keyed by country), the land table and the cell list, and a
listing (names, sizes, mtimes) of the weather files. ``arc/submit_lory_sequence.sh`` writes it in its prepare job; the merge/QA job verifies
the inputs again before it accepts the shards.

    python -m arc.manifest write --output <run dir>/manifest.json ...   (every option: --help)
    python -m arc.manifest verify --manifest <run dir>/manifest.json

Two cost scopes. The flat one of the key runs and C1/C2
(``flat_<finance>_uniform_baseline_water_no_land_rent``): interest rates from the run, one rate
(``waccN``) or a finance table varying by cell or country (``amelired``, ``ameli``), uniform
baseline water from the YAML,
no build or remoteness multipliers, no land rent. The site-cost one of S1/S2
(``site_costs_<finance>_no_land_rent``, finance also ``ameli``): the same finance, and the
multipliers and the water price of a site-cost table (model/site_costs.py), whose source price
must be the YAML's baseline; no land rent.
"""

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

# The model's assumptions other than technology costs; every run records the file's hash
PARAMETERS_YAML = Path(__file__).resolve().parents[1] / "inputs" / "parameters.yaml"
# The cell-country table through which a finance table keyed by country reaches the cells
CELL_COUNTRIES_CSV = Path(__file__).resolve().parents[1] / "inputs" / "cells" / "cell_countries.csv"


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


def _add_write_arguments(parser: argparse.ArgumentParser) -> None:
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
    finance = parser.add_mutually_exclusive_group(required=True)
    finance.add_argument("--finance-csv", help="finance table, by cell or by country")
    finance.add_argument("--interest-rate", type=float, help="one interest rate for every technology and cell")
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
    parser.add_argument("--land-constraint", choices=("after_solve", "in_solve"), required=True)
    parser.add_argument("--capacity-rule", choices=("scaled_reference_design",), required=True)
    parser.add_argument(
        "--land-allocation",
        choices=("colocated", "exclusive"),
        required=True,
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
        type=_cost_scope_id,
        required=True,
        help=(
            "flat_<finance>_uniform_baseline_water_no_land_rent or site_costs_<finance>_no_land_rent "
            "(with --site-costs-csv), finance 'amelired' (Ameli reduced WACC by country), 'ameli' "
            "(another Ameli series by country, e.g. low-carbon) or 'waccN' (uniform N percent)."
        ),
    )
    parser.add_argument(
        "--flat-water-model-currency-per-m3",
        type=float,
        default=None,
        help="Declared baseline water price in the model currency (e.g. 2.0 EUR2020/m3): the uniform "
        "price of a flat run, the source price of a site-cost run; checked against the tech YAML.",
    )
    parser.add_argument("--site-costs-csv", default=None, help="Site-cost table of a site_costs_* scope.")
    parser.add_argument("--weather-land-correction", default=None,
                        help="Land correction of the weather profiles (model/weather_land.py), if the run uses one.")
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--source-diff-sha256", required=True)
    parser.add_argument("--release-source-inventory")
    parser.add_argument("--expected-currency", default="EUR")


# --- writing ---------------------------------------------------------------------------------

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


def _expected_coordinates(land_csv: Path, locations_csv: Path | None) -> tuple[set, str, Path]:
    """The cells a run will solve: the explicit cell list, or the land table's active cells."""
    if locations_csv is not None:
        return _csv_coordinate_set(locations_csv), "explicit_locations_csv", locations_csv
    return _csv_coordinate_set(land_csv, active_only=True), "active_land_csv", land_csv


def validate_coordinate_coverage(
    *,
    table_csv: Path,
    land_csv: Path,
    locations_csv: Path | None,
) -> dict[str, Any]:
    """Require a row of a per-cell table (the site-cost table) for every location that will be run."""

    table_coordinates = _csv_coordinate_set(table_csv)
    expected_coordinates, expected_source, expected_path = _expected_coordinates(land_csv, locations_csv)
    missing = sorted(expected_coordinates - table_coordinates)
    if missing:
        raise ValueError(
            f"{table_csv} does not cover every run coordinate: "
            f"missing={len(missing)} sample={missing[:10]}, expected={expected_path}"
        )
    return {
        "expected_source": expected_source,
        "expected_coordinate_count": len(expected_coordinates),
        "table_coordinate_count": len(table_coordinates),
        "missing_coordinate_count": 0,
    }


def validate_finance_coverage(
    *,
    finance_csv: Path,
    tech_yaml: Path,
    land_csv: Path,
    locations_csv: Path | None,
) -> dict[str, Any]:
    """Require an interest rate for every technology at every location that will be run, resolved
    as the model resolves it (model.technology.finance_overrides)."""
    from model import technology

    _, key = technology.load_finance_table(finance_csv)
    expected_coordinates, expected_source, _ = _expected_coordinates(land_csv, locations_csv)
    _, gaps = technology.finance_overrides(finance_csv, sorted(expected_coordinates), technology.load_tech_inputs(tech_yaml))
    if gaps:
        raise ValueError(f"{finance_csv} leaves {technology.describe_finance_gaps(gaps)}")
    return {
        "key": key,
        "expected_source": expected_source,
        "expected_coordinate_count": len(expected_coordinates),
        "uncovered_coordinate_count": 0,
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
    r"^flat_(amelired|ameli|wacc\d+(?:p\d+)?)_uniform_baseline_water_no_land_rent$"
)


SITE_COSTS_SCOPE_PATTERN = re.compile(r"^site_costs_(amelired|ameli|wacc\d+(?:p\d+)?)_no_land_rent$")

# Columns a site-cost table must have (model/site_costs.py)
SITE_COST_COLUMNS = (
    "latitude", "longitude", "equipment_cost_multiplier", "installation_multiplier", "slope_multiplier",
    "depth_multiplier", "labour_multiplier", "water_source_cost_usd_per_m3", "water_cost_usd_per_m3",
    "land_cost_usd_per_km2_year",
)


def _flat_cost_scope_id(value: str) -> str:
    if not FLAT_COST_SCOPE_PATTERN.match(value):
        raise argparse.ArgumentTypeError(f"Unsupported cost scope: {value}")
    return value


def _cost_scope_id(value: str) -> str:
    if not (FLAT_COST_SCOPE_PATTERN.match(value) or SITE_COSTS_SCOPE_PATTERN.match(value)):
        raise argparse.ArgumentTypeError(f"Unsupported cost scope: {value}")
    return value


def flat_finance_token(cost_scope_id: str) -> str:
    match = FLAT_COST_SCOPE_PATTERN.match(cost_scope_id)
    if match is None:
        raise ValueError(f"Unsupported cost scope: {cost_scope_id}")
    return match.group(1)


def _finance_scope(
    finance_token: str, cost_scope_id: str, *, finance_csv: Path | None = None, interest_rate: float | None = None
) -> dict[str, Any]:
    """The finance part of a cost scope: rates varying by cell or country (Ameli) from a finance
    table, or one uniform rate, given as the run's interest_rate or as a uniform table."""
    if (finance_csv is None) == (interest_rate is None):
        raise ValueError("A run takes its interest rates from a finance table or one rate, not both")
    if finance_csv is not None:
        from model import technology

        table, _ = technology.load_finance_table(finance_csv)  # checks the table's form
        rates = table["interest_rate"].round(12).unique()
        uniform_rate, source = (float(rates[0]) if len(rates) == 1 else None), "finance_csv"
    else:
        uniform_rate, source = float(interest_rate), "interest_rate"
    if finance_token in ("amelired", "ameli"):
        if uniform_rate is not None:
            raise ValueError(f"{finance_token} cost scope requires country-varying interest rates; the run's are uniform")
        return {
            "mode": "ameli_reduced_wacc" if finance_token == "amelired" else "ameli_wacc",
            "source": source,
            "override_column": "interest_rate",
        }
    expected_rate = float(finance_token[4:].replace("p", ".")) / 100.0
    if uniform_rate is None or not math.isclose(uniform_rate, expected_rate, abs_tol=1e-9):
        raise ValueError(
            f"{cost_scope_id} requires a uniform interest rate of {expected_rate}; "
            f"found {'non-uniform rates' if uniform_rate is None else uniform_rate} ({source})"
        )
    return {
        "mode": "uniform_wacc",
        "source": source,
        "override_column": "interest_rate",
        "rate": expected_rate,
    }


def build_flat_amelired_cost_scope(
    *,
    tech_yaml: Path,
    include_site_costs: bool,
    cost_scope_id: str = "flat_amelired_uniform_baseline_water_no_land_rent",
    flat_water_model_currency_per_m3: float | None = None,
    finance_csv: Path | None = None,
    interest_rate: float | None = None,
) -> dict[str, Any]:
    """Structured cost scope for a flat campaign: interest rates only (a table or one rate),
    uniform baseline water, no build/remoteness multipliers, no land rent. Finance is Ameli
    reduced WACC (``amelired``), another Ameli series by country (``ameli``) or a uniform rate
    (``waccN``, checked against the run's rates)."""
    finance_token = flat_finance_token(cost_scope_id)
    resolved_config, _ = _load_tech_chain(tech_yaml)

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

    finance = _finance_scope(finance_token, cost_scope_id, finance_csv=finance_csv, interest_rate=interest_rate)
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



def _read_site_cost_table(site_costs_csv: Path) -> list[dict[str, str]]:
    with site_costs_csv.expanduser().resolve().open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        missing = [column for column in SITE_COST_COLUMNS if column not in (reader.fieldnames or [])]
        if missing:
            raise ValueError(f"Site-cost table {site_costs_csv} lacks {', '.join(missing)}")
        rows = list(reader)
    if not rows:
        raise ValueError(f"Site-cost table {site_costs_csv} has no rows")
    return rows


def build_site_cost_scope(
    *,
    tech_yaml: Path,
    site_costs_csv: Path,
    include_site_costs: bool,
    cost_scope_id: str,
    source_water_model_currency_per_m3: float | None = None,
    finance_csv: Path | None = None,
    interest_rate: float | None = None,
) -> dict[str, Any]:
    """Structured cost scope of a run with site costs: interest rates only (a table or one rate);
    the multipliers and water price of the site-cost table, whose source price must be the
    YAML's baseline; no land rent."""
    match = SITE_COSTS_SCOPE_PATTERN.match(cost_scope_id)
    if match is None:
        raise ValueError(f"Unsupported site-cost scope: {cost_scope_id}")
    resolved_config, _ = _load_tech_chain(tech_yaml)
    source_currency = str(resolved_config.get("spatial_cost_currency", "")).strip().upper()
    if source_currency != "USD":
        raise ValueError(f"Site costs require spatial_cost_currency=USD in the resolved tech config; found {source_currency!r}")
    try:
        baseline = float(resolved_config["water_cost_baseline_usd_per_m3"])
        usage = float(resolved_config["water_usage_m3_per_t_nh3"])
        source_to_model = float(resolved_config["spatial_cost_to_model_currency"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("Resolved tech config must define the baseline water price, the water use and "
                         "spatial_cost_to_model_currency") from exc
    model_currency = str(resolved_config.get("currency", "")).strip().upper() or "EUR"
    if source_water_model_currency_per_m3 is not None and not math.isclose(
        baseline * source_to_model, float(source_water_model_currency_per_m3), rel_tol=1e-5, abs_tol=1e-8
    ):
        raise ValueError(f"The YAML's baseline water price is {baseline * source_to_model:.6f} {model_currency}/m3; "
                         f"the scenario declares {source_water_model_currency_per_m3}")
    rows = _read_site_cost_table(site_costs_csv)
    for number, row in enumerate(rows, start=2):
        if not math.isclose(float(row["water_source_cost_usd_per_m3"]), baseline, rel_tol=1e-6, abs_tol=1e-9):
            raise ValueError(f"{site_costs_csv}:{number}: the water source price is not the YAML baseline {baseline}")
        if float(row["land_cost_usd_per_km2_year"]) != 0.0:
            raise ValueError(f"{site_costs_csv}:{number}: a site-cost scope without land rent needs zero land cost")
        for column in ("equipment_cost_multiplier", "installation_multiplier", "slope_multiplier",
                       "depth_multiplier", "labour_multiplier"):
            value = float(row[column])
            if not math.isfinite(value) or value < 1.0:
                raise ValueError(f"{site_costs_csv}:{number}: {column}={value} is not a multiplier of at least 1")
        # the table rounds both prices alike, so compare with the row's own source price
        if float(row["water_cost_usd_per_m3"]) < float(row["water_source_cost_usd_per_m3"]) - 1e-9:
            raise ValueError(f"{site_costs_csv}:{number}: the water price is below the source price")
    provenance_path = site_costs_csv.with_name(site_costs_csv.stem + "_provenance.json")
    parameters = json.loads(provenance_path.read_text(encoding="utf-8")).get("parameters") if provenance_path.exists() else None
    return {
        "schema_version": 1,
        "id": cost_scope_id,
        "finance": _finance_scope(match.group(1), cost_scope_id, finance_csv=finance_csv, interest_rate=interest_rate),
        "spatial_build_and_remoteness": {
            "mode": "site_cost_table",
            "source": "site_costs_csv",
            "cell_count": len(rows),
            "parameters": parameters,
        },
        "water": {
            "mode": "site_cost_table",
            "source": "site_costs_csv.water_cost_usd_per_m3",
            "source_currency": source_currency,
            "source_price_usd_per_m3": baseline,
            "model_currency": model_currency,
            "source_to_model_currency": source_to_model,
            "source_price_model_currency_per_m3": baseline * source_to_model,
            "usage_m3_per_t_nh3": usage,
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


def build_cost_scope(
    *,
    tech_yaml: Path,
    include_site_costs: bool,
    cost_scope_id: str,
    flat_water_model_currency_per_m3: float | None = None,
    site_costs_csv: Path | None = None,
    finance_csv: Path | None = None,
    interest_rate: float | None = None,
) -> dict[str, Any]:
    """The cost scope of either kind (module docstring); a site-cost scope needs the table. The
    finance is a finance table (finance_csv) or one rate (interest_rate)."""
    if SITE_COSTS_SCOPE_PATTERN.match(cost_scope_id):
        if site_costs_csv is None:
            raise ValueError(f"{cost_scope_id} needs a site-cost table")
        return build_site_cost_scope(
            tech_yaml=tech_yaml, site_costs_csv=site_costs_csv,
            include_site_costs=include_site_costs, cost_scope_id=cost_scope_id,
            source_water_model_currency_per_m3=flat_water_model_currency_per_m3,
            finance_csv=finance_csv, interest_rate=interest_rate,
        )
    if site_costs_csv is not None:
        raise ValueError(f"{cost_scope_id} is a flat scope; a site-cost table cannot be used with it")
    return build_flat_amelired_cost_scope(
        tech_yaml=tech_yaml, include_site_costs=include_site_costs, cost_scope_id=cost_scope_id,
        flat_water_model_currency_per_m3=flat_water_model_currency_per_m3,
        finance_csv=finance_csv, interest_rate=interest_rate,
    )


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


def write_manifest(args: argparse.Namespace) -> None:
    now = datetime.now(timezone.utc).isoformat()
    output = Path(args.output)

    tech_yaml_path = Path(args.tech_yaml)
    finance_csv_path = Path(args.finance_csv) if args.finance_csv else None
    land_csv_path = Path(args.land_csv)
    locations_csv_path = Path(args.locations_csv) if args.locations_csv else None

    inputs: dict[str, Any] = {
        "tech_yaml": _tech_yaml_record(tech_yaml_path),
        "plant_bundle": _plant_record(Path(args.plant_dir)),
        "land_csv": _file_record(land_csv_path),
        "weather": _weather_record(Path(args.weather_dir)),
        "parameters": _file_record(PARAMETERS_YAML),
    }
    if finance_csv_path is not None:
        finance_record = _file_record(finance_csv_path)
        finance_record["columns"] = _csv_columns(finance_csv_path)
        finance_record["coverage"] = validate_finance_coverage(
            finance_csv=finance_csv_path, tech_yaml=tech_yaml_path, land_csv=land_csv_path, locations_csv=locations_csv_path,
        )
        inputs["finance_csv"] = finance_record
        if finance_record["coverage"]["key"] == "country":
            inputs["cell_countries"] = _file_record(CELL_COUNTRIES_CSV)
    else:
        inputs["interest_rate"] = float(args.interest_rate)
    if locations_csv_path is not None:
        inputs["locations_csv"] = _file_record(locations_csv_path)
    site_costs_path = Path(args.site_costs_csv) if args.site_costs_csv else None
    if site_costs_path is not None:
        site_record = _file_record(site_costs_path)
        site_record["coordinate_coverage"] = validate_coordinate_coverage(
            table_csv=site_costs_path, land_csv=land_csv_path, locations_csv=locations_csv_path,
        )
        inputs["site_costs_csv"] = site_record
    if args.weather_land_correction:
        inputs["weather_land_correction"] = _file_record(Path(args.weather_land_correction))

    cost_scope = build_cost_scope(
        tech_yaml=tech_yaml_path,
        finance_csv=finance_csv_path,
        interest_rate=args.interest_rate,
        include_site_costs=args.include_site_costs,
        cost_scope_id=args.cost_scope,
        flat_water_model_currency_per_m3=args.flat_water_model_currency_per_m3,
        site_costs_csv=site_costs_path,
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
            "land_constraint": args.land_constraint,
            "capacity_rule": args.capacity_rule,
            "land_allocation": args.land_allocation,
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


# --- verifying -------------------------------------------------------------------------------

class ManifestInputError(ValueError):
    """Raised when a manifest is malformed or a recorded input has changed."""


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

    _verify_file_record(inputs.get("land_csv"), "inputs.land_csv")
    verified_file_count += 1

    for name in ("finance_csv", "cell_countries", "parameters", "locations_csv", "site_costs_csv", "weather_land_correction"):
        if name in inputs:
            _verify_file_record(inputs[name], f"inputs.{name}")
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


def _verify_command(args: argparse.Namespace) -> None:
    try:
        result = verify_manifest_inputs(Path(args.manifest))
    except (ManifestInputError, OSError) as exc:
        raise SystemExit(f"Campaign input verification failed: {exc}") from exc
    print(
        "Campaign inputs verified: "
        f"{result['byte_hashed_file_count']} byte-hashed files; "
        f"{result['weather_file_count']} weather files"
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    _add_write_arguments(commands.add_parser("write", help="write the manifest of a new run directory"))
    verify = commands.add_parser("verify", help="check that a manifest's inputs have not changed")
    verify.add_argument("--manifest", required=True)
    args = parser.parse_args(argv)
    if args.command == "write":
        write_manifest(args)
    else:
        _verify_command(args)


if __name__ == "__main__":
    main()
