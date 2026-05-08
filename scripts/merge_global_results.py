#!/usr/bin/env python3
"""Merge quadrant global-run CSVs into a canonical scenario output."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


REQUIRED_COLUMNS = [
    "max_gridless_ammonia_capacity_t",
    "max_gridless_ammonia_capacity_mtpa",
    "gridless_capacity_scale_factor",
    "protected_area_pct",
    "slope_suitable_land_pct",
    "land_exclusion_factor",
]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge quadrant global-run CSVs into a canonical scenario CSV.",
    )
    parser.add_argument(
        "run_label",
        help="Base run label, for example way-2050-flat or dea-2050-spatial.",
    )
    parser.add_argument(
        "--results-dir",
        default=str(REPO_ROOT / "results"),
        help="Results root directory containing <run_label>-*/run_global_*.csv folders.",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Destination merged CSV path.",
    )
    parser.add_argument(
        "--expected-quadrants",
        type=int,
        default=4,
        help="Expected number of quadrant CSVs before merging.",
    )
    parser.add_argument(
        "--allow-missing-quadrants",
        action="store_true",
        help="Allow merging fewer files than --expected-quadrants.",
    )
    parser.add_argument(
        "--required-column",
        action="append",
        default=[],
        help="Additional required column name. May be passed multiple times.",
    )
    return parser.parse_args()


def _find_quadrant_files(results_dir: Path, run_label: str) -> list[Path]:
    latest_by_quadrant: dict[Path, Path] = {}
    for path in sorted(results_dir.glob(f"{run_label}-*/run_global_*.csv")):
        latest_by_quadrant[path.parent] = path
    return sorted(latest_by_quadrant.values())


def main() -> None:
    args = _parse_args()
    results_dir = Path(args.results_dir).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()

    quadrant_files = _find_quadrant_files(results_dir, args.run_label)
    if not quadrant_files:
        raise FileNotFoundError(
            f"No quadrant CSVs found for '{args.run_label}' under {results_dir}"
        )
    if not args.allow_missing_quadrants and len(quadrant_files) != args.expected_quadrants:
        raise RuntimeError(
            f"Expected {args.expected_quadrants} quadrant CSVs for '{args.run_label}', found {len(quadrant_files)}"
        )

    frames = [pd.read_csv(path) for path in quadrant_files]
    combined = pd.concat(frames, ignore_index=True)

    before = len(combined)
    combined = combined.drop_duplicates(subset=["latitude", "longitude"], keep="last")
    combined = combined.sort_values(["latitude", "longitude"]).reset_index(drop=True)

    required_columns = REQUIRED_COLUMNS + list(args.required_column)
    missing_columns = [column for column in required_columns if column not in combined.columns]
    if missing_columns:
        raise KeyError(
            "Merged CSV missing required columns: " + ", ".join(missing_columns)
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(output_path, index=False)

    print(f"Merged {len(quadrant_files)} files for {args.run_label}")
    print(f"Rows: {before} -> {len(combined)} after dedupe")
    print(f"Output: {output_path}")
    print("Validated required columns:")
    for column in required_columns:
        print(f"  - {column}")


if __name__ == "__main__":
    main()