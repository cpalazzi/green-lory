#!/usr/bin/env python3
"""Read the run definitions in runs.yaml.

    python arc/run_definitions.py list           # run ids, runner and role
    python arc/run_definitions.py table          # green-lory settings side by side (Markdown)
    python arc/run_definitions.py shell <run id> # shell assignments for arc/submit_lory_sequence.sh
"""

from __future__ import annotations

import argparse
from pathlib import Path
import shlex
import sys
from typing import Any, Mapping

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
RUNS_FILE = REPO_ROOT / "runs.yaml"

RUNNERS = ("green_lory", "derived", "legacy_lcoa", "supply_pilot")

# runs.yaml key -> variable set by configure_scenario in arc/submit_lory_sequence.sh
WRAPPER_VARIABLES = {
    "results_class": "SCENARIO_CLASS",
    "job_token": "SCENARIO_JOB_TOKEN",
    "description": "SCENARIO_DESCRIPTION",
    "tech_yaml": "TECH_YAML",
    "plant_dir": "PLANT_DIR",
    "pv_policy": "PV_POLICY",
    "finance_csv": "OVERRIDE_CSV",
    "finance_csv_sha256": "EXPECTED_OVERRIDE_CSV_SHA256",
    "time_step_h": "TIME_STEP",
    "temporal_accounting": "TEMPORAL_ACCOUNTING_MODE",
    "ramp_limit_basis": "RAMP_LIMIT_BASIS",
    "site_costs_in_headline": "INCLUDE_SITE_COSTS",
    "water_eur2020_per_m3": "FLAT_WATER_MODEL_PER_M3",
    "cost_scope": "COST_SCOPE",
    "land_share": "EXPECTED_LAND_COMPETITION_FRACTION",
    "land_table_sha256": "EXPECTED_LAND_TABLE_SHA256",
    "land_constraint": "LAND_CONSTRAINT",
    "capacity_rule": "CAPACITY_RULE",
    "land_allocation": "LAND_ALLOCATION",
}

ALLOWED_VALUES = {
    "pv_policy": {"both", "fixed"},
    "temporal_accounting": {"snapshot_weighted", "legacy_scaled"},
    "ramp_limit_basis": {"per_hour", "legacy_per_snapshot"},
    "land_constraint": {"after_solve", "in_solve"},
    "capacity_rule": {"scaled_reference_design", "solved_quantity"},
    "land_allocation": {"colocated", "exclusive"},
}

TABLE_KEYS = (
    "role",
    "tech_yaml",
    "plant_dir",
    "pv_policy",
    "finance_csv",
    "time_step_h",
    "temporal_accounting",
    "ramp_limit_basis",
    "site_costs_in_headline",
    "water_eur2020_per_m3",
    "land_share",
    "land_table",
    "land_constraint",
    "capacity_rule",
    "land_allocation",
)


class RunDefinitionError(ValueError):
    """Raised when runs.yaml is malformed or a run id is not usable."""


def load_runs(path: str | Path = RUNS_FILE) -> dict[str, dict[str, Any]]:
    with Path(path).open(encoding="utf-8") as handle:
        document = yaml.safe_load(handle)
    runs = (document or {}).get("runs")
    if not isinstance(runs, dict) or not runs:
        raise RunDefinitionError(f"{path} has no 'runs' mapping")
    for run_id, run in runs.items():
        if not isinstance(run, dict):
            raise RunDefinitionError(f"Run {run_id} is not a mapping")
        if run.get("runner") not in RUNNERS:
            raise RunDefinitionError(f"Run {run_id}: runner must be one of {RUNNERS}")
        if run["runner"] == "green_lory":
            _validate_green_lory_run(run_id, run)
    return runs


def _validate_green_lory_run(run_id: str, run: Mapping[str, Any]) -> None:
    missing = sorted(set(WRAPPER_VARIABLES) - set(run))
    if missing:
        raise RunDefinitionError(f"Run {run_id} is missing {', '.join(missing)}")
    for key, allowed in ALLOWED_VALUES.items():
        if run[key] not in allowed:
            raise RunDefinitionError(f"Run {run_id}: {key}={run[key]!r} is not one of {sorted(allowed)}")
    if not isinstance(run["site_costs_in_headline"], bool):
        raise RunDefinitionError(f"Run {run_id}: site_costs_in_headline must be true or false")
    if not 0.0 < float(run["land_share"]) <= 1.0:
        raise RunDefinitionError(f"Run {run_id}: land_share must lie in (0, 1]")
    if float(run["time_step_h"]) <= 0 or 8760 % float(run["time_step_h"]):
        raise RunDefinitionError(f"Run {run_id}: time_step_h must divide 8760")


def _shell_value(key: str, value: Any) -> str:
    if key == "time_step_h":
        return f"{float(value):.1f}"
    if key == "site_costs_in_headline":
        return "1" if value else "0"
    if key == "water_eur2020_per_m3":
        return repr(float(value))
    if key == "land_share":
        return f"{float(value):.2f}"
    return " ".join(str(value).split())


def shell_assignments(run_id: str, runs: Mapping[str, Mapping[str, Any]] | None = None) -> str:
    runs = load_runs() if runs is None else runs
    if run_id not in runs:
        raise RunDefinitionError(f"Unknown run id {run_id!r}; runs.yaml defines {', '.join(sorted(runs))}")
    run = runs[run_id]
    if run["runner"] != "green_lory":
        raise RunDefinitionError(
            f"Run {run_id} has runner {run['runner']!r}; only green_lory runs are submitted by this wrapper"
        )
    lines = [
        f"{variable}={shlex.quote(_shell_value(key, run[key]))}"
        for key, variable in WRAPPER_VARIABLES.items()
    ]
    return "\n".join(lines)


def markdown_table(runs: Mapping[str, Mapping[str, Any]] | None = None) -> str:
    runs = load_runs() if runs is None else runs
    ids = [run_id for run_id, run in runs.items() if run["runner"] == "green_lory"]
    lines = ["| setting | " + " | ".join(f"`{run_id}`" for run_id in ids) + " |"]
    lines.append("|---|" + "---|" * len(ids))
    for key in TABLE_KEYS:
        cells = [" ".join(str(runs[run_id].get(key, "")).split()) for run_id in ids]
        lines.append(f"| {key} | " + " | ".join(cells) + " |")
    others = [(run_id, run) for run_id, run in runs.items() if run["runner"] != "green_lory"]
    if others:
        lines.append("")
        lines.extend(f"- `{run_id}` ({run['runner']}): {run.get('role', '')}" for run_id, run in others)
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs", type=Path, default=RUNS_FILE, help="run-definition file (default: runs.yaml)")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("list")
    commands.add_parser("table")
    shell = commands.add_parser("shell")
    shell.add_argument("run_id")
    args = parser.parse_args(argv)
    try:
        runs = load_runs(args.runs)
        if args.command == "list":
            for run_id, run in runs.items():
                print(f"{run_id}\t{run['runner']}\t{run.get('role', '')}")
        elif args.command == "table":
            print(markdown_table(runs))
        else:
            print(shell_assignments(args.run_id, runs))
    except (RunDefinitionError, OSError, yaml.YAMLError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
