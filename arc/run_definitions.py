#!/usr/bin/env python3
"""Read the run definitions in runs.yaml.

    python arc/run_definitions.py list           # run and network ids, runner and role
    python arc/run_definitions.py show <id>      # one entry; also accepts a report name such as D
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

# supplier_run of a network that reads the published table rather than one of our runs
ARCHIVED_SUPPLIER = "archived"

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
    "weather_dir": "WEATHER_DIR",
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
    "weather_dir",
)


class RunDefinitionError(ValueError):
    """Raised when runs.yaml is malformed or a run id is not usable."""


def _load_document(path: str | Path) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def load_runs(path: str | Path = RUNS_FILE) -> dict[str, dict[str, Any]]:
    """Run entries with run_defaults filled in, each checked."""
    document = _load_document(path)
    entries = document.get("runs")
    defaults = document.get("run_defaults") or {}
    if not isinstance(entries, dict) or not entries:
        raise RunDefinitionError(f"{path} has no 'runs' mapping")
    if not isinstance(defaults, dict):
        raise RunDefinitionError(f"{path}: run_defaults must be a mapping")
    runs = {}
    for run_id, run in entries.items():
        if not isinstance(run, dict):
            raise RunDefinitionError(f"Run {run_id} is not a mapping")
        # the entry's own keys first, then the defaults it does not override
        run = {**run, **{key: value for key, value in defaults.items() if key not in run}}
        if run.get("runner") not in RUNNERS:
            raise RunDefinitionError(f"Run {run_id}: runner must be one of {RUNNERS}")
        if run["runner"] == "green_lory":
            _validate_green_lory_run(run_id, run)
        if run["runner"] == "derived" and run.get("derived_from") not in entries:
            raise RunDefinitionError(f"Run {run_id}: derived_from must name another run in {path}")
        runs[run_id] = run
    for run_id in runs:
        chain = [run_id]
        while runs[chain[-1]]["runner"] == "derived":
            parent = runs[chain[-1]]["derived_from"]
            if parent in chain:
                raise RunDefinitionError(f"Run {run_id}: derived_from forms a cycle ({' <- '.join(chain + [parent])})")
            chain.append(parent)
    return runs


def load_networks(path: str | Path = RUNS_FILE) -> dict[str, dict[str, Any]]:
    """Network entries with network_defaults filled in, checked against the runs they read."""
    document = _load_document(path)
    runs = load_runs(path)
    defaults = document.get("network_defaults") or {}
    networks = document.get("networks") or {}
    if not isinstance(defaults, dict) or not isinstance(networks, dict):
        raise RunDefinitionError(f"{path}: network_defaults and networks must be mappings")
    clashes = sorted(set(runs) & set(networks))
    if clashes:
        raise RunDefinitionError(f"{path}: ids used for both a run and a network: {', '.join(clashes)}")
    resolved, report_names = {}, {}
    for network_id, network in networks.items():
        if not isinstance(network, dict):
            raise RunDefinitionError(f"Network {network_id} is not a mapping")
        supplier = network.get("supplier_run")
        if supplier != ARCHIVED_SUPPLIER and supplier not in runs:
            raise RunDefinitionError(
                f"Network {network_id}: supplier_run {supplier!r} is neither {ARCHIVED_SUPPLIER!r} nor a run in {path}"
            )
        name = network.get("report_name")
        if name is not None:
            if str(name).casefold() in report_names:
                raise RunDefinitionError(f"{path}: report_name {name!r} is used twice")
            report_names[str(name).casefold()] = network_id
        # the entry's own keys first, then the defaults it does not override
        resolved[network_id] = {**network, **{key: value for key, value in defaults.items() if key not in network}}
    return resolved


def find_entry(
    key: str,
    runs: Mapping[str, Mapping[str, Any]],
    networks: Mapping[str, Mapping[str, Any]],
) -> tuple[str, Mapping[str, Any]]:
    """Look an entry up by run id, network id or report name ("D", "A2", "archival replay")."""
    if key in runs:
        return key, runs[key]
    if key in networks:
        return key, networks[key]
    for network_id, network in networks.items():
        if str(network.get("report_name", "")).casefold() == key.casefold():
            return network_id, network
    raise RunDefinitionError(f"{key!r} is not a run id, network id or report name in runs.yaml")


def _lineage(run_id: str, runs: Mapping[str, Mapping[str, Any]]) -> list[str]:
    chain = [run_id]
    while run_id in runs and runs[run_id].get("runner") == "derived":
        run_id = runs[run_id]["derived_from"]
        chain.append(run_id)
    return chain


def describe_entry(
    key: str,
    runs: Mapping[str, Mapping[str, Any]],
    networks: Mapping[str, Mapping[str, Any]],
) -> str:
    entry_id, entry = find_entry(key, runs, networks)
    text = yaml.safe_dump({entry_id: dict(entry)}, sort_keys=False, width=100, allow_unicode=True).rstrip()
    if entry_id in networks:
        chain = _lineage(entry["supplier_run"], runs)
        steps = [f"{step} ({runs[step].get('role', '')})" if step in runs else step for step in chain]
        text += "\n\n# supplier table: " + "\n#   derived from ".join(steps)
    elif entry.get("runner") == "derived":
        text += "\n\n# derived from: " + " <- ".join(_lineage(entry_id, runs)[1:])
    return text


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


def markdown_table(
    runs: Mapping[str, Mapping[str, Any]] | None = None,
    networks: Mapping[str, Mapping[str, Any]] | None = None,
) -> str:
    runs = load_runs() if runs is None else runs
    networks = {} if networks is None else networks
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
    if networks:
        lines.append("")
        lines.extend(
            f"- `{network_id}` (network{_report_suffix(network)}): {network.get('role', '')}; "
            f"supplier table `{network['supplier_run']}`"
            for network_id, network in networks.items()
        )
    return "\n".join(lines)


def _report_suffix(network: Mapping[str, Any]) -> str:
    name = network.get("report_name")
    return f', "{name}" in the September reports' if name else ""


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs", type=Path, default=RUNS_FILE, help="run-definition file (default: runs.yaml)")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("list")
    show = commands.add_parser("show")
    show.add_argument("id", help="run id, network id or report name (for example D or A2)")
    commands.add_parser("table")
    shell = commands.add_parser("shell")
    shell.add_argument("run_id")
    args = parser.parse_args(argv)
    try:
        runs = load_runs(args.runs)
        networks = load_networks(args.runs)
        if args.command == "list":
            for run_id, run in runs.items():
                print(f"{run_id}\t{run['runner']}\t{run.get('role', '')}")
            for network_id, network in networks.items():
                print(f"{network_id}\tnetwork{_report_suffix(network)}\t{network.get('role', '')}")
        elif args.command == "show":
            print(describe_entry(args.id, runs, networks))
        elif args.command == "table":
            print(markdown_table(runs, networks))
        else:
            print(shell_assignments(args.run_id, runs))
    except (RunDefinitionError, OSError, yaml.YAMLError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
