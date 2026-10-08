"""Run definitions (runs.yaml) and the execution record (run_store.csv).

runs.yaml says what each run is: every setting, by run id, for plant surfaces (`runs`) and
shipping networks (`networks`, with `network_defaults` filled in). run_store.csv says when and
where each run ran: jobs, release, status, output location, headline result. The command line
reads both (`green-lory list | show | table | status | path`, see model/cli.py).

This module only needs PyYAML, so the ARC wrappers can read run definitions with a minimal
Python.
"""

from __future__ import annotations

import csv
import shlex
from pathlib import Path
from typing import Any, Mapping

import yaml

from .config import REPO_ROOT

RUNS_FILE = REPO_ROOT / "runs.yaml"
RUN_STORE_FILE = REPO_ROOT / "run_store.csv"
# Where runs.yaml land tables (land_table: <campaign>/<file>) live on a laptop; on ARC the
# campaign root is green-lory-campaigns/
LOCAL_CAMPAIGNS_DIR = REPO_ROOT / "results" / "campaigns"

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
    "site_costs_csv": "SITE_COSTS_CSV",
    "site_costs_csv_sha256": "EXPECTED_SITE_COSTS_CSV_SHA256",
    "weather_land_correction": "WEATHER_LAND_CORRECTION",
    "weather_land_correction_sha256": "EXPECTED_WEATHER_LAND_CORRECTION_SHA256",
}
# Keys a green_lory run may leave out (the wrapper then gets an empty value): runs without site
# costs use flat build costs and the YAML's uniform water price; runs without a land correction
# use the 1-degree profiles as they are (every run before October 2026)
OPTIONAL_KEYS = ("site_costs_csv", "site_costs_csv_sha256", "weather_land_correction", "weather_land_correction_sha256")

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
    "weather_land_correction",
    "site_costs_csv",
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
    missing = sorted(set(WRAPPER_VARIABLES) - set(OPTIONAL_KEYS) - set(run))
    if missing:
        raise RunDefinitionError(f"Run {run_id} is missing {', '.join(missing)}")
    for key in ("site_costs_csv", "weather_land_correction"):
        if bool(run.get(key)) != bool(run.get(f"{key}_sha256")):
            raise RunDefinitionError(f"Run {run_id}: {key} and {key}_sha256 go together")
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
        f"{variable}={shlex.quote(_shell_value(key, run[key]) if key in run else '')}"
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


# ---------------------------------------------------------------------------------------------
# The execution record
# ---------------------------------------------------------------------------------------------

def load_run_store(path: str | Path = RUN_STORE_FILE) -> list[dict[str, str]]:
    """Every row of run_store.csv, in file order."""
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def run_record(run_id: str, path: str | Path = RUN_STORE_FILE) -> dict[str, str]:
    """The run store row of one run id."""
    for row in load_run_store(path):
        if row["run_id"] == run_id:
            return row
    raise RunDefinitionError(f"{run_id!r} has no row in {path}")


# ---------------------------------------------------------------------------------------------
# Running a defined surface locally
# ---------------------------------------------------------------------------------------------

def surface_arguments(run_id: str, *, campaigns_dir: str | Path = LOCAL_CAMPAIGNS_DIR,
                      runs: Mapping[str, Mapping[str, Any]] | None = None) -> dict[str, Any]:
    """Keyword arguments of model.surface.run_surface for a green_lory run of runs.yaml.

    Checks the pinned hashes of the land table, the finance file and the site-cost table.
    """
    import hashlib

    from .surface import RunSettings  # heavy imports only when a run is asked for

    runs = load_runs() if runs is None else runs
    if run_id not in runs or runs[run_id]["runner"] != "green_lory":
        raise RunDefinitionError(f"{run_id!r} is not a green_lory run of runs.yaml")
    run = runs[run_id]
    land_csv = Path(campaigns_dir) / run["land_table"]
    finance_csv = REPO_ROOT / run["finance_csv"]
    site_costs_csv = REPO_ROOT / run["site_costs_csv"] if run.get("site_costs_csv") else None
    pinned = [("land table", land_csv, run["land_table_sha256"]), ("finance CSV", finance_csv, run["finance_csv_sha256"])]
    if site_costs_csv is not None:
        pinned.append(("site-cost table", site_costs_csv, run["site_costs_csv_sha256"]))
    land_correction = REPO_ROOT / run["weather_land_correction"] if run.get("weather_land_correction") else None
    if land_correction is not None:
        pinned.append(("weather land correction", land_correction, run["weather_land_correction_sha256"]))
    for label, path, expected in pinned:
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != expected:
            raise RunDefinitionError(f"{run_id}: {label} {path} has sha256 {actual}; runs.yaml pins {expected}")
    return {
        "land_csv": land_csv,
        "tech_yaml": REPO_ROOT / run["tech_yaml"],
        "plant_dir": REPO_ROOT / run["plant_dir"],
        "override_csv": finance_csv,
        "site_costs_csv": site_costs_csv,
        "weather_dir": REPO_ROOT / run["weather_dir"],
        "weather_land_correction": land_correction,
        "settings": RunSettings(
            time_step=float(run["time_step_h"]),
            pv_policy=run["pv_policy"],
            land_constraint=run["land_constraint"],
            capacity_rule=run["capacity_rule"],
            land_allocation=run["land_allocation"],
            temporal_accounting_mode=run["temporal_accounting"],
            ramp_limit_basis=run["ramp_limit_basis"],
            include_site_costs=bool(run["site_costs_in_headline"]),
        ),
    }
