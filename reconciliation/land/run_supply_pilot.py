#!/usr/bin/env python3
"""Full-year cost–quantity points under a fixed, shared renewable land budget."""
from __future__ import annotations

import argparse
import copy
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
import json
import multiprocessing
import os
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from arc.release_source_inventory import _sha256 as sha, verify_inventory
from arc.validate_land_campaign_input import validate_land_campaign_input
from model import main as plant_main, location_tools as lt
from model.constants import AMMONIA_HHV_MWH_PER_T
from model import data_paths
from model.data_store import _apply_unit_suffixes
from model.run_global import (
    _load_tech_inputs, _load_tech_meta, _load_land_table, _load_interest_table,
    _build_land_lookup, _build_interest_lookup, _apply_spatial_solar_density_from_tech_config,
    _run_single_location,
)
from reconciliation.land.stage_pv_pilot import TECH_YAML, tech_yaml_dependencies

STATE = {}
ROBUST_GUROBI_OPTIONS = {"Method": 1, "DualReductions": 0,
                         "NumericFocus": 3, "InfUnbdInfo": 1}


def write_json(path, payload):
    path.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")


def validate_bathymetry_input(path, expected_sha256):
    """The weather loader needs this file even for selected onshore cells."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Weather-loader bathymetry dependency missing: {path}")
    if sha(path) != expected_sha256:
        raise ValueError("Bathymetry differs from the controlled weather run")
    return path


def is_proven_infeasible(error):
    """Do not confuse setup, licence, numerical or ambiguous solver failures with infeasibility."""
    while error is not None:
        if isinstance(error, plant_main.OptimizationFailure):
            return error.condition == "infeasible"
        error = error.__cause__
    return False


def solver_failure(error):
    while error is not None:
        if isinstance(error, plant_main.OptimizationFailure):
            return error
        error = error.__cause__
    return None


def run_location_with_retry(*args, attempts, **kwargs):
    """One fresh-network retry for identified numerical/ambiguous terminations."""
    for index, options in enumerate((None, ROBUST_GUROBI_OPTIONS)):
        attempt = {"solver_options_override": options}
        try:
            status, result = _run_single_location(*args, **kwargs, solver_options_override=options)
        except RuntimeError as error:
            failure = solver_failure(error)
            if failure is None:
                raise
            attempt.update(status=failure.status, termination=failure.condition)
            attempts.append(attempt)
            if (index == 0 and os.environ.get("GREEN_LORY_SOLVER", "gurobi").lower() == "gurobi"
                    and failure.condition in {"suboptimal", "numeric", "numerical", "infeasible_or_unbounded"}):
                print(f"RETRY numerical termination {failure.condition}: dual simplex", flush=True)
                continue
            raise
        if status != "done" or result is None:
            raise ValueError("Missing solved point")
        plant_main.require_optimal_termination(result["solver_status"], result["solver_termination"])
        attempt.update(status=result["solver_status"], termination=result["solver_termination"])
        attempts.append(attempt)
        return status, result
    raise AssertionError("Exhausted numerical retry without a result or exception")


def set_quantity(base, quantity_mtpa):
    if not np.isfinite(quantity_mtpa) or quantity_mtpa <= 0:
        raise ValueError("Quantity must be positive and finite")
    if list(base.loads.index) != ["ammonia"] or not base.loads_t.p_set.empty:
        raise ValueError("Pilot requires one constant ammonia load")
    network = copy.deepcopy(base)
    network.loads.at["ammonia", "p_set"] = quantity_mtpa * 1e6 * AMMONIA_HHV_MWH_PER_T / 8760
    return network


PV_POLICIES = ("both", "fixed-only", "tracking-only")


def apply_pv_policy(network, policy):
    """Disable the PV technology excluded by the policy before any solve."""
    if policy not in PV_POLICIES:
        raise ValueError("Unknown PV policy")
    excluded = {"both": None, "fixed-only": "solar_tracking", "tracking-only": "solar"}[policy]
    if excluded is not None:
        if excluded not in network.generators.index:
            raise ValueError(f"Missing {excluded} generator")
        network.generators.loc[excluded, "p_nom"] = 0.
        network.generators.loc[excluded, "p_nom_extendable"] = False
        network.generators.loc[excluded, "p_nom_max"] = 0.
    return network


def validate_result(result, quantity, land, *, constrained):
    expected = quantity * 1e6
    for name in ("annual_ammonia_production_t", "gridless_ammonia_production_t"):
        np.testing.assert_allclose(result[name], expected, rtol=1e-7, atol=1)
    for name, value in {"snapshot_hours": 1, "simulated_hours": 8760,
                        "build_cost_multiplier": 1, "water_cost_usd_per_m3": 2,
                        "land_cost_usd_per_km2_year": 0}.items():
        np.testing.assert_allclose(result[name], value, rtol=1e-8, atol=1e-6)
    for name, value in {"currency": "EUR", "ramp_limit_basis": "per_hour",
                        "temporal_accounting_mode": "snapshot_weighted",
                        "land_allocation": STATE.get("land_allocation", "exclusive")}.items():
        if result[name] != value:
            raise ValueError(f"Unexpected {name}")
    if not result["is_gridless_feasible"] or abs(result["grid_energy_mwh"]) > result["grid_energy_tolerance_mwh"]:
        raise ValueError("Grid use in supply point")
    if abs(result["accumulated_penalty_mwh"]) > 1:
        raise ValueError("Feasibility slack in supply point")
    if not np.isfinite(result["lcoa_eur_per_t"]) or result["lcoa_eur_per_t"] <= 0:
        raise ValueError("Invalid LCOA")
    np.testing.assert_allclose(result["total_cost_eur_per_year"],
        result["plant_objective_cost_eur_per_year"] + result["water_cost_eur_per_year"] +
        result["land_cost_eur_per_year"], rtol=1e-10, atol=1e-3)
    np.testing.assert_allclose(result["lcoa_eur_per_t"],
        result["total_cost_eur_per_year"] / result["annual_ammonia_production_t"], rtol=1e-10)
    np.testing.assert_allclose(result["headline_cost_share_total_pct"], 100, atol=1e-8, rtol=0)
    np.testing.assert_allclose(result["solar_density_mw_per_km2"], land.solar_density_mw_per_km2, rtol=1e-12)
    tech = STATE["tech"]
    tracking_ratio = tech["solar_tracking"]["land_use_km2_per_mw"] / tech["solar"]["land_use_km2_per_mw"]
    wind_density = 1. / tech["wind"]["land_use_km2_per_mw"]
    exclusive_fraction = 1. if STATE.get("land_allocation", "exclusive") == "exclusive" else float(STATE["config"]["wind_land_exclusive_fraction"])
    np.testing.assert_allclose(result["wind_land_exclusive_fraction_effective"], exclusive_fraction, rtol=1e-12)
    wind = max(0., result["wind_mw"]) / wind_density
    solar = (max(0., result["solar_mw"]) + tracking_ratio * max(0., result["solar_tracking_mw"])) / land.solar_density_mw_per_km2
    shared = solar + exclusive_fraction * wind
    if constrained:
        if result["capacity_rule"] != "solved_quantity" or result["quantity_is_maximum"]:
            raise ValueError("Wrong constrained-quantity output contract")
        if any(key.startswith("scaled_design_max_") or key.startswith("max_ammonia_capacity_") for key in result):
            raise ValueError("Constrained point incorrectly carries scaled maximum capacity")
        for used, area in ((wind, land.wind_onshore_area_km2), (solar, land.solar_area_km2),
                           (shared, land.renewable_union_area_km2)):
            if used > area + max(1e-5, area * 1e-6):
                raise ValueError("Independent land budget check failed")
    return {"wind_footprint_km2": wind, "pv_footprint_km2": solar, "shared_budget_used_km2": shared,
            "union_used_fraction": shared / land.renewable_union_area_km2}


def solve_point(site, quantity, directory, *, constrained):
    directory.mkdir(parents=True, exist_ok=False)
    lat, lon = site["latitude"], site["longitude"]
    start = datetime.now(timezone.utc).isoformat()
    print(f"START {site['id']} {'land' if constrained else 'control'} {quantity:.6g} Mt/year", flush=True)
    network = set_quantity(STATE["base"], quantity)
    record = {"site": site["id"], "latitude": lat, "longitude": lon,
              "target_Mtpa": quantity, "constrained": constrained, "started_utc": start}
    record["solver_attempts"] = []
    try:
        status, result = run_location_with_retry(
            lat, lon, STATE["weather"], STATE["interest"], STATE["tech"], STATE["meta"],
            STATE["land_lookup"], network, aggregation_count=1, time_step=1.,
            max_snapshots=None, quiet=True, fail_fast=True,
            land_constraint="in_solve" if constrained else "after_solve",
            capacity_rule="solved_quantity" if constrained else "scaled_reference_design",
            land_allocation=STATE.get("land_allocation", "exclusive"), allow_conservative_union_fallback=False,
            temporal_accounting_mode="snapshot_weighted", ramp_limit_basis="per_hour",
            include_site_costs=True,
            attempts=record["solver_attempts"],
        )
    except RuntimeError as error:
        if not constrained or not is_proven_infeasible(error):
            raise
        record.update(status="infeasible", solver_termination="infeasible",
                      finished_utc=datetime.now(timezone.utc).isoformat())
        write_json(directory / "status.json", record)
        print(f"INFEASIBLE {site['id']} {quantity:.6g} Mt/year", flush=True)
        return record
    if status != "done" or result is None:
        raise ValueError("Missing solved point")
    # _run_single_location returns internal component names; run_global's
    # Data_store normally performs this conversion before exposing CSV results.
    result = _apply_unit_suffixes(result)
    metrics = validate_result(result, quantity, STATE["land_lookup"][(lat, lon)], constrained=constrained)
    if STATE.get("pv_policy", "both") == "fixed-only" and abs(result["solar_tracking_mw"]) > 1e-8:
        raise ValueError("Tracking capacity in fixed-only experiment")
    if STATE.get("pv_policy", "both") == "tracking-only" and abs(result["solar_mw"]) > 1e-8:
        raise ValueError("Fixed capacity in tracking-only experiment")
    result.update(latitude=lat, longitude=lon, target_Mtpa=quantity, site=site["id"])
    raw = directory / "result.csv"
    pd.DataFrame([result]).to_csv(raw, index=False)
    record.update(status="feasible", finished_utc=datetime.now(timezone.utc).isoformat(),
                  result_sha256=sha(raw), LCOA_EUR2020_per_t=float(result["lcoa_eur_per_t"]),
                  annual_cost_EUR2020=float(result["total_cost_eur_per_year"]),
                  solar_MW=float(result["solar_mw"]), tracking_MW=float(result["solar_tracking_mw"]),
                  wind_MW=float(result["wind_mw"]), **metrics)
    write_json(directory / "status.json", record)
    print(f"DONE {site['id']} {quantity:.6g} Mt/year: {record['LCOA_EUR2020_per_t']:.3f} EUR/t", flush=True)
    return record


def run_site(site):
    directory = STATE["output"] / site["id"]
    control = solve_point(site, 1., directory / "control_unconstrained", constrained=False)
    expected_control = site.get("control_LCOA_EUR2020_per_t")
    if expected_control is None:
        print(f"CONTROL {site['id']} recorded without a reference: {control['LCOA_EUR2020_per_t']:.3f} EUR/t", flush=True)
    elif abs(control["LCOA_EUR2020_per_t"] - expected_control) > STATE["config"]["control_lcoa_tolerance_eur_per_t"]:
        raise ValueError(f"Unconstrained control changed at {site['id']}")
    points = []
    for quantity in site["quantities_Mtpa"]:
        key = format(quantity, ".12g").replace(".", "p")
        points.append(solve_point(site, quantity, directory / ("q_" + key), constrained=True))
    feasible = [p for p in points if p["status"] == "feasible"]
    infeasible = [p for p in points if p["status"] == "infeasible"]
    if not feasible:
        raise ValueError(f"No feasible supply point at {site['id']}")
    if infeasible and max(p["target_Mtpa"] for p in feasible) >= min(p["target_Mtpa"] for p in infeasible):
        raise ValueError("Non-monotonic feasibility")
    for point in feasible:
        if point["LCOA_EUR2020_per_t"] < control["LCOA_EUR2020_per_t"] - .15:
            raise ValueError("Constrained cost below unconstrained minimum")
    for first, second in zip(feasible, feasible[1:]):
        if second["LCOA_EUR2020_per_t"] < first["LCOA_EUR2020_per_t"] - .15:
            raise ValueError("Non-monotonic average cost")
    summary = {"site": site["id"], "qa_pass": True, "control": control, "points": points,
               "maximum_feasible_tested_Mtpa": max(p["target_Mtpa"] for p in feasible),
               "minimum_infeasible_tested_Mtpa": min((p["target_Mtpa"] for p in infeasible), default=None),
               "warning": "Tested quantities bracket feasibility; no exact maximum-capacity claim."}
    write_json(directory / "summary.json", summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("land", "weather-run", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--experiment-config", type=Path,
                        default=ROOT / "reconciliation/land/supply_curve/config_v1.json")
    parser.add_argument("--pv-policy", choices=list(PV_POLICIES), default="both")
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    release = verify_inventory(ROOT, ROOT / "source_inventory.json")
    config_path = args.experiment_config.resolve()
    if not config_path.is_relative_to(ROOT):
        raise ValueError("Experiment config must be in the pinned source release")
    config = json.loads(config_path.read_text())
    if config.get("pv_policy", "both") != args.pv_policy:
        raise ValueError("PV policy differs from experiment config")
    if sha(args.land) != config["land_sha256"]:
        raise ValueError("Land differs from completed common-land pilot")
    land_qa = validate_land_campaign_input(args.land, expected_land_fraction=.02)
    tech_path = ROOT / TECH_YAML
    tech = _load_tech_inputs(tech_path)
    if not np.isclose(tech["solar_tracking"]["land_use_km2_per_mw"] / tech["solar"]["land_use_km2_per_mw"],
                      config["tracking_footprint_ratio"], rtol=1e-9, atol=0):
        raise ValueError("Changed tracking packing ratio")
    land_allocation = config.get("land_allocation", "exclusive")
    if land_allocation not in {"colocated", "exclusive"}:
        raise ValueError("Experiment config must declare land_allocation colocated or exclusive")
    if land_allocation == "colocated":
        declared = float(config.get("wind_land_exclusive_fraction", -1))
        if not np.isclose(declared, float(tech["wind"].get("land_use_exclusive_fraction", 0.03)), rtol=1e-9, atol=0):
            raise ValueError("wind_land_exclusive_fraction in the config differs from the technology YAML")
    previous = json.loads((args.weather_run / "manifest.json").read_text())
    previous_inputs = {Path(f["path"]).name: f["sha256"] for f in previous["inputs"]}
    bathymetry = validate_bathymetry_input(
        data_paths.BATHYMETRY_FILE, previous_inputs["model_bathymetry.nc"])
    plant = ROOT / "basic_ammonia_plant_2050_way_tracking"
    files = [*tech_yaml_dependencies(tech_path, ROOT), ROOT / "inputs/amelired_interest_inputs_2050.csv",
             *sorted(plant.glob("*.csv")), bathymetry]
    tech_yaml_changed = []
    for path in files:
        if path.name == "generators.csv" or sha(path) == previous_inputs[path.name]:
            continue
        # The technology overlay may carry revised land-use parameters (tracking footprint
        # ratio, wind exclusive fraction) when the experiment config pins its exact hash; the
        # plant bundle CSVs that hold the solver's costs must still match the reference run.
        if path.name == TECH_YAML.name and config.get("tech_yaml_sha256") == sha(path):
            tech_yaml_changed.append({"path": str(path.resolve()), "sha256": sha(path),
                                      "reference_sha256": previous_inputs[path.name],
                                      "note": "land-use parameters revised; pinned by experiment config"})
            continue
        raise ValueError(f"Changed controlled input: {path.name}")
    generators = pd.read_csv(plant / "generators.csv").set_index("name")
    if not generators.loc[["wind", "solar", "solar_tracking"], "p_nom_extendable"].all():
        raise ValueError("Pilot must permit wind and both PV options (the policy disables one at run time)")
    if generators.loc["grid", "p_nom_extendable"] or generators.loc["grid", "p_nom"] != 0:
        raise ValueError("Grid must be disabled")
    weather_files = []
    for entry in previous["weather_subsets"]:
        path = args.weather_run / "weather_used" / Path(entry["path"]).name
        if sha(path) != entry["sha256"]:
            raise ValueError(f"Weather hash failure: {path}")
        weather_files.append({"path": str(path.resolve()), "sha256": entry["sha256"]})
    manifest = {"source_release": release, "experiment": config, "land_qa": land_qa,
                "pv_policy": args.pv_policy, "tech_yaml_changed_from_reference": tech_yaml_changed,
                "solver_acceptance": "status ok AND termination optimal; physical QA also required",
                "numerical_retry_options": ROBUST_GUROBI_OPTIONS,
                "previous_manifest_sha256": sha(args.weather_run / "manifest.json"),
                "inputs": [{"path": str(p.resolve()), "sha256": sha(p)} for p in [args.land, config_path, *files]],
                "weather_inputs": weather_files,
                "protocol": "One 1 Mt/year unconstrained control per site, then the configured specified-quantity, full-year hourly shared-land solves; only proven solver infeasibility is recorded as infeasible. No grid backstop. No post-solve capacity scaling for constrained points. PV policy is explicitly recorded; fixed-only disables tracking capacity before every solve."}
    if args.preflight_only:
        # Exercise the real loader so missing run-time dependencies cannot pass.
        weather = lt.all_locations(str(args.weather_run / "weather_used"), cache_resources=True)
        weather.close()
        print(json.dumps({"preflight_pass": True, "source_release": release,
                          "pv_policy": args.pv_policy,
                          "cases": len(config["sites"])+sum(len(s["quantities_Mtpa"]) for s in config["sites"])}, indent=2))
        return
    args.output.mkdir(parents=True, exist_ok=False)
    write_json(args.output / "manifest.json", manifest)
    land = _apply_spatial_solar_density_from_tech_config(_load_land_table(args.land), tech)
    STATE.update(config=config, output=args.output, pv_policy=args.pv_policy, tech=tech, meta=_load_tech_meta(tech_path),
                 land_allocation=land_allocation,
                 land_lookup=_build_land_lookup(land),
                 interest=_build_interest_lookup(_load_interest_table(ROOT / "inputs/amelired_interest_inputs_2050.csv")))
    STATE["weather"] = lt.all_locations(str(args.weather_run / "weather_used"), cache_resources=True)
    STATE["base"] = plant_main.generate_network(8760, str(plant), aggregation_count=1, time_step=1.,
                                                temporal_accounting_mode="snapshot_weighted")
    apply_pv_policy(STATE["base"], args.pv_policy)
    try:
        with ProcessPoolExecutor(max_workers=3, mp_context=multiprocessing.get_context("fork")) as pool:
            summaries = list(pool.map(run_site, config["sites"]))
    finally:
        STATE["weather"].close()
    rows = [point for s in summaries for point in s["points"]]
    pd.DataFrame(rows).to_csv(args.output / "supply_points.csv", index=False)
    summary = {"qa_pass": True, "sites": summaries, "manifest_sha256": sha(args.output / "manifest.json"),
               "points_sha256": sha(args.output / "supply_points.csv"),
               "feasible_points": sum(r["status"] == "feasible" for r in rows),
               "infeasible_points": sum(r["status"] == "infeasible" for r in rows)}
    write_json(args.output / "summary.json", summary)
    print(json.dumps({k: v for k, v in summary.items() if k != "sites"}, indent=2), flush=True)


if __name__ == "__main__":
    main()
