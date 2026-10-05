#!/usr/bin/env python3
"""Independently check completed supply points and derive an ideal annual-energy bound."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import xarray as xr

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from arc.release_source_inventory import _sha256 as sha


def check_pv_policy(result, policy):
    if policy not in {"both", "fixed-only"}:
        raise ValueError("Unknown PV policy")
    if policy == "fixed-only" and abs(float(result.solar_tracking_mw)) > 1e-8:
        raise ValueError("Tracking capacity in fixed-only experiment")


def energy_upper_bound(land, capacity_factors, electricity_floor_mwh_per_t):
    """Maximize annual generation over the same exclusive shared-area budget.

    Dropping all temporal, storage and curtailment constraints gives an upper
    bound, not an achievable plant design. Fixed and tracking share one PV area.
    """
    if electricity_floor_mwh_per_t <= 0 or not np.isfinite(electricity_floor_mwh_per_t):
        raise ValueError("Invalid electricity floor")
    if any(not np.isfinite(v) or not 0 <= v <= 1 for v in capacity_factors.values()):
        raise ValueError("Invalid capacity factors")
    fixed = land.solar_density_mw_per_km2 * capacity_factors["solar"] * 8760
    tracking = land.solar_density_mw_per_km2 / 2 * capacity_factors["solar_tracking"] * 8760
    wind = land.wind_density_mw_per_km2 * capacity_factors["wind"] * 8760
    options = sorted([(max(fixed, tracking), land.solar_area_km2),
                      (wind, land.wind_onshore_area_km2)], reverse=True)
    remaining, annual_energy = float(land.renewable_union_area_km2), 0.
    for yield_per_area, available_area in options:
        area = min(remaining, float(available_area))
        annual_energy += area * yield_per_area
        remaining -= area
    return {"ideal_annual_generation_MWh": annual_energy,
            "electricity_floor_MWh_per_t": electricity_floor_mwh_per_t,
            "ideal_energy_bound_Mtpa": annual_energy / electricity_floor_mwh_per_t / 1e6}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("run", "land", "weather-run", "release", "output"):
        parser.add_argument("--" + key, type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    manifest = json.loads((args.run / "manifest.json").read_text())
    policy = manifest.get("pv_policy", "both")
    if policy != manifest["experiment"].get("pv_policy", "both"):
        raise ValueError("PV policy differs from experiment configuration")
    summary = json.loads((args.run / "summary.json").read_text())
    if (manifest["experiment"]["tracking_footprint_ratio"] <= 0 or
            manifest["experiment"]["land_allocation"] not in {"exclusive", "colocated"}):
        raise ValueError("Audit requires a declared land allocation (exclusive or colocated) and a positive tracking ratio")
    if not summary["qa_pass"] or sha(args.run / "manifest.json") != summary["manifest_sha256"]:
        raise ValueError("Run manifest check failed")
    if sha(args.run / "supply_points.csv") != summary["points_sha256"]:
        raise ValueError("Point table hash failed")
    if sha(args.land) != manifest["land_qa"]["file_sha256"]:
        raise ValueError("Wrong land input")
    previous = json.loads((args.weather_run / "manifest.json").read_text())
    if sha(args.weather_run / "manifest.json") != manifest["previous_manifest_sha256"]:
        raise ValueError("Wrong weather-source manifest")
    datasets = {}
    for entry in previous["weather_subsets"]:
        path = args.weather_run / "weather_used" / Path(entry["path"]).name
        if sha(path) != entry["sha256"]:
            raise ValueError("Weather hash failure")
        datasets[entry["resource"]] = xr.open_dataset(path)
    links_path = args.release / "basic_ammonia_plant_2050_way_tracking/links.csv"
    input_hashes = {Path(f["path"]).name: f["sha256"] for f in manifest["inputs"]}
    if sha(links_path) != input_hashes["links.csv"]:
        raise ValueError("Wrong plant conversion input")
    links = pd.read_csv(links_path).set_index("name")
    hb, el = links.loc["ammonia_synthesis"], links.loc["electrolysis"]
    if (hb.bus0, hb.bus1, hb.bus2, el.bus0, el.bus1) != ("power", "ammonia", "hydrogen", "power", "hydrogen"):
        raise ValueError("Unsupported conversion topology for energy bound")
    # The current result contract uses 6.25 MWh/t NH3. Compression and storage
    # losses are deliberately omitted to give an optimistic relaxation.
    floor = 6.25 / hb.efficiency * (1 - hb.efficiency2 / el.efficiency)
    land_table = pd.read_csv(args.land).set_index(["latitude", "longitude"], verify_integrity=True)
    rows, site_rows = [], []
    try:
        for site_summary, site in zip(summary["sites"], manifest["experiment"]["sites"], strict=True):
            if site_summary["site"] != site["id"] or not site_summary["qa_pass"]:
                raise ValueError("Site mismatch")
            location = (site["latitude"], site["longitude"])
            land = land_table.loc[location]
            control_path = args.run / site["id"] / "control_unconstrained/result.csv"
            if sha(control_path) != site_summary["control"]["result_sha256"]:
                raise ValueError("Control hash failed")
            control = pd.read_csv(control_path).iloc[0]
            check_pv_policy(control, policy)
            if (control.solver_status, control.solver_termination) != ("ok", "optimal"):
                raise ValueError("Control is not an accepted optimum")
            if abs(control.lcoa_eur_per_t - site["control_LCOA_EUR2020_per_t"]) > manifest["experiment"]["control_lcoa_tolerance_eur_per_t"]:
                raise ValueError("Control cost changed")
            cf = {name: float(next(iter(ds.data_vars.values())).sel(
                latitude=location[0], longitude=location[1]).mean()) for name, ds in datasets.items()}
            cf["wind"] *= .93  # Matches the current weather-frame wake-loss factor.
            bound = energy_upper_bound(land, cf, floor)
            one_mt_lcoa = None
            points = site_summary["points"]
            if [p["target_Mtpa"] for p in points] != site["quantities_Mtpa"]:
                raise ValueError("Missing or reordered target quantity")
            for point in points:
                quantity = point["target_Mtpa"]
                folder = args.run / site["id"] / ("q_" + format(quantity, ".12g").replace(".", "p"))
                if json.loads((folder / "status.json").read_text()) != point:
                    raise ValueError("Point summary differs from checkpoint")
                attempts = point["solver_attempts"]
                if not attempts or len(attempts) > 2:
                    raise ValueError("Missing or excessive solver attempts")
                if attempts[0]["solver_options_override"] is not None:
                    raise ValueError("First attempt did not use controlled baseline settings")
                if len(attempts) == 2 and attempts[1]["solver_options_override"] != manifest["numerical_retry_options"]:
                    raise ValueError("Unrecorded retry settings")
                if point["status"] == "infeasible":
                    if (point["solver_termination"] != "infeasible" or
                            attempts[-1]["termination"] != "infeasible" or (folder / "result.csv").exists()):
                        raise ValueError("Infeasible point carries an apparent solution")
                elif point["status"] == "feasible":
                    path = folder / "result.csv"
                    if sha(path) != point["result_sha256"]:
                        raise ValueError("Raw solution hash failed")
                    result = pd.read_csv(path).iloc[0]
                    check_pv_policy(result, policy)
                    if ((result.solver_status, result.solver_termination) != ("ok", "optimal") or
                            (attempts[-1]["status"], attempts[-1]["termination"]) != ("ok", "optimal")):
                        raise ValueError("Feasible point is not an accepted optimum")
                    if (result.capacity_method != "solved_quantity" or result.lcoa_land_mode != "enforce" or
                            result.temporal_accounting_mode != "snapshot_weighted" or
                            result.ramp_limit_basis != "per_hour"):
                        raise ValueError("Wrong constrained solve mode")
                    np.testing.assert_allclose(result.simulated_hours, 8760, atol=1e-6)
                    np.testing.assert_allclose(result.snapshot_hours, 1, atol=1e-9)
                    np.testing.assert_allclose(result.solar_density_mw_per_km2, land.solar_density_mw_per_km2, rtol=1e-12)
                    np.testing.assert_allclose(result.annual_ammonia_production_t, quantity * 1e6, rtol=1e-7)
                    np.testing.assert_allclose(result.total_cost_eur_per_year,
                        result.plant_objective_cost_eur_per_year + result.water_cost_eur_per_year +
                        result.land_cost_eur_per_year, rtol=1e-10, atol=1e-3)
                    np.testing.assert_allclose(result.lcoa_eur_per_t,
                        result.total_cost_eur_per_year / result.annual_ammonia_production_t, rtol=1e-10)
                    if abs(result.grid_energy_mwh) > result.grid_energy_tolerance_mwh or abs(result.accumulated_penalty_mwh) > 1:
                        raise ValueError("Grid/slack use")
                    wind = max(0., result.wind_mw) / land.wind_density_mw_per_km2
                    pv = (max(0., result.solar_mw) + 2 * max(0., result.solar_tracking_mw)) / land.solar_density_mw_per_km2
                    for used, available in ((wind, land.wind_onshore_area_km2),
                                             (pv, land.solar_area_km2),
                                             (wind + pv, land.renewable_union_area_km2)):
                        if used > available + max(1e-5, available * 1e-6):
                            raise ValueError("Independent shared-area check failed")
                    if quantity > bound["ideal_energy_bound_Mtpa"] + 1e-6:
                        raise ValueError("Claimed feasible quantity exceeds relaxed annual-energy bound")
                    if quantity == 1:
                        one_mt_lcoa = float(result.lcoa_eur_per_t)
                else:
                    raise ValueError("Unrecognized point status")
                rows.append(point)
            site_rows.append({"site": site["id"], "latitude": location[0], "longitude": location[1],
                "historical_Mtpa": site["historical_Mtpa"],
                "control_LCOA_EUR2020_per_t": site_summary["control"]["LCOA_EUR2020_per_t"],
                "one_Mtpa_LCOA_EUR2020_per_t": one_mt_lcoa,
                "maximum_feasible_tested_Mtpa": site_summary["maximum_feasible_tested_Mtpa"],
                "minimum_infeasible_tested_Mtpa": site_summary["minimum_infeasible_tested_Mtpa"],
                **bound})
    finally:
        for dataset in datasets.values():
            dataset.close()
    args.output.mkdir(parents=True, exist_ok=False)
    pd.DataFrame(rows).to_csv(args.output / "validated_points.csv", index=False)
    pd.DataFrame(site_rows).to_csv(args.output / "site_comparison.csv", index=False)
    report = {"qa_pass": True, "pv_policy": policy, "points_rechecked": len(rows), "site_comparison": site_rows,
              "source_summary_sha256": sha(args.run / "summary.json"), "script_sha256": sha(Path(__file__)),
              "scope": "Three selected sites and unchanged source-substituted land. Annual-energy bound relaxes temporal balance, storage, compression and curtailment constraints under the current model's conversion efficiencies, exclusive shared-area accounting and ratio-2 tracker footprint. It is not an achievable capacity or a universal bound on the historical model."}
    write = args.output / "summary.json"
    write.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
