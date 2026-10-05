#!/usr/bin/env python3
"""Close planned quantities with audited optima or explicit infeasibility evidence.

This does not turn a timed-out Slurm run into a completed solver run. Analytic
certificates live in a new derived result, leaving original checkpoints intact.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import xarray as xr

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from arc.release_source_inventory import _sha256 as sha, verify_inventory
from reconciliation.land.run_supply_pilot import validate_result


def conversion_certificate(generators, links, stores, loads):
    """Non-increasing electricity-equivalent potential for every physical link.

Summing bus balances over a cyclic year cancels internal flows and storage.
With these potentials, only renewable generation can supply the ammonia load.
Ramp-penalty components have zero potential and cannot feed physical buses.
"""
    hb, el = links.loc["ammonia_synthesis"], links.loc["electrolysis"]
    if (hb.bus0, hb.bus1, hb.bus2, el.bus0, el.bus1) != ("power", "ammonia", "hydrogen", "power", "hydrogen"):
        raise ValueError("Unsupported production topology")
    if not 0 < el.efficiency <= 1 or not hb.efficiency > 0 or not hb.efficiency2 < 0:
        raise ValueError("Invalid conversion efficiencies")
    ammonia_potential = (1-hb.efficiency2/el.efficiency)/hb.efficiency
    potentials = {"power": 1., "power_storage": 1., "hydrogen": 1/el.efficiency,
                  "hydrogen_storage": 1/el.efficiency, "ammonia": ammonia_potential,
                  "ramp_penalty": 0., "ramp_penalty_dest": 0.}
    margins = {}
    for name, row in links.iterrows():
        if not np.isfinite(row.p_min_pu) or row.p_min_pu < 0:
            raise ValueError("Reversible links require a different energy proof")
        net = -potentials[row.bus0]
        for port in range(1, 5):
            bus = row.get(f"bus{port}")
            if pd.isna(bus) or bus == "":
                continue
            eff = float(row["efficiency" if port == 1 else f"efficiency{port}"])
            if not np.isfinite(eff):
                raise ValueError("Non-finite link efficiency")
            net += eff*potentials[bus]
        if net > 1e-10:
            raise ValueError(f"Link creates electricity-equivalent energy: {name}")
        margins[name] = float(net)
    for name, row in generators.iterrows():
        if potentials[row.bus] > 0:
            if name in {"wind", "solar", "solar_tracking"}:
                if row.bus != "power" or not row.p_nom_extendable or row.p_nom != 0:
                    raise ValueError("Unexpected renewable definition")
            elif row.p_nom != 0 or row.p_nom_extendable:
                raise ValueError(f"Unbounded external energy source: {name}")
    for name, row in stores.iterrows():
        if not 0 <= row.standing_loss <= 1:
            raise ValueError("Storage can create energy")
        if potentials[row.bus] > 0 and not row.e_cyclic:
            raise ValueError("Physical storage must be cyclic for this certificate")
    if list(loads.index) != ["ammonia"] or loads.loc["ammonia", "bus"] != "ammonia":
        raise ValueError("Unexpected load topology")
    return {"electricity_floor_MWh_per_t": 6.25*ammonia_potential,
            "bus_electricity_potentials": potentials, "link_potential_changes": margins,
            "physical_storage_cyclic": True, "external_grid_disabled": True}


def area_energy_certificate(land, capacity_factors, floor, *, tracking_footprint_ratio=2.,
                            wind_land_exclusive_fraction=1., pv_policy="both"):
    """Primal/dual certificate of the relaxed two-technology area allocation LP.

    max  y_s x_s + y_w x_w   s.t.  x_s <= S,  x_w <= W,  x_s + f x_w <= U,  x >= 0

    where f is the fraction of the wind footprint that competes with PV for the
    shared budget (1 under the exclusive rule; the wind direct-impact fraction
    under co-location) and y_s is the best allowed PV yield per km2 (fixed, or
    tracking at fixed density / tracking_footprint_ratio, subject to pv_policy).
    """
    names = ("solar_area_km2", "wind_onshore_area_km2", "renewable_union_area_km2",
             "solar_density_mw_per_km2", "wind_density_mw_per_km2")
    values = np.array([float(land[k]) for k in names])
    if not np.isfinite(values).all() or (values < 0).any() or (values[-2:] <= 0).any():
        raise ValueError("Invalid land areas/densities")
    if set(capacity_factors) != {"solar", "solar_tracking", "wind"} or any(
            not np.isfinite(v) or not 0 <= v <= 1 for v in capacity_factors.values()):
        raise ValueError("Invalid renewable capacity factors")
    if not np.isfinite(floor) or floor <= 0:
        raise ValueError("Invalid conversion energy floor")
    if not np.isfinite(tracking_footprint_ratio) or tracking_footprint_ratio <= 0:
        raise ValueError("Invalid tracking footprint ratio")
    if not np.isfinite(wind_land_exclusive_fraction) or not 0 <= wind_land_exclusive_fraction <= 1:
        raise ValueError("Invalid wind exclusive fraction")
    if pv_policy not in {"both", "fixed-only", "tracking-only"}:
        raise ValueError("Unknown PV policy")
    solar, wind, union, sdensity, wdensity = values
    fixed_yield = sdensity*capacity_factors["solar"]*8760
    tracking_yield = sdensity/tracking_footprint_ratio*capacity_factors["solar_tracking"]*8760
    pv_yield = {"both": max(fixed_yield, tracking_yield), "fixed-only": fixed_yield, "tracking-only": tracking_yield}[pv_policy]
    yields = np.array([pv_yield, wdensity*capacity_factors["wind"]*8760])
    areas = np.array([solar, wind])
    weights = np.array([1., wind_land_exclusive_fraction])   # union-budget consumption per km2 used
    # Dual: min lambda*U + mu_s*S + mu_w*W  s.t.  lambda*w_i + mu_i >= y_i, all >= 0.
    candidates = []
    for lam in [0., *[y/w for y, w in zip(yields, weights) if w > 0]]:
        mu = np.maximum(yields-lam*weights, 0)
        candidates.append((float(lam*union+mu@areas), float(lam), mu))
    energy, lam, mu = min(candidates, key=lambda v: v[0])
    # Primal: fractional knapsack on the union budget, ordered by yield per unit of budget used;
    # a technology with zero weight never consumes the budget.
    remaining, allocated = union, np.zeros(2)
    order = sorted(range(2), key=lambda i: -(yields[i]/weights[i] if weights[i] > 0 else np.inf))
    for i in order:
        allocated[i] = areas[i] if weights[i] == 0 else min(areas[i], remaining/weights[i])
        remaining -= weights[i]*allocated[i]
    np.testing.assert_allclose(energy, yields@allocated, rtol=1e-12, atol=1e-7)
    if (lam*weights+mu < yields-1e-9).any():
        raise ValueError("Energy dual is not feasible")
    # Conservatively widen the numerical limit, not the physical data assumptions.
    cushioned = energy*(1+1e-6)+1.
    return {"ideal_annual_generation_MWh": energy, "electricity_floor_MWh_per_t": floor,
            "ideal_energy_bound_Mtpa": energy/floor/1e6,
            "certificate_bound_Mtpa": cushioned/floor/1e6,
            "roundoff_cushion_MWh": cushioned-energy,
            "tracking_footprint_ratio": float(tracking_footprint_ratio),
            "wind_land_exclusive_fraction": float(wind_land_exclusive_fraction),
            "pv_policy": pv_policy, "allocated_solar_km2": float(allocated[0]), "allocated_wind_km2": float(allocated[1]),
            "fixed_generation_MWh_per_km2": float(fixed_yield),
            "tracking_generation_MWh_per_km2": float(tracking_yield),
            "wind_generation_MWh_per_km2": float(yields[1]),
            "area_dual_lambda": lam, "area_dual_mu_solar": float(mu[0]), "area_dual_mu_wind": float(mu[1])}


def classify_missing(quantity, certificate):
    if quantity > certificate["certificate_bound_Mtpa"]:
        return "infeasible_analytic"
    return "unresolved"


def profile_diagnostics(values, resource):
    values = np.asarray(values)
    if values.shape != (8760,) or not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("Weather is not one complete nonnegative hourly year")
    if resource == "wind" and (values > 1).any():
        raise ValueError("Wind profile exceeds rated output")
    # Preserve the exact model inputs. PV profiles can exceed the reference
    # nameplate; whether the AC/DC normalization is appropriate is a separate
    # source-data question, not permission to clip a controlled experiment.
    return {"mean": float(values.mean()), "maximum": float(values.max()),
            "hours_above_one": int((values > 1).sum()),
            "clipping_at_one_energy_reduction_percent": float(100*(values.sum()-np.minimum(values, 1).sum())/values.sum()) if values.sum() else 0.}


def main():
    p = argparse.ArgumentParser()
    for key in ("run", "checkpoint-audit", "land", "weather-run", "release", "output"):
        p.add_argument("--"+key, type=Path, required=True)
    a = p.parse_args()
    if a.output.exists():
        raise FileExistsError(a.output)
    manifest = json.loads((a.run/"manifest.json").read_text())
    checkpoint = json.loads((a.checkpoint_audit/"summary.json").read_text())
    if (not checkpoint["checkpoint_qa_pass"] or checkpoint["source_manifest_sha256"] != sha(a.run/"manifest.json") or
            checkpoint["output_sha256"] != sha(a.checkpoint_audit/"validated_checkpoints.csv")):
        raise ValueError("Checkpoint audit provenance failure")
    release = verify_inventory(a.release, a.release/"source_inventory.json")
    if release != manifest["source_release"]:
        raise ValueError("Wrong source release")
    experiment = manifest["experiment"]
    land_policy = {"tracking_footprint_ratio": float(experiment["tracking_footprint_ratio"]),
                   "land_allocation": experiment["land_allocation"],
                   "wind_land_exclusive_fraction": (1. if experiment["land_allocation"] == "exclusive"
                                                    else float(experiment["wind_land_exclusive_fraction"])),
                   "pv_policy": manifest.get("pv_policy", experiment.get("pv_policy", "both"))}
    if experiment["land_allocation"] not in {"exclusive", "colocated"}:
        raise ValueError("Unexpected land policy")
    if sha(a.land) != manifest["land_qa"]["file_sha256"]:
        raise ValueError("Wrong land source")
    for entry in manifest["inputs"]:
        path = Path(entry["path"])
        if "green-lory-releases" in path.parts:
            relative = Path(*path.parts[path.parts.index("green-lory-releases")+2:])
            if sha(a.release/relative) != entry["sha256"]:
                raise ValueError(f"Changed plant/config input: {relative}")
        elif entry["sha256"] != sha(a.land):
            raise ValueError("Unknown external input")
    # Validate the exact hourly resource arrays rather than an unvalidated mean.
    previous = json.loads((a.weather_run/"manifest.json").read_text())
    if sha(a.weather_run/"manifest.json") != manifest["previous_manifest_sha256"]:
        raise ValueError("Wrong weather manifest")
    if {Path(e["path"]).name: e["sha256"] for e in manifest["weather_inputs"]} != {
            Path(e["path"]).name: e["sha256"] for e in previous["weather_subsets"]}:
        raise ValueError("Weather inventories differ")
    datasets = {}
    for entry in previous["weather_subsets"]:
        path = a.weather_run/"weather_used"/Path(entry["path"]).name
        if sha(path) != entry["sha256"]:
            raise ValueError("Weather content changed")
        datasets[entry["resource"]] = xr.open_dataset(path)
    plant = a.release/"basic_ammonia_plant_2050_way_tracking"
    tables = {name: pd.read_csv(plant/(name+".csv")).set_index("name") for name in ("generators", "links", "stores", "loads")}
    conversion = conversion_certificate(**tables)
    floor = conversion["electricity_floor_MWh_per_t"]
    land_table = pd.read_csv(a.land).set_index(["latitude", "longitude"], verify_integrity=True)
    rows, sites, certificates = [], [], []
    try:
        for site in manifest["experiment"]["sites"]:
            coord = (site["latitude"], site["longitude"])
            land = land_table.loc[coord]
            cf, profile_checks = {}, {}
            for resource, ds in datasets.items():
                if len(ds.data_vars) != 1:
                    raise ValueError("Ambiguous weather variable")
                values = next(iter(ds.data_vars.values())).sel(latitude=coord[0], longitude=coord[1]).values
                profile_checks[resource] = profile_diagnostics(values, resource)
                cf[resource] = profile_checks[resource]["mean"]*(.93 if resource == "wind" else 1.)
            certificate = area_energy_certificate(land, cf, floor, tracking_footprint_ratio=land_policy["tracking_footprint_ratio"],
                                                  wind_land_exclusive_fraction=land_policy["wind_land_exclusive_fraction"],
                                                  pv_policy=land_policy["pv_policy"])
            certificates.append({"site": site["id"], "capacity_factors_after_wake": cf,
                                 "unmodified_profile_diagnostics": profile_checks, **certificate})
            control = pd.read_csv(a.run/site["id"]/"control_unconstrained/result.csv").iloc[0]
            control_status = json.loads((a.run/site["id"]/"control_unconstrained/status.json").read_text())
            if sha(a.run/site["id"]/"control_unconstrained/result.csv") != control_status["result_sha256"]:
                raise ValueError("Control changed")
            validate_result(control.to_dict(), 1., land, constrained=False)
            feasible, infeasible, unresolved = [], [], []
            for q in site["quantities_Mtpa"]:
                folder = a.run/site["id"]/("q_"+format(q, ".12g").replace(".", "p"))
                row = {"site": site["id"], "latitude": coord[0], "longitude": coord[1], "target_Mtpa": q,
                       "LCOA_EUR2020_per_t": None, "annual_cost_EUR2020": None,
                       "certificate_bound_Mtpa": certificate["certificate_bound_Mtpa"],
                       "annual_energy_shortfall_MWh": q*1e6*floor-certificate["ideal_annual_generation_MWh"]}
                if (folder/"status.json").exists():
                    point = json.loads((folder/"status.json").read_text())
                    prior = pd.read_csv(a.checkpoint_audit/"validated_checkpoints.csv")
                    checked = prior[(prior.site == site["id"]) & prior.constrained & np.isclose(prior.target_Mtpa, q, rtol=0, atol=1e-10)]
                    if len(checked) != 1 or sha(folder/"status.json") != checked.iloc[0].checkpoint_sha256:
                        raise ValueError("Checkpoint changed since validation")
                    if point["status"] == "feasible":
                        if sha(folder/"result.csv") != point["result_sha256"]:
                            raise ValueError("Solution content changed")
                        result = pd.read_csv(folder/"result.csv").iloc[0]
                        if (result.solver_status, result.solver_termination) != ("ok", "optimal"):
                            raise ValueError("Nonoptimal feasible result")
                        metrics = validate_result(result.to_dict(), q, land, constrained=True)
                        if q > certificate["certificate_bound_Mtpa"]:
                            raise ValueError("Feasible result contradicts energy certificate")
                        row.update(status="feasible", evidence="solver_optimal_and_physical_qa",
                                   LCOA_EUR2020_per_t=float(result.lcoa_eur_per_t),
                                   annual_cost_EUR2020=float(result.total_cost_eur_per_year),
                                   solar_MW=float(result.solar_mw), tracking_MW=float(result.solar_tracking_mw),
                                   wind_MW=float(result.wind_mw), **metrics)
                    else:
                        row.update(status="infeasible_solver", evidence="solver_infeasible_checkpoint")
                    row["checkpoint_sha256"] = sha(folder/"status.json")
                else:
                    status = classify_missing(q, certificate)
                    row.update(status=status, evidence="annual_energy_dual_certificate" if status == "infeasible_analytic" else "no_certificate")
                row["solver_checkpoint_present"] = (folder/"status.json").exists()
                row["solver_directory_present"] = folder.exists()
                row["also_exceeds_energy_certificate"] = q > certificate["certificate_bound_Mtpa"]
                rows.append(row)
                (feasible if row["status"] == "feasible" else unresolved if row["status"] == "unresolved" else infeasible).append(row)
            if not feasible or unresolved or max(r["target_Mtpa"] for r in feasible) >= min(r["target_Mtpa"] for r in infeasible):
                raise ValueError("Planned grid remains unresolved or nonmonotonic")
            if any(b["LCOA_EUR2020_per_t"] < arow["LCOA_EUR2020_per_t"]-.15 for arow, b in zip(feasible, feasible[1:])):
                raise ValueError("Nonmonotonic average costs")
            sites.append({"site": site["id"], "latitude": coord[0], "longitude": coord[1], "historical_Mtpa": site["historical_Mtpa"],
                          "control_LCOA_EUR2020_per_t": float(control.lcoa_eur_per_t),
                          "maximum_feasible_tested_Mtpa": max(r["target_Mtpa"] for r in feasible),
                          "minimum_infeasible_tested_Mtpa": min(r["target_Mtpa"] for r in infeasible),
                          "one_Mtpa_LCOA_EUR2020_per_t": next(r["LCOA_EUR2020_per_t"] for r in feasible if r["target_Mtpa"] == 1),
                          "historical_to_energy_ceiling_ratio": site["historical_Mtpa"]/certificate["ideal_energy_bound_Mtpa"], **certificate})
    finally:
        for ds in datasets.values():
            ds.close()
    a.output.mkdir(parents=True, exist_ok=False)
    pd.DataFrame(rows).to_csv(a.output/"validated_points.csv", index=False)
    pd.DataFrame(sites).to_csv(a.output/"site_comparison.csv", index=False)
    (a.output/"certificates.json").write_text(json.dumps({"conversion": conversion, "sites": certificates}, indent=2)+"\n")
    report = {"created_utc": datetime.now(timezone.utc).isoformat(), "qa_pass": True,
              "planned_quantity_grid_classified": True, "slurm_run_completed": False,
              "scope": "Three-site unchanged-land quantity grid. Solver outcomes and analytical infeasibility kept distinct. No exact maximum or continuous supply-curve claim.",
              "point_counts": pd.Series([r["status"] for r in rows]).value_counts().to_dict(),
              "source_manifest_sha256": sha(a.run/"manifest.json"), "source_release": release,
              "checkpoint_audit_sha256": sha(a.checkpoint_audit/"summary.json"),
              "code_sha256": {str(f.relative_to(ROOT)): sha(f) for f in (Path(__file__), ROOT/"reconciliation/land/run_supply_pilot.py")},
              "outputs_sha256": {name: sha(a.output/name) for name in ("validated_points.csv", "site_comparison.csv", "certificates.json")}}
    (a.output/"summary.json").write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
