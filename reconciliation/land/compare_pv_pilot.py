#!/usr/bin/env python3
"""Audit actual fixed-PV solves against central designs on identical land."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from arc.release_source_inventory import _sha256 as sha


def shared_capacity(row, land, tracking_ratio=2.0):
    """Independent arithmetic, deliberately not calling the production estimator."""
    wind = max(0.0, float(row.wind_mw)) / float(land.wind_density_mw_per_km2)
    solar = (max(0.0, float(row.solar_mw)) + tracking_ratio *
             max(0.0, float(row.solar_tracking_mw))) / float(land.solar_density_mw_per_km2)
    limits = {name: area / used for name, area, used in (
        ("wind", land.wind_onshore_area_km2, wind),
        ("solar", land.solar_area_km2, solar),
        ("renewable_union", land.renewable_union_area_km2, wind + solar),
    ) if used > 1e-9}
    if not limits:
        raise ValueError("No renewable footprint")
    limiting = min(limits, key=limits.get)
    return {
        "capacity_Mtpa": limits[limiting] * float(row.gridless_ammonia_production_t) / 1e6,
        "limiting_constraint": limiting, "wind_footprint_km2": wind,
        "pv_footprint_km2": solar,
    }


def validate_plants(frame):
    for key, expected in {
        "currency": "EUR", "temporal_accounting_mode": "snapshot_weighted",
        "ramp_limit_basis": "per_hour", "land_allocation": "technology_shared",
        "capacity_method": "paper_scaled", "lcoa_land_mode": "postprocess",
    }.items():
        if not frame[key].eq(expected).all():
            raise ValueError(f"Unexpected {key}")
    for key, expected in {
        "annual_ammonia_production_t": 1e6, "simulated_hours": 8760,
        "snapshot_hours": 1, "build_cost_multiplier": 1,
        "water_cost_usd_per_m3": 2, "land_cost_usd_per_km2_year": 0,
    }.items():
        np.testing.assert_allclose(frame[key], expected, rtol=1e-8, atol=1e-6)
    for key in ("site_costs_in_headline", "is_full_year_result", "is_gridless_feasible"):
        if not frame[key].astype(str).str.lower().eq("true").all():
            raise ValueError(f"Failed {key}")
    if not (frame.grid_energy_mwh.abs() <= frame.grid_energy_tolerance_mwh).all():
        raise ValueError("Grid energy exceeds tolerance")
    if not (frame.accumulated_penalty_mwh.abs() <= 1).all():
        raise ValueError("Feasibility slack exceeds tolerance")
    numeric = frame[["lcoa_eur_per_t", "solar_mw", "solar_tracking_mw", "wind_mw",
                     "gridless_ammonia_production_t", "total_cost_eur_per_year"]]
    if not np.isfinite(numeric.to_numpy()).all() or not (frame.lcoa_eur_per_t > 0).all():
        raise ValueError("Invalid plant values")
    np.testing.assert_allclose(frame.total_cost_eur_per_year,
        frame.plant_objective_cost_eur_per_year + frame.water_cost_eur_per_year +
        frame.land_cost_eur_per_year, rtol=1e-10, atol=1e-4)
    np.testing.assert_allclose(frame.lcoa_eur_per_t,
        frame.total_cost_eur_per_year / frame.annual_ammonia_production_t,
        rtol=1e-10, atol=1e-8)
    np.testing.assert_allclose(frame.headline_cost_share_total_pct, 100, rtol=0, atol=1e-8)


def validate_comparison_inputs(fixed_manifest, central_manifest):
    """Hold costs/finance/non-generator bundle fixed; check recorded weather identity."""
    new = {Path(f["path"]).name: f["sha256"] for f in fixed_manifest["inputs"]}
    old = central_manifest["inputs"]
    required = [old["override_csv"], *old["tech_yaml"]["extends_chain"]]
    required += [entry for name, entry in old["plant_bundle"]["files"].items()
                 if name != "generators.csv"]
    for entry in required:
        name = Path(entry["path"]).name
        if new.get(name) != entry["sha256"]:
            raise ValueError(f"Changed controlled input: {name}")
    new_weather = {Path(f["path"]).name: (f["size_bytes"], f["mtime_ns"])
                   for f in fixed_manifest["weather_source_files"]}
    old_weather = {f["name"]: (f["size_bytes"], f["mtime_ns"])
                   for f in old["weather"]["files"]}
    if new_weather != old_weather:
        raise ValueError("Global weather source metadata changed")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("fixed", "central", "central-manifest", "land", "diagnostics", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    manifest = json.loads((args.fixed / "manifest.json").read_text())
    summary = json.loads((args.fixed / "summary.json").read_text())
    if not summary["qa_pass"] or sha(args.fixed / "run_global.csv") != summary["output_sha256"]:
        raise ValueError("Fixed-PV result hash or completion check failed")
    for entry in manifest["weather_subsets"]:
        if sha(args.fixed / "weather_used" / Path(entry["path"]).name) != entry["sha256"]:
            raise ValueError("Weather subset hash failed")
    if sha(args.land) != manifest["land_qa"]["file_sha256"]:
        raise ValueError("Comparison land differs from optimization input")
    validate_comparison_inputs(manifest, json.loads(args.central_manifest.read_text()))
    index = ["latitude", "longitude"]
    fixed = pd.read_csv(args.fixed / "run_global.csv").set_index(index, verify_integrity=True)
    expected = {(-23., -69.), (-23., 117.), (-21., 135.)}
    if set(fixed.index) != expected or summary["rows"] != 3:
        raise ValueError("Unexpected fixed-PV sites")
    central = pd.read_csv(args.central).set_index(index, verify_integrity=True).loc[fixed.index]
    land = pd.read_csv(args.land).set_index(index, verify_integrity=True)
    diagnostics = pd.read_csv(args.diagnostics).set_index(index, verify_integrity=True)
    validate_plants(fixed)
    validate_plants(central)
    np.testing.assert_allclose(fixed.solar_tracking_mw, 0, rtol=0, atol=1e-6)
    np.testing.assert_allclose(fixed.solar_density_mw_per_km2,
        land.loc[fixed.index].solar_density_mw_per_km2, rtol=1e-12)
    rows = []
    for coord, f in fixed.iterrows():
        old, geography = central.loc[coord], land.loc[coord]
        new_cap = shared_capacity(f, geography)
        np.testing.assert_allclose(new_cap["capacity_Mtpa"],
            f.paper_scaled_max_gridless_onshore_ammonia_capacity_mtpa, rtol=1e-9, atol=1e-9)
        for ratio in (1., 1.25, 1.5, 2.):
            old_cap = shared_capacity(old, geography, ratio)
            rows.append({"latitude": coord[0], "longitude": coord[1], "country": geography.country,
                "tracking_footprint_ratio": ratio,
                "historical_Mtpa": diagnostics.loc[coord].historical_capacity_Mtpa,
                "central_original_land_Mtpa": old.paper_scaled_max_gridless_onshore_ammonia_capacity_mtpa,
                "central_common_land_Mtpa": old_cap["capacity_Mtpa"],
                "fixed_common_land_Mtpa": new_cap["capacity_Mtpa"],
                "fixed_capacity_change_pct": 100 * (new_cap["capacity_Mtpa"] / old_cap["capacity_Mtpa"] - 1),
                "central_LCOA_EUR2020_per_t": old.lcoa_eur_per_t,
                "fixed_LCOA_EUR2020_per_t": f.lcoa_eur_per_t,
                "fixed_LCOA_change_pct": 100 * (f.lcoa_eur_per_t / old.lcoa_eur_per_t - 1),
                **{"central_" + k: v for k, v in old_cap.items() if k != "capacity_Mtpa"},
                **{"fixed_" + k: v for k, v in new_cap.items() if k != "capacity_Mtpa"},
                "fixed_solar_MW": f.solar_mw, "fixed_wind_MW": f.wind_mw,
                "central_tracking_MW": old.solar_tracking_mw, "central_wind_MW": old.wind_mw})
    result = pd.DataFrame(rows)
    args.output.mkdir(parents=True, exist_ok=False)
    result.to_csv(args.output / "comparison.csv", index=False)
    report = {"qa_pass": True, "sites": 3, "comparison_sha256": sha(args.output / "comparison.csv"),
        "comparison_script_sha256": sha(Path(__file__)),
        "inputs": {name: sha(getattr(args, name)) for name in ("central", "central_manifest", "land", "diagnostics")},
        "fixed_manifest_sha256": sha(args.fixed / "manifest.json"),
        "fixed_results_sha256": summary["output_sha256"],
        "checks": ["result/weather/land hashes", "matching cost/finance and non-generator plant inputs",
                   "matching recorded parent weather metadata", "full-year hourly production",
                   "grid and feasibility slack", "annual-cost and cost-share closure",
                   "preserved explicit PV density", "independent shared-land capacity reconstruction"],
        "limitations": "Three selected sites, not a global result. Central design permits both PV types and is tracking-dominated. Its capacity is recomputed on the common land, not reoptimized. Packing ratios only change postprocessed land. Ratio 1 is an equal-packing counterfactual, not a claim about real tracker layouts. Global parent weather is identified by metadata, not payload hashes.",
        "at_configured_tracking_ratio_2": result.loc[result.tracking_footprint_ratio == 2].to_dict(orient="records")}
    (args.output / "summary.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
