#!/usr/bin/env python3
"""Independent arithmetic checks and fixed-design land sensitivities.

These calculations do not rerun plant optimisation. They distinguish changing
the land-accounting rule from changing the plant design or available land.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


def sha(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def ratio(available, used):
    return np.divide(available, used, out=np.full(len(used), np.inf), where=used > 1e-9)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--received", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    report = {"script_sha256": sha(__file__), "surfaces": {}, "sensitivity": "Fixed optimised plant design and land budget; no plant reoptimisation or land-fraction fitting"}
    tables = []
    for label, directory in [("rep", "10_replication"), ("central", "20_central")]:
        path = next((args.received / "lory" / directory).rglob("merged/global_run_results.csv"))
        d = pd.read_csv(path)
        wind = d.wind_mw.clip(lower=0) / d.wind_density_mw_per_km2
        solar = d.solar_mw.clip(lower=0) / d.solar_density_mw_per_km2
        tracking = d.solar_tracking_mw.clip(lower=0) / d.solar_tracking_density_mw_per_km2
        tracking_as_fixed = d.solar_tracking_mw.clip(lower=0) / d.solar_density_mw_per_km2
        union = ratio(d.renewable_union_area_km2.to_numpy(), (wind+solar+tracking).to_numpy())
        # The compatibility wind_available_area field includes offshore area.
        # The versioned onshore capacity contract explicitly uses this field.
        wind_limit = ratio(d.wind_onshore_area_km2.to_numpy(), wind.to_numpy())
        solar_limit = ratio(d.solar_available_area_km2.to_numpy(), (solar+tracking).to_numpy())
        shared = np.minimum.reduce([union, wind_limit, solar_limit])
        fixed = ratio(d.renewable_union_area_km2.to_numpy(), (wind+solar+tracking_as_fixed).to_numpy())
        production = d.gridless_ammonia_production_t.to_numpy()
        calculated = (union if label == "rep" else shared) * production
        observed = d.paper_scaled_max_gridless_onshore_ammonia_capacity_t.to_numpy()
        np.testing.assert_allclose(calculated, observed, rtol=1e-8, atol=1e-3)
        np.testing.assert_allclose(d.annual_ammonia_production_t, 1e6, rtol=0, atol=1)
        np.testing.assert_allclose(d.total_cost_eur_per_year / d.annual_ammonia_production_t,
                                   d.lcoa_eur_per_t, rtol=1e-10, atol=1e-8)
        tech_columns = [c for c in d if c.startswith("lcoa_tech_") and c.endswith("_eur_per_t")]
        np.testing.assert_allclose(d[tech_columns].sum(axis=1), d.lcoa_plant_eur_per_t, rtol=1e-8, atol=1e-6)
        expected_water = d.water_cost_eur_per_t if label == "central" else 0.
        np.testing.assert_allclose(d.lcoa_plant_eur_per_t + expected_water, d.lcoa_eur_per_t, rtol=1e-10, atol=1e-8)
        if not (d.grid_energy_mwh.abs() <= d.grid_energy_tolerance_mwh).all():
            raise ValueError("Grid imports exceed declared tolerance")
        if (d.accumulated_penalty_mwh.abs() > 1).any():
            raise ValueError("Non-negligible feasibility-penalty energy")
        table = d[["latitude", "longitude", "country"]].copy()
        table["surface"] = label
        table["fixed_packing_union_Mtpa"] = fixed * production / 1e6
        table["tracking_union_Mtpa"] = union * production / 1e6
        table["tracking_technology_shared_Mtpa"] = shared * production / 1e6
        table["actual_Mtpa"] = observed / 1e6
        tables.append(table)
        sensitivity = {}
        for region in ["Global", "Australia", "Chile", "Mauritania"]:
            selected = table if region == "Global" else table[table.country == region]
            sensitivity[region] = {c: int((selected[c] >= 1).sum()) for c in table if c.endswith("Mtpa")}
        report["surfaces"][label] = {
            "source_sha256": sha(path), "rows_checked": len(d),
            "maximum_capacity_recalculation_error_tpa": float(np.max(np.abs(calculated-observed))),
            "maximum_grid_energy_MWh": float(d.grid_energy_mwh.abs().max()),
            "maximum_penalty_energy_MWh": float(d.accumulated_penalty_mwh.abs().max()),
            "maximum_reference_production_error_tpa": float(abs(d.annual_ammonia_production_t-1e6).max()),
            "maximum_cost_component_closure_EUR_per_t": float(abs(d[tech_columns].sum(axis=1)-d.lcoa_plant_eur_per_t).max()),
            "eligible_cells_by_land_accounting": sensitivity,
        }
    pd.concat(tables).to_csv(args.output / "fixed_design_land_sensitivities.csv", index=False)
    with (args.output / "summary.json").open("x") as handle:
        json.dump(report, handle, indent=2, allow_nan=False)
        handle.write("\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
