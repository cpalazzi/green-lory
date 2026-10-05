#!/usr/bin/env python3
"""Three-cell model/land evidence, with uncertain legacy provenance explicit.

This does not rerun legacy-lcoa. It applies transparent land equations to its
archived designs and keeps the assumed reference production visible.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

ROOT = Path(__file__).resolve().parents[2]
CAMPAIGN = ROOT / "results/campaigns/land_reconcile_20260914_v1"
LEGACY = Path("/Users/carlopalazzi/programming/shipping_sprint/lcoa_model/lcoa-opt")
SITES = [("atacama", -23, -69), ("northwest_australia", -23, 117),
         ("central_australia", -21, 135)]
HHV = 6.25
LHV = 18.6 / 3.6


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def select(frame, lat, lon):
    la = "Latitude" if "Latitude" in frame else "latitude"
    lo = "Longitude" if "Longitude" in frame else "longitude"
    rows = frame[(frame[la] == lat) & (frame[lo] == lon)]
    if len(rows) != 1:
        raise ValueError(f"Expected one row for {lat}, {lon}; found {len(rows)}")
    return rows.iloc[0]


def scaled_capacity(q_mtpa, wind_mw, fixed_mw, tracking_mw, land, ratio=1.):
    """Fixed-design scaling: min of wind, PV and shared-union area ratios."""
    if q_mtpa <= 0 or min(wind_mw, fixed_mw, tracking_mw) < 0:
        raise ValueError("Invalid reference design")
    wind = wind_mw / 5.
    pv = (fixed_mw + ratio * tracking_mw) / land.solar_density_mw_per_km2
    limits = []
    if wind > 0:
        limits.append(land.wind_onshore_area_km2 / wind)
    if pv > 0:
        limits.append(land.solar_area_km2 / pv)
    if wind + pv <= 0:
        raise ValueError("No renewable footprint")
    independent = q_mtpa * min(limits)
    shared = q_mtpa * min(*limits, land.renewable_union_area_km2 / (wind + pv))
    return shared, independent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    sources = {}

    def read_csv(label, path):
        path = Path(path)
        sources[label] = {"path": str(path.resolve()), "sha256": digest(path)}
        return pd.read_csv(path)

    old_manifest = json.loads((CAMPAIGN / "audit/v3/summary.json").read_text())
    historical = read_csv("archived_shipping", old_manifest["inputs"]["historical"]["path"])
    historical = historical.drop_duplicates(["Latitude", "Longitude"], keep="last")
    rep = read_csv("historical_style_green_lory", old_manifest["inputs"]["rep_surface"]["path"])
    land = read_csv("native_paper_land", CAMPAIGN / "replication/native-paper-method-pilot-v1/max_capacities.csv")
    common = read_csv("cmg_control_land", CAMPAIGN / "revised/common-geography-fixed-pilot-v1/max_capacities.csv")
    legacy = read_csv("legacy_lcoa_designs_20231105", LEGACY / "results/2050_lcoa_global_20231105_-180to180mp.csv")
    regressed = read_csv("later_regressed_surface", LEGACY / "results/2050_4.5_lcoa_global_max_capacity.csv")
    grid = read_csv("green_lory_certified_grid", CAMPAIGN / "audit/supply-certified-grid-v1/site_comparison.csv")
    fixed_grid = read_csv("green_lory_fixed_threshold", CAMPAIGN / "audit/fixed-threshold-v2/site_comparison.csv")
    pv = read_csv("green_lory_fixed_tracking", CAMPAIGN / "audit/fixed-vs-tracking-v1/comparison.csv")
    pv = pv[pv.tracking_footprint_ratio == 2.]
    pilot = CAMPAIGN / "arc_received_fixed_v2/fixed_pv_3cells_v2/weather_used"
    weather_rows, weather_inventory = [], []
    for path in sorted((LEGACY / "data").glob("*.nc")):
        if not path.name.startswith(("Solar", "Wind")):
            continue
        with xr.open_dataset(path) as ds:
            weather_inventory.append({"path": str(path), "size_bytes": path.stat().st_size,
                "dimensions": dict(ds.sizes), "time_start": str(ds.time.values[0]),
                "time_end": str(ds.time.values[-1]),
                "longitude_min": float(ds.longitude.min()), "longitude_max": float(ds.longitude.max()),
                "global_attributes": dict(ds.attrs), "full_file_hash_checked": False})

    capacities, energy, archived_designs = [], [], []
    for site, lat, lon in SITES:
        cf = {}
        suffix = "" if lon < -60 else "1" if lon < 60 else "2"
        for stem, var, tech in [("Solar", "Solar", "fixed_pv"),
                               ("SolarTracking", "Solar", "tracking_pv"),
                               ("WindPowers", "Wind", "wind")]:
            source = LEGACY / "data" / f"{stem}{suffix}.nc"
            subset = pilot / f"{stem}.nc"
            with xr.open_dataset(source) as a, xr.open_dataset(subset) as b:
                x = a[var].sel(latitude=lat, longitude=lon).values
                y = b[var].sel(latitude=lat, longitude=lon).values
                assert len(x) == 8760 and np.isfinite(x).all()
                assert np.array_equal(a.time.values, b.time.values)
                np.testing.assert_array_equal(x, y)
                net = x * (.93 if tech == "wind" else 1.)
                cf[tech] = float(net.mean())
                weather_rows.append({"site": site, "latitude": lat, "longitude": lon,
                    "technology": tech, "raw_cf": float(x.mean()), "model_cf": cf[tech],
                    "wake_multiplier": .93 if tech == "wind" else 1.,
                    "legacy_to_pilot_exact_profile_match": True,
                    "profile_float64_sha256": hashlib.sha256(np.asarray(x, dtype="<f8").tobytes()).hexdigest(),
                    "source_path": str(source), "pilot_path": str(subset),
                    "pilot_file_sha256": digest(subset), "raw_peak": float(x.max()),
                    "raw_hours_above_one": int((x > 1).sum())})

        h, l, c, d, r, g, p, later, fg = [select(f, lat, lon) for f in
            (historical, land, common, legacy, rep, grid, pv, regressed, fixed_grid)]
        # Current legacy loads.csv implies 10 Mt/year, but it is NOT a pinned
        # manifest for the November 2023 run. Keep this result conditional.
        q_assumed = 10.
        legacy_shared, legacy_independent = scaled_capacity(q_assumed, d.Wind, d.Solar, d.SolarTracking, l)
        rep_shared, rep_independent = scaled_capacity(r.annual_ammonia_production_t / 1e6,
            r.wind_mw, r.solar_mw, r.solar_tracking_mw, l)
        capacities.append({"site": site, "latitude": lat, "longitude": lon,
            "archived_shipping_capacity_Mtpa": h.Max_capacity,
            "later_regressed_surface_capacity_Mtpa": later.Max_capacity,
            "legacy_lcoa_reference_assumed_Mtpa": q_assumed,
            "legacy_lcoa_native_reconstruction_shared_Mtpa": legacy_shared,
            "legacy_lcoa_native_reconstruction_independent_Mtpa": legacy_independent,
            "legacy_lcoa_reconstruction_status": "conditional_on_unverified_10Mtpa_reference_and_saved_design",
            "historical_style_green_lory_native_fixed_packing_shared_Mtpa": rep_shared,
            "historical_style_green_lory_native_fixed_packing_independent_Mtpa": rep_independent,
            "historical_style_status": "green_lory_emulation_not_legacy_lcoa_rerun;tracking_yield_with_paper_fixed_packing_is_diagnostic",
            "green_lory_cmg_both_reference_scaled_Mtpa": p.central_common_land_Mtpa,
            "green_lory_cmg_fixed_reference_scaled_Mtpa": p.fixed_common_land_Mtpa,
            "green_lory_cmg_maximum_feasible_tested_Mtpa": g.maximum_feasible_tested_Mtpa,
            "green_lory_cmg_ideal_energy_ceiling_Mtpa": g.ideal_energy_bound_Mtpa,
            "native_solar_shipping_area_km2": l.solar_area_km2,
            "cmg_solar_shipping_area_km2": c.solar_area_km2,
            "fixed_pv_density_MW_per_km2": l.solar_density_mw_per_km2,
            "legacy_required_shipping_share_if_shared_pct": 2. * h.Max_capacity / legacy_shared,
            "archived_shipping_LCOA_USD_per_t": h.LCOA,
            "legacy_lcoa_saved_LCOA_labelled_USD_per_t": d.Objective * 1000.,
            "legacy_lcoa_currency_status": "legacy_label_only;cost_year_and_AUD_USD_conversions_unverified",
            "historical_style_green_lory_LCOA_EUR2020_per_t": r.lcoa_eur_per_t,
            "green_lory_both_reference_LCOA_EUR2020_per_t": p.central_LCOA_EUR2020_per_t,
            "green_lory_fixed_reference_LCOA_EUR2020_per_t": p.fixed_LCOA_EUR2020_per_t,
            "green_lory_both_1Mtpa_LCOA_EUR2020_per_t": g.one_Mtpa_LCOA_EUR2020_per_t,
            "green_lory_fixed_1Mtpa_LCOA_EUR2020_per_t": fg.one_Mtpa_LCOA_EUR2020_per_t})
        archived_designs.append({"site": site, "source_row_id": d["Unnamed: 0"],
            "wind_MW": d.Wind, "fixed_MW": d.Solar, "tracking_MW": d.SolarTracking,
            "assumed_reference_Mtpa": q_assumed, "reference_status": "not_manifest_verified",
            "required_shared_area_km2_per_Mtpa": l.renewable_union_area_km2 / legacy_shared})

        for case, row in [("green_lory_historical_style_4h", r),
                          ("green_lory_both_control_1Mtpa", None),
                          ("green_lory_both_land_enforced_1Mtpa", None),
                          ("green_lory_fixed_land_enforced_1Mtpa", None)]:
            if row is None:
                folder = "control_unconstrained" if "control" in case else "q_1"
                run = ("arc_received_fixed_threshold_v2/fixed_land_threshold_3cells_v2" if "fixed" in case
                       else "arc_received_supply_attempt_v4/supply_curve_3cells_v4")
                path = CAMPAIGN / run / site / folder / "result.csv"
                row = read_csv(f"{site}_{case}", path).iloc[0]
            tonnes = row.annual_ammonia_production_t
            dispatched = row.power_bus_generator_supply_mwh
            assert abs(row.grid_energy_mwh) < 1
            available = 8760. * (row.wind_mw * cf["wind"] + row.solar_mw * cf["fixed_pv"]
                                  + row.solar_tracking_mw * cf["tracking_pv"])
            if dispatched <= 0 or available + 1 < dispatched:
                raise ValueError("Energy denominator does not close")
            energy.append({"site": site, "case": case, "production_tpa": tonnes,
                "primary_RE_dispatched_MWh": dispatched, "primary_RE_available_MWh": available,
                "electricity_dispatched_MWh_per_t": dispatched / tonnes,
                "efficiency_HHV_dispatched": tonnes * HHV / dispatched,
                "efficiency_LHV_dispatched": tonnes * LHV / dispatched,
                "efficiency_HHV_available": tonnes * HHV / available,
                "efficiency_LHV_available": tonnes * LHV / available,
                "curtailed_fraction": 1. - dispatched / available,
                "NH3_HHV_MWh_per_t": HHV, "NH3_LHV_MWh_per_t": LHV,
                "denominator": "primary renewable Generator dispatch to power bus; excludes fuel-cell/battery Link recycling",
                "legacy_lcoa_dispatch_efficiency_available": False})

    for name, records in [("capacity_cost_comparison", capacities), ("weather_cf_comparison", weather_rows),
                          ("energy_efficiency_comparison", energy), ("legacy_saved_designs", archived_designs)]:
        pd.DataFrame(records).to_csv(args.output / f"{name}.csv", index=False)
    for name in ["main.py", "main_mp.py", "p_location_class.py", "p_auxiliary.py", "land_availability.ipynb", "Basic_ammonia_plant/loads.csv"]:
        sources[f"legacy_code_{name}"] = {"path": str(LEGACY / name), "sha256": digest(LEGACY / name)}
    payload = {"qa_pass": True, "script_sha256": digest(Path(__file__)), "sources": sources,
        "weather_file_inventory": weather_inventory,
        "all_9_test_profiles_exactly_match": True,
        "legacy_reconstruction_is_verified_rerun": False,
        "native_land_sources_are_historical_vintages": False,
        "legacy_dispatch_efficiency_status": "unavailable_without_validated_dispatch_replay",
        "heating_value_source": "IRENA Innovation Outlook Renewable Ammonia (2022), Annex G: 22.5 MJ/kg HHV, 18.6 MJ/kg LHV",
        "heating_value_source_url": "https://www.irena.org/-/media/Files/IRENA/Agency/Publication/2022/May/IRENA_Innovation_Outlook_Ammonia_2022.pdf"}
    (args.output / "manifest.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(pd.DataFrame(capacities).to_string(index=False))
    print(pd.DataFrame(energy)[["site", "case", "electricity_dispatched_MWh_per_t", "efficiency_HHV_dispatched", "efficiency_LHV_dispatched", "curtailed_fraction"]].to_string(index=False))


if __name__ == "__main__":
    main()
