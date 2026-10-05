#!/usr/bin/env python
"""Merge legacy replication shards and build the replicated supplier table.

Capacity follows the rule recovered from the archived table, implemented in
`legacy_capacity.py` (complete wind/solar overlap, 140 MW/km2 PV without a
latitude adjustment, 7.3 MW/km2 wind, 2 % Table-2 suitable areas):

    Q [Mt/yr] = 1 * min( A_solar * d_pv / P_pv , A_wind * d_wind / P_wind )

with P in MW per 1 Mt/yr design. Both constants are empirical and reported
as such; alternatives can be produced with --d-pv/--d-wind. The output keeps
the archived table's column contract (Index, Latitude, Longitude, iso3,
country, LCOA, Production, Max_capacity, Electricity_Cost_Frac) so that
green-porpoise can consume it unchanged, plus explicit provenance columns.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from legacy_capacity import (D_PV_MW_PER_KM2, D_WIND_MW_PER_KM2, RULE_ID, STATED_RULE_ID, legacy_capacity,  # noqa: E402
                             load_land, load_stated_land, rule_provenance, sha256_file, stated_capacity,
                             stated_land_anchor, stated_rule_provenance)


def iter_summaries(run: Path):
    """Yield summary dicts from a run directory (shard_XX/<cell>/summary.json or flat) or from a
    summaries.jsonl file written by collect_summaries.py."""
    if run.is_file():
        with open(run) as fh:
            for line in fh:
                if line.strip():
                    yield json.loads(line)
        return
    for f in glob.glob(str(run / "shard_*" / "*__*__*" / "summary.json")) + glob.glob(str(run / "*__*__*" / "summary.json")):
        yield json.load(open(f))


def load_shards(run_dir: Path) -> pd.DataFrame:
    rows = []
    for s in iter_summaries(run_dir):
        c = s["capacities_mw"]
        meta = s.get("cell_meta", {})
        rows.append({"latitude": s["lat"], "longitude": s["lon"], "cell": s["cell"], "variant": s["variant"],
                     "wacc_rate": s["wacc_rate"], "lcoa_rerun": s["lcoa_usd_per_t"],
                     "P_wind": c.get("Wind", 0.0), "P_fixed": c.get("Solar", 0.0), "P_tracking": c.get("SolarTracking", 0.0),
                     "P_electrolysis": c.get("Electrolysis", 0.0), "P_hb": c.get("HB", 0.0),
                     "E_h2_store_mwh": c.get("CompressedH2Store", 0.0), "E_battery_mwh": c.get("Battery", 0.0),
                     "E_nh3_store_mwh": c.get("Ammonia", 0.0),
                     "elec_frac_rerun": s["electricity_capex_fraction_of_objective"], "curtailed": s["curtailed_fraction"],
                     "iso3": meta.get("iso3"), "country": meta.get("country"), "wacc_source": meta.get("wacc_source")})
    df = pd.DataFrame(rows)
    if df.duplicated(["latitude", "longitude"]).any():
        raise SystemExit("duplicate cells across shards")
    return df


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", type=Path, required=True, help="run directory with shard_XX/ subdirectories (or cell dirs), or a summaries.jsonl file from collect_summaries.py")
    p.add_argument("--land", type=Path, required=True,
                   help="archived_table_reproduction: legacy_land_areas.csv (2 %% Table-2 areas, centered, no exclusions); "
                        "stated_method: the green-lory land build CSV (paper_2pct_slope15.csv)")
    p.add_argument("--rule", choices=["archived_table_reproduction", "stated_method"], default="archived_table_reproduction")
    p.add_argument("--archived", type=Path, required=True, help="c_NH3_cost_4.5.csv for comparison and metadata")
    p.add_argument("--cells", type=Path, required=True, help="cells_archived_all_v1.csv (expected cell list)")
    p.add_argument("--d-pv", type=float, default=D_PV_MW_PER_KM2)
    p.add_argument("--d-wind", type=float, default=D_WIND_MW_PER_KM2)
    p.add_argument("--anchor", choices=["center", "southwest"], default="center")
    p.add_argument("--water-cost-usd2018-per-t", type=float, default=0.0,
                   help="Flat water cost added to every cell's LCOA (USD2018 per t NH3; the legacy plant has no water "
                        "cost). The key legacy run of 24 Sep 2026 uses 1.5 m3/t x 2 EUR2020/m3 = 3.0 EUR2020/t = "
                        "3.3245 USD2018/t (EUR2020 / 0.9024).")
    p.add_argument("--scenario-id", default=None, help="Scenario id recorded in the contract (default: the replication id)")
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite {args.output}")
    args.output.mkdir(parents=True)

    df = load_shards(args.run)
    cells = pd.read_csv(args.cells)
    expected = set(zip(cells.lat, cells.lon))
    got = set(zip(df.latitude, df.longitude))
    missing = sorted(expected - got)
    land = load_stated_land(args.land) if args.rule == "stated_method" else load_land(args.land, args.anchor)
    arch = pd.read_csv(args.archived).rename(columns={"Latitude": "latitude", "Longitude": "longitude"})
    out = df.merge(land, on=["latitude", "longitude"], how="left").merge(
        arch[["latitude", "longitude", "LCOA", "Max_capacity", "Electricity_Cost_Frac", "iso3", "country"]].rename(
            columns={"LCOA": "LCOA_archived", "Max_capacity": "Max_capacity_archived", "Electricity_Cost_Frac": "Electricity_Cost_Frac_archived",
                     "iso3": "iso3_archived", "country": "country_archived"}), on=["latitude", "longitude"], how="left")
    out["iso3"] = out["iso3"].fillna(out["iso3_archived"]); out["country"] = out["country"].fillna(out["country_archived"])
    out["P_pv"] = out.P_fixed + out.P_tracking
    if args.rule == "stated_method":
        rule = stated_capacity(out.A_solar_km2, out.A_wind_km2, out.P_pv, out.P_wind, out.d_pv_mw_per_km2)
        land_anchor = stated_land_anchor(args.land)
        provenance = stated_rule_provenance(land_anchor)
        rule_id = STATED_RULE_ID
    else:
        rule = legacy_capacity(out.A_solar_km2, out.A_wind_km2, out.P_pv, out.P_wind, args.d_pv, args.d_wind)
        provenance = rule_provenance(args.d_pv, args.d_wind, args.anchor)
        rule_id = RULE_ID
    for c in rule:
        out[c] = rule[c].to_numpy()
    out["lcoa_plant_usd_per_t"] = out.lcoa_rerun
    water_adder = float(getattr(args, "water_cost_usd2018_per_t", 0.0) or 0.0)
    out["water_cost_usd2018_per_t"] = water_adder
    out["LCOA"] = (out.lcoa_rerun + water_adder).round(2)
    out["Production"] = 1000000
    out["Electricity_Cost_Frac"] = out.elec_frac_rerun.round(9)
    out["Index"] = out.apply(lambda r: f"{int(r.latitude)}_{int(r.longitude)}_1000000.0", axis=1)
    out["lcoa_ratio_vs_archived"] = out.LCOA / out.LCOA_archived
    out["capacity_ratio_vs_archived"] = out.Max_capacity / out.Max_capacity_archived.replace(0, np.nan)
    out["capacity_rule"] = provenance.get("capacity_rule", provenance["formula"])
    cols = ["Index", "latitude", "longitude", "iso3", "country", "LCOA", "Production", "Max_capacity", "Electricity_Cost_Frac"]
    contract = out[cols].rename(columns={"latitude": "Latitude", "longitude": "Longitude"})
    contract.to_csv(args.output / "legacy_replicated_suppliers.csv", index=False)
    out.to_csv(args.output / "legacy_replicated_suppliers_full.csv", index=False)
    # green-porpoise supplier contract (same schema as the September green-lory exports)
    export_dir = args.output / "gpo_export"
    export_dir.mkdir()
    exported = contract[contract.Max_capacity > 0].copy()
    exported.to_csv(export_dir / "suppliers_USD2018.csv", index=False)
    contract_json = {
        "schema_version": 1, "scope": "global", "currency": "USD", "price_year": 2018,
        "source_currency": "USD", "source_price_year": 2018, "source_to_output_factor": 1.0,
        "scenario_id": getattr(args, "scenario_id", None) or ("legacy_lcoa_replicated_may2023_xcost45_tracking_4h_ameli" + ("_stated_land" if args.rule == "stated_method" else "")),
        "water_cost_usd2018_per_t_added_to_lcoa": water_adder,
        "run_id": str(args.run.parent.name if args.run.is_file() else args.run.name),
        "capacity_method": rule_id,
        "capacity_variant": args.rule,
        "capacity_rule": provenance.get("capacity_rule", provenance["formula"]),
        "capacity_rule_provenance": provenance,
        "capacity_units": "Mt_NH3_per_year (1 Mt reference multiplier)",
        "capacity_source_column": "Max_capacity",
        "electricity_cost_fraction_basis": "wind + PV annualised generation cost / plant objective (frozen legacy model)",
        "minimum_capacity_Mt_per_year": 0.0,
        "supplier_selection": "No cutoff applied at export; downstream selection applies the 1 Mt/yr cutoff and cheapest-4000 ranking",
        "source_rows": int(len(out)), "exported_positive_capacity_rows": int(len(exported)),
        "excluded_zero_capacity_rows": int((contract.Max_capacity <= 0).sum()),
        "inputs": {"run_dir": str(args.run.resolve()), **({"run_summaries_sha256": sha256_file(args.run)} if args.run.is_file() else {}), "land": {"path": str(args.land.resolve()), "sha256": sha256_file(args.land)},
                   "archived": {"path": str(args.archived.resolve()), "sha256": sha256_file(args.archived)},
                   "cells": {"path": str(args.cells.resolve()), "sha256": sha256_file(args.cells)}},
        "output": str((export_dir / "suppliers_USD2018.csv").resolve()),
        "output_sha256": sha256_file(export_dir / "suppliers_USD2018.csv"),
        "exporter_sha256": sha256_file(Path(__file__)),
        "note": "LCOA is the frozen legacy model's USD2018 output on the replicating configuration; it runs ~4 % above the archived table (LEGACY_REPLICATION_20260915.md). Requires regenerated routes.",
    }
    with open(export_dir / "contract.json", "w") as f:
        json.dump(contract_json, f, indent=2, sort_keys=True)

    ok = out.dropna(subset=["LCOA_archived"])
    pos = ok[(ok.Max_capacity_archived > 0) & (ok.Max_capacity > 0)]
    def q(s): return {str(k): round(float(v), 4) for k, v in s.quantile([.05, .25, .5, .75, .95]).items()}
    summary = {
        "n_cells": int(len(out)), "n_expected": int(len(expected)), "n_missing": len(missing), "missing_first": missing[:20],
        "rule": args.rule, "d_pv_MW_km2": ("latitude-dependent" if args.rule == "stated_method" else args.d_pv),
        "d_wind_MW_km2": (5.0 if args.rule == "stated_method" else args.d_wind), "anchor": (land_anchor if args.rule == "stated_method" else args.anchor),
        "cells_without_land_row": int(out.A_solar_km2.isna().sum()),
        "lcoa_ratio": q(ok.lcoa_ratio_vs_archived),
        "capacity_ratio_positive": q(pos.capacity_ratio_vs_archived),
        "capacity_within_10pct": float((abs(pos.capacity_ratio_vs_archived - 1) < 0.1).mean()),
        "capacity_within_25pct": float((abs(pos.capacity_ratio_vs_archived - 1) < 0.25).mean()),
        "eligible_ge_1Mt_archived": int((ok.Max_capacity_archived >= 1).sum()),
        "eligible_ge_1Mt_replicated": int((ok.Max_capacity >= 1).sum()),
        "eligible_both": int(((ok.Max_capacity_archived >= 1) & (ok.Max_capacity >= 1)).sum()),
        "total_capacity_archived_Mtpa": float(ok.Max_capacity_archived.sum()), "total_capacity_replicated_Mtpa": float(ok.Max_capacity.sum()),
        "australia": {
            "eligible_archived": int(((ok.country == "Australia") & (ok.Max_capacity_archived >= 1)).sum()),
            "eligible_replicated": int(((ok.country == "Australia") & (ok.Max_capacity >= 1)).sum()),
            "capacity_archived_Mtpa": float(ok[ok.country == "Australia"].Max_capacity_archived.sum()),
            "capacity_replicated_Mtpa": float(ok[ok.country == "Australia"].Max_capacity.sum()),
        },
        "limiting_technology_counts": out.limiting_technology.value_counts().to_dict(),
    }
    with open(args.output / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
