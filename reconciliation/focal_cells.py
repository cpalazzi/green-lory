#!/usr/bin/env python3
"""Land, plant cost and capacity of every kept run at the three focal cells.

Reads the accepted outputs named in reconciliation/run_store.csv (local copies under
results/campaigns/) and writes one long table, focal_cells.csv, plus focal_cells.md:

    python reconciliation/focal_cells.py --output results/campaigns/reconciliation_final_20260916_v1/focal_cells_v3

LCOA is reported in EUR2020/t; legacy and contract values in USD2018 are converted with
the documented factor 0.9024 (EUR2020 per USD2018). Capacity is the supplier capacity in
Mt/yr at the stated land share. Nothing is recomputed from the models.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CAMPAIGNS = ROOT / "results" / "campaigns"
EUR2020_PER_USD2018 = 0.9024
FOCAL = {"Atacama": (-23.0, -69.0), "NW Australia": (-23.0, 117.0), "Central Australia": (-21.0, 135.0)}
KEYRUNS = "keyruns_20260923_v1/30_keyruns/{run}/runs/20260928-v4/{stage}/{file}"
VERSCHUUR = "verschuur_reconcile_20260907_v1"

LAND_SOURCES = {
    "legacy land step, 2 %, no exclusions": ("legacy_lcoa_20260915_v1/audit/legacy-land-areas-v1/legacy_land_areas.csv",
                                             "center_solar_shipping_km2", "center_wind_shipping_km2", None),
    "September table (mislabelled), 2 %": (f"{VERSCHUUR}/arc_received_20260914/paper_2pct_slope15.csv",
                                           "solar_area_km2", "wind_onshore_area_km2", "land_exclusion_factor"),
    "centred table, 2 %": ("land_center_20260923_v1/max_capacities_center_2pct_slope15.csv",
                           "solar_area_km2", "wind_onshore_area_km2", "land_exclusion_factor"),
    "centred table, 20 %": ("land_center_20260923_v1/max_capacities_center_20pct_slope15.csv",
                            "solar_area_km2", "wind_onshore_area_km2", "land_exclusion_factor"),
}

# run id -> (label, file, kind); kind fixes the column contract of the file
RUN_SOURCES = {
    "archived": ("archived table (Salmon 2023)", f"{VERSCHUUR}/historical/git-0a63616/data/c_NH3_cost_4.5.csv", "contract_usd"),
    "ll_xcost45_ameli_bflat_wnone_landarch_track_4h": (
        "legacy replication", "legacy_lcoa_20260915_v1/audit/global-supplier-table-arc-v1/legacy_replicated_suppliers_full.csv", "legacy"),
    "rep_way2050_flat_amelired_4h_tracking_nominal_h2_legacyrule": (
        "green-lory reconciling, legacy capacity rule", f"{VERSCHUUR}/lory/global_exports/20260915-v1/rep_legacy_rule/suppliers_USD2018.csv",
        "contract_usd"),
    "ll_way2050_ameli_bflat_wflat_landleg2c_track_4h_glannuity": (
        "legacy key run (stated land method, centred 2 %)",
        "keyruns_20260923_v1/exports_20260928-v4_v1/ll_way2050_ameli_bflat_wflat_landleg2c_track_4h_glannuity/legacy_replicated_suppliers_full.csv",
        "legacy"),
    "gl_dea2050_wacc5_bflat_wflat_land20c_fixed": (
        "green-lory improved, uniform 5 %, 20 %",
        KEYRUNS.format(run="gl_dea2050_wacc5_bflat_wflat_land20c_fixed", stage="global", file="merged/global_run_results.csv"), "lory"),
    "gl_dea2050_ameli_bflat_wflat_land20c_fixed": (
        "green-lory improved, Ameli, 20 %",
        KEYRUNS.format(run="gl_dea2050_ameli_bflat_wflat_land20c_fixed", stage="global", file="merged/global_run_results.csv"), "lory"),
    "gl_dea2050_wacc5_bflat_wflat_land2c_fixed": (
        "green-lory improved, uniform 5 %, 2 %",
        KEYRUNS.format(run="gl_dea2050_wacc5_bflat_wflat_land2c_fixed", stage="derived", file="global_run_results.csv"), "lory"),
}

PILOTS = {
    "gl_way2050_ameli_bflat_wnone_land2c_fixed_pilot3cells": "land_reconcile_20260914_v1/arc_received_colocated_v6_center/supply_curve_v6_fixed_center/supply_points.csv",
    "gl_way2050_ameli_bflat_wnone_land2c_track_pilot3cells": "land_reconcile_20260914_v1/arc_received_colocated_v6_center/supply_curve_v6_track_center/supply_points.csv",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def at_focal(df: pd.DataFrame, lat: str, lon: str) -> dict[str, pd.Series]:
    rows = {}
    for name, (la, lo) in FOCAL.items():
        match = df[(df[lat].round(3) == la) & (df[lon].round(3) == lo)]
        if len(match) != 1:
            raise ValueError(f"{name}: expected one row at ({la}, {lo}), found {len(match)}")
        rows[name] = match.iloc[0]
    return rows


def land_rows(sources: list) -> list[dict]:
    out = []
    for label, (rel, solar, wind, exclusion) in LAND_SOURCES.items():
        path = CAMPAIGNS / rel
        sources.append({"item": f"land: {label}", "path": rel, "sha256": sha256_file(path)})
        for name, row in at_focal(pd.read_csv(path), "latitude", "longitude").items():
            out.append({"cell": name, "table": "land", "source": label,
                        "solar_area_km2": float(row[solar]), "wind_area_km2": float(row[wind]),
                        "exclusion_factor_vs_share": None if exclusion is None
                        else float(row[exclusion]) / float(row.get("land_competition_fraction", 1.0)),
                        "pv_density_mw_per_km2": float(row["solar_density_mw_per_km2"]) if "solar_density_mw_per_km2" in row else None})
    return out


def run_rows(sources: list) -> list[dict]:
    out = []
    for run_id, (label, rel, kind) in RUN_SOURCES.items():
        path = CAMPAIGNS / rel
        sources.append({"item": f"run: {run_id}", "path": rel, "sha256": sha256_file(path)})
        df = pd.read_csv(path)
        if kind == "lory":
            rows = at_focal(df, "latitude", "longitude")
            for name, row in rows.items():
                wind, pv = float(row["wind_mw"]), float(row["solar_mw"]) + float(row["solar_tracking_mw"])
                out.append({"cell": name, "table": "plant", "run_id": run_id, "source": label,
                            "lcoa_eur2020_per_t": float(row["lcoa_eur_per_t"]),
                            "capacity_mt_per_year": float(row["scaled_design_max_gridless_onshore_ammonia_capacity_t"]) / 1e6,
                            "wind_mw_per_mt": wind, "pv_mw_per_mt": pv,
                            "limiting": row["scaled_design_limiting_constraint"]})
        elif kind == "legacy":
            rows = at_focal(df, "latitude", "longitude")
            for name, row in rows.items():
                out.append({"cell": name, "table": "plant", "run_id": run_id, "source": label,
                            "lcoa_eur2020_per_t": float(row["LCOA"]) * EUR2020_PER_USD2018,
                            "capacity_mt_per_year": float(row["Max_capacity"]),
                            "wind_mw_per_mt": float(row["P_wind"]), "pv_mw_per_mt": float(row["P_fixed"]) + float(row["P_tracking"]),
                            "limiting": row["limiting_technology"]})
        else:  # contract_usd: Index, Latitude, Longitude, LCOA (USD2018/t), Max_capacity (Mt/yr)
            rows = at_focal(df, "Latitude", "Longitude")
            for name, row in rows.items():
                out.append({"cell": name, "table": "plant", "run_id": run_id, "source": label,
                            "lcoa_eur2020_per_t": float(row["LCOA"]) * EUR2020_PER_USD2018,
                            "capacity_mt_per_year": float(row["Max_capacity"])})
    return out


def pilot_rows(sources: list) -> list[dict]:
    out = []
    for run_id, rel in PILOTS.items():
        path = CAMPAIGNS / rel
        sources.append({"item": f"pilot: {run_id}", "path": rel, "sha256": sha256_file(path)})
        df = pd.read_csv(path)
        for name, (la, lo) in FOCAL.items():
            site = df[(df.latitude.round(3) == la) & (df.longitude.round(3) == lo)]
            feasible = site[site.status == "feasible"].sort_values("target_Mtpa")
            if feasible.empty:
                raise ValueError(f"{run_id}: no feasible point at {name}")
            top = feasible.iloc[-1]
            # The smallest tested quantity is below the land limit, so its cost is the unconstrained design's.
            out.append({"cell": name, "table": "finite-site pilot", "run_id": run_id,
                        "source": "finite-site supply curve, Way 2050 costs, 2 %, " + ("fixed PV" if "fixed" in run_id else "tracking PV"),
                        "lcoa_eur2020_per_t": float(feasible.LCOA_EUR2020_per_t.iloc[0]),
                        "capacity_mt_per_year": float(top.target_Mtpa),
                        "lcoa_at_capacity_eur2020_per_t": float(top.LCOA_EUR2020_per_t),
                        "first_infeasible_mt_per_year": float(site[site.status == "infeasible"].target_Mtpa.min())})
    return out


def markdown(frame: pd.DataFrame) -> str:
    def cell(value) -> str:
        if isinstance(value, float):
            return "" if pd.isna(value) else f"{value:,.2f}"
        return str(value)
    lines = ["| " + " | ".join(frame.columns) + " |", "|" + "---|" * len(frame.columns)]
    lines += ["| " + " | ".join(cell(v) for v in row) + " |" for row in frame.itertuples(index=False)]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", type=Path, required=True, help="new directory (must not exist)")
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite {args.output}")
    sources: list[dict] = []
    table = pd.DataFrame(land_rows(sources) + run_rows(sources) + pilot_rows(sources))
    lines = ["# Focal cells", "", "LCOA in EUR2020/t; capacity in Mt/yr at the land share of the run.", ""]
    for kind, frame in table.groupby("table", sort=False):
        lines += [f"## {kind}", "", markdown(frame.dropna(axis=1, how="all").drop(columns="table")), ""]
    staging = args.output.with_name(args.output.name + ".partial")
    staging.mkdir(parents=True)  # renamed only when every file is written
    table.to_csv(staging / "focal_cells.csv", index=False)
    (staging / "focal_cells.md").write_text("\n".join(lines))
    (staging / "sources.json").write_text(json.dumps({"script_sha256": sha256_file(Path(__file__)),
                                                      "sources": sources}, indent=2) + "\n")
    staging.rename(args.output)
    print((args.output / "focal_cells.md").read_text())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
