#!/usr/bin/env python3
"""Consolidate every accepted result for the three focal cells into one table.

Focal cells: Atacama (-23, -69), northwest Australia (-23, 117) and central
Australia (-21, 135). Each column names its source file; every source is
hashed into ``sources.json``. Costs are reported in the currency of the source
and, where the documented Way conversion applies (USD2018 x 0.9024 = EUR2020),
also as USD2018-equivalent. Nothing is recomputed from model outputs here
except unit conversions and the ratio columns.

    python build_three_cell_comparison.py --campaigns results/campaigns --output <new dir>
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
from pathlib import Path

import pandas as pd

FX = 0.9024  # EUR2020 per USD2018 (Way YAML documented conversion)
CELLS = [("atacama", -23, -69), ("northwest_australia", -23, 117), ("central_australia", -21, 135)]
IDX = {c[0]: f"{c[1]}_{c[2]}_1000000.0" for c in CELLS}
LEGACY_NAMES = {"atacama": "cm23_m69", "northwest_australia": "cm23_117", "central_australia": "cm21_135"}


def sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


class Sources:
    def __init__(self):
        self.used = {}

    def path(self, key: str, p: Path, required: bool = True) -> Path | None:
        if not p.exists():
            if required:
                raise SystemExit(f"missing source {key}: {p}")
            self.used[key] = {"path": str(p), "status": "absent"}
            return None
        self.used[key] = {"path": str(p.resolve()), "sha256": sha256_file(p)}
        return p


_JSONL_CACHE: dict = {}


def legacy_summary(run_dir: Path, name: str, variant: str, wacc: str) -> dict | None:
    """Per-cell summary from a run directory (shard_XX/<cell>/summary.json or flat) or from a
    summaries.jsonl file written by collect_summaries.py."""
    if run_dir.is_file():
        if run_dir not in _JSONL_CACHE:
            table = {}
            with open(run_dir) as fh:
                for line in fh:
                    if line.strip():
                        s = json.loads(line)
                        table[(s["cell"], s["variant"], s["wacc_name"])] = s
            _JSONL_CACHE[run_dir] = table
        return _JSONL_CACHE[run_dir].get((name, variant, wacc))
    hits = glob.glob(str(run_dir / "shard_*" / f"{name}__{variant}__{wacc}" / "summary.json")) + \
        glob.glob(str(run_dir / f"{name}__{variant}__{wacc}" / "summary.json"))
    if len(hits) > 1:
        raise SystemExit(f"ambiguous legacy result for {name} in {run_dir}")
    return json.load(open(hits[0])) if hits else None


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--campaigns", type=Path, required=True)
    p.add_argument("--legacy-global-run", type=Path, default=None,
                   help="legacy global surface run directory with shard_XX/ (ARC fetch); optional")
    p.add_argument("--legacy-supplier-table", type=Path, default=None,
                   help="legacy_replicated_suppliers_full.csv built from the global legacy run; optional")
    p.add_argument("--networks", type=Path, default=None, help="compare_networks.py output directory; optional")
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    if a.output.exists():
        raise SystemExit(f"refusing to overwrite {a.output}")
    S = Sources()
    C = a.campaigns
    V = C / "verschuur_reconcile_20260907_v1"
    L = C / "land_reconcile_20260914_v1"
    G = C / "legacy_lcoa_20260915_v1"

    archived = pd.read_csv(S.path("archived_supplier_table", V / "historical/git-0a63616/data/c_NH3_cost_4.5.csv"))
    archived = archived.drop_duplicates(["Latitude", "Longitude"], keep="last").set_index(["Latitude", "Longitude"])
    later = pd.read_csv(S.path("legacy_three_cell_capacity_cost", L / "audit/model-evidence-v2/capacity_cost_comparison.csv")).set_index("site")
    attribution = pd.read_csv(S.path("lory_global_three_cell_attribution", V / "comparison/global-20260914-v2/three_cell_attribution.csv"))
    attribution = attribution.set_index(["latitude", "longitude"])
    rep_rule = pd.read_csv(S.path("lory_rep_costs_legacy_rule", V / "lory/global_exports/20260915-v1/rep_legacy_rule/cells_full.csv")).set_index("Index")
    supply = pd.read_csv(S.path("lory_land_enforced_supply_grid", L / "audit/supply-certified-grid-v1/site_comparison.csv")).set_index("site")
    supply_pts = pd.read_csv(S.path("lory_land_enforced_supply_points", L / "audit/supply-certified-grid-v1/validated_points.csv"))
    fixed = pd.read_csv(S.path("lory_fixed_only_threshold", L / "audit/fixed-threshold-v2/site_comparison.csv")).set_index("site")
    energy = pd.read_csv(S.path("energy_efficiency", L / "audit/model-evidence-v2/energy_efficiency_comparison.csv"))
    cf = pd.read_csv(S.path("weather_cf", L / "audit/model-evidence-v2/weather_cf_comparison.csv"))
    three = G / "three_cells_may2023_xcost45_tracking_v1"
    S.path("legacy_three_cell_run_manifest", three / "manifest.json")
    sample_tbl = pd.read_csv(S.path("legacy_sample563_supplier_table", G / "audit/sample563-supplier-table-v2/legacy_replicated_suppliers_full.csv")).set_index("Index")
    legacy_global_tbl = None
    if a.legacy_supplier_table:
        legacy_global_tbl = pd.read_csv(S.path("legacy_global_supplier_table", a.legacy_supplier_table)).set_index("Index")
    focal_net = None
    if a.networks:
        focal_net = pd.read_csv(S.path("network_focal_cells", a.networks / "focal_cells.csv"))

    rows = []
    for site, lat, lon in CELLS:
        idx = IDX[site]
        r = {"site": site, "latitude": lat, "longitude": lon, "Index": idx}
        arc = archived.loc[(lat, lon)]
        r["archived_LCOA_USD_per_t"] = float(arc.LCOA)
        r["archived_Max_capacity_Mtpa"] = float(arc.Max_capacity)
        r["archived_Electricity_Cost_Frac"] = float(arc.Electricity_Cost_Frac)
        r["later_regressed_capacity_Mtpa_not_a_target"] = float(later.loc[site, "later_regressed_surface_capacity_Mtpa"])
        # weather (identical old profile stack in legacy and green-lory pilots)
        for tech in ("fixed_pv", "tracking_pv", "wind"):
            w = cf[(cf.site == site) & (cf.technology == tech)].iloc[0]
            r[f"cf_{tech}_raw"] = float(w.raw_cf)
            r[f"cf_{tech}_model"] = float(w.model_cf)
        # legacy-lcoa replication (frozen 94de8ce, x_Cost RCP4.5 2050, tracking, 4 h, Ameli 5.1 %)
        s = legacy_summary(three, site, "stated_4h_mean", "ameli_reduced")
        if s is None:
            raise SystemExit(f"legacy three-cell result missing for {site}")
        caps = s["capacities_mw"]
        r["legacy_rep_LCOA_USD2018_per_t"] = s["lcoa_usd_per_t"]
        r["legacy_rep_over_archived"] = s["lcoa_usd_per_t"] / float(arc.LCOA)
        r["legacy_rep_wind_MW"] = caps.get("Wind", 0.0)
        r["legacy_rep_fixed_pv_MW"] = caps.get("Solar", 0.0)
        r["legacy_rep_tracking_pv_MW"] = caps.get("SolarTracking", 0.0)
        r["legacy_rep_electrolysis_MW"] = caps.get("Electrolysis", 0.0)
        r["legacy_rep_h2_store_MWh"] = caps.get("CompressedH2Store", 0.0)
        r["legacy_rep_battery_MWh"] = caps.get("Battery", 0.0)
        r["legacy_rep_curtailed_fraction"] = s["curtailed_fraction"]
        r["legacy_rep_electricity_MWh_per_t"] = s["primary_electricity_mwh_per_t_snapshot_basis"]
        r["legacy_rep_electricity_capex_fraction"] = s["electricity_capex_fraction_of_objective"]
        r["legacy_rep_snapshots"] = s["snapshots_used"]
        # same configuration on the workbook 8 % basis and hourly, for reference
        s8 = legacy_summary(three, site, "stated_4h_mean", "workbook8")
        r["legacy_rep_workbook8pct_LCOA_USD2018_per_t"] = None if s8 is None else s8["lcoa_usd_per_t"]
        sh = legacy_summary(three, site, "hourly", "ameli_reduced")
        r["legacy_rep_hourly_LCOA_USD2018_per_t"] = None if sh is None else sh["lcoa_usd_per_t"]
        # the same cells inside the global legacy surface (ARC), if given
        if a.legacy_global_run:
            sg = legacy_summary(a.legacy_global_run, LEGACY_NAMES[site], "stated_4h_mean", "cell_wacc")
            r["legacy_global_surface_LCOA_USD2018_per_t"] = None if sg is None else sg["lcoa_usd_per_t"]
            r["legacy_global_surface_tracking_pv_MW"] = None if sg is None else sg["capacities_mw"].get("SolarTracking", 0.0)
            r["legacy_global_surface_wind_MW"] = None if sg is None else sg["capacities_mw"].get("Wind", 0.0)
        # legacy capacity rule (recovered) applied to the replicated designs
        tbl = legacy_global_tbl if legacy_global_tbl is not None else sample_tbl
        r["legacy_rule_table"] = "global" if legacy_global_tbl is not None else "sample563"
        t = tbl.loc[idx]
        r["legacy_rule_A_solar_2pct_km2"] = float(t.A_solar_km2)
        r["legacy_rule_A_wind_2pct_km2"] = float(t.A_wind_km2)
        r["legacy_rule_Q_pv_limit_Mtpa"] = float(t.Q_pv_limit_Mtpa)
        r["legacy_rule_Q_wind_limit_Mtpa"] = None if t.Q_wind_limit_Mtpa == float("inf") else float(t.Q_wind_limit_Mtpa)
        r["legacy_rule_Max_capacity_Mtpa"] = float(t.Max_capacity)
        r["legacy_rule_over_archived"] = float(t.Max_capacity) / float(arc.Max_capacity)
        r["legacy_rule_limiting_technology"] = t.limiting_technology
        # green-lory replication (4 h historical-style) and central (hourly) surfaces
        g = attribution.loc[(float(lat), float(lon))]
        r["lory_rep_LCOA_USD2018_per_t"] = float(g.cost_rep_USD2018_per_t)
        r["lory_rep_LCOA_EUR2020_per_t"] = float(g.cost_rep_USD2018_per_t) * FX
        r["lory_rep_capacity_Mtpa_september_land_method"] = float(g.capacity_rep_Mtpa)
        r["lory_central_LCOA_USD2018_per_t"] = float(g.cost_central_USD2018_per_t)
        r["lory_central_LCOA_EUR2020_per_t"] = float(g.cost_central_USD2018_per_t) * FX
        r["lory_central_capacity_Mtpa_september_land_method"] = float(g.capacity_central_Mtpa)
        r["lory_central_minus_rep_USD2018_per_t"] = float(g.central_minus_rep_USD2018_per_t)
        # green-lory replication designs under the legacy capacity rule (network run D input)
        d = rep_rule.loc[idx]
        r["lory_rep_wind_MW"] = float(d.P_wind)
        r["lory_rep_pv_MW"] = float(d.P_pv)
        r["lory_rep_capacity_Mtpa_legacy_rule"] = float(d.Max_capacity)
        r["lory_rep_legacy_rule_limiting"] = d.limiting_technology
        # green-lory land-enforced finite-site results (corrected common land, exclusive sharing)
        sp = supply.loc[site]
        r["lory_land_enforced_control_LCOA_EUR2020_per_t"] = float(sp.control_LCOA_EUR2020_per_t)
        r["lory_land_enforced_1Mtpa_LCOA_EUR2020_per_t"] = float(sp.one_Mtpa_LCOA_EUR2020_per_t)
        r["lory_land_enforced_1Mtpa_LCOA_USD2018_per_t"] = float(sp.one_Mtpa_LCOA_EUR2020_per_t) / FX
        r["lory_land_enforced_max_feasible_tested_Mtpa"] = float(sp.maximum_feasible_tested_Mtpa)
        r["lory_land_enforced_energy_ceiling_Mtpa"] = float(sp.ideal_energy_bound_Mtpa)
        r["archived_over_lory_energy_ceiling"] = float(arc.Max_capacity) / float(sp.ideal_energy_bound_Mtpa)
        pts = supply_pts[(supply_pts.site == site) & (supply_pts.status == "feasible")].sort_values("target_Mtpa")
        r["lory_land_enforced_feasible_points_Mtpa_LCOA_EUR2020"] = "; ".join(f"{t.target_Mtpa:g}:{t.LCOA_EUR2020_per_t:.2f}" for t in pts.itertuples())
        fx = fixed.loc[site]
        r["lory_fixed_only_1Mtpa_LCOA_EUR2020_per_t"] = float(fx.one_Mtpa_LCOA_EUR2020_per_t)
        r["lory_fixed_only_control_LCOA_EUR2020_per_t"] = float(fx.control_LCOA_EUR2020_per_t)
        # efficiencies (green-lory land-enforced 1 Mt/yr, both PV options)
        e = energy[(energy.site == site) & (energy.case == "green_lory_both_land_enforced_1Mtpa")].iloc[0]
        r["lory_land_enforced_electricity_MWh_per_t"] = float(e.electricity_dispatched_MWh_per_t)
        r["lory_land_enforced_efficiency_HHV"] = float(e.efficiency_HHV_dispatched)
        r["lory_land_enforced_efficiency_LHV"] = float(e.efficiency_LHV_dispatched)
        r["lory_land_enforced_curtailed_fraction"] = float(e.curtailed_fraction)
        eh = energy[(energy.site == site) & (energy.case == "green_lory_historical_style_4h")].iloc[0]
        r["lory_rep_electricity_MWh_per_t"] = float(eh.electricity_dispatched_MWh_per_t)
        # network roles
        if focal_net is not None:
            for t in focal_net[focal_net.Index == idx].itertuples():
                r[f"network_{t.run}_production_Mtpa"] = 0.0 if not t.selected else t.production_Mtpa
                r[f"network_{t.run}_selected"] = bool(t.selected)
        rows.append(r)

    a.output.mkdir(parents=True)
    df = pd.DataFrame(rows)
    df.to_csv(a.output / "three_cells.csv", index=False)
    df.set_index("site").T.to_csv(a.output / "three_cells_transposed.csv")
    (a.output / "sources.json").write_text(json.dumps({"sources": S.used, "fx_eur2020_per_usd2018": FX,
                                                        "script_sha256": sha256_file(Path(__file__))}, indent=2) + "\n")

    def fmt(v, nd=2):
        return "" if v is None or (isinstance(v, float) and pd.isna(v)) else (f"{v:.{nd}f}" if isinstance(v, (int, float)) else str(v))

    def block(title, items):
        lines = [f"### {title}", "", "| Quantity | Atacama | NW Australia | Central Australia |", "|---|---:|---:|---:|"]
        for label, col, nd in items:
            if col not in df:
                continue
            lines.append(f"| {label} | " + " | ".join(fmt(df.loc[i, col], nd) for i in range(3)) + " |")
        return "\n".join(lines) + "\n"

    md = ["# Three focal cells: consolidated evidence", "",
          "Sources and hashes: `sources.json`. Costs: USD2018 for the archived table and the legacy model; "
          "green-lory EUR2020 converted with USD2018 x 0.9024 = EUR2020. Capacities in Mt NH3/yr.", ""]
    md.append(block("Archived shipping input (Salmon spring-2023 run)", [
        ("LCOA, USD/t", "archived_LCOA_USD_per_t", 2), ("Max_capacity, Mt/yr", "archived_Max_capacity_Mtpa", 3),
        ("Electricity_Cost_Frac", "archived_Electricity_Cost_Frac", 3),
        ("Later regressed notebook capacity (not a target)", "later_regressed_capacity_Mtpa_not_a_target", 3)]))
    md.append(block("Weather (old 2019 profile stack, identical in both models)", [
        ("Fixed PV CF", "cf_fixed_pv_raw", 4), ("Tracking PV CF", "cf_tracking_pv_raw", 4),
        ("Wind CF raw", "cf_wind_raw", 4), ("Wind CF x0.93 wake", "cf_wind_model", 4)]))
    md.append(block("Legacy-lcoa replication (frozen 94de8ce; x_Cost RCP4.5 2050; tracking; 4 h; Ameli 5.1 %)", [
        ("LCOA, USD2018/t", "legacy_rep_LCOA_USD2018_per_t", 2), ("Ratio to archived", "legacy_rep_over_archived", 4),
        ("Same in global ARC surface, USD2018/t", "legacy_global_surface_LCOA_USD2018_per_t", 2),
        ("Same on workbook 8 % basis, USD2018/t", "legacy_rep_workbook8pct_LCOA_USD2018_per_t", 2),
        ("Same hourly, USD2018/t", "legacy_rep_hourly_LCOA_USD2018_per_t", 2),
        ("Wind, MW per Mt/yr", "legacy_rep_wind_MW", 0), ("Fixed PV, MW", "legacy_rep_fixed_pv_MW", 0),
        ("Tracking PV, MW", "legacy_rep_tracking_pv_MW", 0), ("Electrolysis, MW", "legacy_rep_electrolysis_MW", 0),
        ("H2 store, MWh", "legacy_rep_h2_store_MWh", 0), ("Battery, MWh", "legacy_rep_battery_MWh", 0),
        ("Curtailed fraction", "legacy_rep_curtailed_fraction", 4), ("Electricity, MWh/t", "legacy_rep_electricity_MWh_per_t", 3),
        ("Electricity capex fraction", "legacy_rep_electricity_capex_fraction", 3)]))
    md.append(block("Legacy capacity rule (recovered: 140 MW/km2 PV, 7.3 MW/km2 wind, complete overlap, 2 % Table-2 areas)", [
        ("Table used", "legacy_rule_table", 0),
        ("2 % solar area, km2", "legacy_rule_A_solar_2pct_km2", 2), ("2 % wind area, km2", "legacy_rule_A_wind_2pct_km2", 2),
        ("PV-limited quantity, Mt/yr", "legacy_rule_Q_pv_limit_Mtpa", 3), ("Wind-limited quantity, Mt/yr", "legacy_rule_Q_wind_limit_Mtpa", 3),
        ("Replicated Max_capacity, Mt/yr", "legacy_rule_Max_capacity_Mtpa", 3), ("Ratio to archived", "legacy_rule_over_archived", 3),
        ("Limiting technology", "legacy_rule_limiting_technology", 0)]))
    md.append(block("green-lory surfaces (September campaign; 52,702 cells; September land method)", [
        ("Replication LCOA, USD2018/t", "lory_rep_LCOA_USD2018_per_t", 2), ("Replication capacity, Mt/yr", "lory_rep_capacity_Mtpa_september_land_method", 3),
        ("Central LCOA, USD2018/t", "lory_central_LCOA_USD2018_per_t", 2), ("Central capacity, Mt/yr", "lory_central_capacity_Mtpa_september_land_method", 3),
        ("Central minus replication, USD2018/t", "lory_central_minus_rep_USD2018_per_t", 2),
        ("Replication wind, MW", "lory_rep_wind_MW", 0), ("Replication PV (fixed+tracking), MW", "lory_rep_pv_MW", 0),
        ("Replication designs under legacy rule, Mt/yr", "lory_rep_capacity_Mtpa_legacy_rule", 3),
        ("Electricity, MWh/t (replication)", "lory_rep_electricity_MWh_per_t", 3)]))
    md.append(block("green-lory land-enforced finite-site solves (corrected common land, exclusive sharing, hourly)", [
        ("Unconstrained control LCOA, EUR2020/t", "lory_land_enforced_control_LCOA_EUR2020_per_t", 2),
        ("1 Mt/yr LCOA, EUR2020/t", "lory_land_enforced_1Mtpa_LCOA_EUR2020_per_t", 2),
        ("1 Mt/yr LCOA, USD2018/t", "lory_land_enforced_1Mtpa_LCOA_USD2018_per_t", 2),
        ("Highest tested feasible quantity, Mt/yr", "lory_land_enforced_max_feasible_tested_Mtpa", 2),
        ("Annual-energy ceiling, Mt/yr", "lory_land_enforced_energy_ceiling_Mtpa", 3),
        ("Archived capacity / energy ceiling", "archived_over_lory_energy_ceiling", 2),
        ("Feasible points (Mt/yr:EUR/t)", "lory_land_enforced_feasible_points_Mtpa_LCOA_EUR2020", 0),
        ("Fixed-only 1 Mt/yr LCOA, EUR2020/t", "lory_fixed_only_1Mtpa_LCOA_EUR2020_per_t", 2),
        ("Electricity, MWh/t", "lory_land_enforced_electricity_MWh_per_t", 3),
        ("Efficiency HHV", "lory_land_enforced_efficiency_HHV", 4), ("Efficiency LHV", "lory_land_enforced_efficiency_LHV", 4),
        ("Curtailed fraction", "lory_land_enforced_curtailed_fraction", 4)]))
    if focal_net is not None:
        items = [(c.replace("network_", "").replace("_production_Mtpa", "") + ", production Mt/yr", c, 3)
                 for c in df.columns if c.startswith("network_") and c.endswith("_production_Mtpa")]
        md.append(block("Role in the shipping networks (production at the cell, Mt/yr; 0 = selected but idle or not selected)", items))
    (a.output / "three_cells.md").write_text("\n".join(md))
    print("\n".join(md))


if __name__ == "__main__":
    main()
