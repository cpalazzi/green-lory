#!/usr/bin/env python
"""Attribution table for the green-porpoise network comparison runs.

Reads each run's summary.json, country_production.csv and selected_supplier_inputs.csv
(as written by reconciliation/run_historical_network.py) and tabulates the bundle:
supplier contract, cost per delivered tonne, solver gap, active suppliers, production by
country (top N), Australian production and share, and the supply-side statistics of the
selected supplier set (count, eligible capacity, Australian capacity, cost quantiles).

    python compare_network_runs.py --runs label=path [label=path ...] --output DIR

Every number is taken from the run outputs; nothing is recomputed from the model.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def load_run(label: str, path: Path) -> dict:
    s = json.loads((path / "summary.json").read_text())
    m = json.loads((path / "manifest.json").read_text()) if (path / "manifest.json").exists() else {}
    cp = pd.read_csv(path / "country_production.csv")
    sel = pd.read_csv(path / "selected_supplier_inputs.csv")
    total = float(cp.production_t_per_year.sum())
    aus = float(cp.loc[cp.iso3 == "AUS", "production_t_per_year"].sum())
    top = cp.sort_values("production_t_per_year", ascending=False).head(8)
    contract = (m.get("inputs") or {}).get("supplier_contract", {})
    row = {
        "run": label, "job": s.get("slurm_job_id"), "equations": s.get("equations"), "engine": s.get("engine"),
        "supplier_contract": contract.get("path") if isinstance(contract, dict) else contract,
        "supplier_rows_raw": m.get("supplier_rows_raw"), "supplier_rows_selected": m.get("supplier_rows_selected"),
        "onshore_route_mode": m.get("onshore_route_mode"), "pipeline_multiplier": m.get("pipeline_cost_multiplier"),
        "demand_Mt": s.get("demand_t_per_year", np.nan) / 1e6,
        "delivered_cost_USD_per_t": s.get("delivered_cost_USD_per_demand_tonne"),
        "cost_per_produced_t": s.get("cost_USD_per_produced_tonne"),
        "annual_cost_bUSD": s.get("annual_cost_USD", np.nan) / 1e9,
        "relative_gap_pct": 100 * s.get("relative_gap", np.nan), "solve_s": s.get("solve_seconds"),
        "solver_status": s.get("solver_status"), "qa_pass": s.get("qa_pass"),
        "active_suppliers": s.get("active_suppliers_above_one_tonne"),
        "production_Mt": total / 1e6, "australia_Mt": aus / 1e6, "australia_share_pct": 100 * aus / total if total else np.nan,
        "top_producers": "; ".join(f"{r.iso3} {r.production_t_per_year / 1e6:.0f}" for r in top.itertuples()),
        "selected_suppliers": len(sel), "selected_capacity_Mt": float(sel.Max_capacity.sum()),
        "selected_capacity_capped10_Mt": float(np.minimum(sel.Max_capacity, 10).sum()),
        "selected_aus_suppliers": int((sel.iso3 == "AUS").sum()),
        "selected_aus_capacity_capped10_Mt": float(np.minimum(sel.loc[sel.iso3 == "AUS", "Max_capacity"], 10).sum()),
        "selected_lcoa_median": float(sel.LCOA.median()), "selected_lcoa_p10": float(sel.LCOA.quantile(.1)),
        "selected_lcoa_aus_median": float(sel.loc[sel.iso3 == "AUS", "LCOA"].median()) if (sel.iso3 == "AUS").any() else np.nan,
    }
    return row, cp.assign(run=label)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--runs", nargs="+", required=True, help="label=path pairs")
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    rows, countries = [], []
    for item in args.runs:
        label, path = item.split("=", 1)
        row, cp = load_run(label, Path(path))
        rows.append(row); countries.append(cp)
    table = pd.DataFrame(rows).set_index("run")
    table.to_csv(args.output / "attribution_table.csv")
    wide = pd.concat(countries).pivot_table(index=["iso3", "country"], columns="run", values="production_t_per_year", aggfunc="sum").fillna(0) / 1e6
    wide["max"] = wide.max(axis=1)
    wide = wide.sort_values("max", ascending=False).drop(columns="max")
    wide.round(2).to_csv(args.output / "country_production_Mt.csv")
    show = ["job", "delivered_cost_USD_per_t", "relative_gap_pct", "active_suppliers", "australia_Mt", "australia_share_pct",
            "selected_suppliers", "selected_capacity_capped10_Mt", "selected_aus_suppliers", "selected_aus_capacity_capped10_Mt",
            "selected_lcoa_median", "selected_lcoa_aus_median"]
    with pd.option_context("display.width", 250, "display.max_columns", 30):
        print(table[show].round(2).T.to_string())
        print()
        print(wide.head(15).round(1).to_string())
    with open(args.output / "attribution_table.md", "w") as f:
        f.write(table[show].round(2).T.to_markdown() + "\n\n" + wide.head(15).round(1).to_markdown() + "\n")


if __name__ == "__main__":
    main()
