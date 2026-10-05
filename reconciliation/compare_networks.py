#!/usr/bin/env python3
"""Quantitative comparison of green-porpoise network runs.

Reads each run's ``summary.json``, ``manifest.json``, ``suppliers.csv`` and
``country_production.csv`` (as written by ``run_historical_network.py`` /
``sparse_network.py``) and tabulates delivered cost, solver gap, production by
country and Australian subregion, active-supplier identities and overlap with a
reference run, and the role of named focal cells. A run that ended at its time
limit is reported with its gap; nothing is relabelled as accepted.

    python compare_networks.py --run <label>=<dir> [--run ...] \
        --reference <label> --focal "Atacama=-23_-69_1000000.0" ... --output <new dir>
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

# Australian subregions by longitude of the 1-degree cell (explicit, coarse):
# west (WA) < 129 E, centre (NT/SA) 129-141 E, east (QLD/NSW/VIC/TAS) > 141 E.
AUS_BANDS = [("west_lt129E", -np.inf, 129.0), ("centre_129_141E", 129.0, 141.0), ("east_gt141E", 141.0, np.inf)]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_run(label: str, d: Path) -> dict:
    s = json.loads((d / "summary.json").read_text())
    m = json.loads((d / "manifest.json").read_text())
    sup = pd.read_csv(d / "suppliers.csv")
    country = pd.read_csv(d / "country_production.csv")
    failure = json.loads((d / "failure.json").read_text()) if (d / "failure.json").exists() else None
    active = sup[sup.production_t_per_year > 1].copy()
    aus = sup[sup.iso3 == "AUS"]
    bands = {}
    for name, lo, hi in AUS_BANDS:
        part = aus[(aus.Longitude >= lo) & (aus.Longitude < hi)]
        bands[name] = {"production_Mtpa": float(part.production_t_per_year.sum() / 1e6),
                       "active_suppliers": int((part.production_t_per_year > 1).sum()),
                       "selected_capacity_Mtpa": float(part.Max_capacity.sum())}
    contract = m.get("supplier_contract") or {}
    demand = s["demand_t_per_year"]
    covered = m.get("model_total_demand_requirement_t_per_year", demand)
    return {
        "label": label, "dir": str(d.resolve()),
        "summary": s, "manifest_inputs": m.get("inputs"), "failure": failure,
        "endpoint": m.get("endpoint"), "onshore_route_mode": m.get("onshore_route_mode"),
        "pipeline_multiplier": m.get("pipeline_cost_multiplier"),
        "supplier_table": (m.get("inputs", {}).get("suppliers") or {}).get("path"),
        "supplier_table_sha256": (m.get("inputs", {}).get("suppliers") or {}).get("sha256"),
        "supplier_contract_scenario": contract.get("scenario_id"), "capacity_method": contract.get("capacity_method"),
        "demand_all_file_Mtpa": demand / 1e6, "demand_covered_Mtpa": covered / 1e6,
        "supplier_rows_selected": m.get("supplier_rows_selected"), "supplier_ids_effective": m.get("supplier_ids_effective"),
        "suppliers": sup, "active": active, "country": country, "aus_bands": bands,
    }


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", action="append", required=True, help="label=directory (repeatable, order kept)")
    p.add_argument("--reference", required=True, help="label of the reference run for supplier-overlap statistics")
    p.add_argument("--focal", action="append", default=[], help="name=Index of a focal supplier cell (repeatable)")
    p.add_argument("--top", type=int, default=8)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    if a.output.exists():
        raise SystemExit(f"refusing to overwrite {a.output}")
    runs = []
    for item in a.run:
        label, d = item.split("=", 1)
        runs.append(load_run(label, Path(d)))
    labels = [r["label"] for r in runs]
    if a.reference not in labels:
        raise SystemExit(f"reference {a.reference} not among {labels}")
    ref = next(r for r in runs if r["label"] == a.reference)
    ref_active = set(ref["active"].Index)
    ref_prod = ref["active"].set_index("Index").production_t_per_year
    focal = dict(item.split("=", 1) for item in a.focal)

    rows, focal_rows, country_rows, band_rows = [], [], [], []
    for r in runs:
        s = r["summary"]
        act = set(r["active"].Index)
        common = act & ref_active
        prod_common = float(r["active"].set_index("Index").production_t_per_year.reindex(sorted(common)).sum() / 1e6)
        aus = r["country"][r["country"].iso3 == "AUS"].production_t_per_year.sum() / 1e6 if "iso3" in r["country"] else np.nan
        top = r["country"].sort_values("production_t_per_year", ascending=False).head(a.top)
        rows.append({
            "run": r["label"], "endpoint": r["endpoint"], "termination": s.get("termination"), "qa_pass": s.get("qa_pass"),
            "failure": None if r["failure"] is None else r["failure"].get("error"),
            "relative_gap": s.get("relative_gap"), "solve_seconds": s.get("solve_seconds"),
            "annual_cost_USD_bn": s["annual_cost_USD"] / 1e9,
            "delivered_cost_USD_per_demand_t": s["delivered_cost_USD_per_demand_tonne"],
            "cost_USD_per_produced_t": s.get("cost_USD_per_produced_tonne"),
            "lower_bound_cost_USD_per_demand_t": s["lower_bound_USD_per_year"] / s["demand_t_per_year"],
            "demand_all_file_Mtpa": r["demand_all_file_Mtpa"], "demand_covered_Mtpa": r["demand_covered_Mtpa"],
            "total_production_Mtpa": s["total_production_t_per_year"] / 1e6,
            "production_cost_USD_bn": s["cost_components_USD_per_year"]["production"] / 1e9,
            "pipeline_cost_USD_bn": s["cost_components_USD_per_year"]["pipeline"] / 1e9,
            "shipping_cost_USD_bn": s["cost_components_USD_per_year"]["shipping"] / 1e9,
            "storage_cost_USD_bn": s["cost_components_USD_per_year"]["storage"] / 1e9,
            "supplier_rows_selected": r["supplier_rows_selected"], "supplier_ids_effective": r["supplier_ids_effective"],
            "active_suppliers": s["active_suppliers_above_one_tonne"],
            "active_common_with_reference": len(common),
            "share_of_reference_active_retained": len(common) / max(len(ref_active), 1),
            "production_at_common_suppliers_Mtpa": prod_common,
            "australia_production_Mtpa": aus,
            "australia_share_of_production": aus / (s["total_production_t_per_year"] / 1e6),
            "onshore_routes": s.get("onshore_routes"), "offshore_routes": s.get("offshore_routes"),
            "onshore_route_mode": r["onshore_route_mode"], "pipeline_multiplier": r["pipeline_multiplier"],
            "supplier_contract_scenario": r["supplier_contract_scenario"], "capacity_method": r["capacity_method"],
            "top_producers": "; ".join(f"{t.iso3} {t.production_t_per_year/1e6:.1f}" for t in top.itertuples()),
            "supplier_table_sha256": r["supplier_table_sha256"],
        })
        for name, idx in focal.items():
            sel = r["suppliers"][r["suppliers"].Index == idx]
            focal_rows.append({"run": r["label"], "cell": name, "Index": idx,
                               "selected": not sel.empty,
                               "LCOA_USD_per_t": None if sel.empty else float(sel.LCOA.iloc[0]),
                               "Max_capacity_Mtpa": None if sel.empty else float(sel.Max_capacity.iloc[0]),
                               "production_Mtpa": None if sel.empty else float(sel.production_t_per_year.iloc[0] / 1e6)})
        for t in r["country"].itertuples():
            country_rows.append({"run": r["label"], "iso3": t.iso3, "country": t.country, "production_Mtpa": t.production_t_per_year / 1e6})
        for band, v in r["aus_bands"].items():
            band_rows.append({"run": r["label"], "band": band, **v})

    a.output.mkdir(parents=True)
    table = pd.DataFrame(rows)
    table.to_csv(a.output / "networks.csv", index=False)
    pd.DataFrame(focal_rows).to_csv(a.output / "focal_cells.csv", index=False)
    cp = pd.DataFrame(country_rows).pivot(index=["iso3", "country"], columns="run", values="production_Mtpa").fillna(0.0)
    cp = cp[labels].sort_values(a.reference, ascending=False)
    cp.to_csv(a.output / "country_production_Mtpa.csv")
    pd.DataFrame(band_rows).to_csv(a.output / "australia_subregions.csv", index=False)

    # markdown digest
    lines = ["| Run | Endpoint | Gap | USD/t (covered demand) | Production Mt/yr | Active suppliers | Common with " + a.reference +
             " | Australia Mt/yr | Top producers |", "|---|---|---:|---:|---:|---:|---:|---:|---|"]
    for t in table.itertuples():
        lines.append(f"| {t.run} | {t.endpoint} | {t.relative_gap:.2%} | {t.delivered_cost_USD_per_demand_t:.2f} | "
                     f"{t.total_production_Mtpa:.1f} | {t.active_suppliers} | {t.active_common_with_reference} | "
                     f"{t.australia_production_Mtpa:.1f} | {t.top_producers} |")
    lines += ["", "| Run | " + " | ".join(focal) + " |", "|---|" + "---|" * len(focal)]
    fr = pd.DataFrame(focal_rows)
    for lab in labels:
        cells = []
        for name in focal:
            f = fr[(fr.run == lab) & (fr.cell == name)].iloc[0]
            cells.append("not selected" if not f.selected else f"cap {f.Max_capacity_Mtpa:.2f}, prod {f.production_Mtpa:.2f} Mt/yr at {f.LCOA_USD_per_t:.2f} USD/t")
        lines.append(f"| {lab} | " + " | ".join(cells) + " |")
    (a.output / "digest.md").write_text("\n".join(lines) + "\n")
    meta = {"runs": [{k: r[k] for k in ("label", "dir", "endpoint", "supplier_table", "supplier_table_sha256", "capacity_method",
                                        "supplier_contract_scenario", "manifest_inputs", "failure")} for r in runs],
            "reference": a.reference, "focal": focal, "australia_bands": [(n, lo, hi) for n, lo, hi in AUS_BANDS],
            "script_sha256": sha256_file(Path(__file__)),
            "note": "Delivered cost divides the objective by the all-file demand as the runner does; covered-demand cost uses model_total_demand_requirement when present."}
    (a.output / "summary.json").write_text(json.dumps(meta, indent=2, default=str) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
