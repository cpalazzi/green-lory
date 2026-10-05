#!/usr/bin/env python3
"""Recheck downloaded ARC outputs and quantify cost/capacity differences.

Raw downloads are never rewritten. Analysis outputs go to a new directory.
Costs are compared on the documented USD2018 basis, not current exchange rates.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from arc.merge_and_qa_campaign import _run as recheck

CAPACITY = "scaled_design_max_gridless_onshore_ammonia_capacity_t"
FX = .9024


def sha(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def save_json(path, value):
    with path.open("x") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def quantiles(values):
    values = pd.Series(values).dropna()
    if not len(values):
        return {"n": 0}
    if not np.isfinite(values).all():
        raise ValueError("Non-finite values in comparison")
    return {"n": len(values), **{str(q): float(values.quantile(q)) for q in [0, .1, .5, .9, 1]}}


def compact(frame, name):
    result = frame[["latitude", "longitude", "country", "lcoa_eur_per_t", CAPACITY,
                    "max_gridless_onshore_ammonia_capacity_t", "lcoa_plant_eur_per_t",
                    "water_cost_eur_per_t"]].copy()
    result["lcoa_eur_per_t"] /= FX
    result["lcoa_plant_eur_per_t"] /= FX
    result[CAPACITY] /= 1e6
    result["max_gridless_onshore_ammonia_capacity_t"] /= 1e6
    return result.rename(columns={
        "country": f"country_{name}", "lcoa_eur_per_t": f"cost_{name}_USD2018_per_t",
        "lcoa_plant_eur_per_t": f"plant_{name}_USD2018_per_t", CAPACITY: f"capacity_{name}_Mtpa",
        "max_gridless_onshore_ammonia_capacity_t": f"legacy_alias_{name}_Mtpa",
        "water_cost_eur_per_t": f"water_{name}_EUR2020_per_t",
    })


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--campaign", type=Path, required=True)
    p.add_argument("--received", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    reports = sorted((args.received / "lory").rglob("qa/validation.json"))
    if len(reports) != 4:
        raise ValueError(f"Expected four completed surface/diagnostic runs, found {len(reports)}")
    # Reuse the exact validation implementation that was frozen for these runs.
    inventory = json.loads((args.campaign / "source/release-20260907-v1.json").read_text())
    qa_hash = next(e["sha256"] for e in inventory["entries"] if e["path"] == "arc/merge_and_qa_campaign.py")
    if sha(ROOT / "arc/merge_and_qa_campaign.py") != qa_hash:
        raise ValueError("Local QA implementation differs from pinned ARC release")
    land = args.received / "paper_2pct_slope15.csv"
    if sha(land) != "4d0753f77f3574c1616159997481c090c68e383ac02e6b25284561fb0ab1b38f":
        raise ValueError("Land-input checksum mismatch")
    validations, frames = {}, {}
    for qa_path in reports:
        run = qa_path.parent.parent
        qa = json.loads(qa_path.read_text())
        manifest = run / "manifest.json"
        surface = run / "merged/global_run_results.csv"
        if qa["status"] != "passed" or sha(manifest) != qa["manifest_sha256"] or sha(surface) != qa["output_sha256"]:
            raise ValueError(f"Downloaded result integrity failed: {run}")
        shards = [run / "shards" / Path(e["path"]).name for e in qa["inputs"]]
        for entry, shard in zip(qa["inputs"], shards):
            if sha(shard) != entry["sha256"]:
                raise ValueError(f"Shard hash mismatch: {shard}")
        label = ("rep" if qa["scenario_id"].startswith("rep_") else
                 "central" if qa["scenario_id"].startswith("central_") else
                 "hourly" if qa["scenario_id"].endswith("nominal_h2") else "compressor")
        output = args.output / "revalidated" / label / "merged.csv"
        checked = recheck(SimpleNamespace(
            input=[str(s) for s in shards], expected_input_count=len(shards),
            manifest=str(manifest), scenario_id=qa["scenario_id"], stage=qa["stage"],
            expected_locations=str(land if qa["stage"] == "global" else ROOT / "inputs/lory_reconciliation_diagnostic_cells.csv"),
            expected_locations_kind="land" if qa["stage"] == "global" else "explicit",
            expected_currency="EUR", require_interest_overrides=True,
            require_full_year=True, output=str(output)))
        # Serialisation can differ between pandas versions; compare parsed values.
        downloaded = pd.read_csv(surface)
        regenerated = pd.read_csv(output)
        pd.testing.assert_frame_equal(downloaded, regenerated, check_exact=False, rtol=1e-12, atol=1e-10)
        checked["downloaded_surface_sha256"] = sha(surface)
        checked["downloaded_qa_sha256"] = sha(qa_path)
        checked["parsed_output_matches_download"] = True
        save_json(output.parent / "validation.json", checked)
        validations[label] = {k: checked[k] for k in ["status", "row_count", "coordinate_sha256", "positive_corrected_supplier_capacity_rows", "downloaded_surface_sha256"]}
        frames[label] = downloaded
        print(f"Revalidated {label}: {len(downloaded)} locations", flush=True)

    historical = args.campaign / "historical/git-0a63616/data/c_NH3_cost_4.5.csv"
    raw = pd.read_csv(historical)
    for _, duplicate in raw[raw.Index.duplicated(False)].groupby("Index"):
        if (duplicate[["LCOA", "Max_capacity", "Latitude", "Longitude"]].nunique() != 1).any():
            raise ValueError("Historical duplicate IDs have conflicting values")
    old = raw.drop_duplicates("Index", keep="last").rename(columns={
        "Latitude": "latitude", "Longitude": "longitude", "LCOA": "cost_old_USD2018_per_t",
        "Max_capacity": "capacity_old_Mtpa", "country": "country_old"})
    absent = old.merge(frames["rep"][["latitude", "longitude"]], on=["latitude", "longitude"],
                       how="left", indicator=True, validate="one_to_one")
    absent = absent[(absent._merge == "left_only") & (absent.capacity_old_Mtpa >= 1)].drop(columns="_merge")
    land_table = pd.read_csv(land)
    absent = absent.merge(land_table[["latitude", "longitude", "max_capacity_mw", "protected_area_pct"]],
                          on=["latitude", "longitude"], how="left", validate="one_to_one")
    if not absent.max_capacity_mw.eq(0).all():
        raise ValueError("Historical eligible cells absent from new surfaces are not all explained by zero available land")
    absent.to_csv(args.output / "historical_eligible_cells_excluded_by_land.csv", index=False)
    cells = compact(frames["rep"], "rep").merge(compact(frames["central"], "central"),
           on=["latitude", "longitude"], validate="one_to_one")
    cells = cells.merge(old[["Index", "latitude", "longitude", "country_old", "iso3", "cost_old_USD2018_per_t", "capacity_old_Mtpa"]],
                        on=["latitude", "longitude"], how="left", validate="one_to_one")
    cells["central_minus_rep_USD2018_per_t"] = cells.cost_central_USD2018_per_t - cells.cost_rep_USD2018_per_t
    cells["central_minus_rep_pct"] = 100 * (cells.cost_central_USD2018_per_t / cells.cost_rep_USD2018_per_t - 1)
    cells["rep_minus_old_pct"] = 100 * (cells.cost_rep_USD2018_per_t / cells.cost_old_USD2018_per_t - 1)
    cells["rep_capacity_over_old"] = cells.capacity_rep_Mtpa / cells.capacity_old_Mtpa.replace(0, np.nan)
    cells["central_capacity_over_rep"] = cells.capacity_central_Mtpa / cells.capacity_rep_Mtpa.replace(0, np.nan)
    for label in ["old", "rep", "central"]:
        cells[f"eligible_{label}"] = cells[f"capacity_{label}_Mtpa"] >= 1
    cells.to_csv(args.output / "cell_comparison.csv", index=False)
    eligible = {}
    country_tables = []
    selected_ids = {}
    for label in ["old", "rep", "central"]:
        part = old[old.capacity_old_Mtpa >= 1].copy() if label == "old" else cells[cells[f"eligible_{label}"]].copy()
        selected = part.sort_values(f"cost_{label}_USD2018_per_t").head(4000)
        selected_ids[label] = set(zip(selected.latitude, selected.longitude))
        eligible[label] = {"eligible_cells": len(part), "selected_cells": len(selected),
                           "selected_capacity_Mtpa": float(selected[f"capacity_{label}_Mtpa"].sum()),
                           "selected_cost_USD2018_per_t": quantiles(selected[f"cost_{label}_USD2018_per_t"])}
        grouped = part.groupby(f"country_{label}").agg(
            eligible_cells=(f"capacity_{label}_Mtpa", "size"), capacity_Mtpa=(f"capacity_{label}_Mtpa", "sum"))
        grouped.index.name = "country"
        grouped["surface"] = label
        country_tables.append(grouped.reset_index())
    pd.concat(country_tables).to_csv(args.output / "country_eligibility.csv", index=False)
    common = cells[cells.capacity_old_Mtpa.gt(0) & cells.capacity_rep_Mtpa.gt(0)]
    attribution = cells.merge(compact(frames["hourly"], "hourly"), on=["latitude", "longitude"], validate="one_to_one")
    attribution = attribution.merge(compact(frames["compressor"], "compressor"), on=["latitude", "longitude"], validate="one_to_one")
    attribution["hourly_effect_USD2018_per_t"] = attribution.cost_hourly_USD2018_per_t - attribution.cost_rep_USD2018_per_t
    attribution["compressor_effect_USD2018_per_t"] = attribution.cost_compressor_USD2018_per_t - attribution.cost_hourly_USD2018_per_t
    attribution["tank_effect_USD2018_per_t"] = attribution.plant_central_USD2018_per_t - attribution.cost_compressor_USD2018_per_t
    attribution["water_effect_USD2018_per_t"] = attribution.cost_central_USD2018_per_t - attribution.plant_central_USD2018_per_t
    effects = [c for c in attribution if c.endswith("effect_USD2018_per_t")]
    np.testing.assert_allclose(attribution[effects].sum(axis=1), attribution.central_minus_rep_USD2018_per_t, atol=1e-10)
    attribution.to_csv(args.output / "three_cell_attribution.csv", index=False)
    active = pd.read_csv(args.received / "archival-modamb-v2/suppliers.csv")
    active = active.rename(columns={"Latitude": "latitude", "Longitude": "longitude"})
    active = active[active.production_t_per_year > 1].merge(cells[["latitude", "longitude", "capacity_rep_Mtpa", "capacity_central_Mtpa"]],
             on=["latitude", "longitude"], how="left", validate="one_to_one")
    active.to_csv(args.output / "historical_active_supplier_capacity_check.csv", index=False)
    summary = {
        "checked_utc": datetime.now(timezone.utc).isoformat(), "analysis_sha256": sha(__file__),
        "historical_surface_sha256": sha(historical), "validations": validations,
        "price_basis": "USD2018-equivalent = Lory EUR2020 / 0.9024; historical USD table as archived",
        "eligible_and_cheapest_4000_unique_cells": eligible,
        "selection_overlap": {f"old_vs_{label}": len(selected_ids["old"] & selected_ids[label]) for label in ["rep", "central"]},
        "common_old_rep_positive_capacity_cells": len(common),
        "historical_eligible_absent_from_new_surface_due_to_zero_land": len(absent),
        "rep_vs_old_cost_pct": quantiles(common.rep_minus_old_pct),
        "rep_over_old_capacity_ratio": quantiles(common.rep_capacity_over_old),
        "central_vs_rep_cost_pct_rep_eligible": quantiles(cells.loc[cells.eligible_rep, "central_minus_rep_pct"]),
        "old_eligible_fall_below_cutoff_rep": int((cells.eligible_old & ~cells.eligible_rep).sum()) + len(absent),
        "old_eligible_fall_below_cutoff_central": int((cells.eligible_old & ~cells.eligible_central).sum()) + len(absent),
        "active_historical_suppliers": {label: {
            "below_1Mt_cutoff": int((active[f"capacity_{label}_Mtpa"] < 1).sum()),
            "historical_production_at_excluded_cells_Mtpa": float(active.loc[active[f"capacity_{label}_Mtpa"] < 1, "production_t_per_year"].sum()/1e6),
            "missing_cells": int(active[f"capacity_{label}_Mtpa"].isna().sum()),
        } for label in ["rep", "central"]},
        "notes": ["Selection summaries use unique coordinate cells, not the archived duplicate-row selection semantics.",
                  "Sequential three-cell attribution is order-dependent and is not a global causal attribution.",
                  "Costs for zero-capacity sites are excluded from supplier comparisons."]}
    save_json(args.output / "summary.json", summary)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
