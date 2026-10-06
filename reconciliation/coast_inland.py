#!/usr/bin/env python3
"""Coast against inland Australia: plant cost, network production and spatial site costs.

Bands every Australian 1-degree cell by the distance from its centre to the coastline
(the boundary of the Australia polygon in data/countries.geojson; Australia has no land
border), then writes, from the accepted outputs under results/campaigns/:

  lcoa_by_band.csv        LCOA (EUR2020/t) by band for each model step, on all cells and on
                          the cells eligible (>= 1 Mt/yr) in the archived table
  production_by_band.csv  Australian network production (Mt/yr) by band
  spatial_estimate.csv    first-order effect of the spatial water and build/remoteness inputs
                          (inputs/spatial_cost_inputs.csv) on the improved surface
  cells.csv, coast_inland.md, sources.json

    python reconciliation/coast_inland.py --output results/campaigns/reconciliation_final_20260916_v1/coast_inland_v1
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import Point

ROOT = Path(__file__).resolve().parents[1]
C = ROOT / "results" / "campaigns"
V = C / "verschuur_reconcile_20260907_v1"
EUR2020_PER_USD2018 = 0.9024
EUR2020_PER_USD2020 = 0.875506  # ECB 2020 average, as in the DEA co-located YAML
BASE_WATER_USD2020_PER_M3 = 2.28439325  # flat water of the improved runs (2 EUR2020/m3)
WATER_M3_PER_T = 1.5
BANDS = ([-1, 100, 300, 5000], ["coast <100 km", "100-300 km", "inland >300 km"])

# model step -> (file, latitude column, longitude column, LCOA column, factor to EUR2020)
STEPS = {
    "1 archived table (Salmon 2023)": (V / "historical/git-0a63616/data/c_NH3_cost_4.5.csv", "Latitude", "Longitude", "LCOA", EUR2020_PER_USD2018),
    "2 legacy model replicated": (C / "legacy_lcoa_20260915_v1/audit/global-supplier-table-arc-v1/legacy_replicated_suppliers_full.csv",
                                  "latitude", "longitude", "LCOA", EUR2020_PER_USD2018),
    "3 green-lory, legacy-style plant (4 h, Way, tracking)": (V / "comparison/global-20260914-v2/revalidated/rep/merged.csv",
                                                              "latitude", "longitude", "lcoa_eur_per_t", 1.0),
    "4 hourly plant (1 h, Way, tracking, explicit compressor, DEA tank)": (V / "comparison/global-20260914-v2/revalidated/central/merged.csv",
                                                                          "latitude", "longitude", "lcoa_eur_per_t", 1.0),
    "5 improved (1 h, DEA 2050, fixed PV, water)": (
        C / "keyruns_20260923_v1/30_keyruns/gl_dea2050_wacc5_bflat_wflat_land20c_fixed/runs/20260928-v4/global/merged/global_run_results.csv",
        "latitude", "longitude", "lcoa_eur_per_t", 1.0),
}
NETWORKS = {
    "archived table (A2)": V / "arc_received_20260915/networks/deposited-old-routes-20260914-v1",
    "legacy replication (E)": V / "arc_received_20260916/networks/deposited-legacyrep-20260916-v1",
    "reconciling, legacy capacity rule (D)": V / "arc_received_20260916/networks/deposited-rep-legacyrule-20260915-v1",
    "legacy key run, 2 %": V / "arc_received_20260928_keyruns/gpo_ll_way2050_ameli_bflat_wflat_landleg2c_track_4h_glannuity_iso1000_1mt_v1",
    "improved, 20 %": V / "arc_received_20260928_keyruns/gpo_gl_dea2050_wacc5_bflat_wflat_land20c_fixed_iso1000_1mt_v1",
    "improved, 2 %": V / "arc_received_20260928_keyruns/gpo_gl_dea2050_wacc5_bflat_wflat_land2c_fixed_iso1000_1mt_v1",
}
LAND = C / "land_center_20260923_v1/max_capacities_center_20pct_slope15.csv"
SPATIAL = ROOT / "inputs/spatial_cost_inputs.csv"
COUNTRIES = ROOT / "data/countries.geojson"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def australian_cells() -> pd.DataFrame:
    world = gpd.read_file(COUNTRIES)
    name = next(c for c in world.columns if c.lower() in ("admin", "name", "name_en", "country", "sovereignt"))
    australia = world[world[name].astype(str).str.contains("Australia")].to_crs(3577).geometry.union_all()
    land = pd.read_csv(LAND, usecols=["latitude", "longitude", "onshore_land_pct"])
    box = land[land.latitude.between(-44, -10) & land.longitude.between(112, 154) & (land.onshore_land_pct > 0)].copy()
    points = gpd.GeoSeries([Point(xy) for xy in zip(box.longitude, box.latitude)], crs=4326).to_crs(3577)
    inside = points.within(australia.buffer(60_000)).to_numpy()  # cell centres up to 60 km offshore keep coastal cells
    box, points = box[inside].copy(), points[inside]
    box["coast_km"] = [australia.boundary.distance(p) / 1000 for p in points]
    box["band"] = pd.cut(box.coast_km, BANDS[0], labels=BANDS[1])
    return box


def lcoa_by_band(cells: pd.DataFrame) -> pd.DataFrame:
    frame = cells
    for step, (path, lat, lon, col, factor) in STEPS.items():
        values = pd.read_csv(path, usecols=[lat, lon, col]).rename(columns={lat: "latitude", lon: "longitude", col: step})
        values[step] *= factor
        frame = frame.merge(values, on=["latitude", "longitude"], how="inner")
    archived = pd.read_csv(STEPS["1 archived table (Salmon 2023)"][0]).rename(columns={"Latitude": "latitude", "Longitude": "longitude"})
    eligible = frame.merge(archived.loc[archived.Max_capacity >= 1, ["latitude", "longitude"]], on=["latitude", "longitude"])
    rows = []
    for cell_set, subset in (("all cells", frame), ("eligible in the archived table", eligible)):
        for step in STEPS:
            grouped = subset.groupby("band", observed=True)[step]
            median, p10 = grouped.median(), grouped.quantile(0.1)
            rows.append({"cell_set": cell_set, "step": step, "cells": len(subset),
                         **{f"median {band}": median[band] for band in BANDS[1]},
                         **{f"p10 {band}": p10[band] for band in BANDS[1]},
                         "median inland minus coast": median[BANDS[1][2]] - median[BANDS[1][0]],
                         "p10 inland minus coast": p10[BANDS[1][2]] - p10[BANDS[1][0]]})
    return pd.DataFrame(rows)


def production_by_band(cells: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for label, directory in NETWORKS.items():
        suppliers = pd.read_csv(directory / "suppliers.csv").rename(columns={"Latitude": "latitude", "Longitude": "longitude"})
        active = suppliers[(suppliers.iso3 == "AUS") & (suppliers.production_t_per_year > 0)]
        merged = active.merge(cells[["latitude", "longitude", "band"]], on=["latitude", "longitude"], how="left")
        if merged.band.isna().any():
            raise ValueError(f"{label}: Australian suppliers outside the banded cells")
        production = merged.groupby("band", observed=False).production_t_per_year.sum() / 1e6
        rows.append({"network": label, "australia_mt_per_year": production.sum(), **production.to_dict()})
    return pd.DataFrame(rows)


def spatial_estimate(cells: pd.DataFrame) -> pd.DataFrame:
    path, lat, lon, _, _ = STEPS["5 improved (1 h, DEA 2050, fixed PV, water)"]
    surface = pd.read_csv(path, usecols=[lat, lon, "lcoa_eur_per_t", "build_cost_pct",
                                         "scaled_design_max_gridless_onshore_ammonia_capacity_t"])
    spatial = (pd.read_csv(SPATIAL).groupby(["lat", "lon"])
               .agg(water_usd_per_m3=("water_cost_usd_per_m3", "first"), build_multiplier=("build_cost_multiplier", "median"))
               .reset_index().rename(columns={"lat": "latitude", "lon": "longitude"}))
    frame = cells.merge(surface, on=["latitude", "longitude"]).merge(spatial, on=["latitude", "longitude"], how="left")
    frame = frame[frame.scaled_design_max_gridless_onshore_ammonia_capacity_t >= 1e6]
    # water enters after the solve, so its effect is exact; the build effect holds the design fixed (an upper bound)
    frame["water_effect_eur_per_t"] = (frame.water_usd_per_m3 - BASE_WATER_USD2020_PER_M3) * WATER_M3_PER_T * EUR2020_PER_USD2020
    frame["build_effect_eur_per_t"] = frame.lcoa_eur_per_t * frame.build_cost_pct / 100 * (frame.build_multiplier - 1)
    frame["lcoa_with_spatial"] = frame.lcoa_eur_per_t + frame.water_effect_eur_per_t + frame.build_effect_eur_per_t
    grouped = frame.groupby("band", observed=True)
    return pd.DataFrame({"eligible_cells": grouped.size(),
                         "water_usd_per_m3_median": grouped.water_usd_per_m3.median(),
                         "build_multiplier_median": grouped.build_multiplier.median(),
                         "water_effect_eur_per_t_median": grouped.water_effect_eur_per_t.median(),
                         "build_effect_eur_per_t_median": grouped.build_effect_eur_per_t.median(),
                         "lcoa_p10_base": grouped.lcoa_eur_per_t.quantile(0.1),
                         "lcoa_p10_with_spatial": grouped.lcoa_with_spatial.quantile(0.1)}).reset_index()


def markdown(frame: pd.DataFrame) -> str:
    def cell(value) -> str:
        return f"{value:,.1f}" if isinstance(value, float) else str(value)
    lines = ["| " + " | ".join(map(str, frame.columns)) + " |", "|" + "---|" * len(frame.columns)]
    lines += ["| " + " | ".join(cell(v) for v in row) + " |" for row in frame.itertuples(index=False)]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", type=Path, required=True, help="new directory (must not exist)")
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite {args.output}")
    cells = australian_cells()
    tables = {"lcoa_by_band": lcoa_by_band(cells), "production_by_band": production_by_band(cells),
              "spatial_estimate": spatial_estimate(cells)}
    staging = args.output.with_name(args.output.name + ".partial")
    staging.mkdir(parents=True)  # renamed only when every file is written
    cells.to_csv(staging / "cells.csv", index=False)
    lines = ["# Coast against inland Australia", ""]
    for name, table in tables.items():
        table.to_csv(staging / f"{name}.csv", index=False)
        lines += [f"## {name}", "", markdown(table), ""]
    (staging / "coast_inland.md").write_text("\n".join(lines))
    inputs = [path for path, *_ in STEPS.values()] + [d / "suppliers.csv" for d in NETWORKS.values()] + [LAND, SPATIAL, COUNTRIES]
    (staging / "sources.json").write_text(json.dumps({
        "script_sha256": sha256_file(Path(__file__)),
        "inputs": {str(p.relative_to(ROOT)): sha256_file(p) for p in inputs}}, indent=2) + "\n")
    staging.rename(args.output)
    print((args.output / "coast_inland.md").read_text())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
