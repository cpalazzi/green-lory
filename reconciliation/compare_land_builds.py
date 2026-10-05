#!/usr/bin/env python3
"""Compare a green-lory land build with the legacy-lcoa land step, cell by cell.

Legacy step (reconciliation/legacy_lcoa/legacy_land_areas.py): centred 1-degree
cells, MODIS MCD12C1 2022 x Table-2 suitability, 2 % share, no exclusions.
green-lory build (model/land_processing.py): same MODIS and factors, centred
cells, WDPA and >15 degree slope exclusions, any land-competition share (the
100 % table is rescaled here with apply_land_competition_scenario).

At a 2 % share the ratio green-lory / legacy is exactly the exclusion factor of
the cell (same MODIS content, same geometry); at 20 % it is ten times that.
The co-located rule enters through the union area: PV plus 3 % of the wind
footprint must fit inside the classwise nested union, whereas the legacy rule
lets wind and PV overlap completely and limits each technology separately.

    python reconciliation/compare_land_builds.py --green-lory <100pct csv> --tag <name>
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap, LogNorm
import numpy as np
import pandas as pd

import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from model.land_processing import apply_land_competition_scenario  # noqa: E402
from model.land_union import CLASSWISE_NESTED_UNION_AREA_COLUMN  # noqa: E402

C = ROOT / "results/campaigns"
LEGACY = C / "legacy_lcoa_20260915_v1/audit/legacy-land-areas-v1/legacy_land_areas.csv"
STATED = C / "legacy_lcoa_20260915_v1/audit/global-supplier-table-stated-v1/legacy_replicated_suppliers_full.csv"
COUNTRIES = C / "verschuur_reconcile_20260907_v1/comparison/global-20260914-v2/revalidated/central/merged.csv"
GEOJSON = C / "verschuur_reconcile_20260907_v1/arc_received_20260914/countries.geojson"
FOCAL = {"Atacama (-23, -69)": (-23.0, -69.0), "NW Australia (-23, 117)": (-23.0, 117.0), "central Australia (-21, 135)": (-21.0, 135.0)}
D_PV_ARCHIVED, D_WIND_ARCHIVED, D_WIND_STATED = 140.0, 7.3, 5.0


def boundaries(path: Path):
    geo = json.loads(path.read_text()); segs = []
    for f in geo["features"]:
        g = f["geometry"]; polys = g["coordinates"] if g["type"] == "MultiPolygon" else [g["coordinates"]]
        for poly in polys:
            for ring in poly:
                xy = np.asarray(ring)
                for sec in np.split(xy, np.where(np.abs(np.diff(xy[:, 0])) > 180)[0] + 1):
                    if len(sec) > 1: segs.append(sec[:, :2])
    return segs


def raster(lon, lat, values):
    lon = np.round(np.asarray(lon)).astype(int); lat = np.round(np.asarray(lat)).astype(int)
    grid = np.full((180, 360), np.nan); ok = (lon >= -180) & (lon < 180) & (lat >= -90) & (lat < 90)
    grid[lat[ok] + 90, lon[ok] + 180] = np.asarray(values, dtype=float)[ok]
    return np.arange(-180, 181) - 0.5, np.arange(-90, 91) - 0.5, grid   # centred cells


def panel(ax, segs, lon, lat, values, title, norm, cmap, unit):
    ax.set_facecolor("#eef2f5")
    v = np.asarray(values, dtype=float); ok = np.isfinite(v) & (v > 0)
    xe, ye, g = raster(lon, lat, np.where(ok, v, np.nan))
    _, _, zero = raster(lon, lat, np.where(ok, np.nan, 1.0))
    ax.pcolormesh(xe, ye, np.ma.masked_invalid(zero), cmap=ListedColormap(["#c9d1d8"]), zorder=1.5, rasterized=True)
    m = ax.pcolormesh(xe, ye, np.ma.masked_invalid(g), cmap=cmap, norm=norm, zorder=2, rasterized=True)
    ax.add_collection(LineCollection(segs, colors="#55606a", linewidths=.3, zorder=3))
    ax.set_xlim(-180, 180); ax.set_ylim(-60, 80); ax.tick_params(labelsize=7, length=2)
    ax.set_title(f"{title}\n{ok.sum():,} cells; total {np.nansum(v[ok]):,.0f} {unit}", loc="left", fontsize=10)
    return m


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--green-lory", type=Path, required=True, help="green-lory land table at a 100 % share")
    p.add_argument("--tag", required=True)
    p.add_argument("--shares", type=float, nargs="+", default=[0.02, 0.20])
    p.add_argument("--output-root", type=Path, default=C / "reconciliation_final_20260916_v1/land_builds")
    a = p.parse_args()
    out = a.output_root / a.tag; out.mkdir(parents=True, exist_ok=True)

    leg = pd.read_csv(LEGACY)
    gl100 = pd.read_csv(a.green_lory, low_memory=False)
    anchor = set(gl100["cell_anchor"].dropna().unique()) if "cell_anchor" in gl100.columns else {"unknown"}
    frac = gl100["land_competition_fraction"].dropna().unique()
    assert len(frac) == 1 and abs(frac[0] - 1.0) < 1e-9, f"expected a 100 % table, got {frac}"
    stated = pd.read_csv(STATED).drop_duplicates(["latitude", "longitude"])[["latitude", "longitude", "iso3", "country", "d_pv_mw_per_km2"]]
    ctry = pd.read_csv(COUNTRIES, low_memory=False)
    ccols = [c for c in ("iso3", "country") if c in ctry.columns]
    ctry = ctry.drop_duplicates(["latitude", "longitude"])[["latitude", "longitude", *ccols]]

    tables = {s: apply_land_competition_scenario(gl100, s) for s in a.shares}
    base = leg[["latitude", "longitude", "center_cell_area_km2", "center_solar_suitable_km2", "center_wind_suitable_km2",
                "center_solar_shipping_km2", "center_wind_shipping_km2", "center_union_shipping_km2"]].merge(stated, on=["latitude", "longitude"], how="left")
    # Exclusion shares, factor and densities come from the 100 % table: apply_land_competition_scenario
    # folds the share into land_exclusion_factor, so a 2 % table reports 0.02 x factor.
    excl = gl100[["latitude", "longitude", "onshore_land_pct", "protected_area_pct", "slope_suitable_land_pct", "land_exclusion_factor",
                  "solar_density_mw_per_km2", "wind_density_mw_per_km2"]]
    cells = base.merge(excl, on=["latitude", "longitude"], how="left")
    for s, t in tables.items():
        tag = f"{s*100:g}pct"
        sub = t[["latitude", "longitude", "solar_area_km2", "wind_onshore_area_km2", CLASSWISE_NESTED_UNION_AREA_COLUMN]].rename(
            columns={"solar_area_km2": f"gl_solar_km2_{tag}", "wind_onshore_area_km2": f"gl_wind_km2_{tag}",
                     CLASSWISE_NESTED_UNION_AREA_COLUMN: f"gl_union_km2_{tag}"})
        cells = cells.merge(sub, on=["latitude", "longitude"], how="left")
        cells[f"ratio_solar_{tag}_vs_legacy_2pct"] = cells[f"gl_solar_km2_{tag}"] / cells["center_solar_shipping_km2"].replace(0, np.nan)
    cells["max_pv_mw_archived_rule_2pct"] = cells["center_solar_shipping_km2"] * D_PV_ARCHIVED
    cells["max_wind_mw_archived_rule_2pct"] = cells["center_wind_shipping_km2"] * D_WIND_ARCHIVED
    cells["max_pv_mw_stated_rule_2pct"] = cells["center_solar_shipping_km2"] * cells["d_pv_mw_per_km2"]
    cells["max_wind_mw_stated_rule_2pct"] = cells["center_wind_shipping_km2"] * D_WIND_STATED
    for s in a.shares:
        tag = f"{s*100:g}pct"
        cells[f"gl_max_pv_mw_fixed_{tag}"] = cells[f"gl_solar_km2_{tag}"] * cells["solar_density_mw_per_km2"]
        cells[f"gl_max_wind_mw_{tag}"] = cells[f"gl_wind_km2_{tag}"] * cells["wind_density_mw_per_km2"]
    cells.to_csv(out / "cell_comparison.csv", index=False)

    # summaries
    first = f"{a.shares[0]*100:g}pct"
    onshore = gl100[gl100["onshore_land_pct"] > 0].merge(ctry, on=["latitude", "longitude"], how="left")
    summary = {
        "green_lory_table": str(a.green_lory), "cell_anchor": sorted(anchor), "shares": a.shares,
        "legacy_cells": int(len(leg)), "legacy_cells_missing_in_green_lory": int(cells[f"gl_solar_km2_{first}"].isna().sum()),
        "legacy_solar_km2_2pct_total": float(leg.center_solar_shipping_km2.sum()),
        "legacy_wind_km2_2pct_total": float(leg.center_wind_shipping_km2.sum()),
        "exclusion_factor_archived_cells": {"median": float(cells.land_exclusion_factor.median()), "mean_area_weighted": float((cells.land_exclusion_factor * cells.center_solar_suitable_km2).sum() / cells.center_solar_suitable_km2.sum())},
        "exclusion_factor_all_onshore_cells": {"cells": int(len(onshore)), "median": float(onshore.land_exclusion_factor.median())},
    }
    for s in a.shares:
        tag = f"{s*100:g}pct"; t = tables[s]
        summary[f"green_lory_{tag}"] = {
            "solar_km2_archived_cells": float(cells[f"gl_solar_km2_{tag}"].sum()), "wind_km2_archived_cells": float(cells[f"gl_wind_km2_{tag}"].sum()),
            "union_km2_archived_cells": float(cells[f"gl_union_km2_{tag}"].sum()),
            "solar_km2_all_onshore_cells": float(t.solar_area_km2.sum()), "wind_km2_all_onshore_cells": float(t.wind_onshore_area_km2.sum()),
            "ratio_solar_vs_legacy_2pct_median": float(cells[f"ratio_solar_{tag}_vs_legacy_2pct"].median()),
            "ratio_solar_vs_legacy_2pct_total": float(cells[f"gl_solar_km2_{tag}"].sum() / leg.center_solar_shipping_km2.sum()),
        }
    # countries (archived cells: legacy iso3; all onshore cells: September assignment on centred boxes)
    rows = []
    for iso, g in cells.groupby("iso3"):
        r = {"iso3": iso, "country": g.country.iloc[0], "archived_cells": len(g), "legacy_solar_km2_2pct": g.center_solar_shipping_km2.sum(),
             "legacy_wind_km2_2pct": g.center_wind_shipping_km2.sum(), "exclusion_factor_area_weighted": (g.land_exclusion_factor * g.center_solar_suitable_km2).sum() / max(g.center_solar_suitable_km2.sum(), 1e-9)}
        for s in a.shares:
            tag = f"{s*100:g}pct"; r[f"gl_solar_km2_{tag}"] = g[f"gl_solar_km2_{tag}"].sum(); r[f"gl_wind_km2_{tag}"] = g[f"gl_wind_km2_{tag}"].sum()
        rows.append(r)
    countries = pd.DataFrame(rows).sort_values("legacy_solar_km2_2pct", ascending=False)
    countries.to_csv(out / "country_comparison.csv", index=False)
    focal = []
    for name, (lat, lon) in FOCAL.items():
        r = cells[(cells.latitude == lat) & (cells.longitude == lon)]
        if r.empty: continue
        r = r.iloc[0]
        row = {"cell": name, "legacy_solar_km2_2pct": r.center_solar_shipping_km2, "legacy_wind_km2_2pct": r.center_wind_shipping_km2,
               "legacy_union_km2_2pct": r.center_union_shipping_km2, "protected_area_pct": r.protected_area_pct, "slope_suitable_land_pct": r.slope_suitable_land_pct,
               "exclusion_factor": r.land_exclusion_factor, "d_pv_stated": r.d_pv_mw_per_km2, "d_pv_green_lory_fixed": r.solar_density_mw_per_km2,
               "max_pv_mw_archived_rule_2pct": r.max_pv_mw_archived_rule_2pct, "max_wind_mw_archived_rule_2pct": r.max_wind_mw_archived_rule_2pct,
               "max_pv_mw_stated_rule_2pct": r.max_pv_mw_stated_rule_2pct, "max_wind_mw_stated_rule_2pct": r.max_wind_mw_stated_rule_2pct}
        for s in a.shares:
            tag = f"{s*100:g}pct"
            for k in (f"gl_solar_km2_{tag}", f"gl_wind_km2_{tag}", f"gl_union_km2_{tag}", f"gl_max_pv_mw_fixed_{tag}", f"gl_max_wind_mw_{tag}"):
                row[k] = r[k]
        focal.append(row)
    focal = pd.DataFrame(focal); focal.to_csv(out / "focal_cells.csv", index=False)
    (out / "summary.json").write_text(json.dumps(summary, indent=2))

    # maps
    segs = boundaries(GEOJSON)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9})
    fig, axes = plt.subplots(4, 1, figsize=(12, 17), layout="constrained")
    n1 = LogNorm(1.0, 300.0)
    panel(axes[0], segs, leg.longitude, leg.latitude, leg.center_solar_shipping_km2, "A. Legacy land step: PV land at a 2 % share (centred cells, MODIS 2022 x Table 2, no exclusions; archived supplier cells)", n1, "cividis", "km²")
    t2 = tables[a.shares[0]]; t2 = t2[t2.onshore_land_pct > 0]
    panel(axes[1], segs, t2.longitude, t2.latitude, t2.solar_area_km2, f"B. green-lory build [{a.tag}]: PV land at a {a.shares[0]*100:g} % share after WDPA and >15° slope exclusions (centred cells, all onshore cells)", n1, "cividis", "km²")
    g0 = gl100[gl100.onshore_land_pct > 0]
    m3 = panel(axes[2], segs, g0.longitude, g0.latitude, g0.land_exclusion_factor, "C. Exclusion factor of the green-lory build (share of suitable land kept after WDPA and slope exclusions); at 2 % this is exactly B / A", matplotlib.colors.Normalize(0, 1), "viridis", "(sum of factors)")
    if len(a.shares) > 1:
        t20 = tables[a.shares[-1]]; t20 = t20[t20.onshore_land_pct > 0]
        m4 = panel(axes[3], segs, t20.longitude, t20.latitude, t20.solar_area_km2, f"D. green-lory build [{a.tag}]: PV land at a {a.shares[-1]*100:g} % share (ten times B)", LogNorm(10.0, 3000.0), "cividis", "km²")
        fig.colorbar(m4, ax=axes[3], orientation="horizontal", fraction=.05, pad=.02, aspect=60, extend="both", label=f"PV land per cell at {a.shares[-1]*100:g} %, km² (log)")
    fig.colorbar(m3, ax=axes[2], orientation="horizontal", fraction=.05, pad=.02, aspect=60, label="exclusion factor")
    fig.colorbar(plt.cm.ScalarMappable(norm=n1, cmap="cividis"), ax=axes[:2], orientation="horizontal", fraction=.03, pad=.02, aspect=60, extend="both", label="PV land per cell at 2 %, km² (log)")
    fig.suptitle(f"Legacy land step vs green-lory centred land build ({a.tag})", fontsize=13)
    fig.savefig(out / "land_build_comparison_maps.png", dpi=150); plt.close(fig)

    print(json.dumps(summary, indent=2))
    print("\nfocal cells:"); print(focal.T.to_string())
    print("\ntop countries by legacy PV land (2 %):"); print(countries.head(15).to_string(index=False, float_format=lambda x: f"{x:,.0f}" if abs(x) >= 10 else f"{x:.3f}"))
    print(out.resolve())


if __name__ == "__main__":
    main()
