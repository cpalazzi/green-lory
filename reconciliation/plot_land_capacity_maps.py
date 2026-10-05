"""Global maps of land-limited ammonia capacity and of available land, drawn as exact 1-degree rasters.

Each cell is a pcolormesh quad on its true footprint (no markers, so no rendering stripes).

Also prints per-table longitude coverage so that missing columns, if any, would be visible in numbers.
Legacy tables label centred cells (integer label = cell centre); the September green-lory table
labels cells whose MODIS content lies in [lat-1, lat] x [lon, lon+1] (see the 23 Sep audit), so its
raster is drawn on that footprint.
"""
from __future__ import annotations
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import LogNorm
import numpy as np
import pandas as pd

import argparse
CAP = "paper_scaled_max_gridless_onshore_ammonia_capacity_t"

def boundaries(path):
    geo = json.loads(path.read_text()); segs = []
    for f in geo["features"]:
        g = f["geometry"]; polys = g["coordinates"] if g["type"] == "MultiPolygon" else [g["coordinates"]]
        for poly in polys:
            for ring in poly:
                xy = np.asarray(ring)
                for sec in np.split(xy, np.where(np.abs(np.diff(xy[:, 0])) > 180)[0] + 1):
                    if len(sec) > 1: segs.append(sec[:, :2])
    return segs

def raster(lon, lat, values, lon_edge_offset, lat_edge_offset):
    """2-D array on an exact 1-degree grid; edges = label + offset ... label + offset + 1."""
    lon = np.round(np.asarray(lon)).astype(int); lat = np.round(np.asarray(lat)).astype(int)
    grid = np.full((180, 360), np.nan)
    ix = lon + 180; iy = lat + 90
    ok = (ix >= 0) & (ix < 360) & (iy >= 0) & (iy < 180)
    grid[iy[ok], ix[ok]] = np.asarray(values, dtype=float)[ok]
    x_edges = np.arange(-180, 181) + lon_edge_offset; y_edges = np.arange(-90, 91) + lat_edge_offset
    return x_edges, y_edges, grid

def panel(ax, segs, lon, lat, values, title, norm, cmap, offsets, cutoff=True):
    ax.set_facecolor("#eef2f5")
    v = np.asarray(values, dtype=float); ok = np.isfinite(v) & (v > 0)
    xe, ye, g = raster(lon, lat, np.where(ok, v, np.nan), *offsets)
    _, _, zero = raster(lon, lat, np.where(ok, np.nan, 1.0), *offsets)
    ax.pcolormesh(xe, ye, np.ma.masked_invalid(zero), cmap=matplotlib.colors.ListedColormap(["#c9d1d8"]), zorder=1.5, rasterized=True)
    m = ax.pcolormesh(xe, ye, np.ma.masked_invalid(g), cmap=cmap, norm=norm, zorder=2, rasterized=True)
    ax.add_collection(LineCollection(segs, colors="#55606a", linewidths=.3, zorder=3))
    ax.set_xlim(-180, 180); ax.set_ylim(-60, 80); ax.tick_params(labelsize=7, length=2)
    tot = float(np.nansum(v[ok]))
    sub = (f"{ok.sum():,} cells with capacity; {int((v >= 1).sum()):,} at or above 1 Mt/yr; total {tot:,.0f} Mt/yr" if cutoff
           else f"{ok.sum():,} cells; total {tot:,.0f} km²")
    ax.set_title(f"{title}\n{sub}", loc="left", fontsize=10)
    return m

def coverage(name, lon):
    lon = np.round(np.asarray(lon)).astype(int)
    counts = pd.Series(lon).value_counts().reindex(range(-180, 180), fill_value=0)
    empty = counts[counts == 0].index.tolist()
    print(f"{name}: {len(lon):,} cells, {360 - len(empty)} of 360 longitude columns populated; empty columns: {empty}")

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--campaigns", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    a = p.parse_args()
    C = a.campaigns; V = C / "verschuur_reconcile_20260907_v1"; L = C / "legacy_lcoa_20260915_v1"; OUT = a.output_dir
    OUT.mkdir(parents=True, exist_ok=True)
    segs = boundaries(V / "arc_received_20260914/countries.geojson")
    arch = pd.read_csv(V / "historical/git-0a63616/data/c_NH3_cost_4.5.csv").drop_duplicates(["Latitude", "Longitude"], keep="last")
    stated = pd.read_csv(L / "audit/global-supplier-table-stated-v1/legacy_replicated_suppliers_full.csv").drop_duplicates(["latitude", "longitude"])
    recov = pd.read_csv(L / "audit/global-supplier-table-arc-v1/legacy_replicated_suppliers_full.csv").drop_duplicates(["latitude", "longitude"])
    central = pd.read_csv(V / "comparison/global-20260914-v2/revalidated/central/merged.csv", low_memory=False)
    central = central[central[CAP].notna()]
    for name, df, lonc in [("archived", arch, "Longitude"), ("stated", stated, "longitude"), ("recovered", recov, "longitude"), ("green-lory central", central, "longitude")]:
        coverage(name, df[lonc])
    CENTRED = (-0.5, -0.5)   # legacy: label = cell centre
    SEPT = (0.0, -1.0)       # September green-lory: MODIS content of label (lat, lon) is [lat-1, lat] x [lon, lon+1]
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9})
    norm = LogNorm(vmin=0.01, vmax=10.0); cmap = "viridis"
    fig, axes = plt.subplots(4, 1, figsize=(12, 17), layout="constrained")
    panel(axes[0], segs, arch.Longitude, arch.Latitude, arch.Max_capacity, "1. Legacy table as it was: archived Verschuur supplier input (Salmon 2023), Max_capacity", norm, cmap, CENTRED)
    panel(axes[1], segs, stated.longitude, stated.latitude, stated.Max_capacity, "2. Legacy plant with the paper's land method as stated (200 km²/GW wind, latitude-packed PV, exclusions, overlap, 2 %)", norm, cmap, CENTRED)
    panel(axes[2], segs, central.longitude, central.latitude, central[CAP] / 1e6, "3. green-lory September central surface (scaled 1 Mt design, exclusive sharing, tracking at 2x fixed, exclusions, 2 %); drawn on its MODIS footprint [lat-1, lat]", norm, cmap, SEPT)
    m = panel(axes[3], segs, recov.longitude, recov.latitude, recov.Max_capacity, "4. Legacy plant with the recovered constants (140 MW/km² PV, 7.3 MW/km² wind, overlap, no exclusions, 2 %): reproduces panel 1", norm, cmap, CENTRED)
    cb = fig.colorbar(m, ax=axes, orientation="horizontal", fraction=.025, pad=.02, aspect=60, extend="both")
    cb.set_label("Land-limited ammonia capacity per 1-degree cell, Mt NH₃ per year (log scale; grey = zero or no land)")
    fig.suptitle("Land-limited production capacity", fontsize=13)
    out1 = OUT / "land_capacity_maps.png"; fig.savefig(out1, dpi=150); plt.close(fig)

    leg = pd.read_csv(L / "audit/legacy-land-areas-v1/legacy_land_areas.csv")
    sept = pd.read_csv(V / "arc_received_20260914/paper_2pct_slope15.csv", low_memory=False)
    sept = sept[sept.onshore_area_km2 > 0]
    coverage("legacy land step", leg.longitude); coverage("green-lory September land build (onshore cells)", sept.longitude)
    norm2 = LogNorm(vmin=1.0, vmax=300.0)
    fig, axes = plt.subplots(2, 1, figsize=(12, 9), layout="constrained")
    panel(axes[0], segs, leg.longitude, leg.latitude, leg.center_solar_shipping_km2, "A. Legacy land step: 2 % of Table-2 suitable area, MODIS 2022, centred cells, no exclusions (archived supplier cells)", norm2, "cividis", CENTRED, cutoff=False)
    m = panel(axes[1], segs, sept.longitude, sept.latitude, sept.solar_area_km2, "B. green-lory September land build: 2 % of suitable area after WDPA and >15° slope exclusions; drawn on its MODIS footprint [lat-1, lat] x [lon, lon+1]", norm2, "cividis", SEPT, cutoff=False)
    fig.colorbar(m, ax=axes, orientation="horizontal", fraction=.03, pad=.02, aspect=60, extend="both", label="Available PV land per cell at a 2 % share, km² (log scale)")
    fig.suptitle("Available land behind the capacity maps", fontsize=13)
    out2 = OUT / "land_availability_maps.png"; fig.savefig(out2, dpi=150); plt.close(fig)
    print(out1.resolve()); print(out2.resolve())

if __name__ == "__main__":
    main()
