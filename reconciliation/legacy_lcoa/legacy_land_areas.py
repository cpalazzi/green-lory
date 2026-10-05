#!/usr/bin/env python
"""Legacy land step: technology-specific suitable area per 1-degree cell.

This is the land/capacity-availability step that was missing from the
surviving legacy-lcoa code. It follows Salmon & Banares-Alcantara (2022)
section 2.2.1 and Verschuur et al. (2024) section 4.7 as far as the recovered
capacity column allows (see legacy_capacity.py and
LEGACY_REPLICATION_20260915.md section 6): MODIS land-cover class fractions of
the centered 1-degree cell, the Table-2 suitability factors per class for wind
and solar, spherical pixel areas, and the 2 % shipping allocation. No
protected-area or slope exclusion is applied: the archived capacities are
reproduced without them and Salmon's exclusion inputs are not recoverable.

Source substitution is explicit: the only MODIS product available locally is
MCD12C1 2022 Collection 6.1 (0.05 degree class percentages); Salmon's MODIS
year is unknown.

    python legacy_land_areas.py --modis <MCD12C1 .hdf> --cells cells_archived_all_v1.csv \
        --output <new dir>

Output columns keep the names used by the September land audit
(``center_solar_shipping_km2`` etc.) so that build_legacy_supplier_table.py
and legacy_capacity.py consume the table unchanged.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from reconciliation.land.core import Cell, WIND, SOLAR, read_cmg_cell  # noqa: E402

LAND_SHARE = 0.02
CLASS_NAMES = ["water", "evergreen_needleleaf", "evergreen_broadleaf", "deciduous_needleleaf", "deciduous_broadleaf",
               "mixed_forest", "closed_shrubland", "open_shrubland", "woody_savanna", "savanna", "grassland",
               "permanent_wetland", "cropland", "urban", "cropland_natural_mosaic", "snow_ice", "barren"]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--modis", type=Path, required=True, help="MCD12C1 HDF file (Land_Cover_Type_1_Percent)")
    p.add_argument("--cells", type=Path, required=True, help="CSV with lat, lon (or Latitude, Longitude) columns")
    p.add_argument("--anchor", choices=["center", "southwest"], default="center")
    p.add_argument("--land-share", type=float, default=LAND_SHARE)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    if a.output.exists():
        raise SystemExit(f"refusing to overwrite {a.output}")
    from pyhdf.SD import SD, SDC
    cells = pd.read_csv(a.cells).rename(columns={"Latitude": "lat", "Longitude": "lon", "latitude": "lat", "longitude": "lon"})
    cells = cells[["lat", "lon"]].drop_duplicates().reset_index(drop=True)
    sd = SD(str(a.modis), SDC.READ)
    sds = sd.select("Land_Cover_Type_1_Percent")
    rows = []
    t0 = time.time()
    try:
        for i, c in cells.iterrows():
            lat, lon = float(c.lat), float(c.lon)
            f, area, residual = read_cmg_cell(sds, Cell(lat, lon, a.anchor))
            f = f / f.sum(axis=-1, keepdims=True)       # normalise independently rounded percentages
            solar = float(((f @ SOLAR) * area).sum())
            wind = float(((f @ WIND) * area).sum())
            union = float(((f @ np.maximum(SOLAR, WIND)) * area).sum())
            classes = (f * area[..., None]).sum(axis=(0, 1)) / area.sum()
            rows.append({"latitude": lat, "longitude": lon,
                         f"{a.anchor}_cell_area_km2": float(area.sum()),
                         f"{a.anchor}_solar_suitable_km2": solar, f"{a.anchor}_wind_suitable_km2": wind,
                         f"{a.anchor}_solar_shipping_km2": solar * a.land_share,
                         f"{a.anchor}_wind_shipping_km2": wind * a.land_share,
                         f"{a.anchor}_union_shipping_km2": union * a.land_share,
                         "max_abs_pixel_closure_residual": float(np.abs(residual).max()),
                         **{f"class_fraction_{n}": float(v) for n, v in zip(CLASS_NAMES, classes)}})
            if (i + 1) % 1000 == 0:
                print(f"{i + 1}/{len(cells)} cells, {time.time() - t0:.0f} s", flush=True)
    finally:
        sd.end()
    a.output.mkdir(parents=True)
    df = pd.DataFrame(rows)
    df.to_csv(a.output / "legacy_land_areas.csv", index=False)
    meta = {
        "method": "MODIS class fractions of the 1-degree cell x Table-2 suitability factor per class x spherical pixel area, "
                  "summed, x land share; complete wind/solar overlap (each technology has its own budget); no exclusions",
        "anchor": a.anchor, "land_share": a.land_share,
        "table2_factors": {"wind": WIND.tolist(), "solar": SOLAR.tolist(), "class_order": CLASS_NAMES},
        "modis": {"path": str(a.modis.resolve()), "sha256": sha256_file(a.modis),
                  "product_note": "MCD12C1 Collection 6.1, year 2022 (source substitution; Salmon's vintage unknown)"},
        "cells": {"path": str(a.cells.resolve()), "sha256": sha256_file(a.cells), "n": int(len(cells))},
        "output_sha256": sha256_file(a.output / "legacy_land_areas.csv"),
        "script_sha256": sha256_file(Path(__file__)), "elapsed_s": time.time() - t0,
    }
    (a.output / "summary.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(json.dumps({k: v for k, v in meta.items() if k != "table2_factors"}, indent=2))


if __name__ == "__main__":
    main()
