#!/usr/bin/env python3
"""Interest-only finance override CSVs covering every cell of a land table.

Two files, same columns (lat, lon, tech, interest_rate) and the same 15 technology rows
per cell as the September Ameli file:
  uniform_interest_inputs_<rate>_2050.csv        one rate everywhere (flat finance)
  amelired_interest_inputs_2050_center.csv       Ameli reduced WACC by country; cells absent
                                                 from the source file (no country assignment
                                                 in the September build) take the rate of the
                                                 nearest covered cell and are listed in
                                                 amelired_interest_fill_2050_center.csv
Deterministic: identical inputs give identical bytes on any machine.
"""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--land-csv", type=Path, required=True, help="land table whose latitude/longitude labels define the cells")
    p.add_argument("--ameli-csv", type=Path, required=True, help="September Ameli override (lat, lon, tech, interest_rate)")
    p.add_argument("--uniform-rate", type=float, default=0.05)
    p.add_argument("--output-dir", type=Path, required=True)
    a = p.parse_args()

    cells = pd.read_csv(a.land_csv, usecols=["latitude", "longitude"]).drop_duplicates().sort_values(["latitude", "longitude"])
    cells = cells.rename(columns={"latitude": "lat", "longitude": "lon"}).reset_index(drop=True)
    ameli = pd.read_csv(a.ameli_csv)
    techs = sorted(ameli["tech"].unique())
    a.output_dir.mkdir(parents=True, exist_ok=True)

    # uniform
    grid = cells.merge(pd.DataFrame({"tech": techs}), how="cross")
    grid["interest_rate"] = a.uniform_rate
    tag = f"{a.uniform_rate:g}".replace(".", "p")
    uniform_path = a.output_dir / f"uniform_interest_inputs_{tag}_2050.csv"
    grid.sort_values(["lat", "lon", "tech"]).to_csv(uniform_path, index=False, float_format="%.6g")

    # Ameli with nearest-cell fill
    rates = ameli.pivot_table(index=["lat", "lon"], columns="tech", values="interest_rate", aggfunc="first")
    covered = rates.index.to_frame(index=False)
    key = cells.merge(covered.assign(_covered=1), on=["lat", "lon"], how="left")
    missing = key[key["_covered"].isna()][["lat", "lon"]].reset_index(drop=True)
    fill_rows = []
    if not missing.empty:
        rad = np.deg2rad
        c_lat, c_lon = rad(covered["lat"].to_numpy()), rad(covered["lon"].to_numpy())
        for lat, lon in zip(missing["lat"], missing["lon"]):
            d = np.arccos(np.clip(np.sin(rad(lat)) * np.sin(c_lat) + np.cos(rad(lat)) * np.cos(c_lat) * np.cos(rad(lon) - c_lon), -1, 1))
            j = int(np.argmin(d))
            src_lat, src_lon = float(covered["lat"].iloc[j]), float(covered["lon"].iloc[j])
            fill_rows.append({"lat": lat, "lon": lon, "source_lat": src_lat, "source_lon": src_lon, "distance_km": float(d[j] * 6371.0)})
    fills = pd.DataFrame(fill_rows, columns=["lat", "lon", "source_lat", "source_lon", "distance_km"])
    frames = [ameli[["lat", "lon", "tech", "interest_rate"]]]
    for _, f in fills.iterrows():
        src = rates.loc[(f.source_lat, f.source_lon)]
        frames.append(pd.DataFrame({"lat": f.lat, "lon": f.lon, "tech": techs, "interest_rate": [float(src[t]) for t in techs]}))
    ameli_out = pd.concat(frames, ignore_index=True)
    ameli_out = ameli_out.merge(cells, on=["lat", "lon"], how="inner").sort_values(["lat", "lon", "tech"])
    ameli_path = a.output_dir / "amelired_interest_inputs_2050_center.csv"
    ameli_out.to_csv(ameli_path, index=False, float_format="%.6g")
    fill_path = a.output_dir / "amelired_interest_fill_2050_center.csv"
    fills.sort_values(["lat", "lon"]).to_csv(fill_path, index=False, float_format="%.6g")
    for path in (uniform_path, ameli_path, fill_path):
        print(f"{sha256(path)}  {path}  rows={sum(1 for _ in path.open()) - 1}")
    print(f"cells {len(cells):,}; Ameli covered {len(covered):,}; filled from nearest covered cell {len(fills)} (max distance {fills.distance_km.max() if len(fills) else 0:.0f} km)")


if __name__ == "__main__":
    main()
