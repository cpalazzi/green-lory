#!/usr/bin/env python
"""Annual mean capacity factors for every cell of the legacy 2019 weather stack.

Reads the nine NetCDF files (Solar*, SolarTracking*, WindPowers* x longitude
bands) and writes one row per (latitude, longitude) with the annual mean of each
profile, the peak value, and the count of hours above 1.0. Wind is reported raw
(the plant model multiplies it by 0.93 at load time). No file is modified.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--weather-dir", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True, help="new directory (must not exist)")
    args = p.parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite {args.output}")
    args.output.mkdir(parents=True)

    files = sorted(args.weather_dir.glob("*.nc"))
    files = [f for f in files if f.name != "model_bathymetry.nc"]
    manifest = {"weather_dir": str(args.weather_dir), "files": [], "started": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
    frames = []
    for f in files:
        t0 = time.time()
        if "SolarTracking" in f.name:
            tech, var = "tracking_pv", "Solar"
        elif "Solar" in f.name:
            tech, var = "fixed_pv", "Solar"
        elif "WindPower" in f.name:
            tech, var = "wind_raw", "Wind"
        else:
            continue
        ds = xr.open_dataset(f)
        da = ds[var]
        n_time = int(da.sizes["time"])
        mean = da.mean("time").to_dataframe(name="cf").reset_index()
        peak = da.max("time").to_dataframe(name="peak").reset_index()
        above = (da > 1.0).sum("time").to_dataframe(name="hours_above_one").reset_index()
        df = mean.merge(peak, on=["latitude", "longitude"]).merge(above, on=["latitude", "longitude"])
        df["tech"] = tech
        df["source_file"] = f.name
        df["n_time"] = n_time
        df["time_start"] = str(pd.to_datetime(ds["time"].values[0]))
        df["time_end"] = str(pd.to_datetime(ds["time"].values[-1]))
        frames.append(df)
        st = f.stat()
        manifest["files"].append({"name": f.name, "bytes": st.st_size, "mtime": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(st.st_mtime)),
                                  "tech": tech, "variable": var, "n_time": n_time, "n_cells": int(len(df)),
                                  "lon_min": float(df.longitude.min()), "lon_max": float(df.longitude.max()),
                                  "seconds": round(time.time() - t0, 1)})
        print(f"{f.name}: {len(df)} cells, {n_time} steps, lon {df.longitude.min()}..{df.longitude.max()} ({time.time()-t0:.0f}s)", flush=True)
        ds.close()
    long = pd.concat(frames, ignore_index=True)
    long.to_csv(args.output / "annual_cf_long.csv", index=False)
    wide = long.pivot_table(index=["latitude", "longitude"], columns="tech", values="cf").reset_index()
    peaks = long.pivot_table(index=["latitude", "longitude"], columns="tech", values="peak").reset_index()
    peaks.columns = ["latitude", "longitude"] + [f"{c}_peak" for c in peaks.columns[2:]]
    wide = wide.merge(peaks, on=["latitude", "longitude"])
    wide["wind_net_0.93"] = wide["wind_raw"] * 0.93
    wide.to_csv(args.output / "annual_cf_2019.csv", index=False)
    manifest["n_cells_wide"] = int(len(wide))
    manifest["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    with open(args.output / "manifest.json", "w") as fh:
        json.dump(manifest, fh, indent=2)
    print("done", len(wide), "cells", flush=True)


if __name__ == "__main__":
    main()
