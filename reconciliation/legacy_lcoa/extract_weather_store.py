#!/usr/bin/env python
"""Extract per-cell hourly profiles from the nine legacy NetCDF weather files
into a compact store that the harness can read cell by cell.

Why: the legacy files are contiguous NETCDF3_64BIT_OFFSET arrays laid out
(time, latitude, longitude). Reading one cell's 8,760-hour series is a strided
read across the whole 1.5 GB file, which costs 13-70 s per file on ARC's
parallel file system (and the harness hashes all nine files at start-up,
another eight minutes). Sequential block reads are fast, so this tool reads
each file once in time blocks, gathers the requested cells, and writes one
float64 array per source file plus a manifest with the source hashes. Values
are copied unchanged (float64), so a run on the store is numerically identical
to a run on the files.

Band rule (frozen `renewable_data.get_data_from_nc`): a cell reads band 0 if
longitude < -60, band 1 if longitude < 60, else band 2; band k is the k-th
file of each technology in sorted name order (unsuffixed, `1`, `2`). A cell
whose coordinate is not present in its band file is recorded as missing (the
frozen code would fail on it too).

Per-file mode (parallel, e.g. one Slurm array task per file):

    python extract_weather_store.py --weather-dir DIR --cells cells.csv --output STORE --file Solar1.nc

Merge mode (after all nine files): validates the time axes and writes manifest.json:

    python extract_weather_store.py --weather-dir DIR --cells cells.csv --output STORE --merge
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

BLOCK = 730  # time steps per block: 730 x 180 x 120 x 8 B = 126 MB


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


def band_of(lon: float) -> int:
    return 0 if lon < -60 else (1 if lon < 60 else 2)


def technology_files(weather_dir: Path) -> dict[str, list[Path]]:
    files = sorted(f for f in weather_dir.glob("*.nc") if f.name != "model_bathymetry.nc")
    groups = {"Solar": [], "Wind": [], "SolarTracking": []}
    for f in files:
        if "SolarTracking" in f.name:
            groups["SolarTracking"].append(f)
        elif "Solar" in f.name:
            groups["Solar"].append(f)
        elif "WindPower" in f.name:
            groups["Wind"].append(f)
    for tech, lst in groups.items():
        if len(lst) not in (1, 3):
            raise SystemExit(f"expected 1 or 3 files for {tech}, found {[p.name for p in lst]}")
    return groups


def file_role(weather_dir: Path, name: str) -> tuple[str, int, str]:
    """technology, band index and NetCDF variable name of a file (same ordering rule as the harness)."""
    for tech, lst in technology_files(weather_dir).items():
        names = [p.name for p in lst]
        if name in names:
            band = names.index(name) if len(lst) == 3 else -1  # -1: single file serves all bands
            var = "Wind" if tech == "Wind" else "Solar"
            return tech, band, var
    raise SystemExit(f"{name} is not one of the weather files in {weather_dir}")


def extract_file(args):
    import netCDF4
    src = args.weather_dir / args.file
    tech, band, var = file_role(args.weather_dir, args.file)
    cells = pd.read_csv(args.cells)[["lat", "lon"]].drop_duplicates().reset_index(drop=True)
    cells["band"] = [band_of(float(v)) for v in cells.lon]
    wanted = cells if band < 0 else cells[cells.band == band]
    args.output.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    ds = netCDF4.Dataset(src)
    v = ds.variables[var]
    dims = v.dimensions
    if dims != ("time", "latitude", "longitude"):
        raise SystemExit(f"{src.name}: unexpected layout {dims}")
    lats = ds.variables["latitude"][:].astype(float)
    lons = ds.variables["longitude"][:].astype(float)
    lat_index = {float(a): i for i, a in enumerate(lats)}
    lon_index = {float(a): i for i, a in enumerate(lons)}
    rows, missing = [], []
    for lat, lon in zip(wanted.lat, wanted.lon):
        i, j = lat_index.get(float(lat)), lon_index.get(float(lon))
        if i is None or j is None:
            missing.append([float(lat), float(lon)])
        else:
            rows.append((float(lat), float(lon), i, j))
    n_time = v.shape[0]
    out = np.empty((len(rows), n_time), dtype=np.float64)
    ii = np.array([r[2] for r in rows], dtype=int)
    jj = np.array([r[3] for r in rows], dtype=int)
    for start in range(0, n_time, BLOCK):
        stop = min(start + BLOCK, n_time)
        block = np.asarray(v[start:stop, :, :])           # sequential read, (t, lat, lon)
        if np.ma.isMaskedArray(block):
            block = block.filled(np.nan)
        out[:, start:stop] = block[:, ii, jj].T
        print(f"{src.name}: {stop}/{n_time} steps, {time.time() - t0:.0f} s", flush=True)
    time_values = ds.variables["time"]
    try:
        import netCDF4 as nc
        times = nc.num2date(time_values[:], time_values.units, getattr(time_values, "calendar", "standard"),
                            only_use_cftime_datetimes=False, only_use_python_datetimes=True)
        times = np.array(pd.to_datetime(times).values)
    except Exception:  # pragma: no cover
        times = np.asarray(time_values[:])
    ds.close()
    stem = src.stem
    np.save(args.output / f"{stem}.npy", out)
    np.save(args.output / f"{stem}.time.npy", times)
    pd.DataFrame(rows, columns=["lat", "lon", "ilat", "ilon"]).to_csv(args.output / f"{stem}.cells.csv", index=False)
    info = {"source": str(src.resolve()), "source_sha256": sha256_file(src), "source_bytes": src.stat().st_size,
            "technology": tech, "band": band, "variable": var, "n_time": int(n_time), "rows": len(rows),
            "missing_cells": missing, "array": f"{stem}.npy", "array_sha256": sha256_file(args.output / f"{stem}.npy"),
            "cells_csv": f"{stem}.cells.csv", "dtype": "float64", "extracted": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "elapsed_s": round(time.time() - t0, 1)}
    with open(args.output / f"{stem}.json", "w") as f:
        json.dump(info, f, indent=2)
    print(f"{src.name}: {len(rows)} cells, {len(missing)} missing, {info['elapsed_s']} s", flush=True)


def merge(args):
    groups = technology_files(args.weather_dir)
    cells = pd.read_csv(args.cells)[["lat", "lon"]].drop_duplicates()
    files, times = {}, None
    for tech, lst in groups.items():
        for p in lst:
            info_path = args.output / f"{p.stem}.json"
            if not info_path.exists():
                raise SystemExit(f"missing extraction for {p.name}")
            info = json.loads(info_path.read_text())
            if sha256_file(args.output / info["array"]) != info["array_sha256"]:
                raise SystemExit(f"array hash mismatch for {p.name}")
            t = np.load(args.output / f"{p.stem}.time.npy", allow_pickle=False)
            if times is None:
                times = t
            elif not np.array_equal(times, t):
                raise SystemExit(f"time axis of {p.name} differs from the first file")
            files[p.name] = info
    covered = set()
    for tech, lst in groups.items():
        for p in lst:
            c = pd.read_csv(args.output / f"{p.stem}.cells.csv")
            covered |= {(tech, float(a), float(b)) for a, b in zip(c.lat, c.lon)}
    n_missing = {tech: int(sum((tech, float(a), float(b)) not in covered for a, b in zip(cells.lat, cells.lon))) for tech in groups}
    np.save(args.output / "time.npy", times)
    manifest = {"schema": "legacy_weather_store_v1", "weather_dir": str(args.weather_dir.resolve()),
                "cells": {"path": str(args.cells.resolve()), "sha256": sha256_file(args.cells), "n": int(len(cells))},
                "band_rule": "0 if lon < -60 else 1 if lon < 60 else 2; band k = k-th file per technology in sorted name order",
                "files": files, "n_time": int(len(times)), "time_first": str(times[0]), "time_last": str(times[-1]),
                "cells_without_data_by_technology": n_missing, "extractor_sha256": sha256_file(Path(__file__)),
                "merged": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
    with open(args.output / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    print(json.dumps({k: manifest[k] for k in ("n_time", "time_first", "time_last", "cells_without_data_by_technology")}, indent=2))
    print("store manifest written:", args.output / "manifest.json")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--weather-dir", type=Path, required=True)
    p.add_argument("--cells", type=Path, required=True, help="CSV with lat, lon columns")
    p.add_argument("--output", type=Path, required=True, help="store directory")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--file", help="extract this one NetCDF file (name inside --weather-dir)")
    g.add_argument("--all", action="store_true", help="extract every file sequentially, then merge")
    g.add_argument("--merge", action="store_true", help="validate and write manifest.json")
    args = p.parse_args()
    if args.merge:
        merge(args)
    elif args.all:
        for lst in technology_files(args.weather_dir).values():
            for path in lst:
                args.file = path.name
                extract_file(args)
        merge(args)
    else:
        extract_file(args)


if __name__ == "__main__":
    main()
