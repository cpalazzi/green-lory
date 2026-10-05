#!/usr/bin/env python3
"""Read only selected cells from the separate green-condor 2019 CF tiles.

The source stores are never changed. A new evidence directory is required.
This is a technology-and-weather profile comparison, not a weather-only OAT.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    sites = [("atacama", -23, -69), ("northwest_australia", -23, 117),
             ("central_australia", -21, 135)]
    rows, inventory = [], []
    for path in sorted(a.source.glob("global_cf_2019_tiles_*.zarr")):
        ds = xr.open_zarr(path, consolidated=False, chunks=None)
        ys = ds.y.values
        inventory.append({"path": str(path), "sizes": dict(ds.sizes),
            "y_min": float(ys.min()), "y_max": float(ys.max()), "attrs": ds.attrs,
            "group_metadata_sha256": hashlib.sha256((path / "zarr.json").read_bytes()).hexdigest()})
        for site, lat, lon in sites:
            if not np.any(ys == lat):
                continue
            for variable in ("cf_solar", "cf_wind"):
                series = ds[variable].sel(y=lat, x=lon).load()
                x = np.asarray(series.values, dtype=float)
                valid = np.isfinite(x)
                rows.append({"site": site, "latitude": lat, "longitude": lon,
                    "variable": variable, "source_store": str(path), "observations": x.size,
                    "finite_observations": int(valid.sum()), "missing_observations": int((~valid).sum()),
                    "cf_mean_if_complete": float(x.mean()) if valid.all() else None,
                    "peak_if_complete": float(x.max()) if valid.all() else None,
                    "time_start": str(ds.time.values[0]), "time_end": str(ds.time.values[-1]),
                    "additional_wake_multiplier_applied": False,
                    "profile_float64_sha256": hashlib.sha256(np.asarray(x, dtype="<f8").tobytes()).hexdigest()})
                series.to_netcdf(a.output / f"{site}_{variable}.nc")
        ds.close()
    pd.DataFrame(rows).to_csv(a.output / "selected_capacity_factors.csv", index=False)
    complete = len(rows) == 6 and all(r["observations"] == r["finite_observations"] == 8760 for r in rows)
    payload = {"selected_cells_complete": complete, "global_completeness_verified": False,
        "source_stores": inventory, "rows": rows,
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "note": "No tracking profile in this CF stack. No additional 0.93 wake multiplier applied. Technology definitions differ or are unverified against legacy profiles."}
    (a.output / "manifest.json").write_text(json.dumps(payload, indent=2, default=str) + "\n")
    print(json.dumps(payload, indent=2, default=str), flush=True)


if __name__ == "__main__":
    main()
