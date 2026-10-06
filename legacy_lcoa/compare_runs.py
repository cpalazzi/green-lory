#!/usr/bin/env python
"""Cell-by-cell comparison of two legacy-lcoa run directories.

Used to cross-check the ARC execution of the global replication surface against
the Mac execution of the same configuration (same frozen code, same weather
store values, different machine and Gurobi build). Loads every
``summary.json`` below each run (``shard_XX/<cell>__<variant>__<wacc>/`` or the
flat layout), joins on (latitude, longitude, variant, wacc_name) and reports
the distribution of differences in LCOA, objective and solved capacities.

    python compare_runs.py --run-a <dir> --run-b <dir> --output <new dir> \
        [--label-a arc --label-b mac]
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

CAPS = ["Wind", "Solar", "SolarTracking", "Electrolysis", "HB", "CompressedH2Store", "Battery", "Ammonia"]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def iter_summaries(run: Path):
    if run.is_file():   # summaries.jsonl from collect_summaries.py
        with open(run) as fh:
            for line in fh:
                if line.strip():
                    yield json.loads(line)
        return
    for f in glob.glob(str(run / "shard_*" / "*__*__*" / "summary.json")) + glob.glob(str(run / "*__*__*" / "summary.json")):
        yield json.load(open(f))


def load(run: Path) -> tuple[pd.DataFrame, dict]:
    rows = []
    for s in iter_summaries(run):
        c = s["capacities_mw"]
        rows.append({"latitude": s["lat"], "longitude": s["lon"], "variant": s["variant"], "wacc_name": s["wacc_name"],
                     "wacc_rate": s["wacc_rate"], "lcoa": s["lcoa_usd_per_t"], "objective": s["objective_usd_per_year"],
                     "snapshots": s["snapshots_used"], "elapsed_s": s.get("elapsed_s"),
                     **{f"cap_{k}": float(c.get(k, 0.0)) for k in CAPS}})
    df = pd.DataFrame(rows)
    if df.duplicated(["latitude", "longitude", "variant", "wacc_name"]).any():
        raise SystemExit(f"duplicate cells in {run}")
    if run.is_file():
        cm = json.load(open(run.with_name("collect_manifest.json")))
        manifests = sorted(k for k in cm["manifests"] if k.endswith("/manifest.json") or k == "manifest.json")
        loaded = {k: cm["manifests"][k] for k in manifests}
    else:
        manifests = sorted(glob.glob(str(run / "shard_*" / "manifest.json")) + glob.glob(str(run / "manifest.json")))
        loaded = {k: json.load(open(k)) for k in manifests}
    meta = {}
    if manifests:
        m = loaded[manifests[0]]
        meta = {"manifest": manifests[0], "n_manifests": len(manifests), "packages": m.get("packages"), "python": m.get("python"),
                "harness_sha256": m.get("harness"), "commit": m.get("commit"), "era": m.get("era"),
                "capex_source": m.get("capex_source"), "enable_tracking_cost_ratio": m.get("enable_tracking_cost_ratio"),
                "weather_source_kind": m.get("weather_source", {}).get("kind"),
                "weather_files": m.get("weather_files"), "solver": m.get("solver")}
        # every shard manifest of a run must describe the same configuration
        keys = ["harness", "commit", "era", "capex_source", "enable_tracking_cost_ratio", "weather_files", "source_files", "costs_xlsx"]
        for other in manifests[1:]:
            o = loaded[other]
            for k in keys:
                if o.get(k) != m.get(k):
                    raise SystemExit(f"manifest field {k} differs between {manifests[0]} and {other}")
        meta["all_shard_manifests_consistent"] = True
    return df, meta


def q(s: pd.Series) -> dict:
    s = s.dropna()
    if not len(s):
        return {"n": 0}
    return {"n": int(len(s)), "max": float(s.max()), "median": float(s.median()),
            "p99": float(s.quantile(0.99)), "p999": float(s.quantile(0.999))}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run-a", type=Path, required=True)
    p.add_argument("--run-b", type=Path, required=True)
    p.add_argument("--label-a", default="a")
    p.add_argument("--label-b", default="b")
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    if a.output.exists():
        raise SystemExit(f"refusing to overwrite {a.output}")
    da, ma = load(a.run_a)
    db, mb = load(a.run_b)
    key = ["latitude", "longitude", "variant", "wacc_name"]
    both = da.merge(db, on=key, suffixes=(f"_{a.label_a}", f"_{a.label_b}"))
    only_a = len(da) - len(both)
    only_b = len(db) - len(both)
    if not np.allclose(both[f"wacc_rate_{a.label_a}"], both[f"wacc_rate_{a.label_b}"]):
        raise SystemExit("WACC rates differ for the same cells")
    if not (both[f"snapshots_{a.label_a}"] == both[f"snapshots_{a.label_b}"]).all():
        raise SystemExit("snapshot counts differ for the same cells")
    la, lb = both[f"lcoa_{a.label_a}"], both[f"lcoa_{a.label_b}"]
    both["lcoa_abs_diff"] = (la - lb).abs()
    both["lcoa_rel_diff"] = both.lcoa_abs_diff / la
    oa, ob = both[f"objective_{a.label_a}"], both[f"objective_{a.label_b}"]
    both["objective_rel_diff"] = (oa - ob).abs() / oa
    for k in CAPS:
        ca, cb = both[f"cap_{k}_{a.label_a}"], both[f"cap_{k}_{a.label_b}"]
        both[f"cap_{k}_abs_diff_mw"] = (ca - cb).abs()
        scale = np.maximum(np.maximum(ca.abs(), cb.abs()), 1.0)
        both[f"cap_{k}_rel_diff"] = both[f"cap_{k}_abs_diff_mw"] / scale
    a.output.mkdir(parents=True)
    both.to_csv(a.output / "per_cell.csv", index=False)
    summary = {
        "run_a": {"label": a.label_a, "path": str(a.run_a.resolve()), "cells": int(len(da)), **ma},
        "run_b": {"label": a.label_b, "path": str(a.run_b.resolve()), "cells": int(len(db)), **mb},
        "common_cells": int(len(both)), "only_in_a": int(only_a), "only_in_b": int(only_b),
        "lcoa_abs_diff_usd_per_t": q(both.lcoa_abs_diff),
        "lcoa_rel_diff": q(both.lcoa_rel_diff),
        "objective_rel_diff": q(both.objective_rel_diff),
        "cells_lcoa_rel_diff_above_1e-6": int((both.lcoa_rel_diff > 1e-6).sum()),
        "cells_lcoa_rel_diff_above_1e-4": int((both.lcoa_rel_diff > 1e-4).sum()),
        "cells_lcoa_rel_diff_above_1e-3": int((both.lcoa_rel_diff > 1e-3).sum()),
        "capacity_rel_diff": {k: q(both[f"cap_{k}_rel_diff"]) for k in CAPS},
        "capacity_abs_diff_mw": {k: q(both[f"cap_{k}_abs_diff_mw"]) for k in CAPS},
        "cells_any_capacity_rel_diff_above_1e-3": int((both[[f"cap_{k}_rel_diff" for k in CAPS]].max(axis=1) > 1e-3).sum()),
        "elapsed_s_median": {a.label_a: float(both[f"elapsed_s_{a.label_a}"].median()), a.label_b: float(both[f"elapsed_s_{a.label_b}"].median())},
        "interpretation": "LCOA and objective are the replication targets; capacity differences at equal objective indicate "
                          "alternative optima (degenerate LP), not configuration differences.",
        "script_sha256": sha256_file(Path(__file__)),
    }
    with open(a.output / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps({k: v for k, v in summary.items() if k not in ("run_a", "run_b")}, indent=2))
    print("run_a:", {k: ma.get(k) for k in ("packages", "commit", "capex_source", "weather_source_kind")})
    print("run_b:", {k: mb.get(k) for k in ("packages", "commit", "capex_source", "weather_source_kind")})


if __name__ == "__main__":
    main()
