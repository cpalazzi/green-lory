#!/usr/bin/env python
"""Gather every per-cell ``summary.json`` of a legacy run into one JSON-lines file.

Purpose: a global run holds 15,377 cell directories with five small files each;
transferring or scanning them individually on a parallel file system is slow.
This tool reads each ``summary.json`` once and writes ``summaries.jsonl`` (one
JSON object per line, unchanged content) plus ``collect_manifest.json`` with the
run manifests, the per-shard results.csv row counts, the number of summaries
and the SHA-256 of the output. ``build_legacy_supplier_table.py`` and
``compare_runs.py`` accept the JSON-lines file in place of the run directory.

    python collect_summaries.py --run <run dir with shard_XX/ or cell dirs> --output <new dir>
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import time
from pathlib import Path


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def summary_files(run: Path) -> list[str]:
    return sorted(glob.glob(str(run / "shard_*" / "*__*__*" / "summary.json")) + glob.glob(str(run / "*__*__*" / "summary.json")))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    if a.output.exists():
        raise SystemExit(f"refusing to overwrite {a.output}")
    t0 = time.time()
    files = summary_files(a.run)
    a.output.mkdir(parents=True)
    n = 0
    keys = set()
    with open(a.output / "summaries.jsonl", "w") as out:
        for f in files:
            with open(f) as fh:
                s = json.load(fh)
            s["_source_path"] = os.path.relpath(f, a.run)
            key = (s["lat"], s["lon"], s["variant"], s["wacc_name"])
            if key in keys:
                raise SystemExit(f"duplicate cell {key} at {f}")
            keys.add(key)
            out.write(json.dumps(s, separators=(",", ":")) + "\n")
            n += 1
            if n % 2000 == 0:
                print(f"{n}/{len(files)} ({time.time() - t0:.0f} s)", flush=True)
    manifests = {}
    for m in sorted(glob.glob(str(a.run / "shard_*" / "manifest*.json")) + glob.glob(str(a.run / "manifest*.json"))):
        manifests[os.path.relpath(m, a.run)] = json.load(open(m))
    counts = {}
    for r in sorted(glob.glob(str(a.run / "shard_*" / "results.csv")) + glob.glob(str(a.run / "results.csv"))):
        with open(r) as fh:
            counts[os.path.relpath(r, a.run)] = sum(1 for _ in fh) - 1
    meta = {"run": str(a.run.resolve()), "summaries": n, "results_csv_rows": counts,
            "output_sha256": sha256_file(a.output / "summaries.jsonl"), "manifests": manifests,
            "script_sha256": sha256_file(Path(__file__)), "elapsed_s": time.time() - t0}
    with open(a.output / "collect_manifest.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(json.dumps({k: v for k, v in meta.items() if k != "manifests"}, indent=2))


if __name__ == "__main__":
    main()
