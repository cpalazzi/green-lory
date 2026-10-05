#!/usr/bin/env python3
"""Pin the 40-cell native download list, excluding already verified granules."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from reconciliation.land.plan_native_downloads import query_url, parse_response


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024*1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser()
    for key in ("cells", "previous-manifest", "native-manifest", "output"):
        p.add_argument("--"+key, type=Path, required=True)
    a = p.parse_args()
    if a.output.exists():
        raise FileExistsError(a.output)
    previous = json.loads(a.previous_manifest.read_text())
    expected = [r for r in previous["source_files"] if Path(r["path"]).name == "pilot_cells.csv"]
    if len(expected) != 1 or digest(a.cells) != expected[0]["sha256"]:
        raise ValueError("Cell list differs from accepted 40-cell pilot")
    with a.cells.open() as f:
        cells = list(csv.DictReader(f))
    sites = [{"id": f"cell_{i:02d}", "latitude": float(c["latitude"]), "longitude": float(c["longitude"]),
              "country": c["country"]} for i, c in enumerate(cells, 1)]
    native = json.loads(a.native_manifest.read_text())["native_files"]
    received = {}
    for record in native:
        path = Path(record["path"])
        if digest(path) != record["sha256"]:
            raise ValueError("Received native file changed")
        received[path.name] = record
    def query(site):
        url, bounds = query_url(site, 2022, "center")
        with urlopen(url, timeout=30) as response:
            body = response.read()
        parsed = parse_response(json.loads(body), 2022)
        print(f"{site['id']} {site['latitude']},{site['longitude']}: {len(parsed)} intersecting granule(s)", flush=True)
        return site, url, bounds, body, parsed
    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(query, sites))
    queries, granules = [], {}
    for site, url, bounds, body, parsed in results:
        queries.append({**site, "bounds_wsen": bounds, "url": url, "raw_response": site["id"]+".cmr.json",
                        "raw_response_sha256": hashlib.sha256(body).hexdigest()})
        for row in parsed:
            name = row["filename"]
            if name in granules and {k: v for k, v in granules[name].items() if k != "sites"} != row:
                raise ValueError("Conflicting granule metadata")
            if name not in granules:
                granules[name] = {**row, "sites": []}
            granules[name]["sites"].append(site["id"])
    missing = [row for name, row in sorted(granules.items()) if name not in received]
    size = 0
    for row in missing:
        sizes = row["catalogue_size"]
        if len(sizes) != 1 or sizes[0]["SizeUnit"] != "MB":
            raise ValueError("Unsupported catalogue size units")
        size += round(sizes[0]["Size"]*2**20)
    manifest = {"created_utc": datetime.now(timezone.utc).isoformat(), "product": "MCD12Q1", "collection": "061", "year": 2022,
                "anchor": "center", "source_dates_match_paper": False, "sites": sites, "queries": queries,
                "cells_sha256": digest(a.cells), "native_receipt_manifest_sha256": digest(a.native_manifest),
                "granules": sorted(granules.values(), key=lambda r: r["filename"]),
                "already_received": [r for name, r in sorted(received.items()) if name in granules],
                "additional_files": len(missing), "additional_size_bytes": size, "script_sha256": digest(Path(__file__)),
                "parser_sha256": digest(ROOT/"reconciliation/land/plan_native_downloads.py"),
                "scope": "Catalogue intersections for the accepted 40 centered cells; native pixel coverage must still be validated. No raster download or credentials handled."}
    a.output.mkdir(parents=True, exist_ok=False)
    for site, _, _, body, _ in results:
        (a.output/(site["id"]+".cmr.json")).write_bytes(body)
    (a.output/"manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")
    (a.output/"additional_download_urls.txt").write_text("\n".join(r["url"] for r in missing)+"\n")
    (a.output/"all_download_urls.txt").write_text("\n".join(r["url"] for r in manifest["granules"])+"\n")
    lines = ["# Additional native MODIS downloads", "", f"For the 40-cell pilot: {len(missing)} additional original HDF files, approximately {size/2**20:.1f} MiB.",
             "The three verified files already stored in the campaign are excluded. Sign in to Earthdata before following the links. Keep original filenames.",
             "", "| Tile | File | Size (MiB) |", "|---|---|---:|"]
    lines += [f"| {r['tile']} | [{r['filename']}]({r['url']}) | {r['catalogue_size'][0]['Size']:.2f} |" for r in missing]
    (a.output/"DOWNLOADS.md").write_text("\n".join(lines)+"\n")
    print(json.dumps({"sites": len(sites), "total_tiles": len(granules), "additional_files": len(missing), "additional_size_MiB": size/2**20}, indent=2))


if __name__ == "__main__":
    main()
