#!/usr/bin/env python3
"""Pin public NASA catalogue metadata for a small native-MODIS pilot.

No raster download, authentication or credential handling is performed. The
2022 C6.1 plan is a resolution-controlled comparison, not a historical replay.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
from urllib.parse import urlencode, urlparse
from urllib.request import urlopen

CMR = "https://cmr.earthdata.nasa.gov/search/granules.umm_json"
PRODUCT = "MCD12Q1"
VERSION = "061"
PAGE_SIZE = 100


def query_url(site, year, anchor):
    lat, lon = float(site["latitude"]), float(site["longitude"])
    if not all(map(math.isfinite, (lat, lon))) or anchor not in {"center", "southwest"}:
        raise ValueError("Invalid pilot geometry")
    offset = .5 if anchor == "center" else 0
    bounds = (lon - offset, lat - offset, lon - offset + 1, lat - offset + 1)
    west, south, east, north = bounds
    if not (-180 <= west < east <= 180 and -90 <= south < north <= 90):
        raise ValueError("This pilot requires non-dateline, nonpolar cell bounds")
    if not 2001 <= year <= datetime.now(timezone.utc).year:
        raise ValueError("Invalid MODIS year")
    return CMR + "?" + urlencode({
        "short_name": PRODUCT, "version": VERSION, "provider": "LPCLOUD",
        "temporal": f"{year}-01-01T00:00:00Z,{year}-12-31T23:59:59Z",
        "bounding_box": ",".join(format(x, ".10g") for x in bounds),
        "page_size": PAGE_SIZE,
    }), bounds


def parse_response(response, year):
    items = response["items"]
    if not items or int(response["hits"]) != len(items):
        raise ValueError("Empty or paginated catalogue response; no complete download plan")
    rows = []
    for item in items:
        meta, data = item["meta"], item["umm"]
        collection = data["CollectionReference"]
        if (collection["ShortName"], collection["Version"], meta["provider-id"]) != (PRODUCT, VERSION, "LPCLOUD"):
            raise ValueError("Unexpected product, collection or provider")
        name = data["GranuleUR"]
        match = re.fullmatch(rf"MCD12Q1\.A{year}001\.(h\d{{2}}v\d{{2}})\.061\.\d{{13}}", name)
        if not match:
            raise ValueError("Unexpected native granule name/year")
        urls = [link["URL"] for link in data["RelatedUrls"]
                if link["Type"] == "GET DATA" and
                urlparse(link["URL"]).scheme == "https" and
                urlparse(link["URL"]).hostname == "data.lpdaac.earthdatacloud.nasa.gov" and
                urlparse(link["URL"]).path.endswith("/" + name + ".hdf")]
        if len(set(urls)) != 1:
            raise ValueError("Expected one official native HDF download URL")
        rows.append({
            "filename": name + ".hdf", "tile": match.group(1), "url": urls[0],
            "concept_id": meta["concept-id"], "revision_id": meta["revision-id"],
            "collection_concept_id": meta["collection-concept-id"],
            "catalogue_size": data["DataGranule"]["ArchiveAndDistributionInformation"],
            "spatial_extent": data["SpatialExtent"],
        })
    if len({row["tile"] for row in rows}) != len(rows):
        raise ValueError("Multiple revisions/files for one tile; resolve before downloading")
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True,
                        help="JSON containing a sites list with id/latitude/longitude")
    parser.add_argument("--year", type=int, default=2022)
    parser.add_argument("--anchor", choices=["center", "southwest"], default="center")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    config_bytes = args.config.read_bytes()
    sites = json.loads(config_bytes)["sites"]
    ids = [site["id"] for site in sites]
    if not sites or len(set(ids)) != len(ids) or any(not re.fullmatch(r"[a-z0-9_]+", x) for x in ids):
        raise ValueError("Nonempty unique safe site IDs required")
    # Complete and validate every catalogue request before creating outputs.
    raw, queries, granules = {}, [], {}
    for site in sites:
        url, bounds = query_url(site, args.year, args.anchor)
        with urlopen(url, timeout=30) as response:
            body = response.read()
        filename = site["id"] + ".cmr.json"
        raw[filename] = body
        rows = parse_response(json.loads(body), args.year)
        queries.append({"site": site["id"], "bounds_wsen": bounds, "url": url,
                        "raw_response": filename,
                        "raw_response_sha256": hashlib.sha256(body).hexdigest()})
        for row in rows:
            name = row["filename"]
            if name in granules:
                if {k: v for k, v in granules[name].items() if k != "sites"} != row:
                    raise ValueError("Inconsistent metadata across overlapping site queries")
                granules[name]["sites"].append(site["id"])
            else:
                granules[name] = {**row, "sites": [site["id"]]}
    manifest = {
        "schema_version": 1, "created_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": "native_resolution_controlled_pilot_not_historical_replay",
        "product": PRODUCT, "collection": VERSION, "year": args.year,
        "anchor": args.anchor, "source_dates_match_paper": False,
        "required_science_datasets": ["LC_Type1", "QC", "LW"],
        "native_files_required": True, "raster_files_downloaded": False,
        "scope": "Catalogue intersections only; pixel coverage and masks require validation after download.",
        "config_sha256": hashlib.sha256(config_bytes).hexdigest(),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "queries": queries, "granules": sorted(granules.values(), key=lambda x: x["filename"]),
    }
    args.output.mkdir(parents=True, exist_ok=False)
    for name, body in raw.items():
        (args.output / name).write_bytes(body)
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (args.output / "download_urls.txt").write_text("\n".join(g["url"] for g in manifest["granules"]) + "\n")
    print(json.dumps({"manifest": str(args.output / "manifest.json"),
                      "files": [{"filename": g["filename"], "sites": g["sites"],
                                 "size": g["catalogue_size"]} for g in manifest["granules"]]}, indent=2))


if __name__ == "__main__":
    main()
