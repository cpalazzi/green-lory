#!/usr/bin/env python3
"""Controlled native-Q1 versus CMG land pilot on identical exclusion masks.

The fine geographic grid is numerical quadrature, not a finer observation.
Native categories and DEM slopes are sampled without interpolation. The
unmasked native ledger independently clips sinusoidal pixels to the cell.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time

import geopandas as gpd
import numpy as np
import pandas as pd
from pyhdf.SD import SD, SDC
import shapely
import xarray as xr

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from reconciliation.land.core import Cell, RADIUS_KM, SOLAR, WIND, read_cmg_cell, rectangle_area
from reconciliation.land.native_modis import NativeTile, class_histogram
from reconciliation.land.pilot_joint_masks import sha
from model.land_processing import _filter_protected_areas, _geometry_union

MASKS = ("raw", "slope_only", "protection_only", "joint")
FACTORS = {"solar": SOLAR, "wind": WIND, "nested_union": np.maximum(SOLAR, WIND)}


def dem_slopes(elevation, cell):
    """Same centered-gradient physical slope as the preceding CMG pilot."""
    west, south, east, north = cell.bounds
    dy = abs(float(elevation.lat[1] - elevation.lat[0]))
    dx = abs(float(elevation.lon[1] - elevation.lon[0]))
    def select(coord, lo, hi):
        return slice(lo, hi) if float(coord[0]) < float(coord[-1]) else slice(hi, lo)
    band = elevation.sel(lat=select(elevation.lat, south-2*dy, north+2*dy),
                         lon=select(elevation.lon, west-2*dx, east+2*dx))
    band = band.sortby("lat").sortby("lon").transpose("lat", "lon")
    lat, lon = band.lat.values.astype(float), band.lon.values.astype(float)
    z = band.values.astype(float)
    if min(z.shape) < 3 or not np.isfinite(z).all():
        raise ValueError("Missing DEM coverage")
    np.testing.assert_allclose(np.diff(lat), dy, atol=1e-9, rtol=0)
    np.testing.assert_allclose(np.diff(lon), dx, atol=1e-9, rtol=0)
    gy = np.gradient(z, axis=0)/(RADIUS_KM*1000*np.deg2rad(dy))
    gx = np.gradient(z, axis=1)/(RADIUS_KM*1000*np.deg2rad(dx)*np.cos(np.deg2rad(lat))[:, None])
    return lat, lon, np.rad2deg(np.arctan(np.hypot(gx, gy)))


def sample_regular(data, lat, lon, ys, xs):
    """Containing DEM pixel, with no extrapolation across coverage edges."""
    row = np.floor((ys-lat[0])/(lat[1]-lat[0])+.5).astype(int)
    col = np.floor((xs-lon[0])/(lon[1]-lon[0])+.5).astype(int)
    if (row < 0).any() or (row >= len(lat)).any() or (col < 0).any() or (col >= len(lon)).any():
        raise ValueError("Sampling outside DEM coverage")
    return data[row, col]


def integrate(cell, tile, fractions, slope_grid, protected, samples_per_degree, block_rows=100):
    """Joint class/mask area ledgers, with CMG and Q1 on exactly the same grid."""
    if samples_per_degree < 20 or samples_per_degree % 20:
        raise ValueError("Quadrature must align with 0.05-degree CMG boundaries")
    west, south, east, north = cell.bounds
    n = int(round(cell.degree*samples_per_degree))
    step = 1./samples_per_degree
    if not np.isclose(n*step, cell.degree) or fractions.shape != (20, 20, 17) or cell.degree != 1:
        raise ValueError("Pilot supports aligned one-degree cells only")
    fn = fractions/fractions.sum(axis=-1, keepdims=True)
    classes = np.zeros((2, len(MASKS), 17))
    cmg_mask_area = np.zeros((len(MASKS), 400))
    native_qc = np.zeros(256)
    land_water_mismatch = 0.
    if protected is not None:
        shapely.prepare(protected)
    for first in range(0, n, block_rows):
        row = np.arange(first, min(first+block_rows, n))
        ys = (south+(row+.5)*step)[:, None]
        xs = (west+(np.arange(n)+.5)*step)[None, :]
        area = np.broadcast_to(rectangle_area(south+row*step, south+(row+1)*step, step)[:, None], (len(row), n))
        native = tile.sample(xs, ys)
        slope_ok = sample_regular(slope_grid[2], slope_grid[0], slope_grid[1], ys, xs) <= 15.
        protection_ok = np.ones_like(slope_ok) if protected is None else ~shapely.intersects_xy(protected, xs, ys)
        masks = (np.ones_like(slope_ok), slope_ok, protection_ok, slope_ok & protection_ok)
        bins = np.broadcast_to(np.floor((north-ys)/.05).astype(int)*20 + np.floor((xs-west)/.05).astype(int), area.shape)
        if (bins < 0).any() or (bins >= 400).any():
            raise ValueError("CMG index outside study cell")
        for k, mask in enumerate(masks):
            weights = area*mask
            classes[0, k] += class_histogram(native["LC_Type1"], weights)
            cmg_mask_area[k] += np.bincount(bins.ravel(), weights=weights.ravel(), minlength=400)
        native_qc += np.bincount(native["QC"].ravel(), weights=area.ravel(), minlength=256)
        mismatch = ((native["LC_Type1"] == 17) & (native["LW"] == 2)) | ((native["LC_Type1"] != 17) & (native["LW"] == 1))
        land_water_mismatch += float(area[mismatch].sum())
    classes[1] = cmg_mask_area @ fn.reshape(400, 17)
    np.testing.assert_allclose(classes[:, 0].sum(axis=1), cell.area_km2, atol=1e-6, rtol=0)
    if (classes[:, 3] > np.minimum(classes[:, 1], classes[:, 2])+1e-7).any():
        raise ValueError("Joint exclusions exceed a marginal class area")
    if (classes[:, 1:3] > classes[:, :1]+1e-7).any() or (classes < -1e-7).any():
        raise ValueError("Class-area conservation failure")
    if native_qc[255] > 0:
        raise ValueError("Missing native quality flags in study cell")
    return {"classes": classes, "qc_areas_km2": native_qc,
            "land_water_mismatch_km2": land_water_mismatch,
            "mask_areas_km2": cmg_mask_area.sum(axis=1)}


def verify_sources(modis, dem, protected_paths, previous_manifest):
    """Prove the local mask/CMG files are identical to the previous ARC inputs."""
    old = json.loads(previous_manifest.read_text())["source_files"]
    sources = [modis, dem]
    for shp in protected_paths:
        sources.extend(shp.with_suffix(suffix) for suffix in (".cpg", ".dbf", ".prj", ".shp", ".shx"))
    inventory = []
    for path in sources:
        matches = [r for r in old if Path(r["path"]).name == path.name and
                   (path.suffix not in {".cpg", ".dbf", ".prj", ".shp", ".shx"} or Path(r["path"]).parent.name == path.parent.name)]
        if len(matches) != 1:
            raise ValueError(f"Ambiguous/missing previous provenance: {path}")
        print(f"Verify source {path.name} ({path.parent.name})", flush=True)
        digest = sha(path)
        if digest != matches[0]["sha256"] or path.stat().st_size != matches[0]["size_bytes"]:
            raise ValueError(f"Source changed since CMG pilot: {path}")
        inventory.append({"path": str(path.resolve()), "size_bytes": path.stat().st_size, "sha256": digest})
    return inventory


def main():
    p = argparse.ArgumentParser()
    for name in ("native", "plan", "modis", "dem", "previous", "config", "output"):
        p.add_argument("--"+name, type=Path, required=True)
    p.add_argument("--protected", type=Path, nargs="+", required=True)
    p.add_argument("--resolutions", type=int, nargs="+", default=[1200, 2400])
    a = p.parse_args()
    if len(a.resolutions) < 2 or a.resolutions != sorted(set(a.resolutions)):
        raise ValueError("At least two increasing quadrature resolutions required")
    a.output.mkdir(parents=True, exist_ok=False)
    plan = json.loads(a.plan.read_text())
    if (plan["product"], plan["collection"], plan["year"], plan["anchor"]) != ("MCD12Q1", "061", 2022, "center"):
        raise ValueError("Unexpected native download plan")
    native_inventory, tiles = [], {}
    for item in plan["granules"]:
        path = a.native/item["filename"]
        tile = NativeTile.read(path)
        sizes = item["catalogue_size"]
        if len(sizes) != 1 or sizes[0]["SizeUnit"] != "MB" or abs(path.stat().st_size-sizes[0]["Size"]*2**20) > 1:
            raise ValueError("Native file size differs from pinned NASA catalogue")
        native_inventory.append({"path": str(path.resolve()), "size_bytes": path.stat().st_size,
                                 "sha256": sha(path), "url": item["url"], "tile": item["tile"],
                                 "identity_validated": True, "pixel_size_metres": tile.dx})
        tiles[item["tile"]] = tile
    sources = verify_sources(a.modis, a.dem, a.protected, a.previous/"manifest.json")
    code = [Path(__file__), ROOT/"reconciliation/land/native_modis.py", ROOT/"reconciliation/land/core.py",
            ROOT/"reconciliation/land/pilot_joint_masks.py", ROOT/"model/land_processing.py"]
    previous_codes = json.loads((a.previous/"manifest.json").read_text())["code_sha256"]
    for name in ("reconciliation/land/core.py", "model/land_processing.py"):
        if sha(ROOT/name) != previous_codes[name]:
            raise ValueError(f"Common methods changed since previous pilot: {name}")
    manifest = {"created_utc": datetime.now(timezone.utc).isoformat(), "source_dates_match_paper": False,
                "purpose": "three_site_native_resolution_control_not_global", "native_files": native_inventory,
                "unchanged_cmg_and_mask_sources": sources, "config_sha256": sha(a.config),
                "download_plan_sha256": sha(a.plan), "previous_manifest_sha256": sha(a.previous/"manifest.json"),
                "quadrature_samples_per_degree": a.resolutions, "maximum_slope_degrees": 15.,
                "slope_method": "GEBCO centered physical gradient; containing DEM pixel; no elevation-sign exclusion",
                "protection_method": "same designated/inscribed/established non-marine polygon union; quadrature center inclusion; point records excluded",
                "native_method": "categorical containing sinusoidal pixel, plus independent area-clipped unmasked ledger",
                "cmg_method": "normalized 17 class fractions, uniform within each CMG pixel, identical quadrature masks",
                "shipping_fraction": .02, "overlap_method": "classwise nested lower-bound overlap, not observed technology overlap",
                "qc_policy": "report flags without excluding classifications; reject fill; report LC_Type1/LW disagreement",
                "convergence_tolerance_relative": .001,
                "code_sha256": {str(f.relative_to(ROOT)): sha(f) for f in code}}
    (a.output/"manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")
    old_summary = json.loads((a.previous/"summary.json").read_text())
    if not old_summary["qa_pass"] or sha(a.previous/"land_cells.csv") != old_summary["output_sha256"]:
        raise ValueError("Previous CMG pilot failed integrity check")
    previous = pd.read_csv(a.previous/"land_cells.csv").set_index(["latitude", "longitude"])
    sd = SD(str(a.modis), SDC.READ)
    dem = xr.open_dataset(a.dem)
    class_rows, comparisons, diagnostics, geometries = [], [], [], []
    tile_ids = {"atacama": "h11v11", "northwest_australia": "h28v11", "central_australia": "h30v11"}
    try:
        for site in json.loads(a.config.read_text())["sites"]:
            start = time.monotonic()
            cell = Cell(site["latitude"], site["longitude"], "center")
            tile = tiles[tile_ids[site["id"]]]
            exact = tile.unmasked_ledger(cell)
            fractions, areas, closure = read_cmg_cell(sd.select("Land_Cover_Type_1_Percent"), cell)
            cmg_raw = (fractions*areas[..., None]).sum(axis=(0, 1))
            cmg_norm = (fractions/fractions.sum(axis=-1, keepdims=True)*areas[..., None]).sum(axis=(0, 1))
            for k in range(17):
                geometries.append({"site": site["id"], "class_id_cmg": k, "class_id_q1": k or 17,
                                   "native_geometric_km2": exact["class_areas_km2"][k],
                                   "cmg_raw_rounded_km2": cmg_raw[k], "cmg_normalized_km2": cmg_norm[k]})
            polys = []
            for source in a.protected:
                protected = _filter_protected_areas(gpd.read_file(source, bbox=cell.bounds))
                if len(protected):
                    polys.append(protected.to_crs("EPSG:4326").geometry)
            union = None if not polys else _geometry_union(gpd.GeoSeries(pd.concat(polys, ignore_index=True), crs="EPSG:4326"))
            slope_grid = dem_slopes(dem["elevation"], cell)
            for resolution in a.resolutions:
                result = integrate(cell, tile, fractions, slope_grid, union, resolution)
                for source_index, source_name in enumerate(("native_q1", "cmg_uniform")):
                    for mi, mask in enumerate(MASKS):
                        for k in range(17):
                            class_rows.append({"site": site["id"], "samples_per_degree": resolution, "source": source_name,
                                               "mask": mask, "class_id_cmg": k, "area_km2": result["classes"][source_index, mi, k]})
                for technology, factors in FACTORS.items():
                    weighted = result["classes"] @ factors
                    row = {"site": site["id"], "latitude": cell.latitude, "longitude": cell.longitude,
                           "samples_per_degree": resolution, "technology": technology,
                           "cell_area_km2": cell.area_km2, "native_geometric_raw_km2": float(exact["class_areas_km2"]@factors),
                           "cmg_raw_rounded_km2": float(cmg_raw@factors), "cmg_raw_normalized_km2": float(cmg_norm@factors)}
                    for i, source_name in enumerate(("native", "cmg")):
                        for mi, mask in enumerate(MASKS):
                            row[f"{source_name}_{mask}_suitable_km2"] = weighted[i, mi]
                        row[f"{source_name}_shipping_km2"] = weighted[i, 3]*.02
                    row["previous_cmg_shipping_km2"] = float(previous.loc[(cell.latitude, cell.longitude), f"{technology}_shipping_estimate_km2"])
                    row["native_vs_common_grid_cmg_percent"] = 100*(weighted[0, 3]/weighted[1, 3]-1)
                    row["native_vs_previous_cmg_percent"] = 100*(row["native_shipping_km2"]/row["previous_cmg_shipping_km2"]-1)
                    comparisons.append(row)
                diagnostics.append({"site": site["id"], "samples_per_degree": resolution,
                                    "cell_area_error_km2": exact["cell_area_error_km2"],
                                    "class_rounding_max": float(np.abs(closure).max()),
                                    "native_geometric_qc_areas_km2": exact["qc_areas_km2"].tolist(),
                                    "native_geometric_lw_mismatch_km2": exact["land_water_mismatch_km2"],
                                    "quadrature_qc_areas_km2": result["qc_areas_km2"].tolist(),
                                    "quadrature_lw_mismatch_km2": result["land_water_mismatch_km2"],
                                    "physical_mask_areas_km2": dict(zip(MASKS, result["mask_areas_km2"].tolist()))})
                pd.DataFrame(comparisons).to_csv(a.output/"comparison.partial.csv", index=False)
                print(f"{site['id']} {resolution}/degree complete in {time.monotonic()-start:.1f}s", flush=True)
        frame = pd.DataFrame(comparisons)
        low = frame[frame.samples_per_degree == a.resolutions[-2]].set_index(["site", "technology"])
        high = frame[frame.samples_per_degree == a.resolutions[-1]].set_index(["site", "technology"])
        convergence = high[["native_shipping_km2", "cmg_shipping_km2"]]/low[["native_shipping_km2", "cmg_shipping_km2"]]-1
        unmasked_error = high.native_raw_suitable_km2/high.native_geometric_raw_km2-1
        passed = bool((convergence.abs().to_numpy() <= .001).all() and (unmasked_error.abs() <= .001).all())
        pd.DataFrame(class_rows).to_csv(a.output/"joint_classes.csv", index=False)
        pd.DataFrame(geometries).to_csv(a.output/"unmasked_classes.csv", index=False)
        frame.to_csv(a.output/"comparison.csv", index=False)
        convergence.reset_index().to_csv(a.output/"convergence.csv", index=False)
        (a.output/"diagnostics.json").write_text(json.dumps(diagnostics, indent=2)+"\n")
        summary = {"qa_pass": passed, "scope": "three_centered_cells_only_not_global_not_exact_historical_replay",
                   "completed_sites": len(high.index.get_level_values("site").unique()),
                   "max_joint_quadrature_relative_change": float(convergence.abs().to_numpy().max()),
                   "max_unmasked_suitability_quadrature_relative_error": float(unmasked_error.abs().max()),
                   "manifest_sha256": sha(a.output/"manifest.json"),
                   "output_sha256": {f.name: sha(f) for f in a.output.iterdir() if f.name in
                                     {"comparison.csv", "joint_classes.csv", "unmasked_classes.csv", "convergence.csv", "diagnostics.json"}}}
        (a.output/"summary.json").write_text(json.dumps(summary, indent=2)+"\n")
        print(json.dumps(summary, indent=2), flush=True)
        if not passed:
            raise ValueError("Quadrature needs refinement; pilot not accepted")
    finally:
        sd.end()
        dem.close()


if __name__ == "__main__":
    main()
