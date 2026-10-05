#!/usr/bin/env python3
"""Pilot aligned CMG classes with joint DEM-slope/protected raster masks.

Not an exact reproduction of native 500 m class masks: CMG subpixel class
locations remain unknown. Output estimates and sharp bounds expose this.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
import numpy as np
import pandas as pd
import geopandas as gpd
import shapely
import xarray as xr
from pyhdf.SD import SD, SDC

ROOT=Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from reconciliation.land.core import Cell, WIND, SOLAR, RADIUS_KM, read_cmg_cell, masked_suitability, paper_pv_density
from model.land_processing import _filter_protected_areas, _geometry_union


def sha(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda:f.read(2**20),b""): h.update(block)
    return h.hexdigest()


def masked_dem_fraction(elevation, cell, protected, max_slope=15.):
    west,south,east,north=cell.bounds
    if west < -180 or east > 180:
        raise ValueError("Pilot DEM window does not yet support dateline cells")
    dy=abs(float(elevation.lat[1]-elevation.lat[0]))
    dx=abs(float(elevation.lon[1]-elevation.lon[0]))
    def coord_slice(coord,lo,hi):
        return slice(lo,hi) if float(coord[0])<float(coord[-1]) else slice(hi,lo)
    band=elevation.sel(lat=coord_slice(elevation.lat,south-2*dy,north+2*dy),
                       lon=coord_slice(elevation.lon,west-2*dx,east+2*dx)).transpose("lat","lon")
    lat=band.lat.values.astype(float);lon=band.lon.values.astype(float)
    z=band.values.astype(float)
    if not np.isfinite(z).all(): raise ValueError("Missing DEM data in pilot window")
    gy=np.gradient(z,axis=0)/(RADIUS_KM*1000*np.deg2rad(dy))
    gx=np.gradient(z,axis=1)/(RADIUS_KM*1000*np.deg2rad(dx)*np.cos(np.deg2rad(lat))[:,None])
    slope=np.rad2deg(np.arctan(np.hypot(gx,gy)))
    yy=(lat>=south)&(lat<north);xx=(lon>=west)&(lon<east)
    lat=lat[yy];lon=lon[xx];slope=slope[np.ix_(yy,xx)];z=z[np.ix_(yy,xx)]
    xs,ys=np.meshgrid(lon,lat)
    is_protected=np.zeros(z.shape,dtype=bool) if protected is None else shapely.intersects_xy(protected,xs,ys)
    slope_ok=slope<=max_slope
    # MODIS, not positive DEM elevation, identifies land. Below-sea-level land
    # is not excluded merely by sign of elevation.
    surviving=slope_ok & ~is_protected
    row=np.floor((north-ys)/.05).astype(int)
    col=np.floor((xs-west)/.05).astype(int)
    if (row<0).any() or (row>=20).any() or (col<0).any() or (col>=20).any():
        raise ValueError("DEM/CMG alignment failure")
    bins=row*20+col;weights=np.cos(np.deg2rad(ys))
    denom=np.bincount(bins.ravel(),weights=weights.ravel(),minlength=400)
    if (denom==0).any(): raise ValueError("DEM does not cover every CMG pixel")
    def fraction(mask):
        return (np.bincount(bins.ravel(),weights=(weights*mask).ravel(),minlength=400)/denom).reshape(20,20)
    return fraction(surviving),fraction(slope_ok),fraction(is_protected),fraction(z<0)


def validate_cell(row):
    """Every accepted result must satisfy explicit geometric conservation."""
    area=row["cell_area_km2"]
    for name in ("solar","wind","nested_union"):
        raw=row[f"{name}_raw_suitable_km2"]
        slope=row[f"{name}_after_slope_km2"]
        protected=row[f"{name}_after_protected_km2"]
        low=row[f"{name}_joint_lower_km2"]
        estimate=row[f"{name}_joint_estimate_km2"]
        high=row[f"{name}_joint_upper_km2"]
        if not np.isfinite([raw,slope,protected,low,estimate,high]).all():
            raise ValueError("Non-finite land result")
        tol=1e-7
        if not (-tol <= low <= estimate+tol and estimate <= high+tol and high <= raw+tol and raw <= area+tol):
            raise ValueError("Land area or subpixel bounds fail conservation")
        if estimate > min(slope,protected)+tol or max(slope,protected)>raw+tol:
            raise ValueError("Joint exclusions exceed an individual mask")
        if not np.isclose(row[f"{name}_shipping_estimate_km2"],estimate*.02):
            raise ValueError("Shipping competition factor applied incorrectly")
    if row["nested_union_joint_estimate_km2"]+1e-7 < max(row["solar_joint_estimate_km2"],row["wind_joint_estimate_km2"]):
        raise ValueError("Union is smaller than individual technology area")


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--modis",type=Path,required=True)
    p.add_argument("--dem",type=Path,required=True)
    p.add_argument("--protected",type=Path,nargs="+",required=True)
    p.add_argument("--cells",type=Path,required=True)
    p.add_argument("--anchor",choices=["center","southwest"],required=True)
    p.add_argument("--output",type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    source_files=[a.modis,a.dem,a.cells]
    for shp in a.protected:
        source_files.extend(sorted(shp.parent.glob(shp.stem+".*")))
    sources=[]
    for f in source_files:
        print(f"Hashing {f}",flush=True)
        sources.append({"path":str(f.resolve()),"size_bytes":f.stat().st_size,"sha256":sha(f)})
    manifest={"source_files":sources,"anchor":a.anchor,"source_dates_match_paper":False,
              "source_resolution":"CMG 0.05 degree class fractions; joint exclusions on native DEM centers",
              "land_competition_fraction":.02,"maximum_slope_degrees":15,
              "class_subpixel_rule":"normalized fractions, conditional-uniform estimate with sharp min/max suitability bounds",
              "spatial_protection_rule":"union of designated/inscribed/established non-marine polygons; DEM pixel-center inclusion",
              "excluded_point_only_protected_records":True,
              "density":"9 km2/GW at equator and van de Ven Eq6, fixed PV; wind 200 km2/GW",
              "code_sha256":{str(p.relative_to(ROOT)):sha(p) for p in [Path(__file__),ROOT/"reconciliation/land/core.py",ROOT/"model/land_processing.py"]}}
    (a.output/"manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
    sites=pd.read_csv(a.cells);rows=[]
    sd=SD(str(a.modis),SDC.READ);sds=sd.select("Land_Cover_Type_1_Percent")
    dataset=xr.open_dataset(a.dem)
    try:
        elevation=dataset["elevation"]
        for i,site in sites.iterrows():
            started=time.monotonic();cell=Cell(float(site.latitude),float(site.longitude),a.anchor)
            f,area,residual=read_cmg_cell(sds,cell)
            polygons=[]
            for source in a.protected:
                protected=_filter_protected_areas(gpd.read_file(source,bbox=cell.bounds))
                if len(protected): polygons.append(protected.to_crs("EPSG:4326").geometry)
            union=None if not polygons else _geometry_union(gpd.GeoSeries(pd.concat(polygons,ignore_index=True),crs="EPSG:4326"))
            q,slope_ok,protected,below_sea=masked_dem_fraction(elevation,cell,union)
            row={"latitude":cell.latitude,"longitude":cell.longitude,"country":site.country,"cell_anchor":a.anchor,
                 "cell_area_km2":cell.area_km2,"class_rounding_max":float(np.abs(residual).max()),
                 "land_competition_fraction":.02,"wind_density_MW_km2":5.,"solar_fixed_density_MW_km2":float(paper_pv_density(cell.latitude))}
            for name,fac in (("solar",SOLAR),("wind",WIND),("nested_union",np.maximum(SOLAR,WIND))):
                estimate,low,high=masked_suitability(f,q,fac)
                fn=f/f.sum(axis=-1,keepdims=True)
                raw=fn@fac
                row[f"{name}_raw_suitable_km2"]=float((raw*area).sum())
                row[f"{name}_after_slope_km2"]=float((raw*slope_ok*area).sum())
                row[f"{name}_after_protected_km2"]=float((raw*(1-protected)*area).sum())
                for label,value in (("estimate",estimate),("lower",low),("upper",high)):
                    row[f"{name}_joint_{label}_km2"]=float((value*area).sum())
                    row[f"{name}_shipping_{label}_km2"]=row[f"{name}_joint_{label}_km2"]*.02
                row[f"{name}_below_sea_level_class_overlap_estimate_km2"]=float((raw*below_sea*area).sum())
            validate_cell(row)
            rows.append(row)
            pd.DataFrame(rows).to_csv(a.output/"cells.partial.csv",index=False)
            print(f"Cell {i+1}/{len(sites)} {cell.latitude},{cell.longitude}: solar {row['solar_shipping_estimate_km2']:.3f} km2 ({time.monotonic()-started:.1f}s)",flush=True)
        frame=pd.DataFrame(rows)
        frame.to_csv(a.output/"land_cells.csv",index=False)
        (a.output/"summary.json").write_text(json.dumps({"completed_cells":len(frame),"qa_pass":True,
            "scope":"diagnostic_pilot_not_global","manifest_sha256":sha(a.output/"manifest.json"),
            "output_sha256":sha(a.output/"land_cells.csv")},indent=2)+"\n")
    finally:
        sd.end();dataset.close()


if __name__=="__main__":main()
