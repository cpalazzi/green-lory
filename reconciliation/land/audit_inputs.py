#!/usr/bin/env python3
"""Trace current and archived land inputs without modifying either stack."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from reconciliation.land.core import Cell, WIND, SOLAR, paper_pv_density, read_cmg_cell
from model.land_processing import _solar_density_raw


def sha(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(2**20),b""): h.update(chunk)
    return h.hexdigest()


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--modis",type=Path,required=True)
    p.add_argument("--land",type=Path,required=True)
    p.add_argument("--historical",type=Path,required=True)
    p.add_argument("--rep-surface",type=Path,required=True)
    p.add_argument("--regressed-land",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    args=p.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    old=pd.read_csv(args.historical).drop_duplicates(["Latitude","Longitude"],keep="last")
    old=old.rename(columns={"Latitude":"latitude","Longitude":"longitude"})
    land=pd.read_csv(args.land).set_index(["latitude","longitude"])
    rep=pd.read_csv(args.rep_surface)
    rep=rep.rename(columns={"Latitude":"latitude","Longitude":"longitude"})
    # Header mapping is explicit; fail if the published runner schema changes.
    rep=rep.set_index(["latitude","longitude"])
    historic=old.set_index(["latitude","longitude"])
    # Generous physical upper bound: all 2% of a cell is available, all PV,
    # paper 9 km2/GW equatorial density at every latitude, 100% annual capacity
    # factor, only 5 MWh/t electricity. This is intentionally not a prediction.
    lat=old.latitude.to_numpy()
    area=np.array([Cell(float(x),0).area_km2 for x in lat])
    old["generous_capacity_upper_bound_Mtpa"]=area*.02*(1000/9)*8760/5/1e6
    old["exceeds_generous_upper_bound"]=old.Max_capacity>old.generous_capacity_upper_bound_Mtpa
    old.sort_values("Max_capacity",ascending=False).to_csv(args.output/"historical_capacity_bounds.csv",index=False)
    old[old.exceeds_generous_upper_bound].to_csv(args.output/"historical_impossible_under_stated_heuristic.csv",index=False)

    coords={(-23.,-69.),(-23.,117.),(-21.,135.),(-29.,137.),(-33.,151.),(-22.,114.)}
    coords.update(map(tuple,old.nlargest(12,"Max_capacity")[["latitude","longitude"]].to_numpy()))
    # High-capacity Australian sites, and a broader deterministic geographic sample.
    aus=old[old.country=="Australia"].sort_values(["latitude","longitude"])
    coords.update(map(tuple,aus.nlargest(10,"Max_capacity")[["latitude","longitude"]].to_numpy()))
    coords.update(map(tuple,aus.iloc[::max(1,len(aus)//12)][["latitude","longitude"]].to_numpy()))
    from pyhdf.SD import SD, SDC
    sd=SD(str(args.modis),SDC.READ); sds=sd.select("Land_Cover_Type_1_Percent")
    metadata={"sds_info":sds.info(),"sds_attributes":sds.attributes(),
              "grid_metadata":str(sd.attributes().get("StructMetadata.0", "")).split(chr(0))[0]}
    rows=[]
    try:
        for lat,lon in sorted(coords):
            if (lat,lon) not in land.index: continue
            current=land.loc[(lat,lon)]
            row={"latitude":lat,"longitude":lon,"country":historic.loc[(lat,lon),"country"] if (lat,lon) in historic.index else "",
                 "historical_capacity_Mtpa":historic.loc[(lat,lon),"Max_capacity"] if (lat,lon) in historic.index else None,
                 "current_solar_area_km2":float(current.solar_area_km2),
                 "current_wind_area_km2":float(current.wind_onshore_area_km2),
                 "current_land_exclusion_factor":float(current.land_exclusion_factor),
                 "current_physical_exclusion_factor":float(current.land_exclusion_factor/current.land_competition_fraction),
                 "current_onshore_fraction":float(current.onshore_land_pct/100),
                 "current_solar_density_MW_km2":float(current.solar_density_mw_per_km2),
                 "paper_pv_density_MW_km2":float(paper_pv_density(lat))}
            for anchor in ("center","southwest"):
                f,a,closure=read_cmg_cell(sds,Cell(lat,lon,anchor))
                row[f"{anchor}_raw_solar_area_km2"]=float(((f@SOLAR)*a).sum())
                row[f"{anchor}_raw_wind_area_km2"]=float(((f@WIND)*a).sum())
                row[f"{anchor}_water_fraction"]=float((f[...,0]*a).sum()/a.sum())
                row[f"{anchor}_rounding_residual_max"]=float(np.abs(closure).max())
            # Current HDF reader takes [lat-1,lat] and [lon,lon+1], while
            # protected/slope/area are calculated on [lat,lat+1]. Verify data.
            f,a,_=read_cmg_cell(sds,Cell(lat-1,lon,"southwest"))
            row["legacy_north_label_raw_onshore_fraction"]=float(1-f[...,0].mean())
            row["legacy_north_label_raw_solar_fraction"]=float((f@SOLAR).mean())
            row["onshore_legacy_reproduction_error"]=row["legacy_north_label_raw_onshore_fraction"]-row["current_onshore_fraction"]
            # The exported factor already includes land_competition_fraction.
            expected=float((f@SOLAR).mean()*current.area*current.land_exclusion_factor)
            row["solar_legacy_reproduction_error_km2"]=expected-float(current.solar_area_km2)
            if (lat,lon) in rep.index:
                r=rep.loc[(lat,lon)]
                # Read exact installed powers from September output.
                wind=float(r["wind_mw"]); fixed=float(r["solar_mw"]); tracking=float(r["solar_tracking_mw"])
                row.update(rep_wind_MW=wind,rep_fixed_MW=fixed,rep_tracking_MW=tracking)
                footprint=wind/5+(fixed+tracking)/float(paper_pv_density(lat)) if paper_pv_density(lat)>0 else np.nan
                row["rep_design_paper_fixed_footprint_km2"]=footprint
                row["rep_design_required_land_at_historical_capacity_km2"]=footprint*row["historical_capacity_Mtpa"] if row["historical_capacity_Mtpa"] is not None else None
            rows.append(row)
    finally: sd.end()
    diagnostic=pd.DataFrame(rows)
    diagnostic.to_csv(args.output/"cell_diagnostics.csv",index=False)
    diagnostic[["latitude","longitude","country"]].to_csv(args.output/"pilot_cells.csv",index=False)
    density=pd.DataFrame({"latitude":np.arange(0,76)})
    density["current_fixed_MW_km2"]=_solar_density_raw(density.latitude.to_numpy())
    density["current_tracking_MW_km2"]=density.current_fixed_MW_km2/2
    density["paper_equation6_fixed_MW_km2"]=paper_pv_density(density.latitude)
    density.to_csv(args.output/"pv_density_comparison.csv",index=False)
    regression=pd.read_csv(args.regressed_land)
    summary={"inputs":{k:{"path":str(v.resolve()),"sha256":sha(v)} for k,v in vars(args).items() if k!="output"},
             "modis_metadata":metadata,"diagnostic_cells":len(rows),
             "historical_unique_cells":len(old),"historical_max_capacity_Mtpa":float(old.Max_capacity.max()),
             "historical_exceed_generous_upper_bound":int(old.exceeds_generous_upper_bound.sum()),
             "upper_bound_assumptions":"2% entire centered cell; 111.111 MW/km2 PV; 100% capacity factor; 5 MWh/t NH3; all land/classes/exclusions ignored; no co-located extra wind",
             "onshore_legacy_reproduction_max_error":float(diagnostic.onshore_legacy_reproduction_error.abs().max()),
             "solar_legacy_reproduction_max_error_km2":float(diagnostic.solar_legacy_reproduction_error_km2.abs().max()),
             "regressed_land_rows":len(regression),
             "regressed_land_negative_capacity_rows":int((regression.Max_capacity<0).sum()),
             "warning":"The later regression notebook is not established as the provenance of the original paper supplier capacities."}
    (args.output/"summary.json").write_text(json.dumps(summary,indent=2,allow_nan=False)+"\n")
    print(json.dumps({k:v for k,v in summary.items() if k not in {"inputs","modis_metadata"}},indent=2))


if __name__=="__main__": main()
