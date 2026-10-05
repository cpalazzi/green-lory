#!/usr/bin/env python3
"""Global historical-site ablation: corrected MODIS areas, no physical exclusions.

Capacities hold September's replication plant designs fixed. They are not new
LCOA solves, nor upper bounds on every possible plant design.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd
from pyhdf.SD import SD, SDC

ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from reconciliation.land.core import Cell, WIND, SOLAR, read_cmg_cell, paper_pv_density
from reconciliation.land.audit_inputs import sha


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ("modis","historical","rep-surface","output"):
        p.add_argument("--"+name,type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    old=pd.read_csv(a.historical).rename(columns={"Latitude":"latitude","Longitude":"longitude"}).drop_duplicates(["latitude","longitude"],keep="last")
    rep=pd.read_csv(a.rep_surface).set_index(["latitude","longitude"])
    sd=SD(str(a.modis),SDC.READ);sds=sd.select("Land_Cover_Type_1_Percent")
    rows=[]
    try:
        for i,site in old.reset_index(drop=True).iterrows():
            lat,lon=float(site.latitude),float(site.longitude)
            density=float(paper_pv_density(lat))
            row={"latitude":lat,"longitude":lon,"country":site.country,
                 "historical_capacity_Mtpa":site.Max_capacity,"paper_fixed_density_MW_km2":density}
            for anchor in ("center","southwest"):
                f,area,residual=read_cmg_cell(sds,Cell(lat,lon,anchor))
                f=f/f.sum(axis=-1,keepdims=True)
                for name,fac in (("solar",SOLAR),("wind",WIND),("union",np.maximum(SOLAR,WIND))):
                    row[f"{anchor}_{name}_shipping_km2"]=float(((f@fac)*area).sum())*.02
                row[f"{anchor}_cell_area_km2"]=float(area.sum())
            if (lat,lon) in rep.index and density>0:
                r=rep.loc[(lat,lon)]
                wind=max(0,float(r.wind_mw))/5
                pv=max(0,float(r.solar_mw))+max(0,float(r.solar_tracking_mw))
                production=float(r.gridless_ammonia_production_t)/1e6
                row["rep_current_capacity_Mtpa"]=float(r.paper_scaled_max_gridless_onshore_ammonia_capacity_mtpa)
                row["rep_wind_footprint_km2"]=wind
                row["rep_pv_fixed_footprint_km2"]=pv/density
                for anchor in ("center","southwest"):
                    for multiplier in (1.,1.25,1.5,2.):
                        used=wind+pv/density*multiplier
                        capacity=row[f"{anchor}_union_shipping_km2"]/used*production if used>1e-9 else np.nan
                        row[f"{anchor}_packing_{multiplier:g}_no_exclusions_Mtpa"]=capacity
            rows.append(row)
            if (i+1)%1000==0:print(f"{i+1}/{len(old)}",flush=True)
    finally:sd.end()
    d=pd.DataFrame(rows);d.to_csv(a.output/"historical_sites_unmasked.csv",index=False)
    summaries={}
    for name,selected in (("global",d),("Australia",d[d.country=="Australia"])):
        matched=selected.dropna(subset=["center_packing_1_no_exclusions_Mtpa","rep_current_capacity_Mtpa"])
        matched=matched[(matched.rep_current_capacity_Mtpa>0)&(matched.historical_capacity_Mtpa>0)]
        metrics={"all_historical_sites":len(selected),"both_positive_capacity_matched_sites":len(matched)}
        for multiplier in (1.,1.25,1.5,2.):
            column=f"center_packing_{multiplier:g}_no_exclusions_Mtpa"
            metrics[f"packing_{multiplier:g}_historical_exceeds_same_design_no_exclusion_capacity"]=int((matched.historical_capacity_Mtpa>matched[column]).sum())
            metrics[f"packing_{multiplier:g}_median_reconstructed_to_historical_ratio"]=float((matched[column]/matched.historical_capacity_Mtpa).median())
        summaries[name]=metrics
    report={"inputs":{k:{"path":str(v.resolve()),"sha256":sha(v)} for k,v in vars(a).items() if k!="output"},
            "code_sha256":sha(__file__),"summary":summaries,
            "scope":"All archived supplier coordinates; MODIS Table2 and 2% only; no slope or protection; fixed September replication designs, tracking output retained under all footprint counterfactuals; no new fixed-PV plant solves."}
    (a.output/"summary.json").write_text(json.dumps(report,indent=2,allow_nan=False)+"\n")
    print(json.dumps(summaries,indent=2),flush=True)


if __name__=="__main__":main()
