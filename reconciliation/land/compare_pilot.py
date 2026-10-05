#!/usr/bin/env python3
"""Verify ARC pilot artifacts, compare land and hold plant design fixed."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from reconciliation.land.audit_inputs import sha
from reconciliation.land.pilot_joint_masks import validate_cell
from arc.validate_land_campaign_input import validate_land_campaign_input


def load_verified(path):
    summary=json.loads((path/"summary.json").read_text())
    for file,key in (("land_cells.csv","output_sha256"),("manifest.json","manifest_sha256")):
        if sha(path/file)!=summary[key]:raise ValueError(f"Hash failure: {path/file}")
    d=pd.read_csv(path/"land_cells.csv")
    if not summary["qa_pass"] or len(d)!=summary["completed_cells"]:raise ValueError("Incomplete pilot")
    if d.duplicated(["latitude","longitude"]).any():raise ValueError("Duplicate pilot coordinates")
    for _,row in d.iterrows():validate_cell(row)
    return d


def export_land(d,diagnostics,output,stack):
    """Compatible 40-cell pilot input; not a global or empirically calibrated map."""
    z=d[["latitude","longitude","country","cell_anchor"]].copy()
    z["area"]=d.cell_area_km2;z["land_competition_fraction"]=.02
    z["wind_onshore_area_km2"]=d.wind_shipping_estimate_km2
    z["wind_area_km2"]=z.wind_onshore_area_km2
    z["solar_area_km2"]=d.solar_shipping_estimate_km2
    z["wind_density_mw_per_km2"]=5.
    z["solar_density_mw_per_km2"]=d.solar_fixed_density_MW_km2
    z["solar_density_method"]="explicit_fixed_pv_density_v1"
    z["renewable_union_area_km2_classwise_nested_v1"]=d.nested_union_shipping_estimate_km2
    z["renewable_union_area_km2"]=d.nested_union_shipping_estimate_km2
    z["renewable_union_availability_classwise_nested_v1"]=d.nested_union_shipping_estimate_km2/z.area
    z["renewable_union_availability"]=z.renewable_union_availability_classwise_nested_v1
    z["renewable_union_method"]="classwise_nested_overlap_lower_bound"
    z["renewable_union_method_version"]="v1"
    z["max_power_solar_mw"]=z.solar_area_km2*z.solar_density_mw_per_km2
    z["max_power_wind_mw"]=z.wind_onshore_area_km2*5
    z["max_capacity_mw"]=z.max_power_solar_mw+z.max_power_wind_mw
    z["onshore_land_pct"]=[100*(1-diagnostics.loc[(r.latitude,r.longitude),"center_water_fraction"]) for _,r in d.iterrows()]
    z["stack"]=stack
    z["spatial_method"]="cmg_conditional_uniform_joint_dem_masks"
    z["source_substituted"]=True
    z["pilot_only"]=True
    if output.exists():
        if (output/"max_capacities.csv").read_text()!=z.to_csv(index=False):
            raise ValueError(f"Refusing to change existing land stack: {output}")
        validate_land_campaign_input(output/"max_capacities.csv",expected_land_fraction=.02)
        return
    output.mkdir(parents=True,exist_ok=False)
    z.to_csv(output/"max_capacities.csv",index=False)
    qa=validate_land_campaign_input(output/"max_capacities.csv",expected_land_fraction=.02)
    qa.update(stack=stack,pilot_only=True,rows=len(z),warning="Common corrected geography and paper PV packing for controlled comparison; revised engineering assumptions not yet promoted; tracking density must be explicitly configured in the plant YAML.")
    (output/"qa.json").write_text(json.dumps(qa,indent=2)+"\n")
    d.to_csv(output/"land_area_diagnostics_with_bounds.csv",index=False)


def main():
    p=argparse.ArgumentParser()
    for name in ("center","southwest","diagnostics","rep","central","campaign","output"):
        p.add_argument("--"+name,type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    frames={anchor:load_verified(getattr(a,anchor)) for anchor in ("center","southwest")}
    diagnostic=pd.read_csv(a.diagnostics).set_index(["latitude","longitude"])
    expected=set(diagnostic.index)
    for anchor,d in frames.items():
        if set(map(tuple,d[["latitude","longitude"]].to_numpy()))!=expected:raise ValueError("Pilot cell coverage mismatch")
    center=frames["center"]
    land=diagnostic.copy()
    for anchor,d in frames.items():
        land=land.join(d.set_index(["latitude","longitude"]).drop(columns=["country","cell_anchor"]).add_prefix(anchor+"_jointmask_"))
    land["center_current_solar_ratio"]=land.center_jointmask_solar_shipping_estimate_km2/land.current_solar_area_km2.replace(0,np.nan)
    land["southwest_current_solar_ratio"]=land.southwest_jointmask_solar_shipping_estimate_km2/land.current_solar_area_km2.replace(0,np.nan)
    land.reset_index().to_csv(a.output/"land_changes.csv",index=False)
    rows=[]
    for label in ("rep","central"):
        surface=pd.read_csv(getattr(a,label)).set_index(["latitude","longitude"])
        for anchor,d in frames.items():
            for _,site in d.iterrows():
                coord=(site.latitude,site.longitude)
                if coord not in surface.index:continue
                r=surface.loc[coord]
                if not float(r.annual_ammonia_production_t)>0:continue
                for bound in ("lower","estimate","upper"):
                    for packing in (1.,1.25,1.5,2.):
                        density=float(site.solar_fixed_density_MW_km2)
                        wind=max(0,float(r.wind_mw))/5
                        solar=(max(0,float(r.solar_mw))+packing*max(0,float(r.solar_tracking_mw)))/density
                        def limit(area,used):return area/used if used>1e-9 else np.inf
                        union=limit(site[f"nested_union_shipping_{bound}_km2"],wind+solar)
                        shared=min(union,limit(site[f"wind_shipping_{bound}_km2"],wind),limit(site[f"solar_shipping_{bound}_km2"],solar))
                        prod=float(r.gridless_ammonia_production_t)/1e6
                        rows.append({"latitude":coord[0],"longitude":coord[1],"country":site.country,"design":label,"anchor":anchor,"subpixel_bound":bound,"tracking_footprint_ratio":packing,
                                     "historical_Mtpa":diagnostic.loc[coord,"historical_capacity_Mtpa"],
                                     "current_Mtpa":float(r.paper_scaled_max_gridless_onshore_ammonia_capacity_mtpa),
                                     "paper_union_Mtpa":union*prod,"technology_shared_Mtpa":shared*prod,
                                     "wind_footprint_km2":wind,"pv_footprint_km2":solar})
    counter=pd.DataFrame(rows);counter.to_csv(a.output/"frozen_design_counterfactuals.csv",index=False)
    summaries={}
    for anchor in ("center","southwest"):
        ratio=land[anchor+"_current_solar_ratio"].dropna()
        summaries[anchor]={"positive_current_cells":len(ratio),"median_solar_area_ratio":float(ratio.median()),"min_solar_area_ratio":float(ratio.min()),"max_solar_area_ratio":float(ratio.max()),
                           "current_zero_new_positive":int(((land.current_solar_area_km2==0)&(land[f"{anchor}_jointmask_solar_shipping_estimate_km2"]>0)).sum())}
    report={"pilot_cells":len(center),"all_cells_revalidated":True,"land_comparison":summaries,
            "input_hashes":{key:sha(getattr(a,key)) for key in ("diagnostics","rep","central")},
            "pilot_manifests":{key:sha(getattr(a,key)/"manifest.json") for key in ("center","southwest")},
            "scope":"Purposefully selected diagnostic sample, not representative global statistics. Capacity comparisons hold September plants fixed; ratio=1 retains tracking generation with fixed packing, not a fixed-PV reoptimization."}
    (a.output/"summary.json").write_text(json.dumps(report,indent=2,allow_nan=False)+"\n")
    export_land(center,diagnostic,a.campaign/"replication/cmg-paper-method-pilot-v1","replication_source_substituted")
    export_land(center,diagnostic,a.campaign/"revised/common-geography-fixed-pilot-v1","revised_fixed_candidate_controlled_land")
    print(json.dumps(report,indent=2),flush=True)


if __name__=="__main__":main()
