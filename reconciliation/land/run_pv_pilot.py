#!/usr/bin/env python3
"""Three hourly full-year fixed-PV plant solves on corrected, common land."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from model import location_tools as lt
from model.run_global import run_global, _load_tech_inputs
from arc.release_source_inventory import verify_inventory
from arc.validate_land_campaign_input import validate_land_campaign_input
from reconciliation.land.audit_inputs import sha
from reconciliation.land.stage_pv_pilot import TECH_YAML, tech_yaml_dependencies


def main():
    p=argparse.ArgumentParser()
    for key in ("weather","land","output"):p.add_argument("--"+key,type=Path,required=True)
    p.add_argument("--preflight-only", action="store_true", help="Validate release/configuration without loading weather or creating outputs")
    a=p.parse_args()
    release=verify_inventory(ROOT,ROOT/"source_inventory.json")
    land_qa=validate_land_campaign_input(a.land,expected_land_fraction=.02)
    cells=ROOT/"inputs/lory_reconciliation_diagnostic_cells.csv"
    sites=pd.read_csv(cells);coords=list(zip(sites.lat.astype(float),sites.lon.astype(float)))
    plant=ROOT/"basic_ammonia_plant_2050_way"
    generators=pd.read_csv(plant/"generators.csv").set_index("name")
    if bool(generators.loc["solar_tracking","p_nom_extendable"]) or float(generators.loc["solar_tracking","p_nom"])!=0:
        raise ValueError("Fixed-PV bundle permits tracking")
    tech=ROOT/TECH_YAML
    tech_files=tech_yaml_dependencies(tech, ROOT)
    if not _load_tech_inputs(tech):
        raise ValueError("No technology inputs loaded from staged release")
    overrides=ROOT/"inputs/amelired_interest_inputs_2050.csv"
    files=[a.land,cells,*tech_files,overrides,ROOT/"data/countries.geojson",ROOT/"data/model_bathymetry.nc"]+sorted(plant.glob("*.csv"))
    manifest={"source_release":release,"land_qa":land_qa,"scenario":"hourly_central_fixed_pv_corrected_common_land",
              "comparison":"September hourly central technology/cost/ramp/dispatch settings; only PV option and explicitly versioned land/density inputs differ",
              "inputs":[{"path":str(f.resolve()),"sha256":sha(f)} for f in files],"weather_subsets":[]}
    if a.output.exists():
        raise FileExistsError(f"Output already exists: {a.output}")
    if a.preflight_only:
        print(json.dumps({"preflight_pass": True, "source_release": release,
                          "tech_yaml_dependencies": [str(f.relative_to(ROOT)) for f in tech_files],
                          "land_qa": land_qa}, indent=2), flush=True)
        return
    a.output.mkdir(parents=True,exist_ok=False)
    # Materialize only the exact coordinate cross-product used by this pilot.
    # This avoids loading the full global weather cube for three plant solves.
    weather_out=a.output/"weather_used";weather_out.mkdir()
    dataset=lt.all_locations(str(a.weather),cache_resources=False)
    selected={}
    try:
        for key,filename in (("solar","Solar.nc"),("solar_tracking","SolarTracking.nc"),("wind","WindPowers.nc")):
            original=dataset.resources[key]
            subset=original.sel(latitude=sorted(set(sites.lat)),longitude=sorted(set(sites.lon))).load()
            if subset.sizes["time"]!=8760 or not np.isfinite(subset.values).all():raise ValueError("Invalid full-year weather subset")
            selected[key]=subset.values
            target=weather_out/filename
            subset.encoding={}  # Original global chunk sizes may exceed the small subset.
            subset.to_dataset(name=original.name).to_netcdf(target)
            manifest["weather_subsets"].append({"resource":key,"path":str(target.resolve()),"sha256":sha(target),"shape":list(subset.shape),"dimensions":list(subset.dims)})
        if np.allclose(selected["solar"],selected["solar_tracking"]):raise ValueError("Fixed/tracking profiles unexpectedly identical")
    finally:dataset.close()
    manifest["weather_source_files"]=[{"path":str(f.resolve()),"size_bytes":f.stat().st_size,"mtime_ns":f.stat().st_mtime_ns} for f in sorted(a.weather.glob("*.nc"))]
    manifest["weather_identity_scope"]="Exact extracted NetCDF inputs hashed; global parent weather files recorded by path/size/mtime, not claimed content-hashed"
    (a.output/"manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
    output=a.output/"run_global.csv"
    d=run_global(locations=coords,weather_dir=weather_out,land_csv=a.land,override_csv=overrides,tech_yaml=tech,
                 plant_dir=plant,time_step=1.,output_csv=output,quiet=True,threads_per_worker=4,num_workers=3,
                 fail_fast=True,ensure_feasibility=False,land_constraint="after_solve",capacity_rule="scaled_reference_design",
                 land_allocation="exclusive",allow_conservative_union_fallback=False,
                 temporal_accounting_mode="snapshot_weighted",ramp_limit_basis="per_hour",include_site_costs=True)
    if len(d)!=3 or set(zip(d.latitude,d.longitude))!=set(coords):raise ValueError("Missing fixed-PV results")
    np.testing.assert_allclose(d.annual_ammonia_production_t,1e6,rtol=0,atol=1)
    np.testing.assert_allclose(d.solar_tracking_mw,0,atol=1e-6,rtol=0)
    if not (d.solar_mw>0).all():raise ValueError("No fixed solar capacity")
    if not (d.grid_energy_mwh.abs()<=d.grid_energy_tolerance_mwh).all():raise ValueError("Grid use in fixed-PV result")
    if (d.accumulated_penalty_mwh.abs()>1).any():raise ValueError("Feasibility slack in fixed-PV result")
    cols=["latitude","longitude","lcoa_eur_per_t","solar_mw","solar_tracking_mw","wind_mw","solar_density_mw_per_km2","scaled_design_max_gridless_onshore_ammonia_capacity_mtpa"]
    summary={"qa_pass":True,"rows":len(d),"output_sha256":sha(output),"results":d[cols].to_dict(orient="records")}
    (a.output/"summary.json").write_text(json.dumps(summary,indent=2,allow_nan=False)+"\n")
    print(json.dumps(summary,indent=2),flush=True)


if __name__=="__main__":main()
