#!/usr/bin/env python3
"""Export accepted three-cell native land into separately labelled stacks."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from reconciliation.land.pilot_joint_masks import sha
from arc.validate_land_campaign_input import validate_land_campaign_input


def main():
    p = argparse.ArgumentParser()
    for name in ("audit", "base", "config", "replication", "revised"):
        p.add_argument("--"+name, required=True, type=Path)
    a = p.parse_args()
    if a.replication.exists() or a.revised.exists():
        raise FileExistsError("Refusing to overwrite either land stack")
    summary = json.loads((a.audit/"summary.json").read_text())
    if not summary["qa_pass"] or sha(a.audit/"manifest.json") != summary["manifest_sha256"]:
        raise ValueError("Native audit failed")
    for name, digest in summary["output_sha256"].items():
        if sha(a.audit/name) != digest:
            raise ValueError(f"Changed native output: {name}")
    if sha(a.base) != json.loads(a.config.read_text())["land_sha256"]:
        raise ValueError("Base schema/input differs from controlled experiment")
    manifest = json.loads((a.audit/"manifest.json").read_text())
    if sha(a.config) != manifest["config_sha256"]:
        raise ValueError("Site config changed")
    comparison = pd.read_csv(a.audit/"comparison.csv")
    comparison = comparison[comparison.samples_per_degree == comparison.samples_per_degree.max()]
    base = pd.read_csv(a.base).set_index(["latitude", "longitude"], verify_integrity=True)
    unmasked = pd.read_csv(a.audit/"unmasked_classes.csv")
    output = []
    for site, group in comparison.groupby("site", sort=False):
        row = group.iloc[0]
        record = base.loc[(row.latitude, row.longitude)].to_dict()
        record.update(latitude=row.latitude, longitude=row.longitude)
        areas = group.set_index("technology").native_shipping_km2
        record.update(wind_onshore_area_km2=areas.wind, wind_area_km2=areas.wind,
                      solar_area_km2=areas.solar, renewable_union_area_km2=areas.nested_union,
                      renewable_union_area_km2_classwise_nested_v1=areas.nested_union,
                      renewable_union_availability=areas.nested_union/row.cell_area_km2,
                      renewable_union_availability_classwise_nested_v1=areas.nested_union/row.cell_area_km2)
        record["max_power_solar_mw"] = areas.solar*record["solar_density_mw_per_km2"]
        record["max_power_wind_mw"] = areas.wind*record["wind_density_mw_per_km2"]
        record["max_capacity_mw"] = record["max_power_solar_mw"]+record["max_power_wind_mw"]
        water = float(unmasked[(unmasked.site == site) & (unmasked.class_id_cmg == 0)].native_geometric_km2.iloc[0])
        record["onshore_land_pct"] = 100*(1-water/row.cell_area_km2)
        record["spatial_method"] = "native_q1_joint_masks_converged_quadrature_v1"
        record["source_substituted"] = True
        record["pilot_only"] = True
        output.append(record)
    frame = pd.DataFrame(output)[list(pd.read_csv(a.base, nrows=0).columns)]
    for destination, stack in ((a.replication, "replication_native_source_substituted_pilot"),
                               (a.revised, "revised_native_common_geography_fixed_candidate")):
        destination.mkdir(parents=True, exist_ok=False)
        frame["stack"] = stack
        frame.to_csv(destination/"max_capacities.csv", index=False)
        qa = validate_land_campaign_input(destination/"max_capacities.csv", expected_land_fraction=.02)
        qa.update(stack=stack, pilot_only=True, rows=len(frame), source_audit_summary_sha256=sha(a.audit/"summary.json"),
                  source_audit=str(a.audit.resolve()), exporter_sha256=sha(Path(__file__)),
                  warning="Same native geography, paper factors and fixed-PV density in both controlled pilot stacks; not an exact historic replay or a global production replacement. Tracking footprint remains a separately configured assumption.")
        (destination/"qa.json").write_text(json.dumps(qa, indent=2)+"\n")
        print(f"Validated {len(frame)} cells: {destination}")


if __name__ == "__main__":
    main()
