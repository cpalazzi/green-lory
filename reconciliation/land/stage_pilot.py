#!/usr/bin/env python3
"""Create a minimal, immutable pilot source release with a verified inventory."""
from pathlib import Path
import argparse
import json
import shutil
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from arc.release_source_inventory import _file_entry, _tree_sha256, verify_inventory

FILES=[
    "arc/release_source_inventory.py",
    "arc/jobs/04_land_reconstruction_pilot.sh",
    "model/data_paths.py", "model/land_union.py", "model/land_processing.py",
    "reconciliation/land/core.py", "reconciliation/land/pilot_joint_masks.py",
    "reconciliation/land/stage_pilot.py",
    "reconciliation/land/replication/config.json",
    "reconciliation/land/revised/config.json",
    "tests/test_land_reconstruction.py",
]


def main():
    p=argparse.ArgumentParser();p.add_argument("--output",type=Path,required=True)
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    entries=[]
    for name in sorted(FILES):
        target=args.output/name;target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(ROOT/name,target)
        entries.append(_file_entry(args.output,name))
    manifest={"schema_version":1,"scope":"explicit_minimal_land_pilot_source",
              "entry_count":len(entries),"tree_sha256":_tree_sha256(entries),"entries":entries}
    inventory=args.output/"source_inventory.json"
    inventory.write_text(json.dumps(manifest,indent=2,sort_keys=True)+"\n")
    print(json.dumps(verify_inventory(args.output,inventory),indent=2))


if __name__=="__main__":main()
