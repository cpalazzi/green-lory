#!/usr/bin/env python3
"""Pin source and small inputs for the fixed-PV follow-on diagnostic."""
from pathlib import Path
import argparse
import json
import shutil
import sys
import yaml
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from arc.release_source_inventory import _file_entry,_tree_sha256,verify_inventory

TECH_YAML = Path("inputs/tech_config_ammonia_plant_2050_way_eur_explicit_compressor_dea_tank.yaml")


def tech_yaml_dependencies(path: Path, root: Path) -> list[Path]:
    """Resolve the complete extends chain, restricted to portable release files."""
    root = root.resolve()
    dependencies = []
    current = path.resolve()
    while True:
        if not current.is_relative_to(root):
            raise ValueError(f"Tech YAML dependency outside release root: {current}")
        if current in dependencies:
            raise ValueError(f"Circular tech YAML extends chain: {current}")
        raw = yaml.safe_load(current.read_text()) or {}
        if not isinstance(raw, dict):
            raise ValueError(f"Tech YAML must contain a mapping: {current}")
        dependencies.append(current)
        parent = raw.get("extends")
        if parent is None:
            return dependencies
        if not isinstance(parent, str) or not parent.strip() or Path(parent).is_absolute():
            raise ValueError(f"Tech YAML extends must be a nonempty relative path: {current}")
        current = (current.parent / parent).resolve()


def stage_release(root: Path, output: Path, *, extra_names=(),
                  scope="explicit_fixed_pv_pilot_source_and_inputs") -> dict:
    root = root.resolve()
    # Discover semantic dependencies before creating a potentially incomplete release.
    dependencies = tech_yaml_dependencies(root / TECH_YAML, root)
    names={str(f.relative_to(root)) for pattern in ("model/*.py","basic_ammonia_plant_2050_way/*.csv","reconciliation/land/*.py") for f in root.glob(pattern)}
    names.update(["arc/release_source_inventory.py","arc/validate_land_campaign_input.py","arc/jobs/05_fixed_pv_land_pilot.sh",
                  "inputs/amelired_interest_inputs_2050.csv","inputs/lory_reconciliation_diagnostic_cells.csv"])
    names.update(str(path.relative_to(root)) for path in dependencies)
    for name in extra_names:
        source = (root / name).resolve()
        if not source.is_relative_to(root) or not source.is_file():
            raise ValueError(f"Invalid extra release input: {name}")
        names.add(str(source.relative_to(root)))
    output.mkdir(parents=True,exist_ok=False)
    entries=[]
    for name in sorted(names):
        target=output/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(root/name,target)
        entries.append(_file_entry(output,name))
    # Check the relocated chain, not only the source checkout's chain.
    tech_yaml_dependencies(output / TECH_YAML, output)
    inventory=output/"source_inventory.json"
    inventory.write_text(json.dumps({"schema_version":1,"scope":scope,"entry_count":len(entries),"tree_sha256":_tree_sha256(entries),"entries":entries},indent=2,sort_keys=True)+"\n")
    return verify_inventory(output,inventory)


def main():
    p=argparse.ArgumentParser();p.add_argument("--output",type=Path,required=True)
    a=p.parse_args()
    print(json.dumps(stage_release(ROOT, a.output),indent=2))


if __name__=="__main__":main()
