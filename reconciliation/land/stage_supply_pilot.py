#!/usr/bin/env python3
"""Pin source and complete configuration for the land-constrained pilot."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from reconciliation.land.stage_pv_pilot import stage_release


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    extras = [str(p.relative_to(ROOT)) for p in ROOT.glob("basic_ammonia_plant_2050_way_tracking/*.csv")]
    extras += ["reconciliation/land/supply_curve/config_v1.json",
               "reconciliation/land/supply_curve/fixed_only_1mt_v1.json",
               "reconciliation/land/supply_curve/config_v2.json",
               "reconciliation/land/supply_curve/config_v2_tracking.json",
               "arc/jobs/06_land_supply_pilot.sh", "tests/test_supply_certificate.py",
               "tests/test_land_supply_pilot.py"]
    print(json.dumps(stage_release(ROOT, args.output, extra_names=extras,
        scope="land_constrained_supply_curve_pilot"), indent=2))


if __name__ == "__main__":
    main()
