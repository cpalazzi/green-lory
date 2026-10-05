#!/usr/bin/env python3
"""Preserve the native pilot's source, tests and export configuration."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from reconciliation.land import stage_pilot


if __name__ == "__main__":
    stage_pilot.FILES = sorted(set(stage_pilot.FILES) | {
        "arc/validate_land_campaign_input.py",
        "reconciliation/land/native_modis.py",
        "reconciliation/land/run_native_pilot.py",
        "reconciliation/land/export_native_pilot.py",
        "reconciliation/land/stage_native_pilot.py",
        "reconciliation/land/supply_curve/config_v1.json",
        "tests/test_native_modis.py", "tests/test_native_pilot.py",
    })
    stage_pilot.main()
