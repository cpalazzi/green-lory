#!/usr/bin/env python3
"""Validate a fetched, possibly incomplete snapshot without accepting a full run."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from reconciliation.land.run_supply_pilot import validate_result, sha


def main():
    p = argparse.ArgumentParser()
    for name in ("run", "land", "output"):
        p.add_argument("--"+name, type=Path, required=True)
    a = p.parse_args()
    if a.output.exists():
        raise FileExistsError(a.output)
    manifest = json.loads((a.run/"manifest.json").read_text())
    if sha(a.land) != manifest["land_qa"]["file_sha256"]:
        raise ValueError("Land input changed")
    experiment = manifest["experiment"]
    if experiment["land_allocation"] not in {"exclusive", "colocated"} or experiment["tracking_footprint_ratio"] <= 0:
        raise ValueError("Unexpected land allocation")
    land = pd.read_csv(a.land).set_index(["latitude", "longitude"], verify_integrity=True)
    rows, missing = [], []
    for site in experiment["sites"]:
        for constrained, quantity in [(False, 1.)]+[(True, q) for q in site["quantities_Mtpa"]]:
            key = "q_"+format(quantity, ".12g").replace(".", "p") if constrained else "control_unconstrained"
            folder = a.run/site["id"]/key
            if not (folder/"status.json").exists():
                missing.append({"site": site["id"], "target_Mtpa": quantity, "constrained": constrained})
                continue
            point = json.loads((folder/"status.json").read_text())
            if (point["site"], point["target_Mtpa"], point["constrained"], point["latitude"], point["longitude"]) != (
                    site["id"], quantity, constrained, site["latitude"], site["longitude"]):
                raise ValueError("Checkpoint identity differs from manifest")
            attempts = point["solver_attempts"]
            if not 1 <= len(attempts) <= 2 or attempts[0]["solver_options_override"] is not None:
                raise ValueError("Invalid first solver attempt")
            if len(attempts) == 2 and attempts[1]["solver_options_override"] != manifest["numerical_retry_options"]:
                raise ValueError("Unrecorded fallback")
            metrics = {}
            if point["status"] == "feasible":
                if sha(folder/"result.csv") != point["result_sha256"]:
                    raise ValueError("Checkpoint result hash failed")
                frame = pd.read_csv(folder/"result.csv")
                if len(frame) != 1:
                    raise ValueError("Result must have one row")
                result = frame.iloc[0]
                if (result.solver_status, result.solver_termination, attempts[-1]["status"], attempts[-1]["termination"]) != ("ok", "optimal", "ok", "optimal"):
                    raise ValueError("Checkpoint not optimal")
                if result.lcoa_land_mode != ("enforce" if constrained else "postprocess"):
                    raise ValueError("Unexpected solve mode")
                metrics = validate_result(result.to_dict(), quantity,
                    land.loc[(site["latitude"], site["longitude"])], constrained=constrained)
                np.testing.assert_allclose(result.lcoa_eur_per_t, point["LCOA_EUR2020_per_t"], rtol=1e-12)
                if not constrained and abs(result.lcoa_eur_per_t-site["control_LCOA_EUR2020_per_t"]) > experiment["control_lcoa_tolerance_eur_per_t"]:
                    raise ValueError("Control cost changed")
            elif point["status"] == "infeasible":
                if not constrained or point["solver_termination"] != "infeasible" or attempts[-1]["termination"] != "infeasible" or (folder/"result.csv").exists():
                    raise ValueError("Unproven infeasibility")
            else:
                raise ValueError("Unknown checkpoint state")
            rows.append({**point, **metrics, "checkpoint_sha256": sha(folder/"status.json")})
    a.output.mkdir(parents=True, exist_ok=False)
    pd.DataFrame(rows).to_csv(a.output/"validated_checkpoints.csv", index=False)
    summary = {"created_utc": datetime.now(timezone.utc).isoformat(), "checkpoint_qa_pass": True,
               "full_run_accepted": False, "checkpoints_checked": len(rows),
               "constrained_points_checked": sum(r["constrained"] for r in rows), "missing_checkpoints": missing,
               "scope": "Fetched snapshot only; missing endpoints are unresolved, not infeasible. Full-run provenance/energy-bound audit remains required.",
               "source_manifest_sha256": sha(a.run/"manifest.json"), "land_sha256": sha(a.land),
               "output_sha256": sha(a.output/"validated_checkpoints.csv"),
               "code_sha256": {str(path.relative_to(ROOT)): sha(path) for path in
                               (Path(__file__), ROOT/"reconciliation/land/run_supply_pilot.py")}}
    (a.output/"summary.json").write_text(json.dumps(summary, indent=2)+"\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
