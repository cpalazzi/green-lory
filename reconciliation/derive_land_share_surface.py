#!/usr/bin/env python3
"""Derive a land-share variant of a QA-passed surface by rescaling its capacities.

Under --land-constraint after_solve the plant design and LCOA do not depend on the land share;
only the scaled-design capacities do, and linearly.  So the 2 % variant of a 20 % surface is the
same table with every scaled_design capacity multiplied by 0.1.  The derived surface keeps the
parent's LCOA columns untouched and records the multiplier and the parent's hashes.

    python reconciliation/derive_land_share_surface.py --surface <merged csv> --multiplier 0.1 \
        --parent-share 0.20 --output-dir <new dir>
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

CAPACITY_PREFIX = "scaled_design_max_"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--surface", type=Path, required=True, help="merged global_run_results.csv of the parent run")
    p.add_argument("--multiplier", type=float, required=True, help="capacity multiplier, e.g. 0.1 for 20 % -> 2 %")
    p.add_argument("--parent-share", type=float, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    a = p.parse_args()
    a.output_dir.mkdir(parents=True, exist_ok=False)
    df = pd.read_csv(a.surface, low_memory=False)
    if "land_constraint" in df.columns and set(df["land_constraint"].dropna().unique()) - {"after_solve"}:
        raise SystemExit("Only after_solve surfaces can be rescaled post hoc")
    scaled = [c for c in df.columns if c.startswith(CAPACITY_PREFIX)]
    if not scaled:
        raise SystemExit("No scaled_design capacity columns found")
    for c in scaled:
        df[c] = pd.to_numeric(df[c], errors="coerce") * a.multiplier
    if "scaled_design_renewable_capacity_scale_factor" in df.columns:
        df["scaled_design_renewable_capacity_scale_factor"] = pd.to_numeric(df["scaled_design_renewable_capacity_scale_factor"], errors="coerce") * a.multiplier
    df["land_share_multiplier_applied"] = a.multiplier
    df["land_competition_fraction_effective"] = a.parent_share * a.multiplier
    out = a.output_dir / "global_run_results.csv"
    df.to_csv(out, index=False)
    provenance = {
        "schema_version": 1,
        "kind": "post_hoc_land_share_variant",
        "parent_surface": {"path": str(a.surface.resolve()), "sha256": sha256(a.surface)},
        "parent_land_competition_fraction": a.parent_share,
        "multiplier": a.multiplier,
        "effective_land_competition_fraction": a.parent_share * a.multiplier,
        "columns_rescaled": scaled + (["scaled_design_renewable_capacity_scale_factor"] if "scaled_design_renewable_capacity_scale_factor" in df.columns else []),
        "lcoa_columns_untouched": True,
        "justification": "land_constraint=after_solve: the reference design is solved without land; capacity scales linearly with the land share",
        "output": {"path": str(out.resolve()), "sha256": sha256(out), "rows": int(len(df))},
    }
    (a.output_dir / "provenance.json").write_text(json.dumps(provenance, indent=2))
    print(json.dumps(provenance, indent=2))


if __name__ == "__main__":
    main()
