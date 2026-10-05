#!/usr/bin/env python3
"""Export the two validated global surfaces with a declared 1 Mt/year cutoff."""
import argparse
from pathlib import Path
from types import SimpleNamespace
import pandas as pd

try:
    from .export_lory_surface import export
    from .run_historical_network import digest, write_json
except ImportError:
    from export_lory_surface import export
    from run_historical_network import digest, write_json


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--campaign", type=Path, required=True)
    p.add_argument("--received", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    original = args.campaign / "historical/git-0a63616/data/c_NH3_cost_4.5.csv"
    old = pd.read_csv(original)
    reference = old[["country", "iso3"]].drop_duplicates()
    # Keep the historical network's country-code convention for these cells.
    # Natural Earth labels them Somaliland, whereas the archived supplier table
    # records Somalia/SOM. This is an explicit modelling mapping, not a change
    # to the geographic labels or a claim about recognition.
    check = old[(old.Latitude == 10) & old.Longitude.between(44, 48)]
    if len(check) != 5 or not check.iso3.eq("SOM").all():
        raise ValueError("Archived evidence for the Somaliland country-code mapping changed")
    reference = pd.concat([reference, pd.DataFrame([{"country": "Somaliland", "iso3": "SOM"}])], ignore_index=True)
    mapping = args.output / "country_reference.csv"
    reference.to_csv(mapping, index=False)
    write_json(args.output / "country_reference_provenance.json", {
        "source": str(original.resolve()), "source_sha256": digest(original),
        "output_sha256": digest(mapping),
        "explicit_aliases": {"Somaliland": "SOM"},
        "alias_basis": "Preserve archived network code for shared supplier cells at latitude 10, longitudes 44 to 48; retain Natural Earth display labels.",
        "unassigned_policy": "Do not assign countries to unassigned cells. Below-cutoff cells are excluded explicitly, and any eligible unmapped cell is a hard error.",
    })
    for label, directory in [("rep", "10_replication"), ("central", "20_central")]:
        manifests = list((args.received / "lory" / directory).rglob("global/manifest.json"))
        if len(manifests) != 1:
            raise ValueError("Expected exactly one global run per surface")
        run = manifests[0].parent
        export(SimpleNamespace(surface=run / "merged/global_run_results.csv",
                               qa=run / "qa/validation.json", manifest=run / "manifest.json",
                               country_reference=mapping, output=args.output / label,
                               allow_diagnostic=False, minimum_capacity=1.))


if __name__ == "__main__":
    main()
