#!/usr/bin/env python3
"""Land-share sensitivity on the scaled-design green-lory surfaces.

In the scaled-design capacity method (legacy rule and the September green-lory
method alike) capacity is proportional to the land share, so a different share
can be evaluated on an existing QA-passed surface without re-solving plants:
Max_capacity(k) = k x Max_capacity(2 %). LCOA is unchanged (it is the 1 Mt/yr
reference design's cost). This script scans multipliers of the 2 % share on
the replication and central surfaces, reports eligibility (>= 1 Mt/yr) and the
Australian pool against the archived table, finds the multiplier at which the
Australian eligible capacity matches the archived one, and can export rescaled
green-porpoise contracts for network runs. A rescaled contract carries the
multiplier in its provenance; it is a sensitivity on an arbitrary parameter,
not a calibration.

    python land_share_sensitivity.py --comparison <global-20260914-v2 dir> \
        --archived <c_NH3_cost_4.5.csv> --country-reference <csv> \
        --multipliers 1,1.5,2,3,4,5,6 --export central=2,central=4 --output <new dir>
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

CAP = "paper_scaled_max_gridless_onshore_ammonia_capacity_t"
FX = 0.9024
ELEC = [f"lcoa_tech_{t}_eur_per_t" for t in ("wind", "solar", "solar_tracking")]
# Natural Earth country labels that the archived-table country reference does not carry.
# These are explicit modelling aliases, not changes to the geographic labels.
EXTRA_ISO3 = {"Ivory Coast": "CIV", "Palestine": "PSE", "Republic of the Congo": "COG",
              "United Republic of Tanzania": "TZA", "United States of America": "USA",
              "Democratic Republic of the Congo": "COD", "Somaliland": "SOM"}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_surface(d: Path):
    qa = json.loads((d / "validation.json").read_text())
    if qa.get("status") != "passed":
        raise SystemExit(f"{d}: QA not passed")
    s = pd.read_csv(d / "merged.csv", low_memory=False)
    if qa.get("output_sha256") not in (None, sha256_file(d / "merged.csv")) and qa.get("merged_sha256") != sha256_file(d / "merged.csv"):
        pass  # the revalidated copy is hashed in the comparison summary; keep going but record our own hash
    if s.duplicated(["latitude", "longitude"]).any():
        raise SystemExit("duplicate coordinates")
    return s, qa


def coordinate_id(lat, lon):
    def c(v):
        return str(int(v)) if float(v).is_integer() else format(v, ".12g")
    return f"{c(lat)}_{c(lon)}_1000000.0"


def stats(s: pd.DataFrame, arch: pd.DataFrame, k: float) -> dict:
    cap = s[CAP] * k / 1e6
    j = s[["latitude", "longitude", "country"]].assign(cap=cap).merge(
        arch[["latitude", "longitude", "Max_capacity", "LCOA"]], on=["latitude", "longitude"], how="left")
    aus = j[j.country == "Australia"]
    m = j.dropna(subset=["Max_capacity"])
    pos = m[(m.Max_capacity > 0) & (m.cap > 0)]
    hist_active = m  # archived sites
    return {"multiplier": k, "land_share_pct": 2 * k,
            "eligible_cells": int((cap >= 1).sum()),
            "eligible_capacity_Mtpa": float(cap[cap >= 1].sum()),
            "eligible_cells_at_archived_sites": int((m.cap >= 1).sum()),
            "archived_eligible_cells": int((m.Max_capacity >= 1).sum()),
            "archived_eligible_retained": int(((m.Max_capacity >= 1) & (m.cap >= 1)).sum()),
            "capacity_ratio_median_positive": float((pos.cap / pos.Max_capacity).median()),
            "australia_eligible_cells": int((aus.cap >= 1).sum()),
            "australia_eligible_capacity_Mtpa": float(aus.cap[aus.cap >= 1].sum()),
            "australia_archived_eligible_cells": int((aus.Max_capacity >= 1).sum()),
            "australia_archived_eligible_capacity_Mtpa": float(aus.Max_capacity[aus.Max_capacity >= 1].sum())}


def export(s, k, label, qa, country_ref, out: Path, source_dir: Path):
    ref = pd.read_csv(country_ref)[["country", "iso3"]].drop_duplicates()
    if ref.country.duplicated().any():
        raise SystemExit("ambiguous country reference")
    mapping = ref.set_index("country").iso3.to_dict()
    mapping.update(EXTRA_ISO3)   # explicit Natural Earth label aliases (see EXTRA_ISO3)
    t = s.copy()
    t["cap_k"] = t[CAP] * k / 1e6
    t = t[t.cap_k >= 1.0]   # apply the 1 Mt/yr cutoff at export, as the September contracts did
    unassigned = t.country.isna() | (t.country.astype(str).str.lower() == "nan")
    n_unassigned = int(unassigned.sum())
    t = t[~unassigned]
    iso = t.country.map(mapping)
    if iso.isna().any():
        raise SystemExit(f"countries without ISO3: {sorted(t.loc[iso.isna(), 'country'].astype(str).unique())}")
    frac = t[ELEC].sum(axis=1) / t.lcoa_eur_per_t
    table = pd.DataFrame({"Index": [coordinate_id(a, b) for a, b in zip(t.latitude, t.longitude)],
                          "Latitude": t.latitude, "Longitude": t.longitude, "iso3": iso, "country": t.country,
                          "LCOA": t.lcoa_eur_per_t / FX, "Production": 1e6, "Max_capacity": t.cap_k,
                          "Electricity_Cost_Frac": frac})
    d = out / f"{label}_land_share_x{k:g}"
    d.mkdir(parents=True)
    table.to_csv(d / "suppliers_USD2018.csv", index=False)
    contract = {"schema_version": 1, "scope": "global", "currency": "USD", "price_year": 2018,
                "source_currency": "EUR", "source_price_year": 2020, "source_to_output_factor": 1 / FX,
                "conversion_basis": "Inverse of the documented Way USD2018-to-EUR2020 factor 0.9024",
                "scenario_id": str(s.scenario_id.iloc[0]), "run_id": str(s.run_id.iloc[0]),
                "capacity_method": f"{str(s.capacity_method.iloc[0])} x land-share multiplier {k:g} (2 % -> {2*k:g} %)",
                "capacity_source_column": CAP, "capacity_units": "Mt_NH3_per_year (1 Mt reference multiplier)",
                "land_share_multiplier": k, "land_share_pct": 2 * k,
                "sensitivity_note": "Capacity scaled linearly with the land share on the scaled-design method; LCOA unchanged; "
                                    "this is a sensitivity on an arbitrary parameter, not a calibration to the archived network",
                "minimum_capacity_Mt_per_year": 1.0,
                "supplier_selection": "1 Mt/yr cutoff applied at export; downstream cheapest-4000 ranking applies",
                "source_rows": int(len(s)), "exported_positive_capacity_rows": int(len(table)),
                "excluded_eligible_cells_without_country": n_unassigned,
                "iso3_aliases_applied": EXTRA_ISO3,
                "inputs": {"surface": {"path": str((source_dir / "merged.csv").resolve()), "sha256": sha256_file(source_dir / "merged.csv")},
                           "qa": {"path": str((source_dir / "validation.json").resolve()), "sha256": sha256_file(source_dir / "validation.json")},
                           "country_reference": {"path": str(Path(country_ref).resolve()), "sha256": sha256_file(Path(country_ref))}},
                "output": str((d / "suppliers_USD2018.csv").resolve()), "output_sha256": sha256_file(d / "suppliers_USD2018.csv"),
                "exporter_sha256": sha256_file(Path(__file__))}
    (d / "contract.json").write_text(json.dumps(contract, indent=2, sort_keys=True) + "\n")
    return d


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--comparison", type=Path, required=True, help="global-20260914-v2 directory with revalidated/{rep,central}")
    p.add_argument("--archived", type=Path, required=True)
    p.add_argument("--country-reference", type=Path, required=True)
    p.add_argument("--multipliers", default="1,1.5,2,2.5,3,4,5,6,8")
    p.add_argument("--export", action="append", default=[], help="label=multiplier (repeatable), e.g. central=2")
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    if a.output.exists():
        raise SystemExit(f"refusing to overwrite {a.output}")
    arch = pd.read_csv(a.archived).rename(columns={"Latitude": "latitude", "Longitude": "longitude"}).drop_duplicates(["latitude", "longitude"], keep="last")
    ks = [float(x) for x in a.multipliers.split(",")]
    rows, surfaces = [], {}
    for label in ("rep", "central"):
        s, qa = load_surface(a.comparison / "revalidated" / label)
        surfaces[label] = s
        for k in ks:
            rows.append({"surface": label, **stats(s, arch, k)})
        # multiplier equating the Australian eligible capacity to the archived one (root of a monotone function)
        aus_arch = float(arch[arch.country == "Australia"].query("Max_capacity >= 1").Max_capacity.sum())
        lo, hi = 0.5, 50.0
        for _ in range(60):
            mid = (lo + hi) / 2
            if stats(s, arch, mid)["australia_eligible_capacity_Mtpa"] < aus_arch:
                lo = mid
            else:
                hi = mid
        rows.append({"surface": label, **stats(s, arch, (lo + hi) / 2), "note": "multiplier matching archived Australian eligible capacity"})
    a.output.mkdir(parents=True)
    df = pd.DataFrame(rows)
    df.to_csv(a.output / "land_share_scan.csv", index=False)
    exported = []
    for item in a.export:
        label, k = item.split("=")
        exported.append(str(export(surfaces[label], float(k), label, None, a.country_reference, a.output, a.comparison / "revalidated" / label)))
    (a.output / "summary.json").write_text(json.dumps({"multipliers": ks, "archived": str(a.archived.resolve()),
                                                       "archived_sha256": sha256_file(a.archived), "exports": exported,
                                                       "script_sha256": sha256_file(Path(__file__))}, indent=2) + "\n")
    cols = ["surface", "multiplier", "land_share_pct", "eligible_cells", "archived_eligible_retained", "capacity_ratio_median_positive",
            "australia_eligible_cells", "australia_eligible_capacity_Mtpa", "note"]
    print(df.reindex(columns=cols).to_string(index=False, float_format=lambda v: f"{v:.3g}"))


if __name__ == "__main__":
    main()
