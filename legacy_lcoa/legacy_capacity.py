#!/usr/bin/env python
"""Legacy land-to-capacity rule, recovered from the archived shipping input.

The archived supplier table `c_NH3_cost_4.5.csv` (Salmon, spring 2023) carries a
`Max_capacity` column whose generating code was never found. With the frozen
legacy plant replicated (LEGACY_REPLICATION_20260915.md) the column is
reproduced, cell by cell, by

    Q [Mt/yr] = Q_ref * min( A_solar * D_PV / P_pv , A_wind * D_WIND / P_wind )

where Q_ref = 1 Mt/yr is the reference design, P_pv / P_wind are the design's
PV (fixed + tracking) and wind capacities in MW for that reference design,
A_solar / A_wind are the 2 % Table-2 suitable areas of the 1-degree cell for
each technology (Salmon 2022, MODIS land classes, centered cell, no slope or
protected-area exclusions), and the technologies overlap completely (each sees
its own full budget; Salmon 2022 section 2.2.1).

The two densities are EMPIRICAL: they are what the archived table implies, not
what the paper states (about 9 km2/GW with a latitude adjustment for PV, and
200 km2/GW for wind). On the 563-cell latitude-stratified sample the rule
reproduces PV-dominated cells with median ratio 0.998 (IQR 0.977-1.057 on the
high-suitability subset), wind-dominated cells with 0.996, and under-predicts
mixed wind/solar cells by about 20 % at the median (design-dependent). Every
export made with this module carries these flags in its contract.

Usage as a library:

    from legacy_capacity import legacy_capacity, rule_provenance
    q = legacy_capacity(A_solar_km2, A_wind_km2, P_pv_mw, P_wind_mw)   # DataFrame

Usage as a tool (apply the rule to a green-lory surface at the archived sites):

    python legacy_capacity.py apply-lory --surface .../global_run_results.csv \
        --qa .../validation.json --manifest .../manifest.json \
        --land .../historical_sites_unmasked.csv --archived .../c_NH3_cost_4.5.csv \
        --output <new directory>
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

# Two capacity rules are kept, deliberately separate:
#
# archived_table_reproduction (RULE_ID below): the constants that reproduce the archived
#   Max_capacity column, recovered on the 563-cell sample. They are NOT the paper's stated
#   values and no source for them has been found; the stack that produced the archived
#   table is inconsistent with the paper's land method and is archived under this name.
# stated_method: what Salmon & Banares-Alcantara (2022) section 2.2.1 and Verschuur et al.
#   (2024) section 4.7 state: 200 km2/GW of wind, PV packed by latitude after van de Ven
#   et al. (2021) with a First Solar module (about 9 km2/GW at the equator), complete
#   wind/solar overlap, 2 % of the suitable area after protected-area and > 15 degree slope
#   exclusions. Areas and the latitude-dependent density come from the green-lory land build
#   (`paper_2pct_slope15.csv`), which implements exactly that method.
RULE_ID = "legacy_rule_complete_overlap_v1"     # archived_table_reproduction
STATED_RULE_ID = "stated_method_v1"
D_PV_MW_PER_KM2 = 140.0      # 7.1 km2/GW, fixed and tracking alike, no latitude adjustment
D_WIND_MW_PER_KM2 = 7.3      # 137 km2/GW
STATED_D_WIND_MW_PER_KM2 = 5.0   # 200 km2/GW (Salmon 2022 section 2.2.1; Verschuur 2024 section 4.7)
LAND_SHARE = 0.02            # already applied in the 2 % area columns used as input
REFERENCE_MTPA = 1.0
AREA_ANCHOR = "center"
EUR2020_PER_USD2018 = 0.9024  # documented Way conversion used by export_lory_surface.py

PROVENANCE = {
    "rule_id": RULE_ID,
    "formula": "Q = Q_ref * min(A_solar * D_PV / P_pv, A_wind * D_WIND / P_wind)",
    "d_pv_MW_per_km2": D_PV_MW_PER_KM2,
    "d_wind_MW_per_km2": D_WIND_MW_PER_KM2,
    "reference_design_Mtpa": REFERENCE_MTPA,
    "land_share": LAND_SHARE,
    "land_basis": "2 % of Salmon (2022) Table-2 suitable area per technology, MODIS 2022 classes, centered 1-degree cell, no slope/protected exclusions",
    "sharing": "complete overlap (each technology sees its full 2 % budget)",
    "pv_latitude_adjustment": "none",
    "tracking_footprint": "same as fixed (no penalty)",
    "variant": "archived_table_reproduction",
    "constants_status": "empirical, recovered from the archived Max_capacity column with replicated designs; not the paper's stated 9 km2/GW-with-latitude or 200 km2/GW; this stack reproduces the archived (inconsistent) land results and is archived as such",
    "validation": "563-cell sample (LEGACY_REPLICATION_20260915.md section 6): PV-dominated median 0.998, wind-dominated 0.996, mixed cells ~0.80 (design-dependent)",
    "known_residuals": ["mixed wind/solar cells under-predicted by ~20 % at the median",
                        "land vintage (MODIS 2022) and any exclusions Salmon applied are not recoverable"],
}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def legacy_capacity(a_solar_km2, a_wind_km2, p_pv_mw, p_wind_mw,
                    d_pv: float = D_PV_MW_PER_KM2, d_wind: float = D_WIND_MW_PER_KM2,
                    reference_mtpa: float = REFERENCE_MTPA) -> pd.DataFrame:
    """Vectorised rule. Inputs are array-like of equal length.

    Returns a DataFrame with Q_pv_limit_Mtpa, Q_wind_limit_Mtpa, Max_capacity
    (Mt/yr, 0 when no technology limit is finite or the land is missing) and
    limiting_technology ("pv", "wind" or "none").
    """
    a_s = np.asarray(a_solar_km2, dtype=float)
    a_w = np.asarray(a_wind_km2, dtype=float)
    p_pv = np.asarray(p_pv_mw, dtype=float)
    p_w = np.asarray(p_wind_mw, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        q_pv = np.where(p_pv > 0, reference_mtpa * a_s * d_pv / p_pv, np.inf)
        q_w = np.where(p_w > 0, reference_mtpa * a_w * d_wind / p_w, np.inf)
    q = np.minimum(q_pv, q_w)
    bad = ~np.isfinite(q) | (q < 0)
    q = np.where(bad, 0.0, q)
    limiting = np.where(bad, "none", np.where(q_pv <= q_w, "pv", "wind"))
    return pd.DataFrame({"Q_pv_limit_Mtpa": q_pv, "Q_wind_limit_Mtpa": q_w,
                         "Max_capacity": q, "limiting_technology": limiting})


def rule_provenance(d_pv: float = D_PV_MW_PER_KM2, d_wind: float = D_WIND_MW_PER_KM2, anchor: str = AREA_ANCHOR) -> dict:
    out = dict(PROVENANCE)
    out.update({"d_pv_MW_per_km2": d_pv, "d_wind_MW_per_km2": d_wind, "area_anchor": anchor,
                "capacity_rule": f"min(A_solar*{d_pv}/P_pv, A_wind*{d_wind}/P_wind); complete overlap; 2 % Table-2 areas, {anchor} anchor, no exclusions",
                "module_sha256": sha256_file(Path(__file__))})
    if (d_pv, d_wind) != (D_PV_MW_PER_KM2, D_WIND_MW_PER_KM2):
        out["constants_status"] = "NON-DEFAULT constants; " + out["constants_status"]
    return out


def load_land(path: Path, anchor: str = AREA_ANCHOR) -> pd.DataFrame:
    """Legacy land step output (`legacy_land_areas.py`): 2 % Table-2 areas, no exclusions,
    centered cells. Returns A_solar_km2, A_wind_km2 and cell_area_km2."""
    land = pd.read_csv(path)
    cols = {f"{anchor}_solar_shipping_km2": "A_solar_km2", f"{anchor}_wind_shipping_km2": "A_wind_km2",
            f"{anchor}_cell_area_km2": "cell_area_km2"}
    keep = ["latitude", "longitude", *cols]
    if "historical_capacity_Mtpa" in land:
        keep.append("historical_capacity_Mtpa")
    land = land[keep].rename(columns=cols)
    if land.duplicated(["latitude", "longitude"]).any():
        raise ValueError("duplicate coordinates in land table")
    return land


def stated_land_anchor(path: Path) -> str:
    """Cell anchoring recorded in a green-lory land table (`cell_anchor` column); tables built
    before 23 September 2026 carry no column and were labelled inconsistently (MODIS content of
    the cell one degree south, exclusions of the labelled cell), reported as 'southwest_mislabelled'."""
    land = pd.read_csv(path, low_memory=False, usecols=lambda c: c == "cell_anchor")
    if "cell_anchor" not in land:
        return "southwest_mislabelled"
    anchors = set(land["cell_anchor"].dropna().astype(str))
    if len(anchors) != 1:
        raise ValueError(f"land table mixes cell anchors: {sorted(anchors)}")
    return anchors.pop()


def load_stated_land(path: Path) -> pd.DataFrame:
    """green-lory land build at a 2 % share: 2 % of the Table-2 suitable area after
    protected-area and > 15 degree slope exclusions, plus the latitude-packed First Solar PV
    density. Use a centred table (`cell_anchor` = center, build of 23 September 2026) so that
    the cells coincide with the legacy step and the weather nodes; see stated_land_anchor()."""
    land = pd.read_csv(path, low_memory=False)
    need = ["latitude", "longitude", "solar_area_km2", "wind_onshore_area_km2", "solar_density_mw_per_km2",
            "wind_density_mw_per_km2", "onshore_area_km2"]
    missing = [c for c in need if c not in land]
    if missing:
        raise ValueError(f"stated-method land table lacks {missing}")
    out = land[need].rename(columns={"solar_area_km2": "A_solar_km2", "wind_onshore_area_km2": "A_wind_km2",
                                     "onshore_area_km2": "cell_area_km2", "solar_density_mw_per_km2": "d_pv_mw_per_km2",
                                     "wind_density_mw_per_km2": "d_wind_table_mw_per_km2"})
    if out.duplicated(["latitude", "longitude"]).any():
        raise ValueError("duplicate coordinates in land table")
    if not np.allclose(out.d_wind_table_mw_per_km2.dropna(), STATED_D_WIND_MW_PER_KM2):
        raise ValueError("land table wind density is not the stated 5 MW/km2 (200 km2/GW)")
    return out


def stated_capacity(a_solar_km2, a_wind_km2, p_pv_mw, p_wind_mw, d_pv_mw_per_km2,
                    d_wind: float = STATED_D_WIND_MW_PER_KM2, reference_mtpa: float = REFERENCE_MTPA) -> pd.DataFrame:
    """Stated method: same complete-overlap formula with the latitude-dependent PV density
    (fixed and tracking alike, the papers give no tracking footprint) and 5 MW/km2 wind."""
    a_s = np.asarray(a_solar_km2, dtype=float)
    a_w = np.asarray(a_wind_km2, dtype=float)
    p_pv = np.asarray(p_pv_mw, dtype=float)
    p_w = np.asarray(p_wind_mw, dtype=float)
    d_pv = np.asarray(d_pv_mw_per_km2, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        q_pv = np.where(p_pv > 0, reference_mtpa * a_s * d_pv / p_pv, np.inf)
        q_w = np.where(p_w > 0, reference_mtpa * a_w * d_wind / p_w, np.inf)
    q = np.minimum(q_pv, q_w)
    bad = ~np.isfinite(q) | (q < 0)
    q = np.where(bad, 0.0, q)
    limiting = np.where(bad, "none", np.where(q_pv <= q_w, "pv", "wind"))
    return pd.DataFrame({"Q_pv_limit_Mtpa": q_pv, "Q_wind_limit_Mtpa": q_w,
                         "Max_capacity": q, "limiting_technology": limiting})


def stated_rule_provenance(anchor: str = "center") -> dict:
    caveats = ["source vintages substituted (MODIS 2022, WDPA Feb 2026, GEBCO 2025)"]
    if anchor == "center":
        land_cells = "centred 1-degree cells (green-lory land build of 23 September 2026, same cells as the legacy step and the weather nodes)"
    else:
        land_cells = f"{anchor} 1-degree cells (green-lory land build)"
        caveats.insert(0, "cell anchor differs from the centered legacy/weather convention" +
                       ("; MODIS content mislabelled by one degree (build before 23 Sep 2026)" if anchor == "southwest_mislabelled" else " by half a cell"))
    return {"rule_id": STATED_RULE_ID,
            "formula": "Q = Q_ref * min(A_solar * d_pv(lat) / P_pv, A_wind * 5 / P_wind); complete overlap",
            "d_pv": "latitude-packed First Solar module density from the green-lory land build (about 106 MW/km2 at the equator, 78 at 23 degrees), applied to fixed and tracking PV alike",
            "d_wind_MW_per_km2": STATED_D_WIND_MW_PER_KM2,
            "land_basis": f"2 % of Table-2 suitable area after WDPA protected-area and > 15 degree slope exclusions, MODIS MCD12C1 2022 C6.1, {land_cells}",
            "land_cell_anchor": anchor,
            "sources": "Salmon & Banares-Alcantara 2022 section 2.2.1; Verschuur et al. 2024 section 4.7 and supplementary table 2; van de Ven et al. 2021",
            "constants_status": "as stated in the papers; not the constants that reproduce the archived table",
            "known_caveats": caveats,
            "module_sha256": sha256_file(Path(__file__))}


def coordinate_id(lat: float, lon: float) -> str:
    def c(v):
        return str(int(v)) if float(v).is_integer() else format(v, ".12g")
    return f"{c(lat)}_{c(lon)}_1000000.0"


def apply_lory(args):
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite {args.output}")
    qa = json.loads(args.qa.read_text())
    manifest = json.loads(args.manifest.read_text())
    if qa.get("status") != "passed":
        raise SystemExit("surface QA is not 'passed'")
    if qa.get("output_sha256") != sha256_file(args.surface):
        raise SystemExit("QA output_sha256 does not identify this surface file")
    surface = pd.read_csv(args.surface, low_memory=False)
    need = ["latitude", "longitude", "country", "lcoa_eur_per_t", "annual_ammonia_production_t", "currency",
            "wind_mw", "solar_mw", "solar_tracking_mw", "paper_scaled_max_gridless_onshore_ammonia_capacity_t",
            "lcoa_tech_wind_eur_per_t", "lcoa_tech_solar_eur_per_t", "lcoa_tech_solar_tracking_eur_per_t",
            "is_gridless_feasible", "is_full_year_result", "scenario_id", "run_id"]
    missing = [c for c in need if c not in surface]
    if missing:
        raise SystemExit(f"surface lacks columns {missing}")
    if not surface.currency.eq("EUR").all():
        raise SystemExit("surface currency is not uniformly EUR")
    if not np.allclose(surface.annual_ammonia_production_t, 1e6, rtol=0, atol=1):
        raise SystemExit("surface is not a 1 Mt/yr reference design")
    for name in ("is_gridless_feasible", "is_full_year_result"):
        if not surface[name].astype(str).str.lower().isin(["true", "1"]).all():
            raise SystemExit(f"surface is not uniformly {name}")
    if surface.duplicated(["latitude", "longitude"]).any():
        raise SystemExit("duplicate coordinates in surface")

    land = load_land(args.land, args.anchor)
    arch = pd.read_csv(args.archived).rename(columns={"Latitude": "latitude", "Longitude": "longitude"})
    arch = arch[["latitude", "longitude", "iso3", "country", "LCOA", "Max_capacity", "Electricity_Cost_Frac"]].rename(
        columns={"country": "country_archived", "LCOA": "LCOA_archived", "Max_capacity": "Max_capacity_archived",
                 "Electricity_Cost_Frac": "Electricity_Cost_Frac_archived"})
    arch = arch.drop_duplicates(["latitude", "longitude"])
    # the legacy universe: cells that have a legacy land budget (the archived sites)
    cells = land.merge(surface, on=["latitude", "longitude"], how="inner").merge(arch, on=["latitude", "longitude"], how="left")
    n_land_not_in_surface = int(len(land) - len(cells))
    cells["P_pv"] = cells.solar_mw + cells.solar_tracking_mw
    cells["P_wind"] = cells.wind_mw
    rule = legacy_capacity(cells.A_solar_km2, cells.A_wind_km2, cells.P_pv, cells.P_wind, args.d_pv, args.d_wind)
    for c in rule:
        cells[c] = rule[c].to_numpy()
    cells["lory_capacity_Mtpa"] = cells.paper_scaled_max_gridless_onshore_ammonia_capacity_t / 1e6
    cells["LCOA"] = cells.lcoa_eur_per_t / EUR2020_PER_USD2018
    cells["Electricity_Cost_Frac"] = (cells.lcoa_tech_wind_eur_per_t + cells.lcoa_tech_solar_eur_per_t
                                      + cells.lcoa_tech_solar_tracking_eur_per_t) / cells.lcoa_eur_per_t
    if not cells.Electricity_Cost_Frac.between(-1e-8, 1 + 1e-8).all():
        raise SystemExit("generation-cost fractions outside [0, 1]")
    cells["Production"] = 1e6
    cells["Index"] = [coordinate_id(a, b) for a, b in zip(cells.latitude, cells.longitude)]
    # country/iso3 from the archived table (legacy universe); fall back to the surface country + reference
    cells["country"] = cells.country_archived.fillna(cells.country)
    if cells.iso3.isna().any():
        ref = pd.read_csv(args.country_reference)[["country", "iso3"]].drop_duplicates() if args.country_reference else None
        if ref is None or ref.country.duplicated().any():
            raise SystemExit("cells without archived iso3 need an unambiguous --country-reference")
        fill = cells.loc[cells.iso3.isna(), "country"].map(ref.set_index("country").iso3)
        if fill.isna().any():
            raise SystemExit(f"no ISO3 for {sorted(cells.loc[fill.index[fill.isna()], 'country'].astype(str).unique())}")
        cells.loc[fill.index, "iso3"] = fill
    cells["capacity_ratio_vs_archived"] = cells.Max_capacity / cells.Max_capacity_archived.replace(0, np.nan)
    cells["capacity_ratio_vs_lory"] = cells.Max_capacity / cells.lory_capacity_Mtpa.replace(0, np.nan)
    cells["lcoa_ratio_vs_archived"] = cells.LCOA / cells.LCOA_archived

    args.output.mkdir(parents=True)
    contract_cols = ["Index", "latitude", "longitude", "iso3", "country", "LCOA", "Production", "Max_capacity", "Electricity_Cost_Frac"]
    table = cells[contract_cols].rename(columns={"latitude": "Latitude", "longitude": "Longitude"})
    exported = table[table.Max_capacity > 0].copy()
    exported.to_csv(args.output / "suppliers_USD2018.csv", index=False)
    full_cols = ["Index", "latitude", "longitude", "iso3", "country", "LCOA", "LCOA_archived", "lcoa_ratio_vs_archived",
                 "P_wind", "solar_mw", "solar_tracking_mw", "P_pv", "A_solar_km2", "A_wind_km2", "cell_area_km2",
                 "Q_pv_limit_Mtpa", "Q_wind_limit_Mtpa", "Max_capacity", "limiting_technology",
                 "Max_capacity_archived", "capacity_ratio_vs_archived", "lory_capacity_Mtpa", "capacity_ratio_vs_lory",
                 "Electricity_Cost_Frac", "Electricity_Cost_Frac_archived"]
    cells[full_cols].to_csv(args.output / "cells_full.csv", index=False)

    ok = cells.dropna(subset=["Max_capacity_archived"])
    pos = ok[(ok.Max_capacity_archived > 0) & (ok.Max_capacity > 0)]
    def q(s):
        return {str(k): round(float(v), 4) for k, v in s.quantile([.05, .25, .5, .75, .95]).items()}
    aus = ok[ok.country == "Australia"]
    summary = {
        "surface_rows": int(len(surface)), "land_rows": int(len(land)), "cells": int(len(cells)),
        "land_cells_not_in_surface": n_land_not_in_surface,
        "exported_positive_capacity_rows": int(len(exported)),
        "lcoa_ratio_vs_archived": q(ok.lcoa_ratio_vs_archived),
        "capacity_ratio_vs_archived_positive": q(pos.capacity_ratio_vs_archived),
        "capacity_ratio_vs_lory_positive": q(cells[(cells.lory_capacity_Mtpa > 0) & (cells.Max_capacity > 0)].capacity_ratio_vs_lory),
        "eligible_ge_1Mt": {"archived": int((ok.Max_capacity_archived >= 1).sum()),
                            "legacy_rule_on_lory_designs": int((ok.Max_capacity >= 1).sum()),
                            "lory_september_method": int((ok.lory_capacity_Mtpa >= 1).sum()),
                            "both_archived_and_rule": int(((ok.Max_capacity_archived >= 1) & (ok.Max_capacity >= 1)).sum())},
        "total_capacity_Mtpa": {"archived": float(ok.Max_capacity_archived.sum()), "legacy_rule_on_lory_designs": float(ok.Max_capacity.sum()),
                                "lory_september_method": float(ok.lory_capacity_Mtpa.sum())},
        "australia": {"eligible_archived": int((aus.Max_capacity_archived >= 1).sum()),
                      "eligible_legacy_rule_on_lory_designs": int((aus.Max_capacity >= 1).sum()),
                      "eligible_lory_september_method": int((aus.lory_capacity_Mtpa >= 1).sum()),
                      "capacity_archived_Mtpa": float(aus.Max_capacity_archived.sum()),
                      "capacity_legacy_rule_Mtpa": float(aus.Max_capacity.sum()),
                      "capacity_lory_september_Mtpa": float(aus.lory_capacity_Mtpa.sum())},
        "limiting_technology_counts": {k: int(v) for k, v in cells.limiting_technology.value_counts().items()},
        "focal_cells": {f"{int(r.latitude)},{int(r.longitude)}": {"LCOA": round(float(r.LCOA), 2), "LCOA_archived": r.LCOA_archived,
                        "P_wind": round(float(r.P_wind), 1), "P_pv": round(float(r.P_pv), 1),
                        "Max_capacity": round(float(r.Max_capacity), 3), "Max_capacity_archived": r.Max_capacity_archived,
                        "lory_capacity_Mtpa": round(float(r.lory_capacity_Mtpa), 3), "limiting": r.limiting_technology}
                        for _, r in cells[cells.Index.isin(["-23_-69_1000000.0", "-23_117_1000000.0", "-21_135_1000000.0"])].iterrows()},
    }
    contract = {
        "schema_version": 1, "scope": "global", "currency": "USD", "price_year": 2018,
        "source_currency": "EUR", "source_price_year": 2020, "source_to_output_factor": 1 / EUR2020_PER_USD2018,
        "conversion_basis": "Inverse of the documented Way USD2018-to-EUR2020 factor 0.9024; no live FX conversion",
        "scenario_id": manifest["scenario"]["id"], "run_id": manifest["run_id"],
        "derived_from": "green-lory surface costs and designs; capacity replaced by the legacy land rule at the archived supplier sites",
        "capacity_method": f"{RULE_ID} applied to green-lory designs", "capacity_rule_provenance": rule_provenance(args.d_pv, args.d_wind, args.anchor),
        "capacity_source_column": "Max_capacity", "capacity_units": "Mt_NH3_per_year (1 Mt reference multiplier)",
        "universe": "archived supplier sites only (cells with a legacy 2 % land budget); other surface cells are not exported",
        "electricity_cost_fraction_basis": "Wind + fixed-PV + tracking-PV annualised generation cost / headline LCOA; excludes storage and conversion plant",
        "minimum_capacity_Mt_per_year": 0.0,
        "supplier_selection": "No cutoff applied at export; downstream selection applies the 1 Mt/yr cutoff and cheapest-4000 ranking",
        "source_rows": int(len(surface)), "universe_rows": int(len(cells)),
        "exported_positive_capacity_rows": int(len(exported)), "excluded_zero_capacity_rows": int((table.Max_capacity <= 0).sum()),
        "inputs": {name: {"path": str(path.resolve()), "sha256": sha256_file(path)} for name, path in
                   [("surface", args.surface), ("qa", args.qa), ("manifest", args.manifest), ("land", args.land), ("archived", args.archived)]
                   + ([("country_reference", args.country_reference)] if args.country_reference else [])},
        "output": str((args.output / "suppliers_USD2018.csv").resolve()),
        "output_sha256": sha256_file(args.output / "suppliers_USD2018.csv"),
        "exporter_sha256": sha256_file(Path(__file__)),
        "summary": summary,
    }
    with open(args.output / "contract.json", "w") as f:
        json.dump(contract, f, indent=2, sort_keys=True)
    with open(args.output / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)
    a = sub.add_parser("apply-lory", help="apply the rule to a QA-passed green-lory surface at the archived sites")
    for name in ("surface", "qa", "manifest", "land", "archived", "output"):
        a.add_argument(f"--{name}", type=Path, required=True)
    a.add_argument("--country-reference", type=Path, help="country -> iso3 table for cells without an archived iso3")
    a.add_argument("--d-pv", type=float, default=D_PV_MW_PER_KM2)
    a.add_argument("--d-wind", type=float, default=D_WIND_MW_PER_KM2)
    a.add_argument("--anchor", choices=["center", "southwest"], default=AREA_ANCHOR)
    a.set_defaults(func=apply_lory)
    s = sub.add_parser("show", help="print the rule constants and provenance")
    s.set_defaults(func=lambda args: print(json.dumps(rule_provenance(), indent=2)))
    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
