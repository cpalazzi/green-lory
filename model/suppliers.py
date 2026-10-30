"""Supplier tables: green-lory surfaces as supplier tables of the green-porpoise network model.

- ``export``: a QA-passed surface (the global result table) to the green-porpoise supplier
  schema (``Index``, ``LCOA`` in USD2018, ``Max_capacity`` in Mt/yr of a 1 Mt reference plant,
  ``iso3``, ...), written as ``suppliers_USD2018.csv`` with ``suppliers_USD2018_provenance.json``
  beside it: the input paths and hashes, the currency conversion, the capacity cutoff and every
  exclusion, and the CSV's hash, which green-porpoise checks before a network run. Costs use the
  inverse of the documented Way conversion (USD2018 x 0.9024 = EUR2020); no live exchange rate.
  The surface itself is never changed.
- ``derive``: a land-share variant of a QA-passed surface without a new solve. Under
  ``land_constraint: after_solve`` the plant design and LCOA do not depend on the land share;
  only the scaled-design capacities do, and linearly. So the 2 % variant of a 20 % surface is
  the same table with every scaled-design capacity multiplied by 0.1, recorded in
  ``provenance.json`` with the parent's hash; ``export --derived-provenance`` checks it against
  the parent's QA report.

    python -m model.suppliers derive --surface <merged csv> --multiplier 0.1 --parent-share 0.20 \
        --output-dir <new dir>
    python -m model.suppliers export --surface <csv> --qa <validation.json> --manifest <manifest.json> \
        --country-reference inputs/cells/country_reference_20260928_v1.csv --minimum-capacity 1 --drop-unassigned \
        --output <new dir>

The network itself runs in the green-porpoise repository (``legacy_gpo/``; linked here,
untracked, as ``legacy_gpo/``), which reads ``suppliers_USD2018.csv`` and its provenance file
(``--supplier-table``). Exports written before 30 October 2026 name the provenance file
``contract.json``.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

CAPACITY = "scaled_design_max_gridless_onshore_ammonia_capacity_t"
EUR2020_PER_USD2018 = .9024
ELECTRICITY_COSTS = [f"lcoa_tech_{technology}_eur_per_t" for technology in ("wind", "solar", "solar_tracking")]
CAPACITY_PREFIX = "scaled_design_max_"  # every column the land share scales
SUPPLIER_TABLE = "suppliers_USD2018.csv"
SUPPLIER_PROVENANCE = "suppliers_USD2018_provenance.json"


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path, payload):
    with Path(path).open("x", encoding="utf8") as f:
        json.dump(payload, f, indent=2, sort_keys=True, allow_nan=False)
        f.write("\n")


def convert(frame, country_reference, minimum_capacity=0., drop_unassigned=False):
    """Supplier table in the deposited green-porpoise schema.

    drop_unassigned=True removes cells whose country is NaN (no polygon overlap in the country
    geojson) before the ISO3 check; export() records how many cells and how much capacity that
    removed.  Without the flag such cells above the cutoff are an error."""
    if not np.isfinite(minimum_capacity) or minimum_capacity < 0:
        raise ValueError(f"--minimum-capacity must be finite and nonnegative (Mt/yr); got {minimum_capacity}")
    required = ["latitude", "longitude", "country", "lcoa_eur_per_t", CAPACITY,
                "annual_ammonia_production_t", "currency", "grid_energy_mwh",
                "is_gridless_feasible", "is_full_year_result", "preferred_supplier_capacity_column",
                *ELECTRICITY_COSTS]
    missing = set(required) - set(frame)
    if missing:
        raise ValueError(f"result table (--surface) lacks the columns {sorted(missing)}")
    if frame[["latitude", "longitude"]].duplicated().any():
        duplicated = frame.loc[frame[["latitude", "longitude"]].duplicated(), ["latitude", "longitude"]]
        raise ValueError(f"result table (--surface) has {len(duplicated)} repeated cells, e.g. "
                         f"{duplicated.head(3).to_dict('records')}")
    for name in ("is_gridless_feasible", "is_full_year_result"):
        if not frame[name].astype(str).str.lower().isin(["true", "1"]).all():
            raise ValueError(f"result table column {name} must be true in every row; "
                             f"{int((~frame[name].astype(str).str.lower().isin(['true', '1'])).sum())} rows are not")
    if not frame.currency.eq("EUR").all():
        raise ValueError(f"result table column currency must be EUR, the model currency; found {sorted(frame.currency.astype(str).unique())}")
    if not frame.preferred_supplier_capacity_column.eq(CAPACITY).all():
        raise ValueError(f"result table column preferred_supplier_capacity_column must be {CAPACITY}; "
                         f"found {sorted(frame.preferred_supplier_capacity_column.astype(str).unique())}")
    numeric = ["latitude", "longitude", "lcoa_eur_per_t", CAPACITY, "annual_ammonia_production_t", *ELECTRICITY_COSTS]
    bad = {column: int((~np.isfinite(frame[column].to_numpy(float))).sum()) for column in numeric}
    bad = {column: count for column, count in bad.items() if count}
    if bad:
        raise ValueError(f"result table has NaN or infinite values (column: rows): {bad}")
    if (frame.lcoa_eur_per_t <= 0).any() or (frame[CAPACITY] < 0).any():
        raise ValueError(f"result table has {int((frame.lcoa_eur_per_t <= 0).sum())} rows with lcoa_eur_per_t <= 0 "
                         f"and {int((frame[CAPACITY] < 0).sum())} with {CAPACITY} < 0")
    if not np.allclose(frame.annual_ammonia_production_t, 1e6, rtol=0, atol=1):
        raise ValueError("result table column annual_ammonia_production_t must be 1e6 t in every row (the network "
                         "scales the 1 Mt/yr reference design); found "
                         f"{frame.annual_ammonia_production_t.min():.6g} to {frame.annual_ammonia_production_t.max():.6g}")
    reference = country_reference[["country", "iso3"]].drop_duplicates()
    if reference.country.duplicated().any():
        raise ValueError(f"--country-reference gives more than one iso3 for {sorted(reference.country[reference.country.duplicated()].unique())}")
    positive = frame[(frame[CAPACITY] > 0) & (frame[CAPACITY] >= minimum_capacity * 1e6)].copy()
    if drop_unassigned:
        positive = positive[positive.country.notna()].copy()
    iso = positive.country.map(reference.set_index("country").iso3)
    if iso.isna().any():
        unknown = positive.loc[iso.isna(), "country"].fillna("<unassigned>").astype(str)
        raise ValueError(f"--country-reference has no iso3 for {sorted(unknown.unique())} "
                         "(--drop-unassigned drops cells without a country)")
    # Match existing GPO grid IDs exactly, without rounding non-integer cells.
    def coordinate(value):
        return str(int(value)) if float(value).is_integer() else format(value, ".12g")
    ids = positive.latitude.map(coordinate) + "_" + positive.longitude.map(coordinate) + "_1000000.0"
    electricity_fraction = positive[ELECTRICITY_COSTS].sum(axis=1) / positive.lcoa_eur_per_t
    if not electricity_fraction.between(-1e-8, 1 + 1e-8).all():
        raise ValueError(f"Electricity_Cost_Frac (wind and PV cost over LCOA) lies outside 0-1 in "
                         f"{int((~electricity_fraction.between(-1e-8, 1 + 1e-8)).sum())} cells")
    return pd.DataFrame({
        "Index": ids, "Latitude": positive.latitude, "Longitude": positive.longitude,
        "iso3": iso, "country": positive.country,
        "LCOA": positive.lcoa_eur_per_t / EUR2020_PER_USD2018,
        "Production": 1e6,
        "Max_capacity": positive[CAPACITY] / 1e6,
        "Electricity_Cost_Frac": electricity_fraction,
    }).reset_index(drop=True)


def verify_derivation(provenance, qa, surface_sha256):
    """A post-hoc derivative (derive_land_share) is exported against its parent's QA report: the QA must identify the parent surface and the provenance must identify
    this file.  Returns the derivation record for the supplier table's provenance file."""
    if provenance.get("kind") != "post_hoc_land_share_variant":
        raise ValueError(f"--derived-provenance has kind {provenance.get('kind')!r}; only post_hoc_land_share_variant can be exported")
    if provenance["parent_surface"]["sha256"] != qa.get("output_sha256"):
        raise ValueError(f"--derived-provenance was derived from a result table with sha256 {provenance['parent_surface']['sha256']}, "
                         f"but the check report (--qa) is for {qa.get('output_sha256')}")
    if provenance["output"]["sha256"] != surface_sha256:
        raise ValueError(f"--derived-provenance describes a table with sha256 {provenance['output']['sha256']}, "
                         f"but --surface has {surface_sha256}")
    return {"kind": provenance["kind"], "parent_surface": provenance["parent_surface"],
            "multiplier": provenance["multiplier"],
            "effective_land_competition_fraction": provenance["effective_land_competition_fraction"],
            "columns_rescaled": provenance["columns_rescaled"], "lcoa_columns_untouched": provenance["lcoa_columns_untouched"]}


def check_table_identity(frame, manifest):
    """The run's scenario and run ids from its run record (manifest.json), checked against every row of its result table."""
    identity = {"scenario_id": manifest["scenario"]["id"], "run_id": manifest["run_id"]}
    for key, value in identity.items():
        if not frame[key].eq(value).all():
            found = sorted(frame[key].astype(str).unique())[:5]
            raise ValueError(f"result table (--surface) column {key} holds {found}, but its run record (--manifest) says {value!r}")
    return identity


def export(args):
    qa = json.loads(args.qa.read_text())
    manifest = json.loads(args.manifest.read_text())
    if qa.get("status") != "passed":
        raise ValueError(f"check report (--qa) has status {qa.get('status')!r}; only a passed result table can be exported")
    derivation = None
    derived_path = getattr(args, "derived_provenance", None)
    surface_sha256 = digest(args.surface)
    if derived_path is not None:
        derivation = verify_derivation(json.loads(Path(derived_path).read_text()), qa, surface_sha256)
        derivation["provenance_path"] = str(Path(derived_path).resolve())
        derivation["provenance_sha256"] = digest(Path(derived_path))
    elif qa.get("output_sha256") != surface_sha256:
        raise ValueError(f"check report (--qa) was written for a result table with sha256 {qa.get('output_sha256')}; "
                         f"--surface has {surface_sha256}")
    if qa.get("manifest_sha256") != digest(args.manifest):
        raise ValueError(f"check report (--qa) was written for a run record with sha256 {qa.get('manifest_sha256')}; "
                         f"--manifest has {digest(args.manifest)}")
    if not args.allow_diagnostic and qa.get("stage") != "global":
        raise ValueError(f"check report (--qa) is for the {qa.get('stage')!r} stage; only a global result table can be "
                         "exported without --allow-diagnostic")
    if qa.get("full_year_required") is not True:
        raise ValueError("check report (--qa) does not require a full year (full_year_required is not true); "
                         "only full-year results can be exported")
    frame = pd.read_csv(args.surface)
    if len(frame) != qa["expected_row_count"] or len(frame) != qa["row_count"]:
        raise ValueError(f"result table (--surface) has {len(frame)} rows; its check report expects "
                         f"{qa['expected_row_count']} and counted {qa['row_count']}")
    identity = check_table_identity(frame, manifest)
    # The ids the provenance file records: a derived table (the 2 % land share of a 20 % run) is
    # exported under its own scenario id and names the run it came from
    provenance_ids = dict(identity)
    if getattr(args, "scenario_id_override", None):
        provenance_ids["scenario_id"] = args.scenario_id_override
        provenance_ids["parent_scenario_id"] = identity["scenario_id"]
    minimum = getattr(args, "minimum_capacity", 0.)
    drop_unassigned = bool(getattr(args, "drop_unassigned", False))
    unassigned_mask = frame.country.isna() & (frame[CAPACITY] >= minimum * 1e6) & (frame[CAPACITY] > 0)
    supplier = convert(frame, pd.read_csv(args.country_reference), minimum, drop_unassigned=drop_unassigned)
    args.output.mkdir(parents=True, exist_ok=False)
    destination = args.output / SUPPLIER_TABLE
    supplier.to_csv(destination, index=False)
    report = {
        "schema_version": 1, **provenance_ids,
        "scope": "diagnostic_only" if qa["stage"] != "global" else "global",
        "inputs": {name: {"path": str(path.resolve()), "sha256": digest(path)} for name, path in
                   [("surface", args.surface), ("qa", args.qa), ("manifest", args.manifest), ("country_reference", args.country_reference)]},
        "exporter_sha256": digest(__file__),
        "currency": "USD", "price_year": 2018, "source_currency": "EUR", "source_price_year": 2020,
        "source_to_output_factor": 1 / EUR2020_PER_USD2018,
        "conversion_basis": "Inverse of Way YAML documented USD2018-to-EUR2020 factor 0.9024; no live FX conversion",
        "capacity_source_column": CAPACITY, "capacity_units": "Mt_NH3_per_year (1 Mt reference multiplier)",
        "capacity_rule": manifest["execution"]["capacity_rule"],
        "electricity_cost_fraction_basis": "Wind + fixed-PV + tracking-PV annualised generation cost / headline LCOA; excludes storage and conversion plant",
        "source_rows": len(frame), "exported_positive_capacity_rows": len(supplier),
        "excluded_zero_capacity_rows": int((frame[CAPACITY] == 0).sum()),
        "minimum_capacity_Mt_per_year": minimum,
        "excluded_positive_below_cutoff_rows": int(((frame[CAPACITY] > 0) & (frame[CAPACITY] < minimum * 1e6)).sum()),
        "derivation": derivation,
        "scenario_id_override": getattr(args, "scenario_id_override", None),
        "unassigned_country_policy": "dropped" if drop_unassigned else "rejected",
        "dropped_unassigned_rows_at_or_above_cutoff": int(unassigned_mask.sum()) if drop_unassigned else 0,
        "dropped_unassigned_capacity_Mt_per_year": float(frame.loc[unassigned_mask, CAPACITY].sum() / 1e6) if drop_unassigned else 0.0,
        "supplier_selection": "Explicit export capacity cutoff applied; downstream cheapest-site ranking not yet applied",
        "output": str(destination.resolve()), "output_sha256": digest(destination),
    }
    write_json(args.output / SUPPLIER_PROVENANCE, report)
    print(json.dumps(report, indent=2))


def derive_land_share(surface: Path, multiplier: float, parent_share: float, output_dir: Path) -> dict:
    """Rescale the scaled-design capacities of an after_solve surface to another land share.

    Writes global_run_results.csv and provenance.json to output_dir (new) and returns the provenance.
    """
    surface, output_dir = Path(surface), Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    df = pd.read_csv(surface, low_memory=False)
    if "land_constraint" in df.columns and set(df["land_constraint"].dropna().unique()) - {"after_solve"}:
        raise ValueError("Only after_solve surfaces can be rescaled post hoc")
    scaled = [c for c in df.columns if c.startswith(CAPACITY_PREFIX)]
    if not scaled:
        raise ValueError("No scaled_design capacity columns found")
    for c in scaled:
        df[c] = pd.to_numeric(df[c], errors="coerce") * multiplier
    if "scaled_design_renewable_capacity_scale_factor" in df.columns:
        df["scaled_design_renewable_capacity_scale_factor"] = pd.to_numeric(df["scaled_design_renewable_capacity_scale_factor"], errors="coerce") * multiplier
    df["land_share_multiplier_applied"] = multiplier
    df["land_competition_fraction_effective"] = parent_share * multiplier
    out = output_dir / "global_run_results.csv"
    df.to_csv(out, index=False)
    provenance = {
        "schema_version": 1,
        "kind": "post_hoc_land_share_variant",
        "parent_surface": {"path": str(surface.resolve()), "sha256": digest(surface)},
        "parent_land_competition_fraction": parent_share,
        "multiplier": multiplier,
        "effective_land_competition_fraction": parent_share * multiplier,
        "columns_rescaled": scaled + (["scaled_design_renewable_capacity_scale_factor"] if "scaled_design_renewable_capacity_scale_factor" in df.columns else []),
        "lcoa_columns_untouched": True,
        "justification": "land_constraint=after_solve: the reference design is solved without land; capacity scales linearly with the land share",
        "output": {"path": str(out.resolve()), "sha256": digest(out), "rows": int(len(df))},
    }
    (output_dir / "provenance.json").write_text(json.dumps(provenance, indent=2))
    return provenance


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    p = commands.add_parser("export", help="a QA-passed surface as a supplier table of the network model")
    for name in ("surface", "qa", "manifest", "country-reference", "output"):
        p.add_argument(f"--{name}", type=Path, required=True)
    p.add_argument("--allow-diagnostic", action="store_true")
    p.add_argument("--derived-provenance", type=Path, default=None,
                   help="provenance.json of a post-hoc land-share variant; the QA report then identifies the parent surface")
    p.add_argument("--scenario-id-override", default=None, help="scenario id recorded for a derived surface (e.g. the 2 %% variant)")
    p.add_argument("--drop-unassigned", action="store_true",
                   help="Drop cells without a country polygon (NaN country) instead of rejecting the export; counted in the provenance file")
    p.add_argument("--minimum-capacity", type=float, default=0., help="Explicit minimum annual capacity in Mt; raw surface stays unchanged")
    d = commands.add_parser("derive", help="a land-share variant of an after_solve surface")
    d.add_argument("--surface", type=Path, required=True, help="merged global_run_results.csv of the parent run")
    d.add_argument("--multiplier", type=float, required=True, help="capacity multiplier, e.g. 0.1 for 20 %% -> 2 %%")
    d.add_argument("--parent-share", type=float, required=True)
    d.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "export":
        export(args)
    else:
        print(json.dumps(derive_land_share(args.surface, args.multiplier, args.parent_share, args.output_dir), indent=2))


if __name__ == "__main__":
    main()
