#!/usr/bin/env python
"""Rerun the frozen legacy-lcoa plant model (Salmon, commit cd56c11, June 2023)
at selected cells under explicitly named temporal-accounting variants.

The legacy model code is imported unchanged from ``source/cd56c11``. This
harness replaces only three things that the frozen code cannot do here:

* the outer loop (``run_Alli_sites`` / Carlo's later ``run_global``), which
  hard-codes Windows paths and a fixed 4-process pool;
* the weather loader (``all_locations``), which expects nine full NetCDF files
  in a Windows directory - here a shim exposes the same ``Solars``/``Winds``/
  ``SolarTrackings`` lists from any directory of NetCDF files (full stack or
  the verified three-cell subsets);
* the hard-coded ``solver = 'gurobi'`` inside ``main()`` - the solver is a
  command-line option so the same run can be checked with GLPK/HiGHS.

Everything that defines the model is the frozen code: ``generate_network``
(network, cost overrides, water cost, time scaling), ``pyomo_constraints``
(battery coupling, H2 cycling limit, HB ramp limits), ``renewable_data``
(profile extraction, 0.93 wake factor, block-sum aggregation),
``aggregate_data`` (block-mean aggregation) and
``get_results_dict_for_multi_site`` (headline results and LCOA).

Variants (see VARIANTS below) reproduce the different call patterns that
exist in the frozen source; none of them is edited or "corrected" here. The
WACC axis rescales the annualised capital costs from the workbook's 8 %/20 y/
2 % O&M basis to another rate with the same lifetime and O&M fraction; this is
the only arithmetic added by the harness and it is reported explicitly.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import shutil
import sys
import time
import types
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

HERE = Path(__file__).resolve().parent
DEFAULT_SOURCE = HERE / "source" / "cd56c11"
NH3_HHV_MWH_PER_T = 6.25          # legacy convention (p_auxiliary)
H2_HHV_MWH_PER_T = 39.4
WORKBOOK_RATE, WORKBOOK_YEARS, WORKBOOK_OM = 0.08, 20, 0.02  # "Discount Rate Calculation" sheet

# --------------------------------------------------------------------------
# Variants: each entry documents exactly which frozen call pattern it follows.
# keys: weather_agg ("hourly" | "sum4" | "mean4"), net_n_snapshots,
#       net_aggregation_count, net_time_step (None -> frozen default 0.5),
#       results_aggregation_count, results_time_step
# --------------------------------------------------------------------------
VARIANTS = {
    "as_written_alli": dict(
        doc="run_Alli_sites() as written: generate_network(8760/4, aggregation_count=4) -> "
            "int(2190/4)=547 snapshots, hourly (unaggregated) profiles aligned on the first "
            "547 hours, frozen default time_step=0.5, results scaled with aggregation_count=4.",
        weather_agg="hourly", net_n_snapshots=8760 / 4, net_aggregation_count=4,
        net_time_step=None, results_aggregation_count=4, results_time_step=1.0),
    "stated_4h_mean": dict(
        doc="Stated method (Salmon & Banares-Alcantara 2022: 4-hour time step, 2190 periods): "
            "block-mean profiles via aggregate_data(4), generate_network(8760, aggregation_count=4, "
            "time_step=1.0) -> 2190 snapshots, stores x4, marginal x(8784/8760)x4.",
        weather_agg="mean4", net_n_snapshots=8760, net_aggregation_count=4,
        net_time_step=1.0, results_aggregation_count=4, results_time_step=1.0),
    "legacy_4h_sum": dict(
        doc="HYPOTHETICAL, not a frozen call path: renewable_data(aggregation_variable=4) block-SUM "
            "profiles (p_max_pu up to 4) with the block index reset by the harness (the frozen "
            "aggregate() leaves indices 0,4,8,... which would align to NaN against 2190 snapshots), "
            "generate_network(8760, aggregation_count=4) with frozen default time_step=0.5 "
            "-> 2190 snapshots, stores x2. Off by default.",
        weather_agg="sum4", net_n_snapshots=8760, net_aggregation_count=4,
        net_time_step=None, results_aggregation_count=4, results_time_step=1.0),
    "hourly": dict(
        doc="Full hourly year: generate_network(8760, aggregation_count=1, time_step=1.0), "
            "8760 snapshots, stores x1.",
        weather_agg="hourly", net_n_snapshots=8760, net_aggregation_count=1,
        net_time_step=1.0, results_aggregation_count=1, results_time_step=1.0),
    "single_site_csv_path": dict(
        doc="main(file_name=csv, aggregation_count=4) interactive path: block-mean profiles, "
            "generate_network(len(weather)=2190) with frozen defaults aggregation_count=1, "
            "time_step=0.5 -> 2190 snapshots, stores x0.5, marginal x(8784/2190).",
        weather_agg="mean4", net_n_snapshots=2190, net_aggregation_count=1,
        net_time_step=None, results_aggregation_count=4, results_time_step=1.0),
}


# Variant table for the 3 May 2023 state (commit 94de8ce, source/94de8ce): generate_network there
# has no time_step argument, sets snapshots = int(n_snapshots) and multiplies store capital cost by
# aggregation_count only; its run_Alli_sites solved the full 8,760-hour year.
VARIANTS_MAY2023 = {
    "hourly": dict(
        doc="94de8ce run_Alli_sites() as written: generate_network(8760, aggregation_count=1), hourly "
            "profiles, 8760 snapshots, stores x1, no water cost, no hydrogen-store cycling constraint.",
        weather_agg="hourly", net_n_snapshots=8760, net_aggregation_count=1,
        net_time_step=None, results_aggregation_count=1, results_time_step=None),
    "stated_4h_mean": dict(
        doc="94de8ce code with the papers' 4-hour step: block-mean profiles via aggregate_data(4), "
            "generate_network(2190, aggregation_count=4) -> 2190 snapshots, stores x4.",
        weather_agg="mean4", net_n_snapshots=2190, net_aggregation_count=4,
        net_time_step=None, results_aggregation_count=4, results_time_step=None),
}
ERAS = {"june2023": ("cd56c11", VARIANTS), "may2023": ("94de8ce", VARIANTS_MAY2023)}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def annualisation_factor(rate: float, years: int = WORKBOOK_YEARS, om: float = WORKBOOK_OM) -> float:
    """Annual cost per unit overnight cost as the workbook computes it: the 'Discount Rate
    Calculation' sheet discounts CAPEX plus 20 yearly O&M payments with an annuity-DUE
    (payments at the start of years 0..19, 'Net Time' 10.6036 at 8 %) and divides by that
    same annuity, i.e. factor = 1/annuity_due(rate, years) + om = 0.1143 at 8 %/20 y/2 %."""
    if rate == 0:
        return 1.0 / years + om
    annuity_due = 1 + (1 - (1 + rate) ** (-(years - 1))) / rate
    return 1.0 / annuity_due + om


def wacc_scale(rate: float) -> float:
    """Ratio of the annualised cost at `rate` to the workbook basis (8 %, 20 y, 2 % O&M)."""
    return annualisation_factor(rate) / annualisation_factor(WORKBOOK_RATE)


# -- green-lory annuity convention (key legacy run of 24 Sep 2026) ---------------------------
# Equipment of the workbook's Costs sheet mapped to the green-lory technology YAML, whose
# per-technology lifetimes and fixed O&M fractions replace the single workbook factor.
LEGACY_EQUIPMENT_TO_TECH = {
    "Wind": "wind", "Solar": "solar", "SolarTracking": "solar_tracking", "Electrolysis": "electrolysis",
    "BatteryInterfaceIn": "battery_pcs_charge", "Battery": "battery_storage",
    "CompressedH2Store": "compressed_hydrogen_store", "HydrogenFuelCell": "hydrogen_fuel_cell",
    "HB": "ammonia_synthesis", "Ammonia": "ammonia",
}


def ordinary_annuity(rate: float, years: float) -> float:
    """Capital recovery factor with end-of-year payments, as model/run_global.py computes it."""
    if abs(rate) < 1e-12:
        return 1.0 / years
    growth = (1.0 + rate) ** years
    return rate * growth / (growth - 1.0)


def load_tech_yaml_chain(path: Path) -> dict:
    """Resolve a green-lory tech YAML with its optional relative `extends` chain (techs merged)."""
    import yaml
    raw = yaml.safe_load(open(path)) or {}
    parent = raw.pop("extends", None)
    if parent is None:
        return raw
    base = load_tech_yaml_chain(path.parent / parent)
    techs = dict(base.get("techs", {}))
    for name, values in (raw.get("techs") or {}).items():
        techs[name] = {**techs.get(name, {}), **(values or {})}
    merged = {**base, **raw}
    merged["techs"] = techs
    return merged


def load_annuity_terms(yaml_path: Path) -> dict:
    """{equipment: {lifetime_years, fixed_om_fraction, tech}} from the green-lory tech YAML."""
    techs = load_tech_yaml_chain(yaml_path)["techs"]
    terms = {}
    for equipment, tech in LEGACY_EQUIPMENT_TO_TECH.items():
        if tech not in techs:
            raise SystemExit(f"annuity YAML {yaml_path} has no technology {tech!r} for equipment {equipment!r}")
        terms[equipment] = {"tech": tech, "lifetime_years": float(techs[tech]["lifetime_years"]),
                            "fixed_om_fraction": float(techs[tech]["fixed_om_fraction"])}
    return terms


def annualise_green_lory(costs_workbook_annual: "pd.Series", rate: float, terms: dict, workbook_factor: float):
    """Re-annualise the workbook's annual costs with green-lory's convention.

    The Costs sheet (and the x_Cost reconstruction) carries overnight CAPEX x the workbook factor
    (8 %, 20 years annuity-due, 2 % O&M = 0.1143).  Overnight cost is recovered by dividing by that
    factor, then multiplied by CRF(rate, lifetime) + O&M of the mapped green-lory technology.
    Equipment without a mapping (nuclear marginal costs) is left unchanged."""
    out = costs_workbook_annual.copy()
    factors = {}
    for equipment, term in terms.items():
        if equipment not in out.index:
            continue
        overnight = float(costs_workbook_annual[equipment]) / workbook_factor
        factor = ordinary_annuity(rate, term["lifetime_years"]) + term["fixed_om_fraction"]
        out[equipment] = overnight * factor
        factors[equipment] = {**term, "overnight_usd_per_unit": overnight, "annual_factor": factor,
                              "ratio_vs_workbook_at_this_rate": factor / annualisation_factor(rate)}
    return out, factors


class WeatherShim:
    """Stand-in for p_location_class.all_locations built from a directory of NetCDF files.

    The frozen `renewable_data.get_data_from_nc` indexes `Solars[ref]`, `Winds[ref]`,
    `SolarTrackings[ref]` with ref = 0/1/2 by longitude band (<-60, <60, else). With the
    full nine-file stack the three files per technology are sorted by name so that the
    unsuffixed file is band 0, `1` is band 1 and `2` is band 2 (the frozen glob was not
    sorted; the 16 November 2023 commit added sorting "to correct order"). With a subset
    file that already contains every requested cell, the same dataset is used for all
    three bands.
    """

    def __init__(self, weather_dir: Path):
        self.path = str(weather_dir)
        files = sorted(weather_dir.glob("*.nc"))
        self.file_list = [str(f) for f in files]
        self.Solars, self.Winds, self.SolarTrackings = [], [], []
        for f in files:
            name = f.name
            if name == "model_bathymetry.nc":
                continue
            ds = xr.open_dataset(f)
            if "SolarTracking" in name:
                self.SolarTrackings.append(ds)
            elif "Solar" in name:
                self.Solars.append(ds)
            elif "WindPower" in name:
                self.Winds.append(ds)
        for lst in (self.Solars, self.Winds, self.SolarTrackings):
            if len(lst) == 1:
                lst.extend([lst[0], lst[0]])
            if len(lst) != 3:
                raise SystemExit(f"expected 1 or 3 files per technology in {weather_dir}, got {len(lst)}")
        self.manifest = {"kind": "netcdf_files", "path": self.path,
                         "weather_files": {str(Path(f).name): sha256_file(Path(f)) for f in self.file_list}}


class _Values:
    """Mimics the `.values` attribute of a DataArray."""

    def __init__(self, values):
        self.values = values


class _StoreBand:
    """One band of one technology from a compact store, exposing the two accessors the frozen
    `renewable_data.get_data_from_nc` uses: `.loc[dict(latitude=, longitude=)].<Var>.values`
    and `.time.values`."""

    def __init__(self, var: str, array_path: Path, rows: dict, times):
        self.var = var
        self.array = np.load(array_path, mmap_mode="r", allow_pickle=False)
        self.rows = rows
        self.time = _Values(times)
        self.loc = self

    def __getitem__(self, key):
        lat, lon = float(key["latitude"]), float(key["longitude"])
        if (lat, lon) not in self.rows:
            raise KeyError(f"({lat}, {lon}) is not in this band of the weather store")
        series = np.array(self.array[self.rows[(lat, lon)]], dtype=np.float64)  # one contiguous 70 kB read
        return types.SimpleNamespace(**{self.var: _Values(series)})


class CompactWeather:
    """Stand-in for `all_locations` backed by a store written by extract_weather_store.py.

    Same `Solars`/`Winds`/`SolarTrackings` triples and the same band rule as WeatherShim; the
    values are the unchanged float64 series of the source files, so runs are numerically
    identical to runs on the NetCDF files. The store manifest carries the source hashes.
    """

    def __init__(self, store_dir: Path):
        self.path = str(store_dir)
        info = json.loads((store_dir / "manifest.json").read_text())
        if info.get("schema") != "legacy_weather_store_v1":
            raise SystemExit(f"{store_dir} is not a legacy weather store")
        times = pd.to_datetime(np.load(store_dir / "time.npy", allow_pickle=False)).values
        self.file_list = []
        groups = {"Solar": [], "Wind": [], "SolarTracking": []}
        for name, meta in sorted(info["files"].items()):
            stem = Path(name).stem
            cells = pd.read_csv(store_dir / meta["cells_csv"])
            rows = {(float(a), float(b)): int(i) for i, (a, b) in enumerate(zip(cells.lat, cells.lon))}
            band = _StoreBand(meta["variable"], store_dir / meta["array"], rows, times)
            groups[meta["technology"]].append((meta["band"], band))
            self.file_list.append(str(store_dir / meta["array"]))
        self.Solars = [b for _, b in sorted(groups["Solar"], key=lambda x: x[0])]
        self.Winds = [b for _, b in sorted(groups["Wind"], key=lambda x: x[0])]
        self.SolarTrackings = [b for _, b in sorted(groups["SolarTracking"], key=lambda x: x[0])]
        for lst in (self.Solars, self.Winds, self.SolarTrackings):
            if len(lst) == 1:
                lst.extend([lst[0], lst[0]])
            if len(lst) != 3:
                raise SystemExit(f"expected 1 or 3 arrays per technology in {store_dir}, got {len(lst)}")
        self.manifest = {"kind": "compact_store", "path": self.path,
                         "store_manifest_sha256": sha256_file(store_dir / "manifest.json"),
                         "weather_files": {name: meta["source_sha256"] for name, meta in sorted(info["files"].items())},
                         "arrays": {name: meta["array_sha256"] for name, meta in sorted(info["files"].items())}}


def load_frozen_modules(source: Path):
    sys.path.insert(0, str(source))
    aux = importlib.import_module("p_auxiliary")
    plc = importlib.import_module("p_location_class")
    legacy_main = importlib.import_module("main")
    return aux, plc, legacy_main


def run_cell(cell, variant_name, variant, wacc_name, wacc_rate, args, aux, plc, legacy_main,
             weather, costs, efficiencies, out_dir: Path):
    import pypsa  # noqa: F401  (frozen code imports it too)

    lat, lon = float(cell["lat"]), float(cell["lon"])
    run_id = f"{cell['name']}__{variant_name}__{wacc_name}"
    run_dir = out_dir / run_id
    if run_dir.exists():
        if args.skip_existing and (run_dir / "summary.json").exists():
            return json.load(open(run_dir / "summary.json"))
        if args.skip_existing:   # an interrupted or failed attempt: redo it
            shutil.rmtree(run_dir)
        else:
            raise SystemExit(f"refusing to overwrite existing run directory {run_dir}")
    run_dir.mkdir(parents=True)
    t0 = time.time()

    # -- costs on the requested WACC basis (harness arithmetic, reported) ----------------
    if getattr(args, "annuity", "workbook") == "green_lory":
        costs_used, annuity_factors = annualise_green_lory(costs, wacc_rate, args.annuity_terms, args.workbook_factor)
        scale = None
    else:
        scale = wacc_scale(wacc_rate)
        costs_used = costs * scale
        annuity_factors = None

    # -- frozen network construction ------------------------------------------------------
    import inspect
    gn_params = inspect.signature(legacy_main.generate_network).parameters
    kwargs = dict(costs=costs_used, efficiencies=efficiencies,
                  aggregation_count=variant["net_aggregation_count"])
    if variant["net_time_step"] is not None:
        if "time_step" not in gn_params:
            raise SystemExit("this source's generate_network has no time_step argument; use the may2023 variants")
        kwargs["time_step"] = variant["net_time_step"]
    plant_dir = str(args.source / "Basic_ammonia_plant")
    n = legacy_main.generate_network(variant["net_n_snapshots"], plant_dir, **kwargs)
    if args.enable_tracking is not None:
        n.generators.loc["SolarTracking", "p_nom_extendable"] = True
        if "SolarTracking" in costs_used.index:   # explicit row (xcost reconstruction), already WACC-scaled
            n.generators.loc["SolarTracking", "capital_cost"] = float(costs_used["SolarTracking"])
        else:
            n.generators.loc["SolarTracking", "capital_cost"] = float(n.generators.loc["Solar", "capital_cost"]) * args.enable_tracking
    renewables = n.generators.index.to_list()

    # -- frozen weather extraction --------------------------------------------------------
    if variant["weather_agg"] == "sum4":
        loc = plc.renewable_data(weather, lat, lon, renewables, aggregation_variable=4)
        frame = loc.concat.drop(columns="Weights")
    else:
        loc = plc.renewable_data(weather, lat, lon, renewables)  # aggregation_variable=1
        frame = loc.concat.drop(columns="Weights")
        if variant["weather_agg"] == "mean4":
            if hasattr(aux, "aggregate_data"):
                frame = aux.aggregate_data(frame, 4)
            else:  # harness fallback with the same block-mean definition
                frame = frame.groupby(frame.index // 4).mean()
    frame = frame.reset_index(drop=True)
    if not args.no_timeseries:
        frame.to_csv(run_dir / "profiles_used.csv")

    # -- the body of legacy main(multi_site=True), with the solver made explicit ----------
    count = 0
    for name, series in frame.items():
        if name in renewables:
            n.generators_t.p_max_pu[name] = series
            count += 1
        else:
            raise ValueError(f"profile column {name} is not a generator")
    if count != len(renewables):
        raise ValueError("missing renewable profiles")
    n_used = len(n.snapshots)
    solver_options = {}
    if args.solver == "gurobi":
        solver_options = {"Threads": args.threads, "Method": 2, "Crossover": 0} if args.barrier else {"Threads": args.threads}
    n.lopf(solver_name=args.solver, pyomo=True, extra_functionality=aux.pyomo_constraints,
           solver_options=solver_options, solver_logfile=str(run_dir / "solver.log"))
    rd_params = inspect.signature(aux.get_results_dict_for_multi_site).parameters
    rd_kwargs = {"aggregation_count": variant["results_aggregation_count"]}
    if variant["results_time_step"] is not None and "time_step" in rd_params:
        rd_kwargs["time_step"] = variant["results_time_step"]
    output = aux.get_results_dict_for_multi_site(n, **rd_kwargs)

    # -- dispatch and accounting (read-only post-processing) -------------------------------
    gen_p = n.generators_t.p
    gen_pmax = n.generators_t.p_max_pu * n.generators.p_nom_opt
    hours_per_snapshot = 8760.0 / n_used if variant["weather_agg"] != "hourly" or n_used == 8760 else 1.0
    # In the 547-snapshot as-written case each snapshot is one hour of data (the year is truncated).
    if variant_name == "as_written_alli":
        hours_per_snapshot = 1.0
    prod_t = n.loads.p_set.values[0] / NH3_HHV_MWH_PER_T * 8760  # legacy production basis (t/yr)
    dispatched = {g: float(gen_p[g].sum()) for g in ["Wind", "Solar", "SolarTracking"] if g in gen_p}
    available = {g: float(gen_pmax[g].sum()) for g in ["Wind", "Solar", "SolarTracking"] if g in gen_pmax}
    tot_disp, tot_avail = sum(dispatched.values()), sum(available.values())
    capex_gen = {g: float(n.generators.loc[g, "capital_cost"] * n.generators.loc[g, "p_nom_opt"]) for g in n.generators.index}
    capex_link = {l: float(n.links.loc[l, "capital_cost"] * n.links.loc[l, "p_nom_opt"]) for l in n.links.index}
    capex_store = {s: float(n.stores.loc[s, "capital_cost"] * n.stores.loc[s, "e_nom_opt"]) for s in n.stores.index}
    marginal = float(sum((n.links_t.p0[l] * n.links.loc[l, "marginal_cost"]).sum() for l in n.links.index)
                     + sum((n.generators_t.p[g] * n.generators.loc[g, "marginal_cost"]).sum() for g in n.generators.index))
    electricity_capex = capex_gen.get("Wind", 0) + capex_gen.get("Solar", 0) + capex_gen.get("SolarTracking", 0)

    summary = {
        "run_id": run_id, "cell": cell["name"], "lat": lat, "lon": lon,
        "cell_meta": {k: (None if pd.isna(v) else v) for k, v in cell.items() if k not in ("name", "lat", "lon")},
        "variant": variant_name, "variant_doc": variant["doc"],
        "wacc_name": wacc_name, "wacc_rate": wacc_rate, "capital_cost_scale_vs_workbook": scale,
        "annuity": getattr(args, "annuity", "workbook"), "annuity_factors": annuity_factors,
        "tracking_enabled_cost_ratio": args.enable_tracking,
        "solver": args.solver, "snapshots_used": n_used,
        "objective_usd_per_year": float(n.objective),
        "lcoa_usd_per_kg_legacy_definition": float(output["Objective"]),
        "lcoa_usd_per_t": float(output["Objective"]) * 1000.0,
        "production_basis_t_per_year": prod_t,
        "p_set_mw": float(n.loads.p_set.values[0]),
        "capacities_mw": {k: float(v) for k, v in output.items() if k not in ("Objective",)},
        "capex_generators_usd": capex_gen, "capex_links_usd": capex_link, "capex_stores_usd": capex_store,
        "marginal_cost_total_usd": marginal,
        "electricity_capex_fraction_of_objective": electricity_capex / float(n.objective),
        "dispatched_generation_snapshot_sum": dispatched, "available_generation_snapshot_sum": available,
        "curtailed_fraction": (1 - tot_disp / tot_avail) if tot_avail > 0 else None,
        "hours_per_snapshot_assumed_for_energy": hours_per_snapshot,
        "primary_electricity_mwh_per_t_snapshot_basis": tot_disp * hours_per_snapshot / prod_t,
        "elapsed_s": time.time() - t0,
    }
    with open(run_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    if not args.no_timeseries:
        n.generators_t.p.to_csv(run_dir / "generators_t_p.csv")
        n.links_t.p0.to_csv(run_dir / "links_t_p0.csv")
        n.links_t.p1.to_csv(run_dir / "links_t_p1.csv")
        n.stores_t.e.to_csv(run_dir / "stores_t_e.csv")
    n.generators.to_csv(run_dir / "generators.csv")
    n.links.to_csv(run_dir / "links.csv")
    n.stores.to_csv(run_dir / "stores.csv")
    print(f"[{run_id}] LCOA {summary['lcoa_usd_per_t']:.2f} USD/t  snapshots={n_used}  "
          f"wind={summary['capacities_mw'].get('Wind', 0):.0f} solar={summary['capacities_mw'].get('Solar', 0):.0f} MW  "
          f"({summary['elapsed_s']:.0f}s)", flush=True)
    return summary


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--era", choices=list(ERAS), default="june2023",
                   help="which frozen state: june2023 = cd56c11 (default), may2023 = 94de8ce; sets the default "
                        "--source and the variant table")
    p.add_argument("--source", type=Path, default=None, help="frozen legacy source directory (default from --era)")
    p.add_argument("--weather-dir", type=Path, default=None,
                   help="directory with Solar*.nc, SolarTracking*.nc, WindPowers*.nc (full stack or subsets)")
    p.add_argument("--weather-store", type=Path, default=None,
                   help="compact per-cell store written by extract_weather_store.py (fast on network file "
                        "systems; numerically identical to --weather-dir)")
    p.add_argument("--cells", type=Path, required=True, help="CSV with columns name,lat,lon")
    p.add_argument("--costs-xlsx", type=Path, default=None,
                   help="cost workbook (default: source/GeneralSteelData_20230831.xlsx)")
    p.add_argument("--year", type=int, default=2050)
    p.add_argument("--capex-source", choices=["workbook", "xcost45", "xcost26"], default="workbook",
                   help="workbook: the Costs sheet as is (default). xcost45/xcost26: HARNESS RECONSTRUCTION - "
                        "take 2050 overnight CAPEX ($/W) for Wind, Fixed PV, Single Axis PV, FC, Electrolyser, "
                        "Battery Interface and Battery from the RCP 4.5 / RCP 2.6 sheet of x_Cost Forecasting.xlsx, "
                        "annualise with the workbook's own 8 %% annuity-due factor (0.1143), keep the workbook for "
                        "everything else; SolarTracking then gets its own cost row")
    p.add_argument("--variants", default=None,
                   help="comma-separated subset of the era's variant table (default: all except legacy_4h_sum)")
    p.add_argument("--annuity", choices=["workbook", "green_lory"], default="workbook",
                   help="workbook: single 8 %%/20 y/2 %% factor rescaled to the WACC (replication). green_lory: overnight "
                        "CAPEX x (CRF(WACC, lifetime) + O&M) per equipment with lifetimes and O&M from --annuity-yaml "
                        "(key legacy run of 24 Sep 2026; not the archived arithmetic).")
    p.add_argument("--annuity-yaml", type=Path, default=Path(__file__).resolve().parents[2] / "inputs/tech_config_ammonia_plant_2050_way_eur.yaml",
                   help="green-lory technology YAML supplying lifetimes and fixed O&M fractions for --annuity green_lory")
    p.add_argument("--wacc", default="workbook8:0.08,ameli_reduced:0.051",
                   help="comma-separated name:rate pairs; 'workbook8:0.08' is the unscaled workbook basis")
    p.add_argument("--enable-tracking", type=float, default=None, metavar="COST_RATIO",
                   help="HARNESS MODIFICATION (stated-method reconstruction, not frozen code): make SolarTracking "
                        "extendable at COST_RATIO x the fixed-PV annualised cost (x_Cost Forecasting RCP4.5 2050: "
                        "0.2542/0.2401 = 1.0587); the frozen plant CSV disables tracking")
    p.add_argument("--solver", default="gurobi")
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--barrier", action="store_true", help="gurobi: barrier without crossover")
    p.add_argument("--output", type=Path, required=True, help="new campaign run directory (must not exist)")
    p.add_argument("--no-timeseries", action="store_true",
                   help="do not write dispatch/storage time series per cell (global runs); summaries and "
                        "component tables are still written")
    p.add_argument("--skip-existing", action="store_true",
                   help="skip cells whose run directory already holds a summary.json (resumable global runs)")
    args = p.parse_args()

    commit, variant_table = ERAS[args.era]
    if args.source is None:
        args.source = HERE / "source" / commit
    args.source = args.source.resolve()
    costs_xlsx = (args.costs_xlsx or (args.source / "GeneralSteelData_20230831.xlsx")).resolve()
    resumed = False
    if args.output.exists():
        if not (args.skip_existing and (args.output / "manifest.json").exists()):
            raise SystemExit(f"refusing to overwrite existing output directory {args.output}")
        resumed = True   # resume: solved cells are skipped, the original manifest is kept
    else:
        args.output.mkdir(parents=True)

    aux, plc, legacy_main = load_frozen_modules(args.source)
    costs = pd.read_excel(costs_xlsx, sheet_name="Costs").set_index("Equipment")[args.year]
    efficiencies = pd.read_excel(costs_xlsx, sheet_name="Efficiencies").set_index("Equipment")[args.year]
    capex_note = None
    if args.capex_source != "workbook":
        xcost = (args.costs_xlsx.parent if args.costs_xlsx else args.source) / "x_Cost_Forecasting_20230831.xlsx"
        sheet = "RCP 4.5" if args.capex_source == "xcost45" else "RCP 2.6"
        raw = pd.read_excel(xcost, sheet_name=sheet, header=None)
        header = raw.iloc[0].tolist()
        col = header.index(args.year)
        labels = raw.apply(lambda r: next((v for v in r.iloc[:4] if isinstance(v, str)), None), axis=1)
        capex_per_w = {lab: float(raw.iloc[i, col]) for i, lab in labels.items() if lab in
                       ("Wind", "Fixed PV", "Single Axis PV", "FC", "Electrolyser", "Battery Interface", "Battery")}
        f8 = annualisation_factor(WORKBOOK_RATE)
        mapping = {"Wind": "Wind", "Fixed PV": "Solar", "Single Axis PV": "SolarTracking", "FC": "HydrogenFuelCell",
                   "Electrolyser": "Electrolysis", "Battery Interface": "BatteryInterfaceIn", "Battery": "Battery"}
        costs = costs.copy()
        for lab, equip in mapping.items():
            costs[equip] = capex_per_w[lab] * 1e6 * f8   # $/W -> $/MW overnight, then annualised
        capex_note = {"sheet": sheet, "year": args.year, "capex_usd_per_w": capex_per_w,
                      "annualisation_factor": f8, "file": str(xcost), "sha256": sha256_file(xcost)}
        print("CAPEX reconstruction:", capex_note, flush=True)
    args.workbook_factor = annualisation_factor(WORKBOOK_RATE)
    args.annuity_terms = None
    annuity_note = {"convention": args.annuity}
    if args.annuity == "green_lory":
        args.annuity_terms = load_annuity_terms(args.annuity_yaml)
        annuity_note.update({"yaml": str(args.annuity_yaml), "sha256": sha256_file(args.annuity_yaml),
                             "terms": args.annuity_terms, "workbook_factor_divided_out": args.workbook_factor})
        print("annuity convention green_lory:", {k: (v["lifetime_years"], v["fixed_om_fraction"]) for k, v in args.annuity_terms.items()}, flush=True)
    if (args.weather_dir is None) == (args.weather_store is None):
        raise SystemExit("give exactly one of --weather-dir or --weather-store")
    weather = CompactWeather(args.weather_store) if args.weather_store else WeatherShim(args.weather_dir)
    cells = pd.read_csv(args.cells)
    if args.variants:
        variants = [v.strip() for v in args.variants.split(",") if v.strip()]
    else:
        variants = [v for v in variant_table if v != "legacy_4h_sum"]
    for v in variants:
        if v not in variant_table:
            raise SystemExit(f"variant {v} is not defined for era {args.era}: {list(variant_table)}")
    waccs = []
    for item in args.wacc.split(","):
        name, rate = item.split(":")
        waccs.append((name.strip(), float(rate)))

    manifest = {
        "harness": sha256_file(Path(__file__)),
        "source_dir": str(args.source),
        "source_files": {str(f.relative_to(args.source)): sha256_file(f)
                         for f in sorted(args.source.rglob("*")) if f.is_file()},
        "costs_xlsx": {"path": str(costs_xlsx), "sha256": sha256_file(costs_xlsx), "year": args.year,
                       "costs": {k: float(v) for k, v in costs.items()},
                       "efficiencies": {k: float(v) for k, v in efficiencies.items()}},
        "weather_files": weather.manifest["weather_files"],
        "weather_source": weather.manifest,
        "cells": cells.to_dict(orient="records"),
        "era": args.era, "commit": commit,
        "variants": {v: variant_table[v] for v in variants},
        "wacc": {name: {"rate": rate, "capital_cost_scale_vs_workbook": wacc_scale(rate)} for name, rate in waccs},
        "solver": args.solver, "enable_tracking_cost_ratio": args.enable_tracking,
        "capex_source": args.capex_source, "capex_reconstruction": capex_note,
        "annuity": annuity_note,
        "workbook_annualisation_factor_8pct": annualisation_factor(WORKBOOK_RATE),
        "python": sys.version,
        "packages": {},
        "started": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    for mod in ("pypsa", "pyomo", "pandas", "numpy", "xarray"):
        try:
            manifest["packages"][mod] = importlib.import_module(mod).__version__
        except Exception:  # pragma: no cover
            manifest["packages"][mod] = "unknown"
    manifest_name = f"manifest.resume.{time.strftime('%Y%m%dT%H%M%S')}.json" if resumed else "manifest.json"
    with open(args.output / manifest_name, "w") as f:
        json.dump(manifest, f, indent=2)

    rows = []
    per_cell_wacc = "wacc" in cells.columns
    if per_cell_wacc:
        print("per-cell WACC column found in the cells CSV; --wacc list ignored", flush=True)
    for _, cell in cells.iterrows():
        cell_waccs = [("cell_wacc", float(cell["wacc"]))] if per_cell_wacc else waccs
        for vname in variants:
            for wname, wrate in cell_waccs:
                try:
                    s = run_cell(cell, vname, variant_table[vname], wname, wrate, args, aux, plc, legacy_main,
                                 weather, costs, efficiencies, args.output)
                    rows.append({k: v for k, v in s.items() if not isinstance(v, dict)})
                except Exception as exc:  # keep going, record the failure
                    print(f"[{cell['name']}__{vname}__{wname}] FAILED: {exc!r}", flush=True)
                    rows.append({"cell": cell["name"], "variant": vname, "wacc_name": wname, "error": repr(exc)})
                pd.DataFrame(rows).to_csv(args.output / "results.csv", index=False)
    print("done", flush=True)


if __name__ == "__main__":
    main()
