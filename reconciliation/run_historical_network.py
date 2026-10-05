#!/usr/bin/env python3
"""Replay pinned Green Porpoise equations with explicit, hashed inputs.

This is an archival replay, not a claim that every archived equation agrees
with the publication. In particular, the old global-demand constraint and
port-activity inequalities are deliberately retained and reported.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd
import xarray as xr


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


def extract_demand(path, scenario, adoption, efficiency=0.0, nh3_lhv=18.6):
    """Vectorised equivalent of the archived Multi/Reg average, no scaling fit."""
    if not 0 < adoption <= 1 or not 0 <= efficiency < 1 or nh3_lhv <= 0:
        raise ValueError("Invalid adoption or efficiency fraction")
    raw = pd.read_csv(path)
    ssp = "1" if scenario == "2.6" else "2"
    names = [f"SSP{ssp}-RCP{scenario}-{trade}" for trade in ("Multi", "Reg")]
    part = raw[raw.scenario.isin(names)].copy()
    if set(part.scenario) != set(names):
        raise ValueError("Both Multi and Reg trade scenarios must be present")
    if part.groupby("name").id.nunique().max() != 1:
        raise ValueError("Ambiguous demand port identifiers")
    fuel = part.groupby(["name", "scenario"]).Fuel_tons_route_future.sum().unstack()
    # An absent trade record contributes zero in the historical implementation.
    fuel = fuel.reindex(columns=names).fillna(0).sum(axis=1) / 2
    result = fuel.rename("HFO_t_per_year").reset_index()
    result["id"] = result.name.map(part.drop_duplicates("name").set_index("name").id)
    result["Fuel_consumption"] = result.HFO_t_per_year * (39 / nh3_lhv) * adoption * (1 - efficiency)
    return result.sort_values(["Fuel_consumption", "name"], ascending=[False, True]).reset_index(drop=True)


def select_suppliers(raw, count=4000, minimum=1.0):
    numeric = ["LCOA", "Production", "Max_capacity", "Latitude", "Longitude"]
    if not np.isfinite(raw[numeric].to_numpy()).all():
        raise ValueError("Non-finite supplier inputs")
    if not (raw.Production == 1e6).all():
        raise ValueError("Archived capacity equation requires a 1 Mt/year reference plant")
    if (raw.LCOA <= 0).any() or (raw.Max_capacity < 0).any():
        raise ValueError("Invalid supplier cost or capacity")
    # Use the original order of operations, including pandas' default sorting.
    selected = raw.drop(raw.loc[raw.Max_capacity < minimum].index).sort_values("LCOA").head(count)
    duplicates = selected[selected.Index.duplicated(False)]
    # Pyomo sets collapse duplicate IDs; the old dict builder keeps last values.
    # Require identical physical/economic data before reproducing that behaviour.
    for _, group in duplicates.groupby("Index"):
        if (group[numeric].nunique() > 1).any():
            raise ValueError("Duplicate supplier IDs have conflicting physical/economic data")
    effective = selected.drop_duplicates("Index", keep="last").copy()
    effective["subsidy"] = 0.0
    return selected, effective, duplicates


def great_circle_km(suppliers, ports):
    lat1 = np.radians(suppliers.Latitude.to_numpy()[:, None])
    lat2 = np.radians(ports.lat.to_numpy()[None, :])
    dlon = np.radians(suppliers.Longitude.to_numpy()[:, None] - ports.lon.to_numpy()[None, :])
    h = np.sin((lat1-lat2)/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2)**2
    return 6371 * 2 * np.arcsin(np.sqrt(np.clip(h, 0, 1)))


def prepare(args):
    root = args.archive.resolve()
    data = root / "data"
    paths = {
        "suppliers": data / f"c_NH3_cost_{args.scenario}.csv",
        "ports": data / "c_port_data.csv",
        "demand": data / args.demand_file,
        "onshore": data / f"n_onshore_distances_{args.scenario}.nc",
        "offshore": data / f"n_return_costs_{args.scenario}_Panamax.nc",
    }
    contract_path = getattr(args, "supplier_contract", None)
    route_mode = getattr(args, "onshore_routes", "archived")
    contract = None
    if contract_path:
        contract = json.loads(contract_path.read_text())
        paths["suppliers"] = contract_path.parent / "suppliers_USD2018.csv"
        paths["supplier_contract"] = contract_path.resolve()
        if (contract.get("scope") != "global" or contract.get("currency") != "USD"
                or contract.get("price_year") != 2018
                or digest(paths["suppliers"]) != contract.get("output_sha256")):
            raise ValueError("Supplier contract must identify a verified global USD2018 surface")
        if route_mode != "iso3-1000km":
            raise ValueError("New supplier contracts require explicit route regeneration")
    if route_mode != "archived":
        del paths["onshore"]
    hashes = {k: {"path": str(p), "sha256": digest(p)} for k, p in paths.items()}
    sources = {str(p.relative_to(root)): digest(p) for p in sorted((root / "gpo").glob("*.py"))}
    raw = pd.read_csv(paths["suppliers"])
    selected, suppliers, duplicates = select_suppliers(raw, args.suppliers, args.minimum_capacity)
    ports = pd.read_csv(paths["ports"])
    if ports.name.duplicated().any() or ports.id.duplicated().any():
        raise ValueError("Duplicate port identifiers")
    nh3_lhv = 18.8 if args.published_code else 18.6
    demand = extract_demand(paths["demand"], args.scenario, args.adoption, args.efficiency, nh3_lhv)
    missing = demand[~demand.name.isin(ports.name)]
    gc = great_circle_km(suppliers, ports)
    same = suppliers.iso3.to_numpy()[:, None] == ports.iso3.to_numpy()[None, :]
    paper_allowed = same | (gc <= 1000)
    if route_mode == "archived":
        onshore = xr.load_dataset(paths["onshore"]).sel(suppliers=suppliers.Index.tolist(), ports=ports.name.tolist())
    else:
        if suppliers.iso3.isna().any() or ports.iso3.isna().any():
            raise ValueError("Explicit country IDs are required for route regeneration")
        onshore = xr.Dataset({"distances": (("suppliers", "ports"), np.where(paper_allowed, gc, -1.))},
                             coords={"suppliers": suppliers.Index.tolist(), "ports": ports.name.tolist()})
    offshore = xr.load_dataset(paths["offshore"]).sel(supply_ports=ports.name.tolist(), demand_ports=ports.name.tolist())
    if not np.isfinite(onshore.distances.values).all() or not np.isfinite(offshore.shipping_costs.values).all():
        raise ValueError("Non-finite transport tensor")
    if (offshore.shipping_costs.values < 0).any():
        raise ValueError("Negative offshore costs")
    allowed = onshore.distances.values >= 0
    nonzero_allowed = allowed & (gc > 0)
    ratio = onshore.distances.values[nonzero_allowed] / gc[nonzero_allowed]
    report = {
        "schema_version": 1, "endpoint": "archival_equation_replay",
        "scenario": args.scenario, "adoption": args.adoption,
        "vessel_efficiency_improvement": args.efficiency,
        "currency": "USD_2018_as_archived", "hfo_to_nh3_factor": 39 / nh3_lhv,
        "inputs": hashes, "gpo_source_sha256": sources,
        "runner_sha256": digest(__file__),
        "supplier_rows_raw": len(raw), "supplier_rows_selected": len(selected),
        "supplier_ids_effective": len(suppliers), "minimum_capacity_Mt_per_year": args.minimum_capacity,
        "duplicate_selected_rows": duplicates.to_dict("records"),
        "duplicate_policy": "select rows first; identical-physics duplicate IDs collapse; last metadata wins, as archived dict builder",
        "ports": len(ports), "demand_ports": len(demand),
        "demand_t_per_year": float(demand.Fuel_consumption.sum()),
        "model_total_demand_requirement_t_per_year": float(demand.Fuel_consumption.sum()),
        "unmapped_demand_t_per_year": float(missing.Fuel_consumption.sum()),
        "unmapped_demand": missing.to_dict("records"),
        "unmapped_policy": "retained in archived global-demand lower bound; not assigned to a specific port",
        "onshore_banned_fraction": float((~allowed).mean()),
        "onshore_distance_to_great_circle_ratio_quantiles": np.quantile(ratio, [0, .5, 1]).tolist(),
        "onshore_route_disagreements_with_iso3_same_country_or_1000km": int(np.count_nonzero(allowed != paper_allowed)),
        "pipeline_cost_multiplier": args.pipeline_multiplier,
        "onshore_route_mode": route_mode,
        "onshore_regeneration_definition": "Same ISO3 country OR great-circle distance <= 1000 km; radius 6371 km; multiplier applied after eligibility" if route_mode != "archived" else None,
        "supplier_contract": contract,
        "preserved_archival_behaviours": [
            "10 Mt/year upper bound on each supplier, additional to land capacity",
            "port activity is bounded ABOVE by flow/global demand; does not activate storage minimum for positive flows",
            "global demand includes three demand ports missing from the transport port set",
            "no subsidy, no diversification, no forced suppliers",
        ],
        "solver_request": {"threads": args.threads, "relative_gap": args.gap, "time_limit_seconds": args.time_limit},
    }
    if args.published_code:
        published = args.published_code.resolve()
        metadata = json.loads((published / "file_metadata.json").read_text())
        published_hashes = {}
        for entry in metadata:
            name = entry["filename"]
            if Path(name).name != name:
                raise ValueError("Unsafe published file name")
            actual = digest(published / name)
            if actual != entry["content_details"]["sha256_hash"]:
                raise ValueError(f"Published-source checksum mismatch: {name}")
            published_hashes[str(published / name)] = actual
        report.update({
            "endpoint": "deposited_code_replay_with_archived_inputs",
            "published_code_sha256": published_hashes,
            "gpo_source_sha256": {},
            "model_total_demand_requirement_t_per_year": float(demand.loc[demand.name.isin(ports.name), "Fuel_consumption"].sum()),
            "unmapped_policy": "Excluded from individual port demands by deposited code; remains in its total_demand reporting/scaling parameter",
            "preserved_archival_behaviours": [
                "10 Mt/year upper bound on each supplier, additional to land capacity",
                "single port-active binary correctly activated by maritime inflow plus outflow",
                "minimum 1.5 Panamax storage at maritime-active ports; public code does not implement paper's small-port annual-storage cap",
                "no global-demand constraint and no import/export exclusivity constraint",
                "storage cost computed per cubic metre, applied to mass-valued storage variables without density conversion",
                "exact publication input tables were not included in this source-only deposit; recovered project data are explicitly identified",
            ],
        })
    if args.pipeline_multiplier != 1:
        onshore["distances"] = xr.where(onshore.distances >= 0, onshore.distances * args.pipeline_multiplier, onshore.distances)
    return report, suppliers, ports, demand, onshore, offshore


def solve(args, report, suppliers, ports, demand, onshore, offshore):
    import pyomo.environ as pm
    if getattr(args, "published_code", None):
        sys.path.insert(0, str(args.published_code.resolve()))
        mod = importlib.import_module("p_optimisation_parent")
        toolbox = importlib.import_module("p_toolbox")
        if not Path(mod.__file__).resolve().is_relative_to(args.published_code.resolve()):
            raise RuntimeError("Wrong deposited source imported")
        optimizer = mod.Optimiser()
        options = dict(optimizer.opt.options, Threads=args.threads, MIPGap=args.gap)
        instance = optimizer.create_instance(suppliers, ports, onshore, offshore, demand)
        toolbox.fix_variables(instance)
    else:
        sys.path.insert(0, str(args.archive.resolve()))
        configs = importlib.import_module("gpo.configs")
        params = importlib.import_module("gpo.params")
        mod = importlib.import_module("gpo.optimiser")
        if not Path(mod.__file__).resolve().is_relative_to(args.archive.resolve()):
            raise RuntimeError("Wrong GPO source imported")
        active = dict(configs.active_constraints, subsidy=False, supply_diversification=False,
                      cap_port_supply=False, force_production_list=False)
        options = dict(params.optimiser_options, Threads=args.threads, MIPGap=args.gap)
        optimizer = mod.Optimiser(options, params.physical_properties, params.cost_assumptions,
                                 params.init_values, active, suppliers, demand, ports, [], 1.0)
        instance = optimizer.create_instance(suppliers, ports, onshore, offshore, demand, 1.0, [])
        mod.tools.fix_variables(instance)
    # Direct API uses the same Gurobi engine without relying on a gurobi_cl PATH.
    opt = pm.SolverFactory("gurobi_direct")
    opt.options.update(options)
    opt.options["TimeLimit"] = args.time_limit
    opt.options["LogFile"] = str(args.output.resolve() / "gurobi.log")
    started = time.monotonic()
    result = opt.solve(instance, tee=True, load_solutions=False)
    if not len(result.solution):
        result.write(filename=str(args.output / "solver_results.json"), format="json")
        raise RuntimeError(f"No feasible solution: {result.solver.termination_condition}")
    # Pyomo's JSON writer prunes zero entries from the solution dictionaries.
    # Load first, otherwise inactive production variables remain uninitialised.
    instance.solutions.load_from(result)
    result.write(filename=str(args.output / "solver_results.json"), format="json")
    objective = next(instance.component_data_objects(pm.Objective, active=True))
    objective_value = float(pm.value(objective))
    def flows(variable, columns):
        records = [(*key, float(pm.value(value))) for key, value in variable.items()
                   if pm.value(value) > 1e-5]
        return pd.DataFrame(records, columns=[*columns, "tonnes_per_year"])
    land_flows = flows(instance.onshore_transport, ["supplier", "port"])
    sea_flows = flows(instance.offshore_transport, ["supply_port", "demand_port"])
    supply_result = suppliers.copy()
    supply_result["production_t_per_year"] = [pm.value(instance.supplier_production[k]) for k in suppliers.Index]
    supply_result["storage_tonnes"] = [pm.value(instance.supplier_storage[k]) for k in suppliers.Index]
    port_result = ports.copy()
    port_result["demand_t_per_year"] = [pm.value(instance.demands[k]) for k in ports.name]
    port_result["storage_tonnes"] = [pm.value(instance.port_storage[k]) for k in ports.name]
    port_result["onshore_in_t_per_year"] = ports.name.map(land_flows.groupby("port").tonnes_per_year.sum()).fillna(0)
    port_result["offshore_in_t_per_year"] = ports.name.map(sea_flows.groupby("demand_port").tonnes_per_year.sum()).fillna(0)
    port_result["offshore_out_t_per_year"] = ports.name.map(sea_flows.groupby("supply_port").tonnes_per_year.sum()).fillna(0)
    port_result["surplus_t_per_year"] = (port_result.onshore_in_t_per_year + port_result.offshore_in_t_per_year
                                        - port_result.offshore_out_t_per_year - port_result.demand_t_per_year)
    costs = {
        "production": float((supply_result.production_t_per_year * supply_result.LCOA).sum()),
        "pipeline": float(sum(row.tonnes_per_year * pm.value(instance.onshore_distances[row.supplier, row.port])
                              for row in land_flows.itertuples()) * pm.value(instance.onshore_specific_cost)),
        "shipping": float(sum(row.tonnes_per_year * pm.value(instance.offshore_costs[row.supply_port, row.demand_port])
                              for row in sea_flows.itertuples())),
        "storage": float((supply_result.storage_tonnes.sum() + port_result.storage_tonnes.sum()) * pm.value(instance.storage_cost)),
    }
    total = float(supply_result.production_t_per_year.sum())
    lower = float(result.problem.lower_bound)
    upper = float(result.problem.upper_bound)
    closure = abs(sum(costs.values()) - objective_value)
    balance = supply_result.production_t_per_year.to_numpy() - suppliers.Index.map(land_flows.groupby("supplier").tonnes_per_year.sum()).fillna(0).to_numpy()
    summary = {
        "solver_status": str(result.solver.status), "termination": str(result.solver.termination_condition),
        "solve_seconds": time.monotonic() - started, "lower_bound_USD_per_year": lower,
        "upper_bound_USD_per_year": upper, "relative_gap": abs(upper-lower) / max(abs(upper), 1),
        "annual_cost_USD": objective_value, "cost_components_USD_per_year": costs,
        "objective_closure_USD_per_year": closure,
        "total_production_t_per_year": total, "demand_t_per_year": report["demand_t_per_year"],
        "delivered_cost_USD_per_demand_tonne": objective_value / report["demand_t_per_year"],
        "cost_USD_per_produced_tonne": objective_value / total,
        "active_suppliers_above_one_tonne": int((supply_result.production_t_per_year > 1).sum()),
        "onshore_routes": len(land_flows), "offshore_routes": len(sea_flows),
        "maximum_supplier_balance_error_t_per_year": float(np.abs(balance).max()),
        "maximum_port_shortfall_t_per_year": float(max(0, -port_result.surplus_t_per_year.min())),
        "total_port_surplus_t_per_year": float(port_result.surplus_t_per_year.sum()),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"), "slurm_cluster": os.environ.get("SLURM_CLUSTER_NAME"),
    }
    summary["qa_pass"] = bool(closure < max(1, objective_value * 1e-8) and np.abs(balance).max() < 1
                              and summary["maximum_port_shortfall_t_per_year"] < 1
                              and total >= report.get("model_total_demand_requirement_t_per_year", report["demand_t_per_year"]) - 1
                              and summary["relative_gap"] <= args.gap * 1.001)
    for name, frame in [("suppliers", supply_result), ("ports", port_result), ("onshore_flows", land_flows), ("offshore_flows", sea_flows)]:
        frame.to_csv(args.output / f"{name}.csv", index=False)
    country = supply_result.groupby(["iso3", "country"]).production_t_per_year.sum().sort_values(ascending=False)
    country.to_csv(args.output / "country_production.csv")
    write_json(args.output / "summary.json", summary)
    # Check immutable sources and inputs again after execution.
    for entry in report["inputs"].values():
        if digest(entry["path"]) != entry["sha256"]:
            raise RuntimeError("Input changed during solve")
    for relative, expected in report["gpo_source_sha256"].items():
        if digest(args.archive / relative) != expected:
            raise RuntimeError("Archived source changed during solve")
    for path, expected in report.get("published_code_sha256", {}).items():
        if digest(path) != expected:
            raise RuntimeError("Deposited source changed during solve")
    print(json.dumps(summary, indent=2), flush=True)
    if not summary["qa_pass"]:
        raise RuntimeError("Solution failed QA; retain outputs for diagnosis")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--published-code", type=Path, help="Run the checksum-verified Mendeley deposit instead of the later project equations; uses the deposited 18.8 GJ/t demand conversion")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--scenario", choices=["4.5", "2.6"], default="4.5")
    parser.add_argument("--adoption", type=float, default=.7)
    parser.add_argument("--efficiency", type=float, default=0)
    parser.add_argument("--demand-file", default="port_fuel_demand/fuel_demand_port_2050_1000km.csv")
    parser.add_argument("--suppliers", type=int, default=4000)
    parser.add_argument("--minimum-capacity", type=float, default=1)
    parser.add_argument("--pipeline-multiplier", type=float, default=1)
    parser.add_argument("--supplier-contract", type=Path, help="Explicit QA-gated global USD2018 supplier contract; requires regenerated routes")
    parser.add_argument("--onshore-routes", choices=["archived", "iso3-1000km"], default="archived")
    parser.add_argument("--threads", type=int, default=12)
    parser.add_argument("--gap", type=float, default=.001)
    parser.add_argument("--time-limit", type=int, default=36000)
    parser.add_argument("--solver-memory-gb", type=float, default=8, help="Gurobi memory ceiling for the sparse implementation")
    parser.add_argument("--audit-only", action="store_true")
    parser.add_argument("--sparse", action="store_true", help="Use the independently parity-tested sparse matrix implementation of the same equations")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    try:
        prepared = prepare(args)
        report, suppliers, ports, demand, _, _ = prepared
        if args.sparse:
            report["engine"] = "sparse_gurobi_equivalent"
            report["sparse_engine_sha256"] = digest(Path(__file__).with_name("sparse_network.py"))
        write_json(args.output / "manifest.json", report)
        suppliers.to_csv(args.output / "selected_supplier_inputs.csv", index=False)
        demand.to_csv(args.output / "demand_inputs.csv", index=False)
        print(json.dumps({k: v for k, v in report.items() if k not in {"inputs", "gpo_source_sha256", "duplicate_selected_rows"}}, indent=2), flush=True)
        if not args.audit_only:
            if args.sparse:
                from sparse_network import solve_sparse
                solve_sparse(args, *prepared)
            else:
                solve(args, *prepared)
    except Exception as exc:
        write_json(args.output / "failure.json", {"error": repr(exc)})
        raise


if __name__ == "__main__":
    main()
