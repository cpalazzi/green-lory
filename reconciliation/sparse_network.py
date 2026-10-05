"""Sparse matrix implementation of explicitly selected, pinned GPO equations.

Only banned pipeline variables are eliminated. Bounds, cost coefficients,
storage rules and binary inequalities are retained, including known historical
defects. Validate against the original Pyomo implementation before use.
"""
from __future__ import annotations

import importlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix


def solve_sparse(args, report, suppliers, ports, demand, onshore, offshore):
    import gurobipy as gp
    try:
        from .run_historical_network import digest, write_json
    except ImportError:
        from run_historical_network import digest, write_json
    published = bool(getattr(args, "published_code", None))
    if published:
        sys.path.insert(0, str(args.published_code.resolve()))
        module = importlib.import_module("p_optimisation_parent")
        original = module.Optimiser()
    else:
        sys.path.insert(0, str(args.archive.resolve()))
        module = importlib.import_module("gpo.optimiser")
        config = importlib.import_module("gpo.configs")
        params = importlib.import_module("gpo.params")
        active = dict(config.active_constraints, subsidy=False, supply_diversification=False,
                      cap_port_supply=False, force_production_list=False)
        original = module.Optimiser(params.optimiser_options, params.physical_properties,
                                    params.cost_assumptions, params.init_values, active,
                                    suppliers, demand, ports, [], 1.)
    expected_root = args.published_code.resolve() if published else args.archive.resolve()
    if not Path(module.__file__).resolve().is_relative_to(expected_root):
        raise RuntimeError("Wrong source module for sparse coefficient extraction")
    ns, np_ = len(suppliers), len(ports)
    si, pi = np.nonzero(onshore.distances.values >= 0)
    nland = len(si)
    land_index = np.arange(nland)
    from_supplier = coo_matrix((np.ones(nland), (si, land_index)), shape=(ns, nland)).tocsr()
    into_port = coo_matrix((np.ones(nland), (pi, land_index)), shape=(np_, nland)).tocsr()
    sea_index = np.arange(np_ * np_)
    sea_source = np.repeat(np.arange(np_), np_)
    sea_target = np.tile(np.arange(np_), np_)
    sea_in = coo_matrix((np.ones(len(sea_index)), (sea_target, sea_index)), shape=(np_, len(sea_index))).tocsr()
    sea_out = coo_matrix((np.ones(len(sea_index)), (sea_source, sea_index)), shape=(np_, len(sea_index))).tocsr()
    wanted = ports.name.map(demand.set_index("name").Fuel_consumption).fillna(0).to_numpy()
    total_wanted = float(demand.Fuel_consumption.sum())
    model = gp.Model("gpo-deposited" if published else "gpo-archival")
    for key, value in {"Threads": args.threads, "MIPGap": args.gap, "TimeLimit": args.time_limit,
                       "NumericFocus": 1, "Presolve": 2, "Method": 1, "NodeMethod": 1, "PreSparsify": 1,
                       "NodefileStart": .5, "LogFile": str(args.output.resolve()/"gurobi.log")}.items():
        model.setParam(key, value)
    model.setParam("MemLimit", getattr(args, "solver_memory_gb", 8))
    production = model.addMVar(ns, lb=0, ub=1e7, name="production")
    land = model.addMVar(nland, lb=0, ub=1e8, name="pipeline")
    sea = model.addMVar(np_*np_, lb=0, ub=1e8, name="shipping")
    supplier_storage = model.addMVar(ns, lb=-gp.GRB.INFINITY, name="supplier_storage")
    port_storage = model.addMVar(np_, lb=-gp.GRB.INFINITY, name="port_storage")
    supplier_active = model.addMVar(ns, vtype=gp.GRB.BINARY, name="supplier_active")
    model.addConstr(production <= suppliers.Max_capacity.to_numpy() * 1e6, name="capacity")
    model.addConstr(production == from_supplier @ land, name="supplier_balance")
    model.addConstr((production - 50) * 1e-7 <= supplier_active, name="supplier_activation")
    model.addConstr(into_port @ land + (sea_in-sea_out) @ sea >= wanted, name="port_demand")
    model.addConstr(supplier_storage == production / 52, name="supplier_storage_week")
    model.addConstr(port_storage >= (into_port @ land + sea_in @ sea) / 52, name="port_storage_week")
    if published:
        port_active = model.addMVar(np_, vtype=gp.GRB.BINARY, name="port_active")
        model.addConstr((sea_in + sea_out) @ sea / total_wanted <= port_active, name="port_activation")
        model.addConstr(port_storage >= 1.5 * 82618 * port_active, name="ship_storage")
    else:
        port_supply = model.addMVar(np_, vtype=gp.GRB.BINARY, name="port_active_s")
        port_demand = model.addMVar(np_, vtype=gp.GRB.BINARY, name="port_active_d")
        # Deliberately retain reversed implications for the archival replay.
        model.addConstr(sea_out @ sea / total_wanted >= port_supply, name="archival_export_activation")
        model.addConstr(sea_in @ sea / total_wanted >= port_demand, name="archival_import_activation")
        model.addConstr(port_supply + port_demand <= 1, name="archival_no_reexport")
        model.addConstr(port_storage >= 1.5 * params.init_values["G_ship_size"] * port_supply)
        model.addConstr(port_storage >= 1.5 * params.init_values["G_ship_size"] * port_demand)
        model.addConstr(land.sum() >= total_wanted, name="archival_global_demand")
    pipeline_cost = onshore.distances.values[si, pi] * original.G_onshore_specific_cost
    shipping_cost = offshore.shipping_costs.values.ravel()
    storage_cost = original.G_storage_costs
    model.setObjective(suppliers.LCOA.to_numpy() @ production + pipeline_cost @ land
                       + shipping_cost @ sea + storage_cost * (supplier_storage.sum() + port_storage.sum()))
    heuristic_state = None
    heuristic_sha = None
    if published:
        try:
            from .network_heuristic import deposited_rounding_callback
        except ImportError:
            from network_heuristic import deposited_rounding_callback
        heuristic_sha = digest(Path(__file__).with_name("network_heuristic.py"))
        callback, heuristic_state = deposited_rounding_callback(
            gp, production, land, sea, supplier_storage, port_storage,
            supplier_active, port_active, into_port, sea_in, sea_out)
        model.optimize(callback)
    else:
        model.optimize()
    if model.SolCount < 1:
        raise RuntimeError(f"No feasible solution: Gurobi status {model.Status}")
    prod, land_x, sea_x = production.X, land.X, sea.X
    supplier_store, port_store = supplier_storage.X, port_storage.X
    country_frame = suppliers.copy()
    country_frame["production_t_per_year"] = prod
    country_frame["storage_tonnes"] = supplier_store
    land_keep, sea_keep = land_x > 1e-5, sea_x > 1e-5
    land_frame = pd.DataFrame({"supplier": suppliers.Index.to_numpy()[si[land_keep]],
                               "port": ports.name.to_numpy()[pi[land_keep]], "tonnes_per_year": land_x[land_keep]})
    sea_frame = pd.DataFrame({"supply_port": ports.name.to_numpy()[sea_source[sea_keep]],
                              "demand_port": ports.name.to_numpy()[sea_target[sea_keep]], "tonnes_per_year": sea_x[sea_keep]})
    port_frame = ports.copy()
    port_frame["demand_t_per_year"] = wanted
    port_frame["storage_tonnes"] = port_store
    port_frame["onshore_in_t_per_year"] = into_port @ land_x
    port_frame["offshore_in_t_per_year"] = sea_in @ sea_x
    port_frame["offshore_out_t_per_year"] = sea_out @ sea_x
    port_frame["surplus_t_per_year"] = (into_port @ land_x + (sea_in-sea_out) @ sea_x - wanted)
    costs = {"production": float(suppliers.LCOA.to_numpy() @ prod), "pipeline": float(pipeline_cost @ land_x),
             "shipping": float(shipping_cost @ sea_x), "storage": float(storage_cost * (supplier_store.sum()+port_store.sum()))}
    balance = float(np.max(np.abs(prod - from_supplier @ land_x)))
    shortage = float(max(0, -port_frame.surplus_t_per_year.min()))
    capacity_excess = float(max(0, np.max(prod - suppliers.Max_capacity.to_numpy()*1e6)))
    maritime = (sea_in + sea_out) @ sea_x
    port_storage_shortfall = float(max(0, np.max((into_port @ land_x + sea_in @ sea_x)/52 - port_store)))
    activation_violation = 0.
    if published:
        port_frame["maritime_active"] = port_active.X
        activation_violation = float(max(0, np.max(maritime/total_wanted - port_active.X)))
        port_storage_shortfall = max(port_storage_shortfall, float(max(0, np.max(1.5*82618*port_active.X-port_store))))
    closure = abs(sum(costs.values()) - model.ObjVal)
    import os
    summary = {
        "engine": "sparse_gurobi_equivalent", "engine_sha256": digest(__file__),
        "rounding_heuristic": heuristic_state, "rounding_heuristic_sha256": heuristic_sha,
        "equations": "deposited" if published else "archived_project",
        "solver_version": list(gp.gurobi.version()), "solver_status": model.Status, "relative_gap": model.MIPGap,
        "lower_bound_USD_per_year": model.ObjBound, "upper_bound_USD_per_year": model.ObjVal,
        "annual_cost_USD": model.ObjVal, "cost_components_USD_per_year": costs,
        "objective_closure_USD_per_year": closure, "solve_seconds": model.Runtime,
        "total_production_t_per_year": float(prod.sum()), "demand_t_per_year": total_wanted,
        "delivered_cost_USD_per_demand_tonne": model.ObjVal / total_wanted,
        "cost_USD_per_produced_tonne": model.ObjVal / prod.sum(),
        "maximum_supplier_balance_error_t_per_year": balance,
        "maximum_supplier_capacity_excess_t_per_year": capacity_excess,
        "maximum_port_shortfall_t_per_year": shortage,
        "maximum_port_storage_shortfall_tonnes": port_storage_shortfall,
        "maximum_port_activation_violation": activation_violation,
        "total_port_surplus_t_per_year": float(port_frame.surplus_t_per_year.sum()),
        "active_suppliers_above_one_tonne": int(np.count_nonzero(prod > 1)),
        "onshore_routes": len(land_frame), "offshore_routes": len(sea_frame),
        "model_variables": model.NumVars, "model_constraints": model.NumConstrs,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"), "slurm_cluster": os.environ.get("SLURM_CLUSTER_NAME"),
    }
    summary["qa_pass"] = bool(balance < 1 and shortage < 1 and capacity_excess < 1 and port_storage_shortfall < 1
                              and activation_violation < 1e-6 and closure < max(1, model.ObjVal*1e-8)
                              and model.MIPGap <= args.gap*1.001
                              and prod.sum() >= report.get("model_total_demand_requirement_t_per_year", total_wanted)-1)
    for name, frame in [("suppliers", country_frame), ("ports", port_frame), ("onshore_flows", land_frame), ("offshore_flows", sea_frame)]:
        frame.to_csv(args.output/f"{name}.csv", index=False)
    country_frame.groupby(["iso3", "country"]).production_t_per_year.sum().sort_values(ascending=False).to_csv(args.output/"country_production.csv")
    for entry in report["inputs"].values():
        if digest(entry["path"]) != entry["sha256"]:
            raise RuntimeError("Input changed during solve")
    for relative, expected in report["gpo_source_sha256"].items():
        if digest(args.archive/relative) != expected:
            raise RuntimeError("Archived source changed during solve")
    for path, expected in report.get("published_code_sha256", {}).items():
        if digest(path) != expected:
            raise RuntimeError("Published source changed during solve")
    write_json(args.output/"summary.json", summary)
    print(json.dumps(summary, indent=2), flush=True)
    model.dispose()
    if not summary["qa_pass"]:
        raise RuntimeError("Sparse result failed QA")
