"""Feasible rounding heuristic for the deposited GPO storage formulation.

Fixing an LP's flows and rounding activity upwards remains feasible if port
storage is raised to the active-port requirement. This changes the search
strategy, not the feasible set or objective. Gurobi validates each candidate.
"""
import numpy as np


def deposited_rounding_callback(gp, production, land, sea, supplier_storage,
                                port_storage, supplier_active, port_active,
                                into_port, sea_in, sea_out):
    state = {"attempts": 0, "accepted_objectives": [], "error": None}

    def callback(model, where):
        if (where != gp.GRB.Callback.MIPNODE or state["attempts"] >= 3
                or model.cbGet(gp.GRB.Callback.MIPNODE_STATUS) != gp.GRB.OPTIMAL):
            return
        state["attempts"] += 1
        try:
            prod_x = model.cbGetNodeRel(production)
            land_x = model.cbGetNodeRel(land)
            sea_x = model.cbGetNodeRel(sea)
            active = (((sea_in + sea_out) @ sea_x) > 1e-8).astype(float)
            storage = np.maximum((into_port @ land_x + sea_in @ sea_x) / 52,
                                 1.5 * 82618 * active)
            candidates = [
                (production, prod_x), (land, land_x), (sea, sea_x),
                (supplier_active, (prod_x > 50).astype(float)),
                (port_active, active), (supplier_storage, prod_x / 52),
                (port_storage, storage),
            ]
            for variables, values in candidates:
                model.cbSetSolution(variables.tolist(), values.tolist())
            objective = model.cbUseSolution()
            if objective < gp.GRB.INFINITY:
                state["accepted_objectives"].append(objective)
        except Exception as exc:
            # Callback failures must be visible, without invalidating the
            # unchanged optimisation model or pretending a seed was accepted.
            state["error"] = repr(exc)
            state["attempts"] = 3
            print(f"Rounding heuristic unavailable: {exc}", flush=True)

    return callback, state
