"""Debug: build one legacy network (stated_4h_mean, Atacama) and print what the solver sees."""
import sys, importlib
from pathlib import Path
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_legacy_cells as h
src = h.DEFAULT_SOURCE
aux, plc, legacy_main = h.load_frozen_modules(src)
costs = pd.read_excel(src / "GeneralSteelData_20230831.xlsx", sheet_name="Costs").set_index("Equipment")[2050]
effs = pd.read_excel(src / "GeneralSteelData_20230831.xlsx", sheet_name="Efficiencies").set_index("Equipment")[2050]
print("costs:\n", costs); print("effs:\n", effs)
n = legacy_main.generate_network(8760, str(src / "Basic_ammonia_plant"), costs=costs, efficiencies=effs, aggregation_count=4, time_step=1.0)
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
print("snapshots:", len(n.snapshots), n.snapshots[:3])
print("loads:\n", n.loads); print("loads_t.p_set:\n", n.loads_t.p_set.head())
print("generators:\n", n.generators[["bus","p_nom","p_nom_extendable","capital_cost","marginal_cost"]])
print("links:\n", n.links[["bus0","bus1","bus2","efficiency","efficiency2","p_nom_extendable","capital_cost","marginal_cost","p_nom_max","ramp_limit_up"]])
print("stores:\n", n.stores[["bus","e_nom_extendable","e_nom_max","capital_cost","e_cyclic"]])
weather = h.WeatherShim(Path(sys.argv[1]))
loc = plc.renewable_data(weather, -23.0, -69.0, n.generators.index.to_list())
frame = aux.aggregate_data(loc.concat.drop(columns="Weights"), 4).reset_index(drop=True)
print("profiles:\n", frame.describe())
for name, s in frame.items():
    n.generators_t.p_max_pu[name] = s
print("p_max_pu columns:", list(n.generators_t.p_max_pu.columns), n.generators_t.p_max_pu.shape)
n.lopf(solver_name="gurobi", pyomo=True, extra_functionality=aux.pyomo_constraints, solver_options={"Threads": 4})
import pyomo.environ as pm
print("n.objective:", n.objective)
print("model objective value:", pm.value(n.model.objective))
print("p_nom_opt gens:", n.generators.p_nom_opt.to_dict())
print("p_nom_opt links:", n.links.p_nom_opt.to_dict())
print("e_nom_opt stores:", n.stores.e_nom_opt.to_dict())
print("load served (sum loads_t.p):", n.loads_t.p.sum().to_dict() if len(n.loads_t.p) else "empty")
print("gen p sums:", n.generators_t.p.sum().to_dict())
print(aux.get_results_dict_for_multi_site(n, aggregation_count=4, time_step=1.0))
