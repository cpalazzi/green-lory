import sys, importlib, inspect
from pathlib import Path
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_legacy_cells as h
src = h.DEFAULT_SOURCE
aux, plc, legacy_main = h.load_frozen_modules(src)
import pypsa, pyomo.environ as pm
print("pypsa", pypsa.__version__)
costs = pd.read_excel(src / "GeneralSteelData_20230831.xlsx", sheet_name="Costs").set_index("Equipment")[2050]
effs = pd.read_excel(src / "GeneralSteelData_20230831.xlsx", sheet_name="Efficiencies").set_index("Equipment")[2050]
n = legacy_main.generate_network(168, str(src / "Basic_ammonia_plant"), costs=costs, efficiencies=effs, aggregation_count=1, time_step=1.0)
print("dtypes gens:", n.generators.dtypes.to_dict())
print("snapshot_weightings head:\n", n.snapshot_weightings.head())
weather = h.WeatherShim(Path(sys.argv[1]))
loc = plc.renewable_data(weather, -23.0, -69.0, n.generators.index.to_list())
frame = loc.concat.drop(columns="Weights").reset_index(drop=True).iloc[:168]
for name, s in frame.items():
    n.generators_t.p_max_pu[name] = s
n.lopf(solver_name="gurobi", pyomo=True, extra_functionality=aux.pyomo_constraints, solver_options={"Threads": 2})
obj = n.model.objective
expr = obj.expr
print("objective expr type:", type(expr))
try:
    from pyomo.core.expr import current as ec
except Exception:
    ec = None
s = str(expr)
print("objective expr length:", len(s)); print(s[:1500])
print("n.objective:", n.objective)
# source of define_linear_objective in this pypsa
from pypsa import opf
print(inspect.getsource(opf.define_linear_objective)[:3000])
