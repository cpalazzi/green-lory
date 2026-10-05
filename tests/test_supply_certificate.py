from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from scipy.optimize import linprog

from reconciliation.land.certify_supply_pilot import area_energy_certificate, conversion_certificate, classify_missing, profile_diagnostics


def tables():
    root = Path(__file__).resolve().parents[1]/"basic_ammonia_plant_2050_way_tracking"
    return {name: pd.read_csv(root/(name+".csv")).set_index("name") for name in ("generators", "links", "stores", "loads")}


def test_energy_potentials_match_conversion_inputs():
    certificate = conversion_certificate(**tables())
    assert certificate["electricity_floor_MWh_per_t"] == pytest.approx(8.99991878255043)
    assert max(certificate["link_potential_changes"].values()) <= 1e-10


@pytest.mark.parametrize("change", ["grid", "battery", "initial_storage", "reverse_link"])
def test_reject_energy_sources_or_unsupported_topologies(change):
    data = tables()
    if change == "grid": data["generators"].loc["grid", "p_nom"] = 1
    elif change == "battery": data["links"].loc["battery_pcs_discharge", "efficiency"] = 1.1
    elif change == "initial_storage": data["stores"].loc["ammonia", "e_cyclic"] = False
    else: data["links"].loc["hydrogen_fuel_cell", "p_min_pu"] = -1
    with pytest.raises(ValueError): conversion_certificate(**data)


def test_dual_area_bound_matches_independent_linear_program():
    rng = np.random.default_rng(19)
    for _ in range(25):
        solar, wind, union = rng.uniform(.1, 20, 3)
        land = dict(solar_area_km2=solar, wind_onshore_area_km2=wind, renewable_union_area_km2=union,
                    solar_density_mw_per_km2=80., wind_density_mw_per_km2=5.)
        cf = dict(zip(("solar", "solar_tracking", "wind"), rng.uniform(0, 1, 3)))
        certificate = area_energy_certificate(land, cf, 9.)
        yields = np.array([max(80*cf["solar"], 40*cf["solar_tracking"]), 5*cf["wind"]])*8760
        lp = linprog(-yields, A_ub=[[1, 1]], b_ub=[union], bounds=[(0, solar), (0, wind)], method="highs")
        assert lp.success
        assert certificate["ideal_annual_generation_MWh"] == pytest.approx(-lp.fun)
        assert classify_missing(certificate["certificate_bound_Mtpa"], certificate) == "unresolved"
        assert classify_missing(certificate["certificate_bound_Mtpa"]+1e-6, certificate) == "infeasible_analytic"


def test_profile_over_nameplate_is_reported_without_changing_input():
    values = np.full(8760, .3)
    values[0] = 1.12
    check = profile_diagnostics(values, "solar")
    assert check["hours_above_one"] == 1
    assert check["mean"] == pytest.approx(values.mean())
    assert values[0] == 1.12
    with pytest.raises(ValueError): profile_diagnostics(values, "wind")
    with pytest.raises(ValueError): profile_diagnostics(values[:100], "solar")
