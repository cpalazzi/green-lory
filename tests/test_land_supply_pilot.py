import copy
import pickle
from types import SimpleNamespace
from pathlib import Path

import pandas as pd
import pypsa
import pytest

from model.auxiliary import linopy_constraints
from model.land_capacity import RenewableLandBudget, report_land_feasible_quantity
from model.main import OptimizationFailure, generate_network, require_optimal_termination
from model import main as plant_main
from model.run_global import _apply_renewable_land_budget, run_global
from reconciliation.land.run_supply_pilot import is_proven_infeasible, set_quantity
from reconciliation.land import run_supply_pilot


def test_bathymetry_dependency_is_pinned_before_preflight(tmp_path):
    path = tmp_path / "model_bathymetry.nc"
    with pytest.raises(FileNotFoundError, match="bathymetry dependency missing"):
        run_supply_pilot.validate_bathymetry_input(path, "absent")
    path.write_bytes(b"controlled test fixture")
    with pytest.raises(ValueError, match="Bathymetry differs"):
        run_supply_pilot.validate_bathymetry_input(path, "wrong")
    assert run_supply_pilot.validate_bathymetry_input(path, run_supply_pilot.sha(path)) == path


def budget(area=1):
    return RenewableLandBudget(wind_area_km2=area, solar_area_km2=area,
        renewable_union_area_km2=area, wind_density_mw_per_km2=1,
        fixed_solar_density_mw_per_km2=10, tracking_solar_density_mw_per_km2=5,
        union_area_source="explicit", wind_area_source="explicit", tracking_density_source="explicit")


def test_solved_quantity_is_not_scaled_to_maximum():
    output = report_land_feasible_quantity({"annual_ammonia_production_t": 250000,
        "gridless_ammonia_production_t": 250000, "wind_mw": 0,
        "solar_mw": 2, "solar_tracking_mw": 0}, budget())
    assert output["land_feasible_ammonia_production_t"] == 250000
    assert output["land_feasible_gridless_ammonia_production_t"] == 250000
    assert output["renewable_union_land_slack_km2"] == pytest.approx(.8)
    assert output["quantity_is_maximum"] is False
    assert not any("max_" in key for key in output)
    assert output["preferred_supplier_capacity_column"] == ""


def test_solved_quantity_rejects_land_excess():
    with pytest.raises(ValueError, match="exceeds solar"):
        report_land_feasible_quantity({"annual_ammonia_production_t": 1e6,
            "solar_mw": 11, "wind_mw": 0, "solar_tracking_mw": 0}, budget())


def test_solved_quantity_requires_enforced_land():
    with pytest.raises(ValueError, match="requires land_constraint='in_solve'"):
        run_global(capacity_rule="solved_quantity", land_constraint="after_solve")


def test_only_proven_solver_infeasibility_is_classified():
    error = RuntimeError("wrapped")
    error.__cause__ = OptimizationFailure("warning", "infeasible")
    assert is_proven_infeasible(error)
    for condition in ("infeasible_or_unbounded", "numeric", "time_limit"):
        assert not is_proven_infeasible(OptimizationFailure("warning", condition))
    assert not is_proven_infeasible(RuntimeError("licence failure: infeasible"))


def test_suboptimal_ok_is_rejected_before_result_extraction(monkeypatch):
    monkeypatch.setattr(plant_main, "apply_weather_profiles", lambda *args: None)
    monkeypatch.setattr(plant_main.aux, "prepare_ammonia_ramp_constraints", lambda *args: None)
    network = SimpleNamespace(generators=pd.DataFrame(), optimize=lambda **kwargs: ("ok", "suboptimal"))
    with pytest.raises(OptimizationFailure, match="suboptimal"):
        plant_main.main(n=network, multi_site=True)
    require_optimal_termination("ok", "optimal")
    error = pickle.loads(pickle.dumps(OptimizationFailure("ok", "suboptimal")))
    assert error.condition == "suboptimal"


def test_numerical_retry_preserves_infeasible_certificate(monkeypatch):
    calls = []

    def run(*args, **kwargs):
        calls.append(kwargs["solver_options_override"])
        raise OptimizationFailure("ok" if len(calls) == 1 else "warning",
                                  "suboptimal" if len(calls) == 1 else "infeasible")

    monkeypatch.setenv("GREEN_LORY_SOLVER", "gurobi")
    monkeypatch.setattr(run_supply_pilot, "_run_single_location", run)
    attempts = []
    with pytest.raises(OptimizationFailure) as caught:
        run_supply_pilot.run_location_with_retry(attempts=attempts)
    assert is_proven_infeasible(caught.value)
    assert calls == [None, run_supply_pilot.ROBUST_GUROBI_OPTIONS]
    assert [x["termination"] for x in attempts] == ["suboptimal", "infeasible"]


def test_unknown_runtime_failure_is_not_retried(monkeypatch):
    calls = []

    def run(*args, **kwargs):
        calls.append(1)
        raise RuntimeError("license unavailable")

    monkeypatch.setattr(run_supply_pilot, "_run_single_location", run)
    with pytest.raises(RuntimeError, match="license"):
        run_supply_pilot.run_location_with_retry(attempts=[])
    assert len(calls) == 1


@pytest.mark.parametrize("retry_condition", ["optimal", "suboptimal"])
def test_retry_accepts_only_an_optimum_and_is_bounded(monkeypatch, retry_condition):
    calls = []

    def run(*args, **kwargs):
        calls.append(kwargs["solver_options_override"])
        if len(calls) == 1 or retry_condition != "optimal":
            raise OptimizationFailure("ok", "suboptimal")
        return "done", {"solver_status": "ok", "solver_termination": "optimal"}

    monkeypatch.setenv("GREEN_LORY_SOLVER", "gurobi")
    monkeypatch.setattr(run_supply_pilot, "_run_single_location", run)
    attempts = []
    if retry_condition == "optimal":
        status, result = run_supply_pilot.run_location_with_retry(attempts=attempts)
        assert status == "done" and result["solver_termination"] == "optimal"
    else:
        with pytest.raises(OptimizationFailure) as caught:
            run_supply_pilot.run_location_with_retry(attempts=attempts)
        assert not is_proven_infeasible(caught.value)
    assert len(calls) == 2
    assert [x["termination"] for x in attempts] == ["suboptimal", retry_condition]


def test_quantity_changes_only_ammonia_load():
    root = Path(__file__).resolve().parents[1]
    network = generate_network(4, root / "basic_ammonia_plant_2050_way_tracking",
                               time_step=1., temporal_accounting_mode="snapshot_weighted")
    original = copy.deepcopy(network)
    altered = set_quantity(network, .5)
    assert altered.loads.at["ammonia", "p_set"] == pytest.approx(.5e6 * 6.25 / 8760)
    pd.testing.assert_frame_equal(network.loads, original.loads)
    pd.testing.assert_frame_equal(altered.generators, original.generators)
    pd.testing.assert_frame_equal(altered.links, original.links)
    pd.testing.assert_frame_equal(altered.stores, original.stores)


def test_point_wrapper_applies_standard_result_units(monkeypatch, tmp_path):
    raw = {"wind": 1., "solar": 2., "solar_tracking": 3., "accumulated_penalty": 0.,
           "lcoa_eur_per_t": 10., "total_cost_eur_per_year": 1e7,
           "solver_status": "ok", "solver_termination": "optimal"}
    state = {name: None for name in ("base", "weather", "interest", "tech", "meta")}
    state["land_lookup"] = {(0., 0.): None}
    monkeypatch.setattr(run_supply_pilot, "STATE", state)
    monkeypatch.setattr(run_supply_pilot, "set_quantity", lambda *args: None)
    monkeypatch.setattr(run_supply_pilot, "_run_single_location", lambda *args, **kwargs: ("done", raw))

    def validate(result, *args, **kwargs):
        assert result["accumulated_penalty_mwh"] == 0
        assert result["wind_mw"] == 1
        assert result["solar_mw"] == 2
        assert result["solar_tracking_mw"] == 3
        return {}

    monkeypatch.setattr(run_supply_pilot, "validate_result", validate)
    output = run_supply_pilot.solve_point({"id": "test", "latitude": 0., "longitude": 0.},
        1., tmp_path / "point", constrained=False)
    assert output["status"] == "feasible"
    saved = pd.read_csv(tmp_path / "point/result.csv")
    assert saved.wind_mw.iloc[0] == 1


@pytest.mark.parametrize("demand,expected", [(8, "optimal"), (11, "infeasible")])
def test_actual_linopy_shared_land_constraint(demand, expected):
    network = pypsa.Network()
    network.set_snapshots(range(2))
    network.add("Bus", "power")
    for name, cost in (("wind", 1000), ("solar", 2), ("solar_tracking", 1)):
        network.add("Generator", name, bus="power", p_nom_extendable=True, capital_cost=cost)
    network.add("Load", "demand", bus="power", p_set=demand)
    _apply_renewable_land_budget(network, budget(), "exclusive")
    status, condition = network.optimize(solver_name="highs", extra_functionality=linopy_constraints,
                                         solver_options={"log_to_console": False})
    assert condition == expected
    if expected == "optimal":
        assert status == "ok"
        assert network.generators.at["solar", "p_nom_opt"] == pytest.approx(6)
        assert network.generators.at["solar_tracking", "p_nom_opt"] == pytest.approx(2)
        assert network.objective == pytest.approx(14)


def test_fixed_only_policy_disables_tracking_without_changing_other_generators():
    from types import SimpleNamespace
    from reconciliation.land.run_supply_pilot import apply_pv_policy
    import pandas as pd
    import pytest
    network = SimpleNamespace(generators=pd.DataFrame({
        "p_nom": [0., 0., 2.], "p_nom_extendable": [True, True, True],
        "p_nom_max": [100., 100., 100.]}, index=["wind", "solar", "solar_tracking"]))
    before = network.generators.copy()
    apply_pv_policy(network, "both")
    pd.testing.assert_frame_equal(network.generators, before)
    apply_pv_policy(network, "fixed-only")
    pd.testing.assert_frame_equal(network.generators.loc[["wind", "solar"]], before.loc[["wind", "solar"]])
    assert network.generators.loc["solar_tracking", "p_nom"] == 0
    assert not network.generators.loc["solar_tracking", "p_nom_extendable"]
    assert network.generators.loc["solar_tracking", "p_nom_max"] == 0
    with pytest.raises(ValueError): apply_pv_policy(network, "invalid")
