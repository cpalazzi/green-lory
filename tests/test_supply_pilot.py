"""reconciliation/supply_pilot: the finite-site supply-curve experiment's own checks.

Removed with reconciliation/ (CAMPAIGN_PLAN.md step 6).
"""
from __future__ import annotations

import pandas as pd
import pytest

from model.plant import OptimizationFailure
from model.surface import CellFailure
from reconciliation.supply_pilot import run_supply_pilot
from reconciliation.supply_pilot.run_supply_pilot import ROOT, TECH_YAML, is_proven_infeasible, tech_yaml_dependencies


def test_bathymetry_dependency_is_pinned_before_preflight(tmp_path):
    path = tmp_path / "model_bathymetry.nc"
    with pytest.raises(FileNotFoundError, match="bathymetry dependency missing"):
        run_supply_pilot.validate_bathymetry_input(path, "absent")
    path.write_bytes(b"controlled test fixture")
    with pytest.raises(ValueError, match="Bathymetry differs"):
        run_supply_pilot.validate_bathymetry_input(path, "wrong")
    assert run_supply_pilot.validate_bathymetry_input(path, run_supply_pilot.sha(path)) == path


def test_only_proven_solver_infeasibility_is_classified():
    error = RuntimeError("wrapped")
    error.__cause__ = OptimizationFailure("warning", "infeasible")
    assert is_proven_infeasible(error)
    for condition in ("infeasible_or_unbounded", "numeric", "time_limit"):
        assert not is_proven_infeasible(OptimizationFailure("warning", condition))
    assert not is_proven_infeasible(RuntimeError("licence failure: infeasible"))


def _failure(status, condition):
    error = CellFailure("solver", f"Optimisation failed: {condition}")
    error.__cause__ = OptimizationFailure(status, condition)
    return error


def test_numerical_retry_preserves_an_infeasibility_certificate(monkeypatch):
    calls = []

    def run(inputs, lat, lon, *, quiet=True, solver_options_override=None):
        calls.append(solver_options_override)
        raise _failure("ok", "suboptimal") if len(calls) == 1 else _failure("warning", "infeasible")

    monkeypatch.setenv("GREEN_LORY_SOLVER", "gurobi")
    monkeypatch.setattr(run_supply_pilot, "run_cell", run)
    attempts = []
    with pytest.raises(CellFailure) as caught:
        run_supply_pilot.run_location_with_retry(None, 0.0, 0.0, attempts=attempts)
    assert is_proven_infeasible(caught.value)
    assert calls == [None, run_supply_pilot.ROBUST_GUROBI_OPTIONS]
    assert [x["termination"] for x in attempts] == ["suboptimal", "infeasible"]


def test_an_unknown_failure_is_not_retried(monkeypatch):
    calls = []

    def run(*args, **kwargs):
        calls.append(1)
        raise CellFailure("setup", "license unavailable")

    monkeypatch.setattr(run_supply_pilot, "run_cell", run)
    with pytest.raises(CellFailure, match="license"):
        run_supply_pilot.run_location_with_retry(None, 0.0, 0.0, attempts=[])
    assert len(calls) == 1


@pytest.mark.parametrize("retry_condition", ["optimal", "suboptimal"])
def test_the_retry_accepts_only_an_optimum_and_is_bounded(monkeypatch, retry_condition):
    calls = []

    def run(inputs, lat, lon, *, quiet=True, solver_options_override=None):
        calls.append(solver_options_override)
        if len(calls) == 1 or retry_condition != "optimal":
            raise _failure("ok", "suboptimal")
        return {"solver_status": "ok", "solver_termination": "optimal"}

    monkeypatch.setenv("GREEN_LORY_SOLVER", "gurobi")
    monkeypatch.setattr(run_supply_pilot, "run_cell", run)
    attempts = []
    if retry_condition == "optimal":
        result = run_supply_pilot.run_location_with_retry(None, 0.0, 0.0, attempts=attempts)
        assert result["solver_termination"] == "optimal"
    else:
        with pytest.raises(CellFailure) as caught:
            run_supply_pilot.run_location_with_retry(None, 0.0, 0.0, attempts=attempts)
        assert not is_proven_infeasible(caught.value)
    assert len(calls) == 2
    assert [x["termination"] for x in attempts] == ["suboptimal", retry_condition]


def test_a_point_is_saved_with_unit_suffixes(monkeypatch, tmp_path):
    raw = {"wind": 1.0, "solar": 2.0, "solar_tracking": 3.0, "accumulated_penalty": 0.0,
           "lcoa_eur_per_t": 10.0, "total_cost_eur_per_year": 1e7,
           "solver_status": "ok", "solver_termination": "optimal"}
    state = {name: None for name in ("base", "weather", "interest", "tech", "meta")}
    state["land_lookup"] = {(0.0, 0.0): None}
    monkeypatch.setattr(run_supply_pilot, "STATE", state)
    monkeypatch.setattr(run_supply_pilot, "set_quantity", lambda *args: None)
    monkeypatch.setattr(run_supply_pilot, "run_location_with_retry", lambda *args, **kwargs: raw)

    def validate(result, *args, **kwargs):
        assert result["accumulated_penalty_mwh"] == 0
        assert (result["wind_mw"], result["solar_mw"], result["solar_tracking_mw"]) == (1, 2, 3)
        return {}

    monkeypatch.setattr(run_supply_pilot, "validate_result", validate)
    output = run_supply_pilot.solve_point({"id": "test", "latitude": 0.0, "longitude": 0.0},
                                          1.0, tmp_path / "point", constrained=False)
    assert output["status"] == "feasible"
    assert pd.read_csv(tmp_path / "point/result.csv").wind_mw.iloc[0] == 1


def test_the_pilot_yaml_chain_stays_inside_the_repository():
    chain = tech_yaml_dependencies(ROOT / TECH_YAML, ROOT)
    assert [path.name for path in chain] == [
        "tech_config_ammonia_plant_2050_way_eur_explicit_compressor_dea_tank.yaml",
        "tech_config_ammonia_plant_2050_way_eur.yaml",
    ]


@pytest.mark.parametrize("parent,match", [
    ("../outside.yaml", "outside release root"),
    ("/absolute.yaml", "relative path"),
    ("a.yaml", "Circular"),
    ("[]", "relative path"),
])
def test_yaml_dependencies_reject_nonportable_or_circular_parents(tmp_path, parent, match):
    path = tmp_path / "a.yaml"
    path.write_text(f"extends: {parent}\n")
    with pytest.raises(ValueError, match=match):
        tech_yaml_dependencies(path, tmp_path)


def test_a_missing_parent_is_an_error(tmp_path):
    path = tmp_path / "a.yaml"
    path.write_text("extends: missing.yaml\n")
    with pytest.raises(FileNotFoundError):
        tech_yaml_dependencies(path, tmp_path)
