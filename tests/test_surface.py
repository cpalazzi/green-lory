"""model/surface.py: run settings and the numerical retry chain (24 Sep 2026 key runs)."""
from __future__ import annotations

import pytest

from model import surface
from model.plant import OptimizationFailure


def _solver_failure(condition="suboptimal"):
    inner = OptimizationFailure("ok", condition)
    error = surface.CellFailure("solver", f"Optimisation failed at (1.0, 2.0): {inner}")
    error.__cause__ = inner
    return error


def _patch_run_cell(monkeypatch, behaviour):
    calls = []

    def run(inputs, lat, lon, *, quiet=True, solver_options_override=None):
        calls.append(solver_options_override)
        return behaviour(len(calls))

    monkeypatch.setattr(surface, "run_cell", run)
    return calls


def test_suboptimal_is_retried_with_crossover(monkeypatch):
    monkeypatch.setenv("GREEN_LORY_SOLVER", "gurobi")

    def behaviour(call):
        if call == 1:
            raise _solver_failure()
        return {"lcoa_eur_per_t": 1.0}

    calls = _patch_run_cell(monkeypatch, behaviour)
    status, results = surface.solve_cell(None, 1.0, 2.0, fail_fast=True)
    assert status == "done"
    assert calls == [None, surface.CROSSOVER_GUROBI_OPTIONS]
    assert results["solver_numerical_retry"] is True
    assert results["solver_retry_attempts"] == 1
    assert results["solver_retry_termination"] == "suboptimal"
    assert "Crossover" in results["solver_retry_options"]


def test_an_optimal_first_solve_is_marked_as_not_retried(monkeypatch):
    _patch_run_cell(monkeypatch, lambda call: {"lcoa_eur_per_t": 1.0})
    status, results = surface.solve_cell(None, 1.0, 2.0, fail_fast=True)
    assert status == "done" and results["solver_numerical_retry"] is False


@pytest.mark.parametrize("fail_fast", [True, False])
def test_non_numerical_failures_follow_the_fail_fast_contract(monkeypatch, fail_fast):
    monkeypatch.setenv("GREEN_LORY_SOLVER", "gurobi")

    def behaviour(call):
        raise surface.CellFailure("solver", "Optimisation failed at (1.0, 2.0): license unavailable")

    calls = _patch_run_cell(monkeypatch, behaviour)
    if fail_fast:
        with pytest.raises(surface.CellFailure, match="license"):
            surface.solve_cell(None, 1.0, 2.0, fail_fast=True)
    else:
        assert surface.solve_cell(None, 1.0, 2.0, fail_fast=False) == ("solver", None)
    assert len(calls) == 1


def test_the_retry_chain_is_gurobi_only_and_bounded(monkeypatch):
    def behaviour(call):
        raise _solver_failure()

    monkeypatch.setenv("GREEN_LORY_SOLVER", "highs")
    calls = _patch_run_cell(monkeypatch, behaviour)
    with pytest.raises(surface.CellFailure, match="suboptimal"):
        surface.solve_cell(None, 1.0, 2.0, fail_fast=True)
    assert calls == [None]

    monkeypatch.setenv("GREEN_LORY_SOLVER", "gurobi")
    calls = _patch_run_cell(monkeypatch, behaviour)
    with pytest.raises(surface.CellFailure, match="suboptimal"):
        surface.solve_cell(None, 1.0, 2.0, fail_fast=True)
    assert calls == [None, surface.CROSSOVER_GUROBI_OPTIONS, surface.ROBUST_GUROBI_OPTIONS]


def test_dual_simplex_follows_a_stalled_crossover(monkeypatch):
    monkeypatch.setenv("GREEN_LORY_SOLVER", "gurobi")

    def behaviour(call):
        if call < 3:
            raise _solver_failure()
        return {"lcoa_eur_per_t": 1.0}

    calls = _patch_run_cell(monkeypatch, behaviour)
    status, results = surface.solve_cell(None, 1.0, 2.0, fail_fast=True)
    assert status == "done" and results["solver_retry_attempts"] == 2
    assert calls == [None, surface.CROSSOVER_GUROBI_OPTIONS, surface.ROBUST_GUROBI_OPTIONS]
    assert "DualReductions" in results["solver_retry_options"]


# --- settings ----------------------------------------------------------------------------------

def test_capacity_rules_require_their_land_constraint():
    with pytest.raises(ValueError, match="requires land_constraint='in_solve'"):
        surface.RunSettings(capacity_rule="solved_quantity", land_constraint="after_solve")
    with pytest.raises(ValueError, match="requires land_constraint='after_solve'"):
        surface.RunSettings(capacity_rule="scaled_reference_design", land_constraint="in_solve")
    surface.RunSettings(capacity_rule="solved_quantity", land_constraint="in_solve")


@pytest.mark.parametrize("time_step", [0.5, 7.0])
def test_the_time_step_must_divide_the_year_in_whole_hours(time_step):
    with pytest.raises(ValueError, match="time_step"):
        surface.RunSettings(time_step=time_step)


def test_a_surface_needs_a_land_table():
    with pytest.raises(ValueError, match="land_csv is required"):
        surface.run_surface([(-23.0, -69.0)], land_csv=None)
