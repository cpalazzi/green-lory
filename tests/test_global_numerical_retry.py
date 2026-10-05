"""Fresh-network retry of the global runner after a suboptimal Gurobi termination (24 Sep 2026 key runs)."""
from __future__ import annotations

import pytest

from model import run_global
from model.main import OptimizationFailure


def _failure(condition="suboptimal"):
    try:
        raise OptimizationFailure("ok", condition)
    except OptimizationFailure as inner:
        return RuntimeError(f"Optimisation failed at (1.0, 2.0): {inner}"), inner


def test_suboptimal_is_retried_once_with_robust_options(monkeypatch):
    calls = []

    def run(**kwargs):
        calls.append(kwargs["solver_options_override"])
        if len(calls) == 1:
            error, inner = _failure()
            raise error from inner
        return "done", {"lcoa_eur_per_t": 1.0}

    monkeypatch.setenv("GREEN_LORY_SOLVER", "gurobi")
    monkeypatch.setattr(run_global, "_run_single_location", run)
    status, results = run_global._run_single_location_with_numerical_retry(fail_fast=True, lat=1.0, lon=2.0)
    assert status == "done"
    assert calls == [None, run_global.CROSSOVER_GUROBI_OPTIONS]
    assert results["solver_numerical_retry"] is True
    assert results["solver_retry_attempts"] == 1
    assert results["solver_retry_termination"] == "suboptimal"
    assert "Crossover" in results["solver_retry_options"]


def test_optimal_first_solve_is_marked_as_not_retried(monkeypatch):
    monkeypatch.setattr(run_global, "_run_single_location", lambda **kwargs: ("done", {"lcoa_eur_per_t": 1.0}))
    status, results = run_global._run_single_location_with_numerical_retry(fail_fast=True, lat=1.0, lon=2.0)
    assert status == "done" and results["solver_numerical_retry"] is False


@pytest.mark.parametrize("fail_fast", [True, False])
def test_non_numerical_failures_follow_the_fail_fast_contract(monkeypatch, fail_fast):
    calls = []

    def run(**kwargs):
        calls.append(1)
        raise RuntimeError("Optimisation failed at (1.0, 2.0): license unavailable")

    monkeypatch.setenv("GREEN_LORY_SOLVER", "gurobi")
    monkeypatch.setattr(run_global, "_run_single_location", run)
    if fail_fast:
        with pytest.raises(RuntimeError, match="license"):
            run_global._run_single_location_with_numerical_retry(fail_fast=True, lat=1.0, lon=2.0)
    else:
        assert run_global._run_single_location_with_numerical_retry(fail_fast=False, lat=1.0, lon=2.0) == ("solver", None)
    assert len(calls) == 1


def test_retry_is_not_used_with_highs_and_is_bounded(monkeypatch):
    calls = []

    def run(**kwargs):
        calls.append(kwargs["solver_options_override"])
        error, inner = _failure()
        raise error from inner

    monkeypatch.setenv("GREEN_LORY_SOLVER", "highs")
    monkeypatch.setattr(run_global, "_run_single_location", run)
    with pytest.raises(RuntimeError, match="suboptimal"):
        run_global._run_single_location_with_numerical_retry(fail_fast=True, lat=1.0, lon=2.0)
    assert calls == [None]
    monkeypatch.setenv("GREEN_LORY_SOLVER", "gurobi")
    calls.clear()
    with pytest.raises(RuntimeError, match="suboptimal"):
        run_global._run_single_location_with_numerical_retry(fail_fast=True, lat=1.0, lon=2.0)
    assert calls == [None, run_global.CROSSOVER_GUROBI_OPTIONS, run_global.ROBUST_GUROBI_OPTIONS]


def test_second_retry_uses_dual_simplex_after_crossover_also_stalls(monkeypatch):
    calls = []

    def run(**kwargs):
        calls.append(kwargs["solver_options_override"])
        if len(calls) < 3:
            error, inner = _failure()
            raise error from inner
        return "done", {"lcoa_eur_per_t": 1.0}

    monkeypatch.setenv("GREEN_LORY_SOLVER", "gurobi")
    monkeypatch.setattr(run_global, "_run_single_location", run)
    status, results = run_global._run_single_location_with_numerical_retry(fail_fast=True, lat=1.0, lon=2.0)
    assert status == "done" and results["solver_retry_attempts"] == 2
    assert calls == [None, run_global.CROSSOVER_GUROBI_OPTIONS, run_global.ROBUST_GUROBI_OPTIONS]
    assert "DualReductions" in results["solver_retry_options"]
