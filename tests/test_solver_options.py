"""Solver-option assembly: built-in Gurobi defaults, campaign-wide JSON override, per-call retry."""
from __future__ import annotations

import pytest

from model.main import build_solver_options


def test_gurobi_defaults_and_threads(monkeypatch):
    monkeypatch.delenv("GREEN_LORY_SOLVER_OPTIONS_JSON", raising=False)
    monkeypatch.setenv("GREEN_LORY_SOLVER_LOG", "0")
    opts = build_solver_options("gurobi", solver_threads=4)
    assert opts["Method"] == 2 and opts["Crossover"] == 0 and opts["BarConvTol"] == 1e-4 and opts["Threads"] == 4
    assert opts["OutputFlag"] == 0


def test_environment_json_changes_the_first_attempt(monkeypatch):
    monkeypatch.setenv("GREEN_LORY_SOLVER_OPTIONS_JSON", '{"Crossover": 1, "BarConvTol": 1e-8, "BarHomogeneous": 1}')
    opts = build_solver_options("gurobi", solver_threads=2)
    assert opts["Crossover"] == 1 and opts["BarConvTol"] == 1e-8 and opts["BarHomogeneous"] == 1 and opts["Method"] == 2


def test_override_wins_over_environment(monkeypatch):
    monkeypatch.setenv("GREEN_LORY_SOLVER_OPTIONS_JSON", '{"Crossover": 1}')
    opts = build_solver_options("gurobi", solver_threads=2, solver_options_override={"Method": 1, "Crossover": 0})
    assert opts["Method"] == 1 and opts["Crossover"] == 0


def test_bad_json_is_rejected(monkeypatch):
    monkeypatch.setenv("GREEN_LORY_SOLVER_OPTIONS_JSON", "[1, 2]")
    with pytest.raises(ValueError):
        build_solver_options("gurobi")


def test_highs_ignores_gurobi_environment(monkeypatch):
    monkeypatch.setenv("GREEN_LORY_SOLVER_OPTIONS_JSON", '{"Crossover": 1}')
    assert build_solver_options("highs", solver_threads=3) == {"solver": "ipm", "threads": 3}
