"""model/plant.py: component names, solver options, time accounting, ramp limits, the solve contract."""
from __future__ import annotations

import copy
import pickle
from types import SimpleNamespace

import pandas as pd
import pytest

from model import plant
from model.config import PLANTS_DIR

PLANT_DIR = PLANTS_DIR / "ammonia_plant_1mt"


def test_older_component_names_map_to_bundle_names():
    assert plant.normalize_component_name("battery_storage") == "battery_storage"
    assert plant.normalize_component_name("Battery") == "battery_storage"
    assert plant.normalize_component_name("BatteryStorage") == "battery_storage"
    assert plant.normalize_component_name("HB") == "ammonia_synthesis"


# --- solver options ---------------------------------------------------------------------------

def test_gurobi_defaults_and_threads(monkeypatch):
    monkeypatch.delenv("GREEN_LORY_SOLVER_OPTIONS_JSON", raising=False)
    monkeypatch.setenv("GREEN_LORY_SOLVER_LOG", "0")
    opts = plant.solver_options("gurobi", solver_threads=4)
    assert opts["Method"] == 2 and opts["Crossover"] == 0 and opts["BarConvTol"] == 1e-4 and opts["Threads"] == 4
    assert opts["OutputFlag"] == 0


def test_environment_json_changes_the_first_attempt(monkeypatch):
    monkeypatch.setenv("GREEN_LORY_SOLVER_OPTIONS_JSON", '{"Crossover": 1, "BarConvTol": 1e-8, "BarHomogeneous": 1}')
    opts = plant.solver_options("gurobi", solver_threads=2)
    assert opts["Crossover"] == 1 and opts["BarConvTol"] == 1e-8 and opts["BarHomogeneous"] == 1 and opts["Method"] == 2


def test_override_wins_over_environment(monkeypatch):
    monkeypatch.setenv("GREEN_LORY_SOLVER_OPTIONS_JSON", '{"Crossover": 1}')
    opts = plant.solver_options("gurobi", solver_threads=2, override={"Method": 1, "Crossover": 0})
    assert opts["Method"] == 1 and opts["Crossover"] == 0


def test_bad_json_is_rejected(monkeypatch):
    monkeypatch.setenv("GREEN_LORY_SOLVER_OPTIONS_JSON", "[1, 2]")
    with pytest.raises(ValueError):
        plant.solver_options("gurobi")


def test_highs_ignores_gurobi_environment(monkeypatch):
    monkeypatch.setenv("GREEN_LORY_SOLVER_OPTIONS_JSON", '{"Crossover": 1}')
    assert plant.solver_options("highs", solver_threads=3) == {"solver": "ipm", "threads": 3}


# --- time accounting --------------------------------------------------------------------------

def _raw_store_cost():
    return pd.read_csv(PLANT_DIR / "stores.csv", index_col=0).at["compressed_hydrogen_store", "capital_cost"]


def test_snapshot_weighted_mode_uses_physical_duration():
    network = plant.build_network(4, PLANT_DIR, time_step=4.0, temporal_accounting_mode="snapshot_weighted")
    assert len(network.snapshots) == 4
    assert (network.snapshot_weightings["objective"] == 4.0).all()
    assert (network.snapshot_weightings["stores"] == 4.0).all()
    assert network.stores.at["compressed_hydrogen_store", "capital_cost"] == _raw_store_cost()
    assert network._temporal_accounting_mode == "snapshot_weighted"


def test_legacy_mode_scales_store_costs_by_the_time_step():
    network = plant.build_network(4, PLANT_DIR, time_step=4.0, temporal_accounting_mode="legacy_scaled")
    assert network.stores.at["compressed_hydrogen_store", "capital_cost"] == _raw_store_cost() * 4.0
    assert network._temporal_accounting_mode == "legacy_scaled"


def test_unknown_time_accounting_is_rejected():
    with pytest.raises(ValueError, match="temporal_accounting_mode"):
        plant.build_network(4, PLANT_DIR, time_step=4.0, temporal_accounting_mode="aggregated")


def test_store_reporting_scale():
    assert plant.store_reporting_scale(SimpleNamespace(_temporal_accounting_mode="snapshot_weighted"), 4.0) == 1.0
    assert plant.store_reporting_scale(SimpleNamespace(_temporal_accounting_mode="legacy_scaled"), 4.0) == 4.0
    # Networks without the label are historical ones
    assert plant.store_reporting_scale(SimpleNamespace(), 4.0) == 4.0


# --- Haber-Bosch ramp limits ------------------------------------------------------------------

def test_per_hour_ramp_clears_native_limits_and_scales_the_custom_rate():
    network = plant.build_network(4, PLANT_DIR, time_step=4.0, temporal_accounting_mode="snapshot_weighted")
    network._ramp_limit_basis = "per_hour"
    network.links.at["ammonia_synthesis", "ramp_limit_up"] = 0.1
    network.links.at["ammonia_synthesis", "ramp_limit_down"] = 0.15

    plant.prepare_ammonia_ramp_constraints(network)

    assert pd.isna(network.links.at["ammonia_synthesis", "ramp_limit_up"])
    assert pd.isna(network.links.at["ammonia_synthesis", "ramp_limit_down"])
    assert plant._ammonia_ramp_limit_per_snapshot(network, "up") == pytest.approx(0.4)
    assert plant._ammonia_ramp_limit_per_snapshot(network, "down") == pytest.approx(0.6)

    # PyPSA's own link ramp constraints are gone; the duration-aware ones are added instead
    network.optimize.create_model()
    plant.extra_constraints(network, network.snapshots)
    names = set(network.model.constraints)
    assert "Link-ext-p-ramp_limit_up" not in names and "Link-ext-p-ramp_limit_down" not in names
    assert "ammonia_synthesis_ramp_up" in names and "ammonia_synthesis_ramp_down" in names


def test_legacy_ramp_keeps_native_snapshot_rates():
    network = plant.build_network(4, PLANT_DIR, time_step=4.0, temporal_accounting_mode="legacy_scaled")
    network._ramp_limit_basis = "legacy_per_snapshot"
    up_before = network.links.at["ammonia_synthesis", "ramp_limit_up"]
    plant.prepare_ammonia_ramp_constraints(network)
    assert network.links.at["ammonia_synthesis", "ramp_limit_up"] == up_before
    assert plant._ammonia_ramp_limit_per_snapshot(network, "up") == up_before


# --- the solve contract -----------------------------------------------------------------------

def test_suboptimal_ok_is_rejected_before_result_extraction(monkeypatch):
    monkeypatch.setattr(plant, "apply_weather", lambda *args: None)
    monkeypatch.setattr(plant, "prepare_ammonia_ramp_constraints", lambda *args: None)
    network = SimpleNamespace(generators=pd.DataFrame(), optimize=lambda **kwargs: ("ok", "suboptimal"))
    with pytest.raises(plant.OptimizationFailure, match="suboptimal"):
        plant.solve(network, pd.DataFrame(), time_step=1.0)
    plant.require_optimal_termination("ok", "optimal")
    error = pickle.loads(pickle.dumps(plant.OptimizationFailure("ok", "suboptimal")))
    assert error.condition == "suboptimal"


# --- plant variants ---------------------------------------------------------------------------

def test_ammonia_output_changes_only_the_load():
    network = plant.build_network(4, PLANT_DIR, time_step=1.0, temporal_accounting_mode="snapshot_weighted")
    original = copy.deepcopy(network)
    altered = plant.set_ammonia_output(network, 0.5)
    assert altered.loads.at["ammonia", "p_set"] == pytest.approx(0.5e6 * 6.25 / 8760)
    pd.testing.assert_frame_equal(network.loads, original.loads)
    for table in ("generators", "links", "stores"):
        pd.testing.assert_frame_equal(getattr(altered, table), getattr(original, table))


def test_fixed_policy_disables_tracking_and_nothing_else():
    network = SimpleNamespace(generators=pd.DataFrame(
        {"p_nom": [0.0, 0.0, 2.0], "p_nom_extendable": [True, True, True], "p_nom_max": [100.0, 100.0, 100.0]},
        index=["wind", "solar", "solar_tracking"],
    ))
    before = network.generators.copy()
    plant.restrict_pv(network, "both")
    pd.testing.assert_frame_equal(network.generators, before)
    plant.restrict_pv(network, "fixed")
    pd.testing.assert_frame_equal(network.generators.loc[["wind", "solar"]], before.loc[["wind", "solar"]])
    assert network.generators.loc["solar_tracking", "p_nom"] == 0
    assert not network.generators.loc["solar_tracking", "p_nom_extendable"]
    assert network.generators.loc["solar_tracking", "p_nom_max"] == 0
    with pytest.raises(ValueError):
        plant.restrict_pv(network, "invalid")


def test_unit_suffixes():
    named = plant.add_unit_suffixes({"wind": 1.0, "ammonia": 2.0, "penalty_link": 3.0,
                                     "hydrogen_storage_capacity": 4.0, "lcoa_eur_per_t": 5.0})
    assert named == {"wind_mw": 1.0, "ammonia_mwh": 2.0, "penalty_link_mwh": 3.0,
                     "hydrogen_storage_capacity_t": 4.0, "lcoa_eur_per_t": 5.0}
