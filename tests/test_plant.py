"""model/plant.py: component names, solver options, time accounting, ramp limits, the solve contract."""
from __future__ import annotations

import copy
import pickle
from types import SimpleNamespace

import pandas as pd
import pytest

from model import plant, technology
from model.config import DEFAULT_TECH_YAML, PLANTS_DIR

PLANT_DIR = PLANTS_DIR / "ammonia_plant_1mt"


def _prepared(n_snapshots, time_step, mode):
    """The bundle's network with the campaign YAML's technology values, costs included."""
    tech = technology.load_tech_inputs(DEFAULT_TECH_YAML)
    network = plant.build_network(n_snapshots, PLANT_DIR, time_step=time_step, temporal_accounting_mode=mode)
    technology.prepare_base_network(network, tech)
    technology.apply_base_costs(network, tech, time_step=time_step, temporal_accounting_mode=mode)
    return network


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

def test_snapshot_weighted_mode_uses_physical_duration():
    network = plant.build_network(4, PLANT_DIR, time_step=4.0, temporal_accounting_mode="snapshot_weighted")
    assert len(network.snapshots) == 4
    assert (network.snapshot_weightings["objective"] == 4.0).all()
    assert (network.snapshot_weightings["stores"] == 4.0).all()
    assert network.loads.at["ammonia", "p_set"] == pytest.approx(1e6 * 6.25 / 8760)  # the reference 1 Mt/yr
    assert network._temporal_accounting_mode == "snapshot_weighted"


def test_legacy_mode_scales_store_and_variable_costs():
    weighted, legacy = _prepared(4, 4.0, "snapshot_weighted"), _prepared(4, 4.0, "legacy_scaled")
    assert legacy._temporal_accounting_mode == "legacy_scaled"
    store = "compressed_hydrogen_store"
    assert legacy.stores.at[store, "capital_cost"] == pytest.approx(4.0 * weighted.stores.at[store, "capital_cost"])
    assert weighted.links.at["electrolysis", "marginal_cost"] == pytest.approx(0.537)
    assert legacy.links.at["electrolysis", "marginal_cost"] == pytest.approx(0.537 * 24 * 366 / 4)


# --- technology values from the YAML ----------------------------------------------------------

def test_operating_values_come_from_the_yaml():
    network = _prepared(4, 1.0, "snapshot_weighted")
    assert network.links.at["ammonia_synthesis", "p_min_pu"] == 0.3
    assert network.links.at["ammonia_synthesis", "ramp_limit_up"] == 0.4
    assert network.links.at["ammonia_synthesis", "ramp_limit_down"] == 0.4
    # 0.1 % a day, which PyPSA applies hour by hour
    assert (1 - network.stores.at["battery_storage", "standing_loss"]) ** 24 == pytest.approx(0.999, rel=1e-12)
    assert network.stores.at["compressed_hydrogen_store", "standing_loss"] == 0.0


def test_the_bundle_needs_the_yaml_values():
    tech = technology.load_tech_inputs(DEFAULT_TECH_YAML)
    del tech["electrolysis"]["overnight_cost_per_mw"]
    network = plant.build_network(4, PLANT_DIR, time_step=1.0, temporal_accounting_mode="snapshot_weighted")
    with pytest.raises(ValueError, match="electrolysis.overnight_cost_per_mw"):
        technology.prepare_base_network(network, tech)


@pytest.mark.parametrize("hours_per_snapshot", [1.0, 4.0])
def test_a_battery_left_alone_keeps_its_daily_share(hours_per_snapshot):
    """PyPSA's own solve: a full battery with no flows keeps 99.9 % after a day, at any step (the
    first snapshot holds the initial energy, so a day ends one snapshot after 24 hours' worth)."""
    import pypsa

    pytest.importorskip("highspy")
    network = pypsa.Network()
    network.set_snapshots(range(int(24 / hours_per_snapshot) + 1))
    network.snapshot_weightings.loc[:, :] = hours_per_snapshot
    network.add("Bus", "power_storage")
    # alone on its bus the store cannot dispatch; the cost only gives PyPSA an objective
    network.add("Store", "battery_storage", bus="power_storage", e_nom=1000.0, e_initial=1000.0, e_cyclic=False,
                marginal_cost=1.0)
    technology.apply_operating_parameters(network, {"battery_storage": {"standing_loss_fraction_per_day": 0.001}})
    network.optimize(solver_name="highs")
    assert network.stores_t.e["battery_storage"].iloc[-1] == pytest.approx(999.0, rel=1e-9)


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
    network = _prepared(4, 4.0, "legacy_scaled")
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
