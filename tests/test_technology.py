"""model/technology.py: the YAML extends chain, annuities and the per-cell override table."""
from __future__ import annotations

from types import SimpleNamespace

import pandas as pd
import pytest

from model import technology
from model.config import INPUTS_DIR


def test_extends_chain_merges_the_overlay_onto_its_parent():
    yaml_path = INPUTS_DIR / "tech_config_ammonia_plant_2050_way_eur_explicit_compressor_dea_tank.yaml"
    assert [path.name for path in technology.tech_yaml_files(yaml_path)] == [
        "tech_config_ammonia_plant_2050_way_eur_explicit_compressor_dea_tank.yaml",
        "tech_config_ammonia_plant_2050_way_eur.yaml",
    ]
    assert technology.load_tech_config(yaml_path)["techs"]


def test_circular_extends_is_rejected(tmp_path):
    (tmp_path / "a.yaml").write_text("extends: b.yaml\n")
    (tmp_path / "b.yaml").write_text("extends: a.yaml\n")
    with pytest.raises(ValueError, match="Circular"):
        technology.load_tech_config(tmp_path / "a.yaml")
    with pytest.raises(ValueError, match="Circular"):
        technology.tech_yaml_files(tmp_path / "a.yaml")


@pytest.mark.parametrize("yaml_name,expected", [
    ("tech_config_ammonia_plant_2050_dea_colocated.yaml",
     {"electrolysis": 0.84, "battery_pcs_charge": 0.96, "battery_pcs_discharge": 0.96, "hydrogen_fuel_cell": 0.5}),
    ("tech_config_ammonia_plant_2050_way_eur.yaml",
     {"electrolysis": 0.886509, "battery_pcs_charge": 0.98, "battery_pcs_discharge": 0.99, "hydrogen_fuel_cell": 0.540015}),
])
def test_link_efficiencies_come_from_the_yaml_recipe(yaml_name, expected):
    from model import plant
    from model.config import DEFAULT_PLANT_DIR

    network = plant.build_network(2, DEFAULT_PLANT_DIR, time_step=1.0, temporal_accounting_mode="snapshot_weighted")
    technology.apply_link_efficiencies(network, technology.load_tech_inputs(INPUTS_DIR / yaml_name))
    for name, value in expected.items():
        assert network.links.at[name, "efficiency"] == value  # exact: the recipes carry these numbers
    # Three-port links: compression draws power, synthesis draws hydrogen
    assert (network.links.at["hydrogen_compression", "efficiency"], network.links.at["hydrogen_compression", "efficiency2"]) == (1.0, -0.05)
    assert (network.links.at["ammonia_synthesis", "efficiency"], network.links.at["ammonia_synthesis", "efficiency2"]) == (6.25, -7.092)


def test_annuity_factor():
    assert technology.annuity_factor(0.0, 20) == pytest.approx(1 / 20)
    assert technology.annuity_factor(0.05, 20) == pytest.approx(0.0802425872)
    assert technology.annualised_capital_cost(1000.0, 0.05, 20, 0.02) == pytest.approx(1000 * (0.0802425872 + 0.02))


def _override_table():
    return pd.DataFrame({
        "lat": [1.0, 1.0, 2.0],
        "lon": [3.0, 3.0, 4.0],
        "tech": ["Solar", "electrolysis", "solar"],
        "interest_rate": [0.06, 0.07, 0.05],
        "build_cost_multiplier": [1.5, None, 1.0],
        "water_cost_usd_per_m3": [3.0, 3.0, None],
    })


def test_override_table_is_indexed_by_cell(tmp_path):
    path = tmp_path / "overrides.csv"
    _override_table().to_csv(path, index=False)
    index = technology.index_override_table(technology.load_override_table(path))
    cell = technology.cell_overrides(index, 1.0, 3.0)
    assert cell["solar"] == {"interest_rate": 0.06, "build_cost_multiplier": 1.5}
    assert cell["electrolysis"] == {"interest_rate": 0.07}
    assert cell["__location__"] == {"water_cost_usd_per_m3": 3.0}
    assert "__location__" not in technology.cell_overrides(index, 2.0, 4.0)
    assert technology.cell_overrides(index, 9.0, 9.0) is None
    assert technology.interest_rates(cell) == {"solar": 0.06, "electrolysis": 0.07}


def test_cell_override_multiplies_only_the_installation_part():
    network = SimpleNamespace(generators=pd.DataFrame({"capital_cost": [0.0]}, index=["solar"]),
                              links=pd.DataFrame(), stores=pd.DataFrame())
    tech = {"solar": {"component_type": "generator", "tech_cost_per_mw": 100.0, "build_cost_per_mw": 200.0,
                      "overnight_cost_per_mw": 300.0, "lifetime_years": 20, "interest_rate": 0.1,
                      "fixed_om_fraction": 0.0}}
    technology.apply_base_costs(network, tech, time_step=1.0, temporal_accounting_mode="snapshot_weighted")
    assert network.generators.at["solar", "capital_cost"] == pytest.approx(300.0 * technology.annuity_factor(0.1, 20))
    technology.apply_cell_overrides(network, tech, {"solar": {"interest_rate": 0.05, "build_cost_multiplier": 1.5}},
                                    time_step=1.0, temporal_accounting_mode="snapshot_weighted")
    expected = (100.0 + 200.0 * 1.5) * technology.annuity_factor(0.05, 20)
    assert network.generators.at["solar", "capital_cost"] == pytest.approx(expected)
