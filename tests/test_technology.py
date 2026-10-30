"""model/technology.py: the YAML extends chain, annuities and the per-cell override table."""
from __future__ import annotations

from types import SimpleNamespace

import pandas as pd
import pytest

from model import technology
from model.config import TECH_DIR


def test_extends_chain_merges_the_overlay_onto_its_parent():
    yaml_path = TECH_DIR / "tech_config_ammonia_plant_2050_way_eur_explicit_compressor_dea_tank.yaml"
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


@pytest.mark.parametrize("yaml_name,expected,compression_power,synthesis_power", [
    ("tech_config_ammonia_plant_2050_dea_colocated.yaml",
     {"electrolysis": 1 / 1.2083785, "battery_pcs_charge": 0.96, "battery_pcs_discharge": 0.96, "hydrogen_fuel_cell": 0.5}, 0.0327, 0.0872338),
    ("tech_config_ammonia_plant_2050_way_eur.yaml",
     {"electrolysis": 0.886509, "battery_pcs_charge": 0.98, "battery_pcs_discharge": 0.99, "hydrogen_fuel_cell": 0.540015}, 0.05, 0.16),
])
def test_link_efficiencies_come_from_the_yaml_recipe(yaml_name, expected, compression_power, synthesis_power):
    from model import plant
    from model.config import DEFAULT_PLANT_DIR

    network = plant.build_network(2, DEFAULT_PLANT_DIR, time_step=1.0, temporal_accounting_mode="snapshot_weighted")
    technology.apply_link_efficiencies(network, technology.load_tech_inputs(TECH_DIR / yaml_name))
    for name, value in expected.items():
        assert network.links.at[name, "efficiency"] == value  # exact: the recipes carry these numbers
    # Three-port links: compression draws power, synthesis draws hydrogen
    assert (network.links.at["hydrogen_compression", "efficiency"], network.links.at["hydrogen_compression", "efficiency2"]) == (1.0, -compression_power)
    # Synthesis is rated on its electricity input (bus0), with hydrogen on bus2
    assert (network.links.at["ammonia_synthesis", "efficiency"], network.links.at["ammonia_synthesis", "efficiency2"]) == (
        1 / synthesis_power, -1.13472 / synthesis_power)


def test_annuity_factor():
    assert technology.annuity_factor(0.0, 20) == pytest.approx(1 / 20)
    assert technology.annuity_factor(0.05, 20) == pytest.approx(0.0802425872)
    assert technology.annualised_capital_cost(1000.0, 0.05, 20, 0.02) == pytest.approx(1000 * (0.0802425872 + 0.02))


def _solar_network():
    return SimpleNamespace(generators=pd.DataFrame({"capital_cost": [0.0]}, index=["solar"]),
                           links=pd.DataFrame(), stores=pd.DataFrame())


SOLAR = {"solar": {"component_type": "generator", "tech_cost_per_mw": 100.0, "build_cost_per_mw": 200.0,
                   "overnight_cost_per_mw": 300.0, "lifetime_years": 20, "fixed_om_fraction": 0.0}}


def test_costs_take_the_cells_interest_rate_and_multipliers():
    network = _solar_network()
    technology.apply_costs(network, SOLAR, {"solar": {"interest_rate": 0.05, "build_cost_multiplier": 1.5}},
                           time_step=1.0, temporal_accounting_mode="snapshot_weighted")
    expected = (100.0 + 200.0 * 1.5) * technology.annuity_factor(0.05, 20)
    assert network.generators.at["solar", "capital_cost"] == pytest.approx(expected)


def test_costs_need_an_interest_rate_from_the_run():
    with pytest.raises(ValueError, match="No interest rate for solar"):
        technology.apply_costs(_solar_network(), SOLAR, {"solar": {"build_cost_multiplier": 1.5}},
                               time_step=1.0, temporal_accounting_mode="snapshot_weighted")


def test_yamls_give_no_interest_rates_and_consistent_totals(tmp_path):
    base = "techs:\n  solar: {component_type: generator, tech_cost_per_mw: 100, build_cost_per_mw: 200, " \
           "overnight_cost_per_mw: %s, lifetime_years: 20, fixed_om_fraction: 0%s}\n"
    (tmp_path / "rate.yaml").write_text(base % (300, ", interest_rate: 0.1"))
    with pytest.raises(ValueError, match="interest rates come from the run"):
        technology.load_tech_inputs(tmp_path / "rate.yaml")
    (tmp_path / "total.yaml").write_text(base % (310, ""))
    with pytest.raises(ValueError, match="not overnight_cost_per_mw = 310"):
        technology.load_tech_inputs(tmp_path / "total.yaml")


# --- finance tables ---------------------------------------------------------------------------

TECHS = {name: {"component_type": kind} for name, kind in
         (("solar", "generator"), ("wind", "generator"), ("battery_storage", "store"))}


def _finance(tmp_path, rows):
    path = tmp_path / "finance.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def test_a_finance_table_by_cell_with_defaults(tmp_path):
    path = _finance(tmp_path, [
        {"lat": 1, "lon": 3, "tech": "all", "interest_rate": 0.06},
        {"lat": 1, "lon": 3, "tech": "Solar", "interest_rate": 0.04, "source": "a note the model ignores"},
        {"lat": 2, "lon": 4, "tech": "solar", "interest_rate": 0.05},
    ])
    index, gaps = technology.finance_overrides(path, [(1.0, 3.0), (2.0, 4.0)], TECHS)
    assert technology.interest_rates(technology.cell_overrides(index, 1.0, 3.0)) == {
        "battery_storage": 0.06, "solar": 0.04, "wind": 0.06}
    assert gaps == {(2.0, 4.0): ["battery_storage", "wind"]}
    assert "1 cell without an interest rate" in technology.describe_finance_gaps(gaps)


def test_a_finance_table_by_country_reaches_its_cells(tmp_path):
    path = _finance(tmp_path, [{"iso3": "chl", "tech": "all", "interest_rate": 0.09},
                               {"iso3": "CHL", "tech": "wind", "interest_rate": 0.08}])
    index, gaps = technology.finance_overrides(path, [(-23.0, -69.0), (-23.0, 117.0)], TECHS)
    assert technology.interest_rates(technology.cell_overrides(index, -23.0, -69.0)) == {
        "battery_storage": 0.09, "solar": 0.09, "wind": 0.08}
    assert list(gaps) == [(-23.0, 117.0)]  # Australia has no rows


@pytest.mark.parametrize("rows,message", [
    ([{"lat": 1, "lon": 3, "iso3": "CHL", "tech": "all", "interest_rate": 0.06}], "either by cell"),
    ([{"iso3": "CHL", "tech": "onshore_wind", "interest_rate": 0.06}], "technologies the plant does not have"),
    ([{"iso3": "CHL", "tech": "all", "interest_rate": 6}], r"fraction in \[0, 1\)"),
    ([{"iso3": "CHL", "tech": "all", "interest_rate": 0.06}] * 2, "rows given twice"),
    ([{"iso3": "CHL", "tech": "all", "interest_rate": 0.06, "build_cost_multiplier": 1.2}], "belong to the site-cost table"),
])
def test_malformed_finance_tables_are_refused(tmp_path, rows, message):
    with pytest.raises(ValueError, match=message):
        technology.finance_overrides(_finance(tmp_path, rows), [(-23.0, -69.0)], TECHS)


def test_one_rate_equals_a_table_of_that_rate(tmp_path):
    cells = [(-23.0, 117.0), (-22.0, 117.0)]
    path = _finance(tmp_path, [{"lat": lat, "lon": lon, "tech": "all", "interest_rate": 0.05} for lat, lon in cells])
    index, gaps = technology.finance_overrides(path, cells, TECHS)
    assert not gaps and technology.uniform_interest_overrides(cells, TECHS, 0.05) == index
