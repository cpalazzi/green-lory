"""model/site_costs.py: each location factor, the water price, the table and how a run reads it."""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from model import site_costs, technology
from model.land_table import SLOPE_CLASS_COLUMNS


def test_equipment_multiplier_rises_with_port_distance_up_to_the_cap():
    multiplier = site_costs.equipment_multiplier([0.0, 250.0, 1000.0, 5000.0], a_per_1000_km=0.2, cap=1.1)
    np.testing.assert_allclose(multiplier, [1.0, 1.05, 1.1, 1.1])


def test_installation_multiplier_saturates_and_fills_missing_times():
    multiplier = site_costs.installation_multiplier([0.0, 120.0, 240.0, 2000.0, np.nan], b_per_hour=0.09, saturation_min=240.0)
    np.testing.assert_allclose(multiplier, [1.0, 1.18, 1.36, 1.36, 1.36])


def _classes(*rows):
    return pd.DataFrame(rows, columns=list(SLOPE_CLASS_COLUMNS[:3]))


def test_slope_multiplier_takes_the_flattest_land_first():
    classes = _classes(
        (100.0, 0.0, 0.0),   # flat cell
        (30.0, 40.0, 30.0),  # the 20 % share fits in the flattest class
        (10.0, 5.0, 85.0),   # 10 % at 1.0, 5 % at 1.3 and 5 % at 1.6
        (0.0, 0.0, 100.0),   # all land in the steepest class kept
        (0.0, 0.0, 0.0),     # no land up to 20 degrees
    )
    np.testing.assert_allclose(site_costs.slope_multiplier(classes, 0.2),
                               [1.0, 1.0, (10 + 5 * 1.3 + 5 * 1.6) / 20, 1.6, 1.0])
    # the whole cell's land: the area-weighted mean of the classes
    assert site_costs.slope_multiplier(_classes((30.0, 40.0, 30.0)), 1.0)[0] == pytest.approx(0.3 + 0.4 * 1.3 + 0.3 * 1.6)


def test_slope_shares_ignore_land_above_twenty_degrees():
    # the classes are percentages of all land; land above 20 degrees is excluded before the share
    with_steep = site_costs.slope_multiplier(_classes((5.0, 5.0, 40.0)), 0.2)
    without = site_costs.slope_multiplier(_classes((10.0, 10.0, 80.0)), 0.2)
    np.testing.assert_allclose(with_steep, without)


def test_water_pipeline_takes_the_cheapest_diameter():
    pipeline = site_costs.water_pipeline_cost(1.5e6)
    totals = {option["diameter_in"]: option["capital_usd_per_m3_km"] + option["energy_usd_per_m3_km"]
              for option in pipeline["options"]}
    assert pipeline["diameter_in"] == min(totals, key=totals.get)
    assert pipeline["capital_usd_per_m3_km"] + pipeline["energy_usd_per_m3_km"] == pytest.approx(min(totals.values()))
    # a smaller flow pays more per cubic metre (the scale effect behind Zhou and Tol's lower costs)
    small = site_costs.water_pipeline_cost(0.15e6)
    assert small["capital_usd_per_m3_km"] > pipeline["capital_usd_per_m3_km"]


def _land_table():
    return pd.DataFrame({
        "latitude": [-23.0, -23.0, -24.0],
        "longitude": [117.0, 151.0, 152.0],
        "onshore_land_pct": [100.0, 30.0, 0.0],
        "land_competition_fraction": [0.2, 0.2, 0.2],
        SLOPE_CLASS_COLUMNS[0]: [10.0, 100.0, 0.0],
        SLOPE_CLASS_COLUMNS[1]: [5.0, 0.0, 0.0],
        SLOPE_CLASS_COLUMNS[2]: [85.0, 0.0, 0.0],
        SLOPE_CLASS_COLUMNS[3]: [0.0, 0.0, 0.0],
    })


def _geography():
    return pd.DataFrame({
        "latitude": [-23.0, -23.0],
        "longitude": [117.0, 151.0],
        "port_distance_km": [250.0, 60.0],
        "nearest_port_id": ["port854", "port930"],
        "travel_min_city_20k_median": [900.0, 60.0],
        "water_source_kind": ["sea", "sea"],
        "water_source_distance_km": [200.0, 0.0],
    })


REMOTENESS = {"source": "test", "equipment_a_per_1000_km": 0.2, "equipment_cap": 1.1, "installation_city_size": "20k",
              "installation_b_per_hour": 0.09, "installation_saturation_min": 240.0}


def test_site_cost_table_combines_the_factors():
    table, parameters = site_costs.site_cost_table(_land_table(), _geography(), REMOTENESS,
                                                   water_source_usd_per_m3=2.0, water_m3_per_t=1.5)
    assert list(table.latitude) == [-23.0, -23.0]  # the sea cell (no onshore land) is left out
    remote, coastal = table.iloc[0], table.iloc[1]
    assert remote.equipment_cost_multiplier == pytest.approx(1.05)
    assert remote.installation_multiplier == pytest.approx(1.36)
    assert remote.slope_multiplier == pytest.approx(1.225)
    assert coastal.installation_multiplier == pytest.approx(1.09)
    assert coastal.water_cost_usd_per_m3 == pytest.approx(2.0)  # the cell holds its source
    pipeline = parameters["water_pipeline"]
    per_km = (pipeline["capital_usd_per_m3_km"] * (0.3 * 1.05 + 0.7 * 1.36 * 1.225) + pipeline["energy_usd_per_m3_km"])
    assert remote.water_cost_usd_per_m3 == pytest.approx(2.0 + 200.0 * per_km)
    assert parameters["land_competition_fraction"] == 0.2


def test_site_cost_table_refuses_land_tables_without_slope_classes_or_geography():
    with pytest.raises(ValueError, match="slope classes"):
        site_costs.site_cost_table(_land_table().drop(columns=list(SLOPE_CLASS_COLUMNS)), _geography(), REMOTENESS,
                                   water_source_usd_per_m3=2.0, water_m3_per_t=1.5)
    with pytest.raises(ValueError, match="no geography row"):
        site_costs.site_cost_table(_land_table(), _geography().iloc[:1], REMOTENESS,
                                   water_source_usd_per_m3=2.0, water_m3_per_t=1.5)


def test_a_run_reads_the_table_as_overrides(tmp_path):
    table, _ = site_costs.site_cost_table(_land_table(), _geography(), REMOTENESS,
                                          water_source_usd_per_m3=2.0, water_m3_per_t=1.5)
    path = tmp_path / "site_costs.csv"
    table.to_csv(path, index=False)
    overrides = site_costs.load_site_cost_overrides(path, ["solar", "wind", "electrolysis"])
    remote = overrides[(-23.0, 117.0)]
    assert remote["solar"]["build_cost_multiplier"] == pytest.approx(1.36 * 1.225)  # slope on PV and wind
    assert remote["electrolysis"]["build_cost_multiplier"] == pytest.approx(1.36)  # not on the process plant
    assert remote["electrolysis"]["equipment_cost_multiplier"] == pytest.approx(1.05)
    assert remote["__location__"]["water_cost_usd_per_m3"] == pytest.approx(table.water_cost_usd_per_m3.iloc[0])

    finance = {(-23.0, 117.0): {"solar": {"interest_rate": 0.05}, "electrolysis": {"interest_rate": 0.06}}}
    merged = site_costs.merge_overrides(finance, overrides)
    assert merged[(-23.0, 117.0)]["solar"] == {"interest_rate": 0.05, "equipment_cost_multiplier": pytest.approx(1.05),
                                                "build_cost_multiplier": pytest.approx(1.36 * 1.225)}
    assert merged[(-23.0, 151.0)]["wind"]["build_cost_multiplier"] == pytest.approx(1.09)  # no finance row: YAML rate


def test_equipment_multiplier_reaches_the_capital_cost():
    network = SimpleNamespace(generators=pd.DataFrame({"capital_cost": [0.0]}, index=["solar"]),
                              links=pd.DataFrame(), stores=pd.DataFrame())
    tech = {"solar": {"component_type": "generator", "tech_cost_per_mw": 100.0, "build_cost_per_mw": 200.0,
                      "overnight_cost_per_mw": 300.0, "lifetime_years": 20, "interest_rate": 0.1,
                      "fixed_om_fraction": 0.02}}
    technology.apply_cell_overrides(
        network, tech, {"solar": {"equipment_cost_multiplier": 1.1, "build_cost_multiplier": 1.5}},
        time_step=1.0, temporal_accounting_mode="snapshot_weighted")
    expected = (100.0 * 1.1 + 200.0 * 1.5) * (technology.annuity_factor(0.1, 20) + 0.02)  # the YAML's rate
    assert network.generators.at["solar", "capital_cost"] == pytest.approx(expected)
