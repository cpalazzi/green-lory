"""model/sea_cells.py: which cells hold a sea plant, and what a sea run's tables give it."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from model import land, sea_cells, site_costs

WIND_TECH, WIND_BUILD = 775395.0, 96861.0


def test_depth_classes_follow_the_limits():
    assert sea_cells.depth_class([30.0, 70.0, 71.0, 1000.0, 1001.0]).tolist() == ["fixed", "fixed", "floating", "floating", "deep"]


def test_fixed_foundation_follows_deas_function_of_depth():
    assert sea_cells.fixed_foundation_eur_per_mw(30.0) == pytest.approx(334000.0)
    assert sea_cells.fixed_foundation_eur_per_mw(70.0) == pytest.approx(334000.0 * 0.80 / 0.60)


def test_wind_at_sea_costs_deas_figures_at_multipliers_of_one():
    equipment, build = sea_cells.wind_cost_ratios([30.0, 500.0, 3000.0], WIND_TECH, WIND_BUILD)
    overnight = WIND_TECH * equipment + WIND_BUILD * build
    assert overnight[0] == pytest.approx(1.284e6)  # DEA without the substation and export cable
    assert overnight[1] == pytest.approx(1.625e6)
    assert overnight[2] == pytest.approx(3.0 * (WIND_TECH + WIND_BUILD))  # deep: three times onshore


def _land_table():
    cells = {"latitude": [55.0, 40.0, 75.0, -58.0, 43.0, 10.0], "longitude": [2.0, -30.0, -86.0, -60.0, -87.0, 20.0]}
    return pd.DataFrame({
        **cells,
        "cell_anchor": "center",
        "area": 8000.0,
        "offshore_area_km2": [8000.0, 8000.0, 8000.0, 8000.0, 8000.0, 0.0],
        "land_competition_fraction": 0.2,
        "wind_density_mw_per_km2": 5.0,
        "solar_density_mw_per_km2": 78.0,
        "renewable_union_method": land.CLASSWISE_NESTED_UNION_METHOD,
        "renewable_union_method_version": land.CLASSWISE_NESTED_UNION_VERSION,
        **{column: 0.0 for column in ("availability", "wind_availability", "wind_onshore_availability", "wind_offshore_availability",
                                      "solar_availability", "renewable_union_availability",
                                      "renewable_union_availability_classwise_nested_v1", "wind_area_km2", "wind_onshore_area_km2",
                                      "wind_offshore_area_km2", "solar_area_km2", "renewable_union_area_km2",
                                      "renewable_union_area_km2_classwise_nested_v1", "max_power_solar_mw", "max_power_wind_mw",
                                      "max_capacity_mw")},
    })


def _cells():
    table = _land_table()
    keys = table[["latitude", "longitude"]]
    # the Hudson Bay cell freezes, which changes nothing; the lake cell has no sea depth; the
    # Sahara cell has no sea
    elevation = keys.assign(sea_depth_m=[30.0, 4000.0, 180.0, 3600.0, np.nan, np.nan])
    zones = keys.assign(eez_iso3=["GBR", np.nan, "CAN", np.nan, "USA", np.nan], eez_type="200NM", eez_source="centre")
    # the Drake Passage cell lies beyond the geography, so takes the land table's water
    geography = keys[keys.latitude > -55].assign(cell_area_km2=8000.0, sea_km2=[6000.0, 8000.0, 8000.0, 0.0, 0.0])
    return sea_cells.sea_cells(table, geography, elevation, zones)


def test_sea_cells_need_sea_and_a_depth():
    cells = _cells()
    assert cells.latitude.tolist() == [55.0, 40.0, 75.0, -58.0]
    assert cells.sea_share.tolist() == pytest.approx([0.75, 1.0, 1.0, 1.0])
    assert cells.depth_class.tolist() == ["fixed", "deep", "floating", "deep"]


def test_the_land_table_gives_the_plant_the_cells_sea_at_the_share(tmp_path):
    table = sea_cells.sea_land_table(_land_table(), _cells())
    path = tmp_path / "sea_land.csv"
    table.to_csv(path, index=False)
    budget = land.build_renewable_land_budget(table.iloc[0], {}, onshore=True)
    assert budget.wind_area_km2 == pytest.approx(8000.0 * 0.75 * 0.2)
    assert budget.solar_area_km2 == 0.0
    assert table.max_power_wind_mw.iloc[0] == pytest.approx(1200.0 * 5.0)


def _site_costs(tmp_path):
    ports = pd.DataFrame({"id": ["port1", "port2"], "lat": [52.0, -52.0], "lon": [2.0, -58.0], "travel_min_city_20k": [0.0, 300.0]})
    table = sea_cells.sea_site_cost_table(_cells(), ports, wind_tech_cost=WIND_TECH, wind_build_cost=WIND_BUILD, water_source_usd_per_m3=2.0)
    path = tmp_path / "sea_costs.csv"
    table.to_csv(path, index=False)
    return table, path


def test_the_sea_leg_counts_beyond_deas_distance(tmp_path):
    table, _ = _site_costs(tmp_path)
    near, far = table.iloc[0], table.iloc[1]
    assert near.port_distance_km == pytest.approx(333.6, abs=0.5)
    assert near.travel_time_to_city_min == pytest.approx((near.port_distance_km - 175.0) / 20.0 * 60.0)
    assert near.installation_multiplier == pytest.approx(site_costs.remote_multiplier(near.travel_time_to_city_min, 2.0))
    assert far.travel_time_source == "port" and far.water_cost_usd_per_m3 == 2.0


def test_frozen_sea_pays_the_cold_rule(tmp_path):
    table, _ = _site_costs(tmp_path)
    assert table.cold_multiplier.tolist() == [1.0, 1.0, 1.25, 1.0]


def test_a_run_reads_the_depth_class_costs(tmp_path):
    table, path = _site_costs(tmp_path)
    overrides = site_costs.load_site_cost_overrides(path, ["wind", "electrolysis"])
    fixed = overrides[(55.0, 2.0)]
    row = table.iloc[0]
    assert fixed["electrolysis"]["equipment_cost_multiplier"] == pytest.approx(1.72 * row.equipment_cost_multiplier)
    assert fixed["electrolysis"]["build_cost_multiplier"] == pytest.approx(1.72 * row.installation_multiplier)
    overnight = (WIND_TECH * fixed["wind"]["equipment_cost_multiplier"] / row.equipment_cost_multiplier
                 + WIND_BUILD * fixed["wind"]["build_cost_multiplier"] / row.installation_multiplier)
    assert overnight == pytest.approx(0.75e6 + sea_cells.fixed_foundation_eur_per_mw(30.0) + 0.2e6)
    deep = overrides[(40.0, -30.0)]
    assert deep["wind"]["equipment_cost_multiplier"] == pytest.approx(3.0 * table.iloc[1].equipment_cost_multiplier)
    assert deep["electrolysis"]["equipment_cost_multiplier"] == pytest.approx(3.0 * table.iloc[1].equipment_cost_multiplier)


def test_a_flat_run_at_sea_keeps_only_the_depth_class_costs(tmp_path):
    table, _ = _site_costs(tmp_path)
    path = tmp_path / "flat.csv"
    sea_cells.flat_site_cost_table(table).to_csv(path, index=False)
    overrides = site_costs.load_site_cost_overrides(path, ["wind", "electrolysis"])
    fixed = overrides[(55.0, 2.0)]
    assert (WIND_TECH * fixed["wind"]["equipment_cost_multiplier"] + WIND_BUILD * fixed["wind"]["build_cost_multiplier"]
            == pytest.approx(0.75e6 + sea_cells.fixed_foundation_eur_per_mw(30.0) + 0.2e6))
    assert fixed["electrolysis"]["equipment_cost_multiplier"] == pytest.approx(1.72)
    frozen = overrides[(75.0, -86.0)]  # floating, and no cold factor in a flat run
    assert frozen["electrolysis"]["build_cost_multiplier"] == pytest.approx(2.0)


def test_finance_takes_the_zone_its_region_or_the_international_rate():
    by_country = pd.DataFrame({"iso3": ["GBR", "ZAF", "BRA", "SAU", "IDN", "FRA"], "ameli_region": ["UK", "AFR", "CSA", "MEA", "ODA", "WEU"],
                               "interest_rate": [0.044, 0.118, 0.092, 0.068, 0.068, 0.042]})
    cells = pd.DataFrame({"latitude": [55.0, -20.0, 40.0], "longitude": [2.0, 57.0, -30.0], "eez_iso3": ["GBR", "MUS", np.nan]})
    table = sea_cells.sea_finance_table(cells, by_country)
    assert table.interest_rate.tolist() == [0.044, 0.118, 0.118]
    assert table.tech.unique().tolist() == ["all"]
    # international waters take the table's highest rate, whatever it is
    reduced = by_country.assign(interest_rate=by_country.interest_rate.clip(upper=0.051))
    assert sea_cells.sea_finance_table(cells, reduced).interest_rate.tolist() == [0.044, 0.051, 0.051]
    with pytest.raises(ValueError, match="XXX"):
        sea_cells.sea_finance_table(cells.assign(eez_iso3="XXX"), by_country)


def test_protected_water_is_left_out_of_the_sea():
    table, keys = _land_table(), _land_table()[["latitude", "longitude"]]
    elevation = keys.assign(sea_depth_m=[30.0, 4000.0, 180.0, 3600.0, np.nan, np.nan])
    zones = keys.assign(eez_iso3=["GBR", np.nan, "CAN", np.nan, "USA", np.nan], eez_type="200NM", eez_source="centre")
    geography = keys[keys.latitude > -55].assign(cell_area_km2=8000.0, sea_km2=[6000.0, 8000.0, 8000.0, 0.0, 0.0])
    protected = keys.iloc[[0, 3]].assign(protected_water_pct=[40.0, 100.0])  # the other cells are not in the table
    cells = sea_cells.sea_cells(table, geography, elevation, zones, protected)
    assert cells.protected_water_pct.tolist() == [40.0, 0.0, 0.0, 100.0]
    land_table = sea_cells.sea_land_table(table, cells)
    assert land_table.wind_onshore_area_km2.tolist() == pytest.approx([8000.0 * 0.75 * 0.6 * 0.2, 1600.0, 1600.0, 0.0])
    assert land_table.max_power_wind_mw.iloc[0] == pytest.approx(720.0 * 5.0)
    assert sea_cells.sea_land_table(table, _cells()).wind_onshore_area_km2.iloc[0] == pytest.approx(1200.0)  # none protected
