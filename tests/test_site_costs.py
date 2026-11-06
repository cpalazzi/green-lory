"""model/site_costs.py: each location factor, the water price, the table and how a run reads it."""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from model import site_costs, technology
from model.config import SITE_COSTS_DIR
from model.land_table import SLOPE_CLASS_COLUMNS


def test_remote_factor_passes_through_the_anchor():
    # 1 hour, 4 hours, half a day, Shamattawa, 2 days, 4 days and Kerguelen, in minutes
    minutes = np.array([60.0, 240.0, 720.0, 1224.0, 2880.0, 5760.0, 10800.0])
    build = site_costs.remote_multiplier(minutes, site_costs.REMOTE_BUILD_AT_ANCHOR)
    equipment = site_costs.remote_multiplier(minutes, site_costs.REMOTE_EQUIPMENT_AT_ANCHOR)
    np.testing.assert_allclose(build, [1.07, 1.25, 1.66, 2.00, 2.79, 3.62, 4.48], atol=0.006)
    np.testing.assert_allclose(equipment, [1.04, 1.15, 1.40, 1.60, 2.07, 2.57, 3.09], atol=0.006)
    assert site_costs.remote_multiplier(0.0, 2.0) == 1.0


def test_travel_time_is_the_raster_then_the_port_and_the_sea_leg():
    minutes, by_port = site_costs.travel_min_to_city([100.0, np.nan], [30.0, 30.0], [500.0, 500.0])
    np.testing.assert_allclose(minutes, [100.0, 30.0 + 500.0 / 20.0 * 60.0])  # on land the whole leg counts
    assert by_port.tolist() == [False, True]
    minutes, by_port = site_costs.travel_min_to_city([100.0, np.nan], [30.0, 30.0], [100.0, 375.0], at_sea=True)
    np.testing.assert_allclose(minutes, [30.0, 30.0 + 200.0 / 20.0 * 60.0])  # at sea, only beyond 175 km
    assert by_port.tolist() == [True, True]


def test_cold_rule_takes_latitude_or_elevation():
    latitude = [59.0, 60.0, -60.0, 20.0, 20.0]
    elevation = [2999.0, 0.0, 0.0, 3000.0, -40.0]
    np.testing.assert_allclose(site_costs.cold_rule(latitude, elevation), [1.0, 1.25, 1.25, 1.25, 1.0])


def test_remote_rule_against_the_australian_factors_it_was_not_fitted_to():
    sites = pd.read_csv(SITE_COSTS_DIR / "calibration_sites.csv")
    ghd, aurecon = sites[sites.source == "ghd2025"], sites[sites.source == "aurecon2024"]
    build = site_costs.remote_multiplier(ghd.travel_min_city_20k, site_costs.REMOTE_BUILD_AT_ANCHOR)
    residual = build - ghd.installation_om_cost_factor
    assert residual.mean() == pytest.approx(-0.036, abs=0.001)
    assert np.sqrt((residual ** 2).mean()) == pytest.approx(0.087, abs=0.001)
    build = site_costs.remote_multiplier(aurecon.travel_min_city_20k, site_costs.REMOTE_BUILD_AT_ANCHOR)
    assert (build - aurecon.installation_cost_factor).mean() == pytest.approx(-0.19, abs=0.005)
    equipment = site_costs.remote_multiplier(aurecon.travel_min_city_20k, site_costs.REMOTE_EQUIPMENT_AT_ANCHOR)
    assert (equipment - aurecon.equipment_cost_factor).mean() == pytest.approx(0.007, abs=0.001)


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
    totals = {option["diameter_mm"]: option["capital_usd_per_m3_km"] + option["energy_usd_per_m3_km"]
              for option in pipeline["options"]}
    assert pipeline["diameter_mm"] == min(totals, key=totals.get)
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


def _ports():
    return pd.DataFrame({"id": ["port854", "port930"], "lat": [-21.0, -24.0], "lon": [117.0, 152.0],
                         "travel_min_city_20k": [120.0, 0.0]})


def _build(days):
    return 1.0 + np.log2(1.0 + days) / np.log2(1.85)


def _equipment(days):
    return 1.0 + 0.6 * np.log2(1.0 + days) / np.log2(1.85)


def _elevation():
    return pd.DataFrame({"latitude": [-23.0, -23.0], "longitude": [117.0, 151.0], "land_elevation_m": [400.0, 20.0]})


def _table(land=None, geography=None, elevation=None):
    return site_costs.site_cost_table(_land_table() if land is None else land, _geography() if geography is None else geography,
                                      _ports(), _elevation() if elevation is None else elevation,
                                      water_source_usd_per_m3=2.0, water_m3_per_t=1.5)


def test_site_cost_table_combines_the_factors():
    table, parameters = _table()
    assert list(table.latitude) == [-23.0, -23.0]  # the sea cell (no onshore land) is left out
    remote, coastal = table.iloc[0], table.iloc[1]
    assert remote.equipment_cost_multiplier == pytest.approx(_equipment(900.0 / 1440.0))
    assert remote.installation_multiplier == pytest.approx(_build(900.0 / 1440.0))
    assert remote.cold_multiplier == 1.0
    assert remote.slope_multiplier == pytest.approx(1.225)
    assert remote.travel_time_source == "raster" and remote.nearest_port_id == "port854"
    assert coastal.installation_multiplier == pytest.approx(_build(60.0 / 1440.0))
    assert coastal.water_cost_usd_per_m3 == pytest.approx(2.0)  # the cell holds its source
    pipeline = parameters["water_pipeline"]
    per_km = (pipeline["capital_usd_per_m3_km"] * (0.3 * remote.equipment_cost_multiplier + 0.7 * remote.installation_multiplier * 1.225)
              + pipeline["energy_usd_per_m3_km"])
    assert remote.water_cost_usd_per_m3 == pytest.approx(2.0 + 200.0 * per_km)
    assert parameters["land_competition_fraction"] == 0.2
    assert parameters["cells_by_port"] == 0


def test_a_cell_without_a_raster_time_goes_through_the_port():
    table, parameters = _table(geography=_geography().assign(travel_min_city_20k_median=[np.nan, 60.0]))
    remote = table.iloc[0]
    assert remote.travel_time_source == "port" and remote.nearest_port_id == "port854"
    assert remote.travel_time_to_city_min == pytest.approx(120.0 + remote.port_distance_km / 20.0 * 60.0)
    assert remote.installation_multiplier == pytest.approx(_build(remote.travel_time_to_city_min / 1440.0))
    assert parameters["cells_by_port"] == 1


def test_a_port_without_a_travel_time_is_left_out(tmp_path):
    path = tmp_path / "ports.csv"
    _ports().assign(travel_min_city_20k=[np.nan, 0.0]).to_csv(path, index=False)
    assert site_costs.load_ports(path)["id"].tolist() == ["port930"]


def test_cold_raises_the_pipeline_and_both_parts_of_every_technology(tmp_path):
    table, _ = _table(elevation=_elevation().assign(land_elevation_m=[3500.0, 20.0]))
    warm, _ = _table()
    assert table.cold_multiplier.tolist() == [1.25, 1.0]
    pipeline = warm.water_pipeline_cost_usd_per_m3.iloc[0]
    assert pipeline < table.water_pipeline_cost_usd_per_m3.iloc[0] < 1.25 * pipeline  # pumping energy is not scaled
    path = tmp_path / "site_costs.csv"
    table.to_csv(path, index=False)
    cold = site_costs.load_site_cost_overrides(path, ["electrolysis"])[(-23.0, 117.0)]["electrolysis"]
    assert cold["equipment_cost_multiplier"] == pytest.approx(1.25 * table.equipment_cost_multiplier.iloc[0])
    assert cold["build_cost_multiplier"] == pytest.approx(1.25 * table.installation_multiplier.iloc[0])


def test_site_cost_table_refuses_land_tables_without_slope_classes_or_geography():
    with pytest.raises(ValueError, match="slope classes"):
        _table(land=_land_table().drop(columns=list(SLOPE_CLASS_COLUMNS)))
    with pytest.raises(ValueError, match="no geography row"):
        _table(geography=_geography().iloc[:1])


def test_a_run_reads_the_table_as_overrides(tmp_path):
    table, _ = _table()
    path = tmp_path / "site_costs.csv"
    table.to_csv(path, index=False)
    overrides = site_costs.load_site_cost_overrides(path, ["solar", "wind", "electrolysis"])
    remote = overrides[(-23.0, 117.0)]
    build, equipment = table.installation_multiplier.iloc[0], table.equipment_cost_multiplier.iloc[0]
    assert remote["solar"]["build_cost_multiplier"] == pytest.approx(build * 1.225)  # slope on PV and wind
    assert remote["electrolysis"]["build_cost_multiplier"] == pytest.approx(build)  # not on the process plant
    assert remote["electrolysis"]["equipment_cost_multiplier"] == pytest.approx(equipment)
    assert remote["__location__"]["water_cost_usd_per_m3"] == pytest.approx(table.water_cost_usd_per_m3.iloc[0])

    finance = {(-23.0, 117.0): {"solar": {"interest_rate": 0.05}, "electrolysis": {"interest_rate": 0.06}}}
    merged = site_costs.merge_overrides(finance, overrides)
    assert merged[(-23.0, 117.0)]["solar"] == {"interest_rate": 0.05, "equipment_cost_multiplier": pytest.approx(equipment),
                                                "build_cost_multiplier": pytest.approx(build * 1.225)}
    assert merged[(-23.0, 151.0)]["wind"]["build_cost_multiplier"] == pytest.approx(table.installation_multiplier.iloc[1])  # no finance row: YAML rate


def test_equipment_multiplier_reaches_the_capital_cost():
    network = SimpleNamespace(generators=pd.DataFrame({"capital_cost": [0.0]}, index=["solar"]),
                              links=pd.DataFrame(), stores=pd.DataFrame())
    tech = {"solar": {"component_type": "generator", "tech_cost_per_mw": 100.0, "build_cost_per_mw": 200.0,
                      "overnight_cost_per_mw": 300.0, "lifetime_years": 20, "fixed_om_fraction": 0.02}}
    technology.apply_costs(
        network, tech, {"solar": {"interest_rate": 0.1, "equipment_cost_multiplier": 1.1, "build_cost_multiplier": 1.5}},
        time_step=1.0, temporal_accounting_mode="snapshot_weighted")
    expected = (100.0 * 1.1 + 200.0 * 1.5) * (technology.annuity_factor(0.1, 20) + 0.02)
    assert network.generators.at["solar", "capital_cost"] == pytest.approx(expected)
