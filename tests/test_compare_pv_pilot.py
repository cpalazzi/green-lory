import pandas as pd
import pytest

from reconciliation.land.compare_pv_pilot import shared_capacity, validate_comparison_inputs


def test_independent_shared_land_capacity():
    plant = pd.Series(dict(wind_mw=10, solar_mw=50, solar_tracking_mw=100,
                           gridless_ammonia_production_t=1e6))
    land = pd.Series(dict(wind_density_mw_per_km2=5, solar_density_mw_per_km2=100,
                          wind_onshore_area_km2=20, solar_area_km2=20,
                          renewable_union_area_km2=30))
    result = shared_capacity(plant, land, 2)
    assert result["capacity_Mtpa"] == pytest.approx(30 / 4.5)
    assert result["limiting_constraint"] == "renewable_union"
    assert result["wind_footprint_km2"] == 2
    assert result["pv_footprint_km2"] == 2.5
    assert shared_capacity(plant, land, 1)["capacity_Mtpa"] == pytest.approx(30 / 3.5)


def test_no_gridless_production_cannot_create_supply():
    plant = pd.Series(dict(wind_mw=10, solar_mw=0, solar_tracking_mw=0,
                           gridless_ammonia_production_t=0))
    land = pd.Series(dict(wind_density_mw_per_km2=5, solar_density_mw_per_km2=100,
                          wind_onshore_area_km2=20, solar_area_km2=20,
                          renewable_union_area_km2=30))
    assert shared_capacity(plant, land)["capacity_Mtpa"] == 0


def test_comparison_rejects_changed_cost_or_weather_inputs():
    old = {"inputs": {
        "override_csv": {"path": "finance.csv", "sha256": "finance-hash"},
        "tech_yaml": {"extends_chain": [{"path": "parent.yaml", "sha256": "tech-hash"}]},
        "plant_bundle": {"files": {}},
        "weather": {"files": [{"name": "Solar.nc", "size_bytes": 100, "mtime_ns": 200}]},
    }}
    new = {
        "inputs": [{"path": "finance.csv", "sha256": "finance-hash"},
                   {"path": "parent.yaml", "sha256": "tech-hash"}],
        "weather_source_files": [{"path": "/data/Solar.nc", "size_bytes": 100, "mtime_ns": 200}],
    }
    validate_comparison_inputs(new, old)
    new["inputs"][1]["sha256"] = "changed"
    with pytest.raises(ValueError, match="Changed controlled input"):
        validate_comparison_inputs(new, old)
    new["inputs"][1]["sha256"] = "tech-hash"
    new["weather_source_files"][0]["mtime_ns"] = 300
    with pytest.raises(ValueError, match="weather source metadata changed"):
        validate_comparison_inputs(new, old)
