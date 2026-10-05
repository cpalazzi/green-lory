import pandas as pd
import pytest

from reconciliation.land.audit_supply_pilot import energy_upper_bound


def test_energy_relaxation_respects_shared_and_technology_budgets():
    land = pd.Series(dict(solar_density_mw_per_km2=10, wind_density_mw_per_km2=5,
                          solar_area_km2=2, wind_onshore_area_km2=5,
                          renewable_union_area_km2=3))
    bound = energy_upper_bound(land, dict(solar=.5, solar_tracking=.6, wind=.4), 10)
    # Fixed PV is best (5 mean MW/km2), then wind (2); tracking gives only 3.
    assert bound["ideal_annual_generation_MWh"] == pytest.approx((2 * 5 + 1 * 2) * 8760)
    assert bound["ideal_energy_bound_Mtpa"] == pytest.approx(12 * 8760 / 10 / 1e6)


def test_energy_relaxation_handles_wind_led_case():
    land = pd.Series(dict(solar_density_mw_per_km2=10, wind_density_mw_per_km2=5,
                          solar_area_km2=3, wind_onshore_area_km2=1,
                          renewable_union_area_km2=2))
    bound = energy_upper_bound(land, dict(solar=.1, solar_tracking=.3, wind=1), 10)
    assert bound["ideal_annual_generation_MWh"] == pytest.approx((5 + 1.5) * 8760)
