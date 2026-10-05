from types import SimpleNamespace

import pytest

from reconciliation.land.compare_model_evidence import scaled_capacity, HHV, LHV
from reconciliation.land.audit_supply_pilot import check_pv_policy


def test_independent_fixed_policy_audit():
    check_pv_policy(SimpleNamespace(solar_tracking_mw=0.), "fixed-only")
    check_pv_policy(SimpleNamespace(solar_tracking_mw=1.), "both")
    with pytest.raises(ValueError, match="Tracking capacity"):
        check_pv_policy(SimpleNamespace(solar_tracking_mw=1.), "fixed-only")


def test_shared_land_and_independent_land_are_distinct():
    land = SimpleNamespace(solar_density_mw_per_km2=10., solar_area_km2=100.,
                           wind_onshore_area_km2=100., renewable_union_area_km2=100.)
    shared, independent = scaled_capacity(1., 250., 500., 0., land)
    assert shared == pytest.approx(1.)
    assert independent == pytest.approx(2.)
    assert scaled_capacity(2., 500., 1000., 0., land) == pytest.approx((shared, independent))


def test_tracking_footprint_is_separate_from_generation():
    land = SimpleNamespace(solar_density_mw_per_km2=10., solar_area_km2=100.,
                           wind_onshore_area_km2=100., renewable_union_area_km2=100.)
    assert scaled_capacity(1., 0., 0., 500., land, ratio=1.) == pytest.approx((2., 2.))
    assert scaled_capacity(1., 0., 0., 500., land, ratio=2.) == pytest.approx((1., 1.))


def test_heating_value_reporting_conversion():
    assert HHV * 3.6 == pytest.approx(22.5)
    assert LHV * 3.6 == pytest.approx(18.6)
    assert LHV / HHV == pytest.approx(18.6 / 22.5)
