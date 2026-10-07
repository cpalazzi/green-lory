"""model/land.py: land budgets, the capacity rules, land tables and land in the solve."""
import math
import unittest

from model.land import (
    CLASSWISE_NESTED_UNION_AREA_COLUMN,
    CLASSWISE_NESTED_UNION_METHOD,
    CLASSWISE_NESTED_UNION_SOURCE,
    CLASSWISE_NESTED_UNION_VERSION,
    RenewableLandBudget,
    estimate_scaled_design_capacity,
    renewable_land_use_from_results,
)


TECH_INPUTS = {
    "wind": {"land_use_km2_per_mw": 0.2},
    "solar": {"land_use_km2_per_mw": 0.01},
    "solar_tracking": {"land_use_km2_per_mw": 0.02},
}


def make_land_row(fraction: float = 1.0) -> dict[str, float]:
    return {
        "wind_onshore_area_km2": 1_000.0 * fraction,
        "solar_area_km2": 100.0 * fraction,
        "renewable_union_area_km2": 1_050.0 * fraction,
        CLASSWISE_NESTED_UNION_AREA_COLUMN: 1_050.0 * fraction,
        "wind_density_mw_per_km2": 5.0,
        "solar_density_mw_per_km2": 100.0,
    }


class RenewableLandBudgetTest(unittest.TestCase):
    def test_versioned_union_lower_bound_is_preferred_and_labelled(self):
        row = make_land_row()

        budget = RenewableLandBudget.from_land_row(row, TECH_INPUTS)

        self.assertEqual(budget.renewable_union_area_km2, 1_050.0)
        self.assertEqual(budget.union_area_source, CLASSWISE_NESTED_UNION_SOURCE)

        mapping = estimate_scaled_design_capacity(
            {
                "annual_ammonia_production_t": 1_000_000.0,
                "wind_mw": 0.0,
                "solar_mw": 100.0,
                "solar_tracking_mw": 0.0,
            },
            budget,
        ).to_mapping()
        self.assertTrue(mapping["renewable_union_area_is_lower_bound_approximation"])
        self.assertEqual(
            mapping["renewable_union_area_method"], CLASSWISE_NESTED_UNION_METHOD
        )
        self.assertEqual(
            mapping["renewable_union_area_method_version"],
            CLASSWISE_NESTED_UNION_VERSION,
        )

    def test_divergent_union_compatibility_alias_is_rejected(self):
        row = make_land_row()
        row["renewable_union_area_km2"] = 999_999.0

        with self.assertRaisesRegex(ValueError, "compatibility alias diverges"):
            RenewableLandBudget.from_land_row(row, TECH_INPUTS)

    def test_unversioned_explicit_union_is_not_mislabelled_as_v1(self):
        row = make_land_row()
        row.pop(CLASSWISE_NESTED_UNION_AREA_COLUMN)

        budget = RenewableLandBudget.from_land_row(row, TECH_INPUTS)

        self.assertEqual(budget.union_area_source, "explicit_legacy_unversioned")

    def test_tracking_uses_twice_fixed_pv_area(self):
        budget = RenewableLandBudget.from_land_row(make_land_row(), TECH_INPUTS)

        self.assertEqual(budget.fixed_solar_density_mw_per_km2, 100.0)
        self.assertEqual(budget.tracking_solar_density_mw_per_km2, 50.0)
        self.assertEqual(budget.tracking_land_multiplier, 2.0)
        self.assertEqual(budget.tracking_density_source, "tech_land_use_ratio")

        land_use = renewable_land_use_from_results(
            {
                "wind_mw": 0.0,
                "solar_mw": 300.0,
                "solar_tracking_mw": 200.0,
            },
            budget,
        )
        self.assertEqual(land_use.fixed_solar_km2, 3.0)
        self.assertEqual(land_use.tracking_solar_km2, 4.0)
        self.assertEqual(land_use.solar_km2, 7.0)

    def test_old_land_row_without_union_has_no_union_area(self):
        row = make_land_row()
        row.pop("renewable_union_area_km2")
        row.pop(CLASSWISE_NESTED_UNION_AREA_COLUMN)

        budget = RenewableLandBudget.from_land_row(row, TECH_INPUTS)

        self.assertIsNone(budget.renewable_union_area_km2)
        self.assertEqual(budget.union_area_source, "missing")


class ScaledDesignCapacityTest(unittest.TestCase):
    def test_scale_below_one_is_not_floored_to_reference_plant(self):
        budget = RenewableLandBudget(
            wind_area_km2=59.2,
            solar_area_km2=1_000.0,
            renewable_union_area_km2=1_000.0,
            wind_density_mw_per_km2=5.0,
            fixed_solar_density_mw_per_km2=100.0,
            tracking_solar_density_mw_per_km2=50.0,
            union_area_source="explicit",
            wind_area_source="wind_onshore_area_km2",
            tracking_density_source="tech_land_use_ratio",
        )
        results = {
            "annual_ammonia_production_t": 1_000_000.0,
            "gridless_ammonia_production_t": 1_000_000.0,
            "wind_mw": 500.0,  # 100 km2 in the reference design
            "solar_mw": 0.0,
            "solar_tracking_mw": 0.0,
        }

        capacity = estimate_scaled_design_capacity(
            results,
            budget,
            land_allocation="exclusive",
        )

        self.assertAlmostEqual(capacity.scale_factor, 0.592)
        self.assertAlmostEqual(capacity.capacity_t, 592_000.0)
        self.assertEqual(capacity.limiting_constraint, "wind")
        mapped = capacity.to_mapping()
        self.assertAlmostEqual(
            mapped["scaled_design_max_onshore_ammonia_capacity_t"], 592_000.0
        )
        self.assertNotIn("max_ammonia_capacity_t", mapped)

    def test_union_can_bind_when_individual_technology_areas_do_not(self):
        budget = RenewableLandBudget(
            wind_area_km2=100.0,
            solar_area_km2=100.0,
            renewable_union_area_km2=100.0,
            wind_density_mw_per_km2=5.0,
            fixed_solar_density_mw_per_km2=100.0,
            tracking_solar_density_mw_per_km2=50.0,
            union_area_source="explicit",
            wind_area_source="wind_onshore_area_km2",
            tracking_density_source="tech_land_use_ratio",
        )
        results = {
            "annual_ammonia_production_t": 1_000_000.0,
            "wind_mw": 400.0,  # 80 km2
            "solar_mw": 4_000.0,  # 40 km2
            "solar_tracking_mw": 0.0,
        }

        shared = estimate_scaled_design_capacity(
            results,
            budget,
            land_allocation="exclusive",
        )
        colocated_default = estimate_scaled_design_capacity(
            results,
            budget,
            land_allocation="colocated",
        )
        overlap_budget = RenewableLandBudget(
            **{**budget.__dict__, "wind_land_exclusive_fraction": 0.0}
        )
        complete_overlap = estimate_scaled_design_capacity(
            results,
            overlap_budget,
            land_allocation="colocated",
        )

        self.assertEqual(shared.limiting_constraint, "renewable_union")
        self.assertAlmostEqual(shared.scale_factor, 100.0 / 120.0)
        self.assertAlmostEqual(shared.wind_land_exclusive_fraction_effective, 1.0)
        # colocated: only 3 % of the 80 km2 wind footprint competes with PV for the union budget
        self.assertEqual(colocated_default.limiting_constraint, "wind")
        self.assertAlmostEqual(colocated_default.scale_factor, 1.25)
        self.assertAlmostEqual(colocated_default.wind_land_exclusive_fraction_effective, 0.03)
        self.assertAlmostEqual(
            min(100.0 / (40.0 + 0.03 * 80.0), 100.0 / 80.0, 100.0 / 40.0), 1.25
        )
        self.assertEqual(complete_overlap.limiting_constraint, "wind")
        self.assertAlmostEqual(complete_overlap.scale_factor, 1.25)

    def test_colocated_union_binds_through_wind_exclusive_fraction(self):
        budget = RenewableLandBudget(
            wind_area_km2=1_000.0,
            solar_area_km2=1_000.0,
            renewable_union_area_km2=100.0,
            wind_density_mw_per_km2=5.0,
            fixed_solar_density_mw_per_km2=100.0,
            tracking_solar_density_mw_per_km2=50.0,
            union_area_source="explicit",
            wind_area_source="wind_onshore_area_km2",
            tracking_density_source="tech_land_use_ratio",
            wind_land_exclusive_fraction=0.5,
        )
        results = {
            "annual_ammonia_production_t": 1_000_000.0,
            "wind_mw": 400.0,  # 80 km2, of which 40 km2 count against the shared budget
            "solar_mw": 4_000.0,  # 40 km2
            "solar_tracking_mw": 0.0,
        }
        capacity = estimate_scaled_design_capacity(results, budget, land_allocation="colocated")
        self.assertEqual(capacity.limiting_constraint, "renewable_union")
        self.assertAlmostEqual(capacity.scale_factor, 100.0 / 80.0)
        exclusive = estimate_scaled_design_capacity(results, budget, land_allocation="exclusive")
        self.assertAlmostEqual(exclusive.scale_factor, 100.0 / 120.0)

    def test_solved_quantity_report_records_the_exclusive_fraction_not_the_gridless_fraction(self):
        from model.land import report_land_feasible_quantity

        budget = RenewableLandBudget(
            wind_area_km2=1_000.0, solar_area_km2=1_000.0, renewable_union_area_km2=1_000.0,
            wind_density_mw_per_km2=5.0, fixed_solar_density_mw_per_km2=100.0,
            tracking_solar_density_mw_per_km2=50.0, union_area_source="explicit",
            wind_area_source="wind_onshore_area_km2", tracking_density_source="tech_land_use_ratio",
            wind_land_exclusive_fraction=0.03,
        )
        results = {"annual_ammonia_production_t": 1_000_000.0, "gridless_ammonia_production_t": 1_000_000.0,
                   "wind_mw": 400.0, "solar_mw": 4_000.0, "solar_tracking_mw": 0.0}
        report = report_land_feasible_quantity(results, budget, land_allocation="colocated")
        self.assertAlmostEqual(report["wind_land_exclusive_fraction_effective"], 0.03)
        self.assertAlmostEqual(report["land_feasible_gridless_ammonia_production_t"], 1_000_000.0)
        self.assertAlmostEqual(report["renewable_union_land_slack_km2"], 1_000.0 - (40.0 + 0.03 * 80.0))
        exclusive = report_land_feasible_quantity(results, budget, land_allocation="exclusive")
        self.assertAlmostEqual(exclusive["wind_land_exclusive_fraction_effective"], 1.0)
        self.assertAlmostEqual(exclusive["renewable_union_land_slack_km2"], 1_000.0 - 120.0)

    def test_only_the_two_allocation_names_are_accepted(self):
        from model.land import normalise_land_allocation

        self.assertEqual(normalise_land_allocation("colocated"), "colocated")
        self.assertEqual(normalise_land_allocation("exclusive"), "exclusive")
        for old_name in ("technology_shared", "paper_union", "independent_legacy", "shared"):
            with self.assertRaises(ValueError):
                normalise_land_allocation(old_name)

    def test_budget_reads_wind_exclusive_fraction_from_tech_inputs(self):
        inputs = {**TECH_INPUTS, "wind": {**TECH_INPUTS["wind"], "land_use_exclusive_fraction": 0.1}}
        budget = RenewableLandBudget.from_land_row(make_land_row(0.02), inputs)
        self.assertAlmostEqual(budget.wind_land_exclusive_fraction, 0.1)
        default = RenewableLandBudget.from_land_row(make_land_row(0.02), TECH_INPUTS)
        self.assertAlmostEqual(default.wind_land_exclusive_fraction, 0.03)
        with self.assertRaises(ValueError):
            RenewableLandBudget.from_land_row(
                make_land_row(0.02), {**TECH_INPUTS, "wind": {**TECH_INPUTS["wind"], "land_use_exclusive_fraction": 1.5}}
            )

    def test_capacity_is_monotonic_with_competition_fraction(self):
        results = {
            "annual_ammonia_production_t": 1_000_000.0,
            "wind_mw": 500.0,
            "solar_mw": 2_000.0,
            "solar_tracking_mw": 500.0,
        }
        capacities = []
        for fraction in (0.02, 0.50, 1.0):
            budget = RenewableLandBudget.from_land_row(
                make_land_row(fraction),
                TECH_INPUTS,
            )
            capacities.append(
                estimate_scaled_design_capacity(
                    results,
                    budget,
                    land_allocation="exclusive",
                ).capacity_t
            )

        self.assertLess(capacities[0], capacities[1])
        self.assertLess(capacities[1], capacities[2])
        self.assertAlmostEqual(capacities[1] / capacities[0], 25.0)
        self.assertAlmostEqual(capacities[2] / capacities[1], 2.0)
        self.assertTrue(all(math.isfinite(value) for value in capacities))



# --- land tables -----------------------------------------------------------------------------

import tempfile  # noqa: E402
from pathlib import Path  # noqa: E402

import pandas as pd  # noqa: E402
import pypsa  # noqa: E402
import pytest  # noqa: E402

from model import land, plant  # noqa: E402


def _write_land_table(root, **extra):
    frame = pd.DataFrame({
        "latitude": [-23.0], "longitude": [-69.0], "elevation_m": [2000.0],
        land.CLASSWISE_NESTED_UNION_AREA_COLUMN: [10.0],
        land.RENEWABLE_UNION_METHOD_COLUMN: [land.CLASSWISE_NESTED_UNION_METHOD],
        land.RENEWABLE_UNION_METHOD_VERSION_COLUMN: [land.CLASSWISE_NESTED_UNION_VERSION],
        **extra,
    })
    path = Path(root) / "land.csv"
    frame.to_csv(path, index=False)
    return path


def test_loader_accepts_only_centred_tables():
    with tempfile.TemporaryDirectory() as tmp:
        assert len(land.load_land_table(_write_land_table(tmp, cell_anchor=["center"]), {})) == 1
        for extra in ({}, {"cell_anchor": ["southwest"]}):
            with pytest.raises(ValueError, match="centred build"):
                land.load_land_table(_write_land_table(tmp, **extra), {})


def test_loader_requires_versioned_union_provenance():
    with tempfile.TemporaryDirectory() as tmp:
        path = _write_land_table(tmp, cell_anchor=["center"])
        frame = pd.read_csv(path).drop(columns=[land.RENEWABLE_UNION_METHOD_COLUMN])
        frame.to_csv(path, index=False)
        with pytest.raises(ValueError, match="versioned renewable-union provenance"):
            land.load_land_table(path, {})


TRACKING_TECH = {"solar": {"land_use_km2_per_mw": 0.01}, "solar_tracking": {"land_use_km2_per_mw": 0.02}}


def test_explicit_density_survives_and_legacy_tables_use_latitude_packing():
    table = pd.DataFrame({"latitude": [-23.0], "solar_density_mw_per_km2": [83.446157], "solar_area_km2": [100.0]})
    packed = land._apply_tech_solar_density(table, TRACKING_TECH)
    assert packed.solar_density_mw_per_km2.iloc[0] == pytest.approx(73.599292, abs=1e-5)
    table["solar_density_method"] = "explicit_fixed_pv_density_v1"
    explicit = land._apply_tech_solar_density(table, TRACKING_TECH)
    assert explicit.solar_density_mw_per_km2.iloc[0] == 83.446157
    assert explicit.solar_tracking_density_mw_per_km2.iloc[0] == pytest.approx(83.446157 / 2)
    assert explicit.max_power_solar_mw.iloc[0] == pytest.approx(8344.6157)


@pytest.mark.parametrize("density,method", [(0.0, "explicit_fixed_pv_density_v1"),
                                            (float("nan"), "explicit_fixed_pv_density_v1"), (100.0, "typo")])
def test_explicit_density_fails_closed(density, method):
    table = pd.DataFrame({"latitude": [0.0], "solar_density_mw_per_km2": [density], "solar_density_method": [method]})
    with pytest.raises(ValueError):
        land._apply_tech_solar_density(table, {})


# --- land in the solve (finite-site supply curves) ---------------------------------------------

def _budget(area=1.0):
    return RenewableLandBudget(
        wind_area_km2=area, solar_area_km2=area, renewable_union_area_km2=area,
        wind_density_mw_per_km2=1, fixed_solar_density_mw_per_km2=10, tracking_solar_density_mw_per_km2=5,
        union_area_source="explicit", wind_area_source="explicit", tracking_density_source="explicit",
    )


def test_solved_quantity_is_not_scaled_to_a_maximum():
    output = land.report_land_feasible_quantity(
        {"annual_ammonia_production_t": 250000, "gridless_ammonia_production_t": 250000,
         "wind_mw": 0, "solar_mw": 2, "solar_tracking_mw": 0},
        _budget(),
    )
    assert output["land_feasible_ammonia_production_t"] == 250000
    assert output["land_feasible_gridless_ammonia_production_t"] == 250000
    assert output["renewable_union_land_slack_km2"] == pytest.approx(0.8)
    assert output["quantity_is_maximum"] is False
    assert not any("max_" in key for key in output)
    assert output["preferred_supplier_capacity_column"] == ""


def test_solved_quantity_rejects_land_excess():
    with pytest.raises(ValueError, match="exceeds solar"):
        land.report_land_feasible_quantity(
            {"annual_ammonia_production_t": 1e6, "solar_mw": 11, "wind_mw": 0, "solar_tracking_mw": 0}, _budget()
        )


@pytest.mark.parametrize("demand,expected", [(8, "optimal"), (11, "infeasible")])
def test_shared_land_constraint_in_linopy(demand, expected):
    network = pypsa.Network()
    network.set_snapshots(range(2))
    network.add("Bus", "power")
    for name, cost in (("wind", 1000), ("solar", 2), ("solar_tracking", 1)):
        network.add("Generator", name, bus="power", p_nom_extendable=True, capital_cost=cost)
    network.add("Load", "demand", bus="power", p_set=demand)
    land.apply_land_budget(network, _budget(), "exclusive")
    status, condition = network.optimize(
        solver_name="highs", extra_functionality=plant.extra_constraints, solver_options={"log_to_console": False}
    )
    assert condition == expected
    if expected == "optimal":
        assert status == "ok"
        assert network.generators.at["solar", "p_nom_opt"] == pytest.approx(6)
        assert network.generators.at["solar_tracking", "p_nom_opt"] == pytest.approx(2)
        assert network.objective == pytest.approx(14)
