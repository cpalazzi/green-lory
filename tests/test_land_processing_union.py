import sys
import types
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd

from model.land_processing import (
    FINAL_COLUMNS,
    LandAvailabilityConfig,
    _aggregate_availability,
    _aggregate_availability_from_hdf4,
    _apply_land_exclusions,
    apply_land_competition_scenario,
)
from model.land_union import (
    CLASSWISE_NESTED_UNION_AREA_COLUMN,
    CLASSWISE_NESTED_UNION_AVAILABILITY_COLUMN,
    CLASSWISE_NESTED_UNION_METHOD,
    CLASSWISE_NESTED_UNION_VERSION,
    RENEWABLE_UNION_METHOD_COLUMN,
    RENEWABLE_UNION_METHOD_VERSION_COLUMN,
)


class RenewableUnionAggregationTest(unittest.TestCase):
    def test_classwise_union_covers_wind_only_solar_only_and_shared_classes(self):
        cases = (
            (12, 0.05, 0.0, 0.05),  # cropland: wind only
            (13, 0.0, 0.03, 0.03),  # urban: solar only
            (6, 0.5, 0.5, 0.5),  # shrubland: shared
        )
        for modis_class, expected_wind, expected_solar, expected_union in cases:
            with self.subTest(modis_class=modis_class):
                frame = pd.DataFrame(
                    {
                        "latitude": [0.0],
                        "longitude": [0.0],
                        "modis_class": [modis_class],
                        "class_fraction": [1.0],
                    }
                )

                result = _aggregate_availability(frame, include_offshore_wind=False).iloc[0]

                self.assertAlmostEqual(result["wind_onshore_availability"], expected_wind)
                self.assertAlmostEqual(result["solar_availability"], expected_solar)
                self.assertAlmostEqual(result["renewable_union_availability"], expected_union)
                self.assertAlmostEqual(
                    result[CLASSWISE_NESTED_UNION_AVAILABILITY_COLUMN],
                    expected_union,
                )

    def test_shared_class_is_labelled_as_nested_overlap_lower_bound(self):
        frame = pd.DataFrame(
            {
                "latitude": [0.0],
                "longitude": [0.0],
                "modis_class": [6],
                "class_fraction": [1.0],
            }
        )

        result = _aggregate_availability(frame, include_offshore_wind=False).iloc[0]

        # Both technologies claim 50% of this class.  Without pixel masks their
        # union could lie between 50% (perfect overlap) and 100% (disjoint).
        self.assertEqual(result[CLASSWISE_NESTED_UNION_AVAILABILITY_COLUMN], 0.5)
        self.assertEqual(
            result[RENEWABLE_UNION_METHOD_COLUMN], CLASSWISE_NESTED_UNION_METHOD
        )
        self.assertEqual(
            result[RENEWABLE_UNION_METHOD_VERSION_COLUMN],
            CLASSWISE_NESTED_UNION_VERSION,
        )

    def test_union_is_sum_of_classwise_maxima_not_maximum_of_technology_sums(self):
        frame = pd.DataFrame(
            {
                "latitude": [0.0, 0.0, 0.0],
                "longitude": [0.0, 0.0, 0.0],
                "modis_class": [12, 13, 6],
                "class_fraction": [0.2, 0.3, 0.5],
            }
        )

        result = _aggregate_availability(frame, include_offshore_wind=False).iloc[0]

        self.assertAlmostEqual(result["wind_onshore_availability"], 0.26)
        self.assertAlmostEqual(result["solar_availability"], 0.259)
        self.assertAlmostEqual(result["renewable_union_availability"], 0.269)
        self.assertGreater(
            result["renewable_union_availability"],
            max(result["wind_onshore_availability"], result["solar_availability"]),
        )

    def test_hdf4_fast_path_uses_the_same_classwise_union(self):
        cube = np.zeros((1, 1, 17), dtype=float)
        cube[0, 0, 12] = 20.0
        cube[0, 0, 13] = 30.0
        cube[0, 0, 6] = 50.0

        class FakeDataset:
            def info(self):
                return ("Land_Cover_Type_1_Percent", 3, cube.shape, 0, 0)

            def __getitem__(self, key):
                return cube[key]

        class FakeFile:
            def __init__(self, *_args, **_kwargs):
                pass

            def select(self, name):
                self.name = name
                return FakeDataset()

            def end(self):
                pass

        fake_sd_module = types.ModuleType("pyhdf.SD")
        fake_sd_module.SD = FakeFile
        fake_sd_module.SDC = types.SimpleNamespace(READ=0)
        fake_package = types.ModuleType("pyhdf")
        fake_package.SD = fake_sd_module

        config = LandAvailabilityConfig(
            land_cover_path=Path("synthetic.hdf"),
            fine_degree=1.0,
            coarse_degree=1.0,
            lat_bounds=(89.0, 90.0),
            include_offshore_wind=False,
            cell_anchor="southwest",  # one fine pixel per cell: centred cells need an even ratio
        )
        with mock.patch.dict(
            sys.modules,
            {"pyhdf": fake_package, "pyhdf.SD": fake_sd_module},
        ):
            result = _aggregate_availability_from_hdf4(config).iloc[0]

        self.assertAlmostEqual(result["wind_onshore_availability"], 0.26)
        self.assertAlmostEqual(result["solar_availability"], 0.259)
        self.assertAlmostEqual(result["renewable_union_availability"], 0.269)

    def test_union_columns_are_part_of_the_stable_output_schema(self):
        self.assertIn(CLASSWISE_NESTED_UNION_AVAILABILITY_COLUMN, FINAL_COLUMNS)
        self.assertIn(CLASSWISE_NESTED_UNION_AREA_COLUMN, FINAL_COLUMNS)
        self.assertIn("renewable_union_availability", FINAL_COLUMNS)
        self.assertIn("renewable_union_area_km2", FINAL_COLUMNS)
        self.assertIn(RENEWABLE_UNION_METHOD_COLUMN, FINAL_COLUMNS)
        self.assertIn(RENEWABLE_UNION_METHOD_VERSION_COLUMN, FINAL_COLUMNS)


class RenewableUnionScalingTest(unittest.TestCase):
    @staticmethod
    def _baseline_frame() -> pd.DataFrame:
        return pd.DataFrame(
            {
                "latitude": [0.0],
                "longitude": [0.0],
                "area": [1_000.0],
                "land_exclusion_factor": [0.8],
                "constrained_onshore_area_km2": [800.0],
                "wind_onshore_availability": [0.2],
                "wind_offshore_availability": [0.0],
                "wind_availability": [0.2],
                "solar_availability": [0.3],
                CLASSWISE_NESTED_UNION_AVAILABILITY_COLUMN: [0.4],
                "renewable_union_availability": [0.4],
                "wind_onshore_area_km2": [200.0],
                "wind_offshore_area_km2": [0.0],
                "wind_area_km2": [200.0],
                "solar_area_km2": [300.0],
                CLASSWISE_NESTED_UNION_AREA_COLUMN: [400.0],
                "renewable_union_area_km2": [400.0],
                RENEWABLE_UNION_METHOD_COLUMN: [CLASSWISE_NESTED_UNION_METHOD],
                RENEWABLE_UNION_METHOD_VERSION_COLUMN: [
                    CLASSWISE_NESTED_UNION_VERSION
                ],
                "wind_density_mw_per_km2": [5.0],
                "solar_density_mw_per_km2": [100.0],
            }
        )

    def test_union_survives_land_exclusions(self):
        frame = pd.DataFrame(
            {
                "latitude": [0.0],
                "longitude": [0.0],
                "area": [1_000.0],
                "onshore_land_pct": [100.0],
                "onshore_area_km2": [1_000.0],
                "solar_availability": [0.3],
                "wind_onshore_availability": [0.2],
                "wind_offshore_availability": [0.0],
                "wind_availability": [0.2],
                CLASSWISE_NESTED_UNION_AVAILABILITY_COLUMN: [0.4],
                "renewable_union_availability": [0.4],
                "availability": [0.3],
            }
        )
        config = LandAvailabilityConfig(skip_slope_exclusion=True)

        def protected_half(input_frame, *_args, **_kwargs):
            result = input_frame.copy()
            result["protected_area_pct"] = 50.0
            return result

        with mock.patch(
            "model.land_processing._protected_area_pct_from_vectors",
            side_effect=protected_half,
        ):
            result = _apply_land_exclusions(frame, config).iloc[0]

        self.assertAlmostEqual(result["land_exclusion_factor"], 0.5)
        self.assertAlmostEqual(
            result[CLASSWISE_NESTED_UNION_AVAILABILITY_COLUMN], 0.2
        )
        self.assertAlmostEqual(result["renewable_union_availability"], 0.2)
        self.assertAlmostEqual(result["availability"], 0.15)

    def test_two_and_fifty_percent_rescaling_is_linear_and_not_compounded(self):
        baseline = self._baseline_frame()
        two_percent = apply_land_competition_scenario(baseline, 0.02)
        fifty_percent_direct = apply_land_competition_scenario(baseline, 0.50)
        fifty_percent_from_two = apply_land_competition_scenario(two_percent, 0.50)

        two = two_percent.iloc[0]
        direct = fifty_percent_direct.iloc[0]
        sequential = fifty_percent_from_two.iloc[0]

        self.assertAlmostEqual(two["renewable_union_availability"], 0.008)
        self.assertAlmostEqual(two["renewable_union_area_km2"], 8.0)
        self.assertAlmostEqual(
            two[CLASSWISE_NESTED_UNION_AVAILABILITY_COLUMN], 0.008
        )
        self.assertAlmostEqual(two[CLASSWISE_NESTED_UNION_AREA_COLUMN], 8.0)
        self.assertAlmostEqual(two["availability"], 0.006)
        self.assertAlmostEqual(direct["renewable_union_availability"], 0.2)
        self.assertAlmostEqual(direct["renewable_union_area_km2"], 200.0)
        self.assertAlmostEqual(
            direct["renewable_union_availability"],
            sequential["renewable_union_availability"],
        )
        self.assertAlmostEqual(
            direct["renewable_union_area_km2"],
            sequential["renewable_union_area_km2"],
        )
        self.assertAlmostEqual(
            direct["renewable_union_area_km2"] / two["renewable_union_area_km2"],
            25.0,
        )

    def test_legacy_frame_without_union_columns_still_rescales(self):
        legacy = self._baseline_frame().drop(
            columns=[
                CLASSWISE_NESTED_UNION_AVAILABILITY_COLUMN,
                CLASSWISE_NESTED_UNION_AREA_COLUMN,
                "renewable_union_availability",
                "renewable_union_area_km2",
                RENEWABLE_UNION_METHOD_COLUMN,
                RENEWABLE_UNION_METHOD_VERSION_COLUMN,
            ]
        )

        result = apply_land_competition_scenario(legacy, 0.02)

        self.assertNotIn("renewable_union_availability", result.columns)
        self.assertNotIn("renewable_union_area_km2", result.columns)
        self.assertNotIn(CLASSWISE_NESTED_UNION_AVAILABILITY_COLUMN, result.columns)
        self.assertNotIn(CLASSWISE_NESTED_UNION_AREA_COLUMN, result.columns)
        self.assertAlmostEqual(result.iloc[0]["solar_area_km2"], 6.0)

    def test_versioned_column_creates_identical_generic_alias_not_reverse(self):
        versioned = self._baseline_frame().drop(
            columns=["renewable_union_availability", "renewable_union_area_km2"]
        )

        result = apply_land_competition_scenario(versioned, 0.02)

        self.assertTrue(
            result["renewable_union_availability"].equals(
                result[CLASSWISE_NESTED_UNION_AVAILABILITY_COLUMN]
            )
        )
        self.assertTrue(
            result["renewable_union_area_km2"].equals(
                result[CLASSWISE_NESTED_UNION_AREA_COLUMN]
            )
        )


if __name__ == "__main__":
    unittest.main()
