"""model/suppliers.py: green-lory surfaces as supplier tables of the network model."""
import json
from pathlib import Path
import tempfile
import unittest

import pandas as pd

from model.suppliers import CAPACITY, ELECTRICITY_COSTS, convert, derive_land_share, verify_derivation


class SupplierExportTests(unittest.TestCase):
    def setUp(self):
        self.row = {"latitude": -23, "longitude": -69, "country": "Chile",
                    "lcoa_eur_per_t": 180.48, CAPACITY: 2.5e6,
                    "annual_ammonia_production_t": 1e6, "currency": "EUR", "grid_energy_mwh": 0,
                    "is_gridless_feasible": True, "is_full_year_result": True,
                    "preferred_supplier_capacity_column": CAPACITY,
                    **dict.fromkeys(ELECTRICITY_COSTS, 10.)}
        self.reference = pd.DataFrame({"country": ["Chile"], "iso3": ["CHL"]})

    def test_units_and_capacity(self):
        result = convert(pd.DataFrame([self.row]), self.reference).iloc[0]
        self.assertEqual(result.Index, "-23_-69_1000000.0")
        self.assertAlmostEqual(result.LCOA, 200.)
        self.assertEqual(result.Max_capacity, 2.5)
        self.assertAlmostEqual(result.Electricity_Cost_Frac, 30/180.48)

    def test_no_capacity_fallback(self):
        self.row.pop(CAPACITY)
        self.row["max_gridless_onshore_ammonia_capacity_t"] = 9e6
        with self.assertRaises(ValueError):
            convert(pd.DataFrame([self.row]), self.reference)

    def test_unknown_country_rejected(self):
        self.row["country"] = "Unknown"
        with self.assertRaises(ValueError):
            convert(pd.DataFrame([self.row]), self.reference)

    def test_explicit_cutoff_excludes_only_subthreshold_unassigned_cells(self):
        unknown = dict(self.row, latitude=0, country=None)
        unknown[CAPACITY] = 999999.
        frame = pd.DataFrame([self.row, unknown])
        self.assertEqual(len(convert(frame, self.reference, minimum_capacity=1)), 1)
        frame.loc[1, CAPACITY] = 1e6
        with self.assertRaisesRegex(ValueError, "unassigned"):
            convert(frame, self.reference, minimum_capacity=1)

    def test_drop_unassigned_removes_nan_country_cells_only(self):
        unknown = dict(self.row, latitude=0, country=None)
        unknown[CAPACITY] = 2e6
        frame = pd.DataFrame([self.row, unknown])
        with self.assertRaisesRegex(ValueError, "unassigned"):
            convert(frame, self.reference, minimum_capacity=1)
        kept = convert(frame, self.reference, minimum_capacity=1, drop_unassigned=True)
        self.assertEqual(len(kept), 1)
        self.assertEqual(kept.iso3.iloc[0], self.reference.iso3.iloc[0])
        frame.loc[1, "country"] = "Unknown"
        with self.assertRaises(ValueError):
            convert(frame, self.reference, minimum_capacity=1, drop_unassigned=True)

    def test_non_gridless_rejected(self):
        self.row["is_gridless_feasible"] = False
        with self.assertRaises(ValueError):
            convert(pd.DataFrame([self.row]), self.reference)



class DerivationTest(unittest.TestCase):
    def test_derivation_must_match_parent_qa_and_output(self):
        prov = {"kind": "post_hoc_land_share_variant", "parent_surface": {"path": "p", "sha256": "abc"}, "multiplier": 0.1,
                "effective_land_competition_fraction": 0.02, "columns_rescaled": ["x"], "lcoa_columns_untouched": True,
                "output": {"path": "o", "sha256": "def"}}
        record = verify_derivation(prov, {"output_sha256": "abc"}, "def")
        self.assertEqual(record["multiplier"], 0.1)
        with self.assertRaises(ValueError):
            verify_derivation(prov, {"output_sha256": "zzz"}, "def")
        with self.assertRaises(ValueError):
            verify_derivation(prov, {"output_sha256": "abc"}, "other")
        with self.assertRaises(ValueError):
            verify_derivation(dict(prov, kind="something"), {"output_sha256": "abc"}, "def")


class LandShareTest(unittest.TestCase):
    def test_capacities_scale_and_costs_stay(self):
        with tempfile.TemporaryDirectory() as directory:
            surface = Path(directory) / "surface.csv"
            pd.DataFrame({"latitude": [-23.0], "longitude": [117.0], "lcoa_eur_per_t": [420.0],
                          "land_constraint": ["after_solve"], CAPACITY: [5e6],
                          "scaled_design_renewable_capacity_scale_factor": [2.0]}).to_csv(surface, index=False)
            provenance = derive_land_share(surface, 0.1, 0.20, Path(directory) / "two_percent")
            derived = pd.read_csv(Path(directory) / "two_percent" / "global_run_results.csv")
            self.assertAlmostEqual(derived[CAPACITY].iloc[0], 5e5)
            self.assertAlmostEqual(derived["scaled_design_renewable_capacity_scale_factor"].iloc[0], 0.2)
            self.assertEqual(derived["lcoa_eur_per_t"].iloc[0], 420.0)
            self.assertAlmostEqual(provenance["effective_land_competition_fraction"], 0.02)
            self.assertEqual(json.loads((Path(directory) / "two_percent" / "provenance.json").read_text())["multiplier"], 0.1)

    def test_only_after_solve_surfaces_rescale(self):
        with tempfile.TemporaryDirectory() as directory:
            surface = Path(directory) / "surface.csv"
            pd.DataFrame({"latitude": [0.0], "longitude": [0.0], "land_constraint": ["in_solve"], CAPACITY: [1e6]}).to_csv(surface, index=False)
            with self.assertRaises(ValueError):
                derive_land_share(surface, 0.1, 0.20, Path(directory) / "out")
