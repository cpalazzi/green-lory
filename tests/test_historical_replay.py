"""Guard input semantics independently of the large network optimisation."""
from pathlib import Path
import contextlib
import importlib.util
import io
import json
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np
import pandas as pd

from reconciliation.run_historical_network import extract_demand, select_suppliers, great_circle_km


class HistoricalInputTests(unittest.TestCase):
    def test_trade_average_and_conversion(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "demand.csv"
            pd.DataFrame([
                {"id": "p1", "name": "one", "scenario": f"SSP2-RCP4.5-{trade}", "Fuel_tons_route_future": fuel}
                for trade, fuel in [("Multi", 10), ("Multi", 20), ("Reg", 50)]
            ]).to_csv(path, index=False)
            value = extract_demand(path, "4.5", .7).Fuel_consumption.iloc[0]
            self.assertAlmostEqual(value, 40 * 39 / 18.6 * .7)
            improved = extract_demand(path, "4.5", .7, .1).Fuel_consumption.iloc[0]
            self.assertAlmostEqual(improved, value * .9)

    def test_missing_trade_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "demand.csv"
            pd.DataFrame([{"id": "p1", "name": "one", "scenario": "SSP2-RCP4.5-Multi", "Fuel_tons_route_future": 10}]).to_csv(path, index=False)
            with self.assertRaises(ValueError):
                extract_demand(path, "4.5", .7)

    def test_archived_duplicate_semantics(self):
        frame = pd.DataFrame({
            "Index": ["a", "a", "b", "c"], "LCOA": [100, 100, 200, 50],
            "Production": [1e6]*4, "Max_capacity": [2, 2, 3, .5],
            "Latitude": [0]*4, "Longitude": [0]*4, "iso3": ["X", "Y", "Z", "Z"],
        })
        selected, effective, duplicate = select_suppliers(frame, count=2)
        self.assertEqual(len(selected), 2)
        self.assertEqual(len(effective), 1)
        self.assertEqual(len(duplicate), 2)
        self.assertEqual(effective.iloc[0].iso3, "Y")
        frame.loc[1, "Max_capacity"] = 4
        with self.assertRaises(ValueError):
            select_suppliers(frame, count=2)

    def test_reference_plant_units(self):
        frame = pd.DataFrame({"Index": ["a"], "LCOA": [100], "Production": [2e6],
                              "Max_capacity": [2], "Latitude": [0], "Longitude": [0]})
        with self.assertRaises(ValueError):
            select_suppliers(frame)

    def test_geodesic_distance(self):
        suppliers = pd.DataFrame({"Latitude": [0], "Longitude": [0]})
        ports = pd.DataFrame({"lat": [0, 0], "lon": [0, 90]})
        np.testing.assert_allclose(great_circle_km(suppliers, ports), [[0, 6371*np.pi/2]])

    @unittest.skipUnless(importlib.util.find_spec("pyomo"), "Optional solver parity test")
    def test_sparse_matches_original_with_maritime_storage(self):
        import xarray as xr
        from reconciliation.run_historical_network import solve
        from reconciliation.sparse_network import solve_sparse
        campaign = Path(__file__).resolve().parents[1] / "results/campaigns/verschuur_reconcile_20260907_v1"
        archive = campaign / "historical/git-0a63616"
        published = campaign / "historical/mendeley-v1"
        if not published.exists():
            self.skipTest("Published source not extracted")
        suppliers = pd.read_csv(archive / "data/c_NH3_cost_4.5.csv")
        suppliers = suppliers[suppliers.Index.isin(["-28_-69_1000000.0", "-23_-67_1000000.0"])].copy()
        suppliers.loc[suppliers.Index == "-23_-67_1000000.0", "LCOA"] = 10000
        suppliers["subsidy"] = 0
        ports = pd.read_csv(archive / "data/c_port_data.csv")
        ports = ports[ports.name.isin(["Puerto San Antonio_Chile", "Bahia De Valparaiso_Chile"])].copy()
        demand = pd.DataFrame({"name": ports.name, "Fuel_consumption": [100000., 200000.]})
        onshore = xr.load_dataset(archive / "data/n_onshore_distances_4.5.nc").sel(suppliers=suppliers.Index.tolist(), ports=ports.name.tolist())
        onshore.distances.values[:, 1] = -1  # Force a maritime leg and active-port storage.
        offshore = xr.load_dataset(archive / "data/n_return_costs_4.5_Panamax.nc").sel(supply_ports=ports.name.tolist(), demand_ports=ports.name.tolist())
        for published_code in [None, published]:
            with self.subTest(published=bool(published_code)), tempfile.TemporaryDirectory() as directory:
                report = {"demand_t_per_year": 300000., "inputs": {}, "gpo_source_sha256": {}}
                results = []
                for name, function in [("original", solve), ("sparse", solve_sparse)]:
                    output = Path(directory) / name
                    output.mkdir()
                    args = SimpleNamespace(archive=archive, published_code=published_code, output=output,
                                           threads=1, gap=1e-6, time_limit=60, solver_memory_gb=4)
                    with contextlib.redirect_stdout(io.StringIO()):
                        function(args, report, suppliers, ports, demand, onshore, offshore)
                    result = json.loads((output / "summary.json").read_text())
                    self.assertTrue(result["qa_pass"])
                    results.append(result["annual_cost_USD"])
                self.assertAlmostEqual(results[0], results[1], places=3)

    @unittest.skipUnless(importlib.util.find_spec("pyomo"), "Optional pinned GPO solver integration test")
    def test_solver_export_keeps_zero_production_values(self):
        import xarray as xr
        from reconciliation.run_historical_network import solve
        archive = Path(__file__).resolve().parents[1] / "results/campaigns/verschuur_reconcile_20260907_v1/historical/git-0a63616"
        if not archive.exists():
            self.skipTest("Pinned historical data not extracted")
        data = archive / "data"
        suppliers = pd.read_csv(data / "c_NH3_cost_4.5.csv")
        suppliers = suppliers[suppliers.Index.isin(["-28_-69_1000000.0", "-23_-67_1000000.0"])].copy()
        self.assertEqual(len(suppliers), 2)
        suppliers.loc[suppliers.Index == "-23_-67_1000000.0", "LCOA"] = 10000
        suppliers["subsidy"] = 0
        ports = pd.read_csv(data / "c_port_data.csv")
        ports = ports[ports.name.isin(["Puerto San Antonio_Chile", "Bahia De Valparaiso_Chile"])].copy()
        demand = pd.DataFrame({"name": ports.name, "Fuel_consumption": [100000., 200000.]})
        onshore = xr.load_dataset(data / "n_onshore_distances_4.5.nc").sel(suppliers=suppliers.Index.tolist(), ports=ports.name.tolist())
        offshore = xr.load_dataset(data / "n_return_costs_4.5_Panamax.nc").sel(supply_ports=ports.name.tolist(), demand_ports=ports.name.tolist())
        with tempfile.TemporaryDirectory(prefix="gpo-replay-regression-") as directory:
            args = SimpleNamespace(archive=archive, output=Path(directory), threads=1, gap=.001, time_limit=60)
            with contextlib.redirect_stdout(io.StringIO()):
                solve(args, {"demand_t_per_year": 300000., "inputs": {}, "gpo_source_sha256": {}},
                      suppliers, ports, demand, onshore, offshore)
            result = pd.read_csv(Path(directory) / "suppliers.csv")
            self.assertEqual((result.production_t_per_year == 0).sum(), 1)
            self.assertTrue(json.loads((Path(directory) / "summary.json").read_text())["qa_pass"])


if __name__ == "__main__":
    unittest.main()
