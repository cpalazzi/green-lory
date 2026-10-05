import unittest
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

from model import auxiliary as aux
from model.main import generate_network


REPO_ROOT = Path(__file__).resolve().parents[1]
PLANT_DIR = REPO_ROOT / "basic_ammonia_plant_2050_way_tracking"


class TemporalAccountingTest(unittest.TestCase):
    def test_omitted_mode_preserves_legacy_low_level_api(self):
        raw_stores = pd.read_csv(PLANT_DIR / "stores.csv", index_col=0)

        network = generate_network(
            8,
            PLANT_DIR,
            aggregation_count=2,
            time_step=4.0,
        )

        self.assertEqual(len(network.snapshots), 4)
        self.assertEqual(network._temporal_accounting_mode, "legacy_scaled")
        self.assertEqual(
            network.stores.at["compressed_hydrogen_store", "capital_cost"],
            raw_stores.at["compressed_hydrogen_store", "capital_cost"] * 8.0,
        )

    def test_snapshot_weighted_mode_uses_physical_duration(self):
        raw_stores = pd.read_csv(PLANT_DIR / "stores.csv", index_col=0)
        network = generate_network(
            4,
            PLANT_DIR,
            aggregation_count=1,
            time_step=4.0,
            temporal_accounting_mode="snapshot_weighted",
        )

        self.assertEqual(len(network.snapshots), 4)
        self.assertTrue((network.snapshot_weightings["objective"] == 4.0).all())
        self.assertTrue((network.snapshot_weightings["stores"] == 4.0).all())
        self.assertEqual(
            network.stores.at["compressed_hydrogen_store", "capital_cost"],
            raw_stores.at["compressed_hydrogen_store", "capital_cost"],
        )
        self.assertEqual(network._temporal_accounting_mode, "snapshot_weighted")

    def test_legacy_mode_is_explicit_and_retains_store_scaling(self):
        raw_stores = pd.read_csv(PLANT_DIR / "stores.csv", index_col=0)
        network = generate_network(
            4,
            PLANT_DIR,
            aggregation_count=1,
            time_step=4.0,
            temporal_accounting_mode="legacy_scaled",
        )

        self.assertEqual(
            network.stores.at["compressed_hydrogen_store", "capital_cost"],
            raw_stores.at["compressed_hydrogen_store", "capital_cost"] * 4.0,
        )
        self.assertEqual(network._temporal_accounting_mode, "legacy_scaled")

    def test_snapshot_weighted_rejects_overlapping_aggregation(self):
        with self.assertRaisesRegex(ValueError, "aggregation_count=1"):
            generate_network(
                4,
                PLANT_DIR,
                aggregation_count=2,
                time_step=4.0,
                temporal_accounting_mode="snapshot_weighted",
            )

    def test_per_hour_ramp_clears_native_limits_and_preserves_scaled_custom_rate(self):
        network = generate_network(
            4,
            PLANT_DIR,
            aggregation_count=1,
            time_step=4.0,
            temporal_accounting_mode="snapshot_weighted",
        )
        network._ramp_limit_basis = "per_hour"
        network.links.at["ammonia_synthesis", "ramp_limit_up"] = 0.1
        network.links.at["ammonia_synthesis", "ramp_limit_down"] = 0.15

        aux.prepare_ammonia_ramp_constraints(network)

        self.assertTrue(
            pd.isna(network.links.at["ammonia_synthesis", "ramp_limit_up"])
        )
        self.assertTrue(
            pd.isna(network.links.at["ammonia_synthesis", "ramp_limit_down"])
        )
        self.assertAlmostEqual(
            aux._ammonia_ramp_limit_per_snapshot(network, "up"), 0.4
        )
        self.assertAlmostEqual(
            aux._ammonia_ramp_limit_per_snapshot(network, "down"), 0.6
        )

        # Native PyPSA link-ramp constraints are absent after clearing, while
        # Green Lory's duration-aware constraints are installed explicitly.
        network.optimize.create_model()
        aux.linopy_constraints(network, network.snapshots)
        constraint_names = set(network.model.constraints)
        self.assertNotIn("Link-ext-p-ramp_limit_up", constraint_names)
        self.assertNotIn("Link-ext-p-ramp_limit_down", constraint_names)
        self.assertIn("ammonia_synthesis_ramp_up", constraint_names)
        self.assertIn("ammonia_synthesis_ramp_down", constraint_names)

    def test_legacy_ramp_keeps_native_snapshot_rates(self):
        network = generate_network(
            4,
            PLANT_DIR,
            aggregation_count=1,
            time_step=4.0,
            temporal_accounting_mode="legacy_scaled",
        )
        network._ramp_limit_basis = "legacy_per_snapshot"
        up_before = network.links.at["ammonia_synthesis", "ramp_limit_up"]

        aux.prepare_ammonia_ramp_constraints(network)

        self.assertEqual(
            network.links.at["ammonia_synthesis", "ramp_limit_up"], up_before
        )
        self.assertEqual(
            aux._ammonia_ramp_limit_per_snapshot(network, "up"), up_before
        )

    @staticmethod
    def _reporting_network(temporal_mode):
        snapshots = pd.RangeIndex(2)
        links = pd.DataFrame(
            {
                "p_nom_opt": [1.0] * 8,
                "carrier": ["test"] * 8,
                "bus0": ["source"] * 8,
                "bus2": ["secondary"] * 8,
                "efficiency": [1.0] * 8,
            },
            index=[
                "electrolysis",
                "hydrogen_compression",
                "hydrogen_from_storage",
                "ammonia_synthesis",
                "battery_pcs_charge",
                "battery_pcs_discharge",
                "hydrogen_fuel_cell",
                "penalty_link",
            ],
        )
        stores = pd.DataFrame(
            {"e_nom_opt": [10.0]},
            index=["compressed_hydrogen_store"],
        )
        generators = pd.DataFrame(
            {"p_nom_opt": [1.0]},
            index=["wind"],
        )
        link_flows = pd.DataFrame(0.0, index=snapshots, columns=links.index)
        network = SimpleNamespace(
            links=links,
            links_t=SimpleNamespace(p0=link_flows.copy(), p2=link_flows.copy()),
            stores=stores,
            stores_t=SimpleNamespace(
                e=pd.DataFrame(
                    {"compressed_hydrogen_store": [2.0, 3.0]},
                    index=snapshots,
                )
            ),
            generators=generators,
            generators_t=SimpleNamespace(
                p=pd.DataFrame({"wind": [0.0, 0.0]}, index=snapshots)
            ),
            loads=pd.DataFrame({"p_set": [1.0]}, index=["ammonia"]),
            objective=1.0,
            _temporal_accounting_mode=temporal_mode,
        )
        return network

    def test_excel_reports_physical_store_mwh_for_snapshot_weighted_network(self):
        network = self._reporting_network("snapshot_weighted")

        output = aux.get_results_dict_for_excel(
            network,
            scale=2.0,
            aggregation_count=3,
            time_step=4.0,
        )

        self.assertEqual(
            output["Stores"].at[
                "compressed_hydrogen_store", "Storage Capacity (MWh)"
            ],
            20.0,
        )
        self.assertEqual(
            output["Stored energy capacity (MWh)"].iloc[-1, 0],
            6.0,
        )

    def test_excel_preserves_legacy_store_reporting_scale(self):
        network = self._reporting_network("legacy_scaled")

        output = aux.get_results_dict_for_excel(
            network,
            scale=2.0,
            aggregation_count=3,
            time_step=4.0,
        )

        self.assertEqual(
            output["Stores"].at[
                "compressed_hydrogen_store", "Storage Capacity (MWh)"
            ],
            240.0,
        )
        self.assertEqual(
            output["Stored energy capacity (MWh)"].iloc[-1, 0],
            72.0,
        )

    def test_excel_treats_unlabelled_historical_network_as_legacy(self):
        network = self._reporting_network("legacy_scaled")
        del network._temporal_accounting_mode

        output = aux.get_results_dict_for_excel(
            network,
            scale=1.0,
            aggregation_count=2,
            time_step=4.0,
        )

        self.assertEqual(
            output["Stores"].at[
                "compressed_hydrogen_store", "Storage Capacity (MWh)"
            ],
            80.0,
        )


if __name__ == "__main__":
    unittest.main()
