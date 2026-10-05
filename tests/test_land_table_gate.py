import tempfile
import unittest
from pathlib import Path

import pandas as pd

from model.run_global import _load_land_table, run_global


class LandTableGateTest(unittest.TestCase):
    """Runs accept only centred land builds (the earlier tables are offset by one degree)."""

    @staticmethod
    def _write(root, **extra):
        frame = pd.DataFrame(
            {"latitude": [-23.0], "longitude": [-69.0], "elevation_m": [2000.0], **extra}
        )
        path = root / "land.csv"
        frame.to_csv(path, index=False)
        return path

    def test_loader_accepts_only_centred_tables(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            table = _load_land_table(self._write(root, cell_anchor=["center"]))
            self.assertEqual(len(table), 1)
            for extra in ({}, {"cell_anchor": ["southwest"]}):
                with self.subTest(extra=extra):
                    with self.assertRaisesRegex(ValueError, "centred build"):
                        _load_land_table(self._write(root, **extra))

    def test_run_global_requires_a_land_table(self):
        with self.assertRaisesRegex(ValueError, "land_csv is required"):
            run_global(locations=[(-23.0, -69.0)])


if __name__ == "__main__":
    unittest.main()
