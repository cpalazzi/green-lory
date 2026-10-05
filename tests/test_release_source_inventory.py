import json
from pathlib import Path
import subprocess
import tempfile
import unittest

from arc.release_source_inventory import verify_inventory, write_inventory


class ReleaseSourceInventoryTest(unittest.TestCase):
    def _repository(self, root: Path) -> None:
        subprocess.run(["git", "init", "-q"], cwd=root, check=True)
        (root / "tracked.txt").write_text("tracked\n", encoding="utf-8")
        (root / "untracked.txt").write_text("untracked\n", encoding="utf-8")
        (root / "results").mkdir()
        (root / "results" / "output.csv").write_text("generated\n", encoding="utf-8")
        subprocess.run(["git", "add", "tracked.txt"], cwd=root, check=True)

    def test_round_trip_and_results_exclusion(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self._repository(root)
            output = root.parent / f"{root.name}-inventory.json"
            try:
                payload = write_inventory(root, output)
                paths = {entry["path"] for entry in payload["entries"]}
                self.assertEqual(paths, {"tracked.txt", "untracked.txt"})
                summary = verify_inventory(root, output)
                self.assertEqual(summary["status"], "passed")
                self.assertEqual(summary["tree_sha256"], payload["tree_sha256"])
            finally:
                output.unlink(missing_ok=True)

    def test_byte_mutation_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self._repository(root)
            output = root.parent / f"{root.name}-inventory.json"
            try:
                write_inventory(root, output)
                (root / "tracked.txt").write_text("changed\n", encoding="utf-8")
                with self.assertRaisesRegex(ValueError, "differs from inventory"):
                    verify_inventory(root, output)
            finally:
                output.unlink(missing_ok=True)

    def test_tampered_tree_digest_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self._repository(root)
            output = root.parent / f"{root.name}-inventory.json"
            try:
                payload = write_inventory(root, output)
                payload["tree_sha256"] = "0" * 64
                output.write_text(json.dumps(payload), encoding="utf-8")
                with self.assertRaisesRegex(ValueError, "tree digest mismatch"):
                    verify_inventory(root, output)
            finally:
                output.unlink(missing_ok=True)


if __name__ == "__main__":
    unittest.main()
