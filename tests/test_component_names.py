import unittest

from model.main import normalize_component_name


class ComponentNameTest(unittest.TestCase):
    def test_battery_storage_normalizes_to_current_store_name(self):
        self.assertEqual(normalize_component_name("battery_storage"), "battery_storage")
        self.assertEqual(normalize_component_name("Battery"), "battery_storage")
        self.assertEqual(normalize_component_name("BatteryStorage"), "battery_storage")


if __name__ == "__main__":
    unittest.main()