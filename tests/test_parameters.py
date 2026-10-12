"""model/parameters.py and inputs/parameters.yaml: every value in the file is read by the model."""
from __future__ import annotations

import importlib

import pytest

from model import parameters

READERS = ("model.config", "model.land_table", "model.weather", "model.weather_land", "model.site_geography", "model.site_costs")


def test_the_model_reads_every_value_in_the_file():
    for module in READERS:
        importlib.import_module(module)
    read = parameters.keys_read()
    unread = [key for key in parameters.leaf_keys()
              if not any(key == done or key.startswith(done + ".") for done in read)]
    assert not unread, f"inputs/parameters.yaml values that no module reads: {unread}"


def test_a_missing_key_is_named():
    with pytest.raises(KeyError, match="land.no_such_value"):
        parameters.parameter("land.no_such_value")


def test_slope_classes_follow_the_bounds():
    from model.config import SLOPE_CLASS_COLUMNS
    from model.site_costs import SLOPE_CLASS_FACTORS

    assert SLOPE_CLASS_COLUMNS == ("land_pct_slope_upto_8p5deg", "land_pct_slope_8p5_to_14deg",
                                   "land_pct_slope_14_to_20deg", "land_pct_slope_above_20deg")
    assert list(SLOPE_CLASS_FACTORS) == list(SLOPE_CLASS_COLUMNS[:-1])


def test_file_record_is_relative_to_the_repository():
    record = parameters.file_record()
    assert record["path"] == "inputs/parameters.yaml" and len(record["sha256"]) == 64
