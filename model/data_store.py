"""Creates a class in which the optimisation driver stores results."""

from __future__ import annotations

from typing import Any, Dict

import pandas as pd


def _apply_unit_suffixes(location_results: Dict[str, Any]) -> Dict[str, Any]:
    """Return a copy of location results with unit suffixes added where helpful.

    Notes:
    - Latitude/longitude are kept as-is (units are obvious).
    - Interest rates are stored as fractions (e.g. 0.0872) and kept as `interest_rate_*`.
    - Capacities from generators/links are in MW; store energies are in MWh.
    """
    rename: Dict[str, str] = {}
    keys = set(location_results.keys())

    # Interest rates: keep as `interest_rate_*` (suffix can be misleading).

    # Component capacities/energies: names originate from network component labels.
    mw_components = {
        "wind",
        "solar",
        "solar_tracking",
        "grid",
        "ramp_dummy",
        "electrolysis",
        "hydrogen_compression",
        "hydrogen_from_storage",
        "ammonia_synthesis",
        "battery_interface_in",
        "battery_interface_out",
        "hydrogen_fuel_cell",
        "penalty_link",
    }
    mwh_components = {
        "ammonia",
        "compressed_hydrogen_store",
        "battery",
        "accumulated_penalty",
    }
    for key in keys:
        key_lower = str(key).lower()
        if key_lower in mw_components:
            rename[key] = f"{key_lower}_mw"
        elif key_lower in mwh_components:
            rename[key] = f"{key_lower}_mwh"

    # Other scalar quantities where the unit isn't otherwise obvious.
    if "hydrogen_storage_capacity" in keys:
        rename["hydrogen_storage_capacity"] = "hydrogen_storage_capacity_t"
    if "land_capacity_cap" in keys:
        rename["land_capacity_cap"] = "land_capacity_cap_mw"
    # Land availability metadata: be explicit that this is land onshore share.
    if "percent_onshore" in keys:
        rename["percent_onshore"] = "land_onshore_pct"
    if "percent_onshore_pct" in keys:
        rename["percent_onshore_pct"] = "land_onshore_pct"

    if not rename:
        return dict(location_results)

    updated: Dict[str, Any] = {}
    for key, value in location_results.items():
        updated[rename.get(key, key)] = value
    return updated


class Data_store:
    """Class designed to store data from each instance of the optimisation driver case"""

    def __init__(self):
        """Creates an empty dictionary in which data will be stored"""
        self.collated_results = {}

    def add_location(self, lat, lon, country, location_results, name=None):
        """Adds a location to the collated results"""
        dct = {'latitude': lat, 'longitude': lon, 'country': country}
        for key, value in _apply_unit_suffixes(location_results).items():
            dct[key] = value
        if name is None:
            self.collated_results['{a}_{b}'.format(a=lat, b=lon)] = dct
        else:
            self.collated_results[name] = dct

                