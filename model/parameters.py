"""Model parameters: inputs/parameters.yaml, the modelling assumptions other than technology costs.

Modules read their values when imported, under the constant names they use:

    from .parameters import parameter
    WIND_LAND_USE_KM2_PER_GW = parameter("land.wind_land_use_km2_per_gw")

parameter() returns the value at a dotted key (a list or mapping as plain Python) and raises
KeyError when the file lacks it. The keys read are recorded, so that tests/test_parameters.py can
check that the model reads every value in the file. Generated tables and run manifests record
the file's hash (file_record()).
"""
from __future__ import annotations

import copy
import hashlib
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
PARAMETERS_YAML = REPO_ROOT / "inputs" / "parameters.yaml"
_KEYS_READ: set[str] = set()


@lru_cache(maxsize=None)
def _load(path: Path) -> dict:
    data = yaml.safe_load(path.read_text())
    if not isinstance(data, dict):
        raise ValueError(f"{path} must hold a mapping of sections")
    return data


def parameter(key: str) -> Any:
    """The value at a dotted key of inputs/parameters.yaml, e.g. 'site_costs.slope_class_factors'."""
    node: Any = _load(PARAMETERS_YAML)
    for part in key.split("."):
        if not isinstance(node, dict) or part not in node:
            raise KeyError(f"'{key}' is not in {PARAMETERS_YAML}")
        node = node[part]
    _KEYS_READ.add(key)
    return copy.deepcopy(node)


MODEL_PRICE_YEAR = 2020


def eur2020_per(currency: str, year: int) -> float:
    """EUR2020 per unit of `currency` in prices of `year` (inputs/parameters.yaml, currency).

    The year's ECB rate first, then euro-area inflation to 2020. Dividing a EUR2020 result by
    this factor states it in `currency` of `year`.
    """
    code = currency.strip().upper()
    if code == "EUR":
        factor = 1.0
    elif code == "USD":
        rates = parameter("currency.usd_per_eur")
        if year not in rates:
            raise KeyError(f"no USD rate for {year} in {PARAMETERS_YAML} (currency.usd_per_eur)")
        factor = 1.0 / float(rates[year])
    else:
        raise ValueError(f"no exchange rate for {currency!r}: add it to currency in {PARAMETERS_YAML}")
    inflation = parameter("currency.eur_inflation")
    for later in range(year + 1, MODEL_PRICE_YEAR + 1):
        factor *= 1.0 + float(inflation[later])
    for later in range(MODEL_PRICE_YEAR + 1, year + 1):
        factor /= 1.0 + float(inflation[later])
    return factor


def keys_read() -> set[str]:
    """The dotted keys read so far in this process."""
    return set(_KEYS_READ)


def leaf_keys(node: dict | None = None, prefix: str = "") -> list[str]:
    """Every dotted key of the file whose value is not a mapping."""
    node = _load(PARAMETERS_YAML) if node is None else node
    keys = []
    for name, value in node.items():
        key = f"{prefix}{name}"
        keys.extend(leaf_keys(value, f"{key}.") if isinstance(value, dict) else [key])
    return keys


def file_record() -> dict:
    """The file's path (relative to the repository) and SHA-256, for provenance records."""
    return {
        "path": PARAMETERS_YAML.relative_to(REPO_ROOT).as_posix(),
        "sha256": hashlib.sha256(PARAMETERS_YAML.read_bytes()).hexdigest(),
    }
