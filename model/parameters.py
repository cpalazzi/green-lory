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
