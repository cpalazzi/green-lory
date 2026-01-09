"""Deprecated module.

The project no longer applies technology YAML configuration at runtime.

Use the notebook-driven input workflow in `notebooks/00_tech_config.ipynb` to:
- load `inputs/tech_config_ammonia_plant.yaml`
- convert overnight CAPEX + interest rates into annualised PyPSA `capital_cost`
- write the resulting component tables into `basic_ammonia_plant/*.csv`
"""


def load_tech_config(*_args, **_kwargs):  # pragma: no cover
    raise RuntimeError(
        "Runtime tech-config loading has been removed. Use notebooks/00_tech_config.ipynb instead."
    )


def apply_tech_config(*_args, **_kwargs):  # pragma: no cover
    raise RuntimeError(
        "Runtime tech-config application has been removed. Use notebooks/00_tech_config.ipynb instead."
    )
