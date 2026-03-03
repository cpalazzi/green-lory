# Green Lory – User Guide

Green Lory is a PyPSA-based optimization workflow for sizing green ammonia plants and mapping levelized cost of ammonia (LCOA) across locations.

## What You Need
- Python 3.11+
- A solver supported by Linopy/PyPSA (HiGHS recommended; Gurobi optional)
- Dependencies installed from `requirements.txt`

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Quick Start
1. Choose/edit a scenario YAML in `inputs/`.
2. Run `notebooks/00_tech_config.ipynb` to write updated costs/efficiencies into `basic_ammonia_plant/*.csv`.
3. Run either:
- single-site optimization (`notebooks/02_single_site_run.ipynb`), or
- global run (`notebooks/03_global_run.ipynb` / `notebooks/04_global_finance_overrides_run.ipynb`).

## Standard Workflow
1. **Set technology assumptions**
- Edit one scenario file, usually:
  - `inputs/tech_config_ammonia_plant_2030_qld.yaml`, or
  - `inputs/tech_config_ammonia_plant_2030_dea.yaml`.

2. **Compile YAML assumptions into plant CSVs**
- Use `notebooks/00_tech_config.ipynb`.
- This converts overnight CAPEX to annualized `capital_cost`, updates link efficiencies, and writes to `basic_ammonia_plant/`.

3. **Run optimization**
- Single site for design/debug.
- Global sweep for heatmaps and comparative location economics.

4. **Review outputs**
- Results are written to `results/` and notebook outputs.
- LCOA columns are currency-labeled (for example `lcoa_usd_per_t` or `lcoa_eur_per_t`).

## Files You Usually Touch
- `inputs/tech_config_ammonia_plant_2030_*.yaml` (technology/cost assumptions)
- `inputs/example_finance_overrides_spatial.csv` (optional location-level finance/cost overrides)
- `notebooks/00_tech_config.ipynb` (apply YAML assumptions to plant CSVs)
- `notebooks/02_single_site_run.ipynb`, `03_global_run.ipynb`, `04_global_finance_overrides_run.ipynb` (execution)

## Data Inputs
- Weather and auxiliary geospatial data live in `data/`.
- Country tagging for global runs expects `data/countries.geojson`.
- Max-capacity preprocessed inputs are generated via `notebooks/01_max_capacities.ipynb`.

## ARC Cluster
- ARC helper scripts are in `arc/`.
- Start with `arc/README.md` for setup, conda environment build, preflight checks, and global job submission on ARC.
- Local runs should still use `.venv`; ARC runs use conda.

## Troubleshooting
- **Solver errors**: set `GREEN_LORY_SOLVER=highs` (or configure Gurobi correctly).
- **Missing weather profiles**: generator names must align with weather columns.
- **Geo stack errors**: rebuild environment if GDAL/GeoPandas dependencies break.
- **Slow global runs**: reduce spatial extent or increase temporal aggregation.

## Documentation Structure
This repo now maintains two canonical docs:
- `README.md` (this file): user-facing setup and run guidance
- `DEVELOPMENT_NOTES.md`: developer and AI-agent implementation details, conventions, and open technical gaps
