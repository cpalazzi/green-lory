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
2. Run `notebooks/00_tech_config.ipynb` to compile costs into `basic_ammonia_plant/*.csv`.
3. Run `notebooks/01_max_capacities.ipynb` to build a named land-scenario CSV such as `data/max_capacities_paper_2pct.csv`.
4. Run `notebooks/02_spatial_cost_inputs.ipynb` to generate offshore/location cost overrides.
5. Run either:
   - single-site optimization (`notebooks/03_single_site_run.ipynb`), or
   - global run (`notebooks/04_global_run.ipynb`).

## Supported Entry Points
The human-supported workflow in this repo is intentionally narrow. These entrypoints should stay stable and documented:
- `notebooks/00_tech_config.ipynb` to `notebooks/05_run_analysis.ipynb`
- `arc/submit_global_run.sh` for general ARC submissions
- `arc/submit_constrained_reruns.sh` for the canonical DEA/Way rerun set
- `scripts/fetch_travel_time.py` to build `data/travel_time_by_cell.csv`
- `scripts/merge_global_results.py` to merge quadrant outputs into canonical scenario CSVs

Anything else under `scripts/` should be treated as local diagnostics or Copilot convenience code, not as a supported user interface. Those local helpers may exist in a working tree, but they are intentionally kept gitignored.

## Standard Workflow
1. **Set technology assumptions**
   - Edit one scenario file, usually:
     - `inputs/tech_config_ammonia_plant_2030_qld.yaml`, or
     - `inputs/tech_config_ammonia_plant_2030_dea.yaml`.

2. **Compile YAML assumptions into plant CSVs**
   - Use `notebooks/00_tech_config.ipynb`.
   - This converts overnight CAPEX to annualized `capital_cost`, updates link efficiencies, and writes to `basic_ammonia_plant/`.

3. **Generate max-capacity constraints**
   - Use `notebooks/01_max_capacities.ipynb`.
   - Output: named land-scenario files such as `data/max_capacities_paper_2pct.csv` and `data/max_capacities_high_50pct.csv`.
   - The default paper scenario also refreshes the convenience alias `data/max_capacities.csv` for older scripts that still expect it.

4. **Generate spatial cost inputs** (optional but recommended)
   - Use `notebooks/02_spatial_cost_inputs.ipynb`.
   - Output: `inputs/spatial_cost_inputs.csv`.
   - See [Spatial Cost Methodology](#spatial-cost-methodology) below for details.

5. **Run optimization**
   - Single site for design/debug.
   - Global sweep for heatmaps and comparative location economics.

6. **Review outputs**
   - Results are written to `results/` and notebook outputs.
   - LCOA columns are currency-labeled (for example `lcoa_usd_per_t` or `lcoa_eur_per_t`).

## Notebooks

| # | Name | Purpose |
|---|------|---------|
| 00 | `tech_config` | Compile YAML scenario into PyPSA CSV bundle |
| 01 | `max_capacities` | Land/bathymetry preprocessing for capacity limits |
| 02 | `spatial_cost_inputs` | Generate per-location cost overrides (depth, remoteness, labour multipliers) |
| 03 | `single_site_run` | Single-location solve with timeseries plots |
| 04 | `global_run` | Batch sweep + ARC result combiner |
| 05 | `run_analysis` | Post-run comparative analysis and heatmaps |

## Spatial Cost Methodology

Notebook 02 generates per-location, per-technology cost overrides in `inputs/spatial_cost_inputs.csv`. The build cost multiplier adjusts only the **build/installation** portion of CAPEX — equipment costs are unchanged.

### Composite multiplier

```
build_cost_multiplier = depth_mult × remoteness_mult × labour_mult
```

Each factor ≥ 1.0. A cell with shallow water, near a port, and baseline labour costs gets 1.0 × 1.0 × 1.0 = 1.0 (no markup).

### Depth multiplier (offshore only)

Piecewise-linear function of ocean depth, with separate curves for wind (foundation cost) and plant equipment (logistics):

| Depth (m) | Wind | Plant | Regime |
|-----------|------|-------|--------|
| 0 | 1.3 | 1.1 | Shallow / fixed-bottom |
| 60 | 1.8 | 1.3 | Fixed-bottom limit |
| 300 | 2.5 | 1.8 | Floating (semi-sub / spar) |
| 1500 | 3.5 | 2.5 | Deep floating |

Returns 1.0 for onshore cells. Clamped at the last breakpoint beyond 1500 m.

### Remoteness multiplier

**Offshore cells**: linear function of great-circle distance to nearest major port (≥1 Mt/yr throughput). +25% per 1,000 km, extrapolating beyond 2,000 km.

**Onshore cells with travel-time data**: linear function of travel time to nearest city (population ≥50k), from the MAP/Oxford Accessibility to Cities 2015 dataset. Same +25% per 2,000 min scaling. Pre-fetched via `scripts/fetch_travel_time.py` → `data/travel_time_by_cell.csv`.

**Onshore cells without travel-time data** (~2,500 cells: Antarctica, Greenland interior, Arctic): fallback using distance-to-port with a 4× no-road premium, plus an elevation penalty above 1,000 m:

```
no_road_mult  = 1.0 + 4.0 × (port_remoteness − 1.0)
elev_factor   = 1.0 + 0.2 × max(0, elevation_m − 1000) / 1000
remoteness    = no_road_mult × elev_factor
```

### Labour multiplier

Placeholder at 1.0 everywhere. Will accept a per-country index when available.

### Other spatial inputs

| Input | Current default | Notes |
|-------|----------------|-------|
| Interest rate | 10% (uniform) | Per-tech from YAML; awaiting per-country data |
| Water cost | $2.00/m³ (uniform) | Added to LCOA post-solve |
| Land cost | $0/km²/yr | Added to LCOA post-solve |

### How the multiplier enters the model

In `run_global.py`, only the **build** portion of overnight cost is scaled:

```
effective_overnight = tech_cost + (build_cost × build_cost_multiplier)
capital_cost = effective_overnight × (annuity + fixed_O&M_fraction)
```

This means equipment costs are location-invariant; only installation/logistics/civil works scale with geography.

## Running from the Command Line

Each step in the workflow can also be run directly from the terminal. This is
useful for scripting, HPC submission, or quick tests without opening a notebook.

```bash
# Activate the environment
source .venv/bin/activate

# 1. Single-site run (default: Alice Springs, DEA 2030)
python -m model.run_global \
  --tech-yaml inputs/tech_config_ammonia_plant_2030_dea.yaml \
  --interest-csv inputs/spatial_cost_inputs.csv \
  --land-csv data/max_capacities.csv \
  --output-csv results/single_site_test.csv \
  --locations-csv inputs/my_locations.csv \
  --quiet

# 2. Global run (all locations from max-capacities CSV)
python -m model.run_global \
  --tech-yaml inputs/tech_config_ammonia_plant_2030_dea.yaml \
  --interest-csv inputs/spatial_cost_inputs.csv \
  --land-csv data/max_capacities.csv \
  --output-csv results/global_run_results.csv \
  --quiet

# 3. Global run with longitude segment (for parallelism)
python -m model.run_global \
  --tech-yaml inputs/tech_config_ammonia_plant_2030_dea.yaml \
  --interest-csv inputs/spatial_cost_inputs.csv \
  --land-csv data/max_capacities.csv \
  --output-csv results/east1.csv \
  --lon-min 0 --lon-max 90 \
  --quiet

# 4. Quick smoke test (first 10 locations, 1 week of weather)
python -m model.run_global \
  --tech-yaml inputs/tech_config_ammonia_plant_2030_dea.yaml \
  --output-csv results/smoke_test.csv \
  --max-snapshots 168 --limit 10 \
  --quiet
```

### Supported helper scripts

```bash
# Fetch travel-time-to-city data for onshore max-capacity cells
python scripts/fetch_travel_time.py

# Merge 4 quadrant outputs into one canonical scenario CSV
python scripts/merge_global_results.py way-2050-spatial \
   --output results/way_2050_spatial/global_run_results_1h_2050.csv
```

### CLI options

| Flag | Description |
|------|-------------|
| `--tech-yaml PATH` | Scenario YAML file |
| `--interest-csv PATH` | Spatial cost inputs CSV (generated by notebook 02) |
| `--land-csv PATH` | Max-capacities CSV (generated by notebook 01) |
| `--locations-csv PATH` | CSV with `lat,lon` columns to restrict the sweep |
| `--output-csv PATH` | Where to write result rows |
| `--lon-min N` / `--lon-max N` | Longitude bounds for parallel segmentation |
| `--max-snapshots N` | Limit weather timeseries length (e.g. 168 = 1 week) |
| `--limit N` | Process only the first N locations |
| `--threads-per-worker N` | Solver thread count |
| `--quiet` | Suppress solver/PyPSA log output |

## Files You Usually Touch
- `inputs/tech_config_ammonia_plant_2030_*.yaml` (technology/cost assumptions)
- `inputs/spatial_cost_inputs.csv` (per-location cost overrides; generated by notebook 02)
- `notebooks/00_tech_config.ipynb` through `notebooks/04_global_run.ipynb` (execution)

## Data Inputs
- Weather and auxiliary geospatial data live in `data/`.
- Static geospatial inputs now live mostly at `data/` root: `model_bathymetry.nc`, `GEBCO_2025_sub_ice.nc`, `WDPA_Feb2026_Public_shp_*/`, `countries.geojson`, `port_locations.csv`, and `travel_time_by_cell.csv`.
- Only the weather stacks stay under `data/weather_data/`, and DEA reference workbooks stay under `data/dea_reference/`.
- Country tagging for global runs expects `data/countries.geojson`.
- Max-capacity preprocessed inputs are generated via `notebooks/01_max_capacities.ipynb`.
- Named land-cap scenarios are part of the intended workflow: `paper_2pct` for the constrained paper case and `high_50pct` for the high-availability sensitivity.

## ARC Cluster
- ARC helper scripts are in `arc/`.
- Start with `arc/README.md` for setup, conda environment build, preflight checks, and global job submission on ARC.
- The supported ARC entrypoints are `arc/submit_global_run.sh` and `arc/submit_constrained_reruns.sh`.
- Local runs use `.venv`; ARC runs use conda.
- Use `rsync` (not `git pull`) to sync code to ARC — see `DEVELOPMENT_NOTES.md` for details.

## Troubleshooting
- **Solver errors**: set `GREEN_LORY_SOLVER=highs` (or configure Gurobi correctly).
- **Missing weather profiles**: generator names must align with weather columns.
- **Geo stack errors**: rebuild environment if GDAL/GeoPandas dependencies break.
- **Slow global runs**: reduce spatial extent or increase temporal aggregation.

## Documentation Structure
This repo maintains two canonical docs:
- `README.md` (this file): user-facing setup and run guidance
- `scripts/README.md`: supported helper scripts only
- `DEVELOPMENT_NOTES.md`: developer and AI-agent implementation details, conventions, and open technical gaps
