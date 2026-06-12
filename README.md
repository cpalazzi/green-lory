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
3. Run `notebooks/01_max_capacities.ipynb` to build a named land-scenario CSV such as `data/max_capacities_paper_2pct_slope15.csv`.
4. Run `notebooks/02_spatial_cost_inputs.ipynb` to generate offshore/location cost overrides.
5. Run either:
   - single-site optimization (`notebooks/03_single_site_run.ipynb`), or
   - global run (`notebooks/04_global_run.ipynb`).

## Supported Entry Points
The human-supported workflow in this repo is intentionally narrow. These entrypoints should stay stable and documented:
- `notebooks/00_tech_config.ipynb` to `notebooks/05_run_analysis.ipynb`
- `arc/submit_global_run.sh` for general ARC submissions
- `arc/submit_constrained_reruns.sh` for the canonical DEA/Way rerun set
- `scripts/build_ameli_wacc_inputs.py` to build Ameli Fig. 2 WACC override CSVs
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
   - Output: named land-scenario files such as `data/max_capacities_paper_2pct_slope15.csv` and `data/max_capacities_high_50pct_slope15.csv`.

4. **Generate spatial cost inputs** (optional but recommended)
   - Use `notebooks/02_spatial_cost_inputs.ipynb`.
   - Output: `inputs/spatial_cost_inputs.csv`.
   - See [Spatial Cost Methodology](#spatial-cost-methodology) below for details.

5. **Build Ameli finance overrides** (optional)
   - Use `python scripts/build_ameli_wacc_inputs.py` after notebook 02 when you want the Ameli et al. Fig. 2 finance sensitivity.
   - The default reduced-WACC outputs use the short spatial-variant token `amelired`:
     - `inputs/amelired_interest_inputs_2050.csv`
     - `inputs/spatial_cost_inputs_amelired_2050.csv`
   - WACC-only Ameli runs should use `flat_amelired`: Ameli reduced WACC without spatial build, remoteness, water, or land-cost changes.
   - Combined Ameli runs should name the active spatial mechanisms before the finance token. The current combined file is `spatial_build_remote_water_amelired`: active spatial build multipliers, active remoteness, active spatial water costs, Ameli reduced WACC, and no spatial land-cost term.
   - `amelired_interest_inputs_2050.csv` is the WACC-only input for `flat_amelired` runs, including the closest Salmon/Verschuur replication check.
   - `spatial_cost_inputs_amelired_2050.csv` preserves the existing spatial build/remoteness/water columns from the spatial base file and replaces `interest_rate` with Ameli reduced-WACC values. Use it only for `spatial_build_remote_water_amelired` runs. If your target land grid is wider than that base spatial file, rerun notebook 02 on the matching land CSV before using the combined Ameli file.
   - For the WAY 2050 `paper_2pct_slope15` case, the matching spatial base can be generated as `inputs/spatial_cost_inputs_way_2050_paper_2pct_slope15.csv`, then used to rebuild the combined Amelired file on the same active-cell grid.

6. **Run optimization**
   - Single site for design/debug.
   - Global sweep for heatmaps and comparative location economics.

7. **Review outputs**
   - Results are written to `results/` and notebook outputs.
   - LCOA columns are currency-labeled (for example `lcoa_usd_per_t` or `lcoa_eur_per_t`).

## Run Label Vocabulary

Canonical merged result folders under `results/` should use labels that expose the scenario axes:

```text
<tech_source>_<year>_<cost_scope>[_<finance_case>][_<time_step>]_<land_case>
```

Examples:

| Run label | Meaning |
|---|---|
| `way_2050_flat_paper_2pct_slope15` | WAY 2050 technology costs, flat build/water/land costs, standard finance, paper land case |
| `way_2050_flat_amelired_4h_paper_2pct_slope15` | Primary Salmon/Verschuur replication check: WAY 2050 costs, Ameli reduced WACC, four-hour resolution, flat build/remoteness/water/land costs |
| `way_2050_spatial_build_remote_water_paper_2pct_slope15` | Same technology/land case, but with spatial build multipliers, remoteness, and water costs |
| `way_2050_spatial_build_remote_water_amelired_4h_paper_2pct_slope15` | Salmon/Verschuur replication case: spatial build/remoteness/water costs plus Ameli reduced-WACC values at four-hour resolution |
| `way_2050_spatial_build_remote_water_amelired_4h_high_50pct_slope15` | Four-hour temporal aggregation, explicit spatial build/remoteness/water plus Ameli WACC, high land-availability sensitivity |
| `dea_2050_spatial_build_remote_water_100pct_slope15` | DEA 2050, spatial build/remoteness/water costs, full land-availability case with slope >15 degree exclusion |

### Label axes

| Axis | Tokens | Meaning |
|---|---|---|
| Technology source | `dea`, `way`, `qld` | Which technology/cost YAML family generated the plant CSVs |
| Year | `2030`, `2050` | Technology-cost year |
| Cost scope | `flat`, `spatial_build`, `spatial_build_remote`, `spatial_build_remote_water`, `spatial_build_remote_water_land` | Which spatial cost mechanisms are active in the override CSV |
| Finance case | omitted, `amelired` | Omitted means the YAML/default finance assumptions. `amelired` means Ameli reduced-WACC values in the override CSV |
| Time step | omitted, `4h` | Omitted means 1-hour snapshots; `4h` means four-hour aggregation |
| Land case | `100pct_slope15`, `100pct_allslopes`, `paper_2pct_slope15`, `high_50pct_slope15` | Renewable land-capacity scenario |

For Salmon/Verschuur paper-replication runs, use the `4h` time-step token. The papers describe hourly reanalysis weather inputs, but the archived lcoa-opt global runner used a four-hour global optimization (`time_step = 4`, 2190 snapshots). The closest first-pass replication target is `way_2050_flat_amelired_4h_paper_2pct_slope15`, because it keeps build/remoteness/water/land flat and varies only WACC through `inputs/amelired_interest_inputs_2050.csv`. One-hour Amelired runs are higher-resolution Green Lory reruns and should be kept separate from the replication comparison.

`flat` means no spatial override CSV: build-cost multipliers, water costs, land costs, remoteness, and WACC all come from the tech config or model defaults. `spatial_*` labels must name the active override mechanisms after `spatial`: `build` for `build_cost_multiplier`, `remote` for remoteness components, `water` for `water_cost_usd_per_m3`, and `land` for `land_cost_usd_per_km2_year`. Finance is a separate optional case appended after the spatial mechanism list, so `spatial_build_remote_water_amelired` means spatial build/remoteness/water plus Ameli reduced-WACC values in the `interest_rate` column. If a finance-only sensitivity is intentionally run without the spatial cost stack, it is `flat_amelired`. Do not include `land` in the label unless the land-cost column is nonzero and intentionally active.

`100pct` is the full land-availability token. In current land inputs:

| Land case | Meaning |
|---|---|
| `100pct_slope15` | Full available land after protected-area and slope >15 degree exclusions; heavy ARC land build |
| `100pct_allslopes` | Full available land after protected-area exclusions, with no slope exclusion; heavy ARC land build |
| `paper_2pct_slope15` | Paper case derived from `100pct_slope15` by applying a 2% land-competition fraction |
| `high_50pct_slope15` | Sensitivity derived from `100pct_slope15` by applying a 50% land-competition fraction |

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

## Renewable Capacity Limit Outputs

The global-run outputs now include renewable-capacity-derived ammonia ceiling columns such as:

- `max_ammonia_capacity_t`
- `max_ammonia_capacity_mtpa`
- `max_onshore_ammonia_capacity_t`
- `max_gridless_onshore_ammonia_capacity_t`
- `capacity_limit_technology`
- `onshore_capacity_limit_technology`
- `renewable_capacity_scale_factor`
- `onshore_renewable_capacity_scale_factor`
- `wind_mw_per_t_nh3`
- `solar_mw_per_t_nh3`

These are easy to misread. The important point is that the model does **not** treat onshore wind and solar as mutually exclusive technologies when computing the max-tonnes outputs.

### What is enforced in the solve

For land-constrained runs, the model enforces separate renewable caps:

- `wind <= max_power_wind_mw`
- `solar + solar_tracking <= max_power_solar_mw`

So onshore wind and solar can coexist in the same cell. The only shared solar constraint is that fixed-tilt solar and solar-tracking draw from the same solar land budget. Offshore cells typically end up wind-only because the land inputs provide wind capacity there and little or no solar capacity.

The land preprocessing also writes a convenience total:

```text
max_capacity_mw = max_power_wind_mw + max_power_solar_mw
```

That combined total is mainly metadata and a backward-compatible fallback. In the current explicit-cap workflow, the max-tonnes calculation uses the per-technology caps directly.

### How `max_ammonia_capacity_t` is computed

`max_ammonia_capacity_t` is **not** a second optimization. It is an ex-post scaling calculation based on the solved plant mix at that location.

First the run stores the renewable intensity of the solved design:

```text
wind_mw_per_t_nh3  = wind_used_mw  / annual_ammonia_t
solar_mw_per_t_nh3 = solar_used_mw / annual_ammonia_t
```

Then it asks: if we scale this exact solved mix up proportionally, which renewable cap is hit first?

```text
wind scale limit  = wind_cap_mw  / wind_used_mw
solar scale limit = solar_cap_mw / solar_used_mw
renewable_capacity_scale_factor = min(wind scale limit, solar scale limit)
max_ammonia_capacity_t = annual_ammonia_t * renewable_capacity_scale_factor
```

So the max-tonnes estimate can absolutely reflect **both** wind and solar together. If the solved design uses both technologies, the scaled-up ceiling still includes both. The calculation is simply bounded by whichever cap binds first under proportional scaling.

### What `capacity_limit_technology` means

`capacity_limit_technology` is the **first binding renewable cap** in the scaling calculation above. It does **not** mean:

- that only one technology is allowed in the cell, or
- that the reported `max_ammonia_capacity_t` was computed from wind alone or solar alone.

If a cell's solved mix uses both wind and solar, but wind reaches its cap at a smaller scale factor than solar, then:

- `capacity_limit_technology = wind`
- the reported max-tonnes value still includes the scaled solar contribution that fits underneath that wind-limited scale factor.

In other words, "wind-limited" means "wind binds first," not "solar is excluded."

### Why there is also an onshore variant

`max_onshore_ammonia_capacity_t` uses the same proportional-scaling logic, but swaps in the explicitly onshore wind cap (`wind_onshore_area_km2 * wind_density_mw_per_km2`) instead of the broader wind cap when that distinction exists in the land input. This is useful when a cell mixes onshore and offshore wind resource accounting.

`max_gridless_onshore_ammonia_capacity_t` is then derived by multiplying the onshore ceiling by the gridless fraction of the solved ammonia output.

## Running from the Command Line

Each step in the workflow can also be run directly from the terminal. This is
useful for scripting, HPC submission, or quick tests without opening a notebook.

```bash
# Activate the environment
source .venv/bin/activate

# 1. Single-site run (default: Alice Springs, DEA 2030)
python -m model.run_global \
  --tech-yaml inputs/tech_config_ammonia_plant_2030_dea.yaml \
   --override-csv inputs/spatial_cost_inputs.csv \
   --land-csv data/max_capacities_paper_2pct_slope15.csv \
  --output-csv results/single_site_test.csv \
  --locations-csv inputs/my_locations.csv \
  --quiet

# 2. Global run (all locations from max-capacities CSV)
python -m model.run_global \
  --tech-yaml inputs/tech_config_ammonia_plant_2030_dea.yaml \
   --override-csv inputs/spatial_cost_inputs.csv \
   --land-csv data/max_capacities_paper_2pct_slope15.csv \
  --output-csv results/global_run_results.csv \
  --quiet

# 3. Global run with longitude segment (for parallelism)
python -m model.run_global \
  --tech-yaml inputs/tech_config_ammonia_plant_2030_dea.yaml \
   --override-csv inputs/spatial_cost_inputs.csv \
   --land-csv data/max_capacities_paper_2pct_slope15.csv \
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

# Build Ameli Fig. 2 WACC overrides (defaults to the amelired reduced-WACC case)
python scripts/build_ameli_wacc_inputs.py

# Merge 4 quadrant outputs into one canonical scenario CSV
python scripts/merge_global_results.py way-2050-spatial-build-remote-water-paper-2pct-slope15 \
   --output results/way_2050_spatial_build_remote_water_paper_2pct_slope15/global_run_results_1h_2050.csv
```

### CLI options

| Flag | Description |
|------|-------------|
| `--tech-yaml PATH` | Scenario YAML file |
| `--override-csv PATH` | Spatial override CSV (generated by notebook 02) |
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
- Named land-cap scenarios are part of the intended workflow: `100pct` for full land availability, `paper_2pct` for the constrained paper case, and `high_50pct` for the high-availability sensitivity.

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
