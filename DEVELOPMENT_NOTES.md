# Development Notes (Developers and AI Agents)

## Scope
This file is the technical reference for architecture, modeling conventions, cost-split logic, and implementation status.

`README.md` is user-facing. Keep deep technical details here.

## Repository Architecture
- `model/main.py`: single-site orchestration and PyPSA solve entrypoint
- `model/run_global.py`: multi-location orchestration, spatial cost inputs, output assembly
- `model/auxiliary.py`: constraint hooks, weather IO helpers, reporting transforms
- `model/location_tools.py`: weather/location geospatial utilities
- `model/land_processing.py`: land/bathymetry-based max-capacity preprocessing
- `model/data_store.py`: results accumulator for multi-location runs
- `model/plot_global_heatmap.py`: choropleth visualisation of global sweep outputs
- `basic_ammonia_plant/*.csv`: canonical PyPSA topology and techno-economic tables used at runtime
- `inputs/tech_config_ammonia_plant_2030_*.yaml`: scenario assumptions compiled into plant CSVs by notebook
- `notebooks/00_tech_config.ipynb`: YAML -> CSV compiler for costs/efficiencies/link recipes
- `notebooks/01_max_capacities.ipynb`: land/bathymetry capacity preprocessing
- `notebooks/02_spatial_cost_inputs.ipynb`: generate per-location cost overrides CSV (offshore multipliers, etc.)
- `notebooks/03_single_site_run.ipynb`: single-location solve and timeseries visualisation
- `notebooks/04_global_run.ipynb`: global sweep + quadrant results combiner
- `notebooks/05_run_analysis.ipynb`: post-run comparative analysis

## Canonical Modeling Conventions

### Basis and units
- Tech assumptions are expressed on an **HHV output basis**.
- For links in YAML, `overnight_cost_per_mw` is quoted on **MW_out (bus1)**.
- In PyPSA, link `p_nom` is **MW_in (bus0)**.
- `notebooks/00_tech_config.ipynb` performs output-basis to PyPSA-input-basis CAPEX conversion.

### Naming
- Component names must stay in snake_case and align across YAML + CSV + reporting.
- Keep new model documentation in comments/Markdown, not in non-standard CSV columns.

### Runtime config rule
- Runtime tech config application is deprecated.
- The only supported path is: edit YAML -> run notebook -> solve using updated CSV bundle.

## Core Workflow
1. Edit scenario YAML (`inputs/tech_config_ammonia_plant_2030_*.yaml`).
2. Run `notebooks/00_tech_config.ipynb` to compile assumptions into `basic_ammonia_plant/*.csv`.
3. Run single-site or global workflows.
4. Review results and split diagnostics.

## Process Coupling and Constraints
- `main.main()` solves with `extra_functionality=auxiliary.linopy_constraints`.
- Active guardrails include:
  - Battery charge/discharge capacity coupling.
  - Hydrogen storage discharge power linkage to store content/cycling assumptions.
  - Link ramp-rate constraints for ammonia synthesis when limits are provided.
- `ammonia_synthesis` is a multi-port link with fixed stoichiometric/energy coupling through `efficiency` and `efficiency2`.

## Spatial Cost Implementation Summary

### Implemented features
- Split-capex representation in YAML:
  - `tech_cost_per_mw/mwh`
  - `build_cost_per_mw/mwh`
  - `overnight_cost_per_mw/mwh` retained for validation
- Spatial cost inputs via `inputs/spatial_cost_inputs.csv`:
  - `interest_rate`
  - `build_cost_multiplier`
  - `land_cost_usd_per_km2_year`
  - `water_cost_usd_per_m3`
- `model/run_global.py` recomputes annualized costs under spatial cost overrides.
- Water and land cost effects are propagated into results and LCOA reporting.

### Cost split reporting behavior
`run_global.py` computes headline percentages (`build_cost_pct`, `tech_cost_pct`, `om_cost_pct`, `interest_pct`) from split inputs and solved capacities. This requires split arithmetic consistency in YAML.

## DEA Datasheet Reference and Split Derivation Guide

### Local reference workbooks
- `data/dea_reference/data_sheets_for_renewable_fuels (2).xlsx`
- `data/dea_reference/energy_transport_datasheet - 07_0 (1).xlsx`
- `data/dea_reference/Technology_datasheet_for_energy_storage – 0010 (2).xlsx`
- `data/dea_reference/technology_data_for_el_and_dh - 0017_1 (1).xlsx`

Preferred parsing tab: `alldata_flat` with columns:
`ws`, `Technology`, `cat`, `par`, `unit`, `priceyear`, `note`, `ref`, `est`, `year`, `val`

Use `year=2030` and `est=ctrl` for baseline config derivations.

### Split derivation patterns
1. **Percentage split from top-line CAPEX**
- Example: AEC and Hydrogen-to-Ammonia sheets provide explicit equipment/install percentages.
- Rule: `tech = total * equipment_share`, `build = total * installation_share`.

2. **Component regrouping**
- Example: PV/wind sheets provide CAPEX component lines.
- Rule: define deterministic buckets:
  - `tech`: equipment-centric lines
  - `build`: installation/civil/development/grid/soft-cost lines

3. **Mixed MW and MWh decomposition**
- Example: lithium-ion battery has power (MW), energy (MWh), and other project costs (MWh).
- Rule used in this repo: 1-hour reference system, with `other project costs` split 50/50 across MW and MWh assets.

4. **Materials/install percentage on transport curves**
- Example: H2/NH3 transport sheets provide per-capacity-band cost curves with materials/install percentages.
- Rule: select intended capacity band first, then apply percentages.

### Mandatory checks before committing config edits
- `overnight_cost == tech_cost + build_cost` for each technology.
- Unit/currency consistency (`EUR/kW`, `MEUR/MW`, `MEUR/MWh`, etc.).
- Link basis consistency (YAML output basis vs PyPSA input basis).
- Explicit comments for assumptions, proxies, and conversions.

## Wind Technology Simplification (2026-03)

The three wind generator variants (`onshore_wind`, `offshore_wind_fixed`, `offshore_wind_floating`)
have been consolidated into a single `wind` generator. Rationale: only one wind NetCDF profile
exists (`WindPowers*.nc`); the three generators used identical capacity factors.

Offshore cost differentiation should be applied via `build_cost_multiplier` in the finance
overrides CSV rather than through separate generator technologies.

## Offshore Build Cost Overrides

The `build_cost_multiplier` column in the spatial cost inputs CSV is wired into
`run_global.py`. Notebook `02_spatial_cost_inputs.ipynb` generates per-location,
per-tech overrides for offshore cells (`onshore_land_pct == 0`).

### Depth-based multiplier (implemented 2026-04)

Offshore `build_cost_multiplier` is a piecewise-linear function of bathymetry depth
with separate curves for **wind** and **plant equipment** techs. The `elevation_m`
column from `max_capacities.csv` (derived from `model_bathymetry.nc`) is used
as input. Convention: positive = above sea level, negative = below sea level (ocean depth).

Breakpoints and multipliers:

| Depth (m) | Wind mult | Plant mult | Technology regime |
|-----------|-----------|------------|-------------------|
| 0         | 1.3       | 1.1        | Shallow / fixed-bottom |
| 60        | 1.8       | 1.3        | Fixed-bottom limit (DNV, IEA Wind Task 26) |
| 300       | 2.5       | 1.8        | Semi-sub / spar floating proven range |
| 1500      | 3.5       | 2.5        | Deep floating → unmoored / vessel-based |

Between breakpoints, values are linearly interpolated (`np.interp`). Beyond 1500 m
(~80% of ocean cells), multipliers are clamped at the last breakpoint value.

Wind techs: `{wind}`. All other techs use the plant curve.

### Composable multiplier design

The build cost multiplier is a product of three independent factors:
```
build_cost_multiplier = depth_mult × remoteness_mult × labour_mult
```

Each factor is ≥ 1.0. Audit columns (`depth_mult`, `remoteness_mult`, `labour_mult`) are
included in `spatial_cost_inputs.csv` for transparency; `run_global.py` reads only
`build_cost_multiplier`.

### Offshore remoteness multiplier (implemented 2026-04)

Linear function of great-circle distance to nearest major port (throughput ≥ 1 Mt/yr).
Applied to offshore cells only (`onshore_land_pct == 0`).

| Distance (km) | Multiplier |
|---------------|------------|
| 0             | 1.0        |
| 2000          | 1.5        |
| beyond 2000   | extrapolates linearly (no cap) |

Rate: +25% per 1000 km. Pragmatic proxy — inflating `build_cost` also inflates O&M via
`fixed_om_fraction`, so this captures both CAPEX logistics and recurring O&M distance penalty.

Port data: `data/port_locations.csv` (1,369 ports from World Port Index, filtered by throughput).
Distance computation uses `scipy.spatial.cKDTree` on a unit-sphere XYZ embedding.

### Onshore remoteness multiplier (implemented 2026-04)

Linear function of travel time to nearest city (population ≥ 50k).
Applied to onshore/coastal cells only.

| Travel time (min) | Multiplier |
|-------------------|------------|
| 0                 | 1.0        |
| 2000              | 1.5        |
| beyond 2000       | extrapolates linearly (no cap) |

At ~60 km/h average, 2000 min ≈ 2000 km — intentionally comparable to offshore scaling.
Same pragmatic-proxy philosophy as offshore.

Source: MAP/Oxford Accessibility to Cities 2015 v1.0 (Google Earth Engine).
Dataset: `Oxford/MAP/accessibility_to_cities_2015_v1_0`, band `accessibility`, 1 km resolution.
Pre-fetched by `scripts/fetch_travel_time.py` → `data/travel_time_by_cell.csv`.

**Fetch script** (`scripts/fetch_travel_time.py`):
- Requires `earthengine-api` and GEE authentication (`earthengine authenticate`)
- GEE project: `weather-461309`
- Uses `reduceRegions` with `ee.Reducer.mean()` at 1 km scale, batches of 500 features
- Output: 19,644 rows; 17,085 with valid data, 2,559 null (small islands, ice)
- Runtime: ~2 minutes
- Re-runnable if the grid changes — just re-run the script

Travel-time statistics (onshore cells with data):
- Median: 399 min (6.7 h), Mean: 1,353 min (22.5 h), Max: 22,578 min (376 h)
- P75: 1,465 min (~24 h), P90: 4,308 min (~72 h)

### NaN travel-time fallback: no-road premium + elevation penalty (implemented 2026-04)

~2,559 onshore cells have **no travel-time data** from the MAP/Oxford dataset — primarily
Antarctica, interior Greenland, remote Arctic islands, and tiny atolls. Without a fallback
these cells would get `remoteness_mult = 1.0`, making Antarctica appear as cheap to build
on as Western Europe — clearly wrong.

The fallback uses a two-step approach:

**Step 1 — No-road premium.** Start from the offshore distance-to-port remoteness
(which is always available via the cKDTree), then amplify the excess above 1.0 by a
`NO_ROAD_FACTOR` to penalize the absence of road infrastructure:

```python
ONSHORE_NO_ROAD_FACTOR = 4.0

sea_leg = offshore_remoteness_multiplier(distance_to_port_km)
no_road_mult = 1.0 + NO_ROAD_FACTOR × (sea_leg − 1.0)
```

Rationale: the port distance captures the sea leg, but overland transport through
roadless terrain (ice sheets, tundra, dense jungle) is conservatively 4× as expensive
as the sea-distance premium alone.

**Step 2 — Elevation penalty.** Linear penalty for high-altitude construction above
1,000 m:

```python
ELEVATION_PENALTY_THRESHOLD_M = 1000
ELEVATION_PENALTY_PER_1000M   = 0.2   # +20% per 1000 m above threshold

elev_factor = 1.0 + 0.2 × max(0, elevation_m − 1000) / 1000
```

**Combined:**
```python
fallback_remoteness_mult = no_road_mult × elev_factor
```

Representative values (NO_ROAD_FACTOR = 4.0):
| Location | Port dist (km) | Elevation (m) | remoteness_mult |
|----------|---------------|---------------|------------------|
| Antarctica (-71, 109) | 5,360 | 2,146 | 6.382 |
| Greenland (75, -43) | 5,644 | 2,320 | 6.644 |
| Antarctica max | — | — | 8.246 |

Overall fallback range: min 1.057, max 8.246 (across all 2,559 NaN cells).

### Labour multiplier (placeholder)

`labour_mult = 1.0` everywhere. Awaiting per-country labour cost index from Luke.

This keeps the model architecture clean: cost differentiation lives in data, not in generator topology.

### MODIS land-cover availability fix (implemented 2026-07)

**Bugs fixed:**
1. **Solar had no per-class filtering.** `solar_area_km2 = onshore_area_km2` treated all
   land (forests, wetlands, urban, ice) as 100% suitable for solar panels.
2. **Wind availability factors were dead data.** `MODIS_CLASS_WIND_AVAILABILITY` was computed
   into `wind_availability` but never used — `wind_area_km2 = onshore + offshore` ignored
   the per-class weights entirely.

**Fix:** Both `solar_area_km2` and `wind_area_km2` now use MODIS-class-weighted availability:
```python
solar_area_km2 = area × solar_availability    # land-only, weighted by MODIS_CLASS_SOLAR_AVAILABILITY
wind_area_km2  = area × wind_availability     # includes offshore (water class=1.0 in wind map)
```

**MODIS_CLASS_SOLAR_AVAILABILITY** (Salmon & Bañares-Alcántara 2022 / Verschuur et al. 2024):
- 0 (Water): 0.0, 1–5 (Forests): 0.0, 6 (Closed shrubs): 0.5, 7 (Open shrubs): 0.5
- 8 (Woody savannas): 0.2, 9 (Savannas): 0.2, 10 (Grasslands): 0.2
- 11 (Wetlands): 0.0, 12 (Croplands): 0.0, 13 (Urban): 0.03
- 14 (Crop/veg mosaic): 0.0, 15 (Snow/ice): 0.0, 16 (Barren): 1.0

**MODIS_CLASS_WIND_AVAILABILITY** (same references):
- 0 (Water): 1.0 (offshore wind), 1–5 (Forests): 0.0, 6 (Closed shrubs): 0.5, 7 (Open shrubs): 0.5
- 8 (Woody savannas): 0.2, 9 (Savannas): 0.2, 10 (Grasslands): 0.2
- 11 (Wetlands): 0.0, 12 (Croplands): 0.05, 13 (Urban): 0.0
- 14 (Crop/veg mosaic): 0.05, 15 (Snow/ice): 0.0, 16 (Barren): 1.0

**Impact:** Solar capacities substantially reduced (forests/wetlands/urban no longer count).
Wind capacities now properly reflect suitability weighting. A new global run is needed to
see the LCOA impact of tighter capacity limits.

## Longitude Segmentation for Parallel SLURM Runs (2026-03)

`run_global()` accepts `lon_min`/`lon_max` parameters to filter locations by longitude.
The ARC submit wrapper supports `--quadrants` to submit 4 parallel jobs:
- `west2`: [-180, -90)
- `west1`: [-90, 0)
- `east1`: [0, 90)
- `east2`: [90, 180)

Each quadrant runs independently on a 48-CPU `medium` node (~20h each vs ~80h serial).
Results are merged using the combiner cell in notebook 03 or `pd.concat` on the CSVs.

## Config Completion Status (as of 2026-03-06)

### Completed
1. DEA split arithmetic reconciled (`overnight = tech + build`) across all configured technologies.
2. Lithium-ion allocation rule standardized (1-hour, 50/50 split of "other project costs" across MW/MWh).
3. Datasheet-derived split updates applied for:
- solar
- solar tracking
- wind (consolidated from onshore/offshore variants)
- hydrogen compression
4. Updated DEA split ratios mirrored into QLD structured config while keeping QLD overnight costs fixed.
5. Wind simplification: 3 generators → 1 `wind` generator across all model files, YAML configs, and notebooks.
6. Dead code removed: `model/tech_config.py`, `model/storage_cost_comparison.py`.

### Remaining data gaps
1. `hydrogen_from_storage`: placeholder CAPEX/split (no direct valve/regulator line identified).
2. `hydrogen_fuel_cell`: no direct imported DEA line identified; still proxy-converted.
3. `ammonia` storage: still proxy-based pending direct DEA source.
4. QLD-specific EPC/build breakdown evidence is still needed for fully local split localization.
5. ~~Offshore build cost multipliers not yet generated~~ → Implemented depth-based multipliers (2026-04).
6. ~~Remoteness multiplier not yet implemented~~ → Implemented offshore (distance-to-port) + onshore (travel-time-to-city) + NaN fallback (no-road premium + elevation penalty) (2026-04).
7. Labour cost multiplier (country-level) not yet implemented — awaiting Luke's data.
8. Per-country interest rates not yet implemented — awaiting Luke's data.

## ARC Cluster Operations

### Environment policy
- Local development: `.venv`.
- ARC production runs: conda environment (`/data/<group>/<user>/envs/green-lory-env`).
- ARC working directory policy: use `$DATA` (`/data/engs-df-green-ammonia/engs2523`) for repo/env/logs/results; do not run from ARC home (`~/`).

### ARC script layout
- `arc/arc_initial_setup.sh`: one-time ARC bootstrap (clone/update + optional env build submission).
- `arc/build-green-lory-env`: SLURM env build job.
- `arc/load_green_lory_env.sh`: module + conda activation helper for interactive shell use.
- `arc/arc_check_run_inputs.sh`: preflight check for required inputs before run submission.
- `arc/jobs/01_run_global.sh`: full global run SLURM job.
- `arc/submit_global_run.sh`: preflight + submit wrapper.

### Recommended ARC run sequence
1. `bash arc/arc_initial_setup.sh`
2. `sbatch arc/build-green-lory-env` (if env not already built)
3. `source arc/load_green_lory_env.sh`
4. `bash arc/arc_check_run_inputs.sh`
5. `bash arc/submit_global_run.sh <run-label> --quadrants` (4 parallel longitude quadrant jobs)

Single-job alternative (slower): `bash arc/submit_global_run.sh <run-label>`

### SSH command style for interactive agents (password-entry compatible)
Prefer one remote command per SSH invocation when driving from local automation/agent sessions:

```bash
ssh engs2523@arc-login.arc.ox.ac.uk 'cd /data/engs-df-green-ammonia/engs2523/green-lory && bash arc/submit_global_run.sh full-global-2030'
```

Use the same style for monitoring:

```bash
ssh engs2523@arc-login.arc.ox.ac.uk 'squeue -u engs2523'
ssh engs2523@arc-login.arc.ox.ac.uk 'cd /data/engs-df-green-ammonia/engs2523/green-lory && ls -1t logs/arc-full-global-2030-*.log | head -n 1'
```

This keeps each command explicit and works well when password must be entered per SSH command.

### ARC run controls (env vars)
`arc/jobs/01_run_global.sh` supports:
- `ARC_TECH_YAML`
- `ARC_INTEREST_CSV`
- `ARC_LAND_CSV`
- `ARC_LOCATIONS_CSV`
- `ARC_MAX_SNAPSHOTS`
- `ARC_LIMIT`
- `ARC_OUTPUT_CSV`
- `ARC_QUIET`
- `ARC_THREADS_PER_WORKER`
- `ARC_LON_MIN`, `ARC_LON_MAX` (longitude segmentation bounds)
- `ARC_ANACONDA_MODULE`
- `ARC_ENV_PREFIX`
- `ARC_GROUP`, `ARC_WORK_BASE`, `ARC_REPO_DIR`

Default production behavior is full sweep (`run_global`) with output written under `results/<run-label>/`.

## ARC Sync Policy (rsync)

Do **not** use `git pull` on ARC to keep code in sync — the ARC repo clone may
have uncommitted data files, results, or environment artefacts that create merge
conflicts.  Instead, use `rsync` from the local machine:

```bash
# Push code + inputs to ARC (excludes data, results, .venv, __pycache__):
# IMPORTANT: use --exclude '*.nc' (recursive) NOT --exclude 'data/*.nc'
# because data/weather_data/archive/ contains a 290 GB global_cutout_2019.nc
rsync -avz --delete \
  --exclude '.venv/' \
  --exclude '__pycache__/' \
  --exclude '.git/' \
  --exclude '*.nc' \
  --exclude '*.hdf' \
  --exclude '*.geojson' \
  --exclude 'results/' \
  --exclude 'logs/' \
  --exclude 'notebooks/' \
  ./ engs2523@arc-login.arc.ox.ac.uk:/data/engs-df-green-ammonia/engs2523/green-lory/
```

For large data files (max_capacities CSV, weather NetCDFs), use explicit `scp`:

```bash
scp data/max_capacities.csv \
  engs2523@arc-login.arc.ox.ac.uk:/data/engs-df-green-ammonia/engs2523/green-lory/data/
```

Pull results back after ARC jobs complete:

```bash
rsync -avz \
  engs2523@arc-login.arc.ox.ac.uk:/data/engs-df-green-ammonia/engs2523/green-lory/results/ \
  ./results/
```

## Documentation Governance
- Keep only two canonical docs at repo root:
  - `README.md` for users
  - `DEVELOPMENT_NOTES.md` for developers/agents
- Move implementation history and technical change logs into this file instead of creating additional root Markdown documents.

## Fixed Hourly Load vs Annual Demand Target (paper insight, 2026-04)

The model imposes a constant hourly `p_set` on the ammonia bus (7134.7 MW_th,
~10 Mt/yr at HHV=6.25 MWh/t). This forces the optimizer to size H₂ and NH₃
storage to bridge intermittency — the ammonia store acts as the seasonal buffer.

### Why a constant load is a defensible formulation
1. **Ammonia storage is already modelled.** The extendable ammonia store
   (~71 EUR/MWh/yr annualized, ~521 EUR/MWh overnight) decouples HB output
   from the fixed offtake. The optimizer uses it — median bridging capacity
   is ~20 days, P95 ~50 days across viable cells.
2. **Cost impact is small.** Ammonia storage accounts for ~24 EUR/t median,
   or ~3.2% of LCOA, in viable cells. This is the overhead of guaranteeing
   constant delivery vs a hypothetical "produce whenever you want" target.
3. **Bimodal grid distribution.** Grid usage is sharply bimodal: 77.6% of
   cells draw <100 MWh/yr from the backstop (effectively zero), while 22%
   draw >1M MWh (resource-dead zones). Only 0.3% sit in between — there is
   almost no marginal population where flexible demand would flip viability.
4. **Chemical process realism.** HB reactors prefer baseload operation
   (30% turndown, 40%/hr ramp). A fully flexible annual target would
   underestimate real operational constraints.

### Caveat for future work
Real offtake contracts can be seasonal (e.g. shipping windows, agricultural
demand cycles). A future extension could replace the fixed `p_set` with a
periodic profile or an annual energy constraint with a minimum utilization
factor, which would reduce ammonia storage requirements and slightly lower
LCOA in locations with strongly seasonal VRE. This would also require
modelling port/logistics scheduling.

## Repo Cleanup and Rebaseline Plan (2026-05-05)

Goal: make the repository usable by a human operator after the land-availability
rework, cleanly separate expensive ARC-only geospatial preprocessing from cheap
derived land-cap variants, clean up stale results/helpers, and then re-sync ARC
from a rationalized local layout.

Important current findings:
- The heavy land-processing stage should now build reusable baseline tables with
  no land competition baked in; land competition becomes a cheap CSV rescaling
  step performed after the protected-area/slope overlay is finished.
- The first required baseline matrix is:
  - `max_capacities_baseline_slope15.csv`
  - `max_capacities_baseline_allslopes.csv`
- The first required derived competition variants are:
  - `max_capacities_paper_2pct_slope15.csv`
  - `max_capacities_high_50pct_slope15.csv`
- The `allslopes` baseline is intended for future runs where steep terrain is
  represented through spatially varying build cost rather than hard exclusion.
- The canonical local data layout is now:
  - top-level `data/model_bathymetry.nc`
  - top-level `data/GEBCO_2025_sub_ice.nc`
  - top-level `data/WDPA_Feb2026_Public_shp_*/`
  - `data/weather_data/` only for weather stacks
- The protected-area checkpoint CSV is a disposable generated artifact and
  should not be treated as a canonical input.
- `results/` currently mixes canonical merged scenario folders
  (`dea_2050_flat/`, `way_2050_spatial/`, etc.) with older quadrant folders
  (`dea-2050-flat-east1/`, `way-2050-spatial-west2/`, etc.).
- Only a very small set of files under `data/`, `results/`, and `scripts/` are
  tracked; many local helper scripts already behave as local-only utilities.

### Ordered execution plan

1. **Freeze the target repo shape and user workflow**
   - Decide the canonical human workflow to support and document:
     `00_tech_config` -> `01_max_capacities` -> `02_spatial_cost_inputs` ->
     `04_global_run` -> `05_run_analysis`.
   - Decide which local helpers remain intentionally untracked for agent/copilot
     convenience, and which ones should become supported user-facing scripts.
   - Review gate: approve the repo structure and supported workflow before path
     migration or result deletion.

2. **Expose land-availability scenarios as user-facing configuration**
   - Make `land_competition_fraction` explicit in the notebook/CLI entrypoint
     rather than hidden in `model/land_processing.py` defaults.
   - Treat land competition and slope handling as separate scenario axes:
     - slope baselines: `baseline_slope15`, `baseline_allslopes`
     - competition variants: `baseline`, `paper_2pct`, `high_50pct`
   - Heavy ARC builds should only generate the baseline no-competition files.
   - Derived competition files should be produced from a baseline CSV via cheap
     rescaling, not by rerunning the full geospatial overlay.
   - Ensure the output naming makes both axes legible in downstream runs and ARC
     submissions.
   - Status 2026-05-05: CLI defaults now target the no-competition baseline;
     `model/land_processing.py --base-csv ...` can derive competition variants
     from an existing baseline file, and the ARC submission layer should submit
     `baseline_slope15`, `baseline_allslopes`, `paper_2pct_slope15`, and
     `high_50pct_slope15` as the initial supported matrix.
   - Review gate: confirm the first matrix and naming before broader scenario
     expansion.

3. **Rationalize the `data/` layout before any ARC sync**
   - Move descriptive top-level data files so the important static inputs sit in
     `data/` root where practical.
   - Target shape:
     - keep `data/dea_reference/`
     - keep `data/weather_data/`, but move bathymetry out of it
     - keep MODIS, WDPA/protected-area, GEBCO slope raster, bathymetry,
       ports/travel-time/country shapes at top level
  - Keep the deprecated external-data subtree absent from the canonical layout.
   - Treat protected-area checkpoint CSVs as disposable generated artefacts, not
     canonical inputs.
   - Status 2026-05-05: local files migrated to the top-level layout; keep ARC in
     sync with that shape before the next land-build submission.
   - Review gate: approve final data tree before any file moves.

4. **Update all code, notebooks, ARC scripts, and docs to the new paths**
   - Update `model/land_processing.py`, notebooks, ARC job scripts, staging
     helpers, and user docs to a single authoritative path scheme.
   - Eliminate current path inconsistencies such as the stale bathymetry path in
     `notebooks/01_max_capacities.ipynb`.
   - Confirm generated files are still written to stable user-visible locations.
   - Review gate: run focused path validation locally before result cleanup.

5. **Separate supported scripts from local-only helpers**
   - Keep or promote only scripts that a human user should run directly.
   - Leave agent/copilot convenience helpers untracked and clearly labeled as
     local operational tooling rather than supported repo interfaces.
   - Tighten `.gitignore` and docs so this distinction is obvious.
   - Review gate: approve the supported script list before deleting or moving
     helpers.

6. **Refresh documentation to match the cleaned workflow**
   - Update `README.md`, `arc/README.md`, and notebook config cells/comments so
     they describe the new land-availability flow, scenario knobs, data layout,
     and supported run/review sequence.
   - Document which generated files are canonical and which are temporary.
   - Review gate: doc pass should happen before new reruns are launched.

7. **Implement spatially varying water cost**
   - Replace the current flat `$2/m^3` placeholder with a composed cost model.
   - Proposed structure:
     - baseline desalinated water cost
     - access factor based on travel time to nearest city or coast (whichever is
       cheaper/closer in the chosen formulation)
     - pipeline transport cost based on a documented water-pipeline cost proxy
   - Need one short evidence step first: identify and record a defensible Danish
     district-heating / large water-pipeline cost source and convert it into a
     model-ready cost basis.
   - Review gate: approve formula and source assumptions before coding.

8. **Regenerate local inputs after the cleanup lands**
   - Rebuild the baseline max-capacity tables on ARC with the cleaned path
     scheme:
     - `baseline_slope15`
     - `baseline_allslopes`
   - Derive the first competition variants from `baseline_slope15` without
     rerunning the full overlay:
     - `paper_2pct_slope15`
     - `high_50pct_slope15`
   - Regenerate spatial cost inputs using the new water-cost model.
   - Review gate: inspect these artefacts before new global reruns use them.

9. **Prune stale results and standardize canonical outputs**
   - Keep only user-meaningful canonical merged scenario folders at top level in
     `results/`.
   - Remove or archive older quadrant shards and stale scenario folders once the
     replacement outputs are confirmed.
   - Ensure naming makes scenario dimensions explicit: year, finance mode, land
     cap case, and any major methodology variant.
   - Review gate: agree on retention policy before deletion.

10. **Re-run DEA scenarios under the cleaned setup**
   - Produce fresh DEA 2030 flat/spatial and DEA 2050 flat/spatial runs using the
     cleaned repo, cleaned data paths, and current land-availability logic.
   - Validate local merge assumptions and canonical output locations before ARC
     download/analysis.
   - Review gate: inspect merged outputs in notebook 05 before scrapping older
     DEA results.

11. **Prepare ARC for low-transfer sync, then sync code and metadata**
   - Before syncing from local, rearrange ARC-side large data files into the same
     target locations so `rsync` mostly updates code and small metadata rather
     than re-copying bulky rasters/shapefiles.
   - Sync only after the local layout, docs, and run scripts are stable.
   - Preserve the explicit ARC Python path approach (`$ARC_ENV_PREFIX/bin/python`).
   - Review gate: dry-run `rsync` and inspect transfer volume before real sync.

12. **Submit replacement ARC runs and retire superseded outputs**
   - Submit the cleaned DEA and chosen land-cap scenarios from the synced repo.
   - Download and validate merged outputs locally.
   - Only then remove/archive the superseded older runs on both local and ARC
     sides.

### Implementation order for interactive review

For reviewable execution, work the plan in these chunks:
- Chunk A: workflow definition + supported script policy.
- Chunk B: land-cap parameter exposure.
- Chunk C: data-layout migration.
- Chunk D: doc refresh.
- Chunk E: spatial water-cost model.
- Chunk F: regenerate local artefacts.
- Chunk G: result pruning + DEA reruns.
- Chunk H: ARC realignment and sync.

Do not combine Chunk C, Chunk E, and Chunk G in one pass; each changes a
different reproducibility boundary and should be reviewed independently.

## TODO
~~Continue remoteness/adversity factor~~ → Done (2026-04): offshore + onshore remoteness + NaN fallback implemented
~~Fix MODIS land availability for solar and wind~~ → Done (2026-07): added per-class solar factors, wired wind factors into capacity calc
- Integrate per-country labour cost index when Luke provides data (`labour_mult`)
- Integrate per-country cost-of-capital / interest rates when Luke provides data
- Integrate per-country water costs when Luke provides data
- Integrate per-country land costs when Luke provides data
- Luke (as of last update): checking labour costs, water, and CoC; starting on land costs
