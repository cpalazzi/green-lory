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
`run_global.py`. Notebook `02_spatial_cost_inputs.ipynb` generates a base overrides
CSV that applies `build_cost_multiplier = 2.0` for all techs at pure-ocean locations
(`onshore_land_pct == 0`).

<!-- TODO: vary offshore costs by distance-to-coast and fixed/floating wind depth bands -->

Planned refinements:
1. Read bathymetry depth from `model_bathymetry.nc` and distance-to-coast.
2. Apply graduated multipliers per location, e.g.:
   - Shallow fixed-bottom (< 50m depth): `build_cost_multiplier` ~1.5
   - Floating (50–1000m): `build_cost_multiplier` ~2.0
   - Additional remoteness factor based on distance-to-shore
3. Differentiate wind cost from other techs (wind turbines have larger offshore premium).

This keeps the model architecture clean: cost differentiation lives in data, not in generator topology.

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
5. Offshore build cost multipliers not yet generated (see "Offshore Build Cost Overrides" section).

## ARC Cluster Operations

### Environment policy
- Local development: `.venv`.
- ARC production runs: conda environment (`/data/<group>/<user>/envs/green-lory-env`).
- ARC working directory policy: use `$DATA` (`/data/engs-df-green-ammonia/engs2523`) for repo/env/logs/results; do not run from ARC home (`~/`).

### ARC script layout
- `arc/arc_initial_setup.sh`: one-time ARC bootstrap (clone/update + optional env build submission).
- `arc/build-green-lory-env.sh`: SLURM env build job.
- `arc/load_green_lory_env.sh`: module + conda activation helper for interactive shell use.
- `arc/arc_check_run_inputs.sh`: preflight check for required inputs before run submission.
- `arc/jobs/01_run_global.sh`: full global run SLURM job.
- `arc/submit_global_run.sh`: preflight + submit wrapper.

### Recommended ARC run sequence
1. `bash arc/arc_initial_setup.sh`
2. `sbatch arc/build-green-lory-env.sh` (if env not already built)
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
- `ARC_TIME_STEP` (timestep resolution in hours, default 1.0)
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
rsync -avz --delete \
  --exclude '.venv/' \
  --exclude '__pycache__/' \
  --exclude '.git/' \
  --exclude 'data/*.nc' \
  --exclude 'data/*.hdf' \
  --exclude 'data/*.geojson' \
  --exclude 'results/' \
  --exclude 'logs/' \
  --exclude 'slurm-*.out' \
  --exclude 'notebooks/' \
  ./ engs2523@arc-login.arc.ox.ac.uk:/data/engs-df-green-ammonia/engs2523/green-lory/
```

For large data files (max_capacities CSV, weather NetCDFs), use explicit `scp`:

```bash
scp data/20251222_max_capacities.csv \
  engs2523@arc-login.arc.ox.ac.uk:/data/engs-df-green-ammonia/engs2523/green-lory/data/
```

Pull results back after ARC jobs complete:

```bash
rsync -avz \
  engs2523@arc-login.arc.ox.ac.uk:/data/engs-df-green-ammonia/engs2523/green-lory/results/ \
  ./results/
```

## Performance Optimisations (2026-03-08)

### Pre-computation caching in `run_global.py`
- Interest/land DataFrames pre-indexed into `{(lat,lon): ...}` dicts for O(1) per-cell lookup.
- Country assignment uses batch `sjoin` for all locations upfront.
- Base PyPSA network built once and `copy.deepcopy()`-ed per cell (avoids re-importing CSVs).
- Weather extraction bypasses `renewable_data` class, reads directly from cached xarray DataArrays.
- Net effect: per-cell overhead reduced from ~1.5 s to < 20 ms.

### Gurobi solver tuning
- Added `BarConvTol=1e-4` to solver_options in `main.py` (relaxed from default 1e-8).
- Saves ~19% solve time with < 0.1 EUR/t LCOA impact.
- Barrier (Method=2, Crossover=0) remains the optimal solver method for this 8760-snapshot LP.

### Profiling summary
Per-cell time breakdown (8760 snapshots, Gurobi barrier):
- Weather/lookup/deepcopy/cost setup: ~20 ms (0.3%)
- `n.optimize()` linopy model build: ~475 ms (8%)
- `m.solve()` Gurobi barrier: ~4.7 s (91%)
- Post-processing: ~50 ms (1%)

The Gurobi barrier solve is the irreducible bottleneck for full-year (8760) resolution.

### Timestep resolution (`time_step` parameter)
`run_global()` and the CLI accept a `time_step` parameter (default 1.0 h). When
set to e.g. 3.0, hourly weather profiles are block-averaged into 3-hourly
snapshots (8760 → 2920), reducing LP size proportionally.

**100-cell Queensland comparison (3h vs 1h):**

| Metric | 1h | 3h | Difference |
|---|---|---|---|
| LCOA mean | 848.2 EUR/t | 841.8 EUR/t | −6.4 (−0.75%) |
| LCOA max deviation | — | — | −14.8 EUR/t (−1.49%) |
| Battery mean | 6901 MWh | 6685 MWh | −216 (−4.1%) |
| Battery individual | — | — | −39% to +108% |
| Solve speed | ~5 s/cell | ~2.4 s/cell | 2.1× faster |

LCOA accuracy is excellent (< 1.5% deviation). Battery sizing shows higher
variance due to smoothing of intra-day peaks — expected for storage-sensitive
assets. The 3h resolution is suitable for screening runs; 1h remains the
production default.

## Documentation Governance
- Keep only two canonical docs at repo root:
  - `README.md` for users
  - `DEVELOPMENT_NOTES.md` for developers/agents
- Move implementation history and technical change logs into this file instead of creating additional root Markdown documents.
