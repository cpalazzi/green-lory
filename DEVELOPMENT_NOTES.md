# Development Notes (Developers and AI Agents)

## Scope
This file is the technical reference for architecture, modeling conventions, cost-split logic, and implementation status.

`README.md` is user-facing. Keep deep technical details here.

## Repository Architecture
- `model/main.py`: single-site orchestration and PyPSA solve entrypoint
- `model/run_global.py`: multi-location orchestration, finance overrides, output assembly
- `model/auxiliary.py`: constraint hooks, weather IO helpers, reporting transforms
- `model/location_tools.py`: weather/location geospatial utilities
- `model/land_processing.py`: land/bathymetry-based max-capacity preprocessing
- `basic_ammonia_plant/*.csv`: canonical PyPSA topology and techno-economic tables used at runtime
- `inputs/tech_config_ammonia_plant_2030_*.yaml`: scenario assumptions compiled into plant CSVs by notebook
- `notebooks/00_tech_config.ipynb`: YAML -> CSV compiler for costs/efficiencies/link recipes
- `notebooks/0*_*.ipynb`: execution and analysis workflows

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
- Spatial override inputs via `inputs/example_finance_overrides_spatial.csv`:
  - `interest_rate`
  - `build_cost_multiplier`
  - `land_cost_usd_per_km2_year`
  - `water_cost_usd_per_m3`
- `model/run_global.py` recomputes annualized costs under finance overrides.
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

## Config Completion Status (as of 2026-02-27)

### Completed
1. DEA split arithmetic reconciled (`overnight = tech + build`) across all configured technologies.
2. Lithium-ion allocation rule standardized (1-hour, 50/50 split of "other project costs" across MW/MWh).
3. Datasheet-derived split updates applied for:
- solar
- solar tracking
- onshore wind
- offshore wind fixed
- offshore wind floating
- hydrogen compression
4. Updated DEA split ratios mirrored into QLD structured config while keeping QLD overnight costs fixed.

### Remaining data gaps
1. `hydrogen_from_storage`: placeholder CAPEX/split (no direct valve/regulator line identified).
2. `hydrogen_fuel_cell`: no direct imported DEA line identified; still proxy-converted.
3. `ammonia` storage: still proxy-based pending direct DEA source.
4. QLD-specific EPC/build breakdown evidence is still needed for fully local split localization.

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
5. `bash arc/submit_global_run.sh <run-label>`

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
- `ARC_WORKERS`
- `ARC_THREADS_PER_WORKER`
- `ARC_RAM_PER_WORKER_GB`
- `ARC_MAX_RAM_GB`
- `ARC_ANACONDA_MODULE`
- `ARC_ENV_PREFIX`
- `ARC_GROUP`, `ARC_WORK_BASE`, `ARC_REPO_DIR`

Default production behavior is full sweep (`run_global`) with output written under `results/<run-label>/`.

## Documentation Governance
- Keep only two canonical docs at repo root:
  - `README.md` for users
  - `DEVELOPMENT_NOTES.md` for developers/agents
- Move implementation history and technical change logs into this file instead of creating additional root Markdown documents.
