# ARC Cluster Scripts

This folder contains ARC (Oxford) helper scripts for running full global jobs from this repository.

- Local development: prefer `.venv`.
- ARC cluster runs: use conda environment under `/data/<group>/<user>/envs/green-lory-env`.
- ARC login account for this project: `engs2523`.

## Files
- `arc/arc_initial_setup.sh`: one-time setup on ARC login node (clone/update repo, directories, optional env-build submission).
- `arc/build-green-lory-env`: SLURM job script that creates/refreshes the conda env and installs dependencies.
- `arc/load_green_lory_env.sh`: shell helper to load modules + activate the conda env for interactive ARC sessions.
- `arc/arc_check_run_inputs.sh`: preflight checker for required inputs before submitting a global run.
- `arc/jobs/01_run_global.sh`: SLURM job script that executes `model.run_global` end-to-end.
- `arc/submit_global_run.sh`: convenience wrapper to run preflight + submit the SLURM job.
- `scripts/make_flat_finance_overrides.py`: generates a flat finance CSV from the spatial template coverage.

## Typical ARC Workflow

### 1. One-time setup on ARC login node
```bash
ssh engs2523@arc-login.arc.ox.ac.uk
cd /data/engs-df-green-ammonia/engs2523
bash green-lory/arc/arc_initial_setup.sh
```

### 2. Build/refresh environment
```bash
cd /data/engs-df-green-ammonia/engs2523/green-lory
sbatch arc/build-green-lory-env
```

### 3. Optional interactive check with conda env loaded
```bash
cd /data/engs-df-green-ammonia/engs2523/green-lory
source arc/load_green_lory_env.sh
bash arc/arc_check_run_inputs.sh
```

### 4. Submit full global run
```bash
cd /data/engs-df-green-ammonia/engs2523/green-lory
bash arc/submit_global_run.sh full-global-2030
```

Or submit directly:
```bash
cd /data/engs-df-green-ammonia/engs2523/green-lory
sbatch arc/jobs/01_run_global.sh full-global-2030
```

### 5. Select finance mode at submission time

Default (no overrides CSV, use tech-config interest values everywhere):
```bash
cd /data/engs-df-green-ammonia/engs2523/green-lory
bash arc/submit_global_run.sh full-global-2030
```

Spatial finance CSV:
```bash
cd /data/engs-df-green-ammonia/engs2523/green-lory
bash arc/submit_global_run.sh full-global-2030 --finance-mode spatial
```

Flat finance CSV:
```bash
cd /data/engs-df-green-ammonia/engs2523/green-lory
bash arc/submit_global_run.sh full-global-2030 --finance-mode flat
```

Custom finance CSV:
```bash
cd /data/engs-df-green-ammonia/engs2523/green-lory
bash arc/submit_global_run.sh full-global-2030 --finance-mode custom --interest-csv inputs/my_finance.csv
```

Optional: enforce a single global flat rate (e.g. 8%)
```bash
cd /data/engs-df-green-ammonia/engs2523/green-lory
ARC_FLAT_INTEREST_RATE=0.08 bash arc/submit_global_run.sh full-global-2030 --finance-mode flat
```

### 6. Parallelize inside `run_global`

`model.run_global` now supports multiprocessing directly. Control process and thread budgets through ARC env vars:

```bash
cd /data/engs-df-green-ammonia/engs2523/green-lory
ARC_WORKERS=8 ARC_THREADS_PER_WORKER=2 ARC_RAM_PER_WORKER_GB=20 bash arc/submit_global_run.sh full-global-2030-flat --finance-mode flat
```

`run_global` fails fast when requested workers/threads/RAM exceed detected machine allocation.

## Password-based interactive usage (VS Code agent friendly)
If you are entering password for each SSH command, use single-command SSH invocations from your local terminal and provide password when prompted:

```bash
ssh engs2523@arc-login.arc.ox.ac.uk 'cd /data/engs-df-green-ammonia/engs2523/green-lory && bash arc/submit_global_run.sh full-global-2030'
```

Repeat per command as needed:
```bash
ssh engs2523@arc-login.arc.ox.ac.uk 'squeue -u engs2523'
ssh engs2523@arc-login.arc.ox.ac.uk 'tail -n 80 /data/engs-df-green-ammonia/engs2523/green-lory/logs/arc-full-global-2030-*.log'
```

## Environment variables (optional)
You can override defaults without editing scripts:
- `ARC_GROUP` (default: `engs-df-green-ammonia`)
- `ARC_WORK_BASE` (default: `/data/$ARC_GROUP/$USER`)
- `ARC_REPO_DIR` (default: `$ARC_WORK_BASE/green-lory`)
- `ARC_ENV_PREFIX` (default: `$ARC_WORK_BASE/envs/green-lory-env`)
- `ARC_ANACONDA_MODULE` (default: `Anaconda3/2024.06-1`)
- `ARC_TECH_YAML` (default: `inputs/tech_config_ammonia_plant_2030_qld.yaml`)
- `ARC_INTEREST_CSV` (default: unset / no overrides)
- `ARC_FINANCE_MODE` (default: `none`; alternatives: `spatial`, `flat`, `custom`)
- `ARC_FLAT_INTEREST_RATE` (optional constant flat CoC for all tech/location rows)
- `ARC_LAND_CSV` (default: `data/20251222_max_capacities.csv`)
- `ARC_LOCATIONS_CSV` (optional location subset)
- `ARC_MAX_SNAPSHOTS` (optional smoke-test cap)
- `ARC_LIMIT` (optional location cap)
- `ARC_OUTPUT_CSV` (optional explicit output path)
- `ARC_QUIET` (default: `1`)
- `ARC_WORKERS` (default: `1`)
- `ARC_THREADS_PER_WORKER` (default: `1`)
- `ARC_RAM_PER_WORKER_GB` (optional per-worker RAM request for preflight validation)
- `ARC_MAX_RAM_GB` (optional total RAM request for preflight validation)

## Notes
- `arc/jobs/01_run_global.sh` writes outputs under `results/<run-label>/` by default.
- If `data/20251222_max_capacities.csv` is missing, scripts automatically fallback to `data/20251222_land_max_capacity.csv` when available.
