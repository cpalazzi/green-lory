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

Single job (all longitudes):
```bash
cd /data/engs-df-green-ammonia/engs2523/green-lory
bash arc/submit_global_run.sh full-global-2030
```

**Recommended: 4 parallel quadrant jobs** (splits by longitude, each ~20 hours):
```bash
cd /data/engs-df-green-ammonia/engs2523/green-lory
bash arc/submit_global_run.sh full-global-2030 --quadrants
```

This submits 4 SLURM jobs with longitude bounds:
- `west2`: [-180, -90)
- `west1`: [-90, 0)
- `east1`: [0, 90)
- `east2`: [90, 180)

After all quadrant jobs complete, use the results combiner cell in notebook 03
or merge manually:
```bash
cat results/full-global-2030-*/run_global_*.csv | head -1 > combined.csv
tail -q -n +2 results/full-global-2030-*/run_global_*.csv >> combined.csv
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

Custom finance CSV:
```bash
cd /data/engs-df-green-ammonia/engs2523/green-lory
bash arc/submit_global_run.sh full-global-2030 --finance-mode custom --interest-csv inputs/my_finance.csv
```

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
- `ARC_TECH_YAML` (default: `inputs/tech_config_ammonia_plant_2030_dea.yaml`)
- `ARC_INTEREST_CSV` (default: unset / no overrides)
- `ARC_FINANCE_MODE` (default: `none`; alternatives: `spatial`, `custom`)
- `ARC_LAND_CSV` (default: `data/20251222_max_capacities.csv`)
- `ARC_LOCATIONS_CSV` (optional location subset)
- `ARC_MAX_SNAPSHOTS` (optional smoke-test cap)
- `ARC_LIMIT` (optional location cap)
- `ARC_OUTPUT_CSV` (optional explicit output path)
- `ARC_QUIET` (default: `1`)
- `ARC_THREADS_PER_WORKER` (default: all available CPUs)
- `ARC_LON_MIN` / `ARC_LON_MAX` (optional longitude bounds for segmented runs)

## Notes
- `arc/jobs/01_run_global.sh` writes outputs under `results/<run-label>/` by default.
- If `data/20251222_max_capacities.csv` is missing, scripts automatically fallback to `data/20251222_land_max_capacity.csv` when available.
