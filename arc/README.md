# ARC Cluster Scripts

This folder contains ARC (Oxford) helper scripts for running full global jobs from this repository.

- Local development: prefer `.venv`.
- ARC cluster runs: use conda environment under `/data/<group>/<user>/envs/green-lory-env`.
- ARC login account for this project: `engs2523`.

## Files
- `arc/arc_initial_setup.sh`: one-time setup on ARC login node (clone/update repo, directories, optional env-build submission).
- `arc/build-green-lory-env.sh`: SLURM job script that creates/refreshes the conda env and installs dependencies.
- `arc/load_green_lory_env.sh`: shell helper to load modules + activate the conda env for interactive ARC sessions.
- `arc/arc_check_run_inputs.sh`: preflight checker for required inputs before submitting a global run.
- `arc/jobs/01_run_global.sh`: SLURM job script that executes `model.run_global` end-to-end.
- `arc/submit_global_run.sh`: convenience wrapper to run preflight + submit the SLURM job.
- `arc/jobs/00_build_land_constraints.sh`: SLURM job script for either heavy 100pct land builds or cheap derived competition rescaling.
- `arc/submit_land_constraints_matrix.sh`: ARC-side wrapper that submits the 100pct/derived max-capacity matrix.
- `arc/stage_and_submit_land_constraints.sh`: local helper that stages the required data/code to ARC and then calls `arc/submit_land_constraints_matrix.sh` remotely.
- `arc/submit_constrained_reruns.sh`: convenience wrapper for the canonical constrained DEA/Way rerun set; queues dependent merge jobs automatically.
- `scripts/merge_global_results.py`: canonical quadrant merge helper used by ARC workflows.

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
sbatch arc/build-green-lory-env.sh
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

**Recommended: 4 parallel quadrant jobs** (splits by longitude so each shard fits the short partition):
```bash
cd /data/engs-df-green-ammonia/engs2523/green-lory
bash arc/submit_global_run.sh full-global-2030 --quadrants
```

This submits 4 SLURM jobs with longitude bounds:
- `west2`: [-180, -90)
- `west1`: [-90, 0)
- `east1`: [0, 90)
- `east2`: [90, 180)

After all quadrant jobs complete, merge them with the canonical merge helper:
```bash
cd /data/engs-df-green-ammonia/engs2523/green-lory
python scripts/merge_global_results.py full-global-2030 \
	--output results/full_global_2030/global_run_results.csv
```

The notebook combiner cell is still fine for inspection, but `scripts/merge_global_results.py`
is the supported merge path because it deduplicates and validates required columns.

Or submit directly:
```bash
cd /data/engs-df-green-ammonia/engs2523/green-lory
sbatch arc/jobs/01_run_global.sh full-global-2030
```

### 4b. Submit the canonical constrained reruns

This wrapper submits the standard 2050 `flat` and explicit spatial-mechanism cost runs for Way and DEA, plus the Salmon/Verschuur WAY 2050 Amelired replication cases at 4h resolution. It can also include DEA 2030. Each scenario is submitted as 4 quadrant jobs and followed by a dependent merge job.

```bash
cd /data/engs-df-green-ammonia/engs2523/green-lory
bash arc/submit_constrained_reruns.sh --land-csv data/max_capacities_paper_2pct_slope15.csv --include-2030
```

To limit submission to one or more scenarios:

```bash
cd /data/engs-df-green-ammonia/engs2523/green-lory
bash arc/submit_constrained_reruns.sh \
	--land-csv data/max_capacities_high_50pct_slope15.csv \
	--scenario way-2050-spatial-build-remote-water \
	--scenario dea-2050-spatial-build-remote-water
```

For the closest Salmon/Verschuur replication target, use the WACC-only 4h scenario. This keeps build/remoteness/water/land flat and only applies Ameli reduced WACC:

```bash
cd /data/engs-df-green-ammonia/engs2523/green-lory
bash arc/submit_constrained_reruns.sh \
	--land-csv data/max_capacities_paper_2pct_slope15.csv \
	--scenario way-2050-flat-amelired-4h
```

For the spatial-sensitivity comparison, use the explicit spatial build/remoteness/water 4h scenario:

```bash
cd /data/engs-df-green-ammonia/engs2523/green-lory
bash arc/submit_constrained_reruns.sh \
	--land-csv data/max_capacities_paper_2pct_slope15.csv \
	--scenario way-2050-spatial-build-remote-water-amelired-4h
```

The ambiguous `way-2050-spatial-amelired` filter is intentionally unsupported in this wrapper. One-hour Amelired reruns should use a separately named scenario when they are added. The current combined Ameli file has active build/remoteness/water mechanisms and zero land cost, so the canonical spatial-sensitivity label is `spatial-build-remote-water-amelired`, not plain `spatial-amelired`.

When `--land-tag` is omitted, the wrapper infers one from the land CSV name when it follows the `max_capacities_<tag>.csv` pattern. That tag is appended to the run label and merged results directory, so the land-cap choice stays visible in downstream ARC outputs.

### 4c. Submit the land-processing matrix

From your local machine, stage the required data/code and submit the initial 100pct/derived matrix:

```bash
cd /Users/carlopalazzi/programming/pypsa_models/green-lory
bash arc/stage_and_submit_land_constraints.sh
```

This submits four outputs:
- `data/max_capacities_100pct_slope15.csv`
- `data/max_capacities_100pct_allslopes.csv`
- `data/max_capacities_paper_2pct_slope15.csv`
- `data/max_capacities_high_50pct_slope15.csv`

`100pct_slope15` and `100pct_allslopes` are the heavy ARC builds. The `paper_2pct_slope15` and `high_50pct_slope15` files are submitted as dependent rescaling jobs from `100pct_slope15`, so they do not rerun the full geospatial overlay.

### 5. Select override inputs at submission time

The generic `submit_global_run.sh` flag is `--override-mode` because the CSV can
carry more than finance. The flag selects the override file; the run label should
name the active mechanisms. In current Green Lory inputs, the spatial base file
has active build multipliers, active remoteness, active water costs, and zero
land cost, so use labels such as `way-2050-spatial-build-remote-water` rather
than plain `way-2050-spatial` for new runs.

Default (no overrides CSV, use tech-config interest values everywhere):
```bash
cd /data/engs-df-green-ammonia/engs2523/green-lory
bash arc/submit_global_run.sh full-global-2030
```

Spatial override CSV:
```bash
cd /data/engs-df-green-ammonia/engs2523/green-lory
bash arc/submit_global_run.sh full-global-2030 --override-mode spatial
```

Custom override CSV:
```bash
cd /data/engs-df-green-ammonia/engs2523/green-lory
bash arc/submit_global_run.sh full-global-2030 --override-mode custom --override-csv inputs/my_overrides.csv
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
- `ARC_OVERRIDE_CSV` (default: unset / no overrides)
- `ARC_OVERRIDE_MODE` (default: `none`; alternatives: `spatial`, `custom`)
- `ARC_LAND_CSV` (default: `data/max_capacities_paper_2pct_slope15.csv`)
- `ARC_LOCATIONS_CSV` (optional location subset)
- `ARC_MAX_SNAPSHOTS` (optional smoke-test cap)
- `ARC_TIME_STEP` (default: `1.0`; set by constrained scenarios, `4.0` for Salmon/Verschuur Amelired replication)
- `ARC_LIMIT` (optional location cap)
- `ARC_OUTPUT_CSV` (optional explicit output path)
- `ARC_QUIET` (default: `1`)
- `ARC_THREADS_PER_WORKER` (default: all available CPUs)
- `ARC_LON_MIN` / `ARC_LON_MAX` (optional longitude bounds for segmented runs)

## Notes
- `arc/jobs/01_run_global.sh` writes outputs under `results/<run-label>/` by default.
