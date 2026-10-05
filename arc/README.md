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
- `arc/submit_land_center_matrix.sh`: ARC-side wrapper for one 100pct build plus derived shares (default 2 % and 20 %) written to an immutable campaign directory; the heavy job goes to `--clusters=all` and the derived jobs follow it on the accepted cluster. Cells are anchored at their centre (`ARC_CELL_ANCHOR=center`, the default of the template since 23 Sep 2026; `southwest` reproduces the pre-September labelling convention only for audits).
- `arc/stage_and_submit_land_constraints.sh`: local helper that stages the required data/code to ARC and then calls `arc/submit_land_constraints_matrix.sh` remotely.
- `arc/submit_constrained_reruns.sh`: convenience wrapper for the canonical constrained DEA/Way rerun set; queues dependent merge jobs automatically.
- `arc/submit_lory_sequence.sh`: immutable campaign wrapper for the Green Lory reconciliation sequence.
- `arc/merge_and_qa_campaign.py`: merges an explicit shard list and enforces coordinate, uniqueness, currency, finance-override, manifest, and schema gates.
- `arc/write_campaign_manifest.py`: records source and input identities for one immutable campaign run.
- `arc/write_campaign_submission.py`: records job IDs separately so the manifest hash cannot race running shards.
- `scripts/merge_global_results.py`: canonical quadrant merge helper used by ARC workflows.

## Green Lory Reconciliation Campaign

Use `submit_lory_sequence.sh` for the publication-replication and central-realism
runs. It is intentionally additive: the older general-purpose wrappers remain
available, while this path never reuses a mutable shard folder or discovers a
result with a “latest file” glob.

Supported scenarios:

- `rep_way2050_flat_amelired_4h_tracking_nominal_h2`: four-hour historical replication with tracking PV, nominal compressor CAPEX, legacy bundled H2 storage, legacy-scaled temporal accounting and per-snapshot ramps. Land remains a post-processing capacity calculation using `scaled_reference_design` with the `exclusive` allocation (September name `paper_union`). The flat cost scope uses Ameli reduced WACC, unity build/remoteness multipliers, and uniform baseline water at 2 USD/m3; water is reported but excluded from the replication headline, and no land-rent input is available.
- `central_way2050_flat_amelired_1h_fixed_explicit_compressor_dea_tank_colocated` (central case since 16 September 2026): hourly, fixed-tilt PV only (plant bundle without tracking), explicit compressor CAPEX, DEA 2050 tank-only H2 storage, snapshot-weighted accounting, and the `colocated` land allocation: wind and PV draw on the same suitable-land budget and only the exclusive fraction of the wind footprint (3 %, Denholm et al. 2009 direct-impact area) competes with PV. Tracking-only is the named sensitivity.
- `central_way2050_flat_amelired_1h_tracking_explicit_compressor_dea_tank` (September 2026 definition, retained for reproducibility): hourly central case with tracking PV, explicit compressor CAPEX, DEA 2050 tank-only H2 storage, snapshot-weighted temporal accounting and per-hour ramps. Land remains a post-processing capacity calculation using `scaled_reference_design` with the `exclusive` allocation (September name `technology_shared`). The same flat cost scope uses Ameli reduced WACC, unity build/remoteness multipliers, and uniform baseline water at 2 USD/m3; baseline water is included in the headline, while land rent remains unmodelled and zero.

The `flat_amelired` token is deliberate. These baselines do not use the spatial
build, remoteness, or water-access columns in
`inputs/spatial_cost_inputs_amelired_2050.csv`. That combined input belongs in a
separately named `spatial_build_remote_water_amelired` sensitivity; its current
land-rent column is also zero.

Both scientific scenarios disable the feasibility grid backstop. Conservative
or unversioned renewable-union inputs are allowed only for runtime smoke work.
Full-year diagnostic and global submissions require the versioned
`renewable_union_area_km2_classwise_nested_v1` column and fail in preflight if
it is absent. This v1 estimate sums the larger wind/solar suitability fraction
within each MODIS class. Because the underlying eligible footprints are not
spatially resolved, it assumes perfect nested overlap and is a lower bound on
the physical union, not an exact union. `renewable_union_area_km2` remains a
numerically identical compatibility alias in newly generated tables.

Start with a local or ARC-side dry run:

```bash
bash arc/submit_lory_sequence.sh \
  --stage smoke \
  --run-id review-only \
  --dry-run
```

Local dry runs validate the tracked YAML, override, diagnostic-cell, and plant
bundle inputs. Large `data/` inputs are ARC-only and are reported as notes.
Non-dry submissions require the full ARC preflight to pass. Preflight also
requires the interest-only override to cover every explicit diagnostic cell or
every positive-capacity global land cell.

The intended gated sequence is:

```bash
# Three cells over 168 simulated hours: operational check.
bash arc/submit_lory_sequence.sh --stage smoke

# Same three cells over the full weather year: scientific comparison. Point to
# the immutable 2% table produced for this campaign.
bash arc/submit_lory_sequence.sh \
  --stage diagnostic \
  --land-csv /data/<group>/<user>/green-lory-campaigns/<campaign>/land/paper_2pct_slope15.csv

# Submit one global baseline only after reviewing its diagnostic result.
bash arc/submit_lory_sequence.sh \
  --stage global \
  --land-csv /data/<group>/<user>/green-lory-campaigns/<campaign>/land/paper_2pct_slope15.csv \
  --scenario rep_way2050_flat_amelired_4h_tracking_nominal_h2

bash arc/submit_lory_sequence.sh \
  --stage global \
  --land-csv /data/<group>/<user>/green-lory-campaigns/<campaign>/land/paper_2pct_slope15.csv \
  --scenario central_way2050_flat_amelired_1h_tracking_explicit_compressor_dea_tank
```

Every invocation receives a unique UTC/source run ID by default. Passing
`--run-id` is supported, but the wrapper refuses an existing destination.
For ARC reconciliation work, keep staged source and generated artifacts in
separate roots:

```text
/data/<group>/<user>/green-lory-releases/<release-id>/
/data/<group>/<user>/green-lory-campaigns/<campaign>/
├── land/
└── results/
```

The release may link to the existing read-only large `data/` payload; it must
not reuse the legacy repository's mutable `results/` or `logs/` folders.
Campaign results are stored under:

```text
results/campaigns/<campaign>/
├── 00_smoke/<scenario>/runs/<run-id>/smoke/
├── 10_replication/<scenario>/runs/<run-id>/<diagnostic-or-global>/
└── 20_central/<scenario>/runs/<run-id>/<diagnostic-or-global>/
```

Each run contains a byte-immutable `manifest.json`, a separate `submission.json`
for SLURM job IDs, exact `shards/`, the validated `merged/` CSV,
`qa/validation.json`, and SLURM `logs/`. The manifest records the complete
resolved technology-YAML inheritance chain, not only an overlay file. It also
records the override CSV columns and a structured cost scope: Ameli reduced
WACC, no spatial build/remoteness override, unity build multiplier, 2 USD/m3
uniform YAML water, and no land-rent input. A
merge/QA job runs only after all of that run's explicit shard job IDs finish
successfully. It fails if rows are
missing or duplicated, unexpected coordinates appear, currency or finance
metadata disagree, required columns are absent, a result's build/water/land
values contradict the manifest scope, or a failure-sidecar CSV exists.

Useful controls:

- `--campaign`, `--run-id`, and `--results-root` set immutable output identity.
- `--land-csv` pins both scenarios to one versioned campaign land table; the
  manifest hashes that exact file.
- `--cluster` pins every shard and merge/QA job to one ARC cluster. Without it,
  the first returned cluster is used for all remaining jobs in that scenario.
- `--smoke-hours` changes the short-run simulated duration. The 168-hour
  default becomes 168 snapshots in the hourly central case and 42 snapshots in
  the four-hour replication case.
- `--scenario` is repeatable; omitting it selects both baseline scenarios.
- `ARC_CAMPAIGN_THREADS_PER_WORKER`, `ARC_CAMPAIGN_DIAGNOSTIC_WORKERS`, and
  `ARC_CAMPAIGN_GLOBAL_WORKERS` control worker sizing.

Do not promote or delete an older result merely because a replacement was
submitted. Review the merged CSV and require `qa/validation.json` to report
`"status": "passed"` first.

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

For the reconciliation campaign, submit from the immutable source release and
write every artifact below the campaign root. The matrix wrapper and each land
job now refuse an existing output by default:

```bash
campaign_root=/data/<group>/<user>/green-lory-campaigns/lory_reconcile_20260722_v1
ARC_PCT100_SLOPE15_CSV="$campaign_root/land/100pct_slope15.csv" \
ARC_PCT100_ALLSLOPES_CSV="$campaign_root/land/100pct_allslopes.csv" \
ARC_PAPER_2PCT_SLOPE15_CSV="$campaign_root/land/paper_2pct_slope15.csv" \
ARC_HIGH_50PCT_SLOPE15_CSV="$campaign_root/land/high_50pct_slope15.csv" \
ARC_LAND_LOG_DIR="$campaign_root/land/logs" \
bash arc/submit_land_constraints_matrix.sh
```

The older local staging helper below targets the legacy repository and is kept
only for historical/general-purpose maintenance; do not use it for the
reconciliation campaign.

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
