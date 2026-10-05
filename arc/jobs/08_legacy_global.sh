#!/bin/bash
#SBATCH --job-name=glr-legacy-global-v2
#SBATCH --clusters=htc
#SBATCH --partition=short
#SBATCH --array=0-15
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=04:00:00
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=carlo.palazzi@eng.ox.ac.uk
# Legacy-lcoa replication surface: the frozen 3 May 2023 model at every archived supplier
# coordinate (15,377 cells in 16 shards), replicating configuration of
# reconciliation/legacy_lcoa/LEGACY_REPLICATION_20260915.md section 3
# (x_Cost RCP4.5 2050 CAPEX, tracking PV enabled, 4-hour step, Ameli reduced WACC by country).
# Weather comes from the compact per-cell store (extract_weather_store.py): the contiguous
# NetCDF-3 files cost 13-70 s per cell in strided reads on the parallel file system.
set -euo pipefail
: "${LEGACY_RELEASE:?pinned source release directory required}"
: "${LEGACY_OUTPUT:?new campaign output directory required}"
STORE="${LEGACY_STORE:-/data/engs-df-green-ammonia/engs2523/green-lory/data/weather_store_archived15377_v1}"
PY="${LEGACY_ENV:-/data/engs-df-green-ammonia/engs2523/envs/legacy-lcoa-env}/bin/python"
export GRB_LICENSE_FILE="${GRB_LICENSE_FILE:-/apps/system/easybuild/software/Gurobi/10.0.3-GCCcore-12.2.0/gurobi.lic}"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
cd "$LEGACY_RELEASE"
SHARD=$(printf "%02d" "$SLURM_ARRAY_TASK_ID")
date -u; scontrol --oneliner show job "$SLURM_JOB_ID" | tr ' ' '\n' | grep -E '^(JobId|NumCPUs|MinMemoryNode|TimeLimit|MailUser|MailType)=' || true
test -f "$STORE/manifest.json" || { echo "weather store missing: $STORE" >&2; exit 3; }
mkdir -p "$LEGACY_OUTPUT"
"$PY" reconciliation/legacy_lcoa/run_legacy_cells.py \
  --era may2023 --variants stated_4h_mean --enable-tracking 1.0587 --capex-source xcost45 ${LEGACY_EXTRA_ARGS:-} \
  --weather-store "$STORE" \
  --cells "reconciliation/legacy_lcoa/shards/cells_archived_shard_${SHARD}.csv" \
  --solver gurobi --threads "${SLURM_CPUS_PER_TASK:-2}" --no-timeseries --skip-existing \
  --output "$LEGACY_OUTPUT/shard_${SHARD}" 2>&1 | grep --line-buffered -vE 'FutureWarning|DeprecationWarning|warnings.warn|groupby\(|attrs.loc|^INFO:|Solver Results|^#|^-|^ *$|Problem:|Solver:|Solution:|Status:|Return code|Message:|Termination|Wall time|Error rc|Time:|Lower bound|Upper bound|Number of|Sense:|Name:|number of'
date -u
