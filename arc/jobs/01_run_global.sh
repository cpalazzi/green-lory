#!/bin/bash
#SBATCH --job-name=green-lory-global
#SBATCH --partition=short
#SBATCH --clusters=all
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=370G
#SBATCH --time=12:00:00
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=carlo.palazzi@eng.ox.ac.uk

set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: sbatch arc/jobs/01_run_global.sh <run-label> [locations-csv]" >&2
  exit 2
fi

RUN_LABEL="$1"
LOCATIONS_ARG="${2:-${ARC_LOCATIONS_CSV:-}}"

set +eu
if [ -f /etc/profile ]; then
  source /etc/profile
fi
if [ -f /etc/profile.d/modules.sh ]; then
  source /etc/profile.d/modules.sh
fi
if [ -f /etc/profile.d/lmod.sh ]; then
  source /etc/profile.d/lmod.sh
fi
if ! command -v module >/dev/null 2>&1; then
  source /usr/share/lmod/lmod/init/bash || true
fi
set -eu

ARC_ANACONDA_MODULE="${ARC_ANACONDA_MODULE:-Anaconda3/2024.06-1}"
module purge
module load "$ARC_ANACONDA_MODULE"

if [ -n "${EBROOTANACONDA3:-}" ] && [ -f "$EBROOTANACONDA3/etc/profile.d/conda.sh" ]; then
  source "$EBROOTANACONDA3/etc/profile.d/conda.sh"
elif command -v conda >/dev/null 2>&1; then
  eval "$(conda shell.bash hook)"
fi

USER_NAME="${USER:-$(id -un)}"
ARC_GROUP="${ARC_GROUP:-engs-df-green-ammonia}"
ARC_WORK_BASE="${ARC_WORK_BASE:-/data/${ARC_GROUP}/${USER_NAME}}"
DEFAULT_REPO_DIR="${SLURM_SUBMIT_DIR:-}"
if [[ -z "$DEFAULT_REPO_DIR" ]]; then
  DEFAULT_REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
fi
ARC_REPO_DIR="${ARC_REPO_DIR:-$DEFAULT_REPO_DIR}"
ARC_ENV_PREFIX="${ARC_ENV_PREFIX:-${ARC_WORK_BASE}/envs/green-lory-env}"
ARC_LICENSE_DIR="${ARC_LICENSE_DIR:-${ARC_WORK_BASE}/licenses}"

# Use the ARC institutional token-server license (USELIMIT=4096) by pointing
# GRB_LICENSE_FILE directly at the known path — no `module load Gurobi` needed
# (that module ships Python 3.10 gurobipy which segfaults under Python 3.11).
# Fall back to the personal WLS license only if the system file is absent.
ARC_SYSTEM_GRB_LIC="/apps/system/easybuild/software/Gurobi/10.0.3-GCCcore-12.2.0/gurobi.lic"
if [[ -z "${GRB_LICENSE_FILE:-}" ]]; then
  if [[ -f "$ARC_SYSTEM_GRB_LIC" ]]; then
    export GRB_LICENSE_FILE="$ARC_SYSTEM_GRB_LIC"
  elif [[ -f "$ARC_LICENSE_DIR/gurobi.lic" ]]; then
    export GRB_LICENSE_FILE="$ARC_LICENSE_DIR/gurobi.lic"
  fi
fi
echo "Gurobi license: ${GRB_LICENSE_FILE:-(none set)}"

if [[ ! -d "$ARC_ENV_PREFIX" ]]; then
  echo "ERROR: conda env not found: $ARC_ENV_PREFIX" >&2
  exit 2
fi

conda activate "$ARC_ENV_PREFIX" 2>/dev/null || true
# Use the env's Python explicitly so the job works even if conda activate
# silently fails to update PATH (common in non-interactive SLURM batch shells).
ARC_PYTHON="${ARC_ENV_PREFIX}/bin/python"
if [[ ! -x "$ARC_PYTHON" ]]; then
  echo "ERROR: python not found at $ARC_PYTHON" >&2
  exit 2
fi

cd "$ARC_REPO_DIR"
CAMPAIGN_RUN_DIR="${ARC_RUN_OUTPUT_DIR:-}"
if [[ -n "$CAMPAIGN_RUN_DIR" ]]; then
  mkdir -p "$CAMPAIGN_RUN_DIR/logs"
else
  mkdir -p logs "results/${RUN_LABEL}"
fi

if [[ -n "$LOCATIONS_ARG" ]]; then
  export ARC_LOCATIONS_CSV="$LOCATIONS_ARG"
fi

export ARC_TECH_YAML="${ARC_TECH_YAML:?ARC_TECH_YAML must name the technology YAML of the run}"
export ARC_OVERRIDE_CSV="${ARC_OVERRIDE_CSV:-}"
export ARC_SITE_COSTS_CSV="${ARC_SITE_COSTS_CSV:-}"
export ARC_WEATHER_LAND_CORRECTION="${ARC_WEATHER_LAND_CORRECTION:-}"
export ARC_LAND_CSV="${ARC_LAND_CSV:?ARC_LAND_CSV must point at a centred land table (cell_anchor=center)}"
export ARC_WEATHER_DIR="${ARC_WEATHER_DIR:-data/weather_data/salmon2022_pv_wind_2019_hourly_1deg}"

bash arc/arc_check_run_inputs.sh "${ARC_LOCATIONS_CSV:-}" >/dev/null

if [[ -n "$CAMPAIGN_RUN_DIR" ]]; then
  if [[ -z "${ARC_RUN_MANIFEST:-}" || ! -f "$ARC_RUN_MANIFEST" ]]; then
    echo "ERROR: campaign run requires an existing ARC_RUN_MANIFEST" >&2
    exit 2
  fi
  "$ARC_PYTHON" -m arc.manifest verify \
    --manifest "$ARC_RUN_MANIFEST"
  if [[ -z "${ARC_RELEASE_INVENTORY:-}" ]]; then
    echo "ERROR: campaign run requires ARC_RELEASE_INVENTORY" >&2
    exit 2
  fi
  "$ARC_PYTHON" -m arc.release verify \
    --root "$ARC_REPO_DIR" \
    --inventory "$ARC_RELEASE_INVENTORY"
fi

CPUS="${SLURM_CPUS_PER_TASK:-48}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export GREEN_LORY_SOLVER_LOG="${GREEN_LORY_SOLVER_LOG:-0}"
# Default: 4 Gurobi threads per worker, 12 parallel workers (= 48 CPUs).
# Override with ARC_THREADS_PER_WORKER and ARC_NUM_WORKERS env vars.
export ARC_THREADS_PER_WORKER="${ARC_THREADS_PER_WORKER:-4}"
export ARC_NUM_WORKERS="${ARC_NUM_WORKERS:-$((CPUS / ARC_THREADS_PER_WORKER))}"

STAMP="$(date +%Y%m%d-%H%M%S)"
if [[ -n "$CAMPAIGN_RUN_DIR" ]]; then
  if [[ -z "${ARC_OUTPUT_CSV:-}" ]]; then
    echo "ERROR: ARC_RUN_OUTPUT_DIR requires an explicit ARC_OUTPUT_CSV" >&2
    exit 2
  fi
  export ARC_OUTPUT_CSV
else
  export ARC_OUTPUT_CSV="${ARC_OUTPUT_CSV:-results/${RUN_LABEL}/run_global_${RUN_LABEL}_${STAMP}.csv}"
fi
export ARC_QUIET="${ARC_QUIET:-1}"
export ARC_LON_MIN="${ARC_LON_MIN:-}"
export ARC_LON_MAX="${ARC_LON_MAX:-}"
export ARC_TIME_STEP="${ARC_TIME_STEP:-1.0}"
export ARC_FAIL_FAST="${ARC_FAIL_FAST:-1}"  # set to 0 to swallow per-location failures silently
export ARC_LAND_CONSTRAINT="${ARC_LAND_CONSTRAINT:-after_solve}"
export ARC_CAPACITY_RULE="${ARC_CAPACITY_RULE:-scaled_reference_design}"
export ARC_LAND_ALLOCATION="${ARC_LAND_ALLOCATION:-colocated}"
export ARC_TEMPORAL_ACCOUNTING_MODE="${ARC_TEMPORAL_ACCOUNTING_MODE:-snapshot_weighted}"
export ARC_RAMP_LIMIT_BASIS="${ARC_RAMP_LIMIT_BASIS:-per_hour}"
export ARC_INCLUDE_SITE_COSTS="${ARC_INCLUDE_SITE_COSTS:-1}"

if [[ -n "$CAMPAIGN_RUN_DIR" ]]; then
  LOGFILE="$CAMPAIGN_RUN_DIR/logs/worker-${ARC_SHARD:-single}-${STAMP}.log"
else
  LOGFILE="logs/arc-${RUN_LABEL}-${STAMP}.log"
fi
echo "Run label: $RUN_LABEL"
echo "Log file:  $LOGFILE"
echo "Output:    $ARC_OUTPUT_CSV"

echo "Workers: ARC_NUM_WORKERS=${ARC_NUM_WORKERS}, ARC_THREADS_PER_WORKER=${ARC_THREADS_PER_WORKER}" | tee -a "$LOGFILE"
env | grep -E '^(ARC_|GREEN_LORY_|GRB_LICENSE_FILE|OMP_NUM_THREADS|MKL_NUM_THREADS|OPENBLAS_NUM_THREADS)' | sort | tee -a "$LOGFILE"

"$ARC_PYTHON" - <<'PY' 2>&1 | tee -a "$LOGFILE"
import os

import pandas as pd

from model.surface import RunSettings, run_surface


def env(name, default=""):
    return os.environ.get(name, default).strip()


def flag(name, default):
    return env(name, default).lower() not in {"0", "false", "no"}


locations = None
if env("ARC_LOCATIONS_CSV"):
    frame = pd.read_csv(env("ARC_LOCATIONS_CSV"))
    columns = {column.lower(): column for column in frame.columns}
    if "lat" not in columns or "lon" not in columns:
        raise SystemExit(f"Locations CSV must contain lat/lon columns: {env('ARC_LOCATIONS_CSV')}")
    locations = list(zip(frame[columns["lat"]].astype(float), frame[columns["lon"]].astype(float)))
if env("ARC_LIMIT"):
    limit = int(env("ARC_LIMIT"))
    if locations is None:
        land = pd.read_csv(env("ARC_LAND_CSV"))
        columns = {column.lower(): column for column in land.columns}
        lat, lon = columns.get("latitude") or columns.get("lat"), columns.get("longitude") or columns.get("lon")
        if lat is None or lon is None:
            raise SystemExit(f"Land CSV missing latitude/longitude columns: {env('ARC_LAND_CSV')}")
        locations = list(zip(land[lat].astype(float), land[lon].astype(float)))
    locations = locations[:limit]

settings = RunSettings(
    time_step=float(env("ARC_TIME_STEP", "1.0") or 1.0),
    max_snapshots=int(env("ARC_MAX_SNAPSHOTS")) if env("ARC_MAX_SNAPSHOTS") else None,
    pv_policy=env("ARC_PV_POLICY", "both"),
    land_constraint=env("ARC_LAND_CONSTRAINT", "after_solve"),
    capacity_rule=env("ARC_CAPACITY_RULE", "scaled_reference_design"),
    land_allocation=env("ARC_LAND_ALLOCATION", "colocated"),
    temporal_accounting_mode=env("ARC_TEMPORAL_ACCOUNTING_MODE", "snapshot_weighted"),
    ramp_limit_basis=env("ARC_RAMP_LIMIT_BASIS", "per_hour"),
    include_site_costs=flag("ARC_INCLUDE_SITE_COSTS", "1"),
)
result = run_surface(
    locations,
    land_csv=env("ARC_LAND_CSV"),
    tech_yaml=env("ARC_TECH_YAML"),
    plant_dir=env("ARC_PLANT_DIR") or None,
    override_csv=env("ARC_OVERRIDE_CSV") or None,
    site_costs_csv=env("ARC_SITE_COSTS_CSV") or None,
    weather_dir=env("ARC_WEATHER_DIR") or None,
    weather_land_correction=env("ARC_WEATHER_LAND_CORRECTION") or None,
    settings=settings,
    output_csv=env("ARC_OUTPUT_CSV"),
    quiet=flag("ARC_QUIET", "1"),
    threads_per_worker=int(env("ARC_THREADS_PER_WORKER", "1") or 1),
    num_workers=int(env("ARC_NUM_WORKERS", "1") or 1),
    lon_min=float(env("ARC_LON_MIN")) if env("ARC_LON_MIN") else None,
    lon_max=float(env("ARC_LON_MAX")) if env("ARC_LON_MAX") else None,
    fail_fast=flag("ARC_FAIL_FAST", "1"),
)
print(f"Processed {len(result)} locations")
print(f"Saved: {env('ARC_OUTPUT_CSV')}")
PY

echo "Completed run: $RUN_LABEL"
