#!/bin/bash
#SBATCH --job-name=green-lory-global
#SBATCH --partition=medium
#SBATCH --clusters=all
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=370G
#SBATCH --time=2-00:00:00
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

conda activate "$ARC_ENV_PREFIX"

cd "$ARC_REPO_DIR"
mkdir -p logs "results/${RUN_LABEL}"

if [[ -n "$LOCATIONS_ARG" ]]; then
  export ARC_LOCATIONS_CSV="$LOCATIONS_ARG"
fi

export ARC_TECH_YAML="${ARC_TECH_YAML:-inputs/tech_config_ammonia_plant_2030_dea.yaml}"
export ARC_INTEREST_CSV="${ARC_INTEREST_CSV:-}"
export ARC_LAND_CSV="${ARC_LAND_CSV:-data/20251222_max_capacities.csv}"
export ARC_WEATHER_DIR="${ARC_WEATHER_DIR:-data/weather_data}"
if [[ ! -f "$ARC_LAND_CSV" && -f "data/20251222_land_max_capacity.csv" ]]; then
  export ARC_LAND_CSV="data/20251222_land_max_capacity.csv"
fi

bash arc/arc_check_run_inputs.sh "${ARC_LOCATIONS_CSV:-}" >/dev/null

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
export ARC_OUTPUT_CSV="${ARC_OUTPUT_CSV:-results/${RUN_LABEL}/run_global_${RUN_LABEL}_${STAMP}.csv}"
export ARC_QUIET="${ARC_QUIET:-1}"
export ARC_LON_MIN="${ARC_LON_MIN:-}"
export ARC_LON_MAX="${ARC_LON_MAX:-}"
export ARC_TIME_STEP="${ARC_TIME_STEP:-1.0}"
export ARC_FAIL_FAST="${ARC_FAIL_FAST:-1}"  # set to 0 to swallow per-location failures silently
export ARC_ENSURE_FEASIBILITY="${ARC_ENSURE_FEASIBILITY:-1}"  # set to 0 to disable grid backstop (hard infeasibility)

LOGFILE="logs/arc-${RUN_LABEL}-${STAMP}.log"
echo "Run label: $RUN_LABEL"
echo "Log file:  $LOGFILE"
echo "Output:    $ARC_OUTPUT_CSV"

echo "Workers: ARC_NUM_WORKERS=${ARC_NUM_WORKERS}, ARC_THREADS_PER_WORKER=${ARC_THREADS_PER_WORKER}" | tee -a "$LOGFILE"
env | grep -E '^(ARC_|GREEN_LORY_|GRB_LICENSE_FILE|OMP_NUM_THREADS|MKL_NUM_THREADS|OPENBLAS_NUM_THREADS)' | sort | tee -a "$LOGFILE"

python - <<'PY' 2>&1 | tee -a "$LOGFILE"
import os
import pandas as pd
from model.run_global import run_global

locations = None
locations_csv = os.environ.get("ARC_LOCATIONS_CSV", "").strip()
limit_raw = os.environ.get("ARC_LIMIT", "").strip()
max_snapshots_raw = os.environ.get("ARC_MAX_SNAPSHOTS", "").strip()
threads_raw = os.environ.get("ARC_THREADS_PER_WORKER", "1").strip()

if locations_csv:
    df = pd.read_csv(locations_csv)
    cols = {c.lower(): c for c in df.columns}
    if "lat" not in cols or "lon" not in cols:
        raise SystemExit(f"Locations CSV must contain lat/lon columns: {locations_csv}")
    locations = list(zip(df[cols["lat"]].astype(float), df[cols["lon"]].astype(float)))

if limit_raw:
    limit = int(limit_raw)
    if locations is not None:
        locations = locations[:limit]
    else:
        land_csv = os.environ.get("ARC_LAND_CSV")
        ldf = pd.read_csv(land_csv)
        lcols = {c.lower(): c for c in ldf.columns}
        lat_col = lcols.get("latitude") or lcols.get("lat")
        lon_col = lcols.get("longitude") or lcols.get("lon")
        if lat_col is None or lon_col is None:
            raise SystemExit(f"Land CSV missing latitude/longitude columns: {land_csv}")
        locations = list(zip(ldf[lat_col].astype(float), ldf[lon_col].astype(float)))[:limit]

max_snapshots = int(max_snapshots_raw) if max_snapshots_raw else None
quiet_raw = os.environ.get("ARC_QUIET", "1").strip().lower()
quiet = quiet_raw not in {"0", "false", "no"}
threads_per_worker = int(threads_raw) if threads_raw else None
lon_min_raw = os.environ.get("ARC_LON_MIN", "").strip()
lon_max_raw = os.environ.get("ARC_LON_MAX", "").strip()
lon_min = float(lon_min_raw) if lon_min_raw else None
lon_max = float(lon_max_raw) if lon_max_raw else None
time_step_raw = os.environ.get("ARC_TIME_STEP", "1.0").strip()
time_step = float(time_step_raw) if time_step_raw else 1.0
num_workers_raw = os.environ.get("ARC_NUM_WORKERS", "1").strip()
num_workers = int(num_workers_raw) if num_workers_raw else 1
fail_fast_raw = os.environ.get("ARC_FAIL_FAST", "1").strip().lower()
fail_fast = fail_fast_raw in {"1", "true", "yes"}
ensure_feasibility_raw = os.environ.get("ARC_ENSURE_FEASIBILITY", "1").strip().lower()
ensure_feasibility = ensure_feasibility_raw not in {"0", "false", "no"}

result_df = run_global(
    locations=locations,
    land_csv=os.environ.get("ARC_LAND_CSV"),
    interest_csv=(os.environ.get("ARC_INTEREST_CSV") or "").strip() or None,
    tech_yaml=os.environ.get("ARC_TECH_YAML"),
    time_step=time_step,
    max_snapshots=max_snapshots,
    output_csv=os.environ.get("ARC_OUTPUT_CSV"),
    quiet=quiet,
    threads_per_worker=threads_per_worker,
    num_workers=num_workers,
    lon_min=lon_min,
    lon_max=lon_max,
    weather_dir=os.environ.get("ARC_WEATHER_DIR") or None,
    fail_fast=fail_fast,
    ensure_feasibility=ensure_feasibility,
)

print(f"Processed {len(result_df)} locations")
print(f"Saved: {os.environ.get('ARC_OUTPUT_CSV')}")
PY

echo "Completed run: $RUN_LABEL"
