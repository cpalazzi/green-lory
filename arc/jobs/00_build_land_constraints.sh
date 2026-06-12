#!/bin/bash
#SBATCH --job-name=green-lory-land
#SBATCH --partition=short
#SBATCH --clusters=all
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=carlo.palazzi@eng.ox.ac.uk

set -euo pipefail

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

if [[ ! -d "$ARC_ENV_PREFIX" ]]; then
  echo "ERROR: conda env not found: $ARC_ENV_PREFIX" >&2
  exit 2
fi

conda activate "$ARC_ENV_PREFIX" 2>/dev/null || true
ARC_PYTHON="${ARC_ENV_PREFIX}/bin/python"
if [[ ! -x "$ARC_PYTHON" ]]; then
  echo "ERROR: python not found at $ARC_PYTHON" >&2
  exit 2
fi

cd "$ARC_REPO_DIR"
mkdir -p logs

LAND_TAG="${ARC_LAND_TAG:-full_land_constraints}"
LAND_COVER="${ARC_LAND_COVER:-data/MCD12C1.A2022001.061.2023244164746.hdf}"
SLOPE_RASTER="${ARC_SLOPE_RASTER:-data/GEBCO_2025_sub_ice.nc}"
OUTPUT_CSV="${ARC_LAND_OUTPUT_CSV:-data/max_capacities_${LAND_TAG}.csv}"
BASE_LAND_CSV="${ARC_BASE_LAND_CSV:-}"
LAND_COMPETITION_FRACTION="${ARC_LAND_COMPETITION_FRACTION:-1.0}"
SOURCE_LAND_COMPETITION_FRACTION="${ARC_SOURCE_LAND_COMPETITION_FRACTION:-}"
MAX_SLOPE_DEGREES="${ARC_MAX_SLOPE_DEGREES:-15}"
SKIP_SLOPE_EXCLUSION="${ARC_SKIP_SLOPE_EXCLUSION:-0}"
INCLUDE_OFFSHORE_WIND="${ARC_INCLUDE_OFFSHORE_WIND:-1}"
PROTECTED_AREA_0="${ARC_PROTECTED_AREA_0:-data/WDPA_Feb2026_Public_shp_0/WDPA_Feb2026_Public_shp-polygons.shp}"
PROTECTED_AREA_1="${ARC_PROTECTED_AREA_1:-data/WDPA_Feb2026_Public_shp_1/WDPA_Feb2026_Public_shp-polygons.shp}"
PROTECTED_AREA_2="${ARC_PROTECTED_AREA_2:-data/WDPA_Feb2026_Public_shp_2/WDPA_Feb2026_Public_shp-polygons.shp}"

required_paths=()
if [[ -n "$BASE_LAND_CSV" ]]; then
  required_paths+=("$BASE_LAND_CSV")
else
  required_paths+=(
    "$LAND_COVER"
    "$PROTECTED_AREA_0"
    "$PROTECTED_AREA_1"
    "$PROTECTED_AREA_2"
  )
  if [[ "$SKIP_SLOPE_EXCLUSION" != "1" ]]; then
    required_paths+=("$SLOPE_RASTER")
  fi
fi

for path in "${required_paths[@]}"; do
  if [[ ! -f "$path" ]]; then
    echo "ERROR: required input not found: $path" >&2
    exit 2
  fi
done

STAMP="$(date +%Y%m%d-%H%M%S)"
LOGFILE="logs/arc-land-availability-${LAND_TAG}-${STAMP}.log"

mkdir -p "$(dirname "$OUTPUT_CSV")"

echo "Repo:        $ARC_REPO_DIR" | tee -a "$LOGFILE"
echo "Python:      $ARC_PYTHON" | tee -a "$LOGFILE"
echo "Tag:         $LAND_TAG" | tee -a "$LOGFILE"
echo "Base CSV:    ${BASE_LAND_CSV:-<heavy-build>}" | tee -a "$LOGFILE"
echo "Land frac:   $LAND_COMPETITION_FRACTION" | tee -a "$LOGFILE"
echo "Source frac: ${SOURCE_LAND_COMPETITION_FRACTION:-<auto>}" | tee -a "$LOGFILE"
echo "Slope skip:  $SKIP_SLOPE_EXCLUSION" | tee -a "$LOGFILE"
echo "Max slope:   $MAX_SLOPE_DEGREES" | tee -a "$LOGFILE"
echo "Offshore:    $INCLUDE_OFFSHORE_WIND" | tee -a "$LOGFILE"
if [[ -z "$BASE_LAND_CSV" ]]; then
  echo "Land cover:  $LAND_COVER" | tee -a "$LOGFILE"
  if [[ "$SKIP_SLOPE_EXCLUSION" != "1" ]]; then
    echo "Slope:       $SLOPE_RASTER" | tee -a "$LOGFILE"
  fi
  echo "Protected:   $PROTECTED_AREA_0" | tee -a "$LOGFILE"
  echo "Protected:   $PROTECTED_AREA_1" | tee -a "$LOGFILE"
  echo "Protected:   $PROTECTED_AREA_2" | tee -a "$LOGFILE"
fi
echo "Output:      $OUTPUT_CSV" | tee -a "$LOGFILE"

cmd=(
  "$ARC_PYTHON"
  model/land_processing.py
  --output "$OUTPUT_CSV"
  --land-competition-fraction "$LAND_COMPETITION_FRACTION"
)

if [[ "$INCLUDE_OFFSHORE_WIND" == "1" ]]; then
  cmd+=(--include-offshore-wind)
fi

if [[ -n "$BASE_LAND_CSV" ]]; then
  cmd+=(--base-csv "$BASE_LAND_CSV")
  if [[ -n "$SOURCE_LAND_COMPETITION_FRACTION" ]]; then
    cmd+=(--source-land-competition-fraction "$SOURCE_LAND_COMPETITION_FRACTION")
  fi
else
  cmd+=(
    --land-cover "$LAND_COVER"
    --protected-area "$PROTECTED_AREA_0"
    --protected-area "$PROTECTED_AREA_1"
    --protected-area "$PROTECTED_AREA_2"
  )
  if [[ "$SKIP_SLOPE_EXCLUSION" == "1" ]]; then
    cmd+=(--skip-slope-exclusion)
  else
    cmd+=(--slope-raster "$SLOPE_RASTER" --max-slope-degrees "$MAX_SLOPE_DEGREES")
  fi
fi

printf 'Command:     ' | tee -a "$LOGFILE"
printf '%q ' "${cmd[@]}" | tee -a "$LOGFILE"
printf '\n' | tee -a "$LOGFILE"

"${cmd[@]}" 2>&1 | tee -a "$LOGFILE"

ls -lh "$OUTPUT_CSV" | tee -a "$LOGFILE"
