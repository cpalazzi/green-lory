#!/bin/bash
#SBATCH --job-name=glr-weather-profiles
#SBATCH --partition=medium
#SBATCH --clusters=all
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=360G
#SBATCH --time=2-00:00:00
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=carlo.palazzi@eng.ox.ac.uk
# One-degree PV and wind profiles from the ERA5 2019 cutout (model/weather_profiles.py). Three
# modes, each run from a pinned release, writing only to a new campaign directory:
#
#   check: full-load hours at DEA's reference cells (Denmark and its North Sea), about 4 hours:
#     six conversions, each over the whole year whatever the number of cells (wind about 15
#     minutes, PV more than 27), so one hour is too short
#     ARC_RELEASE=<release> ARC_MODE=check ARC_OUTPUT=<campaign>/check.json \
#       sbatch --partition=short --time=12:00:00 --cpus-per-task=4 --mem=32G --output=<campaign>/logs/%x-%j.out arc/jobs/11_weather_profiles.sh
#   convert: one band of 15 cell latitudes per array task (12 tasks cover -90 to 89); the
#     resources are those of December 2025's atlite tiles of the same size, 17 to 50 hours each;
#     --array=8 --partition=short --time=12:00:00 runs one band, the log timing each technology
#     ARC_RELEASE=<release> ARC_MODE=convert ARC_OUTPUT=<campaign>/bands \
#       sbatch --array=0-11 --output=<campaign>/logs/%x-%A_%a.out arc/jobs/11_weather_profiles.sh
#   stack: stitch the bands into the weather directory read by runs (weather_dir); ARC_SEA=1
#   writes a sea run's stack instead, the wind at sea and no PV output (<campaign>/profiles_sea)
#     ARC_RELEASE=<release> ARC_MODE=stack ARC_BANDS=<campaign>/bands ARC_OUTPUT=<campaign>/profiles \
#       sbatch --partition=short --time=04:00:00 --cpus-per-task=8 --mem=64G --output=<campaign>/logs/%x-%j.out arc/jobs/11_weather_profiles.sh
#
# The green-condor environment is borrowed read-only: it has atlite 0.4.1, which green-lory-env lacks.
set -euo pipefail
: "${ARC_RELEASE:?pinned source release directory required}"
: "${ARC_MODE:?check, convert or stack}"
: "${ARC_OUTPUT:?output path required}"
CUTOUT="${ARC_CUTOUT:-/data/engs-df-green-ammonia/engs2523/green-condor/data/global_cutout_2019.nc}"
PY="${ARC_ENV:-/data/engs-df-green-ammonia/engs2523/envs/green-condor-env}/bin/python"
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
cd "$ARC_RELEASE"
date -u; hostname
case "$ARC_MODE" in
  check)
    [[ -e "$ARC_OUTPUT" ]] && { echo "ERROR: refusing to overwrite $ARC_OUTPUT" >&2; exit 2; }
    "$PY" -m model.weather_profiles check --cutout "$CUTOUT" --output "$ARC_OUTPUT"
    ;;
  convert)
    : "${SLURM_ARRAY_TASK_ID:?convert runs as an array job (--array=0-11)}"
    SOUTH=$(( -90 + 15 * SLURM_ARRAY_TASK_ID )); NORTH=$(( SOUTH + 14 ))
    BAND="$ARC_OUTPUT/band_${SOUTH}_${NORTH}.nc"
    [[ -e "$BAND" ]] && { echo "ERROR: refusing to overwrite $BAND" >&2; exit 2; }
    mkdir -p "$ARC_OUTPUT"
    "$PY" -m model.weather_profiles convert --cutout "$CUTOUT" --cells="${SOUTH}:${NORTH}" --output "$BAND"
    sha256sum "$BAND"
    ;;
  stack)
    : "${ARC_BANDS:?directory of band files required}"
    [[ -e "$ARC_OUTPUT" ]] && { echo "ERROR: refusing to overwrite $ARC_OUTPUT" >&2; exit 2; }
    "$PY" -m model.weather_profiles stack --bands "$ARC_BANDS" --output "$ARC_OUTPUT" ${ARC_SEA:+--sea}
    ;;
  *) echo "ERROR: unknown mode $ARC_MODE" >&2; exit 2;;
esac
date -u
