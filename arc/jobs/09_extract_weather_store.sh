#!/bin/bash
#SBATCH --job-name=glr-weather-store
#SBATCH --clusters=htc
#SBATCH --partition=devel
#SBATCH --array=0-8
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=00:10:00
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=carlo.palazzi@eng.ox.ac.uk
# One task per legacy weather file: sequential block reads of the contiguous NetCDF-3 file,
# gathering the requested cells into a float64 array (values unchanged). Run the merge step
# afterwards (extract_weather_store.py --merge) to validate and write the store manifest.
set -euo pipefail
: "${LEGACY_RELEASE:?pinned source release directory required}"
: "${LEGACY_STORE:?store output directory required}"
: "${LEGACY_CELLS:?cells CSV (lat, lon) required}"
WEATHER="${LEGACY_WEATHER:-/data/engs-df-green-ammonia/engs2523/green-lory/data/weather_data}"
PY="${LEGACY_ENV:-/data/engs-df-green-ammonia/engs2523/envs/legacy-lcoa-env}/bin/python"
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
FILES=(Solar.nc Solar1.nc Solar2.nc SolarTracking.nc SolarTracking1.nc SolarTracking2.nc WindPowers.nc WindPowers1.nc WindPowers2.nc)
FILE="${FILES[$SLURM_ARRAY_TASK_ID]}"
cd "$LEGACY_RELEASE"
date -u; hostname
"$PY" reconciliation/legacy_lcoa/extract_weather_store.py --weather-dir "$WEATHER" --cells "$LEGACY_CELLS" \
  --output "$LEGACY_STORE" --file "$FILE"
date -u
