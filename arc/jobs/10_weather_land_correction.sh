#!/bin/bash
#SBATCH --job-name=glr-weather-land
#SBATCH --partition=short
#SBATCH --clusters=all
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=06:00:00
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=carlo.palazzi@eng.ox.ac.uk
# The land correction of the 1-degree weather profiles (model/weather_land.py ratios): reads the
# ERA5 capacity-factor stores of green-condor (0.25 degrees, zarr) and writes the factors of every
# cell with water. Run from a pinned release; the output must not exist yet.
#
#   ARC_RELEASE=<release dir> ARC_OUTPUT=<campaign dir>/weather_land_correction_2019.nc \
#     sbatch --output=<campaign dir>/logs/%x-%j.out arc/jobs/10_weather_land_correction.sh
#
# The green-condor environment is borrowed read-only: it has zarr, which green-lory-env lacks.
set -euo pipefail
: "${ARC_RELEASE:?pinned source release directory required}"
: "${ARC_OUTPUT:?output NetCDF path required}"
STORES="${ARC_ERA5_STORES:-/data/engs-df-green-ammonia/engs2523/green-condor/outputs}"
PY="${ARC_ENV:-/data/engs-df-green-ammonia/engs2523/envs/green-condor-env}/bin/python"
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
if [[ -e "$ARC_OUTPUT" ]]; then
  echo "ERROR: refusing to overwrite $ARC_OUTPUT" >&2
  exit 2
fi
cd "$ARC_RELEASE"
date -u; hostname
"$PY" -m model.weather_land ratios --weights inputs/weather/era5_point_weights_1deg.csv \
  --stores "$STORES" --legacy-dir data/weather_data/salmon2022_pv_wind_2019_hourly_1deg --output "$ARC_OUTPUT"
sha256sum "$ARC_OUTPUT"
date -u
