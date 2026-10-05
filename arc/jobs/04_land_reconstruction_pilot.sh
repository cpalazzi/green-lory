#!/bin/bash
#SBATCH --job-name=glr-land-pilot
#SBATCH --clusters=htc
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=carlo.palazzi@eng.ox.ac.uk
set -euo pipefail
: "${LAND_RELEASE:?Pinned source release required}"
: "${LAND_DATA:?Read-only source data directory required}"
: "${LAND_CELLS:?Explicit pilot cell CSV required}"
: "${LAND_OUTPUT:?New output directory required}"
: "${LAND_ANCHOR:?center or southwest required}"
if [[ -e "$LAND_OUTPUT" ]]; then
  echo "Refusing to overwrite $LAND_OUTPUT" >&2
  exit 2
fi
cd "$LAND_RELEASE"
scontrol show job "$SLURM_JOB_ID"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
/data/engs-df-green-ammonia/engs2523/envs/green-lory-env/bin/python \
  reconciliation/land/pilot_joint_masks.py \
  --modis "$LAND_DATA/MCD12C1.A2022001.061.2023244164746.hdf" \
  --dem "$LAND_DATA/GEBCO_2025_sub_ice.nc" \
  --protected \
    "$LAND_DATA/WDPA_Feb2026_Public_shp_0/WDPA_Feb2026_Public_shp-polygons.shp" \
    "$LAND_DATA/WDPA_Feb2026_Public_shp_1/WDPA_Feb2026_Public_shp-polygons.shp" \
    "$LAND_DATA/WDPA_Feb2026_Public_shp_2/WDPA_Feb2026_Public_shp-polygons.shp" \
  --cells "$LAND_CELLS" --anchor "$LAND_ANCHOR" --output "$LAND_OUTPUT"
