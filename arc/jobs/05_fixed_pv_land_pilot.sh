#!/bin/bash
#SBATCH --job-name=glr-pv-fixed-3
#SBATCH --clusters=htc
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=carlo.palazzi@eng.ox.ac.uk
set -euo pipefail
: "${PV_RELEASE:?Pinned source release required}"
: "${PV_LAND:?Versioned common-land CSV required}"
: "${PV_OUTPUT:?New fixed-PV result directory required}"
cd "$PV_RELEASE"
scontrol show job "$SLURM_JOB_ID"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export GRB_LICENSE_FILE=/apps/system/easybuild/software/Gurobi/10.0.3-GCCcore-12.2.0/gurobi.lic
export PYTHONUNBUFFERED=1 GREEN_LORY_SOLVER_LOG=0
/data/engs-df-green-ammonia/engs2523/envs/green-lory-env/bin/python \
  reconciliation/land/run_pv_pilot.py \
  --weather /data/engs-df-green-ammonia/engs2523/green-lory/data/weather_data \
  --land "$PV_LAND" --output "$PV_OUTPUT"
