#!/bin/bash
#SBATCH --job-name=glr-land-supply-3-v1
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
: "${SUPPLY_RELEASE:?Pinned source release required}"
: "${SUPPLY_LAND:?Versioned common land CSV required}"
: "${SUPPLY_WEATHER_RUN:?Completed fixed-PV run containing hashed weather required}"
: "${SUPPLY_OUTPUT:?New output directory required}"
cd "$SUPPLY_RELEASE"
scontrol show job "$SLURM_JOB_ID"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export GREEN_LORY_SOLVER_THREADS=4 GREEN_LORY_SOLVER_LOG=0 PYTHONUNBUFFERED=1
export GRB_LICENSE_FILE=/apps/system/easybuild/software/Gurobi/10.0.3-GCCcore-12.2.0/gurobi.lic
/data/engs-df-green-ammonia/engs2523/envs/green-lory-env/bin/python \
  reconciliation/land/run_supply_pilot.py --land "$SUPPLY_LAND" \
  --weather-run "$SUPPLY_WEATHER_RUN" --output "$SUPPLY_OUTPUT" \
  --pv-policy "${SUPPLY_PV_POLICY:-both}" \
  --experiment-config "${SUPPLY_CONFIG:-reconciliation/land/supply_curve/config_v1.json}"
