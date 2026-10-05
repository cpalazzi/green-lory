#!/bin/bash
#SBATCH --job-name=gpo-modamb-replay
#SBATCH --partition=short
#SBATCH --clusters=htc
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=carlo.palazzi@eng.ox.ac.uk
set -euo pipefail
if [[ $# -lt 3 ]]; then
  echo "Usage: sbatch $0 <source-release> <historical-archive> <new-output-directory> [runner-options]" >&2
  exit 2
fi
REPLAY_SOURCE="$1"
REPLAY_ARCHIVE="$2"
REPLAY_OUTPUT="$3"
shift 3
REPLAY_ENV="${ARC_ENV_PREFIX:-/data/engs-df-green-ammonia/engs2523/envs/green-lory-env}"
export GRB_LICENSE_FILE="${GRB_LICENSE_FILE:-/apps/system/easybuild/software/Gurobi/10.0.3-GCCcore-12.2.0/gurobi.lic}"
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export PATH="${REPLAY_ENV}/bin:$PATH"
cd "$REPLAY_SOURCE"
date -u
scontrol --oneliner show job "$SLURM_JOB_ID"
"${REPLAY_ENV}/bin/python" reconciliation/run_historical_network.py \
  --archive "$REPLAY_ARCHIVE" --output "$REPLAY_OUTPUT" \
  --threads "${SLURM_CPUS_PER_TASK:-12}" "$@"
date -u
