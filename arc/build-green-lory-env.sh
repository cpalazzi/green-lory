#!/bin/bash
#SBATCH --job-name=build-green-lory-env
#SBATCH --partition=medium
#SBATCH --clusters=all
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=carlo.palazzi@eng.ox.ac.uk

set -euo pipefail

set +u
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
  source /usr/share/lmod/lmod/init/bash
fi
set -u

module purge
ARC_ANACONDA_MODULE="${ARC_ANACONDA_MODULE:-Anaconda3/2024.06-1}"
module load "$ARC_ANACONDA_MODULE"

if [ -n "${EBROOTANACONDA3:-}" ] && [ -f "$EBROOTANACONDA3/etc/profile.d/conda.sh" ]; then
  source "$EBROOTANACONDA3/etc/profile.d/conda.sh"
elif command -v conda >/dev/null 2>&1; then
  eval "$(conda shell.bash hook)"
fi

if ! command -v conda >/dev/null 2>&1; then
  echo "ERROR: conda not found after loading module $ARC_ANACONDA_MODULE" >&2
  exit 2
fi

USER_NAME="${USER:-$(id -un)}"
ARC_GROUP="${ARC_GROUP:-engs-df-green-ammonia}"
ARC_WORK_BASE="${ARC_WORK_BASE:-/data/${ARC_GROUP}/${USER_NAME}}"
ARC_REPO_DIR="${ARC_REPO_DIR:-${ARC_WORK_BASE}/green-lory}"
ARC_ENV_PREFIX="${ARC_ENV_PREFIX:-${ARC_WORK_BASE}/envs/green-lory-env}"
ARC_LOG_DIR="${ARC_ENV_LOGDIR:-${ARC_WORK_BASE}/envs/logs}"
mkdir -p "$ARC_LOG_DIR" "$(dirname "$ARC_ENV_PREFIX")"

if [[ ! -f "$ARC_REPO_DIR/requirements.txt" ]]; then
  echo "ERROR: requirements.txt not found in $ARC_REPO_DIR" >&2
  exit 2
fi

if [[ -d "$ARC_ENV_PREFIX" ]]; then
  rm -rf "${ARC_ENV_PREFIX}.old" || true
  mv "$ARC_ENV_PREFIX" "${ARC_ENV_PREFIX}.old"
fi

conda create -y -p "$ARC_ENV_PREFIX" python=3.11 pip
"$ARC_ENV_PREFIX/bin/python" -m pip install --upgrade pip wheel
"$ARC_ENV_PREFIX/bin/pip" install -r "$ARC_REPO_DIR/requirements.txt"
"$ARC_ENV_PREFIX/bin/pip" install openpyxl

if [[ "${ARC_INSTALL_GUROBI:-0}" == "1" ]]; then
  conda install -y -p "$ARC_ENV_PREFIX" -c gurobi gurobi
fi

"$ARC_ENV_PREFIX/bin/pip" freeze > "$ARC_LOG_DIR/green-lory-env-pip-freeze.txt"
"$ARC_ENV_PREFIX/bin/python" - <<'PY'
import sys
print("Python:", sys.version)
for mod in ("pypsa", "pandas", "xarray", "geopandas"):
    __import__(mod)
print("Core imports OK")
PY

echo "Environment ready: $ARC_ENV_PREFIX"
