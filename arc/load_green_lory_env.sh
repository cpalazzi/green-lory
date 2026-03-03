#!/bin/bash
# Source this script to load ARC modules + activate green-lory conda env.

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  echo "Source this script instead:" >&2
  echo "  source arc/load_green_lory_env.sh" >&2
  exit 1
fi

set -euo pipefail

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

ARC_ANACONDA_MODULE="${ARC_ANACONDA_MODULE:-Anaconda3/2024.06-1}"
module load "$ARC_ANACONDA_MODULE"

if [ -n "${EBROOTANACONDA3:-}" ] && [ -f "$EBROOTANACONDA3/etc/profile.d/conda.sh" ]; then
  source "$EBROOTANACONDA3/etc/profile.d/conda.sh"
elif command -v conda >/dev/null 2>&1; then
  eval "$(conda shell.bash hook)"
fi

if ! command -v conda >/dev/null 2>&1; then
  echo "ERROR: conda not available after loading $ARC_ANACONDA_MODULE" >&2
  return 2
fi

USER_NAME="${USER:-$(id -un)}"
ARC_GROUP="${ARC_GROUP:-engs-df-green-ammonia}"
ARC_WORK_BASE="${ARC_WORK_BASE:-/data/${ARC_GROUP}/${USER_NAME}}"
ARC_ENV_PREFIX="${ARC_ENV_PREFIX:-${ARC_WORK_BASE}/envs/green-lory-env}"
ARC_LICENSE_DIR="${ARC_LICENSE_DIR:-${ARC_WORK_BASE}/licenses}"

if [[ ! -d "$ARC_ENV_PREFIX" ]]; then
  echo "ERROR: env not found: $ARC_ENV_PREFIX" >&2
  return 2
fi

conda activate "$ARC_ENV_PREFIX"

if [[ -z "${GRB_LICENSE_FILE:-}" && -f "$ARC_LICENSE_DIR/gurobi.lic" ]]; then
  export GRB_LICENSE_FILE="$ARC_LICENSE_DIR/gurobi.lic"
fi

echo "Activated conda env: $ARC_ENV_PREFIX"
echo "Python: $(command -v python)"
