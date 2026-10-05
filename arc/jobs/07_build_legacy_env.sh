#!/bin/bash
#SBATCH --job-name=build-legacy-lcoa-env
#SBATCH --clusters=htc
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=carlo.palazzi@eng.ox.ac.uk
# Builds a dedicated environment for the frozen legacy-lcoa model (PyPSA 0.25.1 with the
# Pyomo lopf path, pinned as in lcoa-opt/environment.yaml). Mirrors arc/build-green-lory-env.sh:
# conda for the interpreter, pip for the packages, gurobipy via pip with the ARC token licence.
set -euo pipefail
set +u
[ -f /etc/profile ] && source /etc/profile
[ -f /etc/profile.d/modules.sh ] && source /etc/profile.d/modules.sh
[ -f /etc/profile.d/lmod.sh ] && source /etc/profile.d/lmod.sh
command -v module >/dev/null 2>&1 || source /usr/share/lmod/lmod/init/bash
set -u
module purge
module load "${ARC_ANACONDA_MODULE:-Anaconda3/2024.06-1}"
if [ -n "${EBROOTANACONDA3:-}" ] && [ -f "$EBROOTANACONDA3/etc/profile.d/conda.sh" ]; then
  source "$EBROOTANACONDA3/etc/profile.d/conda.sh"
else
  eval "$(conda shell.bash hook)"
fi
ARC_WORK_BASE="${ARC_WORK_BASE:-/data/engs-df-green-ammonia/engs2523}"
PREFIX="${LEGACY_ENV_PREFIX:-$ARC_WORK_BASE/envs/legacy-lcoa-env}"
LOGDIR="$ARC_WORK_BASE/envs/logs"; mkdir -p "$LOGDIR"
if [[ -d "$PREFIX" ]]; then echo "ERROR: $PREFIX exists; refusing to overwrite" >&2; exit 2; fi
conda create -y -p "$PREFIX" python=3.11 pip
"$PREFIX/bin/python" -m pip install --upgrade pip wheel
"$PREFIX/bin/pip" install "pypsa==0.25.1" "pyomo==6.5.0" "linopy==0.3.8" "pandas==2.2.1" "numpy==1.26.4" \
  "xarray==2024.3.0" "netCDF4==1.6.5" "openpyxl" "geopandas==0.14.3" "shapely" "gurobipy==11.0.3" "highspy" "scipy<1.14"
"$PREFIX/bin/pip" freeze > "$LOGDIR/legacy-lcoa-env-pip-freeze.txt"
export GRB_LICENSE_FILE=/apps/system/easybuild/software/Gurobi/10.0.3-GCCcore-12.2.0/gurobi.lic
"$PREFIX/bin/python" - <<'PY'
import sys, pypsa, pyomo, xarray, netCDF4, openpyxl
print("Python:", sys.version); print("pypsa", pypsa.__version__, "pyomo", pyomo.version.version)
import gurobipy as g
m = g.Model(); print("gurobi", g.gurobi.version(), "licence OK")
PY
