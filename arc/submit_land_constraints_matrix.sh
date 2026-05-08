#!/bin/bash
set -euo pipefail

DEFAULT_REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ARC_REPO_DIR="${ARC_REPO_DIR:-$DEFAULT_REPO_DIR}"
ARC_INCLUDE_OFFSHORE_WIND="${ARC_INCLUDE_OFFSHORE_WIND:-1}"
ARC_BASELINE_MAX_SLOPE_DEGREES="${ARC_BASELINE_MAX_SLOPE_DEGREES:-15}"

BASELINE_SLOPE15_CSV="${ARC_BASELINE_SLOPE15_CSV:-data/max_capacities_baseline_slope15.csv}"
BASELINE_ALLSLOPES_CSV="${ARC_BASELINE_ALLSLOPES_CSV:-data/max_capacities_baseline_allslopes.csv}"
PAPER_2PCT_SLOPE15_CSV="${ARC_PAPER_2PCT_SLOPE15_CSV:-data/max_capacities_paper_2pct_slope15.csv}"
HIGH_50PCT_SLOPE15_CSV="${ARC_HIGH_50PCT_SLOPE15_CSV:-data/max_capacities_high_50pct_slope15.csv}"

cd "$ARC_REPO_DIR"

submit_job() {
  local raw_job_id
  raw_job_id=$(sbatch --parsable "$@")
  printf '%s\n' "$raw_job_id" >&2
  printf '%s\n' "${raw_job_id%%;*}"
}

required=(
  data/MCD12C1.A2022001.061.2023244164746.hdf
  data/GEBCO_2025_sub_ice.nc
  data/WDPA_Feb2026_Public_shp_0/WDPA_Feb2026_Public_shp-polygons.shp
  data/WDPA_Feb2026_Public_shp_1/WDPA_Feb2026_Public_shp-polygons.shp
  data/WDPA_Feb2026_Public_shp_2/WDPA_Feb2026_Public_shp-polygons.shp
  arc/jobs/00_build_land_constraints.sh
  model/land_processing.py
)

for path in "${required[@]}"; do
  if [[ ! -f "$path" ]]; then
    echo "ERROR: missing staged ARC input: $path" >&2
    exit 2
  fi
done

bash -n arc/jobs/00_build_land_constraints.sh

base_export="ALL,ARC_REPO_DIR=${ARC_REPO_DIR},ARC_INCLUDE_OFFSHORE_WIND=${ARC_INCLUDE_OFFSHORE_WIND}"

baseline_slope15_job=$(submit_job \
  --job-name=green-lory-land-slope15 \
  --export="${base_export},ARC_LAND_TAG=baseline_slope15,ARC_LAND_OUTPUT_CSV=${BASELINE_SLOPE15_CSV},ARC_LAND_COMPETITION_FRACTION=1.0,ARC_MAX_SLOPE_DEGREES=${ARC_BASELINE_MAX_SLOPE_DEGREES}" \
  arc/jobs/00_build_land_constraints.sh)

baseline_allslopes_job=$(submit_job \
  --job-name=green-lory-land-allslopes \
  --export="${base_export},ARC_LAND_TAG=baseline_allslopes,ARC_LAND_OUTPUT_CSV=${BASELINE_ALLSLOPES_CSV},ARC_LAND_COMPETITION_FRACTION=1.0,ARC_SKIP_SLOPE_EXCLUSION=1" \
  arc/jobs/00_build_land_constraints.sh)

paper_2pct_job=$(submit_job \
  --job-name=green-lory-land-2pct \
  --dependency="afterok:${baseline_slope15_job}" \
  --export="${base_export},ARC_LAND_TAG=paper_2pct_slope15,ARC_LAND_OUTPUT_CSV=${PAPER_2PCT_SLOPE15_CSV},ARC_BASE_LAND_CSV=${BASELINE_SLOPE15_CSV},ARC_LAND_COMPETITION_FRACTION=0.02,ARC_SOURCE_LAND_COMPETITION_FRACTION=1.0" \
  arc/jobs/00_build_land_constraints.sh)

high_50pct_job=$(submit_job \
  --job-name=green-lory-land-50pct \
  --dependency="afterok:${baseline_slope15_job}" \
  --export="${base_export},ARC_LAND_TAG=high_50pct_slope15,ARC_LAND_OUTPUT_CSV=${HIGH_50PCT_SLOPE15_CSV},ARC_BASE_LAND_CSV=${BASELINE_SLOPE15_CSV},ARC_LAND_COMPETITION_FRACTION=0.50,ARC_SOURCE_LAND_COMPETITION_FRACTION=1.0" \
  arc/jobs/00_build_land_constraints.sh)

echo "Submitted land-processing matrix:"
echo "  baseline_slope15    job=${baseline_slope15_job} output=${BASELINE_SLOPE15_CSV}"
echo "  baseline_allslopes  job=${baseline_allslopes_job} output=${BASELINE_ALLSLOPES_CSV}"
echo "  paper_2pct_slope15  job=${paper_2pct_job} output=${PAPER_2PCT_SLOPE15_CSV} depends_on=${baseline_slope15_job}"
echo "  high_50pct_slope15  job=${high_50pct_job} output=${HIGH_50PCT_SLOPE15_CSV} depends_on=${baseline_slope15_job}"