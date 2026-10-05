#!/bin/bash
set -euo pipefail

DEFAULT_REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ARC_REPO_DIR="${ARC_REPO_DIR:-$DEFAULT_REPO_DIR}"
ARC_INCLUDE_OFFSHORE_WIND="${ARC_INCLUDE_OFFSHORE_WIND:-1}"
ARC_PCT100_MAX_SLOPE_DEGREES="${ARC_PCT100_MAX_SLOPE_DEGREES:-15}"

PCT100_SLOPE15_CSV="${ARC_PCT100_SLOPE15_CSV:-data/max_capacities_100pct_slope15.csv}"
PCT100_ALLSLOPES_CSV="${ARC_PCT100_ALLSLOPES_CSV:-data/max_capacities_100pct_allslopes.csv}"
PAPER_2PCT_SLOPE15_CSV="${ARC_PAPER_2PCT_SLOPE15_CSV:-data/max_capacities_paper_2pct_slope15.csv}"
HIGH_50PCT_SLOPE15_CSV="${ARC_HIGH_50PCT_SLOPE15_CSV:-data/max_capacities_high_50pct_slope15.csv}"
LAND_LOG_DIR="${ARC_LAND_LOG_DIR:-logs}"

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

for output in \
  "$PCT100_SLOPE15_CSV" \
  "$PCT100_ALLSLOPES_CSV" \
  "$PAPER_2PCT_SLOPE15_CSV" \
  "$HIGH_50PCT_SLOPE15_CSV"; do
  if [[ -e "$output" ]]; then
    echo "ERROR: refusing to overwrite immutable land output: $output" >&2
    exit 2
  fi
done

bash -n arc/jobs/00_build_land_constraints.sh

base_export="ALL,ARC_REPO_DIR=${ARC_REPO_DIR},ARC_INCLUDE_OFFSHORE_WIND=${ARC_INCLUDE_OFFSHORE_WIND},ARC_LAND_LOG_DIR=${LAND_LOG_DIR},ARC_ALLOW_LAND_OUTPUT_OVERWRITE=0"

pct100_slope15_job=$(submit_job \
  --job-name=green-lory-land-slope15 \
  --export="${base_export},ARC_LAND_TAG=100pct_slope15,ARC_LAND_OUTPUT_CSV=${PCT100_SLOPE15_CSV},ARC_LAND_COMPETITION_FRACTION=1.0,ARC_MAX_SLOPE_DEGREES=${ARC_PCT100_MAX_SLOPE_DEGREES}" \
  arc/jobs/00_build_land_constraints.sh)

pct100_allslopes_job=$(submit_job \
  --job-name=green-lory-land-allslopes \
  --export="${base_export},ARC_LAND_TAG=100pct_allslopes,ARC_LAND_OUTPUT_CSV=${PCT100_ALLSLOPES_CSV},ARC_LAND_COMPETITION_FRACTION=1.0,ARC_SKIP_SLOPE_EXCLUSION=1" \
  arc/jobs/00_build_land_constraints.sh)

paper_2pct_job=$(submit_job \
  --job-name=green-lory-land-2pct \
  --dependency="afterok:${pct100_slope15_job}" \
  --export="${base_export},ARC_LAND_TAG=paper_2pct_slope15,ARC_LAND_OUTPUT_CSV=${PAPER_2PCT_SLOPE15_CSV},ARC_BASE_LAND_CSV=${PCT100_SLOPE15_CSV},ARC_LAND_COMPETITION_FRACTION=0.02,ARC_SOURCE_LAND_COMPETITION_FRACTION=1.0" \
  arc/jobs/00_build_land_constraints.sh)

high_50pct_job=$(submit_job \
  --job-name=green-lory-land-50pct \
  --dependency="afterok:${pct100_slope15_job}" \
  --export="${base_export},ARC_LAND_TAG=high_50pct_slope15,ARC_LAND_OUTPUT_CSV=${HIGH_50PCT_SLOPE15_CSV},ARC_BASE_LAND_CSV=${PCT100_SLOPE15_CSV},ARC_LAND_COMPETITION_FRACTION=0.50,ARC_SOURCE_LAND_COMPETITION_FRACTION=1.0" \
  arc/jobs/00_build_land_constraints.sh)

echo "Submitted land-processing matrix:"
echo "  100pct_slope15      job=${pct100_slope15_job} output=${PCT100_SLOPE15_CSV}"
echo "  100pct_allslopes    job=${pct100_allslopes_job} output=${PCT100_ALLSLOPES_CSV}"
echo "  paper_2pct_slope15  job=${paper_2pct_job} output=${PAPER_2PCT_SLOPE15_CSV} depends_on=${pct100_slope15_job}"
echo "  high_50pct_slope15  job=${high_50pct_job} output=${HIGH_50PCT_SLOPE15_CSV} depends_on=${pct100_slope15_job}"
