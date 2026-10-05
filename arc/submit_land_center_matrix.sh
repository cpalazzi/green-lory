#!/bin/bash
# Submit one heavy 100 % land build (MODIS + WDPA + >15 degree slope, cells anchored per
# ARC_CELL_ANCHOR) and the cheap derived land-share tables that rescale it.
#
# Run on ARC from inside a pinned release directory:
#   ARC_CAMPAIGN_DIR=/data/.../green-lory-campaigns/land_center_20260923_v1 \
#   bash arc/submit_land_center_matrix.sh
#
# Outputs (never overwritten) land in $ARC_CAMPAIGN_DIR:
#   max_capacities_<prefix>_100pct_slope15.csv
#   max_capacities_<prefix>_<share>pct_slope15.csv   for each share in ARC_DERIVED_SHARES
# Every sbatch line is appended to $ARC_CAMPAIGN_DIR/submissions.tsv.  The heavy job is
# submitted with --clusters=all; the derived jobs depend on it and therefore go to the
# cluster that accepted it (job ids and dependencies are per cluster).
set -euo pipefail

DEFAULT_REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ARC_REPO_DIR="${ARC_REPO_DIR:-$DEFAULT_REPO_DIR}"
ARC_CAMPAIGN_DIR="${ARC_CAMPAIGN_DIR:?set ARC_CAMPAIGN_DIR to the immutable output directory}"
ARC_CELL_ANCHOR="${ARC_CELL_ANCHOR:-center}"
ARC_LAND_TAG_PREFIX="${ARC_LAND_TAG_PREFIX:-$ARC_CELL_ANCHOR}"
ARC_DERIVED_SHARES="${ARC_DERIVED_SHARES:-0.02 0.20}"
ARC_INCLUDE_OFFSHORE_WIND="${ARC_INCLUDE_OFFSHORE_WIND:-1}"
ARC_MAX_SLOPE_DEGREES="${ARC_MAX_SLOPE_DEGREES:-15}"
ARC_HEAVY_MEM="${ARC_HEAVY_MEM:-128G}"
ARC_HEAVY_TIME="${ARC_HEAVY_TIME:-12:00:00}"
ARC_DERIVED_MEM="${ARC_DERIVED_MEM:-16G}"
ARC_DERIVED_TIME="${ARC_DERIVED_TIME:-00:30:00}"
DRY_RUN="${ARC_DRY_RUN:-0}"

cd "$ARC_REPO_DIR"

required=(
  data/MCD12C1.A2022001.061.2023244164746.hdf
  data/GEBCO_2025_sub_ice.nc
  data/WDPA_Feb2026_Public_shp_0/WDPA_Feb2026_Public_shp-polygons.shp
  data/WDPA_Feb2026_Public_shp_1/WDPA_Feb2026_Public_shp-polygons.shp
  data/WDPA_Feb2026_Public_shp_2/WDPA_Feb2026_Public_shp-polygons.shp
  data/model_bathymetry.nc
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

mkdir -p "$ARC_CAMPAIGN_DIR/logs"
heavy_tag="${ARC_LAND_TAG_PREFIX}_100pct_slope15"
heavy_csv="$ARC_CAMPAIGN_DIR/max_capacities_${heavy_tag}.csv"
if [[ -e "$heavy_csv" ]]; then
  echo "ERROR: refusing to overwrite immutable land output: $heavy_csv" >&2
  exit 2
fi

submissions="$ARC_CAMPAIGN_DIR/submissions.tsv"
if [[ ! -f "$submissions" ]]; then
  printf 'submitted_utc\tjob_id\tcluster\tjob_name\toutput\tsbatch_command\n' > "$submissions"
fi

submit() {
  # prints "jobid<TAB>cluster"
  local name="$1" output="$2"; shift 2
  local cmd=(sbatch --parsable "$@")
  if [[ "$DRY_RUN" == "1" ]]; then
    printf 'DRY RUN: %q ' "${cmd[@]}" >&2; printf '\n' >&2
    printf '0\tdry\n'
    return 0
  fi
  local raw
  raw=$("${cmd[@]}")
  local job_id="${raw%%;*}" cluster="${raw#*;}"
  if [[ "$cluster" == "$raw" ]]; then cluster="default"; fi
  printf '%s\t%s\t%s\t%s\t%s\t%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$job_id" "$cluster" "$name" "$output" "$(printf '%q ' "${cmd[@]}")" >> "$submissions"
  printf '%s\t%s\n' "$job_id" "$cluster"
}

base_export="ALL,ARC_REPO_DIR=${ARC_REPO_DIR},ARC_INCLUDE_OFFSHORE_WIND=${ARC_INCLUDE_OFFSHORE_WIND},ARC_LAND_LOG_DIR=${ARC_CAMPAIGN_DIR}/logs,ARC_ALLOW_LAND_OUTPUT_OVERWRITE=0,ARC_CELL_ANCHOR=${ARC_CELL_ANCHOR}"

heavy_name="glr-land-${ARC_LAND_TAG_PREFIX}-100pct"
read -r heavy_job heavy_cluster < <(submit "$heavy_name" "$heavy_csv" \
  --job-name="$heavy_name" \
  --clusters=all \
  --mem="$ARC_HEAVY_MEM" \
  --time="$ARC_HEAVY_TIME" \
  --output="${ARC_CAMPAIGN_DIR}/logs/%x-%j.out" \
  --export="${base_export},ARC_LAND_TAG=${heavy_tag},ARC_LAND_OUTPUT_CSV=${heavy_csv},ARC_LAND_COMPETITION_FRACTION=1.0,ARC_MAX_SLOPE_DEGREES=${ARC_MAX_SLOPE_DEGREES}" \
  arc/jobs/00_build_land_constraints.sh)
echo "heavy build: job=${heavy_job} cluster=${heavy_cluster} output=${heavy_csv}"

cluster_flag=()
if [[ "$heavy_cluster" != "default" && "$heavy_cluster" != "dry" ]]; then
  cluster_flag=(--clusters="$heavy_cluster")
fi

for share in $ARC_DERIVED_SHARES; do
  pct=$(awk -v s="$share" 'BEGIN { printf "%g", s * 100 }')
  tag="${ARC_LAND_TAG_PREFIX}_${pct}pct_slope15"
  csv="$ARC_CAMPAIGN_DIR/max_capacities_${tag}.csv"
  if [[ -e "$csv" ]]; then
    echo "ERROR: refusing to overwrite immutable land output: $csv" >&2
    exit 2
  fi
  name="glr-land-${ARC_LAND_TAG_PREFIX}-${pct}pct"
  read -r job cluster < <(submit "$name" "$csv" \
    --job-name="$name" \
    "${cluster_flag[@]}" \
    --dependency="afterok:${heavy_job}" \
    --mem="$ARC_DERIVED_MEM" \
    --time="$ARC_DERIVED_TIME" \
    --output="${ARC_CAMPAIGN_DIR}/logs/%x-%j.out" \
    --export="${base_export},ARC_LAND_TAG=${tag},ARC_LAND_OUTPUT_CSV=${csv},ARC_BASE_LAND_CSV=${heavy_csv},ARC_LAND_COMPETITION_FRACTION=${share},ARC_SOURCE_LAND_COMPETITION_FRACTION=1.0" \
    arc/jobs/00_build_land_constraints.sh)
  echo "derived ${pct}%: job=${job} cluster=${cluster} output=${csv} depends_on=${heavy_job}"
done
