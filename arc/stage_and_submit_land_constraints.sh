#!/bin/bash
set -euo pipefail

ARC_HOST="${ARC_HOST:-engs2523@arc-login.arc.ox.ac.uk}"
ARC_CONTROL_PATH="${ARC_CONTROL_PATH:-/tmp/arc-green-lory.sock}"
ARC_REPO_DIR="${ARC_REPO_DIR:-/data/engs-df-green-ammonia/engs2523/green-lory}"
ARC_CLEAN_REMOTE="${ARC_CLEAN_REMOTE:-0}"

ssh_opts=()
rsync_ssh="ssh"

ensure_control_master() {
  if ssh -o "ControlPath=${ARC_CONTROL_PATH}" -O check "$ARC_HOST" >/dev/null 2>&1; then
    ssh_opts=( -o "ControlPath=${ARC_CONTROL_PATH}" )
    rsync_ssh="ssh -o ControlPath=${ARC_CONTROL_PATH}"
    return 0
  fi

  echo "Opening SSH control connection to $ARC_HOST"
  ssh -M -o ControlMaster=yes -o ControlPersist=4h -o "ControlPath=${ARC_CONTROL_PATH}" -Nf "$ARC_HOST"
  ssh_opts=( -o "ControlPath=${ARC_CONTROL_PATH}" )
  rsync_ssh="ssh -o ControlPath=${ARC_CONTROL_PATH}"
}

ensure_control_master

remote_prep_cmd=$(cat <<EOF
cd "$ARC_REPO_DIR"
mkdir -p data arc/jobs model logs

normalize_partial() {
  local target="\$1"
  if [[ -e "\$target" ]]; then
    return 0
  fi
  local dir base
  dir="\$(dirname "\$target")"
  base="\$(basename "\$target")"
  shopt -s nullglob
  local matches=("\$dir"/."\$base".*)
  shopt -u nullglob
  if [[ \${#matches[@]} -eq 1 ]]; then
    mv "\${matches[0]}" "\$target"
  fi
}

if [[ "$ARC_CLEAN_REMOTE" == "1" ]]; then
  rm -rf data/WDPA_Feb2026_Public_shp_0 data/WDPA_Feb2026_Public_shp_1 data/WDPA_Feb2026_Public_shp_2
  rm -f data/GEBCO_2025_sub_ice.nc data/MCD12C1.A2022001.061.2023244164746.hdf
fi

rm -f \
  data/max_capacities_100pct_slope15.csv \
  data/max_capacities_100pct_allslopes.csv \
  data/max_capacities_paper_2pct_slope15.csv \
  data/max_capacities_high_50pct_slope15.csv

normalize_partial data/MCD12C1.A2022001.061.2023244164746.hdf
normalize_partial data/GEBCO_2025_sub_ice.nc
normalize_partial data/WDPA_Feb2026_Public_shp_0/WDPA_Feb2026_Public_shp-polygons.shp
normalize_partial data/WDPA_Feb2026_Public_shp_1/WDPA_Feb2026_Public_shp-polygons.shp
normalize_partial data/WDPA_Feb2026_Public_shp_2/WDPA_Feb2026_Public_shp-polygons.shp

EOF
)

ssh "${ssh_opts[@]}" "$ARC_HOST" "$remote_prep_cmd"

rsync_sources=(
  arc/submit_land_constraints_matrix.sh
  arc/jobs/00_build_land_constraints.sh
  model/data_paths.py
  model/land_processing.py
)

while IFS= read -r source; do
  if [[ -n "$source" ]]; then
    rsync_sources+=("$source")
  fi
done < <(
  ssh "${ssh_opts[@]}" "$ARC_HOST" "cd '$ARC_REPO_DIR' && for pair in \
    'data/MCD12C1.A2022001.061.2023244164746.hdf|data/MCD12C1.A2022001.061.2023244164746.hdf' \
    'data/GEBCO_2025_sub_ice.nc|data/GEBCO_2025_sub_ice.nc' \
    'data/WDPA_Feb2026_Public_shp_0|data/WDPA_Feb2026_Public_shp_0/WDPA_Feb2026_Public_shp-polygons.shp' \
    'data/WDPA_Feb2026_Public_shp_1|data/WDPA_Feb2026_Public_shp_1/WDPA_Feb2026_Public_shp-polygons.shp' \
    'data/WDPA_Feb2026_Public_shp_2|data/WDPA_Feb2026_Public_shp_2/WDPA_Feb2026_Public_shp-polygons.shp'; do \
      src=\${pair%%|*}; check=\${pair#*|}; \
      if [[ ! -f \"\$check\" ]]; then printf '%s\\n' \"\$src\"; fi; \
    done"
)

echo "Staging sources:" "${rsync_sources[@]}"

rsync -avR --partial --inplace --append --progress -e "$rsync_ssh" \
  "${rsync_sources[@]}" \
  "$ARC_HOST:$ARC_REPO_DIR/"

ssh "${ssh_opts[@]}" "$ARC_HOST" "cd '$ARC_REPO_DIR' && bash arc/submit_land_constraints_matrix.sh"