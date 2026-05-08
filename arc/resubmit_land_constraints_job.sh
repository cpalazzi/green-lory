#!/bin/bash
set -euo pipefail

ARC_HOST="${ARC_HOST:-engs2523@arc-login.arc.ox.ac.uk}"
ARC_CONTROL_PATH="${ARC_CONTROL_PATH:-/tmp/arc-green-lory.sock}"
ARC_REPO_DIR="${ARC_REPO_DIR:-/data/engs-df-green-ammonia/engs2523/green-lory}"

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

rsync -avR -e "$rsync_ssh" \
  model/data_paths.py \
  model/land_processing.py \
  arc/jobs/00_build_land_constraints.sh \
  arc/submit_land_constraints_matrix.sh \
  "$ARC_HOST:$ARC_REPO_DIR/"

remote_verify_and_submit=$(cat <<'EOF'
cd "$ARC_REPO_DIR"

for shard in 0 1 2; do
  canonical_dir="data/WDPA_Feb2026_Public_shp_${shard}"
  legacy_dir="data/external/wdpa/WDPA_Feb2026_Public_shp_${shard}"
  mkdir -p "$canonical_dir"
  if [[ -d "$legacy_dir" && ! -f "$canonical_dir/WDPA_Feb2026_Public_shp-polygons.shp" ]]; then
    find "$legacy_dir" -maxdepth 1 -type f -exec mv -f {} "$canonical_dir"/ \;
  fi
done

if [[ -f data/external/gebco/GEBCO_2025_sub_ice.nc ]]; then
  legacy_gebco_size="$(stat -c %s data/external/gebco/GEBCO_2025_sub_ice.nc)"
  canonical_gebco_size="0"
  if [[ -f data/GEBCO_2025_sub_ice.nc ]]; then
    canonical_gebco_size="$(stat -c %s data/GEBCO_2025_sub_ice.nc)"
  fi
  if [[ ! -f data/GEBCO_2025_sub_ice.nc || "$legacy_gebco_size" -gt "$canonical_gebco_size" ]]; then
    mkdir -p data
    rm -f data/GEBCO_2025_sub_ice.nc
    mv data/external/gebco/GEBCO_2025_sub_ice.nc data/GEBCO_2025_sub_ice.nc
  fi
fi

if [[ -f data/weather_data/model_bathymetry.nc && ! -f data/model_bathymetry.nc ]]; then
  mv data/weather_data/model_bathymetry.nc data/model_bathymetry.nc
fi

rm -f data/max_capacities_full_land_constraints.csv
rm -rf data/external/wdpa
rmdir data/external/gebco 2>/dev/null || true
rmdir data/external 2>/dev/null || true

bash arc/submit_land_constraints_matrix.sh
EOF
)

ssh "${ssh_opts[@]}" "$ARC_HOST" "ARC_REPO_DIR='$ARC_REPO_DIR' bash -s" <<<"$remote_verify_and_submit"