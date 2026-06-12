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
bash arc/submit_land_constraints_matrix.sh
EOF
)

ssh "${ssh_opts[@]}" "$ARC_HOST" "ARC_REPO_DIR='$ARC_REPO_DIR' bash -s" <<<"$remote_verify_and_submit"