#!/bin/bash
# One-time ARC setup helper for green-lory
set -euo pipefail

USER_NAME="${USER:-$(id -un)}"
ARC_GROUP="${ARC_GROUP:-engs-df-green-ammonia}"
ARC_WORK_BASE="${ARC_WORK_BASE:-/data/${ARC_GROUP}/${USER_NAME}}"
ARC_REPO_NAME="${ARC_REPO_NAME:-green-lory}"
ARC_REPO_URL="${ARC_REPO_URL:-https://github.com/cpalazzi/green-lory.git}"
ARC_REPO_DIR="${ARC_REPO_DIR:-${ARC_WORK_BASE}/${ARC_REPO_NAME}}"
ARC_ENV_PREFIX="${ARC_ENV_PREFIX:-${ARC_WORK_BASE}/envs/green-lory-env}"
ARC_LICENSE_DIR="${ARC_LICENSE_DIR:-${ARC_WORK_BASE}/licenses}"

printf "\n=== Green Lory ARC initial setup ===\n"
printf "User: %s\n" "$USER_NAME"
printf "Group: %s\n" "$ARC_GROUP"
printf "Work base: %s\n" "$ARC_WORK_BASE"
printf "Repo dir: %s\n" "$ARC_REPO_DIR"
printf "Env prefix: %s\n\n" "$ARC_ENV_PREFIX"

mkdir -p "$ARC_WORK_BASE" "${ARC_ENV_PREFIX%/*}/logs" "$ARC_LICENSE_DIR"

if [[ -d "$ARC_REPO_DIR/.git" ]]; then
  printf "Repository exists. Pull latest changes? [y/N] "
  read -r reply
  if [[ "$reply" =~ ^[Yy]$ ]]; then
    git -C "$ARC_REPO_DIR" fetch --all
    git -C "$ARC_REPO_DIR" pull --ff-only
  fi
else
  printf "Cloning repository from %s ...\n" "$ARC_REPO_URL"
  git clone "$ARC_REPO_URL" "$ARC_REPO_DIR"
fi

mkdir -p "$ARC_REPO_DIR/results" "$ARC_REPO_DIR/logs"

if [[ -f "$ARC_LICENSE_DIR/gurobi.lic" ]]; then
  printf "Found Gurobi license: %s\n" "$ARC_LICENSE_DIR/gurobi.lic"
else
  printf "No Gurobi license found at %s/gurobi.lic (optional if using HiGHS).\n" "$ARC_LICENSE_DIR"
fi

if command -v sbatch >/dev/null 2>&1; then
  printf "\nSubmit environment build job now? [y/N] "
  read -r submit_env
  if [[ "$submit_env" =~ ^[Yy]$ ]]; then
    job_out=$(sbatch \
      --export=ALL,ARC_GROUP="$ARC_GROUP",ARC_WORK_BASE="$ARC_WORK_BASE",ARC_REPO_DIR="$ARC_REPO_DIR",ARC_ENV_PREFIX="$ARC_ENV_PREFIX",ARC_LICENSE_DIR="$ARC_LICENSE_DIR" \
      "$ARC_REPO_DIR/arc/build-green-lory-env")
    printf "Submitted: %s\n" "$job_out"
  else
    printf "Skip env build. Run later with:\n"
    printf "  cd %s && sbatch arc/build-green-lory-env\n" "$ARC_REPO_DIR"
  fi
else
  printf "sbatch not available in this shell. Run from ARC login node.\n"
fi

printf "\nNext:\n"
printf "1) cd %s\n" "$ARC_REPO_DIR"
printf "2) source arc/load_green_lory_env.sh\n"
printf "3) bash arc/arc_check_run_inputs.sh\n"
printf "4) bash arc/submit_global_run.sh full-global-2030\n\n"
