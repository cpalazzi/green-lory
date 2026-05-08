#!/bin/bash
set -euo pipefail

usage() {
  cat >&2 <<'EOF'
Usage: bash arc/submit_constrained_reruns.sh --land-csv <path> [--include-2030]
  bash arc/submit_constrained_reruns.sh --land-csv <path> [--land-tag <tag>] [--scenario <name>]...

Submits the constrained 2050 global scenarios as 4 quadrant jobs each and
queues a dependent merge job to write the canonical scenario CSV.

Options:
  --land-csv <path>   Max-capacity CSV to use for all submissions.
  --land-tag <tag>    Optional suffix for run/output naming. If omitted, infer
                      from --land-csv when the file is named max_capacities_<tag>.csv.
  --scenario <name>   Submit only the named scenario. May be passed multiple times.
  --include-2030      Also submit DEA 2030 flat and spatial scenarios.
EOF
}

LAND_CSV=""
LAND_TAG=""
INCLUDE_2030=false
SCENARIO_FILTER=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --land-csv)
      LAND_CSV="$2"
      shift 2
      ;;
    --include-2030)
      INCLUDE_2030=true
      shift
      ;;
    --land-tag)
      LAND_TAG="$2"
      shift 2
      ;;
    --scenario)
      SCENARIO_FILTER+=("$2")
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage
      exit 2
      ;;
  esac
done

if [[ -z "$LAND_CSV" ]]; then
  echo "--land-csv is required" >&2
  usage
  exit 2
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

if [[ ! -f "$LAND_CSV" ]]; then
  echo "Land CSV not found: $LAND_CSV" >&2
  exit 2
fi

ARC_GROUP="${ARC_GROUP:-engs-df-green-ammonia}"
USER_NAME="${USER:-$(id -un)}"
ARC_WORK_BASE="${ARC_WORK_BASE:-/data/${ARC_GROUP}/${USER_NAME}}"
ARC_ENV_PREFIX="${ARC_ENV_PREFIX:-${ARC_WORK_BASE}/envs/green-lory-env}"
ARC_PYTHON="${ARC_PYTHON:-${ARC_ENV_PREFIX}/bin/python}"

if [[ ! -x "$ARC_PYTHON" ]]; then
  echo "Python not found for merge jobs: $ARC_PYTHON" >&2
  exit 2
fi

mkdir -p logs results

normalize_job_id() {
  local raw_job_id="$1"
  printf '%s\n' "${raw_job_id%%;*}"
}

infer_land_tag() {
  local stem
  stem="$(basename "${LAND_CSV%.*}")"
  if [[ "$stem" == "max_capacities" ]]; then
    return 0
  fi
  if [[ "$stem" == max_capacities_* ]]; then
    printf '%s\n' "${stem#max_capacities_}"
  fi
}

slug_for_run_label() {
  printf '%s' "$1" | tr '[:upper:]' '[:lower:]' | tr -cs 'a-z0-9' '-'
}

slug_for_path() {
  printf '%s' "$1" | tr '[:upper:]' '[:lower:]' | tr -cs 'a-z0-9' '_'
}

if [[ -z "$LAND_TAG" ]]; then
  LAND_TAG="$(infer_land_tag || true)"
fi

LAND_TAG_RUN=""
LAND_TAG_PATH=""
if [[ -n "$LAND_TAG" ]]; then
  LAND_TAG_RUN="$(slug_for_run_label "$LAND_TAG")"
  LAND_TAG_PATH="$(slug_for_path "$LAND_TAG")"
fi

qualify_run_label() {
  local base_label="$1"
  if [[ -z "$LAND_TAG_RUN" ]]; then
    printf '%s\n' "$base_label"
  else
    printf '%s-%s\n' "$base_label" "$LAND_TAG_RUN"
  fi
}

qualify_output_csv() {
  local base_output_csv="$1"
  if [[ -z "$LAND_TAG_PATH" ]]; then
    printf '%s\n' "$base_output_csv"
    return 0
  fi

  local parent_dir base_name
  parent_dir="$(dirname "$base_output_csv")"
  base_name="$(basename "$base_output_csv")"
  printf '%s_%s/%s\n' "$parent_dir" "$LAND_TAG_PATH" "$base_name"
}

should_submit_scenario() {
  local scenario_name="$1"
  local requested

  if [[ ${#SCENARIO_FILTER[@]} -eq 0 ]]; then
    return 0
  fi

  for requested in "${SCENARIO_FILTER[@]}"; do
    if [[ "$requested" == "$scenario_name" ]]; then
      return 0
    fi
  done

  return 1
}

submit_scenario() {
  local run_label="$1"
  local tech_yaml="$2"
  local plant_dir="$3"
  local finance_mode="$4"
  local output_csv="$5"
  local qualified_run_label
  local qualified_output_csv

  qualified_run_label="$(qualify_run_label "$run_label")"
  qualified_output_csv="$(qualify_output_csv "$output_csv")"

  local interest_csv=""
  if [[ "$finance_mode" == "spatial" ]]; then
    interest_csv="inputs/spatial_cost_inputs.csv"
  fi

  local -a scenario_env=(
    "ARC_TECH_YAML=$tech_yaml"
    "ARC_PLANT_DIR=$plant_dir"
    "ARC_LAND_CSV=$LAND_CSV"
    "ARC_INTEREST_CSV=$interest_csv"
    "ARC_FAIL_FAST=${ARC_FAIL_FAST:-0}"
  )

  echo
  echo "Scenario: $qualified_run_label"
  echo "  land csv: $LAND_CSV"
  if [[ -n "$LAND_TAG" ]]; then
    echo "  land tag: $LAND_TAG"
  fi
  env "${scenario_env[@]}" bash arc/arc_check_run_inputs.sh >/dev/null

  local bounds=(
    "-180 -90 west2"
    "-90 0 west1"
    "0 90 east1"
    "90 180 east2"
  )
  local -a job_ids=()
  local lo hi quadrant qlabel job_id raw_job_id
  for spec in "${bounds[@]}"; do
    read -r lo hi quadrant <<<"$spec"
    qlabel="${qualified_run_label}-${quadrant}"
    raw_job_id=$(env "${scenario_env[@]}" sbatch --parsable --export="ALL,ARC_LON_MIN=${lo},ARC_LON_MAX=${hi}" arc/jobs/01_run_global.sh "$qlabel")
    job_id=$(normalize_job_id "$raw_job_id")
    job_ids+=("$job_id")
    echo "  ${quadrant}: ${job_id}"
  done

  local deps
  deps=$(IFS=:; echo "${job_ids[*]}")
  local merge_job
  merge_job=$(sbatch --parsable \
    --dependency="afterok:${deps}" \
    --job-name="merge-${qualified_run_label}" \
    --output="logs/merge-${qualified_run_label}-%j.log" \
    --wrap="cd '$REPO_ROOT' && '$ARC_PYTHON' scripts/merge_global_results.py '$qualified_run_label' --output '$qualified_output_csv'")
  echo "  merge: ${merge_job} -> ${qualified_output_csv}"
}

if should_submit_scenario "way-2050-flat"; then
  submit_scenario \
    "way-2050-flat" \
    "inputs/tech_config_ammonia_plant_2050_way_eur.yaml" \
    "basic_ammonia_plant_2050_way" \
    "none" \
    "results/way_2050_flat/global_run_results_1h_2050.csv"
fi

if should_submit_scenario "way-2050-spatial"; then
  submit_scenario \
    "way-2050-spatial" \
    "inputs/tech_config_ammonia_plant_2050_way_eur.yaml" \
    "basic_ammonia_plant_2050_way" \
    "spatial" \
    "results/way_2050_spatial/global_run_results_1h_2050.csv"
fi

if should_submit_scenario "dea-2050-flat"; then
  submit_scenario \
    "dea-2050-flat" \
    "inputs/tech_config_ammonia_plant_2050_dea.yaml" \
    "basic_ammonia_plant_2050" \
    "none" \
    "results/dea_2050_flat/global_run_results_1h_2050.csv"
fi

if should_submit_scenario "dea-2050-spatial"; then
  submit_scenario \
    "dea-2050-spatial" \
    "inputs/tech_config_ammonia_plant_2050_dea.yaml" \
    "basic_ammonia_plant_2050" \
    "spatial" \
    "results/dea_2050_spatial/global_run_results_1h_2050.csv"
fi

if [[ "$INCLUDE_2030" == "true" ]]; then
  if should_submit_scenario "dea-2030-flat"; then
    submit_scenario \
      "dea-2030-flat" \
      "inputs/tech_config_ammonia_plant_2030_dea.yaml" \
      "basic_ammonia_plant" \
      "none" \
      "results/dea_2030_flat/global_run_results_1h_2030.csv"
  fi

  if should_submit_scenario "dea-2030-spatial"; then
    submit_scenario \
      "dea-2030-spatial" \
      "inputs/tech_config_ammonia_plant_2030_dea.yaml" \
      "basic_ammonia_plant" \
      "spatial" \
      "results/dea_2030_spatial/global_run_results_1h_2030.csv"
  fi
fi