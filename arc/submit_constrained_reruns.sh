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
                      Closest Salmon/Verschuur replication uses way-2050-flat-amelired-4h.
  --include-2030      Also submit DEA 2030 flat and spatial scenarios.

Available scenarios:
  way-2050-flat
  way-2050-flat-amelired-4h
  way-2050-spatial-build-remote-water
  way-2050-spatial-build-remote-water-amelired-4h
  dea-2050-flat
  dea-2050-spatial-build-remote-water
  dea-2030-flat
  dea-2030-spatial-build-remote-water
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

job_id_from_parsable() {
  local raw_job_id="$1"
  printf '%s\n' "${raw_job_id%%;*}"
}

job_cluster_from_parsable() {
  local raw_job_id="$1"
  if [[ "$raw_job_id" == *";"* ]]; then
    printf '%s\n' "${raw_job_id#*;}"
  fi
}

infer_land_tag() {
  local stem
  stem="$(basename "${LAND_CSV%.*}")"
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

time_step_tag() {
  local time_step="$1"
  time_step="${time_step%0}"
  time_step="${time_step%.}"
  printf '%sh\n' "${time_step//./p}"
}

if [[ -z "$LAND_TAG" ]]; then
  LAND_TAG="$(infer_land_tag || true)"
fi

if [[ -z "$LAND_TAG" ]]; then
  echo "Could not infer land tag from $LAND_CSV. Use a max_capacities_<land_case>.csv filename or pass --land-tag." >&2
  exit 2
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

AVAILABLE_SCENARIOS=(
  "way-2050-flat"
  "way-2050-flat-amelired-4h"
  "way-2050-spatial-build-remote-water"
  "way-2050-spatial-build-remote-water-amelired-4h"
  "dea-2050-flat"
  "dea-2050-spatial-build-remote-water"
  "dea-2030-flat"
  "dea-2030-spatial-build-remote-water"
)

validate_scenario_filters() {
  local requested known match

  for requested in "${SCENARIO_FILTER[@]}"; do
    match=false
    for known in "${AVAILABLE_SCENARIOS[@]}"; do
      if [[ "$requested" == "$known" ]]; then
        match=true
        break
      fi
    done
    if [[ "$match" != "true" ]]; then
      echo "Unknown scenario: $requested" >&2
      echo "Available scenarios:" >&2
      printf '  %s\n' "${AVAILABLE_SCENARIOS[@]}" >&2
      exit 2
    fi
  done
}

submit_scenario() {
  local run_label="$1"
  local tech_yaml="$2"
  local plant_dir="$3"
  local override_csv="$4"
  local output_csv="$5"
  local time_step="${6:-1.0}"
  local qualified_run_label
  local qualified_output_csv

  qualified_run_label="$(qualify_run_label "$run_label")"
  qualified_output_csv="$(qualify_output_csv "$output_csv")"

  local -a scenario_env=(
    "ARC_TECH_YAML=$tech_yaml"
    "ARC_PLANT_DIR=$plant_dir"
    "ARC_LAND_CSV=$LAND_CSV"
    "ARC_OVERRIDE_CSV=$override_csv"
    "ARC_TIME_STEP=$time_step"
    "ARC_FAIL_FAST=${ARC_FAIL_FAST:-0}"
  )

  echo
  echo "Scenario: $qualified_run_label"
  echo "  land csv: $LAND_CSV"
  echo "  time step: $(time_step_tag "$time_step")"
  if [[ -n "$LAND_TAG" ]]; then
    echo "  land tag: $LAND_TAG"
  fi
  if [[ -n "$override_csv" ]]; then
    echo "  override csv: $override_csv"
  fi
  env "${scenario_env[@]}" bash arc/arc_check_run_inputs.sh >/dev/null

  local bounds=(
    "-180 -90 west2"
    "-90 0 west1"
    "0 90 east1"
    "90 180 east2"
  )
  local -a job_ids=()
  local merge_cluster=""
  local lo hi quadrant qlabel job_id job_cluster raw_job_id
  for spec in "${bounds[@]}"; do
    read -r lo hi quadrant <<<"$spec"
    qlabel="${qualified_run_label}-${quadrant}"
    raw_job_id=$(env "${scenario_env[@]}" sbatch --parsable --export="ALL,ARC_LON_MIN=${lo},ARC_LON_MAX=${hi}" arc/jobs/01_run_global.sh "$qlabel")
    job_id=$(job_id_from_parsable "$raw_job_id")
    job_cluster=$(job_cluster_from_parsable "$raw_job_id")
    if [[ -n "$job_cluster" ]]; then
      if [[ -z "$merge_cluster" ]]; then
        merge_cluster="$job_cluster"
      elif [[ "$merge_cluster" != "$job_cluster" ]]; then
        echo "Shard jobs for ${qualified_run_label} landed on multiple clusters: ${merge_cluster} and ${job_cluster}" >&2
        exit 2
      fi
    fi
    job_ids+=("$job_id")
    echo "  ${quadrant}: ${job_id}${job_cluster:+;${job_cluster}}"
  done

  local deps
  deps=$(IFS=:; echo "${job_ids[*]}")
  local merge_job
  local -a merge_sbatch_args=(
    --parsable
    --partition=short
    --time=01:00:00
    --dependency="afterok:${deps}"
    --job-name="merge-${qualified_run_label}"
    --output="logs/merge-${qualified_run_label}-%j.log"
    --wrap="cd '$REPO_ROOT' && '$ARC_PYTHON' scripts/merge_global_results.py '$qualified_run_label' --output '$qualified_output_csv'"
  )
  if [[ -n "$merge_cluster" ]]; then
    merge_sbatch_args=(--clusters="$merge_cluster" "${merge_sbatch_args[@]}")
  fi
  merge_job=$(sbatch "${merge_sbatch_args[@]}")
  echo "  merge: ${merge_job} -> ${qualified_output_csv}"
}

validate_scenario_filters

if should_submit_scenario "way-2050-flat"; then
  submit_scenario \
    "way-2050-flat" \
    "inputs/tech_config_ammonia_plant_2050_way_eur.yaml" \
    "basic_ammonia_plant_2050_way" \
    "" \
    "results/way_2050_flat/global_run_results_1h_2050.csv" \
    "1.0"
fi

if should_submit_scenario "way-2050-flat-amelired-4h"; then
  submit_scenario \
    "way-2050-flat-amelired-4h" \
    "inputs/tech_config_ammonia_plant_2050_way_eur.yaml" \
    "basic_ammonia_plant_2050_way" \
    "inputs/amelired_interest_inputs_2050.csv" \
    "results/way_2050_flat_amelired_4h/global_run_results_4h_2050.csv" \
    "4.0"
fi

if should_submit_scenario "way-2050-spatial-build-remote-water"; then
  submit_scenario \
    "way-2050-spatial-build-remote-water" \
    "inputs/tech_config_ammonia_plant_2050_way_eur.yaml" \
    "basic_ammonia_plant_2050_way" \
    "inputs/spatial_cost_inputs.csv" \
    "results/way_2050_spatial_build_remote_water/global_run_results_1h_2050.csv" \
    "1.0"
fi

if should_submit_scenario "way-2050-spatial-build-remote-water-amelired-4h"; then
  submit_scenario \
    "way-2050-spatial-build-remote-water-amelired-4h" \
    "inputs/tech_config_ammonia_plant_2050_way_eur.yaml" \
    "basic_ammonia_plant_2050_way" \
    "inputs/spatial_cost_inputs_amelired_2050.csv" \
    "results/way_2050_spatial_build_remote_water_amelired_4h/global_run_results_4h_2050.csv" \
    "4.0"
fi

if should_submit_scenario "dea-2050-flat"; then
  submit_scenario \
    "dea-2050-flat" \
    "inputs/tech_config_ammonia_plant_2050_dea.yaml" \
    "basic_ammonia_plant_2050" \
    "" \
    "results/dea_2050_flat/global_run_results_1h_2050.csv" \
    "1.0"
fi

if should_submit_scenario "dea-2050-spatial-build-remote-water"; then
  submit_scenario \
    "dea-2050-spatial-build-remote-water" \
    "inputs/tech_config_ammonia_plant_2050_dea.yaml" \
    "basic_ammonia_plant_2050" \
    "inputs/spatial_cost_inputs.csv" \
    "results/dea_2050_spatial_build_remote_water/global_run_results_1h_2050.csv" \
    "1.0"
fi

if [[ "$INCLUDE_2030" == "true" ]]; then
  if should_submit_scenario "dea-2030-flat"; then
    submit_scenario \
      "dea-2030-flat" \
      "inputs/tech_config_ammonia_plant_2030_dea.yaml" \
      "basic_ammonia_plant" \
      "" \
      "results/dea_2030_flat/global_run_results_1h_2030.csv" \
      "1.0"
  fi

  if should_submit_scenario "dea-2030-spatial-build-remote-water"; then
    submit_scenario \
      "dea-2030-spatial-build-remote-water" \
      "inputs/tech_config_ammonia_plant_2030_dea.yaml" \
      "basic_ammonia_plant" \
      "inputs/spatial_cost_inputs.csv" \
      "results/dea_2030_spatial_build_remote_water/global_run_results_1h_2030.csv" \
      "1.0"
  fi
fi