#!/bin/bash
# Submit green-lory surface runs (run ids of runs.yaml) in smoke, diagnostic or global stages,
# each into a new run directory: shard folders are never reused, run directories never
# overwritten. On ARC, every check and the run manifests run in a short prepare job; the login
# node only calls sbatch.
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  bash arc/submit_lory_sequence.sh --stage smoke|diagnostic|global [options]

Required:
  --stage <stage>       smoke: 3 cells / 168 simulated hours by default
                        diagnostic: 3 cells / full year
                        global: 4 longitude shards / full year
  --campaign <id>       Campaign ID.
  --scenario <id>       Run id from runs.yaml (runner: green_lory); repeatable.
  --land-csv <path>     Centred land table; its sha256 must match land_table_sha256
                        of the run in runs.yaml.

Options:
  --run-id <id>         Immutable run ID. Default: UTC timestamp + source commit.
  --results-root <path> Campaign result root. Default: results/campaigns/<campaign>.
  --cluster <name>      Pin all jobs for a scenario to one ARC cluster.
  --smoke-hours <n>     Simulated duration for smoke jobs (default: 168 hours).
                        This is 168 snapshots at 1h and 42 snapshots at 4h.
  --global-locations <csv>
                        Restrict the global stage to the lat,lon cells listed
                        (e.g. onshore cells with suitable land); shards still
                        split by longitude.
  --diagnostic-locations <csv>
                        Cells for the smoke and diagnostic stages (default:
                        inputs/cells/diagnostic_cells.csv).
  --dry-run             Check the inputs and print the plan; submit nothing.
  -h, --help            Show this help.

Where it runs:
  ARC login node   Every check and the run manifests run in a prepare job (2 CPUs,
                   16 GB, up to 30 minutes; this command waits for it). Its log is
                   <results root>/logs/prepare-<run id>-<job id>.out. The login node
                   itself only calls sbatch, for the shard and merge/QA jobs, and
                   writes submission.json. With --dry-run the prepare job checks the
                   inputs and prints the plan; only its log is written.
  Mac              --dry-run only: the checks run here with .venv/bin/python
                   (or $PYTHON).

Runs:
  Every setting of a run is defined in runs.yaml; list them with
  python3 -m model.cli list (or table for a side-by-side view).

This wrapper never overwrites an existing run directory and never discovers
shards with a wildcard. Every QA/merge job receives its exact input paths.
EOF
}

ORIGINAL_ARGS=("$@")
CAMPAIGN_ID=""
RUN_STAGE=""
RUN_ID=""
RESULTS_ROOT=""
CAMPAIGN_LAND_CSV="${ARC_CAMPAIGN_LAND_CSV:-}"
REQUESTED_CLUSTER="${ARC_CLUSTER:-}"
SMOKE_HOURS="${ARC_SMOKE_HOURS:-168}"
GLOBAL_LOCATIONS_CSV="${ARC_GLOBAL_LOCATIONS_CSV:-}"
DIAGNOSTIC_LOCATIONS_OVERRIDE="${ARC_DIAGNOSTIC_LOCATIONS:-}"
RELEASE_SOURCE_INVENTORY="${ARC_RELEASE_INVENTORY:-}"
MAIL_USER="${ARC_MAIL_USER:-carlo.palazzi@eng.ox.ac.uk}"
MAIL_TYPE="BEGIN,END,FAIL"
DRY_RUN=false
REQUESTED_SCENARIOS=()
PHASE=""      # internal: "prepare" inside the prepare job
PLAN_FILE=""  # internal: where the prepare job lists the run directories it prepared

while [[ $# -gt 0 ]]; do
  case "$1" in
    --campaign)
      CAMPAIGN_ID="$2"
      shift 2
      ;;
    --stage)
      RUN_STAGE="$2"
      shift 2
      ;;
    --scenario)
      REQUESTED_SCENARIOS+=("$2")
      shift 2
      ;;
    --run-id)
      RUN_ID="$2"
      shift 2
      ;;
    --results-root)
      RESULTS_ROOT="$2"
      shift 2
      ;;
    --land-csv)
      CAMPAIGN_LAND_CSV="$2"
      shift 2
      ;;
    --cluster)
      REQUESTED_CLUSTER="$2"
      shift 2
      ;;
    --smoke-hours)
      SMOKE_HOURS="$2"
      shift 2
      ;;
    --global-locations)
      GLOBAL_LOCATIONS_CSV="$2"
      shift 2
      ;;
    --diagnostic-locations)
      DIAGNOSTIC_LOCATIONS_OVERRIDE="$2"
      shift 2
      ;;
    --dry-run)
      DRY_RUN=true
      shift
      ;;
    --phase)
      PHASE="$2"
      shift 2
      ;;
    --plan-file)
      PLAN_FILE="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ "$RUN_STAGE" != "smoke" && "$RUN_STAGE" != "diagnostic" && "$RUN_STAGE" != "global" ]]; then
  echo "--stage must be smoke, diagnostic, or global" >&2
  exit 2
fi
if [[ -z "$CAMPAIGN_ID" || -z "$CAMPAIGN_LAND_CSV" || ${#REQUESTED_SCENARIOS[@]} -eq 0 ]]; then
  echo "--campaign, --land-csv and at least one --scenario are required" >&2
  exit 2
fi
if [[ ! "$CAMPAIGN_ID" =~ ^[a-zA-Z0-9._-]+$ ]]; then
  echo "Campaign ID may contain only letters, numbers, dot, underscore, and hyphen" >&2
  exit 2
fi
if [[ ! "$SMOKE_HOURS" =~ ^[1-9][0-9]*$ ]]; then
  echo "--smoke-hours must be a positive integer" >&2
  exit 2
fi

# Where this runs decides what it may do:
#   login    an ARC login node: the prepare job does the checks; here only sbatch
#   prepare  inside the prepare job: checks, run directories, manifests
#   local    anywhere without sbatch (the Mac): --dry-run only
if [[ "$PHASE" == "prepare" ]]; then
  if [[ -z "${SLURM_JOB_ID:-}" ]]; then
    echo "--phase prepare runs only inside the prepare job" >&2
    exit 2
  fi
  REPO_ROOT="${ARC_REPO_DIR:?the prepare job needs ARC_REPO_DIR}"
elif [[ -n "$PHASE" ]]; then
  echo "Unknown --phase: $PHASE" >&2
  exit 2
else
  REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
  if command -v sbatch >/dev/null 2>&1 && [[ -z "${SLURM_JOB_ID:-}" ]]; then
    PHASE="login"
  else
    PHASE="local"
    if ! $DRY_RUN; then
      echo "sbatch is unavailable here, or this is already a Slurm job: use --dry-run, or submit from an ARC login node" >&2
      exit 2
    fi
  fi
fi
cd "$REPO_ROOT"

USER_NAME="${USER:-$(id -un)}"
ARC_GROUP="${ARC_GROUP:-engs-df-green-ammonia}"
ARC_WORK_BASE="${ARC_WORK_BASE:-/data/${ARC_GROUP}/${USER_NAME}}"
ARC_ENV_PREFIX="${ARC_ENV_PREFIX:-${ARC_WORK_BASE}/envs/green-lory-env}"
ARC_PYTHON="${ARC_PYTHON:-${ARC_ENV_PREFIX}/bin/python}"
WEATHER_DIR="${ARC_WEATHER_DIR:-data/weather_data/salmon2022_pv_wind_2019_hourly_1deg}"
DIAGNOSTIC_LOCATIONS="${DIAGNOSTIC_LOCATIONS_OVERRIDE:-inputs/cells/diagnostic_cells.csv}"

# Python for the checks: never on the login node
VALIDATION_PYTHON=""
if [[ "$PHASE" == "prepare" ]]; then
  VALIDATION_PYTHON="$ARC_PYTHON"
  if [[ ! -x "$VALIDATION_PYTHON" ]]; then
    echo "ARC Python not found or not executable: $VALIDATION_PYTHON" >&2
    exit 2
  fi
elif [[ "$PHASE" == "local" ]]; then
  VALIDATION_PYTHON="${PYTHON:-}"
  if [[ -z "$VALIDATION_PYTHON" && -x "$REPO_ROOT/.venv/bin/python" ]]; then
    VALIDATION_PYTHON="$REPO_ROOT/.venv/bin/python"
  fi
  VALIDATION_PYTHON="${VALIDATION_PYTHON:-$(command -v python3 || true)}"
  if [[ -z "$VALIDATION_PYTHON" ]]; then
    echo "Python is required for the checks; set PYTHON" >&2
    exit 2
  fi
fi

SOURCE_COMMIT="${ARC_SOURCE_COMMIT:-}"
if [[ -z "$SOURCE_COMMIT" ]] && git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  SOURCE_COMMIT="$(git rev-parse --short=12 HEAD)"
fi
SOURCE_COMMIT="${SOURCE_COMMIT:-unknown}"
if [[ -z "$RUN_ID" ]]; then
  RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)-${SOURCE_COMMIT}"
fi
if [[ ! "$RUN_ID" =~ ^[a-zA-Z0-9._-]+$ ]]; then
  echo "Run ID may contain only letters, numbers, dot, underscore, and hyphen" >&2
  exit 2
fi
RESULTS_ROOT="${RESULTS_ROOT:-results/campaigns/${CAMPAIGN_ID}}"

for i in "${!REQUESTED_SCENARIOS[@]}"; do
  for j in "${!REQUESTED_SCENARIOS[@]}"; do
    if (( j >= i )); then
      continue
    fi
    if [[ "${REQUESTED_SCENARIOS[$i]}" == "${REQUESTED_SCENARIOS[$j]}" ]]; then
      echo "Scenario was requested more than once: ${REQUESTED_SCENARIOS[$i]}" >&2
      exit 2
    fi
  done
done

print_command() {
  printf '  '
  printf '%q ' "$@"
  printf '\n'
}

job_id_from_parsable() {
  printf '%s\n' "${1%%;*}"
}

job_cluster_from_parsable() {
  if [[ "$1" == *";"* ]]; then
    printf '%s\n' "${1#*;}"
  fi
}

render_shell_command() {
  local rendered=""
  local item quoted
  for item in "$@"; do
    printf -v quoted '%q' "$item"
    rendered+="${quoted} "
  done
  printf '%s\n' "${rendered% }"
}

print_campaign_header() {
  echo "Campaign:     $CAMPAIGN_ID"
  echo "Run ID:       $RUN_ID"
  echo "Stage:        $RUN_STAGE"
  echo "Results root: $RESULTS_ROOT"
  echo "Source:       $SOURCE_COMMIT (diff/tree marker: $SOURCE_DIFF_SHA256)"
  if [[ -n "$REQUESTED_CLUSTER" ]]; then
    echo "Cluster:      $REQUESTED_CLUSTER"
  else
    echo "Cluster:      first submitted shard selects; remaining jobs are pinned to it"
  fi
}

print_scenario_summary() {
  echo
  echo "Scenario: $SCENARIO_ID"
  echo "  class:       $SCENARIO_CLASS"
  echo "  tech yaml:   $TECH_YAML"
  echo "  plant dir:   $PLANT_DIR"
  echo "  finance:     ${FINANCE_CSV:-interest rate $INTEREST_RATE for every technology and cell}"
  echo "  site costs table: ${SITE_COSTS_CSV:-<none: flat build costs and water>}"
  echo "  weather land correction: ${WEATHER_LAND_CORRECTION:-<none: 1-degree profiles as they are>}"
  echo "  land csv:    $LAND_CSV"
  echo "  timestep:    ${TIME_STEP}h"
  echo "  land:        $LAND_CONSTRAINT / $CAPACITY_RULE / $LAND_ALLOCATION"
  echo "  temporal:    $TEMPORAL_ACCOUNTING_MODE / $RAMP_LIMIT_BASIS"
  echo "  site costs:  $INCLUDE_SITE_COSTS"
  echo "  locations:   ${LOCATIONS_CSV:-<all positive-capacity land cells>}"
  echo "  snapshots:   ${MAX_SNAPSHOTS:-<full year>}"
  echo "  simulated hours: $SIMULATED_HOURS"
  echo "  run dir:     $RUN_DIR"
}

# Variables the login node needs to submit a prepared run (written by the prepare job)
SUBMIT_ENV_VARIABLES=(
  CAMPAIGN_ID RUN_ID RUN_STAGE SOURCE_COMMIT SOURCE_DIFF_SHA256
  SCENARIO_ID SCENARIO_CLASS SCENARIO_DESCRIPTION SCENARIO_JOB_TOKEN
  RUN_DIR MANIFEST_PATH MANIFEST_SHA256
  TECH_YAML PLANT_DIR PV_POLICY FINANCE_CSV INTEREST_RATE SITE_COSTS_CSV LAND_CSV WEATHER_DIR WEATHER_LAND_CORRECTION LOCATIONS_CSV
  TIME_STEP MAX_SNAPSHOTS SIMULATED_HOURS LAND_CONSTRAINT CAPACITY_RULE LAND_ALLOCATION
  TEMPORAL_ACCOUNTING_MODE RAMP_LIMIT_BASIS INCLUDE_SITE_COSTS
  NUM_WORKERS THREADS_PER_WORKER SLURM_CPUS SLURM_MEMORY
)

write_submit_env() {
  local path="$1" name
  {
    echo "# Written by the prepare job of arc/submit_lory_sequence.sh; read by its login-node step."
    for name in "${SUBMIT_ENV_VARIABLES[@]}"; do
      printf '%s=%q\n' "$name" "${!name}"
    done
  } > "$path"
}

# --- run in the prepare job, or locally for a dry run -------------------------------------------

compute_source_diff_sha256() {
  "$VALIDATION_PYTHON" - "$REPO_ROOT" <<'PY'
import hashlib
from pathlib import Path
import subprocess
import sys

root = Path(sys.argv[1])
digest = hashlib.sha256()
diff = subprocess.run(
    ["git", "diff", "--binary", "HEAD"],
    cwd=root,
    check=True,
    stdout=subprocess.PIPE,
).stdout
digest.update(b"TRACKED-DIFF\0")
digest.update(diff)
untracked = subprocess.run(
    ["git", "ls-files", "--others", "--exclude-standard", "-z"],
    cwd=root,
    check=True,
    stdout=subprocess.PIPE,
).stdout.split(b"\0")
for raw_path in sorted(path for path in untracked if path):
    path = root / raw_path.decode("utf-8", errors="surrogateescape")
    digest.update(b"UNTRACKED\0")
    digest.update(raw_path)
    digest.update(b"\0")
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
print(digest.hexdigest())
PY
}

configure_scenario() {
  SCENARIO_ID="$1"
  # Every per-run setting comes from runs.yaml (see model/runs.py).
  local definition
  if ! definition="$("$VALIDATION_PYTHON" -m model.cli shell "$SCENARIO_ID")"; then
    echo "Run 'python3 -m model.cli list' to see the runs defined in runs.yaml." >&2
    exit 2
  fi
  eval "$definition"
  LAND_CSV="$CAMPAIGN_LAND_CSV"

  if [[ "$RUN_STAGE" == "smoke" ]]; then
    PATH_CLASS="00_smoke"
    LOCATIONS_CSV="$DIAGNOSTIC_LOCATIONS"
    timestep_integer="${TIME_STEP%.*}"
    if (( SMOKE_HOURS % timestep_integer != 0 )); then
      echo "Smoke hours ($SMOKE_HOURS) must be divisible by timestep ($timestep_integer)" >&2
      exit 2
    fi
    MAX_SNAPSHOTS="$((SMOKE_HOURS / timestep_integer))"
    SIMULATED_HOURS="$SMOKE_HOURS"
    EXPECTED_FULL_YEAR="0"
    NUM_WORKERS="${ARC_CAMPAIGN_DIAGNOSTIC_WORKERS:-3}"
  elif [[ "$RUN_STAGE" == "diagnostic" ]]; then
    PATH_CLASS="$SCENARIO_CLASS"
    LOCATIONS_CSV="$DIAGNOSTIC_LOCATIONS"
    MAX_SNAPSHOTS=""
    SIMULATED_HOURS="8760"
    EXPECTED_FULL_YEAR="1"
    NUM_WORKERS="${ARC_CAMPAIGN_DIAGNOSTIC_WORKERS:-3}"
  else
    PATH_CLASS="$SCENARIO_CLASS"
    LOCATIONS_CSV="$GLOBAL_LOCATIONS_CSV"
    MAX_SNAPSHOTS=""
    SIMULATED_HOURS="8760"
    EXPECTED_FULL_YEAR="1"
    NUM_WORKERS="${ARC_CAMPAIGN_GLOBAL_WORKERS:-12}"
  fi
  THREADS_PER_WORKER="${ARC_CAMPAIGN_THREADS_PER_WORKER:-4}"
  SLURM_CPUS="$((NUM_WORKERS * THREADS_PER_WORKER))"
  if [[ "$RUN_STAGE" == "global" ]]; then
    SLURM_MEMORY="${ARC_CAMPAIGN_MEMORY:-370G}"
  else
    SLURM_MEMORY="${ARC_CAMPAIGN_MEMORY:-64G}"
  fi
  RUN_DIR="${RESULTS_ROOT}/${PATH_CLASS}/${SCENARIO_ID}/runs/${RUN_ID}/${RUN_STAGE}"
  MANIFEST_PATH="${RUN_DIR}/manifest.json"
}

validate_plant_dir() {
  local plant_dir="$1"
  local required=(network.csv buses.csv generators.csv links.csv loads.csv stores.csv)
  local file
  if [[ ! -d "$plant_dir" ]]; then
    echo "Plant directory not found: $plant_dir" >&2
    return 2
  fi
  for file in "${required[@]}"; do
    if [[ ! -f "$plant_dir/$file" ]]; then
      echo "Plant bundle is missing $plant_dir/$file" >&2
      return 2
    fi
  done

  "$VALIDATION_PYTHON" - "$plant_dir/generators.csv" "$PV_POLICY" <<'PY'
import csv
import sys

path, pv_policy = sys.argv[1], sys.argv[2]
with open(path, newline="", encoding="utf-8") as handle:
    rows = {row["name"]: row for row in csv.DictReader(handle)}
tracking = rows.get("solar_tracking")
if tracking is None:
    raise SystemExit(f"solar_tracking generator missing from {path}")
# The bundle allows tracking PV; the run's pv_policy switches it off at run time
if tracking.get("p_nom_extendable", "").strip().lower() not in {"1", "true", "yes"}:
    raise SystemExit(f"solar_tracking is not extendable in {path}; pv_policy {pv_policy} cannot apply")
PY
}

validate_explicit_union_area() {
  local land_csv="$1"
  if [[ ! -f "$land_csv" ]]; then
    return 0
  fi
  "$VALIDATION_PYTHON" -m arc.qa land "$land_csv" \
    --expected-land-fraction "$EXPECTED_LAND_COMPETITION_FRACTION"
}

validate_pinned_sha256() {
  local label="$1" path="$2" expected="$3"
  if [[ -z "$expected" || ! -f "$path" ]]; then
    return 0
  fi
  local actual
  actual="$("$VALIDATION_PYTHON" -c 'import hashlib,sys; print(hashlib.sha256(open(sys.argv[1], "rb").read()).hexdigest())' "$path")"
  if [[ "$actual" != "$expected" ]]; then
    echo "$label $path has sha256 $actual; runs.yaml pins $expected for $SCENARIO_ID" >&2
    return 2
  fi
}

validate_release_source_inventory() {
  if [[ -z "$RELEASE_SOURCE_INVENTORY" ]]; then
    if $DRY_RUN; then
      echo "DRY-RUN NOTE: ARC_RELEASE_INVENTORY is not set"
      return 0
    fi
    echo "Scientific campaign submission requires ARC_RELEASE_INVENTORY" >&2
    return 2
  fi
  if [[ ! -f "$RELEASE_SOURCE_INVENTORY" ]]; then
    echo "Release source inventory not found: $RELEASE_SOURCE_INVENTORY" >&2
    return 2
  fi
  "$VALIDATION_PYTHON" -m arc.release verify \
    --root "$REPO_ROOT" \
    --inventory "$RELEASE_SOURCE_INVENTORY"
}

validate_cost_scope_and_coverage() {
  "$VALIDATION_PYTHON" - \
    "$TECH_YAML" \
    "$FINANCE_CSV" \
    "$LAND_CSV" \
    "$LOCATIONS_CSV" \
    "$INCLUDE_SITE_COSTS" \
    "$DRY_RUN" \
    "$COST_SCOPE" \
    "$FLAT_WATER_MODEL_PER_M3" \
    "$SITE_COSTS_CSV" \
    "$INTEREST_RATE" <<'PY'
from pathlib import Path
import sys

from arc.manifest import (
    build_cost_scope,
    validate_coordinate_coverage,
    validate_finance_coverage,
)

tech_yaml = Path(sys.argv[1])
finance_csv = Path(sys.argv[2]) if sys.argv[2] else None
land_csv = Path(sys.argv[3])
locations_csv = Path(sys.argv[4]) if sys.argv[4] else None
include_site_costs = sys.argv[5] == "1"
dry_run = sys.argv[6].lower() == "true"

cost_scope_id = sys.argv[7]
flat_water_model_currency_per_m3 = float(sys.argv[8])
site_costs_csv = Path(sys.argv[9]) if sys.argv[9] else None
interest_rate = float(sys.argv[10]) if sys.argv[10] else None
build_cost_scope(
    tech_yaml=tech_yaml,
    finance_csv=finance_csv,
    interest_rate=interest_rate,
    include_site_costs=include_site_costs,
    cost_scope_id=cost_scope_id,
    flat_water_model_currency_per_m3=flat_water_model_currency_per_m3,
    site_costs_csv=site_costs_csv,
)
if locations_csv is not None or land_csv.is_file():
    if finance_csv is not None:
        coverage = validate_finance_coverage(
            finance_csv=finance_csv, tech_yaml=tech_yaml, land_csv=land_csv, locations_csv=locations_csv,
        )
        print(f"Finance coverage OK ({coverage['key']} table): {coverage['expected_coordinate_count']} cells")
    if site_costs_csv is not None:
        coverage = validate_coordinate_coverage(
            table_csv=site_costs_csv, land_csv=land_csv, locations_csv=locations_csv,
        )
        print(
            "Site-cost coverage OK: "
            f"{coverage['expected_coordinate_count']} expected coordinates, "
            f"{coverage['table_coordinate_count']} table coordinates"
        )
elif not dry_run:
    raise SystemExit(f"Land CSV is required for global coverage checks: {land_csv}")
PY
}

validate_scenario_sources() {
  local required=("$TECH_YAML" "$DIAGNOSTIC_LOCATIONS")
  if [[ -n "$FINANCE_CSV" && -n "$INTEREST_RATE" ]] || [[ -z "$FINANCE_CSV" && -z "$INTEREST_RATE" ]]; then
    echo "A run sets either a finance table (finance_csv) or one interest rate (interest_rate)" >&2
    return 2
  fi
  if [[ -n "$FINANCE_CSV" ]]; then
    required+=("$FINANCE_CSV")
  fi
  if [[ -n "$LOCATIONS_CSV" ]]; then
    required+=("$LOCATIONS_CSV")
  fi
  local file
  for file in "${required[@]}"; do
    if [[ ! -f "$file" ]]; then
      echo "Required scenario source not found: $file" >&2
      return 2
    fi
  done
  validate_plant_dir "$PLANT_DIR"
  validate_pinned_sha256 "Land table" "$LAND_CSV" "$EXPECTED_LAND_TABLE_SHA256"
  if [[ -n "$FINANCE_CSV" ]]; then
    validate_pinned_sha256 "Finance CSV" "$FINANCE_CSV" "$EXPECTED_FINANCE_CSV_SHA256"
  fi
  if [[ -n "$SITE_COSTS_CSV" ]]; then
    [[ -f "$SITE_COSTS_CSV" ]] || { echo "Required site-cost table not found: $SITE_COSTS_CSV" >&2; return 2; }
    validate_pinned_sha256 "Site-cost table" "$SITE_COSTS_CSV" "$EXPECTED_SITE_COSTS_CSV_SHA256"
  fi
  if [[ -n "$WEATHER_LAND_CORRECTION" ]]; then
    if [[ -f "$WEATHER_LAND_CORRECTION" ]]; then
      validate_pinned_sha256 "Weather land correction" "$WEATHER_LAND_CORRECTION" "$EXPECTED_WEATHER_LAND_CORRECTION_SHA256"
    elif $DRY_RUN; then
      echo "DRY-RUN NOTE: ARC-only weather land correction is not present locally: $WEATHER_LAND_CORRECTION"
    else
      echo "Required weather land correction not found: $WEATHER_LAND_CORRECTION" >&2; return 2
    fi
  fi
  validate_release_source_inventory
  validate_explicit_union_area "$LAND_CSV"
  validate_cost_scope_and_coverage

  if $DRY_RUN; then
    [[ -f "$LAND_CSV" ]] || echo "DRY-RUN NOTE: ARC-only land input is not present locally: $LAND_CSV"
    [[ -d "$WEATHER_DIR" ]] || echo "DRY-RUN NOTE: ARC-only weather directory is not present locally: $WEATHER_DIR"
    [[ -f data/countries.geojson ]] || echo "DRY-RUN NOTE: ARC-only country input is not present locally: data/countries.geojson"
    [[ -f data/model_bathymetry.nc ]] || echo "DRY-RUN NOTE: ARC-only bathymetry input is not present locally: data/model_bathymetry.nc"
    return 0
  fi

  env \
    ARC_PYTHON_BIN="$ARC_PYTHON" \
    ARC_TECH_YAML="$TECH_YAML" \
    ARC_FINANCE_CSV="$FINANCE_CSV" \
    ARC_LAND_CSV="$LAND_CSV" \
    ARC_WEATHER_DIR="$WEATHER_DIR" \
    ARC_LOCATIONS_CSV="$LOCATIONS_CSV" \
    bash arc/arc_check_run_inputs.sh "$LOCATIONS_CSV" >/dev/null
}

manifest_command() {
  local manifest_python="$1"
  shift
  MANIFEST_COMMAND=(
    "$manifest_python" -m arc.manifest write
    --output "$MANIFEST_PATH"
    --campaign-id "$CAMPAIGN_ID"
    --run-id "$RUN_ID"
    --scenario-id "$SCENARIO_ID"
    --scenario-class "$SCENARIO_CLASS"
    --stage "$RUN_STAGE"
    --description "$SCENARIO_DESCRIPTION"
    --run-dir "$RUN_DIR"
    --tech-yaml "$TECH_YAML"
    --plant-dir "$PLANT_DIR"
    --land-csv "$LAND_CSV"
    --weather-dir "$WEATHER_DIR"
    --time-step-hours "$TIME_STEP"
    --simulated-hours "$SIMULATED_HOURS"
    --expected-full-year "$EXPECTED_FULL_YEAR"
    --num-workers "$NUM_WORKERS"
    --threads-per-worker "$THREADS_PER_WORKER"
    --slurm-cpus "$SLURM_CPUS"
    --slurm-memory "$SLURM_MEMORY"
    --mail-user "$MAIL_USER"
    --mail-type "$MAIL_TYPE"
    --land-constraint "$LAND_CONSTRAINT"
    --capacity-rule "$CAPACITY_RULE"
    --land-allocation "$LAND_ALLOCATION"
    --temporal-accounting-mode "$TEMPORAL_ACCOUNTING_MODE"
    --ramp-limit-basis "$RAMP_LIMIT_BASIS"
    --include-site-costs "$INCLUDE_SITE_COSTS"
    --cost-scope "$COST_SCOPE"
    --flat-water-model-currency-per-m3 "$FLAT_WATER_MODEL_PER_M3"
    --source-commit "$SOURCE_COMMIT"
    --source-diff-sha256 "$SOURCE_DIFF_SHA256"
    --expected-currency EUR
  )
  if [[ -n "$RELEASE_SOURCE_INVENTORY" ]]; then
    MANIFEST_COMMAND+=(--release-source-inventory "$RELEASE_SOURCE_INVENTORY")
  fi
  if [[ -n "$FINANCE_CSV" ]]; then
    MANIFEST_COMMAND+=(--finance-csv "$FINANCE_CSV")
  else
    MANIFEST_COMMAND+=(--interest-rate "$INTEREST_RATE")
  fi
  if [[ -n "$LOCATIONS_CSV" ]]; then
    MANIFEST_COMMAND+=(--locations-csv "$LOCATIONS_CSV")
  fi
  if [[ -n "$SITE_COSTS_CSV" ]]; then
    MANIFEST_COMMAND+=(--site-costs-csv "$SITE_COSTS_CSV")
  fi
  if [[ -n "$WEATHER_LAND_CORRECTION" ]]; then
    MANIFEST_COMMAND+=(--weather-land-correction "$WEATHER_LAND_CORRECTION")
  fi
  if [[ -n "$MAX_SNAPSHOTS" ]]; then
    MANIFEST_COMMAND+=(--max-snapshots "$MAX_SNAPSHOTS")
  fi
  while [[ $# -gt 0 ]]; do
    MANIFEST_COMMAND+=("$1")
    shift
  done
}

# --- shard and merge/QA jobs: submitted from the login node, or printed for a dry run ----------

submit_or_plan_jobs() {
  scenario_env=(
    "ARC_REPO_DIR=$REPO_ROOT"
    "ARC_CAMPAIGN_ID=$CAMPAIGN_ID"
    "ARC_RUN_ID=$RUN_ID"
    "ARC_SCENARIO_ID=$SCENARIO_ID"
    "ARC_SCENARIO_CLASS=$SCENARIO_CLASS"
    "ARC_SCENARIO_DESCRIPTION=$SCENARIO_DESCRIPTION"
    "ARC_RUN_STAGE=$RUN_STAGE"
    "ARC_RUN_MANIFEST=$MANIFEST_PATH"
    "ARC_RUN_OUTPUT_DIR=$RUN_DIR"
    "ARC_SOURCE_COMMIT=$SOURCE_COMMIT"
    "ARC_RELEASE_INVENTORY=$RELEASE_SOURCE_INVENTORY"
    "ARC_MANIFEST_SHA256=$MANIFEST_SHA256"
    "ARC_TECH_YAML=$TECH_YAML"
    "ARC_PLANT_DIR=$PLANT_DIR"
    "ARC_PV_POLICY=$PV_POLICY"
    "ARC_FINANCE_CSV=$FINANCE_CSV"
    "ARC_INTEREST_RATE=$INTEREST_RATE"
    "ARC_SITE_COSTS_CSV=$SITE_COSTS_CSV"
    "ARC_LAND_CSV=$LAND_CSV"
    "ARC_WEATHER_DIR=$WEATHER_DIR"
    "ARC_WEATHER_LAND_CORRECTION=$WEATHER_LAND_CORRECTION"
    "ARC_LOCATIONS_CSV=$LOCATIONS_CSV"
    "ARC_TIME_STEP=$TIME_STEP"
    "ARC_MAX_SNAPSHOTS=$MAX_SNAPSHOTS"
    "ARC_LIMIT="
    "ARC_FAIL_FAST=1"
    "ARC_LAND_CONSTRAINT=$LAND_CONSTRAINT"
    "ARC_CAPACITY_RULE=$CAPACITY_RULE"
    "ARC_LAND_ALLOCATION=$LAND_ALLOCATION"
    "ARC_TEMPORAL_ACCOUNTING_MODE=$TEMPORAL_ACCOUNTING_MODE"
    "ARC_RAMP_LIMIT_BASIS=$RAMP_LIMIT_BASIS"
    "ARC_INCLUDE_SITE_COSTS=$INCLUDE_SITE_COSTS"
    "ARC_QUIET=1"
    "ARC_THREADS_PER_WORKER=$THREADS_PER_WORKER"
    "ARC_NUM_WORKERS=$NUM_WORKERS"
    "ARC_EXPECTED_CURRENCY=EUR"
  )

  explicit_inputs=()
  job_ids=()
  pinned_cluster="$REQUESTED_CLUSTER"

  if [[ "$RUN_STAGE" == "global" ]]; then
    shard_specs=(
      "west2 -180 -90"
      "west1 -90 0"
      "east1 0 90"
      "east2 90 180"
    )
  else
    shard_specs=("single")
  fi

  for shard_spec in "${shard_specs[@]}"; do
    read -r shard lon_min lon_max <<<"$shard_spec"
    output_csv="$RUN_DIR/shards/${shard}.csv"
    explicit_inputs+=("$output_csv")
    run_label="glr-${SCENARIO_JOB_TOKEN}-${RUN_STAGE}-${RUN_ID}-${shard}"
    run_label="${run_label:0:120}"
    slurm_log="$RUN_DIR/logs/${shard}-%j.out"
    sbatch_args=(
      sbatch --parsable --export=ALL
      --cpus-per-task="$SLURM_CPUS" --mem="$SLURM_MEMORY"
      --mail-user="$MAIL_USER" --mail-type="$MAIL_TYPE"
      --job-name="$run_label"
      --output="$slurm_log"
    )
    if [[ -n "$pinned_cluster" ]]; then
      sbatch_args+=(--clusters="$pinned_cluster")
    fi
    sbatch_args+=(arc/jobs/01_run_global.sh "$run_label")
    if [[ -n "$LOCATIONS_CSV" ]]; then
      sbatch_args+=("$LOCATIONS_CSV")
    fi

    shard_env=(
      "${scenario_env[@]}"
      "ARC_OUTPUT_CSV=$output_csv"
      "ARC_SHARD=$shard"
      "ARC_LON_MIN=${lon_min:-}"
      "ARC_LON_MAX=${lon_max:-}"
    )

    if $DRY_RUN; then
      echo "  planned shard $shard:"
      print_command env "${shard_env[@]}" "${sbatch_args[@]}"
      continue
    fi

    raw_job_id=$(env "${shard_env[@]}" "${sbatch_args[@]}")
    job_id="$(job_id_from_parsable "$raw_job_id")"
    job_cluster="$(job_cluster_from_parsable "$raw_job_id")"
    if [[ ! "$job_id" =~ ^[0-9]+$ ]]; then
      echo "Could not parse SLURM job ID from: $raw_job_id" >&2
      exit 2
    fi
    if [[ -z "$pinned_cluster" && -n "$job_cluster" ]]; then
      pinned_cluster="$job_cluster"
    elif [[ -n "$pinned_cluster" && -n "$job_cluster" && "$pinned_cluster" != "$job_cluster" ]]; then
      echo "SLURM placed $job_id on $job_cluster, expected $pinned_cluster" >&2
      exit 2
    fi
    job_ids+=("$job_id")
    echo "  submitted $shard: ${job_id}${job_cluster:+;${job_cluster}}"
  done

  merged_csv="$RUN_DIR/merged/global_run_results.csv"
  qa_json="$RUN_DIR/qa/validation.json"
  qa_command=(
    "$ARC_PYTHON" -m arc.qa merge
    --expected-input-count "${#explicit_inputs[@]}"
    --output "$merged_csv"
    --qa-output "$qa_json"
    --manifest "$MANIFEST_PATH"
    --scenario-id "$SCENARIO_ID"
    --stage "$RUN_STAGE"
    --expected-currency EUR
    --require-interest-overrides
  )
  for input_csv in "${explicit_inputs[@]}"; do
    qa_command+=(--input "$input_csv")
  done
  if [[ "$RUN_STAGE" == "global" && -z "$LOCATIONS_CSV" ]]; then
    qa_command+=(--expected-locations "$LAND_CSV" --expected-locations-kind land)
  else
    qa_command+=(--expected-locations "$LOCATIONS_CSV" --expected-locations-kind explicit)
  fi
  if [[ "$RUN_STAGE" != "smoke" ]]; then
    qa_command+=(--require-full-year)
  fi
  qa_source_verify=(
    "$ARC_PYTHON" -m arc.release verify
    --root "$REPO_ROOT"
    --inventory "$RELEASE_SOURCE_INVENTORY"
  )
  qa_input_verify=(
    "$ARC_PYTHON" -m arc.manifest verify
    --manifest "$MANIFEST_PATH"
  )
  qa_wrap="cd $(printf '%q' "$REPO_ROOT") && $(render_shell_command "${qa_source_verify[@]}") && $(render_shell_command "${qa_input_verify[@]}") && $(render_shell_command "${qa_command[@]}")"
  qa_label="glr-qa-${SCENARIO_JOB_TOKEN}-${RUN_STAGE}-${RUN_ID}"
  qa_label="${qa_label:0:120}"

  if $DRY_RUN; then
    echo "  planned explicit merge/QA after successful shard jobs:"
    print_command sbatch --parsable --partition=short --time=01:00:00 --cpus-per-task=1 --mem=8G \
      --mail-user="$MAIL_USER" --mail-type="$MAIL_TYPE" \
      --dependency="afterok:<exact-shard-job-ids>" --job-name="$qa_label" \
      --output="$RUN_DIR/logs/merge-qa-%j.out" --wrap="$qa_wrap"
    return 0
  fi

  dependencies="$(IFS=:; echo "${job_ids[*]}")"
  qa_sbatch_args=(
    sbatch --parsable
    --partition=short
    --time=01:00:00
    --cpus-per-task=1
    --mem=8G
    --mail-user="$MAIL_USER"
    --mail-type="$MAIL_TYPE"
    --dependency="afterok:${dependencies}"
    --job-name="$qa_label"
    --output="$RUN_DIR/logs/merge-qa-%j.out"
    --wrap="$qa_wrap"
  )
  if [[ -n "$pinned_cluster" ]]; then
    qa_sbatch_args+=(--clusters="$pinned_cluster")
  fi
  raw_qa_job_id="$("${qa_sbatch_args[@]}")"
  qa_job_id="$(job_id_from_parsable "$raw_qa_job_id")"
  if [[ ! "$qa_job_id" =~ ^[0-9]+$ ]]; then
    echo "Could not parse merge/QA job ID from: $raw_qa_job_id" >&2
    exit 2
  fi

  write_submission_record
  echo "  merge/QA job: $qa_job_id"
  echo "  submission record: $RUN_DIR/submission.json"
  echo "  merged output: $merged_csv"
  echo "  QA report: $qa_json"
}

# The mutable submission facts beside the byte-immutable manifest, written without Python:
# job ids here, the scheduler's view of each job in submission_scheduler.txt.
write_submission_record() {
  local path="$RUN_DIR/submission.json"
  if [[ -e "$path" ]]; then
    echo "Refusing to overwrite submission record: $path" >&2
    exit 2
  fi
  local text
  for text in "$MANIFEST_PATH" "$MANIFEST_SHA256" "$MAIL_USER" "$MAIL_TYPE" "$pinned_cluster"; do
    if [[ "$text" == *[\"\\]* ]]; then
      echo "Cannot write $path: a value contains a quote or backslash: $text" >&2
      exit 2
    fi
  done
  local shard_ids="" id
  for id in "${job_ids[@]}"; do
    shard_ids+="\"$id\", "
  done
  local cluster_json="null"
  if [[ -n "$pinned_cluster" ]]; then
    cluster_json="\"$pinned_cluster\""
  fi
  cat > "$path.tmp" <<EOF
{
  "cluster": $cluster_json,
  "manifest": "$MANIFEST_PATH",
  "manifest_sha256": "$MANIFEST_SHA256",
  "merge_qa_job_id": "$qa_job_id",
  "requested_mail_type": "$MAIL_TYPE",
  "requested_mail_user": "$MAIL_USER",
  "scheduler_snapshots": "submission_scheduler.txt",
  "schema_version": 2,
  "shard_job_ids": [${shard_ids%, }],
  "submitted_utc": "$(date -u +%Y-%m-%dT%H:%M:%S+00:00)"
}
EOF
  mv "$path.tmp" "$path"
  local cluster_option=()
  if [[ -n "$pinned_cluster" ]]; then
    cluster_option=(-M "$pinned_cluster")
  fi
  for id in "${job_ids[@]}" "$qa_job_id"; do
    scontrol "${cluster_option[@]}" --oneliner show job "$id" >> "$RUN_DIR/submission_scheduler.txt" 2>&1 || true
  done
}

# --- login node: the prepare job, then sbatch only ------------------------------------------

if [[ "$PHASE" == "login" ]]; then
  SOURCE_DIFF_SHA256="computed in the prepare job"
  prepare_log_dir="$RESULTS_ROOT/logs"
  PLAN_FILE="$prepare_log_dir/prepare-${RUN_ID}.plan"
  if [[ -e "$PLAN_FILE" ]]; then
    echo "Refusing to reuse $PLAN_FILE: choose another --run-id" >&2
    exit 2
  fi
  mkdir -p "$prepare_log_dir"
  prepare_command=(
    sbatch --wait --parsable
    --partition=short --time=00:30:00 --cpus-per-task=2 --mem=16G
    --mail-user="$MAIL_USER" --mail-type="$MAIL_TYPE"
    --job-name="glr-prepare-${RUN_ID}"
    --output="$prepare_log_dir/prepare-${RUN_ID}-%j.out"
    arc/submit_lory_sequence.sh "${ORIGINAL_ARGS[@]}" --run-id "$RUN_ID"
    --phase prepare --plan-file "$PLAN_FILE"
  )
  echo "Prepare job (checks and manifests on a compute node; waiting for it):"
  print_command "${prepare_command[@]}"
  prepare_status=0
  raw_prepare_job_id="$(ARC_REPO_DIR="$REPO_ROOT" "${prepare_command[@]}")" || prepare_status=$?
  prepare_log="$prepare_log_dir/prepare-${RUN_ID}-$(job_id_from_parsable "${raw_prepare_job_id%%$'\n'*}").out"
  echo
  if [[ -f "$prepare_log" ]]; then
    cat "$prepare_log"
  else
    echo "Prepare log not found: $prepare_log" >&2
  fi
  if (( prepare_status != 0 )); then
    echo "Prepare job failed (exit $prepare_status); nothing was submitted." >&2
    exit 2
  fi
  if $DRY_RUN; then
    exit 0
  fi

  prepared_run_dirs=()
  while IFS= read -r prepared_run_dir; do
    [[ -n "$prepared_run_dir" ]] && prepared_run_dirs+=("$prepared_run_dir")
  done < "$PLAN_FILE"
  if (( ${#prepared_run_dirs[@]} != ${#REQUESTED_SCENARIOS[@]} )); then
    echo "The prepare job prepared ${#prepared_run_dirs[@]} run(s) for ${#REQUESTED_SCENARIOS[@]} scenario(s); nothing was submitted." >&2
    exit 2
  fi
  echo
  echo "Submitting from the login node:"
  for prepared_run_dir in "${prepared_run_dirs[@]}"; do
    # shellcheck source=/dev/null
    source "$prepared_run_dir/submit.env"
    echo "Scenario: $SCENARIO_ID ($RUN_DIR)"
    submit_or_plan_jobs
  done
  exit 0
fi

# --- prepare job, or a local dry run: check every run before creating anything -------------------

if [[ "$PHASE" == "prepare" ]] && ! $DRY_RUN; then
  if [[ -z "$PLAN_FILE" || -e "$PLAN_FILE" ]]; then
    echo "The prepare job needs a new --plan-file" >&2
    exit 2
  fi
fi

SOURCE_DIFF_SHA256="${ARC_SOURCE_DIFF_SHA256:-}"
if [[ -z "$SOURCE_DIFF_SHA256" ]] && git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  if [[ -n "$(git status --porcelain)" ]]; then
    SOURCE_DIFF_SHA256=$(compute_source_diff_sha256)
  else
    SOURCE_DIFF_SHA256="clean"
  fi
fi
SOURCE_DIFF_SHA256="${SOURCE_DIFF_SHA256:-unknown}"

for requested_scenario in "${REQUESTED_SCENARIOS[@]}"; do
  configure_scenario "$requested_scenario"
  if [[ -e "$RUN_DIR" ]]; then
    echo "Refusing to overwrite existing immutable run directory: $RUN_DIR" >&2
    exit 2
  fi
  validate_scenario_sources
done

print_campaign_header

for requested_scenario in "${REQUESTED_SCENARIOS[@]}"; do
  configure_scenario "$requested_scenario"
  print_scenario_summary

  if $DRY_RUN; then
    manifest_command "$VALIDATION_PYTHON"
    echo "  planned manifest command:"
    print_command "${MANIFEST_COMMAND[@]}"
    MANIFEST_SHA256="planned-byte-immutable-manifest-sha256"
    submit_or_plan_jobs
    continue
  fi

  # The prepare job: run directory, manifest, and what the login node needs to submit
  mkdir -p "$RUN_DIR/shards" "$RUN_DIR/merged" "$RUN_DIR/qa" "$RUN_DIR/logs"
  manifest_command "$ARC_PYTHON"
  "${MANIFEST_COMMAND[@]}"
  MANIFEST_SHA256=$("$ARC_PYTHON" -c 'import hashlib,sys; print(hashlib.sha256(open(sys.argv[1], "rb").read()).hexdigest())' "$MANIFEST_PATH")
  write_submit_env "$RUN_DIR/submit.env"
  printf '%s\n' "$RUN_DIR" >> "$PLAN_FILE"
  echo "  prepared: manifest sha256 $MANIFEST_SHA256"
done

if $DRY_RUN; then
  echo
  echo "DRY-RUN complete: no run directories were created and no jobs were submitted."
fi
