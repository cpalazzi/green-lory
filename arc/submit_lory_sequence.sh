#!/bin/bash
# Submit the reconciled Green Lory campaign sequence without reusing mutable shard folders.
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  bash arc/submit_lory_sequence.sh --stage smoke|diagnostic|global [options]

Required:
  --stage <stage>       smoke: 3 cells / 168 simulated hours by default
                        diagnostic: 3 cells / full year
                        global: 4 longitude shards / full year

Options:
  --campaign <id>       Campaign ID (default: lory_reconcile_20260722_v1).
  --scenario <id>       Scenario to submit; repeatable. If omitted, submit both.
  --run-id <id>         Immutable run ID. Default: UTC timestamp + source commit.
  --results-root <path> Campaign result root. Default: results/campaigns/<campaign>.
  --land-csv <path>     Versioned land table for both scenarios. Default:
                        data/max_capacities_paper_2pct_slope15.csv.
  --cluster <name>      Pin all jobs for a scenario to one ARC cluster.
  --smoke-hours <n>     Simulated duration for smoke jobs (default: 168 hours).
                        This is 168 snapshots at 1h and 42 snapshots at 4h.
  --global-locations <csv>
                        Restrict the global stage to the lat,lon cells listed
                        (e.g. onshore cells with suitable land); shards still
                        split by longitude.
  --diagnostic-locations <csv>
                        Cells for the smoke and diagnostic stages (default:
                        inputs/lory_reconciliation_diagnostic_cells.csv).
  --dry-run             Validate local source inputs and print commands only.
  -h, --help            Show this help.

Scenarios:
  rep_way2050_flat_amelired_4h_tracking_nominal_h2
  central_way2050_flat_amelired_1h_tracking_explicit_compressor_dea_tank
  oat_way2050_flat_amelired_1h_tracking_nominal_h2
  oat_way2050_flat_amelired_1h_tracking_explicit_compressor_legacy_tank

This wrapper never overwrites an existing run directory and never discovers
shards with a wildcard. Every QA/merge job receives its exact input paths.
EOF
}

CAMPAIGN_ID="lory_reconcile_20260722_v1"
RUN_STAGE=""
RUN_ID=""
RESULTS_ROOT=""
CAMPAIGN_LAND_CSV="${ARC_CAMPAIGN_LAND_CSV:-}"
REQUESTED_CLUSTER="${ARC_CLUSTER:-}"
SMOKE_HOURS="${ARC_SMOKE_HOURS:-168}"
GLOBAL_LOCATIONS_CSV="${ARC_GLOBAL_LOCATIONS_CSV:-}"
DIAGNOSTIC_LOCATIONS_OVERRIDE="${ARC_DIAGNOSTIC_LOCATIONS:-}"
DEFAULT_COST_SCOPE="flat_amelired_uniform_baseline_water_no_land_rent"
DEFAULT_EXPECTED_LAND_COMPETITION_FRACTION="${ARC_CAMPAIGN_LAND_COMPETITION_FRACTION:-0.02}"
RELEASE_SOURCE_INVENTORY="${ARC_RELEASE_INVENTORY:-}"
MAIL_USER="${ARC_MAIL_USER:-carlo.palazzi@eng.ox.ac.uk}"
MAIL_TYPE="BEGIN,END,FAIL"
DRY_RUN=false
REQUESTED_SCENARIOS=()

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
if [[ ! "$CAMPAIGN_ID" =~ ^[a-zA-Z0-9._-]+$ ]]; then
  echo "Campaign ID may contain only letters, numbers, dot, underscore, and hyphen" >&2
  exit 2
fi
if [[ ! "$SMOKE_HOURS" =~ ^[1-9][0-9]*$ ]]; then
  echo "--smoke-hours must be a positive integer" >&2
  exit 2
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

VALIDATION_PYTHON=""
if command -v python3 >/dev/null 2>&1; then
  VALIDATION_PYTHON="$(command -v python3)"
elif command -v python >/dev/null 2>&1; then
  VALIDATION_PYTHON="$(command -v python)"
else
  echo "Python is required to validate plant bundles" >&2
  exit 2
fi

USER_NAME="${USER:-$(id -un)}"
ARC_GROUP="${ARC_GROUP:-engs-df-green-ammonia}"
ARC_WORK_BASE="${ARC_WORK_BASE:-/data/${ARC_GROUP}/${USER_NAME}}"
ARC_ENV_PREFIX="${ARC_ENV_PREFIX:-${ARC_WORK_BASE}/envs/green-lory-env}"
ARC_PYTHON="${ARC_PYTHON:-${ARC_ENV_PREFIX}/bin/python}"
WEATHER_DIR="${ARC_WEATHER_DIR:-data/weather_data}"
DIAGNOSTIC_LOCATIONS="${DIAGNOSTIC_LOCATIONS_OVERRIDE:-inputs/lory_reconciliation_diagnostic_cells.csv}"

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

SOURCE_DIFF_SHA256="${ARC_SOURCE_DIFF_SHA256:-}"
if [[ -z "$SOURCE_DIFF_SHA256" ]] && git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  if [[ -n "$(git status --porcelain)" ]]; then
    SOURCE_DIFF_SHA256=$(compute_source_diff_sha256)
  else
    SOURCE_DIFF_SHA256="clean"
  fi
fi
SOURCE_DIFF_SHA256="${SOURCE_DIFF_SHA256:-unknown}"

if [[ ${#REQUESTED_SCENARIOS[@]} -eq 0 ]]; then
  REQUESTED_SCENARIOS=(
    "rep_way2050_flat_amelired_4h_tracking_nominal_h2"
    "central_way2050_flat_amelired_1h_tracking_explicit_compressor_dea_tank"
  )
fi

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

configure_scenario() {
  SCENARIO_ID="$1"
  COST_SCOPE="$DEFAULT_COST_SCOPE"
  EXPECTED_LAND_COMPETITION_FRACTION="$DEFAULT_EXPECTED_LAND_COMPETITION_FRACTION"
  PV_POLICY="both"   # both: tracking generator must be extendable; fixed: it must not be
  FLAT_WATER_MODEL_PER_M3="1.751012"   # 2 USD2020/m3 x 0.875506: the September 2026 baseline
  case "$SCENARIO_ID" in
    rep_way2050_flat_amelired_4h_tracking_nominal_h2)
      SCENARIO_CLASS="10_replication"
      SCENARIO_JOB_TOKEN="rep-nomh2"
      SCENARIO_DESCRIPTION="WAY 2050, flat build/remoteness, Ameli reduced WACC, uniform 2 USD/m3 baseline water reported outside the headline, no land rent, 4-hour weather, tracking PV, nominal compressor CAPEX, legacy bundled hydrogen storage"
      TECH_YAML="inputs/tech_config_ammonia_plant_2050_way_eur_nominal_compression.yaml"
      PLANT_DIR="basic_ammonia_plant_2050_way_tracking"
      OVERRIDE_CSV="inputs/amelired_interest_inputs_2050.csv"
      LAND_CSV="${CAMPAIGN_LAND_CSV:-data/max_capacities_paper_2pct_slope15.csv}"
      TIME_STEP="4.0"
      LAND_CONSTRAINT="after_solve"
      CAPACITY_RULE="scaled_reference_design"
      LAND_ALLOCATION="exclusive"
      TEMPORAL_ACCOUNTING_MODE="legacy_scaled"
      RAMP_LIMIT_BASIS="legacy_per_snapshot"
      INCLUDE_SITE_COSTS="0"
      ;;
    oat_way2050_flat_amelired_1h_tracking_nominal_h2|oat_way2050_flat_amelired_1h_tracking_explicit_compressor_legacy_tank)
      SCENARIO_CLASS="15_attribution"
      SCENARIO_JOB_TOKEN="oat-hourly-nomh2"
      SCENARIO_DESCRIPTION="Attribution only: hourly resolution with the replication nominal-compressor and bundled-H2-storage cost assumptions; flat build/remoteness, Ameli reduced WACC, no headline water or land rent"
      TECH_YAML="inputs/tech_config_ammonia_plant_2050_way_eur_nominal_compression.yaml"
      if [[ "$SCENARIO_ID" == *explicit_compressor_legacy_tank ]]; then
        SCENARIO_JOB_TOKEN="oat-compressor"
        SCENARIO_DESCRIPTION="Attribution only: explicit compressor CAPEX added to legacy bundled H2 storage at hourly resolution. Deliberate intermediate double-count-risk case, not a recommended central estimate. Flat build/remoteness, Ameli reduced WACC, no headline water or land rent"
        TECH_YAML="inputs/tech_config_ammonia_plant_2050_way_eur.yaml"
      fi
      PLANT_DIR="basic_ammonia_plant_2050_way_tracking"
      OVERRIDE_CSV="inputs/amelired_interest_inputs_2050.csv"
      LAND_CSV="${CAMPAIGN_LAND_CSV:-data/max_capacities_paper_2pct_slope15.csv}"
      TIME_STEP="1.0"
      LAND_CONSTRAINT="after_solve"
      CAPACITY_RULE="scaled_reference_design"
      LAND_ALLOCATION="exclusive"
      TEMPORAL_ACCOUNTING_MODE="snapshot_weighted"
      RAMP_LIMIT_BASIS="per_hour"
      INCLUDE_SITE_COSTS="0"
      ;;
    central_way2050_flat_amelired_1h_tracking_explicit_compressor_dea_tank)
      SCENARIO_CLASS="20_central"
      SCENARIO_JOB_TOKEN="central-deatank"
      SCENARIO_DESCRIPTION="WAY 2050, flat build/remoteness, Ameli reduced WACC, uniform 2 USD/m3 baseline water included in the headline, no land rent, hourly weather, tracking PV, explicit compressor CAPEX, DEA 2050 tank-only hydrogen storage"
      TECH_YAML="inputs/tech_config_ammonia_plant_2050_way_eur_explicit_compressor_dea_tank.yaml"
      PLANT_DIR="basic_ammonia_plant_2050_way_tracking"
      OVERRIDE_CSV="inputs/amelired_interest_inputs_2050.csv"
      LAND_CSV="${CAMPAIGN_LAND_CSV:-data/max_capacities_paper_2pct_slope15.csv}"
      TIME_STEP="1.0"
      LAND_CONSTRAINT="after_solve"
      CAPACITY_RULE="scaled_reference_design"
      LAND_ALLOCATION="exclusive"
      TEMPORAL_ACCOUNTING_MODE="snapshot_weighted"
      RAMP_LIMIT_BASIS="per_hour"
      INCLUDE_SITE_COSTS="1"
      ;;
    central_way2050_flat_amelired_1h_fixed_explicit_compressor_dea_tank_colocated)
      SCENARIO_CLASS="20_central"
      SCENARIO_JOB_TOKEN="central-fixed-coloc"
      SCENARIO_DESCRIPTION="Central case from 16 September 2026: WAY 2050, flat build/remoteness, Ameli reduced WACC, uniform 2 USD/m3 baseline water included in the headline, no land rent, hourly weather, fixed-tilt PV only (tracking disabled by plant bundle), explicit compressor CAPEX, DEA 2050 tank-only hydrogen storage, co-located wind/PV land (wind exclusive fraction from Denholm 2009)"
      TECH_YAML="inputs/tech_config_ammonia_plant_2050_way_eur_explicit_compressor_dea_tank.yaml"
      PLANT_DIR="basic_ammonia_plant_2050_way"
      OVERRIDE_CSV="inputs/amelired_interest_inputs_2050.csv"
      LAND_CSV="${CAMPAIGN_LAND_CSV:-data/max_capacities_paper_2pct_slope15.csv}"
      TIME_STEP="1.0"
      LAND_CONSTRAINT="after_solve"
      CAPACITY_RULE="scaled_reference_design"
      LAND_ALLOCATION="colocated"
      TEMPORAL_ACCOUNTING_MODE="snapshot_weighted"
      RAMP_LIMIT_BASIS="per_hour"
      INCLUDE_SITE_COSTS="1"
      ;;
    gl_dea2050_wacc5_bflat_wflat_land20c_fixed|gl_dea2050_ameli_bflat_wflat_land20c_fixed)
      SCENARIO_CLASS="30_keyruns"
      if [[ "$SCENARIO_ID" == *_wacc5_* ]]; then
        SCENARIO_JOB_TOKEN="key-dea50-wacc5"
        SCENARIO_DESCRIPTION="Key run 1 (23 Sep 2026): DEA 2050 costs, uniform 5 % WACC, flat build/remoteness, uniform baseline water of 2 EUR2020/m3 (2.284 USD2020 at the ECB 2020 rate) in the headline, no land rent, hourly weather, fixed-tilt PV only, explicit compressor and DEA tank, co-located wind/PV, centred land build at a 20 % share, onshore cells with suitable land"
        OVERRIDE_CSV="inputs/uniform_interest_inputs_0p05_2050.csv"
        COST_SCOPE="flat_wacc5_uniform_baseline_water_no_land_rent"
      else
        SCENARIO_JOB_TOKEN="key-dea50-ameli"
        SCENARIO_DESCRIPTION="Key run 2 (23 Sep 2026): DEA 2050 costs, Ameli reduced WACC by country (cells without a September country assignment take the nearest covered cell's rate, see inputs/amelired_interest_fill_2050_center.csv), flat build/remoteness, uniform baseline water of 2 EUR2020/m3 (2.284 USD2020 at the ECB 2020 rate) in the headline, no land rent, hourly weather, fixed-tilt PV only, explicit compressor and DEA tank, co-located wind/PV, centred land build at a 20 % share, onshore cells with suitable land"
        OVERRIDE_CSV="inputs/amelired_interest_inputs_2050_center.csv"
        COST_SCOPE="flat_amelired_uniform_baseline_water_no_land_rent"
      fi
      TECH_YAML="inputs/tech_config_ammonia_plant_2050_dea_colocated.yaml"
      PLANT_DIR="basic_ammonia_plant_2050"
      LAND_CSV="${CAMPAIGN_LAND_CSV:?--land-csv must point at the centred 20 % land table}"
      EXPECTED_LAND_COMPETITION_FRACTION="0.20"
      PV_POLICY="fixed"
      FLAT_WATER_MODEL_PER_M3="2.0"
      TIME_STEP="1.0"
      LAND_CONSTRAINT="after_solve"
      CAPACITY_RULE="scaled_reference_design"
      LAND_ALLOCATION="colocated"
      TEMPORAL_ACCOUNTING_MODE="snapshot_weighted"
      RAMP_LIMIT_BASIS="per_hour"
      INCLUDE_SITE_COSTS="1"
      ;;
    *)
      echo "Unknown scenario: $SCENARIO_ID" >&2
      echo "Run with --help to list supported scenarios." >&2
      exit 2
      ;;
  esac

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
    ALLOW_CONSERVATIVE_UNION_FALLBACK="1"
  elif [[ "$RUN_STAGE" == "diagnostic" ]]; then
    PATH_CLASS="$SCENARIO_CLASS"
    LOCATIONS_CSV="$DIAGNOSTIC_LOCATIONS"
    MAX_SNAPSHOTS=""
    SIMULATED_HOURS="8760"
    EXPECTED_FULL_YEAR="1"
    NUM_WORKERS="${ARC_CAMPAIGN_DIAGNOSTIC_WORKERS:-3}"
    ALLOW_CONSERVATIVE_UNION_FALLBACK="0"
  else
    PATH_CLASS="$SCENARIO_CLASS"
    LOCATIONS_CSV="$GLOBAL_LOCATIONS_CSV"
    MAX_SNAPSHOTS=""
    SIMULATED_HOURS="8760"
    EXPECTED_FULL_YEAR="1"
    NUM_WORKERS="${ARC_CAMPAIGN_GLOBAL_WORKERS:-12}"
    ALLOW_CONSERVATIVE_UNION_FALLBACK="0"
  fi
  ENSURE_FEASIBILITY="0"
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
extendable = tracking.get("p_nom_extendable", "").strip().lower() in {"1", "true", "yes"}
if pv_policy == "fixed" and extendable:
    raise SystemExit(f"fixed-PV scenario but solar_tracking is extendable in {path}")
if pv_policy != "fixed" and not extendable:
    raise SystemExit(f"solar_tracking is not extendable in {path}")
PY
}

validate_explicit_union_area() {
  local land_csv="$1"
  if [[ ! -f "$land_csv" ]]; then
    return 0
  fi
  "$VALIDATION_PYTHON" arc/validate_land_campaign_input.py "$land_csv" \
    --expected-land-fraction "$EXPECTED_LAND_COMPETITION_FRACTION"
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
  "$VALIDATION_PYTHON" arc/release_source_inventory.py verify \
    --root "$REPO_ROOT" \
    --inventory "$RELEASE_SOURCE_INVENTORY"
}

validate_cost_scope_and_override_coverage() {
  "$VALIDATION_PYTHON" - \
    "$TECH_YAML" \
    "$OVERRIDE_CSV" \
    "$LAND_CSV" \
    "$LOCATIONS_CSV" \
    "$INCLUDE_SITE_COSTS" \
    "$DRY_RUN" \
    "$COST_SCOPE" \
    "$FLAT_WATER_MODEL_PER_M3" <<'PY'
from pathlib import Path
import sys

from arc.write_campaign_manifest import (
    build_flat_amelired_cost_scope,
    validate_override_coordinate_coverage,
)

tech_yaml = Path(sys.argv[1])
override_csv = Path(sys.argv[2])
land_csv = Path(sys.argv[3])
locations_csv = Path(sys.argv[4]) if sys.argv[4] else None
include_site_costs = sys.argv[5] == "1"
dry_run = sys.argv[6].lower() == "true"

cost_scope_id = sys.argv[7]
flat_water_model_currency_per_m3 = float(sys.argv[8])
build_flat_amelired_cost_scope(
    tech_yaml=tech_yaml,
    override_csv=override_csv,
    include_site_costs=include_site_costs,
    cost_scope_id=cost_scope_id,
    flat_water_model_currency_per_m3=flat_water_model_currency_per_m3,
)
if locations_csv is not None or land_csv.is_file():
    coverage = validate_override_coordinate_coverage(
        override_csv=override_csv,
        land_csv=land_csv,
        locations_csv=locations_csv,
    )
    print(
        "Override coverage OK: "
        f"{coverage['expected_coordinate_count']} expected coordinates, "
        f"{coverage['override_coordinate_count']} override coordinates"
    )
elif not dry_run:
    raise SystemExit(f"Land CSV is required for global override coverage: {land_csv}")
PY
}

validate_scenario_sources() {
  local required=("$TECH_YAML" "$OVERRIDE_CSV" "$DIAGNOSTIC_LOCATIONS")
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
  validate_release_source_inventory
  if [[ "$ALLOW_CONSERVATIVE_UNION_FALLBACK" == "0" ]]; then
    validate_explicit_union_area "$LAND_CSV"
  fi
  validate_cost_scope_and_override_coverage

  if $DRY_RUN; then
    [[ -f "$LAND_CSV" ]] || echo "DRY-RUN NOTE: ARC-only land input is not present locally: $LAND_CSV"
    [[ -d "$WEATHER_DIR" ]] || echo "DRY-RUN NOTE: ARC-only weather directory is not present locally: $WEATHER_DIR"
    [[ -f data/countries.geojson ]] || echo "DRY-RUN NOTE: ARC-only country input is not present locally: data/countries.geojson"
    [[ -f data/model_bathymetry.nc ]] || echo "DRY-RUN NOTE: ARC-only bathymetry input is not present locally: data/model_bathymetry.nc"
    return 0
  fi

  if [[ ! -x "$ARC_PYTHON" ]]; then
    echo "ARC Python not found or not executable: $ARC_PYTHON" >&2
    return 2
  fi
  if ! command -v sbatch >/dev/null 2>&1; then
    echo "sbatch is unavailable; run the non-dry submission on an ARC login node" >&2
    return 2
  fi

  env \
    ARC_PYTHON_BIN="$ARC_PYTHON" \
    ARC_TECH_YAML="$TECH_YAML" \
    ARC_OVERRIDE_CSV="$OVERRIDE_CSV" \
    ARC_LAND_CSV="$LAND_CSV" \
    ARC_WEATHER_DIR="$WEATHER_DIR" \
    ARC_LOCATIONS_CSV="$LOCATIONS_CSV" \
    bash arc/arc_check_run_inputs.sh "$LOCATIONS_CSV" >/dev/null
}

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

manifest_command() {
  local manifest_python="$1"
  shift
  MANIFEST_COMMAND=(
    "$manifest_python" arc/write_campaign_manifest.py
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
    --override-csv "$OVERRIDE_CSV"
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
    --ensure-feasibility "$ENSURE_FEASIBILITY"
    --land-constraint "$LAND_CONSTRAINT"
    --capacity-rule "$CAPACITY_RULE"
    --land-allocation "$LAND_ALLOCATION"
    --allow-conservative-union-fallback "$ALLOW_CONSERVATIVE_UNION_FALLBACK"
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
  if [[ -n "$LOCATIONS_CSV" ]]; then
    MANIFEST_COMMAND+=(--locations-csv "$LOCATIONS_CSV")
  fi
  if [[ -n "$MAX_SNAPSHOTS" ]]; then
    MANIFEST_COMMAND+=(--max-snapshots "$MAX_SNAPSHOTS")
  fi
  while [[ $# -gt 0 ]]; do
    MANIFEST_COMMAND+=("$1")
    shift
  done
}

# Validate every requested run before creating any directory or submitting any job.
for requested_scenario in "${REQUESTED_SCENARIOS[@]}"; do
  configure_scenario "$requested_scenario"
  if [[ -e "$RUN_DIR" ]]; then
    echo "Refusing to overwrite existing immutable run directory: $RUN_DIR" >&2
    exit 2
  fi
  validate_scenario_sources
done

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

for requested_scenario in "${REQUESTED_SCENARIOS[@]}"; do
  configure_scenario "$requested_scenario"
  echo
  echo "Scenario: $SCENARIO_ID"
  echo "  class:       $SCENARIO_CLASS"
  echo "  tech yaml:   $TECH_YAML"
  echo "  plant dir:   $PLANT_DIR"
  echo "  override:    $OVERRIDE_CSV"
  echo "  land csv:    $LAND_CSV"
  echo "  timestep:    ${TIME_STEP}h"
  echo "  land:        $LAND_CONSTRAINT / $CAPACITY_RULE / $LAND_ALLOCATION"
  echo "  temporal:    $TEMPORAL_ACCOUNTING_MODE / $RAMP_LIMIT_BASIS"
  echo "  site costs:  $INCLUDE_SITE_COSTS"
  echo "  grid backstop: $ENSURE_FEASIBILITY"
  echo "  union fallback: $ALLOW_CONSERVATIVE_UNION_FALLBACK"
  echo "  locations:   ${LOCATIONS_CSV:-<all positive-capacity land cells>}"
  echo "  snapshots:   ${MAX_SNAPSHOTS:-<full year>}"
  echo "  simulated hours: $SIMULATED_HOURS"
  echo "  run dir:     $RUN_DIR"

  if $DRY_RUN; then
    manifest_command "$VALIDATION_PYTHON"
    echo "  planned manifest command:"
    print_command "${MANIFEST_COMMAND[@]}"
  else
    mkdir -p "$RUN_DIR/shards" "$RUN_DIR/merged" "$RUN_DIR/qa" "$RUN_DIR/logs"
    manifest_command "$ARC_PYTHON"
    "${MANIFEST_COMMAND[@]}"
  fi

  if $DRY_RUN; then
    MANIFEST_SHA256="planned-byte-immutable-manifest-sha256"
  else
    MANIFEST_SHA256=$($ARC_PYTHON -c 'import hashlib,sys; print(hashlib.sha256(open(sys.argv[1], "rb").read()).hexdigest())' "$MANIFEST_PATH")
  fi

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
    "ARC_OVERRIDE_CSV=$OVERRIDE_CSV"
    "ARC_LAND_CSV=$LAND_CSV"
    "ARC_WEATHER_DIR=$WEATHER_DIR"
    "ARC_LOCATIONS_CSV=$LOCATIONS_CSV"
    "ARC_TIME_STEP=$TIME_STEP"
    "ARC_MAX_SNAPSHOTS=$MAX_SNAPSHOTS"
    "ARC_LIMIT="
    "ARC_FAIL_FAST=1"
    "ARC_ENSURE_FEASIBILITY=$ENSURE_FEASIBILITY"
    "ARC_LAND_CONSTRAINT=$LAND_CONSTRAINT"
    "ARC_CAPACITY_RULE=$CAPACITY_RULE"
    "ARC_LAND_ALLOCATION=$LAND_ALLOCATION"
    "ARC_ALLOW_CONSERVATIVE_UNION_FALLBACK=$ALLOW_CONSERVATIVE_UNION_FALLBACK"
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
    "$ARC_PYTHON" arc/merge_and_qa_campaign.py
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
    "$ARC_PYTHON" arc/release_source_inventory.py verify
    --root "$REPO_ROOT"
    --inventory "$RELEASE_SOURCE_INVENTORY"
  )
  qa_input_verify=(
    "$ARC_PYTHON" arc/verify_campaign_manifest_inputs.py
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
    continue
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

  submission_path="$RUN_DIR/submission.json"
  submission_command=(
    "$ARC_PYTHON" arc/write_campaign_submission.py
    --output "$submission_path"
    --manifest "$MANIFEST_PATH"
    --qa-job-id "$qa_job_id"
    --mail-user "$MAIL_USER"
    --mail-type "$MAIL_TYPE"
  )
  for job_id in "${job_ids[@]}"; do
    submission_command+=(--job-id "$job_id")
  done
  if [[ -n "$pinned_cluster" ]]; then
    submission_command+=(--cluster "$pinned_cluster")
  fi
  "${submission_command[@]}"

  echo "  merge/QA job: $qa_job_id"
  echo "  submission record: $submission_path"
  echo "  merged output: $merged_csv"
  echo "  QA report: $qa_json"
done

if $DRY_RUN; then
  echo
  echo "DRY-RUN complete: no directories were created and no jobs were submitted."
fi
