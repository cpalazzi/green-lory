#!/bin/bash
# Submit helper: runs preflight then submits full global SLURM run.
#
# Flags:
#   --quadrants     Submit all 4 longitude-quadrant jobs simultaneously.
#   --phased        Submit west2+west1 immediately; submit east1+east2 with
#                   afterok:west2:west1 dependency so they only run if both
#                   western jobs succeed.  Useful for catching early failures
#                   before committing the full cluster budget.
#   --finance-mode  none|spatial|custom  (default: none)
#   --interest-csv  path to spatial cost CSV (required for --finance-mode custom)
#   --locations-csv path to a lat/lon CSV to restrict the sweep
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: bash arc/submit_global_run.sh <run-label> [locations-csv] [--finance-mode none|spatial|custom] [--interest-csv path] [--quadrants|--phased]" >&2
  exit 2
fi

RUN_LABEL=""
LOCATIONS_CSV="${ARC_LOCATIONS_CSV:-}"
FINANCE_MODE="${ARC_FINANCE_MODE:-none}"
INTEREST_CSV_OVERRIDE=""
QUADRANTS=false
PHASED=false

while [[ $# -gt 0 ]]; do
  case "$1" in
    --locations-csv)
      LOCATIONS_CSV="$2"
      shift 2
      ;;
    --finance-mode|--finance)
      FINANCE_MODE="$2"
      shift 2
      ;;
    --interest-csv)
      INTEREST_CSV_OVERRIDE="$2"
      shift 2
      ;;
    --quadrants)
      QUADRANTS=true
      shift
      ;;
    --phased)
      PHASED=true
      shift
      ;;
    --*)
      echo "Unknown option: $1" >&2
      exit 2
      ;;
    *)
      if [[ -z "$RUN_LABEL" ]]; then
        RUN_LABEL="$1"
      elif [[ -z "$LOCATIONS_CSV" ]]; then
        LOCATIONS_CSV="$1"
      else
        echo "Unexpected positional argument: $1" >&2
        exit 2
      fi
      shift
      ;;
  esac
done

if [[ -z "$RUN_LABEL" ]]; then
  echo "Missing required <run-label>." >&2
  exit 2
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
export ARC_REPO_DIR="${ARC_REPO_DIR:-$REPO_ROOT}"

if [[ -n "$LOCATIONS_CSV" ]]; then
  export ARC_LOCATIONS_CSV="$LOCATIONS_CSV"
fi

case "$FINANCE_MODE" in
  none)
    unset ARC_INTEREST_CSV || true
    ;;
  spatial)
    export ARC_INTEREST_CSV="${INTEREST_CSV_OVERRIDE:-inputs/spatial_cost_inputs.csv}"
    ;;
  custom)
    if [[ -z "$INTEREST_CSV_OVERRIDE" ]]; then
      echo "--finance-mode custom requires --interest-csv <path>" >&2
      exit 2
    fi
    export ARC_INTEREST_CSV="$INTEREST_CSV_OVERRIDE"
    ;;
  *)
    echo "Invalid --finance-mode '$FINANCE_MODE' (expected: none|spatial|custom)" >&2
    exit 2
    ;;
esac

echo "Finance mode: $FINANCE_MODE"
if [[ -n "${ARC_INTEREST_CSV:-}" ]]; then
  echo "Interest CSV: $ARC_INTEREST_CSV"
else
  echo "Interest CSV: <none> (using tech config values everywhere)"
fi

if [[ -x "arc/arc_check_run_inputs.sh" ]]; then
  bash arc/arc_check_run_inputs.sh "${ARC_LOCATIONS_CSV:-}"
fi

_submit_one() {
  # Submits one job and prints the SLURM job ID to stdout (display goes to stderr).
  # Usage: job_id=$(_submit_one label [lon_min lon_max] [--dependency <dep>])
  local label="$1"; shift
  local lon_min="${1:-}"; local lon_max="${2:-}"
  [[ $# -ge 2 ]] && shift 2

  local extra_env=()
  [[ -n "$lon_min" ]] && extra_env+=(--export="ALL,ARC_LON_MIN=${lon_min},ARC_LON_MAX=${lon_max}")

  local dep_args=()
  if [[ "${1:-}" == "--dependency" ]]; then
    dep_args+=(--dependency="$2")
    shift 2
  fi

  local out
  if [[ -n "${ARC_LOCATIONS_CSV:-}" ]]; then
    out=$(sbatch "${dep_args[@]}" "${extra_env[@]}" \
      arc/jobs/01_run_global.sh "$label" "$ARC_LOCATIONS_CSV")
  else
    out=$(sbatch "${dep_args[@]}" "${extra_env[@]}" \
      arc/jobs/01_run_global.sh "$label")
  fi
  # Display to stderr so capturing stdout returns only the job ID
  echo "$out" >&2
  awk '{print $4}' <<<"$out"  # stdout: job ID only ("Submitted batch job <ID> on cluster htc")
}

if $QUADRANTS || $PHASED; then
  QUADRANT_BOUNDS=("-180 -90" "-90 0" "0 90" "90 180")
  QUADRANT_NAMES=("west2" "west1" "east1" "east2")

  if $PHASED; then
    # ── Phased mode ────────────────────────────────────────────────────────
    # Submit west2 + west1 immediately.
    # Submit east1 + east2 with afterok:<west2>:<west1> so they start only if
    # both western jobs complete successfully.
    echo "Submitting phased quadrant jobs for run: $RUN_LABEL"
    echo "(west2 + west1 now;  east1 + east2 afterok both)"
    echo ""

    echo "=== Quadrant west2: lon [-180, -90) ==="
    WEST2_ID=$(_submit_one "${RUN_LABEL}-west2" "-180" "-90")
    echo ""
    echo "=== Quadrant west1: lon [-90, 0) ==="
    WEST1_ID=$(_submit_one "${RUN_LABEL}-west1" "-90" "0")
    echo ""

    DEP="afterok:${WEST2_ID}:${WEST1_ID}"
    echo "=== Quadrant east1: lon [0, 90) — dependency: ${DEP} ==="
    EAST1_ID=$(_submit_one "${RUN_LABEL}-east1" "0" "90" --dependency "$DEP")
    echo ""
    echo "=== Quadrant east2: lon [90, 180) — dependency: ${DEP} ==="
    EAST2_ID=$(_submit_one "${RUN_LABEL}-east2" "90" "180" --dependency "$DEP")

    echo ""
    echo "Phased submission complete:"
    echo "  west2 : $WEST2_ID  (running)"
    echo "  west1 : $WEST1_ID  (running)"
    echo "  east1 : $EAST1_ID  (pending afterok:$WEST2_ID:$WEST1_ID)"
    echo "  east2 : $EAST2_ID  (pending afterok:$WEST2_ID:$WEST1_ID)"
  else
    # ── Simultaneous quadrants mode ─────────────────────────────────────────
    echo "Submitting 4 longitude quadrant jobs for run: $RUN_LABEL"
    JOB_IDS=()
    for i in "${!QUADRANT_BOUNDS[@]}"; do
      read -r lo hi <<<"${QUADRANT_BOUNDS[$i]}"
      qlabel="${RUN_LABEL}-${QUADRANT_NAMES[$i]}"
      echo ""
      echo "=== Quadrant ${QUADRANT_NAMES[$i]}: lon [${lo}, ${hi}) ==="
      job_id=$(_submit_one "$qlabel" "$lo" "$hi")
      JOB_IDS+=("$job_id")
    done
    echo ""
    echo "All quadrant jobs submitted:"
    for i in "${!QUADRANT_NAMES[@]}"; do
      echo "  ${QUADRANT_NAMES[$i]}: ${JOB_IDS[$i]}"
    done
  fi

  echo ""
  echo "Monitor:"
  echo "  squeue -u \$USER"
  echo "  ls -1t results/${RUN_LABEL}-*/run_global_*.csv"
else
  job_id=$(_submit_one "$RUN_LABEL")
  echo ""
  echo "Monitor:"
  echo "  squeue -j $job_id"
  echo "  ls -1t logs/arc-${RUN_LABEL}-*.log | head -n 1"
fi
