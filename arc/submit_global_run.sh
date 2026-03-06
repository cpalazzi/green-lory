#!/bin/bash
# Submit helper: runs preflight then submits full global SLURM run.
# Use --quadrants to submit 4 parallel jobs split by longitude.
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: bash arc/submit_global_run.sh <run-label> [locations-csv] [--finance-mode none|spatial|custom] [--interest-csv path] [--quadrants]" >&2
  exit 2
fi

RUN_LABEL=""
LOCATIONS_CSV="${ARC_LOCATIONS_CSV:-}"
FINANCE_MODE="${ARC_FINANCE_MODE:-none}"
INTEREST_CSV_OVERRIDE=""
QUADRANTS=false

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
    export ARC_INTEREST_CSV="${INTEREST_CSV_OVERRIDE:-inputs/example_finance_overrides_spatial.csv}"
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
  local label="$1"
  local lon_min="${2:-}"
  local lon_max="${3:-}"
  local extra_env=()
  if [[ -n "$lon_min" ]]; then
    extra_env+=(--export="ALL,ARC_LON_MIN=${lon_min},ARC_LON_MAX=${lon_max}")
  fi

  if [[ -n "${ARC_LOCATIONS_CSV:-}" ]]; then
    out=$(sbatch "${extra_env[@]}" arc/jobs/01_run_global.sh "$label" "$ARC_LOCATIONS_CSV")
  else
    out=$(sbatch "${extra_env[@]}" arc/jobs/01_run_global.sh "$label")
  fi
  echo "$out"
  awk '{print $NF}' <<<"$out"
}

if $QUADRANTS; then
  echo "Submitting 4 longitude quadrant jobs for run: $RUN_LABEL"
  QUADRANT_BOUNDS=("-180 -90" "-90 0" "0 90" "90 180")
  QUADRANT_NAMES=("west2" "west1" "east1" "east2")
  JOB_IDS=()
  for i in "${!QUADRANT_BOUNDS[@]}"; do
    read -r lo hi <<<"${QUADRANT_BOUNDS[$i]}"
    qlabel="${RUN_LABEL}-${QUADRANT_NAMES[$i]}"
    echo
    echo "=== Quadrant ${QUADRANT_NAMES[$i]}: lon [${lo}, ${hi}) ==="
    job_id=$(_submit_one "$qlabel" "$lo" "$hi")
    JOB_IDS+=("$job_id")
  done
  echo
  echo "All quadrant jobs submitted:"
  for i in "${!QUADRANT_NAMES[@]}"; do
    echo "  ${QUADRANT_NAMES[$i]}: ${JOB_IDS[$i]}"
  done
  echo
  echo "Monitor:"
  echo "  squeue -u \$USER"
  echo "  ls -1t results/${RUN_LABEL}-*/run_global_*.csv"
else
  out=$(_submit_one "$RUN_LABEL")
  echo
  job_id=$(awk '{print $NF}' <<<"$out")
  echo "Monitor:"
  echo "  squeue -j $job_id"
  echo "  ls -1t logs/arc-${RUN_LABEL}-*.log | head -n 1"
fi
