#!/bin/bash
# Submit helper: runs preflight then submits full global SLURM run.
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: bash arc/submit_global_run.sh <run-label> [locations-csv] [--finance-mode none|spatial|flat|custom] [--interest-csv path]" >&2
  exit 2
fi

RUN_LABEL=""
LOCATIONS_CSV="${ARC_LOCATIONS_CSV:-}"
FINANCE_MODE="${ARC_FINANCE_MODE:-none}"
INTEREST_CSV_OVERRIDE=""

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
  flat)
    export ARC_INTEREST_CSV="${INTEREST_CSV_OVERRIDE:-inputs/example_finance_overrides_flat.csv}"
    if [[ ! -f "$ARC_INTEREST_CSV" ]]; then
      python scripts/make_flat_finance_overrides.py \
        --spatial-csv inputs/example_finance_overrides_spatial.csv \
        --tech-yaml "${ARC_TECH_YAML:-inputs/tech_config_ammonia_plant_2030_dea.yaml}" \
        --output-csv "$ARC_INTEREST_CSV"
    fi
    ;;
  custom)
    if [[ -z "$INTEREST_CSV_OVERRIDE" ]]; then
      echo "--finance-mode custom requires --interest-csv <path>" >&2
      exit 2
    fi
    export ARC_INTEREST_CSV="$INTEREST_CSV_OVERRIDE"
    ;;
  *)
    echo "Invalid --finance-mode '$FINANCE_MODE' (expected: none|spatial|flat|custom)" >&2
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

if [[ -n "${ARC_LOCATIONS_CSV:-}" ]]; then
  out=$(sbatch arc/jobs/01_run_global.sh "$RUN_LABEL" "$ARC_LOCATIONS_CSV")
else
  out=$(sbatch arc/jobs/01_run_global.sh "$RUN_LABEL")
fi

echo "$out"
job_id=$(awk '{print $NF}' <<<"$out")

echo
echo "Monitor:"
echo "  squeue -j $job_id"
echo "  ls -1t logs/arc-${RUN_LABEL}-*.log | head -n 1"
