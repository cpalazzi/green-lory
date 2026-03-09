#!/bin/bash
# Submit 4 SLURM jobs to re-run the 13,714 cells that failed in the original global run.
# Each job processes one longitude quadrant using its per-quadrant locations CSV.
#
# Usage (from the repo root on ARC):
#   bash arc/submit_missing_rerun.sh [run-label]
#
# The run-label defaults to "missing_rerun".  Results land in results/<run-label>/.
# Merge with the original combined CSV once all 4 jobs complete.

set -euo pipefail

RUN_LABEL="${1:-missing_rerun}"
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

echo "=== Green Lory — re-run missing cells ==="
echo "Run label : $RUN_LABEL"
echo "Repo dir  : $REPO_DIR"
echo ""

# Verify per-quadrant CSVs exist
for QUAD in west2 west1 east1 east2; do
  CSV="$REPO_DIR/inputs/missing_cells_${QUAD}.csv"
  if [[ ! -f "$CSV" ]]; then
    echo "ERROR: missing $CSV — run scripts/make_missing_cells_csv.py first" >&2
    exit 2
  fi
  NROWS=$(( $(wc -l < "$CSV") - 1 ))
  echo "  $QUAD : $NROWS cells ($CSV)"
done
echo ""

# Submit one job per quadrant
for QUAD in west2 west1 east1 east2; do
  CSV="$REPO_DIR/inputs/missing_cells_${QUAD}.csv"
  JOB_LABEL="${RUN_LABEL}_${QUAD}"

  JOB_ID=$(
    ARC_LOCATIONS_CSV="$CSV" \
    ARC_OUTPUT_CSV="results/${RUN_LABEL}/run_global_${JOB_LABEL}_\$(date +%Y%m%d-%H%M%S).csv" \
    sbatch \
      --job-name="gl-missing-${QUAD}" \
      --export=ALL \
      "$REPO_DIR/arc/jobs/01_run_global.sh" \
      "$JOB_LABEL" \
      "$CSV" \
      | awk '{print $4}'
  )
  echo "  Submitted $QUAD → SLURM job $JOB_ID"
done

echo ""
echo "Monitor with:  squeue -u \$USER"
echo ""
echo "Once all jobs finish, merge results with the original combined CSV:"
echo "  python3 scripts/merge_results.py results/${RUN_LABEL}/ results/global_run_results.csv results/global_run_results_full.csv"
