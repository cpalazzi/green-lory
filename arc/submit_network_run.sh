#!/bin/bash
# Submit one green-porpoise network comparison on ARC (htc) through the
# arc/jobs/03_replay_historical_network.sh template, from the immutable network
# source release, into a new output directory under the reconciliation campaign.
#
# Run this ON THE ARC LOGIN NODE (or via ssh). It refuses an existing output
# directory, records the sbatch line beside the job logs, and never edits inputs.
#
#   bash arc/submit_network_run.sh --name deposited-legacyrep-20260916-v1 \
#       [--contract <campaign>/lory/global_exports/<id>/<case>/contract.json] \
#       [--cpus 4] [--mem 64G] [--hours 2] [--solver-time 3600] [--solver-mem-gb 48] \
#       [--gap 0.001] [--routes iso3-1000km] [--pipeline-multiplier 1.1] \
#       [--release 20260914-v3] [--clusters all] [--dry-run]
#
# Without --contract the archived supplier table (c_NH3_cost_4.5.csv in the
# historical archive) is used. The five 14 September comparisons and run D used
# 4 CPUs / 64 GB / 2 h with a 3600 s solver limit; an accepted 0.1 % gap needs
# a longer allocation (e.g. --cpus 12 --mem 96G --hours 12 --solver-time 39600).
set -euo pipefail
BASE=/data/engs-df-green-ammonia/engs2523
CAMP=$BASE/green-lory-campaigns/verschuur_reconcile_20260907_v1
RELEASE=20260914-v3
NAME=""; CONTRACT=""; CPUS=4; MEM=64G; HOURS=2; STIME=3600; SMEM=48; GAP=0.001
ROUTES=iso3-1000km; PMULT=1.1; DRY=0
# Submit to whichever cluster has room (arc or htc); the job template's own --clusters=htc
# directive is overridden by this command-line option. Query with sacct/squeue -M all.
CLUSTERS=all
while [[ $# -gt 0 ]]; do
  case "$1" in
    --name) NAME="$2"; shift 2;;
    --contract) CONTRACT="$2"; shift 2;;
    --cpus) CPUS="$2"; shift 2;;
    --mem) MEM="$2"; shift 2;;
    --hours) HOURS="$2"; shift 2;;
    --solver-time) STIME="$2"; shift 2;;
    --solver-mem-gb) SMEM="$2"; shift 2;;
    --gap) GAP="$2"; shift 2;;
    --routes) ROUTES="$2"; shift 2;;
    --pipeline-multiplier) PMULT="$2"; shift 2;;
    --release) RELEASE="$2"; shift 2;;
    --clusters) CLUSTERS="$2"; shift 2;;
    --dry-run) DRY=1; shift;;
    *) echo "unknown option $1" >&2; exit 2;;
  esac
done
[[ -n "$NAME" ]] || { echo "--name is required" >&2; exit 2; }
SRC=$BASE/green-lory-releases/$RELEASE
OUT=$CAMP/networks/$NAME
[[ -d "$SRC" ]] || { echo "release missing: $SRC" >&2; exit 3; }
[[ -e "$OUT" ]] && { echo "refusing to overwrite $OUT" >&2; exit 4; }
if [[ -n "$CONTRACT" ]]; then
  [[ -f "$CONTRACT" ]] || { echo "contract missing: $CONTRACT" >&2; exit 3; }
  [[ -f "$(dirname "$CONTRACT")/suppliers_USD2018.csv" ]] || { echo "suppliers_USD2018.csv missing beside contract" >&2; exit 3; }
  EXTRA="--supplier-contract $CONTRACT"
else
  EXTRA=""
fi
# walltime = solver limit + model build/export margin; refuse an inconsistent request
WALL=$((HOURS * 3600))
(( STIME + 900 <= WALL )) || { echo "solver limit $STIME s leaves less than 15 min of the $HOURS h walltime for build/export" >&2; exit 5; }
mkdir -p "$CAMP/networks/logs"
CMD=(sbatch -M "$CLUSTERS" --partition=short --cpus-per-task="$CPUS" --mem="$MEM" --time="$(printf '%02d:00:00' "$HOURS")" \
  --mail-type=BEGIN,END,FAIL --mail-user=carlo.palazzi@eng.ox.ac.uk -J "gpo-$NAME" \
  -o "$CAMP/networks/logs/$NAME-%j.out" -e "$CAMP/networks/logs/$NAME-%j.err" \
  "$SRC/arc/jobs/03_replay_historical_network.sh" "$SRC" "$CAMP/historical/git-0a63616" "$OUT" \
  --published-code "$CAMP/historical/mendeley-v1" --sparse --solver-memory-gb "$SMEM" --time-limit "$STIME" --gap "$GAP" \
  --onshore-routes "$ROUTES" --pipeline-multiplier "$PMULT" $EXTRA)
echo "${CMD[*]}"
if (( DRY )); then echo "(dry run, not submitted)"; exit 0; fi
cd "$SRC"
RESULT=$("${CMD[@]}")
echo "$RESULT"
JOB=$(echo "$RESULT" | grep -oE '[0-9]+' | head -n 1)
CLUSTER=$(echo "$RESULT" | grep -oE 'on cluster [a-z]+' | awk '{print $3}')
printf '%s\t%s\t%s\t%s\n' "$(date -u +%FT%TZ)" "$JOB" "${CLUSTER:-unknown}" "${CMD[*]}" >> "$CAMP/networks/logs/submissions.tsv"
