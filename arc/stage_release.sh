#!/bin/bash
# Stage an immutable source release on ARC from this checkout (run on the Mac).
#
#   bash arc/stage_release.sh <release-id> [--allow-dirty] [--dry-run]
#
# The release is green-lory-releases/<release-id>/ on ARC: the Git-visible source
# (tracked plus untracked-but-not-ignored files, without results/, the spatial cost
# CSVs and the weather data), the generated finance overrides that runs.yaml pins
# (gitignored, so copied explicitly after checking their sha256), source_inventory.json
# (arc/release.py), RELEASE_NOTES.md (commit, dirty state, extras), and data/ links to
# the shared inputs in green-lory/data/. On ARC the copy is verified against the
# inventory in a short Slurm job (log: release_verification-<job>.out in the release;
# nothing computational runs on the login node), then made read-only. An existing
# release id is refused.
#
# Submit from the release afterwards, e.g.
#   cd <release> && ARC_RELEASE_INVENTORY=$PWD/source_inventory.json \
#     bash arc/submit_lory_sequence.sh --stage smoke ...
#
# Environment: ARC_HOST (default arc-oxford), ARC_CONTROL_PATH (default
# /tmp/arc-green-lory.sock), ARC_WORK_BASE (default /data/engs-df-green-ammonia/engs2523).
set -euo pipefail

usage() { sed -n '2,20p' "$0" | sed 's/^# \{0,1\}//'; }

RELEASE_ID=""; ALLOW_DIRTY=0; DRY_RUN=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --allow-dirty) ALLOW_DIRTY=1; shift;;
    --dry-run) DRY_RUN=1; shift;;
    -h|--help) usage; exit 0;;
    -*) echo "unknown option $1" >&2; exit 2;;
    *) [[ -z "$RELEASE_ID" ]] || { echo "one release id only" >&2; exit 2; }; RELEASE_ID="$1"; shift;;
  esac
done
[[ "$RELEASE_ID" =~ ^[0-9]{8}-[a-z0-9-]+$ ]] || { echo "release id must look like 20261006-keyruns-v1" >&2; exit 2; }

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
HOST="${ARC_HOST:-arc-oxford}"
SOCKET="${ARC_CONTROL_PATH:-/tmp/arc-green-lory.sock}"
BASE="${ARC_WORK_BASE:-/data/engs-df-green-ammonia/engs2523}"
RELEASE="$BASE/green-lory-releases/$RELEASE_ID"
SHARED_DATA="$BASE/green-lory/data"
ARC_PY="$BASE/envs/green-lory-env/bin/python"
PY="${PYTHON:-$REPO_ROOT/.venv/bin/python}"
SSH=(ssh -o "ControlPath=$SOCKET" -o BatchMode=yes "$HOST")

COMMIT="$(git rev-parse HEAD)"
DIRTY="$(git status --porcelain)"
if [[ -n "$DIRTY" && $ALLOW_DIRTY -eq 0 ]]; then
  echo "working tree has uncommitted changes; commit first or pass --allow-dirty (recorded in RELEASE_NOTES.md)" >&2
  exit 3
fi

"${SSH[@]}" true || { echo "no ARC connection through $SOCKET; open the socket first" >&2; exit 4; }
if "${SSH[@]}" test -e "$RELEASE"; then echo "refusing to overwrite $RELEASE" >&2; exit 5; fi

STAGE="$(mktemp -d)"
trap 'rm -rf "$STAGE"' EXIT

# Generated finance overrides pinned by runs.yaml: copied when gitignored, after a hash check.
"$PY" - "$STAGE/extras.txt" <<'PY'
import hashlib, subprocess, sys
from pathlib import Path
from model.runs import load_runs

extras = []
for run_id, run in load_runs().items():
    path, expected = run.get("finance_csv"), run.get("finance_csv_sha256")
    if not path or not expected or path in extras:
        continue
    ignored = subprocess.run(["git", "check-ignore", "-q", path]).returncode == 0
    if not ignored:
        continue  # tracked or untracked-visible: already in the inventory
    if not Path(path).is_file():
        sys.exit(f"{run_id}: {path} is missing; rebuild it with python -m model.finance (see its --help)")
    digest = hashlib.sha256(Path(path).read_bytes()).hexdigest()
    if digest != expected:
        sys.exit(f"{run_id}: {path} has sha256 {digest}, runs.yaml pins {expected}")
    extras.append(path)
Path(sys.argv[1]).write_text("".join(f"{p}\n" for p in extras))
PY

"$PY" -m arc.release write --root "$REPO_ROOT" --output "$STAGE/source_inventory.json" \
  --exclude-prefix inputs/spatial_cost_inputs --exclude-prefix data/weather_data/ >/dev/null
"$PY" - "$STAGE/source_inventory.json" "$STAGE/files.txt" <<'PY'
import json, sys
entries = json.load(open(sys.argv[1]))["entries"]
open(sys.argv[2], "w").write("".join(e["path"] + "\n" for e in entries))
PY
cat "$STAGE/extras.txt" >> "$STAGE/files.txt"
TREE="$("$PY" -c "import json,sys; print(json.load(open(sys.argv[1]))['tree_sha256'])" "$STAGE/source_inventory.json")"

{
  echo "# Release $RELEASE_ID"
  echo
  echo "- staged: $(date -u +%Y-%m-%dT%H:%M:%SZ) from $(hostname -s)"
  echo "- commit: $COMMIT"
  if [[ -n "$DIRTY" ]]; then
    echo "- working tree: dirty (--allow-dirty), status digest $(printf '%s' "$DIRTY" | shasum -a 256 | cut -c1-16)"
  else
    echo "- working tree: clean"
  fi
  echo "- source inventory tree sha256: $TREE ($(wc -l < "$STAGE/files.txt" | tr -d ' ') files incl. extras)"
  echo "- extras outside the inventory (sha256 checked against runs.yaml):"
  while read -r extra; do [[ -n "$extra" ]] && echo "  - $extra $(shasum -a 256 "$extra" | cut -d' ' -f1)"; done < "$STAGE/extras.txt"
  echo "- data/ links: countries.geojson, model_bathymetry.nc, weather_data -> $SHARED_DATA"
} > "$STAGE/RELEASE_NOTES.md"

echo "release:   $RELEASE"
echo "commit:    $COMMIT$([[ -n "$DIRTY" ]] && echo ' (dirty)')"
echo "files:     $(wc -l < "$STAGE/files.txt" | tr -d ' ') (tree $TREE)"
if (( DRY_RUN )); then cat "$STAGE/RELEASE_NOTES.md"; echo "(dry run, nothing copied)"; exit 0; fi

rsync -a --files-from="$STAGE/files.txt" -e "ssh -o ControlPath=$SOCKET -o BatchMode=yes" "$REPO_ROOT/" "$HOST:$RELEASE/"
rsync -a -e "ssh -o ControlPath=$SOCKET -o BatchMode=yes" "$STAGE/source_inventory.json" "$STAGE/RELEASE_NOTES.md" "$HOST:$RELEASE/"
"${SSH[@]}" "set -euo pipefail; cd '$RELEASE'; mkdir -p data
  for item in countries.geojson model_bathymetry.nc weather_data; do
    test -e '$SHARED_DATA/'\$item || { echo \"missing shared input $SHARED_DATA/\$item\" >&2; exit 6; }
    ln -s '$SHARED_DATA/'\$item data/\$item
  done
  sbatch --wait --partition=short --time=00:20:00 --cpus-per-task=1 --mem=4G \
    --mail-user=carlo.palazzi@eng.ox.ac.uk --mail-type=BEGIN,END,FAIL \
    --job-name=glr-verify-release --output='$RELEASE/release_verification-%j.out' \
    --wrap=\"cd '$RELEASE' && '$ARC_PY' -m arc.release verify --root . --inventory source_inventory.json\"
  cat '$RELEASE'/release_verification-*.out
  chmod -R a-w ."
echo "staged and verified: $RELEASE"
