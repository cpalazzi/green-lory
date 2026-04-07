#!/bin/bash
# Preflight input checks for green-lory global runs on ARC.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

LOCATIONS_CSV="${1:-${ARC_LOCATIONS_CSV:-}}"
TECH_YAML="${ARC_TECH_YAML:-inputs/tech_config_ammonia_plant_2030_dea.yaml}"
INTEREST_CSV="${ARC_INTEREST_CSV:-}"
LAND_CSV="${ARC_LAND_CSV:-data/20251222_max_capacities.csv}"

if [[ ! -f "$LAND_CSV" && -f "data/20251222_land_max_capacity.csv" ]]; then
  LAND_CSV="data/20251222_land_max_capacity.csv"
fi

echo "Preflight checks"
echo "  repo:        $REPO_ROOT"
echo "  tech_yaml:   $TECH_YAML"
if [[ -n "$INTEREST_CSV" ]]; then
  echo "  interest:    $INTEREST_CSV"
else
  echo "  interest:    <none>"
fi
echo "  land_csv:    $LAND_CSV"
[[ -n "$LOCATIONS_CSV" ]] && echo "  locations:   $LOCATIONS_CSV"
echo

missing=()
for f in \
  "model/run_global.py" \
  "$TECH_YAML" \
  "$LAND_CSV" \
  "data/countries.geojson"
do
  [[ -f "$f" ]] || missing+=("$f")
done

if [[ -n "$INTEREST_CSV" ]]; then
  [[ -f "$INTEREST_CSV" ]] || missing+=("$INTEREST_CSV")
fi

WEATHER_DIR="${ARC_WEATHER_DIR:-data/weather_data}"
for pattern in "${WEATHER_DIR}/Solar*.nc" "${WEATHER_DIR}/SolarTracking*.nc" "${WEATHER_DIR}/WindPowers*.nc"; do
  if ! compgen -G "$pattern" >/dev/null; then
    missing+=("$pattern")
  fi
done

if [[ -n "$LOCATIONS_CSV" ]]; then
  if [[ ! -f "$LOCATIONS_CSV" ]]; then
    missing+=("$LOCATIONS_CSV")
  fi
fi

if [[ ${#missing[@]} -gt 0 ]]; then
  echo "ERROR: missing required inputs:" >&2
  for item in "${missing[@]}"; do
    echo "  - $item" >&2
  done
  exit 2
fi

PYTHON_BIN="${ARC_PYTHON_BIN:-python}"
if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "ERROR: python not found in PATH ($PYTHON_BIN). Load/activate ARC env first." >&2
  exit 2
fi

if [[ -n "$LOCATIONS_CSV" ]]; then
  "$PYTHON_BIN" - "$LOCATIONS_CSV" <<'PY'
import sys
import csv
path = sys.argv[1]
with open(path, newline="") as handle:
  reader = csv.DictReader(handle)
  if reader.fieldnames is None:
    raise SystemExit(f"Locations CSV has no header: {path}")
  cols = {c.lower(): c for c in reader.fieldnames}
  for req in ("lat", "lon"):
    if req not in cols:
      raise SystemExit(f"Locations CSV missing required column: {req}")
  row_count = sum(1 for _ in reader)
print(f"Locations CSV OK: {path} ({row_count} rows)")
PY
fi

echo "OK: required files are present."
