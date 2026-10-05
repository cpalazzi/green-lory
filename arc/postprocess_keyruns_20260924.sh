#!/bin/bash
# Post-processing of the 24 Sep 2026 key runs, to be run ON ARC after the global QA jobs pass and
# the legacy array has finished.  Every output goes to a new directory; nothing is overwritten.
#   1. export the two green-lory surfaces as supplier contracts (1 Mt/yr cutoff declared);
#   2. derive the 2 % land-share variant of the uniform-5 % surface and export it;
#   3. build the legacy key-run supplier table (stated method on the centred 2 % land table,
#      flat water 3.3245 USD2018/t) with its contract;
#   4. submit the four green-porpoise runs with the settings of the September comparisons.
set -euo pipefail
R="${KEYRUN_RELEASE:-/data/engs-df-green-ammonia/engs2523/green-lory-releases/20260928-keyruns-v4}"
RUN_ID="${KEYRUN_ID:-20260928-v3}"
LEG=/data/engs-df-green-ammonia/engs2523/green-lory-releases/20260924-legacy-v4
NT=/data/engs-df-green-ammonia/engs2523/green-lory-releases/20260916-network-tools-v1
ROOT=/data/engs-df-green-ammonia/engs2523/green-lory-campaigns/keyruns_20260923_v1
LANDC=/data/engs-df-green-ammonia/engs2523/green-lory-campaigns/land_center_20260923_v1
VER=/data/engs-df-green-ammonia/engs2523/green-lory-campaigns/verschuur_reconcile_20260907_v1
PY=/data/engs-df-green-ammonia/engs2523/envs/green-lory-env/bin/python
DERIVE="${DERIVE_SCRIPT:-$ROOT/tools/derive_land_share_surface.py}"
CUTOFF="${CUTOFF_MT:-1.0}"
SUBMIT_NETWORKS="${SUBMIT_NETWORKS:-1}"
EXPORTS="${EXPORTS_DIR:-$ROOT/exports_${RUN_ID}_v1}"
mkdir -p $EXPORTS
cd $R

export_surface() {  # run dir, export name
  local d="$1" name="$2"
  if [[ -f "$EXPORTS/$name/contract.json" ]]; then echo "  $name: contract exists, skipping"; return 0; fi
  test -f "$d/qa/validation.json" && grep -Eq '"status": *"passed"' "$d/qa/validation.json" || { echo "QA not passed for $d" >&2; return 2; }
  $PY reconciliation/export_lory_surface.py --surface "$d/merged/global_run_results.csv" --qa "$d/qa/validation.json" \
     --manifest "$d/manifest.json" --country-reference "$R/reconciliation/country_reference_20260928_v1.csv" --drop-unassigned \
     --output "$EXPORTS/$name" --minimum-capacity "$CUTOFF" | grep -E '"exported_positive_capacity_rows"|"excluded_positive_below_cutoff_rows"|"output_sha256"'
}

W5=$ROOT/30_keyruns/gl_dea2050_wacc5_bflat_wflat_land20c_fixed/runs/$RUN_ID/global
AM=$ROOT/30_keyruns/gl_dea2050_ameli_bflat_wflat_land20c_fixed/runs/$RUN_ID/global
echo "== 1. contracts of the two surfaces"
export_surface "$W5" gl_dea2050_wacc5_bflat_wflat_land20c_fixed
export_surface "$AM" gl_dea2050_ameli_bflat_wflat_land20c_fixed

echo "== 2. 2 % variant of the uniform-5 % surface (post hoc, capacities x 0.1)"
D2=$ROOT/30_keyruns/gl_dea2050_wacc5_bflat_wflat_land2c_fixed/runs/$RUN_ID/derived
if [[ ! -f "$D2/global_run_results.csv" ]]; then
  $PY "$DERIVE" --surface "$W5/merged/global_run_results.csv" --multiplier 0.1 --parent-share 0.20 --output-dir "$D2" | grep -E '"effective_land_competition_fraction"|"sha256"' | head -3
  mkdir -p "$D2/qa" && cp "$W5/qa/validation.json" "$D2/qa/validation.json" && cp "$W5/manifest.json" "$D2/manifest.json"
fi
if [[ -f "$EXPORTS/gl_dea2050_wacc5_bflat_wflat_land2c_fixed/contract.json" ]]; then echo "  2 % contract exists, skipping"; else
$PY reconciliation/export_lory_surface.py --surface "$D2/global_run_results.csv" --qa "$D2/qa/validation.json" --manifest "$D2/manifest.json" \
   --derived-provenance "$D2/provenance.json" --scenario-id-override gl_dea2050_wacc5_bflat_wflat_land2c_fixed \
   --country-reference "$R/reconciliation/country_reference_20260928_v1.csv" --drop-unassigned --output "$EXPORTS/gl_dea2050_wacc5_bflat_wflat_land2c_fixed" \
   --minimum-capacity "$CUTOFF" | grep -E '"exported_positive_capacity_rows"|"excluded_positive_below_cutoff_rows"|"output_sha256"'
fi

echo "== 3. legacy key run supplier table (stated method, centred 2 % land, flat water)"
LOUT=$ROOT/legacy_glannuity_20260924_v1
LTAB=$EXPORTS/ll_way2050_ameli_bflat_wflat_landleg2c_track_4h_glannuity
if [[ -f "$LTAB/gpo_export/contract.json" ]]; then echo "  legacy table exists, skipping"; else
test "$(find $LOUT -name summary.json | wc -l)" -ge 15377 || { echo "legacy array incomplete: $(find $LOUT -name summary.json | wc -l) summaries" >&2; exit 3; }
cd $LEG
$PY reconciliation/legacy_lcoa/build_legacy_supplier_table.py --run "$LOUT" --land "$LANDC/max_capacities_center_2pct_slope15.csv" --rule stated_method \
   --archived "$VER/historical/git-0a63616/data/c_NH3_cost_4.5.csv" --cells reconciliation/legacy_lcoa/cells_archived_all_v1.csv --anchor center \
   --water-cost-usd2018-per-t 3.3245 --scenario-id ll_way2050_ameli_bflat_wflat_landleg2c_track_4h_glannuity --output "$LTAB" | tail -12
cd $R
fi

if [[ "$SUBMIT_NETWORKS" == "1" ]]; then
  echo "== 4. green-porpoise runs"
  cd $NT
  for name in gl_dea2050_wacc5_bflat_wflat_land20c_fixed gl_dea2050_ameli_bflat_wflat_land20c_fixed gl_dea2050_wacc5_bflat_wflat_land2c_fixed; do
    bash arc/submit_network_run.sh --name "gpo_${name}_iso1000_1mt_v1" --contract "$EXPORTS/$name/contract.json" --clusters all 2>&1 | tail -2
  done
  bash arc/submit_network_run.sh --name gpo_ll_way2050_ameli_bflat_wflat_landleg2c_track_4h_glannuity_iso1000_1mt_v1 --contract "$LTAB/gpo_export/contract.json" --clusters all 2>&1 | tail -2
  squeue -M all -u engs2523 -o "%.14i %.8P %.60j %.8T %.10M %R" | grep -v CLUSTER
fi
