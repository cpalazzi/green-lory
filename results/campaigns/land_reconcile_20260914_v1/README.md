# Land reconciliation campaign — 14 September 2026

This campaign separates source-substituted paper-method reconstruction from
the revised-model candidate. It does not replace the production land table or
the archived historical supplier surface.

## Accepted results

| Directory | Contents and interpretation |
|---|---|
| `audit/v3/` | Corrected audit of current land inputs and historical capacity outliers |
| `audit/global-unmasked-v2/` | Global class-only bound, without protected/slope exclusions; frozen plant designs |
| `arc_received_v1/` | Completed centered and southwest-cell joint-mask pilots; raw ARC results and logs |
| `audit/joint-comparison-v1/` | Independently validated 40-cell land comparison and frozen-design packing sensitivities |
| `replication/cmg-paper-method-pilot-v1/` | Validated 40-cell paper-method input with substituted source datasets |
| `revised/common-geography-fixed-pilot-v1/` | Separately labelled 40-cell common-geography fixed-PV candidate input |
| `arc_received_fixed_v2/` | Completed fixed-PV job 8809820, raw results, exact weather inputs, manifest and logs |
| `audit/fixed-vs-tracking-v1/` | Independently validated actual fixed-PV versus central-design comparison on common land |
| `audit/native-modis-pilot-v2/` | Accepted three-cell native class/joint-mask comparison with refined convergence checks; v1 retained as initial resolution test |
| `replication/native-paper-method-pilot-v1/` | Validated three-cell native source-substituted paper-method land input |
| `revised/native-common-geography-pilot-v1/` | Separately labelled validated native common-geography candidate |
| `audit/supply-checkpoints-v4-20260915T1001BST/` | Three controls and 16 constrained points locally validated; partial snapshot, not full-run acceptance |
| `audit/supply-checkpoints-v4-final/` | Final 21 checkpoint records validated: three controls, 16 feasible and two solver-infeasible quantities |
| `audit/supply-certified-grid-v1/` | All 21 planned quantities classified; missing three endpoints closed by explicit analytical energy certificates, not relabelled solver outcomes |

The two land inputs deliberately share corrected geography at this controlled
stage. Neither is an exact historical-source replay or a promoted global map.
The earlier `joint-comparison-v1` packing sensitivities retain the old
optimized tracking generation and are not fixed-PV re-optimizations. The
completed `fixed-vs-tracking-v1` comparison now includes actual fixed-PV
re-optimizations as well as frozen-central-design packing sensitivities.

## Runs and source snapshots

- `sources/pilot-release-v1/`: immutable source for the two completed mask jobs.
- `sources/fixed-pv-release-v1/`: preserved first plant-pilot release, missing
  an inherited technology YAML; job 8807144 failed before optimization.
- `arc_received_fixed_attempt_v1/`: preserved failed-attempt weather, manifest
  and logs. No plant costs or capacities were produced by that attempt.
- `sources/fixed-pv-release-v2/`: corrected release with recursive configuration
  dependency checks and an early preflight mode.
- `sources/supply-curve-release-v1/` and `v2/` (full prefix retained in each
  directory name): preserved cost–quantity attempts. The first failed at the
  output adapter; the second stopped when a numerical solution failed land QA.
- `arc_received_supply_attempt_v1/` and `arc_received_supply_attempt_v2/`:
  failed-attempt raw files and logs. Completed checkpoints are not promoted as
  a complete, accepted supply curve.
- `sources/supply-curve-release-v3/`: strict optimal-termination check and
  recorded homogeneous-barrier retry; job 8811980 failed when numerical
  terminations remained unresolved. Raw files are preserved under
  `arc_received_supply_attempt_v3/`.
- `sources/supply-curve-release-v4/`: dual-simplex numerical fallback,
  submitted as job 8811997 on 15 September; no physical inputs changed.
- `sources/native-modis-download-plan-20260915-v1/`: official NASA catalogue
  metadata and precise three-file URLs for the native-resolution comparison.
- `sources/native-modis-c61-2022/`: three received and verified unmodified HDF sources.
- `sources/native-pilot-release-v1/`: 19-file preserved native-pilot code,
  configuration and test release, verified against accepted output provenance.
- `arc_received_supply_snapshots_v4/20260915T1001BST/`: fetched partial v4
  snapshot, preserved separately from completed or failed runs.
- `arc_received_supply_attempt_v4/`: final raw files and logs from job 8811997,
  which timed out at 11:06:31 BST; scheduler status is recorded alongside.
- `sources/native-modis-40cell-download-plan-v1/`: 40 catalogue queries and
  the 12 additional native tiles needed (52.7 MiB), excluding existing files.
- `sources/fixed-land-threshold-release-v1/`: pinned 58-file fixed-only,
  enforced-land 1 Mt/year experiment; submitted as 8812986. This release also
  preserves the standalone certificate and plotting code for the v4 analysis;
  its original model/helper dependencies remain in `sources/supply-curve-release-v4/`.
- `audit/superseded/`: retained development attempts; not accepted results.

Run IDs, statuses, resource requests and notification settings are recorded in
[the run log](../../../reconciliation/land/RUNS.md). Scientific interpretation
and the remaining validation gates are in
[the land report](../../../reconciliation/land/FINDINGS_20260914.md).
The [completed fixed-PV report](../../../reconciliation/land/FIXED_PV_20260914.md)
sets out the three-site results and their implications.
The [dataset guide](../../../reconciliation/land/DATASETS_20260915.md) separates
the received three-file download from historical-source recovery and later
global-resolution sensitivities.
The [interim supply report](../../../reconciliation/land/SUPPLY_CURVE_20260915.md)
documents locally rechecked checkpoints without promoting the incomplete
21-point curve. Job 8811997 was still running at 10:03 BST on
15 September.
The [native land comparison](../../../reconciliation/land/NATIVE_MODIS_20260915.md)
reports the completed spatial-resolution test and its limits.
The [certified quantity analysis](../../../reconciliation/land/SUPPLY_CERTIFIED_20260915.md)
supersedes the interim supply-run status and closes the planned quantity grid.

Directories are versioned and preserved. The campaign date follows the ARC
clock; a later date typed in a connection message does not relabel these runs.

## 15 September model-evidence handover and completed fixed threshold

The [handover](../../../reconciliation/HANDOVER_20260915.md) is the current
investigation entry point. It distinguishes green-lory from legacy-lcoa and
the frozen shipping input, locates both 2019 weather stacks, and explains
reference-design scaling versus finite-site production.

- `audit/model-evidence-v2/`: current capacity, CF, LCOA and HHV/LHV comparison;
  explicitly conditional legacy-design reconstruction. Earlier v1 is retained.
- `sources/fixed-land-threshold-release-v2/`: pinned dependency-gated retry.
- `arc_received_fixed_threshold_attempt_v1/`: failed 8812986 manifest/logs.
- `arc_received_fixed_threshold_v2/`: completed 8814788, six accepted solves.
- `audit/fixed-threshold-v2/`: independent full-run audit.
- `audit/fixed-threshold-checkpoints-v2/`: six physical/cost checkpoint checks.
- `arc_received_alternative_weather_v2/`: six full-year profiles from the
  separate green-condor 2019 CF tiles, with source metadata and profile hashes.
- `audit/alternative-weather-v1/`: locally verified old/new raw CF comparison;
  not an alternative-weather plant optimisation or global-completeness claim.

Both land stacks remain separate and no global default was promoted.
