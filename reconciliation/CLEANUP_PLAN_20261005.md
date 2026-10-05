# Next steps and cleanup plan (5 October 2026)

Working checklist for review. Nothing in it has been executed. Delete this file once
the cleanup has landed. Verified against the files, ARC and the code on 5 Oct; ARC had
no queued jobs; the test suite passed (184 tests).

## 0. Corrections to the record (verified 5 Oct)

1. **Network run D is still valid.** It combined the September green-lory replication
   surface (`lcoa_land_mode = postprocess`: LCOA and designs never saw a land table) with
   capacity from the legacy land step (`legacy_land_areas.py`, centred cells). The anchor bug
   does not touch it. D is the green-lory reconciling run at network level: 261.23 USD/t,
   Australia 238.5 Mt/yr, against 260.65 / 255.3 for the archived-table control A2. The
   28 Sep handover listed D as superseded; it is not.
2. **Pilots v1-v5 never used the mislabelled table.** Configs v1/v2 pin the 40-cell centred
   table `results/campaigns/land_reconcile_20260914_v1/revised/common-geography-fixed-pilot-v1/max_capacities.csv`
   (sha 7f786de9, `cell_anchor=center`). v5 and v6 have the same land budgets
   (226.8 / 113.1 / 102.4 km²).
3. **The v5 to v6 drop in the supply ceilings is a PV-density change, not a land change.**
   v5 used the table's explicit density (83.45 MW/km² at 23°S); v6 used the run-time
   recomputation in `model/run_global.py:931-951` from the YAML base 0.01 km²/MW
   (73.60 MW/km² at 23°S), which applies whenever a land table has no
   `solar_density_method` column. Three PV density rules are in use:

   | Rule | At 23° | Used by |
   |---|---|---|
   | explicit, literal paper reading (9 km²/GW at the equator) | 83.45 MW/km² | 40-cell pilot table, pilots v1-v5 |
   | land-build column | about 78 MW/km² | legacy stated method (legacy key run) |
   | run-time recomputation from the YAML | 73.6 MW/km² | every `run_global` surface, incl. the key runs, and pilot v6 |

   RUNS.md (28 Sep) and run_store rows 12-13 give the wrong cause.
4. **The decision text and the key runs disagree.** DECISION §7.2 makes land-enforced
   finite-site capacity the method; the key surfaces use the scaled reference design at a
   20 % share (`after_solve`, `scaled_reference_design`). Finite-site curves exist only at
   the three focal cells.
5. Smaller record errors, to fix in the doc rewrite:
   - the one-degree mislabel was first recorded on 14 Sep (`land/FINDINGS_20260914.md:80-85`),
     not 23 Sep;
   - legacy array 8821144 ran from release `20260915-legacy-v2`; run_store says v3;
   - the legacy key run id uses `way2050`, while the run-store grammar defines `xcost45`
     for the x_Cost RCP4.5 sheet it read (`arc/jobs/08_legacy_global.sh:32`); pick one;
   - run_store's September central row carries the network release (20260914-v3) instead
     of the surface release (20260907-v1-c570c5153d04), and `track` where both PV types
     were offered (`both`).

## 1. Next steps, in order

| # | Step | Effort | Needs |
|---|---|---|---|
| 1 | Commit the worktree directly on `main` as area commits; one branch, no PR (decided 5 Oct) | 45 min | done 5 Oct |
| 2 | Decisions a-f (§2) | you | |
| 3 | Fix the run hazards (§5) | 1 h | 1 |
| 4 | Fetch the accepted ARC outputs (key surfaces, legacy key table, centred land tables; about 50 MB of summaries) so they exist in two places | 15 min | socket |
| 5 | Build the final evidence: `three_cells_v3`, one network digest (A2, E, D, four key networks), the centred land comparison and maps | 2-3 h | 4 |
| 6 | Optional compute: corrected run B and legacy key run at 20 % (§2f) | post-processing + 2 network jobs (4 CPUs, 2 h each, `arc`) | 2f |
| 7 | One decision document, one entry README, the legacy README with end-to-end commands and expected values | 2-3 h | 5, 6 |
| 8 | Cleanup commits (§4) and ARC pruning (§4.5) | 1-2 h | 1, 7 |

## 2. Decisions (5 Oct: a, b, d and f agreed as recommended; c agreed in principle, value open; e replaced by a run-definition file)

a. **Network acceptance gap.** Adopt the public deposit's 1.5 % and report each run's gap.
   The 11-hour form of E moved the gap from 0.59 % to 0.557 %; all runs sit at 0.48-0.89 %
   after one hour; differences between runs are 3-60 % of delivered cost. A 0.6 % threshold
   would fail A, A2, A3 and both 20 % key networks.

b. **Name the two green-lory runs.**
   - Reconciling: `rep_way2050_flat_amelired_4h_tracking_nominal_h2` (Way 2050, Ameli,
     4-hour, tracking, nominal compressor, legacy store accounting) with network D.
   - Improved: `gl_dea2050_wacc5_bflat_wflat_land20c_fixed` as base (uniform 5 %, since Ameli
     moves LCOA by under 1 %); Ameli and the 2 % share as sensitivities.

c. **One PV density rule.** Write the chosen fixed-PV density into the land table
   (`solar_density_method`) and remove the run-time recomputation, so land build, pilots,
   global surfaces and the legacy stated method read one number. The value is yours:
   73.6 MW/km² at 23° (what the key runs used), 83.45 (literal paper reading), or an
   empirical figure (Bolinger and Bolinger 2022: 87 MWdc/km², fixed tilt, 2019 US median).
   The key surfaces and the replication surface solved without land, so a change is
   post-processing only; PV-limited capacities scale with the density.

d. **Capacity rule of record for global surfaces.** Keep the scaled reference design and
   state it as a conservative lower bound; quote the three-cell finite-site curves as the
   measured bias (at 2 %: Atacama 3.4 scaled against 4.0 feasible; NW Australia 1.17
   against 1.75; central Australia 0.38 against 2.0; different cost sets, so approximate).
   Global finite-site solves are not proportionate now; a per-cell energy-ceiling bound is
   the cheap middle ground if needed. Rewrite DECISION §7.2 accordingly.

e. **Runs are defined as data, not switches (decided 5 Oct).** One run-definition file
   (for example `runs.yaml`) holds every setting of every run id: cost YAML, plant bundle,
   finance CSV, land table and share, land constraint and capacity rule, allocation, PV
   policy, time step, temporal accounting, ramp basis, site costs, water. The wrapper takes a
   run id and reads its settings; the run manifest records the resolved values. The
   reconciling run is one entry (4-hour step, `legacy_scaled`, `legacy_per_snapshot`, site
   costs out of the headline, nominal compressor, tracking PV), the improved run another.
   `run_store.csv` stays the execution record (jobs, release, status, outputs) under the same
   ids. This replaces the `case` presets in `arc/submit_lory_sequence.sh:240-330`; the
   `oat_*` and both `central_*` presets are not carried over. Model options stay as the
   mechanism; remove only those that reproduce a bug or read obsolete inputs: the
   `southwest` anchor (keep it in the anchor test), the conservative union fallback, the grid
   backstop, the `aggregation_count` plumbing, the legacy `max_*` column family (after
   editing the QA gate), and the run-time PV-density recomputation once (c) is settled.
   `exclusive` allocation, the in-solve land constraint and solved-quantity capacity stay.

f. **Optional symmetry runs.** Recommend one: the corrected run B (reconciling costs with the
   green-lory land method on the centred 2 % table). B is the most-cited divergent network
   and currently rests on the mislabelled table. The legacy key run at 20 % is optional.
   Both are post-processing plus one network hour each.

## 3. Target layout

```
README.md          current model and pipeline; links reconciliation/README.md
GLOSSARY.md        absorbs RUN_STORE.md (grammar, tokens, statuses, network run letters)
model/             green-lory (dead functions removed)
legacy_lcoa/       frozen 94de8ce source, harness, weather store, land step (+ land/core.py),
                   capacity rule, supplier table, environment, README with commands and
                   expected values (optional move up from reconciliation/)
arc/               land-centre, lory-sequence, network, generic post-processing wrappers;
                   jobs 00, 01, 03, 06, 07, 08, 09; README rewritten
reconciliation/    README (entry) · DECISION.md · RUNS.md + run_store.csv ·
                   current analysis scripts · supply_pilot/ · archive/2026-09/ (unedited)
tests/             current code only
results/reconciliation_final_<date>/   curated evidence (§4.3)
data/              inputs (+ countries.geojson)
```

Moving `legacy_lcoa/` to the top level maps the brief ("legacy directory", "model
directory") literally; it costs path updates in jobs 07-09, tests and docs.

## 4. Keep / archive / delete

Everything is committed first (step 1), so code and documents removed later remain in git
history. "Archive" means `reconciliation/archive/2026-09/` for documents cited as evidence;
superseded code is deleted.

Step 1 commits, by area, each with its own tests: `.gitignore`; model and its inputs;
ARC campaign tooling and campaign inputs; reconciliation analysis, network replay and land
pilots; the legacy-lcoa package; documents and run store. The generated interest CSVs
(47 MB) stay out of git (§4.4).

### 4.1 Code

**Delete (dead or scratch)**
- `model/plot_monthly_results.py`
- dead functions: `run_global.py` (`_interest_overrides`, `_match_land_row`,
  `_extract_capacity_caps_from_row`, `_apply_component_caps`, `_apply_land_caps`,
  `_apply_land_caps_fast`, `_estimate_land_used_km2`); `auxiliary.py`
  (`extra_functionalities`, the four `_nh3_ramp_*` and two `_penalise_ramp_*`,
  `pyomo_constraints`, `pyomo_operating_constraints`, `get_solving_info`,
  `convert_network_to_operating`, `linopy_operating_constraints`)
- `legacy_lcoa/debug_one.py` (keep `debug_two.py`, cited), `legacy_lcoa/environment.legacy-min.yaml.bak`,
  `arc/jobs/07_build_legacy_env.sh.bak` (both pin Pyomo 6.6.2, the zero-objective version)
- `legacy_lcoa/source/cd56c11/` (June 2023 code that does not replicate; both commits stay in
  github.com/cpalazzi/lcoa-opt); set `run_legacy_cells.py --era` default to `may2023` and drop
  the exploratory variants
- `arc/`: `jobs/02_notification_test.sh`, `jobs/04_*`, `jobs/05_*`, `resubmit_land_constraints_job.sh`,
  `stage_and_submit_land_constraints.sh`, `submit_land_constraints_matrix.sh` (July matrix),
  `submit_global_run.sh`, `submit_constrained_reruns.sh`; `arc_initial_setup.sh` (or rewrite for releases);
  `scripts/merge_global_results.py`
- `reconciliation/`: `audit_land_and_physics.py`, `check_completed_campaign.py` (broken),
  `prepare_network_surfaces.py` (broken), `land_share_sensitivity.py`, `plot_supplier_comparison.py`,
  `compare_network_runs.py` (fold its selected-supplier statistics into `compare_networks.py`);
  `build_three_cell_comparison.py` and `plot_land_capacity_maps.py` after their replacements exist
- `reconciliation/land/`: everything except `core.py` (move into `legacy_lcoa/`),
  `run_supply_pilot.py` and `supply_curve/config_v3_center*.json`; first move `TECH_YAML` and
  `tech_yaml_dependencies` out of `stage_pv_pilot.py`. Keep the `fixed_pv_3cells_v2` output (hashed
  weather input of every pilot).
- `legacy_lcoa/` diagnostics `analyze_sample_capacity.py`, `test_capacity_hypotheses.py`,
  `compute_annual_cf.py`: delete or move to `legacy_lcoa/diagnostics/`. Keep
  `legacy_capacity.py apply-lory` (it rebuilds D's contract).
- `scripts/`: the six ignored one-offs (`check_weather_coverage`, `diagnose_missing`,
  `make_missing_cells_csv`, `run_3h_comparison`, `plot_gridless_maps`,
  `build_depth_altitude_cost_inputs`); keep `build_ameli_wacc_inputs.py` (provenance of the Ameli
  input) and `fetch_travel_time.py` only if the spatial runs return
- `notebooks/`: duplicates `02_single_site_run` and `03_global_run`; the pre-reconciliation
  `01`, `03_single_site_run`, `04`, `05`; fix or delete `00`
- `tests/`: `test_compare_pv_pilot`, `test_model_evidence`, `test_supply_certificate`,
  `test_supply_energy_bound`, `test_native_download_plan`, `test_native_modis`,
  `test_native_pilot`, the `pilot_joint_masks` half of `test_land_reconstruction`; rename
  `test_paper_ammonia_capacity.py`
- plant bundles: keep `basic_ammonia_plant_2050` (key runs), `_2050_way_tracking` (pilots,
  replication), `_2050_way` (lcoa_validation symlink); delete `basic_ammonia_plant` (2030,
  10 Mt) after switching the `main.py` and `run_global` defaults

**Small refactors (optional):** one solver-retry chain (`run_supply_pilot` reuses
`run_global`'s), one YAML-`extends` resolver (three copies), a generic
`arc/postprocess_campaign.sh` with required arguments.

### 4.2 Documents

- **Keep and rewrite:** `README.md`; `GLOSSARY.md` (absorbs `RUN_STORE.md`);
  `reconciliation/README.md` (entry point); new `reconciliation/DECISION.md` (replaces
  `DECISION_20260916.md` and its post-scripta, folds in LAND_SHARE_STATEMENT §5-6, adds the
  §6 table, the run definitions and decisions a-d); `RUNS.md` with an index from
  `run_store.csv`, absorbing `land/RUNS.md`, with rows added for the networks, the September
  replication surface and pilots v1-v5; `legacy_lcoa/README.md` (absorbs what is still valid in
  PROVENANCE, LEGACY_REPLICATION §3 and §8 and GLOBAL_SURFACE; adds the key run, the network
  step, environment notes and expected values); `arc/README.md`; `data/README.md` (lines 50-54).
- **Archive unedited:** `REPORT_20260914`, `FINDINGS`, `HANDOVER_20260915`,
  `ATTRIBUTION_20260916`, `PLAN_GLOBAL_CAMPAIGN_20260916` (after noting which S-runs happened),
  `LAND_SHARE_STATEMENT_20260916`, `notifications`, `DECISION_20260916`,
  `legacy_lcoa/PROVENANCE.md`, `legacy_lcoa/LEGACY_REPLICATION_20260915.md`, all `land/*.md`,
  `results/INVENTORY_20260916.*`; `DEVELOPMENT_NOTES.md` (pre-reconciliation log).

### 4.3 Local results (9.6 GB, of which about 2 MB current)

**Keep, in one curated folder**
- networks: archival replay 8758215, A2 8805879, E 8823893 (and its 11-hour form 8823894 as gap
  evidence), D 8820849, the four key networks 13296857/60/63/65
- legacy baseline: `legacy_lcoa_20260915_v1/arc_received_global_archived_v2` without the
  extracted `surface/` (370 MB; it is the content of the kept 32 MB tarball);
  `audit/global-supplier-table-arc-v1`, `audit/legacy-land-areas-v1`, `audit/arc-vs-mac-v2`;
  `three_cells_may2023_xcost45_tracking_v1`
- reconciling surface: `verschuur_reconcile_20260907_v1/comparison/global-20260914-v2/revalidated/rep/`
  and D's contract `lory/global_exports/20260915-v1/rep_legacy_rule/`
- pilots v6 (`arc_received_colocated_v6_center`), their weather input (`arc_received_fixed_v2`)
  and control LCOAs (`arc_received_fixed_threshold_v2`)
- inputs: `historical/git-0a63616`, `historical/mendeley-v1`; move `countries.geojson` to `data/`
- `reconciliation_final_20260916_v1/legacy_surface/` and its figure

**Remove (about 9 GB)**
- superseded (5.4 GB): Mac weather store (3.0 GB, rebuildable), Mac global run (0.37 GB),
  sample563 (0.40 GB), rejected three-cell configurations, the September central surface and
  the comparison copies (`global-20260914-v1`, `-v2` except `rep`), land-physics,
  network-attribution, networks v1/v3, runs B, C, F and the land-share networks, pilots v1-v5
  and attempts, land audit and source snapshots (0.31 GB), `interim_center_20260923`,
  `land_share_sensitivity_v1`, the September maps
- pre-reconciliation (3.6 GB): `results/archive/` (1.2 GB) and the 32 May-era top-level
  directories; five of them only after green-dolphin-paper is repointed (§4.6)

Optionally write one compressed tarball with a sha256 manifest to ARC
`green-lory-archive/` before deleting (ARC has no backup, so it is a convenience copy).
Skip the Mac weather store and the Mac global run.

### 4.4 data/ and inputs/

- `data/`: keep `weather_data/` (12.7 GB), the MODIS HDF (1.2 GB) and `model_bathymetry.nc`
  (also linked from lcoa-opt and lcoa_validation). GEBCO (7.0 GB) is linked from
  pypsa-earth-green-auklet: keep, or move it there. WDPA (6.2 GB) is used only by the ARC
  land build, which has its own copy: delete locally if space matters. Remove the unused
  `cutouts/` link. Add `countries.geojson`.
- `inputs/`: the new interest CSVs (`uniform_interest_inputs_0p05_2050.csv`,
  `amelired_interest_inputs_2050_center.csv`, 47 MB) are generated by
  `arc/build_finance_overrides.py` and hashed in the release manifests; keep them out of git.
  The four tracked `spatial_cost_inputs*.csv` (274 MB) stay until the spatial runs are
  decided. Unreferenced: `tech_config_ammonia_plant_2050_way_usd.yaml`, the `amelired_wacc_*`
  map CSVs, `filtered_dea.csv` at the root; scratch: `arc_exchange/`, `Claude outputs/`, `logs/`.

### 4.5 ARC (491 GB of 5 TB used: pruning is for clarity, not space)

- Releases: keep the ten behind accepted results: `20260907-v1-c570c5153d04` (replication
  surface), `20260914-v3` and `20260916-network-tools-v1` (network code),
  `20260915-legacy-env-v1`, `20260915-legacy-v2` (legacy global surface),
  `20260923-land-center-v1`, `20260924-legacy-v4` (legacy key run), `20260928-keyruns-v4`
  (pilots v6), `-v6` (key surfaces), `-v8` (contracts). Delete the other 22 and the two loose
  tarballs (files shared by hardlink survive in the kept releases).
- Campaigns: keep `keyruns_20260923_v1` (drop the probes and failed run directories),
  `land_center_20260923_v1`, `legacy_lcoa_20260915_v1/global_archived_v2` and its tarball, and
  the accepted parts of `verschuur_reconcile_20260907_v1` and `land_reconcile_20260914_v1`
  (pilots v6, `fixed_pv_3cells_v2`, `fixed_land_threshold_3cells_v2`). Remove
  `lory_reconcile_20260722_v1` (July build on the mislabelled table) and the superseded pilots
  and networks.
- `green-lory/` (old mutable clone, code at e258da0 of March 2026): **keep `green-lory/data/`**
  (31 GB): every pinned release links its weather, MODIS, GEBCO, WDPA, bathymetry and
  `countries.geojson` there (checked 5 Oct). Remove `green-lory/results/` (6.8 GB of May-era
  runs) and the superseded `max_capacities*.csv` tables in `green-lory/data/`; leave the rest of
  the checkout alone or reduce it to `data/` once nothing else points into it.
- Size report: `~/glr_du_20261005.txt` on ARC (written by a detached scan).

### 4.6 Outside this repository (repoint before deleting)

- green-dolphin-paper: `sync_lcoa_from_green_lory.py`, `run_way_2050_flat_reference.py`,
  `run_green_lory_max_capacity_scenarios.py` read `results/{dea_2030_flat, dea_2050_flat,
  way_2050_flat, way_2050_flat_paper_2pct_slope15, way_2050_flat_high_50pct_slope15}` by
  absolute path. These May surfaces predate the land fix. Repoint them to a stable surfaces
  index (`results/surfaces/<run_id>/` plus an index CSV) built from the accepted runs.
- lcoa_validation (`greenlory_way_bundle` → `basic_ammonia_plant_2050_way`, `way_eur.yaml`,
  `harness/gl_driver.py`), the lcoa-opt data links → `data/`, and pypsa-earth-green-auklet →
  `data/land/GEBCO_2025_sub_ice.nc`: keep these targets in place.
- green-porpoise has uncommitted April-July edits (`gpo/*.py`, `run_scenarios.py`,
  `data/c_NH3_cost_2.6.csv` overwritten with 2050 technology-year data, a `.bak`). The
  reconciliation ran from frozen copies, so they do not affect it; yours to commit or revert.

## 5. Hazards to fix before any new run

1. Defaults still point at the mislabelled south-west table: `model/data_paths.MAX_CAPACITIES_FILE`
   (hence `run_global`'s default), `arc/jobs/01_run_global.sh:101`, `arc/arc_check_run_inputs.sh:11`,
   `arc/submit_lory_sequence.sh:20-21`. Make the land table a required argument.
2. Nothing checks `cell_anchor`: add a `center` gate to `arc/validate_land_campaign_input.py`
   and the campaign QA.
3. `arc/jobs/06_land_supply_pilot.sh` and `run_supply_pilot.py` default to `config_v1.json`,
   which the code now rejects (`technology_shared`); the job limit is 2 h where v6 needed 8-9 h.
4. `arc/postprocess_keyruns_20260924.sh` defaults to run `20260928-v3` and release v4; the
   accepted post-processing used run `20260928-v4` with releases v7/v8.
5. `run_legacy_cells.py` defaults to `--era june2023`; the replicating configuration is `may2023`.
6. Two capacity column families coexist (`max_*` from separate per-technology caps, written
   but unused downstream; `scaled_design_*`, used by the contracts), and the QA gate requires
   the first. Document or drop one.

## 6. The three focal cells today (seed for three_cells_v3)

LCOA in USD2018/t; capacity in Mt/yr at the stated land share.

| Run | Atacama (−23, −69) | NW Australia (−23, 117) | Central Australia (−21, 135) |
|---|---|---|---|
| Archived table | 212.92; 9.77 | 233.69; 4.02 | 226.39; 4.32 |
| Legacy replication (surface behind E) | 221.33; 9.43 | 242.64; 3.97 | 234.04; 2.42 |
| green-lory reconciling, legacy rule (surface behind D) | 215.54; 9.10 | 235.51; 3.80 | 220.23; 0.91 |
| Legacy key run (x_Cost CAPEX, green-lory annuity, stated land 2 % centred, water) | 203.41; 5.17 | 221.57; 2.18 | 214.60; 1.52 |
| green-lory improved, uniform 5 %, 20 % | 363.94; 34.27 | 389.26; 11.65 | 320.13; 3.82 |
| green-lory improved, Ameli, 20 % | 367.01; 34.27 | 392.56; 11.65 | 322.80; 3.82 |
| green-lory improved, uniform 5 %, 2 % | 363.94; 3.43 | 389.26; 1.17 | 320.13; 0.38 |
| Finite-site, Way costs, fixed PV, 2 % (pilot v6): unconstrained cost / ceiling at its cost, EUR2020/t | 234.7 / 4.0 at 238.9 | 251.8 / 1.75 at 254.1 | 213.2 / 2.0 at 244.8 |

Sources: DECISION §4 (archived, legacy replication, D surface), the key network inputs in
`verschuur_reconcile_20260907_v1/arc_received_20260928_keyruns/*/selected_supplier_inputs.csv`
and `land_reconcile_20260914_v1/arc_received_colocated_v6_center/supply_curve_v6_fixed_center/supply_points.csv`
(re-read 5 Oct).

Approximate split of the improved run's premium over the legacy replication (+37 to +64 %),
using the pilot v6 unconstrained cost as the intermediate step (Way 2050 costs, hourly
green-lory plant, fixed PV, explicit compressor, DEA tank; Ameli, no water): plant model
+18 / +15 / +1 % (260 / 279 / 236 USD2018/t), DEA 2050 cost data +40 / +39 / +36 % on top.
Finance and water differ slightly between the steps.
