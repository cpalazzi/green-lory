# Reconciliation package: legacy-lcoa versus green-lory, and the Verschuur network

This directory is the entry point for the reconciliation of the old `lcoa-opt`
plant model ("legacy-lcoa", Salmon 2023) with the current PyPSA implementation
("green-lory", `model/`), and of the green-porpoise shipping networks that each
produces. The decision and the consolidated comparison are in
[DECISION_20260916.md](DECISION_20260916.md).

## Endpoints

1. Paper-method reproduction: Verschuur et al. (2024) MOD-AMB, RCP4.5, 70 %
   ammonia adoption, Way 2050 costs, no subsidies. The user confirmed this as the
   primary endpoint on 7 September 2026. The archived RCP4.5/90 % project
   network is replayed separately; it is not a published scenario.
2. Revised estimate: a newly generated green-lory surface and green-porpoise
   network, with controlled comparisons that explain the effects of plant
   physics, land capacity, supplier selection and transport assumptions.

A matching map alone is not a replication test. Delivered cost, production
and demand, production by country and Australian subregion, supplier
identities, route feasibility and solver gap are compared, with tolerances
that account for the gap and alternative optima.

## Package map

| Where | What |
|---|---|
| [`legacy_lcoa/`](legacy_lcoa/README.md) | The legacy-lcoa model end to end: frozen source at two commits, environment, weather store, run harness, the recovered land/capacity step, supplier-table builder, cross-check tool; provenance and replication reports |
| `../model/` | The green-lory model (PyPSA/linopy): `main.py` single site, `run_global.py` global sweep, `land_capacity.py`/`land_union.py` land accounting, `result_accounting.py` reporting; inputs in `../inputs/`, compiled plant bundles in `../basic_ammonia_plant*/` |
| [`land/`](land/README.md) | Land reconstruction: paper-method replication stack and the corrected ("revised") stack; finite-site supply-curve and fixed-PV experiments; native MODIS checks |
| `run_historical_network.py`, `sparse_network.py`, `network_heuristic.py` | Green-porpoise network replay with pinned equations (archived project or public deposit), explicit supplier contracts and QA |
| `export_lory_surface.py`, `prepare_network_surfaces.py` | Supplier contracts from QA-passed green-lory surfaces |
| `check_completed_campaign.py`, `audit_land_and_physics.py` | Revalidation of downloaded surfaces and independent land/physics checks |
| `compare_networks.py` | Quantitative comparison of network runs (cost, gap, countries, Australian subregions, supplier overlap, focal cells) |
| `build_three_cell_comparison.py` | One table of every accepted result at the three focal cells with source hashes |
| [`PLAN_GLOBAL_CAMPAIGN_20260916.md`](PLAN_GLOBAL_CAMPAIGN_20260916.md) | State of the green-lory model, recommended implementation, scenario matrix and steps for the global campaign |
| [`LAND_SHARE_STATEMENT_20260916.md`](LAND_SHARE_STATEMENT_20260916.md) | Why a higher land share or lower admission cutoff is defensible, and how the paper's nominal 2 % produced its results |
| [`RUNS.md`](RUNS.md) | Run register: every ARC job, its purpose and whether its QA passed |
| [`HANDOVER_20260915.md`](HANDOVER_20260915.md), [`REPORT_20260914.md`](REPORT_20260914.md), [`FINDINGS.md`](FINDINGS.md) | Dated reports; earlier evidence is not silently revised |
| `../arc/` | ARC job templates and wrappers (`submit_lory_sequence.sh` for green-lory surfaces, `submit_network_run.sh` for network runs, `jobs/08_legacy_global.sh` and `jobs/09_extract_weather_store.sh` for the legacy surface) |

Generated evidence lives under `../results/campaigns/`, never here:

| Campaign | Contents |
|---|---|
| `legacy_lcoa_20260915_v1/` | Legacy three-cell, 563-cell sample and global (ARC + Mac) runs; weather stores; `audit/` supplier tables, land areas and cross-checks |
| `verschuur_reconcile_20260907_v1/` | Frozen historical inputs and public deposit; green-lory replication and central surfaces; supplier contracts; network runs and comparisons |
| `land_reconcile_20260914_v1/` | Land pilots, finite-site supply grid, fixed-PV threshold, native MODIS and alternative-weather evidence |
| `reconciliation_final_20260916_v1/` | The consolidated three-cell table and network comparison behind the decision |

## Three focal cells

Atacama (−23, −69), northwest Australia (−23, 117) and central Australia
(−21, 135): the cells whose changed capacity and cost inputs drove the
divergence of the green-porpoise supplier distribution. Their consolidated
evidence is `results/campaigns/reconciliation_final_20260916_v1/three_cells_v2/`.

## Preservation and naming

Existing green-lory, green-porpoise and paper working files remain untouched
unless a specific, tested change is needed. RCP-named supplier tables are not
overwritten with technology-year data. Every run receives a new directory with
a manifest and input hashes; nothing is promoted without a passed QA record.
Original source data are read-only inputs. `glr` in ARC job names is the
green-lory reconciliation prefix; the numeric Slurm ID is the execution identity.

## Land builds (added 23 September 2026)

- `compare_land_builds.py`: cell-by-cell comparison of a green-lory land table (100 %, rescaled to
  the requested shares) with the legacy land step; writes `cell_comparison.csv`,
  `country_comparison.csv`, `focal_cells.csv`, `summary.json` and `land_build_comparison_maps.png`
  under `results/campaigns/reconciliation_final_20260916_v1/land_builds/<tag>/`.
- `interim_center_20260923`: local centred table with approximate exclusions (see RUNS.md); superseded
  by the ARC build `land_center_20260923_v1` once fetched.
- `plot_land_capacity_maps.py` now renders exact 1-degree rasters (no marker artefacts).
