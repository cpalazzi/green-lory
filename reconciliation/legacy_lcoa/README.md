# legacy-lcoa package

End-to-end reconstruction of the plant model that produced the `LCOA` column of
the Verschuur et al. (2024) shipping input (`green-porpoise/data/c_NH3_cost_4.5.csv`),
plus the land/capacity step that produced its `Max_capacity` column and was
never in any repository. Part of the reconciliation package inside
`green-lory/reconciliation/` (decision of 15 September 2026). Read
[PROVENANCE.md](PROVENANCE.md) first for what is and is not recoverable, then
[LEGACY_REPLICATION_20260915.md](LEGACY_REPLICATION_20260915.md) for the tested
configurations and the recovered capacity rule, then
[GLOBAL_SURFACE_20260916.md](GLOBAL_SURFACE_20260916.md) for the global
replication surface and supplier table.

## Contents

| Path | Role in the pipeline |
|---|---|
| `source/94de8ce/` | Frozen lcoa-opt at commit 94de8ce (3 May 2023), the state that replicates the archived table; `git archive`, nothing edited; plus the cost workbooks (`GeneralSteelData_20230831.xlsx`, `x_Cost_Forecasting_20230831.xlsx`) and `SOURCE_IDENTITY.md` |
| `source/cd56c11/` | Frozen lcoa-opt at cd56c11 (6 June 2023), the last original-author commit; used to show that the June additions (water cost, cycling constraint, ...) are not in the archived table |
| `environment.legacy-min.yaml` | Conda environment: PyPSA 0.25.1 with the Pyomo `lopf` path, **Pyomo 6.5.0** (later Pyomo silently gives a zero objective), Gurobi 11 |
| `extract_weather_store.py` | Step 1: gather per-cell hourly profiles from the nine 2019 NetCDF files into a compact float64 store (values unchanged; needed on network file systems) |
| `run_legacy_cells.py` | Step 2: run the frozen model at listed cells under named temporal-accounting variants, CAPEX sources, tracking and WACC bases; writes one directory per cell with `summary.json`, solved component tables and a run manifest with all input hashes |
| `legacy_land_areas.py` | Step 3: the missing land step: MODIS class fractions x Table-2 suitability x spherical pixel area x 2 % share, per technology, centered 1-degree cells, no exclusions |
| `legacy_capacity.py` | Step 4: the recovered land-to-capacity rule (empirical densities 140 MW/km2 PV and 7.3 MW/km2 wind, complete overlap) as a library and a tool that applies it to a green-lory surface |
| `build_legacy_supplier_table.py` | Step 5: merge run shards, apply the rule to the replicated 1 Mt/yr designs, write the supplier table in the archived column contract plus a `gpo_export/` contract for green-porpoise |
| `collect_summaries.py` | Gather all per-cell `summary.json` of a run into one `summaries.jsonl` (fast transfer from ARC); accepted by the two tools below in place of the run directory |
| `compare_runs.py` | QA: cell-by-cell comparison of two executions of the same configuration (ARC versus Mac); unit tests in `tests/test_legacy_surface_tools.py` |
| `analyze_sample_capacity.py`, `test_capacity_hypotheses.py`, `compute_annual_cf.py` | Diagnostics used to recover the rule and check capacity factors |
| `cells_3.csv`, `cells_sample_v1.csv`, `cells_archived_all_v1.csv`, `shards/` | Cell lists: the three focal cells; the 563-cell latitude-stratified sample; all 15,377 archived supplier cells with Ameli WACC by country; the 16 global shards |

Generated evidence lives under `results/campaigns/legacy_lcoa_20260915_v1/`
(never inside this directory): each run has `manifest.json` (source, input and
weather hashes, package versions), per-cell `summary.json`, and a `results.csv`
roll-up; supplier tables and cross-checks are under `audit/`.

## The replicating configuration

`--era may2023 --variants stated_4h_mean --enable-tracking 1.0587 --capex-source xcost45`
with Ameli reduced WACC by country (5.1 % for Chile and Australia): frozen
3 May 2023 code, overnight CAPEX from the RCP 4.5 sheet of
`x_Cost Forecasting.xlsx` (2050 column) annualised with the workbook's own
8 %/20-year/2 % O&M annuity-due factor 0.1143 and rescaled to the country WACC,
single-axis tracking enabled at 1.0587 x the fixed-PV cost, four-hour block
means (2,190 periods), 1 Mt/yr reference load (713.47 MW HHV). Every departure
from the frozen code is a named harness option and is recorded in the manifest.
On the 563-cell sample the LCOA ratio rerun/archived is 1.040 (IQR 1.032-1.057);
the residual is a uniform finance-convention offset and is not tuned away.

## Running the pipeline

```sh
# 0. environment (Mac: conda; ARC: arc/jobs/07_build_legacy_env.sh)
conda env create -f reconciliation/legacy_lcoa/environment.legacy-min.yaml

# 1. weather store for the cells of interest (nine-file stack on the Mac or ARC)
python reconciliation/legacy_lcoa/extract_weather_store.py \
  --weather-dir ~/programming/shipping_sprint/lcoa_model/lcoa-opt/data \
  --cells reconciliation/legacy_lcoa/cells_archived_all_v1.csv \
  --output results/campaigns/legacy_lcoa_20260915_v1/weather_store_archived15377_mac_v1 --all

# 2. plant model at the three focal cells (full dispatch series kept)
python reconciliation/legacy_lcoa/run_legacy_cells.py \
  --era may2023 --variants stated_4h_mean,hourly --enable-tracking 1.0587 --capex-source xcost45 \
  --weather-store results/campaigns/legacy_lcoa_20260915_v1/weather_store_archived15377_mac_v1 \
  --cells reconciliation/legacy_lcoa/cells_3.csv --solver gurobi --threads 4 \
  --output results/campaigns/legacy_lcoa_20260915_v1/three_cells_<new-id>
#    ... or the global surface, one shard per job (ARC: arc/jobs/08_legacy_global.sh)

# 3. land areas (MODIS MCD12C1 2022 C6.1; source substitution is recorded)
python reconciliation/legacy_lcoa/legacy_land_areas.py \
  --modis ~/programming/shipping_sprint/lcoa_model/lcoa-opt/data/MCD12C1.A2022001.061.2023244164746.hdf \
  --cells reconciliation/legacy_lcoa/cells_archived_all_v1.csv \
  --output results/campaigns/legacy_lcoa_20260915_v1/audit/legacy-land-areas-v1

# 4+5a. supplier table as stated in the papers (consistent variant)
python reconciliation/legacy_lcoa/build_legacy_supplier_table.py --rule stated_method \
  --run results/campaigns/legacy_lcoa_20260915_v1/arc_received_global_archived_v2/summaries/summaries.jsonl \
  --land results/campaigns/verschuur_reconcile_20260907_v1/arc_received_20260914/paper_2pct_slope15.csv \
  --archived results/campaigns/verschuur_reconcile_20260907_v1/historical/git-0a63616/data/c_NH3_cost_4.5.csv \
  --cells reconciliation/legacy_lcoa/cells_archived_all_v1.csv \
  --output results/campaigns/legacy_lcoa_20260915_v1/audit/global-supplier-table-stated-v1

# 4+5b. supplier table with the recovered constants (archived-table reproduction)
python reconciliation/legacy_lcoa/build_legacy_supplier_table.py --rule archived_table_reproduction \
  --run results/campaigns/legacy_lcoa_20260915_v1/arc_received_global_archived_v2/surface \
  --land results/campaigns/legacy_lcoa_20260915_v1/audit/legacy-land-areas-v1/legacy_land_areas.csv \
  --archived results/campaigns/verschuur_reconcile_20260907_v1/historical/git-0a63616/data/c_NH3_cost_4.5.csv \
  --cells reconciliation/legacy_lcoa/cells_archived_all_v1.csv \
  --output results/campaigns/legacy_lcoa_20260915_v1/audit/global-supplier-table-arc-v1

# QA: the ARC execution against the Mac execution of the same shards
python reconciliation/legacy_lcoa/compare_runs.py \
  --run-a results/campaigns/legacy_lcoa_20260915_v1/arc_received_global_archived_v2/surface --label-a arc \
  --run-b results/campaigns/legacy_lcoa_20260915_v1/global_archived_mac_v1 --label-b mac \
  --output results/campaigns/legacy_lcoa_20260915_v1/audit/arc-vs-mac-v1
```

The network step consumes `gpo_export/contract.json` through
`reconciliation/run_historical_network.py --supplier-contract` (ARC:
`arc/submit_network_run.sh`).

## Two capacity variants: as stated, and as archived

The papers state one land method; the archived table was produced by another.
The package therefore keeps two variants, deliberately separate
(`build_legacy_supplier_table.py --rule ...`):

| Variant | Rule | Land input | Result against the archived table | Status |
|---|---|---|---|---|
| `stated_method` | what Salmon 2022 section 2.2.1 and Verschuur 2024 section 4.7 state: 200 km2/GW wind (5 MW/km2), PV packed by latitude after van de Ven 2021 with a First Solar module (about 106 MW/km2 at the equator, 78 at 23 degrees), complete wind/solar overlap, 2 % of the Table-2 suitable area after WDPA and > 15 degree slope exclusions | the green-lory land build `paper_2pct_slope15.csv` (same Table-2 factors, MODIS 2022; south-west-anchored cells, a half-cell offset from the centered weather nodes) | capacity ratio median 0.44 (IQR 0.28-0.63); 2,787 cells at or above 1 Mt/yr against 4,552; Australia 372 cells / 797 Mt/yr against 553 / 1,827 | **the consistent legacy-lcoa**: legacy plant plus the paper's land method; `audit/global-supplier-table-stated-v1/` |
| `archived_table_reproduction` | recovered constants: 140 MW/km2 for all PV without latitude dependence, 7.3 MW/km2 wind, complete overlap, no exclusions | `legacy_land_areas.py` (2 % Table-2 areas, centered cells, no exclusions) | capacity ratio median 0.97; Australia 590 cells / 1,851 Mt/yr against 553 / 1,827 | **archived**: reproduces the published input but is inconsistent with the stated method; no source for its constants was found; `audit/global-supplier-table-arc-v1/` |

The 140 MW/km2 lies between the module-only density of the stated First Solar
module (181 MW/km2, panels touching) and the stated equatorial value (106): a
ground-coverage ratio of about 0.77 applied at every latitude, i.e. the
latitude-dependent row pitch of the prose was absent from the run that wrote
the table. Its land classes were the 2022 ones in aggregate: the recovered rule
reproduces Australian capacities equally in grassland (median ratio 0.99), open
shrubland (1.02) and savanna (1.13) cells, so a reclassification of Australia
between MODIS vintages is excluded as the explanation.

## Acceptance and known residuals

- LCOA: replicated within a uniform +4 % (finance convention) plus a few per
  cent at wind-using cells and about +15 % north of 60 degrees. The archived
  cost ordering of the focal cells (Atacama < central Australia < northwest
  Australia) is reproduced.
- Capacity: PV-dominated cells median ratio 0.998, wind-dominated 0.996, mixed
  wind/solar cells under-predicted by about 20 % at the median because the
  replicated designs use more wind than Salmon's. The two densities are
  empirical (what the archived table implies), not the paper's stated values.
- Not recoverable from local evidence: Salmon's MODIS year, any protected-area
  or slope exclusion he applied, the exact PV-density function, the definition
  of `Electricity_Cost_Frac`, and the runner that wrote the table.
