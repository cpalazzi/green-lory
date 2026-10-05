# Legacy-lcoa global replication surface and supplier table, 16 September 2026

Status: **accepted**. The frozen legacy model, in the replicating configuration
of [LEGACY_REPLICATION_20260915.md](LEGACY_REPLICATION_20260915.md) section 3,
was solved at all 15,377 archived supplier cells on ARC, cross-checked against an
independent execution on the Mac, and combined with the recovered land/capacity
rule into a supplier table in the archived column contract.

## 1. Execution

| Item | Value |
|---|---|
| ARC job | array `8821144_[0-15]`, cluster `htc`, partition `short`, 2 CPUs / 8 GB / 4 h per shard |
| Source release | `/data/engs-df-green-ammonia/engs2523/green-lory-releases/20260915-legacy-v2` (68 files, md5-verified; inventory tree hash `1287efc1...91f6`) |
| Configuration | `--era may2023 --variants stated_4h_mean --enable-tracking 1.0587 --capex-source xcost45 --weather-store ... --solver gurobi --threads 2 --no-timeseries` |
| WACC | per cell from `cells_archived_all_v1.csv` (Ameli reduced 2050 by country: 15,373 cells from the map, 6 distinct rates between 1.8 % and 5.1 %; 4 cells without a map entry use the 5.1 % default) |
| Weather | compact store `weather_store_archived15377_v1` extracted on ARC from the nine 2019 NetCDF files (`extract_weather_store.py`, array 8820743, merge 8821143); source-file hashes equal the Mac stack |
| Outcome | 16 shards COMPLETED between 01:24 and 02:33 UTC; 15,377 cells solved, 2,190 snapshots each, **zero failures**, empty stderr; median 13.3 s per cell, 119.5 core-hours in total |
| Fetched to | `results/campaigns/legacy_lcoa_20260915_v1/arc_received_global_archived_v2/` (`summaries/summaries.jsonl` with all per-cell summaries, SHA-256 verified; Slurm logs; `sacct` record; the full per-cell archive as a tarball) |

## 2. Cross-check: ARC versus Mac

The same configuration was run on the Mac (`global_archived_mac_v1`, Gurobi
11.0.1, 4 workers x 2 threads, weather store extracted locally). On the 10,821
cells solved on both machines by 07:12 UTC (`audit/arc-vs-mac-v1/`):

| Quantity | Maximum difference over 10,821 cells |
|---|---:|
| LCOA, relative | 4.5e-14 |
| Objective, relative | 4.5e-14 |
| Any solved capacity (wind, PV, tracking, electrolysis, HB, stores), relative | 6.2e-12 |

The two executions are numerically identical to floating-point precision. The
ARC surface is therefore the accepted result; the Mac run is its cross-check.

## 3. Supplier table

`audit/global-supplier-table-arc-v1/` (`build_legacy_supplier_table.py`):
LCOA from the surface, capacity from `legacy_capacity.py`
(`legacy_rule_complete_overlap_v1`: 140 MW/km2 PV, 7.3 MW/km2 wind, complete
overlap, 2 % Table-2 areas from `audit/legacy-land-areas-v1/`, centered cells,
no exclusions), archived column contract, plus `gpo_export/` (15,008
positive-capacity rows, USD2018, contract with input hashes). Statistics
against the archived table over the 15,377 cells
(`results/campaigns/reconciliation_final_20260916_v1/legacy_surface/`):

| Quantity | Archived | Replicated |
|---|---:|---:|
| LCOA rerun / archived, median (IQR) | 1 | **1.038** (1.029-1.064) |
| ... PV-only designs (7,278 cells), median (p10-p90) | | 1.037 (1.027-1.068) |
| ... mixed designs (8,094 cells) | | 1.043 (0.988-1.251) |
| ... cells above 60 degrees latitude (3,716) | | 1.083 |
| Capacity rule / archived, median (IQR), positive cells | 1 | **0.972** (0.83-1.12) |
| ... PV-limited cells | | 1.000 (p10-p90 0.83-1.75) |
| ... wind-limited cells | | 0.853 (0.47-1.46) |
| Cells at or above the 1 Mt/yr cutoff | 4,548 | 4,900 (4,367 in common; 181 archived-only, 533 replicated-only) |
| Cheapest-4,000 selection, unique cells in common | 4,000 | 3,656 |
| Total capacity, Mt/yr | 25,271 | 21,545 |
| Australia: eligible cells / capacity, Mt/yr | 553 / 1,827 | 590 / 1,851 |
| Australia west (< 129 E): eligible / capacity | 175 / 614 | 196 / 629 |
| Australia centre (129-141 E) | 203 / 755 | 217 / 766 |
| Australia east (> 141 E) | 175 / 458 | 177 / 456 |
| Cells with zero capacity | 667 | 436 |

The three focal cells: LCOA 221.33 / 242.64 / 234.04 USD/t against 212.92 /
233.69 / 226.39 (ratios 1.040 / 1.038 / 1.034); capacities 9.43 / 3.97 / 2.42
Mt/yr against 9.77 / 4.02 / 4.32. Central Australia is the mixed-design case
(348 MW wind + 3,039 MW tracking per Mt/yr) where the wind-land term limits
the rule at 2.42 Mt/yr; the archived 4.32 lies between the wind-limited and
PV-limited (5.31) values, consistent with a design with less wind than ours.

## 4. What is and is not reproduced

- Reproduced: the cost surface up to a uniform factor of about 1.04 with the
  archived spatial ordering; the eligibility pool (4,367 of 4,548 archived
  eligible cells remain eligible); the Australian supply that drove the network
  divergence (all three subregions within 3 %).
- Residuals, stated as they are: mixed wind/solar cells are under-predicted by
  about 10 % at the median and scatter widely (the replicated plant uses more
  wind than Salmon's); 181 archived-eligible cells fall below the cutoff (115 of
  them wind-limited, mostly Argentina, United States, Kazakhstan) while 533
  cells rise above it (514 PV-limited, mostly Algeria, Australia, Tanzania);
  LCOA runs 8 % high above 60 degrees where designs are wind-heavy; the archived
  `Electricity_Cost_Frac` (median 0.37) is not reproduced by the replicated
  electricity-capex fraction (median 0.48), whose archived definition is unknown.
- Not tuned: neither the 1.04 finance offset nor the capacity densities were
  fitted to the focal cells; the densities come from the 563-cell sample
  (LEGACY_REPLICATION section 6) and are applied unchanged here.

![Legacy replication surface against the archived table](../../results/campaigns/reconciliation_final_20260916_v1/figures/legacy_surface_vs_archived.png)

## 5. Downstream

The exported contract was staged to ARC
(`verschuur_reconcile_20260907_v1/lory/global_exports/20260916-v1/legacy_replicated/`,
hashes verified, read-only) and consumed by network run E
(`deposited-legacyrep-20260916-v1`, job 8823893, and its accepted-target form
`deposited-legacyrep-20260916-long-v1`, job 8823894); see
[../RUNS.md](../RUNS.md) and [../DECISION_20260916.md](../DECISION_20260916.md).
