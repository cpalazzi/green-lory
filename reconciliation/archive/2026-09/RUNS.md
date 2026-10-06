# Reconciliation run register

Primary endpoint: MOD-AMB, RCP4.5/70% adoption, Way 2050 costs, no subsidies.
All September jobs below are on ARC's `htc` cluster. Submission is not a
completion claim; only a passed result QA establishes an accepted output.

| Run | Worker jobs | QA | Purpose |
|---|---|---|---|
| Notification test | 8758069 | Completed, exit 0 | Mail flags verified; initial nonreceipt reported, Slurm email receipt confirmed 14 September |
| `archival-modamb-v1` | 8758130 | Failed during export after optimal solve | Preserved failed attempt; do not use its partial output as an accepted result |
| `archival-modamb-v2` | 8758215 | Completed; in-run QA passed | Pinned 2023 GPO equations, old unprefixed demand, historical surface; USD 254.048771/t |
| Replication `global-20260907-v1` | 8758155, 8758156, 8758157, 8758158 | 8758159 completed and passed 8 September | Four-hour Green Lory historical-assumption emulation; 52,702 locations |
| Central `global-20260907-v1` | 8758172, 8758173, 8758174, 8758175 | 8758176 completed and passed 9 September | Hourly Green Lory, explicit compressor and tank-only hydrogen storage; 52,702 locations |
| Hourly attribution `attribution-20260907-v2` | 8758225 | 8758226 completed and passed | Three full-year cells, nominal compressor and legacy hydrogen storage |
| Compressor attribution `attribution-20260907-v2` | 8758234 | 8758235 completed and passed | Same cells and hourly basis; explicit compressor with legacy tank is an intermediate sensitivity |

All results above were fetched on 14 September into
`results/campaigns/verschuur_reconcile_20260907_v1/arc_received_20260914/`.
The two full surfaces and two diagnostics passed a fresh local merge/QA run,
including checksums, exact coordinate coverage, full-year status and cost closure.
The user confirmed that Slurm emails arrived.

## Network comparisons submitted 14 September

All five use the checksum-verified public deposited equations, RCP4.5/70%,
no efficiency improvement, no subsidies, a 1 Mt/year supplier cutoff and
the archived maritime cost tensor. They request 4 CPUs, 64 GB and two hours
of walltime, with a one-hour solver limit, 48 GB solver memory ceiling and
0.1% optimality-gap acceptance target. BEGIN/END/FAIL mail is enabled.

| Job | Output directory under `networks/` | Comparison |
|---|---|---|
| 8805878 | `deposited-old-20260914-v1` | Historical surface, unprefixed demand, archived pipeline tensor |
| 8805879 | `deposited-old-routes-20260914-v1` | Same surface/demand; regenerate same-ISO3-or-1000km pipelines with a 1.1 distance multiplier |
| 8805880 | `deposited-rep-20260914-v1` | Replication surface with the same regenerated routing and unprefixed demand |
| 8805881 | `deposited-central-20260914-v1` | Hourly central surface with the same regenerated routing and unprefixed demand |
| 8805882 | `deposited-prefixed-20260914-v1` | Historical surface and archived pipelines; prefixed demand-file sensitivity |

All five were confirmed running at 11:39 BST on 14 September, after starting
at 10:48 BST. Their results are not
accepted until a summary records passed QA and the requested solver gap.
The public code's original solver tolerance is 1.5%; these runs request a
stricter tolerance. The historical demand-file discrepancy remains explicit.

Source release: `/data/engs-df-green-ammonia/engs2523/green-lory-releases/20260914-v3`.
Its 124-file inventory has tree hash
`cf16b040c6e82333c54325459a43c615b749e1dcad3953920bfed9d71964107c`.
Remote verification passed before submission. New supplier contracts are under
`lory/global_exports/20260914-v1/{rep,central}/` and preserve USD2018 costs,
annual capacity units, input hashes and the explicit cutoff.

## Original surface release and layout

ARC source release:
`/data/engs-df-green-ammonia/engs2523/green-lory-releases/20260907-v1-c570c5153d04`

Its 115-file source inventory has tree hash
`c570c5153d0478f41a9d6536c97ca353b0003f1d10a8941a0c690c06492bdd4b`.
It passed local unit tests (73 tests), the small-network solver smoke test,
remote source verification, versioned land validation and global WACC-override
coverage checks. The surface runs expect 52,702 grid coordinates each.

ARC campaign root:
`/data/engs-df-green-ammonia/engs2523/green-lory-campaigns/verschuur_reconcile_20260907_v1`

- `historical/git-0a63616/`: unmodified historical source/data extraction.
- `networks/archival-modamb-v1/`: failed export attempt, retained for diagnosis.
- `networks/archival-modamb-v2/`: accepted replay, flows and QA summary.
- `networks/logs/`: network Slurm output.
- `lory/10_replication/`: replication surface; run manifest, exact shard paths,
  effective scheduler records, merged results and QA.
- `lory/20_central/`: central surface with the same structure.
- `lory/15_attribution/`: controlled plant diagnostics.
- `notifications/`: notification test output.

The land input is reused, not rebuilt or overwritten:
`/data/engs-df-green-ammonia/engs2523/green-lory-campaigns/lory_reconcile_20260722_v1/land/paper_2pct_slope15.csv`.
SHA256: `4d0753f77f3574c1616159997481c090c68e383ac02e6b25284561fb0ab1b38f`.
Its renewable-land union is a classwise nested-overlap lower-bound estimate,
not an exact spatial intersection.

Local analysis outputs are separate from downloads:

- `comparison/global-20260914-v2/`: accepted global comparison. It includes
  19 historically eligible cells excluded by zero available land, which v1's
  matched-grid eligibility summary omitted. V1 is retained and superseded.
- `comparison/land-physics-20260914-v2/`: independent land/cost/grid checks.
  V1 stopped because its checker used a total-wind compatibility field at
  coastal cells. V2 explicitly uses onshore wind area and passes.
- `networks/archival-modamb-sparse-v2/`: accepted local sparse parity check.
- `networks/deposited-modamb-sparse-v1/` and `v2/`: unsuccessful local
  deposited-code attempts, stopped by their 4 GB and 8 GB memory ceilings.

```bash
ssh -o ControlPath=/tmp/arc-green-lory.sock arc-oxford \
  'squeue -M htc -u engs2523'
ssh -o ControlPath=/tmp/arc-green-lory.sock arc-oxford \
  'sacct -M htc -j 8758130,8758155,8758156,8758157,8758158,8758159,8758172,8758173,8758174,8758175,8758176 --format=JobID,State,ExitCode,Elapsed -P'
```

## 15 September 2026: the five network comparisons ended as time-limited feasible solutions

All five jobs (8805878–8805882) ran the full one-hour solver limit and were then
marked `FAILED` by the runner because the 0.1 % gap target was not reached
(`failure.json`: "Sparse result failed QA"); every other QA check passed
(port shortfall < 1e-7 t, supplier balance, cost closure). Outputs, Gurobi logs
and the Slurm logs were fetched to `arc_received_20260915/` in the campaign.
Costs are per tonne of model-covered port demand (594.500 Mt/yr; 609.766 for
the prefixed case); the gap is the solver's relative MIP gap at the limit.

| Job | Case | USD/t | Gap | Australia, Mt/yr | Active suppliers | Top producers |
|---|---|---:|---:|---:|---:|---|
| 8805878 | old surface, unprefixed demand, archived pipelines | 259.67 | 0.79 % | 259.4 | 311 | AUS, MRT, CHL, CHN, OMN |
| 8805879 | old surface, regenerated routes × 1.1 | 260.65 | 0.89 % | 255.3 | 317 | AUS, MRT, CHL, OMN, ESH |
| 8805880 | replication surface, regenerated routes | 274.91 | 0.28 % | 25.0 | 394 | SAU, MRT, YEM, CHL, SDN |
| 8805881 | central surface, regenerated routes | 310.21 | 0.25 % | (not in top 8) | 388 | SAU, YEM, DZA, CHL, EGY |
| 8805882 | old surface, archived pipelines, prefixed demand | 259.64 | 0.76 % | 259.9 | 316 | AUS, MRT, CHL, CHN, OMN |

The public deposit's own tolerance is 1.5 %, so these are usable as bounded
provisional results: the bundle differences (15–50 USD/t) are far larger than the
gaps (< 2.4 USD/t). A resubmission at 12 CPUs / 128 GB / 12 h with
`--time-limit 36000` would close them to 0.1 % if an accepted result is wanted;
the exact original command lines are in the fetched Slurm logs.

## 15 September 2026, night: legacy replication surface on ARC (release 20260915-legacy-v2)

**Why the first submission was withdrawn.** Array 8819394 (release `20260915-legacy-v1`,
16 shards × 4 CPUs / 12 GB / 4 h, file-based weather) was cancelled before it started. A
`devel` probe (8819730, 8820045) showed that the legacy weather files are contiguous
`NETCDF3_64BIT_OFFSET` arrays laid out (time, latitude, longitude): one cell's 8,760-hour
series is a strided read over the whole 1.5 GB file, costing 36–72 s cold and 13.5 s warm per
file on `/data` (three files per cell), and the harness hashed all nine files (13.5 GB) at
start-up, about eight minutes. On the Mac's SSD the same reads take about 1 s. A shard of 961
cells would have needed 15–30 h.

**Fix: a compact per-cell weather store.** `reconciliation/legacy_lcoa/extract_weather_store.py`
reads each file once in sequential 730-step blocks and gathers the requested cells into one
float64 array per source file (values unchanged); `run_legacy_cells.py --weather-store` reads
one contiguous 70 kB row per cell and technology through the same `Solars/Winds/SolarTrackings`
interface the frozen `renewable_data` expects. Equivalence test on the Mac
(`three_cells_may2023_xcost45_tracking_store_test_v1` vs `three_cells_may2023_xcost45_tracking_v1`):
LCOA, capacities and objectives identical to the last digit at all three cells; the nine
source hashes are carried in the store manifest and equal the Mac stack's.

- Store: `/data/engs-df-green-ammonia/engs2523/green-lory/data/weather_store_archived15377_v1/`
  (15,377 archived cells, no missing cell in any technology; 3.2 GB), extraction array
  8820743_[0-8] on `devel` (9.1–9.9 min each), merge job 8821143.
- Release `20260915-legacy-v2` (68-file package verified by explicit md5 lists; inventory
  tree hash `1287efc1…91f6`): harness with `--weather-store`, `extract_weather_store.py`,
  `legacy_capacity.py` (the recovered rule as a module with provenance flags), the refactored
  `build_legacy_supplier_table.py`, `arc/jobs/08_legacy_global.sh` (now 2 CPUs / 8 GB / 4 h per
  shard, `--line-buffered` filter) and `arc/jobs/09_extract_weather_store.sh`.
- Global replication surface: array **8821144_[0-15]** on `short`, dependency `afterok:8821143`,
  output `green-lory-campaigns/legacy_lcoa_20260915_v1/global_archived_v2/shard_XX/`
  (`global_archived_v1` holds only the empty directory of the cancelled attempt). Resumable with
  `--skip-existing` if a shard hits the 4 h limit.
- ARC Gurobi cross-check of the three cells: the `devel` attempt (8819730) timed out in the
  start-up hashing; the global surface contains the three cells and serves as the cross-check.

**Network attribution run D submitted.** `deposited-rep-legacyrule-20260915-v1` (job 8820849,
4 CPUs / 64 GB / 2 h, `--time-limit 3600`, same options as 8805880): the green-lory replication
surface's costs and designs with capacity from the legacy land rule (`legacy_capacity.py
apply-lory`, contract `lory/global_exports/20260915-v1/rep_legacy_rule/`, 14,471 positive-capacity
rows at the archived sites; 835 archived sites are absent from the green-lory grid). Against the
archived table at the 14,542 common sites: LCOA ratio median 0.99; eligible ≥ 1 Mt/yr archived
4,529 / legacy rule on green-lory designs 4,379 / September method 1,233; total capacity 25,013 /
18,673 / 4,306 Mt/yr; Australia eligible 545 / 449 / 20 and 1,724 / 1,262 / 281 Mt/yr. The
remaining shortfall is in wind-limited cells (median ratio 0.74 vs 0.95 for PV-limited): the
green-lory replication plant builds more wind than the legacy plant (central Australia 923 MW
wind vs 348 MW in the legacy replication, so 0.91 vs 2.42 Mt/yr under the same rule).

## 16 September 2026, 06:30 UTC: overnight state

- ARC socket dropped at ~00:10 UTC (`mux_client_request_session: read from master failed`);
  ARC jobs continue unattended (network run D 8820849; array 8821144 to its 4 h limit at ~02:41 UTC).
  Fetches and the run-E submission wait for a new socket.
- Mac insurance run (`global_archived_mac_v1`, 4 workers × 2 threads, weather store): 9,272 of
  15,377 cells solved by 06:17 UTC, no failures. Per-cell time rose from 6 s to 12–16 s at about
  00:50 BST on all four workers at once (machine-level; the shards are round-robin so it is not
  cell difficulty) — completion now expected around 11:30 UTC.
- Partial supplier-table preview on the 9,272 solved cells
  (`audit/global-supplier-table-mac-partial-v1`, shards 00–07 complete = an unbiased 60 % sample
  of the archived sites): LCOA rerun/archived median 1.038 (IQR 1.030–1.055); capacity rule
  median 0.974 (IQR 0.84–1.13); eligible ≥ 1 Mt/yr archived 3,129 / replicated 3,376 / common
  3,029; total capacity 17,864 / 14,792 Mt/yr; Australia eligible 429 / 452, capacity 1,370 /
  1,398 Mt/yr. The archived Australian supply that drove the network divergence is reproduced.

## 16 September 2026, morning: legacy global surface complete on ARC; network run D fetched

- ARC array **8821144_[0-15]** (release `20260915-legacy-v2`, weather store
  `weather_store_archived15377_v1`) COMPLETED on `short` between 01:24 and 02:33 UTC
  (elapsed 2 h 44 min to 3 h 52 min per shard at 2 CPUs / 8 GB; peak RSS below 1 GB).
  All 16 shards hold 961 cells (962 for shard 00): **15,377 archived supplier cells, zero
  failures, empty stderr**. Fetched to
  `results/campaigns/legacy_lcoa_20260915_v1/arc_received_global_archived_v2/` (single
  tarball with SHA-256 verified; Slurm logs and the release inventory beside it).
- Network run **D 8820849** (`deposited-rep-legacyrule-20260915-v1`: green-lory replication
  costs and designs, capacity from the legacy land rule, deposited equations, regenerated
  routes x1.1, 4 CPUs / 64 GB, 1 h solver limit) ended like the five 14 September runs: time
  limit reached at 0.60 % gap, every other QA check passed, runner marks it FAILED against
  the 0.1 % target. Fetched to `arc_received_20260916/`. Delivered cost **261.23 USD/t**,
  Australia **238.5 Mt/yr** (old surface with the same routing: 260.65 USD/t, 255.3 Mt/yr;
  green-lory replication surface with the September land method: 274.91 USD/t, 25.0 Mt/yr).
  216 of the 317 active suppliers of the old-surface control are active again.
- Quantitative comparison of the archival replay and runs A-D:
  `verschuur_reconcile_20260907_v1/comparison/networks-20260916-v1/` (`reconciliation/compare_networks.py`).
- The legacy land step is now standalone (`reconciliation/legacy_lcoa/legacy_land_areas.py`);
  `audit/legacy-land-areas-v1/` reproduces the September `global-unmasked-v2` areas for all
  15,377 cells with zero difference.
- Mac insurance run `global_archived_mac_v1` continues (shards 00-07 complete, 08-11 at
  ~70 %, 12-15 pending at 07:40 UTC); it is retained as the cross-check of the ARC execution.

## 16 September 2026, 07:00-07:15 UTC: legacy surface validated; network runs E and long controls submitted

- Cross-check `audit/arc-vs-mac-v1/` (`reconciliation/legacy_lcoa/compare_runs.py`): the ARC
  execution (Gurobi 11.0.3, 2 threads) and the Mac execution (Gurobi 11.0.1, 2 threads) of the
  same frozen configuration agree on all **10,821** cells solved on both by 07:12 UTC: maximum
  relative LCOA difference 4.5e-14, maximum relative capacity difference 6e-12. The ARC surface is
  the accepted global legacy replication surface; the Mac run is its cross-check and continues.
- Global legacy supplier table `audit/global-supplier-table-arc-v1/` (rule
  `legacy_rule_complete_overlap_v1`, land from `audit/legacy-land-areas-v1/`): 15,377 cells, no
  missing cell; LCOA rerun/archived median 1.038 (IQR 1.029-1.064); capacity rule/archived median
  0.972 (IQR 0.83-1.12); eligible ≥ 1 Mt/yr archived 4,548 / replicated 4,900 / both 4,367;
  Australia eligible 553 / 590 and 1,827 / 1,851 Mt/yr. Exported contract
  `gpo_export/` (15,008 positive-capacity rows, USD2018) staged to
  `verschuur_reconcile_20260907_v1/lory/global_exports/20260916-v1/legacy_replicated/` with
  matching hashes and made read-only.
- New wrapper `arc/submit_network_run.sh` (staged as release `20260916-network-tools-v1`) wraps the
  `arc/jobs/03_replay_historical_network.sh` template with the exact options of the 14 September
  runs; it records every sbatch line in `networks/logs/submissions.tsv`.

| Job | Output directory under `networks/` | Resources | Purpose |
|---|---|---|---|
| 8823881 | `deposited-old-routes-20260916-long-v1` | 12 CPUs, 96 GB, 12 h, solver limit 39,600 s, 0.1 % gap | Accepted-target rerun of the old-surface control with regenerated routes (8805879) |
| 8823893 | `deposited-legacyrep-20260916-v1` | 4 CPUs, 64 GB, 2 h, solver limit 3,600 s | **Run E**: legacy-lcoa replicated surface (legacy LCOA and designs, legacy capacity rule), same solver treatment as runs A-D |
| 8823894 | `deposited-legacyrep-20260916-long-v1` | 12 CPUs, 96 GB, 12 h, solver limit 39,600 s, 0.1 % gap | Accepted-target form of run E |

All three use the deposited equations, RCP4.5/70 %, unprefixed demand, regenerated
same-ISO3-or-1000 km routes x1.1, 1 Mt/yr cutoff and the archived maritime tensor.
- 07:56 UTC: the ARC SSH master socket dropped while the full per-cell tarball
  (`global_archived_v2_fetch.tar.gz`, written on ARC beside the run) was being transferred; the
  transfer must be redone once the socket is reopened. The collected `summaries/summaries.jsonl`
  (hash-verified) is already local and is sufficient for every table in this register; the tarball
  is the complete per-cell archive (solver logs and component tables).

## 16 September 2026, 08:35 UTC: network run E fetched; full legacy archive fetched

- **Run E 8823893** (`deposited-legacyrep-20260916-v1`, legacy replicated surface) ended at its
  one-hour solver limit at **0.59 % gap** (runner FAILED against the 0.1 % target; every other
  QA check passed). Delivered cost **268.16 USD/t**, Australia **250.4 Mt/yr** (west/centre/east
  177 / 63 / 10), 299 active suppliers, 247 in common with the old-surface control A2 (8805879:
  260.65 USD/t, 255.3 Mt/yr). The +7.5 USD/t is production cost (+7.26; the replicated plant's
  +3.7 % LCOA offset at the active suppliers); pipeline, shipping and storage differ by < 0.3 USD/t.
  Fetched to `arc_received_20260916/networks/deposited-legacyrep-20260916-v1/`.
- Comparison of the archival replay and runs A-E:
  `comparison/networks-20260916-v2/` (copied to `reconciliation_final_20260916_v1/networks_v2/`);
  three-cell table `reconciliation_final_20260916_v1/three_cells_v2/`.
- Full per-cell legacy archive (`global_archived_v2_fetch.tar.gz`, 33.8 MB, SHA-256 verified)
  extracted to `arc_received_global_archived_v2/surface/` (370 MB, 15,377 cell directories with
  solver logs and component tables); all 15,377 `summary.json` files are byte-equal in content to
  the collected `summaries/summaries.jsonl`.
- Long runs at 08:34 UTC: 8823894 (legacy replicated, 12 threads) at 0.60 % gap after 80 min;
  8823881 (old-surface control) at 1.56 % after 90 min. Both have until about 19:10 UTC.

## 16 September 2026, 10:30 UTC: legacy global replication surface complete (Mac)

`global_archived_mac_v1`: 15,377 / 15,377 cells, no failures (11:10 BST; four workers, 6 s per
cell while the Mac was active, 15 s while idle overnight). Supplier table and contract in
`audit/global-supplier-table-mac-v1/` (`gpo_export/` for green-porpoise, 15,008 rows). Headline
numbers and the anomaly analysis are in `LEGACY_REPLICATION_20260915.md` §8: LCOA ratio median
1.038; eligible ≥ 1 Mt/yr 4,548 archived / 4,900 replicated / 4,367 common; Australia 553 / 590
cells and 1,827 / 1,851 Mt/yr; 28 archived cells at 7–40 × the rule (5,543 Mt/yr, capped to
10 Mt/yr each in the network). Next: network run E with this contract (queued for the bridge).

## 16 September 2026, 10:00-11:00 UTC: land rules revised; land-share sensitivity submitted

- **Allocation modes cleaned up** (green-lory only; the legacy package is unchanged). `model/land_capacity.py`
  now has two rules: `colocated` (base: wind and PV share the same suitable-land budget and only the
  exclusive fraction of the wind footprint, 3 % from Denholm et al. 2009 Table 1, competes with PV;
  per-technology caps kept) and `exclusive` (the September rule: whole wind footprint counts). The
  September names `paper_union`, `technology_shared` and `independent_legacy` are accepted as
  deprecated aliases only. The fraction is a wind property in the technology YAML
  (`wind.land_use_exclusive_fraction`); results and manifests record the effective value.
- **Tracking footprint** in the central configuration set from Bolinger & Bolinger (2022): tracking
  needs 0.35/0.24 = 1.458 times the fixed-tilt area (was 2). **PV policy** is explicit: the new
  central scenario `central_way2050_flat_amelired_1h_fixed_explicit_compressor_dea_tank_colocated`
  is fixed-tilt only; `tracking-only` is a named sensitivity; the supply pilot accepts all three.
  The analytical energy certificate now takes the ratio, the exclusive fraction and the policy.
  155 tests pass.
- **Land-share sensitivity** (`reconciliation/land_share_sensitivity.py`,
  `reconciliation_final_20260916_v1/land_share_sensitivity_v1/`): on the scaled-design surfaces
  capacity is proportional to the share, so the September surfaces were rescaled without new
  solves. Doubling the 2 % share gives Australia 251 eligible cells / 366 Mt/yr (replication) or
  100 / 130 (central) against 553 / 1,762 archived; matching the archived Australian eligible
  capacity needs a share of 12.6 % (replication) or 17.6 % (central). Network runs with the
  rescaled contracts (same solver treatment as A-E):

| Job | Output directory under `networks/` | Supplier input |
|---|---|---|
| 8824427 | `deposited-central-land-share-x2-20260916-v1` | central surface, 4 % share |
| 8824428 | `deposited-central-land-share-x4-20260916-v1` | central surface, 8 % share |
| 8824429 | `deposited-rep-land-share-x2-20260916-v1` | replication surface, 4 % share |
| 8824430 | `deposited-rep-land-share-x4-20260916-v1` | replication surface, 8 % share |

- Long runs at 10:20 UTC: 8823894 (legacy replicated) 0.59 % after 3 h 08 min, hardly moving since
  80 min; 8823881 (control) 1.01 % after 3 h 18 min. Both continue to their 12 h limit.
- Data moved into the repository: the nine legacy weather files, MODIS MCD12C1, `model_bathymetry.nc`
  and GEBCO 2025 now live under `data/` (tracked README with hashes; contents ignored), with
  symlinks left at their previous paths.
- **Co-located supply pilots submitted 10:33 UTC** (template `arc/jobs/06_land_supply_pilot.sh`, release
  `20260916-land-supply-v6-colocated`, 12 CPUs / 32 GB / 2 h each, 24 solves each): **8824523**
  `supply_curve_v2_fixed_colocated` (fixed-only, `config_v2.json`) and **8824524**
  `supply_curve_v2_track_colocated` (tracking-only, `config_v2_tracking.json`); same land, weather
  subsets and plant bundle as the accepted v1/fixed-threshold runs; the technology overlay differs
  only in land-use parameters and is pinned by hash in the config. Release v5 was withdrawn before
  submission (bathymetry file missing from the release, then the revised overlay rejected by the
  controlled-input check); record `supply_curve_v2_colocated_submission_v1.json`.
- Mac insurance run `global_archived_mac_v1` completed all 15,377 cells (about 11:35 UTC); full
  cross-check `audit/arc-vs-mac-v2/` covers every cell (see its summary.json).
- **Co-located pilots v2 failed on their own validation, not on the model**: 8824523 (fixed-only)
  stopped because `config_v2.json` still carried the both-PV control costs from v1 (the fixed-only
  controls are 234.71 / 251.85 / 213.17 EUR/t, exactly the accepted fixed-threshold values, so the
  solves were right); 8824524 (tracking-only) stopped at Atacama 3 Mt/yr because the result checker
  still hard-coded the tracking footprint ratio of 2 and the exclusive union rule. Both fixed in
  `run_supply_pilot.py` (ratio and wind density from the technology YAML, shared budget
  `solar + f x wind`, controls optional and recorded when no reference exists). Release
  `20260916-land-supply-v7-colocated` (tree `c1f87cf3...`), preflights passed, resubmitted with
  `-M all`: **13158922** (fixed-only) and **13158924** (tracking-only) started immediately on the
  `arc` cluster at 11:12 UTC; outputs `supply_curve_v3_{fixed,track}_colocated`.
- `arc/submit_network_run.sh` now submits with `--clusters=all` by default and records the cluster
  chosen; WDPA February 2026 shapefiles copied to `data/land/` (6.4 GB).

## 16 September 2026, 11:45 UTC: the legacy-lcoa "as stated" variant, and the cutoff question

- The 1 Mt/yr cutoff is real in both code lineages: the public deposit's `p_select_locations.py`
  drops `Max_capacity < min_production` (1) and keeps the cheapest `number_of_locations`
  (4,000 passed from `main.py`); the later project's `gpo/select_locations.py` is identical with
  `supplier_number = 4000`. Australia survived it in the archived run because that run's
  capacities were 4 to 6 times the stated method's, not because the cutoff was absent.
- Reclassification hypothesis rejected: the recovered rule on MODIS 2022 classes reproduces the
  archived Australian capacities equally in grassland, open-shrubland and savanna cells (median
  ratios 0.99, 1.02, 1.13; Sahara/Arabia control 0.98); only 8 cells that are barren in 2022 have
  ratios above 2. The deposit holds no land data; Carlo's October 2023 table carries
  availabilities identical to the 2022 fractions (0.499 at both Australian focal cells).
- `legacy_capacity.py` now carries two variants: `stated_method` (paper constants, exclusions,
  latitude-packed PV, land from the green-lory build) and `archived_table_reproduction` (the
  recovered constants, archived as inconsistent with the stated method).
  `audit/global-supplier-table-stated-v1/`: capacity ratio median 0.44 (IQR 0.28-0.63), 2,787
  eligible cells (4,552 archived), Australia 372 cells / 797 Mt/yr (553 / 1,827); 585 archived
  cells have no row in the land build (outside its latitude bounds or offshore-only) and get zero.
  Caveat recorded in the contract: the land build anchors cells at the south-west corner while
  the legacy step and the weather nodes are centered, a half-cell offset that also affects the
  September green-lory surfaces and should be removed in the next land build.
- Network run **F** submitted: `deposited-legacystated-20260916-v1` (legacy costs and designs,
  stated land method), same solver treatment as A-E, `-M all`.

## 16 September 2026, 12:10 UTC: land-share network results (provisional, one-hour solver limit)

All four ended at the limit with 0.7-1.0 % gaps and every other QA check passed (runner marks
FAILED against 0.1 %). Fetched to `arc_received_20260916/`; comparison `comparison/networks-20260916-v3/`.

| Job | Supplier input | USD/t | Gap | Australia, Mt/yr (west / centre / east) | Top producers |
|---|---|---:|---:|---:|---|
| 8824429 | replication surface, 4 % share | 263.84 | 1.04 % | 183.7 (112 / 56 / 15) | AUS, YEM, CHL, MAR, OMN |
| 8824430 | replication surface, 8 % share | 258.18 | 0.67 % | 257.9 (185 / 64 / 9) | AUS, MAR, CHL, MRT, USA |
| 8824427 | central surface, 4 % share | 293.04 | 0.77 % | 97.2 (45 / 36 / 16) | AUS, YEM, SDN, OMN, MRT |
| 8824428 | central surface, 8 % share | 281.30 | 0.73 % | 241.6 (158 / 78 / 5) | AUS, MAR, SDN, MRT, USA |

Reading: the network needs roughly 250-300 Mt/yr of admitted Australian capacity, not the archived
1,762; at 8 % (eligible pool 719-1,081 Mt/yr) Australian production is back at the archived
258 Mt/yr on both surfaces, at 4 % it is already the largest producer. Delivered cost falls with the
share (central 310 -> 293 -> 281 USD/t; replication 275 -> 264 -> 258) because cheaper near-demand
supply replaces long-haul supply. These are share sensitivities on the scaled-design surfaces,
not calibrations; the co-located, fixed-only supply curves supersede them once accepted.
- Pilots v3 (13158922 / 13158924 on `arc`) failed on a reporting bug, not on the solves: in
  `report_land_feasible_quantity` the gridless-energy fraction (1.0) reused the variable name of the
  wind exclusive fraction, so constrained results reported 1.0 and the pilot's consistency check
  stopped at the first constrained point. Fixed (`exclusive_fraction`), regression test added
  (`tests/test_land_capacity.py`), verified locally on a one-week HiGHS solve in both land modes.
  Release `20260916-land-supply-v8-colocated` (tree `20c11548...`), preflights passed, resubmitted
  with `-M all`: **13160582** (fixed-only, `supply_curve_v4_fixed_colocated`) and **13160584**
  (tracking-only, `supply_curve_v4_track_colocated`) on `arc`.
- **Run F 13160450** (`deposited-legacystated-20260916-v1`, legacy costs and designs with the paper's
  land method as stated; `arc` cluster, one-hour solver limit): **270.68 USD/t** at 0.54 % gap,
  Australia **209.8 Mt/yr**, then Yemen 79, Mauritania 44, Chile 39, Bolivia 24, Oman 22; 360 active
  suppliers. Fetched to `arc_received_20260916/networks/`. Reading: even the stated land method
  keeps Australia the leading producer with the legacy plant (797 Mt/yr eligible there), so the
  September green-lory exclusion of Australia came from the exclusive-sharing, tracking-penalised
  land rule and the smaller designs, not from the paper's method.
- September global surface timings (for campaign sizing): replication (4-hour step) 48-83 min per
  quadrant on 48 CPUs; central (hourly) 2 h 46 min to 3 h 03 min per quadrant on 48 CPUs, peak RSS
  26 GB. An hourly 52,702-cell surface is therefore about 3 h of wall time on four concurrent
  48-core jobs (about 580 CPU-hours).

## 23 September 2026: outcomes of the week's jobs; land maps

- **Long accepted-target network runs (12 CPUs, 96 GB, 12 h).** 8823881 (old surface, regenerated
  routes) was killed by the memory limit after 10 h 32 min (99.6 GB RSS) at a 1.0 % gap; 8823894
  (legacy replicated surface) ran its 11-hour solver limit and ended at **0.557 % gap**, 268.08 USD/t,
  Australia 251.9 Mt/yr, against 268.16 USD/t and 250.4 Mt/yr from the one-hour run 8823893. The
  bound moved from 0.59 % to 0.56 % in ten extra hours: the deposited formulation's bound plateaus,
  and the one-hour results are robust. Fetched to `arc_received_20260923_long/`. Recommendation:
  adopt the public deposit's own 1.5 % tolerance (or 0.6 %) as the acceptance target for network
  comparisons and stop pursuing 0.1 %.
- **Co-located pilots v4** (13160582 fixed-only, 13160584 tracking-only, `arc`, 12 CPUs, 2 h):
  TIMEOUT with 20 and 17 of 24 points solved (the near-infeasible high quantities trigger the
  dual-simplex retry and are slow). Partial results fetched to `arc_received_colocated_v4/`.
  Fixed-only, co-located, 1.458 tracking ratio: Atacama feasible to 4.5 Mt/yr at 238.2 EUR/t
  (5.0 unfinished); northwest Australia to 2.0 at 255.2 (2.5 unfinished; was 264.4 under the
  exclusive rule with both PV options); central Australia to 2.0 at 237.1 (was 260.0), 4.32
  infeasible. Tracking-only: Atacama to 4.0 at 225.3, northwest Australia to 1.75 at 244.0,
  central Australia to 1.5 at 223.9 with 2.0 infeasible. Resubmitted unchanged as v5 with a
  12-hour limit on `--clusters=all`; outputs `supply_curve_v5_{fixed,track}_colocated`.
- **Land maps** (`reconciliation/plot_land_capacity_maps.py`,
  `reconciliation_final_20260916_v1/figures/land_capacity_maps.png` and
  `land_availability_maps.png`): archived table 14,710 cells with capacity / 4,548 eligible /
  25,236 Mt/yr; legacy stack as stated 14,194 / 2,784 / 9,692; green-lory September central
  15,885 / 832 / 3,688; legacy with recovered constants 14,941 / 4,900 / 21,511. Available PV land
  at 2 %: legacy step 756,828 km² over the archived cells; green-lory build 641,993 km² over all
  onshore cells; ratio at common cells median 0.85 (exclusions plus the half-cell anchor offset).
- **Heatmap stripes (user query, 23 Sep).** The vertical white lines in `land_capacity_maps.png` and
  `land_availability_maps.png` were a rendering artefact, not missing data. Data check: every
  green-lory table populates all 360 longitude columns (September central 52,702 cells, min 137
  cells per column; 29 per column across Australia); the legacy tables populate 357 of 360, the
  three empty columns being 170-168 W in the open Pacific. Cause: the panels drew 2-point square
  markers (`s=4`) at a horizontal pitch of about 4.4 px (1,800 px wide figure) against a 4.17 px
  marker width, leaving a sub-pixel gap that the rasteriser turned into a white column every few
  degrees (period 5 px in the PNG); the vertical pitch (3.6 px) was smaller than the marker, so no
  horizontal lines appeared. `reconciliation/plot_land_capacity_maps.py` now draws each cell as a
  pcolormesh quad on its exact footprint; both figures regenerated (same numbers, no stripes).
- **One-degree labelling bug in the July/September land build (found 23 Sep).** In
  `_aggregate_availability_from_hdf4` each MODIS band was labelled by its north edge
  (`floor(top)`), while the cell area, WDPA overlay, slope overlay and bathymetry sampling treated
  the label as the south-west corner. Verified on the September table: the MODIS water fraction at
  label (lat, lon) equals the band [lat-1, lat] x [lon, lon+1] exactly (3,000 coastal cells, 100 %
  exact) and the whole 54,000-row table equals the corrected south-west aggregation shifted by one
  degree (max difference 1e-14 with pixel means). Consequence: every green-lory land table used so
  far (`max_capacities_*_slope15.csv`, July build 8253741) combines the suitability of the cell one
  degree south with the exclusions of the labelled cell; relative to the centred weather node the
  MODIS content sits half a degree south and the exclusions half a degree north. The September
  central surface, the supply pilots' land budgets, the "B/A median 0.85" availability ratio and the
  stated-method legacy variant (which read `solar_area_km2` from that build) all inherit it.
  Fix in `model/land_processing.py`: explicit `cell_anchor` (`center` default, `southwest` kept),
  consistent labelling in both aggregation paths, `_cell_polygon` with a dateline-straddling centred
  cell at 180 W, anchored slope/WDPA/bathymetry/density sampling, exact spherical row weights (area
  means, not pixel means) and a `cell_anchor` column in the output. Tests
  `tests/test_land_processing_anchor.py` (12) pin the geometry; suite 167 passing. Validation on the
  real MODIS file: the centred aggregation reproduces the legacy land step
  (`legacy_land_areas.csv`) on all 15,377 archived cells to 3e-14 in every class fraction and to
  1e-13 in area; the south-west aggregation matches the legacy reader with `anchor=southwest` to
  3e-14.
- **Centred land build submitted.** Release `20260923-land-center-v1` (files and sha256 in its
  `SHA256SUMS`; `RELEASE_NOTES.md`; data symlinked to `green-lory/data`), new wrapper
  `arc/submit_land_center_matrix.sh`, template `00_build_land_constraints.sh` with `ARC_CELL_ANCHOR`.
  Jobs on `arc` (short, `--clusters=all`): 13225827 100 % build (128 GB, 12 h; the July build took
  8 h 32 min on htc), 13225828 2 % and 13225829 20 % rescaled tables (afterok). Outputs:
  `green-lory-campaigns/land_center_20260923_v1/max_capacities_center_{100pct,2pct,20pct}_slope15.csv`.
  Latitude bounds stay -75..75: the 585 archived cells above 75 N hold 0.027 % of archived capacity
  and none reaches 1 Mt/yr.
- **Interim centred table and comparison (local, while ARC runs).**
  `results/campaigns/reconciliation_final_20260916_v1/land_builds/interim_center_20260923/`: exact
  centred MODIS content; exclusions approximated from the September south-west overlays (whose
  polygon/raster geometry was correct for their labelled box) by averaging the four quarter-overlapping
  cells. `reconciliation/compare_land_builds.py` (rerun on the ARC tables when they land) gives:
  at 2 % the green-lory/legacy PV-land ratio is the exclusion factor itself (median 0.894 over the
  archived cells, 0.840 area-weighted): 635,809 km² vs the legacy 756,828 km²; at 20 % 6.36 Mkm²
  (8.4x the legacy 2 % pool, 10x the green-lory 2 % pool). Co-location changes nothing in the
  desert cells: the classwise union equals the PV area wherever the wind and PV class factors
  coincide (barren, shrub), so PV plus 3 % of the wind footprint is limited by the same area.
  Focal cells (legacy 2 % PV km² / green-lory 2 % / exclusion factor): Atacama 227.6 / 221.4 /
  0.973; NW Australia 113.2 / 111.6 / 0.986; central Australia 115.3 / 112.0 / 0.972. Country
  factors (area-weighted): Australia 0.762, Russia 0.808, Canada 0.762, USA 0.785, Algeria 0.942,
  Libya 0.993, Sudan 0.980, Mauritania 0.993: protected areas cost Australia a quarter of its
  suitable land, the Sahara almost nothing.
- **Supply pilots v5** (13225807 fixed-only, 13225809 tracking-only) started running on `arc` at
  about 09:27 and 09:41 UTC (12-hour limit).
- **Centred land build finished** (13225827: 5 h 34 min on arc-c310; derived tables 36 s each).
  Tables (sha256 prefixes): 100 % `93a47776`, 2 % `89a60166`, 20 % `12277da1`, in
  `green-lory-campaigns/land_center_20260923_v1/`. Exact statistics: 23,395 cells with land (the
  interim table counted 31,217 because fully-water cells keep a 1e-14 floating residue), 17,136
  with suitable land, exclusion factor median 0.876 over land cells; 2 % PV land 648,765 km² over
  all land cells (interim 650,403). Focal cells (protected %, slope-suitable %, factor, PV km² at
  2 %): Atacama 0.0 / 99.68 / 0.997 / 226.8; NW Australia 0.0 / 99.96 / 1.000 / 113.1; central
  Australia 11.1 / 100 / 0.889 / 102.5. The interim four-cell averaging had smeared protected
  areas across neighbours (Atacama 1.9 %, central Australia 2.8 %); the exact overlay puts the
  protection where it is. Exact `compare_land_builds.py` rerun deferred until the tables can be
  fetched on a wired connection (17 MB each).
- **Key-run campaign set up (23 Sep, evening, mobile connection: source-only transfers).**
  Naming convention and run store: `reconciliation/RUN_STORE.md`, `reconciliation/run_store.csv`.
  Two green-lory scenarios added to `arc/submit_lory_sequence.sh` (class `30_keyruns`):
  `gl_dea2050_wacc5_bflat_wflat2usd_land20c_fixed` (uniform 5 % WACC) and
  `gl_dea2050_ameli_bflat_wflat2usd_land20c_fixed` (Ameli reduced WACC). Both: DEA 2050 costs via
  the new overlay `inputs/tech_config_ammonia_plant_2050_dea_colocated.yaml` (adds the 1.458
  tracking footprint and the 0.03 wind exclusive fraction to `tech_config_ammonia_plant_2050_dea.yaml`),
  plant bundle `basic_ammonia_plant_2050` (fixed PV only, explicit compressor, tank store), hourly,
  co-located land, `postprocess`/`paper_scaled`, uniform 2 USD2020/m³ water (1.75 EUR) in the headline,
  zero land rent, centred 20 % land table, cells restricted by `--global-locations` to
  `land_center_20260923_v1/locations_onshore_suitable_center_20260923.csv` (16,509 cells with at
  least 0.1 % land and 1 km² suitable area at 100 %, north of 60 S; sha `792a3e41`; shards
  2,362 / 3,482 / 6,168 / 4,497). Finance overrides rebuilt for every centred cell by
  `arc/build_finance_overrides.py`: `inputs/uniform_interest_inputs_0p05_2050.csv` and
  `inputs/amelired_interest_inputs_2050_center.csv` (1,656 cells absent from the September Ameli
  file, 33 of them in the run set such as (-30, 126) and (-21, 132), take the nearest covered cell's
  rate; list in `inputs/amelired_interest_fill_2050_center.csv`); both byte-identical when generated
  on ARC. Wrapper changes: `--global-locations`, per-scenario cost scope (`flat_wacc5_...` accepted
  by `write_campaign_manifest.py` and the QA gate, finance mode `uniform_wacc` with its rate),
  PV-policy-aware plant validation, QA expects the explicit cell list. Release
  `20260923-keyruns-v1` (273 inventoried files, tree `62edeeb5`; data symlinked). Stage order:
  smoke (3 cells, 168 h) then global; results under `green-lory-campaigns/keyruns_20260923_v1/`.
- **Deferred:** spatial build/remoteness and spatial water runs (inputs to be reviewed: remoteness
  is a linear +25 % per 2,000 min travel time to a city of 50k, uncapped, median 1.10 / p90 1.97 /
  max 6.97 over land cells; the spatial water column is dominated by a pipeline term from a
  1-degree "distance to water source" that gives 205 km at Atacama and 8 USD/m³, 595 km and
  21 USD/m³ in central Australia, median 9.2 USD/m³ over land cells).

## 24 September 2026

- **Decisions from the evening of 23 Sep (mobile connection).** Water is priced in the model
  currency: the flat case is 2 EUR2020/m³ (entered as 2.2843903 USD2020 in the DEA co-located
  overlay because the site-cost pipeline keeps the spatial input currency); values leave the run
  ids (`wflat`, not `wflat2usd`). The legacy key run is a full working stack close to the original
  runs rather than the original inputs verbatim: Way 2050 x_Cost CAPEX, 4-hour step, tracking on,
  Ameli WACC, rebuilt centred land stack, but annualised with green-lory's per-technology
  convention (`--annuity green_lory`: overnight CAPEX x (CRF(WACC, lifetime) + O&M), lifetimes and
  O&M from `inputs/tech_config_ammonia_plant_2050_way_eur.yaml`). The user declined a network
  run on the intermediate stated-method table. Spatial build/remoteness and water runs stay
  deferred while the user thinks about the coastal-port question.
- **Readability rename** (user request): `--lcoa-land-mode {postprocess, enforce}` is now
  `--land-constraint {after_solve, in_solve}`; `--capacity-method {paper_scaled, solved_quantity}`
  is `--capacity-rule {scaled_reference_design, solved_quantity}`; result columns
  `paper_scaled_*` are `scaled_design_*`; manifest, QA, wrapper, job template, pilots and tests
  follow. Scripts that only read September artefacts keep the old column names. `GLOSSARY.md`
  documents options, run-id tokens, result and land-table columns.
- **Manifest and QA water rule.** The hard "2 USD/m³" check is replaced by a currency-consistent
  one: the manifest records the baseline in the source currency, the conversion and the model
  currency; the wrapper declares the expected model-currency price per scenario
  (`FLAT_WATER_MODEL_PER_M3`, 2.0 for the key runs, 1.751012 for the September scenarios).
- **Legacy harness v4** (`green-lory-releases/20260924-legacy-v4`, hardlink copy of v3 plus the
  changed files): at the Ameli 5.1 % WACC the green-lory annuity lowers annualised costs to
  0.82 (solar, 40 y), 0.89 (wind, H2 store), 0.95 (electrolyser), 0.98 to 0.99 (batteries, HB,
  ammonia store) and raises the fuel cell to 1.04 of the workbook basis. Three-cell devel test
  job 13237552 (`glannuity_three_cells_v1`) before the global array.
- **Three-cell legacy annuity test 13237552** (devel, 2 min 37 s): LCOA at the Ameli 5.1 % WACC
  Atacama 200.09 USD/t, northwest Australia 218.25, central Australia 211.27, against 221.33,
  242.64 and 234.04 with the workbook annuity: a uniform 9.6 to 10.1 % reduction, as expected from
  the longer green-lory lifetimes (40 y PV, 30 y wind). At 8 % the same cells give 253.56, 276.94
  and 267.53. Submitted the global legacy key run as array 13237736 on `arc` (16 shards, 2 CPUs,
  8 GB, 4 h each), output `keyruns_20260923_v1/legacy_glannuity_20260924_v1/`.
- **Key-run releases.** `20260923-keyruns-v1` (273 files) is superseded by `20260924-keyruns-v2`
  (tree `a4b5e1f5`, 437 KB transferred, the rest hardlinked to v1), which carries the renames, the
  EUR water rule and a pandas-free uniform-rate check (the login-node python that validates
  submissions has no pandas). Smoke stage submitted on `htc`: 8887008 (uniform 5 %) and 8887011
  (Ameli) with QA jobs 8887009 and 8887012; a watcher submits the global stage
  (`--global-locations`, 16,509 cells, run id `20260924-v2`) as soon as both QA reports pass.
- **Smoke QA passed for both key runs; global stage submitted automatically** (run id
  `20260924-v2`, 16,509 cells, four longitude shards of 48 CPUs / 370 GB each on `htc`): uniform
  5 % shards 8887019 to 8887022 with QA 8887023; Ameli shards 8887025 to 8887028 with QA 8887029.
  Legacy array 13237736 running on `arc` in parallel.

## 28 September 2026

- **Global key runs of 24 Sep failed at the shard stage** (both scenarios, all eight shards,
  within five minutes of starting): Gurobi barrier without crossover (the fast default of
  `model/main.py`) ended `suboptimal` at one hard cell per shard, (-34, 24), (-51, 166),
  (-53, -73), (-18, -150), and `fail_fast` aborted the shard, leaving 8 finished cells per shard
  and `*_failed_*.csv` lists of the untouched cells. The smoke stage could not have caught this
  (three easy cells, one week). Fix: `model/run_global.py` now retries a cell on a fresh network
  with the robust Gurobi settings already used by the supply pilots (`Method=1`,
  `DualReductions=0`, `NumericFocus=3`, `InfUnbdInfo=1`) after a `suboptimal`, numerical or
  infeasible-or-unbounded termination, once; results carry `solver_numerical_retry` and
  `solver_retry_termination`. Tests in `tests/test_global_numerical_retry.py`. The wrapper gained
  `--diagnostic-locations`; `inputs/keyruns_numerical_retry_cells.csv` lists the four cells for
  a full-year diagnostic before the global resubmission (run id `20260928-v3`).
- **Legacy key run array 13237736 completed** on `arc`: 16 shards, 3 h 12 min to 3 h 40 min each,
  15,377 cell directories. Post-processing (stated-method supplier table on the centred 2 % table,
  flat water 3.3245 USD2018/t, contract) waits for the green-lory surfaces so that the four
  networks are submitted together.
- **Supply pilots v5 completed on 23 Sep** (13225807 fixed-only 4 h 03 min, 13225809 tracking-only
  5 h 33 min; 21 points each, fetched to `land_reconcile_20260914_v1/arc_received_colocated_v5/`,
  152 KB). Co-located land, September south-west land table (mislabelled; to be rerun on the centred
  build). Fixed-only: Atacama 234.7 EUR/t up to 4.0 Mt/yr and 238.2 at 4.5; northwest Australia
  251.8 to 1.75 Mt/yr, 255.2 at 2.0, infeasible at 2.5; central Australia 213.2 at 0.25 rising to
  237.1 at 2.0. Tracking-only: Atacama 222.8 to 3.0 Mt/yr, 225.3 at 4.0; northwest Australia 240.8
  to 1.25, 244.0 at 1.75, infeasible at 2.5; central Australia 208.6 at 0.25 to 223.9 at 1.5.
- **Retry diagnostic submitted** (release `20260928-keyruns-v3`, 277 files, tree `538a0672`):
  full-year solves of the four suboptimal cells, 8923267/8923268 (uniform 5 %, htc) and
  13287849/13287850 (Ameli, arc). A watcher resubmits the global stage (`20260928-v3`) when both
  QA reports pass. Stale QA jobs 8887023 and 8887029 of the failed 24 Sep stage cancelled.
- **Supply pilots rerun on the centred land table** (user question: is the mislabelled table being
  fixed for the pilots). Configs `reconciliation/land/supply_curve/config_v3_center{,_tracking}.json`
  pin `max_capacities_center_2pct_slope15.csv` (sha `89a60166`); same Way overlay, plant bundle,
  Ameli finance and hashed weather subsets as v5 (`fixed_pv_3cells_v2/weather_used`), run from
  release `20260928-keyruns-v3`. Jobs 8923392 (`supply_curve_v6_fixed_center`, htc) and 13287957
  (`supply_curve_v6_track_center`, arc), 12 CPUs / 32 GB / 12 h; v5 took 4 h and 5.5 h.
- **Why NW Australia ends at about 2 Mt/yr at a 2 % share** (v5, fixed-only): the site's design is
  pure PV (5,393 MW and 64.6 km² per Mt/yr); the shared budget is 113.1 km², so PV alone reaches
  1.75 Mt/yr at 251.8 EUR/t; at 2.0 Mt/yr the optimiser swaps in the whole wind allowance
  (566 MW on 113 km², of which only 3 % competes with PV) and LCOA rises to 255.2; at 2.5 Mt/yr
  both caps are exhausted and the point is infeasible. The scaled-design rule gives 1.75 Mt/yr
  for the same cell, a slight underestimate because it cannot re-optimise the mix.
- **Cleanup.** Never-used release `20260923-keyruns-v1` removed (files persist as hardlinks in v2
  and v3); `FAILED.md` written in the two aborted 24 Sep global run directories; the local interim
  land table marked `SUPERSEDED.md`; stale 24 Sep QA jobs cancelled.
- First pilot attempt (8923392, 13287957) failed at once: the pilot requires its experiment config
  inside the pinned release. Release `20260928-keyruns-v4` (279 files, tree `82d30415`, adds the two
  configs; 54 KB transferred) and resubmission: 8923430 (fixed-only, htc) and 13287974
  (tracking-only, arc). Release v3 remains the source of the global key-run stage.
- **Retry diagnostic passed for both scenarios** (8923267 uniform 5 %, 13287849 Ameli; 32 and 25
  minutes for four full-year cells): the retry fired at three of four cells in the uniform run and
  two of four in the Ameli run (which cells stall is numerically arbitrary), every retried cell
  reached an optimum, and the four cells are as poor as expected: (-53, -73) 534 to 538 EUR/t,
  (-34, 24) 394 to 397, (-18, -150) 367 to 370, (-51, 166) 232 to 234 with almost no PV. **Global
  stage resubmitted** (run id `20260928-v3`): uniform 5 % shards 8923443 to 8923446 with QA 8923447
  on htc; Ameli shards 13287985 to 13287988 with QA 13287989 on arc. Post-processing script updated
  to run id `20260928-v3` and release v4 and staged in `keyruns_20260923_v1/tools/`.
- **Resubmitted global stage cancelled after 50 minutes** (uniform 5 % shards 8923443 to 8923446
  on htc; Ameli shards 13287985 to 13287988 still pending on arc). Throughput was 3 to 11 cells per
  minute per shard against 71 in the September hourly run. Cause established from the partial shard
  CSVs saved at cancellation (1,492 cells): 6.2 % of cells (93) end SUBOPTIMAL on the first attempt
  and go through the dual-simplex retry, which takes about 15 minutes per cell and dominates the
  wall time; the token server (105 of 4,096 in use) and node placement (all four shards on the
  192-core htc-c076) are not the cause. A probe on a clean node showed even northwest Australia
  (-23, 117) ending SUBOPTIMAL at full year under the DEA fixed-only configuration, so the barrier-
  without-crossover default is fragile for this cost set (cheap tank and battery, fixed PV only:
  a more degenerate dispatch). `model/main.py` now builds solver options in `build_solver_options`
  with a campaign-wide `GREEN_LORY_SOLVER_OPTIONS_JSON` override (tests in
  `tests/test_solver_options.py`); release `20260928-keyruns-v5` (tree `d79b15ce`). Five probes on
  four stalling cells compare first-attempt settings (baseline; barrier with crossover at 1e-8 and
  at 1e-4; homogeneous barrier; concurrent) in `keyruns_20260923_v1/solver_probe_20260928_v2/`.
- **Solver probes (four stalling cells plus northwest Australia, 4 threads, clean arc nodes).**
  Baseline barrier without crossover: every cell optimal in 52 to 68 iterations, 5 to 6 s each
  (the stalls are numerically marginal and not reproducible cell by cell: the same cells that
  ended SUBOPTIMAL inside the shards solved cleanly here). Barrier with crossover: barrier 5 s
  plus crossover 44 to 77 s per cell. Concurrent (Method=3): 50 to 82 s per cell. Two probes
  (crossover at 1e-8, homogeneous barrier) never ran their intended settings because Slurm's
  `--export` splits on commas and truncated the JSON; campaign-wide settings must be exported in
  the submitting shell, not listed in `--export`. Conclusion: keep the 5-second baseline as the
  first attempt and make the retry cheap: barrier with crossover (about one minute) before the
  dual-simplex fallback (about fifteen minutes). Also found: the single-worker path of
  `run_global` bypassed the retry wrapper (the first probe's failure at northwest Australia).
- **Retry chain and resubmission.** `model/run_global.py` now retries a numerical termination first
  with barrier plus crossover (`CROSSOVER_GUROBI_OPTIONS`, about a minute per cell) and only then
  with the dual-simplex fallback; the single-worker path goes through the same wrapper; results
  carry `solver_retry_attempts` and `solver_retry_options`. Release `20260928-keyruns-v6` (tree
  `961fe350`). Global stage resubmitted as run id `20260928-v4`, all on `arc` (one shard per 48-core
  node): uniform 5 % shards 13288331 to 13288334 with QA 13288335; Ameli shards 13288339 to
  13288342 with QA 13288343. A watcher summarises both surfaces and runs
  `tools/postprocess_keyruns_20260924.sh` (contracts, 2 % variant, legacy table, four networks)
  when both QA reports pass.
- **Both key surfaces solved and QA-passed (run id 20260928-v4).** Shards 38 to 95 minutes each on
  arc; 16,509 cells; 15 % of cells needed the crossover retry (2,470 uniform, 2,428 Ameli), none
  the dual-simplex fallback. Uniform 5 %: LCOA p5/p50/p95 317.6 / 387.0 / 740.1 EUR/t, minimum
  229.1; 6,817 cells at or above 1 Mt/yr; 51,151 Mt/yr of scaled-design capacity; water 0.78 %
  of the headline. Ameli: 317.6 / 386.5 / 734.5, minimum 231.0; 6,816 cells; 51,147 Mt/yr. Focal
  cells (uniform / Ameli): Atacama 328.4 / 331.2 EUR/t, 34.3 Mt/yr, pure PV (4,872 MW per Mt);
  northwest Australia 351.3 / 354.2, 11.7 Mt/yr; central Australia 288.9 / 291.3, 3.8 Mt/yr,
  wind-limited (1,342 MW wind, 2,071 MW PV). DEA 2050 costs sit well above the Way 2050
  trajectories used in September (PV 0.29 versus 0.22 EUR/W, electrolyser 327 versus about
  183 kEUR/MW), hence the roughly 40 % higher LCOA than the September central surface at the same
  cells. The uniform 5 % and Ameli surfaces differ by less than 1 % because the Ameli map gives
  5.1 % to most of the cheap regions.
- **Post-processing stopped at the contract export**: `export_lory_surface.py` refuses country
  names without an explicit ISO3 mapping in the September country reference (East Timor, Ivory
  Coast, North Macedonia, Palestine, Republic of the Congo, United Republic of Tanzania, United
  States of America, unassigned coastal cells). Fixed by an alias table in the exporter and an
  explicit rule for unassigned cells; rerun below.
- **Contract export fixed and post-processing completed** (releases `20260928-keyruns-v7` and
  `v8`): `reconciliation/country_reference_20260928_v1.csv` (194 rows: the September reference plus
  the 17 geojson spellings it lacked, provenance in the sidecar JSON) and `export_lory_surface.py
  --drop-unassigned` (cells with no country polygon are dropped and counted in the contract) and
  `--derived-provenance` (a post-hoc land-share variant exports against its parent's QA report
  with the derivation recorded). Contracts under `keyruns_20260923_v1/exports_20260928-v4_v1/`:
  uniform 5 % at 20 %: 6,815 supplier cells at or above 1 Mt/yr (9,692 below); Ameli at 20 %:
  6,814; uniform 5 % at 2 % (capacities x 0.1): 1,363 (15,146 below). Legacy key run table
  (`ll_way2050_ameli_bflat_wflat_landleg2c_track_4h_glannuity`, stated method on the centred 2 %
  table, water 3.32 USD2018/t): Australia 393 eligible cells and 779 Mt/yr against 553 and 1,827
  archived; limiting technology PV 9,943 cells, wind 4,916, none 585 (above 75 N). Four green-
  porpoise runs submitted on `arc` with the September settings (4 CPUs, 64 GB, 2 h, 1 h solver,
  0.1 % target, same-ISO3-or-1000 km routes x 1.1, 1 Mt cutoff): 13296857 (uniform 5 %, 20 %),
  13296860 (Ameli, 20 %), 13296863 (uniform 5 %, 2 %), 13296865 (legacy key run).
- **First key network finished** (13296857, `gpo_gl_dea2050_wacc5_bflat_wflat_land20c_fixed_iso1000_1mt_v1`):
  one-hour solver limit reached at a 0.75 % gap (2.0 million variables, 21,508 constraints; the
  deposited formulation's bound plateaus as in runs A to F, so the 0.1 % QA target fails and
  `failure.json` records "Sparse result failed QA" while every output is complete and consistent:
  port and supplier balance errors below 1e-7). Delivered cost 370.96 USD/t for 595 Mt/yr of demand,
  against 260.65 (archived control A2) and 268 (legacy replication E): the DEA 2050 plant costs
  carry through. Producers: Australia 221.7 Mt/yr, Morocco 105.6, Sudan 80.1, Mauritania 58.8,
  Western Sahara 24.3, United States 21.1, China 21.1, Argentina 11.1, Saudi Arabia 8.1. The
  acceptance-gap decision (1.5 % as deposited, or 0.6 %) is still open with the user.
- **All four key networks finished** (one-hour solver limit each, gaps 0.48 to 0.75 %, hence
  `FAILED` against the 0.1 % QA target; every balance closed). Delivered cost in USD2018/t and
  production by country in Mt/yr (595 Mt/yr of demand):

| Network | USD/t | Gap | AUS | MAR | SDN | MRT | CHL | USA | CHN | ARG | Other leaders |
|---|---|---|---|---|---|---|---|---|---|---|---|
| green-lory, uniform 5 %, 20 % land (13296857) | 370.96 | 0.75 % | 221.7 | 105.6 | 80.1 | 58.8 | 0.1 | 21.1 | 21.1 | 11.1 | Western Sahara 24.3 |
| green-lory, Ameli, 20 % land (13296860) | 372.58 | 0.73 % | 219.5 | 98.3 | 79.1 | 56.6 | 0.1 | 34.4 | 21.1 | 10.3 | Western Sahara 24.3 |
| green-lory, uniform 5 %, 2 % land (13296863) | 416.90 | 0.48 % | 17.1 | 0.3 | 47.6 | 57.7 | 39.2 | 6.3 | 6.5 | 15.5 | Yemen 110.6, Saudi Arabia 72.6, Oman 36.5 |
| legacy key run, green-lory annuity, stated land 2 % (13296865) | 251.59 | 0.50 % | 202.8 | 6.1 | 0.2 | 39.3 | 36.0 | 29.5 | 8.7 | 7.6 | Yemen 88.4, Western Sahara 28.4 |
| archived control A2 (14 Sep) | 260.65 | | 255.3 | | | | | | | | |
| legacy replication E (16 Sep) | 268.16 | | 250.4 | | | | | | | | |

  Reading: at a 20 % share the green-lory DEA surfaces keep Australia first (about 220 Mt/yr) and
  bring in Morocco, Sudan and Mauritania; finance (uniform 5 % against Ameli) moves the delivered
  cost by 0.4 % and shifts some production from Morocco to the United States. At 2 % the same
  plant costs lose Australia (17 Mt/yr) to Yemen, Saudi Arabia and Mauritania and cost 12 % more:
  the land share, not the plant cost, decides the Australian outcome, as in September. The legacy
  key run (Way 2050 CAPEX, our annuity, stated-method land at 2 %) delivers at 251.6 USD/t, 6 %
  below the replication, with Australia at 203 Mt/yr. Summaries fetched to
  `verschuur_reconcile_20260907_v1/arc_received_20260928_keyruns/`.
- **Supply pilots v6 on the centred 2 % land table finished** (8923430 fixed-only about 9 h;
  13287974 tracking-only about 8 h; both QA passed; fetched to
  `land_reconcile_20260914_v1/arc_received_colocated_v6_center/`, 152 KB). Fixed-only: Atacama
  234.7 EUR/t to 3.0 Mt/yr, 238.9 at 4.0, infeasible from 4.5 (v5 on the mislabelled table: 4.5
  feasible); northwest Australia 251.8 to 1.25, 254.1 at 1.75, infeasible from 2.0 (v5: 2.0
  feasible at 255.2); central Australia 213.2 at 0.25 rising to 244.8 at 2.0 (v5: 237.1), 4.3
  infeasible, the 11 % protected share of the exact cell costing land. Tracking-only: Atacama
  222.8 to 3.0, infeasible from 4.0; northwest Australia 240.8 to 1.25, infeasible from 1.75;
  central Australia 208.6 to 223.9 at 1.5, infeasible from 2.0. The centred cells hold a little
  less usable land than the mislabelled labels did at these three sites, so every ceiling moved
  down one step; the unconstrained controls are unchanged (land does not enter the 1 Mt design).
