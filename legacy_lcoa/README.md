# legacy_lcoa/: the legacy-lcoa plant model, reproduced

End-to-end reconstruction of the plant model that produced the supplier table of Verschuur et
al. (2024) (`green-porpoise/data/c_NH3_cost_4.5.csv`, Salmon 2023): its `LCOA` column from the
frozen lcoa-opt code, and its `Max_capacity` column from a land-to-capacity rule that was never
in any repository and was recovered empirically. Kept as the replication baseline of the
published network; green-lory is the model of record (`../reconciliation/DECISION.md`).

Read [PROVENANCE.md](PROVENANCE.md) first for what is and is not recoverable. The replication
reports are in `reports/`: [LEGACY_REPLICATION_20260915.md](reports/LEGACY_REPLICATION_20260915.md)
(tested configurations, the recovered rule) and
[GLOBAL_SURFACE_20260916.md](reports/GLOBAL_SURFACE_20260916.md) (global surface and supplier
table).

## Contents

| Path | Role |
|---|---|
| `source/94de8ce/` | Frozen lcoa-opt at commit 94de8ce (3 May 2023), the state that replicates the archived table: `git archive`, nothing edited, plus the cost workbooks (`GeneralSteelData_20230831.xlsx`, `x_Cost_Forecasting_20230831.xlsx`) and `SOURCE_IDENTITY.md`. Both lcoa-opt commits remain in github.com/cpalazzi/lcoa-opt |
| `environment.legacy-min.yaml` | Conda environment: PyPSA 0.25.1 with the Pyomo `lopf` path, **Pyomo 6.5.0** (later Pyomo silently returns a zero objective), Gurobi 11. ARC: `../arc/jobs/07_build_legacy_env.sh` |
| `extract_weather_store.py` | Step 1: per-cell hourly profiles from the nine 2019 NetCDF files into a compact store (values unchanged; needed on network file systems). ARC: `../arc/jobs/09_extract_weather_store.sh` |
| `run_legacy_cells.py` | Step 2: the frozen model at listed cells; one directory per cell with `summary.json`, solved component tables and a manifest with every input hash. ARC: `../arc/jobs/08_legacy_global.sh` (16 shards) |
| `legacy_land_areas.py`, `land_core.py` | Step 3, the missing land step: MODIS class fractions x Table-2 suitability x spherical pixel area x 2 % share, per technology, centred 1-degree cells, no exclusions |
| `legacy_capacity.py` | Step 4: the recovered rule (140 MW/km² PV, 7.3 MW/km² wind, complete overlap) as a library, and `apply-lory`, which applies it to a green-lory surface |
| `build_legacy_supplier_table.py` | Step 5: merge shards, apply a capacity rule (`--rule archived_table_reproduction` or `stated_method`), write the supplier table in the archived column contract plus a `gpo_export/` contract for green-porpoise |
| `collect_summaries.py`, `compare_runs.py` | Gather per-cell summaries into one `summaries.jsonl`; compare two executions cell by cell (tests in `../tests/test_legacy_surface_tools.py`) |
| `cells_3.csv`, `cells_sample_v1.csv`, `cells_archived_all_v1.csv`, `shards/` | The three focal cells; the 563-cell latitude-stratified sample; all 15,377 archived supplier cells with Ameli WACC by country; the 16 global shards |
| `diagnostics/` | Scripts used to recover the capacity rule and check capacity factors |
| `debug_two.py` | The check that Pyomo 6.5.0 restores the objective that later Pyomo zeroes (cited in the replication report) |

Generated evidence lives under `../results/campaigns/legacy_lcoa_20260915_v1/` and, for the key
run, `../results/campaigns/keyruns_20260923_v1/`, never inside this directory.

## The two runs kept

| Run id (`../runs.yaml`) | What it is | Harness options |
|---|---|---|
| `ll_xcost45_ameli_bflat_wnone_landarch_track_4h` | **Replication**: reproduces the archived table and, as network E, the published network | `--era may2023 --variants stated_4h_mean --enable-tracking 1.0587 --capex-source xcost45`, per-cell Ameli WACC, workbook annuity; capacity by the recovered rule |
| `ll_way2050_ameli_bflat_wflat_landleg2c_track_4h_glannuity` | **Key run**: the legacy plant as close to the improved green-lory stack as it goes | as above plus `--annuity green_lory`; capacity by the stated method on the centred 2 % land table; water 3.32 USD2018/t added after the solve |

The replicating configuration: frozen 3 May 2023 code; overnight CAPEX from the RCP 4.5 sheet of
`x_Cost Forecasting.xlsx` (2050 column) annualised with the workbook's own 8 %/20-year/2 % O&M
annuity-due factor 0.1143 and rescaled to the country WACC (5.1 % for Chile and Australia);
single-axis tracking enabled at 1.0587 x the fixed-PV cost; four-hour block means (2,190
periods); 1 Mt/yr reference load (713.47 MW HHV). Every departure from the frozen code is a
named harness option recorded in the manifest.

## Running the pipeline

From the repository root, in the legacy environment:

```sh
# 0. environment
conda env create -f legacy_lcoa/environment.legacy-min.yaml

# 1. weather store for the cells of interest (nine-file stack, see ../data/README.md)
python legacy_lcoa/extract_weather_store.py --weather-dir data/weather_data/salmon2022_pv_wind_2019_hourly_1deg \
  --cells legacy_lcoa/cells_archived_all_v1.csv --output <new store dir> --all

# 2. plant model (the global surface runs one shard per ARC array task: arc/jobs/08_legacy_global.sh)
python legacy_lcoa/run_legacy_cells.py --era may2023 --variants stated_4h_mean \
  --enable-tracking 1.0587 --capex-source xcost45 --weather-store <store dir> \
  --cells legacy_lcoa/cells_3.csv --solver gurobi --threads 4 --output <new run dir>
#    key run: add --annuity green_lory

# 3. land areas (MODIS MCD12C1 2022 C6.1)
python legacy_lcoa/legacy_land_areas.py --modis data/MCD12C1.A2022001.061.2023244164746.hdf \
  --cells legacy_lcoa/cells_archived_all_v1.csv --output <new dir>

# 4+5. supplier table: the archived reproduction, or the stated method on a land table
python legacy_lcoa/build_legacy_supplier_table.py --rule archived_table_reproduction \
  --run <run dir or summaries.jsonl> --land <legacy_land_areas.csv> \
  --archived <green-porpoise git 0a63616>/data/c_NH3_cost_4.5.csv \
  --cells legacy_lcoa/cells_archived_all_v1.csv --output <new dir>
python legacy_lcoa/build_legacy_supplier_table.py --rule stated_method --anchor center \
  --run <run dir> --land <centred 2 % land table> --archived <...>/c_NH3_cost_4.5.csv \
  --cells legacy_lcoa/cells_archived_all_v1.csv --output <new dir>
```

The network step consumes `gpo_export/contract.json` through
`python -m networks.replay --supplier-contract` (`../networks/replay.py`; ARC:
`../arc/submit_network_run.sh --contract <contract.json>`).

## Expected values

USD2018/t and Mt/yr; regenerate with the commands above and compare.

| | Atacama (−23, −69) | NW Australia (−23, 117) | Central Australia (−21, 135) |
|---|---|---|---|
| archived table | 212.92; 9.77 | 233.69; 4.02 | 226.39; 4.32 |
| replication | 221.33; 9.43 | 242.64; 3.97 | 234.04; 2.42 |
| key run (stated method, centred 2 %, with water) | 203.41; 5.17 | 221.57; 2.18 | 214.60; 1.51 |

Globally, the replication solves all 15,377 archived cells (ARC array 8821144, zero failures;
a Mac execution agrees to 1e-14); the LCOA ratio to the archived table is 1.040 at the median
of the 563-cell sample (IQR 1.032-1.057). Networks: replication 268.16 USD/t with Australia
250.4 Mt/yr (archived-table control 260.65 and 255.3); key run 251.59 USD/t with Australia
202.8 Mt/yr.

## Two capacity rules: as stated, and as archived

| Rule | Definition | Result against the archived table |
|---|---|---|
| `stated_method` | what Salmon 2022 section 2.2.1 and Verschuur 2024 section 4.7 state: 200 km²/GW wind (5 MW/km²), PV packed by latitude after van de Ven 2021 with a First Solar module (about 106 MW/km² at the equator, 78 at 23°), complete wind/solar overlap, 2 % of the Table-2 suitable area after WDPA and > 15° slope exclusions | the consistent legacy-lcoa; key run on the centred table: Australia 393 eligible cells and 779 Mt/yr against 553 and 1,827 archived |
| `archived_table_reproduction` | recovered constants: 140 MW/km² for all PV without latitude dependence, 7.3 MW/km² wind, complete overlap, no exclusions | capacity ratio median 0.97; reproduces the published input but is inconsistent with the stated method; no source for its constants was found |

The 140 MW/km² lies between the module-only density of the stated First Solar module (181
MW/km², panels touching) and the stated equatorial value (106): a ground-coverage ratio of
about 0.77 at every latitude, so the latitude-dependent row pitch of the prose was absent from
the run that wrote the table.

## Known residuals

- LCOA: a uniform +4 % (finance convention) plus a few per cent at wind-using cells and about
  +15 % north of 60°. The archived cost ordering of the focal cells is reproduced.
- Capacity: PV-dominated cells median ratio 0.998, wind-dominated 0.996; mixed wind/solar
  cells about 20 % low at the median because the replicated designs use more wind than
  Salmon's. The two densities are empirical, not the papers' stated values.
- Not recoverable from local evidence: Salmon's MODIS year, any exclusion he applied, the exact
  PV-density function, the definition of `Electricity_Cost_Frac`, and the runner that wrote the
  table.
