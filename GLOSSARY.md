# Glossary

Every option, run setting, id token, role, network name and result column used in
`runs.yaml`, `run_store.csv` and the reports, in one place. Names are the
current ones; names they replaced are given in brackets. First written 24 September 2026;
runs, roles, network names, plant bundles and terms added 6 October 2026.

## Looking things up

| Question | Where |
|---|---|
| What is run X, or "network D"? | `runs.yaml`; `green-lory show X` (accepts an id or a report name such as `D`) |
| Which runs exist? | `green-lory list` |
| How do the green-lory runs differ? | `green-lory table` |
| When and where did X run, and what did it give? | `run_store.csv` or `green-lory status X` (one row per run id: jobs, release, status, output directory, headline results); [Run store](#run-store) below |
| What does a token in an id mean? | [Run identifiers](#run-identifiers) below |

## Kinds of run (`runner` in `runs.yaml`)

| Runner | What it is | How it runs |
|---|---|---|
| `green_lory` | A global surface: the green-lory plant optimised at every cell, with LCOA, design and capacity per cell | `arc/submit_lory_sequence.sh --scenario <run id>` (smoke, diagnostic, global stages), which calls `model/run_global.py` |
| `derived` | No new solve: the parent run's LCOA and designs with another land share or capacity rule (`derived_from` names the parent) | post-processing script named in the entry |
| `legacy_lcoa` | The frozen lcoa-opt plant model (Salmon 2023) at the archived supplier cells | `legacy_lcoa/run_legacy_cells.py` through `arc/jobs/08_legacy_global.sh` |
| `supply_pilot` | Finite-site supply curves at the three focal cells: the plant re-solved at increasing ammonia quantities with land enforced inside the solve, giving cost against quantity | `reconciliation/supply_pilot/run_supply_pilot.py` through `arc/jobs/06_land_supply_pilot.sh` |
| network | A green-porpoise shipping network built on one supplier table (the `networks` section) | `legacy_gpo/arc/submit_network_run.sh` of the green-porpoise repository (`legacy_gpo/` is linked here) |

## Roles

| Role | Meaning |
|---|---|
| reconciling | green-lory configured like the legacy plant (Way 2050 costs, 4-hour step, legacy accounting) to show that the two models agree when given the same assumptions |
| improved | the green-lory model of record, with the current modelling choices (`MODEL_CHOICES.md`) |
| legacy replication | the frozen legacy model reproducing the published supplier table; the accepted baseline |
| legacy key run | the legacy plant brought as close to the improved stack as it goes (green-lory annuity, rebuilt land table, water) |
| control | the published (archived) supplier table under the same network settings as the comparisons |
| sensitivity | one setting changed from a base run, named in the role (finance, land share) |

## Network names in the September reports

The September reports (`reconciliation/archive/2026-09/`: `DECISION_20260916.md` section 4, `RUNS.md`,
`HANDOVER_20260915.md`) name network runs by letter. All used the deposited equations, the
MOD-AMB RCP4.5 70 % demand, a 1 Mt/yr supplier cutoff and a one-hour solver limit unless stated.
Results are in USD2018 per tonne delivered.

| Name | Network id (if kept) | Supplier table and setting | Result | Job |
|---|---|---|---|---|
| archival replay | `gpo_archived_replay` | archived table; the 2023 project code, archived pipelines | 254.05 USD/t, Australia 258.1 Mt/yr | 8758215 |
| A | not kept | archived table, archived pipelines | 259.67, 259.4 | 8805878 |
| A2 | `gpo_archived_iso1000_1mt` | archived table, regenerated routes: the **control** for every comparison below | 260.65, 255.3 | 8805879 |
| A3 | not kept | archived table, prefixed demand file | 259.64, 259.9 | 8805882 |
| B | not kept | reconciling surface with the September (mislabelled) land capacity | 274.91, 25.0 | 8805880 |
| C | not kept | September central surface with the September land capacity | 310.21, 1.0 | 8805881 |
| D | `gpo_rep_way2050_flat_amelired_4h_tracking_nominal_h2_legacyrule_iso1000_1mt` | reconciling surface's costs and designs with the **legacy capacity rule**: shows that the capacity rule, not the plant cost, moves the network | 261.23, 238.5 | 8820849 |
| E | `gpo_ll_xcost45_ameli_bflat_wnone_landarch_track_4h_iso1000_1mt` | the **legacy replication** surface: the published network reproduced from the replicated legacy model | 268.16, 250.4 | 8823893 (11-hour form 8823894) |
| F | not kept | legacy costs and designs with the stated land method on the September table | 270.68, 209.8 | 13160450 |
| land share x2, x4 | not kept | surfaces of B and C at 4 % and 8 % of suitable land | see the archived DECISION_20260916.md section 4 | 8824427-8824430 |

The four networks of 28 September (key runs) have no letter; their ids are
`gpo_<supplier run id>_iso1000_1mt`.

## Settings in `runs.yaml`

| Key | Meaning |
|---|---|
| `role`, `runner`, `description`, `notes` | See the two tables above; free text otherwise |
| `results_class` | Subdirectory of the campaign the run is written to (`10_replication`, `30_keyruns`, `50_campaign`) |
| `job_token` | Short token in the Slurm job names |
| `tech_yaml` | Technology and cost YAML in `inputs/tech/`; may `extends:` another YAML, whose values it overrides |
| `plant_dir` | Plant bundle, see [Plant bundles](#plant-bundles) |
| `pv_policy` | PV offered to the optimiser: `fixed` (fixed tilt only) or `both` (fixed and single-axis tracking) |
| `finance_csv`, `finance_csv_sha256` | Finance table: interest rates by country (`iso3`) or by cell (`lat, lon`), per technology or `all` (`--finance-csv`; `inputs/README.md`); the hash pins the file. A run sets this or `interest_rate` |
| `interest_rate` | One interest rate (a fraction, e.g. 0.05) for every technology at every cell (`--interest-rate`), instead of a finance table |
| `site_costs_csv`, `site_costs_csv_sha256` | Optional site-cost table of the run's land table (`green-lory site-costs`): equipment and build multipliers and the water price per cell (`--site-costs-csv`); runs without it are flat |
| `weather_land_correction`, `weather_land_correction_sha256` | Optional factors that rescale the profiles of cells with water to their land (`model/weather_land.py`, `--weather-land-correction`); runs before October 2026 have none |
| `time_step_h`, `temporal_accounting`, `ramp_limit_basis` | Weather step and the matching accounting options, see [Run options](#run-options) |
| `site_costs_in_headline` | Whether water (and land rent, zero so far) enters the headline LCOA; false passes `--exclude-site-costs` |
| `water_eur2020_per_m3` | Flat water price |
| `cost_scope` | Label of the active cost mechanisms (build, finance, water, land rent), recorded in the manifest and checked by the QA gate |
| `land_share` | Share of suitable land open to ammonia (0.20 or 0.02) |
| `land_table`, `land_table_sha256` | Land table (path below the ARC campaign root) and its hash; passed with `--land-csv` |
| `land_constraint`, `capacity_rule`, `land_allocation` | See [Run options](#run-options) |
| `derived_from` | Parent run of a `derived` entry |
| `source`, `harness`, `cells`, `annuity`, `release` | Legacy runs: frozen source, harness command, cell lists, annuity convention, ARC release |
| `config` | Supply pilots: the experiment configuration |
| `report_name` | Networks: the name the September reports used (table above) |
| `supplier_run` | Networks: the run whose supplier table the network reads; `archived` is the published table |
| `equations`, `demand`, `routes`, `maritime_costs`, `supplier_cutoff_mt_per_year` | Networks: green-porpoise formulation, demand scenario, pipeline routing (`iso1000`: same country or within 1,000 km, length x 1.1), shipping costs, smallest supplier admitted |
| `solver_time_limit_s`, `solver_gap_target`, `accepted_gap` | Networks: solver limits, and the optimality gap at which a result is accepted (1.5 %, the deposit's tolerance) |

## Run options

`model/surface.py` (`python -m model.surface --help`), run definitions in `runs.yaml`:

| Option | Values | Meaning |
|---|---|---|
| `--land-constraint` [`--lcoa-land-mode`] | `after_solve` [`postprocess`], `in_solve` [`enforce`] | Where the land budget enters. `after_solve`: the plant is optimised as a 1 Mt/yr reference design without land limits, and land only scales the capacity afterwards. `in_solve`: the land areas bound the wind and PV capacities inside the optimisation, so the design and LCOA respond to land. |
| `--capacity-rule` [`--capacity-method`] | `scaled_reference_design` [`paper_scaled`], `solved_quantity` | How a cell's ammonia capacity is reported. `scaled_reference_design`: the reference design is scaled up until wind, PV or their shared footprint fills the available land (the rule of Salmon 2022 / Verschuur 2024, hence the old name). `solved_quantity`: the production solved under `in_solve` land limits, a quantity that was feasible, not a maximum. |
| `--land-allocation` | `colocated`, `exclusive` | `colocated`: PV and wind share the suitable land; only the wind exclusive fraction (0.03 of the turbine footprint, Denholm 2009) competes with PV. `exclusive`: the September 2026 rule, every wind km² is lost to PV. |
| `--temporal-accounting-mode` | `snapshot_weighted`, `legacy_scaled` | How energy and cost are integrated over snapshots longer than one hour. `legacy_scaled` reproduces the archived lcoa-opt arithmetic and is used only for the replication. |
| `--ramp-limit-basis` | `per_hour`, `legacy_per_snapshot` | Whether Haber-Bosch ramp limits are per hour or per snapshot (replication only). |
| `--exclude-site-costs` | flag | Keeps water and land rent out of the headline LCOA (replication only). |
| `--finance-csv` / `--interest-rate` | path / fraction | The run's interest rates: a finance table by country or by cell, per technology or `all` (`inputs/README.md`), or one rate for every technology and cell. One of the two is required. |
| `--land-csv` | path | Land table from `model/land_table.py`; its `land_competition_fraction` column is the share of suitable land open to ammonia (2 % or 20 %). |
| `--locations-csv` / `--global-locations` | path | Restricts the run to listed `lat,lon` cells (the key runs use onshore cells with suitable land). |
| `--pv-policy` (`pv_policy` in `runs.yaml`) | `both`, `fixed`, `tracking` | Which PV the plant may build; `fixed` switches tracking PV off. |
| `weather_dir` (`runs.yaml`) | path | The weather profile folder; every run uses `data/weather_data/salmon2022_pv_wind_2019_hourly_1deg`. Its profiles average the whole 1-degree cell, sea and lakes included. |
| `--weather-land-correction` (`weather_land_correction`) | path | Factors by month and hour that rescale the profiles of cells with at least 1 % water to the cell's land, from ERA5 at 0.25 degrees. |

## Run identifiers

`<model>_<costs>_<finance>_<build>_<water>_<land>_<pv>[_<step>][_<qualifier>]` for plant runs,
for example `gl_dea2050_wacc5_bflat_wflat_land20c_fixed`, and
`gpo_<supplier run id>_<routes>_<cutoff>` for networks. A derived run (another land share, a
water price change) reuses its parent's LCOA and gets its own id. Extra qualifiers go after the
pv slot (`_explicitcomp`, `_offshore`); anything no token covers goes in the run store's
`notes`. Numeric values (the water price, the WACC of the Ameli map) belong in
`run_store.csv` and in the run manifest, not in the id.

| Token | Meaning |
|---|---|
| `gl`, `ll`, `gpo` | green-lory plant model, legacy-lcoa plant model, green-porpoise network |
| `rep_...` | the reconciling run, which keeps its September id (`rep` for replication mode) |
| `dea2050`, `dea2030`, `way2050`, `xcost45` | technology cost basis (see [Sources and terms](#sources-and-terms)) |
| `wacc5`, `ameli`, `amelired`, `amelilc` | uniform 5 % WACC; Ameli reduced WACC by country (`ameli` in key run 2, `amelired` in the reconciling run); Ameli low-carbon (current) WACC by country (campaign runs F and S2) |
| `bflat`, `bspat` | build-cost multiplier 1 everywhere, or depth x remoteness x labour |
| `wflat`, `wspat`, `wnone` | uniform baseline water (2 EUR2020/m³ from 24 Sep 2026), spatial water, no water cost |
| `land20c`, `land2c`, `land20oct`, `land2sw`, `landleg2`, `landleg2c`, `landarch` | land share and land build (`c` centred build of 23 Sep 2026, `oct` the October build of 8 Oct 2026 with slope classes up to 20 degrees and protected land counted on land only, `sw` the mislabelled September build, `leg` the legacy land step or the stated method, `arch` the archived capacity column) |
| `fixed`, `track`, `both` | PV technologies offered |
| `4h` | four-hour weather step (default is hourly) |
| `glannuity` | legacy model with green-lory's per-technology annuity instead of the workbook factor |
| `legacyrule` | capacity from the recovered legacy rule instead of green-lory's land method |
| `pilot3cells` | supply pilot at the three focal cells |
| `iso1000`, `1mt` | networks: routes within the same country or 1,000 km; 1 Mt/yr supplier cutoff |

## Run store

`run_store.csv` has one row per run id: `run_id, model, cost_basis, finance,
build, water, land, land_table, pv, time_step_h, allocation, cells, weather, release, location,
job_ids, status, submitted_utc, upstream_run, network_runs, notes`. Statuses: `planned`,
`submitted`, `running`, `merged`, `qa_passed`, `exported`, `done`, `superseded`, `failed`,
`deferred`. A run defined in `runs.yaml` gets its row (status `planned`) before it is submitted;
a test checks that every definition has one.

## Plant bundle

The plant is one directory of PyPSA component tables, `model/plants/ammonia_plant_1mt/`: the
components and how they connect, with no technology values. Each run sets those from its
technology YAML: the capital costs, every link's efficiencies (from the YAML's
`carriers_in`/`carriers_out` recipe) and the operating values (Haber-Bosch minimum load and ramp
limits, the battery standing loss); the ammonia demand comes from
the reference output in `inputs/parameters.yaml`, and the run's `pv_policy` decides whether
tracking PV may be built. Until 7 October 2026 there were three bundles
(`basic_ammonia_plant_2050`, `_2050_way`, `_2050_way_tracking`), which differed only in four
efficiencies and the tracking flag; the September reports name them.

## Sources and terms

| Term | Meaning |
|---|---|
| Way 2050 | Technology costs after Way et al. (2022), 2050 values, as implemented in lcoa-opt; green-lory's EUR2020 version is `inputs/tech/tech_config_ammonia_plant_2050_way_eur.yaml` (overnight costs reverse-engineered from the lcoa-opt workbook) |
| x_Cost, `xcost45` | The lcoa-opt cost workbook `x_Cost_Forecasting_20230831.xlsx`, RCP4.5 sheet, 2050 column: the Way-based CAPEX that the frozen legacy model reads |
| DEA 2050 | Danish Energy Agency technology catalogue, 2050 values (`inputs/tech/tech_config_ammonia_plant_2050_dea.yaml`) |
| Ameli reduced WACC | Country WACC from Ameli et al. (2021) Fig. 2, reduced case, by TIAM-UCL region (`inputs/finance/ameli2021_reduced_wacc_by_country.csv`) |
| Ameli low-carbon WACC | Country WACC from Ameli et al. (2021) Fig. 2, low-carbon (current) case, by TIAM-UCL region (`inputs/finance/ameli2021_lowcarbon_wacc_by_country.csv`) |
| Workbook annuity | The legacy model's single annuity-due factor 0.1143 (8 %, 20 years, 2 % O&M) applied to every technology; green-lory annualises per technology with its own lifetime and O&M (`glannuity` in a legacy id) |
| Nominal compressor, bundled H2 storage | The legacy cost convention: hydrogen-compressor CAPEX nominal only, compression cost bundled into the hydrogen-store cost. The reconciling run uses it |
| Explicit compressor, DEA tank | The improved convention: the compressor costed explicitly and the store at the DEA 2050 tank-only cost, so compression is not counted twice |
| Archived table | `c_NH3_cost_4.5.csv` in green-porpoise (git 0a63616): the supplier input of Verschuur et al. (2024), produced by Salmon in spring 2023 by code that is in no repository |
| Table 2 | Salmon (2022) Table 2: share of each MODIS land-cover class usable for wind and for PV |
| Stated method | The land-to-capacity method as the papers state it (Salmon 2022 section 2.2.1, Verschuur 2024 section 4.7): 200 km²/GW wind, latitude-packed PV, complete wind/PV overlap, 2 % of the suitable land after WDPA and slope exclusions |
| Recovered rule | The rule that reproduces the archived `Max_capacity` column: 140 MW/km² PV at every latitude, 7.3 MW/km² wind, complete overlap, no exclusions. Found empirically; no source for its constants exists |
| Reference design | The cheapest 1 Mt/yr plant at a cell, solved without land limits |
| Centred build | A land table built with `cell_anchor=center` (fixed 23 Sep 2026). Earlier green-lory tables (`sw`) give each cell the suitability of the cell one degree south |
| Focal cells | Atacama (−23, −69), northwest Australia (−23, 117), central Australia (−21, 135): the cells whose changes drove the network divergence |
| MOD-AMB | The Verschuur et al. (2024) scenario used as the primary endpoint, with RCP4.5 and 70 % ammonia adoption |
| Supplier contract | The network input exported from a surface: `contract.json` plus `suppliers_USD2018.csv` |
| Release | An immutable copy of the source tree on ARC (`green-lory-releases/<id>`) with a hashed file inventory; every ARC job runs from one |
| Campaign | An output tree on ARC (`green-lory-campaigns/<campaign>/`); runs are written below it by results class, run id and stage |
| Accepted gap | The network optimality gap at which a result is accepted: 1.5 %, the public deposit's own tolerance (decided 5 Oct 2026) |

## Result columns worth knowing

| Column | Meaning |
|---|---|
| `lcoa_eur_per_t` | Headline levelised cost of ammonia at the plant gate (EUR2020/t); includes water when `site_costs_in_headline` is true, never land rent so far. |
| `lcoa_plant_eur_per_t` | The same without site costs. |
| `water_cost_usd_per_m3`, `water_cost_pct` | Water price applied (USD2020, the currency of the spatial input files; the manifest records the EUR value) and its share of the headline. |
| `build_cost_multiplier`, `equipment_cost_multiplier` | Capacity-weighted multipliers on the build and equipment shares of CAPEX (1.0 in flat runs). |
| `interest_rate_<tech>` | WACC used per technology after overrides. |
| `annual_ammonia_production_t` | Reference-design production (1 Mt/yr). |
| `scaled_design_max_onshore_ammonia_capacity_t` [`paper_scaled_max_onshore_ammonia_capacity_t`] | Capacity of the scaled reference design on onshore land. |
| `scaled_design_max_gridless_onshore_ammonia_capacity_t` [`paper_scaled_max_gridless_...`] | The same, counting only production that needed no grid imports (all of it: the grid is off); this is what the supplier contract exports as `Max_capacity` (Mt/yr). |
| `scaled_design_limiting_constraint` | Which budget stopped the scaling: `wind`, `solar` or `renewable_union`. |
| `max_ammonia_capacity_t` and the other `max_*` columns | An older capacity family from separate per-technology caps, in surfaces before 7 October 2026; no longer written. |
| `land_constraint`, `capacity_rule`, `land_allocation` | The options above, echoed per row. |
| `is_gridless_feasible`, `grid_energy_share` | Whether the cell's design ran without grid imports (always, since the grid is off in the plant). |

## Land table columns (`model/land_table.py`)

| Column | Meaning |
|---|---|
| `cell_anchor` | Always `center`: the cell labelled (lat, lon) covers lat ± 0.5, lon ± 0.5, like the weather nodes. Tables of September 2026 and earlier were labelled by the south-west corner or the north edge and are refused. |
| `onshore_land_pct`, `area` | MODIS land fraction of the cell and the cell's spherical area (km²). |
| `protected_area_pct`, `slope_suitable_land_pct`, `land_exclusion_factor` | Protected land as a share of the cell (WDPA polygons on the land-cover grid, water inside them left out; the whole protected area until 8 October 2026), share of land at or below the slope limit (20 degrees, 15 until October 2026), and their combined factor (multiplied by the land share in the 2 % / 20 % tables). |
| `land_pct_slope_upto_8p5deg`, `_8p5_to_14deg`, `_14_to_20deg`, `_above_20deg` | Share of the cell's land in each slope class, from which the site costs take the slope factor. |
| `solar_area_km2`, `wind_onshore_area_km2`, `renewable_union_area_classwise_nested_v1_km2` | Suitable area per technology after exclusions and share; the union is a classwise nested lower bound of the physical union. |
| `solar_density_mw_per_km2`, `wind_density_mw_per_km2` | Latitude-packed First Solar fixed-tilt density (about 106 MW/km² at the equator) and 5 MW/km² wind. When the table has no `solar_density_method` column, `model/land.py` recomputes the PV density from the YAML instead (73.6 MW/km² at 23°); which rule to keep is open (`MODEL_CHOICES.md`). |

## Files and where they live

- Land tables: `green-lory-campaigns/land_center_20260923_v1/` on ARC (100 %, 2 %, 20 %).
- Campaign runs: `green-lory-campaigns/<campaign>/<class>/<scenario>/runs/<run-id>/<stage>/` with `manifest.json`, `shards/`, `merged/`, `qa/`.
- Network runs: `green-lory-campaigns/verschuur_reconcile_20260907_v1/networks/` on ARC; local copies under `results/campaigns/verschuur_reconcile_20260907_v1/arc_received_*/` (paths per run in `run_store.csv`).
- Supplier contracts for green-porpoise: `contract.json` plus `suppliers_USD2018.csv` (USD2018 because the network model keeps the deposited currency).
- Legacy replication package: `legacy_lcoa/` (frozen lcoa-opt source, harness, land step, supplier-table builder).
