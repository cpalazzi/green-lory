# Glossary of options, identifiers and result columns

Written 24 September 2026 for a later user of the green-lory stack. Names below are the
current ones; the September 2026 names they replaced are given in brackets.

## Run options (`model/run_global.py`, `arc/submit_lory_sequence.sh`)

| Option | Values | Meaning |
|---|---|---|
| `--land-constraint` [`--lcoa-land-mode`] | `after_solve` [`postprocess`], `in_solve` [`enforce`] | Where the land budget enters. `after_solve`: the plant is optimised as a 1 Mt/yr reference design without land limits, and land only scales the capacity afterwards. `in_solve`: the land areas bound the wind and PV capacities inside the optimisation, so the design and LCOA respond to land. |
| `--capacity-rule` [`--capacity-method`] | `scaled_reference_design` [`paper_scaled`], `solved_quantity` | How a cell's ammonia capacity is reported. `scaled_reference_design`: the reference design is scaled up until wind, PV or their shared footprint fills the available land (the rule of Salmon 2022 / Verschuur 2024, hence the old name). `solved_quantity`: the production solved under `in_solve` land limits, a quantity that was feasible, not a maximum. |
| `--land-allocation` | `colocated`, `exclusive` | `colocated`: PV and wind share the suitable land; only the wind exclusive fraction (0.03 of the turbine footprint, Denholm 2009) competes with PV. `exclusive`: the September 2026 rule, every wind km² is lost to PV. |
| `--temporal-accounting-mode` | `snapshot_weighted`, `legacy_scaled` | How energy and cost are integrated over snapshots longer than one hour. `legacy_scaled` reproduces the archived lcoa-opt arithmetic and is used only for the replication. |
| `--ramp-limit-basis` | `per_hour`, `legacy_per_snapshot` | Whether Haber-Bosch ramp limits are per hour or per snapshot (replication only). |
| `--exclude-site-costs` | flag | Keeps water and land rent out of the headline LCOA (replication only). |
| `--override-csv` | path | Per-cell, per-technology `interest_rate` (and optionally build multipliers, water, land rent). Interest-only files are the *flat* cases. |
| `--land-csv` | path | Land table from `model/land_processing.py`; its `land_competition_fraction` column is the share of suitable land open to ammonia (2 % or 20 %). |
| `--locations-csv` / `--global-locations` | path | Restricts the run to listed `lat,lon` cells (the key runs use onshore cells with suitable land). |
| `--cell-anchor` (land build) | `center`, `southwest` | Where the `(latitude, longitude)` label sits in its 1-degree cell. `center` matches the weather nodes and the legacy land step. |

## Run identifiers

`<model>_<costs>_<finance>_<build>_<water>_<land>_<pv>[_<step>][_<qualifier>]`, see
`reconciliation/RUN_STORE.md`. Numeric values (the water price, the WACC of the Ameli map)
belong in `reconciliation/run_store.csv` and in the run manifest, not in the id.

| Token | Meaning |
|---|---|
| `gl`, `ll`, `gpo` | green-lory plant model, legacy-lcoa plant model, green-porpoise network |
| `dea2050`, `dea2030`, `way2050`, `xcost45` | technology cost basis |
| `wacc5`, `ameli` | uniform 5 % WACC, or Ameli reduced WACC by country |
| `bflat`, `bspat` | build-cost multiplier 1 everywhere, or depth x remoteness x labour |
| `wflat`, `wspat`, `wnone` | uniform baseline water (2 EUR2020/m³ from 24 Sep 2026), spatial water, no water cost |
| `land20c`, `land2c`, `land2sw`, `landleg2c`, `landarch` | land share and land build (`c` centred build of 23 Sep 2026, `sw` the mislabelled September build, `leg` the legacy step, `arch` the archived capacity column) |
| `fixed`, `track`, `both` | PV technologies offered |
| `4h` | four-hour weather step (default is hourly) |
| `glannuity` | legacy model with green-lory's per-technology annuity instead of the workbook factor |

## Result columns worth knowing

| Column | Meaning |
|---|---|
| `lcoa_eur_per_t` | Headline levelised cost of ammonia at the plant gate (EUR2020/t); includes water when `site_costs_in_headline` is true, never land rent so far. |
| `lcoa_plant_eur_per_t` | The same without site costs. |
| `water_cost_usd_per_m3`, `water_cost_pct` | Water price applied (USD2020, the currency of the spatial input files; the manifest records the EUR value) and its share of the headline. |
| `build_cost_multiplier` | Multiplier on the build share of CAPEX (1.0 in flat runs). |
| `interest_rate_<tech>` | WACC used per technology after overrides. |
| `annual_ammonia_production_t` | Reference-design production (1 Mt/yr). |
| `scaled_design_max_onshore_ammonia_capacity_t` [`paper_scaled_max_onshore_ammonia_capacity_t`] | Capacity of the scaled reference design on onshore land. |
| `scaled_design_max_gridless_onshore_ammonia_capacity_t` [`paper_scaled_max_gridless_...`] | The same, counting only production that needed no grid backstop; this is what the supplier contract exports as `Max_capacity` (Mt/yr). |
| `scaled_design_limiting_constraint` | Which budget stopped the scaling: `wind`, `solar` or `renewable_union`. |
| `land_constraint`, `capacity_rule`, `land_allocation` | The options above, echoed per row. |
| `is_gridless_feasible`, `grid_energy_share` | Whether the cell's design ran without the grid backstop. |

## Land table columns (`model/land_processing.py`)

| Column | Meaning |
|---|---|
| `cell_anchor` | `center` or `southwest`, see above. |
| `onshore_land_pct`, `area` | MODIS land fraction of the cell and the cell's spherical area (km²). |
| `protected_area_pct`, `slope_suitable_land_pct`, `land_exclusion_factor` | WDPA share of the cell, share of land at or below 15 degrees, and their combined factor (multiplied by the land share in the 2 % / 20 % tables). |
| `solar_area_km2`, `wind_onshore_area_km2`, `renewable_union_area_classwise_nested_v1_km2` | Suitable area per technology after exclusions and share; the union is a classwise nested lower bound of the physical union. |
| `solar_density_mw_per_km2`, `wind_density_mw_per_km2` | Latitude-packed First Solar fixed-tilt density (about 106 MW/km² at the equator) and 5 MW/km² wind. |

## Files and where they live

- Land tables: `green-lory-campaigns/land_center_20260923_v1/` on ARC (100 %, 2 %, 20 %).
- Campaign runs: `green-lory-campaigns/<campaign>/<class>/<scenario>/runs/<run-id>/<stage>/` with `manifest.json`, `shards/`, `merged/`, `qa/`.
- Supplier contracts for green-porpoise: `contract.json` plus `suppliers_USD2018.csv` (USD2018 because the network model keeps the deposited currency).
- Legacy replication package: `reconciliation/legacy_lcoa/` (frozen lcoa-opt source, harness, land step, supplier-table builder).
