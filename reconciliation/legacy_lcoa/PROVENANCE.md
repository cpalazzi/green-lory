# legacy-lcoa provenance

15 September 2026. Established from the local repositories only (no ARC access
needed): `shipping_sprint/lcoa_model/lcoa-opt` (git), `green-porpoise` (git)
and the papers. Every statement below names its evidence so it can be rechecked.

## 1. What the shipping network actually consumed

The supplier table behind the Verschuur et al. (2024) MOD-AMB network is
`green-porpoise/data/c_NH3_cost_4.5.csv` (columns `Index, Latitude, Longitude,
iso3, country, LCOA, Production, Max_capacity, Electricity_Cost_Frac`; 15,377
rows; `Production` is 1,000,000 everywhere; `Index` is `<lat>_<lon>_1000000.0`).
For the three test cells it holds:

| Cell | LCOA (USD/t) | Max_capacity (Mt/yr) | Electricity_Cost_Frac |
|---|---:|---:|---:|
| Atacama (−23, −69) | 212.92 | 9.769406259 | 0.304209428 |
| NW Australia (−23, 117) | 233.69 | 4.019748515 | 0.329227989 |
| Central Australia (−21, 135) | 226.39 | 4.324243441 | 0.326934196 |

**These exact values were committed to green-porpoise on 2 June 2023**
(commit `0c00555`, "allow data uploads", author Aman Majid; columns then were
`Index, LCOA, Latitude, Longitude, Production, Max_capacity, Electricity_Cost_Frac`),
re-committed with iso3/country columns on 9 August 2023 (`db9de04`), removed
from the tree on 4 June 2024 (`f4f4c14`) and re-added on 25 February 2026
(`1052bf5`). `c_NH3_cost_2.6.csv` (HIGH-AMB) sits beside it. The numbers occur
nowhere in the lcoa-opt repository or its results (checked with grep for the
capacity `9.769406259` and for `212.92` at that coordinate).

Conclusion: the shipping input is **Nicholas Salmon's spring-2023 global run**
(the `Index`/`Production` format is his single-site output convention), produced
before Carlo's first lcoa-opt commit (31 August 2023). Neither the run script
nor the land-to-capacity code that produced `Max_capacity` is in any local
repository. The paper's Methods 4.6-4.7 and Salmon & Bañares-Alcántara (2022)
section 2.2.1 are the only description of that pipeline, so the replication has
to recreate it from the papers (the decision taken on 15 September).

## 2. The legacy-lcoa code state that produced the LCOA column

`lcoa-opt` history: initial commit by `nsalmon11` on 9 March 2023; his last
commit is `cd56c11` (6 June 2023, "Commit before adding alternative grid supply
cost for ammonia plant"); Carlo's commits start 31 August 2023. The frozen copy
in `source/cd56c11/` is therefore the closest surviving state to the June 2023
run. Relevant properties of that state (all verified in the files):

- `Basic_ammonia_plant/loads.csv`: `p_set = 713.4703196 MW` ammonia (HHV
  6.25 MWh/t), i.e. **1 Mt/yr**. Carlo changed it to 7,134.703196 MW (10 Mt/yr)
  on 17 November 2023 (`b47ef15`, commit message misleadingly says "10^6 tonnes").
- `generators.csv`: `Wind` and `Solar` extendable at placeholder capital cost 1;
  **`SolarTracking` is `p_nom_extendable = FALSE` with capital cost 1e9** and no
  cost row in the workbook, so the June 2023 plant had **no tracking PV** despite
  the paper listing it as a technology. `Grid` is fixed at 0 MW. `RampDummy`
  (10 GW, marginal cost 1) feeds the ramp-penalty bus.
- `links.csv`: electrolysis efficiency 0.74 (overridden by the workbook),
  `HydrogenCompression` draws 5 % of the hydrogen flow as electricity
  (`efficiency2 = -0.05`) at capital cost 1,000 USD/MW/yr, `HB` converts 1 MW of
  power into 6.25 MW ammonia while consuming 7.092 MW hydrogen (HHV), with a
  30 % minimum load and 40 % ramp limits per snapshot, `BatteryInterfaceOut`
  carries the PCS cost (148,000, overridden), `HydrogenFuelCell` 120,000 (overridden).
- `stores.csv`: ammonia store 36 USD/MWh/yr (overridden to 5.76), compressed
  H2 store 3,620 (overridden to 2,524.05), battery 0 (overridden to 8,915.99).
- `p_auxiliary.pyomo_constraints`: battery charge/discharge coupling, a
  hydrogen-store cycling constraint that ties `BatteryInterfaceOut` to
  `CompressedH2Store` (`4/8760*0.5*0.5`), HB ramp limits per snapshot.
- `p_location_class.renewable_data`: wind profile × 0.93 wake factor; block
  aggregation by **summing** consecutive hours (`aggregate`), separate from
  `p_auxiliary.aggregate_data` which **averages**.
- `main.generate_network(n_snapshots, ..., aggregation_count, time_step=0.5)`:
  snapshots = `int(n_snapshots / aggregation_count)`; store capital cost
  × `time_step × aggregation_count`; link marginal cost × `24·366/n_snapshots ×
  aggregation_count`; water cost added to electrolysis at 2 USD/kL ÷ 0.7 (AUD).
- `main.run_Alli_sites` (the only global-style loop in the frozen state) calls
  `generate_network(8760/4, ..., aggregation_count=4)`, which yields **547**
  snapshots, with unaggregated hourly profiles; the paper and Salmon (2022)
  describe a **4-hour step with 2,190 periods**. Which call Salmon used for the
  global run is not recoverable; `run_legacy_cells.py` runs both as named variants.
- `get_results_dict_for_multi_site`: `Objective = n.objective / (p_set/6.25 ×
  8760 × 1000)`, i.e. USD per kg on a 1 Mt/yr production basis; the shipping
  table's `LCOA` is this × 1000.

## 3. Cost inputs

`data/GeneralSteelData.xlsx` (first committed 31 August 2023, `5418ad1`; the
working-tree file is byte-identical, md5 `cee43cbf24814d35de7494a23cf1ed07`)
is the workbook `run_Alli_sites` reads. Its `Costs` sheet gives **annualised
USD2018** costs per MW (or MWh) for 2030-2050 on an 8 % discount rate, 20-year
life and 2 % fixed O&M (`Discount Rate Calculation` sheet: CRF 0.1019 + 0.02 =
0.1219). Its `Cost Trajectories` sheet is the Way et al. RCP4.5 / "slow
transition" series (identical to `x_SlowTransitionCosts.xlsx` and to the RCP 4.5
sheet of `x_Cost Forecasting.xlsx`), which the paper assigns to MOD-AMB. 2050
column: Wind 93,432.9; Solar 25,724.7; Electrolysis 33,720.7; BatteryInterfaceIn
and Battery 8,915.99; CompressedH2Store 2,524.05; HydrogenFuelCell 24,919.1; HB
565,000; Ammonia 5.76. Efficiencies 2050: electrolysis 0.8865, fuel cell 0.5400.
The intermediate commit `a3d6533` (17 October 2023) has the same Costs and
Efficiencies sheets.

The paper applies Ameli et al. "Reduced" 2050 WACC by region; the frozen code
loads a `WACCs.csv` (not in the repository) but never uses it, so the WACC step
lived in Salmon's runner. green-lory's `inputs/amelired_wacc_country_map_2050.csv`
gives 5.1 % for both Chile and Australia. `run_legacy_cells.py` therefore runs
each variant on the 8 % workbook basis and rescaled to 5.1 % (same life and O&M).

## 4. Weather

Nine NetCDF files (`Solar`, `SolarTracking`, `WindPowers` × suffix ``/`1`/`2` =
longitude bands <−60°, −60…60°, >60°), 2019 hourly, 180 × 120 cells, in
`lcoa-opt/data/` (13.6 GB). The three test-cell profiles were shown on 15
September to be bit-identical to the subsets used by the September green-lory
runs (`audit/model-evidence-v2/weather_cf_comparison.csv`), so those subsets
(`arc_received_fixed_v2/fixed_pv_3cells_v2/weather_used/`) are valid inputs
for the three-cell reruns. The frozen loader globbed the nine files **without
sorting** (Carlo added sorting on 16 November 2023 "to correct order"); the
band-to-file assignment on Salmon's machine is therefore not certain.

## 5. Carlo's later runs are a different lineage

- 25-26 August 2023: first global sweep in 30° longitude slabs
  (`results/2050_lcoa_global_*long.csv`, merged `results/2050_lcoa_global.csv`).
- 5 November 2023: `results/2050_lcoa_global_20231105_-180to180mp.csv`
  (`main_mp.run_global`, `time_step = 4`, 547 snapshots by construction). Its
  designs are on the 10 Mt/yr basis (Atacama 37,058 MW fixed PV, Objective 0.33084
  → 330.85 USD/t). These are the "November saved designs" used conditionally in
  the 15 September handover; they are **not** the paper's designs.
- 13 November 2023: `..._20231113_-180to180mp.csv`, 1 Mt/yr basis (328.60 / 339.15
  / 335.46 USD/t at the three cells).
- 29 October - 15 December 2023: `land_availability.ipynb` builds
  `data/20231029_land_max_capacity.csv` from MCD12C1 class fractions and the
  Table 2 factors (no protected-area or slope exclusion, area = east-west width
  squared, water availability 1) and then **copies `Max_capacity` from Nick's
  `c_NH3_cost_4.5.csv`**, removes IQR outliers (which drops the two Australian
  test cells) and regression-fills them from availability and area. Its output
  `results/2050_4.5_lcoa_global_max_capacity.csv` combines the 5 November LCOA
  with those capacities (1.355 / 1.024 / 1.051 Mt/yr). It is not an independent
  reconstruction and must not be used as a validation target.

## 6. Consequences for the replication

1. The legacy LCOA to reproduce is 212.92 / 233.69 / 226.39 USD/t at 1 Mt/yr with
   wind + fixed PV only, RCP4.5 costs, reduced WACC and (per the papers) a 4-hour
   step. `run_legacy_cells.py` tests the named call patterns; agreement within a
   few USD/t identifies the temporal accounting Salmon used.
2. The capacity column follows the paper method with **complete wind/solar
   overlap** (Salmon 2022, §2.2.1: "we assume that complete overlap is allowed
   between wind and solar farms"), not the exclusive shared-land rule used in the
   September green-lory capacity work. The reconstruction therefore uses
   `Q = 1 Mt × min(A_wind/(P_wind × 0.2 km²/MW), A_solar/(P_solar × d_lat))`, with
   the technology-specific suitable areas, the 2 % shipping share and the
   latitude-adjusted fixed-PV density, applied to the **replicated 1 Mt designs**
   from item 1 (not the November 10 Mt designs).
3. Remaining unknowns that cannot be closed from local evidence: Salmon's MODIS
   Collection 6 year, WDPA snapshot, slope source (ETOPO1 in Salmon 2022) and
   the exact PV-density function; the September pilots bound these.
