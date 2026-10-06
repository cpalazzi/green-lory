# Legacy-lcoa replication — three-cell results, 15 September 2026

Status: **LCOA replicated to within 4 % at all three cells with a single,
documented configuration; capacity formula under test on a 563-cell sample.**
All numbers below are from `run_legacy_cells.py` runs on Carlo's Mac
(`legacy-lcoa-env`: PyPSA 0.25.1, Pyomo 6.5.0, Gurobi 11.0.1), full nine-file
2019 weather stack, campaign directory
`results/campaigns/legacy_lcoa_20260915_v1/` (each run has `manifest.json`,
per-cell `summary.json`, profiles used, dispatch series).

## 1. Environment note

PyPSA 0.25's Pyomo path builds its objective by assigning
`LinearExpression.linear_vars/linear_coefs`; Pyomo ≥ 6.6 silently ignores that
and the model solves with a **constant-zero objective** (every capacity at its
bound). lcoa-opt's `environment.yaml` pins Pyomo 6.7.1, so that environment
cannot have produced meaningful legacy results after March 2024. Pyomo 6.5.0
restores the objective (`debug_two.py` check: 8760-h objective 3.1e8 vs 0).

## 2. What each configuration gives (USD/t at 1 Mt/yr)

Archived shipping input: Atacama 212.92, NW Australia 233.69, central Australia 226.39.

| Code state | CAPEX source | Tracking | Step | WACC basis | Atacama | NW Aus | C Aus |
|---|---|---|---|---|---:|---:|---:|
| cd56c11 (6 Jun 2023) | workbook Costs sheet | off (as in CSV) | 547 as written | 8 % workbook | **330.85** | **342.88** | **317.38** |
| cd56c11 | workbook | off | 4 h mean, 2190 | 8 % | 337.97 | 352.76 | 311.14 |
| cd56c11 | workbook | off | hourly 8760 | 8 % | 345.70 | 363.88 | 315.41 |
| cd56c11 | workbook | off | 4 h | Ameli 5.1 % | 281.16 | 293.32 | 258.69 |
| 94de8ce (3 May 2023) | workbook | off | hourly | Ameli 5.1 % | 263.86 | 284.98 | 252.17 |
| 94de8ce | workbook | on, 1.0587 × fixed | 4 h | Ameli 5.1 % | 234.71 | 257.34 | 243.01 |
| 94de8ce | x_Cost RCP4.5 2050 | on (own row) | hourly | Ameli 5.1 % | 230.27 | 251.06 | 238.97 |
| **94de8ce** | **x_Cost RCP4.5 2050** | **on** | **4 h** | **Ameli 5.1 %** | **221.33** | **242.64** | **234.04** |
| cd56c11 | x_Cost RCP4.5 2050 | on | 4 h | Ameli 5.1 % | 240.72 | 260.63 | 243.61 |

Ratios of the bold row to the archived values: **1.039 / 1.038 / 1.034** — a
uniform 3.4–3.9 % offset, with the archived cost ordering (Atacama < central
Australia < northwest Australia) reproduced. Every earlier configuration is 20–60 %
high and gets the ordering wrong (central Australia cheapest) because it lets the
plant use the strong central-Australian wind.

The first row reproduces Carlo's 5 November 2023 global run **exactly**
(330.85 / 342.88 / 317.38; designs 3,706 / 4,031 / 2,820 MW PV and 0 / 186 / 785 MW
wind, i.e. the November 10 Mt/yr designs divided by ten). That run therefore was
the surviving `run_Alli_sites`/`run_global` call pattern: 547 snapshots (the
first 23 days of January), workbook costs, 8 % basis. It is not the paper's run:
against the archived table its LCOA ratio is 1.6 at low latitudes and 4.8 at
60–70°, exactly the signature of a January-only optimisation.

## 3. The configuration that replicates the paper

1. **Code state 94de8ce (3 May 2023)**, the last commit before the supplier
   tables were committed to green-porpoise (2 June 2023). Compared with the
   6 June state it has no electrolysis water cost, no hydrogen-store cycling
   constraint, no compressor marginal cost, no 148,000 USD/MW/yr battery-discharge
   cost, HB coefficients 6.27 / −6.976, and `generate_network(8760, …)` with
   plain `aggregation_count` store scaling. Those June additions raise LCOA by
   about 9 % (240.72 vs 221.33 at Atacama).
2. **Overnight CAPEX from `x_Cost Forecasting.xlsx`, sheet RCP 4.5, 2050 column**
   (USD/W: wind 0.906, fixed PV 0.240, single-axis PV 0.254, fuel cell 0.221,
   electrolyser 0.203, battery interface 0.056, battery 0.077/Wh), annualised
   with the workbook's own factor. That factor is **0.1143**, not 0.1219: the
   workbook's "Discount Rate Calculation" sheet discounts CAPEX plus twenty O&M
   payments with an annuity-due (Net Time 10.6036 at 8 %) and divides by the
   same annuity, i.e. 1/10.6036 + 0.02. The workbook's Costs sheet is exactly
   its own CAPEX table × 0.1143; its electrolyser (0.295 USD/W) and wind
   (0.817 USD/W) differ from the x_Cost sheet, and the x_Cost electrolyser
   (0.203) is what brings the plant cost down. HB, ammonia store and hydrogen
   store keep the workbook values (565,000; 5.76; 2,524.05).
3. **Single-axis tracking PV enabled** with its own cost row. The frozen
   `generators.csv` has it disabled (capital cost 1e9, not extendable) and the
   workbook has no row for it, but the paper lists it as a technology and the
   x_Cost sheet carries its trajectory. With it enabled every design at these
   cells is tracking-dominated (Atacama 3,380 MW, NW Australia 3,995 MW, central
   Australia 3,039 MW tracking + 348 MW wind at 4 h).
4. **Four-hour step, 2,190 periods** (block means; Salmon & Bañares-Alcántara
   2022: "a time step of four hours was used"); hourly gives 4 % more.
5. **Ameli reduced WACC 5.1 %** for Chile and Australia, applied as a rescaling
   of the annualised costs on the same 20-year, 2 % O&M basis (factor 0.8486).

Remaining residual: +3.4–3.9 % uniformly. A single scalar of that size is
consistent with a slightly lower WACC (~4.6 %), a longer life, or a small
difference in the annualisation convention; none of these was tuned and the
residual is reported as is. Electricity capex fractions are 0.376 / 0.406 /
0.451 against archived `Electricity_Cost_Frac` 0.304 / 0.329 / 0.327; that
column's definition in Salmon's pipeline is unknown.

## 4. Consequences for the capacity reconstruction

With tracking-dominated designs of 3.0–4.0 GW per Mt/yr and complete wind/solar
overlap (Salmon 2022 §2.2.1), the archived capacities imply an effective PV
density of **146 MW/km² at Atacama and 142 MW/km² at northwest Australia**
(both 23° S) on the corrected 2 % land, and 128 MW/km² at central Australia
(21° S) if its 348 MW of wind is not land-limiting. The paper's "~9 km²/GW"
(111 MW/km²) with a downward latitude adjustment cannot produce these; the
implied values point to a density function nearer 6–7 km²/GW at 20–25° — for
example First Solar module area (≈6 km²/GW) with little row spacing at low
latitude. The 563-cell sample run (`sample563_may2023_xcost45_tracking_4h_v1`)
provides designs across latitudes so that the implied density can be plotted
against latitude and the wind-land treatment tested, instead of being fitted.

The energy-ceiling formula tested earlier (annual energy on 2 % land at
111 MW/km² divided by 6.25 MWh/t) matches these three barren, PV-dominated
cells within 5 % but fails globally (median ratio 1.38, wide spread), so it is
recorded as a coincidence for this land class, not as the method.

## 5. Not yet done

- Exact-match search for the 3.5 % residual is deliberately not attempted.
- ARC cross-check with the ARC Gurobi licence (`arc/jobs/07_build_legacy_env.sh`
  builds the environment; pyomo must be 6.5.0).
- Capacity formula selection and the global legacy surface follow from the
  sample run; the network attribution then uses the replicated designs.

## 6. 563-cell sample: the LCOA replication holds globally and the capacity rule is recovered

Run `sample563_may2023_xcost45_tracking_4h_v1` (the bold configuration of §2 at
563 archived supplier cells, 80 per 10° latitude band plus the three focal
cells; all solved; audit `audit/sample563-analysis-v1/`, `summary_v2.json`,
`sample563_analysis_v2.png`).

**LCOA.** Rerun / archived over 565 matched cells: median **1.040**, IQR
1.032–1.057. For the 371 PV-dominated cells (wind < 5 % of design energy) the
ratio is 1.039 with IQR 1.034–1.043 and no latitude trend below 60°; it widens
where wind is used (mixed cells median 1.050, IQR 1.022–1.113) and north of 60°
(1.15). The paper's cost surface is therefore reproduced by this configuration up
to one uniform factor of about 1.04 (a finance convention, not tuned) plus a
wind-related residual of a few per cent.

**Capacity.** With the replicated designs (MW per Mt/yr) and Salmon (2022)'s
complete wind/solar overlap, the archived capacities imply

- a PV density of **140 MW/km² (7.1 km²/GW) with no latitude dependence**:
  median 140.3 over the PV-dominated cells, 140.8 with IQR 132–143 for the 76
  cells whose land is more than half suitable (barren/shrub), and band medians
  143 / 140 / 143 / 140 for 0–10°, 10–20°, 20–30°, 30–40°;
- a wind density of **7.3 MW/km² (137 km²/GW)**: 7.26–7.42 for seven of the
  eight wind-dominated cells (Patagonia, northern Russia).

Neither is the paper's stated value (~9 km²/GW with a latitude adjustment;
200 km²/GW). The rule

`Q = 1 Mt/yr × min( A_solar(2 %) × 140 / P_pv , A_wind(2 %) × 7.3 / P_wind )`

reproduces the archived capacity with median 0.998 for PV-dominated cells
(0.994, IQR 0.977–1.057, 75 % within ±10 % for the high-suitability subset;
the spread at low-suitability cells is land-data vintage, not the rule) and
0.996 for wind-dominated cells. For the 186 **mixed** cells (5–90 % wind
energy) it under-predicts (median 0.80, IQR 0.61–0.99): the archived capacity
sits between the wind-limited and the PV-limited value, which is what a design
with less wind than ours would give. The LCOA ratio also scatters most there,
so the remaining disagreement is in the wind part of the design, not in the
land rule. Central Australia is such a cell (348 MW wind + 3,039 MW tracking in
the replication): wind-limited 2.42, PV-limited 5.31, archived 4.32 Mt/yr.
Atacama (9.43 vs 9.77) and northwest Australia (3.97 vs 4.02) are PV-only and
fall on the rule.

**What this means for the reconciliation.** Relative to the September green-lory
capacity method (exclusive shared land, 83 MW/km² fixed PV with a latitude
adjustment, tracking counted at twice the fixed footprint → 37–42 MW/km²,
wind 5 MW/km², slope/protected exclusions), the legacy input used effectively
**3.4–3.8 × denser PV, 1.5 × denser wind, full overlap and no exclusions**.
That is the whole of the capacity gap (median new/old ratio 0.217): it is a
difference of land-conversion constants and sharing rule, not of plant physics.
The energy-ceiling formula of §4 is withdrawn as an explanation; it coincided
with the rule only for barren PV-only cells.

## 7. Remaining residuals, stated

- Uniform +4 % LCOA offset; +1–2 % more at wind-using cells; +15 % north of 60°.
- Mixed-cell capacities under-predicted by ~20 % at the median (design-dependent).
- Land is MODIS 2022 / no exclusions (the unmasked audit tables); Salmon's
  land vintage and any exclusions he applied are not recoverable, and explain
  the wider spread at low-suitability cells.
- The archived `Electricity_Cost_Frac` column is not reproduced (0.30–0.33
  archived vs 0.36–0.45 here); its definition is unknown.

## 8. Global replication surface (16 September 2026): all 15,377 archived sites

Run `global_archived_mac_v1` (Mac, `legacy-lcoa-env`, Gurobi 11.0.1, four workers, weather
store `weather_store_archived15377_mac_v1` extracted from the nine files — values unchanged,
proven bit-identical on the three cells; the bold configuration of §2 with the Ameli country
WACC map from `cells_archived_all_v1.csv`). 15,377 cells, no failures. Supplier table and
green-porpoise contract: `audit/global-supplier-table-mac-v1/` (`gpo_export/contract.json`,
15,008 positive-capacity rows including the archived table's 67 border-cell duplicates);
figure `global_replication_v1.png`. The same surface is being produced on ARC
(`global_archived_v2`, Gurobi 11.0.3) as a cross-check.

**LCOA.** Rerun / archived over the 15,377 unique sites: median **1.038**, IQR 1.029–1.064.
PV-dominated cells (8,509 with less than 5 % wind in the design): 1.038, IQR 1.033–1.043;
latitude-band medians 1.031–1.041 from 0° to 60°, 1.08 beyond 60°. Mixed designs (6,805):
1.043, IQR 1.011–1.117; wind-dominated (63): 1.038. The residual is therefore one uniform
factor of 1.04 plus a wind-related excess that grows with the wind share at 30–60°: where the
plant uses wind, Salmon's run was cheaper than the replication, i.e. his wind was cheaper or
more productive than the frozen 0.93-wake profile with x_Cost wind CAPEX. The 563-cell sample
run (uniform 5.1 %) agrees exactly with the global run at every one of its 463 cells whose
country WACC is 5.1 %; the 100 cells that differ are those in Canada, the USA, the UK, western
Europe and Japan, whose Ameli WACC is lower.

**Capacity.** The rule of §6 against the archived `Max_capacity` (positive cells): PV-limited
cells median **1.000** (IQR 0.935–1.178); wind-limited cells 0.85 with a long low tail (mixed
designs 0.858, IQR 0.664–1.033) — the same design-dependent shortfall as in the sample: the
archived capacity of a mixed cell lies between the wind-limited and the PV-limited value.

**Anomalous archived cells.** 28 archived sites carry 20–356 Mt/yr — Algeria 19, Australia 6
(−21/−22° S, 130–132° E and −25° S, 127–128° E), Niger 2, Chad 1 — at 7–40 × the rule; for the
largest the ratio is 40 ≈ 0.8 / 0.02, as if the whole barren fraction of the cell had been
counted without the 2 % share. They hold 5,543 of the archived 25,236 Mt/yr. Excluding them the
replicated total is 1.08 × the archived (21,298 vs 19,693 Mt/yr). In the network each supplier
is capped at 10 Mt/yr anyway (preserved archival behaviour), so their influence there is bounded:
with the cap, eligible capacity is archived 17,491 vs replicated 19,309 Mt/yr, Australia
1,627 vs 1,801.

**Eligibility at 1 Mt/yr.** Archived 4,548 cells, replicated 4,900, common 4,367 (96 % of the
archived eligible set reproduced); 181 archived-only, 533 replicated-only (mostly mid-latitude
cells whose replicated design uses some wind and clears the PV limit). Cheapest-4,000 overlap
3,656. **Australia: 553 vs 590 eligible cells, 1,827 vs 1,851 Mt/yr** (1,632 vs 1,826 without
the six anomalies); Australian cells among the cheapest 4,000: 540 vs 534. The archived
Australian supply that produced the divergent green-porpoise distribution is reproduced by the
replicated pipeline.

**Three focal cells** (unchanged from §2/§6): 221.33 / 242.64 / 234.04 USD/t; 9.43 / 3.97 /
2.42 Mt/yr against 9.77 / 4.02 / 4.32 archived (central Australia is a mixed cell; its
replicated 348 MW of wind makes it wind-limited).

**Status.** Legacy-lcoa is replicated end to end at the archived sites: plant (frozen code,
pinned CAPEX and WACC, 4-hour step, tracking on), weather (hash-pinned 2019 stack), and the
land-to-capacity step (recovered rule, 2 % Table-2 areas). Residuals, all reported and none
tuned: +3.8 % uniform LCOA; +1–8 % more at wind-using cells and +8 % north of 60°; wind-limited
capacities 15 % low at the median; 28 anomalous archived capacities not reproducible;
`Electricity_Cost_Frac` definition unknown.
