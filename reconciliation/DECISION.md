# Legacy-lcoa versus green-lory: results and decision

6 October 2026. Replaces `archive/2026-09/DECISION_20260916.md` and its two post-scripta; the
September reports stay in `archive/2026-09/` unedited. Run ids are defined in `../runs.yaml`
and their executions recorded in `run_store.csv`; terms are in `../GLOSSARY.md`; open modelling
choices are in `../MODEL_CHOICES.md`. Plant LCOA is in EUR2020/t (legacy values converted from
USD2018 with 0.9024); network delivered cost stays in USD2018/t, the network model's currency.

## 1. Question and answer

When the green-porpoise supplier table of Verschuur et al. (2024) (`c_NH3_cost_4.5.csv`,
Salmon 2023) was replaced by a green-lory surface in September, Australia fell from about
255 Mt/yr of supply to 25 Mt/yr or less. The reconciliation had to explain why, reproduce the
legacy model, and decide which model to use.

- **The legacy model is reproduced end to end.** The frozen lcoa-opt code (94de8ce, 3 May 2023)
  with the recovered capacity rule regenerates the archived table (LCOA +4 %, a finance-convention
  offset) and the published network geography (network E against the archived-table control A2:
  Australia 250 against 255 Mt/yr; 268 against 261 USD/t).
- **The divergence came from the land-to-capacity step, not from plant cost.** Green-lory
  configured like the legacy plant agrees with it on cost (within 1 % at the median), and with the
  legacy capacity rule it gives the archived network back (network D: Australia 239 Mt/yr,
  261 USD/t). The archived capacities rest on 140 MW/km² of PV at every latitude, 7.3 MW/km² of
  wind on the same land, and no exclusions; at the focal cells they exceed what the land can
  generate in a year by about a factor of two.
- **Decision.** Green-lory is the model of record for costs, designs and capacity, with the
  choices in `../MODEL_CHOICES.md`. The legacy package (`../legacy_lcoa/`) is kept only as the
  replication baseline of the published network.

## 2. What changed in green-lory

| Area | Before the reconciliation | Now |
|---|---|---|
| Land geometry | MODIS bands labelled by their north edge, exclusions by the south-west corner: each cell carried the suitability of the cell one degree south | centred cells with exact spherical weights (`cell_anchor=center`); tables that are not centred builds are refused |
| Land sharing | wind and PV exclusive (every wind km² lost to PV); tracking at twice the fixed footprint | co-located: only the wind direct-impact share (3 %) excludes PV; tracking at 1.458 x fixed (Bolinger and Bolinger 2022) |
| Land share | 2 % of suitable land | 20 % as base (LUT-consistent), 2 % as sensitivity |
| PV | fixed and tracking offered | fixed tilt only; tracking as a named sensitivity |
| Plant | 4-hour step with legacy store and ramp scaling; nominal compressor with compression bundled into the H2 store cost | hourly, snapshot-weighted accounting, per-hour ramps; explicit compressor and DEA tank (compression counted once) |
| Costs and finance | Way 2050, one finance basis | DEA 2050 (EUR2020), per-technology annuity; per-cell WACC override (uniform 5 % base, Ameli by country as sensitivity) |
| Site costs | water outside the headline | water 2 EUR2020/m³ (1.5 m³/t) in the headline; no land rent |
| Capacity | separate per-technology caps | scaled reference design against three budgets (wind, PV, their shared union); land-enforced finite-site solves available (`in_solve`, `solved_quantity`) |
| Runs | wrapper presets, mutable outputs | runs as data (`runs.yaml`), immutable releases and campaigns, manifests with hashes, QA gates, solver retry chain |

## 3. The focal cells

Atacama (Chile, −23, −69; 2,600 m), the inland Pilbara in northwest Australia (−23, 117;
230 km from the coast) and central Australia (−21, 135; 580 km from the coast). None of the
three is a coastal cell; section 5 compares coast and inland across Australia. Built by `focal_cells.py`; output and source hashes in
`results/campaigns/reconciliation_final_20260916_v1/focal_cells_v3/`.

**Land (suitable km² at a 2 % share).** The centred build changed little at these cells; it
now matches the legacy land step except where a protected area lies inside the cell.

| | Atacama | NW Australia | Central Australia |
|---|---:|---:|---:|
| legacy land step (no exclusions) | 227.6 | 113.2 | 115.3 |
| September table (mislabelled) | 220.3 | 108.4 | 109.7 |
| centred table | 226.8 | 113.1 | 102.5 (11 % protected) |

**Plant cost and capacity.** LCOA in EUR2020/t; capacity in Mt/yr at the stated share.

| Run | Atacama | NW Australia | Central Australia |
|---|---|---|---|
| archived table | 192.1; 9.77 | 210.9; 4.02 | 204.3; 4.32 |
| legacy replication (recovered rule, 2 %) | 199.7; 9.43 | 219.0; 3.97 | 211.2; 2.42 |
| green-lory reconciling, legacy rule (2 %) | 194.5; 9.10 | 212.5; 3.79 | 198.7; 0.91 |
| legacy key run (stated land method, centred 2 %, water) | 183.6; 5.17 | 199.9; 2.18 | 193.7; 1.51 |
| green-lory improved, uniform 5 %, 20 % | 328.4; 34.27 | 351.3; 11.65 | 288.9; 3.82 |
| green-lory improved, uniform 5 %, 2 % | 328.4; 3.43 | 351.3; 1.16 | 288.9; 0.38 |
| finite-site curve, Way 2050 costs, fixed PV, 2 %: unconstrained cost; largest feasible quantity at its cost | 234.7; 4.0 at 238.9 | 251.8; 1.75 at 254.1 | 213.2; 2.0 at 244.8 |

Reading:

1. **Cost ranking reverses.** Legacy and reconciling runs rank Atacama < central Australia < NW
   Australia; the improved model ranks central Australia (289) < Atacama (328) < NW Australia
   (351). The hourly plant does this: with the same Way 2050 costs and tracking still offered,
   moving from the 4-hour legacy-style plant to the hourly plant with explicit compressor and
   DEA tank raised the PV-only plants by 13-15 % (Atacama, the inland Pilbara) but the wind–PV hybrid
   of central Australia by 5 % (September attribution, `archive/2026-09/ATTRIBUTION_20260916.md`
   section 2). Fixed-tilt-only PV adds another 2-5 % (finite-site controls, tracking-only:
   222.8, 240.8, 208.6).
2. **DEA 2050 costs, with water in the headline, add 36-40 %** on top (fixed-PV finite-site
   controls to the improved run: +40 %, +39 %, +36 %). This is a choice of datasheet, not a
   correction: Way 2050 and DEA 2050 are both defensible 2050 projections.
3. **Capacity is set by the design and the rule, not by land area.** At 2 % the improved scaled
   design gives 3.43, 1.16 and 0.38 Mt/yr. The two Australian designs carry wind (0.49 and
   1.34 GW per Mt/yr), and wind at 5 MW/km² exhausts the wind budget first; Atacama's PV-only
   design fills the shared budget. Re-optimising the plant for the land (finite-site curves)
   gives 4.0, 1.75 and 2.0 Mt/yr, so the scaled design is a conservative lower bound, by about a
   factor 5 in central Australia (approximate: the curves use Way 2050 costs).

## 4. Networks

Deposited equations, MOD-AMB RCP4.5 70 % demand (594.5 Mt/yr), 1 Mt/yr cutoff, one-hour solver
limit (gaps 0.48-0.89 %, accepted at the deposit's 1.5 %). Australia split by longitude: west
< 129° E, centre 129-141° E, east > 141° E. Built by `replay/compare.py` of green-porpoise
(`results/campaigns/reconciliation_final_20260916_v1/networks_v3/`).

| Network (September name) | USD/t | Australia: total (west / centre / east), Mt/yr | Chile | Other leading producers |
|---|---:|---|---:|---|
| archived table, control (A2) | 260.65 | 255.3 (179.7 / 61.5 / 14.1) | 39.0 | Mauritania 47.5, Oman 32.0 |
| legacy replication (E) | 268.16 | 250.4 (176.8 / 63.4 / 10.2) | 33.6 | Mauritania 45.5, China 27.0 |
| green-lory reconciling, legacy rule (D) | 261.23 | 238.5 (176.6 / 52.4 / 9.5) | 35.2 | Yemen 38.8, USA 31.9 |
| legacy key run, 2 % | 251.59 | 202.8 (125.6 / 62.4 / 14.7) | 36.0 | Yemen 88.4, Mauritania 39.3 |
| improved, uniform 5 %, 20 % | 370.96 | 221.7 (134.6 / 78.8 / 8.3) | 0.1 | Morocco 105.6, Sudan 80.1, Mauritania 58.8 |
| improved, Ameli, 20 % | 372.58 | 219.5 (134.2 / 76.7 / 8.7) | 0.1 | Morocco 98.3, Sudan 79.1, Mauritania 56.6 |
| improved, uniform 5 %, 2 % | 416.90 | 17.1 (4.8 / 3.1 / 9.2) | 39.2 | Yemen 110.6, Saudi Arabia 72.6, Mauritania 57.7 |

Reading: at a 20 % share the improved model keeps Australia first and moves its production
inland (centre 79 against 62 Mt/yr, west 135 against 180), brings in Morocco and Sudan, and
drops Chile, whose PV-only cells lost their cost advantage (item 1 above). At 2 % the
wind-carrying designs of Australia, Morocco and Sudan become land-limited, and Chile and the
Arabian Peninsula supply instead. The land share decides Australia against the Middle East;
the hourly plant decides Chile against central Australia and North Africa. Finance (uniform
5 % against Ameli) moves delivered cost by 0.4 %.

## 5. Coast against inland Australia

Australian cells banded by the distance from the cell centre to the coast. Built by
`coast_inland.py` (`results/campaigns/reconciliation_final_20260916_v1/coast_inland_v1/`).

**Plant cost never favoured the coast.** In the archived table inland cells (> 300 km) are
cheaper than coastal ones (< 100 km): median 214 against 238 EUR2020/t over all cells, and level
(214 against 212) over the cells eligible for the network. **The coast won on transport.** The
network pipes ammonia from each supplier to a port at 0.026 USD/t/km (deposited model, 7 %), so
at level plant costs a cell 300-600 km inland pays roughly 8-17 USD/t more delivered. In the
archived-table network 69 % of Australian production lay within 300 km of the coast.

**The hourly plant changes the balance.** On the cells eligible in the archived table, the
cheapest decile inland was level with the cheapest coastal decile until the hourly plant, and is
18 EUR2020/t cheaper in the improved model, more than the pipeline to the coast:

| Step | Cheapest decile, inland minus coast (EUR2020/t) |
|---|---:|
| 1 archived table | −0.2 |
| 2 legacy model replicated | 0.0 |
| 3 green-lory, legacy-style plant (4 h, Way, tracking) | −1.5 |
| 4 hourly plant (1 h, Way, tracking, explicit compressor, DEA tank) | −10.7 |
| 5 improved (1 h, DEA 2050, fixed PV, water) | −17.5 |

| Australian production, Mt/yr | coast < 100 km | 100-300 km | inland > 300 km |
|---|---:|---:|---:|
| archived table (A2) | 57.0 | 120.4 | 77.9 |
| improved, 20 % | 12.8 | 45.4 | 163.5 |

The mechanism: a solar-only plant makes hydrogen in a daily pulse that must be compressed and
stored to keep the Haber-Bosch loop above its minimum load through the night. The legacy-style
plant priced this lightly: four-hour block means flatten the solar peak, the compressor carried a
nominal cost, and compression was folded into the store cost. At the focal cells, hourly
resolution and the explicit compressor added 32 USD2018/t to the solar-only plants (Atacama, the
inland Pilbara) and 9 to central Australia's wind–solar plant (`archive/2026-09/REPORT_20260914.md`
section 3.4). Inland Australian designs carry more wind (median 30 % of renewable capacity on
eligible inland cells, 10 % on coastal ones), so they pay less of it.

**Site costs absent from the base case could restore the coast.** Water and build costs are flat
in the improved runs, so nothing but the ammonia pipeline penalises inland sites. With the May
spatial inputs (`inputs/spatial_cost_inputs.csv`, not yet reviewed), water piped inland (median
17.2 against 5.2 USD2020/m³ at the coast) adds 19.6 EUR2020/t inland against 3.8 at the coast,
and the build/remoteness multipliers add 6-8 EUR2020/t everywhere (first order, design held
fixed). The cheapest-decile inland advantage then falls from 19 to 4 EUR2020/t, below the
pipeline cost, and the coast would lead on delivered cost. This is a site-cost result: it
belongs to the spatial runs once their water and remoteness inputs are reviewed, and must not be
obtained by changing the plant model.

## 6. Decisions

Settled on 16 and 23 September and 5 October (details and dates in `../MODEL_CHOICES.md`):
green-lory as the model of record; DEA 2050 costs; uniform 5 % WACC base with Ameli as
sensitivity; hourly; fixed-tilt PV; co-located land at a 20 % share with 2 % as sensitivity;
capacity from the scaled reference design, stated as a conservative lower bound with the
finite-site curves quoting its bias (this replaces the 16 September text, which named
finite-site solves); networks accepted at the deposit's 1.5 % gap; the legacy package as the
replication baseline only. Still open: the fixed-PV density value, the weather data source,
and which further surfaces the papers need.

## 7. Evidence and reproduction

| Item | Location |
|---|---|
| Focal-cell table | `results/campaigns/reconciliation_final_20260916_v1/focal_cells_v3/` (`python reconciliation/focal_cells.py --output <new dir>`) |
| Coast against inland Australia | `results/campaigns/reconciliation_final_20260916_v1/coast_inland_v1/` (`python reconciliation/coast_inland.py --output <new dir>`) |
| Network comparison | `results/campaigns/reconciliation_final_20260916_v1/networks_v3/` (in green-porpoise: `python -m replay.compare --run <label>=<dir> ... --reference <label> --output <new dir>`) |
| Every run's output | `location` column of `run_store.csv` |
| Legacy package and its replication reports | `../legacy_lcoa/README.md`, `../legacy_lcoa/reports/` |
| September evidence (reports, run log, land experiments) | `archive/2026-09/` |
