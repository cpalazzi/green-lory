# Attribution chain — legacy-lcoa → green-lory → green-porpoise (draft, 16 September 2026)

A number at every link, for the three focal cells and for Australia. Sources: archived
shipping input `c_NH3_cost_4.5.csv`; legacy replication `legacy_lcoa_20260915_v1`
(`LEGACY_REPLICATION_20260915.md`); green-lory surfaces `verschuur_reconcile_20260907_v1`
(replication `rep_way2050_flat_amelired_4h_tracking_nominal_h2`, central
`central_way2050_flat_amelired_1h_tracking_explicit_compressor_dea_tank`); land
`land_reconcile_20260914_v1/audit/global-unmasked-v2`; network runs in `RUNS.md`.
Network links D–F are filled in as the ARC runs return.

## 1. Weather

Identical: the nine 2019 NetCDF files (hash-pinned) feed both models; the three-cell profile
identity was verified exactly in the handover, and the legacy weather store carries the same
hashes on the Mac and on ARC.

## 2. Plant cost at 1 Mt/yr (USD2018/t)

| Cell | Archived | Legacy replicated | Green-lory replication (4 h) | Green-lory central (1 h) |
|---|---:|---:|---:|---:|
| Atacama (−23, −69) | 212.92 | 221.33 | 215.54 | 246.94 |
| NW Australia (−23, 117) | 233.69 | 242.64 | 235.51 | 266.85 |
| Central Australia (−21, 135) | 226.39 | 234.04 | 220.24 | 231.20 |
| Global median ratio to archived (15,377 / 14,542 sites) | 1 | 1.038 | 0.99 | 1.097 |

The legacy residual is a uniform +3.8 % (finance convention, not tuned) plus a wind-related
excess; the green-lory replication was calibrated to the archived costs; the central model is
+10 % (hourly resolution, explicit compressor, tank storage).

## 3. Design (MW per 1 Mt/yr; PV = fixed + tracking)

| Cell | Legacy replicated | Green-lory replication | Green-lory central |
|---|---|---|---|
| Atacama | 3,380 PV, 0 wind | 3,500 PV, 0 wind | 3,507 PV, 0 wind |
| NW Australia | 3,995 PV, 0 wind | 4,176 PV, 0 wind | 4,180 PV, 2 wind |
| Central Australia | 3,039 PV, 348 wind | 2,147 PV, 923 wind | 1,777 PV, 1,157 wind |

Same weather, similar PV-only designs; the models diverge where wind competes. Green-lory
(both variants) builds two to three times the legacy plant's wind at central Australia; over
the archived sites 6,868 legacy designs use more than 5 % wind, and the green-lory designs use
wind at more cells and in larger shares (median wind share 0.03, upper quartile 0.35 for the
replication surface).

## 4. Land → capacity

Legacy rule (recovered from the archived table, `legacy_capacity.py`): 2 % Table-2 suitable
area per technology, complete overlap, 140 MW/km² PV (no latitude term, tracking not
penalised), 7.3 MW/km² wind, Q = min(A_solar·140/P_pv, A_wind·7.3/P_wind).
Green-lory September method: exclusive shared land, 83 MW/km² fixed PV with latitude
adjustment, tracking at twice the fixed footprint (37–42 MW/km²), 5 MW/km² wind, slope and
protected-area exclusions.

| Cell (Mt/yr) | Archived | Legacy rule on legacy design | Legacy rule on green-lory rep design | Legacy rule on central design | September method, rep | September method, central |
|---|---:|---:|---:|---:|---:|---:|
| Atacama | 9.77 | 9.43 | 9.10 | 9.09 | 2.32 | 2.31 |
| NW Australia | 4.02 | 3.97 | 3.80 | 3.79 | 0.96 | 0.95 |
| Central Australia | 4.32 | 2.42 (wind-limited) | 0.91 (wind-limited) | 0.73 (wind-limited) | 0.46 | 0.40 |

Reading: the September land constants explain a factor of about 4 at PV-only cells (9.77 →
2.32); the plant's wind share explains the rest at mixed cells (central Australia 4.32 → 2.42
→ 0.91 → 0.73 under the same rule). 28 archived cells (Algeria 19, central Australia 6,
Niger 2, Chad 1) carry 20–356 Mt/yr at 7–40 × the rule and are not reproducible by any
tested method (`LEGACY_REPLICATION_20260915.md` §8).

## 5. Eligibility at 1 Mt/yr and Australian supply

| Supplier set | Eligible cells, global | Eligible cells, Australia | Australian eligible capacity, 10 Mt/yr cap (Mt/yr) |
|---|---:|---:|---:|
| Archived table | 4,548 | 553 | 1,627 |
| Legacy replicated (legacy rule) | 4,900 | 590 | 1,801 |
| Green-lory replication + legacy rule (`20260915-v1/rep_legacy_rule`) | 4,379 | 449 | 1,123 |
| Green-lory central + legacy rule (`20260915-v1/central_legacy_rule`) | 3,972 | 226 | 430 |
| Green-lory replication, September method | 1,233 | 20 | 23 |
| Green-lory central, September method | 819 | 1 | 1 |

(Green-lory rows count the 14,542 archived sites present on its grid; totals at the archived
universe. The Australian archived capacity includes six anomalous cells; without them 1,632
archived vs 1,826 replicated Mt/yr before the cap.)

## 6. Network (deposited equations, RCP4.5/70 %, unprefixed demand, regenerated routes × 1.1, 1 h solver limit)

| Run | Supplier contract | USD/t delivered | Gap | Australia (Mt/yr, share) | Active suppliers |
|---|---|---:|---:|---:|---:|
| A (8805879) | archived table | 260.65 | 0.89 % | 255.3 (43 %) | 317 |
| B (8805880) | green-lory replication, September method | 274.91 | 0.28 % | 25.0 (4 %) | 394 |
| C (8805881) | green-lory central, September method | 310.21 | 0.25 % | 1.0 (0.2 %) | 388 |
| D (8820849) | green-lory replication + legacy rule | pending fetch | | | |
| E | legacy replicated (frozen plant + legacy rule) | queued | | | |
| F (optional) | green-lory central + legacy rule | not submitted | | | |

The archival replay (archived table, archived routes, archival equations) is 254.05 USD/t with
258.1 Mt/yr Australian; A reproduces it with the deposited equations and regenerated routes.

## 7. What the chain says so far

1. Weather and PV-only plant designs agree; the cost surfaces agree to a uniform few per cent.
2. The capacity gap (median new/old 0.217) is almost entirely the land-conversion constants and
   the sharing rule: the recovered legacy rule applied to green-lory's own designs restores 96 %
   of the archived eligible set globally and 81 % of it in Australia.
3. The remainder is the plant's wind share at mixed cells, which the hourly central model
   amplifies (Australia 226 eligible cells under the legacy rule vs 553 archived), because
   wind is the land-hungry technology under every rule.
4. Which of the two land treatments is *realistic* is the decision: the legacy constants
   (7 km²/GW for tracking PV, 137 km²/GW wind, full overlap, no exclusions, plus 28 cells with
   the 2 % share apparently unapplied) are not the paper's stated values and are generous by
   current siting practice; the September constants are conservative and treat tracking
   harshly. That judgement, and the network runs D–F, complete `DECISION.md`.
