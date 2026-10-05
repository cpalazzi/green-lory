# Land-constrained supply curves — 15 September 2026

## Final status

The [certified quantity-grid report](SUPPLY_CERTIFIED_20260915.md) supersedes
the interim status below. Job 8811997 timed out at 11:06:31 BST. All 21 planned
quantities are nevertheless classified using 16 validated feasible optima,
two solver-infeasible checkpoints and three separate analytical certificates.
The controls also pass. This is a completed discrete quantity analysis, not
a successful full Slurm run or an exact maximum-capacity calculation.

## Interim findings retained for provenance

The three **1 Mt/year** checkpoints have passed local result-hash, optimal
termination, production, cost-closure, grid/slack and independently reconstructed
land-use checks. The full 21-point curve is **not yet accepted**: numerical
failures affected higher-quantity endpoints in earlier attempts. Job **8811997**
is running with a dual-simplex numerical fallback. See [the run log](RUNS.md)
for preserved attempts and solver changes.

Update, **10:03 BST**: v4 remains running. A fetched snapshot now has **19
locally validated checkpoints**, comprising all three controls and 16
constrained quantities. The highest checked feasible outputs are Atacama
4.5 Mt/year at 238.22 EUR/t, northwest Australia 2 Mt/year at 264.42 EUR/t,
and central Australia 2 Mt/year at 259.98 EUR/t. Five extreme endpoints
remain unresolved; these are not maximum-capacity estimates. The checkpoint
audit is `audit/supply-checkpoints-v4-20260915T1001BST` in the land campaign.

The important result is that **scaling the cheapest reference plant is not
equivalent to optimizing supply under a land constraint**. This distinction
can change supplier eligibility even without changing a single land input.

| Site | Cheapest central design scaled to common land (Mt/year) | Land-constrained output checked (Mt/year) | Constrained LCOA (EUR 2020/t) | Shared land used |
|---|---:|---:|---:|---:|
| Atacama | 2.699 | 1.000 | 222.83 | 37.1% |
| Northwest Australia | 1.125 | 1.000 | 240.81 | 88.8% |
| Central Australia | 0.376 | 1.000 | 231.11 | approximately 100% |

The existing central Australian reference plant is cheap, at about
208.64 EUR 2020/t, but wind-intensive and land-hungry. At 1 Mt/year with land
enforced, its replacement uses about **119 MW wind, 700 MW fixed PV and
3,053 MW tracking PV**. Wind occupies 23.724 km2 and PV 78.714 km2 within the
102.438 km2 shared budget. The roughly **10.8% cost increase** buys a different
feasible design; it does not create land or demonstrate the historical capacity.

Thus excluding this cell solely because the cheapest reference plant scales
to less than the 1 Mt/year supplier threshold would miss a feasible supplier
under these model assumptions. It does not establish its competitiveness in
the shipping network after transport and other suppliers are considered.

## What is held constant

- Same centered-cell, source-substituted land input and 2% shipping share.
- Same explicit fixed-PV density and tracking footprint twice that of fixed PV.
- Same individual wind/PV areas and exclusive shared renewable-area budget.
- Same Way-based EUR 2020 costs, finance, water assumptions, full 8,760 hourly
  weather observations and hourly ammonia-ramp treatment.
- Wind and both PV types remain available; grid supply is disabled.

Only the specified annual ammonia quantity changes across curve points.
The `solved_quantity` results are **not rescaled again** and are not labelled
maximum capacities. The unconstrained controls reproduce the earlier central
cost basis. The [earlier fixed-PV pilot](FIXED_PV_20260914.md) changed the available
PV technology, but still scaled an unconstrained reference design; it answers
a different question.

## The historical capacity gap remains

An independent annual-energy relaxation bounds supply under the current
assumptions. It maximizes renewable generation per available area, allows an
ideal allocation between wind and PV, and ignores hourly mismatch, storage,
compression and curtailment losses. Current conversion inputs require at least
8.99992 MWh of electricity per tonne NH3 before those additional losses.

| Site | Optimistic energy ceiling (Mt/year) | Archived capacity (Mt/year) |
|---|---:|---:|
| Atacama | 4.991 | 9.769 |
| Northwest Australia | 2.075 | 4.020 |
| Central Australia | 2.060 | 4.324 |

These are **necessary upper bounds, not achievable production forecasts**.
All three archived quantities exceed them. Reoptimizing plant mix alone
therefore cannot reproduce those quantities while preserving the current
land, packing, resource and conversion assumptions. This does not prove which
historical input or implementation was different, or make the bound universal
across different historical technologies and land-allocation rules.

## Next reconciliation steps

At **Green Lory** level:

1. The discrete quantity audit is complete with evidence types distinguished.
   Retain cost as a function of quantity; refine the maximum only if required
   for the subsequent network formulation.
2. Expand the [completed three-site native MODIS comparison](NATIVE_MODIS_20260915.md)
   to the wider diagnostic set. Native joint masks changed these three land
   inputs by less than 0.07%, so they do not explain the large capacity gap.
3. Keep historical-source recovery separate: original MODIS Collection 6 year,
   WDPA month, DEM/registration, class ledgers and original renewable sizing.
4. Test fixed-only versus mixed-PV **under enforced land** before choosing a
   global central technology rule. At low quantities tracking can still be
   cheaper; denser fixed PV becomes more valuable as land binds.

At **Green Porpoise** level:

1. Keep the archived MOD-AMB RCP4.5/70% network replication unchanged.
2. Replace the cheapest-design capacity cutoff with validated land-feasible
   supply information for the revised network. A feasible 1 Mt/year point
   establishes eligibility, not unlimited supply at that average cost.
3. Represent quantity-dependent production costs with a properly constrained
   supply curve or mutually exclusive design alternatives. Do not add the
   capacities of alternative full-plant solutions together; that would count
   the same land repeatedly. Derive incremental costs from **total annual
   cost differences**, not from differences in average LCOA alone.
4. Re-run network comparisons only after the revised land inputs pass the
   pilot/global checks. Attribute land geography, packing, plant reoptimization
   and network allocation changes separately.

## Provenance of the interim 1 Mt/year checks

Raw files are preserved in campaign
`arc_received_supply_attempt_v3/supply_curve_3cells_v3/<site>/q_1/result.csv`.
Each checkpoint records optimal termination; its file hash matches its
`status.json`. Their local recheck is separate from acceptance of the overall
failed job.

- Atacama SHA-256: `c7cbe4d1fdf8a9e8fc47386467e0e5a05e775b1a82b7901cc14beb29b22b6d42`
- Northwest Australia: `25510c1383179e8a4c474f62a2b3915e74a9f99f4297a3c3457cd5f64c1f0358`
- Central Australia: `20974f60279ed14c2e40f2b029153bcfcdddb18781e0a623edcd76db4c1df774`

The accepted common-land CSV has SHA-256
`7f786de930bd7585f9862e3f753df73fa18222d0fb690ea248bd1f46ff417ac5`.
Energy-bound calculation is implemented in `audit_supply_pilot.py`; source
weather and conversion parameters are the same pinned inputs used by the
experiment. The replication and revised land stacks remain separate.
