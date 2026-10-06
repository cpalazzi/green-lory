# Verschuur reconciliation: findings and remaining tests

Primary endpoint: **MOD-AMB, RCP4.5/70% adoption, Way 2050 costs, no subsidies**.
The previously selected RCP4.5/90% case remains a project benchmark, not a
published scenario. Updated 14 September 2026: both full surfaces and the
three-cell attribution diagnostics are downloaded and revalidated. Final
deposited-code network comparisons are still running.

## What is already reproducible

The archived-project equations, the supplier/transport tables recovered from
Green Porpoise commit `0a63616`, and the older unprefixed 2050 demand file give:

- Ammonia production/demand: 601.579879 Mt/year.
- Annual system cost: USD 152.830629 billion on the archived price basis.
- Cost per tonne: USD 254.048771.
- Australian production: 258.092247 Mt/year; 234 active supplier cells globally.
- Solver gap: zero. Mass balance, port demand, capacities and cost closure pass.

The sparse implementation reproduces the original ARC Pyomo objective within
USD 0.001/year, while eliminating banned pipeline variables. Original model:
7,414,740 variables; sparse model: 2,022,330. Small test networks also reproduce
the original and deposited formulations independently. Evidence is in
`results/campaigns/verschuur_reconcile_20260907_v1/networks/archival-modamb-sparse-v2/`.

This is an archival replay, **not an exact reproduction of the publication**.
The first ARC execution reached the optimum but its result exporter failed:
Pyomo's JSON writer removed zero-valued solution entries before loading them.
Loading before serialisation fixes that error; a regression test includes an
inactive supplier. The failed execution is retained for provenance.

## Green Lory: costs and capacities must be separated

These full-year, three-cell diagnostics passed campaign QA. Costs below are
USD2018-equivalent, using the inverse of the documented Way YAML conversion
`USD2018 × 0.9024 = EUR2020`. Capacities are Mt NH3/year.

| Cell (latitude, longitude) | Historical LCOA | Lory replication LCOA | Hourly central LCOA | Historical capacity | Replication corrected capacity | Central corrected capacity |
|---|---:|---:|---:|---:|---:|---:|
| Atacama (-23, -69) | 212.92 | 215.54 | 246.93 | 9.769 | 2.316 | 2.312 |
| Northwest Australia (-23, 117) | 233.69 | 235.51 | 266.85 | 4.020 | 0.955 | 0.951 |
| Central Australia (-21, 135) | 226.39 | 220.23 | 231.20 | 4.324 | 0.459 | 0.397 |

The near agreement in replication LCOA does not imply near agreement in
supplier selection. Both sampled Australian cells fall below the paper's
1 Mt/year cutoff with the corrected capacity calculation. Increasing the
allowed land fraction just to restore these suppliers would be calibration to
a desired map, not validation.

The central configuration combines hourly variability/ramping, explicit
compressor CAPEX, tank-only hydrogen storage, and water costs. Controlled
diagnostics have completed and distinguish their cost effects. The
compressor-plus-legacy-storage intermediate is deliberately an attribution
case with a double-counting risk, not a recommended estimate. Water is added
after plant optimisation and is separable analytically.

The updated capacity method accounts for tracking-PV area and technology-
specific land limits with a shared renewable-land budget. The land union is a
classwise nested-overlap lower-bound estimate, not an exact pixelwise union.
Both reference-plant designs are cost-optimised before proportional land
scaling; neither is a full optimisation of a finite site at arbitrary scale.

## Completed global checks and attribution

Each full surface contains 52,702 coordinates. A fresh local check reproduced
the ARC merge from all four shards, verified the file/manifest hashes and
confirmed exact expected-coordinate coverage. An independent arithmetic check
reconstructed capacity from available onshore land and renewable plant sizes.
The largest capacity discrepancy was below 0.000001 t/year. Both surfaces have
zero grid imports, feasibility-penalty energy below 0.001 MWh/year, reference
production within 0.00005 t/year of 1 Mt, and component-cost closure below
0.000000001 EUR/t. These checks establish internal consistency, not empirical
validation of the land or technology assumptions.

Across 14,035 matched cells with positive historical and replication capacity,
the median replication LCOA difference is **−0.97%**. The median replication /
historical capacity ratio is **0.217**. Thus a near-matching cost surface can
produce a very different network through the capacity filter alone.

| Supplier-pool measure | Archived input | Lory replication | Hourly central |
|---|---:|---:|---:|
| Unique cells meeting 1 Mt/year cutoff | 4,548 | 1,253 | 832 |
| Historically active cells now below cutoff | — | 166 of 234 | 188 of 234 |
| Historical production at those excluded cells, Mt/year | — | 408.48 | 474.10 |

The old optimizer selected 4,000 rows but only 3,996 unique IDs because of
duplicates. The table above counts unique cells before that ranking limit.
Nineteen historically eligible cells are absent from the new surfaces because
their land input permits no capacity, not because a shard is missing. They are
listed separately in the analysis. The new surfaces are not padded to 4,000
suppliers and subthreshold capacities are not promoted to 1 Mt/year.

Among the 1,253 replication-eligible cells, moving to the central configuration
raises LCOA by 12.61% at the median (10th–90th percentile: 8.92–14.14%).
The controlled three-cell cost decomposition is below, in USD2018-equivalent/t.
These are sequential changes in this order, not order-independent global effects.

| Cell | Hourly resolution and ramp accounting | Explicit compressor | Tank-only H2 storage | Water | Total increase |
|---|---:|---:|---:|---:|---:|
| Atacama (−23, −69) | +10.69 | +21.45 | −3.65 | +2.91 | +31.40 |
| Northwest Australia (−23, 117) | +9.28 | +22.91 | −3.77 | +2.91 | +31.34 |
| Central Australia (−21, 135) | +2.62 | +6.72 | −1.29 | +2.91 | +10.97 |

The larger compressor increment at solar-heavy sites helps explain why the
cost ranking can shift even when the same cost catalogue is used globally.

### Land assumptions materially affect the Australian result

At fixed optimised plant designs and unchanged available land, using fixed-PV
packing for tracking PV would yield 175 eligible Australian cells in the
replication case, versus 22 with the specified tracking footprint. For the
central design the corresponding numbers are 66 versus 1. This is a footprint
sensitivity, not a recommendation to ignore tracking spacing. Globally the
corresponding counts are 2,167 versus 1,253, and 1,688 versus 832.

Adding technology-specific onshore limits to the union-only budget does not
change the number of eligible cells in either full surface. Therefore that
additional constraint is not the explanation for the 1,253-to-832 reduction
under the 1 Mt cutoff. Changing the plant designs and retaining the tracking
footprint is the important distinction here.

Australia's eligible production capacity is only 25.01 Mt/year in replication
and 1.02 Mt/year in the central case. The latter cannot reproduce the archived
258.09 Mt/year Australian output, regardless of network solver choices.
This strong result is conditional on the 2% land budget, protected/slope
exclusions, PV footprint and the 1 Mt/year cutoff; it is not evidence that
Australia is intrinsically uncompetitive for ammonia production.

![Supplier eligibility comparison](../results/campaigns/verschuur_reconcile_20260907_v1/comparison/eligible_supplier_maps_20260914.png)

Analysis: `comparison/global-20260914-v2/` and
`comparison/land-physics-20260914-v2/` within the campaign directory.

## Green Porpoise: published code is not the later project code

The paper's [public code deposit](https://data.mendeley.com/datasets/v4yz7778mh/1)
contains seven source/licence files, but not the five numerical input datasets.
All seven files were retrieved and verified against the deposit's SHA256
checksums. Public versions 1 and 2 have identical file checksums.

| Assumption | Deposited code | Later archived project |
|---|---|---|
| HFO-to-NH3 conversion | 39 / 18.8 | 39 / 18.6 |
| Port activation | Positive maritime flow activates a binary | Reversed inequalities do not activate the binary |
| Maritime port storage | At least 1.5 Panamax loads when active | That minimum can be bypassed by zero activity binaries |
| Import/export exclusivity | None | Present, but ineffective with zero binaries |
| Global demand lower bound | None in addition to port demands | Includes demand at names missing from the port set |
| Pipeline cost coefficient | 0.025612 USD/t/km | 0.0256 USD/t/km |

The later project is therefore not a neutral implementation of the deposited
model. Storage logic, demand conversion and demand coverage can change both
delivered costs and the network independently of the LCOA surface.

The deposit also sets a 1.5% solver gap. The current comparison runs request
0.1%, so a small difference from a published rounded cost need not establish
an input error. Regional production and alternative near-optimal networks
still need comparison.

Further source/prose discrepancies remain explicit:

- The paper describes a 10% pipeline overdesign factor. The recovered stored
  distances are great-circle distances without that factor.
- Its prose allows smaller storage at small ports (a one-year cap). The public
  code retains the uncapped 1.5-Panamax requirement.
- The public storage-cost coefficient is calculated per cubic metre but is
  applied to mass-valued storage variables without a density conversion.
- Four border-cell IDs are duplicated among the selected 4,000 supplier rows;
  the effective Pyomo supplier set contains 3,996 IDs. Costs and capacities of
  each duplicate agree, but country metadata can differ.
- Three demand names are absent from the transport port set: Georgetown
  (Guyana), Santa Cruz de la Palma, and Soyo Angola LNG Terminal. They must not
  be silently reassigned to other ports with similar names.

## Exact historical demand is not yet pinned

The older unprefixed demand file with the **later** 18.6 conversion gives
601.580 Mt/year, close to the paper's 601.5 Mt/year. That closeness initially
looked promising, but is insufficient evidence once the deposited 18.8
conversion is checked. Using the deposited conversion gives 595.180 Mt/year
from that file, or 610.437 Mt/year from the prefixed file. The latter has
609.766 Mt/year assigned to ports in the transport set.

No demand rescaling has been applied to force agreement. An exact reproduction
requires either locating the publication's actual input snapshot or documenting
the remaining input discrepancy and its sensitivity. The publication reports
260 USD/t for MOD-AMB; reproducing that rounded scalar alone is not a network
validation test.

## Present recommendation

At Green Lory level, retain the historical-assumption emulation as a regression
baseline and the hourly explicit-compressor/tank-only configuration as the
candidate improved plant estimate. Its physics and accounting are more
defensible, but it still assumes optimistic Way learning, reduced WACC, flat
construction/remoteness costs, uniform water cost and a uniform 2% land share.
Do not yet call it a fully realistic spatial cost surface.

At Green Porpoise level, use the deposited equations as the reproduction
reference; retain the later project equations only as a separate regression
case. Compare the published formulation, paper-prose adjustments and explicit
unit/accounting corrections separately before substituting the new surface.
Use the same demand, currency basis, supplier cutoff and routing rules when
isolating cost or land effects. Report solver bounds and country/subregional
production, not just map appearance.

See [RUNS.md](RUNS.md) for submissions and [notifications.md](notifications.md)
for the notification history. The user confirmed receipt of the Slurm emails
on 14 September; the earlier delivery delay's cause is not established.
