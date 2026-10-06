# Reconciled three-site supply quantities — 15 September 2026

## Result

All **21 planned quantities are now classified**, using 16 validated optimal
solutions, two solver-proven infeasibility results and three independent
analytical infeasibility certificates. The three unconstrained controls also
passed validation. This completes the discrete quantity experiment, not a
global LCOA surface or an exact maximum-capacity calculation.

ARC job **8811997** reached its two-hour limit at **11:06:31 BST**. Its final
state was `TIMEOUT`, not successful completion. The batch received signal 15;
the job-level `0:0` exit field does not override the timeout state. Peak batch
memory was about 4.2 GiB, below the 32 GiB request. Final files and logs are
preserved separately from the earlier live snapshot.

| Site | Highest tested feasible quantity (Mt/year) | LCOA at that quantity (EUR 2020/t) | Optimistic energy ceiling (Mt/year) | Archived capacity (Mt/year) |
|---|---:|---:|---:|---:|
| Atacama | 4.5 | 238.22 | 4.9913 | 9.7694 |
| Northwest Australia | 2.0 | 264.42 | 2.0750 | 4.0197 |
| Central Australia | 2.0 | 259.98 | 2.0595 | 4.3242 |

The feasible quantities are lower bounds on each modelled maximum. The energy
ceilings are optimistic upper bounds, not achievable forecasts. The exact
maximum lies between the two under the stated model assumptions.

![Three-site LCOA against quantity](/Users/carlopalazzi/programming/pypsa_models/green-lory/results/campaigns/land_reconcile_20260914_v1/audit/supply-certified-grid-v1/supply_curves.png)

Lines connect solved quantities only. They do not establish the cost of an
intermediate quantity. Infeasible quantities have no assigned LCOA.

## What happened at the high-demand endpoints

| Site and quantity (Mt/year) | Evidence | Ideal annual electricity shortfall |
|---|---|---:|
| Atacama 5.000 | Analytic certificate; numerical solve interrupted | 0.079 TWh |
| Atacama 9.769 | Analytic certificate; solve never reached | 43.003 TWh |
| Northwest Australia 2.500 | Solver returned infeasible | 3.825 TWh |
| Northwest Australia 4.020 | Analytic certificate; numerical solve interrupted | 17.502 TWh |
| Central Australia 4.324 | Solver returned infeasible | 20.382 TWh |

The two completed infeasibility results arrived at 10:50:53 and 10:59:57 BST
for central Australia and northwest Australia respectively. Both also exceed
the independent energy ceiling. The remaining three quantities were not
relabelled as solver-proven infeasible. Their evidence is explicitly recorded
as analytical, and the original missing checkpoints remain untouched.

## How the analytical certificate works

The source release, technology inputs, land CSV and exact hourly weather files
were verified against the run's hashes. The proof uses the **same** exclusive
shared-area allocation, fixed-PV density, tracker footprint twice fixed PV,
wind density, wake loss and conversion efficiencies as the completed points.

First, each physical bus is assigned an electricity-equivalent value. Hydrogen
is valued at the electricity required by electrolysis; ammonia includes its
hydrogen requirement and direct synthesis electricity. Every allowed link
preserves or loses that value. Checks reject energy-creating cycles,
reversible links outside the proof, active grid imports and noncyclic physical
storage. The ramp-penalty components are isolated from physical energy supply.
Summing over the cyclic year gives a minimum electricity requirement of
**8.9999188 MWh/t NH3**, before compression, storage and curtailment losses.

Second, annual renewable generation is maximized subject only to the separate
wind/PV and shared-area budgets. The best of fixed and tracking PV is allowed.
The calculation provides both a feasible allocation and a matching dual upper
bound, with 25 randomized comparisons against an independent linear-program
solver in the tests. A small numerical cushion of one part per million plus
1 MWh widens the bound before classifying a quantity as infeasible.

Dropping hourly balance, storage sizing, compression, curtailment and ramp
constraints can only make this relaxation more generous. If it cannot supply
a requested quantity, the original model cannot either. These certificates
do not validate the observational data or apply to different land/technology
assumptions. They avoid spending another full Slurm allocation on endpoints
already excluded by conservation of energy.

## What this reconciles

The historical capacities are approximately **1.94–2.10 times** even the
optimistic energy ceilings under the present assumptions. Plant redesign alone
cannot restore those capacities. For illustration, uniformly increasing all
land budgets would require at least that multiplier before the historical
quantities became energetically possible; hourly and storage losses would
require more. This is a diagnostic lower bound, not a recommendation to tune
land allocation to the archived result.

By contrast, supplier eligibility changes substantially when land is enforced
inside the optimization. Central Australia's cheapest reference design scales
to only 0.376 Mt/year. A different design supplies **1 Mt/year at 231.11 EUR/t**
or **2 Mt/year at 259.98 EUR/t**. These solutions use progressively more
space-efficient fixed PV and less wind/tracking.

The [native MODIS test](NATIVE_MODIS_20260915.md) changed suitable land by less
than 0.07% in the same three cells. That effect is negligible relative to the
capacity discrepancy and plant-redesign effect here. Agreement in these
homogeneous cells does not validate heterogeneous cells globally.

## Additional input issue to resolve

The inherited PV profiles sometimes exceed one unit of output per unit of
reference nameplate. Atacama peaks are 1.060 for fixed PV and 1.123 for tracking.
This is not automatically erroneous: output normalized to standard-test DC
power can exceed that reference under different irradiance/temperature, while
AC inverter ratings introduce a different limit. The files lack explanatory
metadata, so the original normalization, inverter treatment, cost basis and
land-density basis should be traced before claiming engineering consistency.
[pvlib's definition of DC reference power](https://pvlib-python.readthedocs.io/en/v0.10.3/reference/generated/pvlib.pvsystem.pvwatts_dc.html).

No profiles were clipped or changed for this controlled comparison. Capping
them at one would reduce Atacama's fixed-PV annual energy by 0.131% and tracking
energy by 1.499%. Corresponding Australian tracking changes are below 0.009%;
fixed PV never exceeds one there. This simple clipping sensitivity cannot
explain the approximately twofold capacity gap, but the normalization remains
an input-quality question for the revised model.

## Next Green Lory and Green Porpoise steps

At Green Lory level:

1. Compare fixed-only and mixed-PV designs at the same land-constrained
   1 Mt/year threshold. Use the existing fixed-only unconstrained controls
   to check the cost basis. This isolates the technology choice the revised
   central model needs; it does not change land or weather. **Submitted as
   8812986, `glr-land-fixed-3-v1`**, after a successful six-solve preflight.
2. Expand native land validation to the existing 40-cell set. NASA's catalogue
   requires **12 additional tiles, 52.7 MiB**, beyond the three already present.
   The [download list](../../results/campaigns/land_reconcile_20260914_v1/sources/native-modis-40cell-download-plan-v1/DOWNLOADS.md)
   has exact filenames and excludes received files.
3. Recover the original MODIS/WDPA/terrain versions and trace historic PV
   density, AC/DC normalization and shared-land rules. Keep observational
   changes separate from suitability and shipping-allocation sensitivities.

At Green Porpoise level, preserve the archived MOD-AMB RCP4.5/70% replay.
For the revised model, use land-feasible quantity/cost information rather than
the scaled cheapest-design cutoff. Do not add alternative full-plant
capacities together or assign all capacity the 1 Mt average LCOA. A discrete
design formulation or independently validated incremental-cost construction
is needed, with increments calculated from **total annual cost differences**.
No global supply table or shipping network has yet been replaced.

## Reproducibility

Campaign paths under `results/campaigns/land_reconcile_20260914_v1/`:

- `arc_received_supply_attempt_v4/`: final raw files, logs and scheduler status.
- `audit/supply-checkpoints-v4-final/`: local checks of 21 completed
  checkpoints, including controls and two infeasible endpoints.
- `audit/supply-certified-grid-v1/`: all 21 planned quantities, site-level
  bounds, explicit certificates, provenance and the figure.
- `sources/native-modis-40cell-download-plan-v1/`: 40 pinned catalogue
  responses, cell definitions, full inventory and additional-only download list.

Historical replication inputs, revised candidate inputs and raw sources remain
separate. The classified result table distinguishes feasible, solver-infeasible
and analytically infeasible records instead of inventing zero costs for failures.
