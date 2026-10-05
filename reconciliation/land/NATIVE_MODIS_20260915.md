# Native land-cover comparison — 15 September 2026

## Finding

**Resolving the native MODIS classes does not explain the large capacity gap
at the three diagnostic sites.** Compared with the corrected coarse-fraction
input, the native calculation changes solar-suitable shipping land by less
than 0.07%. It neither restores historical Australian capacities nor removes
Australia's renewable resource.

These are three relatively homogeneous cells, not a global validation. Both
products also originate from the same MODIS classification: their agreement
checks aggregation and spatial implementation, not independent ground truth.

| Centered one-degree cell | Previous corrected CMG solar area (km2) | Native joint-mask solar area (km2) | Change |
|---|---:|---:|---:|
| Atacama, −23, −69 | 226.8355 | 226.8270 | −0.0037% |
| Northwest Australia, −23, 117 | 113.1331 | 113.0614 | −0.0634% |
| Central Australia, −21, 135 | 102.4384 | 102.4291 | −0.0091% |

Areas include the same **2% shipping allocation, applied once**. They are
suitability-weighted areas, not the total physical land within each cell.
Wind and shared-area results are included in the machine-readable comparison;
the shared-area overlap remains the stated classwise-nested assumption.

## Data and controlled method

The three original MCD12Q1 Collection 6.1 / 2022 HDF files were moved from
Downloads into `sources/native-modis-c61-2022/` in the land campaign. Their
hashes were unchanged by the move; file sizes matched the pinned NASA
catalogue, and embedded identity, year, collection, layer dimensions, class
codes, projection and full study-cell coverage passed checks.

MCD12Q1 is nominally **500 m**, with approximately 463.313 m sinusoidal
pixels—not the paper's approximate 250 m description. The calculation uses
the native categorical `LC_Type1`, retaining quality and land/water flags as
diagnostics. Native water code 17 maps explicitly to CMG's water layer 0.
[NASA product guide](https://modis-land.gsfc.nasa.gov/pdf/MCD12Q1_C6_Userguide04042018.pdf).

The local CMG file, GEBCO 2025 elevation file and all required sidecars of the
three WDPA February 2026 polygon parts were hash-verified against the inputs
used by the preceding ARC pilot. Suitability factors, cell centering, 15-degree
slope threshold, protected-record filtering and the shipping share did not
change. Below-sea-level land was not excluded merely by its elevation sign.
Point-only protected records remain excluded in this controlled comparison.

Two calculations cross-check each other:

1. An unmasked class ledger clips native sinusoidal pixels against the curved
   one-degree cell boundary and checks physical area conservation.
2. A common integration grid samples native classes, the existing DEM slopes
   and protected polygons jointly. The CMG fractions use **exactly those same
   exclusion masks**, making the class-resolution comparison controlled.

The integration was refined from 1,200/2,400 to 2,400/4,800/9,600 samples per
degree. This is numerical integration, not newly observed fine-resolution
land cover or terrain. Between the final two resolutions, the largest relative
change in suitable area was **0.0000803%**; unmasked suitable-area error against
the geometric ledger was at most **0.0000081%**. The accepted result is
`audit/native-modis-pilot-v2/`; v1 is retained as the initial convergence run.

## Why the answer barely changes

Atacama is almost entirely barren land, with suitability factor 1. Both
Australian cells are overwhelmingly open shrubland, assigned factor 0.5 in
the paper-based rules. Moving the small remaining class fractions into their
actual native positions has little effect on weighted area.

Slope excludes about 36.51 km2 of physical area in Atacama and 4.55 km2 in
northwest Australia. Neither cell intersects retained protected polygons.
Central Australia has no pixels above the slope threshold in this DEM
calculation; protection excludes about **1,281.94 km2** of physical area.

At central Australia, two small effects partly cancel. Finer protected-boundary
integration increases the CMG estimate by approximately **0.0439 km2** of
shipping area; resolving the classes then reduces that estimate by about
**0.0532 km2**. The net change from the previous input is only −0.0093 km2.
Attributing the net difference entirely to classification would be misleading.

All Australian study-cell QC flags are zero. Atacama has about 32.41 km2 with
other QC codes and one approximately 0.215 km2 native pixel whose IGBP water
label disagrees with the separate land/water layer. These are reported, not
silently recoded; they are too small to explain the capacity discrepancy.

## Interpretation and model recommendation

The corrected implementation is more defensible geometrically: it has explicit
cell bounds, physical pixel areas, joint exclusions and auditable ledgers.
However, native resolution alone does **not** establish that the resulting
availability is realistic. The 50% shrubland suitability, 2% allocation to
shipping, PV spacing and wind/shared-land assumptions remain substantive
modelling choices. The original source years and exact historic implementation
have not been recovered.

For the revised model, retain fixed PV as the central candidate and test
tracking explicitly under enforced land constraints. Do not increase land
availability merely to reproduce an archived network. Use regional land-use
and competition sensitivities, and an independent land-cover source, to test
realism. Prioritize heterogeneous cells and protected/steep boundaries in the
40-cell expansion; the three homogeneous examples cannot settle those cases.

For historical reproduction, preserve the archived MOD-AMB RCP4.5/70% replay
and the source-substituted reconstruction as different endpoints. Recovering
the original MODIS year/collection, WDPA snapshot, terrain source and renewable
spacing rules is still necessary for an exact input-level reconstruction.

## ARC progress and the more consequential result

Job **8811997**, `glr-land-supply-3-v4`, was running at **10:03 BST**. Here `glr`
means Green Lory reconciliation, `land-supply` denotes enforced-land
cost–quantity curves, `3` is the number of sites, and `v4` is the attempt.

A timestamped local snapshot contains **19 validated checkpoints: three
unconstrained controls and 16 constrained quantities**. Result hashes, optimal
termination, controlled retry settings, production, cost accounting, grid/slack
use and independently reconstructed land footprints passed local checks.
Five high-demand endpoints remain unresolved; no full curve is accepted yet.

| Site | Highest locally checked feasible quantity (Mt/year) | LCOA (EUR 2020/t) |
|---|---:|---:|
| Atacama | 4.5 | 238.22 |
| Northwest Australia | 2.0 | 264.42 |
| Central Australia | 2.0 | 259.98 |

These are tested feasible quantities, **not maximum capacities**. At the high
quantities the designs become almost entirely fixed PV, illustrating the
value of packing when land binds. Central Australia can supply 1 Mt/year at
231.11 EUR/t despite the cheapest reference design scaling to only 0.376 Mt/year.
This plant-redesign effect is much larger than the native-versus-CMG land change.
See the [supply report](SUPPLY_CURVE_20260915.md) for energy bounds and caveats.

Green Porpoise should therefore use validated quantity-dependent supply costs,
not discard a site solely because a scaled least-cost reference design misses
the supplier cutoff. Alternative full-plant solutions must not be summed as
independent capacities using the same land.

## Outputs and next steps

Within `results/campaigns/land_reconcile_20260914_v1/`:

- `audit/native-modis-pilot-v2/`: accepted comparison, class ledgers, quality
  diagnostics, convergence checks and complete source hashes.
- `sources/native-pilot-release-v1/`: preserved 19-file source/config/test
  release; all analysis-code hashes match the accepted run manifest.
- `replication/native-paper-method-pilot-v1/`: validated three-cell,
  source-substituted paper-method input.
- `revised/native-common-geography-pilot-v1/`: separately labelled three-cell
  revised candidate using the same controlled geography and density.
- `arc_received_supply_snapshots_v4/20260915T1001BST/`: immutable partial ARC
  snapshot, not a finished run.
- `audit/supply-checkpoints-v4-20260915T1001BST/`: validated checkpoints and
  explicit list of missing endpoints.

No production land table, archived surface or running job input was changed.
Observations, calculated areas and assumptions remain separately recorded,
with units, provenance and explicit numerical checks.

Next: finish the current supply-run audit; expand native coverage to the
40-cell set; test historic terrain/packing and revised land-policy assumptions
separately; then propagate accepted differences through Green Porpoise. No
further download is required for the completed three-site test. The wider
native pilot will require additional tiles, to be specified before download.

Latest ARC status check: **10:06:41 BST**, still running, elapsed 1h00m29s.
The five outstanding quantities exceed the previously derived optimistic
annual-energy ceilings. If solver termination remains numerically unresolved,
use a separately documented analytic infeasibility certificate—not repeated
unbounded retries or an unsupported claim of solver-proven infeasibility.
