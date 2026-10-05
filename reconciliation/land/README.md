# Land reconstruction and revised inputs

Two independent stacks are maintained here. Neither overwrites September's
inputs or outputs.

- `replication/`: reconstruct Verschuur section 4.7 and Supplementary Table 2,
  with the 2% shipping allocation, 15 degree slope limit, onshore-only wind,
  200 km2/GW wind footprint and approximately 9 km2/GW equatorial fixed-PV
  footprint with van de Ven equation 6 latitude dependence. Preserve the
  archived supplier CSV separately; it is not a raw land-observation table.
- `revised/`: corrected spatial intersection and area accounting, with fixed
  PV as the candidate central technology. Tracking is a separately labelled
  sensitivity requiring its own weather profile, cost and packing assumptions.

The replication target specifies Collection 6 MODIS and WDPA 2022, but not an
unambiguous MODIS product/year or a DEM. Available source data are MODIS
MCD12C1 2022 Collection 6.1 (0.05 degree class fractions), WDPA February 2026
and GEBCO 2025. Reconstructions using these are **source-substituted**, not exact
historical input replays. Native MCD12Q1 is 500 m; the paper's approximate
250 m description is not an exact product specification.

## Working sequence

1. Verify table factors and source metadata; audit current alignment/packing.
2. Bound and trace archived outlier capacities without statistical imputation.
3. Reaggregate MODIS by explicit cell bounds and spherical pixel areas.
4. Pilot spatially joint slope/protected masks, retaining uncertainty from
   0.05 degree class fractions. Test centered and southwest-anchored cells.
5. Compare fixed and tracking with common land before rerunning plant designs.
6. Promote full-resolution/global stacks only after pilot checks and source
   provenance are explicit.

Generated outputs live under
`results/campaigns/land_reconcile_20260914_v1/{replication,revised,audit,sources}/`.
Each output directory is new and immutable. Shared code does not imply shared
assumptions. No default production input is changed by these diagnostics.

Current findings: [land audit and reconstruction](FINDINGS_20260914.md),
[completed fixed-PV comparison](FIXED_PV_20260914.md), and [run record](RUNS.md).

Native-resolution source needs and exact three-file download links are in the
[dataset acquisition guide](DATASETS_20260915.md).
The [interim constrained-supply findings](SUPPLY_CURVE_20260915.md) distinguish
locally rechecked supplier-eligibility points from the still-unaccepted full curve.
The [completed native-MODIS comparison](NATIVE_MODIS_20260915.md) finds less
than 0.07% land change at the three diagnostic sites, with converged joint
masks and separately exported native pilot inputs. It is not a global or
exact historical-source validation.
The [certified supply-quantity report](SUPPLY_CERTIFIED_20260915.md) classifies
all 21 planned quantities despite the ARC timeout, distinguishing optimal
solutions, solver infeasibility and independent energy certificates. The
40-cell expansion needs 12 additional MODIS files (52.7 MiB).
