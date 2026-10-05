# Green Lory supplier reconciliation campaign

This campaign keeps the comparison and the preferred estimate separate:

- `00_smoke`: three diagnostic cells over 168 simulated hours; operational QA only.
- `10_replication`: strict published-method comparison (4-hour legacy temporal
  convention, legacy per-snapshot ramping, tracking PV, nominal compressor,
  bundled hydrogen storage, plant-gate headline). Its scenario ID is
  `rep_way2050_flat_amelired_4h_tracking_nominal_h2`.
- `20_central`: preferred coherent case (hourly snapshot weights, hourly ramp
  basis, tracking PV, explicit compressor, DEA tank-only storage, and uniform
  baseline water in the headline). Its scenario ID is
  `central_way2050_flat_amelired_1h_tracking_explicit_compressor_dea_tank`.
- `30_sensitivities`: one-axis-at-a-time variants after both baselines pass.

Both baseline IDs explicitly say `flat_amelired`: the override contains Ameli
reduced WACC only, build/remoteness multipliers remain one, and water uses the
resolved YAML baseline of 2 USD/m3. The replication reports that water cost but
keeps it outside the plant-gate headline; central includes it. No nonzero
land-rent input is available, so land rent is recorded as unmodelled/zero rather
than described as an active cost. Spatial build/remoteness/water belongs in a
separately named `spatial_build_remote_water_amelired` sensitivity.

Both baselines solve the 1 Mt reference plant without land caps and apply land
availability during capacity estimation. Tracking PV uses its own density. The
full-year diagnostic and global runs require a land table rebuilt with the
versioned classwise-nested v1 renewable-union estimate. It is the lower-bound
union implied by assuming perfect overlap of the wind/solar eligible fractions
inside each MODIS class; the source data do not support an exact spatial union.
Legacy aggregated tables are permitted only for preliminary runtime smoke runs
and are flagged in every result. Full-year diagnostic and global runs require
the versioned v1 columns and method metadata.

ARC artifacts are separated from source releases:

```text
green-lory-campaigns/lory_reconcile_20260722_v1/
├── land/                 # immutable 100%, 2%, and 50% tables plus land logs
└── results/              # phase/scenario/run-id/stage outputs
```

Global QA retains finite zero-capacity rows (for example offshore-only cells)
and reports their count. Green Porpoise supplier eligibility is based on the
explicit corrected capacity column, not on whether a Green Lory LCOA row exists.

The current central capacity method is the strict proportional `paper_scaled`
method. A separate maximum-output optimization is the next scientific phase and
must be reported in its own column rather than replacing this baseline silently.
