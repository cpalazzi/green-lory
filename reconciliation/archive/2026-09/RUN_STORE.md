# Run store and naming convention (from 23 September 2026)

Every surface, land table and network run gets one row in `run_store.csv`
(machine-readable) and a run id built from fixed tokens, in this order:

```
<model>_<costs>_<finance>_<build>_<water>_<land>_<pv>[_<step>]      e.g.
gl_dea2050_wacc5_bflat_wflat_land20c_fixed
```

| Slot | Tokens | Meaning |
|---|---|---|
| model | `gl`, `ll`, `gpo` | green-lory plant, legacy-lcoa plant, green-porpoise network |
| costs | `dea2050`, `dea2030`, `way2050`, `xcost45` | technology cost basis (legacy: x_Cost RCP4.5 sheet) |
| finance | `wacc5`, `wacc10`, `ameli` | uniform WACC in percent, or Ameli reduced WACC by country |
| build | `bflat`, `bspat` | build-cost multiplier 1 everywhere, or depth x remoteness x labour |
| water | `wflat`, `wspat`, `wnone` | uniform baseline water (2 EUR2020/m³ from 24 Sep 2026; the September runs used 2 USD2020), spatial desalination + access, or no water cost (legacy) |
| land | `land20c`, `land2c`, `land2sw`, `landleg2`, `landarch` | share and land build: `c` = centred build 20260923, `sw` = September south-west build, `leg` = legacy land step (Table 2, centred, no exclusions), `arch` = archived capacity column |
| pv | `fixed`, `track`, `both` | PV technologies offered to the optimiser |
| step | `1h` (default, omitted), `4h` | weather time step |

Networks: `gpo_<supplier run id>_<routes>_<cutoff>` (routes `iso1000` = same ISO3 or
<= 1000 km x 1.1; cutoff `1mt`). Post-hoc derivatives (another land share, a water price
change) reuse the parent's LCOA and get their own row with `upstream_run` filled in.

Flexibility: extra qualifiers go after the pv slot (`_explicitcomp`, `_offshore`), and
anything not covered by a token goes in the `notes` column. Land-rent is zero everywhere
until a `rent` token is introduced.

Statuses: `planned`, `submitted`, `running`, `merged`, `qa_passed`, `exported`, `done`,
`superseded`, `failed`. Columns of `run_store.csv`:

`run_id, model, cost_basis, finance, build, water, land, land_table, pv, time_step_h,
allocation, cells, weather, release, location, job_ids, status, submitted_utc,
upstream_run, network_runs, notes`
