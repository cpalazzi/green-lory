# scripts/: input builders

Only the scripts listed here are tracked; other files in this folder are local scratch and
stay gitignored.

## `build_ameli_wacc_inputs.py`

Builds the Ameli et al. (2021) Fig. 2 WACC inputs (`amelired` = the reduced-WACC case) from
`inputs/ameli_tiam_wacc_fig2.csv`: the per-cell interest-only override
`inputs/amelired_interest_inputs_2050.csv` (used by the reconciling run) and, for spatial runs,
`inputs/spatial_cost_inputs_amelired_2050.csv`, which keeps the build, remoteness and water
columns of a spatial base file and replaces `interest_rate`. The finance overrides of the key
runs on the centred cells are built by `arc/build_finance_overrides.py` from these.

```bash
python scripts/build_ameli_wacc_inputs.py   # --help for the inputs and outputs
```

The interest-only output covers the union of the spatial base grid and `--locations-input`;
the combined output covers only the spatial base grid and warns about cells missing from it.

## `fetch_travel_time.py`

Builds `data/travel_time_by_cell.csv`, the travel time to the nearest city of 50,000 people
(MAP/Oxford Accessibility to Cities 2015, through Google Earth Engine) for each onshore cell,
used by the remoteness multiplier of the spatial runs (`model/README.md`). Needs authenticated
Earth Engine access. The spatial runs are deferred while this remoteness measure is reviewed
(travel time to a city is doubtful for coastal greenfield ports), so the script is kept but not
in use.
