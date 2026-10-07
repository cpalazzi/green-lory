# scripts/: input builders

Only the scripts listed here are tracked; other files in this folder are local scratch and
stay gitignored.

Finance inputs (uniform and Ameli WACC tables) are built by `model/finance.py`
(`python -m model.finance --help`); the September builder `build_ameli_wacc_inputs.py` is in
git history.

## `fetch_travel_time.py`

Builds `data/travel_time_by_cell.csv`, the travel time to the nearest city of 50,000 people
(MAP/Oxford Accessibility to Cities 2015, through Google Earth Engine) for each onshore cell,
used by the remoteness multiplier of the spatial runs (`model/README.md`). Needs authenticated
Earth Engine access. The spatial runs are deferred while this remoteness measure is reviewed
(travel time to a city is doubtful for coastal greenfield ports), so the script is kept but not
in use.
