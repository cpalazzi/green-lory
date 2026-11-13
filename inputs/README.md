# inputs/: what the model reads, apart from raw data

This folder holds the model's assumptions and the small tables copied from publications
(tracked), and the tables generated from them that are small enough to track, each with a
`*_provenance.json` beside it giving its inputs' hashes. Generated tables too large for git are
gitignored here, and `runs.yaml` pins their hashes. Elsewhere:

- raw data (land cover, terrain, protected areas, lakes, travel times, weather): `../data/`,
  listed with hashes in `../data/README.md`;
- land tables and other campaign outputs: `../results/campaigns/` and the campaign folders on
  ARC; `../data/README.md` names the accepted ones;
- which of these files each run uses: `../runs.yaml` (`green-lory show <run id>`).

## `parameters.yaml`: the modelling assumptions

Every modelling assumption other than technology costs, one value per key with its source:
the plant's reference output (`plant`); land-cover availability, protected-area statuses, slope classes and PV row geometry (`land`);
the rule for the cells a campaign solves (`run_cells`); the wind wake factor and the land
correction's thresholds (`weather`); what counts as a water source (`water_sources`); and the
remote factor and cold rule, the slope factors and the water pipeline's unit costs (`site_costs`), the costs of plants at sea (`offshore`);
and the physical constants with a source, such as the heating values that turn energy into
tonnes (`physical`).
`model/parameters.py` reads it; the land, geography and site-cost tables and every run manifest
record its hash, so a table built with other values can be told apart. A table built before a
change of these values must be rebuilt (the land tables on ARC).

## `tech/`: technology costs

One YAML per cost case; a YAML may `extends:` another and override its values
(`model/technology.py`). Each file gives every technology's values: overnight costs in EUR2020
per MW (per MWh for storage, on an output basis for conversion links) with the equipment part
(`tech_cost_per_mw`) and the build part (`build_cost_per_mw`) apart, lifetime, fixed O&M and
default interest rate, each link's conversion recipe, and the operating values (the ammonia
synthesis minimum load and ramp limits, the battery's standing loss per day). The plant bundle, `model/plants/ammonia_plant_1mt/`, holds only the plant's
structure.

| File | What it is | Used by |
|---|---|---|
| `tech_config_ammonia_plant_2050_dea_colocated.yaml` | The campaign case: extends the DEA 2050 file with co-located wind and PV, the tracking footprint of Bolinger and Bolinger (2022) and the baseline water price of 2 EUR2020/m3 | the key runs and the campaign; the default of every command (`model/config.py`) |
| `tech_config_ammonia_plant_2050_dea.yaml` | Danish Energy Agency technology data, 2050 values; the data sheets and rows are named in its header | the file above |
| `tech_config_ammonia_plant_2050_way_eur.yaml` | Way et al. (2022) 2050 costs as implemented in lcoa-opt (Salmon and Verschuur), converted to EUR2020 | the Way files below; the legacy annuity in `legacy_lcoa/run_legacy_cells.py` |
| `tech_config_ammonia_plant_2050_way_eur_nominal_compression.yaml` | The Way case with the nominal hydrogen-compressor cost | the reconciling run (`rep_way2050_flat_amelired_4h_tracking_nominal_h2`) |
| `tech_config_ammonia_plant_2050_way_eur_explicit_compressor_dea_tank.yaml` | Overlay of the Way case: the explicit compressor with the DEA 2050 tank-only hydrogen store | the supply pilot (`reconciliation/supply_pilot/`) |

## `finance/`: interest rates

Interest rates come from the run, never from a technology YAML. A run with one rate for every
technology and cell sets `interest_rate` in `runs.yaml` and needs no file. A run with rates that
vary sets `finance_csv` to a finance table, pinned by its hash, in one of two forms:

- **by country**: columns `iso3, tech, interest_rate`. Each cell takes its country's rates through
  `cells/cell_countries.csv`;
- **by cell**: columns `lat, lon, tech, interest_rate`, for anything finer than a country.

`tech` is a technology of the plant (`solar`, `solar_tracking`, `wind`, `electrolysis`,
`ammonia_synthesis`, `battery_storage`, ...: the names of the technology YAMLs) or `all`, which
gives the rate of every technology the table does not name for that country or cell.
`interest_rate` is a fraction. Other columns (a source, a region) are kept for the reader and
ignored by the model; site-cost columns are refused. Every technology at every cell a run solves
must get a rate, or the run stops before solving and lists the gaps
(`model/technology.py`, `finance_overrides`). The tables by country of Ameli et al.'s rates are built in the repository
(`python -m model.countries`, from the two inputs below); other datasets are turned into finance
tables outside it, and the table records the result. Tables by country are small and tracked;
tables by cell are large and gitignored, and `arc/stage_release.sh` copies the one a run pins
into its release after checking the hash.

| File | What it is | Used by |
|---|---|---|
| `ameli2021_tiam_ucl_regions.csv` | The countries of each of the 16 regions of the TIAM-UCL model, transcribed from Ameli et al. (2021), Supplementary Table 4 (doi:10.1038/s41467-021-24305-3), one row per ISO code, with the table's three catch-alls (Other Africa, Other Latin America, Other Asia) as rows without a code and notes where a name needs reading (Congo, Netherlands Antilles, Yugoslavia) | `model/countries.py` (the region of each sovereign country) |
| `ameli2021_wacc_by_region.csv` | Ameli et al.'s (2021) weighted average cost of capital of each region, read from the bubble labels of Fig. 2: the low-carbon (current) and the reduced series | `model/countries.py` (the rates of the tables below) |
| `ameli2021_lowcarbon_wacc_by_country.csv` | Ameli's low-carbon (current) rates by country: one rate for all technologies for each of the 238 countries and territories of Natural Earth's 1:10m map (`data/README.md`), the rate of its region (`ameli_region`), which is the region of its sovereign country (`sovereign_iso3`) in the region list above, or for a sovereign the list does not name the list's catch-all for its continent (`region_basis`); Antarctica (no run cell) takes the highest rate, as international waters do (`python -m model.countries`, 13 November 2026; `*_provenance.json`) | the low-carbon sensitivity of the campaign's `finance` and `all` (those cases from 14 October to 11 November 2026, decision D5) |
| `ameli2021_reduced_wacc_by_country.csv` | The same for Ameli's reduced series (5.1 % in most regions; 1.8 % in Japan, 4.0 % in Western Europe, 4.3 % in the UK, 4.4 % in the United States and 4.6 % in Canada): the series of Verschuur et al. (2024) | the campaign's `finance` and `all` runs from 11 November 2026 |
| `ameli2021_reduced_wacc_sea_cells.csv`, `ameli2021_lowcarbon_wacc_sea_cells.csv` (by cell, gitignored) | The rates of the 40,197 sea cells: each takes the rate of the sovereign country of the exclusive economic zone that holds it and in international waters the highest rate of the table, 5.1 % (reduced) or 11.8 % (low-carbon; `offshore` in `../parameters.yaml`); `sea_finance_table` of `model/sea_cells.py` with the table by country above (rebuilt on 13 November 2026 with the tables by country of `model/countries.py`: the reduced table byte for byte, the low-carbon table with 125 cells of the zones of Fiji and Cyprus at the rates of Other Asia and the Middle East) | the sea runs of `finance` and `all`, and their low-carbon sensitivity |
| `amelired_interest_inputs_2050.csv` (by cell, gitignored) | Ameli reduced rates for 52,704 cells and 15 technologies, built in June 2026 for the replication of the legacy model (25 MB). Tracked until 9 October 2026: `git show 79b72f2:inputs/amelired_interest_inputs_2050.csv` recovers it | the reconciling run; the supply pilot |
| `amelired_interest_inputs_2050_center.csv` (by cell, gitignored) | The table above extended to the 54,360 cells of the September centred land table, each missing cell taking the rates of the nearest covered cell (23 MB) | the Ameli key run of September |
| `amelired_interest_fill_2050_center.csv` (gitignored) | The 1,656 cells that fill added, with the cell each copies and the distance | review |

## `site_costs/`: location factors

The tables behind the equipment, installation and slope multipliers and the water price: the
AEMO consultants' location-factor tables (transcribed), the per-cell geography (travel times,
port distances, water), the remoteness fit and the site-cost table of the October land build.
`site_costs/README.md` describes each file.

## `weather/`

| File | What it is | Used by |
|---|---|---|
| `era5_point_weights_1deg.csv` | For each of the 10,971 cells with at least 1 % water, the land and total area weights of the 25 ERA5 points (0.25 degrees) inside it, from the Global Lakes and Wetlands Database v2 (`python -m model.weather_land weights`) | the land correction of the weather profiles (`arc/jobs/10_weather_land_correction.sh`) |

## `cells/`: cell and country lists

| File | What it is | Used by |
|---|---|---|
| `cell_countries.csv` | Each cell's country for the 54,360 cells of the land tables: the country or territory covering most of the cell in Natural Earth's 1:10m map (`data/README.md`; `assignment` overlap), else the country of the nearest such cell, with the distance (`nearest`: the sea of the land tables' rows; 27 of the 16,885 run cells). A territory keeps its own ISO 3166 code; an area without one takes the code of the country ISO 3166 counts it in (Somaliland Somalia, Northern Cyprus Cyprus, Kosovo Serbia) or, for a dependency, its sovereign's; areas without any code are left out (`python -m model.countries`, 13 November 2026; until then Natural Earth's 1:110m map, with 653 run cells by the nearest rule) | finance tables by country; the results' country column (overlap cells only) |
| `un_m49_regions.csv` | The UN's standard regions (M49, United Nations Statistics Division): each country's or area's region, sub-region and intermediate region by ISO-alpha3 code, as published on the M49 overview page (248 rows, retrieved 10 Nov 2026; `un_m49_regions_provenance.json`). M49 does not list Taiwan, which `model/results.py` gives the region of its neighbours | the regions of the results notebook (`results.cell_regions`) |
| `country_reference_20261113_v1.csv` | The 238 names of the cell-country table with their codes (`python -m model.countries`) | the supplier export of runs solved with the cell-country table of 13 November 2026 (`model/suppliers.py --country-reference`) |
| `country_reference_20260928_v1.csv` | 194 country names with their ISO 3166 alpha-3 codes: the September export's 177 and the aliases its provenance file lists | the cell-country table until 13 November 2026; the supplier export of the runs solved with it |
| `diagnostic_cells.csv` | Three cells (Atacama, north-west Australia, central Australia) | the smoke and diagnostic stages (`arc/submit_lory_sequence.sh`) |
| `diagnostic_cells_sea.csv` | Three sea cells, one per depth class: Dogger Bank (fixed), the Kerguelen shelf (floating, remote) and the mid-Atlantic near the Azores (deep) | the smoke and diagnostic stages of a sea run (`--diagnostic-locations`) |
| `validation_cells_step4.csv` | 1,317 cells (`group`): all of Australia's 786 run cells of the October land table, and 562 of the 563 cells of the legacy sample (`legacy_lcoa/cells_sample_v1.csv`: the three focal cells plus 80 cells from each 10-degree band of absolute latitude between 0 and 70 degrees, drawn on 15 September 2026 from the 15,377 cells of the published supplier table; one has no suitable land in October); 31 cells are in both | the four-hour against hourly comparison of campaign step 4 (`--global-locations`) |

## Left at the top level

`spatial_cost_inputs.csv` (76 MB) holds the per-cell spatial costs of the removed notebook 02.
Only `reconciliation/coast_inland.py` reads it; both go in step 6 of `CAMPAIGN_PLAN.md`.

## References

- Ameli, N., Dessens, O., Winning, M., Cronin, J., Chenet, H., Drummond, P., Calzadilla, A.,
  Anandarajah, G. and Grubb, M. (2021). Higher cost of finance exacerbates a climate investment
  trap in developing economies. *Nature Communications* 12, 4046.
  https://doi.org/10.1038/s41467-021-24305-3
- Bolinger, M. and Bolinger, G. (2022). Land requirements for utility-scale PV: an empirical
  update on power and energy density. *IEEE Journal of Photovoltaics* 12(2).
  https://doi.org/10.1109/JPHOTOV.2021.3136805
- Danish Energy Agency. Technology Data: generation of electricity and district heating;
  renewable fuels; energy storage. The sheets used are named in each YAML.
- Way, R., Ives, M. C., Mealy, P. and Farmer, J. D. (2022). Empirically grounded technology
  forecasts and the energy transition. *Joule* 6(9), 2057-2082.
  https://doi.org/10.1016/j.joule.2022.08.009
