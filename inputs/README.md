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

## `tech/`: technology costs

One YAML per cost case; a YAML may `extends:` another and override its values
(`model/technology.py`). Costs are overnight EUR2020 per MW (per MWh for storage), on an output
basis for conversion links, with the equipment part (`tech_cost_per_mw`) and the build part
(`build_cost_per_mw`) apart.

| File | What it is | Used by |
|---|---|---|
| `tech_config_ammonia_plant_2050_dea_colocated.yaml` | The campaign case: extends the DEA 2050 file with co-located wind and PV, the tracking footprint of Bolinger and Bolinger (2022) and the baseline water price of 2 EUR2020/m3 | the key runs and the campaign; the default of every command (`model/config.py`) |
| `tech_config_ammonia_plant_2050_dea.yaml` | Danish Energy Agency technology data, 2050 values; the data sheets and rows are named in its header | the file above |
| `tech_config_ammonia_plant_2050_way_eur.yaml` | Way et al. (2022) 2050 costs as implemented in lcoa-opt (Salmon and Verschuur), converted to EUR2020 | the Way files below; the legacy annuity in `legacy_lcoa/run_legacy_cells.py` |
| `tech_config_ammonia_plant_2050_way_eur_nominal_compression.yaml` | The Way case with the nominal hydrogen-compressor cost | the reconciling run (`rep_way2050_flat_amelired_4h_tracking_nominal_h2`) |
| `tech_config_ammonia_plant_2050_way_eur_explicit_compressor_dea_tank.yaml` | Overlay of the Way case: the explicit compressor with the DEA 2050 tank-only hydrogen store | the supply pilot (`reconciliation/supply_pilot/`) |

## `finance/`: interest rates

A finance table has one row per cell and technology (`lat, lon, tech, interest_rate`, a
fraction) and is built by `python -m model.finance` (`--help` lists the commands).

| File | What it is | Used by |
|---|---|---|
| `ameli_tiam_wacc_fig2.csv` | The weighted average cost of capital by TIAM-UCL region in the low-carbon, high-carbon and reduced scenarios, read from the bubble labels of Ameli et al. (2021), Fig. 2 | `python -m model.finance ameli` |
| `amelired_interest_inputs_2050.csv` | Ameli reduced rates for 52,704 cells and 15 technologies, built in June 2026 for the replication of the legacy model (25 MB) | the reconciling run; the supply pilot; the source of the filled table below |
| `amelired_interest_inputs_2050_center.csv` | The table above extended to the 54,360 cells of the September centred land table, each missing cell taking the rates of the nearest covered cell (`python -m model.finance fill`). Gitignored (23 MB); `arc/stage_release.sh` copies it into releases after checking its hash | the Ameli key run |
| `amelired_interest_fill_2050_center.csv` | The 1,656 cells the fill added, with the cell each copies and the distance | review |
| `uniform_interest_inputs_0p05_2050.csv` | 5 % for every cell and technology (`python -m model.finance uniform --rate 0.05`). Gitignored (22 MB), copied into releases as above | the uniform-finance key runs |

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
| `country_reference_20260928_v1.csv` | 194 country names with their ISO 3166 alpha-3 codes: the September export's 177 and the aliases its provenance file lists | the supplier export (`model/suppliers.py --country-reference`) |
| `diagnostic_cells.csv` | Three cells (Atacama, north-west Australia, central Australia) | the smoke and diagnostic stages (`arc/submit_lory_sequence.sh`) |

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
