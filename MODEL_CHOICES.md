# Modelling choices of the improved green-lory model

One list of the choices behind the improved runs: which are settled, and which are open. Once
the open ones are settled, this file becomes the model specification and the repository
reorganisation (README, "Planned layout") starts. Written 6 October 2026 from the decisions of
16 September to 5 October; dates say when a choice was made. "Cost to change" says whether
changing the choice needs only post-processing of the existing key surfaces or a new solve.

## Settled

| Choice | Decision | Since | Where it is set |
|---|---|---|---|
| Technology costs | DEA 2050; Way 2050 only for the reconciling run | 16 Sep | `tech_yaml` in `runs.yaml` |
| Plant values checked (D9) | DEA costs per unit of hydrogen or ammonia converted from LHV to the model's HHV buses (hydrogen tank, Haber-Bosch); DEA's compressor with its own electricity use (161.8 kEUR/MW, 9.1 %); Haber-Bosch minimum load 20 % (DEA) and ramps 20 %/h (Armijo and Philibert 2020); key-run LCOA -19 to -22 EUR/t at the focal cells. Ammonia tank 80.3 EUR/MWh since 28 Oct (502 EUR2020/t: Cesaro 2022, the low end of Nayak-Luke et al. 2021, the value of green-caribou; key-run LCOA -1.8 to -5.6 EUR/t at the focal cells), 260 EUR/MWh from 14 Oct (Fletcher et al. 2023, 3,000 AUD/t), which the October campaign used | 14 Oct, 28 Oct | `inputs/tech/tech_config_ammonia_plant_2050_dea.yaml`; `CAMPAIGN_PLAN.md` section 4 |
| Currency | EUR2020 for LCOA; supplier contracts in USD2018, the network model's currency | 23 Sep | result columns, contract export |
| Finance | uniform 5 % WACC as the base; Ameli low-carbon WACC by country in the campaign's finance runs (F, S2) until the collaborator's technology-specific table; Ameli reduced WACC by country moved the median LCOA by under 1 % (key run 2) | 5 Oct, 14 Oct | `interest_rate` (uniform) or `finance_csv` (by country) in `runs.yaml` |
| Time resolution | hourly, one weather year (2019); a four-hour step lowers LCOA by 2.1 % at the median and raises supply by 11 % (step 4, 1,317 sampled cells: Australia and a latitude-stratified global sample), so it is not used | Sep, 14 Oct | `time_step_h` |
| Weather of coastal cells | the 1-degree profiles average the cell, sea included; for the campaign the profiles of cells with at least 1 % water are rescaled to their land with ERA5 at 0.25 degrees | 8 Oct | `weather_land_correction`; thresholds in `inputs/parameters.yaml` (weather) |
| PV | fixed tilt only; tracking-only is a named sensitivity; the optimiser does not choose between them | 16 Sep | `pv_policy`, plant bundle |
| Tracking footprint | 1.458 x the fixed-tilt area (Bolinger and Bolinger 2022) | 16 Sep | `inputs/tech/tech_config_ammonia_plant_2050_dea_colocated.yaml` |
| Wind/PV land sharing | co-located: only the wind direct-impact share (0.03 of the footprint, Denholm et al. 2009) excludes PV | 16 Sep | `land_allocation` |
| Land table | centred 1-degree cells; MODIS MCD12C1 2022 with the Table-2 factors; WDPA exclusions counted on land only (water inside protected areas left out since 8 October); land above 20 degrees excluded (15 until 7 October) and the land's share in three slope classes recorded | 16 and 23 Sep, 7-8 Oct | `land_table` in `runs.yaml` (the October build, `land_center_slope20_20261008_v2` on ARC, for the campaign); the rules in `inputs/parameters.yaml` (land) |
| Land share | 20 % of suitable land as the base, 2 % as the sensitivity | 16 Sep | `land_share`, `land_table` |
| Land and capacity | land applied after the solve; capacity = the 1 Mt/yr reference design scaled until it fills the land, stated as a conservative lower bound, with the three-cell finite-site curves quoted as its bias (at 2 %: Atacama 3.4 against 4.0 Mt/yr feasible, northwest Australia 1.17 against 1.75, central Australia 0.38 against 2.0; approximate, the two use different cost sets) | 5 Oct | `land_constraint`, `capacity_rule` |
| Water | flat 2 EUR2020/m³ (1.62 m³/t: DEA 2050 electrolysis, 9.0 kg of water per kg of hydrogen, air-cooled; 1.5 until 15 Oct) in the headline LCOA. Spatial runs: the same source price at the nearest source plus a pipeline (TWDB Uniform Costing Model, 1.73 USD2020/m³ per 100 km at multipliers of 1); sources are the sea and permanent lakes and reservoirs covering at least 10 % of a cell (Global Lakes and Wetlands Database v2, classes 01-03 and 06; saline lakes only where the JRC Global Surface Water shows water at least half the time) and large rivers covering at least 1 % (classes 04-05), not ephemeral lakes or salt pans; the same price at sea, lake and river, a shadow price that protects freshwater for other uses (**state in the paper**); no pumping for the elevation gain | 23 Sep, 7-9 Oct, 15 Oct | tech YAML, `site_costs_in_headline`; `inputs/parameters.yaml` (water_sources, site_costs.water_pipeline); `CAMPAIGN_PLAN.md` section 0b |
| Build cost, land rent | flat build cost (multiplier 1), no land rent. Spatial runs: equipment factor by distance to the nearest port, installation factor by travel time to a city of 20,000, slope factor on PV and wind (`model/README.md`, cost mechanisms; calibration table open at the step-3 review) | 23 Sep, 7-8 Oct | `cost_scope`, `site_costs_csv`; `inputs/parameters.yaml` (site_costs) |
| Cells | onshore cells with suitable land (16,509) | 23 Sep | `--global-locations`; the rule in `inputs/parameters.yaml` (run_cells) |
| Grid | no grid backstop | Sep | wrapper |
| Networks | 1 Mt/yr supplier cutoff; deposited equations; results accepted at the deposit's 1.5 % gap, each run's gap reported | 16 Sep, 5 Oct | `network_defaults` in `runs.yaml` |

The capacity choice above replaces section 7.2 of the archived 16 September decision, which
named land-enforced finite-site solves; `reconciliation/DECISION.md` states it.

## Open

### 1. Fixed-PV power density (agreed 5 Oct to keep one rule; value open)

Three latitude-packed rules are in use, which differ in their equator value:

| Rule | At the equator | At 23° | Used by |
|---|---|---|---|
| run-time recomputation from the YAML (0.01 km²/MW) whenever a land table has no `solar_density_method` column | 100 MW/km² | 73.6 MW/km² | every `run_global` surface, including the key runs; pilot v6 |
| land-table column (First Solar module packed after van de Ven 2021) | about 106 | about 78 | the legacy stated method (legacy key run) |
| literal paper reading (9 km²/GW at the equator) | 111 | 83.45 | pilots v1 to v5 |

Empirical check: Bolinger and Bolinger (2022) give a 2019 US median of 0.35 MWdc/acre for fixed
tilt, 87 MWdc/km²: above all three rules even at 23°, although US latitudes need wider row
spacing than 23°.

Recommendation: keep the land-table column as the single rule (written once into the land table
with `solar_density_method`, run-time recomputation removed). It is the paper's stated method,
the legacy stated method uses it (so the legacy/green-lory comparison carries no density
artefact), and the empirical median says it is not optimistic. Before comparing with Bolinger,
settle whether the model's PV megawatts are AC or DC: the PV profiles exceed 1.0 at times and
their normalisation basis is undocumented (`data/README.md`), and a DC/AC ratio of about 1.3
would change the comparison more than the choice between these rules.

Cost to change: post-processing. The key surfaces solved without land, so PV-limited capacities
scale by the density ratio (+6 % for the recommended rule); new supplier contracts and four
network jobs (about 2 h each).

### 2. Weather data

Every run, legacy and green-lory, uses the 2019 profile stack made for lcoa-opt (wind x 0.93
wake factor, PV normalisation undocumented). The ERA5/atlite 2019 capacity factors on ARC give
7 to 9 % lower fixed-PV and 26 to 38 % lower raw wind capacity factors at the focal cells, have
no tracking field, and have never been run through a solve. The choice does not affect the
legacy/green-lory comparison (both use the same stack), but it moves absolute LCOA more than any
other open item.

Recommendation: keep the legacy stack for the comparison notebook. For the improved model,
decide whether the papers quote absolute LCOA; if they do, run the 563-cell legacy sample with
both weather sources first (small runs) and adopt ERA5/atlite if the difference holds
globally.

Cost to change: a weather loader for the atlite tiles, then new global solves (about 3 h on four
48-core jobs per surface, each submitted only with your confirmation).

### 3. Which further surfaces the papers need

The 16 September plan (`reconciliation/archive/2026-09/PLAN_GLOBAL_CAMPAIGN_20260916.md`
section 4) proposed eight surfaces S1 to S8 (2030 and 2050; finance; spatial build and
remoteness; spatial water; 2 % land). The three key surfaces cover S1, S3 and S8. Open: whether
the sensitivity paper needs the spatial surfaces (S2, S4, S5) and the 2030 surfaces (S6, S7).
The spatial ones wait for a review of their inputs: travel time to the nearest city is doubtful
as remoteness for coastal greenfield ports, and the water-access costs are unreviewed.

Recommendation: fix this list before the workflow is written, because each surface becomes a
run id; keep offshore cells out until bathymetry exclusions and costs exist. See the next
section for the sensitivity paper.

### 4. Smaller open items

- Co-location reference: the 0.03 wind direct-impact share rests on Denholm et al. (2009); a
  better source is a to-do. Post-processing only.
- Capacity range: the scaled design is a lower bound; a per-cell energy-ceiling upper bound is a
  cheap addition if a reviewer asks for a range. Global finite-site curves are not proportionate.
- The legacy key run's id says `way2050` although it reads the x_Cost sheet (`xcost45`). Keep
  it (results are filed under it) or rename with the old id as an alias.

## Next step: the global LCOA heatmap sensitivity paper

With land applied after the solve, LCOA does not depend on the land table, the land share or
the PV density: those change capacity only. The heatmaps of a cost-sensitivity paper therefore
depend on the weather (open item 2), the technology costs and the cost mechanisms (finance,
build and remoteness, water), not on the land choices still open. In order:

1. **Weather basis.** Run the 563-cell legacy sample (`legacy_lcoa/cells_sample_v1.csv`) with the
   improved settings on the legacy stack and on the ERA5/atlite capacity factors (small runs).
   If the difference holds, the paper's surfaces should use ERA5/atlite, and they need a
   tracking field for the tracking sensitivity.
2. **Review the spatial inputs** before any spatial surface: remoteness for coastal cells
   (distance to a port instead of travel time to a city) and water access (desalination plus
   pipeline from the coast).
3. **One-at-a-time surfaces against S1** (`gl_dea2050_wacc5_bflat_wflat_land20c_fixed`), each a
   new run id in `runs.yaml` with a run-store row: build and remoteness alone (S4), water alone
   (S5), all spatial mechanisms (S2), and DEA 2030 (S6). For finance, Ameli's reduced WACC is
   5.1 % almost everywhere, so it moves the median LCOA by under 1 % against uniform 5 %; a
   borrowing-cost sensitivity needs Ameli's low-carbon (current) WACC series (8 to 12 % in the
   regions that matter; `inputs/finance/ameli2021_lowcarbon_wacc_by_country.csv`) or a
   uniform 10 %. A tracking-only twin of S1 is worth adding: tracking changes the ranking of
   PV-only regions (Chile) against wind–PV hybrids (`reconciliation/DECISION.md` section 3).
4. **Figures**: LCOA maps and maps of the change against S1 for each mechanism, the mechanism
   that dominates each cell, and cost-quantity supply curves by region at 2 % and 20 %.

Each hourly 16,509-cell surface is a global submission (confirm each); about 580 CPU-hours in
September for 52,702 cells, so fewer for the onshore subset.

## Structure

Not modelling choices, but they decide what the reorganised code looks like.

Done on 6 October: the plant bundles live in `model/plants/`; the dead functions and the unused
Pyomo import are gone from `model/`.

Done on 7 October (CAMPAIGN_PLAN.md step 2; the focal-cell regression reproduces the accepted
results):

1. **One source for plant parameters.** One plant bundle, `model/plants/ammonia_plant_1mt/`;
   each run takes the link efficiencies from its YAML recipe and the tracking switch from its
   `pv_policy`. The load stays in the bundle (1 Mt/yr for every run).
2. **One capacity column family.** The `max_*` columns are gone; the QA gate checks the
   `scaled_design_*` columns.
3. **Grid backstop.** Removed: the plant has no grid imports and nothing can switch them on.
4. **Dead options.** Removed: the south-west cell anchor, the conservative union fallback, the
   `aggregation_count` plumbing, the netCDF land-cover path and the unused land-build options.
   Still to settle with open item 1: the run-time PV density recomputation.
