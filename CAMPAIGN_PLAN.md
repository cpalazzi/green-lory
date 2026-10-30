# Plan: rationalised green-lory and the 2026 campaign

Plan of 6 October 2026, decisions completed on 7 October. Status (8 October): steps 1 and 2
done and signed off; step 3 built and at its **Review** (section 3: three points to decide), its
global land build running on ARC. Steps run in order; each ends with a check,
and the gates marked **Review** wait for you.

Working rule (7 October): clean up as we go. Each step deletes what it supersedes, listed under
**Removes**, once its check has passed; git history keeps the code. Inputs that a run definition
in `runs.yaml` still pins stay until that run is retired.

## 0. Decisions to confirm first

| # | Question | Recommendation |
|---|---|---|
| D1 | Base time resolution | **Hourly as the base; 4 h as a documented resolution case.** The 4-hour run is worth having as the bridge to the published method, but it should not be the base because it gives the coastal map: four-hour averaging flattens the solar peak and so hides the cost of buffering solar-only hydrogen, which is the mechanism that moved production inland (`reconciliation/DECISION.md` section 5). If 4 h and 1 h disagree, the paper reports 1 h and explains the difference. The 4 h step also explains only part of the shift: at the focal cells the explicit compressor added about twice as much as the hourly step, and both stay in a 4 h run of the improved plant, so it may still favour inland. |
| D2 | Slope | **Decided 7 October:** a cost factor by slope class on the build part of PV and wind, slope from GEBCO, factors from NREL's terrain multipliers, the plant takes the flattest land first, land above 20° excluded (section 0a). The class bounds follow your question on 15 % grade; confirm at the step 3 review. |
| D3 | Remoteness | **Decided 7 October:** two factors, as AEMO does: an equipment factor by distance to the nearest of all network ports (`data/ports/verschuur2024_ports.csv`), on the equipment part of the cost, and an installation factor by travel time to a city, on the build part, both fitted to the location factors behind AEMO's inputs and used worldwide (section 0a). |
| D4 | Water | **Decided 7 October** (section 0b): desalination at the nearest water source plus a pipeline to the cell, with an official water-pipeline cost replacing the district-heating proxy; sea, permanent lakes and (decided 9 October) large rivers are sources at the same 2 USD/m³; no pumping for the elevation gain; the 1° geometry is kept. |
| D5 | Finance sensitivity | **Ameli low-carbon (current) WACC**: Australia 6.1 %, Middle East 6.8 %, Central and South America 9.2 %, Africa 11.8 %. It would favour Australia over Chile and North Africa. The reduced series is 5.1 % almost everywhere and moves LCOA by under 1 % against the flat 5 %. |
| D6 | Network tolerance | **1 %** (your instruction): the solver stops at a 1 % gap and results are accepted at 1 %. |
| D7 | Weather | The 2019 legacy stack for every run (your instruction), in `data/weather_data/salmon2022_pv_wind_2019_hourly_1deg/`: hourly 2019 output per unit of capacity for fixed PV, tracking PV and wind on a 1° grid, as used by Salmon and Bañares-Alcántara (2022). The comparison with our own ERA5 cutouts is deferred. |
| D8 | Fixed-PV density | The land-table rule (about 78 MW/km² at 23°), written into the land table once; capacity only (open item 1 of `MODEL_CHOICES.md`). |
| D9 | Validation before the campaign | Check the two inputs that drive the coast/inland result: the hydrogen compressor cost (165 kEUR/MW in DEA 2050, derived by scaling a DEA 2030 storage-tank figure) and the Haber-Bosch flexibility (30 % minimum load, 40 %/h ramps, inherited from the legacy model). If the literature supports a more flexible synthesis loop or a different compressor cost, change it for every run, not to reach a map. |

My view on confidence in green-lory: confident in the code. It reproduces the legacy model
when configured like it, the land geometry is fixed, and regression tests pin the accepted
results. I am not yet confident in the two cost inputs of D9, which decide coast against inland
more than anything else in the plant.

## 0a. Location costs: proposed methods for D2 and D3

Both factors use the cost split green-lory already has. Every technology's overnight cost is
`tech_cost_per_mw` (equipment: modules, inverters, turbines, electrolyser stacks) plus
`build_cost_per_mw` (installation, balance of plant, grid, soft costs); today one multiplier
acts on the build part. The proposal adds a second multiplier, on the equipment part. Fixed O&M
is a fraction of the adjusted overnight cost, so it follows both.

### Remoteness: equipment by port, installation by city

The one documented scheme I found is the one behind AEMO's planning inputs (Aurecon 2024,
section 11, derived from the Rawlinsons 2024 construction handbook). It splits a location
factor into two parts:

- **Equipment factor**: transport of plant and materials. It is 1.01 to 1.15, rising with
  distance. Aurecon measured the distance from the state capital. After consultation, AEMO
  re-based it on the nearest port in the final 2025 IASR (AEMO 2025).
- **Installation factor**: labour, materials, and mobilisation from the metropolitan area. It
  is about 1.0 to 1.1 in and near cities, up to about 1.5 for regional towns, and 1.5 to 1.75
  for the most remote towns: Broken Hill 1.53, Ceduna 1.66, Cooktown 1.65, Roxby Downs 1.68,
  Longreach 1.74. Forsayth (2.67) is an outlier.

That is the answer to "city or port": **a port only shortens the equipment journey, and the
workforce still comes from a city.** In Aurecon's table, remote coastal towns cost as much to
build in as remote inland ones. The overall factor is 1.35 at Cooktown and 1.30 at Ceduna, both
on the coast, against 1.26 at Broken Hill and 1.30 at Roxby Downs, both inland.

Method (port set decided 7 October: all ports):

- **Equipment multiplier** = 1 + a × (great-circle distance to the nearest port). It applies to
  the equipment part of every technology.
  - The ports are all 1,377 ports of the shipping network (green-porpoise `c_port_data.csv`: 46
    in Australia, 16 in Chile), in any country, so landlocked cells use a foreign port. The
    plant's equipment then lands where its ammonia would leave, on the assumption that an
    export project upgrades its port for heavy lifts.
  - The file is copied to `data/ports/verschuur2024_ports.csv` with its hash (done 7 October),
    not symlinked: its only other copy sits in `results/`, which gets archived.
  - Great-circle distance is the network's own measure for onshore pipelines, and Aurecon sets
    the equipment factor by approximate distance. The fit therefore uses the great-circle
    distance from each site to its state capital's port, and the coefficient absorbs the road
    detour.
- **Installation multiplier** = 1 + b × (travel time to the nearest city of at least N people),
  capped at a saturation time. It applies to the build part.
  - Travel times come from Nelson et al. (2019): 30 arc-second rasters on the Weiss et al.
    (2018) friction surface, which covers off-road travel.
  - The present fallback (port distance × 4 plus an elevation uplift) goes, unless the raster
    has gaps over cells we use.
- **Calibration**: a, b, N and the saturation time are fitted once to Aurecon's 77 sites, located
  with the GeoNames gazetteer. The fit and its residuals are reported before any run, and nothing
  is fitted to the LCOA map. If the final 2025 IASR publishes port-based factors by REZ, they
  replace Aurecon's capital-based equipment factors.
- Labour level stays 1 everywhere (no country wage factor); finance differences are S2's job.

Expected effect:

- **Installation term.** The present inputs add 25 % to the build part per 2,000 minutes (33
  hours) from a city of 50,000, about +3 % at 4 hours. Aurecon's installation factors are +10 to
  +75 % for regional and remote towns. S1 will therefore penalise remote sites, coastal or
  inland, far more than the current spatial inputs do.
- **Equipment term.** At its 1.15 ceiling it adds about 13 % to wind or electrolyser capex and
  about 5 % to PV capex.

Using Aurecon's factors globally is defensible if the paper says what they are:

- The structure (equipment by port, installation by city) holds anywhere.
- The magnitudes are Australian. Remote work there means fly-in fly-out crews and camps at high
  wages, and all 77 sites are in the NEM states (Queensland, New South Wales, Victoria, South
  Australia, Tasmania).
- Elsewhere the premium may be smaller (cheaper mobile labour) or similar (established mining
  regions such as the Atacama).
- The factors are relative to each country's base cost and carry no wage level, so they measure
  remoteness only.
- The paper states the calibration and reports, per cell, the share of LCOA that remoteness adds
  (from the site-cost table, design held fixed), so readers can judge its weight.

### Slope: a cost factor by slope class

There is no published per-degree slope multiplier for PV or wind in the sources I checked:
NREL's 2024 supply curves, Aurecon's 2024 review for AEMO and NREL's LandBOSSE model. Supply-curve
studies exclude land instead. In NREL's 2024
Reference scenario (Lopez et al.), PV is excluded above a 10 % grade (5.7°) and wind above 25 %
(14°); in the Limited scenario the limits are 5 % and 13 %. NREL derives slope from the 90 m
SRTM. The cost evidence:

- Nafa and Curcio (2024): a 100 MWp PV plant in Japan with rack slopes up to 15°, 58 % of them
  below 6°. Grading was 14 % of capex, +17 % over the ungraded design. Building without grading
  lost 2 to 4 % of yield instead.
- NREL 2024 (Table 12): transmission cost multipliers of 1.1 to 1.5 in hilly terrain and 1.2 to
  2.0 in mountainous terrain, across four US regions. These are the only published terrain
  multipliers for civil works I found.
- Aurecon (2024, wind section): terrain complexity (earthworks for road gradients, crane
  hardstands, hard rock) is causing significant overruns in wind balance-of-plant costs. The
  review gives no numbers.
- Ma et al. (2025): mountain PV in Yunnan, China, on a south-facing hill with flexible mounting
  made for mountain terrain. The two modelled arrays sit on average slopes of 17.7° and 20.4°.
  It shows utility PV built on such slopes but has no flat-ground comparison, so it gives no
  multiplier. It covers PV only.

Decided 7 October: the NREL factors, the flattest land first, and no land above 20° in this
1° model (finer models later can cost steep land instead).

| Slope class (GEBCO) | Factor on build part | Fixed PV capex | Wind capex | Basis |
|---|---|---|---|---|
| below 8.5° (15 % grade) | 1.0 | +0 % | +0 % | Your question: standard single-axis trackers are sold for north-south slopes up to 15 % grade, and wind roads can follow the contours at normal grades |
| 8.5 to 14° (15 to 25 % grade) | 1.3 | +20 % | +3 % | Middle of NREL's hilly range; same order as Nafa and Curcio's +17 %. 14° is NREL's Reference limit for wind |
| 14 to 20° (25 to 36 % grade) | 1.6 | +39 % | +7 % | Middle of NREL's mountainous range (extrapolation) |
| above 20° | excluded | | | Physical limit for this model |

- Why no uplift below 15 % grade:
  - Manufacturers quote standard single-axis trackers (Nextracker NX Horizon, Array DuraTrack)
    for north-south slopes up to 15 % without grading. I have this only from trade-press
    summaries, because the datasheet links are dead; a citable datasheet is needed before the
    paper.
  - Wind roads are the limit for turbines: gravel roads take 6 to 7 % for unassisted transport,
    9 to 10 % with a tractor unit and 15 % with a bulldozer (Miceli 2012, windfarmbop.com). On
    terrain below about 8.5°, roads can run along the contours at those grades.
  - NREL's 10 % PV limit (5.7°) is a siting screen, not a cost threshold.
- The factors apply to PV and wind only. The process plant needs a small pad, sited on the
  flattest ground.
- The plant takes the flattest usable land first. A cell's factor is the area-weighted mean
  over the land its share needs, taken in order of slope class, so most cells keep 1.0.
  - The land share now counts land up to 20°, not 15°.
  - The derived 2 % variants re-use the 20 % costs, so they keep the 20 % slope factor, an upper
    bound.
- Slope and installation factors multiply, as depth, remoteness and labour do now: earthworks
  in a remote place cost more too. A steep, remote cell can reach about 2.8 times the base build
  cost (1.75 × 1.6); the factor maps in step 3 will show how many cells do.
- Slope classes come from GEBCO (decided 7 October), the single file that also gives
  bathymetry, at 15 arc-seconds (about 460 m).
  - A coarse grid flattens slopes, so GEBCO understates steep land compared with the 90 m data
    behind NREL's thresholds. The classes are not adjusted for this; the paper notes it.
  - GEBCO is still finer than the ETOPO1 grid (1 arc-minute) on which Salmon and
    Bañares-Alcántara (2022) applied their 15° exclusion.

## 0b. Water: decisions of 7 October

The spatial water price (USD/m³) is a source price plus the pipeline from the nearest water
source to the cell, added to the LCOA after the solve at 1.5 m³ per tonne of ammonia. Decided:

- **Pipeline cost**: an official water-pipeline cost replaces the district-heating proxy (the DEA
  heat-transmission line, scaled through an equivalent thermal capacity of the water flow,
  2.8 USD/m³ per 100 km at a remoteness factor of 1). Candidates, to be proposed with values at
  the step-3 review:
  - the Texas Water Development Board's Uniform Costing Model, version 2.0 (HDR and Freese and
    Nichols, November 2018): planning-level unit costs
    of transmission pipelines by diameter, ground type and urbanisation, with pump stations,
    30 % engineering and contingency and O&M of 1 % (pipes) and 2.5 % (pumps) a year; sized
    for the plant's own flow;
  - Zhou and Tol (2005): 0.06 USD/m³ per 100 km horizontally and 0.05 USD/m³ per 100 m of lift,
    from large transfers; a check on the scale effect, since a 1 Mt/yr plant needs only
    1.5 million m³ a year.
  The installation part of the pipeline keeps the cell's build factor, which stands for terrain
  along the route. Friction pumping is part of any such transmission cost.
- **No pumping for the elevation gain**: lifting 1.5 m³ by 500 m takes about 3 kWh, about
  0.2 USD per tonne of ammonia.
- **Sources**: the sea and permanent lakes and reservoirs. Ephemeral and salt-pan lakes are not
  sources. With the present rule (any cell at least 10 % water in the 2022 land cover),
  Kati Thanda–Lake Eyre, a salt lake dry in most years, is the nearest source for 87 of
  Australia's 696 land cells, all in the centre, up to 420 km closer than the sea. **Decided 7
  October**: the Global Lakes and Wetlands Database v2 (Lehner et al. 2025; 15 arc-seconds;
  CC BY 4.0) decides which lakes count: classes 01 freshwater lake, 02 saline lake,
  03 reservoir and 06 other permanent waterbody are sources; 20-21 ephemeral and 32 salt pan are
  not. To check in step 3: which class the database gives Kati Thanda–Lake Eyre (if saline
  lake, permanence needs a further test).
  **Decided 9 October**: large rivers (GLWD 04 large river, 05 large estuarine river) are sources
  too, where they cover at least 1 % of the cell (a river is long and narrow, so the 10 % of
  standing water would leave the Amazon, Congo and Siberian rivers out).
- **Same source price for sea and lake**, 2 USD/m³ (coastal desalination). Lake water could be
  treated more cheaply, but charging the desalination price is a deliberate shadow price that
  protects freshwater for other uses. Stated in the code (the tech YAMLs and, from step 3,
  `model/site_costs.py`) and in the paper.
- **Geometry**: distances between 1° cell centres, and a cell that holds a source pays no
  pipeline, though a plant in it may sit 50 km from the water; a stated limitation of the 1°
  model.

## 1. Weather data layout (small)

**Done 7 October.** The focal cells load identical profiles from the new folder (SHA-256 of the
hourly frames unchanged). On ARC the nine files keep their inode numbers and sizes, and the old
names are links, so staged releases resolve the same files.

- Move the nine NetCDF files to `data/weather_data/salmon2022_pv_wind_2019_hourly_1deg/` and the
  ERA5 cutout link to `data/weather_data/cutouts/`. The nine files are `Solar`, `SolarTracking`
  and `WindPowers`, each split into three 120° longitude bands (no suffix: −180 to −61°; `1`:
  −60 to 59°; `2`: 60 to 179°), 8,760 hours of 2019 on a 1° grid.
- Name the source in the weather section of `data/README.md`. The files carry no metadata;
  they are the profiles used by Salmon and Bañares-Alcántara (2022), who state that their
  weather data come from ERA5. `data/README.md` is the tracked description, because git ignores
  everything else under `data/`.
- Make the weather directory a run setting: `weather_dir` in the new `run_defaults` block of
  `runs.yaml`, passed to the wrapper as `WEATHER_DIR`.
- Repoint the lcoa-opt symlinks (and through them lcoa_validation) to the new paths.
- On ARC, move the files to `green-lory/data/weather_data/salmon2022_pv_wind_2019_hourly_1deg/`,
  leaving links at the old paths so that existing releases still run.
- **Removes:** nothing else locally; the old paths go once the lcoa-opt links point to the new
  folder.
- **Check:** the three focal cells give identical profiles before and after.
- **Needs:** ARC write access (see "What I need from you").

## 2. Rationalise the code (behaviour unchanged)

**Done 7 October** (commits eb6cfd9 to 05362ab), waiting for your review.

- **Layout now:**
  - `model/`, 13 modules: `config`, `technology`, `finance`, `weather`, `plant`, `accounting`,
    `land`, `land_table`, `surface`, `runs`, `results`, `suppliers`, `cli`, and one plant bundle,
    `model/plants/ammonia_plant_1mt/`;
  - the network code (replay and comparison) moved to the green-porpoise repository on
    7 October, as `legacy_gpo/` (commits 0d71f8d and bc36b19 there), linked here untracked;
    the supplier export stays here as `model/suppliers.py`;
  - `arc/`, 3 modules: `release`, `manifest`, `qa`, plus the wrappers and `jobs/`;
  - 15 test files, one per module.
- **The `green-lory` command** (`pip install -e .`): `list`, `show`, `table`, `status`, `path`,
  `run <id> --cell=lat,lon` (a defined run at a few cells, locally), `map <id>`, `shell`.
  `run_store.csv` moved to the top level.
- **Check passed:**
  - the focal cells, one week and a full year, for the key run and the reconciling run;
  - the land build rerun for latitudes −24 to −22 (1,080 cells) against the accepted table,
    to 8e-11;
  - the two finance files that `runs.yaml` pins, regenerated byte for byte;
  - the derived 2 % surface and the two contracts of the key runs, regenerated identically
    by the old scripts and the new modules;
  - the network replay's sparse-against-Pyomo parity test;
  - the September focal-cell and coast/inland tables, reproduced by `model/results.py`;
  - `legacy_lcoa/` unchanged.
- **Changed from the plan:**
  - Run definitions are in `model/runs.py` and finance tables in `model/finance.py`, not in
    `config.py` and `technology.py`: each was too large to merge.
  - The ARC wrappers keep their names. One `submit.sh` for surfaces, land and networks is
    deferred to step 5, where the networks should also be submitted by id from `runs.yaml`.
  - New: `submit_lory_sequence.sh` and `stage_release.sh` no longer run Python on the login
    node. Every check and manifest runs in a short prepare or verification job, and the
    login node only calls `sbatch`. Tested with a stand-in `sbatch` on the Mac. The release
    verification job first runs when the step-3 release is staged, the prepare job at the
    first smoke run.
  - New: `python -m model.finance ameli` takes each cell's country as the surface does (the
    largest overlap of the 1° cell) and fills sea cells from the nearest land cell. For S2.
    The September file used the country at the cell centre and 5.1 % for sea cells; with the
    reduced series the two differ at 2,746 cells with land.
- **Still to go:**
  - `scripts/fetch_travel_time.py` moves into `model/site_costs.py` in step 3;
  - `reconciliation/` keeps `focal_cells.py`, `coast_inland.py`, `compare_land_builds.py` and
    `supply_pilot/` until step 6 (they made the evidence behind `DECISION.md`; the comparison
    notebook replaces them);
  - module count: 16 Python modules in `model/` and `arc/` now, plus 5 left in
    `reconciliation/` and `scripts/`.

Target: one library in `model/`, one network package, a thin ARC layer. That is about 14 Python
modules instead of the present 33 outside `tests/` and `legacy_lcoa/`, each with a module
docstring saying what it owns.

| New module | Contents | From |
|---|---|---|
| `model/config.py` | paths, constants, `runs.yaml` and run-store access | `data_paths`, `constants`, `arc/run_definitions` |
| `model/technology.py` | technology YAMLs, annuities, finance overrides, compiling a plant bundle | cost parts of `run_global`, notebook 00, `arc/build_finance_overrides`, `scripts/build_ameli_wacc_inputs` |
| `model/plant.py` | network build, constraints, solve, per-plant results | `main`, `auxiliary`, `data_store` |
| `model/weather.py` | profile loading and aggregation | `location_tools`, weather parts of `run_global` |
| `model/land_table.py` | building land tables from MODIS, WDPA, GEBCO | `land_processing`, `land_union` |
| `model/land.py` | land budget and capacity rules, finite-site supply curves | `land_capacity`, `reconciliation/supply_pilot` |
| `model/site_costs.py` | **every location uplift in one place**: remoteness, slope, water, depth, labour | notebook 02, `scripts/fetch_travel_time`, the archived altitude script |
| `model/surface.py` | the many-cell runner and result accounting | `run_global`, `result_accounting` |
| `model/results.py` | load results by run id, maps, focal-cell and coast/inland tables | `plot_global_heatmap`, `reconciliation/focal_cells`, `coast_inland`, `compare_land_builds` |
| `model/cli.py` | `green-lory show / run / status / path / site-costs` | new |
| `networks/replay.py`, `contracts.py`, `compare.py` | green-porpoise replay, supplier contracts, network comparison | `reconciliation/run_historical_network`, `sparse_network`, `network_heuristic`, `export_lory_surface`, `derive_land_share_surface`, `compare_networks`, `fetch_published_code` |
| `arc/` | `stage_release.sh`, `submit.sh` (surface, land, network by run id), `jobs/`, `qa.py` | the present wrappers, manifest and QA scripts |

Also in this step:
- Install `model/` as a package (`pyproject.toml`, `pip install -e .`).
- Take efficiencies and the load from the YAML at run time and keep one plant bundle.
- Point the technology YAMLs' source notes at `references/`. They still name
  `data/dea_reference/`, which no longer exists; the three DEA data sheets are now linked in
  `references/` from Zotero.
- Remove the dead options: the south-west anchor (except in its test), the conservative union
  fallback, the grid backstop, `aggregation_count`, the `max_*` capacity columns.
- Merge the tests into about eight files by module.
- `legacy_lcoa/` stays as it is.
- **Removes:**
  - every module in the "From" column once its code has moved;
  - `notebooks/00_tech_config.ipynb` and `scripts/` (absorbed into `model/technology.py` and
    `model/site_costs.py`);
  - `inputs/tech_config_ammonia_plant_2030_qld.yaml` and
    `inputs/archive/tech_config_ammonia_plant_2030_qld_original.yaml`, which nothing uses;
  - `inputs/keyruns_numerical_retry_cells.csv`, which only the archived run log mentions.

**Check:** a regression fixture must reproduce the accepted results: the three focal cells (one
week and a full year) for the key run and the reconciling run, a sample of land-table rows, and
the legacy harness at its focal cells. **Review** of the new layout before step 3.

## 3. Site costs and slope (new behaviour)

**Built 8 October; Review.** What exists (details in `inputs/site_costs/README.md` and the
cost-mechanisms table of `model/README.md`):

- `model/site_costs.py` (factors, the water price, the per-cell table, the overrides a run reads;
  `green-lory site-costs`), `model/site_geography.py` (rasters to per-cell and per-site tables,
  the fit), the equipment multiplier in `technology.apply_cell_overrides` and in the cost split,
  `site_costs_csv` as an optional run setting (`runs.yaml`, hash-pinned), the land tables' slope
  classes (`land_pct_slope_*`) with land up to 20 degrees. 16 new unit tests; the land-band
  regression still matches the accepted table in every existing column.
- Data: Nelson's travel-time rasters, GLWD v2, GeoNames, the network ports (`data/README.md`).
- Maps for Australia and Chile and the effect on the key run's designs:
  `results/campaigns/site_costs_review_20261008/` (local; from a local land build of 55 S to
  9 S, `results/campaigns/land_slope20_local_20261007/`).

Found on the way, and decided by the plan's rules:

- **A newer AEMO table.** AEMO's 2025 consultation report says the consultants' updated values
  base equipment factors on the nearest port. The newest table is GHD's 2025 review for the
  2026 cycle (draft Rev C, 28 November 2025, Table 8.3, 110 sites): equipment 1.00/1.05/1.10 by
  distance from a major port, installation from Rawlinsons' regional maps directly. Aurecon's
  Table 11-1 has 77 sites, not 75.
- **Kati Thanda-Lake Eyre** is GLWD class 02 (saline lake), so the permanence test of section 0b
  was needed: a saline lake counts only where the JRC Global Surface Water layer shows water at
  least half the time (1984-2021). Lake Eyre, Torrens, Gairdner and Frome, Makgadikgadi and
  Salinas Grandes fail; the Caspian, Aral, Balkhash, Issyk-Kul, Great Salt Lake, Turkana and the
  Tibetan lakes pass (109 of 134 cells). Without the test, 1,921 of the key runs' cells would
  move a median 244 km closer to water.
- **Water pipeline**: the Uniform Costing Model gives 1.80 USD2020/m3 per 100 km for one plant
  (12-inch pipe, least yearly cost), against 2.8 for the district-heating proxy, which used a
  10 % discount rate (1.57 at 5 %). Zhou and Tol's 0.06 is for transfers a hundred times larger.
- **Water price basis**: the source price is the YAML's baseline, 2 EUR2020/m3 in the campaign
  YAML (2.28 USD2020), the same as the flat runs' price (your EUR basis of 23 September), not
  the 2 USD of section 0b.

To decide at this review:

1. **Which table calibrates remoteness.** Both fits choose the same form (travel time to a city
   of 20,000, saturating at 4 hours) but differ in size: installation up to 1.34 (GHD 2025) or
   1.80 (Aurecon 2024). With the key run's designs held fixed the site costs add a median
   60 EUR/t (+16 %) with GHD, 104 EUR/t (+27 %) with Aurecon; Australia +18 % or +31 %, Chile
   +10 % or +20 %. Recommended: GHD 2025, the newer table, with AEMO's port basis and without
   Aurecon's 2.5-fold residual; it is a draft, so the paper should cite it as such and give
   Aurecon's as the upper bound. **Decided 8 October: GHD 2025, fitted as its sites give it**
   (installation rising to 1.34 at 4 hours from a town of 20,000, then flat). The plateau rests
   on 5 of 110 sites, the farthest 6.7 hours out, while 41 % of the key runs' cells (56 % in
   Australia) lie beyond that; a concave curve, 1 + c ln(1 + t / 2 h), fits as well (leave-one-
   out RMSE 0.071 against 0.070) and would give 1.49 at 12 hours and 1.65 at 24 hours, a
   possible sensitivity.
2. **The slope factor is nearly inactive.** On GEBCO's 15-arc-second grid 94 % of the land from
   55 S to 9 S lies below 8.5 degrees, so with the 20 % share taken flattest first only 6 of
   2,806 cells pay a slope factor (at the full share 1,674 cells, up to 1.54). The slope decision
   then acts through land up to 20 degrees, not through cost. No change proposed: the classes
   were fixed before this was known, and the paper can say so.
3. **The water source price** at 2 EUR2020/m3 (as the flat runs) rather than 2 USD.

Offshore and coastal cells (review of 8 October). No campaign run places a plant at sea (the
choice of 23 September stands: offshore cells stay out until their costs and exclusions exist),
and the plant's capacity uses onshore land only. Three things reach the onshore runs anyway:

- **Mostly-sea cells take a marine wind resource.** The 1-degree profiles behave like averages
  over the whole cell: at equal sea share, cells centred at sea and on land have the same wind,
  and it rises with the sea share. Against fully inland neighbours, the key run's LCOA is lower
  by 0.1 % (cells up to 25 % sea), 2.5 % (25-50 %), 6.3 % (50-75 %) and 10.4 % (over 75 %), the
  wind capacity factor higher by up to 0.23. 2,360 of the 16,509 cells are more than half sea;
  they hold only 1.6 % of the capacity at 20 %, but 175 of the 265 cheapest cells that would meet
  600 Mt/yr (32 Mt/yr between them). The plant sits on the land part and is costed as onshore.
  **Fixed (decided 8 October):** the profiles of cells with at least 1 % water are rescaled to
  the cell's land, by month and hour of day, with the ratio of the land-weighted to the
  area-weighted mean of the cell's ERA5 capacity factors at 0.25 degrees (`model/weather_land.py`;
  the legacy stack stays the source, D7). Runs opt in with `weather_land_correction` in
  `runs.yaml`; the September runs keep the profiles as they were.
- **Protected areas count their water against land.** WDPA areas of the coastal realm (part
  land, part sea) are kept, and their whole area is divided by the cell's land: in 407 cells the
  protected share exceeds the land share, and 519 coastal cells (281 between 50 S and 50 N; 25 in
  Australia, 28 in Chile, 18 in Arabia) lose all their land, some rightly (south-west Tasmania,
  K'gari) and some because of sea. **Fixed (decided 8 October, commit afecb85):** protected land
  is measured on the land cover's grid (WDPA rasterised at 0.005 degrees, each 0.05-degree MODIS
  pixel weighted by its land fraction). Inland cells move by at most 0.3 percentage points of
  protection; between 55 S and 9 S the 69 coastal cells that had lost all their land keep some,
  and coastal suitable land grows 2.8 % in Australia and 2.9 % in Chile.
- **The wind stack's turbine is undocumented** and the stack comes from Salmon and
  Banares-Alcantara's offshore study; ERA5 with an onshore turbine gave 26-38 % less wind at the
  focal cells (D7, deferred).

What exists for a later offshore implementation: profiles at sea, the land tables' offshore
columns (MODIS water, lakes included, no exclusions), GEBCO depths, the port set, DEA's offshore
turbine sheet, and the depth, port-distance and weather-window multipliers of the removed
notebook 02 (git history before 73689c6; placeholders, not sourced). Missing: offshore wind and
foundation costs by depth, where conversion happens (platform or shore) and the export link,
exclusions (depth, shipping lanes, marine protected areas, ice), and an offshore power curve.

Done and still to do in this step (8 October):

- **Land table, October build** (release `20261008-land-slope20-v2`, commit afecb85; jobs 13441136
  in 24 minutes, 13441137-8): `green-lory-campaigns/land_center_slope20_20261008_v2/`, local copy
  under `results/campaigns/`. Against September: land-cover columns identical; inland protection
  within 0.31 points; coastal cells wholly excluded by protection 573 to 26; land up to the slope
  limit 95.3 % to 97.4 % of land; suitable onshore area +2.2 to +2.7 %; cells with suitable land
  18,773 to 19,387. The run cells (`locations_onshore_suitable_center_20261008.csv`, same rule as
  September's 16,509, `land_table.run_cells`): 16,885, none dropped. The first build (release
  `20261008-land-slope20-v1`, jobs 13434202-4, old protected-area rule) was cancelled after 4 h
  at your request; its folder holds a CANCELLED.md note.
- **Site-cost table of the 20 % table** (GHD 2025): `inputs/site_costs/site_costs_center_20pct_slope20.csv`,
  every run cell covered. In half the run cells the equipment and installation factors are at
  their caps (1.10, 1.34); 27 cells pay a slope factor (at most 1.095); water 4.5 USD/m3 median
  (90th percentile 12.1) with the rivers of 9 October, 5.5 (15.6) before. World maps:
  `results/campaigns/site_costs_review_20261008/maps/`; the water map with rivers in
  `results/campaigns/site_costs_review_20261009/maps/`.
- **Weather land correction**: the first factors (job 13441218, release
  `20261008-weather-land-v1`) corrected PV wrongly, because the ERA5 stores hold no PV at sea,
  and nearly removed the wind of some steep coasts (Atacama: ERA5 land capacity factor 0.003);
  superseded (SUPERSEDED.md in its folder, the file removed from the weather folder). The
  corrected version (wind only, calm bins, inland floor; commit 1bd8d0f; job 13442443, release
  `20261008-weather-land-v2`) is `data/weather_data/land_correction/weather_land_correction_2019.nc`
  (on ARC too). It cuts the wind excess of coastal cells over their fully inland neighbours by
  about two thirds (capacity factor above inland: +0.053 to +0.013 at 25-50 % sea, +0.148 to
  +0.049 at 50-75 %, +0.231 to +0.075 above 75 %); the floor binds in 1,120 run cells (the
  Atacama coast keeps its hinterland's wind instead of ERA5's near-zero). Key-run LCOA, full year:
  Gladstone coast (23 S 151 E) 321 to 357 EUR/t, Perth coast 313 to 327, Ningaloo 328 to 331,
  inland Pilbara unchanged. Campaign runs set `weather_land_correction` to this file.
- **Rivers as water sources, decided 9 October** (section 0b): large rivers covering at least
  1 % of a cell are sources, at the same shadow price. 1,082 more cells hold a source (38,457 in
  all); 5,065 of the 16,885 run cells get cheaper water and none dearer. Median distance to water:
  Amazon 556 to 111 km, Congo 398 to 157, Siberia 229 to 94, Sahara 667 to 556 (the Nile);
  Australia unchanged, Chile 111 to 106. Water over the run cells: median 5.54 to 4.48 USD2020/m3;
  at 1.5 m3/t the cells that gain save 6 USD/t of ammonia at the median, 41 at most.
  `cell_geography.csv` keeps the distance without rivers (`water_source_distance_km_without_rivers`).
- Done for runs with site costs (commit d99ad1c): the `site_costs_<finance>_no_land_rent` cost
  scope in the manifest, the QA checks per cell, the wrapper and shard job passing the table.

- `model/site_costs.py` produces one table per land build. For each cell and technology it gives
  the equipment multiplier (port), the build multiplier (installation x slope x depth x labour)
  and the water price. Each factor is a named, documented function with its source;
  `green-lory site-costs --land <table>` writes it.
- The plant applies the equipment multiplier to `tech_cost_per_mw`, next to the existing build
  multiplier on `build_cost_per_mw`.
- Fit the remoteness factors to Aurecon's 77 sites (section 0a) with a small script. Keep the
  site table, its source and the fit report in `inputs/`.
- New inputs, each with a recorded hash in `data/README.md`:
  - Nelson's city travel-time rasters;
  - the network port set, copied to `data/ports/`;
  - the GeoNames gazetteer for Australia, to locate the 77 sites.
- The land build computes the land area in each slope class per cell from GEBCO, as it now
  computes the > 15° share. The slope ban goes; the class factors of D2 replace it.
- Water follows D4 and section 0b. I propose the pipeline values with references for you to
  approve; `model/site_costs.py` carries the shadow-price comment of the source price.
- Write the "cost mechanisms" table in the model documentation: mechanism, module and function,
  input, reference.
- **Check:** unit tests per factor; maps of each factor for Australia and Chile. **Review.**
- **Removes:**
  - `notebooks/02_spatial_cost_inputs.ipynb`;
  - the three `inputs/spatial_cost_inputs_*.csv` variants, replaced by the generated site-cost
    tables (`inputs/spatial_cost_inputs.csv` itself goes in step 6, with
    `reconciliation/coast_inland.py`, which reads it);
  - the elevation uplift and the port-distance fallback.

## 4. Validation (small runs, before ARC time)

- The D9 literature check: compressor cost, storage basis, Haber-Bosch flexibility (30 % minimum
  load, ramps of 40 % of capacity per hour, the legacy model's values); any change goes into the
  YAML with its source. **Done and decided 14 October** (decisions at the end of this item).
  Each candidate value was solved locally with C2's settings on six cells:
  three inland (Chile 23 S 69 W, Pilbara 23 S 117 E, central Australia 21 S 135 E) and three on
  the coast (Gladstone 23 S 151 E, Perth 32 S 115 E, Ningaloo 22 S 114 E).
  - **Two basis errors in `inputs/tech/tech_config_ammonia_plant_2050_dea.yaml`.**
    - Hydrogen tank: DEA sheet 151a gives costs per MWh of hydrogen on LHV (33.3 kWh/kg), but
      the store's bus is on HHV (39.4 kWh/kg). 15,300 becomes 12,900 EUR/MWh.
    - Haber-Bosch: DEA sheet 5.4 gives costs per MW of ammonia on LHV (5.24 MWh/t), but the
      ammonia bus is on HHV (6.25 MWh/t). The ASU mark-up is DEA's 1.09; the YAML's 1.106 was
      1.09 times the 2030 YAML's rounding of 1.380 to 1.4. 957,000 becomes 790,600 EUR/MW.
      Sheet 5.4 says its costs are in 2015 EUR, but that note predates the catalogue's update to
      2020 prices: every value on the sheet is a 2015 value times 1.0634. So no price-year
      change is needed.
    - Together the two fixes lower LCOA by 14 to 16 EUR/t, nearly the same in every cell.
  - **The ammonia tank misconverts its source.** 521.74 EUR/MWh (3,261 EUR/t) comes from
    Fletcher et al. (2023, Queensland green ammonia value chain, p. 35 and Table 54): "A value of
    3000 AUD/t is assumed in this study". The report gives a range of 1,500 to 6,000 AUD/t, and
    the 3,000 is an installed cost. The old conversion made three errors: it read AUD as USD,
    divided by 6.25 MWh/t, and then added a 25 % build mark-up. The correct conversion is
    3,000 AUD/t x 0.7 USD/AUD (the report's rate) x 258.811/292.655 (US CPI, 2022 to 2020)
    / 1.142196 USD/EUR = 1,626 EUR2020/t, or 260 EUR/MWh. The report's own range gives a
    sensitivity of 130 to 520 EUR/MWh. With the basis fixes, LCOA falls by 16 to 26 EUR/t, and
    tanks grow by 2 to 6 % (5 to 27 days of output).
  - **Haber-Bosch flexibility.**
    - The 30 % minimum load is an assumption in Fletcher et al. (2023, p. 35), with no source.
      DEA sheet 5.4, which already supplies the model's Haber-Bosch cost, gives 20 % for 2050
      (range 10 to 30 %) and notes that conventional loops reach 30 % without changes. Moving to
      20 % changes nothing at the inland solar cells and lowers LCOA by 2 to 7 EUR/t elsewhere,
      most on the windy coasts.
    - No source gives a ramp of 40 %/h; the legacy model applied 0.4 per four-hour step. At
      hourly steps the ramp limit never binds: the results are identical without it, at both 30 %
      and 20 % minimum load.
  - **The compressor's cost and electricity assume different inlet pressures.**
    - 165,300 EUR/MW is DEA 151a's compressor, its 2030 value scaled by DEA's 2050/2030 ratio.
      DEA's own 2050 figure, 191.4 EUR/kW of output on LHV, is 161.8 on HHV, within 2 % of the
      YAML. It describes a five-stage compressor from about 1 bar to 200 bar, which uses 9.1 % of
      the hydrogen's HHV (DEA's 90.2 % round trip).
    - The YAML's 5 % electricity is instead Fletcher et al.'s 2.016 kWh/kg, which suits a
      pressurised electrolyser outlet. Our DEA electrolysis sheet does not state an outlet
      pressure.
    - Atmospheric outlet (DEA 151a's cost and electricity together): +0.1 to +0.9 EUR/t.
    - 30-bar outlet (about 95 EUR/kW and 4 %): -10 EUR/t at the inland solar cells and -1 to
      -6 at the coasts. The 95 scales the compression cost in DEA's newer electrolysis sheet
      (30 to 80 bar, August 2026) to 250 bar, so this option would need that sheet to replace
      ours.
    - The compressor cost is the strongest single lever: doubling it raises the inland solar
      cells by 19 to 22 EUR/t; halving it lowers them by 11.

  | LCOA, EUR/t | Chile | Pilbara | central Aus. | Gladstone | Perth | Ningaloo | coast - inland |
  |---|---|---|---|---|---|---|---|
  | C2 as run | 324 | 347 | 284 | 352 | 322 | 326 | 15.2 |
  | + basis fixes | 309 | 331 | 270 | 337 | 306 | 311 | 14.8 |
  | + tank 260 EUR/MWh | 301 | 324 | 268 | 328 | 296 | 300 | 10.6 |
  | + minimum load 20 % | 301 | 324 | 266 | 327 | 289 | 297 | 7.4 |
  | + compressor from 1 bar (DEA pair) | 302 | 324 | 266 | 327 | 289 | 297 | 7.0 |
  | or + compressor from 30 bar | 291 | 314 | 264 | 321 | 287 | 294 | 11.2 |

  - Sources for these values: the DEA sheets and Fletcher et al. (2023), both already in Zotero.
    The other literature consulted only served to check that these values are not outliers.
  - **Decided 14 October**, in `inputs/tech/tech_config_ammonia_plant_2050_dea.yaml` (so every
    `gl_*` run):
    - the two basis fixes;
    - the tank at 260 EUR/MWh;
    - DEA 151a's compressor with its own electricity use: 161,800 EUR/MW, all equipment
      (the sheet's installation row is already in the store), and 0.0914 MWh per MWh;
    - minimum load 20 % (DEA);
    - ramps kept, at 20 % of capacity per hour. This is the manufacturers' figure in Armijo and
      Philibert (2020, p. 1550, from interviews); the IEA's Ammonia Technology Roadmap (2021,
      p. 102) uses the same. Both are already in Zotero.

    The key run's regression rows were regenerated: full-year LCOA -21.7 (Chile), -22.2
    (Pilbara) and -18.7 (central Australia) EUR/t. The ramp limit barely binds: these match
    the six-cell results without it. The Way YAMLs of the replication runs keep the published
    values, and C2's validation run of 12 October predates the change.
- 4 h against 1 h on Australia's 565 eligible cells and the 563-cell legacy sample, both with the
  improved plant, to measure the resolution effect before the global runs. **Done 14 October** on
  ARC (release `20261012-validation-v1`, campaign `validation_step4_20261012_v1`, htc jobs
  9082565-75, both QA passed; smoke and diagnostic stages passed on 12 October): C1 (4 h) against
  C2 (1 h) on 1,317 cells (`inputs/cells/validation_cells_step4.csv`: Australia's 786 run cells,
  562 cells of the legacy sample). The 4 h step lowers LCOA in every cell, by 0.2 % to 5.1 %
  (median 2.1 %, mean 8.5 EUR/t), and keeps the ranking (Spearman 0.994). It raises the supply
  on the cells' land by 11 % in all (median 2.3 %, 95th percentile 47 %): with the variability
  averaged out the plant builds less wind (median share of capacity 0.23 to 0.17) and less
  battery (median 20 %), and where wind land was binding more capacity fits. At 500 Mt/yr of the
  sample's supply the 4 h curve is 8.4 EUR/t (2.7 %) lower. Results and the comparison in
  `results/campaigns/validation_step4_20261012_v1/` (`comparison/`).
  How the 1,317 cells were chosen (`inputs/cells/validation_cells_step4.csv`, column `group`):
  every one of Australia's 786 run cells in the October land table (cells whose country in
  `inputs/cells/cell_countries.csv` is Australia), to see the effect where it matters most for the
  networks; and the 563-cell legacy sample of 15 September 2026 (`legacy_lcoa/cells_sample_v1.csv`),
  the three focal cells plus 80 cells drawn from each 10-degree band of absolute latitude from 0 to
  70 degrees (both hemispheres pooled) among the 15,377 cells of the published supplier table, to
  cover all climates evenly; 562 of the 563 have suitable land in the October table, and 31 cells
  are in both groups. The legacy sample weighs each latitude band equally, not by area or supply,
  and has no cell poleward of 70 degrees; the script that drew it was not kept, the file records
  the cells. The results below hold for each group separately (Australia: LCOA -1.9 % at the
  median; legacy sample: -2.5 %).
  **Decision (14 October): C1 is dropped from the global campaign for now**; the four-hour effect
  is the one measured here, on the sample, and the hourly run (C2) is the base.
- PV profiles above 1.0 (AC/DC basis). Checked 12 October: over the 16,885 run cells, output above
  1.0 is 0.006 % of fixed-tilt PV energy (0.05 % for tracking), at most 1.8 % in one cell of the
  high Andes (28 S, 69 W; tracking 4.9 %): negligible for the fixed-tilt campaign, no clipping
  proposed.
- **Review:** the validation note, before the campaign.

## 5. Campaign

All runs:
- DEA 2050 costs, fixed-tilt PV, co-located land, 2019 weather.
- Land applied after the solve, capacity from the scaled design.
- The new land table (no slope ban), onshore cells with suitable land, 20 % share with 2 % derived.

| Run | Step | Build | Water | Finance | Network |
|---|---|---|---|---|---|
| C1 flat, 4 h: dropped for now (14 Oct); step 4 measured the four-hour effect on 1,317 cells | 4 h | flat | flat 2 EUR/m³ | uniform 5 % | no |
| C2 flat, 1 h (base): `gl_dea2050_wacc5_bflat_wflat_land20oct_fixed` | 1 h | flat | flat | uniform 5 % | yes |
| S1 spatial: `gl_dea2050_wacc5_bspat_wspat_land20oct_fixed` | 1 h | equipment by port; installation by city x slope (x depth, x labour = 1) | spatial | uniform 5 % | yes |
| F finance only: `gl_dea2050_amelilc_bflat_wflat_land20oct_fixed` (added 14 Oct) | 1 h | flat | flat | Ameli low-carbon by country | no |
| S2 spatial + finance: `gl_dea2050_amelilc_bspat_wspat_land20oct_fixed` | 1 h | as S1 | spatial | Ameli low-carbon by country (decided 14 Oct, until the collaborator's table) | yes |
| Build only, water only | derived from S1 and C2 | as S1 / flat | flat / spatial | uniform 5 % | no |
| 2 % variants | derived | | | | C2, S2 |

- **Design (14 October):** the paper (green-serval) asks what spatially varying water, build
  and finance costs do to the global cost of green ammonia. C2, S1, F and S2 give each factor
  alone and together. Water enters the LCOA after the solve, as a cost per tonne, so S1 also
  gives build only (S1 with C2's water) and water only (C2 with S1's water) without another
  solve. F against S2 shows how finance and site costs interact.
- **Order:** new land build (done 8 October) → C2, S1, F, S2, each after your go-ahead →
  supplier contracts → networks at the 1 % tolerance, run from a green-porpoise release
  (`legacy_gpo/arc/`).
- **Compute:** 1.5-5 h on each of four 48-core jobs per hourly surface of about 17,000 cells
  (the global stage of 15 October), and 1-2 h per network.
- Every run is defined in `runs.yaml` with a run-store row before submission.
- **Check:** QA gates as now; the coast/inland and focal-cell tables for each run.

## 6. Results, notebooks and documents

- Notebooks become readers:
  - `results.ipynb`: maps and tables by run id, the coast/inland comparison, delivered-cost
    supply curves.
  - `legacy_vs_green_lory.ipynb`: what the reconciliation showed and why the improved model
    differs.
- **Removes:** `reconciliation/` (git history keeps it), `inputs/spatial_cost_inputs.csv`, this
  plan and the cleanup plan (archived).
- Documents:
  - `README.md`: quickstart: install, run three cells locally in minutes, run on ARC.
  - `MODEL.md`: the model description, absorbing `MODEL_CHOICES.md`, with the cost-mechanisms
    table.
  - `GLOSSARY.md` and `data/README.md` updated.
- **End state:** one documented library, one CLI, run definitions as data, notebooks that only
  read results.

## Decisions and access

- **D2, D3, D4 and D7** were decided on 7 October (table above, sections 0a and 0b). D1, D5,
  D6 and D8 stand as recommended, following your approval of the campaign on 6 October; D9 is
  the step-4 literature check.
- **Signed off on 7 October:** the step-2 layout and all of step 3 as planned, with the slope
  classes (no uplift below 8.5 degrees, 1.3 for 8.5-14, 1.6 for 14-20, land above 20 excluded,
  flattest land first) and the global land build on ARC.
- **9 October:** `inputs/` sorted by topic (`tech/`, `finance/`, `site_costs/`, `weather/`,
  `cells/`) with an index, `inputs/README.md`; every modelling assumption other than technology
  costs moved from the code to `inputs/parameters.yaml`, each with its source (values unchanged:
  the geography and site-cost tables regenerate byte for byte). A uniform rate is now one number
  in `runs.yaml` (`interest_rate`), not a 22 MB table (the uniform key run reproduces its
  regression cells); the countries of each TIAM-UCL region are in `inputs/finance/tiam_regions.csv`;
  the per-cell finance tables are generated and gitignored.
- **12 October:** the plant bundle (`model/plants/ammonia_plant_1mt/`) holds only the plant's
  structure; every technology value, including the Haber-Bosch minimum load and ramp limits and
  the battery standing loss, is in the technology YAMLs. The battery standing loss was wrong: the
  bundle's 0.001 was applied per hour (2.4 % a day), the YAML's 0.1 % a day (DEA 2050) was never
  read; with the fix the focal cells' full-year LCOA falls by 0.006 % to 0.05 % (Chile 328.42 to
  328.26 EUR/t in the key run). The electrolysis variable cost of 0.537 EUR/MWh, in the bundle
  since January 2026 with no source (DEA gives none, the legacy model had none), is dropped at
  your request: full-year LCOA falls by 4.53 EUR/t in the key run (1.3-1.6 %) and 4.3 EUR/t in
  the reconciling run. Finance: interest rates come only from the run, one rate or a finance table
  by country (`iso3`) or by cell, per technology or `all`; every technology at every cell run must
  get a rate (`inputs/README.md`, finance). Country tables reach the cells through
  `inputs/cells/cell_countries.csv`, which also gives the results' countries. The YAMLs' own
  rates (10 %) and `model/finance.py` with the TIAM-UCL region tables are gone; Ameli's
  low-carbon and reduced series are kept as country tables in `inputs/finance/`.
- **14 October:** decided in green-porpoise (the network project, `legacy_gpo/`): the campaign's
  networks use the corrected equations (`--corrected`) and keep the published week of storage at
  each supplier (0.99 USD/t with the corrected unit, the same for every supplier, so no change to
  flows; it partly overlaps the plant's own ammonia tank, which the paper states). C1 is dropped
  from the global campaign (section 4). The step-4 literature check (D9) started.
- **14 October, later:** D9 decided (section 4) and the campaign approved: "Go ahead commit and
  submit all" (C2, S1, F and S2, section 5). That go-ahead also settles the two step-3 points of
  section 3 as proposed: no change to the nearly inactive slope factor, and the water source
  price at 2 EUR2020/m3 at sea, lakes and rivers. S2's interest rates are Ameli's low-carbon
  series by country for now (`inputs/finance/ameli2021_lowcarbon_wacc_by_country.csv`); the
  collaborator's technology-specific rates replace them when their table arrives.
- **15 October:** water use 1.62 m3 per tonne of ammonia instead of 1.5, which had no source and
  was below the stoichiometric 1.59 m3/t (`inputs/tech/tech_config_ammonia_plant_2050_dea.yaml`):
  DEA's 2050 electrolysers use 9.0 kg of water per kg of hydrogen (IEA 2019: about 9 litres),
  at 0.18 t of hydrogen per tonne; air-cooled, treatment losses not counted. The site-cost table
  was regenerated for the larger flow (pipeline 1.73 instead of 1.80 USD2020/m3 per 100 km);
  water per tonne rises by 0.24 to 1.65 EUR/t (median 0.35) over the run cells. The global runs
  submitted on 15 October used 1.5; water is added after the solve, so their LCOA is recomputed
  exactly.
- **28 October:** the ammonia tank is 80.3 EUR/MWh (502 EUR2020/t) instead of 260, aligned
  with green-caribou: Cesaro (2022, Table 5.2), 0.56 USD2019/kg, the low end of the
  0.561-1.062 USD2019/kg in Nayak-Luke et al. (2021); Fletcher et al.'s 3,000 AUD/t is an
  assumption at the top of the literature. The campaign of 15 October (C2, S1, F, S2) used 260,
  so its runs no longer match their definitions; they are rerun with the collaborator's finance
  rates. From the campaign results, the median LCOA falls by 6 to 11 EUR/t across the four runs
  (bounded by the same design with the cheaper tank and by a free tank, which local re-solves
  of 310 cells of C2 showed to save the tank's cost plus a median 0.2 EUR/t, at most 4.5).
  Key-run LCOA at the focal cells: -5.5 (Chile), -5.6 (Pilbara), -1.8 EUR/t (central
  Australia).
- **29 October:** the inputs shared with green-caribou aligned, all in EUR2020 (green-caribou
  had moved its ammonia tank to the IEA's import-terminal tank on 28 October):
  - ammonia tank 250.4 EUR/MWh (IEA 2019 assumptions annex, p. 7: USD2017 97 million for
    56,700 t, 1,565 EUR2020/t), 4 % O&M;
  - electrolyser efficiency 0.828 on HHV (39.4 / 47.61 kWh/kg; DEA's note G adds a row on the
    stack-level consumption) and fixed O&M 4 % (DEA row 36, which excludes stack replacement);
  - battery: DEA's other project costs all on the energy store (note M), the PCS at 72 EUR/kW;
  - hydrogen tank 12,133 EUR/MWh and compressor 152.1 EUR/kW on HHV, 6 % lower: DEA sheet
    151a's 2030 and 2050 component rows carry the 2020 price update (x 1.0634) twice, its totals
    and O&M once.

  Key-run LCOA at the focal cells: +27.9 (Chile), +23.4 (Pilbara), +13.5 EUR/t (central
  Australia). The 28 October bounds no longer apply; C2, S1, F and S2 are rerun.
- **30 October:** the hydrogen compressor's lifetime is 25 years and its fixed O&M 1.2 %, the
  compressor row of DEA's component table (151a text, Table 7, p. 107: 6,000 EUR a year on a
  500,000 EUR compressor), as in green-caribou since 29 October. The sheet's 30 years is the
  lifetime of the whole storage system; the 20 years and 2 % used until now came from the 2030
  YAML without a source. Key-run LCOA at the focal cells: -3.5 (Chile), -3.2 (Pilbara),
  -0.5 EUR/t (central Australia). green-caribou's DEA price-year corrections of 29 October
  need nothing more here: green-lory's Haber-Bosch cost is DEA's 0.18889 MEUR2020 per t/day
  with only DEA's ASU mark-up (no currency factor), its hydrogen tank and compressor were
  corrected on 29 October, and it uses neither DEA's ammonia carrier sheet nor PyPSA
  technology-data. Still open: compression ahead of Haber-Bosch
  (MODEL_CHOICES.md, inputs shared with green-caribou).
- **ARC access:** the control socket closed on 7 October in the evening and was open again on
  8 October, when the global land build was submitted.
- **ARC tooling checked on 7 October** with a smoke stage of the key run from release
  `20261007-shakedown-v1`: release verification job 13396884, prepare jobs 13396930 (dry run) and
  13396965, shard 13396992 (7.5 min) and merge/QA 13396993 (passed). The three cells match the
  regression baseline to 1.2e-5 (LCOA), so the step-2 code and the new prepare-job flow work on
  ARC. Outputs: `green-lory-campaigns/shakedown_20261007_v1/` (local copy under `results/campaigns/`).
- **Still needed:** a go-ahead before each global run, and your review at the gates marked
  **Review** (after steps 2, 3 and 4).

## References for section 0a

All in Zotero (7 October), with the PDFs linked in `references/` (gitignored), except where
noted below.

- AEMO (2025). *2025 IASR Consultation Summary Report*, July 2025, published with the 2025 IASR.
  AEMO accepts basing the equipment cost factors on distance to ports rather than capital
  cities (search the PDF for "nearest port").
  https://www.aemo.com.au/-/media/files/stakeholder_consultation/consultations/nem-consultations/2024/2025-iasr-scenarios/final-docs/2025-iasr-consultation-summary-report.pdf
- Aurecon (Halder, P.) (2024). *2024 Energy Technology Costs and Technical Parameter Review*,
  revision 3, 6 December 2024, for AEMO; section 11 and Table 11-1 (location factors from
  Rawlinsons 2024). Zotero key `prabirhalder2024EnergyTechnology2024`.
  https://aemo.com.au/-/media/files/major-publications/isp/2025/aurecon-2024-energy-technology-costs-and-technical-parameter-review.pdf
- GHD (2025). *2025 Energy Technology Cost and Technical Parameter Review*, draft report Rev C,
  28 November 2025, for AEMO (draft 2026 Forecasting Assumptions Update); section 8 and Tables
  8.1 to 8.3 (location factors from Rawlinsons 2025). Zotero key `ghd2025EnergyTechnology2025`.
  https://www.aemo.com.au/-/media/files/stakeholder_consultation/consultations/nem-consultations/2025/draft-2026-fau/ghd-2025-energy-technology-cost-and-technical-parameter-report.pdf
- Lopez, A. et al. (2025). *Renewable Energy Technical Potential and Supply Curves for the
  Contiguous United States: 2024 Edition*, NREL/TP-6A20-91900, revised June 2025; slope
  exclusions in Tables 7 (wind) and 9 (PV), transmission cost multipliers in Table 12.
- Ma, J. et al. (2025). Simulation study of a 386.4 MW mountain photovoltaic power plant: a
  case study. Scientific Reports 16, 1000. https://doi.org/10.1038/s41598-025-30742-7
- Miceli, F. (2012). Maximum wind farm internal road gradient. Wind Farm BoP blog, 23 May
  2012. https://www.windfarmbop.com/?p=268 (not yet in Zotero).
- Nextracker. NX Horizon datasheet: 15 % north-south slope tolerance. Not yet citable; the
  datasheet links found on 7 October return 404.
- Nafa, A. and Curcio, M. (2024). Optimizing PV solar deployment on hilly terrain:
  techno-economic analysis of grading and dynamic mismatch losses. Proc. IEICES 10, 275-281.
  https://doi.org/10.5109/7323274
- Nelson, A. et al. (2019). A suite of global accessibility indicators. Scientific Data 6, 266.
  https://doi.org/10.1038/s41597-019-0265-5
- Pekel, J.-F., Cottam, A., Gorelick, N. and Belward, A. S. (2016). High-resolution mapping of
  global surface water and its long-term changes. Nature 540, 418-422.
  https://doi.org/10.1038/nature20584 (Global Surface Water v1.4, 1984-2021).
- Salmon, N. and Bañares-Alcántara, R. (2022). A global, spatially granular techno-economic
  analysis of offshore green ammonia production. Journal of Cleaner Production 367, 133045.
  https://doi.org/10.1016/j.jclepro.2022.133045
- Verschuur, J. et al. (2024). Optimal fuel supply of green ammonia to decarbonise global
  shipping. Environmental Research: Infrastructure and Sustainability 4, 015001. The network
  model and its port set.
- HDR and Freese and Nichols (2018). *Uniform Costing Model User's Guide*, version 2.0, for the
  Texas Water Development Board.
  https://www.twdb.texas.gov/publications/reports/contracted_reports/doc/1800012227_UCM2018_User_Guide.pdf
- Lehner, B. et al. (2025). Mapping the world's inland surface waters: an upgrade to the Global
  Lakes and Wetlands Database (GLWD v2). Earth System Science Data 17, 2277.
  https://doi.org/10.5194/essd-17-2277-2025 (data: https://doi.org/10.6084/m9.figshare.28519994).
- Weiss, D. J. et al. (2018). A global map of travel time to cities to assess inequalities in
  accessibility in 2015. Nature 553, 333-336. https://doi.org/10.1038/nature25181
- Zhou, Y. and Tol, R. S. J. (2005). Evaluating the costs of desalination and water transport.
  Water Resources Research 41, W03003. https://doi.org/10.1029/2004WR003749
