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
  supplier tables for the network model → networks at the 1 % tolerance, run from a green-porpoise release
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
- **For you, before publication:** a courtesy email to info@marineregions.org saying that the
  World EEZ v12 layer, taken from Marine Regions' public feed on 30 October 2026, gives the
  country of the offshore cells. The feed's service metadata asks users of a layer to contact
  VLIZ (`data/README.md`, Maritime zones).

## 7. Offshore sites, cold climates, snow and the weather source (agreed 30 October 2026)

The model keeps measuring the cost of production at a site. Delivery (cables to shore, ships to
markets) stays with the network model.

### 7a. Weather source

- **Agreed 3 November** (proposed 30 October): build the one-degree profiles from the ERA5 2019
  cutout on ARC (green-condor,
  0.25 degree, 8,760 hours; `data/README.md`) with atlite, instead of Salmon's profiles. The
  source is then direct; land and sea pixels can be averaged separately in each cell, which
  replaces the land correction of coastal cells and gives offshore cells sea-only profiles; the
  cold rules use the same temperature; and no capacity factor exceeds 1.
- **Turbines:** the existing atlite tiles use a Vestas V112 3 MW onshore, an NREL 5 MW offshore
  turbine and a fixed CSi panel, without tracking, and give 26-38 % lower wind capacity factors
  at the focal cells than the current profiles. The profiles are recomputed (atlite 0.4.1,
  `envs/green-condor-env` on ARC) with turbines matching DEA 2050 (the data sheets' rows; a
  turbine's power curve per unit of capacity is set mainly by its specific power):
  - onshore: DEA 6 MW, 165 m rotor, 281 W/m2, hub height 110 m. atlite's
    `NREL_ReferenceTurbine_2020ATB_7MW` (200 m rotor, 223 W/m2) with its wind speeds multiplied
    by (281/223)^(1/3) = 1.080: below rated power a turbine's output per unit of capacity
    depends on wind speed cubed over specific power, so this keeps the reference turbine's
    power coefficient at each point and moves rated power to the higher wind speed of DEA's
    turbine;
  - offshore: DEA 35 MW, 380 m rotor, 310 W/m2 (309 from the rotor), hub height 220 m. atlite's
    `NREL_ReferenceTurbine_2020ATB_15MW_offshore` (240 m rotor, 332 W/m2) with its wind speeds
    multiplied by (310/332)^(1/3) = 0.978, at 220 m;
  - both through `cutout.wind(turbine=dict(V, POW, P, hub_height), smooth=True)`: atlite
    extrapolates the cutout's 100 m wind to the hub height with the logarithmic profile and
    the cutout's roughness;
  - PV: `cutout.pv(panel="CSi", orientation="latitude_optimal")` for fixed tilt and
    `tracking="horizontal"` (single axis, north-south) for tracking, converted to output per MW
    of AC, the basis of DEA's costs, with DEA's DC/AC ratio of 1.25 and clipping at 1.
- **Air density:** wind speeds normalised by each point's air density before the power curve,
  V (rho / rho0)^(1/3), the IEC 61400-12-1 method for pitch-regulated turbines, with density from
  the point's height in the standard atmosphere (unless atlite 0.4.1 offers it). At 3,000 m the
  density is 0.74 of sea level and wind speeds count x 0.91; at 4,000 m, 0.67 and x 0.87, so
  output below rated power falls by about a third. Neither Salmon's nor Cesaro's thesis mentions
  it for Salmon's profiles.
- **Calibration check:** atlite's full-load hours at DEA's reference sites against DEA's: onshore
  3,800 h (Danish site), offshore fixed 4,650 h (North Sea, 10.3 m/s at 100 m, Table 11), PV
  1,515 h fixed and 1,742 h tracking (AC). DEA's hours are a reference, not a target (Carlo,
  5 November), so a gap adds no factor.
- **What is known of Salmon's profiles (3 November):** ERA5 2019 at one degree, wind through
  "the curve for a Vestas 3 MW wind turbine", PV through pvlib, using "the DC power output from
  solar cells" (Salmon 2023, DPhil thesis, chapter 3, p. 37); the conversion was Zac Cesaro's,
  whose thesis (2022, section 4.5) uses ERA5's 100 m wind and pvlib's PVWatts from ERA5's
  surface solar radiation. Neither names the Vestas model (V90, V112 or V126 at 3 MW have
  472, 305 and 241 W/m2), the hub height or the panel tilt. Against DEA 2050, over five Danish
  land cells and four Danish North Sea cells: onshore wind 5,076 h (DEA 3,800, +34 %), offshore
  wind 6,470 h (DEA 4,650, +39 %), fixed PV 1,039 h (DEA 1,212 per MW of DC, 1,515 per MW of AC),
  tracking PV 1,354 h (DEA 1,742 per MW of AC). DEA's PV costs, which the model uses, are per MW
  of AC (sheet 22: 0.29 MEUR/MW, 0.23 per MW of DC, DC/AC 1.25), so the model counts DC output
  per MW against AC costs per MW, about 25 % too little PV output per euro, while wind runs well
  above DEA's full-load hours.
- **Built 3 November:** `model/weather_profiles.py` (settings in
  `inputs/parameters.yaml`, `weather.profiles`), `arc/jobs/11_weather_profiles.sh` (check,
  convert in 12 bands of 15 cell latitudes, stack), `tests/test_weather_profiles.py`. A local
  trial on the PyPSA-Earth European cutout (2013, 0.3 degrees, so the land weights are uniform)
  gave, after the wake factor of 0.93: onshore 2,976 h with atlite's power-curve smoothing and
  3,731 h without (DEA 3,800), offshore 4,500 and 5,266 h (DEA 4,650), PV 1,351 h fixed and
  1,673 h tracking per MW of AC (DEA 1,515 and 1,742). The ARC check reports both smoothing
  settings on the 2019 cutout. It takes hours, not minutes: job 13603956 timed out at one hour,
  and 13610857 reruns it with a 12-hour limit.
- **Checked 5 November** (job 13610857, 6 h 5 min; `check.json` in
  `green-lory-campaigns/weather_profiles_20261104_v1/` on ARC): 2019 full-load hours over DEA's
  reference cells, wind after the wake factor of 0.93, with the ratio to DEA in brackets.

  | Technology | DEA 2050 | Raw power curve | Smoothed power curve |
  |---|---|---|---|
  | Onshore wind | 3,800 | 3,762 (0.99) | 2,968 (0.78) |
  | Offshore wind | 4,650 | 5,555 (1.19) | 4,702 (1.01) |
  | Fixed PV | 1,515 | 1,334 (0.88) | |
  | Tracking PV | 1,742 | 1,656 (0.95) | |

  - Smoothing: atlite's `windturbine_smooth` shifts the power curve by 1.27 m/s and spreads it
    with a 2.29 m/s kernel, so the offshore curve reaches full output at about 15.8 m/s, not
    10.7 m/s. Against the raw curve it takes off about 21 % at a mean hub speed of 6 m/s and 8 %
    at 14 m/s (Weibull shape 2.1). The "Oversmoothing" warning in the log is harmless: both
    curves give 0.1 % of rated output at 0 m/s, and atlite tests an absolute 0.01 MW, which only
    the 15 MW curve passes.
  - Offshore: the cells read 10.0 to 10.1 m/s at 100 m in 2019, against DEA's 10.3 m/s, so at
    DEA's wind speed the ratio would be about 2 % higher. DEA's hours are net, built up from the
    measured Danish fleet (Table 13). Its note B gives 5 % for outages and grid losses, so a
    factor on the raw curve (0.84, with the wake factor 0.78) would be only partly sourced.
  - PV: 1,334 h is DEA's 2020 plant (1,343 h, performance ratio 0.86). DEA's 2050 figure differs
    only by the performance ratio, 0.97, which includes 0.05 from bifacial modules: full-load
    hours are irradiation x transposition factor x performance ratio (sheet 22, note K). A
    factor of 0.97 / 0.86 before the cap would give about 1,500 h fixed and 1,865 h tracking
    (12.4 % more on the local 2013 cutout, the cap taking 0.4 % in Denmark and 0.7 % in southern
    Spain). Not applied: see the decision below.
  - **Decided (Carlo, 5 November):** one setting for land and sea, and DEA's hours are a
    reference, not a target. Smoothing stays on (`smooth_power_curve: true`), with no wind or PV
    factor. The kernel is the PyPSA-Eur paper's (Hörsch et al. 2018, eq. 8), which smooths its
    onshore and offshore power curves "to account for effects of spatial wind speed variations
    within a grid cell", with parameters fitted to Danish feed-in; atlite's default and
    PyPSA-Earth's current configuration leave it off. A plant here is a farm spread over its
    cell, not one turbine. The raw curve gives
    5,555 h in the North Sea, 19 % above DEA's net figure; smoothed, onshore wind reads 2,968 h,
    within the 2,600 to 3,100 h of Danish turbines installed since 2010 (DEA, chapter 20).
    atlite's efficiency of 0.95 cancels, since the profiles are divided by the smoothed curve's
    maximum, so the wake factor of 0.93 is the only derating.
  - Against Salmon's profiles at the same cells, after the wake factor: onshore wind -37 %,
    offshore wind -22 %, fixed PV +28 %. C2 will move towards solar sites.
- **Aggregation:** each one-degree cell's profile is the average of its 0.25-degree pixels
  weighted by land area (onshore) or by sea area of the depth class (offshore), through atlite's
  `matrix` argument; written in the current file format, so runs only change `weather_dir`
  and drop `weather_land_correction`.
- **Check:** the new profiles against the current ones on the 1,317 step-4 cells, and the
  focal-cell LCOA. The reconciling run keeps Salmon's profiles, as the legacy model used them.
- **Compute:** like the existing tiles, a Slurm job per latitude band. December 2025's tiles (60
  ERA5 rows each, as a band here) took 17 to 50 hours on 32 cores with 360 GB requested. By
  `sbatch --test-only` on 5 November, those resources would start within hours on `short`
  (12 h limit), in 3.5 days on `medium` (2 days) and in 19 days on `long`. So one band runs first
  on `short` to measure the time: its log gives each technology's time even if it runs out, and a
  finished band stays valid for the stack. Band 30-44 N: job 13618227, array task 8, submitted 5
  November from release 20261105-profiles-v1; it ran in 1 h 59 min on 32 cores, 6.7 GB at peak
  (wind 20 min, wind at sea 4, fixed PV 74, tracking PV 16). Sea and land tables stop at 75
  degrees, so the band from 90 to 76 S (array task 0) holds no run cell and is skipped: 11 bands.
- **First look, band 30-44 N (6 November):** over C2's 2,678 land cells there, against Salmon's
  profiles as C2 read them, wind falls to 0.59 of its hours at the median (0.47 to 0.92 between
  the 10th and 90th percentiles) and fixed PV rises to 1.23, almost all of which is the change to
  output per MW of AC (x 1.25). Against NREL's 2024 supply curves (Lopez et al., in Zotero; a 6 MW
  turbine, 170 m rotor, 115 m hub, state averages of developable land), the smoothed hours of the
  five flat Great Plains states are 0.62 to 1.20 of NREL's (median 0.92); hillier states read
  lower, as a cell's average falls below the best sites NREL places turbines on.
- **Why the wind falls (6 November):** not the units (both per MW, at most 1), the year (both
  2019; hourly correlation 0.93, highest with no shift) or the sample. Over 2,168 inland cells of
  band 30-44 N, at the median, the new 1,033 h against Salmon's 1,887 h (before wakes) split into
  DEA's turbine (x 0.73), air density (x 0.91) and smoothing (x 0.83), found by recovering the
  hub-height speed behind each hour and running it through the other curves. Salmon's output
  matches a Vestas V112 at 100 m only with 14 % more wind speed hour by hour (8-20 % between the
  quartiles), as a low-wind turbine of about 210-240 W/m2 would give. Cesaro's thesis (section
  4.5.1) computes speed as sqrt(u2 + v2) and scales 100 to 120 m by 1.023, and used an IEC class III
  curve for India; Salmon's global curve is not documented. DEA's turbine is 281 W/m2, the one
  whose costs the model uses.
- **Global conversion (6 November):** a first attempt (release 20261106-profiles-v2, folder
  `weather_profiles_20261106_v1`, array 13622975) failed at once on every band south of the equator:
  the job script passed `--cells -75:-61`, which the argument parser reads as an option. Fixed in
  fe99393 (`--cells=-75:-61`, with a test) and cancelled. Rerun from release 20261106-profiles-v3
  (fe99393), folder `green-lory-campaigns/weather_profiles_20261106_v2/`: bands 1-11 as array
  13623198 (short, 32 cores, 64 GB; 1 h 11 min to 3 h 21 min each). The stacks (13623199-200)
  wrote every file but failed on `provenance.json` (NetCDF attributes read back as NumPy
  integers); fixed in `write_stack` and rerun on 9 November from release 20261109-profiles-v4
  (jobs 13631579-80) into `land/` and `sea/`, linked as `data/weather_data/era5_2019_1deg` and
  `era5_2019_1deg_sea` (`data/README.md`). Every land run cell (16,885) and sea run cell (40,197)
  has all 8,760 hours. Against Salmon's profiles as C2 read them, on the 1,317 validation cells:
  wind hours 0.58 at the median (0.52 to 0.70 between the 10th and 90th percentiles), fixed PV
  1.17 (1.11 to 1.31). Wind at sea, after the wake factor: median 3,575 h (fixed 2,792, floating
  3,295, deep 3,701); DEA's North Sea cell 4,768 h, Dogger Bank 4,479 h.
- **Campaign `campaign_20261109_v1` (9 November):** smoke stage of the four land and four sea runs
  from release 20261106-profiles-v3 (shards 13631640-50 and 13631677-87, with their QA jobs).
  The stacks are linked into the shared data as `data/weather_data/era5_2019_1deg` and
  `era5_2019_1deg_sea`, the `weather_dir` of the campaign runs.
- **Campaign runs (defined 6 November, `runs.yaml`):** C2, S1, F and S2 read the new profiles and
  drop the land correction; each has a sea run (`_offshore`) on the sea land table
  (`land_sea_20261105_v2`, 40,197 cells), the flat or spatial sea site costs and, for F and S2,
  the finance table by cell. Sea runs use their own smoke and diagnostic cells
  (`inputs/cells/diagnostic_cells_sea.csv`: Dogger Bank, Kerguelen shelf, mid-Atlantic). The runs
  go from release 20261106-profiles-v3, which holds these definitions too; releases
  20261106-campaign-v1 and -v2 were staged before the fix and not used.
- **Atolls (found 5 November, before the bands ran):** 24 run cells of the Tuamotus (14 to 22 S)
  have land in the land table, from MODIS, and none in `cell_geography.csv`, so they would have
  had no land profile and no solution. `band_cells` now gives every cell a land profile, from all
  its points where the geography has no land. They make 0.41 Mt/yr in C2, none reaches 1 Mt/yr,
  and no cell of band 30-44 N is affected, so the other bands need a release with this change.

### 7b. Cold climates

- **Construction:** labour productivity by temperature, from the Alaska school-construction cost
  factors (HMS for Alaska DEED, Bond Reimbursement and Grant Review Committee packet, 21 February
  2019, Equation 1, report pp. 8-9): (40 - T) - 3 % on labour, T the annual mean in Fahrenheit,
  Anchorage (37 F) as zero, from "published rates for modifying the labor productivity for
  temperature below 40 degrees". Used by month: the mean over the 12 months of
  max(0, 4.4 C - monthly mean) x 1.8 % per C, on the labour share of the build cost
  (`installation_multiplier`). The labour share of DEA's build costs is to be set (Alaska's 42 %
  is for a school).
- **Equipment (proposed):** a premium on the equipment part (`tech_cost_per_mw`) where IEA Wind
  Recommended Practice 13 (2017, sections 3.1 and 5) recommends low-temperature versions: annual
  mean below 0 C, or below -20 C for at least an hour on more than nine days a year (a daily
  minimum below -20 C is exactly an hour or more below it).
  - Wind +5 %: arctic equipment (blade heating, cold-rated parts) made the turbines built cost
    "somewhat over 5 %" more, 5-10 % in all (Marjaniemi and Peltola 1998, VTT Julkaisuja 830,
    pp. 49 and 52; 200-600 kW turbines in Finnish Lapland). Their heating used 1-3 % of the
    annual output, which is not counted.
  - Electrolysers, Haber-Bosch with its ASU, compressor and tanks +10 %: enclosures and heat
    tracing. Between USGS's Arctic offshore cost adjustment, +25 % on topsides material only
    (Verma, White and Gautier 2008, USGS Open-File Report 2008-1193, p. 4), and a fully
    enclosed, heated drilling rig in Arctic Alaska, up to USD 3 million against USD 20-25
    million for a new rig, 12-15 % (David 1982, abstract only).
  - PV and batteries: none. Modules are qualified to -40 C; batteries are housed and heated,
    so cold acts as a heating load, not modelled.
  - The all-in Arctic factors (Fairbanks 1.35 and North Slope 1.5 against the US Gulf Coast;
    Muse, Stancil & Co. 2004, p. 19; Robertson 1999, INEEL, p. 10) are mostly labour, camps,
    freight and materials (Robertson: 1.5 falls to 1.2 at an ice-free port through "reduced
    shipping, labor, and materials costs"), which the remoteness factors and the construction
    rule above cover. No source gives a premium as a function of design temperature.
- **Cold rule (decided 3 November; replaces the construction and equipment rules above):** every
  technology costs 25 % more (`tech_cost_per_mw` and `build_cost_per_mw`) at cells at or beyond
  60 degrees latitude, or at 3,000 m or higher: about 20 C colder than sea level at the standard
  lapse rate (6.5 C per km), roughly the drop from the subtropics to 60 degrees. Elevation is the
  mean of the cell's land in GEBCO 2025 (`cell_elevation.csv`), not the land table's
  `elevation_m`, a GEBCO point at the cell centre that can land on a peak: (35 S, 70 W) reads
  4,234 m against a mean of 2,713 m. 360 land cells below 60 degrees reach 3,000 m, against 386
  at the centre. No temperature data needed. MODIS land cover supports the latitude: permanent
  snow and ice is 1 % of land at 55-60 S and 99.6 % at 60-65 S, and in the north it starts to rise
  there, 0.6 % at 55-60 N, 3.5 % at 60-65 N and 25 % at 70-75 N. The design is unchanged, so the case without the
  premium follows from the same results.
  - Source: Yamal LNG, adjusted to Corpus Christi's scope, costs 25 % more, "reflecting the
    higher construction costs in the harsh arctic environment" (Songhurst 2018, OIES NG137,
    p. 40), the one like-for-like Arctic process plant. The figure is all-in, so with the
    remoteness factor it counts some remoteness twice: on the high side by intent.
    Equipment-only figures are lower (drilling rig 12-15 %, Arctic topsides +25 % on material,
    wind 5-10 %). The much larger Arctic factors are mostly logistics, labour and short building
    seasons: the North Slope's 1.5 against the US Gulf Coast falls to 1.2 at an ice-free port
    through "reduced shipping, labor, and materials costs" (Robertson 1999, INEEL, p. 10); North
    Slope schools cost about twice Anchorage's, with logistics +54 % and cold on labour about
    +8 % of the total (HMS for Alaska DEED 2019). Section 7e deals with remoteness.
  - Effect in C2 of 30 October (Salmon's profiles): 4,077 cold cells, 3,691 by latitude and 387
    by altitude (mostly Tibet and the Andes; one cell both). They supply 3.5 % of the cheapest
    2,000 Mt/yr and none with the premium; the 2,000th Mt moves from 283 to 284 EUR/t. Among the
    cheapest 5 % of cells the cold ones fall from 135 to none, including the Altiplano and Puna
    cells of Bolivia, Argentina and Chile (282-285 EUR/t). Salmon and Bañares-Alcántara (2022)
    had no cold premium: they excluded latitudes beyond 85 degrees and Antarctica, and the legacy
    global runs covered 65 S to 65 N (`lcoa-opt/main_mp.py`).
- Fixed O&M follows, as a fraction of the overnight cost.

### 7c. Snow on PV

- A limitation, not modelled (decided 30 October): the profiles come from irradiance and
  temperature only, so snow lying on the modules is not counted. Snowfall comes with cloud, so
  the profiles already give little output during it; what they miss is the cover that stays on
  the modules on clear days after snow. Losses reported for snowy sites: 1-15 % a year, up to
  90 % in a month (Sandia's summary of Marion et al. 2013).

### 7d. Offshore sites

- **Cells:** every cell of the land table with at least 1 % sea: 40,197 sea plants (2,605 fixed,
  4,456 floating, 33,136 deep), 5,172 of them in cells that also have land. A sea plant is a
  separate run with its own land table, weather stack and site-cost table
  (`model/sea_cells.py`), so the plant code is unchanged and a coastal cell can hold both.
  Without those cells only 540 fixed cells would remain, since shallow water lies near land.
  Depth is the mean of the cell's sea (GEBCO, `sea_depth_m`). Country from exclusive economic
  zones (Marine Regions, World EEZ v12, CC BY 4.0, `data/marine_regions/`; the sovereign,
  `iso_sov1`), the nearest zone reaching the cell where a coastal cell's centre is on land; cells
  outside every zone, Antarctica's coast among them, are tagged international waters (24,254) and
  run too. Sea plants stop at 75 degrees, the land table's limit. The Caspian holds no sea plants:
  the lakes database counts it as inland water, so its 18 cells without land hold no plant of
  either kind.
- **Sea ice (decided 5 November):** nothing is left out for ice. Sea is what the cell geography
  reads as sea, and south of 55 S, where that table stops, MODIS's water class; frozen or not, it
  holds a plant, and the cold rule prices it: 7,630 sea plants at or beyond 60 degrees pay the
  25 %. Antarctica's ice shelves read as water too, in MODIS as in GEBCO's sub-ice grid, so they
  hold plants, at the costs of a cold, deep and remote site. Below 60 degrees frozen sea pays no
  premium (605 cells, section 7f).
- **Area (decided 5 November):** the cell's sea at the land table's share (20 %) and the wind
  density of the land runs (5 MW/km2).
- **Flat runs at sea (decided 6 November):** the depth-class costs of wind and the plant are the
  base costs of a plant at sea, so the flat sea runs (C2 and F) keep them, from
  `site_costs_sea_flat_20pct.csv`, with every location factor 1. S1 and S2 add the location factors
  (remote, cold, labour) on top, from `site_costs_sea_20pct.csv`, as on land. Without the base
  costs a flat run would put a sea plant at onshore cost, and open ocean and ice shelves would
  look cheaper than land wherever the wind is strong.
- **Coastal cells (decided 5 November):** a coastal cell holds two independent plants, each on its
  own area, with nothing shared between them: 4,428 of C2's 16,885 land plants have a sea plant in
  the same cell. For the network and the supply curve they are separate suppliers, so their
  capacities add. A sea plant's `Index` ends in `_sea` (`python -m model.suppliers export
  --sea-surface`); the network code in green-porpoise treats `Index` as a key and takes two
  suppliers at one location unchanged. Its country is the zone's sovereign, XIW in international
  waters, since the network needs a code on every supplier. The network routes a sea plant to a
  port like any supplier, by its onshore route; revisit this at the network stage.
- **Maps (decided 5 November):** land and sea maps show every plant. Combined maps show
  `results.lowest_feasible_plant`, the cell's cheapest plant that can make the 1 Mt/yr reference
  output. Every LCOA is that of the 1 Mt/yr design, so a smaller plant cannot make ammonia at
  that cost at scale, and the cheapest colour would otherwise mostly be plants too small to
  matter: in C2, 81 of the 100 cheapest land plants are below 1 Mt/yr (median 0.04). The cutoff
  is the network's, so a combined map shows what the network can choose. In C2 it keeps 95 % of
  the capacity (7,224 of 16,885 plants, 56,735 of 59,844 Mt/yr) and moves the cheapest plant
  from 202 to 221 EUR/t and the 10th percentile from 298 to 294 EUR/t. In the 4,428 cells with
  both plants, by the about 320 km2 a sea plant takes per Mt/yr: both reach 1 Mt/yr in 547, the
  sea plant alone in 2,828, the land plant alone in 284, neither in 769. A cell with neither is
  left blank: with the 6,064 land plants below 1 Mt/yr that have no sea plant, 6,833 cells of
  the land run. A capacity map of the same table shows what the shown plant can make; the two
  plants of a cell add only in the supply curve.
- **Technologies:** wind only, no PV offshore; the plant produces at the site, so DEA's
  substation and export cable are left out of the wind cost.
- **Costs by depth** (2050, DEA electricity catalogue, chapter 21 and its data sheets; MEUR/MW):

  | Depth | Wind | Conversion plant and stores | Sources |
  |---|---|---|---|
  | up to 70 m, fixed | turbine 0.75 + foundation + array cables 0.04 + development 0.16 (1.28 at 30 m); foundation 0.334 at 30 m, scaled by DEA's function of depth, 0.45 MEUR/MW + 5 kEUR/MW per m | x 1.72: on a fixed jacket platform, 42 % of a 300 t/d offshore plant's capital cost (Brener 2022, MSc thesis, Oxford) | DEA p. 208: "fixed foundations can be used in water depths of up to 70 metres"; Table 15, p. 225 (its cost function is calibrated on 20-50 m, Table 11) |
  | 70 to 1,000 m, floating | 0.75 + 0.675 + 0.04 + 0.16 = 1.63 | x 2: a process plant on a moored floating hull. Floating LNG plants, hull, turret, mooring and marine installation included, cost near or above USD 2,000 per t/yr for the first newbuilds, like the costliest onshore plants (Scott 2018, Oil & Gas Journal); later ones about USD 500-1,130 per t, in line with onshore plants, against USD 2,114 for Prelude (Rystad, in World Oil and JPT 2025). x 2 is the first-of-a-kind level | NREL 2016 (Musial et al., p. v and note 2): floating beyond 60 m, water deeper than 1,000 m excluded |
  | deeper than 1,000 m | x 3 on every technology at onshore cost: turbines and plant on ships; speculative. Order of magnitude: floating wind at 500 m depth and 1,000 km from shore, 2.8-4.0 MUSD/MW (Salmon and Bañares-Alcántara 2022, Table 2), 2.5-3.5 times onshore; a first-of-a-kind floating plant, 2-4 times a mature one (Prelude against later floating LNG plants, Rystad) | | |

- **Build factor at sea:** the remote factor of section 7e, with the travel time of the nearest
  port plus the sea leg at 20 km/h beyond 175 km (decided 3 November, replacing 60 km).
- **Water:** desalinated seawater at the source price, no pipeline.
- **Finance:** international waters take the highest rate in Ameli et al.'s table in the finance
  runs (11.8 %, the rate of their Africa region), as a high-risk default: no state's legal
  framework, insurance or offtake support. (Damodaran's US cost of capital for the
  shipbuilding and marine industry, 6.7 % in January 2026, is a company rate, not a project
  rate.) The uniform-rate runs give them 5 % like every cell. The 23 small states with zones but
  no row in Ameli's table (1,396 sea cells) take the rate of their Ameli region (decided 5
  November), since those rates are regional.
- **Fixed foundations:** up to 70 m, DEA's limit (decided 30 October); the plant multipliers
  x 1.72, x 2 and x 3 as in the table.
- **Cost parts (decided 5 November):** the turbine is the equipment part
  (`tech_cost_per_mw`); the foundation, array cables and development are the build part.

### 7e. Remoteness, onshore and offshore (decided 3 November)

Carlo (3 November): a remote place must cost what building there costs, at sea as on land. A
remote cell that looks cheap in the model would invite a case study that the costs do not support.

- **The problem:** the installation factor rises with travel time to a city of 20,000 (Nelson
  et al. 2019) only up to 4 hours and then stays at x1.34 (GHD 2025 fit); the equipment factor
  stops at x1.10. The remotest calibration site is 6.7 hours from such a city, and 41 % of the run
  cells lie beyond it. In C2 (no site costs), cells more than 12 hours away supply 58 % of the
  cheapest 2,000 Mt/yr; with S1's factors (15 October) still 25 %. The cheapest cells of C2 are
  sub-Antarctic islands (Campbell, Macquarie, Auckland, Crozet, Kerguelen, Heard; 202-212 EUR/t,
  28 to 208 hours from a city) and the Magallanes fjords: 25 of the 30 cheapest cells can host
  less than 0.1 Mt/yr. The islands are nature reserves, and the protected-land layer does count
  them: `protected_area_pct` is a share of the whole cell, so Kerguelen's 9.3 % is 91 % of its
  10.2 % land (Crozet 87 %, Auckland 70 %, Macquarie 66 %, Campbell 63 %; checked 5 November).
  Heard reads 0: WDPA tags its reserve as marine, and the land table leaves out marine records.
  Its two cells hold under 0.1 Mt/yr.
- **One measure on land and at sea:** travel time from a city of at least 20,000 people, for
  equipment and build alike. On land, Nelson et al. as now; at sea, the nearest port's own travel
  time plus the sea leg at 20 km/h (Salmon and Bañares-Alcántara 2022), beyond the 175 km that
  DEA's offshore costs include.
- **Calibration beyond 6.7 hours (proposed, replaced by the remote factor below):** public
  cost-factor tables for genuinely remote places, added to
  GHD's sites and located on the same travel-time raster: Alaska's geographic cost factors (HMS
  for Alaska DEED; road-connected towns to fly-in villages, relative to Anchorage, split into
  logistics and labour), and, where found, northern Canada, remote Western Australia and the
  Northern Territory, and the US Defense Department's area cost factors for remote sites
  (Pituffik, formerly Thule, 3.39). The fitted curve keeps rising with travel time instead of
  stopping at 4 hours.
- **Remote cost data found (3 November):** 2,325 rows from 21 public series (session scratch
  folder `remote_costs/`, with `REPORT.md`). The most remote places with published construction
  factors are settlements reached by air, barge or seasonal road, at about x1.5 to x2.4 of the
  nearest supply city for a whole building: Alaska's fly-in and no-barge villages x1.60-1.63 on
  Anchorage (HMS for Alaska DEED, July 2023; up to x2.00 in the 2008-2018 table); the US Defense
  Department's Arctic villages x1.44 and Aleutians x2.25 on Anchorage (UFS 3-701-01, June 2026);
  Shamattawa, a fly-in First Nation, x1.99 on Winnipeg (federal manual, 1996); Australia's remote
  islands x1.75-2.00 on the state capital (Rawlinsons, via Nous 2017); Puerto Williams x1.62 on
  Punta Arenas. HMS's 2018 split for North Slope villages: general requirements (freight, fuel,
  crew travel and per diem, duration, equipment) +54 to +57 %, productivity +18 %, risk +18 %.
  No source covers places without a settlement, airstrip or landing, and none gives coordinates
  or travel times: the places must be geocoded and read on the Nelson et al. raster.
- **Remote factor (decided 3 November):** one rule for every point, on land and at sea, with no
  breakpoints, from one source used as a sensitivity factor rather than a case study: Canada's
  1996 federal capital cost manual (*DIAND Cost Reference Manual*, Real Property Services for
  INAC, July 1996, pp. 1-15 and 1-16). A duplex at Shamattawa, Manitoba, "only accessible by air
  and winter road" about 750 air km from Winnipeg, costs x1.99 Winnipeg's, the product of nature
  of site 1.04, transportation 1.60, schedule 1.00, personnel 1.14, materials 1.05 and
  administration 1.00. Rounded, x2 on the build part and x1.6 (its transportation part) on
  equipment, which is made elsewhere and shipped in. Shamattawa is 20.4 hours (0.85 days) from a
  city of 20,000 on the Nelson et al. raster, which has no road restriction (it models off-road
  travel at land-cover speeds, so fly-in places get finite times).
  - Rule, travel time d in days from a city of 20,000: build (`build_cost_per_mw`) =
    1 + 1.127 log2(1 + d), equipment (`tech_cost_per_mw`) = 1 + 0.676 log2(1 + d), so each
    factor rises by the same step for each doubling of (1 + d) and equals x2 and x1.6 at 0.85
    days. Days, because hours matter little at this scale.
  - Why a logarithm: remote costs grow by steps in kind (daily commute, overnight stays, a camp
    with rotations, flights or winter roads, sealift), each roughly doubling the travel time, so
    each doubling adds about the same. log2(1 + d) is zero in a city, nearly linear for short
    trips (as the Australian fits, linear in hours), and grows ever more slowly for long ones; a
    linear rule through Shamattawa would give x10 on the build part at Kerguelen, and a cap
    would need a cutoff and make every remote place equal. The form is a convention; it has no
    fitted parameter beyond the Shamattawa anchor.
  - Values: 1 hour (0.04 days) build 1.07, equipment 1.04; 4 hours 1.25 and 1.15; half a day 1.66
    and 1.40; Shamattawa (0.85 days) 2.00 and 1.60; 2 days 2.79 and 2.07; 4 days 3.62 and 2.57;
    Kerguelen (7.5 days) 4.48 and 3.09, about x3.3 on a wind and electrolyser plant.
  - Comparison for remote islands (a check, not an input): the US Defense Department's
    construction area cost factors put Pituffik (Thule Air Base, Greenland) at 3.39, Diego Garcia
    at 3.30 and Wake Island at 4.36, on the average of 96 base cities in the contiguous US = 1.00
    (UFS 3-701-01, *DoD Facilities Pricing Guide*, 2026, Table 4-1; the factor covers labour,
    materials and equipment prices adjusted for weather, logistics and labour productivity).
  - Check in the Australian range (up to 6.7 hours), with nothing fitted to it: the build rule is
    0.04 below GHD's installation factors on average (RMSE 0.087; GHD's own fit 0.070) and 0.19
    below Aurecon's; the equipment rule is 0.01 above Aurecon's equipment factors.
  - Every point gets a factor: d from the raster where it has a value; otherwise, and at sea, the
    nearest port's own travel time plus the sea leg at 20 km/h (Salmon and Bañares-Alcántara
    2022). The site-cost table records which.
  - At sea (decided 3 November), the sea leg counts beyond 175 km, DEA's far-shore reference:
    DEA's costs other than
    substations and export cables are about the same at 175 km (DC, 40 m) as at 60 km (AC, 30 m),
    1.31 against 1.28 MEUR/MW, so marine installation to 175 km is in DEA's costs.
  - Effect on C2 with the designs held fixed, cold rule included: cells more than 12 hours from
    a city supply 26 % of the cheapest 2,000 Mt/yr with the present factors and none with this
    rule; the 2,000th Mt stays at 327 EUR/t, and cells 1-7 hours from a city change by +0.3 % at
    the median. Kerguelen's five land cells go from 239-262 to 701-770 EUR/t.
- **Built 5 November:** `model/site_costs.py` (`remote_multiplier`,
  `travel_min_to_city`, `cold_rule`), `python -m model.site_geography ports`
  (`inputs/site_costs/port_travel_times.csv`), and the table
  `inputs/site_costs/site_costs_center_20pct_slope20.csv` rebuilt on the October land table, with
  its new hash in `runs.yaml` for S1 and S2. The offshore step will replace this table.
  - Eight of the 1,377 ports have no raster value, because their coordinates lie out in the water
    (Manila, Jingtang, Qingzhou, Jinzhou Wan, Bajo Grande, Steinkjer, Tarawa, Famagusta). The
    route uses the nearest port that has one.
  - On land the whole sea leg counts, since the 175 km allowance comes from DEA's offshore costs.
    61 of the 18,039 land cells take the port route (0.2 to 6.1 days), one of them in C2.
  - The table reproduces the effect above: 26.4 % to 0 %, 327 EUR/t, +0.3 %, Kerguelen 701-770.
  - The water pipeline follows the cell's multipliers, so its price rises with the rule, which
    the effect above leaves out (it holds C2's flat water price): median change 0, 90th
    percentile +4 USD/t of ammonia, 99th +29, at most +60 (Sahara, over 1,000 km from water).
    The cold factor applies to the pipeline too, but not to its pumping energy.
  - Greenland's ice sheet reaches 15.7 days (x5.6 on build); C2 does not run those cells, and its
    remotest cell is at 8.7 days.
- **No exclusion in this campaign (decided 3 November):** costs only, and every point gets a
  factor ("all points are theoretically reachable"; Carlo). C2 and S2 run on every cell, on land
  and at sea (compute is available). After S2, its results can be used to leave out remote areas
  in other runs and campaigns.

### 7f. Order

1. Profiles from the cutout, checked against the current ones.
2. Economic zones (downloaded on 30 October).
3. Remoteness and cold: the remote factor and the cold rule in the site costs (sections 7b, 7e;
   built 5 November).
4. Offshore land build and site costs.
5. Technology YAML and plant changes, run definitions.
6. The campaign (C2, S1, F, S2, with offshore cells), each stage after a go-ahead.

Sea ice (decided 5 November): no sea-ice data, the cold rule alone, with MODIS land cover
(MCD12C1) as the only cover data. MODIS cannot show sea ice: in every layer of the file (the 2022
land cover), all 4,470 all-sea cells inside the 2019 ice edge read the same as the 30,062 outside
it, 100 % water, and class 15, snow and ice, is permanent ice on land. Antarctica's ice shelves
read as water too (the Amery cell, 71 S 70 E, is 100 % water). What MODIS gives is the sea area
south of 55 S (its water class) and the support for the cold rule's latitude (section 7b).

NSIDC's Sea Ice Index (G02135, version 4.0) was tried for an exclusion and dropped. Against it, the
cold rule alone adds 5,743 cells: 5,138 at or beyond 60 degrees, which pay the 25 %, and 605 below
it, which pay nothing (278 fixed or floating, mostly Canadian and Russian: Gulf of St Lawrence,
Labrador, Okhotsk).

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
  technology-data.
- **30 October, later:** two decisions, shared with green-caribou.
  - Pressure: the electrolyser delivers hydrogen at about 30 bar, as the DEA catalogue assumes
    (renewable fuels, chapter 1.1, p. 30: "both AEC and PEMEC technology can deliver hydrogen at
    pressures as high as 30 bar, which can be sufficient for many Power-to-X (PtX) applications
    (e.g., ammonia"; p. 36: a compressor only for more than 30 bar; sheet 5.4, note C:
    "Assumption of higher pressure electrolysis"). So Haber-Bosch has no feed compressor and
    takes DEA's own electricity, 0.053 of its total input plus the ASU's 0.25 MWh per tonne of
    nitrogen (0.545 MWh/t, 0.0872 MWh per MWh of ammonia; the legacy 0.16 until now), and the
    storage compressor raises 30 to 200 bar (DEA's 1-200 bar cost and electricity x 0.358, by
    isothermal work). Key-run LCOA at the focal cells: -29.0 (Chile), -25.9 (Pilbara), -11.3
    EUR/t (central Australia); about half of it at the windy cells is the cheaper storage
    compressor. Local solves of green-caribou's design (feed compressor, 1 bar inlet) gave +6 to
    +7 EUR/t instead.
  - Currency: a cost in another currency or price year is converted at the ECB's annual rate of
    its price year, then brought to 2020 with the euro-area HICP (`currency` in
    `inputs/parameters.yaml`, `model.parameters.eur2020_per`). The water pipeline's dollars
    were escalated with the US CPI instead: the regenerated site-cost table (S1 and S2 pin its
    new hash) has pipelines 0.5 % cheaper, at most 0.14 USD/m3. The supplier table for the
    green-porpoise network model (each cell's LCOA and capacity in USD2018, the network model's
    currency; `model/suppliers.py`) follows the same rule from 30 October (0.8595 EUR2020 per
    USD2018 instead of the Way factor 0.9024): every supplier's LCOA is 5 % higher in USD2018.
- **30 October, evening:** C2, S1, F and S2 reran from release `20261030-campaign-v1`
  (campaign `campaign_20261030_v1`): smoke and diagnostic QA passed for all four, and the
  diagnostic cells match local solves to 2e-5. The global jobs of S1, F and S2 were cancelled
  before they started, to add a cold uplift to the site costs and offshore sites first; C2's
  global stage finishes (jobs 13578378-82), as the onshore reference for both.
- **3 November:** simplicity first, for the working papers and the thesis of the coming months:
  only data in Salmon's profiles, in the ERA5 2019 cutout, or in a small public dataset. More
  weather years are future work (`FUTURE_WORK.md`). Later the same day: a cold rule by latitude
  instead of temperature (proposed in section 7b), and Salmon's profiles kept only if their
  turbine and panel can be documented, otherwise new profiles from the cutout (section 7a).
  Then: new profiles from the cutout agreed (section 7a; the theses name only a Vestas 3 MW
  turbine); the cold premium is 25 % (Yamal) at 60 degrees latitude or 3,000 m and above
  (section 7b); the remoteness factor is to be reworked for genuinely remote places, on land
  and at sea; no exclusion: C2 and S2 cover every cell, and S2's results can later be used to
  leave out remote areas (section 7e).
  The remote factor is decided: travel time in days, x2 on the build part and x1.6 on equipment
  at Shamattawa's 0.85 days (DIAND 1996), logarithmic in travel time, every point costed; at sea
  the sea leg counts beyond DEA's far-shore 175 km. DIAND 1996, the DoD pricing guide (UFS
  3-701-01, 2026) and Songhurst (2018) added to Zotero.
- **6 November:** flat runs at sea keep the depth-class costs, the base costs of a plant at sea
  (section 7d); green-lory's two commits of 4 and 5 November pushed.
- **5 November:** no sea-ice data: frozen sea counts as sea and the cold rule prices it, with
  MODIS land cover as the only cover data; NSIDC's Sea Ice Index was tried and dropped. Wind
  profiles smoothed on land and at sea, with no wind or PV factor; DEA's hours are a reference,
  not a target (section 7a). Sea plants as separate sea runs (section 7d): coastal cells hold a land and a sea plant
  as separate suppliers; the sea at a 20 % share and 5 MW/km2; the foundation in the build part;
  small states at their Ameli region's rate; sea plants up to 75 degrees. Combined maps show the
  cheapest plant that can make 1 Mt/yr; land and sea maps show every plant.
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
