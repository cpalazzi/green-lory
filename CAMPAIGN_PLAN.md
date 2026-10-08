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
| D4 | Water | **Decided 7 October** (section 0b): desalination at the nearest water source plus a pipeline to the cell, with an official water-pipeline cost replacing the district-heating proxy; sea and permanent lakes are sources at the same 2 USD/m³; no pumping for the elevation gain; the 1° geometry is kept. |
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
   Aurecon's as the upper bound.
2. **The slope factor is nearly inactive.** On GEBCO's 15-arc-second grid 94 % of the land from
   55 S to 9 S lies below 8.5 degrees, so with the 20 % share taken flattest first only 6 of
   2,806 cells pay a slope factor (at the full share 1,674 cells, up to 1.54). The slope decision
   then acts through land up to 20 degrees, not through cost. No change proposed: the classes
   were fixed before this was known, and the paper can say so.
3. **The water source price** at 2 EUR2020/m3 (as the flat runs) rather than 2 USD.

Running and still to do in this step:

- The global land build runs on ARC since 8 October from release `20261008-land-slope20-v1`
  (commit 53faded, verified by job 13434113): job 13434202 builds the 100 % table, 13434203 and
  13434204 derive the 2 % and 20 % tables, into
  `green-lory-campaigns/land_center_slope20_20261008_v1/` (about 5.5 h, as in September).
- Then: compare it with the September table (every column but the slope ones should match where
  slope is not involved), write the site-cost table of the 20 % table into
  `inputs/site_costs/` with your choice of calibration, and the global maps.
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

- The D9 literature check: compressor cost, storage basis, Haber-Bosch flexibility; any change
  goes into the YAML with its source.
- 4 h against 1 h on Australia's 565 eligible cells and the 563-cell legacy sample, both with the
  improved plant, to measure the resolution effect before the global runs.
- PV profiles above 1.0 (AC/DC basis).
- **Review:** the validation note, before the campaign.

## 5. Campaign

All runs:
- DEA 2050 costs, fixed-tilt PV, co-located land, 2019 weather.
- Land applied after the solve, capacity from the scaled design.
- The new land table (no slope ban), onshore cells with suitable land, 20 % share with 2 % derived.

| Run | Step | Build | Water | Finance | Network |
|---|---|---|---|---|---|
| C1 flat, 4 h | 4 h | flat | flat 2 EUR/m³ | uniform 5 % | yes |
| C2 flat, 1 h (base) | 1 h | flat | flat | uniform 5 % | yes |
| S1 spatial | 1 h | equipment by port; installation by city x slope (x depth, x labour = 1) | spatial | uniform 5 % | yes |
| S2 spatial + finance | 1 h | as S1 | spatial | Ameli low-carbon | yes |
| 2 % variants | derived | | | | C2, S2 |

- **Order:** new land build (about 4-8 h on ARC) → C1 → C2 → S1 → S2, each after your go-ahead →
  supplier contracts → networks at the 1 % tolerance, run from a green-porpoise release
  (`legacy_gpo/arc/`).
- **Compute:** about 1 h on four 48-core jobs per hourly surface of about 17,000 cells (less at
  4 h), and 1-2 h per network.
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
- **Open, for you** (8 October):
  1. the three step-3 points of section 3: the remoteness table (GHD 2025 recommended), the
     nearly inactive slope factor (no change proposed), the water source price in EUR;
  2. S2 interest rates (step 5): each cell's country as the surface assigns it (largest
     overlap), sea cells from the nearest land cell (`python -m model.finance ameli`);
  3. network equations for the campaign (step 5): corrected (`legacy_gpo --corrected`,
     recommended; the archived-table control A2c gives 261.64 USD/t against A2's 260.94,
     Australia unchanged, ships calling at 1,190 ports against 808) or as deposited;
  4. supplier-site storage in the corrected networks (step 5): drop it (recommended; the plant
     LCOA already holds its ammonia tank; about 1 USD/t, no change to the network);
  5. Zotero: Lehner et al. (2025), GHD (2025) and Pekel et al. (2016) (see the references below).
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
  8.1 to 8.3 (location factors from Rawlinsons 2025). Not yet in Zotero.
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
  https://doi.org/10.1038/nature20584 (Global Surface Water v1.4, 1984-2021; not yet in Zotero).
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
  https://doi.org/10.5194/essd-17-2277-2025 (data: https://doi.org/10.6084/m9.figshare.28519994;
  not yet in Zotero).
- Weiss, D. J. et al. (2018). A global map of travel time to cities to assess inequalities in
  accessibility in 2015. Nature 553, 333-336. https://doi.org/10.1038/nature25181
- Zhou, Y. and Tol, R. S. J. (2005). Evaluating the costs of desalination and water transport.
  Water Resources Research 41, W03003. https://doi.org/10.1029/2004WR003749
