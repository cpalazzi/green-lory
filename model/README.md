# model/: the green-lory plant model

The package computes, for each 1-degree cell, the levelised cost of ammonia (LCOA) of a plant
powered by wind and PV, and how much ammonia the cell's land can supply. Modules, in the order
a run uses them:

| Module | Role |
|---|---|
| `config.py` | Paths (data, inputs, plant bundle), the default YAML and bundle, physical constants |
| `technology.py` | The technology YAML (with `extends:`): link efficiencies from each link's recipe, annualised costs, and the per-cell override table (interest rates, build-cost multipliers, water and land prices) |
| `finance.py` | The per-cell interest-rate (WACC) tables runs pin as `finance_csv`: uniform, or Ameli et al. (2021) by TIAM-UCL region (`python -m model.finance --help`) |
| `weather.py` | Hourly PV and wind profiles of a cell from the NetCDF stack, and block averages for coarser steps |
| `weather_land.py` | The land correction of the profiles of cells with water: point weights from the lakes database, factors from ERA5 at 0.25 degrees (an ARC job) |
| `plant.py` | One plant at one cell: the PyPSA network from the plant bundle, its constraints (Haber-Bosch ramps, battery coupling, land in the solve), the Gurobi solve and its acceptance check, the plant results |
| `accounting.py` | After the solve: cost splits (equipment, installation, O&M, interest), water and land costs in the headline, grid-use classification |
| `land.py` | Land budgets per cell (co-located or exclusive wind and PV land), the capacity rules (`scaled_reference_design`, `solved_quantity`), loading and checking land tables |
| `land_table.py` | Building land tables from MODIS land cover, WDPA protected areas and GEBCO slope classes (`python -m model.land_table --help`) |
| `site_costs.py` | Every location factor of the cost in one table per land build: equipment (port), installation (city), slope, water (`green-lory site-costs`; the cost mechanisms below) |
| `site_geography.py` | The per-cell and per-site inputs of the site costs, read once from global rasters, and the remoteness fit (`python -m model.site_geography --help`) |
| `surface.py` | Running the plant over many cells, serially or in forked workers, with a numerical retry chain (`python -m model.surface --help`) |
| `runs.py` | Run definitions (`../runs.yaml`) and the execution record (`../run_store.csv`); the `run_surface` arguments of a defined run |
| `results.py` | Reading results by run id: surfaces, network suppliers, rows at the focal cells, coast against inland, maps |
| `suppliers.py` | A QA-passed surface as a supplier table of the green-porpoise network model (`export`), and post-hoc land-share variants (`derive`); the network runs in the green-porpoise repository |
| `cli.py` | The `green-lory` command: `list`, `show`, `table`, `status`, `path`, `run`, `map`, `site-costs`, `shell` |

`plants/ammonia_plant_1mt/` is the plant; its README lists what the bundle fixes and what each
run sets from its YAML. Runs are data: `../runs.yaml` gives every setting of every run by id,
and the ARC job `../arc/jobs/01_run_global.sh` passes one run's settings to
`surface.run_surface`; `green-lory run <id> --cell=lat,lon` does the same at a few cells locally. Options and result columns are described in `../GLOSSARY.md`, the open
modelling choices in `../MODEL_CHOICES.md`.

## One cell, step by step (`surface.run_cell`)

1. Copy the base network: the plant bundle with the YAML's link efficiencies and the run's PV
   policy, read once per run.
2. Set the costs: every YAML cost annualised at its default interest rate
   (`technology.apply_base_costs`), then the cell's overrides: its interest rates, and, in runs
   with site costs, the equipment part times its equipment multiplier and the installation
   part times its build multiplier (`technology.apply_cell_overrides`; see the cost mechanisms
   below).
3. Read the cell's land row. With `after_solve` (all campaign runs) the land does not limit the
   design; with `in_solve` it bounds wind and PV inside the optimisation.
4. Solve the plant for the cell's year of weather (`plant.solve`).
5. Complete the accounting: water and land costs, cost shares, grid use (`accounting.py`), and
   the capacity: the 1 Mt/yr design scaled until it fills the cell's land (`land.py`).

## Site costs: the cost mechanisms

Runs with site costs (S1 and S2 of the campaign) read a site-cost table built for their land
table (`green-lory site-costs --land <table> --output <csv>`, `site_costs.py`); flat runs (C1,
C2) have none and use the YAML's uniform water price. Each technology's overnight cost is
`tech_cost_per_mw` x equipment multiplier + `build_cost_per_mw` x build multiplier, annualised
at the cell's interest rate; fixed O&M is a share of that cost, so it follows both multipliers
(`technology.apply_cell_overrides`). The result rows report the capacity-weighted
`equipment_cost_multiplier` and `build_cost_multiplier` applied (`accounting.cost_splits`).

| Mechanism | Acts on | Module and function | Input | Reference |
|---|---|---|---|---|
| Equipment (transport of plant and materials) | `tech_cost_per_mw`, every technology | `site_costs.equipment_multiplier`: 1 + a x great-circle distance to the nearest network port, capped | `../inputs/site_costs/cell_geography.csv` (`port_distance_km`), from `../data/ports/verschuur2024_ports.csv` | AEMO's equipment cost factors, fitted in `../inputs/site_costs/remoteness_fit.json` to GHD (2025) Table 8.3 or Aurecon (2024) Table 11-1; nearest-port basis as in AEMO (2025) |
| Installation (labour, materials, mobilisation) | `build_cost_per_mw`, every technology | `site_costs.installation_multiplier`: 1 + b x min(travel time to the nearest city of at least N people, saturation) | `cell_geography.csv` (median over the cell's land), from Nelson et al. (2019) | AEMO's installation cost factors, same fit (N and the saturation by leave-one-out cross-validation) |
| Slope | `build_cost_per_mw` of PV and wind | `site_costs.slope_multiplier`: 1.0 / 1.3 / 1.6 for land up to 8.5, 14 and 20 degrees, the plant's land share taken flattest first | the land table's `land_pct_slope_*` columns (`land_table.py`, from GEBCO) | Lopez et al. (2025), Table 12 (terrain multipliers); limits as decided on 7 October 2026 |
| Depth | `build_cost_per_mw` | 1 for every onshore cell (offshore cells are not in the table) | | |
| Labour level | `build_cost_per_mw` | 1 everywhere: no wage level by country; finance differences are the finance table's | | |
| Water | the LCOA, after the solve, at 1.5 m3/t | `site_costs.site_cost_table`: source price + distance to the nearest source x pipeline cost per m3 and km (`site_costs.water_pipeline_cost`); the pipeline's materials (30 %) take the equipment multiplier, the rest the PV and wind build multiplier | `cell_geography.csv`: sea and permanent lakes and reservoirs (GLWD v2 classes 1, 2, 3, 6; saline lakes only where the JRC Global Surface Water shows water at least half the time), at least 10 % of a cell | Source price: the YAML's baseline, also a shadow price at lakes; pipeline: TWDB Uniform Costing Model (HDR and Freese and Nichols 2018); Lehner et al. (2025); Pekel et al. (2016) |
| Land rent | the LCOA | 0 everywhere | | |

Inputs are built once from the rasters of `../data/README.md` by `site_geography.py`
(`python -m model.site_geography cells | sites | fit`); `../inputs/site_costs/README.md`
describes them and the fit.
