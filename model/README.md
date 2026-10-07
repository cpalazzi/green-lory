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
| `plant.py` | One plant at one cell: the PyPSA network from the plant bundle, its constraints (Haber-Bosch ramps, battery coupling, land in the solve), the Gurobi solve and its acceptance check, the plant results |
| `accounting.py` | After the solve: cost splits (equipment, installation, O&M, interest), water and land costs in the headline, grid-use classification |
| `land.py` | Land budgets per cell (co-located or exclusive wind and PV land), the capacity rules (`scaled_reference_design`, `solved_quantity`), loading and checking land tables |
| `land_table.py` | Building land tables from MODIS land cover, WDPA protected areas and GEBCO slope (`python -m model.land_table --help`) |
| `surface.py` | Running the plant over many cells, serially or in forked workers, with a numerical retry chain (`python -m model.surface --help`) |
| `runs.py` | Run definitions (`../runs.yaml`) and the execution record (`../run_store.csv`); the `run_surface` arguments of a defined run |
| `results.py` | Reading results by run id: surfaces, network suppliers, rows at the focal cells, coast against inland, maps |
| `suppliers.py` | A QA-passed surface as a supplier table of the green-porpoise network model (`export`), and post-hoc land-share variants (`derive`); the network runs in the green-porpoise repository |
| `cli.py` | The `green-lory` command: `list`, `show`, `table`, `status`, `path`, `run`, `map`, `shell` |

`plants/ammonia_plant_1mt/` is the plant; its README lists what the bundle fixes and what each
run sets from its YAML. Runs are data: `../runs.yaml` gives every setting of every run by id,
and the ARC job `../arc/jobs/01_run_global.sh` passes one run's settings to
`surface.run_surface`; `green-lory run <id> --cell=lat,lon` does the same at a few cells locally. Options and result columns are described in `../GLOSSARY.md`, the open
modelling choices in `../MODEL_CHOICES.md`.

## One cell, step by step (`surface.run_cell`)

1. Copy the base network: the plant bundle with the YAML's link efficiencies and the run's PV
   policy, read once per run.
2. Set the costs: every YAML cost annualised at its default interest rate
   (`technology.apply_base_costs`), then the cell's overrides: its interest rates, and the
   installation part times its build-cost multiplier (`technology.apply_cell_overrides`).
3. Read the cell's land row. With `after_solve` (all campaign runs) the land does not limit the
   design; with `in_solve` it bounds wind and PV inside the optimisation.
4. Solve the plant for the cell's year of weather (`plant.solve`).
5. Complete the accounting: water and land costs, cost shares, grid use (`accounting.py`), and
   the capacity: the 1 Mt/yr design scaled until it fills the cell's land (`land.py`).

## Spatial cost inputs

Moved here unchanged from the May 2026 README. These inputs are used only by the spatial
runs (`bspat`, `wspat`), which are deferred until the remoteness and water inputs are reviewed;
the key runs are flat.

Notebook 02 generates per-location, per-technology cost overrides in `inputs/spatial_cost_inputs.csv`. Campaign step 3 replaces this method (equipment by port, installation by city, slope classes; `../CAMPAIGN_PLAN.md` section 0a). The build cost multiplier adjusts only the **build/installation** portion of CAPEX — equipment costs are unchanged.

### Composite multiplier

```
build_cost_multiplier = depth_mult × remoteness_mult × labour_mult
```

Each factor ≥ 1.0. A cell with shallow water, near a port, and baseline labour costs gets 1.0 × 1.0 × 1.0 = 1.0 (no markup).

### Depth multiplier (offshore only)

Piecewise-linear function of ocean depth, with separate curves for wind (foundation cost) and plant equipment (logistics):

| Depth (m) | Wind | Plant | Regime |
|-----------|------|-------|--------|
| 0 | 1.3 | 1.1 | Shallow / fixed-bottom |
| 60 | 1.8 | 1.3 | Fixed-bottom limit |
| 300 | 2.5 | 1.8 | Floating (semi-sub / spar) |
| 1500 | 3.5 | 2.5 | Deep floating |

Returns 1.0 for onshore cells. Clamped at the last breakpoint beyond 1500 m.

### Remoteness multiplier

**Offshore cells**: linear function of great-circle distance to nearest major port (≥1 Mt/yr throughput). +25% per 1,000 km, extrapolating beyond 2,000 km.

**Onshore cells with travel-time data**: linear function of travel time to nearest city (population ≥50k), from the MAP/Oxford Accessibility to Cities 2015 dataset. Same +25% per 2,000 min scaling. Pre-fetched via `scripts/fetch_travel_time.py` → `data/travel_time_by_cell.csv`.

**Onshore cells without travel-time data** (~2,500 cells: Antarctica, Greenland interior, Arctic): fallback using distance-to-port with a 4× no-road premium, plus an elevation penalty above 1,000 m:

```
no_road_mult  = 1.0 + 4.0 × (port_remoteness − 1.0)
elev_factor   = 1.0 + 0.2 × max(0, elevation_m − 1000) / 1000
remoteness    = no_road_mult × elev_factor
```

### Labour multiplier

Placeholder at 1.0 everywhere. Will accept a per-country index when available.

### Other spatial inputs

| Input | Current default | Notes |
|-------|----------------|-------|
| Interest rate | 10% (uniform) | Per-tech from YAML; awaiting per-country data |
| Water cost | $2.00/m³ (uniform) | Added to LCOA post-solve |
| Land cost | $0/km²/yr | Added to LCOA post-solve |

### How the multiplier enters the model

In `technology.apply_cell_overrides`, only the **build** portion of overnight cost is scaled:

```
effective_overnight = tech_cost + (build_cost × build_cost_multiplier)
capital_cost = effective_overnight × (annuity + fixed_O&M_fraction)
```

This means equipment costs are location-invariant; only installation/logistics/civil works scale with geography.
