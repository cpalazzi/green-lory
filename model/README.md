# model/: the green-lory plant model

| Module | Role |
|---|---|
| `main.py` | Builds the PyPSA network of one plant from a plant bundle (`generate_network`), applies the weather profiles and solves it (`main`); solver options and the optimal-termination check |
| `run_global.py` | Runs the plant over many cells: reads the technology YAML (with `extends:`), the finance overrides and the land table, applies the YAML costs to the bundle, solves each cell (with a solver retry chain), and writes one results row per cell. Command-line entry point: `python -m model.run_global --help` |
| `auxiliary.py` | Ammonia ramp constraints (linopy), weather loading and aggregation, result dictionaries |
| `land_processing.py` | Builds land tables (`max_capacities_*.csv`) from MODIS land cover, WDPA protected areas and GEBCO slope, per centred 1-degree cell |
| `land_capacity.py` | Renewable land budget per cell (co-located or exclusive wind/PV land), and the capacity rules (`scaled_reference_design`, `solved_quantity`) |
| `land_union.py` | Names and provenance of the wind/PV land-overlap (union) estimate |
| `result_accounting.py` | Final LCOA accounting: site costs (water, land rent), headline shares, grid-backstop reporting |
| `location_tools.py` | Weather data access (the 2019 NetCDF stack) and cell-to-country attribution |
| `data_store.py`, `constants.py`, `data_paths.py` | Result container, physical constants, default data paths |
| `plot_global_heatmap.py` | LCOA maps, used by the notebooks (`plot_monthly_results.py` is unused) |

Plant bundles (`basic_ammonia_plant*/`, at the repository root for now) and the options and
result columns are described in `../GLOSSARY.md`; the open modelling choices in
`../MODEL_CHOICES.md`.

## Spatial cost inputs

Moved here unchanged from the May 2026 README. These inputs are used only by the spatial
runs (`bspat`, `wspat`), which are deferred until the remoteness and water inputs are reviewed;
the key runs are flat.

Notebook 02 generates per-location, per-technology cost overrides in `inputs/spatial_cost_inputs.csv`. The build cost multiplier adjusts only the **build/installation** portion of CAPEX — equipment costs are unchanged.

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

In `run_global.py`, only the **build** portion of overnight cost is scaled:

```
effective_overnight = tech_cost + (build_cost × build_cost_multiplier)
capital_cost = effective_overnight × (annuity + fixed_O&M_fraction)
```

This means equipment costs are location-invariant; only installation/logistics/civil works scale with geography.
