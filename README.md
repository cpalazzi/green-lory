# Green Lory – Global LCOA Model

Green Lory is a PyPSA-based optimisation workflow for sizing green ammonia (and hydrogen) plants and mapping the resulting levelised cost of ammonia (LCOA) across the globe. The codebase can run a single-location design study or sweep over thousands of coordinates to produce a global heatmap.

## Repository layout
- `main.py` – serial driver for single-site sizing; exposes `main()` and the `generate_network()` helper.
- `run_global.py` – orchestration utilities for looping over many coordinates (global sweeps, scenario batches).
- `basic_ammonia_plant/` – canonical PyPSA CSV bundle (`buses.csv`, `generators.csv`, `links.csv`, `loads.csv`, `stores.csv`). This is the starting plant design that each scenario imports.
- `data/` – weather files (NetCDF) and auxiliary CSV/HDF assets consumed by `location_tools` and `data_store`.
- `auxiliary.py`, `location_tools.py`, `data_store.py`, `plot_monthly_results.py`, `reformat_files.py`, `storage_cost_comparison.py` – helper modules for weather wrangling, constraint injection, plotting, and archival scripts.
- `results/` – empty staging area for freshly generated outputs (historic artefacts have been purged).
- `notebooks/` – smoke tests and exploratory notebooks (e.g., `notebooks/dev_smoke_tests.ipynb`).

### Editing the base plant
PyPSA consumes the canonical CSV bundle in `basic_ammonia_plant/`. The project no longer applies a YAML tech config at runtime.

To update techno-economic assumptions, edit `inputs/tech_config_ammonia_plant.yaml` (overnight CAPEX, lifetimes, interest rates, efficiencies) and run `notebooks/00_tech_config.ipynb` to generate the PyPSA-ready CSV tables (annualised `capital_cost`, link cost basis conversion, updated efficiencies). The most important CSV headers are:

- `buses.csv`: `name` (bus id), `carrier` (AC, Hydrogen, Ammonia, …), `control` (Slack/PQ), optional `generator` reference when a bus hosts a fixed generator, plus a free-form `comment` column for documentation.
- `generators.csv`: `name`, `bus`, `control`, `p_nom` (initial capacity), `p_nom_extendable` (bool), `marginal_cost`, `capital_cost`, and `comment`. Add any other PyPSA generator attribute as another column if needed.
- `links.csv`: `name`, `bus0`, `bus1`, optional `bus2` (or higher), `carrier`, `efficiency`/`efficiency2`, `p_nom_extendable`, `p_nom_max`, `p_min_pu`, `capital_cost`, optional `ramp_limit_{up,down}`, and a `comment`. Empty cells simply leave an attribute at its PyPSA default.
- `loads.csv`: `name`, `bus`, `p_set`, `comment` – perfect for the ammonia off‑take constraint.
- `stores.csv`: `name`, `bus`, `carrier`, `e_nom_extendable`, `e_nom_max`, `e_initial`, `e_cyclic`, `capital_cost`, `comment`.

#### Units and basis (important)
This project treats **techno-economic inputs and reported outputs** on an **HHV output basis** (shorthand: `MW_out_HHV` / `MWh_HHV`).

- **YAML inputs (`inputs/tech_config_ammonia_plant.yaml`)**
	- Generators: `overnight_cost_per_mw` is per **MW_out (HHV)**
	- Stores: `overnight_cost_per_mwh` is per **MWh (HHV)**
	- Links: `overnight_cost_per_mw` is per **MW_out on bus1 (HHV)**

- **PyPSA internal basis (CSV bundle in `basic_ammonia_plant/`)**
	- Generators: `p_nom` / `p_nom_opt` are **MW_out**
	- Stores: `e_nom` / `e_nom_opt` are **MWh**
	- Links: `p_nom` / `p_nom_opt` are **MW_in on bus0** (PyPSA convention)
		- Therefore link `capital_cost` in `links.csv` must be **USD/MW_in/year** to be consistent.

`notebooks/00_tech_config.ipynb` handles the conversion you want: it takes YAML link CAPEX quoted per **MW_out** and writes `links.csv.capital_cost` per **MW_in** using the link efficiency.

- **Model outputs (saved CSVs / displayed results)**
	- Link installed capacities are reported as **MW_out on bus1 (HHV)** so they are directly comparable across technologies.

Because PyPSA reads CSV headers literally, stick to the official attribute names listed in the [PyPSA component tables](https://pypsa.readthedocs.io/en/latest/components.html). Any extra column is ignored unless PyPSA knows how to map it, so documentation text belongs in the `comment` column rather than custom headers. When you need a new component, duplicate the relevant row, change the `name`, and adjust costs/limits; no auxiliary builder step is required. Keep every bus, generator, link, load, and store `name` in snake_case (e.g., `hydrogen_compression`) so the tech config and downstream scripts can match them reliably.

### CSV glossary
Quick reference for the headers we keep touching in the PyPSA CSV bundle. Unless stated otherwise, values are on the HHV basis described above; note that PyPSA link `p_nom` is MW_in on bus0 even though we report link capacities as MW_out_HHV in exported results.

| Label | Applies to | Definition | Units |
| --- | --- | --- | --- |
| `name` | All component tables | Unique identifier referenced by the YAML cost map and helper scripts; keep it in snake_case. | - |
| `carrier` | All component tables | Energy-carrier grouping that PyPSA uses for reporting and constraint tagging. | - |
| `bus`, `bus0`, `bus1`, `bus2` | Generators/loads/stores/links | Electrical or process bus connection; links use numbered terminals (bus0 = input, bus1 = primary output, bus2 = optional secondary port). | - |
| `control` | `buses.csv`, `generators.csv` | Power-flow control mode (`Slack`, `PQ`, etc.) enforced by PyPSA. | - |
| `p_nom` | `generators.csv`, `links.csv` | Installed nameplate rating that acts as the starting point for optimisation. | MW |
| `p_nom_extendable` | `generators.csv`, `links.csv` | Boolean switch that lets PyPSA size the component (`True`) or keep it fixed. | boolean |
| `p_nom_max` / `p_nom_min` | `generators.csv`, `links.csv` | Hard upper/lower bounds applied to the optimised `p_nom`. | MW |
| `*_pu` suffix | Many columns | “Per-unit”: a fraction of the component’s nominal rating (typically `p_nom`). Example: `p_min_pu = 0.2` means the dispatch must stay above 20% of `p_nom` each snapshot. | per-unit |
| `p_min_pu` / `p_max_pu` | `generators.csv`, `links.csv`, `generators_t/links_t` | Snapshot min/max dispatch expressed as a per-unit fraction of `p_nom`; 0–1 for typical renewables. | per-unit |
| `marginal_cost` | `generators.csv`, `links.csv` | Variable operating cost that multiplies dispatched energy. | USD/MWh (HHV) |
| `capital_cost` | `generators.csv`, `links.csv`, `stores.csv` | Annualised cost that multiplies installed capacity. For generators: USD/MW_out/year. For stores: USD/MWh/year. For links (PyPSA): USD/MW_in/year (because `p_nom` is MW_in on bus0). | USD/MW/year or USD/MWh/year |
| `efficiency`, `efficiency2` | `links.csv` | Conversion ratios from bus0 to bus1 (and bus2); negative values flag consumption on that port. | dimensionless |
| `p_set` | `loads.csv` | Fixed demand profile that the optimisation must meet each snapshot. | MW |
| `e_nom_extendable` | `stores.csv` | Boolean signalling whether the storage volume may expand. | boolean |
| `e_nom_max` / `e_nom_min` | `stores.csv` | Energy-capacity bounds applied when the store is extendable. | MWh |
| `e_initial` | `stores.csv` | Initial state of charge at the first snapshot. | MWh |
| `e_cyclic` | `stores.csv` | Forces the final state of charge to equal the initial value when set to `True`. | boolean |
| `ramp_limit_up` / `ramp_limit_down` | `links.csv` | Per-unit ramp-rate guardrails relative to `p_nom` that the Linopy constraints enforce snapshot-to-snapshot. | ΔMW/MW per snapshot |
| `comment` | All component tables | Free-form documentation; ignored by PyPSA but priceless for collaborators. | text |

PyPSA-derived result tables append `_opt` fields (e.g., `p_nom_opt`, `e_nom_opt`) after each solve.

For **exported results CSVs** produced by this repo (global sweeps and notebooks), link capacities are converted and reported as **MW_out on bus1 (HHV)** for readability.

- Default CAPEX/interest assumptions live in `inputs/tech_config_ammonia_plant.yaml` (overnight cost, `lifetime_years`, `interest_rate`, `fixed_om_fraction`, plus link `carriers_in`/`carriers_out` recipes when needed).
- `notebooks/00_tech_config.ipynb` converts those overnight assumptions into annualised PyPSA `capital_cost` values using the annuity payment
	$\text{Annuity}(r, n) = \frac{r(1+r)^n}{(1+r)^n - 1}$
	and writes updated `basic_ammonia_plant/*.csv` tables.
- Topology (buses, links, connectivity) still originates from the CSV bundle, so editing the YAML only changes the techno-economic inputs unless you add/remove assets.

#### Link carrier recipes (`carriers_in` / `carriers_out`)
Some links have multiple inputs/outputs (e.g., ammonia synthesis consumes both power and hydrogen). For these, the YAML describes the link using bus-based “recipes”:

- `carriers_in`: mapping of **bus names** to required **inputs per 1 unit of bus1 output**
- `carriers_out`: mapping of **bus names** to produced **outputs per 1 unit of bus1 output**

`notebooks/00_tech_config.ipynb` converts this into PyPSA link coefficients:

- `efficiency` is the bus0→bus1 ratio implied by the recipe
- `efficiency2` (and additional ports if used) capture secondary flows; negative values indicate consumption on that port

The notebook also cross-checks that the written coefficients reproduce the recipe ratios.

### Process coupling and ramping
- **Linopy guardrails** – `main.main()` always calls PyPSA with `extra_functionality=auxiliary.linopy_constraints`, so every optimisation enforces three families of physical constraints: (1) the battery charger/discharger links (`battery_interface_in` and `battery_interface_out`) must share the same install capacity after accounting for their round-trip efficiency; (2) the hydrogen buffer cannot discharge faster than its energy content would allow because the discharger capacity is tied to `compressed_hydrogen_store` via a fixed cycling factor; and (3) the ammonia synthesis block ramp rates (`ramp_limit_up`/`ramp_limit_down` columns in `basic_ammonia_plant/links.csv`) apply snapshot-to-snapshot, preventing unrealistically fast step changes. The same logic has an operating-mode counterpart (`linopy_operating_constraints`) that reuses the fixed capacities from a prior design solve.
- **Hydrogen compression loop** – `hydrogen_compression` represents the power draw needed to push hydrogen into the buffer, while `hydrogen_from_storage` is the low-lift withdrawal leg that simply meters hydrogen back into the process stream. Their capital costs/annuities are set via `inputs/tech_config_ammonia_plant.yaml` and written into the CSV bundle by `notebooks/00_tech_config.ipynb`.
- **Fixed hydrogen/power ratios** – The `ammonia_synthesis` entry in `basic_ammonia_plant/links.csv` is a three-terminal PyPSA link. Bus0 (`power`) draws the electrical load, bus2 (`hydrogen`) withdraws feedstock, and bus1 (`ammonia`) receives the product. PyPSA enforces
	$p_{ammonia} = -\eta_1 \cdot p_{power}$ and $p_{hydrogen} = -\eta_2 \cdot p_{power}$,
	so the `efficiency` (`\eta_1 = 6.25`) and `efficiency2` (`\eta_2 = 7.092`) coefficients lock in the stoichiometric/energy ratios. In practice this means every tonne per hour of ammonia dispatched consumes a fixed amount of power and hydrogen, and no optimisation setting can vary that coupling unless you rewrite the CSV row. The negative sign on `efficiency2` simply tells PyPSA that hydrogen is consumed (not produced) in proportion to the link flow.
## Prerequisites
1. **Python**: target Python 3.11+.
2. **Environment (recommended: `.venv`)**:
	```bash
	python -m venv .venv
	source .venv/bin/activate
	python -m pip install --upgrade pip
	pip install -r requirements.txt
	```
	In VS Code: select the interpreter at `.venv/bin/python` (so notebooks and scripts share the same dependencies).
3. **Core libraries**: managed via `requirements.txt` (PyPSA + geo stack + xarray + plotting).
4. **Solvers**:
	- Default runs target Linopy-compatible solvers (e.g. HiGHS; set `GREEN_LORY_SOLVER=highs`).
	- Gurobi is recommended for large/global sweeps; install with `pip install gurobipy` and ensure your license is configured.
5. **System packages**: macOS/Linux may need GDAL/GEOS/PROJ prerequisites for GeoPandas, depending on how wheels resolve on your machine.

## Preparing inputs
- **Weather data**: `location_tools.all_locations()` stitches every `Solar*.nc`, `SolarTracking*.nc`, and `WindPowers*.nc` file under `data/` into three continuous xarray objects. The bundled NetCDF stacks in this repo are **hourly for 2019** (8760 timestamps from 2019-01-01 00:00 through 2019-12-31 23:00). You can drop in additional longitude slices matching the same time axis and they will be merged automatically. For single-site studies you can still supply your own CSV via `aux.get_weather_data`.
- **Generator list**: every column in your weather file must match an entry in `basic_ammonia_plant/generators.csv`. Remove rows or set profiles to zero if a technology is unavailable.
- **Costs & efficiencies**: edit `inputs/tech_config_ammonia_plant.yaml`, then run `notebooks/00_tech_config.ipynb` to write annualised `capital_cost` values and updated efficiencies into `basic_ammonia_plant/*.csv`.
- **Discounting**: the notebook converts overnight CAPEX into annuity-based `capital_cost` values via each entry's `interest_rate` and `lifetime_years`.
- **Land availability table**: `model/land_processing.py` writes `data/*_land_max_capacity.csv` with per-degree siting metadata (wind/solar availability, km² of usable land, an `onshore_land_pct` derived directly from the MODIS open-water share, and the resulting MW caps). Every column is lowercase (`latitude`, `longitude`, `max_capacity`, …) to simplify downstream joins. Wind capacity density defaults to the Salmon et al. (2021) wake-limited spacing of 200 km²/GW (≈5 MW/km²); solar follows the van de Ven et al. (2021) packing rule using the First Solar (2018) module geometry so higher latitudes automatically receive lower ground coverage ratios. If you need different spacing assumptions edit `_wind_density()` / `_solar_density()` directly before regenerating the CSV. The workflow always re-aggregates the MODIS HDF input—CSV shortcuts were removed—and will fall back to `pyhdf` if your netCDF4 build cannot read HDF4.

## Running scenarios
### Single-location optimisation
```python
from main import main

# Assumes basic_ammonia_plant/ contains the design and weather CSV lives under data/weather_case.csv
o = main(file_name='data/weather_case.csv', get_complete_output=True, extension='_test_case')
```
- If `file_name` is supplied, the helper `aux.get_weather_data` loads the CSV and the script will prompt for solver preferences when not in multi-site mode.
- Set unusable technologies to prohibitive `capital_cost` values or remove them from the CSV bundle.
- Runs default to `gurobi`; set `GREEN_LORY_SOLVER=highs` (or another Linopy-compatible solver) before invoking Python if you need a different backend locally.

### Global heatmap (serial)
```bash
python main.py   # defaults to run_global(2050)
```
- Adjust `run_global` to change the year, longitude window, aggregation (`time_step`), or output filename.
- Produces a `*_lcoa_global_<lon window>long_<timestamp>.csv` file summarising LCOA, sizing, and binding constraints per coordinate.

### Global heatmap (multiprocessing)
```bash
python main_mp.py
```
- Uses a simple multiprocessing pool (`num_processes` constant) and writes `*_lcoa_global_YYYYMMDD_<lon window>mp.csv`.
- A `tqdm` progress bar helps monitor long jobs; redirect output to a log file if running under SLURM.

## Results handling
- New results should be written to `results/` (now empty). `aux.write_results_to_excel` can dump Excel workbooks for detailed inspection; global sweeps save aggregated CSVs.
- If you need to keep historical artefacts, place them under `results/archive/` (not yet created) or external storage before committing new runs.

### What `accumulated_penalty` represents
`accumulated_penalty` is the dedicated store fed by the `penalty_link` in `basic_ammonia_plant/links.csv`. Both components now start at zero capacity but are extendable with prohibitively high annualised CAPEX set in `inputs/tech_config_ammonia_plant.yaml` and written into the CSV bundle via `notebooks/00_tech_config.ipynb`.

## Capital cost roadmap
- Current layer uses a notebook to translate `inputs/tech_config_ammonia_plant.yaml` into the CSV bundle; next up is adding per-region/per-carrier financing inputs plus scenario presets so `interest_rate` and `lifetime_years` can vary spatially.
- The legacy Operating Guide stays retired—this README is the canonical reference and will continue to track future CLI/config improvements.

## Troubleshooting
- **Missing solver**: install/configure a Linopy-compatible solver (HiGHS recommended via `GREEN_LORY_SOLVER=highs`) or configure Gurobi’s license and `gurobi_cl` binary.
- **Geometry errors**: ensure GeoPandas stack is built against compatible GDAL/GEOS libs; recreating the environment usually fixes `fiona` import issues.
- **Country boundaries**: `run_global` requires `data/countries.geojson` (Natural Earth admin-0 polygons or equivalent). Drop the GeoJSON into `data/` before running multi-location sweeps so the output can tag each coordinate with a country.
- **Weather/profile mismatches**: any generator lacking a profile (or vice versa) raises an explicit error during `main()`.
- **Long runtimes**: reduce `time_step` aggregation (e.g., 4 → 6) or limit the lat/lon ranges before committing to a full sweep.

## Next steps
1. Keep `requirements.txt` current and add a minimal, curated `requirements.in` when dependencies stabilise.
2. Add a formal CLI (e.g., `python -m green_lory run --config config.yaml`) so single-site vs global runs share one entry point.
3. Implement per-component, per-location cost-of-capital inputs and feed them into the PyPSA capital cost overrides.
4. Expand automated tests (pytest) to cover CAPEX scaling, multi-site aggregation, and datastore integrity.
