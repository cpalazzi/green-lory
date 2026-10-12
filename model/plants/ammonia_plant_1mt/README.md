# ammonia_plant_1mt: the plant every green-lory run solves

A PyPSA network in CSV form: the components of a green ammonia plant and how they connect. It
holds the plant's structure only. A run reads it once (`model/plant.py`, `build_network`) and
then sets from its technology YAML (`model/technology.py`, `prepare_base_network` and, per cell,
the costs):

- every capital cost, from the annualised overnight cost;
- every link's conversion coefficients (`efficiency`, `efficiency2`), from the YAML's
  `carriers_in`/`carriers_out` recipe;
- the operating values a technology states: the ammonia synthesis minimum load and ramp limits
  and the battery's standing loss.

`build_network` sets the constant ammonia demand from the reference output and the ammonia
heating value (`inputs/parameters.yaml`: 1 Mt/yr, 713.47 MW at 6.25 MWh/t), and the run's
`pv_policy` in `runs.yaml` decides whether tracking PV may be built (`fixed` switches it off).
`prepare_base_network` refuses a YAML that leaves any component other than the formulation
devices without its cost or a link without its recipe.

What the bundle itself fixes:

| File | Content |
|---|---|
| `buses.csv` | power, hydrogen, hydrogen storage, battery, ammonia, and the two ramp-penalty helper buses |
| `generators.csv` | wind, fixed PV and tracking PV (extendable); grid (off: every plant is off-grid); the ramp dummy |
| `links.csv` | electrolysis; hydrogen compression into storage (its power draw is the third port) and withdrawal; ammonia synthesis with its air separation unit (hydrogen is the third port); battery charger and discharger; fuel cell; the ramp penalty |
| `stores.csv` | ammonia (output tracker), compressed hydrogen, battery, penalty integrator |
| `loads.csv` | the constant ammonia demand at the ammonia bus |
| `network.csv` | the network name and the PyPSA version it was written with |

The numbers left in the bundle belong to the formulation, not to any technology: the switched-off
grid; the ramp dummy, penalty link and penalty store, a relief valve priced at 100,000 EUR/MWh so
that a plant uses it only if it could not run otherwise (results report the energy through it);
a 0.0001 EUR/MWh cost on hydrogen compression, too small to change any cost (no recorded
purpose; it looks like a tie-breaker); capacity bounds of 10^6 to 10^7 MW (MWh) that never bind;
and the stores' starting levels, which cyclic operation overrides (the penalty store starts at 0).

Until 7 October 2026 there were three bundles, which differed only in four efficiencies (DEA
or Way 2050) and in whether tracking PV was allowed. Until 12 October 2026 the bundle also held
copies of the costs and efficiencies (overwritten by every run), the minimum load and ramp
limits, an electrolysis variable cost of 0.537 EUR/MWh with no known source (dropped) and a
battery standing loss of 0.001 that PyPSA applied per hour (2.4 % a day; the YAML's 0.1 % a day
is now applied).
