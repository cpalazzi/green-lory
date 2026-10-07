# ammonia_plant_1mt: the plant every green-lory run solves

A PyPSA network in CSV form: the components of a green ammonia plant making 1 Mt/yr and how
they connect. A run reads it once (`model/plant.py`, `build_network`) and then sets from its
technology YAML (`model/technology.py`):

- every capital cost, from the annualised overnight cost (the costs in these files are
  placeholders);
- every link's conversion coefficients (`efficiency`, `efficiency2`), from the YAML's
  `carriers_in`/`carriers_out` recipe (`apply_link_efficiencies`);

and from the run's `pv_policy` in `runs.yaml` whether tracking PV may be built (`fixed`
switches it off).

What the bundle itself fixes:

| File | Content |
|---|---|
| `buses.csv` | power, hydrogen, hydrogen storage, battery, ammonia, and the two ramp-penalty helper buses |
| `generators.csv` | wind, fixed PV and tracking PV (extendable); grid (off: every plant is off-grid); the ramp dummy |
| `links.csv` | electrolysis; hydrogen compression into storage (0.05 MW of power per MW of hydrogen) and withdrawal; ammonia synthesis with its air separation unit (per MW of power, 6.25 MW of ammonia from 7.092 MW of hydrogen; 30 % minimum load; ramps of 40 % of capacity per hour); battery charger and discharger; fuel cell; the ramp penalty |
| `stores.csv` | ammonia (output tracker), compressed hydrogen, battery (0.1 % a day standing loss), penalty integrator |
| `loads.csv` | the constant ammonia output: 713.47 MW, which is 1 Mt/yr at 6.25 MWh/t |
| `network.csv` | the network name and the PyPSA version it was written with |

Until 7 October 2026 there were three bundles, which differed only in four efficiencies (DEA
or Way 2050) and in whether tracking PV was allowed. Git history keeps them.
