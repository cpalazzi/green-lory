# lcoa-opt 2050 Cost Reference

This archive captures the exact techno-economic input files used by the
external lcoa-opt model for its 2050 run path.

Source model:
- `/Users/carlopalazzi/programming/shipping_sprint/lcoa_model/lcoa-opt`

Direct run-time evidence:
- `main.py`, `main_mp.py`, and `main_test.py` load
  `data/GeneralSteelData.xlsx`.
- They read the `Costs` and `Efficiencies` sheets for the selected year.
- Those series are overlaid onto the baseline `Basic_ammonia_plant/`
  network CSVs by exact equipment name.

Archived files:
- `data/GeneralSteelData.xlsx`
- `Basic_ammonia_plant/generators.csv`
- `Basic_ammonia_plant/links.csv`
- `Basic_ammonia_plant/stores.csv`

Relevant 2050 workbook values from `GeneralSteelData.xlsx`:

Costs sheet, 2050 column, annualised USD 2018:

| Equipment | Value |
| --- | ---: |
| Wind | 93432.935367 |
| Solar | 25724.725467 |
| Electrolysis | 33720.742225 |
| BatteryInterfaceIn | 8915.992859 |
| Battery | 8915.992859 |
| CompressedH2Store | 2524.051083 |
| HydrogenFuelCell | 24919.056966 |
| HB | 565000.000000 |
| Ammonia | 5.760000 |

Efficiencies sheet, 2050 column:

| Equipment | Value |
| --- | ---: |
| Electrolysis | 0.8865088651 |
| HydrogenFuelCell | 0.5400151204 |

Correspondence to green-lory `2050_way` YAMLs:

- Direct numeric correspondence is good for `solar`, `wind`, `electrolysis`,
  `hydrogen_fuel_cell`, `battery_pcs_charge`, `battery_storage`,
  `compressed_hydrogen_store`, `ammonia`, and `ammonia_synthesis` after:
  reversing the lcoa-opt annualisation using `0.121852`, converting from
  USD 2018 to the target currency year, and converting PyPSA link costs from
  lcoa-opt input basis to green-lory output basis where needed.
- `solar_tracking` is not sourced from lcoa-opt. The archived plant has
  `SolarTracking` fixed and effectively unusable, so green-lory derives a
  tracker cost from the DEA tracker/fixed ratio as a gap-fill.
- `hydrogen_compression` is not sourced from the workbook. lcoa-opt keeps only
  a negligible baseline `HydrogenCompression` link CAPEX in the plant CSV,
  while the local `2050_way` configs use the DEA 2050 compression cost as a
  separate green-lory component.
- `battery_pcs_discharge` is not a clean one-to-one match. The lcoa-opt
  workbook has a row for `BatteryInterfaceIn` but not `BatteryInterfaceOut`, so
  the external model retains the baseline `BatteryInterfaceOut` cost from the
  plant CSV. The local `2050_way` configs instead place PCS CAPEX on the charge
  side and set discharge CAPEX to zero.
- `tech_cost_per_*` and `build_cost_per_*` splits are not present in lcoa-opt.
  The local `2050_way` configs import those fractions from the DEA 2050 setup.

Interpretation:

- The local `2050_way` configs are a good reconstruction of the main Way-style
  2050 cost and efficiency assumptions used in lcoa-opt.
- They are not a byte-for-byte replica of the external model because green-lory
  uses a different plant topology and separates some components that lcoa-opt
  leaves bundled or partially implicit.
- The main structural mismatch to keep in mind is the battery and hydrogen
  storage chain: green-lory makes an explicit modeling choice for battery PCS
  discharge and hydrogen compression rather than preserving the external model's
  baseline residual values.