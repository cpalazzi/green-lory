# inputs/site_costs/: the inputs of the site costs

`model/site_costs.py` builds a site-cost table for a land table from these files
(`green-lory site-costs --land <table> --output <csv>`); `model/README.md` lists the cost
mechanisms. Everything here except the three transcribed tables is generated, with a
`*_provenance.json` beside it giving the inputs' hashes: the site-cost table by
`model/site_costs.py`, the rest by `model/site_geography.py` from the data of
`../../data/README.md`.

| File | What it is | Written by |
|---|---|---|
| `aurecon2024_location_factors.csv` | Aurecon (2024), *2024 Energy Technology Cost and Technical Parameter Review*, revision 3 (6 December 2024), for AEMO: Table 11-1, the location factors of 77 sites in the NEM states, transcribed by hand on 7 October 2026. Every row passes the table's own arithmetic (installation = (location - 0.6 x equipment) / 0.4 within rounding, O&M = (location + installation) / 2) | hand |
| `ghd2025_location_factors.csv` | GHD (2025), *2025 Energy Technology Cost and Technical Parameter Review*, draft report Rev C (28 November 2025), for AEMO's draft 2026 Forecasting Assumptions Update: Table 8.3, the factors of 110 sites, transcribed on 7 October 2026. Equipment factors by distance from a major port (1.00, 1.05, 1.10); installation factors from Rawlinsons (2025) regional cost maps, including each capital's price level | hand |
| `ghd2025_capital_installation_factors.csv` | GHD (2025) Table 8.2: the capital cities' installation factors relative to Melbourne | hand |
| `cell_geography.csv` | Every 1-degree cell from 55 S to 83 N: travel time to the nearest city of 5,000 to 5 million people (median over the cell's land pixels), distance to the nearest network port, sea and inland-water areas by class, whether the cell holds a water source (sea and lakes at least 10 % of it, or large rivers at least 1 %) and the distance to the nearest one (also with every saline lake, with none, and without the rivers) | `python -m model.site_geography cells` |
| `port_travel_times.csv` | The 1,377 ports of the shipping network with their travel times to a city of 5,000 to 5 million people, at the port. Eight ports have none, because their coordinates lie more than 3 pixels from a land pixel on the raster (among them Manila and Jingtang); the next nearest port stands in for them | `python -m model.site_geography ports` |
| `cell_elevation.csv` | Every 1-degree cell, 90 S to 89 N: the mean elevation of its GEBCO 2025 pixels at or above 0 m (`land_elevation_m`, for the cold rule) and the mean depth of those below (`sea_depth_m`, for a sea plant's depth class). The sub-ice grid, so bedrock under the ice sheets, which lie beyond 60 degrees anyway | `python -m model.site_geography elevation` |
| `protected_areas.csv` | Every 1-degree cell, 89 S to 89 N: its land and water as the land tables count them (`land_km2`, `water_km2`, MODIS MCD12C1), the parts of them in protected areas (`protected_land_km2`, `protected_water_km2`: the WDPA February 2026 polygons of `land.protected_area_statuses`, in the terrestrial and coastal realms on land and the marine and coastal realms on water, `../parameters.yaml`) and their shares (`protected_land_pct`, `protected_water_pct`). The land tables leave the protected land out of the land a plant may use, the sea runs the protected water out of the sea (`model/protected_areas.py`) | `python -m model.site_geography protected-areas` |
| `sea_zones.csv` | Every 1-degree cell: `eez_iso3` and `eez_type`, the sovereign and type of the exclusive economic zone; `eez_source`, `centre` if the zone holds the centre, `nearest` if the nearest zone reaches a coastal cell whose centre is on land, `none` outside every zone (international waters, Antarctica's coast among them) | `python -m model.site_geography sea` |
| `gsw_permanent_water.csv` | For the 134 cells that would hold a water source only through a saline lake: the area where the JRC Global Surface Water occurrence layer shows water in at least half of the observations of 1984-2021, read remotely on 8 October 2026; 109 of them pass the 10 % threshold | the same, kept between runs |
| `calibration_sites.csv` | The sites of both tables, located with the GeoNames gazetteer of Australia (Texas, Kingston SE, Campbelltown and Derwent Valley under their gazetteer names; Cardowan is a homestead, not a town), with their travel times and their distances to the nearest network port and to the state capital's port | `python -m model.site_geography sites` |
| `remoteness_fit.json`, `remoteness_fit.md` | The fit of the equipment and installation factors to each table, and the report with every site's residual: a check since 3 November 2026, not an input | `python -m model.site_geography fit` |
| `site_costs_center_20pct_slope20.csv` | The site costs of every cell with onshore land in the October 20 % land table (`land_center_slope20_20261008_v2`): travel time and its source (`travel_time_source`: `raster` or `port`), the equipment, installation, cold, slope, depth and labour multipliers, the nearest water source and the water price (USD2020/m3). `plant_multiplier`, `wind_equipment_ratio` and `wind_build_ratio` are 1 | `green-lory site-costs --land <land table> --output <csv>` |
| `site_costs_sea_20pct.csv` | The same columns for the 40,197 sea plants of a sea run, with `sea_depth_m`, `depth_class` and `eez_iso3` (see below) | `python -m model.sea_cells` |
| `site_costs_sea_flat_20pct.csv` | The flat runs' table at sea (`bflat`): the same rows with the depth-class costs (`plant_multiplier`, `wind_equipment_ratio`, `wind_build_ratio`) and every location factor 1 | `python -m model.sea_cells --flat-site-costs-output` |

## The remote factor

One rule for every point (`site_costs.remote_factor` in `../parameters.yaml`, decided on 3
November 2026). With d the travel time in days from a city of 20,000:

1. Build factor (`installation_multiplier`) = 1 + 1.127 log2(1 + d); equipment factor
   (`equipment_cost_multiplier`) = 1 + 0.676 log2(1 + d). Both pass through Shamattawa, Manitoba
   (0.85 days on the raster), rounded from the DIAND manual's x1.99 and its transportation
   factor 1.60: x2 and x1.6.
2. d is the median over the cell's land pixels where the raster has a value. Otherwise it is the
   travel time of the nearest port that has one, plus the distance to that port at 20 km/h.
3. At sea (offshore cells, not yet in these tables) d always runs through the port, and only the
   distance beyond 175 km counts, because DEA's offshore costs include marine installation that
   far.

Nothing is fitted beyond the anchor. Against the Australian factors, which the rule did not use,
the build factor is 0.04 below GHD's on average (RMSE 0.087) and 0.19 below Aurecon's, and the
equipment factor 0.01 above Aurecon's (`tests/test_site_costs.py`).

## Plants at sea

A sea run solves the land runs' plant with wind at sea and no PV, so it differs only in its data:
its land table, its weather stack and `site_costs_sea_20pct.csv` (`model/sea_cells.py`). Its rows:

1. A cell holds a sea plant if at least 1 % of it is sea and it has a sea depth. Frozen sea
   counts as sea, and the cold rule prices it. A coastal cell can so hold a land plant in the
   land run and a sea plant in the sea run, each on its own area. The plant may use the cell's
   sea outside protected areas (less the protected share of its water, `protected_areas.csv`), at
   the land table's share; sea runs built before 11 November 2026 counted the protected sea too.
2. `depth_class` follows the mean depth: `fixed` up to 70 m, `floating` to 1,000 m, `deep` beyond.
3. Wind (`offshore` in `../parameters.yaml`): the turbine is the equipment part, and the
   foundation, array cables and development the build part, without DEA's substation and export
   cable. `wind_equipment_ratio` and `wind_build_ratio` are these over the technology YAML's
   onshore costs. In deep water both are 3.
4. Every other technology takes `plant_multiplier`: 1.72, 2 or 3 by class.
5. The remote factor uses the nearest port's travel time plus the sea leg beyond 175 km. The cold
   rule applies by latitude. Water is desalinated at the source price with no pipeline.

## The fit (a check)

Equipment factor = 1 + a x distance, capped at the table's largest factor; installation factor
= 1 + b x min(travel time to the nearest city of at least N people, saturation). The
installation factors of GHD are divided by their capital's factor first, so that both tables
measure remoteness only. N and the saturation are chosen by leave-one-out cross-validation over
nine city sizes and saturations every 30 minutes; b by least squares through 1 at the city.

| Table | Equipment | Installation | Installation fit |
|---|---|---|---|
| GHD 2025 (110 sites) | 1 + 0.204 per 1,000 km to the nearest network port, at most 1.10 (RMSE 0.032) | 1 + 0.0855 per hour to a city of 20,000, saturating at 4 hours (1.34) | RMSE 0.070, R2 0.52 |
| Aurecon 2024 (77 sites) | 1 + 0.096 per 1,000 km to the state capital's port, at most 1.15 (RMSE 0.006) | 1 + 0.201 per hour to a city of 20,000, saturating at 4 hours (1.80) | RMSE 0.167, R2 0.51 |

Both tables choose the same city size and saturation. The installation factors differ by a
factor of about two because Aurecon derives them as the residual of an overall location factor
(installation = (location - 0.6 x equipment) / 0.4, which multiplies the regional uplift by
2.5; its own note 1 says some "appear to be high"), while GHD takes Rawlinsons' regional uplift
directly. The largest misses of both fits are regional cities far from their capital
(Townsville, Mackay, Rockhampton: 1.2 to 1.36 in Aurecon's table at no travel time to a town of
20,000) and, in Aurecon's, Forsayth (2.67). `model/site_costs.py` used GHD's fit from 8
October to 3 November 2026; it stopped at 4 hours, and the remotest site is 6.7 hours from a city.

## Water

Sources are the sea, permanent lakes and reservoirs covering at least 10 % of a cell, and large
rivers covering at least 1 % (GLWD classes 04-05, added on 9 October 2026). The source price is
the technology YAML's baseline water price (2 EUR2020 per m3 in the campaign YAML), at sea, lake
and river alike; inland it is a shadow price that protects freshwater.
The pipeline from the nearest source is costed with the Texas Water Development Board's Uniform
Costing Model (2018 unit costs, rural soil) for one plant's 1.62 million m3 a year: the
least-cost standard diameter is 305 mm (12 inches; 0.70 m/s), a booster station every 70 km, 30 % (pipe)
and 35 % (stations) engineering and contingencies, 1 % and 2.5 % O&M a year, friction pumping
at 0.08 USD/kWh, 5 % over 45 years (pipe) and 20 years (stations): 1.72 USD2020 per m3 and
100 km at multipliers of 1. The dollars (December 2017 for the unit costs, taken as 2017; 2018
for the power cost) are converted by the shared currency rule (`currency` in
`inputs/parameters.yaml`: the price year's ECB rate, then euro-area inflation to 2020) and
stated in USD2020 at the 2020 rate, the site-cost currency. Until 30 October 2026 the unit costs
were escalated with the US CPI and the power cost was not converted: 1.73 (1.80 at the
1.5 million m3 a year used until 15 October 2026).
