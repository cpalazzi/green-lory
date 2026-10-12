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
| `gsw_permanent_water.csv` | For the 134 cells that would hold a water source only through a saline lake: the area where the JRC Global Surface Water occurrence layer shows water in at least half of the observations of 1984-2021, read remotely on 8 October 2026; 109 of them pass the 10 % threshold | the same, kept between runs |
| `calibration_sites.csv` | The sites of both tables, located with the GeoNames gazetteer of Australia (Texas, Kingston SE, Campbelltown and Derwent Valley under their gazetteer names; Cardowan is a homestead, not a town), with their travel times and their distances to the nearest network port and to the state capital's port | `python -m model.site_geography sites` |
| `remoteness_fit.json`, `remoteness_fit.md` | The fit of the equipment and installation factors to each table, and the report with every site's residual | `python -m model.site_geography fit` |
| `site_costs_center_20pct_slope20.csv` | The site costs of every cell with onshore land in the October 20 % land table (`land_center_slope20_20261008_v2`): equipment, installation, slope, depth and labour multipliers, the nearest water source and the water price (USD2020/m3), with GHD's fit | `green-lory site-costs --land <land table> --output <csv>` |

## The fit

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
20,000) and, in Aurecon's, Forsayth (2.67). `model/site_costs.py` uses GHD's fit by default
(`--remoteness-source`), as chosen on 8 October 2026.

## Water

Sources are the sea, permanent lakes and reservoirs covering at least 10 % of a cell, and large
rivers covering at least 1 % (GLWD classes 04-05, added on 9 October 2026). The source price is
the technology YAML's baseline water price (2 EUR2020 per m3 in the campaign YAML), at sea, lake
and river alike; inland it is a shadow price that protects freshwater.
The pipeline from the nearest source is costed with the Texas Water Development Board's Uniform
Costing Model (2018 unit costs, rural soil) for one plant's 1.5 million m3 a year: the
least-cost standard diameter is 305 mm (12 inches; 0.65 m/s), a booster station every 70 km, 30 % (pipe)
and 35 % (stations) engineering and contingencies, 1 % and 2.5 % O&M a year, friction pumping
at 0.08 USD/kWh, 5 % over 45 years (pipe) and 20 years (stations), USD of December 2017
escalated to 2020 with the US CPI: 1.80 USD2020 per m3 and 100 km at multipliers of 1.
