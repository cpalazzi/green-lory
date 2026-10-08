# Remoteness factors fitted to the AEMO location-factor tables

Written by `python -m model.site_geography fit`; read with `model/site_costs.py`.

## aurecon2024 (77 sites)

- Equipment: 1 + 0.0959 per 1,000 km of capital_port_distance_km, capped at 1.15; RMSE 0.006.
- Installation (installation cost factor of Table 11-1 (relative to the state capital)): 1 + 0.2007 per hour to a city of at least 20k people, saturating at 240 min (factor 1.80); RMSE 0.167, leave-one-out RMSE 0.174.

Best saturation for each city size:

| city size | saturation (min) | b per hour | factor at saturation | RMSE | leave-one-out RMSE |
|---|---|---|---|---|---|
| 5k | 210 | 0.2760 | 1.97 | 0.199 | 0.207 |
| 10k | 210 | 0.2181 | 1.76 | 0.205 | 0.214 |
| 20k | 240 | 0.2007 | 1.80 | 0.167 | 0.174 |
| 50k | 270 | 0.1124 | 1.51 | 0.197 | 0.202 |
| 100k | 270 | 0.0927 | 1.42 | 0.209 | 0.214 |
| 200k | 300 | 0.0786 | 1.39 | 0.214 | 0.219 |
| 500k | 300 | 0.0773 | 1.39 | 0.214 | 0.219 |
| 1m | 300 | 0.0773 | 1.39 | 0.214 | 0.219 |
| 5m | 2490 | 0.0065 | 1.27 | 0.239 | 0.242 |

| site | state | travel (min) | installation observed | fitted | equipment observed | fitted |
|---|---|---|---|---|---|---|
| Cooktown | QLD | 119 | 1.65 | 1.40 | 1.15 | 1.15 |
| Port Douglas | QLD | 24 | 1.10 | 1.08 | 1.14 | 1.14 |
| Cairns | QLD | 0 | 1.03 | 1.00 | 1.13 | 1.13 |
| Innisfail | QLD | 45 | 1.25 | 1.15 | 1.12 | 1.13 |
| Forsayth | QLD | 243 | 2.67 | 1.80 | 1.14 | 1.13 |
| Ingham | QLD | 64 | 1.28 | 1.21 | 1.11 | 1.12 |
| Townsville | QLD | 0 | 1.22 | 1.00 | 1.11 | 1.11 |
| Bowen | QLD | 114 | 1.32 | 1.38 | 1.09 | 1.09 |
| Mackay | QLD | 0 | 1.31 | 1.00 | 1.08 | 1.08 |
| Moranbah | QLD | 129 | 1.38 | 1.43 | 1.08 | 1.08 |
| Longreach | QLD | 403 | 1.74 | 1.80 | 1.09 | 1.10 |
| Rockhampton | QLD | 0 | 1.36 | 1.00 | 1.05 | 1.05 |
| Bundaberg | QLD | 0 | 1.13 | 1.00 | 1.03 | 1.03 |
| Gympie | QLD | 34 | 1.10 | 1.11 | 1.02 | 1.01 |
| Nambour | QLD | 5 | 1.04 | 1.02 | 1.01 | 1.01 |
| Maroochydore | QLD | 0 | 1.04 | 1.00 | 1.01 | 1.01 |
| Dalby | QLD | 47 | 1.12 | 1.16 | 1.02 | 1.02 |
| Toowoomba | QLD | 0 | 1.04 | 1.00 | 1.01 | 1.01 |
| Warwick | QLD | 49 | 1.13 | 1.16 | 1.02 | 1.01 |
| Inglewood | QLD | 88 | 1.22 | 1.29 | 1.02 | 1.02 |
| Texas | QLD | 121 | 1.34 | 1.40 | 1.02 | 1.02 |
| Banana | QLD | 89 | 1.36 | 1.30 | 1.05 | 1.04 |
| Moree | NSW | 155 | 1.26 | 1.52 | 1.05 | 1.05 |
| Inverell | NSW | 73 | 1.31 | 1.24 | 1.05 | 1.04 |
| Narrabri | NSW | 102 | 1.32 | 1.34 | 1.04 | 1.04 |
| Gunnedah | NSW | 45 | 1.34 | 1.15 | 1.02 | 1.03 |
| Tamworth | NSW | 1 | 1.08 | 1.00 | 1.03 | 1.03 |
| Walgett | NSW | 168 | 1.43 | 1.56 | 1.05 | 1.05 |
| Armidale | NSW | 0 | 1.19 | 1.00 | 1.04 | 1.04 |
| Tenterfield | NSW | 89 | 1.25 | 1.30 | 1.05 | 1.05 |
| Glen Innes | NSW | 55 | 1.23 | 1.18 | 1.05 | 1.05 |
| Dubbo | NSW | 0 | 1.16 | 1.00 | 1.03 | 1.03 |
| Coonamble | NSW | 97 | 1.44 | 1.32 | 1.04 | 1.04 |
| Coonabarabran | NSW | 88 | 1.39 | 1.29 | 1.04 | 1.03 |
| Parkes | NSW | 61 | 1.21 | 1.20 | 1.03 | 1.03 |
| Forbes | NSW | 69 | 1.21 | 1.23 | 1.03 | 1.03 |
| Condobolin | NSW | 124 | 1.44 | 1.41 | 1.04 | 1.04 |
| Broken Hill | NSW | 178 | 1.53 | 1.60 | 1.08 | 1.09 |
| Deniliquin | NSW | 82 | 1.30 | 1.27 | 1.05 | 1.06 |
| Hay | NSW | 158 | 1.42 | 1.53 | 1.05 | 1.06 |
| Griffith | NSW | 109 | 1.13 | 1.36 | 1.05 | 1.05 |
| Wagga Wagga | NSW | 0 | 1.02 | 1.00 | 1.04 | 1.04 |
| Tumut | NSW | 60 | 1.01 | 1.20 | 1.03 | 1.03 |
| Braidwood | NSW | 43 | 1.09 | 1.14 | 1.02 | 1.02 |
| Bombala | NSW | 107 | 1.32 | 1.36 | 1.04 | 1.04 |
| Gosford | NSW | 0 | 1.04 | 1.00 | 1.01 | 1.01 |
| Newcastle | NSW | 0 | 1.00 | 1.00 | 1.02 | 1.01 |
| Muswellbrook | NSW | 54 | 1.19 | 1.18 | 1.02 | 1.02 |
| Singleton | NSW | 24 | 1.18 | 1.08 | 1.02 | 1.02 |
| Wollongong | NSW | 0 | 1.01 | 1.00 | 1.01 | 1.01 |
| Mansfield | VIC | 68 | 1.18 | 1.23 | 1.02 | 1.01 |
| Kerang | VIC | 75 | 1.02 | 1.25 | 1.02 | 1.02 |
| Kaniva | VIC | 121 | 1.11 | 1.40 | 1.03 | 1.03 |
| Casterton | VIC | 40 | 1.11 | 1.13 | 1.03 | 1.03 |
| Morwell | VIC | 7 | 1.00 | 1.02 | 1.00 | 1.01 |
| Seymour | VIC | 42 | 1.00 | 1.14 | 1.01 | 1.01 |
| Mount Gambier | SA | 0 | 1.33 | 1.00 | 1.03 | 1.04 |
| Kingston SE | SA | 93 | 1.47 | 1.31 | 1.02 | 1.02 |
| Bordertown | SA | 111 | 1.34 | 1.37 | 1.02 | 1.03 |
| Morgan | SA | 71 | 1.35 | 1.24 | 1.02 | 1.01 |
| Renmark | SA | 84 | 1.34 | 1.28 | 1.02 | 1.02 |
| Clare | SA | 54 | 1.24 | 1.18 | 1.01 | 1.01 |
| Balaklava | SA | 38 | 1.11 | 1.13 | 1.01 | 1.01 |
| Maitland | SA | 88 | 1.35 | 1.29 | 1.02 | 1.01 |
| Yorketown | SA | 126 | 1.35 | 1.42 | 1.02 | 1.01 |
| Whyalla | SA | 0 | 1.33 | 1.00 | 1.03 | 1.02 |
| Leigh Creek | SA | 206 | 1.57 | 1.69 | 1.04 | 1.04 |
| Roxby Downs | SA | 203 | 1.68 | 1.68 | 1.05 | 1.05 |
| Port Lincoln | SA | 168 | 1.37 | 1.56 | 1.05 | 1.02 |
| Ceduna | SA | 274 | 1.66 | 1.80 | 1.06 | 1.05 |
| Scottsdale | TAS | 28 | 1.09 | 1.09 | 1.02 | 1.02 |
| George Town | TAS | 25 | 1.09 | 1.08 | 1.02 | 1.02 |
| Smithton | TAS | 75 | 1.08 | 1.25 | 1.03 | 1.03 |
| Queenstown | TAS | 93 | 1.60 | 1.31 | 1.02 | 1.02 |
| Campbelltown | TAS | 35 | 1.24 | 1.12 | 1.01 | 1.01 |
| Derwent Valley | TAS | 10 | 1.11 | 1.03 | 1.01 | 1.00 |
| Oatlands | TAS | 35 | 1.24 | 1.12 | 1.01 | 1.01 |

## ghd2025 (110 sites)

- Equipment: 1 + 0.2037 per 1,000 km of port_distance_km, capped at 1.10; RMSE 0.032.
- Installation (installation factor of Table 8.3 divided by the state capital's factor of Table 8.2): 1 + 0.0855 per hour to a city of at least 20k people, saturating at 240 min (factor 1.34); RMSE 0.070, leave-one-out RMSE 0.071.

Best saturation for each city size:

| city size | saturation (min) | b per hour | factor at saturation | RMSE | leave-one-out RMSE |
|---|---|---|---|---|---|
| 5k | 180 | 0.1214 | 1.36 | 0.093 | 0.093 |
| 10k | 180 | 0.1043 | 1.31 | 0.082 | 0.083 |
| 20k | 240 | 0.0855 | 1.34 | 0.070 | 0.071 |
| 50k | 515 | 0.0443 | 1.38 | 0.077 | 0.078 |
| 100k | 300 | 0.0413 | 1.21 | 0.084 | 0.085 |
| 200k | 330 | 0.0363 | 1.20 | 0.079 | 0.080 |
| 500k | 360 | 0.0343 | 1.21 | 0.079 | 0.080 |
| 1m | 360 | 0.0343 | 1.21 | 0.079 | 0.080 |
| 5m | 2910 | 0.0026 | 1.12 | 0.101 | 0.102 |

| site | state | travel (min) | installation observed | fitted | equipment observed | fitted |
|---|---|---|---|---|---|---|
| Cooktown | QLD | 119 | 1.40 | 1.17 | 1.00 | 1.04 |
| Port Douglas | QLD | 24 | 1.15 | 1.03 | 1.00 | 1.01 |
| Cairns | QLD | 0 | 1.10 | 1.00 | 1.00 | 1.00 |
| Innisfail | QLD | 45 | 1.18 | 1.06 | 1.00 | 1.01 |
| Richmond | QLD | 250 | 1.40 | 1.34 | 1.05 | 1.09 |
| Hughenden | QLD | 231 | 1.40 | 1.33 | 1.05 | 1.07 |
| Ingham | QLD | 64 | 1.19 | 1.09 | 1.00 | 1.02 |
| Townsville | QLD | 0 | 1.16 | 1.00 | 1.00 | 1.00 |
| Cardowan | QLD | 113 | 1.17 | 1.16 | 1.00 | 1.02 |
| Moranbah | QLD | 129 | 1.21 | 1.18 | 1.00 | 1.03 |
| Longreach | QLD | 403 | 1.35 | 1.34 | 1.10 | 1.10 |
| Rockhampton | QLD | 0 | 1.17 | 1.00 | 1.00 | 1.01 |
| Biloela | QLD | 70 | 1.17 | 1.10 | 1.00 | 1.02 |
| Gladstone | QLD | 0 | 1.17 | 1.00 | 1.00 | 1.00 |
| Bundaberg | QLD | 0 | 1.07 | 1.00 | 1.05 | 1.03 |
| Gympie | QLD | 34 | 1.07 | 1.05 | 1.00 | 1.03 |
| Nambour | QLD | 5 | 1.02 | 1.01 | 1.00 | 1.02 |
| Maroochydore | QLD | 0 | 1.02 | 1.00 | 1.00 | 1.02 |
| Dalby | QLD | 47 | 1.06 | 1.07 | 1.05 | 1.04 |
| Toowoomba | QLD | 0 | 1.02 | 1.00 | 1.00 | 1.02 |
| Warwick | QLD | 49 | 1.06 | 1.07 | 1.00 | 1.03 |
| Inglewood | QLD | 88 | 1.10 | 1.13 | 1.05 | 1.05 |
| Texas | QLD | 121 | 1.15 | 1.17 | 1.05 | 1.05 |
| Charleville | QLD | 380 | 1.26 | 1.34 | 1.10 | 1.10 |
| Roma | QLD | 216 | 1.17 | 1.31 | 1.05 | 1.08 |
| Emerald | QLD | 168 | 1.26 | 1.24 | 1.05 | 1.06 |
| Bowen | QLD | 114 | 1.19 | 1.16 | 1.00 | 1.00 |
| Mackay | QLD | 0 | 1.17 | 1.00 | 1.00 | 1.00 |
| Moree | NSW | 155 | 1.13 | 1.22 | 1.05 | 1.08 |
| Narrabri | NSW | 102 | 1.15 | 1.15 | 1.05 | 1.07 |
| Gunnedah | NSW | 45 | 1.15 | 1.06 | 1.05 | 1.05 |
| Armidale | NSW | 0 | 1.10 | 1.00 | 1.00 | 1.05 |
| Glen Innes | NSW | 55 | 1.12 | 1.08 | 1.00 | 1.06 |
| Tamworth | NSW | 1 | 1.05 | 1.00 | 1.05 | 1.04 |
| Inverell | NSW | 73 | 1.15 | 1.10 | 1.05 | 1.07 |
| Dubbo | NSW | 0 | 1.08 | 1.00 | 1.05 | 1.06 |
| Coonabarabran | NSW | 88 | 1.18 | 1.13 | 1.05 | 1.06 |
| Mudgee | NSW | 74 | 1.12 | 1.11 | 1.05 | 1.04 |
| Broken Hill | NSW | 178 | 1.26 | 1.25 | 1.10 | 1.07 |
| Deniliquin | NSW | 82 | 1.15 | 1.12 | 1.10 | 1.05 |
| Hay | NSW | 158 | 1.20 | 1.23 | 1.10 | 1.08 |
| Albury | NSW | 0 | 1.01 | 1.00 | 1.10 | 1.05 |
| Griffith | NSW | 109 | 1.08 | 1.16 | 1.05 | 1.08 |
| Wagga Wagga | NSW | 0 | 1.03 | 1.00 | 1.05 | 1.07 |
| Perisher Valley | NSW | 109 | 1.25 | 1.16 | 1.05 | 1.06 |
| Cabramurra | NSW | 110 | 1.20 | 1.16 | 1.05 | 1.06 |
| Jindabyne | NSW | 92 | 1.19 | 1.13 | 1.05 | 1.06 |
| Canberra | NSW | 1 | 1.02 | 1.00 | 1.05 | 1.04 |
| Braidwood | NSW | 43 | 1.05 | 1.06 | 1.05 | 1.03 |
| Goulburn | NSW | 0 | 1.08 | 1.00 | 1.00 | 1.02 |
| Eden | NSW | 141 | 1.11 | 1.20 | 1.05 | 1.06 |
| Bega | NSW | 118 | 1.12 | 1.17 | 1.05 | 1.05 |
| Bombala | NSW | 107 | 1.15 | 1.15 | 1.05 | 1.06 |
| Singleton | NSW | 24 | 1.08 | 1.03 | 1.00 | 1.01 |
| Newcastle | NSW | 0 | 1.01 | 1.00 | 1.00 | 1.00 |
| Nowra | NSW | 0 | 1.03 | 1.00 | 1.00 | 1.01 |
| Bowral | NSW | 28 | 1.03 | 1.04 | 1.00 | 1.01 |
| Wollongong | NSW | 0 | 1.01 | 1.00 | 1.00 | 1.00 |
| Cobar | NSW | 186 | 1.34 | 1.27 | 1.10 | 1.10 |
| Corryong | VIC | 68 | 1.08 | 1.10 | 1.10 | 1.07 |
| Bright | VIC | 54 | 1.05 | 1.08 | 1.05 | 1.04 |
| Mount Buller | VIC | 93 | 1.05 | 1.13 | 1.00 | 1.03 |
| Omeo | VIC | 89 | 1.05 | 1.13 | 1.05 | 1.05 |
| Mildura | VIC | 0 | 1.05 | 1.00 | 1.10 | 1.07 |
| Ouyen | VIC | 59 | 1.02 | 1.08 | 1.10 | 1.07 |
| Kerang | VIC | 75 | 1.02 | 1.11 | 1.10 | 1.05 |
| Ballarat | VIC | 0 | 1.00 | 1.00 | 1.00 | 1.02 |
| Ararat | VIC | 54 | 1.01 | 1.08 | 1.05 | 1.03 |
| Horsham | VIC | 116 | 1.01 | 1.17 | 1.10 | 1.04 |
| Kaniva | VIC | 121 | 1.06 | 1.17 | 1.10 | 1.05 |
| Casterton | VIC | 40 | 1.06 | 1.06 | 1.10 | 1.02 |
| Portland | VIC | 59 | 1.03 | 1.08 | 1.10 | 1.00 |
| Hamilton | VIC | 57 | 1.03 | 1.08 | 1.10 | 1.02 |
| Warrnambool | VIC | 0 | 1.01 | 1.00 | 1.05 | 1.02 |
| Sale | VIC | 30 | 1.02 | 1.04 | 1.05 | 1.03 |
| Morwell | VIC | 7 | 1.00 | 1.01 | 1.00 | 1.02 |
| Bendigo | VIC | 0 | 1.00 | 1.00 | 1.00 | 1.03 |
| Shepparton | VIC | 0 | 1.01 | 1.00 | 1.05 | 1.03 |
| Seymour | VIC | 42 | 1.00 | 1.06 | 1.00 | 1.02 |
| Launceston | TAS | 0 | 0.95 | 1.00 | 1.00 | 1.01 |
| Scottsdale | TAS | 28 | 1.05 | 1.04 | 1.00 | 1.01 |
| Queenstown | TAS | 93 | 1.26 | 1.13 | 1.10 | 1.02 |
| Smithton | TAS | 75 | 1.07 | 1.11 | 1.10 | 1.00 |
| Hobart | TAS | 0 | 1.00 | 1.00 | 1.00 | 1.00 |
| Derwent Valley Council | TAS | 10 | 1.05 | 1.01 | 1.00 | 1.00 |
| Swansea | TAS | 70 | 1.15 | 1.10 | 1.05 | 1.02 |
| Bothwell | TAS | 30 | 1.10 | 1.04 | 1.05 | 1.01 |
| Murray Bridge | SA | 42 | 1.05 | 1.06 | 1.00 | 1.02 |
| Keith | SA | 128 | 1.15 | 1.18 | 1.05 | 1.04 |
| Naracoorte | SA | 63 | 1.15 | 1.09 | 1.05 | 1.04 |
| Mount Gambier | SA | 0 | 1.15 | 1.00 | 1.05 | 1.02 |
| Renmark | SA | 84 | 1.20 | 1.12 | 1.05 | 1.04 |
| Berri | SA | 95 | 1.15 | 1.14 | 1.00 | 1.04 |
| Kapunda | SA | 19 | 1.05 | 1.03 | 1.00 | 1.01 |
| Port Pirie | SA | 105 | 1.11 | 1.15 | 1.05 | 1.00 |
| Adelaide | SA | 0 | 1.00 | 1.00 | 1.00 | 1.00 |
| Clare | SA | 54 | 1.11 | 1.08 | 1.00 | 1.02 |
| Yorketown | SA | 126 | 1.15 | 1.18 | 1.00 | 1.00 |
| Maitland | SA | 88 | 1.15 | 1.13 | 1.00 | 1.01 |
| Wallaroo | SA | 83 | 1.11 | 1.12 | 1.00 | 1.00 |
| Whyalla | SA | 0 | 1.15 | 1.00 | 1.05 | 1.00 |
| Port Augusta | SA | 46 | 1.15 | 1.07 | 1.05 | 1.01 |
| Leigh Creek | SA | 206 | 1.31 | 1.29 | 1.10 | 1.06 |
| Peterborough | SA | 120 | 1.20 | 1.17 | 1.05 | 1.02 |
| Coober Pedy | SA | 383 | 1.40 | 1.34 | 1.10 | 1.07 |
| Roxby Downs | SA | 203 | 1.31 | 1.29 | 1.10 | 1.06 |
| Cleve | SA | 89 | 1.17 | 1.13 | 1.05 | 1.02 |
| Port Lincoln | SA | 168 | 1.20 | 1.24 | 1.05 | 1.00 |
| Ceduna | SA | 274 | 1.31 | 1.34 | 1.10 | 1.00 |
| Elliston | SA | 193 | 1.25 | 1.28 | 1.05 | 1.03 |

