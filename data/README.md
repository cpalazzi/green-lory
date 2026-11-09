# data/: large inputs (tracked layout, untracked contents)

This directory and this README are tracked; every data file below is ignored by
git (`/data/**` in `.gitignore`) and must be fetched or linked into place. Paths
are relative to this directory. SHA-256 hashes identify the exact files used in
the reconciliation campaigns; verify with `shasum -a 256 <file>`.

## 2019 PV and wind profiles (`weather_data/salmon2022_pv_wind_2019_hourly_1deg/`)

Hourly output per unit of capacity for fixed PV (`Solar`), tracking PV (`SolarTracking`) and
wind (`WindPowers`), 8,760 hours of 2019 on a 1-degree grid: the profiles used by Salmon and
Bañares-Alcántara (2022, *J. Cleaner Production* 367, 133045,
doi:10.1016/j.jclepro.2022.133045) and the original lcoa-opt model. The paper says its weather
data come from the ERA5 reanalysis; the files carry no metadata of their own. Nine NetCDF-3
files, three longitude bands per technology (unsuffixed = 180 W to 61 W, `1` = 60 W to 59 E,
`2` = 60 E to 179 E; the suffix is a longitude partition, not a year), 1.51 GB each. Used
unchanged by the frozen legacy model (`legacy_lcoa/`), by the reconciling run and by every
green-lory run before November 2026 (`weather_dir` in `run_defaults` of `../runs.yaml`); the
campaign runs since read the ERA5 profiles of the next section.

| File | SHA-256 |
|---|---|
| `Solar.nc` | `757911abd570204079203cf298b3c0865dad661e260fbe261c7b8e75e94f9c2a` |
| `Solar1.nc` | `21498760861c1642ad8eef53bd462cf7934d45e08d2eea01b4878ff732198921` |
| `Solar2.nc` | `d42c5c78413f51e86e2ff424b578c5df9bef2365fc2a9fa7e0acd01a0d0ec1ba` |
| `SolarTracking.nc` | `168ad862fc38c385c8fc79acfcc2508f4f063bd19cef9b7492a0639f586da7c5` |
| `SolarTracking1.nc` | `0dd96f4ec3a52fee62125160a7f0af106214147a90cc8741676e44daecb3a430` |
| `SolarTracking2.nc` | `067519348768aed581994bb8a0c64eed10d8e41e59a73555c0e6bae414b5b344` |
| `WindPowers.nc` | `780c675aaca41d628a599a8f4a2c6f2b365e99a529e48fdff8bc3b8bf3a5050d` |
| `WindPowers1.nc` | `27754a50a4139b19a2b82ec27ff0fbff98af248f28020155f3687eed7af4deab` |
| `WindPowers2.nc` | `49e7ec627d3cb4320521fc9a64f2f507cb9456e9e4c18432fb58d5c99aa5ec22` |

Locations:
- On the Mac, this folder. The files were moved here from `weather_data/` on 7 October 2026;
  the three focal cells load identical profiles from the new place.
  `~/programming/shipping_sprint/lcoa_model/lcoa-opt/data/*.nc` link to them (and through those
  links, `lcoa_validation/data/`).
- On ARC, `/data/engs-df-green-ammonia/engs2523/green-lory/data/weather_data/salmon2022_pv_wind_2019_hourly_1deg/`.
  Links with the old names stay in `weather_data/`, so releases staged before 7 October still run.
- A compact per-cell store for the 15,377 archived supplier cells (`extract_weather_store.py`,
  3.2 GB, values unchanged) is at
  `results/campaigns/legacy_lcoa_20260915_v1/weather_store_archived15377_mac_v1/` locally and
  `green-lory/data/weather_store_archived15377_v1/` on ARC.

Method, from the authors' theses: ERA5 at one degree, wind through "the curve for a Vestas 3 MW
wind turbine" and PV through pvlib, using "the DC power output from solar cells" (Salmon 2023,
DPhil thesis, Oxford, chapter 3, p. 37). The conversion was Zac Cesaro's (Salmon's
acknowledgements); Cesaro's thesis (2022, Oxford, section 4.5) uses ERA5's 100 m wind and pvlib's
PVWatts model on ERA5's surface solar radiation. Neither names the Vestas model, the hub height or
the panel tilt. The PV profiles are therefore DC output per MW of modules, which is why they
occasionally exceed 1.0 (Atacama fixed 1.06, tracking 1.12); DEA's PV costs are per MW of AC
(DC/AC 1.25).

The profiles average the whole 1-degree cell, sea and lakes included (at equal sea share, cells
centred at sea and on land have the same wind; it rises with the sea share). Runs from October
2026 rescale the wind of cells with at least 1 % water to the cell's land with a factor file in
`land_correction/` (`model/weather_land.py`): factors by cell, month and hour of day, the ratio
of the land-weighted to the area-weighted mean of the cell's ERA5 wind capacity factors at 0.25
degrees (the green-condor stores listed below), kept between the wind of the cell's fully
inland neighbours and the uncorrected profile. PV is not corrected. The point weights are
tracked in `../inputs/weather/era5_point_weights_1deg.csv`; the factors are computed on ARC by
`../arc/jobs/10_weather_land_correction.sh`.

| File | What it is | SHA-256 |
|---|---|---|
| `land_correction/weather_land_correction_2019.nc` | Wind factors of the 10,971 cells with at least 1 % water (job 13442443, release `20261008-weather-land-v2`, campaign `green-lory-campaigns/weather_land_correction_20261008_v2/`); the inland floor binds in 1,182 cells. On ARC in `green-lory/data/weather_data/land_correction/` (read-only), so releases reach it through their `data/weather_data` link. A first version of 8 October (job 13441218) was withdrawn | `ad49b8120da80c414aef4cfb2d503ad8330f4b1c27697477827a45f368c7c1cf` |

## ERA5 2019 profiles (`weather_data/era5_2019_1deg/`, `weather_data/era5_2019_1deg_sea/`)

Hourly output per MW for fixed PV, tracking PV and wind, 8,760 hours of 2019 on the same 1-degree
grid and in the same file layout as Salmon's profiles, built from the green-condor ERA5 2019
cutout with atlite 0.4.1 (`../model/weather_profiles.py`, settings in `weather.profiles` of
`../inputs/parameters.yaml`; `../CAMPAIGN_PLAN.md`, section 7a). Wind: DEA 2050's turbines (6 MW
at 110 m on land, 35 MW at 220 m at sea) as stretched NREL reference curves, smoothed (Hörsch et
al. 2018, eq. 8), with the speed normalised to the air density at the site; the wake factor is
applied when a run loads them. PV: output per MW of AC (x 1.25 DC/AC, capped at 1), fixed at the
latitude-optimal tilt and single-axis tracking. Each cell's land profile averages its land
pixels and its sea profile its sea pixels; the land stack holds the land profiles, the sea
stack the wind at sea as `WindPowers` and zero PV (`stack --sea`). The campaign runs of November
2026 read them (`weather_dir` in `../runs.yaml`).

Built on ARC from release `20261106-profiles-v3` (bands 1-11, array 13623198) and
`20261109-profiles-v4` (the stacks, jobs 13631579 and 13631580: the first stacks, 13623199-200,
wrote the files but failed on their `provenance.json`), in
`green-lory-campaigns/weather_profiles_20261106_v2/`, linked as `green-lory/data/weather_data/era5_2019_1deg`
and `era5_2019_1deg_sea`. Each folder's `provenance.json` gives the band files, their hashes and
the settings.

| File | SHA-256, land | SHA-256, sea |
|---|---|---|
| `Solar.nc` | `307910d49d993c8e93aa386d26669cc07b4db75f1df0881e93cd7951672237e3` | `d9683aa788903b17baad60e47f3b944adde18c88bea660fb8aaf8862ac2b8ffd` |
| `Solar1.nc` | `d6d81aaa58e46c4101ff26b2777cd34e68002f00b8edd9c190e6550263295d4b` | `6a91393eda136e6b5b46ada54f0b0b5f1c7cf855a0729b6ba67d8da8001f53c8` |
| `Solar2.nc` | `df0bf662525542a4b7aa475ff0be6e7d8438caeb7993f0de02066088d64110c8` | `3c3762fc036f8a714f4e3dafcc90e09ef0b867f26a2f95de22234c97e63a71d9` |
| `SolarTracking.nc` | `76d81f328dfde6940df08256f644ce79c48c4b9aa5520a5d77928d95138f41b9` | `d9683aa788903b17baad60e47f3b944adde18c88bea660fb8aaf8862ac2b8ffd` |
| `SolarTracking1.nc` | `ba80bebf0c486ef4076052ae725665bf2692b8f3001c7af5cf154b8caaab209e` | `6a91393eda136e6b5b46ada54f0b0b5f1c7cf855a0729b6ba67d8da8001f53c8` |
| `SolarTracking2.nc` | `79129a3cf7996c1fc52939feeff6965c7bafb0dccb9e854a70fb0c1929b16464` | `3c3762fc036f8a714f4e3dafcc90e09ef0b867f26a2f95de22234c97e63a71d9` |
| `WindPowers.nc` | `ce814e790a49b1d311f88031b39b80c0a611812169ef9e089f733b298918a95d` | `d4af8ab9dffe1478c0323bd0327f2fa6c76e73a65a50bbcc3cf9346ffa72d5a5` |
| `WindPowers1.nc` | `ad9e4ee9953a8be114defda1c17229e41bdba78df9c7f7995c0fd4a27b04a468` | `7cb0166652424c3e31f997f4f6b66aebb6d4929360320cb0ed361fccc8c993f7` |
| `WindPowers2.nc` | `148f74b8ca973b6df6ded036f6b3bfa33662e525647ef44781373a89244a49d6` | `94ec102a13168743502f1e3d13a5833dbf179416db5cecffcc903f6df37c4912` |

## Land and terrain inputs

| File | Purpose | SHA-256 |
|---|---|---|
| `MCD12C1.A2022001.061.2023244164746.hdf` | MODIS land cover, Collection 6.1, year 2022, 0.05-degree class fractions, global (3600 x 7200 x 17). Used by the legacy land step (`legacy_lcoa/legacy_land_areas.py`) and by the green-lory land build (`model/land_processing.py`). Sufficient for the global legacy reproduction; the historical vintage is unknown | `5b12c573975da5cb1b119973411e119e6b749b88a0056c68668b9a261b3e56cc` |
| `land/GEBCO_2025_sub_ice.nc` | GEBCO 2025 sub-ice bathymetry/topography, 7.5 GB (moved here from the pypsa-earth clone's `data/gebco/` on 16 September 2026; a symlink was left there). Slope classes and the slope limit (20 degrees since 7 October 2026, 15 before) in the green-lory land build; bathymetry and depth-based costs for the offshore implementation (open) | `40080250dd9932367461f0294195ef01b8ab78163c9f93d3b60a4f751ca3aa75` |
| `model_bathymetry.nc` | Coarse bathymetry used by the legacy model and the green-lory preflight (0.5 MB) | `6600d5980390d3fb623c2fd4dca2c131ac69df507b151b7420ec63cef9eaf697` |
| `land/WDPA_Feb2026_Public_shp_{0,1,2}/` | World Database on Protected Areas, February 2026 shapefiles (1.9 + 2.6 + 1.9 GB); protected-area exclusions in the green-lory land build. Copied from `green-lory/data/` on ARC on 16 September 2026 (rsync, sizes verified) |
| `countries.geojson` | Country boundaries for cell attribution and maps (`model/surface.py`). ARC copy at `green-lory/data/`; copied here on 6 October 2026 from `results/campaigns/verschuur_reconcile_20260907_v1/arc_received_20260914/` | `86057d2fad98ac537a66df1da995d2f7904c2c215457077853087d13fa412487` |
| Native MODIS MCD12Q1 500 m tiles (2022, C6.1) | Resolution checks only; three tiles at `results/campaigns/land_reconcile_20260914_v1/sources/native-modis-c61-2022/`, twelve more listed in the download plan there. Not needed for the global reproduction |

Generated land tables (`max_capacities_*.csv`) are campaign outputs, not inputs. The
campaign runs (step 5) are to use the October build of 8 October 2026 (release
`20261008-land-slope20-v2`): slope classes up to 20 degrees and protected land counted on land
only. It is on ARC at
`/data/engs-df-green-ammonia/engs2523/green-lory-campaigns/land_center_slope20_20261008_v2/`, with
local copies in `results/campaigns/land_center_slope20_20261008_v2/`, and awaits the step-3
review:

| File | SHA-256 |
|---|---|
| `max_capacities_center_100pct_slope20.csv` | `e4c76c78dd1950ce685e20d6d74b7092dc962403e05b735deec6d559007a863a` |
| `max_capacities_center_20pct_slope20.csv` | `0495b732e75c3784a9219327243ce212db41ee465dffaa1c22dce9ffa65ec7eb` |
| `max_capacities_center_2pct_slope20.csv` | `f22b9a40b145e0912af2ac3f76ad00fe81b5a1d6b450f3b410a1bd515812f7a4` |
| `locations_onshore_suitable_center_20261008.csv` (the 16,885 run cells, `land_table.run_cells`) | `2299a04e692bd96f93bc2ab1d0d128b50177fb12c0baba5c0fe7e85718ed5a68` |

The key runs of September used the centred build of 23 September 2026 (slope limit 15 degrees)
on ARC,
`/data/engs-df-green-ammonia/engs2523/green-lory-campaigns/land_center_20260923_v1/`, with local
copies in `results/campaigns/land_center_20260923_v1/`:

| File | SHA-256 |
|---|---|
| `max_capacities_center_100pct_slope15.csv` | `93a4777634b3504bfa9a012cf7cdc2b66b8bad16c65b70a00f1267563a0abf83` |
| `max_capacities_center_20pct_slope15.csv` | `12277da14986a8ba9af481505d70072dd20846e2fe975903713ac9384bea8c21` |
| `max_capacities_center_2pct_slope15.csv` | `89a601662b9570d7c42adb439fecd6b9337d59104b3b76e25ad3384bf5855753` |
| `locations_onshore_suitable_center_20260923.csv` (the 16,509 cells of the key runs) | `792a3e41d68c983b3e9462650b73ac5d481f629d8f1449aa00aa484a9b460bf4` |

Earlier tables (`lory_reconcile_20260722_v1/land/*.csv`, `data/max_capacities_*` of May) give
each cell the suitability of the cell one degree south and are refused by the model.

## ERA5 / atlite stack

The 2019 global cutout and its atlite capacity-factor tiles are kept on ARC
because of size. A separate PyPSA-Earth cutout lives in the pypsa-earth clone
and is linked here (the file stays in that repository, which is also worked on):

| Item | Location | Notes |
|---|---|---|
| `weather_data/cutouts/cutout-2013-era5.nc` | symlink to `~/programming/pypsa_models/pypsa-earth-green-auklet/pypsa-earth/cutouts/cutout-2013-era5.nc` | PyPSA-Earth ERA5 cutout, year 2013, 18.4 GB; a PyPSA-Earth artifact, not used by any green-lory campaign yet |

| Item | Location on ARC | Notes |
|---|---|---|
| ERA5 cutout 2019 | `/data/engs-df-green-ammonia/engs2523/green-condor/data/global_cutout_2019.nc` | about 290 GiB, 0.25 degree, 8,760 hours; 100 m wind, radiation, temperature, roughness |
| atlite capacity factors | `/data/engs-df-green-ammonia/engs2523/green-condor/outputs/global_cf_2019_tiles_<start>_<end>.zarr` | 12 stores of 60 latitude rows each (0-59 ... 660-719), about 3.2 GB each; Vestas V112 3 MW onshore, NREL 5 MW offshore, CSi fixed PV; no tracking field. The unsuffixed `global_cf_2019.zarr` covers 85 rows only and is not a complete product. Read by the land correction of the profiles (zarr 3, so with `envs/green-condor-env`) |
| Provenance | `green-lory/data/weather_data/archive/green-condor_global_cutout_2019/` on ARC | notebooks, readmes and scripts that produced the cutout, not the data |

At the three focal cells these profiles give 7 to 9 % lower fixed-PV and 26 to
38 % lower raw wind capacity factors than the legacy stack
(`results/campaigns/land_reconcile_20260914_v1/audit/alternative-weather-v1/`);
they have not been used in any accepted surface.

## Ports (`ports/`)

| File | Purpose | SHA-256 |
|---|---|---|
| `ports/verschuur2024_ports.csv` | The 1,377 ports of the green-porpoise shipping network (Verschuur et al. 2024, *Environ. Res.: Infrastruct. Sustain.* 4, 015001), with location, country and trade columns; 46 in Australia, 16 in Chile. Copied on 7 October 2026 from the archived project snapshot `../results/campaigns/verschuur_reconcile_20260907_v1/historical/git-0a63616/data/c_port_data.csv`, byte for byte. For the equipment (port-distance) factor of the site costs | `497b9bc9fdcb9b67af57c5834f1a068633b0776535e254ec644bc89750f13651` |

## Maritime zones (`marine_regions/`)

| File | Purpose | SHA-256 |
|---|---|---|
| `marine_regions/world_eez_v12_wfs/eez_v12.zip` (`eez.shp`, `.dbf`, `.shx`, `.prj`, `.cst` inside, unpacked beside it) | Exclusive economic zones for the country of offshore cells (`iso_sov1`, the sovereign; `pol_type` 200NM, Overlapping claim or Joint regime); cells outside every zone are international waters. Flanders Marine Institute (2023), Maritime Boundaries Geodatabase: Maritime Boundaries and Exclusive Economic Zones (200NM), version 12, doi:10.14284/632, CC BY 4.0 (the geodatabase v12, doi:10.14284/628). 285 polygons, EPSG:4326. Retrieved on 30 October 2026 from Marine Regions' public feed (layer `MarineRegions:eez`, "Exclusive Economic Zones (200 NM) (v12, world, 2023)"), the same data as the download page, which asks for personal details | zip `7b010436bf816c34600f0acea35a4a6b3db83ce26901c9b4f957634b5d7ba1ab`; `eez.shp` `bb68ebda72f96ecf4f6428cd11079c50ec94eb2ffc4f711e092698dbb79066e5`; `eez.dbf` `badec96cd1b9a1a3a46dc78229ce1329ed48c20178b08cbe6b6fd020acf3cd55` |

## Site-cost inputs (`accessibility/`, `water/`, `gazetteer/`)

Read once by `model/site_geography.py`, which writes the per-cell and per-site tables of
`../inputs/site_costs/` (tracked); a run never reads these files. Downloaded on 7 October 2026;
the figshare files were checked against figshare's MD5 sums.

| File | Purpose | SHA-256 |
|---|---|---|
| `accessibility/nelson2019_travel_time/travel_time_to_cities_<k>.tif`, k = 1, 2, 3, 4, 5, 8, 10, 11, 12 | Nelson et al. (2019, *Scientific Data* 6, 266, doi:10.1038/s41597-019-0265-5): travel time (minutes) in 2015 to the nearest city of a population band (k = 1 to 9) or above a size (10: 20,000; 11: 50,000; 12: 5,000), 30 arc-seconds, on the Weiss et al. (2018) friction surface. figshare doi:10.6084/m9.figshare.7638134 (version 4), CC BY 4.0. `README.txt` and `metadata_v02.csv` beside them give the bands. Layers 6, 7 and 9 are not needed for the city sizes used | 1: `e8b49ec19ee5b50af7a9ace75a676284c9d4f2939ad31353f531100e1bba19a2`; 2: `8612dc288aedc02c9effa2c8165e255d91b803428230bb1d1d8dee4dc104947d`; 3: `c8b73856d5b9375ea4aefb54f320a14d1ea927de1f5c01a9f39ce9cbfca6dbf6`; 4: `17690d821b8f83fb065c6e0347c7518a8b43529cd83bd7a4f6dfd70323c97532`; 5: `0ff0dd92c84ba2736c5547dfe745abc03ab5f7ecd59fbaa09b4fa405ce7a2584`; 8: `16c2aac960a02fc5dcef1fa3dc18cc20a0595a6e3a117122622f7caf29decac8`; 10: `debe0105966c53ce8c7926d29f3a647c339c158b75f07b92a4597297b2a9e4e0`; 11: `80138115b0310b4d9cede0d8327f6ccbe74503433d19de73163f9323564b2add`; 12: `b0840a0cf12a49a2d6edfe078ecf067d318db734648f5332435a66b012743803` |
| `water/glwd_v2/GLWD_v2_0_area_by_class_ha_tif.zip` | Global Lakes and Wetlands Database v2.0 (Lehner et al. 2025, *Earth System Science Data* 17, 2277, doi:10.5194/essd-17-2277-2025): area of each of 33 classes per 15 arc-second pixel. figshare doi:10.6084/m9.figshare.28519994, CC BY 4.0. Classes 00 to 06, 20, 21 and 32 are unzipped beside it (`GLWD_v2_0_area_by_class_ha/`) | `d35407f3d7a904b25557f7759d6a92dbdc08451a94d9fd28cfd3db8f31bd6ad5` |
| `gazetteer/geonames_au/AU.zip` (`AU.txt` inside) | GeoNames gazetteer of Australia (https://download.geonames.org/export/dump/, CC BY 4.0), to place the sites of the AEMO location-factor tables. The dump is updated daily; this copy is of 7 October 2026 | zip `4b9d848097c2c3691d0bc9ade6b65253e50fae1814145a7a55ea758fa02659c0`; `AU.txt` `85697065e9b607379c7be42b84f5ac9476f0a823b6885c2fb31105e95ac71f38` |

The permanence test of saline lakes reads the JRC Global Surface Water occurrence layer (Pekel
et al. 2016, *Nature* 540, 418, doi:10.1038/nature20584; version 1.4, 1984-2021) remotely from
`storage.googleapis.com/global-surface-water/downloads2021/occurrence/`, cell by cell; nothing
is stored here, and the result per cell is `../inputs/site_costs/gsw_permanent_water.csv`.

## Other inputs referenced by scripts

`port_locations.csv`, `travel_time_by_cell.csv` and `single_site_weather_data.csv` on ARC under
`green-lory/data/` were inputs of the spatial-cost notebook and a single-site notebook, both
removed (October 2026); nothing reads them now. The
DEA technology data sheets that the technology YAMLs cite are linked from Zotero in
`../references/` (not tracked). Small tracked inputs are under `inputs/`.
