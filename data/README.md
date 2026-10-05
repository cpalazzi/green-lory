# data/: large inputs (tracked layout, untracked contents)

This directory and this README are tracked; every data file below is ignored by
git (`/data/**` in `.gitignore`) and must be fetched or linked into place. Paths
are relative to this directory. SHA-256 hashes identify the exact files used in
the reconciliation campaigns; verify with `shasum -a 256 <file>`.

## Legacy 2019 generation-profile stack (`weather_data/`)

Nine NetCDF-3 files produced for the original lcoa-opt model: hourly 2019,
1-degree cells, three longitude bands per technology (unsuffixed = 180 W to
61 W, `1` = 60 W to 59 E, `2` = 60 E to 179 E; the suffix is a longitude
partition, not a year). Each file is 1.51 GB. Used unchanged by the frozen
legacy model (`reconciliation/legacy_lcoa/`) and by the green-lory campaigns.

| File | SHA-256 |
|---|---|
| `weather_data/Solar.nc` | `757911abd570204079203cf298b3c0865dad661e260fbe261c7b8e75e94f9c2a` |
| `weather_data/Solar1.nc` | `21498760861c1642ad8eef53bd462cf7934d45e08d2eea01b4878ff732198921` |
| `weather_data/Solar2.nc` | `d42c5c78413f51e86e2ff424b578c5df9bef2365fc2a9fa7e0acd01a0d0ec1ba` |
| `weather_data/SolarTracking.nc` | `168ad862fc38c385c8fc79acfcc2508f4f063bd19cef9b7492a0639f586da7c5` |
| `weather_data/SolarTracking1.nc` | `0dd96f4ec3a52fee62125160a7f0af106214147a90cc8741676e44daecb3a430` |
| `weather_data/SolarTracking2.nc` | `067519348768aed581994bb8a0c64eed10d8e41e59a73555c0e6bae414b5b344` |
| `weather_data/WindPowers.nc` | `780c675aaca41d628a599a8f4a2c6f2b365e99a529e48fdff8bc3b8bf3a5050d` |
| `weather_data/WindPowers1.nc` | `27754a50a4139b19a2b82ec27ff0fbff98af248f28020155f3687eed7af4deab` |
| `weather_data/WindPowers2.nc` | `49e7ec627d3cb4320521fc9a64f2f507cb9456e9e4c18432fb58d5c99aa5ec22` |

Locations: this directory on the Mac (moved here on 16 September 2026;
`~/programming/shipping_sprint/lcoa_model/lcoa-opt/data/*.nc` are symlinks to
these files) and `/data/engs-df-green-ammonia/engs2523/green-lory/data/weather_data/`
on ARC. A compact per-cell store for the 15,377 archived supplier cells
(`extract_weather_store.py`, 3.2 GB, values unchanged) is at
`results/campaigns/legacy_lcoa_20260915_v1/weather_store_archived15377_mac_v1/`
locally and `green-lory/data/weather_store_archived15377_v1/` on ARC.

Known open point: PV profiles occasionally exceed 1.0 (Atacama fixed 1.06,
tracking 1.12); the AC/DC normalisation basis is undocumented.

## Land and terrain inputs

| File | Purpose | SHA-256 |
|---|---|---|
| `MCD12C1.A2022001.061.2023244164746.hdf` | MODIS land cover, Collection 6.1, year 2022, 0.05-degree class fractions, global (3600 x 7200 x 17). Used by the legacy land step (`reconciliation/legacy_lcoa/legacy_land_areas.py`) and by the green-lory land build (`model/land_processing.py`). Sufficient for the global legacy reproduction; the historical vintage is unknown | `5b12c573975da5cb1b119973411e119e6b749b88a0056c68668b9a261b3e56cc` |
| `land/GEBCO_2025_sub_ice.nc` | GEBCO 2025 sub-ice bathymetry/topography, 7.5 GB (moved here from the pypsa-earth clone's `data/gebco/` on 16 September 2026; a symlink was left there). Slope (> 15 degree) exclusions in the green-lory land build; bathymetry and depth-based costs for the offshore implementation (open) | `40080250dd9932367461f0294195ef01b8ab78163c9f93d3b60a4f751ca3aa75` |
| `model_bathymetry.nc` | Coarse bathymetry used by the legacy model and the green-lory preflight (0.5 MB) | `6600d5980390d3fb623c2fd4dca2c131ac69df507b151b7420ec63cef9eaf697` |
| `land/WDPA_Feb2026_Public_shp_{0,1,2}/` | World Database on Protected Areas, February 2026 shapefiles (1.9 + 2.6 + 1.9 GB); protected-area exclusions in the green-lory land build. Copied from `green-lory/data/` on ARC on 16 September 2026 (rsync, sizes verified) |
| `countries.geojson` | Country boundaries for cell attribution and maps. ARC copy at `green-lory/data/`; a local copy is in `results/campaigns/verschuur_reconcile_20260907_v1/arc_received_20260914/` |
| Native MODIS MCD12Q1 500 m tiles (2022, C6.1) | Resolution checks only; three tiles at `results/campaigns/land_reconcile_20260914_v1/sources/native-modis-c61-2022/`, twelve more listed in the download plan there. Not needed for the global reproduction |

Generated land tables (`max_capacities_*.csv`) are campaign outputs, not
inputs: the reconciliation uses
`/data/engs-df-green-ammonia/engs2523/green-lory-campaigns/lory_reconcile_20260722_v1/land/paper_2pct_slope15.csv`
(SHA-256 `4d0753f77f3574c1616159997481c090c68e383ac02e6b25284561fb0ab1b38f`), a
local copy of which is in `arc_received_20260914/` of the network campaign.

## ERA5 / atlite stack

The 2019 global cutout and its atlite capacity-factor tiles are kept on ARC
because of size. A separate PyPSA-Earth cutout lives in the pypsa-earth clone
and is linked here (the file stays in that repository, which is also worked on):

| Item | Location | Notes |
|---|---|---|
| `cutouts/cutout-2013-era5.nc` | symlink to `~/programming/pypsa_models/pypsa-earth-green-auklet/pypsa-earth/cutouts/cutout-2013-era5.nc` | PyPSA-Earth ERA5 cutout, year 2013, 18.4 GB; a PyPSA-Earth artifact, not used by any green-lory campaign yet |

| Item | Location on ARC | Notes |
|---|---|---|
| ERA5 cutout 2019 | `/data/engs-df-green-ammonia/engs2523/green-condor/data/global_cutout_2019.nc` | about 290 GiB, 0.25 degree, 8,760 hours; 100 m wind, radiation, temperature, roughness |
| atlite capacity factors | `/data/engs-df-green-ammonia/engs2523/green-condor/outputs/global_cf_2019_tiles_<start>_<end>.zarr` | 12 stores of 60 latitude rows each (0-59 ... 660-719), about 3.2 GB each; Vestas V112 3 MW onshore, NREL 5 MW offshore, CSi fixed PV; no tracking field. The unsuffixed `global_cf_2019.zarr` covers 85 rows only and is not a complete product |
| Provenance | `green-lory/data/weather_data/archive/green-condor_global_cutout_2019/` on ARC | notebooks, readmes and scripts that produced the cutout, not the data |

At the three focal cells these profiles give 7 to 9 % lower fixed-PV and 26 to
38 % lower raw wind capacity factors than the legacy stack
(`results/campaigns/land_reconcile_20260914_v1/audit/alternative-weather-v1/`);
they have not been used in any accepted surface.

## Other inputs referenced by scripts

`dea_reference/` workbooks (DEA technology datasheets), `port_locations.csv`,
`travel_time_by_cell.csv` and `single_site_weather_data.csv` live on ARC under
`green-lory/data/`; copy them here when running the corresponding notebooks
locally. Small tracked inputs are under `inputs/`.
