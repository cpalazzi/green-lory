# Green-lory: state of the model and plan for the global campaign

16 September 2026. Decisions taken today: land-competition fraction 20 % of
suitable land for the new base (2 % kept as the paper sensitivity), the 1 Mt/yr
admission cutoff kept for green-porpoise, DEA technology inputs preferred, no
global run is submitted without explicit confirmation.

## 1. Reference for a 20 % share

Bogdanov et al. (2019), *Nature Communications* 10:1077, LUT energy system
model, introduction: "we assume that up to 6% of regional area can be used for
PV system installations, 4% of area can be used for wind farm installations".
Those limits are shares of *total* regional area. In our stack suitable land
after protected-area and slope exclusions is 24 % of onshore land (32.1 of
135 million km² between 75 S and 75 N), so 20 % of suitable land is 4.8 % of
total land, inside the LUT range, and it is the order of announced hub leases
(6,500 to 15,000 km² in Australia, 60 to 130 % of a 1-degree cell). The global
demand needs about 29,000 km² of arrays, 4.5 % of even the 2 % budget, so the
share governs per-cell concentration, not aggregate scarcity (see
[LAND_SHARE_STATEMENT_20260916.md](LAND_SHARE_STATEMENT_20260916.md)).

## 2. State of the green-lory model

What exists and is validated:

- **Plant** (`model/main.py`, `basic_ammonia_plant_2050_*`, `inputs/*.yaml`):
  PyPSA/linopy plant with wind, fixed and single-axis PV (policy-selectable,
  fixed is the central choice), HHV-basis electrolysis, explicit compressor,
  DEA tank hydrogen store, fuel cell, battery with PCS, Haber-Bosch with
  minimum load and ramp limits, ammonia store; hourly or 2/4-hour aggregation
  with snapshot-weighted accounting; grid backstop for infeasibility (off in
  science runs). Energy audit closes (9.2 MWh/t, 68 % HHV). Reproduces the
  legacy plant at the focal cells to within 1 % in replication mode. 156 tests.
- **Global runner** (`model/run_global.py`, `arc/submit_lory_sequence.sh`,
  `arc/merge_and_qa_campaign.py`): per-cell finance overrides (Ameli WACC),
  spatial cost overrides (build multiplier = depth x remoteness x labour,
  water cost, land rent), four longitude quadrants of 48 CPUs on ARC, merge
  and QA with hashes, contracts for green-porpoise. An hourly 52,702-cell
  surface took about 3 h of wall time on four concurrent 48-core jobs
  (580 CPU-hours) in September.
- **Land** (`model/land_processing.py`, `model/land_capacity.py`): MODIS
  MCD12C1 classes with the Table-2 factors, WDPA and > 15 degree slope
  exclusions, competition fraction, classwise nested union, latitude-packed
  fixed-PV density (First Solar module), tracking at 1.458 x fixed (Bolinger
  and Bolinger 2022), wind 5 MW/km²; allocation `colocated` (base, wind
  exclusive fraction 0.03 from Denholm et al. 2009) or `exclusive`; capacity
  by scaled reference design (`paper_scaled`) or by land-enforced finite-site
  solves (`solved_quantity`, supply curves).
- **Legacy package** (`reconciliation/legacy_lcoa/`): reproduced end to end
  in two variants (archived-table reproduction; paper method as stated).

Known limitations, in the order they matter:

1. (Corrected 23 Sep 2026.) The July/September land build mislabelled every
   MODIS band by one degree: the suitability at label (lat, lon) is that of the
   box [lat-1, lat] x [lon, lon+1], while the area, WDPA, slope and bathymetry
   overlays used [lat, lat+1] x [lon, lon+1]. Relative to the centred weather
   node the MODIS content sits half a degree south and the exclusions half a
   degree north. Every September surface carries this. Fixed in
   `model/land_processing.py` (centred cells by default, exact row weights);
   the centred build is release `20260923-land-center-v1`, jobs 13225827-9, and
   the campaign below must use its tables (`max_capacities_center_*_slope15.csv`).
2. One weather year (2019) from the legacy profile stack; PV profiles exceed
   1.0 at times (normalisation basis unknown); the ERA5/atlite alternative
   gives 7 to 9 % lower PV and 26 to 38 % lower wind capacity factors at the
   focal cells and has not been run through a solve.
3. Capacity for green-porpoise is still a scaled 1 Mt/yr design; finite-site
   supply curves exist for three cells only.
4. Offshore is unfinished (depth multipliers exist; bathymetry exclusions and
   costs not applied); land rent is zero; no economies of scale.

## 3. Recommended implementation

1. Freeze the plant as it is; DEA 2030 and 2050 configurations as the
   technology basis, Way 2050 retained only for the replication lineage.
2. Rebuild the land stack once with a centered anchor, the co-located rule,
   fixed-PV density and a 100 % build from which 2 % and 20 % (and any other
   share) are derived by the existing cheap rescaling job.
3. Keep the scaled-design capacity with the 1 Mt/yr cutoff as the screening
   surface for green-porpoise at 20 %, and add finite-site supply curves for
   the active supplier cells as the second stage when the networks are stable.
4. Run hourly. The September timing shows a full hourly surface is 3 h of
   wall time; restricting to onshore cells with positive land (about 19,600 of
   52,702) and the fixed-only bundle should bring it near 1 h, so there is no
   need for coarser time steps in the base cases.
5. Publish every accepted surface under one index
   (`results/surfaces/<label>/` with the merged CSV, QA, contract, heatmaps and
   hashes) so the green-dolphin paper, the sensitivity paper and green-porpoise
   symlink to a stable location.

## 4. Scenario design (proposal)

Axes: year (2030, 2050); technology (DEA); finance (uniform YAML default or
Ameli reduced WACC by country); build (flat, or spatial depth x remoteness x
labour); water (flat 2 USD/m³, or spatial desalination and pipeline access);
land (20 % base, 2 % sensitivity); PV (fixed base, tracking sensitivity).

| # | Label | Year | Finance | Build/remoteness | Water | Land | Purpose |
|---|---|---|---|---|---|---|---|
| S1 | `dea_2050_flat_20pct` | 2050 | uniform | flat | flat | 20 % | base surface and base network |
| S2 | `dea_2050_spatial_build_remote_water_amelired_20pct` | 2050 | Ameli | spatial | spatial | 20 % | all spatial mechanisms; second network |
| S3 | `dea_2050_flat_amelired_20pct` | 2050 | Ameli | flat | flat | 20 % | finance alone |
| S4 | `dea_2050_spatial_build_remote_20pct` | 2050 | uniform | spatial | flat | 20 % | build and remoteness alone |
| S5 | `dea_2050_spatial_water_20pct` | 2050 | uniform | flat | spatial | 20 % | water alone |
| S6 | `dea_2030_flat_20pct` | 2030 | uniform | flat | flat | 20 % | 2030 base |
| S7 | `dea_2030_spatial_build_remote_water_amelired_20pct` | 2030 | Ameli | spatial | spatial | 20 % | 2030 all spatial |
| S8 | `dea_2050_flat_2pct` | 2050 | uniform | flat | flat | 2 % | land-share sensitivity, link to September |

S1, S2 and S8 give the sensitivity paper its one-at-a-time decomposition with
S3 to S5 (each mechanism alone against S1, and all together in S2). Networks:
S1, S2, S6, S7 and S8 through green-porpoise with the deposited equations at
the 1 Mt/yr cutoff, alongside the legacy runs A, E and F already done. Eight
hourly surfaces are about 4,600 CPU-hours, or 3 h of wall time each on four
48-core jobs, and all can queue at once on `--clusters=all`.

Open choices for you: the uniform WACC of the flat case (the YAML default is
10 %; a lower uniform value such as 8 % would make the finance sensitivity
symmetric around Ameli's 5 to 10 % range); whether to add a tracking-PV twin of
S1; whether to run Way 2050 at 20 % for continuity with the replication work;
and whether 2 h instead of 1 h is acceptable for S3 to S5 if the queue is slow.

## 5. Deliverables after the campaign

- Legacy: land-availability and LCOA heatmaps for both legacy variants
  (archived reproduction; as stated) and the replication network E.
- Green-lory replication: the September replication surface and network B,
  now with the co-located, 20 % land as a bridge case (S8 at 2 % shows the
  effect of the share alone).
- Key global runs: S1 to S8 surfaces with QA and contracts; networks for the
  five listed; the one-at-a-time decomposition maps and tables for the
  sensitivity paper; supplier-eligibility and production maps for the
  green-dolphin paper.
- A short model description (plant, land, capacity, accounting, limitations)
  suitable for the papers' methods sections, drawn from this document and the
  decision document.

## 6. Steps and what needs confirmation

1. Land build (ARC, `arc/jobs/00_build_land_constraints.sh`, 128 GB, hours):
   add the centered anchor, co-located and fixed-PV settings; derive 2 % and
   20 %. Not a plant run, but a multi-hour job: submitted after your go-ahead.
2. Validate the new land table at the focal cells and against the legacy
   step (areas, densities, anchor), then rerun the three-cell supply pilots on
   it (small runs, no confirmation needed).
3. Stage a release; run S1 first (smoke, diagnostic, global), inspect, then
   the rest. Each global submission waits for your confirmation.
4. Networks with the new contracts; comparison tables and heatmaps; update the
   decision document.
