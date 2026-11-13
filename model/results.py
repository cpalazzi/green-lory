"""Results: surfaces and networks by run id, and the tables and maps the notebooks show.

Reads only; nothing here solves. Run ids are those of runs.yaml; where a run's output is comes
from the ``location`` column of run_store.csv, whose ARC paths map to the local copies under
results/campaigns/.

- ``output_dir``, ``load_surface``, ``read_surface``, ``load_network_suppliers``: outputs by run
  id (``read_surface`` by folder, for an execution run_store.csv no longer points to);
  ``runs_table`` and ``copy_command``: what run_store.csv records and how to copy an output from ARC;
- ``site_cost_table``, ``depth_classes``: a run's site-cost table and the depth class of each sea cell;
- ``at_cells``, ``focal_cell_table``: rows at named cells (the three focal cells by default);
- ``coast_distance_bands``, ``lcoa_by_band``, ``production_by_band``: coast against inland;
- ``lowest_feasible_plant``, ``select_plants``: one plant per cell from a land run and a sea run,
  for combined maps;
- ``plot_map`` (interactive, plotly) and ``plot_map_static`` (matplotlib, for papers): any
  column of a surface on the 1-degree grid;
- ``plot_maps``: one column of several surfaces as map panels on one colour scale (the results
  notebook), with ``summary_note`` and ``site_note`` for a panel's box of results;
- ``supply_curve``, ``cost_at_supply``, ``plot_supply_curves``, ``supply_shares``: the plants
  that can make the reference output, cheapest first, and the cost of the last tonne at a volume;
- ``summary_table``, ``median_by``, ``by_band``, ``cheapest_cells``, ``wind_share``,
  ``compare_surfaces``: tables of one or several surfaces;
- ``cost_factors``, ``cell_regions``, ``plot_region_costs``: what each location factor adds to
  the cost of each cell, as stacked bars by region (UN M49 sub-regions by default);
- ``plot_build_factors``: the remote and cold factors on the capital cost against travel time,
  with the Australian location factors they were checked against and the plants by travel time;
- ``potential_by_cost``, ``plot_potential_by_cost``, ``potential_below``: production potential by
  cost band and UN region, as stacked histograms and a table;
- ``land_rule_list``, ``land_cover_shares``, ``land_share_medians``: what sets the land a plant may
  use, the share of each land-cover class open to wind and PV, and the median share of a cell's
  land that each rule keeps;
- ``potential_by_land_rule``, ``plot_land_rule_maps``, ``land_by_latitude``: production potential
  as the land rules apply one after another, from every km² of land to the run's own potential, by
  cell, and what sets it by band of latitude;
- ``share_sensitivity``, ``plants_by_share``: the effect of a land share on the plants, potential
  and costs of each case, and on the plants of each region.
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Dict, Mapping, Sequence

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from . import runs
from .config import CELLS_DIR, COUNTRIES_GEOJSON, FINANCE_DIR, REPO_ROOT, repo_path
from .parameters import parameter

FOCAL_CELLS = {"Atacama": (-23.0, -69.0), "NW Australia": (-23.0, 117.0), "Central Australia": (-21.0, 135.0)}
CAPACITY_COLUMN = "scaled_design_max_gridless_onshore_ammonia_capacity_t"
REFERENCE_OUTPUT_MT_PER_YEAR = float(parameter("plant.reference_output_mt_per_year"))
COAST_BANDS = ([-1, 100, 300, 5000], ["coast <100 km", "100-300 km", "inland >300 km"])

# Where the result table sits inside a run's output directory, by kind of output
SURFACE_FILES = (
    "merged/global_run_results.csv",  # a global stage of arc/submit_lory_sequence.sh
    "global_run_results.csv",  # a derived land-share variant
    "merged.csv",  # the September revalidated surfaces
    "suppliers_USD2018.csv",  # a supplier table for the network model (model/suppliers.py export)
)
ARC_CAMPAIGN_PREFIXES = ("/data/engs-df-green-ammonia/engs2523/green-lory-campaigns/", "green-lory-campaigns/")


# --- outputs by run id ------------------------------------------------------------------------

def output_dir(run_id: str, *, campaigns_dir: str | Path = runs.LOCAL_CAMPAIGNS_DIR) -> Path:
    """The local directory holding a run's output, from its run_store.csv location."""
    record = runs.run_record(run_id)
    location = record["location"].strip()
    if not location:
        raise runs.RunDefinitionError(f"{run_id} has no output location in run_store.csv (status {record['status']})")
    if "; local copy " in location:
        return Path(campaigns_dir) / location.split("; local copy ", 1)[1].strip()
    for prefix in ARC_CAMPAIGN_PREFIXES:
        if location.startswith(prefix):
            return Path(campaigns_dir) / location[len(prefix):]
    path = Path(location)
    return path if path.is_absolute() else REPO_ROOT / path


def read_surface(directory: str | Path) -> pd.DataFrame:
    """The result table in an output folder (the first of SURFACE_FILES found), such as an
    earlier execution of a run that run_store.csv no longer points to."""
    directory = Path(directory)
    for name in SURFACE_FILES:
        if (directory / name).is_file():
            return pd.read_csv(directory / name, low_memory=False)
    raise FileNotFoundError(f"none of {', '.join(SURFACE_FILES)} in {directory}")


def load_surface(run_id: str, *, campaigns_dir: str | Path = runs.LOCAL_CAMPAIGNS_DIR) -> pd.DataFrame:
    """The result table of a surface (or a supplier table for the network model) by run id."""
    try:
        return read_surface(output_dir(run_id, campaigns_dir=campaigns_dir))
    except FileNotFoundError as error:
        raise FileNotFoundError(f"{run_id}: {error}") from None


def load_network_suppliers(network_id: str, *, campaigns_dir: str | Path = runs.LOCAL_CAMPAIGNS_DIR) -> pd.DataFrame:
    """The suppliers.csv of a network run (production by supplier cell)."""
    return pd.read_csv(output_dir(network_id, campaigns_dir=campaigns_dir) / "suppliers.csv")


def runs_table(run_ids: Mapping[str, str]) -> pd.DataFrame:
    """What run_store.csv records for each run (name -> run id): its finance, build, water and
    weather options, its status, and whether its output is in the local copies."""
    rows = []
    for name, run_id in run_ids.items():
        record = runs.run_record(run_id)
        try:
            local = output_dir(run_id).is_dir()
        except runs.RunDefinitionError:
            local = False
        rows.append({"name": name, "run_id": run_id,
                     **{key: record[key] for key in ("finance", "build", "water", "weather", "status")},
                     "local copy": local})
    return pd.DataFrame(rows).set_index("name")


def copy_command(run_id: str) -> str:
    """The rsync command that copies a run's output folder from ARC to its local copy (through the
    SSH control socket of the arc-oxford host)."""
    local = output_dir(run_id)
    remote = ARC_CAMPAIGN_PREFIXES[0] + local.relative_to(runs.LOCAL_CAMPAIGNS_DIR).as_posix()
    return (f'mkdir -p "{local}" && rsync -rltm -e "ssh -o BatchMode=yes -S /tmp/arc-green-lory.sock" '
            f'arc-oxford:{remote}/ "{local}/"')


# --- site costs -------------------------------------------------------------------------------

# Sea plants by mean water depth (inputs/site_costs/README.md, "Plants at sea"), shallow to deep
DEPTH_CLASSES = {"fixed": "fixed, to 70 m", "floating": "floating, 70 to 1,000 m", "deep": "deep, beyond 1,000 m"}


def site_cost_table(run_id: str) -> pd.DataFrame:
    """The site-cost table of a run (site_costs_csv in runs.yaml; a derived run's is its solved
    parent's), one row per cell."""
    entries = runs.load_runs()
    solved = runs._lineage(run_id, entries)[-1]
    run = entries[solved]
    if not run.get("site_costs_csv"):
        raise runs.RunDefinitionError(f"{solved} has no site_costs_csv in runs.yaml")
    return pd.read_csv(repo_path(run["site_costs_csv"]))


def protected_areas(cells: pd.DataFrame | None = None) -> pd.DataFrame:
    """The protected areas of each cell (inputs/site_costs/protected_areas.csv, written by
    python -m model.site_geography protected-areas): its land and water as the land tables count
    them (land_km2, water_km2), the parts in protected areas (protected_land_km2,
    protected_water_km2) and their shares (protected_land_pct, protected_water_pct), for the cells
    of cells if given."""
    table = pd.read_csv(REPO_ROOT / "inputs" / "site_costs" / "protected_areas.csv")
    return table if cells is None else cells[["latitude", "longitude"]].merge(table, on=["latitude", "longitude"], how="left")


def protected_water_note(table: pd.DataFrame) -> str:
    """A map box for the water of protected_areas: the cells, those with some protected water,
    and the protected share of the water of all the cells."""
    share = table["protected_water_km2"].sum() / table["water_km2"].sum()
    return (f"{len(table):,} cells, {(table['protected_water_pct'] > 0).sum():,} of them with protected water\n"
            f"{100 * share:.1f} % of the water of all of them is protected")


def depth_classes(run_id: str) -> pd.DataFrame:
    """The depth class of each cell of a sea run, named as in DEPTH_CLASSES and ordered from
    shallow to deep, its mean sea depth (m) and the country of the exclusive economic zone that
    holds it (eez_iso3, empty in international waters), from the run's site-cost table."""
    table = site_cost_table(run_id)[["latitude", "longitude", "sea_depth_m", "depth_class", "eez_iso3"]].copy()
    unknown = set(table["depth_class"]) - set(DEPTH_CLASSES)
    if unknown:
        raise ValueError(f"{run_id}: depth classes {sorted(unknown)} are not in DEPTH_CLASSES")
    table["depth_class"] = pd.Categorical(table["depth_class"].map(DEPTH_CLASSES),
                                          categories=list(DEPTH_CLASSES.values()), ordered=True)
    return table


# --- rows at named cells ----------------------------------------------------------------------

def at_cells(frame: pd.DataFrame, cells: Mapping[str, tuple[float, float]] = FOCAL_CELLS, *,
             lat: str = "latitude", lon: str = "longitude") -> pd.DataFrame:
    """One row per named cell, indexed by the cell name; an error if a cell is missing or repeated."""
    picked = []
    for name, (la, lo) in cells.items():
        match = frame[(frame[lat].round(3) == la) & (frame[lon].round(3) == lo)]
        if len(match) != 1:
            raise ValueError(f"{name}: expected one row at ({la}, {lo}), found {len(match)}")
        picked.append(match.assign(cell=name))
    return pd.concat(picked).set_index("cell")


def focal_cell_table(run_ids: list[str], cells: Mapping[str, tuple[float, float]] = FOCAL_CELLS, *,
                     campaigns_dir: str | Path = runs.LOCAL_CAMPAIGNS_DIR) -> pd.DataFrame:
    """LCOA, supplier capacity and design of green-lory surfaces at the named cells.

    One row per run and cell: LCOA in EUR/t, capacity in Mt/yr at the run's land share, wind and
    PV in MW per Mt/yr of ammonia, and the land constraint that limits the capacity.
    """
    rows = []
    for run_id in run_ids:
        for name, row in at_cells(load_surface(run_id, campaigns_dir=campaigns_dir), cells).iterrows():
            rows.append({"run_id": run_id, "cell": name,
                         "lcoa_eur_per_t": float(row["lcoa_eur_per_t"]),
                         "capacity_mt_per_year": float(row[CAPACITY_COLUMN]) / 1e6,
                         "wind_mw_per_mt": float(row["wind_mw"]),
                         "pv_mw_per_mt": float(row["solar_mw"]) + float(row.get("solar_tracking_mw", 0.0)),
                         "limiting": row.get("scaled_design_limiting_constraint")})
    return pd.DataFrame(rows)


# --- coast against inland ---------------------------------------------------------------------

def coast_distance_bands(frame: pd.DataFrame, country: str = "Australia", *, crs: int = 3577,
                         buffer_km: float = 60.0, bands: tuple = COAST_BANDS) -> pd.DataFrame:
    """The cells of one country with the distance from each centre to the coast, in bands.

    The coast is the boundary of the country's polygon in data/countries.geojson, which is the
    coastline only for a country without land borders, such as Australia (the default, in the
    Australian Albers projection, EPSG 3577). Cell centres up to buffer_km offshore are kept, so
    coastal cells whose centre falls in the sea count.
    """
    import geopandas as gpd

    world = gpd.read_file(COUNTRIES_GEOJSON)
    name = next(c for c in world.columns if c.lower() in ("admin", "name", "name_en", "country", "sovereignt"))
    shape = world[world[name].astype(str) == country].to_crs(crs).geometry.union_all()
    points = gpd.GeoSeries(gpd.points_from_xy(frame["longitude"], frame["latitude"]), crs=4326).to_crs(crs)
    inside = points.within(shape.buffer(buffer_km * 1000)).to_numpy()
    banded = frame[inside].copy()
    banded["coast_km"] = [shape.boundary.distance(point) / 1000 for point in points[inside]]
    banded["band"] = pd.cut(banded["coast_km"], bands[0], labels=bands[1])
    return banded


def lcoa_by_band(banded: pd.DataFrame, surfaces: Mapping[str, pd.DataFrame], *,
                 column: str = "lcoa_eur_per_t", bands: tuple = COAST_BANDS) -> pd.DataFrame:
    """Median and 10th percentile of a column by coast band for each named surface.

    Uses the cells common to the banded table and every surface, so the surfaces are compared
    on the same cells.
    """
    frame = banded[["latitude", "longitude", "band"]]
    for label, surface in surfaces.items():
        frame = frame.merge(surface[["latitude", "longitude", column]].rename(columns={column: label}),
                            on=["latitude", "longitude"], how="inner")
    rows = []
    for label in surfaces:
        grouped = frame.groupby("band", observed=True)[label]
        median, p10 = grouped.median(), grouped.quantile(0.1)
        rows.append({"surface": label, "cells": len(frame),
                     **{f"median {band}": median.get(band) for band in bands[1]},
                     **{f"p10 {band}": p10.get(band) for band in bands[1]},
                     "median inland minus coast": median.get(bands[1][-1]) - median.get(bands[1][0]),
                     "p10 inland minus coast": p10.get(bands[1][-1]) - p10.get(bands[1][0])})
    return pd.DataFrame(rows)


def production_by_band(banded: pd.DataFrame, suppliers: Mapping[str, pd.DataFrame], *, iso3: str = "AUS") -> pd.DataFrame:
    """Network production (Mt/yr) of one country by coast band, for each named network run.

    suppliers maps a label to a network's suppliers.csv (load_network_suppliers).
    """
    rows = []
    for label, table in suppliers.items():
        table = table.rename(columns={"Latitude": "latitude", "Longitude": "longitude"})
        active = table[(table["iso3"] == iso3) & (table["production_t_per_year"] > 0)]
        merged = active.merge(banded[["latitude", "longitude", "band"]], on=["latitude", "longitude"], how="left")
        if merged["band"].isna().any():
            raise ValueError(f"{label}: {iso3} suppliers outside the banded cells")
        production = merged.groupby("band", observed=False)["production_t_per_year"].sum() / 1e6
        rows.append({"network": label, "production_mt_per_year": production.sum(), **production.to_dict()})
    return pd.DataFrame(rows)


# --- maps -------------------------------------------------------------------------------------

def lowest_feasible_plant(land: pd.DataFrame, sea: pd.DataFrame,
                          minimum_mt_per_year: float = REFERENCE_OUTPUT_MT_PER_YEAR) -> pd.DataFrame:
    """Each cell's cheapest plant that can make the reference output, from a land and a sea run.

    A cell can hold a land plant and a sea plant, on separate areas, but a map shows one number.
    Every LCOA is that of the reference design (1 Mt/yr), so a plant whose area cannot reach that
    output cannot make ammonia at that cost at scale, and it does not count: the same cutoff as
    the network's suppliers. A cell with no such plant is left out. Column site says which plant
    is shown; its own capacity column gives what the cell can make at that cost, and the supply
    curve, not this table, adds the two plants of a cell.
    """
    plants = pd.concat([land.assign(site="land"), sea.assign(site="sea")], ignore_index=True)
    feasible = plants[plants[CAPACITY_COLUMN] >= minimum_mt_per_year * 1e6]
    cheapest = feasible.sort_values("lcoa_eur_per_t").drop_duplicates(["latitude", "longitude"])
    return cheapest.sort_values(["latitude", "longitude"]).reset_index(drop=True)


# What a map of plants shows, by the sites option of select_plants
SITES = {"land": "land plants", "sea": "sea plants", "both": "the cheaper of each cell's land and sea plants"}


def select_plants(land: pd.DataFrame, sea: pd.DataFrame, sites: str,
                  minimum_mt_per_year: float = REFERENCE_OUTPUT_MT_PER_YEAR) -> pd.DataFrame:
    """The plants a map shows, among those that can make at least minimum_mt_per_year: the land
    plants ("land"), the sea plants ("sea"), or each cell's cheaper plant ("both",
    lowest_feasible_plant). Column site says which."""
    if sites not in SITES:
        raise ValueError(f"sites must be one of {list(SITES)}")
    if sites == "both":
        return lowest_feasible_plant(land, sea, minimum_mt_per_year)
    plants = (land if sites == "land" else sea).assign(site=sites)
    return plants[plants[CAPACITY_COLUMN] >= minimum_mt_per_year * 1e6].reset_index(drop=True)


def _cell_id(lat: float, lon: float, decimals: int = 2) -> str:
    return f"{lat:.{decimals}f}_{lon:.{decimals}f}"


def _build_cell_geojson(
    df: pd.DataFrame,
    lat_col: str = "latitude",
    lon_col: str = "longitude",
    id_col: str = "cell_id",
    cell_size_deg: float = 1.0,
) -> Dict:
    half = cell_size_deg / 2.0
    features = []
    for lat, lon, cell_id in zip(df[lat_col], df[lon_col], df[id_col]):
        lat0 = float(lat)
        lon0 = float(lon)
        polygon = [
            [lon0 - half, lat0 - half],
            [lon0 - half, lat0 + half],
            [lon0 + half, lat0 + half],
            [lon0 + half, lat0 - half],
            [lon0 - half, lat0 - half],
        ]
        features.append(
            {
                "type": "Feature",
                "id": cell_id,
                "properties": {"latitude": lat0, "longitude": lon0},
                "geometry": {"type": "Polygon", "coordinates": [polygon]},
            }
        )
    return {"type": "FeatureCollection", "features": features}


def _country_border_trace(
    geojson_path: Path,
    line_color: str = "rgba(60,60,60,0.45)",
    line_width: float = 0.7,
) -> "go.Scattergeo":
    """Build a Scattergeo line trace of country borders from a GeoJSON file."""
    with open(geojson_path) as fh:
        countries = json.load(fh)

    lats: list = []
    lons: list = []
    for feature in countries.get("features", []):
        geom = feature.get("geometry", {})
        gtype = geom.get("type", "")
        coords = geom.get("coordinates", [])
        if gtype == "Polygon":
            rings = [coords[0]]  # outer ring only
        elif gtype == "MultiPolygon":
            rings = [poly[0] for poly in coords]  # outer ring of each sub-polygon
        else:
            continue
        for ring in rings:
            lons.extend([pt[0] for pt in ring] + [None])
            lats.extend([pt[1] for pt in ring] + [None])

    return go.Scattergeo(
        lon=lons,
        lat=lats,
        mode="lines",
        line=dict(color=line_color, width=line_width),
        hoverinfo="skip",
        showlegend=False,
    )


def plot_map(
    csv_path: "str | Path | pd.DataFrame",
    color_column: str = "auto",
    cell_size_deg: float = 1.0,
    color_scale: str = "Viridis_r",
    range_color: "tuple[float, float] | None" = None,
    percentile_clip: "tuple[float, float]" = (0.02, 0.98),
    show_country_borders: bool = True,
    country_border_color: str = "rgba(60,60,60,0.5)",
    country_border_width: float = 0.5,
    cell_opacity: float = 0.85,
    projection: str = "equirectangular",
    lat_range: "tuple[float, float] | None" = None,
    colorbar_len: float = 0.85,
):
    """An interactive map of one column of a surface on its 1-degree cells (LCOA by default).

    Args:
        csv_path: Path to a results CSV **or** a pre-filtered ``pd.DataFrame``.
        range_color: Explicit (min, max) for the colour scale.  If *None*,
            auto-clipped to *percentile_clip* quantiles.
        percentile_clip: (lo, hi) quantiles used when *range_color* is None.
        show_country_borders: Overlay faint country border lines (default True).
        projection: Plotly geo projection name.  ``"equirectangular"`` (default,
            plate carrée) maps lon→x and lat→y directly so every gridline is a
            perfectly straight horizontal or vertical rule — easiest to read
            lat/lon coordinates from.  Other choices: ``"eckert4"`` (equal-area,
            rectangular feel), ``"natural earth"`` (rounder, publication style),
            ``"robinson"``.
        lat_range: (south, north) latitude limits for the geo background
            (land/ocean fill).  ``None`` (default) auto-derives these from the
            actual data extent — the background clips to the top/bottom of the
            outermost data cells with a half-cell margin, so no empty polar
            background is visible.  Pass an explicit tuple to override.

            Note: ``lataxis_range`` clips Plotly's built-in land/ocean
            background but *not* the choropleth cell polygons.  This is why
            the auto-derived value only needs to match the data extent rather
            than a hard geographic bound.
        colorbar_len: Fraction of the figure height for the colour bar.
            Natural Earth fills the figure differently depending on the
            figure's aspect ratio.  ``0.85`` (default) fits a wide export
            figure (~1400 px wide); narrow notebook panels may need ~0.6–0.7.
    """
    if isinstance(csv_path, pd.DataFrame):
        df = csv_path.copy()
        df.columns = [col.lower() for col in df.columns]
    else:
        csv_path = Path(csv_path)
        df = pd.read_csv(csv_path)
    df.columns = [col.lower() for col in df.columns]
    color_column = color_column.lower()
    if color_column in {"auto", "lcoa_currency_per_t"}:
        inferred = None
        if "currency" in df.columns:
            currencies = (
                df["currency"].dropna().astype(str).str.strip().str.lower().unique()
            )
            if len(currencies) == 1:
                candidate = f"lcoa_{currencies[0]}_per_t"
                if candidate in df.columns:
                    inferred = candidate
        if inferred is None and "lcoa_usd_per_t" in df.columns:
            inferred = "lcoa_usd_per_t"
        if inferred is None:
            lcoa_cols = [
                col for col in df.columns if col.startswith("lcoa_") and col.endswith("_per_t")
            ]
            if lcoa_cols:
                inferred = lcoa_cols[0]
        if inferred is not None:
            color_column = inferred
    lat_col = "latitude" if "latitude" in df.columns else None
    lon_col = "longitude" if "longitude" in df.columns else None
    required = {color_column}
    if lat_col:
        required.add(lat_col)
    if lon_col:
        required.add(lon_col)
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Input is missing required columns: {sorted(missing)}")

    plot_df = df.dropna(subset=[color_column]).copy()
    if plot_df.empty:
        raise ValueError("No rows with colour metric available.")

    # Derive lat_range from actual data extent so the land/ocean background
    # clips to exactly the outermost cell edges with a tiny margin.
    if lat_range is None:
        _half = cell_size_deg / 2.0 + 0.1
        lat_range = (
            float(plot_df[lat_col].min()) - _half,
            float(plot_df[lat_col].max()) + _half,
        )

    # Compute colour range: explicit override or percentile clip.
    if range_color is None:
        lo_q, hi_q = percentile_clip
        range_color = (
            float(plot_df[color_column].quantile(lo_q)),
            float(plot_df[color_column].quantile(hi_q)),
        )

    plot_df["cell_id"] = [_cell_id(lat, lon) for lat, lon in zip(plot_df[lat_col], plot_df[lon_col])]
    geojson = _build_cell_geojson(plot_df, lat_col=lat_col, lon_col=lon_col, cell_size_deg=cell_size_deg)

    wind_power_cols = [
        col
        for col in (
            "wind_mw",
            "wind",
        )
        if col in plot_df.columns
    ]
    if wind_power_cols:
        plot_df["wind_total_mw"] = plot_df[wind_power_cols].fillna(0.0).sum(axis=1)
    solar_col = "solar_mw" if "solar_mw" in plot_df.columns else "solar"
    wind_cost_cols = [
        col
        for col in (
            "tech_share_wind_pct",
        )
        if col in plot_df.columns
    ]
    if wind_cost_cols:
        plot_df["tech_share_wind_total_pct"] = plot_df[wind_cost_cols].fillna(0.0).sum(axis=1)
    land_onshore_col = (
        "land_onshore_pct"
        if "land_onshore_pct" in plot_df.columns
        else "percent_onshore_pct"
        if "percent_onshore_pct" in plot_df.columns
        else "percent_onshore"
    )
    hover_candidates = [
        "country",
        color_column,
        "annual_ammonia_demand_mwh",
        "wind_total_mw",
        solar_col,
        "tech_share_wind_total_pct",
        "tech_share_solar_pct",
        "tech_share_electrolyser_pct",
        land_onshore_col,
        "build_cost_multiplier",
        "elevation_m",
    ]
    hover_data = {col: True for col in hover_candidates if col in plot_df.columns}

    fig = px.choropleth(
        plot_df,
        geojson=geojson,
        locations="cell_id",
        featureidkey="id",
        color=color_column,
        color_continuous_scale=color_scale,
        range_color=range_color,
        hover_data=hover_data or None,
        projection=projection,
    )
    fig.update_traces(marker_line_width=0.0, marker_opacity=cell_opacity)
    fig.update_geos(
        projection_type=projection,
        lataxis_range=list(lat_range),
        # Land / ocean background — subtle, so the LCOA colour dominates
        showland=True,
        landcolor="#f0f0ee",
        showocean=True,
        oceancolor="#dde8f0",
        showlakes=False,
        # Built-in coastlines are used when show_country_borders is False;
        # when True, the GeoJSON overlay provides higher-fidelity borders.
        showcoastlines=not show_country_borders,
        coastlinecolor="rgba(80,80,80,0.5)",
        coastlinewidth=0.5,
        showframe=False,
        bgcolor="white",
    )
    fig.update_layout(
        margin=dict(l=0, r=0, t=30, b=0),
        coloraxis_colorbar=dict(lenmode="fraction", len=colorbar_len, yanchor="middle", y=0.5, thickness=15),
    )

    if show_country_borders:
        geojson_path = COUNTRIES_GEOJSON
        if geojson_path.exists():
            border_trace = _country_border_trace(
                geojson_path,
                line_color=country_border_color,
                line_width=country_border_width,
            )
            fig.add_trace(border_trace)

    return fig


def plot_map_static(
    csv_path: "str | Path | pd.DataFrame",
    color_column: str = "auto",
    cell_size_deg: float = 1.0,
    color_scale: str = "viridis_r",
    range_color: "tuple[float, float] | None" = None,
    percentile_clip: "tuple[float, float]" = (0.02, 0.98),
    show_country_borders: bool = True,
    country_border_color: str = "#444444",
    country_border_width: float = 0.35,
    lat_range: "tuple[float, float] | None" = None,
    lon_range: "tuple[float, float]" = (-180.0, 180.0),
    land_color: str = "#f0f0ee",
    ocean_color: str = "#dde8f0",
    lat_dtick: int = 30,
    lon_dtick: int = 30,
    colorbar_label: str = "LCOA (EUR/t NH\u2083)",
    figsize: "tuple[float, float]" = (14.0, 6.0),
) -> tuple:
    """A static map of one column of a surface, for papers (LCOA by default).

    Uses an equirectangular (plate carrée) projection — latitude and longitude
    map directly to y/x axes — with a proper cartographic cornice: inward tick
    marks on all four sides, labelled parallels and meridians, and a thin
    surrounding frame.

    Returns:
        (fig, ax) — matplotlib Figure and Axes objects.
    """
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    from matplotlib.collections import PatchCollection, LineCollection
    from matplotlib.patches import Polygon as MplPolygon
    from matplotlib.cm import ScalarMappable

    # ── Load & normalise data ────────────────────────────────────────────────
    if isinstance(csv_path, pd.DataFrame):
        df = csv_path.copy()
        df.columns = [col.lower() for col in df.columns]
    else:
        df = pd.read_csv(Path(csv_path))
        df.columns = [col.lower() for col in df.columns]

    color_column = color_column.lower()
    if color_column in {"auto", "lcoa_currency_per_t"}:
        inferred = None
        if "currency" in df.columns:
            currencies = (
                df["currency"].dropna().astype(str).str.strip().str.lower().unique()
            )
            if len(currencies) == 1:
                candidate = f"lcoa_{currencies[0]}_per_t"
                if candidate in df.columns:
                    inferred = candidate
        if inferred is None and "lcoa_usd_per_t" in df.columns:
            inferred = "lcoa_usd_per_t"
        if inferred is None:
            lcoa_cols = [
                c for c in df.columns
                if c.startswith("lcoa_") and c.endswith("_per_t")
            ]
            if lcoa_cols:
                inferred = lcoa_cols[0]
        if inferred is not None:
            color_column = inferred

    lat_col = "latitude"
    lon_col = "longitude"
    plot_df = df.dropna(subset=[color_column]).copy()

    if lat_range is None:
        _half = cell_size_deg / 2.0 + 0.1
        lat_range = (
            float(plot_df[lat_col].min()) - _half,
            float(plot_df[lat_col].max()) + _half,
        )

    if range_color is None:
        lo_q, hi_q = percentile_clip
        range_color = (
            float(plot_df[color_column].quantile(lo_q)),
            float(plot_df[color_column].quantile(hi_q)),
        )

    # ── Build 2D raster grid ─────────────────────────────────────────────────
    half = cell_size_deg / 2.0
    lat_min = float(plot_df[lat_col].min())
    lat_max = float(plot_df[lat_col].max())
    lon_min_d = float(plot_df[lon_col].min())
    lon_max_d = float(plot_df[lon_col].max())

    n_lat = round((lat_max - lat_min) / cell_size_deg) + 1
    n_lon = round((lon_max_d - lon_min_d) / cell_size_deg) + 1
    grid = np.full((n_lat, n_lon), np.nan)

    lat_idx = np.round(
        (plot_df[lat_col].values - lat_min) / cell_size_deg
    ).astype(int)
    lon_idx = np.round(
        (plot_df[lon_col].values - lon_min_d) / cell_size_deg
    ).astype(int)
    valid = (lat_idx >= 0) & (lat_idx < n_lat) & (lon_idx >= 0) & (lon_idx < n_lon)
    grid[lat_idx[valid], lon_idx[valid]] = plot_df[color_column].values[valid]

    img_extent = [lon_min_d - half, lon_max_d + half, lat_min - half, lat_max + half]

    # ── Colormap ──────────────────────────────────────────────────────────────
    cmap_name = color_scale.lower().replace(" ", "_")
    try:
        cmap = plt.get_cmap(cmap_name).copy()
    except (ValueError, AttributeError):
        cmap = plt.get_cmap("viridis_r").copy()
    cmap.set_bad(alpha=0.0)  # NaN cells → fully transparent
    norm = mcolors.Normalize(vmin=range_color[0], vmax=range_color[1])

    # ── Figure & axes ─────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_facecolor(ocean_color)  # ocean background

    # ── Land fill & border source ─────────────────────────────────────────────
    geojson_path = COUNTRIES_GEOJSON
    land_patches = []
    border_segments: list = []

    if geojson_path.exists():
        with open(geojson_path) as fh:
            countries = json.load(fh)

        for feature in countries.get("features", []):
            geom = feature.get("geometry", {})
            gtype = geom.get("type", "")
            coords = geom.get("coordinates", [])

            if gtype == "Polygon":
                rings = [coords[0]]
            elif gtype == "MultiPolygon":
                rings = [p[0] for p in coords]
            else:
                continue

            for ring in rings:
                xy = np.array(ring, dtype=float)
                land_patches.append(MplPolygon(xy, closed=True))
                if show_country_borders:
                    border_segments.append(xy)

        land_col = PatchCollection(
            land_patches, facecolor=land_color, edgecolor="none",
            linewidth=0, zorder=1,
        )
        ax.add_collection(land_col)

    # ── Data raster ───────────────────────────────────────────────────────────
    ax.imshow(
        grid, extent=img_extent, origin="lower", aspect="auto",
        cmap=cmap, norm=norm, interpolation="none", zorder=2,
    )

    # ── Country borders ───────────────────────────────────────────────────────
    if show_country_borders and border_segments:
        lc = LineCollection(
            [seg for seg in border_segments],
            colors=country_border_color,
            linewidths=country_border_width,
            zorder=3,
        )
        ax.add_collection(lc)

    # ── Axis limits ───────────────────────────────────────────────────────────
    ax.set_xlim(lon_range)
    ax.set_ylim(lat_range)

    # ── Tick marks & labels ───────────────────────────────────────────────────
    lon_major = list(range(
        int(np.ceil(lon_range[0] / lon_dtick)) * lon_dtick,
        int(lon_range[1]) + 1,
        lon_dtick,
    ))
    lat_major = [
        l for l in range(-90, 91, lat_dtick)
        if lat_range[0] - 1 <= l <= lat_range[1] + 1
    ]

    def _fmt_lon(v: int) -> str:
        if v == 0:
            return "0°"
        return f"{v}°E" if v > 0 else f"{-v}°W"

    def _fmt_lat(v: int) -> str:
        if v == 0:
            return "0°"
        return f"{v}°N" if v > 0 else f"{-v}°S"

    ax.set_xticks(lon_major)
    ax.set_xticklabels([_fmt_lon(v) for v in lon_major], fontsize=8)
    ax.set_yticks(lat_major)
    ax.set_yticklabels([_fmt_lat(v) for v in lat_major], fontsize=8)

    # Minor ticks at 10° lat / 10° lon
    lon_minor_step = max(lon_dtick // 3, 10)
    lat_minor_step = max(lat_dtick // 3, 5)
    lon_minor = [
        v for v in range(int(lon_range[0]), int(lon_range[1]) + 1, lon_minor_step)
        if v not in lon_major
    ]
    lat_minor = [
        l for l in range(-90, 91, lat_minor_step)
        if l not in lat_major and lat_range[0] - 1 <= l <= lat_range[1] + 1
    ]
    ax.set_xticks(lon_minor, minor=True)
    ax.set_yticks(lat_minor, minor=True)

    # Graticule lines for all ticks (major slightly more visible than minor)
    for _lon in lon_major:
        ax.axvline(_lon, color="#888888", alpha=0.35, linewidth=0.55, zorder=3)
    for _lat in lat_major:
        ax.axhline(_lat, color="#888888", alpha=0.35, linewidth=0.55, zorder=3)
    for _lon in lon_minor:
        ax.axvline(_lon, color="#888888", alpha=0.18, linewidth=0.35, zorder=3)
    for _lat in lat_minor:
        ax.axhline(_lat, color="#888888", alpha=0.18, linewidth=0.35, zorder=3)

    # Inward ticks on all four sides — long enough to be clearly visible
    ax.tick_params(
        axis="both", which="major",
        direction="in", length=10, width=0.9,
        top=True, right=True, zorder=6,
    )
    ax.tick_params(
        axis="both", which="minor",
        direction="in", length=5, width=0.6,
        top=True, right=True,
        labeltop=False, labelright=False,
        labelbottom=False, labelleft=False,
        zorder=6,
    )
    # Re-enable bottom/left labels (minor tick_params blanket-disabled them)
    ax.tick_params(axis="x", which="major", labelbottom=True)
    ax.tick_params(axis="y", which="major", labelleft=True)

    for spine in ax.spines.values():
        spine.set_linewidth(0.8)

    # ── Colorbar ──────────────────────────────────────────────────────────────
    sm = ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cb = fig.colorbar(sm, ax=ax, fraction=0.025, pad=0.01, aspect=30)
    cb.set_label(colorbar_label, fontsize=9)
    cb.ax.tick_params(labelsize=8)

    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))
    return fig, ax


# --- campaign maps, supply curves and tables (notebooks/results.ipynb) -------------------------

# Colours, from the dataviz skill's validated palette, as in the figures of the green-serval paper.
# Each quantity mapped has its own one-hue ramp, light to dark: the palette's blue ramp, and ramps
# through its categorical slots for the others (a light tint, the slot, a dark shade). The factors
# keep the same hue on maps and bars: construction orange, water blue, finance aqua.
RAMPS = {
    "blue": ("#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"),  # water
    "orange": ("#fde4d8", "#eb6834", "#7a2e0e"),  # construction
    "aqua": ("#d2f1e4", "#1baf7a", "#0b4f37"),  # finance
    "violet": ("#c3b9f2", "#6250d6", "#241a66"),  # production potential; its lightest step stays clear of NO_DATA
}
# The levelised cost of ammonia: matplotlib's viridis, reversed so that yellow is the cheapest
# (perceptually uniform, readable with colour-vision deficiency, and with no grey in it)
LCOA_COLOURMAP = "viridis_r"
SEQUENTIAL = RAMPS["blue"]
DIVERGING = ("#0d366b", "#256abf", "#6da7ec", "#b7d3f6", "#f0efec", "#f6c7c3", "#ec8a84", "#d4413f", "#8e1f1d")
SERIES = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4")  # categorical slots 1-5, in this order
FACTOR_COLOURS = {"base": "#b5b4ae", "construction": SERIES[1], "water": SERIES[0], "finance": SERIES[2],
                  "interaction": SERIES[3]}
# Ordered steps of a slate ramp (sea depth), each darker than the grey of land cells
DEPTH_COLOURS = ("#a3b3c6", "#6f86a1", "#25364d")
INK, MUTED, GRID, NO_DATA, ABOVE = "#0b0b0b", "#52514e", "#d9d8d4", "#e4e3df", "#9b9a95"
SCALES = ("linear", "change")


def cell_grid(frame: pd.DataFrame, column: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """A column of a surface on the global one-degree grid, for pcolormesh: the longitude and
    latitude edges, and the values by latitude (rows) and longitude (columns), NaN where the
    surface has no cell. Cell centres are whole degrees."""
    latitude = frame["latitude"].to_numpy(dtype=float)
    longitude = frame["longitude"].to_numpy(dtype=float)
    if not (np.allclose(latitude, np.round(latitude)) and np.allclose(longitude, np.round(longitude))):
        raise ValueError("cell centres must be whole degrees")
    values = np.full((181, 361), np.nan)
    values[np.round(latitude).astype(int) + 90, np.round(longitude).astype(int) + 180] = frame[column].to_numpy(dtype=float)
    return np.arange(-180.5, 181.0), np.arange(-90.5, 91.0), values


def _two_figures(value: float, *, up: bool) -> float:
    """value rounded up or down to two significant figures."""
    if value == 0 or not np.isfinite(value):
        return float(value)
    step = 10.0 ** (np.floor(np.log10(abs(value))) - 1)
    return float((np.ceil if up else np.floor)(value / step) * step)


def _colour_scale(values: pd.Series, scale: str, limits: tuple[float, float] | None, ramp: Sequence[str] | str):
    """The colour map, the normalisation and the colour bar's arrows ("extend") of a map's values."""
    import matplotlib
    from matplotlib.colors import LinearSegmentedColormap, Normalize

    if scale not in SCALES:
        raise ValueError(f"scale must be one of {SCALES}")
    values = values.dropna()
    if limits is None and scale == "change":
        bound = _two_figures(float(values.abs().quantile(0.98)), up=True)
        limits = (-bound, bound)
    elif limits is None:
        limits = (_two_figures(float(values.quantile(0.02)), up=False),
                  _two_figures(float(values.quantile(0.98)), up=True))
    low, high = float(limits[0]), float(limits[1])
    if scale == "change":
        cmap = LinearSegmentedColormap.from_list(scale, DIVERGING)
    else:
        cmap = matplotlib.colormaps[ramp] if isinstance(ramp, str) else LinearSegmentedColormap.from_list(scale, list(ramp))
    norm = Normalize(low, high)
    below, above = bool((values < low).any()), bool((values > high).any())
    extend = "both" if below and above else "min" if below else "max" if above else "neither"
    return cmap, norm, extend


@lru_cache(maxsize=1)
def _country_borders():
    import geopandas as gpd

    return gpd.read_file(COUNTRIES_GEOJSON).boundary


def _degrees(value: float, hemispheres: str) -> str:
    """60°E, 30°S or 0°."""
    return "0°" if value == 0 else f"{abs(value):g}°{hemispheres[0] if value > 0 else hemispheres[1]}"


def _map_axes(ax, extent: tuple[float, float, float, float]) -> None:
    """Country borders, a graticule every 30 degrees of latitude and 60 of longitude, and the extent."""
    from matplotlib.ticker import FuncFormatter

    if COUNTRIES_GEOJSON.exists():
        _country_borders().plot(ax=ax, color=MUTED, linewidth=0.3, zorder=3)
    ax.set_xlim(extent[0], extent[1])
    ax.set_ylim(extent[2], extent[3])
    ax.set_aspect("equal")
    ax.set_xticks(np.arange(-180, 181, 60))
    ax.set_yticks([latitude for latitude in range(-90, 91, 30) if extent[2] <= latitude <= extent[3]])
    ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _: _degrees(value, "EW")))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: _degrees(value, "NS")))
    ax.tick_params(labelsize=7, labelcolor=MUTED, length=0)
    ax.grid(color=GRID, linewidth=0.4)
    for spine in ax.spines.values():
        spine.set_edgecolor(GRID)


def plot_maps(frames: Mapping[str, pd.DataFrame], column: str, *, title: str, label: str, scale: str = "linear",
              limits: tuple[float, float] | None = None, ramp: Sequence[str] | str = SEQUENTIAL, above: str = "",
              categories: Mapping[str, str] | None = None, background: pd.DataFrame | None = None,
              background_label: str = "", notes: Mapping[str, str] | None = None, width: float = 10.0):
    """Maps of one column of several surfaces, one panel each, on one colour scale; returns the figure.

    frames: panel title -> surface ("" for a panel without a title). scale: "linear" for levels, on
    ramp (colours from low to high, such as one of RAMPS, or the name of a matplotlib colour map,
    such as LCOA_COLOURMAP), or "change" for changes, on blue, grey and red, symmetric about zero.
    limits: the ends of the colour scale; by default the 2nd and 98th percentiles of the values
    shown, rounded outwards to two significant figures (for "change", plus and minus the 98th
    percentile of their size). Values beyond the limits take the end colours, a cap, and the colour
    bar then has an arrow. above: a cutoff instead of the cap at the top; values above the upper
    limit take a grey of their own (ABOVE), named above in the key. categories: the colour of each
    value of a categorical column, in the key's order; a key titled label replaces the colour bar.
    background: cells drawn in grey under each panel, such as the land cells under a map of sea
    plants, with background_label in the key. notes: text for a box in the lower left corner of a
    panel (panel title -> text).
    """
    import matplotlib.pyplot as plt
    from matplotlib.colors import BoundaryNorm, ListedColormap
    from matplotlib.patches import Patch

    drawn = [*frames.values(), *([background] if background is not None else [])]
    latitudes = pd.concat([frame["latitude"] for frame in drawn])
    extent = (-180.0, 180.0, max(-90.0, float(latitudes.min()) - 2.0), min(90.0, float(latitudes.max()) + 2.0))
    # the figure's height follows the maps' width (less the tick labels and any colour bar), so that
    # the maps fill their panels and the colour bar is as long as they are
    map_height = (width - (1.8 if categories is None else 0.5)) * (extent[3] - extent[2]) / 360.0
    has_key = categories is not None or (background is not None and bool(background_label)) or bool(above)
    height = len(frames) * (map_height + 0.3 + (0.3 if any(frames) else 0.0)) + 0.5 + (0.6 if has_key else 0.0)
    fig, axes = plt.subplots(len(frames), 1, figsize=(width, height), squeeze=False, layout="constrained")
    axes = list(axes[:, 0])
    values = pd.concat([frame[column] for frame in frames.values()])
    if categories is not None:
        codes = {value: code for code, value in enumerate(categories)}
        cmap = ListedColormap(list(categories.values()))
        norm = BoundaryNorm(np.arange(len(codes) + 1) - 0.5, len(codes))
    else:
        cmap, norm, extend = _colour_scale(values, scale, limits, ramp)
        if above:
            cmap = cmap.with_extremes(over=ABOVE)
    grey_shown = False  # whether any background cell is left uncovered, and so needs its key entry
    for ax, (name, frame) in zip(axes, frames.items()):
        if categories is not None:
            frame = frame.assign(_code=frame[column].astype(object).map(codes).astype(float))
        x, y, cells = cell_grid(frame, "_code" if categories is not None else column)
        if background is not None:
            _, _, grey = cell_grid(background.assign(_drawn=1.0), "_drawn")
            ax.pcolormesh(x, y, grey, cmap=ListedColormap([NO_DATA]), shading="flat", rasterized=True, zorder=1)
            grey_shown = grey_shown or bool((np.isfinite(grey) & ~np.isfinite(cells)).any())
        mesh = ax.pcolormesh(x, y, cells, cmap=cmap, norm=norm, shading="flat", rasterized=True, zorder=1)
        _map_axes(ax, extent)
        if name:
            ax.set_title(name, loc="left", color=INK, fontsize=10)
        if notes and notes.get(name):
            ax.text(0.01, 0.03, notes[name], transform=ax.transAxes, fontsize=8, color=INK, ha="left", va="bottom",
                    zorder=5, bbox={"boxstyle": "round,pad=0.4", "facecolor": "white", "edgecolor": GRID, "alpha": 0.9})
    fig.suptitle(title, color=INK, fontsize=12)
    handles = []
    if categories is not None:
        present = set(values.dropna())
        handles = [Patch(facecolor=colour, label=value) for value, colour in categories.items() if value in present]
    if categories is None and above and bool((values > norm.vmax).any()):
        handles.append(Patch(facecolor=ABOVE, label=above))
    if grey_shown and background_label:
        handles.append(Patch(facecolor=NO_DATA, label=background_label))
    if categories is None:
        bar = fig.colorbar(mesh, ax=axes, extend=extend, shrink=0.95 if len(axes) == 1 else 0.6)
        bar.set_label(label, color=INK)
    if handles:
        fig.legend(handles=handles, title=label if categories is not None else None, loc="outside lower center",
                   ncols=len(handles), frameon=False, fontsize=9)
    return fig


def summary_note(values: pd.Series, unit: str = "", *, signed: bool = False, decimals: int = 0) -> str:
    """The text of a panel's box: the median and the 10th to 90th percentiles of values, and the cells."""
    quantiles = values.quantile([0.1, 0.5, 0.9])
    number = "{:" + ("+" if signed else "") + f",.{decimals}f" + "}"
    low, median, high = (number.format(quantiles[q]) for q in (0.1, 0.5, 0.9))
    unit = f" {unit}" if unit else ""
    return f"Median {median}{unit}\n10th to 90th percentile: {low} to {high}{unit}\n{values.notna().sum():,} cells"


def supply_curve(frame: pd.DataFrame, *, minimum_mt_per_year: float = REFERENCE_OUTPUT_MT_PER_YEAR) -> pd.DataFrame:
    """The plants of a surface that can make at least minimum_mt_per_year, cheapest first, with
    their cumulative production potential in Mt/yr (cumulative_mt_per_year): the supply curve of
    the network model's suppliers, with the cutoff of lowest_feasible_plant."""
    plants = frame[frame[CAPACITY_COLUMN] >= minimum_mt_per_year * 1e6]
    curve = plants.sort_values("lcoa_eur_per_t", kind="stable").reset_index(drop=True)
    curve["cumulative_mt_per_year"] = curve[CAPACITY_COLUMN].cumsum() / 1e6
    return curve


def cost_at_supply(frame: pd.DataFrame, mt_per_year: float, *,
                   minimum_mt_per_year: float = REFERENCE_OUTPUT_MT_PER_YEAR) -> float:
    """The LCOA (EUR/t) of the plant that brings the cumulative potential of the cheapest plants to
    mt_per_year, the cost of the last tonne; NaN if all the plants together make less."""
    curve = supply_curve(frame, minimum_mt_per_year=minimum_mt_per_year)
    reached = curve.loc[curve["cumulative_mt_per_year"] >= mt_per_year, "lcoa_eur_per_t"]
    return float(reached.iloc[0]) if len(reached) else float("nan")


def supply_shares(frames: Mapping[str, pd.DataFrame], mt_per_year: float, *, by: str = "country", top: int = 5,
                  minimum_mt_per_year: float = REFERENCE_OUTPUT_MT_PER_YEAR) -> pd.DataFrame:
    """Where the cheapest mt_per_year of each surface comes from: the share (%) of the potential of
    the plants needed to reach mt_per_year, by the column by (the country by default; "no country"
    where the result table names none, as for the sea plants and some islands). Rows: the top largest of each
    surface, the largest first."""
    shares = {}
    for name, frame in frames.items():
        curve = supply_curve(frame, minimum_mt_per_year=minimum_mt_per_year)
        needed = curve[curve["cumulative_mt_per_year"].shift(fill_value=0.0) < mt_per_year]
        potential = needed.groupby(needed[by].fillna("no country"))[CAPACITY_COLUMN].sum()
        shares[name] = 100.0 * potential / potential.sum()
    table = pd.DataFrame(shares).fillna(0.0)
    rows = set().union(*(table[name].nlargest(top).index for name in table))
    return table.loc[sorted(rows, key=lambda row: -table.loc[row].max())].round(1)


def plot_supply_curves(frames: Mapping[str, pd.DataFrame], *, title: str, xmax: float = 10_000.0,
                       marks: Sequence[float] = (500.0, 2000.0), names: Mapping[str, str] | None = None,
                       minimum_mt_per_year: float = REFERENCE_OUTPUT_MT_PER_YEAR):
    """Supply curves (supply_curve) of up to four surfaces, up to xmax Mt/yr, one colour each
    (SERIES, in the order given) labelled at its right end by its key, and dashed lines at the
    volumes in marks. names: the key's entry for each surface (its key by default)."""
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.ticker import FuncFormatter

    if len(frames) > len(SERIES):
        raise ValueError(f"at most {len(SERIES)} curves")
    fig, ax = plt.subplots(figsize=(9.0, 4.8), layout="constrained")
    shown, ends = [], {}
    for (name, frame), colour in zip(frames.items(), SERIES):
        curve = supply_curve(frame, minimum_mt_per_year=minimum_mt_per_year)
        supply = np.concatenate([[0.0], curve["cumulative_mt_per_year"].to_numpy()])
        cost = curve["lcoa_eur_per_t"].to_numpy()
        cost = np.concatenate([cost[:1], cost])  # each plant's cost over its own stretch of the potential
        ax.step(supply, cost, where="pre", color=colour, linewidth=2, label=(names or {}).get(name, name), zorder=3)
        end = min(int(np.searchsorted(supply, xmax)), len(cost) - 1)
        shown.append(cost[: end + 1])
        ends[name] = float(cost[end])
    for mark in marks:
        ax.axvline(mark, color=MUTED, linewidth=0.8, linestyle="--", zorder=2)
    low = min(float(values.min()) for values in shown)
    high = max(float(values.max()) for values in shown)
    pad = 0.05 * (high - low)
    ax.set_xlim(0, xmax)
    ax.set_ylim(np.floor((low - pad) / 50) * 50, np.ceil((high + pad) / 50) * 50)
    gap = 0.05 * (ax.get_ylim()[1] - ax.get_ylim()[0])  # direct labels at the right edge, nudged apart
    placed: list[float] = []
    for name, y in sorted(ends.items(), key=lambda item: item[1]):
        y = max(y, placed[-1] + gap) if placed else y
        placed.append(y)
        ax.annotate(name, (xmax, y), xytext=(6, 0), textcoords="offset points", color=INK, fontsize=9,
                    va="center", annotation_clip=False)
    handles, _ = ax.get_legend_handles_labels()
    if marks:
        handles.append(Line2D([], [], color=MUTED, linewidth=0.8, linestyle="--",
                              label="Production of " + " and ".join(f"{mark:,g}" for mark in marks) + " Mt/yr"))
    ax.legend(handles=handles, loc="upper left", frameon=False, fontsize=9)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:,.0f}"))
    ax.set_xlabel("Cumulative production potential (Mt/yr)", color=INK)
    ax.set_ylabel("Levelised cost of ammonia (EUR/t)", color=INK)
    ax.set_title(title, loc="left", color=INK, fontsize=12)
    ax.grid(color=GRID, linewidth=0.5)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    return fig


def summary_table(land: Mapping[str, pd.DataFrame], sea: Mapping[str, pd.DataFrame] | None = None, *,
                  volumes: Sequence[float] = (500.0, 2000.0),
                  minimum_mt_per_year: float = REFERENCE_OUTPUT_MT_PER_YEAR) -> pd.DataFrame:
    """The headline numbers of each case (a column per key of land): the cells and the median, 10th
    and 90th percentiles of the LCOA of its land plants and of its sea plants, and the cost of the
    last tonne at each global production volume (cost_at_supply) from its land plants and from its
    land and sea plants together. EUR/t, rounded to whole numbers."""
    columns = {}
    for name, land_plants in land.items():
        column: dict[str, float] = {}
        sites = {"land plants": land_plants, **({"sea plants": sea[name]} if sea is not None else {})}
        for site, plants in sites.items():
            column[f"Cells, {site}"] = len(plants)
            column[f"Median, {site}"] = plants["lcoa_eur_per_t"].median()
            column[f"10th percentile, {site}"] = plants["lcoa_eur_per_t"].quantile(0.1)
            column[f"90th percentile, {site}"] = plants["lcoa_eur_per_t"].quantile(0.9)
        for volume in volumes:
            column[f"Cost at {volume:,g} Mt/yr, land plants"] = cost_at_supply(
                land_plants, volume, minimum_mt_per_year=minimum_mt_per_year)
            if sea is not None:
                column[f"Cost at {volume:,g} Mt/yr, land and sea plants"] = cost_at_supply(
                    pd.concat([land_plants, sea[name]]), volume, minimum_mt_per_year=minimum_mt_per_year)
        columns[name] = column
    return pd.DataFrame(columns).round().astype("Int64")


def median_by(frames: Mapping[str, pd.DataFrame], groups: pd.DataFrame, column: str = "lcoa_eur_per_t") -> pd.DataFrame:
    """The median of column of each surface by group, with the cells of the first surface in each
    group: groups has latitude, longitude and one more column, the group (such as depth_class from
    depth_classes). Rounded to whole numbers."""
    key = ["latitude", "longitude"]
    group = [name for name in groups.columns if name not in key]
    if len(group) != 1:
        raise ValueError("groups needs latitude, longitude and one more column")
    table = None
    for name, frame in frames.items():
        grouped = frame[key + [column]].merge(groups, on=key).groupby(group[0], observed=True)[column]
        if table is None:
            table = grouped.size().rename("cells").to_frame()
        table[name] = grouped.median()
    return table.round().astype("Int64")


def by_band(frame: pd.DataFrame, column: str, by: str, edges: Sequence[float] | None = None) -> pd.DataFrame:
    """The cells and the 10th, 50th and 90th percentiles of column in bands of the column by
    (edges; the lowest edge belongs to the first band) or, without edges, for each value of by;
    where frame has a country column, also the country with the most cells in each."""
    if edges is None:  # values equal to six decimals form one group (rates read from text differ in the last bits)
        groups = frame[by].round(6) if pd.api.types.is_float_dtype(frame[by]) else frame[by]
    else:
        labels = [f"{low:g} to {high:g}" for low, high in zip(edges[:-1], edges[1:])]
        groups = pd.cut(frame[by], list(edges), labels=labels, include_lowest=True)
    grouped = frame.groupby(groups, observed=True)
    table = pd.DataFrame({"cells": grouped.size(), "10th percentile": grouped[column].quantile(0.1),
                          "median": grouped[column].median(), "90th percentile": grouped[column].quantile(0.9)})
    if "country" in frame:
        table["country with the most cells"] = grouped["country"].agg(
            lambda countries: countries.value_counts().index[0] if countries.notna().any() else "")
    table.index.name = by
    return table.round(1)


def wind_share(frame: pd.DataFrame) -> pd.Series:
    """Wind's share of a plant's generating capacity: wind MW over wind and PV MW."""
    pv = frame["solar_mw"] + (frame["solar_tracking_mw"] if "solar_tracking_mw" in frame else 0.0)
    return frame["wind_mw"] / (frame["wind_mw"] + pv)


def cheapest_cells(frame: pd.DataFrame, n: int = 10, *, extra: Sequence[str] = (),
                   minimum_mt_per_year: float = 0.0) -> pd.DataFrame:
    """The n cheapest plants of a surface among those that can make at least minimum_mt_per_year:
    where they are (with the extra columns), their LCOA, wind's share of their generating capacity
    and their production potential (Mt/yr)."""
    cheapest = frame[frame[CAPACITY_COLUMN] >= minimum_mt_per_year * 1e6].nsmallest(n, "lcoa_eur_per_t")
    columns = ["latitude", "longitude", *(["site"] if "site" in cheapest else []), "country", "elevation_m",
               *extra, "lcoa_eur_per_t"]
    table = cheapest[columns].assign(wind_share=wind_share(cheapest),
                                     potential_mt_per_year=cheapest[CAPACITY_COLUMN] / 1e6)
    return table.round({"elevation_m": 0, "lcoa_eur_per_t": 0, "wind_share": 2,
                        "potential_mt_per_year": 1}).reset_index(drop=True)


def compare_surfaces(new: pd.DataFrame, old: pd.DataFrame, *, keep: Sequence[str] = ()) -> pd.DataFrame:
    """The cells of two surfaces, with each one's LCOA and wind share (wind_share; suffixes _new and
    _old), the change in LCOA from old to new in per cent (lcoa_change_pct) and the columns keep of
    new."""
    key = ["latitude", "longitude"]

    def pick(frame: pd.DataFrame, columns: Sequence[str] = ()) -> pd.DataFrame:
        return frame[[*key, *columns]].assign(lcoa_eur_per_t=frame["lcoa_eur_per_t"], wind_share=wind_share(frame))

    both = pick(new, keep).merge(pick(old), on=key, suffixes=("_new", "_old"))
    both["lcoa_change_pct"] = 100.0 * (both["lcoa_eur_per_t_new"] / both["lcoa_eur_per_t_old"] - 1.0)
    return both


def site_note(frame: pd.DataFrame) -> str:
    """The text of a panel's box for a map of land and sea plants (lowest_feasible_plant): the
    number of plants of each site and their median LCOA."""
    lines = []
    for site in ("land", "sea"):
        lcoa = frame.loc[frame["site"] == site, "lcoa_eur_per_t"]
        if len(lcoa):
            lines.append(f"{site.capitalize()} plants: {len(lcoa):,}, median {lcoa.median():,.0f} EUR/t")
    return "\n".join(lines)


# --- what each location factor adds to the cost -----------------------------------------------

# The 16 regions of the finance rates (Ameli et al. 2021, TIAM-UCL), as in the green-serval paper
FINANCE_REGION_NAMES = {
    "AFR": "Africa", "AUS": "Australia and New Zealand", "CAN": "Canada", "CHI": "China",
    "CSA": "Central and South America", "EEU": "Eastern Europe", "FSU": "Former Soviet Union", "IND": "India",
    "JAP": "Japan", "MEA": "Middle East", "MEX": "Mexico", "ODA": "Other developing Asia", "SKO": "South Korea",
    "UK": "United Kingdom", "USA": "United States", "WEU": "Western Europe",
}
# The parts of a cell's LCOA in a campaign's cases, in the order they stack (cost_factors)
FACTOR_NAMES = {"base": "Base case", "construction": "Construction (site costs)", "water": "Water (site costs)",
                "finance": "Finance", "interaction": "Site costs and finance together"}


# Areas of inputs/cells/cell_countries.csv that the UN's M49 table does not list, given the
# region and sub-region of their neighbours
M49_ADDITIONS = {"TWN": ("Asia", "Eastern Asia")}


def cell_regions() -> pd.DataFrame:
    """Each cell's country (iso3, from inputs/cells/cell_countries.csv), its region of the finance
    rates (finance_region, FINANCE_REGION_NAMES, from the Ameli table by country) and its UN region
    and sub-region (un_region, un_sub_region: M49, inputs/cells/un_m49_regions.csv)."""
    countries = pd.read_csv(CELLS_DIR / "cell_countries.csv")[["latitude", "longitude", "iso3"]]
    finance = pd.read_csv(FINANCE_DIR / "ameli2021_lowcarbon_wacc_by_country.csv")[["iso3", "ameli_region"]]
    m49 = pd.read_csv(CELLS_DIR / "un_m49_regions.csv", keep_default_na=False)[["iso3", "region", "sub_region"]]
    m49 = pd.concat([m49, pd.DataFrame([(iso3, *names) for iso3, names in M49_ADDITIONS.items()],
                                       columns=["iso3", "region", "sub_region"])])
    m49 = m49.rename(columns={"region": "un_region", "sub_region": "un_sub_region"}).replace("", np.nan)
    table = countries.merge(finance.drop_duplicates("iso3"), on="iso3", how="left").merge(m49, on="iso3", how="left")
    table["finance_region"] = table["ameli_region"].map(FINANCE_REGION_NAMES)
    return table[["latitude", "longitude", "iso3", "finance_region", "un_region", "un_sub_region"]]


def cost_factors(base: pd.DataFrame, site: pd.DataFrame, finance: pd.DataFrame, both: pd.DataFrame) -> pd.DataFrame:
    """What each location factor adds to the LCOA of each cell (EUR/t), from the four cases of a
    campaign: no location factors (base), site costs (site), finance (finance) and both (both).

    Columns, in FACTOR_NAMES order: base, the base case's LCOA; construction and water, the site
    case's plant cost (lcoa_plant_eur_per_t, which leaves out water) and water cost less the base
    case's; finance, the finance case's LCOA less the base case's; interaction, the rest of the case
    with both, which the two give together, since its cost of capital applies to the raised capital
    cost. They add up to the LCOA of each case. Also the smallest production potential of the four
    cases (potential_mt_per_year) and each cell's regions (cell_regions).
    """
    key = ["latitude", "longitude"]
    columns = ["lcoa_eur_per_t", "lcoa_plant_eur_per_t", "water_cost_eur_per_t", CAPACITY_COLUMN]
    cases = {"base": base, "site": site, "finance": finance, "both": both}
    cells = None
    for name, frame in cases.items():
        picked = frame[key + columns].rename(columns={column: f"{column}_{name}" for column in columns})
        cells = picked if cells is None else cells.merge(picked, on=key)
    table = cells[key].copy()
    table["base"] = cells["lcoa_eur_per_t_base"]
    table["construction"] = cells["lcoa_plant_eur_per_t_site"] - cells["lcoa_plant_eur_per_t_base"]
    table["water"] = cells["water_cost_eur_per_t_site"] - cells["water_cost_eur_per_t_base"]
    table["finance"] = cells["lcoa_eur_per_t_finance"] - cells["lcoa_eur_per_t_base"]
    table["interaction"] = cells["lcoa_eur_per_t_both"] - table[["base", "construction", "water", "finance"]].sum(axis=1)
    table["potential_mt_per_year"] = cells[[f"{CAPACITY_COLUMN}_{name}" for name in cases]].min(axis=1) / 1e6
    return table.merge(cell_regions(), on=key, how="left")


def _stack(ax, position: float, parts: Mapping[str, float], *, horizontal: bool, width: float = 0.6) -> float:
    """One stacked bar of parts (factor -> value) at position: positive parts stack away from zero
    in order, negative parts the other way; returns the total."""
    up = down = 0.0
    for factor, value in parts.items():
        start = up if value >= 0 else down
        style = {"color": FACTOR_COLOURS[factor], "edgecolor": "white", "linewidth": 1.0}
        if horizontal:
            ax.barh(position, value, left=start, height=width, **style)
        else:
            ax.bar(position, value, bottom=start, width=width, **style)
        up, down = (up + value, down) if value >= 0 else (up, down + value)
    return up + down


def _factor_key(ax, factors: Sequence[str], **extra) -> None:
    """A key of the factors drawn, the top of the stack first, and any extra handles."""
    from matplotlib.patches import Patch

    handles = [Patch(facecolor=FACTOR_COLOURS[factor], label=FACTOR_NAMES[factor]) for factor in reversed(factors)]
    ax.legend(handles=handles + list(extra.get("handles", [])), frameon=False, fontsize=9,
              loc=extra.get("loc", "upper left"), bbox_to_anchor=extra.get("bbox_to_anchor"))


def plot_build_factors(surface: pd.DataFrame, site_costs: pd.DataFrame, *, title: str,
                       minimum_mt_per_year: float = REFERENCE_OUTPUT_MT_PER_YEAR):
    """The remote factors of model/site_costs.py against the travel time to a city of 20,000
    people, on the installation (build) part of the capital cost and on its equipment part, alone
    (solid) and with the cold factor (dashed); the calibration point that sets both, Shamattawa
    (crosses, REMOTE_ANCHOR_DAYS); the location factors of the Australian cost reviews
    (inputs/site_costs/calibration_sites.csv: GHD's installation factors, Aurecon's installation
    and equipment factors), which the rule did not use; and, below, the plants of surface that can
    make minimum_mt_per_year by travel time (site_costs, the run's site-cost table), those under
    the cold factor darker. Travel times under 3 minutes are drawn at 3."""
    import matplotlib.pyplot as plt
    from matplotlib.legend_handler import HandlerTuple
    from matplotlib.patches import Patch

    from . import site_costs as rule

    key = ["latitude", "longitude"]
    plants = surface.loc[surface[CAPACITY_COLUMN] >= minimum_mt_per_year * 1e6, key].merge(
        site_costs[key + ["travel_time_to_city_min", "cold_multiplier"]], on=key, how="left")
    floor = 0.05  # hours
    plant_hours = (plants["travel_time_to_city_min"] / 60.0).clip(lower=floor)
    hours = np.logspace(np.log10(floor), np.log10(max(plant_hours.max(), 48.0) * 1.1), 400)
    sites = pd.read_csv(REPO_ROOT / "inputs" / "site_costs" / "calibration_sites.csv")
    ghd, aurecon = sites[sites["source"] == "ghd2025"], sites[sites["source"] == "aurecon2024"]
    calibration_hours = rule.REMOTE_ANCHOR_DAYS * 24.0
    remotest_site = sites["travel_min_city_20k"].max() / 60.0

    fig, (ax, low) = plt.subplots(2, 1, figsize=(10.0, 7.0), sharex=True, height_ratios=(3.0, 1.2), layout="constrained")
    crosses = []
    for at_calibration, colour, part in ((rule.REMOTE_BUILD_AT_ANCHOR, SERIES[1], "installation"),
                                         (rule.REMOTE_EQUIPMENT_AT_ANCHOR, SERIES[0], "equipment")):
        values = rule.remote_multiplier(hours * 60.0, at_calibration)
        ax.plot(hours, values, color=colour, linewidth=2, label=f"Factor on {part}")
        ax.plot(hours, values * rule.COLD_MULTIPLIER, color=colour, linewidth=2, linestyle="--",
                label=f"Factor on {part}, with the cold factor")
        crosses.append(ax.scatter([calibration_hours], [at_calibration], marker="x", s=36, color=colour, linewidths=1.5,
                                  zorder=5))
    ax.scatter(ghd["travel_min_city_20k"].clip(lower=floor * 60) / 60, ghd["installation_om_cost_factor"], marker="o", s=22,
               facecolors="none", edgecolors=SERIES[1], linewidths=1.0, label="GHD 2025: installation, by site")
    ax.scatter(aurecon["travel_min_city_20k"].clip(lower=floor * 60) / 60, aurecon["installation_cost_factor"], marker="s", s=20,
               facecolors="none", edgecolors=SERIES[1], linewidths=1.0, label="Aurecon 2024: installation, by site")
    ax.scatter(aurecon["travel_min_city_20k"].clip(lower=floor * 60) / 60, aurecon["equipment_cost_factor"], marker="s", s=20,
               facecolors="none", edgecolors=SERIES[0], linewidths=1.0, label="Aurecon 2024: equipment, by site")
    for axis in (ax, low):
        for x in (remotest_site, calibration_hours):
            axis.axvline(x, color=MUTED, linewidth=0.8, linestyle=":")
    top = ax.get_ylim()[1]
    ax.text(remotest_site, top, " remotest Australian site", rotation=90, va="top", ha="right", fontsize=8, color=MUTED)
    ax.text(calibration_hours, top, " calibration point", rotation=90, va="top", ha="right", fontsize=8, color=MUTED)
    ax.set_xscale("log")
    ax.set_ylabel("Factor on the capital cost (-)", color=INK)
    ax.set_title(title, loc="left", color=INK, fontsize=12)
    ax.grid(color=GRID, linewidth=0.5)
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles + [tuple(crosses)], labels + ["Calibration point: Shamattawa"],
              handler_map={tuple: HandlerTuple(ndivide=None)}, frameon=False, fontsize=8, loc="upper left", ncol=2)
    bins = np.logspace(np.log10(floor), np.log10(hours[-1]), 40)
    cold = plants["cold_multiplier"] > 1
    low.hist([plant_hours[~cold], plant_hours[cold]], bins=bins, stacked=True, color=["#b5b4ae", MUTED], edgecolor="white", linewidth=0.5)
    low.set_ylabel("Plants", color=INK)
    low.set_xlabel("Travel time to a city of 20,000 people (hours, logarithmic scale)", color=INK)
    low.grid(axis="y", color=GRID, linewidth=0.5)
    low.legend(handles=[Patch(facecolor="#b5b4ae", label="plants"), Patch(facecolor=MUTED, label="plants under the cold factor")],
               frameon=False, fontsize=8, loc="upper right")
    for axis in (ax, low):
        for side in ("top", "right"):
            axis.spines[side].set_visible(False)
    return fig


def plot_region_costs(factors: pd.DataFrame, applied: Sequence[str], *, title: str, by: str = "un_sub_region"):
    """Stacked bars of the mean LCOA by region (the column by) of one case, cheapest at the top: the
    base case's cost and what each of the case's factors (applied, in FACTOR_NAMES order) adds.
    A negative part, such as finance where the cost of capital is below the base case's, is drawn
    left of zero; a black tick and the number at the right give each region's mean, and each label
    gives the region's cells."""
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    parts = ["base", *applied]
    grouped = factors.dropna(subset=[by]).groupby(by)
    means = grouped[parts].mean()
    means["total"] = means[parts].sum(axis=1)
    means = means.sort_values("total", ascending=False)  # barh draws the first row at the bottom
    cells = grouped.size()
    fig, ax = plt.subplots(figsize=(9.0, 0.38 * len(means) + 1.6), layout="constrained")
    for position, (region, row) in enumerate(means.iterrows()):
        total = _stack(ax, position, {factor: float(row[factor]) for factor in parts}, horizontal=True)
        ax.plot([total, total], [position - 0.35, position + 0.35], color=INK, linewidth=1.5)
        right = sum(float(row[factor]) for factor in parts if row[factor] > 0)
        ax.annotate(f"{total:,.0f}", (right, position), xytext=(4, 0), textcoords="offset points", va="center",
                    fontsize=8, color=INK)
    ax.set_yticks(range(len(means)), [f"{region} ({cells[region]:,})" for region in means.index])
    ax.axvline(0.0, color=MUTED, linewidth=0.8)
    tick = Line2D([], [], color=INK, linewidth=1.5, label="Mean of the case")
    _factor_key(ax, parts, handles=[tick], loc="upper left", bbox_to_anchor=(1.01, 1.0))
    ax.set_xlabel("Mean levelised cost of ammonia (EUR/t)", color=INK)
    ax.set_title(title, loc="left", color=INK, fontsize=12)
    ax.grid(axis="x", color=GRID, linewidth=0.5)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    return fig


# The groups of the potential histograms, in stacking order: the land plants by UN region, then the sea plants
POTENTIAL_GROUPS = {"Africa": SERIES[0], "Americas": SERIES[1], "Asia": SERIES[2], "Europe": SERIES[3],
                    "Oceania": SERIES[4], "sea plants": FACTOR_COLOURS["base"]}


def potential_by_cost(land: pd.DataFrame, sea: pd.DataFrame, edges: Sequence[float], *,
                      minimum_mt_per_year: float = REFERENCE_OUTPUT_MT_PER_YEAR) -> pd.DataFrame:
    """The production potential (Gt/yr) of the plants that can make at least minimum_mt_per_year,
    by cost band (edges in EUR/t, each band including its lower edge, and a last band beyond the
    last edge) and group: the land plants by UN region (cell_regions), the sea plants as one group
    ("sea plants"). Bands as rows, groups as columns in POTENTIAL_GROUPS order."""
    key = ["latitude", "longitude"]
    regions = cell_regions()[key + ["un_region"]]
    land = land[land[CAPACITY_COLUMN] >= minimum_mt_per_year * 1e6].merge(regions, on=key, how="left")
    sea = sea[sea[CAPACITY_COLUMN] >= minimum_mt_per_year * 1e6].assign(un_region="sea plants")
    plants = pd.concat([land, sea])[["lcoa_eur_per_t", CAPACITY_COLUMN, "un_region"]]
    bands = pd.cut(plants["lcoa_eur_per_t"], [*edges, np.inf], right=False)
    table = plants.groupby([bands, "un_region"], observed=False)[CAPACITY_COLUMN].sum().unstack(fill_value=0.0) / 1e9
    return table[[group for group in POTENTIAL_GROUPS if group in table]]


def plot_potential_by_cost(tables: Mapping[str, pd.DataFrame], *, title: str):
    """Stacked histograms of production potential by cost band (potential_by_cost), one panel per
    case on shared axes, coloured by group (POTENTIAL_GROUPS); the box of each panel gives the
    potential beyond the last edge, which the bars leave out."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    from matplotlib.ticker import FuncFormatter

    fig, axes = plt.subplots(len(tables), 1, figsize=(10.0, 2.3 * len(tables) + 1.2), sharex=True, sharey=True,
                             squeeze=False, layout="constrained")
    drawn = set()
    for ax, (name, table) in zip(axes[:, 0], tables.items()):
        finite = table[[np.isfinite(band.right) for band in table.index]]
        lefts = [band.left for band in finite.index]
        widths = [band.right - band.left for band in finite.index]
        bottom = np.zeros(len(finite))
        for group, colour in POTENTIAL_GROUPS.items():
            if group in finite and finite[group].sum() > 0:
                ax.bar(lefts, finite[group], width=widths, bottom=bottom, align="edge", color=colour,
                       edgecolor="white", linewidth=0.4)
                bottom += finite[group].to_numpy()
                drawn.add(group)
        beyond = table.iloc[-1]
        ax.text(0.99, 0.94, f"Beyond {table.index[-1].left:,.0f} EUR/t: {beyond.drop('sea plants', errors='ignore').sum():,.1f} Gt/yr "
                f"on land, {beyond.get('sea plants', 0.0):,.1f} Gt/yr at sea", transform=ax.transAxes, ha="right", va="top",
                fontsize=8, color=INK, bbox={"boxstyle": "round,pad=0.4", "facecolor": "white", "edgecolor": GRID, "alpha": 0.9})
        ax.set_title(name, loc="left", color=INK, fontsize=10)
        ax.set_ylabel("Potential (Gt/yr)", color=INK)
        ax.grid(axis="y", color=GRID, linewidth=0.5)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    axes[-1, 0].set_xlabel("Levelised cost of ammonia (EUR/t)", color=INK)
    axes[-1, 0].xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:,.0f}"))
    handles = [Patch(facecolor=colour, label=group) for group, colour in POTENTIAL_GROUPS.items() if group in drawn]
    fig.legend(handles=handles, loc="outside right upper", frameon=False, fontsize=9)
    fig.suptitle(title, color=INK, fontsize=12)
    return fig


def potential_below(tables: Mapping[str, pd.DataFrame], cost: float) -> pd.DataFrame:
    """The potential (Gt/yr) of each group below a cost (EUR/t) in each case (tables of
    potential_by_cost; cost should be one of their edges), of the land plants below it, and of all
    the land plants and all the sea plants, every band included. Cases as columns."""
    columns = {}
    for name, table in tables.items():
        below = table[[band.right <= cost for band in table.index]].sum()
        land_below = below.drop("sea plants", errors="ignore").sum()
        below.index = [f"{group} below {cost:,g} EUR/t" for group in below.index]
        totals = pd.Series({f"land plants below {cost:,g} EUR/t": land_below,
                            "all land plants": table.drop(columns="sea plants", errors="ignore").to_numpy().sum(),
                            "all sea plants": table["sea plants"].sum() if "sea plants" in table else 0.0})
        columns[name] = pd.concat([below, totals])
    return pd.DataFrame(columns).round(1)


# --- the land rules and the production potential ------------------------------------------------

# The land-cover classes of the International Geosphere-Biosphere Programme (IGBP) in MODIS MCD12C1
IGBP_CLASSES = {0: "water", 1: "evergreen needleleaf forest", 2: "evergreen broadleaf forest",
                3: "deciduous needleleaf forest", 4: "deciduous broadleaf forest", 5: "mixed forest",
                6: "closed shrubland", 7: "open shrubland", 8: "woody savanna", 9: "savanna", 10: "grassland",
                11: "permanent wetland", 12: "cropland", 13: "urban and built-up land",
                14: "cropland and natural vegetation mosaic", 15: "snow and ice", 16: "barren land", 17: "unclassified"}


def _land_share(surface: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """A land run's share of the land that remains, and the multiplier a derived land-share variant
    (model/suppliers.py derive) applied to its parent's capacities: the variant keeps the parent's
    area columns, which the multiplier scales to its own share."""
    one = pd.Series(1.0, index=surface.index)
    multiplier = surface["land_share_multiplier_applied"] if "land_share_multiplier_applied" in surface else one
    share = (surface["land_competition_fraction_effective"] if "land_competition_fraction_effective" in surface
             else surface["land_competition_fraction"])
    return share, multiplier


def land_cover_shares() -> pd.DataFrame:
    """The share (%) of each IGBP land-cover class (MODIS MCD12C1) that the land rules open to wind
    and to PV (land.wind_class_availability and land.solar_class_availability in
    inputs/parameters.yaml). Water opens to wind at sea only, which the land runs leave out."""
    wind = parameter("land.wind_class_availability")
    solar = parameter("land.solar_class_availability")
    table = pd.DataFrame({"class": IGBP_CLASSES,
                          "wind (%)": {code: 100.0 * float(wind.get(code, 0.0)) for code in IGBP_CLASSES},
                          "PV (%)": {code: 100.0 * float(solar.get(code, 0.0)) for code in IGBP_CLASSES}})
    table["note"] = ""
    table.loc[0, "note"] = "wind at sea only (sea runs)"
    table.index.name = "IGBP code"
    return table


def _class_shares(availability: Mapping[int, float]) -> str:
    """The land-cover classes open to a technology, by share: "all of barren land; 50 % of ..."."""
    by_share: dict[float, list[str]] = {}
    for code, share in sorted(availability.items()):
        if code != 0 and share > 0:  # water is for wind at sea, which land runs leave out
            by_share.setdefault(float(share), []).append(IGBP_CLASSES[int(code)])
    parts = []
    for share, names in sorted(by_share.items(), reverse=True):
        listed = names[0] if len(names) == 1 else ", ".join(names[:-1]) + " and " + names[-1]
        parts.append(f"{'all' if share >= 1 else f'{100 * share:g} %'} of {listed}")
    return "; ".join(parts) + "; none of the other classes"


def land_rule_list(surface: pd.DataFrame, *, minimum_mt_per_year: float = REFERENCE_OUTPUT_MT_PER_YEAR) -> pd.DataFrame:
    """Everything that sets the land a plant of a land run may use and the capacity that land holds,
    from inputs/parameters.yaml (land) and the run's columns. One row per rule, with what it does."""
    slope_limit = parameter("land.slope_class_upper_deg")[-1]
    statuses = parameter("land.protected_area_statuses")
    wind_gw_km2 = float(parameter("land.wind_land_use_km2_per_gw"))
    solar = surface["solar_density_mw_per_km2"]
    share = 100 * float(_land_share(surface)[0].median())
    rules = {
        "land cover, wind": "the share of each land-cover class (IGBP classes of MODIS MCD12C1) open to wind: "
                            + _class_shares(parameter("land.wind_class_availability")),
        "land cover, PV": "the share of each class open to PV: " + _class_shares(parameter("land.solar_class_availability")),
        "protected areas": "land in protected areas of the World Database on Protected Areas with status "
                           + ", ".join(statuses) + " is left out",
        "slopes": f"land steeper than {slope_limit:g} degrees (GEBCO 2025) is left out",
        "land share": (f"the run takes {share:g} % of the land that remains" if share < 100
                       else "the run takes all the land that remains (no land share)"),
        "wind spacing": f"{wind_gw_km2:g} km² per GW of turbines ({1000 / wind_gw_km2:g} MW/km²)",
        "PV density": f"rows of modules that do not shade each other at noon on the winter solstice: {solar.median():.0f} MW/km² "
                      f"at the median cell ({solar.min():.0f} to {solar.max():.0f})",
        "wind and PV together": f"they share the land, and {100 * surface['wind_land_exclusive_fraction_effective'].median():g} % "
                                "of a wind farm's area is closed to PV",
        "cutoff": f"plants that cannot make {minimum_mt_per_year:g} Mt/yr are left out of the maps, supply curves and "
                  "the network model's suppliers",
    }
    table = pd.DataFrame({"what it does": rules})
    table.index.name = "rule"
    return table


def land_share_medians(surface: pd.DataFrame, *, by: str = "un_sub_region") -> pd.DataFrame:
    """The median share (%) of a cell's land that each land rule keeps, over the cells of each group
    (by: a column of cell_regions) and over all cells: the land-cover shares open to wind and to PV,
    the land outside protected areas, the land not too steep, the run's land share, and all the rules
    together for wind and for PV (the median of each cell's product, not the product of the medians).
    Groups from the most land open to wind under all the rules to the least, then all cells."""
    key = ["latitude", "longitude"]
    land = surface["onshore_area_km2"]
    share, multiplier = _land_share(surface)
    cover = 1.0 / surface["land_exclusion_factor"]
    shares = surface[key].copy()
    shares["land cover, wind"] = surface["wind_onshore_area_km2"] * cover / land
    shares["land cover, PV"] = surface["solar_available_area_km2"] * cover / land
    shares["outside protected areas"] = (1.0 - (surface["protected_area_pct"] / 100.0) / (surface["land_onshore_pct"] / 100.0)).clip(0.0, 1.0)
    shares["not too steep"] = (surface["slope_suitable_land_pct"] / 100.0).clip(0.0, 1.0)
    shares["land share"] = share
    shares["all rules, wind"] = surface["wind_onshore_area_km2"] * multiplier / land
    shares["all rules, PV"] = surface["solar_available_area_km2"] * multiplier / land
    columns = [column for column in shares if column not in key]
    shares = shares.merge(cell_regions()[key + [by]], on=key, how="left")
    grouped = shares.groupby(by)
    table = (100 * grouped[columns].median()).assign(cells=grouped.size()).sort_values("all rules, wind", ascending=False)
    table.loc["all cells"] = [*(100 * shares[columns].median()), len(shares)]
    table.index.name = by
    return table[["cells", *columns]].round(1).astype({"cells": int})

# The land rules of a land table (model/land_table.py), in the order potential_by_land_rule applies
# them, each on top of those before it
LAND_RULES = {
    "all land": "every km² of the cell's land open to wind and PV",
    "land cover": "the share of each land-cover class open to each technology (land.wind_class_availability, land.solar_class_availability)",
    "protected areas": "less protected areas",
    "slopes": "less land steeper than the last slope class",
    "land share": "the run's share of the land that remains (land_competition_fraction)",
    "cutoff": "only the plants that can make the reference output",
}


def potential_by_land_rule(surface: pd.DataFrame, *, minimum_mt_per_year: float = REFERENCE_OUTPUT_MT_PER_YEAR) -> pd.DataFrame:
    """The production potential (Mt/yr) of each plant of a land run as the land rules (LAND_RULES)
    apply one after another, from every km² of the cell's land open to wind and PV to the run's
    own potential (after "land share") and the plants that can make minimum_mt_per_year.

    A run applies land after the solve, so each plant's design scales to the land of each step
    without a new solve, as model/land.py scales it to the run's land: the potential is the design's
    output times the smallest ratio of available to used area of wind, of PV and of the two
    together. The areas before the exclusions and the land share are the run's areas over its
    land_exclusion_factor. Raises ValueError if the last step does not give the run's potential.
    Also the run's land share (land_share). Rows: plants; columns: latitude, longitude and one per rule.
    """
    eps = 1e-9
    wind_used = surface["wind_mw"] / surface["wind_density_mw_per_km2"]
    tracking = surface["solar_tracking_mw"] if "solar_tracking_mw" in surface else pd.Series(0.0, index=surface.index)
    solar_used = (surface["solar_mw"] / surface["solar_density_mw_per_km2"]
                  + (tracking.fillna(0.0) / surface["solar_tracking_density_mw_per_km2"]).fillna(0.0))
    union_used = solar_used + surface["wind_land_exclusive_fraction_effective"] * wind_used

    def potential(wind_km2: pd.Series, solar_km2: pd.Series, union_km2: pd.Series) -> pd.Series:
        ratios = pd.concat([(wind_km2 / wind_used).where(wind_used > eps), (solar_km2 / solar_used).where(solar_used > eps),
                            (union_km2 / union_used).where(union_used > eps)], axis=1).min(axis=1)
        return surface["annual_ammonia_production_t"] * ratios.clip(lower=0.0).fillna(0.0) / 1e6

    land = surface["onshore_area_km2"]
    share, multiplier = _land_share(surface)
    run_areas = tuple(surface[column] * multiplier
                      for column in ("wind_onshore_area_km2", "solar_available_area_km2", "renewable_union_area_km2"))
    cover = 1.0 / (surface["land_exclusion_factor"] * multiplier)
    protected = (1.0 - (surface["protected_area_pct"] / 100.0) / (surface["land_onshore_pct"] / 100.0)).clip(0.0, 1.0)
    slopes = (surface["slope_suitable_land_pct"] / 100.0).clip(0.0, 1.0)
    factors = {"land cover": cover, "protected areas": cover * protected, "slopes": cover * protected * slopes}
    table = surface[["latitude", "longitude"]].copy()
    table["all land"] = potential(land, land, land)
    for rule, factor in factors.items():
        table[rule] = potential(*(area * factor for area in run_areas))
    table["land share"] = potential(*run_areas)
    if not np.allclose(table["land share"], surface[CAPACITY_COLUMN] / 1e6, rtol=1e-9, atol=1e-6):
        raise ValueError("the land rules do not give the run's potential: is it a land run with land applied after the solve?")
    table["cutoff"] = table["land share"].where(table["land share"] >= minimum_mt_per_year, 0.0)
    table["land_share"] = share
    return table


def plot_land_rule_maps(potential: pd.DataFrame, *, title: str, limits: tuple[float, float] = (1.0, 20.0),
                        shares: Sequence[float] | None = None):
    """Maps of each cell's production potential (Mt/yr) as the land rules apply (potential_by_land_rule):
    all land; with the land-cover shares; with protected areas and slopes too; and at each land share
    in shares (by default the run's own, unless it is 100 %), which takes that share of the land
    that remains. One linear colour scale (limits): larger potentials take the darkest colour, cells
    below the lower limit are light grey, and each panel's box gives the potential of all the cells
    and the cells that reach the lower limit."""
    run_share = float(potential["land_share"].iloc[0])
    if shares is None:
        shares = [run_share] if run_share < 1.0 else []
    columns = {"All land": potential["all land"], "+ land cover": potential["land cover"],
               "+ protected areas and slopes": potential["slopes"]}
    for share in shares:
        columns[f"+ {100 * share:g} % share"] = potential["slopes"] * share  # every land constraint scales with the share
    panels = {name: potential.loc[values >= limits[0], ["latitude", "longitude"]].assign(potential_mt_per_year=values[values >= limits[0]])
              for name, values in columns.items()}
    return plot_maps(panels, "potential_mt_per_year", title=title, label="Production potential (Mt/yr)", limits=limits,
                     ramp=RAMPS["violet"], background=potential[["latitude", "longitude"]],
                     background_label=f"cells that cannot make {limits[0]:g} Mt/yr",
                     notes={name: f"{values.sum() / 1000:,.0f} Gt/yr in all\n{len(panels[name]):,} cells can make "
                                  f"{limits[0]:g} Mt/yr" for name, values in columns.items()})


def share_sensitivity(land: Mapping[str, pd.DataFrame], sea: Mapping[str, pd.DataFrame], shares: Sequence[float], *,
                      volumes: Sequence[float] = (500.0, 2000.0),
                      minimum_mt_per_year: float = REFERENCE_OUTPUT_MT_PER_YEAR) -> pd.DataFrame:
    """The land and the sea plants that can make minimum_mt_per_year, their potential (Gt/yr) and the
    cost of the last tonne at each volume (cost_at_supply, land and sea plants together) of each case
    when the plants take each share in shares of the land and sea that the rules leave, instead of
    their own. A run applies land after the solve, so a plant's potential scales with the share.
    Rows: shares (%); columns: (case, quantity). A cost is missing where the plants cannot make the
    volume."""
    minimum = minimum_mt_per_year * 1e6
    counts = {site: f"{site} plants of {minimum_mt_per_year:g} Mt/yr" for site in ("land", "sea")}
    rows = {}
    for share in shares:
        row = {}
        for case, frame in land.items():
            scaled = {site: plants.assign(**{CAPACITY_COLUMN: plants[CAPACITY_COLUMN] * share / _land_share(plants)[0]})
                      for site, plants in (("land", frame), ("sea", sea[case]))}
            for site, plants in scaled.items():
                row[(case, counts[site])] = int((plants[CAPACITY_COLUMN] >= minimum).sum())
            row[(case, "their potential (Gt/yr)")] = sum(
                plants.loc[plants[CAPACITY_COLUMN] >= minimum, CAPACITY_COLUMN].sum() for plants in scaled.values()) / 1e9
            for volume in volumes:
                row[(case, f"cost at {volume:,g} Mt/yr (EUR/t)")] = cost_at_supply(
                    pd.concat(scaled.values()), volume, minimum_mt_per_year=minimum_mt_per_year)
        rows[f"{100 * share:g} %"] = row
    table = pd.DataFrame(rows).T
    table.columns = pd.MultiIndex.from_tuples(table.columns)
    table.index.name = "share"
    for column in table.columns:
        values = table[column].astype(float)
        if column[1] in counts.values():
            table[column] = values.astype(int)
        elif column[1].startswith("cost"):
            table[column] = values.round(0).astype("Int64")
        else:
            table[column] = values.round(1)
    return table


def plants_by_share(surface: pd.DataFrame, shares: Sequence[float], *, by: str = "un_sub_region",
                    minimum_mt_per_year: float = REFERENCE_OUTPUT_MT_PER_YEAR) -> pd.DataFrame:
    """The land plants that can make minimum_mt_per_year at each land share in shares (fractions of
    the land that the rules leave), by region (by: a column of cell_regions) and over all cells, and
    the percentage of the first share's plants that the last share keeps (kept (%)). A run applies
    land after the solve, so a plant's potential scales with the share. Rows: regions from the
    largest kept percentage to the smallest, then all cells."""
    key = ["latitude", "longitude"]
    potential = surface[CAPACITY_COLUMN] / _land_share(surface)[0]  # on all the land that the rules leave
    columns = [f"{100 * share:g} %" for share in shares]
    table = surface[key].assign(**{column: potential * share >= minimum_mt_per_year * 1e6
                                   for column, share in zip(columns, shares)})
    table = table.merge(cell_regions()[key + [by]], on=key, how="left")
    counts = table.groupby(by)[columns].sum()
    counts = counts[counts[columns[0]] > 0]
    counts.loc["all cells"] = table[columns].sum()
    counts = counts.astype(int)
    counts["kept (%)"] = (100.0 * counts[columns[-1]] / counts[columns[0]]).round(0).astype(int)
    order = counts.drop(index="all cells").sort_values("kept (%)", ascending=False).index.tolist()
    counts.index.name = by
    return counts.loc[[*order, "all cells"]]


def _latitude(value: float) -> str:
    return "0°" if value == 0 else f"{abs(value):g}° {'S' if value < 0 else 'N'}"


def land_by_latitude(surface: pd.DataFrame, potential: pd.DataFrame, *,
                     edges: Sequence[float] = (-90.0, -23.5, 23.5, 60.0, 90.0)) -> pd.DataFrame:
    """Medians, by band of latitude (edges in degrees, south to north), of what sets a land plant's
    production potential on all of its cell's land ("all land" of potential_by_land_rule): the
    cell's land (km²), the land the plant needs per Mt/yr there (km²), the PV density (MW/km²),
    wind's share of the generating capacity (%) and the potential (Mt/yr). Rows: bands, south to
    north; a cell at an edge belongs to the band south of it."""
    key = ["latitude", "longitude"]
    cells = surface.merge(potential[key + ["all land"]], on=key)
    tracking = cells["solar_tracking_mw"].fillna(0.0) if "solar_tracking_mw" in cells else 0.0
    labels = [f"{_latitude(south)} to {_latitude(north)}" for south, north in zip(edges[:-1], edges[1:])]
    cells = cells.assign(band=pd.cut(cells["latitude"], list(edges), labels=labels),
                         land_per_mt=(cells["onshore_area_km2"] / cells["all land"]).replace(np.inf, np.nan),
                         wind_pct=100.0 * cells["wind_mw"] / (cells["wind_mw"] + cells["solar_mw"] + tracking))
    grouped = cells.groupby("band", observed=True)
    whole = {"land (km²)": "onshore_area_km2", "land per Mt/yr (km²)": "land_per_mt",
             "PV density (MW/km²)": "solar_density_mw_per_km2", "wind share of capacity (%)": "wind_pct"}
    table = pd.DataFrame({"cells": grouped.size(),
                          **{name: grouped[column].median().round(0).astype("Int64") for name, column in whole.items()},
                          "potential on all land (Mt/yr)": grouped["all land"].median().round(1)})
    table.index.name = "latitude"
    return table
