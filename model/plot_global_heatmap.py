"""Visualise global LCOA outputs on a choropleth grid."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict

import json

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# Repo root is the directory containing the 'model/' package.
REPO_ROOT = Path(__file__).resolve().parent.parent


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


def plot_lcoa_heatmap(
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
    """Render a global LCOA choropleth heatmap.

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
            "cost_share_wind_pct",
        )
        if col in plot_df.columns
    ]
    if wind_cost_cols:
        plot_df["cost_share_wind_total_pct"] = plot_df[wind_cost_cols].fillna(0.0).sum(axis=1)
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
        "cost_share_wind_total_pct",
        "cost_share_solar_pct",
        "cost_share_electrolyser_pct",
        land_onshore_col,
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
        geojson_path = REPO_ROOT / "data" / "countries.geojson"
        if geojson_path.exists():
            border_trace = _country_border_trace(
                geojson_path,
                line_color=country_border_color,
                line_width=country_border_width,
            )
            fig.add_trace(border_trace)

    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Matplotlib (publication) heatmap
# ─────────────────────────────────────────────────────────────────────────────

def plot_lcoa_heatmap_mpl(
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
    """Render a publication-quality LCOA heatmap using matplotlib.

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
    geojson_path = REPO_ROOT / "data" / "countries.geojson"
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

    fig.tight_layout()
    return fig, ax


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render a global LCOA heatmap from run_global outputs.")
    parser.add_argument("csv", type=str, help="Path to the aggregated global results CSV.")
    parser.add_argument("--output", type=str, default=None, help="Optional HTML file for saving the figure.")
    parser.add_argument(
        "--color-column",
        type=str,
        default="auto",
        help="Results column to visualise (default: auto).",
    )
    parser.add_argument(
        "--color-scale",
        type=str,
        default="Viridis_r",
        help="Plotly colour scale name (e.g., Viridis, Viridis_r, Inferno, Cividis).",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    figure = plot_lcoa_heatmap(args.csv, color_column=args.color_column, color_scale=args.color_scale)
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        figure.write_html(output_path)
    else:
        figure.show()


if __name__ == "__main__":
    main()
