"""Visualise global LCOA outputs on a choropleth grid."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict

import pandas as pd
import plotly.express as px


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


def plot_lcoa_heatmap(
    csv_path: str | Path,
    color_column: str = "lcoa_usd_per_t",
    cell_size_deg: float = 1.0,
    color_scale: str = "Cividis",
):
    csv_path = Path(csv_path)
    df = pd.read_csv(csv_path)
    df.columns = [col.lower() for col in df.columns]
    color_column = color_column.lower()
    lat_col = "latitude" if "latitude" in df.columns else None
    lon_col = "longitude" if "longitude" in df.columns else None
    required = {color_column}
    if lat_col:
        required.add(lat_col)
    if lon_col:
        required.add(lon_col)
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV {csv_path} is missing required columns: {sorted(missing)}")

    plot_df = df.dropna(subset=[color_column]).copy()
    if plot_df.empty:
        raise ValueError("No rows with colour metric available.")

    plot_df["cell_id"] = [_cell_id(lat, lon) for lat, lon in zip(plot_df[lat_col], plot_df[lon_col])]
    geojson = _build_cell_geojson(plot_df, lat_col=lat_col, lon_col=lon_col, cell_size_deg=cell_size_deg)

    wind_col = "wind_mw" if "wind_mw" in plot_df.columns else "wind"
    solar_col = "solar_mw" if "solar_mw" in plot_df.columns else "solar"
    land_onshore_col = (
        "land_onshore_pct"
        if "land_onshore_pct" in plot_df.columns
        else "percent_onshore_pct"
        if "percent_onshore_pct" in plot_df.columns
        else "percent_onshore"
    )
    hover_candidates = [
        "country",
        "lcoa_usd_per_t",
        "annual_ammonia_demand_mwh",
        wind_col,
        solar_col,
        "cost_share_wind_pct",
        "cost_share_solar_pct",
        "cost_share_electrolyser_pct",
        land_onshore_col,
    ]
    hover_data = {col: True for col in hover_candidates if col in plot_df.columns}

    fig = px.choropleth_map(
        plot_df,
        geojson=geojson,
        locations="cell_id",
        featureidkey="id",
        color=color_column,
        map_style="carto-positron",
        color_continuous_scale=color_scale,
        opacity=0.9,
        hover_data=hover_data or None,
        center=dict(lat=0.0, lon=0.0),
        zoom=0.5,
    )
    fig.update_traces(marker_line_width=0.2, marker_line_color="rgba(255, 255, 255, 0.3)")
    fig.update_layout(margin=dict(l=0, r=0, t=0, b=0))
    return fig


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render a global LCOA heatmap from run_global outputs.")
    parser.add_argument("csv", type=str, help="Path to the aggregated global results CSV.")
    parser.add_argument("--output", type=str, default=None, help="Optional HTML file for saving the figure.")
    parser.add_argument(
        "--color-column",
        type=str,
        default="lcoa_usd_per_t",
        help="Results column to visualise (default: lcoa_usd_per_t).",
    )
    parser.add_argument(
        "--color-scale",
        type=str,
        default="Cividis",
        help="Plotly colour scale name (e.g., Viridis, Inferno, Cividis).",
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
