"""Fetch MAP/Oxford travel-time-to-city data from Google Earth Engine.

For each onshore 1° grid cell, computes the mean travel time (minutes)
to the nearest city with population ≥ 50k.

Source: MAP/Oxford Accessibility to Cities 2015 v1.0
  https://developers.google.com/earth-engine/datasets/catalog/Oxford_MAP_accessibility_to_cities_2015_v1_0

Output: data/travel_time_by_cell.csv
  Columns: latitude, longitude, travel_time_minutes
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import ee
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
GEE_PROJECT = "weather-461309"
ASSET_ID = "Oxford/MAP/accessibility_to_cities_2015_v1_0"
BATCH_SIZE = 500  # features per GEE request
CELL_SIZE_DEG = 1.0  # grid resolution

REPO_ROOT = Path(__file__).resolve().parent.parent
MAX_CAPACITIES_CSV = REPO_ROOT / "data" / "max_capacities.csv"
OUTPUT_CSV = REPO_ROOT / "data" / "travel_time_by_cell.csv"


def make_cell_polygon(lat: float, lon: float, size: float = CELL_SIZE_DEG) -> ee.Geometry:
    """Create a 1° square polygon centred on (lat, lon)."""
    half = size / 2
    return ee.Geometry.Rectangle([lon - half, lat - half, lon + half, lat + half])


def fetch_batch(image: ee.Image, cells: pd.DataFrame) -> list[dict]:
    """Query GEE for mean travel time over a batch of cell polygons."""
    features = []
    for _, row in cells.iterrows():
        poly = make_cell_polygon(row["latitude"], row["longitude"])
        feat = ee.Feature(poly, {"lat": row["latitude"], "lon": row["longitude"]})
        features.append(feat)

    fc = ee.FeatureCollection(features)
    reduced = image.reduceRegions(
        collection=fc,
        reducer=ee.Reducer.mean(),
        scale=1000,  # 1 km — matches raster resolution
    )
    results = reduced.getInfo()
    return [
        {
            "latitude": f["properties"]["lat"],
            "longitude": f["properties"]["lon"],
            "travel_time_minutes": f["properties"].get("mean"),
        }
        for f in results["features"]
    ]


def main():
    print("Initialising GEE...")
    ee.Initialize(project=GEE_PROJECT)

    import warnings
    warnings.filterwarnings("ignore", category=DeprecationWarning)

    image = ee.Image(ASSET_ID).select("accessibility")

    # Load grid and filter to onshore cells
    mc = pd.read_csv(MAX_CAPACITIES_CSV)
    onshore = mc[mc["onshore_land_pct"] > 0][["latitude", "longitude"]].reset_index(drop=True)
    print(f"Onshore cells to query: {len(onshore):,}")

    # Process in batches
    all_results = []
    n_batches = int(np.ceil(len(onshore) / BATCH_SIZE))

    for i in range(n_batches):
        start = i * BATCH_SIZE
        end = min(start + BATCH_SIZE, len(onshore))
        batch = onshore.iloc[start:end]

        t0 = time.time()
        try:
            results = fetch_batch(image, batch)
            all_results.extend(results)
            elapsed = time.time() - t0
            print(f"  Batch {i+1}/{n_batches} ({start}–{end}): {len(results)} results in {elapsed:.1f}s")
        except Exception as e:
            print(f"  Batch {i+1}/{n_batches} FAILED: {e}")
            # Retry with smaller sub-batches
            sub_size = BATCH_SIZE // 5
            for j in range(0, len(batch), sub_size):
                sub = batch.iloc[j : j + sub_size]
                try:
                    results = fetch_batch(image, sub)
                    all_results.extend(results)
                    print(f"    Sub-batch {j}–{j+len(sub)}: OK")
                except Exception as e2:
                    print(f"    Sub-batch {j}–{j+len(sub)} FAILED: {e2}")

        # Save intermediate results every 10 batches
        if (i + 1) % 10 == 0 or i == n_batches - 1:
            df = pd.DataFrame(all_results)
            df.to_csv(OUTPUT_CSV, index=False)
            valid = df["travel_time_minutes"].notna().sum()
            print(f"  -> Saved {len(df):,} rows ({valid:,} valid) to {OUTPUT_CSV.name}")

    # Final summary
    df = pd.DataFrame(all_results)
    df.to_csv(OUTPUT_CSV, index=False)
    valid = df["travel_time_minutes"].notna().sum()
    null = df["travel_time_minutes"].isna().sum()
    print(f"\nDone. {len(df):,} cells ({valid:,} with data, {null:,} null)")
    if valid > 0:
        vals = df["travel_time_minutes"].dropna()
        print(f"Travel time (min): min={vals.min():.0f}, median={vals.median():.0f}, "
              f"mean={vals.mean():.0f}, max={vals.max():.0f}")
    print(f"Saved to: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
