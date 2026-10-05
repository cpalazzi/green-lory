#!/usr/bin/env python3
"""Static scientific map of supplier eligibility, not an optimised network."""
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import numpy as np
import pandas as pd


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--comparison", type=Path, required=True)
    p.add_argument("--countries", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    cells = pd.read_csv(args.comparison / "cell_comparison.csv")
    omitted = pd.read_csv(args.comparison / "historical_eligible_cells_excluded_by_land.csv")
    boundaries = json.loads(args.countries.read_text())
    segments = []
    for feature in boundaries["features"]:
        geometry = feature["geometry"]
        polygons = geometry["coordinates"] if geometry["type"] == "MultiPolygon" else [geometry["coordinates"]]
        for polygon in polygons:
            for ring in polygon:
                xy = np.asarray(ring)
                # Avoid a spurious line across the antimeridian.
                for section in np.split(xy, np.where(np.abs(np.diff(xy[:, 0])) > 180)[0] + 1):
                    if len(section) > 1:
                        segments.append(section[:, :2])
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})
    fig, axes = plt.subplots(3, 2, figsize=(13, 9), gridspec_kw={"width_ratios": [3.5, 1]}, layout="constrained")
    labels = {"old": "Archived supplier input", "rep": "Lory historical-assumption emulation", "central": "Lory hourly candidate"}
    for row, label in enumerate(["old", "rep", "central"]):
        d = cells[cells[f"capacity_{label}_Mtpa"] >= 1].copy()
        if label == "old":
            d = pd.concat([d, omitted], ignore_index=True)
        for col, ax in enumerate(axes[row]):
            ax.set_facecolor("#f5f8fa")
            ax.add_collection(LineCollection(segments, colors="#adb6bd", linewidths=.4, zorder=1))
            dots = ax.scatter(d.longitude, d.latitude, c=d[f"cost_{label}_USD2018_per_t"],
                              s=5 if col == 0 else 16, marker="s", cmap="viridis", vmin=200, vmax=330,
                              linewidths=0, zorder=2, rasterized=True)
            ax.set_xlim((-180, 180) if col == 0 else (110, 155))
            ax.set_ylim((-58, 80) if col == 0 else (-45, -10))
            ax.tick_params(labelsize=8, length=2, color="#8c989f")
            for spine in ax.spines.values():
                spine.set_color("#c6cdd2")
            ax.set_title(f"{labels[label]} · {len(d):,} eligible cells" if col == 0 else "Australia", loc="left", fontsize=11)
    fig.suptitle("Land limits change the supplier pool before network optimisation", fontsize=15)
    fig.supxlabel("Eligibility: annual land-limited capacity ≥ 1 Mt NH₃. Before cheapest-4,000 selection. No network flows shown.", fontsize=10)
    bar = fig.colorbar(dots, ax=axes, orientation="horizontal", fraction=.04, pad=.03, aspect=50, extend="both")
    bar.set_label("Plant LCOA (USD2018-equivalent per tonne NH₃)")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=180)
    plt.close(fig)
    print(args.output.resolve())


if __name__ == "__main__":
    main()
