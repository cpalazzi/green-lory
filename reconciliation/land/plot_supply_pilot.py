#!/usr/bin/env python3
"""Render the accepted three-site scientific cost–quantity comparison."""
import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import pandas as pd


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--fixed-comparison", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    summary = json.loads((args.audit / "summary.json").read_text())
    if not summary["qa_pass"]:
        raise ValueError("Supply analysis has not passed QA")
    for name, digest in summary.get("outputs_sha256", {}).items():
        if hashlib.sha256((args.audit/name).read_bytes()).hexdigest() != digest:
            raise ValueError(f"Changed supply analysis: {name}")
    points = pd.read_csv(args.audit / "validated_points.csv")
    sites = pd.read_csv(args.audit / "site_comparison.csv")
    fixed = pd.read_csv(args.fixed_comparison)
    fixed = fixed[fixed.tracking_footprint_ratio == 2].set_index(["latitude", "longitude"])
    labels = {"atacama": "Atacama", "northwest_australia": "Northwest Australia",
              "central_australia": "Central Australia"}
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 3, figsize=(12.6, 4.7))
    for ax, (_, site) in zip(axes, sites.iterrows(), strict=True):
        selected = points[points.site == site.site].sort_values("target_Mtpa")
        feasible = selected[selected.status == "feasible"]
        infeasible = selected[selected.status.isin(["infeasible", "infeasible_solver"])]
        analytic = selected[selected.status == "infeasible_analytic"]
        previous = fixed.loc[(site.latitude, site.longitude)]
        ax.plot(feasible.target_Mtpa, feasible.LCOA_EUR2020_per_t, "o-", color="#176b9b", lw=1.6, ms=4)
        ax.axhline(site.control_LCOA_EUR2020_per_t, color="#5a5a5a", lw=1, ls="--")
        ax.scatter([previous.fixed_common_land_Mtpa], [previous.fixed_LCOA_EUR2020_per_t],
                   marker="*", s=95, color="#a36520", zorder=5)
        ax.axvline(1., color="#999999", lw=.8, ls=":")
        ax.axvline(site.ideal_energy_bound_Mtpa, color="#a22c36", lw=1, ls="--")
        ax.set_xlim(0, site.ideal_energy_bound_Mtpa * 1.09)
        values = [*feasible.LCOA_EUR2020_per_t, previous.fixed_LCOA_EUR2020_per_t,
                  site.control_LCOA_EUR2020_per_t]
        margin = max(5, (max(values) - min(values)) * .12)
        ax.set_ylim(min(values) - margin, max(values) + margin)
        # Infeasibility has no price. Marks sit on the x axis, not at a fabricated LCOA.
        visible = infeasible[infeasible.target_Mtpa <= ax.get_xlim()[1]]
        ax.scatter(visible.target_Mtpa, [0] * len(visible), marker="x", color="#a22c36",
                   transform=ax.get_xaxis_transform(), clip_on=False, zorder=6)
        visible_analytic = analytic[analytic.target_Mtpa <= ax.get_xlim()[1]]
        ax.scatter(visible_analytic.target_Mtpa, [0] * len(visible_analytic), marker="^",
                   facecolors="none", edgecolors="#a22c36", transform=ax.get_xaxis_transform(),
                   clip_on=False, zorder=6)
        ax.set_title(labels[site.site], fontsize=12, loc="left", pad=10)
        ax.set_xlabel("Ammonia quantity (Mt/year)")
        ax.set_ylabel("LCOA (EUR 2020/t)")
        ax.grid(axis="y", alpha=.18)
        ax.text(.01, -.28, f"Archived: {site.historical_Mtpa:.2f} Mt/year (outside axis)\n"
                f"Ideal energy bound: {site.ideal_energy_bound_Mtpa:.2f} Mt/year",
                transform=ax.transAxes, fontsize=9, va="top")
    handles = [Line2D([], [], marker="o", color="#176b9b", label="Land-constrained, both PV types"),
               Line2D([], [], ls="--", color="#5a5a5a", label="Unconstrained minimum LCOA"),
               Line2D([], [], marker="*", color="#a36520", ls="None", markersize=10, label="Scaled fixed-only design"),
               Line2D([], [], marker="x", color="#a22c36", ls="None", label="Solver-proven infeasible"),
               Line2D([], [], marker="^", color="#a22c36", markerfacecolor="none", ls="None", label="Analytically infeasible")]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(.5, .97), ncol=2, frameon=False, fontsize=9)
    fig.subplots_adjust(left=.07, right=.985, bottom=.28, top=.73, wspace=.32)
    fig.text(.5, .025, "Same pilot land and costs. Tracking footprint = 2 × fixed PV. Red dashed lines are optimistic energy bounds; dotted lines mark 1 Mt/year.\n"
             "Lines join tested points only; neither interpolation nor an exact maximum capacity has been established.",
             ha="center", fontsize=8.5, color="#444444")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=180, facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    main()
