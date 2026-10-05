#!/usr/bin/env python
"""Compare a legacy sample run with the archived shipping input.

For every solved cell: LCOA ratio (rerun / archived), the design (MW per Mt/yr),
and the PV density implied by the archived capacity under the paper's
complete-overlap rule, Q = A_solar * d / P_pv  ->  d_implied = Q * P_pv / A_solar.
Wind-land feasibility of the archived capacity is checked separately:
Q_wind_limit = A_wind / (0.2 km2/MW * P_wind).

Inputs: the run directory (results.csv + per-cell summary.json), the archived
supplier table, the unmasked land audit (2 % suitable areas, centered cells) and
optionally the corrected pilot land tables. Output: per-cell CSV, summary JSON,
and two figures (implied density vs latitude; LCOA ratio vs latitude).
"""
from __future__ import annotations

import argparse
import glob
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

WIND_KM2_PER_MW = 0.2


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--archived", type=Path, required=True, help="c_NH3_cost_4.5.csv")
    p.add_argument("--unmasked", type=Path, required=True, help="historical_sites_unmasked.csv")
    p.add_argument("--cf", type=Path, required=True, help="annual_cf_2019.csv")
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite {args.output}")
    args.output.mkdir(parents=True)

    rows = []
    for d in sorted(glob.glob(str(args.run / "*__*__*"))):
        f = os.path.join(d, "summary.json")
        if not os.path.exists(f):
            continue
        s = json.load(open(f))
        c = s["capacities_mw"]
        rows.append({"latitude": s["lat"], "longitude": s["lon"], "cell": s["cell"], "variant": s["variant"],
                     "lcoa_rerun": s["lcoa_usd_per_t"], "P_wind": c.get("Wind", 0.0), "P_fixed": c.get("Solar", 0.0),
                     "P_tracking": c.get("SolarTracking", 0.0), "P_electrolysis": c.get("Electrolysis", 0.0),
                     "elec_frac_rerun": s["electricity_capex_fraction_of_objective"],
                     "curtailed": s["curtailed_fraction"], "mwh_per_t": s["primary_electricity_mwh_per_t_snapshot_basis"]})
    df = pd.DataFrame(rows)
    arch = pd.read_csv(args.archived).rename(columns={"Latitude": "latitude", "Longitude": "longitude"})
    arch = arch[["latitude", "longitude", "LCOA", "Max_capacity", "Electricity_Cost_Frac", "country"]]
    un = pd.read_csv(args.unmasked)
    cf = pd.read_csv(args.cf)
    df = df.merge(arch, on=["latitude", "longitude"], how="left").merge(un, on=["latitude", "longitude"], how="left",
                                                                           suffixes=("", "_un")).merge(cf, on=["latitude", "longitude"], how="left")
    df["absl"] = df.latitude.abs()
    df["P_pv"] = df.P_fixed + df.P_tracking
    df["lcoa_ratio"] = df.lcoa_rerun / df.LCOA
    As, Aw = df.center_solar_shipping_km2, df.center_wind_shipping_km2
    df["d_implied_MW_km2"] = df.Max_capacity * df.P_pv / As
    df["km2_per_GW_implied"] = 1000.0 / df.d_implied_MW_km2
    df["Q_wind_limit_Mtpa"] = np.where(df.P_wind > 0, Aw / (WIND_KM2_PER_MW * df.P_wind), np.inf)
    df["archived_exceeds_wind_limit"] = df.Max_capacity > df.Q_wind_limit_Mtpa * 1.001
    df["wind_energy_share"] = (df.P_wind * df["wind_net_0.93"]) / (df.P_wind * df["wind_net_0.93"] + df.P_fixed * df.fixed_pv + df.P_tracking * df.tracking_pv)
    df["paper_d_lat"] = df["paper_fixed_density_MW_km2"]
    # energy-ceiling style check with the replicated design intensity, for reference
    df["Q_energy_111"] = (As * (1000 / 9) * df.tracking_pv * 8760) / (df.mwh_per_t * 1e6)
    df.to_csv(args.output / "per_cell.csv", index=False)

    ok = df.dropna(subset=["LCOA", "Max_capacity"])
    pv = ok[(ok.wind_energy_share < 0.05) & (As.loc[ok.index] > 0.5) & (ok.Max_capacity > 0)]
    summary = {
        "n_solved": int(len(df)), "n_matched": int(len(ok)), "n_pv_dominated": int(len(pv)),
        "lcoa_ratio_quantiles": ok.lcoa_ratio.quantile([.05, .25, .5, .75, .95]).round(4).to_dict(),
        "lcoa_ratio_by_lat_band": {str(k): float(v) for k, v in ok.groupby(pd.cut(ok.absl, [0, 10, 20, 30, 40, 50, 60, 70]), observed=True).lcoa_ratio.median().round(4).items()},
        "d_implied_pv_dominated_by_lat_band": {str(k): {kk: float(vv) for kk, vv in v.items()} for k, v in pv.groupby(pd.cut(pv.absl, [0, 10, 20, 30, 40, 50, 60, 70]), observed=True).d_implied_MW_km2.describe()[["count", "25%", "50%", "75%"]].round(1).iterrows()},
        "share_archived_exceeding_wind_limit_when_wind_used": float(ok[ok.P_wind > 1].archived_exceeds_wind_limit.mean()) if (ok.P_wind > 1).any() else None,
        "n_wind_used": int((ok.P_wind > 1).sum()),
    }
    with open(args.output / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(json.dumps(summary, indent=2, default=str))

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
        sc = ax[0].scatter(pv.absl, pv.d_implied_MW_km2, c=pv.tracking_pv, s=10, cmap="viridis")
        ax[0].axhline(1000 / 9, color="grey", ls="--", lw=0.8, label="9 km²/GW (111 MW/km²)")
        ax[0].set_xlabel("|latitude| (deg)"); ax[0].set_ylabel("implied PV density (MW/km²)")
        ax[0].set_title("PV-dominated cells: Q_arch · P_pv / A_solar"); ax[0].set_ylim(0, 400); ax[0].legend()
        plt.colorbar(sc, ax=ax[0], label="tracking CF")
        ax[1].scatter(ok.absl, ok.lcoa_ratio, s=10, c=ok.wind_energy_share, cmap="coolwarm")
        ax[1].axhline(1.0, color="grey", ls="--", lw=0.8)
        ax[1].set_xlabel("|latitude| (deg)"); ax[1].set_ylabel("LCOA rerun / archived"); ax[1].set_ylim(0.6, 1.6)
        ax[1].set_title("LCOA replication ratio (colour: wind energy share)")
        fig.tight_layout(); fig.savefig(args.output / "sample_analysis.png", dpi=130)
    except Exception as exc:  # pragma: no cover
        print("plot skipped:", exc)


if __name__ == "__main__":
    main()
