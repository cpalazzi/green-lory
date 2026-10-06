#!/usr/bin/env python
"""Test candidate formulas for the archived `Max_capacity` column against all
archived supplier coordinates.

Each hypothesis maps per-cell inputs (2 % shipping-share suitable areas for
solar and wind, annual capacity factors, PV density) to a capacity in Mt/yr and
is scored by the distribution of reconstructed/archived ratios. No fitting is
done; the formulas are fixed a priori and reported side by side.

Inputs
------
--cf        annual_cf_2019.csv from compute_annual_cf.py (fixed_pv, tracking_pv,
            wind_raw, wind_net_0.93 per cell)
--unmasked  historical_sites_unmasked.csv (September audit: Table 2 class
            fractions and 2 % share only, centered and southwest anchors, no
            protected/slope exclusions; includes archived capacity)
--masked    optional: the September global land table (paper_2pct_slope15.csv:
            with protected/slope exclusions, known 1-degree latitude
            misalignment in its MODIS reader) for a with-exclusions variant
--output    new directory
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

HOURS = 8760.0
NH3_HHV = 6.25       # MWh per t, legacy model convention
DESIGN_ELEC = 9.0    # MWh electricity per t (approximate legacy design intensity, for a contrast case)
WIND_D = 5.0         # MW/km2 (200 km2/GW)
PV_D_9 = 1000.0 / 9  # MW/km2 (9 km2/GW, no latitude adjustment)


def score(ratio: pd.Series) -> dict:
    r = ratio.replace([np.inf, -np.inf], np.nan).dropna()
    r = r[r > 0]
    lr = np.log(r)
    return {
        "n": int(len(r)),
        "median": float(r.median()),
        "p10": float(r.quantile(0.10)), "p25": float(r.quantile(0.25)),
        "p75": float(r.quantile(0.75)), "p90": float(r.quantile(0.90)),
        "within_5pct": float((abs(r - 1) <= 0.05).mean()),
        "within_10pct": float((abs(r - 1) <= 0.10).mean()),
        "within_25pct": float((abs(r - 1) <= 0.25).mean()),
        "log_rmse": float(np.sqrt((lr ** 2).mean())),
        "log_mad": float((lr - lr.median()).abs().median()),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cf", type=Path, required=True)
    p.add_argument("--unmasked", type=Path, required=True)
    p.add_argument("--masked", type=Path, default=None)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite {args.output}")
    args.output.mkdir(parents=True)

    cf = pd.read_csv(args.cf)
    un = pd.read_csv(args.unmasked)
    df = un.merge(cf, on=["latitude", "longitude"], how="left", validate="one_to_one")
    df = df[df["historical_capacity_Mtpa"] > 0].copy()
    missing_cf = int(df["fixed_pv"].isna().sum())
    df = df.dropna(subset=["fixed_pv", "wind_raw"])
    if args.masked is not None:
        m = pd.read_csv(args.masked, usecols=["latitude", "longitude", "solar_area_km2", "wind_onshore_area_km2",
                                              "solar_density_mw_per_km2"])
        m = m.rename(columns={"solar_area_km2": "masked_solar_km2", "wind_onshore_area_km2": "masked_wind_km2",
                              "solar_density_mw_per_km2": "masked_pv_density"})
        df = df.merge(m, on=["latitude", "longitude"], how="left")

    hyps = {}
    for anchor in ("center", "southwest"):
        As, Aw = df[f"{anchor}_solar_shipping_km2"], df[f"{anchor}_wind_shipping_km2"]
        dlat = df["paper_fixed_density_MW_km2"]
        e_pv9 = As * PV_D_9 * df["fixed_pv"] * HOURS
        e_pvlat = As * dlat * df["fixed_pv"] * HOURS
        e_trk_lat = As * dlat * df["tracking_pv"] * HOURS
        e_trk_9 = As * PV_D_9 * df["tracking_pv"] * HOURS
        e_w_net = Aw * WIND_D * df["wind_net_0.93"] * HOURS
        e_w_raw = Aw * WIND_D * df["wind_raw"] * HOURS
        hyps[f"{anchor}:E(fixed@9km2/GW + wind_net)/6.25"] = (e_pv9 + e_w_net) / NH3_HHV / 1e6
        hyps[f"{anchor}:E(fixed@9km2/GW + wind_raw)/6.25"] = (e_pv9 + e_w_raw) / NH3_HHV / 1e6
        hyps[f"{anchor}:E(fixed@9km2/GW)/6.25 (PV only)"] = e_pv9 / NH3_HHV / 1e6
        hyps[f"{anchor}:E(fixed@lat-density + wind_net)/6.25"] = (e_pvlat + e_w_net) / NH3_HHV / 1e6
        hyps[f"{anchor}:E(tracking@lat-density + wind_net)/6.25"] = (e_trk_lat + e_w_net) / NH3_HHV / 1e6
        hyps[f"{anchor}:E(tracking@9km2/GW + wind_net)/6.25"] = (e_trk_9 + e_w_net) / NH3_HHV / 1e6
        hyps[f"{anchor}:E(fixed@9km2/GW + wind_net)/9.0"] = (e_pv9 + e_w_net) / DESIGN_ELEC / 1e6
        hyps[f"{anchor}:max(E_fixed@9, E_wind_net)/6.25 (no overlap, best tech)"] = np.maximum(e_pv9, e_w_net) / NH3_HHV / 1e6
    if args.masked is not None:
        As, Aw = df["masked_solar_km2"], df["masked_wind_km2"]
        e_pv9 = As * PV_D_9 * df["fixed_pv"] * HOURS
        e_w_net = Aw * WIND_D * df["wind_net_0.93"] * HOURS
        hyps["masked-Sept-table:E(fixed@9km2/GW + wind_net)/6.25"] = (e_pv9 + e_w_net) / NH3_HHV / 1e6
        hyps["masked-Sept-table:E(fixed@lat-density + wind_net)/6.25"] = (As * df["masked_pv_density"] * df["fixed_pv"] * HOURS + e_w_net) / NH3_HHV / 1e6
    # the September design-scaling reconstruction, for contrast
    if "center_packing_1_no_exclusions_Mtpa" in df:
        hyps["Sept:rep-design scaled, fixed packing, no exclusions (center)"] = df["center_packing_1_no_exclusions_Mtpa"]
    if "rep_current_capacity_Mtpa" in df:
        hyps["Sept:rep surface capacity (paper_scaled, exclusions)"] = df["rep_current_capacity_Mtpa"]

    hist = df["historical_capacity_Mtpa"]
    rows = []
    for name, q in hyps.items():
        ratio = q / hist
        s = score(ratio)
        s_aus = score(ratio[df["country"] == "Australia"])
        rows.append({"hypothesis": name, **{f"all_{k}": v for k, v in s.items()}, **{f"aus_{k}": v for k, v in s_aus.items()}})
        df[f"Q::{name}"] = q
    out = pd.DataFrame(rows).sort_values("all_log_rmse")
    pd.set_option("display.width", 250)
    print(out[["hypothesis", "all_n", "all_median", "all_p25", "all_p75", "all_within_10pct", "all_within_25pct",
               "all_log_rmse", "aus_median", "aus_within_10pct"]].to_string(index=False))
    out.to_csv(args.output / "hypothesis_scores.csv", index=False)
    df.to_csv(args.output / "per_cell_reconstruction.csv", index=False)
    with open(args.output / "summary.json", "w") as f:
        json.dump({"n_archived_positive": int((un["historical_capacity_Mtpa"] > 0).sum()), "n_scored": int(len(df)),
                   "n_missing_cf": missing_cf, "scores": rows}, f, indent=2)
    # the three focal cells
    focal = df[((df.latitude == -23) & (df.longitude == -69)) | ((df.latitude == -23) & (df.longitude == 117)) |
               ((df.latitude == -21) & (df.longitude == 135))]
    cols = ["latitude", "longitude", "historical_capacity_Mtpa"] + [c for c in df.columns if c.startswith("Q::")][:6]
    print(focal[cols].to_string(index=False))


if __name__ == "__main__":
    main()
