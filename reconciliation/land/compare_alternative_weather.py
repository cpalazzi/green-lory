"""Verify the six fetched alternative profiles and compare raw CF definitions."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from reconciliation.land.compare_model_evidence import CAMPAIGN, digest


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    source = CAMPAIGN / "arc_received_alternative_weather_v2/alternative_weather_probe_v2"
    manifest = json.loads((source / "manifest.json").read_text())
    assert manifest["selected_cells_complete"] and len(manifest["rows"]) == 6
    old_path = CAMPAIGN / "audit/model-evidence-v2/weather_cf_comparison.csv"
    old = pd.read_csv(old_path)
    rows, files = [], []
    for entry in manifest["rows"]:
        path = source / f"{entry['site']}_{entry['variable']}.nc"
        tech = "fixed_pv" if entry["variable"] == "cf_solar" else "wind"
        ref = old[(old.site == entry["site"]) & (old.technology == tech)]
        assert len(ref) == 1
        ref = ref.iloc[0]
        with xr.open_dataset(path) as ds, xr.open_dataset(ref.pilot_path) as previous:
            x = np.asarray(ds[entry["variable"]].values, dtype="<f8")
            assert x.shape == (8760,) and np.isfinite(x).all() and (x >= 0).all()
            np.testing.assert_array_equal(ds.time.values, previous.time.values)
            assert hashlib.sha256(x.tobytes()).hexdigest() == entry["profile_float64_sha256"]
            np.testing.assert_allclose(x.mean(), entry["cf_mean_if_complete"], rtol=1e-12)
        rows.append({"site": entry["site"], "latitude": entry["latitude"],
            "longitude": entry["longitude"], "technology": tech,
            "old_profile_raw_cf": ref.raw_cf, "alternative_profile_raw_cf": float(x.mean()),
            "relative_change_pct": 100. * (x.mean() / ref.raw_cf - 1.),
            "old_model_cf": ref.model_cf,
            "alternative_cf_if_same_wake_multiplier": float(x.mean() * ref.wake_multiplier),
            "interpretation": "same year and point coordinates; resolution/technology/normalization may differ; not weather-only OAT",
            "additional_wake_loss_in_alternative_source_verified": False})
        files.append({"path": str(path), "sha256": digest(path)})
    pd.DataFrame(rows).to_csv(a.output / "comparison.csv", index=False)
    report = {"qa_pass": True, "profiles_verified": 6, "hourly_observations_per_profile": 8760,
        "source_manifest_sha256": digest(source / "manifest.json"),
        "old_comparison_sha256": digest(old_path), "script_sha256": digest(Path(__file__)),
        "files": files, "global_completeness_verified": False,
        "plant_reoptimised_with_alternative_weather": False}
    (a.output / "manifest.json").write_text(json.dumps(report, indent=2) + "\n")
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
