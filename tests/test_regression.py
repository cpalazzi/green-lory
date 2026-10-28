"""Regression: the focal cells and a band of the land table reproduce the accepted results.

Three focal cells (inland Chile -23,-69; Pilbara -23,117; central Australia -21,135) for the
improved key run and the reconciling run, over one week and over the full year. The baseline
(tests/regression/focal_cells_baseline.csv) was regenerated on 12 October 2026, when the battery
standing loss became the technology YAML's 0.1 % a day (the plant bundle's 0.001 had been
applied per hour; full-year LCOA -0.006 % to -0.05 %), the ammonia demand was derived from the
heating value, and the electrolysis variable cost of 0.537 EUR/MWh, which had no source, was
dropped (-4.5 EUR/t in the key run, -4.3 in the reconciling run). With the old loss and demand
forced back, the code reproduced the earlier baseline of 7 October 2026 exactly; that
baseline's full-year LCOA equals the accepted ARC surfaces to within 5e-5 (the solver
tolerance). The key run's rows were regenerated again on 14 October 2026 for the values of
campaign check D9 in the DEA 2050 YAML (hydrogen tank and Haber-Bosch costs on the HHV basis,
the ammonia tank from its source in AUD, DEA's compressor with its own electricity use,
Haber-Bosch minimum load 0.2 and ramps 0.2 per hour): full-year LCOA -21.7 EUR/t in Chile,
-22.2 in the Pilbara and -18.7 in central Australia. The reconciling run, on the Way YAML, is
unchanged. On 15 October 2026 the water use became 1.62 m3 per tonne (DEA 2050 electrolysis)
instead of 1.5: the key run's LCOA rises by exactly 0.24 EUR/t (0.12 m3 at 2 EUR/m3), since
water is added after the solve. On 28 October 2026 the ammonia tank became 80.3 EUR/MWh
(502 EUR2020/t, Cesaro 2022, the value of green-caribou) instead of 260 (Fletcher et al. 2023),
and the key run's rows were regenerated: full-year LCOA -5.5 EUR/t in Chile, -5.6 in the
Pilbara and -1.8 in central Australia.

The land build, rerun for latitudes -24 to -22 (1,080 cells, including two focal cells) with a
15-degree slope limit, against tests/regression/land_center_100pct_lat_m24_m22.csv. That file
held the rows of the accepted centred 100 % table of 23 September 2026 until 8 October 2026,
when protected land replaced protected area (model/land_table.py, _protected_land_pct): inland
cells moved by at most 0.18 percentage points of protection, coastal cells with protected
water regained land (Ningaloo, 22 S 114 E: exclusion factor 0 to 0.58).

    python -m pytest -m slow tests/test_regression.py

Needs Gurobi, the weather stack, the land inputs (MODIS, WDPA, GEBCO) and the centred land
tables in results/campaigns/.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from model import config
from model.runs import load_runs, surface_arguments
from model.surface import RunSettings, run_surface

pytestmark = pytest.mark.slow

BASELINE = Path(__file__).parent / "regression" / "focal_cells_baseline.csv"
FOCAL_CELLS = [(-23.0, -69.0), (-23.0, 117.0), (-21.0, 135.0)]
RUN_IDS = ["gl_dea2050_wacc5_bflat_wflat_land20c_fixed", "rep_way2050_flat_amelired_4h_tracking_nominal_h2"]

# Columns of the baseline that the restructured model no longer writes (the historical
# max_* capacity aliases and the removed union fallback flag)
REMOVED_COLUMNS = {
    "capacity_limit_technology", "legacy_capacity_alias_method", "max_ammonia_capacity_mtpa",
    "max_ammonia_capacity_t", "max_gridless_onshore_ammonia_capacity_mtpa",
    "max_gridless_onshore_ammonia_capacity_t", "max_onshore_ammonia_capacity_mtpa",
    "max_onshore_ammonia_capacity_t", "onshore_capacity_limit_technology",
    "onshore_renewable_capacity_scale_factor", "renewable_capacity_scale_factor",
    "renewable_union_area_is_conservative_fallback",
}
TOLERANCES = {  # column -> relative tolerance; everything else numeric must agree to 1e-3
    "lcoa_eur_per_t": 1e-5,
    "lcoa_plant_eur_per_t": 1e-5,
    "total_cost_eur_per_year": 1e-5,
    "scaled_design_max_gridless_onshore_ammonia_capacity_t": 1e-3,
    # Generation used rather than curtailed is not unique at the optimum (curtailment costs
    # nothing), so these dispatch totals differ by up to 0.15 % between solver runs
    "power_bus_generator_supply_mwh": 1e-2,
    "grid_energy_reference_mwh": 1e-2,
    "grid_energy_tolerance_mwh": 1e-2,
}


def _run(run_id: str, horizon: str) -> pd.DataFrame:
    pytest.importorskip("gurobipy")
    try:
        arguments = surface_arguments(run_id, runs=load_runs())
    except FileNotFoundError:
        pytest.skip(f"data for {run_id} not present")
    if not arguments["weather_dir"].exists():
        pytest.skip(f"weather for {run_id} not present")
    settings = arguments.pop("settings")
    if horizon == "week":
        settings = RunSettings(**{**settings.__dict__, "max_snapshots": int(168 / settings.time_step)})
    return run_surface(FOCAL_CELLS, settings=settings, quiet=True, threads_per_worker=4, num_workers=3,
                       fail_fast=True, **arguments)


@pytest.mark.parametrize("horizon", ["week", "year"])
@pytest.mark.parametrize("run_id", RUN_IDS)
def test_focal_cells_reproduce_the_baseline(run_id, horizon):
    baseline = pd.read_csv(BASELINE)
    expected = baseline[(baseline["run"] == run_id) & (baseline["horizon"] == horizon)]
    expected = expected.drop(columns=["run", "horizon"]).set_index(["latitude", "longitude"]).sort_index()
    actual = _run(run_id, horizon).set_index(["latitude", "longitude"]).sort_index()

    assert list(actual.index) == list(expected.index)
    missing = set(expected.columns) - set(actual.columns) - REMOVED_COLUMNS
    assert not missing, f"columns no longer written: {sorted(missing)}"
    for column in sorted(set(expected.columns) & set(actual.columns)):
        want, got = expected[column], actual[column]
        if pd.api.types.is_numeric_dtype(want) and pd.api.types.is_numeric_dtype(got):
            np.testing.assert_allclose(
                got.astype(float).to_numpy(), want.astype(float).to_numpy(),
                rtol=TOLERANCES.get(column, 1e-3), atol=1e-6, equal_nan=True, err_msg=column,
            )
        elif column not in {"run_id", "scenario_id", "manifest_sha256"}:
            # An empty text field reads back from CSV as NaN
            assert list(got.fillna("").astype(str)) == list(want.fillna("").astype(str)), column


LAND_BAND = Path(__file__).parent / "regression" / "land_center_100pct_lat_m24_m22.csv"


def test_land_build_reproduces_a_band_of_the_accepted_table(tmp_path):
    inputs = [config.LAND_COVER_FILE, config.GEBCO_FILE, *config.WDPA_SHAPEFILES]
    if not all(path.exists() for path in inputs):
        pytest.skip("land inputs not present")
    from model import land_table

    output = tmp_path / "band.csv"
    argv = ["--output", str(output), "--land-competition-fraction", "1.0", "--include-offshore-wind",
            "--land-cover", str(config.LAND_COVER_FILE), "--slope-raster", str(config.GEBCO_FILE),
            "--max-slope-degrees", "15", "--min-lat", "-24", "--max-lat", "-22"]
    for shapefile in config.WDPA_SHAPEFILES:
        argv += ["--protected-area", str(shapefile)]
    land_table.main(argv)

    expected = pd.read_csv(LAND_BAND).set_index(["latitude", "longitude"]).sort_index()
    actual = pd.read_csv(output).set_index(["latitude", "longitude"]).sort_index()
    assert list(actual.index) == list(expected.index)
    assert list(actual.columns) == list(expected.columns)
    # protected land never exceeds the land the table counts
    assert (actual["protected_area_pct"] <= actual["onshore_land_pct"] + 1e-9).all()
    classes = actual[list(land_table.SLOPE_CLASS_COLUMNS)]
    has_land = classes.sum(axis=1) > 0
    np.testing.assert_allclose(classes[has_land].sum(axis=1), 100.0, atol=1e-9)
    # land up to 15 degrees lies between the classes up to 14 and up to 20 degrees
    up_to_14 = classes.iloc[:, :2].sum(axis=1)
    assert (actual["slope_suitable_land_pct"] >= up_to_14 - 1e-9).all()
    assert (actual["slope_suitable_land_pct"] <= up_to_14 + classes.iloc[:, 2] + 1e-9).all()
    for column in expected.columns:
        want, got = expected[column], actual[column]
        if pd.api.types.is_numeric_dtype(want):
            np.testing.assert_allclose(got.to_numpy(float), want.to_numpy(float), rtol=1e-9, atol=1e-9,
                                       equal_nan=True, err_msg=column)
        else:
            assert list(got.fillna("").astype(str)) == list(want.fillna("").astype(str)), column


def test_run_cells_reproduce_the_key_run_list():
    from model.land_table import run_cells

    folder = Path(__file__).parents[1] / "results" / "campaigns" / "land_center_20260923_v1"
    full, listed = folder / "max_capacities_center_100pct_slope15.csv", folder / "locations_onshore_suitable_center_20260923.csv"
    if not (full.exists() and listed.exists()):
        pytest.skip("September land tables not present")
    cells = run_cells(pd.read_csv(full))
    expected = pd.read_csv(listed).sort_values(["lat", "lon"]).reset_index(drop=True)
    pd.testing.assert_frame_equal(cells, expected[["lat", "lon"]], check_dtype=False)

