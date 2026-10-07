"""Regression: the focal cells reproduce the accepted results.

Three focal cells (inland Chile -23,-69; Pilbara -23,117; central Australia -21,135) for the
improved key run and the reconciling run, over one week and over the full year. The baseline
(tests/regression/focal_cells_baseline.csv) was produced on 7 October 2026 with the code
before the step-2 restructuring; its full-year LCOA equals the accepted ARC surfaces to within
5e-5 (the solver tolerance).

    python -m pytest -m slow tests/test_regression.py

Needs Gurobi, the weather stack, and the centred land tables in results/campaigns/.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

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
