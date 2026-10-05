"""Tests for the legacy-surface tools: collect_summaries, compare_runs and the
JSON-lines input path of build_legacy_supplier_table, on a tiny synthetic run."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "reconciliation" / "legacy_lcoa"


def _summary(name, lat, lon, lcoa, wind, tracking, wacc=0.051):
    return {"run_id": f"{name}__stated_4h_mean__cell_wacc", "cell": name, "lat": lat, "lon": lon,
            "cell_meta": {"iso3": "XXX", "country": "Testland", "wacc_source": "test"},
            "variant": "stated_4h_mean", "wacc_name": "cell_wacc", "wacc_rate": wacc,
            "snapshots_used": 2190, "objective_usd_per_year": lcoa * 1e6, "lcoa_usd_per_t": lcoa,
            "capacities_mw": {"Wind": wind, "Solar": 0.0, "SolarTracking": tracking, "Electrolysis": 2000.0,
                              "HB": 800.0, "CompressedH2Store": 9000.0, "Battery": 500.0, "Ammonia": 20000.0},
            "electricity_capex_fraction_of_objective": 0.4, "curtailed_fraction": 0.1,
            "primary_electricity_mwh_per_t_snapshot_basis": 9.0, "elapsed_s": 1.0}


def _write_run(root: Path, cells, shards=2, perturb=0.0):
    root.mkdir(parents=True)
    for i, (name, lat, lon, lcoa, wind, tracking) in enumerate(cells):
        shard = root / f"shard_{i % shards:02d}"
        d = shard / f"{name}__stated_4h_mean__cell_wacc"
        d.mkdir(parents=True, exist_ok=True)
        (d / "summary.json").write_text(json.dumps(_summary(name, lat, lon, lcoa * (1 + perturb), wind, tracking)))
    for s in range(shards):
        shard = root / f"shard_{s:02d}"
        (shard / "manifest.json").write_text(json.dumps({"harness": "h", "commit": "94de8ce", "era": "may2023",
                                                         "capex_source": "xcost45", "enable_tracking_cost_ratio": 1.0587,
                                                         "weather_files": {"Solar.nc": "abc"}, "source_files": {}, "costs_xlsx": {},
                                                         "weather_source": {"kind": "compact_store"}, "packages": {"pypsa": "0.25.1"},
                                                         "python": "3.11", "solver": "gurobi"}))
        rows = [c for j, c in enumerate(cells) if j % shards == s]
        pd.DataFrame([{"cell": r[0], "lcoa_usd_per_t": r[3]} for r in rows]).to_csv(shard / "results.csv", index=False)


CELLS = [("cA", -23.0, -69.0, 221.33, 0.0, 3380.0), ("cB", -23.0, 117.0, 242.64, 0.0, 3995.0),
         ("cC", -21.0, 135.0, 234.04, 348.0, 3039.0), ("cD", 10.0, 20.0, 300.0, 1200.0, 0.0)]


def run(script, *args):
    return subprocess.run([sys.executable, str(script), *map(str, args)], capture_output=True, text=True, cwd=ROOT)


def test_collect_summaries_roundtrip(tmp_path):
    run_dir = tmp_path / "run"
    _write_run(run_dir, CELLS)
    out = tmp_path / "collected"
    r = run(TOOLS / "collect_summaries.py", "--run", run_dir, "--output", out)
    assert r.returncode == 0, r.stderr
    lines = [json.loads(l) for l in (out / "summaries.jsonl").read_text().splitlines()]
    assert len(lines) == 4 and {l["cell"] for l in lines} == {"cA", "cB", "cC", "cD"}
    meta = json.loads((out / "collect_manifest.json").read_text())
    assert meta["summaries"] == 4 and sum(meta["results_csv_rows"].values()) == 4
    assert len([k for k in meta["manifests"] if k.endswith("manifest.json")]) == 2
    # refuses to overwrite
    assert run(TOOLS / "collect_summaries.py", "--run", run_dir, "--output", out).returncode != 0


def test_collect_summaries_rejects_duplicate_cells(tmp_path):
    run_dir = tmp_path / "run"
    _write_run(run_dir, CELLS)
    dup = run_dir / "shard_01" / "cA_dup__stated_4h_mean__cell_wacc"
    dup.mkdir()
    (dup / "summary.json").write_text(json.dumps(_summary("cA", -23.0, -69.0, 1.0, 0.0, 1.0)))
    assert run(TOOLS / "collect_summaries.py", "--run", run_dir, "--output", tmp_path / "c").returncode != 0


def _land_and_archived(tmp_path):
    land = pd.DataFrame({"latitude": [c[1] for c in CELLS], "longitude": [c[2] for c in CELLS],
                         "center_solar_shipping_km2": [227.57, 113.18, 115.28, 50.0],
                         "center_wind_shipping_km2": [227.56, 113.18, 115.28, 50.0],
                         "center_cell_area_km2": [12000.0] * 4})
    land_path = tmp_path / "land.csv"
    land.to_csv(land_path, index=False)
    arch = pd.DataFrame({"Index": [f"{int(c[1])}_{int(c[2])}_1000000.0" for c in CELLS],
                         "Latitude": [c[1] for c in CELLS], "Longitude": [c[2] for c in CELLS],
                         "iso3": ["CHL", "AUS", "AUS", "TCD"], "country": ["Chile", "Australia", "Australia", "Chad"],
                         "LCOA": [212.92, 233.69, 226.39, 290.0], "Production": [1e6] * 4,
                         "Max_capacity": [9.769406259, 4.019748515, 4.324243441, 0.3], "Electricity_Cost_Frac": [0.3] * 4})
    arch_path = tmp_path / "archived.csv"
    arch.to_csv(arch_path, index=False)
    cells = pd.DataFrame({"name": [c[0] for c in CELLS], "lat": [c[1] for c in CELLS], "lon": [c[2] for c in CELLS]})
    cells_path = tmp_path / "cells.csv"
    cells.to_csv(cells_path, index=False)
    return land_path, arch_path, cells_path


def test_supplier_table_from_directory_and_jsonl_agree(tmp_path):
    run_dir = tmp_path / "run"
    _write_run(run_dir, CELLS)
    land, arch, cells = _land_and_archived(tmp_path)
    r = run(TOOLS / "collect_summaries.py", "--run", run_dir, "--output", tmp_path / "collected")
    assert r.returncode == 0, r.stderr
    outs = []
    for src, out in [(run_dir, tmp_path / "t_dir"), (tmp_path / "collected" / "summaries.jsonl", tmp_path / "t_jsonl")]:
        r = run(TOOLS / "build_legacy_supplier_table.py", "--run", src, "--land", land, "--archived", arch,
                "--cells", cells, "--output", out)
        assert r.returncode == 0, r.stderr
        outs.append(pd.read_csv(out / "legacy_replicated_suppliers.csv").set_index("Index").sort_index())
    pd.testing.assert_frame_equal(outs[0], outs[1])
    table = outs[0]
    # the recovered rule: PV-limited at cA (227.57*140/3380 = 9.4255), wind-limited at cC (115.28*7.3/348)
    assert table.loc["-23_-69_1000000.0", "Max_capacity"] == pytest.approx(227.57 * 140 / 3380, rel=1e-9)
    assert table.loc["-21_135_1000000.0", "Max_capacity"] == pytest.approx(115.28 * 7.3 / 348, rel=1e-9)
    assert table.loc["10_20_1000000.0", "Max_capacity"] == pytest.approx(50.0 * 7.3 / 1200, rel=1e-9)
    contract = json.loads((tmp_path / "t_jsonl" / "gpo_export" / "contract.json").read_text())
    assert contract["capacity_method"] == "legacy_rule_complete_overlap_v1"
    assert "run_summaries_sha256" in contract["inputs"]
    summary = json.loads((tmp_path / "t_jsonl" / "summary.json").read_text())
    assert summary["n_missing"] == 0 and summary["n_cells"] == 4


def test_compare_runs_identical_and_perturbed(tmp_path):
    a = tmp_path / "a"
    b = tmp_path / "b"
    _write_run(a, CELLS)
    _write_run(b, CELLS)
    r = run(TOOLS / "compare_runs.py", "--run-a", a, "--run-b", b, "--label-a", "x", "--label-b", "y", "--output", tmp_path / "cmp0")
    assert r.returncode == 0, r.stderr
    s = json.loads((tmp_path / "cmp0" / "summary.json").read_text())
    assert s["common_cells"] == 4 and s["lcoa_rel_diff"]["max"] == 0.0 and s["cells_lcoa_rel_diff_above_1e-6"] == 0
    assert s["run_a"]["all_shard_manifests_consistent"] is True
    c = tmp_path / "c"
    _write_run(c, CELLS[:3], perturb=1e-3)   # 0.1 % higher LCOA, one cell missing
    r = run(TOOLS / "compare_runs.py", "--run-a", a, "--run-b", c, "--output", tmp_path / "cmp1")
    assert r.returncode == 0, r.stderr
    s = json.loads((tmp_path / "cmp1" / "summary.json").read_text())
    assert s["common_cells"] == 3 and s["only_in_a"] == 1 and s["cells_lcoa_rel_diff_above_1e-4"] == 3
    assert s["lcoa_rel_diff"]["max"] == pytest.approx(1e-3, rel=1e-6)   # relative to run A
    # jsonl input on one side
    assert run(TOOLS / "collect_summaries.py", "--run", a, "--output", tmp_path / "ca").returncode == 0
    r = run(TOOLS / "compare_runs.py", "--run-a", tmp_path / "ca" / "summaries.jsonl", "--run-b", b, "--output", tmp_path / "cmp2")
    assert r.returncode == 0, r.stderr
    assert json.loads((tmp_path / "cmp2" / "summary.json").read_text())["lcoa_rel_diff"]["max"] == 0.0
