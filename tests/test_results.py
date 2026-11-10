"""model/results.py: outputs by run id, rows at named cells, coast bands and maps."""
from __future__ import annotations

import pandas as pd
import pytest

from model import results, runs
from model.config import COUNTRIES_GEOJSON, REPO_ROOT


def test_run_store_locations_map_to_local_copies(tmp_path):
    # An ARC campaign path maps to the same path under the local campaigns folder
    assert results.output_dir("gl_dea2050_wacc5_bflat_wflat_land20c_fixed", campaigns_dir=tmp_path) == (
        tmp_path / "keyruns_20260923_v1/30_keyruns/gl_dea2050_wacc5_bflat_wflat_land20c_fixed/runs/20260928-v4/global")
    # "...; local copy <dir>" names the local folder
    assert results.output_dir("gl_way2050_ameli_bflat_wnone_land2c_fixed_pilot3cells", campaigns_dir=tmp_path) == (
        tmp_path / "land_reconcile_20260914_v1/arc_received_colocated_v6_center")
    # A repository path stays as it is
    assert results.output_dir("gpo_archived_iso1000_1mt") == (
        REPO_ROOT / "results/campaigns/verschuur_reconcile_20260907_v1/arc_received_20260915/networks/deposited-old-routes-20260914-v1")
    with pytest.raises(runs.RunDefinitionError, match="no output location"):
        results.output_dir("gl_dea2050_ameli_bspat_wspat_land20c_fixed")


def test_surface_file_is_found_by_kind_of_output(tmp_path):
    directory = tmp_path / "keyruns_20260923_v1/30_keyruns/gl_dea2050_wacc5_bflat_wflat_land2c_fixed/runs/20260928-v4/derived"
    directory.mkdir(parents=True)
    with pytest.raises(FileNotFoundError):
        results.load_surface("gl_dea2050_wacc5_bflat_wflat_land2c_fixed", campaigns_dir=tmp_path)
    pd.DataFrame({"latitude": [1.0], "longitude": [2.0]}).to_csv(directory / "global_run_results.csv", index=False)
    assert len(results.load_surface("gl_dea2050_wacc5_bflat_wflat_land2c_fixed", campaigns_dir=tmp_path)) == 1


def test_rows_at_named_cells():
    frame = pd.DataFrame({"latitude": [-23.0, -23.0, -21.0, 0.0], "longitude": [-69.0, 117.0, 135.0, 0.0], "x": [1, 2, 3, 4]})
    picked = results.at_cells(frame)
    assert list(picked.index) == ["Atacama", "NW Australia", "Central Australia"]
    assert list(picked.x) == [1, 2, 3]
    with pytest.raises(ValueError, match="found 0"):
        results.at_cells(frame, {"nowhere": (5.0, 5.0)})
    with pytest.raises(ValueError, match="found 2"):
        results.at_cells(pd.concat([frame, frame]))


def _banded():
    return pd.DataFrame({"latitude": [1.0, 2.0, 3.0], "longitude": [1.0, 1.0, 1.0],
                         "band": pd.Categorical(results.COAST_BANDS[1], categories=results.COAST_BANDS[1])})


def test_lcoa_by_band_uses_the_common_cells():
    surfaces = {"a": pd.DataFrame({"latitude": [1.0, 2.0, 3.0], "longitude": [1.0] * 3, "lcoa_eur_per_t": [400.0, 450.0, 500.0]}),
                "b": pd.DataFrame({"latitude": [1.0, 3.0], "longitude": [1.0] * 2, "lcoa_eur_per_t": [600.0, 550.0]})}
    table = results.lcoa_by_band(_banded(), surfaces).set_index("surface")
    assert (table.cells == 2).all()  # the 100-300 km cell is missing from b, so it is left out of both
    assert table.at["a", "median inland minus coast"] == pytest.approx(100.0)
    assert table.at["b", "median inland minus coast"] == pytest.approx(-50.0)


def test_production_by_band():
    suppliers = pd.DataFrame({"Latitude": [1.0, 3.0, 9.0], "Longitude": [1.0, 1.0, 9.0], "iso3": ["AUS", "AUS", "CHL"],
                              "production_t_per_year": [2e6, 3e6, 7e6]})
    row = results.production_by_band(_banded(), {"net": suppliers}).iloc[0]
    assert row["production_mt_per_year"] == pytest.approx(5.0)
    assert row["coast <100 km"] == pytest.approx(2.0) and row["inland >300 km"] == pytest.approx(3.0)
    outside = suppliers.assign(Latitude=[1.0, 4.0, 9.0])
    with pytest.raises(ValueError, match="outside"):
        results.production_by_band(_banded(), {"net": outside})


@pytest.mark.skipif(not COUNTRIES_GEOJSON.exists(), reason="data/countries.geojson not present")
def test_coast_distance_bands_in_australia():
    cells = pd.DataFrame({"latitude": [-32.0, -25.0, 10.0], "longitude": [116.0, 134.0, 10.0]})  # near Perth, the centre, Africa
    banded = results.coast_distance_bands(cells).set_index("longitude")
    assert list(banded.index) == [116.0, 134.0]
    assert banded.at[116.0, "band"] == "coast <100 km"
    assert banded.at[134.0, "band"] == "inland >300 km"


def _plants(rows):
    return pd.DataFrame(rows, columns=["latitude", "longitude", "lcoa_eur_per_t", results.CAPACITY_COLUMN])


def test_a_combined_map_shows_the_cheapest_plant_that_reaches_the_reference_output():
    land = _plants([(54.0, -4.0, 255.0, 0.03e6),   # cheap but tiny: the sea plant is shown
                    (-23.0, 117.0, 300.0, 5e6),    # cheaper than its sea plant
                    (-21.0, 135.0, 320.0, 2e6),    # no sea plant
                    (10.0, 10.0, 280.0, 0.5e6)])   # neither plant reaches 1 Mt/yr: left out
    sea = _plants([(54.0, -4.0, 500.0, 4e6), (-23.0, 117.0, 450.0, 3e6), (55.0, 2.0, 440.0, 4.4e6), (10.0, 10.0, 600.0, 0.2e6)])
    shown = results.lowest_feasible_plant(land, sea).set_index(["latitude", "longitude"])
    assert sorted(shown.index) == [(-23.0, 117.0), (-21.0, 135.0), (54.0, -4.0), (55.0, 2.0)]
    assert shown.loc[(54.0, -4.0), "site"] == "sea" and shown.loc[(54.0, -4.0), "lcoa_eur_per_t"] == 500.0
    assert shown.loc[(-23.0, 117.0), "site"] == "land" and shown.loc[(-21.0, 135.0), "site"] == "land"
    assert shown.loc[(55.0, 2.0), "site"] == "sea"
    # with the cutoff at zero the map is simply the cheaper of the two
    assert results.lowest_feasible_plant(land, sea, minimum_mt_per_year=0.0).set_index(["latitude", "longitude"]).loc[(54.0, -4.0), "site"] == "land"


def test_maps_draw_a_small_surface():
    import matplotlib
    matplotlib.use("Agg")
    frame = pd.DataFrame({"latitude": [-23.0, -22.0], "longitude": [117.0, 117.0], "lcoa_eur_per_t": [400.0, 420.0],
                          "currency": ["EUR", "EUR"]})
    figure = results.plot_map(frame, show_country_borders=False)
    assert figure.data[0].type == "choropleth" and len(figure.data[0].locations) == 2
    static, axes = results.plot_map_static(frame, show_country_borders=False)
    assert axes.get_images()


def test_run_store_locations_are_folders():
    # text after the folder, such as a remark in brackets, would make output_dir name a folder that does not exist
    for record in runs.load_run_store():
        location = record["location"].strip().replace("; local copy ", ";")
        assert not any(character.isspace() for character in location), record["run_id"]
