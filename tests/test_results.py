"""model/results.py: outputs by run id, rows at named cells, coast bands and maps."""
from __future__ import annotations

import numpy as np
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


def test_copy_command_and_runs_table_name_the_run_store_location():
    run_id = "gl_dea2050_wacc5_bflat_wflat_land20c_fixed"
    command = results.copy_command(run_id)
    assert ("arc-oxford:/data/engs-df-green-ammonia/engs2523/green-lory-campaigns/keyruns_20260923_v1/30_keyruns/"
            f"{run_id}/runs/20260928-v4/global/") in command
    table = results.runs_table({"key run": run_id})
    assert table.loc["key run", "run_id"] == run_id and table.loc["key run", "finance"] == "wacc5"


def test_depth_classes_come_from_the_sea_run_site_costs():
    depth = results.depth_classes("gl_dea2050_wacc5_bflat_wflat_land20oct_fixed_offshore")
    assert list(depth["depth_class"].cat.categories) == list(results.DEPTH_CLASSES.values())
    shallow = depth[depth["depth_class"] == results.DEPTH_CLASSES["fixed"]]
    assert len(shallow) > 0 and shallow["sea_depth_m"].max() <= 70.0
    with pytest.raises(runs.RunDefinitionError, match="no site_costs_csv"):
        results.site_cost_table("gl_dea2050_wacc5_bflat_wflat_land20oct_fixed")


def test_cell_grid_places_each_cell_at_its_degree():
    frame = pd.DataFrame({"latitude": [-23.0, 60.0], "longitude": [-69.0, 180.0], "x": [1.0, 2.0]})
    longitude_edges, latitude_edges, values = results.cell_grid(frame, "x")
    assert values.shape == (len(latitude_edges) - 1, len(longitude_edges) - 1)
    row, column = np.searchsorted(latitude_edges, -23.0) - 1, np.searchsorted(longitude_edges, -69.0) - 1
    assert values[row, column] == 1.0 and int(np.isfinite(values).sum()) == 2
    with pytest.raises(ValueError, match="whole degrees"):
        results.cell_grid(frame.assign(latitude=[-23.5, 60.0]), "x")


def test_selected_plants():
    land = _plants([(0.0, 0.0, 300.0, 4e6), (2.0, 0.0, 200.0, 0.5e6)])
    sea = _plants([(0.0, 0.0, 280.0, 3e6), (1.0, 0.0, 500.0, 3e6)])
    assert len(results.select_plants(land, sea, "land")) == 1  # the 0.5 Mt/yr plant is below the cutoff
    assert len(results.select_plants(land, sea, "land", 0.0)) == 2
    assert list(results.select_plants(land, sea, "sea")["site"]) == ["sea", "sea"]
    assert list(results.select_plants(land, sea, "both")["site"]) == ["sea", "sea"]
    assert "Land plants: 2, median 250 EUR/t" in results.site_note(results.select_plants(land, sea, "land", 0.0))
    with pytest.raises(ValueError, match="sites"):
        results.select_plants(land, sea, "coast")


def test_cost_at_supply_is_the_cost_of_the_last_tonne():
    plants = _plants([(0.0, 0.0, 300.0, 4e6), (1.0, 0.0, 250.0, 2e6), (2.0, 0.0, 200.0, 0.5e6), (3.0, 0.0, 400.0, 10e6)])
    curve = results.supply_curve(plants)
    assert list(curve["lcoa_eur_per_t"]) == [250.0, 300.0, 400.0]  # the 0.5 Mt/yr plant is below the cutoff
    assert list(curve["cumulative_mt_per_year"]) == [2.0, 6.0, 16.0]
    assert results.cost_at_supply(plants, 2.0) == 250.0
    assert results.cost_at_supply(plants, 2.5) == 300.0
    assert results.cost_at_supply(plants, 16.0) == 400.0
    assert np.isnan(results.cost_at_supply(plants, 17.0))
    # 5 Mt/yr needs the plants at 250 (country Y, 2 Mt/yr) and 300 EUR/t (X, 4 Mt/yr)
    shares = results.supply_shares({"a": plants.assign(country=["X", "Y", "Z", "Y"])}, 5.0)
    assert shares.loc["X", "a"] == pytest.approx(66.7) and shares.loc["Y", "a"] == pytest.approx(33.3)
    assert "Z" not in shares.index


def test_summary_table_and_medians_by_group():
    land = _plants([(0.0, 0.0, 300.0, 4e6), (1.0, 0.0, 250.0, 2e6), (2.0, 0.0, 200.0, 0.5e6), (3.0, 0.0, 400.0, 10e6)])
    sea = _plants([(0.0, 1.0, 220.0, 3e6)])
    table = results.summary_table({"C": land}, {"C": sea}, volumes=(2.0,))
    assert table.loc["Median, land plants", "C"] == 275 and table.loc["Cells, sea plants", "C"] == 1
    assert table.loc["Cost at 2 Mt/yr, land plants", "C"] == 250
    assert table.loc["Cost at 2 Mt/yr, land and sea plants", "C"] == 220
    groups = pd.DataFrame({"latitude": [0.0, 1.0, 2.0, 3.0], "longitude": [0.0] * 4, "group": ["a", "a", "b", "b"]})
    medians = results.median_by({"x": land}, groups)
    assert medians.loc["a", "cells"] == 2 and medians.loc["a", "x"] == 275 and medians.loc["b", "x"] == 300


def test_bands_and_values_of_a_column():
    frame = pd.DataFrame({"v": [1.0, 2.0, 3.0, 4.0], "w": [0.0, 0.1, 0.6, 0.0999999999], "country": ["A", "A", "B", "A"]})
    bands = results.by_band(frame, "v", "w", (0, 0.5, 1))
    assert list(bands.index) == ["0 to 0.5", "0.5 to 1"] and list(bands["cells"]) == [3, 1]
    values = results.by_band(frame, "v", "w")  # 0.1 and 0.0999999999 are one value
    assert list(values["cells"]) == [1, 2, 1] and values.loc[0.1, "country with the most cells"] == "A"


def test_compare_surfaces_and_cheapest_cells():
    new = _plants([(0.0, 0.0, 330.0, 4e6), (1.0, 0.0, 275.0, 2e6), (2.0, 0.0, 220.0, 0.5e6)]).assign(
        wind_mw=[10.0, 0.0, 5.0], solar_mw=[10.0, 10.0, 5.0], country=["X", "Y", "Z"], elevation_m=0.0)
    old = new.assign(lcoa_eur_per_t=new["lcoa_eur_per_t"] / 1.1)
    both = results.compare_surfaces(new, old, keep=["country"])
    assert both["lcoa_change_pct"].round(6).eq(10.0).all() and list(both["country"]) == ["X", "Y", "Z"]
    assert both.loc[0, "wind_share_new"] == 0.5 and both.loc[1, "wind_share_old"] == 0.0
    cheapest = results.cheapest_cells(new, n=2, minimum_mt_per_year=1.0)
    assert list(cheapest["lcoa_eur_per_t"]) == [275.0, 330.0]  # the 0.5 Mt/yr cell is left out


def test_map_panels_and_supply_curves_draw():
    import matplotlib
    matplotlib.use("Agg")
    frame = pd.DataFrame({"latitude": [-23.0, -22.0], "longitude": [117.0, 117.0], "lcoa_eur_per_t": [400.0, 420.0],
                          "kind": ["a", "b"]})
    figure = results.plot_maps({"": frame}, "lcoa_eur_per_t", title="t", label="l", limits=(300, 500),
                               ramp=results.LCOA_COLOURMAP)
    assert len(figure.axes) == 2  # the map and its colour bar
    figure = results.plot_maps({"one": frame, "two": frame}, "lcoa_eur_per_t", title="t", label="l", scale="change")
    assert len(figure.axes) == 3
    figure = results.plot_maps({"": frame}, "kind", title="t", label="l",
                               categories={"a": "#000000", "b": "#777777", "c": "#ff0000"},
                               background=frame.assign(latitude=[-23.0, -21.0]), background_label="land")
    assert [text.get_text() for text in figure.legends[0].get_texts()] == ["a", "b", "land"]  # only what is drawn
    with pytest.raises(ValueError, match="scale"):
        results.plot_maps({"": frame}, "lcoa_eur_per_t", title="t", label="l", scale="log")
    plants = _plants([(0.0, 0.0, 300.0, 4e6), (1.0, 0.0, 250.0, 2e6)])
    figure = results.plot_supply_curves({"C": plants}, title="t", xmax=10.0, marks=(2.0,), names={"C": "C: base"})
    assert [text.get_text() for text in figure.axes[0].get_legend().get_texts()] == ["C: base", "Production of 2 Mt/yr"]


def test_map_cutoff_and_the_grey_key_entries():
    import matplotlib
    matplotlib.use("Agg")
    frame = pd.DataFrame({"latitude": [-23.0, -22.0], "longitude": [117.0, 117.0], "lcoa_eur_per_t": [280.0, 6000.0]})
    figure = results.plot_maps({"": frame}, "lcoa_eur_per_t", title="t", label="l", limits=(250, 1000),
                               above="above 1,000", background=frame.iloc[:1], background_label="grey")
    # the data cover every grey cell, so only the cutoff has a key entry
    assert [text.get_text() for text in figure.legends[0].get_texts()] == ["above 1,000"]
    figure = results.plot_maps({"": frame.iloc[:1]}, "lcoa_eur_per_t", title="t", label="l", limits=(250, 1000),
                               above="above 1,000", background=frame, background_label="grey")
    assert [text.get_text() for text in figure.legends[0].get_texts()] == ["grey"]  # no plant above the cutoff


def _case_frame(lcoa, plant, water, potential_mt=(2.0, 2.0)):
    return pd.DataFrame({"latitude": [-23.0, 35.0], "longitude": [-69.0, -100.0], "lcoa_eur_per_t": lcoa,
                         "lcoa_plant_eur_per_t": plant, "water_cost_eur_per_t": water,
                         results.CAPACITY_COLUMN: [1e6 * mt for mt in potential_mt]})


def test_cost_factors_add_up_to_each_case():
    base = _case_frame([303.0, 403.0], [300.0, 400.0], [3.0, 3.0])
    site = _case_frame([363.0, 413.0], [350.0, 405.0], [13.0, 8.0])
    finance = _case_frame([333.0, 383.0], [330.0, 380.0], [3.0, 3.0], potential_mt=(2.0, 0.5))
    both = _case_frame([400.0, 395.0], [387.0, 387.0], [13.0, 8.0])
    factors = results.cost_factors(base, site, finance, both).set_index("longitude")
    row = factors.loc[-69.0]
    assert (row["base"], row["construction"], row["water"], row["finance"]) == (303.0, 50.0, 10.0, 30.0)
    assert row["interaction"] == pytest.approx(400.0 - 303.0 - 50.0 - 10.0 - 30.0)
    assert factors.loc[-100.0, "finance"] == -20.0 and factors.loc[-100.0, "potential_mt_per_year"] == 0.5
    assert row["iso3"] == "CHL" and row["finance_region"] == "Central and South America"
    assert row["un_region"] == "Americas" and row["un_sub_region"] == "Latin America and the Caribbean"
    import matplotlib
    matplotlib.use("Agg")
    figure = results.plot_region_costs(factors.reset_index(), ["construction", "water", "finance", "interaction"], title="t")
    labels = [label.get_text() for label in figure.axes[0].get_yticklabels()]
    # from the bottom up: the dearer region first, so that the cheapest is at the top
    assert labels == ["Latin America and the Caribbean (1)", "Northern America (1)"]
    assert [text.get_text() for text in figure.axes[0].texts] == ["400", "395"]  # each region's mean


def test_regions_of_areas_missing_from_m49():
    regions = results.cell_regions().drop_duplicates("iso3").set_index("iso3")
    assert regions.loc["TWN", "un_sub_region"] == "Eastern Asia" and regions.loc["XKO", "un_region"] == "Europe"
    assert regions.loc["RUS", "un_sub_region"] == "Eastern Europe"  # M49 puts all of Russia in Europe


def test_potential_by_cost_band_and_region():
    land = _case_frame([260.0, 330.0], [0.0, 0.0], [0.0, 0.0], potential_mt=(2.0, 0.5))  # Chile, and a plant below the cutoff
    sea = _case_frame([420.0, 2600.0], [0.0, 0.0], [0.0, 0.0], potential_mt=(3.0, 4.0))
    table = results.potential_by_cost(land, sea, [250, 300, 450])
    assert list(table.columns) == ["Americas", "sea plants"]
    assert table["Americas"].tolist() == [0.002, 0.0, 0.0]  # Gt/yr in [250, 300), [300, 450) and beyond 450
    assert table["sea plants"].tolist() == [0.0, 0.003, 0.004]
    below = results.potential_below({"case": table}, 450)
    assert below.loc["land plants below 450 EUR/t", "case"] == 0.0  # rounded to 0.1 Gt/yr
    import matplotlib
    matplotlib.use("Agg")
    figure = results.plot_potential_by_cost({"case": table}, title="t")
    assert [text.get_text() for text in figure.legends[0].get_texts()] == ["Americas", "sea plants"]
