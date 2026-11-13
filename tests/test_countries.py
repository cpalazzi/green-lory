"""The countries of the cells and the regions of the finance tables (model/countries.py)."""
import pandas as pd
import pytest

from model import countries

LIST = pd.DataFrame({"region": ["WEU", "WEU", "UK", "AUS", "AFR", "WEU"],
                     "iso3": ["DNK", "FRA", "GBR", "AUS", "ZAF", ""],
                     "listed_as": ["Denmark", "France", "United Kingdom", "Australia", "South Africa", "Other"]})
UNITS = pd.DataFrame({
    "iso3": ["DNK", "GRL", "PYF", "FLK", "AUS", "ZAF", "MUS", "TON", "MCO", "ATA"],
    "country": ["Denmark", "Greenland", "French Polynesia", "Falkland Islands", "Australia", "South Africa", "Mauritius",
                "Tonga", "Monaco", "Antarctica"],
    "sovereign_iso3": ["DNK", "DNK", "FRA", "GBR", "AUS", "ZAF", "MUS", "TON", "MCO", "ATA"],
    "continent": ["Europe", "Europe", "Europe", "Europe", "Oceania", "Africa", "Africa", "Oceania", "Europe", "Antarctica"],
})
RATES = pd.DataFrame({"region": ["WEU", "UK", "AUS", "AFR", "ODA"], "lowcarbon": [0.042, 0.044, 0.061, 0.118, 0.068],
                      "reduced": [0.040, 0.043, 0.051, 0.051, 0.051]})


def test_a_territory_takes_its_sovereigns_region_and_an_unlisted_country_its_continents():
    regions = countries.country_regions(UNITS, LIST).set_index("iso3")
    # territories: the region of the sovereign (Greenland Denmark's, French Polynesia France's)
    assert regions.loc[["GRL", "PYF", "FLK"], "ameli_region"].tolist() == ["WEU", "WEU", "UK"]
    assert regions.loc["GRL", "region_basis"] == "sovereign listed"
    # countries the list does not name: the catch-all of the continent; the Pacific islands Other Asia
    assert regions.loc[["MUS", "TON", "MCO"], "ameli_region"].tolist() == ["AFR", "ODA", "WEU"]
    assert regions.loc["TON", "region_basis"] == "continent Oceania"
    assert pd.isna(regions.loc["ATA", "ameli_region"])


def test_finance_tables_give_each_code_its_regions_rate_and_antarctica_the_highest():
    tables = countries.finance_tables(countries.country_regions(UNITS, LIST), RATES)
    low, reduced = (tables[series].set_index("iso3")["interest_rate"] for series in ("lowcarbon", "reduced"))
    assert low[["GRL", "FLK", "TON", "MUS"]].tolist() == [0.042, 0.044, 0.068, 0.118]
    assert reduced[["GRL", "FLK", "TON", "MUS"]].tolist() == [0.040, 0.043, 0.051, 0.051]
    assert low["ATA"] == 0.118 and reduced["ATA"] == 0.051
    assert set(tables["reduced"]["tech"]) == {"all"}


def test_a_continent_without_a_catch_all_is_an_error():
    with pytest.raises(ValueError, match="CONTINENT_REGIONS"):
        countries.country_regions(UNITS.assign(continent="Atlantis"), LIST)


@pytest.mark.skipif(not countries.COUNTRIES_10M_SHAPEFILE.exists(), reason="Natural Earth 1:10m countries not present")
def test_map_units_take_iso_codes_or_their_iso_parent_or_sovereign():
    units = countries.load_units().drop_duplicates(["iso3", "country"]).set_index("country")
    assert units.loc["Somalia", "iso3"] == "SOM" and units.loc["Greenland", "sovereign_iso3"] == "DNK"
    codes = set(countries.load_units()["iso3"])
    # Somaliland counts in Somalia, Kosovo in Serbia; areas without any ISO code are left out
    assert {"SOM", "SRB", "CYP", "GRL", "PYF", "RUS"} <= codes and "-99" not in codes
