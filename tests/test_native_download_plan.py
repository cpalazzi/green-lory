import copy
import unittest
from urllib.parse import parse_qs, urlparse

from reconciliation.land.plan_native_downloads import parse_response, query_url


def response():
    name = "MCD12Q1.A2022001.h28v11.061.2023244133105"
    return {"hits": 1, "items": [{"meta": {
        "provider-id": "LPCLOUD", "concept-id": "G1-LPCLOUD", "revision-id": 2,
        "collection-concept-id": "C1-LPCLOUD"}, "umm": {
        "CollectionReference": {"ShortName": "MCD12Q1", "Version": "061"},
        "GranuleUR": name, "SpatialExtent": {},
        "RelatedUrls": [{"Type": "GET DATA", "URL":
            "https://data.lpdaac.earthdatacloud.nasa.gov/example/" + name + ".hdf"}],
        "DataGranule": {"ArchiveAndDistributionInformation": [{"Size": 7.38, "SizeUnit": "MB"}]},
    }}]}


class NativeDownloadPlanTest(unittest.TestCase):
    def test_exact_centered_and_southwest_bounds(self):
        site = {"latitude": -23, "longitude": 117}
        url, bounds = query_url(site, 2022, "center")
        self.assertEqual(bounds, (116.5, -23.5, 117.5, -22.5))
        self.assertEqual(parse_qs(urlparse(url).query)["version"], ["061"])
        self.assertEqual(query_url(site, 2022, "southwest")[1], (117, -23, 118, -22))

    def test_geometry_fail_closed(self):
        for site in ({"latitude": 0, "longitude": 180},
                     {"latitude": float("nan"), "longitude": 0}):
            with self.assertRaises(ValueError):
                query_url(site, 2022, "center")

    def test_native_filename_and_provenance(self):
        row, = parse_response(response(), 2022)
        self.assertEqual(row["tile"], "h28v11")
        self.assertTrue(row["filename"].endswith(".hdf"))
        self.assertEqual(row["revision_id"], 2)

    def test_reject_empty_pagination_and_duplicate_tile(self):
        fixtures = [{"hits": 0, "items": []}, {**response(), "hits": 2}]
        duplicate = response()
        duplicate["hits"] = 2
        duplicate["items"] *= 2
        for bad in [*fixtures, duplicate]:
            with self.assertRaises(ValueError):
                parse_response(bad, 2022)

    def test_reject_wrong_product_year_collection_or_untrusted_url(self):
        for field, value in (("ShortName", "MCD12C1"), ("Version", "006")):
            bad = response()
            bad["items"][0]["umm"]["CollectionReference"][field] = value
            with self.assertRaises(ValueError):
                parse_response(bad, 2022)
        with self.assertRaises(ValueError):
            parse_response(response(), 2020)
        bad = copy.deepcopy(response())
        bad["items"][0]["umm"]["RelatedUrls"][0]["URL"] = "https://example.com/incorrect.hdf"
        with self.assertRaises(ValueError):
            parse_response(bad, 2022)
