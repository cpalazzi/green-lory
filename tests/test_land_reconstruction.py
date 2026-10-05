import unittest
import numpy as np
from reconciliation.land.core import Cell, WIND, SOLAR, paper_pv_density, read_cmg_cell, masked_suitability, rectangle_area


class LandReconstructionTest(unittest.TestCase):
    def test_centered_and_southwest_bounds(self):
        self.assertEqual(Cell(-23, 117).bounds, (116.5, -23.5, 117.5, -22.5))
        self.assertEqual(Cell(-23, 117, "southwest").bounds, (117, -23, 118, -22))

    def test_area_is_not_squared_east_west_length(self):
        self.assertAlmostEqual(Cell(60, 0).area_km2 / Cell(0, 0).area_km2, .5)

    def test_paper_density_equation_and_hemisphere(self):
        self.assertAlmostEqual(float(paper_pv_density(0)), 1000/9)
        expected = (1000/9) / (np.cos(np.deg2rad(23)) + np.sin(np.deg2rad(23))/np.tan(np.deg2rad(66.55-23)))
        self.assertAlmostEqual(float(paper_pv_density(-23)), expected)
        self.assertEqual(float(paper_pv_density(-23)), float(paper_pv_density(23)))
        self.assertEqual(float(paper_pv_density(70)), 0)

    def test_mask_union_not_product(self):
        # Identical 50% exclusions leave 50%, not 25%. Joint masks own q.
        f = np.zeros((1, 17)); f[0,16] = 1
        est, low, high = masked_suitability(f, np.array([.5]), SOLAR)
        np.testing.assert_allclose([est, low, high], .5)

    def test_subpixel_composition_bounds(self):
        f = np.zeros((1, 17)); f[0,1] = .5; f[0,16] = .5
        est, low, high = masked_suitability(f, np.array([.5]), SOLAR)
        self.assertEqual(float(est[0]), .25)
        self.assertEqual(float(low[0]), 0)
        self.assertEqual(float(high[0]), .5)

    def test_water_mapping_is_not_offshore(self):
        self.assertEqual(WIND[0], 0); self.assertEqual(SOLAR[0], 0)
        self.assertEqual(SOLAR[16], 1)

    def test_cmg_latitude_window_and_fill_rejection(self):
        class FakeSDS:
            def __getitem__(self, item):
                yy, xx, _ = item
                assert all(type(v) is int for v in (yy.start, yy.stop, xx.start, xx.stop))
                self.window = (yy.start, yy.stop, xx.start, xx.stop)
                arr = np.zeros((yy.stop-yy.start, xx.stop-xx.start, 17), dtype=np.uint8)
                arr[...,16] = 100
                return arr
        s = FakeSDS()
        f, a, _ = read_cmg_cell(s, Cell(-23,117,"southwest"))
        self.assertEqual(s.window, (2240,2260,5940,5960))
        self.assertAlmostEqual(float(a.sum()), Cell(-23,117,"southwest").area_km2)
        self.assertTrue((f[...,16] == 1).all())

    def test_dateline_wrap(self):
        class FakeSDS:
            def __getitem__(self,item):
                y,x,_ = item
                a=np.zeros((y.stop-y.start,x.stop-x.start,17));a[...,0]=100
                return a
        f, a, _ = read_cmg_cell(FakeSDS(), Cell(0,-180))
        self.assertEqual(f.shape,(20,20,17))
        self.assertAlmostEqual(float(a.sum()),Cell(0,0).area_km2)

    def test_joint_dem_mask_keeps_negative_flat_land(self):
        import xarray as xr
        from shapely.geometry import box
        from reconciliation.land.pilot_joint_masks import masked_dem_fraction
        # Forty DEM centers per one-degree cell plus a two-cell halo.
        coord=np.arange(-2,42)*.025+.0125
        dem=xr.DataArray(np.full((44,44),-20.),coords={"lat":coord,"lon":coord},dims=("lat","lon"))
        q,slope,protected,below=masked_dem_fraction(dem,Cell(0,0,"southwest"),box(0,0,.5,1))
        np.testing.assert_allclose(slope,1)
        np.testing.assert_allclose(below,1)
        np.testing.assert_allclose(q[:,:10],0)
        np.testing.assert_allclose(q[:,10:],1)
        np.testing.assert_allclose(q,1-protected)
        # Descending latitude coordinates must yield exactly the same CMG bins.
        reverse=masked_dem_fraction(dem.sortby("lat",ascending=False),Cell(0,0,"southwest"),box(0,0,.5,1))
        for actual,expected in zip(reverse,(q,slope,protected,below)):
            np.testing.assert_allclose(actual,expected)

    def test_dem_slope_threshold_in_degrees(self):
        import xarray as xr
        from reconciliation.land.core import RADIUS_KM
        from reconciliation.land.pilot_joint_masks import masked_dem_fraction
        coord=np.arange(-2,42)*.025+.0125
        for degrees,expected in [(14.,1.),(16.,0.)]:
            z=np.broadcast_to((np.deg2rad(coord)*RADIUS_KM*1000*np.tan(np.deg2rad(degrees)))[:,None],(44,44))
            dem=xr.DataArray(z,coords={"lat":coord,"lon":coord},dims=("lat","lon"))
            q,slope,protected,below=masked_dem_fraction(dem,Cell(0,0,"southwest"),None)
            np.testing.assert_allclose(q,expected)
            np.testing.assert_allclose(slope,expected)


if __name__ == "__main__":
    unittest.main()
