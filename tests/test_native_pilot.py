import numpy as np
import pytest
import shapely
import xarray as xr

from reconciliation.land.core import Cell, RADIUS_KM
from reconciliation.land.run_native_pilot import dem_slopes, integrate, sample_regular


class SyntheticTile:
    def sample(self, xs, ys):
        xs, ys = np.broadcast_arrays(xs, ys)
        # Western half barren; eastern half forest.
        return {"LC_Type1": np.where(xs < 0, 16, 1).astype(np.uint8),
                "QC": np.zeros(xs.shape, dtype=np.uint8), "LW": np.full(xs.shape, 2, dtype=np.uint8)}


def test_dem_slope_keeps_below_sea_land_and_physical_gradient():
    lat = np.arange(-.7, .71, .1)
    lon = np.arange(-.7, .71, .1)
    distance = RADIUS_KM*1000*np.deg2rad(lat)
    z = -100+np.broadcast_to(distance[:, None]*np.tan(np.deg2rad(16)), (len(lat), len(lon)))
    data = xr.DataArray(z, dims=("lat", "lon"), coords={"lat": lat, "lon": lon})
    _, _, slopes = dem_slopes(data, Cell(0, 0))
    np.testing.assert_allclose(slopes, 16, atol=1e-10)
    _, _, flat = dem_slopes(data*0-100, Cell(0, 0))
    assert (flat == 0).all()


def test_joint_spatial_class_allocation_and_no_double_exclusion():
    cell = Cell(0, 0)
    fractions = np.zeros((20, 20, 17))
    fractions[..., 1] = .5
    fractions[..., 16] = .5
    coords = np.linspace(-.5, .5, 101)
    slopes = np.broadcast_to(np.where(coords < 0, 20., 0.)[None, :], (101, 101))
    protected = shapely.box(-1, -1, 0, 1)
    result = integrate(cell, SyntheticTile(), fractions, (coords, coords, slopes), protected, 100)
    ledger = result["classes"]
    assert ledger[0, 0, 16] == pytest.approx(cell.area_km2/2)
    assert ledger[0, 3, 16] == 0
    assert ledger[1, 3, 16] == pytest.approx(cell.area_km2/4, rel=.03)
    assert result["mask_areas_km2"][3] == pytest.approx(cell.area_km2/2)
    # Slope and protection exclude the same half: not two independent halves.
    assert ledger[0, 3].sum() == pytest.approx(cell.area_km2/2)


def test_flat_below_sea_grid_has_no_exclusions():
    cell = Cell(0, 0)
    fractions = np.zeros((20, 20, 17))
    fractions[..., 16] = 1
    coords = np.array([-1., 0., 1.])
    result = integrate(cell, SyntheticTile(), fractions, (coords, coords, np.zeros((3, 3))), None, 100)
    np.testing.assert_allclose(result["mask_areas_km2"], cell.area_km2)
    np.testing.assert_allclose(result["classes"][:, 0], result["classes"][:, 3])


def test_regular_sampling_rejects_extrapolation():
    coords = np.array([0., 1., 2.])
    with pytest.raises(ValueError, match="outside DEM"):
        sample_regular(np.zeros((3, 3)), coords, coords, np.array([-1.]), np.array([0.]))
