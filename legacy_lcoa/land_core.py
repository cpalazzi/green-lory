"""Explicit geometry and paper-reference land rules, independent of legacy code."""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np

RADIUS_KM = 6371.0
# C1 encodes water at layer 0; Table 2 and Q1 label water 17.
# This mapping is confirmed against the actual HDF SDS attributes.
WIND = np.array([0, 0, 0, 0, 0, 0, .5, .5, .2, .2, .2, 0, .05, 0, .05, 0, 1.])
SOLAR = np.array([0, 0, 0, 0, 0, 0, .5, .5, .2, .2, .2, 0, 0, .03, 0, 0, 1.])


@dataclass(frozen=True)
class Cell:
    latitude: float
    longitude: float
    anchor: str = "center"
    degree: float = 1.

    @property
    def bounds(self):
        if self.anchor not in {"center", "southwest"}:
            raise ValueError("Cell anchor must be center or southwest")
        if not np.isfinite([self.latitude, self.longitude, self.degree]).all() or self.degree <= 0:
            raise ValueError("Invalid cell coordinates/size")
        offset = .5 * self.degree if self.anchor == "center" else 0.
        west, south = self.longitude - offset, self.latitude - offset
        if south < -90 or south + self.degree > 90 or not -180 <= self.longitude < 180:
            raise ValueError("Cell outside globe")
        return west, south, west + self.degree, south + self.degree

    @property
    def area_km2(self):
        west, south, east, north = self.bounds
        return float(rectangle_area(south, north, east - west))


def rectangle_area(south, north, width):
    """Spherical quadrilateral area; longitudes may cross the dateline."""
    return RADIUS_KM**2 * np.deg2rad(width) * (np.sin(np.deg2rad(north)) - np.sin(np.deg2rad(south)))


def paper_pv_density(latitude, equator_km2_per_gw=9.):
    """MW/km2: approximate paper equatorial footprint times van de Ven Eq. 6.

    Beta equals absolute latitude. Formula is invalid at/above 66.55 degrees;
    set density to zero there, explicitly, rather than inventing a polar clamp.
    The 9 km2/GW normalization is a prose-based reconstruction assumption, not
    proof of the exact historic implementation or an extra GSR multiplier.
    """
    if not np.isfinite(equator_km2_per_gw) or equator_km2_per_gw <= 0:
        raise ValueError("Positive equatorial footprint required")
    lat = np.abs(np.asarray(latitude, dtype=float))
    if not np.isfinite(lat).all() or (lat > 90).any():
        raise ValueError("Invalid latitude")
    valid = lat < 66.55
    beta = np.deg2rad(np.minimum(lat, 66.54))
    altitude = np.deg2rad(66.55 - np.minimum(lat, 66.54))
    packing = 1. / (np.cos(beta) + np.sin(beta) / np.tan(altitude))
    return np.where(valid, (1000. / equator_km2_per_gw) * packing, 0.)


def validate_fractions(percent):
    """Fail on fill/invalid values; expose integer-rounding closure residuals.

    MODIS percentages are independently rounded integers. Preserve values for
    unmasked area accounting; normalize only for conditional allocation bounds.
    """
    raw = np.asarray(percent, dtype=float)
    if raw.shape[-1] != 17 or not np.isfinite(raw).all() or ((raw < 0) | (raw > 100)).any():
        raise ValueError("Invalid MODIS percentages or fill values")
    fractions = raw / 100.
    residual = fractions.sum(axis=-1) - 1.
    if (np.abs(residual) > .05 + 1e-12).any():
        raise ValueError("MODIS class fractions do not approximately close")
    return fractions, residual


def masked_suitability(fractions, surviving_fraction, factors):
    """Conditional-uniform estimate and sharp within-pixel allocation bounds.

    C1 lacks spatial class locations inside 0.05-degree pixels. Joint exclusion
    masks give the surviving fraction q, not its class composition. Lower/upper
    weighted suitability allocate q to classes in ascending/descending factor
    order, respectively. No independence assumption is called exact geometry.
    """
    f = np.asarray(fractions, dtype=float)
    q = np.asarray(surviving_fraction, dtype=float)
    factors = np.asarray(factors, dtype=float)
    if (f < 0).any() or not np.isfinite(f).all() or (f.sum(axis=-1) <= 0).any():
        raise ValueError("Invalid class fractions")
    if not np.isfinite(q).all() or ((q < 0) | (q > 1)).any():
        raise ValueError("Invalid surviving fraction")
    if factors.shape != (f.shape[-1],) or ((factors < 0) | (factors > 1)).any() or not np.isfinite(factors).all():
        raise ValueError("Invalid suitability factors")
    f = f / f.sum(axis=-1, keepdims=True)
    estimate = (f @ factors) * q
    bounds = []
    for order in (np.argsort(factors), np.argsort(factors)[::-1]):
        remaining = np.broadcast_to(q, f.shape[:-1]).copy()
        value = np.zeros(f.shape[:-1])
        for k in order:
            allocated = np.minimum(remaining, f[..., k])
            value += allocated * factors[k]
            remaining = np.maximum(remaining - allocated, 0.)
        bounds.append(value)
    return estimate, bounds[0], bounds[1]


def read_cmg_cell(sds, cell: Cell, fine_degree=.05):
    """Read a correctly bounded CMG cell with exact per-row spherical weights."""
    west, south, east, north = cell.bounds
    indices = np.array([(90-north)/fine_degree, (90-south)/fine_degree,
                        (west+180)/fine_degree, (east+180)/fine_degree])
    if not np.allclose(indices, np.rint(indices), atol=1e-8, rtol=0):
        raise ValueError("Cell edges must align with MODIS pixels")
    y0, y1, x0, x1 = map(int, np.rint(indices))
    nx = int(round(360 / fine_degree))
    if x0 < 0:
        percent = np.concatenate([sds[y0:y1, nx+x0:nx, :], sds[y0:y1, 0:x1, :]], axis=1)
    elif x1 > nx:
        percent = np.concatenate([sds[y0:y1, x0:nx, :], sds[y0:y1, 0:x1-nx, :]], axis=1)
    else:
        percent = sds[y0:y1, x0:x1, :]
    fractions, closure = validate_fractions(percent)
    norths = 90 - np.arange(y0, y1) * fine_degree
    areas = np.broadcast_to(rectangle_area(norths-fine_degree, norths, fine_degree)[:, None], fractions.shape[:2])
    if not np.isclose(areas.sum(), cell.area_km2, atol=1e-7, rtol=1e-10):
        raise ValueError("MODIS pixels do not cover the stated cell")
    return fractions, areas, closure
