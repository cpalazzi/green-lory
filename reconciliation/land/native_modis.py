"""Native MCD12Q1 geometry, identity checks and categorical sampling.

Native sinusoidal pixel areas are converted to the same spherical radius used
by the existing CMG comparison. No categorical interpolation is performed.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re

import numpy as np
from pyhdf.SD import SD, SDC
import shapely

from reconciliation.land.core import Cell, RADIUS_KM


def metadata_value(text, name):
    pattern = rf"(?m)^\s*OBJECT\s*=\s*{re.escape(name)}\s*\n(.*?)^\s*END_OBJECT\s*=\s*{re.escape(name)}\s*$"
    match = re.search(pattern, text.rstrip("\0"), re.DOTALL)
    if match is None:
        raise ValueError(f"Missing HDF identity field: {name}")
    value = re.search(r"(?m)^\s*VALUE\s*=\s*(.*?)\s*$", match.group(1))
    if value is None:
        raise ValueError(f"Missing HDF identity value: {name}")
    return value.group(1).strip('"')


def sinusoidal_xy(longitude, latitude, radius=6371007.181):
    lat = np.deg2rad(np.asarray(latitude, dtype=float))
    return radius * np.deg2rad(longitude) * np.cos(lat), radius * lat


def cell_polygon(cell, radius, edge_step_degrees=.002):
    west, south, east, north = cell.bounds
    if west < -180 or east > 180 or south <= -90 or north >= 90:
        raise ValueError("Native pilot requires a nonpolar, non-dateline cell")
    lat = np.linspace(south, north, max(2, int(np.ceil(cell.degree / edge_step_degrees)) + 1))
    xw, y = sinusoidal_xy(west, lat, radius)
    xe, _ = sinusoidal_xy(east, lat, radius)
    return shapely.Polygon(np.column_stack([np.r_[xw, xe[::-1]], np.r_[y, y[::-1]]]))


def class_histogram(classes, weights):
    values, area = np.broadcast_arrays(classes, weights)
    if not np.isfinite(area).all() or (area < 0).any():
        raise ValueError("Invalid pixel areas")
    present = area > 0
    if not np.isin(values[present], np.arange(1, 18)).all():
        raise ValueError("Missing or invalid native land class in study cell")
    mapped = np.where(values[present] == 17, 0, values[present])
    return np.bincount(mapped.astype(int), weights=area[present], minlength=17)


@dataclass
class NativeTile:
    path: Path
    radius: float
    left: float
    top: float
    dx: float
    dy: float
    layers: dict

    @classmethod
    def read(cls, path, *, year=2022):
        path = Path(path)
        name = re.fullmatch(rf"MCD12Q1\.A{year}001\.h(\d{{2}})v(\d{{2}})\.061\.\d{{13}}\.hdf", path.name)
        if name is None:
            raise ValueError("Unexpected native MODIS filename/year/collection")
        sd = SD(str(path), SDC.READ)
        try:
            attrs = sd.attributes()
            core = attrs["CoreMetadata.0"]
            if (metadata_value(core, "LOCALGRANULEID") != path.name or
                    metadata_value(core, "SHORTNAME") != "MCD12Q1" or
                    int(metadata_value(core, "VERSIONID")) != 61 or
                    metadata_value(core, "RANGEBEGINNINGDATE") != f"{year}-01-01"):
                raise ValueError("HDF identity differs from filename/expected year")
            struct = attrs["StructMetadata.0"].rstrip("\0")
            if "Projection=GCTP_SNSOID" not in struct or "GridOrigin=HDFE_GD_UL" not in struct:
                raise ValueError("Unsupported MODIS projection or grid origin")
            def pair(key):
                match = re.search(rf"{key}=\(([^)]+)\)", struct)
                if match is None:
                    raise ValueError(f"Missing geometry: {key}")
                return tuple(map(float, match.group(1).split(",")))
            left, top = pair("UpperLeftPointMtrs")
            right, bottom = pair("LowerRightMtrs")
            params = pair("ProjParams")
            radius = params[0]
            if not np.isclose(radius, 6371007.181) or any(v != 0 for v in params[1:]):
                raise ValueError("Unexpected sinusoidal projection parameters")
            layers = {}
            for key, limits in (("LC_Type1", (1, 17)), ("QC", (0, 10)), ("LW", (1, 2))):
                sds = sd.select(key)
                data = sds[:]
                if data.shape != (2400, 2400) or data.dtype != np.uint8:
                    raise ValueError(f"Unexpected native layer shape/type: {key}")
                if tuple(sds.attributes()["valid_range"]) != limits or sds.attributes()["_FillValue"] != 255:
                    raise ValueError(f"Unexpected native layer metadata: {key}")
                if not (((data >= limits[0]) & (data <= limits[1])) | (data == 255)).all():
                    raise ValueError(f"Invalid native layer values: {key}")
                layers[key] = data
            h, v = map(int, name.groups())
            width = radius * np.deg2rad(10)
            np.testing.assert_allclose([left, top, right, bottom],
                [(h-18)*width, (9-v)*width, (h-17)*width, (8-v)*width], atol=.002, rtol=0)
            return cls(path, radius, left, top, (right-left)/2400, (top-bottom)/2400, layers)
        finally:
            sd.end()

    def sample(self, longitude, latitude):
        x, y = sinusoidal_xy(longitude, latitude, self.radius)
        row, col = np.broadcast_arrays(np.floor((self.top-y)/self.dy).astype(int),
                                       np.floor((x-self.left)/self.dx).astype(int))
        ny, nx = self.layers["LC_Type1"].shape
        if (row < 0).any() or (row >= ny).any() or (col < 0).any() or (col >= nx).any():
            raise ValueError("Native tile does not cover sample coordinates")
        return {key: data[row, col] for key, data in self.layers.items()}

    def unmasked_ledger(self, cell, edge_step_degrees=.002):
        polygon = cell_polygon(cell, self.radius, edge_step_degrees)
        ny, nx = self.layers["LC_Type1"].shape
        tile_box = shapely.box(self.left, self.top-ny*self.dy, self.left+nx*self.dx, self.top)
        if not shapely.covers(tile_box, polygon):
            raise ValueError("Native tile does not cover the entire study cell")
        xmin, ymin, xmax, ymax = polygon.bounds
        c0, c1 = int(np.floor((xmin-self.left)/self.dx)), int(np.ceil((xmax-self.left)/self.dx))
        r0, r1 = int(np.floor((self.top-ymax)/self.dy)), int(np.ceil((self.top-ymin)/self.dy))
        cols, rows = np.meshgrid(np.arange(c0, c1), np.arange(r0, r1))
        xl = self.left + cols*self.dx
        yt = self.top - rows*self.dy
        pixels = shapely.box(xl, yt-self.dy, xl+self.dx, yt)
        shapely.prepare(polygon)
        inside = shapely.contains_properly(polygon, pixels)
        touches = ~inside & shapely.intersects(polygon, pixels)
        areas = np.where(inside, self.dx*self.dy, 0.)
        areas[touches] = shapely.area(shapely.intersection(pixels[touches], polygon))
        # Equal-area sinusoidal geometry, normalized to the existing comparison sphere.
        areas *= (RADIUS_KM*1000/self.radius)**2 / 1e6
        if not np.isclose(areas.sum(), cell.area_km2, atol=1e-5, rtol=1e-9):
            raise ValueError("Native pixel intersections do not conserve study-cell area")
        values = self.layers["LC_Type1"][r0:r1, c0:c1]
        hist = class_histogram(values, areas)
        qc = self.layers["QC"][r0:r1, c0:c1]
        lw = self.layers["LW"][r0:r1, c0:c1]
        return {"class_areas_km2": hist,
                "qc_areas_km2": np.bincount(qc.ravel(), weights=areas.ravel(), minlength=256),
                "land_water_mismatch_km2": float(areas[((values == 17) & (lw == 2)) | ((values != 17) & (lw == 1))].sum()),
                "cell_area_error_km2": float(areas.sum()-cell.area_km2)}
