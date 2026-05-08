"""File to reduce long periods of renewable data down to its midoids, and then design an ammonia plant off it"""
from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
import geopandas as gpd
from shapely.geometry import Point, box
from shapely.errors import GEOSException

try:
    from . import data_paths
except ImportError:  # pragma: no cover - fallback for direct execution
    PACKAGE_ROOT = Path(__file__).resolve().parent
    if str(PACKAGE_ROOT) not in sys.path:
        sys.path.insert(0, str(PACKAGE_ROOT))
    import data_paths  # type: ignore


LOGGER = logging.getLogger(__name__)

COUNTRY_ASSIGNMENT_CELL_SIZE_DEG = 1.0
COUNTRY_ASSIGNMENT_AREA_CRS = 'EPSG:6933'


def _build_cell_geometries(
    locations: list[tuple[float, float]],
    cell_size_deg: float = COUNTRY_ASSIGNMENT_CELL_SIZE_DEG,
    crs: str = 'EPSG:4326',
) -> gpd.GeoDataFrame:
    """Return coarse grid-cell polygons centred on the provided coordinates."""
    if not locations:
        return gpd.GeoDataFrame(columns=['latitude', 'longitude', 'geometry'], geometry='geometry', crs=crs)

    half_size = float(cell_size_deg) / 2.0
    normalized = [(float(lat), float(lon)) for lat, lon in locations]
    geometries = [
        box(lon - half_size, lat - half_size, lon + half_size, lat + half_size)
        for lat, lon in normalized
    ]
    return gpd.GeoDataFrame(
        {'latitude': [lat for lat, _ in normalized], 'longitude': [lon for _, lon in normalized]},
        geometry=geometries,
        crs=crs,
    )


def build_country_lookup_by_cell_overlap(
    world: gpd.GeoDataFrame,
    locations: list[tuple[float, float]],
    cell_size_deg: float = COUNTRY_ASSIGNMENT_CELL_SIZE_DEG,
) -> dict[tuple[float, float], str]:
    """Assign each coarse cell to the country with the largest overlap area.

    This uses the full cell footprint rather than the cell centroid, so mixed
    coastal cells and border cells are assigned to the country covering the
    largest share of the cell's land polygon overlap.
    """
    lookup = {(float(lat), float(lon)): '' for lat, lon in locations}
    if not locations or world.empty:
        return lookup

    country_frame = world[['country', 'geometry']].dropna(subset=['geometry']).copy()
    if country_frame.empty:
        return lookup
    if country_frame.crs is None:
        country_frame = country_frame.set_crs('EPSG:4326')

    # Some country geometries can be topologically invalid and crash GEOS
    # overlay operations (TopologyException side location conflict). Repair
    # them up front with a zero-width buffer.
    if not country_frame.geometry.is_valid.all():
        country_frame['geometry'] = country_frame.geometry.buffer(0)
        country_frame = country_frame.dropna(subset=['geometry']).copy()
        country_frame = country_frame[country_frame.geometry.is_valid].copy()
        if country_frame.empty:
            return lookup

    cells = _build_cell_geometries(locations, cell_size_deg=cell_size_deg, crs=country_frame.crs)
    candidates = gpd.sjoin(cells, country_frame, how='left', predicate='intersects')
    candidates = candidates.dropna(subset=['index_right']).reset_index().rename(columns={'index': 'cell_idx'})
    if candidates.empty:
        return lookup

    countries_equal_area = country_frame.to_crs(COUNTRY_ASSIGNMENT_AREA_CRS).reset_index(drop=True)
    candidates_equal_area = gpd.GeoDataFrame(
        candidates[['cell_idx', 'latitude', 'longitude', 'country', 'index_right']].copy(),
        geometry=candidates.geometry,
        crs=cells.crs,
    ).to_crs(COUNTRY_ASSIGNMENT_AREA_CRS)

    cell_geometries = gpd.GeoSeries(candidates_equal_area.geometry.reset_index(drop=True), crs=COUNTRY_ASSIGNMENT_AREA_CRS)
    country_geometries = gpd.GeoSeries(
        countries_equal_area.geometry.iloc[candidates_equal_area['index_right'].astype(int).to_numpy()].reset_index(drop=True),
        crs=COUNTRY_ASSIGNMENT_AREA_CRS,
    )
    try:
        overlap_area = cell_geometries.intersection(country_geometries).area
    except GEOSException:
        # Retry once after repairing any residual invalid geometries.
        cell_geometries = cell_geometries.buffer(0)
        country_geometries = country_geometries.buffer(0)
        overlap_area = cell_geometries.intersection(country_geometries).area
    candidates_equal_area = candidates_equal_area.reset_index(drop=True)
    candidates_equal_area['overlap_area'] = overlap_area.to_numpy()
    candidates_equal_area = candidates_equal_area[candidates_equal_area['overlap_area'] > 0].copy()
    if candidates_equal_area.empty:
        return lookup

    best = candidates_equal_area.sort_values(['cell_idx', 'overlap_area'], ascending=[True, False])
    best = best.drop_duplicates(subset=['cell_idx'], keep='first')
    for row in best.itertuples(index=False):
        lookup[(float(row.latitude), float(row.longitude))] = str(row.country) if pd.notna(row.country) else ''
    return lookup


def country_for_cell_overlap(
    world: gpd.GeoDataFrame,
    lat: float,
    lon: float,
    cell_size_deg: float = COUNTRY_ASSIGNMENT_CELL_SIZE_DEG,
) -> str:
    """Return the country covering the largest overlap of the cell footprint."""
    key = (float(lat), float(lon))
    return build_country_lookup_by_cell_overlap(world, [key], cell_size_deg=cell_size_deg).get(key, '')


def _resolve_data_dir(path: str | None) -> Path:
    if path is None:
        return Path(os.getcwd()).resolve()
    resolved = Path(path)
    if not resolved.is_absolute():
        resolved = (Path(os.getcwd()) / resolved).resolve()
    return resolved


def _collect_resource_files(data_dir: Path, prefix: str) -> list[Path]:
    files = []
    for candidate in sorted(data_dir.glob('*.nc')):
        stem = candidate.stem.lower()
        if not stem.startswith(prefix):
            continue
        suffix = stem[len(prefix):]
        if suffix and not suffix.isdigit():
            continue
        files.append(candidate)
    return files


def _load_data_array(files: list[Path], variable: str, descriptor: str) -> tuple[xr.DataArray, xr.Dataset]:
    if not files:
        raise FileNotFoundError(f"No NetCDF files found for '{descriptor}'.")
    dataset = xr.open_mfdataset(
        [str(file) for file in files],
        combine="by_coords",
        parallel=False,
    )
    if variable not in dataset:
        dataset.close()
        raise ValueError(f"Variable '{variable}' not present in files for '{descriptor}'.")
    LOGGER.info("Loaded %s stack (%s files)", descriptor, len(files))
    return dataset[variable], dataset


class all_locations:
    """Container for NetCDF resources stitched into continuous data arrays."""

    def __init__(self, path: str | None, cache_resources: bool = True):
        self.path = _resolve_data_dir(path)
        LOGGER.debug("all_locations data_dir: %s", self.path)
        self.cache_resources = cache_resources
        self._open_handles: list[xr.Dataset] = []
        self.resources: dict[str, xr.DataArray] = {}
        self.solar = self._register_resource('solar', 'solar', 'Solar')
        self.wind = self._register_resource('wind', 'windpowers', 'Wind')
        self.solar_tracking = self._register_resource('solar_tracking', 'solartracking', 'Solar')
        self.bathymetry = xr.open_dataset(data_paths.BATHYMETRY_FILE)
        if self.cache_resources:
            self.bathymetry.load()
        self._open_handles.append(self.bathymetry)

    def _register_resource(self, key: str, prefix: str, variable: str) -> xr.DataArray:
        files = _collect_resource_files(self.path, prefix)
        data_array, dataset = _load_data_array(files, variable, prefix)
        if self.cache_resources:
            data_array = data_array.load()
        self._open_handles.append(dataset)
        self.resources[key] = data_array
        return data_array

    def in_ocean(self, latitude, longitude):
        # Returns true if a location is in the ocean
        # Returns false otherwise
        return self.bathymetry.loc[dict(latitude=latitude, longitude=longitude)].depths.values.tolist()

    def close(self):
        for ds in self._open_handles:
            try:
                ds.close()
            except Exception:  # noqa: BLE001
                LOGGER.debug("Failed to close dataset handle", exc_info=True)
        self._open_handles.clear()

    def __del__(self):  # pragma: no cover
        self.close()


class renewable_data:
    # Data stored for a specific renewable location, including cluster information

    def __init__(self, data, latitude, longitude, renewables, aggregation_variable=1, aggregation_mode=None,
                 wake_losses=0.93):
        """Initialises the data class by importing the relevant file, loading the data, and finding the location.
        Reshapes the data.
        Note that df refers to just the data for the specific location as an xarray; not the data for all locations."""
        self.longitude = longitude
        self.latitude = latitude
        self.wake_losses = 0.93
        self.get_data_from_nc(data)
        self.path = data.path
        LOGGER.debug(
            "The plant is at latitude %s and longitude %s",
            self.latitude,
            self.longitude,
        )

        # Extract the relevant profile
        self.renewables = renewables
        self.get_data_as_list()
        if aggregation_mode == 'optimal_cluster':
            self.consecutive_temporal_cluster(aggregation_variable)
        else:
            self.aggregate(aggregation_variable)
        # self.to_csv()

    def to_csv(self):
        """Sends output weather data to a csv file"""
        output_file_name = '{a}_{b}.csv'.format(a=self.latitude, b=self.longitude)
        drop_cols = [col for col in ['solar_tracking', 'Weights'] if col in self.concat.columns]
        output = self.concat.drop(columns=drop_cols)
        output.rename(columns={'solar': 's', 'wind': 'w'}, inplace=True)
        output.index.name = 't'
        output.index = ['t{a}'.format(a=i) for i in output.index]
        output.to_csv(output_file_name)

    def get_data_from_nc(self, weather_data):
        """Imports the data from the nc files and interprets them into wind and solar profiles"""
        LOGGER.debug("get_data_from_nc called for lon=%s", self.longitude)

        self.data = {}
        selection = dict(latitude=self.latitude, longitude=self.longitude)
        for resource in ['solar', 'wind', 'solar_tracking']:
            data_array = weather_data.resources.get(resource)
            if data_array is None:
                continue
            selected = data_array.sel(selection, method='nearest')
            values = selected.values
            if resource == 'wind':
                values = values * self.wake_losses
            self.data[resource] = values

        if not self.data:
            raise ValueError('No weather resources available for the requested location.')

        reference_resource = 'solar' if 'solar' in self.data else next(iter(self.data))
        self.hourly_data = pd.to_datetime(weather_data.resources[reference_resource].time.values)


    def get_data_as_list(self):
        """Extracts the data required and stores it in lists by hour"""
        df = pd.DataFrame()
        for source in self.renewables:
            try:
                edited_output = np.array(self.data[source])
                df[source] = edited_output
            except KeyError:
                df[source] = np.ones(len(self.data[self.renewables[0]]))

        self.concat = df

    def aggregate(self, aggregation_count):
        """Aggregates self.concat into blocks of fixed numbers of size aggregation_count. aggregation_count must be an integer which is a factor of 24 (i.e. 1, 2, 3, 4, 6, 12, 24)"""
        """To be corrected to work without days/clusters"""
        if self.concat.shape[0] % aggregation_count != 0:
            raise TypeError("Aggregation counter must divide evenly into the total number of data points")

        self.concat['Weights'] = np.ones(self.concat.shape[0]).tolist()
        for i in range(0, self.concat.shape[0] // aggregation_count):
            keep_index = i * aggregation_count
            for j in range(1, aggregation_count):
                drop_index = keep_index + j
                self.concat.loc[keep_index] += self.concat.loc[drop_index]
                self.concat.drop(drop_index, inplace=True)

    def consecutive_temporal_cluster(self, data_reduction_factor):
        """Reduces the data size by clustering adjacent hours until it has reduced in size by data_reduction_factor"""

        if data_reduction_factor < 1:
            raise TypeError("Data reduction factor must be greater than 1")

        self.concat['Weights'] = np.ones(self.concat.shape[0]).tolist()
        columns_to_sum = ['solar', 'wind']

        proximity = []
        for row in range(self.concat.shape[0]):
            if row < self.concat.shape[0] - 1:
                differences = sum(
                    abs(self.concat[element].iloc[row] - self.concat[element].iloc[row + 1]) for element in
                    columns_to_sum)
                proximity.append(
                    2 * differences * self.concat['Weights'].iloc[row] * self.concat['Weights'].iloc[row + 1] \
                    / (self.concat['Weights'].iloc[row] + self.concat['Weights'].iloc[row + 1]))
        proximity.append(1E6)
        self.concat['Proximity'] = proximity

        target_size = self.concat.shape[0] // data_reduction_factor
        while self.concat.shape[0] > target_size:
            keep_index = self.concat['Proximity'].idxmin()
            i_keep_index = self.concat.index.get_indexer([keep_index])[0]
            drop_index = self.concat.index.values[i_keep_index + 1]
            self.concat.loc[keep_index] += self.concat.loc[drop_index]
            self.concat.drop(drop_index, inplace=True)
            if i_keep_index + 1 < len(self.concat):
                differences = sum(
                    abs(self.concat[element].iloc[i_keep_index] / self.concat['Weights'].iloc[i_keep_index] \
                        - self.concat[element].iloc[i_keep_index + 1] / self.concat['Weights'].iloc[i_keep_index + 1]) \
                    for element in columns_to_sum)
                self.concat['Proximity'].iloc[i_keep_index] = 2 * differences * self.concat['Weights'].iloc[
                    i_keep_index] \
                                                              * self.concat['Weights'].iloc[i_keep_index + 1] \
                                                              / (self.concat['Weights'].iloc[i_keep_index] +
                                                                 self.concat['Weights'].iloc[i_keep_index + 1])
        self.concat.drop(columns=['Proximity'], inplace=True)


if __name__ == "__main__":
    data = all_locations(r'C:\Users\worc5561\OneDrive - Nexus365\Coding\Offshore_Wind_model')
    world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres')).rename(columns={'name': 'country'})
    for lat in range(44, 47):
        for lon in range(31, 37):
            try:
                # Only use the countries you're interested in...
                country = world[world.intersects(Point(lon, lat))].iloc[0].country
            except IndexError:
                country = 'None'
            if country == 'Russia':
                location = renewable_data(data, lat, lon, ['wind', 'solar', 'solar_tracking'])
                location.to_csv()
