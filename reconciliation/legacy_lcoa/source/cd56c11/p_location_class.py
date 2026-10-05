"""File to reduce long periods of renewable data down to its midoids, and then design an ammonia plant off it"""
import pandas as pd
import numpy as np
import glob
import xarray as xr
import os
import geopandas as gpd
from shapely.geometry import Point


class all_locations:
    # List of files and relevant information    
    def __init__(self, path):
        if path is None:
            self.path = os.getcwd()
        else:
            self.path = path
        self.file_list = glob.glob(self.path + r'\PowerData\*.nc')

        self.Solars = []
        self.Winds = []
        self.SolarTrackings = []
        for file in self.file_list:
            if 'SolarTracking' in file:
                self.SolarTrackings.append(xr.open_dataset(file))
            elif 'Solar' in file:
                self.Solars.append(xr.open_dataset(file))
            elif 'WindPower' in file:
                self.Winds.append(xr.open_dataset(file))
        self.bathymetry = xr.open_dataset(self.path + r'\PowerData\model_bathymetry.nc')
        self.waccs = pd.read_csv(self.path + r'\Equipment Data\WACCs.csv')

    def in_ocean(self, latitude, longitude):
        # Returns true if a location is in the ocean
        # Returns false otherwise
        return self.bathymetry.loc[dict(latitude=latitude, longitude=longitude)].depths.values.tolist()


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
        print('The plant is at latitude {latitude} and longitude {longitude}\n'.format(
            latitude=self.latitude, longitude=self.longitude))

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
        output = self.concat.drop(columns=['SolarTracking', 'Weights'])
        output.rename(columns={'Solar': 's', 'Wind': 'w'}, inplace=True)
        output.index.name = 't'
        output.index = ['t{a}'.format(a=i) for i in output.index]
        output.to_csv(output_file_name)

    def get_data_from_nc(self, weather_data):
        """Imports the data from the nc files and interprets them into wind and solar profiles"""
        self.data = {}
        if self.longitude < -60:
            ref = 0
        elif self.longitude < 60:
            ref = 1
        else:
            ref = 2

        self.data['Solar'] = weather_data.Solars[ref].loc[
            dict(latitude=self.latitude, longitude=self.longitude)].Solar.values
        self.data['Wind'] = weather_data.Winds[ref].loc[
                                dict(latitude=self.latitude, longitude=self.longitude)].Wind.values * self.wake_losses
        self.data['SolarTracking'] = weather_data.SolarTrackings[ref].loc[
            dict(latitude=self.latitude, longitude=self.longitude)].Solar.values
        self.hourly_data = pd.to_datetime(weather_data.Solars[ref].time.values)

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
        columns_to_sum = ['Solar', 'Wind']

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
                location = renewable_data(data, lat, lon, ['Wind', 'Solar', 'SolarTracking'])
                location.to_csv()
