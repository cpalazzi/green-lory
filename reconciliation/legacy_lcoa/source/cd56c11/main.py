import pypsa
import p_auxiliary as aux
import glob
import p_data_store as pds
import pandas as pd
import p_location_class as plc
import geopandas as gpd
from shapely.geometry import Point
import os

"""File to optimise the size of a green ammonia plant given a specified wind and solar profile"""


def generate_network(n_snapshots, data_file, aggregation_count=1, costs=None, efficiencies=None, time_step=0.5):
    """Generates a network that can be used to run several cases"""
    # ==================================================================================================================
    # Set up network
    # ==================================================================================================================

    # Import a generic network
    network = pypsa.Network(override_component_attrs=aux.create_override_components())

    # Set the time values for the network
    network.set_snapshots(range(int(n_snapshots/aggregation_count)))

    # Import the design of the H2 plant into the network
    network.import_from_csv_folder(data_file)

    if costs is not None:
        for equipment, row in costs.items():
            for df in [network.links, network.generators, network.stores]:
                if equipment in df.index:
                    df.loc[equipment, 'capital_cost'] = row

    if efficiencies is not None:
        for equipment, row in efficiencies.items():
            if equipment in network.links.index:
                network.links.loc[equipment, 'efficiency'] = row

    network.links.loc['HydrogenCompression', 'marginal_cost'] = 0.0001  # Just stops pointless cycling through storage

    water_cost = (39.4/network.links.loc['Electrolysis', 'efficiency'])**(-1) * 10 * 2/0.7  #efficiency/39.4 = t H2/MW
    # 2 USD/kL for water (2/0.7 = AUD); factor of 10 because ~10 t H20/t H2;

    network.links.loc['Electrolysis', 'marginal_cost'] = water_cost

    # Adjust the capital cost of the stores, and the marginal costs, based on the aggregation and number of datapoints
    network.stores.capital_cost *= time_step
    network.links.marginal_cost *= 24*366 / n_snapshots
    # Just in case you have a default that is different to 1 hour in the data
    if aggregation_count is not None:
        network.stores.capital_cost *= aggregation_count
        network.links.marginal_cost *= aggregation_count

    return network


def main(n=None, file_name=None, weather_data=None, multi_site=False, get_complete_output=False,
         extension="", aggregation_count=1, time_step=1.0):
    """Code to execute run at a single location"""
    # Import the weather data
    if file_name is not None and weather_data is None:
        weather_data = aux.get_weather_data(file_name=file_name, aggregation_count=aggregation_count)

    # Import a generic network if needed
    if n is None:
        n = generate_network(len(weather_data), "Basic_ammonia_plant")

    renewables_lst = n.generators.index.to_list()
    # Note: All flows are in MW or MWh, conversions for hydrogen done using HHVs. Hydrogen HHV = 39.4 MWh/t

    # ==================================================================================================================
    # Send the weather data to the model
    # ==================================================================================================================

    count = 0
    for name, dataframe in weather_data.items():
        if name in renewables_lst:
            n.generators_t.p_max_pu[name] = dataframe
            count += 1
        else:
            print(name)
            raise ValueError("You have a columns {a} that isn't in your list of renewables, so I don't know what those "
                             " renewables cost. \n "
                             "Edit generators.csv to include all of the renewables for which you've "
                             " provided data".format(a=name))
    if count != len(renewables_lst):
        raise ValueError("You have renewables for which you haven't provided a weather dataset. "
                         "\n Edit the input file to include all relevant renewables sources, "
                         "or remove the renewables sources from generators.csv")

    # ==================================================================================================================
    # Check if the CAPEX input format in Basic_H2_plant is correct, and fix it up if not
    # ==================================================================================================================

    if not multi_site:
        CAPEX_check = aux.check_CAPEX(file_name=file_name)
        if CAPEX_check is not None:
            for item in [n.generators, n.links, n.stores]:
                item.capital_cost = item.capital_cost * CAPEX_check[1] / 100 + item.capital_cost / CAPEX_check[0]

    # ==================================================================================================================
    # Solve the model
    # ==================================================================================================================

    # Ask the user how they would like to solve, unless they're doing several cases
    if not multi_site:
        solver, _ = aux.get_solving_info(file_name=file_name)
    else:
        solver = 'gurobi'

    # Implement their answer
    n.lopf(solver_name=solver, pyomo=True, extra_functionality=aux.pyomo_constraints)

    # ==================================================================================================================
    # Output results
    # ==================================================================================================================
    if not get_complete_output and not multi_site:
        multi_site = True

    if get_complete_output:
        # Scale if needed
        scale = aux.get_scale(n, file_name=file_name)

        # Put the results in a nice format
        output = aux.get_results_dict_for_excel(n, scale, aggregation_count=aggregation_count, time_step=time_step)

        # Send the results to excel
        aux.write_results_to_excel(output, file_name=file_name[5:], extension=extension)

    if multi_site:
        output = aux.get_results_dict_for_multi_site(n, aggregation_count=aggregation_count, time_step=time_step)

    return output


def run_Alli_sites(year):
    # Download any necessary data
    excel_file = r'Data\GeneralSteelData.xlsx'
    time_step = 4
    costs = pd.read_excel(excel_file, sheet_name='Costs').set_index('Equipment')[year]
    efficiencies = pd.read_excel(excel_file, sheet_name='Efficiencies').set_index('Equipment')[year]

    # Create a network; override the defaults with the relevant cost and efficiency data
    n = generate_network(8760/time_step, 'Basic_ammonia_plant',
                         costs=costs,
                         efficiencies=efficiencies,
                         aggregation_count=time_step)

    # Get a list of the renewables
    renewables = n.generators.index.to_list()

    # Create a place to store the data between runs
    store = pds.Data_store()

    # Get the weather data
    data = plc.all_locations(r'C:\Users\worc5561\OneDrive - Nexus365\Coding\Offshore_Wind_model')

    # Get a map of the world to identify which country the weather data is in
    world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres')).rename(columns={'name': 'country'})

    # List hte countries you're interested in
    # country_lst = ['Ukraine']
    # country_lst = ['Russia']
    #country_lst = pd.read_csv(r'C:\Users\worc5561\OneDrive - Nexus365\Coding\Offshore_Wind_model\Country_lst.csv')
    #country_lst = country_lst.Country.to_list()
    parent_directory = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
    csv_file = parent_directory + r'\LCOH_Optimisation\Data\Extra_locs_2.csv'
    loc_lst = pd.read_csv(csv_file).set_index('ID')

    # Now run the code around the world
    for lat in range(43, 53):
        for lon in range(22, 41):
            # try:
            #     # Only use the countries you're interested in...
            #     country = world[world.intersects(Point(lon, lat))].iloc[0].country
            # except IndexError:
            #     country = 'None'
            #     if country in country_lst:
            ID = '{a}_{b}'.format(a=lat, b=lon)
            if ID in loc_lst.index:
                    # Get the weather data in the target location
                location = plc.renewable_data(data, lat, lon, renewables)

                # Run the code
                result = main(n=n, weather_data=location.concat.drop(columns='Weights'),
                              multi_site=True, aggregation_count=time_step)
                # Add the output to the data store.
                store.add_location(lat, lon, loc_lst.loc[ID, 'Code'], result)
        # # Output the data periodically just in case there's a power outage or similar...
        # if lat % 20 == 0:
        #     df = pd.DataFrame.from_dict(store.collated_results, orient='index')
        #     df.to_csv('{a}_lat_{b}.csv'.format(a=year, b=lat))
    # Output all the data at the end
    df = pd.DataFrame.from_dict(store.collated_results, orient='index')
    df.to_csv('{a}_Extra_2_NH3.csv'.format(a=year))


def run_QLD_sites(year, case):

    # Get the cost data
    scenario = '{a} - {b}'.format(a=year, b=case)
    parent_directory = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
    excel_file = parent_directory + r'\LCOA_Optimisation\Data\GeneralQLDData.xlsx'
    costs = pd.read_excel(excel_file,
                          sheet_name='Costs').set_index('Equipment')[scenario.split('_Cycle')[0]]
    efficiencies = pd.read_excel(excel_file,
                                 sheet_name='Efficiencies').set_index('Equipment')[scenario.split('_Cycle')[0]]

    # Get the location data
    file_lst = glob.glob(parent_directory + r'\LCOH_Optimisation\WeatherData/*.csv')
    length = len(pd.read_csv(file_lst[0]))
    

    # Set up the network design
    time_step = 0.5
    aggregation_count = int(8)       
    # network_design = generate_network(length,
    #                                     'Dummy_QLD',
    #                                     costs=costs,
    #                                     efficiencies=efficiencies,
    #                                     aggregation_count=aggregation_count,
    #                                     time_step=time_step)

    # Get the site information
    distances = pd.read_csv(parent_directory + r'\LCOH_Optimisation\a_Distances.csv'). \
        set_index('Location')

    # Set up the data store
    store = pds.Data_store()
    operating_store = pds.Data_store()

    for file in file_lst:
        network_design = generate_network(length,
                                            'Dummy_QLD',
                                            costs=costs,
                                            efficiencies=efficiencies,
                                            aggregation_count=aggregation_count,
                                            time_step=time_step)
        result = main(n=network_design,
                      file_name=file,
                      get_complete_output=True,
                      multi_site=True,
                      extension='{a}'.format(a=scenario),
                      aggregation_count=aggregation_count,
                      time_step=time_step)
        location = file.split('\\')[-1][:-4]
        try:
            lat = distances.loc[location, 'Latitude']
            lon = distances.loc[location, 'Longitude']
        except KeyError:
            lat, lon = 0, 0
        store.add_location(lat, lon, location, result)
        operating = aux.convert_network_to_operating(network_design,
                                          ammonia_cost_per_ton=500,
                                          aggregation_count=aggregation_count,
                                          file_name='{a}_{b}'.format(a=location, b=scenario),
                                          multi_site=False,
                                          time_step=time_step)
        operating_store.add_location(lat, lon, location, operating)
    df = pd.DataFrame.from_dict(store.collated_results, orient='index')
    df.to_csv('{a}.csv'.format(a=scenario))
    df = pd.DataFrame.from_dict(operating_store.collated_results, orient='index')
    df.to_csv('{a}_operating.csv'.format(a=scenario))


if __name__ == '__main__':
    for year in [2030, 2035, 2040, 2045, 2050]:
        run_Alli_sites(year)
    # for year in [2030, 2040, 2050]:
    #     for case in ['Ammonia_Fix_H_Salt_Cavern_Cycle_4']:
        # for case in ['Salt Cavern_Cycle_4', 'Salt Cavern_Cycle_12', 'Salt Cavern_Cycle_24']:
        #     run_QLD_sites(year, case)
