import logging

import numpy as np
import pandas as pd
import pyomo.environ as pm

from constants import HYDROGEN_HHV_MWH_PER_T, AMMONIA_HHV_MWH_PER_T


def _link_capacity_on_output_basis(network, series):
    """Convert a link capacity series from input (bus0) to output (bus1) basis.

    In PyPSA, Link p_nom/p_nom_opt is defined on bus0 (MW_in). For reporting we
    prefer a bus1 output-basis capacity: MW_out = MW_in * efficiency.
    """
    links = network.links
    if links.empty or series.empty:
        return series

    eff_col = "output_basis_efficiency" if "output_basis_efficiency" in links.columns else "efficiency"
    if eff_col not in links.columns:
        return series

    efficiencies = links[eff_col].reindex(series.index).astype(float).fillna(1.0)
    converted = series.astype(float).copy()
    return converted * efficiencies


COST_COMPONENT_SPECS = {
    "wind": ("generator", "wind"),
    "solar": ("generator", "solar"),
    "solar_tracking": ("generator", "solar_tracking"),
    "electrolyser": ("link", "electrolysis"),
    "ammonia_synthesis": ("link", "ammonia_synthesis"),
    "hydrogen_compression": ("link", "hydrogen_compression"),
    "hydrogen_store": ("store", "compressed_hydrogen_store"),
    "ammonia_store": ("store", "ammonia"),
    "battery": ("store", "battery"),
}

INTEREST_RATE_COLUMN_ALIASES = {
    "electrolysis": "electrolyser",
    "compressed_hydrogen_store": "hydrogen_store",
    "ammonia": "ammonia_store",
}


def _snapshot_weightings(network):
    weightings = getattr(network.snapshot_weightings, "generators", None)
    if weightings is None:
        return pd.Series(1.0, index=network.snapshots)
    if isinstance(weightings, pd.DataFrame):
        weightings = weightings.get("generators")
    weightings = pd.Series(weightings, index=network.snapshots)
    return weightings.reindex(network.snapshots).ffill().bfill().fillna(1.0)


def _component_table(network, component_type):
    if component_type == "generator":
        return network.generators
    if component_type == "link":
        return network.links
    if component_type == "store":
        return network.stores
    raise ValueError(f"Unknown component type '{component_type}'")


def _component_capacity(network, component_type, name):
    table = _component_table(network, component_type)
    if name not in table.index:
        return 0.0
    column = "p_nom_opt" if component_type in {"generator", "link"} else "e_nom_opt"
    value = table.at[name, column]
    if pd.isna(value):
        return 0.0
    return float(value)


def _component_capex_cost(network, component_type, name):
    table = _component_table(network, component_type)
    if name not in table.index:
        return 0.0
    capital_cost = table.at[name, "capital_cost"]
    capacity = _component_capacity(network, component_type, name)
    if pd.isna(capital_cost) or capacity <= 0:
        return 0.0
    return float(capital_cost) * capacity


def _component_dispatch_cost(network, component_type, name, weightings):
    if component_type == "generator":
        table = network.generators
        dispatch_df = getattr(network.generators_t, "p", None)
    elif component_type == "link":
        table = network.links
        dispatch_df = getattr(network.links_t, "p0", None)
    else:
        table = network.stores
        dispatch_df = None

    if name not in table.index:
        return 0.0

    marginal_cost = table.at[name, "marginal_cost"] if "marginal_cost" in table.columns else 0.0
    if pd.isna(marginal_cost) or marginal_cost == 0 or dispatch_df is None or name not in dispatch_df:
        return 0.0

    dispatch = dispatch_df[name].reindex(weightings.index, fill_value=0.0)
    return float((dispatch * weightings).sum() * float(marginal_cost))


def _total_capex(network):
    total = 0.0
    if not network.generators.empty:
        total += float((network.generators["capital_cost"].fillna(0.0) * network.generators["p_nom_opt"].fillna(0.0)).sum())
    if not network.links.empty:
        total += float((network.links["capital_cost"].fillna(0.0) * network.links["p_nom_opt"].fillna(0.0)).sum())
    if not network.stores.empty:
        total += float((network.stores["capital_cost"].fillna(0.0) * network.stores["e_nom_opt"].fillna(0.0)).sum())
    return total


def _component_cost_breakdown(network):
    weightings = _snapshot_weightings(network)
    total_cost = float(network.objective)
    components = {}
    tracked_total = 0.0
    total_capital = _total_capex(network)

    for label, (component_type, name) in COST_COMPONENT_SPECS.items():
        capex = _component_capex_cost(network, component_type, name)
        opex = _component_dispatch_cost(network, component_type, name, weightings)
        total = capex + opex
        tracked_total += total
        components[label] = {
            "capex": capex,
            "opex": opex,
            "total": total,
        }

    other_cost = total_cost - tracked_total
    other_capex = total_capital - sum(entry["capex"] for entry in components.values())
    return components, other_cost, total_capital, total_cost


def _weighted_interest_rate(network, interest_rates):
    if not interest_rates:
        return None

    # Prefer a capex-weighted rate because interest rates impact the annualised
    # capital-cost terms in the objective.
    total_capex_weight = 0.0
    weighted_sum = 0.0

    for tech, rate in interest_rates.items():
        if rate is None:
            continue
        for component_type in ("generator", "link", "store"):
            table = _component_table(network, component_type)
            if tech not in table.index:
                continue
            capacity = _component_capacity(network, component_type, tech)
            if capacity <= 0:
                break
            capital_cost = table.loc[tech].get("capital_cost", 0.0)
            try:
                capital_cost = float(capital_cost)
            except Exception:  # noqa: BLE001
                capital_cost = 0.0
            if not np.isfinite(capital_cost):
                capital_cost = 0.0
            weight = float(capital_cost) * float(capacity)
            if weight > 0:
                total_capex_weight += weight
                weighted_sum += weight * float(rate)
            break

    if total_capex_weight > 0:
        return weighted_sum / total_capex_weight

    # Fallback: capacity-weighted average, then simple average.
    total_capacity = 0.0
    weighted_sum = 0.0

    for tech, rate in interest_rates.items():
        if rate is None:
            continue
        for component_type in ("generator", "link", "store"):
            table = _component_table(network, component_type)
            if tech in table.index:
                capacity = _component_capacity(network, component_type, tech)
                if capacity > 0:
                    total_capacity += capacity
                    weighted_sum += capacity * float(rate)
                break

    if total_capacity > 0:
        return weighted_sum / total_capacity

    valid_rates = [float(rate) for rate in interest_rates.values() if rate is not None]
    if not valid_rates:
        return None
    return sum(valid_rates) / len(valid_rates)


def get_col_widths(dataframe):
    # First we find the maximum length of the index column
    idx_max = max([len(str(s)) for s in dataframe.index.values] + [len(str(dataframe.index.name))])
    # Then, we concatenate this to the max of the lengths of column name and its values for each column, left to right
    return [idx_max] + [max([len(str(s)) for s in dataframe[col].values] + [len(col)]) for col in dataframe.columns]


def get_weather_data(file_name=None, aggregation_count=None):
    """Asks the user where the weather data is, and pulls it in. Keeps asking until it gets a file.
    If a file_name has already been provided this code just imports the data. """
    # import data
    if file_name is None:
        input_check = True
        while input_check:
            try:
                input_check = False
                file = input("What is the name of your weather data file? "
                             "It must be a CSV, but don't include the file extension >> ")
                weather_data = pd.read_csv(file + '.csv')
                weather_data.drop(weather_data.columns[0], axis=1, inplace=True)
            except FileNotFoundError:
                input_check = True
                print("There's no input file there! Try again.")

        # Check the weather data is a year long
        if len(weather_data) < 8700 or len(weather_data) > 8760 + 48:
            logging.warning('Your weather data seems not to be one year long in hourly intervals. \n'
                            'Are you sure the input data is correct?'
                            ' If not, exit the code using ctrl+c and start again.')

    else:
        weather_data = pd.read_csv(file_name)
        weather_data.drop(weather_data.columns[0], axis=1, inplace=True)

    # Just tidy up the data if it needs it...
    if 'grid' not in weather_data.columns:
        weather_data['grid'] = np.zeros(len(weather_data))
    if 'ramp_dummy' not in weather_data.columns:
        weather_data['ramp_dummy'] = np.ones(len(weather_data))

    if aggregation_count is not None:
        print('Aggregating weather data...')
        weather_data = aggregate_data(weather_data, aggregation_count)
        return weather_data
    else:
        return weather_data


def aggregate_data(data, aggregation_count):
    """Aggregates self.concat into blocks of fixed numbers of size aggregation_count.
    aggregation_count must be an integer which is a factor of 24 (i.e. 1, 2, 3, 4, 6, 12, 24)"""
    if len(data) % aggregation_count != 0:
        raise TypeError("Aggregation counter must divide evenly into the total number of data points")

    df = pd.Series(range(len(data) // aggregation_count)).to_frame('snapshot').set_index('snapshot')

    for column in data.columns:
        if np.average(data[column]) > 0:
            df[column] = [np.average(data[column].to_list()[i * aggregation_count:(i + 1) * aggregation_count])
                          for i in df.index]
        else:
            df[column] = np.zeros(len(df))
    return df


def check_CAPEX(file_name=None):
    """Checks if the user has put the CAPEX into annualised format. If not, it helps them do so.
    file_name is the weather data file - if it is not specified the user is asked.
    Otherwise, this function does nothing."""
    if file_name is None:
        check = input('Are your capital costs in the generators.csv, '
                      'components.csv and stores.csv files annualised?'
                      '\n (i.e. have you converted them from their upfront capital cost'
                      ' to the cost that accrues each year under the chosen financial conditions? \n'
                      '(Y/N) >> ')
    else:
        check = 'Y'
    if check != 'Y':
        print('You have selected no annualisation, which means you have entered the upfront capital cost'
              ' of the equipment. \n We have to ask you a few questions to convert these to annualised costs.')
        check2 = True
        while check2:
            try:
                discount = float(input('Enter the weighted average cost of capital in percent (i.e. 7 not 0.07)'))
                years = float(input('Enter the plant operating years.'))
                O_and_M = float(input('Enter the fixed O & M costs as a percentage of installed CAPEX '
                                      '(i.e. 2 not 0.02)'))
                check2 = False
            except ValueError:
                logging.warning('You have to enter a number! Try again.')
                check2 = True

        crf = discount * (1 + discount) ** years / ((1 + discount) ** years - 1)
        if crf < 2 or crf > 20 or O_and_M < 0 or O_and_M > 8:
            print('Your financial parameter inputs are giving some strange results. \n'
                  'You might want to exit the code using ctrl + c and try re-entering them.')

        return crf, O_and_M
    else:
        if file_name is None:
            print('You have selected the annualised capital cost entry. \n'
                  'Make sure that the annualised capital cost data includes any operating costs that you '
                  'estimate based on plant CAPEX.')
        return None


def get_solving_info(file_name=None):
    """Prompts the user for information about the solver and the problem formulation.
    If no file_name is provided then the model will autoselect gurobi and pyomo."""
    if file_name is None:
        solver = input('What solver would you like to use? '
                       'If you leave this blank, the glpk default will be used >> ')
        if solver == '':
            solver = 'glpk'

        formulator = 'p'
        print('In this code, the only option for solving is to use pyomo,'
              ' because additional constraints have been turned on. ')
    else:
        solver = 'gurobi'
        formulator = 'p'
    return solver, formulator


def get_scale(n, file_name=None):
    """Gives the user some information about the solution, and asks if they'd like it to be scaled.
    If the file_name is prespecified the scale is automatically 1 (i.e. no scaling)"""
    if file_name is None:
        print('\nThe unscaled generation capacities are:')
        print(n.generators.rename(columns={'p_nom_opt': 'Rated Capacity (MW)'})[['Rated Capacity (MW)']])
        print('The unscaled hydrogen production is {a} t/year\n'.format(
            a=n.loads.p_set.values[0] / HYDROGEN_HHV_MWH_PER_T * 8760
        ))
        scale = input('Enter a scaling factor for the results, to adjust the production. \n'
                      "If you don't want to scale the results, enter a value of 1 >> ")
        try:
            scale = float(scale)
        except ValueError:
            scale = 1
            print("You didn't enter a number! The results won't be scaled.")
        return scale
    else:
        return 1


def get_results_dict_for_excel(n, scale, aggregation_count=1, operating=False, time_step=1.0):
    """Takes the results and puts them in a dictionary ready to be sent to Excel"""
    # Rename the components:
    links_name_dct = {'p_nom_opt': 'Rated Capacity (MW)',
                      'carrier': 'Carrier',
                      'bus0': 'Primary Energy Source',
                      'bus2': 'Secondary Energy Source'}
    comps = n.links.rename(columns=links_name_dct)[[i for i in links_name_dct.values()]]
    comps["Rated Capacity (MW)"] = _link_capacity_on_output_basis(
        n, comps["Rated Capacity (MW)"]
    ) * scale

    # Get the energy flows
    primary = n.links_t.p0 * scale
    secondary = (n.links_t.p2 * scale).drop(columns=['hydrogen_from_storage', 'electrolysis', 'battery_interface_in',
                                                     'battery_interface_out', 'hydrogen_fuel_cell'])

    # Rescale the energy flows (I know there's hard coding here but these numbers should never change!):
    primary['hydrogen_compression'] /= HYDROGEN_HHV_MWH_PER_T
    primary['hydrogen_from_storage'] /= HYDROGEN_HHV_MWH_PER_T
    primary['hydrogen_fuel_cell'] *= n.links.loc['hydrogen_fuel_cell'].efficiency
    secondary['ammonia_synthesis'] /= HYDROGEN_HHV_MWH_PER_T
    primary['Ammonia production (t/h)'] = secondary['ammonia_synthesis'] / 0.18
    
    # Rename the energy flows so that the units are comprehensible
    primary.rename(columns={
        'electrolysis': 'Electrolysis (MW)',
        'hydrogen_compression': 'Hydrogen to storage (t/h)',
        'hydrogen_from_storage': 'Hydrogen from storage (t/h)',
        'battery_interface_in': 'Battery Charge (MW)',
        'battery_interface_out': 'Battery Discharge (MW)',
        'hydrogen_fuel_cell': 'Power from Fuel cell (MW)',
        'ammonia_synthesis': 'Ammonia synthesis power consumption (MW)'
    }, inplace=True)
    secondary.rename(columns={'hydrogen_compression': 'H2 Compression Power Consumption (MW)',
                              'ammonia_synthesis': 'Ammonia synthesis hydrogen consumption (t/h)'}, inplace=True)

    consumption = pd.merge(primary, secondary, left_index=True, right_index=True)

    # # Just move the penalty link column to the end...
    # cols = list(consumption.columns)
    # cols.append(cols.pop(cols.index('penalty_link')))
    # consumption = consumption.reindex(columns=cols)

    output = {
        'Headlines': pd.DataFrame({
            'Objective function (USD/t)': [
                float(n.objective) / (n.loads.p_set.values[0] / AMMONIA_HHV_MWH_PER_T * 8760)
            ],
            'Production (t/year)': n.loads.p_set.values[0] / AMMONIA_HHV_MWH_PER_T * 8760 * scale
        }, index=['LCOA (USD/t)']),
        'Generators': n.generators.rename(columns={'p_nom_opt': 'Rated Capacity (MW)'})[
                          ['Rated Capacity (MW)']] * scale,
        'Components': comps,
        'Stores': scale * aggregation_count * time_step * n.stores.rename(columns={
                                        'e_nom_opt': 'Storage Capacity (MWh)'})[['Storage Capacity (MWh)']],
        'Energy generation (MW)': n.generators_t.p * scale,
        'Energy consumption': consumption,
        'Stored energy capacity (MWh)': n.stores_t.e * scale * aggregation_count * time_step
    }
    print('get_results_dict_for_excel n.objective: ', n.objective)


    if operating:
        years = len(n.stores_t.e['ammonia'])/8760
        objective = n.stores_t.e['ammonia'].iloc[-1] / AMMONIA_HHV_MWH_PER_T * 1E-6 / years
        output['Headlines'] = pd.DataFrame({
            'Annual Production (t/year)': [objective]}, index=['Production (MMTPA)'])
    return output


def write_results_to_excel(output, file_name="", extension=""):
    """Takes results dictionary and puts them in an Excel file. User determines the file name"""
    if file_name is None:
        incomplete = True
        while incomplete:
            output_file = input("Enter the name of your output data file. \n"
                                "Don't include the file extension. >> ") + '.xlsx'
            try:
                incomplete = False
                with pd.ExcelWriter(output_file, engine='xlsxwriter') as writer:
                    for key in output.keys():
                        dataframe = output[key]
                        dataframe.to_excel(writer, sheet_name=key)
                        worksheet = writer.sheets[key]
                        for i, width in enumerate(get_col_widths(dataframe)):
                            worksheet.set_column(i, i, width)
            except PermissionError:
                incomplete = True
            print('There is a problem writing on that file. Try another excel file name.')
    else:
        output_file = r'Results/' + file_name.split('\\')[-1][:-4] + extension + '.xlsx'
        with pd.ExcelWriter(output_file, engine='xlsxwriter') as writer:
            for key in output.keys():
                dataframe = output[key]
                dataframe.to_excel(writer, sheet_name=key)
                worksheet = writer.sheets[key]
                for i, width in enumerate(get_col_widths(dataframe)):
                    worksheet.set_column(i, i, width)


def get_results_dict_for_multi_site(
    n,
    aggregation_count=1,
    operating=False,
    time_step=1.0,
    interest_rates=None,
):
    """Just a simpler function that only gets the headline information, and nothing to do with times"""
    dct = dict()

    if not operating:
        load = float(n.loads.p_set.values[0])
        production = load / AMMONIA_HHV_MWH_PER_T * 8760
        demand_mwh = load * 8760
        total_cost = float(n.objective)
        dct['lcoa_usd_per_t'] = total_cost / production if production > 0 else np.nan
        dct['annual_ammonia_demand_mwh'] = demand_mwh
        dct['annual_ammonia_production_t'] = production
        dct['total_cost_usd_per_year'] = total_cost

        component_costs, other_cost, total_capital, cost_denominator = _component_cost_breakdown(n)
        if cost_denominator == 0:
            cost_denominator = total_cost
        other_cost = max(other_cost, 0.0)
        for label, values in component_costs.items():
            dct[f'cost_share_{label}_pct'] = (
                values['total'] / cost_denominator * 100 if cost_denominator else np.nan
            )
            dct[f'lcoa_component_{label}_usd_per_t'] = (
                values['total'] / production if production > 0 else np.nan
            )

        residual_share = other_cost / cost_denominator * 100 if cost_denominator else np.nan
        dct['cost_share_other_pct'] = residual_share
        dct['lcoa_component_other_usd_per_t'] = (
            other_cost / production if production > 0 else np.nan
        )
        dct['capital_cost_share_pct'] = (
            total_capital / cost_denominator * 100 if cost_denominator else np.nan
        )
        dct['lcoa_component_capital_usd_per_t'] = (
            total_capital / production if production > 0 else np.nan
        )
    else:
        years = len(n.stores_t.e['ammonia']) / 8784
        dct['lcoa_usd_per_t'] = np.nan
        dct['annual_ammonia_production_t'] = (
            n.stores_t.e['ammonia'].iloc[-1] / AMMONIA_HHV_MWH_PER_T * 1e-6 / years
        )

    if interest_rates:
        avg_rate = _weighted_interest_rate(n, interest_rates)
        if avg_rate is not None:
            dct['interest_rate_overall'] = avg_rate
        for tech, rate in interest_rates.items():
            if rate is None:
                continue
            column_key = INTEREST_RATE_COLUMN_ALIASES.get(tech, tech)
            dct[f'interest_rate_{column_key}'] = float(rate)

    for generator in n.generators.index.to_list():
        dct[generator] = n.generators.loc[generator, 'p_nom_opt']
    link_capacities = _link_capacity_on_output_basis(n, n.links['p_nom_opt'])
    for component in n.links.index.to_list():
        dct[component] = link_capacities.loc[component]
    for store in n.stores.index.to_list():
        dct[store] = n.stores.loc[store, 'e_nom_opt'] * aggregation_count * time_step
    dct['hydrogen_storage_capacity'] = (
        n.stores.loc['compressed_hydrogen_store', 'e_nom_opt'] * aggregation_count * time_step / HYDROGEN_HHV_MWH_PER_T
    )
    return dct


def extra_functionalities(n, snapshots):
    """Could be added later if you wanted to convert the pyomo constraints to linopt, but this is a pain."""
    pass


def _nh3_ramp_down(model, t):
    """Places a cap on how quickly the ammonia plant can ramp down"""
    if t == 0:
        old_rate = model.link_p['ammonia_synthesis', model.t.at(-1)]
    else:
        old_rate = model.link_p['ammonia_synthesis', t - 1]

    return old_rate - model.link_p['ammonia_synthesis', t] <= \
        model.link_p_nom['ammonia_synthesis'] * model.ammonia_synthesis_max_ramp_down
    # Note 20 is the UB of the size of the ammonia plant; essentially if x = 0 then the constraint is not active


def _nh3_ramp_up(model, t):
    """Places a cap on how quickly the ammonia plant can ramp down"""
    if t == 0:
        old_rate = model.link_p['ammonia_synthesis', model.t.at(-1)]
    else:
        old_rate = model.link_p['ammonia_synthesis', t - 1]

    return model.link_p['ammonia_synthesis', t] - old_rate <= \
        model.link_p_nom['ammonia_synthesis'] * model.ammonia_synthesis_max_ramp_up


def _nh3_ramp_down_operating(model, t):
    """Places a cap on how quickly the ammonia plant can ramp down"""
    if t == 0:
        old_rate = model.link_p['ammonia_synthesis', model.t.at(-1)]
    else:
        old_rate = model.link_p['ammonia_synthesis', t - 1]

    return old_rate - model.link_p['ammonia_synthesis', t] <= \
        model.ammonia_synthesis_capacity * model.ammonia_synthesis_max_ramp_down
    # Note 20 is the UB of the size of the ammonia plant; essentially if x = 0 then the constraint is not active


def _nh3_ramp_up_operating(model, t):
    """Places a cap on how quickly the ammonia plant can ramp down"""
    if t == 0:
        old_rate = model.link_p['ammonia_synthesis', model.t.at(-1)]
    else:
        old_rate = model.link_p['ammonia_synthesis', t - 1]

    return model.link_p['ammonia_synthesis', t] - old_rate <= \
        model.ammonia_synthesis_capacity * model.ammonia_synthesis_max_ramp_up



def _penalise_ramp_down(model, t):
    """Places a cap on how quickly the ammonia plant can ramp down"""
    if t == 0:
        old_rate = model.link_p['ammonia_synthesis', model.t.at(-1)]
    else:
        old_rate = model.link_p['ammonia_synthesis', t - 1]

    return model.link_p['penalty_link', t] >= (old_rate - model.link_p['ammonia_synthesis', t])


def _penalise_ramp_up(model, t):
    """Places a cap on how quickly the ammonia plant can ramp down"""
    if t == 0:
        old_rate = model.link_p['ammonia_synthesis', model.t.at(-1)]
    else:
        old_rate = model.link_p['ammonia_synthesis', t - 1]

    return model.link_p['penalty_link', t] >= (model.link_p['ammonia_synthesis', t] - old_rate)


def pyomo_constraints(network, snapshots):
    """Includes a series of additional constraints which make the ammonia plant work as needed:
    i) Battery sizing
    ii) Ramp hard constraints down (Cannot be violated)
    iii) Ramp hard constraints up (Cannot be violated)
    iv) Ramp soft constraints down
    v) Ramp soft constraints up
    (iv) and (v) just softly suppress ramping so that the model doesn't 'zig-zag', which looks a bit odd on operation.
    Makes very little difference on LCOA. """

    # The battery constraint is built here - it doesn't need a special function because it doesn't depend on time
    network.model.battery_interface = pm.Constraint(
        rule=lambda model: network.model.link_p_nom['battery_interface_in'] ==
                           network.model.link_p_nom['battery_interface_out'] /
                           network.links.efficiency["battery_interface_out"])

    # Constrain the maximum discharge of the H2 storage relative to its size
    time_step_cycle = 4/8760*0.5*0.5  # Factor 0.5 for half-hourly time step, 0.5 for oversized storage
    network.model.cycling_limit = pm.Constraint(
        rule=lambda model: network.model.link_p_nom['battery_interface_out'] ==
                           network.model.store_e_nom['compressed_hydrogen_store'] * time_step_cycle)

    # The ammonia synthesis ramp constraints are functions of time, so we need to create some pyomo sets/parameters to represent them.
    network.model.t = pm.Set(initialize=network.snapshots)
    network.model.ammonia_synthesis_max_ramp_down = pm.Param(initialize=network.links.loc['ammonia_synthesis'].ramp_limit_down)
    network.model.ammonia_synthesis_max_ramp_up = pm.Param(initialize=network.links.loc['ammonia_synthesis'].ramp_limit_up)

    # Using those sets/parameters, we can now implement the constraints...
    logging.warning('Pypsa has been overridden - Ramp rates on NH3 plant are included')
    network.model.NH3_pyomo_overwrite_ramp_down = pm.Constraint(network.model.t, rule=_nh3_ramp_down)
    network.model.NH3_pyomo_overwrite_ramp_up = pm.Constraint(network.model.t, rule=_nh3_ramp_up)
    # network.model.NH3_pyomo_penalise_ramp_down = pm.Constraint(network.model.t, rule=_penalise_ramp_down)
    # network.model.NH3_pyomo_penalise_ramp_up = pm.Constraint(network.model.t, rule=_penalise_ramp_up)

def pyomo_operating_constraints(network, snapshots):
    """Exactly as per the other constraints, but excludes any constraints which only apply during design"""
    # The ammonia synthesis ramp constraints are functions of time, so we need to create some pyomo sets/parameters to represent them.
    network.model.t = pm.Set(initialize=network.snapshots)
    network.model.ammonia_synthesis_max_ramp_down = pm.Param(initialize=network.links.loc['ammonia_synthesis'].ramp_limit_down)
    network.model.ammonia_synthesis_max_ramp_up = pm.Param(initialize=network.links.loc['ammonia_synthesis'].ramp_limit_up)
    network.model.ammonia_synthesis_capacity = pm.Param(initialize=network.links.loc['ammonia_synthesis'].p_nom_opt)

    # Using those sets/parameters, we can now implement the constraints...
    logging.warning('Pypsa has been overridden - Ramp rates on NH3 plant are included')
    network.model.NH3_pyomo_overwrite_ramp_down = pm.Constraint(network.model.t, rule=_nh3_ramp_down_operating)
    network.model.NH3_pyomo_overwrite_ramp_up = pm.Constraint(network.model.t, rule=_nh3_ramp_up_operating)


def _prepare_linopy_context(network, snapshots):
    """Return the Linopy model and snapshot index, ensuring the model exists."""
    if getattr(network, "model", None) is None:
        raise RuntimeError(
            "Network model not initialised; use linopy constraints via n.optimize(extra_functionality=...)."
        )
    sns = snapshots if snapshots is not None else network.snapshots
    return network.model, sns


def linopy_constraints(network, snapshots):
    """Linopy equivalent of the legacy pyomo_constraints for PyPSA >= 1.0."""
    model, sns = _prepare_linopy_context(network, snapshots)

    # Keep the battery interface charger/discharger capacities coupled
    try:
        batt_in = model["Link-p_nom"].sel(name="battery_interface_in")
        batt_out = model["Link-p_nom"].sel(name="battery_interface_out")
    except KeyError:
        batt_in = batt_out = None

    if batt_in is not None and batt_out is not None:
        eff = network.links.at["battery_interface_out", "efficiency"]
        model.add_constraints(batt_in == batt_out / eff, name="battery_interface_balance")

        try:
            store_cap = model["Store-e_nom"].sel(name="compressed_hydrogen_store")
        except KeyError:
            store_cap = None

        if store_cap is not None:
            time_step_cycle = 4 / 8760 * 0.5 * 0.5
            model.add_constraints(
                batt_out == store_cap * time_step_cycle,
                name="compressed_h2_cycling_limit",
            )

    # Enforce ammonia synthesis ramp limits
    try:
        ammonia_synthesis_dispatch = model["Link-p"].sel(name="ammonia_synthesis", snapshot=sns)
    except KeyError:
        ammonia_synthesis_dispatch = None

    if ammonia_synthesis_dispatch is not None and "ammonia_synthesis" in network.links.index:
        ammonia_synthesis_prev = ammonia_synthesis_dispatch.roll(snapshot=1, roll_coords=False)

        try:
            ammonia_synthesis_capacity = model["Link-p_nom"].sel(name="ammonia_synthesis")
        except KeyError:
            ammonia_synthesis_capacity = network.links.loc["ammonia_synthesis", "p_nom"]

        ramp_down = network.links.at["ammonia_synthesis", "ramp_limit_down"]
        ramp_up = network.links.at["ammonia_synthesis", "ramp_limit_up"]

        if pd.notna(ramp_down):
            model.add_constraints(
                ammonia_synthesis_prev - ammonia_synthesis_dispatch <= ramp_down * ammonia_synthesis_capacity,
                name="ammonia_synthesis_ramp_down",
            )

        if pd.notna(ramp_up):
            model.add_constraints(
                ammonia_synthesis_dispatch - ammonia_synthesis_prev <= ramp_up * ammonia_synthesis_capacity,
                name="ammonia_synthesis_ramp_up",
            )


def linopy_operating_constraints(network, snapshots):
    """Operational counterpart for fixed-capacity reruns using Linopy."""
    model, sns = _prepare_linopy_context(network, snapshots)

    try:
        ammonia_synthesis_dispatch = model["Link-p"].sel(name="ammonia_synthesis", snapshot=sns)
    except KeyError:
        return

    if "ammonia_synthesis" not in network.links.index:
        return

    ammonia_synthesis_prev = ammonia_synthesis_dispatch.roll(snapshot=1, roll_coords=False)
    ammonia_synthesis_capacity = network.links.loc["ammonia_synthesis", "p_nom"]
    ramp_down = network.links.at["ammonia_synthesis", "ramp_limit_down"]
    ramp_up = network.links.at["ammonia_synthesis", "ramp_limit_up"]

    if pd.notna(ramp_down):
        model.add_constraints(
            ammonia_synthesis_prev - ammonia_synthesis_dispatch <= ramp_down * ammonia_synthesis_capacity,
            name="ammonia_synthesis_operating_ramp_down",
        )

    if pd.notna(ramp_up):
        model.add_constraints(
            ammonia_synthesis_dispatch - ammonia_synthesis_prev <= ramp_up * ammonia_synthesis_capacity,
            name="ammonia_synthesis_operating_ramp_up",
        )


def convert_network_to_operating(n, ammonia_cost_per_ton=500, aggregation_count=1, file_name="", multi_site=False,
                                 time_step=1.0):
    """Takes a designed network built with the designer and fixes the parameters as needs be
    ammonia_cost_per_ton = the cost at which ammonia will be sold; this gives the model a reason to make ammonia"""

    # Sets the expandable parameters to false:
    n.links.p_nom_extendable = [False for _ in range(len(n.links))]
    n.stores.e_nom_extendable = [False for _ in range(len(n.stores))]
    n.generators.p_nom_extendable = [False for _ in range(len(n.generators))]

    # Sets the basic equipment size to its optimum size from the last run...
    n.links.p_nom = n.links.p_nom_opt
    n.stores.e_nom = n.stores.e_nom_opt
    n.generators.p_nom = n.generators.p_nom_opt

    # Sets the ammonia storage to be very cheap and expandable - it's just a measure of production
    n.stores.loc['ammonia', 'capital_cost'] = 0.001
    n.stores.loc['ammonia', 'e_nom_extendable'] = True
    n.stores.loc['ammonia', 'e_nom_max'] = 1E9
    n.stores.loc['ammonia', 'e_cyclic'] = False
    n.stores.loc['ammonia', 'e_initial'] = 0


    # Sets the marginal cost of ammonia production to be negative so the system makes a profit...
    n.links.loc['ammonia_synthesis', 'marginal_cost'] = (
        -ammonia_cost_per_ton / AMMONIA_HHV_MWH_PER_T * time_step * aggregation_count / 10
    )  # 10 is no. of years in dataset

    # Turns the ammonia load off:
    n.loads.loc['ammonia', 'p_set'] = 0

    # Adjust the maximum allowable operating rate of the ammonia plant...
    n.links_t.p_max_pu = aggregate_data(
        pd.read_csv('ammonia_synthesis_p_max_pu.csv').set_index('snapshot').rename(columns={'ammonia_synthesis_max': 'ammonia_synthesis'}), aggregation_count)

    # Re-solves model:
    status, condition = n.optimize(
        solver_name='gurobi',
        extra_functionality=linopy_operating_constraints,
    )
    if status != 'ok':
        raise RuntimeError(f"Operating optimisation failed with status {status} ({condition}).")

    if not multi_site:
        detailed_results = get_results_dict_for_excel(n, 1, aggregation_count, operating=True, time_step=time_step)
        write_results_to_excel(detailed_results, file_name=file_name, extension="_operating")
        results = get_results_dict_for_multi_site(n, aggregation_count, operating=True, time_step=time_step)
    else:
        results = get_results_dict_for_multi_site(n, aggregation_count, operating=True, time_step=time_step)

    return results

