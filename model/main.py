from pathlib import Path
import copy
import glob
import os
import sys

import geopandas as gpd
import pandas as pd
import pypsa
from shapely.geometry import Point

try:
    from . import auxiliary as aux
    from . import data_store as pds
    from . import location_tools as plc
    from .constants import HYDROGEN_HHV_MWH_PER_T
except ImportError:  # pragma: no cover - fallback for direct execution
    PACKAGE_ROOT = Path(__file__).resolve().parent
    if str(PACKAGE_ROOT) not in sys.path:
        sys.path.insert(0, str(PACKAGE_ROOT))
    import auxiliary as aux  # type: ignore
    import data_store as pds  # type: ignore
    import location_tools as plc  # type: ignore
    from constants import HYDROGEN_HHV_MWH_PER_T  # type: ignore

"""File to optimise the size of a green ammonia plant given a specified wind and solar profile"""


REPO_ROOT = Path(__file__).resolve().parents[1]

COMPONENT_NAME_MAP = {
    "Power": "power",
    "Ammonia": "ammonia",
    "PowerStorage": "power_storage",
    "Hydrogen": "hydrogen",
    "HydrogenStorage": "hydrogen_storage",
    "RampPenalty": "ramp_penalty",
    "RampPenaltyDest": "ramp_penalty_dest",
    "Wind": "onshore_wind",
    "wind": "onshore_wind",
    "OnshoreWind": "onshore_wind",
    "OffshoreWindFixed": "offshore_wind_fixed",
    "OffshoreWindFloating": "offshore_wind_floating",
    "Solar": "solar",
    "SolarTracking": "solar_tracking",
    "Grid": "grid",
    "RampDummy": "ramp_dummy",
    "Electrolysis": "electrolysis",
    "HydrogenCompression": "hydrogen_compression",
    "HydrogenFromStorage": "hydrogen_from_storage",
    "HB": "ammonia_synthesis",
    "BatteryInterfaceIn": "battery_interface_in",
    "BatteryInterfaceOut": "battery_interface_out",
    "HydrogenFuelCell": "hydrogen_fuel_cell",
    "PenaltyLink": "penalty_link",
    "CompressedH2Store": "compressed_hydrogen_store",
    "Battery": "battery",
    "BatteryStorage": "battery",
    "battery_storage": "battery",
    "AccumulatedPenalty": "accumulated_penalty",
}

WEATHER_PROFILE_ALIASES = {
    "onshore_wind": ("wind",),
    "offshore_wind_fixed": ("wind", "onshore_wind"),
    "offshore_wind_floating": ("wind", "onshore_wind"),
    "solar_tracking": ("solar",),
    "wind": ("onshore_wind",),
}


def normalize_component_name(name: str | None) -> str | None:
    if name is None:
        return None
    return COMPONENT_NAME_MAP.get(name, name)


def _resolve_csv_folder(data_file):
    """Return absolute path to the component CSV bundle."""
    data_path = Path(data_file)
    if not data_path.is_absolute():
        data_path = REPO_ROOT / data_path
    return data_path


def _ensure_carriers_populated(network: pypsa.Network) -> None:
    """Add missing carrier rows based on component references."""
    existing = set(network.carriers.index)
    referenced = set()
    for component in [network.buses, network.generators, network.links, network.stores, network.loads]:
        if component.empty or "carrier" not in component.columns:
            continue
        referenced.update(component["carrier"].dropna().unique())

    for carrier in sorted(filter(None, referenced)):
        if carrier not in existing:
            network.add("Carrier", carrier)


def _convert_link_cost_value(links, equipment: str, value: float) -> float:
    """Return a capital cost expressed on an output basis in PyPSA's input basis."""
    if equipment not in links.index:
        return value
    if "rating_basis" not in links.columns:
        return value
    if str(links.at[equipment, "rating_basis"]).lower() != "output":
        return value
    column = "output_basis_efficiency" if "output_basis_efficiency" in links.columns else "efficiency"
    efficiency = float(links.at[equipment, column])
    if efficiency <= 0:
        raise ValueError(f"Efficiency must be positive for link '{equipment}'.")
    return value * efficiency


def generate_network(
    n_snapshots,
    data_file,
    aggregation_count=1,
    costs=None,
    efficiencies=None,
    time_step=0.5,
    tech_config_overrides=None,
):
    """Generates a network that can be used to run several cases"""
    # ==================================================================================================================
    # Set up network
    # ==================================================================================================================

    # Import a generic network
    network = pypsa.Network()

    # Import the design of the H2 plant into the network
    network.import_from_csv_folder(_resolve_csv_folder(data_file))

    # Ensure snapshot count matches the requested weather data length
    network.set_snapshots(range(int(n_snapshots/aggregation_count)))

    # Populate carrier table if the CSV bundle omits carriers.csv
    _ensure_carriers_populated(network)

    # Techno-economic inputs are expected to be pre-processed into the CSV bundle
    # (annuitised capital_cost, correct link cost basis, efficiencies, etc.).
    # tech_config_overrides is retained in the signature for compatibility but is
    # intentionally not applied here.

    if costs is not None:
        for equipment, row in costs.items():
            equipment = normalize_component_name(equipment)
            for df in [network.links, network.generators, network.stores]:
                if equipment in df.index:
                    value = row
                    if df is network.links:
                        value = _convert_link_cost_value(network.links, equipment, row)
                    df.loc[equipment, 'capital_cost'] = value

    if efficiencies is not None:
        for equipment, row in efficiencies.items():
            equipment = normalize_component_name(equipment)
            if equipment in network.links.index:
                network.links.loc[equipment, 'efficiency'] = row

    # Adjust the capital cost of the stores, and the marginal costs, based on the aggregation and number of datapoints
    network.stores.capital_cost *= time_step
    network.links.marginal_cost *= 24*366 / n_snapshots
    # Just in case you have a default that is different to 1 hour in the data
    if aggregation_count is not None:
        network.stores.capital_cost *= aggregation_count
        network.links.marginal_cost *= aggregation_count

    return network


def apply_weather_profiles(network, weather_data):
    """Populate generator availability profiles from a weather mapping.

    Args:
        network: pypsa.Network instance whose generators correspond to the
            renewable assets referenced in the weather mapping.
        weather_data: Either a dict-like object or pandas.DataFrame whose
            items iterate over (name, series) pairs. The values must be
            aligned with the network snapshot index.
    """
    if weather_data is None:
        raise ValueError("weather_data must be provided to apply_weather_profiles().")

    weather_frame = pd.DataFrame(weather_data).copy()
    weather_frame = weather_frame.reset_index(drop=True)
    if len(weather_frame) != len(network.snapshots):
        raise ValueError(
            f"Weather/profile length mismatch: got {len(weather_frame)} rows for {len(network.snapshots)} snapshots."
        )

    missing_profiles = []
    for generator_name in network.generators.index.to_list():
        selected_column = None
        if generator_name in weather_frame.columns:
            selected_column = generator_name
        else:
            for candidate in WEATHER_PROFILE_ALIASES.get(generator_name, ()):
                if candidate in weather_frame.columns:
                    selected_column = candidate
                    break

        if selected_column is None and generator_name == "grid":
            network.generators_t.p_max_pu[generator_name] = 0.0
            continue
        if selected_column is None and generator_name == "ramp_dummy":
            network.generators_t.p_max_pu[generator_name] = 1.0
            continue
        if selected_column is None:
            missing_profiles.append(generator_name)
            continue

        network.generators_t.p_max_pu[generator_name] = weather_frame[selected_column].to_numpy()

    if missing_profiles:
        raise ValueError(
            "Missing weather datasets for generators: {missing}. "
            "Provide matching columns in the weather frame or define aliases.".format(
                missing=", ".join(sorted(missing_profiles))
            )
        )
    return network


def main(
    n=None,
    file_name=None,
    weather_data=None,
    multi_site=False,
    get_complete_output=False,
    extension="",
    aggregation_count=1,
    time_step=1.0,
    tech_config_overrides=None,
    interest_rates=None,
    water_cost_usd_per_m3=None,
    water_usage_m3_per_t=None,
    land_cost_usd_per_km2_year=None,
    land_used_km2=None,
):
    """Code to execute run at a single location"""
    # Import the weather data
    if file_name is not None and weather_data is None:
        weather_data = aux.get_weather_data(file_name=file_name, aggregation_count=aggregation_count)

    # Import a generic network if needed
    if n is None:
        n = generate_network(
            len(weather_data),
            "basic_ammonia_plant",
            aggregation_count=aggregation_count,
            time_step=time_step,
            tech_config_overrides=tech_config_overrides,
        )

    # Note: All flows are in MW or MWh, conversions for hydrogen use HYDROGEN_HHV_MWH_PER_T (39.4 MWh/t)

    # ==================================================================================================================
    # Send the weather data to the model
    # ==================================================================================================================

    apply_weather_profiles(n, weather_data)

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
    solver = os.environ.get("GREEN_LORY_SOLVER", "gurobi").lower()
    # Set GREEN_LORY_SOLVER to "highs" or "glpk" if you want to switch solvers locally.

    solver_threads = None
    solver_threads_raw = os.environ.get("GREEN_LORY_SOLVER_THREADS", "").strip()
    if solver_threads_raw:
        solver_threads = int(solver_threads_raw)
        if solver_threads < 1:
            raise ValueError("GREEN_LORY_SOLVER_THREADS must be >= 1")

    solver_options = None
    if solver == "gurobi":
        solver_log_flag = os.environ.get("GREEN_LORY_SOLVER_LOG", "0").lower()
        quiet_log = solver_log_flag in {"0", "false", "no", "off"}
        # Barrier (Method=2) without crossover (Crossover=0) via the direct
        # Gurobi Python API (io_api="direct") is ~24× faster than the default
        # LP-file + concurrent-simplex path for this problem class.
        solver_options = {
            "OutputFlag": 0 if quiet_log else 1,
            "LogToConsole": 0 if quiet_log else 1,
            "Method": 2,       # interior-point barrier
            "Crossover": 0,    # skip crossover — we only need the objective + primals
        }
        if solver_threads is not None:
            solver_options["Threads"] = solver_threads
    elif solver == "highs":
        solver_options = {"solver": "ipm"}   # HiGHS interior-point is similarly faster
        if solver_threads is not None:
            solver_options["threads"] = solver_threads

    optimize_kwargs = {
        "solver_name": solver,
        "extra_functionality": aux.linopy_constraints,
        "io_api": "direct",   # bypass LP-file write/read; use Gurobi Python API directly
    }
    if solver_options is not None:
        optimize_kwargs["solver_options"] = solver_options

    # Implement their answer using the Linopy-based optimiser
    status, condition = n.optimize(**optimize_kwargs)
    if status != 'ok':
        raise RuntimeError(f"Optimisation failed with status {status} ({condition}).")

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
        output = aux.get_results_dict_for_multi_site(
            n,
            aggregation_count=aggregation_count,
            time_step=time_step,
            interest_rates=interest_rates,
            water_cost_usd_per_m3=water_cost_usd_per_m3,
            water_usage_m3_per_t=water_usage_m3_per_t,
            land_cost_usd_per_km2_year=land_cost_usd_per_km2_year,
            land_used_km2=land_used_km2,
        )

    return output
