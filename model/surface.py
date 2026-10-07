"""Run the plant over many cells: a cost and capacity surface.

    results = run_surface(locations, land_csv=..., tech_yaml=..., plant_dir=..., override_csv=...,
                          weather_dir=..., settings=RunSettings(time_step=1.0))

For a run defined in runs.yaml, the ARC job (arc/jobs/01_run_global.sh) calls run_surface with
that run's settings. Locally, `python -m model.surface --help` runs a few cells.

What happens:
1. read_run_inputs() reads everything shared by the cells once: the technology YAML, the
   override table and the land table (each indexed by cell), the weather stack, and the plant
   bundle as a base network.
2. run_cell() handles one cell: it copies the base network, sets the YAML costs and the cell's
   overrides, solves the plant for the cell's weather (plant.solve), then completes the
   accounting (accounting.py) and the land capacity (land.py).
3. solve_cell() retries a numerical solver failure on a fresh network with sturdier Gurobi
   settings (NUMERICAL_RETRY_OPTIONS).
4. run_surface() runs the cells serially or in a pool of forked workers that share the loaded
   inputs, writes each row as it finishes, and finally rewrites the table with its columns in
   a fixed order.
"""
from __future__ import annotations

import argparse
import copy
import io
import json
import logging
import math
import multiprocessing
import os
import sys
import time
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Any, Iterator, Sequence

import geopandas as gpd
import pandas as pd
import pypsa
from shapely.errors import GEOSException
from shapely.geometry import box

from . import accounting, land, plant, technology, weather
from .config import COUNTRIES_GEOJSON, DEFAULT_PLANT_DIR, DEFAULT_TECH_YAML, WEATHER_DIR, repo_path

LOGGER = logging.getLogger(__name__)

# Retry chain after a numerical termination, cheapest first (probes of 28 Sep 2026 on stalling
# cells: the default barrier 5 s; barrier with crossover about a minute; dual simplex with
# numeric focus about fifteen minutes). Each attempt rebuilds the network from scratch.
CROSSOVER_GUROBI_OPTIONS = {"Method": 2, "Crossover": 1, "BarConvTol": 1e-8}
ROBUST_GUROBI_OPTIONS = {"Method": 1, "DualReductions": 0, "NumericFocus": 3, "InfUnbdInfo": 1}
NUMERICAL_RETRY_OPTIONS = (CROSSOVER_GUROBI_OPTIONS, ROBUST_GUROBI_OPTIONS)
NUMERICAL_RETRY_CONDITIONS = frozenset({"suboptimal", "numeric", "numerical", "infeasible_or_unbounded"})

LAND_CONSTRAINTS = ("after_solve", "in_solve")
CAPACITY_RULES = ("scaled_reference_design", "solved_quantity")


@dataclass(frozen=True)
class RunSettings:
    """The settings of a surface run that change results (runs.yaml gives one set per run)."""

    time_step: float = 1.0
    max_snapshots: int | None = None
    pv_policy: str = "both"
    land_constraint: str = "after_solve"
    capacity_rule: str = "scaled_reference_design"
    land_allocation: str = "colocated"
    temporal_accounting_mode: str = "snapshot_weighted"
    ramp_limit_basis: str = "per_hour"
    include_site_costs: bool = True

    def __post_init__(self) -> None:
        if not float(self.time_step).is_integer() or float(self.time_step) < 1:
            raise ValueError("time_step must be a positive whole number of hours")
        if 8760 % int(self.time_step) != 0:
            raise ValueError("time_step must divide the 8760-hour model year exactly")
        if self.pv_policy not in plant.PV_POLICIES:
            raise ValueError(f"pv_policy must be one of {plant.PV_POLICIES}")
        if self.temporal_accounting_mode not in plant.TEMPORAL_ACCOUNTING_MODES:
            raise ValueError(f"temporal_accounting_mode must be one of {plant.TEMPORAL_ACCOUNTING_MODES}")
        if self.ramp_limit_basis not in plant.RAMP_LIMIT_BASES:
            raise ValueError(f"ramp_limit_basis must be one of {plant.RAMP_LIMIT_BASES}")
        if self.land_constraint not in LAND_CONSTRAINTS:
            raise ValueError(f"land_constraint must be one of {LAND_CONSTRAINTS}")
        if self.capacity_rule not in CAPACITY_RULES:
            raise ValueError(f"capacity_rule must be one of {CAPACITY_RULES}")
        if self.capacity_rule == "solved_quantity" and self.land_constraint != "in_solve":
            raise ValueError("capacity_rule='solved_quantity' requires land_constraint='in_solve'")
        if self.capacity_rule == "scaled_reference_design" and self.land_constraint != "after_solve":
            raise ValueError(
                "capacity_rule='scaled_reference_design' requires land_constraint='after_solve'; scaling a "
                "land-constrained reference design would apply the land budget twice"
            )
        land.normalise_land_allocation(self.land_allocation)


@dataclass
class RunInputs:
    """Everything the cells of one run share, read once (pool workers inherit it by fork)."""

    settings: RunSettings
    tech_inputs: dict[str, dict]
    tech_meta: dict[str, Any]
    overrides: dict
    land: dict
    base_network: pypsa.Network
    weather: weather.ProfileStack | None = None


def read_run_inputs(
    settings: RunSettings,
    *,
    land_csv: str | Path,
    tech_yaml: str | Path | None = None,
    plant_dir: str | Path | None = None,
    override_csv: str | Path | None = None,
    weather_dir: str | Path | None = None,
    load_weather: bool = True,
) -> RunInputs:
    """Read a run's shared inputs. The base network has the run's snapshot count, the YAML's
    link efficiencies and the run's PV policy."""
    tech_yaml = repo_path(tech_yaml) if tech_yaml is not None else DEFAULT_TECH_YAML
    plant_dir = plant.check_plant_bundle(plant_dir if plant_dir is not None else DEFAULT_PLANT_DIR)
    tech_inputs = technology.load_tech_inputs(tech_yaml)
    tech_meta = technology.load_tech_meta(tech_yaml)
    land_table = land.load_land_table(land_csv, tech_inputs)
    profiles = weather.ProfileStack(weather_dir if weather_dir is not None else WEATHER_DIR) if load_weather else None

    hours = len(profiles.resources["solar"]["time"]) if profiles is not None else 8760
    n_snapshots = settings.max_snapshots if settings.max_snapshots is not None else hours // int(settings.time_step)
    base_network = plant.build_network(
        n_snapshots,
        plant_dir,
        time_step=settings.time_step,
        temporal_accounting_mode=settings.temporal_accounting_mode,
    )
    technology.apply_link_efficiencies(base_network, tech_inputs)
    plant.restrict_pv(base_network, settings.pv_policy)
    base_network._currency = str(tech_meta.get("currency", "USD")).strip().upper()
    base_network._ramp_limit_basis = settings.ramp_limit_basis
    return RunInputs(
        settings=settings,
        tech_inputs=tech_inputs,
        tech_meta=tech_meta,
        overrides=technology.index_override_table(technology.load_override_table(override_csv)),
        land=land.index_land_table(land_table),
        base_network=base_network,
        weather=profiles,
    )


# ---------------------------------------------------------------------------------------------
# One cell
# ---------------------------------------------------------------------------------------------

class CellFailure(RuntimeError):
    """A cell that could not be run; ``stage`` is weather, setup or solver."""

    def __init__(self, stage: str, message: str):
        self.stage = stage
        super().__init__(message)

    def __reduce__(self):
        return type(self), (self.stage, str(self))


def cell_weather(inputs: RunInputs, lat: float, lon: float) -> pd.DataFrame:
    """One cell's profiles at the run's time step, cut to max_snapshots."""
    settings = inputs.settings
    frame = weather.cell_profiles(inputs.weather, lat, lon)
    if settings.time_step > 1:
        frame = weather.resample(frame, int(settings.time_step))
    if settings.max_snapshots is not None:
        frame = frame.iloc[: settings.max_snapshots].copy()
    return frame


def run_cell(
    inputs: RunInputs,
    lat: float,
    lon: float,
    *,
    quiet: bool = True,
    solver_options_override: dict | None = None,
    weather_frame: pd.DataFrame | None = None,
) -> dict[str, Any]:
    """Solve one cell and return its complete result row (without unit suffixes).

    Raises CellFailure; a solver failure carries the OptimizationFailure as its cause.
    """
    settings = inputs.settings
    try:
        frame = weather_frame if weather_frame is not None else cell_weather(inputs, lat, lon)
    except Exception as exc:  # noqa: BLE001
        raise CellFailure("weather", f"Weather extraction failed at ({lat}, {lon}): {exc}") from exc

    try:
        overrides = technology.cell_overrides(inputs.overrides, lat, lon)
        rates = technology.interest_rates(overrides)
        location_costs = overrides.get("__location__", {}) if overrides else {}
        water_cost_usd_per_m3 = location_costs.get("water_cost_usd_per_m3")
        land_cost_usd_per_km2_year = location_costs.get("land_cost_usd_per_km2_year")
        if water_cost_usd_per_m3 is None:
            water_cost_usd_per_m3 = inputs.tech_meta.get("water_cost_baseline_usd_per_m3")
        water_usage_m3_per_t = inputs.tech_meta.get("water_usage_m3_per_t_nh3")

        network = copy.deepcopy(inputs.base_network)
        network._ramp_limit_basis = settings.ramp_limit_basis
        network._currency = str(inputs.tech_meta.get("currency", "USD")).strip().upper()
        technology.apply_base_costs(
            network, inputs.tech_inputs, time_step=settings.time_step,
            temporal_accounting_mode=settings.temporal_accounting_mode,
        )
        technology.apply_cell_overrides(
            network, inputs.tech_inputs, overrides, time_step=settings.time_step,
            temporal_accounting_mode=settings.temporal_accounting_mode,
        )
        land_row = land.cell_land_row(inputs.land, lat, lon)
        if land_row is None:
            raise ValueError(f"No land row found for ({lat}, {lon})")
        budget = land.build_renewable_land_budget(land_row, inputs.tech_inputs, onshore=True)
        if settings.land_constraint == "in_solve":
            land_cap = land.apply_land_budget(network, budget, settings.land_allocation)
        else:
            land_cap = None
            # The reference design does not depend on the land table: remove any finite
            # placeholder bound of the bundle so it cannot act as a land or resource limit.
            for renewable in plant.RENEWABLES:
                if renewable in network.generators.index and bool(network.generators.at[renewable, "p_nom_extendable"]):
                    network.generators.at[renewable, "p_nom_max"] = math.inf
    except Exception as exc:  # noqa: BLE001
        raise CellFailure("setup", f"Network setup failed at ({lat}, {lon}): {exc}") from exc

    try:
        with _suppress_solver_streams(quiet):
            results = plant.solve(
                network,
                frame,
                time_step=settings.time_step,
                interest_rates=rates,
                water_cost_usd_per_m3=water_cost_usd_per_m3,
                water_usage_m3_per_t=water_usage_m3_per_t,
                land_cost_usd_per_km2_year=land_cost_usd_per_km2_year,
                land_used_km2=None,
                solver_options_override=solver_options_override,
            )
    except Exception as exc:  # noqa: BLE001
        raise CellFailure("solver", f"Optimisation failed at ({lat}, {lon}): {exc}") from exc

    land_use = land.renewable_land_use_from_results(results, budget)
    land_used_km2 = land_use.total_km2
    if land_used_km2 is not None:
        results["land_used_km2"] = float(land_used_km2)
        results["wind_land_used_km2"] = land_use.wind_km2
        results["fixed_solar_land_used_km2"] = land_use.fixed_solar_km2
        results["tracking_solar_land_used_km2"] = land_use.tracking_solar_km2
        results["solar_land_used_km2"] = land_use.solar_km2
    if land_cap is not None:
        results["land_capacity_cap"] = land_cap
        results["land_capacity_cap_mw"] = land_cap
    results.update(land.land_metadata(land_row))
    results["land_constraint"] = settings.land_constraint
    results["capacity_rule"] = settings.capacity_rule
    results["land_allocation"] = settings.land_allocation
    results["wind_land_exclusive_fraction_effective"] = land.effective_wind_land_exclusive_fraction(
        settings.land_allocation, budget
    )
    results["temporal_accounting_mode"] = settings.temporal_accounting_mode
    results["ramp_limit_basis"] = settings.ramp_limit_basis
    results["scenario_id"] = os.environ.get("ARC_SCENARIO_ID", "adhoc")
    results["run_id"] = os.environ.get("ARC_RUN_ID", "adhoc")
    results["manifest_sha256"] = os.environ.get("ARC_MANIFEST_SHA256", "")
    results["interest_overrides_applied"] = bool(overrides)

    splits = accounting.cost_splits(
        network, inputs.tech_inputs, overrides, time_step=settings.time_step,
        temporal_accounting_mode=settings.temporal_accounting_mode,
    )
    results.update(splits)

    output_currency = str(inputs.tech_meta.get("currency", results.get("currency", "USD"))).strip().upper()
    source_currency = str(inputs.tech_meta.get("spatial_cost_currency", "USD")).strip().upper()
    source_to_output_fx = inputs.tech_meta.get("spatial_cost_to_model_currency")
    if source_to_output_fx is None:
        if source_currency != output_currency:
            raise ValueError("Tech YAML must define spatial_cost_to_model_currency when site-cost and model currencies differ")
        source_to_output_fx = 1.0
    results = accounting.finalize_site_cost_accounting(
        results,
        output_currency=output_currency,
        source_currency=source_currency,
        source_to_output_fx=float(source_to_output_fx),
        water_cost_source_per_m3=water_cost_usd_per_m3,
        water_usage_m3_per_t_nh3=water_usage_m3_per_t if water_cost_usd_per_m3 is not None else None,
        land_cost_source_per_km2_year=float(land_cost_usd_per_km2_year) if land_cost_usd_per_km2_year is not None else 0.0,
        land_used_km2=land_used_km2,
        include_site_costs_in_headline=settings.include_site_costs,
    )
    results = accounting.finalize_headline_cost_percentages(results, plant_split_percentages=splits)

    results.setdefault("grid_energy_mwh", 0.0)
    if "power_bus_generator_supply_mwh" not in results:
        raise ValueError("Solved results are missing the electrical grid-share denominator 'power_bus_generator_supply_mwh'")
    results = accounting.finalize_grid_reporting(
        results,
        electrical_reference_energy_mwh=float(results["power_bus_generator_supply_mwh"]),
        absolute_tolerance_mwh=1.0,
        relative_tolerance=1e-6,
    )

    if settings.capacity_rule == "solved_quantity":
        results.update(land.report_land_feasible_quantity(results, budget, land_allocation=settings.land_allocation))
    else:
        capacity = land.estimate_scaled_design_capacity(results, budget, land_allocation=settings.land_allocation)
        results.update(capacity.to_mapping())
    return results


def _optimization_failure(error: BaseException | None) -> plant.OptimizationFailure | None:
    while error is not None:
        if isinstance(error, plant.OptimizationFailure):
            return error
        error = error.__cause__
    return None


def solve_cell(inputs: RunInputs, lat: float, lon: float, *, fail_fast: bool, quiet: bool = True) -> tuple[str, dict | None]:
    """run_cell() with the numerical retry chain. Returns (status, results).

    After a suboptimal or numerical Gurobi termination the cell is solved again on a fresh
    network with each option set of NUMERICAL_RETRY_OPTIONS in turn. Any other failure is
    raised when ``fail_fast``, otherwise logged and returned as (stage, None).
    """
    attempts = [None, *NUMERICAL_RETRY_OPTIONS]
    gurobi = os.environ.get("GREEN_LORY_SOLVER", "gurobi").strip().lower() == "gurobi"
    first_condition = ""
    for index, options in enumerate(attempts):
        try:
            results = run_cell(inputs, lat, lon, quiet=quiet, solver_options_override=options)
        except CellFailure as error:
            failure = _optimization_failure(error)
            retryable = (
                failure is not None and failure.condition in NUMERICAL_RETRY_CONDITIONS
                and gurobi and index < len(attempts) - 1
            )
            if not retryable:
                if fail_fast:
                    raise
                LOGGER.warning("Cell (%s, %s) failed after %d attempt(s): %s", lat, lon, index + 1, error)
                return error.stage, None
            if index == 0:
                first_condition = failure.condition
            LOGGER.warning(
                "Numerical termination %s at (%s, %s) on attempt %d; retrying on a fresh network with %s",
                failure.condition, lat, lon, index + 1, attempts[index + 1],
            )
            continue
        results["solver_numerical_retry"] = index > 0
        results["solver_retry_attempts"] = index
        results["solver_retry_termination"] = first_condition
        results["solver_retry_options"] = json.dumps(options, sort_keys=True) if options else ""
        return "done", results
    raise AssertionError("retry chain exhausted without a result or an exception")


# ---------------------------------------------------------------------------------------------
# Countries
# ---------------------------------------------------------------------------------------------

COUNTRY_ASSIGNMENT_AREA_CRS = "EPSG:6933"


def load_countries(path: str | Path = COUNTRIES_GEOJSON) -> gpd.GeoDataFrame:
    """Country polygons with a 'country' column (Natural Earth admin-0 style GeoJSON)."""
    path = repo_path(path)
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}: add a Natural Earth admin-0 GeoJSON to tag cells by country.")
    frame = gpd.read_file(path)
    for candidate in ("ADMIN", "NAME", "name", "COUNTRY", "Country"):
        if candidate in frame.columns:
            frame = frame.rename(columns={candidate: "country"})
            break
    if "country" not in frame.columns:
        raise ValueError(f"{path} needs a 'country'/'ADMIN' column and valid geometries.")
    return frame[["country", "geometry"]]


def country_lookup(world: gpd.GeoDataFrame, locations: Sequence[tuple[float, float]], cell_size_deg: float = 1.0) -> dict[tuple[float, float], str]:
    """Each cell's country: the one covering the largest share of the cell's 1-degree box."""
    lookup = {(float(lat), float(lon)): "" for lat, lon in locations}
    if not locations or world.empty:
        return lookup
    countries = world[["country", "geometry"]].dropna(subset=["geometry"]).copy()
    if countries.empty:
        return lookup
    if countries.crs is None:
        countries = countries.set_crs("EPSG:4326")
    # Repair invalid geometries, which can otherwise crash GEOS overlay operations
    if not countries.geometry.is_valid.all():
        countries["geometry"] = countries.geometry.buffer(0)
        countries = countries.dropna(subset=["geometry"])
        countries = countries[countries.geometry.is_valid].copy()
        if countries.empty:
            return lookup

    half = float(cell_size_deg) / 2.0
    points = [(float(lat), float(lon)) for lat, lon in locations]
    cells = gpd.GeoDataFrame(
        {"latitude": [lat for lat, _ in points], "longitude": [lon for _, lon in points]},
        geometry=[box(lon - half, lat - half, lon + half, lat + half) for lat, lon in points],
        crs=countries.crs,
    )
    candidates = gpd.sjoin(cells, countries, how="left", predicate="intersects")
    candidates = candidates.dropna(subset=["index_right"]).reset_index().rename(columns={"index": "cell_idx"})
    if candidates.empty:
        return lookup
    countries_equal_area = countries.to_crs(COUNTRY_ASSIGNMENT_AREA_CRS).reset_index(drop=True)
    candidates_equal_area = gpd.GeoDataFrame(
        candidates[["cell_idx", "latitude", "longitude", "country", "index_right"]].copy(),
        geometry=candidates.geometry,
        crs=cells.crs,
    ).to_crs(COUNTRY_ASSIGNMENT_AREA_CRS)
    cell_geometries = gpd.GeoSeries(candidates_equal_area.geometry.reset_index(drop=True), crs=COUNTRY_ASSIGNMENT_AREA_CRS)
    country_geometries = gpd.GeoSeries(
        countries_equal_area.geometry.iloc[candidates_equal_area["index_right"].astype(int).to_numpy()].reset_index(drop=True),
        crs=COUNTRY_ASSIGNMENT_AREA_CRS,
    )
    try:
        overlap_area = cell_geometries.intersection(country_geometries).area
    except GEOSException:
        overlap_area = cell_geometries.buffer(0).intersection(country_geometries.buffer(0)).area
    candidates_equal_area = candidates_equal_area.reset_index(drop=True)
    candidates_equal_area["overlap_area"] = overlap_area.to_numpy()
    candidates_equal_area = candidates_equal_area[candidates_equal_area["overlap_area"] > 0]
    best = candidates_equal_area.sort_values(["cell_idx", "overlap_area"], ascending=[True, False])
    best = best.drop_duplicates(subset=["cell_idx"], keep="first")
    for row in best.itertuples(index=False):
        lookup[(float(row.latitude), float(row.longitude))] = str(row.country) if pd.notna(row.country) else ""
    return lookup


# ---------------------------------------------------------------------------------------------
# Many cells
# ---------------------------------------------------------------------------------------------

# Inputs inherited by forked pool workers (set by run_surface before the pool starts)
_WORKER: dict[str, Any] = {}


def _worker_task(lat_lon: tuple[float, float]) -> tuple[float, float, str, dict | None]:
    lat, lon = lat_lon
    if _WORKER["fail_fast"]:
        # Any exception propagates; the pool re-raises it in the parent with its traceback
        status, results = solve_cell(_WORKER["inputs"], lat, lon, fail_fast=True, quiet=_WORKER["quiet"])
    else:
        try:
            status, results = solve_cell(_WORKER["inputs"], lat, lon, fail_fast=False, quiet=_WORKER["quiet"])
        except Exception as exc:  # noqa: BLE001
            LOGGER.error("Unhandled error in worker at (%s, %s): %s", lat, lon, exc, exc_info=True)
            status, results = "error", None
    return lat, lon, status, results


def default_locations(land_index: dict) -> list[tuple[float, float]]:
    """Every cell of the land table with positive capacity."""
    cells = []
    for row in land_index.values():
        capacity = row.get("max_capacity_mw", row.get("availability", 1.0))
        if pd.notna(capacity) and float(capacity) > 0:
            cells.append((float(row["latitude"]), float(row["longitude"])))
    return cells


def run_surface(
    locations: Sequence[tuple[float, float]] | None,
    *,
    land_csv: str | Path,
    tech_yaml: str | Path | None = None,
    plant_dir: str | Path | None = None,
    override_csv: str | Path | None = None,
    weather_dir: str | Path | None = None,
    settings: RunSettings = RunSettings(),
    output_csv: str | Path | None = None,
    quiet: bool = False,
    threads_per_worker: int | None = None,
    num_workers: int = 1,
    lon_min: float | None = None,
    lon_max: float | None = None,
    fail_fast: bool = False,
) -> pd.DataFrame:
    """Run the plant at every location (default: every cell of the land table with capacity).

    ``threads_per_worker`` sets the solver threads of each worker; ``lon_min``/``lon_max``
    restrict the run to a longitude band [lon_min, lon_max) (how the ARC shards split a surface).
    With ``fail_fast`` any failure stops the run with its traceback.
    """
    if land_csv is None:
        raise ValueError("land_csv is required: pass a centred land table (cell_anchor=center)")
    with _quiet_logging(quiet), _override_env("GREEN_LORY_SOLVER_LOG", "0", quiet):
        started = time.perf_counter()
        inputs = read_run_inputs(
            settings, land_csv=land_csv, tech_yaml=tech_yaml, plant_dir=plant_dir,
            override_csv=override_csv, weather_dir=weather_dir,
        )
        if locations is None:
            location_list = default_locations(inputs.land)
        else:
            location_list = [(float(lat), float(lon)) for lat, lon in locations]
        if lon_min is not None or lon_max is not None:
            lo = float(lon_min) if lon_min is not None else -180.0
            hi = float(lon_max) if lon_max is not None else 180.0
            location_list = [(la, lo_) for la, lo_ in location_list if lo <= lo_ < hi]
            LOGGER.info("Longitude filter [%.1f, %.1f): %d locations", lo, hi, len(location_list))
        countries = country_lookup(load_countries(), location_list)
        LOGGER.info(
            "Inputs read in %.1fs: %d override cells, %d land cells, base network of %d snapshots",
            time.perf_counter() - started, len(inputs.overrides), len(inputs.land), len(inputs.base_network.snapshots),
        )

        rows: dict[str, dict] = {}
        writer = _RowWriter(output_csv)
        progress = _ProgressTracker(len(location_list))

        def record(lat: float, lon: float, status: str, results: dict | None) -> None:
            if results is not None:
                country = countries.get((lat, lon), "")
                row = {"latitude": lat, "longitude": lon, "country": country, **plant.add_unit_suffixes(results)}
                rows[f"{lat}_{lon}"] = row
                writer.write(row)
            progress.update(lat, lon, status)

        threads = 1 if threads_per_worker is None else int(threads_per_worker)
        try:
            with _override_env("GREEN_LORY_SOLVER_THREADS", str(threads), threads_per_worker is not None):
                if num_workers > 1:
                    _run_pool(inputs, location_list, num_workers, threads, quiet, fail_fast, output_csv, record)
                else:
                    for lat, lon in location_list:
                        try:
                            status, results = solve_cell(inputs, lat, lon, fail_fast=fail_fast, quiet=quiet)
                        except Exception as exc:  # noqa: BLE001
                            if fail_fast:
                                raise
                            LOGGER.error("Unhandled error at (%s, %s); skipping: %s", lat, lon, exc, exc_info=True)
                            status, results = "error", None
                        record(lat, lon, status, results)
        finally:
            if inputs.weather is not None:
                inputs.weather.close()
            writer.close()

        frame = order_result_columns(pd.DataFrame.from_dict(rows, orient="index"))
        if output_csv:
            # Rewrite in column order (the incremental file keeps each row's own order)
            path = repo_path(output_csv)
            path.parent.mkdir(parents=True, exist_ok=True)
            frame.to_csv(path, index=False)
        return frame


def _run_pool(inputs, location_list, num_workers, threads, quiet, fail_fast, output_csv, record) -> None:
    """Run the cells in forked workers that inherit the loaded inputs (copy on write)."""
    LOGGER.info("Starting parallel run: %d workers x %d threads = %d CPUs", num_workers, threads, num_workers * threads)
    _WORKER.update(inputs=inputs, quiet=quiet, fail_fast=fail_fast)
    context = multiprocessing.get_context("fork" if sys.platform != "win32" else "spawn")
    received: set[tuple[float, float]] = set()
    try:
        # chunksize=1: a worker killed by the system loses at most one cell
        with context.Pool(processes=num_workers) as pool:
            for lat, lon, status, results in pool.imap_unordered(_worker_task, location_list, chunksize=1):
                received.add((lat, lon))
                record(lat, lon, status, results)
    except Exception as exc:  # noqa: BLE001
        # Cell errors come back as results; reaching here means the pool itself died (memory,
        # Slurm, licence). List the dropped cells for a resubmission and fail the job.
        dropped = [location for location in location_list if location not in received]
        LOGGER.error(
            "Pool iteration aborted after %d/%d locations: %s; %d location(s) were never processed.",
            len(received), len(location_list), exc, len(dropped),
        )
        if dropped and output_csv:
            failed_path = Path(output_csv).with_stem(Path(output_csv).stem + f"_failed_{datetime.now():%Y%m%d-%H%M%S}")
            pd.DataFrame(dropped, columns=["lat", "lon"]).to_csv(failed_path, index=False)
            LOGGER.error("Dropped locations written to %s; resubmit with --locations-csv '%s'", failed_path, failed_path)
        raise
    finally:
        _WORKER.clear()


class _RowWriter:
    """Appends each finished row to the output CSV, for recovery after a crash."""

    def __init__(self, output_csv: str | Path | None):
        self.handle = None
        self.header_written = False
        if output_csv:
            path = repo_path(output_csv)
            path.parent.mkdir(parents=True, exist_ok=True)
            self.handle = open(path, "w", newline="", encoding="utf-8")

    def write(self, row: dict) -> None:
        if self.handle is None:
            return
        pd.DataFrame([row]).to_csv(self.handle, index=False, header=not self.header_written)
        self.header_written = True
        self.handle.flush()

    def close(self) -> None:
        if self.handle is not None:
            self.handle.close()


def order_result_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Headline, land and water, capacities, cost shares, interest rates, then the rest."""
    if df.empty:
        return df
    currency = str(df["currency"].iloc[0]).strip().lower() if "currency" in df.columns else "usd"
    headline = [
        "latitude", "longitude", "country", "scenario_id", "run_id", "manifest_sha256",
        "land_constraint", "capacity_rule", "land_allocation", "temporal_accounting_mode", "ramp_limit_basis",
        "currency", f"lcoa_{currency}_per_t", f"lcoa_plant_{currency}_per_t",
        "annual_ammonia_demand_mwh", "annual_ammonia_production_t", "gridless_ammonia_production_t",
        f"total_cost_{currency}_per_year", f"plant_objective_cost_{currency}_per_year",
        "site_costs_in_headline", "site_cost_source_currency", "site_cost_source_to_output_fx",
        "build_cost_multiplier", "build_cost_pct", "tech_cost_pct", "om_cost_pct", "interest_pct",
        "water_cost_pct", "land_cost_pct", "other_cost_pct", "plant_cost_pct",
        "headline_cost_share_total_pct", "headline_cost_share_residual_pct",
        f"headline_cost_identity_residual_{currency}_per_year", "cost_percentage_basis",
        "uses_grid_backstop", "grid_energy_mwh", "grid_energy_share", "grid_energy_reference_mwh",
        "grid_energy_reference_basis", "grid_energy_tolerance_mwh", "grid_free_result_status",
        "is_gridless_feasible", "simulated_hours", "is_full_year_result",
    ]
    land_and_water = [
        "water_cost_usd_per_m3", "water_usage_m3_per_t_nh3", f"water_cost_{currency}_per_t",
        "land_cost_usd_per_km2_year", "land_used_km2", f"land_cost_{currency}_per_t", "land_capacity_cap_mw",
        "land_onshore_pct", "offshore_sea_pct", "land_cell_area_km2", "onshore_area_km2", "offshore_area_km2",
        "elevation_m", "area_cap_mw",
        "scaled_design_max_onshore_ammonia_capacity_t", "scaled_design_max_onshore_ammonia_capacity_mtpa",
        "scaled_design_max_gridless_onshore_ammonia_capacity_t", "scaled_design_max_gridless_onshore_ammonia_capacity_mtpa",
        "scaled_design_renewable_capacity_scale_factor", "scaled_design_limiting_constraint",
        "preferred_supplier_capacity_column", "renewable_union_area_source",
        "renewable_union_area_is_lower_bound_approximation", "renewable_union_area_method",
        "renewable_union_area_method_version", "wind_mw_per_t_nh3", "solar_mw_per_t_nh3",
        "protected_area_pct", "slope_suitable_land_pct", "steep_slope_pct", "land_exclusion_factor",
        "land_competition_fraction", "constrained_onshore_area_km2", "solar_area_used_km2", "wind_area_used_km2",
        "wind_onshore_area_km2", "wind_offshore_area_km2", "wind_density_mw_per_km2", "solar_density_mw_per_km2",
        "solar_tracking_density_mw_per_km2", "renewable_union_area_km2", land.CLASSWISE_NESTED_UNION_AREA_COLUMN,
        "max_power_wind_mw", "max_power_solar_mw",
    ]
    capacities = [
        "wind_mw", "solar_mw", "solar_tracking_mw", "grid_mw", "ramp_dummy_mw", "electrolysis_mw",
        "hydrogen_compression_mw", "hydrogen_from_storage_mw", "ammonia_synthesis_mw", "battery_pcs_mw",
        "hydrogen_fuel_cell_mw", "penalty_link_mwh", "ammonia_mwh", "compressed_hydrogen_store_mwh",
        "battery_storage_mwh", "accumulated_penalty_mwh", "hydrogen_storage_capacity_t",
    ]
    component_costs = [
        column for column in df.columns
        if (column.startswith("tech_share_") or column.startswith("lcoa_tech_")) and column != "tech_share_other_pct"
    ]
    interest = [column for column in df.columns if column.startswith("interest_rate_")]
    ordered: list[str] = []
    for section in (headline, land_and_water, capacities, component_costs, interest, ["interest_overrides_applied"]):
        ordered.extend(column for column in section if column in df.columns and column not in ordered)
    return df[ordered + [column for column in df.columns if column not in ordered]]


class _ProgressTracker:
    def __init__(self, total: int | None):
        self.total = total
        self.count = 0
        self.bar_width = 28
        self.inline = bool(getattr(sys.stderr, "isatty", lambda: False)())
        # Outside a terminal (notebooks, Slurm logs) print about every 1 %
        self.every = max(1, self.total // 100) if self.total else 1

    def update(self, lat: float, lon: float, status: str = "") -> None:
        self.count += 1
        stream = sys.stderr if self.inline else sys.stdout
        if self.total:
            filled = int(self.bar_width * self.count / self.total)
            bar = "#" * filled + "-" * (self.bar_width - filled)
            prefix = f"[{bar}] {self.count}/{self.total} ({self.count / self.total * 100.0:5.1f}%)"
        else:
            prefix = f"[{self.count}]"
        message = f"{prefix} {(status.upper() if status else ''):>8} lat={lat:7.2f} lon={lon:7.2f}"
        if self.total and self.inline:
            stream.write("\r" + message + ("\n" if self.count >= self.total else ""))
        elif (
            self.total is None or self.count == 1 or self.count >= self.total
            or self.count % self.every == 0 or (status and status.lower() != "done")
        ):
            stream.write(message + "\n")
        stream.flush()


@contextmanager
def _quiet_logging(enabled: bool) -> Iterator[None]:
    """Silence the solver and PyPSA loggers so the progress bar stays readable."""
    if not enabled:
        yield
        return
    previous_disable = logging.root.manager.disable
    logging.disable(logging.WARNING)
    names = [
        "gurobipy", "linopy", "linopy.constants", "linopy.io", "linopy.model", "model.weather", "model.surface",
        "pypsa", "pypsa.network", "pypsa.network.io", "pypsa.optimization", "pypsa.optimization.constraints",
        "pypsa.optimization.optimize",
    ]
    previous_levels = {name: logging.getLogger(name).level for name in names}
    for name in names:
        logging.getLogger(name).setLevel(logging.ERROR)
    try:
        yield
    finally:
        for name, level in previous_levels.items():
            logging.getLogger(name).setLevel(level)
        logging.disable(previous_disable)


@contextmanager
def _suppress_solver_streams(enabled: bool) -> Iterator[None]:
    if not enabled:
        yield
        return
    with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
        yield


@contextmanager
def _override_env(var: str, value: str, enabled: bool) -> Iterator[None]:
    if not enabled:
        yield
        return
    previous = os.environ.get(var)
    os.environ[var] = value
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop(var, None)
        else:
            os.environ[var] = previous


def load_locations_csv(path: str | Path) -> list[tuple[float, float]]:
    """(lat, lon) pairs from a CSV with lat and lon columns."""
    frame = pd.read_csv(path)
    for column in ("lat", "lon"):
        if column not in frame.columns:
            raise ValueError(f"Locations CSV {path} must include '{column}' column")
    return list(zip(frame["lat"], frame["lon"]))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the ammonia plant over many cells (a surface).")
    parser.add_argument("--land-csv", required=True, help="Centred land table (cell_anchor=center).")
    parser.add_argument("--locations-csv", help="CSV with lat,lon columns to restrict the run.")
    parser.add_argument("--limit", type=int, help="Run only the first N locations.")
    parser.add_argument("--tech-yaml", help=f"Technology YAML (default {DEFAULT_TECH_YAML.name}).")
    parser.add_argument("--plant-dir", help=f"Plant bundle (default {DEFAULT_PLANT_DIR.name}).")
    parser.add_argument("--override-csv", help="Per-cell override table (interest rates, build multipliers, water).")
    parser.add_argument("--weather-dir", help="Weather profile folder (default the runs.yaml run_defaults folder).")
    parser.add_argument("--output-csv", default="results/surface.csv", help="Where to write the result table.")
    parser.add_argument("--quiet", action="store_true", help="Silence the solver and PyPSA logs.")
    parser.add_argument("--max-snapshots", type=int, help="Solve only the first N snapshots (quick checks).")
    parser.add_argument("--threads-per-worker", type=int, help="Solver threads per worker.")
    parser.add_argument("--num-workers", type=int, default=1, help="Parallel worker processes (1 = serial).")
    parser.add_argument("--lon-min", type=float)
    parser.add_argument("--lon-max", type=float)
    parser.add_argument("--fail-fast", action="store_true", help="Stop at the first failing cell.")
    parser.add_argument("--time-step", type=float, default=1.0, help="Hours per snapshot (1 or 4 in the runs).")
    parser.add_argument("--pv-policy", choices=plant.PV_POLICIES, default="both", help="Which PV the plant may build.")
    parser.add_argument("--land-constraint", choices=LAND_CONSTRAINTS, default="after_solve")
    parser.add_argument("--capacity-rule", choices=CAPACITY_RULES, default="scaled_reference_design")
    parser.add_argument("--land-allocation", choices=land.LAND_ALLOCATIONS, default="colocated")
    parser.add_argument("--temporal-accounting-mode", choices=plant.TEMPORAL_ACCOUNTING_MODES, default="snapshot_weighted")
    parser.add_argument("--ramp-limit-basis", choices=plant.RAMP_LIMIT_BASES, default="per_hour")
    parser.add_argument("--exclude-site-costs", dest="include_site_costs", action="store_false",
                        help="Keep water and land out of the headline (plant-gate comparison).")
    args = parser.parse_args(argv)
    locations = load_locations_csv(args.locations_csv) if args.locations_csv else None
    if locations is not None and args.limit is not None:
        locations = locations[: args.limit]
    settings = RunSettings(
        time_step=args.time_step, max_snapshots=args.max_snapshots, pv_policy=args.pv_policy,
        land_constraint=args.land_constraint,
        capacity_rule=args.capacity_rule, land_allocation=args.land_allocation,
        temporal_accounting_mode=args.temporal_accounting_mode, ramp_limit_basis=args.ramp_limit_basis,
        include_site_costs=args.include_site_costs,
    )
    frame = run_surface(
        locations, land_csv=args.land_csv, tech_yaml=args.tech_yaml, plant_dir=args.plant_dir,
        override_csv=args.override_csv, weather_dir=args.weather_dir, settings=settings,
        output_csv=args.output_csv, quiet=args.quiet, threads_per_worker=args.threads_per_worker,
        num_workers=args.num_workers, lon_min=args.lon_min, lon_max=args.lon_max, fail_fast=args.fail_fast,
    )
    print(f"Processed {len(frame)} locations. Results saved to {args.output_csv}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
