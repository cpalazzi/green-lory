"""Global orchestration helpers for looping the ammonia plant model over many locations."""
from __future__ import annotations

import argparse
import io
import logging
import os
import sys
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Sequence, Tuple

import geopandas as gpd
import pandas as pd
from shapely.geometry import Point
import yaml

try:
    from . import main as plant_main
    from . import location_tools as lt
    from . import data_store as results_store
except ImportError:  # pragma: no cover - fallback for direct execution
    PACKAGE_ROOT = Path(__file__).resolve().parent
    if str(PACKAGE_ROOT) not in sys.path:
        sys.path.insert(0, str(PACKAGE_ROOT))
    import main as plant_main  # type: ignore
    import location_tools as lt  # type: ignore
    import data_store as results_store  # type: ignore

LOGGER = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_WEATHER_DIR = REPO_ROOT / "data"
DEFAULT_LAND_CSV = DEFAULT_WEATHER_DIR / "20251222_land_max_capacity.csv"
DEFAULT_INTEREST_CSV = REPO_ROOT / "inputs" / "example_finance_overrides.csv"
DEFAULT_TECH_YAML = REPO_ROOT / "inputs" / "tech_config_ammonia_plant.yaml"
RENEWABLES = ["wind", "solar", "solar_tracking"]
_LAT_LON_TOLERANCE = 0.125  # match within 1/8th degree
# Techno-economic inputs are now expected to be pre-processed into the CSV bundle.
# Interest-rate defaults are therefore not sourced from a runtime YAML file.
DEFAULT_INTEREST_RATES: Dict[str, float] = {}


def _annuity_factor(interest_rate: float, lifetime_years: float) -> float:
    if lifetime_years <= 0:
        raise ValueError("lifetime_years must be positive")
    if abs(interest_rate) < 1e-12:
        return 1.0 / lifetime_years
    factor = (1.0 + interest_rate) ** lifetime_years
    return interest_rate * factor / (factor - 1.0)


def _annualised_capital_cost(
    overnight_cost: float,
    interest_rate: float,
    lifetime_years: float,
    fixed_om_fraction: float,
) -> float:
    crf = _annuity_factor(float(interest_rate), float(lifetime_years))
    return float(overnight_cost) * (crf + float(fixed_om_fraction))


def _load_tech_inputs(path: str | Path | None) -> Dict[str, dict]:
    resolved = _resolve_path(path) or DEFAULT_TECH_YAML
    if not resolved.exists():
        LOGGER.warning("Tech YAML %s not found; financing overrides will not affect costs.", resolved)
        return {}
    data = yaml.safe_load(resolved.read_text()) or {}
    techs = data.get("techs")
    if not isinstance(techs, dict):
        LOGGER.warning("Tech YAML %s missing top-level 'techs' mapping; skipping overrides.", resolved)
        return {}
    out: Dict[str, dict] = {}
    for name, raw in techs.items():
        if not isinstance(raw, dict):
            continue
        normalized = plant_main.normalize_component_name(name)
        if normalized is None:
            continue
        out[normalized] = raw
    return out


def _apply_finance_overrides(
    network,
    tech_inputs: Dict[str, dict],
    overrides: Dict[str, Dict[str, float]] | None,
    aggregation_count: int,
    time_step: float,
) -> None:
    """Apply per-tech interest_rate overrides by recomputing capital_cost.

    Convention:
    - YAML overnight costs are quoted on an HHV output basis.
    - PyPSA Links are sized on an input basis (bus0 MW_in), so link capital_cost
      must be USD/MW_in/year.
    - Stores are scaled in generate_network() by time_step * aggregation_count;
      we mirror that scaling here so the objective remains consistent.
    """
    if not overrides or not tech_inputs:
        return

    for tech, params in overrides.items():
        rate = params.get("interest_rate")
        if rate is None:
            continue
        normalized = plant_main.normalize_component_name(tech)
        if normalized is None:
            continue
        raw = tech_inputs.get(normalized)
        if not raw:
            continue

        component_type = str(raw.get("component_type", "")).lower()
        lifetime_years = float(raw.get("lifetime_years", 20.0))
        fixed_om_fraction = float(raw.get("fixed_om_fraction", 0.0))

        if component_type in {"generator", "link"}:
            overnight = raw.get("overnight_cost_per_mw")
            if overnight is None:
                continue
            annual_out = _annualised_capital_cost(float(overnight), float(rate), lifetime_years, fixed_om_fraction)

            if component_type == "generator":
                if normalized in network.generators.index:
                    network.generators.loc[normalized, "capital_cost"] = annual_out
                continue

            # Link: YAML is MW_out on bus1; PyPSA needs MW_in on bus0.
            if normalized in network.links.index:
                efficiency = float(network.links.loc[normalized, "efficiency"])
                if efficiency <= 0:
                    raise ValueError(f"Link '{normalized}' has non-positive efficiency; cannot apply finance overrides")
                network.links.loc[normalized, "capital_cost"] = annual_out * efficiency

        elif component_type == "store":
            overnight = raw.get("overnight_cost_per_mwh")
            if overnight is None:
                continue
            annual = _annualised_capital_cost(float(overnight), float(rate), lifetime_years, fixed_om_fraction)
            if normalized in network.stores.index:
                network.stores.loc[normalized, "capital_cost"] = annual * float(time_step) * float(aggregation_count)



class _ProgressTracker:
    def __init__(self, total: int | None):
        self.total = total
        self.count = 0
        self.bar_width = 28

    def update(self, lat: float, lon: float, status: str = "") -> None:
        self.count += 1
        stream = sys.stderr
        if self.total and self.total > 0:
            filled = int(self.bar_width * self.count / self.total)
            bar = "#" * filled + "-" * (self.bar_width - filled)
            percent = self.count / self.total * 100.0
            prefix = f"[{bar}] {self.count}/{self.total} ({percent:5.1f}%)"
            newline = self.count >= self.total
        else:
            prefix = f"[{self.count}]"
            newline = True
        suffix = status.upper() if status else ""
        message = f"{prefix} {suffix:>8} lat={lat:7.2f} lon={lon:7.2f}"
        if self.total and self.total > 0:
            stream.write("\r" + message)
            if newline:
                stream.write("\n")
        else:
            stream.write(message + "\n")
        stream.flush()


def _resolve_path(value: str | Path | None) -> Path | None:
    if value is None:
        return None
    path = Path(value)
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path


@contextmanager
def _quiet_logging(enabled: bool):
    if not enabled:
        yield
        return
    previous_disable_level = logging.root.manager.disable
    # Disable WARNING and below so the progress bar doesn't get interleaved with log output.
    logging.disable(logging.WARNING)
    logger_names = [
        "gurobipy",
        "linopy",
        "linopy.constants",
        "linopy.io",
        "linopy.model",
        "model.location_tools",
        "model.run_global",
        "pypsa",
        "pypsa.network",
        "pypsa.network.io",
        "pypsa.optimization",
        "pypsa.optimization.constraints",
        "pypsa.optimization.optimize",
    ]
    previous_levels: Dict[str, int] = {}
    for name in logger_names:
        logger = logging.getLogger(name)
        previous_levels[name] = logger.level
        logger.setLevel(logging.ERROR)
    try:
        yield
    finally:
        for name, level in previous_levels.items():
            logging.getLogger(name).setLevel(level)
        logging.disable(previous_disable_level)


@contextmanager
def _suppress_solver_streams(enabled: bool):
    if not enabled:
        yield
        return
    stdout_buffer = io.StringIO()
    stderr_buffer = io.StringIO()
    with redirect_stdout(stdout_buffer), redirect_stderr(stderr_buffer):
        yield


@contextmanager
def _override_env(var: str, value: str, enabled: bool):
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


def _load_interest_table(path: str | Path | None) -> pd.DataFrame:
    resolved = _resolve_path(path)
    if resolved is None:
        return pd.DataFrame(columns=["lat", "lon", "tech", "interest_rate"])
    if not resolved.exists():
        LOGGER.warning("Interest CSV %s not found; continuing without overrides.", resolved)
        return pd.DataFrame(columns=["lat", "lon", "tech", "interest_rate"])
    df = pd.read_csv(resolved)
    column_map = {col.lower(): col for col in df.columns}
    for required in ("lat", "lon", "tech", "interest_rate"):
        if required not in column_map:
            raise ValueError(
                f"Interest CSV {resolved} must include '{required}' column (found {list(df.columns)})"
            )
    cleaned = df.rename(columns={column_map[k]: k for k in ("lat", "lon", "tech", "interest_rate")})
    cleaned["tech"] = cleaned["tech"].map(plant_main.normalize_component_name)
    return cleaned


def _load_land_table(path: str | Path | None) -> pd.DataFrame | None:
    resolved = _resolve_path(path)
    if resolved is None:
        return None
    if not resolved.exists():
        LOGGER.warning("Land-availability CSV %s not found; skipping land caps.", resolved)
        return None
    df = pd.read_csv(resolved)
    df.columns = [col.lower() for col in df.columns]
    expected = {"latitude", "longitude"}
    if not expected.issubset(df.columns):
        LOGGER.warning("Land CSV %s is missing %s", resolved, expected - set(df.columns))
        return None
    return df


def _read_country_file(path: str | Path) -> gpd.GeoDataFrame | None:
    try:
        frame = gpd.read_file(path)
    except Exception as exc:  # noqa: BLE001
        LOGGER.debug("Failed to read country boundaries from %s: %s", path, exc)
        return None
    rename_map = {}
    for candidate in ["ADMIN", "NAME", "name", "COUNTRY", "Country"]:
        if candidate in frame.columns:
            rename_map[candidate] = "country"
            break
    if rename_map:
        frame = frame.rename(columns=rename_map)
    if "country" not in frame.columns:
        LOGGER.debug("Country column missing in %s (columns=%s)", path, list(frame.columns))
        return None
    return frame[["country", "geometry"]]


def _interest_overrides(interest_df: pd.DataFrame, lat: float, lon: float) -> Dict[str, Dict[str, float]] | None:
    if interest_df.empty:
        return None
    mask = (interest_df["lat"] - lat).abs() <= _LAT_LON_TOLERANCE
    mask &= (interest_df["lon"] - lon).abs() <= _LAT_LON_TOLERANCE
    subset = interest_df.loc[mask]
    if subset.empty:
        return None
    overrides: Dict[str, Dict[str, float]] = {}
    for _, row in subset.iterrows():
        tech = row["tech"]
        overrides.setdefault(tech, {})["interest_rate"] = float(row["interest_rate"])
    return overrides or None


def _match_land_row(land_df: pd.DataFrame, lat: float, lon: float) -> pd.Series | None:
    if land_df is None:
        return None
    mask = (land_df["latitude"] - lat).abs() <= _LAT_LON_TOLERANCE
    mask &= (land_df["longitude"] - lon).abs() <= _LAT_LON_TOLERANCE
    subset = land_df.loc[mask]
    if subset.empty:
        return None
    return subset.iloc[0]


def _set_generator_cap(network, generator_name: str, cap_value: float) -> float:
    if cap_value is None or cap_value <= 0:
        return 0.0
    if generator_name not in network.generators.index:
        return 0.0
    if not bool(network.generators.at[generator_name, "p_nom_extendable"]):
        return 0.0
    network.generators.loc[generator_name, "p_nom_max"] = cap_value
    return cap_value


def _apply_land_caps(
    network, land_df: pd.DataFrame | None, lat: float, lon: float
) -> Tuple[float | None, pd.Series | None]:
    row = _match_land_row(land_df, lat, lon) if land_df is not None else None
    if row is None:
        return None, None
    total_cap = 0.0

    wind_cap = row.get("wind_max_capacity")
    if pd.notna(wind_cap):
        wind_cap_val = float(wind_cap)
        if wind_cap_val > 0:
            total_cap += wind_cap_val
            _set_generator_cap(network, "wind", wind_cap_val)

    solar_cap = row.get("solar_max_capacity")
    if pd.notna(solar_cap):
        solar_cap_val = float(solar_cap)
        if solar_cap_val > 0:
            total_cap += solar_cap_val
            _set_generator_cap(network, "solar", solar_cap_val)
            _set_generator_cap(network, "solar_tracking", solar_cap_val)

    if total_cap > 0:
        return total_cap, row

    fallback = row.get("max_capacity")
    if pd.notna(fallback):
        fallback_val = float(fallback)
        fallback_availability = row.get("availability", 1.0)
        if pd.notna(fallback_availability):
            fallback_val *= float(fallback_availability)
        if fallback_val > 0:
            for gen_name in ("wind", "solar", "solar_tracking"):
                _set_generator_cap(network, gen_name, fallback_val)
            return fallback_val, row

    return None, row


def _land_metadata_from_row(row: pd.Series | None) -> Dict[str, float]:
    if row is None:
        return {}

    metadata: Dict[str, float] = {}

    def _maybe(name: str, scale: float = 1.0) -> float | None:
        value = row.get(name)
        if pd.isna(value):
            return None
        return float(value) * scale

    onshore = _maybe("onshore_land_pct", 1.0)
    if onshore is None:
        onshore = _maybe("onshore_fraction", 100.0)
    if onshore is not None:
        metadata["land_onshore_pct"] = onshore

    area = _maybe("area")
    if area is not None:
        metadata["land_cell_area_km2"] = area

    max_capacity = _maybe("max_capacity")
    if max_capacity is not None:
        metadata["area_cap_mw"] = max_capacity

    solar_area = _maybe("solar_area_km2")
    if solar_area is not None:
        metadata["solar_area_used_km2"] = solar_area

    wind_area = _maybe("wind_area_km2")
    if wind_area is not None:
        metadata["wind_area_used_km2"] = wind_area

    return metadata


def _compose_interest_rates(
    overrides: Dict[str, Dict[str, float]] | None,
) -> Dict[str, float]:
    rates: Dict[str, float] = {k: v for k, v in DEFAULT_INTEREST_RATES.items() if v is not None}
    if overrides:
        for tech, params in overrides.items():
            rate = params.get("interest_rate")
            if rate is None:
                continue
            normalized = plant_main.normalize_component_name(tech)
            if normalized is None:
                continue
            rates[normalized] = float(rate)
    return rates


def _load_country_shapes() -> gpd.GeoDataFrame:
    fallback = REPO_ROOT / "data" / "countries.geojson"
    if not fallback.exists():
        raise FileNotFoundError(
            "Missing data/countries.geojson – add a Natural Earth admin-0 GeoJSON to tag locations."
        )
    frame = _read_country_file(fallback)
    if frame is None:
        raise ValueError(
            "Unable to parse data/countries.geojson. Ensure it includes a 'country'/'ADMIN' column and valid geometries."
        )
    return frame


def _country_for(world: gpd.GeoDataFrame, lat: float, lon: float) -> str:
    point = Point(lon, lat)
    matches = world[world.intersects(point)]
    if matches.empty:
        return "None"
    return matches.iloc[0].country


def _build_weather_frame(dataset: lt.all_locations, lat: float, lon: float, aggregation_count: int) -> pd.DataFrame:
    location = lt.renewable_data(
        dataset,
        latitude=lat,
        longitude=lon,
        renewables=RENEWABLES,
        aggregation_variable=aggregation_count,
    )
    frame = location.concat.copy()
    if "Weights" in frame.columns:
        frame = frame.drop(columns="Weights")
    if "solar_tracking" not in frame.columns and "solar" in frame.columns:
        frame["solar_tracking"] = frame["solar"]
    defaults = {"grid": 0.0, "ramp_dummy": 1.0}
    for column, default_value in defaults.items():
        if column not in frame.columns:
            frame[column] = default_value
    frame = frame.reset_index(drop=True)
    return frame


def _default_locations(land_df: pd.DataFrame | None) -> List[Tuple[float, float]]:
    if land_df is None:
        raise ValueError("A land-availability CSV is required when no explicit locations are supplied.")
    filtered = land_df[land_df.get("availability", 0) > 0]
    return list(zip(filtered["latitude"].astype(float), filtered["longitude"].astype(float)))


def run_global(
    locations: Sequence[Tuple[float, float]] | None = None,
    weather_dir: str | Path | None = None,
    land_csv: str | Path | None = None,
    interest_csv: str | Path | None = None,
    aggregation_count: int = 1,
    time_step: float = 1.0,
    max_snapshots: int | None = None,
    output_csv: str | Path | None = None,
    quiet: bool = False,
) -> pd.DataFrame:
    """Run the ammonia plant optimisation for every requested location.

    Args:
        locations: Iterable of (lat, lon) pairs. If omitted, the function iterates over every
            coordinate contained in the land-availability CSV.
        weather_dir: Directory containing the NetCDF stacks consumed by `location_tools`.
        land_csv: CSV with at least `Latitude` and `Longitude` columns plus any available
            capacity metadata.
        interest_csv: CSV with `lat`, `lon`, `tech`, `interest_rate` columns to override
            financing assumptions per technology.
        aggregation_count: Snapshot aggregation factor forwarded to the weather loader.
        time_step: Duration of each snapshot in hours.
        max_snapshots: Optional hard cap on the number of weather snapshots to simulate per
            location (useful for quick smoke tests and notebooks).
        output_csv: Optional destination for the aggregated results table.
    """
    with _quiet_logging(quiet), _override_env("GREEN_LORY_SOLVER_LOG", "0", quiet):
        weather_dir = _resolve_path(weather_dir) or DEFAULT_WEATHER_DIR
        dataset = lt.all_locations(str(weather_dir), cache_resources=True)
        land_df = _load_land_table(land_csv or DEFAULT_LAND_CSV)
        interest_df = _load_interest_table(interest_csv or DEFAULT_INTEREST_CSV)
        tech_inputs = _load_tech_inputs(DEFAULT_TECH_YAML)
        world = _load_country_shapes()
        store = results_store.Data_store()

        if locations is None:
            location_iterable: Iterable[Tuple[float, float]] = _default_locations(land_df)
        else:
            location_iterable = locations

        total_locations = len(location_iterable) if hasattr(location_iterable, "__len__") else None
        progress = _ProgressTracker(total_locations)

        for lat_value, lon_value in location_iterable:
            lat = float(lat_value)
            lon = float(lon_value)
            status = "run"
            try:
                weather_frame = _build_weather_frame(dataset, lat, lon, aggregation_count)
                if max_snapshots is not None:
                    weather_frame = weather_frame.iloc[:max_snapshots].copy()
            except Exception as exc:  # noqa: BLE001
                status = "weather"
                LOGGER.warning("Skipping location (%s, %s) due to weather extraction error: %s", lat, lon, exc)
                progress.update(lat, lon, status)
                continue

            overrides = _interest_overrides(interest_df, lat, lon)
            interest_rates = _compose_interest_rates(overrides)
            network = plant_main.generate_network(
                len(weather_frame),
                "basic_ammonia_plant",
                aggregation_count=aggregation_count,
                time_step=time_step,
                tech_config_overrides=overrides,
            )
            _apply_finance_overrides(
                network,
                tech_inputs=tech_inputs,
                overrides=overrides,
                aggregation_count=aggregation_count,
                time_step=time_step,
            )
            land_cap, land_row = _apply_land_caps(network, land_df, lat, lon)

            try:
                with _suppress_solver_streams(quiet):
                    results = plant_main.main(
                        n=network,
                        weather_data=weather_frame,
                        multi_site=True,
                        aggregation_count=aggregation_count,
                        time_step=time_step,
                        interest_rates=interest_rates,
                    )
            except Exception as exc:  # noqa: BLE001
                status = "solver"
                LOGGER.warning("Optimisation failed for (%s, %s): %s", lat, lon, exc)
                progress.update(lat, lon, status)
                continue

            if land_cap is not None:
                results["land_capacity_cap"] = land_cap
            if land_row is not None:
                results.update(_land_metadata_from_row(land_row))
            results["interest_overrides_applied"] = bool(overrides)

            country = _country_for(world, lat, lon)
            store.add_location(lat, lon, country, results)
            progress.update(lat, lon, "done")

        df = pd.DataFrame.from_dict(store.collated_results, orient='index')
        if output_csv:
            output_path = _resolve_path(output_csv)
            if output_path is None:
                output_path = Path(output_csv)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(output_path, index=False)
        return df


def _load_locations_csv(path: str | Path) -> List[Tuple[float, float]]:
    df = pd.read_csv(path)
    for column in ("lat", "lon"):
        if column not in df.columns:
            raise ValueError(f"Locations CSV {path} must include '{column}' column")
    return list(zip(df["lat"], df["lon"]))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the green ammonia model over many locations.")
    parser.add_argument("--locations-csv", type=str, help="CSV with columns lat,lon to restrict the sweep.")
    parser.add_argument("--interest-csv", type=str, default=None, help="CSV with lat,lon,tech,interest_rate overrides.")
    parser.add_argument("--land-csv", type=str, default=None, help="Optional override for the land availability CSV.")
    parser.add_argument("--output-csv", type=str, default="results/run_global.csv", help="Where to write aggregated results.")
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress solver and PyPSA logging when running large batches.",
    )
    parser.add_argument(
        "--max-snapshots",
        type=int,
        default=None,
        help="Limit the number of weather snapshots per location (useful for notebooks/smoke tests).",
    )
    parser.add_argument("--limit", type=int, default=None, help="Optional limit on the number of locations to process.")
    args = parser.parse_args()

    if args.locations_csv:
        requested_locations = _load_locations_csv(args.locations_csv)
        if args.limit is not None:
            requested_locations = requested_locations[: args.limit]
    else:
        requested_locations = None

    results_df = run_global(
        locations=requested_locations,
        interest_csv=args.interest_csv,
        land_csv=args.land_csv,
        max_snapshots=args.max_snapshots,
        output_csv=args.output_csv,
        quiet=args.quiet,
    )
    print(f"Processed {len(results_df)} locations. Results saved to {args.output_csv}.")
