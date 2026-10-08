"""The green-lory command line.

    green-lory list                      every run and network of runs.yaml, one line each
    green-lory show <id>                 one entry (also a report name such as D or A2)
    green-lory table                     the green-lory runs' settings side by side (Markdown)
    green-lory status [<id>]             the execution record (run_store.csv)
    green-lory path <id>                 where a run's output is
    green-lory run <id> --cell=-23,117   a defined run at a few cells, here on this machine
    green-lory map <id> --output m.html  a map of a run's result (LCOA by default; .png for a static map)
    green-lory site-costs --land <table> --output <csv>   the site-cost table of a land build
    green-lory shell <id>                shell assignments for arc/submit_lory_sequence.sh

`python -m model.cli ...` does the same without installing the package. Building land tables
and running whole surfaces have their own entry points (`python -m model.land_table`,
`python -m model.surface`, and the ARC jobs).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

from . import runs


def _cells(values: list[str]) -> list[tuple[float, float]]:
    cells = []
    for value in values:
        lat, lon = (float(part) for part in value.split(","))
        cells.append((lat, lon))
    return cells


def _run(args: argparse.Namespace) -> int:
    from .surface import RunSettings, load_locations_csv, run_surface

    arguments = runs.surface_arguments(args.id, campaigns_dir=args.campaigns_dir)
    cells = _cells(args.cell) + (load_locations_csv(args.locations_csv) if args.locations_csv else [])
    if not cells:
        raise runs.RunDefinitionError("give --cell lat,lon (repeatable) or --locations-csv")
    settings = arguments.pop("settings")
    if args.week:
        settings = RunSettings(**{**settings.__dict__, "max_snapshots": int(168 / settings.time_step)})
    frame = run_surface(cells, settings=settings, output_csv=args.output, quiet=not args.verbose,
                        num_workers=args.workers, threads_per_worker=args.threads, fail_fast=True, **arguments)
    columns = [column for column in ("latitude", "longitude", "country", "lcoa_eur_per_t", "wind_mw", "solar_mw",
                                     "solar_tracking_mw", "scaled_design_max_gridless_onshore_ammonia_capacity_mtpa")
               if column in frame.columns]
    print(frame[columns].to_string(index=False))
    if args.output:
        print(f"Full results: {args.output}")
    return 0


def _map(args: argparse.Namespace) -> int:
    from . import results

    frame = results.load_surface(args.id)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.suffix == ".html":
        results.plot_map(frame, color_column=args.column).write_html(output)
    else:
        import matplotlib
        matplotlib.use("Agg")
        figure, _ = results.plot_map_static(frame, color_column=args.column)
        figure.savefig(output, dpi=200)
    print(f"Map: {output}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="green-lory", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs", type=Path, default=runs.RUNS_FILE, help="run-definition file (default: runs.yaml)")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("list", help="every run and network")
    show = commands.add_parser("show", help="one entry of runs.yaml")
    show.add_argument("id", help="run id, network id or report name (for example D or A2)")
    commands.add_parser("table", help="the green-lory runs' settings side by side")
    status = commands.add_parser("status", help="the execution record")
    status.add_argument("id", nargs="?", help="one run id (default: every run)")
    path = commands.add_parser("path", help="where a run's output is")
    path.add_argument("id")
    run = commands.add_parser("run", help="a defined run at a few cells, on this machine")
    run.add_argument("id")
    run.add_argument("--cell", action="append", default=[], metavar="LAT,LON", help="a cell, written --cell=-23,117 (repeatable)")
    run.add_argument("--locations-csv", help="CSV with lat,lon columns")
    run.add_argument("--week", action="store_true", help="solve the first week only (a quick check)")
    run.add_argument("--output", help="write the full result table here")
    run.add_argument("--campaigns-dir", type=Path, default=runs.LOCAL_CAMPAIGNS_DIR,
                     help="folder holding the run's land table (default results/campaigns)")
    run.add_argument("--workers", type=int, default=1)
    run.add_argument("--threads", type=int, default=None, help="solver threads per worker")
    run.add_argument("--verbose", action="store_true", help="show the solver and PyPSA logs")
    map_ = commands.add_parser("map", help="a map of a run's result")
    map_.add_argument("id")
    map_.add_argument("--column", default="auto", help="result column to show (default: the LCOA column)")
    map_.add_argument("--output", required=True, help=".html for an interactive map, .png or .pdf for a static one")
    from . import site_costs

    site = commands.add_parser("site-costs", help="the site-cost table of a land build (model/site_costs.py)")
    site_costs.add_arguments(site)
    shell = commands.add_parser("shell", help="shell assignments for the ARC wrapper")
    shell.add_argument("id")
    args = parser.parse_args(argv)

    try:
        if args.command == "run":
            return _run(args)
        if args.command == "map":
            return _map(args)
        if args.command == "site-costs":
            from . import site_costs

            return site_costs.run(args)
        definitions = runs.load_runs(args.runs)
        networks = runs.load_networks(args.runs)
        if args.command == "list":
            for run_id, entry in definitions.items():
                print(f"{run_id}\t{entry['runner']}\t{entry.get('role', '')}")
            for network_id, network in networks.items():
                print(f"{network_id}\tnetwork{runs._report_suffix(network)}\t{network.get('role', '')}")
        elif args.command == "show":
            print(runs.describe_entry(args.id, definitions, networks))
        elif args.command == "table":
            print(runs.markdown_table(definitions, networks))
        elif args.command == "status":
            rows = runs.load_run_store()
            if args.id:
                entry_id, _ = runs.find_entry(args.id, definitions, networks)
                rows = [row for row in rows if row["run_id"] == entry_id]
            for row in rows:
                print(f"{row['run_id']}\t{row['status']}\t{row['release']}\t{row['location']}")
        elif args.command == "path":
            entry_id, _ = runs.find_entry(args.id, definitions, networks)
            print(runs.run_record(entry_id)["location"])
        else:
            print(runs.shell_assignments(args.id, definitions))
    except (runs.RunDefinitionError, OSError, yaml.YAMLError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
