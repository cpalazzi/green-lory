# green-lory

green-lory sizes a green-ammonia plant (wind, fixed or tracking PV, battery, electrolysis,
hydrogen compression and storage, fuel cell, Haber-Bosch) at every 1-degree cell of the world
with PyPSA, and reports the levelised cost of ammonia (LCOA) and the ammonia capacity the
cell's land can host. Its supplier tables feed the green-porpoise shipping-network model.

> **Status, 6 October 2026.** The reconciliation with the legacy plant model (legacy-lcoa,
> Salmon 2023) is complete: results and decision in
> [reconciliation/DECISION.md](reconciliation/DECISION.md). The modelling choices of the
> improved model are being finalised in [MODEL_CHOICES.md](MODEL_CHOICES.md); the library and
> workflow reorganisation follows ([Planned layout](#planned-layout)). Green-lory capacities
> computed before 23 September 2026 are superseded (their land tables were mislabelled by one
> degree), and the May-era directories directly under `results/` predate the reconciliation.

## Where to start

| You want to | Look at |
|---|---|
| see the current results and why legacy-lcoa and green-lory differ | [reconciliation/DECISION.md](reconciliation/DECISION.md) |
| know what a run is, or what "network D" means | [runs.yaml](runs.yaml), or `green-lory show D` |
| look up an option, id token, term or result column | [GLOSSARY.md](GLOSSARY.md) |
| see when and where a run ran, and its headline result | [run_store.csv](run_store.csv), or `green-lory status <run id>` |
| see which modelling choices are settled and which are open | [MODEL_CHOICES.md](MODEL_CHOICES.md) |
| run the legacy model | [legacy_lcoa/README.md](legacy_lcoa/README.md) |
| run on ARC | [arc/README.md](arc/README.md) |
| find or check an input dataset | [data/README.md](data/README.md) |
| read about the model code and the spatial cost inputs | [model/README.md](model/README.md) |

## Repository layout

| Path | Contents |
|---|---|
| `model/` | the green-lory model library: `technology` (YAML costs and efficiencies), `weather`, `plant` (one plant), `accounting`, `land` (land budgets and capacity rules), `land_table` (land tables), `surface` (many cells), `runs` and `cli` (the run definitions and the `green-lory` command), `results` (results by run id, tables, maps); `plants/ammonia_plant_1mt/` is the plant ([model/README.md](model/README.md)) |
| `inputs/` | technology and cost YAMLs, finance (WACC) override CSVs, cell lists |
| `runs.yaml` | run definitions: every setting of every run and network we keep |
| `run_store.csv` | the execution record: when, where and with what result each run ran |
| `legacy_lcoa/` | the legacy-lcoa model reproduced end to end: frozen source, harness, land step, capacity rule, supplier table |
| `networks/` | the green-porpoise shipping network: `replay` (the optimisation), `contracts` (supplier tables from surfaces), `compare` |
| `reconciliation/` | decision, comparison scripts, supply pilot, and the September reports in `archive/` |
| `arc/` | ARC release staging, campaign wrappers, Slurm templates, QA tools |
| `notebooks/` | `02_spatial_cost_inputs` (inputs of the spatial runs; replaced by `model/site_costs.py` in campaign step 3) |
| `scripts/` | input builders (`build_ameli_wacc_inputs.py`, `fetch_travel_time.py`) |
| `tests/` | `pytest` suite |
| `data/` | large inputs; only the layout and README are tracked |
| `results/` | local outputs; current evidence under `results/campaigns/` |

## Setup

Python 3.11 or later, a linear-programming solver supported by linopy (Gurobi on ARC; HiGHS
works locally), and the packages in `requirements.txt` (the pinned environment). Installing the
repository as a package adds the `green-lory` command:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e . --no-deps
python -m pytest -q
green-lory list
```

The legacy model needs its own environment (PyPSA 0.25.1 with Pyomo 6.5.0); see its README.

## Running

**A defined run at one or a few cells.** `green-lory run` takes every setting from
`runs.yaml` and checks the pinned hashes of the land table and finance file (local copies
under `results/campaigns/`). The key run at the Pilbara cell for one week:

```bash
green-lory run gl_dea2050_wacc5_bflat_wflat_land20c_fixed --cell=-23,117 --week
```

**Ad hoc settings.** `model/surface.py` needs a centred land table (`cell_anchor=center`; the
accepted tables are listed in `data/README.md`). Its defaults are the improved runs'
technology YAML and the plant bundle. One week of weather at the cells of a `lat,lon` CSV,
fixed-tilt PV only as in the key runs:

```bash
python -m model.surface \
  --land-csv results/campaigns/land_center_20260923_v1/max_capacities_center_20pct_slope15.csv \
  --locations-csv <cells.csv> --max-snapshots 168 --pv-policy fixed \
  --output-csv results/scratch/smoke.csv --quiet
```

Without `--override-csv` the YAML's own WACC applies. The regression test runs the three focal
cells of the key run and the reconciling run against the accepted results:
`python -m pytest -m slow` (needs Gurobi and the data).

**A defined run on ARC.** Stage a release (`bash arc/stage_release.sh <release id>`), then
submit the run by id from it, in smoke, diagnostic and global stages
(`arc/submit_lory_sequence.sh`); see `arc/README.md`. A 52,702-cell hourly surface took about
3 hours on four 48-core jobs.

**The legacy model.** See `legacy_lcoa/README.md` for the end-to-end commands and expected
values.

## Results

Every campaign result has its own immutable directory with a manifest, shards, the merged table
and a QA report:

```text
green-lory-campaigns/<campaign>/<results class>/<run id>/runs/<execution id>/<stage>/   (ARC)
results/campaigns/<campaign>/...                                                        (local copies)
```

`run_store.csv` gives the location of each run's output (`green-lory path <run id>`). `model/results.py` reads
outputs by run id from the local copies under `results/campaigns/`
(`results.load_surface(run_id)`, `results.focal_cell_table(...)`, coast-against-inland tables),
and `green-lory map <run id> --output lcoa.html` (or `.png`) draws a run's LCOA.

## Planned layout

Once the modelling choices are settled (`MODEL_CHOICES.md`):

- `model/` becomes an installable package, with the logic now in notebooks (plant-bundle
  compilation, spatial cost inputs) moved into functions;
- one command keyed on run ids runs and finds results: `run <id> --stage ...` (wrapping the ARC
  wrapper), `status` (what exists, what is missing, what is out of date because its `runs.yaml`
  entry changed since it ran) and `path <id>`;
- the notebooks only load and inspect results by run id, each naming the command that produces
  its inputs;
- `reconciliation/` is replaced by one notebook comparing the legacy and green-lory results.

A workflow manager (Snakemake) is not planned: the pipeline has few, expensive steps that are
submitted one at a time after review, and its provenance is already recorded by the releases,
manifests and run store. It can be layered on the same command later if the run list grows to
dozens of surfaces.
