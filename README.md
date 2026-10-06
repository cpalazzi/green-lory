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
| know what a run is, or what "network D" means | [runs.yaml](runs.yaml), or `python arc/run_definitions.py show D` |
| look up an option, id token, term or result column | [GLOSSARY.md](GLOSSARY.md) |
| see when and where a run ran, and its headline result | [reconciliation/run_store.csv](reconciliation/run_store.csv) |
| see which modelling choices are settled and which are open | [MODEL_CHOICES.md](MODEL_CHOICES.md) |
| run the legacy model | [legacy_lcoa/README.md](legacy_lcoa/README.md) |
| run on ARC | [arc/README.md](arc/README.md) |
| find or check an input dataset | [data/README.md](data/README.md) |
| read about the model code and the spatial cost inputs | [model/README.md](model/README.md) |

## Repository layout

| Path | Contents |
|---|---|
| `model/` | the green-lory model: `main.py` (one plant), `run_global.py` (many cells, results table), `land_processing.py` (land tables), `land_capacity.py` and `land_union.py` (land budget and capacity rules), `result_accounting.py` (LCOA and site costs); `plants/` holds the plant bundles |
| `inputs/` | technology and cost YAMLs, finance (WACC) override CSVs, cell lists |
| `runs.yaml` | run definitions: every setting of every run and network we keep |
| `legacy_lcoa/` | the legacy-lcoa model reproduced end to end: frozen source, harness, land step, capacity rule, supplier table |
| `reconciliation/` | decision, run store, green-porpoise network replay, supplier-contract export, comparison scripts, supply pilot, and the September reports in `archive/` |
| `arc/` | ARC release staging, campaign wrappers, Slurm templates, QA tools |
| `notebooks/` | `00_tech_config` (YAML to plant bundle) and `02_spatial_cost_inputs` (inputs of the spatial runs) |
| `scripts/` | input builders (`build_ameli_wacc_inputs.py`, `fetch_travel_time.py`) |
| `tests/` | `pytest` suite |
| `data/` | large inputs; only the layout and README are tracked |
| `results/` | local outputs; current evidence under `results/campaigns/` |

## Setup

Python 3.11 or later, a linear-programming solver supported by linopy (Gurobi on ARC; HiGHS
works locally), and the packages in `requirements.txt`:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m pytest -q
```

The legacy model needs its own environment (PyPSA 0.25.1 with Pyomo 6.5.0); see its README.

## Running

**One or a few cells locally.** `model/run_global.py` needs a centred land table
(`cell_anchor=center`; the accepted tables are listed in `data/README.md`). Its defaults are
the improved runs' technology YAML and plant bundle; for another run, read its settings with
`python arc/run_definitions.py show <run id>`. One week of weather at the cells of a `lat,lon`
CSV:

```bash
python -m model.run_global \
  --land-csv results/campaigns/land_center_20260923_v1/max_capacities_center_20pct_slope15.csv \
  --locations-csv <cells.csv> --max-snapshots 168 --no-ensure-feasibility \
  --output-csv results/scratch/smoke.csv --quiet
```

`--no-ensure-feasibility` matters: the command line keeps the grid backstop by default, while
every campaign run removes it. Without `--override-csv` the YAML's own WACC applies.

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

`reconciliation/run_store.csv` gives the location of each run's output.

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
