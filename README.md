# green-lory

green-lory sizes a green-ammonia plant (wind, fixed or tracking PV, battery, electrolysis,
hydrogen compression and storage, fuel cell, Haber-Bosch) at every 1-degree cell of the world
with PyPSA, and reports the levelised cost of ammonia (LCOA) and the ammonia capacity the
cell's land can host. Its supplier tables feed the green-porpoise shipping-network model.

> **Status, 6 October 2026.** The reconciliation with the legacy plant model (legacy-lcoa,
> Salmon 2023) is complete; the modelling choices of the improved model are being finalised in
> [MODEL_CHOICES.md](MODEL_CHOICES.md), and the repository will be reorganised after that
> ([Planned layout](#planned-layout)). Green-lory capacities computed before 23 September 2026
> are superseded: those land tables were mislabelled by one degree. The May-era directories
> directly under `results/` and the notebooks predate the reconciliation.

## Where to start

| You want to | Look at |
|---|---|
| know what a run is, or what "network D" means | [runs.yaml](runs.yaml), or `python arc/run_definitions.py show D` |
| look up an option, id token, term or result column | [GLOSSARY.md](GLOSSARY.md) |
| see when and where a run ran, and its headline result | [reconciliation/run_store.csv](reconciliation/run_store.csv) |
| see which modelling choices are settled and which are open | [MODEL_CHOICES.md](MODEL_CHOICES.md) |
| understand why legacy-lcoa and green-lory gave different networks | [reconciliation/README.md](reconciliation/README.md), then `DECISION_20260916.md` there |
| run the legacy model | [reconciliation/legacy_lcoa/README.md](reconciliation/legacy_lcoa/README.md) |
| run on ARC | [arc/README.md](arc/README.md) |
| find or check an input dataset | [data/README.md](data/README.md) |
| read about the model code and the spatial cost inputs | [model/README.md](model/README.md) |

## Repository layout

| Path | Contents |
|---|---|
| `model/` | the green-lory model: `main.py` (one plant, one cell), `run_global.py` (many cells, results table), `land_processing.py` (land tables), `land_capacity.py` and `land_union.py` (land budget and capacity rules), `result_accounting.py` (LCOA and site costs) |
| `basic_ammonia_plant*/` | plant bundles: PyPSA component tables (GLOSSARY, Plant bundles) |
| `inputs/` | technology and cost YAMLs, finance (WACC) override CSVs, cell lists |
| `runs.yaml` | run definitions: every setting of every run and network we keep |
| `arc/` | ARC job templates, campaign wrappers, release and QA tools |
| `reconciliation/` | the legacy-lcoa versus green-lory reconciliation: reports, decision, run register (`RUNS.md`), run store, network replay, land pilots, and the legacy-lcoa package (`legacy_lcoa/`) |
| `notebooks/` | pre-reconciliation notebooks (see below) |
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
python -m pytest -q   # about 200 tests, under a minute
```

The legacy model needs its own environment (PyPSA 0.25.1 with Pyomo 6.5.0); see its README.

## Running

**One or a few cells locally.** `model/run_global.py` needs a technology YAML, a plant bundle
and a centred land table (`cell_anchor=center`; the accepted tables are on ARC under
`green-lory-campaigns/land_center_20260923_v1/`, fetch one first). For the settings of a defined
run, read its entry with `python arc/run_definitions.py show <run id>`. Example with the
improved base run's settings (finance override omitted, so the YAML's WACC applies) for one week
of weather at the cells listed in a `lat,lon` CSV:

```bash
python -m model.run_global \
  --tech-yaml inputs/tech_config_ammonia_plant_2050_dea_colocated.yaml \
  --plant-dir basic_ammonia_plant_2050 \
  --land-csv <path>/max_capacities_center_20pct_slope15.csv \
  --locations-csv <cells.csv> --max-snapshots 168 \
  --land-allocation colocated --no-ensure-feasibility \
  --output-csv results/scratch/smoke.csv --quiet
```

`--no-ensure-feasibility` matters: the command line keeps the grid backstop by default, while
every campaign run removes it.

**A defined run on ARC.** Runs are defined in `runs.yaml` and submitted by id from an immutable
source release, in stages (smoke, diagnostic, global); see `arc/README.md`:

```bash
bash arc/submit_lory_sequence.sh --dry-run --stage smoke --campaign <campaign> \
  --scenario gl_dea2050_wacc5_bflat_wflat_land20c_fixed --land-csv <centred 20 % table>
```

A global (52,702-cell) surface takes about 3 hours on four 48-core jobs.

**The legacy model.** See `reconciliation/legacy_lcoa/README.md` for the end-to-end commands.

**Notebooks.** `00_tech_config` (YAML to plant bundle), `01_max_capacities`,
`02_spatial_cost_inputs`, `03_single_site_run`, `04_global_run` and `05_run_analysis` date from
May 2026: they default to the 2030 YAML, the 10 Mt bundle and a mislabelled land table, and
`05` reads the May result directories. `02_single_site_run` and `03_global_run` duplicate
`03_single_site_run` and `04_global_run`. All of them will be replaced (next section).

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

- `model/` holds the green-lory code and the plant bundles, installable as a library;
- `legacy_lcoa/` moves to the top level as the self-contained legacy package;
- a workflow (Snakemake is the candidate) defines how each run in `runs.yaml` is produced,
  locally or on ARC;
- the notebooks only load and inspect results, each naming the workflow target it reads;
- `reconciliation/` is replaced by one notebook comparing the legacy and green-lory results and
  explaining the differences; its reports stay in git history.

The working checklist is `reconciliation/CLEANUP_PLAN_20261005.md`. `DEVELOPMENT_NOTES.md` is
the pre-reconciliation development log.
