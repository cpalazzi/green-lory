# reconciliation/: legacy-lcoa versus green-lory, and the shipping networks

The reconciliation of the old `lcoa-opt` plant model ("legacy-lcoa", Salmon 2023) with
green-lory, and of the green-porpoise networks each produces, is complete. Start with
[DECISION.md](DECISION.md): the answer, what changed in green-lory, the three focal cells and
the networks. Once the open modelling choices (`../MODEL_CHOICES.md`) are settled, this
directory will be replaced by one notebook comparing the legacy and green-lory results; git
history keeps everything.

## Contents

| Path | Role |
|---|---|
| [DECISION.md](DECISION.md) | Results and decision (6 October 2026) |
| `compare_land_builds.py` | Cell-by-cell comparison of a green-lory land table with the legacy land step |
| `focal_cells.py` | Land, LCOA and capacity of every kept run at the three focal cells (DECISION.md section 3) |
| `coast_inland.py` | Coast against inland Australia: LCOA by distance to the coast for each model step, network production by band, effect of the spatial water and remoteness inputs (DECISION.md section 5) |
| `supply_pilot/` | Finite-site supply curves at the focal cells: the plant re-solved at fixed quantities with land enforced in the solve (ARC job `../arc/jobs/06_land_supply_pilot.sh`) |
| `archive/2026-09/` | The September reports, run log (`RUNS.md`), id grammar and land experiments, unedited; they name networks by letter (`../GLOSSARY.md` maps the letters) |
| `archive/pre-reconciliation/` | The development log of March to June 2026 |

The legacy model itself is in `../legacy_lcoa/`; the network replay, supplier contracts and
network comparison are in `../networks/`, and the execution record is `../run_store.csv`.

## Evidence

Generated evidence is under `../results/campaigns/`, never here; `../run_store.csv` gives each
run's directory. The consolidated tables behind DECISION.md are in
`../results/campaigns/reconciliation_final_20260916_v1/` (`focal_cells_v3/`, `networks_v3/`).

## Working rules

Every run gets a new output directory with a manifest and input hashes, and nothing is
accepted without a passed QA record. Original source data are read-only inputs. Land fractions
and constants are never tuned to reproduce a map: residuals are reported as they are.
