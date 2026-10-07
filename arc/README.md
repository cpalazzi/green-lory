# arc/: running on Oxford ARC

Everything that runs on ARC runs from an **immutable source release** and writes to a **new
campaign directory**; nothing is overwritten, and only a passed QA record makes a result
accepted. Run definitions are in `../runs.yaml` (`python3 -m model.cli list`).

| Location on ARC | Contents |
|---|---|
| `/data/engs-df-green-ammonia/engs2523/green-lory-releases/<id>/` | read-only source releases with `source_inventory.json` and `RELEASE_NOTES.md` |
| `/data/engs-df-green-ammonia/engs2523/green-lory-campaigns/<campaign>/` | outputs: land tables, surfaces, contracts, networks |
| `/data/engs-df-green-ammonia/engs2523/green-lory/data/` | shared inputs linked into every release (weather stack, MODIS, GEBCO, WDPA, bathymetry, countries) |
| `/data/engs-df-green-ammonia/engs2523/envs/` | `green-lory-env` and `legacy-lcoa-env` |

Clusters `arc` and `htc`; submit with `--clusters=all` (the templates do) and query with
`squeue -M all` / `sacct -M all`. Partitions: `devel` (10 min), `short` (12 h), `medium` (48 h),
`long`. Never submit a global surface without the user's confirmation.

Nothing computational runs on a login node: the wrappers there only call `sbatch`, and every
check, hash and manifest runs in a Slurm job (the prepare job of `submit_lory_sequence.sh`,
the verification job of `stage_release.sh`, the merge/QA jobs).

## Workflow

1. **Stage a release** from a committed checkout on the Mac, with the ARC SSH socket open:
   `bash arc/stage_release.sh 20261006-keyruns-v1`. It copies the Git-visible source and the
   generated finance overrides that `runs.yaml` pins (hash-checked), links the shared data,
   verifies the copy against the inventory in a short Slurm job and makes the release
   read-only.
2. **Land tables** (rarely): `submit_land_center_matrix.sh` runs `jobs/00_build_land_constraints.sh`
   for the centred 100 % build and derives the 2 % and 20 % tables
   (`green-lory-campaigns/land_center_20260923_v1/` is the accepted build).
3. **Plant surfaces**: from the release directory,
   ```bash
   ARC_RELEASE_INVENTORY=$PWD/source_inventory.json bash arc/submit_lory_sequence.sh \
     --stage smoke --campaign <campaign> --scenario <run id> \
     --land-csv <centred table> --results-root /data/.../green-lory-campaigns/<campaign>
   ```
   then `--stage diagnostic` (the same three cells, full year) and, after review and
   confirmation, `--stage global` (four longitude shards, `--global-locations` to restrict the
   cells). The command first runs a prepare job (2 CPUs, 16 GB, up to 30 min) and waits for
   it: every input check, the run directories and the manifests; its log is
   `<results root>/logs/prepare-<run id>-<job>.out`. The login node then submits the shard and
   merge/QA jobs. Each stage writes `manifest.json`, `submit.env`, `submission.json`
   (job ids; `submission_scheduler.txt` holds the scheduler's view), `shards/`, `merged/`,
   `qa/validation.json` and `logs/`; the merge/QA job fails on missing or duplicated rows,
   unexpected coordinates, currency, finance or cost-scope mismatches. `--dry-run` checks the
   inputs and prints the plan (on the Mac it runs locally with `.venv/bin/python`).
4. **Post-processing**: derived land shares and supplier tables for the network model
   (`python -m model.suppliers derive|export`, `../model/suppliers.py`);
   `postprocess_keyruns_20260924.sh` is the sequence used for the key runs. It is a Slurm job
   (2 CPUs, 16 GB, 4 h): submit it with `KEYRUN_RELEASE=<release> KEYRUN_ID=<run id> sbatch
   arc/postprocess_keyruns_20260924.sh`, never run it on a login node.
5. **Networks** run from the green-porpoise repository (`legacy_gpo/`, linked as
   `../legacy_gpo/`): stage a release (`legacy_gpo/arc/stage_release.sh`, run in green-porpoise),
   then `legacy_gpo/arc/submit_network_run.sh --release <id> --name <run> --contract <contract.json>`
   (4 CPUs, 64 GB, 2 h, one-hour solver limit by default; see `legacy_gpo/README.md`).
   Networks before 7 October 2026 ran from green-lory releases (`20260914-v3`,
   `20260916-network-tools-v1`), which still hold that code.
6. **Legacy model**: `jobs/07_build_legacy_env.sh` (environment, once),
   `jobs/09_extract_weather_store.sh` (weather store, once), `jobs/08_legacy_global.sh`
   (16-shard array; `LEGACY_EXTRA_ARGS` adds harness options such as `--annuity green_lory`).
7. **Supply pilots**: `jobs/06_land_supply_pilot.sh` (see `../reconciliation/supply_pilot/`).

## Files

| File | Role |
|---|---|
| `stage_release.sh` | Stage, verify and lock a source release (run on the Mac) |
| `submit_lory_sequence.sh` | Plant-surface campaigns by run id, in smoke, diagnostic and global stages |
| `submit_land_center_matrix.sh` | Centred land build and derived shares |
| `postprocess_keyruns_20260924.sh` | Slurm job: derived 2 % surface, contracts and legacy key table of the 28 September key runs |
| `release.py` | Inventory of a source release: `write` on the Mac, `verify` in a job |
| `manifest.py` | Run manifest (source, inputs, resolved settings, cost scope, hashes): `write`, and `verify` that the inputs have not changed |
| `qa.py` | Quality gates: `land` (a centred land table before a run) and `merge` (the shards of a run, merged and checked) |
| `arc_check_run_inputs.sh` | Presence checks of a run's inputs, in the prepare job and each shard job |
| `build-green-lory-env.sh`, `load_green_lory_env.sh` | Build the green-lory environment; load it in an interactive session |
| `jobs/` | Slurm templates: `00` land build, `01` surface shard, `06` supply pilot, `07` legacy environment, `08` legacy surface, `09` legacy weather store |

## Notes

- No surface has a grid backstop (the option was removed on 7 October 2026). Full-year runs require the versioned
  `renewable_union_area_km2_classwise_nested_v1` column of the land table (a lower bound on the
  physical wind/PV union).
- `--smoke-hours` (default 168) sets the simulated duration of the smoke stage;
  `ARC_CAMPAIGN_THREADS_PER_WORKER`, `ARC_CAMPAIGN_DIAGNOSTIC_WORKERS` and
  `ARC_CAMPAIGN_GLOBAL_WORKERS` size the workers. An hourly 52,702-cell surface took about
  3 hours on four 48-core jobs in September 2026.
- Campaign-wide solver options go in `GREEN_LORY_SOLVER_OPTIONS_JSON`, exported in the shell,
  never through `sbatch --export` (the commas break it). Surfaces retry suboptimal cells with
  barrier plus crossover, then dual simplex.
- The Gurobi licence path is set in the job templates; do not print the licence file.
