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

## Workflow

1. **Stage a release** from a committed checkout on the Mac, with the ARC SSH socket open:
   `bash arc/stage_release.sh 20261006-keyruns-v1`. It copies the Git-visible source and the
   generated finance overrides that `runs.yaml` pins (hash-checked), links the shared data,
   verifies the inventory on ARC and makes the release read-only.
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
   cells). Each stage writes `manifest.json`, `submission.json`, `shards/`, `merged/`,
   `qa/validation.json` and `logs/`; a merge/QA job runs after the shards and fails on missing
   or duplicated rows, unexpected coordinates, currency, finance or cost-scope mismatches.
   `--dry-run` validates inputs and prints the commands.
4. **Post-processing**: derived land shares and supplier contracts
   (`python -m networks.contracts derive|export`, `../networks/contracts.py`);
   `postprocess_keyruns_20260924.sh` is the sequence used for the key runs. It is a Slurm job
   (2 CPUs, 16 GB, 4 h): submit it with `KEYRUN_RELEASE=<release> KEYRUN_ID=<run id> sbatch
   arc/postprocess_keyruns_20260924.sh`, never run it on a login node.
5. **Networks**: `submit_network_run.sh --name <id> --contract <contract.json>` runs
   `jobs/03_replay_historical_network.sh` (`python -m networks.replay`; 4 CPUs, 64 GB, 2 h,
   one-hour solver limit by default).
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
| `submit_network_run.sh` | One green-porpoise network from a supplier contract |
| `postprocess_keyruns_20260924.sh` | Slurm job: derived 2 % surface, contracts and legacy key table of the 28 September key runs |
| `write_campaign_manifest.py`, `write_campaign_submission.py` | Run manifest (source, inputs, resolved settings, hashes) and job ids |
| `merge_and_qa_campaign.py` | Merge an explicit shard list and run the QA gates |
| `release_source_inventory.py`, `verify_campaign_manifest_inputs.py`, `validate_land_campaign_input.py`, `arc_check_run_inputs.sh` | Inventory and input checks (centred land tables only) |
| `build-green-lory-env.sh`, `load_green_lory_env.sh` | Build the green-lory environment; load it in an interactive session |
| `jobs/` | Slurm templates: `00` land build, `01` surface shard, `03` network, `06` supply pilot, `07` legacy environment, `08` legacy surface, `09` legacy weather store |

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
