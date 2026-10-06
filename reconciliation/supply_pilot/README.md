# Finite-site supply curves at the focal cells

The global surfaces report a cell's capacity by scaling the cheapest 1 Mt/yr design until it
fills the cell's land (`scaled_reference_design`). This pilot asks how much more a plant
designed for the land can produce. At each of the three focal cells it solves the plant at a
list of fixed annual ammonia quantities with the wind and PV land budgets enforced inside the
optimisation (`in_solve`), and reports cost against quantity (`solved_quantity`). The results
are in `../DECISION.md` section 3.

| Run id (`../../runs.yaml`) | Config | PV |
|---|---|---|
| `gl_way2050_ameli_bflat_wnone_land2c_fixed_pilot3cells` | `config_v3_center.json` | fixed tilt only |
| `gl_way2050_ameli_bflat_wnone_land2c_track_pilot3cells` | `config_v3_center_tracking.json` | tracking only |

Both use the centred 2 % land table (sha256 pinned in the config), co-located wind and PV
(wind direct-impact share 0.03), tracking at 1.458 times the fixed footprint, Way 2050 costs
with the explicit compressor and DEA tank, Ameli WACC, hourly weather, and no water or grid.

## Protocol

- Each site first repeats the unconstrained 1 Mt/yr control, which must reproduce the
  configured control LCOA within 0.15 EUR2020/t.
- Then it solves each configured quantity with the same costs, finance and weather; only the
  constant ammonia demand changes between points.
- A solver-confirmed `infeasible` is a result. Numerical or ambiguous terminations get one
  fresh-network retry with stronger settings; unresolved failures fail the run and are never
  relabelled as infeasible.
- Feasible points must pass production, grid/slack, cost-closure and land-use checks, and
  costs must not fall with quantity. Tested points bracket the maximum feasible output; they do
  not prove an exact maximum.

## Running

On ARC from a staged release (`../../arc/stage_release.sh`), with the hashed weather subsets of
the accepted fixed-PV run (`fixed_pv_3cells_v2`):

```bash
SUPPLY_RELEASE=<release> SUPPLY_LAND=<centred 2 % table> SUPPLY_WEATHER_RUN=<fixed_pv_3cells_v2> \
SUPPLY_OUTPUT=<new directory> SUPPLY_CONFIG=reconciliation/supply_pilot/config_v3_center.json \
SUPPLY_PV_POLICY=fixed-only sbatch arc/jobs/06_land_supply_pilot.sh
```

Add `--preflight-only` to the command in the job to check inputs without solving. The v6 runs
took 8 to 9 hours on 12 CPUs. Every point has its own directory, so an interrupted run keeps
its finished points.
