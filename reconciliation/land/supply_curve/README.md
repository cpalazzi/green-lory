# Land-constrained cost–quantity pilot

This experiment asks whether changing plant design allows more supply than
scaling the unconstrained minimum-LCOA plant. It does not change either land
stack or promote a global supplier surface.

The three sites use the validated common centered-cell land, explicit fixed-PV
density, 2% shipping allocation, and the current tracking/fixed footprint ratio
of 2. Wind and both PV options remain available. Individual wind/PV areas and
their shared renewable-area budget are enforced inside the optimization;
wind/PV co-location is not implicitly introduced.

Each site first repeats the 1 Mt/year unconstrained central case. Its cost
must reproduce the accepted central value within 0.15 EUR 2020/t. It then
solves seven specified annual quantities with the same costs, finance,
full-year hourly weather and ramp constraints. The configured quantities span
the previous design's capacity, the 1 Mt/year supplier cutoff and the archived
historical capacity. Only the constant ammonia demand changes between points.

The `solved_quantity` output contract reports an achieved land-feasible
quantity and its cost. It does not scale that plant again, label it a maximum,
or populate a preferred maximum-capacity supplier column. The existing
`paper_scaled` path remains restricted to unconstrained reference designs.

Grid power is disabled. A solver-confirmed `infeasible` termination is a
scientific result with no fabricated zero price or capacity. Ambiguous,
numerical, configuration and licence errors fail the run. From attempt v3,
identified numerical or ambiguous Gurobi terminations receive one fresh-network
retry with stronger numerical settings (homogeneous barrier in v3; dual simplex
in v4 after the former remained unresolved). All attempts
are recorded; unresolved failures are not relabelled as physical infeasibility.
Both status `ok` and termination `optimal` are required. Feasible results
must pass production, grid/slack, cost-closure and independently reconstructed
land-use checks. Feasibility and average costs must be monotonic to numerical
tolerances. Tested points can bracket maximum feasible output but do not prove
an exact maximum or define a continuous global supply curve.

Inputs and artifacts are pinned in a separate release. Completed points and
infeasible terminations each have their own directory, so an interrupted run
does not erase finished work. See [the run record](../RUNS.md) for job details.

## Follow-on status, 15 September 2026

The mixed-PV v4 job timed out after two hours. The original solver-only
protocol produced 16 feasible and two infeasible quantity checkpoints. A
separate post-run certificate now classifies the remaining three endpoints
analytically; see `../SUPPLY_CERTIFIED_20260915.md`. Original solver statuses
and raw output are unchanged.

`fixed_only_1mt_v1.json` defines the next controlled test: fixed-only controls
and enforced-land 1 Mt/year plants at the same three sites. The runner's
explicit `--pv-policy fixed-only` disables tracking before solving and
rejects any resulting tracking capacity. Both the policy and configuration
are pinned in the manifest. Default `both` behavior is preserved for the
original experiment. Submission: 8812986, new output
`fixed_land_threshold_3cells_v1`.

That attempt failed before solving because its release lacked the bathymetry
dependency. The v2 retry, **8814788**, completed on 15 September at 15:56:58 BST.
All six solves were fetched and locally rechecked. Fixed-only land-enforced
1 Mt/year costs are 234.72, 251.85 and 236.68 EUR2020/t for Atacama, northwest
Australia and central Australia. See `../../HANDOVER_20260915.md` and campaign
`audit/fixed-threshold-v2/`. Preflight now hashes the bathymetry dependency and
exercises the real weather loader.
