# Land reconstruction runs

## 14 September 2026: joint-mask diagnostic pilot

| Job | Name | Cell convention | Scope | Final status |
|---|---|---|---|---|
| 8806853 | glr-land-center | weather coordinate at cell center | 40 diagnostic cells | Completed, 7m52s |
| 8806854 | glr-land-southwest | coordinate at southwest cell corner | same 40 cells | Completed, 14m03s |

`glr` means Green Lory reconciliation. These jobs use HTC's short partition,
one CPU, 16 GB RAM and a two-hour limit. BEGIN, END and FAIL notifications go
to carlo.palazzi@eng.ox.ac.uk. They do not run the ammonia or shipping models.

ARC source release:
`/data/engs-df-green-ammonia/engs2523/green-lory-releases/20260914-land-v1`.
Its eleven-file inventory was verified on ARC before submission:
`c2bf6527976dd63a40a27f9e0c3e1e624f2f5b6bf24e3476c4746ed9868ff771`.

ARC campaign:
`/data/engs-df-green-ammonia/engs2523/green-lory-campaigns/land_reconcile_20260914_v1`.
Outputs are `joint_pilot_v1_center/` and `joint_pilot_v1_southwest/`, with
job-ID-labelled stdout/stderr under `logs/`. All output paths refuse overwrite.
Raw sources remain in the existing read-only Green Lory data directory.

Local audit `audit/v3` is the accepted baseline for pilot comparison.
`audit/superseded/v1` is an incomplete development attempt; `audit/superseded/v2` is superseded
because its diagnostic reconstruction multiplied the shipping share a second
time. This was an error in the new diagnostic, not evidence of a second
shipping-share multiplication in the existing land pipeline. The corrected
v3 reproduces the old land fractions to 2.3e-16 and solar areas to 9.1e-13 km2.

Pre-submission validation: 90 tests and 27 subtests passed, with one existing
NumPy binary-size warning from the historical-network export test.

Both pilots were fetched and their hashes, coordinate coverage and all 80
cell area-conservation checks independently revalidated. Accepted comparison:
`audit/joint-comparison-v1`. The accepted global unmasked class-only comparison
is `audit/global-unmasked-v2`; its predecessor is in `audit/superseded/`.

## Fixed-PV follow-on: failed first attempt and corrected staging

**8807144 — glr-pv-fixed-3** failed on 14 September at 12:44:12 BST after
4m18s, exit code 1:0. It extracted the full-year weather subsets but produced
no plant-optimization results. The minimal source release included the central
technology overlay but omitted its inherited base YAML. This was a release
packaging error, not plant infeasibility or evidence against fixed PV.

The immutable v1 release and `fixed_pv_3cells_v1` output are preserved on ARC.
Their local copies are `sources/fixed-pv-release-v1` and
`arc_received_fixed_attempt_v1/fixed_pv_3cells_v1`; logs are preserved alongside.

The v2 staging code follows the entire relative YAML `extends` chain, rejects
missing/circular/nonportable dependencies, and checks the relocated chain.
The runner loads its complete technology configuration before weather work;
its manifest now hashes every inherited YAML. A preflight-only mode checks
release integrity, configuration, land and small input files without solving
or creating a result directory. Regression tests include staging the real
configuration and comparing its merged contents with the source checkout.

Local validation: **99 tests and 30 subtests passed**, with the same existing
NumPy warning. The corrected 34-file release is
`/data/engs-df-green-ammonia/engs2523/green-lory-releases/20260914-land-fixed-v2`,
tree SHA-256 `c02a9cff3fa4c2b27be602d114319213322381b894681c5f6c587bc800b2b738`.

The fixed-PV job uses three full-year hourly plant solves (Atacama, northwest
Australia and central Australia), 12 CPUs, 32 GB RAM and a two-hour limit on
HTC short. It retains BEGIN, END and FAIL notifications to
carlo.palazzi@eng.ox.ac.uk. `glr-pv-fixed-3` means Green Lory reconciliation,
fixed photovoltaic generation, three diagnostic sites. The job ID and versioned
output path distinguish attempts; the descriptive name alone does not.

ARC preflight passed with both inherited YAMLs present. Rerun **8809820 —
glr-pv-fixed-3-v2** was submitted on 14 September 2026, using the new
`fixed_pv_3cells_v2` output directory. No fixed-PV result is accepted until
completion, cost/dispatch/land-capacity QA and comparison with the central
tracking design on the same corrected land.

Submission: 18:46:41 BST. The scheduler confirmed 12 CPUs, 32 GB, the two-hour
limit and `MailType=BEGIN,END,FAIL` to the expected address. The job subsequently
started on `htc-c029`. Local comparison QA is implemented in
`compare_pv_pilot.py`; it reconstructs land capacity independently and verifies
matching cost/finance inputs and recorded parent-weather metadata. The central
comparator allows both PV types and is tracking-dominated: disabling tracking
narrows the design choices, so lower unconstrained LCOA is not expected. The
test concerns the tradeoff between cost and land-limited supply.

**Final: completed 0:0**, 18:47:53–18:52:52 BST, elapsed **4m59s**; batch MaxRSS
1,686,316 K. Raw results were fetched to
`arc_received_fixed_v2/fixed_pv_3cells_v2`. Independent comparison and QA passed
under `audit/fixed-vs-tracking-v1`. Local tests, including comparison checks:
**102 passed and 30 subtests passed**, with the existing NumPy warning.

Verified ARC/local SHA-256 digests:

- Manifest: `fd00f90b54b342fd1c31aa9b611a7f3dae0c78dd139970785eb761d2b7624060`
- Results: `352544cf27b68526ae19e465274bc5864ba6034024cf89eec6ecf94aca19e5e2`
- Summary: `09fd803c77c56bfaf385bdc4ea30dafbfe08711774f4394c6c1f4ad0e5fa715a`

See [the completed fixed-PV report](FIXED_PV_20260914.md) for findings and the
remaining Green Lory and Green Porpoise reconciliation gates.

## Land-constrained cost–quantity pilot

The follow-on experiment uses 21 specified-quantity solves and three
unconstrained controls across the same three sites. Quantities include the
1 Mt/year supplier cutoff and each archived historical capacity. Both PV
types and wind are allowed; the shared land budget is enforced in the solve.
No land input, shipping allocation, density, weather or cost assumption changes.

The new opt-in `solved_quantity` mode reports achieved quantity, not a rescaled
or maximum capacity. The old `paper_scaled` mode remains restricted to
unconstrained reference plants. A typed solver termination distinguishes proven
infeasibility from numerical, setup and license failures. Each point has its
own checkpoint directory.

Pre-submission validation: **109 tests and 30 subtests passed**, with the same
existing NumPy warning. Tests include an actual small shared-land optimization
that changes PV mix, and an infeasible demand. ARC preflight passed.

Pinned source release:
`/data/engs-df-green-ammonia/engs2523/green-lory-releases/20260914-land-supply-v1`.
Its 45-file inventory has tree SHA-256
`568e50dc9688c4cbeaf2d16379dbd780066605980fb5c6df07189c187018a2b4`.
Output directory: `supply_curve_3cells_v1` under the existing land campaign.
Weather reuses the content-hashed subsets from completed fixed-PV job 8809820.
Job profile: HTC short, 12 CPUs, 32 GB, two hours, BEGIN/END/FAIL to the usual
Oxford address. See [the experiment protocol](supply_curve/README.md).

Submitted job: **8810426 — glr-land-supply-3-v1**. `glr` denotes Green Lory
reconciliation; `land-supply` distinguishes constrained cost–quantity points
from the earlier fixed-PV reference run; `3` is the number of sites and `v1`
the attempt version. Completion and result checks remain pending.

First attempt **8810426 failed 1:0 after 2m13s**. All three control
optimizations reported optimal termination, but the wrapper checked the
internal result dictionary as if it already had the CSV output's unit suffixes
(`accumulated_penalty_mwh`, etc.). No constrained points were produced. This
was an output-adapter error, not a physical infeasibility result.

The wrapper now calls the same unit-normalization function as the standard
global result store; a regression test covers this integration boundary. The
failed release/output are preserved, with a local copy under
`arc_received_supply_attempt_v1`. The corrected 46-file release is
`20260914-land-supply-v2`, tree SHA-256
`08bba6561bfed8526b59ec8461922dfa75f149e53421faca70c2023d72697e28`.

Corrected attempt **8810489 — glr-land-supply-3-v2** was submitted after a
successful ARC preflight and **112 tests / 30 subtests** locally. It uses
`supply_curve_3cells_v2`; no v1 artifacts are overwritten.

**8810489 failed 1:0**, 22:18:48–22:23:07 BST on 14 September, elapsed 4m19s;
batch MaxRSS 2,759,200 K. The job completed its three controls and 16 feasible
quantity checkpoints before the extreme quantities encountered numerical
termination. Linopy labelled Gurobi `suboptimal` as status `ok`; the existing
model checked status alone. The independent land check rejected the apparent
Atacama 5 Mt/year result (230.430 km2 of PV against a 226.836 km2 budget).
It was not accepted as feasible or relabelled as proven infeasible.

Raw checkpoints and both logs are preserved under
`arc_received_supply_attempt_v2/`. No complete supply curve is accepted from
that failed attempt. Version 3 now requires **both** status `ok` and termination
`optimal` before extracting a solution. Numerical or ambiguous terminations
get one fresh-network Gurobi retry using homogeneous barrier, disabled dual
reductions and stronger numerical settings; all attempts are recorded.
Unresolved numerical, setup or licence failures still fail the job. The land,
costs, weather, quantity points and physical acceptance tolerances are unchanged.

Local validation: **120 tests and 30 subtests passed**, with the existing NumPy
warning. Pinned 48-file release `20260915-land-supply-v3`, tree SHA-256
`e0c196b6e9bb6483fba988f2086295bfc901ef2a94c4b8fd7daedb1ebea7b373`.

ARC preflight passed. Submitted **8811980 — glr-land-supply-3-v3** on
15 September at 08:57:37 BST, output `supply_curve_3cells_v3`. Scheduler
confirmation: HTC short, 12 CPUs, 32 GB, two hours, BEGIN/END/FAIL to
carlo.palazzi@eng.ox.ac.uk. Initial state: pending scheduler priority.

**8811980 failed 1:0**, 08:58:08–09:01:06 BST on 15 September, elapsed 2m58s,
batch MaxRSS 2,955,888 K. The strict check correctly rejected a suboptimal
Atacama 4 Mt/year solve, and homogeneous barrier did not resolve it. Other
high-demand retries returned `unbounded`, not a usable infeasibility conclusion.
No such endpoint was accepted. The third attempt is preserved under
`arc_received_supply_attempt_v3/`. Attempt v4 replaces only the numerical
fallback with dual simplex (`Method=1`, `DualReductions=0`, `NumericFocus=3`,
`InfUnbdInfo=1`); it retains strict acceptance, unchanged physical inputs and
the original first-attempt settings. Tests including bounded retry and
exception propagation: **122 passed and 30 subtests passed**.

Attempt v4 ARC preflight passed. Submitted **8811997 — glr-land-supply-3-v4**,
using the new `supply_curve_3cells_v4` output. Pinned source release
`20260915-land-supply-v4` has 48 files, tree SHA-256
`e918426abff4b867f4d44d35148ed5823469de9a95d317ca1d6b66a6360cf8f5`.
Resources and BEGIN/END/FAIL notification settings remain unchanged.

At **09:10:24 BST on 15 September**, 8811997 was running (started 09:06:12,
elapsed 4m12s), with the dual-simplex fallbacks active. Completion and full-curve
QA remain pending. The three 1 Mt/year optimal checkpoints from v3 were fetched
and locally rechecked independently of that failed job's overall status;
see [the interim supply report](SUPPLY_CURVE_20260915.md).

At **10:03:09 BST**, 8811997 remained running, elapsed 56m57s. Snapshot
`arc_received_supply_snapshots_v4/20260915T1001BST` was fetched and locally
validated: three controls and 16 constrained points passed, including dual-
simplex optima at Atacama 4 and 4.5 Mt/year. Five extreme-quantity checkpoints
are missing and remain unresolved, not infeasible. The partial audit is
`audit/supply-checkpoints-v4-20260915T1001BST`; it explicitly does not accept
the full run. No duplicate job was submitted.

## 15 September 2026: local native-MODIS pilot

Three downloaded Q1 C6.1/2022 HDF files were moved into the source directory,
identity/coverage checked and hashed. Local CMG, GEBCO and all 15 required WDPA
polygon component files match the previous ARC source hashes exactly.

`run_native_pilot.py` ran locally, first at 1,200/2,400 integration samples
per degree (`audit/native-modis-pilot-v1`), then 2,400/4,800/9,600
(`audit/native-modis-pilot-v2`, accepted). These are quadrature resolutions,
not observational resolutions. Suitable-area changes between the finest
two grids are below 0.0001%; native versus previous-CMG changes are below
0.07% for all three sites. No Slurm submission or email applies to this
local land-data calculation.

Both separate three-cell native land exports passed the campaign input gate.
Native geometry/mask tests and the full suite passed: **135 tests and 30
subtests**, with the pre-existing NumPy warning. See the
[native comparison report](NATIVE_MODIS_20260915.md).

The native source/config/test release is preserved under
`sources/native-pilot-release-v1` (19 files), tree SHA-256
`7d03b36525288d794d568134a8478994e14a0126e68140f1a1774c8f953353d9`.
All five analysis-code hashes match the accepted native-v2 manifest.
Final ARC status check this session: 10:06:41 BST, job 8811997 running,
elapsed 1h00m29s; no additional completed checkpoints since the fetched snapshot.

## Final v4 supply status and analytical closure

**8811997 ended TIMEOUT at 11:06:31 BST**, elapsed 2h00m19s. The batch was
cancelled with signal 15; MaxRSS was 4,401,348 KiB. Final raw files, stdout,
stderr and the scheduler record are in `arc_received_supply_attempt_v4/`.
The job-level 0:0 exit field is not a successful-completion indication.

Final checkpoints add solver-proven infeasibility for central Australia
4.324243441 Mt/year and northwest Australia 2.5 Mt/year. Local final-checkpoint
QA passed for three controls, 16 feasible quantities and those two infeasible
quantities (`audit/supply-checkpoints-v4-final/`).

The three remaining planned quantities are closed separately by annual-energy
certificates in `audit/supply-certified-grid-v1/`: Atacama 5 and 9.769406259,
and northwest Australia 4.019748515 Mt/year. The certificate validates source
hashes, non-energy-creating conversion/storage topology and a matching primal/
dual annual land allocation. It does not fabricate solver checkpoints or
change the TIMEOUT state. All 21 planned quantities are now classified;
an exact maximum and a continuous cost curve remain outside this result.

## Fixed-only enforced-land supplier-threshold follow-on

New experiment **8812986 — glr-land-fixed-3-v1** uses three unconstrained
fixed-only controls and three land-constrained **1 Mt/year** solves. Tracking
is disabled before solving; wind and fixed PV remain available. Land, weather,
costs, finance, hourly accounting and physical acceptance tolerances are held
constant. This tests the fixed-only model at the supplier threshold against
the accepted mixed-PV results, not another rescaling of the reference plant.

Pinned 58-file release `20260915-land-fixed-threshold-v1`, tree SHA-256
`67452816a45aee49ad8fa51468186ecf0d4b7ea18b100065ec1a8914df5f342a`.
ARC preflight passed and confirmed six solves. New output:
`fixed_land_threshold_3cells_v1/`; job-ID-labelled logs use
`glr-land-fixed-3-v1-8812986`. The existing template retains HTC short,
12 CPUs, 32 GiB, two hours and BEGIN/END/FAIL mail to the Oxford address.
Local tests: **143 passed and 30 subtests passed**, with the pre-existing
NumPy binary-size warning. No completed-run claim is made at submission.

Submitted 11:28:17 BST. At 11:29:26 BST, 8812986 was **PENDING (Priority)**.
The scheduler confirmed all requested resources and
`MailType=BEGIN,END,FAIL`, `MailUser=carlo.palazzi@eng.ox.ac.uk`.

### Final fixed-threshold result and retry, 15 September

**8812986 FAILED** at 11:32:42 BST, before any optimisation, because the
isolated source release had no `data/model_bathymetry.nc`. The old preflight
did not exercise the weather loader. Raw manifest and logs are preserved in
`arc_received_fixed_threshold_attempt_v1/`; this is not physical infeasibility.

New release `20260915-land-fixed-threshold-v2` adds a bathymetry hash gate and
a real-loader preflight. Its explicit data link points at the controlled ARC
data. The 58-file tree hash is
`8b9aca50a33624b560850d04d9cccec93dd240d8fb3c91799bdaa36717bd2872`.
ARC preflight passed, including the previously missing dependency.

**8814788 — glr-land-fixed-3-v2** was submitted at 15:53:16 BST and completed
15:53:51–15:56:58 BST (3m07s; MaxRSS 2,531,076 KiB; exit 0:0). The template
retained 12 CPUs, 32 GB, two hours and BEGIN/END/FAIL Oxford emails. No physical
or cost inputs changed. All six controls/quantity solves were fetched to
`arc_received_fixed_threshold_v2/`. Full-run and checkpoint audits passed;
source inventory and zero tracking capacity were separately checked.

All sites are feasible at 1 Mt/year. Fixed-only costs: Atacama 234.716679,
northwest Australia 251.850041, central Australia 236.676925 EUR2020/t.
Audit: `audit/fixed-threshold-v2/`, with all six checkpoint checks in
`audit/fixed-threshold-checkpoints-v2/`. The comparison with legacy files,
weather identity and energy efficiency is in `audit/model-evidence-v2/`.
Focused tests: 28 passed. See `../HANDOVER_20260915.md` for interpretation.

### Separate 2019 profile-stack extraction

The large green-condor ERA5 cutout was located on ARC (approximately 290 GiB)
under `green-condor/data/global_cutout_2019.nc`; the green-lory archive folder
contains provenance rather than that binary. Twelve derived CF tiles have
declared 60-row extents; the unsuffixed store has only 85 rows.

The first read-only selected-cell probe failed with `can't start new thread`.
It was repeated with BLAS/OMP/MKL restricted to one thread and a new directory,
`alternative_weather_probe_v2`. Six profiles (fixed PV and wind for three
sites) have all 8,760 finite hours and were fetched and hash-verified locally.
The probe was not a Slurm plant run. Sources were not modified.

`audit/alternative-weather-v1/` compares raw profile CFs on the same 2019 time
axis. Alternative fixed PV is 7–9% lower and raw wind 26–38% lower. This also
changes or may change technology/resolution/normalization, not only weather.
No alternative-weather plant LCOA or capacity is claimed.
