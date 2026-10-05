# Why a higher land-competition share (or a lower admission cutoff) is reasonable, and how the paper's 2 % gave the results it did

16 September 2026. Statement for the reconciliation; numbers from the campaign
files named in [RUNS.md](RUNS.md) and [DECISION_20260916.md](DECISION_20260916.md).

## 1. What the 2 % is, and what it is not

Verschuur et al. (2024, section 4.7) and Salmon and Bañares-Alcántara (2022,
section 2.2.1) justify the 2 % as a global share: shipping fuel is about 2 % of
global final energy (233 Mt of 418 EJ), so "an equivalent fraction of all land"
is assumed available to it, "though sensitivity of this assumption should be
tested in future work". Salmon 2022 states the underlying argument: if ammonia
took more than its share of global energy demand in land, other uses would not
be met, and "the competition factor is applied evenly to on- and offshore
sites". It is a proxy for competition between energy uses, applied uniformly
per cell for simplicity. It is not a physical constraint, not a land-use
planning rule, and not a statement that any particular 1-degree cell can host
only 2 % of its suitable land in renewables.

## 2. What the published run actually used

The archived supplier table cannot be reproduced by the stated land method. It
is reproduced by a constant PV density of 140 MW/km² at every latitude (the
stated method gives 106 at the equator falling to 78 at 23 degrees), 7.3 MW/km²
of wind (stated 5), complete wind/solar overlap and no protected-area or slope
exclusion, all on 2 % of the Table-2 suitable area. Because share and density
multiply, the same table is equally described by the stated densities with a
larger share. Expressed as an equivalent share of each method's land:

| Land method | Median archived / method capacity | The archived "2 %" is worth |
|---|---:|---:|
| Paper method as stated, legacy plant designs (exclusions, latitude packing, overlap) | 2.3 | 4.5 % |
| September green-lory replication (exclusive sharing, tracking at twice fixed) | 4.6 | 9 % |
| September green-lory central | 5.2 | 10 % |
| September green-lory central, Australia only | 9.3 | 18 % |

So the paper's networks are what a consistent method gives at roughly 5 to 10 %
per cell, labelled 2 %. That is the answer to "how did they get those results
with just 2 %": the label and the arithmetic were not the same thing.

## 3. Why the network needs only a fraction of the archived pool

The archived table admitted 1,762 Mt/yr of Australian capacity; the archived
network produced 258 Mt/yr there. With the green-lory surfaces, the runs at 4 %
and 8 % of suitable land (one-hour solver limit, gaps 0.7 to 1.0 %) give:

| Share of suitable land | Replication surface: cost, Australia | Central surface: cost, Australia |
|---|---:|---:|
| 2 % | 275 USD/t, 25 Mt/yr | 310 USD/t, 1 Mt/yr |
| 4 % | 264 USD/t, 184 Mt/yr | 293 USD/t, 97 Mt/yr |
| 8 % | 258 USD/t, 258 Mt/yr | 281 USD/t, 242 Mt/yr |
| archived table | 261 USD/t, 255 Mt/yr | |

About 250 to 300 Mt/yr of admitted Australian capacity is enough to recover
the archived geography; 4 % already makes Australia the largest producer. What
removes Australia at 2 % is not scarcity of land but the admission rule: a 1 Mt
reference plant scaled to each cell's 2 % and a hard 1 Mt/yr cutoff. Australian
cells are savanna, shrubland and grassland (Table-2 factors 0.2 to 0.5), so
their per-cell budgets are half a desert cell's; summed over all cells the
September surfaces still hold 209 to 292 Mt/yr of Australian capacity at 2 %
(272 to 387 with co-location), of which 1 to 25 Mt/yr survives the cutoff.
The cutoff was introduced for tractability ("to maintain the computational
tractability of the optimisation problem", section 4.8) and the paper reports
it did not affect their results, which is true only for capacities several
times the stated method's. The sparse implementation solves the full pool, so
tractability no longer justifies it.

## 4. The physical scale of the task

Land is not scarce for this problem at the aggregate level:

- Suitable land after protected-area and slope exclusions is 32.1 million km²
  (24 % of onshore land between 75 S and 75 N); 2 % of it is 642,000 km².
- The whole MOD-AMB demand of 602 Mt/yr needs about 5,500 TWh/yr, roughly
  2,500 GW of PV at 25 % capacity factor, which at the empirical fixed-tilt
  density of 87 MW/km² (Bolinger and Bolinger 2022) is about 29,000 km² of
  arrays: 4.5 % of the global 2 % budget, 0.09 % of suitable land, 0.02 % of
  onshore land.
- The archived network's Australian production needs about 12,500 km² of
  arrays: 29 % of Australia's 2 % budget and 0.16 % of the country. Chile and
  Oman use 36 % of their 2 % budgets, Mauritania 16 %, China 3 %.
- Real deployment concentrates. Announced Australian hubs are of the order of
  6,500 km² (Asian Renewable Energy Hub, 26 GW) and 15,000 km² (Western Green
  Energy Hub, 70 GW), that is 50 to 130 % of a single 1-degree cell, and single
  solar parks already exceed 500 km² (Talatan, Qinghai). A per-cell cap of 2 %
  of suitable land (100 to 230 km² at the focal cells) forbids projects that
  exist today.

## 5. Position

1. Keep the paper's rationale as an aggregate check, not a per-cell cap: total
   land taken by shipping-fuel ammonia in any network solution should stay
   within about 2 % of suitable land globally and nationally. Every network in
   this reconciliation satisfies that by a factor of 20 or more.
2. Represent site supply as cost against quantity under a per-cell land cap at
   hub scale, for which 10 to 20 % of the cell's suitable land after exclusions
   is consistent with announced projects, with the co-located, fixed-tilt land
   rules of 16 September. Cost then rises within a cell through packing and
   curtailment instead of being cut off; the finite-site solves already show
   the shape of those curves.
3. Until those curves exist globally, lower the admission cutoff (0.25 to
   0.5 Mt/yr) or raise the share to 5 to 10 % on the scaled-design surfaces,
   and label either as a sensitivity on an arbitrary parameter. A share of
   4 to 8 % is not a calibration to the archived map: it is the range in which
   the archived run itself operated once its arithmetic is expressed in the
   stated method's terms.
4. What a higher share does not settle: competing uses within the suitable
   classes (grazing, biodiversity outside WDPA, cultural land), water, the
   opportunity cost of land, and the spacing of transmission and pipelines
   inside a cell. These belong in a land-cost or opportunity-cost term, which
   is the next step if the share is to be defended beyond an order of magnitude.

## 6. One-paragraph version

The paper's 2 % is a global energy-share proxy applied uniformly per cell, not
a physical limit, and the run that produced the published supplier table did
not implement it as stated: its capacities correspond to 5 to 10 % of the
stated method's land, so its networks are those of a 5 to 10 % share. The
demand itself needs under 5 % of the global 2 % budget and 0.02 % of onshore
land; what excludes Australia at a literal 2 % is the per-cell cap combined
with the 1 Mt/yr admission cutoff, both of which forbid the concentration that
real hub projects exhibit. Raising the per-cell share to hub scale, or lowering
the admission cutoff, therefore reproduces the published geography for stated,
physical reasons, while the 2 % remains a valid aggregate check that every
solution here passes with a wide margin.
