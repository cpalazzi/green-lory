# Future work

Items deliberately left out of the 1-degree global model, with what was decided and why.

## Slope (steep land)

**Decided 7 October 2026** (CAMPAIGN_PLAN.md, section 0a, D2):

- Steep land is costed rather than banned, up to 20 degrees. A factor on the build part of PV
  and wind: 1.0 below 8.5 degrees (15 % grade), 1.3 for 8.5-14 degrees, 1.6 for 14-20 degrees.
  Land above 20 degrees is excluded.
- Each plant takes the flattest land first. Slope comes from GEBCO (15 arc-seconds, about
  460 m), the same file as the bathymetry.
- Reason for keeping it simple (user, 7 October): a 90 m slope layer is out of place in a model
  whose weather is 1 degree. Here slope is only a land-availability and cost input, not part of
  the optimisation. "We'll aim to do higher resolution models in future where we include it more
  sensibly."

**Effect in the October campaign:** almost none. On GEBCO 94 % of land between 55 S and 9 S lies
below 8.5 degrees. At the 20 % land share, 27 of 16,885 cells pay a slope factor (at most
1.095), about 0.2 % of cells.

**Evidence is thin:**
- No published slope multiplier exists for wind or PV capital cost.
- The 1.3 and 1.6 are the middle of NREL's terrain multipliers for **transmission lines**
  (Lopez et al. 2025, Table 12: hilly 1.1-1.5, mountainous 1.2-2.0), applied by analogy. The
  paper and `inputs/parameters.yaml` said "civil-works multipliers" until 26 October 2026.
- Nafa and Curcio (2024): grading added 17 % to the capital cost of a 100 MWp hilly PV site in
  Japan. That is one site.
- Ma et al. (2025): mountain PV in Yunnan on 18-20 degree slopes. It gives no flat-ground
  comparison, so no multiplier.
- Aurecon (2024): terrain causes wind balance-of-plant overruns. It gives no numbers.
- Standard single-axis trackers take 15 % north-south grade. This comes from trade press only;
  a datasheet is needed before it can be cited.

**What a finer, local model should add:**
- A 90 m or finer DEM: Copernicus GLO-90, SRTM or Geomorpho90m. GEBCO flattens slopes.
- PV yield by slope and aspect. At this scale yield varies with which way the land faces
  (equator-facing slopes gain, pole-facing lose), and row spacing and shading change on slopes.
  1-degree weather cannot show this.
- Wind: speed-up and turbulence on ridges, and road and crane-pad costs by grade. Roads take
  about 6-15 % grade depending on assistance (Miceli 2012).
- Grading and earthworks costs by slope from engineering estimates, in place of transmission
  analogues.
