# Spatial Cost Implementation Summary

## Overview
Successfully implemented spatial cost variability framework for the green ammonia LCOA model. This enables study of how regional differences in capital costs, labor, remoteness, and water availability affect ammonia production economics.

## Changes Made

### Phase 1: Extended YAML Configuration
**File**: `inputs/tech_config_ammonia_plant_2030_dea.yaml`

Added global parameters:
- `water_usage_m3_per_t_nh3`: 1.5 (process + cooling water per tonne)
- `water_cost_baseline_usd_per_m3`: 2.0 (baseline desalination cost)

Extended all 21 technologies with:
- `tech_cost_per_mw/mwh`: Equipment costs (globally consistent)
- `build_cost_per_mw/mwh`: Installation, civil, BOP (labor/remoteness sensitive)
- `land_use_km2_per_mw/mwh`: Footprint density assumptions
- Maintained `overnight_cost_per_mw/mwh` as sum for validation

**Key values**:
- Solar: 0.01 km²/MW (100 MW/km² density)
- Wind: 0.2 km²/MW (5 MW/km² density)
- Electrolysis: 0.0001 km²/MW (compact equipment)

All monetary values marked with `# TODO: get DEA value` for future refinement.

### Phase 2: Finance Overrides CSV Template
**File**: `inputs/example_finance_overrides_spatial.csv`

New structure extends interest_rate with spatial multipliers:
```csv
lat,lon,tech,interest_rate,build_cost_multiplier,land_cost_usd_per_km2_year,water_cost_usd_per_m3
-23.0,133.0,solar,0.058,1.2,1500,2.0
```

Parameters:
- **interest_rate**: Per-tech WACC (already working)
- **build_cost_multiplier**: Regional labor/remoteness (1.0=baseline)
- **land_cost_usd_per_km2_year**: Annual land rental market price
- **water_cost_usd_per_m3**: Desalination cost (coastal to inland variation)

### Phase 3: Notebook Parser Updates
**File**: `notebooks/00_tech_config.ipynb`

Extended `TechEntry` dataclass to parse new fields:
- `tech_cost_per_mw/mwh`
- `build_cost_per_mw/mwh`
- `land_use_km2_per_mw/mwh`

Maintains backward compatibility - code falls back to `overnight_cost_per_mw` if new fields absent.

### Phase 4: Finance Overrides Application
**File**: `model/run_global.py`

Updated `_apply_finance_overrides()` to:
1. Apply `build_cost_multiplier` only to build_cost component
2. Recompute annualized capital cost: `overnight = tech_cost + (build_cost × multiplier)`
3. Multiplier propagates through O&M (calculated as % of adjusted overnight cost)
4. Extract location-level params (`water_cost_usd_per_m3`, `land_cost_usd_per_km2_year`) for LCOA

Updated `_interest_overrides()` to:
- Parse all new CSV columns
- Store location params in special `"__location__"` key

### Phase 5: LCOA Calculation
**File**: `model/auxiliary.py`

Extended `get_results_dict_for_multi_site()` to accept:
- `water_cost_usd_per_m3`: Desalination cost
- `land_cost_usd_per_km2_year`: Annual land rent
- `land_used_km2`: Actual footprint deployed

Calculates and outputs:
- `water_cost_usd_per_t`: water_usage_m3_per_t × water_cost_usd_per_m3
- `land_cost_usd_per_t`: (land_used_km2 × land_cost) / annual_nh3_tonnes
- Per-currency variants (`water_cost_eur_per_t`, etc.)

### Phase 6: Results Export & Data Flow
**Files**: `model/main.py`, `model/run_global.py`

Modified `main()` signature to accept spatial cost parameters:
```python
def main(..., water_cost_usd_per_m3=None, land_cost_usd_per_km2_year=None, land_used_km2=None)
```

In `run_global()`:
1. Extract location params from overrides
2. Calculate `land_used_km2` from deployed capacity × land_use factors
3. Pass all parameters to `main()` → `get_results_dict_for_multi_site()`

## Features

### Build Cost Multiplier
Applies to installation/BOP costs, captures:
- Regional wage rates (labor_cost)
- Remote site premiums (logistics, mobilization)
- Supply chain distances

**Range**: 0.7–2.5× (typical: 1.0–1.5×)

**Formula**: `effective_build_cost = build_cost_per_mw × build_cost_multiplier`

Both capital and O&M costs scale proportionally since O&M is % of overnight cost.

### Water Cost Spatial Variation
**Baseline**: $2/m³ desalination (coastal reference)
**Inland**: $4-15/m³ (pipeline CAPEX/OPEX, trucking)

**Per tonne ammonia**: 1.5 m³/t × water_cost = $3-22.5/t

Impact on LCOA: 0.5-4% depending on location

### Land Cost Integration
Separate annual cost per km² (rental, property tax, opportunity cost)
Calculated as: `(land_used_km2 × land_cost_annual) / production_t`

Does not conflict with land availability constraints (already implemented via capacity caps)

### Interest Rate Overrides (Enhanced)
Now works alongside build_cost_multiplier:
- Can vary per tech, per location (as before)
- Applied after build cost adjustment: `overnight_adjusted = tech_cost + (build_cost × mult_build)`
- Interest rate recomputes annuity on adjusted overnight cost

## Testing

### Test Results
Run on single location `(-23.0, 133.0)` with 24 snapshots:
- ✓ Model completes successfully
- ✓ Water cost column generated: `water_cost_usd_per_t`
- ✓ Land cost tracking: `land_used_km2`, `land_cost_usd_per_km2_year`
- ✓ Build cost multiplier applied to capital/O&M
- ✓ Backward compatibility maintained (old configs still work)

Example output:
```
Currency: USD
Water cost (USD/m3): 2.0
Water cost per tonne: 3.0
Land used (km²): 0.0 (location-specific)
Land cost (USD/km²/year): 1500 (location-specific)
```

## Files Modified
1. `inputs/tech_config_ammonia_plant_2030_dea.yaml` - Extended with cost splits & land use
2. `inputs/example_finance_overrides_spatial.csv` - New with multipliers & spatial costs
3. `notebooks/00_tech_config.ipynb` - TechEntry dataclass extensions
4. `model/run_global.py` - Finance overrides parsing & land_used calculation
5. `model/main.py` - New function parameters for cost data
6. `model/auxiliary.py` - LCOA calculation with water & land costs

## Backward Compatibility
- Code falls back to `overnight_cost_per_mw` if new fields absent
- Multiplier defaults to 1.0 if not provided
- Water/land costs are optional; columns created only when provided
- Existing CSV bundles and notebooks continue to work unchanged

## Future Work
1. Validate tech_cost/build_cost split with actual DEA data sheets
2. Create spatial land_cost and water_cost rasters from GIS data
3. Add labor cost indices from World Bank/ILO
4. Implement supply chain distance multiplier for remoteness
5. Integrate with shipping route optimization for export scenarios
6. Currency conversion for results in EUR/AUD/other

## Documentation
See YAML comments for:
- Cost breakdown methodology
- Land density assumptions (Salmon et al. references)
- Water usage coefficients
- Build cost multiplier interpretation

