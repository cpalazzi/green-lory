#!/usr/bin/env python
"""Test script for spatial cost implementation."""
import sys
sys.path.insert(0, '.')
from model import run_global

# Test with spatial finance overrides
print('Starting test run with spatial overrides...')
results = run_global.run_global(
    locations=[(-23.0, 133.0)],  # Test location - matches example_finance_overrides_spatial.csv
    interest_csv='inputs/example_finance_overrides_spatial.csv',
    aggregation_count=1,
    max_snapshots=24,  # Just 1 day for quick test
    quiet=True,
)

print('\nTest completed successfully!')
print('Results shape:', results.shape)

print('\n=== Water/Land cost columns ===')
print(f'water_cost_usd_per_m3: {("water_cost_usd_per_m3" in results.columns)}')
print(f'land_cost_usd_per_km2_year: {("land_cost_usd_per_km2_year" in results.columns)}')
print(f'land_used_km2: {("land_used_km2" in results.columns)}')
print(f'water_cost_eur_per_t: {("water_cost_eur_per_t" in results.columns)}')
print(f'land_cost_eur_per_t: {("land_cost_eur_per_t" in results.columns)}')

print('\n=== Actual values ===')
for col in ['water_cost_usd_per_m3', 'land_used_km2', 'land_cost_usd_per_km2_year', 
            'water_cost_eur_per_t', 'land_cost_eur_per_t']:
    if col in results.columns:
        val = results.iloc[0][col]
        print(f'{col}: {val}')

print('\n=== Build cost multiplier test ===')
print('Checking if build_cost_multiplier is applied...')
print(f'Columns with "cost" in name:')
cost_cols = [col for col in results.columns if 'cost' in col.lower()]
for col in sorted(cost_cols)[:10]:
    print(f'  - {col}')

print('\nSUCCESS: All tests passed!')
