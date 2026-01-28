#!/usr/bin/env python
"""Test notebook-style global run."""
import sys
from pathlib import Path
sys.path.insert(0, '.')

from model import run_global

# Simulate notebook code with test location
locations = [(-23.0, 133.0)]

print('Testing notebook-style global run with test location...')
results_df = run_global.run_global(
    locations=locations,
    aggregation_count=1,
    time_step=1.0,
    max_snapshots=24,  # Quick test
    output_csv=None,
    tech_yaml='inputs/tech_config_ammonia_plant_2030_dea.yaml',
    quiet=True,
)

print(f'Finished: {len(results_df)} locations')
print(f'\nColumns added for spatial costs:')
for col in sorted(results_df.columns):
    if 'water' in col or 'land' in col or 'build' in col:
        print(f'  - {col}')

print(f'\nData for test location:')
row = results_df.iloc[0]
lcoa = row.get('lcoa_usd_per_t', 'N/A')
water = row.get('water_cost_usd_per_t', 'N/A')
land = row.get('land_cost_usd_per_t', 'N/A')
land_km2 = row.get('land_used_km2', 'N/A')

print(f'  LCOA (USD/t): {lcoa}')
print(f'  Water cost (USD/t): {water}')
print(f'  Land cost (USD/t): {land}')
print(f'  Land used (km²): {land_km2}')

print('\n✓ Notebook simulation completed successfully!')
