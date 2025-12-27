"""Query TIC catalog for stellar parameters of our stars."""
import pandas as pd
import numpy as np
from astroquery.mast import Catalogs
import time

# Load our stars
df = pd.read_csv('variable_stars/s0070_1-1_all_stars.csv')

# Get stars with TIC IDs
stars_with_tic = df[df['tic_id'].notna()].copy()
print(f"Stars with TIC IDs: {len(stars_with_tic)}")

# TIC parameters we want
tic_params = ['Teff', 'logg', 'rad', 'mass', 'lumclass', 'objType', 'Tmag', 'd']

# Query TIC for each star
results = []
for idx, row in stars_with_tic.iterrows():
    tic_id = int(row['tic_id'])
    try:
        # Query TIC by ID
        tic_data = Catalogs.query_criteria(catalog="TIC", ID=tic_id)

        if len(tic_data) > 0:
            tic_row = tic_data[0]
            result = {
                'star_id': row['star_id'],
                'tic_id': tic_id,
                'ra': row['ra'],
                'dec': row['dec'],
                'amplitude_robust': row['amplitude_robust'],
                'variability_score': row['variability_score'],
            }

            # Add TIC params
            for param in tic_params:
                if param in tic_row.colnames:
                    val = tic_row[param]
                    # Handle masked values
                    if hasattr(val, 'mask') and val.mask:
                        result[param] = None
                    else:
                        result[param] = float(val) if not isinstance(val, str) else val
                else:
                    result[param] = None

            results.append(result)

            # Print progress
            teff = result.get('Teff', None)
            lum = result.get('lumclass', None)
            tmag = result.get('Tmag', None)
            print(f"TIC {tic_id}: Teff={teff}, lumclass={lum}, Tmag={tmag}")
        else:
            print(f"TIC {tic_id}: No data found")

    except Exception as e:
        print(f"TIC {tic_id}: Error - {e}")

    time.sleep(0.3)  # Rate limiting

# Create dataframe
results_df = pd.DataFrame(results)
results_df.to_csv('variable_stars/s0070_1-1_tic_params.csv', index=False)

print(f"\n{'='*60}")
print(f"Queried {len(results)} stars")
print(f"\nSummary:")
print(f"  Stars with Teff: {results_df['Teff'].notna().sum()}")
print(f"  Stars with logg: {results_df['logg'].notna().sum()}")
print(f"  Stars with lumclass: {results_df['lumclass'].notna().sum()}")

# Show Teff distribution
if results_df['Teff'].notna().any():
    print(f"\nTeff range: {results_df['Teff'].min():.0f} - {results_df['Teff'].max():.0f} K")

    # Spectral type estimate
    def teff_to_spectral(teff):
        if teff is None or pd.isna(teff):
            return '?'
        if teff > 30000: return 'O'
        if teff > 10000: return 'B'
        if teff > 7500: return 'A'
        if teff > 6000: return 'F'
        if teff > 5200: return 'G'
        if teff > 3700: return 'K'
        return 'M'

    results_df['spectral'] = results_df['Teff'].apply(teff_to_spectral)
    print("\nSpectral type distribution:")
    print(results_df['spectral'].value_counts())

# Show luminosity class
if results_df['lumclass'].notna().any():
    print("\nLuminosity class distribution:")
    print(results_df['lumclass'].value_counts())

# Show Tmag distribution
if results_df['Tmag'].notna().any():
    print(f"\nTmag range: {results_df['Tmag'].min():.1f} - {results_df['Tmag'].max():.1f}")
    print(f"  Bright (Tmag < 12): {(results_df['Tmag'] < 12).sum()}")
    print(f"  Medium (12-15): {((results_df['Tmag'] >= 12) & (results_df['Tmag'] < 15)).sum()}")
    print(f"  Faint (>= 15): {(results_df['Tmag'] >= 15).sum()}")

print("\nSaved to: variable_stars/s0070_1-1_tic_params.csv")
