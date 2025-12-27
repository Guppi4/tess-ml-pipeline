"""Validate our variable star candidates against VSX (Variable Star Index)."""

import pandas as pd
import numpy as np
from pathlib import Path
from astroquery.vizier import Vizier
from astropy.coordinates import SkyCoord, match_coordinates_sky
import astropy.units as u

def validate_with_vsx(candidates_file: str, max_separation_arcsec: float = 60.0):
    """
    Cross-match our candidates with VSX known variables.

    Args:
        candidates_file: Path to our variable candidates CSV
        max_separation_arcsec: Maximum separation to consider a match

    Returns:
        DataFrame with VSX match info added
    """
    # Load our candidates
    candidates = pd.read_csv(candidates_file)
    print(f"Our candidates: {len(candidates)} stars")

    # Get coordinate bounds
    ra_values = candidates['ra'].values
    dec_values = candidates['dec'].values

    our_coords = SkyCoord(ra=ra_values, dec=dec_values, unit='deg')

    # Query VSX in relevant regions
    Vizier.ROW_LIMIT = -1

    # Handle RA wrap-around
    vsx_list = []

    # Region 1: RA 0-10
    if any((ra_values >= 0) & (ra_values <= 10)):
        print("Querying VSX region RA 0-10...")
        coord = SkyCoord(ra=5, dec=np.median(dec_values), unit='deg')
        result = Vizier.query_region(coord, radius=15*u.deg, catalog='B/vsx/vsx')
        if result:
            vsx_list.append(result[0].to_pandas())

    # Region 2: RA 350-360
    if any((ra_values >= 350) & (ra_values <= 360)):
        print("Querying VSX region RA 350-360...")
        coord = SkyCoord(ra=355, dec=np.median(dec_values), unit='deg')
        result = Vizier.query_region(coord, radius=15*u.deg, catalog='B/vsx/vsx')
        if result:
            vsx_list.append(result[0].to_pandas())

    if not vsx_list:
        print("No VSX data retrieved")
        return candidates

    vsx = pd.concat(vsx_list, ignore_index=True).drop_duplicates(subset='Name')
    print(f"VSX variables in field: {len(vsx)}")

    # Cross-match
    vsx_coords = SkyCoord(ra=vsx['RAJ2000'].values, dec=vsx['DEJ2000'].values, unit='deg')
    idx, sep, _ = match_coordinates_sky(our_coords, vsx_coords)

    # Add VSX info
    candidates['vsx_name'] = vsx.iloc[idx]['Name'].values
    candidates['vsx_type'] = vsx.iloc[idx]['Type'].values
    candidates['vsx_mag_max'] = vsx.iloc[idx]['max'].values
    candidates['vsx_mag_min'] = vsx.iloc[idx]['min'].values
    candidates['vsx_period'] = vsx.iloc[idx]['Period'].values
    candidates['vsx_sep_arcsec'] = sep.arcsec

    # Flag good matches
    candidates['vsx_match'] = candidates['vsx_sep_arcsec'] < max_separation_arcsec

    # Summary
    n_matches = candidates['vsx_match'].sum()
    print(f"\nMatches within {max_separation_arcsec} arcsec: {n_matches}")

    if n_matches > 0:
        print("\n" + "="*80)
        print("CONFIRMED VARIABLES (matches with VSX)")
        print("="*80)

        matches = candidates[candidates['vsx_match']]
        for _, row in matches.iterrows():
            print(f"\n{row['star_id']} -> {row['vsx_name']}")
            print(f"  Separation: {row['vsx_sep_arcsec']:.1f} arcsec")
            print(f"  VSX Type: {row['vsx_type']}")
            print(f"  VSX Mag: {row['vsx_mag_max']:.2f} - {row['vsx_mag_min']:.2f}")
            print(f"  VSX Period: {row['vsx_period']} days")
            print(f"  Our amplitude: {row['amplitude']*100:.1f}%")
            print(f"  Our period (LS): {row['period_1']:.4f} days")

    # Also show close misses
    close_misses = candidates[
        (candidates['vsx_sep_arcsec'] >= max_separation_arcsec) &
        (candidates['vsx_sep_arcsec'] < max_separation_arcsec * 2)
    ]

    if len(close_misses) > 0:
        print(f"\n\nPOSSIBLE MATCHES ({max_separation_arcsec}-{max_separation_arcsec*2} arcsec): {len(close_misses)}")
        for _, row in close_misses.head(5).iterrows():
            print(f"  {row['star_id']} -> {row['vsx_name']} ({row['vsx_type']}) @ {row['vsx_sep_arcsec']:.1f}\"")

    return candidates


if __name__ == "__main__":
    candidates_file = "variable_stars/s0070_1-1_variable_candidates.csv"

    # Run validation with 60 arcsec threshold (3 TESS pixels)
    result = validate_with_vsx(candidates_file, max_separation_arcsec=60.0)

    # Save result with VSX info
    output_file = "variable_stars/s0070_1-1_vsx_validated.csv"
    result.to_csv(output_file, index=False)
    print(f"\nSaved to: {output_file}")
