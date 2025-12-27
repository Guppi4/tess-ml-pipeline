"""Add TIC IDs to streaming catalog with quality metrics."""
import json
from pathlib import Path
from astroquery.mast import Catalogs
from astropy.coordinates import SkyCoord
import astropy.units as u
from tqdm import tqdm
import time

def add_tic_ids_to_catalog(catalog_path: str, output_path: str = None, search_radius: float = 40.0):
    """Add TIC IDs to catalog stars with quality metrics.

    Args:
        catalog_path: Path to catalog JSON
        output_path: Output path (default: overwrite input)
        search_radius: Search radius in arcsec (default 40, TESS PSF ~21"/px)

    Saves for each star:
        - tic_id: TIC identifier (or None)
        - tmag: TESS magnitude
        - sep_arcsec: Angular separation to matched TIC star
        - n_candidates: Number of TIC stars in search radius
        - match_quality: 'high' (<25"), 'medium' (25-40"), 'low' (>40")
    """
    print(f"Loading catalog: {catalog_path}")
    with open(catalog_path, 'r') as f:
        catalog = json.load(f)

    stars = catalog['stars']
    n_stars = len(stars)
    print(f"Stars to process: {n_stars}")
    print(f"Search radius: {search_radius} arcsec")

    matched = 0
    errors = 0
    sep_values = []

    for star_id, star_data in tqdm(stars.items(), desc="Querying TIC"):
        ra = star_data.get('ra')
        dec = star_data.get('dec')

        if ra is None or dec is None:
            continue

        try:
            coord = SkyCoord(ra=ra, dec=dec, unit='deg')
            result = Catalogs.query_region(
                coord,
                radius=search_radius * u.arcsec,
                catalog='TIC'
            )

            if len(result) > 0:
                # Sort by distance
                result = result[result['dstArcSec'].argsort()]

                tic_id = int(result[0]['ID'])
                tmag = float(result[0]['Tmag']) if result[0]['Tmag'] else None
                sep = float(result[0]['dstArcSec'])
                n_candidates = len(result)

                # Quality flag
                if sep < 25:
                    quality = 'high'
                elif sep < 40:
                    quality = 'medium'
                else:
                    quality = 'low'

                stars[star_id]['tic_id'] = tic_id
                stars[star_id]['tmag'] = tmag
                stars[star_id]['sep_arcsec'] = round(sep, 2)
                stars[star_id]['n_candidates'] = n_candidates
                stars[star_id]['match_quality'] = quality

                matched += 1
                sep_values.append(sep)
            else:
                stars[star_id]['tic_id'] = None
                stars[star_id]['tmag'] = None
                stars[star_id]['sep_arcsec'] = None
                stars[star_id]['n_candidates'] = 0
                stars[star_id]['match_quality'] = None

        except Exception as e:
            errors += 1
            stars[star_id]['tic_id'] = None
            stars[star_id]['tmag'] = None
            stars[star_id]['sep_arcsec'] = None
            stars[star_id]['n_candidates'] = 0
            stars[star_id]['match_quality'] = None

        # Rate limiting
        time.sleep(0.1)

    # Save updated catalog
    output = output_path or catalog_path
    with open(output, 'w') as f:
        json.dump(catalog, f, indent=2)

    # Statistics
    print(f"\nResults:")
    print(f"  Matched: {matched}/{n_stars} ({100*matched/n_stars:.1f}%)")
    print(f"  Errors: {errors}")

    if sep_values:
        import statistics
        print(f"\nSeparation statistics:")
        print(f"  Mean: {statistics.mean(sep_values):.1f} arcsec")
        print(f"  Median: {statistics.median(sep_values):.1f} arcsec")
        print(f"  Min: {min(sep_values):.1f} arcsec")
        print(f"  Max: {max(sep_values):.1f} arcsec")

        high = sum(1 for s in sep_values if s < 25)
        medium = sum(1 for s in sep_values if 25 <= s < 40)
        low = sum(1 for s in sep_values if s >= 40)
        print(f"\nQuality breakdown:")
        print(f"  High (<25\"): {high} ({100*high/len(sep_values):.1f}%)")
        print(f"  Medium (25-40\"): {medium} ({100*medium/len(sep_values):.1f}%)")
        print(f"  Low (>40\"): {low} ({100*low/len(sep_values):.1f}%)")

    print(f"\nSaved to: {output}")
    return matched


if __name__ == "__main__":
    catalog_path = "streaming_results/s0070_1-1_catalog.json"
    add_tic_ids_to_catalog(catalog_path, search_radius=40.0)
