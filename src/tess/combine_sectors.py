"""
Cross-sector star matching and lightcurve combination.

Combines photometry from multiple TESS sectors by matching stars via RA/Dec coordinates.
Designed for CVZ (Continuous Viewing Zone) analysis where the same stars appear
across many sectors.

Usage:
    from tess.combine_sectors import combine_cvz_sectors

    combined = combine_cvz_sectors(
        sectors=[61, 62, 63],
        camera="4",
        ccd="2",
        output_name="cvz_south"
    )
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from astropy.coordinates import SkyCoord
from astropy import units as u
from tqdm import tqdm

from .config import get_tess_sector_dir, DATA_DIR


def load_sector_data(sector: int, camera: str, ccd: str) -> Tuple[Dict, pd.DataFrame]:
    """
    Load catalog and photometry for a single sector.

    Returns:
        (catalog_dict, photometry_df) or (None, None) if not found
    """
    sector_dir = get_tess_sector_dir(sector, int(camera), int(ccd))

    # Find catalog file
    catalog_pattern = f"s{sector:04d}_{camera}-{ccd}_catalog.json"
    catalog_path = sector_dir / catalog_pattern

    if not catalog_path.exists():
        print(f"  Warning: Catalog not found for sector {sector}: {catalog_path}")
        return None, None

    # Find photometry file (prefer parquet for speed)
    parquet_path = sector_dir / f"s{sector:04d}_{camera}-{ccd}_photometry.parquet"
    csv_path = sector_dir / f"s{sector:04d}_{camera}-{ccd}_photometry.csv"

    if parquet_path.exists():
        photometry_df = pd.read_parquet(parquet_path)
    elif csv_path.exists():
        photometry_df = pd.read_csv(csv_path)
    else:
        print(f"  Warning: Photometry not found for sector {sector}")
        return None, None

    # Load catalog
    with open(catalog_path) as f:
        catalog_data = json.load(f)

    return catalog_data.get('stars', {}), photometry_df


def crossmatch_catalogs(
    catalogs: Dict[int, Dict],
    tolerance_arcsec: float = 3.0
) -> pd.DataFrame:
    """
    Cross-match star catalogs from multiple sectors.

    Args:
        catalogs: {sector: {star_id: {ra, dec, ...}}}
        tolerance_arcsec: matching tolerance in arcseconds

    Returns:
        DataFrame with columns: unified_id, sector, original_id, ra, dec
    """
    if not catalogs:
        return pd.DataFrame()

    # Use first sector as reference
    sectors = sorted(catalogs.keys())
    ref_sector = sectors[0]
    ref_catalog = catalogs[ref_sector]

    print(f"  Reference sector: {ref_sector} ({len(ref_catalog)} stars)")

    # Build reference coordinates
    ref_stars = []
    for star_id, info in ref_catalog.items():
        if info.get('ra') is not None and info.get('dec') is not None:
            ref_stars.append({
                'unified_id': star_id,  # Use original ID as unified
                'sector': ref_sector,
                'original_id': star_id,
                'ra': info['ra'],
                'dec': info['dec']
            })

    if not ref_stars:
        print("  Error: No stars with valid RA/Dec in reference sector")
        return pd.DataFrame()

    ref_df = pd.DataFrame(ref_stars)
    ref_coords = SkyCoord(ra=ref_df['ra'].values * u.deg, dec=ref_df['dec'].values * u.deg)

    # Match other sectors to reference
    all_matches = [ref_df]

    for sector in sectors[1:]:
        catalog = catalogs[sector]

        # Build coordinates for this sector
        sector_stars = []
        for star_id, info in catalog.items():
            if info.get('ra') is not None and info.get('dec') is not None:
                sector_stars.append({
                    'original_id': star_id,
                    'ra': info['ra'],
                    'dec': info['dec']
                })

        if not sector_stars:
            print(f"  Warning: No valid coordinates in sector {sector}")
            continue

        sector_df = pd.DataFrame(sector_stars)
        sector_coords = SkyCoord(ra=sector_df['ra'].values * u.deg, dec=sector_df['dec'].values * u.deg)

        # Cross-match
        idx, sep2d, _ = sector_coords.match_to_catalog_sky(ref_coords)

        # Filter by tolerance
        matches = sep2d < tolerance_arcsec * u.arcsec
        n_matched = matches.sum()

        print(f"  Sector {sector}: {len(sector_df)} stars, {n_matched} matched ({100*n_matched/len(sector_df):.1f}%)")

        # Build match table
        matched_df = sector_df[matches].copy()
        matched_df['unified_id'] = ref_df.iloc[idx[matches]]['unified_id'].values
        matched_df['sector'] = sector

        all_matches.append(matched_df[['unified_id', 'sector', 'original_id', 'ra', 'dec']])

    return pd.concat(all_matches, ignore_index=True)


def combine_photometry(
    match_table: pd.DataFrame,
    photometry_data: Dict[int, pd.DataFrame]
) -> pd.DataFrame:
    """
    Combine photometry from multiple sectors using match table.

    Args:
        match_table: output from crossmatch_catalogs
        photometry_data: {sector: photometry_df}

    Returns:
        Combined photometry DataFrame with unified star IDs
    """
    if match_table.empty:
        return pd.DataFrame()

    combined = []

    for sector in tqdm(photometry_data.keys(), desc="  Combining photometry"):
        phot_df = photometry_data[sector].copy()

        # Get mapping for this sector
        sector_matches = match_table[match_table['sector'] == sector]
        id_map = dict(zip(sector_matches['original_id'], sector_matches['unified_id']))

        # Filter to matched stars only
        phot_df = phot_df[phot_df['star_id'].isin(id_map.keys())].copy()

        # Replace original IDs with unified IDs
        phot_df['star_id'] = phot_df['star_id'].map(id_map)
        phot_df['sector'] = sector

        combined.append(phot_df)

    if not combined:
        return pd.DataFrame()

    result = pd.concat(combined, ignore_index=True)

    # Sort by star and time
    result = result.sort_values(['star_id', 'btjd']).reset_index(drop=True)

    return result


def combine_cvz_sectors(
    sectors: List[int],
    camera: str = "4",
    ccd: str = "2",
    output_name: str = "cvz_combined",
    tolerance_arcsec: float = 3.0,
    save: bool = True
) -> Dict:
    """
    Main function to combine multiple CVZ sectors.

    Args:
        sectors: list of sector numbers to combine
        camera: camera number (default "4" for CVZ)
        ccd: CCD number (default "2" for south CVZ)
        output_name: name for output files
        tolerance_arcsec: cross-match tolerance
        save: whether to save results to disk

    Returns:
        {
            'combined_photometry': DataFrame,
            'match_table': DataFrame,
            'stats': dict with statistics
        }
    """
    print(f"\n{'='*60}")
    print(f"Combining {len(sectors)} sectors: {sectors}")
    print(f"Camera {camera}, CCD {ccd}")
    print(f"{'='*60}\n")

    # Load all sector data
    print("Loading sector data...")
    catalogs = {}
    photometry = {}

    for sector in sectors:
        cat, phot = load_sector_data(sector, camera, ccd)
        if cat is not None:
            catalogs[sector] = cat
            photometry[sector] = phot
            print(f"  Sector {sector}: {len(cat)} stars, {len(phot)} measurements")

    if len(catalogs) < 2:
        print("Error: Need at least 2 sectors to combine")
        return None

    # Cross-match stars
    print(f"\nCross-matching stars (tolerance: {tolerance_arcsec} arcsec)...")
    match_table = crossmatch_catalogs(catalogs, tolerance_arcsec)

    if match_table.empty:
        print("Error: No matches found")
        return None

    # Count unique stars
    unique_stars = match_table['unified_id'].nunique()
    stars_in_all = match_table.groupby('unified_id')['sector'].nunique()
    stars_in_all_sectors = (stars_in_all == len(catalogs)).sum()

    print(f"\n  Unique matched stars: {unique_stars}")
    print(f"  Stars in ALL {len(catalogs)} sectors: {stars_in_all_sectors}")

    # Combine photometry
    print(f"\nCombining photometry...")
    combined_phot = combine_photometry(match_table, photometry)

    # Statistics
    stats = {
        'n_sectors': len(catalogs),
        'sectors': list(catalogs.keys()),
        'n_unique_stars': unique_stars,
        'n_stars_all_sectors': int(stars_in_all_sectors),
        'n_total_measurements': len(combined_phot),
        'btjd_min': float(combined_phot['btjd'].min()) if len(combined_phot) > 0 else None,
        'btjd_max': float(combined_phot['btjd'].max()) if len(combined_phot) > 0 else None,
    }

    if stats['btjd_min'] and stats['btjd_max']:
        stats['time_span_days'] = stats['btjd_max'] - stats['btjd_min']

    print(f"\n{'='*60}")
    print("Summary:")
    print(f"  Sectors combined: {stats['n_sectors']}")
    print(f"  Unique stars: {stats['n_unique_stars']}")
    print(f"  Stars in all sectors: {stats['n_stars_all_sectors']}")
    print(f"  Total measurements: {stats['n_total_measurements']:,}")
    if stats.get('time_span_days'):
        print(f"  Time span: {stats['time_span_days']:.1f} days")
    print(f"{'='*60}\n")

    # Save results
    if save:
        output_dir = DATA_DIR / "cvz_combined" / output_name
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save combined photometry
        phot_path = output_dir / "combined_photometry.csv"
        combined_phot.to_csv(phot_path, index=False)
        print(f"Saved: {phot_path}")

        # Try parquet for faster loading
        try:
            parquet_path = output_dir / "combined_photometry.parquet"
            combined_phot.to_parquet(parquet_path, index=False)
            print(f"Saved: {parquet_path}")
        except Exception:
            pass

        # Save match table
        match_path = output_dir / "star_matches.csv"
        match_table.to_csv(match_path, index=False)
        print(f"Saved: {match_path}")

        # Save stats
        stats_path = output_dir / "stats.json"
        with open(stats_path, 'w') as f:
            json.dump(stats, f, indent=2)
        print(f"Saved: {stats_path}")

    return {
        'combined_photometry': combined_phot,
        'match_table': match_table,
        'stats': stats
    }


def load_combined_cvz(name: str = "cvz_combined") -> pd.DataFrame:
    """Load previously combined CVZ data."""
    output_dir = DATA_DIR / "cvz_combined" / name

    # Prefer parquet
    parquet_path = output_dir / "combined_photometry.parquet"
    if parquet_path.exists():
        return pd.read_parquet(parquet_path)

    csv_path = output_dir / "combined_photometry.csv"
    if csv_path.exists():
        return pd.read_csv(csv_path)

    raise FileNotFoundError(f"No combined data found at {output_dir}")


if __name__ == "__main__":
    # Example: combine 3 CVZ sectors
    result = combine_cvz_sectors(
        sectors=[61, 62, 63],
        camera="4",
        ccd="2",
        output_name="south_cvz_test"
    )
