"""
FFI Downloader - Downloads TESS Full Frame Images from MAST archive.

Supports:
- Download by sector (auto-discovers available days)
- Download by date range
- Smart gap detection and logging
- Manifest with full metadata
"""

import json
import requests
from bs4 import BeautifulSoup
from tqdm import tqdm
from datetime import datetime, timedelta
from astropy.io import fits
from astropy.time import Time
from typing import List, Dict, Optional, Tuple

from config import FITS_DIR, MANIFEST_DIR, MAST_FFI_BASE_URL, ensure_directories


# =============================================================================
# TESS Sector Information
# =============================================================================

def get_sector_info(sector: int) -> Dict:
    """
    Get information about a TESS sector.

    Returns dict with available years, days, and date range.
    """
    sector_str = f"s{sector:04d}"
    url = f"{MAST_FFI_BASE_URL}/{sector_str}/"

    info = {
        'sector': sector,
        'sector_str': sector_str,
        'years': [],
        'days_by_year': {},
        'total_days': 0,
        'available': False
    }

    try:
        r = requests.get(url, timeout=30)
        if r.status_code == 404:
            return info
        r.raise_for_status()

        soup = BeautifulSoup(r.text, 'html.parser')

        # Find year directories
        for link in soup.find_all('a'):
            href = link.get('href', '')
            if href.endswith('/') and href[:-1].isdigit() and len(href[:-1]) == 4:
                year = href[:-1]
                info['years'].append(year)

        # Get days for each year
        for year in info['years']:
            year_url = f"{url}{year}/"
            try:
                r = requests.get(year_url, timeout=30)
                r.raise_for_status()
                soup = BeautifulSoup(r.text, 'html.parser')

                days = []
                for link in soup.find_all('a'):
                    href = link.get('href', '')
                    if href.endswith('/') and href[:-1].isdigit() and len(href[:-1]) == 3:
                        days.append(href[:-1])

                info['days_by_year'][year] = sorted(days)
                info['total_days'] += len(days)
            except:
                info['days_by_year'][year] = []

        info['available'] = info['total_days'] > 0

        # Calculate date range
        if info['available']:
            all_dates = []
            for year, days in info['days_by_year'].items():
                for day in days:
                    try:
                        date = datetime(int(year), 1, 1) + timedelta(days=int(day) - 1)
                        all_dates.append(date)
                    except:
                        pass

            if all_dates:
                info['date_start'] = min(all_dates).strftime('%Y-%m-%d')
                info['date_end'] = max(all_dates).strftime('%Y-%m-%d')

    except Exception as e:
        info['error'] = str(e)

    return info


def list_available_sectors() -> List[int]:
    """Get list of all available sectors from MAST."""
    url = f"{MAST_FFI_BASE_URL}/"

    try:
        r = requests.get(url, timeout=30)
        r.raise_for_status()
        soup = BeautifulSoup(r.text, 'html.parser')

        sectors = []
        for link in soup.find_all('a'):
            href = link.get('href', '')
            if href.startswith('s') and href.endswith('/'):
                try:
                    sector_num = int(href[1:-1])
                    sectors.append(sector_num)
                except:
                    pass

        return sorted(set(sectors))
    except:
        return []


def get_camera_ccd_combinations(sector: int, year: str, day: str) -> List[Tuple[str, str]]:
    """Get available camera-CCD combinations for a specific day."""
    url = f"{MAST_FFI_BASE_URL}/s{sector:04d}/{year}/{day}/"

    try:
        r = requests.get(url, timeout=30)
        if r.status_code == 404:
            return []
        r.raise_for_status()

        soup = BeautifulSoup(r.text, 'html.parser')

        combinations = []
        for link in soup.find_all('a'):
            href = link.get('href', '')
            if '-' in href and href.endswith('/'):
                parts = href[:-1].split('-')
                if len(parts) == 2 and parts[0].isdigit() and parts[1].isdigit():
                    combinations.append((parts[0], parts[1]))

        return sorted(combinations)
    except:
        return []


# =============================================================================
# File Download Functions
# =============================================================================

def get_mast_url(sector: int, year: str, day: str, camera: str, ccd: str) -> str:
    """Construct MAST FFI archive URL from observation parameters."""
    return f"{MAST_FFI_BASE_URL}/s{sector:04d}/{year}/{day}/{camera}-{ccd}/"


def verify_fits_file(filepath: str) -> Dict:
    """Verify downloaded FITS file and extract metadata."""
    try:
        with fits.open(filepath, mode='readonly') as hdu:
            header = hdu[0].header

            metadata = {
                'filename': filepath.split('/')[-1].split('\\')[-1],
                'filepath': str(filepath),
                'date_obs': header.get('DATE-OBS', ''),
                'date_end': header.get('DATE-END', ''),
                'tstart': header.get('TSTART', 0),
                'tstop': header.get('TSTOP', 0),
                'camera': header.get('CAMERA', 0),
                'ccd': header.get('CCD', 0),
                'sector': header.get('SECTOR', 0),
                'valid': True
            }

            if metadata['date_obs']:
                try:
                    metadata['mjd_obs'] = Time(metadata['date_obs'], format='isot', scale='utc').mjd
                except:
                    metadata['mjd_obs'] = None

            # TESS TSTART/TSTOP are in BTJD (days), need seconds
            metadata['exposure_time'] = header.get('EXPOSURE',
                (metadata['tstop'] - metadata['tstart']) * 86400.0)

            return metadata
    except Exception as e:
        return {
            'filename': filepath.split('/')[-1].split('\\')[-1],
            'filepath': str(filepath),
            'valid': False,
            'error': str(e)
        }


def download_fits_for_day(sector: int, year: str, day: str, camera: str, ccd: str,
                          skip_existing: bool = True) -> Tuple[List[Dict], bool]:
    """
    Download FITS files for a single day.

    Returns:
        Tuple of (metadata_list, day_exists)
    """
    url = get_mast_url(sector, year, day, camera, ccd)

    try:
        r = requests.get(url, timeout=30)
        if r.status_code == 404:
            return [], False  # Day doesn't exist
        r.raise_for_status()
    except requests.exceptions.HTTPError:
        return [], False
    except Exception as e:
        return [{'error': str(e), 'valid': False}], True

    soup = BeautifulSoup(r.text, 'html.parser')

    # Find calibrated FFI files (ffic)
    fits_links = []
    for link in soup.find_all('a'):
        href = link.get('href', '')
        if href.endswith('.fits') and 'ffic' in href:
            fits_links.append(href)

    if not fits_links:
        return [], True  # Day exists but no files

    downloaded_metadata = []

    for href in fits_links:
        file_url = url + href if not href.startswith('http') else href
        filename = href.split('/')[-1]
        output_path = FITS_DIR / filename

        if skip_existing and output_path.exists():
            metadata = verify_fits_file(str(output_path))
            metadata['skipped'] = True
            downloaded_metadata.append(metadata)
            continue

        try:
            r = requests.get(file_url, stream=True, timeout=120)
            r.raise_for_status()

            total_size = int(r.headers.get('content-length', 0))

            with open(output_path, 'wb') as f:
                with tqdm(total=total_size, unit='B', unit_scale=True,
                         desc=filename, leave=False) as pbar:
                    for chunk in r.iter_content(chunk_size=8192):
                        f.write(chunk)
                        pbar.update(len(chunk))

            metadata = verify_fits_file(str(output_path))
            metadata['skipped'] = False
            downloaded_metadata.append(metadata)

        except Exception as e:
            downloaded_metadata.append({
                'filename': filename,
                'valid': False,
                'error': str(e),
                'skipped': False
            })

    return downloaded_metadata, True


# =============================================================================
# Main Download Functions
# =============================================================================

def download_sector(sector: int, camera: str, ccd: str,
                   skip_existing: bool = True) -> Dict:
    """
    Download all available data for a sector.

    This is the RECOMMENDED way to download - it automatically:
    - Discovers all available days
    - Downloads only calibrated FFI files
    - Logs gaps in data
    - Creates comprehensive manifest

    Args:
        sector: TESS sector number (1-98+)
        camera: Camera number (1-4)
        ccd: CCD number (1-4)
        skip_existing: Skip files that already exist

    Returns:
        Manifest dictionary with all metadata
    """
    ensure_directories()

    print(f"\n{'='*60}")
    print(f"TESS FFI Downloader - Sector {sector}, Camera {camera}, CCD {ccd}")
    print(f"{'='*60}")

    # Get sector info
    print("\nDiscovering available data...")
    sector_info = get_sector_info(sector)

    if not sector_info['available']:
        print(f"Sector {sector} not available on MAST")
        return {'error': 'Sector not available', 'sector': sector}

    print(f"  Date range: {sector_info.get('date_start', 'N/A')} to {sector_info.get('date_end', 'N/A')}")
    print(f"  Total observation days: {sector_info['total_days']}")

    # Build list of all days to download
    all_days = []
    for year, days in sector_info['days_by_year'].items():
        for day in days:
            all_days.append((year, day))

    all_days.sort(key=lambda x: (x[0], int(x[1])))

    # Download each day
    all_metadata = []
    missing_days = []
    available_days = []

    print(f"\nDownloading {len(all_days)} days of data...")

    for year, day in tqdm(all_days, desc="Days", unit="day"):
        day_metadata, day_exists = download_fits_for_day(
            sector, year, day, camera, ccd, skip_existing
        )

        if day_exists:
            available_days.append({'year': year, 'day': day})
            all_metadata.extend(day_metadata)
        else:
            missing_days.append({'year': year, 'day': day})

    # Analyze gaps in time coverage
    gaps = analyze_time_gaps(all_metadata)

    # Create manifest
    manifest = create_sector_manifest(
        sector=sector,
        camera=camera,
        ccd=ccd,
        sector_info=sector_info,
        files=all_metadata,
        missing_days=missing_days,
        gaps=gaps
    )

    # Summary
    valid_files = [m for m in all_metadata if m.get('valid', False)]

    print(f"\n{'='*60}")
    print("Download Summary:")
    print(f"  Valid files: {len(valid_files)}")
    print(f"  Missing days: {len(missing_days)}")
    print(f"  Time gaps detected: {len(gaps)}")
    print(f"  Manifest: {manifest['manifest_path']}")
    print(f"{'='*60}")

    return manifest


def download_by_date_range(start_date: str, end_date: str,
                           camera: str, ccd: str,
                           skip_existing: bool = True) -> Dict:
    """
    Download data by calendar date range.

    Args:
        start_date: Start date as 'YYYY-MM-DD'
        end_date: End date as 'YYYY-MM-DD'
        camera, ccd: Camera and CCD numbers
        skip_existing: Skip existing files

    Returns:
        Manifest dictionary
    """
    ensure_directories()

    # Parse dates
    start = datetime.strptime(start_date, '%Y-%m-%d')
    end = datetime.strptime(end_date, '%Y-%m-%d')

    print(f"\nDownloading data from {start_date} to {end_date}")

    # Find which sectors cover this date range
    # (This is approximate - would need sector start/end dates for precision)
    sectors_to_check = list_available_sectors()

    all_metadata = []
    all_missing = []

    current = start
    while current <= end:
        year = str(current.year)
        day = str(current.timetuple().tm_yday).zfill(3)

        # Try each sector until we find data
        for sector in sectors_to_check:
            metadata, exists = download_fits_for_day(
                sector, year, day, camera, ccd, skip_existing
            )
            if exists and metadata:
                all_metadata.extend(metadata)
                break
        else:
            all_missing.append({'date': current.strftime('%Y-%m-%d'), 'year': year, 'day': day})

        current += timedelta(days=1)

    # Create manifest
    gaps = analyze_time_gaps(all_metadata)

    manifest = {
        'created': datetime.now().isoformat(),
        'mode': 'date_range',
        'parameters': {
            'start_date': start_date,
            'end_date': end_date,
            'camera': camera,
            'ccd': ccd
        },
        'summary': {
            'total_files': len(all_metadata),
            'valid_files': sum(1 for m in all_metadata if m.get('valid', False)),
            'missing_days': len(all_missing),
            'gaps': len(gaps)
        },
        'files': all_metadata,
        'missing_days': all_missing,
        'gaps': gaps
    }

    # Save manifest
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    manifest_path = MANIFEST_DIR / f"manifest_dates_{start_date}_{end_date}_{timestamp}.json"
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2, default=str)
    manifest['manifest_path'] = str(manifest_path)

    return manifest


# =============================================================================
# Gap Analysis
# =============================================================================

def analyze_time_gaps(metadata_list: List[Dict], max_gap_hours: float = 2.0) -> List[Dict]:
    """
    Analyze time gaps in observations.

    Args:
        metadata_list: List of file metadata with mjd_obs
        max_gap_hours: Maximum expected gap between observations (TESS FFI cadence)

    Returns:
        List of gap dictionaries with start/end times and duration
    """
    # Get valid MJD values
    mjd_list = []
    for m in metadata_list:
        if m.get('valid') and m.get('mjd_obs'):
            mjd_list.append(m['mjd_obs'])

    if len(mjd_list) < 2:
        return []

    mjd_list.sort()

    gaps = []
    expected_cadence = max_gap_hours / 24.0  # Convert to days

    for i in range(len(mjd_list) - 1):
        gap_days = mjd_list[i + 1] - mjd_list[i]

        # If gap is significantly larger than expected cadence
        if gap_days > expected_cadence * 1.5:
            gaps.append({
                'gap_index': i,
                'mjd_start': mjd_list[i],
                'mjd_end': mjd_list[i + 1],
                'gap_hours': gap_days * 24,
                'gap_days': gap_days,
                'expected_observations_missed': int(gap_days / expected_cadence) - 1
            })

    return gaps


def create_sector_manifest(sector: int, camera: str, ccd: str,
                          sector_info: Dict, files: List[Dict],
                          missing_days: List[Dict], gaps: List[Dict]) -> Dict:
    """Create comprehensive manifest for a sector download."""

    valid_files = [f for f in files if f.get('valid', False)]
    invalid_files = [f for f in files if not f.get('valid', False)]

    # Calculate time coverage
    mjd_values = [f['mjd_obs'] for f in valid_files if f.get('mjd_obs')]

    manifest = {
        'created': datetime.now().isoformat(),
        'mode': 'sector',
        'parameters': {
            'sector': sector,
            'camera': camera,
            'ccd': ccd
        },
        'sector_info': sector_info,
        'summary': {
            'total_files': len(files),
            'valid_files': len(valid_files),
            'invalid_files': len(invalid_files),
            'missing_days': len(missing_days),
            'gaps_detected': len(gaps),
            'mjd_start': min(mjd_values) if mjd_values else None,
            'mjd_end': max(mjd_values) if mjd_values else None,
            'time_span_days': (max(mjd_values) - min(mjd_values)) if len(mjd_values) > 1 else 0,
            'completeness': len(valid_files) / (len(valid_files) + len(missing_days))
                           if (len(valid_files) + len(missing_days)) > 0 else 0
        },
        'files': valid_files,
        'invalid_files': invalid_files,
        'missing_days': missing_days,
        'gaps': gaps
    }

    # Save manifest
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    manifest_name = f"manifest_s{sector:04d}_{camera}-{ccd}_{timestamp}.json"
    manifest_path = MANIFEST_DIR / manifest_name

    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2, default=str)

    manifest['manifest_path'] = str(manifest_path)

    return manifest


# =============================================================================
# Legacy Interface (backward compatibility)
# =============================================================================

def download_fits(inputs: Dict) -> List[str]:
    """
    Legacy interface for backward compatibility.

    Supports both old format (sector, year, day) and new format (sector only).
    """
    ensure_directories()

    sector = int(inputs.get('sector', 0))
    camera = str(inputs.get('camera', '1'))
    ccd = str(inputs.get('ccd', '1'))

    # New mode: download entire sector
    if 'day' not in inputs and 'day_start' not in inputs:
        manifest = download_sector(sector, camera, ccd)
        return [f['filepath'] for f in manifest.get('files', []) if f.get('valid')]

    # Old mode: specific day or range
    year = str(inputs.get('year', ''))

    if 'day_start' in inputs and 'day_end' in inputs:
        day_start = int(inputs['day_start'])
        day_end = int(inputs['day_end'])
        days = [str(d).zfill(3) for d in range(day_start, day_end + 1)]
    else:
        days = [str(inputs['day']).zfill(3)]

    all_metadata = []
    for day in tqdm(days, desc="Days"):
        metadata, _ = download_fits_for_day(sector, year, day, camera, ccd)
        all_metadata.extend(metadata)

    return [m['filepath'] for m in all_metadata if m.get('valid')]


def load_manifest(manifest_path: str) -> Dict:
    """Load a previously created manifest."""
    with open(manifest_path, 'r') as f:
        return json.load(f)


def get_latest_manifest(sector: int = None) -> Optional[Dict]:
    """Find and load the most recent manifest."""
    ensure_directories()

    pattern = f"manifest_s{sector:04d}_*.json" if sector else "manifest_*.json"
    manifests = list(MANIFEST_DIR.glob(pattern))

    if not manifests:
        return None

    latest = max(manifests, key=lambda x: x.stat().st_mtime)
    return load_manifest(str(latest))
