"""
Sector Analyzer - Check sector data quality BEFORE downloading.

This helps you avoid wasting time on sectors with:
- Too many gaps
- Missing days
- Poor coverage

Run this BEFORE starting a download to pick the best sector.
"""

import requests
from bs4 import BeautifulSoup
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import json

from config import MAST_FFI_BASE_URL, MANIFEST_DIR, ensure_directories


@dataclass
class SectorQuality:
    """Quality assessment for a sector."""
    sector: int
    available: bool
    total_days: int
    expected_days: int  # TESS sector is ~27 days
    missing_days: int
    coverage_percent: float
    largest_gap_days: int
    date_start: str
    date_end: str
    quality_score: str  # "excellent", "good", "fair", "poor"
    recommendation: str
    details: Dict


def get_sector_quality(sector: int, camera: str = "1", ccd: str = "1",
                       verbose: bool = True) -> SectorQuality:
    """
    Analyze sector data quality WITHOUT downloading anything.

    This checks:
    - How many days of data exist
    - Gaps in coverage
    - Expected vs actual days
    - Overall quality score

    Args:
        sector: TESS sector number
        camera: Camera (1-4)
        ccd: CCD (1-4)
        verbose: Print progress

    Returns:
        SectorQuality object with assessment
    """
    if verbose:
        print(f"\nAnalyzing Sector {sector} (Camera {camera}, CCD {ccd})...")
        print("-" * 50)

    # Get sector structure
    sector_str = f"s{sector:04d}"
    base_url = f"{MAST_FFI_BASE_URL}/{sector_str}/"

    result = SectorQuality(
        sector=sector,
        available=False,
        total_days=0,
        expected_days=27,  # Standard TESS sector length
        missing_days=0,
        coverage_percent=0.0,
        largest_gap_days=0,
        date_start="",
        date_end="",
        quality_score="unknown",
        recommendation="",
        details={}
    )

    try:
        # Check if sector exists
        r = requests.get(base_url, timeout=30)
        if r.status_code == 404:
            result.recommendation = f"Sector {sector} does not exist on MAST"
            if verbose:
                print(f"  Sector {sector} NOT FOUND")
            return result

        r.raise_for_status()
        soup = BeautifulSoup(r.text, 'html.parser')

        # Find year directories
        years = []
        for link in soup.find_all('a'):
            href = link.get('href', '')
            if href.endswith('/') and href[:-1].isdigit() and len(href[:-1]) == 4:
                years.append(href[:-1])

        if not years:
            result.recommendation = "No data found in sector"
            return result

        if verbose:
            print(f"  Years found: {', '.join(years)}")

        # Collect all available days
        all_days = []  # List of (year, day_of_year, date)

        for year in years:
            year_url = f"{base_url}{year}/"
            try:
                r = requests.get(year_url, timeout=30)
                r.raise_for_status()
                soup = BeautifulSoup(r.text, 'html.parser')

                for link in soup.find_all('a'):
                    href = link.get('href', '')
                    if href.endswith('/') and href[:-1].isdigit() and len(href[:-1]) == 3:
                        day_of_year = int(href[:-1])
                        # Convert to actual date
                        date = datetime(int(year), 1, 1) + timedelta(days=day_of_year - 1)
                        all_days.append({
                            'year': year,
                            'day_of_year': day_of_year,
                            'date': date,
                            'date_str': date.strftime('%Y-%m-%d')
                        })
            except Exception as e:
                if verbose:
                    print(f"  Warning: Could not fetch {year}: {e}")

        if not all_days:
            result.recommendation = "No observation days found"
            return result

        # Sort by date
        all_days.sort(key=lambda x: x['date'])

        result.available = True
        result.total_days = len(all_days)
        result.date_start = all_days[0]['date_str']
        result.date_end = all_days[-1]['date_str']

        if verbose:
            print(f"  Date range: {result.date_start} to {result.date_end}")
            print(f"  Total days with data: {result.total_days}")

        # Analyze gaps
        gaps = analyze_gaps(all_days)
        result.details['gaps'] = gaps

        if gaps:
            result.largest_gap_days = max(g['gap_days'] for g in gaps)
            total_gap_days = sum(g['gap_days'] for g in gaps)
            result.missing_days = total_gap_days

            if verbose:
                print(f"  Gaps detected: {len(gaps)}")
                print(f"  Largest gap: {result.largest_gap_days} days")
                if len(gaps) <= 5:
                    for g in gaps:
                        print(f"    - {g['start']} to {g['end']} ({g['gap_days']} days)")

        # Calculate expected coverage
        date_range = (all_days[-1]['date'] - all_days[0]['date']).days + 1
        result.expected_days = min(date_range, 30)  # Cap at 30
        result.coverage_percent = (result.total_days / result.expected_days) * 100

        if verbose:
            print(f"  Coverage: {result.coverage_percent:.1f}%")

        # Check file availability for specific camera/ccd
        files_info = check_files_availability(sector, all_days[0], camera, ccd)
        result.details['files_per_day'] = files_info.get('files_count', 0)

        if verbose and files_info.get('files_count', 0) > 0:
            print(f"  Files per day: ~{files_info['files_count']}")

        # Calculate quality score
        result.quality_score, result.recommendation = calculate_quality_score(result)

        if verbose:
            print(f"\n  Quality Score: {result.quality_score.upper()}")
            print(f"  Recommendation: {result.recommendation}")

        return result

    except requests.exceptions.RequestException as e:
        result.recommendation = f"Network error: {e}"
        if verbose:
            print(f"  Error: {e}")
        return result


def analyze_gaps(days: List[Dict]) -> List[Dict]:
    """Find gaps in observation days."""
    gaps = []

    for i in range(len(days) - 1):
        current = days[i]['date']
        next_day = days[i + 1]['date']
        diff = (next_day - current).days

        if diff > 1:  # Gap detected
            gaps.append({
                'start': days[i]['date_str'],
                'end': days[i + 1]['date_str'],
                'gap_days': diff - 1,
                'after_day': days[i]['day_of_year']
            })

    return gaps


def check_files_availability(sector: int, sample_day: Dict,
                             camera: str, ccd: str) -> Dict:
    """Check how many files are available for a sample day."""
    url = f"{MAST_FFI_BASE_URL}/s{sector:04d}/{sample_day['year']}/{sample_day['day_of_year']:03d}/{camera}-{ccd}/"

    try:
        r = requests.get(url, timeout=30)
        if r.status_code == 404:
            return {'files_count': 0, 'error': 'Directory not found'}

        r.raise_for_status()
        soup = BeautifulSoup(r.text, 'html.parser')

        files = []
        for link in soup.find_all('a'):
            href = link.get('href', '')
            if href.endswith('.fits') and 'ffic' in href:
                files.append(href)

        return {
            'files_count': len(files),
            'sample_day': sample_day['date_str']
        }
    except:
        return {'files_count': 0, 'error': 'Could not check'}


def calculate_quality_score(result: SectorQuality) -> Tuple[str, str]:
    """Calculate overall quality score and recommendation."""
    coverage = result.coverage_percent
    largest_gap = result.largest_gap_days
    total_days = result.total_days

    # Scoring logic
    if coverage >= 90 and largest_gap <= 1 and total_days >= 25:
        return "excellent", "Perfect for lightcurve analysis. No significant gaps."

    elif coverage >= 80 and largest_gap <= 3 and total_days >= 20:
        return "good", "Good quality data. Minor gaps won't affect most analysis."

    elif coverage >= 60 and largest_gap <= 7 and total_days >= 15:
        return "fair", "Usable but has gaps. May miss some periodic signals."

    elif coverage >= 40 and total_days >= 10:
        return "poor", "Significant gaps. Only suitable for bright variable stars."

    else:
        return "bad", "Too many gaps or insufficient data. Consider another sector."


def compare_sectors(sectors: List[int], camera: str = "1", ccd: str = "1") -> List[SectorQuality]:
    """
    Compare multiple sectors to find the best one.

    Args:
        sectors: List of sector numbers to compare
        camera: Camera number
        ccd: CCD number

    Returns:
        List of SectorQuality sorted by quality (best first)
    """
    print("\n" + "=" * 60)
    print("SECTOR COMPARISON")
    print("=" * 60)

    results = []

    for sector in sectors:
        quality = get_sector_quality(sector, camera, ccd, verbose=False)
        results.append(quality)
        print(f"  Sector {sector:2d}: {quality.quality_score:10s} | "
              f"{quality.total_days:2d} days | "
              f"{quality.coverage_percent:5.1f}% coverage | "
              f"max gap: {quality.largest_gap_days}d")

    # Sort by quality
    quality_order = {'excellent': 0, 'good': 1, 'fair': 2, 'poor': 3, 'bad': 4, 'unknown': 5}
    results.sort(key=lambda x: (quality_order.get(x.quality_score, 5), -x.coverage_percent))

    print("\n" + "-" * 60)
    if results and results[0].available:
        best = results[0]
        print(f"RECOMMENDED: Sector {best.sector}")
        print(f"  {best.recommendation}")
    else:
        print("No suitable sectors found in the given range.")

    return results


def find_best_sector(start_sector: int = 1, end_sector: int = 98,
                     camera: str = "1", ccd: str = "1",
                     min_quality: str = "good") -> Optional[SectorQuality]:
    """
    Find the best available sector in a range.

    Args:
        start_sector: Start of range
        end_sector: End of range
        camera: Camera number
        ccd: CCD number
        min_quality: Minimum acceptable quality ("excellent", "good", "fair", "poor")

    Returns:
        Best SectorQuality or None
    """
    quality_order = {'excellent': 0, 'good': 1, 'fair': 2, 'poor': 3, 'bad': 4}
    min_quality_num = quality_order.get(min_quality, 1)

    print(f"\nSearching for best sector ({start_sector}-{end_sector})...")
    print(f"Minimum quality: {min_quality}")
    print("-" * 50)

    best = None

    for sector in range(start_sector, end_sector + 1):
        try:
            quality = get_sector_quality(sector, camera, ccd, verbose=False)

            if not quality.available:
                continue

            quality_num = quality_order.get(quality.quality_score, 5)

            if quality_num <= min_quality_num:
                if best is None or quality_num < quality_order.get(best.quality_score, 5):
                    best = quality
                    print(f"  Found: Sector {sector} ({quality.quality_score})")

                    if quality.quality_score == "excellent":
                        break  # Can't do better

        except Exception as e:
            continue

    if best:
        print(f"\nBest sector: {best.sector}")
        print(f"  Quality: {best.quality_score}")
        print(f"  Coverage: {best.coverage_percent:.1f}%")
        print(f"  Days: {best.total_days}")
    else:
        print("\nNo suitable sector found.")

    return best


def save_sector_report(quality: SectorQuality, filename: str = None):
    """Save sector quality report to file."""
    ensure_directories()

    if filename is None:
        filename = f"sector_{quality.sector:04d}_report.json"

    filepath = MANIFEST_DIR / filename

    report = {
        'sector': quality.sector,
        'available': quality.available,
        'quality_score': quality.quality_score,
        'recommendation': quality.recommendation,
        'coverage_percent': quality.coverage_percent,
        'total_days': quality.total_days,
        'missing_days': quality.missing_days,
        'largest_gap_days': quality.largest_gap_days,
        'date_start': quality.date_start,
        'date_end': quality.date_end,
        'details': quality.details,
        'analyzed_at': datetime.now().isoformat()
    }

    with open(filepath, 'w') as f:
        json.dump(report, f, indent=2)

    print(f"Report saved to: {filepath}")
    return filepath


# Quick check function for command line
def quick_check(sector: int):
    """Quick quality check for a sector."""
    print("\n" + "=" * 60)
    print(f"QUICK SECTOR CHECK: Sector {sector}")
    print("=" * 60)

    quality = get_sector_quality(sector, verbose=True)

    print("\n" + "=" * 60)

    if quality.quality_score in ['excellent', 'good']:
        print("✓ This sector is RECOMMENDED for your analysis")
    elif quality.quality_score == 'fair':
        print("~ This sector is USABLE but has some gaps")
    else:
        print("✗ This sector is NOT RECOMMENDED - try another one")

    return quality
