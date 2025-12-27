"""
Streaming Pipeline - Process TESS data on-the-fly without storing raw files.

This solves the storage problem:
- Downloads one file at a time
- Extracts photometry
- Deletes the raw file
- Keeps only results (~100 MB instead of ~10 GB)

Supports:
- Resume from interruption
- Progress checkpoints
- Memory-efficient processing
"""

import json
import os
import tempfile
import warnings
import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup
from tqdm import tqdm
from datetime import datetime
from pathlib import Path
from astropy.io import fits
from astropy.stats import sigma_clipped_stats
from astropy.wcs import WCS
from astropy.time import Time
from photutils.detection import DAOStarFinder
from photutils.aperture import CircularAperture, CircularAnnulus, aperture_photometry
from typing import Dict, List, Optional, Tuple

# Suppress WCS warnings about NaN values during coordinate conversion
warnings.filterwarnings('ignore', message='All-NaN slice encountered', category=RuntimeWarning)
# Suppress FITS header fixup warnings (MJD-OBS from DATE-OBS, etc.)
warnings.filterwarnings('ignore', message='.*datfix.*', category=Warning)
warnings.filterwarnings('ignore', category=FutureWarning)

import sys
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

# Async downloader for high-performance parallel downloads
try:
    from .AsyncDownloader import AsyncDownloader, download_files_async
    ASYNC_AVAILABLE = True
except ImportError:
    ASYNC_AVAILABLE = False


class ProgressDisplay:
    """
    Clean progress display for streaming pipeline.
    Shows parameters, live stats, and final summary.
    """

    def __init__(self, sector: int, camera: str, ccd: str, total_files: int, already_processed: int):
        self.sector = sector
        self.camera = camera
        self.ccd = ccd
        self.total_files = total_files
        self.already_processed = already_processed
        self.remaining = total_files - already_processed

        # Stats
        self.processed = 0
        self.successful = 0
        self.failed = 0
        self.stars_detected = 0
        self.measurements = 0
        self.start_time = None
        self.errors = []  # Store errors silently
        self.pbar = None
        self._lock = threading.Lock()  # Thread safety for parallel updates

        # Terminal width
        self.width = min(shutil.get_terminal_size().columns, 80)

    def show_header(self):
        """Display initial parameters."""
        print()
        print("+" + "-" * (self.width - 2) + "+")
        print("|" + " TESS Streaming Pipeline ".center(self.width - 2) + "|")
        print("+" + "-" * (self.width - 2) + "+")
        print("|" + f"  Sector: {self.sector}".ljust(self.width - 2) + "|")
        print("|" + f"  Camera: {self.camera}    CCD: {self.ccd}".ljust(self.width - 2) + "|")
        print("+" + "-" * (self.width - 2) + "+")
        print("|" + f"  Total files:     {self.total_files:,}".ljust(self.width - 2) + "|")
        print("|" + f"  Already done:    {self.already_processed:,}".ljust(self.width - 2) + "|")
        print("|" + f"  To process:      {self.remaining:,}".ljust(self.width - 2) + "|")
        print("+" + "-" * (self.width - 2) + "+")
        print()

    def start(self):
        """Start timing and progress bar."""
        self.start_time = datetime.now()
        self.pbar = tqdm(
            total=self.remaining,
            desc="Processing",
            unit="file",
            ncols=100,
            bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]'
        )

    def update(self, success: bool, stars: int = 0, measurements: int = 0, error: str = None):
        """Update stats after processing a file (thread-safe)."""
        with self._lock:
            self.processed += 1
            if success:
                self.successful += 1
                self.stars_detected = max(self.stars_detected, stars)
                self.measurements += measurements
            else:
                self.failed += 1
                if error:
                    self.errors.append(error)

            if self.pbar:
                self.pbar.update(1)
                self.pbar.set_postfix({
                    'ok': self.successful,
                    'fail': self.failed,
                    'stars': f'{self.stars_detected:,}'
                }, refresh=True)

    def close(self):
        """Close progress bar."""
        if self.pbar:
            self.pbar.close()

    def show_summary(self, output_dir: Path, catalog_file: str, data_file: str, data_size_mb: float):
        """Display final summary."""
        elapsed = (datetime.now() - self.start_time).total_seconds() if self.start_time else 0

        print()
        print()
        print("╔" + "═" * (self.width - 2) + "╗")

        # Status header
        if self.failed == 0:
            status_text = " ✓ COMPLETED SUCCESSFULLY "
        elif self.failed < self.processed * 0.1:
            status_text = " ✓ COMPLETED (minor errors) "
        else:
            status_text = " ⚠ COMPLETED WITH ERRORS "

        print("║" + status_text.center(self.width - 2) + "║")
        print("╠" + "═" * (self.width - 2) + "╣")

        # Results
        print("║" + "  RESULTS:".ljust(self.width - 2) + "║")
        print("║" + f"    Stars detected:    {self.stars_detected:,}".ljust(self.width - 2) + "║")
        print("║" + f"    Epochs processed:  {self.successful:,}".ljust(self.width - 2) + "║")
        print("║" + f"    Total measurements:{self.measurements:,}".ljust(self.width - 2) + "║")
        print("╠" + "─" * (self.width - 2) + "╣")

        # Stats
        print("║" + "  STATS:".ljust(self.width - 2) + "║")
        print("║" + f"    Successful: {self.successful:,}  |  Failed: {self.failed:,}".ljust(self.width - 2) + "║")

        # Time
        if elapsed < 60:
            time_str = f"{elapsed:.1f} seconds"
        elif elapsed < 3600:
            time_str = f"{elapsed/60:.1f} minutes"
        else:
            time_str = f"{elapsed/3600:.1f} hours"
        print("║" + f"    Total time: {time_str}".ljust(self.width - 2) + "║")

        if self.successful > 0 and elapsed > 0:
            rate = self.successful / elapsed * 60
            print("║" + f"    Speed: {rate:.1f} files/min".ljust(self.width - 2) + "║")

        print("╠" + "─" * (self.width - 2) + "╣")

        # Output files
        print("║" + "  OUTPUT FILES:".ljust(self.width - 2) + "║")
        print("║" + f"    {output_dir}/".ljust(self.width - 2) + "║")
        print("║" + f"      ├─ {catalog_file}".ljust(self.width - 2) + "║")
        print("║" + f"      └─ {data_file} ({data_size_mb:.1f} MB)".ljust(self.width - 2) + "║")

        print("╚" + "═" * (self.width - 2) + "╝")

        # Show errors summary if any
        if self.errors:
            print()
            print(f"  ⚠ {len(self.errors)} files had errors (see log for details)")


from .config import (
    BASE_DIR, PHOTOMETRY_RESULTS_DIR, MANIFEST_DIR,
    MAST_FFI_BASE_URL, DAOFIND_FWHM, DAOFIND_THRESHOLD_SIGMA,
    DEFAULT_APERTURE_RADIUS, ANNULUS_R_IN, ANNULUS_R_OUT, ensure_directories
)


# Streaming results directory
STREAMING_DIR = BASE_DIR / "streaming_results"


def ensure_streaming_dirs():
    """Create streaming directories."""
    STREAMING_DIR.mkdir(parents=True, exist_ok=True)
    (STREAMING_DIR / "checkpoints").mkdir(exist_ok=True)


class StreamingProcessor:
    """
    Process TESS FFI data in streaming mode.

    Downloads one file at a time, extracts photometry, deletes raw file.
    Results are accumulated and saved periodically.
    """

    def __init__(self, sector: int, camera: str, ccd: str):
        self.sector = sector
        self.camera = camera
        self.ccd = ccd

        self.session_id = f"s{sector:04d}_{camera}-{ccd}"

        # Results storage
        self.star_catalog = {}  # star_id -> {ra, dec, ...}
        self.photometry_records = []  # List of measurement records
        self.processed_files = set()  # Track what's done
        self.epoch_metadata = []  # Metadata for each epoch

        # Reference epoch data (for cross-matching)
        self.reference_stars = None
        self.reference_epoch_idx = None

        # Checkpoint
        self.checkpoint_path = STREAMING_DIR / "checkpoints" / f"{self.session_id}_checkpoint.json"

        ensure_streaming_dirs()

    def load_checkpoint(self) -> bool:
        """Load previous progress if exists."""
        if self.checkpoint_path.exists():
            try:
                with open(self.checkpoint_path, 'r') as f:
                    data = json.load(f)

                self.processed_files = set(data.get('processed_files', []))
                self.epoch_metadata = data.get('epoch_metadata', [])
                self.star_catalog = data.get('star_catalog', {})
                self.reference_epoch_idx = data.get('reference_epoch_idx')

                # Load photometry from CSV (fallback to parquet for old checkpoints)
                csv_path = STREAMING_DIR / f"{self.session_id}_photometry_checkpoint.csv"
                parquet_path = STREAMING_DIR / f"{self.session_id}_photometry.parquet"

                if csv_path.exists():
                    df = pd.read_csv(csv_path)
                    self.photometry_records = df.to_dict('records')
                elif parquet_path.exists():
                    try:
                        df = pd.read_parquet(parquet_path)
                        self.photometry_records = df.to_dict('records')
                    except ImportError:
                        print("Warning: pyarrow not installed, cannot load old parquet checkpoint")

                # Restore reference_stars flag if we have a star catalog
                if self.star_catalog:
                    self.reference_stars = True  # Flag that catalog exists

                print(f"Resumed from checkpoint: {len(self.processed_files)} files already processed")
                return True
            except Exception as e:
                print(f"Could not load checkpoint: {e}")
        return False

    def save_checkpoint(self):
        """Save current progress."""
        checkpoint_data = {
            'session_id': self.session_id,
            'sector': self.sector,
            'camera': self.camera,
            'ccd': self.ccd,
            'processed_files': list(self.processed_files),
            'epoch_metadata': self.epoch_metadata,
            'star_catalog': self.star_catalog,
            'reference_epoch_idx': self.reference_epoch_idx,
            'last_update': datetime.now().isoformat()
        }

        with open(self.checkpoint_path, 'w') as f:
            json.dump(checkpoint_data, f, indent=2, default=str)

        # Save photometry to CSV (no external dependencies)
        if self.photometry_records:
            df = pd.DataFrame(self.photometry_records)
            csv_path = STREAMING_DIR / f"{self.session_id}_photometry_checkpoint.csv"
            df.to_csv(csv_path, index=False)

    def get_sector_files(self) -> List[Dict]:
        """Get list of all FFI files for this sector/camera/ccd."""
        from .FFIDownloader import get_sector_info

        sector_info = get_sector_info(self.sector)

        if not sector_info['available']:
            return []

        files = []
        for year, days in sector_info['days_by_year'].items():
            for day in days:
                # Get file list for this day
                url = f"{MAST_FFI_BASE_URL}/s{self.sector:04d}/{year}/{day}/{self.camera}-{self.ccd}/"

                try:
                    r = requests.get(url, timeout=30)
                    if r.status_code == 404:
                        continue
                    r.raise_for_status()

                    soup = BeautifulSoup(r.text, 'html.parser')

                    for link in soup.find_all('a'):
                        href = link.get('href', '')
                        if href.endswith('.fits') and 'ffic' in href:
                            files.append({
                                'year': year,
                                'day': day,
                                'filename': href,
                                'url': url + href
                            })
                except:
                    continue

        return files

    def download_and_process_file(self, file_info: Dict, temp_dir: str) -> Optional[Dict]:
        """
        Download a single file, process it, return results.
        File is deleted after processing.
        """
        import gc
        import time

        url = file_info['url']
        filename = file_info['filename']

        temp_path = Path(temp_dir) / filename
        result = None

        try:
            # Download
            r = requests.get(url, stream=True, timeout=120)
            r.raise_for_status()

            with open(temp_path, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)

            # Process
            result = self.process_fits_file(str(temp_path), file_info)

            return result

        except Exception as e:
            return {
                'filename': filename,
                'error': str(e),
                'valid': False
            }
        finally:
            # Force garbage collection to release file handles
            gc.collect()

            # Delete temp file with retry for Windows file locking
            if temp_path.exists():
                for attempt in range(3):
                    try:
                        temp_path.unlink()
                        break
                    except PermissionError:
                        if attempt < 2:
                            time.sleep(0.5)
                            gc.collect()
                        else:
                            # Last resort - ignore and let temp dir cleanup handle it
                            pass

    def process_fits_file(self, filepath: str, file_info: Dict) -> Dict:
        """Process a single FITS file and extract photometry."""
        with fits.open(filepath, mode='readonly') as hdu:
            data = hdu[1].data
            header0 = hdu[0].header
            header1 = hdu[1].header

            # Extract metadata - check both headers for SECTOR/CAMERA/CCD
            # TESS FFIs may have these in either primary or extension header
            sector = header0.get('SECTOR', header1.get('SECTOR', 0))
            camera = header0.get('CAMERA', header1.get('CAMERA', 0))
            ccd = header0.get('CCD', header1.get('CCD', 0))

            # Use self.sector/camera/ccd as fallback if headers don't have them
            if sector == 0:
                sector = self.sector
            if camera == 0:
                camera = int(self.camera)
            if ccd == 0:
                ccd = int(self.ccd)

            metadata = {
                'filename': file_info['filename'],
                'date_obs': header0.get('DATE-OBS', ''),
                'tstart': header0.get('TSTART', 0),
                'tstop': header0.get('TSTOP', 0),
                'sector': sector,
                'camera': camera,
                'ccd': ccd,
            }

            # Calculate time - use BTJD mid-time (scientifically correct for TESS)
            # BTJD = BJD - 2457000.0 (Barycentric TESS Julian Date)
            # TSTART/TSTOP are already in BTJD
            if metadata['tstart'] > 0 and metadata['tstop'] > 0:
                btjd_mid = (metadata['tstart'] + metadata['tstop']) / 2
                metadata['btjd_mid'] = btjd_mid
                metadata['bjd_mid'] = btjd_mid + 2457000.0  # Full BJD for reference
            else:
                # Fallback to UTC MJD if BTJD not available
                if metadata['date_obs']:
                    try:
                        metadata['btjd_mid'] = Time(metadata['date_obs'], format='isot', scale='utc').mjd - 2457000.0
                        metadata['bjd_mid'] = metadata['btjd_mid'] + 2457000.0
                    except:
                        metadata['btjd_mid'] = None
                        metadata['bjd_mid'] = None
                else:
                    metadata['btjd_mid'] = None
                    metadata['bjd_mid'] = None

            # TESS TSTART/TSTOP are in BTJD (days), need seconds for flux calculation
            # Use EXPOSURE header if available (already in seconds), otherwise convert
            metadata['exposure_time'] = header0.get('EXPOSURE',
                (metadata['tstop'] - metadata['tstart']) * 86400.0)  # days to seconds

            # Get gain
            gain = (header1.get('GAINA', 5.2) + header1.get('GAINB', 5.2) +
                   header1.get('GAINC', 5.2) + header1.get('GAIND', 5.2)) / 4

            # Background estimation
            mean, median, std = sigma_clipped_stats(data, sigma=3.0)

            # Get WCS - try multiple approaches for TESS data
            wcs = None
            try:
                # Try primary header first (sometimes has better WCS)
                wcs = WCS(header1)
                # Verify WCS is valid by checking reference pixel
                if wcs.wcs.crpix is None or len(wcs.wcs.crpix) < 2:
                    wcs = None
            except Exception:
                pass

            # If still no valid WCS, try with naxis parameter
            if wcs is None:
                try:
                    wcs = WCS(header1, naxis=2)
                except Exception:
                    pass

            # Use file index for epoch (thread-safe, not len(epoch_metadata) which races)
            epoch_idx = file_info.get('file_idx', len(self.epoch_metadata))

            if self.reference_stars is None:
                # First epoch - detect stars and build reference catalog
                photometry = self._detect_and_measure(
                    data, median, std, gain, metadata, wcs, epoch_idx
                )
            else:
                # Subsequent epochs - forced photometry at reference positions
                photometry = self._forced_photometry(
                    data, median, std, gain, metadata, wcs, epoch_idx
                )

            return {
                'metadata': metadata,
                'photometry': photometry,
                'valid': True,
                'n_measurements': len(photometry)
            }

    def _detect_and_measure(self, data, median, std, gain, metadata, wcs, epoch_idx):
        """Detect stars in first epoch and build reference catalog."""
        # Detect stars
        daofind = DAOStarFinder(
            fwhm=DAOFIND_FWHM,
            threshold=DAOFIND_THRESHOLD_SIGMA * std
        )
        sources = daofind(data - median)

        if sources is None or len(sources) == 0:
            return []

        # Try to import tess-point for coordinate conversion (TESS FFI WCS is unreliable)
        tess_point_available = False
        try:
            from tess_stars2px import tess_stars2px_reverse_function_entry
            tess_point_available = True
        except ImportError:
            pass

        photometry_records = []

        for i, source in enumerate(sources):
            star_id = f"STAR_{i:06d}"
            x, y = source['xcentroid'], source['ycentroid']

            # Get celestial coordinates using tess-point (preferred) or WCS fallback
            ra, dec = None, None

            # Method 1: tess-point (reliable for TESS data)
            if tess_point_available:
                try:
                    ra_val, dec_val, _ = tess_stars2px_reverse_function_entry(
                        self.sector, int(self.camera), int(self.ccd), float(x), float(y)
                    )
                    if 0 <= ra_val <= 360 and -90 <= dec_val <= 90:
                        ra, dec = float(ra_val), float(dec_val)
                except Exception:
                    pass

            # Method 2: WCS fallback (usually doesn't work for TESS FFI)
            if ra is None and wcs is not None:
                try:
                    coords = wcs.all_pix2world([[x, y]], 0)
                    ra_val, dec_val = coords[0]
                    if not np.isnan(ra_val) and not np.isnan(dec_val):
                        if 0 <= ra_val <= 360 and -90 <= dec_val <= 90:
                            ra, dec = float(ra_val), float(dec_val)
                except Exception:
                    pass

            # Perform photometry with LOCAL background estimation (annulus)
            aperture = CircularAperture([(x, y)], r=DEFAULT_APERTURE_RADIUS)
            annulus = CircularAnnulus([(x, y)], r_in=ANNULUS_R_IN, r_out=ANNULUS_R_OUT)

            # Measure in both apertures
            phot = aperture_photometry(data, aperture)
            ann_phot = aperture_photometry(data, annulus)

            # Calculate local background per pixel from annulus
            annulus_area = annulus.area_overlap(data)
            local_bkg_per_pixel = ann_phot['aperture_sum'][0] / annulus_area if annulus_area > 0 else median

            # Subtract local background from aperture
            n_pixels = np.pi * DEFAULT_APERTURE_RADIUS ** 2
            aperture_sum = phot['aperture_sum'][0] - (local_bkg_per_pixel * n_pixels)

            # Calculate flux and error (both in counts/second)
            exposure_time = metadata.get('exposure_time', 1) or 1
            flux = aperture_sum / exposure_time

            # CCD noise model with correct dimensions (using local background):
            # Error in counts, then convert to counts/second
            flux_error_counts = np.sqrt(
                abs(aperture_sum) / gain +
                n_pixels * local_bkg_per_pixel / gain +
                n_pixels * (std / gain) ** 2
            )
            flux_error = flux_error_counts / exposure_time

            # Quality flag
            quality = 0
            if aperture_sum < 0:
                quality = 2
            elif min(x, y, data.shape[1] - x, data.shape[0] - y) < DEFAULT_APERTURE_RADIUS * 2:
                quality = 1

            # Add to catalog
            self.star_catalog[star_id] = {
                'star_id': star_id,
                'ra': ra,
                'dec': dec,
                'ref_x': x,
                'ref_y': y
            }

            # Add photometry record
            record = {
                'star_id': star_id,
                'epoch': epoch_idx,
                'mjd': metadata.get('btjd_mid'),  # BTJD mid-time
                'flux': float(flux),
                'flux_error': float(flux_error),
                'quality': int(quality),
                'x': float(x),
                'y': float(y)
            }
            photometry_records.append(record)

        # Save reference
        self.reference_stars = sources
        self.reference_epoch_idx = epoch_idx

        # Reference catalog is built silently - display shows stats

        return photometry_records

    def _forced_photometry(self, data, median, std, gain, metadata, wcs, epoch_idx):
        """Perform forced photometry at reference star positions."""
        photometry_records = []

        for star_id, star_info in self.star_catalog.items():
            ra, dec = star_info['ra'], star_info['dec']

            # Convert RA/Dec to pixel coordinates for this epoch
            if wcs and ra is not None and dec is not None:
                try:
                    pixel_coords = wcs.all_world2pix([[ra, dec]], 0)
                    x, y = pixel_coords[0]
                    # Check for NaN (WCS can return NaN without exception)
                    if np.isnan(x) or np.isnan(y):
                        x, y = star_info['ref_x'], star_info['ref_y']
                except:
                    # Fall back to reference position
                    x, y = star_info['ref_x'], star_info['ref_y']
            else:
                x, y = star_info['ref_x'], star_info['ref_y']

            # Check bounds
            if x < 0 or y < 0 or x >= data.shape[1] or y >= data.shape[0]:
                record = {
                    'star_id': star_id,
                    'epoch': epoch_idx,
                    'mjd': metadata.get('btjd_mid'),  # BTJD mid-time
                    'flux': np.nan,
                    'flux_error': np.nan,
                    'quality': 3,  # Outside image
                    'x': float(x),
                    'y': float(y)
                }
                photometry_records.append(record)
                continue

            # Perform photometry with LOCAL background estimation (annulus)
            aperture = CircularAperture([(x, y)], r=DEFAULT_APERTURE_RADIUS)
            annulus = CircularAnnulus([(x, y)], r_in=ANNULUS_R_IN, r_out=ANNULUS_R_OUT)

            # Measure in both apertures
            phot = aperture_photometry(data, aperture)
            ann_phot = aperture_photometry(data, annulus)

            # Calculate local background per pixel from annulus
            annulus_area = annulus.area_overlap(data)
            local_bkg_per_pixel = ann_phot['aperture_sum'][0] / annulus_area if annulus_area > 0 else median

            # Subtract local background from aperture
            n_pixels = np.pi * DEFAULT_APERTURE_RADIUS ** 2
            aperture_sum = phot['aperture_sum'][0] - (local_bkg_per_pixel * n_pixels)

            # Calculate flux and error (both in counts/second)
            exposure_time = metadata.get('exposure_time', 1) or 1
            flux = aperture_sum / exposure_time

            # CCD noise model with correct dimensions:
            # - Signal Poisson: aperture_sum / gain (ADU → electrons variance)
            # - Background Poisson: n_pixels * local_bkg / gain (using LOCAL background)
            # - Read noise: n_pixels * (std/gain)² (not n_pixels²!)
            # Error in counts, then convert to counts/second
            flux_error_counts = np.sqrt(
                abs(aperture_sum) / gain +
                n_pixels * local_bkg_per_pixel / gain +
                n_pixels * (std / gain) ** 2
            )
            flux_error = flux_error_counts / exposure_time

            # Quality flag
            quality = 0
            if aperture_sum < 0:
                quality = 2
            elif min(x, y, data.shape[1] - x, data.shape[0] - y) < DEFAULT_APERTURE_RADIUS * 2:
                quality = 1

            record = {
                'star_id': star_id,
                'epoch': epoch_idx,
                'mjd': metadata.get('btjd_mid'),  # BTJD mid-time
                'flux': float(flux),
                'flux_error': float(flux_error),
                'quality': int(quality),
                'x': float(x),
                'y': float(y)
            }
            photometry_records.append(record)

        return photometry_records

    def run(self, resume: bool = True, workers: int = 5) -> Dict:
        """
        Run the streaming pipeline with parallel downloading.

        Args:
            resume: If True, resume from checkpoint if available
            workers: Number of parallel download workers (default 6)

        Returns:
            Summary dictionary
        """
        # Try to resume
        if resume:
            self.load_checkpoint()

        # Get file list (silently)
        print("\n  Fetching file list from MAST...", end="", flush=True)
        files = self.get_sector_files()
        print(" done!")

        if not files:
            print("  No files found!")
            return {'error': 'No files found'}

        # Add unique file index to each file (for thread-safe epoch numbering)
        for idx, f in enumerate(files):
            f['file_idx'] = idx

        # Filter already processed
        remaining_files = [f for f in files if f['filename'] not in self.processed_files]

        if not remaining_files:
            print("  All files already processed!")
            return self._create_summary()

        # Create progress display
        self.display = ProgressDisplay(
            sector=self.sector,
            camera=self.camera,
            ccd=self.ccd,
            total_files=len(files),
            already_processed=len(self.processed_files)
        )
        self.display.show_header()
        print(f"  Using {workers} parallel workers for downloading\n")
        self.display.start()

        # Lock for thread-safe updates to shared state
        self._results_lock = threading.Lock()
        self._checkpoint_counter = 0

        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                # If no reference catalog yet, process first file sequentially
                if self.reference_stars is None and remaining_files:
                    first_file = remaining_files[0]
                    remaining_files = remaining_files[1:]

                    result = self.download_and_process_file(first_file, temp_dir)
                    self._handle_result(first_file, result)

                # Process remaining files in parallel
                if remaining_files:
                    with ThreadPoolExecutor(max_workers=workers) as executor:
                        # Submit all download tasks
                        future_to_file = {
                            executor.submit(self.download_and_process_file, f, temp_dir): f
                            for f in remaining_files
                        }

                        # Process results as they complete
                        for future in as_completed(future_to_file):
                            file_info = future_to_file[future]
                            try:
                                result = future.result()
                                self._handle_result(file_info, result)
                            except Exception as e:
                                self.display.update(success=False, error=str(e))

        except KeyboardInterrupt:
            print("\n\n  Interrupted! Saving checkpoint...")
            self.save_checkpoint()
            raise

        # Close progress bar
        self.display.close()

        # Final save
        self.save_checkpoint()

        # Create final outputs
        return self._finalize()

    def _handle_result(self, file_info: Dict, result: Optional[Dict]):
        """Handle processing result (thread-safe)."""
        with self._results_lock:
            if result and result.get('valid'):
                # Add metadata
                self.epoch_metadata.append(result['metadata'])

                # Add photometry records
                self.photometry_records.extend(result.get('photometry', []))

                # Mark as processed
                self.processed_files.add(file_info['filename'])

                # Update display
                self.display.update(
                    success=True,
                    stars=len(self.star_catalog),
                    measurements=result.get('n_measurements', 0)
                )
            else:
                error_msg = result.get('error', 'Unknown error') if result else 'No result'
                self.display.update(success=False, error=error_msg)

            # Save checkpoint every 50 files (less often for parallel)
            self._checkpoint_counter += 1
            if self._checkpoint_counter % 50 == 0:
                self.save_checkpoint()

    def _process_batch_async(self, files: List[Dict], temp_dir: str,
                              download_workers: int = 10, process_workers: int = 4) -> None:
        """
        Download a batch of files asynchronously, then process them in parallel.

        This is MUCH faster than the sequential download+process approach because:
        - Async I/O overlaps multiple downloads (network-bound)
        - ThreadPool processes files in parallel (CPU-bound)

        Args:
            files: List of file info dicts to process
            temp_dir: Temporary directory for downloads
            download_workers: Concurrent downloads (default 10)
            process_workers: Parallel processing threads (default 4)
        """
        if not ASYNC_AVAILABLE:
            raise RuntimeError("AsyncDownloader not available. Install aiohttp and aiofiles.")

        import gc

        # Download all files in batch asynchronously
        results, stats = download_files_async(
            files=files,
            temp_dir=temp_dir,
            max_concurrent=download_workers
        )

        # Separate successful downloads
        downloaded = [(r.file_info, r.filepath) for r in results if r.success]
        failed = [r for r in results if not r.success]

        # Log failures
        for r in failed:
            self.display.update(success=False, error=f"Download failed: {r.error}")

        if not downloaded:
            return

        # Process downloaded files in parallel
        with ThreadPoolExecutor(max_workers=process_workers) as executor:
            futures = {}
            for file_info, filepath in downloaded:
                future = executor.submit(self._process_downloaded_file, str(filepath), file_info)
                futures[future] = (file_info, filepath)

            for future in as_completed(futures):
                file_info, filepath = futures[future]
                try:
                    result = future.result()
                    self._handle_result(file_info, result)
                except Exception as e:
                    self.display.update(success=False, error=str(e))
                finally:
                    # Clean up file
                    try:
                        if filepath.exists():
                            filepath.unlink()
                    except:
                        pass

        # Force garbage collection
        gc.collect()

    def _process_downloaded_file(self, filepath: str, file_info: Dict) -> Optional[Dict]:
        """Process an already-downloaded FITS file."""
        try:
            return self.process_fits_file(filepath, file_info)
        except Exception as e:
            return {
                'filename': file_info['filename'],
                'error': str(e),
                'valid': False
            }

    def run_async(self, resume: bool = True, download_workers: int = 10,
                  process_workers: int = 4, batch_size: int = 50,
                  max_files: int = None) -> Dict:
        """
        Run the streaming pipeline with high-performance async downloading.

        This method is 5-10x faster than run() because it uses:
        - Async I/O for parallel downloads (aiohttp)
        - Batch processing to maximize throughput
        - Separate download and process phases

        Args:
            resume: If True, resume from checkpoint
            download_workers: Number of concurrent downloads (default 10)
            process_workers: Number of parallel processing threads (default 4)
            batch_size: Files per batch (default 50)
            max_files: Maximum number of files to process (default None = all)

        Returns:
            Summary dictionary
        """
        if not ASYNC_AVAILABLE:
            print("  Warning: aiohttp not available, falling back to sync mode")
            return self.run(resume=resume, workers=process_workers)

        # Try to resume
        if resume:
            self.load_checkpoint()

        # Get file list
        print("\n  Fetching file list from MAST...", end="", flush=True)
        files = self.get_sector_files()
        print(" done!")

        if not files:
            print("  No files found!")
            return {'error': 'No files found'}

        # Filter already processed
        remaining_files = [f for f in files if f['filename'] not in self.processed_files]

        # Limit to max_files if specified
        if max_files and len(remaining_files) > max_files:
            remaining_files = remaining_files[:max_files]
            print(f"  Limiting to first {max_files} files")

        if not remaining_files:
            print("  All files already processed!")
            return self._create_summary()

        # Create progress display (show limited total if max_files specified)
        display_total = len(remaining_files) + len(self.processed_files)
        self.display = ProgressDisplay(
            sector=self.sector,
            camera=self.camera,
            ccd=self.ccd,
            total_files=display_total,
            already_processed=len(self.processed_files)
        )
        self.display.show_header()
        if max_files:
            print(f"  LIMITED MODE: Processing only {max_files} files")
        print(f"  ASYNC MODE: {download_workers} downloads, {process_workers} processors, batch={batch_size}\n")
        self.display.start()

        # Lock for thread-safe updates
        self._results_lock = threading.Lock()
        self._checkpoint_counter = 0

        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                # Process first file sequentially if no reference catalog
                if self.reference_stars is None and remaining_files:
                    first_file = remaining_files[0]
                    remaining_files = remaining_files[1:]
                    result = self.download_and_process_file(first_file, temp_dir)
                    self._handle_result(first_file, result)

                # Process remaining files in batches
                for i in range(0, len(remaining_files), batch_size):
                    batch = remaining_files[i:i + batch_size]
                    self._process_batch_async(
                        batch, temp_dir,
                        download_workers=download_workers,
                        process_workers=process_workers
                    )
                    # Checkpoint after each batch
                    self.save_checkpoint()

        except KeyboardInterrupt:
            print("\n\n  Interrupted! Saving checkpoint...")
            self.save_checkpoint()
            raise

        # Close progress bar
        self.display.close()

        # Final save
        self.save_checkpoint()

        return self._finalize()

    def add_tic_ids(self) -> pd.DataFrame:
        """
        Add TIC IDs to stars using existing FFIStarFinder function.

        Returns:
            DataFrame with star catalog including TIC IDs
        """
        from .FFIStarFinder import add_tic_ids

        # Convert catalog to DataFrame
        records = []
        for star_id, info in self.star_catalog.items():
            records.append({
                'star_id': star_id,
                'ra': info.get('ra'),
                'dec': info.get('dec'),
                'ref_x': info.get('ref_x'),
                'ref_y': info.get('ref_y'),
            })

        df = pd.DataFrame(records)

        # Filter valid coordinates
        valid = df[df['ra'].notna() & df['dec'].notna()].copy()
        print(f"Stars with valid coordinates: {len(valid)}/{len(df)}")

        # Use existing function
        valid = add_tic_ids(valid)

        # Update catalog with TIC info
        for _, row in valid.iterrows():
            star_id = row['star_id']
            tic = row.get('tic')
            if tic is not None:
                self.star_catalog[star_id]['tic_id'] = str(tic)
            else:
                self.star_catalog[star_id]['tic_id'] = star_id  # Keep internal ID

        # Save updated catalog
        self.save_checkpoint()

        # Count matches
        tic_count = valid['tic'].notna().sum()
        print(f"TIC matches: {tic_count}/{len(valid)} ({tic_count/len(valid)*100:.1f}%)")

        return valid

    def _finalize(self) -> Dict:
        """Create final output files."""
        # Save star catalog
        catalog_path = STREAMING_DIR / f"{self.session_id}_catalog.json"
        with open(catalog_path, 'w') as f:
            json.dump({
                'n_stars': len(self.star_catalog),
                'n_epochs': len(self.epoch_metadata),
                'stars': self.star_catalog
            }, f, indent=2, default=str)

        # Save photometry as CSV
        csv_path = STREAMING_DIR / f"{self.session_id}_photometry.csv"
        df = pd.DataFrame(self.photometry_records)
        df.to_csv(csv_path, index=False)

        # Try to save as parquet if pyarrow is available (more compact)
        parquet_path = STREAMING_DIR / f"{self.session_id}_photometry.parquet"
        try:
            df.to_parquet(parquet_path, index=False)
        except ImportError:
            pass  # pyarrow not installed, CSV is enough

        # Create summary
        summary = self._create_summary()
        summary_path = STREAMING_DIR / f"{self.session_id}_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2, default=str)

        # Show nice summary if display exists
        if hasattr(self, 'display') and self.display:
            data_size = csv_path.stat().st_size / 1024 / 1024
            self.display.show_summary(
                output_dir=STREAMING_DIR,
                catalog_file=catalog_path.name,
                data_file=csv_path.name,
                data_size_mb=data_size
            )
        else:
            # Fallback simple output
            print(f"\n  Results saved to: {STREAMING_DIR}")
            print(f"    Stars: {len(self.star_catalog):,}")
            print(f"    Epochs: {len(self.epoch_metadata):,}")

        return summary

    def _create_summary(self) -> Dict:
        """Create summary statistics."""
        df = pd.DataFrame(self.photometry_records) if self.photometry_records else pd.DataFrame()

        summary = {
            'session_id': self.session_id,
            'sector': self.sector,
            'camera': self.camera,
            'ccd': self.ccd,
            'n_stars': len(self.star_catalog),
            'n_epochs': len(self.epoch_metadata),
            'n_measurements': len(self.photometry_records),
            'files_processed': len(self.processed_files),
        }

        if len(df) > 0 and 'mjd' in df.columns:
            mjd_values = df['mjd'].dropna()
            if len(mjd_values) > 0:
                summary['mjd_start'] = float(mjd_values.min())
                summary['mjd_end'] = float(mjd_values.max())
                summary['time_span_days'] = summary['mjd_end'] - summary['mjd_start']

        if len(df) > 0 and 'quality' in df.columns:
            good = (df['quality'] == 0).sum()
            summary['good_measurements'] = int(good)
            summary['completeness'] = float(good / len(df)) if len(df) > 0 else 0

        return summary


def run_streaming_pipeline(sector: int, camera: str = "1", ccd: str = "1",
                          resume: bool = True, workers: int = 5,
                          add_tic: bool = False, async_mode: bool = True,
                          download_workers: int = 10, batch_size: int = 50) -> Dict:
    """
    Convenience function to run streaming pipeline.

    Args:
        sector: TESS sector number
        camera: Camera number (1-4)
        ccd: CCD number (1-4)
        resume: Resume from checkpoint if available
        workers: Number of parallel processing workers (default 5)
        add_tic: If True, add TIC IDs after processing (slow, ~1 hour for 2000 stars)
        async_mode: If True, use high-performance async downloading (5-10x faster)
        download_workers: Concurrent downloads in async mode (default 10)
        batch_size: Files per batch in async mode (default 50)

    Returns:
        Summary dictionary
    """
    processor = StreamingProcessor(sector, camera, ccd)

    if async_mode and ASYNC_AVAILABLE:
        result = processor.run_async(
            resume=resume,
            download_workers=download_workers,
            process_workers=workers,
            batch_size=batch_size
        )
    else:
        if async_mode and not ASYNC_AVAILABLE:
            print("  Note: async mode requires 'pip install aiohttp aiofiles'")
            print("  Falling back to standard mode...\n")
        result = processor.run(resume=resume, workers=workers)

    # Add TIC IDs if requested
    if add_tic and processor.star_catalog:
        print("\n" + "="*50)
        print("Adding TIC IDs (this may take a while)...")
        print("="*50)
        processor.add_tic_ids()

    return result


def get_streaming_results(sector: int, camera: str = "1", ccd: str = "1") -> Tuple[pd.DataFrame, Dict]:
    """
    Load results from a completed or in-progress streaming run.

    Returns:
        Tuple of (photometry_df, catalog_dict)
    """
    session_id = f"s{sector:04d}_{camera}-{ccd}"

    # Try multiple file locations (final, checkpoint)
    csv_path = STREAMING_DIR / f"{session_id}_photometry.csv"
    checkpoint_csv = STREAMING_DIR / f"{session_id}_photometry_checkpoint.csv"
    parquet_path = STREAMING_DIR / f"{session_id}_photometry.parquet"

    catalog_path = STREAMING_DIR / f"{session_id}_catalog.json"
    checkpoint_json = STREAMING_DIR / "checkpoints" / f"{session_id}_checkpoint.json"

    # Load photometry data
    if csv_path.exists():
        df = pd.read_csv(csv_path)
    elif checkpoint_csv.exists():
        df = pd.read_csv(checkpoint_csv)
        print(f"  Using checkpoint data: {checkpoint_csv.name}")
    elif parquet_path.exists():
        df = pd.read_parquet(parquet_path)
    else:
        raise FileNotFoundError(f"No results found for {session_id}")

    # Load catalog
    if catalog_path.exists():
        with open(catalog_path, 'r') as f:
            catalog = json.load(f)
    elif checkpoint_json.exists():
        with open(checkpoint_json, 'r') as f:
            checkpoint = json.load(f)
        # Build catalog structure from checkpoint
        catalog = {
            'n_stars': len(checkpoint.get('star_catalog', {})),
            'n_epochs': len(checkpoint.get('epoch_metadata', [])),
            'stars': checkpoint.get('star_catalog', {})
        }
        print(f"  Using checkpoint catalog: {checkpoint_json.name}")
    else:
        raise FileNotFoundError(f"No catalog found for {session_id}")

    return df, catalog


def convert_to_starcatalog(sector: int, camera: str = "1", ccd: str = "1"):
    """
    Convert streaming results to StarCatalog format for use with LightcurveBuilder.

    This allows you to use the existing pipeline steps after streaming download.
    """
    from .StarCatalog import StarCatalog

    df, catalog_data = get_streaming_results(sector, camera, ccd)

    # Create StarCatalog
    catalog = StarCatalog()

    # Populate stars
    for star_id, star_info in catalog_data['stars'].items():
        catalog.stars[star_id] = {
            'star_id': star_id,
            'ra': star_info.get('ra'),
            'dec': star_info.get('dec'),
            'ref_xcentroid': star_info.get('ref_x'),
            'ref_ycentroid': star_info.get('ref_y'),
            'detection_count': 0,
            'epochs_detected': [],
            'tic_id': star_info.get('tic_id', star_id),
        }
        catalog.photometry[star_id] = {}

    # Populate photometry (convert column names)
    for _, row in df.iterrows():
        star_id = row['star_id']
        epoch = int(row['epoch'])

        if star_id not in catalog.photometry:
            continue

        catalog.photometry[star_id][epoch] = {
            'flux': row['flux'],
            'flux_error': row['flux_error'],
            'quality_flag': int(row['quality']),  # Rename quality -> quality_flag
            'mjd': row['mjd'],
            'xcentroid': row.get('x'),
            'ycentroid': row.get('y'),
        }

        if row['quality'] == 0:
            catalog.stars[star_id]['detection_count'] += 1
            catalog.stars[star_id]['epochs_detected'].append(epoch)

    catalog.n_stars = len(catalog.stars)

    # Build epochs list (mjd column now contains BTJD mid-time)
    epochs = df.groupby('epoch')['mjd'].first().to_dict()
    catalog.epochs = [{'metadata': {'btjd_mid': mjd, 'mjd_obs': mjd}} for mjd in sorted(epochs.values())]

    print(f"Converted to StarCatalog: {catalog.n_stars} stars, {len(catalog.epochs)} epochs")

    return catalog


def build_lightcurves_from_streaming(sector: int, camera: str = "1", ccd: str = "1",
                                     min_detections: int = 3) -> pd.DataFrame:
    """
    Build lightcurve table from streaming results.

    Returns:
        DataFrame with one row per star, columns for each epoch's flux
    """
    df, catalog = get_streaming_results(sector, camera, ccd)

    # Pivot to wide format
    pivot = df.pivot_table(
        index='star_id',
        columns='epoch',
        values=['flux', 'flux_error', 'quality'],
        aggfunc='first'
    )

    # Flatten column names
    pivot.columns = [f'{col[0]}_{col[1]:04d}' for col in pivot.columns]
    pivot = pivot.reset_index()

    # Add star info
    for star_id in pivot['star_id']:
        if star_id in catalog['stars']:
            info = catalog['stars'][star_id]
            pivot.loc[pivot['star_id'] == star_id, 'ra'] = info.get('ra')
            pivot.loc[pivot['star_id'] == star_id, 'dec'] = info.get('dec')

    # Count good detections
    quality_cols = [c for c in pivot.columns if c.startswith('quality_')]
    pivot['n_detections'] = (pivot[quality_cols] == 0).sum(axis=1)

    # Filter
    pivot = pivot[pivot['n_detections'] >= min_detections]

    return pivot
