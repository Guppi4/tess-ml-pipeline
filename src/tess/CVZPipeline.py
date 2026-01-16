"""
CVZ Pipeline - Process TESS Continuous Viewing Zone data.

⚠️  DEPRECATED: This module uses TESScut which is ~100x slower than direct FFI download.
    For CVZ processing, use StreamingPipeline with Camera 4, CCD 2 instead:

    from tess.StreamingPipeline import run_streaming_pipeline

    # South CVZ is always on Camera 4, CCD 2
    for sector in [61, 62, 63, 64, 65, 66, 67, 68, 69, 87, 88, 89, 90, 93, 94, 95, 96]:
        run_streaming_pipeline(sector, "4", "2", cadence_skip=6, workers=10)

    This file is kept for reference only.

---

Original description:
Downloads cutouts from TESScut, performs photometry across multiple sectors,
and builds year-long lightcurves for stars near the ecliptic poles.

Usage (slow, not recommended):
    from tess.CVZPipeline import CVZPipeline

    cvz = CVZPipeline(
        name="south_pole",
        ra=90.0,
        dec=-66.56,
        size_px=150
    )

    cvz.download_cutouts()
    cvz.detect_stars()
    cvz.run_photometry()
    cvz.combine_sectors()

    variables = cvz.find_variables()
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from datetime import datetime
from tqdm import tqdm

from astropy.io import fits
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
from astropy.stats import sigma_clipped_stats
import astropy.units as u

from photutils.detection import DAOStarFinder
from photutils.aperture import CircularAperture, CircularAnnulus, aperture_photometry

from .config import (
    DATA_DIR,
    DAOFIND_FWHM,
    DAOFIND_THRESHOLD_SIGMA,
    DEFAULT_APERTURE_RADIUS,
    ANNULUS_R_IN,
    ANNULUS_R_OUT,
)


# CVZ data directory
CVZ_DIR = DATA_DIR / "cvz"


class CVZPipeline:
    """
    Pipeline for processing TESS Continuous Viewing Zone data.

    Uses TESScut to download small cutouts around a target position,
    then performs photometry across all available sectors.
    """

    # Known ecliptic pole coordinates
    SOUTH_POLE = (90.0, -66.56)  # RA, Dec in degrees
    NORTH_POLE = (270.0, 66.56)  # RA, Dec in degrees

    def __init__(
        self,
        name: str,
        ra: float = None,
        dec: float = None,
        size_px: int = 150,
        pole: str = None
    ):
        """
        Initialize CVZ pipeline.

        Args:
            name: Project name (used for directory)
            ra: Right Ascension in degrees
            dec: Declination in degrees
            size_px: Cutout size in pixels (default 150)
            pole: 'south' or 'north' (sets ra/dec automatically)
        """
        self.name = name
        self.size_px = size_px

        # Set coordinates
        if pole == 'south':
            self.ra, self.dec = self.SOUTH_POLE
        elif pole == 'north':
            self.ra, self.dec = self.NORTH_POLE
        elif ra is not None and dec is not None:
            self.ra, self.dec = ra, dec
        else:
            raise ValueError("Specify ra/dec or pole='south'/'north'")

        # Directories
        self.base_dir = CVZ_DIR / name
        self.cutouts_dir = self.base_dir / "cutouts"
        self.photometry_dir = self.base_dir / "photometry"
        self.combined_dir = self.base_dir / "combined"

        # Create directories
        for d in [self.cutouts_dir, self.photometry_dir, self.combined_dir]:
            d.mkdir(parents=True, exist_ok=True)

        # Paths
        self.config_path = self.base_dir / "config.json"
        self.catalog_path = self.base_dir / "star_catalog.csv"
        self.combined_path = self.combined_dir / "lightcurves.csv"

        # State
        self.sectors = []
        self.star_catalog = None
        self.reference_sector = None

        # Load existing config if resuming, otherwise save new
        if not self._load_config():
            self._save_config()

        # Load star catalog if exists
        if self.catalog_path.exists():
            self.star_catalog = pd.read_csv(self.catalog_path)

        print(f"CVZ Pipeline: {name}")
        print(f"  Coordinates: RA={self.ra:.2f}, Dec={self.dec:.2f}")
        print(f"  Cutout size: {size_px}x{size_px} px")
        print(f"  Directory: {self.base_dir}")

    def _save_config(self):
        """Save configuration to JSON."""
        config = {
            'name': self.name,
            'ra': self.ra,
            'dec': self.dec,
            'size_px': self.size_px,
            'created': datetime.now().isoformat(),
            'sectors': self.sectors,
            'reference_sector': self.reference_sector,
        }
        with open(self.config_path, 'w') as f:
            json.dump(config, f, indent=2)

    def _load_config(self):
        """Load configuration from JSON."""
        if self.config_path.exists():
            with open(self.config_path, 'r') as f:
                config = json.load(f)
            self.sectors = config.get('sectors', [])
            self.reference_sector = config.get('reference_sector')
            return True
        return False

    def get_available_sectors(self) -> List[int]:
        """Query TESScut for available sectors at this position."""
        from astroquery.mast import Tesscut

        coord = SkyCoord(ra=self.ra, dec=self.dec, unit='deg')
        sector_table = Tesscut.get_sectors(coordinates=coord)

        sectors = list(sector_table['sector'])
        print(f"Available sectors: {len(sectors)}")
        print(f"  {sectors}")

        return sectors

    def download_cutouts(self, sectors: List[int] = None, force: bool = False, delay: float = 3.0):
        """
        Download cutouts from TESScut for all available sectors.

        Args:
            sectors: List of sectors to download (None = all available)
            force: Re-download even if file exists
            delay: Delay between downloads in seconds (to avoid rate limiting)
        """
        from astroquery.mast import Tesscut
        import time

        if sectors is None:
            sectors = self.get_available_sectors()

        coord = SkyCoord(ra=self.ra, dec=self.dec, unit='deg')

        print(f"\nDownloading {len(sectors)} sectors...")
        print(f"  (with {delay}s delay between requests to avoid rate limiting)")

        downloaded = []
        for i, sector in enumerate(tqdm(sectors, desc="Downloading")):
            # Delay between requests (except first)
            if i > 0:
                time.sleep(delay)
            # Check if already exists
            fits_path = self.cutouts_dir / f"sector_{sector:03d}.fits"

            if fits_path.exists() and not force:
                downloaded.append(sector)
                continue

            try:
                # Remove existing file if force=True (for Windows compatibility)
                if fits_path.exists() and force:
                    fits_path.unlink()

                # Download cutout
                result = Tesscut.download_cutouts(
                    coordinates=coord,
                    size=self.size_px,
                    sector=sector,
                    path=str(self.cutouts_dir)
                )

                # Rename to standard format
                if result and len(result) > 0:
                    # Find downloaded file
                    import glob
                    pattern = str(self.cutouts_dir / f"*s{sector:04d}*.fits")
                    files = glob.glob(pattern)
                    if files:
                        src_path = Path(files[0])
                        # Use replace() instead of rename() for Windows compatibility
                        src_path.replace(fits_path)
                        downloaded.append(sector)

            except Exception as e:
                print(f"\n  Warning: Sector {sector} failed: {e}")

        self.sectors = sorted(downloaded)
        self._save_config()

        print(f"\nDownloaded: {len(downloaded)} sectors")
        print(f"  Files in: {self.cutouts_dir}")

    def detect_stars(
        self,
        reference_sector: int = None,
        fwhm: float = DAOFIND_FWHM,
        threshold_sigma: float = DAOFIND_THRESHOLD_SIGMA,
        edge_margin: int = 10
    ) -> pd.DataFrame:
        """
        Detect stars in reference sector and build catalog.

        Args:
            reference_sector: Sector to use as reference (None = first available)
            fwhm: FWHM for DAOStarFinder
            threshold_sigma: Detection threshold in sigma
            edge_margin: Exclude stars within this many pixels of edge

        Returns:
            DataFrame with star catalog
        """
        # Find reference sector
        if reference_sector is None:
            if not self.sectors:
                # Find downloaded files
                fits_files = list(self.cutouts_dir.glob("sector_*.fits"))
                self.sectors = sorted([
                    int(f.stem.split('_')[1]) for f in fits_files
                ])

            if not self.sectors:
                raise ValueError("No sectors downloaded. Run download_cutouts() first.")

            reference_sector = self.sectors[0]

        self.reference_sector = reference_sector
        fits_path = self.cutouts_dir / f"sector_{reference_sector:03d}.fits"

        print(f"\nDetecting stars in sector {reference_sector}...")

        with fits.open(fits_path) as hdu:
            pixels = hdu[1].data
            flux_cube = pixels['FLUX']
            quality = pixels['QUALITY']

            # Get WCS with fallback (TESScut usually has WCS in HDU[2], but try others)
            wcs = None
            for hdu_idx in [2, 1, 0]:
                try:
                    test_wcs = WCS(hdu[hdu_idx].header)
                    if test_wcs.has_celestial:
                        wcs = test_wcs
                        break
                except Exception:
                    continue

            if wcs is None:
                raise ValueError(f"No valid WCS found in {fits_path}")

            # Create median frame from good data
            good_mask = quality == 0
            n_good = good_mask.sum()
            print(f"  Good frames: {n_good}/{len(quality)}")

            if n_good == 0:
                raise ValueError(f"No good frames (quality==0) in sector {reference_sector}")

            # Use first 100 good frames for median
            good_indices = np.where(good_mask)[0][:100]
            median_frame = np.nanmedian(flux_cube[good_indices], axis=0)

            if np.all(np.isnan(median_frame)):
                raise ValueError(f"Median frame is all NaN in sector {reference_sector}")

            # Background stats
            mean, median_val, std = sigma_clipped_stats(median_frame, sigma=3.0)
            print(f"  Background: median={median_val:.1f}, std={std:.1f}")

            # Detect stars
            daofind = DAOStarFinder(fwhm=fwhm, threshold=threshold_sigma * std)
            sources = daofind(median_frame - median_val)

            if sources is None or len(sources) == 0:
                print("  No stars detected!")
                return pd.DataFrame()

            print(f"  Detected: {len(sources)} sources")

            # Build catalog with coordinates
            catalog_rows = []
            img_size = flux_cube.shape[1]  # Assuming square

            for i, source in enumerate(sources):
                x = float(source['xcentroid'])
                y = float(source['ycentroid'])

                # Skip stars too close to edge
                if (x < edge_margin or x > img_size - edge_margin or
                    y < edge_margin or y > img_size - edge_margin):
                    continue

                # Get celestial coordinates
                sky = wcs.pixel_to_world(x, y)
                ra = sky.ra.deg
                dec = sky.dec.deg

                catalog_rows.append({
                    'star_id': f"STAR_{i:06d}",
                    'ra': ra,
                    'dec': dec,
                    'ref_x': x,
                    'ref_y': y,
                    'ref_flux': float(source['flux']),
                    'ref_peak': float(source['peak']),
                })

            self.star_catalog = pd.DataFrame(catalog_rows)

            # Save catalog
            self.star_catalog.to_csv(self.catalog_path, index=False)
            self._save_config()

            print(f"  Catalog: {len(self.star_catalog)} stars (after edge filter)")
            print(f"  Saved: {self.catalog_path}")

            return self.star_catalog

    def run_photometry(
        self,
        sectors: List[int] = None,
        aperture_radius: float = DEFAULT_APERTURE_RADIUS,
        annulus_r_in: float = ANNULUS_R_IN,
        annulus_r_out: float = ANNULUS_R_OUT
    ):
        """
        Run aperture photometry for all sectors.

        Args:
            sectors: List of sectors to process (None = all downloaded)
            aperture_radius: Aperture radius in pixels
            annulus_r_in: Inner annulus radius
            annulus_r_out: Outer annulus radius
        """
        # Load catalog if needed
        if self.star_catalog is None:
            if self.catalog_path.exists():
                self.star_catalog = pd.read_csv(self.catalog_path)
            else:
                raise ValueError("No star catalog. Run detect_stars() first.")

        # Get sectors
        if sectors is None:
            fits_files = list(self.cutouts_dir.glob("sector_*.fits"))
            sectors = sorted([int(f.stem.split('_')[1]) for f in fits_files])

        print(f"\nRunning photometry for {len(sectors)} sectors...")
        print(f"  Stars: {len(self.star_catalog)}")
        print(f"  Aperture: r={aperture_radius} px")

        for sector in tqdm(sectors, desc="Photometry"):
            self._photometry_sector(
                sector, aperture_radius, annulus_r_in, annulus_r_out
            )

        print(f"\nPhotometry complete!")
        print(f"  Results in: {self.photometry_dir}")

    def _photometry_sector(
        self,
        sector: int,
        aperture_radius: float,
        annulus_r_in: float,
        annulus_r_out: float
    ):
        """Run photometry for a single sector (fully vectorized)."""
        fits_path = self.cutouts_dir / f"sector_{sector:03d}.fits"
        output_path = self.photometry_dir / f"sector_{sector:03d}.csv"

        if not fits_path.exists():
            return

        with fits.open(fits_path) as hdu:
            pixels = hdu[1].data
            flux_cube = pixels['FLUX']
            time_arr = pixels['TIME']
            quality_arr = pixels['QUALITY']

            # Get WCS with fallback
            wcs = None
            for hdu_idx in [2, 1, 0]:
                try:
                    test_wcs = WCS(hdu[hdu_idx].header)
                    if test_wcs.has_celestial:
                        wcs = test_wcs
                        break
                except Exception:
                    continue

            if wcs is None:
                print(f"  Warning: No WCS in sector {sector}, skipping")
                return

            # Background for error estimation
            _, median_bkg, std_bkg = sigma_clipped_stats(
                np.nanmedian(flux_cube[:10], axis=0), sigma=3.0
            )

            # Get frame dimensions (support non-square)
            height, width = flux_cube.shape[1], flux_cube.shape[2]

            # Pre-compute pixel positions for all stars (with NaN filtering)
            star_ids = []
            positions = []
            for _, star in self.star_catalog.iterrows():
                sky = SkyCoord(ra=star['ra'], dec=star['dec'], unit='deg')
                px, py = wcs.world_to_pixel(sky)
                px, py = float(px), float(py)

                # Skip stars with NaN coordinates or outside image
                if np.isnan(px) or np.isnan(py):
                    continue
                if px < 0 or px >= width or py < 0 or py >= height:
                    continue

                star_ids.append(star['star_id'])
                positions.append((px, py))

            if not positions:
                return

            n_stars = len(positions)
            positions = np.array(positions)
            star_ids = np.array(star_ids)

            # Create apertures ONCE for all stars
            apertures = CircularAperture(positions, r=aperture_radius)
            annuli = CircularAnnulus(positions, r_in=annulus_r_in, r_out=annulus_r_out)

            # Pre-compute areas ONCE (same for all frames with same shape)
            # Guard against empty flux_cube
            if len(flux_cube) == 0:
                return
            reference_frame = flux_cube[0]
            ap_areas = np.array(apertures.area_overlap(reference_frame))
            ann_areas = np.array(annuli.area_overlap(reference_frame))

            # Pre-compute edge flags
            edge_flags = np.zeros(n_stars, dtype=int)
            for i, (x, y) in enumerate(positions):
                if min(x, width - x) < aperture_radius * 2 or min(y, height - y) < aperture_radius * 2:
                    edge_flags[i] = 1  # Near edge

            # Process all epochs - collect as list of DataFrames
            all_epoch_dfs = []
            good_epochs = np.where(quality_arr == 0)[0]

            for epoch_idx in good_epochs:
                frame = flux_cube[epoch_idx]
                btjd = float(time_arr[epoch_idx])

                # Skip frames with all NaN
                if np.all(np.isnan(frame)):
                    continue

                # Single photometry call for ALL stars at once
                # Note: NOT using NaN mask to avoid area mismatch bias
                # NaN pixels will propagate to results, filtered by quality flag
                phot = aperture_photometry(frame, apertures)
                ann_phot = aperture_photometry(frame, annuli)

                # Aperture sums
                aperture_sums = np.array(phot['aperture_sum'])
                annulus_sums = np.array(ann_phot['aperture_sum'])

                # Local background per pixel (vectorized)
                local_bkg = np.where(ann_areas > 0, annulus_sums / ann_areas, median_bkg)

                # Net flux
                net_flux = aperture_sums - local_bkg * ap_areas

                # Error estimate
                flux_err = np.sqrt(np.abs(net_flux) + ap_areas * local_bkg + ap_areas * std_bkg**2)

                # Quality flags (0=good, 1=edge, 2=negative, 3=out of bounds, 4=NaN)
                quality = edge_flags.copy()
                quality[net_flux < 0] = 2  # Negative flux
                quality[~np.isfinite(net_flux)] = 4  # NaN flux (not 3, reserved for out-of-bounds)

                # Vectorized DataFrame creation (no Python loop!)
                epoch_df = pd.DataFrame({
                    'star_id': star_ids,
                    'btjd': btjd,
                    'flux': net_flux,
                    'flux_err': flux_err,
                    'quality': quality,
                    'x': positions[:, 0],
                    'y': positions[:, 1],
                })
                all_epoch_dfs.append(epoch_df)

        # Concatenate all epochs at once
        if all_epoch_dfs:
            df = pd.concat(all_epoch_dfs, ignore_index=True)
            df.to_csv(output_path, index=False)

    def combine_sectors(self) -> pd.DataFrame:
        """
        Combine photometry from all sectors into single lightcurve file.

        Normalizes flux between sectors using sigma-clipped median.

        Returns:
            DataFrame with combined lightcurves
        """
        print("\nCombining sectors...")

        # Find all photometry files
        phot_files = sorted(self.photometry_dir.glob("sector_*.csv"))

        if not phot_files:
            raise ValueError("No photometry files found. Run run_photometry() first.")

        print(f"  Found {len(phot_files)} sector files")

        all_data = []

        for phot_file in tqdm(phot_files, desc="Loading"):
            sector = int(phot_file.stem.split('_')[1])

            df = pd.read_csv(phot_file)

            # Fix flux columns if they were saved as string arrays (bug fix)
            for col in ['flux', 'flux_err']:
                if df[col].dtype == object:
                    df[col] = df[col].apply(lambda x: float(str(x).strip('[]')) if pd.notna(x) else np.nan)

            # Normalize flux per star within this sector
            for star_id in df['star_id'].unique():
                star_mask = df['star_id'] == star_id
                good_mask = star_mask & (df['quality'] == 0)

                if good_mask.sum() < 10:
                    continue

                fluxes = df.loc[good_mask, 'flux'].values

                # Sigma-clipped median for normalization
                clipped_median = np.median(fluxes[
                    (fluxes > np.percentile(fluxes, 5)) &
                    (fluxes < np.percentile(fluxes, 95))
                ])

                if clipped_median <= 0:
                    continue

                # Normalize
                star_data = df[star_mask].copy()
                star_data['flux_norm'] = star_data['flux'] / clipped_median
                star_data['flux_err_norm'] = star_data['flux_err'] / clipped_median
                star_data['sector'] = sector

                all_data.append(star_data[['star_id', 'btjd', 'flux_norm', 'flux_err_norm', 'quality', 'sector']])

        # Check if any data was collected
        if not all_data:
            print("  Warning: No valid data to combine (all stars had <10 good points)")
            return pd.DataFrame()

        # Combine
        combined = pd.concat(all_data, ignore_index=True)
        combined = combined.sort_values(['star_id', 'btjd']).reset_index(drop=True)

        # Rename for compatibility with LightcurveBuilder
        combined = combined.rename(columns={
            'flux_norm': 'flux',
            'flux_err_norm': 'flux_error',
            'quality': 'quality_flag'
        })

        # Save
        combined.to_csv(self.combined_path, index=False)

        # Statistics
        n_stars = combined['star_id'].nunique()
        n_points = len(combined)
        time_span = combined['btjd'].max() - combined['btjd'].min()

        print(f"\nCombined lightcurves:")
        print(f"  Stars: {n_stars}")
        print(f"  Total points: {n_points:,}")
        print(f"  Time span: {time_span:.1f} days")
        print(f"  Saved: {self.combined_path}")

        return combined

    def load_lightcurves(self) -> pd.DataFrame:
        """Load combined lightcurves from file."""
        if not self.combined_path.exists():
            raise ValueError("No combined lightcurves. Run combine_sectors() first.")
        return pd.read_csv(self.combined_path)

    def get_lightcurve(self, star_id: str) -> pd.DataFrame:
        """Get lightcurve for a single star."""
        combined = self.load_lightcurves()
        return combined[combined['star_id'] == star_id].copy()

    def get_lightcurve_collection(self):
        """
        Build LightcurveCollection from combined data.

        Returns:
            LightcurveCollection object compatible with existing analysis tools
        """
        from .LightcurveBuilder import Lightcurve, LightcurveCollection

        combined = self.load_lightcurves()
        catalog = pd.read_csv(self.catalog_path)

        collection = LightcurveCollection()

        for star_id in combined['star_id'].unique():
            star_data = combined[combined['star_id'] == star_id].copy()

            # Get star info from catalog
            star_info = catalog[catalog['star_id'] == star_id]
            if len(star_info) > 0:
                info_dict = star_info.iloc[0].to_dict()
            else:
                info_dict = {}

            # Create Lightcurve object
            lc = Lightcurve(star_id, star_data, info_dict)
            collection.add(lc)

        return collection

    def find_variables(
        self,
        min_amplitude: float = 0.01,
        min_chi2: float = 2.0,
        min_points: int = 100
    ) -> pd.DataFrame:
        """
        Find variable star candidates in combined data.

        Args:
            min_amplitude: Minimum amplitude (robust, p95-p5)
            min_chi2: Minimum reduced chi-squared
            min_points: Minimum number of good data points

        Returns:
            DataFrame with variable candidates ranked by variability
        """
        print("\nSearching for variable stars...")

        combined = self.load_lightcurves()
        catalog = pd.read_csv(self.catalog_path)

        candidates = []

        for star_id in tqdm(combined['star_id'].unique(), desc="Analyzing"):
            star_data = combined[combined['star_id'] == star_id]
            good_data = star_data[star_data['quality_flag'] == 0]

            if len(good_data) < min_points:
                continue

            flux = good_data['flux'].values
            flux_err = good_data['flux_error'].values

            # Filter out NaN/inf and zero errors
            valid_mask = np.isfinite(flux) & np.isfinite(flux_err) & (flux_err > 0)
            flux = flux[valid_mask]
            flux_err = flux_err[valid_mask]

            if len(flux) < min_points:
                continue

            # Calculate statistics
            median_flux = np.median(flux)
            std_flux = np.std(flux)

            if median_flux <= 0 or not np.isfinite(median_flux):
                continue

            # Robust amplitude
            p5, p95 = np.percentile(flux, [5, 95])
            amplitude = (p95 - p5) / median_flux

            # Reduced chi-squared (safe division)
            chi2 = np.sum(((flux - median_flux) / flux_err) ** 2) / max(len(flux) - 1, 1)

            # MAD
            mad = np.median(np.abs(flux - median_flux))

            # Skip if any result is invalid
            if not np.isfinite(amplitude) or not np.isfinite(chi2):
                continue

            if amplitude >= min_amplitude and chi2 >= min_chi2:
                # Get coordinates
                star_info = catalog[catalog['star_id'] == star_id]
                ra = star_info['ra'].values[0] if len(star_info) > 0 else None
                dec = star_info['dec'].values[0] if len(star_info) > 0 else None

                candidates.append({
                    'star_id': star_id,
                    'ra': ra,
                    'dec': dec,
                    'n_points': len(good_data),
                    'amplitude': amplitude,
                    'reduced_chi2': chi2,
                    'mad': mad,
                    'std': std_flux,
                    'median_flux': median_flux,
                })

        if not candidates:
            print("  No variable candidates found with current criteria.")
            return pd.DataFrame()

        result = pd.DataFrame(candidates)
        result = result.sort_values('amplitude', ascending=False).reset_index(drop=True)

        print(f"\nFound {len(result)} variable candidates!")
        print(result[['star_id', 'amplitude', 'reduced_chi2', 'n_points']].head(20))

        return result

    # =========================================================================
    # GRID MODE - for downloading multiple cutouts to cover larger area
    # =========================================================================

    def calculate_grid_positions(self, grid_size: int = 10, overlap_px: int = 5) -> List[Dict]:
        """
        Calculate grid positions for multiple cutouts.

        Args:
            grid_size: Number of cutouts per side (10 = 10x10 = 100 cutouts)
            overlap_px: Overlap between adjacent cutouts in pixels

        Returns:
            List of dicts with grid_i, grid_j, ra, dec for each position
        """
        # TESS pixel scale: ~21 arcsec/pixel
        pixel_scale = 21.0 / 3600  # degrees per pixel
        step_deg = (self.size_px - overlap_px) * pixel_scale

        # Generate grid centered on target
        half_grid = (grid_size - 1) / 2
        positions = []

        for i in range(grid_size):
            for j in range(grid_size):
                # Offset from center in grid units
                di = i - half_grid
                dj = j - half_grid

                # Convert to sky coordinates
                # Dec is straightforward
                dec = self.dec + dj * step_deg
                # RA needs cos(dec) correction for spherical geometry
                ra = self.ra + di * step_deg / np.cos(np.radians(dec))

                positions.append({
                    'grid_i': i,
                    'grid_j': j,
                    'ra': ra,
                    'dec': dec,
                    'pos_id': f"pos_{i:02d}_{j:02d}",
                })

        return positions

    def download_grid(
        self,
        grid_size: int = 10,
        sectors: List[int] = None,
        overlap_px: int = 5,
        delay: float = 3.0,
        force: bool = False
    ):
        """
        Download a grid of cutouts to cover ~1000 stars.

        Args:
            grid_size: Number of cutouts per side (10 = 10x10 grid)
            sectors: List of sectors to download (None = all available)
            overlap_px: Overlap between adjacent cutouts
            delay: Delay between downloads (rate limiting)
            force: Re-download even if exists
        """
        from astroquery.mast import Tesscut
        import time

        # Get available sectors
        if sectors is None:
            sectors = self.get_available_sectors()

        # Calculate grid positions
        positions = self.calculate_grid_positions(grid_size, overlap_px)

        print(f"\nGrid download: {grid_size}x{grid_size} = {len(positions)} positions")
        print(f"Sectors: {len(sectors)}")
        print(f"Total cutouts: {len(positions) * len(sectors)}")
        print(f"Estimated time: {len(positions) * len(sectors) * delay / 60:.1f} minutes")

        # Create grid cutouts directory
        grid_cutouts_dir = self.base_dir / "grid_cutouts"
        grid_cutouts_dir.mkdir(parents=True, exist_ok=True)

        # Save grid config
        grid_config = {
            'grid_size': grid_size,
            'overlap_px': overlap_px,
            'positions': positions,
            'sectors': sectors,
        }
        with open(self.base_dir / "grid_config.json", 'w') as f:
            json.dump(grid_config, f, indent=2)

        # Download each position for each sector
        total = len(positions) * len(sectors)
        downloaded = 0
        skipped = 0

        for pos in tqdm(positions, desc="Grid positions"):
            pos_id = pos['pos_id']
            pos_dir = grid_cutouts_dir / pos_id
            pos_dir.mkdir(exist_ok=True)

            coord = SkyCoord(ra=pos['ra'], dec=pos['dec'], unit='deg')

            for sector in sectors:
                fits_path = pos_dir / f"sector_{sector:03d}.fits"

                if fits_path.exists() and not force:
                    skipped += 1
                    continue

                # Rate limiting
                time.sleep(delay)

                try:
                    if fits_path.exists() and force:
                        fits_path.unlink()

                    result = Tesscut.download_cutouts(
                        coordinates=coord,
                        size=self.size_px,
                        sector=sector,
                        path=str(pos_dir)
                    )

                    if result and len(result) > 0:
                        # Rename to standard format
                        import glob
                        pattern = str(pos_dir / f"*s{sector:04d}*.fits")
                        files = glob.glob(pattern)
                        if files:
                            src_path = Path(files[0])
                            src_path.replace(fits_path)
                            downloaded += 1

                except Exception as e:
                    print(f"\n  Warning: {pos_id} sector {sector} failed: {e}")

        print(f"\nGrid download complete!")
        print(f"  Downloaded: {downloaded}")
        print(f"  Skipped (existing): {skipped}")
        print(f"  Directory: {grid_cutouts_dir}")

    def detect_stars_grid(
        self,
        reference_sector: int = None,
        fwhm: float = DAOFIND_FWHM,
        threshold_sigma: float = DAOFIND_THRESHOLD_SIGMA,
        edge_margin: int = 10
    ) -> pd.DataFrame:
        """
        Detect stars from all grid cutouts.

        Returns combined star catalog with unique IDs per position.
        """
        # Load grid config
        grid_config_path = self.base_dir / "grid_config.json"
        if not grid_config_path.exists():
            raise ValueError("No grid config. Run download_grid() first.")

        with open(grid_config_path) as f:
            grid_config = json.load(f)

        positions = grid_config['positions']
        sectors = grid_config['sectors']

        if reference_sector is None:
            reference_sector = sectors[0]

        self.reference_sector = reference_sector
        grid_cutouts_dir = self.base_dir / "grid_cutouts"

        print(f"\nDetecting stars in {len(positions)} grid positions...")
        print(f"  Reference sector: {reference_sector}")

        all_stars = []
        total_detected = 0

        for pos in tqdm(positions, desc="Detecting"):
            pos_id = pos['pos_id']
            fits_path = grid_cutouts_dir / pos_id / f"sector_{reference_sector:03d}.fits"

            if not fits_path.exists():
                continue

            try:
                with fits.open(fits_path) as hdu:
                    pixels = hdu[1].data
                    flux_cube = pixels['FLUX']
                    quality = pixels['QUALITY']

                    # Get WCS
                    wcs = None
                    for hdu_idx in [2, 1, 0]:
                        try:
                            test_wcs = WCS(hdu[hdu_idx].header)
                            if test_wcs.has_celestial:
                                wcs = test_wcs
                                break
                        except:
                            continue

                    if wcs is None:
                        continue

                    # Median frame from good data
                    good_mask = quality == 0
                    if good_mask.sum() == 0:
                        continue

                    good_indices = np.where(good_mask)[0][:100]
                    median_frame = np.nanmedian(flux_cube[good_indices], axis=0)

                    if np.all(np.isnan(median_frame)):
                        continue

                    # Background stats
                    mean, median_val, std = sigma_clipped_stats(median_frame, sigma=3.0)

                    # Detect stars
                    daofind = DAOStarFinder(fwhm=fwhm, threshold=threshold_sigma * std)
                    sources = daofind(median_frame - median_val)

                    if sources is None or len(sources) == 0:
                        continue

                    img_size = flux_cube.shape[1]

                    for i, source in enumerate(sources):
                        x = float(source['xcentroid'])
                        y = float(source['ycentroid'])

                        # Skip edge stars
                        if (x < edge_margin or x > img_size - edge_margin or
                            y < edge_margin or y > img_size - edge_margin):
                            continue

                        # Get celestial coordinates
                        sky = wcs.pixel_to_world(x, y)
                        ra = sky.ra.deg
                        dec = sky.dec.deg

                        # Unique star ID including position
                        star_id = f"{pos_id}_STAR_{i:04d}"

                        all_stars.append({
                            'star_id': star_id,
                            'pos_id': pos_id,
                            'ra': ra,
                            'dec': dec,
                            'ref_x': x,
                            'ref_y': y,
                            'ref_flux': float(source['flux']),
                            'ref_peak': float(source['peak']),
                        })

                    total_detected += len(sources)

            except Exception as e:
                print(f"\n  Warning: {pos_id} detection failed: {e}")

        if not all_stars:
            print("  No stars detected!")
            return pd.DataFrame()

        self.star_catalog = pd.DataFrame(all_stars)
        self.star_catalog.to_csv(self.catalog_path, index=False)
        self._save_config()

        print(f"\nStar detection complete!")
        print(f"  Total detected: {total_detected}")
        print(f"  After edge filter: {len(self.star_catalog)}")
        print(f"  Saved: {self.catalog_path}")

        return self.star_catalog

    def run_photometry_grid(
        self,
        sectors: List[int] = None,
        aperture_radius: float = DEFAULT_APERTURE_RADIUS,
        annulus_r_in: float = ANNULUS_R_IN,
        annulus_r_out: float = ANNULUS_R_OUT
    ):
        """
        Run photometry for all grid positions.
        """
        # Load grid config
        grid_config_path = self.base_dir / "grid_config.json"
        if not grid_config_path.exists():
            raise ValueError("No grid config. Run download_grid() first.")

        with open(grid_config_path) as f:
            grid_config = json.load(f)

        positions = grid_config['positions']

        if sectors is None:
            sectors = grid_config['sectors']

        if self.star_catalog is None:
            if self.catalog_path.exists():
                self.star_catalog = pd.read_csv(self.catalog_path)
            else:
                raise ValueError("No star catalog. Run detect_stars_grid() first.")

        grid_cutouts_dir = self.base_dir / "grid_cutouts"

        print(f"\nRunning photometry for {len(sectors)} sectors...")
        print(f"  Grid positions: {len(positions)}")
        print(f"  Stars: {len(self.star_catalog)}")

        for sector in tqdm(sectors, desc="Sectors"):
            sector_dfs = []

            for pos in positions:
                pos_id = pos['pos_id']
                fits_path = grid_cutouts_dir / pos_id / f"sector_{sector:03d}.fits"

                if not fits_path.exists():
                    continue

                # Get stars for this position
                pos_stars = self.star_catalog[self.star_catalog['pos_id'] == pos_id]
                if len(pos_stars) == 0:
                    continue

                # Run photometry for this position (returns DataFrame)
                pos_df = self._photometry_grid_position(
                    fits_path, pos_stars, aperture_radius, annulus_r_in, annulus_r_out
                )
                if len(pos_df) > 0:
                    sector_dfs.append(pos_df)

            # Save sector results
            if sector_dfs:
                df = pd.concat(sector_dfs, ignore_index=True)
                output_path = self.photometry_dir / f"sector_{sector:03d}.csv"
                df.to_csv(output_path, index=False)

        print(f"\nPhotometry complete!")
        print(f"  Results in: {self.photometry_dir}")

    def _photometry_grid_position(
        self,
        fits_path: Path,
        stars: pd.DataFrame,
        aperture_radius: float,
        annulus_r_in: float,
        annulus_r_out: float
    ) -> pd.DataFrame:
        """Run fully vectorized photometry for one grid position."""

        with fits.open(fits_path) as hdu:
            pixels = hdu[1].data
            flux_cube = pixels['FLUX']
            time_arr = pixels['TIME']
            quality_arr = pixels['QUALITY']

            # Get WCS
            wcs = None
            for hdu_idx in [2, 1, 0]:
                try:
                    test_wcs = WCS(hdu[hdu_idx].header)
                    if test_wcs.has_celestial:
                        wcs = test_wcs
                        break
                except:
                    continue

            if wcs is None:
                return pd.DataFrame()

            # Background stats
            _, median_bkg, std_bkg = sigma_clipped_stats(
                np.nanmedian(flux_cube[:10], axis=0), sigma=3.0
            )

            # Get frame dimensions (support non-square)
            height, width = flux_cube.shape[1], flux_cube.shape[2]

            # Get pixel positions for stars
            star_ids = []
            positions = []

            for _, star in stars.iterrows():
                sky = SkyCoord(ra=star['ra'], dec=star['dec'], unit='deg')
                px, py = wcs.world_to_pixel(sky)
                px, py = float(px), float(py)

                if np.isnan(px) or np.isnan(py):
                    continue
                if px < 0 or px >= width or py < 0 or py >= height:
                    continue

                star_ids.append(star['star_id'])
                positions.append((px, py))

            if not positions:
                return pd.DataFrame()

            n_stars = len(positions)
            positions = np.array(positions)
            star_ids = np.array(star_ids)

            # Create apertures once
            apertures = CircularAperture(positions, r=aperture_radius)
            annuli = CircularAnnulus(positions, r_in=annulus_r_in, r_out=annulus_r_out)

            # Pre-compute areas ONCE (same for all frames)
            # Guard against empty flux_cube
            if len(flux_cube) == 0:
                return pd.DataFrame()
            reference_frame = flux_cube[0]
            ap_areas = np.array(apertures.area_overlap(reference_frame))
            ann_areas = np.array(annuli.area_overlap(reference_frame))

            # Edge flags
            edge_flags = np.zeros(n_stars, dtype=int)
            for i, (x, y) in enumerate(positions):
                if min(x, width - x) < aperture_radius * 2 or min(y, height - y) < aperture_radius * 2:
                    edge_flags[i] = 1

            # Process good epochs - collect as DataFrames
            all_epoch_dfs = []
            good_epochs = np.where(quality_arr == 0)[0]

            for epoch_idx in good_epochs:
                frame = flux_cube[epoch_idx]
                btjd = float(time_arr[epoch_idx])

                if np.all(np.isnan(frame)):
                    continue

                # Vectorized photometry (no NaN mask to avoid area mismatch)
                phot = aperture_photometry(frame, apertures)
                ann_phot = aperture_photometry(frame, annuli)

                aperture_sums = np.array(phot['aperture_sum'])
                annulus_sums = np.array(ann_phot['aperture_sum'])

                local_bkg = np.where(ann_areas > 0, annulus_sums / ann_areas, median_bkg)
                net_flux = aperture_sums - local_bkg * ap_areas
                flux_err = np.sqrt(np.abs(net_flux) + ap_areas * local_bkg + ap_areas * std_bkg**2)

                # Quality flags (0=good, 1=edge, 2=negative, 3=out of bounds, 4=NaN)
                quality = edge_flags.copy()
                quality[net_flux < 0] = 2
                quality[~np.isfinite(net_flux)] = 4

                # Vectorized DataFrame creation
                epoch_df = pd.DataFrame({
                    'star_id': star_ids,
                    'btjd': btjd,
                    'flux': net_flux,
                    'flux_err': flux_err,
                    'quality': quality,
                    'x': positions[:, 0],
                    'y': positions[:, 1],
                })
                all_epoch_dfs.append(epoch_df)

        if all_epoch_dfs:
            return pd.concat(all_epoch_dfs, ignore_index=True)
        return pd.DataFrame()

    def status(self):
        """Print current pipeline status."""
        print(f"\n{'='*60}")
        print(f"CVZ Pipeline Status: {self.name}")
        print(f"{'='*60}")
        print(f"Coordinates: RA={self.ra:.2f}, Dec={self.dec:.2f}")
        print(f"Cutout size: {self.size_px}x{self.size_px} px")
        print()

        # Cutouts
        cutout_files = list(self.cutouts_dir.glob("sector_*.fits"))
        print(f"Cutouts: {len(cutout_files)} files")
        if cutout_files:
            sectors = sorted([int(f.stem.split('_')[1]) for f in cutout_files])
            print(f"  Sectors: {sectors}")

        # Catalog
        if self.catalog_path.exists():
            catalog = pd.read_csv(self.catalog_path)
            print(f"Star catalog: {len(catalog)} stars")
        else:
            print("Star catalog: Not created")

        # Photometry
        phot_files = list(self.photometry_dir.glob("sector_*.csv"))
        print(f"Photometry: {len(phot_files)} sectors processed")

        # Combined
        if self.combined_path.exists():
            combined = pd.read_csv(self.combined_path)
            n_stars = combined['star_id'].nunique()
            n_points = len(combined)
            time_span = combined['btjd'].max() - combined['btjd'].min()
            print(f"Combined: {n_stars} stars, {n_points:,} points, {time_span:.1f} days")
        else:
            print("Combined: Not created")

        print(f"{'='*60}")


def run_cvz_pipeline(
    name: str = "south_pole",
    pole: str = "south",
    size_px: int = 150,
    skip_download: bool = False
):
    """
    Convenience function to run full CVZ pipeline.

    Args:
        name: Project name
        pole: 'south' or 'north'
        size_px: Cutout size in pixels
        skip_download: Skip download step (use existing cutouts)

    Returns:
        CVZPipeline instance
    """
    cvz = CVZPipeline(name=name, pole=pole, size_px=size_px)

    if not skip_download:
        cvz.download_cutouts()

    cvz.detect_stars()
    cvz.run_photometry()
    cvz.combine_sectors()

    return cvz
