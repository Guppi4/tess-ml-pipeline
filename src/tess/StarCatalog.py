"""
Star Catalog - Cross-matching stars across epochs.

This module provides functionality to:
1. Build a master star catalog from a reference epoch
2. Match stars across multiple epochs using celestial coordinates
3. Track star identity throughout the observation period
"""

import json
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from astropy.wcs import WCS
from astropy.io.fits import Header
from astropy.coordinates import SkyCoord
import astropy.units as u
from photutils.detection import DAOStarFinder
from photutils.aperture import CircularAperture, CircularAnnulus, aperture_photometry
from tqdm import tqdm

from .config import (
    CALIBRATED_DATA_DIR,
    PHOTOMETRY_RESULTS_DIR,
    DAOFIND_FWHM,
    DAOFIND_THRESHOLD_SIGMA,
    DEFAULT_APERTURE_RADIUS,
    ANNULUS_R_IN,
    ANNULUS_R_OUT,
    CROSSMATCH_TOLERANCE_ARCSEC,
    MIN_EPOCHS_FOR_STAR,
    ensure_directories
)
from .FFICalibrate import load_calibration, load_epoch_data


class StarCatalog:
    """
    A catalog of stars with consistent IDs across all epochs.

    The catalog is built from a reference epoch and stars from other
    epochs are matched to this reference using celestial coordinates.
    """

    def __init__(self):
        self.stars = {}  # star_id -> {ra, dec, pixel_x, pixel_y, ...}
        self.photometry = {}  # star_id -> {epoch_idx: {flux, error, ...}}
        self.epochs = []  # List of epoch metadata
        self.n_stars = 0
        self.reference_epoch = None

    def detect_stars_in_epoch(self, data: np.ndarray, wcs_header: dict,
                               median: float, std: float) -> pd.DataFrame:
        """
        Detect stars in a single epoch.

        Returns:
            DataFrame with columns: xcentroid, ycentroid, ra, dec, peak, flux, ...
        """
        # Run DAOStarFinder
        daofind = DAOStarFinder(
            fwhm=DAOFIND_FWHM,
            threshold=DAOFIND_THRESHOLD_SIGMA * std
        )

        # Subtract background
        sources = daofind(data - median)

        if sources is None or len(sources) == 0:
            return pd.DataFrame()

        # Convert to DataFrame
        df = sources.to_pandas()

        # Add celestial coordinates
        if wcs_header:
            try:
                header = Header()
                for k, v in wcs_header.items():
                    try:
                        header[k] = v
                    except:
                        pass
                wcs = WCS(header)

                positions = list(zip(df['xcentroid'], df['ycentroid']))
                world_coords = wcs.all_pix2world(positions, 0)
                ra_vals = world_coords[:, 0]
                dec_vals = world_coords[:, 1]

                # Check for NaN values (WCS can return NaN without exception)
                valid_mask = ~np.isnan(ra_vals) & ~np.isnan(dec_vals)
                if not np.all(valid_mask):
                    print(f"  Warning: {np.sum(~valid_mask)} sources have invalid WCS coordinates")

                df['ra'] = ra_vals
                df['dec'] = dec_vals
            except Exception as e:
                print(f"WCS conversion failed: {e}")
                df['ra'] = np.nan
                df['dec'] = np.nan
        else:
            df['ra'] = np.nan
            df['dec'] = np.nan

        return df

    def build_from_reference(self, calibration_metadata: dict,
                             reference_epoch_idx: int = 0) -> int:
        """
        Build the star catalog from a reference epoch.

        Args:
            calibration_metadata: Loaded calibration metadata
            reference_epoch_idx: Index of epoch to use as reference

        Returns:
            Number of stars in catalog
        """
        self.epochs = calibration_metadata['epochs']
        self.reference_epoch = reference_epoch_idx

        # Load reference epoch data
        ref_epoch = self.epochs[reference_epoch_idx]
        data, wcs_header = load_epoch_data(reference_epoch_idx)

        median = ref_epoch['stats']['median']
        std = ref_epoch['stats']['std']

        # Detect stars
        print(f"Detecting stars in reference epoch {reference_epoch_idx}...")
        sources = self.detect_stars_in_epoch(data, wcs_header, median, std)

        if len(sources) == 0:
            print("No stars detected in reference epoch!")
            return 0

        # Build catalog
        for idx, row in sources.iterrows():
            star_id = f"STAR_{idx:06d}"

            self.stars[star_id] = {
                'star_id': star_id,
                'ra': row['ra'],
                'dec': row['dec'],
                'ref_xcentroid': row['xcentroid'],
                'ref_ycentroid': row['ycentroid'],
                'ref_peak': row.get('peak', 0),
                'ref_flux': row.get('flux', 0),
                'detection_count': 0,
                'epochs_detected': []
            }

            self.photometry[star_id] = {}

        self.n_stars = len(self.stars)
        print(f"Catalog built with {self.n_stars} stars from reference epoch")

        return self.n_stars

    def match_epoch(self, epoch_idx: int, sources: pd.DataFrame,
                    tolerance_arcsec: float = None) -> dict:
        """
        Match detected stars from an epoch to the catalog.

        Args:
            epoch_idx: Index of epoch being matched
            sources: DataFrame of detected sources with ra, dec columns
            tolerance_arcsec: Matching tolerance in arcseconds

        Returns:
            Dictionary mapping source index to star_id (or None if no match)
        """
        if tolerance_arcsec is None:
            tolerance_arcsec = CROSSMATCH_TOLERANCE_ARCSEC

        if len(sources) == 0 or len(self.stars) == 0:
            return {}

        # Get catalog coordinates
        catalog_ids = list(self.stars.keys())
        catalog_ra = [self.stars[sid]['ra'] for sid in catalog_ids]
        catalog_dec = [self.stars[sid]['dec'] for sid in catalog_ids]

        # Create SkyCoord objects
        catalog_coords = SkyCoord(ra=catalog_ra * u.degree, dec=catalog_dec * u.degree)
        source_coords = SkyCoord(ra=sources['ra'].values * u.degree,
                                 dec=sources['dec'].values * u.degree)

        # Match using astropy's match_to_catalog_sky
        idx, sep2d, _ = source_coords.match_to_catalog_sky(catalog_coords)

        # Build matches
        matches = {}
        tolerance = tolerance_arcsec * u.arcsec

        for source_idx in range(len(sources)):
            matched_catalog_idx = idx[source_idx]
            separation = sep2d[source_idx]

            if separation <= tolerance:
                matches[source_idx] = catalog_ids[matched_catalog_idx]
            else:
                matches[source_idx] = None

        return matches

    def perform_photometry_epoch(self, epoch_idx: int) -> dict:
        """
        Perform aperture photometry for all catalog stars in a specific epoch.

        This uses FORCED photometry - we measure at the catalog positions,
        not at the detected positions. This ensures consistent measurements
        even if a star is not detected by DAOStarFinder in this epoch.

        Returns:
            Dictionary with photometry results for each star
        """
        epoch_meta = self.epochs[epoch_idx]
        data, wcs_header = load_epoch_data(epoch_idx)

        median = epoch_meta['stats']['median']
        std = epoch_meta['stats']['std']
        mjd = epoch_meta['metadata'].get('mjd_obs')
        exposure_time = epoch_meta['metadata'].get('exposure_time', 1)
        gain = epoch_meta['metadata'].get('gain_avg', 5.2)

        # Get WCS for coordinate conversion
        if wcs_header:
            header = Header()
            for k, v in wcs_header.items():
                try:
                    header[k] = v
                except:
                    pass
            wcs = WCS(header)
        else:
            print(f"Warning: No WCS for epoch {epoch_idx}")
            return {}

        results = {}

        # For each star, convert RA/Dec to pixel and measure flux
        for star_id, star_info in self.stars.items():
            ra = star_info['ra']
            dec = star_info['dec']

            # Convert to pixel coordinates for this epoch
            try:
                pixel_coords = wcs.all_world2pix([[ra, dec]], 0)
                x, y = pixel_coords[0]
                # Check for NaN (WCS can return NaN without exception)
                if np.isnan(x) or np.isnan(y):
                    raise ValueError("WCS returned NaN")
            except Exception as e:
                results[star_id] = {
                    'flux': np.nan,
                    'flux_error': np.nan,
                    'quality_flag': 4,  # WCS conversion failed
                    'btjd': mjd
                }
                continue

            # Check if position is within image bounds
            if x < 0 or y < 0 or x >= data.shape[1] or y >= data.shape[0]:
                results[star_id] = {
                    'flux': np.nan,
                    'flux_error': np.nan,
                    'xcentroid': x,
                    'ycentroid': y,
                    'quality_flag': 3,  # Outside image
                    'btjd': mjd
                }
                continue

            # Check if position is near edge
            edge_distance = min(x, y, data.shape[1] - x, data.shape[0] - y)
            near_edge = edge_distance < DEFAULT_APERTURE_RADIUS * 2

            # Perform aperture photometry with LOCAL background (annulus)
            aperture = CircularAperture([(x, y)], r=DEFAULT_APERTURE_RADIUS)
            annulus = CircularAnnulus([(x, y)], r_in=ANNULUS_R_IN, r_out=ANNULUS_R_OUT)

            # Measure in both apertures
            phot_table = aperture_photometry(data, aperture)
            ann_phot = aperture_photometry(data, annulus)

            # Calculate local background per pixel from annulus
            annulus_area = annulus.area_overlap(data)
            local_bkg_per_pixel = ann_phot['aperture_sum'][0] / annulus_area if annulus_area > 0 else median

            # Subtract local background from aperture
            n_pixels = np.pi * DEFAULT_APERTURE_RADIUS ** 2
            aperture_sum = phot_table['aperture_sum'][0] - (local_bkg_per_pixel * n_pixels)

            # Calculate flux (counts/second)
            flux = aperture_sum / exposure_time if exposure_time > 0 else aperture_sum

            # Calculate flux error (CCD noise model with correct dimensions)
            # Using LOCAL background instead of global median
            # Error in counts, then convert to counts/second
            flux_error_counts = np.sqrt(
                abs(aperture_sum) / gain +
                n_pixels * local_bkg_per_pixel / gain +
                n_pixels * (std / gain) ** 2
            )
            flux_error = flux_error_counts / exposure_time if exposure_time > 0 else flux_error_counts

            # Determine quality flag
            if aperture_sum < 0:
                quality_flag = 2  # Negative flux (likely noise)
            elif near_edge:
                quality_flag = 1  # Near edge
            else:
                quality_flag = 0  # Good measurement

            # Signal-to-noise
            snr = flux / flux_error if flux_error > 0 else 0

            results[star_id] = {
                'flux': float(flux),
                'flux_error': float(flux_error),
                'aperture_sum': float(aperture_sum),
                'xcentroid': float(x),
                'ycentroid': float(y),
                'quality_flag': int(quality_flag),
                'snr': float(snr),
                'background': float(local_bkg_per_pixel),
                'btjd': mjd
            }

        return results

    def process_all_epochs(self, calibration_metadata: dict = None,
                          reference_epoch_idx: int = 0) -> None:
        """
        Process all epochs: build catalog from reference and measure photometry.

        This is the main entry point for building the complete catalog.
        """
        if calibration_metadata is None:
            calibration_metadata = load_calibration()

        # Build catalog from reference
        self.build_from_reference(calibration_metadata, reference_epoch_idx)

        if self.n_stars == 0:
            print("No stars in catalog, aborting.")
            return

        # Process each epoch
        print(f"\nProcessing {len(self.epochs)} epochs...")

        for epoch_idx in tqdm(range(len(self.epochs)), desc="Processing epochs"):
            # Perform forced photometry
            epoch_results = self.perform_photometry_epoch(epoch_idx)

            # Store results
            for star_id, result in epoch_results.items():
                self.photometry[star_id][epoch_idx] = result

                # Update detection count for good measurements
                if result['quality_flag'] == 0:
                    self.stars[star_id]['detection_count'] += 1
                    self.stars[star_id]['epochs_detected'].append(epoch_idx)

        # Summary
        detected_counts = [self.stars[sid]['detection_count'] for sid in self.stars]
        print(f"\nProcessing complete:")
        print(f"  Total stars: {self.n_stars}")
        print(f"  Stars with >{MIN_EPOCHS_FOR_STAR} detections: "
              f"{sum(1 for c in detected_counts if c >= MIN_EPOCHS_FOR_STAR)}")
        print(f"  Median detections per star: {np.median(detected_counts):.0f}")

    def get_lightcurve(self, star_id: str) -> pd.DataFrame:
        """
        Get lightcurve for a specific star.

        Returns:
            DataFrame with columns: mjd, flux, flux_error, quality_flag, ...
        """
        if star_id not in self.photometry:
            return pd.DataFrame()

        records = []
        for epoch_idx, data in self.photometry[star_id].items():
            record = {'epoch': epoch_idx}
            record.update(data)
            records.append(record)

        df = pd.DataFrame(records)
        if len(df) > 0:
            # Support both btjd (new) and mjd (legacy) column names
            time_col = 'btjd' if 'btjd' in df.columns else 'mjd' if 'mjd' in df.columns else None
            if time_col:
                df = df.sort_values(time_col)

        return df

    def get_all_lightcurves(self, min_detections: int = None) -> dict:
        """
        Get lightcurves for all stars meeting detection threshold.

        Args:
            min_detections: Minimum number of good detections required

        Returns:
            Dictionary of star_id -> DataFrame
        """
        if min_detections is None:
            min_detections = MIN_EPOCHS_FOR_STAR

        lightcurves = {}

        for star_id, star_info in self.stars.items():
            if star_info['detection_count'] >= min_detections:
                lightcurves[star_id] = self.get_lightcurve(star_id)

        return lightcurves

    def to_master_table(self) -> pd.DataFrame:
        """
        Export catalog to a master table with all photometry.

        Returns:
            DataFrame with one row per star, columns for each epoch's flux
        """
        records = []

        for star_id, star_info in self.stars.items():
            record = {
                'star_id': star_id,
                'ra': star_info['ra'],
                'dec': star_info['dec'],
                'detection_count': star_info['detection_count']
            }

            # Add flux for each epoch
            for epoch_idx in range(len(self.epochs)):
                if epoch_idx in self.photometry[star_id]:
                    phot = self.photometry[star_id][epoch_idx]
                    record[f'flux_{epoch_idx:04d}'] = phot['flux']
                    record[f'err_{epoch_idx:04d}'] = phot['flux_error']
                    record[f'flag_{epoch_idx:04d}'] = phot['quality_flag']
                else:
                    record[f'flux_{epoch_idx:04d}'] = np.nan
                    record[f'err_{epoch_idx:04d}'] = np.nan
                    record[f'flag_{epoch_idx:04d}'] = 5  # Not processed

            records.append(record)

        return pd.DataFrame(records)

    def save(self, filename: str = None) -> str:
        """
        Save catalog to disk.
        """
        ensure_directories()

        if filename is None:
            filename = 'star_catalog.json'

        filepath = PHOTOMETRY_RESULTS_DIR / filename

        # Prepare data for JSON
        data = {
            'n_stars': self.n_stars,
            'n_epochs': len(self.epochs),
            'reference_epoch': self.reference_epoch,
            'stars': self.stars,
            'photometry': {
                star_id: {
                    str(epoch_idx): phot_data
                    for epoch_idx, phot_data in epoch_phots.items()
                }
                for star_id, epoch_phots in self.photometry.items()
            },
            'epoch_mjds': [e['metadata'].get('mjd_obs') for e in self.epochs]
        }

        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2, default=str)

        # Also save master table as CSV
        master_table = self.to_master_table()
        csv_path = PHOTOMETRY_RESULTS_DIR / 'master_photometry.csv'
        master_table.to_csv(csv_path, index=False)

        print(f"Catalog saved to {filepath}")
        print(f"Master table saved to {csv_path}")

        return str(filepath)

    @classmethod
    def load(cls, filename: str = None) -> 'StarCatalog':
        """
        Load catalog from disk.
        """
        if filename is None:
            filename = 'star_catalog.json'

        filepath = PHOTOMETRY_RESULTS_DIR / filename

        with open(filepath, 'r') as f:
            data = json.load(f)

        catalog = cls()
        catalog.n_stars = data['n_stars']
        catalog.reference_epoch = data['reference_epoch']
        catalog.stars = data['stars']

        # Convert epoch keys back to integers
        catalog.photometry = {
            star_id: {
                int(epoch_idx): phot_data
                for epoch_idx, phot_data in epoch_phots.items()
            }
            for star_id, epoch_phots in data['photometry'].items()
        }

        return catalog


    def add_tic_ids(self, checkpoint_path=None) -> int:
        """
        Add TIC IDs to all stars using FFIStarFinder.add_tic_ids().

        Stars without TIC match keep their internal STAR_XXXXXX ID.
        Supports checkpointing to resume after failures.

        Args:
            checkpoint_path: Path to save/load checkpoint (optional).
                            If None, uses default path in data/tess/ directory.

        Returns:
            Number of stars with TIC matches
        """
        from .FFIStarFinder import add_tic_ids
        from .config import DATA_DIR

        # Default checkpoint path
        if checkpoint_path is None:
            checkpoint_path = DATA_DIR / "tic_checkpoint.json"

        # Convert to DataFrame
        records = []
        for star_id, info in self.stars.items():
            records.append({
                'star_id': star_id,
                'ra': info.get('ra'),
                'dec': info.get('dec'),
            })

        df = pd.DataFrame(records)
        valid = df[df['ra'].notna() & df['dec'].notna()].copy()

        print(f"Stars with valid coordinates: {len(valid)}/{len(df)}")

        # Use existing function with checkpoint
        valid = add_tic_ids(valid, checkpoint_path=checkpoint_path)

        # Update catalog
        tic_count = 0
        for _, row in valid.iterrows():
            star_id = row['star_id']
            tic = row.get('tic')
            if tic is not None:
                self.stars[star_id]['tic_id'] = str(tic)
                tic_count += 1
            else:
                self.stars[star_id]['tic_id'] = star_id

        print(f"TIC matches: {tic_count}/{len(valid)} ({tic_count/len(valid)*100:.1f}%)")
        return tic_count


def build_star_catalog(reference_epoch: int = 0) -> StarCatalog:
    """
    Convenience function to build a complete star catalog.

    Args:
        reference_epoch: Index of epoch to use as reference (default: 0, first epoch)

    Returns:
        Populated StarCatalog object
    """
    # Load calibration
    calibration = load_calibration()

    # Create and populate catalog
    catalog = StarCatalog()
    catalog.process_all_epochs(calibration, reference_epoch)

    # Save
    catalog.save()

    return catalog
