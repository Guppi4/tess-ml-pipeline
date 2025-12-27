"""
FFI Calibration - Background estimation and noise removal for TESS FFIs.

Now includes metadata preservation to maintain temporal information.
"""

import json
import numpy as np
from astropy.io import fits
from astropy.stats import sigma_clipped_stats
from astropy.time import Time
from astropy.wcs import WCS
from tqdm import tqdm
from datetime import datetime

from .config import CALIBRATED_DATA_DIR, SIGMA_CLIP_VALUE, ensure_directories


def extract_fits_metadata(fits_file: str, hdu) -> dict:
    """
    Extract all relevant metadata from a FITS file.

    Args:
        fits_file: Path to FITS file
        hdu: Opened HDU object

    Returns:
        Dictionary with metadata
    """
    header0 = hdu[0].header
    header1 = hdu[1].header

    metadata = {
        'filename': str(fits_file).split('/')[-1].split('\\')[-1],
        'filepath': str(fits_file),

        # Temporal info
        'date_obs': header0.get('DATE-OBS', ''),
        'date_end': header0.get('DATE-END', ''),
        'tstart': header0.get('TSTART', 0),
        'tstop': header0.get('TSTOP', 0),

        # Observation parameters
        'sector': header0.get('SECTOR', 0),
        'camera': header0.get('CAMERA', 0),
        'ccd': header0.get('CCD', 0),

        # Detector info
        'gain_a': header1.get('GAINA', 0),
        'gain_b': header1.get('GAINB', 0),
        'gain_c': header1.get('GAINC', 0),
        'gain_d': header1.get('GAIND', 0),
    }

    # Calculate derived values
    # TESS TSTART/TSTOP are in BTJD (days), need seconds for flux calculation
    # Use EXPOSURE header if available, otherwise convert days to seconds
    metadata['exposure_time'] = header0.get('EXPOSURE',
        (metadata['tstop'] - metadata['tstart']) * 86400.0)
    metadata['gain_avg'] = (metadata['gain_a'] + metadata['gain_b'] +
                           metadata['gain_c'] + metadata['gain_d']) / 4

    # Calculate MJD
    if metadata['date_obs']:
        try:
            metadata['mjd_obs'] = Time(metadata['date_obs'], format='isot', scale='utc').mjd
        except:
            metadata['mjd_obs'] = None
    else:
        metadata['mjd_obs'] = None

    if metadata['date_end']:
        try:
            metadata['mjd_end'] = Time(metadata['date_end'], format='isot', scale='utc').mjd
        except:
            metadata['mjd_end'] = None
    else:
        metadata['mjd_end'] = None

    return metadata


def calibrate_single_file(fits_file: str) -> dict:
    """
    Calibrate a single FITS file and return all data with metadata.

    Returns:
        Dictionary with 'data', 'stats', 'metadata', 'wcs'
    """
    with fits.open(fits_file, mode='readonly') as hdu:
        data = hdu[1].data

        # Background estimation
        mean, median, std = sigma_clipped_stats(data, sigma=SIGMA_CLIP_VALUE)

        # Extract metadata
        metadata = extract_fits_metadata(fits_file, hdu)

        # Extract WCS for coordinate conversion later
        try:
            wcs_header = dict(hdu[1].header)
        except:
            wcs_header = None

        return {
            'data': data,
            'stats': {
                'mean': float(mean),
                'median': float(median),
                'std': float(std)
            },
            'metadata': metadata,
            'wcs_header': wcs_header
        }


def calibrate_background(fits_files: list, manifest: dict = None) -> dict:
    """
    Calibrate FFI images by estimating and removing background noise.

    Now preserves all metadata for each epoch.

    Args:
        fits_files: List of paths to FITS files to calibrate
        manifest: Optional manifest from downloader (for additional metadata)

    Returns:
        Dictionary with:
            - epochs: List of calibrated data per epoch
            - summary: Overall statistics
    """
    ensure_directories()

    epochs = []
    failed_files = []

    for i, fits_file in enumerate(tqdm(fits_files, desc="Calibrating FFIs", unit="file")):
        try:
            result = calibrate_single_file(fits_file)
            result['epoch_index'] = i
            epochs.append(result)

        except Exception as e:
            print(f"\nFailed to process {fits_file}: {e}")
            failed_files.append({
                'filepath': str(fits_file),
                'error': str(e),
                'epoch_index': i
            })

    # Sort epochs by MJD to ensure temporal order
    epochs.sort(key=lambda x: x['metadata'].get('mjd_obs', 0) or 0)

    # Re-index after sorting
    for i, epoch in enumerate(epochs):
        epoch['epoch_index'] = i

    # Create calibration result
    calibration_result = {
        'created': datetime.now().isoformat(),
        'n_epochs': len(epochs),
        'n_failed': len(failed_files),
        'epochs': epochs,
        'failed_files': failed_files,
        'time_coverage': {
            'mjd_start': min(e['metadata']['mjd_obs'] for e in epochs if e['metadata'].get('mjd_obs')),
            'mjd_end': max(e['metadata']['mjd_obs'] for e in epochs if e['metadata'].get('mjd_obs')),
        } if epochs else {}
    }

    # Save calibration data
    save_calibration(calibration_result)

    print(f"\nCalibration complete:")
    print(f"  Processed: {len(epochs)} files")
    print(f"  Failed: {len(failed_files)} files")
    print(f"  Results saved to {CALIBRATED_DATA_DIR}")

    return calibration_result


def save_calibration(calibration_result: dict):
    """
    Save calibration results to disk.

    Saves:
    - calibration_metadata.json: All metadata (without pixel data)
    - epoch_XXXX.npy: Pixel data for each epoch
    - epoch_XXXX_wcs.json: WCS header for each epoch
    """
    # Prepare metadata (without large arrays)
    metadata_for_json = {
        'created': calibration_result['created'],
        'n_epochs': calibration_result['n_epochs'],
        'n_failed': calibration_result['n_failed'],
        'time_coverage': calibration_result['time_coverage'],
        'failed_files': calibration_result['failed_files'],
        'epochs': []
    }

    # Save each epoch's data separately
    for epoch in calibration_result['epochs']:
        epoch_idx = epoch['epoch_index']

        # Save pixel data
        data_path = CALIBRATED_DATA_DIR / f'epoch_{epoch_idx:04d}.npy'
        np.save(data_path, epoch['data'])

        # Save WCS header
        if epoch['wcs_header']:
            wcs_path = CALIBRATED_DATA_DIR / f'epoch_{epoch_idx:04d}_wcs.json'
            with open(wcs_path, 'w') as f:
                # Convert header to serializable format
                wcs_dict = {k: str(v) if not isinstance(v, (int, float, bool, type(None))) else v
                           for k, v in epoch['wcs_header'].items()}
                json.dump(wcs_dict, f, indent=2)

        # Add to metadata (without data array)
        epoch_meta = {
            'epoch_index': epoch_idx,
            'stats': epoch['stats'],
            'metadata': epoch['metadata'],
            'data_path': str(data_path),
        }
        metadata_for_json['epochs'].append(epoch_meta)

    # Save metadata JSON
    metadata_path = CALIBRATED_DATA_DIR / 'calibration_metadata.json'
    with open(metadata_path, 'w') as f:
        json.dump(metadata_for_json, f, indent=2, default=str)

    # Also save legacy format for backward compatibility
    save_legacy_format(calibration_result)


def save_legacy_format(calibration_result: dict):
    """Save in old format for backward compatibility."""
    data_arrays = [e['data'] for e in calibration_result['epochs']]
    means = [e['stats']['mean'] for e in calibration_result['epochs']]
    medians = [e['stats']['median'] for e in calibration_result['epochs']]
    stds = [e['stats']['std'] for e in calibration_result['epochs']]

    np.save(CALIBRATED_DATA_DIR / 'data_arrays.npy', data_arrays)
    np.save(CALIBRATED_DATA_DIR / 'means.npy', means)
    np.save(CALIBRATED_DATA_DIR / 'medians.npy', medians)
    np.save(CALIBRATED_DATA_DIR / 'stds.npy', stds)


def load_calibration() -> dict:
    """
    Load calibration results from disk.

    Returns:
        Dictionary with calibration data and metadata
    """
    metadata_path = CALIBRATED_DATA_DIR / 'calibration_metadata.json'

    if not metadata_path.exists():
        raise FileNotFoundError(f"Calibration metadata not found at {metadata_path}")

    with open(metadata_path, 'r') as f:
        metadata = json.load(f)

    return metadata


def load_epoch_data(epoch_index: int) -> tuple:
    """
    Load pixel data for a specific epoch.

    Returns:
        Tuple of (data_array, wcs_header)
    """
    data_path = CALIBRATED_DATA_DIR / f'epoch_{epoch_index:04d}.npy'
    wcs_path = CALIBRATED_DATA_DIR / f'epoch_{epoch_index:04d}_wcs.json'

    data = np.load(data_path)

    wcs_header = None
    if wcs_path.exists():
        with open(wcs_path, 'r') as f:
            wcs_header = json.load(f)

    return data, wcs_header


def get_epoch_wcs(epoch_index: int) -> WCS:
    """Get WCS object for a specific epoch."""
    _, wcs_header = load_epoch_data(epoch_index)

    if wcs_header:
        from astropy.io.fits import Header
        header = Header()
        for k, v in wcs_header.items():
            try:
                header[k] = v
            except:
                pass
        return WCS(header)

    return None
