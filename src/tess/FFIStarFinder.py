"""
FFI Star Finder - Star detection and aperture photometry for TESS FFIs.

Uses DAOStarFinder algorithm (Stetson, 1987) for source detection and
performs curve-of-growth optimized aperture photometry.
"""

import codecs
codecs.register(lambda name: codecs.lookup('utf-8') if name == 'cp1255' else None)

import time as time_module
from datetime import datetime

import numpy as np
import pandas as pd
from astropy.io import fits
from astropy.time import Time
from astropy.wcs import WCS
from astroquery.mast import Catalogs
from photutils.aperture import CircularAperture, aperture_photometry
from photutils.detection import DAOStarFinder
from tqdm import tqdm

from .config import (
    PHOTOMETRY_RESULTS_DIR,
    DAOFIND_FWHM,
    DAOFIND_THRESHOLD_SIGMA,
    DEFAULT_APERTURE_RADIUS,
    COG_RADII_MIN,
    COG_RADII_MAX,
    TIC_SEARCH_RADIUS,
    TIC_QUERY_DELAY,
    ensure_directories,
)


def closest_source(sources, position):
    """
    Identify the closest source to a given position in an image.

    Args:
        sources: Table of sources with 'xcentroid' and 'ycentroid' columns
        position: Tuple of (x, y) coordinates

    Returns:
        Row from sources table for the closest source
    """
    distances = np.sqrt(
        (sources['xcentroid'] - position[0]) ** 2 +
        (sources['ycentroid'] - position[1]) ** 2
    )
    return sources[np.argmin(distances)]


def compare_to_background(phot_table, median):
    """
    Compare aperture sums to background level for signal-to-noise estimation.

    A source should be at least 5-10 times brighter than the median background
    to be clearly detectable.

    Args:
        phot_table: Photometry table with 'aperture_sum' column
        median: Median background level

    Returns:
        Updated photometry table with 'aperture_to_background' column
    """
    phot_table['aperture_to_background'] = phot_table['aperture_sum'] / median
    phot_table['median'] = median
    return phot_table


def add_tic_ids(df):
    """
    Add TIC (TESS Input Catalog) IDs to a DataFrame by cross-matching coordinates.

    Finds the NEAREST TIC match (not just first result) and saves separation.
    Note: This function includes rate limiting to avoid overwhelming MAST server.

    Args:
        df: DataFrame with 'ra' and 'dec' columns

    Returns:
        DataFrame with added 'tic' and 'tic_sep_arcsec' columns
    """
    from astropy.coordinates import SkyCoord
    import astropy.units as u

    tic_ids = []
    tic_separations = []

    for i, row in tqdm(df.iterrows(), total=len(df), desc="Querying TIC", unit="star"):
        catalogData = Catalogs.query_object(
            f"{row['ra']} {row['dec']}",
            radius=TIC_SEARCH_RADIUS,
            catalog="TIC"
        )
        if len(catalogData) > 0:
            # Find NEAREST match, not just first result
            source_coord = SkyCoord(ra=row['ra'] * u.deg, dec=row['dec'] * u.deg)
            tic_coords = SkyCoord(ra=catalogData['ra'] * u.deg, dec=catalogData['dec'] * u.deg)
            separations = source_coord.separation(tic_coords)
            nearest_idx = np.argmin(separations)

            tic_id = catalogData['ID'][nearest_idx]
            tic_sep = separations[nearest_idx].to(u.arcsec).value
        else:
            tic_id = None
            tic_sep = None

        tic_ids.append(tic_id)
        tic_separations.append(tic_sep)
        time_module.sleep(TIC_QUERY_DELAY)

    df['tic'] = tic_ids
    df['tic_sep_arcsec'] = tic_separations
    return df


def curve_of_growth(data, positions, radii):
    """
    Generate curve of growth by measuring flux at multiple aperture radii.

    Args:
        data: 2D image array
        positions: List of (x, y) positions
        radii: Iterable of aperture radii to test

    Returns:
        List of total flux values for each radius
    """
    fluxes = []
    for radius in radii:
        apertures = CircularAperture(positions, r=radius)
        phot_table = aperture_photometry(data, apertures)
        flux = np.sum(phot_table['aperture_sum'])
        fluxes.append(flux)
    return fluxes


def optimal_radius(fluxes, radii):
    """
    Find optimal aperture radius from curve of growth.

    The optimal radius is where the growth curve flattens (minimum slope).

    Args:
        fluxes: List of flux values from curve_of_growth
        radii: Corresponding radii

    Returns:
        Optimal aperture radius
    """
    slopes = [fluxes[i + 1] - fluxes[i] for i in range(len(fluxes) - 1)]
    optimal_idx = np.argmin(slopes)
    return radii[optimal_idx + 1]


def perform_photometry(data, sources):
    """
    Perform aperture photometry with curve-of-growth optimized radius.

    Args:
        data: 2D image array
        sources: Table of detected sources

    Returns:
        Photometry table with aperture sums
    """
    positions = list(zip(sources['xcentroid'], sources['ycentroid']))
    radii = range(COG_RADII_MIN, COG_RADII_MAX)

    # Generate curve of growth
    fluxes = curve_of_growth(data, positions, radii)

    # Determine optimal radius
    optimal_rad = optimal_radius(fluxes, radii)
    apertures = CircularAperture(positions, r=optimal_rad)
    phot_table = aperture_photometry(data, apertures)

    return phot_table


def calculate_flux(phot_table, exposure_time):
    """
    Calculate flux from aperture sums.

    Args:
        phot_table: Photometry table with 'aperture_sum' column
        exposure_time: Exposure time in seconds

    Returns:
        Updated photometry table with 'flux' column
    """
    phot_table['flux'] = phot_table['aperture_sum'] / exposure_time
    return phot_table


def calculate_flux_error(phot_table, n_pixels, median, gain, std, exposure_time=1.0):
    """
    Calculate flux uncertainty using CCD noise model.

    Error sources:
    1. Poisson noise from signal: aperture_sum / gain
    2. Poisson noise from background: n_pixels * median * gain
    3. Read noise from detector: n_pixels^2 * std^2

    Args:
        phot_table: Photometry table with 'aperture_sum' column
        n_pixels: Number of pixels in aperture
        median: Median background level
        gain: Detector gain (e-/ADU)
        std: Background standard deviation
        exposure_time: Exposure time in seconds (to convert to counts/second)

    Returns:
        Updated photometry table with 'flux_error' column (in counts/second)
    """
    # CCD noise model with correct dimensions:
    # - Signal Poisson: aperture_sum / gain
    # - Background Poisson: n_pixels * median / gain
    # - Read noise: n_pixels * (std/gain)² (not n_pixels²!)
    # Calculate error in counts, then convert to counts/second
    flux_error_counts = np.sqrt(
        phot_table['aperture_sum'] / gain +
        n_pixels * median / gain +
        n_pixels * (std / gain) ** 2
    )
    phot_table['flux_error'] = flux_error_counts / exposure_time
    return phot_table


def format_time(time_string):
    """Convert ISO time string to human-readable format."""
    dt = datetime.strptime(time_string, "%Y-%m-%dT%H:%M:%S.%f")
    return dt.strftime("%I:%M:%S %p on %B %d, %Y")


def process_fits_file(fits_files, data_arrays, means, medians, stds):
    """
    Process FITS files and perform initial photometry on detected sources.

    Uses DAOStarFinder for source detection and extracts:
    - Photometry table
    - Detected sources
    - Exposure time
    - Gain values
    - Observation timestamps

    Args:
        fits_files: List of FITS file paths
        data_arrays: Calibrated image arrays
        means: Background mean values
        medians: Background median values
        stds: Background standard deviations

    Returns:
        List of result tuples for each file
    """
    results = []

    for fits_file, data_array, mean, median, std in tqdm(
        zip(fits_files, data_arrays, means, medians, stds),
        total=len(fits_files),
        desc="Processing FITS",
        unit="file"
    ):
        with fits.open(fits_file) as hdu:
            data = data_array

            # Find stars using DAOStarFinder (Stetson 1987)
            daofind = DAOStarFinder(
                fwhm=DAOFIND_FWHM,
                threshold=DAOFIND_THRESHOLD_SIGMA * std
            )
            sources = daofind(data - median)

            if sources is None or len(sources) == 0:
                print(f"No sources found in {fits_file}")
                continue

            # Perform aperture photometry
            phot_table = perform_photometry(data, sources)

            # Calculate exposure time
            # TESS TSTART/TSTOP are in BTJD (days), need seconds for flux
            header0 = hdu[0].header
            exposure_time = header0.get('EXPOSURE',
                (header0['TSTOP'] - header0['TSTART']) * 86400.0)

            # Calculate flux
            phot_table = calculate_flux(phot_table, exposure_time)

            # Estimate pixels in aperture
            n_pixels = np.pi * DEFAULT_APERTURE_RADIUS ** 2

            # Extract and average gain values
            gain = (
                hdu[1].header['GAINA'] +
                hdu[1].header['GAINB'] +
                hdu[1].header['GAINC'] +
                hdu[1].header['GAIND']
            ) / 4

            # Calculate flux error (pass exposure_time for consistent units)
            phot_table = calculate_flux_error(phot_table, n_pixels, median, gain, std, exposure_time)

            # Get timestamps
            header = hdu[0].header
            date_obs = header['DATE-OBS']
            date_end = header['DATE-END']

            mjd_obs = Time(date_obs, format='isot', scale='utc').mjd
            mjd_end = Time(date_end, format='isot', scale='utc').mjd

            results.append((
                phot_table, sources, exposure_time, n_pixels,
                median, gain, std, data, date_obs, mjd_obs, mjd_end
            ))

    return results


def find_stars(fits_files, data_arrays, means, medians, stds):
    """
    Main function to detect stars and perform photometry on all FITS files.

    Args:
        fits_files: List of FITS file paths
        data_arrays: Calibrated image arrays
        means: Background mean values
        medians: Background median values
        stds: Background standard deviations

    Returns:
        Status message
    """
    ensure_directories()

    results = process_fits_file(fits_files, data_arrays, means, medians, stds)

    print(f"Processing {len(fits_files)} files...")

    for i, (result, fits_file) in enumerate(zip(results, fits_files)):
        (phot_table, sources, exposure_time, n_pixels,
         median, gain, std, data, date_obs, mjd_obs, mjd_end) = result

        # Perform final aperture photometry
        positions = [(source['xcentroid'], source['ycentroid']) for source in sources]
        apertures = CircularAperture(positions, r=DEFAULT_APERTURE_RADIUS)
        phot_table = aperture_photometry(data, apertures)

        # Add timestamps
        phot_table['time'] = format_time(date_obs)
        phot_table['MJD_OBS'] = mjd_obs

        # Convert pixel to celestial coordinates
        with fits.open(fits_file) as hdu:
            w = WCS(hdu[1].header)
            positions = list(zip(sources['xcentroid'], sources['ycentroid']))
            world_coords = w.all_pix2world(positions, 0)
            ra_vals = world_coords[:, 0]
            dec_vals = world_coords[:, 1]

            # Check for NaN values (WCS can return NaN without exception)
            valid_mask = ~np.isnan(ra_vals) & ~np.isnan(dec_vals)
            if not np.all(valid_mask):
                print(f"  Warning: {np.sum(~valid_mask)} sources have invalid WCS coordinates")

            phot_table['ra'] = ra_vals
            phot_table['dec'] = dec_vals

        # Calculate flux and errors (both in counts/second)
        phot_table = calculate_flux(phot_table, exposure_time)
        phot_table = calculate_flux_error(phot_table, n_pixels, median, gain, std, exposure_time)

        # Add signal-to-noise ratio
        phot_table = compare_to_background(phot_table, median)

        # Save results
        output_path = PHOTOMETRY_RESULTS_DIR / f'photometry_results_{i}.csv'
        phot_table.to_pandas().to_csv(output_path, index=False)

        print(f"Saved: {output_path.name}")

    print(f"\nPhotometry complete. Results saved to {PHOTOMETRY_RESULTS_DIR}")
    return "Find stars in the calibrated images."
