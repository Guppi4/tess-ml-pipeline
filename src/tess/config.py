"""
Configuration settings for TESS FFI Data Processing pipeline.

All paths and parameters are centralized here to avoid hardcoding.
"""

import os
from pathlib import Path

# ============================================================================
# Directory Paths
# ============================================================================

# Base directory (project root)
BASE_DIR = Path(__file__).resolve().parent.parent.parent

# Data directories (relative to project root)
FITS_DIR = BASE_DIR / "FITS"
CALIBRATED_DATA_DIR = BASE_DIR / "calibrated_data"
PHOTOMETRY_RESULTS_DIR = BASE_DIR / "photometry_results"
LIGHTCURVES_DIR = BASE_DIR / "lightcurves"
MANIFEST_DIR = BASE_DIR / "manifests"

# ============================================================================
# Cross-matching Parameters
# ============================================================================

# Tolerance for matching stars between epochs (arcseconds)
CROSSMATCH_TOLERANCE_ARCSEC = 3.0

# Minimum number of epochs a star must be detected to be included in catalog
MIN_EPOCHS_FOR_STAR = 3

# ============================================================================
# MAST Archive Settings
# ============================================================================

MAST_FFI_BASE_URL = "https://archive.stsci.edu/missions/tess/ffi"

# ============================================================================
# DAOStarFinder Parameters
# ============================================================================

# Full Width at Half Maximum for star detection (pixels)
DAOFIND_FWHM = 5.0

# Detection threshold in units of standard deviation
DAOFIND_THRESHOLD_SIGMA = 3.0

# ============================================================================
# Photometry Parameters
# ============================================================================

# Default aperture radius (pixels) - used if curve of growth fails
DEFAULT_APERTURE_RADIUS = 3.0

# Local background annulus radii (pixels)
# Inner radius should be larger than aperture to avoid star flux
# Outer radius should be small enough to capture local background variations
ANNULUS_R_IN = 5.0
ANNULUS_R_OUT = 8.0

# Range of radii for curve of growth analysis
COG_RADII_MIN = 1
COG_RADII_MAX = 10

# Sigma clipping threshold for background estimation
SIGMA_CLIP_VALUE = 3.0

# ============================================================================
# TIC Catalog Query Settings
# ============================================================================

# Search radius in degrees for TIC cross-matching
# Was 0.001 (~3.6"), too strict for TESS pixel scale (~21")
TIC_SEARCH_RADIUS = 0.02

# Delay between TIC queries (seconds) to avoid rate limiting
TIC_QUERY_DELAY = 0.5

# ============================================================================
# Helper Functions
# ============================================================================

def ensure_directories():
    """Create all required output directories if they don't exist."""
    for directory in [FITS_DIR, CALIBRATED_DATA_DIR, PHOTOMETRY_RESULTS_DIR, LIGHTCURVES_DIR, MANIFEST_DIR]:
        directory.mkdir(parents=True, exist_ok=True)


def get_mast_url(sector: str, year: str, day: str, camera: str, ccd: str) -> str:
    """
    Construct MAST FFI archive URL from observation parameters.

    Args:
        sector: TESS sector number
        year: Observation year
        day: Day of year
        camera: Camera number (1-4)
        ccd: CCD number (1-4)

    Returns:
        Full URL to the FFI directory on MAST
    """
    return f"{MAST_FFI_BASE_URL}/s{sector}/{year}/{day}/{camera}-{ccd}/"
