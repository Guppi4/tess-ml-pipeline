"""
Configuration settings for photometry pipeline.

Supports multiple satellites (TESS, Kepler, K2, etc.)
All paths and parameters are centralized here.
"""

import os
from pathlib import Path
from typing import Optional

# ============================================================================
# Directory Paths
# ============================================================================

# Base directory (project root)
BASE_DIR = Path(__file__).resolve().parent.parent.parent

# Main data directory (all satellite data goes here)
DATA_DIR = BASE_DIR / "data"

# Legacy paths (for backward compatibility)
FITS_DIR = BASE_DIR / "FITS"
CALIBRATED_DATA_DIR = BASE_DIR / "calibrated_data"
PHOTOMETRY_RESULTS_DIR = BASE_DIR / "photometry_results"
LIGHTCURVES_DIR = BASE_DIR / "lightcurves"
MANIFEST_DIR = BASE_DIR / "manifests"

# ============================================================================
# Satellite-specific path helpers
# ============================================================================

def get_satellite_dir(satellite: str = "tess") -> Path:
    """Get root directory for a satellite's data."""
    return DATA_DIR / satellite


def get_tess_sector_dir(sector: int, camera: int = None, ccd: int = None) -> Path:
    """
    Get directory for TESS sector data.

    Args:
        sector: Sector number (1-99)
        camera: Camera number (1-4), optional
        ccd: CCD number (1-4), optional

    Returns:
        Path to sector/camera/ccd directory

    Examples:
        get_tess_sector_dir(70) -> data/tess/sector_070/
        get_tess_sector_dir(70, 1, 1) -> data/tess/sector_070/cam1_ccd1/
    """
    path = DATA_DIR / "tess" / f"sector_{sector:03d}"
    if camera is not None and ccd is not None:
        path = path / f"cam{camera}_ccd{ccd}"
    return path


def get_exports_dir() -> Path:
    """Get directory for ML exports."""
    return DATA_DIR / "exports"


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
ANNULUS_R_IN = 5.0
ANNULUS_R_OUT = 8.0

# Range of radii for curve of growth analysis
COG_RADII_MIN = 1
COG_RADII_MAX = 10

# Sigma clipping threshold for background estimation
SIGMA_CLIP_VALUE = 3.0

# ============================================================================
# Known Artifact Windows
# ============================================================================

# Time periods with known data quality issues (e.g., scattered light from Moon)
# Format: {sector: [(btjd_start, btjd_end, description), ...]}
# Source: https://heasarc.gsfc.nasa.gov/docs/tess/sector_summary.html
ARTIFACT_WINDOWS = {
    70: [
        (3215.0, 3221.0, "Lunar scattered light - Orbit 147b"),
    ],
    # Add more sectors as needed:
    # 71: [(start, end, "description"), ...],
}


def get_artifact_windows(sector: int) -> list:
    """
    Get list of artifact windows for a sector.

    Returns:
        List of (btjd_start, btjd_end, description) tuples
    """
    return ARTIFACT_WINDOWS.get(sector, [])


def is_in_artifact_window(btjd: float, sector: int) -> bool:
    """Check if a BTJD time is within any artifact window for the sector."""
    for start, end, _ in get_artifact_windows(sector):
        if start <= btjd <= end:
            return True
    return False


# ============================================================================
# TIC Catalog Query Settings
# ============================================================================

# Search radius in degrees for TIC cross-matching
TIC_SEARCH_RADIUS = 0.02

# Delay between TIC queries (seconds) to avoid rate limiting
TIC_QUERY_DELAY = 0.5

# ============================================================================
# Helper Functions
# ============================================================================

def ensure_directories():
    """Create all required output directories if they don't exist."""
    # Legacy directories
    for directory in [FITS_DIR, CALIBRATED_DATA_DIR, PHOTOMETRY_RESULTS_DIR,
                      LIGHTCURVES_DIR, MANIFEST_DIR]:
        directory.mkdir(parents=True, exist_ok=True)

    # New data structure
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    (DATA_DIR / "tess").mkdir(exist_ok=True)
    (DATA_DIR / "exports" / "features").mkdir(parents=True, exist_ok=True)
    (DATA_DIR / "exports" / "timeseries").mkdir(parents=True, exist_ok=True)


def ensure_tess_sector_dir(sector: int, camera: int, ccd: int) -> Path:
    """Create and return directory for TESS sector data."""
    path = get_tess_sector_dir(sector, camera, ccd)
    path.mkdir(parents=True, exist_ok=True)
    return path


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
