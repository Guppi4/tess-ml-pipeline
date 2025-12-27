"""
TESS FFI Data Processing Pipeline

A toolkit for downloading, calibrating, and analyzing TESS Full Frame Images
to detect stars and generate photometric lightcurves.
"""

__version__ = "1.0.0"

from . import config
from .FFIDownloader import download_fits
from .FFICalibrate import calibrate_background
from .FFIStarFinder import find_stars
from .FFILcCreator import create_lightcurve

__all__ = [
    "config",
    "download_fits",
    "calibrate_background",
    "find_stars",
    "create_lightcurve",
]
