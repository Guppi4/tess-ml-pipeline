"""
TESS FFI Data Processing Pipeline

A toolkit for downloading, calibrating, and analyzing TESS Full Frame Images
to detect stars and generate photometric lightcurves.
"""

__version__ = "1.0.0"

# Core modules - import on demand to avoid circular imports
# Use: from tess.StreamingPipeline import run_streaming_pipeline

__all__ = [
    "StreamingPipeline",
    "LightcurveBuilder",
    "MLExport",
    "VariableStarFinder",
    "StarCatalog",
    "FFIDownloader",
    "FFICalibrate",
    "FFIStarFinder",
    "config",
]
