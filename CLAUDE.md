# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**TESS FFI Data Processing Pipeline** - Extract stellar photometry from NASA's TESS Full Frame Images, build lightcurves, find variable stars, and export for machine learning.

### Key Innovation: Streaming Mode

The pipeline processes TESS data **on-the-fly** without storing raw FITS files (~10GB). Only photometry results are kept (~100MB), making it practical for any computer.

## Installation

```bash
pip install -e .           # Basic install
pip install -e ".[gui]"    # With PyQt5 GUI
pip install -e ".[dev]"    # With testing tools
```

## Architecture

```
src/tess/
├── cli.py                 # Command-line interface (tess-ffi)
├── config.py              # Central configuration
├── StreamingPipeline.py   # Main processor (download + photometry)
├── FFIDownloader.py       # Download-only (legacy path)
├── FFICalibrate.py        # Background calibration (legacy path)
├── FFIStarFinder.py       # Star detection utilities
├── StarCatalog.py         # Cross-epoch star matching
├── LightcurveBuilder.py   # Lightcurve generation + statistics
├── VariableStarFinder.py  # Variable star detection + ranking
├── MLExport.py            # ML feature extraction
└── main.py                # Interactive menu
```

## Data Flow

```
MAST Archive
     │
     ▼
┌────────────────────────────────────────────────────────┐
│              StreamingPipeline.py                       │
│  for each FITS file:                                   │
│    download → calibrate → detect → photometry → delete │
└────────────────────────────────────────────────────────┘
     │
     ▼  streaming_results/*.csv
     │
┌────────────────────────────────────────────────────────┐
│              LightcurveBuilder.py                       │
│  • Build time series per star                          │
│  • Calculate: amplitude, chi², stetson_j, MAD, SNR     │
└────────────────────────────────────────────────────────┘
     │
     ├────────────────────────┐
     ▼                        ▼
┌─────────────────┐    ┌─────────────────┐
│ VariableStarFinder │ │    MLExport     │
│ • Lomb-Scargle    │ │ • features.csv  │
│ • TIC enrichment  │ │ • timeseries.npz│
│ • Ranking         │ │                 │
└─────────────────┘    └─────────────────┘
```

## CLI Commands

```bash
# Main workflow (recommended)
tess-ffi process --sector 70 --camera 1 --ccd 1   # Download + process
tess-ffi lightcurves --stats                       # View statistics
tess-ffi find-variables --sector 70                # Find variables
tess-ffi export --name dataset1                    # Export for ML

# Utilities
tess-ffi status                                    # Show data status
tess-ffi catalog --sector 70                       # Build star catalog
tess-ffi interactive                               # Interactive menu

# Legacy path (if raw FITS files needed)
tess-ffi download --sector 70                      # Download only (~10GB)
tess-ffi calibrate                                 # Background calibration
```

## Key Modules

### StreamingPipeline.py
Main processing engine. Downloads FFI files one at a time, extracts photometry, deletes raw files. Supports parallel downloads, checkpointing, and resume.

**Key functions:**
- `run_streaming_pipeline(sector, camera, ccd)` - Main entry point
- `convert_to_starcatalog(sector, camera, ccd)` - Convert results to StarCatalog format

### LightcurveBuilder.py
Builds time series for each star and calculates variability metrics.

**Key classes:**
- `Lightcurve` - Single star's time series with statistics
- `LightcurveCollection` - Container for all lightcurves

**Key metrics:**
- `amplitude` - (max-min)/median (sensitive to outliers!)
- `amplitude_robust` - (p95-p5)/median (use this instead)
- `reduced_chi2` - Variability significance (>1 means real variability)
- `stetson_j` - Correlated variability index
- `outlier_fraction` - Fraction of 3σ outliers

### VariableStarFinder.py
Finds and ranks variable stars using multiple metrics.

**Key functions:**
- `load_and_build_lightcurves(min_snr=5.0)` - Load and filter data
- `calculate_periodicity()` - Lomb-Scargle periodogram
- `get_variable_candidates(min_amplitude=0.01)` - Get ranked candidates

### MLExport.py
Extracts features for machine learning classification.

**Statistical features:** mean, std, skewness, kurtosis, MAD, IQR, percentiles, amplitude, beyond_Nsigma

**Periodic features:** Top N frequencies/periods from Lomb-Scargle, power ratios

**Time series export:** Padded arrays shape (n_samples, max_length, 3) for LSTM/Transformer

## Configuration (config.py)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `DAOFIND_FWHM` | 5.0 | Star detection FWHM (pixels) |
| `DAOFIND_THRESHOLD_SIGMA` | 3.0 | Detection threshold |
| `DEFAULT_APERTURE_RADIUS` | 3.0 | Photometry aperture (pixels) |
| `CROSSMATCH_TOLERANCE_ARCSEC` | 3.0 | Star matching tolerance |
| `ANNULUS_R_IN` | 6.0 | Background annulus inner radius |
| `ANNULUS_R_OUT` | 10.0 | Background annulus outer radius |

## Quality Flags

| Flag | Meaning |
|------|---------|
| 0 | Good measurement |
| 1 | Near image edge |
| 2 | Negative flux (noise) |
| 3 | Outside image bounds |

## Output Directories

```
streaming_results/     # Main photometry output
variable_stars/        # Variable star analysis
ml_data/features/      # ML features (CSV, NPY)
ml_data/timeseries/    # Padded sequences (NPZ)
lightcurves/           # Saved plots
photometry_results/    # Converted StarCatalog format
```

## Common Tasks

### Process a new sector
```python
from tess.StreamingPipeline import run_streaming_pipeline
run_streaming_pipeline(sector=70, camera="1", ccd="1", workers=5)
```

### Find variable stars
```python
from tess.VariableStarFinder import VariableStarFinder
finder = VariableStarFinder(sector=70)
finder.load_and_build_lightcurves(min_snr=5.0)
finder.calculate_periodicity()
candidates = finder.get_variable_candidates(min_amplitude=0.01)
```

### Export for ML
```python
from tess.StreamingPipeline import convert_to_starcatalog
from tess.LightcurveBuilder import LightcurveCollection
from tess.MLExport import export_for_ml

catalog = convert_to_starcatalog(70, "1", "1")
collection = LightcurveCollection(catalog)
export_for_ml(collection, name="sector70", min_completeness=0.6)
```

## Important Notes

- **Use `amplitude_robust` instead of `amplitude`** - The simple amplitude is extremely sensitive to single outliers
- **Check `reduced_chi2`** - If chi² < 1, the "variability" is likely just noise
- **TIC matching uses 40 arcsec radius** - Increased from default to improve match rate
- **Streaming mode auto-resumes** - Safe to interrupt with Ctrl+C
- **TESS WCS headers are unreliable** - Pipeline uses tess-point library for pixel→RA/Dec

## References

- TESS Mission: https://tess.mit.edu/
- MAST Archive: https://archive.stsci.edu/tess/
- DAOStarFinder: Stetson (1987), PASP 99, 191
- TIC Catalog: Stassun et al. (2019), AJ 158, 138
