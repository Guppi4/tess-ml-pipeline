# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Session Management

**At session START:** Read `state.md` to understand current project state.

**During session:** Update `state.md` periodically after completing significant tasks:
- Current branch and recent commits
- What was done
- Next steps / pending tasks
- Any known issues

**On `/compact` or `/clear`:** Update `state.md` first, then compact.

PreCompact hook configured in `.claude/settings.local.json` will remind about this.

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
├── DataCleaner.py         # Data cleaning with masks (NEW)
├── CVZPipeline.py         # DEPRECATED - use StreamingPipeline for CVZ
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
│  (supports cadence_skip for faster processing)         │
└────────────────────────────────────────────────────────┘
     │
     ▼  data/tess/sector_XXX/camY_ccdZ/photometry.parquet
     │
┌────────────────────────────────────────────────────────┐
│              DataCleaner.py (NEW)                       │
│  • epoch_qc: bad epoch detection                       │
│  • star_qc: per-star quality metrics                   │
│  • Common-mode correction (flux_cm)                    │
│  • Mask-based cleaning (preserves all data)            │
└────────────────────────────────────────────────────────┘
     │
     ▼  cleaned/photometry_with_masks.parquet
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
     │
     ▼  variable_stars/sector_XXX/
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
- `run_streaming_pipeline(sector, camera, ccd, cadence_skip=1)` - Main entry point
- `convert_to_starcatalog(sector, camera, ccd)` - Convert results to StarCatalog format

**cadence_skip parameter:**
Controls how many files to skip for faster processing. TESS sector 70 has ~10,730 files at 200-second cadence (18 images/hour).

| cadence_skip | Files processed | Time resolution |
|--------------|-----------------|-----------------|
| 1 (default)  | All (~10,730)   | 200 sec (~18/hour) |
| 6            | ~1,789          | 20 min (3/hour) |
| 9            | ~1,193          | 30 min (2/hour) |
| 18           | ~596            | 1 hour (1/hour) |

**Note:** If you resume with a different cadence_skip than before, you'll get a warning. The checkpoint stores the original cadence_skip value.

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

### DataCleaner.py
Comprehensive data cleaning with mask-based approach (preserves all data).

**Key features:**
- `compute_epoch_qc()` - Detect bad epochs (high scatter, low completeness)
- `compute_star_qc()` - Per-star quality metrics (completeness, MAD, edge flags)
- `find_quiet_stars()` - Find stable stars for common-mode calculation
- `compute_common_mode()` - Calculate systematic trends from quiet stars
- `create_masks()` - Apply all quality filters as bit masks
- `export_for_ml()` - Export cleaned data for ML (anomaly/classification modes)

**Mask bits (uint16):**
| Bit | Name | Description |
|-----|------|-------------|
| 0 | quality | TESS quality flag |
| 1 | invalid | NaN/negative flux |
| 2 | bad_epoch | High scatter epoch |
| 3 | outlier_pos | Positive outlier (5σ) |
| 4 | outlier_neg | Negative outlier (10σ) |
| 5 | edge | Star near CCD edge |
| 6 | artifact | Known artifact window |
| 7 | low_snr | Low signal-to-noise |

**Usage:**
```python
from tess.DataCleaner import clean_sector
clean_sector(61, '4', '2')  # Creates qc/, cleaned/, ml/ folders
```

### VariableStarFinder.py
Finds and ranks variable stars using multiple metrics. Saves results to `variable_stars/sector_XXX/`.

**Key functions:**
- `load_and_build_lightcurves(min_snr=5.0)` - Load and filter data
- `calculate_periodicity()` - Lomb-Scargle periodogram
- `get_variable_candidates(min_amplitude=0.01)` - Get ranked candidates
- `cross_match_vsx()` - Check against AAVSO Variable Star Index

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
data/
├── tess/
│   └── sector_XXX/
│       └── camY_ccdZ/
│           ├── sXXX_Y-Z_photometry.parquet   # Raw photometry
│           ├── sXXX_Y-Z_catalog.json         # Star catalog with TIC IDs
│           ├── qc/                           # Quality control (DataCleaner)
│           │   ├── epoch_qc.parquet
│           │   ├── star_qc.parquet
│           │   └── bad_epochs.json
│           ├── cleaned/                      # Cleaned data
│           │   └── photometry_with_masks.parquet
│           └── ml/                           # ML exports
│               ├── ml_anomaly.parquet
│               └── ml_classification.parquet
├── exports/
│   ├── features/       # ML features (CSV, NPY)
│   └── timeseries/     # Padded sequences (NPZ)

variable_stars/                    # Variable star analysis (per sector)
├── sector_061/
│   ├── plots/                     # Overview plots (top 20)
│   ├── sXXX_Y-Z_variable_candidates.csv
│   ├── sXXX_Y-Z_all_stars.csv
│   └── TIC_XXXXXXXXX/             # Per-star detailed analysis
│       ├── VSX_rawlc.png          # For VSX submission
│       ├── VSX_phase.png
│       └── period_check_2x.png
└── sector_070/
    └── ...
```

**Note:** Legacy `streaming_results/` directory is still supported for backwards compatibility.

## Common Tasks

### Process a new sector
```python
from tess.StreamingPipeline import run_streaming_pipeline

# Full resolution (all files, ~10,730 for sector 70)
run_streaming_pipeline(sector=70, camera="1", ccd="1", workers=5)

# Faster processing with cadence_skip (every 6th file = 3 observations/hour)
run_streaming_pipeline(sector=70, camera="1", ccd="1", workers=10, cadence_skip=6)
```

### Clean data (after processing)
```python
from tess.DataCleaner import clean_sector

# Run full cleaning pipeline: epoch_qc, star_qc, common-mode, masks, ML export
clean_sector(61, '4', '2')

# Or step by step:
from tess.DataCleaner import DataCleaner
cleaner = DataCleaner(61, '4', '2')
cleaner.load_data()
cleaner.compute_epoch_qc()
cleaner.compute_star_qc()
cleaner.find_quiet_stars()
cleaner.compute_common_mode()
cleaner.create_masks()
cleaner.export_for_ml(mode='classification')
```

### Find variable stars
```python
from tess.VariableStarFinder import VariableStarFinder
finder = VariableStarFinder(sector=61, camera='4', ccd='2')
finder.load_and_build_lightcurves(min_snr=5.0)
finder.calculate_periodicity()
finder.calculate_variability_scores()
finder.save_results()  # Saves to variable_stars/sector_061/
finder.plot_top_candidates(n_plots=20)
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

## Notebooks

| Notebook | Purpose |
|----------|---------|
| `notebooks/ml_variable_stars.ipynb` | ML analysis: anomaly detection, clustering, t-SNE/UMAP, classification baseline |
| `notebooks/anomaly_detection.ipynb` | Periodogram analysis, BLS transit search (sector 70) |
| `notebooks/eda_tess.ipynb` | Exploratory data analysis |

### ml_variable_stars.ipynb

Main ML notebook for sector 61 data. Sections:
1. Data Loading — loads `ml/ml_classification.parquet`
2. Feature Engineering — RobustScaler, correlation analysis
3. Anomaly Detection — Isolation Forest, LOF, Mahalanobis (consensus voting)
4. Clustering — K-Means (elbow/silhouette), DBSCAN
5. Dimensionality Reduction — t-SNE, UMAP visualization
6. Classification Baseline — pseudo-labels, Random Forest, XGBoost
7. Feature Importance — RF importance, SHAP values
8. Results Summary — statistics, top anomalies table
9. Anomaly Viewer — lightcurves for top-N anomalies
10. Export — ml_results.csv, trained models

**Note:** Classification uses pseudo-labels derived from features (not real labels). For real classification, cross-match with VSX or add manual labels.

## Important Notes

- **Use `amplitude_robust` instead of `amplitude`** - The simple amplitude is extremely sensitive to single outliers
- **Check `reduced_chi2`** - If chi² < 1, the "variability" is likely just noise
- **TIC matching uses 40 arcsec radius** - Increased from default to improve match rate
- **Streaming mode auto-resumes** - Safe to interrupt with Ctrl+C
- **TESS WCS headers are unreliable** - Pipeline uses tess-point library for pixel→RA/Dec
- **Filter artifacts before analysis** - Use `ARTIFACT_WINDOWS` in config.py (e.g., lunar light in sector 70: BTJD 3215-3221)
- **Trim sector edges** - First ~0.5d and last ~1d often have instrumental trends

## References

- TESS Mission: https://tess.mit.edu/
- MAST Archive: https://archive.stsci.edu/tess/
- DAOStarFinder: Stetson (1987), PASP 99, 191
- TIC Catalog: Stassun et al. (2019), AJ 158, 138
