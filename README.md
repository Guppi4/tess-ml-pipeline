# TESS FFI Data Processing Pipeline

A complete toolkit for extracting stellar photometry from NASA's TESS Full Frame Images, building lightcurves, finding variable stars, and preparing data for machine learning.

## What is This For?

**TESS** (Transiting Exoplanet Survey Satellite) observes the sky in 27-day sectors, capturing **Full Frame Images (FFIs)** every 30 minutes (or 10 min in extended mission). Each FFI covers a 24° × 96° field containing thousands of stars.

This pipeline lets you:
- Extract brightness measurements (photometry) for all detected stars
- Track stars across hundreds of observations to build **lightcurves**
- Find **variable stars** (eclipsing binaries, pulsators, transiting planets)
- Export data for **machine learning** classification

## Quick Start (5 minutes)

```bash
# Install
git clone https://github.com/Guppi4/tess-ml-pipeline.git
cd tess-ml-pipeline
pip install -e .

# Process one CCD from sector 70 (downloads ~500 MB, not 10 GB!)
tess-ffi process --sector 70 --camera 1 --ccd 1

# See what stars you found
tess-ffi lightcurves --stats

# Find variable star candidates
tess-ffi find-variables --sector 70

# Export for machine learning
tess-ffi export --name my_dataset
```

## How It Works

```
                           TESS FFI Data Pipeline
    ═══════════════════════════════════════════════════════════════

    MAST Archive (NASA)
         │
         │  ~270 FITS files per CCD (~10 GB)
         │
         ▼
    ┌─────────────────────────────────────────────────────────────┐
    │                    tess-ffi process                         │
    │  ┌─────────┐   ┌─────────┐   ┌─────────┐   ┌─────────────┐ │
    │  │Download │ → │ Calibr. │ → │ Detect  │ → │ Photometry  │ │
    │  │ 1 file  │   │  (bkg)  │   │ stars   │   │ (aperture)  │ │
    │  └─────────┘   └─────────┘   └─────────┘   └─────────────┘ │
    │       │                                           │         │
    │       └──────── delete file ◄────────────────────┘         │
    │                 keep only results (~100 MB)                 │
    └─────────────────────────────────────────────────────────────┘
         │
         │  data/tess/sector_XXX/camY_ccdZ/*.csv
         │
         ▼
    ┌─────────────────────────────────────────────────────────────┐
    │                  tess-ffi lightcurves                       │
    │                                                             │
    │   Build time series for each star:                          │
    │                                                             │
    │   Flux ▲      ╭─╮                                          │
    │        │  ────╯   ╰───────  ← possible transit/eclipse     │
    │        └────────────────► Time (27 days)                   │
    │                                                             │
    │   Statistics: amplitude, chi², Stetson J, SNR              │
    └─────────────────────────────────────────────────────────────┘
         │
         ├──────────────────────────────┐
         ▼                              ▼
    ┌────────────────────┐    ┌────────────────────┐
    │ tess-ffi           │    │ tess-ffi export    │
    │ find-variables     │    │                    │
    │                    │    │ ML-ready formats:  │
    │ • Lomb-Scargle     │    │ • features.csv     │
    │ • TIC enrichment   │    │ • timeseries.npz   │
    │ • Ranking          │    │                    │
    └────────────────────┘    └────────────────────┘
```

## Command Reference

### Main Workflow (Recommended)

| Step | Command | What it does |
|------|---------|--------------|
| 1 | `tess-ffi process --sector 70 --camera 1 --ccd 1` | Download & process FFIs (streaming mode) |
| 2 | `tess-ffi lightcurves --stats` | Build lightcurves, show statistics |
| 3 | `tess-ffi find-variables --sector 70` | Find variable star candidates |
| 4 | `tess-ffi export --name dataset1` | Export for machine learning |

### All Commands

```bash
# Data acquisition
tess-ffi process     # Streaming: download + process + delete (recommended)
tess-ffi download    # Download only (keeps ~10 GB of FITS files)

# Analysis
tess-ffi catalog     # Build star catalog with cross-matching
tess-ffi lightcurves # View lightcurves and statistics
tess-ffi find-variables  # Find variable stars

# Export
tess-ffi export      # Export features and time series for ML

# Utilities
tess-ffi status      # Show what data exists
tess-ffi calibrate   # Background calibration (for legacy download path)
tess-ffi interactive # Interactive menu
```

### Common Options

```bash
# Process with more parallel downloads (faster)
tess-ffi process --sector 70 --workers 10

# Resume interrupted processing
tess-ffi process --sector 70  # auto-resumes by default

# Add TIC IDs during processing (slower, queries MAST)
tess-ffi process --sector 70 --add-tic

# View specific star's lightcurve
tess-ffi lightcurves --star STAR_000123

# Export only high-quality data
tess-ffi export --name clean_data --min-completeness 0.8 --min-snr 10
```

### cadence_skip - Faster Processing

TESS sector 70 has ~10,730 files at 200-second cadence (18 images/hour). Use `cadence_skip` to process every Nth file for faster results:

```python
from tess.StreamingPipeline import run_streaming_pipeline

# All files (default) - best accuracy
run_streaming_pipeline(sector=70, camera="1", ccd="1", cadence_skip=1)

# Every 6th file - 3 observations per hour (~1,789 files)
run_streaming_pipeline(sector=70, camera="1", ccd="1", cadence_skip=6)

# Every 9th file - 2 observations per hour (~1,193 files)
run_streaming_pipeline(sector=70, camera="1", ccd="1", cadence_skip=9)

# Every 18th file - 1 observation per hour (~596 files)
run_streaming_pipeline(sector=70, camera="1", ccd="1", cadence_skip=18)
```

| cadence_skip | Files | Time resolution | Use case |
|--------------|-------|-----------------|----------|
| 1 | ~10,730 | 200 sec | Full precision |
| 6 | ~1,789 | 20 min | Good for most variables |
| 9 | ~1,193 | 30 min | Quick survey |
| 18 | ~596 | 1 hour | Initial exploration |

## Understanding the Output

### Directory Structure

```
project/
├── data/
│   ├── tess/
│   │   └── sector_070/
│   │       └── cam1_ccd1/
│   │           ├── s0070_1-1_photometry.csv         # All measurements
│   │           └── s0070_1-1_photometry_checkpoint.csv  # Resume data
│   │
│   └── exports/                    # Machine learning exports
│       ├── features/*.csv              # Statistical features
│       └── timeseries/*.npz            # Padded sequences
│
├── variable_stars/             # Variable star analysis
│   ├── s0070_1-1_candidates.csv    # Top candidates
│   ├── s0070_1-1_all_stars.csv     # All analyzed stars
│   └── plots/                      # Lightcurve plots
│
└── lightcurves/                # Saved plots
```

**Note:** Legacy `streaming_results/` directory is still supported for backwards compatibility.

### Photometry File Columns

| Column | Description | Units |
|--------|-------------|-------|
| `star_id` | Unique identifier | STAR_XXXXXX |
| `epoch` | Observation index | 0, 1, 2, ... |
| `mjd` | Modified Julian Date | days |
| `flux` | Brightness | counts/sec |
| `flux_error` | Uncertainty | counts/sec |
| `quality` | Quality flag | 0=good, 1=edge, 2=negative, 3=outside |
| `x`, `y` | Pixel position | pixels |

### Quality Flags Explained

| Flag | Meaning | Should you use it? |
|------|---------|-------------------|
| 0 | Good measurement | ✓ Yes |
| 1 | Near CCD edge | ⚠ Use with caution |
| 2 | Negative flux (noise) | ✗ Exclude |
| 3 | Star outside image | ✗ Exclude |

### Variability Metrics

| Metric | What it measures | High value means |
|--------|-----------------|------------------|
| `amplitude` | (max-min)/median | Large brightness changes |
| `amplitude_robust` | (p95-p5)/median | Changes ignoring outliers |
| `reduced_chi2` | Scatter vs errors | Real variability (if >1) |
| `stetson_j` | Correlated changes | Consistent variability |
| `mad` | Median abs. deviation | Scatter from median |

## Two Processing Paths

### Path 1: Streaming (Recommended)

Best for most users. Processes data on-the-fly, uses only ~500 MB disk space.

```bash
tess-ffi process --sector 70 --camera 1 --ccd 1
# Results in data/tess/sector_070/cam1_ccd1/
```

### Path 2: Legacy (Full Download)

Use this if you need the raw FITS files for custom analysis.

```bash
tess-ffi download --sector 70 --camera 1 --ccd 1  # ~10 GB
tess-ffi calibrate                                 # Background estimation
tess-ffi catalog --source calibrated               # Star detection + photometry
```

## Python API

```python
from tess.StreamingPipeline import run_streaming_pipeline, convert_to_starcatalog
from tess.LightcurveBuilder import LightcurveCollection
from tess.VariableStarFinder import VariableStarFinder
from tess.MLExport import export_for_ml

# Process sector (with optional cadence_skip for faster processing)
run_streaming_pipeline(sector=70, camera="1", ccd="1", workers=10, cadence_skip=6)

# Build lightcurves
catalog = convert_to_starcatalog(70, "1", "1")
collection = LightcurveCollection(catalog)

# Find variables
finder = VariableStarFinder(sector=70)
finder.load_and_build_lightcurves(min_snr=5.0)
finder.calculate_periodicity()
candidates = finder.get_variable_candidates(min_amplitude=0.01)

# Export for ML
export_for_ml(collection, name="sector70", min_completeness=0.6)
```

## Science Background

### How Aperture Photometry Works

```
Raw CCD Image:              Aperture Extraction:

  ░░░░░░░░░░░░░               radius = 3 pixels
  ░░░░░░░░░░░░░                   ╭───────╮
  ░░░░▓█▓░░░░░░               ╭───│ · · · │───╮
  ░░░▓███▓░░░░░               │ · │ ·███· │ · │  Sum all pixels
  ░░░░▓█▓░░░░░░     →         │ · │ ██★██ │ · │  inside circle
  ░░░░░░░░░░░░░               │ · │ ·███· │ · │  = aperture_sum
  ░░░░░░░░░░░░░               ╰───│ · · · │───╯
                                  ╰───────╯
```

1. **Background subtraction**: Remove sky glow using sigma-clipped statistics
2. **Star detection**: DAOStarFinder algorithm (Stetson 1987) finds star-like sources
3. **Aperture photometry**: Sum pixel values in circular aperture around each star
4. **Error estimation**: Combine Poisson noise + background noise + read noise

### What Can You Find?

- **Eclipsing binaries**: Periodic dips when one star blocks another
- **Pulsating stars**: Regular brightness oscillations (Cepheids, RR Lyrae, δ Scuti)
- **Exoplanet transits**: Tiny dips when a planet crosses its star
- **Stellar flares**: Sudden brightness increases (especially M dwarfs)
- **Rotational modulation**: Starspots causing periodic variations

## Configuration

Key parameters in `src/tess/config.py`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `DAOFIND_FWHM` | 5.0 px | Expected star width (TESS PSF) |
| `DAOFIND_THRESHOLD_SIGMA` | 3.0 | Detection significance |
| `DEFAULT_APERTURE_RADIUS` | 3.0 px | Photometry aperture size |
| `CROSSMATCH_TOLERANCE_ARCSEC` | 3.0" | Star matching tolerance |
| `MIN_EPOCHS_FOR_STAR` | 3 | Minimum detections to keep star |

## Requirements

- Python 3.8+
- numpy, pandas, matplotlib, scipy
- astropy, photutils, astroquery
- requests, beautifulsoup4, tqdm
- Optional: tess-point (better coordinate conversion)
- Optional: PyQt5 (GUI)

## References

- TESS Mission: https://tess.mit.edu/
- MAST Archive: https://archive.stsci.edu/tess/
- DAOStarFinder: Stetson (1987), PASP 99, 191
- TIC Catalog: Stassun et al. (2019), AJ 158, 138

## License

MIT License
