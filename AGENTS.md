# AGENTS.md

This file guides Codex when working in this repo.

## Session workflow
- At session start: read `state.md` for current project status.
- During work: update `state.md` after significant tasks (branch/commits, what changed, next steps, issues).
- Before `/compact` or `/clear`: update `state.md` first.

## Project overview
TESS FFI Data Processing Pipeline: extract stellar photometry from TESS Full Frame Images, build lightcurves, find variable stars, export ML datasets. Streaming mode processes files on-the-fly without storing raw FITS (~10GB); keeps photometry results (~100MB). Config supports multi-satellite paths (TESS, Kepler, K2), but current tooling targets TESS.

## Core structure
```
src/tess/
  cli.py                 # CLI entry (tess-ffi)
  config.py              # Central configuration
  StreamingPipeline.py   # Main processor (download + photometry)
  FFIDownloader.py       # Download-only (legacy)
  FFICalibrate.py        # Background calibration (legacy)
  FFIStarFinder.py       # Star detection utilities
  StarCatalog.py         # Cross-epoch star matching
  LightcurveBuilder.py   # Lightcurve generation + stats
  VariableStarFinder.py  # Variable star detection + ranking
  MLExport.py            # ML feature extraction
  AsyncDownloader.py     # Optional async downloads (aiohttp/aiofiles)
  SectorAnalyzer.py      # Sector quality check before download
  TESS_FFI_App.py        # PyQt GUI (legacy path)
  main.py                # Interactive menu
```

## Repo layout
- `scripts/` utilities for TIC queries, VSX validation, cleaning plots, checkpoint sync
- `docs/VSX_PLOTTING_GUIDE.md` for VSX submission plot format
- `notebooks/` EDA and anomaly detection notebooks
- `data/` and `variable_stars/` hold generated outputs (mostly in .gitignore)

## Data flow (high level)
- MAST archive -> StreamingPipeline (download -> calibrate -> detect -> photometry -> delete raw FITS)
- Photometry CSVs -> LightcurveBuilder -> VariableStarFinder and/or MLExport

## CLI
Main workflow:
```
tess-ffi process --sector 70 --camera 1 --ccd 1
tess-ffi lightcurves --stats
tess-ffi find-variables --sector 70
tess-ffi export --name dataset1
```
Utilities:
```
tess-ffi status
tess-ffi catalog --sector 70
tess-ffi interactive
```
Deprecated:
```
tess-ffi fix-coords --sector 70 --camera 1 --ccd 1  # use process --add-tic instead
```
Legacy (raw FITS):
```
tess-ffi download --sector 70
tess-ffi calibrate
```

## Key functions and behaviors
- `run_streaming_pipeline(sector, camera, ccd, cadence_skip=1)` is the main entry point.
- `convert_to_starcatalog(sector, camera, ccd)` converts results for StarCatalog.
- `cadence_skip` trades time resolution for speed; resume warns if mismatch with checkpoint.
- `SectorAnalyzer.get_sector_quality()` can be used to check sector coverage before downloading.

Lightcurve metrics:
- Prefer `amplitude_robust` over `amplitude` (outlier sensitive).
- Check `reduced_chi2` (values < 1 often mean noise).

Variable stars:
- `load_and_build_lightcurves(min_snr=5.0)`
- `calculate_periodicity()` uses Lomb-Scargle
- `get_variable_candidates(min_amplitude=0.01)` for ranking

ML export:
- `export_for_ml(collection, name="...", min_completeness=0.6)` outputs features + padded time series.

## Configuration defaults (config.py)
- `DAOFIND_FWHM` 5.0
- `DAOFIND_THRESHOLD_SIGMA` 3.0
- `DEFAULT_APERTURE_RADIUS` 3.0
- `CROSSMATCH_TOLERANCE_ARCSEC` 3.0
- `ANNULUS_R_IN` 5.0
- `ANNULUS_R_OUT` 8.0
- `MIN_EPOCHS_FOR_STAR` 3
- `TIC_SEARCH_RADIUS` 0.02

## Output directories
```
data/tess/sector_XXX/camY_ccdZ/
  sXXX_Y-Z_photometry.csv
  sXXX_Y-Z_photometry_checkpoint.csv
data/exports/
  features/
  timeseries/
  sonification/
  vsx_submissions/
variable_stars/   # in .gitignore
  _overview/
  TIC_XXXXXXXXX/
lightcurves/
photometry_results/
```
Legacy `streaming_results/` is still supported.
Legacy paths also include `FITS/`, `calibrated_data/`, and `manifests/`.

## Important notes
- TIC matching uses 40 arcsec radius (higher match rate).
- Streaming mode auto-resumes; safe to interrupt.
- TESS WCS headers are unreliable; tess-point used for pixel->RA/Dec.
- Filter artifacts via `ARTIFACT_WINDOWS` in `config.py`.
- Trim sector edges: first ~0.5d and last ~1d can show trends.
- Async downloads require `aiohttp` and `aiofiles` (not in base deps).
