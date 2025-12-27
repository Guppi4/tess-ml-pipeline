"""
TESS FFI Pipeline - Command Line Interface

Usage:
    tess-ffi <command> [options]

Commands:
    download    Download FFI data from MAST
    process     Run streaming pipeline (recommended)
    calibrate   Calibrate downloaded FITS files
    catalog     Build star catalog
    lightcurves Generate lightcurves
    export      Export data for ML
    status      Show current data status
    interactive Launch interactive menu

Examples:
    tess-ffi download --sector 70
    tess-ffi process --sector 70 --camera 1 --ccd 1
    tess-ffi lightcurves --star STAR_000123
    tess-ffi export --min-snr 5 --format features
"""

import argparse
import sys
import json
from pathlib import Path


def create_parser():
    """Create argument parser with all commands."""
    parser = argparse.ArgumentParser(
        prog='tess-ffi',
        description='TESS FFI Data Processing Pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s download --sector 70
  %(prog)s process --sector 70 --camera 1 --ccd 1
  %(prog)s status
  %(prog)s interactive
        """
    )

    subparsers = parser.add_subparsers(dest='command', help='Available commands')

    # =========================================================================
    # DOWNLOAD command
    # =========================================================================
    download_parser = subparsers.add_parser(
        'download',
        help='Download FFI data from MAST archive'
    )
    download_parser.add_argument(
        '--sector', '-s',
        type=int,
        required=True,
        help='TESS sector number (1-98)'
    )
    download_parser.add_argument(
        '--camera', '-c',
        type=str,
        default='1',
        help='Camera number (1-4, default: 1)'
    )
    download_parser.add_argument(
        '--ccd',
        type=str,
        default='1',
        help='CCD number (1-4, default: 1)'
    )
    download_parser.add_argument(
        '--list-sectors',
        action='store_true',
        help='List available sectors and exit'
    )
    download_parser.add_argument(
        '--info',
        action='store_true',
        help='Show sector info without downloading'
    )

    # =========================================================================
    # PROCESS command (streaming pipeline)
    # =========================================================================
    process_parser = subparsers.add_parser(
        'process',
        help='Run streaming pipeline (download + process on-the-fly)'
    )
    process_parser.add_argument(
        '--sector', '-s',
        type=int,
        required=True,
        help='TESS sector number'
    )
    process_parser.add_argument(
        '--camera', '-c',
        type=str,
        default='1',
        help='Camera number (1-4, default: 1)'
    )
    process_parser.add_argument(
        '--ccd',
        type=str,
        default='1',
        help='CCD number (1-4, default: 1)'
    )
    process_parser.add_argument(
        '--workers', '-w',
        type=int,
        default=5,
        help='Number of parallel download workers (default: 5)'
    )
    process_parser.add_argument(
        '--no-resume',
        action='store_true',
        help='Start fresh, ignore checkpoint'
    )
    process_parser.add_argument(
        '--check-quality',
        action='store_true',
        help='Check sector quality before processing'
    )
    process_parser.add_argument(
        '--add-tic',
        action='store_true',
        help='Add TIC IDs after processing (slow, ~1 hour for 2000 stars)'
    )
    process_parser.add_argument(
        '--no-async',
        action='store_true',
        help='Disable async downloads (use if aiohttp not available)'
    )
    process_parser.add_argument(
        '--download-workers',
        type=int,
        default=10,
        help='Concurrent downloads in async mode (default: 10)'
    )
    process_parser.add_argument(
        '--batch-size',
        type=int,
        default=50,
        help='Files per batch in async mode (default: 50)'
    )

    # =========================================================================
    # CALIBRATE command
    # =========================================================================
    calibrate_parser = subparsers.add_parser(
        'calibrate',
        help='Calibrate downloaded FITS files'
    )
    calibrate_parser.add_argument(
        '--force',
        action='store_true',
        help='Re-calibrate even if already done'
    )

    # =========================================================================
    # CATALOG command
    # =========================================================================
    catalog_parser = subparsers.add_parser(
        'catalog',
        help='Build star catalog with cross-matching'
    )
    catalog_parser.add_argument(
        '--sector', '-s',
        type=int,
        help='Sector number (for streaming data)'
    )
    catalog_parser.add_argument(
        '--camera', '-c',
        type=str,
        default='1',
        help='Camera number'
    )
    catalog_parser.add_argument(
        '--ccd',
        type=str,
        default='1',
        help='CCD number'
    )
    catalog_parser.add_argument(
        '--add-tic',
        action='store_true',
        help='Add TIC IDs (slow, ~2 hours)'
    )
    catalog_parser.add_argument(
        '--source',
        choices=['streaming', 'calibrated', 'auto'],
        default='auto',
        help='Data source (default: auto-detect)'
    )

    # =========================================================================
    # LIGHTCURVES command
    # =========================================================================
    lc_parser = subparsers.add_parser(
        'lightcurves',
        help='Generate and view lightcurves'
    )
    lc_parser.add_argument(
        '--star',
        type=str,
        help='Specific star ID (e.g., STAR_000123)'
    )
    lc_parser.add_argument(
        '--ra',
        type=float,
        help='RA coordinate to find nearest star'
    )
    lc_parser.add_argument(
        '--dec',
        type=float,
        help='Dec coordinate to find nearest star'
    )
    lc_parser.add_argument(
        '--save-plots',
        action='store_true',
        help='Save lightcurve plots'
    )
    lc_parser.add_argument(
        '--max-plots',
        type=int,
        default=100,
        help='Maximum number of plots to save (default: 100)'
    )
    lc_parser.add_argument(
        '--stats',
        action='store_true',
        help='Show statistics table'
    )
    lc_parser.add_argument(
        '--export-stats',
        type=str,
        metavar='FILE',
        help='Export statistics to CSV file'
    )

    # =========================================================================
    # EXPORT command (ML)
    # =========================================================================
    export_parser = subparsers.add_parser(
        'export',
        help='Export data for machine learning'
    )
    export_parser.add_argument(
        '--name',
        type=str,
        default='tess_dataset',
        help='Dataset name (default: tess_dataset)'
    )
    export_parser.add_argument(
        '--min-completeness',
        type=float,
        default=0.5,
        help='Minimum data completeness 0-1 (default: 0.5)'
    )
    export_parser.add_argument(
        '--min-snr',
        type=float,
        default=3.0,
        help='Minimum signal-to-noise ratio (default: 3.0)'
    )
    export_parser.add_argument(
        '--format',
        choices=['all', 'features', 'timeseries'],
        default='all',
        help='Export format (default: all)'
    )
    export_parser.add_argument(
        '--max-length',
        type=int,
        default=200,
        help='Max sequence length for timeseries (default: 200)'
    )

    # =========================================================================
    # FIX-COORDS command [DEPRECATED]
    # =========================================================================
    fix_parser = subparsers.add_parser(
        'fix-coords',
        help='[DEPRECATED] Add TIC IDs to existing data. Use --add-tic with process instead.'
    )
    fix_parser.add_argument(
        '--sector', '-s',
        type=int,
        help='Sector number (auto-detect if not specified)'
    )
    fix_parser.add_argument(
        '--camera', '-c',
        type=str,
        default='1',
        help='Camera number'
    )
    fix_parser.add_argument(
        '--ccd',
        type=str,
        default='1',
        help='CCD number'
    )
    fix_parser.add_argument(
        '--add-tic',
        action='store_true',
        help='Also add TIC IDs (slow)'
    )
    fix_parser.add_argument(
        '--force',
        action='store_true',
        help='Force re-conversion even if coordinates look valid'
    )

    # =========================================================================
    # FIND-VARIABLES command
    # =========================================================================
    var_parser = subparsers.add_parser(
        'find-variables',
        help='Find variable stars in photometry data'
    )
    var_parser.add_argument(
        '--sector', '-s',
        type=int,
        required=True,
        help='TESS sector number'
    )
    var_parser.add_argument(
        '--camera', '-c',
        type=str,
        default='1',
        help='Camera number (1-4, default: 1)'
    )
    var_parser.add_argument(
        '--ccd',
        type=str,
        default='1',
        help='CCD number (1-4, default: 1)'
    )
    var_parser.add_argument(
        '--min-amplitude',
        type=float,
        default=0.01,
        help='Minimum variability amplitude (default: 0.01 = 1%%)'
    )
    var_parser.add_argument(
        '--min-chi2',
        type=float,
        default=2.0,
        help='Minimum reduced chi-squared (default: 2.0)'
    )
    var_parser.add_argument(
        '--min-completeness',
        type=float,
        default=0.5,
        help='Minimum data completeness 0-1 (default: 0.5)'
    )
    var_parser.add_argument(
        '--min-snr',
        type=float,
        default=5.0,
        help='Minimum signal-to-noise ratio (default: 5.0)'
    )
    var_parser.add_argument(
        '--top-n',
        type=int,
        default=100,
        help='Number of top candidates (default: 100)'
    )
    var_parser.add_argument(
        '--enrich-tic',
        action='store_true',
        help='Query TIC for stellar parameters (slow, ~8 min)'
    )
    var_parser.add_argument(
        '--plot',
        action='store_true',
        help='Generate lightcurve plots for top candidates'
    )
    var_parser.add_argument(
        '--output-dir',
        type=str,
        help='Output directory (default: variable_stars/)'
    )

    # =========================================================================
    # STATUS command
    # =========================================================================
    status_parser = subparsers.add_parser(
        'status',
        help='Show current data status'
    )
    status_parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Show detailed information'
    )

    # =========================================================================
    # INTERACTIVE command
    # =========================================================================
    interactive_parser = subparsers.add_parser(
        'interactive',
        help='Launch interactive menu'
    )

    return parser


# =============================================================================
# Command handlers
# =============================================================================

def cmd_download(args):
    """Handle download command."""
    from . import FFIDownloader as dffi

    if args.list_sectors:
        print("Fetching available sectors from MAST...")
        sectors = dffi.list_available_sectors()
        if sectors:
            print(f"\nAvailable sectors: {len(sectors)}")
            print(f"Range: {min(sectors)} to {max(sectors)}")
            recent = sorted(sectors)[-10:]
            print(f"Most recent: {', '.join(map(str, recent))}")
        return

    if args.info:
        print(f"\nFetching info for sector {args.sector}...")
        info = dffi.get_sector_info(args.sector)
        if info['available']:
            print(f"  Status: Available")
            print(f"  Date range: {info.get('date_start', 'N/A')} to {info.get('date_end', 'N/A')}")
            print(f"  Total days: {info['total_days']}")
            print(f"  Expected files: ~{info['total_days'] * 10}")
        else:
            print(f"  Status: Not available")
        return

    # Download
    print(f"\nDownloading Sector {args.sector}, Camera {args.camera}, CCD {args.ccd}")
    dffi.download_sector(args.sector, args.camera, args.ccd)


def cmd_process(args):
    """Handle process (streaming) command."""
    from .StreamingPipeline import run_streaming_pipeline
    from .SectorAnalyzer import quick_check

    if args.check_quality:
        print(f"\nChecking sector {args.sector} quality...")
        quality = quick_check(args.sector)
        print(f"  Quality: {quality.quality_score}")
        print(f"  Coverage: {quality.coverage_percent:.1f}%")

        if quality.quality_score in ['poor', 'bad']:
            response = input("\nSector has significant gaps. Continue? (y/n): ").strip().lower()
            if response != 'y':
                print("Aborted.")
                return

    async_mode = not args.no_async
    print(f"\nStarting streaming pipeline...")
    print(f"  Sector: {args.sector}")
    print(f"  Camera: {args.camera}")
    print(f"  CCD: {args.ccd}")
    print(f"  Workers: {args.workers}")
    print(f"  Resume: {not args.no_resume}")
    print(f"  Add TIC: {args.add_tic}")
    print(f"  Async mode: {async_mode}" + (f" ({args.download_workers} downloads, batch={args.batch_size})" if async_mode else ""))

    run_streaming_pipeline(
        sector=args.sector,
        camera=args.camera,
        ccd=args.ccd,
        resume=not args.no_resume,
        workers=args.workers,
        add_tic=args.add_tic,
        async_mode=async_mode,
        download_workers=args.download_workers,
        batch_size=args.batch_size
    )


def cmd_calibrate(args):
    """Handle calibrate command."""
    from . import FFICalibrate as cffi
    from .config import FITS_DIR, CALIBRATED_DATA_DIR

    fits_files = sorted(FITS_DIR.glob('*.fits'))

    if not fits_files:
        print(f"No FITS files found in {FITS_DIR}")
        print("Use 'tess-ffi download' first or use streaming mode.")
        return

    print(f"Found {len(fits_files)} FITS files")

    # Check existing
    metadata_path = CALIBRATED_DATA_DIR / 'calibration_metadata.json'
    if metadata_path.exists() and not args.force:
        with open(metadata_path, 'r') as f:
            meta = json.load(f)
        print(f"\nExisting calibration found ({meta['n_epochs']} epochs)")
        print("Use --force to re-calibrate")
        return

    print("\nStarting calibration...")
    cffi.calibrate_background([str(f) for f in fits_files])


def cmd_catalog(args):
    """Handle catalog command."""
    from .StreamingPipeline import convert_to_starcatalog, STREAMING_DIR
    from .StarCatalog import build_star_catalog, StarCatalog
    from .config import CALIBRATED_DATA_DIR, PHOTOMETRY_RESULTS_DIR

    # Auto-detect source
    has_streaming = False
    has_calibrated = (CALIBRATED_DATA_DIR / 'calibration_metadata.json').exists()

    if STREAMING_DIR.exists():
        streaming_files = list(STREAMING_DIR.glob("*_photometry*.csv"))
        has_streaming = len(streaming_files) > 0

    source = args.source
    if source == 'auto':
        if has_streaming and not has_calibrated:
            source = 'streaming'
        elif has_calibrated and not has_streaming:
            source = 'calibrated'
        elif has_streaming:
            source = 'streaming'  # Prefer streaming
        else:
            print("No data found. Run 'tess-ffi process' first.")
            return

    if source == 'streaming':
        if not args.sector:
            # Try to detect from files
            streaming_files = list(STREAMING_DIR.glob("*_photometry*.csv"))
            if streaming_files:
                filename = streaming_files[0].stem
                parts = filename.replace("_photometry_checkpoint", "").replace("_photometry", "").split("_")
                sector = int(parts[0][1:])
                camera_ccd = parts[1].split("-")
                camera, ccd = camera_ccd[0], camera_ccd[1]
            else:
                print("No streaming data found. Specify --sector")
                return
        else:
            sector = args.sector
            camera = args.camera
            ccd = args.ccd

        print(f"Converting streaming data: Sector {sector}, Camera {camera}, CCD {ccd}")
        catalog = convert_to_starcatalog(sector, camera, ccd)
        catalog.save()
        print(f"Catalog saved: {catalog.n_stars} stars")

        if args.add_tic:
            print("\nAdding TIC IDs (this takes ~2 hours)...")
            catalog.add_tic_ids()
            catalog.save()

    else:  # calibrated
        print("Building catalog from calibrated data...")
        catalog = build_star_catalog(reference_epoch=0)
        print(f"Catalog built: {catalog.n_stars} stars")


def cmd_lightcurves(args):
    """Handle lightcurves command."""
    from .StarCatalog import StarCatalog
    from .LightcurveBuilder import LightcurveCollection
    from .config import PHOTOMETRY_RESULTS_DIR

    catalog_path = PHOTOMETRY_RESULTS_DIR / 'star_catalog.json'

    if not catalog_path.exists():
        print("Star catalog not found.")
        print("Run 'tess-ffi catalog' first.")
        return

    print("Loading catalog...")
    catalog = StarCatalog.load()
    print(f"Loaded: {catalog.n_stars} stars")

    print("Building lightcurves...")
    collection = LightcurveCollection(catalog)
    print(f"Built: {len(collection)} lightcurves")

    # Specific star
    if args.star:
        lc = collection.get(args.star)
        if lc:
            print(f"\n{'='*50}")
            print(f"Star: {args.star}")
            print(f"  RA, Dec: {lc.star_info.get('ra', 'N/A'):.6f}, {lc.star_info.get('dec', 'N/A'):.6f}")
            print(f"  Detections: {lc.stats['n_good']}/{lc.stats['n_total']} ({lc.stats['completeness']:.1%})")
            print(f"  Mean flux: {lc.stats.get('mean_flux', 0):.2f}")
            print(f"  Amplitude: {lc.stats.get('amplitude', 0)*100:.2f}%")
            print(f"  Median SNR: {lc.stats.get('median_snr', 0):.1f}")

            if args.save_plots:
                path = lc.save_plot(normalized=True)
                print(f"\nPlot saved: {path}")
        else:
            print(f"Star {args.star} not found")
        return

    # Find by coordinates
    if args.ra is not None and args.dec is not None:
        # Find nearest star
        import numpy as np
        min_dist = float('inf')
        nearest_id = None

        for star_id, star_info in catalog.stars.items():
            ra = star_info.get('ra')
            dec = star_info.get('dec')
            if ra is not None and dec is not None:
                dist = np.sqrt((ra - args.ra)**2 + (dec - args.dec)**2)
                if dist < min_dist:
                    min_dist = dist
                    nearest_id = star_id

        if nearest_id:
            print(f"\nNearest star: {nearest_id} (distance: {min_dist:.6f} deg)")
            lc = collection.get(nearest_id)
            if lc:
                print(f"  Detections: {lc.stats['n_good']}/{lc.stats['n_total']}")
        return

    # Statistics
    if args.stats or args.export_stats:
        stats_df = collection.get_statistics_table()

        if args.stats:
            print(f"\n{'='*50}")
            print("Lightcurve Statistics")
            print(f"{'='*50}")
            print(f"Total stars: {len(stats_df)}")
            print(f"Median completeness: {stats_df['completeness'].median():.1%}")
            print(f"Median SNR: {stats_df['median_snr'].median():.1f}")

            if 'amplitude' in stats_df.columns:
                variable = stats_df[stats_df['amplitude'] > 0.01]
                print(f"Potentially variable (>1%): {len(variable)}")

        if args.export_stats:
            stats_df.to_csv(args.export_stats, index=False)
            print(f"\nStatistics saved: {args.export_stats}")
        return

    # Save plots
    if args.save_plots:
        print(f"\nSaving up to {args.max_plots} plots...")
        paths = collection.save_all_plots(max_plots=args.max_plots, normalized=True)
        print(f"Saved {len(paths)} plots")
        return

    # Default: show summary
    stats_df = collection.get_statistics_table()
    print(f"\n{'='*50}")
    print("Summary")
    print(f"{'='*50}")
    print(f"Total stars: {len(stats_df)}")
    print(f"Median completeness: {stats_df['completeness'].median():.1%}")
    print(f"Median SNR: {stats_df['median_snr'].median():.1f}")
    print(f"\nUse --star STAR_ID to view specific star")
    print("Use --save-plots to save lightcurve images")
    print("Use --stats to see detailed statistics")


def cmd_export(args):
    """Handle export command."""
    from .StarCatalog import StarCatalog
    from .LightcurveBuilder import LightcurveCollection
    from .MLExport import MLDataset, export_for_ml
    from .config import PHOTOMETRY_RESULTS_DIR

    catalog_path = PHOTOMETRY_RESULTS_DIR / 'star_catalog.json'

    if not catalog_path.exists():
        print("Star catalog not found.")
        print("Run 'tess-ffi catalog' first.")
        return

    print("Loading data...")
    catalog = StarCatalog.load()
    collection = LightcurveCollection(catalog)

    print(f"Total lightcurves: {len(collection)}")

    # Filter preview
    filtered = collection.filter_by_quality(args.min_completeness, args.min_snr)
    print(f"After quality filter: {len(filtered)}")

    print(f"\nExporting dataset '{args.name}'...")
    print(f"  Min completeness: {args.min_completeness}")
    print(f"  Min SNR: {args.min_snr}")
    print(f"  Format: {args.format}")

    dataset = MLDataset(collection)

    if args.format in ['all', 'features']:
        dataset.extract_features(args.min_completeness, args.min_snr)

    if args.format in ['all', 'timeseries']:
        dataset.prepare_timeseries(max_length=args.max_length, normalize=True)

    saved = dataset.save(args.name)

    print("\nExport complete!")
    print("Files saved:")
    for name, path in saved.items():
        print(f"  {name}: {path}")


def cmd_fix_coords(args):
    """Fix coordinates in existing data using tess-point (no download needed)."""
    import json
    from pathlib import Path
    from tqdm import tqdm

    print("=" * 60)
    print("WARNING: fix-coords is DEPRECATED")
    print("TIC IDs are now added automatically during processing.")
    print("Use: tess-ffi process --sector X --add-tic")
    print("=" * 60)
    print()

    from .StreamingPipeline import STREAMING_DIR

    # Find existing data
    checkpoint_files = list((STREAMING_DIR / "checkpoints").glob("*.json")) if STREAMING_DIR.exists() else []

    if not checkpoint_files:
        print("No existing data found. Run 'tess-ffi process' first.")
        return

    # Auto-detect or use provided sector
    if args.sector:
        sector = args.sector
        camera = int(args.camera)
        ccd = int(args.ccd)
        checkpoint_path = STREAMING_DIR / "checkpoints" / f"s{sector:04d}_{args.camera}-{args.ccd}_checkpoint.json"
        if not checkpoint_path.exists():
            print(f"No checkpoint found for sector {sector}")
            return
    else:
        # Use first found
        checkpoint_path = checkpoint_files[0]
        # Parse sector from filename
        parts = checkpoint_path.stem.replace("_checkpoint", "").split("_")
        sector = int(parts[0][1:])
        cam_ccd = parts[1].split("-")
        camera, ccd = int(cam_ccd[0]), int(cam_ccd[1])

    print(f"Fixing coordinates for: Sector {sector}, Camera {camera}, CCD {ccd}")

    # Load checkpoint
    with open(checkpoint_path, 'r') as f:
        checkpoint = json.load(f)

    stars = checkpoint.get('star_catalog', {})
    print(f"Stars in catalog: {len(stars)}")

    # Check if conversion needed - count how many have valid coords
    valid_count = 0
    for info in stars.values():
        ra = info.get('ra', 0) or 0
        dec = info.get('dec', 0) or 0
        if 0 < ra < 360 and -90 < dec < 90:
            valid_count += 1

    need_conversion = valid_count < len(stars) * 0.9  # Less than 90% valid

    if not need_conversion and not args.force:
        print(f"Coordinates already valid: {valid_count}/{len(stars)}")
        if not args.add_tic:
            return
    else:
        # Use tess-point for conversion (no download needed!)
        print("\nUsing tess-point for pixel -> RA/Dec conversion...")

        try:
            from tess_stars2px import tess_stars2px_reverse_function_entry
        except ImportError:
            print("tess-point not installed. Run: pip install tess-point")
            return

        updated = 0
        errors = 0

        for star_id, info in tqdm(stars.items(), desc="Converting coordinates"):
            # Get pixel coordinates
            x = info.get('ref_x')
            y = info.get('ref_y')

            # If ref_x/ref_y not set, try ra/dec (which contain pixel values)
            if x is None:
                x = info.get('ra')
            if y is None:
                y = info.get('dec')

            if x is not None and y is not None:
                try:
                    # tess_stars2px_reverse returns (ra, dec, pointing_data)
                    # Note: column = x, row = y
                    ra, dec, _ = tess_stars2px_reverse_function_entry(
                        sector, camera, ccd, float(x), float(y)
                    )

                    if 0 <= ra <= 360 and -90 <= dec <= 90:
                        info['ra'] = float(ra)
                        info['dec'] = float(dec)
                        info['ref_x'] = float(x)
                        info['ref_y'] = float(y)
                        updated += 1
                    else:
                        errors += 1
                except Exception:
                    errors += 1

        print(f"\nUpdated {updated}/{len(stars)} coordinates")
        if errors:
            print(f"Errors/invalid: {errors}")

        # Save
        checkpoint['star_catalog'] = stars
        with open(checkpoint_path, 'w') as f:
            json.dump(checkpoint, f, indent=2, default=str)

        print(f"Saved to {checkpoint_path}")

    # Add TIC if requested
    if args.add_tic:
        print("\nAdding TIC IDs (this will take ~1-2 hours)...")
        from .StreamingPipeline import StreamingProcessor

        processor = StreamingProcessor(sector, str(camera), str(ccd))
        processor.star_catalog = stars
        processor.add_tic_ids()

        # Save again
        checkpoint['star_catalog'] = processor.star_catalog
        with open(checkpoint_path, 'w') as f:
            json.dump(checkpoint, f, indent=2, default=str)

    print("\nDone!")


def cmd_status(args):
    """Handle status command."""
    from .config import FITS_DIR, CALIBRATED_DATA_DIR, PHOTOMETRY_RESULTS_DIR, MANIFEST_DIR, BASE_DIR
    from .StreamingPipeline import STREAMING_DIR

    print("\n" + "=" * 60)
    print("  TESS Pipeline Status")
    print("=" * 60)

    # FITS files
    fits_files = list(FITS_DIR.glob('*.fits'))
    print(f"\n1. FITS Files: {len(fits_files)}")
    if fits_files and args.verbose:
        total_size = sum(f.stat().st_size for f in fits_files) / (1024**3)
        print(f"   Total size: {total_size:.2f} GB")

    # Streaming results
    print(f"\n2. Streaming Results:")
    if STREAMING_DIR.exists():
        checkpoints = list(STREAMING_DIR.glob("*_photometry*.csv"))
        summaries = list(STREAMING_DIR.glob("*_summary.json"))
        print(f"   Data files: {len(checkpoints)}")
        print(f"   Completed runs: {len(summaries)}")

        if args.verbose and summaries:
            for s in summaries[:3]:
                with open(s, 'r') as f:
                    data = json.load(f)
                print(f"     - {data['session_id']}: {data['n_stars']} stars, {data['n_epochs']} epochs")
    else:
        print("   Not started")

    # Calibration
    print(f"\n3. Calibration:")
    metadata_path = CALIBRATED_DATA_DIR / 'calibration_metadata.json'
    if metadata_path.exists():
        with open(metadata_path, 'r') as f:
            cal = json.load(f)
        print(f"   Epochs: {cal['n_epochs']}")
        if args.verbose:
            print(f"   Failed: {cal['n_failed']}")
    else:
        print("   Not done")

    # Star catalog
    print(f"\n4. Star Catalog:")
    catalog_path = PHOTOMETRY_RESULTS_DIR / 'star_catalog.json'
    if catalog_path.exists():
        with open(catalog_path, 'r') as f:
            cat = json.load(f)
        print(f"   Stars: {cat['n_stars']}")
        print(f"   Epochs: {cat['n_epochs']}")
    else:
        print("   Not built")

    # ML data
    print(f"\n5. ML Data:")
    ml_dir = BASE_DIR / 'ml_data'
    if ml_dir.exists():
        datasets = list(ml_dir.glob('*_metadata.json'))
        print(f"   Datasets: {len(datasets)}")
        if args.verbose and datasets:
            for d in datasets:
                print(f"     - {d.stem.replace('_metadata', '')}")
    else:
        print("   Not exported")

    print()


def cmd_interactive(args):
    """Launch interactive menu."""
    from main import main as interactive_main
    interactive_main()


def cmd_find_variables(args):
    """Find variable stars in photometry data."""
    from .VariableStarFinder import VariableStarFinder

    finder = VariableStarFinder(args.sector, args.camera, args.ccd)

    # Load and filter lightcurves
    n_stars = finder.load_and_build_lightcurves(
        min_completeness=args.min_completeness,
        min_snr=args.min_snr
    )

    if n_stars == 0:
        print("No stars found after quality filtering.")
        return

    # Calculate periodicity
    finder.calculate_periodicity()

    # Optionally enrich with TIC stellar parameters
    if args.enrich_tic:
        finder.enrich_with_tic()

    # Calculate variability scores
    finder.calculate_variability_scores()

    # Get candidates using robust metrics
    candidates = finder.get_variable_candidates(
        min_amplitude=args.min_amplitude,  # Now uses amplitude_robust
        min_chi2=None,
        max_outlier_fraction=0.05,  # Reject stars with >5% outlier points
        top_n=args.top_n
    )

    print(f"\nFound {len(candidates)} variable candidates")

    # Save results
    finder.save_results(args.output_dir)

    # Print summary
    finder.print_summary()

    # Optionally generate plots
    if args.plot:
        n_plots = min(20, len(candidates))
        finder.plot_top_candidates(n_plots=n_plots, output_dir=args.output_dir)


def main():
    """Main entry point."""
    parser = create_parser()
    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        print("\n" + "=" * 60)
        print("Quick start:")
        print("  tess-ffi process --sector 70    # Download & process")
        print("  tess-ffi status                 # Check status")
        print("  tess-ffi interactive            # Interactive menu")
        print("=" * 60)
        return

    # Ensure directories exist
    from .config import ensure_directories
    ensure_directories()

    # Dispatch to command handler
    commands = {
        'download': cmd_download,
        'process': cmd_process,
        'calibrate': cmd_calibrate,
        'catalog': cmd_catalog,
        'lightcurves': cmd_lightcurves,
        'export': cmd_export,
        'fix-coords': cmd_fix_coords,
        'find-variables': cmd_find_variables,
        'status': cmd_status,
        'interactive': cmd_interactive,
    }

    handler = commands.get(args.command)
    if handler:
        try:
            handler(args)
        except KeyboardInterrupt:
            print("\n\nInterrupted by user.")
        except Exception as e:
            print(f"\nError: {e}")
            if hasattr(args, 'verbose') and args.verbose:
                import traceback
                traceback.print_exc()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
