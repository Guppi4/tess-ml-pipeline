"""
TESS FFI Data Processing - Main CLI Application

Interactive command-line interface for the TESS FFI processing pipeline.

Features:
- Smart sector-based downloading (auto-discovers available days)
- Proper cross-matching of stars across epochs
- Gap detection and logging
- Quality flags and metrics
- ML export functionality
"""

import os
import json

from .config import (
    FITS_DIR, CALIBRATED_DATA_DIR, PHOTOMETRY_RESULTS_DIR,
    MANIFEST_DIR, ensure_directories
)
from . import FFIDownloader as dffi
from . import FFICalibrate as cffi


def clear_screen():
    """Clear terminal screen."""
    os.system('cls' if os.name == 'nt' else 'clear')


def print_header():
    """Print application header."""
    print("\n" + "=" * 60)
    print("  TESS FFI Data Processing Pipeline v2.0")
    print("  For lightcurve generation and ML classification")
    print("=" * 60)


def print_main_menu():
    """Display main menu options."""
    options = [
        "STREAMING MODE (recommended) - process without storing files",
        "Download FFI data (requires ~10GB per sector)",
        "Calibrate images",
        "Build star catalog (cross-matching)",
        "Generate lightcurves",
        "Export for ML",
        "View data status",
        "Exit"
    ]

    print("\nMain Menu:")
    for i, desc in enumerate(options, 1):
        print(f"  {i}) {desc}")
    print()


# =============================================================================
# Streaming Mode (Recommended)
# =============================================================================

def streaming_menu():
    """Streaming mode - process on-the-fly without storing raw files."""
    from .StreamingPipeline import run_streaming_pipeline, get_streaming_results, STREAMING_DIR
    from .SectorAnalyzer import get_sector_quality, compare_sectors, quick_check

    print("\n" + "=" * 60)
    print("STREAMING MODE")
    print("=" * 60)
    print("\nThis mode:")
    print("  - Downloads one file at a time")
    print("  - Extracts photometry immediately")
    print("  - Deletes raw file after processing")
    print("  - Needs only ~500 MB free space (instead of ~10 GB)")
    print("  - Can be paused and resumed anytime")

    print("\n" + "-" * 50)
    print("Options:")
    print("  1) Check sector quality FIRST (recommended)")
    print("  2) Compare multiple sectors")
    print("  3) Start processing run")
    print("  4) Resume previous run")
    print("  5) View completed results")
    print("  6) Back")

    choice = input("\nSelect: ").strip()

    if choice == "1":
        # Check single sector quality
        try:
            sector = int(input("\nSector number to check (1-98): ").strip())
            camera = input("Camera (1-4, default 1): ").strip() or "1"
            ccd = input("CCD (1-4, default 1): ").strip() or "1"

            quality = quick_check(sector)

            if quality.quality_score in ['excellent', 'good']:
                proceed = input("\nStart processing this sector? (y/n): ").strip().lower()
                if proceed == 'y':
                    run_streaming_pipeline(sector, camera, ccd, resume=True)

        except ValueError:
            print("Invalid input")

    elif choice == "2":
        # Compare multiple sectors
        try:
            print("\nEnter sector range to compare:")
            start = int(input("  Start sector (default 60): ").strip() or "60")
            end = int(input("  End sector (default 70): ").strip() or "70")
            camera = input("  Camera (default 1): ").strip() or "1"
            ccd = input("  CCD (default 1): ").strip() or "1"

            sectors = list(range(start, end + 1))
            results = compare_sectors(sectors, camera, ccd)

            if results and results[0].available:
                proceed = input(f"\nProcess best sector ({results[0].sector})? (y/n): ").strip().lower()
                if proceed == 'y':
                    run_streaming_pipeline(results[0].sector, camera, ccd, resume=True)

        except ValueError:
            print("Invalid input")

    elif choice == "3" or choice == "4":
        # Start or resume processing
        try:
            sector = int(input("\nSector number (1-98): ").strip())

            # Quick quality check first
            print("\nQuick quality check...")
            quality = get_sector_quality(sector, verbose=False)

            if quality.available:
                print(f"  Quality: {quality.quality_score}")
                print(f"  Coverage: {quality.coverage_percent:.1f}%")
                print(f"  Days: {quality.total_days}")

                if quality.quality_score in ['poor', 'bad']:
                    print(f"\n  WARNING: This sector has significant gaps!")
                    confirm = input("  Continue anyway? (y/n): ").strip().lower()
                    if confirm != 'y':
                        return

            camera = input("\nCamera (1-4, default 1): ").strip() or "1"
            ccd = input("CCD (1-4, default 1): ").strip() or "1"

            print(f"\nWill process: Sector {sector}, Camera {camera}, CCD {ccd}")
            print(f"Estimated time: ~{quality.total_days * 2} minutes")

            confirm = input("Start? (y/n): ").strip().lower()

            if confirm == 'y':
                resume = (choice == "4")
                run_streaming_pipeline(sector, camera, ccd, resume=resume)

        except ValueError:
            print("Invalid input")
        except KeyboardInterrupt:
            print("\n\nProcessing paused. You can resume later with option 4.")

    elif choice == "5":
        # Show completed results
        print("\nLooking for completed results...")

        if STREAMING_DIR.exists():
            summaries = list(STREAMING_DIR.glob("*_summary.json"))
            if summaries:
                print(f"\nFound {len(summaries)} completed runs:")
                for s in summaries:
                    with open(s, 'r') as f:
                        data = json.load(f)
                    print(f"\n  {data['session_id']}:")
                    print(f"    Stars: {data['n_stars']}")
                    print(f"    Epochs: {data['n_epochs']}")
                    print(f"    Completeness: {data.get('completeness', 0):.1%}")
            else:
                print("No completed runs found.")
        else:
            print("No streaming results directory found.")

    elif choice == "6":
        return


# =============================================================================
# Download Menu
# =============================================================================

def download_menu():
    """Handle download options."""
    print("\n" + "-" * 50)
    print("Download Options:")
    print("-" * 50)
    print("  1) Download by sector (recommended)")
    print("  2) Download by date range")
    print("  3) Show available sectors")
    print("  4) Back to main menu")

    choice = input("\nSelect option: ").strip()

    if choice == "1":
        download_by_sector()
    elif choice == "2":
        download_by_dates()
    elif choice == "3":
        show_available_sectors()
    elif choice == "4":
        return
    else:
        print("Invalid option")


def download_by_sector():
    """Download all data for a sector."""
    print("\n--- Download by Sector ---")
    print("This will automatically find and download all available days.\n")

    try:
        sector = int(input("Enter sector number (1-98): ").strip())

        # Show sector info first
        print("\nFetching sector information...")
        info = dffi.get_sector_info(sector)

        if not info['available']:
            print(f"Sector {sector} is not available on MAST.")
            return

        print(f"\nSector {sector} info:")
        print(f"  Date range: {info.get('date_start', 'N/A')} to {info.get('date_end', 'N/A')}")
        print(f"  Available days: {info['total_days']}")

        # Camera and CCD
        print("\nCamera options: 1, 2, 3, 4")
        camera = input("Enter camera (default 1): ").strip() or "1"

        print("CCD options: 1, 2, 3, 4")
        ccd = input("Enter CCD (default 1): ").strip() or "1"

        # Confirm
        print(f"\nWill download: Sector {sector}, Camera {camera}, CCD {ccd}")
        print(f"Expected: ~{info['total_days'] * 10} files (~{info['total_days'] * 340}MB)")

        confirm = input("\nProceed? (y/n): ").strip().lower()
        if confirm == 'y':
            dffi.download_sector(sector, camera, ccd)

    except ValueError:
        print("Invalid input. Please enter a number.")
    except Exception as e:
        print(f"Error: {e}")


def download_by_dates():
    """Download data by calendar date range."""
    print("\n--- Download by Date Range ---")
    print("Enter dates in YYYY-MM-DD format\n")

    try:
        start_date = input("Start date: ").strip()
        end_date = input("End date: ").strip()

        camera = input("Camera (default 1): ").strip() or "1"
        ccd = input("CCD (default 1): ").strip() or "1"

        confirm = input(f"\nDownload {start_date} to {end_date}? (y/n): ").strip().lower()
        if confirm == 'y':
            dffi.download_by_date_range(start_date, end_date, camera, ccd)

    except Exception as e:
        print(f"Error: {e}")


def show_available_sectors():
    """Show list of available sectors."""
    print("\nFetching available sectors from MAST...")

    sectors = dffi.list_available_sectors()

    if sectors:
        print(f"\nAvailable sectors: {len(sectors)}")
        print(f"Range: {min(sectors)} to {max(sectors)}")

        # Show recent sectors
        recent = sorted(sectors)[-10:]
        print(f"\nMost recent: {', '.join(map(str, recent))}")
    else:
        print("Could not fetch sector list.")


# =============================================================================
# Calibration Menu
# =============================================================================

def calibrate_menu():
    """Handle calibration."""
    from .StreamingPipeline import STREAMING_DIR

    print("\n--- Calibration ---")

    # Check for streaming data first
    if STREAMING_DIR.exists():
        streaming_files = list(STREAMING_DIR.glob("*_photometry*.csv"))
        if streaming_files:
            print("NOTE: You have STREAMING data available!")
            print("Streaming mode already includes calibration.")
            print("Skip this step and go to 'Build star catalog' (option 4).")
            print()

    fits_files = sorted(FITS_DIR.glob('*.fits'))

    if not fits_files:
        print(f"No FITS files found in {FITS_DIR}")
        print("If you used Streaming Mode, skip this step - data is already calibrated.")
        print("Go directly to 'Build star catalog' (option 4).")
        return

    print(f"Found {len(fits_files)} FITS files.")

    # Check for existing calibration
    metadata_path = CALIBRATED_DATA_DIR / 'calibration_metadata.json'
    if metadata_path.exists():
        print("\nExisting calibration found!")
        with open(metadata_path, 'r') as f:
            meta = json.load(f)
        print(f"  Epochs: {meta['n_epochs']}")
        print(f"  Created: {meta['created']}")

        overwrite = input("\nRe-calibrate? (y/n): ").strip().lower()
        if overwrite != 'y':
            return

    confirm = input("\nStart calibration? (y/n): ").strip().lower()
    if confirm == 'y':
        cffi.calibrate_background([str(f) for f in fits_files])


# =============================================================================
# Star Catalog Menu
# =============================================================================

def build_catalog_menu():
    """Build star catalog with cross-matching."""
    from .StarCatalog import build_star_catalog, StarCatalog
    from .StreamingPipeline import convert_to_starcatalog, STREAMING_DIR

    print("\n--- Build Star Catalog ---")

    # Check for streaming results FIRST
    streaming_checkpoints = list(STREAMING_DIR.glob("*_photometry_checkpoint.csv")) if STREAMING_DIR.exists() else []
    streaming_final = list(STREAMING_DIR.glob("*_photometry.csv")) if STREAMING_DIR.exists() else []
    streaming_files = streaming_final or streaming_checkpoints

    # Check for old-style calibration
    metadata_path = CALIBRATED_DATA_DIR / 'calibration_metadata.json'
    has_calibration = metadata_path.exists()

    if streaming_files and not has_calibration:
        # Only streaming data available
        print("Found STREAMING data (recommended):")
        for f in streaming_files[:5]:
            print(f"  - {f.name}")

        print("\nOptions:")
        print("  1) Convert streaming data to catalog")
        print("  2) Back")

        choice = input("\nSelect: ").strip()

        if choice == "1":
            # Parse sector from filename
            filename = streaming_files[0].stem
            parts = filename.replace("_photometry_checkpoint", "").replace("_photometry", "").split("_")
            sector = int(parts[0][1:])  # s0070 -> 70
            camera_ccd = parts[1].split("-")
            camera, ccd = camera_ccd[0], camera_ccd[1]

            print(f"\nConverting: Sector {sector}, Camera {camera}, CCD {ccd}")
            catalog = convert_to_starcatalog(sector, camera, ccd)

            # Save
            catalog.save()
            print(f"\nCatalog saved! {catalog.n_stars} stars")

            # Ask about TIC
            add_tic = input("\nAdd TIC IDs? (slow, ~2 hours) (y/n): ").strip().lower()
            if add_tic == 'y':
                catalog.add_tic_ids()
                catalog.save()
        return

    elif streaming_files and has_calibration:
        # Both available
        print("Found data sources:")
        print(f"  1) Streaming results ({len(streaming_files)} files)")
        print(f"  2) Calibrated data (old pipeline)")

        source = input("\nUse which source? (1/2): ").strip()

        if source == "1":
            filename = streaming_files[0].stem
            parts = filename.replace("_photometry_checkpoint", "").replace("_photometry", "").split("_")
            sector = int(parts[0][1:])
            camera_ccd = parts[1].split("-")
            camera, ccd = camera_ccd[0], camera_ccd[1]

            print(f"\nConverting: Sector {sector}, Camera {camera}, CCD {ccd}")
            catalog = convert_to_starcatalog(sector, camera, ccd)
            catalog.save()
            print(f"\nCatalog saved! {catalog.n_stars} stars")
            return

    # Old pipeline path
    if not has_calibration:
        print("No data found.")
        print("Please use Streaming Mode (option 1) to download data first.")
        return

    with open(metadata_path, 'r') as f:
        cal_meta = json.load(f)

    print(f"Calibration: {cal_meta['n_epochs']} epochs available")

    # Check for existing catalog
    catalog_path = PHOTOMETRY_RESULTS_DIR / 'star_catalog.json'
    if catalog_path.exists():
        print("\nExisting catalog found!")
        with open(catalog_path, 'r') as f:
            cat_meta = json.load(f)
        print(f"  Stars: {cat_meta['n_stars']}")
        print(f"  Epochs: {cat_meta['n_epochs']}")

        rebuild = input("\nRebuild catalog? (y/n): ").strip().lower()
        if rebuild != 'y':
            return

    print("\nThis will:")
    print("  1. Detect stars in reference epoch")
    print("  2. Perform forced photometry on ALL epochs")
    print("  3. Create master catalog with consistent star IDs")
    print("  4. Track gaps and quality flags")

    ref_epoch = input("\nReference epoch index (default 0, best quality): ").strip()
    ref_epoch = int(ref_epoch) if ref_epoch else 0

    confirm = input("\nProceed? (y/n): ").strip().lower()
    if confirm == 'y':
        catalog = build_star_catalog(reference_epoch=ref_epoch)
        print(f"\nCatalog built with {catalog.n_stars} stars")


# =============================================================================
# Lightcurve Menu
# =============================================================================

def lightcurve_menu():
    """Build and view lightcurves."""
    from .StarCatalog import StarCatalog
    from .LightcurveBuilder import LightcurveCollection

    print("\n--- Lightcurves ---")

    catalog_path = PHOTOMETRY_RESULTS_DIR / 'star_catalog.json'

    if not catalog_path.exists():
        print("Star catalog not found.")
        print("Please build catalog first (option 3).")
        return

    print("Loading catalog...")
    catalog = StarCatalog.load()
    print(f"Loaded: {catalog.n_stars} stars")

    print("\nBuilding lightcurves...")
    collection = LightcurveCollection(catalog)
    print(f"Built: {len(collection)} lightcurves")

    # Statistics
    stats_df = collection.get_statistics_table()

    print("\n--- Lightcurve Statistics ---")
    print(f"Total stars: {len(stats_df)}")

    if len(stats_df) > 0:
        print(f"Median completeness: {stats_df['completeness'].median():.1%}")
        print(f"Median SNR: {stats_df['median_snr'].median():.1f}")

        # Variable stars
        if 'amplitude' in stats_df.columns:
            variable = stats_df[stats_df['amplitude'] > 0.01]
            print(f"Potentially variable (>1% amplitude): {len(variable)}")

    # Options
    print("\nOptions:")
    print("  1) Save lightcurve plots")
    print("  2) View specific star")
    print("  3) Export statistics")
    print("  4) Back")

    choice = input("\nSelect: ").strip()

    if choice == "1":
        max_plots = input("Max plots to save (default 100): ").strip()
        max_plots = int(max_plots) if max_plots else 100
        paths = collection.save_all_plots(max_plots=max_plots, normalized=True)
        print(f"Saved {len(paths)} plots to {PHOTOMETRY_RESULTS_DIR.parent / 'lightcurves'}")

    elif choice == "2":
        star_id = input("Enter star ID (e.g., STAR_000123): ").strip()
        lc = collection.get(star_id)
        if lc:
            print(f"\nStar: {star_id}")
            print(f"  RA, Dec: {lc.star_info.get('ra', 'N/A'):.6f}, {lc.star_info.get('dec', 'N/A'):.6f}")
            print(f"  Detections: {lc.stats['n_good']}/{lc.stats['n_total']} ({lc.stats['completeness']:.1%})")
            print(f"  Mean flux: {lc.stats.get('mean_flux', 0):.2f}")
            print(f"  Amplitude: {lc.stats.get('amplitude', 0)*100:.2f}%")

            save = input("\nSave plot? (y/n): ").strip().lower()
            if save == 'y':
                path = lc.save_plot(normalized=True)
                print(f"Saved to {path}")
        else:
            print(f"Star {star_id} not found")

    elif choice == "3":
        stats_path = PHOTOMETRY_RESULTS_DIR / 'lightcurve_statistics.csv'
        stats_df.to_csv(stats_path, index=False)
        print(f"Saved to {stats_path}")


# =============================================================================
# ML Export Menu
# =============================================================================

def ml_export_menu():
    """Export data for ML classification."""
    from .StarCatalog import StarCatalog
    from .LightcurveBuilder import LightcurveCollection
    from .MLExport import export_for_ml, MLDataset

    print("\n--- ML Export ---")

    catalog_path = PHOTOMETRY_RESULTS_DIR / 'star_catalog.json'

    if not catalog_path.exists():
        print("Star catalog not found.")
        print("Please build catalog first (option 3).")
        return

    print("\nExport settings:")

    min_completeness = input("Min completeness (default 0.5 = 50%): ").strip()
    min_completeness = float(min_completeness) if min_completeness else 0.5

    min_snr = input("Min SNR (default 3.0): ").strip()
    min_snr = float(min_snr) if min_snr else 3.0

    dataset_name = input("Dataset name (default 'tess_dataset'): ").strip()
    dataset_name = dataset_name if dataset_name else 'tess_dataset'

    print("\nLoading data...")
    catalog = StarCatalog.load()
    collection = LightcurveCollection(catalog)

    print(f"Total lightcurves: {len(collection)}")

    # Filter preview
    filtered = collection.filter_by_quality(min_completeness, min_snr)
    print(f"After quality filter: {len(filtered)}")

    confirm = input("\nExport? (y/n): ").strip().lower()
    if confirm == 'y':
        saved = export_for_ml(
            collection,
            name=dataset_name,
            min_completeness=min_completeness,
            min_snr=min_snr
        )
        print("\nExport complete!")
        print("Files saved:")
        for name, path in saved.items():
            print(f"  {name}: {path}")


# =============================================================================
# Status Menu
# =============================================================================

def status_menu():
    """View current data status."""
    print("\n" + "=" * 50)
    print("Data Status")
    print("=" * 50)

    # FITS files
    fits_files = list(FITS_DIR.glob('*.fits'))
    print(f"\n1. FITS Files: {len(fits_files)}")
    if fits_files:
        total_size = sum(f.stat().st_size for f in fits_files) / (1024**3)
        print(f"   Total size: {total_size:.2f} GB")

    # Manifests
    if MANIFEST_DIR.exists():
        manifests = list(MANIFEST_DIR.glob('manifest_*.json'))
        print(f"\n2. Download Manifests: {len(manifests)}")
        if manifests:
            latest = max(manifests, key=lambda x: x.stat().st_mtime)
            with open(latest, 'r') as f:
                m = json.load(f)
            print(f"   Latest: {latest.name}")
            if 'summary' in m:
                print(f"   Files: {m['summary'].get('valid_files', 0)}")
                print(f"   Gaps: {m['summary'].get('gaps_detected', 0)}")

    # Calibration
    metadata_path = CALIBRATED_DATA_DIR / 'calibration_metadata.json'
    print(f"\n3. Calibration:")
    if metadata_path.exists():
        with open(metadata_path, 'r') as f:
            cal = json.load(f)
        print(f"   Epochs: {cal['n_epochs']}")
        print(f"   Failed: {cal['n_failed']}")
        if 'time_coverage' in cal and cal['time_coverage']:
            tc = cal['time_coverage']
            print(f"   MJD: {tc.get('mjd_start', 0):.2f} - {tc.get('mjd_end', 0):.2f}")
    else:
        print("   Not done")

    # Star catalog
    catalog_path = PHOTOMETRY_RESULTS_DIR / 'star_catalog.json'
    print(f"\n4. Star Catalog:")
    if catalog_path.exists():
        with open(catalog_path, 'r') as f:
            cat = json.load(f)
        print(f"   Stars: {cat['n_stars']}")
        print(f"   Epochs: {cat['n_epochs']}")
    else:
        print("   Not built")

    # ML data
    ml_dir = FITS_DIR.parent / 'ml_data'
    print(f"\n5. ML Data:")
    if ml_dir.exists():
        datasets = list(ml_dir.glob('*_metadata.json'))
        print(f"   Datasets: {len(datasets)}")
    else:
        print("   Not exported")


# =============================================================================
# Main Loop
# =============================================================================

def main():
    """Main entry point."""
    ensure_directories()

    while True:
        clear_screen()
        print_header()
        print_main_menu()

        choice = input("Select option (1-8): ").strip()

        if choice == "1":
            streaming_menu()
        elif choice == "2":
            download_menu()
        elif choice == "3":
            calibrate_menu()
        elif choice == "4":
            build_catalog_menu()
        elif choice == "5":
            lightcurve_menu()
        elif choice == "6":
            ml_export_menu()
        elif choice == "7":
            status_menu()
        elif choice == "8":
            print("\nGoodbye!")
            break
        else:
            print("Invalid option")

        input("\nPress Enter to continue...")


if __name__ == "__main__":
    main()
