"""
Download CVZ sectors with nice progress display.
Usage: python scripts/download_cvz.py
"""

import sys
import time
from datetime import datetime, timedelta

# Add src to path
sys.path.insert(0, 'src')

from tess.StreamingPipeline import run_streaming_pipeline

# Configuration
SECTORS = [61, 62, 63]
CAMERA = "4"
CCD = "2"
CADENCE_SKIP = 6
WORKERS = 5  # Reduced from 10 for stability

def print_header():
    print("\n" + "="*70)
    print("   🛰️  TESS CVZ DOWNLOADER - South Ecliptic Pole")
    print("="*70)
    print(f"   Sectors: {SECTORS}")
    print(f"   Camera {CAMERA}, CCD {CCD}")
    print(f"   Workers: {WORKERS}, Cadence skip: {CADENCE_SKIP}")
    print("="*70 + "\n")

def print_sector_status(completed, current, remaining):
    """Print visual status of all sectors."""
    print("\n" + "-"*50)
    print("   📊 OVERALL PROGRESS")
    print("-"*50)

    for sector in SECTORS:
        if sector in completed:
            icon = "✅"
            status = "Complete"
        elif sector == current:
            icon = "⏳"
            status = "Downloading..."
        else:
            icon = "⬜"
            status = "Pending"

        print(f"   {icon} Sector {sector}: {status}")

    print("-"*50)
    print(f"   Progress: {len(completed)}/{len(SECTORS)} sectors")
    print("-"*50 + "\n")

def main():
    print_header()

    completed = []
    start_time = time.time()

    for i, sector in enumerate(SECTORS):
        remaining = SECTORS[i+1:] if i+1 < len(SECTORS) else []

        print_sector_status(completed, sector, remaining)

        sector_start = time.time()
        print(f"🚀 Starting Sector {sector}...\n")

        try:
            run_streaming_pipeline(
                sector=sector,
                camera=CAMERA,
                ccd=CCD,
                cadence_skip=CADENCE_SKIP,
                workers=WORKERS
            )

            sector_time = time.time() - sector_start
            print(f"\n✅ Sector {sector} complete! ({sector_time/60:.1f} min)")
            completed.append(sector)

        except KeyboardInterrupt:
            print(f"\n\n⚠️  Interrupted at sector {sector}")
            print("   Run again to resume from checkpoint.")
            break
        except Exception as e:
            print(f"\n❌ Error in sector {sector}: {e}")
            print("   Run again to retry.")
            break

    # Final summary
    total_time = time.time() - start_time

    print("\n" + "="*70)
    print("   🏁 DOWNLOAD COMPLETE")
    print("="*70)

    for sector in SECTORS:
        icon = "✅" if sector in completed else "❌"
        print(f"   {icon} Sector {sector}")

    print("-"*50)
    print(f"   Total time: {total_time/60:.1f} min ({total_time/3600:.1f} hours)")
    print(f"   Sectors completed: {len(completed)}/{len(SECTORS)}")
    print("="*70)

    if len(completed) == len(SECTORS):
        print("\n   🎉 All done! Now run:")
        print("   python -c \"from tess.combine_sectors import combine_cvz_sectors; combine_cvz_sectors([61,62,63], '4', '2')\"")
        print()

if __name__ == "__main__":
    main()
