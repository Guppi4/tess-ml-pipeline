"""
Verification script for TIC 610495795
Uses TESScut to get image cutout and verify star position.
"""

import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
import astropy.units as u
from astroquery.mast import Tesscut
import os

# TIC 610495795 coordinates (from MAST catalog)
TIC_RA = 0.406943688937586  # degrees
TIC_DEC = -9.26197775721615  # degrees
TIC_ID = 610495795
TIC_TMAG = 18.55  # T magnitude from TIC

# Our detected star coordinates (from pipeline)
OUR_RA = 0.3997931753444407
OUR_DEC = -9.255102158000499

print("=" * 60)
print("VERIFICATION: TIC 610495795")
print("=" * 60)
print(f"\nTIC catalog coordinates:")
print(f"  RA:  {TIC_RA:.6f} deg")
print(f"  Dec: {TIC_DEC:.6f} deg")
print(f"  Tmag: {TIC_TMAG}")
print(f"\nOur pipeline coordinates:")
print(f"  RA:  {OUR_RA:.6f} deg")
print(f"  Dec: {OUR_DEC:.6f} deg")

# Calculate separation
tic_coord = SkyCoord(ra=TIC_RA*u.deg, dec=TIC_DEC*u.deg)
our_coord = SkyCoord(ra=OUR_RA*u.deg, dec=OUR_DEC*u.deg)
sep = tic_coord.separation(our_coord)
print(f"\nSeparation: {sep.arcsec:.2f} arcsec")

# Step 1: Get TESScut cutout
print("\n" + "=" * 60)
print("STEP 1: Downloading TESScut cutout...")
print("=" * 60)

# Use TIC coordinates as center
cutout_coord = tic_coord
cutout_size = 20  # pixels (smaller = faster download)

print(f"Center: RA={TIC_RA:.4f}, Dec={TIC_DEC:.4f}")
print(f"Size: {cutout_size}x{cutout_size} pixels")
print(f"Sector: 70")

try:
    # Download cutout
    hdulist = Tesscut.get_cutouts(coordinates=cutout_coord, size=cutout_size, sector=70)

    if len(hdulist) == 0:
        print("No cutouts found!")
    else:
        print(f"Found {len(hdulist)} cutout(s)")

        # Use first cutout
        hdu = hdulist[0]

        print(f"\nHDUs: {[h.name for h in hdu]}")

        # Get image data - TESScut returns a cube (time, y, x)
        # We'll use the first frame
        flux_cube = hdu[1].data['FLUX']
        header = hdu[2].header  # WCS is in aperture extension

        print(f"Flux cube shape: {flux_cube.shape}")
        n_frames = flux_cube.shape[0]
        print(f"Number of frames: {n_frames}")

        # Use middle frame (avoid edge effects)
        frame_idx = n_frames // 2
        image = flux_cube[frame_idx]
        print(f"Using frame {frame_idx}, shape: {image.shape}")

        # Get WCS
        wcs = WCS(header)
        print(f"WCS: {wcs}")

        # Step 2: Convert coordinates to pixels in cutout
        print("\n" + "=" * 60)
        print("STEP 2: Converting coordinates...")
        print("=" * 60)

        tic_x, tic_y = wcs.world_to_pixel(tic_coord)
        our_x, our_y = wcs.world_to_pixel(our_coord)

        print(f"TIC 610495795 in cutout: ({tic_x:.2f}, {tic_y:.2f})")
        print(f"Our detection in cutout: ({our_x:.2f}, {our_y:.2f})")

        pix_sep = np.sqrt((tic_x - our_x)**2 + (tic_y - our_y)**2)
        print(f"Pixel separation: {pix_sep:.2f} pixels")
        print(f"TESS pixel scale: ~21 arcsec/pixel")
        print(f"Expected pixel separation: {sep.arcsec/21:.2f} pixels")

        # Step 3: Measure flux at both positions
        print("\n" + "=" * 60)
        print("STEP 3: Measuring flux...")
        print("=" * 60)

        from photutils.aperture import CircularAperture, CircularAnnulus, aperture_photometry

        aperture_radius = 3.0

        ny, nx = image.shape
        print(f"Cutout size: {nx}x{ny} pixels")

        # Check if positions are in cutout
        tic_in = 0 <= tic_x < nx and 0 <= tic_y < ny
        our_in = 0 <= our_x < nx and 0 <= our_y < ny

        print(f"TIC in cutout: {tic_in}")
        print(f"Our in cutout: {our_in}")

        if tic_in:
            tic_ap = CircularAperture((tic_x, tic_y), r=aperture_radius)
            tic_ann = CircularAnnulus((tic_x, tic_y), r_in=6, r_out=10)

            tic_phot = aperture_photometry(image, tic_ap)
            tic_ann_phot = aperture_photometry(image, tic_ann)

            tic_bkg = tic_ann_phot['aperture_sum'][0] / tic_ann.area
            tic_flux = tic_phot['aperture_sum'][0] - tic_bkg * tic_ap.area

            print(f"\nTIC 610495795:")
            print(f"  Aperture sum: {tic_phot['aperture_sum'][0]:.1f}")
            print(f"  Background: {tic_bkg:.1f}/pixel")
            print(f"  Net flux: {tic_flux:.1f}")

        if our_in:
            our_ap = CircularAperture((our_x, our_y), r=aperture_radius)
            our_ann = CircularAnnulus((our_x, our_y), r_in=6, r_out=10)

            our_phot = aperture_photometry(image, our_ap)
            our_ann_phot = aperture_photometry(image, our_ann)

            our_bkg = our_ann_phot['aperture_sum'][0] / our_ann.area
            our_flux = our_phot['aperture_sum'][0] - our_bkg * our_ap.area

            print(f"\nOur detection:")
            print(f"  Aperture sum: {our_phot['aperture_sum'][0]:.1f}")
            print(f"  Background: {our_bkg:.1f}/pixel")
            print(f"  Net flux: {our_flux:.1f}")

        # Step 4: Visualization
        print("\n" + "=" * 60)
        print("STEP 4: Creating visualization...")
        print("=" * 60)

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        # Left: Cutout with both positions
        ax1 = axes[0]
        vmin, vmax = np.nanpercentile(image, [5, 99])
        im = ax1.imshow(image, origin='lower', cmap='gray', vmin=vmin, vmax=vmax)

        if tic_in:
            ax1.plot(tic_x, tic_y, 'r+', ms=15, mew=2, label=f'TIC 610495795')
            circle1 = plt.Circle((tic_x, tic_y), aperture_radius, fill=False, color='red', lw=2)
            ax1.add_patch(circle1)

        if our_in:
            ax1.plot(our_x, our_y, 'gx', ms=15, mew=2, label=f'Our detection')
            circle2 = plt.Circle((our_x, our_y), aperture_radius, fill=False, color='green', lw=2)
            ax1.add_patch(circle2)

        ax1.legend(loc='upper right')
        ax1.set_xlabel('X pixel')
        ax1.set_ylabel('Y pixel')
        ax1.set_title(f'TESScut: Sector 70\nSeparation: {sep.arcsec:.1f}" ({pix_sep:.1f} pix)')
        plt.colorbar(im, ax=ax1, label='Flux')

        # Right: Zoom on center
        ax2 = axes[1]
        zoom = 15
        cx, cy = nx//2, ny//2
        x1, x2 = max(0, cx-zoom), min(nx, cx+zoom)
        y1, y2 = max(0, cy-zoom), min(ny, cy+zoom)

        zoom_img = image[y1:y2, x1:x2]
        vmin2, vmax2 = np.nanpercentile(zoom_img, [1, 99.5])
        im2 = ax2.imshow(zoom_img, origin='lower', cmap='gray', vmin=vmin2, vmax=vmax2,
                         extent=[x1, x2, y1, y2])

        if tic_in:
            ax2.plot(tic_x, tic_y, 'r+', ms=20, mew=3, label='TIC (red)')
        if our_in:
            ax2.plot(our_x, our_y, 'gx', ms=20, mew=3, label='Our (green)')

        ax2.legend(loc='upper right')
        ax2.set_xlabel('X pixel')
        ax2.set_ylabel('Y pixel')
        ax2.set_title('Zoom on target region')
        plt.colorbar(im2, ax=ax2, label='Flux')

        plt.suptitle(f'TIC 610495795 Verification - TESS Sector 70', fontsize=14, y=1.02)
        plt.tight_layout()

        output_path = 'variable_stars/plots/TIC_610495795_verification.png'
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"\nSaved: {output_path}")
        plt.close()

        # Summary
        print("\n" + "=" * 60)
        print("SUMMARY")
        print("=" * 60)
        print(f"TIC 610495795: pixel ({tic_x:.1f}, {tic_y:.1f})")
        print(f"Our detection: pixel ({our_x:.1f}, {our_y:.1f})")
        print(f"Separation: {sep.arcsec:.1f}\" = {pix_sep:.1f} pixels")

        if tic_in and our_in:
            flux_ratio = our_flux / tic_flux if tic_flux > 0 else 0
            print(f"\nFlux at TIC position: {tic_flux:.1f}")
            print(f"Flux at Our position: {our_flux:.1f}")
            print(f"Flux ratio (Our/TIC): {flux_ratio:.2f}")

            if pix_sep < 2:
                print("\n>>> Positions nearly identical - SAME STAR!")
            elif pix_sep < 5:
                print("\n>>> Positions close - likely same star with WCS offset")
            else:
                print("\n>>> Positions differ significantly - check manually")

except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()

print("\nDone!")
