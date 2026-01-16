"""Make clean 2-panel plot like top_periodic_candidates."""
import sys
sys.path.insert(0, 'src')

import numpy as np
import matplotlib.pyplot as plt
from tess.StreamingPipeline import convert_to_starcatalog
from tess.LightcurveBuilder import LightcurveCollection

print("Loading catalog...")
catalog = convert_to_starcatalog(70, '1', '1')
print("Building lightcurves...")
collection = LightcurveCollection(catalog)

# Get STAR_000088 (TIC 610538470)
print("Getting STAR_000088...")
lc = collection.get('STAR_000088')
data = lc.get_good_data()

btjd = data['btjd']
flux = data['flux']
flux_norm = flux / np.median(flux)

# Period from previous analysis
period = 0.87  # days

# Phase fold
phase = ((btjd - btjd.min()) % period) / period

print("Creating plot...")
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# Left: Raw lightcurve
ax1 = axes[0]
ax1.scatter(btjd, flux_norm, s=3, alpha=0.5, c='steelblue')
ax1.axhline(1.0, color='gray', linestyle='--', alpha=0.5)
ax1.set_xlabel('BTJD')
ax1.set_ylabel('Normalized Flux')
ax1.set_title('STAR_000088 | TIC 610538470 | Raw')

# Right: Phase-folded
ax2 = axes[1]

# Extended phase for visibility (-0.5 to 1.0)
phase_ext = np.concatenate([phase - 1, phase])
flux_ext = np.concatenate([flux_norm, flux_norm])

ax2.scatter(phase_ext, flux_ext, s=3, alpha=0.3, c='steelblue')

# Binned median (like orange line in good plot)
n_bins = 50
bins = np.linspace(-0.5, 1.0, n_bins + 1)
bin_centers = (bins[:-1] + bins[1:]) / 2
bin_medians = []
for i in range(n_bins):
    mask = (phase_ext >= bins[i]) & (phase_ext < bins[i+1])
    if mask.sum() > 0:
        bin_medians.append(np.median(flux_ext[mask]))
    else:
        bin_medians.append(np.nan)

ax2.plot(bin_centers, bin_medians, 'o-', color='darkorange', markersize=4, linewidth=1.5, label='Binned median')

ax2.set_xlabel('Phase')
ax2.set_ylabel('Normalized Flux')
ax2.set_title(f'Phase-folded | P = {period:.2f} days')
ax2.set_xlim(-0.5, 1.0)
ax2.legend(loc='upper right', fontsize=8)

plt.tight_layout()
output_path = 'variable_stars/TIC_610538470/TIC_610538470_clean.png'
plt.savefig(output_path, dpi=150, bbox_inches='tight')
print(f'Saved: {output_path}')
plt.close()
