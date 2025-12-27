"""
Lightcurve Builder - Generate and analyze lightcurves from star catalog.

Features:
- Proper handling of missing data (NaN, not substitute)
- Quality flags for each measurement
- Normalization options
- Detrending
- Statistics and quality metrics
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from pathlib import Path

from .config import LIGHTCURVES_DIR, MIN_EPOCHS_FOR_STAR, ensure_directories
from .StarCatalog import StarCatalog


class Lightcurve:
    """
    Represents a single star's lightcurve with quality information.
    """

    def __init__(self, star_id: str, data: pd.DataFrame, star_info: dict = None):
        """
        Args:
            star_id: Unique identifier for the star
            data: DataFrame with columns: mjd, flux, flux_error, quality_flag, ...
            star_info: Additional star information (ra, dec, etc.)
        """
        self.star_id = star_id
        self.data = data.copy()
        self.star_info = star_info or {}

        # Ensure sorted by time
        if 'mjd' in self.data.columns:
            self.data = self.data.sort_values('mjd').reset_index(drop=True)

        # Calculate basic stats
        self._calculate_statistics()

    def _calculate_statistics(self):
        """Calculate lightcurve statistics."""
        good_mask = self.data['quality_flag'] == 0
        good_data = self.data[good_mask]

        self.stats = {
            'n_total': len(self.data),
            'n_good': good_mask.sum(),
            'completeness': good_mask.sum() / len(self.data) if len(self.data) > 0 else 0,
        }

        if len(good_data) > 0:
            flux = good_data['flux'].values
            median_flux = np.median(flux)

            self.stats.update({
                'mean_flux': np.mean(flux),
                'median_flux': median_flux,
                'std_flux': np.std(flux),
                'min_flux': np.min(flux),
                'max_flux': np.max(flux),
                'amplitude': (np.max(flux) - np.min(flux)) / median_flux if median_flux != 0 else 0,
                'median_error': np.median(good_data['flux_error']),
                'median_snr': np.median(good_data.get('snr', flux / good_data['flux_error'])),
            })

            # Time coverage
            mjd = good_data['mjd'].values
            self.stats['mjd_start'] = np.min(mjd)
            self.stats['mjd_end'] = np.max(mjd)
            self.stats['time_span_days'] = np.max(mjd) - np.min(mjd)

            # Variability metrics
            # Median Absolute Deviation (robust measure of scatter)
            self.stats['mad'] = np.median(np.abs(flux - median_flux))

            # Robust amplitude using percentiles (insensitive to outliers)
            p5, p95 = np.percentile(flux, [5, 95])
            self.stats['amplitude_robust'] = (p95 - p5) / median_flux if median_flux != 0 else 0

            # IQR-based amplitude
            p25, p75 = np.percentile(flux, [25, 75])
            self.stats['iqr'] = p75 - p25

            # Count outliers (points > 3 sigma from mean)
            if np.std(flux) > 0:
                z_scores = np.abs((flux - np.mean(flux)) / np.std(flux))
                self.stats['n_outliers'] = int(np.sum(z_scores > 3))
                self.stats['outlier_fraction'] = np.sum(z_scores > 3) / len(flux)
            else:
                self.stats['n_outliers'] = 0
                self.stats['outlier_fraction'] = 0.0

            # Reduced chi-squared (variability significance)
            if 'flux_error' in good_data.columns:
                errors = good_data['flux_error'].values
                chi2 = np.sum(((flux - np.mean(flux)) / errors) ** 2)
                dof = len(flux) - 1
                self.stats['reduced_chi2'] = chi2 / dof if dof > 0 else 0

            # Stetson J index (correlated variability)
            if len(flux) > 2:
                self.stats['stetson_j'] = self._calculate_stetson_j(
                    flux, good_data['flux_error'].values
                )
        else:
            self.stats.update({
                'mean_flux': np.nan,
                'median_flux': np.nan,
                'std_flux': np.nan,
                'amplitude': np.nan,
            })

    def _calculate_stetson_j(self, flux: np.ndarray, errors: np.ndarray) -> float:
        """Calculate Stetson J variability index."""
        n = len(flux)
        if n < 2:
            return 0.0

        mean_flux = np.mean(flux)
        residuals = (flux - mean_flux) / errors

        # Pair consecutive observations
        j_sum = 0.0
        for i in range(n - 1):
            p = residuals[i] * residuals[i + 1]
            j_sum += np.sign(p) * np.sqrt(np.abs(p))

        return j_sum * np.sqrt(n / (n - 1))

    def normalize(self, method: str = 'median') -> 'Lightcurve':
        """
        Return normalized lightcurve.

        Args:
            method: 'median', 'mean', or 'first'

        Returns:
            New Lightcurve object with normalized flux
        """
        good_mask = self.data['quality_flag'] == 0
        good_flux = self.data.loc[good_mask, 'flux']

        if len(good_flux) == 0:
            return self

        if method == 'median':
            norm_factor = np.median(good_flux)
        elif method == 'mean':
            norm_factor = np.mean(good_flux)
        elif method == 'first':
            norm_factor = good_flux.iloc[0]
        else:
            raise ValueError(f"Unknown normalization method: {method}")

        new_data = self.data.copy()
        new_data['flux'] = new_data['flux'] / norm_factor
        new_data['flux_error'] = new_data['flux_error'] / norm_factor

        return Lightcurve(self.star_id, new_data, self.star_info)

    def detrend(self, method: str = 'linear') -> 'Lightcurve':
        """
        Remove systematic trends from lightcurve.

        Args:
            method: 'linear', 'polynomial', or 'median_filter'

        Returns:
            New Lightcurve object with detrended flux
        """
        good_mask = self.data['quality_flag'] == 0
        good_data = self.data[good_mask].copy()

        if len(good_data) < 3:
            return self

        mjd = good_data['mjd'].values
        flux = good_data['flux'].values

        if method == 'linear':
            # Fit linear trend
            slope, intercept, _, _, _ = stats.linregress(mjd, flux)
            trend = slope * mjd + intercept

        elif method == 'polynomial':
            # Fit polynomial
            coeffs = np.polyfit(mjd, flux, deg=2)
            trend = np.polyval(coeffs, mjd)

        elif method == 'median_filter':
            # Running median
            from scipy.ndimage import median_filter
            trend = median_filter(flux, size=5)

        else:
            raise ValueError(f"Unknown detrending method: {method}")

        # Apply detrending
        new_data = self.data.copy()
        detrended_flux = flux - trend + np.median(flux)
        new_data.loc[good_mask, 'flux'] = detrended_flux

        return Lightcurve(self.star_id, new_data, self.star_info)

    def get_good_data(self) -> pd.DataFrame:
        """Return only good quality measurements."""
        return self.data[self.data['quality_flag'] == 0].copy()

    def to_arrays(self, good_only: bool = True) -> tuple:
        """
        Return time, flux, error as numpy arrays.

        Args:
            good_only: If True, only return good quality data

        Returns:
            Tuple of (mjd, flux, flux_error)
        """
        if good_only:
            data = self.get_good_data()
        else:
            data = self.data

        return (
            data['mjd'].values,
            data['flux'].values,
            data['flux_error'].values
        )

    def plot(self, ax=None, show_errors: bool = True,
             show_bad: bool = True, normalized: bool = False,
             **kwargs) -> plt.Axes:
        """
        Plot the lightcurve.

        Args:
            ax: Matplotlib axes (creates new if None)
            show_errors: Show error bars
            show_bad: Show bad quality points (in gray)
            normalized: Normalize before plotting
            **kwargs: Additional arguments for plt.errorbar

        Returns:
            Matplotlib axes
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(12, 4))

        lc = self.normalize() if normalized else self

        good_data = lc.get_good_data()
        bad_data = lc.data[lc.data['quality_flag'] != 0]

        # Plot good data
        if show_errors:
            ax.errorbar(good_data['mjd'], good_data['flux'],
                       yerr=good_data['flux_error'],
                       fmt='o', markersize=4, capsize=2,
                       label='Good', **kwargs)
        else:
            ax.plot(good_data['mjd'], good_data['flux'], 'o',
                   markersize=4, label='Good', **kwargs)

        # Plot bad data
        if show_bad and len(bad_data) > 0:
            ax.plot(bad_data['mjd'], bad_data['flux'], 'x',
                   color='gray', markersize=4, alpha=0.5, label='Flagged')

        ax.set_xlabel('MJD')
        ax.set_ylabel('Flux' + (' (normalized)' if normalized else ''))
        ax.set_title(f'Lightcurve: {self.star_id}')

        # Add stats annotation
        stats_text = (f"N={self.stats['n_good']}/{self.stats['n_total']} "
                     f"({self.stats['completeness']:.0%})")
        ax.annotate(stats_text, xy=(0.02, 0.98), xycoords='axes fraction',
                   ha='left', va='top', fontsize=9)

        return ax

    def save_plot(self, filename: str = None, **kwargs) -> str:
        """Save lightcurve plot to file."""
        ensure_directories()

        if filename is None:
            filename = f"lc_{self.star_id}.png"

        filepath = LIGHTCURVES_DIR / filename

        fig, ax = plt.subplots(figsize=(12, 4))
        self.plot(ax=ax, **kwargs)
        plt.tight_layout()
        plt.savefig(filepath, dpi=150)
        plt.close()

        return str(filepath)


class LightcurveCollection:
    """
    Collection of lightcurves with batch processing capabilities.
    """

    def __init__(self, catalog: StarCatalog = None):
        """
        Args:
            catalog: StarCatalog object to build lightcurves from
        """
        self.lightcurves = {}
        self.catalog = catalog

        if catalog is not None:
            self._build_from_catalog()

    def _build_from_catalog(self):
        """Build lightcurves from catalog."""
        lc_dict = self.catalog.get_all_lightcurves(min_detections=MIN_EPOCHS_FOR_STAR)

        for star_id, lc_df in lc_dict.items():
            star_info = self.catalog.stars.get(star_id, {})
            self.lightcurves[star_id] = Lightcurve(star_id, lc_df, star_info)

        print(f"Built {len(self.lightcurves)} lightcurves")

    def add(self, lightcurve: Lightcurve):
        """Add a lightcurve to the collection."""
        self.lightcurves[lightcurve.star_id] = lightcurve

    def get(self, star_id: str) -> Lightcurve:
        """Get lightcurve by star ID."""
        return self.lightcurves.get(star_id)

    def filter_by_quality(self, min_completeness: float = 0.5,
                          min_snr: float = 5.0) -> 'LightcurveCollection':
        """
        Filter lightcurves by quality metrics.

        Args:
            min_completeness: Minimum fraction of good epochs
            min_snr: Minimum median SNR

        Returns:
            New LightcurveCollection with filtered lightcurves
        """
        new_collection = LightcurveCollection()

        for star_id, lc in self.lightcurves.items():
            if lc.stats.get('completeness', 0) >= min_completeness:
                if lc.stats.get('median_snr', 0) >= min_snr:
                    new_collection.add(lc)

        return new_collection

    def filter_variable(self, min_amplitude: float = 0.01,
                        min_chi2: float = 2.0) -> 'LightcurveCollection':
        """
        Filter to keep only variable stars.

        Args:
            min_amplitude: Minimum fractional amplitude
            min_chi2: Minimum reduced chi-squared

        Returns:
            New LightcurveCollection with variable stars
        """
        new_collection = LightcurveCollection()

        for star_id, lc in self.lightcurves.items():
            if lc.stats.get('amplitude', 0) >= min_amplitude:
                if lc.stats.get('reduced_chi2', 0) >= min_chi2:
                    new_collection.add(lc)

        return new_collection

    def get_statistics_table(self) -> pd.DataFrame:
        """
        Get summary statistics for all lightcurves.

        Returns:
            DataFrame with one row per lightcurve
        """
        records = []

        for star_id, lc in self.lightcurves.items():
            record = {
                'star_id': star_id,
                'ra': lc.star_info.get('ra'),
                'dec': lc.star_info.get('dec'),
            }
            record.update(lc.stats)
            records.append(record)

        return pd.DataFrame(records)

    def normalize_all(self, method: str = 'median') -> 'LightcurveCollection':
        """Normalize all lightcurves."""
        new_collection = LightcurveCollection()

        for star_id, lc in self.lightcurves.items():
            new_collection.add(lc.normalize(method))

        return new_collection

    def save_all_plots(self, max_plots: int = None, **kwargs) -> list:
        """
        Save plots for all lightcurves.

        Args:
            max_plots: Maximum number of plots to save (None for all)
            **kwargs: Arguments passed to Lightcurve.save_plot

        Returns:
            List of saved file paths
        """
        paths = []

        for i, (star_id, lc) in enumerate(self.lightcurves.items()):
            if max_plots is not None and i >= max_plots:
                break

            path = lc.save_plot(**kwargs)
            paths.append(path)

        return paths

    def __len__(self):
        return len(self.lightcurves)

    def __iter__(self):
        return iter(self.lightcurves.values())


def build_lightcurves(catalog: StarCatalog = None,
                      min_detections: int = None) -> LightcurveCollection:
    """
    Convenience function to build lightcurves from catalog.

    Args:
        catalog: StarCatalog (loads from disk if None)
        min_detections: Minimum epochs for inclusion

    Returns:
        LightcurveCollection object
    """
    if catalog is None:
        catalog = StarCatalog.load()

    if min_detections is not None:
        # Update global config temporarily
        from . import config
        old_val = config.MIN_EPOCHS_FOR_STAR
        config.MIN_EPOCHS_FOR_STAR = min_detections

    collection = LightcurveCollection(catalog)

    if min_detections is not None:
        config.MIN_EPOCHS_FOR_STAR = old_val

    return collection
