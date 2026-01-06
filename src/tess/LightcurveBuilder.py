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
            data: DataFrame with columns: btjd, flux, flux_error, quality_flag, ...
            star_info: Additional star information (ra, dec, etc.)
        """
        self.star_id = star_id
        self.data = data.copy()
        self.star_info = star_info or {}

        # Backwards compatibility: handle column name variations
        # quality vs quality_flag
        if 'quality' in self.data.columns and 'quality_flag' not in self.data.columns:
            self.data = self.data.rename(columns={'quality': 'quality_flag'})

        # mjd vs btjd - merge the two columns (some rows have mjd, others have btjd)
        if 'mjd' in self.data.columns:
            if 'btjd' not in self.data.columns:
                self.data = self.data.rename(columns={'mjd': 'btjd'})
            else:
                # Fill missing btjd values with mjd values
                self.data['btjd'] = self.data['btjd'].fillna(self.data['mjd'])

        # Ensure sorted by time
        if 'btjd' in self.data.columns:
            self.data = self.data.sort_values('btjd').reset_index(drop=True)

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
            btjd = good_data['btjd'].values
            self.stats['btjd_start'] = np.min(btjd)
            self.stats['btjd_end'] = np.max(btjd)
            self.stats['time_span_days'] = np.max(btjd) - np.min(btjd)

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

        btjd = good_data['btjd'].values
        flux = good_data['flux'].values

        if method == 'linear':
            # Fit linear trend
            slope, intercept, _, _, _ = stats.linregress(btjd, flux)
            trend = slope * btjd + intercept

        elif method == 'polynomial':
            # Fit polynomial
            coeffs = np.polyfit(btjd, flux, deg=2)
            trend = np.polyval(coeffs, btjd)

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

    def get_good_data(self, exclude_artifacts: bool = False, sector: int = None) -> pd.DataFrame:
        """
        Return only good quality measurements.

        Args:
            exclude_artifacts: If True, also exclude known artifact windows
            sector: Sector number (required if exclude_artifacts=True)

        Returns:
            DataFrame with good quality data
        """
        mask = self.data['quality_flag'] == 0

        if exclude_artifacts:
            if sector is None:
                raise ValueError("sector is required when exclude_artifacts=True")
            from .config import get_artifact_windows
            for start, end, _ in get_artifact_windows(sector):
                mask = mask & ~((self.data['btjd'] >= start) & (self.data['btjd'] <= end))

        return self.data[mask].copy()

    def filter_artifacts(self, sector: int) -> 'Lightcurve':
        """
        Return a new Lightcurve with artifact windows removed.

        Args:
            sector: TESS sector number

        Returns:
            New Lightcurve instance with artifacts filtered out
        """
        from .config import get_artifact_windows

        mask = pd.Series(True, index=self.data.index)
        windows = get_artifact_windows(sector)

        for start, end, desc in windows:
            window_mask = (self.data['btjd'] >= start) & (self.data['btjd'] <= end)
            n_removed = window_mask.sum()
            if n_removed > 0:
                print(f"Filtering {n_removed} points: {desc} (BTJD {start}-{end})")
            mask = mask & ~window_mask

        filtered_data = self.data[mask].copy()
        return Lightcurve(self.star_id, filtered_data, self.star_info)

    def to_arrays(self, good_only: bool = True) -> tuple:
        """
        Return time, flux, error as numpy arrays.

        Args:
            good_only: If True, only return good quality data

        Returns:
            Tuple of (btjd, flux, flux_error)
        """
        if good_only:
            data = self.get_good_data()
        else:
            data = self.data

        return (
            data['btjd'].values,
            data['flux'].values,
            data['flux_error'].values
        )

    def fold(self, period: float, t0: float = None) -> pd.DataFrame:
        """
        Phase-fold the lightcurve at a given period.

        Args:
            period: Folding period in days
            t0: Reference time (epoch). If None, uses first observation.

        Returns:
            DataFrame with 'phase', 'flux', 'flux_error' columns, sorted by phase
        """
        good_data = self.get_good_data()
        if len(good_data) == 0:
            return pd.DataFrame()

        times = good_data['btjd'].values
        flux = good_data['flux'].values
        errors = good_data['flux_error'].values

        if t0 is None:
            t0 = times[0]

        # Calculate phase (0 to 1)
        phase = ((times - t0) / period) % 1.0

        # Center phase around 0 (-0.5 to 0.5) for better visualization
        phase = np.where(phase > 0.5, phase - 1.0, phase)

        result = pd.DataFrame({
            'phase': phase,
            'flux': flux,
            'flux_error': errors,
            'btjd': times
        }).sort_values('phase').reset_index(drop=True)

        return result

    def plot_folded(self, period: float, t0: float = None, ax=None,
                    normalized: bool = True, bin_phase: bool = False,
                    n_bins: int = 50) -> plt.Axes:
        """
        Plot phase-folded lightcurve.

        Args:
            period: Folding period in days
            t0: Reference epoch
            ax: Matplotlib axes
            normalized: Normalize flux before plotting
            bin_phase: Bin the folded data for cleaner visualization
            n_bins: Number of phase bins

        Returns:
            Matplotlib axes
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 5))

        lc = self.normalize() if normalized else self
        folded = lc.fold(period, t0)

        if len(folded) == 0:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center')
            return ax

        if bin_phase:
            # Bin the data
            bins = np.linspace(-0.5, 0.5, n_bins + 1)
            bin_centers = (bins[:-1] + bins[1:]) / 2
            bin_flux = []
            bin_err = []

            for i in range(n_bins):
                mask = (folded['phase'] >= bins[i]) & (folded['phase'] < bins[i+1])
                if mask.sum() > 0:
                    bin_flux.append(np.median(folded.loc[mask, 'flux']))
                    bin_err.append(np.std(folded.loc[mask, 'flux']) / np.sqrt(mask.sum()))
                else:
                    bin_flux.append(np.nan)
                    bin_err.append(np.nan)

            ax.errorbar(bin_centers, bin_flux, yerr=bin_err, fmt='o-',
                       markersize=6, capsize=2, label='Binned')
        else:
            ax.scatter(folded['phase'], folded['flux'], s=10, alpha=0.6)

        ax.axhline(1.0 if normalized else np.median(folded['flux']),
                  color='gray', linestyle='--', alpha=0.5)
        ax.set_xlabel('Phase')
        ax.set_ylabel('Normalized Flux' if normalized else 'Flux')
        ax.set_title(f'{self.star_id} | Period = {period:.4f} days')
        ax.set_xlim(-0.5, 0.5)

        return ax

    def bls_periodogram(self, min_period: float = 0.5, max_period: float = 15.0,
                        n_periods: int = 5000) -> dict:
        """
        Compute Box Least Squares periodogram for transit/eclipse detection.

        BLS is optimized for finding periodic box-shaped dips (transits, eclipses).

        Args:
            min_period: Minimum period to search (days)
            max_period: Maximum period to search (days)
            n_periods: Number of trial periods

        Returns:
            dict with 'periods', 'power', 'best_period', 'best_t0', 'best_duration', 'best_depth'
        """
        from astropy.timeseries import BoxLeastSquares

        good_data = self.get_good_data()
        if len(good_data) < 20:
            return {'periods': [], 'power': [], 'best_period': None}

        times = good_data['btjd'].values
        flux = good_data['flux'].values
        errors = good_data['flux_error'].values

        # Normalize
        median_flux = np.median(flux)
        norm_flux = flux / median_flux
        norm_err = errors / median_flux

        # Create BLS model
        model = BoxLeastSquares(times, norm_flux, dy=norm_err)

        # Search periods
        periods = np.linspace(min_period, max_period, n_periods)
        periodogram = model.power(periods, duration=np.linspace(0.01, 0.2, 10))

        # Find best period
        best_idx = np.argmax(periodogram.power)
        best_period = periodogram.period[best_idx]

        # Get transit parameters at best period
        stats = model.compute_stats(best_period,
                                    periodogram.duration[best_idx],
                                    periodogram.transit_time[best_idx])

        return {
            'periods': periodogram.period,
            'power': periodogram.power,
            'best_period': best_period,
            'best_t0': periodogram.transit_time[best_idx],
            'best_duration': periodogram.duration[best_idx],
            'best_depth': stats['depth'][0] if 'depth' in stats else None,
            'snr': periodogram.power[best_idx]
        }

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
            ax.errorbar(good_data['btjd'], good_data['flux'],
                       yerr=good_data['flux_error'],
                       fmt='o', markersize=4, capsize=2,
                       label='Good', **kwargs)
        else:
            ax.plot(good_data['btjd'], good_data['flux'], 'o',
                   markersize=4, label='Good', **kwargs)

        # Plot bad data
        if show_bad and len(bad_data) > 0:
            ax.plot(bad_data['btjd'], bad_data['flux'], 'x',
                   color='gray', markersize=4, alpha=0.5, label='Flagged')

        ax.set_xlabel('BTJD')
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
