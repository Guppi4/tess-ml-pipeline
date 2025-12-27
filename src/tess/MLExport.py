"""
ML Export - Prepare lightcurve data for machine learning classification.

Exports data in formats suitable for:
- Time series classification (LSTM, Transformer)
- Feature-based classification (Random Forest, XGBoost)
- Anomaly detection

Includes:
- Data quality metrics
- Feature extraction
- Multiple output formats
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from scipy import stats
from astropy.timeseries import LombScargle

from config import BASE_DIR, ensure_directories
from LightcurveBuilder import Lightcurve, LightcurveCollection


# Output directory
ML_DATA_DIR = BASE_DIR / "ml_data"


def ensure_ml_directories():
    """Create ML data directories."""
    ML_DATA_DIR.mkdir(parents=True, exist_ok=True)
    (ML_DATA_DIR / "features").mkdir(exist_ok=True)
    (ML_DATA_DIR / "timeseries").mkdir(exist_ok=True)


class FeatureExtractor:
    """
    Extract features from lightcurves for ML classification.
    """

    @staticmethod
    def extract_statistical_features(lc: Lightcurve) -> Dict:
        """
        Extract statistical features from a lightcurve.

        Returns:
            Dictionary of feature name -> value
        """
        good_data = lc.get_good_data()

        if len(good_data) < 3:
            return {}

        flux = good_data['flux'].values
        errors = good_data['flux_error'].values
        mjd = good_data['mjd'].values

        features = {}

        # Basic statistics
        features['mean'] = np.mean(flux)
        features['median'] = np.median(flux)
        features['std'] = np.std(flux)
        features['skewness'] = stats.skew(flux)
        features['kurtosis'] = stats.kurtosis(flux)

        # Robust statistics
        features['mad'] = np.median(np.abs(flux - np.median(flux)))
        features['iqr'] = np.percentile(flux, 75) - np.percentile(flux, 25)

        # Percentiles
        for p in [5, 10, 25, 75, 90, 95]:
            features[f'percentile_{p}'] = np.percentile(flux, p)

        # Range and amplitude
        features['range'] = np.max(flux) - np.min(flux)
        features['amplitude'] = features['range'] / features['median'] if features['median'] != 0 else 0

        # Variability indices
        features['rms'] = np.sqrt(np.mean(flux ** 2))
        features['coefficient_of_variation'] = features['std'] / features['mean'] if features['mean'] != 0 else 0

        # Beyond N-sigma statistics
        for n in [1, 2, 3]:
            beyond = np.sum(np.abs(flux - features['mean']) > n * features['std'])
            features[f'beyond_{n}sigma'] = beyond / len(flux)

        # Weighted mean and variance (using errors)
        if np.all(errors > 0):
            weights = 1 / errors ** 2
            features['weighted_mean'] = np.sum(flux * weights) / np.sum(weights)
            features['weighted_std'] = np.sqrt(np.sum(weights * (flux - features['weighted_mean']) ** 2) / np.sum(weights))

        # Time-domain features
        features['time_span'] = mjd[-1] - mjd[0]
        features['n_observations'] = len(flux)

        # Flux differences
        diff = np.diff(flux)
        features['mean_diff'] = np.mean(np.abs(diff))
        features['max_diff'] = np.max(np.abs(diff))

        # Autocorrelation at lag 1
        if len(flux) > 1:
            features['autocorr_lag1'] = np.corrcoef(flux[:-1], flux[1:])[0, 1]

        return features

    @staticmethod
    def extract_periodic_features(lc: Lightcurve, n_frequencies: int = 5) -> Dict:
        """
        Extract periodicity-related features using Lomb-Scargle periodogram.

        Lomb-Scargle is correct for non-uniformly sampled data (unlike FFT).
        This is important for TESS FFI which has gaps during data downlink.

        Args:
            lc: Lightcurve object
            n_frequencies: Number of top frequencies to return

        Returns:
            Dictionary of periodic features
        """
        good_data = lc.get_good_data()

        if len(good_data) < 10:
            return {}

        flux = good_data['flux'].values
        mjd = good_data['mjd'].values

        features = {}

        # Normalize flux
        flux_norm = flux - np.mean(flux)

        # Lomb-Scargle periodogram (handles non-uniform sampling correctly)
        # Frequency range: from 1/time_span to Nyquist (1/2*median_cadence)
        time_span = mjd[-1] - mjd[0]
        if time_span <= 0:
            return {}

        median_cadence = np.median(np.diff(mjd))
        if median_cadence <= 0:
            return {}

        min_freq = 1.0 / time_span
        max_freq = 1.0 / (2 * median_cadence)  # Nyquist frequency

        # Compute Lomb-Scargle periodogram
        ls = LombScargle(mjd, flux_norm)
        freqs, power = ls.autopower(
            minimum_frequency=min_freq,
            maximum_frequency=max_freq
        )

        if len(power) == 0:
            return {}

        # Total power
        features['total_power'] = float(np.sum(power))

        # Top frequencies
        top_indices = np.argsort(power)[-n_frequencies:][::-1]
        for i, idx in enumerate(top_indices):
            features[f'freq_{i+1}'] = float(freqs[idx])
            features[f'power_{i+1}'] = float(power[idx])
            features[f'period_{i+1}'] = float(1 / freqs[idx]) if freqs[idx] != 0 else np.inf

        # Ratio of power in top frequency to total
        if features['total_power'] > 0:
            features['power_ratio_top'] = float(power[top_indices[0]] / features['total_power'])
        else:
            features['power_ratio_top'] = 0.0

        return features

    @staticmethod
    def extract_all_features(lc: Lightcurve) -> Dict:
        """
        Extract all features from a lightcurve.

        Returns:
            Dictionary of all features
        """
        features = {}

        # Add star info
        features['star_id'] = lc.star_id
        features['ra'] = lc.star_info.get('ra', np.nan)
        features['dec'] = lc.star_info.get('dec', np.nan)

        # Quality metrics from stats
        features['completeness'] = lc.stats.get('completeness', 0)
        features['n_good'] = lc.stats.get('n_good', 0)
        features['median_snr'] = lc.stats.get('median_snr', 0)

        # Statistical features
        stat_features = FeatureExtractor.extract_statistical_features(lc)
        features.update(stat_features)

        # Periodic features
        periodic_features = FeatureExtractor.extract_periodic_features(lc)
        features.update(periodic_features)

        return features


class MLDataset:
    """
    Dataset of lightcurves prepared for ML.
    """

    def __init__(self, lightcurves: LightcurveCollection = None):
        """
        Args:
            lightcurves: LightcurveCollection to process
        """
        self.lightcurves = lightcurves
        self.features_df = None
        self.timeseries_data = None
        self.metadata = {}

    def extract_features(self, min_completeness: float = 0.5,
                         min_snr: float = 3.0) -> pd.DataFrame:
        """
        Extract features for all lightcurves.

        Args:
            min_completeness: Minimum data completeness to include
            min_snr: Minimum median SNR to include

        Returns:
            DataFrame with one row per star, columns are features
        """
        print("Extracting features...")

        feature_list = []

        for lc in self.lightcurves:
            # Quality filter
            if lc.stats.get('completeness', 0) < min_completeness:
                continue
            if lc.stats.get('median_snr', 0) < min_snr:
                continue

            features = FeatureExtractor.extract_all_features(lc)
            feature_list.append(features)

        self.features_df = pd.DataFrame(feature_list)
        print(f"Extracted features for {len(self.features_df)} stars")

        return self.features_df

    def prepare_timeseries(self, max_length: int = 200,
                           normalize: bool = True,
                           padding: str = 'zero') -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Prepare time series data for sequence models (LSTM, Transformer).

        Args:
            max_length: Maximum sequence length (pad/truncate to this)
            normalize: Whether to normalize each lightcurve
            padding: 'zero' for zero-padding, 'nan' for NaN-padding

        Returns:
            Tuple of:
                - X: array of shape (n_samples, max_length, n_features)
                - mask: boolean array of shape (n_samples, max_length) - True where data exists
                - star_ids: list of star IDs
        """
        print("Preparing time series data...")

        n_samples = len(self.lightcurves.lightcurves)

        # Features per timestep: flux, flux_error, quality_flag
        n_features = 3

        X = np.zeros((n_samples, max_length, n_features))
        mask = np.zeros((n_samples, max_length), dtype=bool)
        star_ids = []

        for i, lc in enumerate(self.lightcurves):
            star_ids.append(lc.star_id)

            # Get data
            mjd, flux, error = lc.to_arrays(good_only=False)

            if normalize:
                median_flux = np.nanmedian(flux)
                if median_flux != 0 and not np.isnan(median_flux):
                    flux = flux / median_flux
                    error = error / median_flux

            # Get quality flags
            quality = lc.data['quality_flag'].values

            # Determine length
            n_points = min(len(flux), max_length)

            # Fill arrays
            X[i, :n_points, 0] = flux[:n_points]
            X[i, :n_points, 1] = error[:n_points]
            X[i, :n_points, 2] = quality[:n_points]
            mask[i, :n_points] = (quality[:n_points] == 0)

            # Replace NaN with padding value
            if padding == 'zero':
                X[i] = np.nan_to_num(X[i], nan=0.0)

        self.timeseries_data = {
            'X': X,
            'mask': mask,
            'star_ids': star_ids,
            'max_length': max_length,
            'normalized': normalize
        }

        print(f"Prepared {n_samples} time series, shape: {X.shape}")

        return X, mask, star_ids

    def save(self, name: str = "dataset") -> Dict[str, str]:
        """
        Save dataset to disk.

        Args:
            name: Base name for saved files

        Returns:
            Dictionary of saved file paths
        """
        ensure_ml_directories()

        saved_files = {}

        # Save features
        if self.features_df is not None:
            features_path = ML_DATA_DIR / "features" / f"{name}_features.csv"
            self.features_df.to_csv(features_path, index=False)
            saved_files['features_csv'] = str(features_path)

            # Also save as numpy array (for faster loading)
            features_np_path = ML_DATA_DIR / "features" / f"{name}_features.npy"
            np.save(features_np_path, self.features_df.values)
            saved_files['features_npy'] = str(features_np_path)

            # Save column names
            columns_path = ML_DATA_DIR / "features" / f"{name}_columns.json"
            with open(columns_path, 'w') as f:
                json.dump(list(self.features_df.columns), f)
            saved_files['columns'] = str(columns_path)

        # Save time series
        if self.timeseries_data is not None:
            ts_path = ML_DATA_DIR / "timeseries" / f"{name}_timeseries.npz"
            np.savez(ts_path,
                     X=self.timeseries_data['X'],
                     mask=self.timeseries_data['mask'])
            saved_files['timeseries'] = str(ts_path)

            # Save star IDs
            ids_path = ML_DATA_DIR / "timeseries" / f"{name}_star_ids.json"
            with open(ids_path, 'w') as f:
                json.dump(self.timeseries_data['star_ids'], f)
            saved_files['star_ids'] = str(ids_path)

        # Save metadata
        self.metadata = {
            'name': name,
            'n_samples': len(self.lightcurves.lightcurves),
            'saved_files': saved_files
        }

        metadata_path = ML_DATA_DIR / f"{name}_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(self.metadata, f, indent=2)
        saved_files['metadata'] = str(metadata_path)

        print(f"\nDataset saved:")
        for key, path in saved_files.items():
            print(f"  {key}: {path}")

        return saved_files

    @classmethod
    def load(cls, name: str = "dataset") -> 'MLDataset':
        """
        Load dataset from disk.

        Args:
            name: Base name of saved files

        Returns:
            MLDataset object
        """
        dataset = cls()

        # Load metadata
        metadata_path = ML_DATA_DIR / f"{name}_metadata.json"
        with open(metadata_path, 'r') as f:
            dataset.metadata = json.load(f)

        # Load features
        features_path = ML_DATA_DIR / "features" / f"{name}_features.csv"
        if features_path.exists():
            dataset.features_df = pd.read_csv(features_path)

        # Load time series
        ts_path = ML_DATA_DIR / "timeseries" / f"{name}_timeseries.npz"
        if ts_path.exists():
            ts_data = np.load(ts_path)
            ids_path = ML_DATA_DIR / "timeseries" / f"{name}_star_ids.json"
            with open(ids_path, 'r') as f:
                star_ids = json.load(f)

            dataset.timeseries_data = {
                'X': ts_data['X'],
                'mask': ts_data['mask'],
                'star_ids': star_ids
            }

        return dataset


def export_for_ml(lightcurves: LightcurveCollection,
                  name: str = "tess_lightcurves",
                  min_completeness: float = 0.5,
                  min_snr: float = 3.0,
                  max_sequence_length: int = 200) -> Dict[str, str]:
    """
    Convenience function to export lightcurves for ML.

    Args:
        lightcurves: LightcurveCollection to export
        name: Dataset name
        min_completeness: Minimum data completeness
        min_snr: Minimum signal-to-noise ratio
        max_sequence_length: Maximum length for time series

    Returns:
        Dictionary of saved file paths
    """
    dataset = MLDataset(lightcurves)

    # Extract features
    dataset.extract_features(min_completeness, min_snr)

    # Prepare time series
    dataset.prepare_timeseries(max_length=max_sequence_length, normalize=True)

    # Save
    return dataset.save(name)


def get_train_test_split(features_df: pd.DataFrame,
                         test_size: float = 0.2,
                         random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split features into train and test sets.

    Returns:
        Tuple of (train_df, test_df)
    """
    from sklearn.model_selection import train_test_split

    train_df, test_df = train_test_split(
        features_df,
        test_size=test_size,
        random_state=random_state
    )

    return train_df, test_df
