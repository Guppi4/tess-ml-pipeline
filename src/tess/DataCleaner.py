"""
Data Cleaner for TESS Photometry

Cleans photometry data using masks (not deletion) to preserve data for different use cases:
- Transits/EA: minimal filtering (keep negative dips)
- Variable stars: balanced filtering
- ML: strict filtering with feature export

Architecture:
- photometry_with_masks.parquet: all data + masks + flux_cm
- epoch_qc.parquet: quality metrics per epoch
- star_qc.parquet: quality metrics per star (with TIC ID)
- ml_features.parquet: tabular features for ML

Mask bits (uint16):
    bit 0 (1):    quality != 0
    bit 1 (2):    NaN/inf/flux_error <= 0
    bit 2 (4):    bad_epoch (Moon, momentum dump, scattered light)
    bit 3 (8):    outlier_positive (cosmic ray, glitch)
    bit 4 (16):   outlier_negative (rarely used)
    bit 5 (32):   edge (first/last 0.5 days)
    bit 6 (64):   artifact_window (from config)
    bit 7 (128):  low_snr (flux/flux_error < threshold)
"""

import json
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime

warnings.filterwarnings('ignore', category=FutureWarning)

from .config import (
    get_tess_sector_dir, get_artifact_windows, is_in_artifact_window
)


# Mask bit definitions
MASK_QUALITY = 1        # bit 0: quality != 0
MASK_INVALID = 2        # bit 1: NaN/inf/flux_error <= 0
MASK_BAD_EPOCH = 4      # bit 2: bad epoch (scattered light, momentum dump)
MASK_OUTLIER_POS = 8    # bit 3: positive outlier (cosmic ray)
MASK_OUTLIER_NEG = 16   # bit 4: negative outlier (rarely used)
MASK_EDGE = 32          # bit 5: edge of observation
MASK_ARTIFACT = 64      # bit 6: known artifact window
MASK_LOW_SNR = 128      # bit 7: low signal-to-noise


class DataCleaner:
    """
    Clean TESS photometry data using masks.

    Usage:
        cleaner = DataCleaner(sector=61, camera=4, ccd=2)
        cleaner.run()

        # Access results
        df = cleaner.get_cleaned_data()
        epoch_qc = cleaner.epoch_qc
        star_qc = cleaner.star_qc
    """

    def __init__(self, sector: int, camera: int, ccd: int):
        """
        Initialize DataCleaner.

        Args:
            sector: TESS sector number
            camera: Camera number (1-4)
            ccd: CCD number (1-4)
        """
        self.sector = sector
        self.camera = camera
        self.ccd = ccd
        self.session_id = f"s{sector:04d}_{camera}-{ccd}"

        # Paths
        self.data_dir = get_tess_sector_dir(sector, camera, ccd)
        self.qc_dir = self.data_dir / "qc"
        self.cleaned_dir = self.data_dir / "cleaned"
        self.ml_dir = self.data_dir / "ml"

        # Data containers
        self.df: Optional[pd.DataFrame] = None
        self.catalog: Optional[Dict] = None
        self.epoch_qc: Optional[pd.DataFrame] = None
        self.star_qc: Optional[pd.DataFrame] = None
        self.quiet_stars: Optional[List[str]] = None
        self.common_mode: Optional[pd.Series] = None

        # Parameters (can be adjusted)
        self.params = {
            'edge_days': 0.5,           # trim first/last N days
            'outlier_sigma_pos': 5.0,   # positive outlier threshold
            'outlier_sigma_neg': 10.0,  # negative outlier threshold (less strict!)
            'snr_threshold': 3.0,       # minimum SNR
            'quiet_star_mad_percentile': 30,  # percentile for quiet stars
            'quiet_star_completeness': 0.9,   # min completeness for quiet stars
            'bad_epoch_scatter_sigma': 5.0,   # epoch scatter threshold
            'bad_epoch_median_sigma': 3.0,    # epoch median shift threshold
            'momentum_dump_gap_hours': 3.0,   # gap threshold for momentum dumps
        }

    def load_data(self) -> bool:
        """Load photometry and catalog data."""
        # Try parquet first, then CSV
        parquet_path = self.data_dir / f"{self.session_id}_photometry.parquet"
        csv_path = self.data_dir / f"{self.session_id}_photometry.csv"

        if parquet_path.exists():
            self.df = pd.read_parquet(parquet_path)
            print(f"Loaded {len(self.df):,} rows from {parquet_path.name}")
        elif csv_path.exists():
            self.df = pd.read_csv(csv_path)
            print(f"Loaded {len(self.df):,} rows from {csv_path.name}")
        else:
            raise FileNotFoundError(f"No photometry file found in {self.data_dir}")

        # Load catalog
        catalog_path = self.data_dir / f"{self.session_id}_catalog.json"
        if catalog_path.exists():
            with open(catalog_path, 'r') as f:
                self.catalog = json.load(f)
            print(f"Loaded catalog with {len(self.catalog.get('stars', {}))} stars")
        else:
            # Try checkpoint
            checkpoint_path = self.data_dir / "checkpoints" / f"{self.session_id}_checkpoint.json"
            if checkpoint_path.exists():
                with open(checkpoint_path, 'r') as f:
                    data = json.load(f)
                self.catalog = {
                    'stars': data.get('star_catalog', {}),
                    'n_stars': len(data.get('star_catalog', {}))
                }
                print(f"Loaded catalog from checkpoint with {self.catalog['n_stars']} stars")
            else:
                print("Warning: No catalog found, TIC matching will not be available")
                self.catalog = {'stars': {}}

        # Rename flux to flux_raw if needed
        if 'flux' in self.df.columns and 'flux_raw' not in self.df.columns:
            self.df = self.df.rename(columns={'flux': 'flux_raw'})

        return True

    def has_tic_ids(self) -> bool:
        """Check if catalog has TIC IDs."""
        if not self.catalog or 'stars' not in self.catalog:
            return False

        stars = self.catalog['stars']
        if not stars:
            return False

        # Check first star
        first_star = next(iter(stars.values()))
        tic_id = first_star.get('tic_id')

        # TIC ID exists and is not just the internal star_id
        return tic_id is not None and not str(tic_id).startswith('STAR_')

    def add_tic_ids(self) -> pd.DataFrame:
        """
        Add TIC IDs to catalog using FFIStarFinder function.

        Returns:
            DataFrame with star catalog including TIC IDs
        """
        from .FFIStarFinder import add_tic_ids

        if self.catalog is None:
            raise ValueError("No catalog loaded. Call load_data() first.")

        stars = self.catalog.get('stars', {})
        if not stars:
            print("No stars in catalog")
            return pd.DataFrame()

        # Convert to DataFrame
        records = []
        for star_id, info in stars.items():
            records.append({
                'star_id': star_id,
                'ra': info.get('ra'),
                'dec': info.get('dec'),
                'ref_x': info.get('ref_x'),
                'ref_y': info.get('ref_y'),
            })

        df = pd.DataFrame(records)

        # Filter valid coordinates
        valid = df[df['ra'].notna() & df['dec'].notna()].copy()
        print(f"Stars with valid coordinates: {len(valid)}/{len(df)}")

        if len(valid) == 0:
            print("No stars with valid coordinates for TIC matching")
            return df

        # Use existing function
        print("Querying TIC catalog (this may take a while)...")
        valid = add_tic_ids(valid)

        # Update catalog with TIC info
        tic_count = 0
        for _, row in valid.iterrows():
            star_id = row['star_id']
            tic = row.get('tic')
            if star_id in self.catalog['stars']:
                if tic is not None and not pd.isna(tic):
                    self.catalog['stars'][star_id]['tic_id'] = str(int(tic))
                    tic_count += 1
                else:
                    self.catalog['stars'][star_id]['tic_id'] = star_id

        print(f"TIC matches: {tic_count}/{len(valid)} ({tic_count/len(valid)*100:.1f}%)")

        # Save updated catalog
        catalog_path = self.data_dir / f"{self.session_id}_catalog.json"
        with open(catalog_path, 'w') as f:
            json.dump(self.catalog, f, indent=2, default=str)
        print(f"Saved updated catalog to {catalog_path.name}")

        return valid

    def _add_tic_to_dataframe(self):
        """Add tic_id column to main dataframe from catalog."""
        if self.df is None or self.catalog is None:
            return

        stars = self.catalog.get('stars', {})
        tic_map = {sid: info.get('tic_id', sid) for sid, info in stars.items()}
        self.df['tic_id'] = self.df['star_id'].map(tic_map)

    def compute_epoch_qc(self) -> pd.DataFrame:
        """
        Compute quality metrics per epoch.

        Detects:
        - Momentum dumps (large time gaps)
        - Scattered light (elevated median)
        - Bad epochs (high scatter, low valid count)

        Returns:
            DataFrame with epoch QC metrics
        """
        if self.df is None:
            raise ValueError("No data loaded. Call load_data() first.")

        df = self.df.copy()

        # Basic epoch stats
        # Use 'size' for n_total (count excludes NaN, size doesn't)
        epoch_stats = df.groupby('epoch').agg({
            'btjd': 'first',
            'flux_raw': ['median', 'mean', 'std'],
            'flux_error': 'median',
            'quality': lambda x: (x == 0).sum(),
            'star_id': 'size'  # size counts all rows including NaN
        }).reset_index()

        epoch_stats.columns = [
            'epoch', 'btjd', 'median_flux', 'mean_flux', 'std_flux',
            'median_error', 'n_valid', 'n_total'
        ]

        # Handle epochs with n_total=0 or NaN
        epoch_stats['frac_good'] = np.where(
            epoch_stats['n_total'] > 0,
            epoch_stats['n_valid'] / epoch_stats['n_total'],
            0.0  # Set to 0 if no data
        )

        # Detect momentum dumps (large gaps)
        epoch_stats = epoch_stats.sort_values('btjd').reset_index(drop=True)
        epoch_stats['time_gap'] = epoch_stats['btjd'].diff()
        median_gap = epoch_stats['time_gap'].median()
        gap_threshold = self.params['momentum_dump_gap_hours'] / 24.0  # convert to days
        epoch_stats['is_after_gap'] = epoch_stats['time_gap'] > gap_threshold

        # Compute robust stats using only good quality data
        good_df = df[df['quality'] == 0].copy()
        if len(good_df) > 0:
            good_stats = good_df.groupby('epoch').agg({
                'flux_raw': ['median', lambda x: np.median(np.abs(x - np.median(x)))]
            }).reset_index()
            good_stats.columns = ['epoch', 'median_flux_good', 'mad_flux']
            epoch_stats = epoch_stats.merge(good_stats, on='epoch', how='left')
        else:
            epoch_stats['median_flux_good'] = epoch_stats['median_flux']
            epoch_stats['mad_flux'] = epoch_stats['std_flux']

        # Normalize median flux (to detect scattered light)
        # Use nanmedian to handle epochs with no good data
        overall_median = np.nanmedian(epoch_stats['median_flux_good'])
        epoch_stats['median_flux_norm'] = epoch_stats['median_flux_good'] / overall_median

        # Normalize scatter
        overall_mad = np.nanmedian(epoch_stats['mad_flux'])
        epoch_stats['scatter_norm'] = epoch_stats['mad_flux'] / overall_mad

        # Detect bad epochs
        scatter_sigma = self.params['bad_epoch_scatter_sigma']
        median_sigma = self.params['bad_epoch_median_sigma']

        # Use nanmedian to handle NaN values in scatter_norm and median_flux_norm
        scatter_norm_valid = epoch_stats['scatter_norm'].dropna()
        median_norm_valid = epoch_stats['median_flux_norm'].dropna()

        mad_of_mad = np.nanmedian(np.abs(scatter_norm_valid - 1.0)) if len(scatter_norm_valid) > 0 else 1.0
        mad_of_median = np.nanmedian(np.abs(median_norm_valid - 1.0)) if len(median_norm_valid) > 0 else 1.0

        # Handle NaN: treat NaN scatter_norm/median_flux_norm as bad
        is_high_scatter = (epoch_stats['scatter_norm'] > 1.0 + scatter_sigma * mad_of_mad) | epoch_stats['scatter_norm'].isna()
        is_shifted = (np.abs(epoch_stats['median_flux_norm'] - 1.0) > median_sigma * mad_of_median) | epoch_stats['median_flux_norm'].isna()
        is_low_valid = epoch_stats['frac_good'] < 0.5
        is_after_gap = epoch_stats['is_after_gap']
        is_no_data = epoch_stats['n_valid'] == 0  # epochs with no good measurements

        # Mark bad epochs
        epoch_stats['is_bad'] = is_high_scatter | is_shifted | is_low_valid | is_after_gap | is_no_data

        # Reason for being bad
        reasons = []
        for i in range(len(epoch_stats)):
            r = []
            if is_no_data.iloc[i]:
                r.append('no_data')
            if is_high_scatter.iloc[i] and not is_no_data.iloc[i]:
                r.append('high_scatter')
            if is_shifted.iloc[i] and not is_no_data.iloc[i]:
                r.append('shifted_median')
            if is_low_valid.iloc[i] and not is_no_data.iloc[i]:
                r.append('low_valid')
            if is_after_gap.iloc[i]:
                r.append('after_momentum_dump')
            reasons.append('|'.join(r) if r else '')
        epoch_stats['reason'] = reasons

        # Add common-mode placeholder (will be filled later)
        epoch_stats['cm'] = np.nan

        self.epoch_qc = epoch_stats

        n_bad = epoch_stats['is_bad'].sum()
        print(f"Epoch QC: {n_bad}/{len(epoch_stats)} epochs marked as bad ({n_bad/len(epoch_stats)*100:.1f}%)")

        return epoch_stats

    def compute_star_qc(self) -> pd.DataFrame:
        """
        Compute quality metrics per star.

        Returns:
            DataFrame with star QC metrics including TIC ID
        """
        if self.df is None:
            raise ValueError("No data loaded. Call load_data() first.")

        df = self.df.copy()

        # Count total and valid measurements per star
        star_stats = df.groupby('star_id').agg({
            'epoch': 'count',
            'quality': lambda x: (x == 0).sum(),
            'flux_raw': ['median', 'mean', 'std'],
            'flux_error': 'median'
        }).reset_index()

        star_stats.columns = [
            'star_id', 'n_total', 'n_valid',
            'median_flux', 'mean_flux', 'std_flux', 'median_error'
        ]

        # Completeness
        n_epochs = df['epoch'].nunique()
        star_stats['completeness'] = star_stats['n_valid'] / n_epochs

        # Robust scatter (MAD) for good quality only
        good_df = df[df['quality'] == 0].copy()
        if len(good_df) > 0:
            mad_per_star = good_df.groupby('star_id')['flux_raw'].apply(
                lambda x: np.median(np.abs(x - np.median(x))) if len(x) > 0 else np.nan
            ).reset_index()
            mad_per_star.columns = ['star_id', 'robust_scatter_mad']
            star_stats = star_stats.merge(mad_per_star, on='star_id', how='left')
        else:
            star_stats['robust_scatter_mad'] = star_stats['std_flux']

        # SNR estimate
        star_stats['snr'] = star_stats['median_flux'] / star_stats['median_error']

        # Add TIC ID and coordinates from catalog
        if self.catalog and 'stars' in self.catalog:
            stars = self.catalog['stars']
            star_stats['tic_id'] = star_stats['star_id'].apply(
                lambda x: stars.get(x, {}).get('tic_id', x)
            )
            star_stats['ra'] = star_stats['star_id'].apply(
                lambda x: stars.get(x, {}).get('ra')
            )
            star_stats['dec'] = star_stats['star_id'].apply(
                lambda x: stars.get(x, {}).get('dec')
            )
            star_stats['ref_x'] = star_stats['star_id'].apply(
                lambda x: stars.get(x, {}).get('ref_x')
            )
            star_stats['ref_y'] = star_stats['star_id'].apply(
                lambda x: stars.get(x, {}).get('ref_y')
            )
        else:
            star_stats['tic_id'] = star_stats['star_id']
            star_stats['ra'] = np.nan
            star_stats['dec'] = np.nan
            star_stats['ref_x'] = np.nan
            star_stats['ref_y'] = np.nan

        # Mark edge stars (near CCD boundary)
        edge_threshold = 50  # pixels from edge
        is_edge = (
            (star_stats['ref_x'] < edge_threshold) |
            (star_stats['ref_y'] < edge_threshold) |
            (star_stats['ref_x'] > 2048 - edge_threshold) |
            (star_stats['ref_y'] > 2048 - edge_threshold)
        )
        star_stats['is_edge_star'] = is_edge

        self.star_qc = star_stats

        print(f"Star QC: {len(star_stats)} stars analyzed")
        print(f"  Completeness range: {star_stats['completeness'].min():.2f} - {star_stats['completeness'].max():.2f}")
        print(f"  Edge stars: {star_stats['is_edge_star'].sum()}")

        return star_stats

    def find_quiet_stars(self) -> List[str]:
        """
        Find "quiet" stars for common-mode calculation.

        Criteria:
        - High completeness (>90%)
        - Low scatter (bottom 30% of MAD)
        - Not on edge
        - Positive flux

        Returns:
            List of star_ids for quiet stars
        """
        if self.star_qc is None:
            self.compute_star_qc()

        sq = self.star_qc.copy()

        # Filter criteria
        min_completeness = self.params['quiet_star_completeness']
        mad_percentile = self.params['quiet_star_mad_percentile']

        # Get MAD threshold (percentile of non-NaN values)
        valid_mad = sq['robust_scatter_mad'].dropna()
        if len(valid_mad) > 0:
            mad_threshold = np.percentile(valid_mad, mad_percentile)
        else:
            mad_threshold = np.inf

        # Apply filters
        is_complete = sq['completeness'] >= min_completeness
        is_low_scatter = sq['robust_scatter_mad'] <= mad_threshold
        is_not_edge = ~sq['is_edge_star']
        is_positive = sq['median_flux'] > 0

        quiet_mask = is_complete & is_low_scatter & is_not_edge & is_positive

        quiet_stars = sq.loc[quiet_mask, 'star_id'].tolist()

        # Mark in star_qc
        self.star_qc['is_quiet_star'] = self.star_qc['star_id'].isin(quiet_stars)

        self.quiet_stars = quiet_stars

        print(f"Found {len(quiet_stars)} quiet stars ({len(quiet_stars)/len(sq)*100:.1f}%)")

        return quiet_stars

    def compute_common_mode(self) -> pd.Series:
        """
        Compute common-mode correction using quiet stars.

        Common-mode captures systematic effects that affect all stars:
        - Scattered light
        - Pointing drift
        - Thermal effects

        Returns:
            Series indexed by epoch with common-mode values
        """
        if self.df is None:
            raise ValueError("No data loaded. Call load_data() first.")

        if self.quiet_stars is None:
            self.find_quiet_stars()

        if not self.quiet_stars:
            print("Warning: No quiet stars found, using all stars for common-mode")
            quiet_mask = self.df['quality'] == 0
        else:
            quiet_mask = (
                self.df['star_id'].isin(self.quiet_stars) &
                (self.df['quality'] == 0)
            )

        quiet_df = self.df[quiet_mask].copy()

        if len(quiet_df) == 0:
            print("Warning: No valid data for common-mode calculation")
            self.common_mode = pd.Series(1.0, index=self.df['epoch'].unique())
            return self.common_mode

        # Normalize each star to median=1
        star_medians = quiet_df.groupby('star_id')['flux_raw'].transform('median')
        quiet_df['flux_norm'] = quiet_df['flux_raw'] / star_medians

        # Compute robust median per epoch
        cm = quiet_df.groupby('epoch')['flux_norm'].median()

        # Fill missing epochs with 1.0
        all_epochs = self.df['epoch'].unique()
        cm = cm.reindex(all_epochs, fill_value=1.0)

        self.common_mode = cm

        # Update epoch_qc with cm values
        if self.epoch_qc is not None:
            self.epoch_qc['cm'] = self.epoch_qc['epoch'].map(cm)

        print(f"Common-mode range: {cm.min():.4f} - {cm.max():.4f}")

        return cm

    def create_masks(self) -> pd.DataFrame:
        """
        Create all mask columns and add flux_cm.

        Adds columns:
        - flux_cm: common-mode corrected flux
        - mask_bits: uint16 with all mask bits
        - mask_reason: string explanation

        Returns:
            DataFrame with masks added
        """
        if self.df is None:
            raise ValueError("No data loaded. Call load_data() first.")

        df = self.df.copy()

        # Ensure QC is computed
        if self.epoch_qc is None:
            self.compute_epoch_qc()
        if self.star_qc is None:
            self.compute_star_qc()
        if self.common_mode is None:
            self.compute_common_mode()

        # Initialize mask_bits
        df['mask_bits'] = np.uint16(0)
        reasons = [''] * len(df)

        # --- Bit 0: quality != 0 ---
        mask = df['quality'] != 0
        df.loc[mask, 'mask_bits'] |= MASK_QUALITY
        for i in df[mask].index:
            reasons[i] += 'quality|'

        # --- Bit 1: NaN/inf/flux_error <= 0 ---
        mask = (
            df['flux_raw'].isna() |
            df['flux_error'].isna() |
            np.isinf(df['flux_raw']) |
            np.isinf(df['flux_error']) |
            (df['flux_error'] <= 0)
        )
        df.loc[mask, 'mask_bits'] |= MASK_INVALID
        for i in df[mask].index:
            reasons[i] += 'invalid|'

        # --- Bit 2: bad_epoch ---
        bad_epochs = self.epoch_qc[self.epoch_qc['is_bad']]['epoch'].tolist()
        mask = df['epoch'].isin(bad_epochs)
        df.loc[mask, 'mask_bits'] |= MASK_BAD_EPOCH
        for i in df[mask].index:
            reasons[i] += 'bad_epoch|'

        # --- Compute flux_cm first (needed for outlier detection) ---
        df['flux_cm'] = df['flux_raw'] / df['epoch'].map(self.common_mode)

        # --- Bits 3-4: outliers (asymmetric: stricter for positive) ---
        # Per-star outlier detection
        sigma_pos = self.params['outlier_sigma_pos']
        sigma_neg = self.params['outlier_sigma_neg']

        # Only compute on valid points
        valid_mask = (df['mask_bits'] & (MASK_QUALITY | MASK_INVALID)) == 0

        for star_id in df['star_id'].unique():
            star_mask = (df['star_id'] == star_id) & valid_mask
            if star_mask.sum() < 5:
                continue

            star_flux = df.loc[star_mask, 'flux_cm']
            median = star_flux.median()
            mad = np.median(np.abs(star_flux - median))

            if mad == 0:
                continue

            # Normalized residuals
            residuals = (star_flux - median) / (mad * 1.4826)  # scale to sigma

            # Positive outliers (cosmic rays)
            pos_outlier = residuals > sigma_pos
            if pos_outlier.any():
                idx = star_flux[pos_outlier].index
                df.loc[idx, 'mask_bits'] |= MASK_OUTLIER_POS
                for i in idx:
                    reasons[i] += 'outlier_pos|'

            # Negative outliers (very rare, keep transits)
            neg_outlier = residuals < -sigma_neg
            if neg_outlier.any():
                idx = star_flux[neg_outlier].index
                df.loc[idx, 'mask_bits'] |= MASK_OUTLIER_NEG
                for i in idx:
                    reasons[i] += 'outlier_neg|'

        # --- Bit 5: edge (first/last N days) ---
        edge_days = self.params['edge_days']
        btjd_min = df['btjd'].min()
        btjd_max = df['btjd'].max()

        mask = (df['btjd'] < btjd_min + edge_days) | (df['btjd'] > btjd_max - edge_days)
        df.loc[mask, 'mask_bits'] |= MASK_EDGE
        for i in df[mask].index:
            reasons[i] += 'edge|'

        # --- Bit 6: artifact_window (from config) ---
        artifact_windows = get_artifact_windows(self.sector)
        if artifact_windows:
            for start, end, desc in artifact_windows:
                mask = (df['btjd'] >= start) & (df['btjd'] <= end)
                df.loc[mask, 'mask_bits'] |= MASK_ARTIFACT
                for i in df[mask].index:
                    reasons[i] += f'artifact:{desc}|'

        # --- Bit 7: low_snr ---
        snr_threshold = self.params['snr_threshold']
        mask = (df['flux_raw'] / df['flux_error']) < snr_threshold
        df.loc[mask, 'mask_bits'] |= MASK_LOW_SNR
        for i in df[mask].index:
            reasons[i] += 'low_snr|'

        # Clean up reasons (remove trailing |)
        df['mask_reason'] = [r.rstrip('|') for r in reasons]

        # Add tic_id directly to local df (not via _add_tic_to_dataframe which writes to self.df)
        if self.catalog and 'stars' in self.catalog:
            stars = self.catalog.get('stars', {})
            tic_map = {sid: info.get('tic_id', sid) for sid, info in stars.items()}
            df['tic_id'] = df['star_id'].map(tic_map)

        if 'tic_id' in df.columns:
            # Reorder columns
            cols = ['star_id', 'tic_id', 'epoch', 'btjd', 'flux_raw', 'flux_error',
                    'flux_cm', 'quality', 'mask_bits', 'mask_reason']
            other_cols = [c for c in df.columns if c not in cols]
            df = df[[c for c in cols if c in df.columns] + other_cols]

        self.df = df

        # Summary
        n_total = len(df)
        n_any_mask = (df['mask_bits'] > 0).sum()
        print(f"\nMask summary:")
        print(f"  Total points: {n_total:,}")
        print(f"  Any mask: {n_any_mask:,} ({n_any_mask/n_total*100:.1f}%)")
        print(f"  Clean points: {n_total - n_any_mask:,} ({(n_total-n_any_mask)/n_total*100:.1f}%)")

        # Per-mask summary
        for name, bit in [
            ('quality', MASK_QUALITY),
            ('invalid', MASK_INVALID),
            ('bad_epoch', MASK_BAD_EPOCH),
            ('outlier_pos', MASK_OUTLIER_POS),
            ('outlier_neg', MASK_OUTLIER_NEG),
            ('edge', MASK_EDGE),
            ('artifact', MASK_ARTIFACT),
            ('low_snr', MASK_LOW_SNR),
        ]:
            count = ((df['mask_bits'] & bit) > 0).sum()
            if count > 0:
                print(f"    {name}: {count:,} ({count/n_total*100:.2f}%)")

        return df

    def run(self, add_tic: bool = True) -> pd.DataFrame:
        """
        Run full cleaning pipeline.

        Args:
            add_tic: If True, add TIC IDs if not present

        Returns:
            Cleaned DataFrame with masks
        """
        print(f"\n{'='*60}")
        print(f"DataCleaner: Sector {self.sector}, Camera {self.camera}, CCD {self.ccd}")
        print(f"{'='*60}\n")

        # Load data
        print("Loading data...")
        self.load_data()

        # Add TIC IDs if needed
        if add_tic and not self.has_tic_ids():
            print("\nAdding TIC IDs...")
            self.add_tic_ids()
        elif self.has_tic_ids():
            print("TIC IDs already present")

        # Compute QC
        print("\nComputing epoch QC...")
        self.compute_epoch_qc()

        print("\nComputing star QC...")
        self.compute_star_qc()

        print("\nFinding quiet stars...")
        self.find_quiet_stars()

        print("\nComputing common-mode...")
        self.compute_common_mode()

        print("\nCreating masks...")
        self.create_masks()

        print("\n" + "="*60)
        print("Cleaning complete!")
        print("="*60)

        return self.df

    def save(self):
        """Save all outputs."""
        # Create directories
        self.qc_dir.mkdir(parents=True, exist_ok=True)
        self.cleaned_dir.mkdir(parents=True, exist_ok=True)

        # Save epoch_qc
        if self.epoch_qc is not None:
            path = self.qc_dir / "epoch_qc.parquet"
            self.epoch_qc.to_parquet(path, index=False)
            print(f"Saved {path.name}")

            # Also save bad_epochs.json
            bad_epochs = self.epoch_qc[self.epoch_qc['is_bad']][['epoch', 'btjd', 'reason']].to_dict('records')
            bad_path = self.qc_dir / "bad_epochs.json"
            with open(bad_path, 'w') as f:
                json.dump(bad_epochs, f, indent=2)
            print(f"Saved {bad_path.name}")

        # Save star_qc
        if self.star_qc is not None:
            path = self.qc_dir / "star_qc.parquet"
            self.star_qc.to_parquet(path, index=False)
            print(f"Saved {path.name}")

        # Save main data with masks
        if self.df is not None:
            path = self.cleaned_dir / "photometry_with_masks.parquet"
            self.df.to_parquet(path, index=False)
            print(f"Saved {path.name}")

    def get_cleaned_data(self, preset: str = 'balanced') -> pd.DataFrame:
        """
        Get cleaned data with specified preset.

        Args:
            preset: 'minimal', 'balanced', or 'strict'
                - minimal: only quality!=0 and invalid
                - balanced: + bad_epoch + outlier_pos
                - strict: + edge + low_snr

        Returns:
            Filtered DataFrame
        """
        if self.df is None:
            raise ValueError("No data. Call run() first.")

        df = self.df.copy()

        if preset == 'minimal':
            # Only base quality
            mask = MASK_QUALITY | MASK_INVALID
        elif preset == 'balanced':
            # Recommended for variable star search
            mask = MASK_QUALITY | MASK_INVALID | MASK_BAD_EPOCH | MASK_OUTLIER_POS | MASK_ARTIFACT
        elif preset == 'strict':
            # Maximum cleaning
            mask = MASK_QUALITY | MASK_INVALID | MASK_BAD_EPOCH | MASK_OUTLIER_POS | MASK_OUTLIER_NEG | MASK_EDGE | MASK_ARTIFACT | MASK_LOW_SNR
        else:
            raise ValueError(f"Unknown preset: {preset}")

        return df[(df['mask_bits'] & mask) == 0]

    def export_for_ml(self, mode: str = 'anomaly') -> Path:
        """
        Export features for ML (vectorized, fast).

        Args:
            mode: 'anomaly' or 'classification'
                - anomaly: minimal cleaning, keep flux_raw and flux_cm
                - classification: stricter cleaning, min_completeness=0.8

        Returns:
            Path to saved file
        """
        if self.df is None or self.star_qc is None:
            raise ValueError("No data. Call run() first.")

        self.ml_dir.mkdir(parents=True, exist_ok=True)

        # Get cleaned data based on mode
        if mode == 'anomaly':
            # Minimal cleaning
            mask = MASK_QUALITY | MASK_INVALID | MASK_BAD_EPOCH | MASK_ARTIFACT
            min_completeness = 0.5
        elif mode == 'classification':
            # Stricter cleaning
            mask = MASK_QUALITY | MASK_INVALID | MASK_BAD_EPOCH | MASK_OUTLIER_POS | MASK_EDGE | MASK_ARTIFACT
            min_completeness = 0.8
        else:
            raise ValueError(f"Unknown mode: {mode}")

        # Filter data
        df = self.df[(self.df['mask_bits'] & mask) == 0].copy()

        # Filter stars by completeness
        good_stars = self.star_qc[
            self.star_qc['completeness'] >= min_completeness
        ]['star_id'].tolist()
        df = df[df['star_id'].isin(good_stars)]

        n_epochs_total = self.df['epoch'].nunique()

        print(f"\nML Export ({mode}):")
        print(f"  Stars: {df['star_id'].nunique()}")
        print(f"  Points: {len(df):,}")

        # Vectorized feature computation using groupby
        print("  Computing features (vectorized)...")

        # Helper functions for aggregation
        def mad(x):
            """Median absolute deviation."""
            med = np.median(x)
            return np.median(np.abs(x - med))

        def amplitude(x):
            """(max-min)/median."""
            med = np.median(x)
            return (np.max(x) - np.min(x)) / med if med > 0 else np.nan

        def amplitude_robust(x):
            """(p95-p5)/median."""
            med = np.median(x)
            return (np.percentile(x, 95) - np.percentile(x, 5)) / med if med > 0 else np.nan

        def beyond_nsigma(x, n=2):
            """Fraction of points beyond n*sigma."""
            med = np.median(x)
            m = np.median(np.abs(x - med)) * 1.4826  # MAD to sigma
            if m == 0:
                return 0.0
            return np.sum(np.abs(x - med) > n * m) / len(x)

        # Group by star_id
        grouped = df.groupby('star_id')

        # Basic aggregations (fast)
        features_df = grouped.agg(
            n_points=('flux_cm', 'count'),
            mean_flux=('flux_cm', 'mean'),
            median_flux=('flux_cm', 'median'),
            std_flux=('flux_cm', 'std'),
            min_flux=('flux_cm', 'min'),
            max_flux=('flux_cm', 'max'),
            p5_flux=('flux_cm', lambda x: np.percentile(x, 5)),
            p95_flux=('flux_cm', lambda x: np.percentile(x, 95)),
            median_error=('flux_error', 'median'),
            skewness=('flux_cm', lambda x: pd.Series(x).skew()),
            kurtosis=('flux_cm', lambda x: pd.Series(x).kurtosis()),
        ).reset_index()

        # MAD (slightly slower but still vectorized)
        mad_series = grouped['flux_cm'].apply(mad).reset_index()
        mad_series.columns = ['star_id', 'mad_flux']
        features_df = features_df.merge(mad_series, on='star_id')

        # Computed metrics
        features_df['completeness'] = features_df['n_points'] / n_epochs_total
        features_df['amplitude'] = (features_df['max_flux'] - features_df['min_flux']) / features_df['median_flux']
        features_df['amplitude_robust'] = (features_df['p95_flux'] - features_df['p5_flux']) / features_df['median_flux']
        features_df['snr'] = features_df['median_flux'] / features_df['median_error']

        # Beyond sigma (need to compute per star)
        beyond_2s = grouped['flux_cm'].apply(lambda x: beyond_nsigma(x, 2)).reset_index()
        beyond_2s.columns = ['star_id', 'beyond_2sigma']
        features_df = features_df.merge(beyond_2s, on='star_id')

        beyond_3s = grouped['flux_cm'].apply(lambda x: beyond_nsigma(x, 3)).reset_index()
        beyond_3s.columns = ['star_id', 'beyond_3sigma']
        features_df = features_df.merge(beyond_3s, on='star_id')

        # Reduced chi2 (need flux_error too)
        def calc_chi2(group):
            flux = group['flux_cm'].values
            err = group['flux_error'].values
            med = np.median(flux)
            if len(flux) <= 1:
                return np.nan
            return np.sum(((flux - med) / err)**2) / (len(flux) - 1)

        chi2_series = grouped.apply(calc_chi2).reset_index()
        chi2_series.columns = ['star_id', 'reduced_chi2']
        features_df = features_df.merge(chi2_series, on='star_id')

        # Add TIC ID from star_qc
        tic_map = self.star_qc.set_index('star_id')['tic_id'].to_dict()
        features_df['tic_id'] = features_df['star_id'].map(tic_map)

        # Filter stars with too few points
        features_df = features_df[features_df['n_points'] >= 10]

        # Clean up intermediate columns
        features_df = features_df.drop(columns=['min_flux', 'max_flux', 'p5_flux', 'p95_flux'], errors='ignore')

        # Reorder columns
        cols = ['star_id', 'tic_id', 'n_points', 'completeness',
                'mean_flux', 'median_flux', 'std_flux', 'mad_flux',
                'amplitude', 'amplitude_robust', 'reduced_chi2',
                'skewness', 'kurtosis', 'beyond_2sigma', 'beyond_3sigma',
                'median_error', 'snr']
        features_df = features_df[[c for c in cols if c in features_df.columns]]

        # Save
        path = self.ml_dir / f"ml_{mode}.parquet"
        features_df.to_parquet(path, index=False)
        print(f"  Saved: {path.name}")

        return path

    def generate_report(self) -> Dict:
        """Generate summary report."""
        if self.df is None:
            raise ValueError("No data. Call run() first.")

        report = {
            'sector': self.sector,
            'camera': self.camera,
            'ccd': self.ccd,
            'generated': datetime.now().isoformat(),
            'data': {
                'n_stars': self.df['star_id'].nunique(),
                'n_epochs': self.df['epoch'].nunique(),
                'n_points': len(self.df),
                'btjd_range': [float(self.df['btjd'].min()), float(self.df['btjd'].max())],
            },
            'quality': {
                'clean_points': int((self.df['mask_bits'] == 0).sum()),
                'clean_fraction': float((self.df['mask_bits'] == 0).mean()),
            },
            'epochs': {
                'n_bad': int(self.epoch_qc['is_bad'].sum()) if self.epoch_qc is not None else 0,
            },
            'stars': {
                'n_quiet': len(self.quiet_stars) if self.quiet_stars else 0,
                'with_tic': int(self.star_qc['tic_id'].apply(lambda x: not str(x).startswith('STAR_')).sum()) if self.star_qc is not None else 0,
            },
            'common_mode': {
                'min': float(self.common_mode.min()) if self.common_mode is not None else None,
                'max': float(self.common_mode.max()) if self.common_mode is not None else None,
            },
            'parameters': self.params,
        }

        return report


def clean_sector(sector: int, camera: int, ccd: int, add_tic: bool = True) -> DataCleaner:
    """
    Convenience function to clean a sector.

    Args:
        sector: TESS sector number
        camera: Camera number (1-4)
        ccd: CCD number (1-4)
        add_tic: If True, add TIC IDs if not present

    Returns:
        DataCleaner instance with cleaned data
    """
    cleaner = DataCleaner(sector, camera, ccd)
    cleaner.run(add_tic=add_tic)
    cleaner.save()

    # Export for ML
    cleaner.export_for_ml(mode='anomaly')
    cleaner.export_for_ml(mode='classification')

    # Save report
    report = cleaner.generate_report()
    report_path = cleaner.qc_dir / "cleaning_report.json"
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2, default=str)
    print(f"Saved {report_path.name}")

    return cleaner
