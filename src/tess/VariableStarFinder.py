"""
Variable Star Finder - Find and rank variable stars in TESS photometry data.

Uses existing pipeline components:
- LightcurveBuilder: amplitude, chi2, stetson_j metrics
- MLExport: Lomb-Scargle periodicity
- TIC catalog: stellar parameters enrichment
"""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from tqdm import tqdm
import time

from .config import BASE_DIR, DATA_DIR
from .StreamingPipeline import convert_to_starcatalog, get_streaming_results
from .LightcurveBuilder import Lightcurve, LightcurveCollection
from .MLExport import FeatureExtractor


# Output directory
VARIABLE_STARS_DIR = BASE_DIR / "variable_stars"


def ensure_variable_dirs():
    """Create output directories."""
    VARIABLE_STARS_DIR.mkdir(parents=True, exist_ok=True)
    (VARIABLE_STARS_DIR / "plots").mkdir(exist_ok=True)


class VariableStarFinder:
    """Find and rank variable stars in TESS photometry data."""

    def __init__(self, sector: int, camera: str = "1", ccd: str = "1"):
        """
        Args:
            sector: TESS sector number
            camera: Camera number (1-4)
            ccd: CCD number (1-4)
        """
        self.sector = sector
        self.camera = camera
        self.ccd = ccd
        self.session_id = f"s{sector:04d}_{camera}-{ccd}"

        self.collection = None
        self.periodic_features = {}
        self.tic_params = {}
        self.variability_df = None

    def load_and_build_lightcurves(self, min_completeness: float = 0.5,
                                    min_snr: float = 5.0) -> int:
        """
        Load streaming results and build lightcurves with quality filtering.

        Args:
            min_completeness: Minimum fraction of good epochs (0-1)
            min_snr: Minimum median signal-to-noise ratio

        Returns:
            Number of lightcurves after filtering
        """
        print(f"Loading data for sector {self.sector}, camera {self.camera}, CCD {self.ccd}...")

        # Convert streaming data to StarCatalog format
        catalog = convert_to_starcatalog(self.sector, self.camera, self.ccd)

        # Build lightcurves
        print("Building lightcurves...")
        collection = LightcurveCollection(catalog)

        # Apply quality filters
        print(f"Filtering: completeness >= {min_completeness}, SNR >= {min_snr}")
        self.collection = collection.filter_by_quality(min_completeness, min_snr)

        print(f"Loaded {len(self.collection.lightcurves)} lightcurves (from {len(collection.lightcurves)} total)")
        return len(self.collection.lightcurves)

    def calculate_periodicity(self, n_frequencies: int = 5) -> int:
        """
        Calculate Lomb-Scargle periodicity features for all stars.

        Args:
            n_frequencies: Number of top frequencies to extract

        Returns:
            Number of stars with periodicity calculated
        """
        if self.collection is None:
            raise ValueError("No lightcurves loaded. Call load_and_build_lightcurves() first.")

        print("Calculating periodicity (Lomb-Scargle)...")

        for star_id, lc in tqdm(self.collection.lightcurves.items(), desc="Periodicity"):
            features = FeatureExtractor.extract_periodic_features(lc, n_frequencies)
            if features:
                self.periodic_features[star_id] = features

        print(f"Periodicity calculated for {len(self.periodic_features)} stars")
        return len(self.periodic_features)

    def enrich_with_tic(self, fields: List[str] = None) -> int:
        """
        Query TIC for additional stellar parameters.

        Args:
            fields: List of TIC fields to retrieve (default: Teff, logg, rad, mass, lumclass)

        Returns:
            Number of stars enriched
        """
        if self.collection is None:
            raise ValueError("No lightcurves loaded. Call load_and_build_lightcurves() first.")

        if fields is None:
            fields = ['Teff', 'logg', 'rad', 'mass', 'lumclass', 'objType', 'lum', 'Vmag', 'Bmag']

        from astroquery.mast import Catalogs

        # Get stars with TIC IDs
        stars_with_tic = []
        for star_id, lc in self.collection.lightcurves.items():
            tic_id = lc.star_info.get('tic_id')
            if tic_id and str(tic_id).replace('.0', '').isdigit():
                stars_with_tic.append((star_id, int(float(tic_id))))

        print(f"Querying TIC for {len(stars_with_tic)} stars...")

        for star_id, tic_id in tqdm(stars_with_tic, desc="TIC Query"):
            try:
                result = Catalogs.query_criteria(
                    catalog="Tic",
                    ID=tic_id
                )

                if len(result) > 0:
                    row = result[0]
                    self.tic_params[star_id] = {
                        'tic_id': tic_id,
                    }
                    for field in fields:
                        val = row.get(field)
                        # Convert masked values to None
                        if hasattr(val, 'mask') or val is np.ma.masked:
                            val = None
                        elif isinstance(val, (np.floating, float)) and np.isnan(val):
                            val = None
                        self.tic_params[star_id][field] = val

            except Exception as e:
                continue

            time.sleep(0.3)  # Rate limiting

        print(f"Enriched {len(self.tic_params)} stars with TIC data")
        return len(self.tic_params)

    def calculate_variability_scores(self) -> pd.DataFrame:
        """
        Calculate composite variability score for all stars.

        Combines:
        - amplitude: fractional brightness change
        - reduced_chi2: statistical significance
        - stetson_j: correlated variability
        - power_ratio_top: periodicity strength

        Returns:
            DataFrame sorted by variability_score
        """
        if self.collection is None:
            raise ValueError("No lightcurves loaded. Call load_and_build_lightcurves() first.")

        print("Calculating variability scores...")

        records = []

        for star_id, lc in self.collection.lightcurves.items():
            record = {
                'star_id': star_id,
                'ra': lc.star_info.get('ra'),
                'dec': lc.star_info.get('dec'),
                'tic_id': lc.star_info.get('tic_id'),
                'tmag': lc.star_info.get('tmag'),
                'match_quality': lc.star_info.get('match_quality'),

                # Variability metrics from LightcurveBuilder
                'amplitude': lc.stats.get('amplitude', 0),
                'amplitude_robust': lc.stats.get('amplitude_robust', 0),  # 5-95 percentile based
                'reduced_chi2': lc.stats.get('reduced_chi2', 1),
                'stetson_j': lc.stats.get('stetson_j', 0),
                'mad': lc.stats.get('mad', 0),
                'iqr': lc.stats.get('iqr', 0),
                'std_flux': lc.stats.get('std_flux', 0),
                'n_outliers': lc.stats.get('n_outliers', 0),
                'outlier_fraction': lc.stats.get('outlier_fraction', 0),
                'completeness': lc.stats.get('completeness', 0),
                'median_snr': lc.stats.get('median_snr', 0),
                'n_good': lc.stats.get('n_good', 0),
                'time_span_days': lc.stats.get('time_span_days', 0),
            }

            # Add periodicity features
            periodic = self.periodic_features.get(star_id, {})
            record.update({
                'period_1': periodic.get('period_1'),
                'power_1': periodic.get('power_1'),
                'period_2': periodic.get('period_2'),
                'power_2': periodic.get('power_2'),
                'power_ratio_top': periodic.get('power_ratio_top', 0),
                'total_power': periodic.get('total_power', 0),
            })

            # Add TIC stellar parameters
            tic = self.tic_params.get(star_id, {})
            record.update({
                'Teff': tic.get('Teff'),
                'logg': tic.get('logg'),
                'rad': tic.get('rad'),
                'mass': tic.get('mass'),
                'lumclass': tic.get('lumclass'),
            })

            records.append(record)

        df = pd.DataFrame(records)

        # Calculate composite variability score using ROBUST metrics
        # Component 1: Robust amplitude (5-95 percentile, log scale)
        df['amp_score'] = np.log10(df['amplitude_robust'].clip(lower=1e-5)) + 5

        # Component 2: Chi-squared excess (log scale) - variability significance
        df['chi2_score'] = np.log10((df['reduced_chi2']).clip(lower=0.01))

        # Component 3: Stetson J (absolute value) - correlated variability
        df['stetson_score'] = np.abs(df['stetson_j'])

        # Component 4: Periodicity strength
        df['periodic_score'] = df['power_ratio_top'].fillna(0)

        # Component 5: Penalize high outlier fraction (likely artifacts)
        df['outlier_penalty'] = 1 - df['outlier_fraction'].clip(upper=0.1) * 10

        # Normalize each component to [0, 1]
        for col in ['amp_score', 'chi2_score', 'stetson_score', 'periodic_score']:
            col_min = df[col].min()
            col_max = df[col].max()
            if col_max > col_min:
                df[f'{col}_norm'] = (df[col] - col_min) / (col_max - col_min)
            else:
                df[f'{col}_norm'] = 0.5

        # Weighted combination (using robust amplitude, penalizing outliers)
        weights = {
            'amp_score_norm': 0.35,      # Robust amplitude is primary
            'chi2_score_norm': 0.25,     # Statistical significance
            'stetson_score_norm': 0.20,  # Correlated variability
            'periodic_score_norm': 0.20, # Periodicity strength
        }

        df['variability_score'] = sum(df[col] * w for col, w in weights.items())
        df['variability_score'] *= df['outlier_penalty']  # Penalize high outlier stars

        # Sort by score
        self.variability_df = df.sort_values('variability_score', ascending=False).reset_index(drop=True)

        print(f"Calculated scores for {len(self.variability_df)} stars")
        return self.variability_df

    def get_variable_candidates(self, min_amplitude: float = 0.01,
                                 min_chi2: float = None,
                                 min_score: float = None,
                                 max_outlier_fraction: float = 0.05,
                                 top_n: int = 100) -> pd.DataFrame:
        """
        Filter to keep only likely variable stars.

        Args:
            min_amplitude: Minimum ROBUST fractional amplitude (5-95 percentile, default 1%)
            min_chi2: Minimum reduced chi-squared (optional)
            min_score: Minimum variability score (optional)
            max_outlier_fraction: Maximum fraction of outlier points (default 5%)
            top_n: Maximum number of candidates to return

        Returns:
            DataFrame of variable candidates sorted by variability_score
        """
        if self.variability_df is None:
            raise ValueError("No scores calculated. Call calculate_variability_scores() first.")

        df = self.variability_df.copy()

        # Apply filters using ROBUST amplitude (insensitive to outliers)
        mask = df['amplitude_robust'] >= min_amplitude

        # Filter out stars with too many outliers (likely artifacts)
        mask &= (df['outlier_fraction'] <= max_outlier_fraction)

        # Chi2 filter only if specified
        if min_chi2 is not None:
            mask &= (df['reduced_chi2'] >= min_chi2)

        if min_score is not None:
            mask &= (df['variability_score'] >= min_score)

        candidates = df[mask].head(top_n)

        return candidates

    def save_results(self, output_dir: str = None) -> Tuple[Path, Path]:
        """
        Save variable star candidates to CSV.

        Args:
            output_dir: Output directory (default: variable_stars/)

        Returns:
            Tuple of (full_table_path, top_candidates_path)
        """
        if self.variability_df is None:
            raise ValueError("No scores calculated. Call calculate_variability_scores() first.")

        ensure_variable_dirs()

        if output_dir is None:
            output_dir = VARIABLE_STARS_DIR
        else:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)

        # Select output columns (including robust metrics)
        output_cols = [
            'star_id', 'tic_id', 'ra', 'dec', 'tmag', 'match_quality',
            'amplitude', 'amplitude_robust', 'reduced_chi2', 'stetson_j', 'mad',
            'n_outliers', 'outlier_fraction',
            'period_1', 'power_ratio_top',
            'completeness', 'median_snr', 'n_good', 'time_span_days',
            'Teff', 'logg', 'rad', 'mass', 'lumclass',
            'variability_score'
        ]

        # Keep only existing columns
        available_cols = [c for c in output_cols if c in self.variability_df.columns]

        # Full table
        full_path = output_dir / f"{self.session_id}_all_stars.csv"
        self.variability_df[available_cols].to_csv(full_path, index=False)

        # Top candidates
        top = self.get_variable_candidates(top_n=100)
        top_path = output_dir / f"{self.session_id}_variable_candidates.csv"
        top[available_cols].to_csv(top_path, index=False)

        # Summary
        summary = {
            'sector': self.sector,
            'camera': self.camera,
            'ccd': self.ccd,
            'total_stars': len(self.variability_df),
            'stars_with_tic': int((self.variability_df['tic_id'].notna()).sum()),
            'variable_candidates': len(top),
            'median_amplitude': float(top['amplitude'].median()) if len(top) > 0 else 0,
            'max_amplitude': float(top['amplitude'].max()) if len(top) > 0 else 0,
        }

        summary_path = output_dir / f"{self.session_id}_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)

        print(f"\nResults saved:")
        print(f"  All stars: {full_path}")
        print(f"  Variable candidates: {top_path}")
        print(f"  Summary: {summary_path}")

        return full_path, top_path

    def plot_top_candidates(self, n_plots: int = 20, output_dir: str = None) -> int:
        """
        Generate lightcurve plots for top variable candidates.

        Args:
            n_plots: Number of plots to generate
            output_dir: Output directory for plots

        Returns:
            Number of plots generated
        """
        if self.variability_df is None:
            raise ValueError("No scores calculated. Call calculate_variability_scores() first.")

        ensure_variable_dirs()

        if output_dir is None:
            output_dir = VARIABLE_STARS_DIR / "plots"
        else:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)

        top = self.get_variable_candidates(top_n=n_plots)

        print(f"Generating {len(top)} lightcurve plots...")

        for rank, (idx, row) in enumerate(top.iterrows(), 1):
            star_id = row['star_id']
            lc = self.collection.get(star_id)

            if lc is None:
                continue

            # Normalize for plotting
            lc_norm = lc.normalize('median')

            # Create informative filename
            tic_str = f"TIC{int(row['tic_id'])}" if pd.notna(row['tic_id']) else star_id
            filename = f"var_{rank:03d}_{tic_str}_amp{row['amplitude']*100:.1f}pct.png"

            # Plot
            fig, ax = plt.subplots(figsize=(12, 4))
            lc_norm.plot(ax=ax, normalized=True, show_errors=True)

            # Title with metrics
            period_str = f"P={row['period_1']:.2f}d" if pd.notna(row['period_1']) else "P=?"
            ax.set_title(
                f"#{rank} {tic_str} | Amp={row['amplitude']*100:.2f}% | "
                f"chi2={row['reduced_chi2']:.1f} | {period_str} | "
                f"Score={row['variability_score']:.3f}"
            )

            plt.tight_layout()
            plt.savefig(output_dir / filename, dpi=150)
            plt.close()

        print(f"Plots saved to: {output_dir}")
        return len(top)

    def print_summary(self):
        """Print summary statistics."""
        if self.variability_df is None:
            print("No data calculated yet.")
            return

        df = self.variability_df

        print("\n" + "="*60)
        print(f"VARIABLE STAR FINDER - Sector {self.sector} Camera {self.camera} CCD {self.ccd}")
        print("="*60)

        print(f"\nTotal stars analyzed: {len(df)}")
        print(f"Stars with TIC match: {(df['tic_id'].notna()).sum()}")

        if len(self.tic_params) > 0:
            print(f"Stars with TIC parameters: {len(self.tic_params)}")

        # Variability stats using robust amplitude
        var_candidates = self.get_variable_candidates(min_amplitude=0.01, max_outlier_fraction=0.05)
        print(f"\nVariable candidates (robust amp >= 1%, outliers <= 5%): {len(var_candidates)}")

        if len(var_candidates) > 0:
            print(f"\nTop 10 variable stars (by robust metrics):")
            print("-"*70)

            top10 = var_candidates.head(10)
            for i, (idx, row) in enumerate(top10.iterrows(), 1):
                tic_str = f"TIC {int(row['tic_id'])}" if pd.notna(row['tic_id']) else row['star_id']
                period_str = f"P={row['period_1']:.2f}d" if pd.notna(row['period_1']) else ""
                outliers = f"out={row['n_outliers']:.0f}" if 'n_outliers' in row else ""
                print(f"  {i:2d}. {tic_str:20s} RobAmp={row['amplitude_robust']*100:5.2f}% chi2={row['reduced_chi2']:5.2f} {outliers} {period_str}")

        print("="*60)


def find_variables(sector: int, camera: str = "1", ccd: str = "1",
                   min_completeness: float = 0.5, min_snr: float = 5.0,
                   min_amplitude: float = 0.01, min_chi2: float = 2.0,
                   enrich_tic: bool = False, plot: bool = False,
                   top_n: int = 100) -> pd.DataFrame:
    """
    Convenience function to find variable stars.

    Args:
        sector: TESS sector number
        camera: Camera number
        ccd: CCD number
        min_completeness: Minimum data completeness
        min_snr: Minimum signal-to-noise ratio
        min_amplitude: Minimum variability amplitude
        min_chi2: Minimum chi-squared
        enrich_tic: Query TIC for stellar parameters
        plot: Generate lightcurve plots
        top_n: Number of top candidates

    Returns:
        DataFrame of variable candidates
    """
    finder = VariableStarFinder(sector, camera, ccd)

    # Load and filter
    finder.load_and_build_lightcurves(min_completeness, min_snr)

    # Calculate periodicity
    finder.calculate_periodicity()

    # Optionally enrich with TIC
    if enrich_tic:
        finder.enrich_with_tic()

    # Calculate scores
    finder.calculate_variability_scores()

    # Get candidates
    candidates = finder.get_variable_candidates(min_amplitude, min_chi2, top_n=top_n)

    # Save results
    finder.save_results()

    # Print summary
    finder.print_summary()

    # Optionally plot
    if plot:
        finder.plot_top_candidates(n_plots=min(20, len(candidates)))

    return candidates


if __name__ == "__main__":
    # Example usage
    candidates = find_variables(
        sector=70,
        camera="1",
        ccd="1",
        min_completeness=0.5,
        min_snr=5.0,
        enrich_tic=False,  # Set to True for TIC enrichment (~8 min)
        plot=True
    )
