"""DataManager: gestor centralitzat de dades, splits i normalització."""
import json
from pathlib import Path
from typing import Tuple

import numpy as np
import polars as pl

from src.features.normalization import FeatureNormalizer


class DataManager:
    """
    Gestor centralitzat de dades per a ML trading.

    Encapsula:
    - Carrega de dades ja splitejades (train/val/test) amb features Norm_
    - Normalització/denormalització (data is pre-normalized in parquets)
    - Metadades: features, dates, scaler params, feature groups
    """

    def __init__(self, interval: str = "5m", project_root: Path = None):
        """
        Initialize DataManager.

        Args:
            interval: "5m", "15m", "60m", "240m", "1440m"
            project_root: Project root path (default: current directory)
        """
        self.interval = interval
        self.project_root = Path(project_root or ".")
        self.data_dir = self.project_root / "data" / "processed" / interval
        self.metadata_file = self.project_root / "data" / "processed" / "metadata.json"

        # Load metadata
        if not self.metadata_file.exists():
            raise FileNotFoundError(
                f"Metadata file not found: {self.metadata_file}\n"
                "Run NB01 first to generate the metadata."
            )

        with open(self.metadata_file, encoding="utf-8") as f:
            all_metadata = json.load(f)

        if interval not in all_metadata:
            raise ValueError(
                f"Interval '{interval}' not found in metadata. "
                f"Available: {list(all_metadata.keys())}"
            )

        self.metadata = all_metadata[interval]

        # Initialize normalizer (mostly no-op since data is pre-normalized)
        self.normalizer = FeatureNormalizer(self.metadata)

        # Norm_ features → passed to the RL agent as observations
        self.features = self.metadata["features"]

        # Raw price columns → kept in parquets for env PnL/reward calculations, NOT agent obs
        self.price_cols = ["open", "high", "low", "close"]

    def load_train(self) -> pl.DataFrame:
        """Load training data (already split and normalized)."""
        path = self.data_dir / "train.parquet"
        if not path.exists():
            raise FileNotFoundError(f"Train parquet not found: {path}")
        return pl.read_parquet(path)

    def load_val(self) -> pl.DataFrame:
        """Load validation data (already split and normalized)."""
        path = self.data_dir / "val.parquet"
        if not path.exists():
            raise FileNotFoundError(f"Val parquet not found: {path}")
        return pl.read_parquet(path)

    def load_test(self) -> pl.DataFrame:
        """Load test data (already split and normalized)."""
        path = self.data_dir / "test.parquet"
        if not path.exists():
            raise FileNotFoundError(f"Test parquet not found: {path}")
        return pl.read_parquet(path)

    def load_all(self) -> Tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
        """Load all three splits (train, val, test)."""
        return self.load_train(), self.load_val(), self.load_test()

    def normalize_array(self, arr: np.ndarray) -> np.ndarray:
        """
        No-op: data is already normalized in the saved parquets.

        Kept for backward compatibility with NB02/NB03 code that calls this.

        Args:
            arr: (n_samples, n_features) array (already Norm_ transformed)

        Returns:
            Same array as float32
        """
        return self.normalizer.normalize_array(arr)

    def denormalize_array(self, arr: np.ndarray) -> np.ndarray:
        """
        Reverse RobustScaler normalization (for backtesting/analysis).

        Only robust_scaled features are invertible. Log-ratios, oscillator/100,
        and percentage/100 transformations are not reversed.

        Args:
            arr: Normalized array (n_samples, n_features)

        Returns:
            Partially denormalized array, float32
        """
        return self.normalizer.denormalize_array(arr)

    def info(self):
        """Print summary of splits, features and normalization."""
        m = self.metadata
        cat_info = self.normalizer.get_category_info()

        print(f"\n{'='*70}")
        print(f"DataManager: {self.interval.upper()}")
        print(f"{'='*70}")

        print(f"\nWalk-Forward Splits:")
        print(f"  Train: {m['n_train']:>10,} samples  {m['date_train_start'][:10]} → {m['date_train_end'][:10]}")
        print(f"  Val:   {m['n_val']:>10,} samples  {m['date_val_start'][:10]} → {m['date_val_end'][:10]}")
        print(f"  Test:  {m['n_test']:>10,} samples  {m['date_test_start'][:10]} → {m['date_test_end'][:10]}")

        print(f"\nFeature Groups ({m['n_features']} Norm_ features):")
        for group, count in sorted(cat_info.items()):
            if count > 0:
                print(f"  {group:20s}: {count:3d} features")

        print(f"\nPre-baked Normalization (applied in NB01 step 10):")
        print(f"  A. Log-ratios:     log(close/indicator) for MAs, pivots, ichimoku")
        print(f"  B. Oscillators:    /100 → [0,1] for RSI, Stochastic, MFI, etc.")
        print(f"  C. Percentages:    /100 for candle_body_pct, dist_from_*, etc.")
        print(f"  D. Passthrough:    already bounded (bb_percent_b, log_returns, etc.)")
        print(f"  E. RobustScaler:   fit on train only (zscore, sharpe, ad_line, trend_strength, etc.)")
        print(f"  F. Binary:         no transform (patterns, candle_direction)")

        print(f"\nParquet columns: {m['n_features']} Norm_ + 4 price (open/high/low/close) + timestamp")
        print(f"  → price cols are for env PnL calculations, NOT passed to agent")

        print(f"\nModel Input:")
        print(f"  obs_dim: {m['obs_dim']} Norm_ features")

        print(f"{'='*70}\n")

    def __repr__(self) -> str:
        return (
            f"DataManager(interval='{self.interval}', "
            f"features={self.metadata['n_features']}, "
            f"train={self.metadata['n_train']:,}, "
            f"val={self.metadata['n_val']:,}, "
            f"test={self.metadata['n_test']:,})"
        )
