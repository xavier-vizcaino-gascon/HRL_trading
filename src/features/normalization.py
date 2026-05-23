"""Feature categorization and per-category normalization.

Since NB01 now bakes normalization into the saved parquets (all features have
``Norm_`` prefix and are already transformed), the runtime normalizer is minimal.
The ``FeatureNormalizer`` class is kept for backward compatibility and for
optional re-normalization or denormalization in backtesting.
"""
import re
from typing import Dict, List

import numpy as np
from sklearn.preprocessing import RobustScaler


def categorize_features(feature_list: List[str]) -> Dict[str, List[str]]:
    """
    Categorize Norm_ features by transformation group.

    Groups (matching NB01 step 10):
        - log_ratio: log(close/indicator) for MAs, ichimoku, psar, pivots, MACD, volume, mom
        - oscillator: oscillators [0,100] divided by 100
        - percentage: percentage features divided by 100
        - passthrough: already bounded features copied as-is
        - robust_scaled: unbounded features normalized with RobustScaler
        - binary: binary/categorical features copied as-is

    Args:
        feature_list: List of feature column names (with or without Norm_ prefix)

    Returns:
        Dict mapping group name -> [feature names]
    """
    categories = {
        "log_ratio": [],
        "oscillator": [],
        "percentage": [],
        "passthrough": [],
        "robust_scaled": [],
        "binary": [],
    }

    # Base names for each group (without Norm_ prefix)
    group_map = {
        "log_ratio": {
            "wma_10", "wma_20", "wma_50", "wma_200", "psar",
            "ichimoku_conversion", "ichimoku_base", "ichimoku_span_a", "ichimoku_span_b",
            "pivot_point", "pivot_r1", "pivot_r2", "pivot_s1", "pivot_s2",
            "macd", "macd_signal", "macd_histogram", "keltner_pct_b",
            "volume", "mom_10", "mom_20",
        },
        "oscillator": {
            "rsi", "stoch_k", "stoch_d", "plus_di", "minus_di", "mfi",
            "price_position_10", "price_position_20", "price_position_50",
        },
        "percentage": {
            "donchian_width_pct", "candle_body_pct", "candle_range_pct",
            "pivot_distance_pct",
            "dist_from_high_10", "dist_from_low_10",
            "dist_from_high_20", "dist_from_low_20",
            "dist_from_high_50", "dist_from_low_50",
        },
        "passthrough": {
            "bb_percent_b", "bb_bandwidth", "cmf",
            "log_returns_1", "log_returns_5", "log_returns_10",
            "drawdown",
        },
        "robust_scaled": {
            "zscore", "sharpe_ratio", "drawdown_pct",
            "ad_line", "trend_strength",
            "cumulative_returns",
        },
        "binary": {
            "candle_body_ratio", "candle_upper_wick_ratio", "candle_lower_wick_ratio",
            "candle_direction",
            "pattern_doji", "pattern_hammer", "pattern_inverted_hammer",
            "pattern_marubozu", "pattern_spinning_top",
            "pattern_bullish_engulfing", "pattern_bearish_engulfing",
            "pattern_strong_momentum",
        },
    }

    for feature in feature_list:
        # Strip Norm_ prefix for matching
        base = feature.replace("Norm_", "") if feature.startswith("Norm_") else feature

        matched = False
        for group_name, base_set in group_map.items():
            if base in base_set:
                categories[group_name].append(feature)
                matched = True
                break

        if not matched:
            categories["binary"].append(feature)

    return categories


class FeatureNormalizer:
    """Per-category feature normalization/denormalization.

    With the new NB01 pipeline, data in saved parquets is already normalized
    (Norm_ prefix features). This class is kept for:
    - Backward compatibility with existing code
    - Optional denormalization for backtesting analysis
    - Re-normalization if needed
    """

    def __init__(self, metadata: dict):
        """
        Initialize normalizer from metadata.

        Args:
            metadata: Metadata dict with features, feature_groups, scaler params
        """
        self.metadata = metadata
        self.features = metadata["features"]
        self.categories = metadata.get(
            "feature_groups", categorize_features(metadata["features"])
        )

        # Initialize RobustScaler from saved params
        self.scaler = RobustScaler()
        if "scaler_center" in metadata:
            self.scaler.center_ = np.array(metadata["scaler_center"], dtype=np.float64)
            self.scaler.scale_ = np.array(metadata["scaler_scale"], dtype=np.float64)

        self.scaler_features = metadata.get("scaler_features", [])

    def normalize_array(self, arr: np.ndarray) -> np.ndarray:
        """
        No-op: data is already normalized in the saved parquets.

        Kept for backward compatibility. Returns the input array as float32.

        Args:
            arr: (n_samples, n_features) array (already normalized)

        Returns:
            Same array as float32
        """
        return arr.astype(np.float32)

    def denormalize_array(self, arr: np.ndarray) -> np.ndarray:
        """
        Reverse normalization for RobustScaler features only.

        Only the robust_scaled group needs inverse transform. Other groups
        (log_ratio, oscillator, percentage, passthrough, binary) are not
        easily invertible or don't need inversion for analysis.

        Args:
            arr: Normalized array (n_samples, n_features)

        Returns:
            Partially denormalized array, float32
        """
        arr_denorm = arr.astype(np.float64).copy()

        # Find indices of robust_scaled features
        robust_features = self.categories.get("robust_scaled", [])
        idx_robust = [
            i for i, f in enumerate(self.features) if f in robust_features
        ]

        if idx_robust and hasattr(self.scaler, "center_") and self.scaler.center_ is not None:
            arr_denorm[:, idx_robust] = self.scaler.inverse_transform(
                arr[:, idx_robust]
            )

        return arr_denorm.astype(np.float32)

    def get_category_info(self) -> Dict[str, int]:
        """Return count of features per category."""
        return {
            cat: len(feats) for cat, feats in self.categories.items() if feats
        }
