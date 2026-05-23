"""
Feature engineering module for quantitative trading.

This module provides:
- Base classes for indicators
- Technical indicator implementations
- Feature pipeline for batch processing
- Configuration system for indicators

Example usage:
    from src.features import FeaturePipeline
    from src.features.indicators import SMA, RSI, BollingerBands

    # Create pipeline
    pipeline = FeaturePipeline([
        SMA(periods=[20, 50]),
        RSI(period=14),
        BollingerBands(period=20)
    ])

    # Calculate features
    df_with_features = pipeline.calculate(df)

    # Or load from config
    pipeline = FeaturePipeline.from_config("config/features/default.yaml")
    df_with_features = pipeline.calculate(df)
"""

from .base import BaseIndicator, IndicatorConfig, IndicatorGroup
from .pipeline import FeaturePipeline, FeatureSelector

__all__ = [
    "BaseIndicator",
    "IndicatorConfig",
    "IndicatorGroup",
    "FeaturePipeline",
    "FeatureSelector",
]
