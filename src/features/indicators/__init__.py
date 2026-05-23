"""
Technical indicators for feature engineering.

This module provides a comprehensive set of technical indicators organized by category:

- **Trend**: Moving averages, MACD, ADX, etc.
- **Momentum**: RSI, Stochastic, Williams %R, etc.
- **Volatility**: Bollinger Bands, ATR, Keltner Channels, etc.
- **Volume**: OBV, VWAP, MFI, Accumulation/Distribution, etc.
- **Statistical**: Returns, skewness, kurtosis, z-score, etc.
- **Price Action**: Candle patterns, pivot points, gaps, etc.

All indicators inherit from BaseIndicator and follow a consistent interface:
- calculate(df: pl.DataFrame) -> pl.DataFrame
- required_columns property
- output_columns property

Example usage:
    from src.features.indicators import SMA, RSI, BollingerBands

    # Single indicator
    sma = SMA(periods=[20, 50])
    df_with_sma = sma.calculate(df)

    # Multiple indicators
    indicators = [
        SMA(periods=[20, 50]),
        RSI(period=14),
        BollingerBands(period=20)
    ]

    result = df
    for indicator in indicators:
        result = indicator.calculate(result)
"""

# Trend indicators
# Momentum indicators
from .momentum import CCI, ROC, RSI, Momentum, StochasticOscillator, UltimateOscillator, WilliamsR

# Price action features
from .price_action import (
    CandleMetrics,
    CandlestickPatterns,
    FractalIndicator,
    GapDetection,
    PivotPoints,
    PricePosition,
    TrendStrength,
)

# Statistical features
from .statistical import (
    Autocorrelation,
    Drawdown,
    HigherMoments,
    HurstExponent,
    PercentileRank,
    Returns,
    RollingBeta,
    SharpeRatio,
    ZScore,
)
from .trend import ADX, EMA, MACD, SMA, WMA, IchimokuCloud, ParabolicSAR

# Volatility indicators
from .volatility import (
    ATR,
    BollingerBands,
    DonchianChannels,
    HistoricalVolatility,
    KeltnerChannels,
    StandardDeviation,
    UlcerIndex,
)

# Volume indicators
from .volume import (
    MFI,
    OBV,
    VWAP,
    AccumulationDistribution,
    ChaikinMoneyFlow,
    EaseOfMovement,
    ForceIndex,
    VolumeROC,
    VolumeWeightedMACD,
)

__all__ = [
    # Trend
    "SMA",
    "EMA",
    "WMA",
    "MACD",
    "ADX",
    "ParabolicSAR",
    "IchimokuCloud",
    # Momentum
    "RSI",
    "StochasticOscillator",
    "WilliamsR",
    "CCI",
    "ROC",
    "Momentum",
    "UltimateOscillator",
    # Volatility
    "BollingerBands",
    "ATR",
    "KeltnerChannels",
    "StandardDeviation",
    "HistoricalVolatility",
    "DonchianChannels",
    "UlcerIndex",
    # Volume
    "OBV",
    "VWAP",
    "MFI",
    "AccumulationDistribution",
    "ChaikinMoneyFlow",
    "VolumeROC",
    "ForceIndex",
    "EaseOfMovement",
    "VolumeWeightedMACD",
    # Statistical
    "Returns",
    "HigherMoments",
    "ZScore",
    "PercentileRank",
    "Autocorrelation",
    "HurstExponent",
    "RollingBeta",
    "SharpeRatio",
    "Drawdown",
    # Price Action
    "CandleMetrics",
    "PivotPoints",
    "GapDetection",
    "CandlestickPatterns",
    "PricePosition",
    "FractalIndicator",
    "TrendStrength",
]
