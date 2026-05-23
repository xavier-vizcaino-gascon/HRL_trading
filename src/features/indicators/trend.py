"""
Trend indicators for technical analysis.

These indicators help identify market direction and trend strength:
- Moving averages (SMA, EMA, WMA)
- MACD (Moving Average Convergence Divergence)
- ADX (Average Directional Index)
- Parabolic SAR
"""

import polars as pl

from ..base import BaseIndicator, IndicatorConfig


class SMA(BaseIndicator):
    """
    Simple Moving Average (SMA).

    Calculates the arithmetic mean of prices over a specified period.
    """

    def __init__(
        self, periods: list[int] = [10, 20, 50, 200], config: IndicatorConfig | None = None
    ):
        """
        Initialize SMA indicator.

        Args:
            periods: List of periods to calculate (default: [10, 20, 50, 200])
            config: Optional configuration
        """
        super().__init__(config)
        self.periods = periods or self.config.get("periods", [10, 20, 50, 200])

    def calculate(self, df: pl.DataFrame) -> pl.DataFrame:
        """Calculate SMA for each period."""
        self.validate_input(df)

        result = df
        for period in self.periods:
            col_name = f"sma_{period}"
            result = result.with_columns(
                pl.col("close").rolling_mean(window_size=period).alias(col_name)
            )

        return result

    @property
    def required_columns(self) -> list[str]:
        return ["close"]

    @property
    def output_columns(self) -> list[str]:
        return [f"sma_{period}" for period in self.periods]


class EMA(BaseIndicator):
    """
    Exponential Moving Average (EMA).

    Gives more weight to recent prices, making it more responsive than SMA.
    """

    def __init__(self, periods: list[int] = [12, 26, 50], config: IndicatorConfig | None = None):
        """
        Initialize EMA indicator.

        Args:
            periods: List of periods to calculate (default: [12, 26, 50])
            config: Optional configuration
        """
        super().__init__(config)
        self.periods = periods or self.config.get("periods", [12, 26, 50])

    def calculate(self, df: pl.DataFrame) -> pl.DataFrame:
        """Calculate EMA for each period."""
        self.validate_input(df)

        result = df
        for period in self.periods:
            col_name = f"ema_{period}"
            # Polars EWM (Exponentially Weighted Moving average)
            # span parameter corresponds to the period
            result = result.with_columns(
                pl.col("close").ewm_mean(span=period, adjust=False).alias(col_name)
            )

        return result

    @property
    def required_columns(self) -> list[str]:
        return ["close"]

    @property
    def output_columns(self) -> list[str]:
        return [f"ema_{period}" for period in self.periods]


class WMA(BaseIndicator):
    """
    Weighted Moving Average (WMA).

    Linearly weighted moving average giving more weight to recent prices.
    """

    def __init__(self, periods: list[int] = [10, 20], config: IndicatorConfig | None = None):
        """
        Initialize WMA indicator.

        Args:
            periods: List of periods to calculate (default: [10, 20])
            config: Optional configuration
        """
        super().__init__(config)
        self.periods = periods or self.config.get("periods", [10, 20])

    def calculate(self, df: pl.DataFrame) -> pl.DataFrame:
        """Calculate WMA for each period."""
        self.validate_input(df)

        result = df
        for period in self.periods:
            col_name = f"wma_{period}"
            # Create weights: 1, 2, 3, ..., period
            weights = list(range(1, period + 1))
            weight_sum = sum(weights)

            # Calculate WMA using rolling window with custom aggregation
            result = result.with_columns(
                pl.col("close")
                .rolling_map(
                    function=lambda s: (
                        sum(s[i] * weights[i] for i in range(len(s))) / weight_sum
                        if len(s) == period
                        else None
                    ),
                    window_size=period,
                )
                .alias(col_name)
            )

        return result

    @property
    def required_columns(self) -> list[str]:
        return ["close"]

    @property
    def output_columns(self) -> list[str]:
        return [f"wma_{period}" for period in self.periods]


class MACD(BaseIndicator):
    """
    Moving Average Convergence Divergence (MACD).

    Shows relationship between two moving averages.
    Outputs: MACD line, signal line, and histogram.
    """

    def __init__(
        self,
        fast_period: int = 12,
        slow_period: int = 26,
        signal_period: int = 9,
        config: IndicatorConfig | None = None,
    ):
        """
        Initialize MACD indicator.

        Args:
            fast_period: Fast EMA period (default: 12)
            slow_period: Slow EMA period (default: 26)
            signal_period: Signal line EMA period (default: 9)
            config: Optional configuration
        """
        super().__init__(config)
        self.fast_period = fast_period or self.config.get("fast_period", 12)
        self.slow_period = slow_period or self.config.get("slow_period", 26)
        self.signal_period = signal_period or self.config.get("signal_period", 9)

    def calculate(self, df: pl.DataFrame) -> pl.DataFrame:
        """Calculate MACD, signal line, and histogram."""
        self.validate_input(df)

        # Calculate fast and slow EMAs
        fast_ema = pl.col("close").ewm_mean(span=self.fast_period, adjust=False)
        slow_ema = pl.col("close").ewm_mean(span=self.slow_period, adjust=False)

        # MACD line = fast EMA - slow EMA
        macd_line = fast_ema - slow_ema

        result = df.with_columns(macd_line.alias("macd"))

        # Signal line = EMA of MACD line
        result = result.with_columns(
            pl.col("macd").ewm_mean(span=self.signal_period, adjust=False).alias("macd_signal")
        )

        # Histogram = MACD - Signal
        result = result.with_columns(
            (pl.col("macd") - pl.col("macd_signal")).alias("macd_histogram")
        )

        return result

    @property
    def required_columns(self) -> list[str]:
        return ["close"]

    @property
    def output_columns(self) -> list[str]:
        return ["macd", "macd_signal", "macd_histogram"]


class ADX(BaseIndicator):
    """
    Average Directional Index (ADX).

    Measures trend strength (not direction).
    Also calculates +DI and -DI for directional movement.
    """

    def __init__(self, period: int = 14, config: IndicatorConfig | None = None):
        """
        Initialize ADX indicator.

        Args:
            period: Calculation period (default: 14)
            config: Optional configuration
        """
        super().__init__(config)
        self.period = period or self.config.get("period", 14)

    def calculate(self, df: pl.DataFrame) -> pl.DataFrame:
        """Calculate ADX, +DI, and -DI."""
        self.validate_input(df)

        # Calculate directional movement
        high_diff = pl.col("high") - pl.col("high").shift(1)
        low_diff = pl.col("low").shift(1) - pl.col("low")

        # +DM and -DM
        plus_dm = pl.when((high_diff > low_diff) & (high_diff > 0)).then(high_diff).otherwise(0)

        minus_dm = pl.when((low_diff > high_diff) & (low_diff > 0)).then(low_diff).otherwise(0)

        # True Range
        tr1 = pl.col("high") - pl.col("low")
        tr2 = (pl.col("high") - pl.col("close").shift(1)).abs()
        tr3 = (pl.col("low") - pl.col("close").shift(1)).abs()
        tr = pl.max_horizontal(tr1, tr2, tr3)

        result = df.with_columns(
            [plus_dm.alias("_plus_dm"), minus_dm.alias("_minus_dm"), tr.alias("_tr")]
        )

        # Smooth using EMA (Wilder's smoothing = EMA with alpha = 1/period)
        alpha = 1.0 / self.period
        result = result.with_columns(
            [
                pl.col("_plus_dm").ewm_mean(alpha=alpha, adjust=False).alias("_plus_dm_smooth"),
                pl.col("_minus_dm").ewm_mean(alpha=alpha, adjust=False).alias("_minus_dm_smooth"),
                pl.col("_tr").ewm_mean(alpha=alpha, adjust=False).alias("_atr"),
            ]
        )

        # Calculate +DI and -DI
        result = result.with_columns(
            [
                (100 * pl.col("_plus_dm_smooth") / pl.col("_atr")).alias("plus_di"),
                (100 * pl.col("_minus_dm_smooth") / pl.col("_atr")).alias("minus_di"),
            ]
        )

        # Calculate DX and ADX
        result = result.with_columns(
            (
                100
                * (pl.col("plus_di") - pl.col("minus_di")).abs()
                / (pl.col("plus_di") + pl.col("minus_di"))
            ).alias("_dx")
        )

        result = result.with_columns(pl.col("_dx").ewm_mean(alpha=alpha, adjust=False).alias("adx"))

        # Drop intermediate columns
        result = result.drop(
            ["_plus_dm", "_minus_dm", "_tr", "_plus_dm_smooth", "_minus_dm_smooth", "_atr", "_dx"]
        )

        return result

    @property
    def required_columns(self) -> list[str]:
        return ["high", "low", "close"]

    @property
    def output_columns(self) -> list[str]:
        return ["adx", "plus_di", "minus_di"]


class ParabolicSAR(BaseIndicator):
    """
    Parabolic SAR (Stop and Reverse).

    Provides potential entry/exit points based on price direction.
    """

    def __init__(
        self,
        acceleration: float = 0.02,
        maximum: float = 0.2,
        config: IndicatorConfig | None = None,
    ):
        """
        Initialize Parabolic SAR indicator.

        Args:
            acceleration: Acceleration factor (default: 0.02)
            maximum: Maximum acceleration (default: 0.2)
            config: Optional configuration
        """
        super().__init__(config)
        self.acceleration = acceleration or self.config.get("acceleration", 0.02)
        self.maximum = maximum or self.config.get("maximum", 0.2)

    def calculate(self, df: pl.DataFrame) -> pl.DataFrame:
        """Calculate Parabolic SAR."""
        self.validate_input(df)

        # Parabolic SAR requires iterative calculation
        # This is a simplified vectorized approximation
        high = df["high"].to_numpy()
        low = df["low"].to_numpy()
        close = df["close"].to_numpy()

        sar = []
        af = self.acceleration
        trend = 1  # 1 for uptrend, -1 for downtrend
        ep = high[0]  # Extreme point
        sar_value = low[0]

        for i in range(len(close)):
            sar.append(sar_value)

            if i == 0:
                continue

            # Update SAR
            sar_value = sar_value + af * (ep - sar_value)

            # Check for reversal
            if trend == 1:  # Uptrend
                if low[i] < sar_value:
                    # Reverse to downtrend
                    trend = -1
                    sar_value = ep
                    ep = low[i]
                    af = self.acceleration
                else:
                    # Continue uptrend
                    if high[i] > ep:
                        ep = high[i]
                        af = min(af + self.acceleration, self.maximum)
            else:  # Downtrend
                if high[i] > sar_value:
                    # Reverse to uptrend
                    trend = 1
                    sar_value = ep
                    ep = high[i]
                    af = self.acceleration
                else:
                    # Continue downtrend
                    if low[i] < ep:
                        ep = low[i]
                        af = min(af + self.acceleration, self.maximum)

        result = df.with_columns(pl.Series("psar", sar))

        return result

    @property
    def required_columns(self) -> list[str]:
        return ["high", "low", "close"]

    @property
    def output_columns(self) -> list[str]:
        return ["psar"]


class IchimokuCloud(BaseIndicator):
    """
    Ichimoku Cloud (Ichimoku Kinko Hyo).

    Comprehensive indicator showing support/resistance and trend direction.
    """

    def __init__(
        self,
        conversion_period: int = 9,
        base_period: int = 26,
        span_b_period: int = 52,
        displacement: int = 26,
        config: IndicatorConfig | None = None,
    ):
        """
        Initialize Ichimoku Cloud indicator.

        Args:
            conversion_period: Tenkan-sen period (default: 9)
            base_period: Kijun-sen period (default: 26)
            span_b_period: Senkou Span B period (default: 52)
            displacement: Displacement for cloud (default: 26)
            config: Optional configuration
        """
        super().__init__(config)
        self.conversion_period = conversion_period or self.config.get("conversion_period", 9)
        self.base_period = base_period or self.config.get("base_period", 26)
        self.span_b_period = span_b_period or self.config.get("span_b_period", 52)
        self.displacement = displacement or self.config.get("displacement", 26)

    def _calculate_midpoint(self, df: pl.DataFrame, period: int, col_name: str) -> pl.Expr:
        """Calculate midpoint of high/low over period."""
        high_max = pl.col("high").rolling_max(window_size=period)
        low_min = pl.col("low").rolling_min(window_size=period)
        return ((high_max + low_min) / 2).alias(col_name)

    def calculate(self, df: pl.DataFrame) -> pl.DataFrame:
        """Calculate Ichimoku Cloud components."""
        self.validate_input(df)

        # Tenkan-sen (Conversion Line): (9-period high + 9-period low) / 2
        result = df.with_columns(
            self._calculate_midpoint(df, self.conversion_period, "ichimoku_conversion")
        )

        # Kijun-sen (Base Line): (26-period high + 26-period low) / 2
        result = result.with_columns(
            self._calculate_midpoint(result, self.base_period, "ichimoku_base")
        )

        # Senkou Span A (Leading Span A): (Conversion + Base) / 2, shifted forward
        result = result.with_columns(
            ((pl.col("ichimoku_conversion") + pl.col("ichimoku_base")) / 2)
            .shift(-self.displacement)
            .alias("ichimoku_span_a")
        )

        # Senkou Span B (Leading Span B): (52-period high + 52-period low) / 2, shifted forward
        high_max = pl.col("high").rolling_max(window_size=self.span_b_period)
        low_min = pl.col("low").rolling_min(window_size=self.span_b_period)
        result = result.with_columns(
            ((high_max + low_min) / 2).shift(-self.displacement).alias("ichimoku_span_b")
        )

        # Chikou Span (Lagging Span): Close shifted backward
        result = result.with_columns(
            pl.col("close").shift(self.displacement).alias("ichimoku_chikou")
        )

        return result

    @property
    def required_columns(self) -> list[str]:
        return ["high", "low", "close"]

    @property
    def output_columns(self) -> list[str]:
        return [
            "ichimoku_conversion",
            "ichimoku_base",
            "ichimoku_span_a",
            "ichimoku_span_b",
            "ichimoku_chikou",
        ]
