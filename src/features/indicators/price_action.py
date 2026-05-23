"""
Price action features for technical analysis.

These features analyze candlestick patterns and price structure:
- Candle body/wick ratios
- High-Low range metrics
- Pivot points
- Support/Resistance levels
- Gap detection
- Basic candlestick patterns
"""

import polars as pl

from ..base import BaseIndicator, IndicatorConfig


class CandleMetrics(BaseIndicator):
    """
    Basic candlestick metrics.

    Calculates body size, wick sizes, and ratios.
    """

    def __init__(self, config: IndicatorConfig | None = None):
        """Initialize Candle Metrics calculator."""
        super().__init__(config)

    def calculate(self, df: pl.DataFrame) -> pl.DataFrame:
        """Calculate candle metrics."""
        self.validate_input(df)

        # Body size (absolute and percentage)
        body = (pl.col("close") - pl.col("open")).abs()
        body_pct = 100 * body / pl.col("open")

        # Upper wick
        upper_wick = pl.col("high") - pl.max_horizontal(pl.col("open"), pl.col("close"))

        # Lower wick
        lower_wick = pl.min_horizontal(pl.col("open"), pl.col("close")) - pl.col("low")

        # Total range
        total_range = pl.col("high") - pl.col("low")
        range_pct = 100 * total_range / pl.col("open")

        # Body to range ratio
        body_range_ratio = body / total_range

        # Upper wick ratio
        upper_wick_ratio = upper_wick / total_range

        # Lower wick ratio
        lower_wick_ratio = lower_wick / total_range

        # Direction: 1 for bullish (close > open), -1 for bearish, 0 for doji
        direction = (
            pl.when(pl.col("close") > pl.col("open"))
            .then(1)
            .when(pl.col("close") < pl.col("open"))
            .then(-1)
            .otherwise(0)
        )

        result = df.with_columns(
            [
                body.alias("candle_body"),
                body_pct.alias("candle_body_pct"),
                upper_wick.alias("candle_upper_wick"),
                lower_wick.alias("candle_lower_wick"),
                total_range.alias("candle_range"),
                range_pct.alias("candle_range_pct"),
                body_range_ratio.alias("candle_body_ratio"),
                upper_wick_ratio.alias("candle_upper_wick_ratio"),
                lower_wick_ratio.alias("candle_lower_wick_ratio"),
                direction.alias("candle_direction"),
            ]
        )

        return result

    @property
    def required_columns(self) -> list[str]:
        return ["open", "high", "low", "close"]

    @property
    def output_columns(self) -> list[str]:
        return [
            "candle_body",
            "candle_body_pct",
            "candle_upper_wick",
            "candle_lower_wick",
            "candle_range",
            "candle_range_pct",
            "candle_body_ratio",
            "candle_upper_wick_ratio",
            "candle_lower_wick_ratio",
            "candle_direction",
        ]


class PivotPoints(BaseIndicator):
    """
    Pivot Points.

    Classic support/resistance levels based on previous period's OHLC.
    """

    def __init__(self, config: IndicatorConfig | None = None):
        """Initialize Pivot Points calculator."""
        super().__init__(config)

    def calculate(self, df: pl.DataFrame) -> pl.DataFrame:
        """Calculate pivot points."""
        self.validate_input(df)

        # Use previous period's data
        prev_high = pl.col("high").shift(1)
        prev_low = pl.col("low").shift(1)
        prev_close = pl.col("close").shift(1)

        # Pivot point (PP)
        pp = (prev_high + prev_low + prev_close) / 3

        # Support and Resistance levels
        r1 = (2 * pp) - prev_low
        s1 = (2 * pp) - prev_high

        r2 = pp + (prev_high - prev_low)
        s2 = pp - (prev_high - prev_low)

        r3 = prev_high + 2 * (pp - prev_low)
        s3 = prev_low - 2 * (prev_high - pp)

        result = df.with_columns(
            [
                pp.alias("pivot_point"),
                r1.alias("pivot_r1"),
                r2.alias("pivot_r2"),
                r3.alias("pivot_r3"),
                s1.alias("pivot_s1"),
                s2.alias("pivot_s2"),
                s3.alias("pivot_s3"),
            ]
        )

        # Distance from pivot (normalized)
        result = result.with_columns(
            [
                (100 * (pl.col("close") - pl.col("pivot_point")) / pl.col("pivot_point")).alias(
                    "pivot_distance_pct"
                )
            ]
        )

        return result

    @property
    def required_columns(self) -> list[str]:
        return ["high", "low", "close"]

    @property
    def output_columns(self) -> list[str]:
        return [
            "pivot_point",
            "pivot_r1",
            "pivot_r2",
            "pivot_r3",
            "pivot_s1",
            "pivot_s2",
            "pivot_s3",
            "pivot_distance_pct",
        ]


class GapDetection(BaseIndicator):
    """
    Gap Detection.

    Identifies price gaps between candles.
    """

    def __init__(self, min_gap_pct: float = 0.5, config: IndicatorConfig | None = None):
        """
        Initialize Gap Detection.

        Args:
            min_gap_pct: Minimum gap size as percentage (default: 0.5%)
            config: Optional configuration
        """
        super().__init__(config)
        self.min_gap_pct = min_gap_pct or self.config.get("min_gap_pct", 0.5)

    def calculate(self, df: pl.DataFrame) -> pl.DataFrame:
        """Detect gaps."""
        self.validate_input(df)

        prev_high = pl.col("high").shift(1)
        prev_low = pl.col("low").shift(1)
        prev_close = pl.col("close").shift(1)

        # Gap up: current low > previous high
        gap_up = pl.col("low") > prev_high
        gap_up_size = pl.col("low") - prev_high
        gap_up_pct = 100 * gap_up_size / prev_close

        # Gap down: current high < previous low
        gap_down = pl.col("high") < prev_low
        gap_down_size = prev_low - pl.col("high")
        gap_down_pct = 100 * gap_down_size / prev_close

        # Combined gap indicator: 1 for gap up, -1 for gap down, 0 for no gap
        has_gap = (
            pl.when(gap_up & (gap_up_pct >= self.min_gap_pct))
            .then(1)
            .when(gap_down & (gap_down_pct >= self.min_gap_pct))
            .then(-1)
            .otherwise(0)
        )

        # Gap size (absolute)
        gap_size = (
            pl.when(has_gap == 1)
            .then(gap_up_size)
            .when(has_gap == -1)
            .then(gap_down_size)
            .otherwise(0)
        )

        # Gap percentage
        gap_pct = (
            pl.when(has_gap == 1)
            .then(gap_up_pct)
            .when(has_gap == -1)
            .then(gap_down_pct)
            .otherwise(0)
        )

        result = df.with_columns(
            [has_gap.alias("gap"), gap_size.alias("gap_size"), gap_pct.alias("gap_pct")]
        )

        return result

    @property
    def required_columns(self) -> list[str]:
        return ["high", "low", "close"]

    @property
    def output_columns(self) -> list[str]:
        return ["gap", "gap_size", "gap_pct"]


class CandlestickPatterns(BaseIndicator):
    """
    Basic Candlestick Patterns.

    Detects common single and two-candle patterns.
    """

    def __init__(self, config: IndicatorConfig | None = None):
        """Initialize Candlestick Patterns detector."""
        super().__init__(config)

    def calculate(self, df: pl.DataFrame) -> pl.DataFrame:
        """Detect candlestick patterns."""
        self.validate_input(df)

        # Calculate basic metrics first
        body = (pl.col("close") - pl.col("open")).abs()
        total_range = pl.col("high") - pl.col("low")
        body_ratio = body / total_range
        upper_wick = pl.col("high") - pl.max_horizontal(pl.col("open"), pl.col("close"))
        lower_wick = pl.min_horizontal(pl.col("open"), pl.col("close")) - pl.col("low")

        # Previous candle metrics
        prev_body = body.shift(1)
        prev_range = total_range.shift(1)
        prev_close = pl.col("close").shift(1)
        prev_open = pl.col("open").shift(1)

        result = df

        # 1. Doji: small body relative to range
        is_doji = body_ratio < 0.1
        result = result.with_columns(is_doji.cast(pl.Int8).alias("pattern_doji"))

        # 2. Hammer/Hanging Man: small body at top, long lower wick
        is_hammer = (body_ratio < 0.3) & (lower_wick > 2 * body) & (upper_wick < body)
        result = result.with_columns(is_hammer.cast(pl.Int8).alias("pattern_hammer"))

        # 3. Inverted Hammer/Shooting Star: small body at bottom, long upper wick
        is_inverted_hammer = (body_ratio < 0.3) & (upper_wick > 2 * body) & (lower_wick < body)
        result = result.with_columns(
            is_inverted_hammer.cast(pl.Int8).alias("pattern_inverted_hammer")
        )

        # 4. Marubozu: very small wicks, large body
        is_marubozu = body_ratio > 0.9
        result = result.with_columns(is_marubozu.cast(pl.Int8).alias("pattern_marubozu"))

        # 5. Spinning Top: small body in middle of range
        is_spinning_top = (
            (body_ratio < 0.3) & (body_ratio > 0.1) & (upper_wick > body) & (lower_wick > body)
        )
        result = result.with_columns(is_spinning_top.cast(pl.Int8).alias("pattern_spinning_top"))

        # 6. Bullish Engulfing: current bullish candle engulfs previous bearish
        is_bullish_engulfing = (
            (pl.col("close") > pl.col("open"))
            & (prev_close < prev_open)
            & (pl.col("close") > prev_open)
            & (pl.col("open") < prev_close)
        )
        result = result.with_columns(
            is_bullish_engulfing.cast(pl.Int8).alias("pattern_bullish_engulfing")
        )

        # 7. Bearish Engulfing: current bearish candle engulfs previous bullish
        is_bearish_engulfing = (
            (pl.col("close") < pl.col("open"))
            & (prev_close > prev_open)
            & (pl.col("close") < prev_open)
            & (pl.col("open") > prev_close)
        )
        result = result.with_columns(
            is_bearish_engulfing.cast(pl.Int8).alias("pattern_bearish_engulfing")
        )

        # 8. Strong Momentum: large body with small wicks
        is_strong_momentum = (body_ratio > 0.7) & (body > prev_body * 1.5)
        result = result.with_columns(
            is_strong_momentum.cast(pl.Int8).alias("pattern_strong_momentum")
        )

        return result

    @property
    def required_columns(self) -> list[str]:
        return ["open", "high", "low", "close"]

    @property
    def output_columns(self) -> list[str]:
        return [
            "pattern_doji",
            "pattern_hammer",
            "pattern_inverted_hammer",
            "pattern_marubozu",
            "pattern_spinning_top",
            "pattern_bullish_engulfing",
            "pattern_bearish_engulfing",
            "pattern_strong_momentum",
        ]


class PricePosition(BaseIndicator):
    """
    Price Position Relative to Range.

    Shows where price is within various ranges.
    """

    def __init__(self, periods: list[int] = [10, 20, 50], config: IndicatorConfig | None = None):
        """
        Initialize Price Position calculator.

        Args:
            periods: List of periods to calculate (default: [10, 20, 50])
            config: Optional configuration
        """
        super().__init__(config)
        self.periods = periods or self.config.get("periods", [10, 20, 50])

    def calculate(self, df: pl.DataFrame) -> pl.DataFrame:
        """Calculate price position metrics."""
        self.validate_input(df)

        result = df

        for period in self.periods:
            # Position within high-low range
            high_max = pl.col("high").rolling_max(window_size=period)
            low_min = pl.col("low").rolling_min(window_size=period)
            range_size = high_max - low_min

            # Position: 0 = at bottom, 100 = at top
            position = 100 * (pl.col("close") - low_min) / range_size

            result = result.with_columns(position.alias(f"price_position_{period}"))

            # Distance from high (negative values)
            dist_from_high = 100 * (pl.col("close") - high_max) / high_max
            result = result.with_columns(dist_from_high.alias(f"dist_from_high_{period}"))

            # Distance from low (positive values)
            dist_from_low = 100 * (pl.col("close") - low_min) / low_min
            result = result.with_columns(dist_from_low.alias(f"dist_from_low_{period}"))

        return result

    @property
    def required_columns(self) -> list[str]:
        return ["high", "low", "close"]

    @property
    def output_columns(self) -> list[str]:
        outputs = []
        for period in self.periods:
            outputs.extend(
                [f"price_position_{period}", f"dist_from_high_{period}", f"dist_from_low_{period}"]
            )
        return outputs


class FractalIndicator(BaseIndicator):
    """
    Fractal Indicator.

    Identifies fractal highs and lows (Williams Fractals).
    """

    def __init__(self, period: int = 2, config: IndicatorConfig | None = None):
        """
        Initialize Fractal Indicator.

        Args:
            period: Number of bars on each side (default: 2 for 5-bar pattern)
            config: Optional configuration
        """
        super().__init__(config)
        self.period = period or self.config.get("period", 2)

    def calculate(self, df: pl.DataFrame) -> pl.DataFrame:
        """Detect fractal patterns."""
        self.validate_input(df)

        # Fractal Up: middle high is highest among period bars on each side
        is_fractal_up = pl.lit(True)
        for i in range(1, self.period + 1):
            is_fractal_up = (
                is_fractal_up
                & (pl.col("high") > pl.col("high").shift(i))
                & (pl.col("high") > pl.col("high").shift(-i))
            )

        # Fractal Down: middle low is lowest among period bars on each side
        is_fractal_down = pl.lit(True)
        for i in range(1, self.period + 1):
            is_fractal_down = (
                is_fractal_down
                & (pl.col("low") < pl.col("low").shift(i))
                & (pl.col("low") < pl.col("low").shift(-i))
            )

        result = df.with_columns(
            [
                is_fractal_up.cast(pl.Int8).alias("fractal_up"),
                is_fractal_down.cast(pl.Int8).alias("fractal_down"),
            ]
        )

        # Mark fractal prices
        result = result.with_columns(
            [
                pl.when(pl.col("fractal_up") == 1)
                .then(pl.col("high"))
                .otherwise(None)
                .alias("fractal_up_price"),
                pl.when(pl.col("fractal_down") == 1)
                .then(pl.col("low"))
                .otherwise(None)
                .alias("fractal_down_price"),
            ]
        )

        return result

    @property
    def required_columns(self) -> list[str]:
        return ["high", "low"]

    @property
    def output_columns(self) -> list[str]:
        return ["fractal_up", "fractal_down", "fractal_up_price", "fractal_down_price"]


class TrendStrength(BaseIndicator):
    """
    Trend Strength Indicator.

    Measures strength of uptrend or downtrend based on consecutive movements.
    """

    def __init__(self, config: IndicatorConfig | None = None):
        """Initialize Trend Strength calculator."""
        super().__init__(config)

    def calculate(self, df: pl.DataFrame) -> pl.DataFrame:
        """Calculate trend strength."""
        self.validate_input(df)

        # Direction: 1 for up, -1 for down
        direction = (
            pl.when(pl.col("close") > pl.col("close").shift(1))
            .then(1)
            .when(pl.col("close") < pl.col("close").shift(1))
            .then(-1)
            .otherwise(0)
        )

        result = df.with_columns(direction.alias("_direction"))

        # Count consecutive same-direction moves using cumsum of direction changes
        direction_change = (pl.col("_direction") != pl.col("_direction").shift(1)).cast(pl.Int32)
        streak_id = direction_change.cum_sum()

        result = result.with_columns(streak_id.alias("_streak_id"))

        # Count streak length
        result = result.with_columns(
            pl.col("_streak_id").cum_count().over("_streak_id").alias("trend_strength")
        )

        # Make negative for downtrends
        result = result.with_columns(
            (pl.col("trend_strength") * pl.col("_direction")).alias("trend_strength")
        )

        # Drop intermediate columns
        result = result.drop(["_direction", "_streak_id"])

        return result

    @property
    def required_columns(self) -> list[str]:
        return ["close"]

    @property
    def output_columns(self) -> list[str]:
        return ["trend_strength"]
