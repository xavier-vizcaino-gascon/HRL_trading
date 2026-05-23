"""
Volatility indicators for technical analysis.

These indicators measure price volatility and market uncertainty:
- Bollinger Bands
- ATR (Average True Range)
- Keltner Channels
- Standard Deviation
- Historical Volatility
- Donchian Channels
"""

import numpy as np
import polars as pl

from ..base import BaseIndicator, IndicatorConfig


class BollingerBands(BaseIndicator):
    """
    Bollinger Bands.

    Volatility bands placed above and below a moving average.
    Outputs: middle band (SMA), upper band, lower band, and bandwidth.
    """

    def __init__(
        self, period: int = 20, std_dev: float = 2.0, config: IndicatorConfig | None = None
    ):
        """
        Initialize Bollinger Bands.

        Args:
            period: Moving average period (default: 20)
            std_dev: Number of standard deviations (default: 2.0)
            config: Optional configuration
        """
        super().__init__(config)
        self.period = period or self.config.get("period", 20)
        self.std_dev = std_dev or self.config.get("std_dev", 2.0)

    def calculate(self, df: pl.DataFrame) -> pl.DataFrame:
        """Calculate Bollinger Bands."""
        self.validate_input(df)

        # Middle band = SMA
        middle = pl.col("close").rolling_mean(window_size=self.period)

        # Standard deviation
        std = pl.col("close").rolling_std(window_size=self.period)

        # Upper and lower bands
        upper = middle + (self.std_dev * std)
        lower = middle - (self.std_dev * std)

        # Bandwidth (measures volatility)
        bandwidth = (upper - lower) / middle

        # %B (shows where price is relative to bands)
        percent_b = (pl.col("close") - lower) / (upper - lower)

        result = df.with_columns(
            [
                middle.alias("bb_middle"),
                upper.alias("bb_upper"),
                lower.alias("bb_lower"),
                bandwidth.alias("bb_bandwidth"),
                percent_b.alias("bb_percent_b"),
            ]
        )

        return result

    @property
    def required_columns(self) -> list[str]:
        return ["close"]

    @property
    def output_columns(self) -> list[str]:
        return ["bb_middle", "bb_upper", "bb_lower", "bb_bandwidth", "bb_percent_b"]


class ATR(BaseIndicator):
    """
    Average True Range (ATR).

    Measures market volatility by calculating average of true ranges.
    """

    def __init__(self, period: int = 14, config: IndicatorConfig | None = None):
        """
        Initialize ATR indicator.

        Args:
            period: Calculation period (default: 14)
            config: Optional configuration
        """
        super().__init__(config)
        self.period = period or self.config.get("period", 14)

    def calculate(self, df: pl.DataFrame) -> pl.DataFrame:
        """Calculate ATR."""
        self.validate_input(df)

        # Calculate True Range
        tr1 = pl.col("high") - pl.col("low")
        tr2 = (pl.col("high") - pl.col("close").shift(1)).abs()
        tr3 = (pl.col("low") - pl.col("close").shift(1)).abs()
        tr = pl.max_horizontal(tr1, tr2, tr3)

        result = df.with_columns(tr.alias("_tr"))

        # ATR is Wilder's smoothing of TR (EMA with alpha = 1/period)
        alpha = 1.0 / self.period
        result = result.with_columns(pl.col("_tr").ewm_mean(alpha=alpha, adjust=False).alias("atr"))

        # Also calculate ATR as percentage of price (normalized ATR)
        result = result.with_columns((100 * pl.col("atr") / pl.col("close")).alias("atr_percent"))

        # Drop intermediate column
        result = result.drop("_tr")

        return result

    @property
    def required_columns(self) -> list[str]:
        return ["high", "low", "close"]

    @property
    def output_columns(self) -> list[str]:
        return ["atr", "atr_percent"]


class KeltnerChannels(BaseIndicator):
    """
    Keltner Channels.

    Volatility-based envelopes around an EMA using ATR for width.
    """

    def __init__(
        self,
        ema_period: int = 20,
        atr_period: int = 10,
        atr_multiplier: float = 2.0,
        config: IndicatorConfig | None = None,
    ):
        """
        Initialize Keltner Channels.

        Args:
            ema_period: EMA period for middle line (default: 20)
            atr_period: ATR calculation period (default: 10)
            atr_multiplier: ATR multiplier for bands (default: 2.0)
            config: Optional configuration
        """
        super().__init__(config)
        self.ema_period = ema_period or self.config.get("ema_period", 20)
        self.atr_period = atr_period or self.config.get("atr_period", 10)
        self.atr_multiplier = atr_multiplier or self.config.get("atr_multiplier", 2.0)

    def calculate(self, df: pl.DataFrame) -> pl.DataFrame:
        """Calculate Keltner Channels."""
        self.validate_input(df)

        # Middle line = EMA
        middle = pl.col("close").ewm_mean(span=self.ema_period, adjust=False)

        # Calculate ATR
        tr1 = pl.col("high") - pl.col("low")
        tr2 = (pl.col("high") - pl.col("close").shift(1)).abs()
        tr3 = (pl.col("low") - pl.col("close").shift(1)).abs()
        tr = pl.max_horizontal(tr1, tr2, tr3)

        alpha = 1.0 / self.atr_period
        atr = tr.ewm_mean(alpha=alpha, adjust=False)

        # Upper and lower channels
        upper = middle + (self.atr_multiplier * atr)
        lower = middle - (self.atr_multiplier * atr)

        result = df.with_columns(
            [
                middle.alias("keltner_middle"),
                upper.alias("keltner_upper"),
                lower.alias("keltner_lower"),
            ]
        )

        return result

    @property
    def required_columns(self) -> list[str]:
        return ["high", "low", "close"]

    @property
    def output_columns(self) -> list[str]:
        return ["keltner_middle", "keltner_upper", "keltner_lower"]


class StandardDeviation(BaseIndicator):
    """
    Standard Deviation.

    Measures dispersion of prices from the mean.
    """

    def __init__(self, periods: list[int] = [10, 20], config: IndicatorConfig | None = None):
        """
        Initialize Standard Deviation indicator.

        Args:
            periods: List of periods to calculate (default: [10, 20])
            config: Optional configuration
        """
        super().__init__(config)
        self.periods = periods or self.config.get("periods", [10, 20])

    def calculate(self, df: pl.DataFrame) -> pl.DataFrame:
        """Calculate Standard Deviation for each period."""
        self.validate_input(df)

        result = df
        for period in self.periods:
            col_name = f"std_{period}"
            result = result.with_columns(
                pl.col("close").rolling_std(window_size=period).alias(col_name)
            )

            # Also calculate normalized (percentage) std
            col_name_pct = f"std_{period}_pct"
            result = result.with_columns(
                (100 * pl.col(col_name) / pl.col("close")).alias(col_name_pct)
            )

        return result

    @property
    def required_columns(self) -> list[str]:
        return ["close"]

    @property
    def output_columns(self) -> list[str]:
        outputs = []
        for period in self.periods:
            outputs.extend([f"std_{period}", f"std_{period}_pct"])
        return outputs


class HistoricalVolatility(BaseIndicator):
    """
    Historical Volatility (HV).

    Annualized volatility based on log returns.
    """

    def __init__(
        self, period: int = 20, trading_periods: int = 252, config: IndicatorConfig | None = None
    ):
        """
        Initialize Historical Volatility indicator.

        Args:
            period: Calculation period (default: 20)
            trading_periods: Number of trading periods per year (default: 252 for daily data)
            config: Optional configuration
        """
        super().__init__(config)
        self.period = period or self.config.get("period", 20)
        self.trading_periods = trading_periods or self.config.get("trading_periods", 252)

    def calculate(self, df: pl.DataFrame) -> pl.DataFrame:
        """Calculate Historical Volatility."""
        self.validate_input(df)

        # Calculate log returns
        log_returns = (pl.col("close") / pl.col("close").shift(1)).log()

        result = df.with_columns(log_returns.alias("_log_returns"))

        # Calculate rolling standard deviation of log returns
        result = result.with_columns(
            pl.col("_log_returns").rolling_std(window_size=self.period).alias("_std_returns")
        )

        # Annualize volatility
        result = result.with_columns(
            (pl.col("_std_returns") * np.sqrt(self.trading_periods) * 100).alias(
                "historical_volatility"
            )
        )

        # Drop intermediate columns
        result = result.drop(["_log_returns", "_std_returns"])

        return result

    @property
    def required_columns(self) -> list[str]:
        return ["close"]

    @property
    def output_columns(self) -> list[str]:
        return ["historical_volatility"]


class DonchianChannels(BaseIndicator):
    """
    Donchian Channels.

    Shows highest high and lowest low over a period.
    """

    def __init__(self, period: int = 20, config: IndicatorConfig | None = None):
        """
        Initialize Donchian Channels.

        Args:
            period: Calculation period (default: 20)
            config: Optional configuration
        """
        super().__init__(config)
        self.period = period or self.config.get("period", 20)

    def calculate(self, df: pl.DataFrame) -> pl.DataFrame:
        """Calculate Donchian Channels."""
        self.validate_input(df)

        # Upper channel = highest high
        upper = pl.col("high").rolling_max(window_size=self.period)

        # Lower channel = lowest low
        lower = pl.col("low").rolling_min(window_size=self.period)

        # Middle channel = average of upper and lower
        middle = (upper + lower) / 2

        # Channel width (volatility measure)
        width = upper - lower
        width_pct = 100 * width / middle

        result = df.with_columns(
            [
                upper.alias("donchian_upper"),
                middle.alias("donchian_middle"),
                lower.alias("donchian_lower"),
                width.alias("donchian_width"),
                width_pct.alias("donchian_width_pct"),
            ]
        )

        return result

    @property
    def required_columns(self) -> list[str]:
        return ["high", "low"]

    @property
    def output_columns(self) -> list[str]:
        return [
            "donchian_upper",
            "donchian_middle",
            "donchian_lower",
            "donchian_width",
            "donchian_width_pct",
        ]


class UlcerIndex(BaseIndicator):
    """
    Ulcer Index.

    Measures downside volatility (drawdown depth and duration).
    """

    def __init__(self, period: int = 14, config: IndicatorConfig | None = None):
        """
        Initialize Ulcer Index.

        Args:
            period: Calculation period (default: 14)
            config: Optional configuration
        """
        super().__init__(config)
        self.period = period or self.config.get("period", 14)

    def calculate(self, df: pl.DataFrame) -> pl.DataFrame:
        """Calculate Ulcer Index."""
        self.validate_input(df)

        # Calculate rolling maximum
        rolling_max = pl.col("close").rolling_max(window_size=self.period)

        # Calculate percentage drawdown from peak
        drawdown = 100 * (pl.col("close") - rolling_max) / rolling_max

        # Square the drawdown
        squared_dd = drawdown**2

        result = df.with_columns(squared_dd.alias("_squared_dd"))

        # Calculate Ulcer Index (sqrt of mean squared drawdown)
        result = result.with_columns(
            pl.col("_squared_dd").rolling_mean(window_size=self.period).sqrt().alias("ulcer_index")
        )

        # Drop intermediate column
        result = result.drop("_squared_dd")

        return result

    @property
    def required_columns(self) -> list[str]:
        return ["close"]

    @property
    def output_columns(self) -> list[str]:
        return ["ulcer_index"]
