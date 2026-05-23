"""
Momentum indicators for technical analysis.

These indicators measure the rate of price change and overbought/oversold conditions:
- RSI (Relative Strength Index)
- Stochastic Oscillator
- Williams %R
- CCI (Commodity Channel Index)
- ROC (Rate of Change)
- MOM (Momentum)
"""

import polars as pl

from ..base import BaseIndicator, IndicatorConfig


class RSI(BaseIndicator):
    """
    Relative Strength Index (RSI).

    Momentum oscillator measuring speed and magnitude of price changes.
    Values range from 0 to 100. Typically, >70 is overbought, <30 is oversold.
    """

    def __init__(self, period: int = 14, config: IndicatorConfig | None = None):
        """
        Initialize RSI indicator.

        Args:
            period: Calculation period (default: 14)
            config: Optional configuration
        """
        super().__init__(config)
        self.period = period or self.config.get("period", 14)

    def calculate(self, df: pl.DataFrame) -> pl.DataFrame:
        """Calculate RSI."""
        self.validate_input(df)

        # Calculate price changes
        delta = pl.col("close") - pl.col("close").shift(1)

        # Separate gains and losses
        gain = pl.when(delta > 0).then(delta).otherwise(0)
        loss = pl.when(delta < 0).then(-delta).otherwise(0)

        result = df.with_columns([gain.alias("_gain"), loss.alias("_loss")])

        # Use Wilder's smoothing (EMA with alpha = 1/period)
        alpha = 1.0 / self.period
        result = result.with_columns(
            [
                pl.col("_gain").ewm_mean(alpha=alpha, adjust=False).alias("_avg_gain"),
                pl.col("_loss").ewm_mean(alpha=alpha, adjust=False).alias("_avg_loss"),
            ]
        )

        # Calculate RS and RSI
        result = result.with_columns(
            pl.when(pl.col("_avg_loss") == 0)
            .then(100)
            .otherwise(100 - (100 / (1 + pl.col("_avg_gain") / pl.col("_avg_loss"))))
            .alias("rsi")
        )

        # Drop intermediate columns
        result = result.drop(["_gain", "_loss", "_avg_gain", "_avg_loss"])

        return result

    @property
    def required_columns(self) -> list[str]:
        return ["close"]

    @property
    def output_columns(self) -> list[str]:
        return ["rsi"]


class StochasticOscillator(BaseIndicator):
    """
    Stochastic Oscillator.

    Compares closing price to price range over a period.
    Outputs %K (fast) and %D (slow) lines.
    """

    def __init__(
        self,
        k_period: int = 14,
        d_period: int = 3,
        smooth_k: int = 3,
        config: IndicatorConfig | None = None,
    ):
        """
        Initialize Stochastic Oscillator.

        Args:
            k_period: Period for %K calculation (default: 14)
            d_period: Period for %D smoothing (default: 3)
            smooth_k: Smoothing period for %K (default: 3)
            config: Optional configuration
        """
        super().__init__(config)
        self.k_period = k_period or self.config.get("k_period", 14)
        self.d_period = d_period or self.config.get("d_period", 3)
        self.smooth_k = smooth_k or self.config.get("smooth_k", 3)

    def calculate(self, df: pl.DataFrame) -> pl.DataFrame:
        """Calculate Stochastic %K and %D."""
        self.validate_input(df)

        # Calculate highest high and lowest low over period
        highest_high = pl.col("high").rolling_max(window_size=self.k_period)
        lowest_low = pl.col("low").rolling_min(window_size=self.k_period)

        # Calculate raw %K
        raw_k = 100 * (pl.col("close") - lowest_low) / (highest_high - lowest_low)

        result = df.with_columns(raw_k.alias("_raw_k"))

        # Smooth %K
        result = result.with_columns(
            pl.col("_raw_k").rolling_mean(window_size=self.smooth_k).alias("stoch_k")
        )

        # Calculate %D (SMA of %K)
        result = result.with_columns(
            pl.col("stoch_k").rolling_mean(window_size=self.d_period).alias("stoch_d")
        )

        # Drop intermediate column
        result = result.drop("_raw_k")

        return result

    @property
    def required_columns(self) -> list[str]:
        return ["high", "low", "close"]

    @property
    def output_columns(self) -> list[str]:
        return ["stoch_k", "stoch_d"]


class WilliamsR(BaseIndicator):
    """
    Williams %R.

    Momentum indicator measuring overbought/oversold levels.
    Values range from -100 to 0.
    """

    def __init__(self, period: int = 14, config: IndicatorConfig | None = None):
        """
        Initialize Williams %R indicator.

        Args:
            period: Calculation period (default: 14)
            config: Optional configuration
        """
        super().__init__(config)
        self.period = period or self.config.get("period", 14)

    def calculate(self, df: pl.DataFrame) -> pl.DataFrame:
        """Calculate Williams %R."""
        self.validate_input(df)

        # Calculate highest high and lowest low
        highest_high = pl.col("high").rolling_max(window_size=self.period)
        lowest_low = pl.col("low").rolling_min(window_size=self.period)

        # Williams %R = -100 * (Highest High - Close) / (Highest High - Lowest Low)
        williams_r = -100 * (highest_high - pl.col("close")) / (highest_high - lowest_low)

        result = df.with_columns(williams_r.alias("williams_r"))

        return result

    @property
    def required_columns(self) -> list[str]:
        return ["high", "low", "close"]

    @property
    def output_columns(self) -> list[str]:
        return ["williams_r"]


class CCI(BaseIndicator):
    """
    Commodity Channel Index (CCI).

    Measures deviation from average price.
    Typically, values above +100 indicate overbought, below -100 indicate oversold.
    """

    def __init__(
        self, period: int = 20, constant: float = 0.015, config: IndicatorConfig | None = None
    ):
        """
        Initialize CCI indicator.

        Args:
            period: Calculation period (default: 20)
            constant: Scaling constant (default: 0.015)
            config: Optional configuration
        """
        super().__init__(config)
        self.period = period or self.config.get("period", 20)
        self.constant = constant or self.config.get("constant", 0.015)

    def calculate(self, df: pl.DataFrame) -> pl.DataFrame:
        """Calculate CCI."""
        self.validate_input(df)

        # Calculate Typical Price
        tp = (pl.col("high") + pl.col("low") + pl.col("close")) / 3

        result = df.with_columns(tp.alias("_tp"))

        # Calculate SMA of Typical Price
        result = result.with_columns(
            pl.col("_tp").rolling_mean(window_size=self.period).alias("_sma_tp")
        )

        # Calculate Mean Deviation
        result = result.with_columns(
            (pl.col("_tp") - pl.col("_sma_tp"))
            .abs()
            .rolling_mean(window_size=self.period)
            .alias("_mean_dev")
        )

        # Calculate CCI
        result = result.with_columns(
            ((pl.col("_tp") - pl.col("_sma_tp")) / (self.constant * pl.col("_mean_dev"))).alias(
                "cci"
            )
        )

        # Drop intermediate columns
        result = result.drop(["_tp", "_sma_tp", "_mean_dev"])

        return result

    @property
    def required_columns(self) -> list[str]:
        return ["high", "low", "close"]

    @property
    def output_columns(self) -> list[str]:
        return ["cci"]


class ROC(BaseIndicator):
    """
    Rate of Change (ROC).

    Measures percentage change in price over a period.
    """

    def __init__(self, periods: list[int] = [10, 20], config: IndicatorConfig | None = None):
        """
        Initialize ROC indicator.

        Args:
            periods: List of periods to calculate (default: [10, 20])
            config: Optional configuration
        """
        super().__init__(config)
        self.periods = periods or self.config.get("periods", [10, 20])

    def calculate(self, df: pl.DataFrame) -> pl.DataFrame:
        """Calculate ROC for each period."""
        self.validate_input(df)

        result = df
        for period in self.periods:
            col_name = f"roc_{period}"
            # ROC = ((Close - Close_n_periods_ago) / Close_n_periods_ago) * 100
            roc = (
                100
                * (pl.col("close") - pl.col("close").shift(period))
                / pl.col("close").shift(period)
            )
            result = result.with_columns(roc.alias(col_name))

        return result

    @property
    def required_columns(self) -> list[str]:
        return ["close"]

    @property
    def output_columns(self) -> list[str]:
        return [f"roc_{period}" for period in self.periods]


class Momentum(BaseIndicator):
    """
    Momentum (MOM).

    Measures absolute price change over a period.
    """

    def __init__(self, periods: list[int] = [10, 20], config: IndicatorConfig | None = None):
        """
        Initialize Momentum indicator.

        Args:
            periods: List of periods to calculate (default: [10, 20])
            config: Optional configuration
        """
        super().__init__(config)
        self.periods = periods or self.config.get("periods", [10, 20])

    def calculate(self, df: pl.DataFrame) -> pl.DataFrame:
        """Calculate Momentum for each period."""
        self.validate_input(df)

        result = df
        for period in self.periods:
            col_name = f"mom_{period}"
            # MOM = Close - Close_n_periods_ago
            mom = pl.col("close") - pl.col("close").shift(period)
            result = result.with_columns(mom.alias(col_name))

        return result

    @property
    def required_columns(self) -> list[str]:
        return ["close"]

    @property
    def output_columns(self) -> list[str]:
        return [f"mom_{period}" for period in self.periods]


class UltimateOscillator(BaseIndicator):
    """
    Ultimate Oscillator.

    Combines three different time periods to reduce false signals.
    """

    def __init__(
        self,
        period1: int = 7,
        period2: int = 14,
        period3: int = 28,
        config: IndicatorConfig | None = None,
    ):
        """
        Initialize Ultimate Oscillator.

        Args:
            period1: Short period (default: 7)
            period2: Medium period (default: 14)
            period3: Long period (default: 28)
            config: Optional configuration
        """
        super().__init__(config)
        self.period1 = period1 or self.config.get("period1", 7)
        self.period2 = period2 or self.config.get("period2", 14)
        self.period3 = period3 or self.config.get("period3", 28)

    def calculate(self, df: pl.DataFrame) -> pl.DataFrame:
        """Calculate Ultimate Oscillator."""
        self.validate_input(df)

        # Calculate Buying Pressure
        true_low = pl.min_horizontal(pl.col("low"), pl.col("close").shift(1))
        bp = pl.col("close") - true_low

        # Calculate True Range
        tr1 = pl.col("high") - pl.col("low")
        tr2 = (pl.col("high") - pl.col("close").shift(1)).abs()
        tr3 = (pl.col("low") - pl.col("close").shift(1)).abs()
        tr = pl.max_horizontal(tr1, tr2, tr3)

        result = df.with_columns([bp.alias("_bp"), tr.alias("_tr")])

        # Calculate averages for each period
        for period in [self.period1, self.period2, self.period3]:
            bp_sum = pl.col("_bp").rolling_sum(window_size=period)
            tr_sum = pl.col("_tr").rolling_sum(window_size=period)
            result = result.with_columns((bp_sum / tr_sum).alias(f"_avg_{period}"))

        # Calculate Ultimate Oscillator
        result = result.with_columns(
            (
                100
                * (
                    (4 * pl.col(f"_avg_{self.period1}"))
                    + (2 * pl.col(f"_avg_{self.period2}"))
                    + pl.col(f"_avg_{self.period3}")
                )
                / 7
            ).alias("ultimate_oscillator")
        )

        # Drop intermediate columns
        result = result.drop(
            ["_bp", "_tr", f"_avg_{self.period1}", f"_avg_{self.period2}", f"_avg_{self.period3}"]
        )

        return result

    @property
    def required_columns(self) -> list[str]:
        return ["high", "low", "close"]

    @property
    def output_columns(self) -> list[str]:
        return ["ultimate_oscillator"]
