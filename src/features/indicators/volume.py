"""
Volume indicators for technical analysis.

These indicators analyze trading volume to confirm trends and identify potential reversals:
- OBV (On-Balance Volume)
- VWAP (Volume Weighted Average Price)
- Volume Profile
- MFI (Money Flow Index)
- Accumulation/Distribution
- Chaikin Money Flow
- Volume Rate of Change
"""

import polars as pl

from ..base import BaseIndicator, IndicatorConfig


class OBV(BaseIndicator):
    """
    On-Balance Volume (OBV).

    Cumulative volume indicator that adds/subtracts volume based on price direction.
    """

    def __init__(self, config: IndicatorConfig | None = None):
        """Initialize OBV indicator."""
        super().__init__(config)

    def calculate(self, df: pl.DataFrame) -> pl.DataFrame:
        """Calculate OBV."""
        self.validate_input(df)

        # Determine direction: 1 if close > previous close, -1 if close < previous close, 0 if equal
        direction = (
            pl.when(pl.col("close") > pl.col("close").shift(1))
            .then(1)
            .when(pl.col("close") < pl.col("close").shift(1))
            .then(-1)
            .otherwise(0)
        )

        # Volume * direction
        signed_volume = direction * pl.col("volume")

        # Cumulative sum
        result = df.with_columns(signed_volume.cum_sum().alias("obv"))

        return result

    @property
    def required_columns(self) -> list[str]:
        return ["close", "volume"]

    @property
    def output_columns(self) -> list[str]:
        return ["obv"]


class VWAP(BaseIndicator):
    """
    Volume Weighted Average Price (VWAP).

    Average price weighted by volume over a period.
    """

    def __init__(self, period: int | None = None, config: IndicatorConfig | None = None):
        """
        Initialize VWAP indicator.

        Args:
            period: Rolling period (None for cumulative, default: None)
            config: Optional configuration
        """
        super().__init__(config)
        self.period = period or self.config.get("period", None)

    def calculate(self, df: pl.DataFrame) -> pl.DataFrame:
        """Calculate VWAP."""
        self.validate_input(df)

        # Typical price
        typical_price = (pl.col("high") + pl.col("low") + pl.col("close")) / 3

        # Price * Volume
        pv = typical_price * pl.col("volume")

        result = df.with_columns([pv.alias("_pv"), typical_price.alias("_tp")])

        if self.period is None:
            # Cumulative VWAP
            result = result.with_columns(
                (pl.col("_pv").cum_sum() / pl.col("volume").cum_sum()).alias("vwap")
            )
        else:
            # Rolling VWAP
            result = result.with_columns(
                (
                    pl.col("_pv").rolling_sum(window_size=self.period)
                    / pl.col("volume").rolling_sum(window_size=self.period)
                ).alias("vwap")
            )

        # Drop intermediate columns
        result = result.drop(["_pv", "_tp"])

        return result

    @property
    def required_columns(self) -> list[str]:
        return ["high", "low", "close", "volume"]

    @property
    def output_columns(self) -> list[str]:
        return ["vwap"]


class MFI(BaseIndicator):
    """
    Money Flow Index (MFI).

    Volume-weighted RSI showing buying/selling pressure.
    """

    def __init__(self, period: int = 14, config: IndicatorConfig | None = None):
        """
        Initialize MFI indicator.

        Args:
            period: Calculation period (default: 14)
            config: Optional configuration
        """
        super().__init__(config)
        self.period = period or self.config.get("period", 14)

    def calculate(self, df: pl.DataFrame) -> pl.DataFrame:
        """Calculate MFI."""
        self.validate_input(df)

        # Typical Price
        typical_price = (pl.col("high") + pl.col("low") + pl.col("close")) / 3

        # Raw Money Flow
        money_flow = typical_price * pl.col("volume")

        result = df.with_columns([typical_price.alias("_tp"), money_flow.alias("_mf")])

        # Determine if money flow is positive or negative
        positive_flow = (
            pl.when(pl.col("_tp") > pl.col("_tp").shift(1)).then(pl.col("_mf")).otherwise(0)
        )

        negative_flow = (
            pl.when(pl.col("_tp") < pl.col("_tp").shift(1)).then(pl.col("_mf")).otherwise(0)
        )

        result = result.with_columns(
            [positive_flow.alias("_pos_flow"), negative_flow.alias("_neg_flow")]
        )

        # Calculate rolling sums
        result = result.with_columns(
            [
                pl.col("_pos_flow").rolling_sum(window_size=self.period).alias("_pos_mf"),
                pl.col("_neg_flow").rolling_sum(window_size=self.period).alias("_neg_mf"),
            ]
        )

        # Money Flow Ratio
        result = result.with_columns((pl.col("_pos_mf") / pl.col("_neg_mf")).alias("_mf_ratio"))

        # MFI = 100 - (100 / (1 + Money Flow Ratio))
        result = result.with_columns((100 - (100 / (1 + pl.col("_mf_ratio")))).alias("mfi"))

        # Drop intermediate columns
        result = result.drop(
            ["_tp", "_mf", "_pos_flow", "_neg_flow", "_pos_mf", "_neg_mf", "_mf_ratio"]
        )

        return result

    @property
    def required_columns(self) -> list[str]:
        return ["high", "low", "close", "volume"]

    @property
    def output_columns(self) -> list[str]:
        return ["mfi"]


class AccumulationDistribution(BaseIndicator):
    """
    Accumulation/Distribution Line (A/D).

    Cumulative indicator relating price and volume to show money flow.
    """

    def __init__(self, config: IndicatorConfig | None = None):
        """Initialize A/D indicator."""
        super().__init__(config)

    def calculate(self, df: pl.DataFrame) -> pl.DataFrame:
        """Calculate A/D Line."""
        self.validate_input(df)

        # Money Flow Multiplier
        mfm = ((pl.col("close") - pl.col("low")) - (pl.col("high") - pl.col("close"))) / (
            pl.col("high") - pl.col("low")
        )

        # Money Flow Volume
        mfv = mfm * pl.col("volume")

        # A/D Line = cumulative sum of MFV
        result = df.with_columns(mfv.cum_sum().alias("ad_line"))

        return result

    @property
    def required_columns(self) -> list[str]:
        return ["high", "low", "close", "volume"]

    @property
    def output_columns(self) -> list[str]:
        return ["ad_line"]


class ChaikinMoneyFlow(BaseIndicator):
    """
    Chaikin Money Flow (CMF).

    Measures buying and selling pressure over a period.
    """

    def __init__(self, period: int = 20, config: IndicatorConfig | None = None):
        """
        Initialize CMF indicator.

        Args:
            period: Calculation period (default: 20)
            config: Optional configuration
        """
        super().__init__(config)
        self.period = period or self.config.get("period", 20)

    def calculate(self, df: pl.DataFrame) -> pl.DataFrame:
        """Calculate CMF."""
        self.validate_input(df)

        # Money Flow Multiplier
        mfm = ((pl.col("close") - pl.col("low")) - (pl.col("high") - pl.col("close"))) / (
            pl.col("high") - pl.col("low")
        )

        # Money Flow Volume
        mfv = mfm * pl.col("volume")

        result = df.with_columns(mfv.alias("_mfv"))

        # CMF = Sum(MFV, period) / Sum(Volume, period)
        result = result.with_columns(
            (
                pl.col("_mfv").rolling_sum(window_size=self.period)
                / pl.col("volume").rolling_sum(window_size=self.period)
            ).alias("cmf")
        )

        # Drop intermediate column
        result = result.drop("_mfv")

        return result

    @property
    def required_columns(self) -> list[str]:
        return ["high", "low", "close", "volume"]

    @property
    def output_columns(self) -> list[str]:
        return ["cmf"]


class VolumeROC(BaseIndicator):
    """
    Volume Rate of Change (VROC).

    Measures percentage change in volume over a period.
    """

    def __init__(self, periods: list[int] = [10, 20], config: IndicatorConfig | None = None):
        """
        Initialize VROC indicator.

        Args:
            periods: List of periods to calculate (default: [10, 20])
            config: Optional configuration
        """
        super().__init__(config)
        self.periods = periods or self.config.get("periods", [10, 20])

    def calculate(self, df: pl.DataFrame) -> pl.DataFrame:
        """Calculate VROC for each period."""
        self.validate_input(df)

        result = df
        for period in self.periods:
            col_name = f"vroc_{period}"
            # VROC = ((Volume - Volume_n_periods_ago) / Volume_n_periods_ago) * 100
            vroc = (
                100
                * (pl.col("volume") - pl.col("volume").shift(period))
                / pl.col("volume").shift(period)
            )
            result = result.with_columns(vroc.alias(col_name))

        return result

    @property
    def required_columns(self) -> list[str]:
        return ["volume"]

    @property
    def output_columns(self) -> list[str]:
        return [f"vroc_{period}" for period in self.periods]


class ForceIndex(BaseIndicator):
    """
    Force Index.

    Combines price change and volume to measure buying/selling pressure.
    """

    def __init__(self, period: int = 13, config: IndicatorConfig | None = None):
        """
        Initialize Force Index.

        Args:
            period: EMA smoothing period (default: 13)
            config: Optional configuration
        """
        super().__init__(config)
        self.period = period or self.config.get("period", 13)

    def calculate(self, df: pl.DataFrame) -> pl.DataFrame:
        """Calculate Force Index."""
        self.validate_input(df)

        # Raw Force Index = (Close - Previous Close) * Volume
        raw_fi = (pl.col("close") - pl.col("close").shift(1)) * pl.col("volume")

        result = df.with_columns(raw_fi.alias("_raw_fi"))

        # Smooth with EMA
        result = result.with_columns(
            pl.col("_raw_fi").ewm_mean(span=self.period, adjust=False).alias("force_index")
        )

        # Drop intermediate column
        result = result.drop("_raw_fi")

        return result

    @property
    def required_columns(self) -> list[str]:
        return ["close", "volume"]

    @property
    def output_columns(self) -> list[str]:
        return ["force_index"]


class EaseOfMovement(BaseIndicator):
    """
    Ease of Movement (EMV).

    Relates price change to volume, showing how easily price moves.
    """

    def __init__(self, period: int = 14, config: IndicatorConfig | None = None):
        """
        Initialize EMV indicator.

        Args:
            period: SMA smoothing period (default: 14)
            config: Optional configuration
        """
        super().__init__(config)
        self.period = period or self.config.get("period", 14)

    def calculate(self, df: pl.DataFrame) -> pl.DataFrame:
        """Calculate EMV."""
        self.validate_input(df)

        # Distance moved = midpoint change
        midpoint = (pl.col("high") + pl.col("low")) / 2
        distance = midpoint - midpoint.shift(1)

        # Box ratio = (Volume / scale) / (High - Low)
        # Using 1,000,000 as scale factor
        box_ratio = (pl.col("volume") / 1_000_000) / (pl.col("high") - pl.col("low"))

        # EMV = Distance / Box Ratio
        raw_emv = distance / box_ratio

        result = df.with_columns(raw_emv.alias("_raw_emv"))

        # Smooth with SMA
        result = result.with_columns(
            pl.col("_raw_emv").rolling_mean(window_size=self.period).alias("emv")
        )

        # Drop intermediate column
        result = result.drop("_raw_emv")

        return result

    @property
    def required_columns(self) -> list[str]:
        return ["high", "low", "volume"]

    @property
    def output_columns(self) -> list[str]:
        return ["emv"]


class VolumeWeightedMACD(BaseIndicator):
    """
    Volume-Weighted MACD.

    MACD indicator using volume-weighted prices.
    """

    def __init__(
        self,
        fast_period: int = 12,
        slow_period: int = 26,
        signal_period: int = 9,
        config: IndicatorConfig | None = None,
    ):
        """
        Initialize Volume-Weighted MACD.

        Args:
            fast_period: Fast EMA period (default: 12)
            slow_period: Slow EMA period (default: 26)
            signal_period: Signal line period (default: 9)
            config: Optional configuration
        """
        super().__init__(config)
        self.fast_period = fast_period or self.config.get("fast_period", 12)
        self.slow_period = slow_period or self.config.get("slow_period", 26)
        self.signal_period = signal_period or self.config.get("signal_period", 9)

    def calculate(self, df: pl.DataFrame) -> pl.DataFrame:
        """Calculate Volume-Weighted MACD."""
        self.validate_input(df)

        # Volume-weighted price
        vw_price = pl.col("close") * pl.col("volume")

        result = df.with_columns(vw_price.alias("_vw_price"))

        # Volume-weighted EMAs
        fast_ema = pl.col("_vw_price").ewm_mean(span=self.fast_period, adjust=False) / pl.col(
            "volume"
        ).ewm_mean(span=self.fast_period, adjust=False)

        slow_ema = pl.col("_vw_price").ewm_mean(span=self.slow_period, adjust=False) / pl.col(
            "volume"
        ).ewm_mean(span=self.slow_period, adjust=False)

        # MACD line
        macd_line = fast_ema - slow_ema

        result = result.with_columns(macd_line.alias("vw_macd"))

        # Signal line
        result = result.with_columns(
            pl.col("vw_macd")
            .ewm_mean(span=self.signal_period, adjust=False)
            .alias("vw_macd_signal")
        )

        # Histogram
        result = result.with_columns(
            (pl.col("vw_macd") - pl.col("vw_macd_signal")).alias("vw_macd_histogram")
        )

        # Drop intermediate column
        result = result.drop("_vw_price")

        return result

    @property
    def required_columns(self) -> list[str]:
        return ["close", "volume"]

    @property
    def output_columns(self) -> list[str]:
        return ["vw_macd", "vw_macd_signal", "vw_macd_histogram"]
