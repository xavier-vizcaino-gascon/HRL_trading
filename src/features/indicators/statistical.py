"""
Statistical features for technical analysis.

These features provide statistical measures and transformations of price data:
- Returns (simple, log)
- Higher moments (skewness, kurtosis)
- Z-score normalization
- Percentile ranks
- Autocorrelation
- Entropy measures
"""

import numpy as np
import polars as pl
from loguru import logger

from ..base import BaseIndicator, IndicatorConfig


class Returns(BaseIndicator):
    """
    Calculate various types of returns.

    Outputs: simple returns, log returns, and cumulative returns.
    """

    def __init__(self, periods: list[int] = [1, 5, 10], config: IndicatorConfig | None = None):
        """
        Initialize Returns calculator.

        Args:
            periods: List of periods to calculate (default: [1, 5, 10])
            config: Optional configuration
        """
        super().__init__(config)
        self.periods = periods or self.config.get("periods", [1, 5, 10])

    def calculate(self, df: pl.DataFrame) -> pl.DataFrame:
        """Calculate returns."""
        self.validate_input(df)

        result = df

        for period in self.periods:
            # Simple returns
            simple_ret = (pl.col("close") - pl.col("close").shift(period)) / pl.col("close").shift(
                period
            )
            result = result.with_columns(simple_ret.alias(f"returns_{period}"))

            # Log returns
            log_ret = (pl.col("close") / pl.col("close").shift(period)).log()
            result = result.with_columns(log_ret.alias(f"log_returns_{period}"))

        # Cumulative returns (since start)
        result = result.with_columns(
            ((pl.col("close") / pl.col("close").first()) - 1).alias("cumulative_returns")
        )

        return result

    @property
    def required_columns(self) -> list[str]:
        return ["close"]

    @property
    def output_columns(self) -> list[str]:
        outputs = []
        for period in self.periods:
            outputs.extend([f"returns_{period}", f"log_returns_{period}"])
        outputs.append("cumulative_returns")
        return outputs


class HigherMoments(BaseIndicator):
    """
    Calculate higher statistical moments.

    Outputs: skewness and kurtosis over rolling windows.
    """

    def __init__(self, period: int = 20, config: IndicatorConfig | None = None):
        """
        Initialize Higher Moments calculator.

        Args:
            period: Rolling window period (default: 20)
            config: Optional configuration
        """
        super().__init__(config)
        self.period = period or self.config.get("period", 20)

    def calculate(self, df: pl.DataFrame) -> pl.DataFrame:
        """Calculate skewness and kurtosis."""
        self.validate_input(df)

        # Calculate log returns first
        log_returns = (pl.col("close") / pl.col("close").shift(1)).log()

        result = df.with_columns(log_returns.alias("_log_returns"))

        # Skewness: measure of asymmetry
        result = result.with_columns(
            pl.col("_log_returns")
            .rolling_map(
                function=lambda s: (
                    float(skew_val)
                    if len(s) >= 3 and (skew_val := pl.Series(s).skew()) is not None
                    else None
                ),
                window_size=self.period,
            )
            .alias("skewness")
        )

        # Kurtosis: measure of tail heaviness
        result = result.with_columns(
            pl.col("_log_returns")
            .rolling_map(
                function=lambda s: (
                    float(kurt_val)
                    if len(s) >= 4 and (kurt_val := pl.Series(s).kurtosis()) is not None
                    else None
                ),
                window_size=self.period,
            )
            .alias("kurtosis")
        )

        # Drop intermediate column
        result = result.drop("_log_returns")

        return result

    @property
    def required_columns(self) -> list[str]:
        return ["close"]

    @property
    def output_columns(self) -> list[str]:
        return ["skewness", "kurtosis"]


class ZScore(BaseIndicator):
    """
    Z-score normalization.

    Standardizes prices using rolling mean and standard deviation.
    """

    def __init__(self, period: int = 20, config: IndicatorConfig | None = None):
        """
        Initialize Z-score calculator.

        Args:
            period: Rolling window period (default: 20)
            config: Optional configuration
        """
        super().__init__(config)
        self.period = period or self.config.get("period", 20)

    def calculate(self, df: pl.DataFrame) -> pl.DataFrame:
        """Calculate Z-score."""
        self.validate_input(df)

        # Z-score = (X - mean) / std
        rolling_mean = pl.col("close").rolling_mean(window_size=self.period)
        rolling_std = pl.col("close").rolling_std(window_size=self.period)

        z_score = (pl.col("close") - rolling_mean) / rolling_std

        result = df.with_columns(z_score.alias("zscore"))

        return result

    @property
    def required_columns(self) -> list[str]:
        return ["close"]

    @property
    def output_columns(self) -> list[str]:
        return ["zscore"]


class PercentileRank(BaseIndicator):
    """
    Percentile rank.

    Shows where current price ranks relative to historical range.
    """

    def __init__(self, period: int = 100, config: IndicatorConfig | None = None):
        """
        Initialize Percentile Rank calculator.

        Args:
            period: Historical period for ranking (default: 100)
            config: Optional configuration
        """
        super().__init__(config)
        self.period = period or self.config.get("period", 100)

    def calculate(self, df: pl.DataFrame) -> pl.DataFrame:
        """Calculate percentile rank."""
        self.validate_input(df)

        # Percentile rank = percentage of values less than current value
        result = df.with_columns(
            pl.col("close")
            .rolling_map(
                function=lambda s: (s < s[-1]).sum() / len(s) * 100 if len(s) > 0 else None,
                window_size=self.period,
            )
            .alias("percentile_rank")
        )

        return result

    @property
    def required_columns(self) -> list[str]:
        return ["close"]

    @property
    def output_columns(self) -> list[str]:
        return ["percentile_rank"]


class Autocorrelation(BaseIndicator):
    """
    Autocorrelation.

    Measures correlation of returns with lagged returns.
    """

    def __init__(
        self, lags: list[int] = [1, 5, 10], period: int = 50, config: IndicatorConfig | None = None
    ):
        """
        Initialize Autocorrelation calculator.

        Args:
            lags: List of lag periods (default: [1, 5, 10])
            period: Rolling window for correlation (default: 50)
            config: Optional configuration
        """
        super().__init__(config)
        self.lags = lags or self.config.get("lags", [1, 5, 10])
        self.period = period or self.config.get("period", 50)

    def calculate(self, df: pl.DataFrame) -> pl.DataFrame:
        """Calculate autocorrelation."""
        self.validate_input(df)

        # Calculate returns
        returns = (pl.col("close") / pl.col("close").shift(1)).log()

        result = df.with_columns(returns.alias("_returns"))

        # Calculate autocorrelation for each lag
        for lag in self.lags:
            lagged_returns = pl.col("_returns").shift(lag)

            # Rolling correlation between returns and lagged returns
            result = result.with_columns(
                pl.rolling_corr(pl.col("_returns"), lagged_returns, window_size=self.period).alias(
                    f"autocorr_{lag}"
                )
            )

        # Drop intermediate column
        result = result.drop("_returns")

        return result

    @property
    def required_columns(self) -> list[str]:
        return ["close"]

    @property
    def output_columns(self) -> list[str]:
        return [f"autocorr_{lag}" for lag in self.lags]


class HurstExponent(BaseIndicator):
    """
    Hurst Exponent.

    Measures long-term memory and mean reversion vs trending behavior.
    H < 0.5: mean reverting, H = 0.5: random walk, H > 0.5: trending
    """

    def __init__(self, period: int = 100, config: IndicatorConfig | None = None):
        """
        Initialize Hurst Exponent calculator.

        Args:
            period: Calculation period (default: 100)
            config: Optional configuration
        """
        super().__init__(config)
        self.period = period or self.config.get("period", 100)

    def _calculate_hurst(self, prices: np.ndarray) -> float | None:
        """Calculate Hurst exponent for a price series."""
        if len(prices) < 20:
            return None

        # Remove trend
        prices = np.array(prices)
        log_prices = np.log(prices)

        # Calculate lags
        lags = range(2, min(20, len(prices) // 2))
        tau = []
        lagvec = []

        for lag in lags:
            # Calculate standard deviation of differences
            pp = np.subtract(log_prices[lag:], log_prices[:-lag])
            lagvec.append(lag)
            tau.append(np.std(pp))

        # Linear fit of log(tau) vs log(lags)
        if len(lagvec) > 1:
            poly = np.polyfit(np.log(lagvec), np.log(tau), 1)
            return poly[0]  # Slope is Hurst exponent
        return None

    def calculate(self, df: pl.DataFrame) -> pl.DataFrame:
        """Calculate Hurst Exponent."""
        self.validate_input(df)

        # Calculate rolling Hurst exponent
        result = df.with_columns(
            pl.col("close")
            .rolling_map(
                function=lambda s: self._calculate_hurst(np.array(s)), window_size=self.period
            )
            .alias("hurst_exponent")
        )

        return result

    @property
    def required_columns(self) -> list[str]:
        return ["close"]

    @property
    def output_columns(self) -> list[str]:
        return ["hurst_exponent"]


class RollingBeta(BaseIndicator):
    """
    Rolling Beta.

    Measures sensitivity to market movements (requires benchmark data).
    """

    def __init__(
        self,
        period: int = 50,
        benchmark_col: str = "benchmark_close",
        config: IndicatorConfig | None = None,
    ):
        """
        Initialize Rolling Beta calculator.

        Args:
            period: Rolling window period (default: 50)
            benchmark_col: Column name for benchmark prices (default: "benchmark_close")
            config: Optional configuration
        """
        super().__init__(config)
        self.period = period or self.config.get("period", 50)
        self.benchmark_col = benchmark_col or self.config.get("benchmark_col", "benchmark_close")

    def calculate(self, df: pl.DataFrame) -> pl.DataFrame:
        """Calculate rolling beta."""
        # Validate both asset and benchmark columns exist
        if self.benchmark_col not in df.columns:
            logger.warning(
                f"Benchmark column '{self.benchmark_col}' not found, skipping beta calculation"
            )
            return df.with_columns(pl.lit(None).alias("beta"))

        # Calculate returns
        asset_returns = (pl.col("close") / pl.col("close").shift(1)).log()
        benchmark_returns = (pl.col(self.benchmark_col) / pl.col(self.benchmark_col).shift(1)).log()

        result = df.with_columns(
            [asset_returns.alias("_asset_returns"), benchmark_returns.alias("_benchmark_returns")]
        )

        # Calculate rolling covariance
        rolling_cov = pl.rolling_cov(
            pl.col("_asset_returns"), pl.col("_benchmark_returns"), window_size=self.period
        )

        # Calculate rolling variance of benchmark
        rolling_var = pl.col("_benchmark_returns").rolling_var(window_size=self.period)

        # Beta = Cov(asset, benchmark) / Var(benchmark)
        result = result.with_columns((rolling_cov / rolling_var).alias("beta"))

        # Drop intermediate columns
        result = result.drop(["_asset_returns", "_benchmark_returns"])

        return result

    @property
    def required_columns(self) -> list[str]:
        return ["close"]  # benchmark_col is optional

    @property
    def output_columns(self) -> list[str]:
        return ["beta"]


class SharpeRatio(BaseIndicator):
    """
    Rolling Sharpe Ratio.

    Risk-adjusted return measure.
    """

    def __init__(
        self,
        period: int = 50,
        risk_free_rate: float = 0.0,
        annualization_factor: int = 252,
        config: IndicatorConfig | None = None,
    ):
        """
        Initialize Sharpe Ratio calculator.

        Args:
            period: Rolling window period (default: 50)
            risk_free_rate: Annual risk-free rate (default: 0.0)
            annualization_factor: Trading periods per year (default: 252)
            config: Optional configuration
        """
        super().__init__(config)
        self.period = period or self.config.get("period", 50)
        self.risk_free_rate = risk_free_rate or self.config.get("risk_free_rate", 0.0)
        self.annualization_factor = annualization_factor or self.config.get(
            "annualization_factor", 252
        )

    def calculate(self, df: pl.DataFrame) -> pl.DataFrame:
        """Calculate rolling Sharpe ratio."""
        self.validate_input(df)

        # Calculate returns
        returns = (pl.col("close") / pl.col("close").shift(1)).log()

        result = df.with_columns(returns.alias("_returns"))

        # Periodic risk-free rate
        rf_periodic = self.risk_free_rate / self.annualization_factor

        # Rolling mean and std of excess returns
        excess_returns = pl.col("_returns") - rf_periodic
        rolling_mean = excess_returns.rolling_mean(window_size=self.period)
        rolling_std = pl.col("_returns").rolling_std(window_size=self.period)

        # Annualized Sharpe Ratio
        sharpe = (rolling_mean / rolling_std) * np.sqrt(self.annualization_factor)

        result = result.with_columns(sharpe.alias("sharpe_ratio"))

        # Drop intermediate column
        result = result.drop("_returns")

        return result

    @property
    def required_columns(self) -> list[str]:
        return ["close"]

    @property
    def output_columns(self) -> list[str]:
        return ["sharpe_ratio"]


class Drawdown(BaseIndicator):
    """
    Drawdown analysis.

    Measures peak-to-trough decline.
    """

    def __init__(self, config: IndicatorConfig | None = None):
        """Initialize Drawdown calculator."""
        super().__init__(config)

    def calculate(self, df: pl.DataFrame) -> pl.DataFrame:
        """Calculate drawdown metrics."""
        self.validate_input(df)

        # Running maximum
        running_max = pl.col("close").cum_max()

        # Drawdown
        drawdown = (pl.col("close") - running_max) / running_max

        # Drawdown in percentage
        drawdown_pct = drawdown * 100

        result = df.with_columns(
            [
                running_max.alias("running_max"),
                drawdown.alias("drawdown"),
                drawdown_pct.alias("drawdown_pct"),
            ]
        )

        return result

    @property
    def required_columns(self) -> list[str]:
        return ["close"]

    @property
    def output_columns(self) -> list[str]:
        return ["running_max", "drawdown", "drawdown_pct"]
