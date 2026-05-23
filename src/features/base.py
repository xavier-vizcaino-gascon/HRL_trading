"""
Base classes and utilities for feature engineering.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import polars as pl
from loguru import logger


@dataclass
class IndicatorConfig:
    """Configuration for an indicator."""

    enabled: bool = True
    params: dict[str, Any] = field(default_factory=dict)

    def get(self, key: str, default: Any = None) -> Any:
        """Get parameter value with default."""
        return self.params.get(key, default)


class BaseIndicator(ABC):
    """
    Base class for all technical indicators.

    All indicators must implement the calculate() method which takes
    a Polars DataFrame with OHLCV data and returns the same DataFrame
    with additional indicator columns.
    """

    def __init__(self, config: IndicatorConfig | None = None):
        """
        Initialize indicator.

        Args:
            config: Optional configuration for the indicator
        """
        self.config = config or IndicatorConfig()
        self._validate_config()

    def _validate_config(self) -> None:
        """Validate configuration parameters. Override in subclasses."""
        pass

    @abstractmethod
    def calculate(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Calculate indicator values.

        Args:
            df: Input DataFrame with OHLCV data

        Returns:
            DataFrame with additional indicator columns
        """
        pass

    @property
    @abstractmethod
    def required_columns(self) -> list[str]:
        """
        List of required columns in input DataFrame.

        Returns:
            List of column names
        """
        pass

    @property
    @abstractmethod
    def output_columns(self) -> list[str]:
        """
        List of columns that will be added by this indicator.

        Returns:
            List of column names
        """
        pass

    def validate_input(self, df: pl.DataFrame) -> None:
        """
        Validate input DataFrame has required columns.

        Args:
            df: Input DataFrame

        Raises:
            ValueError: If required columns are missing
        """
        missing = set(self.required_columns) - set(df.columns)
        if missing:
            raise ValueError(
                f"{self.__class__.__name__} requires columns {missing} "
                f"which are not present in the DataFrame"
            )

    def __call__(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Convenience method to calculate indicator.

        Args:
            df: Input DataFrame

        Returns:
            DataFrame with indicator columns
        """
        return self.calculate(df)

    def __repr__(self) -> str:
        """String representation."""
        params_str = ", ".join(f"{k}={v}" for k, v in self.config.params.items())
        return f"{self.__class__.__name__}({params_str})"


class IndicatorGroup(BaseIndicator):
    """
    Group of related indicators that can be calculated together.

    This is useful for organizing related indicators and calculating
    them efficiently in batch.
    """

    def __init__(
        self, indicators: list[BaseIndicator] | None = None, config: IndicatorConfig | None = None
    ):
        """
        Initialize indicator group.

        Args:
            indicators: List of indicators in this group
            config: Optional configuration
        """
        super().__init__(config)
        self.indicators = indicators or []

    def add(self, indicator: BaseIndicator) -> "IndicatorGroup":
        """
        Add an indicator to the group.

        Args:
            indicator: Indicator to add

        Returns:
            Self for method chaining
        """
        self.indicators.append(indicator)
        return self

    def calculate(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Calculate all indicators in the group.

        Args:
            df: Input DataFrame

        Returns:
            DataFrame with all indicator columns
        """
        result = df
        for indicator in self.indicators:
            if indicator.config.enabled:
                try:
                    result = indicator.calculate(result)
                except Exception as e:
                    logger.error(f"Error calculating {indicator}: {e}")
                    raise
        return result

    @property
    def required_columns(self) -> list[str]:
        """Get all required columns from all indicators."""
        columns = set()
        for indicator in self.indicators:
            columns.update(indicator.required_columns)
        return list(columns)

    @property
    def output_columns(self) -> list[str]:
        """Get all output columns from all indicators."""
        columns = []
        for indicator in self.indicators:
            columns.extend(indicator.output_columns)
        return columns


def ensure_sorted(df: pl.DataFrame, time_col: str = "timestamp") -> pl.DataFrame:
    """
    Ensure DataFrame is sorted by time.

    Args:
        df: Input DataFrame
        time_col: Name of time column

    Returns:
        Sorted DataFrame
    """
    if time_col in df.columns:
        return df.sort(time_col)
    return df


def validate_ohlcv(df: pl.DataFrame) -> None:
    """
    Validate DataFrame has standard OHLCV columns.

    Args:
        df: Input DataFrame

    Raises:
        ValueError: If required OHLCV columns are missing
    """
    required = ["open", "high", "low", "close", "volume"]
    missing = set(required) - set(df.columns)
    if missing:
        raise ValueError(f"DataFrame missing required OHLCV columns: {missing}")
