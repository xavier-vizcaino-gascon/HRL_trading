"""
Feature pipeline for orchestrating indicator calculations.

Provides a clean interface for applying multiple indicators to OHLCV data
with configuration management, error handling, and performance optimization.
"""

from pathlib import Path
from typing import Any

import polars as pl
import yaml
from loguru import logger

from .base import BaseIndicator, IndicatorConfig, ensure_sorted, validate_ohlcv


class FeaturePipeline:
    """
    Pipeline for calculating multiple technical indicators.

    The pipeline manages:
    - Sequential indicator calculation
    - Error handling and recovery
    - Performance tracking
    - Optional data validation
    """

    def __init__(
        self,
        indicators: list[BaseIndicator] | None = None,
        validate_input: bool = True,
        validate_output: bool = False,
        sort_by_time: bool = True,
        time_column: str = "timestamp",
    ):
        """
        Initialize feature pipeline.

        Args:
            indicators: List of indicators to calculate
            validate_input: Validate OHLCV data before processing
            validate_output: Validate output has no infinite/null values
            sort_by_time: Sort data by timestamp before processing
            time_column: Name of timestamp column
        """
        self.indicators = indicators or []
        self.validate_input = validate_input
        self.validate_output = validate_output
        self.sort_by_time = sort_by_time
        self.time_column = time_column

        # Statistics
        self._stats = {
            "total_indicators": 0,
            "successful_indicators": 0,
            "failed_indicators": 0,
            "total_features": 0,
            "processing_time_ms": 0,
        }

    def add(self, indicator: BaseIndicator) -> "FeaturePipeline":
        """
        Add an indicator to the pipeline.

        Args:
            indicator: Indicator to add

        Returns:
            Self for method chaining
        """
        self.indicators.append(indicator)
        return self

    def add_multiple(self, indicators: list[BaseIndicator]) -> "FeaturePipeline":
        """
        Add multiple indicators to the pipeline.

        Args:
            indicators: List of indicators to add

        Returns:
            Self for method chaining
        """
        self.indicators.extend(indicators)
        return self

    def calculate(self, df: pl.DataFrame, ignore_errors: bool = False) -> pl.DataFrame:
        """
        Calculate all indicators in the pipeline.

        Args:
            df: Input DataFrame with OHLCV data
            ignore_errors: Continue processing if an indicator fails

        Returns:
            DataFrame with all indicator columns added

        Raises:
            ValueError: If input validation fails
            Exception: If indicator calculation fails (unless ignore_errors=True)
        """
        import time

        start_time = time.time()

        # Reset statistics
        self._stats = {
            "total_indicators": len(self.indicators),
            "successful_indicators": 0,
            "failed_indicators": 0,
            "total_features": 0,
            "processing_time_ms": 0,
        }

        # Input validation
        if self.validate_input:
            validate_ohlcv(df)
            logger.debug("Input OHLCV validation passed")

        # Sort by time if requested
        if self.sort_by_time:
            df = ensure_sorted(df, self.time_column)

        # Store original columns to count new features
        original_columns = set(df.columns)

        result = df

        # Calculate each indicator
        for indicator in self.indicators:
            if not indicator.config.enabled:
                logger.debug(f"Skipping disabled indicator: {indicator}")
                continue

            try:
                logger.debug(f"Calculating {indicator}")
                result = indicator.calculate(result)
                self._stats["successful_indicators"] += 1

            except Exception as e:
                self._stats["failed_indicators"] += 1
                error_msg = f"Error calculating {indicator}: {e}"

                if ignore_errors:
                    logger.warning(error_msg)
                    continue
                else:
                    logger.error(error_msg)
                    raise

        # Count new features
        new_columns = set(result.columns) - original_columns
        self._stats["total_features"] = len(new_columns)

        # Output validation
        if self.validate_output:
            self._validate_output(result, new_columns)

        # Calculate processing time
        self._stats["processing_time_ms"] = int((time.time() - start_time) * 1000)

        logger.info(
            f"Pipeline completed: {self._stats['successful_indicators']}/{self._stats['total_indicators']} "
            f"indicators, {self._stats['total_features']} features added "
            f"in {self._stats['processing_time_ms']:.2f}ms"
        )

        return result

    def _validate_output(self, df: pl.DataFrame, new_columns: set) -> None:
        """Validate output data quality."""
        for col in new_columns:
            # Check for infinite values
            if df[col].is_infinite().any():
                logger.warning(f"Column {col} contains infinite values")

            # Check for null percentage
            null_count = df[col].null_count()
            null_pct = 100 * null_count / len(df)
            if null_pct > 50:
                logger.warning(f"Column {col} has {null_pct:.1f}% null values")

    def get_statistics(self) -> dict[str, Any]:
        """Get pipeline execution statistics."""
        return self._stats.copy()

    def get_output_columns(self) -> list[str]:
        """Get list of all output columns from all indicators."""
        columns = []
        for indicator in self.indicators:
            if indicator.config.enabled:
                columns.extend(indicator.output_columns)
        return columns

    def get_required_columns(self) -> list[str]:
        """Get list of all required input columns."""
        columns = set()
        for indicator in self.indicators:
            if indicator.config.enabled:
                columns.update(indicator.required_columns)
        return list(columns)

    @classmethod
    def from_config(cls, config_path: Path, **kwargs) -> "FeaturePipeline":
        """
        Create pipeline from YAML configuration file.

        Args:
            config_path: Path to YAML config file
            **kwargs: Additional arguments for pipeline initialization

        Returns:
            Configured FeaturePipeline

        Example config.yaml:
            indicators:
              - type: SMA
                params:
                  periods: [20, 50, 200]
              - type: RSI
                params:
                  period: 14
              - type: BollingerBands
                enabled: true
                params:
                  period: 20
                  std_dev: 2.0
        """
        with open(config_path) as f:
            config = yaml.safe_load(f)

        indicators = []

        for ind_config in config.get("indicators", []):
            ind_type = ind_config["type"]
            enabled = ind_config.get("enabled", True)
            params = ind_config.get("params", {})

            # Import indicator class dynamically
            try:
                # Try importing from indicators module
                from . import indicators as ind_module

                indicator_class = getattr(ind_module, ind_type)

                # Create IndicatorConfig
                ind_conf = IndicatorConfig(enabled=enabled, params=params)

                # Instantiate indicator with params and config
                indicator = indicator_class(**params, config=ind_conf)
                indicators.append(indicator)

                logger.debug(f"Loaded indicator {ind_type} from config")

            except (AttributeError, ImportError) as e:
                logger.error(f"Failed to load indicator {ind_type}: {e}")
                raise ValueError(f"Unknown indicator type: {ind_type}")

        pipeline_kwargs = config.get("pipeline", {})
        pipeline_kwargs.update(kwargs)

        return cls(indicators=indicators, **pipeline_kwargs)

    def to_config(self, config_path: Path) -> None:
        """
        Save pipeline configuration to YAML file.

        Args:
            config_path: Path to save config file
        """
        config = {
            "pipeline": {
                "validate_input": self.validate_input,
                "validate_output": self.validate_output,
                "sort_by_time": self.sort_by_time,
                "time_column": self.time_column,
            },
            "indicators": [],
        }

        for indicator in self.indicators:
            ind_config = {
                "type": indicator.__class__.__name__,
                "enabled": indicator.config.enabled,
                "params": indicator.config.params,
            }
            config["indicators"].append(ind_config)

        with open(config_path, "w") as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)

        logger.info(f"Pipeline configuration saved to {config_path}")

    def __repr__(self) -> str:
        """String representation."""
        enabled_count = sum(1 for ind in self.indicators if ind.config.enabled)
        return f"FeaturePipeline(indicators={enabled_count}/{len(self.indicators)} enabled)"

    def __len__(self) -> int:
        """Number of indicators in pipeline."""
        return len(self.indicators)


class FeatureSelector:
    """
    Select subset of features based on criteria.

    Can be used after pipeline to filter features before feeding to model.
    """

    def __init__(
        self, max_null_pct: float = 50.0, remove_infinite: bool = True, remove_constant: bool = True
    ):
        """
        Initialize feature selector.

        Args:
            max_null_pct: Maximum allowed null percentage (0-100)
            remove_infinite: Remove columns with infinite values
            remove_constant: Remove columns with constant values
        """
        self.max_null_pct = max_null_pct
        self.remove_infinite = remove_infinite
        self.remove_constant = remove_constant

    def select(self, df: pl.DataFrame, exclude_columns: list[str] | None = None) -> pl.DataFrame:
        """
        Select features based on quality criteria.

        Args:
            df: Input DataFrame
            exclude_columns: Columns to exclude from filtering (e.g., OHLCV, timestamp)

        Returns:
            DataFrame with filtered columns
        """
        exclude_columns = exclude_columns or []
        columns_to_keep = set(exclude_columns)
        columns_to_check = [col for col in df.columns if col not in exclude_columns]

        removed = []

        for col in columns_to_check:
            keep = True

            # Check null percentage
            null_pct = 100 * df[col].null_count() / len(df)
            if null_pct > self.max_null_pct:
                logger.debug(f"Removing {col}: {null_pct:.1f}% null values")
                keep = False

            # Check for infinite values
            if keep and self.remove_infinite:
                try:
                    if df[col].is_infinite().any():
                        logger.debug(f"Removing {col}: contains infinite values")
                        keep = False
                except Exception:
                    pass  # Skip non-numeric columns

            # Check for constant values
            if keep and self.remove_constant:
                try:
                    unique_count = df[col].n_unique()
                    if unique_count == 1:
                        logger.debug(f"Removing {col}: constant value")
                        keep = False
                except Exception:
                    pass  # Skip non-numeric columns

            if keep:
                columns_to_keep.add(col)
            else:
                removed.append(col)

        logger.info(
            f"Feature selection: kept {len(columns_to_keep)}, removed {len(removed)} columns"
        )

        return df.select(sorted(columns_to_keep))
