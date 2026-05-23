"""
Custom exception classes for the trading system.

Provides specific exceptions for different failure scenarios to enable
better error handling and debugging.
"""


class TradingSystemError(Exception):
    """Base exception for all trading system errors."""

    pass


# Connector Exceptions


class ConnectorError(TradingSystemError):
    """Base exception for connector-related errors."""

    pass


class ConnectionError(ConnectorError):
    """Failed to connect to exchange/broker."""

    pass


class AuthenticationError(ConnectorError):
    """Authentication failed (invalid API keys, etc.)."""

    pass


class RateLimitError(ConnectorError):
    """API rate limit exceeded."""

    pass


class InsufficientFundsError(ConnectorError):
    """Insufficient balance to execute order."""

    pass


class InvalidOrderError(ConnectorError):
    """Order parameters are invalid."""

    pass


# Data Pipeline Exceptions


class DataError(TradingSystemError):
    """Base exception for data-related errors."""

    pass


class DataNotFoundError(DataError):
    """Requested data not available."""

    pass


class DataValidationError(DataError):
    """Data failed validation checks."""

    pass


class StorageError(DataError):
    """Failed to read/write data from storage."""

    pass


# Feature Engineering Exceptions


class FeatureError(TradingSystemError):
    """Base exception for feature engineering errors."""

    pass


class InvalidFeatureConfigError(FeatureError):
    """Feature configuration is invalid."""

    pass


class FeatureCalculationError(FeatureError):
    """Error during feature calculation."""

    pass


# RL Environment Exceptions


class EnvironmentError(TradingSystemError):
    """Base exception for RL environment errors."""

    pass


class InvalidActionError(EnvironmentError):
    """Action is not valid in current state."""

    pass


class EnvironmentResetError(EnvironmentError):
    """Failed to reset environment."""

    pass


# Execution Exceptions


class ExecutionError(TradingSystemError):
    """Base exception for order execution errors."""

    pass


class RiskLimitError(ExecutionError):
    """Action blocked by risk management rules."""

    pass


class OrderRejectedError(ExecutionError):
    """Order was rejected by exchange."""

    pass


class PortfolioError(ExecutionError):
    """Error in portfolio management."""

    pass


# Model Exceptions


class ModelError(TradingSystemError):
    """Base exception for model-related errors."""

    pass


class ModelLoadError(ModelError):
    """Failed to load model."""

    pass


class ModelSaveError(ModelError):
    """Failed to save model."""

    pass


class InferenceError(ModelError):
    """Error during model inference."""

    pass


# Configuration Exceptions


class ConfigurationError(TradingSystemError):
    """Base exception for configuration errors."""

    pass


class InvalidConfigError(ConfigurationError):
    """Configuration file is invalid or malformed."""

    pass


class MissingConfigError(ConfigurationError):
    """Required configuration is missing."""

    pass
