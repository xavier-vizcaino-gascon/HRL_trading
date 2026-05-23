"""
Centralized logging configuration using Loguru.

Provides structured logging with rotation, compression, and different log levels
for different components.
"""

import sys
from pathlib import Path

from loguru import logger


def setup_logging(
    log_level: str = "INFO",
    log_dir: Path | None = None,
    log_to_console: bool = True,
    log_to_file: bool = True,
    rotation: str = "100 MB",
    retention: str = "30 days",
    compression: str = "zip",
) -> None:
    """
    Configure application-wide logging.

    Args:
        log_level: Minimum log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_dir: Directory for log files (default: ./logs)
        log_to_console: Enable console output
        log_to_file: Enable file output
        rotation: Log file rotation trigger (size or time)
        retention: How long to keep old logs
        compression: Compression format for rotated logs
    """
    # Remove default handler
    logger.remove()

    # Console handler
    if log_to_console:
        logger.add(
            sys.stdout,
            format="<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
            "<level>{message}</level>",
            level=log_level,
            colorize=True,
        )

    # File handler
    if log_to_file:
        if log_dir is None:
            log_dir = Path("./logs")

        log_dir.mkdir(parents=True, exist_ok=True)

        # Main application log
        logger.add(
            log_dir / "app_{time:YYYY-MM-DD}.log",
            format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{function}:{line} | {message}",
            level=log_level,
            rotation=rotation,
            retention=retention,
            compression=compression,
            enqueue=True,  # Thread-safe
        )

        # Error-only log
        logger.add(
            log_dir / "errors_{time:YYYY-MM-DD}.log",
            format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{function}:{line} | {message}",
            level="ERROR",
            rotation=rotation,
            retention=retention,
            compression=compression,
            enqueue=True,
        )

        # Trading operations log (separate for audit trail)
        logger.add(
            log_dir / "trading_{time:YYYY-MM-DD}.log",
            format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {message}",
            level="INFO",
            rotation="1 day",
            retention="1 year",  # Keep trading logs longer for compliance
            compression=compression,
            enqueue=True,
            filter=lambda record: record["extra"].get("type") == "trading",
        )

    logger.info(f"Logging initialized at {log_level} level")


def get_logger(name: str):
    """
    Get a logger instance for a specific module.

    Args:
        name: Logger name (typically __name__)

    Returns:
        Logger instance
    """
    return logger.bind(module=name)


def log_trade_event(event_type: str, **kwargs) -> None:
    """
    Log a trading event for audit trail.

    Args:
        event_type: Type of event (ORDER_PLACED, ORDER_FILLED, etc.)
        **kwargs: Event details
    """
    logger.bind(type="trading").info(
        f"TRADE_EVENT: {event_type}", extra={"event_type": event_type, **kwargs}
    )
