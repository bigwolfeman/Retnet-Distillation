"""
Structured logging configuration for curriculum training system.

Provides JSON-formatted logs compatible with wandb and production monitoring systems.
"""

import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional


class JSONFormatter(logging.Formatter):
    """
    Format log records as JSON for structured logging.

    Compatible with wandb, Datadog, CloudWatch, and other log aggregators.
    """

    def format(self, record: logging.LogRecord) -> str:
        """
        Format a log record as a JSON string.

        Args:
            record: Log record to format

        Returns:
            JSON-formatted log string
        """
        log_data: Dict[str, Any] = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }

        # Add exception info if present
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)

        # Add extra fields from record
        if hasattr(record, "extra_fields"):
            log_data.update(record.extra_fields)

        return json.dumps(log_data)


def setup_logging(
    log_level: str = "INFO",
    log_file: Optional[Path] = None,
    json_format: bool = True,
) -> logging.Logger:
    """
    Configure root logger with structured JSON output.

    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional file path to write logs to
        json_format: If True, use JSON formatter. Otherwise use plain text.

    Returns:
        Configured logger instance
    """
    # Get root logger
    logger = logging.getLogger()
    logger.setLevel(getattr(logging, log_level.upper()))

    # Remove existing handlers
    logger.handlers.clear()

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)

    if json_format:
        console_handler.setFormatter(JSONFormatter())
    else:
        console_handler.setFormatter(
            logging.Formatter(
                "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
        )

    logger.addHandler(console_handler)

    # File handler (if specified)
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, mode="a")
        file_handler.setLevel(logging.DEBUG)  # File gets all levels

        if json_format:
            file_handler.setFormatter(JSONFormatter())
        else:
            file_handler.setFormatter(
                logging.Formatter(
                    "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S",
                )
            )

        logger.addHandler(file_handler)

    return logger


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance for a specific module.

    Args:
        name: Logger name (usually __name__)

    Returns:
        Logger instance
    """
    return logging.getLogger(name)


def log_with_extra(logger: logging.Logger, level: str, message: str, **extra_fields: Any):
    """
    Log a message with additional structured fields.

    Args:
        logger: Logger instance
        level: Log level (info, warning, error, etc.)
        message: Log message
        **extra_fields: Additional key-value pairs to include in JSON output
    """
    record = logger.makeRecord(
        logger.name,
        getattr(logging, level.upper()),
        "(unknown file)",
        0,
        message,
        (),
        None,
    )
    record.extra_fields = extra_fields
    logger.handle(record)


# Example usage
if __name__ == "__main__":
    # Setup logging
    logger = setup_logging(log_level="DEBUG", json_format=True)

    # Basic logging
    logger.info("Training started")
    logger.debug("Loading dataset")
    logger.warning("High GPU memory usage detected")

    # Structured logging with extra fields
    module_logger = get_logger(__name__)
    log_with_extra(
        module_logger,
        "info",
        "Batch processed",
        step=100,
        loss=0.234,
        em=0.823,
        gpu_mem_gb=28.5,
    )

    # Error logging with exception
    try:
        raise ValueError("Invalid band configuration")
    except Exception:
        logger.exception("Failed to load curriculum config")
