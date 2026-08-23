"""Centralized logging configuration — console output plus one rotating
log file per day under `backend/logs/`, in a single consistent format
used everywhere in the backend. No module should ever call `print()`;
use `get_logger(__name__)` instead.
"""
import logging
import logging.handlers
from pathlib import Path

_LOG_FORMAT = "%(asctime)s %(levelname)s %(name)s\n%(message)s\n"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
_LOG_FILENAME = "app.log"


def configure_logging(level: str = "INFO", logs_root: Path | None = None) -> None:
    """Points the root logger at a console handler and a file handler
    that rotates at midnight, so each calendar day gets its own file
    (`app.log` for today, `app.log.YYYY-MM-DD` for prior days).
    """
    logs_root = logs_root or Path("logs")
    logs_root.mkdir(parents=True, exist_ok=True)

    formatter = logging.Formatter(fmt=_LOG_FORMAT, datefmt=_DATE_FORMAT)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    file_handler = logging.handlers.TimedRotatingFileHandler(
        filename=logs_root / _LOG_FILENAME,
        when="midnight",
        interval=1,
        backupCount=30,
        encoding="utf-8",
    )
    file_handler.suffix = "%Y-%m-%d"
    file_handler.setFormatter(formatter)

    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    root_logger.handlers.clear()
    root_logger.addHandler(console_handler)
    root_logger.addHandler(file_handler)


def get_logger(name: str) -> logging.Logger:
    """The one way any module in this backend should get a logger —
    keeps every log line flowing through the same console+file
    configuration set up by `configure_logging`.
    """
    return logging.getLogger(name)
