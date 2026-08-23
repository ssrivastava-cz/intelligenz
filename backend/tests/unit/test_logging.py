"""Unit tests for the centralized logging configuration.

`configure_logging` replaces the root logger's handlers — genuinely
global state — so every test here restores the original handlers/level
afterward to avoid bleeding into the rest of the test suite.
"""
import logging

import pytest

from app.core.logging import configure_logging, get_logger


@pytest.fixture(autouse=True)
def _restore_root_logger():
    root_logger = logging.getLogger()
    original_handlers = list(root_logger.handlers)
    original_level = root_logger.level
    yield
    for handler in root_logger.handlers:
        handler.close()
    root_logger.handlers.clear()
    root_logger.handlers.extend(original_handlers)
    root_logger.setLevel(original_level)


def test_configure_logging_creates_the_logs_directory_and_a_log_file(tmp_path):
    logs_root = tmp_path / "logs"

    configure_logging(logs_root=logs_root)
    get_logger("test.module").info("hello")

    assert logs_root.is_dir()
    assert (logs_root / "app.log").is_file()


def test_log_file_contains_timestamp_level_module_and_message(tmp_path):
    logs_root = tmp_path / "logs"
    configure_logging(logs_root=logs_root)
    logger = get_logger("app.services.redmine_service")

    logger.info("Retrieved ticket 12345")

    content = (logs_root / "app.log").read_text(encoding="utf-8")
    assert "INFO" in content
    assert "app.services.redmine_service" in content
    assert "Retrieved ticket 12345" in content


def test_log_entries_include_a_real_timestamp(tmp_path):
    import re

    logs_root = tmp_path / "logs"
    configure_logging(logs_root=logs_root)

    get_logger("test.module").info("hello")

    content = (logs_root / "app.log").read_text(encoding="utf-8")
    assert re.search(r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}", content)


def test_configure_logging_captures_warning_and_error_levels(tmp_path):
    logs_root = tmp_path / "logs"
    configure_logging(logs_root=logs_root)
    logger = get_logger("app.services.redmine_service")

    logger.warning("Failed to download attachment design.pdf (404)")
    logger.error("Unexpected failure retrieving ticket")

    content = (logs_root / "app.log").read_text(encoding="utf-8")
    assert "WARNING" in content
    assert "Failed to download attachment design.pdf (404)" in content
    assert "ERROR" in content
    assert "Unexpected failure retrieving ticket" in content


def test_configure_logging_also_writes_to_the_console(tmp_path, capsys):
    logs_root = tmp_path / "logs"
    configure_logging(logs_root=logs_root)

    get_logger("test.module").info("hello console")

    captured = capsys.readouterr()
    assert "hello console" in captured.err or "hello console" in captured.out


def test_get_logger_returns_a_standard_library_logger():
    logger = get_logger("some.module")

    assert isinstance(logger, logging.Logger)
    assert logger.name == "some.module"
