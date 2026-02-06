# src/utils/logging_config.py
from __future__ import annotations

import logging
import os
import sys
from pathlib import Path


class ExtraFormatter(logging.Formatter):
    _standard_keys = {
        "name",
        "msg",
        "args",
        "levelname",
        "levelno",
        "pathname",
        "filename",
        "module",
        "exc_info",
        "exc_text",
        "stack_info",
        "lineno",
        "funcName",
        "created",
        "msecs",
        "relativeCreated",
        "thread",
        "threadName",
        "processName",
        "process",
        "message",
    }

    def format(self, record: logging.LogRecord) -> str:
        base = super().format(record)
        extras = {
            k: v for k, v in record.__dict__.items() if k not in self._standard_keys
        }
        if not extras:
            return base
        extra_str = " ".join(f"{k}={v}" for k, v in sorted(extras.items()))
        return f"{base} | {extra_str}"


def setup_logging(level: int = logging.INFO) -> None:
    """
    Standard project logging:
    2026-01-14 09:49:59 | INFO | module.name | message
    """

    root = logging.getLogger()

    # evita duplicar handlers se setup_logging() for chamado mais de uma vez
    if root.handlers:
        root.setLevel(level)
        return

    root.setLevel(level)

    formatter = ExtraFormatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)
    root.addHandler(handler)

    log_dir_env = os.getenv("LOG_DIR")
    log_dir = Path(log_dir_env) if log_dir_env else Path("logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler(log_dir / "pipeline.log", encoding="utf-8")
    file_handler.setFormatter(formatter)
    root.addHandler(file_handler)

    # opcional: silenciar libs muito verbosas
    logging.getLogger("yfinance").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
