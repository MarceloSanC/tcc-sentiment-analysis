# src/utils/logging_config.py
from __future__ import annotations

import logging
import sys


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

    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler.setFormatter(formatter)
    root.addHandler(handler)

    # opcional: silenciar libs muito verbosas
    logging.getLogger("yfinance").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
