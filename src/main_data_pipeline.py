from __future__ import annotations

import argparse
import logging
import subprocess
import sys

from src.utils.logging_config import setup_logging

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run data ingestion/preprocessing pipelines sequentially (no training)"
    )
    parser.add_argument("--asset", required=True, help="Asset symbol (e.g. AAPL)")
    parser.add_argument(
        "--overwrite-indicators",
        action="store_true",
        help="Overwrite existing technical indicators if they exist",
    )
    return parser.parse_args()


def _run(module: str, args: list[str]) -> subprocess.CompletedProcess[str]:
    cmd = [sys.executable, "-m", module] + args
    logger.info("Running pipeline step", extra={"module_name": module, "cmd_args": args})
    return subprocess.run(cmd, check=False, text=True)


def main() -> None:
    setup_logging(logging.INFO)
    args = parse_args()

    asset = args.asset.strip().upper()
    overwrite_indicators = args.overwrite_indicators

    steps = [
        ("src.main_candles", ["--asset", asset]),
        ("src.main_news_dataset", ["--asset", asset]),
        ("src.main_sentiment", ["--asset", asset]),
        ("src.main_sentiment_features", ["--asset", asset]),
        (
            "src.main_technical_indicators",
            ["--asset", asset] + (["--overwrite"] if overwrite_indicators else []),
        ),
        ("src.main_fundamentals", ["--asset", asset]),
        ("src.main_dataset_tft", ["--asset", asset]),
    ]

    for module, module_args in steps:
        result = _run(module, module_args)
        if result.returncode == 0:
            continue

        raise RuntimeError(
            f"Pipeline step failed: {module}\n"
            f"stdout: {result.stdout}\n"
            f"stderr: {result.stderr}"
        )

    logger.info("All data pipeline steps completed", extra={"asset": asset})


if __name__ == "__main__":
    main()
